#!/usr/bin/env python3
"""
Salvus mesh.h5 → SPECFEM GLL 插值点云
=====================================

读取 SinoScope 作者提供的 Salvus 0.11.x 谱元网格 (mesh.h5)，
提取 GLL 点坐标与弹性参数，供 build_hybrid_gll.py 插值到 SPECFEM mesh0。

Salvus MODEL/data 维度标签:
  [ ETA | QKAPPA | QMU | RHO | VPH | VPV | VSH | VSV | z_node_1D ]

单位: 坐标 m (ECEF), 速度 m/s, 密度 kg/m³ → 输出 km/s, g/cm³
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger('EASTASIA-FWI.LoadSalvusMesh')

# Salvus data 通道下标
_SALVUS_IDX: Dict[str, int] = {
    'eta': 0,
    'rho': 3,
    'vph': 4,
    'vpv': 5,
    'vsh': 6,
    'vsv': 7,
}

PARAMS: Tuple[str, ...] = ('vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho')


@dataclass
class SalvusPointCloud:
    """Salvus GLL 点云，duck-type 兼容 ModelGrid 的范围属性。"""

    xyz: np.ndarray                    # (N, 3) float32, ECEF m
    arrays: Dict[str, np.ndarray]      # 各参数 (N,) float32, km/s 或 g/cm³
    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]
    depth_range: Tuple[float, float]
    _tree: Optional[cKDTree] = None
    k_neighbors: int = 4

    def _ensure_tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.xyz)
        return self._tree

    def interpolate(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        depth: np.ndarray,
        earth_radius_m: float = 6371000.0,
    ) -> Dict[str, np.ndarray]:
        """
        将查询点 (lat, lon, depth km) 转为 ECEF，k-NN 反距离加权插值。

        Args:
            lat: 地心纬度（度）
            lon: 经度（度）
            depth: 球面深度（km，向下为正）
            earth_radius_m: 地球半径（m）

        Returns:
            六个弹性参数字典，单位 km/s 或 g/cm³
        """
        lat_r = np.radians(lat.astype(np.float64))
        lon_r = np.radians(lon.astype(np.float64))
        r_m = earth_radius_m - depth.astype(np.float64) * 1000.0
        cos_lat = np.cos(lat_r)
        qx = r_m * cos_lat * np.cos(lon_r)
        qy = r_m * cos_lat * np.sin(lon_r)
        qz = r_m * np.sin(lat_r)
        query = np.column_stack([qx, qy, qz]).astype(np.float32)

        tree = self._ensure_tree()
        k = min(self.k_neighbors, len(self.xyz))
        dist, idx = tree.query(query, k=k)
        if k == 1:
            dist = dist[:, np.newaxis]
            idx = idx[:, np.newaxis]

        # 反距离加权；重合点 dist=0
        with np.errstate(divide='ignore'):
            w = 1.0 / np.maximum(dist, 1e-6)
        w = np.where(dist <= 1e-6, 1e12, w)
        w /= w.sum(axis=1, keepdims=True)

        out: Dict[str, np.ndarray] = {}
        for p in PARAMS:
            vals = self.arrays[p][idx]  # (nq, k)
            out[p] = np.sum(w * vals, axis=1).astype(np.float64)
        # Salvus 反演 eta 可偏离 1 较多；SPECFEM 稳定运行需限制在物理合理区间
        out['eta'] = np.clip(out['eta'], 0.5, 1.5)
        return out


def _cache_key(
    mesh_path: Path,
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
    depth_bounds: Tuple[float, float],
    subsample: int,
) -> str:
    st = mesh_path.stat()
    raw = (
        f"{mesh_path.resolve()}|{st.st_size}|{st.st_mtime}|"
        f"{lat_bounds}|{lon_bounds}|{depth_bounds}|{subsample}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _lon_in_bounds(lon: np.ndarray, lon_min: float, lon_max: float) -> np.ndarray:
    """经度区间判断，支持跨 0° 或 180°。"""
    lon = np.mod(lon + 360.0, 360.0)
    lo = lon_min % 360.0
    hi = lon_max % 360.0
    if lo <= hi:
        return (lon >= lo) & (lon <= hi)
    return (lon >= lo) | (lon <= hi)


def load_salvus_point_cloud(
    mesh_path: Path,
    lat_bounds: Tuple[float, float] = (-15.0, 65.0),
    lon_bounds: Tuple[float, float] = (45.0, 170.0),
    depth_bounds_km: Tuple[float, float] = (-20.0, 1100.0),
    subsample: int = 1,
    cache_dir: Optional[Path] = None,
    chunk_elems: int = 80_000,
) -> SalvusPointCloud:
    """
    从 Salvus mesh.h5 构建区域点云；支持磁盘缓存。

    Args:
        mesh_path: mesh.h5 路径
        lat_bounds: 保留纬度范围（度）
        lon_bounds: 保留经度范围（度，-180~180 或 0~360 均可）
        depth_bounds_km: 保留深度范围（km，相对球面）
        subsample: 单元 stride 降采样（>1 时仅用于快速测试）
        cache_dir: 缓存目录，默认 mesh 同目录
        chunk_elems: HDF5 分块读取单元数

    Returns:
        SalvusPointCloud
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Salvus mesh 不存在: {mesh_path}")

    cache_dir = cache_dir or mesh_path.parent
    key = _cache_key(mesh_path, lat_bounds, lon_bounds, depth_bounds_km, subsample)
    cache_npz = cache_dir / f"{mesh_path.stem}.salvus_{key}.npz"

    if cache_npz.exists():
        logger.info('加载 Salvus 点云缓存: %s', cache_npz.name)
        z = np.load(cache_npz)
        cloud = SalvusPointCloud(
            xyz=z['xyz'],
            arrays={p: z[p] for p in PARAMS},
            lat_range=(float(z['lat_min']), float(z['lat_max'])),
            lon_range=(float(z['lon_min']), float(z['lon_max'])),
            depth_range=(float(z['dep_min']), float(z['dep_max'])),
        )
        logger.info('  点数 %s', f"{len(cloud.xyz):,}")
        return cloud

    logger.info('从 Salvus mesh 构建点云: %s', mesh_path.name)
    logger.info('  区域 lat %s lon %s depth %s km',
                lat_bounds, lon_bounds, depth_bounds_km)

    xyz_parts: list[np.ndarray] = []
    param_parts: Dict[str, list[np.ndarray]] = {p: [] for p in PARAMS}

    with h5py.File(mesh_path, 'r') as f:
        m = f['MODEL']
        radius = float(m.attrs['radius'])
        coords_d = m['coordinates']
        data_d = m['data']
        n_elem = coords_d.shape[0]
        logger.info('  单元数 %s, GLL/单元 %d', f"{n_elem:,}", coords_d.shape[1])

        elem_indices = np.arange(0, n_elem, subsample, dtype=np.int64)
        for i0 in range(0, len(elem_indices), chunk_elems):
            sel = elem_indices[i0:i0 + chunk_elems]
            coords = coords_d[sel]          # (chunk, 27, 3)
            data = data_d[sel]              # (chunk, 9, 27)

            xyz = coords.reshape(-1, 3).astype(np.float64)
            r = np.linalg.norm(xyz, axis=1)
            lat = np.degrees(np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1.0), -1.0, 1.0)))
            lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
            depth = (radius - r) / 1000.0

            mask = (
                (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
                & _lon_in_bounds(lon, lon_bounds[0], lon_bounds[1])
                & (depth >= depth_bounds_km[0]) & (depth <= depth_bounds_km[1])
            )
            if not mask.any():
                continue

            xyz_parts.append(xyz[mask].astype(np.float32))
            flat_data = data.reshape(data.shape[0], data.shape[1], -1)  # (chunk, 9, 27)
            for p in PARAMS:
                ch = _SALVUS_IDX[p]
                vals = flat_data[:, ch, :].reshape(-1)[mask]
                if p == 'rho':
                    vals = vals / 1000.0          # kg/m³ → g/cm³
                else:
                    vals = vals / 1000.0          # m/s → km/s
                param_parts[p].append(vals.astype(np.float32))

            if (i0 // chunk_elems) % 5 == 0:
                n_so_far = sum(len(a) for a in xyz_parts)
                logger.info('    已扫描 %d/%d 单元, 保留 %s 点',
                            min(i0 + chunk_elems, len(elem_indices)),
                            len(elem_indices), f"{n_so_far:,}")

    if not xyz_parts:
        raise ValueError('区域过滤后 Salvus 点云为空，请放宽 lat/lon/depth 边界')

    xyz_all = np.concatenate(xyz_parts, axis=0)
    arrays = {p: np.concatenate(param_parts[p], axis=0) for p in PARAMS}

    r = np.linalg.norm(xyz_all.astype(np.float64), axis=1)
    lat_all = np.degrees(np.arcsin(np.clip(xyz_all[:, 2] / np.maximum(r, 1.0), -1.0, 1.0)))
    lon_all = np.degrees(np.arctan2(xyz_all[:, 1], xyz_all[:, 0]))
    dep_all = (6371000.0 - r) / 1000.0  # 近似，仅用于范围报告

    cloud = SalvusPointCloud(
        xyz=xyz_all,
        arrays=arrays,
        lat_range=(float(lat_all.min()), float(lat_all.max())),
        lon_range=(float(lon_all.min()), float(lon_all.max())),
        depth_range=(float(dep_all.min()), float(dep_all.max())),
    )
    logger.info('  完成: %s 点, lat %s lon %s depth %s km',
                f"{len(cloud.xyz):,}", cloud.lat_range, cloud.lon_range, cloud.depth_range)

    np.savez_compressed(
        cache_npz,
        xyz=cloud.xyz,
        **cloud.arrays,
        lat_min=cloud.lat_range[0],
        lat_max=cloud.lat_range[1],
        lon_min=cloud.lon_range[0],
        lon_max=cloud.lon_range[1],
        dep_min=cloud.depth_range[0],
        dep_max=cloud.depth_range[1],
    )
    logger.info('  缓存已写: %s (%.1f MB)', cache_npz.name, cache_npz.stat().st_size / 1e6)
    return cloud


def compare_with_public_h5(
    salvus: SalvusPointCloud,
    h5_path: Path,
    n_sample: int = 5000,
    seed: int = 42,
) -> None:
    """在 Salvus 点云上抽样，与公开规则网格 H5 对比速度差。"""
    from convert_nc_to_gll import load_h5_as_dataset
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(salvus.xyz), size=min(n_sample, len(salvus.xyz)), replace=False)
    xyz = salvus.xyz[idx].astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2] / r, -1, 1)))
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    depth = (6371000.0 - r) / 1000.0

    ds = load_h5_as_dataset(h5_path)
    depth_ax = np.asarray(ds['depth'].values, dtype=np.float64)
    lat_ax = np.asarray(ds['latitude'].values, dtype=np.float64)
    lon_ax = np.asarray(ds['longitude'].values, dtype=np.float64)
    vsv_h5 = np.asarray(ds['vsv'].values, dtype=np.float64)

    interp = RegularGridInterpolator(
        (depth_ax, lat_ax, lon_ax), vsv_h5,
        method='linear', bounds_error=False, fill_value=np.nan,
    )
    lon_q = np.mod(lon + 360.0, 360.0)
    lon_ax_n = np.mod(lon_ax, 360.0)
    interp2 = RegularGridInterpolator(
        (depth_ax, lat_ax, lon_ax_n), vsv_h5,
        method='linear', bounds_error=False, fill_value=np.nan,
    )
    vsv_pub = interp2(np.column_stack([depth, lat, lon_q]))

    vsv_sem = salvus.arrays['vsv'][idx].astype(np.float64)
    ok = np.isfinite(vsv_pub) & (depth >= 20.0)
    if ok.sum() == 0:
        print('无有效对比点（深度>=20 km）')
        return
    diff = vsv_sem[ok] - vsv_pub[ok]
    rel = diff / vsv_pub[ok] * 100.0
    print(f'VSV Salvus vs 公开H5 (N={ok.sum()}):')
    print(f'  绝对差 km/s: mean={diff.mean():+.4f}  std={diff.std():.4f}  '
          f'max|.|={np.abs(diff).max():.4f}')
    print(f'  相对差 %:    mean={rel.mean():+.2f}  std={rel.std():.2f}  '
          f'max|.|={np.abs(rel).max():.2f}')
    ds.close()


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    ap = argparse.ArgumentParser(description='Salvus mesh.h5 点云加载与对比')
    ap.add_argument('--mesh-h5', type=Path, required=True)
    ap.add_argument('--public-h5', type=Path, default=None)
    ap.add_argument('--subsample', type=int, default=1)
    args = ap.parse_args()

    cloud = load_salvus_point_cloud(args.mesh_h5, subsample=args.subsample)
    print(f'点云: {len(cloud.xyz):,} 点')
    print(f'VSV range: {cloud.arrays["vsv"].min():.3f} .. {cloud.arrays["vsv"].max():.3f} km/s')
    if args.public_h5:
        compare_with_public_h5(cloud, args.public_h5)
