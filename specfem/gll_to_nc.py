"""
EASTASIA-FWI 原生 GLL → 规则网格 NetCDF
========================================

将 SPECFEM GLL 二进制（如 model_updated/）按 DATABASES_MPI 坐标
插值到与公开 EMC NetCDF 相同的 (depth, latitude, longitude) 网格，
用于量化「原生 GLL → 规则网格导出」相对公开 FWEA23 的差异。

科学用途:
  - 估计公开 NC 相对作者原生终模型的导出损失
  - 与 NC→GLL（convert_nc_to_gll.py）往返误差互补，不替代 T2/T3

用法:
    conda activate eastasia_fwi
    cd specfem

    # 默认: model_updated → 按 FWEA23.r0.0-n4.nc 网格写出并对比
    python gll_to_nc.py \\
        --mesh-dir FWEA23/DATABASES_MPI \\
        --gll-dir model_updated \\
        --template-nc models/FWEA23.r0.0-n4.nc \\
        --output-nc models/FWEA23_from_GLL.nc

    # 只写出、不对比
    python gll_to_nc.py ... --no-compare

    # 快速试跑（前 12 个 proc）
    python gll_to_nc.py ... --max-procs 12

作者: EASTASIA-FWI Team
日期: 2026-08
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

SCRIPT_DIR = Path(__file__).parent
NGLLX = NGLLY = NGLLZ = 5
NGLL3 = 125
R_EARTH_KM = 6371.0

DEFAULT_PARAMS = ['vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho', 'qmu']
PARAM_UNITS = {
    'vpv': 'km/s', 'vph': 'km/s', 'vsv': 'km/s', 'vsh': 'km/s',
    'eta': '1', 'rho': 'g/cm3', 'qmu': '1',
    'alpha': 'km/s', 'beta': 'km/s',
}
DEPTH_BINS = [
    (0, 35), (35, 100), (100, 220), (220, 410),
    (410, 660), (660, 1000),
]
FILL_VALUE = 9999.0


# ─── I/O ─────────────────────────────────────────────────────
def _read_fortran_record(f) -> bytes:
    rec_len = struct.unpack('i', f.read(4))[0]
    data = f.read(rec_len)
    tail = struct.unpack('i', f.read(4))[0]
    if tail != rec_len:
        raise ValueError(f"Fortran record marker mismatch: {rec_len} vs {tail}")
    return data


def read_gll_flat(path: Path) -> NDArray[np.float32]:
    """读取 proc***_reg1_*.bin → 1D float32（NGLL3 × nspec）。"""
    with open(path, 'rb') as f:
        rec_len = struct.unpack('i', f.read(4))[0]
        payload = f.read(rec_len)
        f.read(4)
    return np.frombuffer(payload, dtype=np.float32).copy()


def read_solver_xyz(solver_path: Path
                    ) -> Tuple[int, int, NDArray, NDArray, NDArray, NDArray]:
    """读取 solver_data.bin 前段：nspec, nglob, x, y, z, ibool(1-based)。"""
    with open(solver_path, 'rb') as f:
        nspec = struct.unpack('i', _read_fortran_record(f))[0]
        nglob = struct.unpack('i', _read_fortran_record(f))[0]
        x_buf = _read_fortran_record(f)
        y_buf = _read_fortran_record(f)
        z_buf = _read_fortran_record(f)
        dtype_xyz = np.float32 if len(x_buf) == nglob * 4 else np.float64
        x = np.frombuffer(x_buf, dtype=dtype_xyz).copy()
        y = np.frombuffer(y_buf, dtype=dtype_xyz).copy()
        z = np.frombuffer(z_buf, dtype=dtype_xyz).copy()
        ibool = np.frombuffer(_read_fortran_record(f), dtype=np.int32)
        ibool = ibool.reshape((NGLLX, NGLLY, NGLLZ, nspec), order='F')
    return nspec, nglob, x, y, z, ibool


def xyz_to_lat_lon_depth(
    x: NDArray, y: NDArray, z: NDArray
) -> Tuple[NDArray, NDArray, NDArray]:
    """SPECFEM 非量纲 (x,y,z) → lat°, lon°[0,360), depth_km。"""
    r = np.maximum(np.sqrt(x ** 2 + y ** 2 + z ** 2), 1e-6)
    lat = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    lon = np.mod(lon, 360.0)
    depth = R_EARTH_KM * (1.0 - r)
    return lat, lon, depth


# ─── 收集 GLL 唯一点 ─────────────────────────────────────────
def discover_procs(mesh_dir: Path, gll_dir: Path, probe: str = 'vpv') -> List[int]:
    mesh_procs = {int(p.name[4:10]) for p in mesh_dir.glob('proc*_reg1_solver_data.bin')}
    gll_procs = {int(p.name[4:10]) for p in gll_dir.glob(f'proc*_reg1_{probe}.bin')}
    common = sorted(mesh_procs & gll_procs)
    if not common:
        raise FileNotFoundError(
            f"未找到共有 proc: mesh={mesh_dir}, gll={gll_dir} (probe={probe})"
        )
    return common


def discover_params(gll_dir: Path, requested: Sequence[str]) -> List[str]:
    available = {
        p.name.split('_reg1_')[1].replace('.bin', '')
        for p in gll_dir.glob('proc000000_reg1_*.bin')
    }
    params = [p for p in requested if p in available]
    if not params:
        raise FileNotFoundError(f"GLL 目录无请求参数: {requested} @ {gll_dir}")
    return params


def collect_unique_gll_points(
    mesh_dir: Path,
    gll_dir: Path,
    params: Sequence[str],
    procs: Sequence[int],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    depth_min: float,
    depth_max: float,
    margin_deg: float = 0.5,
    margin_km: float = 15.0,
) -> Tuple[NDArray, NDArray, NDArray, Dict[str, NDArray]]:
    """
    收集落在模板网格邻域内的 GLL 唯一点。

    每个 proc 内按 ibool 去重；跨 proc 再按圆整坐标去重，控制内存。
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    lon0 = lon_min - margin_deg
    lon1 = lon_max + margin_deg
    lat0 = lat_min - margin_deg
    lat1 = lat_max + margin_deg
    z0 = depth_min - margin_km
    z1 = depth_max + margin_km

    lat_chunks: List[NDArray] = []
    lon_chunks: List[NDArray] = []
    dep_chunks: List[NDArray] = []
    val_chunks: Dict[str, List[NDArray]] = {p: [] for p in params}

    iterator = tqdm(procs, desc='读取 GLL', unit='proc') if tqdm else procs
    n_kept = 0
    for pid in iterator:
        solver = mesh_dir / f'proc{pid:06d}_reg1_solver_data.bin'
        nspec, nglob, x, y, z, ibool = read_solver_xyz(solver)
        lat_g, lon_g, dep_g = xyz_to_lat_lon_depth(x, y, z)
        iglob = ibool.ravel(order='F') - 1  # 1-based → 0-based
        unique_iglob, first_idx = np.unique(iglob, return_index=True)

        lat_u = lat_g[unique_iglob]
        lon_u = lon_g[unique_iglob]
        dep_u = dep_g[unique_iglob]
        # 与模板经度约定对齐（FWEA23 为 65–150，保持 [0,360)）
        if lon_min >= 0.0:
            lon_u = np.mod(lon_u, 360.0)
        else:
            lon_u = np.where(lon_u > 180.0, lon_u - 360.0, lon_u)

        mask = (
            (lon_u >= lon0) & (lon_u <= lon1)
            & (lat_u >= lat0) & (lat_u <= lat1)
            & (dep_u >= z0) & (dep_u <= z1)
        )
        if not np.any(mask):
            continue

        sel = unique_iglob[mask]
        first_sel = first_idx[mask]
        lat_chunks.append(lat_u[mask].astype(np.float64))
        lon_chunks.append(lon_u[mask].astype(np.float64))
        dep_chunks.append(dep_u[mask].astype(np.float64))

        for p in params:
            path = gll_dir / f'proc{pid:06d}_reg1_{p}.bin'
            flat = read_gll_flat(path)
            if flat.size != NGLL3 * nspec:
                raise ValueError(
                    f"{path.name}: 点数 {flat.size} ≠ NGLL3*nspec={NGLL3 * nspec}"
                )
            val_chunks[p].append(flat[first_sel].astype(np.float64))
        n_kept += int(mask.sum())

    if not lat_chunks:
        raise RuntimeError('域内无 GLL 点：请检查 mesh / 模板网格范围是否匹配')

    lat = np.concatenate(lat_chunks)
    lon = np.concatenate(lon_chunks)
    dep = np.concatenate(dep_chunks)
    vals = {p: np.concatenate(val_chunks[p]) for p in params}
    print(f'  初收集（域内唯一点/proc）: {n_kept:,}')

    # 跨进程去重：圆整到约 10 m / 0.001°
    key = np.stack([
        np.round(lat / 0.001),
        np.round(lon / 0.001),
        np.round(dep / 0.01),
    ], axis=1).astype(np.int64)
    # 结构化唯一
    uniq_view = np.ascontiguousarray(key).view(
        np.dtype((np.void, key.dtype.itemsize * key.shape[1]))
    )
    _, uniq_idx = np.unique(uniq_view, return_index=True)
    uniq_idx.sort()
    lat = lat[uniq_idx]
    lon = lon[uniq_idx]
    dep = dep[uniq_idx]
    vals = {p: vals[p][uniq_idx] for p in params}
    print(f'  跨 proc 去重后: {len(lat):,}')
    return lat, lon, dep, vals


# ─── 插值到规则网格 ───────────────────────────────────────────
def _scaled_coords(
    lon: NDArray, lat: NDArray, depth: NDArray, lat_ref: float
) -> NDArray:
    """各向异性尺度：水平 ~度，深度使 10 km ≈ 0.25°。"""
    cos_ref = max(np.cos(np.radians(lat_ref)), 0.2)
    depth_to_deg = 0.25 / 10.0  # 10 km → 0.25°
    return np.column_stack([
        lon * cos_ref,
        lat,
        depth * depth_to_deg,
    ])


def interpolate_idw_multi(
    lon_src: NDArray,
    lat_src: NDArray,
    dep_src: NDArray,
    values: Dict[str, NDArray],
    lon_t: NDArray,
    lat_t: NDArray,
    dep_t: NDArray,
    k: int = 8,
    max_dist_deg: float = 1.5,
    chunk: int = 200_000,
) -> Dict[str, NDArray]:
    """
    cKDTree + 反距离加权；一次邻域查询，同时插值多个参数。

    每个参数输出 shape (ndepth, nlat, nlon)；超出 max_dist_deg 填 NaN。
    """
    from scipy.spatial import cKDTree

    lat_ref = float(np.median(lat_t))
    tree = cKDTree(_scaled_coords(lon_src, lat_src, dep_src, lat_ref))

    nlon, nlat, ndep = len(lon_t), len(lat_t), len(dep_t)
    total = ndep * nlat * nlon
    tgt_lon = np.empty(total, dtype=np.float64)
    tgt_lat = np.empty(total, dtype=np.float64)
    tgt_dep = np.empty(total, dtype=np.float64)
    idx = 0
    for id_ in range(ndep):
        for ila in range(nlat):
            sl = slice(idx, idx + nlon)
            tgt_lon[sl] = lon_t
            tgt_lat[sl] = lat_t[ila]
            tgt_dep[sl] = dep_t[id_]
            idx += nlon

    coords_t = _scaled_coords(tgt_lon, tgt_lat, tgt_dep, lat_ref)
    param_names = list(values.keys())
    flat_out = {p: np.full(total, np.nan, dtype=np.float64) for p in param_names}

    try:
        from tqdm import tqdm
        ranges = tqdm(range(0, total, chunk), desc='IDW 插值', unit='chunk')
    except ImportError:
        ranges = range(0, total, chunk)

    eps = 1e-12
    for start in ranges:
        stop = min(start + chunk, total)
        dist, nn = tree.query(coords_t[start:stop], k=k, workers=-1)
        if k == 1:
            dist = dist[:, None]
            nn = nn[:, None]
        exact = dist[:, 0] < 1e-12
        w = 1.0 / np.maximum(dist, eps) ** 2
        w[~np.isfinite(w)] = 0.0
        too_far = dist[:, 0] > max_dist_deg
        den = np.sum(w, axis=1)
        for p in param_names:
            nn_vals = values[p][nn]
            est = np.sum(w * nn_vals, axis=1) / np.maximum(den, eps)
            est[exact] = values[p][nn[exact, 0]]
            est[too_far] = np.nan
            flat_out[p][start:stop] = est

    shape = (ndep, nlat, nlon)
    return {p: flat_out[p].reshape(shape).astype(np.float32) for p in param_names}


def voigt_alpha_beta(
    vpv: NDArray, vph: NDArray, vsv: NDArray, vsh: NDArray
) -> Tuple[NDArray, NDArray]:
    """各向同性 Voigt 平均（与常见 EMC 导出一致）。"""
    alpha = np.sqrt((vpv ** 2 + 4.0 * vph ** 2) / 5.0)
    beta = np.sqrt((2.0 * vsv ** 2 + vsh ** 2) / 3.0)
    return alpha.astype(np.float32), beta.astype(np.float32)


# ─── 写 NetCDF / 对比 ─────────────────────────────────────────
def write_netcdf(
    path: Path,
    lon: NDArray,
    lat: NDArray,
    depth: NDArray,
    fields: Dict[str, NDArray],
    template_attrs: Optional[dict] = None,
    fill_value: float = FILL_VALUE,
) -> None:
    import xarray as xr

    data_vars = {}
    for name, arr in fields.items():
        # NaN → fill
        out = np.array(arr, dtype=np.float32, copy=True)
        out[~np.isfinite(out)] = fill_value
        data_vars[name] = xr.DataArray(
            out,
            dims=('depth', 'latitude', 'longitude'),
            coords={
                'depth': depth,
                'latitude': lat,
                'longitude': lon,
            },
            attrs={
                'units': PARAM_UNITS.get(name, ''),
                'long_name': name,
                'missing_value': fill_value,
            },
        )

    ds = xr.Dataset(data_vars)
    ds['depth'].attrs.update({'units': 'km', 'positive': 'down', 'long_name': 'depth'})
    ds['latitude'].attrs.update({'units': 'degrees_north', 'long_name': 'latitude'})
    ds['longitude'].attrs.update({'units': 'degrees_east', 'long_name': 'longitude'})

    attrs = {
        'title': 'FWEA23 resampled from native SPECFEM GLL (model_updated)',
        'summary': (
            'Regular-grid velocity model interpolated from author native GLL '
            'bins using inverse-distance weighting in (lon, lat, depth). '
            'Intended for quantifying EMC export loss vs public FWEA23 NetCDF.'
        ),
        'Conventions': 'CF-1.0',
        'source_gll': 'model_updated (SPECFEM3D Globe proc*_reg1_*.bin)',
        'interpolation': 'cKDTree IDW (k neighbors)',
        'geospatial_vertical_positive': 'down',
    }
    if template_attrs:
        for key in ('id', 'reference', 'author_name', 'author_institution'):
            if key in template_attrs:
                attrs[f'template_{key}'] = template_attrs[key]
    ds.attrs.update(attrs)

    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {
            'dtype': 'float32',
            'zlib': True,
            'complevel': 4,
            '_FillValue': fill_value,
        }
        for name in fields
    }
    ds.to_netcdf(path, encoding=encoding)
    ds.close()
    print(f'  写出: {path} ({path.stat().st_size / 1e6:.1f} MB)')


def compare_to_template(
    fields: Dict[str, NDArray],
    template_nc: Path,
    depth: NDArray,
    report_path: Path,
    fill_value: float = FILL_VALUE,
) -> None:
    """与公开 NC 对比（仅双方均有限值的格点）。"""
    import xarray as xr

    ds = xr.open_dataset(template_nc)
    lines: List[str] = []
    lines.append('GLL→NC vs 公开 EMC NetCDF 对比报告')
    lines.append(f'模板: {template_nc}')
    lines.append('=' * 72)

    print('\n' + '=' * 72)
    print('  GLL→NC vs 公开 FWEA23')
    print('=' * 72)

    for name, arr in fields.items():
        if name not in ds:
            continue
        ref = ds[name].values.astype(np.float64)
        ours = arr.astype(np.float64)
        valid = (
            np.isfinite(ours) & np.isfinite(ref)
            & (ours != fill_value) & (ref != fill_value)
            & (np.abs(ref) < 0.5 * fill_value)
            & (np.abs(ours) < 0.5 * fill_value)
        )
        if valid.sum() < 10:
            msg = f'{name}: 有效对比点不足'
            print(f'  {msg}')
            lines.append(msg)
            continue

        diff = ours[valid] - ref[valid]
        abs_d = np.abs(diff)
        rel = abs_d / np.maximum(np.abs(ref[valid]), 1e-6) * 100.0
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        corr = float(np.corrcoef(ours[valid], ref[valid])[0, 1])
        header = (
            f'\n{name.upper():6s}  n={int(valid.sum()):,}  '
            f'mean|Δ|={abs_d.mean():.4f}  RMSE={rmse:.4f}  '
            f'rel%={rel.mean():.3f}  R={corr:.5f}  bias={diff.mean():+.4f}'
        )
        print(header)
        lines.append(header)

        # 分深度（用 depth 坐标掩码）
        dep3 = np.broadcast_to(depth.reshape(-1, 1, 1), ours.shape)
        print(f'  {"深度段":<12} {"点数":>12} {"mean|Δ|":>10} {"rel%":>8}')
        lines.append(f'  {"深度段":<12} {"点数":>12} {"mean|Δ|":>10} {"rel%":>8}')
        for z0, z1 in DEPTH_BINS:
            m = valid & (dep3 >= z0) & (dep3 < z1)
            if m.sum() < 10:
                continue
            d = ours[m] - ref[m]
            ad = np.abs(d)
            rp = ad / np.maximum(np.abs(ref[m]), 1e-6) * 100.0
            row = f'  {z0:4d}-{z1:<4d} km {int(m.sum()):12,} {ad.mean():10.4f} {rp.mean():8.3f}'
            print(row)
            lines.append(row)

    ds.close()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n  报告 → {report_path}')


# ─── CLI ─────────────────────────────────────────────────────
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='原生 GLL → 规则网格 NetCDF（对齐公开 FWEA23 网格）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python gll_to_nc.py \\
      --mesh-dir FWEA23/DATABASES_MPI \\
      --gll-dir model_updated \\
      --template-nc models/FWEA23.r0.0-n4.nc \\
      --output-nc models/FWEA23_from_GLL.nc
""",
    )
    parser.add_argument('--mesh-dir', type=str, required=True,
                        help='DATABASES_MPI（含 solver_data.bin，提供坐标）')
    parser.add_argument('--gll-dir', type=str, default=str(SCRIPT_DIR / 'model_updated'),
                        help='原生 GLL 目录（默认 model_updated/）')
    parser.add_argument('--template-nc', type=str,
                        default=str(SCRIPT_DIR / 'models' / 'FWEA23.r0.0-n4.nc'),
                        help='公开 NC，提供目标 lon/lat/depth 网格')
    parser.add_argument('--output-nc', type=str,
                        default=str(SCRIPT_DIR / 'models' / 'FWEA23_from_GLL.nc'),
                        help='输出 NetCDF 路径')
    parser.add_argument('--params', type=str, default=','.join(DEFAULT_PARAMS),
                        help='逗号分隔参数列表')
    parser.add_argument('--k', type=int, default=8, help='IDW 邻居数（默认 8）')
    parser.add_argument('--max-dist-deg', type=float, default=1.5,
                        help='尺度空间最大距离，超出填 NaN/fill')
    parser.add_argument('--max-procs', type=int, default=None,
                        help='仅用前 N 个 proc（调试）')
    parser.add_argument('--no-compare', action='store_true',
                        help='只写出 NC，不与模板对比')
    parser.add_argument('--report', type=str,
                        default=str(SCRIPT_DIR / 'figures' / 'gll_to_nc' / 'report.txt'),
                        help='对比报告路径')
    parser.add_argument('--no-voigt', action='store_true',
                        help='不写 alpha/beta Voigt 平均')
    args = parser.parse_args(argv)

    mesh_dir = Path(args.mesh_dir)
    gll_dir = Path(args.gll_dir)
    template_nc = Path(args.template_nc)
    output_nc = Path(args.output_nc)

    if not mesh_dir.is_dir():
        print(f'❌ mesh-dir 不存在: {mesh_dir}')
        return 1
    if not gll_dir.is_dir():
        print(f'❌ gll-dir 不存在: {gll_dir}')
        return 1
    if not template_nc.is_file():
        print(f'❌ template-nc 不存在: {template_nc}')
        return 1

    import xarray as xr

    print('🎯 GLL → 规则网格 NetCDF')
    print(f'   mesh:     {mesh_dir}')
    print(f'   gll:      {gll_dir}')
    print(f'   template: {template_nc}')
    print(f'   output:   {output_nc}')

    ds_t = xr.open_dataset(template_nc)
    lon_t = ds_t['longitude'].values.astype(np.float64)
    lat_t = ds_t['latitude'].values.astype(np.float64)
    dep_t = ds_t['depth'].values.astype(np.float64)
    # 经度统一到与 GLL 收集一致的范围
    if lon_t.min() < 0 and lon_t.max() <= 180:
        pass  # already [-180,180] style
    template_attrs = dict(ds_t.attrs)
    print(f'   网格: lon {lon_t.min():.2f}~{lon_t.max():.2f} ({len(lon_t)}), '
          f'lat {lat_t.min():.2f}~{lat_t.max():.2f} ({len(lat_t)}), '
          f'depth {dep_t.min():.1f}~{dep_t.max():.1f} ({len(dep_t)})')

    req_params = [p.strip() for p in args.params.split(',') if p.strip()]
    params = discover_params(gll_dir, req_params)
    print(f'   参数: {params}')

    procs = discover_procs(mesh_dir, gll_dir, probe=params[0])
    if args.max_procs is not None:
        procs = procs[: args.max_procs]
        print(f'   ⚠ 调试模式: 仅 {len(procs)} 个 proc')
    else:
        print(f'   进程数: {len(procs)}')

    lat_s, lon_s, dep_s, vals = collect_unique_gll_points(
        mesh_dir, gll_dir, params, procs,
        lon_min=float(lon_t.min()), lon_max=float(lon_t.max()),
        lat_min=float(lat_t.min()), lat_max=float(lat_t.max()),
        depth_min=float(dep_t.min()), depth_max=float(dep_t.max()),
    )

    print(f'\n▶ IDW 插值 ({len(params)} 参数，一次邻域查询) ...')
    fields = interpolate_idw_multi(
        lon_s, lat_s, dep_s, vals,
        lon_t, lat_t, dep_t,
        k=args.k, max_dist_deg=args.max_dist_deg,
    )
    probe = params[0]
    n_valid = int(np.isfinite(fields[probe]).sum())
    n_all = fields[probe].size
    print(f'   有效格点 ({probe}): {n_valid:,}/{n_all:,} ({100 * n_valid / n_all:.1f}%)')

    if (not args.no_voigt
            and all(k in fields for k in ('vpv', 'vph', 'vsv', 'vsh'))):
        fields['alpha'], fields['beta'] = voigt_alpha_beta(
            fields['vpv'], fields['vph'], fields['vsv'], fields['vsh']
        )
        print('   已计算 Voigt alpha/beta')

    # 从模板拷贝 vp0/vs0（参考场，非 GLL 插值）
    for ref_name in ('vp0', 'vs0'):
        if ref_name in ds_t:
            fields[ref_name] = ds_t[ref_name].values.astype(np.float32)
            print(f'   已从模板拷贝 {ref_name}（非 GLL 插值）')

    ds_t.close()

    print('\n▶ 写出 NetCDF ...')
    write_netcdf(
        output_nc, lon_t, lat_t, dep_t, fields,
        template_attrs=template_attrs, fill_value=FILL_VALUE,
    )

    if not args.no_compare:
        cmp_fields = {k: v for k, v in fields.items() if k not in ('vp0', 'vs0')}
        compare_to_template(
            cmp_fields, template_nc, dep_t, Path(args.report), FILL_VALUE,
        )

    print('\n✅ 完成')
    print('说明: 本对比度量的是「原生 GLL 重采样到规则网格」与公开 NC 的差异，')
    print('      即 EMC 导出损失的上界估计；不能替代 T2/T3 波形检验。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
