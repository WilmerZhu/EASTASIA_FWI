#!/usr/bin/env python3
"""
EASTASIA-FWI 混合模型 GLL 构建
================================

以一个已知正确的 GLL 模型为底图（默认 FWEA23 原作者的 model_updated/），
将另一个公开速度模型的值嵌入指定区域，输出可直接供 MODEL=GLL 正演使用的
proc***_reg1_{vpv,vph,vsv,vsh,eta,rho,qmu}.bin。

设计要点:
  1. 底图现成，不需要 mesher 计算参考模型，也不需要从弹性模量反算速度。
     solver_data.bin 只读文件开头的坐标记录（无条件写入，不受 Par_file 开关影响）。
  2. 六个弹性参数 (vpv/vph/vsv/vsh/eta/rho) 必须整组来自同一模型，且共用同一
     权重场。混搭会破坏 Vp/Vs 一致性，极端情况下体积模量为负导致 solver 发散。
  3. 替换区边界用升余弦 taper 向内渐变，w 在 --region / --depth-range 指定的
     外边界上恰好为 0，保证过渡完整落在模型数据域内部，不产生人工散射体。
  4. qmu 一律保留底图（EARA2024 / SinoScope 均未发布 Q 模型）。

权重定义（taper 向内）:

    w = w_horiz(lat, lon) x w_depth(depth) x in_model_domain

    w_horiz : 在 --region 的四条边上为 0，向内经 --horiz-taper 度升到 1
    w_depth : 在 --depth-range 的上下界上为 0，向内经 --depth-taper 升到 1

    输出 V = w * V_model + (1 - w) * V_base

用法:
    # 基准 run：不指定模型文件，等价于把底图原样拷贝一份
    python build_hybrid_gll.py --mesh-dir mesh_coords \\
        --base-gll-dir model_updated --output-dir BASE/DATA/GLL

    # 嵌入 EARA2024
    python build_hybrid_gll.py --mesh-dir mesh_coords \\
        --base-gll-dir model_updated --output-dir EARA2024_hy/DATA/GLL \\
        --model-nc models/EARA2024.r0.0-n4.nc

    # 嵌入 SinoScope1.0 (HDF5)
    python build_hybrid_gll.py --mesh-dir mesh_coords \\
        --base-gll-dir model_updated --output-dir SinoScope_hy/DATA/GLL \\
        --model-h5 models/FWI_SinoScope_1.0_JMa_ET_AL_2022.h5

    # 覆盖诊断：只统计不写文件
    python build_hybrid_gll.py ... --model-nc ... --dry-run

配套文档: Models_Hybrid_workflow.md

作者: EASTASIA-FWI Team
日期: 2026-08
版本: v1.0
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import struct
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from convert_nc_to_gll import (  # noqa: E402
    NGLLX,
    NGLLY,
    NGLLZ,
    R_EARTH_KM,
    _fill_nan_nearest_3d,
    load_h5_as_dataset,
    read_gll_binary,
    read_solver_data,
    write_gll_binary,
)
from load_salvus_mesh import SalvusPointCloud, load_salvus_point_cloud  # noqa: E402

# ── 常量 ────────────────────────────────────────────────────────────────────

#: 必须整组替换的六个弹性参数
PARAMS: Tuple[str, ...] = ('vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho')

#: 1DREF/STW105 地表椭率 ε。用于把自由表面半径拆成椭率与地形两部分，
#: 见 SurfaceDatum.to_depth 的推导
ELL_SURFACE: float = 1.0 / 299.8

#: R220 的归一化半径。地形拉伸只作用于 R220 以上，见 add_topography.f90:69
R220_NORM: float = 6151.0 / R_EARTH_KM

#: 核幔边界半径（km）。reg1 的底面恰在此处，用作深度映射的标定点
R_CMB_KM: float = 3480.0

#: CMB 自检容差（km）。超过它说明深度映射有系统性问题
CMB_TOLERANCE_KM: float = 3.0

#: 自由表面查表剔除阈值（km）。低于局部上包络这么多的格子判为没有地表点
SURFACE_REJECT_KM: float = 8.0

#: 直接从底图拷贝、不参与插值的参数
COPY_PARAMS: Tuple[str, ...] = ('qmu',)

#: 体积模量为正的充要条件 Vp > (2/sqrt3) * Vs
KAPPA_RATIO: float = 2.0 / np.sqrt(3.0)

#: 值域合理性检查区间 (min, max)，单位 km/s 或 g/cm3
VALUE_RANGE: Dict[str, Tuple[float, float]] = {
    'vpv': (1.0, 15.0),
    'vph': (1.0, 15.0),
    'vsv': (0.3, 8.0),
    'vsh': (0.3, 8.0),
    'eta': (0.5, 1.5),
    'rho': (1.0, 6.0),
}

#: 清单文件名，记录网格指纹与改动文件校验和
MANIFEST_NAME: str = 'manifest.txt'

logger = logging.getLogger('EASTASIA-FWI.BuildHybridGLL')


# ── 校验工具 ────────────────────────────────────────────────────────────────


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """整文件 sha256。"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def mesh_prefix_fingerprint(solver_path: Path) -> Tuple[int, str]:
    """
    对 solver_data.bin 的坐标部分（前 6 条记录 nspec/nglob/x/y/z/ibool）取指纹。

    这段内容完全决定了 GLL 点的物理位置，也就决定了插值取值的正确性。
    本地建模型时记录它，服务器 mesher 跑完后重新核对，可以拦住
    「网格几何变了但 NSPEC 没变」这种不会报错的静默错误。

    逐条读 record marker 再跳过，因此与 CUSTOM_REAL 是 4 还是 8 字节无关。

    Args:
        solver_path: proc***_reg1_solver_data.bin 路径

    Returns:
        (坐标部分字节数, sha256 十六进制串)
    """
    with open(solver_path, 'rb') as f:
        for _ in range(6):
            nbytes = struct.unpack('i', f.read(4))[0]
            f.seek(nbytes + 4, 1)   # 跳过数据体与尾部 marker
        length = f.tell()
        f.seek(0)
        digest = hashlib.sha256(f.read(length)).hexdigest()
    return length, digest


# ── 深度基准 ────────────────────────────────────────────────────────────────


def xyz_to_lat_lon_radius(x: np.ndarray, y: np.ndarray, z: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    非量纲笛卡尔坐标 → (地心纬度°, 经度°, 归一化半径)。

    半径原样返回，不在这里换算深度。深度换算需要自由表面作参照，见 SurfaceDatum。
    """
    r = np.sqrt(x.astype(np.float64) ** 2 + y.astype(np.float64) ** 2
                + z.astype(np.float64) ** 2)
    safe = np.maximum(r, 1e-30)
    lat = np.degrees(np.arcsin(np.clip(z / safe, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon, r


def _fill_nan_nearest_2d(arr: np.ndarray) -> np.ndarray:
    """用最近有效值填 2D 数组里的 NaN。空格子（网格覆盖不到的角落）靠它补齐。"""
    from scipy.ndimage import distance_transform_edt
    bad = ~np.isfinite(arr)
    if not bad.any():
        return arr
    if bad.all():
        raise ValueError('自由表面网格全空，无法填补')
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    out = arr.copy()
    out[bad] = arr[tuple(idx)][bad]
    return out


@dataclass
class SurfaceDatum:
    """
    自由表面半径查表，把 GLL 点深度定义成到局部地表的径向距离。

    为什么需要它：mesher 的调用顺序是 get_model → add_topography → get_ellipticity
    （compute_element_properties.f90:217-297），也就是说 SPECFEM 查模型用的是**完美
    球面坐标**，而写进 solver_data.bin 的坐标是**加完地形和椭率**的。直接用
    R⊕(1-r) 当深度会带上一个随纬度变化的偏差：椭率因子 1-(2/3)·ell·P₂(cosθ) 在
    赤道让半径涨 7 km、在 60°N 让它缩 9 km，研究区内跨度 16 km。这个量在地幔里只
    值 0.3% 的速度误差，但在 10 km 深处的地壳里能到 8.7%，做地壳替换时必须修掉。

    修法：地形和椭率都是纯径向缩放，用同一条径向线上的自由表面半径作参照，两者会
    一起约掉。表面半径直接从网格自身取——这样无论 SPECFEM 对地形做过什么平滑，
    我们都自动跟它一致，不存在复现其处理流程时对不上的风险。

    Attributes:
        lat_min: 表格纬度起点（度）
        lon_min: 表格经度起点（度，[0,360) 约定）
        step: 格距（度）
        r_surf: (nlat, nlon) 归一化自由表面半径
    """

    lat_min: float
    lon_min: float
    step: float
    r_surf: np.ndarray

    def query(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """双线性取自由表面半径，超出表格范围时钳制到边缘。"""
        nlat, nlon = self.r_surf.shape
        fi = np.clip((lat - self.lat_min) / self.step, 0.0, nlat - 1.0)
        fj = np.clip((normalize_lon(lon) - self.lon_min) / self.step, 0.0, nlon - 1.0)
        i0 = np.floor(fi).astype(np.intp)
        j0 = np.floor(fj).astype(np.intp)
        i1 = np.minimum(i0 + 1, nlat - 1)
        j1 = np.minimum(j0 + 1, nlon - 1)
        ti, tj = fi - i0, fj - j0
        g = self.r_surf
        return ((1 - ti) * ((1 - tj) * g[i0, j0] + tj * g[i0, j1])
                + ti * ((1 - tj) * g[i1, j0] + tj * g[i1, j1]))

    def to_depth(self, lat: np.ndarray, lon: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        把变形后的归一化半径换成 SPECFEM 查模型时用的球面深度（km）。

        记球面半径 r_s、归一化地形 e、椭率因子 A = 1-(2/3)·ell·P₂(cosθ)。变形是
            r_def = [r_s + γ(r_s)·e] · A,   γ = (r_s-R220)/(1-R220)，R220 以下 γ=0
        表面处 r_s=1、γ=1，故 r_surf = (1+e)·A。ell 在地表到 R220 之间只变 0.5%，
        取 A 为常数后两式相减：

            R220 以上：(r_surf - r_def)/A = u·(1 + e/u₂₂₀)
            R220 以下：(r_surf - r_def)/A = u + e            （u = 1-r_s 为球面深度）

        两式在 u=u₂₂₀ 处连续。A 由纬度解析给出，e 再从 r_surf/A-1 反解，于是 u 可
        逐点求出。

        残差来自把 ell(r) 当常数：ell 从地表的 1/299.8 降到 CMB 的约 0.0026，因此
        误差随深度增长、在地表为零。实测（tests/test_depth_datum.py，144 分片）
        CMB 处约 1 km、1000 km 深处约 0.8·P₂ km，地壳内可忽略；相比之下直接用
        R⊕(1-r) 在同一套网格上偏差达 11.5 km。

        Args:
            lat: 地心纬度（度）
            lon: 经度（度）
            r: 归一化半径（solver_data.bin 中坐标的模）

        Returns:
            球面深度（km，向下为正）
        """
        cos_theta = np.sin(np.radians(lat))          # θ 为余纬，cosθ = sin(lat)
        p2 = 0.5 * (3.0 * cos_theta ** 2 - 1.0)
        a_ell = 1.0 - (2.0 / 3.0) * ELL_SURFACE * p2

        r_surf = self.query(lat, lon)
        elev = r_surf / a_ell - 1.0                  # 归一化地形（海域为负）
        s = (r_surf - r) / a_ell

        u220 = 1.0 - R220_NORM
        shallow = s <= (u220 + elev)
        # 浅部按地形拉伸反演，深部地形不再参与，直接减掉表面抬升
        u = np.where(shallow,
                     s * u220 / np.maximum(u220 + elev, 1e-9),
                     s - elev)
        return u * R_EARTH_KM


def _surface_bins_one_proc(pid: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    统计单个 proc 落在各格子里的最大半径。

    reg1 是横向分域的，每个 slice 都从地表贯通到 CMB，所以格子里的最大半径就是
    该处的自由表面半径。同一单元内 k 方向的 GLL 点共享 (ξ,η)，经纬度几乎相同，
    因此"有点的格子必有表面点"，不会出现拿深部点冒充地表的情况。

    Returns:
        (非空格子的展平下标, 对应的最大归一化半径)
    """
    mesh_dir: Path = _SG['mesh_dir']
    lat_min: float = _SG['lat_min']
    lon_min: float = _SG['lon_min']
    step: float = _SG['step']
    nlat: int = _SG['nlat']
    nlon: int = _SG['nlon']

    path = mesh_dir / f"proc{pid:06d}_reg1_solver_data.bin"
    _nspec, _nglob, x, y, z, _ibool, _ = read_solver_data(path, extract_velocities=False)
    lat, lon, r = xyz_to_lat_lon_radius(x, y, z)

    ii = np.clip(np.rint((lat - lat_min) / step).astype(np.intp), 0, nlat - 1)
    jj = np.clip(np.rint((normalize_lon(lon) - lon_min) / step).astype(np.intp), 0, nlon - 1)
    flat = ii * nlon + jj

    order = np.argsort(flat, kind='stable')
    fs, rs = flat[order], r[order]
    starts = np.flatnonzero(np.r_[True, fs[1:] != fs[:-1]])
    return fs[starts], np.maximum.reduceat(rs, starts)


#: _surface_bins_one_proc 的工作进程全局状态
_SG: Dict[str, Any] = {}


def _init_surface_worker(mesh_dir: str, lat_min: float, lon_min: float,
                         step: float, nlat: int, nlon: int) -> None:
    _SG.update(mesh_dir=Path(mesh_dir), lat_min=lat_min, lon_min=lon_min,
               step=step, nlat=nlat, nlon=nlon)


def build_surface_datum(mesh_dir: Path, nproc: int, jobs: int,
                        step: float = 0.1) -> SurfaceDatum:
    """
    扫一遍全部 proc 的坐标，建立自由表面半径查表。

    逐 proc 建表会在分片边界上缺数据，所以统一扫全域一次；顺带也只读一遍文件。

    Args:
        mesh_dir: 含 proc*_reg1_solver_data.bin 的目录
        nproc: MPI 进程数
        jobs: 并行度
        step: 格距（度）

    Returns:
        SurfaceDatum
    """
    # 全球格网累加，省去先探边界的一遍 IO；用完再裁到实际覆盖范围
    lat_min, lon_min = -90.0, 0.0
    nlat = int(np.ceil(180.0 / step)) + 1
    nlon = int(np.ceil(360.0 / step)) + 1

    acc = np.zeros(nlat * nlon, dtype=np.float64)
    init = (str(mesh_dir), lat_min, lon_min, step, nlat, nlon)
    with Pool(processes=jobs, initializer=_init_surface_worker, initargs=init) as pool:
        for keys, vals in pool.imap_unordered(_surface_bins_one_proc, range(nproc)):
            np.maximum.at(acc, keys, vals)

    grid = acc.reshape(nlat, nlon)
    rows = np.flatnonzero(grid.any(axis=1))
    cols = np.flatnonzero(grid.any(axis=0))
    if rows.size == 0:
        raise ValueError(f"{mesh_dir} 的坐标全部为空，无法建立自由表面查表")

    # 裁到覆盖范围并留 2° 余量，让边缘点的双线性 stencil 仍有真实数据
    pad = int(np.ceil(2.0 / step))
    i0, i1 = max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad, nlat - 1)
    j0, j1 = max(int(cols[0]) - pad, 0), min(int(cols[-1]) + pad, nlon - 1)
    grid = grid[i0:i1 + 1, j0:j1 + 1].copy()
    lat_min += i0 * step
    lon_min += j0 * step

    grid[grid <= 0.0] = np.nan

    # chunk 侧边界上，同一根径向线的浅部与深部点可能被舍入到相邻格子，于是个别
    # 格子里只剩深部点，其"最大半径"根本不是地表。这类格子必须在填补之前剔除，
    # 否则最近邻填补会把错值扩散到整片空格，后续任何邻域判据都失效。
    #
    # 判据取局部上包络：1° 窗口内真实地形起伏最多约 4 km，而最近的单元层面在
    # 15 km 深处，取 8 km 阈值可以把两者分开。
    from scipy.ndimage import maximum_filter
    win = 2 * int(round(1.0 / step)) + 1
    envelope = maximum_filter(np.nan_to_num(grid, nan=-1.0), size=win, mode='nearest')
    bad = ~np.isfinite(grid) | (grid < envelope - SURFACE_REJECT_KM / R_EARTH_KM)
    n_deep = int((bad & np.isfinite(grid)).sum())
    if n_deep:
        logger.info('自由表面查表: 剔除 %d 个只含深部点的格子', n_deep)
    grid[bad] = np.nan
    grid = _fill_nan_nearest_2d(grid)

    return SurfaceDatum(lat_min=lat_min, lon_min=lon_min, step=step, r_surf=grid)


# ── CRUST1.0 Moho ───────────────────────────────────────────────────────────


@dataclass
class MohoGrid:
    """
    CRUST1.0 的 Moho 深度（km，地表以下），1°×1° 规则网格。

    深度基准与 SPECFEM 一致：model_crust_1_0.f90:181 把 moho 算成沉积层加三层结晶
    地壳的**厚度之和**，不含水层与冰层，也就是"地表以下"而非"海平面以下"。对应到
    crust1.bnds 就是第 3 列（上层沉积顶面）减第 9 列（Moho 面）。

    注意 SPECFEM 查询时会做 CAP 平滑（crust_1_0_CAPsmoothed），这里用双线性，两者
    在 Moho 起伏剧烈处可差几 km。替换上界的 taper 宽度需覆盖这个量级。
    """

    lat: np.ndarray            # 升序，纬度（度）
    lon: np.ndarray            # 升序，经度（度，[0,360)）
    depth: np.ndarray          # (nlat, nlon)，km

    def query(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """双线性取 Moho 深度，超出网格时钳制到边缘。"""
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (self.lat, self.lon), self.depth, method='linear',
            bounds_error=False, fill_value=None,
        )
        lon_n = np.clip(normalize_lon(lon), self.lon[0], self.lon[-1])
        lat_c = np.clip(lat, self.lat[0], self.lat[-1])
        return np.asarray(interp(np.column_stack([lat_c, lon_n])))


def load_crust1_moho(bnds_path: Path) -> MohoGrid:
    """
    读取 crust1.bnds，返回 Moho 深度网格。

    文件格式（DATA/crust1.0/readme.txt）：360×180 = 64800 行，每行 9 个数，为各层
    顶面高程（km，向上为正）。纬度自 89.5°N 递减到 89.5°S，经度自 179.5°W 递增到
    179.5°E，经度为内层循环。

    Args:
        bnds_path: crust1.bnds 路径

    Returns:
        MohoGrid（纬度经度均已转成升序，经度归一化到 [0,360)）
    """
    raw = np.loadtxt(bnds_path)
    if raw.shape != (64800, 9):
        raise ValueError(f"{bnds_path.name} 形状 {raw.shape} 不是预期的 (64800, 9)")

    # 第 3 列是上层沉积顶面（水与冰之下），第 9 列是 Moho
    moho = (raw[:, 2] - raw[:, 8]).reshape(180, 360)

    lat = np.arange(89.5, -90.0, -1.0)
    lon = np.arange(-179.5, 180.0, 1.0)

    moho = moho[::-1, :]                    # 纬度转升序
    lat = lat[::-1]

    lon_n = normalize_lon(lon)
    order = np.argsort(lon_n)               # -179.5..179.5 → 0.5..359.5
    lon_n = lon_n[order]
    moho = moho[:, order]

    # 经度首尾各补一格，让 0°/360° 附近的双线性有完整 stencil
    lon_ext = np.r_[lon_n[-1] - 360.0, lon_n, lon_n[0] + 360.0]
    moho_ext = np.c_[moho[:, -1], moho, moho[:, 0]]

    return MohoGrid(lat=lat, lon=lon_ext, depth=moho_ext)


def find_crust1_bnds(start: Path) -> Optional[Path]:
    """在常见位置查找 crust1.bnds。找不到返回 None，由调用方要求显式指定。"""
    candidates = [
        start / 'DATA' / 'crust1.0' / 'crust1.bnds',
        start / 'specfem3d_globe_code_new' / 'DATA' / 'crust1.0' / 'crust1.bnds',
        start / 'specfem3d_globe' / 'DATA' / 'crust1.0' / 'crust1.bnds',
        start.parent / 'specfem3d_globe' / 'DATA' / 'crust1.0' / 'crust1.bnds',
    ]
    return next((c for c in candidates if c.exists()), None)


def moho_span_in_region(moho: MohoGrid, region: 'ReplaceRegion') -> Tuple[float, float]:
    """统计替换区水平范围内 Moho 深度的最浅与最深值（km）。"""
    lat = np.arange(region.lat_min, region.lat_max + 0.25, 0.25)
    lon = np.arange(region.lon_min, region.lon_max + 0.25, 0.25)
    la, lo = np.meshgrid(lat, lon, indexing='ij')
    d = moho.query(la.ravel(), lo.ravel())
    return float(np.nanmin(d)), float(np.nanmax(d))


# ── 配置 ────────────────────────────────────────────────────────────────────


@dataclass
class ReplaceRegion:
    """
    替换区几何与 taper 参数。

    lat/lon/depth 的边界值是 **w = 0 的外边界**；taper 从边界向内侧渐变到 1。
    这样可以保证过渡带完整落在模型数据域内部，不会被数据域掩码截断成硬边界。

    上界有两种模式。`fixed` 用固定深度 `depth_min`；`moho` 让上界逐点跟随
    CRUST1.0 的 Moho，即 depth_min(lat,lon) = moho(lat,lon) + moho_offset。后者
    配合 `--base-from-mesh` 使用，效果是地壳一律保留 CRUST1.0、Moho 以下才换成
    外部模型，三个候选模型因此共用同一套地壳。
    """

    lat_min: float = 10.0
    lat_max: float = 55.0
    lon_min: float = 80.0
    lon_max: float = 150.0
    horiz_taper: float = 5.0
    depth_min: float = 75.0
    depth_max: float = 1000.0
    depth_taper_top: float = 30.0
    depth_taper_bot: float = 50.0
    depth_min_mode: str = 'fixed'      # 'fixed' | 'moho'
    moho_offset: float = 0.0

    #: moho 模式下 depth_min 的可能范围，由 MohoGrid 在替换区内实测后填入，
    #: 供 validate / summary / 数据域检查使用
    moho_span: Optional[Tuple[float, float]] = None

    def effective_depth_min(self) -> Tuple[float, float]:
        """替换区上界的 (最浅, 最深) 可能取值，单位 km。"""
        if self.depth_min_mode == 'moho' and self.moho_span is not None:
            return (self.moho_span[0] + self.moho_offset,
                    self.moho_span[1] + self.moho_offset)
        return (self.depth_min, self.depth_min)

    def validate(self) -> None:
        """检查几何自洽性，taper 过宽会让 w 永远达不到 1。"""
        if self.lat_max - self.lat_min <= 2 * self.horiz_taper:
            raise ValueError(
                f"纬度跨度 {self.lat_max - self.lat_min:.1f}° 不足以容纳两侧各 "
                f"{self.horiz_taper:.1f}° 的 taper"
            )
        if self.lon_max - self.lon_min <= 2 * self.horiz_taper:
            raise ValueError(
                f"经度跨度 {self.lon_max - self.lon_min:.1f}° 不足以容纳两侧各 "
                f"{self.horiz_taper:.1f}° 的 taper"
            )
        if self.depth_min_mode not in ('fixed', 'moho'):
            raise ValueError(f"未知的上界模式: {self.depth_min_mode}")
        d_top = self.effective_depth_min()[1]      # 最深的上界最吃紧
        if self.depth_max - d_top <= self.depth_taper_top + self.depth_taper_bot:
            raise ValueError(
                f"深度跨度 {self.depth_max - d_top:.1f} km 不足以容纳 "
                f"{self.depth_taper_top:.1f} + {self.depth_taper_bot:.1f} km 的 taper"
            )

    def summary(self) -> str:
        d_lo, d_hi = self.effective_depth_min()
        if self.depth_min_mode == 'moho':
            top = (f"上界跟随 CRUST1.0 Moho{self.moho_offset:+g} km"
                   f"（区内 {d_lo:.1f} ~ {d_hi:.1f} km）")
            full = f"[{d_hi + self.depth_taper_top:.1f}, {self.depth_max - self.depth_taper_bot}] km"
        else:
            top = f"上界固定 {self.depth_min} km"
            full = f"[{d_lo + self.depth_taper_top}, {self.depth_max - self.depth_taper_bot}] km"
        return (
            f"lat [{self.lat_min}, {self.lat_max}] "
            f"lon [{self.lon_min}, {self.lon_max}] "
            f"(水平 taper {self.horiz_taper}°, 满权重区 "
            f"lat [{self.lat_min + self.horiz_taper}, {self.lat_max - self.horiz_taper}] "
            f"lon [{self.lon_min + self.horiz_taper}, {self.lon_max - self.horiz_taper}]); "
            f"{top}, 下界 {self.depth_max} km "
            f"(taper 上 {self.depth_taper_top} / 下 {self.depth_taper_bot} km, "
            f"满权重区 {full})"
        )


@dataclass
class ModelGrid:
    """预处理后的模型规则网格，坐标已升序、NaN 已填补、单位已统一。"""

    axes: Tuple[np.ndarray, np.ndarray, np.ndarray]  # 按 dim_order 排列
    dim_order: Tuple[str, str, str]                   # ('dep'|'lat'|'lon') 的排列
    arrays: Dict[str, np.ndarray]
    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]
    depth_range: Tuple[float, float]


# ── 权重场 ──────────────────────────────────────────────────────────────────


def _ramp_up(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """升余弦：x<=a 时 0，x>=b 时 1，中间平滑过渡。b 必须大于 a。"""
    if b <= a:
        return (x >= b).astype(np.float64)
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * t))


def normalize_lon(lon: np.ndarray | float) -> np.ndarray:
    """经度统一到 [0, 360)，便于与 NC 坐标比较。"""
    return np.mod(np.asarray(lon, dtype=np.float64), 360.0)


def compute_weight(
    lat: np.ndarray,
    lon: np.ndarray,
    depth: np.ndarray,
    region: ReplaceRegion,
    grid: Optional[ModelGrid | SalvusPointCloud] = None,
    moho: Optional[MohoGrid] = None,
) -> np.ndarray:
    """
    计算逐点替换权重 w ∈ [0, 1]。

    Args:
        lat: 纬度（度）
        lon: 经度（度，任意约定，内部归一化到 [0,360)）
        depth: 深度（km，向下为正）
        region: 替换区几何
        grid: 模型网格；若给出，落在其数据域外的点强制 w = 0
        moho: CRUST1.0 Moho 网格；region.depth_min_mode == 'moho' 时必需

    Returns:
        与输入同形状的权重数组（float64）
    """
    lon_n = normalize_lon(lon)

    # 水平：四条边各自向内 ramp，取最小值
    w_lat = np.minimum(
        _ramp_up(lat, region.lat_min, region.lat_min + region.horiz_taper),
        _ramp_up(-lat, -region.lat_max, -region.lat_max + region.horiz_taper),
    )
    w_lon = np.minimum(
        _ramp_up(lon_n, region.lon_min, region.lon_min + region.horiz_taper),
        _ramp_up(-lon_n, -region.lon_max, -region.lon_max + region.horiz_taper),
    )
    w = w_lat * w_lon

    # 垂直上界：固定深度，或逐点跟随 CRUST1.0 Moho
    if region.depth_min_mode == 'moho':
        if moho is None:
            raise ValueError("depth_min_mode='moho' 需要提供 MohoGrid")
        d_top = moho.query(lat, lon) + region.moho_offset
    else:
        d_top = np.full(np.shape(depth), region.depth_min, dtype=np.float64)

    w_dep = np.minimum(
        _ramp_up(depth - d_top, 0.0, region.depth_taper_top),
        _ramp_up(-depth, -region.depth_max, -region.depth_max + region.depth_taper_bot),
    )
    w *= w_dep

    # 模型数据域外强制归零
    if grid is not None:
        lo0, lo1 = normalize_lon(grid.lon_range[0]), normalize_lon(grid.lon_range[1])
        inside = (
            (lat >= grid.lat_range[0]) & (lat <= grid.lat_range[1])
            & (lon_n >= lo0) & (lon_n <= lo1)
            & (depth >= grid.depth_range[0]) & (depth <= grid.depth_range[1])
        )
        w = np.where(inside, w, 0.0)

    return w


def check_region_inside_model(
    region: ReplaceRegion,
    grid: ModelGrid | SalvusPointCloud,
) -> List[str]:
    """
    检查替换区是否完整落在模型数据域内。

    若不满足，数据域掩码会把 taper 截断成硬边界，在计算域内制造人工散射体。
    返回警告信息列表（空列表表示通过）。
    """
    warnings: List[str] = []
    lo0, lo1 = normalize_lon(grid.lon_range[0]), normalize_lon(grid.lon_range[1])
    d_shallow = region.effective_depth_min()[0]
    checks = [
        ('纬度下界', region.lat_min, grid.lat_range[0], '>='),
        ('纬度上界', region.lat_max, grid.lat_range[1], '<='),
        ('经度下界', normalize_lon(region.lon_min), lo0, '>='),
        ('经度上界', normalize_lon(region.lon_max), lo1, '<='),
        ('深度上界（最浅处）', d_shallow, grid.depth_range[0], '>='),
        ('深度下界', region.depth_max, grid.depth_range[1], '<='),
    ]
    for name, want, have, op in checks:
        ok = want >= have if op == '>=' else want <= have
        if not ok:
            warnings.append(
                f"{name} {want:g} 超出模型数据域 {have:g}，taper 将被截断成硬边界"
            )
    return warnings


# ── 模型加载 ────────────────────────────────────────────────────────────────


def load_model_grid(
    model_path: Path,
    kind: str,
    region: Optional['ReplaceRegion'] = None,
    salvus_subsample: int = 1,
) -> ModelGrid | SalvusPointCloud:
    """
    读取模型文件并预处理成可直接插值的规则网格或 Salvus 点云。

    处理内容：识别维度顺序、翻转降序坐标、rho 单位统一到 g/cm³、
    补齐缺失参数、填补 NaN、降为 float32 以控制内存。

    Args:
        model_path: 模型文件路径
        kind: 'nc' | 'h5' | 'salvus'
        region: Salvus 模式下的区域裁剪（替换区 + 余量）
        salvus_subsample: Salvus 单元 stride（>1 仅用于快速测试）

    Returns:
        ModelGrid 或 SalvusPointCloud
    """
    import xarray as xr

    if kind == 'salvus':
        pad = 5.0
        if region is not None:
            lat_b = (region.lat_min - pad, region.lat_max + pad)
            lon_b = (region.lon_min - pad, region.lon_max + pad)
            d_top = region.effective_depth_min()[0]
            dep_b = (d_top - 20.0, region.depth_max + 50.0)
        else:
            lat_b = (-15.0, 65.0)
            lon_b = (45.0, 170.0)
            dep_b = (-20.0, 1100.0)
        return load_salvus_point_cloud(
            model_path,
            lat_bounds=lat_b,
            lon_bounds=lon_b,
            depth_bounds_km=dep_b,
            subsample=salvus_subsample,
        )

    if kind == 'h5':
        ds = load_h5_as_dataset(model_path)
    elif kind == 'nc':
        ds = xr.open_dataset(model_path)
    else:
        raise ValueError(f"不支持的模型格式: {kind}")

    sample = next((p for p in PARAMS if p in ds), None)
    if sample is None:
        raise ValueError(f"{model_path.name} 中找不到任何弹性参数 {PARAMS}")

    # 识别维度顺序
    dim_order: List[str] = []
    for d in ds[sample].dims:
        dl = str(d).lower()
        if 'lon' in dl:
            dim_order.append('lon')
        elif 'lat' in dl:
            dim_order.append('lat')
        elif 'dep' in dl:
            dim_order.append('dep')
        else:
            raise ValueError(f"无法识别的维度: {d}")
    if len(dim_order) != 3:
        raise ValueError(f"{model_path.name} 维度数不为 3: {ds[sample].dims}")

    coord_name = {'lon': 'longitude', 'lat': 'latitude', 'dep': 'depth'}
    raw_axes = {k: np.asarray(ds[coord_name[k]].values, dtype=np.float64) for k in dim_order}
    raw_axes['lon'] = normalize_lon(raw_axes['lon'])

    # 降序坐标需翻转，RegularGridInterpolator 要求严格升序
    flip_axis: Dict[str, bool] = {}
    for k, arr in raw_axes.items():
        flip_axis[k] = bool(arr.size > 1 and arr[-1] < arr[0])
        if flip_axis[k]:
            raw_axes[k] = arr[::-1]

    arrays: Dict[str, np.ndarray] = {}
    for p in PARAMS:
        if p in ds:
            arr = np.asarray(ds[p].values, dtype=np.float64)
        elif p == 'eta':
            # 无 eta 的模型按各向同性处理
            arr = np.ones_like(np.asarray(ds[sample].values, dtype=np.float64))
        else:
            raise ValueError(
                f"{model_path.name} 缺少参数 {p}，无法整组替换。"
                f"请先补齐或改用其它模型（见 Models_Hybrid_workflow.md §4.1）"
            )

        # rho 单位：kg/m³ → g/cm³
        if p == 'rho' and np.nanmean(arr) > 100:
            arr = arr / 1000.0

        for i, k in enumerate(dim_order):
            if flip_axis[k]:
                arr = np.flip(arr, axis=i)

        arrays[p] = _fill_nan_nearest_3d(arr).astype(np.float32)

    ds.close()

    return ModelGrid(
        axes=tuple(raw_axes[k] for k in dim_order),  # type: ignore[arg-type]
        dim_order=tuple(dim_order),                  # type: ignore[arg-type]
        arrays=arrays,
        lat_range=(float(raw_axes['lat'].min()), float(raw_axes['lat'].max())),
        lon_range=(float(raw_axes['lon'].min()), float(raw_axes['lon'].max())),
        depth_range=(float(raw_axes['dep'].min()), float(raw_axes['dep'].max())),
    )


def interpolate_model(
    grid: ModelGrid | SalvusPointCloud,
    lat: np.ndarray,
    lon: np.ndarray,
    depth: np.ndarray,
) -> Dict[str, np.ndarray]:
    """在给定的散点上插值全部六个参数（规则网格或 Salvus 点云）。"""
    if isinstance(grid, SalvusPointCloud):
        return grid.interpolate(lat, lon, depth)

    from scipy.interpolate import RegularGridInterpolator

    query = {'lat': lat, 'lon': normalize_lon(lon), 'dep': depth}
    pts = np.column_stack([query[k] for k in grid.dim_order])

    out: Dict[str, np.ndarray] = {}
    for p in PARAMS:
        interp = RegularGridInterpolator(
            grid.axes, grid.arrays[p], method='linear',
            bounds_error=False, fill_value=np.nan,
        )
        out[p] = interp(pts)
    return out


# ── 逐 proc 处理 ────────────────────────────────────────────────────────────

#: 工作进程全局状态，由 _init_worker 填充（避免每个 proc 重复加载模型）
_G: Dict[str, Any] = {}


def _init_worker(mesh_dir: str, base_dir: str, out_dir: str,
                 model_path: Optional[str], model_kind: Optional[str],
                 region: ReplaceRegion, dry_run: bool, skip_unchanged: bool,
                 base_from_mesh: bool, datum: SurfaceDatum,
                 moho: Optional[MohoGrid], salvus_subsample: int) -> None:
    """工作进程初始化：加载一次模型网格，后续所有 proc 复用。"""
    _G['mesh_dir'] = Path(mesh_dir)
    _G['base_dir'] = Path(base_dir)
    _G['out_dir'] = Path(out_dir)
    _G['region'] = region
    _G['dry_run'] = dry_run
    _G['skip_unchanged'] = skip_unchanged
    _G['base_from_mesh'] = base_from_mesh
    _G['datum'] = datum
    _G['moho'] = moho
    _G['grid'] = (
        load_model_grid(
            Path(model_path), model_kind,  # type: ignore[arg-type]
            region=region, salvus_subsample=salvus_subsample,
        )
        if model_path else None
    )


def process_one_proc(pid: int) -> Dict[str, Any]:
    """
    处理单个 MPI 进程分片。

    Args:
        pid: proc 编号

    Returns:
        统计与自检结果字典，含 'ok'、'pid'、'errors'、点数统计
    """
    mesh_dir: Path = _G['mesh_dir']
    base_dir: Path = _G['base_dir']
    out_dir: Path = _G['out_dir']
    region: ReplaceRegion = _G['region']
    grid: Optional[ModelGrid] = _G['grid']
    dry_run: bool = _G['dry_run']
    base_from_mesh: bool = _G['base_from_mesh']
    datum: SurfaceDatum = _G['datum']
    moho: Optional[MohoGrid] = _G['moho']

    tag = f"proc{pid:06d}_reg1"
    errors: List[str] = []

    # ── 1. 坐标与底图来源 ──
    # 坐标记录无条件写入，不受 Par_file 开关影响；弹性模量记录只在需要时才读
    solver_path = mesh_dir / f"{tag}_solver_data.bin"
    nspec, nglob, x, y, z, ibool, ref_vel = read_solver_data(
        solver_path, extract_velocities=base_from_mesh)
    mesh_len, mesh_hash = mesh_prefix_fingerprint(solver_path)

    # ibool 把 GLL 点映射到全局点；在唯一点上算完再展开，省约 46% 计算
    idx = ibool.ravel(order='F') - 1
    lat_g, lon_g, r_g = xyz_to_lat_lon_radius(x, y, z)
    dep_g = datum.to_depth(lat_g, lon_g, r_g)
    shape = (NGLLX, NGLLY, NGLLZ, nspec)

    # ── 2. 底图 ──
    base: Dict[str, np.ndarray] = {}
    if base_from_mesh:
        if ref_vel is None:
            raise RuntimeError(f"{tag}: solver_data.bin 中未能反算出速度场")
        for p in PARAMS:
            base[p] = ref_vel[p].astype(np.float32)
    else:
        for p in PARAMS:
            base[p] = read_gll_binary(base_dir / f"{tag}_{p}.bin", nspec)

    n_total = int(np.prod(shape))

    # ── 3. 权重 ──
    if grid is None:
        w_gll = np.zeros(shape, dtype=np.float64)
    else:
        w_g = compute_weight(lat_g, lon_g, dep_g, region, grid, moho)
        w_gll = w_g[idx].reshape(shape, order='F')

    mask = w_gll > 0.0
    n_partial = int(mask.sum())
    n_full = int((w_gll >= 1.0 - 1e-12).sum())

    # ── 4. 插值与混合 ──
    out: Dict[str, np.ndarray] = {p: base[p].copy() for p in PARAMS}

    if grid is not None and n_partial > 0:
        # 只在需要的全局点上插值
        need_g = np.zeros(lat_g.shape, dtype=bool)
        need_g[idx[mask.ravel(order='F')]] = True
        sel = np.flatnonzero(need_g)
        interp_g = interpolate_model(grid, lat_g[sel], lon_g[sel], dep_g[sel])

        for p in PARAMS:
            full_g = np.full(lat_g.shape, np.nan, dtype=np.float64)
            full_g[sel] = interp_g[p]
            model_gll = full_g[idx].reshape(shape, order='F')

            # 插值出 NaN 的点退回底图，避免污染
            bad = mask & ~np.isfinite(model_gll)
            if bad.any():
                model_gll = np.where(bad, base[p], model_gll)

            wm = w_gll[mask]
            out[p][mask] = (
                wm * model_gll[mask] + (1.0 - wm) * base[p][mask]
            ).astype(out[p].dtype)

    # ── 5. 自检 ──
    # 底图自身在近地表含有非常规值（海水层、FWI 更新的浅部节点），
    # 因此值域判据取「物理区间 ∪ 底图包络」，且只作用于被修改的点。
    base_env: Dict[str, Tuple[float, float]] = {
        p: (float(base[p].min()), float(base[p].max())) for p in PARAMS
    }

    for p in PARAMS:
        if not np.isfinite(out[p]).all():
            errors.append(f"S1 {p}: 存在 NaN/Inf")

        vals = out[p][mask]
        if vals.size == 0:
            continue
        lo = min(VALUE_RANGE[p][0], base_env[p][0])
        hi = max(VALUE_RANGE[p][1], base_env[p][1])
        if vals.min() < lo or vals.max() > hi:
            errors.append(
                f"S3 {p}: 修改点值域 [{vals.min():.4f}, {vals.max():.4f}] "
                f"超出容许区间 [{lo:.4f}, {hi:.4f}]"
            )

    for vp, vs in (('vpv', 'vsv'), ('vph', 'vsh')):
        bad = out[vp] <= KAPPA_RATIO * out[vs]
        if bad.any():
            errors.append(f"S2 {vp}/{vs}: {int(bad.sum())} 点体积模量非正")

    if grid is not None:
        keep = ~mask
        for p in PARAMS:
            if not np.array_equal(out[p][keep], base[p][keep]):
                errors.append(f"S4 {p}: w=0 区域与底图不一致")

    if errors:
        return {'ok': False, 'pid': pid, 'errors': errors,
                'n_total': n_total, 'n_partial': n_partial, 'n_full': n_full}

    # ── 6. 写出 ──
    # files: [(文件名, 'hybrid'|'base', sha256 或 '')]
    # 'base' 表示内容与底图逐位相同，--skip-unchanged 时不落盘，
    # 由服务器端从已有的 model_updated/ 补齐
    files: List[Tuple[str, str, str]] = []
    skip_unchanged: bool = _G['skip_unchanged']

    for p in PARAMS:
        fname = f"{tag}_{p}.bin"
        same_as_base = np.array_equal(out[p], base[p])
        if dry_run:
            files.append((fname, 'base' if same_as_base else 'hybrid', ''))
            continue
        if same_as_base and skip_unchanged:
            files.append((fname, 'base', ''))
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        fpath = out_dir / fname
        write_gll_binary(fpath, out[p], dtype=np.float32)
        # 回读计算校验和，顺带确认写入完整
        files.append((fname, 'base' if same_as_base else 'hybrid', sha256_file(fpath)))

    # qmu 恒等于底图，永远不必上传
    if not dry_run and not skip_unchanged:
        for p in COPY_PARAMS:
            src = base_dir / f"{tag}_{p}.bin"
            if src.exists():
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out_dir / f"{tag}_{p}.bin")

    # 相对底图的扰动幅度，用于确认嵌入确实生效
    dln: Dict[str, Tuple[float, float]] = {}
    for p in ('vsv', 'vpv', 'rho'):
        if n_partial > 0:
            r = (out[p][mask] - base[p][mask]) / np.maximum(np.abs(base[p][mask]), 1e-6)
            dln[p] = (float(np.abs(r).max()), float(np.sqrt(np.mean(r ** 2))))
        else:
            dln[p] = (0.0, 0.0)

    return {'ok': True, 'pid': pid, 'errors': [],
            'n_total': n_total, 'n_partial': n_partial, 'n_full': n_full,
            'dln': dln, 'files': files,
            'mesh': (nspec, nglob, mesh_len, mesh_hash),
            'lat': (float(lat_g.min()), float(lat_g.max())),
            'lon': (float(normalize_lon(lon_g).min()), float(normalize_lon(lon_g).max())),
            'dep': (float(dep_g.min()), float(dep_g.max()))}


def write_manifest(path: Path, results: List[Dict[str, Any]], header: Dict[str, str]) -> Tuple[int, int]:
    """
    写出清单，供上传后在服务器端核对。

    格式（'#' 开头为注释）：
        MESH  <pid> <nspec> <nglob> <坐标字节数> <sha256>
        FILE  <文件名> <hybrid|base> <sha256 或 -->

    `base` 行表示该文件与底图逐位相同，服务器端从 model_updated/ 补齐即可。

    Args:
        path: 清单输出路径
        results: 各 proc 的处理结果
        header: 写入注释区的构建参数

    Returns:
        (hybrid 文件数, base 文件数)
    """
    ordered = sorted(results, key=lambda r: r['pid'])
    n_hybrid = n_base = 0
    lines: List[str] = ['# EASTASIA-FWI build_hybrid_gll.py 清单']
    for k, v in header.items():
        lines.append(f'# {k}: {v}')
    lines.append('#')
    lines.append('# MESH <pid> <nspec> <nglob> <coord_bytes> <sha256>')
    lines.append('# FILE <name> <hybrid|base> <sha256>')

    for r in ordered:
        nspec, nglob, mlen, mhash = r['mesh']
        lines.append(f"MESH {r['pid']:06d} {nspec} {nglob} {mlen} {mhash}")
    for r in ordered:
        for fname, kind, digest in r['files']:
            lines.append(f"FILE {fname} {kind} {digest or '--'}")
            if kind == 'hybrid':
                n_hybrid += 1
            else:
                n_base += 1

    path.write_text('\n'.join(lines) + '\n')
    return n_hybrid, n_base


# ── 主流程 ──────────────────────────────────────────────────────────────────


def setup_logger(level: str = 'INFO') -> logging.Logger:
    """统一日志初始化，输出到 stdout 便于 Slurm 收集。"""
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                               datefmt='%H:%M:%S'))
        logger.addHandler(handler)
    return logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='以底图 GLL 为基础，嵌入外部速度模型，生成混合 GLL 模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='详见 Models_Hybrid_workflow.md',
    )
    req = parser.add_argument_group('必选')
    req.add_argument('--mesh-dir', required=True,
                     help='含 proc*_reg1_solver_data.bin 的目录（只读坐标）')
    req.add_argument('--base-gll-dir', required=True,
                     help='底图 GLL 目录，通常为 model_updated/。'
                          '--base-from-mesh 时仍需要它提供 qmu')
    req.add_argument('--output-dir', required=True, help='输出 GLL 目录')

    base = parser.add_argument_group('底图来源')
    base.add_argument('--base-from-mesh', action='store_true',
                      help='弹性参数底图改为从 --mesh-dir 的 solver_data.bin 反算，'
                           '即 mesher 当时用的 CRUST1.0 + s362ani。配合 '
                           '--depth-min-mode moho 可让各候选模型共用同一套地壳。'
                           'qmu 仍取自 --base-gll-dir')

    mdl = parser.add_argument_group('模型（全部省略 = 基准 run，仅拷贝底图）')
    grp = mdl.add_mutually_exclusive_group()
    grp.add_argument('--model-nc', help='NetCDF 模型文件')
    grp.add_argument('--model-h5', help='HDF5 规则网格 (如 FWI_SinoScope_1.0_JMa_ET_AL_2022.h5)')
    grp.add_argument('--model-salvus', help='Salvus 谱元 mesh.h5（SinoScope 作者原生网格）')
    mdl.add_argument('--salvus-subsample', type=int, default=1,
                     help='Salvus 单元 stride，>1 仅用于快速测试（默认 1=全分辨率）')

    reg = parser.add_argument_group('替换区（边界处 w=0，taper 向内）')
    reg.add_argument('--region', nargs=4, type=float,
                     metavar=('LAT0', 'LAT1', 'LON0', 'LON1'),
                     default=[10.0, 55.0, 80.0, 150.0], help='水平外边界，默认 10 55 80 150')
    reg.add_argument('--horiz-taper', type=float, default=5.0,
                     help='水平 taper 宽度（度），默认 5.0')
    reg.add_argument('--depth-range', nargs=2, type=float, metavar=('D0', 'D1'),
                     default=[75.0, 1000.0], help='深度外边界（km），默认 75 1000')
    reg.add_argument('--depth-taper', nargs=2, type=float, metavar=('WTOP', 'WBOT'),
                     default=[30.0, 50.0], help='深度 taper 上/下宽度（km），默认 30 50')
    reg.add_argument('--depth-min-mode', choices=['fixed', 'moho'], default='fixed',
                     help="上界模式：fixed 用 --depth-range 的上界；"
                          "moho 逐点跟随 CRUST1.0 Moho，此时 --depth-range 的上界被忽略")
    reg.add_argument('--moho-offset', type=float, default=0.0,
                     help='moho 模式下相对 Moho 的偏移（km，正为更深），默认 0')
    reg.add_argument('--crust1-bnds', default=None,
                     help='crust1.bnds 路径，默认在 SPECFEM 源码树的 '
                          'DATA/crust1.0/ 下自动查找')

    misc = parser.add_argument_group('其它')
    misc.add_argument('--nproc', type=int, default=144, help='MPI 进程数，默认 144')
    misc.add_argument('--surface-bin', type=float, default=0.1,
                      help='自由表面查表的格距（度），默认 0.1')
    misc.add_argument('--jobs', type=int, default=0, help='并行进程数，默认 CPU 核数')
    misc.add_argument('--skip-unchanged', action='store_true',
                      help='与底图逐位相同的文件不落盘（含 qmu），由服务器端从 '
                           'model_updated/ 补齐。可省约一半磁盘与上传量')
    misc.add_argument('--dry-run', action='store_true', help='只统计不写文件')
    misc.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING'])

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    log = setup_logger(args.log_level)

    mesh_dir = Path(args.mesh_dir)
    base_dir = Path(args.base_gll_dir)
    out_dir = Path(args.output_dir)

    model_path: Optional[Path] = None
    model_kind: Optional[str] = None
    if args.model_nc:
        model_path, model_kind = Path(args.model_nc), 'nc'
    elif args.model_h5:
        model_path, model_kind = Path(args.model_h5), 'h5'
    elif args.model_salvus:
        model_path, model_kind = Path(args.model_salvus), 'salvus'

    region = ReplaceRegion(
        lat_min=args.region[0], lat_max=args.region[1],
        lon_min=args.region[2], lon_max=args.region[3],
        horiz_taper=args.horiz_taper,
        depth_min=args.depth_range[0], depth_max=args.depth_range[1],
        depth_taper_top=args.depth_taper[0], depth_taper_bot=args.depth_taper[1],
        depth_min_mode=args.depth_min_mode, moho_offset=args.moho_offset,
    )

    # 底图来自网格时它并不存在于磁盘上，每个文件都必须落盘
    skip_unchanged = args.skip_unchanged
    if args.base_from_mesh and skip_unchanged:
        log.warning('⚠️  --base-from-mesh 的底图不在磁盘上，已忽略 --skip-unchanged')
        skip_unchanged = False

    log.info('=' * 68)
    log.info('EASTASIA-FWI 混合模型 GLL 构建')
    log.info('=' * 68)
    log.info('坐标源:   %s', mesh_dir)
    log.info('底图:     %s',
             f'{mesh_dir} 反算的 CRUST1.0+s362ani（qmu 取自 {base_dir}）'
             if args.base_from_mesh else str(base_dir))
    log.info('输出:     %s', out_dir)
    log.info('模型:     %s', f'{model_path} ({model_kind})' if model_path else '无（基准 run）')
    log.info('进程数:   %d', args.nproc)

    # ── 前置检查 ──
    try:
        for d, name in ((mesh_dir, '坐标源'), (base_dir, '底图')):
            if not d.is_dir():
                raise FileNotFoundError(f"{name}目录不存在: {d}")
        missing = [
            pid for pid in range(args.nproc)
            if not (mesh_dir / f"proc{pid:06d}_reg1_solver_data.bin").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"坐标源缺少 {len(missing)} 个 solver_data.bin，首个: proc{missing[0]:06d}"
            )
        if model_path and not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    except (FileNotFoundError, ValueError) as exc:
        log.error('前置检查失败: %s', exc)
        return 1

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, args.nproc))
    if model_kind == 'salvus' and jobs > 1:
        log.warning('⚠️  Salvus 点云较大，每个 worker 会独立建 KDTree；'
                    '建议 --jobs 1~4（当前 %d）', jobs)
    log.info('并行度: %d', jobs)

    # ── 自由表面查表：深度基准的前提 ──
    log.info('扫描 %d 个分片建立自由表面查表（格距 %.2f°）…', args.nproc, args.surface_bin)
    try:
        datum = build_surface_datum(mesh_dir, args.nproc, jobs, args.surface_bin)
    except (ValueError, OSError) as exc:
        log.error('自由表面查表构建失败: %s', exc)
        return 1
    log.info('自由表面半径: %.6f ~ %.6f（等效地形 %.2f ~ %.2f km）',
             datum.r_surf.min(), datum.r_surf.max(),
             (datum.r_surf.min() - 1.0) * R_EARTH_KM,
             (datum.r_surf.max() - 1.0) * R_EARTH_KM)

    # ── Moho 网格 ──
    moho: Optional[MohoGrid] = None
    if region.depth_min_mode == 'moho':
        bnds = Path(args.crust1_bnds) if args.crust1_bnds else find_crust1_bnds(script_dir)
        if bnds is None or not bnds.exists():
            log.error('找不到 crust1.bnds，请用 --crust1-bnds 指定')
            return 1
        log.info('CRUST1.0 Moho: %s', bnds)
        try:
            moho = load_crust1_moho(bnds)
        except (ValueError, OSError) as exc:
            log.error('crust1.bnds 读取失败: %s', exc)
            return 1
        region.moho_span = moho_span_in_region(moho, region)
        log.info('替换区内 Moho 深度: %.1f ~ %.1f km', *region.moho_span)

    if model_path:
        log.info('替换区: %s', region.summary())
        try:
            region.validate()
        except ValueError as exc:
            log.error('替换区几何检查失败: %s', exc)
            return 1

    # ── 模型域与替换区的相容性 ──
    if model_path:
        try:
            probe = load_model_grid(
                model_path, model_kind,  # type: ignore[arg-type]
                region=region, salvus_subsample=args.salvus_subsample,
            )
        except (ValueError, OSError) as exc:
            log.error('模型加载失败: %s', exc)
            return 1
        kind_label = 'Salvus 点云' if isinstance(probe, SalvusPointCloud) else '规则网格'
        log.info('模型数据域 (%s): lat %s  lon %s  depth %s km',
                 kind_label, probe.lat_range, probe.lon_range, probe.depth_range)
        if isinstance(probe, SalvusPointCloud):
            log.info('  Salvus GLL 点数: %s', f'{len(probe.xyz):,}')
        for msg in check_region_inside_model(region, probe):
            log.warning('⚠️  %s', msg)
        del probe

    if args.dry_run:
        log.info('--dry-run: 只统计，不写文件')
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── 并行处理 ──
    init_args = (str(mesh_dir), str(base_dir), str(out_dir),
                 str(model_path) if model_path else None, model_kind,
                 region, args.dry_run, skip_unchanged,
                 args.base_from_mesh, datum, moho, args.salvus_subsample)

    results: List[Dict[str, Any]] = []
    with Pool(processes=jobs, initializer=_init_worker, initargs=init_args) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one_proc, range(args.nproc)), 1):
            results.append(res)
            if not res['ok']:
                log.error('proc%06d 自检失败:', res['pid'])
                for e in res['errors']:
                    log.error('    %s', e)
            if i % 24 == 0 or i == args.nproc:
                log.info('进度 %d/%d', i, args.nproc)

    # ── 汇总 ──
    failed = [r for r in results if not r['ok']]
    n_total = sum(r['n_total'] for r in results)
    n_partial = sum(r['n_partial'] for r in results)
    n_full = sum(r['n_full'] for r in results)

    log.info('-' * 68)
    log.info('GLL 总点数      : %s', f'{n_total:,}')
    if model_path:
        log.info('S5 w>0  (受影响): %s (%.2f%%)', f'{n_partial:,}', n_partial / n_total * 100)
        log.info('S5 w==1 (全替换): %s (%.2f%%)', f'{n_full:,}', n_full / n_total * 100)
        if n_partial == 0:
            log.error('替换区与 chunk 无交集，请检查 --region / --depth-range')
            return 1

    ok_res = [r for r in results if r['ok']]
    if model_path and ok_res:
        for p in ('vsv', 'vpv', 'rho'):
            peak = max(r['dln'][p][0] for r in ok_res)
            # 各 proc 的 RMS 按其修改点数加权合并
            num = sum(r['dln'][p][1] ** 2 * r['n_partial'] for r in ok_res)
            den = max(sum(r['n_partial'] for r in ok_res), 1)
            log.info('相对底图扰动 %-3s : peak %6.2f%%  rms %6.2f%%',
                     p, peak * 100, np.sqrt(num / den) * 100)

    if ok_res:
        d_lo = min(r['dep'][0] for r in ok_res)
        d_hi = max(r['dep'][1] for r in ok_res)
        log.info('chunk 实际 lat  : %.2f ~ %.2f',
                 min(r['lat'][0] for r in ok_res), max(r['lat'][1] for r in ok_res))
        log.info('chunk 实际 lon  : %.2f ~ %.2f',
                 min(r['lon'][0] for r in ok_res), max(r['lon'][1] for r in ok_res))
        log.info('chunk 实际 depth: %.1f ~ %.1f km', d_lo, d_hi)

        # 深度基准自检：reg1 的顶面是自由表面、底面恰在 CMB，两端都是已知常数。
        # 偏差超限说明自由表面查表或地形/椭率反演出了系统性问题
        cmb_err = abs(d_hi - (R_EARTH_KM - R_CMB_KM))
        log.info('深度基准自检    : 顶 %+.2f km / 底 %+.2f km（CMB 应为 %.0f km）',
                 d_lo, d_hi - (R_EARTH_KM - R_CMB_KM), R_EARTH_KM - R_CMB_KM)
        if cmb_err > CMB_TOLERANCE_KM or abs(d_lo) > CMB_TOLERANCE_KM:
            log.error('深度基准偏差超过 %.1f km，自由表面查表或深度反演有问题，'
                      '拒绝写出结果', CMB_TOLERANCE_KM)
            return 1

    if failed:
        log.error('%d/%d 个 proc 自检失败，未写出完整结果', len(failed), args.nproc)
        return 1

    if args.dry_run:
        log.info('✅ 诊断完成（未写文件）')
        log.info('=' * 68)
        return 0

    # ── 清单 ──
    header = {
        '生成时间': __import__('datetime').datetime.now().isoformat(timespec='seconds'),
        '坐标源': str(mesh_dir.resolve()),
        '底图': (f'{mesh_dir.resolve()} 反算 (CRUST1.0+s362ani)'
                 if args.base_from_mesh else str(base_dir.resolve())),
        '模型': str(model_path) if model_path else '(无，等价于底图)',
        'nproc': str(args.nproc),
        'region': ' '.join(str(v) for v in args.region),
        'horiz_taper': str(args.horiz_taper),
        'depth_range': ' '.join(str(v) for v in args.depth_range),
        'depth_taper': ' '.join(str(v) for v in args.depth_taper),
        'depth_min_mode': args.depth_min_mode,
        'moho_offset': str(args.moho_offset),
        'surface_bin': str(args.surface_bin),
        'skip_unchanged': str(skip_unchanged),
    }
    n_hybrid, n_base_files = write_manifest(out_dir / MANIFEST_NAME, ok_res, header)

    n_files = len(list(out_dir.glob('proc*_reg1_*.bin')))
    size_mb = sum(f.stat().st_size for f in out_dir.glob('proc*_reg1_*.bin')) / 1e6
    log.info('落盘文件      : %d 个, %.1f GB', n_files, size_mb / 1000)
    log.info('清单          : hybrid %d 个 / base %d 个 -> %s',
             n_hybrid, n_base_files, MANIFEST_NAME)
    if args.skip_unchanged:
        log.info('base 文件未落盘，服务器端由 run_all_test.sh prepare 从底图补齐')
    log.info('✅ 完成 -> %s', out_dir)
    log.info('=' * 68)
    return 0


if __name__ == '__main__':
    sys.exit(main())
