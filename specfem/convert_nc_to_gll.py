"""
EASTASIA-FWI 速度模型 → GLL 格式转换
=====================================

将公开速度模型 (NetCDF / HDF5 / CSV) 插值到 SPECFEM3D Globe 的 GLL 网格，
生成 proc***_reg1_{vpv,vph,vsv,vsh,eta,rho}.bin，供 MODEL=GLL 正演使用。

支持的模型格式:
  - NetCDF (.nc):  EMC 标准格式 (FWEA23, EARA2024 等)
  - HDF5 (.h5):    SinoScope 等自定义格式
  - CSV (.csv):    通用表格格式

前置条件:
  必须先运行 meshfem (MODEL=s362ani) 生成 DATABASES_MPI/proc*_reg1_solver_data.bin，
  提供 GLL 网格坐标。run_forward.bash 会自动处理此步骤。

工作流程:
  1. 读取 solver_data.bin 获取 GLL 点坐标 (x,y,z)、ibool，并反算 s362ani+CRUST1.0 速度
  2. 唯一点去重 (ibool 展开 ~90 万点 → ~20 万唯一点，节省 ~75% 计算)
  3. 坐标转换: 非量纲 (x,y,z) → 地理 (lat, lon, depth_km)
  4. NaN 预填充: 消除 NC 中 NaN 对插值 stencil 的污染
  5. 扰动模式 (--use-perturbation, FWEA23/EARA 必需):
       δV = interp(V_nc − V_ref_nc)     # 仅插值平滑扰动
       V_gll = V_ref_gll + δV           # 间断面来自 mesher 精确参考
       域外 δV→0 → 自动保持 V_ref_gll
     绝对值模式 (SinoScope 等无参考场):
       V_gll = interp(V_nc)，域外用 V_ref_gll / 1DREF
  6. 写出 Fortran 无格式二进制 GLL 文件

用法:
    # FWEA23 (NetCDF, 各向异性, 含扰动插值)
    python convert_nc_to_gll.py \\
        --mesh-dir DATABASES_MPI \\
        --model-nc ../models/FWEA23.r0.0-n4.nc \\
        --output-dir DATA/GLL \\
        --anisotropic --use-perturbation --horizontal-fill blend

    # EARA2024 (NetCDF, 逐分量参考场)
    python convert_nc_to_gll.py \\
        --mesh-dir DATABASES_MPI \\
        --model-nc ../models/EARA2024.r0.0-n4.nc \\
        --output-dir DATA/GLL \\
        --anisotropic --use-perturbation --horizontal-fill blend

    # SinoScope (HDF5, P 各向同性 + S 各向异性)
    python convert_nc_to_gll.py \\
        --mesh-dir DATABASES_MPI \\
        --model-h5 ../models/FWI_SinoScope_1.0_JMa_ET_AL_2022.h5 \\
        --output-dir DATA/GLL \\
        --anisotropic --horizontal-fill blend

    # FWEA23 验证: 对比原始 GLL 文件
    python convert_nc_to_gll.py \\
        --mesh-dir DATABASES_MPI \\
        --model-nc ../models/FWEA23.r0.0-n4.nc \\
        --output-dir DATA/GLL \\
        --anisotropic --use-perturbation \\
        --reference-gll-dir ../model_updated

    # double 精度 (SPECFEM 以 --enable-double-precision 编译时)
    python convert_nc_to_gll.py ... --double

Par_file 配置:
  MODEL = GLL          (v7.0.0, 1D Q 衰减)
  MODEL = gll_qmu      (v7.0.0, 3D Q 衰减, 需 qmu.bin)

作者: EASTASIA-FWI Team
日期: 2026-03
"""

import sys
import struct
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set
import argparse
import numpy as np
from numpy.typing import DTypeLike

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

NGLLX, NGLLY, NGLLZ = 5, 5, 5
GLL_PER_ELEM = NGLLX * NGLLY * NGLLZ  # 125
R_EARTH_KM = 6371.0
R_EARTH_M = 6371000.0

# SPECFEM 非量纲化常数（用于从弹性模量反算速度）
_PI = 3.14159265358979323846
_GRAV = 6.6723e-11
_RHOAV = 5514.3
_scaleval = np.sqrt(_PI * _GRAV * _RHOAV)
_SCALE_V = 1000.0 / (R_EARTH_M * _scaleval)   # km/s → non-dim
_SCALE_RHO = 1000.0 / _RHOAV                   # g/cm³ → non-dim

# 1DREF 参考模型: 优先在 specfem/ 同级查找，其次在项目根目录
DEFAULT_REF_MODEL_PATHS = [
    'specfem3d_globe_code_new/utils/models_s362ani/REF',
    '../specfem3d_globe/utils/models_s362ani/REF',
]


def load_1dref_model(ref_path: Path) -> dict:
    """
    加载 1DREF (STW105) 参考模型。

    REF 文件格式（跳过前 3 行头部）：
    radius(m)  density(kg/m3)  vpv(m/s)  vsv(m/s)  Qkappa  Qmu  vph(m/s)  vsh(m/s)  eta

    返回 dict，包含按深度(km)线性插值的函数，单位已转为 km/s 和 g/cm3。
    """
    from scipy.interpolate import interp1d

    raw = np.loadtxt(ref_path, skiprows=3)

    # REF 文件头第 3 行 "750 180 358 717 739"：index 358 = CMB（1-based）
    # GLL reg1 只覆盖壳幔，排除外核(0-179)/内核(180-357)
    # 读取 CMB 边界索引
    with open(ref_path) as f:
        f.readline(); f.readline()
        header3 = f.readline().split()
    # header3[2] = NR_OC_REF（外核最后一层的 1-based 索引）
    # 壳幔从下一层开始：0-based index = int(header3[2])
    mantle_start = int(header3[2])  # 358 → 0-based idx 358 = 壳幔侧 CMB
    data = raw[mantle_start:]

    radius_m = data[:, 0]
    depth_km = (R_EARTH_M - radius_m) / 1000.0

    # 翻转使 depth 递增（原始是 radius 递增 = depth 递减）
    depth_km = depth_km[::-1]
    data = data[::-1]

    # 跳过海水层（vsv=0 的点），从固体地壳开始
    # 1DREF 最浅 ~3 km 是海水层：vsv=0, vpv=1.45, rho=1.02
    vsv_col = data[:, 3]
    solid_mask = vsv_col > 0
    if solid_mask.any():
        first_solid = np.argmax(solid_mask)
        if first_solid > 0:
            depth_km = depth_km[first_solid:]
            data = data[first_solid:]

    ref = {}
    # 速度: m/s → km/s;  密度: kg/m3 → g/cm3
    # 用边界值填充（不外推），避免 depth<0 或 depth>CMB 时产生不物理的值
    def make_interp(vals):
        return interp1d(depth_km, vals, bounds_error=False,
                        fill_value=(vals[0], vals[-1]))  # type: ignore[arg-type]

    ref['vpv'] = make_interp(data[:, 2] / 1000.0)
    ref['vsv'] = make_interp(data[:, 3] / 1000.0)
    ref['vph'] = make_interp(data[:, 6] / 1000.0)
    ref['vsh'] = make_interp(data[:, 7] / 1000.0)
    ref['rho'] = make_interp(data[:, 1] / 1000.0)
    ref['eta'] = make_interp(data[:, 8])
    ref['qmu'] = make_interp(data[:, 5])
    ref['depth_km'] = depth_km
    return ref


def get_1dref_values(ref: dict, depth_km: np.ndarray, varnames: List[str],
                     dtype: DTypeLike = np.float32) -> dict:
    """按深度从 1DREF 模型获取值。rho 已为 g/cm3，速度已为 km/s。"""
    result = {}
    for v in varnames:
        if v in ref:
            result[v] = ref[v](depth_km).astype(dtype)
    return result


def xyz_to_lat_lon_depth(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SPECFEM3D Globe 非量纲化坐标 (x,y,z) → (lat°, lon°, depth_km)。

    solver_data.bin 中的坐标已被 R_EARTH 归一化（r ≈ 1.0 为地表），
    需乘以 R_EARTH_KM 才能得到真实半径。
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-6)
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    depth = R_EARTH_KM * (1.0 - r)
    return lat, lon, depth


def _read_fortran_record(f) -> bytes:
    """读取一条 Fortran 无格式记录（含 record marker）。"""
    rec_len = struct.unpack('i', f.read(4))[0]
    data = f.read(rec_len)
    tail = struct.unpack('i', f.read(4))[0]
    if tail != rec_len:
        raise ValueError(f"Fortran record marker mismatch: {rec_len} vs {tail}")
    return data


def _skip_fortran_record(f):
    """跳过一条 Fortran 无格式记录。"""
    rec_len = struct.unpack('i', f.read(4))[0]
    f.seek(rec_len, 1)
    f.read(4)


def read_solver_data(solver_path: Path, extract_velocities: bool = False
                     ) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                Optional[Dict[str, np.ndarray]]]:
    """
    读取 proc***_reg1_solver_data.bin（Fortran 无格式，含 record marker）。

    当 extract_velocities=True 时，继续读取弹性模量记录并反算
    s362ani+CRUST1.0 速度场（km/s, g/cm³），作为深部/域外 3D 参考。

    solver_data.bin 记录顺序 (reg1, TRANSVERSE_ISOTROPY, !ANISOTROPIC_3D_MANTLE):
        1. nspec  2. nglob  3-5. x,y,z  6. ibool  7. idoubling  8. ispec_is_tiso
        9-17. 9×Jacobian  18. rho  19. kappav  20. muv  21. kappah  22. muh  23. eta

    Returns:
        nspec, nglob, x, y, z, ibool, ref_velocities
        ref_velocities: None 或 dict {'vpv','vph','vsv','vsh','eta','rho'} (4D arrays, km/s & g/cm³)
    """
    with open(solver_path, 'rb') as f:
        nspec = struct.unpack('i', _read_fortran_record(f))[0]
        nglob = struct.unpack('i', _read_fortran_record(f))[0]
        x_buf = _read_fortran_record(f)
        y_buf = _read_fortran_record(f)
        z_buf = _read_fortran_record(f)
        nbytes = len(x_buf)
        dtype_xyz = np.float32 if nbytes == nglob * 4 else np.float64
        x = np.frombuffer(x_buf, dtype=dtype_xyz)
        y = np.frombuffer(y_buf, dtype=dtype_xyz)
        z = np.frombuffer(z_buf, dtype=dtype_xyz)
        ibool_buf = _read_fortran_record(f)
        ibool = np.frombuffer(ibool_buf, dtype=np.int32)
        ibool = ibool.reshape((NGLLX, NGLLY, NGLLZ, nspec), order='F')

        if not extract_velocities:
            return nspec, nglob, x, y, z, ibool, None

        cr = dtype_xyz
        ngll = GLL_PER_ELEM * nspec

        # 7. idoubling, 8. ispec_is_tiso
        _skip_fortran_record(f)
        _skip_fortran_record(f)
        # 9-17. 9 个 Jacobian 数组
        for _ in range(9):
            _skip_fortran_record(f)
        # 18. rhostore  19. kappavstore  20. muvstore
        rho_nd = np.frombuffer(_read_fortran_record(f), dtype=cr).copy()
        kappav = np.frombuffer(_read_fortran_record(f), dtype=cr).copy()
        muv = np.frombuffer(_read_fortran_record(f), dtype=cr).copy()

        # 尝试读 21-23（TRANSVERSE_ISOTROPY 时存在）
        aniso = False
        try:
            kappah = np.frombuffer(_read_fortran_record(f), dtype=cr).copy()
            if len(kappah) == ngll:
                muh = np.frombuffer(_read_fortran_record(f), dtype=cr).copy()
                eta_nd = np.frombuffer(_read_fortran_record(f), dtype=cr).copy()
                aniso = True
        except Exception:
            pass

    safe_rho = np.maximum(rho_nd, 1e-20)
    vpv = np.sqrt(np.maximum((kappav + 4.0 / 3.0 * muv) / safe_rho, 0.0)) / _SCALE_V
    vsv = np.sqrt(np.maximum(muv / safe_rho, 0.0)) / _SCALE_V
    rho_out = rho_nd / _SCALE_RHO

    if aniso:
        vph = np.sqrt(np.maximum((kappah + 4.0 / 3.0 * muh) / safe_rho, 0.0)) / _SCALE_V
        vsh = np.sqrt(np.maximum(muh / safe_rho, 0.0)) / _SCALE_V
        eta_out = eta_nd
    else:
        vph, vsh = vpv.copy(), vsv.copy()
        eta_out = np.ones(ngll, dtype=cr)

    shape = (NGLLX, NGLLY, NGLLZ, nspec)
    ref_vel = {
        'vpv': vpv.reshape(shape, order='F'),
        'vph': vph.reshape(shape, order='F'),
        'vsv': vsv.reshape(shape, order='F'),
        'vsh': vsh.reshape(shape, order='F'),
        'eta': eta_out.reshape(shape, order='F'),
        'rho': rho_out.reshape(shape, order='F'),
    }
    return nspec, nglob, x, y, z, ibool, ref_vel


def _fill_nan_nearest_3d(arr: np.ndarray) -> np.ndarray:
    """用最近有效值填充 3D 数组中的 NaN，消除插值 stencil 的 NaN 污染。"""
    from scipy.ndimage import distance_transform_edt
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr
    filled = arr.copy()
    indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
    filled[nan_mask] = arr[tuple(indices)][nan_mask]
    return filled


def _build_nc_ref_map(ds, varnames: List[str]) -> Dict[str, str]:
    """
    构建 NC 变量 → 参考场变量名映射。

    优先逐分量参考 (vpv_ref 等，EARA2024)，其次 vp0/vs0 (FWEA23)。
    """
    ref_nc_map: Dict[str, str] = {}
    for v in varnames:
        per_component_ref = f'{v}_ref'
        if per_component_ref in ds:
            ref_nc_map[v] = per_component_ref
        elif v in ('vpv', 'vph', 'vp') and 'vp0' in ds:
            ref_nc_map[v] = 'vp0'
        elif v in ('vsv', 'vsh', 'vs') and 'vs0' in ds:
            ref_nc_map[v] = 'vs0'
        elif v in ('vpv', 'vph', 'vp') and 'vp_ref' in ds:
            ref_nc_map[v] = 'vp_ref'
        elif v in ('vsv', 'vsh', 'vs') and 'vs_ref' in ds:
            ref_nc_map[v] = 'vs_ref'
        elif v == 'rho' and 'rho_ref' in ds:
            ref_nc_map[v] = 'rho_ref'
        elif v == 'eta' and 'eta_ref' in ds:
            ref_nc_map[v] = 'eta_ref'
    return ref_nc_map


def interpolate_from_netcdf(
    lat: np.ndarray, lon: np.ndarray, depth: np.ndarray,
    ds, varnames: List[str], fill: float = np.nan,
    dtype: DTypeLike = np.float32,
    ref_model: Optional[dict] = None,
    horizontal_fill_mode: str = 'nearest',
    blend_width_deg: float = 5.0,
    interp_method: str = 'linear',
    use_perturbation: bool = False,
    quiet: bool = False,
) -> Tuple[dict, Set[str]]:
    """
    从 xarray Dataset 在 (lat, lon, depth) 插值。

    扰动模式 (use_perturbation=True，且变量有 NC 参考场):
      仅返回 δV = interp(V_nc − V_ref_nc)。
      域外（深部/水平）δV → 0（可经 blend 平滑过渡到 0）。
      调用方必须执行 V_gll = V_ref_gll + δV，间断面由 mesher 参考保留。

    绝对值模式 (无参考场或 use_perturbation=False):
      返回 interp(V_nc)；域外用 1DREF / nearest / blend。

    Returns:
        (result, perturbation_keys)
        perturbation_keys: 结果中为扰动量、需叠加 V_ref_gll 的变量名集合
    """
    from scipy.interpolate import RegularGridInterpolator

    def _log(msg: str) -> None:
        if not quiet:
            print(msg)

    lon_nc = ds['longitude'].values.copy()
    lat_nc = ds['latitude'].values.copy()
    depth_nc = ds['depth'].values.copy()

    lon = lon.copy()
    lon[lon < 0] += 360
    if lon_nc.max() <= 180 and lon_nc.min() >= -180:
        lon_nc[lon_nc < 0] += 360

    for name, arr in [('longitude', lon_nc), ('latitude', lat_nc), ('depth', depth_nc)]:
        if arr[-1] < arr[0]:
            raise ValueError(f"NC 坐标 {name} 非递增，需要翻转")

    sample_var = next((v for v in varnames if v in ds), None)
    if sample_var is None:
        return {}, set()
    dims = [str(d) for d in ds[sample_var].dims]
    dim_order = []
    for d in dims:
        if 'lon' in d.lower():
            dim_order.append('lon')
        elif 'lat' in d.lower():
            dim_order.append('lat')
        elif 'dep' in d.lower():
            dim_order.append('dep')
    coord_map = {'lon': lon_nc, 'lat': lat_nc, 'dep': depth_nc}
    query_map = {'lon': lon, 'lat': lat, 'dep': depth}
    points = tuple(coord_map[k] for k in dim_order)
    pts = np.column_stack([query_map[k] for k in dim_order])

    depth_max_nc = depth_nc.max()
    deep_mask = depth > depth_max_nc

    lat_min_nc, lat_max_nc = lat_nc.min(), lat_nc.max()
    lon_min_nc, lon_max_nc = lon_nc.min(), lon_nc.max()
    horizontal_outside = (
        (lat < lat_min_nc) | (lat > lat_max_nc) |
        (lon < lon_min_nc) | (lon > lon_max_nc)
    )
    # 域外总掩码：扰动模式下这些点 δV 必须归零
    domain_outside = deep_mask | horizontal_outside

    def _outside_distance_deg() -> np.ndarray:
        dlat = np.where(lat < lat_min_nc, lat_min_nc - lat,
                        np.where(lat > lat_max_nc, lat - lat_max_nc, 0.0))
        dlon = np.where(lon < lon_min_nc, lon_min_nc - lon,
                        np.where(lon > lon_max_nc, lon - lon_max_nc, 0.0))
        return np.sqrt(dlat**2 + dlon**2)

    ref_nc_map: Dict[str, str] = {}
    if use_perturbation:
        ref_nc_map = _build_nc_ref_map(ds, varnames)
        if ref_nc_map:
            _log(f"  扰动插值(仅δV): {', '.join(f'{k}←{v}' for k, v in ref_nc_map.items())}")
            _log("  ★ 正确公式: V_gll = V_ref_gll(mesher) + interp(δV_nc)")
        else:
            _log("  ⚠️ --use-perturbation 但 NC 无参考场 (vp0/vs0 或 *_ref)，回退绝对值插值")

    n_total = len(lat)
    n_deep = int(deep_mask.sum())
    n_h_out = int(horizontal_outside.sum())
    _log(f"  域外统计: 总点数={n_total}, 深度域外={n_deep} ({n_deep/n_total*100:.1f}%), "
         f"水平域外={n_h_out} ({n_h_out/n_total*100:.1f}%), 模式={horizontal_fill_mode}")
    _log(f"  插值方法: {interp_method}, NaN预填充: 开启")

    result = {}
    perturbation_keys: Set[str] = set()

    for v in varnames:
        if v not in ds:
            continue
        arr = ds[v].values.copy()
        if v == 'rho' and np.nanmean(arr) > 100:
            arr = arr / 1000.0

        is_delta = bool(use_perturbation and v in ref_nc_map)
        if is_delta:
            ref_name = ref_nc_map[v]
            ref_arr_nc = ds[ref_name].values.copy()
            if v == 'rho' and np.nanmean(ref_arr_nc) > 100:
                ref_arr_nc = ref_arr_nc / 1000.0
            arr_interp = arr - ref_arr_nc
            perturbation_keys.add(v)
        else:
            arr_interp = arr

        # NaN 预填充：使线性/cubic stencil 不受 NaN 污染
        n_nan_before = int(np.isnan(arr_interp).sum())
        arr_filled = _fill_nan_nearest_3d(arr_interp)
        if n_nan_before > 0 and v == varnames[0]:
            n_total_nc = arr_interp.size
            _log(f"  NaN预填充: {n_nan_before}/{n_total_nc} "
                 f"({n_nan_before/n_total_nc*100:.1f}%) 个NC网格点")

        interp_func = RegularGridInterpolator(
            points, arr_filled, method=interp_method,
            bounds_error=False, fill_value=np.nan)
        out = interp_func(pts)
        nan_mask = ~np.isfinite(out)

        if is_delta:
            # ── 扰动模式：域外 δV → 0（叠加 V_ref_gll 后即保持 3D 参考）──
            # 深部：严格归零
            if deep_mask.any():
                out[deep_mask] = 0.0
                nan_mask[deep_mask] = False

            # 水平域外：blend 时近边界保留部分 nearest δV，远边界→0；否则 δV=0
            h_mask = horizontal_outside & nan_mask
            if h_mask.any():
                if horizontal_fill_mode == 'blend':
                    nn_interp = RegularGridInterpolator(
                        points, arr_filled, method='nearest',
                        bounds_error=False, fill_value=None)
                    nn_vals = nn_interp(pts[h_mask]).astype(dtype)
                    dist = _outside_distance_deg()[h_mask]
                    width = max(float(blend_width_deg), 1e-6)
                    w_zero = np.clip(dist / width, 0.0, 1.0).astype(dtype)
                    out[h_mask] = (1.0 - w_zero) * nn_vals  # → 0
                else:
                    # nearest / ref：域外扰动归零，避免把异常外推进吸收边界
                    out[h_mask] = 0.0
                nan_mask[h_mask] = False

            # 其余无效点（如 NC 内部 NaN 残差）→ 0
            if nan_mask.any():
                out[nan_mask] = 0.0
            still_nan = ~np.isfinite(out)
            if still_nan.any():
                out[still_nan] = 0.0

        else:
            # ── 绝对值模式：域外用 1DREF / nearest / blend ──
            if ref_model is not None and v in ref_model and deep_mask.any():
                out[deep_mask] = ref_model[v](depth[deep_mask]).astype(dtype)
                nan_mask[deep_mask] = False

            if horizontal_fill_mode == 'ref' and ref_model is not None and v in ref_model:
                h_mask = horizontal_outside & nan_mask
                if h_mask.any():
                    out[h_mask] = ref_model[v](depth[h_mask]).astype(dtype)
                    nan_mask[h_mask] = False
            elif horizontal_fill_mode == 'blend' and ref_model is not None and v in ref_model:
                h_mask = horizontal_outside & nan_mask
                if h_mask.any():
                    nn_interp = RegularGridInterpolator(
                        points, arr_filled, method='nearest',
                        bounds_error=False, fill_value=None)
                    nn_vals = nn_interp(pts[h_mask]).astype(dtype)
                    ref_vals = ref_model[v](depth[h_mask]).astype(dtype)
                    dist = _outside_distance_deg()[h_mask]
                    width = max(float(blend_width_deg), 1e-6)
                    w_ref = np.clip(dist / width, 0.0, 1.0).astype(dtype)
                    out[h_mask] = (1.0 - w_ref) * nn_vals + w_ref * ref_vals
                    nan_mask[h_mask] = False
            elif horizontal_fill_mode not in ('nearest', 'ref', 'blend'):
                raise ValueError(f"未知 horizontal_fill_mode: {horizontal_fill_mode}")

            if nan_mask.any():
                interp_nn = RegularGridInterpolator(
                    points, arr_filled, method='nearest',
                    bounds_error=False, fill_value=None)
                out[nan_mask] = interp_nn(pts[nan_mask])

            still_nan = ~np.isfinite(out)
            if still_nan.any():
                if ref_model is not None and v in ref_model:
                    out[still_nan] = ref_model[v](depth[still_nan]).astype(dtype)
                else:
                    out[still_nan] = fill

        result[v] = out.astype(dtype)

    if perturbation_keys:
        n_out = int(domain_outside.sum())
        _log(f"  扰动键: {sorted(perturbation_keys)} "
             f"(域外 {n_out} 点 δV→0，由调用方叠加 V_ref_gll)")

    return result, perturbation_keys


def interpolate_from_csv(
    lat: np.ndarray, lon: np.ndarray, depth: np.ndarray,
    csv_path: Path, varnames: List[str], fill: float = np.nan,
    dtype: DTypeLike = np.float32,
) -> dict:
    """从 CSV（列: lon, lat, depth, vp, vs, rho...）插值。"""
    import pandas as pd
    from scipy.interpolate import LinearNDInterpolator

    df = pd.read_csv(csv_path)
    cols = [c.strip().lower() for c in df.columns]
    lon_src = np.asarray(df['lon'].values if 'lon' in cols else df['longitude'].values, dtype=np.float64)
    lat_src = np.asarray(df['lat'].values if 'lat' in cols else df['latitude'].values, dtype=np.float64)
    dep_src = np.asarray(df['depth'].values, dtype=np.float64)
    pts_src = np.column_stack([lon_src, lat_src, dep_src])
    pts_query = np.column_stack([lon, lat, depth])

    result = {}
    for v in varnames:
        vc = v.lower()
        if vc not in cols and v not in df.columns:
            continue
        col = [c for c in df.columns if c.strip().lower() == vc][0]
        vals = df[col].values.astype(np.float64)
        interp = LinearNDInterpolator(pts_src, vals, fill_value=fill)
        out = interp(pts_query).astype(dtype)
        out[~np.isfinite(out)] = fill
        result[v] = out
    return result


def write_gll_binary(path: Path, data: np.ndarray, dtype: DTypeLike = np.float32) -> None:
    """
    按 Fortran unformatted 格式写出 (NGLLX, NGLLY, NGLLZ, nspec)。
    需与 SPECFEM 的 CUSTOM_REAL 一致：float32 (4B) 或 float64 (8B)。
    Intel Fortran 与 gfortran 默认均使用 4 字节 record marker。
    """
    flat = np.asarray(data, dtype=dtype).flatten(order='F')
    nbytes = len(flat) * np.dtype(dtype).itemsize
    with open(path, 'wb') as f:
        f.write(struct.pack('i', nbytes))
        f.write(flat.tobytes())
        f.write(struct.pack('i', nbytes))


def read_gll_binary(path: Path, nspec: int, dtype: DTypeLike = np.float32) -> np.ndarray:
    """
    读取 Fortran unformatted 的 GLL bin（单变量）并返回 4D 数组。
    """
    with open(path, 'rb') as f:
        rec_len = struct.unpack('i', f.read(4))[0]
        payload = f.read(rec_len)
        tail = struct.unpack('i', f.read(4))[0]
    if rec_len != tail:
        raise ValueError(f"{path.name}: record marker mismatch {rec_len} vs {tail}")

    expected = NGLLX * NGLLY * NGLLZ * nspec * np.dtype(dtype).itemsize
    if rec_len != expected:
        raise ValueError(
            f"{path.name}: record bytes={rec_len}, expected={expected} "
            f"(nspec={nspec}, dtype={np.dtype(dtype).name})"
        )
    arr = np.frombuffer(payload, dtype=dtype)
    return arr.reshape((NGLLX, NGLLY, NGLLZ, nspec), order='F')


def load_h5_as_dataset(h5_path: Path):
    """
    读取 SinoScope 等 HDF5 模型文件，返回 xarray Dataset。

    HDF5 结构:
      /model_coordinates/{longitude, latitude, depth_in_meter}
      /model_parameters/{vp, vsv, vsh, rho}

    单位转换: 速度 m/s → km/s, 深度 m → km, 密度保持 kg/m3（后续自动转换）。
    变量映射: vp → vpv = vph（P 波各向同性），补 eta = 1.0。
    """
    import h5py
    import xarray as xr

    with h5py.File(h5_path, 'r') as f:
        lon = f['/model_coordinates/longitude'][:]
        lat = f['/model_coordinates/latitude'][:]
        depth_m = f['/model_coordinates/depth_in_meter'][:]
        depth_km = depth_m / 1000.0

        vp_raw = f['/model_parameters/vp'][:] / 1000.0
        vsv_raw = f['/model_parameters/vsv'][:] / 1000.0
        vsh_raw = f['/model_parameters/vsh'][:] / 1000.0
        rho_raw = f['/model_parameters/rho'][:]

    # HDF5 数据形状: (nlat, nlon, ndep) → 转置为 (ndep, nlat, nlon) 与 EMC NC 一致
    vp_3d = vp_raw.transpose(2, 0, 1)
    vsv_3d = vsv_raw.transpose(2, 0, 1)
    vsh_3d = vsh_raw.transpose(2, 0, 1)
    rho_3d = rho_raw.transpose(2, 0, 1)

    ds = xr.Dataset(
        {
            'vpv': (['depth', 'latitude', 'longitude'], vp_3d),
            'vph': (['depth', 'latitude', 'longitude'], vp_3d),
            'vsv': (['depth', 'latitude', 'longitude'], vsv_3d),
            'vsh': (['depth', 'latitude', 'longitude'], vsh_3d),
            'eta': (['depth', 'latitude', 'longitude'], np.ones_like(vp_3d)),
            'rho': (['depth', 'latitude', 'longitude'], rho_3d),
        },
        coords={'depth': depth_km, 'latitude': lat, 'longitude': lon},
    )

    return ds


def voigt_vp(vpv: np.ndarray, vph: np.ndarray) -> np.ndarray:
    return np.sqrt((2 * vpv**2 + vph**2) / 3.0).astype(vpv.dtype)


def voigt_vs(vsv: np.ndarray, vsh: np.ndarray) -> np.ndarray:
    return np.sqrt((2 * vsv**2 + vsh**2) / 3.0).astype(vsv.dtype)


def process_one_proc(
    proc_id: int,
    mesh_dir: Path,
    output_dir: Path,
    model_nc: Optional[Path],
    model_csv: Optional[Path],
    anisotropic: bool,
    fill_value: float,
    dtype: DTypeLike = np.float32,
    global_min: Optional[dict] = None,
    global_max: Optional[dict] = None,
    ref_model: Optional[dict] = None,
    apply_physical_floor: bool = False,
    horizontal_fill_mode: str = 'nearest',
    blend_width_deg: float = 5.0,
    reference_gll_dir: Optional[Path] = None,
    compare_all_procs: bool = False,
    interp_method: str = 'linear',
    use_perturbation: bool = False,
    model_h5: Optional[Path] = None,
    preloaded_ds=None,
    ref_gll_dir: Optional[Path] = None,
    use_3d_fallback: bool = False,
    quiet: bool = False,
) -> bool:
    solver = mesh_dir / f"proc{proc_id:06d}_reg1_solver_data.bin"
    if not solver.exists():
        return False

    def _log(msg: str) -> None:
        if not quiet:
            print(msg)

    # 扰动模式必须读取 mesher 的 3D 参考；绝对值模式在 use_3d_fallback 时读取
    need_ref_gll = use_perturbation or use_3d_fallback
    nspec, nglob, x, y, z, ibool, ref_vel_4d = read_solver_data(
        solver, extract_velocities=(need_ref_gll and ref_gll_dir is None))

    # 外部 GLL 参考目录（调试用）可替代 solver_data 反算
    def _load_ref_gll_var(varname: str) -> Optional[np.ndarray]:
        if ref_vel_4d is not None and varname in ref_vel_4d:
            return ref_vel_4d[varname].astype(dtype, copy=False)
        if ref_gll_dir is not None:
            ref_path = ref_gll_dir / f"proc{proc_id:06d}_reg1_{varname}.bin"
            if ref_path.exists():
                try:
                    return read_gll_binary(ref_path, nspec, dtype=dtype)
                except Exception as e:
                    if not quiet:
                        print(f"   [3D-REF] ⚠️ {varname}: 读取失败 ({e})")
        return None

    if use_perturbation and ref_vel_4d is None and ref_gll_dir is None:
        print(f"❌ proc{proc_id:06d}: --use-perturbation 需要从 solver_data.bin "
              f"反算 V_ref_gll，但未能提取弹性模量。请确认 Phase 2 使用 "
              f"MODEL=s362ani（含 TRANSVERSE_ISOTROPY）。")
        return False

    # ibool: (NGLLX, NGLLY, NGLLZ, nspec)，Fortran 1-based → 0-based
    iglob_flat = ibool.ravel(order='F') - 1

    # 唯一点去重：ibool 中相邻元素共享 GLL 节点，
    # 对 nglob 个唯一点插值后 scatter 回 904500 个条目，避免重复计算
    unique_iglob, inverse_idx = np.unique(iglob_flat, return_inverse=True)
    x_unique = x[unique_iglob]
    y_unique = y[unique_iglob]
    z_unique = z[unique_iglob]

    lat_uniq, lon_uniq, depth_uniq = xyz_to_lat_lon_depth(x_unique, y_unique, z_unique)

    if not quiet:
        _log(f"   [诊断 proc{proc_id:06d}] nspec={nspec}, nglob={nglob}, "
             f"ibool展开={len(iglob_flat)}, 唯一点={len(unique_iglob)} "
             f"(节省{(1-len(unique_iglob)/len(iglob_flat))*100:.0f}%)")
        _log(f"      r (非量纲): {np.sqrt(x**2+y**2+z**2).min():.6f} ~ "
             f"{np.sqrt(x**2+y**2+z**2).max():.6f}")
        _log(f"      lat: {lat_uniq.min():.2f}° ~ {lat_uniq.max():.2f}°")
        _log(f"      lon: {lon_uniq.min():.2f}° ~ {lon_uniq.max():.2f}°")
        _log(f"      depth: {depth_uniq.min():.2f} ~ {depth_uniq.max():.2f} km")

    # 后续插值使用唯一点坐标
    lat_flat, lon_flat, depth_flat = lat_uniq, lon_uniq, depth_uniq

    # 加载模型: 预加载 Dataset / NC / HDF5 / CSV（优先级依次）
    ds_to_close = None
    perturbation_keys: Set[str] = set()
    if preloaded_ds is not None:
        ds = preloaded_ds
    elif model_nc and model_nc.exists():
        import xarray as xr
        ds = xr.open_dataset(model_nc)
        ds_to_close = ds
    elif model_h5 and model_h5.exists():
        ds = load_h5_as_dataset(model_h5)
    else:
        ds = None

    if ds is not None:
        if anisotropic and all(v in ds for v in ['vpv', 'vph', 'vsv', 'vsh']):
            varnames = ['vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho', 'qmu']
            varnames = [v for v in varnames if v in ds or v in ['vpv', 'vph', 'vsv', 'vsh', 'rho']]
        else:
            if 'vp' in ds and 'vs' in ds:
                varnames = ['vp', 'vs', 'rho']
            else:
                varnames = ['vpv', 'vph', 'vsv', 'vsh', 'rho']
                if 'eta' in ds:
                    varnames.insert(4, 'eta')
        interped, perturbation_keys = interpolate_from_netcdf(
            lat_flat, lon_flat, depth_flat, ds, varnames,
            fill_value, dtype, ref_model=ref_model,
            horizontal_fill_mode=horizontal_fill_mode,
            blend_width_deg=blend_width_deg,
            interp_method=interp_method,
            use_perturbation=use_perturbation,
            quiet=quiet)
        if ds_to_close is not None:
            ds_to_close.close()
    elif model_csv and model_csv.exists():
        varnames = ['vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho'] if anisotropic else ['vp', 'vs', 'rho']
        interped = interpolate_from_csv(lat_flat, lon_flat, depth_flat, model_csv, varnames, fill_value, dtype)
        ds = None
    else:
        print(f"❌ 未找到模型文件: {model_nc or model_csv or model_h5}")
        return False

    if use_perturbation and not perturbation_keys:
        print(f"❌ proc{proc_id:06d}: --use-perturbation 已开启，但未能对任何变量构建 "
              f"δV（NC 缺少 vp0/vs0 或 *_ref）。拒绝输出错误的绝对值 GLL。")
        return False

    # 从唯一点插值结果 scatter 回 (NGLLX, NGLLY, NGLLZ, nspec)
    def to_gll(v: np.ndarray) -> np.ndarray:
        return v[inverse_idx].reshape((NGLLX, NGLLY, NGLLZ, nspec), order='F')

    # 安全值与默认值（密度单位 g/cm³）
    vmin = {'vp': 0.5, 'vs': 0.3, 'rho': 1.0, 'vpv': 0.5, 'vph': 0.5,
            'vsv': 0.3, 'vsh': 0.3, 'eta': 0.5, 'qmu': 20.0}
    default = {'vp': 6.0, 'vs': 3.5, 'rho': 3.3, 'vpv': 6.0, 'vph': 6.0,
               'vsv': 3.5, 'vsh': 3.5, 'eta': 1.0, 'qmu': 150.0}
    if 'rho' not in interped:
        interped['rho'] = np.full(lat_flat.shape, default['rho'], dtype=dtype)
    if anisotropic and 'eta' not in interped:
        interped['eta'] = np.full(lat_flat.shape, default['eta'], dtype=dtype)

    # 域外掩码（仅用于无 NC 参考场的绝对值变量）
    replace_mask: Optional[np.ndarray] = None
    if use_3d_fallback and ds is not None:
        depth_all = depth_flat[inverse_idx]
        lat_all = lat_flat[inverse_idx]
        lon_all = lon_flat[inverse_idx]
        depth_nc = ds['depth'].values
        lat_nc_arr = ds['latitude'].values
        lon_nc_arr = ds['longitude'].values
        depth_max_nc = float(depth_nc.max())
        lat_min_nc_v = float(lat_nc_arr.min())
        lat_max_nc_v = float(lat_nc_arr.max())
        lon_min_nc_v = float(lon_nc_arr.min())
        lon_max_nc_v = float(lon_nc_arr.max())

        lon_all_adj = lon_all.copy()
        lon_all_adj[lon_all_adj < 0] += 360
        if float(np.max(lon_nc_arr)) <= 180:
            tmp = np.asarray(lon_nc_arr, dtype=np.float64).copy()
            tmp[tmp < 0] += 360
            lon_min_nc_v, lon_max_nc_v = float(tmp.min()), float(tmp.max())

        deep_mask_all = depth_all > depth_max_nc
        horiz_outside_all = (
            (lat_all < lat_min_nc_v) | (lat_all > lat_max_nc_v) |
            (lon_all_adj < lon_min_nc_v) | (lon_all_adj > lon_max_nc_v)
        )
        replace_mask = deep_mask_all | horiz_outside_all
        if not quiet:
            n_all = len(replace_mask)
            n_replace = int(replace_mask.sum())
            _log(f"   [3D-REF] 域外点 {n_replace}/{n_all} "
                 f"({n_replace / n_all * 100:.1f}%): "
                 f"深部={int(deep_mask_all.sum())}, "
                 f"水平={int(horiz_outside_all.sum())}")

    first_pert = sorted(perturbation_keys)[0] if perturbation_keys else None
    for k, v in interped.items():
        delta_or_abs = to_gll(v)

        if k in perturbation_keys:
            # ★ 正确转换: V_gll = V_ref_gll + δV
            ref_arr = _load_ref_gll_var(k)
            if ref_arr is None:
                print(f"❌ proc{proc_id:06d}: 变量 {k} 有 δV 但缺少 V_ref_gll")
                return False
            vv = ref_arr + delta_or_abs
            if not quiet and k == first_pert:
                _log(f"   [PERT] {k}: V_gll = V_ref_gll + δV "
                     f"(mean|δV|={float(np.mean(np.abs(delta_or_abs))):.4f}, "
                     f"max|δV|={float(np.max(np.abs(delta_or_abs))):.4f})")
        else:
            vv = delta_or_abs
            # 绝对值变量：域外用 mesher 3D 参考替换（保留间断面）
            if replace_mask is not None and use_3d_fallback:
                ref_arr = _load_ref_gll_var(k)
                if ref_arr is not None:
                    flat = vv.ravel(order='F').copy()
                    flat[replace_mask] = ref_arr.ravel(order='F')[replace_mask]
                    vv = flat.reshape((NGLLX, NGLLY, NGLLZ, nspec), order='F')

        nan_count = int((~np.isfinite(vv)).sum())
        if nan_count > 0:
            print(f"  ⚠️ {k}: {nan_count} 个 NaN 残留，用默认值 {default.get(k, 3.5)} 填充")
            vv = np.asarray(vv, dtype=dtype).copy()
            vv[~np.isfinite(vv)] = default.get(k, 3.5)
        if apply_physical_floor and k in vmin and np.any(vv < vmin[k]):
            vv = np.asarray(vv, dtype=dtype).copy()
            vv[vv < vmin[k]] = vmin[k]
        interped[k] = vv

    if not quiet and 'rho' in interped:
        _log(f"  rho mean={float(np.nanmean(interped['rho'])):.2f} g/cm³")

    vals = "  ".join(f"{k}:{v.min():.3f}~{v.max():.3f}" for k, v in interped.items())
    _log(f"   proc{proc_id:06d} {vals}")

    if global_min is not None and global_max is not None:
        for k, v in interped.items():
            vmin_val, vmax_val = float(v.min()), float(v.max())
            global_min[k] = min(global_min.get(k, vmin_val), vmin_val)
            global_max[k] = max(global_max.get(k, vmax_val), vmax_val)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"proc{proc_id:06d}_reg1_"

    if anisotropic and all(k in interped for k in ['vpv', 'vph', 'vsv', 'vsh']):
        for name in ['vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho']:
            write_gll_binary(Path(str(prefix) + f"{name}.bin"), interped[name], dtype)
        if 'qmu' in interped:
            write_gll_binary(Path(str(prefix) + "qmu.bin"), interped['qmu'], dtype)
    else:
        if 'vp' in interped and 'vs' in interped:
            write_gll_binary(Path(str(prefix) + "vp.bin"), interped['vp'], dtype)
            write_gll_binary(Path(str(prefix) + "vs.bin"), interped['vs'], dtype)
        else:
            vp = voigt_vp(interped['vpv'], interped['vph'])
            vs = voigt_vs(interped['vsv'], interped['vsh'])
            write_gll_binary(Path(str(prefix) + "vp.bin"), vp, dtype)
            write_gll_binary(Path(str(prefix) + "vs.bin"), vs, dtype)
        write_gll_binary(Path(str(prefix) + "rho.bin"), interped['rho'], dtype)

    # 与参考 GLL 自动对比（默认仅 proc000000，避免日志过长）
    if reference_gll_dir is not None and (compare_all_procs or proc_id == 0) and not quiet:
        compare_vars = ['vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho'] if anisotropic else ['vp', 'vs', 'rho']
        print(f"   [参考对比 proc{proc_id:06d}]")
        for name in compare_vars:
            ref_path = reference_gll_dir / f"proc{proc_id:06d}_reg1_{name}.bin"
            if not ref_path.exists():
                print(f"      {name}: 参考文件不存在，跳过")
                continue
            try:
                ref_arr = read_gll_binary(ref_path, nspec, dtype=dtype)
                cur_arr = interped[name] if name in interped else None
                if cur_arr is None:
                    print(f"      {name}: 当前结果无该变量，跳过")
                    continue
                diff = cur_arr - ref_arr
                rmse = float(np.sqrt(np.mean(diff**2)))
                mean_abs = float(np.mean(np.abs(diff)))
                print(f"      {name}: rmse={rmse:.4f}, mean|Δ|={mean_abs:.4f}, "
                      f"cur[{cur_arr.min():.3f},{cur_arr.max():.3f}] "
                      f"ref[{ref_arr.min():.3f},{ref_arr.max():.3f}]")
            except Exception as e:
                print(f"      {name}: 对比失败 ({e})")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='NetCDF/HDF5/CSV → SPECFEM GLL 格式转换（自动从 solver_data.bin 反算 3D 参考）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # FWEA23 (NetCDF, 含扰动插值)
  python convert_nc_to_gll.py --mesh-dir DATABASES_MPI \\
      --model-nc models/FWEA23.r0.0-n4.nc --output-dir DATA/GLL --use-perturbation

  # SinoScope (HDF5, 无扰动)
  python convert_nc_to_gll.py --mesh-dir DATABASES_MPI \\
      --model-h5 models/SinoScope.h5 --output-dir DATA/GLL
""")
    # ── 必选参数 ──
    parser.add_argument('--mesh-dir', type=str, required=True,
                        help='DATABASES_MPI 目录（含 solver_data.bin）')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--model-nc', type=str, default=None,
                             help='NetCDF 模型（EMC .nc/.nc4）')
    input_group.add_argument('--model-h5', type=str, default=None,
                             help='HDF5 模型（SinoScope 等）')
    input_group.add_argument('--model-csv', type=str, default=None,
                             help='CSV 模型（lon, lat, depth, vp, vs, rho）')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='GLL 输出目录（如 DATA/GLL）')

    # ── 常用选项 ──
    parser.add_argument('--use-perturbation', action='store_true',
                        help='正确扰动转换: V_gll=V_ref_gll+interp(V_nc-V_ref_nc) '
                             '(FWEA23/EARA2024 必需；要求 NC 含 vp0/vs0 或 *_ref，'
                             '且必须能从 solver_data.bin 反算 V_ref_gll)')
    parser.add_argument('--double', action='store_true',
                        help='输出 float64（SPECFEM double precision 编译时使用）')

    # ── 高级选项（默认值已是最优配置，一般不需要改）──
    advanced = parser.add_argument_group('高级选项')
    advanced.add_argument('--no-3d-fallback', action='store_true',
                          help='禁用 3D 参考（仅绝对值模式调试用；与 --use-perturbation 互斥）')
    advanced.add_argument('--no-anisotropic', action='store_true',
                          help='输出各向同性 vp,vs,rho（默认输出各向异性 vpv,vph,vsv,vsh,eta,rho）')
    advanced.add_argument('--ref-model', type=str, default=None,
                          help='手动指定 1DREF 路径（默认自动查找）')
    advanced.add_argument('--horizontal-fill', type=str, default='blend',
                          choices=['nearest', 'ref', 'blend'],
                          help='水平域外填充策略（默认 blend；扰动模式下域外 δV→0）')
    advanced.add_argument('--blend-width-deg', type=float, default=5.0,
                          help='blend 过渡宽度（默认 5°）')
    advanced.add_argument('--interp-method', type=str, default='linear',
                          choices=['linear', 'cubic'],
                          help='插值方法（默认 linear）')
    advanced.add_argument('--reference-gll-dir', type=str, default=None,
                          help='对比参考 GLL 目录（如 model_updated/，调试用）')
    advanced.add_argument('--verbose', action='store_true',
                          help='打印每个 proc 的详细诊断（默认仅显示进度条）')
    args = parser.parse_args()

    # 从简化的 flag 派生内部变量（保持向后兼容）
    args.anisotropic = not args.no_anisotropic
    args.use_3d_fallback = not args.no_3d_fallback
    args.fill_value = 4.5
    args.apply_physical_floor = False
    args.compare_all_procs = False
    args.ref_gll_dir = None

    if args.use_perturbation and args.no_3d_fallback:
        print("❌ --use-perturbation 与 --no-3d-fallback 互斥")
        print("   正确扰动转换必须使用 mesher 的 V_ref_gll (solver_data.bin 反算)")
        return 1
    if args.use_perturbation:
        # 扰动模式强制启用 3D 参考
        args.use_3d_fallback = True

    dtype = np.float64 if args.double else np.float32

    mesh_dir = Path(args.mesh_dir)
    output_dir = Path(args.output_dir)
    model_nc = Path(args.model_nc) if args.model_nc else None
    model_h5 = Path(args.model_h5) if args.model_h5 else None
    model_csv = Path(args.model_csv) if args.model_csv else None
    reference_gll_dir = Path(args.reference_gll_dir) if args.reference_gll_dir else None

    if not mesh_dir.exists():
        print(f"❌ 网格目录不存在: {mesh_dir}")
        print("请先运行 meshfem (xgenerate_databases) 生成 DATABASES_MPI")
        return 1

    # 统计 proc 数量
    procs = sorted([int(p.name[4:10]) for p in mesh_dir.glob("proc*_reg1_solver_data.bin")])
    if not procs:
        print(f"❌ 未找到 solver_data.bin: {mesh_dir}/proc*_reg1_solver_data.bin")
        return 1

    # 加载 1DREF 参考模型
    ref_model = None
    if args.ref_model:
        ref_path = Path(args.ref_model)
    else:
        ref_path = None
        for rp in DEFAULT_REF_MODEL_PATHS:
            candidate = script_dir / rp
            if candidate.exists():
                ref_path = candidate
                break
            candidate = project_root / rp
            if candidate.exists():
                ref_path = candidate
                break

    if ref_path and ref_path.exists():
        ref_model = load_1dref_model(ref_path)
        ref_info = f"{ref_path} (深部用 1DREF 填充)"
    else:
        ref_info = "未找到（深部将用最近邻外推）"

    model_display = model_nc or model_h5 or model_csv
    use_3d_fallback = args.use_3d_fallback

    print(f"🎯 模型 → GLL 转换")
    print(f"   网格目录: {mesh_dir}")
    print(f"   模型: {model_display} ({'NC' if model_nc else 'HDF5' if model_h5 else 'CSV'})")
    if args.use_perturbation:
        print("   模式: ★ 扰动叠加  V_gll = V_ref_gll + interp(δV_nc)")
        print("   3D 参考: solver_data.bin 反算 (强制)")
    else:
        print("   模式: 绝对值插值  V_gll = interp(V_nc)")
        print(f"   3D 参考: {'solver_data.bin 域外替换' if use_3d_fallback else '1DREF (降级)'}")
    print(f"   输出: {output_dir} ({'float64' if dtype == np.float64 else 'float32'})")
    print(f"   进程数: {len(procs)}")
    if reference_gll_dir:
        print(f"   对比参考: {reference_gll_dir}")
    print()

    ok = 0
    global_min = {}
    global_max = {}
    quiet = not args.verbose

    # 预加载模型 Dataset：避免每个 proc 重复打开文件
    preloaded_ds = None
    ds_to_close = None
    if model_h5:
        preloaded_ds = load_h5_as_dataset(model_h5)
        print(f"  [HDF5] lon: {preloaded_ds.longitude.values.min():.1f}°~"
              f"{preloaded_ds.longitude.values.max():.1f}° "
              f"lat: {preloaded_ds.latitude.values.min():.1f}°~"
              f"{preloaded_ds.latitude.values.max():.1f}° "
              f"depth: {preloaded_ds.depth.values.min():.1f}~"
              f"{preloaded_ds.depth.values.max():.1f} km")
        print(f"  [HDF5] vpv=vph (P各向同性), eta=1.0, vsv/vsh 独立")
    elif model_nc:
        import xarray as xr
        print(f"  [NC] 预加载: {model_nc}")
        preloaded_ds = xr.open_dataset(model_nc)
        ds_to_close = preloaded_ds
        print(f"  [NC] lon: {float(preloaded_ds.longitude.min()):.1f}°~"
              f"{float(preloaded_ds.longitude.max()):.1f}° "
              f"lat: {float(preloaded_ds.latitude.min()):.1f}°~"
              f"{float(preloaded_ds.latitude.max()):.1f}° "
              f"depth: {float(preloaded_ds.depth.min()):.1f}~"
              f"{float(preloaded_ds.depth.max()):.1f} km")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    def _run_proc(pid: int, proc_quiet: bool) -> bool:
        return process_one_proc(
            pid, mesh_dir, output_dir, model_nc, model_csv,
            args.anisotropic, args.fill_value, dtype,
            global_min=global_min, global_max=global_max,
            ref_model=ref_model,
            apply_physical_floor=args.apply_physical_floor,
            horizontal_fill_mode=args.horizontal_fill,
            blend_width_deg=args.blend_width_deg,
            reference_gll_dir=reference_gll_dir,
            compare_all_procs=args.compare_all_procs,
            interp_method=args.interp_method,
            use_perturbation=args.use_perturbation,
            model_h5=model_h5,
            preloaded_ds=preloaded_ds,
            use_3d_fallback=use_3d_fallback,
            quiet=proc_quiet,
        )

    # 首个 proc 始终打印诊断（确认扰动公式 / 域外统计），其余走进度条
    if _run_proc(procs[0], proc_quiet=False):
        ok += 1

    remaining = procs[1:]
    if remaining:
        if quiet and tqdm is not None:
            remaining_iter = tqdm(
                remaining, desc="NC→GLL", unit="proc",
                initial=1, total=len(procs),
                dynamic_ncols=True, mininterval=0.5,
            )
        else:
            remaining_iter = remaining
        for pid in remaining_iter:
            if _run_proc(pid, proc_quiet=quiet):
                ok += 1
                if quiet and tqdm is not None and hasattr(remaining_iter, 'set_postfix'):
                    remaining_iter.set_postfix(ok=ok, refresh=False)

    if ds_to_close is not None:
        ds_to_close.close()

    print(f"\n✅ 完成: {ok}/{len(procs)} 个进程")
    if global_min:
        print(f"\n   全局值域汇总 (所有 {ok} 个 proc):")
        for k in global_min:
            print(f"      {k}: {global_min[k]:.4f} ~ {global_max[k]:.4f}")
    print(f"   输出目录: {output_dir}")
    print("\nPar_file 配置:")
    print("   MODEL = GLL")
    print(f"   PATHNAME_GLL_modeldir = {output_dir}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
