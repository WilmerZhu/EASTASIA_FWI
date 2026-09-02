"""
2_2_Model_similarity.py:
速度模型结构相似性分析模块（1D SSIM + 2D SSIM(z)）
================================================================

功能描述:
----
用 Wang et al. (2004) SSIM 从两个互补维度量化模型间结构相似性：

**1D SSIM** — 背景剖面一致性
1. 共同地理范围（所有已加载模型覆盖的交集，与 2_1 standardized 对比一致）
2. 原始绝对速度面积加权横向平均 → 1D 剖面 V(z)，插值到统一深度网格
3. 沿深度的高斯窗口求局部 SSIM(z)，整体值 = 局部 SSIM 图的均值
4. 反映 410/660 等间断面形态与绝对速度水平的一致性

**2D SSIM(z)** — 逐深度横向结构一致性
1. 每个公共深度层重采样到该模型对的最粗分辨率
2. 去掉该层横向平均（扰动场），排除 1D 背景对分数的贡献
3. 在 6° 横向高斯窗口下求 SSIM，得到 SSIM 随深度的曲线

两者分工明确：1D 量背景，2D 量同一深度的横向异常图案。
均为模型空间先验指标，不能直接用于 FWI 初始模型排序。

使用方法:
----
```python
config = ModelSimilarityConfig()
analyzer = VelocityModelSimilarity(config)
analyzer.load_models()
analyzer.compare_all_models()
analyzer.save_results()
```

参考文献:
----
- Wang, Z. et al. (2004) IEEE TIP 13(4): 600-612. SSIM
- 1D 剖面定义对齐 2_1_Model_compare.py

作者: EASTASIA-FWI Team
日期: 2026-09
版本: v19.0 (1D SSIM + 2D SSIM(z))
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import warnings
import json

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter, gaussian_filter, gaussian_filter1d
from tqdm import tqdm

try:
    import dtcwt
    HAS_DTCWT = True
except ImportError:
    HAS_DTCWT = False

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import BaseConfig

warnings.filterwarnings('ignore')


# ==================== 配置类 ====================

class ModelSimilarityConfig:
    """模型相似性分析配置类 (v18.0 — 1D SSIM 主路径)"""

    def __init__(self):

        # ============ 目标模型配置 ============
        self.models = {
            '2022_SinoScope1.0': {
                'netcdf_file': '2022_SinoScope1.0_standardized.nc',
                'processed_dir': '2022_SinoScope1.0',
                'name': 'SinoScope1.0',
                'color': '#E41A1C',
            },
            '2024_EARA2024': {
                'netcdf_file': '2024_EARA2024_standardized.nc',
                'processed_dir': '2024_EARA2024',
                'name': 'EARA2024',
                'color': '#4DAF4A',
            },
            '2024_FWEA23': {
                'netcdf_file': '2024_FWEA23_standardized.nc',
                'processed_dir': '2024_FWEA23',
                'name': 'FWEA23',
                'color': '#377EB8',
            }
        }

        # ============ 分析参数 ============
        self.analysis = {
            'target_features': ['vs', 'vp'],
            'primary_feature': 'vs',
            'compute_depth_wise': True,
            # 空间 CW-SSIM 图默认关闭；当前产品只需要整体 + 深度曲线
            'compute_cwssim': False,
            'compute_cwssim_spatial': False,
            'cwssim_spatial_depths': [20, 60, 100, 160, 200, 300, 400, 660, 800, 900, 1000],
            'depth_grid_step_km': 20.0,
            'depth_range_km': (0.0, 1000.0),
        }

        # ============ 1D SSIM 参数（原始 Vs/Vp 剖面，对齐 2_1）============
        self.ssim = {
            'mode': '1d_profile',
            # 沿深度的高斯窗口（km）。20 km 采样下 200 km → 11 点，对应 Wang 默认
            'win_km': 200.0,
            'win_size': 11,
            'gaussian_sigma_per_pixel': 1.5 / 11.0,
            'K1': 0.01,
            'K2': 0.03,
            # 直接比较原始绝对速度 1D 剖面
            'perturbation': 'none',
            'use_area_weights': True,
        }

        # ============ 2D SSIM 参数（逐深度切片的横向结构）============
        self.ssim_2d = {
            # 所有模型对共用同一分析网格与同一区域。若按模型对各取最粗分辨率，
            # 0.25° 的对会保留 1° 以下短波长内容（分歧最大的部分），而 1.0° 的对
            # 已被块平均掉，两者的分数不是同一个量，矩阵不可比。
            'resolution_mode': 'global_coarsest',   # 'pair_coarsest' 为旧行为
            'region_mode': 'all_models',            # 'pair' 为旧行为
            # 高斯窗口的物理尺度。σ=2.0° ≈ 220 km，与被比模型的横向分辨率同量级；
            # 更小的窗口是在噪声尺度上比结构，会系统性压低分数。
            'sigma_deg': 2.0,
            'truncate_sigma': 3.0,
            'K1': 0.01,
            'K2': 0.03,
            # 'relative': δlnV = (V - V̄_layer)/V̄_layer，去掉 1D 背景且无量纲；
            # 'absolute': km/s 残差，深部背景速度高会把对比度项抬高（旧行为）
            'perturbation': 'relative',
            # 逐层用两模型合并 RMS 归一，再按 ±clip_sigma 映射到非负区间。
            # 等价于把两张图用完全相同的对称色标出图后再比较，
            # 消除"平移量任取"的随意性，且各深度处于同一尺度。
            'normalize': 'pooled_rms',
            'clip_sigma': 3.0,
            'use_area_weights': True,
        }

        # ============ CW-SSIM 参数（可选对照，默认不计算）============
        self.cwssim = {
            # 分辨率策略: 'pair_coarsest' — 按模型对取最粗
            #   SinoScope vs * → 1.0°;  EARA vs FWEA → 0.25°
            'resolution_mode': 'pair_coarsest',
            'nlevels': 4,
            'K': 1e-6,
            # Huang: 6°×6° 空间滑窗
            'local_win_deg': 6.0,
            'local_win_size': None,
            'level_weights': [0.05, 0.15, 0.35, 0.45],
            'depth_adaptive_weights': {
                100: [0.15, 0.25, 0.35, 0.25],
                410: [0.08, 0.18, 0.37, 0.37],
                660: [0.05, 0.15, 0.35, 0.45],
                1000: [0.03, 0.12, 0.35, 0.50],
            },
            'use_depth_adaptive_weights': True,
            'use_gaussian_window': False,
            # Huang 原做法
            'normalize_method': 'minmax',
            'percentile_range': (2.0, 98.0),
            # Huang 空间滑窗主路径; 'local_filter' 仅作快速备选
            'spatial_method': 'huang_patch',
        }

        # ============ 掩膜与采样 ============
        self.masking = {
            # 仅当前模型对有效交集（Huang）
            'use_all_model_common_mask': False,
            'fill_method': 'nearest',  # 避免 nanmean 假高分
            'boundary_buffer_px': 0,
            'downsample_method': 'block_mean',
        }

        # ============ 零假设基线（附录，默认关）============
        self.null_test = {
            'enabled': False,
            'n_realizations': 20,
            'random_seed': 42,
        }

        # ============ 伴随指标（附录，默认关）============
        self.companion = {
            'compute_signed_correlation': False,
            'compute_sign_agreement': False,
            'sign_threshold_percentile': 50.0,
        }

        # ============ 可视化配置 ============
        self.visualization = {
            'dpi': 300,
            'figsize_depth_curve': (12, 10),
            'figsize_heatmap': (10, 9),
            'figsize_cwssim_spatial': (16, 10),
            'figsize_overview': (20, 16),
            'heatmap': {
                'ssim_vmin': 0.0,
                'ssim_vmax': 1.0,
                'cwssim_vmin': 0.0,
                'cwssim_vmax': 1.0,
                'cmap': 'RdYlGn',
                'annot_fontsize': 28,
                'label_fontsize': 18,
                'title_fontsize': 20,
                'cbar_fontsize': 14,
                'cell_linewidth': 2,
                'cell_linecolor': 'white',
            },
            'cmap_ssim': 'RdYlGn',
            'cmap_cwssim': 'RdYlGn',
            'discontinuities': [
                ('Moho', 40),
                ('LAB', 100),
                ('410km', 410),
                ('660km', 660),
            ],
            'add_coastlines': True,
            'coastline_color': 'black',
            'coastline_linewidth': 0.8,
        }

        # ============ 输出参数 ============
        self.output = {
            # 项目规范: 同时保存 jpg + pdf，均 300 dpi
            'save_formats': ['jpg'],
            'save_dpi': 300,
            'figure_prefix': '2-2_',
        }

        # ============ 日志配置 ============
        self.logging = {
            'level': 'INFO',
            'console_output': True,
        }


# ==================== NetCDF 模型加载类 ====================

class VelocityModelNetCDF:
    """基于 NetCDF 的速度模型数据类（保持原生网格，无插值）"""

    def __init__(self, netcdf_path: Path, model_key: str,
                 config: ModelSimilarityConfig, logger: logging.Logger):
        self.path = netcdf_path
        self.model_key = model_key
        self.name = config.models[model_key]['name']
        self.color = config.models[model_key]['color']
        self.logger = logger
        self.ds: Optional[xr.Dataset] = None

        self._lon: np.ndarray = np.array([])
        self._lat: np.ndarray = np.array([])
        self._depth: np.ndarray = np.array([])

        self._load_netcdf()

    def _load_netcdf(self):
        """加载 NetCDF 数据"""
        self.logger.info(f"📦 加载 NetCDF: {self.name} ({self.path.name})")
        self.ds = xr.open_dataset(self.path)

        self._lon = np.sort(self.ds['longitude'].values)
        self._lat = np.sort(self.ds['latitude'].values)
        self._depth = np.sort(self.ds['depth'].values)

        self.logger.info(
            f"  ✅ 维度 ({len(self._lat)}, {len(self._lon)}, {len(self._depth)})  "
            f"纬度 [{self._lat.min():.2f}°, {self._lat.max():.2f}°]  "
            f"经度 [{self._lon.min():.2f}°, {self._lon.max():.2f}°]  "
            f"深度 [{self._depth.min():.0f}, {self._depth.max():.0f}] km"
        )

    @property
    def lon(self) -> np.ndarray:
        return self._lon

    @property
    def lat(self) -> np.ndarray:
        return self._lat

    @property
    def depth(self) -> np.ndarray:
        return self._depth

    def has_parameter(self, param: str) -> bool:
        if self.ds is None:
            return False
        return param in self.ds

    def get_parameter(self, param: str) -> np.ndarray:
        """获取 3D 数组，统一为 (lon, lat, depth) 维度顺序"""
        if self.ds is None or param not in self.ds:
            raise ValueError(f"参数 {param} 不存在于模型 {self.name}")

        data = self.ds[param].values
        n_lon, n_lat, n_depth = len(self._lon), len(self._lat), len(self._depth)

        if data.shape == (n_lat, n_lon, n_depth):
            return np.transpose(data, (1, 0, 2))
        elif data.shape == (n_lon, n_lat, n_depth):
            return data
        else:
            raise ValueError(
                f"维度无法识别: {data.shape}, 期望 ({n_lat},{n_lon},{n_depth}) 或 ({n_lon},{n_lat},{n_depth})"
            )

    def get_depth_slice(self, param: str, depth_km: float) -> Tuple[np.ndarray, float]:
        """获取指定深度的 2D 切片 (lon, lat)，返回 (slice, actual_depth)"""
        d_idx = int(np.argmin(np.abs(self._depth - depth_km)))
        data = self.get_parameter(param)
        return data[:, :, d_idx], float(self._depth[d_idx])

    def estimate_resolution(self) -> Dict[str, float]:
        """估算模型空间分辨率 (度)"""
        lon_res = float(np.median(np.diff(self._lon))) if len(self._lon) > 1 else 1.0
        lat_res = float(np.median(np.diff(self._lat))) if len(self._lat) > 1 else 1.0
        return {'lon': lon_res, 'lat': lat_res}

    def close(self):
        """显式关闭 NetCDF 文件句柄"""
        if self.ds is not None:
            try:
                self.ds.close()
            except Exception:
                # 解释器退出期 netCDF4 C 扩展可能已被回收，此时忽略
                pass
            finally:
                self.ds = None

    def __enter__(self) -> 'VelocityModelNetCDF':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # 注意: 刻意不实现 __del__。在解释器关闭阶段调用 xr.Dataset.close()
    # 会触及已被回收的 netCDF4 C 扩展，实测导致 TypeError 乃至 SIGSEGV(139)，
    # 并因 stdout 缓冲未刷新而丢失全部日志。改由 close()/上下文管理器显式释放。


# ==================== 工具函数 ====================

def find_common_depth_indices(
    depths1: np.ndarray, depths2: np.ndarray, tolerance_km: float = 5.0
) -> List[Tuple[int, int, float]]:
    """
    找到两个深度数组中匹配的深度层索引。

    Returns:
        [(idx1, idx2, depth_km), ...] 匹配的深度层列表
    """
    matched = []
    for i, d1 in enumerate(depths1):
        j = int(np.argmin(np.abs(depths2 - d1)))
        if abs(depths2[j] - d1) <= tolerance_km:
            matched.append((i, j, float(d1)))
    return matched


def nearest_neighbor_sample(
    model_lon: np.ndarray, model_lat: np.ndarray,
    data_2d: np.ndarray,
    target_lons: np.ndarray, target_lats: np.ndarray
) -> np.ndarray:
    """
    在目标坐标上使用最近邻取值采样 2D 数据（非插值）。

    Args:
        model_lon, model_lat: 模型原生坐标
        data_2d: 形状 (n_lon, n_lat)
        target_lons, target_lats: 目标坐标

    Returns:
        形状 (n_target_lon, n_target_lat) 的数组
    """
    interp = RegularGridInterpolator(
        (model_lon, model_lat), data_2d,
        method='nearest',
        bounds_error=False,
        fill_value=np.nan,
    )
    tlon, tlat = np.meshgrid(target_lons, target_lats, indexing='ij')
    points = np.column_stack([tlon.ravel(), tlat.ravel()])
    return interp(points).reshape(len(target_lons), len(target_lats))


def block_mean_sample(
    model_lon: np.ndarray, model_lat: np.ndarray,
    data_2d: np.ndarray,
    target_lons: np.ndarray, target_lats: np.ndarray
) -> np.ndarray:
    """
    面积加权块平均重采样（抗混叠），用于将细网格模型降采样到粗分析网格。

    最近邻抽样会把短波长内容折叠成虚假长波长（混叠），污染小波粗尺度层级。
    本函数对落入每个目标格元的所有源网格点做 cos(lat) 加权平均。
    若某目标格元内无源网格点（源比目标粗），退化为最近邻取值。

    Args:
        model_lon, model_lat: 模型原生坐标（升序）
        data_2d: 形状 (n_lon, n_lat)，可含 NaN
        target_lons, target_lats: 目标坐标（升序，等间距）

    Returns:
        形状 (n_target_lon, n_target_lat) 的数组；全 NaN 格元返回 NaN
    """
    d_lon = float(np.median(np.diff(target_lons))) if len(target_lons) > 1 else 1.0
    d_lat = float(np.median(np.diff(target_lats))) if len(target_lats) > 1 else 1.0

    lon_edges = np.concatenate([target_lons - d_lon / 2.0, [target_lons[-1] + d_lon / 2.0]])
    lat_edges = np.concatenate([target_lats - d_lat / 2.0, [target_lats[-1] + d_lat / 2.0]])

    n_i, n_j = len(target_lons), len(target_lats)

    # 每个源点归属的目标格元索引；落在目标网格外的源点直接剔除
    i_src = np.searchsorted(lon_edges, model_lon, side='right') - 1
    j_src = np.searchsorted(lat_edges, model_lat, side='right') - 1
    keep_i = (i_src >= 0) & (i_src < n_i)
    keep_j = (j_src >= 0) & (j_src < n_j)

    ii = i_src[keep_i]
    jj = j_src[keep_j]
    sub = np.asarray(data_2d, dtype=np.float64)[np.ix_(keep_i, keep_j)]
    w_lat = np.cos(np.radians(np.asarray(model_lat, dtype=np.float64)[keep_j]))

    valid = np.isfinite(sub)
    weight_2d = np.broadcast_to(w_lat[np.newaxis, :], sub.shape)
    contrib = np.where(valid, sub * weight_2d, 0.0)
    weight = np.where(valid, weight_2d, 0.0)

    num = np.zeros((n_i, n_j), dtype=np.float64)
    den = np.zeros((n_i, n_j), dtype=np.float64)
    idx_i = np.broadcast_to(ii[:, np.newaxis], sub.shape)
    idx_j = np.broadcast_to(jj[np.newaxis, :], sub.shape)
    np.add.at(num, (idx_i.ravel(), idx_j.ravel()), contrib.ravel())
    np.add.at(den, (idx_i.ravel(), idx_j.ravel()), weight.ravel())

    out = np.full((n_i, n_j), np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = num[ok] / den[ok]

    # 空格元（源比目标粗）用最近邻补
    if not ok.all():
        nn = nearest_neighbor_sample(model_lon, model_lat, data_2d,
                                     target_lons, target_lats)
        out[~ok] = nn[~ok]

    return out


def fill_invalid_nearest(data_2d: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    用最近邻外延填充无效区，避免常数填充在掩膜边界制造人工阶跃。

    高通小波系数对阶跃极其敏感；而常数填充还会让两图在该区同时趋于常数，
    使 num/den → K/K = 1，产生虚假的完美相似。最近邻外延保持场的连续性。

    Args:
        data_2d: 待填充数组
        valid: 有效掩膜（True 为有效）

    Returns:
        填充后的数组（无 NaN）；若全无效则返回全 0
    """
    from scipy.ndimage import distance_transform_edt

    if valid.all():
        return np.asarray(data_2d, dtype=np.float64)
    if not valid.any():
        return np.zeros_like(data_2d, dtype=np.float64)

    # distance_transform_edt 对 valid==False 的点给出最近 valid 点的索引
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = np.asarray(data_2d, dtype=np.float64).copy()
    filled[~valid] = filled[tuple(i[~valid] for i in idx)]
    return filled


def erode_mask(valid: np.ndarray, buffer_px: int) -> np.ndarray:
    """
    对有效掩膜做腐蚀，排除距无效区 buffer_px 格以内的缓冲带。

    小波系数在掩膜边界附近受填充值污染，污染范围约等于最粗层级的支撑尺度。

    Args:
        valid: 有效掩膜
        buffer_px: 缓冲宽度（网格数）；<= 0 时原样返回

    Returns:
        腐蚀后的掩膜
    """
    if buffer_px <= 0 or valid.all():
        return valid
    from scipy.ndimage import binary_erosion
    struct = np.ones((2 * buffer_px + 1, 2 * buffer_px + 1), dtype=bool)
    return binary_erosion(valid, structure=struct, border_value=0)


def latitude_weights(lats: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    构造 cos(lat) 面积权重，形状为 (n_lon, n_lat)。

    等经纬网格上格元物理面积正比于 cos(lat)。10-55°N 权重差 1.72 倍，
    不加权会系统性高估高纬度区域的贡献。

    Args:
        lats: 纬度数组
        shape: 目标形状 (n_lon, n_lat)

    Returns:
        权重数组，形状 (n_lon, n_lat)
    """
    w = np.cos(np.radians(np.asarray(lats, dtype=np.float64)))
    return np.broadcast_to(w[np.newaxis, :], shape).copy()


def weighted_pearson(
    x: np.ndarray, y: np.ndarray,
    weights: np.ndarray, valid: np.ndarray
) -> float:
    """
    面积加权的带符号 Pearson 相关系数。

    这是 CW-SSIM 极性盲区的直接补充: CW-SSIM 对结构反相给出 1.0，
    而本指标在反相时给出接近 -1 的值。

    Args:
        x, y: 两个场（同形状）
        weights: 面积权重
        valid: 有效掩膜

    Returns:
        r ∈ [-1, 1]；有效点不足时返回 nan
    """
    m = valid & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float('nan')

    w = weights[m]
    xv, yv = x[m], y[m]
    w_sum = w.sum()

    xm = xv - (w * xv).sum() / w_sum
    ym = yv - (w * yv).sum() / w_sum

    cov = (w * xm * ym).sum()
    var_x = (w * xm ** 2).sum()
    var_y = (w * ym ** 2).sum()

    if var_x <= 0 or var_y <= 0:
        return float('nan')
    return float(cov / np.sqrt(var_x * var_y))


def sign_agreement(
    x: np.ndarray, y: np.ndarray,
    weights: np.ndarray, valid: np.ndarray,
    threshold_percentile: float = 50.0
) -> float:
    """
    符号一致率: 在两个场的异常幅度均超过阈值的区域内，符号相同的面积占比。

    阈值按分位数自适应（参考 Shephard et al. 2017 的 depth-dependent threshold），
    而非固定绝对值——固定阈值在不同深度会截取到差异极大的面积比例。
    近零区被排除，因为"两个模型都接近区域均值"不构成有意义的共识。

    Args:
        x, y: 两个场（应为去均值后的扰动）
        weights: 面积权重
        valid: 有效掩膜
        threshold_percentile: 幅度阈值分位数

    Returns:
        一致率 ∈ [0, 1]；有效点不足时返回 nan
    """
    m = valid & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float('nan')

    w = weights[m]
    xv, yv = x[m], y[m]
    w_sum = w.sum()
    xm = xv - (w * xv).sum() / w_sum
    ym = yv - (w * yv).sum() / w_sum

    tx = np.percentile(np.abs(xm), threshold_percentile)
    ty = np.percentile(np.abs(ym), threshold_percentile)
    strong = (np.abs(xm) >= tx) & (np.abs(ym) >= ty)

    if strong.sum() < 10:
        return float('nan')

    same = np.sign(xm[strong]) == np.sign(ym[strong])
    return float((w[strong] * same).sum() / w[strong].sum())


def phase_randomize(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    相位随机化 surrogate: 保留功率谱、随机化相位。

    产生"功率谱与原场相同但空间结构无关"的场，作为相似性指标的零假设基线
    (Theiler et al., 1992)。比白噪声更严格——白噪声的谱结构完全不同，
    对比它得到的高分值毫无意义。

    Args:
        img: 输入 2D 场（不含 NaN）
        rng: 随机数生成器

    Returns:
        相位随机化后的场
    """
    mean_val = float(np.mean(img))
    spec = np.fft.fft2(img - mean_val)
    phase = rng.uniform(0, 2 * np.pi, spec.shape)
    return np.real(np.fft.ifft2(np.abs(spec) * np.exp(1j * phase))) + mean_val


def remove_layer_mean(
    data: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    去掉面积加权横向平均，得到该深度的相对扰动。

    绝对速度的 SSIM 会被 1D 背景主导（两模型都像 PREM，分数虚高）。
    对每个模型各自去该层均值，等价于相对横向平均的 δV 对比。
    """
    out = np.array(data, dtype=np.float64, copy=True)
    m = valid & np.isfinite(data)
    if not m.any():
        return out
    w = weights[m]
    wsum = float(w.sum())
    if wsum <= 0.0:
        return out
    mean_val = float((w * data[m]).sum() / wsum)
    out[m] = data[m] - mean_val
    return out


class SSIMCalculator:
    """
    Wang et al. (2004) 结构相似性指数 (SSIM)。

    在 2D 深度切片上计算局部 SSIM 图，再对有效区做面积加权平均。
    结构项为局部协方差：极性相反时分数下降，这是与 CW-SSIM 的关键差别。
    """

    def __init__(
        self,
        k1: float = 0.01,
        k2: float = 0.03,
        gaussian_sigma_per_pixel: float = 1.5 / 11.0,
    ) -> None:
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.gaussian_sigma_per_pixel = float(gaussian_sigma_per_pixel)

    def compute(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        valid: np.ndarray,
        win_size: int,
        weights: Optional[np.ndarray] = None,
        stat_mask: Optional[np.ndarray] = None,
        data_range: Optional[float] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        计算两张 2D 场的 SSIM。

        Args:
            img1, img2: 同形状的 2D 场（建议已去横向平均）
            valid: 有效掩膜
            win_size: 高斯窗口像素数（奇数）
            weights: 面积权重；None 时等权
            stat_mask: 参与全局平均的掩膜
            data_range: 动态范围 L；None 时用有效区 2–98 百分位跨度

        Returns:
            (全局 SSIM, 局部 SSIM 图)；无效区为 NaN
        """
        img1 = np.asarray(img1, dtype=np.float64)
        img2 = np.asarray(img2, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)

        if img1.shape != img2.shape:
            raise ValueError(f"输入形状不一致: {img1.shape} vs {img2.shape}")
        if not valid.any():
            return float('nan'), np.full(img1.shape, np.nan)

        x = fill_invalid_nearest(img1, valid)
        y = fill_invalid_nearest(img2, valid)

        win_size = int(win_size)
        if win_size % 2 == 0:
            win_size += 1
        min_dim = int(min(img1.shape))
        if min_dim < 3:
            return float('nan'), np.full(img1.shape, np.nan)
        if win_size > min_dim:
            win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
            win_size = max(3, win_size)

        sigma = max(0.5, self.gaussian_sigma_per_pixel * win_size)

        # Wang SSIM 按非负图像设计。扰动场若零均值，反相时亮度项与结构项
        # 同号为负，乘积变正，会把反相关误判为相似。两场共用同一平移到
        # 正值区间，极性只留在结构项（协方差）里。
        vals = np.concatenate([img1[valid], img2[valid]])
        vals = vals[np.isfinite(vals)]
        if vals.size < 10:
            return float('nan'), np.full(img1.shape, np.nan)
        lo, hi = np.percentile(vals, [2.0, 98.0])
        span = float(hi - lo)
        if span < 1e-12:
            span = float(np.ptp(vals))
        if span < 1e-12:
            return 1.0, np.ones(img1.shape, dtype=np.float64)

        if data_range is None:
            data_range = span
        x = x - float(lo)
        y = y - float(lo)

        c1 = (self.k1 * data_range) ** 2
        c2 = (self.k2 * data_range) ** 2

        mu1 = gaussian_filter(x, sigma=sigma, mode='reflect')
        mu2 = gaussian_filter(y, sigma=sigma, mode='reflect')
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = gaussian_filter(x * x, sigma=sigma, mode='reflect') - mu1_sq
        sigma2_sq = gaussian_filter(y * y, sigma=sigma, mode='reflect') - mu2_sq
        sigma12 = gaussian_filter(x * y, sigma=sigma, mode='reflect') - mu12
        sigma1_sq = np.maximum(sigma1_sq, 0.0)
        sigma2_sq = np.maximum(sigma2_sq, 0.0)

        num = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

        if stat_mask is None:
            stat_mask = valid
        m = stat_mask & np.isfinite(ssim_map)
        if not m.any():
            global_val = float('nan')
        elif weights is None:
            global_val = float(np.mean(ssim_map[m]))
        else:
            w = weights[m]
            wsum = float(w.sum())
            global_val = (
                float((w * ssim_map[m]).sum() / wsum) if wsum > 0 else float('nan')
            )

        ssim_map = np.where(valid, ssim_map, np.nan)
        return global_val, ssim_map

    def compute_1d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        win_size: int,
        data_range: Optional[float] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        两条 1D 剖面的 SSIM（Wang 2004，沿深度的高斯局部统计）。

        用于原始绝对速度 V(z)。不做扰动化平移：Vs/Vp 已为正值，
        亮度项接近 1，分数由对比度与结构项（间断面形态是否对齐）决定。

        Args:
            x, y: 同长度 1D 剖面（km/s）
            win_size: 高斯窗口采样点数（奇数）
            data_range: 动态范围 L；None 时用两剖面 2–98 百分位跨度

        Returns:
            (整体 SSIM, 局部 SSIM(z))
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        if x.shape != y.shape:
            raise ValueError(f"1D 剖面长度不一致: {x.shape} vs {y.shape}")
        n = int(x.size)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 5:
            return float('nan'), np.full(n, np.nan)

        x_f = x.copy()
        y_f = y.copy()
        if not valid.all():
            idx = np.arange(n)
            x_f[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
            y_f[~valid] = np.interp(idx[~valid], idx[valid], y[valid])

        win_size = int(win_size)
        if win_size % 2 == 0:
            win_size += 1
        win_size = max(3, min(win_size, n if n % 2 == 1 else n - 1))
        sigma = max(0.5, self.gaussian_sigma_per_pixel * win_size)

        vals = np.concatenate([x[valid], y[valid]])
        if data_range is None:
            lo, hi = np.percentile(vals, [2.0, 98.0])
            data_range = float(hi - lo)
            if data_range < 1e-12:
                data_range = float(np.ptp(vals)) if np.ptp(vals) > 0 else 1.0

        c1 = (self.k1 * data_range) ** 2
        c2 = (self.k2 * data_range) ** 2

        mu1 = gaussian_filter1d(x_f, sigma=sigma, mode='reflect')
        mu2 = gaussian_filter1d(y_f, sigma=sigma, mode='reflect')
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = gaussian_filter1d(x_f * x_f, sigma=sigma, mode='reflect') - mu1_sq
        sigma2_sq = gaussian_filter1d(y_f * y_f, sigma=sigma, mode='reflect') - mu2_sq
        sigma12 = gaussian_filter1d(x_f * y_f, sigma=sigma, mode='reflect') - mu12
        sigma1_sq = np.maximum(sigma1_sq, 0.0)
        sigma2_sq = np.maximum(sigma2_sq, 0.0)

        num = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        ssim_map = np.where(valid, ssim_map, np.nan)
        global_val = float(np.nanmean(ssim_map)) if np.isfinite(ssim_map).any() else float('nan')
        return global_val, ssim_map


def level_wavelength_bands(
    resolution_deg: float, nlevels: int
) -> List[Tuple[float, float]]:
    """
    计算各小波层级对应的物理波长区间（km）。

    DT-CWT 层级 ℓ 的高通子带对应波长约 [2^ℓ·Δ, 2^(ℓ+1)·Δ]，
    Δ 为分析网格间距。此映射是解释 level_weights 的前提。

    Args:
        resolution_deg: 分析分辨率（度）
        nlevels: 分解层数

    Returns:
        [(λ_min, λ_max), ...] 长度为 nlevels
    """
    delta_km = resolution_deg * 111.195
    return [(2 ** lev * delta_km, 2 ** (lev + 1) * delta_km)
            for lev in range(1, nlevels + 1)]


# ==================== CW-SSIM 计算器 ====================

class CWSSIMCalculator:
    """
    Complex Wavelet Structural Similarity (CW-SSIM) 计算器

    主路径使用 dtcwt (Dual-Tree Complex Wavelet Transform)；
    备选路径使用 Gabor 滤波器组 + scipy，无额外依赖。

    参考:
    - Wang & Simoncelli (2005), ICASSP
    - Sampat et al. (2009), IEEE TIP
    - Huang et al. (2024), Earthquake Science 37(6): 514-528
    """

    def __init__(self, nlevels: int = 4, K: float = 1e-6,
                 level_weights: Optional[List[float]] = None,
                 depth_adaptive_weights: Optional[Dict[str, Any]] = None,
                 use_depth_adaptive_weights: bool = False,
                 normalize_method: str = 'minmax',
                 percentile_range: Tuple[float, float] = (2.0, 98.0),
                 use_gaussian_window: bool = False):
        self.nlevels = nlevels
        self.K = K
        self._use_dtcwt = HAS_DTCWT
        self._level_weights = level_weights
        self._depth_adaptive_weights = depth_adaptive_weights or {}
        self._use_depth_adaptive_weights = use_depth_adaptive_weights
        self._normalize_method = normalize_method
        self._percentile_range = percentile_range
        self._use_gaussian_window = use_gaussian_window

    def _get_level_weights(self, depth_km: Optional[float] = None) -> np.ndarray:
        """获取归一化层级权重；若启用深度自适应且提供 depth_km，则按深度区间选用"""
        if (self._use_depth_adaptive_weights and depth_km is not None
                and self._depth_adaptive_weights):
            sorted_limits = sorted(
                self._depth_adaptive_weights.keys(),
                key=lambda k: float(k)
            )
            for dmax in sorted_limits:
                if float(depth_km) <= float(dmax):
                    w = np.asarray(
                        self._depth_adaptive_weights[dmax], dtype=np.float64
                    )
                    if len(w) == self.nlevels:
                        return w / w.sum()
                    break
        if self._level_weights is not None and len(self._level_weights) == self.nlevels:
            w = np.asarray(self._level_weights, dtype=np.float64)
        else:
            w = np.ones(self.nlevels)
        return w / w.sum()

    def _local_filter(self, arr: np.ndarray, size: int) -> np.ndarray:
        """局部平滑：均匀窗口或高斯窗口"""
        if self._use_gaussian_window:
            sigma = max(0.5, size / 3.0)
            return gaussian_filter(arr, sigma=sigma, mode='nearest')
        return uniform_filter(arr, size=size, mode='nearest')

    # ---------- 全局 CW-SSIM ----------

    def compute_global(
        self, img1: np.ndarray, img2: np.ndarray,
        depth_km: Optional[float] = None
    ) -> float:
        """
        计算两张 2D 图像的全局 CW-SSIM 值。

        Args:
            depth_km: 深度 (km)，用于深度自适应尺度权重

        Returns:
            CW-SSIM ∈ [0, 1]，1 = 完全一致
        """
        img1 = np.asarray(img1, dtype=np.float64)
        img2 = np.asarray(img2, dtype=np.float64)

        if img1.shape != img2.shape:
            raise ValueError(f"输入形状不一致: {img1.shape} vs {img2.shape}")

        if np.all(np.isnan(img1)) or np.all(np.isnan(img2)):
            return 0.0

        # 无效区用最近邻外延（非常数 nanmean），避免缺失区虚假高分
        valid = np.isfinite(img1) & np.isfinite(img2)
        if valid.any() and not valid.all():
            img1 = fill_invalid_nearest(img1, valid)
            img2 = fill_invalid_nearest(img2, valid)
        else:
            img1 = np.nan_to_num(img1, nan=0.0)
            img2 = np.nan_to_num(img2, nan=0.0)

        img1, img2 = self._normalize_pair(img1, img2, valid=valid if valid.any() else None)

        min_dim = min(img1.shape)
        use_dtcwt = self._use_dtcwt and min_dim >= (2 ** self.nlevels + 1)
        if not use_dtcwt and self._use_dtcwt:
            # 调用方应已告警；此处静默回退以保持可运行
            pass

        if use_dtcwt:
            return self._cwssim_dtcwt(img1, img2, depth_km)
        return self._cwssim_gabor(img1, img2, depth_km)

    # ---------- 局部 CW-SSIM 空间分布图 ----------

    def compute_local(
        self, img1: np.ndarray, img2: np.ndarray, win_size: int = 7,
        depth_km: Optional[float] = None,
        valid: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        stat_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        计算局部 CW-SSIM 图 + 掩膜感知、面积加权的全局均值。

        这是本模块的主计算路径。相比 compute_global，它保留了 CW-SSIM 的
        平移不变性（该特性来自局部窗口内取模，全局求和会使其消失）。

        Args:
            img1, img2: 两个 2D 场，形状必须一致
            win_size: 系数域局部平均窗口（网格数）
            depth_km: 深度 (km)，用于深度自适应尺度权重
            valid: 有效掩膜；无效区用最近邻外延填充（而非常数），避免人工阶跃
            weights: 面积权重（cos(lat)）；None 时等权
            stat_mask: 参与全局均值统计的掩膜（通常为 valid 腐蚀掉边界缓冲带后）；
                       None 时退回 valid

        Returns:
            (global_cwssim, local_map)；local_map 在 valid 之外为 NaN
        """
        img1 = np.asarray(img1, dtype=np.float64)
        img2 = np.asarray(img2, dtype=np.float64)

        if img1.shape != img2.shape:
            raise ValueError(f"输入形状不一致: {img1.shape} vs {img2.shape}")

        if valid is None:
            valid = np.isfinite(img1) & np.isfinite(img2)
        else:
            valid = valid & np.isfinite(img1) & np.isfinite(img2)

        if not valid.any():
            return float('nan'), np.full(img1.shape, np.nan)

        # 最近邻外延填充，避免常数填充导致缺失区虚假高相似
        img1 = fill_invalid_nearest(img1, valid)
        img2 = fill_invalid_nearest(img2, valid)

        # 归一化只依据有效区的统计量
        img1, img2 = self._normalize_pair(img1, img2, valid=valid)

        min_dim = min(img1.shape)
        use_dtcwt = self._use_dtcwt and min_dim >= (2 ** self.nlevels + 1)

        if use_dtcwt:
            _, local_map = self._cwssim_local_dtcwt(img1, img2, win_size, depth_km)
        else:
            _, local_map = self._cwssim_local_gabor(img1, img2, win_size, depth_km)

        if stat_mask is None:
            stat_mask = valid

        m = stat_mask & np.isfinite(local_map)
        if not m.any():
            global_val = float('nan')
        elif weights is None:
            global_val = float(np.mean(local_map[m]))
        else:
            w = weights[m]
            global_val = float((w * local_map[m]).sum() / w.sum())

        # 展示与后续统计一律排除无效区
        local_map = np.where(valid, local_map, np.nan)
        return global_val, local_map

    def compute_local_by_level(
        self, img1: np.ndarray, img2: np.ndarray, win_size: int,
        valid: np.ndarray, weights: np.ndarray, stat_mask: np.ndarray,
    ) -> List[float]:
        """
        分层级返回未加权的 CW-SSIM，用于判断 level_weights 的影响。

        深度自适应权重是主观设定的；只看加权后的单一数值无法判断结论
        对权重有多敏感。逐层级值让这一敏感性可见。

        Args:
            img1, img2: 两个 2D 场
            win_size: 局部平均窗口
            valid: 有效掩膜
            weights: 面积权重
            stat_mask: 统计掩膜

        Returns:
            长度为 nlevels 的列表，每项为该层级的面积加权 CW-SSIM
        """
        img1 = fill_invalid_nearest(np.asarray(img1, dtype=np.float64), valid)
        img2 = fill_invalid_nearest(np.asarray(img2, dtype=np.float64), valid)
        img1, img2 = self._normalize_pair(img1, img2, valid=valid)

        min_dim = min(img1.shape)
        if not (self._use_dtcwt and min_dim >= (2 ** self.nlevels + 1)):
            return [float('nan')] * self.nlevels

        maps = self._level_maps_dtcwt(img1, img2, win_size)
        out = []
        for lev_map in maps:
            m = stat_mask & np.isfinite(lev_map)
            if not m.any():
                out.append(float('nan'))
            else:
                w = weights[m]
                out.append(float((w * lev_map[m]).sum() / w.sum()))
        return out

    def null_distribution(
        self, img1: np.ndarray, img2: np.ndarray, win_size: int,
        depth_km: Optional[float], valid: np.ndarray,
        weights: np.ndarray, stat_mask: np.ndarray,
        n_realizations: int, rng: np.random.Generator,
    ) -> Dict[str, float]:
        """
        相位随机化零假设基线。

        对 img2 反复做相位随机化（保留功率谱、打乱相位），得到"谱相同但结构
        无关"的 CW-SSIM 分布。没有这条基线，CW-SSIM 的数值没有标尺——
        实测 660 km 处 0.136 看似"不相似"，但相对 null 的 z 值仍达 8.4。

        Args:
            img1, img2: 两个 2D 场
            win_size: 局部平均窗口
            depth_km: 深度，用于层级权重
            valid: 有效掩膜
            weights: 面积权重
            stat_mask: 统计掩膜
            n_realizations: 重复次数
            rng: 随机数生成器

        Returns:
            {'null_mean', 'null_std', 'null_p95'}
        """
        base2 = fill_invalid_nearest(np.asarray(img2, dtype=np.float64), valid)
        vals = []
        for _ in range(max(1, n_realizations)):
            surrogate = phase_randomize(base2, rng)
            val, _ = self.compute_local(
                img1, surrogate, win_size=win_size, depth_km=depth_km,
                valid=valid, weights=weights, stat_mask=stat_mask,
            )
            if np.isfinite(val):
                vals.append(val)

        if not vals:
            return {'null_mean': float('nan'), 'null_std': float('nan'),
                    'null_p95': float('nan')}
        arr = np.asarray(vals)
        return {
            'null_mean': float(arr.mean()),
            'null_std': float(arr.std()),
            'null_p95': float(np.percentile(arr, 95)),
        }

    # ---------- 归一化 ----------

    def _normalize_pair(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        valid: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对两张图像做独立归一化到 [0, 1]。

        - minmax: Huang et al. (2024) 做法，消除绝对速度差异（对离群值极敏感）
        - percentile: 稳健归一化，抗异常值（推荐）

        统计量仅在 valid 区域上估计，避免填充值污染归一化尺度。
        """
        p_low, p_high = self._percentile_range

        def _stats_region(img: np.ndarray) -> np.ndarray:
            if valid is None:
                return img[np.isfinite(img)]
            return img[valid & np.isfinite(img)]

        def _norm_minmax(img: np.ndarray) -> np.ndarray:
            region = _stats_region(img)
            if region.size == 0:
                return np.zeros_like(img)
            vmin, vmax = float(region.min()), float(region.max())
            drange = vmax - vmin
            if drange < 1e-12:
                return np.zeros_like(img)
            return (img - vmin) / drange

        def _norm_percentile(img: np.ndarray) -> np.ndarray:
            region = _stats_region(img)
            if region.size == 0:
                return np.zeros_like(img)
            vmin = float(np.percentile(region, p_low))
            vmax = float(np.percentile(region, p_high))
            drange = vmax - vmin
            if drange < 1e-12:
                return np.zeros_like(img)
            out = (img - vmin) / drange
            return np.clip(out, 0.0, 1.0)

        norm_fn = _norm_percentile if self._normalize_method == 'percentile' else _norm_minmax
        return norm_fn(img1), norm_fn(img2)

    # ---------- dtcwt 实现 ----------

    @staticmethod
    def _pad_to_even(img: np.ndarray) -> np.ndarray:
        """将图像填充到偶数尺寸，避免 dtcwt 内部填充产生大量警告"""
        h, w = img.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            return np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
        return img

    def _cwssim_dtcwt(
        self, img1: np.ndarray, img2: np.ndarray,
        depth_km: Optional[float] = None
    ) -> float:
        """使用 dtcwt 的全局 CW-SSIM（层级加权）"""
        img1 = self._pad_to_even(img1)
        img2 = self._pad_to_even(img2)
        transform = dtcwt.Transform2d()
        c1 = transform.forward(img1, nlevels=self.nlevels)
        c2 = transform.forward(img2, nlevels=self.nlevels)

        weights = self._get_level_weights(depth_km)

        level_vals = []
        for lev in range(self.nlevels):
            hp1 = c1.highpasses[lev]
            hp2 = c2.highpasses[lev]
            orient_vals = []
            for orient in range(hp1.shape[2]):
                cx = hp1[:, :, orient].ravel()
                cy = hp2[:, :, orient].ravel()
                num = 2.0 * np.abs(np.sum(cx * np.conj(cy))) + self.K
                den = np.sum(np.abs(cx) ** 2) + np.sum(np.abs(cy) ** 2) + self.K
                orient_vals.append(num / den)
            level_vals.append(np.mean(orient_vals))

        return float(np.dot(weights, level_vals))

    def _level_maps_dtcwt(
        self, img1: np.ndarray, img2: np.ndarray, win_size: int
    ) -> List[np.ndarray]:
        """
        返回各小波层级的局部 CW-SSIM 图（已上采样到原图尺寸）。

        供加权合成与分层级诊断共用，避免重复做 DT-CWT。
        """
        from scipy.ndimage import zoom as nd_zoom

        orig_shape = img1.shape
        img1 = self._pad_to_even(img1)
        img2 = self._pad_to_even(img2)
        transform = dtcwt.Transform2d()
        c1 = transform.forward(img1, nlevels=self.nlevels)
        c2 = transform.forward(img2, nlevels=self.nlevels)

        level_maps: List[np.ndarray] = []
        for lev in range(self.nlevels):
            hp1 = c1.highpasses[lev]
            hp2 = c2.highpasses[lev]
            lev_map = np.zeros(hp1.shape[:2], dtype=np.float64)
            n_orient = hp1.shape[2]

            for o in range(n_orient):
                cx = hp1[:, :, o]
                cy = hp2[:, :, o]

                cross_real = self._local_filter(np.real(cx * np.conj(cy)), win_size)
                cross_imag = self._local_filter(np.imag(cx * np.conj(cy)), win_size)
                cross_mag = np.sqrt(cross_real ** 2 + cross_imag ** 2)

                auto1 = self._local_filter(np.abs(cx) ** 2, win_size)
                auto2 = self._local_filter(np.abs(cy) ** 2, win_size)

                num = 2.0 * cross_mag + self.K
                den = auto1 + auto2 + self.K
                lev_map += num / den

            lev_map /= n_orient
            scale = np.array(orig_shape, dtype=float) / np.array(lev_map.shape, dtype=float)
            up = nd_zoom(lev_map, scale, order=1)
            level_maps.append(up[:orig_shape[0], :orig_shape[1]])

        return level_maps

    def _cwssim_local_dtcwt(
        self, img1: np.ndarray, img2: np.ndarray, win_size: int,
        depth_km: Optional[float] = None
    ) -> Tuple[float, np.ndarray]:
        """使用 dtcwt 的局部 CW-SSIM（支持高斯窗口）"""
        level_maps = self._level_maps_dtcwt(img1, img2, win_size)
        weights = self._get_level_weights(depth_km)
        local_map = np.average(level_maps, axis=0, weights=weights)
        return float(np.mean(local_map)), local_map

    # ---------- Gabor 备选实现 ----------

    @staticmethod
    def _make_gabor_kernel(size: int, sigma: float, freq: float, theta: float) -> np.ndarray:
        """生成 2D 复 Gabor 核"""
        half = size // 2
        y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
        x_rot = x * np.cos(theta) + y * np.sin(theta)
        y_rot = -x * np.sin(theta) + y * np.cos(theta)
        gaussian = np.exp(-(x_rot ** 2 + y_rot ** 2) / (2 * sigma ** 2))
        sinusoid = np.exp(1j * 2 * np.pi * freq * x_rot)
        kernel = gaussian * sinusoid
        kernel /= np.abs(kernel).sum() + 1e-12
        return kernel

    def _cwssim_gabor(
        self, img1: np.ndarray, img2: np.ndarray,
        depth_km: Optional[float] = None
    ) -> float:
        """使用 Gabor 滤波器的全局 CW-SSIM（层级加权）"""
        from scipy.ndimage import convolve

        orientations = np.linspace(0, np.pi, 6, endpoint=False)
        scales = [2 ** i for i in range(1, self.nlevels + 1)]
        weights = self._get_level_weights(depth_km)

        level_vals = []
        for scale in scales:
            sigma = scale * 0.8
            freq = 1.0 / scale
            ksize = max(3, int(sigma * 4) | 1)

            orient_vals = []
            for theta in orientations:
                kernel = self._make_gabor_kernel(ksize, sigma, freq, theta)
                c1 = convolve(img1, np.real(kernel)) + 1j * convolve(img1, np.imag(kernel))
                c2 = convolve(img2, np.real(kernel)) + 1j * convolve(img2, np.imag(kernel))

                cx = c1.ravel()
                cy = c2.ravel()
                num = 2.0 * np.abs(np.sum(cx * np.conj(cy))) + self.K
                den = np.sum(np.abs(cx) ** 2) + np.sum(np.abs(cy) ** 2) + self.K
                orient_vals.append(num / den)
            level_vals.append(np.mean(orient_vals))

        return float(np.dot(weights, level_vals))

    def _cwssim_local_gabor(
        self, img1: np.ndarray, img2: np.ndarray, win_size: int,
        depth_km: Optional[float] = None
    ) -> Tuple[float, np.ndarray]:
        """使用 Gabor 滤波器的局部 CW-SSIM（支持高斯窗口）"""
        from scipy.ndimage import convolve

        orientations = np.linspace(0, np.pi, 6, endpoint=False)
        scales = [2 ** i for i in range(1, self.nlevels + 1)]
        weights = self._get_level_weights(depth_km)

        level_maps = []
        for scale in scales:
            sigma = scale * 0.8
            freq = 1.0 / scale
            ksize = max(3, int(sigma * 4) | 1)

            orient_sum = np.zeros_like(img1, dtype=np.float64)
            for theta in orientations:
                kernel = self._make_gabor_kernel(ksize, sigma, freq, theta)
                c1 = convolve(img1, np.real(kernel)) + 1j * convolve(img1, np.imag(kernel))
                c2 = convolve(img2, np.real(kernel)) + 1j * convolve(img2, np.imag(kernel))

                cross_real = self._local_filter(np.real(c1 * np.conj(c2)), win_size)
                cross_imag = self._local_filter(np.imag(c1 * np.conj(c2)), win_size)
                cross_mag = np.sqrt(cross_real ** 2 + cross_imag ** 2)

                auto1 = self._local_filter(np.abs(c1) ** 2, win_size)
                auto2 = self._local_filter(np.abs(c2) ** 2, win_size)

                num = 2.0 * cross_mag + self.K
                den = auto1 + auto2 + self.K
                orient_sum += num / den

            level_maps.append(orient_sum / len(orientations))

        local_map = np.average(level_maps, axis=0, weights=weights)
        return float(np.mean(local_map)), local_map


# ==================== 速度模型相似性分析器 ====================

class VelocityModelSimilarity:
    """速度模型结构相似性分析器 (v18.0 — 1D SSIM 主路径)"""

    def __init__(self, config: Optional[ModelSimilarityConfig] = None):
        self.config = config or ModelSimilarityConfig()
        self.base_config = BaseConfig()
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.ModelSimilarity',
            self.config.logging['level'],
        )
        self.logger.propagate = False

        self.models_dir = self.base_config.dirs['models'] / 'processed'
        self.output_dir = self.base_config.dirs['results'] / 'model_similarity'
        self.figures_dir = self.base_config.dirs['figures'] / 'model_similarity'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, VelocityModelNetCDF] = {}
        self.results: Dict[str, Any] = {}
        self.analysis_metadata: Dict[str, Any] = {}
        self.profiles: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        ss = self.config.ssim
        self.ssim_calc = SSIMCalculator(
            k1=ss.get('K1', 0.01),
            k2=ss.get('K2', 0.03),
            gaussian_sigma_per_pixel=ss.get('gaussian_sigma_per_pixel', 1.5 / 11.0),
        )

        s2 = self.config.ssim_2d
        self.ssim2d_calc = SSIMCalculator(
            k1=s2.get('K1', 0.01),
            k2=s2.get('K2', 0.03),
            gaussian_sigma_per_pixel=s2.get('gaussian_sigma_per_pixel', 1.5 / 11.0),
        )

        cw = self.config.cwssim
        self.cwssim_calc = CWSSIMCalculator(
            nlevels=cw['nlevels'],
            K=cw['K'],
            level_weights=cw.get('level_weights'),
            depth_adaptive_weights=cw.get('depth_adaptive_weights'),
            use_depth_adaptive_weights=cw.get('use_depth_adaptive_weights', False),
            normalize_method=cw.get('normalize_method', 'minmax'),
            percentile_range=tuple(cw.get('percentile_range', (2.0, 98.0))),
            use_gaussian_window=cw.get('use_gaussian_window', False),
        )
        self._rng = np.random.default_rng(self.config.null_test.get('random_seed', 42))
        self._backend_warned_pairs: set = set()

        self.logger.info("=" * 80)
        self.logger.info("🚀 速度模型结构相似性分析器 v19.0 (1D SSIM + 2D SSIM(z))")
        self.logger.info("=" * 80)
        self._print_config_summary()

    def _print_config_summary(self) -> None:
        """打印配置摘要"""
        ss = self.config.ssim
        s2 = self.config.ssim_2d
        print("\n📋 相似性分析配置 (v19.0 — 1D SSIM + 2D SSIM(z))")
        print("-" * 60)
        print(f"目标模型: {len(self.config.models)} 个")
        for info in self.config.models.values():
            print(f"  • {info['name']}")
        print("\n🔬 1D SSIM（对齐 2_1 剖面）:")
        print("  场: 原始绝对 Vs / Vp 的横向面积加权平均 V(z)")
        print(f"  深度窗口: {ss.get('win_km')} km  → 整体 1D SSIM + 局部 SSIM(z)")
        print("\n🔬 2D SSIM(z)（逐深度切片）:")
        print(f"  场: 去横向平均后的扰动 | 横向窗口: {s2.get('win_deg')}°")
        print("  分辨率: 按模型对最粗 (EARA–FWEA→0.25°, Sino对→1.0°)")
        print(f"\n📊 分析特征: {', '.join(self.config.analysis['target_features'])}")
        print(f"📁 输出: {self.figures_dir}")
        print("-" * 60)

    # ---------- 模型加载 ----------

    def load_models(self) -> None:
        """加载所有 NetCDF 模型"""
        self.logger.info("\n📦 加载 NetCDF 模型...")

        for model_key, info in self.config.models.items():
            nc_path = self.models_dir / info['processed_dir'] / info['netcdf_file']
            if not nc_path.exists():
                self.logger.warning(f"⚠️ 文件不存在: {nc_path}")
                continue
            try:
                model = VelocityModelNetCDF(nc_path, model_key, self.config, self.logger)
                self.models[model_key] = model
            except Exception as e:
                self.logger.error(f"❌ 加载失败 {model_key}: {e}")

        self.logger.info(f"✅ 成功加载 {len(self.models)} 个模型")
        if len(self.models) < 2:
            raise RuntimeError("至少需要 2 个模型才能进行相似性分析")

        self.analysis_metadata = {
            'version': 'v19.0-1D+2D-SSIM',
            'metrics': ['ssim_1d', 'ssim_2d'],
            'ssim_1d': {
                'field': 'raw absolute V(z), area-weighted lateral mean',
                'win_km': self.config.ssim.get('win_km'),
            },
            'ssim_2d': {
                'field': f"depth slice, perturbation={self.config.ssim_2d.get('perturbation')}",
                'win_deg': self.config.ssim_2d.get('win_deg'),
                'resolution_mode': self.config.cwssim.get('resolution_mode', 'pair_coarsest'),
            },
            'models': list(self.models.keys()),
            'pair_resolutions_deg': {},
        }
        # 预记录各对分析分辨率
        keys = list(self.models.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                m1, m2 = self.models[keys[i]], self.models[keys[j]]
                cov = self._get_common_coverage(m1, m2)
                self.analysis_metadata['pair_resolutions_deg'][
                    f"{keys[i]}_vs_{keys[j]}"
                ] = float(cov['resolution_deg'])
                self.logger.info(
                    f"🧭 {m1.name} vs {m2.name}: 分析分辨率="
                    f"{cov['resolution_deg']:.2f}°, 网格="
                    f"{len(cov['lons'])}×{len(cov['lats'])}"
                )

    def close(self) -> None:
        """关闭所有模型文件句柄"""
        for model in self.models.values():
            model.close()

    # ---------- 按模型对覆盖与采样 ----------

    def _get_common_coverage(
        self,
        m1: VelocityModelNetCDF,
        m2: VelocityModelNetCDF,
    ) -> Dict[str, Any]:
        """
        模型对公共覆盖 + 按最粗分辨率构建分析网格（Huang/公平对比）。

        - EARA vs FWEA → 0.25°
        - SinoScope vs * → 1.0°
        """
        lon_min = max(float(m1.lon.min()), float(m2.lon.min()))
        lon_max = min(float(m1.lon.max()), float(m2.lon.max()))
        lat_min = max(float(m1.lat.min()), float(m2.lat.min()))
        lat_max = min(float(m1.lat.max()), float(m2.lat.max()))

        res1 = m1.estimate_resolution()
        res2 = m2.estimate_resolution()
        pair_res = float(max(res1['lon'], res2['lon'], res1['lat'], res2['lat']))

        # 若两模型分辨率相同且等于 pair_res，优先用较细原生网格（对齐模型节点）
        fine = m1 if (len(m1.lon) * len(m1.lat) >= len(m2.lon) * len(m2.lat)) else m2
        fine_res = float(min(
            fine.estimate_resolution()['lon'],
            fine.estimate_resolution()['lat'],
        ))

        if abs(fine_res - pair_res) < 1e-6:
            mask_lon = (fine.lon >= lon_min - 1e-6) & (fine.lon <= lon_max + 1e-6)
            mask_lat = (fine.lat >= lat_min - 1e-6) & (fine.lat <= lat_max + 1e-6)
            target_lons = np.asarray(fine.lon[mask_lon])
            target_lats = np.asarray(fine.lat[mask_lat])
        else:
            lon0 = np.ceil(lon_min / pair_res - 1e-12) * pair_res
            lat0 = np.ceil(lat_min / pair_res - 1e-12) * pair_res
            target_lons = np.arange(lon0, lon_max + pair_res * 0.5, pair_res)
            target_lats = np.arange(lat0, lat_max + pair_res * 0.5, pair_res)
            target_lons = target_lons[target_lons <= lon_max + 1e-9]
            target_lats = target_lats[target_lats <= lat_max + 1e-9]

        if len(target_lons) < 2 or len(target_lats) < 2:
            raise ValueError(
                f"模型覆盖区域不足: {m1.name} 与 {m2.name} 的公共区域网格仅 "
                f"{len(target_lons)}×{len(target_lats)}"
            )

        return {
            'lons': target_lons,
            'lats': target_lats,
            'resolution_deg': pair_res,
            'grid_resolution': pair_res,
            'pair_analysis_resolution_deg': pair_res,
            'resolution_lon': pair_res,
            'resolution_lat': pair_res,
        }

    def _warn_backend_if_needed(self, pair_label: str, patch_px: int) -> None:
        """Huang 滑窗过小时显式告警（不再静默混用后端）"""
        need = 2 ** self.config.cwssim['nlevels'] + 1
        if patch_px >= need or not HAS_DTCWT:
            return
        if pair_label in self._backend_warned_pairs:
            return
        self._backend_warned_pairs.add(pair_label)
        self.logger.warning(
            f"  ⚠️ {pair_label}: Huang 滑窗={patch_px}px < DT-CWT 所需 {need}px "
            f"(nlevels={self.config.cwssim['nlevels']})，将回退 Gabor。"
            f"EARA–FWEA@0.25° 不受影响；Sino@1°+6° 属已知限制。"
        )

    def _resample_slice(
        self,
        model: VelocityModelNetCDF,
        data_3d: np.ndarray,
        d_idx: int,
        coverage: Dict[str, Any],
    ) -> np.ndarray:
        """将模型深度层重采样到分析网格（块平均或最近邻）"""
        method = self.config.masking.get('downsample_method', 'block_mean')
        src = data_3d[:, :, d_idx]
        if method == 'nearest':
            return nearest_neighbor_sample(
                model.lon, model.lat, src, coverage['lons'], coverage['lats'],
            )
        return block_mean_sample(
            model.lon, model.lat, src, coverage['lons'], coverage['lats'],
        )

    def _get_ssim_win_size(self, resolution_deg: Optional[float] = None) -> int:
        """
        2D SSIM 的横向高斯窗口像素数。

        用 win_deg 保证不同分辨率模型对的物理窗口一致：
        6° @ 0.25° → 25 px；6° @ 1.0° → 7 px。SSIM 在 7 px 仍稳定，
        不像 DT-CWT 那样需要 2^nlevels 的最小尺寸。
        """
        ss = self.config.ssim_2d
        res = resolution_deg or 1.0
        if ss.get('win_deg') is not None:
            size = int(np.ceil(float(ss['win_deg']) / res))
        else:
            size = int(ss.get('win_size', 11))
        return max(3, size | 1)

    def _get_local_win_size(self, resolution_deg: Optional[float] = None) -> int:
        """
        获取局部窗口大小（网格数）。

        local_win_deg° → 像素数 = ceil(local_win_deg / res_deg)，保证为奇数。
        """
        cw = self.config.cwssim
        res = resolution_deg or cw.get('pair_fallback_resolution_deg', 1.0)
        if cw.get('local_win_deg') is not None:
            size = int(np.ceil(cw['local_win_deg'] / res))
        elif cw.get('local_win_size') is not None:
            size = int(cw['local_win_size'])
        else:
            size = 7
        return max(3, size | 1)

    def _build_masks(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        feature: str,
        depth_km: float,
        coverage: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构建有效掩膜、面积权重与统计掩膜。

        Returns:
            (valid, weights, stat_mask)
        """
        valid = np.isfinite(s1) & np.isfinite(s2)

        if self.config.masking.get('use_all_model_common_mask', True):
            method = self.config.masking.get('downsample_method', 'block_mean')
            for m in self.models.values():
                if not m.has_parameter(feature):
                    continue
                try:
                    slice_m, _ = m.get_depth_slice(feature, depth_km)
                    if method == 'nearest':
                        s_m = nearest_neighbor_sample(
                            m.lon, m.lat, slice_m,
                            coverage['lons'], coverage['lats'],
                        )
                    else:
                        s_m = block_mean_sample(
                            m.lon, m.lat, slice_m,
                            coverage['lons'], coverage['lats'],
                        )
                    valid = valid & np.isfinite(s_m)
                except Exception:
                    continue

        weights = latitude_weights(coverage['lats'], s1.shape)
        buffer_px = int(self.config.masking.get('boundary_buffer_px', 0))
        stat_mask = erode_mask(valid, buffer_px)
        if not stat_mask.any():
            stat_mask = valid
        return valid, weights, stat_mask

    def _matched_depths(
        self,
        m1: VelocityModelNetCDF,
        m2: VelocityModelNetCDF,
    ) -> List[Tuple[int, int, float]]:
        """获取公共深度层，并按配置裁剪深度范围"""
        matched = find_common_depth_indices(m1.depth, m2.depth)
        d_min, d_max = self.config.analysis.get('depth_range_km', (0.0, 1000.0))
        return [
            (i1, i2, d) for i1, i2, d in matched
            if d_min - 1e-6 <= d <= d_max + 1e-6
        ]

    def _common_geographic_bbox(self) -> Dict[str, float]:
        """所有已加载模型覆盖的经纬度交集（对齐 2_1 共同区域）。"""
        return {
            'lon_min': max(float(m.lon.min()) for m in self.models.values()),
            'lon_max': min(float(m.lon.max()) for m in self.models.values()),
            'lat_min': max(float(m.lat.min()) for m in self.models.values()),
            'lat_max': min(float(m.lat.max()) for m in self.models.values()),
        }

    def _depth_grid(self) -> np.ndarray:
        """统一深度网格（km）。"""
        d_min, d_max = self.config.analysis.get('depth_range_km', (0.0, 1000.0))
        dz = float(self.config.analysis.get('depth_grid_step_km', 20.0))
        n = int(np.floor((d_max - d_min) / dz + 1e-9)) + 1
        return np.round(d_min + dz * np.arange(n), 6)

    def _extract_1d_profile(
        self,
        model: VelocityModelNetCDF,
        feature: str,
        bbox: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        在共同地理范围内对原始绝对速度做面积加权横向平均，得到 V(z)。

        与 2_1_Model_compare.calculate_1d_profile 同一物理定义，
        仅将 nanmean 改为 cos(lat) 面积加权。
        """
        lon_m = (model.lon >= bbox['lon_min'] - 1e-6) & (model.lon <= bbox['lon_max'] + 1e-6)
        lat_m = (model.lat >= bbox['lat_min'] - 1e-6) & (model.lat <= bbox['lat_max'] + 1e-6)
        if not lon_m.any() or not lat_m.any():
            return model.depth.copy(), np.full(model.depth.shape, np.nan)

        data = model.get_parameter(feature)
        depth_m = np.ones(data.shape[2], dtype=bool)
        sub = data[np.ix_(lon_m, lat_m, depth_m)]
        lats = np.asarray(model.lat[lat_m], dtype=np.float64)
        w_lat = np.cos(np.radians(lats))
        if not self.config.ssim.get('use_area_weights', True):
            w_lat = np.ones_like(w_lat)
        w = w_lat[np.newaxis, :, np.newaxis]
        valid = np.isfinite(sub)
        ww = np.where(valid, w, 0.0)
        num = np.sum(np.where(valid, sub, 0.0) * ww, axis=(0, 1))
        den = np.sum(ww, axis=(0, 1))
        mean = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
        return np.asarray(model.depth, dtype=np.float64), mean

    def _ensure_1d_profiles(self, feature: str) -> None:
        """提取并缓存所有模型在共同区域、统一深度网格上的 1D 剖面。"""
        if feature in self.profiles and self.profiles[feature]:
            return
        bbox = self._common_geographic_bbox()
        target_z = self._depth_grid()
        self.logger.info(
            f"  1D 剖面区域: lon [{bbox['lon_min']:.1f}, {bbox['lon_max']:.1f}], "
            f"lat [{bbox['lat_min']:.1f}, {bbox['lat_max']:.1f}], "
            f"深度网格 {len(target_z)} 层"
        )
        self.profiles[feature] = {}
        for key, model in self.models.items():
            if not model.has_parameter(feature):
                continue
            z_src, v_src = self._extract_1d_profile(model, feature, bbox)
            m = np.isfinite(z_src) & np.isfinite(v_src)
            if m.sum() < 3:
                self.logger.warning(f"  ⚠️ {model.name} {feature} 1D 剖面有效点不足")
                continue
            v_grid = np.interp(target_z, z_src[m], v_src[m], left=np.nan, right=np.nan)
            z_min, z_max = float(z_src[m].min()), float(z_src[m].max())
            v_grid[(target_z < z_min - 1e-6) | (target_z > z_max + 1e-6)] = np.nan
            self.profiles[feature][key] = {'depth': target_z, 'value': v_grid}
            self.logger.info(
                f"    {model.name} {feature.upper()}: "
                f"[{np.nanmin(v_grid):.3f}, {np.nanmax(v_grid):.3f}] km/s"
            )

    def _get_1d_win_size(self, depth_km: np.ndarray) -> int:
        """由 win_km 和深度步长得到奇数窗口。"""
        ss = self.config.ssim
        if len(depth_km) >= 2:
            dz = float(np.median(np.diff(depth_km)))
        else:
            dz = float(self.config.analysis.get('depth_grid_step_km', 20.0))
        if dz <= 0:
            dz = 20.0
        win_km = ss.get('win_km')
        if win_km is not None:
            size = int(np.round(float(win_km) / dz))
        else:
            size = int(ss.get('win_size', 11))
        n = int(len(depth_km))
        size = max(3, size | 1)
        if n >= 3:
            size = min(size, n if n % 2 == 1 else n - 1)
        return max(3, size)

    def compute_1d_metrics(
        self,
        m1_key: str,
        m2_key: str,
        feature: str,
    ) -> pd.DataFrame:
        """
        1D SSIM：对原始 Vs/Vp 的横向平均剖面 V(z) 计算局部 SSIM(z)。

        整体 1D SSIM = 局部 SSIM(z) 的均值（Wang 局部图平均）。

        Returns:
            DataFrame: depth_km, v1, v2, ssim_1d, ssim_1d_overall, ssim_1d_win_pt
        """
        self._ensure_1d_profiles(feature)
        profs = self.profiles.get(feature, {})
        if m1_key not in profs or m2_key not in profs:
            self.logger.warning(f"  ⚠️ {m1_key} vs {m2_key}: 缺少 1D 剖面")
            return pd.DataFrame()

        z = profs[m1_key]['depth']
        v1 = profs[m1_key]['value']
        v2 = profs[m2_key]['value']
        m1 = self.models[m1_key]
        m2 = self.models[m2_key]
        win_size = self._get_1d_win_size(z)

        overall, ssim_z = self.ssim_calc.compute_1d(v1, v2, win_size=win_size)
        self.logger.info(
            f"  📊 1D SSIM: {m1.name} vs {m2.name}  {feature.upper()}  "
            f"整体={overall:.4f}  窗口={win_size}pt"
        )

        return pd.DataFrame({
            'depth_km': z,
            'v1': v1,
            'v2': v2,
            'ssim_1d': ssim_z,
            'ssim_1d_overall': overall,
            'ssim_1d_win_pt': win_size,
        })

    def compute_depth_wise_metrics(
        self,
        m1_key: str,
        m2_key: str,
        feature: str,
    ) -> pd.DataFrame:
        """
        2D SSIM(z)：逐深度切片在去横向平均的扰动场上计算 SSIM。

        与 1D SSIM 分工明确——1D 量背景剖面，本指标量同一深度的横向结构。

        Returns:
            DataFrame: depth_km, ssim, valid_frac, analysis_resolution_deg, ssim_win_px
        """
        m1 = self.models[m1_key]
        m2 = self.models[m2_key]
        coverage = self._get_common_coverage(m1, m2)
        matched_depths = self._matched_depths(m1, m2)

        if not matched_depths:
            self.logger.warning(f"  ⚠️ {m1.name} vs {m2.name}: 无匹配深度层")
            return pd.DataFrame()

        res = coverage['resolution_deg']
        ssim_win = self._get_ssim_win_size(res)
        self.logger.info(
            f"  🗺️ 2D SSIM(z): {m1.name} vs {m2.name}  {feature.upper()}  "
            f"({len(matched_depths)} 层, 分辨率={res}°, 窗口={ssim_win}px)"
        )

        data1_3d = m1.get_parameter(feature)
        data2_3d = m2.get_parameter(feature)
        cfg2d = self.config.ssim_2d
        use_area = bool(cfg2d.get('use_area_weights', True))
        pert_mode = cfg2d.get('perturbation', 'layer_mean')
        do_r = bool(self.config.companion.get('compute_signed_correlation', False))
        do_sign = bool(self.config.companion.get('compute_sign_agreement', False))
        sign_pct = float(self.config.companion.get('sign_threshold_percentile', 50.0))

        rows: List[Dict[str, Any]] = []
        for d_idx1, d_idx2, depth_km in tqdm(matched_depths, desc="  深度层", leave=False):
            s1 = self._resample_slice(m1, data1_3d, d_idx1, coverage)
            s2 = self._resample_slice(m2, data2_3d, d_idx2, coverage)
            valid, weights, stat_mask = self._build_masks(
                s1, s2, feature, depth_km, coverage,
            )
            area_w = weights if use_area else None

            if pert_mode == 'layer_mean':
                p1 = remove_layer_mean(s1, valid, weights)
                p2 = remove_layer_mean(s2, valid, weights)
            else:
                p1, p2 = s1, s2

            ssim_val, _ = self.ssim2d_calc.compute(
                p1, p2,
                valid=valid,
                win_size=ssim_win,
                weights=area_w,
                stat_mask=stat_mask,
            )

            row: Dict[str, Any] = {
                'depth_km': depth_km,
                'ssim': ssim_val,
                'valid_frac': float(valid.mean()) if valid.size else 0.0,
                'analysis_resolution_deg': res,
                'ssim_win_px': ssim_win,
            }
            if do_r:
                row['r_signed'] = weighted_pearson(p1, p2, weights, valid)
            if do_sign:
                row['sign_agreement'] = sign_agreement(
                    p1, p2, weights, valid, sign_pct,
                )
            rows.append(row)

        df = pd.DataFrame(rows)
        if len(df):
            self.logger.info(
                f"      2D SSIM: [{df['ssim'].min():.3f}, {df['ssim'].max():.3f}]  "
                f"mean={df['ssim'].mean():.3f}"
            )
        return df

    def _compute_cwssim_spatial_huang(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        coverage: Dict[str, Any],
        depth_km: float,
        valid: np.ndarray,
        weights: np.ndarray,
        stat_mask: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Huang (2024) 空间滑窗法：每个网格点取 local_win_deg° patch 计算 CW-SSIM。

        Sino@1° 时 patch 可能不足以支撑 DT-CWT，见 _warn_backend_if_needed。
        """
        res = coverage['resolution_deg']
        half = self._get_local_win_size(res) // 2
        n_lon, n_lat = s1.shape
        cwssim_grid = np.full((n_lon, n_lat), np.nan, dtype=np.float64)

        ij_list = [
            (i, j) for i in range(half, n_lon - half)
            for j in range(half, n_lat - half)
            if valid[i, j]
        ]
        for i, j in tqdm(ij_list, desc=f"  {depth_km:.0f} km", leave=False):
            patch1 = s1[i - half:i + half + 1, j - half:j + half + 1]
            patch2 = s2[i - half:i + half + 1, j - half:j + half + 1]
            pvalid = np.isfinite(patch1) & np.isfinite(patch2)
            if np.sum(pvalid) < 10:
                continue
            p1 = fill_invalid_nearest(patch1, pvalid)
            p2 = fill_invalid_nearest(patch2, pvalid)
            cwssim_grid[i, j] = self.cwssim_calc.compute_global(
                p1, p2, depth_km=depth_km,
            )

        # Huang: 有效 patch 的算术平均（不对面积再加权）
        m = stat_mask & np.isfinite(cwssim_grid)
        if not m.any():
            return float('nan'), cwssim_grid
        global_mean = float(np.mean(cwssim_grid[m]))
        return global_mean, cwssim_grid

    # ---------- CW-SSIM 空间分布 ----------

    def compute_cwssim_spatial_pair(
        self,
        m1_key: str,
        m2_key: str,
        feature: str,
        depths: Optional[List[float]] = None,
    ) -> Dict[float, Tuple[float, np.ndarray, Dict, np.ndarray]]:
        """
        计算指定深度的 CW-SSIM 局部空间分布图。

        默认使用 compute_local（全图一次 DT-CWT + 系数域局部平均）。
        可选 huang_patch（慢，且在 1° 下易回退 Gabor）。

        Returns:
            {depth_km: (global_cwssim, local_map, coverage, valid_mask)}
        """
        if depths is None:
            depths = self.config.analysis['cwssim_spatial_depths']

        m1 = self.models[m1_key]
        m2 = self.models[m2_key]
        coverage = self._get_common_coverage(m1, m2)
        res = coverage['resolution_deg']
        win_size = self._get_local_win_size(res)
        method = self.config.cwssim.get('spatial_method', 'huang_patch')
        pair_label = f"{m1.name} vs {m2.name}"
        self._warn_backend_if_needed(pair_label, win_size)

        self.logger.info(
            f"  🗺️ CW-SSIM 空间分布 ({method}/Huang): {pair_label}  "
            f"(分辨率={res}°, 滑窗={self.config.cwssim.get('local_win_deg')}°="
            f"{win_size}px)"
        )

        results: Dict[float, Tuple[float, np.ndarray, Dict, np.ndarray]] = {}
        # 空间切片使用全部公共深度，不受 depth_range_km（仅约束逐层曲线）限制
        matched = find_common_depth_indices(m1.depth, m2.depth)
        depth_map = {d: (i1, i2) for i1, i2, d in matched}
        if not depth_map:
            self.logger.warning("    ⚠️ 无匹配深度层")
            return results

        data1_3d = m1.get_parameter(feature)
        data2_3d = m2.get_parameter(feature)

        for target_depth in depths:
            closest_depth = min(depth_map.keys(), key=lambda d: abs(d - target_depth))
            if abs(closest_depth - target_depth) > 10.0:
                continue
            d_idx1, d_idx2 = depth_map[closest_depth]
            s1 = self._resample_slice(m1, data1_3d, d_idx1, coverage)
            s2 = self._resample_slice(m2, data2_3d, d_idx2, coverage)
            valid, weights, stat_mask = self._build_masks(
                s1, s2, feature, closest_depth, coverage,
            )

            try:
                if method == 'local_filter':
                    global_val, local_map = self.cwssim_calc.compute_local(
                        s1, s2, win_size=win_size, depth_km=closest_depth,
                        valid=valid, weights=weights, stat_mask=stat_mask,
                    )
                else:
                    # 默认 Huang 6° 滑窗
                    global_val, local_map = self._compute_cwssim_spatial_huang(
                        s1, s2, coverage, closest_depth, valid, weights, stat_mask,
                    )
                    local_map = np.where(valid, local_map, np.nan)
                results[closest_depth] = (global_val, local_map, coverage, valid)
                self.logger.info(
                    f"    ✅ {closest_depth:.0f} km: CW-SSIM = {global_val:.4f}  "
                    f"(valid={valid.mean()*100:.1f}%)"
                )
            except Exception as e:
                self.logger.warning(f"    ⚠️ {closest_depth:.0f} km 失败: {e}")

        return results

    # ---------- 批量分析 ----------

    def compare_all_models(self, features: Optional[List[str]] = None) -> None:
        """对所有模型对、所有特征执行完整相似性分析"""
        if features is None:
            features = self.config.analysis['target_features']

        model_keys = list(self.models.keys())
        pairs = [
            (model_keys[i], model_keys[j])
            for i in range(len(model_keys))
            for j in range(i + 1, len(model_keys))
        ]

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"🔬 批量相似性分析: {len(pairs)} 模型对 × {len(features)} 特征")
        self.logger.info("=" * 80)

        for feature in features:
            models_with_feature = [
                k for k in model_keys if self.models[k].has_parameter(feature)
            ]
            if len(models_with_feature) < 2:
                self.logger.warning(f"特征 {feature} 可用模型不足 2 个，跳过")
                continue

            feature_pairs = [
                (models_with_feature[i], models_with_feature[j])
                for i in range(len(models_with_feature))
                for j in range(i + 1, len(models_with_feature))
            ]

            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"特征: {feature.upper()}  ({len(feature_pairs)} 对)")
            self.logger.info(f"{'=' * 60}")

            for m1_key, m2_key in feature_pairs:
                pair_id = f"{m1_key}_vs_{m2_key}_{feature}"
                try:
                    df_1d = self.compute_1d_metrics(m1_key, m2_key, feature)
                    self.results[f"{pair_id}_1d"] = df_1d

                    if self.config.analysis['compute_depth_wise']:
                        df = self.compute_depth_wise_metrics(m1_key, m2_key, feature)
                        self.results[f"{pair_id}_depth"] = df

                    if self.config.analysis['compute_cwssim_spatial']:
                        cs = self.compute_cwssim_spatial_pair(m1_key, m2_key, feature)
                        self.results[f"{pair_id}_cwssim_spatial"] = cs
                except Exception as e:
                    self.logger.error(f"❌ {m1_key} vs {m2_key} ({feature}): {e}")
                    import traceback
                    traceback.print_exc()

        self.logger.info("\n✅ 批量相似性分析完成")

    # ---------- 结果保存 ----------

    def save_results(self) -> None:
        """保存分析结果与分析元数据"""
        self.logger.info("\n💾 保存分析结果...")

        features = set()
        for key in self.results:
            for suffix in ('_depth', '_1d'):
                if key.endswith(suffix):
                    features.add(key[:-len(suffix)].split('_')[-1])

        for feature in features:
            feat_dir = self.output_dir / feature
            feat_dir.mkdir(parents=True, exist_ok=True)

            meta_file = feat_dir / f'analysis_metadata_{feature}.json'
            meta = dict(self.analysis_metadata)
            meta['feature'] = feature
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            self.logger.info(f"  ✅ {meta_file.name}")

            suffix_depth = f'_{feature}_depth'
            suffix_1d = f'_{feature}_1d'
            depth_keys = [k for k in self.results if k.endswith(suffix_depth)]
            all_summaries: List[Dict[str, Any]] = []

            for k1d in [k for k in self.results if k.endswith(suffix_1d)]:
                df_1d = self.results[k1d]
                if not len(df_1d):
                    continue
                csv_1d = feat_dir / f"{k1d}.csv"
                df_1d.to_csv(csv_1d, index=False, float_format='%.6f')
                self.logger.info(f"  ✅ {csv_1d.name}")

            for dk in depth_keys:
                df = self.results[dk]
                csv_file = feat_dir / f"{dk}.csv"
                df.to_csv(csv_file, index=False, float_format='%.6f')
                self.logger.info(f"  ✅ {csv_file.name}")

                pair_name = dk[:-len(suffix_depth)]
                ssim_mean = float(df['ssim'].mean()) if 'ssim' in df.columns and len(df) else float('nan')
                ssim_std = float(df['ssim'].std()) if 'ssim' in df.columns and len(df) else float('nan')
                summary: Dict[str, Any] = {
                    'pair': pair_name,
                    'feature': feature,
                    'ssim_2d_mean': ssim_mean,
                    'ssim_2d_std': ssim_std,
                }
                df_1d = self.results.get(f"{pair_name}{suffix_1d}")
                if df_1d is not None and len(df_1d):
                    summary['ssim_1d_overall'] = float(df_1d['ssim_1d_overall'].iloc[0])
                if 'cwssim' in df.columns:
                    summary['cwssim_mean'] = float(df['cwssim'].mean())
                    summary['cwssim_std'] = float(df['cwssim'].std())
                if 'r_signed' in df.columns:
                    summary['r_signed_mean'] = float(df['r_signed'].mean())
                if 'sign_agreement' in df.columns:
                    summary['sign_agreement_mean'] = float(df['sign_agreement'].mean())
                if 'z_score' in df.columns:
                    summary['z_score_mean'] = float(df['z_score'].mean())
                if 'null_mean' in df.columns:
                    summary['null_mean_avg'] = float(df['null_mean'].mean())
                if 'valid_frac' in df.columns:
                    summary['valid_frac_mean'] = float(df['valid_frac'].mean())
                all_summaries.append(summary)

            if all_summaries:
                df_summary = pd.DataFrame(all_summaries)
                summary_file = feat_dir / f'similarity_summary_{feature}.csv'
                df_summary.to_csv(summary_file, index=False, float_format='%.6f')
                self.logger.info(f"  ✅ {summary_file.name}")

                json_data = {}
                for s in all_summaries:
                    json_data[s['pair']] = {
                        k: (float(v) if isinstance(v, (np.floating, float)) else v)
                        for k, v in s.items()
                    }
                json_file = feat_dir / f'similarity_summary_{feature}.json'
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                self.logger.info(f"  ✅ {json_file.name}")

            if feature in self.profiles and self.profiles[feature]:
                prof_rows = []
                for key, prof in self.profiles[feature].items():
                    name = self.models[key].name if key in self.models else key
                    for z, v in zip(prof['depth'], prof['value']):
                        prof_rows.append({
                            'model': name, 'model_key': key,
                            'depth_km': float(z), feature: float(v) if np.isfinite(v) else np.nan,
                        })
                if prof_rows:
                    prof_file = feat_dir / f'1d_profiles_{feature}.csv'
                    pd.DataFrame(prof_rows).to_csv(
                        prof_file, index=False, float_format='%.6f',
                    )
                    self.logger.info(f"  ✅ {prof_file.name}")

        self.logger.info("✅ 所有结果已保存")


class SimilarityVisualization:
    """结构相似性可视化 (v18.0 — 1D SSIM 主路径)"""

    def __init__(self, analyzer: VelocityModelSimilarity):
        self.analyzer = analyzer
        self.config = analyzer.config
        self.logger = analyzer.logger
        self.figures_dir = analyzer.figures_dir

        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.4)

    def _pair_label(self, pair_name: str) -> str:
        """将 '{m1_key}_vs_{m2_key}' 转为 'Name1 vs Name2' 可读标签"""
        parts = pair_name.split('_vs_')
        if len(parts) == 2:
            n1 = self.analyzer.models.get(parts[0])
            n2 = self.analyzer.models.get(parts[1])
            return f"{n1.name if n1 else parts[0]} vs {n2.name if n2 else parts[1]}"
        return pair_name

    def _parse_pair_indices(
        self, pair_name: str, model_keys: List[str]
    ) -> Tuple[Optional[int], Optional[int]]:
        """解析 pair_name 并返回在 model_keys 中的索引对"""
        parts = pair_name.split('_vs_')
        if len(parts) == 2 and parts[0] in model_keys and parts[1] in model_keys:
            return model_keys.index(parts[0]), model_keys.index(parts[1])
        return None, None

    def _get_plot_extent(self, pair_name: str) -> Tuple[float, float, float, float]:
        """获取空间分布图显示范围：模型对公共 lon/lat。"""
        parts = pair_name.split('_vs_')
        if len(parts) == 2 and parts[0] in self.analyzer.models and parts[1] in self.analyzer.models:
            m1 = self.analyzer.models[parts[0]]
            m2 = self.analyzer.models[parts[1]]
            return (
                float(max(m1.lon.min(), m2.lon.min())),
                float(min(m1.lon.max(), m2.lon.max())),
                float(max(m1.lat.min(), m2.lat.min())),
                float(min(m1.lat.max(), m2.lat.max())),
            )
        return (80.0, 150.0, 10.0, 55.0)

    def _save_figure(self, fig: plt.Figure, name: str):
        """保存图表（多格式）"""
        prefix = self.config.output['figure_prefix']
        for fmt in self.config.output['save_formats']:
            filepath = self.figures_dir / f"{prefix}{name}.{fmt}"
            fig.savefig(filepath, dpi=self.config.output['save_dpi'],
                        bbox_inches='tight', format=fmt)
        self.logger.info(f"  ✅ 保存: {prefix}{name}")
        plt.close(fig)

    def _add_discontinuities(self, ax, orientation: str = 'horizontal'):
        """添加地质不连续面标记"""
        for name, depth in self.config.visualization['discontinuities']:
            if orientation == 'horizontal':
                ax.axhline(y=depth, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
                ax.text(ax.get_xlim()[1], depth, f' {name}',
                        va='center', fontsize=8, color='gray')
            else:
                ax.axvline(x=depth, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # ---------- 1. SSIM 深度曲线 ----------

    def _plot_depth_curve(
        self,
        ax,
        suffix: str,
        column: str,
        feature: str,
    ) -> int:
        """在给定坐标轴上绘制各模型对的 SSIM(z) 曲线，返回绘制条数。"""
        keys = sorted(k for k in self.analyzer.results if k.endswith(suffix))
        pair_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']
        n_plotted = 0
        for idx, key in enumerate(keys):
            df = self.analyzer.results[key]
            if column not in df.columns or not len(df):
                continue
            ax.plot(
                df[column], df['depth_km'], '-o',
                color=pair_colors[idx % len(pair_colors)],
                linewidth=2, markersize=3,
                label=self._pair_label(key[:-len(suffix)]),
            )
            n_plotted += 1
        if n_plotted:
            ax.set_ylabel('Depth (km)', fontsize=12)
            ax.invert_yaxis()
            ax.set_xlim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            self._add_discontinuities(ax, 'horizontal')
        return n_plotted

    def plot_depth_ssim(self, feature: str = 'vs') -> Optional[plt.Figure]:
        """绘制 1D SSIM(z) 与 2D SSIM(z) 随深度变化曲线（并排）"""
        has_1d = any(k.endswith(f'_{feature}_1d') for k in self.analyzer.results)
        has_2d = any(k.endswith(f'_{feature}_depth') for k in self.analyzer.results)
        if not (has_1d or has_2d):
            return None

        self.logger.info(f"\n🎨 SSIM 深度曲线: {feature.upper()}")

        fig, (ax_1d, ax_2d) = plt.subplots(1, 2, figsize=(14, 9))

        self._plot_depth_curve(ax_1d, f'_{feature}_1d', 'ssim_1d', feature)
        ax_1d.set_xlabel('1D SSIM', fontsize=13)
        ax_1d.set_title(
            '1D SSIM(z)\nraw mean V(z) profiles',
            fontsize=13, fontweight='bold',
        )
        ax_1d.legend(fontsize=10, loc='lower left')

        self._plot_depth_curve(ax_2d, f'_{feature}_depth', 'ssim', feature)
        ax_2d.set_xlabel('2D SSIM', fontsize=13)
        ax_2d.set_title(
            '2D SSIM(z)\nlateral structure per depth slice',
            fontsize=13, fontweight='bold',
        )
        ax_2d.legend(fontsize=10, loc='lower right')

        fig.suptitle(
            f'Structural Similarity vs Depth — {feature.upper()}',
            fontsize=16, fontweight='bold', y=1.00,
        )
        plt.tight_layout()
        self._save_figure(fig, f'depth_ssim_{feature}')
        return fig

    def plot_depth_cwssim(self, feature: str = 'vs') -> Optional[plt.Figure]:
        """兼容旧入口，转发到 SSIM 深度曲线。"""
        return self.plot_depth_ssim(feature)

    # ---------- 2. 整体 SSIM 热图矩阵 ----------

    def _build_matrix(self, feature: str, metric: str) -> Optional[np.ndarray]:
        """
        构建 N×N 相似性矩阵。

        Args:
            metric: '1d' 取整体 1D SSIM；'2d' 取 2D SSIM(z) 的深度平均
        """
        model_keys = list(self.analyzer.models.keys())
        n = len(model_keys)
        matrix = np.ones((n, n))
        found = False

        suffix = f'_{feature}_1d' if metric == '1d' else f'_{feature}_depth'
        for key in sorted(k for k in self.analyzer.results if k.endswith(suffix)):
            df = self.analyzer.results[key]
            if not len(df):
                continue
            i, j = self._parse_pair_indices(key[:-len(suffix)], model_keys)
            if i is None:
                continue
            if metric == '1d':
                if 'ssim_1d_overall' not in df.columns:
                    continue
                val = float(df['ssim_1d_overall'].iloc[0])
            else:
                if 'ssim' not in df.columns:
                    continue
                val = float(np.nanmean(df['ssim']))
            matrix[i, j] = val
            matrix[j, i] = val
            found = True

        return matrix if found else None

    def _draw_heatmap(self, ax, matrix: np.ndarray, label: str) -> None:
        """在给定坐标轴绘制上三角相似性热图。"""
        hm_cfg = self.config.visualization['heatmap']
        names = [self.analyzer.models[k].name for k in self.analyzer.models]
        mask = np.tril(np.ones_like(matrix, dtype=bool), k=-1)
        sns.heatmap(
            pd.DataFrame(matrix, index=names, columns=names),
            mask=mask, annot=True, fmt='.3f', cmap=hm_cfg['cmap'],
            vmin=hm_cfg.get('ssim_vmin', 0.0), vmax=hm_cfg.get('ssim_vmax', 1.0),
            square=True,
            linewidths=hm_cfg['cell_linewidth'],
            linecolor=hm_cfg['cell_linecolor'],
            annot_kws={'fontsize': 18, 'fontweight': 'bold'},
            cbar_kws={'label': label, 'shrink': 0.8},
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                           fontsize=12, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=12, fontweight='bold')

    def plot_ssim_heatmap(self, feature: str = 'vs') -> Optional[plt.Figure]:
        """绘制 N×N 相似性矩阵：整体 1D SSIM 与深度平均 2D SSIM 并排"""
        m_1d = self._build_matrix(feature, '1d')
        m_2d = self._build_matrix(feature, '2d')
        if m_1d is None and m_2d is None:
            return None

        self.logger.info(f"\n🎨 整体 SSIM 热图: {feature.upper()}")
        hm_cfg = self.config.visualization['heatmap']
        panels = [(m, lab) for m, lab in
                  ((m_1d, '1D SSIM'), (m_2d, '2D SSIM (depth-averaged)'))
                  if m is not None]

        fig, axes = plt.subplots(1, len(panels), figsize=(9 * len(panels), 8))
        if len(panels) == 1:
            axes = [axes]

        for ax, (matrix, label) in zip(axes, panels):
            self._draw_heatmap(ax, matrix, label)
            ax.set_title(label, fontsize=15, fontweight='bold', pad=12)

        fig.suptitle(
            f'Model Similarity Matrix — {feature.upper()}',
            fontsize=hm_cfg['title_fontsize'], fontweight='bold', y=1.01,
        )
        plt.tight_layout()
        self._save_figure(fig, f'ssim_heatmap_{feature}')
        return fig

    def plot_cwssim_heatmap(self, feature: str = 'vs') -> Optional[plt.Figure]:
        """兼容旧入口，转发到整体 SSIM 热图。"""
        return self.plot_ssim_heatmap(feature)

    # ---------- 4. CW-SSIM 空间分布图 ----------

    def plot_cwssim_spatial(self, feature: str = 'vs') -> Optional[plt.Figure]:
        """绘制 CW-SSIM 局部空间分布图"""
        suffix = f'_{feature}_cwssim_spatial'
        spatial_keys = [k for k in self.analyzer.results if k.endswith(suffix)]
        if not spatial_keys:
            return None

        self.logger.info(f"\n🎨 CW-SSIM 空间分布: {feature.upper()}")

        for sk in spatial_keys:
            cs_data = self.analyzer.results[sk]
            pair_name = sk[:-len(suffix)]
            title = self._pair_label(pair_name)

            n_depths = len(cs_data)
            if n_depths == 0:
                continue

            n_cols = min(3, n_depths)
            n_rows = int(np.ceil(n_depths / n_cols))

            if HAS_CARTOPY:
                fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
            else:
                fig, axes_arr = plt.subplots(
                    n_rows, n_cols,
                    figsize=(6 * n_cols, 5 * n_rows),
                    squeeze=False,
                )

            for idx, (depth_km, item) in enumerate(sorted(cs_data.items())):
                # 兼容旧格式 (3元组) 与新格式 (4元组含 valid_mask)
                if len(item) == 4:
                    global_val, local_map, coverage, valid_mask = item
                    # 公空缺：仅显示两模型均有有效数据的区域
                    plot_raw = np.where(valid_mask.T, local_map.T, np.nan)
                else:
                    global_val, local_map, coverage = item
                    plot_raw = local_map.T
                plot_data = np.ma.masked_invalid(plot_raw)

                lons = coverage['lons']
                lats = coverage['lats']

                if HAS_CARTOPY:
                    proj = ccrs.PlateCarree()
                    ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection=proj)
                    # 与 2-1 一致：先 set_extent（使用模型网格范围），再绘制
                    lon_min, lon_max, lat_min, lat_max = self._get_plot_extent(pair_name)
                    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
                    if self.config.visualization['add_coastlines']:
                        ax.coastlines(
                            linewidth=self.config.visualization['coastline_linewidth'],
                            color=self.config.visualization['coastline_color'],
                        )
                    im = ax.pcolormesh(
                        lons, lats, plot_data,
                        cmap=self.config.visualization['cmap_cwssim'],
                        vmin=0.0, vmax=1.0,
                        transform=proj, shading='nearest',
                    )
                    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.4, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                else:
                    row, col = divmod(idx, n_cols)
                    ax = axes_arr[row, col]
                    im = ax.pcolormesh(
                        lons, lats, plot_data,
                        cmap=self.config.visualization['cmap_cwssim'],
                        vmin=0.0, vmax=1.0, shading='nearest',
                    )
                    ax.set_xlabel('Longitude (°)')
                    ax.set_ylabel('Latitude (°)')
                    # 与 2-1 一致：使用模型网格范围
                    lon_min, lon_max, lat_min, lat_max = self._get_plot_extent(pair_name)
                    ax.set_xlim(lon_min, lon_max)
                    ax.set_ylim(lat_min, lat_max)

                ax.set_title(f'{depth_km:.0f} km  (CW-SSIM={global_val:.3f})',
                             fontsize=12, fontweight='bold')
                plt.colorbar(im, ax=ax, shrink=0.7, label='CW-SSIM')

            if not HAS_CARTOPY:
                for idx in range(n_depths, n_rows * n_cols):
                    row, col = divmod(idx, n_cols)
                    axes_arr[row, col].set_visible(False)

            fig.suptitle(
                f'Local CW-SSIM Spatial Distribution — {title}\n{feature.upper()}',
                fontsize=15, fontweight='bold', y=1.02,
            )
            plt.tight_layout()
            safe_pair = pair_name.replace('/', '_')
            self._save_figure(fig, f'cwssim_spatial_{safe_pair}_{feature}')

        return None

    # ---------- 5. 综合概览面板 ----------

    def plot_1d_profiles(self, feature: str = 'vs') -> Optional[plt.Figure]:
        """绘制共同区域原始 1D 速度剖面（对齐 2_1）。"""
        profs = self.analyzer.profiles.get(feature, {})
        if not profs:
            return None

        self.logger.info(f"\n🎨 1D 速度剖面: {feature.upper()}")
        fig, ax = plt.subplots(figsize=(6, 8))
        for key, prof in profs.items():
            model = self.analyzer.models.get(key)
            name = model.name if model else key
            color = model.color if model else '#333333'
            z = prof['depth']
            v = prof['value']
            m = np.isfinite(z) & np.isfinite(v)
            ax.plot(v[m], z[m], color=color, linewidth=1.8, label=name)

        ax.set_xlabel(f'{feature.upper()} (km/s)', fontsize=14)
        ax.set_ylabel('Depth (km)', fontsize=14)
        ax.set_title(
            f'1D Mean {feature.upper()} Profiles (common region)',
            fontsize=14, fontweight='bold',
        )
        ax.invert_yaxis()
        self._add_discontinuities(ax, 'horizontal')
        ax.legend(fontsize=11, loc='lower left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_figure(fig, f'1d_profile_{feature}')
        return fig

    def plot_similarity_overview(self, feature: str = 'vs') -> Optional[plt.Figure]:
        """1D 剖面 + 1D SSIM(z) + 2D SSIM(z) + 两个相似性矩阵"""
        has_1d = any(k.endswith(f'_{feature}_1d') for k in self.analyzer.results)
        has_2d = any(k.endswith(f'_{feature}_depth') for k in self.analyzer.results)
        if not (has_1d or has_2d):
            return None

        self.logger.info(f"\n🎨 综合概览面板: {feature.upper()}")

        fig = plt.figure(figsize=(20, 11))
        gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.30)
        ax_prof = fig.add_subplot(gs[0, 0])
        ax_1d = fig.add_subplot(gs[0, 1])
        ax_2d = fig.add_subplot(gs[0, 2])
        ax_hm1 = fig.add_subplot(gs[1, 0])
        ax_hm2 = fig.add_subplot(gs[1, 1])

        for key, prof in self.analyzer.profiles.get(feature, {}).items():
            model = self.analyzer.models.get(key)
            name = model.name if model else key
            color = model.color if model else '#333333'
            z, v = prof['depth'], prof['value']
            m = np.isfinite(z) & np.isfinite(v)
            ax_prof.plot(v[m], z[m], color=color, linewidth=1.8, label=name)
        ax_prof.set_xlabel(f'{feature.upper()} (km/s)', fontsize=12)
        ax_prof.set_ylabel('Depth (km)', fontsize=12)
        ax_prof.set_title('1D Mean Profiles', fontsize=14, fontweight='bold')
        ax_prof.invert_yaxis()
        ax_prof.legend(fontsize=9, loc='lower left')
        ax_prof.grid(True, alpha=0.3)
        self._add_discontinuities(ax_prof)

        self._plot_depth_curve(ax_1d, f'_{feature}_1d', 'ssim_1d', feature)
        ax_1d.set_xlabel('1D SSIM', fontsize=12)
        ax_1d.set_title('1D SSIM vs Depth', fontsize=14, fontweight='bold')
        ax_1d.legend(fontsize=8, loc='lower left')

        self._plot_depth_curve(ax_2d, f'_{feature}_depth', 'ssim', feature)
        ax_2d.set_xlabel('2D SSIM', fontsize=12)
        ax_2d.set_title('2D SSIM vs Depth', fontsize=14, fontweight='bold')
        ax_2d.legend(fontsize=8, loc='lower right')

        m_1d = self._build_matrix(feature, '1d')
        m_2d = self._build_matrix(feature, '2d')
        if m_1d is not None:
            self._draw_heatmap(ax_hm1, m_1d, '1D SSIM')
            ax_hm1.set_title('Overall 1D SSIM', fontsize=14, fontweight='bold')
        else:
            ax_hm1.set_visible(False)
        if m_2d is not None:
            self._draw_heatmap(ax_hm2, m_2d, '2D SSIM')
            ax_hm2.set_title('2D SSIM (depth-averaged)', fontsize=14, fontweight='bold')
        else:
            ax_hm2.set_visible(False)

        fig.suptitle(
            f'Structural Similarity — {feature.upper()}',
            fontsize=18, fontweight='bold', y=0.98,
        )
        self._save_figure(fig, f'similarity_overview_{feature}')
        return fig


# ==================== 主函数 ====================

def main():
    """主函数 — 1D SSIM + 2D SSIM(z)"""
    print("\n" + "=" * 80)
    print("🚀 EASTASIA-FWI 速度模型结构相似性分析 (v19.0)")
    print("   框架: Wang et al. (2004) SSIM — 1D 剖面 + 2D 逐深度切片")
    print("=" * 80)
    print("\n🎯 主路径:")
    print("  • 1D SSIM: 共同区域横向平均的原始 Vs / Vp 剖面 V(z)")
    print("  • 2D SSIM(z): 逐深度切片、去横向平均后的横向结构")
    print("=" * 80 + "\n")

    analyzer = None
    try:
        print("📋 步骤 1/5: 初始化配置...")
        config = ModelSimilarityConfig()
        analyzer = VelocityModelSimilarity(config)

        print("\n📦 步骤 2/5: 加载 NetCDF 模型...")
        analyzer.load_models()

        print("\n🔬 步骤 3/5: 批量相似性分析...")
        analyzer.compare_all_models()

        print("\n💾 步骤 4/5: 保存结果...")
        analyzer.save_results()

        print("\n🎨 步骤 5/5: 生成可视化...")
        visualizer = SimilarityVisualization(analyzer)

        features_done = set()
        for key in analyzer.results:
            for suffix in ('_depth', '_1d'):
                if key.endswith(suffix):
                    features_done.add(key[:-len(suffix)].split('_')[-1])

        for feature in sorted(features_done):
            print(f"\n  📊 {feature.upper()}...")
            visualizer.plot_1d_profiles(feature)
            visualizer.plot_depth_ssim(feature)
            visualizer.plot_ssim_heatmap(feature)
            visualizer.plot_similarity_overview(feature)
            if analyzer.config.analysis.get('compute_cwssim_spatial'):
                visualizer.plot_cwssim_spatial(feature)

        print("\n" + "=" * 80)
        print("✅ 分析完成 (1D SSIM + 2D SSIM(z))")
        print(f"📁 结果: {analyzer.output_dir}")
        print(f"📁 图表: {analyzer.figures_dir}")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if analyzer is not None:
            analyzer.close()


if __name__ == "__main__":
    main()
