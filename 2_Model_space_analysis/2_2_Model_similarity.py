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
1. 所有模型对重采样到同一区域、同一分辨率（本模型库为 1.0°）。
   若按模型对各取最粗分辨率，0.25° 的对会保留 1° 以下短波长内容
   （分歧最大的部分），与 1.0° 的对不是同一个量，矩阵不可比。
2. δlnV = (V - V̄_layer)/V̄_layer。用无量纲扰动而非 km/s 残差，
   否则深部背景速度更高会把深度趋势混进 SSIM 的对比度项。
3. 逐层按两模型合并 RMS 归一并截断到 ±3σ 后映射为非负图像——
   等价于把两张图用完全相同的对称色标出图再比较，避免"平移量任取"
   导致分数不可复现。仅消去两模型共有的振幅随深度衰减趋势，
   模型间的振幅差异仍由对比度项惩罚。
4. 高斯窗口 σ = 2.0° (≈220 km)，与被比模型的横向分辨率同量级；
   更小的窗口是在噪声尺度上比结构，会系统性压低分数。
5. 深度平均按层厚（梯形）加权，避免采样更密的深度段权重偏高。
6. 相位随机化 surrogate 给出零假设基线，用于判断得分是否显著。

同时输出结构项 (σ₁₂+C₃)/(σ₁σ₂+C₃)，把"图案是否一致"与
"振幅是否一致"分开：振幅减半时 SSIM≈0.79 而结构项≈1.00。

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
- Theiler, J. et al. (1992) Physica D 58: 77-94. 相位随机化 surrogate
- 1D 剖面定义对齐 2_1_Model_compare.py

作者: EASTASIA-FWI Team
日期: 2026-09
版本: v20.0 (1D SSIM + 2D SSIM(z)，统一网格 + 零假设基线)
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
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
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
            # 取全部模型的有效交集，保证每个模型对在同一批格点上打分
            'use_all_model_common_mask': True,
            'fill_method': 'nearest',  # 避免 nanmean 假高分
            'boundary_buffer_px': 0,
            'downsample_method': 'block_mean',
        }

        # ============ 零假设基线（相位随机化，给出"随机水平"参考）============
        self.null_test = {
            'enabled': True,
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
            # 论文图左侧马赛克显示哪个指标: '1d' 或 '2d'（深度平均）
            'matrix_metric': '1d',
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
            # SSIM(z) 曲线的深度向高斯平滑尺度 (km, σ)，设 0 关闭。
            # 只做圆角级平滑（半个采样间隔），基本不改变曲线形态
            'ssim_smooth_km': 10.0,
            # SSIM(z) 曲线上标记点的深度间隔 (km)，设 0 则每个采样层都标
            'marker_interval_km': 20.0,
            # 相位随机化零假设带。数值仍写入 similarity_summary_{feature}.csv
            # 的 ssim_2d_null_mean，需要时置 True 即可重新画出
            'show_null_band': False,
            'cmap_ssim': 'RdYlGn',
            'cmap_cwssim': 'RdYlGn',
            'discontinuities': [
                ('Moho', 40),
                ('LAB', 100),
                ('410 km', 410),
                ('660 km', 660),
            ],
            'add_coastlines': True,
            'coastline_color': 'black',
            'coastline_linewidth': 0.8,
        }

        # ============ 输出参数 ============
        self.output = {
            # 项目规范: 同时保存 jpg + pdf，均 300 dpi
            'save_formats': ['jpg', 'pdf'],
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


def layer_perturbation(
    data: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    relative: bool = True,
) -> np.ndarray:
    """
    去掉面积加权横向平均，得到该深度的横向扰动。

    绝对速度的 SSIM 会被 1D 背景主导（两模型都像 PREM，分数虚高）。
    relative=True 时进一步除以该层均值得到无量纲的 δlnV——km/s 残差在深部
    会因背景速度更高而系统性偏大，把深度趋势混进 SSIM 的对比度项。

    Args:
        data: 该深度层的绝对速度 (km/s)
        valid: 有效掩膜
        weights: 面积权重
        relative: True 返回 (V - V̄)/V̄；False 返回 V - V̄

    Returns:
        扰动场；无效点保持原值（后续由掩膜排除）
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
    if relative:
        if abs(mean_val) < 1e-9:
            return out
        out[m] = (data[m] - mean_val) / mean_val
    else:
        out[m] = data[m] - mean_val
    return out


def pooled_rms(
    p1: np.ndarray,
    p2: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    两个扰动场合并后的面积加权 RMS，作为该深度层的共同归一化尺度。

    用合并尺度（而非各自 RMS）归一，是为了保留两模型之间的振幅差异——
    那是 SSIM 对比度项该捕捉的真实分歧；被消去的只是"扰动幅度随深度衰减"
    这一两模型共有的趋势，否则深部会因振幅小而分数虚低。

    Args:
        p1, p2: 两个扰动场
        valid: 有效掩膜
        weights: 面积权重

    Returns:
        RMS 标量；有效点不足或退化时返回 nan
    """
    m = valid & np.isfinite(p1) & np.isfinite(p2)
    if m.sum() < 10:
        return float('nan')
    w = weights[m]
    wsum = float(w.sum())
    if wsum <= 0.0:
        return float('nan')
    ms = float((w * (p1[m] ** 2 + p2[m] ** 2)).sum() / (2.0 * wsum))
    return float(np.sqrt(ms)) if ms > 0 else float('nan')


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

    @staticmethod
    def _masked_average(
        field: np.ndarray,
        mask: np.ndarray,
        weights: Optional[np.ndarray],
    ) -> float:
        """掩膜内的（面积加权）平均"""
        m = mask & np.isfinite(field)
        if not m.any():
            return float('nan')
        if weights is None:
            return float(np.mean(field[m]))
        w = weights[m]
        wsum = float(w.sum())
        return float((w * field[m]).sum() / wsum) if wsum > 0 else float('nan')

    def compute(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        valid: np.ndarray,
        sigma_px: float,
        data_range: float,
        weights: Optional[np.ndarray] = None,
        stat_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, float]:
        """
        计算两张 2D 场的 SSIM（Wang et al. 2004 完整三项式）。

        输入必须已经归一化到给定动态范围内的非负区间——归一化是调用方的
        职责，本函数不再从数据里估计平移量或 L。用数据相关的分位数做平移会
        让 SSIM 的绝对值依赖于该分位数的取值，无法在不同深度/模型对之间比较。

        Args:
            img1, img2: 同形状 2D 场，已归一化（建议区间 [0, data_range]）
            valid: 有效掩膜
            sigma_px: 高斯窗口标准差（像素）
            data_range: 动态范围 L，决定稳定化常数 C1=(K1·L)², C2=(K2·L)²
            weights: 面积权重；None 时等权
            stat_mask: 参与全局平均的掩膜；None 时用 valid

        Returns:
            (全局 SSIM, 局部 SSIM 图, 全局结构项)
            结构项 = 局部相关系数 (σ₁₂+C3)/(σ₁σ₂+C3)，用于把
            "图案是否一致"与"振幅是否一致"分开诊断。无效区为 NaN。
        """
        img1 = np.asarray(img1, dtype=np.float64)
        img2 = np.asarray(img2, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)

        if img1.shape != img2.shape:
            raise ValueError(f"输入形状不一致: {img1.shape} vs {img2.shape}")
        if not valid.any() or min(img1.shape) < 3:
            return float('nan'), np.full(img1.shape, np.nan), float('nan')
        if not np.isfinite(data_range) or data_range <= 0:
            return float('nan'), np.full(img1.shape, np.nan), float('nan')

        # 无效区用最近邻外延，避免常数填充在掩膜边界制造人工阶跃
        x = fill_invalid_nearest(img1, valid)
        y = fill_invalid_nearest(img2, valid)

        sigma = max(0.5, float(sigma_px))
        c1 = (self.k1 * data_range) ** 2
        c2 = (self.k2 * data_range) ** 2
        c3 = c2 / 2.0

        mu1 = gaussian_filter(x, sigma=sigma, mode='reflect')
        mu2 = gaussian_filter(y, sigma=sigma, mode='reflect')
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = gaussian_filter(x * x, sigma=sigma, mode='reflect') - mu1_sq
        sigma2_sq = gaussian_filter(y * y, sigma=sigma, mode='reflect') - mu2_sq
        sigma12 = gaussian_filter(x * y, sigma=sigma, mode='reflect') - mu12
        sigma1_sq = np.maximum(sigma1_sq, 0.0)
        sigma2_sq = np.maximum(sigma2_sq, 0.0)

        num = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

        struct_den = np.sqrt(sigma1_sq * sigma2_sq) + c3
        struct_map = np.divide(
            sigma12 + c3, struct_den,
            out=np.zeros_like(sigma12), where=struct_den > 0,
        )

        if stat_mask is None:
            stat_mask = valid
        global_val = self._masked_average(ssim_map, stat_mask, weights)
        global_struct = self._masked_average(struct_map, stat_mask, weights)

        return global_val, np.where(valid, ssim_map, np.nan), global_struct

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
        self.logger.info("🚀 速度模型结构相似性分析器 v20.0 (1D SSIM + 2D SSIM(z))")
        self.logger.info("=" * 80)
        self._print_config_summary()

    def _print_config_summary(self) -> None:
        """打印配置摘要"""
        ss = self.config.ssim
        s2 = self.config.ssim_2d
        print("\n📋 相似性分析配置 (v20.0 — 1D SSIM + 2D SSIM(z))")
        print("-" * 60)
        print(f"目标模型: {len(self.config.models)} 个")
        for info in self.config.models.values():
            print(f"  • {info['name']}")
        print("\n🔬 1D SSIM（对齐 2_1 剖面）:")
        print("  场: 原始绝对 Vs / Vp 的横向面积加权平均 V(z)")
        print(f"  深度窗口: {ss.get('win_km')} km  → 整体 1D SSIM + 局部 SSIM(z)")
        print("\n🔬 2D SSIM(z)（逐深度切片）:")
        print(f"  场: δlnV = (V-V̄)/V̄，逐层按合并 RMS 归一至 ±{s2.get('clip_sigma')}σ")
        print(f"  窗口: 高斯 σ = {s2.get('sigma_deg')}° (≈220 km)")
        print(f"  网格: {s2.get('region_mode')} 区域 + {s2.get('resolution_mode')} 分辨率")
        print(f"  零假设: 相位随机化 × {self.config.null_test.get('n_realizations')}"
              f" ({'开' if self.config.null_test.get('enabled') else '关'})")
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

        cfg2d = self.config.ssim_2d
        self.analysis_metadata = {
            'version': 'v20.0-1D+2D-SSIM',
            'metrics': ['ssim_1d', 'ssim_2d'],
            'ssim_1d': {
                'field': 'raw absolute V(z), area-weighted lateral mean',
                'win_km': self.config.ssim.get('win_km'),
            },
            'ssim_2d': {
                'field': f"depth slice, perturbation={cfg2d.get('perturbation')}",
                'normalize': cfg2d.get('normalize'),
                'clip_sigma': cfg2d.get('clip_sigma'),
                'sigma_deg': cfg2d.get('sigma_deg'),
                'region_mode': cfg2d.get('region_mode'),
                'resolution_mode': cfg2d.get('resolution_mode'),
                'depth_average': 'trapezoidal (layer-thickness weighted)',
                'null_test': dict(self.config.null_test),
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
        构建分析网格。

        默认 region_mode='all_models' + resolution_mode='global_coarsest'：
        所有模型对共用同一区域与同一分辨率（本模型库为 1.0°），
        这样 N×N 矩阵里的每个数才是同一个量。
        旧行为（按模型对各取公共区域与最粗分辨率）保留在
        region_mode='pair' / resolution_mode='pair_coarsest'。
        """
        cfg = self.config.ssim_2d
        if cfg.get('region_mode', 'all_models') == 'all_models':
            bbox = self._common_geographic_bbox()
            lon_min, lon_max = bbox['lon_min'], bbox['lon_max']
            lat_min, lat_max = bbox['lat_min'], bbox['lat_max']
        else:
            lon_min = max(float(m1.lon.min()), float(m2.lon.min()))
            lon_max = min(float(m1.lon.max()), float(m2.lon.max()))
            lat_min = max(float(m1.lat.min()), float(m2.lat.min()))
            lat_max = min(float(m1.lat.max()), float(m2.lat.max()))

        if cfg.get('resolution_mode', 'global_coarsest') == 'global_coarsest':
            candidates = list(self.models.values()) or [m1, m2]
        else:
            candidates = [m1, m2]
        pair_res = float(max(
            v for m in candidates
            for v in m.estimate_resolution().values()
        ))

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

    def _get_ssim_sigma_px(self, resolution_deg: float) -> float:
        """
        2D SSIM 高斯窗口的像素标准差。

        以物理尺度 sigma_deg 定义（默认 2.0° ≈ 220 km），保证不同分辨率下
        比较的是同一空间尺度上的结构。1.0° 网格 → 2 px；0.25° → 8 px。
        """
        sigma_deg = float(self.config.ssim_2d.get('sigma_deg', 2.0))
        res = float(resolution_deg) if resolution_deg else 1.0
        return max(0.5, sigma_deg / res)

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        在共同地理范围内对原始绝对速度做面积加权横向平均，得到 V(z)。

        与 2_1_Model_compare.calculate_1d_profile 同一物理定义，
        仅将 nanmean 改为 cos(lat) 面积加权。同时返回加权标准差，
        用于在剖面图上标出该深度的横向变化幅度。

        Returns:
            (depth, mean, std)，均为模型原生深度网格上的 1D 数组
        """
        lon_m = (model.lon >= bbox['lon_min'] - 1e-6) & (model.lon <= bbox['lon_max'] + 1e-6)
        lat_m = (model.lat >= bbox['lat_min'] - 1e-6) & (model.lat <= bbox['lat_max'] + 1e-6)
        if not lon_m.any() or not lat_m.any():
            nan_like = np.full(model.depth.shape, np.nan)
            return model.depth.copy(), nan_like, nan_like.copy()

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

        dev2 = np.where(valid, (sub - mean[np.newaxis, np.newaxis, :]) ** 2, 0.0)
        var = np.divide(
            np.sum(dev2 * ww, axis=(0, 1)), den,
            out=np.full(den.shape, np.nan), where=den > 0,
        )
        std = np.sqrt(np.maximum(var, 0.0))
        return np.asarray(model.depth, dtype=np.float64), mean, std

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
            z_src, v_src, s_src = self._extract_1d_profile(model, feature, bbox)
            m = np.isfinite(z_src) & np.isfinite(v_src)
            if m.sum() < 3:
                self.logger.warning(f"  ⚠️ {model.name} {feature} 1D 剖面有效点不足")
                continue
            out_of_range = (
                (target_z < float(z_src[m].min()) - 1e-6)
                | (target_z > float(z_src[m].max()) + 1e-6)
            )
            v_grid = np.interp(target_z, z_src[m], v_src[m], left=np.nan, right=np.nan)
            v_grid[out_of_range] = np.nan
            s_grid = np.interp(
                target_z, z_src[m], np.nan_to_num(s_src[m], nan=0.0),
                left=np.nan, right=np.nan,
            )
            s_grid[out_of_range] = np.nan
            self.profiles[feature][key] = {
                'depth': target_z, 'value': v_grid, 'std': s_grid,
            }
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
        2D SSIM(z)：逐深度切片在归一化的横向扰动场上计算 SSIM。

        与 1D SSIM 分工明确——1D 量背景剖面，本指标量同一深度的横向结构。
        每层流程：
        1. 重采样到全局统一分析网格（所有模型对一致）
        2. δlnV = (V - V̄_layer)/V̄_layer
        3. 逐层用两模型合并 RMS 归一，按 ±clip_sigma 映射到非负区间
        4. σ = sigma_deg 的高斯窗口下算 SSIM，面积加权平均
        5. 若开启零假设，用相位随机化 surrogate 给出随机水平

        Returns:
            DataFrame: depth_km, ssim, ssim_struct, ssim_null, rms1, rms2, ...
        """
        m1 = self.models[m1_key]
        m2 = self.models[m2_key]
        coverage = self._get_common_coverage(m1, m2)
        matched_depths = self._matched_depths(m1, m2)

        if not matched_depths:
            self.logger.warning(f"  ⚠️ {m1.name} vs {m2.name}: 无匹配深度层")
            return pd.DataFrame()

        cfg2d = self.config.ssim_2d
        res = float(coverage['resolution_deg'])
        sigma_px = self._get_ssim_sigma_px(res)
        clip_sigma = float(cfg2d.get('clip_sigma', 3.0))
        data_range = 2.0 * clip_sigma
        use_area = bool(cfg2d.get('use_area_weights', True))
        relative = cfg2d.get('perturbation', 'relative') == 'relative'
        normalize = cfg2d.get('normalize', 'pooled_rms')

        n_null = (
            int(self.config.null_test.get('n_realizations', 20))
            if self.config.null_test.get('enabled', False) else 0
        )
        do_r = bool(self.config.companion.get('compute_signed_correlation', False))
        do_sign = bool(self.config.companion.get('compute_sign_agreement', False))
        sign_pct = float(self.config.companion.get('sign_threshold_percentile', 50.0))

        self.logger.info(
            f"  🗺️ 2D SSIM(z): {m1.name} vs {m2.name}  {feature.upper()}  "
            f"({len(matched_depths)} 层, {res:.2f}°, σ={sigma_px:.1f}px"
            f"{f', null×{n_null}' if n_null else ''})"
        )

        data1_3d = m1.get_parameter(feature)
        data2_3d = m2.get_parameter(feature)

        rows: List[Dict[str, Any]] = []
        for d_idx1, d_idx2, depth_km in tqdm(matched_depths, desc="  深度层", leave=False):
            s1 = self._resample_slice(m1, data1_3d, d_idx1, coverage)
            s2 = self._resample_slice(m2, data2_3d, d_idx2, coverage)
            valid, weights, stat_mask = self._build_masks(
                s1, s2, feature, depth_km, coverage,
            )
            area_w = weights if use_area else None

            p1 = layer_perturbation(s1, valid, weights, relative=relative)
            p2 = layer_perturbation(s2, valid, weights, relative=relative)

            # 归一化到统一的无量纲尺度，等价于用同一对称色标出两张图
            if normalize == 'pooled_rms':
                scale = pooled_rms(p1, p2, valid, weights)
            else:
                scale = 1.0
            if not np.isfinite(scale) or scale <= 0:
                continue
            u1 = np.clip(p1 / scale, -clip_sigma, clip_sigma) + clip_sigma
            u2 = np.clip(p2 / scale, -clip_sigma, clip_sigma) + clip_sigma

            ssim_val, _, struct_val = self.ssim2d_calc.compute(
                u1, u2, valid=valid, sigma_px=sigma_px, data_range=data_range,
                weights=area_w, stat_mask=stat_mask,
            )

            row: Dict[str, Any] = {
                'depth_km': depth_km,
                'ssim': ssim_val,
                'ssim_struct': struct_val,
                'rms1': float(np.sqrt(np.nanmean(p1[valid] ** 2))) if valid.any() else np.nan,
                'rms2': float(np.sqrt(np.nanmean(p2[valid] ** 2))) if valid.any() else np.nan,
                'valid_frac': float(valid.mean()) if valid.size else 0.0,
                'analysis_resolution_deg': res,
                'ssim_sigma_px': sigma_px,
            }

            if n_null:
                base = fill_invalid_nearest(u2 - clip_sigma, valid)
                null_vals = []
                for _ in range(n_null):
                    surrogate = phase_randomize(base, self._rng) + clip_sigma
                    nv, _, _ = self.ssim2d_calc.compute(
                        u1, surrogate, valid=valid, sigma_px=sigma_px,
                        data_range=data_range, weights=area_w, stat_mask=stat_mask,
                    )
                    if np.isfinite(nv):
                        null_vals.append(nv)
                if null_vals:
                    row['ssim_null'] = float(np.mean(null_vals))
                    row['ssim_null_std'] = float(np.std(null_vals))

            if do_r:
                row['r_signed'] = weighted_pearson(p1, p2, weights, valid)
            if do_sign:
                row['sign_agreement'] = sign_agreement(
                    p1, p2, weights, valid, sign_pct,
                )
            rows.append(row)

        df = pd.DataFrame(rows)
        if len(df):
            msg = (
                f"      2D SSIM: [{df['ssim'].min():.3f}, {df['ssim'].max():.3f}]  "
                f"mean={self._depth_average(df, 'ssim'):.3f}"
            )
            if 'ssim_null' in df.columns:
                msg += f"  (null≈{df['ssim_null'].mean():.3f})"
            self.logger.info(msg)
        return df

    @staticmethod
    def _depth_average(df: pd.DataFrame, column: str) -> float:
        """
        按层厚（梯形）加权的深度平均。

        不同模型对的公共深度层数与间隔不同（51 层 @20 km vs 101 层 @10 km），
        算术平均会让采样更密的深度段获得更高权重。
        """
        if column not in df.columns or not len(df):
            return float('nan')
        z = df['depth_km'].to_numpy(dtype=float)
        v = df[column].to_numpy(dtype=float)
        m = np.isfinite(z) & np.isfinite(v)
        if m.sum() < 2:
            return float(np.nanmean(v)) if m.any() else float('nan')
        z, v = z[m], v[m]
        order = np.argsort(z)
        z, v = z[order], v[order]
        span = z[-1] - z[0]
        if span <= 0:
            return float(np.mean(v))
        return float(np.trapz(v, z) / span)

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
                summary: Dict[str, Any] = {
                    'pair': pair_name,
                    'feature': feature,
                    'ssim_2d_mean': self._depth_average(df, 'ssim'),
                    'ssim_2d_std': (
                        float(df['ssim'].std()) if 'ssim' in df.columns and len(df)
                        else float('nan')
                    ),
                    'ssim_2d_struct_mean': self._depth_average(df, 'ssim_struct'),
                }
                if 'ssim_null' in df.columns:
                    summary['ssim_2d_null_mean'] = self._depth_average(df, 'ssim_null')
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
                    std = prof.get('std', np.full_like(prof['value'], np.nan))
                    for z, v, s in zip(prof['depth'], prof['value'], std):
                        prof_rows.append({
                            'model': name, 'model_key': key,
                            'depth_km': float(z),
                            feature: float(v) if np.isfinite(v) else np.nan,
                            f'{feature}_std': float(s) if np.isfinite(s) else np.nan,
                        })
                if prof_rows:
                    prof_file = feat_dir / f'1d_profiles_{feature}.csv'
                    pd.DataFrame(prof_rows).to_csv(
                        prof_file, index=False, float_format='%.6f',
                    )
                    self.logger.info(f"  ✅ {prof_file.name}")

        self.logger.info("✅ 所有结果已保存")


class SimilarityVisualization:
    """
    结构相似性可视化 (v20.0)。

    产出论文图：单参数版 (a) 1D 平均剖面 + (b) 2D SSIM(z)；
    以及 Vs/Vp 合并版（线型区分参数，SSIM(z) 按参数分栏）。

    两个面板共用 config.models 的配色，SSIM(z) 曲线按模型对顺序取用同一组色，
    并各配一种标记形状。注意色号在 (a) 指模型、在 SSIM(z) 指模型对，二者不构成
    对应关系，模型对的身份只由图例给出。
    """

    # 仅在模型配色不可用时兜底
    _PAIR_COLORS = ['#D1495B', '#00798C', '#EDAE49', '#8E7DBE', '#66A182']
    # 模型对的标记形状，曲线重叠时不靠颜色也能区分
    _PAIR_MARKERS = ['o', 's', '^', 'D', 'v']
    _FEAT_STYLES = {'vs': '-', 'vp': (0, (5, 2))}

    def __init__(self, analyzer: VelocityModelSimilarity):
        self.analyzer = analyzer
        self.config = analyzer.config
        self.logger = analyzer.logger
        self.figures_dir = analyzer.figures_dir

        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.4)

    @staticmethod
    def _feat_label(feature: str) -> str:
        """特征名转地震学惯用写法: vs → Vs, vp → Vp"""
        return {'vs': 'Vs', 'vp': 'Vp'}.get(feature.lower(), feature.upper())

    @staticmethod
    def _feat_math(feature: str) -> str:
        """特征名转数学排版: vs → $V_S$"""
        sub = feature[1].upper() if len(feature) > 1 else feature.upper()
        return rf'$V_{sub}$'

    def _model_label(self, model_key: str) -> str:
        """模型名附原生网格间距，如 'SinoScope1.0  (1.0°)'"""
        model = self.analyzer.models.get(model_key)
        if model is None:
            return model_key
        try:
            res = model.estimate_resolution()
            r_lon, r_lat = float(res['lon']), float(res['lat'])
        except Exception:
            return model.name

        def fmt(r: float) -> str:
            """至少保留一位小数，让 1.0° 与 0.25° 在图例里对齐"""
            s = f'{r:.2f}'.rstrip('0')
            return f'{s}0' if s.endswith('.') else s

        grid = (f'{fmt(r_lon)}°' if abs(r_lon - r_lat) < 1e-6
                else f'{fmt(r_lon)}°×{fmt(r_lat)}°')
        return f'{model.name}  ({grid})'

    def _marker_step(self, depth: np.ndarray) -> int:
        """
        标记间隔换算成采样点步长，由 visualization['marker_interval_km'] 给定。

        当前深度采样即 20 km，故步长为 1（每层一个标记）；若把
        analysis['depth_grid_step_km'] 调细，标记密度不会跟着膨胀。
        """
        want_km = float(self.config.visualization.get('marker_interval_km', 0.0))
        if want_km <= 0 or len(depth) < 2:
            return 1
        dz = float(np.median(np.diff(depth)))
        if not np.isfinite(dz) or dz <= 0:
            return 1
        return max(1, int(round(want_km / dz)))

    def _smooth_depth(self, depth: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        沿深度做高斯平滑，σ 由 visualization['ssim_smooth_km'] 给定。

        逐层独立打分含估计噪声，而相邻深度的横向结构本身是相关的，因此平滑
        是合理的展示方式。用掩膜归一化实现：无效层不参与加权，也不被填补，
        原有的深度缺口在图上仍然保持为断开。

        Args:
            depth: 深度序列 (km)，需单调
            values: 同长度的待平滑值，允许含 NaN

        Returns:
            平滑后的序列；σ<=0 或采样不足时原样返回
        """
        sigma_km = float(self.config.visualization.get('ssim_smooth_km', 0.0))
        valid = np.isfinite(values)
        if sigma_km <= 0 or valid.sum() < 3 or len(depth) < 3:
            return values

        dz = float(np.median(np.diff(depth)))
        if not np.isfinite(dz) or dz <= 0:
            return values
        sigma_px = sigma_km / dz
        if sigma_px < 0.3:
            return values

        num = gaussian_filter1d(np.where(valid, values, 0.0), sigma_px,
                                mode='nearest')
        den = gaussian_filter1d(valid.astype(float), sigma_px, mode='nearest')
        out = np.where(den > 1e-3, num / np.maximum(den, 1e-12), np.nan)
        out[~valid] = np.nan
        return out

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

    # ---------- 论文图: 1D SSIM 矩阵 + 2D SSIM(z) 曲线 ----------

    def _build_matrix(self, feature: str, metric: str) -> Optional[np.ndarray]:
        """
        构建 N×N 相似性矩阵。

        Args:
            feature: 'vs' / 'vp'
            metric: '1d' 取整体 1D SSIM；'2d' 取 2D SSIM(z) 的层厚加权深度平均

        Returns:
            对称矩阵（对角为 1）；无可用结果时返回 None
        """
        model_keys = list(self.analyzer.models.keys())
        matrix = np.ones((len(model_keys), len(model_keys)))
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
                val = self.analyzer._depth_average(df, 'ssim')
            if not np.isfinite(val):
                continue
            matrix[i, j] = val
            matrix[j, i] = val
            found = True

        return matrix if found else None

    def _draw_matrix(
        self,
        ax,
        matrix: np.ndarray,
        cbar_label: str,
        cmap: str = 'YlGnBu',
    ) -> None:
        """
        绘制上三角相似性马赛克图。

        矩阵对称，只画上三角即可无损呈现全部模型对；对角线为自比较恒等于 1，
        留空——若参与配色会占据色标高端，把关心的差异压缩到几乎同色。
        进一步去掉全空的首列与末行，使每个坐标标签都对应到实际单元格。
        色标横放在空出的左下角三角区。
        """
        names = [self.analyzer.models[k].name for k in self.analyzer.models]
        n = len(names)
        if n < 2:
            return

        # 紧凑上三角: 行取 names[:-1]，列取 names[1:]，保留 i <= j 的单元
        sub = matrix[:-1, 1:]
        m = n - 1
        keep = np.triu(np.ones((m, m), dtype=bool), k=0)
        vmin = np.floor(np.nanmin(sub[keep]) * 20.0) / 20.0
        vmax = 1.0

        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(alpha=0.0)
        im = ax.imshow(
            np.ma.masked_array(sub, mask=~keep),
            cmap=cmap_obj, vmin=vmin, vmax=vmax, origin='upper', aspect='auto',
        )

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        for i in range(m):
            for j in range(i, m):
                val = sub[i, j]
                # 深色底用白字，保证标注在任何色阶上都可读
                txt_color = 'white' if norm(val) > 0.55 else '#1A1A1A'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=15, fontweight='bold', color=txt_color)
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor='white', linewidth=2.5, zorder=3,
                ))

        ax.set_xticks(np.arange(m))
        ax.set_yticks(np.arange(m))
        ax.set_xticklabels(names[1:], rotation=25, ha='left', fontsize=11)
        ax.set_yticklabels(names[:-1], fontsize=11)
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(which='both', length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # 色标用 inset 承载（不抢占父轴空间，轴框才能与相邻面板严格对齐），
        # 横放在左下角的空白三角区
        cax = ax.inset_axes([0.03, 0.15, 0.40, 0.045])
        cbar = ax.figure.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label(cbar_label, fontsize=10.5, labelpad=2)
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.xaxis.set_label_position('bottom')
        cbar.outline.set_visible(False)

    def _decorate_depth_axis(self, ax) -> None:
        """统一各面板的深度轴：反向、间断面虚线、次刻度、网格与边框。"""
        ax.set_ylim(
            float(self.config.analysis['depth_range_km'][1]),
            float(self.config.analysis['depth_range_km'][0]),
        )
        for _, depth in self.config.visualization['discontinuities']:
            ax.axhline(depth, color='#9A9A9A', linestyle=(0, (4, 3)),
                       linewidth=0.8, zorder=2)

        ax.yaxis.set_minor_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, axis='x', alpha=0.25, linestyle='-', linewidth=0.6)
        ax.grid(False, axis='y')
        ax.grid(False, which='minor')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.tick_params(which='minor', length=2.5)

    def _add_discontinuity_labels(self, ax) -> None:
        """
        在最右面板外侧挂一根只带间断面刻度的深度轴。

        名称放在图外，各面板内部就不必再写文字——原先挤在最左面板左缘的
        Moho/LAB 标签紧贴顶边且会压到剖面曲线。所有面板共享同一深度轴，
        标注一次即可。
        """
        z_lo, z_hi = ax.get_ylim()
        ax_r = ax.twinx()
        ax_r.set_ylim(z_lo, z_hi)
        depths = [d for _, d in self.config.visualization['discontinuities']]
        names = [n for n, _ in self.config.visualization['discontinuities']]
        ax_r.set_yticks(depths)
        ax_r.set_yticklabels(names, fontsize=9, color='#6E6E6E')
        ax_r.tick_params(axis='y', length=3, color='#9A9A9A', pad=2)
        ax_r.grid(False)
        for side in ('top', 'bottom', 'left', 'right'):
            ax_r.spines[side].set_visible(False)

    def _draw_profiles(self, ax, feature: str) -> int:
        """
        绘制共同区域的 1D 平均速度剖面 V(z) 与 ±1σ 横向变化带。

        剖面定义与 2_1_Model_compare.calculate_1d_profile 一致（仅横向平均
        改为 cos(lat) 面积加权）。阴影是该深度层的横向标准差——它与 1D SSIM
        高、2D SSIM 低这一组合互为印证：平均剖面几乎重合，但各自的横向
        变化范围很宽，说明分歧藏在横向图案里而非背景。

        图例带各模型的原生网格间距：SinoScope1.0 的 1.0° 决定了 2D SSIM 的
        公共分析网格，也是它的两个模型对得分偏低的一部分原因，标出来读者才
        不会把分辨率差异误读成结构分歧。

        Returns:
            绘制的模型数量
        """
        profs = self.analyzer.profiles.get(feature, {})
        n_plotted = 0
        for key, prof in profs.items():
            model = self.analyzer.models.get(key)
            name = self._model_label(key)
            color = model.color if model else '#333333'
            z = np.asarray(prof['depth'], dtype=float)
            v = np.asarray(prof['value'], dtype=float)
            m = np.isfinite(z) & np.isfinite(v)
            if not m.any():
                continue
            ax.plot(v[m], z[m], color=color, linewidth=1.2,
                    label=name, zorder=3)

            std = np.asarray(prof.get('std', np.full_like(v, np.nan)), dtype=float)
            ms = m & np.isfinite(std)
            if ms.any():
                ax.fill_betweenx(
                    z[ms], v[ms] - std[ms], v[ms] + std[ms],
                    color=color, alpha=0.20, linewidth=0, zorder=1,
                )
            n_plotted += 1

        if not n_plotted:
            return 0

        sub = feature[1].upper() if len(feature) > 1 else feature.upper()
        ax.set_xlabel(rf'$V_{sub}$ (km/s)', fontsize=12)
        ax.set_ylabel('Depth (km)', fontsize=12)
        self._decorate_depth_axis(ax)
        ax.legend(fontsize=9.5, loc='lower left', handlelength=1.6)
        return n_plotted

    def _draw_ssim_curves(self, ax, feature: str) -> int:
        """
        绘制 2D SSIM(z)：逐深度横向结构相似性曲线与相位随机化零假设带。

        不画 1D 剖面的局部 SSIM(z)：V(z) 是单调光滑曲线，没有可供结构项
        发挥的图案，局部统计量退化——亮度项恒为 1（对绝对速度差异不敏感），
        梯度平缓的深度段又被稳定化常数 C₂ 支配而机械趋于 1，曲线不可解读。
        整体 1D SSIM 仍保留在 similarity_summary_{feature}.csv 中。

        Returns:
            绘制的模型对数量
        """
        suffix = f'_{feature}_depth'
        keys = sorted(k for k in self.analyzer.results if k.endswith(suffix))

        # 直接取 (a) 的模型配色，两个面板色系统一
        model_colors = [m.color for m in self.analyzer.models.values()]
        if not model_colors:
            model_colors = self._PAIR_COLORS

        show_null = bool(self.config.visualization.get('show_null_band', False))
        null_depth: List[np.ndarray] = []
        null_value: List[np.ndarray] = []
        handles: List[Any] = []
        labels: List[str] = []
        n_plotted = 0
        v_max = 0.0

        for idx, key in enumerate(keys):
            df = self.analyzer.results[key]
            if 'ssim' not in df.columns or not len(df):
                continue
            pair = key[:-len(suffix)]
            color = model_colors[idx % len(model_colors)]
            marker = self._PAIR_MARKERS[idx % len(self._PAIR_MARKERS)]
            z = df['depth_km'].to_numpy(dtype=float)
            s_raw = df['ssim'].to_numpy(dtype=float)
            s = self._smooth_depth(z, s_raw)

            # 白色描边让曲线在交叉处仍可分辨
            ax.plot(s, z, color=color, linewidth=1.4, zorder=3, path_effects=[
                pe.Stroke(linewidth=2.8, foreground='white'), pe.Normal(),
            ])
            # 标记取平滑后的同一组值，严格落在曲线上——画在原始值上会与曲线
            # 错开，读者无从判断该信哪一个。标记形状按模型对区分，
            # 曲线重叠段不靠颜色也能分开
            step = self._marker_step(z)
            ax.plot(s[::step], z[::step], linestyle='none', marker=marker,
                    markersize=2.8, color=color, markeredgecolor='white',
                    markeredgewidth=0.35, zorder=5)

            agg = self.analyzer._depth_average(df, 'ssim')
            handles.append(Line2D(
                [], [], color=color, linewidth=1.4, marker=marker,
                markersize=4.0, markeredgecolor='white', markeredgewidth=0.4,
            ))
            labels.append(
                f'{self._pair_label(pair)}'
                + (f'  ({agg:.2f})' if np.isfinite(agg) else '')
            )
            v_max = max(v_max, float(np.nanmax(s_raw)))
            if show_null and 'ssim_null' in df.columns:
                null_depth.append(z)
                null_value.append(df['ssim_null'].to_numpy(dtype=float))
            n_plotted += 1

        if not n_plotted:
            return 0

        ax.set_xlim(0.0, min(1.0, max(0.7, np.ceil(v_max * 10.0) / 10.0 + 0.05)))

        # 零假设包络：保留功率谱、随机化相位的 surrogate 得分上界。
        # 落在带内即与"结构无关但粗糙度相同"不可区分。
        if null_depth:
            z_ref = max(null_depth, key=len)
            stack = np.vstack([
                np.interp(z_ref, z, v) for z, v in zip(null_depth, null_value)
            ])
            ax.fill_betweenx(
                z_ref, ax.get_xlim()[0], np.nanmax(stack, axis=0),
                color='#B0B0B0', alpha=0.35, linewidth=0, zorder=1,
            )
            handles.append(Patch(facecolor='#B0B0B0', alpha=0.35))
            labels.append('Null (phase-randomized)')

        # 不写成 "Depth-wise SSIM"：1D 剖面 SSIM 恰恰是沿深度方向算的，
        # 只说 depth-wise 会与它混淆，保留 2D 才点明每个测量做在一张水平切片上
        ax.set_xlabel(
            f'Depth-wise 2D SSIM  ({self._feat_math(feature)})',
            fontsize=12,
        )
        self._decorate_depth_axis(ax)

        # 括号内是层厚加权深度平均值，平滑尺度见 ssim_smooth_km——两者都由
        # 图注说明，图内不再放标题文字
        leg = ax.legend(
            handles=handles, labels=labels, fontsize=9.5, loc='lower right',
            handlelength=2.2, handletextpad=0.7,
        )
        leg.set_frame_on(True)
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_alpha(0.82)
        leg.get_frame().set_linewidth(0)
        return n_plotted

    def plot_ssim_figure(self, feature: str = 'vs') -> Optional[plt.Figure]:
        """
        论文用单幅图，两个面板共享深度轴：

        (a) 共同区域 1D 平均剖面 V(z) 与 ±1σ 横向变化带
        (b) 2D SSIM(z)——逐深度横向结构相似性，图例括号内为深度平均值

        整体 1D SSIM 只进 similarity_summary_{feature}.csv，不进图：3 个模型
        只有 3 个数，与 (a) 的"剖面几乎重合"是同一件事的两种表述，画成矩阵
        反而容易与 (b) 的逐层横向得分混读。
        """
        has_prof = bool(self.analyzer.profiles.get(feature))
        has_2d = any(k.endswith(f'_{feature}_depth') for k in self.analyzer.results)
        if not (has_prof or has_2d):
            return None

        self.logger.info(f"\n🎨 论文图 (剖面 + 2D SSIM): {feature.upper()}")

        rc = {
            'font.size': 11,
            'axes.linewidth': 0.9,
            'axes.edgecolor': '#4A4A4A',
            'xtick.color': '#4A4A4A',
            'ytick.color': '#4A4A4A',
            'axes.labelcolor': '#1A1A1A',
            'legend.frameon': False,
        }
        with plt.rc_context(rc):
            fig, axes = plt.subplots(
                1, 2, figsize=(10.6, 5.6), sharey=True,
                gridspec_kw={'wspace': 0.07, 'width_ratios': [0.92, 1.0]},
            )
            drawn = [
                self._draw_profiles(axes[0], feature),
                self._draw_ssim_curves(axes[1], feature),
            ]

            for ax, tag, ok in zip(axes, ('a', 'b'), drawn):
                if not ok:
                    ax.set_visible(False)
                    continue
                ax.annotate(
                    f'({tag})', xy=(0, 1), xycoords='axes fraction',
                    xytext=(2, 14), textcoords='offset points',
                    fontsize=13, fontweight='bold', va='top', ha='left',
                )

            if drawn[-1]:
                self._add_discontinuity_labels(axes[-1])

            self._save_figure(fig, f'ssim_{feature}')
        return fig

    # ---------- 论文图: Vs 与 Vp 合并版 ----------

    def _draw_profiles_joint(self, ax, features: Tuple[str, ...]) -> int:
        """
        Vs 与 Vp 平均剖面共用同一速度轴（同为 km/s，不需要第二套刻度）。

        两族曲线在图面上天然分离：量值接近的地壳 Vp（~5-6 km/s）出现在图顶，
        而同样量值的下地幔 Vs 出现在图底，深度轴把它们拉开，无需人为偏移。
        参数用线型区分（实线 Vs / 虚线 Vp），模型用颜色区分。

        Returns:
            绘制的曲线数量
        """
        n_plotted = 0
        anchors: Dict[str, float] = {}
        z_ref = 0.62 * float(self.config.analysis['depth_range_km'][1])

        for feature in features:
            ls = self._FEAT_STYLES.get(feature, '-')
            ref_vals: List[float] = []
            for key, prof in self.analyzer.profiles.get(feature, {}).items():
                model = self.analyzer.models.get(key)
                color = model.color if model else '#333333'
                z = np.asarray(prof['depth'], dtype=float)
                v = np.asarray(prof['value'], dtype=float)
                m = np.isfinite(z) & np.isfinite(v)
                if not m.any():
                    continue
                ax.plot(v[m], z[m], color=color, linewidth=1.2,
                        linestyle=ls, zorder=3)

                std = np.asarray(prof.get('std', np.full_like(v, np.nan)),
                                 dtype=float)
                ms = m & np.isfinite(std)
                if ms.any():
                    ax.fill_betweenx(
                        z[ms], v[ms] - std[ms], v[ms] + std[ms],
                        color=color, alpha=0.18, linewidth=0, zorder=1,
                    )
                ref_vals.append(float(np.interp(z_ref, z[m], v[m])))
                n_plotted += 1
            if ref_vals:
                anchors[feature] = float(np.mean(ref_vals))

        if not n_plotted:
            return 0

        ax.set_xlabel('Velocity (km/s)', fontsize=12)
        ax.set_ylabel('Depth (km)', fontsize=12)
        self._decorate_depth_axis(ax)

        # 参数名直接标注在各自曲线族旁，指向两族之间的空白，比图例更好读
        x0, x1 = ax.get_xlim()
        offset = 0.05 * (x1 - x0)
        order = sorted(anchors, key=lambda f: anchors[f])
        for i, feature in enumerate(order):
            to_right = (i == 0)
            ax.text(
                anchors[feature] + (offset if to_right else -offset), z_ref,
                self._feat_math(feature),
                ha='left' if to_right else 'right', va='center',
                fontsize=13, color='#4A4A4A', zorder=4,
            )

        handles = [
            Line2D([], [], color=self.analyzer.models[k].color,
                   linewidth=1.2, label=self._model_label(k))
            for k in self.analyzer.models
        ]
        ax.legend(handles=handles, fontsize=9.5, loc='lower left',
                  handlelength=1.6)
        return n_plotted

    def plot_ssim_joint_figure(
        self, features: Tuple[str, ...] = ('vs', 'vp')
    ) -> Optional[plt.Figure]:
        """
        Vs 与 Vp 合并的论文图，各面板共享深度轴：

        (a) 共同区域 1D 平均剖面与 ±1σ 横向变化带（Vs 与 Vp 同轴，km/s）
        (b)(c) 逐深度 2D SSIM(z) 与相位随机化零假设带，按参数分栏

        SSIM(z) 不把两个参数叠在同一横轴上：6 条本身就逐层抖动的曲线全落在
        0.1-0.4 的窄带内，其中两个 SinoScope 模型对几乎重合，再靠线型区分参数
        只会更乱。分栏后每栏 3 条实线，两栏强制共用横轴上限，Vs 与 Vp 的高低
        仍可直接对读。

        Vp 与 Vs 并列是有信息量的：三个模型都给出原生 vpv/vph，δlnVp 与
        δlnVs 的逐层相关只有 0.25-0.98、Vp/Vs 比值在空间上变化，Vp 不是 Vs
        的确定性换算。但 EARA2024 与 FWEA23 的 δlnVp/δlnVs 斜率接近经验值
        0.5-0.7，说明其 Vp 受 Vs 约束较强——因此 Vp 得分低于 Vs 不能直接读作
        "Vp 结构分歧更大"，其中混入了各模型 Vp-Vs 耦合假设本身的差异。

        Args:
            features: 参与合并的参数，(a) 中按线型 _FEAT_STYLES 区分，
                SSIM(z) 各占一栏

        Returns:
            图对象；无可用数据时返回 None
        """
        feats = tuple(
            f for f in features
            if self.analyzer.profiles.get(f)
            or any(k.endswith(f'_{f}_depth') for k in self.analyzer.results)
        )
        if len(feats) < 2:
            return None

        self.logger.info(
            f"\n🎨 论文图 (剖面 + 2D SSIM, 合并): "
            f"{' + '.join(self._feat_label(f) for f in feats)}"
        )

        rc = {
            'font.size': 11,
            'axes.linewidth': 0.9,
            'axes.edgecolor': '#4A4A4A',
            'xtick.color': '#4A4A4A',
            'ytick.color': '#4A4A4A',
            'axes.labelcolor': '#1A1A1A',
            'legend.frameon': False,
        }
        with plt.rc_context(rc):
            fig, axes = plt.subplots(
                1, 1 + len(feats), figsize=(4.9 + 3.7 * len(feats), 5.6),
                sharey=True,
                gridspec_kw={
                    'wspace': 0.09,
                    # (a) 横跨 Vs 与 Vp 两个速度量程，需要比单参数栏更宽
                    'width_ratios': [1.32] + [1.0] * len(feats),
                },
            )
            drawn = [self._draw_profiles_joint(axes[0], feats)]
            for ax, feature in zip(axes[1:], feats):
                drawn.append(self._draw_ssim_curves(ax, feature))

            # 各 SSIM 栏强制同一横轴上限，跨参数的高低才能直接目视对比
            ssim_axes = [ax for ax, ok in zip(axes[1:], drawn[1:]) if ok]
            if ssim_axes:
                x_max = max(ax.get_xlim()[1] for ax in ssim_axes)
                for ax in ssim_axes:
                    ax.set_xlim(0.0, x_max)

            for ax, tag, ok in zip(axes, 'abcdef', drawn):
                if not ok:
                    ax.set_visible(False)
                    continue
                ax.annotate(
                    f'({tag})', xy=(0, 1), xycoords='axes fraction',
                    xytext=(2, 14), textcoords='offset points',
                    fontsize=13, fontweight='bold', va='top', ha='left',
                )

            if drawn[-1]:
                self._add_discontinuity_labels(axes[-1])

            self._save_figure(fig, f"ssim_{'_'.join(feats)}")
        return fig

    def plot_ssim_matrix(self, feature: str = 'vs') -> Optional[plt.Figure]:
        """
        N×N 上三角相似性矩阵，单独出图。

        当前 3 个模型只有 3 个数，信息量不足以在主图里占版面（数值已在
        similarity_summary_{feature}.csv 与主图图例中给出），因此 main()
        不调用；模型库扩展到多模型后这张矩阵才值得单独成图。
        """
        metric = self.config.visualization.get('matrix_metric', '1d')
        matrix = self._build_matrix(feature, metric)
        if matrix is None:
            return None

        self.logger.info(f"\n🎨 相似性矩阵: {feature.upper()}")
        feat = self._feat_label(feature)
        label = (f'1D SSIM  ({feat} profile)' if metric == '1d'
                 else f'2D SSIM  ({feat} lateral structure)')

        with plt.rc_context({'legend.frameon': False}):
            fig, ax = plt.subplots(figsize=(6.4, 5.8))
            self._draw_matrix(ax, matrix, label)
            self._save_figure(fig, f'ssim_matrix_{metric}_{feature}')
        return fig

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

# ==================== 主函数 ====================

def main():
    """主函数 — 1D SSIM + 2D SSIM(z)"""
    print("\n" + "=" * 80)
    print("🚀 EASTASIA-FWI 速度模型结构相似性分析 (v20.0)")
    print("   框架: Wang et al. (2004) SSIM — 1D 剖面 + 2D 逐深度切片")
    print("=" * 80)
    print("\n🎯 主路径:")
    print("  • 1D SSIM: 共同区域横向平均的原始 Vs / Vp 剖面 V(z)")
    print("  • 2D SSIM(z): 统一网格上 δlnV 的横向结构，含零假设基线")
    print("  • 产品: 单幅论文图 2-2_ssim_{vs,vp}.jpg/pdf")
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
            visualizer.plot_ssim_figure(feature)
            if analyzer.config.analysis.get('compute_cwssim_spatial'):
                visualizer.plot_cwssim_spatial(feature)

        # Vs + Vp 合并版（线型区分参数），供正文单图使用
        joint = tuple(f for f in ('vs', 'vp') if f in features_done)
        if len(joint) > 1:
            print(f"\n  📊 {' + '.join(f.upper() for f in joint)} (合并)...")
            visualizer.plot_ssim_joint_figure(joint)

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
