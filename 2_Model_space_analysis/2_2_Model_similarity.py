"""
2_2_Model_similarity.py:
多尺度速度模型相似性分析模块
================================================================

功能描述:
----
基于绝对速度值的多尺度结构相似性量化。核心使用 CW-SSIM 指标衡量速度模型的
空间结构一致性，结合最近邻采样策略避免空间域插值操作。

核心功能:
----
1. ✅ CW-SSIM (Complex Wavelet SSIM): 复小波域多尺度结构相似性
   - 天然抗平移/缩放畸变 (Wang & Simoncelli, 2005)
   - 独立归一化（min-max / 稳健百分位），仅比较空间模式
   - 深度自适应尺度权重，符合地震学分层特征
   - 降采样到统一分析分辨率，保证跨模型可比性
   - 局部滑动窗口（均匀/高斯），产生空间分布图

使用方法:
----
```python
from 2_Model_space_analysis.2_2_Model_similarity import (
    ModelSimilarityConfig, VelocityModelSimilarity
)

config = ModelSimilarityConfig()
analyzer = VelocityModelSimilarity(config)
analyzer.load_models()
analyzer.compare_all_models()
analyzer.save_results()
```

配置说明:
----
通过 ModelSimilarityConfig 类配置参数:
- cwssim: 小波分解层数、稳定常数 K、分析分辨率、深度自适应权重
- analysis: 目标特征、深度范围、计算开关
- normalize_method: 'minmax' | 'percentile'（稳健归一化）

输出文件:
----
- similarity_summary_{feature}.csv: 模型对 CW-SSIM 摘要
- depth_cwssim_{pair}.csv: 深度逐层 CW-SSIM
- 2-2_depth_cwssim.{png,pdf}: CW-SSIM 深度曲线
- 2-2_cwssim_heatmap.{png,pdf}: N×N CW-SSIM 矩阵
- 2-2_cwssim_spatial.{png,pdf}: CW-SSIM 空间分布图
- 2-2_similarity_overview.{png,pdf}: CW-SSIM 综合面板

科学原理:
----
- CW-SSIM 基于 Dual-Tree Complex Wavelet Transform (DT-CWT)，
  通过分析复小波系数的相位一致性衡量结构相似性。
  核心公式: CW-SSIM = (2|Σc_x c_y*| + K) / (Σ|c_x|² + Σ|c_y|² + K)
- Wang & Simoncelli (2005) ICASSP: 平移不变性、复小波域相位比较
- Sampat et al. (2009) IEEE TIP: 完整多尺度、方向子带 CW-SSIM
- Huang et al. (2024) Earthquake Science 37(6): 514-528:
  中国大陆岩石圈速度模型结构相似性分析。DOI: 10.1016/j.eqs.2024.05.004

空间分布计算方式 (Huang 论文方法):
----
采用 Huang et al. (2024) 的「空间域滑动窗口」：
  - 每个网格点取 local_win_deg°×local_win_deg° 物理 patch，单独计算该 patch 的 CW-SSIM
  - 物理尺度明确（如 6°≈600 km），参数由 config.cwssim.local_win_deg 控制
  - 计算较慢（约 10–30 分钟/深度层），带进度条

作者: EASTASIA-FWI Team
日期: 2026-03
版本: v14.0
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
from scipy.ndimage import uniform_filter, gaussian_filter
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
    """模型相似性分析配置类 (v13.0 — CW-SSIM)"""

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
            'compute_cwssim_spatial': True,
            # 地壳: 5–150 km 参考 Huang；地幔: 200–1000 km 覆盖 410/660 等关键界面
            'cwssim_spatial_depths': [5, 20, 60, 100, 150, 200, 300, 410, 660, 800, 900, 1000],
        }

        # ============ CW-SSIM 参数（东亚区域优化，非盲目对齐 Huang 2024）============
        self.cwssim = {
            'nlevels': 4,
            # K: 稳定常数。Sampat 2009 用 0.03；1e-6 对高幅系数更稳健，可调
            'K': 1e-6,
            # 空间滑窗（Huang 方法）：每个网格点取 local_win_deg°×local_win_deg° patch 计算 CW-SSIM
            # Huang 中国大陆用 6°；东亚区域 6° 仍合理，可设 local_win_deg_range=(4,12) 做 grid-search
            'local_win_deg': 6.0,
            'local_win_size': None,  # 若 None 则从 local_win_deg 推导
            'local_win_deg_range': None,  # 若设 (min, max) 可做 grid-search 找最优窗口
            # 分析分辨率：按模型对取最粗（pair_analysis_resolution_deg），公平比较
            'analysis_resolution_deg': 0.25,  # 仅作 fallback；实际使用 coverage 中的 pair 分辨率
            'level_weights': [0.05, 0.15, 0.35, 0.45],
            # 深度自适应尺度权重: (depth_max_km, weights) 按深度区间选用
            # 浅层偏细尺度、深层偏粗尺度，符合地震学分层特征
            'depth_adaptive_weights': {
                100: [0.15, 0.25, 0.35, 0.25],   # 0-100 km 地壳/LAB
                410: [0.08, 0.18, 0.37, 0.37],  # 100-410 km 上地幔
                660: [0.05, 0.15, 0.35, 0.45],   # 410-660 km 过渡带
                1000: [0.03, 0.12, 0.35, 0.50], # 660-1000 km 下地幔顶部
            },
            'use_depth_adaptive_weights': True,
            'use_gaussian_window': False,  # 局部 CW-SSIM 用高斯窗口（更平滑）
            'normalize_method': 'minmax',  # 'minmax' | 'percentile'
            'percentile_range': (2.0, 98.0),   # 稳健归一化百分位
        }

        # ============ 可视化配置 ============
        self.visualization = {
            'dpi': 300,
            'figsize_depth_curve': (12, 10),
            'figsize_heatmap': (10, 9),
            'figsize_cwssim_spatial': (16, 10),
            'figsize_overview': (20, 16),
            'heatmap': {
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
            'save_formats': ['png'],
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
        if self.ds is not None:
            self.ds.close()

    def __del__(self):
        self.close()


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

        img1 = np.nan_to_num(img1, nan=float(np.nanmean(img1)))
        img2 = np.nan_to_num(img2, nan=float(np.nanmean(img2)))

        img1, img2 = self._normalize_pair(img1, img2)

        min_dim = min(img1.shape)
        use_dtcwt = self._use_dtcwt and min_dim >= (2 ** self.nlevels + 1)

        if use_dtcwt:
            return self._cwssim_dtcwt(img1, img2, depth_km)
        return self._cwssim_gabor(img1, img2, depth_km)

    # ---------- 局部 CW-SSIM 空间分布图 ----------

    def compute_local(
        self, img1: np.ndarray, img2: np.ndarray, win_size: int = 7,
        depth_km: Optional[float] = None
    ) -> Tuple[float, np.ndarray]:
        """
        计算局部 CW-SSIM 图 + 全局均值。

        Args:
            depth_km: 深度 (km)，用于深度自适应尺度权重

        Returns:
            (global_cwssim, local_map)
        """
        img1 = np.asarray(img1, dtype=np.float64)
        img2 = np.asarray(img2, dtype=np.float64)

        img1 = np.nan_to_num(img1, nan=float(np.nanmean(img1)) if not np.all(np.isnan(img1)) else 0.0)
        img2 = np.nan_to_num(img2, nan=float(np.nanmean(img2)) if not np.all(np.isnan(img2)) else 0.0)

        img1, img2 = self._normalize_pair(img1, img2)

        min_dim = min(img1.shape)
        use_dtcwt = self._use_dtcwt and min_dim >= (2 ** self.nlevels + 1)

        if use_dtcwt:
            return self._cwssim_local_dtcwt(img1, img2, win_size, depth_km)
        return self._cwssim_local_gabor(img1, img2, win_size, depth_km)

    # ---------- 归一化 ----------

    def _normalize_pair(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对两张图像做独立归一化到 [0, 1]。

        - minmax: Huang et al. (2024) 做法，消除绝对速度差异
        - percentile: 稳健归一化，抗异常值
        """
        p_low, p_high = self._percentile_range

        def _norm_minmax(img: np.ndarray) -> np.ndarray:
            vmin, vmax = np.nanmin(img), np.nanmax(img)
            drange = vmax - vmin
            if drange < 1e-12:
                return np.zeros_like(img)
            return (img - vmin) / drange

        def _norm_percentile(img: np.ndarray) -> np.ndarray:
            vmin = float(np.nanpercentile(img, p_low))
            vmax = float(np.nanpercentile(img, p_high))
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

    def _cwssim_local_dtcwt(
        self, img1: np.ndarray, img2: np.ndarray, win_size: int,
        depth_km: Optional[float] = None
    ) -> Tuple[float, np.ndarray]:
        """使用 dtcwt 的局部 CW-SSIM（支持高斯窗口）"""
        from scipy.ndimage import zoom as nd_zoom

        orig_shape = img1.shape
        img1 = self._pad_to_even(img1)
        img2 = self._pad_to_even(img2)
        transform = dtcwt.Transform2d()
        c1 = transform.forward(img1, nlevels=self.nlevels)
        c2 = transform.forward(img2, nlevels=self.nlevels)

        level_maps = []
        for lev in range(self.nlevels):
            hp1 = c1.highpasses[lev]
            hp2 = c2.highpasses[lev]
            lev_map = np.zeros(hp1.shape[:2])
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
            level_maps.append(nd_zoom(lev_map, scale, order=1))

        weights = self._get_level_weights(depth_km)
        local_map = np.average(level_maps, axis=0, weights=weights)
        local_map = local_map[:orig_shape[0], :orig_shape[1]]
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
    """速度模型多尺度相似性分析器 (v13.0)"""

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

        self.logger.info("=" * 80)
        self.logger.info("🚀 多尺度速度模型相似性分析器 v14.0 初始化")
        self.logger.info(f"   CW-SSIM 后端: {'dtcwt (DT-CWT)' if HAS_DTCWT else 'Gabor 滤波器组 (备选)'}")
        self.logger.info("=" * 80)
        self._print_config_summary()

    def _print_config_summary(self):
        """打印配置摘要"""
        cw = self.config.cwssim
        print("\n📋 相似性分析配置 (v14.0 — CW-SSIM，按模型对最粗分辨率)")
        print("-" * 60)
        print(f"目标模型: {len(self.config.models)} 个")
        for info in self.config.models.values():
            print(f"  • {info['name']}")
        print(f"\n🔬 核心指标:")
        win_deg = cw.get('local_win_deg')
        win_info = f"{win_deg}°" if win_deg else f"{cw.get('local_win_size', 12)}px"
        print(f"  CW-SSIM — 复小波结构相似性 (nlevels={cw['nlevels']}, "
              f"分析分辨率=按模型对最粗, 局部窗口≈{win_info})")
        print(f"  空间分布: Huang 方法 — {win_deg or 6}°×{win_deg or 6}° 空间滑窗")
        print(f"  CW-SSIM 后端: {'dtcwt' if HAS_DTCWT else 'Gabor (备选)'}")
        print(f"  归一化: {cw.get('normalize_method', 'minmax')} | "
              f"深度自适应权重: {'开' if cw.get('use_depth_adaptive_weights') else '关'} | "
              f"高斯窗口: {'开' if cw.get('use_gaussian_window') else '关'}")
        print(f"\n📊 分析特征: {', '.join(self.config.analysis['target_features'])}")
        print(f"📁 输出: {self.figures_dir}")
        print("-" * 60)

    # ---------- 模型加载 ----------

    def load_models(self):
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

    # ---------- 公共覆盖区域 ----------

    def _get_common_coverage(
        self, m1: VelocityModelNetCDF, m2: VelocityModelNetCDF
    ) -> Dict[str, Any]:
        """
        获取两个模型的公共空间覆盖和网格坐标。

        使用较细分辨率模型的原生网格（在公共范围内），与 2_1 一致，
        避免 np.arange 构建的网格与模型不对齐导致边缘 NaN 或多余块。
        """
        lon_min = max(m1.lon.min(), m2.lon.min())
        lon_max = min(m1.lon.max(), m2.lon.max())
        lat_min = max(m1.lat.min(), m2.lat.min())
        lat_max = min(m1.lat.max(), m2.lat.max())

        res1 = m1.estimate_resolution()
        res2 = m2.estimate_resolution()
        coarse_lon = max(res1['lon'], res2['lon'])
        coarse_lat = max(res1['lat'], res2['lat'])

        # 使用较细模型的原生网格（在公共范围内），与 2_1 地图范围一致
        n1 = len(m1.lon) * len(m1.lat)
        n2 = len(m2.lon) * len(m2.lat)
        if n1 >= n2:
            src_lons, src_lats = m1.lon, m1.lat
        else:
            src_lons, src_lats = m2.lon, m2.lat

        mask_lon = (src_lons >= lon_min - 1e-6) & (src_lons <= lon_max + 1e-6)
        mask_lat = (src_lats >= lat_min - 1e-6) & (src_lats <= lat_max + 1e-6)
        target_lons = np.asarray(src_lons[mask_lon])
        target_lats = np.asarray(src_lats[mask_lat])

        if len(target_lons) < 2 or len(target_lats) < 2:
            # 回退：用 np.arange 构建
            target_lons = np.arange(lon_min, lon_max + coarse_lon * 0.5, coarse_lon)
            target_lats = np.arange(lat_min, lat_max + coarse_lat * 0.5, coarse_lat)
            target_lons = target_lons[target_lons <= lon_max]
            target_lats = target_lats[target_lats <= lat_max]

        if len(target_lons) < 2 or len(target_lats) < 2:
            raise ValueError(
                f"模型覆盖区域不足: {m1.name} 与 {m2.name} 的公共区域网格仅 "
                f"{len(target_lons)}×{len(target_lats)}"
            )

        # 网格实际分辨率（较细模型的间距）
        grid_res_lon = float(np.median(np.diff(target_lons))) if len(target_lons) > 1 else coarse_lon
        grid_res_lat = float(np.median(np.diff(target_lats))) if len(target_lats) > 1 else coarse_lat
        grid_resolution = min(grid_res_lon, grid_res_lat)
        # 模型对分析分辨率：取两者最粗，公平比较（避免粗模型被上采样出虚假细尺度）
        pair_analysis_resolution_deg = max(res1['lon'], res2['lon'], res1['lat'], res2['lat'])

        return {
            'lons': target_lons,
            'lats': target_lats,
            'resolution_lon': coarse_lon,
            'resolution_lat': coarse_lat,
            'grid_resolution': grid_resolution,
            'pair_analysis_resolution_deg': pair_analysis_resolution_deg,
        }

    def _sample_slice_to_coverage(
        self,
        model: VelocityModelNetCDF, data_3d: np.ndarray,
        d_idx: int, coverage: Dict[str, Any],
    ) -> np.ndarray:
        """从已缓存的 3D 数组取一个深度层，最近邻采样到公共粗网格"""
        return nearest_neighbor_sample(
            model.lon, model.lat, data_3d[:, :, d_idx],
            coverage['lons'], coverage['lats'],
        )

    def _get_local_win_size(self, resolution_deg: Optional[float] = None) -> int:
        """
        获取 Huang 空间滑窗的 patch 大小（像素）。

        local_win_deg°×local_win_deg° 物理窗口 → 像素数 = ceil(local_win_deg / res_deg)
        结果保证为奇数。
        """
        cw = self.config.cwssim
        res = resolution_deg or cw['analysis_resolution_deg']
        if cw.get('local_win_deg') is not None:
            size = int(np.ceil(cw['local_win_deg'] / res))
        elif cw.get('local_win_size') is not None:
            size = int(cw['local_win_size'])
        else:
            size = 12  # 6°/0.5° 默认
        return size | 1  # 保证奇数

    def _compute_cwssim_spatial_huang(
        self,
        s1: np.ndarray, s2: np.ndarray,
        coverage: Dict[str, Any],
        depth_km: float,
    ) -> Tuple[float, np.ndarray]:
        """
        Huang (2024) 空间滑窗法：每个网格点取 local_win_deg°×local_win_deg° patch，
        单独计算该 patch 的 CW-SSIM。

        Returns:
            (global_mean, cwssim_grid)
        """
        pair_res = coverage['pair_analysis_resolution_deg']
        half = self._get_local_win_size(pair_res) // 2

        n_lon, n_lat = s1.shape
        cwssim_grid = np.full((n_lon, n_lat), np.nan, dtype=np.float64)

        ij_list = [
            (i, j) for i in range(half, n_lon - half)
            for j in range(half, n_lat - half)
        ]
        for i, j in tqdm(ij_list, desc=f"  {depth_km:.0f} km", leave=False):
            patch1 = s1[i - half:i + half + 1, j - half:j + half + 1]
            patch2 = s2[i - half:i + half + 1, j - half:j + half + 1]
            valid = np.isfinite(patch1) & np.isfinite(patch2)
            if np.sum(valid) < 10:
                continue
            p1 = np.where(valid, patch1, np.nanmean(patch1))
            p2 = np.where(valid, patch2, np.nanmean(patch2))
            cwssim_grid[i, j] = self.cwssim_calc.compute_global(p1, p2, depth_km=depth_km)

        valid_vals = cwssim_grid[np.isfinite(cwssim_grid)]
        global_mean = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0
        return global_mean, cwssim_grid

    # ---------- 降采样 ----------

    def _downsample_for_cwssim(
        self, s1: np.ndarray, s2: np.ndarray,
        coverage: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将深度切片降采样到模型对的分析分辨率。

        按模型对使用最粗分辨率（pair_analysis_resolution_deg），确保公平比较：
        - SinoScope (1°) vs EARA (0.25°) → 分析于 1°
        - EARA (0.25°) vs FWEA23 (0.25°) → 分析于 0.25°
        避免粗模型被上采样产生虚假细尺度结构。
        """
        from scipy.ndimage import zoom as nd_zoom

        target_res = coverage['pair_analysis_resolution_deg']
        src_res = coverage['grid_resolution']

        if src_res >= target_res * 0.95:
            return s1, s2

        factor = src_res / target_res
        s1_ds = nd_zoom(s1, factor, order=1)
        s2_ds = nd_zoom(s2, factor, order=1)
        return s1_ds, s2_ds

    # ---------- 逐深度层三指标计算 ----------

    def compute_depth_wise_metrics(
        self,
        m1_key: str, m2_key: str,
        feature: str,
    ) -> pd.DataFrame:
        """
        逐深度层计算 CW-SSIM。

        Returns:
            DataFrame with columns: depth_km, cwssim
        """
        m1 = self.models[m1_key]
        m2 = self.models[m2_key]
        coverage = self._get_common_coverage(m1, m2)
        matched_depths = find_common_depth_indices(m1.depth, m2.depth)

        if not matched_depths:
            self.logger.warning(f"  ⚠️ {m1.name} vs {m2.name}: 无匹配深度层")
            return pd.DataFrame(columns=['depth_km', 'cwssim'])

        pair_res = coverage['pair_analysis_resolution_deg']
        self.logger.info(
            f"  📊 逐深度层分析: {m1.name} vs {m2.name}  "
            f"({len(matched_depths)} 层, {feature.upper()}, 分析分辨率={pair_res}°)"
        )

        data1_3d = m1.get_parameter(feature)
        data2_3d = m2.get_parameter(feature)

        rows = []
        for d_idx1, d_idx2, depth_km in tqdm(matched_depths, desc="  深度层", leave=False):
            s1 = self._sample_slice_to_coverage(m1, data1_3d, d_idx1, coverage)
            s2 = self._sample_slice_to_coverage(m2, data2_3d, d_idx2, coverage)

            s1_ds, s2_ds = self._downsample_for_cwssim(s1, s2, coverage)
            cwssim_mean = self.cwssim_calc.compute_global(
                s1_ds, s2_ds, depth_km=depth_km,
            )

            rows.append({
                'depth_km': depth_km,
                'cwssim': cwssim_mean,
            })

        df = pd.DataFrame(rows)
        self.logger.info(
            f"    CW-SSIM: [{df['cwssim'].min():.3f}, {df['cwssim'].max():.3f}]"
        )
        return df

    # ---------- CW-SSIM 空间分布 ----------

    def compute_cwssim_spatial_pair(
        self,
        m1_key: str, m2_key: str,
        feature: str,
        depths: Optional[List[float]] = None,
    ) -> Dict[float, Tuple[float, np.ndarray, Dict]]:
        """
        计算指定深度的 CW-SSIM 局部空间分布图。

        Returns:
            {depth_km: (global_cwssim, local_map, coverage)}
        """
        if depths is None:
            depths = self.config.analysis['cwssim_spatial_depths']

        m1 = self.models[m1_key]
        m2 = self.models[m2_key]
        coverage = self._get_common_coverage(m1, m2)

        pair_res = coverage['pair_analysis_resolution_deg']
        win_deg = self.config.cwssim.get('local_win_deg') or 6.0
        self.logger.info(
            f"  🗺️ CW-SSIM 空间分布 (Huang 方法): {m1.name} vs {m2.name}  "
            f"(分辨率={pair_res}°, 滑窗={win_deg}°×{win_deg}°, 预计较慢)"
        )

        results = {}
        matched = find_common_depth_indices(m1.depth, m2.depth)
        depth_map = {d: (i1, i2) for i1, i2, d in matched}

        if not depth_map:
            self.logger.warning(f"    ⚠️ 无匹配深度层")
            return results

        data1_3d = m1.get_parameter(feature)
        data2_3d = m2.get_parameter(feature)

        for target_depth in depths:
            closest_depth = min(depth_map.keys(), key=lambda d: abs(d - target_depth))
            if abs(closest_depth - target_depth) > 10.0:
                continue
            d_idx1, d_idx2 = depth_map[closest_depth]
            s1 = self._sample_slice_to_coverage(m1, data1_3d, d_idx1, coverage)
            s2 = self._sample_slice_to_coverage(m2, data2_3d, d_idx2, coverage)

            # 公空缺修复：仅在两模型均有有效数据的区域计算/显示 CW-SSIM
            valid_mask = np.isfinite(s1) & np.isfinite(s2)

            s1_ds, s2_ds = self._downsample_for_cwssim(s1, s2, coverage)

            try:
                global_val, local_map = self._compute_cwssim_spatial_huang(
                    s1_ds, s2_ds, coverage, closest_depth,
                )
                if local_map.shape != s1.shape:
                    from scipy.ndimage import zoom as nd_zoom
                    scale = np.array(s1.shape, dtype=float) / np.array(local_map.shape, dtype=float)
                    local_map = nd_zoom(local_map, scale, order=1)
                results[closest_depth] = (global_val, local_map, coverage, valid_mask)
                self.logger.info(f"    ✅ {closest_depth:.0f} km: CW-SSIM = {global_val:.4f}")
            except Exception as e:
                self.logger.warning(f"    ⚠️ {closest_depth:.0f} km 失败: {e}")

        return results

    # ---------- 批量分析 ----------

    def compare_all_models(self, features: Optional[List[str]] = None):
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
            models_with_feature = [k for k in model_keys if self.models[k].has_parameter(feature)]
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
                    # 逐深度层三指标
                    if self.config.analysis['compute_depth_wise']:
                        df = self.compute_depth_wise_metrics(m1_key, m2_key, feature)
                        self.results[f"{pair_id}_depth"] = df

                    # CW-SSIM 空间分布
                    if self.config.analysis['compute_cwssim_spatial']:
                        cs = self.compute_cwssim_spatial_pair(m1_key, m2_key, feature)
                        self.results[f"{pair_id}_cwssim_spatial"] = cs

                except Exception as e:
                    self.logger.error(f"❌ {m1_key} vs {m2_key} ({feature}): {e}")
                    import traceback
                    traceback.print_exc()

        self.logger.info("\n✅ 批量相似性分析完成")

    # ---------- 结果保存 ----------

    def save_results(self):
        """保存分析结果"""
        self.logger.info("\n💾 保存分析结果...")

        features = set()
        for key in self.results:
            if key.endswith('_depth'):
                feat = key[:-len('_depth')].split('_')[-1]
                features.add(feat)

        for feature in features:
            feat_dir = self.output_dir / feature
            feat_dir.mkdir(parents=True, exist_ok=True)

            # 保存深度指标 CSV
            suffix_depth = f'_{feature}_depth'
            depth_keys = [k for k in self.results if k.endswith(suffix_depth)]
            all_summaries = []

            for dk in depth_keys:
                df = self.results[dk]
                csv_file = feat_dir / f"{dk}.csv"
                df.to_csv(csv_file, index=False, float_format='%.6f')
                self.logger.info(f"  ✅ {csv_file.name}")

                pair_name = dk[:-len(suffix_depth)]
                all_summaries.append({
                    'pair': pair_name,
                    'feature': feature,
                    'cwssim_mean': df['cwssim'].mean(),
                })

            if all_summaries:
                df_summary = pd.DataFrame(all_summaries)
                summary_file = feat_dir / f'similarity_summary_{feature}.csv'
                df_summary.to_csv(summary_file, index=False, float_format='%.6f')
                self.logger.info(f"  ✅ {summary_file.name}")

            # 保存 JSON 摘要
            json_data = {}
            for s in all_summaries:
                json_data[s['pair']] = {
                    k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in s.items()
                }
            json_file = feat_dir / f'similarity_summary_{feature}.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"  ✅ {json_file.name}")

        self.logger.info("✅ 所有结果已保存")


# ==================== 可视化系统 ====================

class SimilarityVisualization:
    """多尺度相似性分析可视化 (v10.0)"""

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
        """
        获取 2-2 空间分布图的显示范围，与 2-1 一致：使用模型网格的 lon/lat 范围。
        2-1 使用 model.get_horizontal_slice 返回的 lon_grid/lat_grid 的 min/max。
        """
        parts = pair_name.split('_vs_')
        if len(parts) >= 1 and parts[0] in self.analyzer.models:
            m = self.analyzer.models[parts[0]]
            return (
                float(m.lon.min()), float(m.lon.max()),
                float(m.lat.min()), float(m.lat.max()),
            )
        # 回退：从 valid_standardized_region.json 读取
        region_path = self.analyzer.base_config.dirs['models'] / 'metadata' / 'valid_standardized_region.json'
        if region_path.exists():
            try:
                with open(region_path, 'r', encoding='utf-8') as f:
                    region = json.load(f)
                lon = region.get('lon', [80.0, 150.0])
                lat = region.get('lat', [10.0, 55.0])
                return (float(lon[0]), float(lon[1]), float(lat[0]), float(lat[1]))
            except Exception:
                pass
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

    # ---------- 1. CW-SSIM 深度曲线 ----------

    def plot_depth_cwssim(self, feature: str = 'vs') -> Optional[plt.Figure]:
        """绘制所有模型对的 CW-SSIM(z) 深度曲线"""
        suffix = f'_{feature}_depth'
        depth_keys = [k for k in self.analyzer.results if k.endswith(suffix)]
        if not depth_keys:
            return None

        self.logger.info(f"\n🎨 CW-SSIM 深度曲线: {feature.upper()}")

        fig, ax = plt.subplots(figsize=self.config.visualization['figsize_depth_curve'])

        pair_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']

        # 排序确保图例顺序一致（SinoScope vs EARA, SinoScope vs FWEA, EARA vs FWEA）
        depth_keys = sorted(depth_keys)

        for idx, dk in enumerate(depth_keys):
            df = self.analyzer.results[dk]
            pair_name = dk[:-len(suffix)]
            label = self._pair_label(pair_name)

            color = pair_colors[idx % len(pair_colors)]
            ax.plot(df['cwssim'], df['depth_km'], '-o', color=color, linewidth=2,
                    markersize=3, label=label)

        ax.set_xlabel('CW-SSIM', fontsize=14)
        ax.set_ylabel('Depth (km)', fontsize=14)
        ax.set_title(f'Complex Wavelet SSIM vs Depth — {feature.upper()}',
                     fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        self._add_discontinuities(ax, 'horizontal')
        ax.legend(fontsize=11, loc='lower left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_figure(fig, f'depth_cwssim_{feature}')
        return fig

    # ---------- 2. CW-SSIM 热图矩阵 ----------

    def plot_cwssim_heatmap(self, feature: str = 'vs') -> Optional[plt.Figure]:
        """绘制 N×N CW-SSIM 相似性热图"""
        suffix = f'_{feature}_depth'
        depth_keys = [k for k in self.analyzer.results if k.endswith(suffix)]
        if not depth_keys:
            return None

        self.logger.info(f"\n🎨 CW-SSIM 热图: {feature.upper()}")

        model_keys = list(self.analyzer.models.keys())
        n = len(model_keys)
        matrix = np.ones((n, n))

        for dk in sorted(depth_keys):
            df = self.analyzer.results[dk]
            pair_name = dk[:-len(suffix)]
            i, j = self._parse_pair_indices(pair_name, model_keys)
            if i is not None:
                mean_cwssim = df['cwssim'].mean()
                matrix[i, j] = mean_cwssim
                matrix[j, i] = mean_cwssim

        names = [self.analyzer.models[k].name for k in model_keys]
        df_matrix = pd.DataFrame(matrix, index=names, columns=names)
        # 遮罩下三角，显示上三角+对角线（常规相似性矩阵约定）
        mask = np.tril(np.ones_like(matrix, dtype=bool), k=-1)

        hm_cfg = self.config.visualization['heatmap']
        fig, ax = plt.subplots(figsize=self.config.visualization['figsize_heatmap'])

        sns.heatmap(
            df_matrix, mask=mask, annot=True, fmt='.3f',
            cmap=hm_cfg['cmap'],
            vmin=hm_cfg['cwssim_vmin'], vmax=hm_cfg['cwssim_vmax'],
            square=True,
            linewidths=hm_cfg['cell_linewidth'],
            linecolor=hm_cfg['cell_linecolor'],
            annot_kws={'fontsize': hm_cfg['annot_fontsize'], 'fontweight': 'bold'},
            cbar_kws={'label': 'CW-SSIM Index', 'shrink': 0.8},
            ax=ax,
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                           fontsize=hm_cfg['label_fontsize'], fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=hm_cfg['label_fontsize'], fontweight='bold')

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=hm_cfg['cbar_fontsize'])
        cbar.ax.set_ylabel('CW-SSIM Index', fontsize=hm_cfg['cbar_fontsize'] + 2,
                           rotation=270, labelpad=20)

        ax.set_title(
            f'Model Similarity Matrix — {feature.upper()}\n'
            f'Complex Wavelet SSIM (depth-averaged)',
            fontsize=hm_cfg['title_fontsize'], fontweight='bold', pad=15,
        )

        plt.tight_layout()
        self._save_figure(fig, f'cwssim_heatmap_{feature}')
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

    # ---------- 5. 综合概览面板 ----------

    def plot_similarity_overview(self, feature: str = 'vs') -> Optional[plt.Figure]:
        """绘制 CW-SSIM 综合面板（2×1 布局）"""
        suffix_depth = f'_{feature}_depth'
        depth_keys = [k for k in self.analyzer.results if k.endswith(suffix_depth)]

        if not depth_keys:
            return None

        self.logger.info(f"\n🎨 综合概览面板: {feature.upper()}")

        depth_keys = sorted(depth_keys)

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        pair_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']

        def _get_label(dk):
            if dk.endswith(suffix_depth):
                return self._pair_label(dk[:-len(suffix_depth)])
            return dk

        # (0,0) CW-SSIM vs Depth
        ax_cwssim = fig.add_subplot(gs[0, 0])
        for idx, dk in enumerate(depth_keys):
            df = self.analyzer.results[dk]
            ax_cwssim.plot(df['cwssim'], df['depth_km'], '-o', color=pair_colors[idx % 5],
                           linewidth=2, markersize=3, label=_get_label(dk))
        ax_cwssim.set_xlabel('CW-SSIM', fontsize=12)
        ax_cwssim.set_ylabel('Depth (km)', fontsize=12)
        ax_cwssim.set_title('CW-SSIM vs Depth', fontsize=14, fontweight='bold')
        ax_cwssim.invert_yaxis()
        ax_cwssim.set_xlim(0, 1.05)
        ax_cwssim.legend(fontsize=9)
        ax_cwssim.grid(True, alpha=0.3)
        self._add_discontinuities(ax_cwssim)

        # (0,1) CW-SSIM Heatmap
        ax_heatmap = fig.add_subplot(gs[0, 1])
        model_keys = list(self.analyzer.models.keys())
        n = len(model_keys)
        matrix = np.ones((n, n))
        for dk in depth_keys:
            df = self.analyzer.results[dk]
            pair_name = dk[:-len(suffix_depth)]
            i, j = self._parse_pair_indices(pair_name, model_keys)
            if i is not None:
                val = df['cwssim'].mean()
                matrix[i, j] = val
                matrix[j, i] = val

        hm_cfg = self.config.visualization['heatmap']
        names = [self.analyzer.models[k].name for k in model_keys]
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        sns.heatmap(pd.DataFrame(matrix, index=names, columns=names),
                    mask=mask, annot=True, fmt='.3f', cmap=hm_cfg['cmap'],
                    vmin=hm_cfg['cwssim_vmin'], vmax=hm_cfg['cwssim_vmax'],
                    square=True,
                    annot_kws={'fontsize': 16, 'fontweight': 'bold'},
                    cbar_kws={'shrink': 0.8}, ax=ax_heatmap)
        ax_heatmap.set_title('CW-SSIM Matrix', fontsize=14, fontweight='bold')

        # (1,0) CW-SSIM 统计对比
        ax_stats = fig.add_subplot(gs[1, 0])
        stats_data = []
        for idx, dk in enumerate(depth_keys):
            df = self.analyzer.results[dk]
            stats_data.append({
                'pair': _get_label(dk),
                'mean': df['cwssim'].mean(),
                'std': df['cwssim'].std(),
                'min': df['cwssim'].min(),
                'max': df['cwssim'].max(),
            })
        df_stats = pd.DataFrame(stats_data)
        x_pos = np.arange(len(df_stats))
        bars = ax_stats.bar(x_pos, df_stats['mean'], color=pair_colors[:len(df_stats)],
                           yerr=df_stats['std'], capsize=5, alpha=0.8)
        ax_stats.set_xticks(x_pos)
        ax_stats.set_xticklabels(df_stats['pair'], rotation=45, ha='right', fontsize=9)
        ax_stats.set_ylabel('CW-SSIM', fontsize=12)
        ax_stats.set_title('CW-SSIM Statistics', fontsize=14, fontweight='bold')
        ax_stats.set_ylim(0, 1.1)
        ax_stats.grid(True, alpha=0.3, axis='y')
        for i, (mean, min_v, max_v) in enumerate(zip(df_stats['mean'], df_stats['min'], df_stats['max'])):
            ax_stats.text(i, mean + 0.05, f'{mean:.3f}', ha='center', fontsize=9)
            ax_stats.text(i, min_v - 0.05, f'{min_v:.3f}', ha='center', fontsize=8, alpha=0.7)
            ax_stats.text(i, max_v + 0.05, f'{max_v:.3f}', ha='center', fontsize=8, alpha=0.7)

        # (1,1) CW-SSIM 深度分布箱线图
        ax_box = fig.add_subplot(gs[1, 1])
        all_cwssim = []
        all_labels = []
        for dk in depth_keys:
            df = self.analyzer.results[dk]
            all_cwssim.append(df['cwssim'].values)
            all_labels.append(_get_label(dk))
        bp = ax_box.boxplot(all_cwssim, labels=all_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], pair_colors[:len(all_cwssim)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_box.set_ylabel('CW-SSIM', fontsize=12)
        ax_box.set_title('CW-SSIM Distribution', fontsize=14, fontweight='bold')
        ax_box.set_ylim(0, 1.1)
        ax_box.grid(True, alpha=0.3, axis='y')
        plt.setp(ax_box.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

        fig.suptitle(
            f'Similarity Overview — {feature.upper()}',
            fontsize=18, fontweight='bold', y=1.01,
        )
        plt.tight_layout()
        self._save_figure(fig, f'similarity_overview_{feature}')
        return fig


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🚀 EASTASIA-FWI 多尺度速度模型相似性分析 (v13.0)")
    print("   CW-SSIM")
    print("=" * 80)
    print("\n🎯 核心指标:")
    print("  • CW-SSIM — 复小波结构相似性 (抗畸变, 多尺度)")
    print(f"\n🔧 CW-SSIM 后端: {'dtcwt (DT-CWT)' if HAS_DTCWT else 'Gabor 滤波器组'}")
    print("=" * 80 + "\n")

    try:
        # 1. 初始化
        print("📋 步骤 1/5: 初始化配置...")
        config = ModelSimilarityConfig()
        analyzer = VelocityModelSimilarity(config)

        # 2. 加载模型
        print("\n📦 步骤 2/5: 加载 NetCDF 模型...")
        analyzer.load_models()

        # 3. 批量分析
        print("\n🔬 步骤 3/5: 批量相似性分析...")
        analyzer.compare_all_models()

        # 4. 保存结果
        print("\n💾 步骤 4/5: 保存结果...")
        analyzer.save_results()

        # 5. 可视化
        print("\n🎨 步骤 5/5: 生成可视化...")
        visualizer = SimilarityVisualization(analyzer)

        features_done = set()
        for key in analyzer.results:
            if key.endswith('_depth'):
                feat = key[:-len('_depth')].split('_')[-1]
                features_done.add(feat)

        for feature in features_done:
            print(f"\n  📊 {feature.upper()}...")
            visualizer.plot_depth_cwssim(feature)
            visualizer.plot_cwssim_heatmap(feature)
            visualizer.plot_cwssim_spatial(feature)
            visualizer.plot_similarity_overview(feature)

        print("\n" + "=" * 80)
        print("✅ 多尺度相似性分析完成!")
        print(f"📁 结果: {analyzer.output_dir}")
        print(f"📁 图表: {analyzer.figures_dir}")
        print("\n📊 输出文件:")
        for feature in features_done:
            print(f"\n  {feature.upper()}:")
            print(f"    • 2-2_depth_cwssim_{feature}        — CW-SSIM 深度曲线")
            print(f"    • 2-2_cwssim_heatmap_{feature}      — CW-SSIM 热图矩阵")
            print(f"    • 2-2_cwssim_spatial_*_{feature}    — CW-SSIM 空间分布")
            print(f"    • 2-2_similarity_overview_{feature} — 综合概览面板")
        print("\n🎯 下一步:")
        print("  • 基于相似性结果进行模型选择与融合 (4_Assembly/)")
        print("  • 选择代表性模型进行 SPECFEM3D 正演 (3_Data_space_simulation/)")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
