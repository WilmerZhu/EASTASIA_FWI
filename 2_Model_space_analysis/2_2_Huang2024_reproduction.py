"""
2_2_Huang2024_reproduction.py:
Huang et al. (2024) 论文复现 — USTClitho2.0 vs FWEA18 结构相似性分析
================================================================

功能描述:
----
复现 Huang F, Bao XY, Dai QA, Li XF (2024). Structural similarity of lithospheric
velocity models of Chinese mainland. Earthquake Science, 37(6): 514-528.
DOI: 10.1016/j.eqs.2024.05.004

使用 CW-SSIM 方法分析 USTClitho2.0 与 FWEA18 两个模型在中国大陆区域的结构相似性。
论文使用 4 个模型 (Bao15, Shen16, FWEA18, USTClitho2.0)，本脚本仅复现 USTClitho2.0 vs FWEA18 部分。

核心功能:
----
1. ✅ 加载 USTClitho2.0 和 FWEA18 标准化 NetCDF 数据
2. ✅ 中国大陆区域 (90°E–135°E, 18°N–50°N)
3. ✅ 深度层: 5, 20, 60, 100, 150 km
4. ✅ 水平网格: 0.5°×0.5° (论文对齐 FWEA18 0.25° 到 0.5°)
5. ✅ 全区域 CW-SSIM: 6°×6° 滑动窗口 (论文 grid-search 4°–10° 得 knee point)
6. ✅ 区域 CW-SSIM: 华南、东北等子区域
7. ✅ 输出: 深度曲线、空间分布图、直方图

使用方法:
----
1. 先运行 1_5 处理 USTClitho2.0 和 FWEA18（若尚未处理）:
   ```bash
   python 1_Data_preparation/1_5_Process_velocity_models.py
   ```
   并在 1_5 中设置 model_names = ['2022_USTClitho2.0', '2018_FWEA18']

2. 运行复现脚本:
   ```bash
   python 2_Model_space_analysis/2_2_Huang2024_reproduction.py
   ```

输出文件:
----
- Huang2024_cwssim_depth_curve.png: CW-SSIM vs 深度
- Huang2024_cwssim_spatial_*.png: 各深度 CW-SSIM 空间分布
- Huang2024_cwssim_histogram.png: CW-SSIM 直方图 (对应论文 Figure 13)
- Huang2024_results.csv: 数值结果

作者: EASTASIA-FWI Team
日期: 2026-03
版本: v1.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter, gaussian_filter
from tqdm import tqdm

try:
    import pyrtools as pt
    HAS_PYRTOOLS = True
except ImportError:
    pt = None
    HAS_PYRTOOLS = False

try:
    import dtcwt
    HAS_DTCWT = True
except ImportError:
    HAS_DTCWT = False

try:
    import pygmt
    HAS_PYGMT = True
except ImportError:
    pygmt = None
    HAS_PYGMT = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    ccrs = None
    cfeature = None
    HAS_CARTOPY = False

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import BaseConfig

warnings.filterwarnings('ignore')


# ==================== 论文参数配置 ====================

# Huang et al. (2024) 论文参数
HUANG2024_REGION = {
    'lon_min': 90.0,
    'lon_max': 135.0,
    'lat_min': 18.0,
    'lat_max': 50.0,
    'name': 'Chinese mainland',
}

# 论文深度层 (km)
HUANG2024_DEPTHS = [5, 20, 60, 100, 150]

# 论文水平网格 (度)
HUANG2024_GRID_RES = 0.5

# 全区域 CW-SSIM 滑动窗口 (论文 grid-search 得 6° 最优)
HUANG2024_BOX_SIZE_DEG = 6.0

# 论文子区域 (用于区域 CW-SSIM)
HUANG2024_SUBREGIONS = {
    'Southeast_China': {'lon': (110.0, 120.0), 'lat': (27.0, 34.0)},
    'Northeast_China': {'lon': (120.0, 130.0), 'lat': (42.0, 49.0)},
    'Sichuan_Basin': {'lon': (100.0, 108.0), 'lat': (28.0, 34.0)},
    'Ordos_Basin': {'lon': (106.0, 112.0), 'lat': (34.0, 40.0)},
}


# ==================== 模型配置 ====================

MODELS_CONFIG = {
    '2022_USTClitho2.0': {
        'processed_dir': '2022_USTClitho2.0',
        'netcdf_file': '2022_USTClitho2.0_original.nc',
        'name': 'USTClitho2.0',
        'color': '#377EB8',
    },
    '2018_FWEA18': {
        'processed_dir': '2018_FWEA18',
        'netcdf_file': '2018_FWEA18_original.nc',
        'name': 'FWEA18',
        'color': '#E41A1C',
    },
}


# ==================== 工具函数 ====================

def nearest_neighbor_sample(
    model_lon: np.ndarray, model_lat: np.ndarray,
    data_2d: np.ndarray,
    target_lons: np.ndarray, target_lats: np.ndarray
) -> np.ndarray:
    """在目标坐标上使用最近邻采样 2D 数据"""
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
    CW-SSIM 计算器 — 精确匹配 Huang et al. (2024) 使用的 MATLAB cwssim_index.m:
      1. buildSCFpyr (Steerable Complex Frequency Pyramid)
      2. 仅取最粗层 (coarsest level) 各方向子带
      3. 7×7 滑动窗口计算局部 CW-SSIM map
      4. Gaussian 空间加权求和
      5. 各方向平均
    Python 端使用 pyrtools.pyramids.SteerablePyramidFreq 实现。
    方法优先级: Steerable Pyramid (pyrtools) → DTCWT → Gabor
    """
    def __init__(self, n_orientations: int = 16, K: float = 0):
        self.n_orientations = n_orientations
        self.K = K
        self.winsize = 7
        self._use_pyrtools = HAS_PYRTOOLS
        self._use_dtcwt = HAS_DTCWT

    def _normalize_minmax(self, img: np.ndarray) -> np.ndarray:
        """论文: 每图独立 min-max 归一化到 [0, 1]"""
        vmin, vmax = np.nanmin(img), np.nanmax(img)
        drange = vmax - vmin
        if drange < 1e-12:
            return np.zeros_like(img)
        return (img - vmin) / drange

    @staticmethod
    def _pad_to_even(img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            return np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
        return img

    def compute_global(
        self, img1: np.ndarray, img2: np.ndarray, normalize: bool = True
    ) -> float:
        """计算全局 CW-SSIM。normalize=True 时每块独立 min-max；False 时假定已按研究区归一化。"""
        img1 = np.asarray(img1, dtype=np.float64)
        img2 = np.asarray(img2, dtype=np.float64)
        if img1.shape != img2.shape:
            raise ValueError(f"形状不一致: {img1.shape} vs {img2.shape}")
        if np.all(np.isnan(img1)) or np.all(np.isnan(img2)):
            return 0.0
        img1 = np.nan_to_num(img1, nan=float(np.nanmean(img1)))
        img2 = np.nan_to_num(img2, nan=float(np.nanmean(img2)))
        if normalize:
            img1 = self._normalize_minmax(img1)
            img2 = self._normalize_minmax(img2)

        if self._use_pyrtools:
            return self._cwssim_steerable(img1, img2)

        min_dim = min(img1.shape)
        nlevels_dtcwt = 4
        if self._use_dtcwt and min_dim >= (2 ** nlevels_dtcwt + 1):
            return self._cwssim_dtcwt(img1, img2, nlevels_dtcwt)
        return self._cwssim_gabor(img1, img2)

    def _cwssim_steerable(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        基于 MATLAB cwssim_index.m 的 CW-SSIM 计算:
          buildSCFpyr → 最粗层各方向子带 → 7×7 滑窗局部 CW-SSIM → Gaussian 加权 → 方向平均

        pyrtools 的 level-0 子带为全分辨率 (13×13 for 13×13 input)，
        而 MATLAB 的 level-0 子带在提取前已下采样 (7×7)。
        由于 MATLAB 使用平滑低通过渡带 (raised cosine) 而非硬截断，
        直接频域裁剪会引入系统性偏差，因此保留 pyrtools 全分辨率子带。
        """
        from scipy.signal import fftconvolve
        import warnings as _w

        n_or = self.n_orientations
        K = self.K
        winsize = self.winsize
        window = np.ones((winsize, winsize), dtype=np.float64)
        window /= window.sum()

        min_dim = min(img1.shape)
        level = max(1, int(np.floor(np.log2(min_dim))) - 2)

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            pyr1 = pt.pyramids.SteerablePyramidFreq(
                img1, height=level, order=n_or - 1, is_complex=True,
            )
            pyr2 = pt.pyramids.SteerablePyramidFreq(
                img2, height=level, order=n_or - 1, is_complex=True,
            )

        coarsest = level - 1
        first_key = (coarsest, 0)
        if first_key not in pyr1.pyr_coeffs:
            return 0.0

        s = np.array(pyr1.pyr_coeffs[first_key].shape, dtype=int)

        band_cssim = []
        for o in range(n_or):
            key = (coarsest, o)
            if key not in pyr1.pyr_coeffs:
                continue
            band1 = pyr1.pyr_coeffs[key]
            band2 = pyr2.pyr_coeffs[key]

            if min(band1.shape) < winsize:
                c1 = band1.ravel()
                c2 = band2.ravel()
                num = 2.0 * np.abs(np.sum(c1 * np.conj(c2))) + K
                den = np.sum(np.abs(c1) ** 2) + np.sum(np.abs(c2) ** 2) + K
                band_cssim.append(num / den)
                continue

            corr = band1 * np.conj(band2)
            varr = np.abs(band1) ** 2 + np.abs(band2) ** 2

            corr_band = fftconvolve(corr, window, mode='valid')
            varr_band = fftconvolve(varr, window, mode='valid')

            cssim_map = (2.0 * np.abs(corr_band) + K) / (varr_band + K)

            map_h, map_w = cssim_map.shape
            sigma = s[0] / 4.0
            if sigma < 0.5:
                sigma = 0.5
            gy = np.arange(map_h, dtype=np.float64) - (map_h - 1) / 2.0
            gx = np.arange(map_w, dtype=np.float64) - (map_w - 1) / 2.0
            gy2, gx2 = np.meshgrid(gy, gx, indexing='ij')
            w = np.exp(-(gy2 ** 2 + gx2 ** 2) / (2.0 * sigma ** 2))
            w /= w.sum()

            band_cssim.append(float(np.sum(cssim_map.real * w)))

        if not band_cssim:
            return 0.0
        return float(np.mean(band_cssim))

    def _cwssim_dtcwt(self, img1: np.ndarray, img2: np.ndarray,
                      nlevels: int = 4) -> float:
        """DTCWT 备选实现"""
        img1 = self._pad_to_even(img1)
        img2 = self._pad_to_even(img2)
        transform = dtcwt.Transform2d()  # type: ignore[attr-defined]
        c1 = transform.forward(img1, nlevels=nlevels)
        c2 = transform.forward(img2, nlevels=nlevels)

        level_vals = []
        for lev in range(nlevels):
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
        return float(np.mean(level_vals))

    def _cwssim_gabor(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Gabor 备选实现"""
        from scipy.ndimage import convolve
        orientations = np.linspace(0, np.pi, self.n_orientations, endpoint=False)
        nlevels = 4
        scales = [2 ** i for i in range(1, nlevels + 1)]
        level_vals = []
        for scale in scales:
            sigma = scale * 0.8
            freq = 1.0 / scale
            ksize = max(3, int(sigma * 4) | 1)
            y, x = np.mgrid[-ksize//2:ksize//2+1, -ksize//2:ksize//2+1].astype(np.float64)
            orient_vals = []
            for theta in orientations:
                x_rot = x * np.cos(theta) + y * np.sin(theta)
                gaussian = np.exp(-(x_rot ** 2 + y ** 2) / (2 * sigma ** 2))
                sinusoid = np.exp(1j * 2 * np.pi * freq * x_rot)
                kernel = gaussian * sinusoid
                kernel /= np.abs(kernel).sum() + 1e-12
                c1 = convolve(img1, np.real(kernel)) + 1j * convolve(img1, np.imag(kernel))
                c2 = convolve(img2, np.real(kernel)) + 1j * convolve(img2, np.imag(kernel))
                cx, cy = c1.ravel(), c2.ravel()
                num = 2.0 * np.abs(np.sum(cx * np.conj(cy))) + self.K
                den = np.sum(np.abs(cx) ** 2) + np.sum(np.abs(cy) ** 2) + self.K
                orient_vals.append(num / den)
            level_vals.append(np.mean(orient_vals))
        return float(np.mean(level_vals))

    def compute_local(
        self, img1: np.ndarray, img2: np.ndarray, win_size: int = 7
    ) -> Tuple[float, np.ndarray]:
        """计算局部 CW-SSIM 图 + 全局均值"""
        img1 = np.asarray(img1, dtype=np.float64)
        img2 = np.asarray(img2, dtype=np.float64)
        img1 = np.nan_to_num(img1, nan=float(np.nanmean(img1)) if not np.all(np.isnan(img1)) else 0.0)
        img2 = np.nan_to_num(img2, nan=float(np.nanmean(img2)) if not np.all(np.isnan(img2)) else 0.0)
        img1 = self._normalize_minmax(img1)
        img2 = self._normalize_minmax(img2)

        nlevels_dtcwt = 4
        min_dim = min(img1.shape)
        if self._use_dtcwt and min_dim >= (2 ** nlevels_dtcwt + 1):
            return self._cwssim_local_dtcwt(img1, img2, win_size, nlevels_dtcwt)
        return self._cwssim_local_gabor(img1, img2, win_size)

    def _cwssim_local_dtcwt(
        self, img1: np.ndarray, img2: np.ndarray, win_size: int,
        nlevels: int = 4
    ) -> Tuple[float, np.ndarray]:
        from scipy.ndimage import zoom as nd_zoom
        orig_shape = img1.shape
        img1 = self._pad_to_even(img1)
        img2 = self._pad_to_even(img2)
        transform = dtcwt.Transform2d()  # type: ignore[attr-defined]
        c1 = transform.forward(img1, nlevels=nlevels)
        c2 = transform.forward(img2, nlevels=nlevels)

        level_maps = []
        for lev in range(nlevels):
            hp1, hp2 = c1.highpasses[lev], c2.highpasses[lev]
            lev_map = np.zeros(hp1.shape[:2])
            for o in range(hp1.shape[2]):
                cx, cy = hp1[:, :, o], hp2[:, :, o]
                cross_real = uniform_filter(np.real(cx * np.conj(cy)), win_size, mode='nearest')
                cross_imag = uniform_filter(np.imag(cx * np.conj(cy)), win_size, mode='nearest')
                cross_mag = np.sqrt(cross_real ** 2 + cross_imag ** 2)
                auto1 = uniform_filter(np.abs(cx) ** 2, win_size, mode='nearest')
                auto2 = uniform_filter(np.abs(cy) ** 2, win_size, mode='nearest')
                num = 2.0 * cross_mag + self.K
                den = auto1 + auto2 + self.K
                lev_map += num / den
            lev_map /= hp1.shape[2]
            scale = np.array(orig_shape, dtype=float) / np.array(lev_map.shape, dtype=float)
            level_maps.append(nd_zoom(lev_map, scale, order=1))
        local_map = np.mean(level_maps, axis=0)[:orig_shape[0], :orig_shape[1]]
        return float(np.mean(local_map)), local_map

    def _cwssim_local_gabor(
        self, img1: np.ndarray, img2: np.ndarray, win_size: int
    ) -> Tuple[float, np.ndarray]:
        from scipy.ndimage import convolve
        nlevels = 4
        orientations = np.linspace(0, np.pi, self.n_orientations, endpoint=False)
        scales = [2 ** i for i in range(1, nlevels + 1)]
        level_maps = []
        for scale in scales:
            sigma = scale * 0.8
            freq = 1.0 / scale
            ksize = max(3, int(sigma * 4) | 1)
            y, x = np.mgrid[-ksize//2:ksize//2+1, -ksize//2:ksize//2+1].astype(np.float64)
            orient_sum = np.zeros_like(img1, dtype=np.float64)
            for theta in orientations:
                x_rot = x * np.cos(theta) + y * np.sin(theta)
                gaussian = np.exp(-(x_rot ** 2 + y ** 2) / (2 * sigma ** 2))
                sinusoid = np.exp(1j * 2 * np.pi * freq * x_rot)
                kernel = gaussian * sinusoid
                kernel /= np.abs(kernel).sum() + 1e-12
                c1 = convolve(img1, np.real(kernel)) + 1j * convolve(img1, np.imag(kernel))
                c2 = convolve(img2, np.real(kernel)) + 1j * convolve(img2, np.imag(kernel))
                cross_real = uniform_filter(np.real(c1 * np.conj(c2)), win_size, mode='nearest')
                cross_imag = uniform_filter(np.imag(c1 * np.conj(c2)), win_size, mode='nearest')
                cross_mag = np.sqrt(cross_real ** 2 + cross_imag ** 2)
                auto1 = uniform_filter(np.abs(c1) ** 2, win_size, mode='nearest')
                auto2 = uniform_filter(np.abs(c2) ** 2, win_size, mode='nearest')
                orient_sum += (2.0 * cross_mag + self.K) / (auto1 + auto2 + self.K)
            level_maps.append(orient_sum / len(orientations))
        local_map = np.mean(level_maps, axis=0)
        return float(np.mean(local_map)), local_map


# ==================== 主分析类 ====================

class Huang2024Reproduction:
    """Huang et al. (2024) 论文复现 — USTClitho2.0 vs FWEA18"""

    def __init__(self):
        self.base_config = BaseConfig()
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.Huang2024',
            'INFO',
        )
        self.models_dir = self.base_config.dirs['models'] / 'processed'
        self.output_dir = self.base_config.dirs['results'] / 'Huang2024_reproduction'
        self.figures_dir = self.base_config.dirs['figures'] / 'Huang2024_reproduction'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, xr.Dataset] = {}
        self.results: Dict[str, Any] = {}

        self.cwssim = CWSSIMCalculator(n_orientations=16, K=0)

        # 目标网格 (论文 0.5°)
        self.lons = np.arange(
            HUANG2024_REGION['lon_min'],
            HUANG2024_REGION['lon_max'] + HUANG2024_GRID_RES * 0.5,
            HUANG2024_GRID_RES,
        )
        self.lats = np.arange(
            HUANG2024_REGION['lat_min'],
            HUANG2024_REGION['lat_max'] + HUANG2024_GRID_RES * 0.5,
            HUANG2024_GRID_RES,
        )
        self.logger.info(
            f"📐 论文区域: {HUANG2024_REGION['lon_min']}°E–{HUANG2024_REGION['lon_max']}°E, "
            f"{HUANG2024_REGION['lat_min']}°N–{HUANG2024_REGION['lat_max']}°N"
        )
        self.logger.info(
            f"   网格: {len(self.lons)}×{len(self.lats)} ({HUANG2024_GRID_RES}°)"
        )

    def load_models(self):
        """加载 USTClitho2.0 和 FWEA18"""
        self.logger.info("\n📦 加载模型...")
        for key, info in MODELS_CONFIG.items():
            nc_path = self.models_dir / info['processed_dir'] / info['netcdf_file']
            if not nc_path.exists():
                self.logger.warning(f"⚠️ 文件不存在: {nc_path}")
                self.logger.warning("  请先运行 1_5 处理 USTClitho2.0 和 FWEA18:")
                self.logger.warning(
                    "  config.runtime['model_names'] = ['2022_USTClitho2.0', '2018_FWEA18']"
                )
                continue
            try:
                ds = xr.open_dataset(nc_path)
                self.models[key] = ds
                self.logger.info(
                    f"  ✅ {info['name']}: "
                    f"lon [{float(ds.longitude.min()):.1f}, {float(ds.longitude.max()):.1f}, "
                    f"lat [{float(ds.latitude.min()):.1f}, {float(ds.latitude.max()):.1f}], "
                    f"depth [{float(ds.depth.min()):.0f}, {float(ds.depth.max()):.0f}] km"
                )
            except Exception as e:
                self.logger.error(f"❌ 加载失败 {key}: {e}")

        if len(self.models) < 2:
            raise RuntimeError("至少需要 USTClitho2.0 和 FWEA18 两个模型")

    def _get_slice(
        self, model_key: str, param: str, depth_km: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取指定深度的 2D 切片，采样到论文网格。

        使用深度线性插值：当目标深度不在模型深度层中时，
        在相邻两层之间线性插值（而非最近邻），
        避免 FWEA18 在 5 km 时错误使用 0 km 地表层。
        """
        ds = self.models[model_key]
        use_param = param if param in ds else ('vs' if 'vs' in ds else 'vsv')
        if use_param not in ds:
            raise ValueError(f"模型 {model_key} 无参数 vs/vsv")

        data = ds[use_param].values
        lon = np.sort(ds.longitude.values)
        lat = np.sort(ds.latitude.values)
        depth = np.sort(ds.depth.values)

        if data.ndim != 3:
            raise ValueError("数据维度应为 3D")

        def _extract_depth_slice(d_idx: int) -> np.ndarray:
            if data.shape[0] == len(lon):
                return data[:, :, d_idx]
            return data[:, :, d_idx].T

        if depth_km in depth:
            d_idx = int(np.argmin(np.abs(depth - depth_km)))
            slice_2d = _extract_depth_slice(d_idx)
        else:
            idx_hi = int(np.searchsorted(depth, depth_km))
            if idx_hi == 0:
                slice_2d = _extract_depth_slice(0)
            elif idx_hi >= len(depth):
                slice_2d = _extract_depth_slice(len(depth) - 1)
            else:
                d_lo, d_hi = float(depth[idx_hi - 1]), float(depth[idx_hi])
                w = (depth_km - d_lo) / (d_hi - d_lo)
                sl_lo = _extract_depth_slice(idx_hi - 1)
                sl_hi = _extract_depth_slice(idx_hi)
                slice_2d = sl_lo * (1.0 - w) + sl_hi * w
                self.logger.debug(
                    f"  深度插值 {model_key}: {depth_km} km = "
                    f"{d_lo:.0f}×{1-w:.2f} + {d_hi:.0f}×{w:.2f}"
                )

        sampled = nearest_neighbor_sample(lon, lat, slice_2d, self.lons, self.lats)
        return sampled, lon, lat

    def compute_regional_cwssim(
        self, region_name: str, region: Dict
    ) -> Dict[float, float]:
        """计算子区域 CW-SSIM vs 深度"""
        lon_min, lon_max = region['lon']
        lat_min, lat_max = region['lat']
        mask_lon = (self.lons >= lon_min) & (self.lons <= lon_max)
        mask_lat = (self.lats >= lat_min) & (self.lats <= lat_max)
        sub_lons = self.lons[mask_lon]
        sub_lats = self.lats[mask_lat]

        results = {}
        m1_key, m2_key = list(self.models.keys())[:2]
        for depth_km in HUANG2024_DEPTHS:
            s1, _, _ = self._get_slice(m1_key, 'vs', depth_km)
            s2, _, _ = self._get_slice(m2_key, 'vs', depth_km)
            s1_sub = s1[np.ix_(mask_lon, mask_lat)]
            s2_sub = s2[np.ix_(mask_lon, mask_lat)]
            valid = np.isfinite(s1_sub) & np.isfinite(s2_sub)
            if np.sum(valid) < 10:
                continue
            s1_clean = np.where(valid, s1_sub, np.nanmean(s1_sub))
            s2_clean = np.where(valid, s2_sub, np.nanmean(s2_sub))
            cwssim = self.cwssim.compute_global(s1_clean, s2_clean)
            results[depth_km] = cwssim
        return results

    def compute_entire_region_cwssim_spatial(
        self, depth_km: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        全区域 CW-SSIM 空间分布 — 论文方法: 6°×6° 滑动窗口
        论文: "normalized each image with its maximum and minimum velocity values
        in the study area" — 先对整张切片做研究区 min-max 归一化，再在滑窗内算 CW-SSIM
        """
        m1_key, m2_key = list(self.models.keys())[:2]
        s1, _, _ = self._get_slice(m1_key, 'vs', depth_km)
        s2, _, _ = self._get_slice(m2_key, 'vs', depth_km)

        # 论文: "normalized each image with its maximum and minimum velocity values in the study area"
        vmin1, vmax1 = np.nanmin(s1), np.nanmax(s1)
        vmin2, vmax2 = np.nanmin(s2), np.nanmax(s2)
        dr1 = vmax1 - vmin1
        dr2 = vmax2 - vmin2
        if dr1 > 1e-12:
            s1_norm = np.where(np.isfinite(s1), (s1 - vmin1) / dr1, np.nan)
        else:
            s1_norm = np.zeros_like(s1)
        if dr2 > 1e-12:
            s2_norm = np.where(np.isfinite(s2), (s2 - vmin2) / dr2, np.nan)
        else:
            s2_norm = np.zeros_like(s2)

        box_deg = HUANG2024_BOX_SIZE_DEG
        n_per_deg = int(1.0 / HUANG2024_GRID_RES)
        box_px = int(box_deg * n_per_deg)
        if box_px % 2 == 0:
            box_px += 1
        half = box_px // 2

        cwssim_grid = np.full((len(self.lons), len(self.lats)), np.nan)
        n_total = box_px * box_px
        min_valid = int(n_total * 0.5)  # patch 至少 50% 有效数据
        ij_list = [
            (i, j) for i in range(half, len(self.lons) - half)
            for j in range(half, len(self.lats) - half)
        ]
        for i, j in tqdm(ij_list, desc=f"  {depth_km} km", leave=False):
            patch1 = s1_norm[i - half:i + half + 1, j - half:j + half + 1]
            patch2 = s2_norm[i - half:i + half + 1, j - half:j + half + 1]
            valid = np.isfinite(patch1) & np.isfinite(patch2)
            if np.sum(valid) < min_valid:
                continue
            p1 = np.where(valid, patch1, np.nanmean(patch1))
            p2 = np.where(valid, patch2, np.nanmean(patch2))
            cwssim_grid[i, j] = self.cwssim.compute_global(p1, p2, normalize=False)

        return cwssim_grid, s1, s2

    def run_all(self):
        """执行完整分析"""
        self.load_models()

        m1_key, m2_key = list(self.models.keys())[:2]
        pair_name = f"{MODELS_CONFIG[m1_key]['name']}-{MODELS_CONFIG[m2_key]['name']}"

        self.logger.info("\n" + "=" * 60)
        self.logger.info("🔬 区域 CW-SSIM (论文 Figure 5–8)")
        self.logger.info("=" * 60)
        for region_name, region in HUANG2024_SUBREGIONS.items():
            res = self.compute_regional_cwssim(region_name, region)
            self.results[f'regional_{region_name}'] = res
            for d, v in res.items():
                self.logger.info(f"  {region_name} @ {d} km: CW-SSIM = {v:.4f}")

        self.logger.info("\n" + "=" * 60)
        self.logger.info("🔬 全区域 CW-SSIM 空间分布 (论文 Figure 9–11)")
        self.logger.info("   ⏳ 约 91×65 网格×5 深度，每点 6°×6° patch，预计 10–30 分钟")
        self.logger.info("=" * 60)
        for depth_km in tqdm(HUANG2024_DEPTHS, desc="深度层"):
            cwssim_grid, s1, s2 = self.compute_entire_region_cwssim_spatial(depth_km)
            self.results[f'spatial_{depth_km}km'] = {
                'cwssim_grid': cwssim_grid,
                'lons': self.lons,
                'lats': self.lats,
            }
            valid = np.isfinite(cwssim_grid)
            if np.any(valid):
                mean_val = np.nanmean(cwssim_grid)
                self.logger.info(f"  {depth_km} km: mean CW-SSIM = {mean_val:.4f}")

        self.logger.info("\n✅ 分析完成")

    def save_results(self):
        """保存数值结果与空间网格（供 plot_only 复用）"""
        rows = []
        for key, val in self.results.items():
            if isinstance(val, dict) and 'depth' not in str(val):
                if 'cwssim_grid' in val:
                    continue
                for d, v in val.items():
                    rows.append({'region': key, 'depth_km': d, 'cwssim': v})
        if rows:
            df = pd.DataFrame(rows)
            out = self.output_dir / 'Huang2024_results.csv'
            df.to_csv(out, index=False, float_format='%.6f')
            self.logger.info(f"  ✅ {out}")

        # 保存空间网格数据，供 --plot-only 直接画图
        spatial_data = {k: v for k, v in self.results.items() if 'spatial_' in k and isinstance(v, dict)}
        if spatial_data:
            npz_path = self.output_dir / 'Huang2024_spatial_results.npz'
            save_dict = {}
            for k, v in spatial_data.items():
                save_dict[f'{k}_grid'] = v['cwssim_grid']
            save_dict['lons'] = self.lons
            save_dict['lats'] = self.lats
            np.savez_compressed(npz_path, **save_dict)
            self.logger.info(f"  ✅ {npz_path} (供 --plot-only 使用)")

    def load_saved_results(self) -> bool:
        """从磁盘加载已保存结果，用于仅画图模式。返回是否加载成功。
        支持部分加载：仅有 CSV 时可画深度曲线；有 npz 时可画全部图表。"""
        npz_path = self.output_dir / 'Huang2024_spatial_results.npz'
        csv_path = self.output_dir / 'Huang2024_results.csv'

        if not csv_path.exists() and not npz_path.exists():
            self.logger.warning("⚠️ 未找到任何已保存结果，请先完整运行计算")
            return False

        # 加载空间网格（若有 npz）
        if npz_path.exists():
            data = np.load(npz_path)
            self.lons = data['lons']
            self.lats = data['lats']
            for depth_km in HUANG2024_DEPTHS:
                key = f'spatial_{depth_km}km_grid'
                if key in data:
                    self.results[f'spatial_{depth_km}km'] = {
                        'cwssim_grid': data[key],
                        'lons': self.lons,
                        'lats': self.lats,
                    }
            self.logger.info(f"  ✅ 已加载 {npz_path}")
        else:
            self.logger.warning(f"⚠️ 未找到 {npz_path}，仅可画深度曲线（空间图需完整运行）")

        # 加载区域深度曲线（CSV 中 region 已含 regional_ 前缀）
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                rk = str(row['region'])  # 如 regional_Southeast_China
                if rk not in self.results:
                    self.results[rk] = {}
                self.results[rk][int(row['depth_km'])] = float(row['cwssim'])
            self.logger.info(f"  ✅ 已加载 {csv_path}")

        return True

    def plot_all(self):
        """生成所有图表。若仅有 CSV 无 npz，则只画深度曲线。"""
        self.logger.info("\n🎨 生成图表...")

        # 1. 深度曲线（依赖 CSV / regional_*）
        self._plot_depth_curve()
        # 2. 空间分布（依赖 npz）
        if any(k.startswith('spatial_') for k in self.results):
            self._plot_spatial_maps()
        else:
            self.logger.info("  ⏭️ 跳过空间分布图（无 spatial 数据）")
        # 3. 直方图（依赖 npz）
        if any(k.startswith('spatial_') for k in self.results):
            self._plot_histogram()
        else:
            self.logger.info("  ⏭️ 跳过直方图（无 spatial 数据）")

    def _plot_depth_curve(self):
        """CW-SSIM vs 深度 (论文 Figure 6, 8)"""
        fig, ax = plt.subplots(figsize=(8, 6))
        for region_name, res in self.results.items():
            if not region_name.startswith('regional_'):
                continue
            depths = sorted(res.keys())
            vals = [res[d] for d in depths]
            ax.plot(depths, vals, '-o', markersize=6, linewidth=2,
                    label=region_name.replace('regional_', ''))
        ax.set_xlabel('Depth (km)', fontsize=12)
        ax.set_ylabel('CW-SSIM', fontsize=12)
        ax.set_title(
            'Huang et al. (2024) Reproduction\n'
            'USTClitho2.0 vs FWEA18 — CW-SSIM vs Depth',
            fontsize=14, fontweight='bold',
        )
        ax.set_xlim(0, 160)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(
            self.figures_dir / 'Huang2024_cwssim_depth_curve.png',
            dpi=300, bbox_inches='tight',
        )
        plt.close(fig)
        self.logger.info("  ✅ Huang2024_cwssim_depth_curve.png")

    def _plot_spatial_maps(self):
        """全区域 CW-SSIM 空间分布 (论文 Figure 9–11)，使用 PyGMT 绘制"""
        region = [95, 133, 22, 48]
        n_depths = len(HUANG2024_DEPTHS)
        n_cols = 3
        n_rows = (n_depths + n_cols - 1) // n_cols

        if HAS_PYGMT:
            self._plot_spatial_maps_pygmt(region, n_rows, n_cols)
        else:
            self._plot_spatial_maps_matplotlib(region, n_rows, n_cols)

    def _plot_spatial_maps_pygmt(
        self, region: List[float], n_rows: int, n_cols: int
    ):
        """PyGMT 绘制空间分布图 — Lambert 圆锥投影（复现 Huang et al. 2024 Figure 11）"""
        if not HAS_PYGMT or pygmt is None:
            self._plot_spatial_maps_matplotlib(region, n_rows, n_cols)
            return

        m1_key, m2_key = list(MODELS_CONFIG.keys())[:2]
        pair_name = f"{MODELS_CONFIG[m1_key]['name']}-{MODELS_CONFIG[m2_key]['name']}"

        # Lambert Conformal Conic — 中国大陆标准配置 (匹配论文 Figure 9-12)
        # L<lon0>/<lat0>/<lat1>/<lat2>/<width>
        projection = "L112.5/34/27/43/?"

        pygmt.config(
            MAP_FRAME_TYPE="plain",
            MAP_GRID_PEN_PRIMARY="0.15p,gray70,.",
            FONT_ANNOT_PRIMARY="7p,Helvetica,black",
            FONT_LABEL="8p,Helvetica,black",
            FONT_TITLE="9p,Helvetica,black",
            MAP_FRAME_PEN="0.6p,black",
            MAP_TITLE_OFFSET="0.1c",
        )

        # CW-SSIM 色标 (红→白→蓝, 0.3–0.7, 与论文 Figure 11 一致)
        cpt_file = str(self.output_dir / "Huang2024_cwssim.cpt")
        pygmt.makecpt(cmap="polar", series="0.3/0.7/0.001", reverse=True, output=cpt_file)

        # 构造块体边界 (虚线)
        tectonic_file = (
            Path(__file__).parent.parent
            / "5_Visualization" / "data_EastAsia"
            / "china-geospatial-data-UTF8" / "CN-block-L1.gmt"
        )

        # 地质单元标注位置 (Huang et al. 2024 Figure 11)
        region_labels = {
            'TP':  (94, 31),
            'QB':  (96, 37),
            'OB':  (108, 38),
            'AB':  (113, 44),
            'SLB': (125, 46),
            'BB':  (118, 39),
            'HBP': (116, 35),
            'SCB': (105, 29),
            'SC':  (111, 25),
        }

        fig = pygmt.Figure()

        with fig.subplot(
            nrows=n_rows,
            ncols=n_cols,
            figsize=("25c", "18c"),
            margins=["0.3c", "0.6c"],
            autolabel="(a)+jTL+o0.15c/0.15c",
            frame=["a10f5", "WSne"],
        ):
            for idx, depth_km in enumerate(HUANG2024_DEPTHS):
                key = f'spatial_{depth_km}km'
                if key not in self.results:
                    continue
                data = self.results[key]
                cwssim_grid = data['cwssim_grid'].copy()
                lons, lats = data['lons'], data['lats']

                # Gaussian 预平滑: 填充 NaN → 平滑 → 恢复边界 NaN
                # sigma=2 grid points = 1° (保留更多空间细节)
                valid_mask = np.isfinite(cwssim_grid)
                fill_val = float(np.nanmean(cwssim_grid))
                smooth_arr = np.where(valid_mask, cwssim_grid, fill_val)
                smooth_arr = gaussian_filter(smooth_arr, sigma=2.0)
                cwssim_smoothed = np.where(valid_mask, smooth_arr, np.nan)

                lon_idx, lat_idx = np.where(np.isfinite(cwssim_smoothed))
                xyz = pd.DataFrame({
                    'x': lons[lon_idx],
                    'y': lats[lat_idx],
                    'z': cwssim_smoothed[np.isfinite(cwssim_smoothed)],
                })

                surf_region = [
                    region[0] - 3, region[1] + 3,
                    region[2] - 3, region[3] + 3,
                ]
                smoothed_grid = pygmt.surface(
                    data=xyz,
                    spacing="0.2",
                    region=surf_region,
                    tension=0.25,
                )
                smoothed_grid = smoothed_grid.clip(0.2, 0.85)

                with fig.set_panel(panel=idx):
                    fig.grdimage(
                        grid=smoothed_grid,
                        region=region,
                        projection=projection,
                        cmap=cpt_file,
                        frame=[
                            f"+t{pair_name} ({depth_km} km)",
                            "a10f5",
                        ],
                    )
                    fig.coast(
                        resolution="intermediate",
                        shorelines="0.3p,black",
                    )
                    if tectonic_file.exists():
                        fig.plot(data=str(tectonic_file), pen="0.6p,black,--")
                    for label, (lon, lat) in region_labels.items():
                        if (region[0] <= lon <= region[1]
                                and region[2] <= lat <= region[3]):
                            fig.text(
                                x=lon, y=lat,
                                text=label,
                                font="9p,Helvetica-Bold,black",
                                justify="CM",
                            )

        # 共享色标 — 底部居中水平
        fig.colorbar(
            cmap=cpt_file,
            position="JBC+o0c/1c+w12c/0.35c+h",
            frame=["xa0.1f0.05", "+lCWSSIM_INDEX"],
        )

        out_png = self.figures_dir / 'Huang2024_cwssim_spatial_USTClitho2.0_FWEA18.png'
        out_pdf = self.figures_dir / 'Huang2024_cwssim_spatial_USTClitho2.0_FWEA18.pdf'
        fig.savefig(str(out_png), dpi=300, crop=True)
        fig.savefig(str(out_pdf), dpi=300, crop=True)
        self.logger.info("  ✅ Huang2024_cwssim_spatial (Lambert, PyGMT)")

    def _plot_spatial_maps_matplotlib(
        self, region: List[float], n_rows: int, n_cols: int
    ):
        """Matplotlib/Cartopy 备选绘制"""
        lon_min, lon_max, lat_min, lat_max = region
        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

        for idx, depth_km in enumerate(HUANG2024_DEPTHS):
            key = f'spatial_{depth_km}km'
            if key not in self.results:
                continue
            data = self.results[key]
            cwssim_grid = data['cwssim_grid']
            lons, lats = data['lons'], data['lats']

            if HAS_CARTOPY:
                proj = ccrs.PlateCarree()
                ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection=proj)
                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
                ax.coastlines(linewidth=0.8, color='black')
                ax.add_feature(
                    cfeature.BORDERS,
                    linestyle='--',
                    edgecolor='black',
                    linewidth=0.6,
                    alpha=0.8,
                )
                plot_data = np.ma.masked_invalid(cwssim_grid.T)
                im = ax.pcolormesh(
                    lons, lats, plot_data,
                    cmap='RdYlBu_r',
                    vmin=0.3,
                    vmax=0.7,
                    shading='nearest',
                    transform=proj,
                )
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.4, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
            else:
                ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                plot_data = np.ma.masked_invalid(cwssim_grid.T)
                im = ax.pcolormesh(
                    lons, lats, plot_data,
                    cmap='RdYlBu_r',
                    vmin=0.3,
                    vmax=0.7,
                    shading='nearest',
                )
                ax.set_xlabel('Longitude (°E)')
                ax.set_ylabel('Latitude (°N)')
                ax.set_xlim(lon_min, lon_max)
                ax.set_ylim(lat_min, lat_max)
                ax.set_aspect('equal')

            ax.set_title(f'{depth_km} km  (mean={np.nanmean(cwssim_grid):.3f})')
            plt.colorbar(im, ax=ax, shrink=0.8, label='CW-SSIM')
        fig.suptitle(
            'Huang et al. (2024) Reproduction\n'
            'USTClitho2.0 vs FWEA18 — CW-SSIM Spatial Distribution (6°×6° box)',
            fontsize=14, fontweight='bold', y=1.02,
        )
        plt.tight_layout()
        fig.savefig(
            self.figures_dir / 'Huang2024_cwssim_spatial_USTClitho2.0_FWEA18.png',
            dpi=300, bbox_inches='tight',
        )
        plt.close(fig)
        self.logger.info("  ✅ Huang2024_cwssim_spatial_USTClitho2.0_FWEA18.png (Matplotlib)")

    def _plot_histogram(self):
        """CW-SSIM 直方图 (论文 Figure 13)"""
        all_vals = []
        for depth_km in HUANG2024_DEPTHS:
            key = f'spatial_{depth_km}km'
            if key not in self.results:
                continue
            grid = self.results[key]['cwssim_grid']
            valid = np.isfinite(grid)
            all_vals.extend(grid[valid].ravel().tolist())

        if not all_vals:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_vals, bins=20, range=(0.1, 0.9), edgecolor='black', alpha=0.7)
        ax.set_xlabel('CW-SSIM', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(
            'Huang et al. (2024) Reproduction\n'
            'USTClitho2.0 vs FWEA18 — CW-SSIM Histogram (all depths)',
            fontsize=14, fontweight='bold',
        )
        ax.axvline(float(np.mean(all_vals)), color='red', linestyle='--', label=f'Mean={np.mean(all_vals):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(
            self.figures_dir / 'Huang2024_cwssim_histogram.png',
            dpi=300, bbox_inches='tight',
        )
        plt.close(fig)
        self.logger.info("  ✅ Huang2024_cwssim_histogram.png")


# ==================== 主函数 ====================

def main():
    """主函数。支持 --plot-only 从已保存结果直接画图，跳过耗时计算。"""
    plot_only = '--plot-only' in sys.argv or '-p' in sys.argv

    print("\n" + "=" * 70)
    print("🚀 Huang et al. (2024) 论文复现")
    print("   Structural similarity of lithospheric velocity models of Chinese mainland")
    print("   USTClitho2.0 vs FWEA18")
    if plot_only:
        print("   [仅画图模式 — 从 results/Huang2024_reproduction/ 加载]")
    print("=" * 70)
    print(f"\n📐 区域: {HUANG2024_REGION['lon_min']}°E–{HUANG2024_REGION['lon_max']}°E, "
          f"{HUANG2024_REGION['lat_min']}°N–{HUANG2024_REGION['lat_max']}°N")
    print(f"📏 深度: {HUANG2024_DEPTHS} km")
    if not plot_only:
        print(f"📏 网格: {HUANG2024_GRID_RES}°")
        print(f"📏 滑窗: {HUANG2024_BOX_SIZE_DEG}°×{HUANG2024_BOX_SIZE_DEG}°")
        backend = 'Steerable Pyramid (pyrtools)' if HAS_PYRTOOLS else ('dtcwt' if HAS_DTCWT else 'Gabor')
        print(f"🔧 CW-SSIM 后端: {backend}")
    print(f"🗺️ 空间图绘制: {'PyGMT' if HAS_PYGMT else 'Matplotlib (无 PyGMT)'}")
    print("=" * 70 + "\n")

    try:
        repro = Huang2024Reproduction()
        if plot_only:
            if not repro.load_saved_results():
                print("❌ 请先完整运行一次以生成 Huang2024_spatial_results.npz")
                return
            repro.plot_all()
        else:
            repro.run_all()
            repro.save_results()
            repro.plot_all()

        print("\n" + "=" * 70)
        print("✅ 完成!")
        print(f"📁 结果: {repro.output_dir}")
        print(f"📁 图表: {repro.figures_dir}")
        print("\n💡 仅重新画图: python 2_2_Huang2024_reproduction.py --plot-only")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
