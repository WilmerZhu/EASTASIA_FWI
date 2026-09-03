"""
EASTASIA-FWI 速度相聚类可视化模块
====================================================================

功能描述:
- K 选择诊断图: BIC 曲线、拐点弦距、四带合并面板
- 聚类结果图: 深度切片、垂直剖面、簇剖面、特征分布、簇中心
- 结构对照: Slab2 俯冲板片几何、地质省与 CN 地块边界叠绘（均为定性对照）
- GMM 与 KMeans 对比图、后验概率分布图

科学原理:
- facies 解释标注依据 δlnVp/δlnVs 的联合符号与幅值（见 interpret_facies）
- Slab2 与地质边界仅作定性对照，不参与任何定量评分

说明:
本模块自 2_3_Model_clustering.py 拆分而来，仅承担可视化职责。配置对象为
2_3 中定义的 ClusteringConfig；为避免循环导入，此处以 Any 标注其类型。

作者: EASTASIA-FWI Team
日期: 2026-09-02
版本: v1.0
"""

import sys
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import BaseConfig

# 与 2_3_Model_clustering 保持一致的随机种子，确保抽样绘图可复现
RANDOM_SEED = 42



def apply_clustering_plot_style(config: Optional[Any] = None) -> Dict[str, float]:
    """
    统一聚类可视化 Matplotlib 字号（rcParams + 返回字号字典供显式调用）。

    Returns:
        fonts: 含 base/tick/label/title/suptitle/legend/annotation/clabel
    """
    defaults = {
        'base': 16,
        'tick': 15,
        'label': 17,
        'title': 19,
        'suptitle': 24,
        'legend': 15,
        'annotation': 14,
        'clabel': 13,
    }
    fonts = dict(defaults)
    if config is not None:
        fonts.update(config.visualization.get('fonts') or {})
    plt.rcParams.update({
        'font.size': float(fonts['base']),
        'axes.titlesize': float(fonts['title']),
        'axes.labelsize': float(fonts['label']),
        'xtick.labelsize': float(fonts['tick']),
        'ytick.labelsize': float(fonts['tick']),
        'legend.fontsize': float(fonts['legend']),
        'figure.titlesize': float(fonts['suptitle']),
    })
    return fonts


def overlay_slab_on_section(
    ax,
    sec_mask: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    *,
    with_legend: bool = False,
    legend_fontsize: float = 12,
) -> None:
    """
    在经度–深度剖面上叠绘 Slab2(dep+thk) 体：灰罩、轮廓、顶/底线。

    Args:
        sec_mask: (n_lon, n_depth) 布尔掩膜
        lons, depths: 1D 坐标
    """
    if sec_mask is None or not np.any(sec_mask):
        return
    extent = [
        float(np.min(lons)),
        float(np.max(lons)),
        float(np.max(depths)),
        float(np.min(depths)),
    ]
    shade = np.ma.masked_where(~sec_mask.T, np.ones(sec_mask.T.shape))
    ax.imshow(
        shade,
        aspect='auto',
        origin='upper',
        cmap='Greys',
        interpolation='nearest',
        extent=extent,
        vmin=0,
        vmax=1,
        alpha=0.12,
        zorder=3,
    )
    Lon, Dep = np.meshgrid(lons, depths)
    ax.contour(
        Lon,
        Dep,
        sec_mask.T.astype(float),
        levels=[0.5],
        colors='k',
        linewidths=0.7,
        zorder=5,
    )
    has = np.any(sec_mask, axis=1)
    z_top = np.full(len(lons), np.nan)
    z_bot = np.full(len(lons), np.nan)
    for j in range(len(lons)):
        if not has[j]:
            continue
        zs = depths[sec_mask[j, :]]
        z_top[j] = float(np.min(zs))
        z_bot[j] = float(np.max(zs))
    ax.plot(
        lons,
        z_top,
        color='k',
        linewidth=0.7,
        linestyle='-',
        zorder=6,
        label='Slab top',
    )
    ax.plot(
        lons,
        z_bot,
        color='k',
        linewidth=0.7,
        linestyle='--',
        zorder=6,
        label='Slab bottom',
    )
    if with_legend:
        ax.legend(loc='lower left', fontsize=legend_fontsize, framealpha=0.85)


def perturbation_clim(
    cube: np.ndarray, feat_idx: int, pct: float = 98.0
) -> Tuple[float, float]:
    """对称色标：±百分位 |δ|。"""
    vals = cube[:, :, :, feat_idx]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -0.05, 0.05
    m = float(np.nanpercentile(np.abs(vals), pct))
    m = max(m, 1e-4)
    return -m, m


# ==================== 等大地图子图网格 ====================
def make_equal_map_subplots(
    nrows: int,
    ncols: int,
    ccrs=None,
    *,
    panel_w: float = 6.0,
    panel_h: float = 5.0,
    wspace: float = 0.28,
    hspace: float = 0.38,
    top: float = 0.92,
    bottom: float = 0.06,
    left: float = 0.05,
    right: float = 0.98,
):
    """
    创建等大子图网格（每格 panel_w × panel_h inch）。
    固定 GridSpec，避免 tight_layout 把子图挤成不同大小。
    """
    fig = plt.figure(figsize=(panel_w * ncols, panel_h * nrows))
    gs = fig.add_gridspec(
        nrows,
        ncols,
        wspace=wspace,
        hspace=hspace,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        width_ratios=[1] * ncols,
        height_ratios=[1] * nrows,
    )
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            if ccrs is not None:
                axes[r, c] = fig.add_subplot(
                    gs[r, c], projection=ccrs.PlateCarree()
                )
            else:
                axes[r, c] = fig.add_subplot(gs[r, c])
    return fig, axes


# ==================== 生成30+种颜色 ====================
def generate_distinct_colors(n_colors: int) -> List:
    """生成视觉上可区分的颜色"""
    if n_colors <= 10:
        return plt.cm.tab10(np.linspace(0, 1, n_colors))
    elif n_colors <= 20:
        return plt.cm.tab20(np.linspace(0, 1, n_colors))
    else:
        # 组合多个colormap
        base_colors = []
        base_colors.extend(plt.cm.tab20(np.linspace(0, 1, 20)))
        base_colors.extend(plt.cm.Set3(np.linspace(0, 1, 12)))
        
        if n_colors > 32:
            # 添加更多颜色
            base_colors.extend(plt.cm.Pastel1(np.linspace(0, 1, 9)))
            base_colors.extend(plt.cm.Pastel2(np.linspace(0, 1, 8)))
        
        # 选择前n个
        return base_colors[:n_colors]


# ==================== facies 自动解释（启发式，供剖面图标注） ====================
# 各深度带 δlnVs 分类阈值 (weak, strong)；数值为无量纲对数扰动。
# 依据：地壳扰动幅度 ~±10%，上地幔 ~±3%，过渡带/下地幔 ~±1–2%。
# 解释仅为基于扰动符号与幅度的推测标签，规则表见 2_3_Model_clustering_methods.md。
FACIES_DVS_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    'crust': (0.03, 0.10),
    'shallow': (0.010, 0.030),
    'lithosphere': (0.010, 0.030),
    'transition_zone': (0.005, 0.015),
    'lower_mantle': (0.004, 0.012),
}


def interpret_facies(band: str, dlnvp: float, dlnvs: float) -> str:
    """
    按深度带与簇平均扰动给出速度相的推测构造解释（英文短标签）。

    主判据为 δlnVs 的符号与幅度（对温度/部分熔融最敏感）；
    若 δlnVp 与 δlnVs 符号相反且幅度均显著，追加 [Vp-Vs decoupled]
    标记（提示成分/流体等非热成因）。

    Args:
        band: 深度带名（crust/shallow/lithosphere/transition_zone/lower_mantle）
        dlnvp: 簇平均 δlnVp
        dlnvs: 簇平均 δlnVs

    Returns:
        推测解释短标签（英文，用于图面标注）
    """
    b = str(band).lower()
    weak, strong = FACIES_DVS_THRESHOLDS.get(b, (0.010, 0.030))
    v = float(dlnvs) if np.isfinite(dlnvs) else 0.0

    if b == 'crust':
        if v <= -strong:
            name = 'Very slow crust (thick sediments / basin)'
        elif v <= -weak:
            name = 'Slow crust (sedimentary / warm)'
        elif v >= strong:
            name = 'Very fast crust (cratonic / mafic)'
        elif v >= weak:
            name = 'Fast crust (crystalline basement)'
        else:
            name = 'Average crust'
    elif b in ('lithosphere', 'shallow', 'all'):
        if v >= strong:
            name = 'Fast anomaly (cratonic root / slab)'
        elif v >= weak:
            name = 'Moderately fast mantle (cold)'
        elif v <= -strong:
            name = 'Slow anomaly (asthenospheric / back-arc)'
        elif v <= -weak:
            name = 'Moderately slow mantle (warm)'
        else:
            name = 'Ambient upper mantle'
    elif b == 'transition_zone':
        if v >= weak:
            name = 'Fast TZ anomaly (stagnant slab)'
        elif v <= -weak:
            name = 'Slow TZ anomaly (warm / hydrated)'
        else:
            name = 'Ambient transition zone'
    elif b == 'lower_mantle':
        if v >= weak:
            name = 'Fast anomaly (slab remnant)'
        elif v <= -weak:
            name = 'Slow anomaly (thermal upwelling)'
        else:
            name = 'Ambient lower mantle'
    else:
        name = 'Unclassified'

    if (
        np.isfinite(dlnvp)
        and abs(v) >= weak
        and abs(float(dlnvp)) >= 0.5 * weak
        and np.sign(float(dlnvp)) != np.sign(v)
    ):
        name += ' [Vp-Vs decoupled]'
    return name


class GeologyConcordanceEvaluator:
    """
    结构参考数据辅助类（仅定性叠图与空间分带，无定量评分）

    - Slab2 俯冲板片（dep/thk，与 5_1_Basemap 同源）：切片等深线、剖面
      体轮廓、三维体掩膜叠绘
    - Moho 点数据：插值成网格，仅供 moho_4band 空间分带
    - USGS 地质省 / CN 地块边界线：浅部切片叠绘
    """

    def __init__(self, config: Any, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.base_config = BaseConfig()
        self.fonts = apply_clustering_plot_style(config)
        self._moho_grid: Optional[np.ndarray] = None
        self._slab_grid: Optional[np.ndarray] = None
        self._moho_meta: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._slab_meta: Optional[Tuple[np.ndarray, np.ndarray]] = None
        # 三维板片体掩膜缓存: (mask_3d, lons, lats, depths)
        self._slab_vol_cache: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = None

    @staticmethod
    def _import_cartopy():
        """尝试导入 cartopy；失败则返回 (None, None)"""
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            return ccrs, cfeature
        except ImportError:
            return None, None

    def _add_coastlines(
        self,
        ax,
        extent: Sequence[float],
        ccrs=None,
        cfeature=None,
        *,
        draw_labels: bool = True,
        left_labels: bool = True,
        bottom_labels: bool = True,
        match_depth_slices: bool = False,
    ) -> None:
        """
        在地图轴上添加海岸线/国界，并标注经纬度。
        extent: [lon_min, lon_max, lat_min, lat_max]
        match_depth_slices: True 时样式对齐 2-3-4（线宽/字号）
        """
        lon0, lon1, lat0, lat1 = [float(x) for x in extent]
        if ccrs is not None and cfeature is not None:
            try:
                if match_depth_slices:
                    ax.coastlines(
                        resolution='50m', color='gray', linewidth=0.45, zorder=3
                    )
                    ax.add_feature(
                        cfeature.BORDERS,
                        linestyle='--',
                        linewidth=0.5,
                        edgecolor='gray',
                        zorder=3,
                    )
                    label_size = 15
                    gl_lw, gl_alpha = 0.5, 0.5
                else:
                    ax.coastlines(
                        resolution='50m', color='gray', linewidth=0.4, zorder=6
                    )
                    ax.add_feature(
                        cfeature.BORDERS,
                        linestyle=':',
                        linewidth=0.4,
                        edgecolor='gray',
                        zorder=5,
                    )
                    label_size = 8
                    gl_lw, gl_alpha = 0.4, 0.45
                ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
                gl = ax.gridlines(
                    crs=ccrs.PlateCarree(),
                    draw_labels=draw_labels,
                    linewidth=gl_lw,
                    color='gray',
                    alpha=gl_alpha,
                    linestyle='--',
                    zorder=4,
                )
                gl.top_labels = False
                gl.right_labels = False
                gl.left_labels = left_labels
                gl.bottom_labels = bottom_labels
                try:
                    from cartopy.mpl.gridliner import (
                        LONGITUDE_FORMATTER,
                        LATITUDE_FORMATTER,
                    )
                    gl.xformatter = LONGITUDE_FORMATTER
                    gl.yformatter = LATITUDE_FORMATTER
                except Exception:
                    pass
                gl.xlabel_style = {'size': label_size}
                gl.ylabel_style = {'size': label_size}
                # 与 2-3-4 一致：填满子图；固定框比例（高/宽=5/6，对应 6×5 inch）
                if match_depth_slices:
                    try:
                        ax.set_aspect('auto')
                        ax.set_box_aspect(5.0 / 6.0)
                    except Exception:
                        pass
                return
            except Exception:
                pass
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(lat0, lat1)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    def _make_map_subplots(self, nrows: int, ncols: int, ccrs=None, **kwargs):
        """创建等大子图网格（委托模块级函数）。"""
        return make_equal_map_subplots(nrows, ncols, ccrs, **kwargs)

    def _plot_facies_slice_imshow(
        self,
        ax,
        label_slice: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        cmap,
        n_clusters: int,
        ccrs=None,
    ) -> None:
        """
        与 2-3-4 深度切片相同的填色方式：
        imshow + aspect='auto'，填满子图，比例一致。
        """
        extent = [
            float(lons.min()),
            float(lons.max()),
            float(lats.min()),
            float(lats.max()),
        ]
        plot_kw: Dict[str, Any] = {}
        if ccrs is not None:
            plot_kw['transform'] = ccrs.PlateCarree()
        ax.imshow(
            label_slice,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            interpolation='nearest',
            extent=extent,
            vmin=-0.5,
            vmax=max(n_clusters - 0.5, 0.5),
            zorder=1,
            **plot_kw,
        )

    def get_moho_grid(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> Optional[np.ndarray]:
        """获取插值到模型网格的 Moho 深度场（仅用于 moho_4band 分带）"""
        self._ensure_moho_grid(lons, lats)
        return self._moho_grid

    def _ensure_moho_grid(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> None:
        """插值 Moho 到模型经纬网格（缓存；不用于符合度）"""
        if (
            self._moho_meta is not None
            and np.array_equal(self._moho_meta[0], lons)
            and np.array_equal(self._moho_meta[1], lats)
            and self._moho_grid is not None
        ):
            return

        geo_cfg = self.config.geology_concordance
        data_dir = self.base_config.dirs['data']
        moho_name = geo_cfg.get('moho_file', '莫霍面深度数据.txt')
        self._moho_grid = self._interpolate_points_to_grid(
            data_dir / moho_name, lons, lats, kind='moho'
        )
        self._moho_meta = (lons, lats)

    def _slab_data_root(self) -> Path:
        slab_cfg = self.config.geology_concordance.get('slab') or {}
        rel = slab_cfg.get(
            'vis_data_subdir', '5_Visualization/data_EastAsia/Slab2'
        )
        return self.base_config.dirs['project_root'] / rel

    def _ensure_slab_grid(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> None:
        """
        拼接东亚 Slab2 depth .grd → 模型网格正深度场 (km)。
        重叠区取最浅板片顶面（nanmin |z|）。
        """
        if (
            self._slab_meta is not None
            and np.array_equal(self._slab_meta[0], lons)
            and np.array_equal(self._slab_meta[1], lats)
            and self._slab_grid is not None
        ):
            return

        slab_cfg = self.config.geology_concordance.get('slab') or {}
        if not slab_cfg.get('enabled', True):
            self._slab_grid = None
            self._slab_meta = (lons, lats)
            return

        root = self._slab_data_root()
        dist = root / slab_cfg.get('distribute_subdir', 'Slab2Distribute_Mar2018')
        regions = list(
            slab_cfg.get(
                'regions',
                [
                    'kur', 'ryu', 'izu', 'man', 'phi', 'sum', 'sul',
                    'him', 'hin', 'mak', 'cot', 'hal', 'png', 'sol',
                ],
            )
        )
        use_abs = bool(slab_cfg.get('use_absolute_depth', True))
        pattern_tmpl = slab_cfg.get('grd_glob', '{region}_slab2_dep_*.grd')

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        composite = np.full(lon_grid.shape, np.nan, dtype=np.float64)
        n_loaded = 0

        for region in regions:
            matches = sorted(dist.glob(pattern_tmpl.format(region=region)))
            if not matches:
                continue
            grd_path = matches[0]
            try:
                with xr.open_dataset(grd_path) as ds:
                    if 'z' in ds:
                        da = ds['z']
                    else:
                        da = ds[list(ds.data_vars)[0]]
                    # 坐标名 x=lon, y=lat
                    xname = 'x' if 'x' in da.dims else da.dims[-1]
                    yname = 'y' if 'y' in da.dims else da.dims[0]
                    # 裁到模型范围附近以加速
                    pad = 2.0
                    da = da.sel(
                        {
                            xname: slice(
                                float(lons.min()) - pad, float(lons.max()) + pad
                            ),
                            yname: slice(
                                float(lats.min()) - pad, float(lats.max()) + pad
                            ),
                        }
                    )
                    if da.size == 0:
                        continue
                    interp = da.interp(
                        {xname: xr.DataArray(lons, dims='lon'),
                         yname: xr.DataArray(lats, dims='lat')},
                        method='linear',
                    )
                    # 结果维度 (lat, lon)
                    arr = np.asarray(interp.transpose('lat', 'lon').values, dtype=np.float64)
                if use_abs:
                    arr = np.abs(arr)
                # 无效值：Slab2 常用极大值/NaN
                arr[~np.isfinite(arr)] = np.nan
                arr[(arr <= 0) | (arr > 800)] = np.nan
                if not np.any(np.isfinite(arr)):
                    continue
                if np.all(np.isnan(composite)):
                    composite = arr
                else:
                    composite = np.where(
                        np.isnan(composite),
                        arr,
                        np.where(np.isnan(arr), composite, np.fmin(composite, arr)),
                    )
                n_loaded += 1
            except Exception as e:
                self.logger.debug(f"  Slab2 {grd_path.name} 读取失败: {e}")
                continue

        if n_loaded == 0 or not np.any(np.isfinite(composite)):
            self.logger.warning(
                f"  ⚠️ 未加载到有效 Slab2 网格（目录: {dist}）"
            )
            self._slab_grid = None
        else:
            self._slab_grid = composite.astype(np.float32)
            self.logger.info(
                f"  ✅ Slab2 深度场: {n_loaded} 个板片 → 网格 {composite.shape}, "
                f"深度 [{np.nanmin(composite):.1f}, {np.nanmax(composite):.1f}] km, "
                f"覆盖 {np.sum(np.isfinite(composite))}/{composite.size}"
            )
        self._slab_meta = (lons, lats)

    def _interp_slab_grd_to_model(
        self,
        grd_path: Path,
        lons: np.ndarray,
        lats: np.ndarray,
        *,
        use_absolute: bool = False,
        valid_range: Optional[Tuple[float, float]] = None,
    ) -> Optional[np.ndarray]:
        """将单个 Slab2 .grd 双线性插值到模型 (lat, lon) 网格"""
        try:
            with xr.open_dataset(grd_path) as ds:
                if 'z' in ds:
                    da = ds['z']
                else:
                    da = ds[list(ds.data_vars)[0]]
                xname = 'x' if 'x' in da.dims else da.dims[-1]
                yname = 'y' if 'y' in da.dims else da.dims[0]
                pad = 2.0
                da = da.sel(
                    {
                        xname: slice(
                            float(lons.min()) - pad, float(lons.max()) + pad
                        ),
                        yname: slice(
                            float(lats.min()) - pad, float(lats.max()) + pad
                        ),
                    }
                )
                if da.size == 0:
                    return None
                interp = da.interp(
                    {
                        xname: xr.DataArray(lons, dims='lon'),
                        yname: xr.DataArray(lats, dims='lat'),
                    },
                    method='linear',
                )
                arr = np.asarray(
                    interp.transpose('lat', 'lon').values, dtype=np.float64
                )
            if use_absolute:
                arr = np.abs(arr)
            arr[~np.isfinite(arr)] = np.nan
            if valid_range is not None:
                lo, hi = valid_range
                arr[(arr < lo) | (arr > hi)] = np.nan
            if not np.any(np.isfinite(arr)):
                return None
            return arr.astype(np.float32)
        except Exception as e:
            self.logger.debug(f"  Slab2 {grd_path.name} 插值失败: {e}")
            return None

    def build_slab_volume_mask(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        depths: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        用 Slab2 dep + thk 构建与聚类同维度的三维板片掩膜。

        对每个板片区域：z_top = |dep|, z_bot = z_top + thk；
        体素满足 z_top ≤ z ≤ z_bot 则记为板片内。多板片取并集。

        Returns:
            mask: (n_lat, n_lon, n_depth) bool，或不可用时 None
        """
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        depths = np.asarray(depths, dtype=float)

        if self._slab_vol_cache is not None:
            m, lo, la, de = self._slab_vol_cache
            if (
                np.array_equal(lo, lons)
                and np.array_equal(la, lats)
                and np.array_equal(de, depths)
            ):
                return m

        slab_cfg = self.config.geology_concordance.get('slab') or {}
        if not slab_cfg.get('enabled', True):
            return None

        root = self._slab_data_root()
        dist = root / slab_cfg.get('distribute_subdir', 'Slab2Distribute_Mar2018')
        regions = list(
            slab_cfg.get(
                'regions',
                [
                    'kur', 'ryu', 'izu', 'man', 'phi', 'sum', 'sul',
                    'him', 'hin', 'mak', 'cot', 'hal', 'png', 'sol',
                ],
            )
        )
        use_abs = bool(slab_cfg.get('use_absolute_depth', True))
        dep_tmpl = slab_cfg.get('grd_glob', '{region}_slab2_dep_*.grd')
        thk_tmpl = slab_cfg.get('thk_glob', '{region}_slab2_thk_*.grd')

        n_lat, n_lon, n_dep = len(lats), len(lons), len(depths)
        mask = np.zeros((n_lat, n_lon, n_dep), dtype=bool)
        # 深度广播: (1,1,nz)
        z_axis = depths.reshape(1, 1, -1)
        n_loaded = 0

        for region in regions:
            dep_matches = sorted(dist.glob(dep_tmpl.format(region=region)))
            thk_matches = sorted(dist.glob(thk_tmpl.format(region=region)))
            if not dep_matches or not thk_matches:
                continue
            dep = self._interp_slab_grd_to_model(
                dep_matches[0],
                lons,
                lats,
                use_absolute=use_abs,
                valid_range=(0.0, 800.0),
            )
            thk = self._interp_slab_grd_to_model(
                thk_matches[0],
                lons,
                lats,
                use_absolute=False,
                valid_range=(1.0, 300.0),
            )
            if dep is None or thk is None:
                continue
            valid = np.isfinite(dep) & np.isfinite(thk) & (thk > 0)
            if not np.any(valid):
                continue
            z_top = np.where(valid, dep, np.nan)[:, :, np.newaxis]
            z_bot = np.where(valid, dep + thk, np.nan)[:, :, np.newaxis]
            local = (
                np.isfinite(z_top)
                & (z_axis >= z_top)
                & (z_axis <= z_bot)
            )
            mask |= local
            n_loaded += 1

        if n_loaded == 0 or not np.any(mask):
            self.logger.warning("  ⚠️ 未能构建 Slab2 三维体掩膜（缺 dep/thk）")
            self._slab_vol_cache = None
            return None

        frac = float(np.mean(mask))
        self.logger.info(
            f"  ✅ Slab2 体掩膜: {n_loaded} 板片, 形状 {mask.shape}, "
            f"占体素 {100.0 * frac:.2f}% ({int(mask.sum())}/{mask.size})"
        )
        self._slab_vol_cache = (mask, lons.copy(), lats.copy(), depths.copy())
        return mask

    def plot_slab_volume_comparison(
        self,
        labels_3d: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Path,
        model_name: str,
        pert_cube: Optional[np.ndarray] = None,
    ) -> None:
        """
        三维同维度对比可视化：聚类体 vs Slab2(dep+thk) 体掩膜。
        输出: 2-3-11_slab_volume_3d
          行0 — 水平切片 + 板片体边界
          行1 — 聚类垂直剖面 + 板片
          行2–3 — δlnVp / δlnVs 垂直剖面 + 板片（若提供 pert_cube）
        不修改现有切片 MI/ρ 符合度逻辑。
        """
        self.fonts = apply_clustering_plot_style(self.config)
        slab_cfg = self.config.geology_concordance.get('slab') or {}
        vol_cfg = slab_cfg.get('volume_viz') or {}
        if not vol_cfg.get('enabled', True):
            return

        lons = np.asarray(metadata['lons'], dtype=float)
        lats = np.asarray(metadata['lats'], dtype=float)
        depths = np.asarray(metadata['depths'], dtype=float)
        mask = self.build_slab_volume_mask(lons, lats, depths)
        if mask is None:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        slice_zs = list(vol_cfg.get('slice_depths_km', [100.0, 200.0, 400.0]))[:3]
        sec_lats = list(vol_cfg.get('section_latitudes', [20.0, 30.0, 40.0]))[:3]
        n_col = max(len(slice_zs), len(sec_lats), 1)

        n_clusters = int(len(np.unique(labels_3d[labels_3d >= 0])))
        colors = generate_distinct_colors(max(n_clusters, 2))
        cmap = mcolors.ListedColormap(colors[: max(n_clusters, 1)])
        map_extent = [
            float(lons.min()),
            float(lons.max()),
            float(lats.min()),
            float(lats.max()),
        ]
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        sec_extent = [
            float(lons.min()),
            float(lons.max()),
            float(depths.max()),
            float(depths.min()),
        ]

        has_pert = (
            pert_cube is not None
            and np.ndim(pert_cube) == 4
            and pert_cube.shape[-1] >= 2
            and pert_cube.shape[:3] == labels_3d.shape
        )
        n_rows = 4 if has_pert else 2

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
            use_cartopy = True
        except ImportError:
            use_cartopy = False
            ccrs = cfeature = LONGITUDE_FORMATTER = LATITUDE_FORMATTER = None

        fig = plt.figure(figsize=(6.2 * n_col, 4.8 * n_rows))
        gs = fig.add_gridspec(
            n_rows,
            n_col,
            wspace=0.16,
            hspace=0.32,
            left=0.05,
            right=0.90 if has_pert else 0.98,
            top=0.93,
            bottom=0.04,
            width_ratios=[1] * n_col,
            height_ratios=[1.0] * n_rows,
        )
        projection = ccrs.PlateCarree() if use_cartopy else None
        plot_tf = {'transform': projection} if use_cartopy else {}

        # —— 上行：水平切片 ——
        for col, z_target in enumerate(slice_zs):
            if use_cartopy:
                ax = fig.add_subplot(gs[0, col], projection=projection)
            else:
                ax = fig.add_subplot(gs[0, col])
            z_idx = int(np.argmin(np.abs(depths - float(z_target))))
            actual_z = float(depths[z_idx])
            lab = np.ma.masked_where(
                labels_3d[:, :, z_idx] < 0, labels_3d[:, :, z_idx]
            )
            slab_slice = mask[:, :, z_idx]
            ax.imshow(
                lab,
                aspect='auto',
                origin='lower',
                cmap=cmap,
                interpolation='nearest',
                extent=map_extent,
                vmin=-0.5,
                vmax=max(n_clusters - 0.5, 0.5),
                zorder=1,
                **plot_tf,
            )
            if np.any(slab_slice):
                # 板片体水平截面：半透明灰罩 + 黑边界
                shade = np.ma.masked_where(~slab_slice, np.ones(slab_slice.shape))
                ax.imshow(
                    shade,
                    aspect='auto',
                    origin='lower',
                    cmap='Greys',
                    interpolation='nearest',
                    extent=map_extent,
                    vmin=0,
                    vmax=1,
                    alpha=0.12,
                    zorder=3,
                    **plot_tf,
                )
                ax.contour(
                    lon_grid,
                    lat_grid,
                    slab_slice.astype(float),
                    levels=[0.5],
                    colors='k',
                    linewidths=0.7,
                    zorder=5,
                    **plot_tf,
                )
            if use_cartopy:
                ax.coastlines(
                    resolution='50m', color='gray', linewidth=0.45, zorder=6
                )
                ax.add_feature(
                    cfeature.BORDERS,
                    linestyle='--',
                    linewidth=0.4,
                    edgecolor='gray',
                    zorder=5,
                )
                ax.set_extent(map_extent, crs=projection)
                try:
                    ax.set_aspect('auto')
                except Exception:
                    pass
                gl = ax.gridlines(
                    crs=projection,
                    draw_labels=True,
                    linewidth=0.4,
                    color='gray',
                    alpha=0.45,
                    linestyle='--',
                )
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 11}
                gl.ylabel_style = {'size': 11}
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
            else:
                ax.set_xlim(map_extent[0], map_extent[1])
                ax.set_ylim(map_extent[2], map_extent[3])
                ax.set_aspect('auto')

            # 与同列下行剖面一一对应：每个子图只画一条纬度切线（同 2-3-5）
            if col < len(sec_lats):
                lat_t = float(sec_lats[col])
                lat_idx = int(np.argmin(np.abs(lats - lat_t)))
                actual_sec_lat = float(lats[lat_idx])
                if use_cartopy:
                    ax.plot(
                        [lons.min(), lons.max()],
                        [actual_sec_lat, actual_sec_lat],
                        color='red',
                        linewidth=1.8,
                        linestyle='-',
                        transform=projection,
                        zorder=7,
                        label='Section Line',
                    )
                    ax.scatter(
                        [lons.min(), lons.max()],
                        [actual_sec_lat, actual_sec_lat],
                        c='red',
                        s=40,
                        marker='o',
                        edgecolors='black',
                        linewidths=0.8,
                        transform=projection,
                        zorder=8,
                    )
                else:
                    ax.axhline(
                        actual_sec_lat,
                        color='red',
                        linewidth=1.8,
                        label='Section Line',
                    )
                ax.legend(loc='upper right', fontsize=self.fonts['legend'], framealpha=0.85)

            ax.set_title(
                f'{actual_z:.0f} km',
                fontsize=self.fonts['title'],
                fontweight='bold',
                pad=4,
            )
            ax.tick_params(labelsize=self.fonts['tick'])

        for col in range(len(slice_zs), n_col):
            fig.add_subplot(gs[0, col]).axis('off')

        # —— 行1：聚类垂直剖面 ——
        for col, lat_t in enumerate(sec_lats):
            ax = fig.add_subplot(gs[1, col])
            lat_idx = int(np.argmin(np.abs(lats - float(lat_t))))
            actual_lat = float(lats[lat_idx])
            sec_lab = labels_3d[lat_idx, :, :]  # (lon, depth)
            sec_mask = mask[lat_idx, :, :]
            sec_plot = np.ma.masked_where(sec_lab < 0, sec_lab)
            ax.imshow(
                sec_plot.T,
                aspect='auto',
                origin='upper',
                cmap=cmap,
                interpolation='nearest',
                extent=sec_extent,
                vmin=-0.5,
                vmax=max(n_clusters - 0.5, 0.5),
                zorder=1,
            )
            overlay_slab_on_section(
                ax,
                sec_mask,
                lons,
                depths,
                with_legend=(col == 0),
                legend_fontsize=self.fonts['legend'],
            )
            F = self.fonts
            ax.set_xlabel('Longitude (°)', fontsize=F['label'])
            ax.set_ylabel('Depth (km)', fontsize=F['label'])
            ax.tick_params(labelsize=F['tick'])
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_title(
                f'Facies  |  Lat={actual_lat:.1f}°',
                fontsize=F['title'],
                fontweight='bold',
                pad=4,
            )
            ax.set_ylim(float(depths.max()), float(depths.min()))

        for col in range(len(sec_lats), n_col):
            fig.add_subplot(gs[1, col]).axis('off')

        # —— 行2–3：δlnVp / δlnVs 垂直剖面 ——
        if has_pert:
            feat_titles = [r'$\delta\ln V_p$', r'$\delta\ln V_s$']
            for r_off, (fi, ftitle) in enumerate(zip((0, 1), feat_titles)):
                row = 2 + r_off
                vmin, vmax = perturbation_clim(pert_cube, fi)
                last_im = None
                for col, lat_t in enumerate(sec_lats):
                    ax = fig.add_subplot(gs[row, col])
                    lat_idx = int(np.argmin(np.abs(lats - float(lat_t))))
                    actual_lat = float(lats[lat_idx])
                    sec = np.ma.masked_invalid(pert_cube[lat_idx, :, :, fi])
                    last_im = ax.imshow(
                        sec.T,
                        aspect='auto',
                        origin='upper',
                        cmap='RdBu_r',
                        interpolation='nearest',
                        extent=sec_extent,
                        vmin=vmin,
                        vmax=vmax,
                        zorder=1,
                    )
                    overlay_slab_on_section(
                        ax, mask[lat_idx, :, :], lons, depths, with_legend=False
                    )
                    ax.set_title(
                        f'{ftitle}  |  Lat={actual_lat:.1f}°',
                        fontsize=self.fonts['title'],
                        fontweight='bold',
                        pad=4,
                    )
                    ax.set_xlabel('Longitude (°)', fontsize=self.fonts['label'])
                    if col == 0:
                        ax.set_ylabel('Depth (km)', fontsize=self.fonts['label'])
                    ax.tick_params(labelsize=self.fonts['tick'])
                    ax.set_ylim(float(depths.max()), float(depths.min()))
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    if col == len(sec_lats) - 1 and last_im is not None:
                        cbar = fig.colorbar(
                            last_im, ax=ax, fraction=0.046, pad=0.02
                        )
                        cbar.set_label(ftitle, fontsize=self.fonts['label'])
                        cbar.ax.tick_params(labelsize=self.fonts['tick'])
                for col in range(len(sec_lats), n_col):
                    fig.add_subplot(gs[row, col]).axis('off')

        fig.suptitle(
            f'{model_name} — Facies / dlnV vs Slab2 Volume '
            f'(dep+thk mask; black = slab body)',
            fontsize=self.fonts['suptitle'],
            fontweight='bold',
            y=0.985,
        )

        stem = '2-3-11_slab_volume_3d'
        save_fmts = set(self.config.visualization.get('save_formats', ['jpg']))
        save_fmts.add('png')
        for fmt in sorted(save_fmts):
            fig.savefig(
                output_dir / f'{stem}.{fmt}',
                dpi=self.config.visualization['dpi'],
            )
        plt.close(fig)
        self.logger.info(f"      ✅ 保存: {stem} ({', '.join(sorted(save_fmts))})")

    def _interpolate_points_to_grid(
        self,
        path: Path,
        lons: np.ndarray,
        lats: np.ndarray,
        kind: str,
    ) -> Optional[np.ndarray]:
        """读取散点结构深度并插值到规则网格"""
        if not path.exists():
            self.logger.warning(f"  ⚠️ 结构文件不存在: {path}")
            return None
        try:
            df = pd.read_csv(path, sep=r'\s+', engine='python', encoding='utf-8')
            cols = list(df.columns)
            if len(cols) < 3:
                raise ValueError(f"列数不足: {cols}")
            lon_col, lat_col, depth_col = cols[0], cols[1], cols[2]
            pts = df[[lon_col, lat_col, depth_col]].apply(pd.to_numeric, errors='coerce')
            pts = pts.dropna()
            # 限制到模型范围附近
            lon_min, lon_max = float(lons.min()), float(lons.max())
            lat_min, lat_max = float(lats.min()), float(lats.max())
            pad = 2.0
            pts = pts[
                (pts[lon_col] >= lon_min - pad)
                & (pts[lon_col] <= lon_max + pad)
                & (pts[lat_col] >= lat_min - pad)
                & (pts[lat_col] <= lat_max + pad)
            ]
            if len(pts) < 20:
                self.logger.warning(f"  ⚠️ {kind} 有效点过少 ({len(pts)})")
                return None

            lon_grid, lat_grid = np.meshgrid(lons, lats)
            grid = griddata(
                pts[[lon_col, lat_col]].values,
                pts[depth_col].values,
                (lon_grid, lat_grid),
                method='linear',
            )
            # 线性插值空洞用最近邻填补
            nan_mask = ~np.isfinite(grid)
            if np.any(nan_mask):
                grid_nn = griddata(
                    pts[[lon_col, lat_col]].values,
                    pts[depth_col].values,
                    (lon_grid, lat_grid),
                    method='nearest',
                )
                grid[nan_mask] = grid_nn[nan_mask]

            self.logger.info(
                f"  ✅ {kind} 插值完成: {len(pts)} 点 → "
                f"网格 {grid.shape}, 深度 "
                f"[{np.nanmin(grid):.1f}, {np.nanmax(grid):.1f}] km"
            )
            return grid.astype(np.float32)
        except Exception as e:
            self.logger.warning(f"  ⚠️ 读取/插值 {kind} 失败: {e}")
            return None

    # ------------------------------------------------------------------ #
    # 地质省 / CN 地块边界线叠图工具（供 3×3 深度切片图复用）
    # ------------------------------------------------------------------ #
    def _geology_dir(self) -> Path:
        surf = (self.config.geology_concordance.get('surface_units') or {})
        sub = surf.get('geology_subdir', 'geology')
        return self.base_config.dirs['data'] / sub

    def get_overlap_region(self) -> List[float]:
        """三模型公共制图范围 [lon_min, lon_max, lat_min, lat_max]"""
        surf = self.config.geology_concordance.get('surface_units') or {}
        region = surf.get('overlap_region')
        if region and len(region) == 4:
            return [float(x) for x in region]

        models = self.config.data.get('target_models') or []
        pattern = self.config.data.get(
            'netcdf_pattern',
            'processed/{model_name}/{model_name}_original.nc',
        )
        models_base = self.base_config.dirs['data'] / self.config.data.get(
            'models_base_dir', 'models'
        )
        lon_mins, lon_maxs, lat_mins, lat_maxs = [], [], [], []
        for model_name in models:
            nc = models_base / pattern.format(model_name=model_name)
            if not nc.exists():
                continue
            with xr.open_dataset(nc) as ds:
                lons = ds['longitude'].values
                lats = ds['latitude'].values
                lon_mins.append(float(lons.min()))
                lon_maxs.append(float(lons.max()))
                lat_mins.append(float(lats.min()))
                lat_maxs.append(float(lats.max()))
        if not lon_mins:
            return [80.0, 150.0, 10.0, 55.0]
        return [max(lon_mins), min(lon_maxs), max(lat_mins), min(lat_maxs)]

    @staticmethod
    def _parse_gmt_segments(
        path: Path,
        as_polygons: bool = False,
    ) -> List[Dict[str, Any]]:
        """解析 GMT 多段线/多边形 → [{coords, name, code}, ...]"""
        if not path.exists():
            return []
        text = path.read_text(encoding='utf-8', errors='replace')
        segments: List[List[str]] = []
        cur: Optional[List[str]] = None
        for line in text.splitlines(keepends=True):
            if line.startswith('>'):
                if cur is not None:
                    segments.append(cur)
                cur = [line]
            elif cur is None:
                continue
            else:
                cur.append(line)
        if cur is not None:
            segments.append(cur)

        out: List[Dict[str, Any]] = []
        for seg in segments:
            body = ''.join(seg)
            name_m = re.search(r'# @D[^|]*\|[^|]*\|[^|]*\|"([^"]*)"', body)
            code_m = re.search(
                r'# @D[^|]*\|[^|]*\|(\d+)\|"', body
            )
            name = name_m.group(1) if name_m else ''
            code = int(code_m.group(1)) if code_m else -1
            coords: List[Tuple[float, float]] = []
            for ln in seg:
                s = ln.strip()
                if not s or s.startswith('#') or s == '>':
                    continue
                toks = s.split()
                if len(toks) < 2:
                    continue
                try:
                    coords.append((float(toks[0]), float(toks[1])))
                except ValueError:
                    continue
            min_pts = 3 if as_polygons else 2
            if len(coords) < min_pts:
                continue
            if as_polygons and coords[0] != coords[-1]:
                coords = coords + [coords[0]]
            out.append({'coords': coords, 'name': name, 'code': code})
        return out

    def _load_overlay_lines(self) -> List[Dict[str, Any]]:
        """加载叠图用地质省线 + CN 地块线"""
        surf = self.config.geology_concordance.get('surface_units') or {}
        geo_dir = self._geology_dir()
        lines: List[Dict[str, Any]] = []
        for fname in surf.get('province_line_files', []):
            path = geo_dir / fname
            segs = self._parse_gmt_segments(path, as_polygons=False)
            # 过滤缓存不存在时回退原始多边形边界
            if not segs:
                alt = geo_dir / fname.replace(
                    '_nosouth_rim_v3_80_150_10_55', ''
                ).replace('_nosouth_north_rim_v2_80_150_10_55', '')
                if 'prv1ec' in fname:
                    segs = self._parse_gmt_segments(
                        geo_dir / 'prv1ec.gmt', as_polygons=True
                    )
                elif 'prv3bl' in fname:
                    segs = self._parse_gmt_segments(
                        geo_dir / 'prv3bl.gmt', as_polygons=True
                    )
                # 多边形当线画
            for s in segs:
                lines.append({**s, 'kind': 'province', 'file': fname})
        for fname in surf.get('block_line_files', []):
            path = geo_dir / fname
            for s in self._parse_gmt_segments(path, as_polygons=False):
                lines.append({**s, 'kind': 'block', 'file': fname})
        return lines

    def _draw_geology_lines_on_ax(
        self,
        ax,
        lines: Optional[List[Dict[str, Any]]] = None,
        region: Optional[Sequence[float]] = None,
        transform=None,
    ) -> None:
        """在 matplotlib 轴上叠绘地质省/地块线"""
        lines = lines if lines is not None else getattr(
            self, '_surface_overlay_lines', None
        )
        if not lines:
            lines = self._load_overlay_lines()
        region = list(region) if region is not None else self.get_overlap_region()
        lon0, lon1, lat0, lat1 = region
        plot_kw = {'transform': transform} if transform is not None else {}
        for seg in lines:
            coords = seg['coords']
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            # 段中心不在范围内则跳过
            if (
                max(xs) < lon0
                or min(xs) > lon1
                or max(ys) < lat0
                or min(ys) > lat1
            ):
                continue
            kind = seg.get('kind', 'province')
            if kind == 'block':
                ax.plot(
                    xs, ys, color='red', linewidth=0.55, alpha=0.95, zorder=5,
                    **plot_kw,
                )
            else:
                ax.plot(
                    xs, ys, color='k', linewidth=0.45, alpha=0.85, zorder=4,
                    **plot_kw,
                )


class EnhancedClusteringVisualizer:
    """增强的聚类可视化器（包含BIC分析、概率分布、GMM vs K-means对比）"""
    
    def __init__(self, config: Any, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.base_config = BaseConfig()
        self.fonts = apply_clustering_plot_style(config)
        self.processor = None
    
    def set_processor(self, processor):
        """设置数据处理器引用"""
        self.processor = processor
    
    def plot_k_selection_panels(
        self,
        per_band: Dict[str, Any],
        output_dir: Path,
        model_name: str,
    ) -> bool:
        """
        K 选择判据图（各深度带合成一张）。

        每格画归一化 BIC(K) 与首末两点连成的弦，红星标在曲线偏离弦最远处，
        竖线标出该偏离量——判据（max-distance-to-chord）由此在图上自证，
        无需正文解释阈值。曲线单调降至 K 上界即说明 BIC 无内部极小、
        不能直接用 argmin 选 K。

        Args:
            per_band: 分层聚类结果的 per_band 字典
            output_dir: 输出目录
            model_name: 模型名（用于标题与日志）

        Returns:
            True 表示已出图；False 表示本模式不适用（调用方应回退逐带绘图）
        """
        bands = [
            (name, info)
            for name, info in per_band.items()
            if isinstance(info.get('bic_analysis'), dict)
            and 'bics' in info['bic_analysis']
            and info['bic_analysis'].get('selection_rule') != 'stability'
        ]
        if not bands:
            return False

        self.logger.info("    📊 绘制 K 选择判据图（合成）...")
        self.fonts = apply_clustering_plot_style(self.config)

        n = len(bands)
        n_cols = 2 if n > 1 else 1
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5.9 * n_cols, 4.1 * n_rows), squeeze=False
        )
        C_BIC, C_CHORD, C_SEL = '#1f6fb4', '#9a9a9a', '#d62728'

        for ax, (name, info) in zip(axes.ravel(), bands):
            ba = info['bic_analysis']
            ks = np.asarray(ba['n_clusters_range'], dtype=int)
            bic = np.asarray(ba['bics'], dtype=float)
            finite = np.isfinite(bic)
            k_sel = int(ba['optimal_n'])
            rule = str(ba.get('selection_rule', ''))

            span = float(np.nanmax(bic[finite]) - np.nanmin(bic[finite]))
            y = ((bic - np.nanmin(bic[finite])) / span if span > 0
                 else np.zeros_like(bic))
            x = np.arange(len(ks), dtype=float)
            xf, yf = x[finite], y[finite]

            # 首末连弦：拐点判据的几何构造
            chord = yf[0] + (yf[-1] - yf[0]) * (xf - xf[0]) / max(
                xf[-1] - xf[0], 1e-12
            )
            ax.plot(xf, chord, ls='--', color=C_CHORD, lw=1.3, zorder=2,
                    label='Chord (endpoints)')
            ax.plot(xf, yf, 'o-', color=C_BIC, lw=2.0, ms=5, zorder=3,
                    label='BIC (normalized)')

            is_fixed = rule == 'fixed_k'
            # 先验固定 K 时，拐点仍绘出但降为诊断标记，星号标在实际采用的 K
            k_knee = int(ba.get('knee_n', k_sel))
            if is_fixed and k_knee in ks:
                jk = int(np.where(ks == k_knee)[0][0])
                if finite[jk]:
                    jkf = int(np.where(xf == x[jk])[0][0])
                    ax.vlines(x[jk], yf[jkf], chord[jkf], color=C_CHORD,
                              lw=1.4, ls=':', zorder=4)
                    ax.plot([x[jk]], [yf[jkf]], 'D', color='none',
                            mec=C_CHORD, mew=1.6, ms=8, zorder=5,
                            label='BIC knee (diagnostic only)')

            j = int(np.where(ks == k_sel)[0][0])
            if finite[j]:
                jf = int(np.where(xf == x[j])[0][0])
                if not is_fixed:
                    ax.vlines(x[j], yf[jf], chord[jf], color=C_SEL, lw=1.6,
                              ls=':', zorder=4)
                ax.plot([x[j]], [yf[jf]], '*', color=C_SEL, ms=19, mec='k',
                        mew=0.6, zorder=6,
                        label=('Prescribed K (prior)' if is_fixed
                               else 'Selected K (max distance to chord)'))

            ax.set_xticks(x)
            ax.set_xticklabels([str(k) for k in ks])
            ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
            ax.set_ylim(-0.08, 1.12)
            ax.set_xlabel('Number of components K', fontsize=10)
            ax.set_ylabel('Normalized BIC', fontsize=10)
            ax.grid(alpha=0.25, ls=':')

            if is_fixed:
                note = f'prescribed; BIC knee = {k_knee}'
            else:
                note = 'BIC knee' if rule == 'bic_knee' else rule
            k_argmin = ba.get('argmin_n')
            note += f'; argmin BIC = {k_argmin}' if k_argmin is not None else ''
            ax.set_title(
                f"{name}  |  {info.get('description', '')}\n"
                f"N = {info.get('n_points', 0):,}   →   K* = {k_sel}  ({note})",
                fontsize=10, fontweight='bold',
            )

        for ax in axes.ravel()[len(bands):]:
            ax.axis('off')

        h, l = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=10,
                   bbox_to_anchor=(0.5, -0.004))
        fig.suptitle(
            f'{model_name} — Selection of K per depth band',
            fontsize=13.5, fontweight='bold', y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.965))

        for fmt in self.config.visualization.get('save_formats', ['jpg']):
            fig.savefig(
                output_dir / f'2-3-1_k_selection.{fmt}',
                dpi=self.config.visualization['dpi'],
            )
        plt.close(fig)
        self.logger.info("      ✅ 保存: 2-3-1_k_selection")
        return True

    def plot_k_robustness(
        self,
        k_robustness: Dict[str, Any],
        output_dir: Path,
        model_name: str,
    ) -> bool:
        """
        K 扫描稳健性图：证明结论不依赖 K 的具体取值。

        左格为相邻 K 的 ARI——K 不同则粒度不同，该值必然小于 1，只用于定位
        分区结构趋于稳定的区间，不应解读为"K 选得对不对"。右格为快/慢三分类
        相对 K* 的一致度，跨模型投票只依赖这个三分类，故它才是结论稳健性的
        直接证据；灰色参考线标在 0.9。

        Args:
            k_robustness: perform_depth_stratified_clustering 返回的 k_robustness
            output_dir: 输出目录
            model_name: 模型名

        Returns:
            True 表示已出图，False 表示无可用数据
        """
        bands = [(n, d) for n, d in (k_robustness or {}).items() if d.get('k_values')]
        if not bands:
            return False

        apply_clustering_plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        palette = plt.get_cmap('tab10')

        for i, (name, info) in enumerate(bands):
            color = palette(i % 10)
            label = name.replace('_', ' ').title()
            k_sel = int(info['k_selected'])

            # 左：相邻 K 的 ARI，横坐标取两个 K 的中点
            pairs = info.get('ari_consecutive', {})
            if pairs:
                xs, ys = [], []
                for key, val in pairs.items():
                    a, b = key.split('->')
                    xs.append((int(a) + int(b)) / 2.0)
                    ys.append(val)
                idx = np.argsort(xs)
                axes[0].plot(
                    np.asarray(xs)[idx], np.asarray(ys)[idx],
                    'o-', color=color, label=label, lw=1.8, ms=4.5,
                )

            # 右：快/慢三分类相对 K* 的一致度
            tri = info.get('trichotomy_agreement', {})
            if tri:
                ks = sorted(int(k) for k in tri)
                vs = [tri[str(k)] for k in ks]
                axes[1].plot(ks, vs, 'o-', color=color, label=label, lw=1.8, ms=4.5)
                axes[1].plot(
                    [k_sel], [tri[str(k_sel)]], marker='*', ms=15,
                    color=color, mec='k', mew=0.7, ls='none', zorder=5,
                )

        axes[0].set_xlabel('K (midpoint of adjacent pair)')
        axes[0].set_ylabel('ARI between adjacent K')
        axes[0].set_title('Partition change with K', fontsize=11.5, fontweight='bold')
        axes[0].set_ylim(0, 1.02)

        axes[1].axhline(0.9, color='0.5', ls='--', lw=1.0, zorder=1)
        axes[1].set_xlabel('Number of clusters K')
        axes[1].set_ylabel('Fast / slow agreement with K*')
        axes[1].set_title(
            'Robustness of the fast–slow classification',
            fontsize=11.5, fontweight='bold',
        )
        axes[1].set_ylim(0, 1.02)

        for ax in axes:
            ax.grid(alpha=0.3, ls=':')

        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc='lower center',
            ncol=min(len(labels), 4), frameon=False, fontsize=10,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.suptitle(
            f'{model_name} — Sensitivity of the results to K '
            '(star = selected K)',
            fontsize=13, fontweight='bold', y=0.99,
        )
        fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))

        for fmt in self.config.visualization.get('save_formats', ['jpg']):
            fig.savefig(
                output_dir / f'2-3-3_k_robustness.{fmt}',
                dpi=self.config.visualization['dpi'],
            )
        plt.close(fig)
        self.logger.info("      ✅ 保存: 2-3-3_k_robustness")
        return True

    def plot_bic_analysis(
        self,
        bic_analysis: Dict[str, Any],
        output_dir: Path,
        model_name: str
    ) -> None:
        """
        BIC 选 K 决策图。

        bic_eff 模式：双轴对比未修正 BIC（灰，单调下降至 K 上界）与有效
        样本量修正 BIC_eff（蓝，中等 K 处出现真实极小值），红星标注选定 K，
        并注明去相关长度与 N_eff——直观说明 K 由数据的独立信息量决定，
        而非由体元数或人为阈值决定。
        其余模式：单轴 BIC(K) + 拐点增益内嵌图。
        """
        self.logger.info("    📊 绘制BIC决策图...")

        if 'bics' not in bic_analysis:
            # fixed_k 模式未做 BIC 扫描，无曲线可画
            self.logger.info(
                f"      ⏭️ [{bic_analysis.get('band_name', '?')}] "
                f"K 为先验固定（{bic_analysis.get('optimal_n')}），跳过 BIC 图"
            )
            return

        n_range = np.asarray(bic_analysis['n_clusters_range'], dtype=int)
        bics = np.asarray(bic_analysis['bics'], dtype=float)
        optimal_n = int(bic_analysis['optimal_n'])
        optimal_bic = float(bic_analysis['optimal_bic'])
        argmin_n = int(bic_analysis.get('argmin_n', optimal_n))
        rule = bic_analysis.get('selection_rule', 'bic_min')

        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.plot(
            n_range, bics, 'o-', color='steelblue',
            linewidth=2, markersize=8, label='BIC',
        )
        sel_label = (
            f'Selected K = {optimal_n} (BIC knee)'
            if rule == 'bic_knee' else f'Selected K = {optimal_n} (min BIC)'
        )
        ax.axvline(
            optimal_n, color='red', linestyle='--', linewidth=1.5,
            label=sel_label,
        )
        ax.scatter(
            [optimal_n], [optimal_bic], s=220, c='red', marker='*',
            zorder=10, edgecolors='black', linewidth=1.5,
        )
        if argmin_n != optimal_n:
            ax.scatter(
                [argmin_n], [float(bic_analysis.get('argmin_bic', np.nan))],
                s=140, c='gray', marker='*', zorder=9,
                edgecolors='black', linewidth=1.0,
                label=f'min BIC at K = {argmin_n}',
            )
        ax.set_xlabel('Number of Clusters K')
        ax.set_ylabel('BIC (lower is better)')
        ax.set_xticks(n_range)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f'{model_name} — BIC Model Selection',
            fontweight='bold',
        )

        # 内嵌图：各 K 到首末弦的归一化距离（拐点判据本身的可视化）
        dist = np.asarray(
            bic_analysis.get('knee_distances', [np.nan] * len(bics)), dtype=float
        )
        finite = np.isfinite(dist)
        if np.sum(finite) > 0:
            ax_in = ax.inset_axes([0.42, 0.45, 0.53, 0.4])
            ax_in.plot(
                n_range[finite], dist[finite], 'o-',
                color='gray', linewidth=1.2, markersize=4,
            )
            ax_in.axvline(optimal_n, color='red', linestyle='--', linewidth=1)
            ax_in.set_ylabel('Distance to chord', fontsize=11)
            ax_in.tick_params(labelsize=10)
            ax_in.grid(True, alpha=0.3)

        band = bic_analysis.get('band_name', 'full')
        for fmt in self.config.visualization.get('save_formats', ['jpg']):
            filepath = output_dir / f'2-3-1_bic_analysis_{band}.{fmt}'
            fig.savefig(
                filepath, dpi=self.config.visualization['dpi'], bbox_inches='tight'
            )
        plt.close(fig)
        self.logger.info(f"      ✅ 保存: 2-3-1_bic_analysis_{band}")
    
    def plot_probability_distribution(
        self, 
        probabilities: np.ndarray,
        labels: np.ndarray,
        output_dir: Path,
        model_name: str
    ) -> None:
        """绘制概率分布和置信度分析"""
        self.logger.info("    📊 绘制概率分布分析...")
        
        n_clusters = probabilities.shape[1]
        max_probs = np.max(probabilities, axis=1)
        
        # 计算熵
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        
        # 创建图形
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 最大概率分布直方图
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(max_probs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(max_probs), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean={np.mean(max_probs):.3f}')
        ax1.axvline(np.median(max_probs), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median={np.median(max_probs):.3f}')
        threshold = self.config.auto_gmm['probability_threshold']
        ax1.axvline(threshold, color='green', linestyle='--', 
                   linewidth=2, label=f'Threshold={threshold}')
        ax1.set_xlabel('Maximum Probability', fontsize=16)
        ax1.set_ylabel('Frequency', fontsize=16)
        ax1.set_title('Distribution of Maximum Probabilities', fontsize=17, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 置信度分类
        ax2 = fig.add_subplot(gs[0, 1])
        high_conf = np.sum(max_probs > threshold)
        medium_conf = np.sum((max_probs > 0.5) & (max_probs <= threshold))
        low_conf = np.sum(max_probs <= 0.5)
        
        categories = ['High\n(>{})'.format(threshold), 
                     'Medium\n(0.5-{})'.format(threshold), 
                     'Low\n(<0.5)']
        counts = [high_conf, medium_conf, low_conf]
        colors_conf = ['#2ecc71', '#f39c12', '#e74c3c']
        
        bars = ax2.bar(categories, counts, color=colors_conf, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Number of Samples', fontsize=16)
        ax2.set_title('Confidence Level Distribution', fontsize=17, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加百分比标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}\n({count/len(max_probs):.1%})',
                    ha='center', va='bottom', fontsize=15, fontweight='bold')
        
        # 3. 熵分布
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(entropy, bins=50, color='coral', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(entropy), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean={np.mean(entropy):.3f}')
        ax3.set_xlabel('Entropy (Uncertainty)', fontsize=16)
        ax3.set_ylabel('Frequency', fontsize=16)
        ax3.set_title('Clustering Uncertainty Distribution', fontsize=17, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 每个簇的平均概率
        ax4 = fig.add_subplot(gs[1, :])
        cluster_avg_probs = []
        cluster_ids = []
        
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if np.any(mask):
                cluster_probs = probabilities[mask, cluster_id]
                cluster_avg_probs.append(cluster_probs)
                cluster_ids.append(cluster_id)
        
        # 箱线图
        colors = generate_distinct_colors(len(cluster_ids))
        bp = ax4.boxplot(cluster_avg_probs, positions=cluster_ids, widths=0.6,
                        patch_artist=True, showmeans=True, meanline=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_xlabel('Cluster ID', fontsize=16)
        ax4.set_ylabel('Assignment Probability', fontsize=16)
        ax4.set_title('Probability Distribution by Cluster', fontsize=17, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim([0, 1.05])
        
        # 5. 概率矩阵热图
        ax5 = fig.add_subplot(gs[2, :2])
        
        # 采样（如果样本太多）
        n_samples_to_show = min(1000, len(probabilities))
        sample_indices = np.random.choice(len(probabilities), n_samples_to_show, replace=False)
        sample_indices = np.sort(sample_indices)
        
        # 按照最大概率排序
        sorted_indices = sample_indices[np.argsort(np.argmax(probabilities[sample_indices], axis=1))]
        
        im = ax5.imshow(probabilities[sorted_indices].T, aspect='auto', 
                       cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
        ax5.set_xlabel('Sample Index (sorted by cluster)', fontsize=16)
        ax5.set_ylabel('Cluster ID', fontsize=16)
        ax5.set_title(f'Probability Matrix (showing {n_samples_to_show} samples)', 
                     fontsize=17, fontweight='bold')
        ax5.set_yticks(range(n_clusters))
        plt.colorbar(im, ax=ax5, label='Probability')
        
        # 6. 统计表格
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        stats_data = [
            ['Total Samples', f'{len(probabilities):,}'],
            ['Number of Clusters', f'{n_clusters}'],
            ['', ''],
            ['Mean Max Prob', f'{np.mean(max_probs):.4f}'],
            ['Median Max Prob', f'{np.median(max_probs):.4f}'],
            ['Std Max Prob', f'{np.std(max_probs):.4f}'],
            ['', ''],
            ['High Confidence', f'{high_conf:,} ({high_conf/len(max_probs):.1%})'],
            ['Medium Confidence', f'{medium_conf:,} ({medium_conf/len(max_probs):.1%})'],
            ['Low Confidence', f'{low_conf:,} ({low_conf/len(max_probs):.1%})'],
            ['', ''],
            ['Mean Entropy', f'{np.mean(entropy):.4f}'],
            ['Median Entropy', f'{np.median(entropy):.4f}']
        ]
        
        table = ax6.table(cellText=stats_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in [0, 1]:
            table[(i, 0)].set_facecolor('#3498db')
            table[(i, 1)].set_facecolor('#3498db')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        
        plt.suptitle(f'{model_name} - GMM Probability & Confidence Analysis', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        filepath = output_dir / '2-3-2_probability_analysis.png'
        plt.savefig(filepath, dpi=self.config.visualization['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"      ✅ 保存: 2-3-2_probability_analysis.png")
    
    def plot_gmm_vs_kmeans_comparison(
        self,
        gmm_results: Dict[str, Any],
        kmeans_results: Dict[str, Any],
        output_dir: Path,
        model_name: str
    ) -> None:
        """绘制GMM vs K-means对比分析"""
        self.logger.info("    📊 绘制GMM vs K-means对比...")
        
        # 创建图形
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 轮廓系数对比
        ax1 = fig.add_subplot(gs[0, 0])
        
        kmeans_n_list = kmeans_results['n_clusters_tested']
        kmeans_sil_scores = [kmeans_results['metrics'][k]['silhouette_score'] 
                            for k in kmeans_n_list]
        
        ax1.plot(kmeans_n_list, kmeans_sil_scores, 'o-', 
                color='steelblue', linewidth=2, markersize=8, label='K-means')
        
        # GMM最优点
        gmm_n = gmm_results['n_clusters']
        gmm_sil = gmm_results['metrics']['silhouette_score']
        ax1.scatter([gmm_n], [gmm_sil], s=300, c='red', marker='*', 
                   zorder=10, edgecolors='black', linewidth=2, label='GMM (Optimal)')
        
        ax1.set_xlabel('Number of Clusters', fontsize=16)
        ax1.set_ylabel('Silhouette Score', fontsize=16)
        ax1.set_title('Silhouette Score Comparison', fontsize=17, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # 2. Davies-Bouldin对比
        ax2 = fig.add_subplot(gs[0, 1])
        
        kmeans_db_scores = [kmeans_results['metrics'][k]['davies_bouldin_score'] 
                           for k in kmeans_n_list]
        
        ax2.plot(kmeans_n_list, kmeans_db_scores, 'o-', 
                color='coral', linewidth=2, markersize=8, label='K-means')
        
        gmm_db = gmm_results['metrics']['davies_bouldin_score']
        ax2.scatter([gmm_n], [gmm_db], s=300, c='red', marker='*', 
                   zorder=10, edgecolors='black', linewidth=2, label='GMM (Optimal)')
        
        ax2.set_xlabel('Number of Clusters', fontsize=16)
        ax2.set_ylabel('Davies-Bouldin Score', fontsize=16)
        ax2.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=17, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Calinski-Harabasz对比
        ax3 = fig.add_subplot(gs[0, 2])
        
        kmeans_ch_scores = [kmeans_results['metrics'][k]['calinski_harabasz_score'] 
                           for k in kmeans_n_list]
        
        ax3.plot(kmeans_n_list, kmeans_ch_scores, 'o-', 
                color='forestgreen', linewidth=2, markersize=8, label='K-means')
        
        gmm_ch = gmm_results['metrics']['calinski_harabasz_score']
        ax3.scatter([gmm_n], [gmm_ch], s=300, c='red', marker='*', 
                   zorder=10, edgecolors='black', linewidth=2, label='GMM (Optimal)')
        
        ax3.set_xlabel('Number of Clusters', fontsize=16)
        ax3.set_ylabel('Calinski-Harabasz Score', fontsize=16)
        ax3.set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=17, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 计算时间对比
        ax4 = fig.add_subplot(gs[1, 0])
        
        kmeans_times = [kmeans_results['metrics'][k]['fit_time'] for k in kmeans_n_list]
        
        ax4.plot(kmeans_n_list, kmeans_times, 'o-', 
                color='purple', linewidth=2, markersize=8, label='K-means')
        
        gmm_time = gmm_results['metrics']['fit_time']
        ax4.scatter([gmm_n], [gmm_time], s=300, c='red', marker='*', 
                   zorder=10, edgecolors='black', linewidth=2, label='GMM')
        
        ax4.set_xlabel('Number of Clusters', fontsize=16)
        ax4.set_ylabel('Fit Time (seconds)', fontsize=16)
        ax4.set_title('Computational Time Comparison', fontsize=17, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 聚类平衡性对比
        ax5 = fig.add_subplot(gs[1, 1])
        
        kmeans_balance = [kmeans_results['metrics'][k]['cluster_balance'] 
                         for k in kmeans_n_list]
        
        ax5.plot(kmeans_n_list, kmeans_balance, 'o-', 
                color='brown', linewidth=2, markersize=8, label='K-means')
        
        gmm_balance = gmm_results['metrics']['cluster_balance']
        ax5.scatter([gmm_n], [gmm_balance], s=300, c='red', marker='*', 
                   zorder=10, edgecolors='black', linewidth=2, label='GMM')
        
        ax5.set_xlabel('Number of Clusters', fontsize=16)
        ax5.set_ylabel('Cluster Balance (std/mean)', fontsize=16)
        ax5.set_title('Cluster Balance Comparison (Lower is Better)', 
                     fontsize=17, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 标签一致性分析（如果有）
        ax6 = fig.add_subplot(gs[1, 2])
        
        if 'comparison_with_gmm' in kmeans_results and kmeans_results['comparison_with_gmm']:
            comp = kmeans_results['comparison_with_gmm']
            
            metrics_names = ['ARI', 'NMI', 'Agreement\nRatio']
            metrics_values = [
                comp['adjusted_rand_index'],
                comp['normalized_mutual_info'],
                comp['agreement_ratio']
            ]
            
            bars = ax6.bar(metrics_names, metrics_values, 
                          color=['#3498db', '#e74c3c', '#2ecc71'], 
                          alpha=0.7, edgecolor='black', linewidth=2)
            
            ax6.set_ylabel('Score', fontsize=16)
            ax6.set_title(f'Label Agreement (n={gmm_n})', fontsize=17, fontweight='bold')
            ax6.set_ylim([0, 1])
            ax6.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, metrics_values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', 
                        fontsize=15, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                    fontsize=24, transform=ax6.transAxes)
            ax6.set_title('Label Agreement', fontsize=17, fontweight='bold')
        
        # 7. 对比表格
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # 找出K-means最优结果
        best_k = kmeans_results['best_k']
        kmeans_best = kmeans_results['metrics'][best_k]
        
        table_data = [
            ['Metric', 'GMM (BIC-Optimal)', f'K-means (Best k={best_k})', 'Winner'],
            ['Number of Clusters', f'{gmm_n}', f'{best_k}', '-'],
            ['Silhouette Score ↑', f"{gmm_sil:.4f}", f"{kmeans_best['silhouette_score']:.4f}", 
             'GMM' if gmm_sil > kmeans_best['silhouette_score'] else 'K-means'],
            ['Davies-Bouldin ↓', f"{gmm_db:.4f}", f"{kmeans_best['davies_bouldin_score']:.4f}",
             'GMM' if gmm_db < kmeans_best['davies_bouldin_score'] else 'K-means'],
            ['Calinski-Harabasz ↑', f"{gmm_ch:.2f}", f"{kmeans_best['calinski_harabasz_score']:.2f}",
             'GMM' if gmm_ch > kmeans_best['calinski_harabasz_score'] else 'K-means'],
            ['Fit Time (s)', f"{gmm_time:.2f}", f"{kmeans_best['fit_time']:.2f}",
             'GMM' if gmm_time < kmeans_best['fit_time'] else 'K-means'],
            ['Cluster Balance ↓', f"{gmm_balance:.4f}", f"{kmeans_best['cluster_balance']:.4f}",
             'GMM' if gmm_balance < kmeans_best['cluster_balance'] else 'K-means'],
            ['Probability Support', 'Yes ✓', 'No ✗', 'GMM'],
            ['Soft Clustering', 'Yes ✓', 'No ✗', 'GMM']
        ]
        
        table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # 设置表头样式
        for j in range(4):
            table[(0, j)].set_facecolor('#3498db')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        # 高亮Winner列
        for i in range(1, len(table_data)):
            winner = table_data[i][3]
            if winner == 'GMM':
                table[(i, 3)].set_facecolor('#2ecc71')
                table[(i, 3)].set_text_props(weight='bold')
            elif winner == 'K-means':
                table[(i, 3)].set_facecolor('#f39c12')
                table[(i, 3)].set_text_props(weight='bold')
        
        plt.suptitle(f'{model_name} - GMM vs K-means Comprehensive Comparison', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        filepath = output_dir / '2-3-3_gmm_vs_kmeans_comparison.png'
        plt.savefig(filepath, dpi=self.config.visualization['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"      ✅ 保存: 2-3-3_gmm_vs_kmeans_comparison.png")
        
class BasicClusteringVisualizer:
    """基础聚类可视化器（深度切片、垂直剖面等）- 修正版（含海岸线）"""
    
    def __init__(self, config: Any, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.base_config = BaseConfig()
        self.fonts = apply_clustering_plot_style(config)
        self.processor = None
    
    def set_processor(self, processor):
        """设置数据处理器引用"""
        self.processor = processor

    def _ensure_geology_basemap_png(
        self,
        map_extent: Sequence[float],
        cache_dir: Path,
        dpi: int = 200,
    ) -> Path:
        """
        按模型制图范围缓存 USGS 拼合基岩地质底图（仿 5_9，无图例）。

        Args:
            map_extent: [lon_min, lon_max, lat_min, lat_max]
            cache_dir: 缓存目录
            dpi: PNG 分辨率

        Returns:
            PNG 路径
        """
        lon0, lon1, lat0, lat1 = [float(x) for x in map_extent]
        tag = (
            f'{lon0:.1f}_{lon1:.1f}_{lat0:.1f}_{lat1:.1f}'.replace('.', 'p')
        )
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # 文件名带 bathy 标记，避免沿用旧的纯色海水缓存
        # _bathy_otop_：地质后仅洋区重绘水深，避免年代面挡住陆架/洋区
        out_png = cache_dir / f'geology_basemap_{tag}_bathy_otop_dpi{dpi}.png'
        if out_png.exists() and out_png.stat().st_size > 1000:
            return out_png

        import importlib.util

        geology_py = project_root / '5_Visualization' / '5_9_Geology.py'
        if not geology_py.exists():
            raise FileNotFoundError(f'找不到地质图模块: {geology_py}')
        mod_name = 'eastasia_geology_5_9'
        # 始终重载，确保 ocean_bathymetry 等改动生效
        spec = importlib.util.spec_from_file_location(mod_name, geology_py)
        if spec is None or spec.loader is None:
            raise ImportError(f'无法加载地质图模块: {geology_py}')
        geology_mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = geology_mod
        spec.loader.exec_module(geology_mod)
        EastAsiaGeologyMap = geology_mod.EastAsiaGeologyMap

        plotter = EastAsiaGeologyMap(output_dir=str(cache_dir))
        plotter.config.layers['cn_blocks'] = True
        plotter.config.layers['province_overlay'] = True
        plotter.config.layers['ocean_bathymetry'] = True
        plotter.config.layers['legend'] = False
        plotter.config.layers['title'] = False
        plotter.render_composite_png_for_embed(
            region=[lon0, lon1, lat0, lat1],
            out_path=out_png,
            dpi=dpi,
            force_convert=False,
        )
        return out_png

    def plot_depth_slices(
        self,
        labels_3d: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Path,
        algorithm_name: str,
        model_name: str,
        concordance_eval: Optional['GeologyConcordanceEvaluator'] = None,
    ) -> None:
        """
        3×3 深度切片（子图等大）：
        - 第1排：首格 USGS 基岩地质底图（无图例）；其后浅部 facies+地质线（默认 10/40 km）
        - 第2–3排：叠 Slab2 等深线（默认 100/150/200/500/600/700 km）
        子图标题只保留深度与叠图类型（Slab2/地质线均为定性对照）。
        """
        self.logger.info(f"    📊 绘制 3×3 深度切片 ({algorithm_name})...")
        self.fonts = apply_clustering_plot_style(self.config)

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
            use_cartopy = True
        except ImportError:
            self.logger.warning("    ⚠️ cartopy未安装，使用基础版本（无海岸线）")
            use_cartopy = False
            ccrs = cfeature = LONGITUDE_FORMATTER = LATITUDE_FORMATTER = None

        layout = self.config.visualization.get('slice_layout') or {}
        n_rows = int(layout.get('n_rows', 3))
        n_cols = int(layout.get('n_cols', 3))
        lead_geo_map = bool(layout.get('row1_leading_geology_map', True))
        geo_depths = list(
            layout.get('geology_depths_km')
            or (self.config.geology_concordance.get('surface_units') or {}).get(
                'overlay_depths_km', [10.0, 40.0]
            )
        )
        slab_depths = list(
            layout.get('slab_depths_km', [100.0, 150.0, 200.0, 500.0, 600.0, 700.0])
        )
        # 第1排：可选首格地质底图 + 浅部 facies 槽位
        n_facies_geo = n_cols - 1 if lead_geo_map else n_cols
        geo_depths = (geo_depths + [None] * n_facies_geo)[:n_facies_geo]
        n_slab_slots = n_rows * n_cols - n_cols
        slab_depths = (slab_depths + [None] * n_slab_slots)[:n_slab_slots]

        lats = np.asarray(metadata['lats'], dtype=float)
        lons = np.asarray(metadata['lons'], dtype=float)
        depths = np.asarray(metadata['depths'], dtype=float)
        map_extent = [
            float(lons.min()),
            float(lons.max()),
            float(lats.min()),
            float(lats.max()),
        ]
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        n_clusters = int(len(np.unique(labels_3d[labels_3d >= 0])))
        colors = generate_distinct_colors(
            max(n_clusters, self.config.visualization['n_colors'])
        )
        cmap = mcolors.ListedColormap(colors[: max(n_clusters, 1)])

        # Slab2 / 地质线
        slab_grid = None
        geo_region = None
        if concordance_eval is not None:
            try:
                concordance_eval._ensure_slab_grid(lons, lats)
                slab_grid = concordance_eval._slab_grid
            except Exception as e:
                self.logger.warning(f"      ⚠️ Slab2 加载失败: {e}")
            try:
                if not getattr(concordance_eval, '_surface_overlay_lines', None):
                    concordance_eval._surface_overlay_lines = (
                        concordance_eval._load_overlay_lines()
                    )
                geo_region = concordance_eval.get_overlap_region()
            except Exception as e:
                self.logger.warning(f"      ⚠️ 地质线加载失败: {e}")

        slab_levels = [50, 100, 150, 200, 300, 400, 500, 600]
        # 右侧留出色标空间（仅最右列挂簇色标）
        fig, axes = make_equal_map_subplots(
            n_rows,
            n_cols,
            ccrs if use_cartopy else None,
            panel_w=6.4,
            panel_h=5.2,
            wspace=0.12,
            hspace=0.18,
            top=0.90,
            bottom=0.04,
            left=0.035,
            right=0.90,
        )
        projection = ccrs.PlateCarree() if use_cartopy else None
        plot_tf = {'transform': projection} if use_cartopy else {}

        def _style_ax(ax) -> None:
            if use_cartopy:
                ax.coastlines(
                    resolution='50m', color='gray', linewidth=0.45, zorder=6
                )
                ax.add_feature(
                    cfeature.BORDERS,
                    linestyle='--',
                    linewidth=0.4,
                    edgecolor='gray',
                    zorder=5,
                )
                ax.set_extent(map_extent, crs=projection)
                try:
                    ax.set_aspect('auto')
                except Exception:
                    pass
                gl = ax.gridlines(
                    crs=projection,
                    draw_labels=True,
                    linewidth=0.5,
                    color='gray',
                    alpha=0.5,
                    linestyle='--',
                    zorder=4,
                )
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 13}
                gl.ylabel_style = {'size': 13}
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
            else:
                ax.set_xlim(map_extent[0], map_extent[1])
                ax.set_ylim(map_extent[2], map_extent[3])
                ax.set_aspect('auto')
                F = self.fonts
                ax.set_xlabel('Longitude', fontsize=F['label'])
                ax.set_ylabel('Latitude', fontsize=F['label'])
                ax.tick_params(labelsize=F['tick'])
                ax.grid(True, alpha=0.3, linestyle='--')

        def _draw_facies(ax, z_target: float):
            z_idx = int(np.argmin(np.abs(depths - float(z_target))))
            actual_z = float(depths[z_idx])
            label_slice = np.ma.masked_where(
                labels_3d[:, :, z_idx] < 0, labels_3d[:, :, z_idx]
            )
            im = ax.imshow(
                label_slice,
                aspect='auto',
                origin='lower',
                cmap=cmap,
                interpolation='nearest',
                extent=map_extent,
                vmin=-0.5,
                vmax=max(n_clusters - 0.5, 0.5),
                zorder=1,
                **plot_tf,
            )
            return actual_z, im

        def _add_cluster_cbar(ax, im) -> None:
            """最右列子图挂簇色标。"""
            if n_clusters <= 0:
                return
            # 簇多时抽稀刻度，避免文字挤叠
            if n_clusters <= 12:
                ticks = list(range(n_clusters))
            elif n_clusters <= 24:
                ticks = list(range(0, n_clusters, 2))
                if ticks[-1] != n_clusters - 1:
                    ticks.append(n_clusters - 1)
            else:
                ticks = list(range(0, n_clusters, 3))
                if ticks[-1] != n_clusters - 1:
                    ticks.append(n_clusters - 1)
            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.02,
                ticks=ticks,
            )
            cbar.ax.tick_params(labelsize=max(10, int(self.fonts['annotation']) - 2))
            cbar.set_label('Cluster ID', fontsize=self.fonts['label'])

        # —— 第1排：首格基岩地质底图 + 浅部 facies 叠地质线 ——
        facies_start_col = 0
        if lead_geo_map:
            ax0 = axes[0, 0]
            try:
                # output_dir 是叶子图件目录（figures/model_clustering/<scheme>/<model>），
                # 其父目录即 scheme 根，各模型共享同一份地质底图缓存避免重复渲染
                out_res = Path(output_dir).resolve()
                cache_dir = out_res.parent / '_geology_basemap_cache'
                geo_png = self._ensure_geology_basemap_png(
                    map_extent, cache_dir=cache_dir, dpi=220
                )
                img = plt.imread(str(geo_png))
                ax0.imshow(
                    img,
                    extent=map_extent,
                    origin='upper',
                    aspect='auto',
                    zorder=1,
                    **plot_tf,
                )
            except Exception as e:
                self.logger.warning(f'      ⚠️ 地质底图渲染失败: {e}')
                ax0.text(
                    0.5,
                    0.5,
                    'Bedrock geology unavailable',
                    transform=ax0.transAxes,
                    ha='center',
                    va='center',
                    fontsize=self.fonts['annotation'],
                )
            _style_ax(ax0)
            ax0.set_title(
                'Bedrock Geology',
                fontsize=self.fonts['title'],
                fontweight='bold',
                pad=6,
            )
            facies_start_col = 1

        for i, z_target in enumerate(geo_depths):
            col = facies_start_col + i
            if col >= n_cols:
                break
            ax = axes[0, col]
            if z_target is None:
                ax.axis('off')
                continue
            actual_z, im = _draw_facies(ax, float(z_target))
            if concordance_eval is not None and geo_region is not None:
                concordance_eval._draw_geology_lines_on_ax(
                    ax, region=geo_region, transform=projection
                )
            _style_ax(ax)
            ax.set_title(
                f'{actual_z:.0f} km · Geology',
                fontsize=self.fonts['title'],
                fontweight='bold',
                pad=6,
            )
            if col == n_cols - 1:
                _add_cluster_cbar(ax, im)

        # —— 第2–3排：Slab2 ——
        for i, z_target in enumerate(slab_depths):
            row = 1 + i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            if z_target is None:
                ax.axis('off')
                continue
            actual_z, im = _draw_facies(ax, float(z_target))
            if slab_grid is not None and np.any(np.isfinite(slab_grid)):
                levels = [
                    lv for lv in slab_levels
                    if np.nanmin(slab_grid) <= lv <= np.nanmax(slab_grid)
                ]
                if levels:
                    cs = ax.contour(
                        lon_grid,
                        lat_grid,
                        slab_grid,
                        levels=levels,
                        colors='k',
                        linewidths=0.75,
                        alpha=0.9,
                        zorder=4,
                        **plot_tf,
                    )
                    ax.clabel(cs, inline=True, fontsize=self.fonts['clabel'], fmt='%.0f')
            _style_ax(ax)
            ax.set_title(
                f'{actual_z:.0f} km · Slab2',
                fontsize=self.fonts['title'],
                fontweight='bold',
                pad=6,
            )
            if col == n_cols - 1:
                _add_cluster_cbar(ax, im)

        fig.suptitle(
            f'{model_name} — {algorithm_name.upper()} Depth Slices '
            f'(row1 bedrock+facies; row2–3 Slab2)',
            fontsize=self.fonts['suptitle'],
            fontweight='bold',
            y=0.975,
        )
        # 略增行距；右侧已预留色标空间，勿再把 right 拉满
        fig.subplots_adjust(top=0.90, hspace=0.22, wspace=0.14, right=0.90)

        out_stem = f'2-3-4_{algorithm_name}_depth_slices'
        save_fmts = set(self.config.visualization.get('save_formats', ['jpg']))
        save_fmts.add('png')  # 兼容旧深度切片命名
        for fmt in sorted(save_fmts):
            fig.savefig(
                output_dir / f'{out_stem}.{fmt}',
                dpi=self.config.visualization['dpi'],
            )
        plt.close(fig)
        self.logger.info(f"      ✅ 保存: {out_stem} ({', '.join(sorted(save_fmts))})")

    def plot_vertical_sections(
        self,
        labels_3d: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Path,
        algorithm_name: str,
        model_name: str,
        concordance_eval: Optional['GeologyConcordanceEvaluator'] = None,
        pert_cube: Optional[np.ndarray] = None,
    ) -> None:
        """
        绘制垂直剖面（带地理位置标注）。

        行布局：
          0 — 剖面位置地图
          1 — 聚类 facies + Slab2
          2–3 — δlnVp / δlnVs 原始扰动 + Slab2（若提供 pert_cube）
        """
        self.logger.info(f"    📊 绘制垂直剖面 ({algorithm_name})...")
        self.fonts = apply_clustering_plot_style(self.config)

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            use_cartopy = True
        except ImportError:
            self.logger.warning("    ⚠️ cartopy未安装，使用基础版本")
            use_cartopy = False
            ccrs = cfeature = None

        section_lats = self.config.visualization['section_positions']['latitudes']
        lats = np.asarray(metadata['lats'], dtype=float)
        lons = np.asarray(metadata['lons'], dtype=float)
        depths = np.asarray(metadata['depths'], dtype=float)

        n_sections = min(len(section_lats), 3)
        selected_lats = section_lats[:n_sections]

        n_clusters = len(np.unique(labels_3d[labels_3d >= 0]))
        colors = generate_distinct_colors(
            max(n_clusters, self.config.visualization['n_colors'])
        )
        cmap = mcolors.ListedColormap(colors[:n_clusters])

        # Slab2 三维体掩膜（与聚类同网格）
        slab_mask = None
        if concordance_eval is not None:
            try:
                slab_mask = concordance_eval.build_slab_volume_mask(
                    lons, lats, depths
                )
            except Exception as e:
                self.logger.warning(f"      ⚠️ 剖面 Slab2 体掩膜失败: {e}")

        if pert_cube is None and self.processor is not None:
            pert_cube = getattr(self.processor, 'last_feature_cube', None)
        has_pert = (
            pert_cube is not None
            and np.ndim(pert_cube) == 4
            and pert_cube.shape[-1] >= 2
            and pert_cube.shape[:3] == labels_3d.shape
        )
        n_rows = 4 if has_pert else 2
        right_margin = 0.90 if has_pert else 0.98

        panel_w, panel_h = 6.0, 4.2
        fig = plt.figure(figsize=(panel_w * n_sections, panel_h * n_rows))
        gs = fig.add_gridspec(
            n_rows,
            n_sections,
            wspace=0.28,
            hspace=0.38,
            left=0.06,
            right=right_margin,
            top=0.93,
            bottom=0.05,
            width_ratios=[1] * n_sections,
            height_ratios=[1.0] * n_rows,
        )
        projection = ccrs.PlateCarree() if use_cartopy else None
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        slab_footprint = (
            np.any(slab_mask, axis=2) if slab_mask is not None else None
        )
        sec_extent = [
            float(lons.min()),
            float(lons.max()),
            float(depths.max()),
            float(depths.min()),
        ]

        # 第一行：剖面位置地图
        for idx in range(n_sections):
            if use_cartopy:
                ax_map = fig.add_subplot(gs[0, idx], projection=projection)
            else:
                ax_map = fig.add_subplot(gs[0, idx])

            target_lat = selected_lats[idx]
            lat_idx = int(np.argmin(np.abs(lats - target_lat)))
            actual_lat = float(lats[lat_idx])

            depth_idx_bg = int(np.argmin(np.abs(depths - 200)))
            slice_bg = labels_3d[:, :, depth_idx_bg]
            slice_masked_bg = np.ma.masked_where(slice_bg < 0, slice_bg)
            extent_map = [
                float(lons.min()),
                float(lons.max()),
                float(lats.min()),
                float(lats.max()),
            ]
            imshow_kw: Dict[str, Any] = {}
            if use_cartopy:
                imshow_kw['transform'] = projection
            ax_map.imshow(
                slice_masked_bg,
                aspect='auto',
                origin='lower',
                cmap=cmap,
                interpolation='nearest',
                extent=extent_map,
                alpha=0.5,
                vmin=0,
                vmax=max(n_clusters - 1, 0),
                **imshow_kw,
            )

            if slab_footprint is not None and np.any(slab_footprint):
                ax_map.contour(
                    lon_grid,
                    lat_grid,
                    slab_footprint.astype(float),
                    levels=[0.5],
                    colors='k',
                    linewidths=0.55,
                    zorder=3,
                    **imshow_kw,
                )

            if use_cartopy:
                ax_map.coastlines(
                    resolution='50m', color='gray', linewidth=0.45, zorder=3
                )
                ax_map.add_feature(
                    cfeature.BORDERS,
                    linestyle='--',
                    linewidth=0.4,
                    edgecolor='gray',
                    zorder=3,
                )
                ax_map.plot(
                    [lons.min(), lons.max()],
                    [actual_lat, actual_lat],
                    'r-',
                    linewidth=3,
                    transform=projection,
                    zorder=4,
                    label='Section Line',
                )
                ax_map.scatter(
                    [lons.min(), lons.max()],
                    [actual_lat, actual_lat],
                    c='red',
                    s=100,
                    marker='o',
                    edgecolors='black',
                    linewidth=2,
                    transform=projection,
                    zorder=5,
                )
                ax_map.set_extent(extent_map, crs=projection)
                try:
                    ax_map.set_aspect('auto')
                except Exception:
                    pass
                gl = ax_map.gridlines(
                    draw_labels=True,
                    linewidth=0.5,
                    color='gray',
                    alpha=0.5,
                    linestyle='--',
                )
                gl.top_labels = False
                gl.right_labels = False
            else:
                ax_map.axhline(actual_lat, color='r', linewidth=2)
                ax_map.set_xlim(extent_map[0], extent_map[1])
                ax_map.set_ylim(extent_map[2], extent_map[3])
                ax_map.set_aspect('auto')

            ax_map.set_title(
                f'Section Location: Lat={actual_lat:.1f}°',
                fontsize=self.fonts['title'],
                fontweight='bold',
            )
            if use_cartopy:
                ax_map.legend(loc='upper right', fontsize=self.fonts['legend'])
            ax_map.tick_params(labelsize=self.fonts['tick'])

        # 第二行：聚类 facies + Slab2
        for idx, target_lat in enumerate(selected_lats):
            ax = fig.add_subplot(gs[1, idx])
            lat_idx = int(np.argmin(np.abs(lats - target_lat)))
            actual_lat = float(lats[lat_idx])
            section_data = labels_3d[lat_idx, :, :]
            section_masked = np.ma.masked_where(section_data < 0, section_data)
            ax.imshow(
                section_masked.T,
                aspect='auto',
                origin='upper',
                cmap=cmap,
                interpolation='nearest',
                extent=sec_extent,
                vmin=0,
                vmax=max(n_clusters - 1, 0),
                zorder=1,
            )
            if slab_mask is not None:
                overlay_slab_on_section(
                    ax,
                    slab_mask[lat_idx, :, :],
                    lons,
                    depths,
                    with_legend=(idx == 0),
                    legend_fontsize=self.fonts['legend'],
                )
            ax.set_title(
                f'Facies  |  Lat={actual_lat:.1f}°',
                fontsize=self.fonts['title'],
                fontweight='bold',
            )
            ax.set_xlabel('Longitude (°)', fontsize=self.fonts['label'])
            ax.set_ylabel('Depth (km)', fontsize=self.fonts['label'])
            ax.tick_params(labelsize=self.fonts['tick'])
            ax.set_ylim(float(depths.max()), float(depths.min()))
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # 第三/四行：δlnVp / δlnVs + Slab2
        if has_pert:
            feat_titles = [
                r'$\delta\ln V_p$',
                r'$\delta\ln V_s$',
            ]
            for r_off, (fi, ftitle) in enumerate(zip((0, 1), feat_titles)):
                row = 2 + r_off
                vmin, vmax = perturbation_clim(pert_cube, fi)
                last_im = None
                for idx, target_lat in enumerate(selected_lats):
                    ax = fig.add_subplot(gs[row, idx])
                    lat_idx = int(np.argmin(np.abs(lats - target_lat)))
                    actual_lat = float(lats[lat_idx])
                    sec = np.ma.masked_invalid(pert_cube[lat_idx, :, :, fi])
                    last_im = ax.imshow(
                        sec.T,
                        aspect='auto',
                        origin='upper',
                        cmap='RdBu_r',
                        interpolation='nearest',
                        extent=sec_extent,
                        vmin=vmin,
                        vmax=vmax,
                        zorder=1,
                    )
                    if slab_mask is not None:
                        overlay_slab_on_section(
                            ax,
                            slab_mask[lat_idx, :, :],
                            lons,
                            depths,
                            with_legend=False,
                        )
                    ax.set_title(
                        f'{ftitle}  |  Lat={actual_lat:.1f}°',
                        fontsize=self.fonts['title'],
                        fontweight='bold',
                    )
                    ax.set_xlabel('Longitude (°)', fontsize=self.fonts['label'])
                    if idx == 0:
                        ax.set_ylabel('Depth (km)', fontsize=self.fonts['label'])
                    ax.tick_params(labelsize=self.fonts['tick'])
                    ax.set_ylim(float(depths.max()), float(depths.min()))
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    if idx == n_sections - 1 and last_im is not None:
                        cbar = fig.colorbar(
                            last_im, ax=ax, fraction=0.046, pad=0.02
                        )
                        cbar.set_label(ftitle, fontsize=self.fonts['label'])
                        cbar.ax.tick_params(labelsize=self.fonts['tick'])

        fig.suptitle(
            f'{model_name} - {algorithm_name.upper()} Vertical Sections'
            + ('  |  facies + dlnV + Slab2' if has_pert else '')
            + ('  (black: Slab2 body)' if slab_mask is not None else ''),
            fontsize=self.fonts['suptitle'],
            fontweight='bold',
        )

        filepath = output_dir / f'2-3-5_{algorithm_name}_vertical_sections.png'
        fig.savefig(filepath, dpi=self.config.visualization['dpi'])
        # 同步 jpg
        try:
            fig.savefig(
                filepath.with_suffix('.jpg'),
                dpi=self.config.visualization['dpi'],
            )
        except Exception:
            pass
        plt.close(fig)

        self.logger.info(f"      ✅ 保存: 2-3-5_{algorithm_name}_vertical_sections.png")

    def plot_cluster_profiles(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        spatial_indices: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Path,
        algorithm_name: str,
        model_name: str,
        per_band: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        绘制各簇深度–特征剖面（δlnVp 与 δlnVs 同轴叠画）。

        有 per_band 时按深度带分行组图，带内统一深度轴与扰动轴量程，
        便于比较 Vp/Vs 耦合与簇间幅度。
        """
        self.logger.info(f"    📊 绘制聚类速度剖面 ({algorithm_name})...")
        self.fonts = apply_clustering_plot_style(self.config)

        labels = np.asarray(labels)
        X = np.asarray(X)
        spatial_indices = np.asarray(spatial_indices)
        depths_all = np.asarray(metadata['depths'], dtype=float)

        # 反标准化到扰动/原始特征空间
        X_plot = X.copy()
        if (
            self.processor is not None
            and hasattr(self.processor, 'scaler')
            and self.processor.scaler is not None
        ):
            try:
                X_plot = self.processor.scaler.inverse_transform(X)
            except Exception:
                X_plot = X

        feature_names = list(
            metadata.get('feature_display_names')
            or metadata.get('features', [f'F{i}' for i in range(X_plot.shape[1])])
        )
        n_features = min(2, X_plot.shape[1])  # 默认叠画前两特征 (vp/vs)
        feat_space = str(metadata.get('feature_space', '') or '')
        is_pert = 'perturbation' in feat_space or any(
            'dln' in str(n).lower() or 'δ' in str(n) for n in feature_names
        )
        xlabel = r'$\delta\ln V$' if is_pert else 'Velocity (km/s)'

        def _feat_label(name: str) -> str:
            n = str(name).lower()
            if is_pert:
                if 'vp' in n:
                    return r'$\delta\ln V_p$'
                if 'vs' in n:
                    return r'$\delta\ln V_s$'
            return str(name)

        # 特征样式：同轴叠画，线型区分 Vp/Vs，阴影用低透明度避免混色
        feat_styles = [
            {
                'color': '#2563EB',
                'ls': '-',
                'label': _feat_label(feature_names[0] if n_features > 0 else 'vp'),
            },
            {
                'color': '#DC2626',
                'ls': '--',
                'label': _feat_label(feature_names[1] if n_features > 1 else 'vs'),
            },
        ]

        def _depth_profile(
            cluster_id: int,
        ) -> Optional[Dict[str, Any]]:
            """单簇按深度分箱的均值±IQR 剖面。"""
            mask = labels == cluster_id
            n_pts = int(np.sum(mask))
            if n_pts < 5:
                return None
            z = depths_all[spatial_indices[mask, 2]]
            feats = X_plot[mask, :n_features]
            z_min, z_max = float(np.min(z)), float(np.max(z))
            if not np.isfinite(z_min) or z_max <= z_min:
                return None
            # 仅在簇实际深度跨度内分箱（避免全深度空箱）
            n_bins = int(np.clip(np.ceil((z_max - z_min) / 5.0) + 1, 8, 40))
            bins = np.linspace(z_min, z_max, n_bins)
            centers: List[float] = []
            means = [[] for _ in range(n_features)]
            q1s = [[] for _ in range(n_features)]
            q3s = [[] for _ in range(n_features)]
            for j in range(len(bins) - 1):
                right = bins[j + 1] if j < len(bins) - 2 else bins[j + 1] + 1e-9
                dm = (z >= bins[j]) & (z < right)
                if not np.any(dm):
                    continue
                centers.append(0.5 * (bins[j] + bins[j + 1]))
                for fi in range(n_features):
                    vals = feats[dm, fi]
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        means[fi].append(np.nan)
                        q1s[fi].append(np.nan)
                        q3s[fi].append(np.nan)
                    else:
                        means[fi].append(float(np.mean(vals)))
                        q1s[fi].append(float(np.percentile(vals, 25)))
                        q3s[fi].append(float(np.percentile(vals, 75)))
            if not centers:
                return None
            return {
                'n': n_pts,
                'z': np.asarray(centers, dtype=float),
                'mean': [np.asarray(m, dtype=float) for m in means],
                'q1': [np.asarray(q, dtype=float) for q in q1s],
                'q3': [np.asarray(q, dtype=float) for q in q3s],
                'z_span': (z_min, z_max),
                # 簇整体平均（供 facies 解释标注）
                'mu': [
                    float(np.nanmean(feats[:, fi])) for fi in range(n_features)
                ],
            }

        # ---------- 按带组织簇列表 ----------
        band_groups: List[Tuple[str, str, List[int], Tuple[float, float]]] = []
        if per_band:
            for bname, binfo in per_band.items():
                pmask = binfo.get('point_mask')
                if pmask is None:
                    continue
                pmask = np.asarray(pmask, dtype=bool)
                cids = sorted(
                    int(c) for c in np.unique(labels[pmask & (labels >= 0)])
                )
                if not cids:
                    continue
                z0, z1 = binfo.get('depth_range', [np.nan, np.nan])
                k = int(binfo.get('n_clusters', len(cids)))
                subtitle = f'{bname}  ({z0:.0f}–{z1:.0f} km, K={k})'
                band_groups.append((bname, subtitle, cids, (float(z0), float(z1))))
        if not band_groups:
            cids = [int(c) for c in np.unique(labels[labels >= 0])]
            z_lo = float(np.nanmin(depths_all))
            z_hi = float(np.nanmax(depths_all))
            band_groups = [('all', 'all depths', cids, (z_lo, z_hi))]

        max_cid = max((max(g[2]) for g in band_groups if g[2]), default=0)
        colors = generate_distinct_colors(
            max(max_cid + 1, self.config.visualization['n_colors'])
        )

        # 预计算剖面
        profiles: Dict[int, Dict[str, Any]] = {}
        for _, _, cids, _ in band_groups:
            for cid in cids:
                if cid not in profiles:
                    prof = _depth_profile(cid)
                    if prof is not None:
                        profiles[cid] = prof

        # facies 解释所需的 Vp/Vs 特征索引（扰动域才有意义）
        vp_idx = next(
            (
                i for i, nm in enumerate(feature_names[:n_features])
                if 'vp' in str(nm).lower()
            ),
            0,
        )
        vs_idx = next(
            (
                i for i, nm in enumerate(feature_names[:n_features])
                if 'vs' in str(nm).lower()
            ),
            max(n_features - 1, 0),
        )

        # ---------- 布局：每带独立行块，带内统一坐标 ----------
        n_cols = 5
        row_heights: List[float] = []
        band_row_spans: List[
            Tuple[int, int, str, str, List[int], Tuple[float, float]]
        ] = []
        row_cursor = 0
        for bname, subtitle, cids, z_rng in band_groups:
            cids_ok = [c for c in cids if c in profiles]
            if not cids_ok:
                continue
            n_r = (len(cids_ok) + n_cols - 1) // n_cols
            band_row_spans.append((row_cursor, n_r, bname, subtitle, cids_ok, z_rng))
            row_heights.extend([1.0] * n_r)
            row_cursor += n_r

        if not band_row_spans:
            self.logger.warning("      ⚠️ 无可用簇剖面，跳过 2-3-6")
            return

        F = self.fonts
        n_rows_total = len(row_heights)
        fig_w = 3.8 * n_cols
        fig_h = 3.3 * n_rows_total + 0.7 * len(band_row_spans)
        fig = plt.figure(figsize=(fig_w, max(fig_h, 6.0)))
        height_ratios: List[float] = []
        for _start, n_r, _bname, _subtitle, _cids_ok, _z_rng in band_row_spans:
            height_ratios.append(0.38)  # band 标题行
            height_ratios.extend([1.0] * n_r)

        gs = fig.add_gridspec(
            len(height_ratios),
            n_cols,
            height_ratios=height_ratios,
            hspace=0.55,
            wspace=0.32,
            left=0.07,
            right=0.98,
            top=0.93,
            bottom=0.05,
        )

        # 带内共享 xlim：取各簇均值与 IQR 的 2–98 分位
        band_xlim: Dict[str, Tuple[float, float]] = {}
        for _start, _n_r, _bname, subtitle, cids_ok, _z_rng in band_row_spans:
            vals: List[float] = []
            for cid in cids_ok:
                p = profiles[cid]
                for fi in range(n_features):
                    vals.extend(p['mean'][fi][np.isfinite(p['mean'][fi])].tolist())
                    vals.extend(p['q1'][fi][np.isfinite(p['q1'][fi])].tolist())
                    vals.extend(p['q3'][fi][np.isfinite(p['q3'][fi])].tolist())
            if not vals:
                band_xlim[subtitle] = (-0.05, 0.05)
                continue
            lo, hi = np.percentile(vals, [2, 98])
            pad = 0.12 * max(hi - lo, 1e-4)
            if is_pert:
                m = max(abs(lo), abs(hi)) + pad
                band_xlim[subtitle] = (-m, m)
            else:
                band_xlim[subtitle] = (float(lo - pad), float(hi + pad))

        legend_handles: List[Any] = []
        legend_done = False
        h_cursor = 0
        for _start, n_r, bname, subtitle, cids_ok, z_rng in band_row_spans:
            ax_t = fig.add_subplot(gs[h_cursor, :])
            ax_t.axis('off')
            ax_t.text(
                0.0,
                0.3,
                subtitle,
                transform=ax_t.transAxes,
                fontsize=F['title'],
                fontweight='bold',
                color='0.2',
                va='center',
            )
            h_cursor += 1

            xlim = band_xlim.get(subtitle, (-0.05, 0.05))
            n_band = sum(int(profiles[c]['n']) for c in cids_ok)
            n_band = max(n_band, 1)

            for r in range(n_r):
                for c in range(n_cols):
                    idx = r * n_cols + c
                    ax = fig.add_subplot(gs[h_cursor, c])
                    if idx >= len(cids_ok):
                        ax.axis('off')
                        continue
                    cid = cids_ok[idx]
                    p = profiles[cid]
                    cc = colors[cid % len(colors)]
                    z_lo, z_hi = p['z_span']
                    z_pad = 0.03 * max(z_hi - z_lo, 1.0)
                    pct = 100.0 * float(p['n']) / float(n_band)

                    for fi in range(n_features):
                        st = feat_styles[fi % len(feat_styles)]
                        mu = p['mean'][fi]
                        q1 = p['q1'][fi]
                        q3 = p['q3'][fi]
                        zz = p['z']
                        ok = np.isfinite(mu) & np.isfinite(zz)
                        if not np.any(ok):
                            continue
                        ax.fill_betweenx(
                            zz[ok],
                            q1[ok],
                            q3[ok],
                            color=st['color'],
                            alpha=0.12,
                            linewidth=0,
                            zorder=1,
                        )
                        (line,) = ax.plot(
                            mu[ok],
                            zz[ok],
                            linestyle=st['ls'],
                            color=st['color'],
                            linewidth=2.0,
                            label=st['label'],
                            zorder=3,
                        )
                        if not legend_done:
                            legend_handles.append(line)
                    if len(legend_handles) >= n_features:
                        legend_done = True

                    if is_pert:
                        ax.axvline(
                            0.0,
                            color='0.5',
                            linewidth=0.7,
                            linestyle=':',
                            zorder=0,
                        )
                    ax.set_xlim(xlim)
                    ax.set_ylim(z_hi + z_pad, z_lo - z_pad)  # 深度向下
                    ax.spines['left'].set_color(cc)
                    ax.spines['left'].set_linewidth(2.5)
                    for spine in ('top', 'right'):
                        ax.spines[spine].set_visible(False)

                    ax.set_title(
                        f'C{cid}  ({pct:.1f}%)',
                        fontsize=F['title'],
                        fontweight='bold',
                        color=cc,
                        pad=6,
                    )
                    # facies 推测解释（仅扰动域有意义）
                    if is_pert and p.get('mu') is not None:
                        interp = interpret_facies(
                            bname,
                            p['mu'][vp_idx] if vp_idx < len(p['mu']) else np.nan,
                            p['mu'][vs_idx] if vs_idx < len(p['mu']) else np.nan,
                        )
                        # 放在子图标题正下方（轴内顶部）
                        ax.text(
                            0.5,
                            0.985,
                            interp,
                            transform=ax.transAxes,
                            ha='center',
                            va='top',
                            fontsize=max(F['annotation'] - 3, 9),
                            color='0.25',
                            style='italic',
                            wrap=True,
                            bbox=dict(
                                facecolor='white',
                                alpha=0.75,
                                edgecolor='none',
                                pad=1.5,
                            ),
                            zorder=6,
                        )
                    ax.grid(True, axis='both', alpha=0.25, linestyle='--', linewidth=0.6)
                    ax.tick_params(labelsize=F['tick'])

                    if c == 0:
                        ax.set_ylabel('Depth (km)', fontsize=F['label'])
                    else:
                        ax.set_ylabel('')
                    if r == n_r - 1:
                        ax.set_xlabel(xlabel, fontsize=F['label'])
                    else:
                        ax.set_xlabel('')
                h_cursor += 1

        if legend_handles:
            fig.legend(
                legend_handles,
                [h.get_label() for h in legend_handles],
                loc='upper right',
                ncol=n_features,
                fontsize=F['legend'],
                frameon=True,
                fancybox=False,
                edgecolor='0.7',
                bbox_to_anchor=(0.98, 0.995),
            )

        title_x = 'dlnV' if is_pert else 'velocity'
        fig.suptitle(
            f'{model_name} — {algorithm_name.upper()} Cluster Profiles '
            f'(vp & vs on same axis; mean ± IQR; {title_x})',
            fontsize=F['suptitle'],
            fontweight='bold',
            y=0.995,
        )

        filepath = output_dir / f'2-3-6_{algorithm_name}_cluster_profiles.png'
        fig.savefig(
            filepath, dpi=self.config.visualization['dpi'], bbox_inches='tight'
        )
        # 同步 jpg
        try:
            fig.savefig(
                filepath.with_suffix('.jpg'),
                dpi=self.config.visualization['dpi'],
                bbox_inches='tight',
            )
        except Exception:
            pass
        plt.close(fig)
        self.logger.info(f"      ✅ 保存: 2-3-6_{algorithm_name}_cluster_profiles.png")
    
    def plot_feature_distributions(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Path,
        algorithm_name: str,
        model_name: str,
        per_band: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        绘制聚类特征分布图。

        有 per_band 时按深度带分面（行=带，列=特征）；
        每格为小提琴 + 内嵌箱线（无离群点），簇按中位数排序。
        """
        self.logger.info(f"    📊 绘制特征分布 ({algorithm_name})...")
        self.fonts = apply_clustering_plot_style(self.config)

        feature_names = list(
            metadata.get('feature_display_names')
            or metadata.get(
                'features',
                [f'Feature {i + 1}' for i in range(X.shape[1])],
            )
        )
        n_features = min(2, X.shape[1])  # 默认 vp/vs 两列
        labels = np.asarray(labels)
        X = np.asarray(X)

        # 反标准化到扰动/原始特征空间（与训练一致）
        X_plot = X.copy()
        if (
            self.processor is not None
            and hasattr(self.processor, 'scaler')
            and self.processor.scaler is not None
        ):
            try:
                X_plot = self.processor.scaler.inverse_transform(X)
            except Exception:
                X_plot = X

        max_cid = int(labels.max()) if np.any(labels >= 0) else 0
        colors = generate_distinct_colors(
            max(max_cid + 1, self.config.visualization['n_colors'])
        )
        rng = np.random.default_rng(RANDOM_SEED)
        max_sample = 3000

        def _cluster_series(
            point_mask: np.ndarray, feat_idx: int
        ) -> Tuple[List[int], List[np.ndarray], List[int]]:
            """返回按中位数排序的 (cluster_ids, samples, counts)"""
            cids: List[int] = []
            samples: List[np.ndarray] = []
            counts: List[int] = []
            local = labels[point_mask]
            xloc = X_plot[point_mask, feat_idx]
            for cid in np.unique(local[local >= 0]):
                vals = xloc[local == cid]
                vals = vals[np.isfinite(vals)]
                if vals.size < 5:
                    continue
                n_tot = int(vals.size)
                if n_tot > max_sample:
                    vals = rng.choice(vals, size=max_sample, replace=False)
                cids.append(int(cid))
                samples.append(vals.astype(float))
                counts.append(n_tot)
            if not cids:
                return [], [], []
            order = np.argsort([float(np.median(s)) for s in samples])
            return (
                [cids[i] for i in order],
                [samples[i] for i in order],
                [counts[i] for i in order],
            )

        def _draw_violin_panel(
            ax,
            cids: List[int],
            samples: List[np.ndarray],
            counts: List[int],
            feat_name: str,
            panel_title: str,
            ylabel: bool,
        ) -> None:
            F = self.fonts
            if not cids:
                ax.axis('off')
                ax.set_title(f'{panel_title}\n(no data)', fontsize=F['title'])
                return
            # 长表供 seaborn
            rows_v: List[float] = []
            rows_c: List[str] = []
            for cid, samp in zip(cids, samples):
                rows_v.extend(samp.tolist())
                rows_c.extend([f'C{cid}'] * len(samp))
            df = pd.DataFrame({'value': rows_v, 'cluster': rows_c})
            order = [f'C{c}' for c in cids]
            palette = {
                f'C{c}': colors[c % len(colors)] for c in cids
            }
            try:
                sns.violinplot(
                    data=df,
                    x='cluster',
                    y='value',
                    order=order,
                    hue='cluster',
                    hue_order=order,
                    palette=palette,
                    ax=ax,
                    cut=0,
                    inner=None,
                    linewidth=0.6,
                    density_norm='width',
                    legend=False,
                )
            except TypeError:
                # 兼容旧版 seaborn
                sns.violinplot(
                    data=df,
                    x='cluster',
                    y='value',
                    order=order,
                    palette=palette,
                    ax=ax,
                    cut=0,
                    inner=None,
                    linewidth=0.6,
                    scale='width',
                )
            # 内嵌箱线（无 fliers）
            bp = ax.boxplot(
                samples,
                positions=np.arange(len(cids)),
                widths=0.12,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )
            for patch, cid in zip(bp['boxes'], cids):
                patch.set_facecolor('white')
                patch.set_alpha(0.85)
                patch.set_edgecolor(colors[cid % len(colors)])
            for key in ('whiskers', 'caps', 'medians'):
                for line in bp[key]:
                    line.set_color('0.2')
                    line.set_linewidth(0.8)
            # 簇 ID + 带内占比（%）
            n_tot = max(int(sum(counts)), 1)
            ax.set_xticks(np.arange(len(cids)))
            ax.set_xticklabels(
                [
                    f'C{c}\n{100.0 * n / n_tot:.1f}%'
                    for c, n in zip(cids, counts)
                ],
                fontsize=F['tick'],
                linespacing=1.2,
            )
            ax.tick_params(axis='x', pad=2, labelsize=F['tick'])
            ax.tick_params(axis='y', labelsize=F['tick'])
            ax.axhline(0.0, color='gray', linewidth=0.7, linestyle='--', alpha=0.7)
            ax.set_xlabel('')
            ax.set_ylabel(feat_name if ylabel else '', fontsize=F['label'])
            ax.set_title(panel_title, fontsize=F['title'], fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')

        # ---------- 按深度带分面 ----------
        band_items: List[Tuple[str, np.ndarray, str]] = []
        if per_band:
            for bname, binfo in per_band.items():
                pmask = binfo.get('point_mask')
                if pmask is None:
                    continue
                pmask = np.asarray(pmask, dtype=bool)
                if not np.any(pmask):
                    continue
                z0, z1 = binfo.get('depth_range', [np.nan, np.nan])
                k = int(binfo.get('n_clusters', 0))
                subtitle = f'{bname}  ({z0:.0f}–{z1:.0f} km, K={k})'
                band_items.append((bname, pmask, subtitle))

        if band_items:
            n_rows = len(band_items)
            n_cols = n_features
            # 按各带簇数动态加宽，避免浅部 K≈10 时标签挤叠
            max_k = 2
            for bname, pmask, _sub in band_items:
                k_cfg = int(per_band.get(bname, {}).get('n_clusters', 0) or 0)
                k_obs = int(len(np.unique(labels[pmask & (labels >= 0)])))
                max_k = max(max_k, k_cfg, k_obs)
            # 百分比标注较短：略宽于紧凑版，避免小提琴过挤
            panel_w = max(6.0, 0.90 * max_k)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(panel_w * n_cols, 3.8 * n_rows),
                squeeze=False,
            )
            for r, (_bname, pmask, subtitle) in enumerate(band_items):
                for c in range(n_features):
                    feat_name = (
                        feature_names[c]
                        if c < len(feature_names)
                        else f'Feature {c + 1}'
                    )
                    cids, samples, counts = _cluster_series(pmask, c)
                    _draw_violin_panel(
                        axes[r, c],
                        cids,
                        samples,
                        counts,
                        feat_name,
                        panel_title=f'{subtitle}  ·  {feat_name}',
                        ylabel=(c == 0),
                    )
            F = self.fonts
            fig.suptitle(
                f'{model_name} — {algorithm_name.upper()} Feature Distributions '
                f'(by depth band; violin + quartile box)',
                fontsize=F['suptitle'],
                fontweight='bold',
                y=0.998,
            )
            fig.tight_layout(rect=[0, 0.01, 1, 0.96], h_pad=1.0, w_pad=0.6)
        else:
            # 回退：全部簇一行
            n_cols = n_features
            n_all = int(len(np.unique(labels[labels >= 0])))
            n_all = max(n_all, 8)
            fig, axes = plt.subplots(
                1,
                n_cols,
                figsize=(max(6.0, 0.75 * n_all) * n_cols, 4.5),
                squeeze=False,
            )
            all_mask = labels >= 0
            for c in range(n_features):
                feat_name = (
                    feature_names[c]
                    if c < len(feature_names)
                    else f'Feature {c + 1}'
                )
                cids, samples, counts = _cluster_series(all_mask, c)
                _draw_violin_panel(
                    axes[0, c],
                    cids,
                    samples,
                    counts,
                    feat_name,
                    panel_title=feat_name,
                    ylabel=(c == 0),
                )
            F = self.fonts
            fig.suptitle(
                f'{model_name} — {algorithm_name.upper()} Feature Distributions '
                f'(violin + quartile box)',
                fontsize=F['suptitle'],
                fontweight='bold',
            )
            fig.tight_layout(rect=[0, 0.01, 1, 0.94], w_pad=0.6)

        filepath = output_dir / f'2-3-7_{algorithm_name}_feature_distributions.png'
        fig.savefig(
            filepath, dpi=self.config.visualization['dpi'], bbox_inches='tight'
        )
        plt.close(fig)
        self.logger.info(
            f"      ✅ 保存: 2-3-7_{algorithm_name}_feature_distributions.png"
        )

    def plot_cluster_centers(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        metadata: Dict[str, Any],
        gmm_model: Optional[GaussianMixture],
        output_dir: Path,
        algorithm_name: str,
        model_name: str
    ) -> None:
        """绘制聚类中心的速度结构"""
        self.logger.info(f"    📊 绘制聚类中心 ({algorithm_name})...")
        
        unique_labels = np.unique(labels[labels >= 0])
        n_clusters = len(unique_labels)
        
        # 获取聚类中心
        if gmm_model and hasattr(gmm_model, 'means_'):
            centers = gmm_model.means_
        else:
            # 手动计算中心
            centers = []
            for label in unique_labels:
                mask = labels == label
                if np.sum(mask) > 0:
                    centers.append(np.mean(X[mask], axis=0))
            centers = np.array(centers)
        
        # 反归一化
        if self.processor and hasattr(self.processor, 'scaler') and self.processor.scaler:
            centers_original = self.processor.scaler.inverse_transform(centers)
        else:
            centers_original = centers
        
        # 创建图形
        n_features = centers.shape[1]
        feature_names = metadata.get('features', [f'Feature {i+1}' for i in range(n_features)])
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 生成颜色
        colors = generate_distinct_colors(max(n_clusters, self.config.visualization['n_colors']))
        
        # 1. 热力图
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(centers_original.T, aspect='auto', cmap='RdYlBu_r')
        ax1.set_xlabel('Cluster ID', fontsize=15)
        ax1.set_ylabel('Features', fontsize=15)
        ax1.set_title('Cluster Centers Heatmap', fontsize=16, fontweight='bold')
        ax1.set_yticks(range(n_features))
        ax1.set_yticklabels(feature_names[:n_features], fontsize=14)
        ax1.set_xticks(range(n_clusters))
        plt.colorbar(im, ax=ax1, label='Value (km/s or g/cm³)')
        
        # 2. 平行坐标图
        ax2 = fig.add_subplot(gs[0, 1])
        centers_norm = (centers_original - centers_original.min(axis=0)) / \
                      (centers_original.max(axis=0) - centers_original.min(axis=0) + 1e-10)
        
        x = np.arange(n_features)
        for i in range(n_clusters):
            ax2.plot(x, centers_norm[i], 'o-', 
                    color=colors[i % len(colors)], 
                    label=f'C{i}', alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Features', fontsize=15)
        ax2.set_ylabel('Normalized Value', fontsize=15)
        ax2.set_title('Cluster Centers - Parallel Coordinates', fontsize=16, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(feature_names[:n_features], rotation=45, ha='right', fontsize=14)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, ncol=max(1, n_clusters//10))
        ax2.grid(True, alpha=0.3)
        
        # 3. Vp-Vs关系
        ax3 = fig.add_subplot(gs[0, 2])
        if 'vp' in feature_names and 'vs' in feature_names:
            vp_idx = feature_names.index('vp')
            vs_idx = feature_names.index('vs')
            
            for i in range(n_clusters):
                ax3.scatter(centers_original[i, vs_idx], centers_original[i, vp_idx],
                          s=200, c=[colors[i % len(colors)]], edgecolors='black', linewidth=2,
                          label=f'Cluster {i}', zorder=10, alpha=0.8)
            
            vs_range = np.linspace(centers_original[:, vs_idx].min(), 
                                  centers_original[:, vs_idx].max(), 100)
            ax3.plot(vs_range, vs_range * 1.732, 'k--', alpha=0.3, linewidth=2, label='Vp/Vs=1.732')
            
            ax3.set_xlabel('Vs (km/s)', fontsize=15)
            ax3.set_ylabel('Vp (km/s)', fontsize=15)
            ax3.set_title('Cluster Centers - Vp vs Vs', fontsize=16, fontweight='bold')
            ax3.legend(fontsize=14, ncol=2)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Vp/Vs plot\nnot available', 
                    ha='center', va='center', fontsize=16, transform=ax3.transAxes)
            ax3.set_title('Cluster Centers - Vp vs Vs', fontsize=16, fontweight='bold')
        
        # 4. 聚类大小分布
        ax4 = fig.add_subplot(gs[1, 0])
        unique_labels_list, counts = np.unique(labels[labels >= 0], return_counts=True)
        bars = ax4.bar(unique_labels_list, counts, 
                      color=[colors[i % len(colors)] for i in range(len(unique_labels_list))],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('Cluster ID', fontsize=22)
        ax4.set_ylabel('Number of Points', fontsize=22)
        ax4.set_title('Cluster Size Distribution', fontsize=22, fontweight='bold')
        ax4.set_xticks(unique_labels_list)
        ax4.grid(True, alpha=0.3, axis='y')
        
        total = np.sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}\n({count/total:.1%})',
                    ha='center', va='bottom', fontsize=14)
        
        # 5. 特征值表格
        ax5 = fig.add_subplot(gs[1, 1:])
        ax5.axis('off')
        
        # 创建表格数据
        table_data = [['Cluster'] + feature_names[:n_features]]
        for i in range(n_clusters):
            row = [f'C{i}'] + [f'{centers_original[i, j]:.3f}' for j in range(n_features)]
            table_data.append(row)
        
        table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.1] + [0.9/n_features]*n_features)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # 设置表头样式
        for j in range(len(table_data[0])):
            table[(0, j)].set_facecolor('#3498db')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        # 设置聚类行颜色
        for i in range(1, len(table_data)):
            table[(i, 0)].set_facecolor(colors[(i-1) % len(colors)])
            table[(i, 0)].set_text_props(weight='bold')
        
        plt.suptitle(f'{model_name} - {algorithm_name.upper()} Cluster Centers Analysis',
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        filepath = output_dir / f'2-3-8_{algorithm_name}_cluster_centers.png'
        plt.savefig(filepath, dpi=self.config.visualization['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"      ✅ 保存: 2-3-8_{algorithm_name}_cluster_centers.png")

