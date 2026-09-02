"""
EASTASIA-FWI 跨模型速度相投票分析模块 (v1.0)
====================================================================================

科学目标：
- 借鉴 Lekić et al. (2012, EPSL) 的聚类投票思路：各模型独立做体素级
  GMM 聚类（2_3 模块）后，将每个簇按平均 δlnVs 归为 慢/中性/快 三类，
  在公共网格上跨模型投票，得到结构稳健性地图
- 所有模型一致判为"快"（或"慢"）的区域 = 跨模型稳健结构
  （克拉通根、俯冲板片、软流圈低速区等）
- 模型间分歧区 = 模型不确定区，是后续波形评估（Phase D）需要
  重点裁决、模型融合（Phase E）需要谨慎处理的区域

与 4_Fusion/4_1_Voting_map.py 的区别：
- 4_1 对速度值做 median 投票，目标是构建融合"超级模型"（数据域融合）
- 本模块对聚类标签的结构分类投票，目标是评估结构稳健性（模型空间分析）

方法流程：
1. 读取 2_3 输出：cluster_labels (NetCDF) + 每簇 δlnVp/δlnVs 均值 (CSV)
   + 分带 K (metadata JSON)
2. 按深度带阈值（与 2_3 FACIES_DVS_THRESHOLDS 的 weak 阈值一致）将每个
   簇归类：δlnVs ≥ +阈值 → fast(+1)；≤ −阈值 → slow(−1)；否则 neutral(0)
3. 各模型分类体最近邻重采样到公共网格（模型覆盖交集，0.5°）
4. 投票和 S = Σ votes ∈ [−3, +3]（3 模型），|S|=3 为全体一致
5. 可视化：
   2-4   多深度投票地图（7 级离散发散色标，慢=红 快=蓝）+ 一致率曲线
   2-4-2 投票体经度–深度剖面（叠 Slab2 dep+thk 体几何，复用 2_3）
   2-4-3 跨模型快/慢簇平均 δlnVs(z) 剖面（cf. Lekić 2012 Fig.5）
   2-4-4 分歧归因图（2:1 分歧体素上标出少数派模型 + 占比随深度）
   输出 NetCDF 投票体与统计 JSON

作者：EASTASIA-FWI Team
日期：2026-09-01
版本：v1.1
"""

import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import BaseConfig

warnings.filterwarnings('ignore')


@dataclass
class FaciesVotingConfig:
    """跨模型速度相投票配置类"""

    # 参与投票的模型（2_3 聚类结果目录名）
    models: List[str] = field(default_factory=lambda: [
        '2024_FWEA23',
        '2024_EARA2024',
        '2022_SinoScope1.0',
    ])

    # 2_3 聚类方案（决定输入目录与分带结构）
    clustering_scheme: str = 'moho_4band'

    # 深度带顺序（与 2_3 moho_4band 一致，用于按簇 id 偏移映射簇→带）
    band_order: List[str] = field(default_factory=lambda: [
        'crust', 'lithosphere', 'transition_zone', 'lower_mantle',
    ])

    # 簇→快/慢/中性归类的 δlnVs 阈值（与 2_3 FACIES_DVS_THRESHOLDS 的
    # weak 阈值一致，保持两模块解释体系统一）
    class_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'crust': 0.03,
        'lithosphere': 0.010,
        'transition_zone': 0.005,
        'lower_mantle': 0.004,
    })

    # 公共网格（取模型覆盖交集；分辨率折中于 0.25° 与 1° 之间）
    common_grid: Dict[str, Any] = field(default_factory=lambda: {
        'dlat': 0.5,
        'dlon': 0.5,
        'depth_step_km': 20.0,   # 与三模型深度采样均可最近邻对齐
    })

    # 出图深度（km）
    map_depths_km: List[float] = field(default_factory=lambda: [
        60, 100, 200, 300, 440, 560, 700, 900,
    ])

    # 投票剖面纬度（°N）与分歧归因图深度（km）
    section_lats: List[float] = field(default_factory=lambda: [30, 35, 40])
    dissent_depths_km: List[float] = field(default_factory=lambda: [
        100, 300, 440, 560,
    ])

    # 各模型标识色（分歧归因图 / 平均剖面图共用）
    model_colors: Dict[str, str] = field(default_factory=lambda: {
        '2024_FWEA23': '#1b9e77',
        '2024_EARA2024': '#d95f02',
        '2022_SinoScope1.0': '#7570b3',
    })

    # 原始模型 NetCDF（重算 δlnVs 用，与 2_3 数据源一致）
    model_nc_pattern: str = 'models/processed/{model}/{model}_original.nc'

    # 可视化配置
    visualization: Dict[str, Any] = field(default_factory=lambda: {
        'dpi': 300,
        'pdf_dpi': 300,
        'figure_format': ['jpg', 'pdf'],
        # 7 级发散色标（ColorBrewer RdBu，慢=红、快=蓝，符合层析成像惯例）
        'vote_colors': [
            '#b2182b', '#ef8a62', '#fddbc7', '#f7f7f7',
            '#d1e5f0', '#67a9cf', '#2166ac',
        ],
        'nan_color': '0.88',
    })

    # 日志配置
    logging: Dict[str, Any] = field(default_factory=lambda: {
        'level': 'INFO',
        'console_output': True,
    })


class FaciesVotingAnalyzer:
    """跨模型速度相投票分析主类"""

    def __init__(self, output_dir: Optional[str] = None):
        # 1. 全局配置
        self.base_config = BaseConfig()
        # 2. 模块配置
        self.config = FaciesVotingConfig()
        # 3. 输出目录
        scheme = self.config.clustering_scheme
        self.input_root = (
            self.base_config.dirs['results'] / 'model_clustering' / scheme
        )
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.input_root / 'voting'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 4. 日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.FaciesVoting', self.config.logging['level']
        )
        # 5. 区域参数
        bounds = self.base_config.get_region_bounds()
        self.region = [
            bounds['lon_min'], bounds['lon_max'],
            bounds['lat_min'], bounds['lat_max'],
        ]

        self.logger.info("🎯 跨模型速度相投票模块初始化完成")
        self.logger.info(f"  输入: {self.input_root}")
        self.logger.info(f"  输出: {self.output_dir}")
        self.logger.info(f"  模型: {', '.join(self.config.models)}")

    # ------------------------------------------------------------------
    # 数据加载与簇分类
    # ------------------------------------------------------------------

    def _cluster_class_lut(self, model: str) -> np.ndarray:
        """
        构建 簇id → 快/慢/中性 查找表。

        依据 cluster_statistics.csv 的每簇 δlnVs 均值与所属深度带的阈值：
        vs_mean ≥ +thr → +1 (fast)；vs_mean ≤ −thr → −1 (slow)；否则 0。

        Args:
            model: 模型目录名

        Returns:
            长度为全局簇数的 int8 数组（下标 = 全局簇 id）
        """
        mdir = self.input_root / model
        stats = pd.read_csv(mdir / 'cluster_statistics.csv')
        meta = json.load(open(mdir / 'analysis_metadata.json'))
        band_k: Dict[str, int] = meta['band_optimal_k']

        n_total = int(stats['cluster_id'].max()) + 1
        # 簇 id 按带顺序偏移编码（2_3 offset_labels_across_bands=True）
        band_of_id: List[str] = []
        for bname in self.config.band_order:
            band_of_id.extend([bname] * int(band_k[bname]))
        if len(band_of_id) != n_total:
            raise ValueError(
                f"[{model}] 簇数不匹配: 分带合计 {len(band_of_id)} "
                f"vs CSV {n_total}"
            )

        lut = np.zeros(n_total, dtype=np.int8)
        for _, row in stats.iterrows():
            cid = int(row['cluster_id'])
            thr = self.config.class_thresholds[band_of_id[cid]]
            vs_mean = float(row['vs_mean'])
            if vs_mean >= thr:
                lut[cid] = 1
            elif vs_mean <= -thr:
                lut[cid] = -1
        self.logger.info(
            f"  [{model}] {n_total} 簇分类: "
            f"fast={int((lut == 1).sum())}, slow={int((lut == -1).sum())}, "
            f"neutral={int((lut == 0).sum())}"
        )
        return lut

    def _load_class_volume(self, model: str) -> xr.DataArray:
        """
        加载模型聚类标签并转为分类体（-1/0/+1，无效处 NaN）。

        Returns:
            DataArray(latitude, longitude, depth)，float32
        """
        mdir = self.input_root / model
        ds = xr.open_dataset(mdir / 'gmm_clustering_results.nc')
        labels = ds['cluster_labels'].astype(np.int32)
        lut = self._cluster_class_lut(model)

        lab = labels.values
        cls = np.full(lab.shape, np.nan, dtype=np.float32)
        valid = lab >= 0
        cls[valid] = lut[lab[valid]]
        da = xr.DataArray(
            cls, coords=labels.coords, dims=labels.dims, name='facies_class'
        )
        ds.close()
        return da

    # ------------------------------------------------------------------
    # 公共网格与投票
    # ------------------------------------------------------------------

    def _build_common_grid(
        self, volumes: List[xr.DataArray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """公共网格 = 各模型经纬度覆盖交集 × 统一深度层"""
        g = self.config.common_grid
        lat0 = max(float(v.latitude.min()) for v in volumes)
        lat1 = min(float(v.latitude.max()) for v in volumes)
        lon0 = max(float(v.longitude.min()) for v in volumes)
        lon1 = min(float(v.longitude.max()) for v in volumes)
        dep1 = min(float(v.depth.max()) for v in volumes)

        lats = np.arange(np.ceil(lat0 * 2) / 2, lat1 + 1e-6, g['dlat'])
        lons = np.arange(np.ceil(lon0 * 2) / 2, lon1 + 1e-6, g['dlon'])
        deps = np.arange(0.0, dep1 + 1e-6, g['depth_step_km'])
        self.logger.info(
            f"  公共网格: lat {lats[0]:.1f}–{lats[-1]:.1f} × "
            f"lon {lons[0]:.1f}–{lons[-1]:.1f} (0.5°), "
            f"depth 0–{deps[-1]:.0f} km ({len(deps)} 层)"
        )
        return lats, lons, deps

    def run_voting(self) -> xr.Dataset:
        """
        执行跨模型投票。

        Returns:
            Dataset(vote_sum, n_fast, n_slow, n_valid) on 公共网格
        """
        self.logger.info("📥 加载各模型分类体...")
        volumes = [self._load_class_volume(m) for m in self.config.models]
        lats, lons, deps = self._build_common_grid(volumes)

        self.logger.info("🗳️ 最近邻重采样并投票...")
        classes = np.stack(
            [
                v.sel(
                    latitude=lats, longitude=lons, depth=deps, method='nearest'
                ).values
                for v in volumes
            ],
            axis=0,
        )  # (n_models, nlat, nlon, ndep)
        # 缓存供分歧归因图复用
        self._classes_common = classes

        valid = np.isfinite(classes)
        n_valid = valid.sum(axis=0).astype(np.int8)
        vote_sum = np.where(
            n_valid == len(volumes), np.nansum(classes, axis=0), np.nan
        ).astype(np.float32)
        n_fast = np.nansum(classes == 1, axis=0).astype(np.int8)
        n_slow = np.nansum(classes == -1, axis=0).astype(np.int8)

        ds = xr.Dataset(
            {
                'vote_sum': (('latitude', 'longitude', 'depth'), vote_sum),
                'n_fast': (('latitude', 'longitude', 'depth'), n_fast),
                'n_slow': (('latitude', 'longitude', 'depth'), n_slow),
                'n_valid': (('latitude', 'longitude', 'depth'), n_valid),
            },
            coords={'latitude': lats, 'longitude': lons, 'depth': deps},
            attrs={
                'title': 'Cross-model velocity-facies voting',
                'models': ', '.join(self.config.models),
                'method': (
                    'Per-model GMM facies -> slow/neutral/fast by cluster-mean '
                    'dlnVs -> nearest-neighbor to common grid -> vote sum '
                    '(Lekic et al., 2012 style)'
                ),
                'vote_convention': '-3 = all slow, +3 = all fast (3 models)',
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
        )
        out_nc = self.output_dir / 'facies_voting.nc'
        ds.to_netcdf(out_nc)
        self.logger.info(f"  💾 投票体已保存: {out_nc}")
        return ds

    # ------------------------------------------------------------------
    # 统计与可视化
    # ------------------------------------------------------------------

    def _agreement_profile(self, ds: xr.Dataset) -> pd.DataFrame:
        """逐深度统计一致率（全体一致快/慢、≥2 票一致）"""
        rows = []
        for d in ds.depth.values:
            s = ds['vote_sum'].sel(depth=d).values
            n = np.isfinite(s).sum()
            if n == 0:
                continue
            rows.append({
                'depth_km': float(d),
                'frac_all_fast': float((s == 3).sum() / n),
                'frac_all_slow': float((s == -3).sum() / n),
                'frac_majority': float((np.abs(s) >= 2).sum() / n),
            })
        return pd.DataFrame(rows)

    def plot_voting_maps(self, ds: xr.Dataset) -> Path:
        """
        绘制投票地图（8 个深度切片 + 一致率-深度曲线）。

        Returns:
            主图路径（jpg）
        """
        self.logger.info("🖌️ 绘制投票地图...")
        viz = self.config.visualization
        cmap = ListedColormap(viz['vote_colors'])
        cmap.set_bad(viz['nan_color'])
        norm = BoundaryNorm(np.arange(-3.5, 4.0, 1.0), cmap.N)

        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            3, 3, figsize=(16.5, 14.2),
            subplot_kw={'projection': proj},
        )
        fig.subplots_adjust(
            left=0.05, right=0.97, top=0.90, bottom=0.10,
            hspace=0.18, wspace=0.12,
        )
        axes = axes.ravel()

        lon0, lon1 = float(ds.longitude.min()), float(ds.longitude.max())
        lat0, lat1 = float(ds.latitude.min()), float(ds.latitude.max())

        im = None
        for i, dkm in enumerate(self.config.map_depths_km):
            ax = axes[i]
            sl = ds['vote_sum'].sel(depth=dkm, method='nearest')
            im = ax.pcolormesh(
                ds.longitude, ds.latitude,
                np.ma.masked_invalid(sl.values),
                cmap=cmap, norm=norm, transform=proj, shading='nearest',
            )
            ax.coastlines(resolution='50m', linewidth=0.7, color='0.25')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='0.55')
            ax.set_extent([lon0, lon1, lat0, lat1], crs=proj)
            gl = ax.gridlines(
                draw_labels=True, linewidth=0.3, color='0.7',
                alpha=0.5, linestyle=':',
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 9}
            gl.ylabel_style = {'size': 9}
            ax.set_title(
                f'{float(sl.depth):.0f} km', fontsize=14, fontweight='bold'
            )

        # 第 9 格：一致率随深度曲线（复用地图格的精确位置）
        pos = axes[-1].get_position()
        axes[-1].remove()
        ax_prof = fig.add_axes(
            [pos.x0 + 0.02, pos.y0, pos.width - 0.02, pos.height]
        )
        prof = self._agreement_profile(ds)
        ax_prof.plot(
            prof['frac_all_fast'] * 100, prof['depth_km'],
            color='#2166ac', lw=2, label='All fast (3/3)',
        )
        ax_prof.plot(
            prof['frac_all_slow'] * 100, prof['depth_km'],
            color='#b2182b', lw=2, label='All slow (3/3)',
        )
        ax_prof.plot(
            prof['frac_majority'] * 100, prof['depth_km'],
            color='0.4', lw=1.5, ls='--', label=r'Majority ($|S|\geq 2$)',
        )
        for dkm in self.config.map_depths_km:
            ax_prof.axhline(dkm, color='0.85', lw=0.6, zorder=0)
        ax_prof.set_ylim(float(ds.depth.max()), 0)
        ax_prof.set_xlabel('Area fraction (%)', fontsize=11)
        ax_prof.set_ylabel('Depth (km)', fontsize=11)
        ax_prof.set_title(
            'Cross-model agreement vs depth', fontsize=13, fontweight='bold'
        )
        ax_prof.legend(fontsize=9, loc='center right', framealpha=0.9)
        ax_prof.grid(alpha=0.3, linestyle=':')

        # 共享色标（固定位置，避免挤压地图行）
        cax = fig.add_axes([0.18, 0.045, 0.55, 0.016])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_ticks(range(-3, 4))
        cbar.set_ticklabels([
            'All slow\n(3/3)', 'Slow\n(2/3)', 'Slow\n(1/3)', 'Neutral',
            'Fast\n(1/3)', 'Fast\n(2/3)', 'All fast\n(3/3)',
        ])
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(
            'Facies vote sum  S = N(fast) − N(slow)', fontsize=12
        )

        model_names = ' · '.join(
            m.split('_', 1)[1] for m in self.config.models
        )
        fig.suptitle(
            f'Cross-Model Velocity-Facies Voting ({model_names})\n'
            r'GMM facies classified by cluster-mean $\delta\ln V_S$; '
            'red = consistently slow, blue = consistently fast',
            fontsize=17, fontweight='bold', y=0.975,
        )

        out = None
        for fmt in viz['figure_format']:
            out_path = self.output_dir / f'2-4_facies_voting_map.{fmt}'
            fig.savefig(out_path, dpi=viz['dpi'])
            if fmt == 'jpg':
                out = out_path
        plt.close(fig)
        self.logger.info(f"  ✅ 保存: {out}")
        return out

    # ------------------------------------------------------------------
    # 2-4-2 投票垂直剖面（叠 Slab2 体）
    # ------------------------------------------------------------------

    def _build_slab_mask(
        self, lons: np.ndarray, lats: np.ndarray, deps: np.ndarray
    ) -> Optional[np.ndarray]:
        """复用 2_3 的 GeologyConcordanceEvaluator 构建 Slab2 体掩膜"""
        try:
            import importlib.util
            path = Path(__file__).parent / '2_3_Model_clustering.py'
            spec = importlib.util.spec_from_file_location('mc23', path)
            mc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mc)
            self._mc23 = mc
            evaluator = mc.GeologyConcordanceEvaluator(
                mc.ClusteringConfig(), self.logger
            )
            return evaluator.build_slab_volume_mask(lons, lats, deps)
        except Exception as e:
            self.logger.warning(f"  ⚠️ Slab2 体掩膜构建失败，剖面不叠板片: {e}")
            self._mc23 = None
            return None

    def plot_voting_sections(self, ds: xr.Dataset) -> Optional[Path]:
        """
        投票体经度–深度剖面（2-4-2）：位置图 + 各纬度剖面，叠 Slab2 几何。

        展示跨模型共识结构的垂向延展（板片进入过渡带、岩石圈根深度等）。
        """
        self.logger.info("🖌️ 绘制投票垂直剖面...")
        viz = self.config.visualization
        cmap = ListedColormap(viz['vote_colors'])
        cmap.set_bad(viz['nan_color'])
        norm = BoundaryNorm(np.arange(-3.5, 4.0, 1.0), cmap.N)

        lats = ds.latitude.values
        lons = ds.longitude.values
        deps = ds.depth.values
        slab_mask = self._build_slab_mask(lons, lats, deps)

        sec_lats = self.config.section_lats
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(15, 11))
        gs = plt.GridSpec(
            2, 2, figure=fig, hspace=0.28, wspace=0.16,
            left=0.07, right=0.96, top=0.90, bottom=0.12,
        )

        # 位置图：100 km 投票切片 + 剖面线
        ax_map = fig.add_subplot(gs[0, 0], projection=proj)
        sl = ds['vote_sum'].sel(depth=100, method='nearest')
        ax_map.pcolormesh(
            lons, lats, np.ma.masked_invalid(sl.values),
            cmap=cmap, norm=norm, transform=proj, shading='nearest', alpha=0.9,
        )
        ax_map.coastlines(resolution='50m', linewidth=0.7, color='0.25')
        for slat in sec_lats:
            ax_map.plot(
                [lons.min(), lons.max()], [slat, slat],
                color='k', lw=1.6, ls='--', transform=proj,
            )
            ax_map.text(
                lons.min() + 1.0, slat + 0.8, f'{slat}°N',
                fontsize=11, fontweight='bold', transform=proj,
            )
        ax_map.set_extent(
            [lons.min(), lons.max(), lats.min(), lats.max()], crs=proj
        )
        gl = ax_map.gridlines(
            draw_labels=True, linewidth=0.3, color='0.7',
            alpha=0.5, linestyle=':',
        )
        gl.top_labels = False
        gl.right_labels = False
        ax_map.set_title(
            'Section locations (vote @ 100 km)', fontsize=13, fontweight='bold'
        )

        # 剖面
        im = None
        positions = [(0, 1), (1, 0), (1, 1)]
        for (r, c), slat in zip(positions, sec_lats):
            ax = fig.add_subplot(gs[r, c])
            li = int(np.argmin(np.abs(lats - slat)))
            sec = ds['vote_sum'].values[li]  # (nlon, ndep)
            im = ax.pcolormesh(
                lons, deps, np.ma.masked_invalid(sec.T),
                cmap=cmap, norm=norm, shading='nearest',
            )
            if slab_mask is not None:
                self._mc23.overlay_slab_on_section(
                    ax, slab_mask[li], lons, deps
                )
            for z in (410, 660):
                ax.axhline(z, color='0.35', lw=0.8, ls=':')
            ax.set_ylim(float(deps.max()), 0)
            ax.set_xlabel('Longitude (°E)', fontsize=11)
            ax.set_ylabel('Depth (km)', fontsize=11)
            ax.set_title(
                f'{lats[li]:.1f}°N section', fontsize=13, fontweight='bold'
            )
            ax.tick_params(labelsize=10)

        cax = fig.add_axes([0.20, 0.068, 0.55, 0.016])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_ticks(range(-3, 4))
        cbar.set_ticklabels([
            'All slow', '2 slow', '1 slow', 'Neutral',
            '1 fast', '2 fast', 'All fast',
        ])
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Facies vote sum  S = N(fast) − N(slow)', fontsize=12)

        fig.suptitle(
            'Cross-Model Facies Voting — Vertical Sections\n'
            'black contour/lines = Slab2 body (dep + thk); '
            'dotted lines = 410 / 660 km',
            fontsize=16, fontweight='bold', y=0.975,
        )

        out = None
        for fmt in viz['figure_format']:
            p = self.output_dir / f'2-4-2_voting_sections.{fmt}'
            fig.savefig(p, dpi=viz['dpi'])
            if fmt == 'jpg':
                out = p
        plt.close(fig)
        self.logger.info(f"  ✅ 保存: {out}")
        return out

    # ------------------------------------------------------------------
    # 2-4-3 跨模型快/慢簇平均 δlnVs 剖面（Lekić 2012 Fig.5 的对应物）
    # ------------------------------------------------------------------

    def _dlnvs_volume(self, model: str) -> Optional[xr.DataArray]:
        """
        从原始模型 NetCDF 重算 δlnVs（相对逐深度水平平均，同 2_3 预处理）。
        """
        nc = (
            self.base_config.dirs['data']
            / self.config.model_nc_pattern.format(model=model)
        )
        if not nc.exists():
            self.logger.warning(f"  ⚠️ [{model}] 原始 NetCDF 不存在: {nc}")
            return None
        ds = xr.open_dataset(nc)
        vs = ds['vs'].transpose('latitude', 'longitude', 'depth')
        lnv = np.log(vs.where(vs > 0))
        dln = lnv - lnv.mean(dim=('latitude', 'longitude'), skipna=True)
        out = dln.astype(np.float32).rename('dlnvs')
        ds.close()
        return out

    def plot_class_mean_profiles(self) -> Optional[Path]:
        """
        跨模型快/慢簇平均 δlnVs(z) 剖面（2-4-3）。

        对每个模型：按 2_4 的快/慢分类（簇平均 δlnVs 阈值法）在原生网格上
        逐深度平均 δlnVs。左右面板分别为 0–410 km 与 410–1000 km
        （幅值差一个量级，分开显示）。可暴露模型间振幅系统差与快慢不对称。
        """
        self.logger.info("🖌️ 绘制跨模型快/慢簇平均剖面...")
        viz = self.config.visualization

        profiles: Dict[str, Dict[str, Any]] = {}
        for model in self.config.models:
            cls = self._load_class_volume(model)
            dln = self._dlnvs_volume(model)
            if dln is None:
                continue
            # 对齐到标签网格（同源数据，坐标一致；nearest 仅为保险）
            dln = dln.sel(
                latitude=cls.latitude, longitude=cls.longitude,
                depth=cls.depth, method='nearest',
            )
            deps = cls.depth.values
            fast = np.full(len(deps), np.nan)
            slow = np.full(len(deps), np.nan)
            c = cls.values
            d = dln.values
            for k in range(len(deps)):
                ck, dk = c[:, :, k], d[:, :, k]
                m_f = (ck == 1) & np.isfinite(dk)
                m_s = (ck == -1) & np.isfinite(dk)
                if m_f.sum() > 10:
                    fast[k] = float(dk[m_f].mean())
                if m_s.sum() > 10:
                    slow[k] = float(dk[m_s].mean())
            profiles[model] = {'depth': deps, 'fast': fast, 'slow': slow}

        if not profiles:
            self.logger.warning("  ⚠️ 无可用模型剖面，跳过 2-4-3")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 8.5))
        panels = [(ax1, 0, 410, 'Crust + Lithosphere (0–410 km)'),
                  (ax2, 410, 1000, 'Transition Zone + Lower Mantle (410–1000 km)')]
        for ax, z0, z1, title in panels:
            for model, p in profiles.items():
                color = self.config.model_colors.get(model, '0.3')
                short = model.split('_', 1)[1]
                m = (p['depth'] >= z0) & (p['depth'] <= z1)
                ax.plot(
                    p['fast'][m] * 100, p['depth'][m],
                    color=color, lw=2, ls='-', label=f'{short} fast',
                )
                ax.plot(
                    p['slow'][m] * 100, p['depth'][m],
                    color=color, lw=2, ls='--', label=f'{short} slow',
                )
            ax.axvline(0, color='0.5', lw=0.8)
            for z in (410, 660):
                if z0 < z < z1:
                    ax.axhline(z, color='0.8', lw=0.8, ls=':')
            ax.set_ylim(z1, z0)
            ax.set_xlabel(r'Mean $\delta\ln V_S$ within class (%)', fontsize=12)
            ax.set_ylabel('Depth (km)', fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3, linestyle=':')
            ax.tick_params(labelsize=11)
        ax1.legend(fontsize=10, loc='lower left', framealpha=0.9)

        fig.suptitle(
            'Mean Velocity Profiles of Fast / Slow Facies Classes\n'
            'solid = fast class, dashed = slow class '
            '(cf. Lekić et al., 2012, Fig. 5)',
            fontsize=15, fontweight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        out = None
        for fmt in viz['figure_format']:
            p = self.output_dir / f'2-4-3_class_mean_profiles.{fmt}'
            fig.savefig(p, dpi=viz['dpi'])
            if fmt == 'jpg':
                out = p
        plt.close(fig)
        self.logger.info(f"  ✅ 保存: {out}")
        return out

    # ------------------------------------------------------------------
    # 2-4-4 分歧归因（少数派模型）
    # ------------------------------------------------------------------

    def plot_dissent_maps(self, ds: xr.Dataset) -> Optional[Path]:
        """
        分歧归因图（2-4-4）：2:1 分歧体素上标出"少数派"是哪个模型。

        定义：三模型中恰有两个分类相同、第三个不同（含中性）时，
        该第三个模型记为少数派；三者互不相同记为 'all differ'。
        若分歧系统性集中于某一模型（如分辨率最粗者），则属于方法学
        差异而非结构不确定，可为 Phase E 融合权重提供依据。
        """
        if getattr(self, '_classes_common', None) is None:
            self.logger.warning("  ⚠️ 无缓存分类体，请先运行 run_voting()")
            return None
        self.logger.info("🖌️ 绘制分歧归因图...")
        viz = self.config.visualization
        classes = self._classes_common  # (3, nlat, nlon, ndep)
        c0, c1, c2 = classes[0], classes[1], classes[2]
        valid = np.isfinite(c0) & np.isfinite(c1) & np.isfinite(c2)

        # 编码: -2=无效, -1=一致, 0/1/2=少数派模型 idx, 3=三者互异
        dissent = np.full(c0.shape, -2, dtype=np.int8)
        eq01, eq02, eq12 = (c0 == c1), (c0 == c2), (c1 == c2)
        dissent[valid & eq01 & eq02] = -1                     # 全一致
        dissent[valid & eq12 & ~eq01] = 0                     # 模型0 少数派
        dissent[valid & eq02 & ~eq01] = 1                     # 模型1 少数派
        dissent[valid & eq01 & ~eq02] = 2                     # 模型2 少数派
        dissent[valid & ~eq01 & ~eq02 & ~eq12] = 3            # 三者互异

        m_colors = [
            self.config.model_colors.get(m, '0.3') for m in self.config.models
        ]
        cmap = ListedColormap(['#f2f2f2'] + m_colors + ['#636363'])
        cmap.set_bad('white')
        norm = BoundaryNorm(np.arange(-1.5, 4.5, 1.0), cmap.N)
        plot_arr = np.where(dissent == -2, np.nan, dissent).astype(np.float32)

        lats, lons = ds.latitude.values, ds.longitude.values
        deps = ds.depth.values
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(17, 10))
        gs = plt.GridSpec(
            2, 3, figure=fig, width_ratios=[1, 1, 0.75],
            hspace=0.24, wspace=0.14,
            left=0.05, right=0.97, top=0.87, bottom=0.13,
        )

        im = None
        for i, dkm in enumerate(self.config.dissent_depths_km[:4]):
            ax = fig.add_subplot(gs[i // 2, i % 2], projection=proj)
            k = int(np.argmin(np.abs(deps - dkm)))
            im = ax.pcolormesh(
                lons, lats, np.ma.masked_invalid(plot_arr[:, :, k]),
                cmap=cmap, norm=norm, transform=proj, shading='nearest',
            )
            ax.coastlines(resolution='50m', linewidth=0.7, color='0.25')
            ax.set_extent(
                [lons.min(), lons.max(), lats.min(), lats.max()], crs=proj
            )
            gl = ax.gridlines(
                draw_labels=True, linewidth=0.3, color='0.7',
                alpha=0.5, linestyle=':',
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 9}
            gl.ylabel_style = {'size': 9}
            ax.set_title(
                f'{deps[k]:.0f} km', fontsize=14, fontweight='bold'
            )

        # 右侧：各模型少数派占比随深度
        ax_prof = fig.add_subplot(gs[:, 2])
        n_val = valid.sum(axis=(0, 1)).astype(float)
        n_val[n_val == 0] = np.nan
        for idx, model in enumerate(self.config.models):
            frac = (dissent == idx).sum(axis=(0, 1)) / n_val * 100
            ax_prof.plot(
                frac, deps, color=m_colors[idx], lw=2,
                label=model.split('_', 1)[1],
            )
        frac_ad = (dissent == 3).sum(axis=(0, 1)) / n_val * 100
        ax_prof.plot(frac_ad, deps, color='#636363', lw=1.5, ls='--',
                     label='All differ')
        for z in (410, 660):
            ax_prof.axhline(z, color='0.85', lw=0.8, ls=':')
        ax_prof.set_ylim(float(deps.max()), 0)
        ax_prof.set_xlabel('Dissenting-voxel fraction (%)', fontsize=11)
        ax_prof.set_ylabel('Depth (km)', fontsize=11)
        ax_prof.set_title(
            'Who disagrees, vs depth', fontsize=13, fontweight='bold'
        )
        ax_prof.legend(fontsize=10, loc='lower right', framealpha=0.9)
        ax_prof.grid(alpha=0.3, linestyle=':')

        cax = fig.add_axes([0.10, 0.05, 0.45, 0.018])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_ticks(range(-1, 4))
        cbar.set_ticklabels(
            ['Consensus']
            + [m.split('_', 1)[1] for m in self.config.models]
            + ['All differ']
        )
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Dissenting model (odd-one-out in 2:1 splits)',
                       fontsize=12)

        fig.suptitle(
            'Disagreement Attribution — Which Model Is the Odd One Out?\n'
            'colored = that model classifies differently from the other two',
            fontsize=16, fontweight='bold', y=0.97,
        )

        out = None
        for fmt in viz['figure_format']:
            p = self.output_dir / f'2-4-4_dissent_attribution.{fmt}'
            fig.savefig(p, dpi=viz['dpi'])
            if fmt == 'jpg':
                out = p
        plt.close(fig)
        self.logger.info(f"  ✅ 保存: {out}")
        return out

    def run(self) -> Dict[str, Any]:
        """完整流程：投票 + 统计 + 可视化"""
        ds = self.run_voting()
        prof = self._agreement_profile(ds)
        prof.to_csv(self.output_dir / 'agreement_profile.csv', index=False)
        fig_path = self.plot_voting_maps(ds)
        fig_sections = self.plot_voting_sections(ds)
        fig_profiles = self.plot_class_mean_profiles()
        fig_dissent = self.plot_dissent_maps(ds)

        summary = {
            'models': self.config.models,
            'scheme': self.config.clustering_scheme,
            'class_thresholds': self.config.class_thresholds,
            'grid': {
                'nlat': int(ds.sizes['latitude']),
                'nlon': int(ds.sizes['longitude']),
                'ndepth': int(ds.sizes['depth']),
            },
            'mean_frac_all_agree': float(
                (prof['frac_all_fast'] + prof['frac_all_slow']).mean()
            ),
            'figure': str(fig_path),
            'figure_sections': str(fig_sections),
            'figure_class_profiles': str(fig_profiles),
            'figure_dissent': str(fig_dissent),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(self.output_dir / 'voting_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary


def main():
    """主函数"""
    print("🎯 EASTASIA-FWI 跨模型速度相投票分析")
    print("=" * 60)

    try:
        analyzer = FaciesVotingAnalyzer()
        summary = analyzer.run()

        print("\n✅ 投票分析完成!")
        print(f"📁 输出目录: {analyzer.output_dir}")
        print(f"🗺️ 主图: {summary['figure']}")
        print(
            f"📊 全深度平均全体一致率: "
            f"{summary['mean_frac_all_agree'] * 100:.1f}%"
        )

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断处理")
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)


if __name__ == "__main__":
    main()
