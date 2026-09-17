"""
5_12_Model_compare.py:
模型对比可视化（自 2_1_Model_compare.py 拆出）
================================================================

功能描述:
----------
承担 2_1 的全部出图。地图类产品用 PyGMT（grdimage + coast，对齐 5_8），
1D 剖面仍用 Matplotlib。输出目录与文件名前缀保持 2-1_，写入 figures/model_compare。

产品:
- 论文图: 2-1_paper_dlnv_maps / paper_vs_maps / paper_vp_maps
- 水平切片: 绝对速度、两两差异、绝对值/扰动并列
- 垂直剖面、切片位置示意图
- 1D 平均剖面与径向各向异性

作者: EASTASIA-FWI Team
日期: 2026-09-16
版本: v1.0
"""

from __future__ import annotations

import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygmt
import xarray as xr
from matplotlib.figure import Figure

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from config.base_config import BaseConfig  # noqa: E402
from utils.file_utils import extract_module_prefix  # noqa: E402


class ModelCompareVizConfig:
    """2_1 可视化配置：PyGMT 地图 + Matplotlib 剖面"""

    def __init__(self) -> None:
        self.figure_prefix = '2-1_'
        self.save_formats = ['jpg', 'pdf']
        self.dpi = 300

        self.pygmt = {
            'MAP_FRAME_TYPE': 'plain',
            'MAP_FRAME_PEN': '0.6p,black',
            'FONT_ANNOT_PRIMARY': '8p,Helvetica,black',
            'FONT_LABEL': '9p,Helvetica,black',
            'FONT_TITLE': '10p,Helvetica-Bold,black',
            'MAP_TITLE_OFFSET': '4p',
            'MAP_TICK_LENGTH_PRIMARY': '3p',
        }
        self.coast = {
            'shorelines': '0.25p,gray20',
            'borders': ['1/0.2p,gray40'],
            'resolution': 'i',
        }
        self.axes = {'xaxis': 'xa20f10', 'yaxis': 'ya15f5'}

        self.paper = {
            'depths': [100.0, 200.0, 500.0, 800.0],
            # X = 线性经纬度，子图尺寸按宽高显式给定，避免 Q 投影把行距撑开
            'panel_width': '4.70c',
            'panel_w': 4.70,
            'panel_h': 3.05,
            'margin_x': 0.18,
            'margin_y': 0.12,
            'cbar_gap': 1.20,
            'dlnv_cmap': 'seis',
            'abs_cmap': 'seis',
            'abs_reverse': False,
            'dlnv_percentile': 98.0,
            'abs_percentile': (2.0, 98.0),
            'cbar_width': '3.50c',
            'cbar_height': '0.14c',
            'cbar_offset': '0.98c',
        }

        self.logging = {'level': 'INFO'}


class ModelCompareVisualizer:
    """
    2_1 模型对比可视化器。

    从 ModelComparator 取已加载模型与模块配置，不重复读 NetCDF。
    """

    def __init__(self, comparator: Any) -> None:
        self.cmp = comparator
        self.base_config = comparator.base_config
        self.cmp_cfg = comparator.config
        self.config = ModelCompareVizConfig()
        self.logger = comparator.logger
        self.output_dir = comparator.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.module_prefix = extract_module_prefix(Path(__file__), default='5-12')
        pygmt.config(**self.config.pygmt)

        paper_cfg = getattr(comparator.config, 'paper_figure', {}) or {}
        if paper_cfg.get('depths'):
            self.config.paper['depths'] = [float(d) for d in paper_cfg['depths']]
        self._apply_mpl_style()

        r = self._common_region()
        self.region = [r[0], r[1], r[2], r[3]]
        self.logger.info("🗺️ 2_1 可视化已拆至 5_12（地图 PyGMT）")

    def _apply_mpl_style(self) -> None:
        """1D 剖面沿用 2_1 的字体与字号。"""
        fonts = self.cmp_cfg.plot_params
        plt.rcParams['font.sans-serif'] = fonts.get('font_family', ['Arial', 'DejaVu Sans'])
        plt.rcParams['axes.unicode_minus'] = False
        sizes = fonts.get('font_sizes', {})
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': sizes.get('tick', 15),
            'axes.titlesize': sizes.get('title', 20),
            'axes.labelsize': sizes.get('xlabel', 16),
            'legend.fontsize': sizes.get('legend', 15),
            'figure.dpi': fonts.get('dpi', 300),
        })

    # ------------------------------------------------------------------ #
    # 调度
    # ------------------------------------------------------------------ #
    def plot_all(self) -> None:
        """对应原 2_1.save_results 的全部出图步骤。"""
        prefix = self.config.figure_prefix
        mode = self.cmp_cfg.plot_mode

        self.logger.info("\n📊 1. 综合速度剖面对比（VS + VP）...")
        fig = self.plot_1d_velocity_comparison()
        self._save_mpl(fig, f'{prefix}velocity_comparison')
        plt.close(fig)

        self.logger.info("\n📊 2. S波各向异性对比...")
        fig = self.plot_s_wave_anisotropy_comparison()
        if fig is not None:
            self._save_mpl(fig, f'{prefix}s_wave_anisotropy')
            plt.close(fig)

        self.logger.info("\n📊 3. P波各向异性对比...")
        fig = self.plot_p_wave_anisotropy_comparison()
        if fig is not None:
            self._save_mpl(fig, f'{prefix}p_wave_anisotropy')
            plt.close(fig)

        self.logger.info("\n📊 4. 单个模型详细剖面...")
        for key, model in self.cmp.models.items():
            fig = self.plot_individual_model_profile(key)
            if fig is None:
                continue
            safe = model.metadata.name.replace(' ', '_').replace('.', '_')
            self._save_mpl(fig, f'{prefix}profile_{safe}')
            plt.close(fig)

        self.logger.info("\n🗺️  5. 切片位置示意图...")
        self.plot_slice_location_map()

        if mode.get('enable_absolute_only', True):
            self.logger.info("\n🆕 6. 独立绝对速度对比图（VS）...")
            self.plot_horizontal_row_maps('vs', field='abs', stem='absolute_vs')

        if mode.get('enable_model_differences', True):
            self.logger.info("\n🆕 7. 两两模型差异对比图（VS）...")
            self.plot_difference_maps('vs')

        h_mode = str(mode.get('horizontal_slices', 'both'))
        self.logger.info(f"\n🗺️  8. 水平切片对比（VS，模式: {h_mode}）...")
        self.plot_horizontal_comparison('vs', h_mode)

        v_mode = str(mode.get('vertical_profiles', 'perturbation'))
        self.logger.info(f"\n📐 9. 垂直剖面对比（VS，模式: {v_mode}）...")
        self.plot_vertical_profiles('vs', v_mode)

        depths = self.config.paper['depths']
        self.logger.info(f"\n📄 10. 论文主图（深度 {depths} km）...")
        self.plot_paper_maps('dlnv', 'vs', 'paper_dlnv_maps')
        self.plot_paper_maps('abs', 'vs', 'paper_vs_maps')
        self.plot_paper_maps('abs', 'vp', 'paper_vp_maps')

    # ------------------------------------------------------------------ #
    # 论文图：模型 × 深度
    # ------------------------------------------------------------------ #
    def plot_paper_maps(self, field: str, param: str, stem: str) -> Optional[str]:
        """
        3 模型 × 4 深度论文图。

        绝对速度与扰动都用 GMT seis（红=低速，蓝=高速）；
        每列按该深度自身振幅定范围，列内三模型共用。版式按发表图处理。
        """
        models = [
            (k, m) for k, m in self.cmp.models.items() if m.has_parameter(param)
        ]
        if not models:
            self.logger.warning(f"  ⚠️ 无模型含 {param}，跳过 {stem}")
            return None

        depths = list(self.config.paper['depths'])
        n_rows, n_cols = len(models), len(depths)
        grids: List[List[Optional[xr.DataArray]]] = []
        actual: List[float] = []
        col_vals: List[List[np.ndarray]] = [[] for _ in depths]

        for col, depth in enumerate(depths):
            z_used: List[float] = []
            row_grids: List[Optional[xr.DataArray]] = []
            for _, model in models:
                grid, z = self._extract_grid(model, param, depth, field)
                row_grids.append(grid)
                z_used.append(z)
                if grid is not None:
                    valid = grid.values[np.isfinite(grid.values)]
                    if valid.size:
                        col_vals[col].append(valid)
            grids.append(row_grids)
            actual.append(float(np.median(z_used)))

        grids_rc = [[grids[c][r] for c in range(n_cols)] for r in range(n_rows)]
        series = [self._column_series(vals, field) for vals in col_vals]
        self.logger.info(
            f"  {stem}: "
            + ', '.join(f'{z:.0f} km {s}' for z, s in zip(actual, series))
        )

        letters = 'abcdefghijklmnopqrstuvwxyz'
        paper = self.config.paper
        pygmt.config(
            MAP_FRAME_TYPE='plain',
            MAP_FRAME_PEN='0.4p,black',
            FONT_ANNOT_PRIMARY='6.5p,Helvetica,black',
            FONT_LABEL='7.5p,Helvetica,black',
            MAP_TICK_LENGTH_PRIMARY='1.8p',
            MAP_ANNOT_OFFSET_PRIMARY='2p',
            FORMAT_GEO_MAP='dddF',
        )
        pw = float(paper['panel_w'])
        ph = float(paper['panel_h'])
        mx = float(paper['margin_x'])
        my = float(paper['margin_y'])
        fig_w = n_cols * pw + (n_cols - 1) * mx
        fig_h = n_rows * ph + (n_rows - 1) * my + float(paper['cbar_gap'])
        proj = f'X{pw:.2f}c/{ph:.2f}c'
        cbar_label = 'dlnVs (%)' if field == 'dlnv' else f'{param.capitalize()} (km/s)'
        if field == 'dlnv' and param.lower() != 'vs':
            cbar_label = f'dln{param.capitalize()} (%)'

        fig = pygmt.Figure()
        with fig.subplot(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(f'{fig_w:.2f}c', f'{fig_h:.2f}c'),
            margins=[f'{mx:.2f}c', f'{my:.2f}c'],
        ):
            idx = 0
            for row, (_, model) in enumerate(models):
                for col in range(n_cols):
                    with fig.set_panel(panel=idx):
                        self._apply_paper_cpt(field, series[col])
                        self._draw_paper_map(
                            fig, proj, grids_rc[row][col],
                            left=(col == 0),
                            bottom=(row == n_rows - 1),
                        )
                        letter = letters[idx]
                        tag = (
                            f'({letter}) {model.metadata.name}'
                            if col == 0 else f'({letter})'
                        )
                        fig.text(
                            position='TL',
                            offset='0.10c/-0.10c',
                            text=tag,
                            font='8p,Helvetica-Bold,black',
                            fill='white@10',
                            clearance='0.04c/0.03c+tO',
                            justify='TL',
                        )
                        if row == 0:
                            fig.text(
                                position='TR',
                                offset='-0.10c/-0.10c',
                                text=f'{actual[col]:.0f} km',
                                font='8p,Helvetica-Bold,black',
                                fill='white@10',
                                clearance='0.04c/0.03c+tO',
                                justify='TR',
                            )
                        if row == n_rows - 1:
                            fig.colorbar(
                                position=(
                                    f"JBC+o0c/{paper['cbar_offset']}"
                                    f"+w{paper['cbar_width']}/{paper['cbar_height']}+h+e"
                                ),
                                frame=self._cbar_frame(series[col], cbar_label),
                            )
                    idx += 1

        return self._save_gmt(fig, f'{self.config.figure_prefix}{stem}')

    def _apply_paper_cpt(self, field: str, series: str) -> None:
        """论文图共用同一套 CPT 参数，只改 series。"""
        paper = self.config.paper
        pygmt.makecpt(
            cmap=paper['dlnv_cmap'] if field == 'dlnv' else paper['abs_cmap'],
            series=series,
            continuous=True,
            reverse=False,
            background=True,
        )

    def _draw_paper_map(
        self,
        fig: pygmt.Figure,
        projection: str,
        grid: Optional[xr.DataArray],
        left: bool,
        bottom: bool,
    ) -> None:
        """论文子图：无 +t 标题，避免 GMT 把 km 当单位。"""
        wesn = f"{'W' if left else 'w'}{'S' if bottom else 's'}en"
        fig.basemap(
            region=self.region,
            projection=projection,
            frame=['xa20f10', 'ya15f5', wesn],
        )
        if grid is not None:
            fig.grdimage(
                grid=grid,
                region=self.region,
                projection=projection,
                cmap=True,
                nan_transparent=True,
            )
        fig.coast(
            region=self.region,
            projection=projection,
            shorelines='0.2p,gray25',
            borders=['1/0.15p,gray40'],
            resolution=self.config.coast['resolution'],
        )

    # ------------------------------------------------------------------ #
    # 水平切片（按深度一张）
    # ------------------------------------------------------------------ #
    def plot_horizontal_row_maps(
        self, param: str, field: str, stem: str
    ) -> None:
        """每个深度一行：各模型并排，共享色标。"""
        depths = list(self.cmp_cfg.slice_params['horizontal_depths'])
        models = [
            (k, m) for k, m in self.cmp.models.items() if m.has_parameter(param)
        ]
        for depth in depths:
            items: List[Tuple[str, xr.DataArray]] = []
            vals: List[np.ndarray] = []
            for _, model in models:
                grid, _ = self._extract_grid(model, param, depth, field)
                if grid is None:
                    continue
                items.append((model.metadata.name, grid))
                valid = grid.values[np.isfinite(grid.values)]
                if valid.size:
                    vals.append(valid)
            if not items:
                continue
            series = self._column_series(vals, field)
            self._plot_named_row(
                items, series, field, param,
                title=self._gmt_text(f'{field.upper()} {param.upper()} {depth:.0f}km'),
                stem=f'{self.config.figure_prefix}{stem}_{int(depth)}km',
            )

    def plot_difference_maps(self, param: str) -> None:
        """两两模型 ΔV 水平切片。"""
        depths = list(self.cmp_cfg.slice_params['horizontal_depths'])
        keys = [k for k, m in self.cmp.models.items() if m.has_parameter(param)]
        pairs = list(combinations(keys, 2))
        if not pairs:
            return
        for depth in depths:
            items: List[Tuple[str, xr.DataArray]] = []
            absvals: List[np.ndarray] = []
            for k1, k2 in pairs:
                g1, _ = self._extract_grid(self.cmp.models[k1], param, depth, 'abs')
                g2, _ = self._extract_grid(self.cmp.models[k2], param, depth, 'abs')
                if g1 is None or g2 is None:
                    continue
                diff = self._difference_grid(g1, g2)
                if diff is None:
                    continue
                n1 = self.cmp.models[k1].metadata.name
                n2 = self.cmp.models[k2].metadata.name
                items.append((f'{n1}-{n2}', diff))
                valid = diff.values[np.isfinite(diff.values)]
                if valid.size:
                    absvals.append(np.abs(valid))
            if not items:
                continue
            lim = self._round_dlnv_limit(absvals) if absvals else 0.3
            series = f'{-lim:.1f}/{lim:.1f}/{(lim / 5.0):.2f}'
            self._plot_named_row(
                items, series, 'diff', param,
                title=self._gmt_text(f'd{param.upper()} {depth:.0f}km'),
                stem=f'{self.config.figure_prefix}difference_{param}_{int(depth)}km',
                cmap='polar',
            )

    def plot_horizontal_comparison(self, param: str, mode: str) -> None:
        """绝对值 / 扰动 / 两者并列。"""
        depths = list(self.cmp_cfg.slice_params['horizontal_depths'])
        models = [
            (k, m) for k, m in self.cmp.models.items() if m.has_parameter(param)
        ]
        fields = {
            'absolute': ['abs'],
            'perturbation': ['dlnv'],
            'both': ['abs', 'dlnv'],
        }.get(mode, ['abs', 'dlnv'])

        for depth in depths:
            n_rows, n_cols = len(fields), len(models)
            fig = pygmt.Figure()
            proj = f"M{self.config.paper['panel_width']}"
            with fig.subplot(
                nrows=n_rows, ncols=n_cols,
                figsize=('22c', f'{5.6 * n_rows:.1f}c'),
                margins=['0.2c', '0.4c'],
                title=self._gmt_text(f'{param.upper()} {depth:.0f}km'),
            ):
                idx = 0
                for field in fields:
                    vals = []
                    grids: List[Optional[xr.DataArray]] = []
                    names: List[str] = []
                    for _, model in models:
                        grid, _ = self._extract_grid(model, param, depth, field)
                        grids.append(grid)
                        names.append(model.metadata.name)
                        if grid is not None:
                            v = grid.values[np.isfinite(grid.values)]
                            if v.size:
                                vals.append(v)
                    series = self._column_series(vals, field)
                    for col, grid in enumerate(grids):
                        with fig.set_panel(panel=idx):
                            pygmt.makecpt(
                                cmap='seis',
                                series=series,
                                continuous=True,
                                reverse=False,
                                background=True,
                            )
                            tag = field if n_rows > 1 else ''
                            title = self._gmt_text(f'{names[col]} {tag}'.strip())
                            self._draw_map_panel(
                                fig, proj, grid, title,
                                left=(col == 0),
                                bottom=(idx // n_cols == n_rows - 1),
                            )
                            if col == n_cols - 1:
                                lab = (
                                    f'dln{param.upper()} (%)' if field == 'dlnv'
                                    else f'{param.upper()} (km/s)'
                                )
                                fig.colorbar(
                                    position='JMR+o0.25c/0c+w4.2c/0.25c',
                                    frame=self._cbar_frame(series, lab),
                                )
                        idx += 1
            self._save_gmt(
                fig,
                f'{self.config.figure_prefix}horizontal_slice_{param}_{int(depth)}km',
            )

    # ------------------------------------------------------------------ #
    # 垂直剖面与位置图
    # ------------------------------------------------------------------ #
    def plot_vertical_profiles(self, param: str, mode: str) -> None:
        """经度 / 纬度垂直剖面，PyGMT grdimage。"""
        fields = {
            'absolute': ['abs'],
            'perturbation': ['dlnv'],
            'both': ['abs', 'dlnv'],
        }.get(mode, ['dlnv'])
        lons = list(self.cmp_cfg.slice_params['vertical_profiles']['longitude'])
        lats = list(self.cmp_cfg.slice_params['vertical_profiles']['latitude'])
        fig_i = 0
        for direction, positions in (('longitude', lons), ('latitude', lats)):
            for pos in positions:
                fig_i += 1
                self._plot_one_vertical(
                    param, direction, float(pos), fields,
                    f'{self.config.figure_prefix}vertical_profile_{fig_i}',
                )

    def _plot_one_vertical(
        self,
        param: str,
        direction: str,
        position: float,
        fields: Sequence[str],
        stem: str,
    ) -> None:
        models = [
            (k, m) for k, m in self.cmp.models.items() if m.has_parameter(param)
        ]
        n_rows, n_cols = len(fields), len(models)
        fig = pygmt.Figure()
        tag = f'Lon={position:.0f}' if direction == 'longitude' else f'Lat={position:.0f}'
        with fig.subplot(
            nrows=n_rows, ncols=n_cols,
            figsize=(f'{5.5 * n_cols:.1f}c', f'{6.2 * n_rows:.1f}c'),
            margins=['0.25c', '0.35c'],
            title=self._gmt_text(f'{param.upper()} {tag}'),
        ):
            idx = 0
            for field in fields:
                packed: List[Optional[xr.DataArray]] = []
                vals: List[np.ndarray] = []
                names: List[str] = []
                x_min, x_max = np.inf, -np.inf
                z_max = 0.0
                for _, model in models:
                    grid = self._extract_profile_grid(model, param, direction, position, field)
                    packed.append(grid)
                    names.append(model.metadata.name)
                    if grid is not None:
                        v = grid.values[np.isfinite(grid.values)]
                        if v.size:
                            vals.append(v)
                        x_min = min(x_min, float(grid['x'].min()))
                        x_max = max(x_max, float(grid['x'].max()))
                        z_max = max(z_max, float(grid['depth'].max()))
                series = self._column_series(vals, field)
                region = [x_min, x_max, 0.0, max(z_max, 1000.0)]
                for col, grid in enumerate(packed):
                    with fig.set_panel(panel=idx):
                        pygmt.makecpt(
                            cmap='seis',
                            series=series,
                            continuous=True,
                            reverse=False,
                            background=True,
                        )
                        fig.basemap(
                            region=region,
                            projection='X6c/-5.2c',
                            frame=[
                                'xa20f10+lLongitude (deg)' if direction == 'latitude'
                                else 'xa10f5+lLatitude (deg)',
                                'ya200f100+lDepth (km)',
                                f'WSne+t{self._gmt_text(names[col])}',
                            ],
                        )
                        if grid is not None:
                            fig.grdimage(
                                grid=grid, region=region, projection='X6c/-5.2c',
                                cmap=True, nan_transparent=True,
                            )
                        if col == n_cols - 1:
                            lab = (
                                f'dln{param.upper()} (%)' if field == 'dlnv'
                                else f'{param.upper()} (km/s)'
                            )
                            fig.colorbar(
                                position='JMR+o0.25c/0c+w4.6c/0.25c',
                                frame=self._cbar_frame(series, lab),
                            )
                    idx += 1
        self._save_gmt(fig, stem)

    def plot_slice_location_map(self) -> Optional[str]:
        """水平切片深度说明 + 经纬向剖面位置。"""
        lon0, lon1, lat0, lat1 = self.region
        fig = pygmt.Figure()
        fig.basemap(
            region=[lon0 - 2, lon1 + 2, lat0 - 2, lat1 + 2],
            projection='M14c',
            frame=['xa20f10', 'ya15f5', f'WSne+t{self._gmt_text("Slice locations")}'],
        )
        fig.coast(
            shorelines=self.config.coast['shorelines'],
            borders=self.config.coast['borders'],
            resolution=self.config.coast['resolution'],
            land='247/247/242',
            water='230/240/250',
        )
        fig.plot(
            x=[lon0, lon1, lon1, lon0, lon0],
            y=[lat0, lat0, lat1, lat1, lat0],
            pen='1p,black,-',
            straight_line=True,
        )
        for lon in self.cmp_cfg.slice_params['vertical_profiles']['longitude']:
            fig.plot(x=[lon, lon], y=[lat0, lat1], pen='1.1p,red', straight_line=True)
            fig.text(x=lon, y=lat1 + 0.8, text=f'{lon:.0f}E',
                     font='8p,Helvetica-Bold,red')
        for lat in self.cmp_cfg.slice_params['vertical_profiles']['latitude']:
            fig.plot(x=[lon0, lon1], y=[lat, lat], pen='1.1p,blue', straight_line=True)
            fig.text(x=lon1 + 1.2, y=lat, text=f'{lat:.0f}N',
                     font='8p,Helvetica-Bold,blue')
        depths = '/'.join(
            f'{d:.0f}' for d in self.cmp_cfg.slice_params['horizontal_depths']
        )
        fig.text(
            x=(lon0 + lon1) / 2.0, y=lat0 - 1.6,
            text=self._gmt_text(f'H-slices: {depths}'),
            font='8p,Helvetica,black',
        )
        return self._save_gmt(fig, f'{self.config.figure_prefix}slice_locations')

    # ------------------------------------------------------------------ #
    # 1D 剖面（Matplotlib，逻辑对齐原 2_1）
    # ------------------------------------------------------------------ #
    def plot_1d_velocity_comparison(self) -> Figure:
        """共同区域 VS + VP 平均剖面。"""
        self.logger.info("🎨 绘制共同区域1D剖面对比图（standardized，VS + VP）...")
        fig, ax = plt.subplots(figsize=self.cmp_cfg.plot_params['figsize_1d'])
        depth_range = self.cmp_cfg.profile_params['depth_range']
        fonts = self.cmp_cfg.plot_params['font_sizes']
        vs_annotated = False
        vp_annotated = False

        for model_key, model in self.cmp.models.items():
            try:
                profile = model.calculate_1d_profile(
                    depth_range=depth_range, spatial_averaging=True
                )
                if 'depth' not in profile or len(profile['depth']) == 0:
                    continue
                depths = profile['depth']
                color = model.metadata.color
                model_plotted = False
                for target_param in ('vs', 'vp'):
                    if target_param not in profile:
                        continue
                    values = profile[target_param]
                    valid_mask = ~np.isnan(values['mean'])
                    if valid_mask.sum() == 0:
                        continue
                    ax.plot(
                        values['mean'][valid_mask], depths[valid_mask],
                        color=color, linestyle='-', linewidth=1.5,
                        label=model.metadata.full_name if not model_plotted else None,
                        alpha=0.9,
                    )
                    model_plotted = True
                    if target_param == 'vs' and not vs_annotated and valid_mask.sum() > 20:
                        idx = np.where(valid_mask)[0][valid_mask.sum() // 2]
                        ax.text(
                            values['mean'][idx] - 0.3, depths[idx], 'VS',
                            color='darkred', fontsize=16, va='center', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                      edgecolor='darkred', alpha=0.9, linewidth=2),
                        )
                        vs_annotated = True
                    elif target_param == 'vp' and not vp_annotated and valid_mask.sum() > 20:
                        idx = np.where(valid_mask)[0][valid_mask.sum() // 2]
                        ax.text(
                            values['mean'][idx] + 0.3, depths[idx], 'VP',
                            color='darkblue', fontsize=16, va='center', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                      edgecolor='darkblue', alpha=0.9, linewidth=2),
                        )
                        vp_annotated = True
                    if self.cmp_cfg.plot_params['show_uncertainty']:
                        std_mask = valid_mask & ~np.isnan(values['std'])
                        if std_mask.sum() > 0:
                            k = self.cmp_cfg.plot_params['uncertainty_multiplier']
                            ax.fill_betweenx(
                                depths[std_mask],
                                values['mean'][std_mask] - k * values['std'][std_mask],
                                values['mean'][std_mask] + k * values['std'][std_mask],
                                color=color,
                                alpha=self.cmp_cfg.plot_params['uncertainty_alpha'],
                                linewidth=0,
                            )
                if model_plotted:
                    self.logger.info(f"  ✅ {model.metadata.name} 绘制成功")
            except Exception as exc:  # noqa: BLE001
                self.logger.error(f"绘制失败 {model_key}: {exc}")

        ax.set_ylim(depth_range[1], 0)
        ax.set_xlim(1, 13)
        ax.set_xlabel('Velocity (km/s)', fontsize=fonts['xlabel'])
        ax.set_ylabel('Depth (km)', fontsize=fonts['ylabel'])
        ax.set_title('Velocity Profile Comparison (VS & VP)',
                     fontsize=fonts['title'], pad=20)
        for z0 in (60, 410, 660):
            if z0 <= depth_range[1]:
                ax.axhline(z0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.legend(
            loc='upper right', bbox_to_anchor=(0.98, 0.98),
            title='Velocity Models', frameon=True, fontsize=fonts['legend'],
        )
        ax.grid(True, linestyle=':', alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_s_wave_anisotropy_comparison(self) -> Optional[Figure]:
        return self._plot_anisotropy(
            ('vsv', 'vsh'),
            'S-wave Anisotropy Comparison (VSV vs VSH)',
            'S-wave Velocity (km/s)',
            checker='has_s_wave_anisotropy',
        )

    def plot_p_wave_anisotropy_comparison(self) -> Optional[Figure]:
        return self._plot_anisotropy(
            ('vpv', 'vph'),
            'P-wave Anisotropy Comparison (VPV vs VPH)',
            'P-wave Velocity (km/s)',
            checker='has_p_wave_anisotropy',
        )

    def _plot_anisotropy(
        self,
        params: Tuple[str, str],
        title: str,
        xlabel: str,
        checker: str,
    ) -> Optional[Figure]:
        keys = [k for k, m in self.cmp.models.items() if getattr(m, checker)()]
        if not keys:
            self.logger.warning(f"  ⚠️ 无模型含 {params}")
            return None
        fig, ax = plt.subplots(figsize=self.cmp_cfg.plot_params['figsize_1d'])
        depth_range = self.cmp_cfg.profile_params['depth_range']
        fonts = self.cmp_cfg.plot_params['font_sizes']
        styles = ['-', '--']
        for key in keys:
            model = self.cmp.models[key]
            try:
                profile = model.calculate_1d_profile(depth_range=depth_range)
                if 'depth' not in profile or len(profile['depth']) == 0:
                    continue
                z = profile['depth']
                for p, ls in zip(params, styles):
                    if p not in profile:
                        continue
                    v = profile[p]
                    m = ~np.isnan(v['mean'])
                    if not m.any():
                        continue
                    ax.plot(
                        v['mean'][m], z[m], color=model.metadata.color,
                        linestyle=ls, linewidth=1.5, alpha=0.9,
                        label=f'{model.metadata.full_name} {p.upper()}',
                    )
            except Exception as exc:  # noqa: BLE001
                self.logger.error(f"绘制各向异性失败 {key}: {exc}")
        ax.set_ylim(depth_range[1], 0)
        ax.set_xlabel(xlabel, fontsize=fonts['xlabel'])
        ax.set_ylabel('Depth (km)', fontsize=fonts['ylabel'])
        ax.set_title(title, fontsize=fonts['title'], pad=20)
        for z0 in (60, 410, 660):
            if z0 <= depth_range[1]:
                ax.axhline(z0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.legend(loc='upper right', fontsize=fonts['legend'] - 1, ncol=1)
        ax.grid(True, linestyle=':', alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_individual_model_profile(
        self, model_key: str, use_original_region: bool = True
    ) -> Optional[Figure]:
        """单模型完整剖面；默认 original 区域。"""
        if model_key not in self.cmp.models:
            self.logger.warning(f"模型 {model_key} 不存在")
            return None
        if (
            use_original_region
            and self.cmp_cfg.region_data_source.get('individual') == 'original'
        ):
            model = self.cmp._load_model_individual(model_key)
            region_title = 'Original Region'
        else:
            model = self.cmp.models[model_key]
            region_title = 'Common Region'
        if model is None:
            return None

        self.logger.info(f"🎨 绘制单个模型剖面: {model.metadata.full_name} ({region_title})")
        fig, ax = plt.subplots(figsize=self.cmp_cfg.plot_params['figsize_1d_individual'])
        depth_range = self.cmp_cfg.profile_params['depth_range']
        try:
            profile = model.calculate_1d_profile(depth_range=depth_range)
            if 'depth' not in profile or len(profile['depth']) == 0:
                self.logger.warning(f"模型 {model_key} 无有效数据")
                return None
            depths = profile['depth']
            param_styles = {
                'vp': {'color': '#E41A1C', 'linestyle': '-', 'linewidth': 3,
                       'label': 'VP', 'alpha': 1.0},
                'vs': {'color': '#377EB8', 'linestyle': '-', 'linewidth': 3,
                       'label': 'VS', 'alpha': 1.0},
                'vpv': {'color': '#FF7F00', 'linestyle': '-', 'linewidth': 2.5,
                        'label': 'VPV', 'alpha': 0.8},
                'vph': {'color': '#FF7F00', 'linestyle': '--', 'linewidth': 2.5,
                        'label': 'VPH', 'alpha': 0.8},
                'vsv': {'color': '#4DAF4A', 'linestyle': '-', 'linewidth': 2.5,
                        'label': 'VSV', 'alpha': 0.8},
                'vsh': {'color': '#4DAF4A', 'linestyle': '--', 'linewidth': 2.5,
                        'label': 'VSH', 'alpha': 0.8},
            }
            for param, style in param_styles.items():
                if param not in profile:
                    continue
                values = profile[param]
                valid_mask = ~np.isnan(values['mean'])
                if valid_mask.sum() == 0:
                    continue
                ax.plot(
                    values['mean'][valid_mask], depths[valid_mask],
                    color=style['color'], linestyle=style['linestyle'],
                    linewidth=style['linewidth'], label=style['label'],
                    alpha=style['alpha'],
                )
            ax.set_ylim(depth_range[1], 0)
            ax.set_xlabel('Velocity (km/s)', fontsize=14)
            ax.set_ylabel('Depth (km)', fontsize=14)
            ax.set_title(
                f'{model.metadata.full_name}\n1D Velocity Profile ({region_title})',
                fontsize=16, pad=20, fontweight='bold',
            )
            for z0 in (60, 410, 660):
                if z0 <= depth_range[1]:
                    ax.axhline(z0, color='gray', linestyle='--',
                               linewidth=1.5, alpha=0.5)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                iso_h = [h for h, lab in zip(handles, labels) if lab in ('VP', 'VS')]
                iso_l = [lab for lab in labels if lab in ('VP', 'VS')]
                if iso_h:
                    legend1 = ax.legend(
                        iso_h, iso_l, loc='upper right',
                        bbox_to_anchor=(0.98, 0.98), title='Isotropic',
                        frameon=True, fontsize=11,
                    )
                    ax.add_artist(legend1)
                an_h = [h for h, lab in zip(handles, labels) if lab not in ('VP', 'VS')]
                an_l = [lab for lab in labels if lab not in ('VP', 'VS')]
                if an_h:
                    ax.legend(
                        an_h, an_l, loc='upper right',
                        bbox_to_anchor=(0.98, 0.85), title='Anisotropic',
                        frameon=True, fontsize=11,
                    )
            ax.grid(True, linestyle=':', alpha=0.3)
            fig.tight_layout()
            return fig
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"绘制单个模型剖面失败 {model_key}: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # 网格与色标
    # ------------------------------------------------------------------ #
    def _extract_grid(
        self, model: Any, param: str, depth: float, field: str
    ) -> Tuple[Optional[xr.DataArray], float]:
        z_idx = int(np.argmin(np.abs(model.depth - depth)))
        actual = float(model.depth[z_idx])
        try:
            if field == 'dlnv':
                lon_g, lat_g, data = model.get_horizontal_slice_dlnv(param, depth)
            else:
                lon_g, lat_g, data = model.get_horizontal_slice(param, depth)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"  ⚠️ {model.metadata.name} {param}@{depth} km: {exc}")
            return None, actual
        grid = self._as_grid(lon_g, lat_g, data)
        return grid, actual

    def _extract_profile_grid(
        self, model: Any, param: str, direction: str,
        position: float, field: str,
    ) -> Optional[xr.DataArray]:
        try:
            if field == 'dlnv':
                xg, zg, data = model.get_vertical_profile_dlnv(param, direction, position)
            else:
                xg, zg, data = model.get_vertical_profile(param, direction, position)
        except Exception:  # noqa: BLE001
            return None
        x = np.asarray(xg[:, 0] if xg.ndim == 2 else xg, dtype=float)
        z = np.asarray(zg[0, :] if zg.ndim == 2 else model.depth, dtype=float)
        arr = np.asarray(data, dtype=float)
        if arr.shape == (len(z), len(x)):
            arr = arr.T
        if arr.shape != (len(x), len(z)):
            return None
        return xr.DataArray(
            arr.T, dims=['depth', 'x'],
            coords={'depth': z, 'x': x},
        )

    @staticmethod
    def _as_grid(
        lon_grid: np.ndarray, lat_grid: np.ndarray, data: np.ndarray
    ) -> Optional[xr.DataArray]:
        """保留原生坐标顺序，避免 unique+sort 把纬度翻转。"""
        lon_g = np.asarray(lon_grid, dtype=float)
        lat_g = np.asarray(lat_grid, dtype=float)
        z = np.asarray(data, dtype=float)
        if lon_g.ndim == 2:
            lon = lon_g[0, :]
            lat = lat_g[:, 0]
        else:
            lon = lon_g
            lat = lat_g
        if z.shape == (len(lon), len(lat)):
            z = z.T
        if z.shape != (len(lat), len(lon)):
            return None
        if lat.size >= 2 and lat[0] > lat[-1]:
            lat = lat[::-1]
            z = z[::-1, :]
        if lon.size >= 2 and lon[0] > lon[-1]:
            lon = lon[::-1]
            z = z[:, ::-1]
        return xr.DataArray(z, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon})

    def _difference_grid(
        self, a: xr.DataArray, b: xr.DataArray
    ) -> Optional[xr.DataArray]:
        try:
            if a.sizes.get('lon', 0) * a.sizes.get('lat', 0) >= b.sizes.get('lon', 0) * b.sizes.get('lat', 0):
                fine, coarse = a, b
                sign = 1.0
            else:
                fine, coarse = b, a
                sign = -1.0
            other = coarse.interp(lat=fine['lat'], lon=fine['lon'], method='linear')
            return (fine - other) * sign
        except Exception:  # noqa: BLE001
            return None

    def _column_series(self, values: List[np.ndarray], field: str) -> str:
        if not values:
            return '3.0/5.0/0.1' if field == 'abs' else '-5.0/5.0/1.0'
        flat = np.concatenate(values)
        if field == 'dlnv':
            lim = self._round_dlnv_limit([np.abs(flat)])
            step = 0.5 if lim >= 2.0 else 0.2
            return f'{-lim:.1f}/{lim:.1f}/{step:.1f}'
        if field == 'diff':
            lim = self._round_dlnv_limit([np.abs(flat)])
            return f'{-lim:.1f}/{lim:.1f}/{(max(lim / 5.0, 0.1)):.2f}'
        lo_p, hi_p = self.config.paper['abs_percentile']
        vmin = float(np.floor(np.percentile(flat, lo_p) * 10.0 + 1e-9) / 10.0)
        vmax = float(np.ceil(np.percentile(flat, hi_p) * 10.0 - 1e-9) / 10.0)
        if vmax <= vmin:
            vmax = vmin + 0.1
        step = 0.1 if (vmax - vmin) <= 1.2 else 0.2
        return f'{vmin:.1f}/{vmax:.1f}/{step:.1f}'

    def _round_dlnv_limit(self, abs_groups: List[np.ndarray]) -> float:
        pct = float(self.config.paper['dlnv_percentile'])
        p = float(np.percentile(np.concatenate(abs_groups), pct))
        return max(1.0, float(np.ceil(p * 2.0) / 2.0))

    @staticmethod
    def _cbar_frame(series: str, label: str) -> str:
        """色标主刻度约 6 档；绝对速度保持 1 位小数。"""
        parts = str(series).split('/')
        try:
            vmin, vmax, step = float(parts[0]), float(parts[1]), float(parts[2])
        except (IndexError, ValueError):
            return f'xaf+l{label}'
        span = max(vmax - vmin, step)
        annot = step
        while span / annot > 8.0:
            annot *= 2.0
        if annot >= 1.0:
            a_txt, f_txt = f'{annot:.1f}', f'{step:.1f}'
        else:
            a_txt, f_txt = f'{annot:.2f}'.rstrip('0').rstrip('.'), f'{step:.2f}'.rstrip('0').rstrip('.')
        if label:
            return f'xa{a_txt}f{f_txt}+l{label}'
        return f'xa{a_txt}f{f_txt}'

    @staticmethod
    def _gmt_text(text: str) -> str:
        """GMT 标题：保留空格，但把 `100 km` 粘成 `100km`，避免被解析成长度单位。"""
        cleaned = str(text).replace('°', '')
        return re.sub(r'(\d)\s*km\b', r'\1km', cleaned, flags=re.IGNORECASE)

    def _draw_map_panel(
        self,
        fig: pygmt.Figure,
        projection: str,
        grid: Optional[xr.DataArray],
        title: str,
        left: bool,
        bottom: bool,
    ) -> None:
        wesn = f"{'W' if left else 'w'}{'S' if bottom else 's'}en"
        fig.basemap(
            region=self.region,
            projection=projection,
            frame=[
                self.config.axes['xaxis'],
                self.config.axes['yaxis'],
                f'{wesn}+t{title}',
            ],
        )
        if grid is not None:
            fig.grdimage(
                grid=grid,
                region=self.region,
                projection=projection,
                cmap=True,
                nan_transparent=True,
            )
        fig.coast(
            region=self.region,
            projection=projection,
            shorelines=self.config.coast['shorelines'],
            borders=self.config.coast['borders'],
            resolution=self.config.coast['resolution'],
        )

    def _plot_named_row(
        self,
        items: List[Tuple[str, xr.DataArray]],
        series: str,
        field: str,
        param: str,
        title: str,
        stem: str,
        cmap: Optional[str] = None,
    ) -> None:
        n = len(items)
        fig = pygmt.Figure()
        proj = f"M{self.config.paper['panel_width']}"
        pygmt.makecpt(
            cmap=cmap or 'seis',
            series=series,
            continuous=True,
            reverse=False,
            background=True,
        )
        with fig.subplot(
            nrows=1, ncols=n,
            figsize=(f'{6.4 * n:.1f}c', '7.2c'),
            margins=['0.2c', '0.45c'],
            title=title,
        ):
            for i, (name, grid) in enumerate(items):
                with fig.set_panel(panel=i):
                    self._draw_map_panel(
                        fig, proj, grid, self._gmt_text(name),
                        left=(i == 0), bottom=True,
                    )
                    if i == n - 1:
                        lab = (
                            f'dln{param.upper()} (%)' if field == 'dlnv'
                            else (f'Δ{param.upper()} (km/s)' if field == 'diff'
                                  else f'{param.upper()} (km/s)')
                        )
                        fig.colorbar(
                            position='JMR+o0.3c/0c+w5c/0.28c',
                            frame=self._cbar_frame(series, lab),
                        )
        self._save_gmt(fig, stem)

    def _common_region(self) -> List[float]:
        lons, lons2, lats, lats2 = [], [], [], []
        for model in self.cmp.models.values():
            lons.append(float(np.min(model.lon)))
            lons2.append(float(np.max(model.lon)))
            lats.append(float(np.min(model.lat)))
            lats2.append(float(np.max(model.lat)))
        if not lons:
            return self.base_config.get_gmt_region('common')
        return [max(lons), min(lons2), max(lats), min(lats2)]

    def _save_gmt(self, fig: pygmt.Figure, stem: str) -> str:
        saved = ''
        for fmt in self.config.save_formats:
            fp = self.output_dir / f'{stem}.{fmt}'
            fig.savefig(str(fp), dpi=self.config.dpi, crop=True)
            saved = str(fp)
            self.logger.info(f"  ✅ {fp.name}")
        return saved

    def _save_mpl(self, fig: Figure, stem: str) -> None:
        for fmt in self.config.save_formats:
            fp = self.output_dir / f'{stem}.{fmt}'
            fig.savefig(fp, dpi=self.config.dpi, bbox_inches='tight', facecolor='white')
            self.logger.info(f"  ✅ {fp.name}")


def main() -> None:
    print("🗺️ EASTASIA-FWI 2_1 模型对比可视化（5_12 / PyGMT）")
    print("=" * 70)
    try:
        sys.path.insert(0, str(project_root / '2_Model_space_analysis'))
        import importlib.util
        cmp_py = project_root / '2_Model_space_analysis' / '2_1_Model_compare.py'
        spec = importlib.util.spec_from_file_location('model_compare_21', cmp_py)
        if spec is None or spec.loader is None:
            raise ImportError(cmp_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        comparator = mod.ModelComparator()
        comparator.load_models()
        viz = ModelCompareVisualizer(comparator)
        viz.plot_all()
        print(f"\n📁 输出: {viz.output_dir}")
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ {exc}")
        import traceback
        traceback.print_exc()
    print("=" * 70)


if __name__ == '__main__':
    main()
