"""
5_7_Moho_LAB.py:
中国大陆莫霍面与岩石圈底界（LAB）深度可视化模块
================================================================

功能描述:
----------
基于 PyGMT + matplotlib 绘制中国大陆 (东亚区域) 的莫霍面 (Moho) 与
岩石圈—软流圈边界 (LAB) 深度的散点分布图和统计图。
数据来源为过去 20 余年地震学研究 (接收函数 / 接收函数+面波联合反演 /
人工地震 等) 汇编整理结果，包含 1.7 万余个 Moho 深度点与近 4000 个
LAB 深度点。

核心功能:
----------
1. ✅ Moho 深度空间分布图（PyGMT）
   - 散点按深度着色
   - 可按方法类型 (1=接收函数, 2=接收函数+面波, 3=人工地震) 区分
2. ✅ LAB 深度空间分布图（PyGMT）
3. ✅ 深度统计直方图（matplotlib）
   - Moho 总体 + 按方法类型分布
   - LAB 总体分布
4. ✅ 与 5_1 / 5_2 一致的目录、命名与配置约定

输入数据:
----------
- data/莫霍面深度数据.txt : 经度 纬度 Moho深度(km) 类型(1/2/3)
- data/LAB深度数据.txt    : 经度 纬度 LAB深度(km)

输出文件:
----------
- 5-7_moho_depth_map.jpg/pdf
- 5-7_lab_depth_map.jpg/pdf
- 5-7_moho_lab_statistics.jpg/pdf

作者: EASTASIA-FWI Team
日期: 2026-05-12
版本: v1.0
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import pygmt
import matplotlib.pyplot as plt
import matplotlib as mpl

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import BaseConfig  # noqa: E402

# 可视化模块内 utils
sys.path.insert(0, str(Path(__file__).parent))
from utils.file_utils import extract_module_prefix, generate_filename  # noqa: E402


class MohoLABConfig:
    """Moho / LAB 深度可视化配置类"""

    def __init__(self):
        # PyGMT 全局配置（与 5_1/5_2 风格一致）
        self.pygmt = {
            'map_frame_type': "plain",
            'map_grid_pen_primary': "0.3p,dimgrey",
            'map_annot_oblique': "30",
            'map_annot_offset_primary': "5p",
            'map_annot_offset_secondary': "5p",
            'font_annot_primary': "10p,4",
            'font_label': "10p,28,black",
            'map_frame_width': "2p",
            'map_frame_pen': "0.5p",
            'map_tick_length_primary': "5p",
            'map_tick_pen_primary': "0.5p,black,-",
            'map_label_offset': "5p",
        }

        self.projection = {
            'type': "M",
            'width': "15c",
            'frame_interval': "a10f1",
        }

        # 数据源
        self.data_sources = {
            'elevation': "@earth_relief_01m",
        }

        # 颜色与符号
        self.style = {
            'topo_cmap': "geo",
            'moho_cmap': "haxby",      # 颜色由浅到深表征 Moho 深度
            'moho_series': "20/70/5",   # km
            'lab_cmap': "polar",        # LAB 深度色标
            'lab_series': "60/200/10",  # km
            'point_size': "0.10c",      # 散点尺寸
            'point_pen': "0.05p,black",
        }

        # 数据过滤范围（剔除明显异常值）
        self.filters = {
            'moho_depth_range': (10.0, 90.0),   # km
            'lab_depth_range': (30.0, 280.0),   # km
        }

        # Moho 方法类型标签
        self.moho_type_labels = {
            1: "接收函数 (RF)",
            2: "RF + 面波联合反演",
            3: "人工地震 (Active source)",
        }
        self.moho_type_labels_en = {
            1: "Receiver Function",
            2: "RF + Surface Wave",
            3: "Active Source",
        }

        # 色标条
        self.colorbar = {
            'moho_position': "JBC+o0c/0.8c+w8c/0.3c+h",
            'moho_frame': ["x+lMoho depth", "y+lkm"],
            'lab_position': "JBC+o0c/0.8c+w8c/0.3c+h",
            'lab_frame': ["x+lLAB depth", "y+lkm"],
        }

        # 输出
        self.output = {
            'dpi_jpg': 300,
            'dpi_pdf': 300,
            'crop': True,
            'save_jpg': True,
            'save_pdf': True,
        }

        # matplotlib 统计图配置
        self.statistics = {
            'figsize': (14, 9),
            'bins_moho': 60,
            'bins_lab': 50,
            'hist_color_total': '#1f77b4',
            'hist_colors_type': {
                1: '#d62728',
                2: '#2ca02c',
                3: '#ff7f0e',
            },
        }

        self.logging = {'level': 'INFO'}


class MohoLABPlotter:
    """Moho / LAB 深度专业绘图类"""

    def __init__(self,
                 config_file: Optional[Path] = None,
                 output_dir: Optional[str] = None,
                 moho_file: Optional[Path] = None,
                 lab_file: Optional[Path] = None,
                 csrm_moho_file: Optional[Path] = None):
        self.base_config = BaseConfig(config_file)
        self.config = MohoLABConfig()

        # 输出目录（与 5_1/5_2 一致：项目根 figures/）
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.base_config.dirs['figures']
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.MohoLABPlotter',
            self.config.logging['level'],
        )

        # 区域 & 投影
        self.region = [
            self.base_config.region['lon_min'],
            self.base_config.region['lon_max'],
            self.base_config.region['lat_min'],
            self.base_config.region['lat_max'],
        ]
        self.projection = f"{self.config.projection['type']}{self.config.projection['width']}"

        # 默认数据文件
        data_dir = self.base_config.dirs['data']
        self.moho_file = Path(moho_file) if moho_file else data_dir / '莫霍面深度数据.txt'
        self.lab_file = Path(lab_file) if lab_file else data_dir / 'LAB深度数据.txt'
        self.csrm_moho_file = (
            Path(csrm_moho_file) if csrm_moho_file
            else data_dir / 'CSRM1.0_Moho.mod'
        )

        # PyGMT 配置
        self._setup_pygmt_config()

        # 模块前缀
        self.module_prefix = extract_module_prefix(Path(__file__), default="5-7")

        # matplotlib 中文/字体配置
        mpl.rcParams['axes.unicode_minus'] = False
        for f in ['PingFang SC', 'Heiti SC', 'STHeiti', 'Songti SC',
                  'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']:
            try:
                mpl.rcParams['font.sans-serif'] = [f] + list(mpl.rcParams['font.sans-serif'])
                break
            except Exception:
                continue

        self.logger.info("🗻 Moho/LAB 深度绘图器初始化完成")
        self._print_config_summary()

    # ------------------------------------------------------------------ #
    # 配置 & 数据加载
    # ------------------------------------------------------------------ #
    def _setup_pygmt_config(self):
        c = self.config.pygmt
        pygmt.config(
            MAP_FRAME_TYPE=c['map_frame_type'],
            MAP_GRID_PEN_PRIMARY=c['map_grid_pen_primary'],
            MAP_ANNOT_OBLIQUE=c['map_annot_oblique'],
            MAP_ANNOT_OFFSET_PRIMARY=c['map_annot_offset_primary'],
            MAP_ANNOT_OFFSET_SECONDARY=c['map_annot_offset_secondary'],
            FONT_ANNOT_PRIMARY=c['font_annot_primary'],
            FONT_LABEL=c['font_label'],
            MAP_FRAME_WIDTH=c['map_frame_width'],
            MAP_FRAME_PEN=c['map_frame_pen'],
            MAP_TICK_LENGTH_PRIMARY=c['map_tick_length_primary'],
            MAP_TICK_PEN_PRIMARY=c['map_tick_pen_primary'],
            MAP_LABEL_OFFSET=c['map_label_offset'],
        )

    def _print_config_summary(self):
        print("\n📋 Moho/LAB 深度可视化配置摘要")
        print("-" * 60)
        print(f"研究区域: {self.base_config.region['name']}")
        r = self.base_config.region
        print(f"纬度范围: {r['lat_min']}° ~ {r['lat_max']}°")
        print(f"经度范围: {r['lon_min']}° ~ {r['lon_max']}°")
        print(f"Moho 文件: {self.moho_file}")
        print(f"LAB  文件: {self.lab_file}")
        print(f"输出目录: {self.output_dir}")
        print("-" * 60)

    def load_moho_data(self) -> pd.DataFrame:
        """加载 Moho 数据，跳过表头，列：lon lat depth type"""
        if not self.moho_file.exists():
            raise FileNotFoundError(f"Moho 数据文件不存在: {self.moho_file}")

        df = pd.read_csv(
            self.moho_file,
            sep=r'\s+',
            header=0,
            names=['longitude', 'latitude', 'depth', 'type'],
            engine='python',
            comment='#',
        )
        # 强制数值化（非数值的表头行已被 header=0 跳过；这里清理异常）
        for c in ['longitude', 'latitude', 'depth']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['type'] = pd.to_numeric(df['type'], errors='coerce').fillna(0).astype(int)

        n0 = len(df)
        # 深度范围过滤
        dmin, dmax = self.config.filters['moho_depth_range']
        df = df.dropna(subset=['longitude', 'latitude', 'depth'])
        df = df[(df['depth'] >= dmin) & (df['depth'] <= dmax)]
        # 地理范围过滤
        df = df[(df['longitude'] >= self.region[0]) & (df['longitude'] <= self.region[1])
                & (df['latitude'] >= self.region[2]) & (df['latitude'] <= self.region[3])]
        # 只保留类型 1/2/3
        df = df[df['type'].isin([1, 2, 3])].reset_index(drop=True)

        self.logger.info(f"✅ Moho 数据加载: 原始 {n0} → 过滤后 {len(df)} 条")
        for t in sorted(df['type'].unique()):
            label = self.config.moho_type_labels.get(int(t), f"type {t}")
            print(f"   类型 {int(t)} ({label}): {(df['type'] == t).sum()} 条")
        return df

    def load_lab_data(self) -> pd.DataFrame:
        """加载 LAB 数据，列：lon lat depth"""
        if not self.lab_file.exists():
            raise FileNotFoundError(f"LAB 数据文件不存在: {self.lab_file}")

        df = pd.read_csv(
            self.lab_file,
            sep=r'\s+',
            header=0,
            names=['longitude', 'latitude', 'depth'],
            engine='python',
            comment='#',
        )
        for c in ['longitude', 'latitude', 'depth']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        n0 = len(df)
        dmin, dmax = self.config.filters['lab_depth_range']
        df = df.dropna()
        df = df[(df['depth'] >= dmin) & (df['depth'] <= dmax)]
        df = df[(df['longitude'] >= self.region[0]) & (df['longitude'] <= self.region[1])
                & (df['latitude'] >= self.region[2]) & (df['latitude'] <= self.region[3])]
        df = df.reset_index(drop=True)

        self.logger.info(f"✅ LAB 数据加载: 原始 {n0} → 过滤后 {len(df)} 条")
        return df

    def load_csrm_moho(self) -> pd.DataFrame:
        """加载 CSRM1.0 Moho 模型文件（# lon lat moho(km) mohounc(km)）"""
        if not self.csrm_moho_file.exists():
            raise FileNotFoundError(
                f"CSRM Moho 文件不存在: {self.csrm_moho_file}"
            )
        df = pd.read_csv(
            self.csrm_moho_file,
            sep=r'\s+',
            header=0,
            names=['longitude', 'latitude', 'depth', 'uncertainty'],
            engine='python',
            comment='#',
            na_values=['nan', 'NaN', 'NAN'],
        )
        for c in ['longitude', 'latitude', 'depth', 'uncertainty']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        n0 = len(df)
        df = df.dropna(subset=['longitude', 'latitude', 'depth'])
        df = df[(df['longitude'] >= self.region[0])
                & (df['longitude'] <= self.region[1])
                & (df['latitude'] >= self.region[2])
                & (df['latitude'] <= self.region[3])]
        df = df.reset_index(drop=True)
        self.logger.info(
            f"✅ CSRM1.0 Moho 加载: 原始 {n0} → 有效 {len(df)} 节点 "
            f"(depth {df['depth'].min():.1f}–{df['depth'].max():.1f} km)"
        )
        return df

    def plot_csrm_moho_raw(self,
                           df: Optional[pd.DataFrame] = None,
                           save_file: Optional[str] = None) -> str:
        """绘制 CSRM1.0 Moho 模型原始网格点（混合 0.2°/0.4° 分辨率，散点显示）"""
        if df is None:
            df = self.load_csrm_moho()
        if df.empty:
            self.logger.warning("⚠️ CSRM Moho 数据为空，跳过")
            return ""

        if save_file is None:
            save_file = generate_filename(
                self.module_prefix, "moho_CSRM1.0_raw", "jpg")

        self.logger.info(
            f"🎨 [CSRM1.0 Moho raw] 节点数={len(df)}"
        )

        fig = pygmt.Figure()
        fig.basemap(region=self.region, projection=self.projection,
                    frame=["a10f5", "WSen"])
        pygmt.makecpt(cmap=self.config.style['moho_cmap'],
                      series=self.config.style['moho_series'],
                      continuous=True, reverse=True)
        fig.plot(
            x=df['longitude'].values,
            y=df['latitude'].values,
            fill=df['depth'].values,
            style="c0.06c",
            cmap=True,
        )
        fig.coast(region=self.region, projection=self.projection,
                  shorelines="0.4p,black",
                  borders=["1/0.4p,gray40"])
        fig.colorbar(position=self.config.colorbar['moho_position'],
                     frame=["x+lMoho depth (CSRM1.0 raw)", "y+lkm"])

        return self._save_figure(fig, save_file)

    def plot_csrm_moho_surface(self,
                               df: Optional[pd.DataFrame] = None,
                               spacing: float = 0.2,
                               tension: float = 0.25,
                               save_file: Optional[str] = None) -> str:
        """绘制 CSRM1.0 Moho 模型 surface 光滑图

        CSRM1.0 是 0.2°/0.4° 混合分辨率网格（密区 0.2°、稀区 0.4°），
        因此先 blockmean 到统一 spacing，再用 surface 做样条插值光滑，
        最后基于数据覆盖范围用 cKDTree 把空白区域遮成 NaN。
        """
        if df is None:
            df = self.load_csrm_moho()
        if df.empty:
            self.logger.warning("⚠️ CSRM Moho 数据为空，跳过")
            return ""

        if save_file is None:
            save_file = generate_filename(
                self.module_prefix, "moho_CSRM1.0_surface", "jpg")

        sub = df[['longitude', 'latitude', 'depth']]
        self.logger.info(
            f"🎨 [CSRM1.0 Moho] 节点数={len(sub)}, "
            f"surface 重采样 spacing={spacing}° tension={tension}"
        )

        blocked = pygmt.blockmean(
            data=sub.values, region=self.region,
            spacing=f"{spacing}/{spacing}",
        )
        grid = pygmt.surface(
            data=blocked, region=self.region,
            spacing=f"{spacing}/{spacing}", tension=tension,
        )

        # 用 cKDTree 遮掉无数据区域（CSRM 海域本就是 nan，这里再确保陆海边界干净）
        from scipy.spatial import cKDTree
        mask_radius = max(spacing * 3.0, 0.6)
        lons_g = grid['x'].values
        lats_g = grid['y'].values
        lon_mesh, lat_mesh = np.meshgrid(lons_g, lats_g)
        tree = cKDTree(sub[['longitude', 'latitude']].values)
        dist, _ = tree.query(
            np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()]), k=1)
        grid.values[dist.reshape(lon_mesh.shape) > mask_radius] = np.nan

        fig = pygmt.Figure()
        fig.basemap(region=self.region, projection=self.projection,
                    frame=["a10f5", "WSen"])
        pygmt.makecpt(cmap=self.config.style['moho_cmap'],
                      series=self.config.style['moho_series'],
                      continuous=True, reverse=True)
        fig.grdimage(grid=grid, projection=self.projection,
                     region=self.region, cmap=True, nan_transparent=True)
        fig.coast(region=self.region, projection=self.projection,
                  shorelines="0.4p,black",
                  borders=["1/0.4p,gray40"])
        fig.colorbar(position=self.config.colorbar['moho_position'],
                     frame=["x+lMoho depth (CSRM1.0 surface)", "y+lkm"])

        return self._save_figure(fig, save_file)

    # ------------------------------------------------------------------ #
    # 底图
    # ------------------------------------------------------------------ #
    def _draw_basemap(self, fig: pygmt.Figure):
        """绘制简洁底图（海岸线 + 国界 + 浅色陆地/水体），避免 earth_relief 缓存依赖"""
        fig.coast(
            region=self.region,
            projection=self.projection,
            land="245/245/235",
            water="220/235/245",
            shorelines="0.4p,black",
            borders=["1/0.4p,gray40"],
            frame=["a10f5", "WSen"],
            resolution="l",
        )

    # ------------------------------------------------------------------ #
    # Moho 地图
    # ------------------------------------------------------------------ #
    def plot_moho_map(self, df: pd.DataFrame,
                      save_file: Optional[str] = None) -> str:
        self.logger.info("🎨 绘制 Moho 深度分布图...")
        if save_file is None:
            save_file = generate_filename(self.module_prefix, "moho_depth_map", "jpg")

        fig = pygmt.Figure()
        self._draw_basemap(fig)

        # 深度色标
        pygmt.makecpt(cmap=self.config.style['moho_cmap'],
                      series=self.config.style['moho_series'],
                      continuous=True, reverse=True)

        # 按类型用不同符号区分
        symbols = {1: 'c', 2: 't', 3: 's'}  # circle / triangle / square
        labels = self.config.moho_type_labels_en
        # 先绘制变色散点（不带 label，避免 PyGMT 警告）
        for t in [1, 2, 3]:
            sub = df[df['type'] == t]
            if sub.empty:
                continue
            fig.plot(
                x=sub['longitude'].values,
                y=sub['latitude'].values,
                fill=sub['depth'].values,
                style=f"{symbols[t]}{self.config.style['point_size']}",
                pen=self.config.style['point_pen'],
                cmap=True,
            )
        # 再画小型"代理点"用于生成方法分类图例（位于区域外，不影响视觉）
        proxy_lon = self.region[0] - 100
        proxy_lat = self.region[2] - 100
        for t in [1, 2, 3]:
            n = int((df['type'] == t).sum())
            if n == 0:
                continue
            fig.plot(
                x=[proxy_lon], y=[proxy_lat],
                fill="gray40",
                style=f"{symbols[t]}0.25c",
                pen="0.3p,black",
                label=f"{labels[t]} (N={n})",
            )

        fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p0.5p,black")
        fig.colorbar(position=self.config.colorbar['moho_position'],
                     frame=self.config.colorbar['moho_frame'])

        return self._save_figure(fig, save_file)

    # ------------------------------------------------------------------ #
    # Moho 按方法类型分图（3 张独立 JPG）
    # ------------------------------------------------------------------ #
    def plot_moho_by_type(self, df: pd.DataFrame) -> Dict[str, str]:
        """为每种方法（type=1/2/3）单独画一张 Moho 深度分布图。"""
        outputs: Dict[str, str] = {}
        symbols = {1: 'c', 2: 't', 3: 's'}
        labels_en = self.config.moho_type_labels_en
        # 文件名后缀（避免空格/中文）
        suffix_map = {
            1: "moho_depth_map_type1_RF",
            2: "moho_depth_map_type2_RF_SurfWave",
            3: "moho_depth_map_type3_ActiveSource",
        }

        for t in [1, 2, 3]:
            sub = df[df['type'] == t]
            if sub.empty:
                self.logger.warning(f"⚠️ Moho 类型 {t} 无数据，跳过")
                continue

            self.logger.info(
                f"🎨 绘制 Moho 类型 {t} ({labels_en[t]}) 分布图: N={len(sub)}"
            )
            save_file = generate_filename(self.module_prefix, suffix_map[t], "jpg")

            fig = pygmt.Figure()
            self._draw_basemap(fig)

            pygmt.makecpt(
                cmap=self.config.style['moho_cmap'],
                series=self.config.style['moho_series'],
                continuous=True,
                reverse=True,
            )

            fig.plot(
                x=sub['longitude'].values,
                y=sub['latitude'].values,
                fill=sub['depth'].values,
                style=f"{symbols[t]}{self.config.style['point_size']}",
                pen=self.config.style['point_pen'],
                cmap=True,
            )

            # 代理点（区域外）用于图例
            fig.plot(
                x=[self.region[0] - 100], y=[self.region[2] - 100],
                fill="gray40",
                style=f"{symbols[t]}0.25c",
                pen="0.3p,black",
                label=f"{labels_en[t]} (N={len(sub)})",
            )
            fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p0.5p,black")
            fig.colorbar(
                position=self.config.colorbar['moho_position'],
                frame=self.config.colorbar['moho_frame'],
            )

            outputs[f"moho_type{t}_map"] = self._save_figure(fig, save_file)

        return outputs

    # ------------------------------------------------------------------ #
    # LAB 地图
    # ------------------------------------------------------------------ #
    def plot_lab_map(self, df: pd.DataFrame,
                     save_file: Optional[str] = None) -> str:
        self.logger.info("🎨 绘制 LAB 深度分布图...")
        if save_file is None:
            save_file = generate_filename(self.module_prefix, "lab_depth_map", "jpg")

        fig = pygmt.Figure()
        self._draw_basemap(fig)

        pygmt.makecpt(cmap=self.config.style['lab_cmap'],
                      series=self.config.style['lab_series'],
                      continuous=True, reverse=True)

        fig.plot(
            x=df['longitude'].values,
            y=df['latitude'].values,
            fill=df['depth'].values,
            style=f"c{self.config.style['point_size']}",
            pen=self.config.style['point_pen'],
            cmap=True,
        )
        # 代理点用于图例（区域外不可见）
        fig.plot(
            x=[self.region[0] - 100], y=[self.region[2] - 100],
            fill="gray40", style="c0.25c", pen="0.3p,black",
            label=f"LAB samples (N={len(df)})",
        )

        fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p0.5p,black")
        fig.colorbar(position=self.config.colorbar['lab_position'],
                     frame=self.config.colorbar['lab_frame'])

        return self._save_figure(fig, save_file)

    # ------------------------------------------------------------------ #
    # 统计图（matplotlib）
    # ------------------------------------------------------------------ #
    def plot_statistics(self,
                        moho: pd.DataFrame,
                        lab: pd.DataFrame,
                        save_file: Optional[str] = None) -> str:
        self.logger.info("📊 绘制 Moho/LAB 深度统计图...")
        if save_file is None:
            save_file = generate_filename(self.module_prefix, "moho_lab_statistics", "jpg")

        cfg = self.config.statistics
        fig, axes = plt.subplots(2, 2, figsize=cfg['figsize'])

        # (0,0) Moho 总分布
        ax = axes[0, 0]
        ax.hist(moho['depth'], bins=cfg['bins_moho'],
                color=cfg['hist_color_total'], edgecolor='black', alpha=0.85)
        ax.set_xlabel('Moho depth (km)')
        ax.set_ylabel('Count')
        ax.set_title(f"Moho depth distribution (N={len(moho)})")
        ax.grid(alpha=0.3)
        ax.axvline(moho['depth'].mean(), color='red', linestyle='--', lw=1,
                   label=f"mean={moho['depth'].mean():.1f} km")
        ax.axvline(moho['depth'].median(), color='green', linestyle=':', lw=1.5,
                   label=f"median={moho['depth'].median():.1f} km")
        ax.legend()

        # (0,1) Moho 按方法堆叠
        ax = axes[0, 1]
        type_labels = self.config.moho_type_labels_en
        data_list, labels_list, colors_list = [], [], []
        for t in [1, 2, 3]:
            sub = moho[moho['type'] == t]['depth']
            if len(sub) == 0:
                continue
            data_list.append(sub.values)
            labels_list.append(f"{type_labels[t]} (N={len(sub)})")
            colors_list.append(cfg['hist_colors_type'][t])
        ax.hist(data_list, bins=cfg['bins_moho'], stacked=True,
                color=colors_list, edgecolor='black', alpha=0.85,
                label=labels_list)
        ax.set_xlabel('Moho depth (km)')
        ax.set_ylabel('Count')
        ax.set_title("Moho depth by method")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # (1,0) LAB 分布
        ax = axes[1, 0]
        ax.hist(lab['depth'], bins=cfg['bins_lab'],
                color='#9467bd', edgecolor='black', alpha=0.85)
        ax.set_xlabel('LAB depth (km)')
        ax.set_ylabel('Count')
        ax.set_title(f"LAB depth distribution (N={len(lab)})")
        ax.grid(alpha=0.3)
        ax.axvline(lab['depth'].mean(), color='red', linestyle='--', lw=1,
                   label=f"mean={lab['depth'].mean():.1f} km")
        ax.axvline(lab['depth'].median(), color='green', linestyle=':', lw=1.5,
                   label=f"median={lab['depth'].median():.1f} km")
        ax.legend()

        # (1,1) Moho vs LAB 联合散点（取相近台站对的简单密度概念：直接画两者直方对照）
        ax = axes[1, 1]
        ax.hist(moho['depth'], bins=cfg['bins_moho'],
                color=cfg['hist_color_total'], alpha=0.55,
                label=f"Moho (N={len(moho)})", density=True)
        ax.hist(lab['depth'], bins=cfg['bins_lab'],
                color='#9467bd', alpha=0.55,
                label=f"LAB  (N={len(lab)})", density=True)
        ax.set_xlabel('Depth (km)')
        ax.set_ylabel('Density')
        ax.set_title('Moho vs LAB depth (normalized)')
        ax.grid(alpha=0.3)
        ax.legend()

        fig.suptitle('Crustal Moho & Lithospheric LAB Depth Statistics — East Asia',
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        # 保存 jpg + pdf
        out_jpg = self.output_dir / save_file
        fig.savefig(out_jpg, dpi=self.config.output['dpi_jpg'], bbox_inches='tight')
        if self.config.output['save_pdf']:
            out_pdf = out_jpg.with_suffix('.pdf')
            fig.savefig(out_pdf, dpi=self.config.output['dpi_pdf'], bbox_inches='tight')
            self.logger.info(f"📄 统计 PDF 已保存: {out_pdf}")
        plt.close(fig)
        self.logger.info(f"🖼️ 统计 JPG 已保存: {out_jpg}")
        return str(out_jpg)

    # ------------------------------------------------------------------ #
    # Type 2 (RF + 面波联合反演) 网格平滑成 Moho 面
    # ------------------------------------------------------------------ #
    def plot_moho_type2_surface(self,
                                df: pd.DataFrame,
                                spacing: float = 0.25,
                                tension: float = 0.25,
                                save_file: Optional[str] = None) -> str:
        """对 type=2 的规则网格点用 pygmt.surface 做样条光滑，
        生成连续的 Moho 深度面 (类似论文中网格化 Moho 图)。
        """
        sub = df[df['type'] == 2][['longitude', 'latitude', 'depth']]
        if sub.empty:
            self.logger.warning("⚠️ type=2 数据为空，跳过 surface 平滑图")
            return ""

        self.logger.info(
            f"🎨 [Type 2 surface] 输入点数 N={len(sub)}, "
            f"spacing={spacing}°, tension={tension}"
        )
        if save_file is None:
            save_file = generate_filename(
                self.module_prefix, "moho_type2_surface", "jpg")

        # blockmean 先去掉同格重复，再 surface 光滑成网格
        blocked = pygmt.blockmean(
            data=sub.values,
            region=self.region,
            spacing=f"{spacing}/{spacing}",
        )
        grid = pygmt.surface(
            data=blocked,
            region=self.region,
            spacing=f"{spacing}/{spacing}",
            tension=tension,
        )
        # 只保留有数据覆盖的区域：用 KDTree 计算每个网格节点到最近 type2 点的距离，
        # 超过 mask_radius 的节点设为 NaN，避免 surface 外推到无观测区域。
        from scipy.spatial import cKDTree
        mask_radius = max(spacing * 3.0, 1.0)  # 度，约 100 km

        # 提取 grid 的坐标轴
        lons = grid['x'].values  # type: ignore[index]
        lats = grid['y'].values  # type: ignore[index]
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        tree = cKDTree(sub[['longitude', 'latitude']].values)
        dist, _ = tree.query(
            np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()]), k=1
        )
        dist = dist.reshape(lon_mesh.shape)
        # 在原始 grid 的 numpy 视图上写 NaN（xarray 会同步）
        grid.values[dist > mask_radius] = np.nan  # type: ignore[attr-defined]

        fig = pygmt.Figure()
        # 仅画框架（不画陆地/海洋底色，避免遮挡数据/外推区域可视化）
        fig.basemap(region=self.region, projection=self.projection,
                    frame=["a10f5", "WSen"])
        # Moho 深度色标（与散点图保持一致）
        pygmt.makecpt(cmap=self.config.style['moho_cmap'],
                      series=self.config.style['moho_series'],
                      continuous=True, reverse=True)
        fig.grdimage(grid=grid, projection=self.projection,
                     region=self.region, cmap=True, nan_transparent=True)
        # 叠加海岸线和国界（轻量）
        fig.coast(region=self.region, projection=self.projection,
                  shorelines="0.4p,black",
                  borders=["1/0.4p,gray40"])
        fig.colorbar(position=self.config.colorbar['moho_position'],
                     frame=["x+lMoho depth (Type 2 surface)", "y+lkm"])

        return self._save_figure(fig, save_file)

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    def _save_figure(self, fig: pygmt.Figure, save_file: str) -> str:
        out_jpg = self.output_dir / save_file
        if self.config.output['save_jpg']:
            fig.savefig(str(out_jpg),
                        dpi=self.config.output['dpi_jpg'],
                        crop=self.config.output['crop'])
            self.logger.info(f"🖼️ JPG 已保存: {out_jpg}")
        if self.config.output['save_pdf']:
            out_pdf = out_jpg.with_suffix('.pdf')
            fig.savefig(str(out_pdf),
                        dpi=self.config.output['dpi_pdf'],
                        crop=self.config.output['crop'])
            self.logger.info(f"📄 PDF 已保存: {out_pdf}")
        return str(out_jpg)

    # ------------------------------------------------------------------ #
    # 一键全流程
    # ------------------------------------------------------------------ #
    def plot_all(self) -> Dict[str, str]:
        moho = self.load_moho_data()
        lab = self.load_lab_data()
        outputs = {
            'moho_map': self.plot_moho_map(moho),
            'lab_map': self.plot_lab_map(lab),
            'statistics': self.plot_statistics(moho, lab),
        }
        # 按方法类型分别画 3 张 Moho 图
        outputs.update(self.plot_moho_by_type(moho))
        # Type 2 网格点经 surface 光滑后的 Moho 深度面
        outputs['moho_type2_surface'] = self.plot_moho_type2_surface(moho)
        # 另一个外部 Moho 模型：CSRM1.0（原始网格点 + surface 光滑）
        try:
            csrm = self.load_csrm_moho()
            outputs['moho_CSRM1.0_raw'] = self.plot_csrm_moho_raw(csrm)
            outputs['moho_CSRM1.0_surface'] = self.plot_csrm_moho_surface(csrm)
        except FileNotFoundError as e:
            self.logger.warning(f"⚠️ 跳过 CSRM Moho: {e}")
        return outputs


def main():
    print("🗻 EASTASIA-FWI 莫霍面 / LAB 深度可视化系统")
    print("=" * 70)
    try:
        plotter = MohoLABPlotter()
        outputs = plotter.plot_all()
        print("\n✅ 全部绘图完成:")
        for k, v in outputs.items():
            print(f"  - {k}: {v}")
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
