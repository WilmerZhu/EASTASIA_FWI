"""
5_9_Geology.py:
东亚及邻区多幅 USGS 地质图绘制模块
================================================================

功能描述:
----------
参考 geology_Alps.py 的 PyGMT 流程，绘制 USGS World Geologic Maps 系列：
- geo3al  : 中国及邻区（Far East）基岩地质年代
- geo3bl  : 东南亚—澳大利亚（SE Asia / Australasia）
- geo8apg : 南亚—印度次大陆（South Asia）
- prv1ec  : 欧亚地质省边界（非年代图，线/分区叠图）
- prv3bl  : 东南亚—澳大利亚地质省边界
- CN-block: 中国及邻区活动地块（L1 / L1-deduced / L2）
并支持拼合图（age layers + tectonic block / province outlines）。

科学说明:
----------
- geo3al / geo3bl / geo8apg 为地表基岩地质，语义对应上地壳。
- prv1ec / prv3bl 是 geologic province，不是年代面，不能用 age CPT 填色。
- geo8apg 是南亚年代多边形（GLG），已参与拼合填色；不是地质省面。
- CN-block 为活动地块边界（线划叠图），默认叠在年代填色之上。
- 默认 CPT: geoage_EASTASIA_GLG.cpt
  （geoage.cpt 色核 + EURO 火成岩图案 + 区域 GLG 别名）

数据引用:
----------
Steinshouer et al. (1999), USGS OFR 97-470-F / C / E / B.
https://docs.gmt-china.org/latest/dataset/geo3al/

作者: EASTASIA-FWI Team
日期: 2026-08-04
版本: v1.5
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pygmt
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import linemerge, unary_union

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import BaseConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from utils.file_utils import extract_module_prefix, generate_filename  # noqa: E402


@dataclass
class GeologyDataset:
    """单幅地质图数据集描述"""

    key: str
    gmt_file: str
    title: str
    # 'age' = 年代填色；'province' = 地质省边界
    kind: str
    age_field: Optional[str]
    region: List[float]  # [lon_min, lon_max, lat_min, lat_max]
    description: str  # 输出文件名后缀
    # 拼合图中是否参与年代填色
    use_in_composite: bool = True
    pen: Optional[str] = None  # province 线型


# USGS 年代地质图四幅 + 可选地质省（geo8alg 为线划索引，默认不单独出图）
DEFAULT_DATASETS: Dict[str, GeologyDataset] = {
    'geo3al': GeologyDataset(
        key='geo3al',
        gmt_file='geo3al.gmt',
        title='Far East (China & adjacent)',
        kind='age',
        age_field='GEN_GLG',
        region=[70.0, 150.0, 13.0, 55.0],
        description='geo3al_fareast',
    ),
    'geo3bl': GeologyDataset(
        key='geo3bl',
        gmt_file='geo3bl.gmt',
        title='SE Asia & Australasia',
        kind='age',
        age_field='GEN_GLG',
        region=[90.0, 165.0, -14.0, 28.0],
        description='geo3bl_seasia',
    ),
    'geo8apg': GeologyDataset(
        key='geo8apg',
        gmt_file='geo8apg.gmt',
        title='South Asia (India & adjacent)',
        kind='age',
        age_field='GLG',
        region=[60.0, 102.0, 1.0, 39.0],
        description='geo8apg_southasia',
    ),
    'geo1ec': GeologyDataset(
        key='geo1ec',
        gmt_file='geo1ec.gmt',
        title='Northern Eurasia (former USSR)',
        kind='age',
        age_field='GLG',
        region=[60.0, 150.0, 25.0, 70.0],
        description='geo1ec_ussr',
    ),
    'prv1ec': GeologyDataset(
        key='prv1ec',
        gmt_file='prv1ec.gmt',
        title='Eurasia geologic provinces',
        kind='province',
        age_field=None,
        region=[60.0, 150.0, 10.0, 60.0],
        description='prv1ec_provinces',
        use_in_composite=False,
        pen='0.30p,red',
    ),
    'prv3bl': GeologyDataset(
        key='prv3bl',
        gmt_file='prv3bl.gmt',
        title='SE Asia & Australasia geologic provinces',
        kind='province',
        age_field=None,
        region=[90.0, 165.0, -14.0, 28.0],
        description='prv3bl_provinces',
        use_in_composite=False,
        pen='0.30p,red',
    ),
}


class GeologyConfig:
    """多幅地质图模块配置"""

    def __init__(self) -> None:
        # 与 5_1_Basemap 保持同一套 PyGMT / 投影规格
        self.pygmt = {
            'map_frame_type': 'plain',
            'map_grid_pen_primary': '0.3p,dimgrey',
            'map_annot_oblique': '30',
            'map_annot_offset_primary': '5p',
            'map_annot_offset_secondary': '5p',
            'font_annot_primary': '10p,4',
            'font_label': '10p,28,black',
            'map_frame_width': '2p',
            'map_frame_pen': '0.5p',
            'map_tick_length_primary': '5p',
            'map_tick_pen_primary': '0.5p,black,-',
            'map_label_offset': '5p',
        }

        self.projection = {
            'type': 'M',           # 墨卡托
            'width': '15c',        # 与 Basemap 一致
            'frame_interval': 'a10f1',
            # 实际 frame = ['WseN', frame_interval]，在模块初始化时组装
            'frame': ['WseN', 'a10f1'],
        }

        # 拼合图范围默认占位；运行时由 BaseConfig.region 覆盖（与 Basemap 一致）
        self.composite_region = [60.0, 150.0, -10.0, 60.0]

        # 与 2_3_Model_clustering 三模型公共范围（由 original.nc 求交）
        self.model_overlap = {
            'models': [
                '2022_SinoScope1.0',
                '2024_EARA2024',
                '2024_FWEA23',
            ],
            # 相对 base_config.dirs['data']
            'netcdf_pattern': (
                'models/processed/{model_name}/{model_name}_original.nc'
            ),
            'description': 'model_overlap_geology',
        }

        # 拼合填色顺序：先画的在下；后画覆盖重叠区（Far East 优先）
        self.composite_age_order: List[str] = [
            'geo1ec',   # 前苏联 / 北方
            'geo3bl',   # 东南亚
            'geo8apg',  # 南亚
            'geo3al',   # 中国及邻区（最上）
        ]

        self.data = {
            'geology_dir': 'geology',
            'cmap_file': 'geoage_EASTASIA_GLG.cpt',
            'cmap_euro_file': 'geoage_EURO_GLG.cpt',
            'cmap_geoage_file': 'geoage.cpt',
            'legend_file': 'age_legend_eastasia.txt',
            'legend_euro_file': 'age_legend.txt',
            # 兼容旧字段；实际叠图用 block_layers
            'block_file': 'CN-block-L1.gmt',
            # 中国大陆及周边活动地块（gmt-china geospatial-data）
            'block_layers': [
                {
                    'file': 'CN-block-L2.gmt',
                    'pen': '0.25p,red',
                    'label': 'CN block L2',
                },
                {
                    'file': 'CN-block-L1-deduced.gmt',
                    'pen': '0.30p,red',
                    'label': 'CN block L1 deduced',
                },
                {
                    'file': 'CN-block-L1.gmt',
                    'pen': '0.35p,red',
                    'label': 'CN block L1',
                },
            ],
        }

        # 默认只出一张东亚拼合图；需要单幅时自行传入 dataset_keys
        self.datasets_to_plot: List[str] = [
            'composite',
        ]

        self.layers = {
            # 与 comprehensive Basemap 一致：不描海岸线
            'coast': False,
            'geology': True,
            # 洋区：优先水深阴影；失败时回退纯色 ocean_mask
            'ocean_bathymetry': True,
            'ocean_mask': True,  # 仅填洋面，不画 shoreline
            'cn_blocks': True,  # 叠中国及邻区活动地块边界
            'province_overlay': True,  # prv1ec + prv3bl 地质省边界
            'legend': True,
            'title': True,
        }

        # 拼合图叠绘的地质省图层（按顺序绘制）
        self.province_overlay_keys: List[str] = ['prv1ec', 'prv3bl']

        self.style = {
            'shoreline': '0.35p,black',
            'coast_resolution': 'l',   # low：避免 intermediate 碎岛过密
            'area_thresh': '500',     # 滤掉小岛屿/湖，轮廓更干净
            'ocean_fill': 'cadetblue1',
            'land_fill': 'white',
            # 洋区水深：geo/topo 色标裁到负高程；shading 增强立体感
            'bathy_cmap': 'geo',
            'bathy_series': '-8000/0',
            'bathy_shading': True,
            'block_pen': '0.30p,red',  # 回退笔（单文件模式）
            'province_pen': '0.30p,red',
            'province_fill': None,  # 仅线框
            # 地质省通用过滤：轻度过滤 + 去掉覆盖区最南侧外轮廓
            'province_filter': {
                'enabled': True,
                'max_lon_span': 55.0,
                'max_lat_span': 35.0,
                'min_frac_in_region': 0.25,
                'exclude_name_keywords': ['OCS'],
                'drop_south_outer_rim': True,
                'drop_north_outer_rim': False,
                'rim_probe_deg': 0.20,
                'frame_buffer_deg': 0.05,
                'drop_connector_boxes': [],
            },
            # 按数据集覆盖过滤参数（缓存文件名必须互不覆盖）
            'province_filter_by_key': {
                'prv1ec': {
                    # 区域与 Basemap 对齐后重建缓存
                    'cache_name': 'prv1ec_nosouth_rim_v3.gmt',
                    # 南界去掉后残留的中亚拼接直线
                    'drop_connector_boxes': [
                        {
                            'lon': [80.0, 85.5],
                            'lat': [39.5, 46.5],
                            'min_length_deg': 0.6,
                            'near_meridian_tol_deg': 35.0,
                        },
                    ],
                },
                'prv3bl': {
                    'cache_name': 'prv3bl_nosouth_north_rim_v2.gmt',
                    # 东南亚岛弧省区跨度可能更大；同时去掉数据南北外轮廓
                    'max_lon_span': 70.0,
                    'max_lat_span': 40.0,
                    'drop_south_outer_rim': True,
                    'drop_north_outer_rim': True,
                    'drop_connector_boxes': [],
                },
            },
            # 图例：下移后与地图同宽（必须同时给 width/height，否则 GMT 会按内容缩框）
            'legend_match_map_width': True,
            'legend_offset_below': '1.20c',
            'legend_height': '4.0c',   # 4 列时行数略增
            'legend_spacing': '1.12',
            'legend_box': '+p0.6p+g255',
            'legend_position': 'x0/0+w15c/4.0c+jLT+l1.12',
        }

        self.output = {
            'dpi_jpg': 300,
            'dpi_pdf': 300,
            'crop': True,
            'save_jpg': True,
            'save_pdf': True,
        }

        self.logging = {'level': 'INFO'}


class EastAsiaGeologyMap:
    """多幅 USGS 地质图绘制器"""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        config: Optional[GeologyConfig] = None,
        datasets: Optional[Dict[str, GeologyDataset]] = None,
    ) -> None:
        self.base_config = BaseConfig()
        self.config = config or GeologyConfig()
        self.datasets = datasets or dict(DEFAULT_DATASETS)

        self.output_dir = (
            Path(output_dir) if output_dir else self.base_config.dirs['figures']
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.GeologyMap',
            self.config.logging['level'],
        )

        # 与 5_1_Basemap 相同：区域取自 BaseConfig
        bounds = self.base_config.get_region_bounds()
        self.region = [
            bounds['lon_min'],
            bounds['lon_max'],
            bounds['lat_min'],
            bounds['lat_max'],
        ]
        self.config.composite_region = list(self.region)

        # 投影 / 图框与 Basemap 对齐
        proj = self.config.projection
        self.projection = f"{proj['type']}{proj['width']}"
        interval = proj.get('frame_interval', 'a10f1')
        self.config.projection['frame'] = ['WseN', interval]

        self.geology_dir = (
            self.base_config.dirs['data'] / self.config.data['geology_dir']
        )
        self.cmap_file = self.geology_dir / self.config.data['cmap_file']
        self.legend_file = self.geology_dir / self.config.data['legend_file']
        self.block_file = self.geology_dir / self.config.data['block_file']
        self.block_layers = list(self.config.data.get('block_layers', []))
        self.legend_plot_file = self.geology_dir / 'age_legend_eastasia_plot.txt'

        self.module_prefix = extract_module_prefix(Path(__file__), default='5-9')

        self._setup_pygmt_config()
        self.logger.info('🗺️ 东亚多幅地质图模块初始化完成 (v1.5)')
        self._print_config_summary()

    def _setup_pygmt_config(self) -> None:
        c = self.config.pygmt
        kwargs = dict(
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
            MAP_LABEL_OFFSET=c['map_label_offset'],
        )
        if c.get('map_tick_pen_primary'):
            kwargs['MAP_TICK_PEN_PRIMARY'] = c['map_tick_pen_primary']
        pygmt.config(**kwargs)

    def _print_config_summary(self) -> None:
        self.logger.info(f"  CPT: {self.cmap_file.name}")
        self.logger.info(f"  图例: {self.legend_file.name}")
        self.logger.info(
            f"  区域/投影: {self.region} / {self.projection} "
            f"(与 Basemap 一致)"
        )
        self.logger.info(f"  将绘制: {self.config.datasets_to_plot}")
        self.logger.info(f"  输出: {self.output_dir}")

    # ------------------------------------------------------------------ #
    # 数据准备
    # ------------------------------------------------------------------ #
    def prepare_age_dat(
        self, dataset: GeologyDataset, force: bool = False
    ) -> Path:
        """gmt convert -aZ=<age_field> → *.dat"""
        gmt_path = self.geology_dir / dataset.gmt_file
        if not gmt_path.exists():
            raise FileNotFoundError(f'找不到地质数据: {gmt_path}')
        if not dataset.age_field:
            raise ValueError(f'{dataset.key} 不是年代图，无需 convert Z')

        out = self.geology_dir / f'{dataset.key}_{dataset.age_field}.dat'
        if (
            out.exists()
            and not force
            and out.stat().st_mtime >= gmt_path.stat().st_mtime
        ):
            self.logger.info(f'  ✅ 缓存: {out.name}')
            return out

        self.logger.info(
            f'  🔄 gmt convert {gmt_path.name} -aZ={dataset.age_field!r}'
        )
        result = subprocess.run(
            ['gmt', 'convert', str(gmt_path), f'-aZ={dataset.age_field}'],
            check=True,
            capture_output=True,
            text=True,
        )
        out.write_text(result.stdout, encoding='utf-8')
        self.logger.info(
            f'  ✅ {out.name} ({out.stat().st_size / 1e6:.1f} MB)'
        )
        return out

    def _legend_position(self) -> str:
        """图例 position：x0 左对齐，宽=地图投影宽，高显式给定以防缩框。"""
        style = self.config.style
        if not style.get('legend_match_map_width', True):
            return style['legend_position']
        map_width = self.config.projection['width']
        height = style.get('legend_height', '3.4c')
        spacing = style.get('legend_spacing', '1.12')
        return f'x0/0+w{map_width}/{height}+jLT+l{spacing}'

    def _add_legend(self, fig: pygmt.Figure, legend_path: Path) -> None:
        """
        将图例置于地图正下方，框宽与 basemap 投影宽度一致。

        GMT 若只给 +w 不给 height，常按文字内容收缩图例框；
        因此同时指定 width/height，并用 shift_origin 放到地图下方。
        """
        style = self.config.style
        dy = style.get('legend_offset_below', '1.20c')
        pos = self._legend_position()
        self.logger.info(f'  图例: shift -{dy}, position={pos}')
        fig.shift_origin(yshift=f'-{dy}')
        fig.legend(
            spec=str(legend_path),
            position=pos,
            box=style['legend_box'],
        )

    def prepare_legend(self) -> Path:
        """清理图例标签前分号"""
        legend_src = self.legend_file
        if not legend_src.exists():
            euro = self.geology_dir / self.config.data['legend_euro_file']
            if euro.exists():
                legend_src = euro
                self.logger.warning(f'  ⚠️ 回退图例: {euro.name}')
            else:
                raise FileNotFoundError(f'找不到图例: {self.legend_file}')

        lines_out: List[str] = []
        for line in legend_src.read_text(encoding='utf-8').splitlines():
            if line.startswith('S ') and ';' in line:
                line = re.sub(r'(?<=\s);(?=\S)', '', line)
            lines_out.append(line)
        self.legend_plot_file.write_text(
            '\n'.join(lines_out) + '\n', encoding='utf-8'
        )
        return self.legend_plot_file

    def _resolve_cmap(self) -> Path:
        if self.cmap_file.exists():
            return self.cmap_file
        for key in ('cmap_geoage_file', 'cmap_euro_file'):
            alt = self.geology_dir / self.config.data[key]
            if alt.exists():
                self.logger.warning(f'  ⚠️ CPT 回退: {alt.name}')
                return alt
        raise FileNotFoundError(f'找不到 CPT: {self.cmap_file}')

    # ------------------------------------------------------------------ #
    # 绘图
    # ------------------------------------------------------------------ #
    def _draw_basemap(self, fig: pygmt.Figure, region: Sequence[float]) -> None:
        fig.basemap(
            region=list(region),
            projection=self.projection,
            frame=self.config.projection['frame'],
        )
        # 默认关闭海岸线（与 comprehensive Basemap 一致）；开启时也仅填陆地、不描岸线
        if self.config.layers['coast']:
            style = self.config.style
            fig.coast(
                region=list(region),
                projection=self.projection,
                resolution=style['coast_resolution'],
                area_thresh=style['area_thresh'],
                land=style['land_fill'],
            )

    def _elevation_grid_path(self) -> Path:
        """本地 EastAsia / earth_relief 网格；缺失时回退 GMT 远程别名。"""
        elev_dir = Path(__file__).parent / 'data_EastAsia' / 'elevation'
        for name in (
            'EastAsia.grd',
            'earth_relief_01m.grd',
            'earth_relief_30s.grd',
        ):
            path = elev_dir / name
            if path.exists() and path.stat().st_size > 1000:
                return path
        return Path('@earth_relief_01m')

    def _draw_ocean_bathymetry(
        self, fig: pygmt.Figure, region: Sequence[float]
    ) -> bool:
        """
        洋区水深阴影底图（兼容不支持 wet-clip 的 GMT）。

        流程：全图 grdimage(relief) → 陆地填白；随后年代地质盖在陆上。
        地质多边形常伸入陆架/洋区，需在地质之后再调用
        ``_overlay_ocean_bathymetry`` 仅洋区重绘水深。成功返回 True。
        """
        if not self.config.layers.get('ocean_bathymetry', False):
            return False
        style = self.config.style
        grid = self._elevation_grid_path()
        region_list = list(region)
        try:
            pygmt.makecpt(
                cmap=style.get('bathy_cmap', 'geo'),
                series=style.get('bathy_series', '-8000/0'),
                background=True,
            )
            fig.grdimage(
                grid=str(grid),
                region=region_list,
                projection=self.projection,
                cmap=True,
                shading=bool(style.get('bathy_shading', True)),
            )
            # 陆地抹白，避免陆上地形干扰基岩地质
            fig.coast(
                region=region_list,
                projection=self.projection,
                resolution=style['coast_resolution'],
                area_thresh=style['area_thresh'],
                land=style.get('land_fill', 'white'),
            )
            self.logger.info(f'  ✅ 洋区水深阴影(底层): {grid}')
            return True
        except Exception as e:
            self.logger.warning(f'  ⚠️ 水深阴影失败，回退纯色洋面: {e}')
            return False

    def _overlay_ocean_bathymetry(
        self, fig: pygmt.Figure, region: Sequence[float]
    ) -> bool:
        """
        地质填色之后：仅在洋区重绘水深（陆地为 NaN，透明）。

        覆盖伸入海域的 USGS 年代多边形，避免“水层/陆架地质”挡住水深纹理。
        海陆掩膜按裁切后的地形网格间距生成，并用最近邻对齐，避免宽区域
        下 ``grdsample`` 与 ``grdlandmask`` 节点数不一致。
        """
        if not self.config.layers.get('ocean_bathymetry', False):
            return False
        import numpy as np

        style = self.config.style
        grid = self._elevation_grid_path()
        region_list = [float(v) for v in region]
        try:
            bathy = pygmt.grdcut(grid=str(grid), region=region_list)
            ydim, xdim = str(bathy.dims[0]), str(bathy.dims[1])
            y = np.asarray(bathy[ydim].values, dtype=float)
            x = np.asarray(bathy[xdim].values, dtype=float)
            if y.size < 2 or x.size < 2:
                raise ValueError('地形网格节点过少，无法生成洋区掩膜')
            dy = abs(float(y[1] - y[0]))
            dx = abs(float(x[1] - x[0]))
            # 用节点数锁定网格，避免浮点 spacing 造成 ±1 列偏差
            mask = pygmt.grdlandmask(
                region=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
                spacing=[f'{x.size}+n', f'{y.size}+n'],
                mask_values=[1.0, np.nan],
                resolution=style.get('coast_resolution', 'l'),
                area_thresh=style.get('area_thresh', '500'),
            )
            # 统一坐标名后按 bathy 网格最近邻对齐
            rename = {}
            if str(mask.dims[0]) != ydim:
                rename[str(mask.dims[0])] = ydim
            if str(mask.dims[1]) != xdim:
                rename[str(mask.dims[1])] = xdim
            if rename:
                mask = mask.rename(rename)
            if mask.shape != bathy.shape:
                mask = mask.interp(
                    {ydim: bathy[ydim], xdim: bathy[xdim]},
                    method='nearest',
                )
            mask_vals = np.asarray(mask.values, dtype=float)
            bathy_vals = np.asarray(bathy.values, dtype=float)
            ocean_vals = np.where(
                np.isfinite(mask_vals) & (mask_vals > 0.5),
                bathy_vals,
                np.nan,
            )
            ocean = bathy.copy(data=ocean_vals)

            pygmt.makecpt(
                cmap=style.get('bathy_cmap', 'geo'),
                series=style.get('bathy_series', '-8000/0'),
                background=True,
            )
            fig.grdimage(
                grid=ocean,
                region=region_list,
                projection=self.projection,
                cmap=True,
                shading=bool(style.get('bathy_shading', True)),
                nan_transparent=True,
            )
            self.logger.info(
                f'  ✅ 洋区水深阴影(叠于地质之上，仅湿区) grid={ocean_vals.shape}'
            )
            return True
        except Exception as e:
            self.logger.warning(f'  ⚠️ 洋区水深重绘失败: {e}')
            return False

    def _draw_ocean_mask(
        self, fig: pygmt.Figure, region: Sequence[float]
    ) -> None:
        """用海水色覆盖洋面；不绘制海岸线描边。"""
        if not self.config.layers['ocean_mask']:
            return
        style = self.config.style
        fig.coast(
            region=list(region),
            projection=self.projection,
            resolution=style['coast_resolution'],
            area_thresh=style['area_thresh'],
            water=style['ocean_fill'],
        )

    def _draw_age_layer(
        self,
        fig: pygmt.Figure,
        dataset: GeologyDataset,
        region: Sequence[float],
        cmap: Path,
        force_convert: bool,
    ) -> None:
        dat = self.prepare_age_dat(dataset, force=force_convert)
        self.logger.info(
            f'  填色 {dataset.key}: Z={dataset.age_field}, CPT={cmap.name}'
        )
        fig.plot(
            data=str(dat),
            region=list(region),
            projection=self.projection,
            fill='+z',
            cmap=str(cmap),
            close=True,
        )

    @staticmethod
    def _iter_lines(geom) -> Iterable[LineString]:
        """展开 shapely 线几何为 LineString 列表"""
        if geom is None or geom.is_empty:
            return []
        if isinstance(geom, LineString):
            return [geom]
        if isinstance(geom, MultiLineString):
            return list(geom.geoms)
        if isinstance(geom, GeometryCollection):
            out: List[LineString] = []
            for g in geom.geoms:
                out.extend(EastAsiaGeologyMap._iter_lines(g))
            return out
        if isinstance(geom, (Polygon, MultiPolygon)):
            return EastAsiaGeologyMap._iter_lines(geom.boundary)
        return []

    def _write_line_gmt(
        self,
        path: Path,
        segs: Sequence[Sequence[Tuple[float, float]]],
        header: str,
    ) -> None:
        lines = [header if header.endswith('\n') else header + '\n']
        for xy in segs:
            if len(xy) < 2:
                continue
            lines.append('>\n')
            for lon, lat in xy:
                lines.append(f'{lon:.6f} {lat:.6f}\n')
        path.write_text(''.join(lines), encoding='utf-8')

    @staticmethod
    def _edge_in_connector_box(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        boxes: Sequence[Dict],
    ) -> bool:
        """是否为指定框内的近南北向长连接边"""
        mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        length = math.hypot(x1 - x0, y1 - y0)
        ang = abs(math.degrees(math.atan2(y1 - y0, x1 - x0)))
        ang2 = ang if ang <= 180.0 else 360.0 - ang
        # 与南北向夹角 = |ang2 - 90|
        for b in boxes:
            lon_a, lon_b = b['lon']
            lat_a, lat_b = b['lat']
            if not (lon_a <= mx <= lon_b and lat_a <= my <= lat_b):
                continue
            if length < float(b.get('min_length_deg', 0.6)):
                continue
            tol = float(b.get('near_meridian_tol_deg', 35.0))
            if abs(ang2 - 90.0) <= tol:
                return True
        return False

    def _filter_edges_drop_outer_rims(
        self,
        coords: Sequence[Tuple[float, float]],
        union,
        region: Sequence[float],
        rim_probe: float,
        frame_buf: float,
        drop_south: bool = True,
        drop_north: bool = False,
        connector_boxes: Optional[Sequence[Dict]] = None,
    ) -> List[List[Tuple[float, float]]]:
        """保留省界，但去掉贴图框边、南北最外侧数据边界、以及指定连接直线"""
        if len(coords) < 2 or union is None or union.is_empty:
            return []
        lon0, lon1, lat0, lat1 = [float(x) for x in region]
        frame = box(lon0, lat0, lon1, lat1).boundary.buffer(frame_buf)
        boxes = list(connector_boxes or [])

        def drop_edge(x0: float, y0: float, x1: float, y1: float) -> bool:
            seg = LineString([(x0, y0), (x1, y1)])
            if seg.is_empty:
                return True
            if seg.intersects(frame):
                if seg.difference(frame).length < 0.3 * max(seg.length, 1e-12):
                    return True
            if boxes and self._edge_in_connector_box(x0, y0, x1, y1, boxes):
                return True
            mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            south_out = not union.contains(Point(mx, my - rim_probe))
            north_out = not union.contains(Point(mx, my + rim_probe))
            south_in = union.contains(Point(mx, my - rim_probe))
            north_in = union.contains(Point(mx, my + rim_probe))
            if drop_south and south_out and north_in:
                return True
            if drop_north and north_out and south_in:
                return True
            return False

        segs: List[List[Tuple[float, float]]] = []
        cur: List[Tuple[float, float]] = [coords[0]]
        for i in range(1, len(coords)):
            x0, y0 = cur[-1]
            x1, y1 = coords[i]
            if drop_edge(x0, y0, x1, y1):
                if len(cur) >= 2:
                    segs.append(cur)
                cur = [(x1, y1)]
            else:
                cur.append((x1, y1))
        if len(cur) >= 2:
            segs.append(cur)
        return segs

    def _province_filter_for(self, dataset: GeologyDataset) -> Dict[str, Any]:
        """合并通用过滤参数与按数据集覆盖项"""
        base = dict(self.config.style.get('province_filter') or {})
        by_key = self.config.style.get('province_filter_by_key') or {}
        overrides = dict(by_key.get(dataset.key) or {})
        merged = {**base, **overrides}
        if 'cache_name' not in merged:
            merged['cache_name'] = f'{dataset.key}_filtered.gmt'
        return merged

    def prepare_province_gmt(
        self,
        dataset: GeologyDataset,
        region: Sequence[float],
        force: bool = False,
    ) -> Path:
        """
        地质省线划（prv1ec / prv3bl）：
        - 轻度过滤（过大/跨日界线/OCS）
        - 去掉覆盖区最南/最北侧外轮廓（数据框假边界），其余省界保留
        """
        src = self.geology_dir / dataset.gmt_file
        if not src.exists():
            raise FileNotFoundError(f'找不到地质省数据: {src}')

        filt = self._province_filter_for(dataset)
        if not filt.get('enabled', True):
            return src

        # 缓存名带区域标签，避免不同制图范围共用错误裁剪结果
        lon0, lon1, lat0, lat1 = [float(x) for x in region]
        base_name = str(filt['cache_name'])
        stem = Path(base_name).stem
        region_tag = f'{lon0:.0f}_{lon1:.0f}_{lat0:.0f}_{lat1:.0f}'
        cache = self.geology_dir / f'{stem}_{region_tag}.gmt'
        if cache.exists() and not force and cache.stat().st_mtime >= src.stat().st_mtime:
            self.logger.info(f'  ✅ 地质省缓存 [{dataset.key}]: {cache.name}')
            return cache

        clip = box(lon0, lat0, lon1, lat1)
        max_lon_span = float(filt.get('max_lon_span', 55.0))
        max_lat_span = float(filt.get('max_lat_span', 35.0))
        min_frac = float(filt.get('min_frac_in_region', 0.25))
        keywords = [k.upper() for k in filt.get('exclude_name_keywords', ['OCS'])]
        drop_south = bool(filt.get('drop_south_outer_rim', True))
        drop_north = bool(filt.get('drop_north_outer_rim', False))
        # 兼容旧配置键 south_probe_deg
        rim_probe = float(
            filt.get('rim_probe_deg', filt.get('south_probe_deg', 0.20))
        )
        frame_buf = float(filt.get('frame_buffer_deg', 0.05))
        connector_boxes = list(filt.get('drop_connector_boxes') or [])

        text = src.read_text(encoding='utf-8', errors='replace')
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

        if not segments:
            self.logger.warning(f'  ⚠️ {dataset.key} 分段失败，使用原始文件')
            return src

        clipped_polys: List = []
        n_total = 0
        for seg in segments:
            n_total += 1
            body = ''.join(seg)
            name_m = re.search(r'# @D[^|]*\|[^|]*\|[^|]*\|"([^"]*)"', body)
            name = name_m.group(1) if name_m else ''
            if any(k in name.upper() for k in keywords):
                continue

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
            if len(coords) < 3:
                continue

            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            if max(lons) - min(lons) > max_lon_span:
                continue
            if max(lats) - min(lats) > max_lat_span:
                continue
            if any(abs(lons[i] - lons[i - 1]) > 180 for i in range(1, len(lons))):
                continue
            inside = sum(
                1 for lon, lat in coords
                if lon0 <= lon <= lon1 and lat0 <= lat <= lat1
            )
            if inside / len(coords) < min_frac:
                continue

            ring = coords if coords[0] == coords[-1] else coords + [coords[0]]
            try:
                poly = Polygon(ring)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                inter = poly.intersection(clip)
            except Exception:
                continue
            if inter.is_empty:
                continue
            if isinstance(inter, Polygon):
                clipped_polys.append(inter)
            elif isinstance(inter, MultiPolygon):
                clipped_polys.extend(list(inter.geoms))

        if not clipped_polys:
            self.logger.warning('  ⚠️ 无有效地质省多边形')
            return src

        union = unary_union(clipped_polys)
        line_segs: List[List[Tuple[float, float]]] = []
        for poly in clipped_polys:
            for line in self._iter_lines(poly.boundary):
                xy = [(float(x), float(y)) for x, y in line.coords]
                if len(xy) < 2:
                    continue
                if drop_south or drop_north or connector_boxes:
                    line_segs.extend(
                        self._filter_edges_drop_outer_rims(
                            xy,
                            union,
                            region,
                            rim_probe,
                            frame_buf,
                            drop_south=drop_south,
                            drop_north=drop_north,
                            connector_boxes=connector_boxes,
                        )
                    )
                else:
                    line_segs.append(xy)

        rim_note = []
        if drop_south:
            rim_note.append('south')
        if drop_north:
            rim_note.append('north')
        rim_txt = '+'.join(rim_note) if rim_note else 'none'
        self._write_line_gmt(
            cache,
            line_segs,
            f'# {dataset.key} province lines; dropped outer rims: {rim_txt}\n',
        )
        self.logger.info(
            f'  ✅ 地质省线划 [{dataset.key}]: 多边形 '
            f'{len(clipped_polys)}/{n_total}, '
            f'线段 {len(line_segs)}（已去外轮廓: {rim_txt}）→ {cache.name}'
        )
        return cache

    def _draw_province(
        self,
        fig: pygmt.Figure,
        dataset: GeologyDataset,
        region: Sequence[float],
    ) -> None:
        gmt_path = self.geology_dir / dataset.gmt_file
        if not gmt_path.exists():
            self.logger.warning(f'  ⚠️ 缺少地质省数据: {gmt_path.name}')
            return
        try:
            gmt_path = self.prepare_province_gmt(dataset, region, force=False)
        except Exception as e:
            self.logger.warning(
                f'  ⚠️ 地质省过滤失败 [{dataset.key}]，用原文件: {e}'
            )

        pen = dataset.pen or self.config.style['province_pen']
        self.logger.info(
            f'  地质省边界 [{dataset.key}]: {gmt_path.name} ({pen})'
        )
        fig.plot(
            data=str(gmt_path),
            region=list(region),
            projection=self.projection,
            pen=pen,
            close=False,
        )

    def _draw_cn_blocks(
        self,
        fig: pygmt.Figure,
        region: Sequence[float],
    ) -> None:
        """叠绘中国及邻区活动地块边界（原始线划）"""
        layers = self.block_layers
        if not layers:
            if self.block_file.exists():
                layers = [{
                    'file': self.block_file.name,
                    'pen': self.config.style['block_pen'],
                    'label': 'CN block',
                }]
            else:
                self.logger.warning('  ⚠️ 未配置地块文件')
                return

        for layer in layers:
            path = self.geology_dir / layer['file']
            if not path.exists():
                self.logger.warning(f'  ⚠️ 缺少地块数据: {path.name}')
                continue
            pen = layer.get('pen', self.config.style['block_pen'])
            label = layer.get('label', path.stem)
            self.logger.info(f'  地块边界: {path.name} ({pen}) [{label}]')
            fig.plot(
                data=str(path),
                region=list(region),
                projection=self.projection,
                pen=pen,
            )

    def plot_dataset(
        self,
        key: str,
        force_convert: bool = False,
        show: bool = False,
    ) -> pygmt.Figure:
        """绘制单幅数据集"""
        if key == 'composite':
            return self.plot_composite(force_convert=force_convert, show=show)

        if key not in self.datasets:
            raise KeyError(f'未知数据集: {key}')

        ds = self.datasets[key]
        cmap = self._resolve_cmap()
        legend = (
            self.prepare_legend() if self.config.layers['legend'] else None
        )

        self.logger.info(f'🎨 绘制 {ds.key}: {ds.title}')
        fig = pygmt.Figure()
        self._draw_basemap(fig, ds.region)

        if ds.kind == 'age' and self.config.layers['geology']:
            self._draw_age_layer(fig, ds, ds.region, cmap, force_convert)
            self._draw_ocean_mask(fig, ds.region)
        elif ds.kind == 'province':
            self._draw_province(fig, ds, ds.region)
            self._draw_ocean_mask(fig, ds.region)

        if self.config.layers['cn_blocks']:
            self._draw_cn_blocks(fig, ds.region)

        if self.config.layers['title']:
            fig.text(
                text=ds.title,
                position='TC',
                offset='0/0.4c',
                font='12p,Helvetica-Bold,black',
                no_clip=True,
            )

        # 图例最后绘制（shift_origin 下移，须在地图/标题之后）
        if (
            self.config.layers['legend']
            and legend is not None
            and ds.kind == 'age'
        ):
            self._add_legend(fig, legend)

        self._save_figure(fig, ds.description)
        if show:
            fig.show()
        self.logger.info(f'✅ 完成 {ds.key}')
        return fig

    def get_model_overlap_region(self) -> List[float]:
        """
        读取与 2_3_Model_clustering 相同的三个模型 original.nc，
        返回经纬度公共范围 [lon_min, lon_max, lat_min, lat_max]。
        """
        try:
            import xarray as xr
        except ImportError as e:
            raise ImportError('计算模型公共范围需要 xarray') from e

        cfg = self.config.model_overlap
        pattern = cfg['netcdf_pattern']
        data_root = self.base_config.dirs['data']
        lon_mins: List[float] = []
        lon_maxs: List[float] = []
        lat_mins: List[float] = []
        lat_maxs: List[float] = []

        for model_name in cfg['models']:
            nc_path = data_root / pattern.format(model_name=model_name)
            if not nc_path.exists():
                raise FileNotFoundError(f'找不到模型 NetCDF: {nc_path}')
            with xr.open_dataset(nc_path) as ds:
                lons = ds['longitude'].values
                lats = ds['latitude'].values
                lon_mins.append(float(lons.min()))
                lon_maxs.append(float(lons.max()))
                lat_mins.append(float(lats.min()))
                lat_maxs.append(float(lats.max()))
            self.logger.info(
                f'  模型范围 {model_name}: '
                f'lon[{lon_mins[-1]:.2f},{lon_maxs[-1]:.2f}] '
                f'lat[{lat_mins[-1]:.2f},{lat_maxs[-1]:.2f}]'
            )

        region = [
            max(lon_mins),
            min(lon_maxs),
            max(lat_mins),
            min(lat_maxs),
        ]
        if region[0] >= region[1] or region[2] >= region[3]:
            raise ValueError(f'三模型无有效公共范围: {region}')
        self.logger.info(
            f'  ✅ 三模型公共范围: lon[{region[0]:.2f},{region[1]:.2f}] '
            f'lat[{region[2]:.2f},{region[3]:.2f}]'
        )
        return region

    def render_composite_png_for_embed(
        self,
        region: Sequence[float],
        out_path: Path,
        dpi: int = 200,
        force_convert: bool = False,
        map_width_cm: float = 16.0,
    ) -> Path:
        """
        渲染拼合基岩地质图 PNG，供 matplotlib 子图 imshow 嵌入。

        - 无图例、无标题
        - 使用线性经纬度投影 (X)，与 PlateCarree + aspect=auto 对齐
        - 图框无标注（由外层 axes 负责刻度/海岸线）

        Args:
            region: [lon_min, lon_max, lat_min, lat_max]
            out_path: 输出 PNG 路径
            dpi: 分辨率
            force_convert: 是否强制重建 age .dat
            map_width_cm: 地图宽度（cm）；高度按区域纵横比推算

        Returns:
            实际写入的 PNG 路径
        """
        region_list = [float(x) for x in region]
        lon0, lon1, lat0, lat1 = region_list
        if lon1 <= lon0 or lat1 <= lat0:
            raise ValueError(f'无效区域: {region_list}')

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 线性投影：经度跨度 / 纬度跨度 → 宽高比
        aspect = (lon1 - lon0) / (lat1 - lat0)
        map_h = map_width_cm / aspect
        saved_proj = self.projection
        saved_frame = list(self.config.projection.get('frame', []))
        saved_layers = dict(self.config.layers)

        try:
            self.projection = f'X{map_width_cm:.2f}c/{map_h:.2f}c'
            # 仅画框，无标注（避免与外层 cartopy 刻度重复）
            self.config.projection['frame'] = ['wesn']
            self.config.layers['legend'] = False
            self.config.layers['title'] = False

            cmap = self._resolve_cmap()
            order = getattr(
                self.config,
                'composite_age_order',
                ['geo1ec', 'geo3bl', 'geo8apg', 'geo3al'],
            )

            self.logger.info(
                f'🎨 嵌入用地质底图: region={region_list}, '
                f'proj={self.projection} → {out_path.name}'
            )
            fig = pygmt.Figure()
            self._draw_basemap(fig, region_list)

            # 底层水深 → 年代地质 → 仅洋区再叠水深（盖住伸入海区的地质面）
            used_bathy = self._draw_ocean_bathymetry(fig, region_list)

            for key in order:
                if key not in self.datasets:
                    continue
                ds = self.datasets[key]
                if ds.kind != 'age' or not ds.use_in_composite:
                    continue
                if not (self.geology_dir / ds.gmt_file).exists():
                    self.logger.warning(f'  ⚠️ 跳过缺失: {ds.gmt_file}')
                    continue
                self._draw_age_layer(fig, ds, region_list, cmap, force_convert)

            if used_bathy:
                self._overlay_ocean_bathymetry(fig, region_list)
            else:
                self._draw_ocean_mask(fig, region_list)

            if self.config.layers.get('province_overlay', True):
                for pkey in self.config.province_overlay_keys:
                    if pkey not in self.datasets:
                        continue
                    self._draw_province(fig, self.datasets[pkey], region_list)

            if self.config.layers.get('cn_blocks', True):
                self._draw_cn_blocks(fig, region_list)

            fig.savefig(str(out_path), dpi=dpi, crop=True)
            self.logger.info(f'  💾 嵌入底图: {out_path}')
            return out_path
        finally:
            self.projection = saved_proj
            self.config.projection['frame'] = saved_frame
            self.config.layers.clear()
            self.config.layers.update(saved_layers)

    def plot_composite(
        self,
        force_convert: bool = False,
        show: bool = False,
        region: Optional[Sequence[float]] = None,
        description: str = 'eastasia_geology',
    ) -> pygmt.Figure:
        """
        四幅年代地质图拼合。

        Args:
            region: 制图范围 [lon_min, lon_max, lat_min, lat_max]；
                    默认使用 BaseConfig / Basemap 区域。
            description: 输出文件名后缀（经 generate_filename）
        """
        region_list = list(region) if region is not None else list(self.region)
        cmap = self._resolve_cmap()
        legend = (
            self.prepare_legend() if self.config.layers['legend'] else None
        )
        order = getattr(
            self.config,
            'composite_age_order',
            ['geo1ec', 'geo3bl', 'geo8apg', 'geo3al'],
        )

        self.logger.info(
            f'🎨 绘制拼合地质图 [{description}]: '
            f'region={region_list}; layers=' + ' → '.join(order)
        )
        fig = pygmt.Figure()
        self._draw_basemap(fig, region_list)

        for key in order:
            if key not in self.datasets:
                self.logger.warning(f'  ⚠️ 未知数据集: {key}')
                continue
            ds = self.datasets[key]
            if ds.kind != 'age' or not ds.use_in_composite:
                continue
            if not (self.geology_dir / ds.gmt_file).exists():
                self.logger.warning(f'  ⚠️ 跳过缺失: {ds.gmt_file}')
                continue
            self._draw_age_layer(fig, ds, region_list, cmap, force_convert)

        self._draw_ocean_mask(fig, region_list)

        if self.config.layers['province_overlay']:
            for pkey in self.config.province_overlay_keys:
                if pkey not in self.datasets:
                    self.logger.warning(f'  ⚠️ 未知地质省数据集: {pkey}')
                    continue
                self._draw_province(fig, self.datasets[pkey], region_list)

        if self.config.layers['cn_blocks']:
            self._draw_cn_blocks(fig, region_list)

        # 图例最后绘制（会 shift_origin，须放在地图与标题之后）
        if self.config.layers['legend'] and legend is not None:
            self._add_legend(fig, legend)

        self._save_figure(fig, description)
        if show:
            fig.show()
        self.logger.info(f'✅ 拼合地质图完成 → {description}')
        return fig

    def _save_figure(
        self, fig: pygmt.Figure, description: str
    ) -> Dict[str, Path]:
        saved: Dict[str, Path] = {}
        out_cfg = self.config.output
        for fmt, dpi_key, flag in (
            ('jpg', 'dpi_jpg', 'save_jpg'),
            ('pdf', 'dpi_pdf', 'save_pdf'),
        ):
            if not out_cfg[flag]:
                continue
            name = generate_filename(self.module_prefix, description, fmt)
            path = self.output_dir / name
            fig.savefig(
                str(path), dpi=out_cfg[dpi_key], crop=out_cfg['crop']
            )
            saved[fmt] = path
            self.logger.info(f'  💾 {path.name}')
        return saved

    def run(
        self,
        dataset_keys: Optional[Sequence[str]] = None,
        force_convert: bool = False,
        with_blocks: bool = False,
        show: bool = False,
    ) -> Dict[str, pygmt.Figure]:
        """批量绘制"""
        if with_blocks:
            self.config.layers['cn_blocks'] = True
        keys = list(dataset_keys or self.config.datasets_to_plot)
        figs: Dict[str, pygmt.Figure] = {}
        for key in keys:
            try:
                figs[key] = self.plot_dataset(
                    key, force_convert=force_convert, show=show
                )
            except Exception as e:
                self.logger.error(f'❌ {key} 失败: {e}')
                raise
        return figs


def main() -> int:
    """主函数：东亚全图 + 三模型公共范围地质图"""
    print('🎯 EASTASIA-FWI 东亚拼合地质图 (5_9)')
    print('    geo1ec + geo8apg + geo3bl + geo3al + blocks/provinces')
    print('=' * 60)
    try:
        plotter = EastAsiaGeologyMap()
        plotter.config.layers['cn_blocks'] = True

        # 1) 与 Basemap 同范围的全图
        plotter.plot_composite(
            force_convert=False,
            show=False,
            description='eastasia_geology',
        )

        # 2) SinoScope / EARA2024 / FWEA23 公共范围
        overlap = plotter.get_model_overlap_region()
        desc = plotter.config.model_overlap['description']
        plotter.plot_composite(
            force_convert=False,
            show=False,
            region=overlap,
            description=desc,
        )

        print(f'\n✅ 完成 → {plotter.output_dir}')
        print('  全图: 5-9_eastasia_geology.jpg / .pdf')
        print(f'  模型公共范围: 5-9_{desc}.jpg / .pdf')
        print(
            f'  公共范围: lon[{overlap[0]:.1f},{overlap[1]:.1f}] '
            f'lat[{overlap[2]:.1f},{overlap[3]:.1f}]'
        )
        print('  边界: CN-block + prv1ec + prv3bl（细红色实线）')
        print('=' * 60)
        return 0
    except KeyboardInterrupt:
        print('\n⚠️ 用户中断')
        return 1
    except Exception as e:
        print(f'\n❌ 失败: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
