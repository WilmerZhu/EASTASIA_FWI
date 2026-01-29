"""
5_1_Basemap.py: 
东亚地区基础地图绘制模块
基于PyGMT绘制东亚地区基础地形图
包含地形、海底年龄、火山、断层和板块边界
配置参数已集成在模块内部

功能特性：
- 高分辨率地形绘制
- 海底年龄可视化
- 火山分布显示
- 断层和板块边界
- 俯冲带等深线
- 多种色标显示
"""
import pygmt
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig

# 导入可视化模块工具函数（从模块内 utils 目录导入）
sys.path.insert(0, str(Path(__file__).parent))
from utils.file_utils import extract_module_prefix, generate_filename


class BasemapConfig:
    """基础地图绘制配置类 - 集成在模块内部"""
    
    def __init__(self):
        """初始化基础地图绘制配置参数"""
        # PyGMT配置
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
            'map_label_offset': "5p"
        }
        
        # 地图投影和尺寸配置
        self.projection = {
            'type': "M",           # 墨卡托投影
            'width': "15c",        # 地图宽度
            'frame_interval': "a10f1"  # 框架间隔
        }
        
        # 数据源配置
        self.data_sources = {
            'elevation': "@earth_relief_01m",
            'seafloor_age': "@earth_age_01m_g",
            'ridge': "@ridge.txt",
            'data_resolution': "01m"
        }
        
        # 色彩配置
        self.colors = {
            'topo_cmap': "terra",
            'topo_series': "-10000/8000",
            'age_cmap': "seis", 
            'age_series': "0/200/10",
            'slab_cmap': "viridis",
            'slab_series': "-700/0/20",
            'volcano_color': "red",
            'fault_pen': "0.5p,black",
            'boundary_pen': "1.0p,red",
            'slab_pen': "0.35p"
        }
        
        # 图层显示配置
        self.layers = {
            'include_seafloor': True,
            'include_volcanoes': True,
            'include_boundaries': True,
            'include_faults': False,  # 默认关闭活动断层
            'include_slabs': True,
            'include_colorbars': True
        }
        
        # 输出配置
        self.output = {
            'dpi_jpg': 300,
            'dpi_pdf': 300,
            'crop': True,
            'save_jpg': True,
            'save_pdf': True
        }
        
        # 数据目录配置（相对于可视化模块）
        self.data_dirs = {
            'base': 'data_EastAsia',
            'elevation': 'elevation',
            'tectonics': 'tectonics',
            'volcano': 'volcano',
            'slab': 'Slab2',
            'faults': 'gem-global-active-faults-master'
        }
        
        # 色标条配置
        self.colorbar = {
            'topo_position': "JBL+o-4.5c/0.5c+w4c/0.3c+h",
            'topo_frame': ["x+lElevation", "y+lm"],
            'age_position': "JBC+o0c/0.5c+w4c/0.3c+h",
            'age_frame': ["a+lSeafloor Age", "y+lMa"],
            'slab_position': "JBR+o-4.5c/0.5c+w4c/0.3c+h",
            'slab_frame': ["x+lSlab Depth", "y+lkm"],
            'simple_position': "JMR+o1c/0c+w8c/0.5c"
        }
        
        # 火山数据配置
        self.volcano = {
            'style': "kvolcano/0.25c",
            'fill': "red",
            'pen': "0.1p,black",
            'default_data': {
                'longitude': [120.5, 121.0, 122.5, 130.8, 138.7, 140.9, 142.5, 127.3, 128.1, 129.5, 131.2, 133.8, 135.4, 137.1],
                'latitude': [23.8, 25.2, 24.8, 31.6, 35.4, 38.1, 40.5, 33.2, 34.7, 36.1, 37.9, 39.3, 41.2, 42.8]
            }
        }
        
        # 日志配置
        self.logging = {
            'level': 'INFO',
            'console_output': True
        }


class EastAsiaBasemap:
    """东亚地区基础地图绘制类
    
    专门用于绘制东亚地区的基础地形图和地球物理要素
    """
    
    def __init__(self, config_file: Optional[Path] = None, output_dir: Optional[str] = None):
        """
        初始化东亚基础地图绘制器
        
        Args:
            config_file: 配置文件路径
            output_dir: 输出目录
        """
        # 加载基础配置
        self.base_config = BaseConfig(config_file)
        
        # 加载模块配置
        self.config = BasemapConfig()
        
        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.base_config.dirs['figures']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.BasemapPlotter',
            self.config.logging['level']
        )
        
        # 设置地图区域 - 使用基础配置中的东亚区域
        self.region = [
            self.base_config.region['lon_min'], 
            self.base_config.region['lon_max'],
            self.base_config.region['lat_min'], 
            self.base_config.region['lat_max']
        ]
        
        # 地图投影
        self.projection = f"{self.config.projection['type']}{self.config.projection['width']}"
        
        # 设置数据目录
        self._setup_data_directories()
        
        # 设置PyGMT配置
        self._setup_pygmt_config()
        
        # 自动提取模块号（从文件名如 5_1_Basemap.py 提取 5-1）
        self.module_prefix = extract_module_prefix(Path(__file__), default="5-1")
        
        self.logger.info("🗺️ 东亚基础地图绘制器初始化完成")
        self._print_config_summary()

    def _setup_data_directories(self):
        """设置数据目录结构"""
        vis_root = self.base_config.dirs['project_root'] / '5_Visualization'
        data_base = vis_root / self.config.data_dirs['base']
        
        self.data_dirs = {
            'base': data_base,
            'elevation': data_base / self.config.data_dirs['elevation'],
            'tectonics': data_base / self.config.data_dirs['tectonics'],
            'volcano': data_base / self.config.data_dirs['volcano'],
            'slab': data_base / self.config.data_dirs['slab'],
            'faults': data_base / self.config.data_dirs['faults']
        }
        
        # 创建数据目录
        for dir_path in self.data_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 基础地图绘制配置摘要")
        print("-" * 60)
        
        # 研究区域配置
        print(f"研究区域: {self.base_config.region['name']}")
        region = self.base_config.region
        print(f"纬度范围: {region['lat_min']}° ~ {region['lat_max']}°")
        print(f"经度范围: {region['lon_min']}° ~ {region['lon_max']}°")
        
        # 投影和分辨率配置
        print(f"地图投影: {self.projection}")
        print(f"数据分辨率: {self.config.data_sources['data_resolution']}")
        
        # 图层配置
        layers = self.config.layers
        print(f"图层配置:")
        print(f"  海底年龄: {'开启' if layers['include_seafloor'] else '关闭'}")
        print(f"  火山分布: {'开启' if layers['include_volcanoes'] else '关闭'}")
        print(f"  板块边界: {'开启' if layers['include_boundaries'] else '关闭'}")
        print(f"  活动断层: {'开启' if layers['include_faults'] else '关闭'}")
        print(f"  俯冲带: {'开启' if layers['include_slabs'] else '关闭'}")
        
        # 输出配置
        print(f"输出目录: {self.output_dir}")
        print(f"图片分辨率: jpg {self.config.output['dpi_jpg']}dpi, PDF {self.config.output['dpi_pdf']}dpi")
        
        print("-" * 60)

    def _setup_pygmt_config(self):
        """设置PyGMT全局配置"""
        pygmt_config = self.config.pygmt
        pygmt.config(
            MAP_FRAME_TYPE=pygmt_config['map_frame_type'],
            MAP_GRID_PEN_PRIMARY=pygmt_config['map_grid_pen_primary'],
            MAP_ANNOT_OBLIQUE=pygmt_config['map_annot_oblique'],
            MAP_ANNOT_OFFSET_PRIMARY=pygmt_config['map_annot_offset_primary'],
            MAP_ANNOT_OFFSET_SECONDARY=pygmt_config['map_annot_offset_secondary'],
            FONT_ANNOT_PRIMARY=pygmt_config['font_annot_primary'],
            FONT_LABEL=pygmt_config['font_label'],
            MAP_FRAME_WIDTH=pygmt_config['map_frame_width'],
            MAP_FRAME_PEN=pygmt_config['map_frame_pen'],
            MAP_TICK_LENGTH_PRIMARY=pygmt_config['map_tick_length_primary'],
            MAP_TICK_PEN_PRIMARY=pygmt_config['map_tick_pen_primary'],
            MAP_LABEL_OFFSET=pygmt_config['map_label_offset'],
        )
    

    def prepare_elevation_data(self) -> str:
        """准备地形数据"""
        elevation_file = self.data_dirs['elevation'] / "EastAsia.grd"
        
        try:
            if not elevation_file.exists():
                self.logger.info("📊 准备地形数据...")
                # 裁切全球地形数据到东亚区域
                pygmt.grdcut(
                    grid=self.config.data_sources['elevation'],
                    region=self.region,
                    outgrid=str(elevation_file),
                )
                self.logger.info(f"✅ 地形数据已保存: {elevation_file}")
            else:
                self.logger.info(f"📊 使用现有地形数据: {elevation_file}")
                
            return str(elevation_file)
            
        except Exception as e:
            self.logger.error(f"地形数据准备失败: {e}")
            return self.config.data_sources['elevation']  # 降级使用全球数据

    def create_colormaps(self):
        """创建颜色映射文件"""
        try:
            colors = self.config.colors
            
            # 地形色标
            topo_cpt = self.data_dirs['elevation'] / "EastAsiaTopo.cpt"
            pygmt.makecpt(
                cmap=colors['topo_cmap'],
                series=colors['topo_series'],
                output=str(topo_cpt),
            )
            
            # 海底年龄色标
            age_cpt = self.data_dirs['tectonics'] / "OC_Age.cpt"
            pygmt.makecpt(
                cmap=colors['age_cmap'],
                series=colors['age_series'],
                output=str(age_cpt),
            )
            
            # 俯冲带深度色标
            slab_cpt = self.data_dirs['slab'] / "slab_depth.cpt"
            pygmt.makecpt(
                cmap=colors['slab_cmap'],
                series=colors['slab_series'],
                output=str(slab_cpt),
            )
            
            self.logger.info("🎨 色标文件创建完成")
            
            return {
                'topo': str(topo_cpt),
                'age': str(age_cpt),
                'slab': str(slab_cpt)
            }
            
        except Exception as e:
            self.logger.error(f"色标创建失败: {e}")
            return {
                'topo': self.config.colors['topo_cmap'],
                'age': self.config.colors['age_cmap'],
                'slab': self.config.colors['slab_cmap']
            }

    def plot_basic_basemap(self) -> pygmt.Figure:
        """绘制基础底图"""
        self.logger.info("🗺️ 开始绘制基础底图...")
        
        # 准备数据
        elevation_file = self.prepare_elevation_data()
        colormaps = self.create_colormaps()
        
        # 创建图形
        fig = pygmt.Figure()
        
        # 绘制地形
        try:
            fig.grdimage(
                grid=elevation_file,
                shading=True,
                region=self.region,
                projection=self.projection,
                frame=["WseN",self.config.projection['frame_interval']],
                cmap=colormaps['topo'],
            )
            self.logger.info("✅ 地形图层绘制完成")
        except Exception as e:
            self.logger.error(f"地形绘制失败: {e}")
            # 降级处理
            fig.coast(
                region=self.region,
                projection=self.projection,
                land="lightgray",
                water="lightblue",
                frame=[self.config.projection['frame_interval']]
            )
        
        return fig

    def add_seafloor_age(self, fig: pygmt.Figure, colormaps: Dict[str, str]) -> pygmt.Figure:
        """添加海底年龄图层"""
        try:
            self.logger.info("🌊 添加海底年龄图层...")
            
            # 使用GMT自带数据
            seafloor_grid = self.config.data_sources['seafloor_age']
            
            fig.grdimage(
                grid=seafloor_grid,
                shading=self.config.data_sources['elevation'] + "+d",
                region=self.region,
                cmap=colormaps['age'],
                nan_transparent=True,
            )
            
            self.logger.info("✅ 海底年龄图层添加完成")
            
        except Exception as e:
            self.logger.warning(f"海底年龄图层添加失败: {e}")
        
        return fig

    def add_volcanoes(self, fig: pygmt.Figure) -> pygmt.Figure:
        """添加火山数据"""
        try:
            # 查找火山数据文件
            volcano_files = [
                self.data_dirs['volcano'] / "VHolo.csv",
                self.data_dirs['volcano'] / "EastAsia_volcanoes.csv"
            ]
            
            volcano_file = None
            for vf in volcano_files:
                if vf.exists():
                    volcano_file = vf
                    break
            
            if not volcano_file:
                # 创建示例火山数据
                volcano_file = self.data_dirs['volcano'] / "VHolo.csv"
                volcano_data = self.config.volcano['default_data']
                pd.DataFrame(volcano_data).to_csv(volcano_file, index=False)
                self.logger.info(f"📄 创建示例火山数据: {volcano_file}")
            
            # 绘制火山
            volcano_config = self.config.volcano
            fig.plot(
                data=str(volcano_file),
                style=volcano_config['style'],
                fill=volcano_config['fill'],
                pen=volcano_config['pen'],
            )
            
            self.logger.info("🌋 火山图层添加完成")
            
        except Exception as e:
            self.logger.warning(f"火山数据添加失败: {e}")
        
        return fig

    def add_plate_boundaries(self, fig: pygmt.Figure) -> pygmt.Figure:
        """添加板块边界"""
        try:
            self.logger.info("🔄 添加板块边界...")
            
            # 优先使用本地boundaries.gmt文件
            boundaries_file = self.data_dirs['tectonics'] / "boundaries.gmt"
            
            if boundaries_file.exists():
                fig.plot(
                    data=str(boundaries_file),
                    pen=self.config.colors['fault_pen'],
                )
                self.logger.info(f"✅ 使用本地板块边界数据: {boundaries_file}")
            else:
                # 降级处理：使用GMT内置数据
                try:
                    fig.plot(
                        data=self.config.data_sources['ridge'],
                        pen=self.config.colors['boundary_pen'],
                        region=self.region
                    )
                    self.logger.info("✅ 使用GMT内置洋脊数据")
                except:
                    # 最后降级：只添加海岸线和国界
                    fig.coast(
                        borders="1/0.5p,black",
                        shorelines="1/0.3p,black"
                    )
                    self.logger.info("✅ 降级使用海岸线和国界")
            
        except Exception as e:
            self.logger.warning(f"板块边界添加失败: {e}")
        
        return fig

    def add_active_faults(self, fig: pygmt.Figure) -> pygmt.Figure:
        """添加活动断层"""
        try:
            self.logger.info("🔗 添加活动断层...")
            
            # 查找活动断层文件
            fault_files = [
                self.data_dirs['faults'] / "gem_active_faults.txt",
                self.data_dirs['faults'] / "gem_active_faults.gmt"
            ]
            
            fault_file = None
            for ff in fault_files:
                if ff.exists():
                    fault_file = ff
                    break
            
            if fault_file:
                fig.plot(
                    data=str(fault_file),
                    pen=self.config.colors['fault_pen'],
                )
                self.logger.info(f"✅ 活动断层已添加: {fault_file}")
            else:
                self.logger.warning("⚠️ 活动断层文件不存在")
            
        except Exception as e:
            self.logger.warning(f"活动断层添加失败: {e}")
        
        return fig

    def add_slab_contours(self, fig: pygmt.Figure, colormaps: Dict[str, str]) -> pygmt.Figure:
        """添加俯冲带等深线 - 按深度着色"""
        try:
            self.logger.info("📐 添加俯冲带等深线...")
            
            # 查找俯冲带数据文件
            slab_contours_dir = self.data_dirs['slab'] / "Slab2_CONTOURS"
            slab_files = []
            
            if slab_contours_dir.exists():
                slab_files = list(slab_contours_dir.glob("*_slab2_dep_*.in"))
                
            if not slab_files:
                # 尝试在主slab目录查找
                slab_files = list(self.data_dirs['slab'].glob("*_slab2_dep_*.in"))
            
            if not slab_files:
                # 创建示例俯冲带数据（包含深度值）
                sample_slab_file = self.data_dirs['slab'] / "sample_slab.txt"
                sample_data = """140.0 38.0 -100
141.0 39.0 -150
142.0 40.0 -200
143.0 41.0 -250
>
130.0 30.0 -50
131.0 31.0 -100
132.0 32.0 -150
133.0 33.0 -200
>
150.0 45.0 -80
151.0 46.0 -120
152.0 47.0 -160
153.0 48.0 -200"""
                
                with open(sample_slab_file, 'w') as f:
                    f.write(sample_data)
                slab_files = [sample_slab_file]
                self.logger.info("📄 创建示例俯冲带数据")
            
            # 绘制俯冲带等深线
            success_count = 0
            slab_pen = self.config.colors['slab_pen']
            
            for slab_file in slab_files:
                try:
                    # 对于 .in 文件，使用深度着色
                    if str(slab_file).endswith('.in'):
                        fig.plot(
                            data=str(slab_file),
                            incols=[0, 1, 2],  # lon, lat, depth
                            pen=f"{slab_pen}+z",      # +z 表示按第三列（深度）着色
                            cmap=colormaps['slab'],
                        )
                        success_count += 1
                        self.logger.debug(f"使用深度着色绘制: {slab_file.name}")
                    else:
                        # 对于其他格式文件，手动解析
                        lons, lats, depths = [], [], []
                        with open(slab_file, 'r') as file:
                            for line in file:
                                line = line.strip()
                                if line.startswith('>'):
                                    if lons:  # 如果已有数据，添加分隔符
                                        lons.append(np.nan)
                                        lats.append(np.nan)
                                        depths.append(np.nan)
                                elif line and not line.startswith('#'):
                                    parts = line.split()
                                    if len(parts) >= 3:
                                        try:
                                            lons.append(float(parts[0]))
                                            lats.append(float(parts[1]))
                                            depths.append(float(parts[2]))
                                        except ValueError:
                                            continue
                        
                        # 绘制俯冲带线条（按深度着色）
                        if lons and lats and depths:
                            fig.plot(
                                x=lons,
                                y=lats,
                                z=depths,
                                pen=f"{slab_pen}+z",      # +z 表示按深度着色
                                cmap=colormaps['slab'],
                            )
                            success_count += 1
                            self.logger.debug(f"使用深度着色绘制: {slab_file.name}")
                        
                except Exception as e:
                    self.logger.debug(f"俯冲带文件 {slab_file.name} 处理失败: {e}")
                    continue
            
            self.logger.info(f"✅ 成功添加 {success_count} 个俯冲带等深线")
            
        except Exception as e:
            self.logger.warning(f"俯冲带等深线添加失败: {e}")
        
        return fig

    def add_colorbars(self, fig: pygmt.Figure, colormaps: Dict[str, str]) -> pygmt.Figure:
        """添加色标条"""
        try:
            self.logger.info("🎨 添加色标条...")
            
            colorbar_config = self.config.colorbar
            
            # 地形色标条（左下）
            fig.colorbar(
                position=colorbar_config['topo_position'],
                frame=colorbar_config['topo_frame'],
                cmap=colormaps['topo'],
            )
            
            # 海底年龄色标条（中下）
            fig.colorbar(
                frame=colorbar_config['age_frame'],
                position=colorbar_config['age_position'],
                cmap=colormaps['age'],
            )
            
            # 俯冲带深度色标条（右下）
            fig.colorbar(
                position=colorbar_config['slab_position'],
                frame=colorbar_config['slab_frame'],
                cmap=colormaps['slab'],
            )
            
            self.logger.info("✅ 色标条添加完成")
            
        except Exception as e:
            self.logger.warning(f"色标条添加失败: {e}")
        
        return fig

    def plot_comprehensive_basemap(self, 
                                #   title: str = "East Asia Geological and Tectonic Map",
                                  save_file: Optional[str] = None,
                                  **layer_options) -> str:
        """绘制完整的东亚基础地图"""
        
        self.logger.info("🎨 开始绘制完整的东亚基础地图...")
        
        # 如果没有指定文件名，自动生成
        if save_file is None:
            save_file = generate_filename(self.module_prefix, "comprehensive_basemap")
        
        # 合并默认图层配置和用户选项
        layers = self.config.layers.copy()
        layers.update(layer_options)
        
        try:
            # 创建色标
            colormaps = self.create_colormaps()
            
            # 绘制基础底图
            fig = self.plot_basic_basemap()
            
            # 添加海底年龄（如果启用）
            if layers.get('include_seafloor', True):
                fig = self.add_seafloor_age(fig, colormaps)
            
            # 添加火山（如果启用）
            if layers.get('include_volcanoes', True):
                fig = self.add_volcanoes(fig)
            
            # 添加活动断层（如果启用）
            if layers.get('include_faults', False):
                fig = self.add_active_faults(fig)
            
            # 添加板块边界（如果启用）
            if layers.get('include_boundaries', True):
                fig = self.add_plate_boundaries(fig)
            
            # 添加俯冲带等深线（如果启用）
            if layers.get('include_slabs', True):
                fig = self.add_slab_contours(fig, colormaps)
            
            # 添加色标条
            if layers.get('include_colorbars', True):
                fig = self.add_colorbars(fig, colormaps)
            
            # 保存图像
            output_config = self.config.output
            output_path = self.output_dir / save_file
            
            if output_config['save_jpg']:
                fig.savefig(str(output_path), dpi=output_config['dpi_jpg'], crop=output_config['crop'])
                self.logger.info(f"✅ jpg图像保存至: {output_path}")
            
            if output_config['save_pdf']:
                pdf_path = output_path.with_suffix('.pdf')
                fig.savefig(str(pdf_path), dpi=output_config['dpi_pdf'], crop=output_config['crop'])
                self.logger.info(f"✅ PDF版本保存至: {pdf_path}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"完整基础地图绘制失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def plot_simple_basemap(self, 
                        #    title: str = "East Asia Simple Basemap",
                           save_file: Optional[str] = None) -> str:
        """绘制简化版基础地图"""
        
        self.logger.info("🗺️ 绘制简化版基础地图...")
        
        # 如果没有指定文件名，自动生成
        if save_file is None:
            save_file = generate_filename(self.module_prefix, "simple_basemap")
        
        try:
            # 创建基础底图
            fig = self.plot_basic_basemap()
            
            # 只添加海岸线和边界
            fig.coast(
                shorelines="1/0.3p,black",
                borders="1/0.5p,black"
            )
            
            # 添加地形色标
            colormaps = self.create_colormaps()
            colorbar_config = self.config.colorbar
            fig.colorbar(
                position=colorbar_config['simple_position'],
                frame=colorbar_config['topo_frame'],
                cmap=colormaps['topo']
            )
            
            # 保存图像
            output_config = self.config.output
            output_path = self.output_dir / save_file
            
            fig.savefig(str(output_path), dpi=output_config['dpi_jpg'], crop=output_config['crop'])
            self.logger.info(f"✅ 简化基础地图保存至: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"简化基础地图绘制失败: {e}")
            return ""


def main():
    """主函数 - 东亚基础地图绘制"""
    print("🗺️ EASTASIA-FWI 东亚基础地图绘制系统")
    print("="*60)
    
    try:
        # 创建基础地图绘制器
        basemap_plotter = EastAsiaBasemap()
        
        print("🎨 开始绘制东亚基础地图...")
        
        # 绘制完整版基础地图
        print("\n🌍 绘制完整版地质构造图...")
        comprehensive_map = basemap_plotter.plot_comprehensive_basemap(
            # title="East Asia Geological and Tectonic Features",
            # save_file 参数可选，不指定则自动生成
            include_seafloor=True,
            include_volcanoes=True,
            include_boundaries=True,
            include_faults=False,  # 默认不包含活动断层
            include_slabs=True
        )
        
        if comprehensive_map:
            print(f"✅ 完整版基础地图: {Path(comprehensive_map).name}")
        
        # 绘制简化版基础地图
        print("\n🗺️ 绘制简化版基础地图...")
        simple_map = basemap_plotter.plot_simple_basemap(
            # title="East Asia Topographic Map",
            # save_file 参数可选，不指定则自动生成
        )
        
        if simple_map:
            print(f"✅ 简化版基础地图: {Path(simple_map).name}")
        
        print(f"\n🎉 基础地图绘制完成!")
        print(f"📁 图片保存在: {basemap_plotter.output_dir}")
        print("="*60)

    except KeyboardInterrupt:
        print("\n⚠️  用户中断绘制")
    except Exception as e:
        print(f"❌ 基础地图绘制失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()