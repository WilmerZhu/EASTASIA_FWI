"""
东亚地区GCMT震源分布和统计可视化模块
基于PyGMT绘制GCMT地震事件分布图和统计分析
参考GCMT_EastAsia.py的绘图风格
配置参数已集成在模块内部

功能特性：
- GCMT震源机制解分布
- 震级-深度-时间统计分析
- 多维度直方图统计
- 高质量地震学可视化
- 独立的筛选参数控制
- 数据去重处理（修复2021年后重复问题）
"""
import pygmt
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import glob
from typing import Optional, Dict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig


class GCMTPlottingConfig:
    """GCMT震源可视化配置类 - 集成在模块内部"""
    
    def __init__(self):
        """初始化GCMT可视化配置参数"""
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
            'data_resolution': "01m"
        }
        
        # GCMT筛选默认配置
        self.default_filters = {
            'magnitude_range': (5.5, 7.0),
            'depth_range': (0, 700),
            'time_range': ("2000-01-01", "2024-12-31"),
            'use_mb': True,
            'use_ms': True,
            'min_magnitude_for_meca': 5.5
        }
        
        # 数据去重配置
        self.deduplication = {
            'key_columns': ['Date', 'Latitude', 'Longitude', 'Depth', 'Event_ID'],  # 用于识别重复的关键列
            'tolerance': {
                'time_seconds': 5,      # 时间容差（秒）
                'location_km': 1.0,     # 位置容差（km）
                'depth_km': 5.0         # 深度容差（km）
            },
            'keep_first': True,         # 保留第一个记录
            'report_duplicates': True   # 报告重复统计
        }
        
        # 震源机制解绘制配置
        self.focal_mechanism = {
            'min_scale': 0.1,
            'max_scale': 0.5,
            'base_scale': 0.2,
            'scale_increment': 0.05,
            'pen': "0.2p,gray30,solid",
            'extensionfill': "cornsilk",
            'fallback_style': "c",
            'fallback_fill': "orange",
            'fallback_pen': "0.3p,black"
        }
        
        # 色彩配置
        self.colors = {
            'topo_cmap': "geo",
            'depth_cmap': "seis",
            'depth_series': "0/700",
            'volcano_color': "red",
            'fault_pen': "0.5p,black",
            'boundary_pen': "0.5p,black",
            'slab_pen': "0.35p,pink",
            'histogram_fill': "lightblue"
        }
        
        # 统计图配置
        self.statistics = {
            'histogram_transparency': 30,
            'year_max_count': 150,
            'magnitude_step': 0.1,
            'magnitude_max_count': 300,
            'depth_step': 50,
            'depth_max_count': 1200,
            'histogram_pen': "0.5p,black"
        }
        
        # 布局配置（参考GCMT_EastAsia.py）
        self.layout = {
            'main_shift_x': "17c",
            'histogram_width': "10c",
            'histogram_height': "3c",
            'histogram_shift_y': "4.75c"
        }
        
        # 输出配置
        self.output = {
            'dpi_png': 600,
            'dpi_pdf': 720,
            'crop': True,
            'save_png': True,
            'save_pdf': True
        }
        
        # 数据目录配置（相对于可视化模块）
        self.data_dirs = {
            'base': 'data_EastAsia',
            'elevation': 'elevation',
            'gcmt': 'GCMT',
            'slab': 'Slab2',
            'volcano': 'volcano',
            'faults': 'gem-global-active-faults-master',
            'tectonics': 'tectonics'
        }
        
        # 火山数据配置
        self.volcano = {
            'style': "kvolcano/0.25c",
            'fill': "red",
            'pen': "0.1p,black",
            'default_data': {
                'longitude': [120.5, 121.0, 122.5, 130.8, 138.7, 140.9, 142.5],
                'latitude': [23.8, 25.2, 24.8, 31.6, 35.4, 38.1, 40.5]
            }
        }
        
        # 数据文件搜索模式
        self.data_search = {
            'gcmt_patterns': [
                "gcmt_catalogs*.csv",
                "*GCMT*.csv", 
                "*gcmt*.txt"
            ],
            'slab_pattern': "**/*_slab2_dep_*.in",
            'slab_max_files': 5
        }
        
        # 日志配置
        self.logging = {
            'level': 'INFO',
            'console_output': True
        }


class GCMTPlotter:
    """GCMT地震事件专业绘图类
    
    专门用于绘制GCMT地震事件的分布图和统计分析图
    参考传统地震学绘图风格，包含数据去重功能
    """
    
    def __init__(self, config_file: Optional[Path] = None, output_dir: Optional[str] = None):
        """
        初始化GCMT绘图器
        
        Args:
            config_file: 配置文件路径
            output_dir: 输出目录
        """
        # 加载基础配置
        self.base_config = BaseConfig(config_file)
        
        # 加载模块配置
        self.config = GCMTPlottingConfig()
        
        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.base_config.dirs['figures']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.GCMTPlotter',
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
        
        self.logger.info("🎯 GCMT地震事件绘图器初始化完成")
        self._print_config_summary()

    def _setup_data_directories(self):
        """设置数据目录结构"""
        vis_root = self.base_config.dirs['project_root'] / '5_Visualization'
        data_base = vis_root / self.config.data_dirs['base']
        
        self.data_dirs = {
            'base': data_base,
            'elevation': data_base / self.config.data_dirs['elevation'],
            'gcmt': data_base / self.config.data_dirs['gcmt'],
            'slab': data_base / self.config.data_dirs['slab'],
            'volcano': data_base / self.config.data_dirs['volcano'],
            'faults': data_base / self.config.data_dirs['faults'],
            'tectonics': data_base / self.config.data_dirs['tectonics']
        }
        
        # 创建数据目录
        for dir_path in self.data_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 GCMT震源可视化配置摘要")
        print("-" * 60)
        
        # 研究区域配置
        print(f"研究区域: {self.base_config.region['name']}")
        region = self.base_config.region
        print(f"纬度范围: {region['lat_min']}° ~ {region['lat_max']}°")
        print(f"经度范围: {region['lon_min']}° ~ {region['lon_max']}°")
        
        # 投影和筛选配置
        print(f"地图投影: {self.projection}")
        filters = self.config.default_filters
        print(f"默认筛选:")
        print(f"  震级范围: {filters['magnitude_range'][0]} ~ {filters['magnitude_range'][1]}")
        print(f"  深度范围: {filters['depth_range'][0]} ~ {filters['depth_range'][1]} km")
        print(f"  时间范围: {filters['time_range'][0]} ~ {filters['time_range'][1]}")
        print(f"  震源机制解最小震级: {filters['min_magnitude_for_meca']}")
        
        # 去重配置
        dedup = self.config.deduplication
        print(f"数据去重:")
        print(f"  时间容差: {dedup['tolerance']['time_seconds']} 秒")
        print(f"  位置容差: {dedup['tolerance']['location_km']} km")
        print(f"  深度容差: {dedup['tolerance']['depth_km']} km")
        
        # 输出配置
        print(f"输出目录: {self.output_dir}")
        output = self.config.output
        print(f"图片分辨率: PNG {output['dpi_png']}dpi, PDF {output['dpi_pdf']}dpi")
        
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

    def deduplicate_gcmt_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        GCMT数据去重处理 - 修复2021年后的重复问题
        
        Args:
            df: 原始GCMT数据DataFrame
            
        Returns:
            去重后的DataFrame
        """
        try:
            original_count = len(df)
            self.logger.info(f"🔄 开始GCMT数据去重，原始数据: {original_count} 个事件")
            
            dedup_config = self.config.deduplication
            
            # 确保必要的列存在
            required_cols = ['Date', 'Latitude', 'Longitude']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"缺少必要的列: {missing_cols}，将使用简单去重")
                # 使用所有列进行完全重复去重
                df_dedup = df.drop_duplicates(keep='first')
            else:
                # 智能去重：基于时间、位置和深度的相似性
                self.logger.info("🧠 执行智能去重...")
                
                # 首先处理时间数据
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                # 按Event_ID去重（如果存在）
                if 'Event_ID' in df.columns:
                    self.logger.info("📝 基于Event_ID去重...")
                    df_dedup = df.drop_duplicates(subset=['Event_ID'], keep='first')
                    removed_by_event_id = original_count - len(df_dedup)
                    if removed_by_event_id > 0:
                        self.logger.info(f"   通过Event_ID去重移除: {removed_by_event_id} 个重复事件")
                else:
                    # 基于时间-位置-深度的智能去重
                    self.logger.info("📍 基于时间-位置-深度智能去重...")
                    
                    # 计算时间差和位置差
                    tolerance = dedup_config['tolerance']
                    
                    # 排序以便处理
                    df_sorted = df.sort_values(['Date', 'Latitude', 'Longitude'])
                    
                    # 标记重复事件
                    duplicate_mask = []
                    
                    for i in range(len(df_sorted)):
                        is_duplicate = False
                        
                        # 与前面的事件比较（查找窗口内的重复）
                        for j in range(max(0, i-50), i):  # 限制查找窗口以提高效率
                            row_i = df_sorted.iloc[i]
                            row_j = df_sorted.iloc[j]
                            
                            # 时间差检查
                            if pd.notna(row_i['Date']) and pd.notna(row_j['Date']):
                                time_diff = abs((row_i['Date'] - row_j['Date']).total_seconds())
                                if time_diff > tolerance['time_seconds']:
                                    continue
                            
                            # 位置差检查（简化为度数差）
                            lat_diff = abs(row_i['Latitude'] - row_j['Latitude'])
                            lon_diff = abs(row_i['Longitude'] - row_j['Longitude'])
                            
                            # 粗略换算：1度约等于111km
                            location_diff_km = ((lat_diff**2 + lon_diff**2)**0.5) * 111
                            
                            if location_diff_km > tolerance['location_km']:
                                continue
                            
                            # 深度差检查
                            if 'Depth' in df_sorted.columns:
                                if pd.notna(row_i['Depth']) and pd.notna(row_j['Depth']):
                                    depth_diff = abs(row_i['Depth'] - row_j['Depth'])
                                    if depth_diff > tolerance['depth_km']:
                                        continue
                            
                            # 如果所有条件都满足，标记为重复
                            is_duplicate = True
                            break
                        
                        duplicate_mask.append(not is_duplicate)  # True表示保留
                    
                    df_dedup = df_sorted[duplicate_mask].copy()
                
                # 最终的严格重复检查（完全相同的行）
                strict_before = len(df_dedup)
                df_dedup = df_dedup.drop_duplicates(keep='first')
                strict_removed = strict_before - len(df_dedup)
                
                if strict_removed > 0:
                    self.logger.info(f"📋 最终严格去重移除: {strict_removed} 个完全重复事件")
            
            # 统计结果
            final_count = len(df_dedup)
            total_removed = original_count - final_count
            
            self.logger.info(f"✅ 数据去重完成:")
            self.logger.info(f"   原始事件数: {original_count}")
            self.logger.info(f"   去重后事件数: {final_count}")
            self.logger.info(f"   移除重复事件: {total_removed} ({total_removed/original_count*100:.1f}%)")
            
            # 保存去重后的数据
            if dedup_config['report_duplicates'] and total_removed > 0:
                dedup_file = self.base_config.dirs['catalogs'] / f"gcmt_catalogs_deduped_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df_dedup.to_csv(dedup_file, index=False)
                self.logger.info(f"💾 去重后数据保存至: {dedup_file}")
            
            return df_dedup
            
        except Exception as e:
            self.logger.error(f"数据去重失败: {e}")
            self.logger.warning("回退到简单重复去重...")
            return df.drop_duplicates(keep='first')

    def filter_gcmt_data(self, 
                        df: pd.DataFrame,
                        magnitude_range: tuple = (5.5, 7.0),
                        depth_range: tuple = (0, 700),
                        time_range: tuple = ("2000-01-01", "2024-12-31"),
                        geographic_bounds: Optional[Dict[str, float]] = None,
                        use_mb: bool = True,
                        use_ms: bool = True,
                        perform_deduplication: bool = True) -> pd.DataFrame:
        """
        筛选GCMT数据 - 包含数据去重功能
        
        Args:
            df: 原始GCMT数据DataFrame
            magnitude_range: 震级范围 (min_mag, max_mag)
            depth_range: 深度范围 (min_depth, max_depth) km
            time_range: 时间范围 (start_date, end_date) 格式: "YYYY-MM-DD"
            geographic_bounds: 地理范围 {'lat_min': ?, 'lat_max': ?, 'lon_min': ?, 'lon_max': ?}
            use_mb: 是否使用mb震级
            use_ms: 是否使用Ms震级
            perform_deduplication: 是否执行数据去重
            
        Returns:
            筛选后的DataFrame
        """
        try:
            original_count = len(df)
            self.logger.info(f"📊 开始筛选GCMT数据，原始数据: {original_count} 个事件")
            
            # 1. 数据去重（优先执行）
            if perform_deduplication:
                df = self.deduplicate_gcmt_data(df)
            
            # 统一列名（适配不同格式）
            column_mapping = {
                'Mag1': 'Magnitude1_mb',
                'Mag2': 'Magnitude2_Ms',
                'Strike1': 'First_Nodal_Strike',
                'Dip1': 'First_Nodal_Dip',
                'Rake1': 'First_Nodal_Rake'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name not in df.columns:
                    df[new_name] = df[old_name]
            
            # 时间数据处理
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
            
            # 计算主要震级
            magnitude_columns = []
            if use_mb and 'Magnitude1_mb' in df.columns:
                magnitude_columns.append('Magnitude1_mb')
            if use_ms and 'Magnitude2_Ms' in df.columns:
                magnitude_columns.append('Magnitude2_Ms')
            
            if magnitude_columns:
                # 取多个震级列的最大值作为主震级
                df['MainMagnitude'] = df[magnitude_columns].max(axis=1)
            else:
                df['MainMagnitude'] = 5.0  # 默认值
                self.logger.warning("未找到有效的震级列，使用默认值5.0")
            
            # 2. 震级筛选
            min_mag, max_mag = magnitude_range
            df = df[
                (df['MainMagnitude'] >= min_mag) & 
                (df['MainMagnitude'] <= max_mag)
            ]
            self.logger.info(f"🔍 震级筛选 [{min_mag}, {max_mag}]: {len(df)} 个事件")
            
            # 3. 深度筛选
            if 'Depth' in df.columns:
                min_depth, max_depth = depth_range
                df = df[
                    (df['Depth'] >= min_depth) & 
                    (df['Depth'] <= max_depth)
                ]
                self.logger.info(f"🔍 深度筛选 [{min_depth}, {max_depth}] km: {len(df)} 个事件")
            
            # 4. 时间筛选
            start_date, end_date = time_range
            if 'Date' in df.columns:
                df = df[
                    (df['Date'] >= pd.to_datetime(start_date)) & 
                    (df['Date'] <= pd.to_datetime(end_date))
                ]
                self.logger.info(f"🔍 时间筛选 [{start_date}, {end_date}]: {len(df)} 个事件")
            
            # 5. 地理范围筛选
            if geographic_bounds is None:
                # 使用基础配置的东亚区域
                geographic_bounds = {
                    'lat_min': self.base_config.region['lat_min'],
                    'lat_max': self.base_config.region['lat_max'],
                    'lon_min': self.base_config.region['lon_min'],
                    'lon_max': self.base_config.region['lon_max']
                }
            
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                df = df[
                    (df['Latitude'] >= geographic_bounds['lat_min']) &
                    (df['Latitude'] <= geographic_bounds['lat_max']) &
                    (df['Longitude'] >= geographic_bounds['lon_min']) &
                    (df['Longitude'] <= geographic_bounds['lon_max'])
                ]
                self.logger.info(f"🔍 地理筛选: {len(df)} 个事件")
            
            # 筛选结果统计
            self.logger.info(f"✅ 筛选完成: {len(df)} 个事件 (从 {original_count} 筛选)")
            if len(df) > 0:
                self.logger.info(f"   震级范围: {df['MainMagnitude'].min():.1f} - {df['MainMagnitude'].max():.1f}")
                if 'Depth' in df.columns:
                    self.logger.info(f"   深度范围: {df['Depth'].min():.0f} - {df['Depth'].max():.0f} km")
                if 'Year' in df.columns:
                    self.logger.info(f"   时间范围: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"GCMT数据筛选失败: {e}")
            return df

    def prepare_data(self, 
                    gcmt_file: Optional[str] = None,
                    magnitude_range: tuple = None,
                    depth_range: tuple = None,
                    time_range: tuple = None,
                    geographic_bounds: Optional[Dict[str, float]] = None,
                    use_mb: bool = None,
                    use_ms: bool = None,
                    perform_deduplication: bool = True) -> pd.DataFrame:
        """
        准备和筛选GCMT数据 - 包含去重功能
        
        Args:
            gcmt_file: GCMT数据文件路径
            magnitude_range: 震级范围 (min_mag, max_mag)
            depth_range: 深度范围 (min_depth, max_depth) km
            time_range: 时间范围 (start_date, end_date)
            geographic_bounds: 地理范围
            use_mb: 是否使用mb震级
            use_ms: 是否使用Ms震级
            perform_deduplication: 是否执行数据去重
            
        Returns:
            筛选后的DataFrame
        """
        # 使用默认配置填补None参数
        defaults = self.config.default_filters
        magnitude_range = magnitude_range or defaults['magnitude_range']
        depth_range = depth_range or defaults['depth_range']
        time_range = time_range or defaults['time_range']
        use_mb = use_mb if use_mb is not None else defaults['use_mb']
        use_ms = use_ms if use_ms is not None else defaults['use_ms']
        
        try:
            if gcmt_file and Path(gcmt_file).exists():
                # 使用提供的文件
                if gcmt_file.endswith('.csv'):
                    df = pd.read_csv(gcmt_file)
                else:
                    df = pd.read_csv(gcmt_file, sep='\t')
                self.logger.info(f"📊 加载GCMT数据: {gcmt_file}")
            else:
                # 查找默认数据文件
                events_dir = self.base_config.dirs['catalogs']
                possible_files = [
                    events_dir / "gcmt_events.csv",
                    events_dir / "EastAsia_GCMT_events.csv",
                    self.data_dirs['gcmt'] / "EastAsia_GCMT.txt"
                ]
                
                # 添加模式搜索
                search_config = self.config.data_search
                for pattern in search_config['gcmt_patterns']:
                    possible_files.extend(list(events_dir.glob(pattern)))
                
                df = None
                for file_path in possible_files:
                    if file_path.exists():
                        try:
                            if file_path.suffix == '.csv':
                                df = pd.read_csv(file_path)
                            else:
                                df = pd.read_csv(file_path, sep='\t')
                            self.logger.info(f"📊 找到并加载数据: {file_path}")
                            break
                        except Exception as e:
                            self.logger.debug(f"文件 {file_path} 加载失败: {e}")
                            continue
                
                if df is None:
                    raise FileNotFoundError("未找到可用的GCMT数据文件")
            
            # 应用筛选条件（包含去重）
            df = self.filter_gcmt_data(
                df, 
                magnitude_range=magnitude_range,
                depth_range=depth_range,
                time_range=time_range,
                geographic_bounds=geographic_bounds,
                use_mb=use_mb,
                use_ms=use_ms,
                perform_deduplication=perform_deduplication
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"GCMT数据准备失败: {e}")
            raise

    def prepare_basemap_data(self):
        """准备基础地图数据"""
        try:
            # 地形数据
            elevation_file = self.data_dirs['elevation'] / "EastAsia.grd"
            if not elevation_file.exists():
                self.logger.info("📊 准备地形数据...")
                pygmt.grdcut(
                    grid=self.config.data_sources['elevation'],
                    region=self.region,
                    outgrid=str(elevation_file),
                )
            
            # 俯冲带深度色标
            slab_cpt = self.data_dirs['gcmt'] / "focal_depth.cpt"
            colors = self.config.colors
            pygmt.makecpt(
                cmap=colors['depth_cmap'],
                series=colors['depth_series'],
                output=str(slab_cpt),
            )
            
            return {
                'elevation': str(elevation_file),
                'slab_cpt': str(slab_cpt)
            }
            
        except Exception as e:
            self.logger.warning(f"基础地图数据准备失败: {e}")
            return {
                'elevation': self.config.data_sources['elevation'],
                'slab_cpt': self.config.colors['depth_cmap']
            }

    def add_geological_features(self, fig: pygmt.Figure):
        """添加地质构造要素"""
        try:
            # 火山数据
            volcano_file = self.data_dirs['volcano'] / "VHolo.csv"
            volcano_config = self.config.volcano
            
            if volcano_file.exists():
                fig.plot(
                    data=str(volcano_file),
                    style=volcano_config['style'],
                    fill=volcano_config['fill'],
                    pen=volcano_config['pen'],
                )
                self.logger.debug("✅ 火山数据已添加")
            else:
                # 创建示例火山数据
                sample_volcanoes = pd.DataFrame(volcano_config['default_data'])
                volcano_file.parent.mkdir(parents=True, exist_ok=True)
                sample_volcanoes.to_csv(volcano_file, index=False)
                
                fig.plot(
                    data=str(volcano_file),
                    style=volcano_config['style'],
                    fill=volcano_config['fill'],
                    pen=volcano_config['pen'],
                )
            
            # 活动断层
            faults_file = self.data_dirs['faults'] / "gem_active_faults.txt"
            if faults_file.exists():
                fig.plot(
                    data=str(faults_file),
                    pen=self.config.colors['fault_pen'],
                )
                self.logger.debug("✅ 断层数据已添加")
            
            # 板块边界
            boundaries_file = self.data_dirs['tectonics'] / "boundaries.gmt"
            if boundaries_file.exists():
                fig.plot(
                    data=str(boundaries_file),
                    pen=self.config.colors['boundary_pen'],
                )
                self.logger.debug("✅ 板块边界已添加")
            
            # 俯冲带等深线
            search_config = self.config.data_search
            slab_pattern = str(self.data_dirs['slab'] / search_config['slab_pattern'])
            slab_files = glob.glob(slab_pattern, recursive=True)
            
            if slab_files:
                for slab_file in slab_files[:search_config['slab_max_files']]:  # 限制文件数量
                    try:
                        # 读取并处理俯冲带数据
                        lons, lats = [], []
                        with open(slab_file, 'r') as file:
                            for line in file:
                                if line.startswith('>'):
                                    if lons:  
                                        lons.append(np.nan)
                                        lats.append(np.nan)
                                else:
                                    parts = line.split()
                                    if len(parts) >= 3:
                                        lons.append(float(parts[0]))
                                        lats.append(float(parts[1]))
                        
                        if lons:
                            fig.plot(
                                x=lons,
                                y=lats,
                                pen=self.config.colors['slab_pen'],  # 参考GCMT_EastAsia.py使用粉色
                            )
                    except Exception as e:
                        self.logger.debug(f"俯冲带文件 {slab_file} 处理失败: {e}")
                
                self.logger.debug("✅ 俯冲带等深线已添加")
            
        except Exception as e:
            self.logger.warning(f"地质构造要素添加失败: {e}")

    def plot_focal_mechanisms_map(self, 
                                 gcmt_data: pd.DataFrame,
                                 min_magnitude: float = 6.0,
                                 save_file: str = "5-2-gcmt_focal_mechanisms.png") -> str:
        """绘制震源机制解分布图"""
        
        self.logger.info(f"🎯 开始绘制震源机制解分布图...")
        
        try:
            # 准备基础地图数据
            basemap_data = self.prepare_basemap_data()
            
            # 创建图形
            fig = pygmt.Figure()
            
            # 绘制基础地形
            fig.grdimage(
                grid=basemap_data['elevation'],
                region=self.region,
                projection=self.projection,
                frame=[self.config.projection['frame_interval'], "WSen"],
                cmap=self.config.colors['topo_cmap'],
            )
            
            # 添加地质构造要素
            self.add_geological_features(fig)
            
            # 筛选需要绘制震源机制解的事件
            large_events = gcmt_data[gcmt_data['MainMagnitude'] >= min_magnitude].copy()
            
            self.logger.info(f"📊 绘制 {len(large_events)} 个震源机制解 (M≥{min_magnitude})")
            
            # 绘制震源机制解
            success_count = 0
            fallback_count = 0
            focal_config = self.config.focal_mechanism
            
            for _, event in large_events.iterrows():
                try:
                    # 检查必需的震源机制参数
                    required_cols = ['First_Nodal_Strike', 'First_Nodal_Dip', 'First_Nodal_Rake']
                    if all(col in event.index and pd.notna(event[col]) for col in required_cols):
                        
                        # 计算震源机制球大小（基于震级）
                        mag = event['MainMagnitude']
                        scale_factor = focal_config['base_scale'] + (mag - min_magnitude) * focal_config['scale_increment']
                        scale_factor = max(focal_config['min_scale'], min(focal_config['max_scale'], scale_factor))
                        
                        # 震源机制参数
                        focal_mechanism = {
                            "strike": float(event['First_Nodal_Strike']),
                            "dip": float(event['First_Nodal_Dip']),
                            "rake": float(event['First_Nodal_Rake']),
                            "magnitude": float(mag),
                            "depth": float(event.get('Depth', 10)),
                        }
                        
                        # 绘制震源机制球
                        fig.meca(
                            spec=focal_mechanism,
                            scale=f"{scale_factor:.2f}c",
                            longitude=float(event['Longitude']),
                            latitude=float(event['Latitude']),
                            depth=float(event.get('Depth', 10)),
                            cmap=basemap_data['slab_cpt'],
                            extensionfill=focal_config['extensionfill'],
                            pen=focal_config['pen'],
                        )
                        success_count += 1
                        
                    else:
                        # 用圆圈表示没有震源机制解的事件
                        mag = event['MainMagnitude']
                        size = focal_config['base_scale'] + (mag - min_magnitude) * focal_config['scale_increment']
                        
                        fig.plot(
                            x=float(event['Longitude']),
                            y=float(event['Latitude']),
                            style=f"{focal_config['fallback_style']}{size:.2f}c",
                            fill=focal_config['fallback_fill'],
                            pen=focal_config['fallback_pen']
                        )
                        fallback_count += 1
                        
                except Exception as e:
                    self.logger.debug(f"事件绘制失败: {e}")
                    fallback_count += 1
            
            # 添加海岸线
            fig.coast(
                resolution="f",
                shorelines="1/0.01p",
            )
            
            self.logger.info(f"✅ 成功绘制 {success_count} 个震源机制解，{fallback_count} 个圆圈标记")
            
            # 保存图片
            output_config = self.config.output
            output_path = self.output_dir / save_file
            
            if output_config['save_png']:
                fig.savefig(str(output_path), dpi=output_config['dpi_png'], crop=output_config['crop'])
                self.logger.info(f"✅ PNG图像保存至: {output_path}")
            
            if output_config['save_pdf']:
            pdf_path = output_path.with_suffix('.pdf')
                fig.savefig(str(pdf_path), dpi=output_config['dpi_pdf'], crop=output_config['crop'])
                self.logger.info(f"✅ PDF版本保存至: {pdf_path}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"震源机制解分布图绘制失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def plot_comprehensive_statistics(self, 
                                    gcmt_data: pd.DataFrame,
                                    min_magnitude_for_meca: float = 6.0,
                                    save_file: str = "5-2-gcmt_comprehensive_analysis.png") -> str:
        """绘制综合统计分析图（主地图+统计直方图）- 参考GCMT_EastAsia.py"""
        
        self.logger.info(f"📊 开始绘制GCMT综合分析图...")
        
        try:
            # 准备基础地图数据
            basemap_data = self.prepare_basemap_data()
            
            # 创建图形
            fig = pygmt.Figure()
            
            # === 主地图部分 ===
            # 绘制基础地形
            fig.grdimage(
                grid=basemap_data['elevation'],
                region=self.region,
                projection=self.projection,
                frame=[self.config.projection['frame_interval'], "WSen"],
                cmap=self.config.colors['topo_cmap'],
            )
            
            # 添加地质构造要素
            self.add_geological_features(fig)
            
            # 绘制震源机制解
            large_events = gcmt_data[gcmt_data['MainMagnitude'] >= min_magnitude_for_meca].copy()
            focal_config = self.config.focal_mechanism
            
            for _, event in large_events.iterrows():
                try:
                    required_cols = ['First_Nodal_Strike', 'First_Nodal_Dip', 'First_Nodal_Rake']
                    if all(col in event.index and pd.notna(event[col]) for col in required_cols):
                        
                        mag = event['MainMagnitude']
                        
                        # 震源机制参数
                        focal_mechanism = {
                            "strike": float(event['First_Nodal_Strike']),
                            "dip": float(event['First_Nodal_Dip']),
                            "rake": float(event['First_Nodal_Rake']),
                            "magnitude": float(mag),
                            "depth": float(event.get('Depth', 10)),
                        }
                        
                        fig.meca(
                            spec=focal_mechanism,
                            scale="0.25c",  # 参考GCMT_EastAsia.py
                            longitude=float(event['Longitude']),
                            latitude=float(event['Latitude']),
                            depth=float(event.get('Depth', 10)),
                            cmap=basemap_data['slab_cpt'],
                            extensionfill=focal_config['extensionfill'],
                            pen=focal_config['pen'],
                        )
                        
                except Exception as e:
                    self.logger.debug(f"震源机制绘制失败: {e}")
            
            # 添加海岸线
            fig.coast(
                resolution="f",
                shorelines="1/0.01p",
            )
            
            # === 统计直方图部分（参考GCMT_EastAsia.py的布局）===
            # 移动原点到右侧绘制统计图
            layout = self.config.layout
            statistics = self.config.statistics
            
            fig.shift_origin(xshift=layout['main_shift_x'])
            
            # 1. 年份分布直方图
            if 'Year' in gcmt_data.columns and gcmt_data['Year'].notna().any():
                year_data = gcmt_data['Year'].dropna()
                year_min, year_max = int(year_data.min()), int(year_data.max())
                
                # 动态调整Y轴范围
                max_count = max(statistics['year_max_count'], year_data.value_counts().max() * 1.2)
                
                fig.histogram(
                    data=year_data,
                    projection=f"X{layout['histogram_width']}/{layout['histogram_height']}",
                    region=[year_min, year_max, 0, max_count],
                    series=f"{year_min}/{year_max}/1",
                    frame=["WSrt", "xaf+lYear", "yaf+lCounts"],
                    pen=statistics['histogram_pen'],
                    fill=self.config.colors['histogram_fill'],
                    transparency=statistics['histogram_transparency'],
                )
                self.logger.debug("✅ 年份分布直方图已绘制")
            
            # 2. 震级分布直方图（下移）
            fig.shift_origin(yshift=layout['histogram_shift_y'])
            
            if 'MainMagnitude' in gcmt_data.columns:
                mag_data = gcmt_data['MainMagnitude'].dropna()
                mag_min, mag_max = mag_data.min(), mag_data.max()
                mag_step = statistics['magnitude_step']
                max_mag_count = max(statistics['magnitude_max_count'], mag_data.value_counts().max() * 1.2)
                
                fig.histogram(
                    data=mag_data,
                    projection=f"X{layout['histogram_width']}/{layout['histogram_height']}",
                    region=[mag_min, mag_max, 0, max_mag_count],
                    series=f"{mag_min}/{mag_max}/{mag_step}",
                    frame=["WSrt", "xaf+lMagnitude", "yaf+lCounts"],
                    pen=statistics['histogram_pen'],
                    fill=self.config.colors['histogram_fill'],
                    transparency=statistics['histogram_transparency'],
                )
                self.logger.debug("✅ 震级分布直方图已绘制")
            
            # 3. 深度分布直方图（再下移）
            fig.shift_origin(yshift=layout['histogram_shift_y'])
            
            if 'Depth' in gcmt_data.columns and gcmt_data['Depth'].notna().any():
                depth_data = gcmt_data['Depth'].dropna()
                depth_min, depth_max = 0, min(700, depth_data.max())
                depth_step = statistics['depth_step']
                max_depth_count = max(statistics['depth_max_count'], len(depth_data) * 0.3)
                
                fig.histogram(
                    data=depth_data,
                    projection=f"X{layout['histogram_width']}/{layout['histogram_height']}",
                    region=[depth_min, depth_max, 1, max_depth_count],
                    series=f"{depth_min}/{depth_max}/{depth_step}",
                    frame=["WSrt", "xaf+lDepth (km)", "yaf+lCounts"],
                    pen=statistics['histogram_pen'],
                    cmap=basemap_data['slab_cpt'],
                    transparency=statistics['histogram_transparency'],
                )
                self.logger.debug("✅ 深度分布直方图已绘制")
            
            # 保存图片
            output_config = self.config.output
            output_path = self.output_dir / save_file
            
            if output_config['save_png']:
                fig.savefig(str(output_path), dpi=output_config['dpi_pdf'], crop=output_config['crop'])
                self.logger.info(f"✅ PNG图像保存至: {output_path}")
            
            if output_config['save_pdf']:
            pdf_path = output_path.with_suffix('.pdf')
                fig.savefig(str(pdf_path), dpi=output_config['dpi_pdf'], crop=output_config['crop'])
                self.logger.info(f"✅ PDF版本保存至: {pdf_path}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"GCMT综合分析图绘制失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def plot_all_gcmt_analysis(self, 
                              gcmt_file: Optional[str] = None,
                              # === 可修改的筛选参数 ===
                              magnitude_range: tuple = None,
                              depth_range: tuple = None,
                              time_range: tuple = None,
                              geographic_bounds: Optional[Dict[str, float]] = None,
                              use_mb: bool = None,
                              use_ms: bool = None,
                              min_magnitude_for_meca: float = None,
                              perform_deduplication: bool = True) -> Dict[str, str]:
        """
        生成完整的GCMT分析图集 - 带独立筛选参数和去重功能
        
        Args:
            gcmt_file: GCMT数据文件路径
            magnitude_range: 震级范围 (min_mag, max_mag)
            depth_range: 深度范围 (min_depth, max_depth) km
            time_range: 时间范围 (start_date, end_date)
            geographic_bounds: 地理范围
            use_mb: 是否使用mb震级
            use_ms: 是否使用Ms震级
            min_magnitude_for_meca: 震源机制解最小震级阈值
            perform_deduplication: 是否执行数据去重
            
        Returns:
            生成的图片文件路径字典
        """
        
        self.logger.info("🎨 开始生成完整的GCMT分析图集...")
        
        # 使用默认配置填补None参数
        defaults = self.config.default_filters
        magnitude_range = magnitude_range or defaults['magnitude_range']
        depth_range = depth_range or defaults['depth_range']
        time_range = time_range or defaults['time_range']
        use_mb = use_mb if use_mb is not None else defaults['use_mb']
        use_ms = use_ms if use_ms is not None else defaults['use_ms']
        min_magnitude_for_meca = min_magnitude_for_meca or defaults['min_magnitude_for_meca']
        
        # 打印筛选参数
        self.logger.info("📋 使用的筛选参数:")
        self.logger.info(f"   震级范围: {magnitude_range[0]} - {magnitude_range[1]}")
        self.logger.info(f"   深度范围: {depth_range[0]} - {depth_range[1]} km")
        self.logger.info(f"   时间范围: {time_range[0]} - {time_range[1]}")
        if geographic_bounds:
            self.logger.info(f"   地理范围: {geographic_bounds}")
        self.logger.info(f"   使用mb: {use_mb}, 使用Ms: {use_ms}")
        self.logger.info(f"   震源机制解最小震级: {min_magnitude_for_meca}")
        self.logger.info(f"   执行数据去重: {perform_deduplication}")
        
        results = {}
        
        try:
            # 准备数据（包含去重）
            gcmt_data = self.prepare_data(
                gcmt_file=gcmt_file,
                magnitude_range=magnitude_range,
                depth_range=depth_range,
                time_range=time_range,
                geographic_bounds=geographic_bounds,
                use_mb=use_mb,
                use_ms=use_ms,
                perform_deduplication=perform_deduplication
            )
            
            if len(gcmt_data) == 0:
                self.logger.warning("没有可用的GCMT数据")
                return results
            
            # 1. 震源机制解分布图
            self.logger.info("  🎯 生成震源机制解分布图...")
            focal_map = self.plot_focal_mechanisms_map(
                gcmt_data, 
                min_magnitude=min_magnitude_for_meca
            )
            if focal_map:
                results['focal_mechanisms'] = focal_map
            
            # 2. 综合分析图（地图+统计）
            self.logger.info("  📊 生成综合分析图...")
            comprehensive_analysis = self.plot_comprehensive_statistics(
                gcmt_data,
                min_magnitude_for_meca=min_magnitude_for_meca
            )
            if comprehensive_analysis:
                results['comprehensive_analysis'] = comprehensive_analysis
            
            self.logger.info("✅ GCMT分析图集生成完成!")
            
        except Exception as e:
            self.logger.error(f"GCMT分析图集生成失败: {e}")
            import traceback
            traceback.print_exc()
        
        return results


def find_gcmt_data_file() -> Optional[str]:
    """查找GCMT数据文件"""
    try:
        base_config = BaseConfig()
        events_dir = base_config.dirs['catalogs']
        
        # 使用配置中的搜索模式
        config = GCMTPlottingConfig()
        patterns = config.data_search['gcmt_patterns']
        
        for pattern in patterns:
            files = list(events_dir.glob(pattern))
            if files:
                latest_file = sorted(files)[-1]
                return str(latest_file)
        
        return None
        
    except Exception as e:
        logging.error(f"查找GCMT数据文件失败: {e}")
        return None


def main():
    """主函数 - GCMT震源分布和统计可视化"""
    print("🎯 EASTASIA-FWI GCMT震源分布和统计可视化系统（含去重功能）")
    print("="*70)
    
    # === 可修改的筛选参数（参考GCMT_EastAsia.py的设置）===
    MAGNITUDE_RANGE = (5.5, 7.0)  # 震级范围
    DEPTH_RANGE = (0, 700)        # 深度范围（km）
    TIME_RANGE = ("2000-01-01", "2024-12-31")  # 时间范围
    GEOGRAPHIC_BOUNDS = None      # 地理范围（None表示使用默认东亚区域）
    USE_MB = True                 # 是否使用mb震级
    USE_MS = True                 # 是否使用Ms震级
    MIN_MAGNITUDE_FOR_MECA = 5.5  # 震源机制解最小震级阈值
    PERFORM_DEDUPLICATION = True  # 是否执行数据去重（修复2021年后重复问题）
    
    try:
        # 查找GCMT数据文件
        gcmt_file = find_gcmt_data_file()
        
        if not gcmt_file:
            print("❌ 未找到GCMT数据文件")
            print("   请先运行 1_2_Process_gcmt_catalogs.py 生成GCMT事件数据")
            return
        
        print(f"📊 找到GCMT数据文件: {Path(gcmt_file).name}")
        
        # 创建GCMT绘图器
        gcmt_plotter = GCMTPlotter()
        
        print("\n🎨 开始生成GCMT分析图集...")
        print(f"📋 筛选参数: 震级{MAGNITUDE_RANGE}, 深度{DEPTH_RANGE}km, 时间{TIME_RANGE}")
        print(f"🔄 数据去重: {'开启' if PERFORM_DEDUPLICATION else '关闭'}")
        
        # 生成完整分析图集
        results = gcmt_plotter.plot_all_gcmt_analysis(
            gcmt_file=gcmt_file,
            magnitude_range=MAGNITUDE_RANGE,
            depth_range=DEPTH_RANGE,
            time_range=TIME_RANGE,
            geographic_bounds=GEOGRAPHIC_BOUNDS,
            use_mb=USE_MB,
            use_ms=USE_MS,
            min_magnitude_for_meca=MIN_MAGNITUDE_FOR_MECA,
            perform_deduplication=PERFORM_DEDUPLICATION
        )
        
        # 结果汇总
        print(f"\n🎉 GCMT分析图集生成完成!")
        print("="*70)
        print(f"📁 所有图片保存在: {gcmt_plotter.output_dir}")
        print(f"\n📊 生成的图表:")
        
        if results:
            for analysis_type, file_path in results.items():
                print(f"  ✅ {analysis_type}: {Path(file_path).name}")
        else:
            print("  ⚠️  没有成功生成图表")
        
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断绘制")
    except Exception as e:
        print(f"❌ GCMT可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()