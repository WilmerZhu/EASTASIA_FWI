"""
EASTASIA-FWI 事件采样与 CMT3D 目录集成
=========================================

功能描述:
- 永久台站-GCMT 事件匹配与随机加权采样
- CMT3D (3D 波形矫正震源) 目录读取、筛选、测试事件选择
- SPECFEM3D Globe CMTSOLUTION 文件生成（标准格式 + ECEF 格式）
- 台站覆盖质量评估
- 2D/3D 可视化

科学原理:
- CMT3D (Sawade et al., 2022) 通过 GLAD-M25 三维模型矫正震源机制，
  相比传统 GCMT 在东亚区域具有更高的震源参数精度
- 正演测试优先使用 CMT3D 震源，以减少震源误差对波形评估的影响

作者: EASTASIA-FWI Team
日期: 2026-03-25
版本: v2.0
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from typing import Dict, Optional, Any, List
import sys
from datetime import datetime
from geopy.distance import geodesic
import warnings
import random

# 尝试导入Plotly用于交互式3D可视化
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

warnings.filterwarnings("ignore")

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig


class EventSamplingConfig:
    """事件采样配置类 - 集成在模块内部"""
    
    def __init__(self):
        """初始化事件采样配置参数"""
        
        # 永久台站配置
        self.permanent_stations = {
            'use_permanent_stations_only': True,
            'station_file_path': 'EastAsia_permanent_stations_filtered.csv',
            'min_permanent_stations': 10,
            'require_network_diversity': True
        }
        
        # 事件筛选条件
        self.event_filters = {
            'magnitude_range': (6.0, 7.0),
            'depth_range': (10, 600),
            'time_range': ('2010-01-01', '2024-12-31'),
            'magnitude_types': ['mb', 'Ms'],
            'prefer_mb': True,
            'remove_duplicates': True,
            'quality_threshold': 0.5
        }
        
        # 台站覆盖要求
        self.station_coverage = {
            'min_stations_per_event': 20,
            'max_epicentral_distance': 100.0,
            'min_epicentral_distance': 2.0,
            'optimal_station_count': 50,
            'azimuthal_gap_weight': 0.3,
            'distance_weight': 0.7
        }
        
        # 空间采样策略
        self.spatial_sampling = {
            'method': 'random_weighted',
            'min_inter_event_distance': 50.0,
            'coverage_weight': 0.7,
            'magnitude_weight': 0.3,
            'depth_bins': {
                'shallow': (0, 40),
                'intermediate': (40, 150),
                'deep': (150, 600)
            },
            'target_events_per_depth_bin': 30,
            'depth_balance_weight': 0.3,
            'random_seed': 42,
            'allow_close_events': False,
            'spatial_balance': True
        }
        
        # SPECFEM3D 文件生成配置
        self.specfem_output = {
            'generate_stations_file': True,
            'generate_cmtsolution_files': True,
            'coordinate_precision': 4,
            'elevation_precision': 1,
            'default_burial_depth': 0.0,
            'time_shift': 0.0,
            'half_duration_method': 'magnitude_based',
            'cmt_precision': 6
        }
        
        # 输出设置
        self.output = {
            'max_total_events': 100,
            'save_event_list': True,
            'create_stations_file': True,
            'create_cmtsolution_files': True,
            'add_timestamp': True,
            'generate_summary_report': True
        }
        
        # CMT3D 目录配置（Sawade et al., 2022）
        self.cmt3d = {
            'catalog_file': 'cmt3d.txt',
            'individual_dir': 'finalcatalog',
            'data_dir': 'data/catalogs/CMT3D',
            # 三模型公共覆盖区（FWEA23 ∩ SinoScope1.0 ∩ EARA2024）
            'region': {
                'lat_min': 10.0, 'lat_max': 55.0,
                'lon_min': 80.0, 'lon_max': 150.0,
            },
            'magnitude_range': (5.5, 7.0),
            'depth_range': (10, 600),
            'n_test_events': 5,
            # SPECFEM v7.0.0 格式选项
            'use_ecef_format': True,        # 匹配 constants.h.in: USE_ECEF_CMTSOLUTION = .true.
            'source_decay_mimic_triangle': 1.628,  # constants.h.in 编译时常量
            'r_earth_m': 6371000.0,
            # 深度分层选择策略
            'depth_strategy': {
                'shallow': (10, 50),         # 1-2 个事件
                'intermediate': (50, 200),   # 1-2 个事件
                'deep': (200, 600),          # 1 个事件
            },
        }

        # 可视化设置（英文标签）
        self.visualization = {
            'create_maps': True,
            'create_statistics': True,
            'create_3d_plot': True,
            'create_interactive_3d': True,  # 新增：创建交互式3D图
            'figure_size': (14, 10),
            'figure_size_3d': (12, 9),
            'dpi': 300,
            'save_formats': ['png', 'pdf'],
            'station_color': 'gray',
            'station_marker': '^',
            'station_size': 30,
            'event_color_by': 'depth',
            'depth_colormap': 'plasma_r',
            'magnitude_scaling': 40,
            # 3D可视化配置
            'plot_3d': {
                'elevation_angle': 20,
                'azimuth_angle': 45,
                'show_depth_plane': True,
                'depth_plane_alpha': 0.1,
                'station_3d_height': 0.5,  # 台站在3D中的显示高度
                'grid_alpha': 0.3
            },
            # 交互式3D配置
            'interactive_3d': {
                'marker_size_stations': 8,
                'marker_size_events': 6,
                'opacity_stations': 0.8,
                'opacity_events': 0.9,
                'colorscale': 'Plasma_r',  # Plotly颜色标尺
                'show_depth_surfaces': True,
                'depth_surfaces': [0, -100, -200, -410, -660],  # 地质界面深度
                'surface_opacity': 0.3,
                'camera_eye': {'x': 1.5, 'y': 1.5, 'z': 1.2},  # 初始视角
                'hover_template_events': '<b>Event</b><br>' +
                                       'Longitude: %{x:.2f}°<br>' +
                                       'Latitude: %{y:.2f}°<br>' +
                                       'Depth: %{customdata[0]:.1f} km<br>' +
                                       'Magnitude: %{customdata[1]:.2f}<br>' +
                                       'Station Count: %{customdata[2]}<br>' +
                                       'Coverage Score: %{customdata[3]:.3f}<extra></extra>',
                'hover_template_stations': '<b>Station</b><br>' +
                                         'Network: %{customdata[0]}<br>' +
                                         'Station: %{customdata[1]}<br>' +
                                         'Longitude: %{x:.3f}°<br>' +
                                         'Latitude: %{y:.3f}°<extra></extra>'
            },
            # 英文标签配置
            'labels': {
                'station_label': 'Permanent Stations',
                'event_label': 'Selected Events',
                'depth_label': 'Depth (km)',
                'magnitude_label': 'Magnitude',
                'latitude_label': 'Latitude',
                'longitude_label': 'Longitude',
                'station_count_label': 'Station Count per Event',
                'coverage_score_label': 'Coverage Score',
                'event_count_label': 'Event Count',
                'distance_label': 'Epicentral Distance (°)',
                'azimuth_label': 'Azimuth (°)'
            }
        }
        
        # 日志配置
        self.logging = {
            'level': 'INFO',
            'console_output': True,
            'file_prefix': 'event_sampling'
        }


class EventSampler:
    """东亚地区事件采样器
    
    基于永久台站进行GCMT事件智能采样
    支持多种采样策略和质量评估
    """
    
    def __init__(self, config_file: Optional[Path] = None, output_dir: Optional[str] = None):
        """
        初始化事件采样器
        
        Args:
            config_file: 配置文件路径
            output_dir: 输出目录
        """
        # 加载基础配置
        self.base_config = BaseConfig(config_file)
        
        # 加载模块配置
        self.config = EventSamplingConfig()
        
        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.base_config.dirs['figures']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.EventSampler',
            self.config.logging['level']
        )
        
        # 数据存储
        self.permanent_stations = None
        self.gcmt_events = None
        self.selected_events = None
        self.sampling_stats = {}
        
        # 设置matplotlib和随机种子
        self._setup_matplotlib()
        self._setup_random_seed()
        
        # 检查Plotly可用性
        if not PLOTLY_AVAILABLE and self.config.visualization['create_interactive_3d']:
            self.logger.warning("⚠️ Plotly未安装，将跳过交互式3D可视化。请运行: pip install plotly")
            self.config.visualization['create_interactive_3d'] = False
        
        self.logger.info("✅ 事件采样器初始化完成")
        self._print_config_summary()
    
    def _setup_matplotlib(self):
        """设置matplotlib参数"""
        viz_config = self.config.visualization
        plt.rcParams['figure.figsize'] = viz_config['figure_size']
        plt.rcParams['savefig.dpi'] = viz_config['dpi']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10

    def _setup_random_seed(self):
        """设置随机种子"""
        seed = self.config.spatial_sampling['random_seed']
        random.seed(seed)
        np.random.seed(seed)

    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 事件采样配置摘要")
        print("-" * 60)
        
        # 研究区域配置
        print(f"研究区域: {self.base_config.region['name']}")
        region = self.base_config.region
        print(f"纬度范围: {region['lat_min']}° ~ {region['lat_max']}°")
        print(f"经度范围: {region['lon_min']}° ~ {region['lon_max']}°")
        
        # 采样配置
        filters = self.config.event_filters
        coverage = self.config.station_coverage
        sampling = self.config.spatial_sampling
        
        print(f"事件筛选:")
        print(f"  震级范围: {filters['magnitude_range'][0]} ~ {filters['magnitude_range'][1]}")
        print(f"  深度范围: {filters['depth_range'][0]} ~ {filters['depth_range'][1]} km")
        print(f"  时间范围: {filters['time_range'][0]} ~ {filters['time_range'][1]}")
        
        print(f"台站覆盖:")
        print(f"  最小台站数: {coverage['min_stations_per_event']}")
        print(f"  最大震中距: {coverage['max_epicentral_distance']}°")
        
        print(f"采样策略:")
        print(f"  方法: {sampling['method']}")
        print(f"  最大事件数: {self.config.output['max_total_events']}")
        print(f"  深度分层: {len(sampling['depth_bins'])} 层")
        
        # 输出配置
        print(f"输出目录: {self.output_dir}")
        viz = self.config.visualization
        viz_types = []
        if viz['create_maps']:
            viz_types.append("2D地图")
        if viz['create_3d_plot']:
            viz_types.append("3D分布图")
        if viz['create_interactive_3d'] and PLOTLY_AVAILABLE:
            viz_types.append("交互式3D")
        if viz['create_statistics']:
            viz_types.append("统计图")
        print(f"可视化: {' + '.join(viz_types)}")
        
        print("-" * 60)

    def load_permanent_stations(self, station_file: Optional[Path] = None) -> pd.DataFrame:
        """
        加载永久台站数据
        
        Args:
            station_file: 台站文件路径
            
        Returns:
            永久台站数据框
        """
        if station_file is None:
            # 查找筛选后的永久台站文件
            stations_dir = self.base_config.dirs['stations']
            possible_files = [
                stations_dir / 'EastAsia_permanent_stations_filtered.csv',
                stations_dir / 'EastAsia_permanent_stations.csv',
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    station_file = file_path
                    break
            else:
                raise FileNotFoundError("未找到永久台站文件")
        
        try:
            self.permanent_stations = pd.read_csv(station_file)
            self.logger.info(f"📡 加载永久台站数据: {len(self.permanent_stations)} 个台站")
            self.logger.info(f"   文件路径: {station_file}")
            
            # 数据预处理
            self.permanent_stations = self._preprocess_stations(self.permanent_stations)
            
            return self.permanent_stations
            
        except Exception as e:
            self.logger.error(f"❌ 加载永久台站数据失败: {e}")
            raise
    
    def _preprocess_stations(self, stations_df: pd.DataFrame) -> pd.DataFrame:
        """预处理永久台站数据"""
        initial_count = len(stations_df)
        
        # 确保必要的列存在
        required_cols = ['Network', 'Station', 'Latitude', 'Longitude']
        missing_cols = [col for col in required_cols if col not in stations_df.columns]
        if missing_cols:
            raise ValueError(f"永久台站数据缺少必要列: {missing_cols}")
        
        # 数据清理
        stations_df = stations_df.dropna(subset=required_cols)
        
        # 坐标验证
        stations_df = stations_df[
            (stations_df['Latitude'] >= -90) & (stations_df['Latitude'] <= 90) &
            (stations_df['Longitude'] >= -180) & (stations_df['Longitude'] <= 180)
        ]
        
        # 区域筛选
        region = self.base_config.region
        stations_df = stations_df[
            (stations_df['Latitude'] >= region['lat_min']) &
            (stations_df['Latitude'] <= region['lat_max']) &
            (stations_df['Longitude'] >= region['lon_min']) &
            (stations_df['Longitude'] <= region['lon_max'])
        ]
        
        # 创建StationID
        if 'StationID' not in stations_df.columns:
            stations_df['StationID'] = stations_df['Network'] + '.' + stations_df['Station']
        
        processed_count = len(stations_df)
        self.logger.info(f"🔧 永久台站预处理: {initial_count} → {processed_count}")
        
        return stations_df

    def load_gcmt_events(self, gcmt_file: Optional[Path] = None) -> pd.DataFrame:
        """
        加载GCMT事件数据
        
        Args:
            gcmt_file: GCMT事件文件路径
            
        Returns:
            GCMT事件数据框
        """
        if gcmt_file is None:
            # 查找最新的GCMT文件（优先去重版本）
            events_dir = self.base_config.dirs['catalogs']
            gcmt_files = list(events_dir.glob("gcmt_catalogs_deduped_*.csv"))
            
            if not gcmt_files:
                gcmt_files = list(events_dir.glob("gcmt_catalogs_*.csv"))
            
            if not gcmt_files:
                gcmt_files = list(events_dir.glob("gcmt_catalogs_*.tsv"))
            
            if not gcmt_files:
                raise FileNotFoundError("未找到GCMT事件文件")
            
            gcmt_file = max(gcmt_files, key=lambda x: x.stat().st_mtime)
        
        try:
            # 根据文件扩展名选择分隔符
            separator = '\t' if gcmt_file.suffix == '.tsv' else ','
            self.gcmt_events = pd.read_csv(gcmt_file, sep=separator)
            
            self.logger.info(f"📊 加载GCMT事件数据: {len(self.gcmt_events)} 个事件")
            self.logger.info(f"   文件路径: {gcmt_file.name}")
            
            # 数据预处理
            self.gcmt_events = self._preprocess_events(self.gcmt_events)
            
            return self.gcmt_events
            
        except Exception as e:
            self.logger.error(f"❌ 加载GCMT事件失败: {e}")
            raise
    
    def _preprocess_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """预处理GCMT事件数据"""
        initial_count = len(events_df)
        
        # 标准化列名
        column_mapping = {
            'longitude': 'Longitude',
            'latitude': 'Latitude',
            'depth': 'Depth'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in events_df.columns and new_col not in events_df.columns:
                events_df[new_col] = events_df[old_col]
        
        # 处理震级数据
        events_df = self._process_magnitude_data(events_df)
        
        # 确保必要的列存在
        required_cols = ['Longitude', 'Latitude', 'Depth', 'Magnitude']
        missing_cols = [col for col in required_cols if col not in events_df.columns]
        if missing_cols:
            raise ValueError(f"GCMT事件数据缺少必要列: {missing_cols}")
        
        # 数据质量检查
        events_df = events_df.dropna(subset=required_cols)
        
        # 坐标和深度验证
        events_df = events_df[
            (events_df['Latitude'] >= -90) & (events_df['Latitude'] <= 90) &
            (events_df['Longitude'] >= -180) & (events_df['Longitude'] <= 180) &
            (events_df['Depth'] >= 0) & (events_df['Depth'] <= 1000)
        ]
        
        # 区域筛选
        region = self.base_config.region
        events_df = events_df[
            (events_df['Longitude'] >= region['lon_min']) &
            (events_df['Longitude'] <= region['lon_max']) &
            (events_df['Latitude'] >= region['lat_min']) &
            (events_df['Latitude'] <= region['lat_max'])
        ]
        
        processed_count = len(events_df)
        self.logger.info(f"🔧 GCMT事件预处理: {initial_count} → {processed_count}")
        
        return events_df
    
    def _process_magnitude_data(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """处理震级数据"""
        def get_event_magnitude(row):
            """获取事件的最佳震级"""
            mb = row.get('Magnitude1_mb')
            ms = row.get('Magnitude2_Ms')
            
            # 优先使用mb震级
            if self.config.event_filters.get('prefer_mb', True):
                if pd.notna(mb) and mb > 0:
                    return float(mb), "mb"
                elif pd.notna(ms) and ms > 0:
                    return float(ms), "Ms"
            else:
                if pd.notna(ms) and ms > 0:
                    return float(ms), "Ms"
                elif pd.notna(mb) and mb > 0:
                    return float(mb), "mb"
            
            # 回退到其他震级
            if 'Magnitude' in row and pd.notna(row['Magnitude']):
                return float(row['Magnitude']), "M"
            
            return 5.0, "M"  # 默认值
        
        # 计算统一的震级
        magnitude_data = events_df.apply(get_event_magnitude, axis=1)
        events_df['Magnitude'] = magnitude_data.apply(lambda x: x[0])
        events_df['MagnitudeType'] = magnitude_data.apply(lambda x: x[1])
        
        return events_df

    def apply_event_filters(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        应用事件筛选条件
        
        Args:
            events_df: 事件数据框
            
        Returns:
            筛选后的事件数据框
        """
        filters = self.config.event_filters
        initial_count = len(events_df)
        
        self.logger.info("🎯 开始应用事件筛选条件...")
        
        # 震级筛选
        mag_min, mag_max = filters['magnitude_range']
        events_df = events_df[
            (events_df['Magnitude'] >= mag_min) &
            (events_df['Magnitude'] <= mag_max)
        ]
        mag_filtered = len(events_df)
        
        # 深度筛选
        depth_min, depth_max = filters['depth_range']
        events_df = events_df[
            (events_df['Depth'] >= depth_min) &
            (events_df['Depth'] <= depth_max)
        ]
        depth_filtered = len(events_df)
        
        # 时间筛选
        if 'Date' in events_df.columns:
            events_df = self._apply_time_filter(events_df, filters)
        time_filtered = len(events_df)
        
        # 记录筛选统计
        self.logger.info("📊 事件筛选统计:")
        self.logger.info(f"  原始事件: {initial_count}")
        self.logger.info(f"  震级筛选后: {mag_filtered} (移除 {initial_count - mag_filtered})")
        self.logger.info(f"  深度筛选后: {depth_filtered} (移除 {mag_filtered - depth_filtered})")
        self.logger.info(f"  时间筛选后: {time_filtered} (移除 {depth_filtered - time_filtered})")
        
        return events_df
    
    def _apply_time_filter(self, events_df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """应用时间筛选"""
        try:
            time_min, time_max = filters['time_range']
            time_min = pd.to_datetime(time_min)
            time_max = pd.to_datetime(time_max)
            
            if 'Date' in events_df.columns:
                events_df['Date'] = pd.to_datetime(events_df['Date'], errors='coerce')
                events_df = events_df[
                    (events_df['Date'] >= time_min) &
                    (events_df['Date'] <= time_max)
                ]
            
            return events_df
            
        except Exception as e:
            self.logger.warning(f"时间筛选失败: {e}")
            return events_df
    
    def calculate_station_coverage(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算台站覆盖
        
        Args:
            events_df: 事件数据框
            
        Returns:
            包含覆盖信息的事件数据框
        """
        if self.permanent_stations is None:
            raise ValueError("请先加载永久台站数据")
        
        self.logger.info("🔍 计算台站-事件覆盖关系...")
        
        coverage_results = []
        coverage_config = self.config.station_coverage
        
        for idx, event in events_df.iterrows():
            coverage = self._calculate_single_event_coverage(event, coverage_config)
            coverage['event_index'] = idx
            coverage_results.append(coverage)
        
        # 创建覆盖数据框
        coverage_df = pd.DataFrame(coverage_results)
        
        # 合并到事件数据
        events_with_coverage = events_df.copy()
        events_with_coverage = events_with_coverage.merge(
            coverage_df, 
            left_index=True, 
            right_on='event_index', 
            how='left'
        )
        
        self.logger.info(f"✅ 台站覆盖计算完成")
        
        return events_with_coverage
    
    def _calculate_single_event_coverage(self, event_row: pd.Series, 
                                       coverage_config: Dict) -> Dict[str, Any]:
        """计算单个事件的台站覆盖"""
        event_lat = event_row['Latitude']
        event_lon = event_row['Longitude']
        
        # 计算所有台站到事件的距离
        valid_stations = []
        distances = []
        azimuths = []
        
        max_distance = coverage_config['max_epicentral_distance']
        min_distance = coverage_config['min_epicentral_distance']
        
        for _, station in self.permanent_stations.iterrows():
            # 计算震中距（度）
            distance_km = geodesic(
                (event_lat, event_lon),
                (station['Latitude'], station['Longitude'])
            ).kilometers
            
            distance_deg = distance_km / 111.0  # 转换为度
            
            if min_distance <= distance_deg <= max_distance:
                distances.append(distance_deg)
                valid_stations.append(station)
                
                # 计算方位角
                azimuth = self._calculate_azimuth(
                    event_lat, event_lon, 
                    station['Latitude'], station['Longitude']
                )
                azimuths.append(azimuth)
        
        # 计算覆盖质量评分
        station_count = len(valid_stations)
        min_required = coverage_config['min_stations_per_event']
        optimal_count = coverage_config['optimal_station_count']
        
        if station_count >= min_required:
            # 基本覆盖评分
            basic_score = min(1.0, station_count / optimal_count)
            
            # 方位角覆盖评分（降低方位角间隙的影响）
            azimuth_score = self._calculate_azimuthal_coverage_score(azimuths) if azimuths else 0.0
            
            # 距离分布评分
            distance_score = self._calculate_distance_distribution_score(distances) if distances else 0.0
            
            # 综合评分
            coverage_score = (
                basic_score * 0.5 + 
                azimuth_score * coverage_config['azimuthal_gap_weight'] + 
                distance_score * coverage_config['distance_weight']
            )
        else:
            coverage_score = station_count / min_required * 0.3  # 惩罚不足的情况
        
        return {
            'station_count': station_count,
            'mean_distance': np.mean(distances) if distances else 0,
            'min_distance': np.min(distances) if distances else 0,
            'max_distance': np.max(distances) if distances else 0,
            'azimuthal_gap': self._calculate_max_azimuthal_gap(azimuths) if azimuths else 360,
            'coverage_score': coverage_score,
            'network_diversity': len(set([s['Network'] for s in valid_stations]))
        }

    def _calculate_azimuth(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算方位角"""
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon_rad = np.radians(lon2 - lon1)
        
        x = np.sin(dlon_rad) * np.cos(lat2_rad)
        y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
        
        azimuth = np.degrees(np.arctan2(x, y))
        return (azimuth + 360) % 360

    def _calculate_azimuthal_coverage_score(self, azimuths: List[float]) -> float:
        """计算方位角覆盖评分"""
        if len(azimuths) < 4:
            return 0.3
        
        # 计算最大方位角间隙
        sorted_azimuths = sorted(azimuths)
        max_gap = 0
        
        for i in range(len(sorted_azimuths)):
            gap = sorted_azimuths[(i + 1) % len(sorted_azimuths)] - sorted_azimuths[i]
            if gap < 0:
                gap += 360
            max_gap = max(max_gap, gap)
        
        # 理想情况下最大间隙应该小于90度
        score = max(0, 1.0 - max_gap / 180.0)
        return score

    def _calculate_distance_distribution_score(self, distances: List[float]) -> float:
        """计算距离分布评分"""
        if len(distances) < 3:
            return 0.5
        
        # 理想的距离分布应该有多样性
        distance_std = np.std(distances)
        distance_mean = np.mean(distances)
        
        # 变异系数
        cv = distance_std / distance_mean if distance_mean > 0 else 0
        
        # 适中的变异系数得分最高
        optimal_cv = 0.3
        score = max(0, 1.0 - abs(cv - optimal_cv) / optimal_cv)
        
        return score

    def _calculate_max_azimuthal_gap(self, azimuths: List[float]) -> float:
        """计算最大方位角间隙"""
        if len(azimuths) < 2:
            return 360.0
        
        sorted_azimuths = sorted(azimuths)
        max_gap = 0
        
        for i in range(len(sorted_azimuths)):
            gap = sorted_azimuths[(i + 1) % len(sorted_azimuths)] - sorted_azimuths[i]
            if gap < 0:
                gap += 360
            max_gap = max(max_gap, gap)
        
        return max_gap

    def random_weighted_sampling(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        随机加权采样
        
        Args:
            events_df: 带覆盖信息的事件数据框
            
        Returns:
            采样后的事件数据框
        """
        self.logger.info("🎲 开始随机加权采样...")
        
        spatial_config = self.config.spatial_sampling
        max_total_events = self.config.output['max_total_events']
        
        # 1. 基础筛选：必须有足够的台站覆盖
        min_stations = self.config.station_coverage['min_stations_per_event']
        qualified_events = events_df[events_df['station_count'] >= min_stations].copy()
        
        self.logger.info(f"   符合台站覆盖要求的事件: {len(qualified_events)}")
        
        if len(qualified_events) == 0:
            self.logger.warning("⚠️ 没有符合台站覆盖要求的事件")
            return pd.DataFrame()
        
        # 2. 计算综合权重
        qualified_events = self._calculate_sampling_weights(qualified_events, spatial_config)
        
        # 3. 深度分层采样
        sampled_events = self._depth_stratified_sampling(qualified_events, spatial_config)
        
        # 4. 随机采样到目标数量
        if len(sampled_events) > max_total_events:
            # 按权重进行加权随机采样
            weights = sampled_events['sampling_weight'].values
            weights = weights / weights.sum()  # 归一化
            
            selected_indices = np.random.choice(
                sampled_events.index, 
                size=max_total_events, 
                replace=False, 
                p=weights
            )
            sampled_events = sampled_events.loc[selected_indices]
        
        # 5. 应用最小间距约束
        if not spatial_config.get('allow_close_events', False):
            sampled_events = self._apply_distance_constraint(sampled_events, spatial_config)
        
        self.logger.info(f"🎲 随机加权采样完成: {len(events_df)} → {len(sampled_events)} 个事件")
        
        return sampled_events
    
    def _calculate_sampling_weights(self, events_df: pd.DataFrame, 
                                  spatial_config: Dict) -> pd.DataFrame:
        """计算采样权重"""
        coverage_weight = spatial_config['coverage_weight']
        magnitude_weight = spatial_config['magnitude_weight']
        
        # 归一化覆盖评分
        coverage_scores = events_df['coverage_score']
        norm_coverage = (coverage_scores - coverage_scores.min()) / (coverage_scores.max() - coverage_scores.min() + 1e-10)
        
        # 归一化震级
        magnitudes = events_df['Magnitude']
        norm_magnitude = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min() + 1e-10)
        
        # 计算综合权重
        events_df['sampling_weight'] = (
            coverage_weight * norm_coverage + 
            magnitude_weight * norm_magnitude
        )
        
        return events_df
    
    def _depth_stratified_sampling(self, events_df: pd.DataFrame, 
                                 spatial_config: Dict) -> pd.DataFrame:
        """深度分层采样"""
        depth_bins = spatial_config['depth_bins']
        target_per_bin = spatial_config['target_events_per_depth_bin']
        
        sampled_events = []
        
        for bin_name, (min_depth, max_depth) in depth_bins.items():
            bin_events = events_df[
                (events_df['Depth'] >= min_depth) &
                (events_df['Depth'] < max_depth)
            ].copy()
            
            if len(bin_events) == 0:
                continue
            
            # 在每个深度段内按权重采样
            if len(bin_events) <= target_per_bin:
                sampled_events.append(bin_events)
            else:
                weights = bin_events['sampling_weight'].values
                weights = weights / weights.sum()
                
                selected_indices = np.random.choice(
                    bin_events.index,
                    size=target_per_bin,
                    replace=False,
                    p=weights
                )
                sampled_events.append(bin_events.loc[selected_indices])
            
            self.logger.info(f"   深度段 {bin_name}: {len(bin_events)} → {len(sampled_events[-1])} 个事件")
        
        if sampled_events:
            return pd.concat(sampled_events, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _apply_distance_constraint(self, events_df: pd.DataFrame, 
                                 spatial_config: Dict) -> pd.DataFrame:
        """应用最小间距约束"""
        min_distance_km = spatial_config['min_inter_event_distance']
        
        if len(events_df) <= 1:
            return events_df
        
        # 按权重排序，优先保留高权重事件
        events_sorted = events_df.sort_values('sampling_weight', ascending=False)
        
        selected_events = []
        selected_coords = []
        
        for _, event in events_sorted.iterrows():
            event_coord = (event['Latitude'], event['Longitude'])
            
            # 检查与已选事件的距离
            too_close = False
            for selected_coord in selected_coords:
                distance_km = geodesic(event_coord, selected_coord).kilometers
                if distance_km < min_distance_km:
                    too_close = True
                    break
            
            if not too_close:
                selected_events.append(event)
                selected_coords.append(event_coord)
        
        result_df = pd.DataFrame(selected_events)
        self.logger.info(f"   距离约束筛选: {len(events_df)} → {len(result_df)} 个事件")
        
        return result_df
    
    def generate_specfem_stations_file(self, output_file: Optional[Path] = None) -> Path:
        """
        生成SPECFEM3D STATIONS文件
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            STATIONS文件路径
        """
        if self.permanent_stations is None:
            raise ValueError("请先加载永久台站数据")
        
        if output_file is None:
            output_file = self.base_config.dirs['data'] / 'stations' / 'STATIONS'
        
        self.logger.info(f"📄 生成SPECFEM3D STATIONS文件: {output_file}")
        
        # 创建输出目录
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成STATIONS文件内容
        with open(output_file, 'w', encoding='utf-8') as f:
            # 文件头注释
            f.write(f"# STATIONS file for EASTASIA-FWI SPECFEM3D Globe simulation\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total permanent stations: {len(self.permanent_stations)}\n")
            f.write(f"# Region: {self.base_config.region['name']}\n")
            f.write("#\n")
            f.write("# Format: STATION NETWORK LATITUDE LONGITUDE ELEVATION BURIAL\n")
            f.write("#\n")
            
            # 写入台站数据
            specfem_config = self.config.specfem_output
            coord_precision = specfem_config['coordinate_precision']
            elev_precision = specfem_config['elevation_precision']
            burial_depth = specfem_config['default_burial_depth']
            
            for _, station in self.permanent_stations.iterrows():
                station_name = station['Station'][:8]  # SPECFEM限制8字符
                network_name = station['Network'][:8]
                
                f.write(f"{station_name:<8} "
                       f"{network_name:<8} "
                       f"{station['Latitude']:>10.{coord_precision}f} "
                       f"{station['Longitude']:>11.{coord_precision}f} "
                       f"{station.get('Elevation', 0.0):>8.{elev_precision}f} "
                       f"{burial_depth:>6.1f}\n")
        
        self.logger.info(f"✅ STATIONS文件生成完成: {output_file}")
        self.logger.info(f"   包含 {len(self.permanent_stations)} 个永久台站")
        
        return output_file
    
    def generate_cmtsolution_files(self, events_df: pd.DataFrame, 
                                 output_dir: Optional[Path] = None) -> List[Path]:
        """
        生成SPECFEM3D CMTSOLUTION文件
        
        Args:
            events_df: 选中的事件数据框
            output_dir: 输出目录
            
        Returns:
            生成的CMTSOLUTION文件路径列表
        """
        if output_dir is None:
            output_dir = self.base_config.dirs['data'] / 'events' / 'CMT_files'
        
        self.logger.info(f"📄 生成CMTSOLUTION文件到: {output_dir}")
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        success_count = 0
        
        for idx, event in events_df.iterrows():
            try:
                # 创建事件ID
                event_id = self._generate_event_id(event)
                
                # 生成CMTSOLUTION文件内容
                cmt_content = self._generate_cmtsolution_content(event, event_id)
                
                # 保存CMTSOLUTION文件
                cmt_filename = f"CMTSOLUTION_{event_id}"
                cmt_file = output_dir / cmt_filename
                
                with open(cmt_file, 'w', encoding='utf-8') as f:
                    f.write(cmt_content)
                
                generated_files.append(cmt_file)
                success_count += 1
                
            except Exception as e:
                self.logger.warning(f"生成CMTSOLUTION文件失败 (事件 {idx}): {e}")
                continue
        
        self.logger.info(f"✅ CMTSOLUTION文件生成完成: {success_count}/{len(events_df)} 个")
        
        return generated_files
    
    def _generate_event_id(self, event: pd.Series) -> str:
        """生成事件ID"""
        try:
            if 'Date' in event and pd.notna(event['Date']):
                dt = pd.to_datetime(event['Date'])
                time_str = str(event.get('Time', '000000')).replace(':', '')[:6]
                return f"{dt.strftime('%Y%m%d')}_{time_str}"
            else:
                # 使用坐标和震级生成ID
                lat = int(event['Latitude'] * 100)
                lon = int(event['Longitude'] * 100) 
                mag = int(event['Magnitude'] * 100)
                return f"EVENT_{lat}_{lon}_{mag}"
                
        except Exception:
            # 回退方案
            return f"EVENT_{hash(str(event['Latitude']) + str(event['Longitude']))}"
    
    def _generate_cmtsolution_content(self, event: pd.Series, event_id: str) -> str:
        """生成CMTSOLUTION文件内容"""
        # 获取基本参数
        lat = event['Latitude']
        lon = event['Longitude'] 
        depth = event['Depth']
        magnitude = event['Magnitude']
        
        # 获取震级信息
        mb = event.get('Magnitude1_mb', 0.0) if pd.notna(event.get('Magnitude1_mb')) else 0.0
        ms = event.get('Magnitude2_Ms', 0.0) if pd.notna(event.get('Magnitude2_Ms')) else 0.0
        
        # 获取时间信息
        if 'Date' in event and pd.notna(event['Date']):
            date_str = str(event['Date'])[:10]  # YYYY-MM-DD
            time_str = str(event.get('Time', '00:00:00.000'))
        else:
            date_str = '2000/01/01'
            time_str = '00:00:00.000'
        
        # 获取矩张量信息
        mt_components = {}
        mt_cols = ['Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp']
        for col in mt_cols:
            if col in event and pd.notna(event[col]):
                mt_components[col] = float(event[col])
            else:
                mt_components[col] = 0.0
        
        # 计算半持续时间
        half_duration = self._calculate_half_duration(magnitude)
        
        # 获取位置信息
        location = event.get('Location', f'Lat={lat:.3f} Lon={lon:.3f}')
        
        # 生成CMTSOLUTION内容
        cmt_precision = self.config.specfem_output['cmt_precision']
        
        cmt_content = f"""PDE {date_str} {time_str} {lat:8.4f} {lon:9.4f} {depth:5.1f} {mb:3.1f} {ms:3.1f} {location}
event name: {event_id}
time shift: {self.config.specfem_output['time_shift']:.1f}
half duration: {half_duration:.1f}
latitude: {lat:8.4f}
longitude: {lon:9.4f}
depth: {depth:5.1f}
Mrr: {mt_components['Mrr']:.{cmt_precision}e}
Mtt: {mt_components['Mtt']:.{cmt_precision}e}
Mpp: {mt_components['Mpp']:.{cmt_precision}e}
Mrt: {mt_components['Mrt']:.{cmt_precision}e}
Mrp: {mt_components['Mrp']:.{cmt_precision}e}
Mtp: {mt_components['Mtp']:.{cmt_precision}e}
"""
        return cmt_content
    
    def _calculate_half_duration(self, magnitude: float) -> float:
        """计算半持续时间"""
        method = self.config.specfem_output['half_duration_method']
        
        if method == 'magnitude_based':
            # 基于震级的经验公式
            if magnitude <= 5.5:
                return 1.0
            elif magnitude <= 6.0:
                return 2.0 + (magnitude - 5.5) * 2.0
            elif magnitude <= 7.0:
                return 3.0 + (magnitude - 6.0) * 5.0
            else:
                return 8.0 + (magnitude - 7.0) * 10.0
        else:
            # 默认值
            return max(1.0, magnitude - 4.0)

    def _sort_events_by_time(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        按时间顺序对事件进行排序
        
        Args:
            events_df: 事件数据框
            
        Returns:
            按时间排序后的事件数据框
        """
        if events_df is None or len(events_df) == 0:
            return events_df
        
        try:
            # 尝试使用Date列排序
            if 'Date' in events_df.columns:
                # 转换Date列为datetime类型
                events_df = events_df.copy()
                events_df['DateTime'] = pd.to_datetime(events_df['Date'], errors='coerce')
                
                # 如果有Time列，尝试合并时间信息
                if 'Time' in events_df.columns:
                    try:
                        # 合并日期和时间
                        date_time_str = events_df['Date'].astype(str) + ' ' + events_df['Time'].astype(str)
                        events_df['DateTime'] = pd.to_datetime(date_time_str, errors='coerce')
                    except:
                        pass  # 如果合并失败，使用只有日期的DateTime
                
                # 按时间排序
                events_df = events_df.sort_values('DateTime', ascending=True)
                
                # 删除临时的DateTime列
                events_df = events_df.drop('DateTime', axis=1)
                
                self.logger.info(f"   事件已按时间排序: {events_df['Date'].iloc[0]} 到 {events_df['Date'].iloc[-1]}")
                
            elif 'datetime' in events_df.columns:
                # 如果有datetime列
                events_df = events_df.sort_values('datetime', ascending=True)
                self.logger.info(f"   事件已按datetime排序")
                
            else:
                # 如果没有时间列，按索引排序
                self.logger.warning("   没有找到时间列，保持原有顺序")
            
            # 重置索引
            events_df = events_df.reset_index(drop=True)
            
            return events_df
            
        except Exception as e:
            self.logger.error(f"   按时间排序失败: {e}")
            return events_df

    def save_selected_events(self, save_file: Optional[str] = None) -> str:
        """
        保存选择的事件到CSV文件（已按时间排序）
        
        Args:
            save_file: 保存文件名
            
        Returns:
            CSV文件路径
        """
        if self.selected_events is None:
            raise ValueError("请先完成事件采样")
        
        if save_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_file = f"selected_events_{timestamp}.csv"
        
        # 确保事件是按时间排序的
        sorted_events = self._sort_events_by_time(self.selected_events)
        
        save_path = self.base_config.dirs['events'] / save_file
        sorted_events.to_csv(save_path, index=False)
        
        self.logger.info(f"💾 保存选择的事件到: {save_path}")
        self.logger.info(f"   总计 {len(sorted_events)} 个事件（按时间排序）")
        
        return str(save_path)

    def generate_par_catalog(self, save_file: Optional[str] = None) -> str:
        """
        生成PAR格式的事件目录文件（用于波形下载，已按时间排序）
        
        Args:
            save_file: 保存文件名
            
        Returns:
            PAR文件路径
        """
        if self.selected_events is None:
            raise ValueError("请先完成事件采样")
        
        if save_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_file = f"selected_events_catalog_{timestamp}.par"
        
        self.logger.info(f"📝 生成PAR格式事件目录: {len(self.selected_events)} 个事件")
        
        # 确保事件是按时间排序的
        sorted_events = self._sort_events_by_time(self.selected_events)
        
        catalog_lines = []
        successful_count = 0
        failed_count = 0
        
        for _, event in sorted_events.iterrows():
            try:
                # 处理时间数据
                if 'Date' in event and pd.notna(event['Date']) and 'Time' in event and pd.notna(event['Time']):
                    # 如果有完整的日期时间信息
                    date_str = str(event['Date']).strip()
                    time_str = str(event['Time']).strip()
                    
                    # 尝试解析时间
                    try:
                        if len(time_str.split(':')) >= 2:  # 有时分秒格式
                            event_time = pd.to_datetime(f"{date_str} {time_str}")
                        else:
                            event_time = pd.to_datetime(date_str)
                    except:
                        # 回退到只使用日期
                        event_time = pd.to_datetime(date_str)
                        event_time = event_time.replace(hour=0, minute=0, second=0)
                
                else:
                    # 如果没有时间信息，使用默认时间
                    event_time = pd.to_datetime('2000-01-01 00:00:00')
                    self.logger.warning(f"事件缺少时间信息，使用默认时间")
                
                # 创建事件名称
                event_name = f"{event_time.strftime('%Y%m%d.%H.%M')}"
                
                # 获取震级
                magnitude = None
                if 'Magnitude' in event and pd.notna(event['Magnitude']):
                    magnitude = event['Magnitude']
                elif 'Magnitude1_mb' in event and pd.notna(event['Magnitude1_mb']):
                    magnitude = event['Magnitude1_mb']
                elif 'Magnitude2_Ms' in event and pd.notna(event['Magnitude2_Ms']):
                    magnitude = event['Magnitude2_Ms']
                
                if magnitude is None or magnitude <= 0:
                    self.logger.warning(f"事件 {event_name} 没有有效震级，使用默认值")
                    magnitude = 6.0  # 默认震级
                
                # 获取位置信息
                lat = event['Latitude'] if pd.notna(event['Latitude']) else 0.0
                lon = event['Longitude'] if pd.notna(event['Longitude']) else 0.0
                depth = event['Depth'] if pd.notna(event['Depth']) else 10.0
                
                # 生成PAR格式行
                # 格式：event_name YYYYMMDD HH MM SS.sss lat lon depth 8.4 25.6 mag ML
                par_line = (
                    f"{event_name} {event_time.strftime('%Y%m%d')} "
                    f"{event_time.strftime('%H')} {event_time.strftime('%M')} "
                    f"{event_time.strftime('%S')}.000 "
                    f"{lat:.4f} {lon:.4f} "
                    f"{depth:.1f} 8.4 25.6 {magnitude:.1f} ML"
                )
                
                catalog_lines.append(par_line)
                successful_count += 1
            
            except Exception as e:
                self.logger.warning(f"处理事件失败: {e}")
                failed_count += 1
                continue
        
        if not catalog_lines:
            raise ValueError("没有生成有效的事件目录行")
        
        # 保存PAR文件
        par_file_path = self.base_config.dirs['events'] / save_file
        par_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(par_file_path, 'w', encoding='utf-8') as f:
            # 添加文件头注释
            f.write(f"# EASTASIA-FWI Selected Events Catalog (PAR format)\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total events: {len(catalog_lines)}\n")
            f.write(f"# Format: event_name YYYYMMDD HH MM SS.sss lat lon depth 8.4 25.6 mag ML\n")
            f.write(f"# Region: {self.base_config.region['name']}\n")
            f.write("#\n")
            
            for line in catalog_lines:
                f.write(line + '\n')
        
        self.logger.info(f"✅ PAR事件目录已保存: {par_file_path}")
        self.logger.info(f"   成功处理: {successful_count} 个事件")
        if failed_count > 0:
            self.logger.warning(f"   处理失败: {failed_count} 个事件")
        
        return str(par_file_path)

    def sample_events_complete_workflow(self, gcmt_file: Optional[Path] = None,
                                      station_file: Optional[Path] = None) -> pd.DataFrame:
        """
        执行完整的事件采样工作流
        
        Args:
            gcmt_file: GCMT事件文件路径
            station_file: 永久台站文件路径
            
        Returns:
            采样后的事件数据框（按时间排序）
        """
        self.logger.info("🚀 开始完整的事件采样工作流...")
        
        try:
            # 1. 加载永久台站数据
            self.logger.info("📡 步骤1: 加载永久台站数据")
            self.load_permanent_stations(station_file)
            
            # 2. 加载GCMT事件数据
            self.logger.info("📊 步骤2: 加载GCMT事件数据")
            events_df = self.load_gcmt_events(gcmt_file)
            
            # 3. 应用基础事件筛选
            self.logger.info("🎯 步骤3: 应用事件筛选条件")
            events_df = self.apply_event_filters(events_df)
            
            if len(events_df) == 0:
                self.logger.warning("⚠️ 筛选后没有符合条件的事件")
                return pd.DataFrame()
            
            # 4. 计算台站-事件覆盖关系
            self.logger.info("🔍 步骤4: 计算台站-事件覆盖关系")
            events_with_coverage = self.calculate_station_coverage(events_df)
            
            # 5. 随机加权采样
            self.logger.info("🎲 步骤5: 随机加权采样")
            sampled_events = self.random_weighted_sampling(events_with_coverage)
            
            if len(sampled_events) == 0:
                self.logger.warning("⚠️ 采样后没有事件")
                return pd.DataFrame()
            
            # 6. 按时间排序事件
            self.logger.info("🕒 步骤6: 按时间排序事件")
            self.selected_events = self._sort_events_by_time(sampled_events)
            
            # 7. 生成统计信息
            self._generate_sampling_statistics()
            
            self.logger.info(f"✅ 事件采样工作流完成: 最终选择 {len(self.selected_events)} 个事件（按时间排序）")
            
            return self.selected_events
            
        except Exception as e:
            self.logger.error(f"❌ 事件采样工作流失败: {e}")
            raise
    
    def _generate_sampling_statistics(self):
        """生成采样统计信息"""
        if self.selected_events is None or len(self.selected_events) == 0:
            self.sampling_stats = {}
            return
        
        events = self.selected_events
        
        self.sampling_stats = {
            'total_events': len(events),
            'magnitude_range': (events['Magnitude'].min(), events['Magnitude'].max()),
            'magnitude_mean': events['Magnitude'].mean(),
            'depth_range': (events['Depth'].min(), events['Depth'].max()),
            'depth_mean': events['Depth'].mean(),
            'spatial_coverage': {
                'lon_range': (events['Longitude'].min(), events['Longitude'].max()),
                'lat_range': (events['Latitude'].min(), events['Latitude'].max())
            },
            'station_coverage': {
                'mean_stations_per_event': events['station_count'].mean(),
                'min_stations_per_event': events['station_count'].min(),
                'max_stations_per_event': events['station_count'].max(),
                'mean_coverage_score': events['coverage_score'].mean()
            },
            'depth_distribution': self._calculate_depth_distribution(events)
        }
    
    def _calculate_depth_distribution(self, events: pd.DataFrame) -> Dict:
        """计算深度分布统计"""
        depth_bins = self.config.spatial_sampling['depth_bins']
        distribution = {}
        
        for bin_name, (min_depth, max_depth) in depth_bins.items():
            count = len(events[
                (events['Depth'] >= min_depth) & 
                (events['Depth'] < max_depth)
            ])
            distribution[bin_name] = count
        
        return distribution

    def create_station_event_map(self, save_file: Optional[str] = None) -> str:
        """
        创建台站-事件分布地图（英文标签）
        
        Args:
            save_file: 保存文件名
            
        Returns:
            图片文件路径
        """
        if self.selected_events is None or self.permanent_stations is None:
            raise ValueError("请先完成事件采样")
        
        if save_file is None:
            save_file = "5-5-event_sampling_2d_distribution.png"
        
        self.logger.info("🗺️ 创建台站-事件分布地图...")
        
        # 获取英文标签
        labels = self.config.visualization['labels']
        
        fig = plt.figure(figsize=self.config.visualization['figure_size'])
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # 设置地图范围
        region = self.base_config.region
        ax.set_extent([region['lon_min'], region['lon_max'], 
                      region['lat_min'], region['lat_max']], 
                     crs=ccrs.PlateCarree())
        
        # 添加地理要素
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # 绘制永久台站
        viz_config = self.config.visualization
        ax.scatter(self.permanent_stations['Longitude'], self.permanent_stations['Latitude'],
                  s=viz_config['station_size'], c=viz_config['station_color'], 
                  marker=viz_config['station_marker'], alpha=0.8, 
                  label=f'{labels["station_label"]} ({len(self.permanent_stations)})',
                  transform=ccrs.PlateCarree(), edgecolors='black', linewidth=0.3)
        
        # 绘制选中事件 - 按深度着色
        events = self.selected_events
        scatter = ax.scatter(events['Longitude'], events['Latitude'],
                           s=viz_config['magnitude_scaling'], c=events['Depth'],
                           cmap=viz_config['depth_colormap'], alpha=0.8, 
                           edgecolors='black', linewidth=0.5,
                           label=f'{labels["event_label"]} ({len(events)})',
                           transform=ccrs.PlateCarree())
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(labels['depth_label'], fontsize=12)
        
        # 添加网格和标签
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        
        # 添加图例
        ax.legend(loc='lower right', fontsize=12)
        
        # 设置标题（英文）
        plt.title('EASTASIA-FWI: Permanent Stations and Selected Events Distribution\n'
                 f'Stations: {len(self.permanent_stations)}, Events: {len(events)}, '
                 f'Avg Coverage Score: {events["coverage_score"].mean():.3f}',
                 fontsize=14, fontweight='bold')
        
        # 保存图片
        save_path = self.output_dir / save_file
        for fmt in self.config.visualization['save_formats']:
            output_file = save_path.with_suffix(f'.{fmt}')
            plt.savefig(output_file, dpi=self.config.visualization['dpi'], 
                       bbox_inches='tight', facecolor='white')
        
        plt.close()
        
        self.logger.info(f"✅ 台站-事件分布地图保存: {save_path}")
        
        return str(save_path)

    def create_3d_distribution_plot(self, save_file: Optional[str] = None) -> str:
        """
        创建三维事件分布图（静态版本 - 无海岸线）
        
        Args:
            save_file: 保存文件名
            
        Returns:
            图片文件路径
        """
        if self.selected_events is None or self.permanent_stations is None:
            raise ValueError("请先完成事件采样")
        
        if save_file is None:
            save_file = "5-5-event_sampling_3d_distribution.png"
        
        self.logger.info("📊 创建三维事件分布图...")
        
        # 获取英文标签和配置
        labels = self.config.visualization['labels']
        viz_config = self.config.visualization
        plot_3d_config = viz_config['plot_3d']
        
        # 创建3D图形
        fig = plt.figure(figsize=viz_config['figure_size_3d'])
        ax = fig.add_subplot(111, projection='3d')
        
        events = self.selected_events
        stations = self.permanent_stations
        region = self.base_config.region
        
        # 1. 添加深度参考面
        if plot_3d_config['show_depth_plane']:
            xx, yy = np.meshgrid(
                np.linspace(region['lon_min'], region['lon_max'], 10),
                np.linspace(region['lat_min'], region['lat_max'], 10)
            )
            
            # 添加几个代表性深度平面
            depth_planes = [0, -100, -200, -410, -660]
            plane_colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'red']
            plane_alphas = [0.1, 0.08, 0.08, 0.1, 0.1]
            
            for i, depth in enumerate(depth_planes):
                if depth >= -events['Depth'].max() - 50:  # 稍微放宽条件
                    ax.plot_surface(xx, yy, depth * np.ones_like(xx),
                                  alpha=plane_alphas[i],
                                  color=plane_colors[i])
        
        # 2. 绘制台站（放在700km深度）
        ax.scatter(stations['Longitude'], stations['Latitude'], 
                  np.full(len(stations), -700),
                  s=viz_config['station_size'], c='darkgray',
                  marker='^', alpha=0.9,  # 使用三角形标记
                  label=f'{labels["station_label"]} ({len(stations)})',
                  edgecolors='black', linewidth=0.5)
        
        # 3. 绘制事件 - 以深度作为Z轴，颜色也按深度着色
        scatter = ax.scatter(events['Longitude'], events['Latitude'], -events['Depth'],
                           s=60, c=events['Depth'], 
                           cmap=viz_config['depth_colormap'], alpha=0.8,
                           label=f'{labels["event_label"]} ({len(events)})',
                           edgecolors='black', linewidth=0.3)
        
        # 4. 设置坐标轴标签
        ax.set_xlabel(labels['longitude_label'], fontsize=12)
        ax.set_ylabel(labels['latitude_label'], fontsize=12)
        ax.set_zlabel(f'Elevation / -{labels["depth_label"]}', fontsize=12)
        
        # 5. 设置视角
        ax.view_init(elev=plot_3d_config['elevation_angle'], 
                    azim=plot_3d_config['azimuth_angle'])
        
        # 6. 设置坐标轴范围
        ax.set_xlim(region['lon_min'], region['lon_max'])
        ax.set_ylim(region['lat_min'], region['lat_max'])
        ax.set_zlim(-700, 50)  # 从700km深度到地表以上50km
        
        # 7. 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
        cbar.set_label(labels['depth_label'], fontsize=12)
        
        # 8. 添加网格
        ax.grid(True, alpha=plot_3d_config['grid_alpha'])
        
        # 9. 添加图例
        ax.legend(loc='upper left', fontsize=10)
        
        # 10. 设置标题
        plt.suptitle('EASTASIA-FWI: 3D Distribution\n'
                    f'Stations at 700km depth, Events: {events["Depth"].min():.0f}-{events["Depth"].max():.0f} km',
                    fontsize=14, fontweight='bold')
        
        # 11. 保存图片
        save_path = self.output_dir / save_file
        for fmt in self.config.visualization['save_formats']:
            output_file = save_path.with_suffix(f'.{fmt}')
            plt.savefig(output_file, dpi=self.config.visualization['dpi'], 
                       bbox_inches='tight', facecolor='white')
        
        plt.close()
        
        self.logger.info(f"✅ 三维事件分布图保存: {save_path}")
        
        return str(save_path)

    def create_interactive_3d_plot(self, save_file: Optional[str] = None) -> str:
        """
        创建交互式三维事件分布图（优化版本 - 无海岸线）
        
        Args:
            save_file: 保存文件名
            
        Returns:
            HTML文件路径
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("⚠️ Plotly未安装，无法创建交互式3D图")
            return ""
        
        if self.selected_events is None or self.permanent_stations is None:
            raise ValueError("请先完成事件采样")
        
        if save_file is None:
            save_file = "5-5-event_sampling_interactive_3d.html"
        
        self.logger.info("🌐 创建交互式三维事件分布图...")
        
        # 获取配置和标签
        labels = self.config.visualization['labels']
        interactive_config = self.config.visualization['interactive_3d']
        
        events = self.selected_events
        stations = self.permanent_stations
        region = self.base_config.region
        
        # 创建3D散点图
        fig = go.Figure()
        
        # 1. 添加深度参考面（修复显示问题）
        if interactive_config['show_depth_surfaces']:
            self.logger.info("   添加深度参考面...")
            lon_range = np.linspace(region['lon_min'], region['lon_max'], 20)
            lat_range = np.linspace(region['lat_min'], region['lat_max'], 20)
            lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
            
            # 定义重要的地质界面深度
            depth_surfaces = [0, -100, -200, -410, -660]
            surface_names = ['Surface', '100km', '200km', '410km Discontinuity', '660km Discontinuity']
            surface_colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'red']
            
            for i, depth in enumerate(depth_surfaces):
                # 放宽显示条件，确保界面能够显示
                if depth >= -700:  # 只要深度不超过700km就显示
                    depth_grid = np.full_like(lon_grid, depth)
                    
                    fig.add_trace(go.Surface(
                        x=lon_grid,
                        y=lat_grid,
                        z=depth_grid,
                        colorscale=[[0, surface_colors[i]], [1, surface_colors[i]]],
                        opacity=0.35,  # 增加透明度
                        showscale=False,
                        name=surface_names[i],
                        hovertemplate=f'<b>{surface_names[i]}</b><br>Depth: {-depth} km<extra></extra>',
                        showlegend=True,
                        visible=True  # 确保可见
                    ))
                    self.logger.info(f"      添加深度面: {surface_names[i]} ({depth} km)")
        
        # 2. 添加台站数据（放在最底部700km深度，使用三角锥形）
        station_customdata = np.column_stack([
            stations['Network'].values,
            stations['Station'].values
        ])
        
        fig.add_trace(go.Scatter3d(
            x=stations['Longitude'],
            y=stations['Latitude'],
            z=np.full(len(stations), -700),  # 放在700km深度
            mode='markers',
            marker=dict(
                size=interactive_config['marker_size_stations'],
                color='darkgray',
                symbol='diamond-open',  # 使用空心钻石（类似三角锥）
                opacity=interactive_config['opacity_stations'],
                line=dict(width=2, color='black')
            ),
            customdata=station_customdata,
            hovertemplate=interactive_config['hover_template_stations'],
            name=f'{labels["station_label"]} ({len(stations)})',
            showlegend=True
        ))
        
        # 3. 计算深度范围用于色标
        depth_min = events['Depth'].min()
        depth_max = events['Depth'].max()
        
        # 4. 添加事件数据（按深度着色，优化色标）
        event_customdata = np.column_stack([
            events['Depth'].values,
            events['Magnitude'].values,
            events['station_count'].values,
            events['coverage_score'].values
        ])
        
        fig.add_trace(go.Scatter3d(
            x=events['Longitude'],
            y=events['Latitude'],
            z=-events['Depth'],  # 负深度
            mode='markers',
            marker=dict(
                size=interactive_config['marker_size_events'],
                color=events['Depth'],
                colorscale=interactive_config['colorscale'],
                opacity=interactive_config['opacity_events'],
                cmin=depth_min,  # 设置色标最小值
                cmax=depth_max,  # 设置色标最大值
                colorbar=dict(
                    title=dict(
                        text=labels['depth_label'],
                        side="right"
                    ),
                    tickmode="array",
                    tickvals=np.linspace(depth_min, depth_max, 6),  # 6个刻度
                    ticktext=[f"{int(val)}" for val in np.linspace(depth_min, depth_max, 6)],
                    thickness=15,
                    len=0.7,
                    x=1.02,
                    xanchor="left"
                ),
                line=dict(width=0.5, color='black')
            ),
            customdata=event_customdata,
            hovertemplate=interactive_config['hover_template_events'],
            name=f'{labels["event_label"]} ({len(events)})',
            showlegend=True
        ))
        
        # 5. 设置布局（优化坐标轴和视角）
        fig.update_layout(
            title=dict(
                text='EASTASIA-FWI: Interactive 3D Distribution<br>' +
                     f'<sub>Stations: {len(stations)} (at 700km depth), Events: {len(events)}, ' +
                     f'Depth Range: {depth_min:.0f}-{depth_max:.0f} km</sub>',
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title=labels['longitude_label'],
                yaxis_title=labels['latitude_label'],
                zaxis_title=f'Elevation / -{labels["depth_label"]}',
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.5)  # 优化初始视角
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=0.8, z=1.2),  # 调整Z轴比例
                xaxis=dict(
                    range=[region['lon_min'], region['lon_max']],
                    backgroundcolor="rgba(240,240,240,0.8)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                    tickmode='linear',
                    tick0=region['lon_min'],
                    dtick=10  # 每10度一个刻度
                ),
                yaxis=dict(
                    range=[region['lat_min'], region['lat_max']],
                    backgroundcolor="rgba(240,240,240,0.8)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                    tickmode='linear',
                    tick0=region['lat_min'],
                    dtick=10  # 每10度一个刻度
                ),
                zaxis=dict(
                    range=[-700, 50],  # 从700km深度到地表以上50km
                    backgroundcolor="rgba(240,240,240,0.8)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                    tickmode='array',
                    tickvals=[-700, -600, -500, -400, -300, -200, -100, 0],
                    ticktext=['700', '600', '500', '400', '300', '200', '100', '0']
                )
            ),
            width=1200,
            height=900,
            margin=dict(l=0, r=50, b=0, t=80),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10)
            ),
            # 添加操作说明
            annotations=[
                dict(
                    text="💡 操作提示：<br>" +
                         "• 拖拽：旋转视角<br>" +
                         "• 滚轮：缩放<br>" +
                         "• Shift+拖拽：平移<br>" +
                         "• 悬停：查看详情<br>" +
                         "• 双击：重置视角<br>" +
                         "• 图例：点击显示/隐藏",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.02, y=0.02,
                    xanchor="left", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1,
                    font=dict(size=10)
                )
            ]
        )
        
        # 保存HTML文件
        save_path = self.output_dir / save_file
        
        # 添加自定义配置
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'eastasia_3d_events',
                'height': 900,
                'width': 1200,
                'scale': 2
            }
        }
        
        fig.write_html(
            str(save_path),
            config=config,
            include_plotlyjs=True,  # 包含plotly.js，确保离线可用
            div_id="eastasia-3d-plot"
        )
        
        self.logger.info(f"✅ 交互式三维事件分布图保存: {save_path}")
        self.logger.info("   💡 新增功能：")
        self.logger.info("      - 深度界面：修复显示问题，确保所有界面可见")
        self.logger.info("      - 地质意义：410km和660km不连续面")
        
        return str(save_path)

    def create_sampling_statistics_plot(self, save_file: Optional[str] = None) -> str:
        """
        创建采样统计图表（英文标签）
        
        Args:
            save_file: 保存文件名
            
        Returns:
            图片文件路径
        """
        if self.selected_events is None:
            raise ValueError("请先完成事件采样")
        
        if save_file is None:
            save_file = "5-5-event_sampling_statistics.png"
        
        self.logger.info("📊 创建采样统计图表...")
        
        # 获取英文标签
        labels = self.config.visualization['labels']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('EASTASIA-FWI: Event Sampling Statistical Analysis', 
                    fontsize=16, fontweight='bold')
        
        events = self.selected_events
        
        # 1. 震级分布直方图
        ax = axes[0, 0]
        ax.hist(events['Magnitude'], bins=12, alpha=0.7, 
               color='skyblue', edgecolor='black')
        ax.set_xlabel(labels['magnitude_label'])
        ax.set_ylabel(labels['event_count_label'])
        ax.set_title('Magnitude Distribution')
        ax.grid(True, alpha=0.3)
        ax.axvline(events['Magnitude'].mean(), color='red', 
                  linestyle='--', label=f'Mean: {events["Magnitude"].mean():.2f}')
        ax.legend()
        
        # 2. 深度分布直方图
        ax = axes[0, 1]
        ax.hist(events['Depth'], bins=15, alpha=0.7, 
               color='lightgreen', edgecolor='black')
        ax.set_xlabel(labels['depth_label'])
        ax.set_ylabel(labels['event_count_label'])
        ax.set_title('Depth Distribution')
        ax.grid(True, alpha=0.3)
        ax.axvline(events['Depth'].mean(), color='red', 
                  linestyle='--', label=f'Mean: {events["Depth"].mean():.1f}km')
        ax.legend()
        
        # 3. 台站数分布
        ax = axes[0, 2]
        ax.hist(events['station_count'], bins=12, alpha=0.7, 
               color='orange', edgecolor='black')
        ax.set_xlabel(labels['station_count_label'])
        ax.set_ylabel(labels['event_count_label'])
        ax.set_title('Station Coverage Distribution')
        ax.grid(True, alpha=0.3)
        ax.axvline(events['station_count'].mean(), color='red', 
                  linestyle='--', label=f'Mean: {events["station_count"].mean():.1f}')
        ax.legend()
        
        # 4. 震级vs深度散点图
        ax = axes[1, 0]
        scatter = ax.scatter(events['Magnitude'], events['Depth'],
                           c=events['coverage_score'], cmap='viridis', 
                           alpha=0.7, s=60, edgecolors='black')
        ax.set_xlabel(labels['magnitude_label'])
        ax.set_ylabel(labels['depth_label'])
        ax.set_title('Magnitude vs Depth')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        plt.colorbar(scatter, ax=ax, label=labels['coverage_score_label'])
        
        # 5. 覆盖评分分布
        ax = axes[1, 1]
        ax.hist(events['coverage_score'], bins=12, alpha=0.7,
               color='purple', edgecolor='black')
        ax.set_xlabel(labels['coverage_score_label'])
        ax.set_ylabel(labels['event_count_label'])
        ax.set_title('Coverage Score Distribution')
        ax.grid(True, alpha=0.3)
        ax.axvline(events['coverage_score'].mean(), color='red',
                  linestyle='--', label=f'Mean: {events["coverage_score"].mean():.3f}')
        ax.legend()
        
        # 6. 震中距分布
        ax = axes[1, 2]
        if 'mean_distance' in events.columns:
            ax.hist(events['mean_distance'], bins=12, alpha=0.7,
                   color='coral', edgecolor='black')
            ax.set_xlabel(labels['distance_label'])
            ax.set_ylabel(labels['event_count_label'])
            ax.set_title('Mean Epicentral Distance Distribution')
            ax.grid(True, alpha=0.3)
            ax.axvline(events['mean_distance'].mean(), color='red',
                      linestyle='--', label=f'Mean: {events["mean_distance"].mean():.1f}°')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Distance data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Mean Epicentral Distance')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.output_dir / save_file
        for fmt in self.config.visualization['save_formats']:
            output_file = save_path.with_suffix(f'.{fmt}')
            plt.savefig(output_file, dpi=self.config.visualization['dpi'], 
                       bbox_inches='tight', facecolor='white')
        
        plt.close()
        
        self.logger.info(f"✅ 采样统计图表保存: {save_path}")
        
        return str(save_path)

    def generate_comprehensive_report(self, save_file: Optional[Path] = None) -> str:
        """
        生成综合分析报告
        
        Args:
            save_file: 保存文件路径
            
        Returns:
            报告文件路径
        """
        if save_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_file = self.base_config.dirs['results'] / f"event_sampling_report_{timestamp}.txt"
        
        self.logger.info("📋 生成综合分析报告...")
        
        # 确保目录存在
        save_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("EASTASIA-FWI 事件采样分析报告\n")
                f.write("="*80 + "\n\n")
                
                # 基本信息
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"研究区域: {self.base_config.region['name']}\n")
                f.write(f"采样方法: 随机加权采样\n\n")
                
                # 永久台站信息
                if self.permanent_stations is not None:
                    f.write("📡 永久台站信息\n")
                    f.write("-"*50 + "\n")
                    f.write(f"永久台站总数: {len(self.permanent_stations)}\n")
                    f.write(f"涉及台网数: {self.permanent_stations['Network'].nunique()}\n\n")
                
                # 事件采样结果
                if self.selected_events is not None:
                    stats = self.sampling_stats
                    f.write("📊 事件采样结果\n")
                    f.write("-"*50 + "\n")
                    f.write(f"最终选择事件数: {stats['total_events']}\n")
                    f.write(f"震级范围: {stats['magnitude_range'][0]:.2f} - {stats['magnitude_range'][1]:.2f}\n")
                    f.write(f"平均震级: {stats['magnitude_mean']:.2f}\n")
                    f.write(f"深度范围: {stats['depth_range'][0]:.1f} - {stats['depth_range'][1]:.1f} km\n")
                    f.write(f"平均深度: {stats['depth_mean']:.1f} km\n\n")
                    
                    # 台站覆盖统计
                    coverage = stats['station_coverage']
                    f.write("🎯 台站覆盖统计\n")
                    f.write("-"*50 + "\n")
                    f.write(f"平均每事件台站数: {coverage['mean_stations_per_event']:.1f}\n")
                    f.write(f"台站数范围: {coverage['min_stations_per_event']} - {coverage['max_stations_per_event']}\n")
                    f.write(f"平均覆盖评分: {coverage['mean_coverage_score']:.3f}\n\n")
                    
                    # 深度分布
                    depth_dist = stats['depth_distribution']
                    f.write("📏 深度分布\n")
                    f.write("-"*50 + "\n")
                    for bin_name, count in depth_dist.items():
                        f.write(f"{bin_name.title()}: {count} 个事件\n")
                    f.write("\n")
                
                # 配置摘要
                f.write("⚙️ 配置摘要\n")
                f.write("-"*50 + "\n")
                f.write(f"震级范围: {self.config.event_filters['magnitude_range']}\n")
                f.write(f"深度范围: {self.config.event_filters['depth_range']} km\n")
                f.write(f"最小台站数: {self.config.station_coverage['min_stations_per_event']}\n")
                f.write(f"最大震中距: {self.config.station_coverage['max_epicentral_distance']}°\n")
                f.write(f"最大总事件数: {self.config.output['max_total_events']}\n")
                f.write(f"采样方法: {self.config.spatial_sampling['method']}\n\n")
                
                f.write(f"报告生成完成。\n")
                f.write("="*80 + "\n")
            
            self.logger.info(f"✅ 综合分析报告生成: {save_file}")
            
            return str(save_file)
            
        except Exception as e:
            self.logger.error(f"❌ 生成报告失败: {e}")
            raise

    # ================================================================
    #  CMT3D 目录集成（Sawade et al., 2022, GJI）
    # ================================================================

    def load_cmt3d_catalog(self, catalog_file: Optional[Path] = None) -> pd.DataFrame:
        """
        读取 CMT3D 目录（cmt3d.txt，9382 事件，每事件 13 行）

        Args:
            catalog_file: cmt3d.txt 路径，默认从 config 读取

        Returns:
            包含所有 CMT3D 事件的 DataFrame
        """
        if catalog_file is None:
            catalog_file = self.base_config.dirs['project_root'] / \
                           self.config.cmt3d['data_dir'] / self.config.cmt3d['catalog_file']
        catalog_file = Path(catalog_file)

        if not catalog_file.exists():
            raise FileNotFoundError(f"CMT3D 目录文件不存在: {catalog_file}")

        self.logger.info(f"📖 读取 CMT3D 目录: {catalog_file}")
        events = []
        with open(catalog_file, 'r') as f:
            lines = f.readlines()

        idx = 0
        while idx < len(lines):
            if idx + 13 > len(lines):
                break
            block = [l.rstrip('\n') for l in lines[idx:idx+13]]
            evt = self._parse_cmt3d_event_block(block)
            if evt is not None:
                events.append(evt)
            idx += 13

        df = pd.DataFrame(events)
        self.logger.info(f"✅ CMT3D 目录加载完成: {len(df)} 事件")
        return df

    def _parse_cmt3d_event_block(self, block: List[str]) -> Optional[Dict]:
        """解析 CMT3D 单个事件的 13 行数据块"""
        try:
            # PDE 行: " PDE 2001  1  1  8 54 31.60   6.6300  126.9000  33.0 0.0 6.0 None"
            pde = block[0]
            parts = pde.split()
            yr = int(parts[1])
            mo = int(parts[2])
            da = int(parts[3])
            ho = int(parts[4])
            mi = int(parts[5])
            sec = float(parts[6])
            pde_lat = float(parts[7])
            pde_lon = float(parts[8])
            pde_dep = float(parts[9])
            pde_mag = float(parts[11]) if len(parts) > 11 else 0.0

            def _val(line):
                return float(line.split(':')[1].strip())

            return {
                'event_name': block[1].split(':')[1].strip(),
                'origin_year': yr, 'origin_month': mo, 'origin_day': da,
                'origin_hour': ho, 'origin_minute': mi, 'origin_second': sec,
                'pde_latitude': pde_lat, 'pde_longitude': pde_lon,
                'pde_depth': pde_dep, 'pde_magnitude': pde_mag,
                'time_shift': _val(block[2]),
                'half_duration': _val(block[3]),
                'latitude': _val(block[4]),
                'longitude': _val(block[5]),
                'depth': _val(block[6]),
                'Mrr': _val(block[7]),
                'Mtt': _val(block[8]),
                'Mpp': _val(block[9]),
                'Mrt': _val(block[10]),
                'Mrp': _val(block[11]),
                'Mtp': _val(block[12]),
                'pde_line': pde,
            }
        except (ValueError, IndexError) as e:
            return None

    def filter_cmt3d_for_region(self, cmt3d_df: pd.DataFrame,
                                 region: Optional[Dict] = None,
                                 mag_range: Optional[tuple] = None,
                                 dep_range: Optional[tuple] = None) -> pd.DataFrame:
        """
        按区域、震级、深度筛选 CMT3D 事件

        Args:
            cmt3d_df: CMT3D 目录 DataFrame
            region: {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
            mag_range: (min_mag, max_mag)
            dep_range: (min_dep, max_dep)

        Returns:
            筛选后的 DataFrame
        """
        cfg = self.config.cmt3d
        region = region or cfg['region']
        mag_range = mag_range or cfg['magnitude_range']
        dep_range = dep_range or cfg['depth_range']

        n_before = len(cmt3d_df)
        df = cmt3d_df.copy()

        # 区域筛选（使用质心位置，非 PDE 位置）
        df = df[(df['latitude'] >= region['lat_min']) &
                (df['latitude'] <= region['lat_max']) &
                (df['longitude'] >= region['lon_min']) &
                (df['longitude'] <= region['lon_max'])]
        n_region = len(df)

        # 震级筛选
        df = df[(df['pde_magnitude'] >= mag_range[0]) &
                (df['pde_magnitude'] <= mag_range[1])]
        n_mag = len(df)

        # 深度筛选
        df = df[(df['depth'] >= dep_range[0]) &
                (df['depth'] <= dep_range[1])]

        self.logger.info(f"📊 CMT3D 筛选: {n_before} → 区域 {n_region} → 震级 {n_mag} → 深度 {len(df)}")
        return df.reset_index(drop=True)

    def select_cmt3d_test_events(self, filtered_df: pd.DataFrame,
                                  n_events: Optional[int] = None) -> pd.DataFrame:
        """
        从筛选后的 CMT3D 事件中选择测试事件（深度分层 + 空间分散）

        Args:
            filtered_df: 已筛选的 CMT3D DataFrame
            n_events: 选择事件数，默认从 config 读取

        Returns:
            选中的测试事件 DataFrame
        """
        n_events = n_events or self.config.cmt3d['n_test_events']
        depth_strat = self.config.cmt3d['depth_strategy']

        if len(filtered_df) < n_events:
            self.logger.warning(f"⚠️ 可用事件 {len(filtered_df)} < 目标 {n_events}，返回全部")
            return filtered_df

        selected = []
        remaining = filtered_df.copy()

        # 按深度分层选择
        for bin_name, (d_min, d_max) in depth_strat.items():
            bin_events = remaining[(remaining['depth'] >= d_min) &
                                   (remaining['depth'] < d_max)]
            if len(bin_events) == 0:
                continue
            # 优先选信噪比高的（震级较大）
            bin_events = bin_events.sort_values('pde_magnitude', ascending=False)
            n_pick = min(2, len(bin_events), n_events - len(selected))
            if n_pick > 0:
                selected.append(bin_events.head(n_pick))

        if selected:
            selected_df = pd.concat(selected, ignore_index=True)
        else:
            selected_df = pd.DataFrame()

        # 若不足 n_events，从剩余中补充（按震级排序）
        if len(selected_df) < n_events:
            used_names = set(selected_df['event_name']) if len(selected_df) > 0 else set()
            extra = remaining[~remaining['event_name'].isin(used_names)]
            extra = extra.sort_values('pde_magnitude', ascending=False)
            n_extra = n_events - len(selected_df)
            if len(extra) > 0:
                selected_df = pd.concat([selected_df, extra.head(n_extra)], ignore_index=True)

        selected_df = selected_df.head(n_events)
        self.logger.info(f"✅ 选择 {len(selected_df)} 个 CMT3D 测试事件:")
        for _, evt in selected_df.iterrows():
            self.logger.info(f"   {evt['event_name']}: "
                           f"Mw{evt['pde_magnitude']:.1f}, "
                           f"{evt['latitude']:.2f}°N {evt['longitude']:.2f}°E, "
                           f"depth {evt['depth']:.1f} km")
        return selected_df

    def export_cmt3d_to_specfem(self, events_df: pd.DataFrame,
                                 output_dir: Optional[Path] = None,
                                 use_ecef: Optional[bool] = None) -> List[Path]:
        """
        将 CMT3D 事件导出为 SPECFEM CMTSOLUTION 文件

        Args:
            events_df: CMT3D 事件 DataFrame
            output_dir: 输出目录
            use_ecef: 是否使用 ECEF 格式（匹配 USE_ECEF_CMTSOLUTION 编译选项）

        Returns:
            生成的 CMTSOLUTION 文件路径列表
        """
        import math

        if output_dir is None:
            output_dir = self.output_dir / 'CMTSOLUTION_CMT3D'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        use_ecef = use_ecef if use_ecef is not None else self.config.cmt3d['use_ecef_format']
        files = []

        for _, evt in events_df.iterrows():
            if use_ecef:
                content = self._format_ecef_cmtsolution(evt)
            else:
                content = self._format_standard_cmtsolution(evt)

            fname = output_dir / f"CMTSOLUTION_{evt['event_name']}"
            with open(fname, 'w') as f:
                f.write(content)
            files.append(fname)

        fmt_label = "ECEF (USE_ECEF_CMTSOLUTION=.true.)" if use_ecef else "标准 (lat/lon/depth, dyne-cm)"
        self.logger.info(f"📄 生成 {len(files)} 个 CMTSOLUTION 文件 [{fmt_label}]: {output_dir}")
        return files

    def _format_standard_cmtsolution(self, evt: pd.Series) -> str:
        """CMT3D → 标准 SPECFEM CMTSOLUTION（lat/lon/depth, dyne-cm）"""
        return (
            f"{evt['pde_line']}\n"
            f"event name:     {evt['event_name']}\n"
            f"time shift:{evt['time_shift']:14.4f}\n"
            f"half duration:{evt['half_duration']:11.4f}\n"
            f"latitude:{evt['latitude']:16.4f}\n"
            f"longitude:{evt['longitude']:15.4f}\n"
            f"depth:{evt['depth']:19.4f}\n"
            f"Mrr:{evt['Mrr']:20.6e}\n"
            f"Mtt:{evt['Mtt']:20.6e}\n"
            f"Mpp:{evt['Mpp']:20.6e}\n"
            f"Mrt:{evt['Mrt']:20.6e}\n"
            f"Mrp:{evt['Mrp']:20.6e}\n"
            f"Mtp:{evt['Mtp']:20.6e}\n"
        )

    def _format_ecef_cmtsolution(self, evt: pd.Series) -> str:
        """CMT3D → ECEF CMTSOLUTION（x/y/z 米, N*m，匹配 USE_ECEF_CMTSOLUTION=.true.）"""
        import math

        cfg = self.config.cmt3d
        R_EARTH = cfg['r_earth_m']
        TRIANGLE = cfg['source_decay_mimic_triangle']

        # 地理坐标 → ECEF
        lat_rad = math.radians(evt['latitude'])
        lat_geocentric = math.atan(0.9933056 * math.tan(lat_rad))
        theta = math.pi / 2.0 - lat_geocentric
        phi = math.radians(evt['longitude'])
        r = R_EARTH - evt['depth'] * 1000.0

        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)

        # 力矩张量旋转 (r,θ,φ) → (x,y,z) ECEF，并转换单位 dyne-cm → N*m
        st, ct = math.sin(theta), math.cos(theta)
        sp, cp = math.sin(phi), math.cos(phi)
        scale = 1.0e-7  # dyne-cm → N*m

        Mrr = evt['Mrr'] * scale
        Mtt = evt['Mtt'] * scale
        Mpp = evt['Mpp'] * scale
        Mrt = evt['Mrt'] * scale
        Mrp = evt['Mrp'] * scale
        Mtp = evt['Mtp'] * scale

        Mxx = (st*st*cp*cp*Mrr + ct*ct*cp*cp*Mtt + sp*sp*Mpp
               + 2*st*ct*cp*cp*Mrt - 2*st*sp*cp*Mrp - 2*ct*sp*cp*Mtp)
        Myy = (st*st*sp*sp*Mrr + ct*ct*sp*sp*Mtt + cp*cp*Mpp
               + 2*st*ct*sp*sp*Mrt + 2*st*sp*cp*Mrp + 2*ct*sp*cp*Mtp)
        Mzz = ct*ct*Mrr + st*st*Mtt - 2*st*ct*Mrt
        Mxy = (st*st*sp*cp*Mrr + ct*ct*sp*cp*Mtt - sp*cp*Mpp
               + 2*st*ct*sp*cp*Mrt + st*(cp*cp-sp*sp)*Mrp + ct*(cp*cp-sp*sp)*Mtp)
        Mxz = (st*ct*cp*Mrr - st*ct*cp*Mtt
               + (ct*ct-st*st)*cp*Mrt - ct*sp*Mrp + st*sp*Mtp)
        Myz = (st*ct*sp*Mrr - st*ct*sp*Mtt
               + (ct*ct-st*st)*sp*Mrt + ct*cp*Mrp - st*cp*Mtp)

        # tau = half_duration / SOURCE_DECAY_MIMIC_TRIANGLE
        tau = evt['half_duration'] / TRIANGLE

        return (
            f"{evt['pde_line']}\n"
            f"event name:     {evt['event_name']}\n"
            f"t0(s):         {evt['time_shift']:+.10E}\n"
            f"tau(s):        {tau:+.10E}\n"
            f"x(m):          {x:+.10E}\n"
            f"y(m):          {y:+.10E}\n"
            f"z(m):          {z:+.10E}\n"
            f"Mxx(N*m):      {Mxx:+.10E}\n"
            f"Myy(N*m):      {Myy:+.10E}\n"
            f"Mzz(N*m):      {Mzz:+.10E}\n"
            f"Mxy(N*m):      {Mxy:+.10E}\n"
            f"Mxz(N*m):      {Mxz:+.10E}\n"
            f"Myz(N*m):      {Myz:+.10E}\n"
        )

    def run_cmt3d_selection_workflow(self, catalog_file: Optional[Path] = None,
                                     output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        CMT3D 测试事件选择完整工作流

        Returns:
            {'events': DataFrame, 'cmtsolution_files': List[Path], ...}
        """
        self.logger.info("🎯 CMT3D 测试事件选择工作流")
        self.logger.info("=" * 60)

        # 1. 加载 CMT3D 目录
        cmt3d_all = self.load_cmt3d_catalog(catalog_file)

        # 2. 筛选区域内事件
        cmt3d_filtered = self.filter_cmt3d_for_region(cmt3d_all)

        if len(cmt3d_filtered) == 0:
            self.logger.warning("⚠️ 无符合条件的 CMT3D 事件")
            return {'events': pd.DataFrame(), 'cmtsolution_files': []}

        # 3. 选择测试事件
        selected = self.select_cmt3d_test_events(cmt3d_filtered)

        # 4. 生成 CMTSOLUTION 文件（同时生成两种格式）
        ecef_files = self.export_cmt3d_to_specfem(
            selected,
            output_dir=Path(output_dir or self.output_dir) / 'CMTSOLUTION_CMT3D_ECEF',
            use_ecef=True
        )
        standard_files = self.export_cmt3d_to_specfem(
            selected,
            output_dir=Path(output_dir or self.output_dir) / 'CMTSOLUTION_CMT3D_standard',
            use_ecef=False
        )

        # 5. 保存选择结果
        csv_path = self.output_dir / 'cmt3d_selected_events.csv'
        selected.to_csv(csv_path, index=False)
        self.logger.info(f"📊 事件列表: {csv_path}")

        return {
            'events': selected,
            'cmtsolution_ecef': ecef_files,
            'cmtsolution_standard': standard_files,
            'csv_path': csv_path,
            'total_in_catalog': len(cmt3d_all),
            'total_in_region': len(cmt3d_filtered),
        }

    def run_complete_workflow(self, gcmt_file: Optional[Path] = None,
                            station_file: Optional[Path] = None,
                            generate_specfem_files: bool = True,
                            generate_par_catalog: bool = True) -> Dict[str, Any]:
        """
        运行完整的工作流程
        
        Args:
            gcmt_file: GCMT事件文件路径
            station_file: 永久台站文件路径
            generate_specfem_files: 是否生成SPECFEM文件
            generate_par_catalog: 是否生成PAR目录文件
            
        Returns:
            完整结果字典
        """
        self.logger.info("🚀 开始完整的事件采样工作流...")
        
        try:
            # 1. 事件采样
            selected_events = self.sample_events_complete_workflow(gcmt_file, station_file)
            
            if len(selected_events) == 0:
                self.logger.warning("⚠️ 没有选中的事件")
                return {'selected_events_count': 0}
            
            # 2. 生成SPECFEM文件
            specfem_results = {}
            if generate_specfem_files:
                self.logger.info("📄 生成SPECFEM3D文件...")
                
                # 生成STATIONS文件
                if self.config.output['create_stations_file']:
                    stations_file = self.generate_specfem_stations_file()
                    specfem_results['stations_file'] = str(stations_file)
                
                # 生成CMTSOLUTION文件
                if self.config.output['create_cmtsolution_files']:
                    cmt_files = self.generate_cmtsolution_files(selected_events)
                    specfem_results['cmtsolution_files'] = len(cmt_files)
                    specfem_results['cmtsolution_dir'] = str(cmt_files[0].parent) if cmt_files else ""
            
            # 3. 生成PAR目录文件（新增）
            if generate_par_catalog:
                self.logger.info("📝 生成PAR格式事件目录...")
                par_file = self.generate_par_catalog()
                specfem_results['par_catalog'] = par_file
            
            # 4. 创建可视化
            visualizations = {}
            viz_config = self.config.visualization
            
            if viz_config['create_maps']:
                visualizations['station_event_map'] = self.create_station_event_map()
            
            if viz_config['create_3d_plot']:
                visualizations['3d_distribution'] = self.create_3d_distribution_plot()
            
            # 新增：创建交互式3D图
            if viz_config['create_interactive_3d'] and PLOTLY_AVAILABLE:
                visualizations['interactive_3d'] = self.create_interactive_3d_plot()
            
            if viz_config['create_statistics']:
                visualizations['sampling_statistics'] = self.create_sampling_statistics_plot()
            
            # 5. 保存事件列表
            if self.config.output['save_event_list']:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                event_list_file = self.base_config.dirs['events'] / f"selected_events_{timestamp}.csv"
                selected_events.to_csv(event_list_file, index=False)
                specfem_results['event_list'] = str(event_list_file)
            
            # 6. 生成报告
            report_file = None
            if self.config.output['generate_summary_report']:
                report_file = self.generate_comprehensive_report()
            
            # 7. 汇总结果
            results = {
                'selected_events_count': len(selected_events),
                'permanent_stations_count': len(self.permanent_stations) if self.permanent_stations is not None else 0,
                'sampling_statistics': self.sampling_stats,
                'specfem_files': specfem_results,
                'visualizations': visualizations,
                'report_file': report_file
            }
            
            self.logger.info("✅ 完整工作流程完成")
            self.logger.info(f"📊 最终结果:")
            self.logger.info(f"   永久台站数: {results['permanent_stations_count']}")
            self.logger.info(f"   选择事件数: {results['selected_events_count']}")
            if specfem_results:
                self.logger.info(f"   SPECFEM文件: {len(specfem_results)} 类")
            self.logger.info(f"   可视化图表: {len(visualizations)} 个")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 完整工作流程失败: {e}")
            raise


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='EASTASIA-FWI 事件采样系统')
    parser.add_argument('--mode', choices=['gcmt', 'cmt3d', 'both'], default='cmt3d',
                        help='工作模式: gcmt=传统GCMT采样, cmt3d=CMT3D测试事件, both=两者都运行')
    args = parser.parse_args()

    print("🎯 EASTASIA-FWI 事件采样与 CMT3D 集成系统")
    print("=" * 70)

    try:
        sampler = EventSampler()

        # CMT3D 测试事件选择（正演测试优先）
        if args.mode in ('cmt3d', 'both'):
            print("\n" + "─" * 70)
            print("📌 CMT3D 测试事件选择（3D 波形矫正震源）")
            print("─" * 70)

            result = sampler.run_cmt3d_selection_workflow()

            if len(result['events']) > 0:
                print(f"\n✅ CMT3D 事件选择完成!")
                print(f"   目录总量: {result['total_in_catalog']}")
                print(f"   区域内: {result['total_in_region']}")
                print(f"   选中: {len(result['events'])} 个测试事件")
                print(f"\n📄 CMTSOLUTION 文件:")
                print(f"   ECEF 格式: {result['cmtsolution_ecef'][0].parent}")
                print(f"   标准格式:  {result['cmtsolution_standard'][0].parent}")
                print(f"   事件列表:  {result['csv_path']}")

                print(f"\n📋 选中事件列表:")
                for _, evt in result['events'].iterrows():
                    print(f"   {evt['event_name']}: "
                          f"Mw{evt['pde_magnitude']:.1f}, "
                          f"{evt['latitude']:.2f}°N {evt['longitude']:.2f}°E, "
                          f"depth {evt['depth']:.1f} km")

                print(f"\n⚠️  当前编译使用 USE_ECEF_CMTSOLUTION=.true.:")
                print(f"   → 正演请使用 ECEF 格式 CMTSOLUTION")
                print(f"   → 若改为 .false. 并重新编译，可使用标准格式")
            else:
                print("\n⚠️ 区域内无符合条件的 CMT3D 事件")

        # 传统 GCMT 采样工作流
        if args.mode in ('gcmt', 'both'):
            print("\n" + "─" * 70)
            print("📌 传统 GCMT 事件采样")
            print("─" * 70)
            results = sampler.run_complete_workflow()

            if results['selected_events_count'] > 0:
                print(f"\n✅ GCMT 事件采样完成!")
                print(f"   永久台站数: {results['permanent_stations_count']}")
                print(f"   选择事件数: {results['selected_events_count']}")
            else:
                print("\n⚠️ 未选择到符合条件的 GCMT 事件")

        print("\n" + "=" * 70)
        print(f"📁 输出目录: {sampler.output_dir}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断分析")
    except Exception as e:
        print(f"\n❌ 事件采样失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()