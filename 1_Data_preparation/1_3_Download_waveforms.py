"""
1_3_Download_waveforms.py: 
地震波形数据批量下载模块
基于FDSN服务批量下载东亚地区地震事件的波形数据
配置参数已集成在模块内部
新增功能：支持下载指定永久台站的波形数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
import logging
import time
import json
import shutil
import warnings
import sys

warnings.filterwarnings("ignore")

# ObsPy imports
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import (
    RectangularDomain,
    Restrictions,
    MassDownloader,
)

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig


class DownloadConfig:
    """波形数据下载配置类 - 集成在模块内部"""
    
    def __init__(self):
        """初始化下载配置参数"""
        # FDSN服务配置
        self.fdsn = {
            'providers': ['AUSPASS', 'ETH', 'GEOFON',  
                'GEONET', 'GFZ', 'IPGP', 'IRIS', 
                'RESIF', 'SCEDC', 'USP'],
            'network': "*",
            'timeout': 300,  # 超时时间（秒）
            'channel_priorities': ["BH*", "HH*",],
            'location_priorities': ["", "00", "10", "20"],
            'min_interstation_distance': 1000  # 最小台间距（米）
        }
        
        # 时间窗口配置
        self.time_window = {
            'pre_event_minutes': 0.0,      # 事件前时间（分钟）
            'post_event_minutes': 60.0,    # 事件后时间（分钟）
            'minimum_length_ratio': 0.9    # 最小长度比例
        }
        
        # 事件选择条件
        self.event_selection = {
            'magnitude_range': [6.0, 7.0],         # 震级范围
            'time_range': ['2010-01-01', '2024-12-31'],  # 时间范围
            'max_events': 100,                       # 最大事件数
            'spatial_decimation': True,            # 启用空间去重
            'min_distance_km': 100                 # 最小事件间距（km）
        }
        
        # 永久台站下载配置（新增）
        self.permanent_stations = {
            'use_permanent_only': False,           # 是否只下载永久台站
            'station_file': 'EastAsia_permanent_stations_filtered.csv',  # 永久台站文件
            'network_list': None,                  # 指定网络列表（None表示使用所有）
            'station_list': None                   # 指定台站列表（None表示使用所有）
        }
        
        # 重试机制配置
        self.retry = {
            'max_retries': 3,          # 最大重试轮数（每轮尝试所有Provider）
            'base_delay': 3,           # 基础延迟（秒）
            'timeout_delay': 8,        # 超时后延迟（秒）
            'server_error_delay': 15   # 服务器错误延迟（秒）
        }
        
        # 数据质量控制
        self.quality_control = {
            'min_mseed_files': 450,      # 最少mseed文件数
            'min_xml_files': 100,        # 最少xml文件数
            'min_total_files': 300       # 最少总文件数
        }
        
        # 性能优化
        self.performance = {
            'progress_interval': 1,        # 进度显示间隔
            'auto_save_interval': 5,       # 自动保存间隔
            'enable_progress_bar': True    # 启用进度条
        }
        
        # 输出配置
        self.output = {
            'save_catalog': True,          # 保存下载目录
            'save_progress': True,         # 保存进度文件
            'generate_report': True,       # 生成报告
            'cleanup_on_complete': False   # 完成后清理临时文件
        }
        
        # 日志配置
        self.logging = {
            'level': 'INFO',
            'console_output': True,
            'file_prefix': 'download',
            'obspy_log_level': 'WARNING'   # 减少ObsPy日志噪音
        }
    
    def get_selection_criteria(self) -> Dict[str, Any]:
        """获取事件选择条件"""
        return self.event_selection.copy()
    
    def get_provider_config(self) -> Dict[str, Any]:
        """获取FDSN Provider配置"""
        return {
            'providers': self.fdsn['providers'],
            'timeout': self.fdsn['timeout'],
            'network': self.fdsn['network']
        }
    
    def get_download_restrictions(self) -> Dict[str, Any]:
        """获取下载限制条件"""
        return {
            'channel_priorities': self.fdsn['channel_priorities'],
            'location_priorities': self.fdsn['location_priorities'],
            'minimum_length': self.time_window['minimum_length_ratio'],
            'min_interstation_distance': self.fdsn['min_interstation_distance']
        }


class WaveformDownloader:
    """
    波形数据下载器
    
    功能包括：
    - 从GCMT结果生成下载目录
    - 批量下载波形数据
    - 智能重试机制（每轮尝试所有Provider）
    - 断点续传功能
    - 数据质量控制
    - 详细统计报告
    - 支持只下载永久台站数据（新增）
    
    Attributes:
        base_config: 基础配置
        config: 下载配置
        logger: 日志记录器
        output_paths: 输出路径字典
        download_stats: 下载统计信息
        permanent_stations_df: 永久台站数据（新增）
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        初始化波形下载器
        
        Args:
            config_file: 自定义配置文件路径
        """
        # 加载基础配置
        self.base_config = BaseConfig(config_file)
        
        # 加载模块配置
        self.config = DownloadConfig()
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.WaveformDownloader',
            self.config.logging['level']
        )
        
        # 初始化数据存储
        self._initialize_storage()
        
        # 创建输出目录
        self._setup_output_directories()
        
        # 初始化Provider轮询状态
        self.provider_index = 0
        
        # 永久台站数据（新增）
        self.permanent_stations_df = None
        self.use_permanent_stations = False
        
        self.logger.info("📡 波形数据下载器初始化完成")
        self._print_config_summary()

    def _initialize_storage(self):
        """初始化数据存储容器"""
        self.download_stats = {
            'successful_events': [],
            'failed_events': [],
            'skipped_events': [],
            'no_data_events': [],
            'total_processed': 0,
            'start_time': None,
            'provider_usage': {}
        }
        self.processing_start_time = None

    def _setup_output_directories(self):
        """设置并创建输出目录"""
        self.output_paths = self._get_output_paths()
        
        # 创建必要的目录
        for path_name, path in self.output_paths.items():
            if path_name in ['catalog_file', 'progress_file', 'report_file']:
                # 这些是文件，创建其父目录
                path.parent.mkdir(parents=True, exist_ok=True)
                # 如果存在同名目录，删除它
                if path.exists() and path.is_dir():
                    shutil.rmtree(path)
            else:
                # 这些是目录，直接创建
                path.mkdir(parents=True, exist_ok=True)

    def _get_output_paths(self) -> Dict[str, Path]:
        """获取输出路径"""
        base_dir = self.base_config.dirs['data'] / 'events'
        
        return {
            'waveforms': base_dir / 'waveforms',
            'responses': base_dir / 'responses', 
            'catalog_file': base_dir / 'download_catalog.par',
            'progress_file': base_dir / 'download_progress.json',
            'report_file': base_dir / 'download_report.txt'
        }

    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 波形数据下载配置摘要")
        print("-" * 60)
        
        # 研究区域配置
        print(f"研究区域: {self.base_config.region['name']}")
        region = self.base_config.region
        print(f"纬度范围: {region['lat_min']}° ~ {region['lat_max']}°")
        print(f"经度范围: {region['lon_min']}° ~ {region['lon_max']}°")
        
        # FDSN配置
        providers_list = ', '.join(self.config.fdsn['providers'])
        print(f"FDSN Providers: {len(self.config.fdsn['providers'])} 个")
        print(f"Provider列表: {providers_list}")
        print(f"查询超时: {self.config.fdsn['timeout']} 秒")
        
        # 事件选择配置
        mag_range = self.config.event_selection['magnitude_range']
        print(f"震级范围: {mag_range[0]} ~ {mag_range[1]}")
        print(f"最大事件数: {self.config.event_selection['max_events']}")
        
        # 时间窗口配置
        time_win = self.config.time_window
        print(f"时间窗口: 前{time_win['pre_event_minutes']}分钟 ~ 后{time_win['post_event_minutes']}分钟")
        
        # 输出配置
        print(f"数据输出: {self.output_paths['waveforms']}")
        print(f"响应输出: {self.output_paths['responses']}")
        
        print("-" * 60)

    def _configure_obspy_logging(self):
        """配置ObsPy日志以减少噪音"""
        obspy_loggers = [
            'obspy',
            'obspy.clients.fdsn.mass_downloader'
        ]
        
        for logger_name in obspy_loggers:
            obspy_logger = logging.getLogger(logger_name)
            obspy_logger.handlers.clear()
            obspy_logger.setLevel(logging.ERROR)  # 设为ERROR进一步减少噪音
            obspy_logger.propagate = False

    def load_permanent_stations(self, station_file: Optional[Path] = None) -> pd.DataFrame:
        """
        加载永久台站数据（新增方法）
        
        Args:
            station_file: 台站文件路径
            
        Returns:
            永久台站数据DataFrame
        """
        if station_file is None:
            # 查找默认的永久台站文件
            stations_dir = self.base_config.dirs['stations']
            possible_files = [
                stations_dir / self.config.permanent_stations['station_file'],
                stations_dir / 'EastAsia_permanent_stations_filtered.csv',
                stations_dir / 'EastAsia_permanent_stations.csv',
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    station_file = file_path
                    break
            else:
                raise FileNotFoundError("未找到永久台站文件")
        
        if station_file is None:
            raise FileNotFoundError("未找到永久台站文件")
        
        try:
            if station_file is None:
                raise ValueError("station_file 不能为 None")
            
            self.permanent_stations_df = pd.read_csv(station_file)
            self.use_permanent_stations = True
            
            if self.permanent_stations_df is None or self.permanent_stations_df.empty:
                raise ValueError("加载的永久台站数据为空")
            
            self.logger.info(f"✅ 加载永久台站数据: {len(self.permanent_stations_df)} 个台站")
            self.logger.info(f"   文件路径: {station_file}")
            
            # 统计网络信息
            networks = self.permanent_stations_df['Network'].unique()
            self.logger.info(f"   涉及网络: {len(networks)} 个 ({', '.join(sorted(networks)[:10])}...)")
            
            # 验证必要的列
            required_cols = ['Network', 'Station', 'Latitude', 'Longitude']
            missing_cols = [col for col in required_cols if col not in self.permanent_stations_df.columns]
            if missing_cols:
                raise ValueError(f"永久台站文件缺少必要列: {missing_cols}")
            
            return self.permanent_stations_df
            
        except Exception as e:
            self.logger.error(f"❌ 加载永久台站数据失败: {e}")
            raise

    def load_gcmt_events(self, gcmt_file: Optional[Path] = None) -> pd.DataFrame:
        """
        加载GCMT事件数据
        
        Args:
            gcmt_file: GCMT事件文件路径
            
        Returns:
            事件数据DataFrame
        """
        if gcmt_file is None:
            # 查找最新的GCMT文件
            gcmt_files = list(self.base_config.dirs['catalogs'].glob("gcmt_catalogs_*.csv"))
            if not gcmt_files:
                raise FileNotFoundError("未找到GCMT事件文件，请先运行 1_2_Process_gcmt_catalogs.py")
            gcmt_file = max(gcmt_files, key=lambda x: x.stat().st_mtime)
        
        try:
            df = pd.read_csv(gcmt_file)
            self.logger.info(f"✅ 加载了 {len(df)} 个GCMT事件 (文件: {gcmt_file.name})")
            return df
        except Exception as e:
            self.logger.error(f"❌ 加载GCMT文件失败: {e}")
            raise

    def select_download_events(self, df: pd.DataFrame, 
                             criteria: Optional[Dict] = None) -> pd.DataFrame:
        """
        根据条件筛选用于下载的事件
        
        Args:
            df: GCMT事件DataFrame
            criteria: 筛选条件字典
            
        Returns:
            筛选后的事件DataFrame
        """
        if criteria is None:
            criteria = self.config.get_selection_criteria()
        
        self.logger.info("🔍 开始事件筛选...")
        initial_count = len(df)
        filtered_df = df.copy()
        
        # 统一处理震级数据
        magnitude = []
        for _, row in filtered_df.iterrows():
            mag_mb = row.get('Magnitude1_mb')
            mag_ms = row.get('Magnitude2_Ms')
            
            if pd.notna(mag_mb) and mag_mb > 0:
                magnitude.append(mag_mb)
            elif pd.notna(mag_ms) and mag_ms > 0:
                magnitude.append(mag_ms)
            else:
                magnitude.append(np.nan)
        
        filtered_df['Unified_Magnitude'] = magnitude
        
        # 移除无有效震级的事件
        filtered_df = filtered_df[pd.notna(filtered_df['Unified_Magnitude'])]
        self.logger.info(f"移除无震级事件: {initial_count} → {len(filtered_df)}")
        
        # 震级筛选
        if 'magnitude_range' in criteria:
            mag_min, mag_max = criteria['magnitude_range']
            mag_filter = (
                (filtered_df['Unified_Magnitude'] >= mag_min) & 
                (filtered_df['Unified_Magnitude'] <= mag_max)
            )
            filtered_df = filtered_df[mag_filter]
            self.logger.info(f"震级筛选({mag_min}-{mag_max})后剩余: {len(filtered_df)} 个事件")
        
        # 时间筛选
        if 'time_range' in criteria:
            try:
                start_date, end_date = criteria['time_range']
                filtered_df['DateTime'] = pd.to_datetime(filtered_df['Date'])
                time_filter = (
                    (filtered_df['DateTime'] >= start_date) & 
                    (filtered_df['DateTime'] <= end_date)
                )
                filtered_df = filtered_df[time_filter]
                self.logger.info(f"时间筛选后剩余: {len(filtered_df)} 个事件")
            except Exception as e:
                self.logger.warning(f"时间筛选失败: {e}")
        
        # 限制事件数量
        if 'max_events' in criteria and len(filtered_df) > criteria['max_events']:
            filtered_df = filtered_df.nlargest(criteria['max_events'], 'Unified_Magnitude')
            self.logger.info(f"限制事件数量至: {len(filtered_df)} 个")
        
        # 空间去重（如果配置）
        if criteria.get('spatial_decimation', False):
            filtered_df = self._apply_spatial_decimation(filtered_df, criteria)
        
        return filtered_df.sort_values('Date').reset_index(drop=True)

    def _apply_spatial_decimation(self, df: pd.DataFrame, criteria: Dict) -> pd.DataFrame:
        """应用空间去重，避免事件过于密集"""
        min_distance = criteria.get('min_distance_km', 50)
        
        if len(df) <= 1:
            return df
        
        # 按震级排序，优先保留大震级事件
        df_sorted = df.sort_values('Unified_Magnitude', ascending=False)
        selected_indices = []
        
        for idx, event in df_sorted.iterrows():
            keep_event = True
            
            # 检查与已选事件的距离
            for selected_idx in selected_indices:
                selected_event = df_sorted.loc[selected_idx]
                distance = self._calculate_distance(
                    event['Latitude'], event['Longitude'],
                    selected_event['Latitude'], selected_event['Longitude']
                )
                
                if distance < min_distance:
                    keep_event = False
                    break
            
            if keep_event:
                selected_indices.append(idx)
        
        filtered_df = df_sorted.loc[selected_indices]
        removed_count = len(df) - len(filtered_df)
        
        if removed_count > 0:
            self.logger.info(f"空间去重移除 {removed_count} 个事件")
        
        return filtered_df

    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """计算两点间的球面距离 (km)"""
        R = 6371.0  # 地球半径
        
        lat1_rad, lon1_rad = np.radians([lat1, lon1])
        lat2_rad, lon2_rad = np.radians([lat2, lon2])
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c

    def generate_download_catalog(self, events_df: pd.DataFrame) -> Path:
        """
        生成下载目录文件（.par格式）
        
        Args:
            events_df: 筛选后的事件DataFrame
            
        Returns:
            生成的目录文件路径
        """
        if events_df.empty:
            raise ValueError("没有事件数据生成下载目录")
        
        self.logger.info(f"📝 生成 {len(events_df)} 个事件的下载目录...")
        
        catalog_lines = []
        
        for _, event in events_df.iterrows():
            try:
                # 解析时间
                event_time = pd.to_datetime(event['Date'] + ' ' + str(event['Time']))
                
                # 创建事件名称
                event_name = f"{event_time.strftime('%Y%m%d.%H.%M')}"
                
                # 获取震级
                magnitude = event.get('Unified_Magnitude')
                if pd.isna(magnitude) or magnitude <= 0:
                    self.logger.warning(f"事件 {event_name} 没有有效震级，跳过")
                    continue
                
                # 格式化为 .par 格式, 8.4 25.6 为占位符
                par_line = (
                    f"{event_name} {event_time.strftime('%Y%m%d')} "
                    f"{event_time.strftime('%H')} {event_time.strftime('%M')} "
                    f"{event_time.strftime('%S')}.000 "
                    f"{event['Latitude']:.4f} {event['Longitude']:.4f} "
                    f"{event['Depth']:.1f} 8.4 25.6 {magnitude:.1f} ML"
                )
                
                catalog_lines.append(par_line)
                
            except Exception as e:
                self.logger.warning(f"处理事件失败: {e}")
                continue
        
        if not catalog_lines:
            raise ValueError("没有生成有效的事件目录行")
        
        # 保存目录文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        catalog_file = self.output_paths['catalog_file'].parent / f"download_catalog_{timestamp}.par"
        
        with open(catalog_file, 'w') as f:
            for line in catalog_lines:
                f.write(line + '\n')
        
        self.logger.info(f"✅ 下载目录已保存: {catalog_file}")
        self.logger.info(f"包含 {len(catalog_lines)} 个事件")
        
        return catalog_file

    def load_catalog(self, catalog_file: Path) -> List[str]:
        """
        加载地震目录
        
        Args:
            catalog_file: 目录文件路径
            
        Returns:
            目录行列表
        """
        try:
            with open(catalog_file, 'r') as f:
                # 跳过注释行
                catalog = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            self.logger.info(f"📖 成功加载地震目录，共 {len(catalog)} 个事件")
            return catalog
        except Exception as e:
            self.logger.error(f"❌ 加载目录文件失败: {e}")
            raise

    def parse_event(self, event_line: str) -> Optional[Dict]:
        """
        解析事件行数据
        
        Args:
            event_line: 事件行字符串
            
        Returns:
            事件信息字典，解析失败时返回None
        """
        try:
            event_parts = event_line.split()
            
            if len(event_parts) < 8:
                raise ValueError(f"事件数据格式不正确")
            
            event_info = {
                'dir': event_parts[0],
                'ymd': event_parts[1],
                'hour': event_parts[2],
                'min': event_parts[3],
                'sec': event_parts[4],
                'lat': float(event_parts[5]),
                'lon': float(event_parts[6]),
                'depth': float(event_parts[7])
            }
            
            # 验证数据范围
            if not (-90 <= event_info['lat'] <= 90):
                raise ValueError(f"纬度超出范围: {event_info['lat']}")
            if not (-180 <= event_info['lon'] <= 180):
                raise ValueError(f"经度超出范围: {event_info['lon']}")
            if event_info['depth'] < 0:
                raise ValueError(f"深度不能为负数: {event_info['depth']}")
            
            # 创建时间对象
            time_str = f"{event_info['ymd']} {event_info['hour']}:{event_info['min']}:{event_info['sec']}"
            event_info['origin_time'] = UTCDateTime(time_str)
            
            return event_info
            
        except Exception as e:
            self.logger.debug(f"解析事件数据失败: {event_line}, 错误: {e}")
            return None

    def check_event_exists(self, event_dir: str) -> bool:
        """
        检查事件数据是否已存在且完整
        
        Args:
            event_dir: 事件目录名
            
        Returns:
            事件是否存在且完整
        """
        data_path = self.output_paths['waveforms'] / event_dir
        response_path = self.output_paths['responses'] / event_dir
        
        # 检查目录是否存在且非空
        data_exists = data_path.exists() and any(data_path.iterdir())
        response_exists = response_path.exists() and any(response_path.iterdir())
        
        if not (data_exists and response_exists):
            return False
        
        # 检查文件数量
        mseed_count = len(list(data_path.glob("*.mseed")))
        xml_count = len(list(response_path.glob("*.xml")))
        
        quality_config = self.config.quality_control
        sufficient = (
            mseed_count >= quality_config['min_mseed_files'] and
            xml_count >= quality_config['min_xml_files'] and
            (mseed_count + xml_count) >= quality_config['min_total_files']
        )
        
        return sufficient

    def _get_next_provider(self) -> str:
        """获取下一个Provider"""
        providers = self.config.fdsn['providers']
        current_provider = providers[self.provider_index % len(providers)]
        
        # 确保provider_usage字典存在
        if 'provider_usage' not in self.download_stats:
            self.download_stats['provider_usage'] = {}
        
        # 更新统计
        if current_provider not in self.download_stats['provider_usage']:
            self.download_stats['provider_usage'][current_provider] = 0
        self.download_stats['provider_usage'][current_provider] += 1
        
        # 递增索引，下次将使用不同的Provider
        self.provider_index += 1
        
        return current_provider

    def get_geographic_bounds(self) -> Dict[str, float]:
        """获取地理边界"""
        return {
            'lat_min': self.base_config.region['lat_min'],
            'lat_max': self.base_config.region['lat_max'],
            'lon_min': self.base_config.region['lon_min'], 
            'lon_max': self.base_config.region['lon_max']
        }

    def _build_station_restrictions(self) -> Optional[Dict[str, str]]:
        """
        构建台站限制字典
        
        Returns:
            包含network和station字符串的字典
        """
        if not self.use_permanent_stations or self.permanent_stations_df is None:
            return None
        
        # 按网络分组台站
        networks = set()
        stations = set()
        
        for _, station in self.permanent_stations_df.iterrows():
            network = station['Network']
            station_code = station['Station']
            networks.add(network)
            stations.add(station_code)
        
        # 返回network和station的逗号分隔字符串（ObsPy要求的格式）
        return {
            'network': ','.join(sorted(networks)),
            'station': ','.join(sorted(stations))
        }
        
    def download_event_data(self, event_info: Dict, retry_round: int = 1) -> Union[bool, str]:
        """
        下载单个事件的数据（支持永久台站限制）
        
        Args:
            event_info: 事件信息字典
            retry_round: 当前重试轮数
            
        Returns:
            下载结果：True(成功), False(部分数据), "no_data"(无数据)
        """
        event_dir = event_info['dir']
        origin_time = event_info['origin_time']
        
        try:
            # 获取当前Provider
            current_provider = self.config.fdsn['providers'][self.provider_index % len(self.config.fdsn['providers'])]
            
            mode_str = " [仅永久台站]" if self.use_permanent_stations else ""
            self.logger.debug(f"      🔄 下载事件 {event_dir} from {current_provider}{mode_str}")
            
            # 配置下载路径
            data_path = self.output_paths['waveforms'] / event_dir
            response_path = self.output_paths['responses'] / event_dir
            
            # 定义地理域
            bounds = self.get_geographic_bounds()
            domain = RectangularDomain(
                minlatitude=bounds['lat_min'],
                maxlatitude=bounds['lat_max'],
                minlongitude=bounds['lon_min'],
                maxlongitude=bounds['lon_max']
            )
            
            # 定义限制条件
            time_config = self.config.time_window
            restrictions_params = {
                'starttime': origin_time - time_config['pre_event_minutes'] * 60,
                'endtime': origin_time + time_config['post_event_minutes'] * 60,
                'reject_channels_with_gaps': True,
                'minimum_length': time_config['minimum_length_ratio'],
                'channel_priorities': self.config.fdsn['channel_priorities'],
                'location_priorities': self.config.fdsn['location_priorities']
            }
            
            # 如果使用永久台站，添加台站限制
            if self.use_permanent_stations:
                station_restrictions = self._build_station_restrictions()
                if station_restrictions:
                    restrictions_params['network'] = station_restrictions['network']
                    restrictions_params['station'] = station_restrictions['station']
            else:
                restrictions_params['network'] = self.config.fdsn['network']
                restrictions_params['minimum_interstation_distance_in_m'] = self.config.fdsn['min_interstation_distance']
            
            restrictions = Restrictions(**restrictions_params)
            
            # 执行下载
            download_start = time.time()
            
            mdl = MassDownloader(providers=[current_provider])
            mdl.download(
                domain,
                restrictions,
                mseed_storage=str(data_path),
                stationxml_storage=str(response_path)
            )
            
            download_duration = time.time() - download_start
            
            # 验证下载结果（等待文件写入完成）
            time.sleep(1)
            
            # 检查是否有数据下载
            if data_path.exists() and response_path.exists():
                mseed_files = list(data_path.glob("*.mseed"))
                xml_files = list(response_path.glob("*.xml"))
                mseed_count = len(mseed_files)
                xml_count = len(xml_files)
                
                if mseed_count > 0 or xml_count > 0:
                    # 有数据下载
                    self.logger.debug(f"      📥 下载完成 (mseed={mseed_count}, xml={xml_count}), 耗时: {download_duration:.1f}秒")
                    
                    # 检查是否满足质量要求
                    if self.check_event_exists(event_dir):
                        return True
                    else:
                        return False  # 有数据但不足
                else:
                    # 完全没有数据
                    return "no_data"
            else:
                # 目录都不存在
                return "no_data"
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"      ❌ 下载异常: {error_msg}")
            return False

    def download_with_retry(self, event_info: Dict) -> Tuple[bool, str]:
        """
        带重试机制的下载（修改：每轮尝试所有Provider）
        
        策略：
        1. 每轮尝试所有Provider（每个Provider可能有不同台站的数据）
        2. 如果数据量不足，进行下一轮重试
        3. 最多重试 max_retries 轮
        
        Args:
            event_info: 事件信息字典
            
        Returns:
            (下载是否成功, 结果描述)
        """
        retry_config = self.config.retry
        max_retries = retry_config['max_retries']
        event_dir = event_info['dir']
        all_providers = self.config.fdsn['providers']
        
        self.logger.info(f"🎯 开始下载事件 {event_dir}，将尝试所有 {len(all_providers)} 个数据源")
        
        for retry_round in range(1, max_retries + 1):
            self.logger.info(f"📡 第 {retry_round} 轮下载：尝试所有数据源...")
            round_success_count = 0  # 本轮成功下载的Provider数
            
            # 尝试所有Provider
            for provider_idx, provider in enumerate(all_providers, 1):
                # 设置当前provider
                current_provider_idx = all_providers.index(provider)
                self.provider_index = current_provider_idx
                
                self.logger.info(f"   [{provider_idx}/{len(all_providers)}] 尝试数据源: {provider}")
                
                # 下载
                result = self.download_event_data(event_info, retry_round)
                
                # 记录provider使用
                if provider not in self.download_stats['provider_usage']:
                    self.download_stats['provider_usage'][provider] = 0
                self.download_stats['provider_usage'][provider] += 1
                
                if result is True:
                    # 下载成功
                    round_success_count += 1
                    self.logger.info(f"   ✅ {provider} 下载成功")
                elif result is False:
                    # 有部分数据
                    round_success_count += 1
                    self.logger.info(f"   ⚠️  {provider} 部分数据下载")
                else:
                    # 无数据
                    self.logger.debug(f"   ➖ {provider} 无可用数据")
                
                # 短暂延迟，避免请求过快
                time.sleep(1)
            
            # 检查本轮下载后的总数据量
            data_path = self.output_paths['waveforms'] / event_dir
            response_path = self.output_paths['responses'] / event_dir
            
            if self.check_event_exists(event_dir):
                mseed_count = len(list(data_path.glob("*.mseed")))
                xml_count = len(list(response_path.glob("*.xml")))
                
                self.logger.info(f"🎉 事件 {event_dir} 下载完成！")
                self.logger.info(f"   第 {retry_round} 轮完成，累积数据: mseed={mseed_count}, xml={xml_count}")
                self.logger.info(f"   成功数据源数量: {round_success_count}/{len(all_providers)}")
                return True, "success"
            else:
                # 数据量仍不足
                if data_path.exists() and response_path.exists():
                    mseed_count = len(list(data_path.glob("*.mseed")))
                    xml_count = len(list(response_path.glob("*.xml")))
                    self.logger.warning(f"⚠️  第 {retry_round} 轮完成，数据量不足: mseed={mseed_count}, xml={xml_count}")
                    self.logger.warning(f"   需要: mseed>={self.config.quality_control['min_mseed_files']}, xml>={self.config.quality_control['min_xml_files']}")
                else:
                    self.logger.warning(f"⚠️  第 {retry_round} 轮完成，但无有效数据")
                
                # 如果还有重试机会，等待后继续
                if retry_round < max_retries:
                    delay = retry_config['base_delay']
                    self.logger.info(f"   等待 {delay} 秒后开始第 {retry_round + 1} 轮重试...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"❌ 事件 {event_dir} 下载失败：已完成 {max_retries} 轮尝试，数据量仍不足")
                    return False, "insufficient_data"
        
        return False, "failed"

    def save_progress(self, current_index: int):
        """
        保存下载进度
        
        Args:
            current_index: 当前处理的事件索引
        """
        progress_data = {
            'current_index': current_index,
            'download_stats': self.download_stats,
            'provider_index': self.provider_index,
            'use_permanent_stations': self.use_permanent_stations,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            progress_file = self.output_paths['progress_file']
            
            # 确保进度文件不是目录
            if progress_file.exists() and progress_file.is_dir():
                shutil.rmtree(progress_file)
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"⚠️  保存进度失败: {e}")

    def load_progress(self) -> Optional[Dict]:
        """
        加载之前的下载进度
        
        Returns:
            进度数据字典，如果不存在则返回None
        """
        progress_file = self.output_paths['progress_file']
        
        # 如果是目录，删除它
        if progress_file.exists() and progress_file.is_dir():
            shutil.rmtree(progress_file)
            return None
        
        if not progress_file.exists():
            return None
        
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # 恢复Provider索引状态
            if 'provider_index' in progress_data:
                self.provider_index = progress_data['provider_index']
            
            # 恢复下载统计
            if 'download_stats' in progress_data:
                saved_stats = progress_data['download_stats']
                # 合并保存的统计和默认统计
                for key in self.download_stats:
                    if key in saved_stats:
                        self.download_stats[key] = saved_stats[key]
                # 确保provider_usage字典存在
                if 'provider_usage' not in self.download_stats:
                    self.download_stats['provider_usage'] = {}
            
            # 恢复永久台站使用状态
            if 'use_permanent_stations' in progress_data:
                self.use_permanent_stations = progress_data['use_permanent_stations']
                
            self.logger.info(f"📋 发现之前的下载进度，从事件 {progress_data['current_index']} 继续")
            return progress_data
        except Exception as e:
            self.logger.warning(f"⚠️  加载进度文件失败: {e}")
            return None

    def list_existing_catalogs(self) -> List[Path]:
        """
        列出现有的下载目录文件
        
        Returns:
            现有目录文件路径列表
        """
        catalog_dir = self.output_paths['catalog_file'].parent
        # 搜索所有.par文件，包括selected events
        catalog_files = list(catalog_dir.glob("*.par"))
        
        # 按修改时间排序，最新的在前
        catalog_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return catalog_files

    def select_existing_catalog(self) -> Optional[Path]:
        """
        交互式选择现有的下载目录文件（支持selected events）
        
        Returns:
            选择的目录文件路径，如果没有选择则返回None
        """
        existing_catalogs = self.list_existing_catalogs()
        
        if not existing_catalogs:
            print("📭 没有找到现有的下载目录文件")
            return None
        
        print(f"\n📋 发现 {len(existing_catalogs)} 个目录文件:")
        print("="*80)
        
        for i, catalog_file in enumerate(existing_catalogs, 1):
            # 获取文件信息
            stat = catalog_file.stat()
            modified_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            # 读取目录文件获取事件数量
            try:
                with open(catalog_file, 'r') as f:
                    lines = f.readlines()
                    event_count = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            except Exception:
                event_count = "未知"
            
            # 标识是否为selected events文件
            is_selected = "selected_events" in catalog_file.name
            file_type = " [SELECTED EVENTS] ⭐" if is_selected else ""
            
            print(f"{i:2d}. {catalog_file.name}{file_type}")
            print(f"    📅 修改时间: {modified_time}")
            print(f"    📊 事件数量: {event_count}")
            print(f"    📁 文件大小: {stat.st_size} 字节")
            print()
        
        while True:
            try:
                choice = input(f"请选择目录文件 (1-{len(existing_catalogs)}, 0=不选择): ").strip()
                
                if choice == '0':
                    return None
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(existing_catalogs):
                    selected_file = existing_catalogs[choice_idx]
                    
                    # 如果选择的是selected events文件，询问是否使用永久台站
                    if "selected_events" in selected_file.name:
                        print(f"✅ 已选择: {selected_file.name} [SELECTED EVENTS]")
                        use_perm = input("\n是否仅下载永久台站数据? (Y/n): ").strip().lower()
                        if use_perm != 'n':
                            try:
                                self.load_permanent_stations()
                                if self.permanent_stations_df is not None:
                                    print(f"✅ 已加载 {len(self.permanent_stations_df)} 个永久台站")
                                else:
                                    print("⚠️  永久台站数据加载失败")
                            except Exception as e:
                                print(f"⚠️  加载永久台站失败: {e}")
                                print("   将下载所有台站数据")
                    else:
                        print(f"✅ 已选择: {selected_file.name}")
                    
                    return selected_file
                else:
                    print(f"❌ 请输入 1-{len(existing_catalogs)} 之间的数字")
                    
            except ValueError:
                print("❌ 请输入有效的数字")
            except KeyboardInterrupt:
                print("\n🛑 操作已取消")
                return None

    def preview_catalog_content(self, catalog_file: Path, max_lines: int = 10) -> None:
        """
        预览目录文件内容
        
        Args:
            catalog_file: 目录文件路径
            max_lines: 最大显示行数
        """
        try:
            with open(catalog_file, 'r') as f:
                lines = f.readlines()
            
            # 过滤注释行
            event_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            comment_lines = [line for line in lines if line.strip().startswith('#')]
            
            print(f"\n📖 目录文件预览: {catalog_file.name}")
            print("="*80)
            
            # 显示注释信息
            if comment_lines:
                print("文件头信息:")
                for line in comment_lines[:5]:
                    print(f"  {line.strip()}")
                print()
            
            print(f"总事件数: {len(event_lines)}")
            print(f"显示前 {min(max_lines, len(event_lines))} 个事件:")
            print("-"*80)
            
            for i, line in enumerate(event_lines[:max_lines], 1):
                # 解析事件信息用于显示
                parts = line.strip().split()
                if len(parts) >= 8:
                    event_name = parts[0]
                    lat, lon, depth = parts[5], parts[6], parts[7]
                    print(f"{i:2d}. {event_name} | Lat: {lat}° Lon: {lon}° Depth: {depth}km")
                else:
                    print(f"{i:2d}. {line.strip()}")
            
            if len(event_lines) > max_lines:
                print(f"... 还有 {len(event_lines) - max_lines} 个事件")
                
        except Exception as e:
            print(f"❌ 预览文件失败: {e}")

    def validate_catalog_file(self, catalog_file: Path) -> bool:
        """
        验证目录文件格式是否正确
        
        Args:
            catalog_file: 目录文件路径
            
        Returns:
            文件格式是否有效
        """
        try:
            with open(catalog_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                self.logger.error(f"目录文件为空: {catalog_file}")
                return False
            
            valid_lines = 0
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # 验证事件行格式
                event_info = self.parse_event(line)
                if event_info is None:
                    self.logger.warning(f"第 {line_num} 行格式不正确: {line}")
                else:
                    valid_lines += 1
            
            if valid_lines == 0:
                self.logger.error(f"目录文件中没有有效的事件行: {catalog_file}")
                return False
            
            # 计算有效行比例（排除注释）
            non_comment_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            success_rate = valid_lines / len(non_comment_lines) if non_comment_lines else 0
            if success_rate < 0.8:  # 至少80%的行有效
                self.logger.warning(f"目录文件有效行比例较低: {success_rate:.1%}")
                return False
            
            self.logger.info(f"✅ 目录文件验证通过: {valid_lines} 个有效事件")
            return True
            
        except Exception as e:
            self.logger.error(f"验证目录文件失败: {e}")
            return False

    def batch_download(self, catalog_file: Path) -> Dict:
        """
        批量下载多个事件的数据
        
        Args:
            catalog_file: 目录文件路径
            
        Returns:
            下载统计字典
        """
        self.processing_start_time = time.time()
        self.download_stats['start_time'] = self.processing_start_time
        
        # 配置ObsPy日志
        self._configure_obspy_logging()
        
        # 加载目录
        catalog = self.load_catalog(catalog_file)
        
        # 检查之前的进度
        progress_data = self.load_progress()
        if progress_data:
            start_idx = progress_data.get('current_index', 0)
        else:
            start_idx = 0
        
        total_events = len(catalog)
        mode_info = " [仅永久台站]" if self.use_permanent_stations else ""
        self.logger.info(f"📊 开始批量下载{mode_info}: {total_events - start_idx} 个待处理事件 (总计 {total_events})")
        self.logger.info(f"🌐 可用数据源: {', '.join(self.config.fdsn['providers'])}")
        
        if self.use_permanent_stations:
            if self.permanent_stations_df is not None:
                self.logger.info(f"🎯 永久台站数量: {len(self.permanent_stations_df)}")
            else:
                self.logger.warning("⚠️  永久台站模式已启用，但数据未加载")
        
        try:
            for i in range(start_idx, total_events):
                event_line = catalog[i]
                event_info = self.parse_event(event_line)
                
                if event_info is None:
                    self.logger.warning(f"⚠️  跳过无效事件 {i}")
                    continue
                
                event_dir = event_info['dir']
                
                # 检查是否已存在
                if self.check_event_exists(event_dir):
                    self.download_stats['skipped_events'].append(event_dir)
                    self.logger.debug(f"📁 事件 {event_dir} 已存在，跳过")
                    continue
                
                print(f"\n{'='*80}")
                print(f"🎯 处理事件 {i+1}/{total_events}: {event_dir}{mode_info}")
                print(f"{'='*80}")
                
                # 下载数据
                success, reason = self.download_with_retry(event_info)
                
                if success:
                    self.download_stats['successful_events'].append(event_dir)
                else:
                    if reason == "no_data":
                        self.download_stats['no_data_events'].append(event_dir)
                    else:
                        self.download_stats['failed_events'].append(event_dir)
                
                # 定期保存进度
                if i % self.config.performance['auto_save_interval'] == 0:
                    self.save_progress(i)
                
                # 显示进度
                if (i - start_idx + 1) % self.config.performance['progress_interval'] == 0:
                    self._print_progress(i - start_idx + 1, total_events - start_idx)
            
            # 最终统计
            self.download_stats['total_processed'] = total_events
            total_time = (time.time() - self.processing_start_time) / 60
            
            # 生成报告
            self._generate_download_report(total_time)
            
            # 清理进度文件
            progress_file = self.output_paths['progress_file']
            if progress_file.exists() and progress_file.is_file():
                progress_file.unlink()
            
            return self._get_summary_stats()
            
        except KeyboardInterrupt:
            self.logger.info("\n🛑 用户中断下载")
            self.save_progress(i if 'i' in locals() else start_idx)
            raise
        except Exception as e:
            self.logger.error(f"💥 下载过程出错: {e}")
            if 'i' in locals():
                self.save_progress(i)
            raise

    def _print_progress(self, current: int, total: int):
        """打印进度信息"""
        if self.processing_start_time is None:
            return
        
        progress_pct = (current / total) * 100
        elapsed_time = (time.time() - self.processing_start_time) / 60
        
        successful = len(self.download_stats['successful_events'])
        failed = len(self.download_stats['failed_events'])
        skipped = len(self.download_stats['skipped_events'])
        
        print(f"\n{'='*60}")
        print(f"📊 下载进度: {current}/{total} ({progress_pct:.1f}%)")
        print(f"成功: {successful} | 失败: {failed} | 跳过: {skipped}")
        print(f"已用时间: {elapsed_time:.1f} 分钟")
        
        # 显示Provider使用统计
        if self.download_stats.get('provider_usage'):
            print("🌐 数据源使用情况:")
            for provider, count in self.download_stats['provider_usage'].items():
                print(f"  {provider}: {count} 次")
        
        print(f"{'='*60}")

    def _generate_download_report(self, total_time: float):
        """生成下载报告"""
        try:
            report_file = self.output_paths['report_file']
            
            successful = len(self.download_stats['successful_events'])
            failed = len(self.download_stats['failed_events'])
            skipped = len(self.download_stats['skipped_events'])
            no_data = len(self.download_stats['no_data_events'])
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("EASTASIA-FWI 波形数据下载报告\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"下载时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总耗时: {total_time:.2f} 分钟\n")
                f.write(f"研究区域: {self.base_config.region['name']}\n")
                
                # 永久台站模式说明
                if self.use_permanent_stations and self.permanent_stations_df is not None:
                    f.write(f"下载模式: 仅永久台站数据\n")
                    f.write(f"永久台站数量: {len(self.permanent_stations_df)}\n")
                else:
                    f.write(f"下载模式: 所有台站数据\n")
                f.write("\n")
                
                f.write("📊 下载统计\n")
                f.write("-"*50 + "\n")
                f.write(f"成功下载: {successful} 个事件\n")
                f.write(f"下载失败: {failed} 个事件\n")
                f.write(f"跳过已存在: {skipped} 个事件\n")
                f.write(f"无可用数据: {no_data} 个事件\n")
                
                if successful + failed > 0:
                    success_rate = successful / (successful + failed) * 100
                    f.write(f"成功率: {success_rate:.1f}%\n")
                
                # Provider使用统计
                if self.download_stats.get('provider_usage'):
                    f.write(f"\n🌐 数据源使用统计\n")
                    f.write("-"*50 + "\n")
                    for provider, count in self.download_stats['provider_usage'].items():
                        f.write(f"{provider}: {count} 次尝试\n")
                
                f.write(f"\n📁 数据存储位置\n")
                f.write("-"*50 + "\n")
                f.write(f"波形数据: {self.output_paths['waveforms']}\n")
                f.write(f"响应文件: {self.output_paths['responses']}\n")
                
                # 列出部分成功/失败事件
                if successful > 0:
                    f.write(f"\n✅ 成功下载的事件 (前10个):\n")
                    for event in self.download_stats['successful_events'][:10]:
                        f.write(f"  {event}\n")
                
                if failed > 0:
                    f.write(f"\n❌ 下载失败的事件 (前10个):\n")
                    for event in self.download_stats['failed_events'][:10]:
                        f.write(f"  {event}\n")
            
            self.logger.info(f"✅ 下载报告已保存: {report_file}")
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")

    def _get_summary_stats(self) -> Dict:
        """获取汇总统计信息"""
        return {
            'total_events': self.download_stats.get('total_processed', 0),
            'successful': len(self.download_stats['successful_events']),
            'failed': len(self.download_stats['failed_events']),
            'skipped': len(self.download_stats['skipped_events']),
            'no_data': len(self.download_stats['no_data_events']),
            'provider_usage': self.download_stats.get('provider_usage', {}),
            'use_permanent_stations': self.use_permanent_stations,
            'permanent_stations_count': len(self.permanent_stations_df) if self.permanent_stations_df is not None else 0,
            'waveforms_dir': str(self.output_paths['waveforms']),
            'responses_dir': str(self.output_paths['responses']),
            'report_file': str(self.output_paths['report_file'])
        }

    def select_catalog_source(self) -> Optional[Path]:
        """
        选择下载目录来源
        
        Returns:
            选择的目录文件路径，如果取消则返回None
        """
        print("\n🔧 选择下载目录来源:")
        print("1. 从GCMT数据生成新的下载目录")
        print("2. 使用现有的下载目录文件 (.par)")
        print("3. 指定目录文件路径")
        
        while True:
            try:
                choice = input("\n请选择 (1/2/3): ").strip()
                if choice in ['1', '2', '3']:
                    break
                print("❌ 请输入 1、2 或 3")
            except KeyboardInterrupt:
                print("\n🛑 操作已取消")
                return None
        
        if choice == '1':
            # 从GCMT生成新目录
            return self._generate_new_catalog()
        elif choice == '2':
            # 选择现有目录文件
            return self._select_existing_catalog()
        elif choice == '3':
            # 指定目录文件路径
            return self._specify_catalog_path()
    
    def _generate_new_catalog(self) -> Optional[Path]:
        """从GCMT数据生成新的下载目录"""
        # 加载GCMT事件数据
        gcmt_df = self.load_gcmt_events()
        print(f"📊 加载了 {len(gcmt_df)} 个GCMT事件")
        
        # 筛选下载事件
        selection_criteria = {
            'magnitude_range': [6.0, 7.0],  # 6.0-7.0震级范围
            'time_range': ['2015-01-01', '2023-12-31'],
            'max_events': 10,  # 减少测试数量
            'spatial_decimation': True,
            'min_distance_km': 50  # 最小间距50km
        }
        
        selected_events = self.select_download_events(gcmt_df, selection_criteria)
        
        if selected_events.empty:
            print("❌ 没有符合条件的事件")
            return None
        
        print(f"✅ 筛选出 {len(selected_events)} 个事件用于下载")
        
        # 显示事件信息摘要
        print(f"\n📋 事件摘要:")
        print(f"  震级范围: {selected_events['Unified_Magnitude'].min():.1f} - {selected_events['Unified_Magnitude'].max():.1f}")
        print(f"  时间范围: {selected_events['Date'].min()} ~ {selected_events['Date'].max()}")
        print(f"  空间范围: {selected_events['Latitude'].min():.1f}°-{selected_events['Latitude'].max():.1f}°N, "
              f"{selected_events['Longitude'].min():.1f}°-{selected_events['Longitude'].max():.1f}°E")
        
        # 生成下载目录
        return self.generate_download_catalog(selected_events)
    
    def _select_existing_catalog(self) -> Optional[Path]:
        """选择现有的下载目录文件"""
        catalog_file = self.select_existing_catalog()
        
        if catalog_file is None:
            print("❌ 没有选择目录文件")
            return None
        
        # 预览目录内容
        preview = input("\n是否预览目录文件内容? (y/N): ").strip().lower()
        if preview == 'y':
            self.preview_catalog_content(catalog_file)
        
        # 验证目录文件
        if not self.validate_catalog_file(catalog_file):
            print("❌ 目录文件验证失败")
            return None
        
        return catalog_file
    
    def _specify_catalog_path(self) -> Optional[Path]:
        """指定目录文件路径"""
        while True:
            try:
                file_path = input("\n请输入目录文件路径 (.par文件): ").strip()
                
                if not file_path:
                    print("❌ 路径不能为空")
                    continue
                
                catalog_file = Path(file_path).expanduser().resolve()
                
                if not catalog_file.exists():
                    print(f"❌ 文件不存在: {catalog_file}")
                    continue
                
                if not catalog_file.suffix.lower() == '.par':
                    print(f"❌ 文件不是 .par 格式: {catalog_file.suffix}")
                    continue
                
                print(f"✅ 找到文件: {catalog_file}")
                
                # 如果是selected events文件，询问是否使用永久台站
                if "selected_events" in catalog_file.name:
                    use_perm = input("\n是否仅下载永久台站数据? (Y/n): ").strip().lower()
                    if use_perm != 'n':
                        try:
                            self.load_permanent_stations()
                            if self.permanent_stations_df is not None:
                                print(f"✅ 已加载 {len(self.permanent_stations_df)} 个永久台站")
                            else:
                                print("⚠️  永久台站数据加载失败")
                        except Exception as e:
                            print(f"⚠️  加载永久台站失败: {e}")
                            print("   将下载所有台站数据")
                
                # 预览目录内容
                preview = input("\n是否预览目录文件内容? (y/N): ").strip().lower()
                if preview == 'y':
                    self.preview_catalog_content(catalog_file)
                
                # 验证目录文件
                if not self.validate_catalog_file(catalog_file):
                    print("❌ 目录文件验证失败")
                    retry = input("是否重新选择文件? (y/N): ").strip().lower()
                    if retry != 'y':
                        return None
                    continue
                
                return catalog_file
                
            except KeyboardInterrupt:
                print("\n🛑 操作已取消")
                return None
            except Exception as e:
                print(f"❌ 处理路径失败: {e}")
                retry = input("是否重新输入? (y/N): ").strip().lower()
                if retry != 'y':
                    return None


def main():
    """主函数 - 执行波形数据下载"""
    print("📡 EASTASIA-FWI 波形数据下载系统")
    print("   新增功能: 支持下载selected events的永久台站数据")
    print("="*60)
    
    try:
        # 初始化下载器
        downloader = WaveformDownloader()
        
        # 选择下载目录来源
        catalog_file = downloader.select_catalog_source()
        
        if catalog_file is None:
            print("❌ 没有可用的下载目录")
            return
        
        # 确认下载
        mode_info = " [仅永久台站]" if downloader.use_permanent_stations else ""
        response = input(f"\n是否开始下载波形数据{mode_info}? (y/N): ").strip().lower()
        if response != 'y':
            print("下载已取消")
            return
        
        # 执行批量下载
        print(f"\n🚀 开始下载波形数据{mode_info}...")
        print(f"📋 使用目录文件: {catalog_file.name}")
        results = downloader.batch_download(catalog_file)
        
        # 显示结果
        print(f"\n🎉 下载完成!")
        print(f"📊 统计结果:")
        print(f"  成功下载: {results['successful']} 个事件")
        print(f"  下载失败: {results['failed']} 个事件") 
        print(f"  跳过已存在: {results['skipped']} 个事件")
        print(f"  无可用数据: {results['no_data']} 个事件")
        
        if results['successful'] + results['failed'] > 0:
            success_rate = results['successful'] / (results['successful'] + results['failed']) * 100
            print(f"  成功率: {success_rate:.1f}%")
        
        # 显示永久台站模式信息
        if results['use_permanent_stations']:
            print(f"\n🎯 永久台站模式:")
            print(f"  永久台站数量: {results['permanent_stations_count']}")
        
        # 显示Provider使用统计
        if results['provider_usage']:
            print(f"\n🌐 数据源使用统计:")
            for provider, count in results['provider_usage'].items():
                print(f"  {provider}: {count} 次")
        
        print(f"\n📁 数据保存位置:")
        print(f"  波形数据: {results['waveforms_dir']}")
        print(f"  响应文件: {results['responses_dir']}")
        print(f"  详细报告: {results['report_file']}")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
    except Exception as e:
        print(f"❌ 程序执行错误: {e}")
        logging.exception("详细错误信息:")


if __name__ == "__main__":
    main()