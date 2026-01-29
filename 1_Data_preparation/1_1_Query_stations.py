"""
1_1_Query_stations.py: 
台站查询与筛选模块
基于FDSN查询东亚地区地震台站信息
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from obspy.clients.fdsn import Client
from obspy.core.inventory import Inventory
from obspy import UTCDateTime
import logging
from tqdm import tqdm
import warnings
from datetime import datetime
import time
import sys
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig


class StationQueryConfig:
    """台站查询配置参数"""
    
    def __init__(self):
        # FDSN服务配置
        self.fdsn = {
            'providers': ['AUSPASS', 'ETH', 'GEOFON',  
                'GEONET', 'GFZ', 'IPGP', 'IRIS',
                'RESIF', 'SCEDC', 'USP'],
            'timeout': 600,
            'max_retries': 3,
            'retry_delay': 5,
            'use_progress_bar': True
        }
        
        # 查询时间范围
        self.time_range = {
            'start_time': '1980-01-01',
            'end_time': '2026-01-01',
            'auto_update_end': False
        }
        
        # 区域筛选配置（相对于基础配置的buffer）
        self.query_bounds = {
            'buffer_degrees': 0.0
        }
        
        # 台站质量筛选配置
        self.quality_filters = {
            'preferred_networks': [],
            'excluded_networks': [],
            'min_elevation': -2000,
            'max_elevation': 9000,
            'require_coordinates': True
        }
        
        # 输出配置
        self.output = {
            'save_individual_providers': False,
            'save_combined_data': True,
            'formats': ['csv', 'xml'],
            'generate_report': True,
            'create_specfem_format': True
        }
        
        # 日志配置
        self.logging = {
            'level': 'INFO',
            'console_output': True
        }
        
        # 错误处理配置
        self.error_handling = {
            'max_failures_per_provider': 3,
            'continue_on_provider_failure': True,
            'retry_failed_providers': False,
            'save_error_log': True
        }


class StationQuery:
    """台站查询和管理类
    
    基于FDSN的台站查询器，提供东亚地区地震台站信息查询、筛选和存储功能。
    
    主要功能：
    - 多Provider台站查询
    - 基础质量筛选
    - 标准化数据输出
    - 详细的统计报告
    """
    
    def __init__(self):
        """初始化台站查询器"""
        # 加载基础配置
        self.base_config = BaseConfig()
        
        # 加载模块配置
        self.config = StationQueryConfig()
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.StationQuery',
            self.config.logging['level']
        )
        
        # 初始化数据存储
        self._initialize_storage()
        
        # 设置输出路径
        self.output_base = self.base_config.dirs['stations']
        self._setup_output_paths()
        
        self.logger.info("🚀 台站查询器初始化完成")
        self._print_config_summary()

    def _initialize_storage(self):
        """初始化数据存储容器"""
        self.all_stations: List[Dict] = []
        self.total_inventory = Inventory(networks=[], source="EASTASIA-FWI")
        self.failed_providers: List[str] = []
        self.provider_stats: Dict[str, Dict] = {}
        self.query_start_time = None

    def _setup_output_paths(self):
        """设置输出路径"""
        self.output_paths = {
            'base': self.output_base,
            'combined_csv': self.output_base / f"{self.base_config.region['name']}_stations.csv",
            'combined_xml': self.output_base / f"{self.base_config.region['name']}_stations.xml",
            'specfem_stations': self.output_base / "STATIONS",
            'report': self.output_base / "station_query_report.txt",
            'providers': self.output_base / 'providers',
            'metadata': self.output_base / "query_metadata.json"
        }
        
        # 确保目录存在
        for path in self.output_paths.values():
            if hasattr(path, 'parent'):
                path.parent.mkdir(parents=True, exist_ok=True)

    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 台站查询配置摘要")
        print("-" * 60)
        
        # FDSN Provider配置
        providers_list = ', '.join(self.config.fdsn['providers'])
        print(f"FDSN Providers: {len(self.config.fdsn['providers'])} 个")
        print(f"Provider列表: {providers_list}")
        print(f"查询超时: {self.config.fdsn['timeout']} 秒")
        
        # 研究区域配置
        print(f"查询区域: {self.base_config.region['name']}")
        region = self.base_config.region
        print(f"纬度范围: {region['lat_min']}° ~ {region['lat_max']}°")
        print(f"经度范围: {region['lon_min']}° ~ {region['lon_max']}°")
        print(f"缓冲区: {self.config.query_bounds.get('buffer_degrees', 0.0)}°")
        
        # 时间范围配置
        time_config = self.config.time_range
        print(f"时间范围: {time_config['start_time']} ~ {time_config['end_time']}")
        
        # 输出配置
        print(f"输出目录: {self.output_base}")
        
        print("-" * 60)

    def query_single_provider(self, provider: str, 
                            start_time: Optional[str] = None,
                            end_time: Optional[str] = None,
                            retry_count: int = 0) -> Tuple[List[Dict], Optional[Inventory]]:
        """
        查询单个FDSN Provider的台站信息
        
        Args:
            provider: FDSN Provider名称
            start_time: 开始时间，默认使用配置
            end_time: 结束时间，默认使用配置
            retry_count: 重试计数
            
        Returns:
            Tuple[台站信息列表, inventory对象]
        """
        # 获取时间范围
        start_time = start_time or self.config.time_range['start_time']
        end_time = end_time or self.config.time_range['end_time']
        
        # 自动更新结束时间
        if self.config.time_range['auto_update_end']:
            end_time = datetime.now().strftime('%Y-%m-%d')
        
        try:
            self.logger.info(f"📡 正在查询 {provider} (尝试 {retry_count + 1}/{self.config.fdsn['max_retries']})")
            
            # 创建客户端连接
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                client = Client(provider, timeout=self.config.fdsn['timeout'])
                
                # 获取查询边界参数
                buffer = self.config.query_bounds.get('buffer_degrees', 0.0)
                bounds = {
                    'minlatitude': self.base_config.region['lat_min'] - buffer,
                    'maxlatitude': self.base_config.region['lat_max'] + buffer,
                    'minlongitude': self.base_config.region['lon_min'] - buffer,
                    'maxlongitude': self.base_config.region['lon_max'] + buffer
                }
                
                # 执行台站查询
                inventory = client.get_stations(
                    starttime=UTCDateTime(start_time),
                    endtime=UTCDateTime(end_time),
                    level="station",
                    **bounds
                )
                
                # 处理查询结果
                if inventory and len(inventory) > 0:
                    stations_list = self._parse_inventory(inventory, provider)
                    
                    # 记录成功统计
                    self.provider_stats[provider] = {
                        'station_count': len(stations_list),
                        'network_count': len(inventory),
                        'success': True,
                        'retry_count': retry_count,
                        'error': None
                    }
                    
                    self.logger.info(f"✅ {provider}: {len(stations_list)} 台站, {len(inventory)} 台网")
                    return stations_list, inventory
                else:
                    # 无数据返回
                    self.logger.warning(f"⚠️  {provider}: 查询成功但无数据返回")
                    self._record_failed_provider(provider, "No data returned", retry_count)
                    
        except Exception as e:
            error_msg = str(e)
            
            # 判断是否需要重试
            if retry_count < self.config.fdsn['max_retries'] - 1:
                self.logger.warning(f"⚠️  {provider} 查询失败，准备重试: {error_msg}")
                time.sleep(self.config.fdsn.get('retry_delay', 5))
                return self.query_single_provider(provider, start_time, end_time, retry_count + 1)
            else:
                self.logger.error(f"❌ {provider} 查询最终失败: {error_msg}")
                self.failed_providers.append(provider)
                self._record_failed_provider(provider, error_msg, retry_count)
            
        return [], None

    def _record_failed_provider(self, provider: str, error_msg: str, retry_count: int):
        """记录失败的Provider统计信息"""
        self.provider_stats[provider] = {
            'station_count': 0,
            'network_count': 0,
            'success': False,
            'retry_count': retry_count,
            'error': error_msg
        }

    def _parse_inventory(self, inventory: Inventory, provider: str) -> List[Dict]:
        """
        解析inventory对象为标准化字典列表
        
        Args:
            inventory: ObsPy Inventory对象
            provider: Provider名称
            
        Returns:
            标准化的台站信息字典列表
        """
        stations_list = []
        
        try:
            for network in inventory:
                for station in network:
                    # 提取基础台站信息
                    station_info = {
                        "Provider": provider,
                        "Network": network.code,
                        "Station": station.code,
                        "Latitude": float(station.latitude) if station.latitude is not None else 0.0,
                        "Longitude": float(station.longitude) if station.longitude is not None else 0.0,
                        "Elevation": float(station.elevation) if station.elevation is not None else 0.0,
                        "Site": station.site.name if station.site and station.site.name else "",
                        "StartDate": station.start_date.strftime('%Y-%m-%d') if station.start_date else "",
                        "EndDate": station.end_date.strftime('%Y-%m-%d') if station.end_date else "",
                        "StationID": f"{network.code}.{station.code}",
                        "CreationDate": network.start_date.strftime('%Y-%m-%d') if network.start_date else "",
                        "NetworkDescription": network.description or ""
                    }
                    
                    # 数据质量检查
                    if self._validate_station_data(station_info):
                        stations_list.append(station_info)
                        
        except Exception as e:
            self.logger.error(f"解析inventory时出错: {e}")
            
        return stations_list

    def _validate_station_data(self, station_info: Dict) -> bool:
        """
        验证台站数据的基本有效性
        
        Args:
            station_info: 台站信息字典
            
        Returns:
            True if valid, False otherwise
        """
        # 检查必需字段
        required_fields = ['Network', 'Station', 'Latitude', 'Longitude']
        for field in required_fields:
            if field not in station_info or station_info[field] is None:
                return False
                
        # 检查坐标范围
        lat, lon = station_info['Latitude'], station_info['Longitude']
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return False
            
        # 检查是否在研究区域内（考虑缓冲区）
        buffer = self.config.query_bounds.get('buffer_degrees', 0.0)
        if not (self.base_config.region['lat_min'] - buffer <= lat <= self.base_config.region['lat_max'] + buffer and
                self.base_config.region['lon_min'] - buffer <= lon <= self.base_config.region['lon_max'] + buffer):
            return False
            
        return True

    def query_all_providers(self, 
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None,
                          save_individual: Optional[bool] = None) -> pd.DataFrame:
        """
        查询所有配置的FDSN Provider
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            save_individual: 是否保存单个Provider数据
            
        Returns:
            合并后的台站数据DataFrame
        """
        self.query_start_time = time.time()
        providers = self.config.fdsn['providers']
        
        self.logger.info(f"🌐 开始查询 {len(providers)} 个FDSN Provider")
        self.logger.info(f"📍 查询区域: {self.base_config.region['name']}")
        
        if save_individual is None:
            save_individual = self.config.output['save_individual_providers']
        
        # 串行查询所有Provider
        if self.config.fdsn.get('use_progress_bar', True):
            providers_iter = tqdm(providers, desc="查询台站")
        else:
            providers_iter = providers
        
        for provider in providers_iter:
            try:
                stations_list, inventory = self.query_single_provider(provider, start_time, end_time)
                
                if stations_list:
                    self.all_stations.extend(stations_list)
                    
                    if inventory:
                        self.total_inventory += inventory
                    
                    if save_individual:
                        self._save_provider_data(provider, stations_list, inventory)
                        
            except Exception as e:
                self.logger.error(f"Provider {provider} 处理异常: {e}")
        
        # 后处理和保存数据
        if self.all_stations:
            df_all = pd.DataFrame(self.all_stations)
            df_all = self._postprocess_dataframe(df_all)
            
            if not df_all.empty:
                self._save_combined_data(df_all)
                self._generate_comprehensive_report(df_all)
                
                total_time = time.time() - self.query_start_time
                successful_count = sum(1 for stats in self.provider_stats.values() if stats['success'])
                self.logger.info(f"✅ 查询完成! 从 {successful_count}/{len(providers)} 个Provider获取 "
                               f"{len(df_all)} 个台站，耗时 {total_time:.1f} 秒")
                return df_all
        
        self.logger.warning("⚠️  所有Provider都未能返回有效数据")
        return pd.DataFrame()

    def _postprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        后处理DataFrame：清理、筛选、排序
        
        Args:
            df: 原始台站数据DataFrame
            
        Returns:
            处理后的DataFrame
        """
        self.logger.info("🔧 开始数据后处理...")
        initial_count = len(df)
        
        try:
            # 1. 数据清理
            df = self._clean_dataframe(df)
            
            # 2. 应用质量筛选器
            df = self._apply_quality_filters(df)
            
            # 3. 数据排序和索引重置
            df = df.sort_values(['Network', 'Station']).reset_index(drop=True)
            
            # 4. 添加统计信息
            df['QueryDate'] = datetime.now().strftime('%Y-%m-%d')
            df['DataSource'] = f"EASTASIA-FWI-{self.base_config.region['name']}"
            
            processed_count = len(df)
            self.logger.info(f"✅ 数据后处理完成: {initial_count} → {processed_count} 台站")
            
        except Exception as e:
            self.logger.error(f"数据后处理错误: {e}")
            
        return df

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理DataFrame中的无效数据"""
        # 移除坐标缺失的记录
        df = df.dropna(subset=['Latitude', 'Longitude'])
        
        # 移除无效坐标
        mask = ((df['Latitude'] >= -90) & (df['Latitude'] <= 90) & 
                (df['Longitude'] >= -180) & (df['Longitude'] <= 180))
        df = df.loc[mask].copy()
        
        # 填充缺失的数值字段
        df['Elevation'] = pd.to_numeric(df['Elevation'], errors='coerce').fillna(0)
        
        # 填充缺失的字符串字段
        string_fields = ['Site', 'StartDate', 'EndDate', 'NetworkDescription']
        for field in string_fields:
            if field in df.columns:
                df[field] = df[field].fillna('')
        
        return df

    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用质量筛选器
        
        Args:
            df: 输入DataFrame
            
        Returns:
            筛选后的DataFrame
        """
        self.logger.info("🔍 应用质量筛选器...")
        initial_count = len(df)
        filters = self.config.quality_filters
        
        try:
            # 海拔筛选
            min_elev = filters.get('min_elevation', -10000)
            max_elev = filters.get('max_elevation', 10000)
            mask = (df['Elevation'] >= min_elev) & (df['Elevation'] <= max_elev)
            df = df.loc[mask].copy()
            
            # 排除特定台网
            excluded_networks = filters.get('excluded_networks', [])
            if excluded_networks:
                mask = ~df['Network'].isin(excluded_networks)
                df = df.loc[mask].copy()
            
            # 优先台网筛选（如果配置了）
            preferred_networks = filters.get('preferred_networks', [])
            if preferred_networks:
                df['network_priority'] = df['Network'].apply(
                    lambda x: 0 if x in preferred_networks else 1
                )
                # 按优先级排序，但不过滤
                df = df.sort_values('network_priority').drop('network_priority', axis=1)
            
            filtered_count = initial_count - len(df)
            self.logger.info(f"✅ 质量筛选完成: 过滤 {filtered_count} 个台站")
            
        except Exception as e:
            self.logger.error(f"质量筛选错误: {e}")
            
        return df

    def _save_provider_data(self, provider: str, stations_list: List[Dict], 
                          inventory: Optional[Inventory]):
        """保存单个Provider的数据"""
        try:
            provider_dir = self.output_paths['providers'] / provider
            provider_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存CSV格式
            if stations_list:
                df = pd.DataFrame(stations_list)
                csv_file = provider_dir / f"{provider}_stations.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8')
            
            # 保存XML格式
            if inventory:
                xml_file = provider_dir / f"{provider}_stations.xml"
                inventory.write(str(xml_file), format="STATIONXML")
                
        except Exception as e:
            self.logger.error(f"保存Provider {provider} 数据失败: {e}")

    def _save_combined_data(self, df: pd.DataFrame):
        """保存合并后的数据"""
        try:
            self.logger.info("💾 保存合并数据...")
            
            # 保存CSV格式
            csv_path = self.output_paths['combined_csv']
            df.to_csv(csv_path, index=False, encoding='utf-8')
            self.logger.info(f"✅ 台站数据已保存: {csv_path} ({len(df)} 台站)")
            
            # 保存XML格式
            if len(self.total_inventory) > 0 and 'xml' in self.config.output.get('formats', []):
                xml_path = self.output_paths['combined_xml']
                self.total_inventory.write(str(xml_path), format="STATIONXML")
                self.logger.info(f"✅ 台站inventory已保存: {xml_path}")
            
            # 创建SPECFEM格式文件
            if self.config.output.get('create_specfem_format', True):
                self._create_specfem_stations_file(df)
            
            # 保存查询元数据
            self._save_query_metadata(df)
                
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")

    def _create_specfem_stations_file(self, df: pd.DataFrame) -> Optional[Path]:
        """创建SPECFEM3D格式的STATIONS文件"""
        try:
            specfem_file = self.output_paths['specfem_stations']
            self.logger.info(f"📄 创建SPECFEM3D台站文件...")
            
            with open(specfem_file, 'w', encoding='utf-8') as f:
                f.write(f"# STATIONS file for EASTASIA-FWI\n")
                f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total stations: {len(df)}\n")
                f.write(f"# Region: {self.base_config.region['name']}\n")
                f.write(f"# Time range: {self.config.time_range['start_time']} to {self.config.time_range['end_time']}\n")
                f.write(f"# Providers: {', '.join(self.config.fdsn['providers'])}\n")
                f.write("#\n")
                f.write("# Format: STATION NETWORK LATITUDE LONGITUDE ELEVATION BURIAL\n")
                
                for _, station in df.iterrows():
                    f.write(f"{station['Station']:<8} "
                           f"{station['Network']:<8} "
                           f"{station['Latitude']:>10.4f} "
                           f"{station['Longitude']:>11.4f} "
                           f"{station['Elevation']:>8.1f} "
                           f"0.0\n")
            
            self.logger.info(f"✅ SPECFEM3D台站文件已创建: {specfem_file}")
            return specfem_file
            
        except Exception as e:
            self.logger.error(f"创建SPECFEM文件失败: {e}")
            return None

    def _save_query_metadata(self, df: pd.DataFrame):
        """保存查询元数据"""
        try:
            metadata_file = self.output_paths['metadata']
            
            metadata = {
                'query_info': {
                    'timestamp': datetime.now().isoformat(),
                    'region': self.base_config.region,
                    'total_stations': len(df),
                    'total_networks': df['Network'].nunique(),
                    'total_providers': df['Provider'].nunique(),
                    'query_duration_seconds': time.time() - (self.query_start_time or 0)
                },
                'provider_stats': self.provider_stats,
                'failed_providers': self.failed_providers,
                'config_summary': {
                    'providers_queried': self.config.fdsn['providers'],
                    'time_range': self.config.time_range,
                    'query_bounds': {
                        'base_region': self.base_config.get_region_bounds(),
                        'buffer_degrees': self.config.query_bounds.get('buffer_degrees', 0.0)
                    },
                    'quality_filters': self.config.quality_filters
                },
                'data_statistics': self.get_summary_statistics(df) if not df.empty else {}
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"✅ 查询元数据已保存: {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"保存元数据失败: {e}")

    def _generate_comprehensive_report(self, df: pd.DataFrame):
        """生成详细的台站查询报告"""
        try:
            report_path = self.output_paths['report']
            self.logger.info("📊 生成查询报告...")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                # 报告头部
                f.write("="*80 + "\n")
                f.write("EASTASIA-FWI 台站查询详细报告\n")
                f.write("="*80 + "\n\n")
                
                # 基本信息
                f.write(f"查询时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"研究区域: {self.base_config.region['name']}\n")
                f.write(f"查询耗时: {time.time() - (self.query_start_time or 0):.1f} 秒\n\n")
                
                # 查询统计
                f.write("📊 查询统计\n")
                f.write("-"*50 + "\n")
                f.write(f"总台站数量: {len(df):,}\n")
                f.write(f"台网数量: {df['Network'].nunique()}\n")
                f.write(f"Provider数量: {df['Provider'].nunique()}\n")
                successful_providers = sum(1 for stats in self.provider_stats.values() if stats['success'])
                f.write(f"成功Provider: {successful_providers}/{len(self.provider_stats)}\n")
                f.write(f"失败Provider: {len(self.failed_providers)}\n\n")
                
                # Provider详细统计
                f.write("🌐 Provider详细统计\n")
                f.write("-"*60 + "\n")
                f.write(f"{'Provider':<15} {'状态':<8} {'台站数':<8} {'台网数':<8} {'重试':<6}\n")
                f.write("-"*60 + "\n")
                
                for provider, stats in self.provider_stats.items():
                    status = "✅ 成功" if stats['success'] else "❌ 失败"
                    retry_info = f"{stats.get('retry_count', 0)}"
                    f.write(f"{provider:<15} {status:<8} {stats['station_count']:<8} "
                           f"{stats.get('network_count', 0):<8} {retry_info:<6}\n")
                
                if self.failed_providers:
                    f.write(f"\n❌ 失败的Provider详情:\n")
                    for provider in self.failed_providers:
                        if provider in self.provider_stats:
                            error = self.provider_stats[provider].get('error', 'Unknown error')
                            f.write(f"  • {provider}: {error}\n")
                
                # 数据分布统计
                f.write(f"\n📈 数据分布统计\n")
                f.write("-"*50 + "\n")
                
                # 按Provider统计
                f.write("按Provider分布:\n")
                provider_stats = df['Provider'].value_counts()
                for provider, count in provider_stats.head(10).items():
                    f.write(f"  {provider:<12}: {count:>5} 台站 ({count/len(df)*100:.1f}%)\n")
                
                # 按台网统计
                f.write(f"\n按台网分布 (Top 15):\n")
                network_stats = df['Network'].value_counts()
                for network, count in network_stats.head(15).items():
                    f.write(f"  {network:<8}: {count:>4} 台站\n")
                
                # 地理分布统计
                f.write(f"\n🌍 地理分布统计\n")
                f.write("-"*50 + "\n")
                f.write(f"纬度范围: {df['Latitude'].min():.3f}° ~ {df['Latitude'].max():.3f}°\n")
                f.write(f"经度范围: {df['Longitude'].min():.3f}° ~ {df['Longitude'].max():.3f}°\n")
                f.write(f"海拔范围: {df['Elevation'].min():.0f} ~ {df['Elevation'].max():.0f} m\n")
                f.write(f"平均海拔: {df['Elevation'].mean():.0f} m\n")
                f.write(f"海拔标准差: {df['Elevation'].std():.0f} m\n\n")
                
                # 配置摘要
                f.write("🔧 查询配置摘要\n")
                f.write("-"*50 + "\n")
                f.write(f"时间范围: {self.config.time_range['start_time']} ~ {self.config.time_range['end_time']}\n")
                f.write(f"查询超时: {self.config.fdsn['timeout']} 秒\n")
                f.write(f"最大重试: {self.config.fdsn['max_retries']} 次\n")
                f.write(f"缓冲区域: {self.config.query_bounds.get('buffer_degrees', 0.0)}°\n")
            
            self.logger.info(f"✅ 详细报告已保存: {report_path}")
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")

    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        获取台站数据的汇总统计信息
        
        Args:
            df: 台站数据DataFrame
            
        Returns:
            统计信息字典
        """
        if df.empty:
            return {}
        
        def safe_float(val: Any, default: float = 0.0) -> float:
            """安全转换为 float"""
            try:
                result = float(val)
                return result if result == result else default  # NaN check
            except (TypeError, ValueError):
                return default
        
        return {
            'total_stations': len(df),
            'total_networks': df['Network'].nunique(),
            'total_providers': df['Provider'].nunique(),
            'coordinate_bounds': {
                'lat_min': safe_float(df['Latitude'].min()),
                'lat_max': safe_float(df['Latitude'].max()),
                'lon_min': safe_float(df['Longitude'].min()),
                'lon_max': safe_float(df['Longitude'].max()),
            },
            'elevation_stats': {
                'min': safe_float(df['Elevation'].min()),
                'max': safe_float(df['Elevation'].max()),
                'mean': safe_float(df['Elevation'].mean()),
                'std': safe_float(df['Elevation'].std())
            },
            'top_networks': df['Network'].value_counts().head(10).to_dict(),
            'top_providers': df['Provider'].value_counts().to_dict()
        }


def main():
    """主函数 - 执行台站查询"""
    print("🚀 EASTASIA-FWI 台站查询系统")
    print("="*60)
    
    try:
        # 初始化查询器
        query = StationQuery()
        
        # 执行查询
        df_stations = query.query_all_providers()
        
        if not df_stations.empty:
            print(f"\n✅ 查询成功! 获取 {len(df_stations):,} 个台站信息")
            
            # 基础统计
            print(f"\n📊 基础统计:")
            print(f"  台网数量: {df_stations['Network'].nunique()}")
            print(f"  Provider数量: {df_stations['Provider'].nunique()}")
            print(f"  纬度范围: {df_stations['Latitude'].min():.2f}° ~ {df_stations['Latitude'].max():.2f}°")
            print(f"  经度范围: {df_stations['Longitude'].min():.2f}° ~ {df_stations['Longitude'].max():.2f}°")
            print(f"  海拔范围: {df_stations['Elevation'].min():.0f} ~ {df_stations['Elevation'].max():.0f} m")
            
            # 显示主要台网
            print(f"\n📡 主要台网 (Top 10):")
            top_networks = df_stations['Network'].value_counts().head(10)
            for network, count in top_networks.items():
                percentage = count / len(df_stations) * 100
                print(f"  {network}: {count} 台站 ({percentage:.1f}%)")
            
            # 显示Provider分布
            print(f"\n🌐 Provider分布:")
            for provider, count in df_stations['Provider'].value_counts().items():
                percentage = count / len(df_stations) * 100
                print(f"  {provider}: {count} 台站 ({percentage:.1f}%)")
            
            # 显示输出文件
            print(f"\n📁 输出文件:")
            for name, path in query.output_paths.items():
                if hasattr(path, 'exists') and path.exists():
                    if name == 'combined_csv':
                        file_size = path.stat().st_size / 1024 / 1024  # MB
                        print(f"  ✅ {name}: {path} ({file_size:.1f} MB)")
                    else:
                        print(f"  ✅ {name}: {path}")
            
        else:
            print("❌ 未能获取到台站信息")
            print("\n可能的原因:")
            print("  1. 网络连接问题")
            print("  2. FDSN服务暂时不可用") 
            print("  3. 查询参数设置问题")
            print("  4. 目标区域没有可用台站")
            
            # 显示失败的Provider详情
            if query.failed_providers:
                print(f"\n❌ 失败的Provider ({len(query.failed_providers)} 个):")
                for provider in query.failed_providers:
                    if provider in query.provider_stats:
                        error = query.provider_stats[provider].get('error', 'Unknown error')
                        print(f"  • {provider}: {error}")
            
    except KeyboardInterrupt:
        print("\n⚠️  用户中断查询")
    except Exception as e:
        print(f"❌ 程序执行错误: {e}")
        logging.exception("详细错误信息:")


if __name__ == "__main__":
    main()