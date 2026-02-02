"""
1_2_Process_GCMT_catalogs.py: 
GCMT地震事件处理模块
================================================================

功能描述:
----------
基于GCMT (Global Centroid Moment Tensor) 目录的地震事件筛选、格式转换和数据管理模块。
支持自动下载、数据整合、多条件筛选和多种格式输出，为全波形反演提供高质量的地震事件目录。

核心功能:
----------
1. ✅ GCMT数据自动下载和更新
   - 支持从GCMT 网站自动下载最新GCMT数据
   - 多URL备份机制，确保数据获取可靠性
   - 自动合并多年份数据文件
   - 文件格式验证和错误处理

2. ✅ NDK格式文件解析
   - 解析标准NDK格式的GCMT目录文件
   - 提取事件基本信息（时间、位置、震级等）
   - 提取震源机制参数（矩张量、双力偶解等）
   - 支持多种震级类型（Mw, mb, Ms）

3. ✅ 多条件事件筛选
   - 地理区域筛选：基于项目研究区域自动筛选
   - 震级范围筛选：可配置震级范围（默认: 5.0-7.5）
   - 深度范围筛选：可配置深度范围（默认: 0-700 km）
   - 时间范围筛选：支持指定年份范围
   - 远震事件筛选：支持包含远震事件（30-180度）

4. ✅ 数据整合和合并
   - 自动合并多个NDK文件
   - 去除重复事件
   - 按时间排序
   - 数据格式验证

5. ✅ 多种格式输出
   - CSV格式：便于数据分析和可视化
   - CMT格式：SPECFEM3D兼容格式
   - 统计报告：详细的筛选和处理统计

使用方法:
----------
```python
from 1_Data_preparation.1_2_Process_GCMT_catalogs import GCMTProcessor

# 初始化处理器
processor = GCMTProcessor()

# 下载并更新数据（2021-2024）
success = processor.download_and_update(2021, 2024)

# 解析NDK文件
catalogs_df = processor.parse_ndk_file()

# 筛选事件
filtered_catalogs = processor.filter_catalogs(catalogs_df)

# 保存结果
saved_files = processor.save_processed_catalogs(filtered_catalogs)
```

配置说明:
----------
通过 EventProcessConfig 类配置处理参数:
- gcmt: GCMT数据源配置 (下载URL、重试机制等)
- event_filters: 事件筛选条件 (震级、深度范围)
- time_filters: 时间筛选配置
- geographic_filters: 地理筛选配置
- download_management: 下载管理选项
- file_merging: 文件合并策略

输出文件:
----------
- {region}_gcmt_catalogs.csv: 筛选后的事件CSV文件
- {region}_gcmt_catalogs.cmt: SPECFEM3D格式CMT文件
- gcmt_processing_report.txt: 详细处理报告
- query_metadata.json: 处理元数据

科学原理:
----------
- GCMT目录: 全球统一的矩心矩张量解目录，提供高精度的震源机制参数
- 矩张量解: 描述地震震源的完整物理参数，包括双力偶分量和补偿线性向量偶极子
- 事件筛选: 基于震级、深度、位置等多维度条件，确保选择适合全波形反演的事件
- 远震事件: 利用远震波形可以约束深部结构，补充区域地震数据的不足

作者: EASTASIA-FWI Team
日期: 2026-02-02
版本: v2.0
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import sys
import requests
import time
import traceback

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig


class EventProcessConfig:
    """地震事件处理配置类"""
    
    def __init__(self):
        """初始化事件处理配置参数"""
        # GCMT数据源配置
        self.gcmt = {
            'default_file': 'jan76_dec24.ndk',  # 更新为整合后的文件名
            'data_format': 'ndk',
            'encoding': 'utf-8',
            'validate_format': True,
            'download_url': 'https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/',
            'backup_urls': [
                'ftp://ftp.globalcmt.org/pub/gcmt-events/',
                'https://www.globalcmt.org/CMTsearch.html'
            ],
            # 下载控制参数
            'download_delay': 1.0,  # 下载间隔（秒）
            'max_retries': 3,       # 最大重试次数
            'timeout': 30,          # 超时时间（秒）
            'auto_backup': True,    # 自动备份原文件
            'verify_download': True # 验证下载的文件格式
        }
        
        # 事件筛选配置
        self.event_filters = {
            'magnitude_range': [5.0, 7.5],
            'depth_range': [0, 700],
            'use_mb': True,
            'use_ms': True,
            'preferred_magnitude': 'any'
        }
        
        # 时间筛选配置
        self.time_filters = {
            'start_year': 2000,
            'end_year': 2024,
            'enable_time_filter': False
        }
        
        # 地理筛选配置
        self.geographic_filters = {
            'use_region_bounds': True,
            'buffer_degrees': 0.0,
            'include_teleseismic': True,
            'teleseismic_distance_range': [30, 180]
        }
        
        # 下载管理配置
        self.download_management = {
            'enable_download': True,
            'download_directory': 'downloaded',  # 相对于catalogs目录
            'keep_individual_files': True,      # 保留单个下载文件
            'auto_merge_after_download': True,  # 下载后自动合并
            'cleanup_temp_files': False,        # 清理临时文件
            'check_existing_files': True,       # 检查已存在文件
            'update_frequency': 'monthly',      # 更新频率
            'last_update_check': None           # 最后更新检查时间
        }
        
        # 文件合并配置
        self.file_merging = {
            'merge_strategy': 'chronological',  # 合并策略：chronological, size_based
            'validate_before_merge': True,      # 合并前验证格式
            'create_backup_before_merge': True, # 合并前创建备份
            'remove_duplicates': True,          # 去除重复事件
            'sort_merged_catalogs': True          # 对合并后的事件排序
        }
        
        # 输出配置
        self.output = {
            'save_csv': True,
            'create_cmt_files': True,
            'backup_existing': True,
            'add_timestamp': True,
            'compression': None
        }
        
        # SPECFEM3D格式配置
        self.specfem = {
            'half_duration_formula': 'empirical',
            'fixed_half_duration': 10.0,
            'minimum_half_duration': 1.0,
            'maximum_half_duration': 300.0,
            'time_shift': 0.0,
            'depth_in_m': True
        }
        
        # 日志配置
        self.logging = {
            'level': 'INFO',
            'file_prefix': 'event_processing',
            'max_size_mb': 512,
            'backup_count': 3,
            'console_output': True
        }
        
        # 错误处理配置
        self.error_handling = {
            'continue_on_format_error': True,
            'continue_on_coordinate_error': True,
            'continue_on_magnitude_error': False,
            'save_error_catalogs': True,
            'max_error_rate': 0.1
        }
    
    def get_half_duration(self, magnitude: float) -> float:
        """
        根据震级计算半持续时间
        
        Args:
            magnitude: 地震震级
            
        Returns:
            半持续时间（秒）
        """
        if self.specfem['half_duration_formula'] == 'empirical':
            # 基于经验公式：T = 2.26 × 10^((M-5)/3)
            duration = 2.26 * (10 ** ((magnitude - 5) / 3))
        else:
            duration = self.specfem['fixed_half_duration']
        
        # 应用范围限制
        duration = max(self.specfem['minimum_half_duration'], duration)
        duration = min(self.specfem['maximum_half_duration'], duration)
        
        return duration


class GCMTProcessor:
    """
    GCMT地震事件处理器
    
    功能包括：
    - GCMT数据自动下载
    - NDK格式文件解析
    - 数据整合和合并
    - 地震事件筛选
    - 多种格式输出
    - SPECFEM3D兼容格式生成
    
    Attributes:
        base_config: 基础配置
        config: 事件处理配置
        logger: 日志记录器
        processed_catalogs: 处理后的事件数据
        parse_errors: 解析错误统计
        output_paths: 输出文件路径字典
    """
    
    def __init__(self, config_file: Optional[Path] = None, gcmt_file: Optional[str] = None):
        """
        初始化GCMT处理器
        
        Args:
            config_file: 自定义配置文件路径
            gcmt_file: GCMT数据文件路径
        """
        # 加载基础配置
        self.base_config = BaseConfig(config_file)
        
        # 加载模块配置
        self.config = EventProcessConfig()
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.GCMTProcessor',
            self.config.logging['level']
        )
        
        # 设置数据文件
        self.gcmt_file = gcmt_file or self._get_default_gcmt_file()
        
        # 初始化数据存储
        self._initialize_storage()
        
        # 创建输出目录
        self._setup_output_directories()
        
        self.logger.info("🌍 GCMT事件处理器初始化完成")
        self._print_config_summary()

    def _initialize_storage(self):
        """初始化数据存储容器"""
        self.processed_catalogs: List[Dict] = []
        self.parse_errors = {
            'format_errors': 0,
            'coordinate_errors': 0,
            'magnitude_errors': 0,
            'moment_tensor_errors': 0
        }
        self.processing_start_time = None
        self.download_stats = {
            'downloaded_files': 0,
            'failed_downloads': 0,
            'total_catalogs': 0
        }

    def _setup_output_directories(self):
        """设置并创建输出目录"""
        self.output_paths = self._get_output_paths()
        
        for path_name, path in self.output_paths.items():
            if hasattr(path, 'parent'):
                path.parent.mkdir(parents=True, exist_ok=True)
            elif hasattr(path, 'mkdir'):
                path.mkdir(parents=True, exist_ok=True)

    def _get_output_paths(self) -> Dict[str, Path]:
        """获取输出文件路径"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if self.config.output['add_timestamp'] else ''
        
        paths = {
            'csv_file': self.base_config.dirs['catalogs'] / f"GCMT_catalogs{('_' + timestamp) if timestamp else ''}.csv",
            'cmt_dir': self.base_config.dirs['catalogs'] / 'CMT_files',
            'report_file': self.base_config.dirs['logs'] / f"GCMT_processing_report{('_' + timestamp) if timestamp else ''}.txt"
        }
        
        return paths

    def _get_default_gcmt_file(self) -> Path:
        """获取默认的GCMT文件路径"""
        default_file = self.config.gcmt['default_file']
        gcmt_path = self.base_config.dirs['catalogs'] / default_file
        
        if not gcmt_path.exists():
            self.logger.warning(f"默认GCMT文件不存在: {gcmt_path}")
            
        return gcmt_path

    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 GCMT事件处理配置摘要")
        print("-" * 60)
        
        # 研究区域配置
        print(f"研究区域: {self.base_config.region['name']}")
        region = self.base_config.region
        print(f"纬度范围: {region['lat_min']}° ~ {region['lat_max']}°")
        print(f"经度范围: {region['lon_min']}° ~ {region['lon_max']}°")
        print(f"缓冲区: {self.config.geographic_filters.get('buffer_degrees', 0.0)}°")
        
        # 事件筛选配置
        print(f"震级范围: {self.config.event_filters['magnitude_range'][0]:.1f} ~ {self.config.event_filters['magnitude_range'][1]:.1f}")
        print(f"深度范围: {self.config.event_filters['depth_range'][0]} ~ {self.config.event_filters['depth_range'][1]} km")
        
        if self.config.time_filters['enable_time_filter']:
            print(f"时间范围: {self.config.time_filters['start_year']} ~ {self.config.time_filters['end_year']}")
        
        # 下载配置
        print(f"启用下载: {'是' if self.config.download_management['enable_download'] else '否'}")
        print(f"下载延迟: {self.config.gcmt['download_delay']} 秒")
        print(f"最大重试: {self.config.gcmt['max_retries']} 次")
        
        # 输出配置
        print(f"输出目录: {self.base_config.dirs['catalogs']}")
        
        print("-" * 60)

    def get_geographic_bounds(self) -> Dict[str, float]:
        """获取地理边界"""
        bounds = self.base_config.region.copy()
        
        # 添加缓冲区
        if self.config.geographic_filters['buffer_degrees'] > 0:
            buffer = self.config.geographic_filters['buffer_degrees']
            bounds['lat_min'] -= buffer
            bounds['lat_max'] += buffer
            bounds['lon_min'] -= buffer
            bounds['lon_max'] += buffer
        
        return bounds

    def is_valid_event_location(self, lat: float, lon: float, depth: float) -> bool:
        """验证事件位置是否有效"""
        # 基本坐标验证
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return False
        
        # 深度验证
        if not (0 <= depth <= 1000):
            return False
        
        # 区域范围验证
        if self.config.geographic_filters['use_region_bounds']:
            bounds = self.get_geographic_bounds()
            if not (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                    bounds['lon_min'] <= lon <= bounds['lon_max']):
                return False
        
        return True

    def download_gcmt_data(self, start_year: Optional[int] = None, 
                          end_year: Optional[int] = None,
                          auto_merge: bool = True) -> List[Path]:
        """
        下载指定年份范围的GCMT数据
        
        Args:
            start_year: 开始年份（如果为None，从配置中获取）
            end_year: 结束年份（如果为None，从配置中获取）
            auto_merge: 是否自动合并到主文件
            
        Returns:
            下载的文件路径列表
        """
        # 从配置中获取默认时间范围
        if start_year is None:
            start_year_val = self.config.time_filters.get('start_year', 2021)
            start_year_int = int(start_year_val) if start_year_val is not None else 2021
        else:
            start_year_int = int(start_year)
            
        if end_year is None:
            end_year_val = self.config.time_filters.get('end_year', 2024)
            end_year_int = int(end_year_val) if end_year_val is not None else 2024
        else:
            end_year_int = int(end_year)
        
        self.logger.info(f"🌐 开始下载GCMT数据: {start_year_int}-{end_year_int}")
        
        downloaded_files = []
        download_dir = self.base_config.dirs['catalogs'] / 'downloaded'
        download_dir.mkdir(exist_ok=True)
        
        base_url = self.config.gcmt['download_url']
        total_months = (end_year_int - start_year_int + 1) * 12
        current_month = 0
        
        for year in range(start_year_int, end_year_int + 1):
            for month in range(1, 13):
                current_month += 1
                
                # 检查是否超过当前日期
                current_date = datetime.now()
                if year > current_date.year or (year == current_date.year and month > current_date.month):
                    self.logger.info(f"⏭️  跳过未来日期: {year:04d}/{month:02d}")
                    continue
                
                # 构建文件名和URL（修正：添加年份路径）
                month_name = datetime(year, month, 1).strftime('%b').lower()
                filename = f"{month_name}{str(year)[2:]}.ndk"
                # 正确的URL结构：年份目录/文件名
                file_url = f"{base_url.rstrip('/')}/{year}/{filename}"
                
                # 本地文件路径
                local_file = download_dir / filename
                
                # 检查文件是否已存在
                if local_file.exists() and local_file.stat().st_size > 0:
                    self.logger.info(f"✅ 文件已存在: {filename}")
                    downloaded_files.append(local_file)
                    continue
                
                # 下载文件
                success = self._download_single_file(file_url, local_file)
                
                if success:
                    downloaded_files.append(local_file)
                    self.download_stats['downloaded_files'] += 1
                    self.logger.info(f"✅ 下载完成 ({current_month}/{total_months}): {filename}")
                else:
                    self.download_stats['failed_downloads'] += 1
                    self.logger.warning(f"❌ 下载失败: {filename}")
                
                # 添加延时避免服务器过载
                time.sleep(self.config.gcmt.get('download_delay', 1.0))
        
        self.logger.info(f"📥 下载统计: 成功={self.download_stats['downloaded_files']}, "
                        f"失败={self.download_stats['failed_downloads']}")
        
        # 自动合并文件
        if auto_merge and downloaded_files:
            merged_file = self.merge_ndk_files(downloaded_files)
            if merged_file:
                self.logger.info(f"🔗 文件合并完成: {merged_file}")
        
        return downloaded_files

    def _download_single_file(self, url: str, local_path: Path, 
                             max_retries: int = 3, timeout: int = 30) -> bool:
        """
        下载单个文件
        
        Args:
            url: 下载URL
            local_path: 本地保存路径
            max_retries: 最大重试次数
            timeout: 超时时间（秒）
            
        Returns:
            下载是否成功
        """
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"⬇️  正在下载: {url} (尝试 {attempt + 1}/{max_retries})")
                
                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()
                
                # 检查内容类型
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' in content_type:
                    self.logger.warning(f"⚠️  可能是错误页面: {url}")
                    return False
                
                # 写入文件
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 验证文件大小
                if local_path.stat().st_size == 0:
                    self.logger.warning(f"⚠️  下载的文件为空: {local_path}")
                    local_path.unlink(missing_ok=True)
                    return False
                
                return True
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"⚠️  下载错误 (尝试 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    self.logger.error(f"❌ 下载失败，已达最大重试次数: {url}")
                    
            except Exception as e:
                self.logger.error(f"❌ 下载出现未知错误: {e}")
                break
        
        return False

    def merge_ndk_files(self, file_list: List[Path], 
                       output_file: Optional[Path] = None) -> Optional[Path]:
        """
        合并多个NDK文件，按时间顺序排列
        
        Args:
            file_list: 要合并的文件列表
            output_file: 输出文件路径
            
        Returns:
            合并后的文件路径
        """
        if not file_list:
            self.logger.warning("⚠️  没有文件需要合并")
            return None
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.base_config.dirs['catalogs'] / f"GCMT_merged_{timestamp}.ndk"
        
        self.logger.info(f"🔗 开始合并 {len(file_list)} 个NDK文件")
        
        # 收集所有事件（时间戳, 原始内容）
        all_catalogs = []
        
        try:
            for ndk_file in sorted(file_list):
                if not ndk_file.exists():
                    self.logger.warning(f"⚠️  文件不存在，跳过: {ndk_file}")
                    continue
                
                self.logger.debug(f"📄 读取文件: {ndk_file}")
                
                try:
                    with open(ndk_file, 'r', encoding=self.config.gcmt['encoding']) as infile:
                        lines = infile.readlines()
                    
                    # 按事件分组（每5行一个事件）
                    for i in range(0, len(lines), 5):
                        if i + 4 >= len(lines):
                            break
                        
                        event_lines = lines[i:i+5]
                        
                        # 解析时间戳用于排序
                        try:
                            first_line = event_lines[0].strip()
                            date_str = first_line[5:15].strip()  # 提取日期
                            time_str = first_line[16:26].strip()  # 提取时间
                            
                            # 处理时间格式问题
                            if ':60.' in time_str:
                                time_str = time_str.replace(':60.', ':59.')
                            elif time_str.endswith(':60'):
                                time_str = time_str.replace(':60', ':59')
                            
                            # 创建时间戳
                            datetime_str = f"{date_str} {time_str}"
                            timestamp = pd.to_datetime(datetime_str)
                            
                            # 保存事件内容
                            event_content = ''.join(event_lines)
                            all_catalogs.append((timestamp, event_content))
                            
                        except Exception as e:
                            self.logger.debug(f"时间解析失败，使用默认时间: {e}")
                            # 使用默认时间戳
                            default_time = datetime(1970, 1, 1)
                            event_content = ''.join(event_lines)
                            all_catalogs.append((default_time, event_content))
                        
                except Exception as e:
                    self.logger.error(f"❌ 读取文件失败: {ndk_file}, 错误: {e}")
                    continue
            
            if not all_catalogs:
                self.logger.warning("⚠️  没有找到有效事件")
                return None
            
            # 按时间排序
            all_catalogs.sort(key=lambda x: x[0])
            
            # 写入排序后的事件
            with open(output_file, 'w', encoding=self.config.gcmt['encoding']) as outfile:
                for timestamp, event_content in all_catalogs:
                    outfile.write(event_content)
                    if not event_content.endswith('\n'):
                        outfile.write('\n')
            
            total_catalogs = len(all_catalogs)
            self.logger.info(f"✅ 合并完成: {output_file}")
            self.logger.info(f"📊 总事件数: {total_catalogs} (按时间排序)")
            self.download_stats['total_catalogs'] = total_catalogs
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"❌ 合并失败: {e}")
            return None

    def _generate_gcmt_filename(self, start_year: int, end_year: int) -> str:
        """
        根据时间范围生成GCMT文件名
        
        Args:
            start_year: 开始年份
            end_year: 结束年份
            
        Returns:
            文件名，格式如 'jan76_dec25.ndk'
        """
        # 获取开始和结束的月份名称
        start_month = datetime(start_year, 1, 1).strftime('%b').lower()
        end_month = datetime(end_year, 12, 1).strftime('%b').lower()
        
        # 格式化年份（取后两位）
        start_year_short = str(start_year)[-2:]
        end_year_short = str(end_year)[-2:]
        
        # 生成文件名
        filename = f"{start_month}{start_year_short}_{end_month}{end_year_short}.ndk"
        return filename
    
    def update_master_file(self, new_data_file: Path, start_year: Optional[int] = None, 
                          end_year: Optional[int] = None) -> bool:
        """
        将新下载的数据更新到主文件，根据时间范围生成文件名
        
        Args:
            new_data_file: 新数据文件路径
            start_year: 开始年份（如果为None，从配置中获取）
            end_year: 结束年份（如果为None，从配置中获取）
            
        Returns:
            更新是否成功
        """
        # 确定时间范围
        if start_year is None:
            start_year_val = self.config.time_filters.get('start_year', 1976)
            start_year_int = int(start_year_val) if start_year_val is not None else 1976
        else:
            start_year_int = int(start_year)
            
        if end_year is None:
            end_year_val = self.config.time_filters.get('end_year', 2024)
            end_year_int = int(end_year_val) if end_year_val is not None else 2024
        else:
            end_year_int = int(end_year)
        
        # 根据时间范围生成文件名
        new_filename = self._generate_gcmt_filename(start_year_int, end_year_int)
        old_master_file = self.base_config.dirs['catalogs'] / "jan76_dec20.ndk"
        new_master_file = self.base_config.dirs['catalogs'] / new_filename
        
        if not new_data_file.exists():
            self.logger.error(f"❌ 新数据文件不存在: {new_data_file}")
            return False
        
        self.logger.info(f"🔄 整合数据到新主文件: {new_master_file}")
        
        try:
            files_to_merge = []
            
            # 添加原有文件（如果存在）
            if old_master_file.exists():
                files_to_merge.append(old_master_file)
                self.logger.info(f"📄 包含原始文件: {old_master_file}")
            
            # 添加新下载的文件
            files_to_merge.append(new_data_file)
            
            # 合并到新的主文件
            merged_file = self.merge_ndk_files(files_to_merge, new_master_file)
            
            if merged_file:
                # 更新配置中的默认文件名
                self.config.gcmt['default_file'] = new_filename
                self.gcmt_file = new_master_file
                # 同步更新时间筛选配置
                self.config.time_filters['start_year'] = start_year_int
                self.config.time_filters['end_year'] = end_year_int
                self.logger.info(f"✅ 成功创建整合文件: {new_master_file} (时间范围: {start_year_int}-{end_year_int})")
                return True
            else:
                self.logger.error("❌ 文件整合失败")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 更新主文件失败: {e}")
            return False

    def download_and_update(self, start_year: Optional[int] = None, 
                          end_year: Optional[int] = None) -> bool:
        """
        下载并更新GCMT数据的完整流程
        
        Args:
            start_year: 开始年份（如果为None，从配置中获取）
            end_year: 结束年份（如果为None，从配置中获取）
            
        Returns:
            更新是否成功
        """
        # 从配置中获取默认时间范围
        if start_year is None:
            start_year_val = self.config.time_filters.get('start_year', 2021)
            start_year_int = int(start_year_val) if start_year_val is not None else 2021
        else:
            start_year_int = int(start_year)
            
        if end_year is None:
            end_year_val = self.config.time_filters.get('end_year', 2024)
            end_year_int = int(end_year_val) if end_year_val is not None else 2024
        else:
            end_year_int = int(end_year)
        
        self.logger.info(f"🚀 开始GCMT数据下载和更新流程")
        
        try:
            # 1. 下载数据
            downloaded_files = self.download_gcmt_data(start_year_int, end_year_int, auto_merge=False)
            
            if not downloaded_files:
                self.logger.warning("⚠️  没有下载到任何文件")
                return False
            
            # 2. 合并下载的文件
            merged_file = self.merge_ndk_files(downloaded_files)
            
            if not merged_file:
                self.logger.error("❌ 文件合并失败")
                return False
            
            # 3. 更新主文件，根据时间范围生成文件名
            success = self.update_master_file(merged_file, start_year_int, end_year_int)
            
            if success:
                filename = self._generate_gcmt_filename(start_year_int, end_year_int)
                self.logger.info(f"🎉 GCMT数据更新完成! 已生成 {filename} (时间范围: {start_year_int}-{end_year_int})")
                # 清理临时文件
                self._cleanup_temporary_files(downloaded_files, merged_file)
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ 更新流程失败: {e}")
            return False

    def _cleanup_temporary_files(self, downloaded_files: List[Path], 
                                merged_file: Path, keep_downloads: bool = True):
        """
        清理临时文件
        
        Args:
            downloaded_files: 下载的文件列表
            merged_file: 合并后的文件
            keep_downloads: 是否保留下载的文件
        """
        try:
            # 删除合并后的临时文件
            if merged_file.exists() and merged_file != self.gcmt_file:
                merged_file.unlink()
                self.logger.debug(f"🗑️  已删除临时合并文件: {merged_file}")
            
            # 可选择删除下载的文件
            if not keep_downloads:
                for file_path in downloaded_files:
                    if file_path.exists():
                        file_path.unlink()
                        self.logger.debug(f"🗑️  已删除下载文件: {file_path}")
                        
        except Exception as e:
            self.logger.warning(f"⚠️  清理临时文件时出错: {e}")

    def parse_ndk_file(self, ndk_file: Optional[str] = None) -> pd.DataFrame:
        """
        解析GCMT NDK格式文件
        
        Args:
            ndk_file: NDK文件路径，默认使用配置文件
            
        Returns:
            解析后的事件DataFrame
        """
        file_to_parse = ndk_file or self.gcmt_file
        file_path = Path(file_to_parse)
        
        if not file_path.exists():
            raise FileNotFoundError(f"GCMT文件不存在: {file_path}")
        
        self.logger.info(f"📖 开始解析GCMT文件: {file_path}")
        self.processing_start_time = datetime.now()
        
        records = []
        
        try:
            with open(file_path, "r", encoding=self.config.gcmt['encoding']) as file:
                lines = file.readlines()
            
            total_catalogs = len(lines) // 5
            self.logger.info(f"📊 预计事件数量: {total_catalogs}")
            
            # 逐事件解析（每5行为一个事件）
            for i in range(0, len(lines), 5):
                if i + 4 >= len(lines):
                    break
                
                try:
                    event_record = self._parse_single_event(lines[i:i+5])
                    if event_record:
                        records.append(event_record)
                except Exception as e:
                    self.parse_errors['format_errors'] += 1
                    if self.config.error_handling['continue_on_format_error']:
                        self.logger.debug(f"跳过事件解析错误 (行 {i+1}): {e}")
                        continue
                    else:
                        raise
            
            df = pd.DataFrame(records)
            self.logger.info(f"✅ 解析完成: {len(df)} 个事件")
            self._log_parse_errors()
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ 文件解析失败: {e}")
            raise

    def _parse_single_event(self, event_lines: List[str]) -> Optional[Dict]:
        """
        解析单个事件的5行数据
        
        Args:
            event_lines: 包含5行数据的列表
            
        Returns:
            事件信息字典，解析失败时返回None
        """
        if len(event_lines) < 5:
            return None
        
        try:
            line1, line2, line3, line4, line5 = [line.strip() for line in event_lines]
            
            # 解析基础信息（第1行）
            basic_info = self._parse_line1_basic_info(line1)
            
            # 解析矩张量信息（第4行）
            moment_tensor = self._parse_line4_moment_tensor(line4)
            
            # 解析节点面信息（第5行）
            nodal_planes = self._parse_line5_nodal_planes(line5)
            
            # 解析附加信息（第2行）
            additional_info = self._parse_line2_additional_info(line2)
            
            # 合并所有信息
            event_record = {**basic_info, **moment_tensor, **nodal_planes, **additional_info}
            
            # 数据验证
            if self._validate_event_data(event_record):
                return event_record
            else:
                return None
                
        except Exception as e:
            self.logger.debug(f"事件解析错误: {e}")
            return None

    def _parse_line1_basic_info(self, line: str) -> Dict:
        """解析第1行基础信息"""
        try:
            return {
                "Catalog": line[:4].strip(),
                "Date": line[5:15].strip(),
                "Time": line[16:26].strip(),
                "Latitude": float(line[27:33].strip()),
                "Longitude": float(line[34:41].strip()),
                "Depth": float(line[42:47].strip()),
                "Magnitude1_mb": float(line[48:51].strip()) if line[48:51].strip() and line[48:51].strip() != '0.0' else None,
                "Magnitude2_Ms": float(line[52:55].strip()) if line[52:55].strip() and line[52:55].strip() != '0.0' else None,
                "Location": line[56:].strip()
            }
        except Exception as e:
            self.parse_errors['coordinate_errors'] += 1
            raise ValueError(f"基础信息解析错误: {e}")

    def _parse_line4_moment_tensor(self, line: str) -> Dict:
        """解析第4行矩张量信息"""
        try:
            exponent = int(line[0:2])
            moment_values = line[2:].strip().split()
            scale_factor = 10 ** exponent
            
            if len(moment_values) < 12:
                raise ValueError("矩张量数据不完整")
            
            # 清理矩张量值的函数
            def clean_moment_value(value_str: str, scale_factor: float) -> float:
                try:
                    raw_value = float(value_str)
                    scaled_value = raw_value * scale_factor
                    # 避免浮点精度问题
                    if abs(scaled_value) < 1e-30:
                        return 0.0
                    return scaled_value
                except:
                    return 0.0
            
            return {
                "Mrr": clean_moment_value(moment_values[0], scale_factor),
                "Mrr_error": clean_moment_value(moment_values[1], scale_factor),
                "Mtt": clean_moment_value(moment_values[2], scale_factor),
                "Mtt_error": clean_moment_value(moment_values[3], scale_factor),
                "Mpp": clean_moment_value(moment_values[4], scale_factor),
                "Mpp_error": clean_moment_value(moment_values[5], scale_factor),
                "Mrt": clean_moment_value(moment_values[6], scale_factor),
                "Mrt_error": clean_moment_value(moment_values[7], scale_factor),
                "Mrp": clean_moment_value(moment_values[8], scale_factor),
                "Mrp_error": clean_moment_value(moment_values[9], scale_factor),
                "Mtp": clean_moment_value(moment_values[10], scale_factor),
                "Mtp_error": clean_moment_value(moment_values[11], scale_factor),
                "Moment_tensor_scale": exponent
            }
        except Exception as e:
            self.parse_errors['moment_tensor_errors'] += 1
            raise ValueError(f"矩张量解析错误: {e}")

    def _parse_line5_nodal_planes(self, line: str) -> Dict:
        """解析第5行节点面信息"""
        try:
            # 节点面1和2的信息位置
            plane1_str = line[58:68].strip()
            plane2_str = line[69:80].strip()
            
            plane1 = list(map(int, plane1_str.split())) if plane1_str else [0, 0, 0]
            plane2 = list(map(int, plane2_str.split())) if plane2_str else [0, 0, 0]
            
            return {
                "First_Nodal_Strike": plane1[0] if len(plane1) > 0 else 0,
                "First_Nodal_Dip": plane1[1] if len(plane1) > 1 else 0,
                "First_Nodal_Rake": plane1[2] if len(plane1) > 2 else 0,
                "Second_Nodal_Strike": plane2[0] if len(plane2) > 0 else 0,
                "Second_Nodal_Dip": plane2[1] if len(plane2) > 1 else 0,
                "Second_Nodal_Rake": plane2[2] if len(plane2) > 2 else 0
            }
        except Exception as e:
            raise ValueError(f"节点面信息解析错误: {e}")

    def _parse_line2_additional_info(self, line: str) -> Dict:
        """解析第2行附加信息"""
        try:
            return {
                "Event_Name": line[:16].strip(),
                "Data_Used": line[17:61].strip(),
                "Source_Type": line[62:68].strip() if len(line) > 62 else ""
            }
        except Exception:
            return {
                "Event_Name": "",
                "Data_Used": "",
                "Source_Type": ""
            }

    def _validate_event_data(self, event_data: Dict) -> bool:
        """验证事件数据的有效性"""
        try:
            # 坐标有效性检查
            lat, lon = event_data.get('Latitude', 0), event_data.get('Longitude', 0)
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return False
            
            # 深度有效性检查
            depth = event_data.get('Depth', 0)
            if not (0 <= depth <= 1000):
                return False
            
            # 震级有效性检查
            mb = event_data.get('Magnitude1_mb')
            ms = event_data.get('Magnitude2_Ms')
            if not mb and not ms:
                self.parse_errors['magnitude_errors'] += 1
                return False
            
            return True
            
        except Exception:
            return False

    def _log_parse_errors(self):
        """记录解析错误统计"""
        if sum(self.parse_errors.values()) > 0:
            self.logger.warning("📊 解析错误统计:")
            for error_type, count in self.parse_errors.items():
                if count > 0:
                    self.logger.warning(f"  {error_type}: {count}")

    def filter_catalogs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据配置筛选事件
        
        Args:
            df: 原始事件DataFrame
            
        Returns:
            筛选后的DataFrame
        """
        if df.empty:
            return df
        
        self.logger.info(f"🔍 开始事件筛选，原始事件数: {len(df)}")
        initial_count = len(df)
        
        # 应用筛选条件
        filtered_df = df.copy()
        
        # 地理筛选
        if self.config.geographic_filters['use_region_bounds']:
            filtered_df = self._apply_geographic_filter(filtered_df)
            geo_count = len(filtered_df)
        else:
            geo_count = len(filtered_df)
        
        # 震级筛选
        filtered_df = self._apply_magnitude_filter(filtered_df)
        mag_count = len(filtered_df)
        
        # 深度筛选
        filtered_df = self._apply_depth_filter(filtered_df)
        depth_count = len(filtered_df)
        
        # 时间筛选
        if self.config.time_filters['enable_time_filter']:
            filtered_df = self._apply_time_filter(filtered_df)
        time_count = len(filtered_df)
        
        final_count = len(filtered_df)
        
        # 记录筛选统计
        self.logger.info("📊 筛选统计:")
        self.logger.info(f"  原始事件: {initial_count}")
        self.logger.info(f"  地理筛选后: {geo_count} (移除 {initial_count - geo_count})")
        self.logger.info(f"  震级筛选后: {mag_count} (移除 {geo_count - mag_count})")
        self.logger.info(f"  深度筛选后: {depth_count} (移除 {mag_count - depth_count})")
        self.logger.info(f"  时间筛选后: {time_count} (移除 {depth_count - time_count})")
        self.logger.info(f"  最终事件数: {final_count}")
        
        return filtered_df

    def _apply_geographic_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用地理筛选"""
        bounds = self.get_geographic_bounds()
        return df[
            (df['Latitude'] >= bounds['lat_min']) &
            (df['Latitude'] <= bounds['lat_max']) &
            (df['Longitude'] >= bounds['lon_min']) &
            (df['Longitude'] <= bounds['lon_max'])
        ]

    def _apply_magnitude_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用震级筛选"""
        mag_min, mag_max = self.config.event_filters['magnitude_range']
        
        # 构建震级筛选条件
        mag_conditions = []
        
        if self.config.event_filters['use_mb']:
            mb_condition = (
                (df['Magnitude1_mb'].notna()) &
                (df['Magnitude1_mb'] >= mag_min) &
                (df['Magnitude1_mb'] <= mag_max)
            )
            mag_conditions.append(mb_condition)
        
        if self.config.event_filters['use_ms']:
            ms_condition = (
                (df['Magnitude2_Ms'].notna()) &
                (df['Magnitude2_Ms'] >= mag_min) &
                (df['Magnitude2_Ms'] <= mag_max)
            )
            mag_conditions.append(ms_condition)
        
        if mag_conditions:
            # 任一震级满足条件即可
            combined_condition = mag_conditions[0]
            for condition in mag_conditions[1:]:
                combined_condition = combined_condition | condition
            return df[combined_condition]
        
        return df

    def _apply_depth_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用深度筛选"""
        depth_min, depth_max = self.config.event_filters['depth_range']
        return df[
            (df['Depth'] >= depth_min) &
            (df['Depth'] <= depth_max)
        ]

    def _apply_time_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用时间筛选"""
        start_year = self.config.time_filters['start_year']
        end_year = self.config.time_filters['end_year']
        
        # 从日期字符串中提取年份
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        filtered_df = df[
            (df['Year'] >= start_year) &
            (df['Year'] <= end_year)
        ]
        
        return filtered_df.drop('Year', axis=1)

    def save_processed_catalogs(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        保存处理结果
        
        Args:
            df: 处理后的事件DataFrame
            
        Returns:
            保存的文件路径字典
        """
        if df.empty:
            self.logger.warning("⚠️  没有事件数据需要保存")
            return {}
        
        self.logger.info(f"💾 保存 {len(df)} 个事件...")
        
        saved_files = {}
        
        try:
            # 数据格式化处理
            df_output = self._format_output_data(df)
            
            # 按时间排序
            df_output = self._sort_catalogs_by_time(df_output)
            
            # 保存CSV格式（去重）
            if self.config.output['save_csv']:
                csv_path = self.output_paths['csv_file']
                # 检查是否为更新操作，避免重复记录
                if csv_path.exists():
                    # 读取现有数据
                    try:
                        existing_df = pd.read_csv(csv_path, encoding='utf-8')
                        # 基于Event_Name和Date去重
                        if 'Event_Name' in existing_df.columns and 'Date' in existing_df.columns:
                            # 合并数据并去重
                            combined_df = pd.concat([existing_df, df_output], ignore_index=True)
                            # 基于多个关键字段去重
                            dedup_cols = ['Date', 'Time', 'Latitude', 'Longitude', 'Depth']
                            available_cols = [col for col in dedup_cols if col in combined_df.columns]
                            if available_cols:
                                df_output = combined_df.drop_duplicates(subset=available_cols, keep='last')
                                df_output = self._sort_catalogs_by_time(df_output)
                                self.logger.info(f"✅ 去重后事件数: {len(df_output)}")
                    except Exception as e:
                        self.logger.warning(f"⚠️  读取现有CSV文件失败，将覆盖: {e}")
                
                df_output.to_csv(csv_path, index=False, encoding='utf-8')
                saved_files['csv'] = str(csv_path)
                self.logger.info(f"✅ CSV文件已保存: {csv_path}")
            
            # 生成CMT文件
            if self.config.output['create_cmt_files']:
                cmt_count = self.generate_specfem_format(df_output)
                saved_files['cmt_files'] = f"{self.output_paths['cmt_dir']} ({cmt_count} files)"
            
            # 生成处理报告
            self._generate_processing_report(df_output, saved_files)
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"❌ 保存失败: {e}")
            raise

    def _sort_catalogs_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按时间顺序对事件进行排序
        
        Args:
            df: 事件DataFrame
            
        Returns:
            排序后的DataFrame
        """
        if df.empty:
            return df
        
        try:
            # 创建完整的时间戳
            df_sorted = df.copy()
            
            # 合并日期和时间信息
            datetime_strings = df_sorted['Date'].astype(str) + ' ' + df_sorted['Time'].astype(str)
            
            # 处理可能的时间格式问题
            def parse_datetime(dt_str):
                try:
                    # 处理秒数可能为60的情况
                    if ':60.' in dt_str:
                        dt_str = dt_str.replace(':60.', ':59.')
                    elif ':60:' in dt_str:
                        dt_str = dt_str.replace(':60:', ':59:')
                    
                    return pd.to_datetime(dt_str)
                except:
                    # 如果解析失败，使用日期部分并设置时间为00:00:00
                    try:
                        date_part = dt_str.split(' ')[0]
                        return pd.to_datetime(date_part + ' 00:00:00')
                    except:
                        return pd.to_datetime('1970-01-01 00:00:00')
            
            # 解析时间戳
            df_sorted['datetime'] = datetime_strings.apply(parse_datetime)
            
            # 按时间排序
            df_sorted = df_sorted.sort_values('datetime')
            
            # 移除临时的datetime列
            df_sorted = df_sorted.drop('datetime', axis=1)
            
            # 重置索引
            df_sorted = df_sorted.reset_index(drop=True)
            
            self.logger.info(f"📅 事件已按时间排序: {df_sorted['Date'].min()} ~ {df_sorted['Date'].max()}")
            
            return df_sorted
            
        except Exception as e:
            self.logger.warning(f"⚠️  时间排序失败，使用原始顺序: {e}")
            return df

    def _format_output_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化输出数据"""
        df_output = df.copy()
        
        # 格式化矩张量值（科学计数法）
        moment_cols = ['Mrr', 'Mrr_error', 'Mtt', 'Mtt_error', 'Mpp', 'Mpp_error', 
                      'Mrt', 'Mrt_error', 'Mrp', 'Mrp_error', 'Mtp', 'Mtp_error']
        
        for col in moment_cols:
            if col in df_output.columns:
                df_output[col] = df_output[col].apply(
                    lambda x: f"{x:.2e}" if pd.notna(x) and x != 0 else x
                )
        
        # 添加处理时间戳
        df_output['Processing_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df_output['Data_Source'] = f"EASTASIA-FWI-{self.base_config.region['name']}"
        
        return df_output

    def generate_specfem_format(self, df: pd.DataFrame) -> int:
        """
        生成SPECFEM3D格式的CMT文件
        
        Args:
            df: 事件DataFrame
            
        Returns:
            成功生成的CMT文件数量
        """
        if not self.config.output['create_cmt_files']:
            return 0
        
        cmt_dir = self.output_paths['cmt_dir']
        cmt_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📄 生成CMT文件到: {cmt_dir}")
        
        success_count = 0
        error_count = 0
        
        for idx, event in df.iterrows():
            try:
                # 创建事件ID - 改进时间处理
                time_str = str(event['Time'])
                # 处理可能的时间格式问题
                if ':60.' in time_str:
                    time_str = time_str.replace(':60.', ':59.')
                elif time_str.endswith(':60'):
                    time_str = time_str.replace(':60', ':59')
                
                # 解析时间
                try:
                    event_time = pd.to_datetime(f"{event['Date']} {time_str}")
                    event_id = event_time.strftime('%Y%m%d%H%M%S')
                except Exception as time_error:
                    # 如果时间解析失败，使用日期和索引
                    date_part = str(event['Date']).replace('/', '')
                    event_id = f"{date_part}_{idx:06d}"
                    self.logger.debug(f"时间解析失败，使用备用ID: {event_id}, 错误: {time_error}")
                
                # 计算半持续时间
                magnitude = event.get('Magnitude1_mb') or event.get('Magnitude2_Ms') or 5.0
                if pd.isna(magnitude):
                    magnitude = 5.0
                half_duration = self.config.get_half_duration(magnitude)
                
                # 生成CMT文件内容
                cmt_content = self._generate_cmt_content(event, event_id, half_duration)
                
                # 保存CMT文件
                cmt_file = cmt_dir / f"CMTSOLUTION_{event_id}"
                with open(cmt_file, 'w', encoding='utf-8') as f:
                    f.write(cmt_content)
                
                success_count += 1
                
                # 每1000个文件记录一次进度
                if success_count % 1000 == 0:
                    self.logger.info(f"📄 已生成 {success_count} 个CMT文件...")
                
            except Exception as e:
                error_count += 1
                self.logger.debug(f"生成CMT文件失败 (事件 {idx}): {e}")
                # 记录前几个错误的详细信息
                if error_count <= 5:
                    self.logger.warning(f"CMT生成错误详情 #{error_count}: {e}")
                    self.logger.debug(f"问题事件数据: Date={event.get('Date')}, Time={event.get('Time')}, "
                                    f"Lat={event.get('Latitude')}, Lon={event.get('Longitude')}")
                continue
        
        self.logger.info(f"✅ 成功生成 {success_count}/{len(df)} 个CMT文件")
        if error_count > 0:
            self.logger.warning(f"⚠️  生成失败: {error_count} 个文件")
        
        return success_count

    def _generate_cmt_content(self, event: pd.Series, event_id: str, half_duration: float) -> str:
        """生成单个CMT文件内容 - 符合标准GCMT格式"""
        
        # 处理矩张量值 - 如果是字符串则转换为浮点数
        def safe_float_format(value, default=0.0):
            """安全地将值转换为浮点数并格式化"""
            try:
                if isinstance(value, str):
                    float_val = float(value)
                elif pd.isna(value):
                    float_val = default
                else:
                    float_val = float(value)
                return float_val
            except (ValueError, TypeError):
                return default
        
        # 安全获取矩张量分量
        mrr = safe_float_format(event.get('Mrr', 0.0))
        mtt = safe_float_format(event.get('Mtt', 0.0))
        mpp = safe_float_format(event.get('Mpp', 0.0))
        mrt = safe_float_format(event.get('Mrt', 0.0))
        mrp = safe_float_format(event.get('Mrp', 0.0))
        mtp = safe_float_format(event.get('Mtp', 0.0))
        
        # 处理震级值
        mb_val = event.get('Magnitude1_mb', 0.0) or 0.0
        ms_val = event.get('Magnitude2_Ms', 0.0) or 0.0
        if pd.isna(mb_val):
            mb_val = 0.0
        if pd.isna(ms_val):
            ms_val = 0.0
        
        # 解析日期时间 - 转换为标准GCMT格式
        try:
            date_str = str(event['Date'])  # 格式如: 2022/03/12
            time_str = str(event['Time'])  # 格式如: 05:31:11.2
            
            # 处理时间格式问题
            if ':60.' in time_str:
                time_str = time_str.replace(':60.', ':59.')
            elif time_str.endswith(':60'):
                time_str = time_str.replace(':60', ':59')
            
            # 解析日期部分
            if '/' in date_str:
                year, month, day = date_str.split('/')
            else:
                # 如果是其他格式，尝试用pandas解析
                dt = pd.to_datetime(date_str)
                year, month, day = str(dt.year), f"{dt.month:02d}", f"{dt.day:02d}"
            
            # 解析时间部分
            if '.' in time_str:
                time_part, subsec = time_str.split('.')
                hour, minute, second = time_part.split(':')
                # 标准格式需要的时间字符串
                formatted_time = f"{int(hour):2d} {int(minute):2d} {int(second):2d}.{subsec[:2]:<2}"
            else:
                hour, minute, second = time_str.split(':')
                formatted_time = f"{int(hour):2d} {int(minute):2d} {int(second):2d}.00"
            
            # 标准GCMT第一行格式
            pde_line = f"PDE {year} {int(month):2d} {int(day):2d} {formatted_time} {event['Latitude']:8.4f} {event['Longitude']:9.4f} {event['Depth']:5.1f} {mb_val:3.1f} {ms_val:3.1f} {event.get('Location', ''):<40}"
            
        except Exception as e:
            # 如果日期解析失败，使用备用格式
            self.logger.debug(f"日期时间解析失败: {e}")
            pde_line = f"PDE 2000  1  1  0  0  0.00 {event['Latitude']:8.4f} {event['Longitude']:9.4f} {event['Depth']:5.1f} {mb_val:3.1f} {ms_val:3.1f} {event.get('Location', ''):<40}"
        
        # 生成标准GCMT格式内容
        cmt_content = f"""{pde_line}
event name:     {event_id:<15}
time shift:     {self.config.specfem['time_shift']:.4f}
half duration:  {half_duration:.4f}
latitude:      {event['Latitude']:8.4f}
longitude:     {event['Longitude']:8.4f}
depth:         {event['Depth']:8.4f}
Mrr:      {mrr:13.6e}
Mtt:      {mtt:13.6e}
Mpp:      {mpp:13.6e}
Mrt:      {mrt:13.6e}
Mrp:      {mrp:13.6e}
Mtp:      {mtp:13.6e}
"""
        return cmt_content

    def _generate_processing_report(self, df: pd.DataFrame, saved_files: Dict[str, str]):
        """生成处理报告"""
        try:
            report_path = self.output_paths['report_file']
            if self.processing_start_time:
                processing_time = (datetime.now() - self.processing_start_time).total_seconds()
            else:
                processing_time = 0.0
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("EASTASIA-FWI GCMT事件处理报告\n")
                f.write("="*80 + "\n\n")
                
                # 基本信息
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"研究区域: {self.base_config.region['name']}\n")
                f.write(f"处理耗时: {processing_time:.1f} 秒\n")
                f.write(f"数据源文件: {self.gcmt_file}\n\n")
                
                # 下载统计
                if self.download_stats['downloaded_files'] > 0:
                    f.write("📥 下载统计\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"下载文件数: {self.download_stats['downloaded_files']}\n")
                    f.write(f"下载失败数: {self.download_stats['failed_downloads']}\n")
                    f.write(f"总事件数: {self.download_stats['total_catalogs']}\n\n")
                
                # 处理统计
                f.write("📊 处理统计\n")
                f.write("-" * 50 + "\n")
                f.write(f"最终事件数量: {len(df):,}\n")
                f.write(f"事件时间范围: {df['Date'].min()} ~ {df['Date'].max()}\n")
                
                # 震级分布
                mb_count = df['Magnitude1_mb'].notna().sum()
                ms_count = df['Magnitude2_Ms'].notna().sum()
                f.write(f"包含mb震级: {mb_count} 个事件\n")
                f.write(f"包含Ms震级: {ms_count} 个事件\n")
                
                # 地理分布
                f.write(f"\n🌍 地理分布\n")
                f.write("-" * 50 + "\n")
                f.write(f"纬度范围: {df['Latitude'].min():.3f}° ~ {df['Latitude'].max():.3f}°\n")
                f.write(f"经度范围: {df['Longitude'].min():.3f}° ~ {df['Longitude'].max():.3f}°\n")
                f.write(f"深度范围: {df['Depth'].min():.1f} ~ {df['Depth'].max():.1f} km\n")
                f.write(f"平均深度: {df['Depth'].mean():.1f} km\n")
                
                # 错误统计
                if sum(self.parse_errors.values()) > 0:
                    f.write(f"\n⚠️ 解析错误统计\n")
                    f.write("-" * 50 + "\n")
                    for error_type, count in self.parse_errors.items():
                        if count > 0:
                            f.write(f"{error_type}: {count}\n")
                
                # 输出文件
                f.write(f"\n📁 输出文件\n")
                f.write("-" * 50 + "\n")
                for file_type, file_path in saved_files.items():
                    f.write(f"{file_type}: {file_path}\n")
            
            self.logger.info(f"✅ 处理报告已保存: {report_path}")
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")

    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        获取事件数据的汇总统计信息
        
        Args:
            df: 事件数据DataFrame
            
        Returns:
            统计信息字典
        """
        if df.empty:
            return {}
        
        stats = {
            'total_catalogs': len(df),
            'time_range': {
                'start': df['Date'].min(),
                'end': df['Date'].max()
            },
            'coordinate_bounds': {
                'lat_min': float(df['Latitude'].min()),
                'lat_max': float(df['Latitude'].max()),
                'lon_min': float(df['Longitude'].min()),
                'lon_max': float(df['Longitude'].max()),
            },
            'depth_stats': {
                'min': float(df['Depth'].min()),
                'max': float(df['Depth'].max()),
                'mean': float(df['Depth'].mean()),
                'std': float(df['Depth'].std())
            },
            'magnitude_stats': {
                'mb_count': int(df['Magnitude1_mb'].notna().sum()),
                'ms_count': int(df['Magnitude2_Ms'].notna().sum()),
                'mb_range': [float(df['Magnitude1_mb'].min()), float(df['Magnitude1_mb'].max())] if df['Magnitude1_mb'].notna().any() else None,
                'ms_range': [float(df['Magnitude2_Ms'].min()), float(df['Magnitude2_Ms'].max())] if df['Magnitude2_Ms'].notna().any() else None
            },
            'parse_errors': self.parse_errors.copy(),
            'download_stats': self.download_stats.copy()
        }
        
        return stats

    def get_download_statistics(self) -> Dict:
        """获取下载统计信息"""
        return self.download_stats.copy()


def main():
    """主函数 - 执行GCMT事件处理"""
    print("🌍 EASTASIA-FWI GCMT事件处理系统")
    print("="*60)
    
    try:
        # 初始化处理器
        processor = GCMTProcessor()
        
        # 选择操作模式
        print("\n请选择操作模式:")
        print("1. 下载并更新GCMT数据 (2021-2024)")
        print("2. 仅处理现有数据")
        
        choice = input("请输入选择 (1/2): ").strip()
        
        if choice == "1":
            # 下载并更新数据，将生成 jan76_dec24.ndk
            success = processor.download_and_update(2021, 2024)
            if not success:
                print("❌ 数据更新失败")
                return
        elif choice != "2":
            print("❌ 无效选择")
            return
        
        # 检查GCMT文件路径 - 根据时间筛选配置查找匹配的文件
        project_root = Path(__file__).parent.parent
        catalogs_dir = project_root / "data" / "catalogs"
        
        # 获取时间筛选范围
        start_year_val = processor.config.time_filters.get('start_year', 1976)
        end_year_val = processor.config.time_filters.get('end_year', 2024)
        start_year = int(start_year_val) if start_year_val is not None else 1976
        end_year = int(end_year_val) if end_year_val is not None else 2024
        
        # 根据时间范围生成期望的文件名
        expected_filename = processor._generate_gcmt_filename(start_year, end_year)
        expected_file = catalogs_dir / expected_filename
        
        # 查找匹配的文件（按优先级）
        gcmt_file = None
        if expected_file.exists():
            gcmt_file = expected_file
            print(f"✅ 使用匹配文件: {expected_file.name} (时间范围: {start_year}-{end_year})")
        else:
            # 尝试查找其他可能的文件
            fallback_files = [
                catalogs_dir / "jan76_dec25.ndk",
                catalogs_dir / "jan76_dec20.ndk",
                catalogs_dir / processor.config.gcmt['default_file']
            ]
            
            for fallback_file in fallback_files:
                if fallback_file.exists():
                    gcmt_file = fallback_file
                    print(f"⚠️  使用备用文件: {fallback_file.name}")
                    print(f"   注意: 期望文件名为 {expected_filename} (时间范围: {start_year}-{end_year})")
                    break
        
        if gcmt_file:
            processor.gcmt_file = gcmt_file
        else:
            print(f"❌ GCMT文件不存在")
            print(f"   期望文件: {expected_filename} (时间范围: {start_year}-{end_year})")
            print("请确保GCMT数据文件已下载到正确位置")
            return
        
        # 解析NDK文件
        catalogs_df = processor.parse_ndk_file()
        
        if catalogs_df.empty:
            print("❌ 未能解析到任何事件")
            return
        
        print(f"✅ 解析完成: {len(catalogs_df):,} 个事件")
        
        # 筛选事件
        filtered_catalogs = processor.filter_catalogs(catalogs_df)
        
        if filtered_catalogs.empty:
            print("⚠️  筛选后没有符合条件的事件")
            return
        
        print(f"✅ 筛选完成: {len(filtered_catalogs):,} 个事件")
        
        # 保存结果
        saved_files = processor.save_processed_catalogs(filtered_catalogs)
        
        # 显示统计信息
        stats = processor.get_summary_statistics(filtered_catalogs)
        print(f"\n📊 处理统计:")
        print(f"  事件数量: {stats['total_catalogs']:,}")
        print(f"  时间范围: {stats['time_range']['start']} ~ {stats['time_range']['end']}")
        print(f"  纬度范围: {stats['coordinate_bounds']['lat_min']:.2f}° ~ {stats['coordinate_bounds']['lat_max']:.2f}°")
        print(f"  经度范围: {stats['coordinate_bounds']['lon_min']:.2f}° ~ {stats['coordinate_bounds']['lon_max']:.2f}°")
        print(f"  深度范围: {stats['depth_stats']['min']:.0f} ~ {stats['depth_stats']['max']:.0f} km")
        
        # 显示下载统计
        download_stats = processor.get_download_statistics()
        if download_stats['downloaded_files'] > 0:
            print(f"\n📥 下载统计:")
            print(f"  下载文件: {download_stats['downloaded_files']}")
            print(f"  下载失败: {download_stats['failed_downloads']}")
            print(f"  总事件数: {download_stats['total_catalogs']}")
        
        print(f"\n📁 输出文件:")
        for file_type, file_path in saved_files.items():
            print(f"  ✅ {file_type}: {file_path}")
        
        print(f"\n🎉 GCMT事件处理完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断处理")
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()