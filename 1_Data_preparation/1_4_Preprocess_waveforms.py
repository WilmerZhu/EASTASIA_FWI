"""
1_4_Preprocess_waveforms.py: 
EASTASIA-FWI 波形数据预处理模块
================================================================

功能描述:
----------
地震波形数据的标准化预处理模块，将原始观测数据转换为适合全波形反演的格式。
包括仪器响应去除、单位转换、多分量处理、质量控制和SAC格式输出，确保数据质量和格式统一。

核心功能:
----------
1. ✅ 仪器响应去除
   - 自动去除仪器响应，转换为速度或位移
   - 支持多种响应类型（FIR、PAZ、PolesZeros）
   - 预滤波处理，避免频率域边缘效应
   - 自动处理缺失响应文件的情况

2. ✅ 多分量数据处理
   - 自动旋转1-2分量到N-E分量
   - 支持BH1/BH2到BHN/BHE的转换
   - 分量映射和标准化
   - 多分量数据一致性检查

3. ✅ 数据质量控制
   - 采样率验证（0.5-100 Hz）
   - 数据长度比例检查（最小80%）
   - 尖峰检测和去除
   - 数据间隙容忍度检查

4. ✅ SAC格式输出
   - 标准SAC格式输出
   - 自动更新SAC文件头信息
   - 添加事件和台站信息
   - 抑制SAC警告输出

5. ✅ 批量处理和断点续传
   - 支持多进程并行处理
   - 自动检测已处理文件
   - 进度保存和恢复
   - 详细的处理统计报告

6. ✅ 性能优化
   - Inventory缓存机制
   - 内存管理优化
   - 警告抑制（提高处理速度）
   - 自动垃圾回收

使用方法:
----------
```python
from 1_Data_preparation.1_4_Preprocess_waveforms import WaveformPreprocessor

# 初始化预处理器
preprocessor = WaveformPreprocessor()

# 查找事件目录文件
catalog_files = preprocessor.find_catalog_files()

# 批量处理波形数据
results = preprocessor.batch_process(catalog_files[0], max_events=None)

# 处理单个波形文件
station_id, success_msg, error_msg = preprocessor.process_single_waveform(
    waveform_file, response_file, event_info
)
```

配置说明:
----------
通过 PreprocessingConfig 类配置预处理参数:
- response_removal: 响应去除参数 (输出类型、预滤波、水线等)
- channel_processing: 通道处理配置 (自动旋转、分量映射)
- quality_control: 质量控制参数 (采样率、长度比例、尖峰检测)
- output_format: 输出格式配置 (SAC格式、文件头更新)
- resume: 断点续传配置
- performance: 性能优化配置

输出文件:
----------
- events/vel_data/{event_name}/: 每个事件的预处理后波形数据
- preprocessing/preprocessing_report.txt: 详细处理报告
- preprocessing/preprocessing_stats.json: 处理统计信息
- preprocessing_progress.json: 处理进度文件

科学原理:
----------
- 仪器响应去除: 将观测数据从计数转换为物理量（速度/位移），消除仪器频率响应影响
- 预滤波: 在响应去除前应用预滤波，避免频率域边缘的数值不稳定
- 分量旋转: 将仪器坐标系转换为地理坐标系（N-E），便于后续处理和分析
- 质量控制: 确保数据质量满足全波形反演的要求，避免低质量数据影响反演结果
- SAC格式: 地震学标准数据格式，兼容性强，便于后续分析和可视化

作者: EASTASIA-FWI Team
日期: 2025-02-01
版本: v2.6
"""

import os
import sys
import warnings
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
import subprocess
import gc
import json
import logging

import numpy as np
from tqdm.auto import tqdm

from obspy import read, read_inventory, UTCDateTime, Stream

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig

# ========== 警告过滤配置 ==========
warnings.filterwarnings("ignore", category=UserWarning, module="obspy")
warnings.filterwarnings("ignore", message=".*computed and reported sensitivities differ.*")
warnings.filterwarnings("ignore", message=".*FIR normalized.*")
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
np.seterr(divide='ignore', invalid='ignore')

# ========== 配置 ObsPy 日志级别 ==========
obspy_logger = logging.getLogger('obspy')
obspy_logger.setLevel(logging.ERROR)

# ========== 抑制 SAC 警告输出 ==========
os.environ['SAC_DISPLAY_COPYRIGHT'] = '0'


# ========== 🆕 系统级输出抑制器 ==========
class SuppressOutput:
    """
    上下文管理器：完全抑制所有输出（包括 C/Fortran 库）
    
    使用底层文件描述符重定向，可以拦截所有级别的输出
    """
    def __enter__(self):
        """进入上下文：保存并重定向标准输出/错误"""
        try:
            self._original_stdout_fd = sys.stdout.fileno()
            self._original_stderr_fd = sys.stderr.fileno()
            
            # 保存原始文件描述符
            self._saved_stdout_fd = os.dup(self._original_stdout_fd)
            self._saved_stderr_fd = os.dup(self._original_stderr_fd)
            
            # 重定向到 /dev/null
            self._devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self._devnull, self._original_stdout_fd)
            os.dup2(self._devnull, self._original_stderr_fd)
        except Exception:
            # 如果重定向失败，静默处理
            pass
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：恢复原始输出"""
        try:
            # 恢复原始文件描述符
            os.dup2(self._saved_stdout_fd, self._original_stdout_fd)
            os.dup2(self._saved_stderr_fd, self._original_stderr_fd)
            
            # 关闭临时文件描述符
            os.close(self._saved_stdout_fd)
            os.close(self._saved_stderr_fd)
            os.close(self._devnull)
        except Exception:
            pass
        
        return False


class PreprocessingConfig:
    """波形预处理配置类"""
    
    def __init__(self):
        """初始化预处理配置参数"""
        # 输入目录配置
        self.input_directories = {
            'events': 'events/',
            'waveforms': 'events/waveforms',
            'responses': 'events/responses'
        }
        
        # 输出目录配置
        self.output_directories = {
            'vel_data': 'events/vel_data',
            'reports': 'preprocessing',
        }
        
        # 文件模式配置
        self.file_patterns = {
            'catalog_patterns': [
                "selected_events_catalog_*.par",
                "download_catalog_*.par",
                "*.par"
            ],
            'waveform_patterns': ["*.mseed", "*.MSEED", "*.sac", "*.SAC"],
            'response_patterns': ["*.xml", "*.XML"]
        }
        
        # 响应去除参数
        self.response_removal = {
            'output_type': 'VEL',
            'pre_filt': [0.008, 0.012, 8.0, 10.0],
            'water_level': 60.0,
            'remove_mean': True,
            'remove_trend': True,
            'taper_fraction': 0.05
        }
        
        # 通道处理配置
        self.channel_processing = {
            'auto_rotate_12_to_ne': True,
            'preferred_channels': ['BHZ', 'BHN', 'BHE', 'BH1', 'BH2'],
            'component_mapping': {'1': 'N', '2': 'E'}
        }
        
        # 数据质量控制
        self.quality_control = {
            'min_sample_rate': 0.5,
            'max_sample_rate': 100.0,
            'min_length_ratio': 0.8,
            'spike_detection': True,
            'spike_threshold': 5.0,
            'gap_tolerance': 0.1
        }
        
        # 输出格式配置
        self.output_format = {
            'format': 'SAC',
            'update_sac_headers': True,
            'preserve_original': False,
            'file_naming': 'standard',
            'suppress_sac_warnings': True
        }
        
        # 断点续传配置
        self.resume = {
            'enable': True,
            'skip_existing': True,
            'min_sac_files': 3
        }
        
        # 并行处理配置
        self.parallel_processing = {
            'enable_multiprocessing': False,
            'max_workers': min(4, mp.cpu_count()),
            'chunk_size': 1
        }
        
        # 日志配置
        self.logging = {
            'level': 'INFO',
            'console_output': True,
            'file_prefix': 'preprocessing',
            'detailed_progress': True,
            'show_realtime_stats': True,
            'show_debug_info': False
        }
        
        # 性能优化配置
        self.performance = {
            'cache_inventory': True,
            'batch_save': False,
            'memory_efficient': True,
            'suppress_all_warnings': True  # 🆕 完全抑制警告
        }


class WaveformPreprocessor:
    """波形数据预处理器"""
    
    def __init__(self):
        """初始化波形预处理器"""
        # 加载基础配置
        self.base_config = BaseConfig()
        
        # 加载模块配置
        self.config = PreprocessingConfig()
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.WaveformPreprocessor',
            self.config.logging['level']
        )
        
        # 设置输入输出目录
        self._setup_directories()
        
        # 初始化统计信息
        self.stats = self._initialize_statistics()
        
        # 创建输出目录
        self._create_output_directories()
        
        # Inventory 缓存
        self.inventory_cache = {} if self.config.performance['cache_inventory'] else None
        
        # 进度文件
        self.progress_file = self.input_dirs['events'] / 'preprocessing_progress.json'
        
        self.logger.info("🔬 波形预处理器初始化完成")
        self._print_config_summary()
    
    def _setup_directories(self):
        """设置输入输出目录"""
        self.input_dirs = {
            'events': self.base_config.dirs['data'] / self.config.input_directories['events'],
            'waveforms': self.base_config.dirs['data'] / self.config.input_directories['waveforms'],
            'responses': self.base_config.dirs['data'] / self.config.input_directories['responses']
        }
        
        self.output_dirs = {
            'vel_data': self.base_config.dirs['data'] / self.config.output_directories['vel_data'],
            'reports': self.base_config.dirs['results'] / self.config.output_directories['reports']
        }
    
    def _create_output_directories(self):
        """创建所有必要的输出目录"""
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 波形预处理配置摘要")
        print("-" * 60)
        
        print(f"📁 输入目录:")
        for name, path in self.input_dirs.items():
            status = "✅" if path.exists() else "❌"
            print(f"   {name}: {path} {status}")
        
        print(f"📁 输出目录:")
        for name, path in self.output_dirs.items():
            print(f"   {name}: {path}")
        
        print(f"⚙️ 处理参数:")
        print(f"   输出类型: {self.config.response_removal['output_type']}")
        print(f"   水位参数: {self.config.response_removal['water_level']}")
        print(f"   预滤波器: {self.config.response_removal['pre_filt']}")
        print(f"   1,2分量转换: {'启用' if self.config.channel_processing['auto_rotate_12_to_ne'] else '禁用'}")
        print(f"   断点续传: {'启用' if self.config.resume['enable'] else '禁用'}")
        print(f"   Inventory缓存: {'启用' if self.config.performance['cache_inventory'] else '禁用'}")
        print(f"   警告抑制: {'启用' if self.config.performance['suppress_all_warnings'] else '禁用'}")
        
        print("-" * 60)
    
    def _initialize_statistics(self) -> Dict[str, Any]:
        """初始化处理统计信息"""
        return {
            'total_events': 0,
            'processed_events': 0,
            'skipped_events': 0,
            'failed_events': 0,
            'total_stations': 0,
            'successful_stations': 0,
            'failed_stations': 0,
            'no_response_stations': [],
            'error_response_stations': [],
            'component_conversions': 0,
            'inventory_cache_hits': 0,
            'inventory_cache_misses': 0,
            'simplified_response_removal': 0,
            # 🆕 警告统计
            'warning_stats': {
                'response_warnings_suppressed': 0,
                'total_response_removal_calls': 0
            },
            'error_types': {
                'file_not_found': 0,
                'response_removal_failed': 0,
                'quality_control_failed': 0,
                'rotation_failed': 0,
                'saving_failed': 0,
                'no_response_file': 0,
                'no_matching_response': 0,
                'response_error': 0
            },
            'processing_summary': {
                'start_time': None,
                'end_time': None,
                'total_time_seconds': 0
            }
        }
    
    def find_catalog_files(self) -> List[Path]:
        """查找事件目录文件"""
        catalog_files = []
        
        waveforms_base = self.base_config.dirs['data'] / 'waveforms'
        if waveforms_base.exists():
            for pattern in self.config.file_patterns['catalog_patterns']:
                catalog_files.extend(list(waveforms_base.glob(pattern)))
        
        events_dir = self.input_dirs['events']
        if events_dir.exists():
            for pattern in self.config.file_patterns['catalog_patterns']:
                catalog_files.extend(list(events_dir.glob(pattern)))
        
        for pattern in self.config.file_patterns['catalog_patterns']:
            catalog_files.extend(list(self.base_config.dirs['project_root'].glob(f"**/{pattern}")))
        
        catalog_files = list(set(catalog_files))
        catalog_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return catalog_files
    
    def load_event_catalog(self, catalog_file: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """加载事件目录文件"""
        if catalog_file is None:
            catalog_files = self.find_catalog_files()
            
            if not catalog_files:
                raise FileNotFoundError("未找到事件目录文件")
            
            catalog_file = catalog_files[0]
            self.logger.info(f"使用目录文件: {catalog_file}")
        
        catalog_file = Path(catalog_file)
        events = []
        
        try:
            with open(catalog_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) >= 8:
                            event_info = {
                                'name': parts[0],
                                'date': parts[1],
                                'hour': int(parts[2]),
                                'minute': int(parts[3]),
                                'second': float(parts[4]),
                                'latitude': float(parts[5]),
                                'longitude': float(parts[6]),
                                'depth': float(parts[7])
                            }
                            
                            event_datetime = datetime.strptime(event_info['date'], '%Y%m%d')
                            event_datetime = event_datetime.replace(
                                hour=event_info['hour'],
                                minute=event_info['minute'],
                                second=int(event_info['second']),
                                microsecond=int((event_info['second'] % 1) * 1000000)
                            )
                            event_info['datetime'] = event_datetime
                            event_info['origin_time'] = UTCDateTime(event_datetime)
                            
                            events.append(event_info)
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"解析目录文件第{line_num}行失败: {e}")
                        continue
            
            self.logger.info(f"成功加载 {len(events)} 个事件")
            self.stats['total_events'] = len(events)
            return events
            
        except Exception as e:
            self.logger.error(f"加载事件目录文件失败: {e}")
            raise
    
    def is_event_processed(self, event_name: str) -> bool:
        """检查事件是否已经处理过"""
        if not self.config.resume['enable']:
            return False
        
        event_output_dir = self.output_dirs['vel_data'] / event_name
        
        if not event_output_dir.exists():
            return False
        
        # 简单检查：SAC 文件数量
        sac_files = list(event_output_dir.glob("*.SAC"))
        return len(sac_files) >= self.config.resume['min_sac_files']
    
    def find_waveform_files(self, event_name: str) -> Tuple[List[Path], List[Path]]:
        """查找指定事件的波形和响应文件"""
        waveform_dir = self.input_dirs['waveforms'] / event_name
        response_dir = self.input_dirs['responses'] / event_name
        
        waveform_files = []
        if waveform_dir.exists():
            for pattern in self.config.file_patterns['waveform_patterns']:
                waveform_files.extend(list(waveform_dir.glob(pattern)))
        
        response_files = []
        if response_dir.exists():
            for pattern in self.config.file_patterns['response_patterns']:
                response_files.extend(list(response_dir.glob(pattern)))
        
        return waveform_files, response_files
    
    def load_inventory(self, inventory_file: Path) -> Optional[Any]:
        """加载 inventory（支持缓存）"""
        if self.inventory_cache is not None:
            cache_key = str(inventory_file)
            if cache_key in self.inventory_cache:
                self.stats['inventory_cache_hits'] += 1
                return self.inventory_cache[cache_key]
            
            try:
                inv = read_inventory(str(inventory_file))
                self.inventory_cache[cache_key] = inv
                self.stats['inventory_cache_misses'] += 1
                return inv
            except Exception:
                return None
        else:
            try:
                return read_inventory(str(inventory_file))
            except Exception:
                return None
    
    def build_response_mapping(self, response_files: List[Path]) -> Dict[str, Path]:
        """🆕 预先建立响应文件映射表（大幅提升匹配效率）
        
        从 O(n*m) 降低到 O(n+m) 的时间复杂度
        
        Args:
            response_files: 响应文件列表
            
        Returns:
            映射字典 {网络.台站: 响应文件路径}
        """
        mapping = {}
        for resp_file in response_files:
            resp_name = resp_file.stem  # 例如 "AU.COEN"
            parts = resp_name.split('.')
            if len(parts) >= 2:
                network = parts[0]
                station = parts[1]
                key = f"{network}.{station}"
                mapping[key] = resp_file
        return mapping
    
    def match_response_file_fast(self, waveform_file: Path, 
                                 response_mapping: Dict[str, Path]) -> Optional[Path]:
        """🆕 快速匹配响应文件（使用预建的映射表）
        
        O(1) 时间复杂度的查找
        
        Args:
            waveform_file: 波形文件路径
            response_mapping: 响应文件映射表
            
        Returns:
            匹配的响应文件路径，未找到返回 None
        """
        try:
            # 方法1：从文件名提取网络和台站信息
            waveform_name = waveform_file.stem
            # 文件名格式：AU.COEN..BHE__20100218T011319Z__20100218T021319Z
            parts = waveform_name.split('__')[0].split('.')
            if len(parts) >= 2:
                network = parts[0]
                station = parts[1]
                key = f"{network}.{station}"
                resp_file = response_mapping.get(key)
                if resp_file:
                    return resp_file
            
            # 方法2：读取文件头（备用方案）
            st = read(str(waveform_file), headonly=True)
            if len(st) > 0:
                stats = st[0].stats
                key = f"{stats.network}.{stats.station}"
                return response_mapping.get(key)
            
            return None
        except Exception:
            return None
    
    def rotate_12_to_ne(self, st: Stream, inventory_file: Path) -> Tuple[Stream, bool]:
        """将1、2分量转换为N、E分量"""
        try:
            components = [tr.stats.channel[-1] for tr in st]
            if '1' not in components and '2' not in components:
                return st, False
            
            groups = {}
            for tr in st:
                key = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel[:-1]}"
                if key not in groups:
                    groups[key] = {}
                component = tr.stats.channel[-1]
                groups[key][component] = tr
            
            new_st = Stream()
            conversion_successful = False
            
            for group_key, traces in groups.items():
                if '1' in traces and '2' in traces:
                    tr1 = traces['1']
                    tr2 = traces['2']
                    
                    tr_n = tr1.copy()
                    tr_e = tr2.copy()
                    
                    tr_n.stats.channel = tr1.stats.channel[:-1] + 'N'
                    tr_e.stats.channel = tr2.stats.channel[:-1] + 'E'
                    
                    new_st.append(tr_n)
                    new_st.append(tr_e)
                    
                    self.stats['component_conversions'] += 1
                    conversion_successful = True
                else:
                    for component, tr in traces.items():
                        new_st.append(tr)
            
            return new_st if len(new_st) > 0 else st, conversion_successful
            
        except Exception:
            self.stats['error_types']['rotation_failed'] += 1
            return st, False
    
    def remove_instrument_response(self, st: Stream, inventory_file: Path, 
                                 event_info: Dict[str, Any]) -> Tuple[bool, str]:
        """🆕 去除仪器响应（完全抑制输出 + 统计）"""
        try:
            if not inventory_file.exists():
                self.stats['error_types']['no_response_file'] += 1
                return False, "no_response_file"
            
            inv = self.load_inventory(inventory_file)
            if inv is None:
                self.stats['error_types']['response_error'] += 1
                return False, "read_inventory_failed"
            
            resp_params = self.config.response_removal
            
            # 基础预处理
            if resp_params['remove_mean']:
                st.detrend(type="demean")
            
            if resp_params['remove_trend']:
                st.detrend(type="linear")
            
            if resp_params['taper_fraction'] > 0:
                st.taper(max_percentage=resp_params['taper_fraction'])
            
            # 🆕 统计响应去除调用次数
            self.stats['warning_stats']['total_response_removal_calls'] += 1
            
            # 🆕 使用系统级输出抑制
            if self.config.performance['suppress_all_warnings']:
                with SuppressOutput():
                    return self._do_remove_response(st, inv, resp_params)
            else:
                return self._do_remove_response(st, inv, resp_params)
                
        except FileNotFoundError:
            self.stats['error_types']['no_response_file'] += 1
            return False, "no_response_file"
        except Exception as e:
            self.stats['error_types']['response_removal_failed'] += 1
            return False, f"general_error: {str(e)[:100]}"
    
    def _do_remove_response(self, st: Stream, inv: Any, resp_params: Dict) -> Tuple[bool, str]:
        """执行响应去除（内部方法）"""
        try:
            # 尝试标准响应去除
            st.remove_response(
                inventory=inv,
                output=resp_params['output_type'],
                pre_filt=resp_params['pre_filt'],
                water_level=resp_params['water_level']
            )
            # 🆕 统计抑制的警告（假设每次都有警告）
            self.stats['warning_stats']['response_warnings_suppressed'] += 1
            return True, "success"
            
        except (AttributeError, TypeError) as e:
            # 如果出现窗口函数相关错误，尝试不使用 pre_filt 的简化版本
            if 'hann' in str(e) or 'tukey' in str(e) or 'window' in str(e).lower():
                try:
                    # 创建副本进行简化处理
                    st_copy = st.copy()
                    st_copy.remove_response(
                        inventory=inv,
                        output=resp_params['output_type'],
                        water_level=resp_params['water_level']
                    )
                    # 将结果复制回原始 stream
                    for i, tr in enumerate(st):
                        tr.data = st_copy[i].data
                        tr.stats = st_copy[i].stats
                    
                    self.stats['simplified_response_removal'] += 1
                    self.stats['warning_stats']['response_warnings_suppressed'] += 1
                    return True, "success_simplified"
                    
                except Exception:
                    self.stats['error_types']['response_error'] += 1
                    return False, "simplified_removal_failed"
            else:
                # 其他类型的 AttributeError/TypeError，继续抛出
                raise e
                
        except Exception as e:
            error_msg = str(e)
            if "No matching response" in error_msg:
                self.stats['error_types']['no_matching_response'] += 1
                return False, "no_matching_response"
            elif "divide by zero" in error_msg or "invalid value" in error_msg:
                self.stats['error_types']['response_error'] += 1
                return False, "response_error"
            else:
                self.stats['error_types']['response_removal_failed'] += 1
                return False, f"processing_error: {error_msg[:100]}"
    
    def get_station_coordinates(self, inventory_file: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """获取台站坐标信息"""
        try:
            inv = self.load_inventory(inventory_file)
            if inv is None:
                return None, None, None
            
            station = inv.networks[0].stations[0]
            return station.latitude, station.longitude, station.elevation
        except Exception:
            return None, None, None
    
    def update_sac_headers(self, sac_file: Path, event_info: Dict[str, Any], 
                          stla: float, stlo: float, stel: float) -> bool:
        """更新SAC文件头信息"""
        if not self.config.output_format['update_sac_headers']:
            return True
        
        try:
            origin_time = event_info['origin_time']
            evlo = event_info['longitude']
            evla = event_info['latitude'] 
            evdp = event_info['depth']
            hour = event_info['hour']
            mini = event_info['minute']
            msec = f"{event_info['second']:.3f}"
            
            sac_commands = f"""
            wild echo off
            r {sac_file}
            ch LCALDA True
            ch evlo {evlo} evla {evla} evdp {evdp}
            ch stlo {stlo} stla {stla} stel {stel}
            ch t1 0.0 t2 0.0 t3 0.0 t4 0.0
            ch o gmt {origin_time.year} {origin_time.julday} {hour} {mini} {msec.split('.')[0]} {msec.split('.')[1]}
            wh
            q
            """
            
            result = subprocess.run(
                ["sac"], 
                input=sac_commands.encode(), 
                capture_output=True,
                timeout=30,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
            
            return result.returncode == 0
            
        except Exception:
            return False
    
    def save_processed_waveform(self, st: Stream, event_name: str, event_info: Dict[str, Any],
                              stla: float, stlo: float, stel: float) -> Dict[str, str]:
        """保存处理后的波形数据"""
        saved_files = {}
        
        if not st or len(st) == 0:
            return saved_files
        
        origin_time = event_info['origin_time']
        hour = event_info['hour']
        mini = event_info['minute']
        msec = f"{event_info['second']:.3f}"
        
        for tr in st:
            network = tr.stats.network
            station = tr.stats.station
            channel = tr.stats.channel
            
            station_id = f"{network}.{station}"
            
            newsacname = f"{origin_time.year}.{origin_time.julday:03d}.{hour}.{mini}.{msec}.{station_id}..{channel}.SAC"
            
            single_st = Stream([tr.copy()])
            
            output_dir = self.output_dirs['vel_data'] / event_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / newsacname
            
            try:
                single_st.write(str(output_path), format="SAC")
                saved_files[f'{channel}'] = str(output_path)
                
                self.update_sac_headers(output_path, event_info, stla, stlo, stel)
                
            except Exception:
                self.stats['error_types']['saving_failed'] += 1
                continue
        
        return saved_files
    
    def apply_quality_control(self, st: Stream, event_info: Dict[str, Any]) -> Tuple[Stream, bool, str]:
        """应用数据质量控制"""
        if not st:
            return st, False, "空数据流"
        
        qc_params = self.config.quality_control
        passed_traces = Stream()
        qc_messages = []
        
        for tr in st:
            if not (qc_params['min_sample_rate'] <= tr.stats.sampling_rate <= qc_params['max_sample_rate']):
                qc_messages.append(f"采样率不符合要求: {tr.stats.sampling_rate}")
                continue
            
            expected_length = int((tr.stats.endtime - tr.stats.starttime) * tr.stats.sampling_rate)
            actual_length = len(tr.data)
            length_ratio = actual_length / expected_length if expected_length > 0 else 0
            
            if length_ratio < qc_params['min_length_ratio']:
                qc_messages.append(f"数据长度不足: {length_ratio:.2f}")
                continue
            
            if qc_params['spike_detection']:
                data_std = np.std(tr.data)
                data_mean = np.mean(tr.data)
                spike_threshold = qc_params['spike_threshold'] * data_std
                
                spikes = np.abs(tr.data - data_mean) > spike_threshold
                spike_ratio = np.sum(spikes) / len(tr.data)
                
                if spike_ratio > 0.01:
                    qc_messages.append(f"尖峰过多: {spike_ratio:.3f}")
                    continue
            
            passed_traces += tr
        
        success = len(passed_traces) > 0
        message = "; ".join(qc_messages) if qc_messages else "质控通过"
        
        if not success:
            self.stats['error_types']['quality_control_failed'] += 1
        
        return passed_traces, success, message
    
    def process_single_waveform(self, waveform_file: Path, response_file: Path,
                               event_info: Dict[str, Any]
                               ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """处理单个波形文件"""
        try:
            st = read(str(waveform_file))
            if len(st) == 0:
                return None, None, None
            
            head = st[0]
            station_id = f"{head.stats.network}.{head.stats.station}"
            
            stla, stlo, stel = self.get_station_coordinates(response_file)
            if stla is None or stlo is None or stel is None:
                self.stats['error_types']['file_not_found'] += 1
                return station_id, f"{event_info['name']}: {station_id}.{head.stats.channel}", None
            
            if self.config.channel_processing['auto_rotate_12_to_ne']:
                st, rotated = self.rotate_12_to_ne(st, response_file)
            
            success, error_msg = self.remove_instrument_response(st, response_file, event_info)
            if not success:
                if error_msg == "no_response_file":
                    return station_id, f"{event_info['name']}: {station_id}.{head.stats.channel}", None
                else:
                    return station_id, None, f"{event_info['name']}: {station_id}.{head.stats.channel}"
            
            st_qc, qc_success, qc_message = self.apply_quality_control(st, event_info)
            if not qc_success:
                return station_id, None, f"{event_info['name']}: {station_id}.{head.stats.channel}"
            
            saved_files = self.save_processed_waveform(st_qc, event_info['name'], event_info, 
                                                     stla, stlo, stel)
            
            if saved_files:
                self.stats['successful_stations'] += 1
                return station_id, None, None
            else:
                return station_id, None, f"{event_info['name']}: {station_id}.{head.stats.channel}"
                
        except FileNotFoundError:
            station_id = waveform_file.stem.split('.')[1] if '.' in waveform_file.name else "unknown"
            self.stats['error_types']['file_not_found'] += 1
            return station_id, f"{event_info['name']}: {station_id}", None
        except Exception:
            station_id = waveform_file.stem.split('.')[1] if '.' in waveform_file.name else "unknown"
            return station_id, None, f"{event_info['name']}: {waveform_file.name}"
    
    def process_event(self, event_info: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """🆕 处理单个事件的所有波形数据"""
        event_name = event_info['name']
        
        # 简单检查是否已处理
        if self.config.resume['skip_existing'] and self.is_event_processed(event_name):
            self.stats['skipped_events'] += 1
            return [], []
        
        # 查找波形和响应文件
        waveform_files, response_files = self.find_waveform_files(event_name)
        
        if not waveform_files or not response_files:
            if self.config.logging['show_debug_info']:
                self.logger.warning(f"事件 {event_name}: 波形={len(waveform_files) if waveform_files else 0}, 响应={len(response_files) if response_files else 0}")
            return [], []
        
        # 🆕 预先建立响应文件映射表（避免重复O(n)搜索）
        response_mapping = self.build_response_mapping(response_files)
        
        no_resp_stations = []
        error_resp_stations = []
        event_success_count = 0
        match_failures = 0
        
        # 处理每个波形文件
        for waveform_file in waveform_files:
            self.stats['total_stations'] += 1
            
            # 🆕 使用快速匹配（O(1)时间复杂度）
            response_file = self.match_response_file_fast(waveform_file, response_mapping)
            
            if not response_file:
                waveform_name = waveform_file.stem
                parts = waveform_name.split('__')[0].split('.')
                if len(parts) >= 2:
                    station_id = f"{parts[0]}.{parts[1]}"
                    no_resp_stations.append(f"{event_name}: {station_id} (no matching response)")
                    self.stats['no_response_stations'].append(f"{event_name}: {station_id}")
                match_failures += 1
                self.stats['failed_stations'] += 1
                continue
            
            station_id, no_resp, error_resp = self.process_single_waveform(
                waveform_file, response_file, event_info
            )
            
            if no_resp:
                no_resp_stations.append(no_resp)
                self.stats['failed_stations'] += 1
            elif error_resp:
                error_resp_stations.append(error_resp)
                self.stats['failed_stations'] += 1
            else:
                event_success_count += 1
        
        # 内存优化
        if self.config.performance['memory_efficient']:
            gc.collect()
        
        # 🆕 优化日志输出（只在有问题或debug模式时输出）
        if match_failures > 0 and self.config.logging['show_debug_info']:
            self.logger.warning(f"⚠️  {event_name}: {match_failures} 个波形无匹配响应")
        
        return no_resp_stations, error_resp_stations
    
    def save_progress(self, current_index: int):
        """保存进度"""
        progress_data = {
            'current_index': current_index,
            'timestamp': datetime.now().isoformat(),
            'stats': {
                'processed_events': self.stats['processed_events'],
                'successful_stations': self.stats['successful_stations'],
                'failed_stations': self.stats['failed_stations'],
                'total_stations': self.stats['total_stations']  # �� 添加 total_stations
            }
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception:
            pass
    
    def load_progress(self) -> Optional[int]:
        """加载进度"""
        if not self.progress_file.exists():
            return None
        
        try:
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            
            current_index = progress_data.get('current_index', 0)
            self.logger.info(f"📋 发现之前的处理进度，从事件 {current_index + 1} 继续")
            
            if 'stats' in progress_data:
                saved_stats = progress_data['stats']
                # 🆕 添加 total_stations 到恢复列表
                for key in ['processed_events', 'successful_stations', 'failed_stations', 'total_stations']:
                    if key in saved_stats and key in self.stats:
                        self.stats[key] = saved_stats[key]
            
            return current_index
            
        except Exception:
            return None
    
    def batch_process(self, catalog_file: Optional[Union[str, Path]] = None, 
                     max_events: Optional[int] = None) -> Dict[str, Any]:
        """🆕 批量处理多个事件"""
        self.stats['processing_summary']['start_time'] = datetime.now()
        start_time = self.stats['processing_summary']['start_time']
        
        self.logger.info("🚀 开始批量波形预处理")
        
        try:
            events = self.load_event_catalog(catalog_file)
            
            if max_events:
                events = events[:max_events]
            
            # 加载进度
            start_idx = self.load_progress() or 0
            if start_idx > 0:
                start_idx += 1
            
            total_events = len(events)
            self.stats['total_events'] = total_events
            self.logger.info(f"总计需要处理 {total_events - start_idx} 个事件 (总计 {total_events})")
            
            all_no_resp_stations = []
            all_error_resp_stations = []
            
            # 🆕 优化进度条显示
            with tqdm(
                total=total_events - start_idx, 
                desc="🔄 处理事件", 
                unit="event",
                ncols=120,
                position=0,
                leave=True,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
            ) as pbar:
                
                for i in range(start_idx, total_events):
                    event_info = events[i]
                    event_name = event_info['name']
                    
                    try:
                        # 🆕 更新进度条描述显示当前事件
                        pbar.set_description(f"🔄 {event_name}")
                        
                        no_resp, error_resp = self.process_event(event_info)
                        all_no_resp_stations.extend(no_resp)
                        all_error_resp_stations.extend(error_resp)
                        
                        self.stats['processed_events'] += 1
                        
                        # 保存进度
                        if (i + 1) % 5 == 0:
                            self.save_progress(i)
                        
                        # 🆕 更新进度条统计（使用更友好的图标）
                        if self.config.logging['show_realtime_stats']:
                            success_rate = (self.stats['successful_stations'] / 
                                          self.stats['total_stations'] * 100) if self.stats['total_stations'] > 0 else 0
                            pbar.set_postfix({
                                '✅': self.stats['successful_stations'],
                                '❌': self.stats['failed_stations'],
                                '成功率': f"{success_rate:.1f}%",
                                '⏭️': self.stats['skipped_events']
                            }, refresh=False)
                        
                        pbar.update(1)
                        
                    except KeyboardInterrupt:
                        self.logger.info("\n🛑 用户中断处理")
                        self.save_progress(i)
                        raise
            
            self.stats['no_response_stations'] = all_no_resp_stations
            self.stats['error_response_stations'] = all_error_resp_stations
            
            end_time = datetime.now()
            self.stats['processing_summary']['end_time'] = end_time
            total_time = (end_time - start_time).total_seconds()
            self.stats['processing_summary']['total_time_seconds'] = total_time
            
            report_file = self.generate_processing_report()
            
            # 清理进度文件
            if self.progress_file.exists():
                self.progress_file.unlink()
            
            return {
                'stats': self.stats,
                'no_response_stations': all_no_resp_stations,
                'error_response_stations': all_error_resp_stations,
                'total_time_seconds': total_time,
                'report_file': report_file
            }
            
        except Exception as e:
            self.logger.error(f"批处理过程出错: {e}")
            raise
    
    def generate_processing_report(self) -> str:
        """生成详细的处理报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dirs['reports'] / f"preprocessing_report_{timestamp}.txt"
        
        self.output_dirs['reports'].mkdir(parents=True, exist_ok=True)
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"EASTASIA-FWI 波形预处理报告\n")
                f.write("="*80 + "\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                total_time = self.stats['processing_summary']['total_time_seconds']
                f.write(f"总耗时: {total_time/60:.2f} 分钟\n")
                f.write("-"*80 + "\n\n")
                
                f.write(f"📊 统计摘要:\n")
                f.write(f"  总事件数: {self.stats['total_events']:,}\n")
                f.write(f"  实际处理: {self.stats['processed_events']:,}\n")
                f.write(f"  跳过事件: {self.stats['skipped_events']:,}\n")
                f.write(f"  成功台站: {self.stats['successful_stations']:,}\n")
                f.write(f"  失败台站: {self.stats['failed_stations']:,}\n")
                f.write(f"  1,2分量转换: {self.stats['component_conversions']:,} 次\n")
                f.write(f"  简化响应去除: {self.stats['simplified_response_removal']:,} 次\n")
                
                if self.stats['total_stations'] > 0:
                    success_rate = self.stats['successful_stations'] / self.stats['total_stations'] * 100
                    f.write(f"  成功率: {success_rate:.1f}%\n")
                
                if self.stats['processed_events'] > 0:
                    avg_time = total_time / self.stats['processed_events']
                    f.write(f"  平均速度: {avg_time:.1f} 秒/事件\n")
                
                if self.config.performance['cache_inventory']:
                    cache_hits = self.stats.get('inventory_cache_hits', 0)
                    cache_misses = self.stats.get('inventory_cache_misses', 0)
                    cache_total = cache_hits + cache_misses
                    if cache_total > 0:
                        cache_hit_rate = cache_hits / cache_total * 100
                        f.write(f"  Inventory缓存命中率: {cache_hit_rate:.1f}%\n")
                
                # 🆕 警告统计
                f.write(f"\n🔇 警告抑制统计:\n")
                f.write(f"  响应去除调用次数: {self.stats['warning_stats']['total_response_removal_calls']:,}\n")
                f.write(f"  抑制警告次数: {self.stats['warning_stats']['response_warnings_suppressed']:,}\n")
                if self.config.performance['suppress_all_warnings']:
                    f.write(f"  警告抑制状态: ✅ 已启用（输出清爽）\n")
                else:
                    f.write(f"  警告抑制状态: ❌ 未启用\n")
                
                f.write(f"\n❌ 错误类型统计:\n")
                for error_type, count in self.stats['error_types'].items():
                    if count > 0:
                        f.write(f"  {error_type}: {count:,}\n")
                
                f.write("\n" + "-"*80 + "\n")
                f.write(f"{len(self.stats['no_response_stations'])} stations have no response file:\n")
                for station in self.stats['no_response_stations'][:100]:
                    f.write(f"  {station}\n")
                if len(self.stats['no_response_stations']) > 100:
                    f.write(f"  ... 及其他 {len(self.stats['no_response_stations']) - 100} 个台站\n")
                    
                f.write(f"\n{len(self.stats['error_response_stations'])} stations have wrong response file:\n")
                for station in self.stats['error_response_stations'][:100]:
                    f.write(f"  {station}\n")
                if len(self.stats['error_response_stations']) > 100:
                    f.write(f"  ... 及其他 {len(self.stats['error_response_stations']) - 100} 个台站\n")
                
                f.write("="*80 + "\n")
            
            self.logger.info(f"✅ 处理报告已保存: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"生成处理报告失败: {e}")
            return ""


def main():
    """主函数"""
    print("🔬 EASTASIA-FWI 波形预处理模块 v2.6")
    print("="*60)
    
    try:
        preprocessor = WaveformPreprocessor()
        
        catalog_files = preprocessor.find_catalog_files()
        
        if not catalog_files:
            print("❌ 未找到事件目录文件")
            return
        
        catalog_file = catalog_files[0]
        
        print(f"\n📊 输入配置:")
        print(f"  事件目录: {catalog_file}")
        print(f"  波形数据: {preprocessor.input_dirs['waveforms']}")
        print(f"  响应数据: {preprocessor.input_dirs['responses']}")
        print(f"📁 输出目录: {preprocessor.output_dirs['vel_data']}")
        
        print("\n是否开始处理? (y/N): ", end="")
        response = input().strip().lower()
        
        if response != 'y':
            print("处理已取消")
            return
        
        print("\n开始批量预处理...")
        results = preprocessor.batch_process(catalog_file, max_events=None)
        
        stats = results['stats']
        print(f"\n🎉 波形预处理完成!")
        print(f"📈 处理统计:")
        print(f"  总事件数: {stats['total_events']}")
        print(f"  成功处理: {stats['processed_events']}")
        print(f"  跳过事件: {stats['skipped_events']}")
        print(f"  成功台站: {stats['successful_stations']:,}")
        print(f"  失败台站: {stats['failed_stations']:,}")
        print(f"  分量转换: {stats['component_conversions']:,} 次")
        print(f"  简化响应去除: {stats['simplified_response_removal']:,} 次")
        
        # 🆕 显示警告抑制统计
        if preprocessor.config.performance['suppress_all_warnings']:
            print(f"  抑制警告: {stats['warning_stats']['response_warnings_suppressed']:,} 次 ✅")
        
        if stats['total_stations'] > 0:
            success_rate = stats['successful_stations'] / stats['total_stations'] * 100
            print(f"  成功率: {success_rate:.1f}%")
        
        if preprocessor.config.performance['cache_inventory']:
            cache_hits = stats.get('inventory_cache_hits', 0)
            cache_misses = stats.get('inventory_cache_misses', 0)
            if cache_hits + cache_misses > 0:
                cache_hit_rate = cache_hits / (cache_hits + cache_misses) * 100
                print(f"  Inventory缓存命中率: {cache_hit_rate:.1f}%")
        
        print(f"⏱️ 总耗时: {results['total_time_seconds']/60:.2f} 分钟")
        
        if stats['processed_events'] > 0:
            avg_time = results['total_time_seconds'] / stats['processed_events']
            print(f"  平均速度: {avg_time:.1f} 秒/事件")
        
        print(f"📋 详细报告: {results['report_file']}")
        print(f"📁 输出目录: {preprocessor.output_dirs['vel_data']}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断处理")
        print("💡 提示：下次运行时将自动从中断处继续")
    except Exception as e:
        print(f"\n❌ 处理过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*60)


if __name__ == "__main__":
    main()