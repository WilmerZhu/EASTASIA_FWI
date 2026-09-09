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
   - 按台站合并三分量后，用 StationXML 方位角真旋转到 Z/N/E
   - 支持 BH1/BH2（及 HH*）经 inventory 旋转，禁止仅改通道名的假旋转
   - 多分量数据一致性检查

3. ✅ 数据质量控制
   - 采样率验证（0.5-100 Hz）
   - StationXML 完整性（最少响应级数、输出须为 COUNTS）
   - 去响应后位移振幅/有限性门限（拒绝残缺响应伪位移）
   - 去响应前 counts 上做单点毛刺检测（不用全迹振幅离群，避免误杀面波）
   - 台站级完整 ZNE 门槛：缺水平向整台丢弃，禁止写出残缺三分量

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
- events/disp_data/{event_name}/: 每个事件的预处理后位移 SAC（与 SPECFEM 正演 DISP 对齐）
- preprocessing/preprocessing_report.txt: 详细处理报告
- preprocessing/preprocessing_stats.json: 处理统计信息
- preprocessing_progress.json: 处理进度文件

科学原理:
----------
- 仪器响应去除: 将观测数据从计数转换为物理位移（DISP），消除仪器频率响应影响
- 预滤波: 在响应去除前应用预滤波，避免频率域边缘的数值不稳定
- 分量旋转: 将仪器坐标系转换为地理坐标系（N-E），便于后续处理和分析
- 质量控制: 确保数据质量满足全波形反演的要求，避免低质量数据影响反演结果
- SAC格式: 地震学标准数据格式；输出位移以便与 SPECFEM3D Globe 合成波形直接对比

作者: EASTASIA-FWI Team
日期: 2025-02-01
版本: v2.8
"""

import os
import sys
import argparse
import warnings
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Set
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
        # 输入目录配置（相对 data/）
        # 优先用下载会话 events/<YYYYMMDD>_download_XX/{waveforms,responses}；
        # 若无会话则回退到 events/waveforms 与 events/responses。
        self.input_directories = {
            'events': 'events/',
            'waveforms': 'events/waveforms',
            'responses': 'events/responses',
            # 非空则强制使用该会话子目录名，如 '20260805_download_01'
            'download_session': '20260805_download_02',
        }
        
        # 输出目录配置
        self.output_directories = {
            # 位移 SAC：与 SPECFEM 正演 *.sem.sac（DISP）物理量一致
            'disp_data': 'events/disp_data',
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
        
        # 响应去除参数（输出位移，便于合成 vs 观测直接对比）
        self.response_removal = {
            'output_type': 'DISP',
            'pre_filt': [0.008, 0.012, 8.0, 10.0],
            'water_level': 60.0,
            'remove_mean': True,
            'remove_trend': True,
            'taper_fraction': 0.05
        }
        
        # 通道处理配置（按台站合并后用 inventory 真旋转到 ZNE）
        self.channel_processing = {
            'auto_rotate_to_zne': True,
            'preferred_channels': ['BHZ', 'BHN', 'BHE', 'BH1', 'BH2', 'HHZ', 'HHN', 'HHE', 'HH1', 'HH2'],
            'component_mapping': {'1': 'N', '2': 'E'},
        }
        
        # 数据质量控制
        self.quality_control = {
            'min_sample_rate': 0.5,
            'max_sample_rate': 100.0,
            'min_length_ratio': 0.8,
            # 毛刺检测：仅在去响应前的 counts 上做；用邻域残差抓单点尖峰，
            # 避免“全迹均值±kσ”把地震面波当成尖峰误杀水平向
            'spike_detection': True,
            'spike_threshold': 12.0,
            'spike_max_ratio': 0.005,
            'gap_tolerance': 0.1,
            # StationXML 完整性：残缺响应（如仅 1 级 PAZ、输出非 COUNTS）禁止去响应
            'min_response_stages': 3,
            'require_counts_output': True,
            # 去响应后位移振幅门限（米）；残缺响应常给出 ~1e3 m 量级伪位移
            'disp_amp_min_m': 1.0e-12,
            'disp_amp_max_m': 1.0e-1,
            'max_nonfinite_ratio': 0.01,
            # 禁止去掉 pre_filt 的“简化”去响应（易在 DISP 上爆炸）
            'allow_simplified_response_removal': False,
            # 波形对比需要 NE→RT：必须完整垂直+水平，禁止写出残缺三分量
            'require_complete_zne': True,
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
    
    def __init__(
        self,
        download_session: Optional[str] = None,
        skip_existing: Optional[bool] = None,
    ):
        """
        初始化波形预处理器。

        Args:
            download_session: 覆盖配置中的下载会话名（如 20260805_download_02）
            skip_existing: 覆盖断点续传跳过已处理事件；False 强制重跑
        """
        self.base_config = BaseConfig()
        self.config = PreprocessingConfig()

        if download_session is not None:
            self.config.input_directories['download_session'] = download_session.strip()
        if skip_existing is not None:
            self.config.resume['skip_existing'] = bool(skip_existing)

        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.WaveformPreprocessor',
            self.config.logging['level']
        )

        self._setup_directories()
        self.stats = self._initialize_statistics()
        self._create_output_directories()
        self.inventory_cache = {} if self.config.performance['cache_inventory'] else None
        self.progress_file = self.input_dirs['events'] / 'preprocessing_progress.json'

        self.logger.info("🔬 波形预处理器初始化完成")
        self._print_config_summary()
    
    def _resolve_download_session(self, events_dir: Path) -> Optional[Path]:
        """
        解析下载会话目录（含 waveforms/ 与 responses/）。

        优先用配置里的 download_session；否则取 events/ 下最新的 *_download_*。
        """
        session_name = (self.config.input_directories.get('download_session') or '').strip()
        if session_name:
            cand = events_dir / session_name
            if (cand / 'waveforms').is_dir() and (cand / 'responses').is_dir():
                return cand
            self.logger.warning(
                f"配置的 download_session 不存在或不完整: {cand}，改为自动探测")

        sessions = [
            p for p in events_dir.glob('*_download_*')
            if p.is_dir() and (p / 'waveforms').is_dir() and (p / 'responses').is_dir()
        ]
        if not sessions:
            return None
        sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return sessions[0]

    def _setup_directories(self):
        """设置输入输出目录；hybrid 数据在下载会话子目录内时自动对齐。"""
        events_dir = self.base_config.dirs['data'] / self.config.input_directories['events']
        wf_default = self.base_config.dirs['data'] / self.config.input_directories['waveforms']
        resp_default = self.base_config.dirs['data'] / self.config.input_directories['responses']

        session = self._resolve_download_session(events_dir)
        if session is not None:
            wf_dir = session / 'waveforms'
            resp_dir = session / 'responses'
            self.logger.info(f"使用下载会话输入: {session.name}")
        else:
            wf_dir, resp_dir = wf_default, resp_default

        self.input_dirs = {
            'events': events_dir,
            'waveforms': wf_dir,
            'responses': resp_dir,
        }

        self.output_dirs = {
            'disp_data': self.base_config.dirs['data'] / self.config.output_directories['disp_data'],
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
        print(f"   ZNE真旋转: {'启用' if self.config.channel_processing.get('auto_rotate_to_zne', True) else '禁用'}")
        print(f"   完整ZNE门槛: {'启用' if self.config.quality_control.get('require_complete_zne', True) else '禁用'}")
        print(f"   响应完整性QC: stages>={self.config.quality_control.get('min_response_stages', 3)}, "
              f"振幅门限 [{self.config.quality_control.get('disp_amp_min_m')}, "
              f"{self.config.quality_control.get('disp_amp_max_m')}] m")
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
            'incomplete_response_rejected': 0,
            'amplitude_qc_rejected': 0,
            'incomplete_zne_rejected': 0,
            # 警告统计
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
                'response_error': 0,
                'incomplete_response': 0,
                'amplitude_qc_failed': 0,
                'incomplete_zne': 0,
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
        
        event_output_dir = self.output_dirs['disp_data'] / event_name
        
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
    
    def _station_key_from_waveform_path(self, waveform_file: Path) -> Optional[str]:
        """从 mseed 文件名解析 network.station。"""
        try:
            parts = waveform_file.stem.split('__')[0].split('.')
            if len(parts) >= 2:
                return f"{parts[0]}.{parts[1]}"
        except Exception:
            pass
        return None

    def group_waveforms_by_station(
        self, waveform_files: List[Path]
    ) -> Dict[str, List[Path]]:
        """按 network.station 分组波形文件（三分量需合并后再旋转/去响应）。"""
        groups: Dict[str, List[Path]] = {}
        for wf in waveform_files:
            key = self._station_key_from_waveform_path(wf)
            if key is None:
                continue
            groups.setdefault(key, []).append(wf)
        return groups

    def validate_response_inventory(
        self, inv: Any, st: Stream
    ) -> Tuple[bool, str]:
        """
        检查 StationXML 是否足够完整，可安全去响应到物理量。

        残缺响应（如仅 1 级 PolesZeros、输出为 V 而非 COUNTS）会导致
        DISP 振幅偏差达数个数量级，必须拒绝而非静默处理。
        """
        qc = self.config.quality_control
        min_stages = int(qc.get('min_response_stages', 3))
        require_counts = bool(qc.get('require_counts_output', True))

        if inv is None or len(inv) == 0:
            return False, "empty_inventory"

        for tr in st:
            resp = self._get_channel_response(inv, tr)
            if resp is None:
                return False, f"no_matching_response:{tr.id}"

            stages = getattr(resp, 'response_stages', None) or []
            n_stages = len(stages)
            if n_stages < min_stages:
                return False, f"too_few_stages:{tr.id}:{n_stages}<{min_stages}"

            if require_counts:
                out_units = ''
                # 优先用最后一级输出单位；完整响应应为 COUNTS
                if stages:
                    last_out = str(getattr(stages[-1], 'output_units', '') or '')
                    if last_out:
                        out_units = last_out
                if not out_units:
                    sens = getattr(resp, 'instrument_sensitivity', None)
                    if sens is not None:
                        out_units = str(getattr(sens, 'output_units', '') or '')
                units_u = out_units.upper()
                if 'COUNT' not in units_u and 'DIGIT' not in units_u:
                    return False, f"non_counts_output:{tr.id}:{out_units or 'unknown'}"

        return True, "ok"

    def _get_channel_response(self, inv: Any, tr) -> Optional[Any]:
        """匹配通道响应；兼容 mseed 与 StationXML location 码不一致。"""
        t0 = tr.stats.starttime
        candidates = [
            tr.id,
            f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location or ''}."
            f"{tr.stats.channel}",
            f"{tr.stats.network}.{tr.stats.station}..{tr.stats.channel}",
            f"{tr.stats.network}.{tr.stats.station}.00.{tr.stats.channel}",
        ]
        for seed_id in candidates:
            try:
                return inv.get_response(seed_id, t0)
            except Exception:
                continue
        # 最后按网台通道在 inventory 内直接查找
        try:
            for network in inv:
                if network.code != tr.stats.network:
                    continue
                for station in network:
                    if station.code != tr.stats.station:
                        continue
                    for channel in station:
                        if channel.code != tr.stats.channel:
                            continue
                        if channel.response is not None:
                            return channel.response
        except Exception:
            pass
        return None

    def rotate_to_zne(self, st: Stream, inv: Any) -> Tuple[Stream, bool]:
        """
        使用 StationXML 方位角将分量旋转到 Z/N/E。

        取代仅改通道名的假旋转；要求同台站三分量已在同一 Stream 中。
        """
        if not self.config.channel_processing.get('auto_rotate_to_zne', True):
            return st, False
        if inv is None or len(st) == 0:
            return st, False

        comps = {tr.stats.channel[-1].upper() for tr in st}
        needs_rotate = bool(comps & {'1', '2'}) or (
            ('N' in comps or 'E' in comps) and 'Z' in comps
        )
        # 已是 ZNE 且无 12：仍可用 inventory 校正方位（若有 N/E）
        try:
            st_work = st.copy()
            # 合并同分量碎片，旋转要求对齐
            try:
                st_work.merge(method=1, fill_value=0)
            except Exception:
                pass

            before_ids = [tr.id for tr in st_work]
            if needs_rotate or ({'1', '2'} & comps):
                st_work.rotate(method="->ZNE", inventory=inv)
                self.stats['component_conversions'] += 1
                return st_work, True

            # 仅有 Z 时不强制旋转
            if 'Z' in comps and not ({'N', 'E', '1', '2'} & comps):
                return st_work, False

            # 已有 NE：用 inventory 再规范化到 ZNE（校正非正交/方位）
            if {'N', 'E'} <= comps or ({'Z', 'N', 'E'} <= comps):
                try:
                    st_work.rotate(method="->ZNE", inventory=inv)
                    if [tr.id for tr in st_work] != before_ids:
                        self.stats['component_conversions'] += 1
                    return st_work, True
                except Exception:
                    return st_work, False

            return st_work, False
        except Exception as e:
            self.stats['error_types']['rotation_failed'] += 1
            if self.config.logging.get('show_debug_info'):
                self.logger.debug(f"旋转失败: {e}")
            return st, False

    def remove_instrument_response(self, st: Stream, inventory_file: Path,
                                 event_info: Dict[str, Any],
                                 inv: Optional[Any] = None) -> Tuple[bool, str]:
        """去除仪器响应；拒绝不完整响应，默认禁止无 pre_filt 回退。"""
        try:
            if inv is None:
                if not inventory_file.exists():
                    self.stats['error_types']['no_response_file'] += 1
                    return False, "no_response_file"
                inv = self.load_inventory(inventory_file)
            if inv is None:
                self.stats['error_types']['response_error'] += 1
                return False, "read_inventory_failed"

            ok_resp, resp_msg = self.validate_response_inventory(inv, st)
            if not ok_resp:
                self.stats['incomplete_response_rejected'] += 1
                self.stats['error_types']['incomplete_response'] += 1
                return False, f"incomplete_response:{resp_msg}"

            resp_params = self.config.response_removal

            if resp_params['remove_mean']:
                st.detrend(type="demean")
            if resp_params['remove_trend']:
                st.detrend(type="linear")
            if resp_params['taper_fraction'] > 0:
                st.taper(max_percentage=resp_params['taper_fraction'])

            self.stats['warning_stats']['total_response_removal_calls'] += 1

            if self.config.performance['suppress_all_warnings']:
                with SuppressOutput():
                    return self._do_remove_response(st, inv, resp_params)
            return self._do_remove_response(st, inv, resp_params)

        except FileNotFoundError:
            self.stats['error_types']['no_response_file'] += 1
            return False, "no_response_file"
        except Exception as e:
            self.stats['error_types']['response_removal_failed'] += 1
            return False, f"general_error: {str(e)[:100]}"

    def _do_remove_response(self, st: Stream, inv: Any, resp_params: Dict) -> Tuple[bool, str]:
        """执行响应去除；默认失败时不回退到无 pre_filt。"""
        try:
            st.remove_response(
                inventory=inv,
                output=resp_params['output_type'],
                pre_filt=resp_params['pre_filt'],
                water_level=resp_params['water_level']
            )
            self.stats['warning_stats']['response_warnings_suppressed'] += 1
            return True, "success"

        except (AttributeError, TypeError) as e:
            err = str(e).lower()
            allow_simple = self.config.quality_control.get(
                'allow_simplified_response_removal', False
            )
            if allow_simple and ('hann' in err or 'tukey' in err or 'window' in err):
                try:
                    st_copy = st.copy()
                    st_copy.remove_response(
                        inventory=inv,
                        output=resp_params['output_type'],
                        water_level=resp_params['water_level']
                    )
                    for i, tr in enumerate(st):
                        tr.data = st_copy[i].data
                        tr.stats = st_copy[i].stats
                    self.stats['simplified_response_removal'] += 1
                    return True, "success_simplified"
                except Exception:
                    self.stats['error_types']['response_error'] += 1
                    return False, "simplified_removal_failed"
            self.stats['error_types']['response_removal_failed'] += 1
            return False, f"remove_response_failed:{str(e)[:80]}"

        except Exception as e:
            error_msg = str(e)
            if "No matching response" in error_msg:
                self.stats['error_types']['no_matching_response'] += 1
                return False, "no_matching_response"
            if "divide by zero" in error_msg or "invalid value" in error_msg:
                self.stats['error_types']['response_error'] += 1
                return False, "response_error"
            self.stats['error_types']['response_removal_failed'] += 1
            return False, f"processing_error: {error_msg[:100]}"

    def get_station_coordinates(
        self, inventory_file: Path, inv: Optional[Any] = None
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """获取台站坐标信息"""
        try:
            if inv is None:
                inv = self.load_inventory(inventory_file)
            if inv is None:
                return None, None, None
            station = inv.networks[0].stations[0]
            return station.latitude, station.longitude, station.elevation
        except Exception:
            return None, None, None

    def _channel_orientation(
        self, inv: Any, tr
    ) -> Tuple[Optional[float], Optional[float]]:
        """从 inventory 读取分量方位角/倾角 (cmpaz, cmpinc)。"""
        try:
            net = tr.stats.network
            sta = tr.stats.station
            loc = tr.stats.location or ''
            cha = tr.stats.channel
            for network in inv:
                if network.code != net:
                    continue
                for station in network:
                    if station.code != sta:
                        continue
                    for channel in station:
                        if channel.code != cha:
                            continue
                        if (channel.location_code or '') != loc and loc != '':
                            continue
                        az = getattr(channel, 'azimuth', None)
                        dip = getattr(channel, 'dip', None)
                        # SAC cmpinc: 0=up, 90=horizontal；ObsPy dip: -90=up, 0=horizontal
                        cmpinc = None
                        if dip is not None:
                            cmpinc = float(dip) + 90.0
                        return (
                            float(az) if az is not None else None,
                            cmpinc,
                        )
        except Exception:
            pass
        # 按通道尾字符回退
        comp = tr.stats.channel[-1].upper()
        if comp == 'Z':
            return 0.0, 0.0
        if comp == 'N':
            return 0.0, 90.0
        if comp == 'E':
            return 90.0, 90.0
        return None, None

    def update_sac_headers(
        self,
        sac_file: Path,
        event_info: Dict[str, Any],
        stla: float,
        stlo: float,
        stel: float,
        tr_ref=None,
        inv: Optional[Any] = None,
    ) -> bool:
        """用 ObsPy 写入 SAC 头（含 idep/cmpaz/cmpinc）；不依赖外部 sac 命令。"""
        if not self.config.output_format['update_sac_headers']:
            return True

        try:
            st = read(str(sac_file))
            if len(st) == 0:
                return False
            tr = st[0]
            origin_time = event_info['origin_time']
            # 确保 sac 头存在
            if not hasattr(tr.stats, 'sac') or tr.stats.sac is None:
                tr.stats.sac = {}

            sac = tr.stats.sac
            sac['evlo'] = float(event_info['longitude'])
            sac['evla'] = float(event_info['latitude'])
            sac['evdp'] = float(event_info['depth'])
            sac['stlo'] = float(stlo)
            sac['stla'] = float(stla)
            sac['stel'] = float(stel)
            sac['lcalda'] = 1
            # IDEP: 6 = displacement (meters)，与 SPECFEM DISP 一致
            out_type = self.config.response_removal.get('output_type', 'DISP').upper()
            if out_type == 'DISP':
                sac['idep'] = 6
            elif out_type == 'VEL':
                sac['idep'] = 7
            elif out_type == 'ACC':
                sac['idep'] = 8

            sac['nzyear'] = int(origin_time.year)
            sac['nzjday'] = int(origin_time.julday)
            sac['nzhour'] = int(event_info['hour'])
            sac['nzmin'] = int(event_info['minute'])
            sec = float(event_info['second'])
            sac['nzsec'] = int(sec)
            sac['nzmsec'] = int(round((sec % 1) * 1000))
            sac['o'] = 0.0

            ref = tr_ref if tr_ref is not None else tr
            if inv is not None:
                cmpaz, cmpinc = self._channel_orientation(inv, ref)
                if cmpaz is not None:
                    sac['cmpaz'] = float(cmpaz)
                if cmpinc is not None:
                    sac['cmpinc'] = float(cmpinc)
            else:
                comp = ref.stats.channel[-1].upper()
                if comp == 'Z':
                    sac['cmpaz'], sac['cmpinc'] = 0.0, 0.0
                elif comp == 'N':
                    sac['cmpaz'], sac['cmpinc'] = 0.0, 90.0
                elif comp == 'E':
                    sac['cmpaz'], sac['cmpinc'] = 90.0, 90.0

            tr.write(str(sac_file), format='SAC')
            return True
        except Exception:
            # 回退：尝试外部 sac（可选）
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
                ch idep 6
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
                    stdout=subprocess.DEVNULL,
                )
                return result.returncode == 0
            except Exception:
                return False

    def save_processed_waveform(
        self,
        st: Stream,
        event_name: str,
        event_info: Dict[str, Any],
        stla: float,
        stlo: float,
        stel: float,
        inv: Optional[Any] = None,
    ) -> Dict[str, str]:
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
            newsacname = (
                f"{origin_time.year}.{origin_time.julday:03d}."
                f"{hour}.{mini}.{msec}.{station_id}..{channel}.SAC"
            )

            output_dir = self.output_dirs['disp_data'] / event_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / newsacname

            try:
                Stream([tr.copy()]).write(str(output_path), format="SAC")
                saved_files[channel] = str(output_path)
                self.update_sac_headers(
                    output_path, event_info, stla, stlo, stel,
                    tr_ref=tr, inv=inv,
                )
            except Exception:
                self.stats['error_types']['saving_failed'] += 1
                continue

        return saved_files

    @staticmethod
    def _component_letters(st: Stream) -> Set[str]:
        """通道末字符集合（Z/N/E/1/2/...）。"""
        out: Set[str] = set()
        for tr in st:
            chan = (tr.stats.channel or '').upper()
            if chan:
                out.add(chan[-1])
        return out

    def has_complete_zne(self, st: Stream) -> bool:
        """
        是否具备可旋转/可评测的完整三分量。

        接受 Z+N+E，或已带 Z+1+2（旋转前）；旋转后应落到 ZNE。
        """
        comps = self._component_letters(st)
        if {'Z', 'N', 'E'} <= comps:
            return True
        if {'Z', '1', '2'} <= comps:
            return True
        return False

    @staticmethod
    def _glitch_ratio(data: np.ndarray, threshold: float) -> float:
        """
        单点毛刺比例：邻域残差 |x_i - 0.5(x_{i-1}+x_{i+1})| 相对 MAD。

        面波大振幅在邻域上光滑，不会被当成毛刺；真正的单样本尖峰会被抓住。
        """
        x = np.asarray(data, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 3:
            return 0.0
        resid = np.abs(x[1:-1] - 0.5 * (x[:-2] + x[2:]))
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        if mad > 0:
            scale = 1.4826 * mad
        else:
            scale = float(np.std(resid))
        if scale <= 0:
            return 0.0
        return float(np.sum(resid > threshold * scale) / resid.size)

    def apply_quality_control(
        self, st: Stream, event_info: Dict[str, Any],
        check_spikes: bool = True,
    ) -> Tuple[Stream, bool, str]:
        """
        采样率/长度/毛刺等基础 QC。

        Args:
            st: 输入数据流
            event_info: 事件元数据（当前未用于门限，保留接口）
            check_spikes: True 时做单点毛刺检测（应在去响应前的 counts 上开启）

        Returns:
            (通过的道, 是否仍有有效道, 说明文字)
        """
        if not st:
            return st, False, "空数据流"

        qc_params = self.config.quality_control
        passed_traces = Stream()
        qc_messages = []
        spike_thr = float(qc_params.get('spike_threshold', 12.0))
        spike_max = float(qc_params.get('spike_max_ratio', 0.005))

        for tr in st:
            if not (qc_params['min_sample_rate'] <= tr.stats.sampling_rate
                    <= qc_params['max_sample_rate']):
                qc_messages.append(
                    f"{tr.id}:采样率不符合要求:{tr.stats.sampling_rate}"
                )
                continue

            expected_length = int(
                (tr.stats.endtime - tr.stats.starttime) * tr.stats.sampling_rate
            )
            actual_length = len(tr.data)
            length_ratio = actual_length / expected_length if expected_length > 0 else 0

            if length_ratio < qc_params['min_length_ratio']:
                qc_messages.append(f"{tr.id}:数据长度不足:{length_ratio:.2f}")
                continue

            data = np.asarray(tr.data, dtype=float)
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                qc_messages.append(f"{tr.id}:无有效采样点")
                continue

            if check_spikes and qc_params.get('spike_detection', True):
                spike_ratio = self._glitch_ratio(finite, spike_thr)
                if spike_ratio > spike_max:
                    qc_messages.append(f"{tr.id}:毛刺过多:{spike_ratio:.4f}")
                    continue

            passed_traces += tr

        success = len(passed_traces) > 0
        message = "; ".join(qc_messages) if qc_messages else "质控通过"
        if not success:
            self.stats['error_types']['quality_control_failed'] += 1
        return passed_traces, success, message

    def apply_amplitude_qc(self, st: Stream) -> Tuple[Stream, bool, str]:
        """去响应后振幅/有限性 QC（针对 DISP，单位 m）。不做尖峰检测。"""
        if not st:
            return st, False, "空数据流"

        qc = self.config.quality_control
        out_type = self.config.response_removal.get('output_type', 'DISP').upper()
        amp_min = float(qc.get('disp_amp_min_m', 1e-12))
        amp_max = float(qc.get('disp_amp_max_m', 1e-1))
        max_nan = float(qc.get('max_nonfinite_ratio', 0.01))

        # VEL/ACC 时放宽量级（仍做有限性检查）
        if out_type == 'VEL':
            amp_min, amp_max = 1e-12, 1.0
        elif out_type == 'ACC':
            amp_min, amp_max = 1e-12, 10.0

        passed = Stream()
        messages = []
        for tr in st:
            data = np.asarray(tr.data, dtype=float)
            if data.size == 0:
                messages.append(f"{tr.id}:empty")
                continue
            nonfinite = ~np.isfinite(data)
            ratio = float(np.sum(nonfinite) / data.size)
            if ratio > max_nan:
                messages.append(f"{tr.id}:nonfinite={ratio:.3f}")
                continue
            peak = float(np.nanmax(np.abs(data)))
            if not np.isfinite(peak) or peak < amp_min or peak > amp_max:
                messages.append(f"{tr.id}:peak={peak:.3e}")
                continue
            passed += tr

        ok = len(passed) > 0
        if not ok:
            self.stats['amplitude_qc_rejected'] += 1
            self.stats['error_types']['amplitude_qc_failed'] += 1
        return passed, ok, ("; ".join(messages) if messages else "amp_ok")

    def _reject_incomplete_zne(
        self, st: Stream, station_id: str, event_name: str, stage: str,
    ) -> Optional[Tuple[Optional[str], Optional[str], Optional[str]]]:
        """
        若要求完整 ZNE 且当前流不满足，返回失败三元组；否则返回 None 继续。
        """
        if not self.config.quality_control.get('require_complete_zne', True):
            return None
        if self.has_complete_zne(st):
            return None
        comps = sorted(self._component_letters(st))
        self.stats['incomplete_zne_rejected'] += 1
        self.stats['error_types']['incomplete_zne'] += 1
        return (
            station_id,
            None,
            f"{event_name}: {station_id}:incomplete_zne@{stage}:{comps}",
        )

    def _clear_event_disp_dir(self, event_name: str) -> None:
        """强制重跑时清空该事件旧 disp SAC，避免残留残缺三分量。"""
        event_dir = self.output_dirs['disp_data'] / event_name
        if not event_dir.is_dir():
            return
        removed = 0
        for path in list(event_dir.glob('*.SAC')) + list(event_dir.glob('*.sac')):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        if removed and self.config.logging.get('show_debug_info'):
            self.logger.debug(f"清空 {event_name} 旧 SAC {removed} 个")

    def process_station(
        self,
        station_id: str,
        waveform_files: List[Path],
        response_file: Path,
        event_info: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        按台站处理：合并三分量 → 真旋转 ZNE → counts QC → 去响应 →
        振幅 QC → 完整 ZNE 门槛 → 写 SAC。

        Returns:
            (station_id, no_resp_msg, error_msg)；成功时后两者为 None。
        """
        event_name = event_info['name']
        try:
            st = Stream()
            for wf in waveform_files:
                try:
                    st += read(str(wf))
                except Exception:
                    continue
            if len(st) == 0:
                return station_id, None, f"{event_name}: {station_id}:empty_stream"

            inv = self.load_inventory(response_file)
            stla, stlo, stel = self.get_station_coordinates(response_file, inv=inv)
            if stla is None or stlo is None:
                self.stats['error_types']['file_not_found'] += 1
                return station_id, f"{event_name}: {station_id}", None

            # 真旋转（需三分量在同一 Stream）
            st, _ = self.rotate_to_zne(st, inv)

            # 旋转后仍无完整三分量：整台丢弃（不写残缺 SAC）
            bad = self._reject_incomplete_zne(st, station_id, event_name, 'after_rotate')
            if bad is not None:
                return bad

            # counts 域毛刺/长度 QC（去响应前）
            st_qc, qc_ok, qc_msg = self.apply_quality_control(
                st, event_info, check_spikes=True,
            )
            if not qc_ok:
                return station_id, None, f"{event_name}: {station_id}:qc_failed ({qc_msg})"
            bad = self._reject_incomplete_zne(
                st_qc, station_id, event_name, 'after_counts_qc',
            )
            if bad is not None:
                return bad

            success, error_msg = self.remove_instrument_response(
                st_qc, response_file, event_info, inv=inv
            )
            if not success:
                if error_msg.startswith('incomplete_response') or error_msg in (
                    'no_response_file', 'read_inventory_failed', 'no_matching_response'
                ):
                    return station_id, f"{event_name}: {station_id} ({error_msg})", None
                return station_id, None, f"{event_name}: {station_id} ({error_msg})"

            # 去响应后只做长度/采样率（不再做尖峰）+ 振幅门限
            st_len, len_ok, len_msg = self.apply_quality_control(
                st_qc, event_info, check_spikes=False,
            )
            if not len_ok:
                return station_id, None, f"{event_name}: {station_id}:post_qc ({len_msg})"

            st_amp, amp_ok, amp_msg = self.apply_amplitude_qc(st_len)
            if not amp_ok:
                return station_id, None, f"{event_name}: {station_id}:amp_qc ({amp_msg})"

            # 任一水平向振幅不合格 → 整台丢弃，禁止只留 Z
            bad = self._reject_incomplete_zne(
                st_amp, station_id, event_name, 'after_amp_qc',
            )
            if bad is not None:
                return bad

            saved = self.save_processed_waveform(
                st_amp, event_name, event_info, stla, stlo, stel or 0.0, inv=inv
            )
            if saved:
                self.stats['successful_stations'] += 1
                return station_id, None, None
            return station_id, None, f"{event_name}: {station_id}:save_failed"

        except FileNotFoundError:
            self.stats['error_types']['file_not_found'] += 1
            return station_id, f"{event_name}: {station_id}", None
        except Exception as e:
            return station_id, None, f"{event_name}: {station_id}:{str(e)[:80]}"

    def process_single_waveform(
        self,
        waveform_file: Path,
        response_file: Path,
        event_info: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """兼容旧接口：单文件视为单台站处理。"""
        station_id = self._station_key_from_waveform_path(waveform_file) or "unknown"
        return self.process_station(
            station_id, [waveform_file], response_file, event_info
        )

    def process_event(self, event_info: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """按台站合并三分量后处理单个事件的全部波形。"""
        event_name = event_info['name']

        if self.config.resume['skip_existing'] and self.is_event_processed(event_name):
            self.stats['skipped_events'] += 1
            return [], []

        # 强制重跑：先清旧 SAC，避免残缺三分量残留
        if not self.config.resume['skip_existing']:
            self._clear_event_disp_dir(event_name)

        waveform_files, response_files = self.find_waveform_files(event_name)

        if not waveform_files or not response_files:
            if self.config.logging['show_debug_info']:
                self.logger.warning(
                    f"事件 {event_name}: 波形={len(waveform_files) if waveform_files else 0}, "
                    f"响应={len(response_files) if response_files else 0}"
                )
            return [], []

        response_mapping = self.build_response_mapping(response_files)
        station_groups = self.group_waveforms_by_station(waveform_files)

        no_resp_stations: List[str] = []
        error_resp_stations: List[str] = []
        match_failures = 0

        for station_id, wf_list in station_groups.items():
            self.stats['total_stations'] += 1
            response_file = response_mapping.get(station_id)

            if not response_file:
                # 用任一文件再试快速匹配
                response_file = self.match_response_file_fast(wf_list[0], response_mapping)

            if not response_file:
                no_resp_stations.append(f"{event_name}: {station_id} (no matching response)")
                self.stats['no_response_stations'].append(f"{event_name}: {station_id}")
                match_failures += 1
                self.stats['failed_stations'] += 1
                continue

            sid, no_resp, error_resp = self.process_station(
                station_id, wf_list, response_file, event_info
            )

            if no_resp:
                no_resp_stations.append(no_resp)
                self.stats['failed_stations'] += 1
            elif error_resp:
                error_resp_stations.append(error_resp)
                self.stats['failed_stations'] += 1

        if self.config.performance['memory_efficient']:
            gc.collect()

        if match_failures > 0 and self.config.logging['show_debug_info']:
            self.logger.warning(f"⚠️  {event_name}: {match_failures} 个台站无匹配响应")

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
                     max_events: Optional[int] = None,
                     event_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """🆕 批量处理多个事件"""
        self.stats['processing_summary']['start_time'] = datetime.now()
        start_time = self.stats['processing_summary']['start_time']
        
        self.logger.info("🚀 开始批量波形预处理")
        
        try:
            events = self.load_event_catalog(catalog_file)

            if event_names:
                allow = {n.strip() for n in event_names if n and str(n).strip()}
                events = [e for e in events if e.get('name') in allow]
                self.logger.info(f"按 --events 过滤后剩余 {len(events)} 个事件")
            
            if max_events:
                events = events[:max_events]
            
            # 强制重跑或指定事件列表时，不从旧进度续跑
            start_idx = 0
            if self.config.resume['skip_existing'] and not event_names:
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
                f.write(f"  ZNE真旋转: {self.stats['component_conversions']:,} 次\n")
                f.write(f"  拒绝残缺响应: {self.stats.get('incomplete_response_rejected', 0):,}\n")
                f.write(f"  拒绝异常振幅: {self.stats.get('amplitude_qc_rejected', 0):,}\n")
                f.write(f"  拒绝残缺ZNE: {self.stats.get('incomplete_zne_rejected', 0):,}\n")
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
    parser = argparse.ArgumentParser(
        description='EASTASIA-FWI 波形预处理（DISP SAC，台站级真旋转+响应QC）'
    )
    parser.add_argument(
        '--session', type=str, default=None,
        help='下载会话名，如 20260805_download_02（覆盖配置）',
    )
    parser.add_argument(
        '--catalog', type=str, default=None,
        help='事件目录 .par（默认取最新一个，30 事件请显式指定）',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='强制重跑（不跳过已有 disp_data）',
    )
    parser.add_argument(
        '--events', type=str, nargs='+', default=None,
        help='只处理指定事件名（如 20120812.10.47），可多个',
    )
    parser.add_argument(
        '--yes', '-y', action='store_true',
        help='跳过交互确认',
    )
    parser.add_argument(
        '--max-events', type=int, default=None,
        help='最多处理事件数（调试用）',
    )
    args = parser.parse_args()

    print("🔬 EASTASIA-FWI 波形预处理模块 v2.8")
    print("=" * 60)

    try:
        preprocessor = WaveformPreprocessor(
            download_session=args.session,
            skip_existing=False if args.force else None,
        )

        if args.catalog:
            catalog_file = Path(args.catalog).expanduser()
            if not catalog_file.is_file():
                cand = preprocessor.input_dirs['events'] / args.catalog
                catalog_file = cand if cand.is_file() else catalog_file
            if not catalog_file.is_file():
                print(f"❌ 目录文件不存在: {args.catalog}")
                return
            catalog_file = catalog_file.resolve()
        else:
            catalog_files = preprocessor.find_catalog_files()
            if not catalog_files:
                print("❌ 未找到事件目录文件")
                return
            catalog_file = catalog_files[0]

        print(f"\n📊 输入配置:")
        print(f"  事件目录: {catalog_file}")
        print(f"  波形数据: {preprocessor.input_dirs['waveforms']}")
        print(f"  响应数据: {preprocessor.input_dirs['responses']}")
        print(f"📁 输出目录: {preprocessor.output_dirs['disp_data']}")
        if args.force:
            print("  模式: --force（不跳过已处理事件，并清空对应事件旧 SAC）")
        if args.events:
            print(f"  事件过滤: {', '.join(args.events)}")

        if not args.yes:
            print("\n是否开始处理? (y/N): ", end="")
            response = input().strip().lower()
            if response != 'y':
                print("处理已取消")
                return

        print("\n开始批量预处理...")
        results = preprocessor.batch_process(
            catalog_file,
            max_events=args.max_events,
            event_names=args.events,
        )

        stats = results['stats']
        print(f"\n🎉 波形预处理完成!")
        print(f"📈 处理统计:")
        print(f"  总事件数: {stats['total_events']}")
        print(f"  成功处理: {stats['processed_events']}")
        print(f"  跳过事件: {stats['skipped_events']}")
        print(f"  成功台站: {stats['successful_stations']:,}")
        print(f"  失败台站: {stats['failed_stations']:,}")
        print(f"  拒绝残缺响应: {stats.get('incomplete_response_rejected', 0):,}")
        print(f"  拒绝异常振幅: {stats.get('amplitude_qc_rejected', 0):,}")
        print(f"  拒绝残缺ZNE: {stats.get('incomplete_zne_rejected', 0):,}")
        print(f"  ZNE旋转: {stats['component_conversions']:,} 次")
        print(f"  简化响应去除: {stats['simplified_response_removal']:,} 次")

        if preprocessor.config.performance['suppress_all_warnings']:
            print(
                f"  抑制警告: "
                f"{stats['warning_stats']['response_warnings_suppressed']:,} 次"
            )

        if stats['total_stations'] > 0:
            success_rate = stats['successful_stations'] / stats['total_stations'] * 100
            print(f"  成功率: {success_rate:.1f}%")

        print(f"⏱️ 总耗时: {results['total_time_seconds']/60:.2f} 分钟")
        print(f"📋 详细报告: {results['report_file']}")
        print(f"📁 输出目录: {preprocessor.output_dirs['disp_data']}")

    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断处理")
        print("💡 提示：下次运行时将自动从中断处继续（或加 --force 全量重跑）")
    except Exception as e:
        print(f"\n❌ 处理过程出错: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)


if __name__ == "__main__":
    main()