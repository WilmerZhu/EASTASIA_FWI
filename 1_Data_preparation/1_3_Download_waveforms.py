"""
1_3_Download_waveforms.py: 
地震波形数据批量下载模块
================================================================

功能描述:
----------
基于FDSN (Federation of Digital Seismograph Networks) 协议批量下载东亚地区地震事件的波形数据。
支持多Provider智能轮询、永久台站筛选、断点续传和数据质量控制，为全波形反演提供高质量的观测数据。

核心功能:
----------
1. ✅ 批量波形数据下载
   - 从GCMT事件目录自动生成下载任务
   - 支持11个主要FDSN Provider (AUSPASS, ETH, GEOFON, GEONET, GFZ, IPGP, IRIS, RESIF, SCEDC, USP)
   - 智能Provider轮询机制，提高下载成功率
   - 自动重试机制（每轮尝试所有Provider）

2. ✅ 永久台站筛选（新增）
   - 支持仅下载永久台站的波形数据
   - 基于永久台站列表文件筛选
   - 可指定网络和台站列表
   - 提高数据质量和一致性

3. ✅ 时间窗口配置
   - 可配置事件前后时间窗口（默认: 事件后60分钟）
   - 最小数据长度比例要求
   - 自动处理数据间隙

4. ✅ 通道优先级
   - 优先下载宽频带数据 (BH*, HH*)
   - 位置代码优先级 (空, 00, 10, 20)
   - 自动选择最佳数据质量

5. ✅ 断点续传功能
   - 自动检测已下载的数据
   - 跳过已完成的事件
   - 支持中断后继续下载

6. ✅ 数据质量控制
   - 最小文件数量验证
   - 数据完整性检查
   - 详细的下载统计报告

使用方法:
----------
```python
from 1_Data_preparation.1_3_Download_waveforms import WaveformDownloader

# 初始化下载器
downloader = WaveformDownloader()

# 选择事件目录文件
catalog_file = downloader.select_catalog_source()

# 批量下载波形数据
results = downloader.batch_download(catalog_file)

# 使用永久台站模式（test_database/精选目录交互时选 1；选 2 为不限制台站）
downloader.load_permanent_stations()
results = downloader.batch_download(catalog_file)
```

配置说明:
----------
通过 DownloadConfig 类配置下载参数:
- fdsn: FDSN服务配置 (providers, timeout, channel优先级等)
- time_window: 时间窗口配置 (事件前后时间)
- event_selection: 事件选择条件 (震级范围、时间范围、最大事件数)
- permanent_stations: 永久台站配置 (台站文件、网络列表等)
- retry: 重试机制配置
- quality_control: 数据质量控制阈值

输出文件:
----------
- events/YYYYMMDD_download_NN/waveforms/{event_dir}/: 单次下载会话内各事件波形（NN 为当日起流水号）
- events/YYYYMMDD_download_NN/responses/{event_dir}/: 同会话仪器响应
- events/YYYYMMDD_download_NN/download_progress.json: 本会话断点进度（完成后删除；重启程序时可选择沿用该会话以续跑）
- events/YYYYMMDD_download_NN/download_report.txt: 本会话报告
- events/download_catalog*.par: 在 events 根目录生成的目录文件（选项 1）

科学原理:
----------
- FDSN协议: 国际标准的地震数据交换协议，确保数据格式统一和互操作性
- 多Provider下载: 提高数据覆盖率和可用性，减少单点故障影响
- 永久台站: 长期运行的台站数据质量更稳定，适合全波形反演
- 时间窗口: 确保包含完整的P波、S波和面波信号，满足全波形反演需求
- 通道优先级: 宽频带数据提供更宽的频率范围，提高反演分辨率

作者: EASTASIA-FWI Team
日期: 2026-02-02
版本: v2.0 (支持永久台站)
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from datetime import datetime
import logging
import time
import json
import re
import shutil
import warnings
import sys

warnings.filterwarnings("ignore")

# ObsPy imports
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
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
        # FDSN 服务：永久台按「对东亚/西太有波形」选源，不是按台站 CSV 的 Provider 列
        # （CSV 里 204/219 条元数据来自 IRIS，但 JP/TW/GE/G/AU/NZ 的波形常在原台网节点）
        self.fdsn = {
            'providers': [
                'IRIS', 'GEOFON', 'GFZ', 'IESDMC', 'IPGP',
                'AUSPASS', 'GEONET', 'ORFEUS', 'ETH', 'BGR',
                'RESIF', 'SCEDC', 'USP',
            ],
            # 永久台模式：排除加州/巴西/RESIF（无东亚波形，且易 400/重置）
            'permanent_providers': [
                'IRIS',      # GSN / IC / 多数元数据与波形
                'GEOFON',    # GE 台网主归档
                'GFZ',       # 与 GEOFON 部分重叠，可补漏
                'IESDMC',    # 台湾 BATS (TW)
                'IPGP',      # GEOSCOPE (G)
                'AUSPASS',   # 澳大利亚 AU / S1
                'GEONET',    # 新西兰 NZ
                'ORFEUS',    # EIDA 路由，补欧洲镜像上的 GE 等
                'ETH',
                'BGR',
            ],
            'network': "*",
            'timeout': 300,  # 超时时间（秒）
            'channel_priorities': ["BH*", "HH*",],
            'location_priorities': ["", "00", "10", "20"],
            'min_interstation_distance': 1000,  # 永久台：只丢掉同址重复探头
            # 全网：不抽稀。密集台阵留给 1_8 再筛
            'all_stations_min_interstation_m': 0,
        }
        
        # 时间窗口配置（震前 padding 供去响应 taper / 预滤波稳定）
        self.time_window = {
            'pre_event_minutes': 5.0,
            'post_event_minutes': 60.0,
            'minimum_length_ratio': 0.9,           # 永久台：接近整窗
            'all_stations_minimum_length_ratio': 0.5,  # 全网：短记录也收下，交给 1_8
        }
        
        # 事件选择条件
        self.event_selection = {
            'magnitude_range': [6.0, 7.0],         # 震级范围
            'time_range': ['2010-01-01', '2024-12-31'],  # 时间范围
            'max_events': 100,                       # 最大事件数
            'spatial_decimation': True,            # 启用空间去重
            'min_distance_km': 100                 # 最小事件间距（km）
        }
        
        # 永久台站下载配置
        # 清单筛选只保证「长期在网」，不保证单事件波形可用；
        # 正演台站应先全网下载，再由 1_8 按 SNR/完整性筛选。
        self.permanent_stations = {
            'use_permanent_only': True,           # 交互默认；CLI --all-stations 可覆盖
            'station_file': 'EastAsia_permanent_stations_filtered.csv',
            'network_list': None,
            'station_list': None,
        }
        
        # 重试机制配置
        self.retry = {
            'max_retries': 3,          # 最大重试轮数（每轮尝试所有Provider）
            'base_delay': 3,           # 基础延迟（秒）
            'timeout_delay': 8,        # 超时后延迟（秒）
            'server_error_delay': 15,  # 服务器错误延迟（秒）
            'provider_network_retries': 1,  # 同一数据源连接重置后再试次数
        }
        
        # 数据质量控制
        # 永久台：相对清单的覆盖率（fill=继续补漏，accept=失败下限）
        # 全网：不设文件数目标；扫完区域源即结束，有波形就算成功。
        #       台站好坏由 1_4 / 1_8 按波形筛，下载阶段尽量多收。
        self.quality_control = {
            'permanent_channel_factor': 3.0,
            'permanent_min_components': 3,
            'permanent_fill_frac': 0.90,
            'permanent_fill_complete_frac': 0.80,
            'permanent_accept_frac': 0.35,
            'permanent_coverage_frac': 0.35,
            'permanent_min_mseed_floor': 60,
            'permanent_min_xml_floor': 20,
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
    - 断点续传：进度在会话目录的 download_progress.json；再次运行时可选择沿用未完成会话或新建 YYYYMMDD_download_NN
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
    
    def __init__(
        self,
        config_file: Optional[Path] = None,
        station_mode: Optional[str] = None,
    ):
        """
        初始化波形下载器

        Args:
            config_file: 自定义配置文件路径
            station_mode: 'permanent' / 'all'；None 则跟随配置默认
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

        self.processing_start_time = None
        
        # 初始化数据存储
        self._initialize_storage()
        
        # 创建输出目录
        self._setup_output_directories()
        
        # 初始化Provider轮询状态
        self.provider_index = 0
        
        # 永久台站数据
        self.permanent_stations_df = None
        self.use_permanent_stations = False
        if station_mode == 'all':
            self.use_permanent_stations = False
            self.permanent_stations_df = None
        elif station_mode == 'permanent':
            self.load_permanent_stations()
        else:
            self._apply_permanent_station_config_default()
        
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

    def _setup_output_directories(self):
        """设置并创建输出目录（波形/响应目录在每次 batch_download 会话中创建）"""
        self.output_paths = self._get_output_paths()
        events_base = self.output_paths['events_base']
        events_base.mkdir(parents=True, exist_ok=True)
        catalog_file = self.output_paths['catalog_file']
        catalog_file.parent.mkdir(parents=True, exist_ok=True)
        if catalog_file.exists() and catalog_file.is_dir():
            shutil.rmtree(catalog_file)

    def _get_output_paths(self) -> Dict[str, Path]:
        """获取输出路径（会话开始前 waveforms/responses 为占位，batch_download 时会更新）"""
        events_base = self.base_config.dirs['data'] / 'events'
        return {
            'events_base': events_base,
            'session_dir': events_base,
            'waveforms': events_base / 'waveforms',
            'responses': events_base / 'responses',
            'catalog_file': events_base / 'download_catalog.par',
            'progress_file': events_base / 'download_progress.json',
            'report_file': events_base / 'download_report.txt',
        }

    def _begin_download_session(self) -> Path:
        """
        为本次 batch_download 分配 data/events/YYYYMMDD_download_NN/，
        并在其下创建 waveforms、responses，更新进度与报告路径。
        """
        events_base = self.output_paths['events_base']
        events_base.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        pat = re.compile(rf"^{re.escape(date_str)}_download_(\d+)$")
        max_seq = 0
        for child in events_base.iterdir():
            if not child.is_dir():
                continue
            m = pat.match(child.name)
            if m:
                max_seq = max(max_seq, int(m.group(1)))
        seq = max_seq + 1
        # 与首日首次示例一致：01, 02, …；序号 ≥100 时自然变为三位
        session_dir = events_base / f"{date_str}_download_{seq:02d}"
        session_dir.mkdir(parents=True, exist_ok=True)
        waveforms = session_dir / 'waveforms'
        responses = session_dir / 'responses'
        waveforms.mkdir(exist_ok=True)
        responses.mkdir(exist_ok=True)
        self.output_paths['session_dir'] = session_dir
        self.output_paths['waveforms'] = waveforms
        self.output_paths['responses'] = responses
        self.output_paths['progress_file'] = session_dir / 'download_progress.json'
        self.output_paths['report_file'] = session_dir / 'download_report.txt'
        self.logger.info(f"📁 本次下载会话目录: {session_dir}")
        return session_dir

    _SESSION_DIR_NAME_RE = re.compile(r"^\d{8}_download_\d+$")

    def _attach_existing_session(self, session_dir: Path) -> None:
        """沿用已有会话目录，不新建 YYYYMMDD_download_NN。"""
        session_dir = Path(session_dir).resolve()
        if not session_dir.is_dir():
            raise FileNotFoundError(f"会话目录不存在: {session_dir}")
        waveforms = session_dir / "waveforms"
        responses = session_dir / "responses"
        waveforms.mkdir(parents=True, exist_ok=True)
        responses.mkdir(parents=True, exist_ok=True)
        self.output_paths["session_dir"] = session_dir
        self.output_paths["waveforms"] = waveforms
        self.output_paths["responses"] = responses
        self.output_paths["progress_file"] = session_dir / "download_progress.json"
        self.output_paths["report_file"] = session_dir / "download_report.txt"

    def _peek_session_progress(self, progress_path: Path) -> Dict[str, Any]:
        """读取进度 JSON 摘要（无副作用）。"""
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _list_resumable_session_dirs(self) -> List[Path]:
        """列出含 download_progress.json 的 YYYYMMDD_download_NN 目录，按进度文件修改时间降序。"""
        events_base = self.output_paths["events_base"]
        if not events_base.is_dir():
            return []
        found: List[Tuple[float, Path]] = []
        for p in events_base.iterdir():
            if not p.is_dir() or not self._SESSION_DIR_NAME_RE.match(p.name):
                continue
            prog = p / "download_progress.json"
            if prog.is_file():
                try:
                    mtime = prog.stat().st_mtime
                except OSError:
                    continue
                found.append((mtime, p))
        found.sort(key=lambda x: -x[0])
        return [x[1] for x in found]

    def _interactive_pick_resume_session(
        self, catalog_file: Path, catalog_line_count: int
    ) -> Optional[Path]:
        """
        若存在未完成会话，询问沿用或新建。

        Returns:
            选择沿用的会话目录，或 None 表示新建会话。
        """
        if not sys.stdin.isatty() or getattr(self, 'auto_confirm', False):
            return None
        candidates = self._list_resumable_session_dirs()
        if not candidates:
            return None
        cur_cat = str(catalog_file.resolve())
        print("\n♻️ 发现未完成的下载会话（含 download_progress.json）:")
        print("  0) 新建会话（分配新的 YYYYMMDD_download_NN）")
        for idx, sd in enumerate(candidates, 1):
            pdata = self._peek_session_progress(sd / "download_progress.json")
            saved_cat = pdata.get("catalog_path", "")
            match = "✓" if saved_cat and saved_cat == cur_cat else "✗"
            nxt = pdata.get("next_event_index")
            if nxt is None:
                nxt = pdata.get("current_index", "?")
            n_total = pdata.get("catalog_event_count", "?")
            ts = pdata.get("timestamp", "")
            print(f"  {idx}) {sd.name}")
            print(
                f"      下一事件序号: {nxt} / 进度内总行数: {n_total}  |  "
                f"与当前 .par 路径一致: {match}"
            )
            if saved_cat:
                print(f"      进度记录目录: {saved_cat}")
            if ts:
                print(f"      保存时间: {ts}")
        while True:
            choice = input(
                f"\n请选择 0-{len(candidates)} (0=新建会话): "
            ).strip()
            if choice == "0":
                return None
            if not choice:
                print("❌ 请输入 0（新建会话）或 1–N（沿用对应会话）")
                continue
            try:
                k = int(choice)
            except ValueError:
                print("❌ 请输入数字")
                continue
            if 1 <= k <= len(candidates):
                picked = candidates[k - 1]
                pdata = self._peek_session_progress(picked / "download_progress.json")
                saved_cat = pdata.get("catalog_path", "")
                if saved_cat and saved_cat != cur_cat:
                    print(
                        "\n⚠️ 该会话进度关联的目录文件与当前选择不同。"
                        f"\n   进度内: {saved_cat}"
                        f"\n   当前:   {cur_cat}"
                    )
                    ok = input("仍沿用此会话目录续跑? (y/N): ").strip().lower()
                    if ok != "y":
                        print("已取消沿用，请重新选择。")
                        continue
                saved_n = pdata.get("catalog_event_count")
                if (
                    isinstance(saved_n, int)
                    and saved_n != catalog_line_count
                ):
                    print(
                        f"\n⚠️ 当前 .par 有效行数 ({catalog_line_count}) 与进度内记录 ({saved_n}) 不一致，续跑可能错位。"
                    )
                    ok2 = input("仍继续? (y/N): ").strip().lower()
                    if ok2 != "y":
                        continue
                return picked
            print(f"❌ 请输入 0 到 {len(candidates)} 之间的整数")

    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 波形数据下载配置摘要")
        print("-" * 60)
        
        # 研究区域配置
        print(f"研究区域: {self.base_config.region['name']}")
        region = self.base_config.region
        print(f"纬度范围: {region['lat_min']}° ~ {region['lat_max']}°")
        print(f"经度范围: {region['lon_min']}° ~ {region['lon_max']}°")
        
        # FDSN配置（永久台用东亚/西太源；全网模式另含 RESIF/SCEDC/USP）
        providers = self._provider_list()
        print(f"FDSN Providers: {len(providers)} 个")
        print(f"Provider列表: {', '.join(providers)}")
        print(f"查询超时: {self.config.fdsn['timeout']} 秒")
        
        # 事件选择配置
        mag_range = self.config.event_selection['magnitude_range']
        print(f"震级范围: {mag_range[0]} ~ {mag_range[1]}")
        print(f"最大事件数: {self.config.event_selection['max_events']}")
        
        # 时间窗口配置
        time_win = self.config.time_window
        print(f"时间窗口: 前{time_win['pre_event_minutes']}分钟 ~ 后{time_win['post_event_minutes']}分钟")

        # 永久台站模式（由 permanent_stations['use_permanent_only'] 或交互覆盖）
        cfg_flag = bool(self.config.permanent_stations.get('use_permanent_only', False))
        if self.use_permanent_stations and self.permanent_stations_df is not None:
            print(f"台站模式: 仅永久台站 "
                  f"({len(self.permanent_stations_df)} 台, "
                  f"文件={self.config.permanent_stations.get('station_file')})")
        elif cfg_flag:
            print("台站模式: 配置要求仅永久台站，但清单尚未加载成功 → 将回退为不限制")
        else:
            print("台站模式: 不限制（network=*，含临时台阵）")
        
        # 输出配置（具体会话子目录在每次 batch_download 开始时创建）
        print(f"事件根目录: {self.output_paths['events_base']}")
        print(f"波形/响应: <事件根>/YYYYMMDD_download_NN/{{waveforms,responses}}/")
        
        print("-" * 60)

    def _configure_obspy_logging(self):
        """
        配置 ObsPy / MassDownloader 日志。

        ObsPy MassDownloader(configure_logging=True) 每次实例化都会
        logger.addHandler(StreamHandler)，多 Provider 轮询后同一条 INFO
        会打印几十遍。此处统一关 ObsPy 自带 handler，改由本模块控制级别。
        """
        level_name = str(self.config.logging.get('obspy_log_level', 'WARNING')).upper()
        level = getattr(logging, level_name, logging.WARNING)

        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            if not isinstance(logger_name, str):
                continue
            if not logger_name.startswith('obspy'):
                continue
            child_logger = logging.getLogger(logger_name)
            child_logger.handlers.clear()
            child_logger.setLevel(level)
            child_logger.propagate = False

        # 保证 mass_downloader 日志器已创建并清干净（即使尚未被 import 使用）
        md_logger = logging.getLogger('obspy.clients.fdsn.mass_downloader')
        md_logger.handlers.clear()
        md_logger.setLevel(level)
        md_logger.propagate = False

    def _provider_list(self) -> List[str]:
        """
        返回本次实际轮询的 FDSN 数据源。

        东亚/西太波形用 regional 节点（permanent_providers）；
        全网模式同样走这组，不查 SCEDC/USP/RESIF。
        """
        plist = self.config.fdsn.get('permanent_providers')
        if plist:
            return list(plist)
        return list(self.config.fdsn['providers'])

    def _count_active_permanent_stations(
        self, origin_time: Optional[UTCDateTime] = None
    ) -> int:
        """
        统计发震时刻仍在运行的永久台数量。

        清单含 1990s–2020s 陆续建站的台；2006 年事件若按全部 219 台设门槛会虚高。
        StartDate/EndDate 缺失视为该端无约束。
        """
        df = self._active_permanent_frame(origin_time)
        if df is None or df.empty:
            return 0
        return int(len(df))

    def _quality_thresholds(
        self, origin_time: Optional[UTCDateTime] = None
    ) -> Dict[str, Any]:
        """
        返回接受下限 / 补全目标。

        永久台按发震时刻在网台数缩放。min_* 是全部轮次结束后的失败下限；
        fill_* 是提前结束多源轮询的补全目标（MassDownloader 单次易漏台）。
        """
        qc = self.config.quality_control
        if (self.use_permanent_stations
                and self.permanent_stations_df is not None
                and len(self.permanent_stations_df) > 0):
            n_sta = self._count_active_permanent_stations(origin_time)
            accept_frac = float(qc.get(
                'permanent_accept_frac', qc.get('permanent_coverage_frac', 0.35)
            ))
            fill_frac = float(qc.get('permanent_fill_frac', 0.90))
            complete_frac = float(qc.get('permanent_fill_complete_frac', 0.80))
            ch = float(qc.get('permanent_channel_factor', 3.0))
            accept_sta = max(1, int(n_sta * accept_frac))
            fill_sta = max(accept_sta, int(n_sta * fill_frac))
            min_mseed = max(
                int(qc.get('permanent_min_mseed_floor', 60)),
                int(n_sta * accept_frac * ch),
            )
            min_xml = max(
                int(qc.get('permanent_min_xml_floor', 20)),
                accept_sta,
            )
            return {
                'n_active': n_sta,
                'accept_frac': accept_frac,
                'fill_frac': fill_frac,
                'complete_frac': complete_frac,
                'accept_stations': accept_sta,
                'fill_stations': fill_sta,
                'min_mseed_files': min_mseed,
                'min_xml_files': min_xml,
                'min_total_files': min_mseed + min_xml,
                'fill_mseed_files': int(n_sta * fill_frac * ch),
                'fill_xml_files': fill_sta,
            }
        return {
            'n_active': 0,
            'accept_frac': 0.0,
            'fill_frac': 1.0,
            'complete_frac': 1.0,
            'accept_stations': 0,
            'fill_stations': 0,
            'min_mseed_files': 1,
            'min_xml_files': 0,
            'min_total_files': 1,
            'fill_mseed_files': 0,
            'fill_xml_files': 0,
        }

    @staticmethod
    def _unwrap_exception(exc: Exception) -> Exception:
        """展开异常链并返回最内层异常，便于准确分类网络错误。"""
        current: Exception = exc
        visited = set()
        while True:
            next_exc = getattr(current, '__cause__', None) or getattr(current, '__context__', None)
            if next_exc is None:
                return current
            if id(next_exc) in visited:
                return current
            visited.add(id(next_exc))
            if not isinstance(next_exc, Exception):
                return current
            current = next_exc

    def _classify_download_exception(self, exc: Exception) -> Tuple[str, bool]:
        """
        对下载异常进行分类。

        Returns:
            (错误摘要, 是否属于网络可重试错误)
        """
        root_exc = self._unwrap_exception(exc)
        msg = str(exc) if str(exc) else repr(exc)
        root_msg = str(root_exc) if str(root_exc) else repr(root_exc)

        network_error_types = (
            ConnectionResetError,
            TimeoutError,
            ConnectionError,
            BrokenPipeError,
        )
        if isinstance(root_exc, network_error_types):
            return f"{type(root_exc).__name__}: {root_msg}", True

        combined = f"{msg} {root_msg} {type(exc).__name__} {type(root_exc).__name__}"
        # ObsPy/requests 某些版本在日志格式化阶段会把 ConnectionResetError 二次包装成 splitlines 异常
        if 'splitlines' in combined.lower():
            return (
                "ObsPy日志格式化触发 splitlines 异常（底层多为连接重置，按可重试网络错误处理）",
                True,
            )
        if 'ConnectionResetError' in combined:
            return f"ConnectionResetError: {root_msg}", True

        return f"{type(exc).__name__}: {msg}", False

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

    def _apply_permanent_station_config_default(self) -> None:
        """
        按 DownloadConfig.permanent_stations['use_permanent_only'] 设置默认台站模式。

        注意：此前该开关只存在于配置字典，从未写入 self.use_permanent_stations，
        导致改 True 后 hybrid 目录仍走 network=* 全网下载。
        """
        want = bool(self.config.permanent_stations.get('use_permanent_only', False))
        if not want:
            self.use_permanent_stations = False
            self.permanent_stations_df = None
            return
        try:
            self.load_permanent_stations()
        except Exception as e:
            self.logger.warning(f"⚠️  按配置加载永久台站失败，回退不限制: {e}")
            self.use_permanent_stations = False
            self.permanent_stations_df = None

    def _should_offer_permanent_station_mode(self, catalog_file: Path) -> bool:
        """精选/测试库/hybrid 目录询问台站范围（永久清单 vs 研究区全网）。"""
        name_low = catalog_file.name.lower()
        if any(key in name_low for key in (
            'selected_events', 'hybrid', 'test_database',
        )):
            return True
        parts_lower = [p.lower() for p in catalog_file.parts]
        return any(p in parts_lower for p in ('test_database', 'hybrid_30events',
                                              'hybrid_10events'))

    def _offer_permanent_stations_interactive(self) -> None:
        """
        交互选择台站范围。

        使用 1/2 明确区分，避免将输入「1」误当作 Y/n 中的肯定答复而强制永久台站模式。
        默认提示跟随 DownloadConfig.permanent_stations['use_permanent_only']。
        """
        cfg_default = 1 if self.config.permanent_stations.get(
            'use_permanent_only', False) else 2
        print("\n台站范围（测试库/精选目录）:")
        print("  1) 仅下载永久台站清单内台站（见 DownloadConfig.permanent_stations）")
        print("  2) 不限制台站 — 使用研究区内 FDSN 默认规则（network *、最小台间距等）")
        print(f"  （配置默认: {cfg_default}；直接回车采用默认）")
        while True:
            raw = input("请选择 1 或 2: ").strip()
            choice = raw if raw else str(cfg_default)
            if choice == "2":
                self.use_permanent_stations = False
                self.permanent_stations_df = None
                print("✅ 已选：不限制台站清单")
                return
            if choice == "1":
                break
            print("❌ 请输入 1 或 2（或回车用配置默认）")
        try:
            self.load_permanent_stations()
            if self.permanent_stations_df is not None:
                print(f"✅ 已加载 {len(self.permanent_stations_df)} 个永久台站")
            else:
                print("⚠️  永久台站数据加载失败")
        except Exception as e:
            print(f"⚠️  加载永久台站失败: {e}")
            print("   将使用不限制台站清单模式")
            self.use_permanent_stations = False
            self.permanent_stations_df = None

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

    @staticmethod
    def _origin_utcdatetime(
        year: int, month: int, day: int, hour: int, minute: int, second: float
    ) -> UTCDateTime:
        """由年月日时分及分数秒构造 UTCDateTime（ObsPy 的 datetime 后端要求秒为整数 + 微秒）。"""
        sec_whole = int(second)
        micro = int(round((second - sec_whole) * 1e6))
        if micro >= 1_000_000:
            sec_whole += 1
            micro -= 1_000_000
        elif micro < 0:
            sec_whole -= 1
            micro += 1_000_000
        return UTCDateTime(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=sec_whole,
            microsecond=micro,
        )

    def parse_event(self, event_line: str) -> Optional[Dict]:
        """
        解析 .par 事件行（MassDownloader / 本仓库统一格式）

        标准格式（与 download_catalog_1113_test.par、generate_download_catalog 一致）::
            事件目录名 YYYYMMDD HH MM SS[.sss] lat lon depth ... Mag ML
            例: 20140313.17.06 20140313 17 06 50.800 33.6800 131.8200 79.0 ...

        兼容旧版测试库行（CMT 事件 ID 前缀）::
            C200801090826A lat lon depth mag year month day hour min sec.ss

        Args:
            event_line: 事件行字符串

        Returns:
            事件信息字典，解析失败时返回 None
        """
        try:
            parts = event_line.split()
            if len(parts) < 8:
                raise ValueError("列数不足")

            # 标准行：第 2 列为 8 位日期
            if len(parts[1]) == 8 and parts[1].isdigit():
                ymd = parts[1]
                y, mo, d = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8])
                h, mi = int(parts[2]), int(parts[3])
                s = float(parts[4])
                event_info = {
                    'dir': parts[0],
                    'ymd': ymd,
                    'hour': parts[2],
                    'min': parts[3],
                    'sec': parts[4],
                    'lat': float(parts[5]),
                    'lon': float(parts[6]),
                    'depth': float(parts[7]),
                    'origin_time': self._origin_utcdatetime(y, mo, d, h, mi, s),
                }
            # 旧版：CMT 事件名 + 经纬深 + 震级 + 年月日时分秒
            elif len(parts) >= 11 and parts[0].startswith('C'):
                lat, lon, dep = float(parts[1]), float(parts[2]), float(parts[3])
                y = int(parts[5])
                mo = int(parts[6])
                d = int(parts[7])
                h = int(parts[8])
                mi = int(parts[9])
                s = float(parts[10])
                ymd = f"{y:04d}{mo:02d}{d:02d}"
                event_info = {
                    'dir': parts[0],
                    'ymd': ymd,
                    'hour': f"{h:02d}",
                    'min': f"{mi:02d}",
                    'sec': str(s),
                    'lat': lat,
                    'lon': lon,
                    'depth': dep,
                    'origin_time': self._origin_utcdatetime(y, mo, d, h, mi, s),
                }
            else:
                raise ValueError("无法识别行格式")

            if not (-90 <= event_info['lat'] <= 90):
                raise ValueError(f"纬度超出范围: {event_info['lat']}")
            if not (-180 <= event_info['lon'] <= 180):
                raise ValueError(f"经度超出范围: {event_info['lon']}")
            if event_info['depth'] < 0:
                raise ValueError(f"深度不能为负数: {event_info['depth']}")

            return event_info

        except Exception as e:
            self.logger.debug(f"解析事件数据失败: {event_line}, 错误: {e}")
            return None

    def check_event_exists(
        self, event_dir: str, origin_time: Optional[UTCDateTime] = None
    ) -> bool:
        """
        事件是否已达补全目标（可跳过）。

        永久台看清单覆盖。全网模式永不因文件数跳过——MassDownloader
        对已有文件是增量写入，重跑只会补漏。
        """
        if not self.use_permanent_stations:
            return False
        cov = self._event_coverage(event_dir, origin_time)
        return bool(cov['fill_ok'])

    def _get_next_provider(self) -> str:
        """获取下一个Provider"""
        providers = self._provider_list()
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

    def _count_event_files(self, event_dir: str) -> Tuple[int, int]:
        """返回事件目录中的 (mseed 数, xml 数)。"""
        data_path = self.output_paths['waveforms'] / event_dir
        response_path = self.output_paths['responses'] / event_dir
        mseed_count = len(list(data_path.glob("*.mseed"))) if data_path.exists() else 0
        xml_count = len(list(response_path.glob("*.xml"))) if response_path.exists() else 0
        return mseed_count, xml_count

    @staticmethod
    def _parse_mseed_netsta_channel(filename: str) -> Optional[Tuple[str, str, str]]:
        """从 MassDownloader 文件名解析 (network, station, channel)。"""
        head = Path(filename).name.split('__')[0]
        parts = head.split('.')
        if len(parts) >= 4:
            return parts[0], parts[1], parts[3]
        if len(parts) >= 2:
            return parts[0], parts[1], ''
        return None

    def _downloaded_channels(self, event_dir: str) -> Dict[str, Set[str]]:
        """NET.STA -> 已下载通道代码集合。"""
        data_path = self.output_paths['waveforms'] / event_dir
        result: Dict[str, Set[str]] = {}
        if not data_path.exists():
            return result
        for path in data_path.glob("*.mseed"):
            parsed = self._parse_mseed_netsta_channel(path.name)
            if parsed is None:
                continue
            network, station, channel = parsed
            sid = f"{network}.{station}"
            result.setdefault(sid, set()).add(channel)
        return result

    def _downloaded_xml_ids(self, event_dir: str) -> Set[str]:
        """已有 StationXML 的 NET.STA 集合。"""
        response_path = self.output_paths['responses'] / event_dir
        ids: Set[str] = set()
        if not response_path.exists():
            return ids
        for path in response_path.glob("*.xml"):
            parts = path.stem.split('.')
            if len(parts) >= 2:
                ids.add(f"{parts[0]}.{parts[1]}")
        return ids

    def _active_station_ids(self, origin_time: Optional[UTCDateTime] = None) -> Set[str]:
        """发震时刻在网永久台的 NET.STA 集合。"""
        df = self._active_permanent_frame(origin_time)
        if df is None or df.empty:
            return set()
        if 'StationID' in df.columns:
            ids = {
                str(item).strip()
                for item in df['StationID'].dropna()
                if str(item).strip()
            }
            if ids:
                return ids
        return {
            f"{str(row['Network']).strip()}.{str(row['Station']).strip()}"
            for _, row in df.iterrows()
        }

    def _event_coverage(
        self, event_dir: str, origin_time: Optional[UTCDateTime] = None
    ) -> Dict[str, Any]:
        """
        统计相对永久台清单的覆盖情况。

        fill_ok: 达到补全目标，可停止继续请求。
        accept_ok: 达到失败下限；全部轮次结束后仍不足则判失败。
        missing_ids: 缺波形、缺 xml、或不足三分量的台，供后续轮次定向补拉。
        """
        n_mseed, n_xml_files = self._count_event_files(event_dir)
        thr = self._quality_thresholds(origin_time)

        if not (self.use_permanent_stations and self.permanent_stations_df is not None):
            n_sta = len(self._downloaded_channels(event_dir))
            # 全网不设「够了」：fill_ok 恒为 False，避免跳过事件或提前停源
            return {
                'n_active': 0,
                'n_present': n_sta,
                'n_complete': 0,
                'n_mseed': n_mseed,
                'n_xml': n_xml_files,
                'present_frac': 0.0,
                'complete_frac': 0.0,
                'accept_ok': n_mseed > 0,
                'fill_ok': False,
                'accept_frac': 0.0,
                'fill_frac': 1.0,
                'missing_ids': set(),
                'thresholds': thr,
            }

        qc = self.config.quality_control
        min_comp = int(qc.get('permanent_min_components', 3))
        active = self._active_station_ids(origin_time)
        channels = self._downloaded_channels(event_dir)
        xml_ids = self._downloaded_xml_ids(event_dir)
        n_active = len(active)

        present: Set[str] = set()
        complete: Set[str] = set()
        missing: Set[str] = set()
        for sid in active:
            chans = channels.get(sid, set())
            has_xml = sid in xml_ids
            letters = {code[-1].upper() for code in chans if code}
            if chans:
                present.add(sid)
            if chans and has_xml and len(letters) >= min_comp:
                complete.add(sid)
            else:
                missing.add(sid)

        present_frac = (len(present) / n_active) if n_active else 0.0
        xml_frac = (len(xml_ids & active) / n_active) if n_active else 0.0
        complete_frac = (len(complete) / n_active) if n_active else 0.0
        accept_frac = float(thr['accept_frac'])
        fill_frac = float(thr['fill_frac'])
        need_complete = float(thr['complete_frac'])

        accept_ok = (
            n_active > 0
            and present_frac >= accept_frac
            and xml_frac >= accept_frac
        )
        fill_ok = (
            n_active > 0
            and present_frac >= fill_frac
            and xml_frac >= fill_frac
            and complete_frac >= need_complete
        )

        return {
            'n_active': n_active,
            'n_present': len(present),
            'n_complete': len(complete),
            'n_mseed': n_mseed,
            'n_xml': n_xml_files,
            'present_frac': present_frac,
            'complete_frac': complete_frac,
            'accept_ok': accept_ok,
            'fill_ok': fill_ok,
            'accept_frac': accept_frac,
            'fill_frac': fill_frac,
            'missing_ids': missing,
            'thresholds': thr,
        }

    def _format_coverage(self, cov: Dict[str, Any]) -> str:
        """覆盖率摘要，供日志使用。"""
        if cov['n_active'] <= 0:
            return (
                f"台站 {cov['n_present']}, mseed={cov['n_mseed']}, "
                f"xml={cov['n_xml']}（全网尽量收齐，不设文件数门槛）"
            )
        n_miss = len(cov['missing_ids'])
        return (
            f"台站 {cov['n_present']}/{cov['n_active']} ({cov['present_frac']:.0%}), "
            f"三分量 {cov['n_complete']} ({cov['complete_frac']:.0%}), "
            f"待补 {n_miss}, mseed={cov['n_mseed']}, xml={cov['n_xml']} "
            f"| 补全目标 ≥{cov['fill_frac']:.0%} 台且三分量 ≥{cov['thresholds']['complete_frac']:.0%} "
            f"| 失败下限 ≥{cov['accept_frac']:.0%} 台"
        )

    def _active_permanent_frame(
        self, origin_time: Optional[UTCDateTime] = None
    ) -> Optional[pd.DataFrame]:
        """按发震时刻筛出当时在网的永久台。"""
        df = self.permanent_stations_df
        if df is None or df.empty:
            return df
        if origin_time is None:
            return df
        try:
            t = pd.Timestamp(origin_time.datetime)
        except Exception:
            return df
        mask = pd.Series(True, index=df.index)
        if 'StartDate' in df.columns:
            start = pd.to_datetime(df['StartDate'], errors='coerce')
            mask &= start.isna() | (start <= t)
        if 'EndDate' in df.columns:
            end = pd.to_datetime(df['EndDate'], errors='coerce')
            mask &= end.isna() | (end >= t)
        return df.loc[mask]

    def _station_restriction_chunks(
        self,
        origin_time: Optional[UTCDateTime] = None,
        event_dir: Optional[str] = None,
        missing_only: bool = False,
    ) -> List[Dict[str, str]]:
        """
        按台网拆成多次 FDSN 查询，每次只带该网在永久台清单中的台站代码。

        不可把 219 个台站一次塞进 Restrictions.station：IRIS 会 ConnectionReset。
        也不可只限制 network、不限制 station：JP 等台网的 HH* 会把 Hi-net 整网拉下来。
        missing_only=True 时只请求仍缺波形/xml/三分量的台，用于后续轮次补漏。
        """
        if not self.use_permanent_stations:
            return []
        df = self._active_permanent_frame(origin_time)
        if df is None or df.empty:
            return []

        if missing_only and event_dir:
            missing = self._event_coverage(event_dir, origin_time)['missing_ids']
            if not missing:
                return []
            netsta = df['Network'].astype(str).str.strip() + '.' + df['Station'].astype(str).str.strip()
            if 'StationID' in df.columns:
                sid = df['StationID'].astype(str).str.strip()
                df = df.loc[sid.isin(missing) | netsta.isin(missing)]
            else:
                df = df.loc[netsta.isin(missing)]
            if df.empty:
                return []

        chunks: List[Dict[str, str]] = []
        nets_acc: List[str] = []
        stas_acc: List[str] = []
        max_station_chars = 200  # 远小于原先 219 台拼接（约 988 字符）

        def flush() -> None:
            if not nets_acc:
                return
            chunks.append({
                'network': ','.join(nets_acc),
                'station': ','.join(stas_acc),
            })
            nets_acc.clear()
            stas_acc.clear()

        for network, grp in df.groupby('Network'):
            net = str(network).strip()
            if not net:
                continue
            stations = sorted({
                str(code).strip()
                for code in grp['Station'].dropna()
                if str(code).strip()
            })
            if not stations:
                continue
            station_param = ','.join(stations)
            if len(station_param) > max_station_chars:
                flush()
                step = 15
                for i in range(0, len(stations), step):
                    chunks.append({
                        'network': net,
                        'station': ','.join(stations[i:i + step]),
                    })
                continue
            candidate_len = len(','.join(stas_acc + stations)) if stas_acc else len(station_param)
            code_clash = bool(set(stations) & set(stas_acc))
            if stas_acc and (candidate_len > max_station_chars or code_clash):
                flush()
            nets_acc.append(net)
            stas_acc.extend(stations)
        flush()
        return chunks

    def _build_station_restrictions(self) -> Optional[Dict[str, str]]:
        """兼容旧接口：返回全部台网代码（不含台站列表）。"""
        chunks = self._station_restriction_chunks()
        if not chunks:
            return None
        networks = [chunk['network'] for chunk in chunks]
        return {'network': ','.join(sorted(set(networks)))}
        
    def download_event_data(self, event_info: Dict, retry_round: int = 1) -> Union[bool, str]:
        """
        下载单个事件的数据（支持永久台站限制）
        
        Args:
            event_info: 事件信息字典
            retry_round: 当前重试轮数
            
        Returns:
            True: 已达补全目标
            False: 有部分文件但未达补全目标（应继续其它源/下一轮）
            "no_data": 该源无可用数据
            "network_error": 连接重置/超时等，不应计为部分成功
        """
        event_dir = event_info['dir']
        origin_time = event_info['origin_time']
        providers = self._provider_list()
        
        try:
            # 获取当前Provider
            current_provider = providers[self.provider_index % len(providers)]
            
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
            
            # 定义限制条件（永久台按台网分批，避免超长 station= 列表）
            time_config = self.config.time_window
            if self.use_permanent_stations:
                min_len = float(time_config['minimum_length_ratio'])
                reject_gaps = True
            else:
                min_len = float(time_config.get(
                    'all_stations_minimum_length_ratio', 0.5))
                reject_gaps = False
            base_restrictions: Dict[str, Any] = {
                'starttime': origin_time - time_config['pre_event_minutes'] * 60,
                'endtime': origin_time + time_config['post_event_minutes'] * 60,
                'reject_channels_with_gaps': reject_gaps,
                'minimum_length': min_len,
                'channel_priorities': self.config.fdsn['channel_priorities'],
                'location_priorities': self.config.fdsn['location_priorities']
            }

            if self.use_permanent_stations:
                restriction_jobs: List[Optional[Dict[str, str]]] = list(
                    self._station_restriction_chunks(
                        origin_time,
                        event_dir=event_dir,
                        missing_only=(retry_round > 1),
                    )
                )
                if not restriction_jobs:
                    if retry_round > 1:
                        cov = self._event_coverage(event_dir, origin_time)
                        return True if cov['fill_ok'] else False
                    self.logger.warning("      ⚠️ 发震时刻无有效永久台，跳过该源")
                    return "no_data"
            else:
                restriction_jobs = [None]
            
            # 执行下载：把 timeout 传给 Client，避免 MassDownloader 使用默认 120s
            download_start = time.time()
            timeout = int(self.config.fdsn.get('timeout', 300))
            try:
                fdsn_client = Client(current_provider, timeout=timeout)
                providers_arg: List[Any] = [fdsn_client]
            except Exception as client_exc:
                self.logger.debug(
                    f"      Client({current_provider}) 初始化失败，回退名称: {client_exc}"
                )
                providers_arg = [current_provider]

            mdl = MassDownloader(
                providers=providers_arg,
                configure_logging=False,  # 避免每次实例化叠加 StreamHandler
            )

            saw_network_error = False
            for job in restriction_jobs:
                restrictions_params = dict(base_restrictions)
                if job is not None:
                    restrictions_params['network'] = job['network']
                    restrictions_params['station'] = job['station']
                else:
                    restrictions_params['network'] = self.config.fdsn['network']
                    restrictions_params['minimum_interstation_distance_in_m'] = (
                        int(self.config.fdsn.get(
                            'all_stations_min_interstation_m',
                            self.config.fdsn['min_interstation_distance'],
                        ))
                    )
                restrictions = Restrictions(**restrictions_params)
                try:
                    mdl.download(
                        domain,
                        restrictions,
                        mseed_storage=str(data_path),
                        stationxml_storage=str(response_path)
                    )
                except KeyboardInterrupt:
                    raise
                except ConnectionResetError as chunk_exc:
                    saw_network_error = True
                    net_label = job['network'] if job else '*'
                    self.logger.debug(
                        f"      {current_provider}/{net_label} 连接重置: {chunk_exc}"
                    )
                except Exception as chunk_exc:
                    error_msg, is_retryable = self._classify_download_exception(chunk_exc)
                    net_label = job['network'] if job else '*'
                    if is_retryable:
                        saw_network_error = True
                        self.logger.debug(
                            f"      {current_provider}/{net_label} 网络异常: {error_msg}"
                        )
                    else:
                        self.logger.debug(
                            f"      {current_provider}/{net_label} 子查询失败: {error_msg}"
                        )
            
            download_duration = time.time() - download_start
            time.sleep(1)

            mseed_count, xml_count = self._count_event_files(event_dir)
            if mseed_count > 0 or xml_count > 0:
                self.logger.debug(
                    f"      📥 下载完成 (mseed={mseed_count}, xml={xml_count}), "
                    f"耗时: {download_duration:.1f}秒"
                )
                cov = self._event_coverage(event_dir, origin_time)
                return True if cov['fill_ok'] else False
            if saw_network_error:
                return "network_error"
            return "no_data"
                
        except KeyboardInterrupt:
            raise
        except ConnectionResetError as e:
            error_msg = f"ConnectionResetError: {str(e)}"
            self.logger.warning(f"      ⚠️ 网络异常（可重试）: {error_msg}")
            return "network_error"
        except Exception as e:
            error_msg, is_retryable_network_error = self._classify_download_exception(e)
            if is_retryable_network_error:
                self.logger.warning(f"      ⚠️ 网络异常（可重试）: {error_msg}")
                return "network_error"
            self.logger.error(f"      ❌ 下载异常: {error_msg}")
            return False

    def download_with_retry(self, event_info: Dict) -> Tuple[bool, str]:
        """
        带重试机制的下载。

        永久台策略：
        1. 每轮扫完配置中的全部 Provider（永久台默认 10 个东亚/西太源）
        2. 未达补全目标不提前结束；第 2 轮起只请求仍缺的台
        3. 一轮下来文件数不再增加则停止空转
        4. 全部轮次结束后：过失败下限算成功，否则 insufficient_data
        """
        retry_config = self.config.retry
        max_retries = retry_config['max_retries']
        event_dir = event_info['dir']
        origin_time = event_info.get('origin_time')
        all_providers = self._provider_list()
        provider_network_retries = int(retry_config.get('provider_network_retries', 1))
        
        self.logger.info(
            f"🎯 开始下载事件 {event_dir}，将尝试 {len(all_providers)} 个数据源: "
            f"{', '.join(all_providers)}"
        )
        if self.use_permanent_stations and self.permanent_stations_df is not None:
            n_active = self._count_active_permanent_stations(origin_time)
            n_chunks = len(self._station_restriction_chunks(origin_time))
            thr0 = self._quality_thresholds(origin_time)
            self.logger.info(
                f"🎯 发震时刻有效永久台: {n_active} / {len(self.permanent_stations_df)} "
                f"| 首轮 {n_chunks} 批查询 "
                f"| 补全目标 {thr0['fill_frac']:.0%} 台且三分量 {thr0['complete_frac']:.0%} "
                f"| 失败下限 {thr0['accept_frac']:.0%} 台"
            )
        
        prev_mseed = -1
        prev_present = -1
        prev_complete = -1
        prev_xml = -1
        last_cov: Optional[Dict[str, Any]] = None

        for retry_round in range(1, max_retries + 1):
            n_missing = 0
            if self.use_permanent_stations:
                last_cov = self._event_coverage(event_dir, origin_time)
                n_missing = len(last_cov['missing_ids'])
                if retry_round > 1:
                    self.logger.info(
                        f"📡 第 {retry_round} 轮：定向补漏 {n_missing} 个未补全台，"
                        f"尝试 {len(all_providers)} 个数据源..."
                    )
                else:
                    self.logger.info(f"📡 第 {retry_round} 轮下载：尝试所有数据源...")
            else:
                self.logger.info(f"📡 第 {retry_round} 轮下载：尝试所有数据源...")
            round_success_count = 0
            
            for provider_idx, provider in enumerate(all_providers, 1):
                self.provider_index = all_providers.index(provider)
                
                self.logger.info(f"   [{provider_idx}/{len(all_providers)}] 尝试数据源: {provider}")
                
                result = self.download_event_data(event_info, retry_round)
                net_attempt = 0
                while result == "network_error" and net_attempt < provider_network_retries:
                    delay = int(retry_config.get('timeout_delay', 8))
                    net_attempt += 1
                    self.logger.warning(
                        f"   ⚠️  {provider} 网络中断，{delay}s 后重试该源 "
                        f"({net_attempt}/{provider_network_retries})..."
                    )
                    time.sleep(delay)
                    result = self.download_event_data(event_info, retry_round)
                
                if provider not in self.download_stats['provider_usage']:
                    self.download_stats['provider_usage'][provider] = 0
                self.download_stats['provider_usage'][provider] += 1
                
                if result is True:
                    round_success_count += 1
                    self.logger.info(f"   ✅ {provider} 已达补全目标")
                elif result is False:
                    round_success_count += 1
                    self.logger.info(f"   ⚠️  {provider} 本源结束，继续补漏")
                elif result == "network_error":
                    self.logger.warning(f"   ❌ {provider} 网络失败，跳过该源")
                else:
                    self.logger.debug(f"   ➖ {provider} 无可用数据")

                cov = self._event_coverage(event_dir, origin_time)
                last_cov = cov
                self.logger.info(f"   📊 {self._format_coverage(cov)}")
                # 全网模式必须扫完所有区域源，避免 IRIS 先到门槛就丢掉 TW/GE/G
                if self.use_permanent_stations and cov['fill_ok']:
                    self.logger.info("   ✅ 已达补全目标，停止后续数据源")
                    break
                
                time.sleep(1)
            
            cov = last_cov or self._event_coverage(event_dir, origin_time)
            if not self.use_permanent_stations:
                # 全网：每个源已经按研究区拉过一遍，再轮询收益很小
                if cov['n_mseed'] > 0:
                    self.logger.info(
                        f"🎉 事件 {event_dir} 全网一轮完成 | {self._format_coverage(cov)}"
                    )
                    return True, "success"
                self.logger.warning(
                    f"⚠️  第 {retry_round} 轮全网无波形: {self._format_coverage(cov)}"
                )
            elif cov['fill_ok']:
                self.logger.info(f"🎉 事件 {event_dir} 已达补全目标！")
                self.logger.info(f"   第 {retry_round} 轮结束 | {self._format_coverage(cov)}")
                self.logger.info(f"   写出数据的源: {round_success_count}/{len(all_providers)}")
                return True, "success"

            no_new_files = (
                cov['n_mseed'] == prev_mseed
                and cov['n_present'] == prev_present
                and cov['n_complete'] == prev_complete
                and cov['n_xml'] == prev_xml
            )
            prev_mseed = cov['n_mseed']
            prev_present = cov['n_present']
            prev_complete = cov['n_complete']
            prev_xml = cov['n_xml']

            if no_new_files and retry_round > 1:
                if cov['accept_ok']:
                    self.logger.info(
                        f"ℹ️  第 {retry_round} 轮无新增波形，剩余台在 FDSN 上可能本来就没有。"
                        f"已过失败下限，结束补全。"
                    )
                    self.logger.info(f"   {self._format_coverage(cov)}")
                    return True, "success"
                self.logger.warning(
                    f"⚠️  第 {retry_round} 轮无新增，且未过失败下限: {self._format_coverage(cov)}"
                )
            else:
                self.logger.warning(
                    f"⚠️  第 {retry_round} 轮未达补全目标: {self._format_coverage(cov)}"
                )
            
            if retry_round < max_retries:
                delay = retry_config['base_delay']
                self.logger.info(f"   等待 {delay} 秒后开始第 {retry_round + 1} 轮补漏...")
                time.sleep(delay)
            else:
                if cov['accept_ok']:
                    self.logger.info(
                        f"🎉 事件 {event_dir} 未达补全目标，但已过失败下限，记为成功。"
                    )
                    self.logger.info(f"   {self._format_coverage(cov)}")
                    return True, "success"
                self.logger.error(
                    f"❌ 事件 {event_dir} 下载失败：{max_retries} 轮后仍低于失败下限。"
                )
                self.logger.error(f"   {self._format_coverage(cov)}")
                return False, "insufficient_data"
        
        return False, "failed"

    def save_progress(
        self,
        catalog_file: Path,
        next_event_index: int,
        catalog_event_count: int,
    ) -> None:
        """
        保存下载进度。

        Args:
            catalog_file: 当前使用的 .par 路径
            next_event_index: 下次应从 catalog 中取的第 **0-based** 行索引（已完成 0..i-1 时写入 i）
            catalog_event_count: 当前目录总行数，用于续跑校验
        """
        progress_data = {
            'next_event_index': next_event_index,
            'catalog_path': str(catalog_file.resolve()),
            'catalog_event_count': catalog_event_count,
            'download_stats': self.download_stats,
            'provider_index': self.provider_index,
            'use_permanent_stations': self.use_permanent_stations,
            'session_dir': str(self.output_paths['session_dir']),
            'timestamp': datetime.now().isoformat(),
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
            
            # 续跑时台站模式以本次交互选择为准，不覆盖 self.use_permanent_stations

            nxt = progress_data.get('next_event_index')
            if nxt is None:
                nxt = progress_data.get('current_index', 0)
                self.logger.warning(
                    "⚠️ 进度文件为旧格式(current_index)，下一事件索引可能与旧版保存语义不一致；建议重新全量下载"
                )
            self.logger.info(f"📋 发现之前的下载进度，从 catalog 行索引 {nxt} 继续")
            return progress_data
        except Exception as e:
            self.logger.warning(f"⚠️  加载进度文件失败: {e}")
            return None

    def list_existing_catalogs(self) -> List[Path]:
        """
        列出现有的下载目录文件（events 根目录与 test_database 子目录下的 .par）
        
        Returns:
            现有目录文件路径列表
        """
        catalog_dir = self.output_paths['events_base']
        catalog_files = list(catalog_dir.glob("*.par"))
        sub = catalog_dir / "test_database"
        if sub.is_dir():
            catalog_files.extend(sub.glob("*.par"))
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
        
        events_base = self.output_paths['events_base']
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
            
            try:
                rel_display = catalog_file.relative_to(events_base)
            except ValueError:
                rel_display = catalog_file
            
            is_special = self._should_offer_permanent_station_mode(catalog_file)
            file_type = " [测试库/精选事件] ⭐" if is_special else ""
            
            print(f"{i:2d}. {rel_display}{file_type}")
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
                    try:
                        rel_display = selected_file.relative_to(self.output_paths['events_base'])
                    except ValueError:
                        rel_display = selected_file
                    print(f"✅ 已选择: {rel_display}")
                    if self._should_offer_permanent_station_mode(selected_file):
                        self._offer_permanent_stations_interactive()
                    else:
                        # hybrid 等普通目录不弹交互，沿用配置默认
                        self._apply_permanent_station_config_default()
                        if self.use_permanent_stations and self.permanent_stations_df is not None:
                            print(f"📌 台站模式（配置 use_permanent_only=True）: "
                                  f"仅永久台站 {len(self.permanent_stations_df)} 台")
                        else:
                            print("📌 台站模式（配置）: 不限制 network=*")
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
                ev = self.parse_event(line.strip())
                if ev is not None:
                    print(
                        f"{i:2d}. {ev['dir']} | Lat: {ev['lat']}° "
                        f"Lon: {ev['lon']}° Depth: {ev['depth']}km"
                    )
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

    def batch_download(
        self, catalog_file: Path, max_events: Optional[int] = None,
    ) -> Dict:
        """
        批量下载多个事件的数据

        Args:
            catalog_file: 目录文件路径
            max_events: 最多处理事件数（调试用）

        Returns:
            下载统计字典
        """
        self._initialize_storage()
        self.processing_start_time = time.time()
        self._configure_obspy_logging()
        catalog = self.load_catalog(catalog_file)
        if max_events is not None and max_events > 0:
            catalog = catalog[:max_events]
        total_events = len(catalog)

        resume_dir = self._interactive_pick_resume_session(catalog_file, total_events)
        if resume_dir is not None:
            self._attach_existing_session(resume_dir)
            self.logger.info(f"♻️ 沿用未完成会话: {resume_dir}")
        else:
            self.provider_index = 0
            self._begin_download_session()

        self.download_stats['start_time'] = self.processing_start_time

        progress_data = self.load_progress()
        if not progress_data:
            self.provider_index = 0
            start_idx = 0
        else:
            start_idx = progress_data.get("next_event_index")
            if start_idx is None:
                start_idx = progress_data.get("current_index", 0)
        if self.download_stats.get('start_time') is None:
            self.download_stats['start_time'] = self.processing_start_time
        if start_idx > total_events:
            self.logger.warning(
                f"⚠️ 进度中的起始索引 {start_idx} 超过当前目录行数 {total_events}，从 0 开始"
            )
            start_idx = 0
        
        mode_info = " [仅永久台站]" if self.use_permanent_stations else ""
        self.logger.info(f"📊 开始批量下载{mode_info}: {total_events - start_idx} 个待处理事件 (总计 {total_events})")
        self.logger.info(f"🌐 可用数据源: {', '.join(self._provider_list())}")
        
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
                if self.check_event_exists(event_dir, event_info.get('origin_time')):
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
                
                # 定期保存：下一待处理索引为 i+1（已完成第 i 行）
                if (i + 1) % self.config.performance['auto_save_interval'] == 0:
                    self.save_progress(catalog_file, i + 1, total_events)
                
                # 显示进度
                if (i - start_idx + 1) % self.config.performance['progress_interval'] == 0:
                    self._print_progress(i - start_idx + 1, total_events - start_idx)
            
            # 最终统计
            self.download_stats['total_processed'] = total_events
            if self.processing_start_time is None:
                self.logger.warning("⚠️ 下载开始时间缺失，总耗时将记为 0 分钟")
                total_time = 0.0
            else:
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
            # 中断于第 i 个事件处理中：下次仍从 i 重试
            idx = i if "i" in locals() else start_idx
            self.save_progress(catalog_file, idx, total_events)
            raise
        except Exception as e:
            self.logger.error(f"💥 下载过程出错: {e}")
            if "i" in locals():
                self.save_progress(catalog_file, i, total_events)
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
                f.write(f"会话目录: {self.output_paths['session_dir']}\n")
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
            'session_dir': str(self.output_paths['session_dir']),
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
                
                if self._should_offer_permanent_station_mode(catalog_file):
                    self._offer_permanent_stations_interactive()
                else:
                    self._apply_permanent_station_config_default()
                    if self.use_permanent_stations and self.permanent_stations_df is not None:
                        print(f"📌 台站模式（配置 use_permanent_only=True）: "
                              f"仅永久台站 {len(self.permanent_stations_df)} 台")
                    else:
                        print("📌 台站模式（配置）: 不限制 network=*")
                
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


def _resolve_catalog_arg(raw: str, events_dir: Path) -> Path:
    """解析 --catalog：绝对路径、相对路径，或 data/events 下的文件名。"""
    p = Path(raw).expanduser()
    candidates = [p]
    if not p.is_absolute():
        candidates.append(Path.cwd() / p)
        candidates.append(events_dir / p)
        if not p.suffix:
            candidates.append(events_dir / f'{p}.par')
            candidates.append(events_dir / f'download_catalog_{p}.par')
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f'找不到目录文件: {raw}')


def main():
    """主函数 - 执行波形数据下载"""
    print("📡 EASTASIA-FWI 波形数据下载系统")
    print("   支持: 永久台站筛选 | 研究区全网 | 未完成会话续跑")
    print("="*60)

    parser = argparse.ArgumentParser(
        description='东亚研究区地震波形批量下载（FDSN MassDownloader）',
    )
    parser.add_argument(
        '--catalog', type=str, default=None,
        help='目录 .par（路径或 data/events 下文件名，如 download_catalog_hybrid_30events.par）',
    )
    parser.add_argument(
        '--all-stations', action='store_true',
        help='研究区内不限制台站清单（network=*），供后续 1_8 按波形质量筛选',
    )
    parser.add_argument(
        '--permanent', action='store_true',
        help='仅下载永久台站清单内台站',
    )
    parser.add_argument(
        '--yes', '-y', action='store_true',
        help='跳过交互确认',
    )
    parser.add_argument(
        '--max-events', type=int, default=None,
        help='最多下载事件数（调试用）',
    )
    parser.add_argument(
        '--min-distance-km', type=float, default=None,
        help='全网模式最小台间距（km），覆盖配置 all_stations_min_interstation_m',
    )
    args = parser.parse_args()

    if args.all_stations and args.permanent:
        raise SystemExit('不能同时指定 --all-stations 与 --permanent')

    try:
        station_mode = None
        if args.all_stations:
            station_mode = 'all'
        elif args.permanent:
            station_mode = 'permanent'
        downloader = WaveformDownloader(station_mode=station_mode)
        downloader.auto_confirm = bool(args.yes)
        if args.min_distance_km is not None:
            downloader.config.fdsn['all_stations_min_interstation_m'] = (
                int(args.min_distance_km * 1000.0)
            )

        if args.catalog:
            catalog_file = _resolve_catalog_arg(
                args.catalog, downloader.output_paths['events_base'])
            print(f'✅ 目录文件: {catalog_file}')
        else:
            catalog_file = downloader.select_catalog_source()

        if catalog_file is None:
            print("❌ 没有可用的下载目录")
            return

        mode_info = " [仅永久台站]" if downloader.use_permanent_stations else " [研究区全网]"
        if not args.yes:
            response = input(f"\n是否开始下载波形数据{mode_info}? (y/N): ").strip().lower()
            if response != 'y':
                print("下载已取消")
                return

        print(f"\n🚀 开始下载波形数据{mode_info}...")
        print(f"📋 使用目录文件: {catalog_file.name}")
        results = downloader.batch_download(
            catalog_file, max_events=args.max_events,
        )
        
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
        print(f"  会话目录: {results['session_dir']}")
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