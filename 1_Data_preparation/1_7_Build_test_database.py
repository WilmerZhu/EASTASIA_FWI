"""
1_7_Build_test_database.py:
SPECFEM3D 正演验证标准测试数据库构建模块
================================================================

功能描述:
----------
基于 CMT3D 震源目录（Sawade et al., 2022, GJI）构建标准测试数据库，
用于 SPECFEM3D Globe 正演模拟流程验证、模型转换精度和波形拟合能力评估。

CMT3D 震源参数经过三维结构模型（GLAD-M25）全波形拟合修正，
比标准 GCMT 1D 解更接近真实地球结构，适合作为正演模拟的 ground truth 震源。

核心功能:
----------
1. ✅ CMT3D 目录解析
   - 解析 CMTSOLUTION 格式（13行/事件）
   - 支持 cmt3d.txt 合并文件和 finalcatalog/ 单事件文件
   - 提取完整震源参数（位置、深度、矩张量、半持续时间）

2. ✅ 东亚区域事件筛选
   - 地理区域筛选: -15°~60°N, 60°~170°E
   - 震级范围筛选: 可配置（默认 5.5-7.0）
   - 深度范围筛选: 可配置（默认 0-700 km）
   - 时间范围筛选: CMT3D 覆盖 2001-2019

3. ✅ 智能事件采样
   - 深度分层采样: 浅源(0-70km)/中源(70-300km)/深源(300-700km)
   - 震级均匀分布
   - 空间去重: 避免事件过于密集
   - 覆盖不同构造区域（俯冲带、碰撞带、板内）
   - hybrid_30 模式: 与 hybrid_10events 互补的 30 个空间均匀事件（2010+）
   - eastasia_model_30 模式: 从 EastAsia_model 波形库 Mw≥5.5 候选中
     选取 30 个空间均匀、远离研究区边界的正演测试事件

4. ✅ 永久台站集成
   - 使用 293 个经筛选的永久台站
   - 生成 SPECFEM3D STATIONS 文件
   - 核心台网: IC, IU, II, G, GE, AU, MY, JP, MM, KZ

5. ✅ SPECFEM3D 输出
   - CMTSOLUTION 文件（可直接用于正演）
   - STATIONS 文件
   - 事件-台站对应表
   - 下载任务清单（衔接 1_3 下载模块）

6. ✅ 统计报告和可视化
   - 事件分布统计
   - 深度/震级直方图
   - 空间覆盖分析

科学原理:
----------
- CMT3D 目录使用三维地球模型 GLAD-M25 进行全波形拟合，震源参数更准确
- 测试事件需覆盖不同深度范围以验证不同速度结构的正演效果
- 永久台站数据质量稳定，适合长期基准比较
- 参考 FWEA23 研究中使用的质控流程

作者: EASTASIA-FWI Team
日期: 2026-03-31
版本: v1.0
"""

import argparse
import re
import sys
import json
import math
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import BaseConfig

warnings.filterwarnings('ignore')


# ================================================================
# 数据类定义
# ================================================================

@dataclass
class CMT3DEvent:
    """CMT3D 事件数据类"""
    event_name: str
    pde_line: str
    time_shift: float
    half_duration: float
    latitude: float
    longitude: float
    depth: float
    mrr: float
    mtt: float
    mpp: float
    mrt: float
    mrp: float
    mtp: float

    # 从 PDE 行解析的附加信息
    pde_year: int = 0
    pde_month: int = 0
    pde_day: int = 0
    pde_hour: int = 0
    pde_minute: int = 0
    pde_second: float = 0.0
    pde_lat: float = 0.0
    pde_lon: float = 0.0
    pde_depth: float = 0.0
    pde_mb: float = 0.0
    pde_ms: float = 0.0

    @property
    def magnitude_mw(self) -> float:
        """从标量地震矩计算矩震级 Mw"""
        m0 = self.scalar_moment
        if m0 > 0:
            return (2.0 / 3.0) * (math.log10(m0) - 16.1)
        return 0.0

    @property
    def scalar_moment(self) -> float:
        """计算标量地震矩 M0 (dyn·cm)"""
        return math.sqrt(0.5 * (
            self.mrr**2 + self.mtt**2 + self.mpp**2 +
            2.0 * (self.mrt**2 + self.mrp**2 + self.mtp**2)
        ))

    @property
    def best_magnitude(self) -> float:
        """优先返回 Mw，其次 mb/Ms"""
        mw = self.magnitude_mw
        if mw > 0:
            return mw
        if self.pde_mb > 0:
            return self.pde_mb
        if self.pde_ms > 0:
            return self.pde_ms
        return 0.0

    @property
    def depth_category(self) -> str:
        """深度分类"""
        if self.depth <= 70:
            return 'shallow'
        elif self.depth <= 300:
            return 'intermediate'
        else:
            return 'deep'

    @property
    def origin_datetime(self) -> Optional[datetime]:
        """事件发震时间"""
        try:
            sec_int = int(self.pde_second)
            microsec = int((self.pde_second - sec_int) * 1e6)
            return datetime(
                self.pde_year, self.pde_month, self.pde_day,
                self.pde_hour, self.pde_minute, min(sec_int, 59),
                microsec
            )
        except (ValueError, OverflowError):
            return None

    def to_cmtsolution(self) -> str:
        """输出标准 CMTSOLUTION 格式（13行）"""
        return (
            f"{self.pde_line}\n"
            f"event name:     {self.event_name}\n"
            f"time shift:{self.time_shift:12.4f}\n"
            f"half duration:{self.half_duration:9.4f}\n"
            f"latitude:{self.latitude:14.4f}\n"
            f"longitude:{self.longitude:13.4f}\n"
            f"depth:{self.depth:17.4f}\n"
            f"Mrr:{self.mrr:18.6e}\n"
            f"Mtt:{self.mtt:18.6e}\n"
            f"Mpp:{self.mpp:18.6e}\n"
            f"Mrt:{self.mrt:18.6e}\n"
            f"Mrp:{self.mrp:18.6e}\n"
            f"Mtp:{self.mtp:18.6e}\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_name': self.event_name,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'depth': self.depth,
            'magnitude_mw': round(self.magnitude_mw, 2),
            'pde_mb': self.pde_mb,
            'pde_ms': self.pde_ms,
            'time_shift': self.time_shift,
            'half_duration': self.half_duration,
            'depth_category': self.depth_category,
            'scalar_moment': self.scalar_moment,
            'pde_year': self.pde_year,
            'pde_month': self.pde_month,
            'pde_day': self.pde_day,
            'pde_hour': self.pde_hour,
            'pde_minute': self.pde_minute,
            'pde_second': self.pde_second,
            'mrr': self.mrr,
            'mtt': self.mtt,
            'mpp': self.mpp,
            'mrt': self.mrt,
            'mrp': self.mrp,
            'mtp': self.mtp,
        }


# ================================================================
# 配置类
# ================================================================

class TestDatabaseConfig:
    """标准测试数据库构建配置"""

    def __init__(self):
        # CMT3D 数据源
        self.cmt3d = {
            'catalog_file': 'cmt3d.txt',
            'catalog_dir': 'finalcatalog',
            'base_dir': 'CMT3D',
            'source': 'Sawade et al., 2022, GJI',
            'reference_model': 'GLAD-M25',
        }

        # FWEA23 参考数据源
        self.fwea23 = {
            'excel_file': '1-s2.0-S0012821X24001973-mmc2.xlsx',
            'source': 'Tao et al., 2024, EPSL (FWEA23)',
            'total_events': 141,
            'time_match_tolerance_min': 2,
        }

        # 事件筛选条件
        self.event_filters = {
            'magnitude_range': [5.5, 7.0],
            'depth_range': [0, 700],
            'time_range': [2001, 2019],
        }

        # 智能采样策略
        self.sampling = {
            'target_count': 80,
            'min_count': 60,
            'max_count': 100,
            # 深度分层配额比例（浅源:中源:深源）
            'depth_quotas': {
                'shallow': 0.40,       # 0-70 km
                'intermediate': 0.35,  # 70-300 km
                'deep': 0.25,          # 300-700 km
            },
            'depth_boundaries': [0, 70, 300, 700],
            # 空间去重：最小事件间距（度）
            'min_event_distance_deg': 1.0,
            # 震级优先：优先选择震级更大的事件（信噪比更高）
            'prefer_larger_magnitude': True,
            # 随机种子（可复现）
            'random_seed': 42,
        }

        # 正演 30 事件目录：与 download_catalog_hybrid_10events.par 互补
        # CMT3D 截止 2019.11，故 2010 年以后 = 2010–2019
        self.hybrid_forward = {
            'output_subdir': 'hybrid_30events',
            'root_par_name': 'download_catalog_hybrid_30events.par',
            'id_map_name': 'hybrid_30events_id_to_dir.csv',
            'events_list_name': 'events.list',
            'exclude_par': 'download_catalog_hybrid_10events.par',
            'exclude_id_map': 'hybrid_10events_id_to_dir.csv',
            'exclude_distance_deg': 1.5,
            'target_count': 30,
            'time_range': [2010, 2019],
            'magnitude_range': [5.5, 6.8],
            'depth_range': [0, 700],
            # 与 hybrid_10events 及 FWEA23∩EARA 骨架重叠；排除印度洋中脊
            'lat_range': [-10.0, 55.0],
            'lon_range': [80.0, 148.0],
            'exclude_south_west': {'lat_max': 0.0, 'lon_max': 95.0},
            # 相对 10 事件的 5/2/3 深度结构按比例放大，并略增中源以服务岩石圈 FWI
            'depth_quotas': {
                'shallow': 15,
                'intermediate': 8,
                'deep': 7,
            },
            'min_event_distance_deg': 4.0,
            'relax_distance_deg': 2.5,
            'year_weight': 0.35,
            'magnitude_weight': 0.15,
            # 地理种子：大陆内部 + 俯冲带四角，促使空间铺开
            'seed_points': [
                (50.0, 80.0),    # 西准噶尔 / 阿尔泰
                (48.0, 140.0),   # 千岛 / 东北深源
                (8.0, 95.0),     # 安达曼 / 苏门答腊
                (8.0, 126.0),    # 菲律宾 / 苏拉威西
                (32.0, 88.0),    # 青藏高原
                (24.0, 121.0),   # 台湾
                (36.0, 108.0),   # 鄂尔多斯 / 华北
                (42.0, 130.0),   # 日本海 / 东北深源
            ],
            'seed_search_radius_deg': 12.0,
        }

        # EastAsia_model 正演测试事件：从已有波形库中采样，避开研究区边缘
        self.eastasia_model_forward = {
            'dataset_subdir': 'EastAsia_model',
            'output_subdir': 'eastasia_model_30events',
            'root_par_name': 'download_catalog_eastasia_model_30events.par',
            'id_map_name': 'eastasia_model_30events_id_to_dir.csv',
            'events_list_name': 'events.list',
            'target_count': 30,
            'magnitude_range': [5.5, 7.5],
            'depth_range': [0, 700],
            'time_range': [2008, 2025],
            # 距 base_config 边界的最小距离（度），避免边缘事件
            'min_interior_margin_deg': 5.0,
            # 硬排除：南界以南、东缘以远（lat<0° 或 lon>143°）
            'min_latitude_deg': 0.0,
            'max_longitude_deg': 143.0,
            # 大陆内部偏好区（评分加权）
            'mainland_lat_range': [5.0, 50.0],
            'mainland_lon_range': [70.0, 130.0],
            'mainland_weight': 0.30,
            'require_waveform': True,
            'depth_quotas': {
                'shallow': 14,
                'intermediate': 8,
                'deep': 8,
            },
            'min_event_distance_deg': 3.5,
            'relax_distance_deg': 2.0,
            'year_weight': 0.25,
            'magnitude_weight': 0.20,
            'coverage_weight': 0.20,
            # 分时段配额（优先于单一观测窗口）；合计须等于 target_count
            'temporal_quotas': [
                {'start_year': 2008, 'end_year': 2013, 'count': 7},
                {'start_year': 2014, 'end_year': 2018, 'count': 15},
                {'start_year': 2019, 'end_year': 2025, 'count': 8},
            ],
            # 单一观测窗口（temporal_quotas 为空时启用）
            'use_optimal_observation_window': False,
            'observation_window_years': 5,
            'observation_window': None,
            'seed_points': [
                (50.0, 80.0),    # 阿尔泰
                (42.0, 85.0),    # 天山
                (32.0, 88.0),    # 青藏高原
                (36.0, 108.0),   # 华北
                (28.0, 104.0),   # 四川盆地
                (24.0, 110.0),   # 华南
                (30.0, 95.0),    # 藏东
                (38.0, 118.0),   # 华东
            ],
            'seed_search_radius_deg': 10.0,
            'stations_catalog': 'catalog/stations_catalog.csv',
            'waveform_coverage_csv': 'catalog/event_waveform_coverage.csv',
        }

        # 论文地球透视图：事件震源球 + 台站小三角（单图叠加）
        self.eastasia_paper_globe = {
            'figure_stem': '1-7_eastasia_model_30events_paper_globe',
            'save_formats': ['jpg', 'pdf'],
            'dpi': 300,
            'globe_width': '14c',
            'altitude': 2.65,
            'study_region_pen': '2.2p,black',
            'grid_interval': 45,
            'station_size': 't0.05c',
            'station_fill': '180/190/200',
            'station_pen': '0.04p,120/130/140',
            'meca_pen': '0.20p,30/30/30',
            'depth_cmap': 'seis',
            'depth_range': [0, 700],
        }
        # 台站网络详图（单独成图，按台网分色 + 统计图例）
        self.eastasia_station_detail = {
            'figure_stem': '1-7_eastasia_model_stations_network',
            'save_formats': ['jpg', 'pdf'],
            'dpi': 300,
            'region': [55.0, 155.0, -12.0, 62.0],
            'projection_width': '18c',
            'frame': ['xa20f10', 'ya10f5', 'WSen'],
            'top_networks': 14,
            'station_size': 't0.07c',
        }

        # 论文展示图：30 事件 + 1_8 按实测波形筛选的质量台站
        self.paper_map = {
            'region': [65.0, 155.0, -14.0, 58.0],
            'projection_width': '17c',
            'frame': ['xa20f10g20', 'ya10f5g10', 'WSen'],
            'save_formats': ['jpg', 'pdf'],
            'dpi': 300,
            'figure_stem': '1-7_paper_events_stations',
            'stations_name': 'STATIONS_quality',
            'station_size': 'i0.15c',
            'station_fill': '255/255/255',
            'station_pen': '0.45p,20/55/105',
        }

        # 永久台站配置
        self.stations = {
            'station_file': 'EastAsia_permanent_stations_filtered.csv',
            'core_networks': ['IC', 'IU', 'II', 'G', 'GE', 'AU', 'MY', 'JP', 'MM', 'KZ'],
        }

        # 质量控制标准（参考 FWEA23）
        self.quality_control = {
            'snr_body_wave': 5.0,
            'snr_surface_wave': 3.0,
            'min_period': 10.0,  # 秒，匹配 SPECFEM 模拟周期
        }

        # 输出配置
        self.output = {
            'output_subdir': 'test_database',
            'save_cmtsolution': True,
            'save_stations': True,
            'save_catalog_csv': True,
            'save_download_list': True,
            'save_report': True,
        }

        # 日志配置
        self.logging = {
            'level': 'INFO',
            'console_output': True,
        }


# ================================================================
# 主功能类
# ================================================================

class TestDatabaseBuilder:
    """
    标准测试数据库构建器

    从 CMT3D 目录中选取 50-100 个代表性事件，
    配合 293 个永久台站，构建 SPECFEM3D 正演验证用标准测试数据集。
    """

    def __init__(self, output_dir: Optional[str] = None):
        # 加载全局配置
        self.base_config = BaseConfig()

        # 加载模块配置
        self.config = TestDatabaseConfig()

        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = (
                self.base_config.dirs['data'] / 'events' / self.config.output['output_subdir']
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.TestDatabaseBuilder',
            self.config.logging['level']
        )

        # 设置区域参数
        region_bounds = self.base_config.get_region_bounds()
        self.region = [
            region_bounds['lon_min'], region_bounds['lon_max'],
            region_bounds['lat_min'], region_bounds['lat_max']
        ]

        # 数据存储
        self.all_events: List[CMT3DEvent] = []
        self.filtered_events: List[CMT3DEvent] = []
        self.selected_events: List[CMT3DEvent] = []
        self.stations_df: Optional[pd.DataFrame] = None
        self.waveform_coverage: Optional[Dict[str, int]] = None
        self.observation_window: Optional[Dict[str, Any]] = None

        # EastAsia_model 数据集路径
        ea_cfg = self.config.eastasia_model_forward
        self.eastasia_model_dir = (
            self.base_config.dirs['events'] / ea_cfg['dataset_subdir']
        )

        self.logger.info("🎯 标准测试数据库构建器初始化完成")
        self._print_config_summary()

    # ----------------------------------------------------------------
    # 配置与诊断
    # ----------------------------------------------------------------

    def _print_config_summary(self):
        """打印配置摘要"""
        cfg = self.config
        region = self.base_config.region
        print("\n📋 标准测试数据库构建配置")
        print("-" * 60)
        print(f"研究区域: {region['name']}")
        print(f"纬度范围: {region['lat_min']}° ~ {region['lat_max']}°")
        print(f"经度范围: {region['lon_min']}° ~ {region['lon_max']}°")
        print(f"震源目录: CMT3D ({cfg.cmt3d['source']})")
        print(f"参考模型: {cfg.cmt3d['reference_model']}")
        print(f"震级范围: {cfg.event_filters['magnitude_range'][0]} ~ "
              f"{cfg.event_filters['magnitude_range'][1]}")
        print(f"深度范围: {cfg.event_filters['depth_range'][0]} ~ "
              f"{cfg.event_filters['depth_range'][1]} km")
        print(f"目标事件数: {cfg.sampling['target_count']} "
              f"({cfg.sampling['min_count']}-{cfg.sampling['max_count']})")
        print(f"最小事件间距: {cfg.sampling['min_event_distance_deg']}°")
        print(f"输出目录: {self.output_dir}")
        print("-" * 60)

    # ----------------------------------------------------------------
    # CMT3D 目录解析
    # ----------------------------------------------------------------

    def parse_cmt3d_catalog(self, source: str = 'auto') -> List[CMT3DEvent]:
        """
        解析 CMT3D 震源目录

        Args:
            source: 数据源
                - 'auto': 优先使用 cmt3d.txt，否则读 finalcatalog/
                - 'merged': 仅使用 cmt3d.txt
                - 'individual': 仅使用 finalcatalog/ 中的单事件文件

        Returns:
            解析后的事件列表
        """
        cmt3d_base = self.base_config.dirs['catalogs'] / self.config.cmt3d['base_dir']

        if source == 'auto':
            merged_file = cmt3d_base / self.config.cmt3d['catalog_file']
            if merged_file.exists():
                source = 'merged'
            else:
                source = 'individual'

        if source == 'merged':
            events = self._parse_merged_catalog(
                cmt3d_base / self.config.cmt3d['catalog_file']
            )
        else:
            events = self._parse_individual_catalogs(
                cmt3d_base / self.config.cmt3d['catalog_dir']
            )

        self.all_events = events
        self.logger.info(f"📖 CMT3D 目录解析完成: {len(events)} 个事件")
        return events

    def _parse_merged_catalog(self, catalog_file: Path) -> List[CMT3DEvent]:
        """解析合并的 cmt3d.txt 文件"""
        if not catalog_file.exists():
            raise FileNotFoundError(f"CMT3D 目录文件不存在: {catalog_file}")

        self.logger.info(f"📖 解析合并文件: {catalog_file}")

        events = []
        error_count = 0

        with open(catalog_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_expected = len(lines) // 13
        self.logger.info(f"📊 预计事件数: {total_expected}")

        for i in range(0, len(lines), 13):
            if i + 12 >= len(lines):
                break
            event_lines = lines[i:i + 13]
            try:
                event = self._parse_single_cmt3d_event(event_lines)
                if event is not None:
                    events.append(event)
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    self.logger.debug(f"事件解析错误 (行 {i + 1}): {e}")

        if error_count > 0:
            self.logger.warning(f"⚠️  解析错误: {error_count} 个事件")

        return events

    def _parse_individual_catalogs(self, catalog_dir: Path) -> List[CMT3DEvent]:
        """解析 finalcatalog/ 目录中的单事件文件"""
        if not catalog_dir.exists():
            raise FileNotFoundError(f"CMT3D 事件目录不存在: {catalog_dir}")

        self.logger.info(f"📖 解析事件目录: {catalog_dir}")

        event_files = sorted(catalog_dir.iterdir())
        events = []
        error_count = 0

        for event_file in event_files:
            if not event_file.is_file():
                continue
            try:
                with open(event_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                if len(lines) >= 13:
                    event = self._parse_single_cmt3d_event(lines[:13])
                    if event is not None:
                        events.append(event)
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    self.logger.debug(f"文件解析错误 ({event_file.name}): {e}")

        if error_count > 0:
            self.logger.warning(f"⚠️  解析错误: {error_count} 个文件")

        return events

    def _parse_single_cmt3d_event(self, lines: List[str]) -> Optional[CMT3DEvent]:
        """
        解析单个 CMT3D 事件（13 行 CMTSOLUTION 格式）

        行格式:
         1: PDE line (PDE year month day hour min sec lat lon depth mb ms location)
         2: event name:     XXXXXXXX
         3: time shift:     XX.XXXX
         4: half duration:  XX.XXXX
         5: latitude:       XX.XXXX
         6: longitude:      XX.XXXX
         7: depth:          XX.XXXX
         8: Mrr:            X.XXXXXXe+XX
         9: Mtt:            X.XXXXXXe+XX
        10: Mpp:            X.XXXXXXe+XX
        11: Mrt:            X.XXXXXXe+XX
        12: Mrp:            X.XXXXXXe+XX
        13: Mtp:            X.XXXXXXe+XX
        """
        if len(lines) < 13:
            return None

        stripped = [line.strip() for line in lines]

        def _extract_value(line: str) -> str:
            """提取冒号后的值"""
            return line.split(':', 1)[1].strip()

        try:
            pde_line = stripped[0]
            event_name = _extract_value(stripped[1])
            time_shift = float(_extract_value(stripped[2]))
            half_duration = float(_extract_value(stripped[3]))
            latitude = float(_extract_value(stripped[4]))
            longitude = float(_extract_value(stripped[5]))
            depth = float(_extract_value(stripped[6]))
            mrr = float(_extract_value(stripped[7]))
            mtt = float(_extract_value(stripped[8]))
            mpp = float(_extract_value(stripped[9]))
            mrt = float(_extract_value(stripped[10]))
            mrp = float(_extract_value(stripped[11]))
            mtp = float(_extract_value(stripped[12]))

            event = CMT3DEvent(
                event_name=event_name,
                pde_line=pde_line,
                time_shift=time_shift,
                half_duration=half_duration,
                latitude=latitude,
                longitude=longitude,
                depth=depth,
                mrr=mrr, mtt=mtt, mpp=mpp,
                mrt=mrt, mrp=mrp, mtp=mtp,
            )

            # 解析 PDE 行中的附加信息
            self._parse_pde_line(event, pde_line)

            return event

        except (ValueError, IndexError) as e:
            self.logger.debug(f"CMTSOLUTION 解析失败: {e}")
            return None

    def _parse_pde_line(self, event: CMT3DEvent, pde_line: str):
        """解析 PDE 行中的时间和震级信息"""
        try:
            parts = pde_line.strip().split()
            if len(parts) < 11 or not parts[0].startswith('PDE'):
                return

            # PDE/PDEW/PDEQ YYYY MM DD HH MM SS.SS LAT LON DEP MB MS ...
            event.pde_year = int(parts[1])
            event.pde_month = int(parts[2])
            event.pde_day = int(parts[3])
            event.pde_hour = int(parts[4])
            event.pde_minute = int(parts[5])
            event.pde_second = float(parts[6])
            event.pde_lat = float(parts[7])
            event.pde_lon = float(parts[8])
            event.pde_depth = float(parts[9])
            event.pde_mb = float(parts[10]) if parts[10] != 'None' else 0.0
            if len(parts) > 11:
                try:
                    event.pde_ms = float(parts[11])
                except ValueError:
                    # SPECFEM CMTSOLUTION: mb 后紧跟震级，再往后是地名
                    if len(parts) > 12:
                        try:
                            event.pde_ms = float(parts[12])
                        except ValueError:
                            pass
        except (ValueError, IndexError):
            pass

    # ----------------------------------------------------------------
    # 事件筛选
    # ----------------------------------------------------------------

    def filter_events(self, events: Optional[List[CMT3DEvent]] = None) -> List[CMT3DEvent]:
        """
        根据区域、震级、深度条件筛选事件

        Args:
            events: 事件列表，默认使用 self.all_events

        Returns:
            筛选后的事件列表
        """
        if events is None:
            events = self.all_events

        if not events:
            self.logger.warning("⚠️  没有可筛选的事件")
            return []

        initial_count = len(events)
        self.logger.info(f"🔍 开始事件筛选，原始事件数: {initial_count}")

        region = self.base_config.region
        mag_range = self.config.event_filters['magnitude_range']
        depth_range = self.config.event_filters['depth_range']
        time_range = self.config.event_filters['time_range']

        filtered = []
        stats = {'region': 0, 'magnitude': 0, 'depth': 0, 'time': 0}

        for ev in events:
            # 区域筛选
            if not (region['lat_min'] <= ev.latitude <= region['lat_max'] and
                    region['lon_min'] <= ev.longitude <= region['lon_max']):
                stats['region'] += 1
                continue

            # 深度筛选
            if not (depth_range[0] <= ev.depth <= depth_range[1]):
                stats['depth'] += 1
                continue

            # 震级筛选（使用 Mw）
            mw = ev.best_magnitude
            if not (mag_range[0] <= mw <= mag_range[1]):
                stats['magnitude'] += 1
                continue

            # 时间筛选
            if ev.pde_year < time_range[0] or ev.pde_year > time_range[1]:
                stats['time'] += 1
                continue

            filtered.append(ev)

        self.filtered_events = filtered

        self.logger.info("📊 筛选统计:")
        self.logger.info(f"  原始事件: {initial_count}")
        self.logger.info(f"  区域排除: {stats['region']}")
        self.logger.info(f"  深度排除: {stats['depth']}")
        self.logger.info(f"  震级排除: {stats['magnitude']}")
        self.logger.info(f"  时间排除: {stats['time']}")
        self.logger.info(f"  筛选后: {len(filtered)}")

        return filtered

    # ----------------------------------------------------------------
    # FWEA23 参考事件匹配
    # ----------------------------------------------------------------

    @staticmethod
    def _parse_gcmt_event_id(gcmt_id: str) -> Optional[Dict[str, int]]:
        """
        解析 GCMT 事件 ID 提取时间信息

        GCMT ID 格式: C200604300043A
        C + YYYY + MM + DD + HH + MM + suffix
        """
        m = re.match(r'[A-Z](\d{4})(\d{2})(\d{2})(\d{2})(\d{2})\w*', str(gcmt_id))
        if m:
            return {
                'year': int(m.group(1)),
                'month': int(m.group(2)),
                'day': int(m.group(3)),
                'hour': int(m.group(4)),
                'minute': int(m.group(5)),
            }
        return None

    def load_fwea23_reference_events(self) -> pd.DataFrame:
        """
        加载 FWEA23 模型原作者使用的 141 个参考事件

        Returns:
            包含 (event_id, longitude, latitude, depth, year, month, day, hour, minute) 的 DataFrame
        """
        test_db_dir = self.base_config.dirs['data'] / 'events' / 'test_database'
        excel_path = test_db_dir / self.config.fwea23['excel_file']

        if not excel_path.exists():
            raise FileNotFoundError(f"FWEA23 参考事件文件不存在: {excel_path}")

        self.logger.info(f"📖 加载 FWEA23 参考事件: {excel_path}")

        raw = pd.read_excel(excel_path, header=None, skiprows=2)
        events = []
        for _, row in raw.iterrows():
            gcmt_id = str(row.iloc[0])
            time_info = self._parse_gcmt_event_id(gcmt_id)
            if time_info is None:
                continue
            try:
                events.append({
                    'fwea23_event_id': gcmt_id,
                    'longitude': float(row.iloc[1]),
                    'latitude': float(row.iloc[2]),
                    'depth': float(row.iloc[3]),
                    **time_info,
                })
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(events)
        self.logger.info(f"📊 FWEA23 参考事件: {len(df)} 个")
        return df

    def match_fwea23_with_cmt3d(
        self,
        fwea23_df: pd.DataFrame,
        cmt3d_events: List[CMT3DEvent],
    ) -> List[CMT3DEvent]:
        """
        将 FWEA23 参考事件与 CMT3D 目录按时间匹配，获取完整矩张量信息

        匹配规则: 年、月、日、时 完全相同，分钟差 ≤ tolerance

        Args:
            fwea23_df: FWEA23 参考事件 DataFrame
            cmt3d_events: CMT3D 全部事件列表

        Returns:
            匹配成功的 CMT3DEvent 列表（带完整矩张量）
        """
        tolerance = self.config.fwea23['time_match_tolerance_min']
        self.logger.info(
            f"🔗 开始 FWEA23-CMT3D 时间匹配 (容差: ±{tolerance} 分钟)..."
        )

        # 建立 CMT3D 时间索引加速查找
        cmt3d_index: Dict[str, List[CMT3DEvent]] = {}
        for ev in cmt3d_events:
            key = f"{ev.pde_year:04d}{ev.pde_month:02d}{ev.pde_day:02d}{ev.pde_hour:02d}"
            cmt3d_index.setdefault(key, []).append(ev)

        matched_events: List[CMT3DEvent] = []
        unmatched_ids: List[str] = []

        for _, ref in fwea23_df.iterrows():
            key = f"{ref['year']:04d}{ref['month']:02d}{ref['day']:02d}{ref['hour']:02d}"
            candidates = cmt3d_index.get(key, [])

            best_match = None
            best_dt = float('inf')

            for cmt_ev in candidates:
                dt_min = abs(cmt_ev.pde_minute - ref['minute'])
                if dt_min <= tolerance and dt_min < best_dt:
                    best_dt = dt_min
                    best_match = cmt_ev

            if best_match is not None:
                matched_events.append(best_match)
            else:
                unmatched_ids.append(ref['fwea23_event_id'])

        self.logger.info(f"✅ 匹配成功: {len(matched_events)}/{len(fwea23_df)}")
        if unmatched_ids:
            self.logger.warning(
                f"⚠️  未匹配: {len(unmatched_ids)} 个事件 "
                f"(前5: {unmatched_ids[:5]})"
            )

        return matched_events

    # ----------------------------------------------------------------
    # 智能事件采样
    # ----------------------------------------------------------------

    def select_representative_events(
        self,
        events: Optional[List[CMT3DEvent]] = None
    ) -> List[CMT3DEvent]:
        """
        从筛选后的事件中智能采样代表性事件

        采样策略:
        1. 按深度分层（浅/中/深）分配配额
        2. 每个深度层内按震级降序排列（优先高信噪比事件）
        3. 贪心空间去重：新事件与已选事件距离 >= min_event_distance_deg

        Args:
            events: 候选事件列表，默认使用 self.filtered_events

        Returns:
            选中的代表性事件列表
        """
        if events is None:
            events = self.filtered_events

        if not events:
            self.logger.warning("⚠️  没有候选事件可供采样")
            return []

        cfg = self.config.sampling
        target = cfg['target_count']
        min_dist = cfg['min_event_distance_deg']
        rng = np.random.RandomState(cfg['random_seed'])

        self.logger.info(f"🎲 开始智能采样 (目标: {target}, 最小间距: {min_dist}°)")

        # 按深度分组
        depth_groups: Dict[str, List[CMT3DEvent]] = {
            'shallow': [],
            'intermediate': [],
            'deep': [],
        }
        for ev in events:
            depth_groups[ev.depth_category].append(ev)

        for cat, evs in depth_groups.items():
            self.logger.info(f"  {cat}: {len(evs)} 个候选事件")

        # 计算每个深度层的配额
        quotas: Dict[str, int] = {}
        for cat, ratio in cfg['depth_quotas'].items():
            available = len(depth_groups[cat])
            quota = min(int(target * ratio) + 1, available)
            quotas[cat] = quota

        # 如果某层候选不足，将余额分配给其他层
        total_assigned = sum(quotas.values())
        if total_assigned < target:
            deficit = target - total_assigned
            for cat in ['shallow', 'intermediate', 'deep']:
                if deficit <= 0:
                    break
                available_extra = len(depth_groups[cat]) - quotas[cat]
                add = min(deficit, available_extra)
                quotas[cat] += add
                deficit -= add

        self.logger.info(f"📊 深度配额: {quotas}")

        # 对每个深度层执行空间去重采样
        selected: List[CMT3DEvent] = []
        for cat in ['shallow', 'intermediate', 'deep']:
            candidates = depth_groups[cat].copy()
            quota = quotas[cat]

            if cfg['prefer_larger_magnitude']:
                candidates.sort(key=lambda e: e.best_magnitude, reverse=True)
            else:
                rng.shuffle(candidates)

            layer_selected = self._spatial_decimation(
                candidates, quota, min_dist, already_selected=selected
            )
            selected.extend(layer_selected)
            self.logger.info(
                f"  {cat}: 选取 {len(layer_selected)}/{quota} 个事件"
            )

        # 最终排序（按时间）
        selected.sort(key=lambda e: (e.pde_year, e.pde_month, e.pde_day))

        self.selected_events = selected
        self.logger.info(f"✅ 智能采样完成: 共选取 {len(selected)} 个事件")

        return selected

    def _spatial_decimation(
        self,
        candidates: List[CMT3DEvent],
        quota: int,
        min_dist_deg: float,
        already_selected: Optional[List[CMT3DEvent]] = None,
    ) -> List[CMT3DEvent]:
        """
        贪心空间去重采样

        Args:
            candidates: 候选事件（已按优先级排序）
            quota: 目标数量
            min_dist_deg: 最小事件间距（度）
            already_selected: 已选事件列表（跨层检查）

        Returns:
            选中的事件列表
        """
        selected = []
        ref_positions = []

        if already_selected:
            ref_positions = [(e.latitude, e.longitude) for e in already_selected]

        for ev in candidates:
            if len(selected) >= quota:
                break

            too_close = False
            for ref_lat, ref_lon in ref_positions:
                dist = self._angular_distance(
                    ev.latitude, ev.longitude, ref_lat, ref_lon
                )
                if dist < min_dist_deg:
                    too_close = True
                    break

            if not too_close:
                selected.append(ev)
                ref_positions.append((ev.latitude, ev.longitude))

        return selected

    @staticmethod
    def _angular_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine 公式计算两点间的角距离（度）"""
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
        return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(a))))

    # ----------------------------------------------------------------
    # hybrid_30：与 10 事件互补的空间均匀采样
    # ----------------------------------------------------------------

    def load_excluded_hybrid_events(self) -> List[Dict[str, Any]]:
        """读取既有 hybrid_10events，作为排除列表。"""
        cfg = self.config.hybrid_forward
        events_dir = self.base_config.dirs['events']
        excluded: List[Dict[str, Any]] = []

        id_map = events_dir / cfg['exclude_id_map']
        if id_map.exists():
            df = pd.read_csv(id_map)
            for _, row in df.iterrows():
                excluded.append({
                    'event_name': str(row.get('event', '')).strip(),
                    'dir': str(row.get('dir', '')).strip(),
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                })
            self.logger.info(f"🚫 从 {id_map.name} 读取 {len(excluded)} 个既有事件")

        if excluded:
            return excluded

        par_file = events_dir / cfg['exclude_par']
        if not par_file.exists():
            self.logger.warning(f"⚠️  未找到既有 10 事件目录: {par_file}")
            return []

        with open(par_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                excluded.append({
                    'event_name': '',
                    'dir': parts[0],
                    'lat': float(parts[5]),
                    'lon': float(parts[6]),
                })
        self.logger.info(f"🚫 从 {par_file.name} 读取 {len(excluded)} 个既有事件")
        return excluded

    def _event_dir_name(self, ev: CMT3DEvent) -> str:
        """与 MassDownloader / hybrid_10events 一致的事件目录名 YYYYMMDD.HH.MM"""
        dt = ev.origin_datetime
        if dt is None:
            return f"{ev.pde_year:04d}{ev.pde_month:02d}{ev.pde_day:02d}.{ev.pde_hour:02d}.{ev.pde_minute:02d}"
        return dt.strftime('%Y%m%d.%H.%M')

    def _is_excluded_event(
        self,
        ev: CMT3DEvent,
        excluded: List[Dict[str, Any]],
        dist_deg: float,
    ) -> bool:
        """按事件名、目录名或空间邻近判断是否与既有 10 事件重复。"""
        ev_dir = self._event_dir_name(ev)
        for ref in excluded:
            ref_name = str(ref.get('event_name', '')).strip()
            if ref_name and ev.event_name == ref_name:
                return True
            ref_dir = str(ref.get('dir', '')).strip()
            if ref_dir and ev_dir == ref_dir:
                return True
            dist = self._angular_distance(
                ev.latitude, ev.longitude, float(ref['lat']), float(ref['lon'])
            )
            if dist < dist_deg:
                return True
        return False

    def _in_hybrid_forward_region(self, ev: CMT3DEvent) -> bool:
        """正演子区域：模型重叠区，去掉印度洋中脊等无约束角落。"""
        cfg = self.config.hybrid_forward
        lat_min, lat_max = cfg['lat_range']
        lon_min, lon_max = cfg['lon_range']
        if not (lat_min <= ev.latitude <= lat_max and lon_min <= ev.longitude <= lon_max):
            return False
        sw = cfg.get('exclude_south_west')
        if sw and ev.latitude < sw['lat_max'] and ev.longitude < sw['lon_max']:
            return False
        return True

    def select_spatially_uniform_events(
        self,
        events: Optional[List[CMT3DEvent]] = None,
    ) -> List[CMT3DEvent]:
        """
        空间均匀采样 30 个事件，排除既有 hybrid_10events。

        策略:
        1. 剔除与旧 10 事件同名或 <1.5° 的事件
        2. 按深度配额（浅 15 / 中 8 / 深 7）
        3. 地理种子点就近取较新事件，铺开覆盖
        4. 最远点采样（FPS）填满配额，间距优先、年份次之
        """
        if events is None:
            events = self.filtered_events

        cfg = self.config.hybrid_forward
        excluded = self.load_excluded_hybrid_events()
        exclude_dist = float(cfg['exclude_distance_deg'])
        target = int(cfg['target_count'])
        min_dist = float(cfg['min_event_distance_deg'])
        relax_dist = float(cfg['relax_distance_deg'])
        quotas: Dict[str, int] = dict(cfg['depth_quotas'])

        pool = [
            ev for ev in events
            if not self._is_excluded_event(ev, excluded, exclude_dist)
            and self._in_hybrid_forward_region(ev)
        ]
        self.logger.info(
            f"🎲 hybrid_30 候选: {len(events)} → 排除后 {len(pool)} "
            f"(排除半径 {exclude_dist}°, 正演子区域)"
        )

        groups: Dict[str, List[CMT3DEvent]] = {
            'shallow': [], 'intermediate': [], 'deep': [],
        }
        for ev in pool:
            groups[ev.depth_category].append(ev)
        for cat, evs in groups.items():
            self.logger.info(f"  {cat}: {len(evs)} 个候选, 配额 {quotas[cat]}")

        selected: List[CMT3DEvent] = []
        selected_names = set()

        def _layer_count(cat: str) -> int:
            return sum(1 for s in selected if s.depth_category == cat)

        def _too_close(ev: CMT3DEvent, dist_lim: float) -> bool:
            return any(
                self._angular_distance(
                    ev.latitude, ev.longitude, s.latitude, s.longitude
                ) < dist_lim
                for s in selected
            )

        # 1) 地理种子：每个种子在搜索半径内取较新、较大震级事件
        seed_radius = float(cfg['seed_search_radius_deg'])
        for lat0, lon0 in cfg['seed_points']:
            if len(selected) >= target:
                break
            eligible: List[CMT3DEvent] = []
            for cat, quota in quotas.items():
                if _layer_count(cat) >= quota:
                    continue
                for ev in groups[cat]:
                    if ev.event_name in selected_names:
                        continue
                    if _too_close(ev, min_dist):
                        continue
                    dist = self._angular_distance(
                        ev.latitude, ev.longitude, lat0, lon0
                    )
                    if dist <= seed_radius:
                        eligible.append(ev)
            if not eligible:
                continue
            eligible.sort(
                key=lambda e: (e.pde_year, e.best_magnitude),
                reverse=True,
            )
            pick = eligible[0]
            selected.append(pick)
            selected_names.add(pick.event_name)
            self.logger.info(
                f"  种子 ({lat0:.0f}N, {lon0:.0f}E) → {pick.event_name} "
                f"{pick.pde_year} Mw{pick.best_magnitude:.2f} {pick.depth_category}"
            )

        # 2) 各深度层最远点补齐
        for cat in ['shallow', 'intermediate', 'deep']:
            need = quotas[cat] - _layer_count(cat)
            if need <= 0:
                continue
            candidates = [
                ev for ev in groups[cat]
                if ev.event_name not in selected_names
            ]
            added = self._farthest_point_select(
                candidates, need, min_dist, already_selected=selected
            )
            for ev in added:
                selected.append(ev)
                selected_names.add(ev.event_name)
            self.logger.info(
                f"  {cat} FPS: 需要 {need}, 补入 {len(added)}"
            )

        # 3) 仍不足则放宽间距，跨层补齐
        if len(selected) < target:
            remaining = [ev for ev in pool if ev.event_name not in selected_names]
            deficit = target - len(selected)
            added = self._farthest_point_select(
                remaining, deficit, relax_dist, already_selected=selected
            )
            selected.extend(added)
            self.logger.info(
                f"  放宽间距 {relax_dist}° 后补入 {len(added)} 个"
            )

        selected.sort(
            key=lambda e: (
                e.pde_year, e.pde_month, e.pde_day,
                e.pde_hour, e.pde_minute,
            )
        )
        if len(selected) > target:
            selected = selected[:target]

        self.selected_events = selected
        self.logger.info(f"✅ hybrid_30 采样完成: {len(selected)} 个事件")
        return selected

    def _farthest_point_select(
        self,
        candidates: List[CMT3DEvent],
        quota: int,
        min_dist_deg: float,
        already_selected: Optional[List[CMT3DEvent]] = None,
        scoring_cfg: Optional[Dict[str, Any]] = None,
        coverage_lookup: Optional[Dict[str, int]] = None,
    ) -> List[CMT3DEvent]:
        """
        最远点采样：优先拉开空间距离，其次偏好较新年份、较大震级与更高波形覆盖。
        """
        cfg = scoring_cfg or self.config.hybrid_forward
        year_w = float(cfg.get('year_weight', 0.35))
        mag_w = float(cfg.get('magnitude_weight', 0.15))
        cov_w = float(cfg.get('coverage_weight', 0.0))
        mainland_w = float(cfg.get('mainland_weight', 0.0))
        time_range = cfg.get('time_range', [2000, 2025])
        mag_range = cfg.get('magnitude_range', [5.5, 7.5])
        year0, year1 = time_range
        mag0, mag1 = mag_range
        year_span = max(1, year1 - year0)
        mag_span = max(0.1, mag1 - mag0)
        max_coverage = 1
        if coverage_lookup:
            max_coverage = max(coverage_lookup.values() or [1])

        selected: List[CMT3DEvent] = []
        refs = [
            (e.latitude, e.longitude)
            for e in (already_selected or [])
        ]
        remaining = list(candidates)

        while len(selected) < quota and remaining:
            best: Optional[CMT3DEvent] = None
            best_score = -1.0e9
            for ev in remaining:
                if refs:
                    min_d = min(
                        self._angular_distance(
                            ev.latitude, ev.longitude, rlat, rlon
                        )
                        for rlat, rlon in refs
                    )
                else:
                    min_d = 90.0
                if min_d < min_dist_deg:
                    continue
                year_s = (ev.pde_year - year0) / year_span
                mag_s = (ev.best_magnitude - mag0) / mag_span
                cov_s = 0.0
                if coverage_lookup and ev.event_name in coverage_lookup:
                    cov_s = coverage_lookup[ev.event_name] / max_coverage
                mainland_s = self._mainland_interior_score(ev, cfg)
                score = (
                    min_d
                    + year_w * 8.0 * year_s
                    + mag_w * 4.0 * mag_s
                    + cov_w * 6.0 * cov_s
                    + mainland_w * 6.0 * mainland_s
                )
                if score > best_score:
                    best_score = score
                    best = ev
            if best is None:
                break
            selected.append(best)
            remaining.remove(best)
            refs.append((best.latitude, best.longitude))

        return selected

    # ----------------------------------------------------------------
    # eastasia_model_30：从已有波形库选取 30 个内部正演测试事件
    # ----------------------------------------------------------------

    @property
    def _eastasia_dataset_dir(self) -> Path:
        return self.eastasia_model_dir

    def parse_eastasia_model_catalog(self) -> List[CMT3DEvent]:
        """
        解析 EastAsia_model/src_rec 下全部 CMTSOLUTION 文件。

        Returns:
            解析成功的事件列表
        """
        src_rec = self._eastasia_dataset_dir / 'src_rec'
        if not src_rec.exists():
            raise FileNotFoundError(f"EastAsia_model 目录不存在: {src_rec}")

        events: List[CMT3DEvent] = []
        for cmt_file in sorted(src_rec.glob('CMTSOLUTION_*')):
            try:
                lines = cmt_file.read_text(encoding='utf-8', errors='replace').splitlines()
                event = self._parse_single_cmt3d_event(lines[:13])
                if event is not None:
                    events.append(event)
            except OSError as exc:
                self.logger.debug(f"读取失败 {cmt_file.name}: {exc}")

        self.all_events = events
        self.logger.info(f"📖 EastAsia_model 目录解析完成: {len(events)} 个事件")
        return events

    def load_eastasia_waveform_coverage(self) -> Dict[str, int]:
        """
        加载每事件 SAC 覆盖台站数（若 catalog 不存在则现场统计）。

        Returns:
            Event_ID → N_Stations 字典
        """
        cfg = self.config.eastasia_model_forward
        cov_path = self._eastasia_dataset_dir / cfg['waveform_coverage_csv']
        coverage: Dict[str, int] = {}

        if cov_path.exists():
            df = pd.read_csv(cov_path)
            for _, row in df.iterrows():
                coverage[str(row['Event_ID'])] = int(row.get('N_Stations', 0))
            self.logger.info(f"📊 加载波形覆盖统计: {len(coverage)} 个事件")
        else:
            data_obs = self._eastasia_dataset_dir / 'data_obs'
            for ev_dir in sorted(data_obs.iterdir()):
                if ev_dir.is_dir():
                    stations = {
                        p.name.split('.')[0] + '.' + p.name.split('.')[1]
                        for p in ev_dir.glob('*.sac')
                        if len(p.stem.split('.')) >= 2
                    }
                    coverage[ev_dir.name] = len(stations)
            self.logger.info(f"📊 现场统计波形覆盖: {len(coverage)} 个事件")

        self.waveform_coverage = coverage
        return coverage

    def _distance_to_region_boundary(self, latitude: float, longitude: float) -> float:
        """计算事件到研究区域边界的最小距离（度）。"""
        region = self.base_config.region
        dist_lon = min(
            longitude - region['lon_min'],
            region['lon_max'] - longitude,
        )
        dist_lat = min(
            latitude - region['lat_min'],
            region['lat_max'] - latitude,
        )
        return min(dist_lon, dist_lat)

    def _is_interior_event(self, ev: CMT3DEvent, min_margin_deg: float) -> bool:
        """判断事件是否远离研究区边界。"""
        return self._distance_to_region_boundary(ev.latitude, ev.longitude) >= min_margin_deg

    def _passes_mainland_hard_filter(
        self,
        ev: CMT3DEvent,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """硬排除 lat<min_latitude 或 lon>max_longitude 的边界事件。"""
        cfg = cfg or self.config.eastasia_model_forward
        min_lat = cfg.get('min_latitude_deg')
        max_lon = cfg.get('max_longitude_deg')
        if min_lat is not None and ev.latitude < float(min_lat):
            return False
        if max_lon is not None and ev.longitude > float(max_lon):
            return False
        return True

    @staticmethod
    def _mainland_interior_score(
        ev: CMT3DEvent,
        cfg: Dict[str, Any],
    ) -> float:
        """大陆内部偏好评分 [0, 1]：位于 mainland_lat/lon_range 内为 1。"""
        lat_range = cfg.get('mainland_lat_range', [5.0, 50.0])
        lon_range = cfg.get('mainland_lon_range', [70.0, 130.0])
        lat0, lat1 = float(lat_range[0]), float(lat_range[1])
        lon0, lon1 = float(lon_range[0]), float(lon_range[1])
        if lat0 <= ev.latitude <= lat1 and lon0 <= ev.longitude <= lon1:
            return 1.0
        d_lat = 0.0
        if ev.latitude < lat0:
            d_lat = lat0 - ev.latitude
        elif ev.latitude > lat1:
            d_lat = ev.latitude - lat1
        d_lon = 0.0
        if ev.longitude < lon0:
            d_lon = lon0 - ev.longitude
        elif ev.longitude > lon1:
            d_lon = ev.longitude - lon1
        penalty = math.sqrt(d_lat ** 2 + d_lon ** 2)
        return max(0.0, 1.0 - penalty / 20.0)

    def _has_waveform_data(self, event_name: str) -> bool:
        """检查 data_obs 中是否存在该事件的波形目录。"""
        return (self._eastasia_dataset_dir / 'data_obs' / event_name).is_dir()

    def load_or_compute_observation_window(self) -> Dict[str, Any]:
        """
        加载或计算观测台站数最多的事件时间窗口。

        优先读取 5_11 生成的 catalog/optimal_event_window.json；
        若不存在则基于波形覆盖统计现场计算。

        Returns:
            含 start_year/end_year 的窗口字典
        """
        cfg = self.config.eastasia_model_forward
        manual = cfg.get('observation_window')
        if manual and isinstance(manual, dict):
            self.observation_window = manual
            return manual

        window_path = self._eastasia_dataset_dir / 'catalog' / 'optimal_event_window.json'
        if window_path.exists():
            with open(window_path, 'r', encoding='utf-8') as f:
                self.observation_window = json.load(f)
            self.logger.info(
                f"📅 加载最优观测窗口: "
                f"{self.observation_window['start_year']}–"
                f"{self.observation_window['end_year']}"
            )
            return self.observation_window

        if self.waveform_coverage is None:
            self.load_eastasia_waveform_coverage()

        window_years = int(cfg.get('observation_window_years', 5))
        min_events = int(cfg.get('target_count', 30))
        year_stats: Dict[int, Dict[str, float]] = {}

        for ev in self.all_events:
            if ev.pde_year <= 0:
                continue
            n_sta = (self.waveform_coverage or {}).get(ev.event_name, 0)
            if n_sta <= 0:
                continue
            bucket = year_stats.setdefault(ev.pde_year, {'n': 0, 'sum_sta': 0.0, 'max_sta': 0})
            bucket['n'] += 1
            bucket['sum_sta'] += n_sta
            bucket['max_sta'] = max(bucket['max_sta'], n_sta)

        if not year_stats:
            self.observation_window = {}
            return {}

        years = sorted(year_stats.keys())
        best: Optional[Dict[str, Any]] = None
        for start in range(years[0], years[-1] - window_years + 2):
            end = start + window_years - 1
            n_events = sum(year_stats[y]['n'] for y in range(start, end + 1) if y in year_stats)
            if n_events < min_events:
                continue
            sta_vals = [
                year_stats[y]['sum_sta'] / year_stats[y]['n']
                for y in range(start, end + 1) if y in year_stats and year_stats[y]['n'] > 0
            ]
            if not sta_vals:
                continue
            mean_sta = sum(sta_vals) / len(sta_vals)
            if best is None or mean_sta > best['mean_stations_per_event']:
                best = {
                    'start_year': start,
                    'end_year': end,
                    'window_years': window_years,
                    'n_events': n_events,
                    'mean_stations_per_event': round(mean_sta, 1),
                    'selection_criterion': 'max_mean_stations_per_event',
                }

        self.observation_window = best or {}
        if best:
            self.logger.info(
                f"📅 计算最优观测窗口: {best['start_year']}–{best['end_year']} "
                f"(均值 {best['mean_stations_per_event']:.0f} 台站/事件)"
            )
        return self.observation_window

    def filter_eastasia_model_candidates(
        self,
        events: Optional[List[CMT3DEvent]] = None,
    ) -> List[CMT3DEvent]:
        """
        筛选 EastAsia_model 正演候选事件池（Mw≥5.5、有波形、远离边界）。

        Args:
            events: 原始事件列表，默认 self.all_events

        Returns:
            候选事件列表
        """
        if events is None:
            events = self.all_events

        cfg = self.config.eastasia_model_forward
        mag_range = cfg['magnitude_range']
        depth_range = cfg['depth_range']
        margin = float(cfg['min_interior_margin_deg'])
        region = self.base_config.region

        if cfg['require_waveform'] and self.waveform_coverage is None:
            self.load_eastasia_waveform_coverage()

        temporal_quotas = cfg.get('temporal_quotas') or []
        time_window: Optional[Dict[str, Any]] = None
        year_min: Optional[int] = None
        year_max: Optional[int] = None

        if temporal_quotas:
            year_min = min(int(q['start_year']) for q in temporal_quotas)
            year_max = max(int(q['end_year']) for q in temporal_quotas)
        elif cfg.get('use_optimal_observation_window', True):
            time_window = self.load_or_compute_observation_window()
            if time_window:
                year_min = int(time_window['start_year'])
                year_max = int(time_window['end_year'])

        filtered: List[CMT3DEvent] = []
        stats = {
            'region': 0, 'magnitude': 0, 'depth': 0,
            'edge': 0, 'mainland': 0, 'no_waveform': 0, 'time_window': 0,
        }

        for ev in events:
            if not (region['lat_min'] <= ev.latitude <= region['lat_max'] and
                    region['lon_min'] <= ev.longitude <= region['lon_max']):
                stats['region'] += 1
                continue
            if not (depth_range[0] <= ev.depth <= depth_range[1]):
                stats['depth'] += 1
                continue
            mw = ev.best_magnitude
            if not (mag_range[0] <= mw <= mag_range[1]):
                stats['magnitude'] += 1
                continue
            if not self._is_interior_event(ev, margin):
                stats['edge'] += 1
                continue
            if not self._passes_mainland_hard_filter(ev, cfg):
                stats['mainland'] += 1
                continue
            if cfg['require_waveform'] and not self._has_waveform_data(ev.event_name):
                stats['no_waveform'] += 1
                continue
            if year_min is not None and year_max is not None and (
                ev.pde_year < year_min or ev.pde_year > year_max
            ):
                stats['time_window'] += 1
                continue
            filtered.append(ev)

        self.filtered_events = filtered
        window_msg = ''
        if temporal_quotas:
            parts = [
                f"{q['start_year']}–{q['end_year']}×{q['count']}"
                for q in temporal_quotas
            ]
            window_msg = f", 时段配额 {' + '.join(parts)}"
        elif time_window:
            window_msg = (
                f", 观测窗口 {time_window['start_year']}–{time_window['end_year']}"
            )
        self.logger.info(
            f"🔍 EastAsia_model 候选池: {len(events)} → {len(filtered)} "
            f"(边缘排除 {stats['edge']}, 边界 {margin}°{window_msg})"
        )
        self.logger.info(
            f"  区域/深度/震级/边界/大陆硬筛/无波形/时间窗排除: "
            f"{stats['region']}/{stats['depth']}/{stats['magnitude']}/"
            f"{stats['edge']}/{stats['mainland']}/{stats['no_waveform']}/"
            f"{stats['time_window']}"
        )
        return filtered

    @staticmethod
    def _allocate_depth_sub_quotas(
        bucket_count: int,
        depth_quotas: Dict[str, int],
    ) -> Dict[str, int]:
        """按全局深度配额比例分配单个时段桶内的深度目标数。"""
        total = sum(depth_quotas.values())
        if total <= 0 or bucket_count <= 0:
            return {cat: 0 for cat in depth_quotas}

        raw = {
            cat: bucket_count * quota / total
            for cat, quota in depth_quotas.items()
        }
        sub = {cat: int(raw[cat]) for cat in depth_quotas}
        remainder = bucket_count - sum(sub.values())
        if remainder > 0:
            order = sorted(depth_quotas, key=lambda c: raw[c] - sub[c], reverse=True)
            for cat in order:
                if remainder <= 0:
                    break
                sub[cat] += 1
                remainder -= 1
        return sub

    def _select_events_in_temporal_bucket(
        self,
        bucket_pool: List[CMT3DEvent],
        bucket_quota: int,
        depth_sub_quotas: Dict[str, int],
        already_selected: List[CMT3DEvent],
        selected_names: set,
        cfg: Dict[str, Any],
        coverage: Dict[str, int],
        min_dist: float,
        relax_dist: float,
        bucket_label: str,
    ) -> List[CMT3DEvent]:
        """在单个时段桶内做深度分层 + 空间均匀采样。"""
        selected: List[CMT3DEvent] = []

        def _layer_count(cat: str) -> int:
            return sum(1 for s in selected if s.depth_category == cat)

        def _too_close(ev: CMT3DEvent, dist_lim: float) -> bool:
            refs = already_selected + selected
            return any(
                self._angular_distance(
                    ev.latitude, ev.longitude, s.latitude, s.longitude
                ) < dist_lim
                for s in refs
            )

        groups: Dict[str, List[CMT3DEvent]] = {
            'shallow': [], 'intermediate': [], 'deep': [],
        }
        for ev in bucket_pool:
            groups[ev.depth_category].append(ev)

        seed_radius = float(cfg['seed_search_radius_deg'])
        for lat0, lon0 in cfg['seed_points']:
            if len(selected) >= bucket_quota:
                break
            eligible: List[CMT3DEvent] = []
            for cat, quota in depth_sub_quotas.items():
                if _layer_count(cat) >= quota:
                    continue
                for ev in groups[cat]:
                    if ev.event_name in selected_names or _too_close(ev, min_dist):
                        continue
                    if self._angular_distance(
                        ev.latitude, ev.longitude, lat0, lon0
                    ) <= seed_radius:
                        eligible.append(ev)
            if not eligible:
                continue
            eligible.sort(
                key=lambda e: (
                    self._mainland_interior_score(e, cfg),
                    coverage.get(e.event_name, 0),
                    e.best_magnitude,
                ),
                reverse=True,
            )
            pick = eligible[0]
            selected.append(pick)
            selected_names.add(pick.event_name)

        for cat in ['shallow', 'intermediate', 'deep']:
            need = depth_sub_quotas[cat] - _layer_count(cat)
            if need <= 0:
                continue
            candidates = [
                ev for ev in groups[cat]
                if ev.event_name not in selected_names
            ]
            added = self._farthest_point_select(
                candidates, need, min_dist,
                already_selected=already_selected + selected,
                scoring_cfg=cfg,
                coverage_lookup=coverage,
            )
            for ev in added:
                selected.append(ev)
                selected_names.add(ev.event_name)

        if len(selected) < bucket_quota:
            remaining = [
                ev for ev in bucket_pool
                if ev.event_name not in selected_names
            ]
            deficit = bucket_quota - len(selected)
            added = self._farthest_point_select(
                remaining, deficit, relax_dist,
                already_selected=already_selected + selected,
                scoring_cfg=cfg,
                coverage_lookup=coverage,
            )
            for ev in added:
                selected.append(ev)
                selected_names.add(ev.event_name)
            if added:
                self.logger.info(
                    f"  {bucket_label}: 放宽间距 {relax_dist}° 补入 {len(added)} 个"
                )

        if len(selected) > bucket_quota:
            selected.sort(
                key=lambda e: (
                    self._mainland_interior_score(e, cfg),
                    coverage.get(e.event_name, 0),
                ),
                reverse=True,
            )
            drop = selected[bucket_quota:]
            for ev in drop:
                selected_names.discard(ev.event_name)
            selected = selected[:bucket_quota]

        return selected

    def select_eastasia_model_30_events(
        self,
        events: Optional[List[CMT3DEvent]] = None,
    ) -> List[CMT3DEvent]:
        """
        从 EastAsia_model 候选池中采样正演测试事件。

        策略：
        - 候选池已剔除研究区边缘事件
        - 按 temporal_quotas 分时段配额（默认 2008–13×5, 2014–18×15, 2019–25×5）
        - 各时段内深度分层 + 空间均匀采样，评分含波形覆盖台站数
        - 事件目录名使用 event_name（与 data_obs 一致）
        """
        if events is None:
            events = self.filtered_events

        cfg = self.config.eastasia_model_forward
        target = int(cfg['target_count'])
        min_dist = float(cfg['min_event_distance_deg'])
        relax_dist = float(cfg['relax_distance_deg'])
        depth_quotas: Dict[str, int] = dict(cfg['depth_quotas'])
        temporal_quotas = cfg.get('temporal_quotas') or []
        coverage = self.waveform_coverage or {}

        pool = list(events)
        self.logger.info(f"🎲 eastasia_model 候选池: {len(pool)} 个事件")

        if temporal_quotas:
            quota_sum = sum(int(q['count']) for q in temporal_quotas)
            if quota_sum != target:
                self.logger.warning(
                    f"⚠️  temporal_quotas 合计 {quota_sum} ≠ target_count {target}"
                )

        groups: Dict[str, List[CMT3DEvent]] = {
            'shallow': [], 'intermediate': [], 'deep': [],
        }
        for ev in pool:
            groups[ev.depth_category].append(ev)
        for cat, evs in groups.items():
            self.logger.info(f"  {cat}: {len(evs)} 个候选, 全局配额 {depth_quotas[cat]}")

        selected: List[CMT3DEvent] = []
        selected_names: set = set()

        if temporal_quotas:
            for bucket in temporal_quotas:
                y0 = int(bucket['start_year'])
                y1 = int(bucket['end_year'])
                bucket_quota = int(bucket['count'])
                bucket_label = f"{y0}–{y1}"
                bucket_pool = [
                    ev for ev in pool
                    if y0 <= ev.pde_year <= y1
                ]
                depth_sub = self._allocate_depth_sub_quotas(
                    bucket_quota, depth_quotas
                )
                self.logger.info(
                    f"📅 时段 {bucket_label}: 候选 {len(bucket_pool)}, "
                    f"目标 {bucket_quota}, 深度 {depth_sub}"
                )
                bucket_selected = self._select_events_in_temporal_bucket(
                    bucket_pool=bucket_pool,
                    bucket_quota=bucket_quota,
                    depth_sub_quotas=depth_sub,
                    already_selected=selected,
                    selected_names=selected_names,
                    cfg=cfg,
                    coverage=coverage,
                    min_dist=min_dist,
                    relax_dist=relax_dist,
                    bucket_label=bucket_label,
                )
                for ev in bucket_selected:
                    margin = self._distance_to_region_boundary(
                        ev.latitude, ev.longitude
                    )
                    self.logger.info(
                        f"  ✓ {ev.event_name} {ev.pde_year} "
                        f"Mw{ev.best_magnitude:.2f} {ev.depth_category} "
                        f"边距{margin:.1f}° sta={coverage.get(ev.event_name, 0)}"
                    )
                selected.extend(bucket_selected)
                self.logger.info(
                    f"  → {bucket_label} 已选 {len(bucket_selected)}/{bucket_quota}"
                )
        else:
            # 兼容旧逻辑：单一观测窗口 + 全局空间采样
            def _layer_count(cat: str) -> int:
                return sum(1 for s in selected if s.depth_category == cat)

            def _too_close(ev: CMT3DEvent, dist_lim: float) -> bool:
                return any(
                    self._angular_distance(
                        ev.latitude, ev.longitude, s.latitude, s.longitude
                    ) < dist_lim
                    for s in selected
                )

            seed_radius = float(cfg['seed_search_radius_deg'])
            for lat0, lon0 in cfg['seed_points']:
                if len(selected) >= target:
                    break
                eligible: List[CMT3DEvent] = []
                for cat, quota in depth_quotas.items():
                    if _layer_count(cat) >= quota:
                        continue
                    for ev in groups[cat]:
                        if ev.event_name in selected_names or _too_close(ev, min_dist):
                            continue
                        if self._angular_distance(
                            ev.latitude, ev.longitude, lat0, lon0
                        ) <= seed_radius:
                            eligible.append(ev)
                if not eligible:
                    continue
                eligible.sort(
                    key=lambda e: (
                        coverage.get(e.event_name, 0),
                        e.pde_year,
                        e.best_magnitude,
                    ),
                    reverse=True,
                )
                pick = eligible[0]
                selected.append(pick)
                selected_names.add(pick.event_name)

            for cat in ['shallow', 'intermediate', 'deep']:
                need = depth_quotas[cat] - _layer_count(cat)
                if need <= 0:
                    continue
                candidates = [
                    ev for ev in groups[cat]
                    if ev.event_name not in selected_names
                ]
                added = self._farthest_point_select(
                    candidates, need, min_dist,
                    already_selected=selected,
                    scoring_cfg=cfg,
                    coverage_lookup=coverage,
                )
                for ev in added:
                    selected.append(ev)
                    selected_names.add(ev.event_name)

            if len(selected) < target:
                remaining = [
                    ev for ev in pool if ev.event_name not in selected_names
                ]
                deficit = target - len(selected)
                added = self._farthest_point_select(
                    remaining, deficit, relax_dist,
                    already_selected=selected,
                    scoring_cfg=cfg,
                    coverage_lookup=coverage,
                )
                selected.extend(added)

        selected.sort(
            key=lambda e: (
                e.pde_year, e.pde_month, e.pde_day,
                e.pde_hour, e.pde_minute,
            )
        )
        if len(selected) > target:
            selected = selected[:target]

        self.selected_events = selected
        margins = [
            self._distance_to_region_boundary(e.latitude, e.longitude)
            for e in selected
        ]
        year_hist: Dict[int, int] = {}
        for ev in selected:
            year_hist[ev.pde_year] = year_hist.get(ev.pde_year, 0) + 1
        self.logger.info(
            f"✅ eastasia_model 采样完成: {len(selected)} 个事件, "
            f"边界距离 {min(margins):.1f}°–{max(margins):.1f}°"
        )
        if temporal_quotas:
            for bucket in temporal_quotas:
                y0, y1 = int(bucket['start_year']), int(bucket['end_year'])
                n = sum(1 for e in selected if y0 <= e.pde_year <= y1)
                self.logger.info(
                    f"  时段 {y0}–{y1}: {n}/{int(bucket['count'])}"
                )
        return selected

    def _eastasia_event_dir_name(self, ev: CMT3DEvent) -> str:
        """EastAsia_model 波形目录名与 event_name 一致。"""
        return ev.event_name

    def save_eastasia_model_download_catalog(
        self,
        events: Optional[List[CMT3DEvent]] = None,
    ) -> Path:
        """写出 EastAsia_model 30 事件下载/正演目录 (.par)。"""
        if events is None:
            events = self.selected_events

        cfg = self.config.eastasia_model_forward
        lines = []
        for ev in events:
            dt = ev.origin_datetime
            if dt is None:
                continue
            mag = ev.best_magnitude
            event_dir = self._eastasia_event_dir_name(ev)
            ymd = dt.strftime('%Y%m%d')
            hh = dt.strftime('%H')
            mm = dt.strftime('%M')
            sec = f"{ev.pde_second:.3f}"
            line = (
                f"{event_dir} {ymd} {hh} {mm} {sec} "
                f"{ev.latitude:.4f} {ev.longitude:.4f} {ev.depth:.1f} "
                f"8.4 25.6 {float(mag):.1f} Mw"
            )
            lines.append(line)

        text = '\n'.join(lines) + '\n'
        local_par = self.output_dir / 'download_catalog.par'
        root_par = self.base_config.dirs['events'] / cfg['root_par_name']
        for path in (local_par, root_par):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
        self.logger.info(
            f"✅ eastasia_model_30 目录: {root_par} ({len(lines)} 个事件)"
        )
        return root_par

    def save_eastasia_model_id_mapping(
        self,
        events: Optional[List[CMT3DEvent]] = None,
    ) -> Path:
        """写出 event_name ↔ 波形目录对照表。"""
        if events is None:
            events = self.selected_events

        cfg = self.config.eastasia_model_forward
        coverage = self.waveform_coverage or {}
        records = []
        for ev in events:
            dt = ev.origin_datetime
            margin = self._distance_to_region_boundary(ev.latitude, ev.longitude)
            records.append({
                'event': ev.event_name,
                'dir': self._eastasia_event_dir_name(ev),
                'origin_pde': dt.isoformat() if dt is not None else '',
                'lat': round(ev.latitude, 4),
                'lon': round(ev.longitude, 4),
                'depth_km': round(ev.depth, 2),
                'Mw': round(ev.best_magnitude, 2),
                'depth_category': ev.depth_category,
                'boundary_margin_deg': round(margin, 2),
                'n_stations': coverage.get(ev.event_name, 0),
                'waveform_dir': str(
                    self._eastasia_dataset_dir / 'data_obs' / ev.event_name
                ),
            })

        df = pd.DataFrame(records)
        local_csv = self.output_dir / cfg['id_map_name']
        root_csv = self.base_config.dirs['events'] / cfg['id_map_name']
        for path in (local_csv, root_csv):
            df.to_csv(path, index=False, encoding='utf-8')

        list_path = self.output_dir / cfg['events_list_name']
        with open(list_path, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(rec['dir'] + '\n')

        self.logger.info(f"✅ 事件对照表: {root_csv}")
        return root_csv

    def copy_eastasia_model_cmtsolution_files(
        self,
        events: Optional[List[CMT3DEvent]] = None,
    ) -> Path:
        """从 EastAsia_model/src_rec 复制原始 CMTSOLUTION 文件。"""
        if events is None:
            events = self.selected_events

        src_rec = self._eastasia_dataset_dir / 'src_rec'
        cmt_dir = self.output_dir / 'CMTSOLUTION'
        cmt_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for ev in events:
            src = src_rec / f"CMTSOLUTION_{ev.event_name}"
            dst = cmt_dir / f"CMTSOLUTION_{ev.event_name}"
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                with open(dst, 'w', encoding='utf-8') as f:
                    f.write(ev.to_cmtsolution())

        self.logger.info(f"✅ CMTSOLUTION 复制完成: {cmt_dir} ({copied}/{len(events)})")
        return cmt_dir

    def load_eastasia_model_stations(self) -> pd.DataFrame:
        """
        加载 EastAsia_model 台站目录（已有波形对应的台站坐标）。

        Returns:
            台站 DataFrame
        """
        cfg = self.config.eastasia_model_forward
        sta_path = self._eastasia_dataset_dir / cfg['stations_catalog']
        if not sta_path.exists():
            raise FileNotFoundError(f"EastAsia_model 台站目录不存在: {sta_path}")

        df = pd.read_csv(sta_path)
        rename_map = {
            'Network': 'Network', 'Station': 'Station',
            'Latitude': 'Latitude', 'Longitude': 'Longitude',
            'Elevation': 'Elevation',
        }
        for col in rename_map:
            if col not in df.columns:
                raise ValueError(f"台站目录缺少列: {col}")

        self.stations_df = df[list(rename_map.keys())].copy()
        self.logger.info(f"✅ 加载 EastAsia_model 台站: {len(df)} 个")
        return self.stations_df

    def plot_eastasia_model_selection_map(
        self,
        events: Optional[List[CMT3DEvent]] = None,
    ) -> Optional[Path]:
        """绘制 30 个选定事件与候选池对比分布图。"""
        if events is None:
            events = self.selected_events
        if not events:
            return None

        import matplotlib.pyplot as plt

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            use_cartopy = True
        except ImportError:
            use_cartopy = False

        region = self.base_config.region
        margin = float(self.config.eastasia_model_forward['min_interior_margin_deg'])
        lon_min, lon_max = region['lon_min'], region['lon_max']
        lat_min, lat_max = region['lat_min'], region['lat_max']
        inner = [
            lon_min + margin, lon_max - margin,
            lat_min + margin, lat_max - margin,
        ]
        depth_colors = {
            'shallow': '#d62728',
            'intermediate': '#ff7f0e',
            'deep': '#1f77b4',
        }

        fig_w, fig_h = 12, 8
        if use_cartopy:
            proj = ccrs.PlateCarree()
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), subplot_kw={'projection': proj})
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
            ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor='#dbe9f4', zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
            ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
            plot_kw = {'transform': proj}
        else:
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, linewidth=0.3, alpha=0.5)
            plot_kw = {}

        ax.plot(
            [inner[0], inner[1], inner[1], inner[0], inner[0]],
            [inner[2], inner[2], inner[3], inner[3], inner[2]],
            'k--', linewidth=1.0, alpha=0.6, label=f'Interior zone (margin {margin}°)',
            **plot_kw,
        )

        pool = self.filtered_events or []
        if pool:
            ax.scatter(
                [e.longitude for e in pool],
                [e.latitude for e in pool],
                s=12, c='0.75', alpha=0.5, marker='.',
                label=f'Candidates (n={len(pool)})', zorder=3,
                **plot_kw,
            )

        for cat, color in depth_colors.items():
            subset = [e for e in events if e.depth_category == cat]
            if not subset:
                continue
            sizes = [50 + 20 * (e.best_magnitude - 5.5) for e in subset]
            ax.scatter(
                [e.longitude for e in subset],
                [e.latitude for e in subset],
                s=sizes, c=color, alpha=0.9, edgecolors='k', linewidths=0.5,
                label=f'Selected {cat} (n={len(subset)})', zorder=5,
                **plot_kw,
            )

        ax.set_title('EastAsia_model forward test: 30 selected events')
        if not use_cartopy:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        ax.legend(loc='lower left', frameon=True, fontsize=8)

        fig.tight_layout()
        fig_dir = self.base_config.dirs['figures']
        fig_dir.mkdir(parents=True, exist_ok=True)
        stem = '1-7_eastasia_model_30events_distribution'
        saved = None
        for ext, dpi in (('jpg', 300), ('pdf', 300)):
            out = fig_dir / f'{stem}.{ext}'
            fig.savefig(out, dpi=dpi, bbox_inches='tight')
            local = self.output_dir / f'{stem}.{ext}'
            fig.savefig(local, dpi=dpi, bbox_inches='tight')
            saved = out
        plt.close(fig)
        self.logger.info(f"✅ 分布图已保存: {saved}")
        return saved

    def _load_eastasia_stations_catalog(
        self,
        stations_file: Optional[Path] = None,
    ) -> pd.DataFrame:
        """加载 EastAsia_model 台站目录并统一列名。"""
        if stations_file is not None:
            sta_path = Path(stations_file)
        else:
            cfg_ea = self.config.eastasia_model_forward
            sta_path = self._eastasia_dataset_dir / cfg_ea['stations_catalog']

        if not sta_path.exists():
            raise FileNotFoundError(f"台站目录不存在: {sta_path}")

        df = pd.read_csv(sta_path)
        rename = {
            'Latitude': 'latitude', 'Longitude': 'longitude',
            'Station': 'station', 'Network': 'network',
            'Provider': 'provider', 'Elevation': 'elevation',
        }
        for old, new in rename.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        return df

    @staticmethod
    def _build_network_color_map(
        networks: pd.Series,
        top_n: int = 14,
    ) -> Tuple[Dict[str, str], List[str]]:
        """为台网生成 GMT RGB 配色，小台网合并为 Other。"""
        import matplotlib

        counts = networks.value_counts()
        top_nets = counts.head(top_n).index.tolist()
        cmap = matplotlib.colormaps['tab20']
        tab20b = matplotlib.colormaps['tab20b']
        palettes = [cmap(i) for i in range(20)] + [tab20b(i) for i in range(20)]

        color_map: Dict[str, str] = {}
        for i, net in enumerate(top_nets):
            rgb = palettes[i % len(palettes)]
            color_map[str(net)] = (
                f"{int(rgb[0] * 255)}/{int(rgb[1] * 255)}/{int(rgb[2] * 255)}"
            )
        color_map['Other'] = '150/150/150'
        return color_map, top_nets

    def _save_paper_figure(
        self,
        fig: Any,
        stem: str,
        cfg: Dict[str, Any],
    ) -> Path:
        """保存论文图到 figures/ 与 output_dir。"""
        fig_dir = self.base_config.dirs['figures']
        fig_dir.mkdir(parents=True, exist_ok=True)
        dpi = int(cfg.get('dpi', 300))
        saved = fig_dir / f'{stem}.jpg'
        for fmt in cfg.get('save_formats', ['jpg', 'pdf']):
            out = fig_dir / f'{stem}.{fmt}'
            fig.savefig(str(out), dpi=dpi)
            local = self.output_dir / f'{stem}.{fmt}'
            fig.savefig(str(local), dpi=dpi)
            self.logger.info(f"  ✅ {out}")
        return saved

    def plot_eastasia_model_paper_globe(
        self,
        events_csv: Optional[Path] = None,
        stations_file: Optional[Path] = None,
    ) -> Path:
        """
        绘制论文用单幅地球透视图：台站小三角 + 事件震源机制解叠加。

        Args:
            events_csv: 事件 CSV，默认 output_dir/test_database_events.csv
            stations_file: 台站文件，默认 EastAsia_model catalog/stations_catalog.csv

        Returns:
            主输出 jpg 路径
        """
        import pygmt

        events_path = Path(events_csv) if events_csv else (
            self.output_dir / 'test_database_events.csv'
        )
        if not events_path.exists():
            raise FileNotFoundError(f"事件文件不存在: {events_path}")

        events_df = pd.read_csv(events_path)
        if events_df.empty:
            raise ValueError("事件 CSV 为空")

        stations_df = self._load_eastasia_stations_catalog(stations_file)

        region = self.base_config.region
        center_lon = (region['lon_min'] + region['lon_max']) / 2.0
        center_lat = (region['lat_min'] + region['lat_max']) / 2.0
        bnd_lon = [
            region['lon_min'], region['lon_max'],
            region['lon_max'], region['lon_min'], region['lon_min'],
        ]
        bnd_lat = [
            region['lat_min'], region['lat_min'],
            region['lat_max'], region['lat_max'], region['lat_min'],
        ]

        gcfg = self.config.eastasia_paper_globe
        fig_dir = self.base_config.dirs['figures']
        fig_dir.mkdir(parents=True, exist_ok=True)
        tmp_cpt = fig_dir / '_tmp_1-7_globe_depth.cpt'
        tmp_legend = fig_dir / '_tmp_1-7_globe_legend.txt'

        depth0, depth1 = gcfg['depth_range']
        pygmt.config(
            FONT_ANNOT_PRIMARY='7.5p,Helvetica',
            FONT_LABEL='8.5p,Helvetica',
            MAP_FRAME_TYPE='plain',
            MAP_FRAME_PEN='0.6p,50/50/50',
            MAP_GRID_PEN_PRIMARY='0.12p,210/210/210,-',
        )
        pygmt.makecpt(
            cmap=gcfg['depth_cmap'],
            series=f'{depth0}/{depth1}',
            output=str(tmp_cpt),
        )

        projection = (
            f"G{center_lon}/{center_lat}/{gcfg['altitude']}/"
            f"0/0/0/0/0/{gcfg['globe_width']}"
        )
        grid = f"g{int(gcfg['grid_interval'])}"

        fig = pygmt.Figure()
        fig.coast(
            region='g',
            projection=projection,
            frame=grid,
            land='#CFCFCF',
            water='white',
            borders='1/0.15p,180/180/180',
            resolution='i',
        )
        fig.plot(
            x=bnd_lon, y=bnd_lat,
            pen=gcfg['study_region_pen'],
            projection=projection,
        )

        # 台站（底层，小三角）
        fig.plot(
            x=stations_df['longitude'].astype(float).values,
            y=stations_df['latitude'].astype(float).values,
            style=gcfg['station_size'],
            fill=gcfg['station_fill'],
            pen=gcfg['station_pen'],
            projection=projection,
        )

        # 事件震源球（顶层）
        mt_cols = ['mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp']
        has_mt = all(c in events_df.columns for c in mt_cols)
        n_meca = 0
        for _, ev in events_df.iterrows():
            lon = float(ev['longitude'])
            lat = float(ev['latitude'])
            dep = float(ev['depth'])
            mag = float(ev['magnitude_mw'])
            scale = 0.16 + (mag - 5.5) * 0.10
            scale = max(0.14, min(0.34, scale))
            try:
                if not has_mt:
                    raise ValueError('no mt')
                comps = [float(ev[c]) for c in mt_cols]
                max_abs = max(abs(v) for v in comps) or 1.0
                exp = int(math.floor(math.log10(max_abs)))
                sf = 10.0 ** exp
                fig.meca(
                    spec={
                        'mrr': comps[0] / sf, 'mtt': comps[1] / sf,
                        'mff': comps[2] / sf, 'mrt': comps[3] / sf,
                        'mrf': comps[4] / sf, 'mtf': comps[5] / sf,
                        'exponent': exp,
                    },
                    convention='mt',
                    scale=f'{scale:.2f}c',
                    longitude=lon,
                    latitude=lat,
                    depth=dep,
                    cmap=str(tmp_cpt),
                    pen=gcfg['meca_pen'],
                )
                n_meca += 1
            except Exception:
                fig.plot(
                    x=lon, y=lat,
                    style=f'c{scale:.2f}c',
                    fill=dep,
                    cmap=str(tmp_cpt),
                    pen='0.25p,30/30/30',
                    projection=projection,
                )

        n_sh = int((events_df['depth_category'] == 'shallow').sum()) if 'depth_category' in events_df.columns else 0
        n_in = int((events_df['depth_category'] == 'intermediate').sum()) if 'depth_category' in events_df.columns else 0
        n_dp = int((events_df['depth_category'] == 'deep').sum()) if 'depth_category' in events_df.columns else 0
        mw_min = float(events_df['magnitude_mw'].min())
        mw_max = float(events_df['magnitude_mw'].max())

        fig.colorbar(
            cmap=str(tmp_cpt),
            position='JBC+o0c/0.55c+w10c/0.24c+h',
            frame=['xa100f50+lDepth (km)', 'y'],
        )

        legend_txt = '\n'.join([
            'N 1',
            f'S 0.30c t 0.08c {gcfg["station_fill"]} {gcfg["station_pen"]} 0.55c '
            f'Stations (N={len(stations_df)})',
            'G 0.06c',
            'S 0.30c c 0.18c 200/50/50 0.20p,30/30/30 0.55c '
            f'Events (N={len(events_df)})',
            'G 0.06c',
            'L 7p,Helvetica L Focal mechanisms colored by depth',
            f'L 7p,Helvetica L Mw {mw_min:.1f}-{mw_max:.1f}; '
            f'shallow {n_sh}, inter {n_in}, deep {n_dp}',
        ]) + '\n'
        tmp_legend.write_text(legend_txt, encoding='ascii')
        fig.legend(
            spec=str(tmp_legend),
            position='JTR+jTR+o0.25c/0.25c',
            box='+gwhite@15+p0.35p,160/160/160',
        )

        saved = self._save_paper_figure(fig, gcfg['figure_stem'], gcfg)
        tmp_cpt.unlink(missing_ok=True)
        tmp_legend.unlink(missing_ok=True)
        self.logger.info(
            f"📄 论文地球透视图: {n_meca}/{len(events_df)} 震源球 + "
            f"{len(stations_df)} 台站（单图叠加）"
        )
        return saved

    def plot_eastasia_model_station_network_detail(
        self,
        stations_file: Optional[Path] = None,
    ) -> Path:
        """
        绘制 EastAsia_model 台站网络详图（单独成图）。

        按台网分色、展示主要台网统计与 Provider 信息，供论文补充说明台站覆盖。

        Args:
            stations_file: 台站 CSV，默认 EastAsia_model catalog/stations_catalog.csv

        Returns:
            主输出 jpg 路径
        """
        import pygmt

        stations_df = self._load_eastasia_stations_catalog(stations_file)
        scfg = self.config.eastasia_station_detail
        region = list(scfg['region'])
        study = self.base_config.region
        bnd_lon = [
            study['lon_min'], study['lon_max'],
            study['lon_max'], study['lon_min'], study['lon_min'],
        ]
        bnd_lat = [
            study['lat_min'], study['lat_min'],
            study['lat_max'], study['lat_max'], study['lat_min'],
        ]

        top_n = int(scfg['top_networks'])
        color_map, top_nets = self._build_network_color_map(
            stations_df['network'], top_n=top_n,
        )
        stations_df = stations_df.copy()
        stations_df['net_group'] = stations_df['network'].astype(str).where(
            stations_df['network'].astype(str).isin(top_nets), 'Other'
        )
        stations_df['color'] = stations_df['net_group'].map(color_map)

        fig_dir = self.base_config.dirs['figures']
        fig_dir.mkdir(parents=True, exist_ok=True)
        tmp_legend = fig_dir / '_tmp_1-7_station_legend.txt'

        pygmt.config(
            FONT_ANNOT_PRIMARY='8p,Helvetica',
            FONT_LABEL='9p,Helvetica',
            MAP_FRAME_TYPE='plain',
            MAP_FRAME_PEN='0.7p,50/50/50',
            MAP_GRID_PEN_PRIMARY='0.15p,220/220/220,-',
        )

        fig = pygmt.Figure()
        fig.basemap(
            region=region,
            projection=f"M{scfg['projection_width']}",
            frame=scfg['frame'],
        )
        fig.coast(
            land='#F5F2EA',
            water='#D8E6F0',
            shorelines='0.35p,100/100/100',
            borders='1/0.2p,170/170/170',
            resolution='i',
        )

        boundaries = (
            self.base_config.dirs['project_root'] /
            '5_Visualization' / 'data_EastAsia' / 'tectonics' / 'boundaries.gmt'
        )
        if boundaries.exists():
            fig.plot(data=str(boundaries), pen='0.5p,90/90/90')

        fig.plot(x=bnd_lon, y=bnd_lat, pen='2p,black')

        for color, group in stations_df.groupby('color', sort=False):
            fig.plot(
                x=group['longitude'].astype(float).values,
                y=group['latitude'].astype(float).values,
                style=scfg['station_size'],
                fill=color,
                pen='0.05p,50/50/50',
            )

        net_counts = stations_df['network'].value_counts()
        n_networks = stations_df['network'].nunique()
        n_providers = stations_df['provider'].nunique() if 'provider' in stations_df.columns else 0
        provider_top = ''
        if 'provider' in stations_df.columns:
            prov_counts = stations_df['provider'].value_counts()
            provider_top = ', '.join(
                f"{p}({c})" for p, c in prov_counts.head(4).items()
            )

        legend_lines = ['N 1', 'G 0.04c']
        for net in top_nets:
            cnt = int(net_counts.get(net, 0))
            color = color_map[str(net)]
            legend_lines.append(
                f'S 0.26c t 0.07c {color} 0.06p,50/50/50 0.62c {net} ({cnt})'
            )
        n_other = int((stations_df['net_group'] == 'Other').sum())
        if n_other > 0:
            legend_lines.append(
                f'S 0.26c t 0.07c {color_map["Other"]} 0.06p,50/50/50 '
                f'0.62c Other ({n_other})'
            )
        legend_lines.extend([
            'G 0.08c',
            f'L 7.5p,Helvetica L Total: {len(stations_df)} stations, '
            f'{n_networks} networks',
        ])
        if provider_top:
            legend_lines.append(
                f'L 7.5p,Helvetica L Providers: {provider_top}'
            )
        legend_lines.append(
            'L 7.5p,Helvetica L Black box: EASTASIA-FWI study region'
        )

        tmp_legend.write_text('\n'.join(legend_lines) + '\n', encoding='ascii')
        fig.legend(
            spec=str(tmp_legend),
            position='JBR+jBR+o0.3c/0.3c',
            box='+gwhite@12+p0.4p,150/150/150',
        )
        fig.basemap(map_scale='jBL+w1000k+f+lkm+o0.5c/0.5c')

        saved = self._save_paper_figure(fig, scfg['figure_stem'], scfg)
        tmp_legend.unlink(missing_ok=True)
        self.logger.info(
            f"📡 台站网络详图: {len(stations_df)} 台站, "
            f"{n_networks} 台网, top {top_n} 分色"
        )
        return saved

    def save_hybrid_download_catalog(
        self,
        events: Optional[List[CMT3DEvent]] = None,
    ) -> Path:
        """
        写出与 hybrid_10events 相同格式的 .par（震级类型为 Mw），
        同时放到输出子目录和 data/events 根目录。
        """
        if events is None:
            events = self.selected_events

        cfg = self.config.hybrid_forward
        lines = []
        for ev in events:
            dt = ev.origin_datetime
            if dt is None:
                continue
            mag = ev.best_magnitude
            event_dir = self._event_dir_name(ev)
            ymd = dt.strftime('%Y%m%d')
            hh = dt.strftime('%H')
            mm = dt.strftime('%M')
            sec = f"{ev.pde_second:.3f}"
            line = (
                f"{event_dir} {ymd} {hh} {mm} {sec} "
                f"{ev.latitude:.4f} {ev.longitude:.4f} {ev.depth:.1f} "
                f"8.4 25.6 {float(mag):.1f} Mw"
            )
            lines.append(line)

        text = '\n'.join(lines) + '\n'
        local_par = self.output_dir / 'download_catalog.par'
        root_par = self.base_config.dirs['events'] / cfg['root_par_name']
        for path in (local_par, root_par):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
        self.logger.info(
            f"✅ hybrid_30 下载目录: {root_par} ({len(lines)} 个事件)"
        )
        return root_par

    def save_id_mapping_csv(
        self,
        events: Optional[List[CMT3DEvent]] = None,
    ) -> Path:
        """写出 event_name ↔ 目录名对照表，格式对齐 hybrid_10events_id_to_dir.csv。"""
        if events is None:
            events = self.selected_events

        records = []
        for ev in events:
            dt = ev.origin_datetime
            origin_iso = dt.isoformat() if dt is not None else ''
            records.append({
                'event': ev.event_name,
                'dir': self._event_dir_name(ev),
                'origin_pde': origin_iso,
                'lat': round(ev.latitude, 4),
                'lon': round(ev.longitude, 4),
                'depth_km': round(ev.depth, 2),
                'Mw': round(ev.best_magnitude, 2),
                'depth_category': ev.depth_category,
            })
        df = pd.DataFrame(records)
        cfg = self.config.hybrid_forward
        local_csv = self.output_dir / cfg['id_map_name']
        root_csv = self.base_config.dirs['events'] / cfg['id_map_name']
        for path in (local_csv, root_csv):
            df.to_csv(path, index=False, encoding='utf-8')

        list_path = self.output_dir / cfg['events_list_name']
        with open(list_path, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(rec['dir'] + '\n')

        self.logger.info(f"✅ 事件对照表: {root_csv}")
        return root_csv

    def plot_hybrid_selection_map(
        self,
        events: Optional[List[CMT3DEvent]] = None,
    ) -> Optional[Path]:
        """绘制旧 10 事件与新 30 事件对比分布图（英文标签）。"""
        if events is None:
            events = self.selected_events
        if not events:
            return None

        import matplotlib.pyplot as plt

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            use_cartopy = True
        except ImportError:
            use_cartopy = False

        excluded = self.load_excluded_hybrid_events()
        cfg = self.config.hybrid_forward
        lon_min, lon_max = cfg['lon_range']
        lat_min, lat_max = cfg['lat_range']
        depth_colors = {
            'shallow': '#d62728',
            'intermediate': '#ff7f0e',
            'deep': '#1f77b4',
        }

        fig_w, fig_h = 12, 8
        if use_cartopy:
            proj = ccrs.PlateCarree()
            fig, ax = plt.subplots(
                figsize=(fig_w, fig_h), subplot_kw={'projection': proj}
            )
            ax.set_extent(
                [lon_min, lon_max, lat_min, lat_max],
                crs=proj,
            )
            ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor='#dbe9f4', zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':', zorder=1)
            ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
            scatter_crs = proj
        else:
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, linewidth=0.3, alpha=0.5)
            scatter_crs = None

        plot_kw = {'transform': scatter_crs} if use_cartopy else {}

        if excluded:
            ax.scatter(
                [r['lon'] for r in excluded],
                [r['lat'] for r in excluded],
                marker='s', s=70, facecolors='none', edgecolors='0.3',
                linewidths=1.2, label='Previous 10 events', zorder=4,
                **plot_kw,
            )

        for cat, color in depth_colors.items():
            subset = [e for e in events if e.depth_category == cat]
            if not subset:
                continue
            sizes = [40 + 18 * (e.best_magnitude - 5.5) for e in subset]
            ax.scatter(
                [e.longitude for e in subset],
                [e.latitude for e in subset],
                s=sizes, c=color, alpha=0.85, edgecolors='k', linewidths=0.4,
                label=f'New {cat} (n={len(subset)})', zorder=5,
                **plot_kw,
            )

        ax.set_title('Hybrid forward events: 30 new vs 10 previous')
        if not use_cartopy:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        ax.legend(loc='lower left', frameon=True, fontsize=9)

        fig.tight_layout()
        fig_dir = self.base_config.dirs['figures']
        fig_dir.mkdir(parents=True, exist_ok=True)
        stem = '1-7_hybrid_30events_distribution'
        saved = None
        for ext, dpi in (('jpg', 300), ('pdf', 300)):
            out = fig_dir / f'{stem}.{ext}'
            fig.savefig(out, dpi=dpi, bbox_inches='tight')
            local = self.output_dir / f'{stem}.{ext}'
            fig.savefig(local, dpi=dpi, bbox_inches='tight')
            saved = out
        plt.close(fig)
        self.logger.info(f"✅ 分布图已保存: {saved}")
        return saved

    def _load_paper_stations(
        self,
        stations_file: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        加载论文图台站。优先 STATIONS_quality（1_8 按实测波形筛选）。

        Args:
            stations_file: 显式路径；为空则用配置中的质量台站文件

        Returns:
            含 station/network/latitude/longitude 的 DataFrame
        """
        if stations_file is not None:
            sta_path = Path(stations_file)
        else:
            sta_path = (
                self.base_config.dirs['stations'] /
                self.config.paper_map.get('stations_name', 'STATIONS_quality')
            )
            if not sta_path.exists():
                sta_path = self.output_dir / 'STATIONS'

        if not sta_path.exists():
            raise FileNotFoundError(f"台站文件不存在: {sta_path}")

        first = ''
        with open(sta_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    first = line.strip()
                    break
        parts = first.split()
        is_specfem = (
            sta_path.suffix.lower() != '.csv'
            and len(parts) >= 4
            and not first.lower().startswith(('network', 'station'))
        )
        if is_specfem:
            df = pd.read_csv(
                sta_path,
                sep=r'\s+',
                header=None,
                engine='python',
                names=['station', 'network', 'latitude', 'longitude',
                       'elevation', 'burial'],
            )
        else:
            raw = pd.read_csv(sta_path, encoding='utf-8')
            df = raw.rename(columns={
                'Latitude': 'latitude', 'Longitude': 'longitude',
                'Station': 'station', 'Network': 'network',
            })
        if df.empty:
            raise ValueError(f"台站文件为空: {sta_path}")
        self.logger.info(f"  台站文件: {sta_path} ({len(df)} 台)")
        return df

    def plot_paper_events_stations(
        self,
        events_csv: Optional[Path] = None,
        stations_file: Optional[Path] = None,
    ) -> Path:
        """
        绘制论文用单图：筛选的 30 个 CMT3D 事件 + 实测波形质量台站。

        默认使用 1_8 从下载会话筛选的 STATIONS_quality，而不是永久台清单。
        台站为倒三角，事件为按深度着色的震源机制解（GMT seis，0–700 km），
        震源球大小随 Mw 缩放。无整图大标题。输出 jpg + pdf。

        Args:
            events_csv: 事件 CSV，默认 output_dir/test_database_events.csv
            stations_file: SPECFEM STATIONS 或台站 CSV；
                默认 data/stations/STATIONS_quality

        Returns:
            主输出 jpg 路径
        """
        import pygmt

        events_path = Path(events_csv) if events_csv else (
            self.output_dir / 'test_database_events.csv'
        )
        if not events_path.exists():
            raise FileNotFoundError(f"事件文件不存在: {events_path}")

        events_df = pd.read_csv(events_path)
        if len(events_df) == 0:
            raise ValueError("事件 CSV 为空")

        stations_df = self._load_paper_stations(stations_file)

        n_ev = len(events_df)
        n_sta = len(stations_df)
        self.logger.info(f"📄 论文图: {n_ev} 个事件 + {n_sta} 个质量台站")

        cfg = self.config.paper_map
        region = list(cfg['region'])

        fig_dir = self.base_config.dirs['figures']
        fig_dir.mkdir(parents=True, exist_ok=True)
        tmp_cpt = fig_dir / '_tmp_1-7_depth.cpt'
        tmp_legend = fig_dir / '_tmp_1-7_legend.txt'

        pygmt.config(
            FONT_ANNOT_PRIMARY='8p,Helvetica',
            FONT_LABEL='9p,Helvetica',
            FONT_TITLE='10p,Helvetica-Bold',
            MAP_FRAME_TYPE='plain',
            MAP_FRAME_PEN='0.8p,black',
            MAP_TICK_LENGTH_PRIMARY='3.5p',
            MAP_TICK_PEN_PRIMARY='0.6p,black',
            MAP_GRID_PEN_PRIMARY='0.15p,210/210/210,-',
        )
        pygmt.makecpt(cmap='seis', series='0/700', output=str(tmp_cpt))

        fig = pygmt.Figure()
        fig.basemap(
            region=region,
            projection=f"M{cfg['projection_width']}",
            frame=cfg['frame'],
        )
        fig.coast(
            land='#F7F4ED',
            water='#DCE8F2',
            shorelines='0.32p,90/90/90',
            borders='1/0.2p,170/170/170',
            resolution='f',
        )

        boundaries = (
            self.base_config.dirs['project_root'] /
            '5_Visualization' / 'data_EastAsia' / 'tectonics' / 'boundaries.gmt'
        )
        if boundaries.exists():
            fig.plot(data=str(boundaries), pen='0.6p,70/70/70')

        fig.plot(
            x=stations_df['longitude'].values,
            y=stations_df['latitude'].values,
            style=cfg['station_size'],
            fill=cfg['station_fill'],
            pen=cfg['station_pen'],
        )

        mt_cols = ['mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp']
        has_mt = all(c in events_df.columns for c in mt_cols)
        n_meca = 0
        for _, ev in events_df.iterrows():
            lon = float(ev['longitude'])
            lat = float(ev['latitude'])
            dep = float(ev['depth'])
            mag = float(ev['magnitude_mw'])
            scale = 0.20 + (mag - 5.5) * 0.12
            scale = max(0.18, min(0.38, scale))
            try:
                if not has_mt:
                    raise ValueError('no mt')
                comps = [float(ev[c]) for c in mt_cols]
                max_abs = max(abs(v) for v in comps) or 1.0
                exp = int(math.floor(math.log10(max_abs)))
                sf = 10.0 ** exp
                fig.meca(
                    spec={
                        'mrr': comps[0] / sf, 'mtt': comps[1] / sf,
                        'mff': comps[2] / sf, 'mrt': comps[3] / sf,
                        'mrf': comps[4] / sf, 'mtf': comps[5] / sf,
                        'exponent': exp,
                    },
                    convention='mt',
                    scale=f'{scale:.2f}c',
                    longitude=lon,
                    latitude=lat,
                    depth=dep,
                    cmap=str(tmp_cpt),
                    pen='0.22p,25/25/25',
                )
                n_meca += 1
            except Exception:
                fig.plot(
                    x=lon, y=lat,
                    style=f'c{scale:.2f}c',
                    fill='orange',
                    pen='0.3p,black',
                )

        self.logger.info(f"  震源球 {n_meca}/{n_ev}")

        fig.colorbar(
            cmap=str(tmp_cpt),
            position='JBC+o0c/0.62c+w9.2c/0.26c+h',
            frame=['xa100f50+lDepth (km)', 'y'],
        )

        n_sh = int((events_df['depth_category'] == 'shallow').sum()) if 'depth_category' in events_df.columns else 0
        n_in = int((events_df['depth_category'] == 'intermediate').sum()) if 'depth_category' in events_df.columns else 0
        n_dp = int((events_df['depth_category'] == 'deep').sum()) if 'depth_category' in events_df.columns else 0
        mw_min = float(events_df['magnitude_mw'].min())
        mw_max = float(events_df['magnitude_mw'].max())

        # GMT legend 仅用 ASCII，避免 en-dash 在 PostScript 中乱码
        legend_txt = '\n'.join([
            'N 1',
            f'S 0.30c i 0.15c 255/255/255 0.45p,20/55/105 0.50c Stations (n={n_sta})',
            'G 0.10c',
            f'S 0.30c c 0.20c 200/20/20 0.22p,25/25/25 0.50c Events (n={n_ev})',
            'G 0.03c',
            'L 7.5p,Helvetica L CMT3D beachballs by depth',
            f'L 7.5p,Helvetica L Size scales with Mw ({mw_min:.1f}-{mw_max:.1f})',
            'G 0.08c',
            f'S 0.30c c 0.15c 200/0/0 0.2p,40/40/40 0.50c Shallow 0-70 km ({n_sh})',
            f'S 0.30c c 0.15c 250/200/20 0.2p,40/40/40 0.50c Intermediate 70-300 km ({n_in})',
            f'S 0.30c c 0.15c 30/90/180 0.2p,40/40/40 0.50c Deep 300-700 km ({n_dp})',
        ]) + '\n'
        tmp_legend.write_text(legend_txt, encoding='ascii')
        fig.legend(
            spec=str(tmp_legend),
            position='JTR+jTR+o0.22c/0.22c',
            box='+gwhite@12+p0.4p,130/130/130',
        )

        fig.basemap(map_scale='jBL+w1000k+f+lkm+o0.45c/0.45c')

        stem = self.config.paper_map['figure_stem']
        dpi = int(self.config.paper_map['dpi'])
        saved = fig_dir / f'{stem}.jpg'
        for fmt in self.config.paper_map['save_formats']:
            out = fig_dir / f'{stem}.{fmt}'
            fig.savefig(str(out), dpi=dpi)
            local = self.output_dir / f'{stem}.{fmt}'
            fig.savefig(str(local), dpi=dpi)
            self.logger.info(f"  ✅ {out}")

        tmp_cpt.unlink(missing_ok=True)
        tmp_legend.unlink(missing_ok=True)
        return saved

    # ----------------------------------------------------------------
    # 永久台站
    # ----------------------------------------------------------------

    def load_permanent_stations(self) -> pd.DataFrame:
        """
        加载永久台站数据

        Returns:
            台站 DataFrame
        """
        station_file = (
            self.base_config.dirs['stations'] /
            self.config.stations['station_file']
        )
        if not station_file.exists():
            raise FileNotFoundError(f"永久台站文件不存在: {station_file}")

        self.logger.info(f"📡 加载永久台站: {station_file}")

        df = pd.read_csv(station_file, encoding='utf-8')
        self.stations_df = df

        self.logger.info(f"✅ 加载 {len(df)} 个永久台站")

        # 统计台网分布
        if 'Network' in df.columns:
            network_counts = df['Network'].value_counts()
            self.logger.info("📊 台网分布:")
            for net, count in network_counts.head(15).items():
                self.logger.info(f"  {net}: {count} 台站")

        return df

    def generate_specfem_stations_file(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> Path:
        """
        生成 SPECFEM3D Globe STATIONS 文件

        格式: STATION NETWORK LATITUDE LONGITUDE ELEVATION BURIAL_DEPTH

        Args:
            df: 台站 DataFrame，默认使用 self.stations_df

        Returns:
            输出文件路径
        """
        if df is None:
            df = self.stations_df
        if df is None:
            raise ValueError("没有台站数据，请先调用 load_permanent_stations()")

        output_file = self.output_dir / 'STATIONS'

        lines = []
        for _, row in df.iterrows():
            station = str(row.get('Station', '')).strip()
            network = str(row.get('Network', '')).strip()
            lat = float(row.get('Latitude', 0.0))
            lon = float(row.get('Longitude', 0.0))
            elev = float(row.get('Elevation', 0.0))
            burial = 0.0  # 地表台站

            lines.append(
                f"{station:<8s} {network:<6s} {lat:12.4f} {lon:12.4f} {elev:8.1f} {burial:6.1f}"
            )

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

        self.logger.info(f"✅ STATIONS 文件已生成: {output_file} ({len(lines)} 台站)")
        return output_file

    # ----------------------------------------------------------------
    # 输出模块
    # ----------------------------------------------------------------

    def save_cmtsolution_files(
        self,
        events: Optional[List[CMT3DEvent]] = None
    ) -> Path:
        """
        保存每个事件的 CMTSOLUTION 文件

        Args:
            events: 事件列表，默认使用 self.selected_events

        Returns:
            CMTSOLUTION 文件目录路径
        """
        if events is None:
            events = self.selected_events

        cmt_dir = self.output_dir / 'CMTSOLUTION'
        cmt_dir.mkdir(parents=True, exist_ok=True)

        for ev in events:
            cmt_file = cmt_dir / f"CMTSOLUTION_{ev.event_name}"
            with open(cmt_file, 'w', encoding='utf-8') as f:
                f.write(ev.to_cmtsolution())

        self.logger.info(f"✅ CMTSOLUTION 文件已生成: {cmt_dir} ({len(events)} 个文件)")
        return cmt_dir

    def save_catalog_csv(
        self,
        events: Optional[List[CMT3DEvent]] = None
    ) -> Path:
        """
        保存事件目录 CSV

        Args:
            events: 事件列表，默认使用 self.selected_events

        Returns:
            CSV 文件路径
        """
        if events is None:
            events = self.selected_events

        records = [ev.to_dict() for ev in events]
        df = pd.DataFrame(records)

        csv_file = self.output_dir / 'test_database_events.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8')

        self.logger.info(f"✅ 事件目录 CSV 已保存: {csv_file}")
        return csv_file

    def save_download_list(
        self,
        events: Optional[List[CMT3DEvent]] = None
    ) -> Path:
        """
        生成波形下载任务清单（衔接 1_3_Download_waveforms.py）

        每行与 `data/events/download_catalog_1113_test.par` 及
        `WaveformDownloader.generate_download_catalog` 一致，字段顺序为::

            事件目录名 YYYYMMDD HH MM SS.sss lat lon depth 8.4 25.6 Mag ML

        其中事件目录名为 ``%Y%m%d.%H.%M``（与 MassDownloader 输出子目录命名一致）。

        Args:
            events: 事件列表，默认使用 self.selected_events

        Returns:
            下载清单文件路径
        """
        if events is None:
            events = self.selected_events

        download_file = self.output_dir / 'download_catalog.par'

        lines = []
        for ev in events:
            dt = ev.origin_datetime
            if dt is None:
                continue
            mag = ev.best_magnitude
            if mag is None or (isinstance(mag, float) and math.isnan(mag)) or mag <= 0:
                self.logger.warning(f"事件 {ev.event_name} 无有效震级，跳过写入下载清单")
                continue
            # 与 1_3 generate_download_catalog 对齐；秒使用 PDE 分数秒
            event_dir = dt.strftime('%Y%m%d.%H.%M')
            ymd = dt.strftime('%Y%m%d')
            hh = dt.strftime('%H')
            mm = dt.strftime('%M')
            sec = f"{ev.pde_second:.3f}"
            line = (
                f"{event_dir} {ymd} {hh} {mm} {sec} "
                f"{ev.latitude:.4f} {ev.longitude:.4f} {ev.depth:.1f} "
                f"8.4 25.6 {float(mag):.1f} ML"
            )
            lines.append(line)

        with open(download_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

        self.logger.info(f"✅ 下载任务清单已生成: {download_file} ({len(lines)} 个事件)")
        return download_file

    def save_metadata(
        self,
        events: Optional[List[CMT3DEvent]] = None
    ) -> Path:
        """
        保存构建元数据 JSON

        Args:
            events: 事件列表，默认使用 self.selected_events

        Returns:
            元数据文件路径
        """
        if events is None:
            events = self.selected_events

        stations_count = len(self.stations_df) if self.stations_df is not None else 0

        depths = [ev.depth for ev in events]
        mags = [ev.best_magnitude for ev in events]

        metadata = {
            'project': 'EASTASIA-FWI',
            'module': '1_7_Build_test_database',
            'version': 'v1.0',
            'created': datetime.now().isoformat(),
            'source_catalog': {
                'name': 'CMT3D',
                'reference': self.config.cmt3d['source'],
                'reference_model': self.config.cmt3d['reference_model'],
                'total_global_events': len(self.all_events),
            },
            'region': dict(self.base_config.region),
            'filters': {
                'magnitude_range': self.config.event_filters['magnitude_range'],
                'depth_range': self.config.event_filters['depth_range'],
                'time_range': self.config.event_filters['time_range'],
            },
            'sampling': {
                'target_count': self.config.sampling['target_count'],
                'min_event_distance_deg': self.config.sampling['min_event_distance_deg'],
                'depth_quotas': self.config.sampling['depth_quotas'],
                'random_seed': self.config.sampling['random_seed'],
            },
            'result': {
                'filtered_events': len(self.filtered_events),
                'selected_events': len(events),
                'permanent_stations': stations_count,
                'depth_distribution': {
                    'shallow_0_70km': sum(1 for e in events if e.depth_category == 'shallow'),
                    'intermediate_70_300km': sum(1 for e in events if e.depth_category == 'intermediate'),
                    'deep_300_700km': sum(1 for e in events if e.depth_category == 'deep'),
                },
                'magnitude_range': [round(min(mags), 2), round(max(mags), 2)] if mags else [],
                'depth_range': [round(min(depths), 1), round(max(depths), 1)] if depths else [],
                'time_range': [
                    min(e.pde_year for e in events),
                    max(e.pde_year for e in events),
                ] if events else [],
            },
            'quality_control': dict(self.config.quality_control),
            'output_dir': str(self.output_dir),
        }

        meta_file = self.output_dir / 'test_database_metadata.json'
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"✅ 元数据已保存: {meta_file}")
        return meta_file

    # ----------------------------------------------------------------
    # 统计报告
    # ----------------------------------------------------------------

    def generate_report(
        self,
        events: Optional[List[CMT3DEvent]] = None
    ) -> Path:
        """
        生成详细的构建报告

        Args:
            events: 事件列表，默认使用 self.selected_events

        Returns:
            报告文件路径
        """
        if events is None:
            events = self.selected_events

        report_file = self.output_dir / 'test_database_report.txt'
        stations_count = len(self.stations_df) if self.stations_df is not None else 0

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EASTASIA-FWI 标准测试数据库构建报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"震源目录: CMT3D ({self.config.cmt3d['source']})\n")
            f.write(f"参考模型: {self.config.cmt3d['reference_model']}\n\n")

            # 筛选统计
            f.write("📊 事件筛选统计\n")
            f.write("-" * 60 + "\n")
            f.write(f"CMT3D 全球事件总数: {len(self.all_events)}\n")
            f.write(f"区域筛选后: {len(self.filtered_events)}\n")
            f.write(f"最终选取: {len(events)}\n\n")

            # 深度分布
            f.write("📐 深度分布\n")
            f.write("-" * 60 + "\n")
            depth_cats = {'shallow': 0, 'intermediate': 0, 'deep': 0}
            for ev in events:
                depth_cats[ev.depth_category] += 1

            f.write(f"浅源 (0-70 km):   {depth_cats['shallow']} 个事件\n")
            f.write(f"中源 (70-300 km):  {depth_cats['intermediate']} 个事件\n")
            f.write(f"深源 (300-700 km): {depth_cats['deep']} 个事件\n\n")

            # 震级分布
            mags = [ev.best_magnitude for ev in events]
            if mags:
                f.write("📈 震级分布 (Mw)\n")
                f.write("-" * 60 + "\n")
                f.write(f"范围: {min(mags):.2f} ~ {max(mags):.2f}\n")
                f.write(f"平均: {np.mean(mags):.2f}\n")
                f.write(f"中位数: {np.median(mags):.2f}\n\n")

            # 时间分布
            years = [ev.pde_year for ev in events]
            if years:
                f.write("📅 时间分布\n")
                f.write("-" * 60 + "\n")
                f.write(f"范围: {min(years)} ~ {max(years)}\n\n")

            # 台站信息
            f.write("📡 永久台站\n")
            f.write("-" * 60 + "\n")
            f.write(f"台站总数: {stations_count}\n")
            if self.stations_df is not None and 'Network' in self.stations_df.columns:
                network_counts = self.stations_df['Network'].value_counts()
                for net, count in network_counts.head(15).items():
                    f.write(f"  {net}: {count} 台站\n")
            f.write("\n")

            # 质量控制标准
            qc = self.config.quality_control
            f.write("🔧 质量控制标准（参考 FWEA23）\n")
            f.write("-" * 60 + "\n")
            f.write(f"体波 SNR 阈值: >= {qc['snr_body_wave']}\n")
            f.write(f"面波 SNR 阈值: >= {qc['snr_surface_wave']}\n")
            f.write(f"最小周期: T > {qc['min_period']} s\n\n")

            # 事件列表
            f.write("📋 事件列表\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'No.':<5} {'Event':<12} {'Date':<12} {'Lat':>8} {'Lon':>9} "
                    f"{'Depth':>7} {'Mw':>5} {'Category':<14}\n")
            f.write("-" * 80 + "\n")

            for i, ev in enumerate(events, 1):
                date_str = f"{ev.pde_year:04d}-{ev.pde_month:02d}-{ev.pde_day:02d}"
                f.write(
                    f"{i:<5d} {ev.event_name:<12s} {date_str:<12s} "
                    f"{ev.latitude:8.3f} {ev.longitude:9.3f} "
                    f"{ev.depth:7.1f} {ev.best_magnitude:5.2f} {ev.depth_category:<14s}\n"
                )

            f.write("-" * 80 + "\n")

            # 输出文件
            f.write("\n📁 输出文件\n")
            f.write("-" * 60 + "\n")
            f.write(f"CMTSOLUTION 目录: {self.output_dir / 'CMTSOLUTION'}\n")
            f.write(f"STATIONS 文件: {self.output_dir / 'STATIONS'}\n")
            f.write(f"事件 CSV: {self.output_dir / 'test_database_events.csv'}\n")
            f.write(f"下载清单: {self.output_dir / 'download_catalog.par'}\n")
            f.write(f"元数据: {self.output_dir / 'test_database_metadata.json'}\n")

        self.logger.info(f"✅ 构建报告已保存: {report_file}")
        return report_file

    def print_summary(self, events: Optional[List[CMT3DEvent]] = None):
        """在终端打印选取结果摘要"""
        if events is None:
            events = self.selected_events

        if not events:
            print("⚠️  没有选取任何事件")
            return

        depths = [ev.depth for ev in events]
        mags = [ev.best_magnitude for ev in events]
        years = [ev.pde_year for ev in events]

        depth_cats = {'shallow': 0, 'intermediate': 0, 'deep': 0}
        for ev in events:
            depth_cats[ev.depth_category] += 1

        stations_count = len(self.stations_df) if self.stations_df is not None else 0

        print("\n" + "=" * 60)
        print("📊 标准测试数据库构建结果")
        print("=" * 60)
        print(f"选取事件数: {len(events)}")
        print(f"永久台站数: {stations_count}")
        print(f"时间范围: {min(years)} ~ {max(years)}")
        print(f"震级范围: Mw {min(mags):.2f} ~ {max(mags):.2f} (均值 {np.mean(mags):.2f})")
        print(f"深度范围: {min(depths):.1f} ~ {max(depths):.1f} km")
        print(f"深度分布:")
        print(f"  浅源 (0-70 km):    {depth_cats['shallow']:3d} ({depth_cats['shallow']/len(events)*100:.0f}%)")
        print(f"  中源 (70-300 km):  {depth_cats['intermediate']:3d} ({depth_cats['intermediate']/len(events)*100:.0f}%)")
        print(f"  深源 (300-700 km): {depth_cats['deep']:3d} ({depth_cats['deep']/len(events)*100:.0f}%)")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)

    # ----------------------------------------------------------------
    # 完整流程
    # ----------------------------------------------------------------

    def run(self, mode: str = 'fwea23') -> Dict[str, Any]:
        """
        执行完整的测试数据库构建流程

        Args:
            mode: 构建模式
                - 'fwea23': 从 FWEA23 原作者 141 事件中筛选，与 CMT3D 匹配（推荐）
                - 'cmt3d': 直接从 CMT3D 目录全量筛选+采样
                - 'hybrid_30': 2010+ CMT3D 空间均匀 30 事件，排除既有 hybrid_10events
                - 'eastasia_model_30': EastAsia_model 波形库 Mw≥5.5 内部 30 事件

        Returns:
            构建结果字典
        """
        if mode == 'fwea23':
            return self._run_fwea23_mode()
        if mode == 'hybrid_30':
            return self._run_hybrid_30_mode()
        if mode == 'eastasia_model_30':
            return self._run_eastasia_model_30_mode()
        return self._run_cmt3d_mode()

    def _run_fwea23_mode(self) -> Dict[str, Any]:
        """
        FWEA23 模式: 从原作者 141 事件筛选 → CMT3D 时间匹配 → 智能采样 → 输出

        流程:
        1. 解析 CMT3D 全量目录
        2. 加载 FWEA23 参考 141 事件
        3. FWEA23 × CMT3D 时间匹配 → 得到带矩张量的事件
        4. 按项目标准筛选（区域、震级、深度）
        5. 智能采样 → ~80 个事件
        6. 生成输出文件
        """
        self.logger.info("🚀 开始构建标准测试数据库 (FWEA23 模式)")
        start_time = datetime.now()

        # 1. 解析 CMT3D 全量目录
        print("\n[1/7] 解析 CMT3D 震源目录...")
        cmt3d_all = self.parse_cmt3d_catalog()

        # 2. 加载 FWEA23 参考事件
        print("[2/7] 加载 FWEA23 参考事件...")
        fwea23_df = self.load_fwea23_reference_events()

        # 3. 时间匹配
        print("[3/7] FWEA23 × CMT3D 时间匹配...")
        matched_events = self.match_fwea23_with_cmt3d(fwea23_df, cmt3d_all)
        self.all_events = matched_events

        # 4. 按项目标准筛选
        print("[4/7] 按项目标准筛选事件...")
        filtered = self.filter_events(matched_events)

        # 5. 智能采样
        print("[5/7] 智能事件采样...")
        if len(filtered) <= self.config.sampling['max_count']:
            self.selected_events = filtered
            self.logger.info(
                f"✅ 筛选后事件 ({len(filtered)}) 未超过上限 "
                f"({self.config.sampling['max_count']})，直接全部采用"
            )
        else:
            self.select_representative_events(filtered)

        # 6. 加载永久台站
        print("[6/7] 加载永久台站...")
        self.load_permanent_stations()

        # 7. 生成输出文件
        print("[7/7] 生成输出文件...")
        output_files = self._save_all_outputs()

        elapsed = (datetime.now() - start_time).total_seconds()
        self.print_summary()
        print(f"\n⏱️  总耗时: {elapsed:.1f} 秒")
        self.logger.info(f"🎉 标准测试数据库构建完成 (FWEA23 模式, 耗时 {elapsed:.1f} 秒)")

        return {
            'mode': 'fwea23',
            'fwea23_total': len(fwea23_df),
            'cmt3d_matched': len(matched_events),
            'filtered_events': len(self.filtered_events),
            'selected_events': len(self.selected_events),
            'stations': len(self.stations_df) if self.stations_df is not None else 0,
            'output_files': output_files,
            'elapsed_seconds': elapsed,
        }

    def _run_cmt3d_mode(self) -> Dict[str, Any]:
        """CMT3D 模式: 直接从全量目录筛选 + 采样"""
        self.logger.info("🚀 开始构建标准测试数据库 (CMT3D 模式)")
        start_time = datetime.now()

        print("\n[1/6] 解析 CMT3D 震源目录...")
        events = self.parse_cmt3d_catalog()

        print("[2/6] 筛选东亚区域事件...")
        filtered = self.filter_events(events)

        print("[3/6] 智能事件采样...")
        self.select_representative_events(filtered)

        print("[4/6] 加载永久台站...")
        self.load_permanent_stations()

        print("[5/6] 生成输出文件...")
        output_files = self._save_all_outputs()

        print("[6/6] 完成")
        elapsed = (datetime.now() - start_time).total_seconds()
        self.print_summary()
        print(f"\n⏱️  总耗时: {elapsed:.1f} 秒")
        self.logger.info(f"🎉 标准测试数据库构建完成 (CMT3D 模式, 耗时 {elapsed:.1f} 秒)")

        return {
            'mode': 'cmt3d',
            'total_events': len(self.all_events),
            'filtered_events': len(self.filtered_events),
            'selected_events': len(self.selected_events),
            'stations': len(self.stations_df) if self.stations_df is not None else 0,
            'output_files': output_files,
            'elapsed_seconds': elapsed,
        }

    def _run_hybrid_30_mode(self) -> Dict[str, Any]:
        """
        hybrid_30 模式: CMT3D 2010–2019 → 排除既有 10 事件 → 空间均匀 30 事件。
        """
        cfg = self.config.hybrid_forward
        self.config.event_filters['time_range'] = list(cfg['time_range'])
        self.config.event_filters['magnitude_range'] = list(cfg['magnitude_range'])
        self.config.event_filters['depth_range'] = list(cfg['depth_range'])
        self.config.sampling['target_count'] = int(cfg['target_count'])
        self.config.sampling['min_event_distance_deg'] = float(cfg['min_event_distance_deg'])
        self.config.sampling['depth_quotas'] = {
            cat: n / float(cfg['target_count']) for cat, n in cfg['depth_quotas'].items()
        }

        self.logger.info("🚀 开始构建 hybrid_30 正演事件目录")
        start_time = datetime.now()

        print("\n[1/6] 解析 CMT3D 震源目录...")
        events = self.parse_cmt3d_catalog()

        print("[2/6] 筛选 2010–2019 东亚事件...")
        filtered = self.filter_events(events)

        print("[3/6] 空间均匀采样（排除既有 10 事件）...")
        self.select_spatially_uniform_events(filtered)

        print("[4/6] 加载永久台站...")
        self.load_permanent_stations()

        print("[5/6] 生成输出文件...")
        output_files = self._save_all_outputs()
        output_files['root_par'] = str(self.save_hybrid_download_catalog())
        output_files['id_map'] = str(self.save_id_mapping_csv())
        map_path = self.plot_hybrid_selection_map()
        if map_path is not None:
            output_files['map'] = str(map_path)
        try:
            paper_map = self.plot_paper_events_stations()
            output_files['paper_map'] = str(paper_map)
        except Exception as e:
            self.logger.warning(f"论文图绘制失败: {e}")

        print("[6/6] 完成")
        elapsed = (datetime.now() - start_time).total_seconds()
        self.print_summary()
        print(f"\n⏱️  总耗时: {elapsed:.1f} 秒")
        self.logger.info(f"🎉 hybrid_30 构建完成 (耗时 {elapsed:.1f} 秒)")

        return {
            'mode': 'hybrid_30',
            'total_events': len(self.all_events),
            'filtered_events': len(self.filtered_events),
            'selected_events': len(self.selected_events),
            'stations': len(self.stations_df) if self.stations_df is not None else 0,
            'output_files': output_files,
            'elapsed_seconds': elapsed,
        }

    def _run_eastasia_model_30_mode(self) -> Dict[str, Any]:
        """
        eastasia_model_30 模式: 从 EastAsia_model 已有波形库中
        选取 30 个 Mw≥5.5、远离研究区边界的正演测试事件。
        """
        cfg = self.config.eastasia_model_forward
        self.config.event_filters['magnitude_range'] = list(cfg['magnitude_range'])
        self.config.event_filters['depth_range'] = list(cfg['depth_range'])
        self.config.event_filters['time_range'] = list(cfg['time_range'])
        self.config.sampling['target_count'] = int(cfg['target_count'])

        self.logger.info("🚀 开始构建 eastasia_model_30 正演事件目录")
        start_time = datetime.now()

        print("\n[1/7] 解析 EastAsia_model CMTSOLUTION...")
        self.parse_eastasia_model_catalog()

        print("[2/7] 加载波形覆盖统计...")
        self.load_eastasia_waveform_coverage()

        if cfg.get('temporal_quotas'):
            print("[3/7] 使用分时段配额（跳过单一观测窗口）...")
        else:
            print("[3/7] 确定最优观测时间窗口...")
            self.load_or_compute_observation_window()

        print("[4/7] 筛选候选事件（Mw≥5.5，远离边界，时段范围内）...")
        self.filter_eastasia_model_candidates()

        print(f"[5/7] 分时段空间均匀采样 {cfg['target_count']} 个事件...")
        self.select_eastasia_model_30_events()

        print("[6/7] 加载 EastAsia_model 台站...")
        self.load_eastasia_model_stations()

        print("[7/7] 生成输出文件...")
        output_files: Dict[str, str] = {}
        output_files['cmtsolution_dir'] = str(self.copy_eastasia_model_cmtsolution_files())
        output_files['stations_file'] = str(self.generate_specfem_stations_file())
        output_files['catalog_csv'] = str(self.save_catalog_csv())
        output_files['root_par'] = str(self.save_eastasia_model_download_catalog())
        output_files['id_map'] = str(self.save_eastasia_model_id_mapping())
        output_files['metadata'] = str(self.save_metadata())
        output_files['report'] = str(self.generate_report())
        map_path = self.plot_eastasia_model_selection_map()
        if map_path is not None:
            output_files['map'] = str(map_path)

        print("[+] 生成论文地球透视图（事件+台站叠加）...")
        try:
            paper_globe = self.plot_eastasia_model_paper_globe()
            output_files['paper_globe'] = str(paper_globe)
        except Exception as exc:
            self.logger.warning(f"⚠️  论文地球透视图生成失败: {exc}")

        print("[+] 生成台站网络详图...")
        try:
            station_detail = self.plot_eastasia_model_station_network_detail()
            output_files['station_detail'] = str(station_detail)
        except Exception as exc:
            self.logger.warning(f"⚠️  台站网络详图生成失败: {exc}")

        elapsed = (datetime.now() - start_time).total_seconds()
        self.print_summary()
        if cfg.get('temporal_quotas'):
            parts = [
                f"{q['start_year']}–{q['end_year']}×{q['count']}"
                for q in cfg['temporal_quotas']
            ]
            print(f"\n📅 时段配额: {' + '.join(parts)}")
        elif self.observation_window:
            mean_sta = self.observation_window.get('mean_stations_per_event', 0)
            print(
                f"\n📅 观测窗口: {self.observation_window['start_year']}–"
                f"{self.observation_window['end_year']} "
                f"(均值 {mean_sta:.0f} 台站/事件)"
            )
        print(f"\n⏱️  总耗时: {elapsed:.1f} 秒")
        self.logger.info(f"🎉 eastasia_model_30 构建完成 (耗时 {elapsed:.1f} 秒)")

        return {
            'mode': 'eastasia_model_30',
            'total_events': len(self.all_events),
            'filtered_events': len(self.filtered_events),
            'selected_events': len(self.selected_events),
            'observation_window': self.observation_window,
            'stations': len(self.stations_df) if self.stations_df is not None else 0,
            'output_files': output_files,
            'elapsed_seconds': elapsed,
        }

    def _save_all_outputs(self) -> Dict[str, str]:
        """统一生成所有输出文件"""
        output_files = {}

        if self.config.output['save_cmtsolution']:
            output_files['cmtsolution_dir'] = str(self.save_cmtsolution_files())

        if self.config.output['save_stations']:
            output_files['stations_file'] = str(self.generate_specfem_stations_file())

        if self.config.output['save_catalog_csv']:
            output_files['catalog_csv'] = str(self.save_catalog_csv())

        if self.config.output['save_download_list']:
            output_files['download_list'] = str(self.save_download_list())

        output_files['metadata'] = str(self.save_metadata())

        if self.config.output['save_report']:
            output_files['report'] = str(self.generate_report())

        return output_files


# ================================================================
# 主函数
# ================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='EASTASIA-FWI 标准测试数据库 / 正演事件目录构建'
    )
    parser.add_argument(
        '--mode',
        choices=['fwea23', 'cmt3d', 'hybrid_30', 'eastasia_model_30'],
        default='fwea23',
        help='fwea23: FWEA23×CMT3D; cmt3d: 全量采样; '
             'hybrid_30: CMT3D 30事件; eastasia_model_30: 波形库30事件',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='输出目录（hybrid_30 / eastasia_model_30 有各自默认子目录）',
    )
    parser.add_argument(
        '--plot-paper',
        action='store_true',
        help='仅绘制 30 事件 + 测试台站论文图，不重新采样',
    )
    args = parser.parse_args()

    mode_desc = {
        'fwea23': 'FWEA23 参考事件 → CMT3D 匹配',
        'cmt3d': 'CMT3D 全量目录 → 筛选采样',
        'hybrid_30': 'CMT3D 2010+ → 排除 hybrid_10events → 空间均匀 30 事件',
        'eastasia_model_30': 'EastAsia_model 波形库 → Mw≥5.5 内部区 → 30 事件',
    }

    print("🎯 EASTASIA-FWI 标准测试数据库构建")
    print("=" * 60)
    print(f"构建模式: {args.mode} ({mode_desc.get(args.mode, '')})")
    print("用于 SPECFEM3D Globe 正演验证")
    print("=" * 60)

    try:
        output_dir = args.output_dir
        if output_dir is None:
            if args.mode == 'hybrid_30' or args.plot_paper:
                output_dir = str(
                    BaseConfig().dirs['events'] / 'hybrid_30events'
                )
            elif args.mode == 'eastasia_model_30':
                output_dir = str(
                    BaseConfig().dirs['events'] / 'eastasia_model_30events'
                )
        builder = TestDatabaseBuilder(output_dir=output_dir)

        if args.plot_paper:
            print("📄 绘制论文图：30 个筛选事件 + 实测波形质量台站")
            out = builder.plot_paper_events_stations()
            print(f"✅ 已保存: {out}")
            print("=" * 60)
            return
        results = builder.run(mode=args.mode)

        print(f"\n✅ 构建完成!")
        print(f"📁 输出目录: {builder.output_dir}")
        print(f"📊 选取 {results['selected_events']} 个事件 + "
              f"{results['stations']} 个永久台站")

        if results.get('mode') == 'fwea23':
            print(f"📖 FWEA23 参考事件: {results.get('fwea23_total', 0)}")
            print(f"🔗 CMT3D 匹配成功: {results.get('cmt3d_matched', 0)}")

        print("\n📋 输出文件:")
        for key, path in results['output_files'].items():
            print(f"  ✅ {key}: {path}")

        print("\n💡 后续步骤:")
        print("  1. 使用 1_3_Download_waveforms.py 下载波形数据")
        if args.mode == 'hybrid_30':
            print("     - 使用 data/events/download_catalog_hybrid_30events.par")
        elif args.mode == 'eastasia_model_30':
            print("     - 使用 data/events/download_catalog_eastasia_model_30events.par")
            print("     - 波形已在 data/events/EastAsia_model/data_obs/ 中")
        else:
            print("     - 使用 download_catalog.par 作为事件目录")
        print("     - 启用永久台站模式 (use_permanent_only=True)")
        print("  2. 使用 1_4_Preprocess_waveforms.py 预处理波形")
        print("  3. 将 CMTSOLUTION 和 STATIONS 文件部署到 SPECFEM3D")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断处理")
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)


if __name__ == "__main__":
    main()
