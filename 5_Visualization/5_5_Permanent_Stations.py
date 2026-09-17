"""
EASTASIA-FWI 台站网络分析、可视化与 SPECFEM STATIONS 导出
============================================================

功能描述:
- 所有台站 / 永久台站分布图（PyGMT，含震源球叠加）
- 永久台网时间线分析、密集台网筛选
- SPECFEM3D Globe STATIONS 文件导出（按模型 chunk 几何裁剪）
- 台站去重处理（基于 StationID，排除合成台站 SY 等）
- 跨台网近距去重：缓解中亚等地 KN/KR/AD 多套台网同址叠加导致的图上过密

科学原理:
- 正演测试使用永久台站（>5 年且活跃），确保长期观测记录可用
- 吉尔吉斯斯坦一带 KNET(KN) 与 KRNET(KR) 二选一（保留研究区内覆盖更大者）；ACROSS(AD) 强震网整网排除
- 不同速度模型 chunk 覆盖区域不同，需独立生成 STATIONS 文件

作者: EASTASIA-FWI Team
日期: 2026-03-25
版本: v2.0
"""

import math
import pygmt
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from datetime import datetime
import warnings
from matplotlib import cm

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig, strip_model_year


class StationAnalysisConfig:
    """台站分析配置参数（完善版）"""
    
    def __init__(self):
        # 可视化配置
        self.visualization = {
            'dpi': 600,
            'pdf_dpi': 720,
            'figure_format': ['png', 'pdf'],
            'map_projection': "M15c",
            'max_legend_networks': 50  # 增加图例显示的网络数量
        }
        
        # 地图配置
        self.map = {
            'resolution': "f",
            'land_color': "lightgray", 
            'water_color': "lightblue",
            'shoreline_pen': "0.1p",
            'area_threshold': 1000,
            'border_pen': "1/0.5p,black,-"
        }
        
        # 台站标记配置
        self.markers = {
            'all_stations': {
                'symbol': "t0.25c",  # 小圆形
                'pen': "0.1p,black"
            },
            'permanent_stations': {
                'symbol': "t0.4c",  # 三角形
                'pen': "0.2p,black"
            }
        }
        
        # 永久台站筛选条件
        self.permanent_criteria = {
            'min_operation_years': 5,  # 最少运行年数
            'must_be_active': True     # 必须当前活跃（EndDate为空）
        }
        
        # 密集台网筛选条件
        self.dense_network_criteria = {
            'distance_threshold_degrees': 0.5,  # 0.5度范围
            'min_stations_threshold': 5         # 超过5个台站
        }

        # 跨台网近距去重（中亚等同址多套台网叠加时稀疏化）
        self.cross_network_dedup = {
            'enabled': True,
            # 最小球面距离（度），小于此距离的台站只保留优先级更高者
            'min_separation_deg': 0.5,
            # 台网优先级：前项优先保留（全波形/骨干网优先）
            'network_priority': (
                'IC', 'IU', 'II', 'G', 'GE', 'AU', 'JP', 'MM', 'MY', 'TW', 'KS',
                'KN', 'KR', 'TM', 'TJ', 'KZ', 'CB', 'KC', 'CK',
                'RM', 'PS', 'NQ', 'NK', 'S1',
            ),
        }

        # 吉尔吉斯两套宽频网 KN(KNET) 与 KR(KRNET) 二选一：保留研究区内包络面积更大者
        self.kyrgyz_broadband_exclusive = {
            'enabled': True,
            'network_codes': ('KN', 'KR'),
        }
        
        # 排除的台网列表
        self.excluded_networks = {
            'SY', 'NT',   # Synthetic Seismograms (合成地震图)
            'AD',         # ACROSS 强震网（与宽频用途重叠，本项目不采用）
        }
        
        # 去重配置（参考原始代码）
        self.deduplication = {
            'method': 'station_id',  # 基于StationID去重
            'priority_providers': ['IRIS', 'GEOFON', 'GFZ', 'AUSPASS', 'ETH', 'GEONET', 'IPGP', 'RESIF', 'SCEDC', 'USP'],
            'keep_criteria': 'priority_provider'
        }
        
        # 时间线图配置
        self.timeline = {
            'figure_size': (18, 10),  # 年度部署统计图尺寸
            'timeline_figure_size': (9, 8),  # Timeline图尺寸 (3:4比例)
            'colors': ['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC'],
            # 字体配置
            'font_sizes': {
                'suptitle': 22,
                'title': 18,
                'label': 16,
                'tick': 14,
                'annotation': 14,
                'network_label': 13,
                'duration_label': 12,
                'info_text': 14
            }
        }
        
        # 震源球叠加配置
        self.focal_mechanism = {
            'min_magnitude': 7.0,           # 显示震源球的最小震级
            'max_events': 50,              # 最大显示事件数
            'scale': {
                'min': 0.15,                # 最小比例
                'max': 0.45,                # 最大比例
                'base': 0.2,                # 基础比例
                'increment': 0.05           # 每级增量
            },
            'style': {
                'pen': "0.2p,gray30,solid",
                'extensionfill': "cornsilk",
                'compressionfill': "red",
                'fallback_symbol': "c",     # 无机制解时使用圆圈
                'fallback_fill': "orange",
                'fallback_pen': "0.3p,black"
            },
            'depth_cmap': "jet",            # 深度色标
            'depth_range': (0, 700)         # 深度范围 (km)
        }


class StationNetworkAnalyzer:
    """台站网络分析器（完善图例和颜色版）"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """初始化台站网络分析器"""
        
        # 加载基础配置
        self.base_config = BaseConfig()
        self.config = StationAnalysisConfig()
        
        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.base_config.dirs['figures']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.StationAnalyzer',
            'INFO'
        )
        
        # 设置地图区域 - 确保使用正确的东亚研究区域
        region_bounds = self.base_config.get_region_bounds()
        self.region = [
            region_bounds['lon_min'], region_bounds['lon_max'],
            region_bounds['lat_min'], region_bounds['lat_max']
        ]
        
        self.logger.info(f"🌏 东亚研究区域: 经度 {self.region[0]}°-{self.region[1]}°, 纬度 {self.region[2]}°-{self.region[3]}°")
        
        # 设置PyGMT配置
        self._setup_pygmt_config()
        
        # 生成颜色方案（参考原代码）
        self._generate_network_colors()
        
        # 数据存储
        self.all_stations = None
        self.permanent_stations = None
        self.filtered_permanent_stations = None
        self.network_stats = None
        self.permanent_network_color_map = None  # 新增：存储永久台站的颜色映射
        
        self.logger.info("🎯 台站网络分析器（完善图例和颜色版）初始化完成")

    def _setup_pygmt_config(self):
        """设置PyGMT全局配置"""
        pygmt.config(
            MAP_FRAME_TYPE="plain",
            MAP_GRID_PEN_PRIMARY="0.3p,dimgrey", 
            MAP_ANNOT_OBLIQUE="30",
            MAP_ANNOT_OFFSET_PRIMARY="5p",
            MAP_ANNOT_OFFSET_SECONDARY="5p",
            FONT_ANNOT_PRIMARY="10p,4",
            FONT_LABEL="10p,28,black",
            MAP_FRAME_WIDTH="2p",
            MAP_FRAME_PEN="0.5p",
            MAP_TICK_LENGTH_PRIMARY="5p",
            MAP_TICK_PEN_PRIMARY="0.5p,black",
            MAP_LABEL_OFFSET="5p",
        )

    def _generate_network_colors(self):
        """生成网络配色方案（参考原5_3代码）"""
        # 使用matplotlib的高质量配色方案
        tab20_colors = [cm.get_cmap('tab20')(i) for i in range(20)]
        tab20b_colors = [cm.get_cmap('tab20b')(i) for i in range(20)]
        all_colors = tab20_colors + tab20b_colors
        
        self.colors_combined = [
            f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            for r, g, b, _ in all_colors
        ]
        
        self.logger.info(f"🎨 生成了 {len(self.colors_combined)} 种网络配色")

    def load_station_data(self, station_file: Optional[str] = None) -> pd.DataFrame:
        """
        加载台站数据
        
        Args:
            station_file: 台站数据文件路径
            
        Returns:
            台站数据DataFrame
        """
        try:
            if station_file and Path(station_file).exists():
                data_file = Path(station_file)
            else:
                # 查找默认数据文件
                stations_dir = self.base_config.dirs['stations']
                possible_files = [
                    stations_dir / f"{self.base_config.region['name']}_stations.csv",
                    stations_dir / "EastAsia_stations.csv",
                    stations_dir / "stations.csv"
                ]
                
                data_file = None
                for file_path in possible_files:
                    if file_path.exists():
                        data_file = file_path
                        break
                
                if data_file is None:
                    raise FileNotFoundError("未找到可用的台站数据文件")
            
            # 读取数据 - 根据您提供的CSV结构
            self.all_stations = pd.read_csv(data_file)
            self.logger.info(f"📊 加载台站数据: {data_file}")
            self.logger.info(f"   原始台站总数: {len(self.all_stations)}")
            
            # 数据预处理
            self._preprocess_station_data()
            
            # 识别永久台站
            self._identify_permanent_stations()
            
            # 筛选密集台网
            self._filter_dense_networks()

            # KN/KR 二选一（保留研究区内空间覆盖更大者）
            self._apply_kyrgyz_kn_kr_exclusive()

            # 跨台网近距去重
            self._deduplicate_cross_network_proximity()
            
            # 生成网络统计
            self._generate_network_statistics()
            
            return self.all_stations
            
        except Exception as e:
            self.logger.error(f"加载台站数据失败: {e}")
            raise

    def _preprocess_station_data(self):
        """预处理台站数据（包含正确的去重逻辑）"""
        try:
            original_count = len(self.all_stations)
            
            # 确保必要的列存在
            required_columns = ['Network', 'Station', 'Latitude', 'Longitude']
            missing_columns = [col for col in required_columns if col not in self.all_stations.columns]
            if missing_columns:
                raise ValueError(f"缺少必要的列: {missing_columns}")
            
            # 1. 排除合成台站和测试台网
            excluded_networks = self.config.excluded_networks
            excluded_mask = self.all_stations['Network'].isin(excluded_networks)
            excluded_count = excluded_mask.sum()
            
            if excluded_count > 0:
                excluded_networks_found = self.all_stations[excluded_mask]['Network'].unique()
                self.logger.info(
                    f"🚫 排除台网: {list(excluded_networks_found)} ({excluded_count} 个台站)"
                )
                self.all_stations = self.all_stations[~excluded_mask]
            
            # 2. 确保坐标有效
            coord_before = len(self.all_stations)
            self.all_stations = self.all_stations.dropna(subset=['Latitude', 'Longitude'])
            coord_dropped = coord_before - len(self.all_stations)
            if coord_dropped > 0:
                self.logger.info(f"🧹 删除无效坐标台站: {coord_dropped} 个")
            
            # 3. 严格的地理边界检查 - 确保在东亚研究区域内
            region_before = len(self.all_stations)
            region_mask = (
                (self.all_stations['Longitude'] >= self.region[0]) & 
                (self.all_stations['Longitude'] <= self.region[1]) &
                (self.all_stations['Latitude'] >= self.region[2]) & 
                (self.all_stations['Latitude'] <= self.region[3])
            )
            self.all_stations = self.all_stations[region_mask]
            region_dropped = region_before - len(self.all_stations)
            if region_dropped > 0:
                self.logger.info(f"🌏 删除区域外台站: {region_dropped} 个")
            
            # 4. 创建StationID用于去重（参考原始代码逻辑）
            self.all_stations['StationID'] = self.all_stations['Network'] + '.' + self.all_stations['Station']
            
            # 5. 执行去重处理（参考原始代码逻辑）
            self._remove_duplicate_stations()
            
            # 6. 处理时间数据
            if 'StartDate' in self.all_stations.columns:
                self.all_stations['StartDate'] = pd.to_datetime(self.all_stations['StartDate'], errors='coerce')
                self.all_stations['StartYear'] = self.all_stations['StartDate'].dt.year
            
            if 'EndDate' in self.all_stations.columns:
                # EndDate为空表示台站仍在运行
                self.all_stations['EndDate'] = pd.to_datetime(self.all_stations['EndDate'], errors='coerce')
                self.all_stations['EndYear'] = self.all_stations['EndDate'].dt.year
                
            # 7. 计算运行时长
            current_date = datetime.now()
            if 'StartDate' in self.all_stations.columns:
                # 如果EndDate为空，使用当前时间
                self.all_stations['EffectiveEndDate'] = self.all_stations['EndDate'].fillna(current_date)
                self.all_stations['OperationYears'] = (
                    (self.all_stations['EffectiveEndDate'] - self.all_stations['StartDate']).dt.days / 365.25
                )
            
            final_count = len(self.all_stations)
            self.logger.info(f"✅ 数据预处理完成:")
            self.logger.info(f"   原始台站: {original_count} 个")
            self.logger.info(f"   最终有效台站: {final_count} 个")
            self.logger.info(f"   涉及网络: {self.all_stations['Network'].nunique()} 个")
            
            # 显示地理分布范围
            lon_range = f"{self.all_stations['Longitude'].min():.2f}° - {self.all_stations['Longitude'].max():.2f}°"
            lat_range = f"{self.all_stations['Latitude'].min():.2f}° - {self.all_stations['Latitude'].max():.2f}°"
            self.logger.info(f"   经度范围: {lon_range}")
            self.logger.info(f"   纬度范围: {lat_range}")
            
        except Exception as e:
            self.logger.error(f"数据预处理失败: {e}")
            raise

    def _remove_duplicate_stations(self):
        """
        去重处理：基于StationID和Provider优先级（参考原始代码逻辑）
        """
        self.logger.info("🔄 执行台站去重处理...")
        initial_count = len(self.all_stations)
        
        try:
            dedup_config = self.config.deduplication
            priority_providers = dedup_config.get('priority_providers', [])
            
            # 确保Provider列存在，如果不存在则使用DataSource或其他标识
            provider_column = None
            for col in ['Provider', 'DataSource', 'QuerySource']:
                if col in self.all_stations.columns:
                    provider_column = col
                    break
            
            if provider_column is None:
                self.logger.warning("⚠️ 未找到Provider信息列，跳过基于优先级的去重")
                # 简单基于StationID去重，保留第一个
                self.all_stations = self.all_stations.drop_duplicates(subset=['StationID'], keep='first')
            else:
                # 添加Provider优先级
                self.all_stations['provider_priority'] = self.all_stations[provider_column].apply(
                    lambda x: priority_providers.index(x) if x in priority_providers else len(priority_providers)
                )
                
                # 排序并去重：优先级高的排在前面
                df_sorted = self.all_stations.sort_values('provider_priority')
                self.all_stations = df_sorted.drop_duplicates(subset=['StationID'], keep='first')
                
                # 删除临时列
                if 'provider_priority' in self.all_stations.columns:
                    self.all_stations = self.all_stations.drop('provider_priority', axis=1)
            
            # 重置索引
            self.all_stations = self.all_stations.reset_index(drop=True)
            
            removed_count = initial_count - len(self.all_stations)
            self.logger.info(f"✅ 去重完成: 移除 {removed_count} 个重复台站")
            
            if removed_count > 0:
                self.logger.info(f"   去重前: {initial_count} 个台站记录")
                self.logger.info(f"   去重后: {len(self.all_stations)} 个唯一台站")
                
                # 显示去重效果统计
                unique_stations = self.all_stations['StationID'].nunique()
                self.logger.info(f"   唯一台站数: {unique_stations}")
            
        except Exception as e:
            self.logger.error(f"去重处理错误: {e}")
            # 如果去重失败，至少执行基本去重
            self.all_stations = self.all_stations.drop_duplicates(subset=['StationID'], keep='first')

    def _identify_permanent_stations(self):
        """识别永久台站：持续时间>5年且EndDate为空"""
        try:
            criteria = self.config.permanent_criteria
            
            # 筛选条件
            mask = (
                # 运行时间超过5年
                (self.all_stations['OperationYears'] > criteria['min_operation_years']) &
                # EndDate为空（仍在运行）
                (self.all_stations['EndDate'].isna())
            )
            
            self.permanent_stations = self.all_stations[mask].copy()
            
            self.logger.info(f"🏛️ 识别永久台站: {len(self.permanent_stations)} 个")
            self.logger.info(f"   筛选条件: 运行时间>{criteria['min_operation_years']}年 且 当前仍活跃")
            
            if len(self.permanent_stations) > 0:
                # 永久台站网络统计
                perm_networks = self.permanent_stations['Network'].value_counts()
                self.logger.info(f"   涉及网络数: {len(perm_networks)}")
                self.logger.info(f"   最大网络: {perm_networks.index[0]} ({perm_networks.iloc[0]} 个台站)")
                
                # 显示前5个网络
                self.logger.info("   主要永久台站网络:")
                for i, (network, count) in enumerate(perm_networks.head(5).items()):
                    self.logger.info(f"     {i+1}. {network}: {count} 个台站")
            
        except Exception as e:
            self.logger.error(f"永久台站识别失败: {e}")
            self.permanent_stations = pd.DataFrame()

    def _filter_dense_networks(self):
        """筛选密集台网：0.5度范围内超过5个台站的台网需要筛选"""
        try:
            if self.permanent_stations is None or len(self.permanent_stations) == 0:
                self.filtered_permanent_stations = pd.DataFrame()
                return
                
            criteria = self.config.dense_network_criteria
            distance_threshold = criteria['distance_threshold_degrees']
            min_stations = criteria['min_stations_threshold']
            
            # 按网络分组处理
            filtered_stations = []
            dense_networks = []
            
            for network in self.permanent_stations['Network'].unique():
                network_stations = self.permanent_stations[
                    self.permanent_stations['Network'] == network
                ].copy()
                
                if len(network_stations) <= min_stations:
                    # 台站数不超过阈值，直接保留
                    filtered_stations.append(network_stations)
                else:
                    # 检查是否为密集台网
                    is_dense = self._check_network_density(
                        network_stations, distance_threshold, min_stations
                    )
                    
                    if is_dense:
                        dense_networks.append(network)
                        self.logger.info(f"⚠️  排除密集台网: {network} ({len(network_stations)} 个台站)")
                        # 完全排除密集台网
                        continue
                    else:
                        filtered_stations.append(network_stations)
            
            # 合并筛选后的台站
            if filtered_stations:
                self.filtered_permanent_stations = pd.concat(filtered_stations, ignore_index=True)
            else:
                self.filtered_permanent_stations = pd.DataFrame()
            
            self.logger.info(f"🔍 密集台网筛选完成:")
            self.logger.info(f"   原永久台站数: {len(self.permanent_stations)}")
            self.logger.info(f"   排除密集台网数: {len(dense_networks)}")
            self.logger.info(f"   筛选后台站数: {len(self.filtered_permanent_stations)}")
            
            if dense_networks:
                self.logger.info(f"   排除的密集台网: {dense_networks}")
            
        except Exception as e:
            self.logger.error(f"密集台网筛选失败: {e}")
            self.filtered_permanent_stations = self.permanent_stations.copy()

    def _check_network_density(self, network_stations: pd.DataFrame, 
                              distance_threshold: float, min_stations: int) -> bool:
        """检查网络是否为密集台网（改进算法）"""
        try:
            coords = network_stations[['Latitude', 'Longitude']].values
            n_stations = len(coords)
            
            if n_stations <= min_stations:
                return False
            
            # 计算所有台站对之间的距离
            dense_stations = 0
            
            for i in range(n_stations):
                nearby_count = 0
                for j in range(n_stations):
                    if i != j:
                        # 计算度数距离
                        lat_diff = abs(coords[i][0] - coords[j][0])
                        lon_diff = abs(coords[i][1] - coords[j][1])
                        
                        # 使用更严格的距离计算
                        euclidean_distance = np.sqrt(lat_diff**2 + lon_diff**2)
                        
                        if euclidean_distance <= distance_threshold:
                            nearby_count += 1
                
                # 如果一个台站周围有超过阈值数量的邻近台站
                if nearby_count >= min_stations:
                    dense_stations += 1
            
            # 如果超过60%的台站都处于密集区域，认为是密集台网
            density_ratio = dense_stations / n_stations
            is_dense = density_ratio > 0.6
            
            if is_dense:
                self.logger.debug(f"   密集台网检测: {network_stations['Network'].iloc[0]} - "
                               f"密集台站比例: {density_ratio:.2f}")
            
            return is_dense
            
        except Exception as e:
            self.logger.warning(f"密集度检查失败: {e}")
            return False

    @staticmethod
    def _angular_distance_deg(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """两点间大圆距离（度）"""
        r = math.radians
        dlat = r(lat2 - lat1)
        dlon = r(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(r(lat1)) * math.cos(r(lat2)) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(min(1.0, math.sqrt(max(0.0, a))))
        return math.degrees(c)

    @staticmethod
    def _network_bbox_area_deg2(sub: pd.DataFrame) -> float:
        """台网在研究区内的经纬度包络面积（度²），单点取小正数避免 0。"""
        if sub is None or len(sub) == 0:
            return -1.0
        dlat = float(sub['Latitude'].max() - sub['Latitude'].min())
        dlon = float(sub['Longitude'].max() - sub['Longitude'].min())
        dlat = max(dlat, 0.02)
        dlon = max(dlon, 0.02)
        return dlat * dlon

    def _apply_kyrgyz_kn_kr_exclusive(self) -> None:
        """
        KN(KNET) 与 KR(KRNET) 仅保留一套：保留东亚研究区内包络面积更大者；
        面积并列时保留台站数更多者。
        """
        cfg = self.config.kyrgyz_broadband_exclusive
        if not cfg.get('enabled', True):
            return
        if self.filtered_permanent_stations is None or len(self.filtered_permanent_stations) == 0:
            return

        kn_code, kr_code = cfg['network_codes']
        df = self.filtered_permanent_stations
        m_kn = df['Network'].astype(str) == kn_code
        m_kr = df['Network'].astype(str) == kr_code
        if not m_kn.any() or not m_kr.any():
            return

        sub_kn = df[m_kn]
        sub_kr = df[m_kr]
        area_kn = self._network_bbox_area_deg2(sub_kn)
        area_kr = self._network_bbox_area_deg2(sub_kr)

        if area_kn > area_kr:
            keep, drop = kn_code, kr_code
        elif area_kr > area_kn:
            keep, drop = kr_code, kn_code
        else:
            if len(sub_kn) >= len(sub_kr):
                keep, drop = kn_code, kr_code
            else:
                keep, drop = kr_code, kn_code

        n_drop = int((df['Network'].astype(str) == drop).sum())
        self.filtered_permanent_stations = df[df['Network'].astype(str) != drop].copy()
        self.logger.info(
            f"🇰🇬 吉尔吉斯宽频网二选一: 保留 {keep} (包络 {max(area_kn, area_kr):.4f} 度²), "
            f"移除 {drop} ({n_drop} 台)"
        )

    def _deduplicate_cross_network_proximity(self) -> None:
        """
        跨台网近距去重。

        中亚等地同时存在 KNET(KN)、吉尔吉斯数字网(KR)、ACROSS 强震网(AD) 等，
        目录上合法但空间高度重叠；全波形应用宜保留骨干/宽频台，同址强震点降级。
        """
        cfg = self.config.cross_network_dedup
        if not cfg.get('enabled', True):
            return
        if self.filtered_permanent_stations is None or len(self.filtered_permanent_stations) == 0:
            return

        df = self.filtered_permanent_stations
        n_before = len(df)
        priority: tuple = cfg['network_priority']
        min_sep = float(cfg['min_separation_deg'])

        def prio(net: str) -> int:
            try:
                return priority.index(str(net))
            except ValueError:
                return len(priority)

        work = df.copy()
        work['_p'] = work['Network'].astype(str).map(prio)
        if 'OperationYears' in work.columns:
            work['_oy'] = pd.to_numeric(work['OperationYears'], errors='coerce').fillna(0)
        else:
            work['_oy'] = 0.0
        work = work.sort_values(['_p', '_oy'], ascending=[True, False])

        kept_latlon: List[tuple] = []
        kept_rows: List[pd.Series] = []
        for _, row in work.iterrows():
            lat = float(row['Latitude'])
            lon = float(row['Longitude'])
            conflict = False
            for klat, klon in kept_latlon:
                if self._angular_distance_deg(lat, lon, klat, klon) < min_sep:
                    conflict = True
                    break
            if not conflict:
                kept_latlon.append((lat, lon))
                kept_rows.append(row.drop(labels=['_p', '_oy'], errors='ignore'))

        self.filtered_permanent_stations = pd.DataFrame(kept_rows).reset_index(drop=True)
        n_after = len(self.filtered_permanent_stations)
        removed = n_before - n_after
        if removed > 0:
            self.logger.info(
                f"🧭 跨台网近距去重: 移除 {removed} 个台站 "
                f"({n_before} → {n_after}, 最小间距 {min_sep}°)"
            )
        else:
            self.logger.info(
                f"🧭 跨台网近距去重: 无需移除 (最小间距 {min_sep}°)"
            )

    def _generate_network_statistics(self):
        """生成网络统计信息"""
        try:
            if self.filtered_permanent_stations is not None and len(self.filtered_permanent_stations) > 0:
                # 使用筛选后的永久台站进行统计
                stats_data = self.filtered_permanent_stations
            else:
                # 如果没有永久台站，使用所有台站
                stats_data = self.all_stations
            
            # 按网络分组统计
            network_groups = stats_data.groupby('Network')
            
            self.network_stats = pd.DataFrame({
                'StationCount': network_groups.size(),
                'MeanLongitude': network_groups['Longitude'].mean(),
                'MeanLatitude': network_groups['Latitude'].mean()
            })
            
            # 添加时间统计
            if 'StartYear' in stats_data.columns:
                self.network_stats['MinStartYear'] = network_groups['StartYear'].min()
                self.network_stats['MaxStartYear'] = network_groups['StartYear'].max()
                if 'OperationYears' in stats_data.columns:
                    self.network_stats['MeanOperationYears'] = network_groups['OperationYears'].mean()
            
            # 按台站数排序
            self.network_stats = self.network_stats.sort_values('StationCount', ascending=False)
            
            self.logger.info(f"📊 网络统计生成完成:")
            self.logger.info(f"   网络总数: {len(self.network_stats)}")
            if len(self.network_stats) > 0:
                self.logger.info(f"   最大网络: {self.network_stats.index[0]} ({self.network_stats.iloc[0]['StationCount']} 个台站)")
            
        except Exception as e:
            self.logger.error(f"生成网络统计失败: {e}")
            self.network_stats = pd.DataFrame()

    def _create_network_legend(self, station_data: pd.DataFrame, 
                             color_map: Dict[str, str]) -> str:
        """创建网络图例文件（参考原5_3代码格式）"""
        try:
            legend_file = self.output_dir / "station_network_legend_temp.txt"
            
            # 统计网络台站数量
            station_counts = station_data['Network'].value_counts().sort_values(ascending=False)
            
            # 创建图例内容
            legend_content = """H 13.5p,Helvetica-Bold Network Info
N 3
G 0.2c
"""
            
            # 限制显示的网络数量
            max_networks = min(self.config.visualization['max_legend_networks'], len(station_counts))
            
            for i, (network, count) in enumerate(station_counts.head(max_networks).items()):
                color = color_map.get(network, '#FF4500')
                legend_content += f"S 0.3c t 0.25c {color} 0.2p,black 0.8c {network} ({count})\n"
            
            # 如果有更多网络，添加说明
            if len(station_counts) > max_networks:
                remaining = len(station_counts) - max_networks
                legend_content += f"T ... and {remaining} more networks\n"
            
            # 保存图例文件
            with open(legend_file, 'w') as f:
                f.write(legend_content)
            
            return str(legend_file)
            
        except Exception as e:
            self.logger.error(f"网络图例创建失败: {e}")
            return ""

    def _create_complete_network_legend(self, station_data: pd.DataFrame, 
                                      color_map: Dict[str, str]) -> str:
        """创建完整的网络图例文件（显示所有网络）"""
        try:
            legend_file = self.output_dir / "station_network_legend_complete.txt"
            
            # 统计网络台站数量
            station_counts = station_data['Network'].value_counts().sort_values(ascending=False)
            
            # 创建图例内容
            legend_content = """H 13.5p,Helvetica-Bold Network Info
N 3
G 0.15c
"""
            
            # 显示所有网络（不限制数量）
            for i, (network, count) in enumerate(station_counts.items()):
                color = color_map.get(network, '#FF4500')
                legend_content += f"S 0.3c t 0.45c {color} 0.2p,black 0.8c {network} ({count})\n"
                
            
            # 保存图例文件
            with open(legend_file, 'w') as f:
                f.write(legend_content)
            
            self.logger.info(f"✅ 完整网络图例文件创建: {legend_file} (包含 {len(station_counts)} 个网络)")
            
            return str(legend_file)
            
        except Exception as e:
            self.logger.error(f"完整网络图例创建失败: {e}")
            return ""

    def plot_all_stations_distribution(self, 
                                     save_file: str = "5-4-all_stations_distribution.png") -> str:
        """绘制所有台站分布图"""
        
        self.logger.info("🗺️ 开始绘制所有台站分布图...")
        
        try:
            if self.all_stations is None or len(self.all_stations) == 0:
                self.logger.warning("没有台站数据")
                return ""
            
            # 创建图形
            fig = pygmt.Figure()
            
            # 绘制基础地图
            map_config = self.config.map
            fig.coast(
                resolution=map_config['resolution'],
                land=map_config['land_color'],
                water=map_config['water_color'],
                shorelines=map_config['shoreline_pen'],
                area_thresh=map_config['area_threshold'],
                region=self.region,
                projection=self.config.visualization['map_projection'],
                frame=["a10f1", "WSen+t'All Seismic Stations in East Asia (Deduplicated)'"],
                borders=[map_config['border_pen']], 
            )
            
            # 生成颜色映射
            colors = self.colors_combined
            station_counts = self.all_stations['Network'].value_counts().sort_values(ascending=False)
            color_map = {}
            for i, network in enumerate(station_counts.index):
                color_map[network] = colors[i % len(colors)]
            
            # 按网络分组绘制台站
            marker_config = self.config.markers['all_stations']
            for network, color in color_map.items():
                network_stations = self.all_stations[self.all_stations['Network'] == network]
                
                if len(network_stations) > 0:
                    fig.plot(
                        x=network_stations['Longitude'],
                        y=network_stations['Latitude'],
                        style=marker_config['symbol'],
                        pen=marker_config['pen'],
                        fill=color,
                    )
            
            # 添加图例
            legend_file = self._create_network_legend(self.all_stations, color_map)
            if legend_file and Path(legend_file).exists():
                try:
                    pygmt.config(FONT_ANNOT_PRIMARY="10p,5,black")
                    fig.legend(
                        spec=legend_file,
                        position="JBL+w15c/7c+jTM+o0c/-3.5c+l1.5",
                        box="+gwhite+p1p",
                    )
                    # 删除临时文件
                    Path(legend_file).unlink()
                except Exception as e:
                    self.logger.warning(f"图例添加失败: {e}")
            
            # 保存图片
            output_path = self.output_dir / save_file
            viz_config = self.config.visualization
            fig.savefig(str(output_path), dpi=viz_config['dpi'], crop=True)
            
            # 同时保存PDF版本
            if 'pdf' in viz_config['figure_format']:
                pdf_path = output_path.with_suffix('.pdf')
                fig.savefig(str(pdf_path), dpi=viz_config['pdf_dpi'], crop=True)
            
            self.logger.info(f"✅ 所有台站分布图保存至: {output_path}")
            self.logger.info(f"   台站总数: {len(self.all_stations)}")
            self.logger.info(f"   网络总数: {len(station_counts)}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"所有台站分布图绘制失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def plot_permanent_stations_distribution(self, 
                                           save_file: str = "5-4-permanent_stations_distribution.png") -> str:
        """绘制永久台站分布图（完整图例版）"""
        
        self.logger.info("🏛️ 开始绘制永久台站分布图（完整图例版）...")
        
        try:
            if self.filtered_permanent_stations is None or len(self.filtered_permanent_stations) == 0:
                self.logger.warning("没有永久台站数据")
                return ""
            
            # 创建图形
            fig = pygmt.Figure()
            
            # 绘制基础地图
            map_config = self.config.map
            fig.coast(
                resolution=map_config['resolution'],
                land=map_config['land_color'],
                water=map_config['water_color'],
                shorelines=map_config['shoreline_pen'],
                area_thresh=map_config['area_threshold'],
                region=self.region,
                projection=self.config.visualization['map_projection'],
                frame=["a10f1", "WSen+t'Permanent Seismic Stations in East Asia (>5 years, Active, Filtered)'"],
                borders=[map_config['border_pen']], 
            )
            
            # 生成颜色映射
            colors = self.colors_combined
            station_counts = self.filtered_permanent_stations['Network'].value_counts().sort_values(ascending=False)
            color_map = {}
            for i, network in enumerate(station_counts.index):
                color_map[network] = colors[i % len(colors)]
            
            # 存储颜色映射供时间线图使用
            self.permanent_network_color_map = color_map
            
            # 按网络分组绘制台站
            marker_config = self.config.markers['permanent_stations']
            for network, color in color_map.items():
                network_stations = self.filtered_permanent_stations[
                    self.filtered_permanent_stations['Network'] == network
                ]
                
                if len(network_stations) > 0:
                    fig.plot(
                        x=network_stations['Longitude'],
                        y=network_stations['Latitude'],
                        style=marker_config['symbol'],
                        pen=marker_config['pen'],
                        fill=color,
                    )
            
            # 添加完整图例
            legend_file = self._create_complete_network_legend(self.filtered_permanent_stations, color_map)
            if legend_file and Path(legend_file).exists():
                try:
                    pygmt.config(FONT_ANNOT_PRIMARY="10p,5,black")  # 缩小字体以容纳更多内容
                    fig.legend(
                        spec=legend_file,
                        position="JBL+w15c/7c+jTM+o0c/-3.5c+l1.5",
                        box="+gwhite+p1p",
                    )
                    # 删除临时文件
                    Path(legend_file).unlink()
                    self.logger.info("✅ 完整网络图例已添加")
                except Exception as e:
                    self.logger.warning(f"完整图例添加失败: {e}")
            
            # 保存图片
            output_path = self.output_dir / save_file
            viz_config = self.config.visualization
            fig.savefig(str(output_path), dpi=viz_config['dpi'], crop=True)
            
            # 同时保存PDF版本
            if 'pdf' in viz_config['figure_format']:
                pdf_path = output_path.with_suffix('.pdf')
                fig.savefig(str(pdf_path), dpi=viz_config['pdf_dpi'], crop=True)
            
            self.logger.info(f"✅ 永久台站分布图保存至: {output_path}")
            self.logger.info(f"   永久台站数: {len(self.filtered_permanent_stations)}")
            self.logger.info(f"   网络总数: {len(station_counts)}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"永久台站分布图绘制失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def load_gcmt_data(self, 
                       gcmt_file: Optional[str] = None,
                       min_magnitude: float = 6.0,
                       depth_range: tuple = (0, 700),
                       max_events: int = 100) -> pd.DataFrame:
        """
        加载并筛选 GCMT 地震数据
        
        Args:
            gcmt_file: GCMT 数据文件路径 (可选)
            min_magnitude: 最小震级
            depth_range: 深度范围 (km)
            max_events: 最大事件数
            
        Returns:
            筛选后的 GCMT DataFrame
        """
        self.logger.info(f"📊 加载 GCMT 数据 (M >= {min_magnitude})...")
        
        try:
            # 查找 GCMT 数据文件
            if gcmt_file and Path(gcmt_file).exists():
                df = pd.read_csv(gcmt_file)
                self.logger.info(f"   使用指定文件: {gcmt_file}")
            else:
                # 在默认位置查找
                catalogs_dir = self.base_config.dirs.get('catalogs', 
                    self.base_config.dirs['data'] / 'catalogs')
                
                # 搜索可能的文件名模式
                possible_patterns = [
                    "gcmt_catalogs_deduped_*.csv",  # 去重后的文件
                    "gcmt_catalogs_*.csv",          # 原始文件
                    "gcmt_events.csv",
                    "*GCMT*.csv",
                ]
                
                df = None
                for pattern in possible_patterns:
                    files = list(catalogs_dir.glob(pattern))
                    if files:
                        # 选择最新的文件
                        latest_file = sorted(files, key=lambda x: x.stat().st_mtime)[-1]
                        try:
                            df = pd.read_csv(latest_file)
                            self.logger.info(f"   找到数据文件: {latest_file.name}")
                            break
                        except Exception as e:
                            self.logger.debug(f"   读取失败: {e}")
                            continue
                
                # 也检查 5_Visualization/data_EastAsia/gcmt 目录
                if df is None:
                    gcmt_dir = self.base_config.dirs['project_root'] / '5_Visualization' / 'data_EastAsia' / 'gcmt'
                    if gcmt_dir.exists():
                        for f in gcmt_dir.glob("*.txt"):
                            try:
                                df = pd.read_csv(f, sep='\t')
                                self.logger.info(f"   找到数据文件: {f.name}")
                                break
                            except:
                                continue
                
                if df is None:
                    self.logger.warning("❌ 未找到 GCMT 数据文件")
                    return pd.DataFrame()
            
            self.logger.info(f"   原始数据: {len(df)} 条记录, 列: {list(df.columns)[:10]}...")
            
            # 处理震级列 - GCMT 格式通常有 Magnitude1_mb 和 Magnitude2_Ms
            if 'MainMagnitude' not in df.columns:
                # 尝试从 mb 和 Ms 计算主震级
                if 'Magnitude1_mb' in df.columns and 'Magnitude2_Ms' in df.columns:
                    # 优先使用 Ms，如果没有则使用 mb
                    df['MainMagnitude'] = df['Magnitude2_Ms'].fillna(df['Magnitude1_mb'])
                    self.logger.info("   使用 Ms/mb 作为 MainMagnitude")
                elif 'Magnitude1_mb' in df.columns:
                    df['MainMagnitude'] = df['Magnitude1_mb']
                elif 'Magnitude2_Ms' in df.columns:
                    df['MainMagnitude'] = df['Magnitude2_Ms']
                elif 'Mw' in df.columns:
                    df['MainMagnitude'] = df['Mw']
                elif 'magnitude' in df.columns:
                    df['MainMagnitude'] = df['magnitude']
                else:
                    self.logger.warning("   未找到震级列")
                    return pd.DataFrame()
            
            # 筛选数据
            # 震级筛选
            df = df[df['MainMagnitude'] >= min_magnitude].copy()
            self.logger.info(f"   震级筛选后: {len(df)} 条")
            
            # 深度筛选
            if 'Depth' in df.columns:
                df = df[(df['Depth'] >= depth_range[0]) & (df['Depth'] <= depth_range[1])]
                self.logger.info(f"   深度筛选后: {len(df)} 条")
            
            # 区域筛选
            region = self.base_config.region
            df = df[
                (df['Longitude'] >= region['lon_min']) & 
                (df['Longitude'] <= region['lon_max']) &
                (df['Latitude'] >= region['lat_min']) & 
                (df['Latitude'] <= region['lat_max'])
            ]
            self.logger.info(f"   区域筛选后: {len(df)} 条")
            
            # 检查震源机制解列
            focal_cols = ['First_Nodal_Strike', 'First_Nodal_Dip', 'First_Nodal_Rake']
            has_focal = all(col in df.columns for col in focal_cols)
            if has_focal:
                focal_valid = df[focal_cols].notna().all(axis=1).sum()
                self.logger.info(f"   有震源机制解的事件: {focal_valid} 条")
            else:
                self.logger.warning(f"   ⚠️ 数据中没有震源机制解列")
            
            # 按震级排序并限制数量
            df = df.sort_values('MainMagnitude', ascending=False).head(max_events)
            
            self.logger.info(f"✅ 最终筛选后事件数: {len(df)}")
            return df
            
        except Exception as e:
            self.logger.error(f"GCMT 数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def plot_permanent_stations_with_focal_mechanisms(self,
                                                      gcmt_file: Optional[str] = None,
                                                      min_magnitude: float = None,
                                                      max_events: int = None,
                                                      save_file: str = "5-4-stations_with_focal_mechanisms.png") -> str:
        """
        绘制永久台站分布图，叠加震源球
        
        Args:
            gcmt_file: GCMT 数据文件路径 (可选)
            min_magnitude: 显示震源球的最小震级 (默认使用配置值)
            max_events: 最大显示事件数 (默认使用配置值)
            save_file: 输出文件名
            
        Returns:
            输出文件路径
        """
        self.logger.info("🎯 开始绘制台站分布图 + 震源球叠加...")
        
        # 获取配置
        focal_config = self.config.focal_mechanism
        min_magnitude = min_magnitude or focal_config['min_magnitude']
        max_events = max_events or focal_config['max_events']
        
        try:
            if self.filtered_permanent_stations is None or len(self.filtered_permanent_stations) == 0:
                self.logger.warning("没有永久台站数据")
                return ""
            
            # 加载 GCMT 数据
            gcmt_data = self.load_gcmt_data(
                gcmt_file=gcmt_file,
                min_magnitude=min_magnitude,
                depth_range=focal_config['depth_range'],
                max_events=max_events
            )
            
            # 创建图形
            fig = pygmt.Figure()
            
            # 绘制基础地图
            map_config = self.config.map
            fig.coast(
                resolution=map_config['resolution'],
                land=map_config['land_color'],
                water=map_config['water_color'],
                shorelines=map_config['shoreline_pen'],
                area_thresh=map_config['area_threshold'],
                region=self.region,
                projection=self.config.visualization['map_projection'],
                frame=["a10f1", f"WSen+t'Permanent Stations with Focal Mechanisms (M >= {min_magnitude})'"],
                borders=[map_config['border_pen']], 
            )
            
            # 创建深度色标文件
            depth_cpt = self.output_dir / "focal_depth_temp.cpt"
            pygmt.makecpt(
                cmap=focal_config['depth_cmap'],
                series=[focal_config['depth_range'][0], focal_config['depth_range'][1], 50],
                output=str(depth_cpt)
            )
            
            # ===== 先绘制台站（底层）=====
            self.logger.info("   绑制台站...")
            colors = self.colors_combined
            station_counts = self.filtered_permanent_stations['Network'].value_counts().sort_values(ascending=False)
            color_map = {}
            for i, network in enumerate(station_counts.index):
                color_map[network] = colors[i % len(colors)]
            
            marker_config = self.config.markers['permanent_stations']
            for network, color in color_map.items():
                network_stations = self.filtered_permanent_stations[
                    self.filtered_permanent_stations['Network'] == network
                ]
                
                if len(network_stations) > 0:
                    fig.plot(
                        x=network_stations['Longitude'],
                        y=network_stations['Latitude'],
                        style=marker_config['symbol'],
                        pen=marker_config['pen'],
                        fill=color,
                    )
            
            # ===== 再绘制震源球（顶层）=====
            success_count = 0
            fallback_count = 0
            scale_config = focal_config['scale']
            style_config = focal_config['style']
            
            if len(gcmt_data) > 0:
                self.logger.info(f"   绑制 {len(gcmt_data)} 个震源球...")
                
                for idx, (_, event) in enumerate(gcmt_data.iterrows()):
                    try:
                        # 检查是否有震源机制解参数
                        focal_cols = ['First_Nodal_Strike', 'First_Nodal_Dip', 'First_Nodal_Rake']
                        has_focal = all(col in event.index for col in focal_cols)
                        
                        if has_focal:
                            strike = event.get('First_Nodal_Strike')
                            dip = event.get('First_Nodal_Dip')
                            rake = event.get('First_Nodal_Rake')
                            
                            # 检查值是否有效
                            if pd.notna(strike) and pd.notna(dip) and pd.notna(rake):
                                # 计算震源球大小 (增大基础比例)
                                mag = event['MainMagnitude']
                                scale_factor = 0.15 + (mag - min_magnitude) * 0.08  # 增大震源球
                                scale_factor = max(0.25, min(0.6, scale_factor))
                                
                                # 震源机制参数
                                focal_mechanism = {
                                    "strike": float(strike),
                                    "dip": float(dip),
                                    "rake": float(rake),
                                    "magnitude": float(mag),
                                    "depth": float(event.get('Depth', 10)),
                                }
                                
                                # 绘制震源球
                                fig.meca(
                                    spec=focal_mechanism,
                                    scale=f"{scale_factor:.2f}c",
                                    longitude=float(event['Longitude']),
                                    latitude=float(event['Latitude']),
                                    depth=float(event.get('Depth', 10)),
                                    cmap=str(depth_cpt),
                                    extensionfill=style_config['extensionfill'],
                                    pen=style_config['pen'],
                                )
                                success_count += 1
                                
                                if idx < 3:  # 打印前几个用于调试
                                    self.logger.info(f"      事件 {idx+1}: M{mag:.1f}, ({event['Latitude']:.2f}, {event['Longitude']:.2f}), strike={strike:.0f}")
                            else:
                                # 值无效，使用圆圈
                                mag = event['MainMagnitude']
                                size = 0.3 + (mag - min_magnitude) * 0.05
                                
                                fig.plot(
                                    x=float(event['Longitude']),
                                    y=float(event['Latitude']),
                                    style=f"{style_config['fallback_symbol']}{size:.2f}c",
                                    fill=style_config['fallback_fill'],
                                    pen=style_config['fallback_pen']
                                )
                                fallback_count += 1
                        else:
                            # 没有震源机制解列
                            mag = event['MainMagnitude']
                            size = 0.3 + (mag - min_magnitude) * 0.05
                            
                            fig.plot(
                                x=float(event['Longitude']),
                                y=float(event['Latitude']),
                                style=f"{style_config['fallback_symbol']}{size:.2f}c",
                                fill=style_config['fallback_fill'],
                                pen=style_config['fallback_pen']
                            )
                            fallback_count += 1
                            
                    except Exception as e:
                        self.logger.warning(f"   震源球绑制失败 (事件 {idx+1}): {e}")
                        continue
                
                self.logger.info(f"   ✅ 震源球: {success_count} 个, 圆圈(无机制解): {fallback_count} 个")
            
            # 添加色标 (震源深度)
            fig.colorbar(
                cmap=str(depth_cpt),
                frame=["a100f50", "x+lDepth", "y+lkm"],
                position="JBC+w8c/0.35c+h+o0c/1.5c"
            )
            
            # 删除临时文件
            if depth_cpt.exists():
                depth_cpt.unlink()
            
            # 保存图片
            output_path = self.output_dir / save_file
            viz_config = self.config.visualization
            fig.savefig(str(output_path), dpi=viz_config['dpi'], crop=True)
            
            # 同时保存PDF版本
            if 'pdf' in viz_config['figure_format']:
                pdf_path = output_path.with_suffix('.pdf')
                fig.savefig(str(pdf_path), dpi=viz_config['pdf_dpi'], crop=True)
            
            self.logger.info(f"✅ 台站+震源球分布图保存至: {output_path}")
            self.logger.info(f"   永久台站数: {len(self.filtered_permanent_stations)}")
            self.logger.info(f"   震源球数: {success_count + fallback_count}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"台站+震源球分布图绘制失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _plot_network_operation_timeline_complete(self, ax, colors: List[str]):
        """绘制完整的网络运行时间线（所有网络，颜色一致）"""
        try:
            # 获取字体配置
            font_sizes = self.config.timeline['font_sizes']
            
            # 准备网络时间数据
            network_times = []
            current_year = datetime.now().year
            
            # 获取所有网络（不限制数量）
            all_networks = self.filtered_permanent_stations['Network'].value_counts().sort_values(ascending=False)
            
            self.logger.info(f"📅 绑制所有 {len(all_networks)} 个永久台网的时间线")
            
            for network in all_networks.index:
                net_data = self.filtered_permanent_stations[
                    self.filtered_permanent_stations['Network'] == network
                ]
                
                if len(net_data) > 0 and 'StartYear' in net_data.columns:
                    min_year = net_data['StartYear'].min()
                    # 永久台站的EndDate都是NaN，所以用当前年份
                    max_year = current_year
                    
                    duration = max_year - min_year
                    
                    # 从永久台站颜色映射中获取对应颜色
                    network_color = self.permanent_network_color_map.get(network, colors[0])
                    
                    network_times.append({
                        'network': network,
                        'start': min_year,
                        'end': max_year,
                        'duration': duration,
                        'count': len(net_data),
                        'color': network_color  # 使用与分布图一致的颜色
                    })
            
            # 按开始时间排序
            network_times.sort(key=lambda x: x['start'])
            
            if len(network_times) == 0:
                ax.text(0.5, 0.5, 'No Network Timeline Data', transform=ax.transAxes, 
                       ha='center', fontsize=font_sizes['label'])
                return
            
            # 绘制时间线
            for i, net_time in enumerate(network_times):
                y_pos = i
                
                # 绘制时间条（使用与分布图一致的颜色）
                ax.barh(y_pos, net_time['end'] - net_time['start'], 
                       left=net_time['start'], height=0.6,
                       color=net_time['color'], alpha=0.8, 
                       edgecolor='black', linewidth=0.5)
                
                # 添加网络标签
                label = f"{net_time['network']} ({net_time['count']})"
                ax.text(net_time['start'] - 1, y_pos, label, 
                       ha='right', va='center', fontsize=font_sizes['network_label'])
                
                # 添加持续时间
                if net_time['duration'] > 0:
                    ax.text((net_time['start'] + net_time['end']) / 2, y_pos, 
                           f"{net_time['duration']}y", 
                           ha='center', va='center', fontsize=font_sizes['duration_label'], 
                           color='white', weight='bold')
            
            # 设置图形属性
            ax.set_ylim(-0.5, len(network_times) - 0.5)
            
            # 设置x轴范围：最小年份-10年 到 当前年份+2年
            min_start_year = min(nt['start'] for nt in network_times)
            ax.set_xlim(min_start_year - 5, current_year + 1.5)
            
            ax.set_xlabel('Year', fontsize=font_sizes['label'])
            ax.set_title(f'Complete Network Operation Timeline (All {len(network_times)} Networks)', 
                        fontsize=font_sizes['title'], fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=font_sizes['tick'])
            
            # 添加当前年份线
            ax.axvline(x=current_year, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7)
            ax.text(current_year, ax.get_ylim()[1] * 0.98, f'{current_year}',
                   ha='center', va='top', fontsize=font_sizes['annotation'], 
                   color='red', weight='bold')
            
            self.logger.info(f"✅ 完整网络时间线绑制完成: {len(network_times)} 个网络")
            
        except Exception as e:
            self.logger.warning(f"完整网络时间线绑制失败: {e}")
            ax.text(0.5, 0.5, 'Timeline Error', transform=ax.transAxes, ha='center')

    def plot_permanent_networks_timeline(self, 
                                       save_file: str = "5-4-permanent_networks_timeline.png") -> str:
        """
        绘制永久台网时间线分析图（分开为两个独立图）
        
        Returns:
            返回时间线图的路径（主图）
        """
        self.logger.info("📅 开始绘制永久台网时间线分析图（分开绘制版）...")
        
        try:
            if (self.filtered_permanent_stations is None or 
                len(self.filtered_permanent_stations) == 0 or
                'StartYear' not in self.filtered_permanent_stations.columns):
                self.logger.warning("没有永久台站时间数据")
                return ""
            
            if self.permanent_network_color_map is None:
                self.logger.warning("没有永久台站颜色映射，请先运行永久台站分布图生成")
                return ""
            
            # 分别绘制两个图
            # 1. 年度部署统计图
            deployment_path = self.plot_annual_deployment_separate(
                save_file="5-4-annual_deployment.png"
            )
            
            # 2. 网络时间线图（主图）
            timeline_path = self.plot_network_timeline_separate(
                save_file=save_file
            )
            
            return timeline_path
            
        except Exception as e:
            self.logger.error(f"永久台网时间线分析图绘制失败: {e}")
            return ""

    def plot_annual_deployment_separate(self, 
                                        save_file: str = "5-4-annual_deployment.png") -> str:
        """
        单独绘制年度永久台站部署统计图
        
        Args:
            save_file: 输出文件名
            
        Returns:
            输出文件路径
        """
        self.logger.info("📊 绑制年度永久台站部署统计图...")
        
        try:
            # 获取字体配置
            font_sizes = self.config.timeline['font_sizes']
            
            # 设置matplotlib样式
            plt.style.use('default')
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(16, 8))
            fig.suptitle('Annual Deployment of Permanent Stations', 
                        fontsize=font_sizes['suptitle'], fontweight='bold')
            
            # 颜色配置
            colors = self.config.timeline['colors']
            
            # 绘制年度部署统计
            self._plot_annual_permanent_deployment(ax, colors)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            output_path = self.output_dir / save_file
            plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
            
            # 同时保存PDF版本
            if 'pdf' in self.config.visualization['figure_format']:
                pdf_path = output_path.with_suffix('.pdf')
                plt.savefig(str(pdf_path), dpi=300, bbox_inches='tight', facecolor='white')
            
            plt.close()
            
            self.logger.info(f"✅ 年度部署统计图保存至: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"年度部署统计图绑制失败: {e}")
            plt.close()
            return ""

    def plot_network_timeline_separate(self, 
                                       save_file: str = "5-4-permanent_networks_timeline.png") -> str:
        """
        单独绘制网络运行时间线图（3:4比例）
        
        Args:
            save_file: 输出文件名
            
        Returns:
            输出文件路径
        """
        self.logger.info("📅 绑制网络运行时间线图（3:4比例）...")
        
        try:
            # 获取字体配置
            font_sizes = self.config.timeline['font_sizes']
            
            # 设置matplotlib样式
            plt.style.use('default')
            
            # 创建图形 - 3:4 比例 (宽:高)
            fig_size = self.config.timeline['timeline_figure_size']  # (12, 16)
            fig, ax = plt.subplots(figsize=fig_size)
            # fig.suptitle('Complete Network Operation Timeline', 
            #             fontsize=font_sizes['suptitle'], fontweight='bold', y=0.98)
            
            # 颜色配置
            colors = self.config.timeline['colors']
            
            # 绘制完整网络时间线
            self._plot_network_operation_timeline_complete(ax, colors)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # 保存图片
            output_path = self.output_dir / save_file
            plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
            
            # 同时保存PDF版本
            if 'pdf' in self.config.visualization['figure_format']:
                pdf_path = output_path.with_suffix('.pdf')
                plt.savefig(str(pdf_path), dpi=300, bbox_inches='tight', facecolor='white')
            
            plt.close()
            
            self.logger.info(f"✅ 网络时间线图保存至: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"网络时间线图绑制失败: {e}")
            plt.close()
            return ""

    def _plot_annual_permanent_deployment(self, ax, colors: List[str]):
        """绘制年度永久台站部署统计"""
        try:
            # 获取字体配置
            font_sizes = self.config.timeline['font_sizes']
            
            # 统计每年新增永久台站
            year_counts = self.filtered_permanent_stations['StartYear'].value_counts().sort_index()
            
            # 过滤有效年份范围
            current_year = datetime.now().year
            valid_years = year_counts[
                (year_counts.index >= 1970) & (year_counts.index <= current_year)
            ]
            
            if len(valid_years) == 0:
                ax.text(0.5, 0.5, 'No Timeline Data', transform=ax.transAxes, 
                       ha='center', fontsize=font_sizes['label'])
                return
            
            # 计算累计数量
            cumulative = valid_years.cumsum()
            
            # 创建双Y轴
            ax2 = ax.twinx()
            
            # 条形图 - 年度新增
            bars = ax.bar(valid_years.index, valid_years.values, 
                         color=colors[0], alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # 线图 - 累计总数
            ax2.plot(valid_years.index, cumulative.values, color=colors[1], 
                    linewidth=3, marker='o', markersize=5, label='Cumulative')
            
            # 设置标签
            ax.set_xlabel('Year', fontsize=font_sizes['label'])
            ax.set_ylabel('New Permanent Stations', fontsize=font_sizes['label'], color=colors[0])
            ax2.set_ylabel('Cumulative Count', fontsize=font_sizes['label'], color=colors[1])
            ax.set_title('Annual Deployment of Permanent Stations (Deduplicated & Filtered)', 
                        fontsize=font_sizes['title'], fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
            
            # 着色协调
            ax.tick_params(axis='y', labelcolor=colors[0], labelsize=font_sizes['tick'])
            ax.tick_params(axis='x', labelsize=font_sizes['tick'])
            ax2.tick_params(axis='y', labelcolor=colors[1], labelsize=font_sizes['tick'])
            
            # 添加统计信息
            total_stations = len(self.filtered_permanent_stations)
            years_span = valid_years.index.max() - valid_years.index.min() + 1
            avg_per_year = total_stations / years_span if years_span > 0 else 0
            
            info_text = f"Total: {total_stations} permanent stations\n"
            info_text += f"Average: {avg_per_year:.1f} stations/year\n"
            info_text += f"Networks: {self.filtered_permanent_stations['Network'].nunique()}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=font_sizes['info_text'], va='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
        except Exception as e:
            self.logger.warning(f"年度部署统计图绑制失败: {e}")
            ax.text(0.5, 0.5, 'Timeline Error', transform=ax.transAxes, ha='center')

    def save_permanent_stations_data(self):
        """保存永久台站数据到CSV文件"""
        try:
            if self.filtered_permanent_stations is not None and len(self.filtered_permanent_stations) > 0:
                # 保存筛选后的永久台站数据
                output_file = self.base_config.dirs['stations'] / 'EastAsia_permanent_stations_filtered.csv'
                self.filtered_permanent_stations.to_csv(output_file, index=False)
                self.logger.info(f"💾 筛选后的永久台站数据保存至: {output_file}")
                
                # 同时保存原始永久台站数据（未筛选密集台网）
                if self.permanent_stations is not None and len(self.permanent_stations) > 0:
                    output_file_orig = self.base_config.dirs['stations'] / 'EastAsia_permanent_stations_original.csv'
                    self.permanent_stations.to_csv(output_file_orig, index=False)
                    self.logger.info(f"💾 原始永久台站数据保存至: {output_file_orig}")
                
                # 同时保存去重后的所有台站数据
                if self.all_stations is not None and len(self.all_stations) > 0:
                    output_file_all = self.base_config.dirs['stations'] / 'EastAsia_all_stations_cleaned.csv'
                    self.all_stations.to_csv(output_file_all, index=False)
                    self.logger.info(f"💾 去重后的所有台站数据保存至: {output_file_all}")
        
        except Exception as e:
            self.logger.warning(f"保存永久台站数据失败: {e}")

    # ================================================================
    #  SPECFEM3D Globe STATIONS 文件导出
    # ================================================================

    def export_specfem_stations(self,
                                 output_file: Optional[str] = None,
                                 region_bounds: Optional[Dict] = None,
                                 use_filtered: bool = True) -> Path:
        """
        将永久台站导出为 SPECFEM3D Globe STATIONS 格式

        SPECFEM 格式: STATION_NAME NETWORK_NAME LATITUDE LONGITUDE ELEVATION BURIAL_DEPTH

        Args:
            output_file: 输出文件路径
            region_bounds: 可选的区域裁剪 {'lat_min','lat_max','lon_min','lon_max'}
            use_filtered: True=筛选后永久台站, False=全部永久台站

        Returns:
            输出文件路径
        """
        stations = self.filtered_permanent_stations if use_filtered else self.permanent_stations
        if stations is None or len(stations) == 0:
            raise ValueError("永久台站数据为空，请先运行 load_station_data()")

        df = stations.copy()

        # 按区域裁剪
        if region_bounds:
            df = df[(df['Latitude'] >= region_bounds['lat_min']) &
                    (df['Latitude'] <= region_bounds['lat_max']) &
                    (df['Longitude'] >= region_bounds['lon_min']) &
                    (df['Longitude'] <= region_bounds['lon_max'])]
            self.logger.info(f"🌏 区域裁剪: {len(stations)} → {len(df)} 个台站")

        if len(df) == 0:
            self.logger.warning("⚠️ 区域裁剪后无台站")

        # 生成 SPECFEM STATIONS 格式
        if output_file is None:
            output_file = self.base_config.dirs['stations'] / 'SPECFEM_STATIONS'
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            for _, row in df.iterrows():
                sta = str(row['Station']).strip()
                net = str(row['Network']).strip()
                lat = row['Latitude']
                lon = row['Longitude']
                elev = row.get('Elevation', 0.0)
                if elev is None or (isinstance(elev, float) and np.isnan(elev)):
                    elev = 0.0
                f.write(f"{sta:<8s} {net:<4s} {lat:10.4f} {lon:11.4f} {elev:8.1f} {0.0:6.1f}\n")

        self.logger.info(f"📄 SPECFEM STATIONS 导出: {len(df)} 个台站 → {output_file}")
        return output_file

    def export_specfem_stations_per_model(self,
                                           output_dir: Optional[str] = None) -> Dict[str, Path]:
        """
        为核心三模型分别生成裁剪后的 STATIONS 文件（边界取 BaseConfig.model_regions），
        并额外导出三模型交集区（BaseConfig.common_region）的 STATIONS_common。

        Returns:
            {'FWEA23': Path, 'EARA2024': Path, 'SinoScope1.0': Path, 'common': Path}
        """
        models = {
            strip_model_year(name): self.base_config.get_region_bounds(name)
            for name in self.base_config.core_models
        }

        if output_dir is None:
            output_dir = self.base_config.dirs['stations']
        output_dir = Path(output_dir)

        results = {}
        for name, bounds in models.items():
            out_file = output_dir / f'STATIONS_{name}'
            results[name] = self.export_specfem_stations(
                output_file=out_file,
                region_bounds=bounds,
                use_filtered=True
            )

        # 导出公共台站集（三模型交集区域）
        common_bounds = self.base_config.get_region_bounds('common')
        common_file = output_dir / 'STATIONS_common'
        results['common'] = self.export_specfem_stations(
            output_file=common_file,
            region_bounds=common_bounds,
            use_filtered=True
        )

        self.logger.info(f"✅ 已为 {len(models)} 个模型 + common 生成 STATIONS 文件")
        return results

    def run_complete_analysis(self, station_file: Optional[str] = None) -> Dict[str, str]:
        """
        运行完整的台站网络分析
        
        Args:
            station_file: 台站数据文件路径
            
        Returns:
            生成的文件路径字典
        """
        self.logger.info("🚀 开始运行完整的台站网络分析（完善图例和颜色版）...")
        
        results = {}
        
        try:
            # 1. 加载和处理数据
            self.logger.info("📊 加载台站数据...")
            self.load_station_data(station_file)
            
            # 2. 保存永久台站数据
            self.save_permanent_stations_data()
            
            # 3. 生成所有台站分布图
            self.logger.info("🗺️ 生成所有台站分布图...")
            all_stations_map = self.plot_all_stations_distribution()
            if all_stations_map:
                results['all_stations_distribution'] = all_stations_map
            
            # 4. 生成永久台站分布图（完整图例版）
            self.logger.info("🏛️ 生成永久台站分布图（完整图例版）...")
            permanent_stations_map = self.plot_permanent_stations_distribution()
            if permanent_stations_map:
                results['permanent_stations_distribution'] = permanent_stations_map
            
            # 5. 生成永久台网时间线分析图（分开绘制）
            self.logger.info("📅 生成永久台网时间线分析图（分开绘制版）...")
            
            # 5.1 年度部署统计图
            deployment_plot = self.plot_annual_deployment_separate()
            if deployment_plot:
                results['annual_deployment'] = deployment_plot
            
            # 5.2 网络时间线图（3:4比例）
            timeline_plot = self.plot_network_timeline_separate()
            if timeline_plot:
                results['permanent_networks_timeline'] = timeline_plot
            
            # 6. 保存分析摘要
            self._save_analysis_summary(results)
            
            self.logger.info("🎉 完整台站网络分析完成！")
            self.logger.info(f"📁 所有文件保存在: {self.output_dir}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"台站网络分析失败: {e}")
            import traceback
            traceback.print_exc()
            return results

    def _save_analysis_summary(self, results: Dict[str, str]):
        """保存分析摘要（修正版）"""
        try:
            summary_file = self.output_dir / "station_analysis_summary.txt"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("EASTASIA-FWI 台站网络分析摘要（完善图例和颜色版）\n")
                f.write("=" * 60 + "\n")
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 研究区域信息
                f.write("🌏 研究区域:\n")
                f.write("-" * 30 + "\n")
                f.write(f"经度范围: {self.region[0]}° - {self.region[1]}°\n")
                f.write(f"纬度范围: {self.region[2]}° - {self.region[3]}°\n\n")
                
                # 数据统计
                f.write("📊 数据统计:\n")
                f.write("-" * 30 + "\n")
                if self.all_stations is not None:
                    f.write(f"所有台站总数（去重后）: {len(self.all_stations)}\n")
                    f.write(f"所有台站网络数: {self.all_stations['Network'].nunique()}\n")
                
                if self.permanent_stations is not None:
                    f.write(f"永久台站总数: {len(self.permanent_stations)}\n")
                    f.write(f"永久台站网络数: {self.permanent_stations['Network'].nunique()}\n")
                
                if self.filtered_permanent_stations is not None:
                    f.write(f"筛选后永久台站数: {len(self.filtered_permanent_stations)}\n")
                    f.write(f"筛选后永久台站网络数: {self.filtered_permanent_stations['Network'].nunique()}\n")
                
                # 筛选条件
                f.write(f"\n🔍 筛选条件:\n")
                f.write("-" * 30 + "\n")
                criteria = self.config.permanent_criteria
                f.write(f"永久台站条件: 运行时间>{criteria['min_operation_years']}年 且 当前仍活跃\n")
                
                dense_criteria = self.config.dense_network_criteria
                f.write(f"密集台网筛选: {dense_criteria['distance_threshold_degrees']}度范围内超过{dense_criteria['min_stations_threshold']}个台站的台网被排除\n")
                
                excluded_networks = self.config.excluded_networks
                f.write(f"排除的台网类型: {list(excluded_networks)}\n")
                
                dedup_config = self.config.deduplication
                f.write(f"去重方法: {dedup_config['method']}，基于StationID\n")
                f.write(f"Provider优先级: {dedup_config['priority_providers'][:5]}...\n")
                
                # 生成的文件
                f.write(f"\n📁 生成的文件:\n")
                f.write("-" * 30 + "\n")
                for key, path in results.items():
                    f.write(f"{key}: {Path(path).name}\n")
                
                # 网络统计（前10）
                if self.network_stats is not None and len(self.network_stats) > 0:
                    f.write(f"\n📈 主要网络统计 (前10):\n")
                    f.write("-" * 30 + "\n")
                    for i, (network, stats) in enumerate(self.network_stats.head(10).iterrows()):
                        f.write(f"{i+1:2d}. {network}: {stats['StationCount']} 个台站\n")
            
            self.logger.info(f"📄 分析摘要保存至: {summary_file}")
            
        except Exception as e:
            self.logger.warning(f"保存分析摘要失败: {e}")


def find_station_data_file() -> Optional[str]:
    """查找台站数据文件"""
    try:
        base_config = BaseConfig()
        stations_dir = base_config.dirs['stations']
        
        # 支持多种文件格式和命名
        patterns = [
            f"{base_config.region['name']}_stations.csv",
            "EastAsia_stations.csv",
            "*_stations.csv",
            "stations.csv"
        ]
        
        for pattern in patterns:
            files = list(stations_dir.glob(pattern))
            if files:
                # 返回最新的文件
                latest_file = sorted(files, key=lambda x: x.stat().st_mtime)[-1]
                return str(latest_file)
        
        return None
        
    except Exception as e:
        logging.error(f"查找台站数据文件失败: {e}")
        return None


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='EASTASIA-FWI 台站网络分析系统')
    parser.add_argument('--mode', choices=['analysis', 'specfem', 'both'], default='both',
                        help='工作模式: analysis=可视化分析, specfem=SPECFEM STATIONS 导出, both=两者')
    args = parser.parse_args()

    print("🎯 EASTASIA-FWI 台站网络分析与 SPECFEM STATIONS 导出")
    print("=" * 70)
    
    try:
        # 查找台站数据文件
        station_file = find_station_data_file()
        
        if not station_file:
            print("❌ 错误: 未找到台站数据文件")
            print("   请确保以下文件之一存在:")
            print("   - data/stations/EastAsia_stations.csv")
            print("   - data/stations/stations.csv")
            return
        
        print(f"📊 使用台站数据: {Path(station_file).name}")
        
        analyzer = StationNetworkAnalyzer()

        # 先加载数据（两种模式都需要）
        print("\n🔧 加载和处理数据...")
        analyzer.load_station_data(station_file)

        # SPECFEM STATIONS 导出
        if args.mode in ('specfem', 'both'):
            print("\n" + "─" * 70)
            print("📌 SPECFEM STATIONS 文件导出（三模型 + 公共集）")
            print("─" * 70)
            specfem_results = analyzer.export_specfem_stations_per_model()
            for model_name, fpath in specfem_results.items():
                print(f"   {model_name:<15s} → {fpath.name}")

        # 可视化分析
        results = {}
        if args.mode in ('analysis', 'both'):
            print("\n" + "─" * 70)
            print("📌 台站可视化分析")
            print("─" * 70)
            results = analyzer.run_complete_analysis(station_file)

            # 震源球叠加图
            focal_result = analyzer.plot_permanent_stations_with_focal_mechanisms(
                min_magnitude=7.0, max_events=50
            )
            if focal_result:
                results['stations_with_focal_mechanisms'] = focal_result

        # 显示统计摘要
        print(f"\n📊 数据统计摘要:")
        if analyzer.all_stations is not None:
            print(f"   所有台站（去重后）: {len(analyzer.all_stations)} 个")
        if analyzer.permanent_stations is not None:
            print(f"   永久台站（原始）: {len(analyzer.permanent_stations)} 个")
        if analyzer.filtered_permanent_stations is not None:
            print(f"   永久台站（筛选后）: {len(analyzer.filtered_permanent_stations)} 个")

        if results:
            print(f"\n📁 可视化保存在: {analyzer.output_dir}")
        
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断分析")
    except Exception as e:
        print(f"\n❌ 台站网络分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()