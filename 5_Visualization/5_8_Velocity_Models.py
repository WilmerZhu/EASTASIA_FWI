"""
速度模型可视化模块
对现有的速度模型进行各种可视化分析

功能特性：
- 多模型数据加载和管理
- 空间覆盖范围可视化
- 深度覆盖范围分析
- 模型层次结构树状图
- 半圆球形深度分布图
- 模型共同覆盖区域分析
- 特定模型的共同覆盖区域分析

配置管理：
- 所有配置参数集成在模块内部
- 使用BaseConfig作为基础配置
"""

import sys
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import to_rgba  
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# 地图绘制相关
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig


class VelModelVisualizationConfig:
    """速度模型可视化配置类"""
    
    def __init__(self):
        """初始化可视化配置参数"""
        # 速度模型参数映射配置
        self.param_mapping = {
            'longitude': 'longitude',
            'latitude': 'latitude',
            'depth(km)': 'depth',
            'vp(km/s)': 'vp',
            'vsv(km/s)': 'vsv',
            'vsh(km/s)': 'vsh',
            'vs_voigt(km/s)': 'vs_voigt',
            'vs(km/s)': 'vs',
            'density(kg/m3)': 'density'
        }
        
        # 参数显示名称配置
        self.param_display_names = {
            'vp': 'P-wave Velocity',
            'vsv': 'SV-wave Velocity',
            'vsh': 'SH-wave Velocity',
            'vs_voigt': 'S-wave Velocity Voigt',
            'vs': 'S-wave Velocity',
            'density': 'Density'
        }
        
        # 深度分层配置
        self.depth_layers = {
            'crust': {
                'range': (0, 50),
                'label': 'Crust (0-50 km)',
                'color': '#FF6B6B'
            },
            'upper_mantle': {
                'range': (50, 200),
                'label': 'Upper Mantle (50-200 km)',
                'color': '#4ECDC4'
            },
            'transition_zone': {
                'range': (200, 600),
                'label': 'Transition Zone (200-600 km)',
                'color': '#45B7D1'
            },
            'lower_mantle': {
                'range': (600, 1500),
                'label': 'Lower Mantle (600-1500 km)',
                'color': '#96CEB4'
            }
        }
        
        # 色标配置
        self.colormaps = {
            'vp': 'seismic',
            'vs': 'seismic',
            'vsv': 'seismic',
            'vsh': 'seismic',
            'vs_voigt': 'seismic',
            'density': 'viridis'
        }
        
        # 参数值范围配置
        self.param_ranges = {
            'vp': (3.0, 15.0),
            'vs': (1.5, 8.5),
            'vsv': (1.5, 8.5),
            'vsh': (1.5, 8.5),
            'vs_voigt': (1.5, 8.5),
            'density': (2000, 6000)
        }
        
        # 数据验证配置
        self.validation = {
            'min_data_points': 100,
            'lon_range': (-180, 180),
            'lat_range': (-90, 90),
            'depth_range': (0, 3000),
            'invalid_value_threshold': 9000,
            'nan_threshold': 0.95  # 超过95%为NaN的参数将被忽略
        }
        
        # 可视化设置
        self.visualization = {
            'figure_size': (16, 12),
            'dpi': 300,
            'font_size': 14,
            'title_font_size': 20,
            'label_font_size': 16,
            'tick_font_size': 14,
            'legend_font_size': 12,
            'legend_title_size': 14,
            'alpha': 0.7,
            'point_size': 15,
            'grid_alpha': 0.3,
            'bbox_alpha': 0.8
        }
        
        # 地图投影配置
        self.projection = {
            'central_longitude': 115.0,  # 东亚中心
            'land_color': '#f2f2f2',
            'ocean_color': '#e6f3ff',
            'coastline_color': '#666666',
            'border_color': '#888888'
        }
        
        # 共同覆盖区域配置
        self.common_coverage = {
            'enabled': True,
            'highlight_color': '#FF4444',
            'highlight_alpha': 0.3,
            'border_width': 3,
            'show_percentage': True,
            'min_overlap_models': 2  # 最少重叠模型数
        }
        
        # 特定模型对比配置（来自2_1_Model_compare.py的三个模型）
        self.compare_models = {
            'target_models': [
                '2022_SinoScope1.0',
                '2024_EARA2024', 
                '2024_FWEA23'
            ],
            'colors': {
                '2022_SinoScope1.0': '#1f77b4',  # 蓝色
                '2024_EARA2024': '#ff7f0e',      # 橙色
                '2024_FWEA23': '#2ca02c'          # 绿色
            }
        }
        
        # 日志配置
        self.logging = {
            'level': 'INFO',
            'console_output': True
        }


class VelocityModelVisualizer:
    """速度模型可视化器"""
    
    def __init__(self):
        """初始化速度模型可视化器"""
        # 加载基础配置
        self.base_config = BaseConfig()
        
        # 加载模块配置
        self.config = VelModelVisualizationConfig()
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.VelModelVisualizer',
            self.config.logging['level']
        )
        
        # 初始化数据存储
        self.models = {}  # 存储加载的模型数据
        self.model_metadata = {}  # 存储模型元数据
        
        # 设置目录
        self._setup_directories()
        
        # 设置matplotlib样式
        self._setup_matplotlib_style()
        
        # 抑制警告
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        self.logger.info("🌍 速度模型可视化器初始化完成")
        self._print_config_summary()
    
    def _setup_directories(self):
        """设置目录结构"""
        self.dirs = {
            'models': self.base_config.dirs['models'],
            'figures': self.base_config.dirs['figures'] / '5-8_vel_models',
            'reports': self.base_config.dirs['results'] / '5-8_vel_models' / 'reports'
        }
        
        # 创建输出目录
        for dir_name in ['figures', 'reports']:
            self.dirs[dir_name].mkdir(parents=True, exist_ok=True)
    
    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 速度模型可视化配置摘要")
        print("-" * 60)
        print(f"研究区域: {self.base_config.region['name']}")
        print(f"纬度范围: {self.base_config.region['lat_min']}° ~ {self.base_config.region['lat_max']}°")
        print(f"经度范围: {self.base_config.region['lon_min']}° ~ {self.base_config.region['lon_max']}°")
        print(f"深度范围: {self.base_config.region['depth_min']} ~ {self.base_config.region['depth_max']} km")
        print(f"模型目录: {self.dirs['models']}")
        print(f"输出目录: {self.dirs['figures']}")
        print("-" * 60)
    
    def _setup_matplotlib_style(self):
        """设置matplotlib样式"""
        plt.style.use('default')
        viz_cfg = self.config.visualization
        plt.rcParams.update({
            'font.size': viz_cfg['font_size'],
            'axes.titlesize': viz_cfg['title_font_size'],
            'axes.labelsize': viz_cfg['label_font_size'],
            'xtick.labelsize': viz_cfg['tick_font_size'],
            'ytick.labelsize': viz_cfg['tick_font_size'],
            'legend.fontsize': viz_cfg['legend_font_size'],
            'legend.title_fontsize': viz_cfg['legend_title_size'],
            'figure.dpi': viz_cfg['dpi'],
            'savefig.dpi': viz_cfg['dpi'],
            'figure.figsize': viz_cfg['figure_size'],
            'axes.grid': True,
            'grid.alpha': self.config.visualization['grid_alpha']
        })
    
    def load_models(self) -> Dict[str, pd.DataFrame]:
        """加载速度模型数据"""
        try:
            models_dir = self.dirs['models']
            self.logger.info(f"📁 搜索模型文件: {models_dir}")
            
            if not models_dir.exists():
                self.logger.warning(f"模型目录不存在: {models_dir}")
                return {}
            
            # 查找CSV文件
            csv_files = list(models_dir.glob("*.csv"))
            
            if not csv_files:
                self.logger.warning("未找到CSV格式的模型文件")
                return {}
            
            self.logger.info(f"找到 {len(csv_files)} 个模型文件")
            
            loaded_models = {}
            
            for csv_file in csv_files:
                try:
                    model_name = csv_file.stem
                    self.logger.info(f"  🔄 加载模型: {model_name}")
                    
                    # 读取CSV文件
                    df = pd.read_csv(csv_file)
                    
                    # 标准化数据
                    standardized_df = self._standardize_data(df, model_name)
                    
                    if standardized_df is not None:
                        loaded_models[model_name] = standardized_df
                        
                        # 提取元数据
                        metadata = self._extract_metadata(standardized_df, model_name)
                        self.model_metadata[model_name] = metadata
                        
                        self.logger.info(f"    ✅ 成功: {len(standardized_df)} 个有效数据点")
                    else:
                        self.logger.warning(f"    ❌ 跳过: 数据标准化失败")
                        
                except Exception as e:
                    self.logger.error(f"    ❌ 加载失败: {e}")
                    continue
            
            self.models.update(loaded_models)
            self.logger.info(f"🎯 总共成功加载 {len(loaded_models)} 个模型")
            
            return loaded_models
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            traceback.print_exc()
            return {}
    
    def _standardize_data(self, df: pd.DataFrame, model_name: str) -> Optional[pd.DataFrame]:
        """标准化数据格式"""
        try:
            df_copy = df.copy()
            
            # 应用列名映射
            for old_name, new_name in self.config.param_mapping.items():
                if old_name in df_copy.columns and old_name != new_name:
                    df_copy = df_copy.rename(columns={old_name: new_name})
            
            # 检查必要列
            required_cols = ['longitude', 'latitude', 'depth']
            missing_cols = [col for col in required_cols if col not in df_copy.columns]
            
            if missing_cols:
                self.logger.warning(f"    模型 {model_name} 缺少必要列: {missing_cols}")
                return None
            
            # 数值类型转换
            numeric_cols = ['longitude', 'latitude', 'depth'] + [
                col for col in df_copy.columns 
                if col in ['vp', 'vs', 'vsv', 'vsh', 'vs_voigt', 'density']
            ]
            
            for col in numeric_cols:
                if col in df_copy.columns:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            
            # 数据验证
            valid_coords = df_copy.dropna(subset=['longitude', 'latitude', 'depth'])
            min_points = self.config.validation['min_data_points']
            
            if len(valid_coords) < min_points:
                self.logger.warning(f"    模型 {model_name} 有效数据点过少: {len(valid_coords)} < {min_points}")
                return None
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"数据标准化失败: {e}")
            return None
    
    def _get_available_params(self, df: pd.DataFrame) -> List[str]:
        """获取模型中可用的参数"""
        param_cols = ['vp', 'vs', 'vsv', 'vsh', 'vs_voigt', 'density']
        available = []
        
        nan_threshold = self.config.validation['nan_threshold']
        
        for param in param_cols:
            if param in df.columns:
                # 检查非NaN值的比例
                valid_ratio = 1 - df[param].isna().sum() / len(df)
                if valid_ratio > (1 - nan_threshold):
                    available.append(param)
        
        return available
    
    def _extract_metadata(self, df: pd.DataFrame, model_name: str) -> Dict:
        """提取模型元数据"""
        # 空间范围
        bounds = {
            'longitude': (float(df['longitude'].min()), float(df['longitude'].max())),
            'latitude': (float(df['latitude'].min()), float(df['latitude'].max())),
            'depth': (float(df['depth'].min()), float(df['depth'].max()))
        }
        
        # 分辨率估计
        unique_lons = int(df['longitude'].nunique())
        unique_lats = int(df['latitude'].nunique())
        unique_depths = int(df['depth'].nunique())
        
        # 参数可用性和统计
        available_params = self._get_available_params(df)
        param_stats = {}
        
        for param in available_params:
            if param in df.columns:
                valid_data = df[param].dropna()
                if len(valid_data) > 0:
                    param_stats[param] = {
                        'min': float(valid_data.min()),
                        'max': float(valid_data.max()),
                        'mean': float(valid_data.mean()),
                        'std': float(valid_data.std()),
                        'count': int(len(valid_data))
                    }
        
        metadata = {
            'model_name': model_name,
            'point_count': int(len(df)),
            'bounds': bounds,
            'unique_lons': unique_lons,
            'unique_lats': unique_lats,
            'unique_depths': unique_depths,
            'available_params': available_params,
            'param_stats': param_stats
        }
        
        return metadata
    
    def find_common_coverage_region(self, model_names: Optional[List[str]] = None) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        计算指定模型的共同覆盖区域
        
        Args:
            model_names: 要计算的模型名称列表，如果为None则使用所有模型
            
        Returns:
            共同覆盖区域的边界字典，如果不存在则返回None
        """
        # 确定要使用的模型
        if model_names is None:
            target_metadata = self.model_metadata
        else:
            target_metadata = {name: self.model_metadata[name] 
                             for name in model_names 
                             if name in self.model_metadata}
        
        if len(target_metadata) < self.config.common_coverage['min_overlap_models']:
            self.logger.warning(f"模型数量少于最小重叠要求: {len(target_metadata)}")
            return None
        
        try:
            # 获取所有模型的边界
            all_bounds = [meta['bounds'] for meta in target_metadata.values()]
            
            # 计算共同区域
            common_lon_min = max(bounds['longitude'][0] for bounds in all_bounds)
            common_lon_max = min(bounds['longitude'][1] for bounds in all_bounds)
            common_lat_min = max(bounds['latitude'][0] for bounds in all_bounds)
            common_lat_max = min(bounds['latitude'][1] for bounds in all_bounds)
            common_depth_min = max(bounds['depth'][0] for bounds in all_bounds)
            common_depth_max = min(bounds['depth'][1] for bounds in all_bounds)
            
            # 检查是否存在有效的共同区域
            if (common_lon_min >= common_lon_max or 
                common_lat_min >= common_lat_max or 
                common_depth_min >= common_depth_max):
                self.logger.warning("模型之间没有共同覆盖区域")
                return None
            
            common_region = {
                'longitude': (common_lon_min, common_lon_max),
                'latitude': (common_lat_min, common_lat_max),
                'depth': (common_depth_min, common_depth_max)
            }
            
            # 计算覆盖百分比
            region_area = (common_lon_max - common_lon_min) * (common_lat_max - common_lat_min)
            total_area = (self.base_config.region['lon_max'] - self.base_config.region['lon_min']) * \
                        (self.base_config.region['lat_max'] - self.base_config.region['lat_min'])
            coverage_percent = (region_area / total_area) * 100 if total_area > 0 else 0
            
            self.logger.info(f"✅ 找到共同覆盖区域:")
            self.logger.info(f"   经度: {common_lon_min:.2f}° ~ {common_lon_max:.2f}°")
            self.logger.info(f"   纬度: {common_lat_min:.2f}° ~ {common_lat_max:.2f}°")
            self.logger.info(f"   深度: {common_depth_min:.0f} ~ {common_depth_max:.0f} km")
            self.logger.info(f"   覆盖率: {coverage_percent:.1f}%")
            
            common_region['coverage_percent'] = coverage_percent
            
            return common_region
            
        except Exception as e:
            self.logger.error(f"计算共同覆盖区域失败: {e}")
            traceback.print_exc()
            return None
    
    def plot_three_models_common_coverage(self, output_file: Optional[str] = None):
        """
        绘制2_1代码中三个特定模型的共同覆盖区域图
        仅显示：2022_SinoScope1.0, 2024_EARA2024, 2024_FWEA23
        """
        # 获取目标模型列表
        target_models = self.config.compare_models['target_models']
        
        # 检查模型是否已加载
        available_models = [name for name in target_models if name in self.model_metadata]
        
        if len(available_models) < 2:
            self.logger.warning(f"未找到足够的目标模型。需要: {target_models}, 找到: {available_models}")
            return
        
        if len(available_models) < len(target_models):
            missing = set(target_models) - set(available_models)
            self.logger.warning(f"部分模型未找到: {missing}")
        
        try:
            self.logger.info(f"🗺️ 绘制三个特定模型的共同覆盖区域图...")
            self.logger.info(f"   目标模型: {', '.join(available_models)}")
            
            # 计算这三个模型的共同覆盖区域
            common_region = self.find_common_coverage_region(available_models)
            
            # 读取研究区域范围
            region = self.base_config.region
            lon_range = [region['lon_min'], region['lon_max']]
            lat_range = [region['lat_min'], region['lat_max']]
            
            # 扩大显示范围
            plot_lon_range = [lon_range[0] - 5, lon_range[1] + 5]
            plot_lat_range = [lat_range[0] - 5, lat_range[1] + 5]
            
            # 设置图形
            plt.figure(figsize=(15, 10), dpi=300)
            
            # 墨卡托投影
            ax = plt.axes(projection=ccrs.Mercator(
                central_longitude=np.mean(plot_lon_range),
                min_latitude=plot_lat_range[0],
                max_latitude=plot_lat_range[1]
            ))
            
            # 设置地图范围
            ax.set_extent(plot_lon_range + plot_lat_range, crs=ccrs.PlateCarree())
            
            # 添加地图特征
            ax.add_feature(cfeature.LAND, facecolor=self.config.projection['land_color'])
            ax.add_feature(cfeature.OCEAN, facecolor=self.config.projection['ocean_color'])
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), 
                          linewidth=0.8, edgecolor=self.config.projection['coastline_color'])
            ax.add_feature(cfeature.BORDERS.with_scale('50m'), 
                          linestyle=':', linewidth=0.6, edgecolor=self.config.projection['border_color'])
            ax.add_feature(cfeature.LAKES.with_scale('50m'), 
                          facecolor=self.config.projection['ocean_color'], 
                          edgecolor=self.config.projection['coastline_color'], linewidth=0.5)
            
            # 网格线
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size': 10}
            
            # 准备图例
            custom_handles = []
            custom_labels = []
            
            # 添加研究区域
            study_handle = mpatches.Rectangle((0, 0), 1, 1, facecolor='none', 
                                             edgecolor='red', linewidth=2.5)
            custom_handles.append(study_handle)
            custom_labels.append('Study Area')
            
            study_area = plt.Rectangle(
                (lon_range[0], lat_range[0]),
                lon_range[1] - lon_range[0],
                lat_range[1] - lat_range[0],
                transform=ccrs.PlateCarree(),
                edgecolor='red',
                facecolor='none',
                linewidth=2.5,
                linestyle='-'
            )
            ax.add_patch(study_area)
            
            # 获取颜色配置
            model_colors = self.config.compare_models['colors']
            
            # 按模型名称排序（保持一致性）
            sorted_models = sorted([(name, self.model_metadata[name]) 
                                   for name in available_models], 
                                  key=lambda x: x[0])
            
            # 绘制每个模型区域
            for i, (model_name, meta) in enumerate(sorted_models):
                if 'bounds' not in meta:
                    continue
                
                bounds = meta['bounds']
                if 'longitude' not in bounds or 'latitude' not in bounds:
                    continue
                
                lon_min, lon_max = bounds['longitude']
                lat_min, lat_max = bounds['latitude']
                
                # 获取模型颜色
                color = model_colors.get(model_name, '#1f77b4')
                color_rgb = to_rgba(color)  # 修改这里
                
                # 创建矩形
                rect_x = [lon_min, lon_max, lon_max, lon_min, lon_min]
                rect_y = [lat_min, lat_min, lat_max, lat_max, lat_min]
                
                # 绘制边界框
                poly = plt.Polygon(list(zip(rect_x, rect_y)),
                                  transform=ccrs.PlateCarree(),
                                  edgecolor=color,
                                  facecolor=(*color_rgb[:3], 0.15),
                                  linewidth=2.5)
                ax.add_patch(poly)
                
                # 添加模型编号
                plt.text(lon_min + 2.0, lat_min + 2.0,
                        f"{i+1}",
                        transform=ccrs.PlateCarree(),
                        fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, 
                                 boxstyle="circle,pad=0.4",
                                 edgecolor=color,
                                 linewidth=2),
                        ha='center', va='center',
                        zorder=100)
                
                # 构建图例标签
                model_label = f"{i+1} | {model_name}"
                depth_info = ""
                if 'bounds' in meta and 'depth' in meta['bounds']:
                    depth_min, depth_max = meta['bounds']['depth']
                    depth_info = f" | {depth_min:.0f}-{depth_max:.0f}km"
                
                params_info = ""
                if 'available_params' in meta and meta['available_params']:
                    simple_params = [p.split('(')[0] if '(' in p else p 
                                   for p in meta['available_params']]
                    params_info = f" | {', '.join(simple_params)}"
                
                full_label = f"{model_label}{depth_info}{params_info}"
                
                # 添加到图例
                patch = mpatches.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=(*color_rgb[:3], 0.6),
                    edgecolor=color,
                    linewidth=2
                )
                custom_handles.append(patch)
                custom_labels.append(full_label)
            
            # 绘制共同覆盖区域
            if common_region:
                common_lon_min, common_lon_max = common_region['longitude']
                common_lat_min, common_lat_max = common_region['latitude']
                
                # 绘制共同区域矩形
                common_rect_x = [common_lon_min, common_lon_max, common_lon_max, 
                                common_lon_min, common_lon_min]
                common_rect_y = [common_lat_min, common_lat_min, common_lat_max, 
                                common_lat_max, common_lat_min]
                
                highlight_color = self.config.common_coverage['highlight_color']
                highlight_rgb = to_rgba(highlight_color)  # 修改这里
                
                common_poly = plt.Polygon(
                    list(zip(common_rect_x, common_rect_y)),
                    transform=ccrs.PlateCarree(),
                    edgecolor=highlight_color,
                    facecolor=(*highlight_rgb[:3], self.config.common_coverage['highlight_alpha']),
                    linewidth=self.config.common_coverage['border_width'],
                    linestyle='--',
                    zorder=50
                )
                ax.add_patch(common_poly)
                
                # 添加共同区域标签
                coverage_percent = common_region.get('coverage_percent', 0)
                common_label_text = f"Common Coverage Region\n({coverage_percent:.1f}% of study area)"
                
                # 在共同区域中心添加文本
                center_lon = (common_lon_min + common_lon_max) / 2
                center_lat = (common_lat_min + common_lat_max) / 2
                
                plt.text(center_lon, center_lat, common_label_text,
                        transform=ccrs.PlateCarree(),
                        fontsize=13, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.95,
                                 boxstyle="round,pad=0.6",
                                 edgecolor=highlight_color,
                                 linewidth=2.5),
                        zorder=100)
                
                # 添加共同区域到图例
                common_handle = mpatches.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=(*highlight_rgb[:3], self.config.common_coverage['highlight_alpha']),
                    edgecolor=highlight_color,
                    linewidth=self.config.common_coverage['border_width'],
                    linestyle='--'
                )
                custom_handles.append(common_handle)
                custom_labels.append(f'Common Coverage Region ({coverage_percent:.1f}%)')
            
            # 添加标题
            plt.title('Common Coverage Region of Three Seismic Models\n(SinoScope1.0, EARA2024, FWEA23)',
                     fontsize=16, fontweight='bold', pad=20)
            
            # 创建图例
            lgd = ax.legend(
                custom_handles,
                custom_labels,
                loc='center right',
                bbox_to_anchor=(1.5, 0.5),
                title='Models & Common Region',
                title_fontsize=14,
                fontsize=11,
                frameon=True,
                framealpha=0.95,
                facecolor='#f8f8f8',
                ncol=1,
                markerscale=1.3,
                labelspacing=0.8,
                handletextpad=1.5
            )
            
            lgd._legend_box.align = "left"
            plt.subplots_adjust(left=0.05, right=0.72, top=0.93, bottom=0.05)
            
            # 保存图形
            if output_file:
                save_path = Path(output_file)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.dirs['figures'] / f"2-1_three_models_common_coverage_{timestamp}.png"
            
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight',
                       bbox_extra_artists=[lgd], pad_inches=0.5)
            self.logger.info(f"  💾 保存到: {save_path}")
            
            plt.close()
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"三模型共同覆盖区域图绘制失败: {e}")
            traceback.print_exc()
    
    def plot_spatial_coverage(self, output_file: Optional[str] = None):
        """
        绑制模型空间覆盖范围图（地图与模型列表分离版）
        
        改进：
        - 地图和模型信息表格分开绑制，避免叠加
        - 增大所有字体
        - 使用更清晰的布局
        """
        if not self.models:
            self.logger.warning("未加载任何模型")
            return
        
        try:
            self.logger.info("🗺️ 绘制模型空间覆盖范围图...")
            
            # 获取字体配置
            viz_cfg = self.config.visualization
            
            # 计算所有模型的共同覆盖区域
            common_region = self.find_common_coverage_region()
            
            # 读取研究区域范围
            region = self.base_config.region
            lon_range = [region['lon_min'], region['lon_max']]
            lat_range = [region['lat_min'], region['lat_max']]
            
            # 扩大显示范围
            plot_lon_range = [lon_range[0] - 5, lon_range[1] + 5]
            plot_lat_range = [lat_range[0] - 5, lat_range[1] + 5]
            
            # 创建地图图形
            fig = plt.figure(figsize=(16, 12), dpi=300)
            
            # 地图
            ax_map = fig.add_axes([0.05, 0.05, 0.90, 0.88], 
                                  projection=ccrs.Mercator(
                                      central_longitude=np.mean(plot_lon_range),
                                      min_latitude=plot_lat_range[0],
                                      max_latitude=plot_lat_range[1]
                                  ))
            
            # 设置地图范围
            ax_map.set_extent(plot_lon_range + plot_lat_range, crs=ccrs.PlateCarree())
            
            # 添加地图特征
            ax_map.add_feature(cfeature.LAND, facecolor=self.config.projection['land_color'])
            ax_map.add_feature(cfeature.OCEAN, facecolor=self.config.projection['ocean_color'])
            ax_map.add_feature(cfeature.COASTLINE.with_scale('50m'), 
                              linewidth=1.0, edgecolor=self.config.projection['coastline_color'])
            ax_map.add_feature(cfeature.BORDERS.with_scale('50m'), 
                              linestyle=':', linewidth=0.6, edgecolor=self.config.projection['border_color'])
            ax_map.add_feature(cfeature.LAKES.with_scale('50m'), 
                              facecolor=self.config.projection['ocean_color'], 
                              edgecolor=self.config.projection['coastline_color'], linewidth=0.5)
            
            # 网格线
            gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': viz_cfg['tick_font_size']}
            gl.ylabel_style = {'size': viz_cfg['tick_font_size']}
            
            # 添加研究区域
            study_area = plt.Rectangle(
                (lon_range[0], lat_range[0]),
                lon_range[1] - lon_range[0],
                lat_range[1] - lat_range[0],
                transform=ccrs.PlateCarree(),
                edgecolor='red',
                facecolor='none',
                linewidth=3.0,
                linestyle='-',
                zorder=90
            )
            ax_map.add_patch(study_area)
            
            # 按模型名称排序
            sorted_models = sorted(self.model_metadata.items(), key=lambda x: x[0])
            
            # 生成颜色方案
            colors_list = plt.cm.tab20(np.linspace(0, 1, 20)).tolist() + \
                         plt.cm.tab20b(np.linspace(0, 1, 20)).tolist()
            model_colors = np.array(colors_list[:len(sorted_models)])
            
            # 绘制每个模型区域
            for i, (model_name, meta) in enumerate(sorted_models):
                if 'bounds' not in meta:
                    continue
                
                bounds = meta['bounds']
                if 'longitude' not in bounds or 'latitude' not in bounds:
                    continue
                
                lon_min, lon_max = bounds['longitude']
                lat_min, lat_max = bounds['latitude']
                
                # 创建矩形
                rect_x = [lon_min, lon_max, lon_max, lon_min, lon_min]
                rect_y = [lat_min, lat_min, lat_max, lat_max, lat_min]
                
                # 绘制边界框
                poly = plt.Polygon(list(zip(rect_x, rect_y)),
                                  transform=ccrs.PlateCarree(),
                                  edgecolor=model_colors[i],
                                  facecolor=(*model_colors[i][:3], 0.15),
                                  linewidth=2.5)
                ax_map.add_patch(poly)
                
                # 添加模型编号
                ax_map.text(lon_min + 1.5, lat_min + 1.5,
                           f"{i+1}",
                           transform=ccrs.PlateCarree(),
                           fontsize=viz_cfg['font_size'], fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.8, 
                                    boxstyle="circle,pad=0.3",
                                    edgecolor=model_colors[i], linewidth=1.5),
                           ha='center', va='center',
                           zorder=100)
            
            # 绘制共同覆盖区域
            if common_region:
                common_lon_min, common_lon_max = common_region['longitude']
                common_lat_min, common_lat_max = common_region['latitude']
                
                common_rect_x = [common_lon_min, common_lon_max, common_lon_max, 
                                common_lon_min, common_lon_min]
                common_rect_y = [common_lat_min, common_lat_min, common_lat_max, 
                                common_lat_max, common_lat_min]
                
                highlight_color = self.config.common_coverage['highlight_color']
                highlight_rgb = to_rgba(highlight_color)
                
                common_poly = plt.Polygon(
                    list(zip(common_rect_x, common_rect_y)),
                    transform=ccrs.PlateCarree(),
                    edgecolor=highlight_color,
                    facecolor=(*highlight_rgb[:3], self.config.common_coverage['highlight_alpha']),
                    linewidth=self.config.common_coverage['border_width'],
                    linestyle='--',
                    zorder=50
                )
                ax_map.add_patch(common_poly)
                
                # 添加共同区域标签
                coverage_percent = common_region.get('coverage_percent', 0)
                center_lon = (common_lon_min + common_lon_max) / 2
                center_lat = (common_lat_min + common_lat_max) / 2
                
                ax_map.text(center_lon, center_lat, 
                           f"Common Coverage\n({coverage_percent:.1f}%)",
                           transform=ccrs.PlateCarree(),
                           fontsize=viz_cfg['font_size'], fontweight='bold',
                           ha='center', va='center',
                           bbox=dict(facecolor='white', alpha=0.9,
                                    boxstyle="round,pad=0.5",
                                    edgecolor=highlight_color,
                                    linewidth=2),
                           zorder=100)
            
            # 添加标题
            ax_map.set_title('Spatial Coverage of Seismic Velocity Models',
                            fontsize=viz_cfg['title_font_size'], fontweight='bold', pad=15)
            
            # 添加简洁图例（右侧）
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', linewidth=3, linestyle='-', label='Study Area'),
            ]
            if common_region:
                coverage_percent = common_region.get('coverage_percent', 0)
                legend_elements.append(
                    Line2D([0], [0], color=self.config.common_coverage['highlight_color'], 
                          linewidth=3, linestyle='--', label=f'Common Coverage ({coverage_percent:.1f}%)')
                )
            
            ax_map.legend(handles=legend_elements, loc='upper right', fontsize=viz_cfg['legend_font_size'])
            
            # 保存图形
            if output_file:
                save_path = Path(output_file)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.dirs['figures'] / f"5-8_spatial_coverage_{timestamp}.png"
            
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight', pad_inches=0.3)
            self.logger.info(f"  💾 保存到: {save_path}")
            
            plt.close()
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"空间覆盖图绑制失败: {e}")
            traceback.print_exc()
    
    def plot_model_info_table(self, output_file: Optional[str] = None):
        """
        绑制模型信息汇总表格（独立图）
        
        包含信息：编号、模型名、年份、数据点数、空间范围、深度范围、参数
        """
        if not self.model_metadata:
            self.logger.warning("未加载任何模型元数据")
            return
        
        try:
            self.logger.info("📋 绑制模型信息汇总表格...")
            
            # 获取字体配置
            viz_cfg = self.config.visualization
            
            # 按模型名称排序
            sorted_models = sorted(self.model_metadata.items(), key=lambda x: x[0])
            
            # 生成颜色方案
            colors_list = plt.cm.tab20(np.linspace(0, 1, 20)).tolist() + \
                         plt.cm.tab20b(np.linspace(0, 1, 20)).tolist()
            model_colors = colors_list[:len(sorted_models)]
            
            # 准备表格数据
            table_headers = ['No.', 'Model Name', 'Year', 'Data Points', 
                           'Lon Range (°E)', 'Lat Range (°N)', 'Depth (km)', 'Parameters']
            table_rows = []
            cell_colors = []
            
            for i, (model_name, meta) in enumerate(sorted_models):
                # 提取年份
                year = model_name.split('_')[0] if '_' in model_name else '-'
                
                # 数据点数
                point_count = f"{meta.get('point_count', 0):,}"
                
                # 空间范围
                bounds = meta.get('bounds', {})
                lon_range = "-"
                lat_range = "-"
                depth_range = "-"
                
                if 'longitude' in bounds:
                    lon_min, lon_max = bounds['longitude']
                    lon_range = f"{lon_min:.1f} ~ {lon_max:.1f}"
                
                if 'latitude' in bounds:
                    lat_min, lat_max = bounds['latitude']
                    lat_range = f"{lat_min:.1f} ~ {lat_max:.1f}"
                
                if 'depth' in bounds:
                    d_min, d_max = bounds['depth']
                    depth_range = f"{d_min:.0f} ~ {d_max:.0f}"
                
                # 参数
                params = meta.get('available_params', [])
                params_str = ', '.join(params) if params else '-'
                # 截断过长的参数字符串
                if len(params_str) > 30:
                    params_str = params_str[:27] + '...'
                
                table_rows.append([
                    str(i + 1),
                    model_name,
                    year,
                    point_count,
                    lon_range,
                    lat_range,
                    depth_range,
                    params_str
                ])
                
                # 使用模型颜色作为行背景
                row_color = [(*model_colors[i][:3], 0.15)] * len(table_headers)
                cell_colors.append(row_color)
            
            # 计算图形高度（根据行数动态调整）
            n_rows = len(table_rows)
            fig_height = max(8, 2 + n_rows * 0.45)
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(18, fig_height), dpi=300)
            ax.axis('off')
            
            # 绘制表格
            table = ax.table(
                cellText=table_rows,
                colLabels=table_headers,
                cellLoc='center',
                loc='center',
                colWidths=[0.04, 0.18, 0.05, 0.10, 0.12, 0.12, 0.12, 0.22],
                cellColours=cell_colors
            )
            
            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.0, 2.0)
            
            # 设置表头样式
            for j in range(len(table_headers)):
                cell = table[(0, j)]
                cell.set_facecolor('#2E75B6')
                cell.set_text_props(color='white', fontweight='bold', fontsize=12)
                cell.set_height(0.06)
            
            # 设置数据行样式
            for i in range(1, n_rows + 1):
                for j in range(len(table_headers)):
                    cell = table[(i, j)]
                    cell.set_height(0.045)
                    # 第一列（编号）加粗
                    if j == 0:
                        cell.set_text_props(fontweight='bold')
            
            # 添加标题
            ax.set_title('Seismic Velocity Model Database Summary',
                        fontsize=viz_cfg['title_font_size'], fontweight='bold', 
                        pad=20, y=1.02)
            
            # 添加统计信息
            total_points = sum(meta.get('point_count', 0) for meta in self.model_metadata.values())
            info_text = f"Total: {len(sorted_models)} models | {total_points:,} data points"
            ax.text(0.5, -0.02, info_text, transform=ax.transAxes,
                   fontsize=viz_cfg['font_size'], ha='center', va='top',
                   style='italic', color='#666666')
            
            plt.tight_layout()
            
            # 保存图形
            if output_file:
                save_path = Path(output_file)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.dirs['figures'] / f"5-8_model_info_table_{timestamp}.png"
            
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight', pad_inches=0.3)
            self.logger.info(f"  💾 模型信息表格保存到: {save_path}")
            
            plt.close()
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"模型信息表格绑制失败: {e}")
            traceback.print_exc()
    
    def plot_depth_coverage(self, output_file: Optional[str] = None):
        """绘制模型深度覆盖范围条形图"""
        if not self.models:
            self.logger.warning("未加载任何模型")
            return
        
        try:
            self.logger.info("🏔️ 绘制模型深度覆盖范围图...")
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(12, 8), dpi=self.config.visualization['dpi'])
            
            # 按模型名称排序
            sorted_models = sorted(self.model_metadata.items(), key=lambda x: x[0])
            
            # 准备数据
            model_names_short = []
            colors = cm.tab20(np.linspace(0, 1, len(sorted_models)))
            
            for i, (name, metadata) in enumerate(sorted_models):
                model_names_short.append(name[:25] + ('...' if len(name) > 25 else ''))
                depth_min, depth_max = metadata['bounds']['depth']
                
                # 绘制深度范围条
                ax.barh(i, depth_max - depth_min, left=depth_min,
                        color=colors[i], alpha=self.config.visualization['alpha'],
                        height=0.7)
                
                # 添加深度范围标签
                ax.text(depth_max + max(50, (depth_max - depth_min) * 0.05), i,
                        f"{depth_min:.0f}-{depth_max:.0f} km",
                        va='center', fontsize=self.config.visualization['font_size'] - 1)
            
            # 设置坐标轴
            ax.set_yticks(range(len(model_names_short)))
            ax.set_yticklabels(model_names_short,
                               fontsize=self.config.visualization['font_size'] - 1)
            ax.set_xlabel('Depth (km)', fontsize=self.config.visualization['label_font_size'])
            ax.set_title('Seismic Models Depth Coverage Range',
                         fontsize=self.config.visualization['title_font_size'], fontweight='bold')
            ax.grid(True, alpha=self.config.visualization['grid_alpha'], axis='x')
            ax.invert_yaxis()
            
            # 添加深度层标识线
            depth_boundaries = [50, 200, 660, 1000]
            depth_labels = ['Moho', 'Upper Mantle', 'Transition Zone', 'Lower Mantle']
            
            for depth, label in zip(depth_boundaries, depth_labels):
                ax.axvline(x=depth, color='gray', linestyle='--', alpha=0.6, linewidth=1)
                ax.text(depth, len(model_names_short), label,
                        rotation=90, ha='right', va='bottom',
                        fontsize=self.config.visualization['font_size'] - 2,
                        alpha=0.7)
            
            plt.tight_layout()
            
            # 保存图形
            if output_file:
                save_path = Path(output_file)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.dirs['figures'] / f"5-8_depth_coverage_{timestamp}.png"
            
            plt.savefig(str(save_path), dpi=self.config.visualization['dpi'],
                       bbox_inches='tight')
            self.logger.info(f"  💾 保存到: {save_path}")
            
            plt.close()
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"深度覆盖图绘制失败: {e}")
            traceback.print_exc()
    
    def plot_semicircular_depth_distribution(self, output_file: Optional[str] = None):
        """绘制150度扇形深度分布图"""
        if not self.models:
            self.logger.warning("未加载任何模型")
            return
        
        try:
            self.logger.info("🌐 绘制扇形深度分布图...")
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(18, 12), dpi=300, facecolor='white')
            
            # 按模型名称排序
            sorted_models = sorted(self.model_metadata.items(), key=lambda x: x[0])
            
            # 使用与空间分布图一致的颜色方案
            colors_list = plt.cm.tab20(np.linspace(0, 1, 20)).tolist() + \
                         plt.cm.tab20b(np.linspace(0, 1, 20)).tolist()
            model_colors = np.array(colors_list[:len(sorted_models)])
            
            # 设置坐标系
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-0.4, 1.6)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # 地球物理参数
            cmb_depth_km = 2900  # 核幔边界深度
            
            # 扇形角度范围 - 150度
            center_angle = np.pi / 2
            half_span = np.radians(75)
            start_angle = center_angle - half_span
            end_angle = center_angle + half_span
            
            # 深度映射：地核区域占1/2半径
            core_radius_ratio = 0.5
            
            def depth_to_radius(depth):
                """将深度转换为半径位置"""
                depth = min(depth, cmb_depth_km)
                return 1.0 - (depth / cmb_depth_km) * (1.0 - core_radius_ratio)
            
            # 绘制重要深度圈层
            important_depths = [40, 100, 410, 660, 1000, cmb_depth_km]
            
            for depth in important_depths:
                radius = depth_to_radius(depth)
                theta = np.linspace(start_angle, end_angle, 100)
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                
                if depth == cmb_depth_km:
                    line_color, line_width, line_alpha = '#8B4513', 2.5, 0.9
                elif depth in [410, 660]:
                    line_color, line_width, line_alpha = '#B8860B', 2.0, 0.8
                elif depth == 40:
                    line_color, line_width, line_alpha = '#CD853F', 1.8, 0.8
                else:
                    line_color, line_width, line_alpha = '#D2B48C', 1.5, 0.7
                
                ax.plot(x, y, color=line_color, linewidth=line_width, alpha=line_alpha)
            
            # 绘制深度刻度
            left_scale_angle = start_angle + np.radians(5)
            right_scale_angle = end_angle - np.radians(5)
            
            scale_start_radius = depth_to_radius(0)
            scale_end_radius = depth_to_radius(cmb_depth_km)
            
            # 左侧刻度轴
            left_x_start = scale_start_radius * np.cos(left_scale_angle)
            left_y_start = scale_start_radius * np.sin(left_scale_angle)
            left_x_end = scale_end_radius * np.cos(left_scale_angle)
            left_y_end = scale_end_radius * np.sin(left_scale_angle)
            
            ax.plot([left_x_start, left_x_end], [left_y_start, left_y_end],
                    color='#333333', linewidth=3, alpha=0.8)
            
            # 右侧刻度轴
            right_x_start = scale_start_radius * np.cos(right_scale_angle)
            right_y_start = scale_start_radius * np.sin(right_scale_angle)
            right_x_end = scale_end_radius * np.cos(right_scale_angle)
            right_y_end = scale_end_radius * np.sin(right_scale_angle)
            
            ax.plot([right_x_start, right_x_end], [right_y_start, right_y_end],
                    color='#333333', linewidth=3, alpha=0.8)
            
            # 绘制深度刻度和标签
            for depth in important_depths:
                radius = depth_to_radius(depth)
                
                # 左侧刻度
                left_tick_x = radius * np.cos(left_scale_angle)
                left_tick_y = radius * np.sin(left_scale_angle)
                left_tick_angle = left_scale_angle - np.pi/2
                
                tick_length = 0.10 if depth in [cmb_depth_km, 410, 660] else 0.08
                
                left_tick_end_x = left_tick_x + tick_length * np.cos(left_tick_angle)
                left_tick_end_y = left_tick_y + tick_length * np.sin(left_tick_angle)
                
                ax.plot([left_tick_x, left_tick_end_x], [left_tick_y, left_tick_end_y],
                        color='#333333', linewidth=2)
                
                label_offset = 0.15
                left_label_x = left_tick_x + label_offset * np.cos(left_tick_angle)
                left_label_y = left_tick_y + label_offset * np.sin(left_tick_angle)
                
                ax.text(left_label_x, left_label_y, f'{depth:.0f} km',
                        ha='center', va='center', fontsize=8,
                        color='#666666', weight='normal',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                                 alpha=0.8, edgecolor='none'))
                
                # 右侧刻度
                right_tick_x = radius * np.cos(right_scale_angle)
                right_tick_y = radius * np.sin(right_scale_angle)
                right_tick_angle = right_scale_angle + np.pi/2
                
                right_tick_end_x = right_tick_x + tick_length * np.cos(right_tick_angle)
                right_tick_end_y = right_tick_y + tick_length * np.sin(right_tick_angle)
                
                ax.plot([right_tick_x, right_tick_end_x], [right_tick_y, right_tick_end_y],
                        color='#333333', linewidth=2)
                
                right_label_x = right_tick_x + label_offset * np.cos(right_tick_angle)
                right_label_y = right_tick_y + label_offset * np.sin(right_tick_angle)
                
                ax.text(right_label_x, right_label_y, f'{depth:.0f} km',
                        ha='center', va='center', fontsize=8,
                        color='#666666', weight='normal',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                                 alpha=0.8, edgecolor='none'))
            
            # 添加深度轴标签
            left_tick_angle = left_scale_angle - np.pi/2
            left_axis_label_x = left_x_start + 0.3 * np.cos(left_tick_angle)
            left_axis_label_y = (left_y_start + left_y_end) / 2
            
            ax.text(left_axis_label_x, left_axis_label_y, 'Depth (km)',
                    ha='center', va='center', fontsize=13,
                    color='#333333', weight='bold',
                    rotation=np.degrees(left_scale_angle) - 90)
            
            right_tick_angle = right_scale_angle + np.pi/2
            right_axis_label_x = right_x_start + 0.3 * np.cos(right_tick_angle)
            right_axis_label_y = (right_y_start + right_y_end) / 2
            
            ax.text(right_axis_label_x, right_axis_label_y, 'Depth (km)',
                    ha='center', va='center', fontsize=13,
                    color='#333333', weight='bold',
                    rotation=np.degrees(right_scale_angle) + 90)
            
            # 为每个模型绘制深度范围
            model_count = len(sorted_models)
            if model_count > 0:
                angle_margin = np.radians(8)
                available_angle_range = (end_angle - start_angle) - 2 * angle_margin
                angles = np.linspace(start_angle + angle_margin,
                                   end_angle - angle_margin,
                                   model_count)
                
                for i, (model_name, meta) in enumerate(sorted_models):
                    color = model_colors[i]
                    
                    depth_max = min(meta['bounds']['depth'][1], cmb_depth_km)
                    depth_min = meta['bounds']['depth'][0]
                    
                    radius_min = depth_to_radius(depth_min)
                    radius_max = depth_to_radius(depth_max)
                    
                    angle = angles[i]
                    
                    x_start = radius_max * np.cos(angle)
                    y_start = radius_max * np.sin(angle)
                    x_end = radius_min * np.cos(angle)
                    y_end = radius_min * np.sin(angle)
                    
                    # 根据模型平面成像面积调整线宽
                    if 'bounds' in meta:
                        lon_range = meta['bounds']['longitude'][1] - meta['bounds']['longitude'][0]
                        lat_range = meta['bounds']['latitude'][1] - meta['bounds']['latitude'][0]
                        area = lon_range * lat_range
                        
                        min_area, max_area = 100, 8000
                        min_linewidth, max_linewidth = 4, 18
                        
                        normalized_area = np.clip((area - min_area) / (max_area - min_area), 0, 1)
                        line_width = min_linewidth + (max_linewidth - min_linewidth) * normalized_area
                    else:
                        line_width = 8
                    
                    ax.plot([x_start, x_end], [y_start, y_end],
                           color=color, linewidth=line_width, alpha=0.9,
                           solid_capstyle='round')
                    
                    # 模型序号标签
                    surface_radius = depth_to_radius(0)
                    label_extension = 0.25
                    label_radius = surface_radius + label_extension
                    
                    label_x = label_radius * np.cos(angle)
                    label_y = label_radius * np.sin(angle)
                    
                    display_name = f"{i+1}"
                    
                    bbox_props = dict(
                        boxstyle="circle,pad=0.3",
                        facecolor='white',
                        edgecolor=color,
                        alpha=0.9,
                        linewidth=2
                    )
                    
                    ax.text(label_x, label_y, display_name,
                           ha='center', va='center',
                           fontsize=12, color=color, weight='bold',
                           bbox=bbox_props)
            
            # 添加标题
            ax.text(0, 1.6, 'Depth Distribution of Seismic Models in East Asia',
                   ha='center', va='center', fontsize=20, weight='bold',
                   color='#2F4F4F')
            
            plt.tight_layout()
            
            # 保存图形
            if output_file:
                save_path = Path(output_file)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.dirs['figures'] / f"5-8_semicircular_depth_distribution_{timestamp}.png"
            
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none', pad_inches=0.5)
            self.logger.info(f"  💾 保存到: {save_path}")
            
            plt.close()
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"扇形深度分布图绘制失败: {e}")
            traceback.print_exc()
    
    def create_model_hierarchy_tree(self, output_file: Optional[str] = None):
        """创建模型分类的径向树状图"""
        self.logger.info("🌳 创建模型层次结构树状图...")
        
        try:
            # 导入需要的库
            try:
                import networkx as nx
            except ImportError:
                self.logger.warning("NetworkX不可用，跳过树状图生成")
                return None
            
            # 创建有向图
            G = nx.DiGraph()
            
            # 添加根节点
            G.add_node("Seismic Models", level=0, size=3000)
            
            # 深度分类
            depth_categories = {
                "Lithosphere (<200km)": lambda meta: meta.get('bounds', {}).get('depth', [0, 0])[1] < 200,
                "Upper Mantle (200-1000km)": lambda meta: 200 <= meta.get('bounds', {}).get('depth', [0, 0])[1] <= 1000,
                "Whole Mantle (>1000km)": lambda meta: meta.get('bounds', {}).get('depth', [0, 0])[1] > 1000
            }
            
            # 参数分类
            param_categories = {
                "Vp": lambda meta: any(p == 'vp' for p in meta.get('available_params', [])),
                "Vs": lambda meta: any(p == 'vs' for p in meta.get('available_params', [])),
                "Vsv": lambda meta: any(p == 'vsv' for p in meta.get('available_params', [])),
                "Vsh": lambda meta: any(p == 'vsh' for p in meta.get('available_params', [])),
                "Vs_Voigt": lambda meta: any(p == 'vs_voigt' for p in meta.get('available_params', [])),
                "Density": lambda meta: any(p == 'density' for p in meta.get('available_params', []))
            }
            
            # 颜色映射
            depth_colors = {
                "Lithosphere (<200km)": "#2C7BB6",
                "Upper Mantle (200-1000km)": "#2077B4",
                "Whole Mantle (>1000km)": "#1F77B4"
            }
            
            param_colors = {
                "Vp": "#FF7F0E", "Vs": "#D62728", "Vsv": "#9467BD",
                "Vsh": "#8C564B", "Vs_Voigt": "#E377C2", "Density": "#7F7F7F"
            }
            
            # 添加深度分类节点
            for depth_cat in depth_categories:
                G.add_node(depth_cat, level=1, size=2000, color=depth_colors[depth_cat])
                G.add_edge("Seismic Models", depth_cat, weight=3)
            
            # 添加参数分类和模型节点
            for depth_cat in depth_categories:
                for param_cat in param_categories:
                    node_name = f"{param_cat}"
                    models_in_category = []
                    
                    for model_name, meta in self.model_metadata.items():
                        if depth_categories[depth_cat](meta) and param_categories[param_cat](meta):
                            models_in_category.append(model_name)
                    
                    if models_in_category:
                        if not G.has_node(node_name):
                            G.add_node(node_name, level=2, size=1200, color=param_colors[param_cat])
                        G.add_edge(depth_cat, node_name, weight=2)
                        
                        for model_name in models_in_category:
                            model_id = f"{model_name} ({depth_cat})"
                            G.add_node(model_id, level=3, size=600, model_name=model_name, real_name=model_name)
                            G.add_edge(node_name, model_id, weight=1)
            
            # 准备绘图
            plt.figure(figsize=(16, 16), facecolor='white')
            ax = plt.gca()
            
            # 使用shell布局
            shells = [
                ["Seismic Models"],
                [n for n, d in G.nodes(data=True) if d.get('level') == 1],
                [n for n, d in G.nodes(data=True) if d.get('level') == 2],
                [n for n, d in G.nodes(data=True) if d.get('level') == 3]
            ]
            pos = nx.shell_layout(G, shells, scale=2)
            
            # 获取不同级别的节点
            nodes_by_level = {}
            level_colors = {}
            
            for node, data in G.nodes(data=True):
                level = data.get('level', 0)
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)
                
                if level == 0:
                    level_colors[node] = '#2077B4'
                elif level == 1:
                    level_colors[node] = depth_colors.get(node, '#1F77B4')
                elif level == 2:
                    level_colors[node] = param_colors.get(node, '#FF7F0E')
                else:
                    level_colors[node] = '#2CA02C'
            
            # 绘制边
            edge_colors = []
            edge_widths = []
            
            for u, v, data in G.edges(data=True):
                if G.nodes[u]['level'] == 0:
                    edge_colors.append(level_colors[v])
                    edge_widths.append(3)
                elif G.nodes[u]['level'] == 1:
                    edge_colors.append(level_colors[v])
                    edge_widths.append(2)
                else:
                    edge_colors.append('#CCCCCC')
                    edge_widths.append(1)
            
            nx.draw_networkx_edges(
                G, pos,
                width=edge_widths,
                edge_color=edge_colors,
                alpha=0.7,
                connectionstyle="arc3,rad=0.1"
            )
            
            # 按层级绘制节点
            for level, nodes in nodes_by_level.items():
                node_colors = [level_colors[n] for n in nodes]
                node_sizes = [G.nodes[n].get('size', 300) for n in nodes]
                
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=nodes,
                    node_size=node_sizes,
                    node_color=node_colors,
                    alpha=0.8
                )
            
            # 标签布局和样式
            for node, (x, y) in pos.items():
                if G.nodes[node].get('level') == 3:
                    label = G.nodes[node].get('real_name', node)
                    label = label[:15] + '...' if len(label) > 15 else label
                    plt.text(x*1.05, y*1.05, label,
                            fontsize=8,
                            ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7,
                                     edgecolor=level_colors[node], boxstyle='round'),
                            zorder=100)
                else:
                    plt.text(x, y, node,
                            fontsize=12 if G.nodes[node].get('level') == 0 else 10,
                            ha='center', va='center',
                            color='white' if G.nodes[node].get('level') < 3 else 'black',
                            weight='bold',
                            bbox=dict(facecolor=level_colors[node], alpha=0.9, boxstyle='round'),
                            zorder=100)
            
            # 添加标题
            plt.title('Seismic Models Hierarchy Tree', fontsize=24, fontweight='bold', pad=20)
            
            # 添加图例
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2077B4',
                          markersize=15, label='Root'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=list(depth_colors.values())[0],
                          markersize=15, label='Layer'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=list(param_colors.values())[0],
                          markersize=15, label='Parameter'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2CA02C',
                          markersize=15, label='Model')
            ]
            plt.legend(handles=legend_elements, loc='lower right', fontsize=12)
            
            plt.axis('off')
            plt.tight_layout()
            
            # 保存图表
            if output_file:
                save_path = Path(output_file)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.dirs['figures'] / f"5-8_model_hierarchy_tree_{timestamp}.png"
            
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            self.logger.info(f"  💾 树状图已保存到: {save_path}")
            
            plt.close()
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"模型层次树状图创建失败: {e}")
            traceback.print_exc()
            return None
    
    def show_models_summary(self):
        """显示模型摘要信息"""
        if not self.models:
            self.logger.warning("未加载任何模型")
            return
        
        print("\n" + "="*80)
        print("🌍 EASTASIA-FWI 速度模型库摘要信息")
        print("="*80)
        
        # 基本统计
        total_points = sum(len(df) for df in self.models.values())
        
        print(f"📊 模型总数: {len(self.models)}")
        print(f"📈 数据点总数: {total_points:,}")
        print(f"🗂️ 数据目录: {self.dirs['models']}")
        
        # 区域覆盖范围
        if self.model_metadata:
            all_lons = [meta['bounds']['longitude'] for meta in self.model_metadata.values()]
            all_lats = [meta['bounds']['latitude'] for meta in self.model_metadata.values()]
            all_depths = [meta['bounds']['depth'] for meta in self.model_metadata.values()]
            
            if all_lons and all_lats and all_depths:
                lon_min = min(min(lon) for lon in all_lons)
                lon_max = max(max(lon) for lon in all_lons)
                lat_min = min(min(lat) for lat in all_lats)
                lat_max = max(max(lat) for lat in all_lats)
                depth_min = min(min(depth) for depth in all_depths)
                depth_max = max(max(depth) for depth in all_depths)
                
                print(f"\n🌐 总覆盖范围:")
                print(f"   经度: {lon_min:.2f}° ~ {lon_max:.2f}°")
                print(f"   纬度: {lat_min:.2f}° ~ {lat_max:.2f}°")
                print(f"   深度: {depth_min:.1f} ~ {depth_max:.1f} km")
        
        # 参数可用性统计
        param_availability = defaultdict(int)
        for meta in self.model_metadata.values():
            for param in meta['available_params']:
                param_availability[param] += 1
        
        if param_availability:
            print(f"\n📋 参数可用性:")
            for param, count in sorted(param_availability.items()):
                percentage = count / len(self.models) * 100
                display_name = self.config.param_display_names.get(param, param)
                print(f"   {display_name}: {count}/{len(self.models)} 模型 ({percentage:.1f}%)")
        
        # 详细模型信息
        print(f"\n📑 模型详细信息:")
        print(f"{'模型名称':<30} {'数据点':<10} {'参数':<25} {'深度范围(km)':<15}")
        print("-" * 85)
        
        sorted_models = sorted(self.model_metadata.items(), key=lambda x: x[0])
        
        for name, metadata in sorted_models:
            points = f"{metadata['point_count']:,}"
            params = ', '.join(metadata.get('available_params', []))[:23]
            if len(', '.join(metadata.get('available_params', []))) > 23:
                params += '...'
            depth_range = f"{metadata['bounds']['depth'][0]:.0f}-{metadata['bounds']['depth'][1]:.0f}"
            
            print(f"{name[:29]:<30} {points:<10} {params:<25} {depth_range:<15}")
        
        print("="*80)
    
    def generate_all_plots(self):
        """生成所有可视化图表"""
        self.logger.info("🎨 开始生成所有可视化图表...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. 空间覆盖图
            self.logger.info("  📍 生成空间覆盖图...")
            self.plot_spatial_coverage(
                str(self.dirs['figures'] / f"5-8_spatial_coverage_{timestamp}.png")
            )
            
            # 2. 模型信息汇总表格（新增）
            self.logger.info("  📋 生成模型信息汇总表格...")
            self.plot_model_info_table(
                str(self.dirs['figures'] / f"5-8_model_info_table_{timestamp}.png")
            )
            
            # 3. 三个特定模型的共同覆盖区域图
            self.logger.info("  🎯 生成三模型共同覆盖区域图...")
            self.plot_three_models_common_coverage(
                str(self.dirs['figures'] / f"5-8_three_models_common_coverage_{timestamp}.png")
            )
            
            # 4. 深度覆盖图
            self.logger.info("  🏔️ 生成深度覆盖图...")
            self.plot_depth_coverage(
                str(self.dirs['figures'] / f"5-8_depth_coverage_{timestamp}.png")
            )
            
            # 5. 半圆形深度分布图
            self.logger.info("  🌐 生成扇形深度分布图...")
            self.plot_semicircular_depth_distribution(
                str(self.dirs['figures'] / f"5-8_semicircular_depth_distribution_{timestamp}.png")
            )
            
            # 6. 模型层次结构树状图
            self.logger.info("  🌳 生成模型层次结构树状图...")
            self.create_model_hierarchy_tree(
                str(self.dirs['figures'] / f"5-8_model_hierarchy_tree_{timestamp}.png")
            )
            
            self.logger.info(f"✅ 所有可视化图表已生成完成！")
            self.logger.info(f"📁 查看图表: {self.dirs['figures']}")
            
        except Exception as e:
            self.logger.error(f"批量生成失败: {e}")


def main():
    """主函数 - 速度模型可视化"""
    print("🌍 EASTASIA-FWI 速度模型可视化系统")
    print("   对现有速度模型进行各种可视化分析")
    print("="*70)
    
    try:
        # 创建可视化器
        visualizer = VelocityModelVisualizer()
        
        # 加载模型
        print("\n📁 加载速度模型...")
        models = visualizer.load_models()
        
        if not models:
            print("⚠️ 未找到任何模型文件")
            print(f"请确保在目录下放置CSV格式的模型文件: {visualizer.dirs['models']}")
            return
        
        # 显示摘要
        visualizer.show_models_summary()
        
        # 生成所有可视化图表
        print("\n🎨 生成可视化图表...")
        visualizer.generate_all_plots()
        
        print("\n✅ 可视化完成！")
    
    except KeyboardInterrupt:
        print("\n👋 用户中断程序")
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()