"""
2_1_Model_compare.py:
速度模型对比分析模块（双区域1D对比）
================================================================

功能描述:
----------
基于标准化NetCDF数据的多维度速度模型对比分析。对齐 1_5 双区域输出，
共同区域（standardized）用于多模型公平对比，原始区域（original）用于
单模型完整可视化。

核心功能:
----------
1. ✅ 双区域数据源（对齐 1_5 设计）
   - 共同区域对比：*_standardized.nc（多模型交集区域）
   - 单独模型可视化：*_original.nc（完整模型覆盖）
2. ✅ 1D速度剖面对比（VS + VP，各向异性 VSV/VSH、VPV/VPH）
3. ✅ 水平切片对比（绝对速度 / dlnV扰动 / 两者）
4. ✅ 垂直剖面对比（经度/纬度方向，灵活绘图模式）
5. ✅ 两两模型差异图（ΔV = V_A - V_B）
6. ✅ 切片位置示意图（Cartopy地理参考）

使用方法:
----------
```python
from 2_Model_space_analysis.2_1_Model_compare import ModelComparator

comparator = ModelComparator()
comparator.load_models()
comparator.save_results()
```

配置说明:
----------
通过 ModelCompareConfig 类配置参数:
- target_models: 目标模型列表（含 standardized/original 文件路径）
- profile_params: 1D剖面参数（深度范围、空间平均）
- plot_mode: 绘图模式（absolute / perturbation / both）
- slice_params: 切片可视化参数（深度列表、色标、海岸线）

输出文件:
----------
- 2-1_velocity_comparison.png: VS+VP 综合对比
- 2-1_s_wave_anisotropy.png: S波各向异性（VSV vs VSH）
- 2-1_p_wave_anisotropy.png: P波各向异性（VPV vs VPH）
- 2-1_profile_{model}.png: 单模型完整剖面（original区域）
- 2-1_horizontal_slice_vs_{depth}km.png: 水平切片
- 2-1_vertical_profile_{n}.png: 垂直剖面
- 2-1_absolute_vs_{depth}km.png: 绝对速度并排对比
- 2-1_difference_vs_{depth}km.png: 两两模型差异
- 2-1_slice_locations.png: 切片位置示意图
- 2-1_comparison_report.json / _summary.txt: 统计报告

科学原理:
----------
- dlnV = ln(V/V_ref) × 100%（对数速度扰动，百分比）
  小扰动近似: dlnV ≈ (V - V_ref)/V_ref × 100%
  正值 → 高速异常，负值 → 低速异常
- ΔV = V_A - V_B (km/s)（模型间绝对差异）
- 各向异性: 径向各向异性由 VSV/VSH、VPV/VPH 差异体现

作者: EASTASIA-FWI Team
日期: 2026-02-27
版本: v2.7
"""

import copy
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig

warnings.filterwarnings('ignore')


@dataclass
class ModelMetadata:
    """速度模型元数据类"""
    name: str
    full_name: str
    year: int
    reference: str
    region: str
    method: str
    color: str
    linestyle: str
    marker: str = 'o'
    # NetCDF元数据
    netcdf_file: Optional[str] = None
    actual_resolution: Optional[str] = None
    actual_depth_range: Optional[Tuple[float, float]] = None
    actual_spatial_range: Optional[Dict[str, Tuple[float, float]]] = None
    data_shape: Optional[Tuple[int, int, int]] = None


class ModelCompareConfig:
    """模型对比分析配置类（v2.7 - 双区域1D对比）"""
    
    def __init__(self):
        """初始化配置参数"""
        
        # ============ 目标模型配置（优化颜色方案）============
        # ============ v2.7 双区域数据源（对齐1_5）============
        # 共同区域对比: standardized.nc（模型交集的公平对比）
        # 单独模型可视化: original.nc（每个模型的完整覆盖范围）
        self.region_data_source = {
            'comparison': 'standardized',   # 多模型1D对比、水平切片、垂直剖面
            'individual': 'original',      # 单个模型1D完整剖面
        }
        
        # 模型数据根目录（对齐 1_5 输出，可覆盖）
        self.models_processed_base: Optional[Path] = None  # None = base_config.dirs['models']/processed
        
        self.target_models = {
            '2022_SinoScope1.0': {
                'netcdf_file': '2022_SinoScope1.0_standardized.nc',
                'netcdf_file_original': '2022_SinoScope1.0_original.nc',
                'processed_subdir': '2022_SinoScope1.0',
                'metadata': ModelMetadata(
                    name="SinoScope1.0",
                    full_name="SinoScope 1.0",
                    year=2022,
                    reference="Ma et al., 2022, GRL",
                    region="China and adjacent regions",
                    method="Full-waveform tomography",
                    color="#E41A1C",  # 红色
                    linestyle="-",
                    marker='o'
                )
            },
            '2024_EARA2024': {
                'netcdf_file': '2024_EARA2024_standardized.nc',
                'netcdf_file_original': '2024_EARA2024_original.nc',
                'processed_subdir': '2024_EARA2024',
                'metadata': ModelMetadata(
                    name="EARA2024",
                    full_name="EARA2024",
                    year=2024,
                    reference="Xi et al., 2024, GJI",
                    region="East Asia and northwestern Pacific",
                    method="Full-waveform tomography",
                    color="#028BFB",  # 蓝色
                    linestyle="-",
                    marker='s'
                )
            },
            '2024_FWEA23': {
                'netcdf_file': '2024_FWEA23_standardized.nc',
                'netcdf_file_original': '2024_FWEA23_original.nc',
                'processed_subdir': '2024_FWEA23',
                'metadata': ModelMetadata(
                    name="FWEA23",
                    full_name="FWEA23",
                    year=2024,
                    reference="Liu et al., 2024, EPSL",
                    region="East Asia FWI model",
                    method="Full-waveform tomography",
                    color="#05F811",  # 绿色
                    linestyle="-",
                    marker='^'
                )
            }
        }
        
        # ============ 1D剖面参数 ============
        self.profile_params = {
            'depth_range': (0, 1000),  # km
            'spatial_averaging': True,
        }
        
        # ============ 🆕 绘图模式选择（v2.6增强）============
        self.plot_mode = {
            'horizontal_slices': 'both',           # 选项: 'absolute', 'perturbation', 'both'
            'vertical_profiles': 'perturbation',   # 选项: 'absolute', 'perturbation', 'both'
            # 🆕 新增：独立的绝对速度和差异对比
            'enable_absolute_only': True,          # 是否生成独立的绝对速度图
            'enable_model_differences': True,      # 是否生成两两模型差异图
        }
        
        # ============ 切片可视化参数 ============
        self.slice_params = {
            # 水平切片深度（km）
            'horizontal_depths': [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900],
            # 垂直剖面位置
            'vertical_profiles': {
                'longitude': [100, 105, 110, 115, 120, 125, 130],  # 经度剖面
                'latitude': [25, 30, 35, 40, 45]                   # 纬度剖面
            },
            # 可视化参数
            'parameter': 'vs',
            'cmap': 'jet_r',  # 绝对速度色标
            'cmap_perturbation': 'jet_r',  # dlnV扰动图配色
            'cmap_difference': 'RdBu_r',   # 🆕 模型差异配色（红蓝对称）
            'vmin_percentile': 2,
            'vmax_percentile': 98,
            'dlnv_percentile': 95,  # dlnV的百分位数（对称范围）
            'diff_percentile': 95,  # 🆕 差异的百分位数（对称范围）
            # 海岸线配置
            'add_coastlines': True,
            'coastline_color': 'black',
            'coastline_linewidth': 0.5,
            # Colorbar配置（右侧垂直放置）
            'colorbar_orientation': 'vertical',
            'colorbar_position': 'right',
            'colorbar_pad': 0.02,
            'colorbar_shrink': 0.5,
        }
        
        # ============ 对比分析参数 ============
        self.comparison_params = {
            'anisotropy_analysis': True,
            's_wave_anisotropy': True,
            'p_wave_anisotropy': True,
            'calculate_differences': True,
            'calculate_dlnv': True,
        }
        
        # ============ 可视化参数 ============
        self.plot_params = {
            'figsize_1d': (6, 8),
            'figsize_1d_individual': (8, 10),
            'figsize_slice_single': (15, 5),   # 只画一种类型时的尺寸
            'figsize_slice_both': (15, 10),    # 画两种类型时的尺寸
            'figsize_profile': (15, 8),
            'figsize_location_map': (10, 8),
            'dpi': 300,
            'font_family': ['Arial', 'DejaVu Sans', 'sans-serif'],
            'font_sizes': {
                'title': 20,
                'xlabel': 16,
                'ylabel': 16,
                'legend': 15,
                'tick': 15,
                'colorbar': 15
            },
            'show_uncertainty': True,
            'uncertainty_alpha': 0.15,
            'uncertainty_multiplier': 3,
            'anisotropy_linestyles': {
                'vsv': '--',
                'vsh': ':',
                'vpv': '--',
                'vph': ':'
            }
        }
        
        # ============ 输出参数 ============
        self.output_params = {
            'save_formats': ['png'],
            'save_dpi': 300,
            'save_bbox': 'tight',
            'figure_prefix': '2-1_',
        }
        
        # ============ 日志配置 ============
        self.logging = {
            'level': 'INFO',
            'console_output': True
        }


class VelocityModelNetCDF:
    """基于NetCDF的速度模型数据类"""
    
    def __init__(self, netcdf_path: Path, metadata: ModelMetadata, logger):
        """
        初始化模型
        
        Args:
            netcdf_path: NetCDF文件路径
            metadata: 模型元数据
            logger: 日志器
        """
        self.path = netcdf_path
        self.metadata = metadata
        self.logger = logger
        
        # 坐标数组（load_netcdf 中赋值）
        self._lon: np.ndarray = np.array([])
        self._lat: np.ndarray = np.array([])
        self._depth: np.ndarray = np.array([])
        
        # Dataset 引用
        self.ds: xr.Dataset = xr.Dataset()
        
        # 一维参考模型缓存
        self._reference_1d_profile: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        
        self._load_netcdf()
    
    def _load_netcdf(self) -> None:
        """加载NetCDF数据"""
        try:
            self.logger.info(f"📦 加载NetCDF: {self.metadata.name}")
            self.logger.info(f"  文件: {self.path}")
            
            self.ds = xr.open_dataset(self.path)
            
            self._lon = self.ds['longitude'].values
            self._lat = self.ds['latitude'].values
            self._depth = self.ds['depth'].values
            
            # 更新元数据
            self.metadata.netcdf_file = self.path.name
            self.metadata.data_shape = (len(self._lat), len(self._lon), len(self._depth))
            self.metadata.actual_depth_range = (float(self._depth.min()), float(self._depth.max()))
            self.metadata.actual_spatial_range = {
                'lat': (float(self._lat.min()), float(self._lat.max())),
                'lon': (float(self._lon.min()), float(self._lon.max()))
            }
            
            # 估算分辨率
            lat_res = np.mean(np.diff(np.sort(self._lat)))
            lon_res = np.mean(np.diff(np.sort(self._lon)))
            self.metadata.actual_resolution = f"{lat_res:.2f}°×{lon_res:.2f}°"
            
            # 统计信息
            self.logger.info(f"✅ NetCDF加载成功")
            self.logger.info(f"  维度: {self.metadata.data_shape}")
            self.logger.info(f"  纬度: {self._lat.min():.2f}° ~ {self._lat.max():.2f}°")
            self.logger.info(f"  经度: {self._lon.min():.2f}° ~ {self._lon.max():.2f}°")
            self.logger.info(f"  深度: {self._depth.min():.1f} ~ {self._depth.max():.1f} km")
            self.logger.info(f"  分辨率: {self.metadata.actual_resolution}")
            
            available_params = [str(v) for v in self.ds.data_vars if v not in ['longitude', 'latitude', 'depth']]
            self.logger.info(f"  可用参数: {', '.join(available_params)}")
            
        except Exception as e:
            self.logger.error(f"❌ NetCDF加载失败: {e}")
            raise
    
    @property
    def lon(self) -> np.ndarray:
        """经度坐标"""
        return self._lon
    
    @property
    def lat(self) -> np.ndarray:
        """纬度坐标"""
        return self._lat
    
    @property
    def depth(self) -> np.ndarray:
        """深度坐标"""
        return self._depth
    
    def get_parameter(self, param: str) -> np.ndarray:
        """
        获取指定参数的3D数组
        
        Args:
            param: 参数名称（vs, vp, vsv, vsh等）
            
        Returns:
            3D数组 (lat, lon, depth)
        """
        if param not in self.ds:
            raise ValueError(f"参数 {param} 不存在于模型中")
        
        data = self.ds[param].values
        data_transposed = np.transpose(data, (1, 0, 2))
        
        return data_transposed
    
    def calculate_1d_profile(self, depth_range: Tuple[float, float],
                            spatial_averaging: bool = True) -> Dict[str, Any]:
        """
        计算1D平均速度剖面（基于NetCDF）
        
        Args:
            depth_range: 深度范围 (min, max)
            spatial_averaging: 是否进行空间平均
            
        Returns:
            剖面字典
        """
        depth_mask = (self._depth >= depth_range[0]) & (self._depth <= depth_range[1])
        depths = self._depth[depth_mask]
        
        if len(depths) == 0:
            self.logger.warning(f"深度范围 {depth_range} 内无数据")
            return {'depth': np.array([]), 'n_depths': 0}
        
        profile = {
            'depth': depths,
            'n_depths': len(depths)
        }
        
        for param in ['vs', 'vp', 'vsv', 'vsh', 'vpv', 'vph']:
            if param not in self.ds:
                continue
            
            try:
                data_3d = self.get_parameter(param)
                data_depth = data_3d[:, :, depth_mask]
                
                if spatial_averaging:
                    mean_values = np.nanmean(data_depth, axis=(0, 1))
                    std_values = np.nanstd(data_depth, axis=(0, 1))
                    count_values = np.sum(~np.isnan(data_depth), axis=(0, 1))
                else:
                    mean_values = np.nanmedian(data_depth, axis=(0, 1))
                    std_values = np.nanstd(data_depth, axis=(0, 1))
                    count_values = np.sum(~np.isnan(data_depth), axis=(0, 1))
                
                profile[param] = {
                    'mean': mean_values,
                    'std': std_values,
                    'count': count_values
                }
                
            except Exception as e:
                self.logger.warning(f"计算 {param} 剖面失败: {e}")
        
        return profile
    
    def get_1d_reference_profile(self, param: str) -> Dict[str, np.ndarray]:
        """
        获取一维参考剖面（用于计算dlnV扰动）
        
        Args:
            param: 参数名称
            
        Returns:
            {'depth': depths, 'value': reference_values}
        """
        if self._reference_1d_profile is None:
            self._reference_1d_profile = {}
        
        if param not in self._reference_1d_profile:
            profile = self.calculate_1d_profile(
                depth_range=(0, float(self._depth.max())),
                spatial_averaging=True
            )
            
            if param in profile:
                self._reference_1d_profile[param] = {
                    'depth': profile['depth'],
                    'value': profile[param]['mean']
                }
        
        return self._reference_1d_profile.get(param, {'depth': np.array([]), 'value': np.array([])})
    
    def get_horizontal_slice(self, param: str, depth: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取指定深度的水平切片
        
        Args:
            param: 参数名称
            depth: 深度值（km）
            
        Returns:
            (lon_grid, lat_grid, data_slice)
        """
        depth_idx = np.argmin(np.abs(self._depth - depth))
        actual_depth = self._depth[depth_idx]
        
        self.logger.info(f"  提取水平切片: {param} @ {actual_depth:.1f} km")
        
        data_3d = self.get_parameter(param)
        data_slice = data_3d[:, :, depth_idx]
        
        lon_grid, lat_grid = np.meshgrid(self._lon, self._lat)
        
        return lon_grid, lat_grid, data_slice
    
    def get_horizontal_slice_dlnv(self, param: str, depth: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取指定深度的dlnV扰动水平切片（相对于1D参考模型，百分比显示）
        
        dlnV定义：dlnV = ln(V/V_ref) × 100%
        
        Args:
            param: 参数名称
            depth: 深度值（km）
            
        Returns:
            (lon_grid, lat_grid, dlnv_slice_percent)
        """
        lon_grid, lat_grid, data_slice = self.get_horizontal_slice(param, depth)
        
        ref_profile = self.get_1d_reference_profile(param)
        
        if len(ref_profile['depth']) == 0:
            self.logger.warning(f"无法获取 {param} 的1D参考剖面")
            return lon_grid, lat_grid, np.zeros_like(data_slice)
        
        depth_idx = np.argmin(np.abs(self._depth - depth))
        actual_depth = self._depth[depth_idx]
        
        ref_value = np.interp(actual_depth, ref_profile['depth'], ref_profile['value'])
        
        # 计算dlnV扰动并转换为百分比
        with np.errstate(divide='ignore', invalid='ignore'):
            dlnv_slice = np.log(data_slice / ref_value) * 100.0
            dlnv_slice[~np.isfinite(dlnv_slice)] = np.nan
        
        return lon_grid, lat_grid, dlnv_slice
    
    def get_vertical_profile(self, param: str, 
                            direction: str, 
                            position: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取垂直剖面
        
        Args:
            param: 参数名称
            direction: 'longitude' 或 'latitude'
            position: 位置值
            
        Returns:
            (horizontal_coords, depth_grid, data_profile)
        """
        data_3d = self.get_parameter(param)
        
        if direction == 'longitude':
            lon_idx = np.argmin(np.abs(self._lon - position))
            actual_pos = self._lon[lon_idx]
            data_profile = data_3d[:, lon_idx, :]
            horizontal_coords = self._lat
            label = f"Lon={actual_pos:.1f}°"
            
        elif direction == 'latitude':
            lat_idx = np.argmin(np.abs(self._lat - position))
            actual_pos = self._lat[lat_idx]
            data_profile = data_3d[lat_idx, :, :]
            horizontal_coords = self._lon
            label = f"Lat={actual_pos:.1f}°"
        else:
            raise ValueError(f"未知方向: {direction}")
        
        self.logger.info(f"  提取垂直剖面: {param} @ {label}")
        
        depth_grid, horiz_grid = np.meshgrid(self._depth, horizontal_coords)
        
        return horiz_grid, depth_grid, data_profile
    
    def get_vertical_profile_dlnv(self, param: str, 
                                  direction: str, 
                                  position: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取垂直剖面的dlnV扰动（相对于1D参考模型，百分比显示）
        
        dlnV定义：dlnV = ln(V/V_ref) × 100%
        
        Args:
            param: 参数名称
            direction: 'longitude' 或 'latitude'
            position: 位置值
            
        Returns:
            (horizontal_coords, depth_grid, dlnv_profile_percent)
        """
        horiz_grid, depth_grid, data_profile = self.get_vertical_profile(param, direction, position)
        
        ref_profile = self.get_1d_reference_profile(param)
        
        if len(ref_profile['depth']) == 0:
            self.logger.warning(f"无法获取 {param} 的1D参考剖面")
            return horiz_grid, depth_grid, np.zeros_like(data_profile)
        
        dlnv_profile = np.zeros_like(data_profile)
        
        for i in range(data_profile.shape[1]):
            depth_val = self._depth[i]
            ref_value = np.interp(depth_val, ref_profile['depth'], ref_profile['value'])
            
            # 计算dlnV扰动并转换为百分比
            with np.errstate(divide='ignore', invalid='ignore'):
                dlnv_profile[:, i] = np.log(data_profile[:, i] / ref_value) * 100.0
                dlnv_profile[~np.isfinite(dlnv_profile)] = np.nan
        
        return horiz_grid, depth_grid, dlnv_profile
    
    def has_parameter(self, param: str) -> bool:
        """检查是否有指定参数"""
        return param in self.ds
    
    def has_s_wave_anisotropy(self) -> bool:
        """检查是否有S波各向异性"""
        return self.has_parameter('vsv') and self.has_parameter('vsh')
    
    def has_p_wave_anisotropy(self) -> bool:
        """检查是否有P波各向异性"""
        return self.has_parameter('vpv') and self.has_parameter('vph')
    
    def get_statistics(self, param: str) -> Dict[str, float]:
        """获取参数统计信息"""
        if not self.has_parameter(param):
            return {}
        
        data = self.get_parameter(param)
        valid_data = data[~np.isnan(data)]
        
        return {
            'min': float(valid_data.min()),
            'max': float(valid_data.max()),
            'mean': float(valid_data.mean()),
            'std': float(valid_data.std()),
            'median': float(np.median(valid_data)),
            'valid_percentage': float(len(valid_data) / data.size * 100)
        }
    
    def close(self):
        """关闭NetCDF文件"""
        self.ds.close()
    
    def __del__(self):
        """析构函数"""
        self.close()

# ==================== 模型对比分析器类 ====================

class ModelComparator:
    """速度模型对比分析器（v2.7 - 双区域1D对比）"""
    
    def __init__(self):
        """初始化对比分析器"""
        # 加载基础配置
        self.base_config = BaseConfig()
        
        # 加载模块配置
        self.config = ModelCompareConfig()
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.ModelCompare',
            self.config.logging['level']
        )
        
        # 初始化模型存储
        self.models: Dict[str, VelocityModelNetCDF] = {}  # 共同区域对比（standardized）
        self.models_individual: Dict[str, VelocityModelNetCDF] = {}  # 单独模型（original，按需加载）
        
        # 设置绘图样式
        self._setup_plotting_style()
        
        # 输出目录
        self.output_dir = self.base_config.dirs['figures'] / 'model_compare'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("="*80)
        self.logger.info("🚀 速度模型对比分析器初始化 (v2.7 - 双区域1D对比)")
        self.logger.info("="*80)
        self._print_config_summary()
    
    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 模型对比分析配置 (v2.7 - 双区域1D对比)")
        print("-" * 60)
        print(f"目标模型数量: {len(self.config.target_models)}")
        print(f"\n🗺️  双区域数据源（对齐1_5）:")
        print(f"  共同区域对比: {self.config.region_data_source['comparison']}.nc")
        print(f"  单独模型可视化: {self.config.region_data_source['individual']}.nc")
        for key, info in self.config.target_models.items():
            meta = info['metadata']
            print(f"  - {meta.full_name} ({meta.year}) - 颜色: {meta.color}")
        
        print(f"\n🆕 灵活绘图模式:")
        print(f"  水平切片模式: {self.config.plot_mode['horizontal_slices']}")
        print(f"  垂直剖面模式: {self.config.plot_mode['vertical_profiles']}")
        print(f"  🆕 独立绝对速度图: {self.config.plot_mode['enable_absolute_only']}")
        print(f"  🆕 两两模型差异图: {self.config.plot_mode['enable_model_differences']}")
        
        print(f"\n🔥 核心功能:")
        print(f"  1. 综合对比图（所有模型）")
        print(f"  2. 单个模型详细图")
        print(f"  3. 🆕 独立绝对速度对比图（所有模型在同一深度）")
        print(f"  4. 🆕 两两模型差异图（Model_A - Model_B）")
        print(f"  5. 水平切片（可选绝对值/扰动/两者）+ 海岸线")
        print(f"  6. 垂直剖面（可选绝对值/扰动/两者）")
        print(f"  7. 切片位置示意图（地理分布）")
        
        print(f"\n🆕 可视化增强:")
        print(f"  • dlnV扰动单位: 百分比 (%)")
        print(f"  • 差异单位: km/s (ΔV = V_A - V_B)")
        print(f"  • 切片色标: seismic / RdBu_r")
        print(f"  • 水平切片colorbar: 右侧垂直放置")
        
        print(f"\n输出目录: {self.output_dir}")
        print("-" * 60)
    
    def _setup_plotting_style(self):
        """设置绘图样式"""
        plt.rcParams['font.sans-serif'] = self.config.plot_params['font_family']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': self.config.plot_params['font_sizes']['tick'],
            'axes.titlesize': self.config.plot_params['font_sizes']['title'],
            'axes.labelsize': self.config.plot_params['font_sizes']['xlabel'],
            'legend.fontsize': self.config.plot_params['font_sizes']['legend'],
            'figure.dpi': self.config.plot_params['dpi']
        })
    
    def _get_model_path(self, model_key: str, region_type: str = 'standardized') -> Path:
        """
        获取模型NetCDF文件路径（对齐1_5输出结构）
        
        Args:
            model_key: 模型键名
            region_type: 'standardized'（共同区域）或 'original'（原始模型区域）
            
        Returns:
            NetCDF文件完整路径
        """
        if model_key not in self.config.target_models:
            raise KeyError(f"未知模型: {model_key}")
        
        model_info = self.config.target_models[model_key]
        subdir = model_info.get('processed_subdir', model_key)
        
        if region_type == 'standardized':
            nc_file = model_info.get('netcdf_file', f'{model_key}_standardized.nc')
        else:
            nc_file = model_info.get('netcdf_file_original', f'{model_key}_original.nc')
        
        # 路径对齐 1_5：data/models/processed/{model_name}/（可配置 models_processed_base 覆盖）
        base = self.config.models_processed_base
        models_dir = (Path(base) if base is not None else 
                     self.base_config.dirs['models'] / 'processed')
        return models_dir / subdir / nc_file
    
    def load_models(self):
        """
        加载共同区域NetCDF模型（standardized），用于多模型对比。
        单独模型（original）由 _load_model_individual() 按需加载。
        """
        region_type = self.config.region_data_source.get('comparison', 'standardized')
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"📦 加载NetCDF模型（共同区域对比: {region_type}）")
        self.logger.info("="*80)
        
        for model_key, model_info in self.config.target_models.items():
            netcdf_path = self._get_model_path(model_key, region_type)
            
            if not netcdf_path.exists():
                self.logger.warning(f"⚠️  NetCDF文件不存在: {netcdf_path}")
                continue
            
            try:
                model = VelocityModelNetCDF(
                    netcdf_path,
                    model_info['metadata'],
                    self.logger
                )
                self.models[model_key] = model
                
            except Exception as e:
                self.logger.error(f"❌ 加载失败 {model_key}: {e}")
        
        self.logger.info(f"\n✅ 成功加载 {len(self.models)} 个NetCDF模型（共同区域）")
        
        if len(self.models) == 0:
            raise RuntimeError("未能加载任何模型")
    
    def _load_model_individual(self, model_key: str) -> Optional[VelocityModelNetCDF]:
        """
        按需加载单个模型的原始区域数据（用于单独模型1D可视化）
        
        Args:
            model_key: 模型键名
            
        Returns:
            VelocityModelNetCDF 或 None
        """
        if model_key in self.models_individual:
            return self.models_individual[model_key]
        
        netcdf_path = self._get_model_path(model_key, 'original')
        
        if not netcdf_path.exists():
            self.logger.warning(f"⚠️  单独模型NetCDF不存在（将回退到共同区域）: {netcdf_path}")
            return self.models.get(model_key)
        
        try:
            model_info = self.config.target_models[model_key]
            metadata_copy = copy.deepcopy(model_info['metadata'])
            model = VelocityModelNetCDF(
                netcdf_path,
                metadata_copy,
                self.logger
            )
            self.models_individual[model_key] = model
            return model
        except Exception as e:
            self.logger.error(f"❌ 加载单独模型失败 {model_key}: {e}")
            return self.models.get(model_key)
    
    # ==================== 1D速度剖面对比 ====================
    
    def plot_1d_velocity_comparison(self) -> Figure:
        """
        绘制1D速度剖面对比图（共同区域，使用 standardized 数据）
        
        多模型在共同覆盖区域的公平对比，VS和VP用实线、仅颜色区分。
        
        Returns:
            综合速度剖面对比图
        """
        self.logger.info("🎨 绘制共同区域1D剖面对比图（standardized，VS + VP）...")
        
        fig, ax = plt.subplots(figsize=self.config.plot_params['figsize_1d'])
        
        depth_range = self.config.profile_params['depth_range']
        plotted_models = []
        
        vs_annotated = False
        vp_annotated = False
        
        for model_key, model in self.models.items():
            try:
                profile = model.calculate_1d_profile(
                    depth_range=depth_range,
                    spatial_averaging=True
                )
                
                if 'depth' not in profile or len(profile['depth']) == 0:
                    continue
                
                depths = profile['depth']
                color = model.metadata.color
                model_plotted = False
                
                for target_param in ['vs', 'vp']:
                    if target_param in profile:
                        values = profile[target_param]
                        valid_mask = ~np.isnan(values['mean'])
                        
                        if valid_mask.sum() > 0:
                            label = model.metadata.full_name if not model_plotted else None
                            
                            ax.plot(
                                values['mean'][valid_mask], depths[valid_mask],
                                color=color,
                                linestyle='-',
                                linewidth=1.5,
                                label=label,
                                alpha=0.9,
                            )
                            
                            model_plotted = True
                            
                            # 添加参数标注
                            if target_param == 'vs' and not vs_annotated and valid_mask.sum() > 20:
                                mid_idx = valid_mask.sum() // 2
                                valid_indices = np.where(valid_mask)[0]
                                if len(valid_indices) > mid_idx:
                                    idx = valid_indices[mid_idx]
                                    ax.text(
                                        values['mean'][idx] - 0.3, depths[idx],
                                        'VS',
                                        color='darkred',
                                        fontsize=16,
                                        va='center',
                                        fontweight='bold',
                                        bbox=dict(boxstyle='round,pad=0.5', 
                                                facecolor='white', 
                                                edgecolor='darkred',
                                                alpha=0.9,
                                                linewidth=2)
                                    )
                                    vs_annotated = True
                            
                            elif target_param == 'vp' and not vp_annotated and valid_mask.sum() > 20:
                                mid_idx = valid_mask.sum() // 2
                                valid_indices = np.where(valid_mask)[0]
                                if len(valid_indices) > mid_idx:
                                    idx = valid_indices[mid_idx]
                                    ax.text(
                                        values['mean'][idx] + 0.3, depths[idx],
                                        'VP',
                                        color='darkblue',
                                        fontsize=16,
                                        va='center',
                                        fontweight='bold',
                                        bbox=dict(boxstyle='round,pad=0.5', 
                                                facecolor='white', 
                                                edgecolor='darkblue',
                                                alpha=0.9,
                                                linewidth=2)
                                    )
                                    vp_annotated = True
                            
                            # 不确定性阴影
                            if self.config.plot_params['show_uncertainty']:
                                valid_std_mask = valid_mask & ~np.isnan(values['std'])
                                if valid_std_mask.sum() > 0:
                                    multiplier = self.config.plot_params['uncertainty_multiplier']
                                    ax.fill_betweenx(
                                        depths[valid_std_mask],
                                        values['mean'][valid_std_mask] - multiplier * values['std'][valid_std_mask],
                                        values['mean'][valid_std_mask] + multiplier * values['std'][valid_std_mask],
                                        color=color,
                                        alpha=self.config.plot_params['uncertainty_alpha'],
                                        linewidth=0
                                    )
                
                if model_plotted:
                    plotted_models.append(model_key)
                    self.logger.info(f"  ✅ {model.metadata.name} 绘制成功")
                
            except Exception as e:
                self.logger.error(f"绘制失败 {model_key}: {e}")
        
        # 设置图形属性
        ax.invert_yaxis()
        ax.set_ylim(depth_range[1], 0)
        ax.set_xlim(1, 13)
        ax.set_xlabel("Velocity (km/s)", 
                     fontsize=self.config.plot_params['font_sizes']['xlabel'])
        ax.set_ylabel("Depth (km)", 
                     fontsize=self.config.plot_params['font_sizes']['ylabel'])
        ax.set_title("Velocity Profile Comparison (VS & VP)", 
                    fontsize=self.config.plot_params['font_sizes']['title'], 
                    pad=20)
        
        # 地质分界面
        for depth in [60, 410, 660]:
            if depth <= depth_range[1]:
                ax.axhline(y=depth, color='gray', linestyle='--', 
                          linewidth=1.5, alpha=0.5)
                # ax.text(1.2, depth-10, f'{depth} km', 
                #        fontsize=9, color='gray', va='top')
        
        ax.legend(
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            title='Velocity Models',
            frameon=True,
            fontsize=self.config.plot_params['font_sizes']['legend']
        )
        
        ax.grid(True, linestyle=':', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    # ==================== S波各向异性对比图 ====================
    
    def plot_s_wave_anisotropy_comparison(self) -> Optional[Figure]:
        """绘制S波各向异性对比图（VSV实线，VSH虚线）"""
        self.logger.info("🎨 绘制S波各向异性对比图...")
        
        models_with_aniso = [k for k, m in self.models.items() 
                            if m.has_s_wave_anisotropy()]
        
        if len(models_with_aniso) == 0:
            self.logger.warning("没有模型包含S波各向异性数据")
            return None
        
        fig, ax = plt.subplots(figsize=self.config.plot_params['figsize_1d'])
        depth_range = self.config.profile_params['depth_range']
        
        for model_key in models_with_aniso:
            model = self.models[model_key]
            
            try:
                profile = model.calculate_1d_profile(depth_range=depth_range)
                
                if 'depth' not in profile or len(profile['depth']) == 0:
                    continue
                
                depths = profile['depth']
                color = model.metadata.color
                
                # VSV: 实线
                if 'vsv' in profile:
                    vsv = profile['vsv']
                    valid_mask = ~np.isnan(vsv['mean'])
                    if valid_mask.sum() > 0:
                        ax.plot(
                            vsv['mean'][valid_mask], depths[valid_mask],
                            color=color,
                            linestyle='-',
                            linewidth=1.5,
                            label=f"{model.metadata.full_name} VSV",
                            alpha=0.9
                        )
                
                # VSH: 虚线
                if 'vsh' in profile:
                    vsh = profile['vsh']
                    valid_mask = ~np.isnan(vsh['mean'])
                    if valid_mask.sum() > 0:
                        ax.plot(
                            vsh['mean'][valid_mask], depths[valid_mask],
                            color=color,
                            linestyle='--',
                            linewidth=1.5,
                            label=f"{model.metadata.full_name} VSH",
                            alpha=0.9
                        )
            
            except Exception as e:
                self.logger.error(f"绘制S波各向异性失败 {model_key}: {e}")
        
        ax.invert_yaxis()
        ax.set_ylim(depth_range[1], 0)
        ax.set_xlabel("S-wave Velocity (km/s)", 
                     fontsize=self.config.plot_params['font_sizes']['xlabel'])
        ax.set_ylabel("Depth (km)", 
                     fontsize=self.config.plot_params['font_sizes']['ylabel'])
        ax.set_title("S-wave Anisotropy Comparison (VSV vs VSH)", 
                    fontsize=self.config.plot_params['font_sizes']['title'], 
                    pad=20)
        
        for depth in [60, 410, 660]:
            if depth <= depth_range[1]:
                ax.axhline(y=depth, color='gray', linestyle='--', 
                          linewidth=1.5, alpha=0.5)
        
        ax.legend(loc='upper right', 
                 fontsize=self.config.plot_params['font_sizes']['legend']-1,
                 ncol=1)
        ax.grid(True, linestyle=':', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    # ==================== P波各向异性对比图 ====================
    
    def plot_p_wave_anisotropy_comparison(self) -> Optional[Figure]:
        """绘制P波各向异性对比图（VPV实线，VPH虚线）"""
        self.logger.info("🎨 绘制P波各向异性对比图...")
        
        models_with_aniso = [k for k, m in self.models.items() 
                            if m.has_p_wave_anisotropy()]
        
        if len(models_with_aniso) == 0:
            self.logger.warning("没有模型包含P波各向异性数据")
            return None
        
        fig, ax = plt.subplots(figsize=self.config.plot_params['figsize_1d'])
        depth_range = self.config.profile_params['depth_range']
        
        for model_key in models_with_aniso:
            model = self.models[model_key]
            
            try:
                profile = model.calculate_1d_profile(depth_range=depth_range)
                
                if 'depth' not in profile or len(profile['depth']) == 0:
                    continue
                
                depths = profile['depth']
                color = model.metadata.color
                
                # VPV: 实线
                if 'vpv' in profile:
                    vpv = profile['vpv']
                    valid_mask = ~np.isnan(vpv['mean'])
                    if valid_mask.sum() > 0:
                        ax.plot(
                            vpv['mean'][valid_mask], depths[valid_mask],
                            color=color,
                            linestyle='-',
                            linewidth=1.5,
                            label=f"{model.metadata.full_name} VPV",
                            alpha=0.9
                        )
                
                # VPH: 虚线
                if 'vph' in profile:
                    vph = profile['vph']
                    valid_mask = ~np.isnan(vph['mean'])
                    if valid_mask.sum() > 0:
                        ax.plot(
                            vph['mean'][valid_mask], depths[valid_mask],
                            color=color,
                            linestyle='--',
                            linewidth=1.5,
                            label=f"{model.metadata.full_name} VPH",
                            alpha=0.9
                        )
            
            except Exception as e:
                self.logger.error(f"绘制P波各向异性失败 {model_key}: {e}")
        
        ax.invert_yaxis()
        ax.set_ylim(depth_range[1], 0)
        ax.set_xlabel("P-wave Velocity (km/s)", 
                     fontsize=self.config.plot_params['font_sizes']['xlabel'])
        ax.set_ylabel("Depth (km)", 
                     fontsize=self.config.plot_params['font_sizes']['ylabel'])
        ax.set_title("P-wave Anisotropy Comparison (VPV vs VPH)", 
                    fontsize=self.config.plot_params['font_sizes']['title'], 
                    pad=20)
        
        for depth in [60, 410, 660]:
            if depth <= depth_range[1]:
                ax.axhline(y=depth, color='gray', linestyle='--', 
                          linewidth=1.5, alpha=0.5)
        
        ax.legend(loc='upper right', 
                 fontsize=self.config.plot_params['font_sizes']['legend']-1,
                 ncol=1)
        ax.grid(True, linestyle=':', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    # ==================== 单个模型详细剖面（v2.7 使用 original 数据）====================
    
    def plot_individual_model_profile(self, model_key: str, 
                                      use_original_region: bool = True) -> Optional[Figure]:
        """
        绘制单个模型的完整速度剖面（v2.7 对齐1_5：单独模型可视化使用 original 区域）
        
        Args:
            model_key: 模型键名
            use_original_region: 若 True，使用 *_original.nc（完整模型区域）；
                               若 False，使用 standardized（共同区域，兼容旧逻辑）
            
        Returns:
            单个模型的完整剖面图
        """
        if model_key not in self.models:
            self.logger.warning(f"模型 {model_key} 不存在")
            return None
        
        # v2.7: 单独模型默认使用 original 数据（完整覆盖范围）
        if use_original_region and self.config.region_data_source.get('individual') == 'original':
            model = self._load_model_individual(model_key)
            region_label = "Original Region"
        else:
            model = self.models[model_key]
            region_label = "Common Region"
        
        if model is None:
            return None
        
        self.logger.info(f"🎨 绘制单个模型剖面: {model.metadata.full_name} ({region_label})")
        
        fig, ax = plt.subplots(figsize=self.config.plot_params['figsize_1d_individual'])
        depth_range = self.config.profile_params['depth_range']
        
        try:
            profile = model.calculate_1d_profile(depth_range=depth_range)
            
            if 'depth' not in profile or len(profile['depth']) == 0:
                self.logger.warning(f"模型 {model_key} 无有效数据")
                return None
            
            depths = profile['depth']
            
            param_styles = {
                'vp': {'color': '#E41A1C', 'linestyle': '-', 'linewidth': 3, 'label': 'VP', 'alpha': 1.0},
                'vs': {'color': '#377EB8', 'linestyle': '-', 'linewidth': 3, 'label': 'VS', 'alpha': 1.0},
                'vpv': {'color': '#FF7F00', 'linestyle': '-', 'linewidth': 2.5, 'label': 'VPV', 'alpha': 0.8},
                'vph': {'color': '#FF7F00', 'linestyle': '--', 'linewidth': 2.5, 'label': 'VPH', 'alpha': 0.8},
                'vsv': {'color': '#4DAF4A', 'linestyle': '-', 'linewidth': 2.5, 'label': 'VSV', 'alpha': 0.8},
                'vsh': {'color': '#4DAF4A', 'linestyle': '--', 'linewidth': 2.5, 'label': 'VSH', 'alpha': 0.8},
            }
            
            for param, style in param_styles.items():
                if param in profile:
                    values = profile[param]
                    valid_mask = ~np.isnan(values['mean'])
                    
                    if valid_mask.sum() > 0:
                        ax.plot(
                            values['mean'][valid_mask], depths[valid_mask],
                            color=style['color'],
                            linestyle=style['linestyle'],
                            linewidth=style['linewidth'],
                            label=style['label'],
                            alpha=style['alpha']
                        )
            
            ax.invert_yaxis()
            ax.set_ylim(depth_range[1], 0)
            ax.set_xlabel("Velocity (km/s)", fontsize=14)
            ax.set_ylabel("Depth (km)", fontsize=14)
            region_title = "Original Region" if use_original_region and self.config.region_data_source.get('individual') == 'original' else "Common Region"
            ax.set_title(f"{model.metadata.full_name}\n1D Velocity Profile ({region_title})", 
                        fontsize=16, pad=20, fontweight='bold')
            
            for depth in [60, 410, 660]:
                if depth <= depth_range[1]:
                    ax.axhline(y=depth, color='gray', linestyle='--', 
                              linewidth=1.5, alpha=0.5)
                    # ax.text(1.2, depth-10, f'{depth} km', 
                    #        fontsize=10, color='gray', va='top')
            
            handles, labels = ax.get_legend_handles_labels()
            
            if len(handles) > 0:
                iso_handles = [h for h, l in zip(handles, labels) if l in ['VP', 'VS']]
                iso_labels = [l for l in labels if l in ['VP', 'VS']]
                
                if iso_handles:
                    legend1 = ax.legend(
                        iso_handles, iso_labels,
                        loc='upper right',
                        bbox_to_anchor=(0.98, 0.98),
                        title='Isotropic',
                        frameon=True,
                        fontsize=11
                    )
                    ax.add_artist(legend1)
                
                aniso_handles = [h for h, l in zip(handles, labels) if l not in ['VP', 'VS']]
                aniso_labels = [l for l in labels if l not in ['VP', 'VS']]
                
                if aniso_handles:
                    ax.legend(
                        aniso_handles, aniso_labels,
                        loc='upper right',
                        bbox_to_anchor=(0.98, 0.85),
                        title='Anisotropic',
                        frameon=True,
                        fontsize=11
                    )
            
            ax.grid(True, linestyle=':', alpha=0.3)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"绘制单个模型剖面失败 {model_key}: {e}")
            return None
    
    # ==================== 切片位置示意图 ====================
    
    def plot_slice_location_map(self) -> Figure:
        """
        绘制所有切片位置的示意图
        
        功能说明：
        - 显示水平切片的深度信息（文本标注）
        - 显示经度剖面的位置（红色竖线）
        - 显示纬度剖面的位置（蓝色横线）
        - 使用Cartopy添加地理信息和海岸线
        
        Returns:
            切片位置示意图
        """
        self.logger.info("🗺️  绘制切片位置示意图...")
        
        # 获取任意一个模型的空间范围
        first_model = next(iter(self.models.values()))
        spatial_range = first_model.metadata.actual_spatial_range
        if spatial_range is None:
            raise ValueError(f"模型 {first_model.metadata.name} 缺少空间范围信息")
        
        lon_min, lon_max = spatial_range['lon']
        lat_min, lat_max = spatial_range['lat']
        
        fig = plt.figure(figsize=self.config.plot_params['figsize_location_map'])
        ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # type: ignore[assignment]
        
        # 设置地图范围
        ax.set_extent([lon_min-2, lon_max+2, lat_min-2, lat_max+2], crs=ccrs.PlateCarree())
        
        # 添加地理要素
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':', edgecolor='gray')
        
        # 添加网格线
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                         alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # 绘制研究区域边界
        rect = Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=2, edgecolor='black', facecolor='none',
            linestyle='--', transform=ccrs.PlateCarree(), zorder=5
        )
        ax.add_patch(rect)
        
        # 绘制经度剖面位置（红色竖线）
        lon_profiles = self.config.slice_params['vertical_profiles']['longitude']
        for lon in lon_profiles:
            ax.plot([lon, lon], [lat_min, lat_max], 
                   color='red', linewidth=2, linestyle='-', 
                   alpha=0.7, transform=ccrs.PlateCarree(), zorder=3)
            # 添加经度标签
            ax.text(lon, lat_max + 0.5, f'{lon}°E', 
                   color='red', fontsize=10, ha='center', va='bottom',
                   fontweight='bold', transform=ccrs.PlateCarree())
        
        # 绘制纬度剖面位置（蓝色横线）
        lat_profiles = self.config.slice_params['vertical_profiles']['latitude']
        for lat in lat_profiles:
            ax.plot([lon_min, lon_max], [lat, lat], 
                   color='blue', linewidth=2, linestyle='-', 
                   alpha=0.7, transform=ccrs.PlateCarree(), zorder=3)
            # 添加纬度标签
            ax.text(lon_max + 0.5, lat, f'{lat}°N', 
                   color='blue', fontsize=10, ha='left', va='center',
                   fontweight='bold', transform=ccrs.PlateCarree())
        
        # 添加水平切片深度信息（在地图外显示）
        depths_str = ', '.join([f'{d}km' for d in self.config.slice_params['horizontal_depths']])
        ax.text(0.5, -0.08, f'Horizontal Slice Depths: {depths_str}', 
               transform=ax.transAxes, ha='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Longitude Profiles'),
            Line2D([0], [0], color='blue', linewidth=2, label='Latitude Profiles'),
            Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Study Region')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11, frameon=True)
        
        # 设置标题
        ax.set_title('Slice Locations for Model Comparison\n(Horizontal Slices + Vertical Profiles)', 
                    fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        self.logger.info("  ✅ 切片位置示意图绘制完成")
        return fig

    # ==================== 🆕 独立的绝对速度对比图 ====================
    
    def plot_horizontal_slices_absolute_only(self, param: str = 'vs') -> List[Figure]:
        """
        🆕 绘制独立的绝对速度对比图（所有模型在同一深度）
        
        功能说明：
        - 每个深度生成一张图
        - 所有模型并排显示在同一行
        - 统一的colorbar范围
        - 添加海岸线
        - colorbar放置在右侧（垂直方向）
        
        Args:
            param: 要可视化的参数（vs, vp等）
            
        Returns:
            图表列表
        """
        self.logger.info(f"🆕 绘制独立绝对速度对比图: {param.upper()}")
        
        depths = self.config.slice_params['horizontal_depths']
        cmap = self.config.slice_params['cmap']
        
        figures = []
        
        for depth in depths:
            self.logger.info(f"  生成深度 {depth} km 的绝对速度对比...")
            
            n_models = len(self.models)
            figsize = (5*n_models, 5)
            
            fig = plt.figure(figsize=figsize)
            
            # ========== 收集所有模型数据，计算统一colorbar范围 ==========
            all_data = []
            for model in self.models.values():
                if model.has_parameter(param):
                    try:
                        _, _, data_slice = model.get_horizontal_slice(param, depth)
                        all_data.append(data_slice[~np.isnan(data_slice)])
                    except Exception:
                        pass
            
            if len(all_data) == 0:
                self.logger.warning(f"  深度 {depth} km 无有效数据")
                plt.close(fig)
                continue
            
            all_data_flat = np.concatenate(all_data)
            vmin = np.percentile(all_data_flat, self.config.slice_params['vmin_percentile'])
            vmax = np.percentile(all_data_flat, self.config.slice_params['vmax_percentile'])
            
            # ========== 绘制每个模型 ==========
            for idx, (model_key, model) in enumerate(self.models.items(), 1):
                if not model.has_parameter(param):
                    continue
                
                try:
                    lon_grid, lat_grid, data_slice = model.get_horizontal_slice(param, depth)
                    
                    ax: GeoAxes = fig.add_subplot(1, n_models, idx, projection=ccrs.PlateCarree())  # type: ignore[assignment]
                    
                    ax.set_extent([lon_grid.min(), lon_grid.max(), 
                                  lat_grid.min(), lat_grid.max()], 
                                 crs=ccrs.PlateCarree())
                    
                    if self.config.slice_params['add_coastlines']:
                        ax.add_feature(cfeature.COASTLINE, 
                                      linewidth=self.config.slice_params['coastline_linewidth'],
                                      edgecolor=self.config.slice_params['coastline_color'],
                                      zorder=5)
                        ax.add_feature(cfeature.BORDERS, 
                                      linewidth=0.3, linestyle=':', 
                                      edgecolor='gray', alpha=0.5, zorder=4)
                    
                    im = ax.pcolormesh(
                        lon_grid, lat_grid, data_slice,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        shading='auto',
                        transform=ccrs.PlateCarree(),
                        zorder=1
                    )
                    
                    gl = ax.gridlines(draw_labels=True, linewidth=0.5, 
                                     color='gray', alpha=0.3, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    
                    cbar = plt.colorbar(
                        im, 
                        ax=ax, 
                        orientation=self.config.slice_params['colorbar_orientation'],
                        pad=self.config.slice_params['colorbar_pad'], 
                        shrink=self.config.slice_params['colorbar_shrink']
                    )
                    cbar.set_label(f'{param.upper()} (km/s)', 
                                  fontsize=self.config.plot_params['font_sizes']['colorbar'])
                    
                    ax.set_title(f"{model.metadata.full_name}",
                                fontsize=self.config.plot_params['font_sizes']['title'],
                                color=model.metadata.color,
                                fontweight='bold')
                    
                except Exception as e:
                    self.logger.warning(f"  绘制 {model.metadata.name} @ {depth}km 失败: {e}")
            
            # 总标题
            plt.suptitle(f'Absolute Velocity Comparison - {param.upper()} @ {depth} km',
                        fontsize=self.config.plot_params['font_sizes']['title']+2,
                        y=0.98,
                        fontweight='bold')
            
            plt.tight_layout(rect=(0, 0, 1, 0.96))
            figures.append(fig)
        
        self.logger.info(f"✅ 成功生成 {len(figures)} 个独立绝对速度对比图")
        return figures
    
    # ==================== 🆕 两两模型差异对比图 ====================
    
    def plot_horizontal_slices_model_differences(self, param: str = 'vs') -> List[Figure]:
        """
        🆕 绘制两两模型间的速度差异对比图（Model_A - Model_B）
        
        功能说明：
        - 每个深度生成一张图
        - 显示所有模型对之间的差异（Model_A - Model_B）
        - 使用RdBu_r色标（红色表示Model_A更快，蓝色表示Model_B更快）
        - 添加海岸线
        - colorbar放置在右侧（垂直方向）
        
        Args:
            param: 要可视化的参数（vs, vp等）
            
        Returns:
            图表列表
        """
        self.logger.info(f"🆕 绘制两两模型差异对比图: {param.upper()}")
        
        depths = self.config.slice_params['horizontal_depths']
        cmap_diff = self.config.slice_params['cmap_difference']
        
        # 获取所有模型键的组合
        model_keys = list(self.models.keys())
        model_pairs = list(combinations(model_keys, 2))
        
        self.logger.info(f"  模型对数量: {len(model_pairs)}")
        for pair in model_pairs:
            m1 = self.models[pair[0]].metadata.full_name
            m2 = self.models[pair[1]].metadata.full_name
            self.logger.info(f"    - {m1} vs {m2}")
        
        figures = []
        
        for depth in depths:
            self.logger.info(f"  生成深度 {depth} km 的差异对比...")
            
            n_pairs = len(model_pairs)
            ncols = min(3, n_pairs)  # 每行最多3个子图
            nrows = (n_pairs + ncols - 1) // ncols
            
            figsize = (5*ncols, 5*nrows)
            fig = plt.figure(figsize=figsize)
            
            # ========== 收集所有差异数据，计算统一colorbar范围 ==========
            all_diffs = []
            
            for model_key_a, model_key_b in model_pairs:
                model_a = self.models[model_key_a]
                model_b = self.models[model_key_b]
                
                if not (model_a.has_parameter(param) and model_b.has_parameter(param)):
                    continue
                
                try:
                    _, _, data_a = model_a.get_horizontal_slice(param, depth)
                    _, _, data_b = model_b.get_horizontal_slice(param, depth)
                    
                    # 计算差异: Model_A - Model_B
                    diff = data_a - data_b
                    all_diffs.append(diff[~np.isnan(diff)])
                except Exception:
                    pass
            
            if len(all_diffs) == 0:
                self.logger.warning(f"  深度 {depth} km 无有效差异数据")
                plt.close(fig)
                continue
            
            all_diffs_flat = np.concatenate(all_diffs)
            diff_max = np.percentile(np.abs(all_diffs_flat), 
                                    self.config.slice_params['diff_percentile'])
            vmin_diff = -diff_max
            vmax_diff = diff_max
            
            # ========== 绘制每对模型的差异 ==========
            for idx, (model_key_a, model_key_b) in enumerate(model_pairs, 1):
                model_a = self.models[model_key_a]
                model_b = self.models[model_key_b]
                
                if not (model_a.has_parameter(param) and model_b.has_parameter(param)):
                    continue
                
                try:
                    lon_grid_a, lat_grid_a, data_a = model_a.get_horizontal_slice(param, depth)
                    lon_grid_b, lat_grid_b, data_b = model_b.get_horizontal_slice(param, depth)
                    
                    # 确保网格一致（插值到统一网格）
                    if not (np.allclose(lon_grid_a, lon_grid_b) and np.allclose(lat_grid_a, lat_grid_b)):
                        self.logger.warning(f"  模型网格不一致，跳过 {model_a.metadata.name} vs {model_b.metadata.name}")
                        continue
                    
                    # 计算差异: Model_A - Model_B
                    diff = data_a - data_b
                    
                    ax: GeoAxes = fig.add_subplot(nrows, ncols, idx, projection=ccrs.PlateCarree())  # type: ignore[assignment]
                    
                    ax.set_extent([lon_grid_a.min(), lon_grid_a.max(), 
                                  lat_grid_a.min(), lat_grid_a.max()], 
                                 crs=ccrs.PlateCarree())
                    
                    if self.config.slice_params['add_coastlines']:
                        ax.add_feature(cfeature.COASTLINE, 
                                      linewidth=self.config.slice_params['coastline_linewidth'],
                                      edgecolor=self.config.slice_params['coastline_color'],
                                      zorder=5)
                        ax.add_feature(cfeature.BORDERS, 
                                      linewidth=0.3, linestyle=':', 
                                      edgecolor='gray', alpha=0.5, zorder=4)
                    
                    im = ax.pcolormesh(
                        lon_grid_a, lat_grid_a, diff,
                        cmap=cmap_diff,
                        vmin=vmin_diff,
                        vmax=vmax_diff,
                        shading='auto',
                        transform=ccrs.PlateCarree(),
                        zorder=1
                    )
                    
                    # 添加网格线
                    gl = ax.gridlines(draw_labels=True, linewidth=0.5, 
                                     color='gray', alpha=0.3, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    
                    # Colorbar（右侧垂直放置）
                    cbar = plt.colorbar(
                        im, 
                        ax=ax, 
                        orientation=self.config.slice_params['colorbar_orientation'],
                        pad=self.config.slice_params['colorbar_pad'], 
                        shrink=self.config.slice_params['colorbar_shrink']
                    )
                    cbar.set_label(f'ΔV (km/s)', 
                                  fontsize=self.config.plot_params['font_sizes']['colorbar'])
                    
                    # 标题（显示模型对）
                    title_text = f"{model_a.metadata.full_name}\nminus\n{model_b.metadata.full_name}"
                    ax.set_title(title_text,
                                fontsize=self.config.plot_params['font_sizes']['title']-1,
                                fontweight='bold')
                    
                    # 统计信息
                    mean_diff = np.nanmean(diff)
                    std_diff = np.nanstd(diff)
                    ax.text(0.02, 0.02, f'Mean: {mean_diff:.3f} km/s\nStd: {std_diff:.3f} km/s',
                           transform=ax.transAxes,
                           fontsize=8,
                           va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', alpha=0.8))
                    
                except Exception as e:
                    self.logger.warning(f"  绘制差异失败 {model_a.metadata.name} vs {model_b.metadata.name}: {e}")
            
            # 总标题
            plt.suptitle(f'Model Difference Comparison - {param.upper()} @ {depth} km\n(Red: Model A faster | Blue: Model B faster)',
                        fontsize=self.config.plot_params['font_sizes']['title']+2,
                        y=0.98,
                        fontweight='bold')
            
            plt.tight_layout(rect=(0, 0, 1, 0.96))
            figures.append(fig)
        
        self.logger.info(f"✅ 成功生成 {len(figures)} 个两两模型差异对比图")
        return figures
    
    # ==================== 水平切片对比（灵活绘图模式）====================
    
    def plot_horizontal_slices_comparison(self, param: str = 'vs') -> List[Figure]:
        """
        绘制水平切片对比图（灵活绘图模式：绝对速度/扰动/两者）
        
        功能增强：
        - 灵活绘图模式：可选择只画绝对速度、只画扰动、或两者都画
        - dlnV单位：百分比 (%)
        - 添加海岸线数据
        - 切片色标：seismic
        - colorbar放置在右侧（垂直方向）
        
        dlnV定义：dlnV = ln(V/V_ref) × 100%
        
        Args:
            param: 要可视化的参数（vs, vp等）
            
        Returns:
            图表列表
        """
        mode = self.config.plot_mode['horizontal_slices']
        self.logger.info(f"🎨 绘制水平切片对比图（模式: {mode}）: {param.upper()}")
        
        depths = self.config.slice_params['horizontal_depths']
        cmap_abs = self.config.slice_params['cmap']
        cmap_dlnv = self.config.slice_params['cmap_perturbation']
        
        figures = []
        
        for depth in depths:
            self.logger.info(f"  生成深度 {depth} km 的水平切片...")
            
            n_models = len(self.models)
            
            # ========== 根据绘图模式决定子图布局 ==========
            if mode == 'both':
                nrows, ncols = 2, n_models
                figsize = self.config.plot_params['figsize_slice_both']
            else:
                nrows, ncols = 1, n_models
                figsize = self.config.plot_params['figsize_slice_single']
            
            fig = plt.figure(figsize=figsize)
            
            # ========== 收集数据，计算统一colorbar范围 ==========
            all_data_abs = []
            all_data_dlnv = []
            
            for model in self.models.values():
                if model.has_parameter(param):
                    try:
                        if mode in ['absolute', 'both']:
                            _, _, data_slice = model.get_horizontal_slice(param, depth)
                            all_data_abs.append(data_slice[~np.isnan(data_slice)])
                        
                        if mode in ['perturbation', 'both']:
                            _, _, dlnv_slice = model.get_horizontal_slice_dlnv(param, depth)
                            all_data_dlnv.append(dlnv_slice[~np.isnan(dlnv_slice)])
                    except Exception:
                        pass
            
            # 检查是否有有效数据
            if mode in ['absolute', 'both'] and len(all_data_abs) == 0:
                self.logger.warning(f"  深度 {depth} km 无绝对速度数据")
                plt.close(fig)
                continue
            
            if mode in ['perturbation', 'both'] and len(all_data_dlnv) == 0:
                self.logger.warning(f"  深度 {depth} km 无扰动数据")
                if mode == 'perturbation':
                    plt.close(fig)
                    continue
            
            # 计算colorbar范围
            if mode in ['absolute', 'both'] and len(all_data_abs) > 0:
                all_data_abs_flat = np.concatenate(all_data_abs)
                vmin_abs = np.percentile(all_data_abs_flat, self.config.slice_params['vmin_percentile'])
                vmax_abs = np.percentile(all_data_abs_flat, self.config.slice_params['vmax_percentile'])
            else:
                vmin_abs, vmax_abs = None, None
            
            if mode in ['perturbation', 'both'] and len(all_data_dlnv) > 0:
                all_data_dlnv_flat = np.concatenate(all_data_dlnv)
                dlnv_max = np.percentile(np.abs(all_data_dlnv_flat), 
                                        self.config.slice_params['dlnv_percentile'])
                vmin_dlnv, vmax_dlnv = -dlnv_max, dlnv_max
            else:
                vmin_dlnv, vmax_dlnv = -10.0, 10.0
            
            # ========== 绘制每个模型 ==========
            for idx, (model_key, model) in enumerate(self.models.items(), 1):
                if not model.has_parameter(param):
                    continue
                
                try:
                    # ========== 绘制绝对速度（第一行或唯一行）==========
                    if mode in ['absolute', 'both']:
                        lon_grid, lat_grid, data_slice = model.get_horizontal_slice(param, depth)
                        
                        ax_abs: GeoAxes = fig.add_subplot(nrows, ncols, idx, projection=ccrs.PlateCarree())  # type: ignore[assignment]
                        
                        ax_abs.set_extent([lon_grid.min(), lon_grid.max(), 
                                          lat_grid.min(), lat_grid.max()], 
                                         crs=ccrs.PlateCarree())
                        
                        if self.config.slice_params['add_coastlines']:
                            ax_abs.add_feature(cfeature.COASTLINE, 
                                              linewidth=self.config.slice_params['coastline_linewidth'],
                                              edgecolor=self.config.slice_params['coastline_color'],
                                              zorder=5)
                            ax_abs.add_feature(cfeature.BORDERS, 
                                              linewidth=0.3, linestyle=':', 
                                              edgecolor='gray', alpha=0.5, zorder=4)
                        
                        im_abs = ax_abs.pcolormesh(
                            lon_grid, lat_grid, data_slice,
                            cmap=cmap_abs,
                            vmin=vmin_abs,
                            vmax=vmax_abs,
                            shading='auto',
                            transform=ccrs.PlateCarree(),
                            zorder=1
                        )
                        
                        gl_abs = ax_abs.gridlines(draw_labels=True, linewidth=0.5, 
                                                 color='gray', alpha=0.3, linestyle='--')
                        gl_abs.top_labels = False
                        gl_abs.right_labels = False
                        
                        cbar_abs = plt.colorbar(
                            im_abs, 
                            ax=ax_abs, 
                            orientation=self.config.slice_params['colorbar_orientation'],
                            pad=self.config.slice_params['colorbar_pad'], 
                            shrink=self.config.slice_params['colorbar_shrink']
                        )
                        cbar_abs.set_label(f'{param.upper()} (km/s)', 
                                          fontsize=self.config.plot_params['font_sizes']['colorbar'])
                        
                        title_text = f"{model.metadata.full_name}"
                        if mode == 'both':
                            title_text += "\nAbsolute Velocity"
                        ax_abs.set_title(title_text,
                                        fontsize=self.config.plot_params['font_sizes']['title'],
                                        color=model.metadata.color,
                                        fontweight='bold')
                    
                    # ========== 绘制dlnV扰动（第二行或唯一行）==========
                    if mode in ['perturbation', 'both']:
                        lon_grid, lat_grid, dlnv_slice = model.get_horizontal_slice_dlnv(param, depth)
                        
                        if mode == 'both':
                            subplot_idx = ncols + idx
                        else:
                            subplot_idx = idx
                        
                        ax_dlnv: GeoAxes = fig.add_subplot(nrows, ncols, subplot_idx, projection=ccrs.PlateCarree())  # type: ignore[assignment]
                        
                        ax_dlnv.set_extent([lon_grid.min(), lon_grid.max(), 
                                           lat_grid.min(), lat_grid.max()], 
                                          crs=ccrs.PlateCarree())
                        
                        if self.config.slice_params['add_coastlines']:
                            ax_dlnv.add_feature(cfeature.COASTLINE, 
                                               linewidth=self.config.slice_params['coastline_linewidth'],
                                               edgecolor=self.config.slice_params['coastline_color'],
                                               zorder=5)
                            ax_dlnv.add_feature(cfeature.BORDERS, 
                                               linewidth=0.3, linestyle=':', 
                                               edgecolor='gray', alpha=0.5, zorder=4)
                        
                        im_dlnv = ax_dlnv.pcolormesh(
                            lon_grid, lat_grid, dlnv_slice,
                            cmap=cmap_dlnv,
                            vmin=vmin_dlnv,
                            vmax=vmax_dlnv,
                            shading='auto',
                            transform=ccrs.PlateCarree(),
                            zorder=1
                        )
                        
                        gl_dlnv = ax_dlnv.gridlines(draw_labels=True, linewidth=0.5, 
                                                   color='gray', alpha=0.3, linestyle='--')
                        gl_dlnv.top_labels = False
                        gl_dlnv.right_labels = False
                        
                        cbar_dlnv = plt.colorbar(
                            im_dlnv, 
                            ax=ax_dlnv, 
                            orientation=self.config.slice_params['colorbar_orientation'],
                            pad=self.config.slice_params['colorbar_pad'], 
                            shrink=self.config.slice_params['colorbar_shrink']
                        )
                        cbar_dlnv.set_label('dlnV (%)', 
                                           fontsize=self.config.plot_params['font_sizes']['colorbar'])
                        
                        if mode == 'both':
                            title_text = "Velocity Perturbation\n(relative to 1D average)"
                        else:
                            title_text = f"{model.metadata.full_name}\nVelocity Perturbation"
                        
                        ax_dlnv.set_title(title_text,
                                         fontsize=self.config.plot_params['font_sizes']['title'],
                                         color=model.metadata.color if mode == 'perturbation' else 'black',
                                         fontweight='bold' if mode == 'perturbation' else 'normal')
                    
                except Exception as e:
                    self.logger.warning(f"  绘制 {model.metadata.name} @ {depth}km 失败: {e}")
            
            # ========== 总标题 ==========
            if mode == 'absolute':
                suptitle = f'Horizontal Slice Comparison - {param.upper()} @ {depth} km\n(Absolute Velocity)'
            elif mode == 'perturbation':
                suptitle = f'Horizontal Slice Comparison - {param.upper()} @ {depth} km\n(dlnV Perturbation in %)'
            else:
                suptitle = f'Horizontal Slice Comparison - {param.upper()} @ {depth} km\n(Absolute Velocity + dlnV Perturbation in %)'
            
            plt.suptitle(suptitle,
                        fontsize=self.config.plot_params['font_sizes']['title']+2,
                        y=0.98,
                        fontweight='bold')
            
            plt.tight_layout(rect=(0, 0, 1, 0.96))
            
            figures.append(fig)
        
        self.logger.info(f"✅ 成功生成 {len(figures)} 个水平切片对比图（模式: {mode}）")
        return figures
    
    # ==================== 垂直剖面对比（保持原有实现）====================
    
    def plot_vertical_profiles_comparison(self, param: str = 'vs') -> List[Figure]:
        """
        绘制垂直剖面对比图（灵活绘图模式：绝对速度/扰动/两者）
        
        功能增强：
        - 灵活绘图模式：可选择只画绝对速度、只画扰动、或两者都画
        - dlnV单位：百分比 (%)
        - 切片色标：seismic
        
        dlnV定义：dlnV = ln(V/V_ref) × 100%
        
        Args:
            param: 要可视化的参数（vs, vp等）
            
        Returns:
            图表列表
        """
        mode = self.config.plot_mode['vertical_profiles']
        self.logger.info(f"🎨 绘制垂直剖面对比图（模式: {mode}）: {param.upper()}")
        
        figures = []
        
        # 1. 经度剖面
        lon_positions = self.config.slice_params['vertical_profiles']['longitude']
        for lon in lon_positions:
            fig = self._plot_single_vertical_profile_flexible(param, 'longitude', lon, mode)
            if fig:
                figures.append(fig)
        
        # 2. 纬度剖面
        lat_positions = self.config.slice_params['vertical_profiles']['latitude']
        for lat in lat_positions:
            fig = self._plot_single_vertical_profile_flexible(param, 'latitude', lat, mode)
            if fig:
                figures.append(fig)
        
        self.logger.info(f"✅ 成功生成 {len(figures)} 个垂直剖面对比图（模式: {mode}）")
        return figures
    
    def _plot_single_vertical_profile_flexible(
        self, param: str, direction: str, position: float, mode: str
    ) -> Optional[Figure]:
        """
        绘制单个垂直剖面对比（灵活绘图模式）
        
        Args:
            param: 参数名称
            direction: 'longitude' 或 'latitude'
            position: 位置值
            mode: 'absolute', 'perturbation', 或 'both'
            
        Returns:
            垂直剖面对比图
        """
        
        n_models = len(self.models)
        
        # ========== 根据绘图模式决定子图布局 ==========
        if mode == 'both':
            nrows, ncols = 2, n_models
            figsize = (5*ncols, 8)
        else:
            nrows, ncols = 1, n_models
            figsize = (5*ncols, 4)
        
        fig = plt.figure(figsize=figsize)
        
        # ========== 收集数据 ==========
        all_data_abs = []
        all_data_dlnv = []
        
        for model in self.models.values():
            if model.has_parameter(param):
                try:
                    if mode in ['absolute', 'both']:
                        _, _, data_profile = model.get_vertical_profile(param, direction, position)
                        all_data_abs.append(data_profile[~np.isnan(data_profile)])
                    
                    if mode in ['perturbation', 'both']:
                        _, _, dlnv_profile = model.get_vertical_profile_dlnv(param, direction, position)
                        all_data_dlnv.append(dlnv_profile[~np.isnan(dlnv_profile)])
                except Exception:
                    pass
        
        # 检查是否有有效数据
        if mode in ['absolute', 'both'] and len(all_data_abs) == 0:
            pos_label = f"Lon={position}°" if direction == 'longitude' else f"Lat={position}°"
            self.logger.warning(f"  {pos_label} 无绝对速度数据")
            plt.close(fig)
            return None
        
        # 计算统一的colorbar范围
        if mode in ['absolute', 'both'] and len(all_data_abs) > 0:
            all_data_abs_flat = np.concatenate(all_data_abs)
            vmin_abs = np.percentile(all_data_abs_flat, self.config.slice_params['vmin_percentile'])
            vmax_abs = np.percentile(all_data_abs_flat, self.config.slice_params['vmax_percentile'])
        else:
            vmin_abs, vmax_abs = None, None
        
        if mode in ['perturbation', 'both'] and len(all_data_dlnv) > 0:
            all_data_dlnv_flat = np.concatenate(all_data_dlnv)
            dlnv_max = np.percentile(np.abs(all_data_dlnv_flat), 
                                    self.config.slice_params['dlnv_percentile'])
            vmin_dlnv, vmax_dlnv = -dlnv_max, dlnv_max
        else:
            vmin_dlnv, vmax_dlnv = -10.0, 10.0
        
        # ========== 绘制每个模型 ==========
        for idx, (model_key, model) in enumerate(self.models.items(), 1):
            if not model.has_parameter(param):
                continue
            
            try:
                pos_label = f"Lon={position}°" if direction == 'longitude' else f"Lat={position}°"
                
                # ========== 绝对速度（第一行或唯一行）==========
                if mode in ['absolute', 'both']:
                    horiz_grid, depth_grid, data_profile = model.get_vertical_profile(
                        param, direction, position
                    )
                    
                    ax_abs = fig.add_subplot(nrows, ncols, idx)
                    
                    im_abs = ax_abs.pcolormesh(
                        horiz_grid, depth_grid, data_profile,
                        cmap=self.config.slice_params['cmap'],
                        vmin=vmin_abs,
                        vmax=vmax_abs,
                        shading='auto'
                    )
                    
                    cbar_abs = plt.colorbar(im_abs, ax=ax_abs)
                    cbar_abs.set_label(f'{param.upper()} (km/s)', 
                                      fontsize=self.config.plot_params['font_sizes']['colorbar'])
                    
                    title_text = f"{model.metadata.full_name}\n{pos_label}"
                    if mode == 'absolute':
                        title_text = f"{model.metadata.full_name}\n{pos_label}"
                    
                    ax_abs.set_title(title_text,
                                    fontsize=self.config.plot_params['font_sizes']['title'],
                                    color=model.metadata.color,
                                    fontweight='bold')
                    
                    if direction == 'longitude':
                        ax_abs.set_xlabel('Latitude (°)', 
                                         fontsize=self.config.plot_params['font_sizes']['xlabel'])
                    else:
                        ax_abs.set_xlabel('Longitude (°)', 
                                         fontsize=self.config.plot_params['font_sizes']['xlabel'])
                    
                    ax_abs.set_ylabel('Depth (km)', 
                                     fontsize=self.config.plot_params['font_sizes']['ylabel'])
                    ax_abs.invert_yaxis()
                    
                    for boundary_depth in [60, 410, 660]:
                        if boundary_depth <= depth_grid.max():
                            ax_abs.axhline(y=boundary_depth, color='white', 
                                          linestyle='--', linewidth=1, alpha=0.7)
                
                # ========== dlnV扰动（第二行或唯一行）==========
                if mode in ['perturbation', 'both']:
                    horiz_grid, depth_grid, dlnv_profile = model.get_vertical_profile_dlnv(
                        param, direction, position
                    )
                    
                    if mode == 'both':
                        subplot_idx = ncols + idx
                    else:
                        subplot_idx = idx
                    
                    ax_dlnv = fig.add_subplot(nrows, ncols, subplot_idx)
                    
                    im_dlnv = ax_dlnv.pcolormesh(
                        horiz_grid, depth_grid, dlnv_profile,
                        cmap=self.config.slice_params['cmap_perturbation'],
                        vmin=vmin_dlnv,
                        vmax=vmax_dlnv,
                        shading='auto'
                    )
                    
                    cbar_dlnv = plt.colorbar(im_dlnv, ax=ax_dlnv)
                    cbar_dlnv.set_label('dlnV (%)',
                                       fontsize=self.config.plot_params['font_sizes']['colorbar'])
                    
                    if mode == 'both':
                        title_text = "Velocity Perturbation"
                    else:
                        title_text = f"{model.metadata.full_name}\n{pos_label}\nVelocity Perturbation"
                    
                    ax_dlnv.set_title(title_text,
                                     fontsize=self.config.plot_params['font_sizes']['title'],
                                     color=model.metadata.color if mode == 'perturbation' else 'black',
                                     fontweight='bold' if mode == 'perturbation' else 'normal')
                    
                    if direction == 'longitude':
                        ax_dlnv.set_xlabel('Latitude (°)', 
                                          fontsize=self.config.plot_params['font_sizes']['xlabel'])
                    else:
                        ax_dlnv.set_xlabel('Longitude (°)', 
                                          fontsize=self.config.plot_params['font_sizes']['xlabel'])
                    
                    ax_dlnv.set_ylabel('Depth (km)', 
                                      fontsize=self.config.plot_params['font_sizes']['ylabel'])
                    ax_dlnv.invert_yaxis()
                    
                    for boundary_depth in [60, 410, 660]:
                        if boundary_depth <= depth_grid.max():
                            ax_dlnv.axhline(y=boundary_depth, color='white', 
                                           linestyle='--', linewidth=1, alpha=0.7)
                
            except Exception as e:
                self.logger.warning(f"  绘制 {model.metadata.name} 剖面失败: {e}")
        
        # ========== 总标题 ==========
        if mode == 'absolute':
            suptitle = f'Vertical Profile Comparison - {param.upper()} @ {pos_label}\n(Absolute Velocity)'
        elif mode == 'perturbation':
            suptitle = f'Vertical Profile Comparison - {param.upper()} @ {pos_label}\n(dlnV Perturbation in %)'
        else:
            suptitle = f'Vertical Profile Comparison - {param.upper()} @ {pos_label}\n(Absolute Velocity + dlnV Perturbation in %)'
        
        plt.suptitle(suptitle,
                    fontsize=self.config.plot_params['font_sizes']['title']+2,
                    y=0.98,
                    fontweight='bold')
        
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        
        return fig
    
    # ==================== 保存结果 ====================
    
    def save_results(self):
        """保存所有分析结果"""
        self.logger.info("\n" + "="*80)
        self.logger.info("💾 保存分析结果...")
        self.logger.info("="*80)
        
        prefix = self.config.output_params['figure_prefix']
        
        try:
            # 1. 速度剖面对比（VS + VP）
            self.logger.info("\n📊 1. 综合速度剖面对比（VS + VP）...")
            fig1 = self.plot_1d_velocity_comparison()
            self._save_figure(fig1, self.output_dir / f"{prefix}velocity_comparison")
            plt.close(fig1)
            
            # 2. S波各向异性对比
            self.logger.info("\n📊 2. S波各向异性对比（VSV实线 vs VSH虚线）...")
            fig2 = self.plot_s_wave_anisotropy_comparison()
            if fig2:
                self._save_figure(fig2, self.output_dir / f"{prefix}s_wave_anisotropy")
                plt.close(fig2)
            
            # 3. P波各向异性对比
            self.logger.info("\n📊 3. P波各向异性对比（VPV实线 vs VPH虚线）...")
            fig3 = self.plot_p_wave_anisotropy_comparison()
            if fig3:
                self._save_figure(fig3, self.output_dir / f"{prefix}p_wave_anisotropy")
                plt.close(fig3)
            
            # 4. 单个模型详细剖面（v2.7 使用 original 数据，展示完整模型区域）
            self.logger.info("\n📊 4. 单个模型详细剖面（original 区域）...")
            for model_key, model in self.models.items():
                fig = self.plot_individual_model_profile(model_key)
                if fig:
                    safe_name = model.metadata.name.replace(' ', '_').replace('.', '_')
                    self._save_figure(fig, 
                                    self.output_dir / f"{prefix}profile_{safe_name}")
                    plt.close(fig)
            
            # 5. 切片位置示意图
            self.logger.info("\n🗺️  5. 切片位置示意图...")
            fig_loc = self.plot_slice_location_map()
            self._save_figure(fig_loc, self.output_dir / f"{prefix}slice_locations")
            plt.close(fig_loc)
            
            # 6. 🆕 独立绝对速度对比图
            if self.config.plot_mode['enable_absolute_only']:
                self.logger.info(f"\n🆕 6. 独立绝对速度对比图（VS）...")
                abs_figs = self.plot_horizontal_slices_absolute_only('vs')
                for i, fig in enumerate(abs_figs):
                    depth = self.config.slice_params['horizontal_depths'][i]
                    self._save_figure(fig, 
                                    self.output_dir / f"{prefix}absolute_vs_{depth}km")
                    plt.close(fig)
            
            # 7. 🆕 两两模型差异对比图
            if self.config.plot_mode['enable_model_differences']:
                self.logger.info(f"\n🆕 7. 两两模型差异对比图（VS）...")
                diff_figs = self.plot_horizontal_slices_model_differences('vs')
                for i, fig in enumerate(diff_figs):
                    depth = self.config.slice_params['horizontal_depths'][i]
                    self._save_figure(fig, 
                                    self.output_dir / f"{prefix}difference_vs_{depth}km")
                    plt.close(fig)
            
            # 8. 水平切片对比图（灵活模式）
            h_mode = self.config.plot_mode['horizontal_slices']
            self.logger.info(f"\n🗺️  8. 水平切片对比（VS，模式: {h_mode}）...")
            slice_figs_vs = self.plot_horizontal_slices_comparison('vs')
            for i, fig in enumerate(slice_figs_vs):
                depth = self.config.slice_params['horizontal_depths'][i]
                self._save_figure(fig, 
                                self.output_dir / f"{prefix}horizontal_slice_vs_{depth}km")
                plt.close(fig)
            
            # 9. 垂直剖面对比图（灵活模式）
            v_mode = self.config.plot_mode['vertical_profiles']
            self.logger.info(f"\n📐 9. 垂直剖面对比（VS，模式: {v_mode}）...")
            profile_figs = self.plot_vertical_profiles_comparison('vs')
            for i, fig in enumerate(profile_figs):
                self._save_figure(fig, 
                                self.output_dir / f"{prefix}vertical_profile_{i+1}")
                plt.close(fig)
            
            # 10. 生成统计报告
            self.logger.info("\n📋 10. 生成统计报告...")
            report = self.generate_comparison_report()
            
            report_file = self.output_dir / f"{prefix}comparison_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self._save_text_report(report, self.output_dir / f"{prefix}comparison_summary.txt")
            
            self.logger.info("\n" + "="*80)
            self.logger.info(f"✅ 所有结果已保存到: {self.output_dir}")
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")
            raise
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """生成详细的对比分析报告"""
        self.logger.info("  生成对比分析报告...")
        
        report = {
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'analysis_version': 'v2.7-DualRegion',
                'n_models': len(self.models),
                'data_source': 'Standardized (comparison) + Original (individual) NetCDF files',
                'perturbation_definition': 'dlnV = ln(V/V_ref) × 100%',
                'difference_definition': 'ΔV = V_A - V_B (km/s)',
                'features': [
                    'Velocity profile comparison (VS + VP, solid lines)',
                    'S-wave anisotropy (VSV solid vs VSH dashed)',
                    'P-wave anisotropy (VPV solid vs VPH dashed)',
                    '🆕 Independent absolute velocity comparison',
                    '🆕 Pairwise model difference comparison (Model_A - Model_B)',
                    'Horizontal slices with coastlines and flexible plot mode',
                    'Vertical profiles with flexible plot mode',
                    'Slice location map showing all profile positions',
                    'dlnV perturbations in percentage (%)',
                    'Model differences in km/s (ΔV)'
                ]
            },
            'models': {},
            'color_scheme': {
                'SinoScope1.0': '#E41A1C (red)',
                'EARA2024': '#028BFB (blue)',
                'FWEA23': '#05F811 (green)'
            },
            'plot_modes': {
                'horizontal_slices': self.config.plot_mode['horizontal_slices'],
                'vertical_profiles': self.config.plot_mode['vertical_profiles'],
                'enable_absolute_only': self.config.plot_mode['enable_absolute_only'],
                'enable_model_differences': self.config.plot_mode['enable_model_differences']
            },
            'visualization_settings': {
                'slice_colormap': self.config.slice_params['cmap'],
                'perturbation_colormap': self.config.slice_params['cmap_perturbation'],
                'difference_colormap': self.config.slice_params['cmap_difference'],
                'coastlines_enabled': self.config.slice_params['add_coastlines'],
                'colorbar_orientation': self.config.slice_params['colorbar_orientation'],
                'colorbar_position': self.config.slice_params['colorbar_position'],
                'dlnv_unit': 'percent (%)',
                'difference_unit': 'km/s'
            }
        }
        
        # 收集各模型信息
        for model_key, model in self.models.items():
            stats = model.get_statistics('vs')
            
            report['models'][model_key] = {
                'full_name': model.metadata.full_name,
                'year': model.metadata.year,
                'reference': model.metadata.reference,
                'color': model.metadata.color,
                'data_shape': model.metadata.data_shape,
                'depth_range': model.metadata.actual_depth_range,
                'spatial_range': model.metadata.actual_spatial_range,
                'resolution': model.metadata.actual_resolution,
                'vs_statistics': stats,
                'has_s_wave_anisotropy': model.has_s_wave_anisotropy(),
                'has_p_wave_anisotropy': model.has_p_wave_anisotropy()
            }
        
        return report
    
    def _save_figure(self, fig: Figure, filepath: Path):
        """保存图表"""
        for fmt in self.config.output_params['save_formats']:
            output_file = filepath.with_suffix(f'.{fmt}')
            fig.savefig(
                output_file,
                dpi=self.config.output_params['save_dpi'],
                bbox_inches=self.config.output_params['save_bbox']
            )
        self.logger.info(f"  ✅ {filepath.name}")
    
    def _save_text_report(self, report: Dict[str, Any], filepath: Path):
        """保存文本格式报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EASTASIA-FWI 速度模型对比分析报告 (v2.7 - 双区域1D对比)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"生成时间: {report['metadata']['generation_time']}\n")
            f.write(f"分析版本: {report['metadata']['analysis_version']}\n")
            f.write(f"数据源: {report['metadata']['data_source']}\n")
            f.write(f"扰动定义: {report['metadata']['perturbation_definition']}\n")
            f.write(f"差异定义: {report['metadata']['difference_definition']}\n")
            f.write(f"对比模型数量: {report['metadata']['n_models']}\n\n")
            
            f.write("🎨 颜色方案:\n")
            for model, color in report['color_scheme'].items():
                f.write(f"  {model}: {color}\n")
            f.write("\n")
            
            f.write("🆕 绘图模式:\n")
            for plot_type, mode in report['plot_modes'].items():
                f.write(f"  {plot_type}: {mode}\n")
            f.write("\n")
            
            f.write("🆕 可视化设置:\n")
            for key, value in report['visualization_settings'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("✨ 核心功能:\n")
            for feature in report['metadata']['features']:
                f.write(f"  • {feature}\n")
            f.write("\n")
            
            # 模型详细信息
            f.write("📋 模型详细信息:\n")
            f.write("-"*80 + "\n")
            for model_key, info in report['models'].items():
                f.write(f"\n{info['full_name']} ({info['year']}):\n")
                f.write(f"  颜色: {info['color']}\n")
                f.write(f"  参考文献: {info['reference']}\n")
                f.write(f"  数据形状: {info['data_shape']}\n")
                f.write(f"  分辨率: {info['resolution']}\n")
                f.write(f"  深度范围: {info['depth_range'][0]:.1f} ~ {info['depth_range'][1]:.1f} km\n")
                
                if info['spatial_range']:
                    spatial = info['spatial_range']
                    f.write(f"  空间范围: {spatial['lat'][0]:.2f}°-{spatial['lat'][1]:.2f}°N, ")
                    f.write(f"{spatial['lon'][0]:.2f}°-{spatial['lon'][1]:.2f}°E\n")
                
                f.write(f"  S波各向异性: {'✓' if info['has_s_wave_anisotropy'] else '✗'}\n")
                f.write(f"  P波各向异性: {'✓' if info['has_p_wave_anisotropy'] else '✗'}\n")
                
                if info['vs_statistics']:
                    vs_stats = info['vs_statistics']
                    f.write(f"  VS统计:\n")
                    f.write(f"    范围: {vs_stats['min']:.3f} ~ {vs_stats['max']:.3f} km/s\n")
                    f.write(f"    平均: {vs_stats['mean']:.3f} ± {vs_stats['std']:.3f} km/s\n")
                    f.write(f"    有效数据: {vs_stats['valid_percentage']:.1f}%\n")
        
        self.logger.info(f"  ✅ {filepath.name}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🚀 EASTASIA-FWI 速度模型对比分析 (v2.7 - 双区域1D对比)")
    print("="*80)
    print("\n🎨 核心功能:")
    print("  1. 综合对比图（所有模型）")
    print("  2. 单个模型详细图")
    print("  3. 切片位置示意图（地理分布）")
    print("  4. 🆕 独立绝对速度对比图（所有模型在同一深度）")
    print("  5. 🆕 两两模型差异图（Model_A - Model_B）")
    print("  6. 灵活绘图模式水平切片（可选：绝对/扰动/两者）")
    print("  7. 灵活绘图模式垂直剖面（可选：绝对/扰动/两者）")
    print("\n🆕 可视化增强:")
    print("  • 灵活绘图模式：'absolute', 'perturbation', 'both'")
    print("  • dlnV扰动单位: 百分比 (%)")
    print("  • 模型差异单位: km/s (ΔV = V_A - V_B)")
    print("  • 切片色标: seismic / RdBu_r")
    print("  • 水平切片colorbar: 右侧垂直放置")
    print("  • 水平切片: 添加海岸线数据")
    print("  • 新增位置示意图: 显示所有切片位置")
    print("\n🔬 地震学标准:")
    print("  • dlnV扰动定义: dlnV = ln(V/V_ref) × 100%")
    print("  • 小扰动近似: dlnV ≈ (V - V_ref)/V_ref × 100% = dV/V × 100%")
    print("  • 正值表示高速异常，负值表示低速异常")
    print("  • 模型差异: ΔV = V_A - V_B (正值表示模型A更快)")
    print("="*80 + "\n")
    
    try:
        comparator = ModelComparator()
        comparator.load_models()
        
        if len(comparator.models) == 0:
            print("❌ 未能加载任何模型")
            return
        
        print(f"\n📊 已加载 {len(comparator.models)} 个NetCDF模型")
        print(f"\n🆕 绘图模式设置:")
        print(f"  水平切片模式: {comparator.config.plot_mode['horizontal_slices']}")
        print(f"  垂直剖面模式: {comparator.config.plot_mode['vertical_profiles']}")
        print(f"  独立绝对速度图: {comparator.config.plot_mode['enable_absolute_only']}")
        print(f"  两两模型差异图: {comparator.config.plot_mode['enable_model_differences']}")
        
        # 执行对比分析
        print("\n🔍 正在执行对比分析...\n")
        comparator.save_results()
        
        print("\n" + "="*80)
        print("✅ 速度模型对比分析完成!")
        print(f"📁 结果保存在: {comparator.output_dir}")
        print("\n📊 生成的文件:")
        print("\n  综合对比图:")
        print("    - 2-1_velocity_comparison.png (VS+VP，实线)")
        print("    - 2-1_s_wave_anisotropy.png (VSV实线 vs VSH虚线)")
        print("    - 2-1_p_wave_anisotropy.png (VPV实线 vs VPH虚线)")
        print("\n  单个模型详细图:")
        for model_key, model in comparator.models.items():
            safe_name = model.metadata.name.replace(' ', '_').replace('.', '_')
            print(f"    - 2-1_profile_{safe_name}.png")
        print("\n  切片位置示意图:")
        print("    - 2-1_slice_locations.png（显示所有剖面位置）")
        
        if comparator.config.plot_mode['enable_absolute_only']:
            print(f"\n  🆕 独立绝对速度对比图:")
            for depth in comparator.config.slice_params['horizontal_depths']:
                print(f"    - 2-1_absolute_vs_{depth}km.png")
        
        if comparator.config.plot_mode['enable_model_differences']:
            print(f"\n  🆕 两两模型差异对比图:")
            for depth in comparator.config.slice_params['horizontal_depths']:
                print(f"    - 2-1_difference_vs_{depth}km.png")
        
        print(f"\n  水平切片（模式: {comparator.config.plot_mode['horizontal_slices']}）:")
        for depth in comparator.config.slice_params['horizontal_depths']:
            print(f"    - 2-1_horizontal_slice_vs_{depth}km.png")
        print(f"\n  垂直剖面（模式: {comparator.config.plot_mode['vertical_profiles']}）:")
        print("    - 2-1_vertical_profile_*.png")
        print("\n  报告:")
        print("    - 2-1_comparison_report.json")
        print("    - 2-1_comparison_summary.txt")
        
        print("\n🆕 新增功能说明:")
        print("    - 独立绝对速度图: 所有模型在同一深度的并排对比")
        print("    - 两两模型差异图: 显示Model_A - Model_B (km/s)")
        print("      • 红色: Model_A更快（高速）")
        print("      • 蓝色: Model_B更快（低速）")
        print("    - dlnV = ln(V/V_ref) × 100%: 对数速度扰动（百分比）")
        print("    - ΔV = V_A - V_B: 绝对速度差异（km/s）")
        print("    - 海岸线: 使用Cartopy添加地理参考")
        print("    - colorbar: 右侧垂直放置")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()