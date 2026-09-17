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
6. ✅ 切片位置示意图（PyGMT 地理参考；出图见 5_12）

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
可视化由 `5_Visualization/5_12_Model_compare.py` 承担（地图 PyGMT，1D Matplotlib），
前缀仍为 `2-1_`，写入 `figures/model_compare`，格式 jpg + pdf（300 dpi）。
本模块负责加载 NetCDF、统计报告与 Markdown 综述。

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
import importlib.util
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import xarray as xr

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
        
        # ============ 论文主图参数（模型 × 深度 的 dlnVs 切片矩阵）============
        self.paper_figure = {
            # 行=模型，列=深度。当前 3 模型 × 4 深度 = 3×4
            'depths': [100, 200, 500, 800],  # km；取最近层
            'param': 'vs',
            'cmap': 'seis',                 # GMT seis：红(低速)→黄→绿→蓝(高速)
            # 色标范围模式：
            #   'per_depth' — 每个深度按自身振幅取对称范围，各列一条 colorbar。
            #     扰动振幅随深度衰减一个量级（100 km 可达 ±6%，800 km 仅 ±1-2%），
            #     全图共用一个范围会让深部两列几乎全是零色，看不出任何结构。
            #     代价是不同列的颜色深浅不可跨深度比较，需在图注说明。
            #   'fixed' — 全图共用 dlnv_limit，可跨深度比较振幅
            'dlnv_limit_mode': 'per_depth',
            'dlnv_limit': 6.0,              # 'fixed' 模式下的 ±范围
            'dlnv_percentile': 98.0,        # 'per_depth' 模式下取该分位数定范围
            'figsize': (11.8, 7.4),         # 英寸：3×4 地图 + 底部 colorbar 行
            'panel_letters': True,
            'include_1d': False,            # True 则左侧再加一栏全深度 1D 剖面
            '1d_sigma': 1.0,                # 1D 阴影为 ±N σ
            '1d_xlim': (3.0, 7.0),          # VS (km/s)
            '1d_params': ('vs',),           # 同轴绘制的 1D 参数
            'save_formats': ['jpg', 'pdf'], # 论文图：jpg + 矢量 pdf，不用 png
            # 绝对波速论文图（paper_vs_maps / paper_vp_maps）
            'abs_cmap': 'seis',             # 与扰动图同一套 GMT seis：红=低速，蓝=高速
            'abs_vmin_percentile': 2.0,
            'abs_vmax_percentile': 98.0,
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
        """关闭NetCDF文件；解释器退出时 netCDF4 可能已拆掉，忽略即可。"""
        ds = getattr(self, 'ds', None)
        if ds is None:
            return
        try:
            ds.close()
        except Exception:
            pass
    
    def __del__(self):
        """析构函数"""
        try:
            self.close()
        except Exception:
            pass

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
        print(f"  • 出图模块: 5_Visualization/5_12_Model_compare.py（PyGMT）")
        print(f"  • dlnV扰动单位: 百分比 (%)")
        print(f"  • 差异单位: km/s (ΔV = V_A - V_B)")
        print(f"  • 地图色标: GMT seis（绝对速度与扰动）；绝对速度取到 0.1 km/s")
        
        print(f"\n输出目录: {self.output_dir}")
        print("-" * 60)
    
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

    def _get_visualizer(self):
        """延迟加载 5_12，避免与 5_12.main 循环导入。"""
        viz_py = project_root / '5_Visualization' / '5_12_Model_compare.py'
        if not viz_py.exists():
            raise FileNotFoundError(f'找不到模型对比可视化模块: {viz_py}')
        mod_name = 'eastasia_model_compare_viz'
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
        else:
            spec = importlib.util.spec_from_file_location(mod_name, viz_py)
            if spec is None or spec.loader is None:
                raise ImportError(f'无法加载可视化模块: {viz_py}')
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
        return mod.ModelCompareVisualizer(self)

    def save_results(self):
        """保存全部分析结果：出图交给 5_12，本模块只写统计报告。"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("💾 保存分析结果（可视化 → 5_12 / PyGMT）...")
        self.logger.info("=" * 80)

        prefix = self.config.output_params['figure_prefix']
        try:
            visualizer = self._get_visualizer()
            visualizer.plot_all()

            self.logger.info("\n📋 生成统计报告...")
            report = self.generate_comparison_report()

            report_file = self.output_dir / f"{prefix}comparison_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            self._save_text_report(report, self.output_dir / f"{prefix}comparison_summary.txt")
            self._save_markdown_report(report, self.output_dir / f"{prefix}model_comparison.md")

            self.logger.info("\n" + "=" * 80)
            self.logger.info(f"✅ 所有结果已保存到: {self.output_dir}")
            self.logger.info("=" * 80)
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
    
    def _dlnv_rms_by_depth(
        self, depths: List[float], param: str = 'vs'
    ) -> Dict[str, Dict[float, float]]:
        """
        逐深度统计各模型 dlnV 的面积加权 RMS（%）。
        
        用 cos(lat) 做面积权重，否则高纬格点会被过度计权。这张表量化了
        "扰动振幅随深度衰减"，也是论文主图逐深度色标的依据。
        
        Args:
            depths: 目标深度列表（km，取最近层）
            param: 速度参数
            
        Returns:
            {model_key: {depth: rms_percent}}，取不到值时该项缺省
        """
        result: Dict[str, Dict[float, float]] = {}
        for model_key, model in self.models.items():
            if not model.has_parameter(param):
                continue
            per_depth: Dict[float, float] = {}
            for depth in depths:
                try:
                    _, lat_grid, dlnv = model.get_horizontal_slice_dlnv(param, depth)
                except Exception as e:
                    self.logger.debug(f"    {model_key} @{depth}km dlnV 取值失败: {e}")
                    continue
                valid = np.isfinite(dlnv)
                if not np.any(valid):
                    continue
                weights = np.cos(np.deg2rad(lat_grid))[valid]
                values = dlnv[valid]
                rms = float(np.sqrt(np.sum(weights * values ** 2) / np.sum(weights)))
                per_depth[float(depth)] = rms
            if per_depth:
                result[model_key] = per_depth
        return result
    
    def _save_markdown_report(self, report: Dict[str, Any], filepath: Path) -> None:
        """
        保存 Markdown 直观对比报告。
        
        报告与图片同目录，因此图片用纯文件名相对引用，GitHub / VSCode /
        Typora 都能直接渲染。只嵌入实际存在的文件，避免死链。
        
        Args:
            report: generate_comparison_report() 的输出
            filepath: 目标 .md 路径
        """
        prefix = str(self.config.output_params['figure_prefix'])
        fig_dir = filepath.parent
        paper_depths = [float(d) for d in self.config.paper_figure['depths']]
        all_depths = list(self.config.slice_params['horizontal_depths'])
        
        def find(stem: str) -> Optional[str]:
            """按 jpg → png → pdf 优先级找已生成的图"""
            for ext in ('jpg', 'png', 'pdf'):
                if (fig_dir / f'{stem}.{ext}').exists():
                    return f'{stem}.{ext}'
            return None
        
        def embed(stem: str, caption: str) -> List[str]:
            """可嵌入的位图直接展示，pdf 只给链接"""
            name = find(stem)
            if name is None:
                return []
            if name.endswith('.pdf'):
                return [f'[{caption}（PDF）]({name})', '']
            return [f'![{caption}]({name})', '', f'*{caption}*', '']
        
        lines: List[str] = []
        model_items = list(self.models.items())
        names = ' / '.join(m.metadata.full_name for _, m in model_items)
        
        # ---------- 标题 ----------
        lines += [
            f'# East Asia Velocity Model Comparison — {names}',
            '',
            f'> 由 `2_Model_space_analysis/2_1_Model_compare.py` 自动生成  ',
            f'> 生成时间: {report["metadata"]["generation_time"][:19].replace("T", " ")}  ',
            f'> 分析版本: {report["metadata"]["analysis_version"]}  ',
            f'> 对比模型: {report["metadata"]["n_models"]} 个',
            '',
            '**约定**',
            '',
            f'- 速度扰动: `{report["metadata"]["perturbation_definition"]}`'
            '，参考剖面为各模型自身的水平平均 1D 剖面',
            f'- 模型差异: `{report["metadata"]["difference_definition"]}`',
            '- 多模型对比统一在 **standardized**（模型交集区域）上进行；'
            '单模型剖面用 **original**（模型完整覆盖）',
            '',
        ]
        
        # ---------- 1. 模型一览 ----------
        lines += [
            '## 1. 模型一览',
            '',
            '分辨率为各模型**原生网格间距**；空间与深度范围来自 standardized 文件，'
            '三者已被 `1_5_Process_velocity_models.py` 裁剪到同一公共区域，'
            '故此列一致，不代表各模型的原始覆盖。',
            '',
            '| 模型 | 年份 | 反演方法 | 原生分辨率 | 深度范围 (km) | '
            '公共区域范围 | 径向各向异性 (S / P) | 参考文献 |',
            '|---|---|---|---|---|---|---|---|',
        ]
        for model_key, model in model_items:
            info = report['models'].get(model_key, {})
            meta = model.metadata
            dr = info.get('depth_range') or (np.nan, np.nan)
            sr = info.get('spatial_range')
            if sr:
                span = (f'{sr["lat"][0]:.1f}–{sr["lat"][1]:.1f}°N, '
                        f'{sr["lon"][0]:.1f}–{sr["lon"][1]:.1f}°E')
            else:
                span = '—'
            aniso = ('✓' if info.get('has_s_wave_anisotropy') else '✗') + ' / ' + \
                    ('✓' if info.get('has_p_wave_anisotropy') else '✗')
            lines.append(
                f'| **{meta.full_name}** | {meta.year} | {meta.method} | '
                f'{info.get("resolution", "—")} | {dr[0]:.0f}–{dr[1]:.0f} | '
                f'{span} | {aniso} | {meta.reference} |'
            )
        lines += ['']
        
        # ---------- 2. VS 统计 ----------
        lines += [
            '## 2. VS 统计（共同区域）',
            '',
            '| 模型 | 最小 (km/s) | 最大 (km/s) | 平均 ± 标准差 (km/s) | 有效格点 |',
            '|---|---|---|---|---|',
        ]
        for model_key, model in model_items:
            s = (report['models'].get(model_key) or {}).get('vs_statistics')
            if not s:
                continue
            lines.append(
                f'| **{model.metadata.full_name}** | {s["min"]:.3f} | '
                f'{s["max"]:.3f} | {s["mean"]:.3f} ± {s["std"]:.3f} | '
                f'{s["valid_percentage"]:.1f}% |'
            )
        lines += [
            '',
            '极小值（< 2 km/s）来自最浅层的水体与沉积层格点，'
            '不参与地幔结构讨论；有效格点比例差异反映各模型在公共区域内的覆盖缺口。',
            '',
        ]
        
        # ---------- 3. 扰动振幅随深度衰减 ----------
        self.logger.info("  统计逐深度 dlnV 面积加权 RMS...")
        rms_table = self._dlnv_rms_by_depth(paper_depths, 'vs')
        if rms_table:
            header = ' | '.join(f'{d:.0f} km' for d in paper_depths)
            lines += [
                '## 3. dlnVS 扰动振幅随深度衰减',
                '',
                '面积加权（cos φ）RMS，单位 %。振幅从上地幔顶部到 800 km '
                '衰减近一个量级，这是论文主图按深度分别定色标的原因。',
                '',
                f'| 模型 | {header} |',
                '|---' * (len(paper_depths) + 1) + '|',
            ]
            for model_key, model in model_items:
                per_depth = rms_table.get(model_key)
                if not per_depth:
                    continue
                cells = ' | '.join(
                    f'{per_depth[d]:.2f}' if d in per_depth else '—'
                    for d in paper_depths
                )
                lines.append(f'| **{model.metadata.full_name}** | {cells} |')
            lines += ['']
            
            # 最深一层的模型间振幅离散度：影响后续加权融合与波形拟合的可比性
            z_deep = paper_depths[-1]
            deep = {
                self.models[k].metadata.full_name: v[z_deep]
                for k, v in rms_table.items() if z_deep in v
            }
            if len(deep) >= 2:
                hi = max(deep, key=lambda k: deep[k])
                lo = min(deep, key=lambda k: deep[k])
                lines += [
                    f'{z_deep:.0f} km 处 **{hi}** 的扰动振幅是 **{lo}** 的 '
                    f'{deep[hi] / max(deep[lo], 1e-6):.1f} 倍。'
                    '深部振幅差异既可能来自真实结构，也可能来自阻尼/正则化强度不同，'
                    '融合加权时不能仅按振幅取舍。',
                    '',
                ]
        
        # ---------- 4. 论文主图 ----------
        depth_str = ' / '.join(f'{d:.0f}' for d in paper_depths)
        paper_block = embed(
            f'{prefix}paper_dlnv_maps',
            f'Paper figure: dlnVS at {depth_str} km '
            f'({len(model_items)} models × {len(paper_depths)} depths)'
        )
        if paper_block:
            mode = str(self.config.paper_figure.get('dlnv_limit_mode', 'per_depth'))
            note = ('每列（深度）按自身振幅取对称色标，故**不同深度之间颜色深浅不可比**'
                    if mode == 'per_depth'
                    else f'全图共用 ±{self.config.paper_figure["dlnv_limit"]:.1f}% 色标，'
                         '深度之间可直接比较振幅')
            lines += [
                '## 4. 论文主图：dlnVS 深度切片矩阵',
                '',
                f'行 = 模型，列 = 深度（{depth_str} km）。各模型保持**原生网格**，'
                'SinoScope1.0 的 1° 块状特征未作平滑，可直接看出分辨率差异。',
                f'色标 GMT `seis`（红 = 低速，蓝 = 高速）；{note}。'
                '**同一列内三模型共用色标**，因此列内的颜色深浅可直接比较模型间振幅强弱。',
                '',
            ] + paper_block
            pdf = fig_dir / f'{prefix}paper_dlnv_maps.pdf'
            if pdf.exists():
                lines += [f'矢量版本: [{pdf.name}]({pdf.name})', '']

        for abs_param, abs_title, sec in (('vs', 'VS', '4b'), ('vp', 'VP', '4c')):
            abs_block = embed(
                f'{prefix}paper_{abs_param}_maps',
                f'Paper figure: {abs_title} at {depth_str} km '
                f'({len(model_items)} models × {len(paper_depths)} depths)'
            )
            if abs_block:
                lines += [
                    f'## {sec}. 论文图：绝对 {abs_title} 深度切片矩阵',
                    '',
                    f'版式与 dlnV 主图相同；色标为绝对 {abs_title}（km/s），'
                    '每列按该深度所有模型的 2–98 分位数定范围，列内三模型可直接比较快慢。',
                    '',
                ] + abs_block
                abs_pdf = fig_dir / f'{prefix}paper_{abs_param}_maps.pdf'
                if abs_pdf.exists():
                    lines += [f'矢量版本: [{abs_pdf.name}]({abs_pdf.name})', '']
        
        # ---------- 5. 1D 剖面与各向异性 ----------
        block: List[str] = []
        block += embed(f'{prefix}velocity_comparison',
                       'Horizontally averaged 1-D VS and VP profiles')
        block += embed(f'{prefix}s_wave_anisotropy',
                       'S-wave radial anisotropy: VSV (solid) vs VSH (dashed)')
        block += embed(f'{prefix}p_wave_anisotropy',
                       'P-wave radial anisotropy: VPV (solid) vs VPH (dashed)')
        if block:
            lines += ['## 5. 一维平均剖面与径向各向异性', ''] + block
        
        # 单模型完整剖面
        indiv: List[str] = []
        for _, model in model_items:
            safe = model.metadata.name.replace(' ', '_').replace('.', '_')
            indiv += embed(f'{prefix}profile_{safe}',
                           f'{model.metadata.full_name}: full-coverage 1-D profile')
        if indiv:
            lines += ['### 5.1 单模型完整覆盖剖面（original 区域）', ''] + indiv
        
        # ---------- 6. 水平切片 ----------
        lines += ['## 6. 水平切片逐深度对比', '']
        loc = embed(f'{prefix}slice_locations',
                    'Locations of all horizontal slices and vertical profiles')
        if loc:
            lines += loc
        for depth in paper_depths:
            d = int(round(depth))
            sub: List[str] = []
            sub += embed(f'{prefix}absolute_vs_{d}km',
                         f'Absolute VS at {d} km')
            sub += embed(f'{prefix}horizontal_slice_vs_{d}km',
                         f'VS slice at {d} km (absolute + perturbation)')
            sub += embed(f'{prefix}difference_vs_{d}km',
                         f'Pairwise VS difference at {d} km')
            if sub:
                lines += [f'### 6.{paper_depths.index(depth) + 1} {d} km', ''] + sub
        
        others = [d for d in all_depths if float(d) not in paper_depths]
        if others:
            links = []
            for d in others:
                for stem, tag in (
                    (f'{prefix}absolute_vs_{d}km', 'abs'),
                    (f'{prefix}horizontal_slice_vs_{d}km', 'slice'),
                    (f'{prefix}difference_vs_{d}km', 'diff'),
                ):
                    name = find(stem)
                    if name:
                        links.append(f'[{d} km {tag}]({name})')
            if links:
                lines += [f'### 6.{len(paper_depths) + 1} 其余深度（仅链接）', '',
                          ' · '.join(links), '']
        
        # ---------- 7. 垂直剖面 ----------
        vert: List[str] = []
        for i in range(1, len(self.config.slice_params['vertical_profiles']['longitude'])
                       + len(self.config.slice_params['vertical_profiles']['latitude']) + 1):
            name = find(f'{prefix}vertical_profile_{i}')
            if name is None:
                continue
            if i <= 2:
                vert += embed(f'{prefix}vertical_profile_{i}',
                              f'Vertical cross-section #{i}')
            else:
                vert.append(f'- [Vertical cross-section #{i}]({name})')
        if vert:
            lines += ['## 7. 垂直剖面', ''] + vert + ['']
        
        # ---------- 8. 阅读提示 ----------
        lines += [
            '## 8. 阅读提示',
            '',
            '- **不要跨深度比较颜色深浅**（逐深度色标模式下），'
            '振幅对比请看第 3 节的 RMS 表',
            '- 各模型参考 1D 剖面不同（PREM / ak135 系列），'
            '扰动图之间的零点因此不完全等价；跨模型融合前需统一到同一参考模型',
            '- 分辨率差异会直接体现为切片的"块状"程度，'
            '不代表结构本身的差异；定量的结构相似性见 `2_2_Model_similarity.py`',
            '- 深度参照面（地表 vs 海平面）在各原始模型中定义不同，'
            '浅部（< 40 km）对比需谨慎',
            '',
            '---',
            '',
            f'完整数值见 [`{prefix}comparison_report.json`]'
            f'({prefix}comparison_report.json)、'
            f'[`{prefix}comparison_summary.txt`]({prefix}comparison_summary.txt)。',
            '',
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
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
    print("  • 出图: 5_Visualization/5_12_Model_compare.py（地图 PyGMT）")
    print("  • 切片色标: GMT seis（绝对速度与扰动）；绝对速度取到 0.1 km/s")
    print("  • 输出: jpg + pdf，300 dpi")
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
        print("    - 2-1_velocity_comparison.jpg / .pdf (VS+VP，实线)")
        print("    - 2-1_s_wave_anisotropy.jpg / .pdf (VSV实线 vs VSH虚线)")
        print("    - 2-1_p_wave_anisotropy.jpg / .pdf (VPV实线 vs VPH虚线)")
        print("\n  单个模型详细图:")
        for model_key, model in comparator.models.items():
            safe_name = model.metadata.name.replace(' ', '_').replace('.', '_')
            print(f"    - 2-1_profile_{safe_name}.jpg / .pdf")
        print("\n  切片位置示意图:")
        print("    - 2-1_slice_locations.jpg / .pdf")
        
        if comparator.config.plot_mode['enable_absolute_only']:
            print(f"\n  独立绝对速度对比图:")
            for depth in comparator.config.slice_params['horizontal_depths']:
                print(f"    - 2-1_absolute_vs_{depth}km.jpg / .pdf")
        
        if comparator.config.plot_mode['enable_model_differences']:
            print(f"\n  两两模型差异对比图:")
            for depth in comparator.config.slice_params['horizontal_depths']:
                print(f"    - 2-1_difference_vs_{depth}km.jpg / .pdf")
        
        print(f"\n  水平切片（模式: {comparator.config.plot_mode['horizontal_slices']}）:")
        for depth in comparator.config.slice_params['horizontal_depths']:
            print(f"    - 2-1_horizontal_slice_vs_{depth}km.jpg / .pdf")
        print(f"\n  垂直剖面（模式: {comparator.config.plot_mode['vertical_profiles']}）:")
        print("    - 2-1_vertical_profile_*.jpg / .pdf")
        paper_depths = comparator.config.paper_figure['depths']
        print(f"\n  论文主图（{len(comparator.models)}模型 × {len(paper_depths)}深度）:")
        print(f"    - 2-1_paper_dlnv_maps.jpg / .pdf  ({paper_depths} km)")
        print(f"    - 2-1_paper_vs_maps.jpg / .pdf  ({paper_depths} km)")
        print(f"    - 2-1_paper_vp_maps.jpg / .pdf  ({paper_depths} km)")
        print("\n  报告:")
        print("    - 2-1_model_comparison.md  ← 直观对比（含图表，可直接浏览）")
        print("    - 2-1_comparison_report.json")
        print("    - 2-1_comparison_summary.txt")
        
        print("\n说明:")
        print("    - 可视化已拆至 5_12_Model_compare.py，地图用 PyGMT")
        print("    - dlnV = ln(V/V_ref) × 100%: 对数速度扰动（百分比）")
        print("    - ΔV = V_A - V_B: 绝对速度差异（km/s）")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()