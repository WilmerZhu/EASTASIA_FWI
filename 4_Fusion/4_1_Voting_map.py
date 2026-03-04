"""
EASTASIA-FWI 速度模型投票图（Voting Map）分析模块 (v2.1 - 增强版)
====================================================================

核心改进 (v2.1):
1. ✅ 使用模型原始数据 (*_original.nc)，保留完整覆盖范围
2. ✅ 统一网格到研究区域（-15°~60°N, 60°~170°E, 0~1000km）
3. ✅ Median投票智能处理不同模型的覆盖差异（部分NaN）
4. ✅ 与2_Model_space_analysis区分：
   - 2_Model_space_analysis: 共同覆盖区域（聚类/SSIM）
   - 4_Assembly: 原始数据，融合为研究区域集成模型
5. 🆕 不确定性量化增强：Bootstrap置信区间、模型间相关性分析
6. 🆕 空间平滑选项：高斯平滑减少模型边界不连续
7. 🆕 可视化增强：模型差异对比图、垂直剖面图、改进的综合面板

科学原理：
- Median投票: v_median = median(v1, v2, ..., vn)
  对异常值鲁棒，智能处理部分NaN
- 变异系数: CV = σ/μ × 100%
- 模型一致性指数: MAI = 1 - CV/100
- Bootstrap置信区间: 重采样估计不确定性
- 模型间相关性: 避免相似模型重复计票

适用场景：
- 构建研究区域的"超级模型"（Super Model）
- 利用不同模型的优势区域
- SPECFEM3D正演模拟的初始模型

作者：EASTASIA-FWI Team
日期：2025-01-20
版本：v2.1 (增强版)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
from dataclasses import dataclass
import json
import sys
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

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
    color: str
    weight: float = 1.0
    confidence: float = 1.0
    
    # 原始NetCDF元数据
    original_netcdf_file: Optional[str] = None
    original_resolution: Optional[str] = None
    original_depth_range: Optional[Tuple[float, float]] = None
    original_spatial_range: Optional[Dict[str, Tuple[float, float]]] = None
    original_data_shape: Optional[Tuple[int, int, int]] = None


@dataclass
class BootstrapResult:
    """Bootstrap分析结果"""
    median: np.ndarray
    ci_lower: np.ndarray  # 置信区间下界
    ci_upper: np.ndarray  # 置信区间上界
    ci_width: np.ndarray  # 置信区间宽度
    confidence_level: float = 0.95
    n_bootstrap: int = 1000


class VotingMapConfig:
    """投票图配置类（v2.1 - 增强版）"""
    
    def __init__(self):
        """初始化配置参数"""
        
        # ============ 目标模型配置（使用原始数据）============
        self.target_models = {
            '2022_SinoScope1.0': {
                'original_netcdf_file': '2022_SinoScope1.0_original.nc',
                'processed_dir': 'processed/2022_SinoScope1.0',
                'metadata': ModelMetadata(
                    name="SinoScope1.0",
                    full_name="SinoScope 1.0",
                    year=2022,
                    reference="Ma et al., 2022, GRL",
                    color="#E41A1C",
                    weight=1.0,
                    confidence=0.8
                )
            },
            '2024_EARA2024': {
                'original_netcdf_file': '2024_EARA2024_original.nc',
                'processed_dir': 'processed/2024_EARA2024',
                'metadata': ModelMetadata(
                    name="EARA2024",
                    full_name="EARA2024",
                    year=2024,
                    reference="Xi et al., 2024, GJI",
                    color="#028BFB",
                    weight=1.0,
                    confidence=0.90
                )
            },
            '2024_FWEA23': {
                'original_netcdf_file': '2024_FWEA23_original.nc',
                'processed_dir': 'processed/2024_FWEA23',
                'metadata': ModelMetadata(
                    name="FWEA23",
                    full_name="FWEA23",
                    year=2024,
                    reference="Liu et al., 2024, EPSL",
                    color="#05F811",
                    weight=1.0,
                    confidence=0.9
                )
            }
        }
        
        # ============ 研究区域统一网格 ============
        self.unified_grid = {
            'lat_min': -15.0,
            'lat_max': 60.0,
            'lon_min': 60.0,
            'lon_max': 170.0,
            'depth_min': 0.0,
            'depth_max': 1000.0,
            'lat_step': 0.25,
            'lon_step': 0.25,
            'depth_step': 10.0,
            'interpolation_method': 'linear',
            'allow_partial_coverage': True,
            'min_models_required': 1
        }
        
        # ============ 投票策略配置 ============
        self.voting_strategies = {
            'median': {
                'enabled': True,
                'name': 'Median Voting',
                'description': 'Robust to outliers and partial coverage',
                'handle_nan': 'ignore'
            },
            'mean': {
                'enabled': True,
                'name': 'Mean Voting',
                'description': 'Simple average, sensitive to outliers',
                'handle_nan': 'ignore'
            },
            'weighted_mean': {
                'enabled': True,
                'name': 'Weighted Mean',
                'description': 'Weighted by model confidence',
                'handle_nan': 'ignore'
            }
        }
        
        # ============ 🆕 空间平滑配置 ============
        self.spatial_smoothing = {
            'enabled': True,
            'sigma_horizontal': 1.0,  # 水平方向高斯核标准差（网格点数）
            'sigma_vertical': 0.5,    # 垂直方向高斯核标准差
            'apply_to': ['median', 'mean', 'weighted_mean'],
            'preserve_nan': True      # 保持NaN区域
        }
        
        # ============ 🆕 不确定性量化增强配置 ============
        self.uncertainty = {
            'bootstrap': {
                'enabled': False,
                'n_bootstrap': 50,       # Bootstrap重采样次数
                'confidence_level': 0.95,  # 置信水平
                'random_seed': 42
            },
            'model_correlation': {
                'enabled': True,
                'method': 'pearson',       # 'pearson' 或 'spearman'
                'sample_points': 10000     # 相关性计算采样点数
            },
            'coefficient_of_variation': True,
            'model_agreement_index': True,
            'coverage_count': True
        }
        
        # ============ 可视化参数（增强版）============
        self.visualization = {
            # 深度切片设置
            'horizontal_depths': [20, 50, 100, 200, 400, 660, 800, 1000],
            'comparison_depths': [50, 200, 500],
            'parameters': ['vs', 'vp'],
            
            # 🆕 垂直剖面设置
            'vertical_profiles': {
                'enabled': True,
                'lon_profiles': [90, 105, 120, 135],  # 经度剖面
                'lat_profiles': [0, 20, 35, 45],       # 纬度剖面
            },
            
            # 🆕 模型差异对比设置
            'model_difference': {
                'enabled': True,
                'reference': 'median',  # 参考模型：'median' 或具体模型名
                'depths': [100, 410, 660],
            },
            
            # 配色方案
            'cmap_velocity': 'jet_r',
            'cmap_perturbation': 'RdBu_r',
            'cmap_uncertainty': 'YlOrRd',
            'cmap_agreement': 'RdYlGn',
            'cmap_coverage': 'Blues',
            'cmap_correlation': 'coolwarm',
            
            # 地图元素
            'add_coastlines': True,
            'coastline_color': 'black',
            'coastline_linewidth': 0.5,
            'add_boundaries': True,
            
            # 🆕 地震学重要界面
            'seismic_boundaries': {
                'Moho': 35,
                'LAB': 100,
                '410': 410,
                '660': 660
            },
            
            # 图幅尺寸
            'figsize_horizontal': (15, 10),
            'figsize_comprehensive': (22, 16),
            'figsize_comparison': (20, 8),
            'figsize_coverage': (18, 12),
            'figsize_1d': (16, 10),
            'figsize_vertical_profile': (18, 8),
            'figsize_model_diff': (20, 12),
            'figsize_correlation': (12, 10),
            'figsize_bootstrap': (18, 12),
            'dpi': 300
        }
        
        # ============ 输出配置 ============
        self.output = {
            'save_voting_models': True,
            'save_smoothed_models': True,
            'save_coverage_maps': True,
            'save_uncertainty_maps': True,
            'save_bootstrap_results': True,
            'save_statistics': True,
            'figure_prefix': '4-1_',
            'formats': ['jpg', 'pdf'],
            'dpi': 300,
            'pdf_dpi': 300
        }
        
        # ============ 日志配置 ============
        self.logging = {
            'level': 'INFO',
            'console_output': True
        }


class VelocityModelOriginal:
    """基于原始NetCDF的速度模型数据类"""
    
    def __init__(self, netcdf_path: Path, metadata: ModelMetadata, logger):
        """初始化模型（加载原始数据）"""
        self.path = netcdf_path
        self.metadata = metadata
        self.logger = logger
        self.ds: Optional[xr.Dataset] = None
        
        self._lon = None
        self._lat = None
        self._depth = None
        
        self.load_netcdf()
    
    def load_netcdf(self):
        """加载原始NetCDF数据"""
        try:
            self.logger.info(f"📦 加载原始NetCDF: {self.metadata.name}")
            self.logger.info(f"   文件: {self.path.name}")
            
            self.ds = xr.open_dataset(self.path)
            
            self._lon = self.ds['longitude'].values
            self._lat = self.ds['latitude'].values
            self._depth = self.ds['depth'].values
            
            self.metadata.original_data_shape = (len(self._lat), len(self._lon), len(self._depth))
            self.metadata.original_depth_range = (float(self._depth.min()), float(self._depth.max()))
            self.metadata.original_spatial_range = {
                'lat': (float(self._lat.min()), float(self._lat.max())),
                'lon': (float(self._lon.min()), float(self._lon.max()))
            }
            
            lat_res = np.mean(np.diff(np.sort(self._lat)))
            lon_res = np.mean(np.diff(np.sort(self._lon)))
            self.metadata.original_resolution = f"{lat_res:.2f}°×{lon_res:.2f}°"
            
            self.logger.info(f"   ✅ 加载成功: {self.metadata.original_data_shape}")
            self.logger.info(f"   分辨率: {self.metadata.original_resolution}")
            self.logger.info(f"   纬度: {self._lat.min():.2f}° ~ {self._lat.max():.2f}°")
            self.logger.info(f"   经度: {self._lon.min():.2f}° ~ {self._lon.max():.2f}°")
            self.logger.info(f"   深度: {self._depth.min():.1f} ~ {self._depth.max():.1f} km")
            
        except Exception as e:
            self.logger.error(f"❌ NetCDF加载失败: {e}")
            raise
    
    @property
    def lon(self) -> np.ndarray:
        return self._lon
    
    @property
    def lat(self) -> np.ndarray:
        return self._lat
    
    @property
    def depth(self) -> np.ndarray:
        return self._depth
    
    def get_parameter(self, param: str) -> np.ndarray:
        """获取指定参数的3D数组 (lat, lon, depth)"""
        if param not in self.ds:
            raise ValueError(f"参数 {param} 不存在于模型中")
        
        data = self.ds[param].values
        data_transposed = np.transpose(data, (1, 0, 2))
        return data_transposed
    
    def has_parameter(self, param: str) -> bool:
        """检查是否有指定参数"""
        return param in self.ds
    
    def close(self):
        """关闭NetCDF文件"""
        if self.ds is not None:
            self.ds.close()

# ==================== 主分析类（第一部分：初始化）====================

class VotingMapAnalyzer:
    """速度模型投票图分析器 (v2.1 - 增强版)"""
    
    def __init__(self):
        """初始化分析器"""
        self.base_config = BaseConfig()
        self.config = VotingMapConfig()
        
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.VotingMap',
            self.config.logging['level']
        )
        
        # 模型存储
        self.models: Dict[str, VelocityModelOriginal] = {}
        
        # 统一网格
        self.unified_grid_coords = None
        
        # 插值后的模型数据
        self.interpolated_models: Dict[str, Dict[str, np.ndarray]] = {}
        
        # 投票结果（包括平滑版本）
        self.voting_results: Dict[str, Dict[str, np.ndarray]] = {}
        self.voting_results_smoothed: Dict[str, Dict[str, np.ndarray]] = {}
        
        # 一致性指标
        self.consistency_metrics: Dict[str, Dict[str, np.ndarray]] = {}
        
        # 覆盖度统计
        self.coverage_count: Dict[str, np.ndarray] = {}
        
        # Bootstrap结果
        self.bootstrap_results: Dict[str, BootstrapResult] = {}
        
        # 模型相关性矩阵
        self.model_correlation: Dict[str, np.ndarray] = {}
        
        # 输出目录
        self.output_dir = self.base_config.dirs['figures'] / 'model_voting_map'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = self.base_config.dirs['results'] / 'model_voting_map'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_plotting_style()
        
        self.logger.info("="*80)
        self.logger.info("🚀 速度模型投票图分析器初始化 (v2.1 - 增强版)")
        self.logger.info("="*80)
        self._print_config_summary()
    
    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 投票图分析配置 (v2.1 - 增强版)")
        print("-" * 60)
        print(f"🔥 核心功能:")
        print(f"   • 使用模型原始数据（*_original.nc）")
        print(f"   • 多种投票策略（Median/Mean/Weighted Mean）")
        print(f"   • 空间平滑选项")
        print(f"   • Bootstrap置信区间")
        print(f"   • 模型间相关性分析")
        
        print(f"\n目标模型数量: {len(self.config.target_models)}")
        for key, info in self.config.target_models.items():
            meta = info['metadata']
            print(f"   - {meta.full_name} (置信度: {meta.confidence})")
        
        print(f"\n🗺️ 研究区域统一网格:")
        grid = self.config.unified_grid
        print(f"   纬度: {grid['lat_min']}° ~ {grid['lat_max']}° (步长: {grid['lat_step']}°)")
        print(f"   经度: {grid['lon_min']}° ~ {grid['lon_max']}° (步长: {grid['lon_step']}°)")
        print(f"   深度: {grid['depth_min']} ~ {grid['depth_max']} km (步长: {grid['depth_step']} km)")
        
        n_lat = int((grid['lat_max'] - grid['lat_min']) / grid['lat_step']) + 1
        n_lon = int((grid['lon_max'] - grid['lon_min']) / grid['lon_step']) + 1
        n_depth = int((grid['depth_max'] - grid['depth_min']) / grid['depth_step']) + 1
        print(f"   网格点数: {n_lat} × {n_lon} × {n_depth} = {n_lat*n_lon*n_depth:,}")
        
        print(f"\n🗳️ 投票策略:")
        for key, strategy in self.config.voting_strategies.items():
            if strategy['enabled']:
                print(f"   ✓ {strategy['name']}")
        
        print(f"\n📊 不确定性量化:")
        if self.config.uncertainty['bootstrap']['enabled']:
            print(f"   ✓ Bootstrap ({self.config.uncertainty['bootstrap']['n_bootstrap']} 次重采样)")
        if self.config.uncertainty['model_correlation']['enabled']:
            print(f"   ✓ 模型相关性分析 ({self.config.uncertainty['model_correlation']['method']})")
        
        print(f"\n🔧 空间平滑:")
        if self.config.spatial_smoothing['enabled']:
            print(f"   ✓ 水平sigma: {self.config.spatial_smoothing['sigma_horizontal']}")
            print(f"   ✓ 垂直sigma: {self.config.spatial_smoothing['sigma_vertical']}")
        else:
            print(f"   ✗ 未启用")
        
        print(f"\n输出目录:")
        print(f"   图表: {self.output_dir}")
        print(f"   数据: {self.results_dir}")
        print("-" * 60)
    
    def _setup_plotting_style(self):
        """设置绘图样式"""
        plt.rcParams.update({
            'font.family': ['Arial', 'DejaVu Sans', 'sans-serif'],
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'figure.dpi': self.config.visualization['dpi']
        })
    
    def load_models(self):
        """加载所有原始NetCDF模型"""
        self.logger.info("\n" + "="*80)
        self.logger.info("📦 加载原始NetCDF模型文件")
        self.logger.info("="*80)
        
        models_dir = self.base_config.dirs['data'] / 'fwi-models'
        
        for model_key, model_info in tqdm(self.config.target_models.items(), 
                                          desc="加载模型"):
            netcdf_path = models_dir / model_info['processed_dir'] / model_info['original_netcdf_file']
            
            if not netcdf_path.exists():
                self.logger.warning(f"⚠️ NetCDF文件不存在: {netcdf_path}")
                continue
            
            try:
                model = VelocityModelOriginal(
                    netcdf_path,
                    model_info['metadata'],
                    self.logger
                )
                self.models[model_key] = model
                
            except Exception as e:
                self.logger.error(f"❌ 加载失败 {model_key}: {e}")
        
        self.logger.info(f"\n✅ 成功加载 {len(self.models)} 个原始NetCDF模型")
        
        if len(self.models) == 0:
            raise RuntimeError("未能加载任何模型")
    
    def create_unified_grid(self):
        """创建研究区域统一网格"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🗺️ 创建研究区域统一网格")
        self.logger.info("="*80)
        
        grid_cfg = self.config.unified_grid
        
        lat_grid = np.arange(grid_cfg['lat_min'], 
                            grid_cfg['lat_max'] + grid_cfg['lat_step'], 
                            grid_cfg['lat_step'])
        lon_grid = np.arange(grid_cfg['lon_min'], 
                            grid_cfg['lon_max'] + grid_cfg['lon_step'], 
                            grid_cfg['lon_step'])
        depth_grid = np.arange(grid_cfg['depth_min'], 
                              grid_cfg['depth_max'] + grid_cfg['depth_step'], 
                              grid_cfg['depth_step'])
        
        self.unified_grid_coords = {
            'lat': lat_grid,
            'lon': lon_grid,
            'depth': depth_grid
        }
        
        grid_shape = (len(lat_grid), len(lon_grid), len(depth_grid))
        
        self.logger.info(f"✅ 研究区域网格创建成功")
        self.logger.info(f"   纬度点数: {len(lat_grid)}")
        self.logger.info(f"   经度点数: {len(lon_grid)}")
        self.logger.info(f"   深度点数: {len(depth_grid)}")
        self.logger.info(f"   网格总形状: {grid_shape}")
        self.logger.info(f"   网格总点数: {np.prod(grid_shape):,}")
    
    def interpolate_models_to_unified_grid(self, param: str = 'vs'):
        """将原始模型插值到研究区域统一网格"""
        self.logger.info("\n" + "="*80)
        self.logger.info(f"🔄 原始模型插值到研究区域网格: {param.upper()}")
        self.logger.info("="*80)
        
        if self.unified_grid_coords is None:
            raise RuntimeError("统一网格未创建")
        
        lat_target = self.unified_grid_coords['lat']
        lon_target = self.unified_grid_coords['lon']
        depth_target = self.unified_grid_coords['depth']
        
        lat_3d, lon_3d, depth_3d = np.meshgrid(
            lat_target, lon_target, depth_target, indexing='ij'
        )
        
        for model_key, model in tqdm(self.models.items(), desc="插值模型"):
            if not model.has_parameter(param):
                self.logger.warning(f"   ⚠️ {model.metadata.name} 无 {param.upper()} 参数")
                continue
            
            try:
                data_original = model.get_parameter(param)
                
                # 创建插值器
                interpolator = RegularGridInterpolator(
                    (model.lat, model.lon, model.depth),
                    data_original,
                    method=self.config.unified_grid['interpolation_method'],
                    bounds_error=False,
                    fill_value=np.nan
                )
                
                # 插值到统一网格
                points = np.stack([lat_3d.ravel(), lon_3d.ravel(), depth_3d.ravel()], axis=-1)
                data_interpolated = interpolator(points).reshape(lat_3d.shape)
                
                # 存储
                if model_key not in self.interpolated_models:
                    self.interpolated_models[model_key] = {}
                
                self.interpolated_models[model_key][param] = data_interpolated
                
                # 统计
                total_points = data_interpolated.size
                valid_points = np.sum(~np.isnan(data_interpolated))
                valid_ratio = valid_points / total_points * 100
                
                self.logger.info(f"   ✅ {model.metadata.name}: {valid_ratio:.1f}% 有效覆盖")
                
            except Exception as e:
                self.logger.error(f"   ❌ 插值失败 {model.metadata.name}: {e}")
        
        self.logger.info(f"\n✅ 插值完成，共 {len(self.interpolated_models)} 个模型")
    
    def compute_voting_results(self, param: str = 'vs'):
        """计算投票结果（多种策略）"""
        self.logger.info("\n" + "="*80)
        self.logger.info(f"🗳️ 计算投票结果: {param.upper()}")
        self.logger.info("="*80)
        
        if not self.interpolated_models:
            raise RuntimeError("模型未插值")
        
        # 收集模型数据
        model_data_list = []
        model_weights = []
        model_names = []
        
        for model_key, model_data in self.interpolated_models.items():
            if param in model_data:
                model_data_list.append(model_data[param])
                meta = self.config.target_models[model_key]['metadata']
                model_weights.append(meta.confidence)
                model_names.append(meta.name)
        
        if len(model_data_list) == 0:
            raise RuntimeError(f"无可用的 {param} 数据")
        
        model_stack = np.stack(model_data_list, axis=0)
        weights_array = np.array(model_weights)
        
        self.logger.info(f"   参与投票的模型: {len(model_names)}")
        for name, weight in zip(model_names, model_weights):
            self.logger.info(f"      - {name} (权重: {weight})")
        
        if param not in self.voting_results:
            self.voting_results[param] = {}
        
        # 1. Median投票
        if self.config.voting_strategies['median']['enabled']:
            self.logger.info("\n   📊 计算中位数（Median Voting）...")
            median = np.nanmedian(model_stack, axis=0)
            self.voting_results[param]['median'] = median
            self._log_voting_stats('Median', median)
        
        # 2. Mean投票
        if self.config.voting_strategies['mean']['enabled']:
            self.logger.info("\n   📊 计算均值（Mean Voting）...")
            mean = np.nanmean(model_stack, axis=0)
            self.voting_results[param]['mean'] = mean
            self._log_voting_stats('Mean', mean)
        
        # 3. 加权均值投票
        if self.config.voting_strategies['weighted_mean']['enabled']:
            self.logger.info("\n   📊 计算加权均值（Weighted Mean Voting）...")
            weighted_mean = self._compute_weighted_mean(model_stack, weights_array)
            self.voting_results[param]['weighted_mean'] = weighted_mean
            self._log_voting_stats('Weighted Mean', weighted_mean)
        
        # 统计覆盖度
        coverage_count = np.sum(~np.isnan(model_stack), axis=0)
        self.coverage_count[param] = coverage_count
        
        self.logger.info(f"\n   覆盖度统计:")
        total_points = coverage_count.size
        for n in range(len(model_names) + 1):
            n_points = np.sum(coverage_count == n)
            ratio = n_points / total_points * 100
            self.logger.info(f"      {n}个模型覆盖: {ratio:.1f}%")
        
        self.logger.info(f"\n✅ 投票结果计算完成")
    
    def _compute_weighted_mean(self, model_stack: np.ndarray, 
                               weights: np.ndarray) -> np.ndarray:
        """计算加权均值（处理NaN）"""
        # 扩展权重维度以匹配数据
        weights_expanded = weights[:, np.newaxis, np.newaxis, np.newaxis]
        weights_3d = np.broadcast_to(weights_expanded, model_stack.shape)
        
        # 创建mask
        mask = ~np.isnan(model_stack)
        
        # 计算加权和
        weighted_sum = np.nansum(model_stack * weights_3d * mask, axis=0)
        weight_sum = np.sum(weights_3d * mask, axis=0)
        
        # 避免除零
        weight_sum[weight_sum == 0] = np.nan
        
        return weighted_sum / weight_sum
    
    def _log_voting_stats(self, name: str, data: np.ndarray):
        """记录投票统计信息"""
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            self.logger.info(f"      ✅ {name}: {np.min(valid_data):.3f} ~ {np.max(valid_data):.3f} km/s")
            self.logger.info(f"         有效点: {len(valid_data):,} ({len(valid_data)/data.size*100:.1f}%)")
    
    def apply_spatial_smoothing(self, param: str = 'vs'):
        """应用空间平滑"""
        if not self.config.spatial_smoothing['enabled']:
            self.logger.info("   ⏭️ 空间平滑未启用，跳过")
            return
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"🔧 应用空间平滑: {param.upper()}")
        self.logger.info("="*80)
        
        sigma_h = self.config.spatial_smoothing['sigma_horizontal']
        sigma_v = self.config.spatial_smoothing['sigma_vertical']
        
        if param not in self.voting_results_smoothed:
            self.voting_results_smoothed[param] = {}
        
        for strategy in self.config.spatial_smoothing['apply_to']:
            if strategy not in self.voting_results.get(param, {}):
                continue
            
            data = self.voting_results[param][strategy].copy()
            
            # 记录NaN位置
            nan_mask = np.isnan(data)
            
            # 用均值填充NaN进行平滑
            data_filled = data.copy()
            data_filled[nan_mask] = np.nanmean(data)
            
            # 3D高斯平滑 (lat, lon, depth)
            sigma = (sigma_h, sigma_h, sigma_v)
            smoothed = gaussian_filter(data_filled, sigma=sigma)
            
            # 恢复NaN（如果配置要求）
            if self.config.spatial_smoothing['preserve_nan']:
                smoothed[nan_mask] = np.nan
            
            self.voting_results_smoothed[param][strategy] = smoothed
            
            self.logger.info(f"   ✅ {strategy}: sigma=({sigma_h}, {sigma_h}, {sigma_v})")
        
        self.logger.info(f"\n✅ 空间平滑完成")
    
    def compute_consistency_metrics(self, param: str = 'vs'):
        """计算一致性指标"""
        self.logger.info("\n" + "="*80)
        self.logger.info(f"📊 计算一致性指标: {param.upper()}")
        self.logger.info("="*80)
        
        model_data_list = []
        for model_key, model_data in self.interpolated_models.items():
            if param in model_data:
                model_data_list.append(model_data[param])
        
        model_stack = np.stack(model_data_list, axis=0)
        
        if param not in self.consistency_metrics:
            self.consistency_metrics[param] = {}
        
        # 变异系数 (CV)
        if self.config.uncertainty['coefficient_of_variation']:
            self.logger.info("\n   📈 计算变异系数（CV）...")
            mean = np.nanmean(model_stack, axis=0)
            std = np.nanstd(model_stack, axis=0)
            cv = np.abs(std / mean) * 100
            cv[np.isinf(cv)] = np.nan
            self.consistency_metrics[param]['coefficient_of_variation'] = cv
            
            valid_cv = cv[~np.isnan(cv)]
            if len(valid_cv) > 0:
                self.logger.info(f"      CV范围: {np.min(valid_cv):.2f}% ~ {np.max(valid_cv):.2f}%")
                self.logger.info(f"      平均CV: {np.mean(valid_cv):.2f}%")
        
        # 模型一致性指数 (MAI)
        if self.config.uncertainty['model_agreement_index']:
            self.logger.info("\n   📈 计算模型一致性指数（MAI）...")
            mai = 1.0 - np.clip(cv / 100.0, 0, 1)
            self.consistency_metrics[param]['model_agreement_index'] = mai
            
            valid_mai = mai[~np.isnan(mai)]
            if len(valid_mai) > 0:
                self.logger.info(f"      MAI范围: {np.min(valid_mai):.3f} ~ {np.max(valid_mai):.3f}")
                self.logger.info(f"      平均MAI: {np.mean(valid_mai):.3f}")
        
        self.logger.info(f"\n✅ 一致性指标计算完成")
    
    def compute_bootstrap_confidence_interval(self, param: str = 'vs'):
        """计算Bootstrap置信区间"""
        if not self.config.uncertainty['bootstrap']['enabled']:
            self.logger.info("   ⏭️ Bootstrap未启用，跳过")
            return
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"🎲 计算Bootstrap置信区间: {param.upper()}")
        self.logger.info("="*80)
        
        n_bootstrap = self.config.uncertainty['bootstrap']['n_bootstrap']
        confidence_level = self.config.uncertainty['bootstrap']['confidence_level']
        random_seed = self.config.uncertainty['bootstrap']['random_seed']
        
        np.random.seed(random_seed)
        
        # 收集模型数据
        model_data_list = []
        for model_key, model_data in self.interpolated_models.items():
            if param in model_data:
                model_data_list.append(model_data[param])
        
        n_models = len(model_data_list)
        model_stack = np.stack(model_data_list, axis=0)
        
        self.logger.info(f"   模型数量: {n_models}")
        self.logger.info(f"   Bootstrap次数: {n_bootstrap}")
        self.logger.info(f"   置信水平: {confidence_level*100:.0f}%")
        
        # Bootstrap重采样
        grid_shape = model_stack.shape[1:]
        bootstrap_medians = np.zeros((n_bootstrap,) + grid_shape)
        
        for i in tqdm(range(n_bootstrap), desc="Bootstrap重采样"):
            # 有放回抽样
            indices = np.random.choice(n_models, size=n_models, replace=True)
            resampled = model_stack[indices]
            bootstrap_medians[i] = np.nanmedian(resampled, axis=0)
        
        # 计算置信区间
        alpha = 1 - confidence_level
        ci_lower = np.nanpercentile(bootstrap_medians, alpha/2*100, axis=0)
        ci_upper = np.nanpercentile(bootstrap_medians, (1-alpha/2)*100, axis=0)
        ci_width = ci_upper - ci_lower
        median = np.nanmedian(bootstrap_medians, axis=0)
        
        self.bootstrap_results[param] = BootstrapResult(
            median=median,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_width=ci_width,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap
        )
        
        # 统计
        valid_width = ci_width[~np.isnan(ci_width)]
        if len(valid_width) > 0:
            self.logger.info(f"\n   ✅ Bootstrap完成:")
            self.logger.info(f"      置信区间宽度: {np.mean(valid_width):.4f} ± {np.std(valid_width):.4f} km/s")
            self.logger.info(f"      最大宽度: {np.max(valid_width):.4f} km/s")
            self.logger.info(f"      最小宽度: {np.min(valid_width):.4f} km/s")
    
    def compute_model_correlation(self, param: str = 'vs'):
        """计算模型间相关性矩阵"""
        if not self.config.uncertainty['model_correlation']['enabled']:
            self.logger.info("   ⏭️ 模型相关性分析未启用，跳过")
            return
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"📊 计算模型间相关性: {param.upper()}")
        self.logger.info("="*80)
        
        method = self.config.uncertainty['model_correlation']['method']
        n_sample = self.config.uncertainty['model_correlation']['sample_points']
        
        # 收集模型数据
        model_keys = []
        model_data_list = []
        
        for model_key, model_data in self.interpolated_models.items():
            if param in model_data:
                model_keys.append(model_key)
                model_data_list.append(model_data[param].ravel())
        
        n_models = len(model_keys)
        
        # 找到所有模型都有数据的点
        valid_mask = np.ones(model_data_list[0].shape, dtype=bool)
        for data in model_data_list:
            valid_mask &= ~np.isnan(data)
        
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            self.logger.warning("   ⚠️ 无共同有效数据点，跳过相关性分析")
            return
        
        # 随机采样（如果点太多）
        if len(valid_indices) > n_sample:
            np.random.seed(42)
            sample_indices = np.random.choice(valid_indices, size=n_sample, replace=False)
        else:
            sample_indices = valid_indices
        
        self.logger.info(f"   共同有效点: {len(valid_indices):,}")
        self.logger.info(f"   采样点数: {len(sample_indices):,}")
        self.logger.info(f"   相关性方法: {method}")
        
        # 计算相关性矩阵
        correlation_matrix = np.zeros((n_models, n_models))
        
        corr_func = pearsonr if method == 'pearson' else spearmanr
        
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                elif i < j:
                    data_i = model_data_list[i][sample_indices]
                    data_j = model_data_list[j][sample_indices]
                    corr, _ = corr_func(data_i, data_j)
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
        
        self.model_correlation[param] = {
            'matrix': correlation_matrix,
            'model_keys': model_keys,
            'model_names': [self.config.target_models[k]['metadata'].name for k in model_keys]
        }
        
        self.logger.info(f"\n   ✅ 相关性矩阵:")
        names = self.model_correlation[param]['model_names']
        for i, name_i in enumerate(names):
            corr_str = " ".join([f"{correlation_matrix[i,j]:.3f}" for j in range(n_models)])
            self.logger.info(f"      {name_i}: [{corr_str}]")

    # ==================== 可视化方法（增强版）====================
    
    def plot_coverage_map(self, param: str = 'vs'):
        """绘制模型覆盖度地图（改进版：离散配色 + 统计饼图）"""
        self.logger.info(f"\n🗺️ 绘制模型覆盖度地图: {param.upper()}")
        
        if param not in self.coverage_count:
            self.logger.warning("   ⚠️ 无覆盖度数据")
            return None
        
        coverage_data = self.coverage_count[param]
        depths = self.config.visualization['horizontal_depths']
        n_models = len(self.models)
        
        ncols = 4
        nrows = (len(depths) + ncols - 1) // ncols
        
        fig = plt.figure(figsize=(20, 5*nrows + 2))
        
        # 创建离散配色
        colors = ['#FFFFFF']  # 0个模型 = 白色
        blues = plt.cm.Blues(np.linspace(0.3, 1, n_models))
        colors.extend([plt.matplotlib.colors.rgb2hex(c[:3]) for c in blues])
        cmap = ListedColormap(colors)
        bounds = np.arange(-0.5, n_models + 1.5, 1)
        norm = BoundaryNorm(bounds, cmap.N)
        
        for idx, depth_km in enumerate(depths, 1):
            depth_idx = np.argmin(np.abs(self.unified_grid_coords['depth'] - depth_km))
            actual_depth = self.unified_grid_coords['depth'][depth_idx]
            
            ax = fig.add_subplot(nrows, ncols, idx, projection=ccrs.PlateCarree())
            
            lat_grid = self.unified_grid_coords['lat']
            lon_grid = self.unified_grid_coords['lon']
            
            ax.set_extent([lon_grid.min(), lon_grid.max(), 
                          lat_grid.min(), lat_grid.max()], 
                         crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', alpha=0.5)
            
            coverage_slice = coverage_data[:, :, depth_idx]
            lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
            
            im = ax.pcolormesh(lon_2d, lat_2d, coverage_slice,
                              cmap=cmap, norm=norm, shading='auto',
                              transform=ccrs.PlateCarree())
            
            ax.set_title(f'Depth: {actual_depth:.0f} km', fontsize=14, fontweight='bold')
            
            # 统计信息
            total = coverage_slice.size
            text_lines = []
            for n in range(n_models + 1):
                n_pts = np.sum(coverage_slice == n)
                text_lines.append(f"{n}M: {n_pts/total*100:.1f}%")
            
            ax.text(0.02, 0.98, '\n'.join(text_lines), transform=ax.transAxes,
                   fontsize=9, va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Colorbar
        cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Number of Models with Valid Data', fontsize=12)
        cbar.set_ticks(range(n_models + 1))
        
        plt.suptitle(f'Model Coverage Map: {param.upper()}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        
        output_file = self.output_dir / f"{self.config.output['figure_prefix']}coverage_{param}.jpg"
        fig.savefig(output_file, dpi=self.config.output['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"   ✅ 已保存: {output_file.name}")
        return output_file
    
    def plot_median_horizontal_slices(self, param: str = 'vs', use_smoothed: bool = False):
        """绘制Median投票结果水平切片（增强版：添加海岸线和网格）"""
        self.logger.info(f"\n🎨 绘制Median水平切片: {param.upper()}")
        
        # 选择数据源
        if use_smoothed and param in self.voting_results_smoothed:
            data_source = self.voting_results_smoothed
            suffix = '_smoothed'
        else:
            data_source = self.voting_results
            suffix = ''
        
        if param not in data_source or 'median' not in data_source[param]:
            self.logger.warning(f"   ⚠️ 无Median结果")
            return []
        
        median_data = data_source[param]['median']
        depths = self.config.visualization['horizontal_depths']
        
        figures = []
        saved_files = []
        
        for depth_km in depths:
            depth_idx = np.argmin(np.abs(self.unified_grid_coords['depth'] - depth_km))
            actual_depth = self.unified_grid_coords['depth'][depth_idx]
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            lat_grid = self.unified_grid_coords['lat']
            lon_grid = self.unified_grid_coords['lon']
            
            ax.set_extent([lon_grid.min(), lon_grid.max(), 
                          lat_grid.min(), lat_grid.max()], 
                         crs=ccrs.PlateCarree())
            
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', alpha=0.5)
            
            median_slice = median_data[:, :, depth_idx]
            lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
            
            # 不设置固定色标范围，让matplotlib自动适应数据范围
            im = ax.pcolormesh(lon_2d, lat_2d, median_slice,
                              cmap=self.config.visualization['cmap_velocity'],
                              shading='auto', transform=ccrs.PlateCarree())
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, 
                             color='gray', alpha=0.3, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size': 10}
            
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', 
                               pad=0.02, shrink=0.8, aspect=30)
            cbar.set_label(f'{param.upper()} (km/s)', fontsize=12)
            
            # 统计信息
            valid_data = median_slice[~np.isnan(median_slice)]
            if len(valid_data) > 0:
                stats_text = f'Mean: {np.mean(valid_data):.3f} km/s\nStd: {np.std(valid_data):.3f} km/s'
                ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                       fontsize=9, va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            smoothed_label = ' (Smoothed)' if use_smoothed else ''
            ax.set_title(f'Voting Map (Median){smoothed_label}\n{param.upper()} @ {actual_depth:.0f} km',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存
            filename = self.output_dir / f"{self.config.output['figure_prefix']}median_{param}_{int(actual_depth)}km{suffix}.jpg"
            fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
            saved_files.append(filename)
            
            figures.append(fig)
            plt.close(fig)
        
        self.logger.info(f"   ✅ 保存 {len(saved_files)} 个水平切片图")
        return saved_files
    
    def plot_vertical_cross_sections(self, param: str = 'vs'):
        """绘制垂直剖面图（新增功能）"""
        if not self.config.visualization['vertical_profiles']['enabled']:
            return []
        
        self.logger.info(f"\n🎨 绘制垂直剖面: {param.upper()}")
        
        if param not in self.voting_results or 'median' not in self.voting_results[param]:
            self.logger.warning(f"   ⚠️ 无Median结果")
            return []
        
        median_data = self.voting_results[param]['median']
        lat_grid = self.unified_grid_coords['lat']
        lon_grid = self.unified_grid_coords['lon']
        depth_grid = self.unified_grid_coords['depth']
        
        lon_profiles = self.config.visualization['vertical_profiles']['lon_profiles']
        lat_profiles = self.config.visualization['vertical_profiles']['lat_profiles']
        
        saved_files = []
        
        # 经度剖面（沿纬度方向）
        fig, axes = plt.subplots(2, len(lon_profiles), figsize=(5*len(lon_profiles), 12))
        if len(lon_profiles) == 1:
            axes = axes.reshape(2, 1)
        
        for i, lon_val in enumerate(lon_profiles):
            lon_idx = np.argmin(np.abs(lon_grid - lon_val))
            actual_lon = lon_grid[lon_idx]
            
            # 提取剖面 (lat, depth)
            profile = median_data[:, lon_idx, :]
            
            ax = axes[0, i]
            lat_2d, depth_2d = np.meshgrid(lat_grid, depth_grid, indexing='ij')
            
            # 不设置固定色标范围，让matplotlib自动适应数据范围
            im = ax.pcolormesh(lat_2d, depth_2d, profile,
                              cmap=self.config.visualization['cmap_velocity'],
                              shading='auto')
            
            ax.invert_yaxis()
            ax.set_xlabel('Latitude (°)', fontsize=11)
            ax.set_ylabel('Depth (km)', fontsize=11)
            ax.set_title(f'Lon = {actual_lon:.1f}°E', fontsize=12, fontweight='bold')
            
            # 添加地震学界面
            for name, depth_val in self.config.visualization['seismic_boundaries'].items():
                if depth_val <= depth_grid.max():
                    ax.axhline(y=depth_val, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
                    ax.text(lat_grid.min() + 1, depth_val - 10, name, fontsize=8, color='white')
            
            plt.colorbar(im, ax=ax, label=f'{param.upper()} (km/s)', shrink=0.8)
            
            # CV剖面
            if param in self.consistency_metrics:
                cv_data = self.consistency_metrics[param]['coefficient_of_variation']
                cv_profile = cv_data[:, lon_idx, :]
                
                ax2 = axes[1, i]
                im2 = ax2.pcolormesh(lat_2d, depth_2d, cv_profile,
                                    cmap='YlOrRd', vmin=0, vmax=10,
                                    shading='auto')
                ax2.invert_yaxis()
                ax2.set_xlabel('Latitude (°)', fontsize=11)
                ax2.set_ylabel('Depth (km)', fontsize=11)
                ax2.set_title(f'CV @ Lon = {actual_lon:.1f}°E', fontsize=12)
                plt.colorbar(im2, ax=ax2, label='CV (%)', shrink=0.8)
        
        plt.suptitle(f'Vertical Cross-Sections (Longitude): {param.upper()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}vertical_lon_{param}.jpg"
        fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
        saved_files.append(filename)
        plt.close(fig)
        
        # 纬度剖面（沿经度方向）
        fig, axes = plt.subplots(2, len(lat_profiles), figsize=(5*len(lat_profiles), 12))
        if len(lat_profiles) == 1:
            axes = axes.reshape(2, 1)
        
        for i, lat_val in enumerate(lat_profiles):
            lat_idx = np.argmin(np.abs(lat_grid - lat_val))
            actual_lat = lat_grid[lat_idx]
            
            # 提取剖面 (lon, depth)
            profile = median_data[lat_idx, :, :]
            
            ax = axes[0, i]
            lon_2d, depth_2d = np.meshgrid(lon_grid, depth_grid, indexing='ij')
            
            # 不设置固定色标范围，让matplotlib自动适应数据范围
            im = ax.pcolormesh(lon_2d, depth_2d, profile,
                              cmap=self.config.visualization['cmap_velocity'],
                              shading='auto')
            
            ax.invert_yaxis()
            ax.set_xlabel('Longitude (°)', fontsize=11)
            ax.set_ylabel('Depth (km)', fontsize=11)
            ax.set_title(f'Lat = {actual_lat:.1f}°N', fontsize=12, fontweight='bold')
            
            for name, depth_val in self.config.visualization['seismic_boundaries'].items():
                if depth_val <= depth_grid.max():
                    ax.axhline(y=depth_val, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
            
            plt.colorbar(im, ax=ax, label=f'{param.upper()} (km/s)', shrink=0.8)
            
            # CV剖面
            if param in self.consistency_metrics:
                cv_data = self.consistency_metrics[param]['coefficient_of_variation']
                cv_profile = cv_data[lat_idx, :, :]
                
                ax2 = axes[1, i]
                im2 = ax2.pcolormesh(lon_2d, depth_2d, cv_profile,
                                    cmap='YlOrRd', vmin=0, vmax=10,
                                    shading='auto')
                ax2.invert_yaxis()
                ax2.set_xlabel('Longitude (°)', fontsize=11)
                ax2.set_ylabel('Depth (km)', fontsize=11)
                ax2.set_title(f'CV @ Lat = {actual_lat:.1f}°N', fontsize=12)
                plt.colorbar(im2, ax=ax2, label='CV (%)', shrink=0.8)
        
        plt.suptitle(f'Vertical Cross-Sections (Latitude): {param.upper()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}vertical_lat_{param}.jpg"
        fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
        saved_files.append(filename)
        plt.close(fig)
        
        self.logger.info(f"   ✅ 保存 {len(saved_files)} 个垂直剖面图")
        return saved_files
    
    def plot_model_difference_panel(self, param: str = 'vs'):
        """绘制模型间差异对比面板（新增功能）"""
        if not self.config.visualization['model_difference']['enabled']:
            return None
        
        self.logger.info(f"\n🎨 绘制模型差异对比: {param.upper()}")
        
        if param not in self.voting_results or 'median' not in self.voting_results[param]:
            self.logger.warning(f"   ⚠️ 无Median结果")
            return None
        
        median_data = self.voting_results[param]['median']
        depths = self.config.visualization['model_difference']['depths']
        
        lat_grid = self.unified_grid_coords['lat']
        lon_grid = self.unified_grid_coords['lon']
        
        model_keys = list(self.interpolated_models.keys())
        n_models = len(model_keys)
        n_depths = len(depths)
        
        fig = plt.figure(figsize=(5*n_depths, 4*n_models))
        
        for row, model_key in enumerate(model_keys):
            if param not in self.interpolated_models[model_key]:
                continue
            
            model_data = self.interpolated_models[model_key][param]
            meta = self.config.target_models[model_key]['metadata']
            
            for col, depth_km in enumerate(depths):
                depth_idx = np.argmin(np.abs(self.unified_grid_coords['depth'] - depth_km))
                actual_depth = self.unified_grid_coords['depth'][depth_idx]
                
                ax = fig.add_subplot(n_models, n_depths, row * n_depths + col + 1,
                                    projection=ccrs.PlateCarree())
                
                ax.set_extent([lon_grid.min(), lon_grid.max(),
                              lat_grid.min(), lat_grid.max()],
                             crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                
                # 计算差异: Model - Median
                model_slice = model_data[:, :, depth_idx]
                median_slice = median_data[:, :, depth_idx]
                diff = model_slice - median_slice
                
                lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
                
                # 动态色标范围
                vmax = np.nanpercentile(np.abs(diff), 95)
                vmax = max(vmax, 0.1)  # 最小0.1 km/s
                
                im = ax.pcolormesh(lon_2d, lat_2d, diff,
                                  cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                                  shading='auto', transform=ccrs.PlateCarree())
                
                if row == 0:
                    ax.set_title(f'{actual_depth:.0f} km', fontsize=12, fontweight='bold')
                if col == 0:
                    ax.text(-0.15, 0.5, f'{meta.name}\n- Median',
                           transform=ax.transAxes, fontsize=10,
                           va='center', ha='center', rotation=90)
                
                plt.colorbar(im, ax=ax, shrink=0.7, label='Δ (km/s)')
        
        plt.suptitle(f'Model Differences from Median: {param.upper()}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}model_difference_{param}.jpg"
        fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"   ✅ 已保存: {filename.name}")
        return filename
    
    def plot_1d_profile_comparison(self, param: str = 'vs'):
        """绘制1D剖面对比图（独立大图）"""
        self.logger.info(f"\n🎨 绘制1D剖面对比: {param.upper()}")
        
        if param not in self.voting_results:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=self.config.visualization['figsize_1d'])
        
        depth = self.unified_grid_coords['depth']
        
        # 1. 速度剖面
        ax1 = axes[0]
        
        # Median剖面
        median_data = self.voting_results[param]['median']
        median_profile = np.nanmean(median_data, axis=(0, 1))
        ax1.plot(median_profile, depth, label='Median', 
                linewidth=3, color='red', zorder=10)
        
        # 各模型剖面
        for model_key, model_data in self.interpolated_models.items():
            if param in model_data:
                profile = np.nanmean(model_data[param], axis=(0, 1))
                meta = self.config.target_models[model_key]['metadata']
                ax1.plot(profile, depth, label=meta.name,
                        linewidth=1.5, color=meta.color, linestyle='--', alpha=0.7)
        
        ax1.invert_yaxis()
        ax1.set_xlabel(f'{param.upper()} (km/s)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Depth (km)', fontsize=12, fontweight='bold')
        ax1.set_title('Velocity Profiles', fontsize=13, fontweight='bold')
        ax1.legend(loc='lower left', fontsize=9)
        ax1.grid(True, linestyle=':', alpha=0.4)
        
        # 添加地震学界面
        for name, boundary in self.config.visualization['seismic_boundaries'].items():
            if boundary <= depth.max():
                ax1.axhline(y=boundary, color='gray', linestyle='--', linewidth=1, alpha=0.6)
                ax1.text(ax1.get_xlim()[1], boundary, f' {name}', fontsize=8, va='center')
        
        # 2. 差异剖面
        ax2 = axes[1]
        
        for model_key, model_data in self.interpolated_models.items():
            if param in model_data:
                model_profile = np.nanmean(model_data[param], axis=(0, 1))
                diff_profile = model_profile - median_profile
                meta = self.config.target_models[model_key]['metadata']
                ax2.plot(diff_profile, depth, label=f'{meta.name} - Median',
                        linewidth=2, color=meta.color)
        
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.invert_yaxis()
        ax2.set_xlabel(f'Δ{param.upper()} (km/s)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Depth (km)', fontsize=12, fontweight='bold')
        ax2.set_title('Difference from Median', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, linestyle=':', alpha=0.4)
        
        # 3. CV剖面
        ax3 = axes[2]
        
        if param in self.consistency_metrics:
            cv_data = self.consistency_metrics[param]['coefficient_of_variation']
            cv_profile = np.nanmean(cv_data, axis=(0, 1))
            ax3.plot(cv_profile, depth, linewidth=2.5, color='orange')
            ax3.fill_betweenx(depth, 0, cv_profile, alpha=0.3, color='orange')
        
        ax3.invert_yaxis()
        ax3.set_xlabel('CV (%)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Depth (km)', fontsize=12, fontweight='bold')
        ax3.set_title('Coefficient of Variation', fontsize=13, fontweight='bold')
        ax3.grid(True, linestyle=':', alpha=0.4)
        ax3.set_xlim(0, None)
        
        for name, boundary in self.config.visualization['seismic_boundaries'].items():
            if boundary <= depth.max():
                ax3.axhline(y=boundary, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        
        plt.suptitle(f'1D Profile Analysis: {param.upper()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}1d_profiles_{param}.jpg"
        fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"   ✅ 已保存: {filename.name}")
        return filename
    
    def plot_bootstrap_uncertainty(self, param: str = 'vs'):
        """绘制Bootstrap置信区间可视化"""
        if param not in self.bootstrap_results:
            self.logger.info(f"   ⏭️ 无Bootstrap结果，跳过")
            return None
        
        self.logger.info(f"\n🎨 绘制Bootstrap不确定性: {param.upper()}")
        
        bootstrap = self.bootstrap_results[param]
        depths = self.config.visualization['comparison_depths']
        
        lat_grid = self.unified_grid_coords['lat']
        lon_grid = self.unified_grid_coords['lon']
        
        fig = plt.figure(figsize=self.config.visualization['figsize_bootstrap'])
        
        for i, depth_km in enumerate(depths):
            depth_idx = np.argmin(np.abs(self.unified_grid_coords['depth'] - depth_km))
            actual_depth = self.unified_grid_coords['depth'][depth_idx]
            
            # 置信区间宽度
            ax = fig.add_subplot(2, len(depths), i + 1, projection=ccrs.PlateCarree())
            
            ax.set_extent([lon_grid.min(), lon_grid.max(),
                          lat_grid.min(), lat_grid.max()],
                         crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            
            ci_width_slice = bootstrap.ci_width[:, :, depth_idx]
            lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
            
            vmax = np.nanpercentile(ci_width_slice, 95)
            im = ax.pcolormesh(lon_2d, lat_2d, ci_width_slice,
                              cmap='YlOrRd', vmin=0, vmax=vmax,
                              shading='auto', transform=ccrs.PlateCarree())
            
            ax.set_title(f'CI Width @ {actual_depth:.0f} km', fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.7, label='km/s')
            
            # 下半部分：1D剖面的置信区间
            ax2 = fig.add_subplot(2, len(depths), len(depths) + i + 1)
            
            depth_array = self.unified_grid_coords['depth']
            median_profile = np.nanmean(bootstrap.median, axis=(0, 1))
            ci_lower_profile = np.nanmean(bootstrap.ci_lower, axis=(0, 1))
            ci_upper_profile = np.nanmean(bootstrap.ci_upper, axis=(0, 1))
            
            ax2.fill_betweenx(depth_array, ci_lower_profile, ci_upper_profile,
                             alpha=0.3, color='blue', label=f'{bootstrap.confidence_level*100:.0f}% CI')
            ax2.plot(median_profile, depth_array, 'b-', linewidth=2, label='Median')
            
            ax2.invert_yaxis()
            ax2.set_xlabel(f'{param.upper()} (km/s)', fontsize=10)
            ax2.set_ylabel('Depth (km)', fontsize=10)
            ax2.set_title('Bootstrap CI Profile', fontsize=11)
            ax2.legend(fontsize=8)
            ax2.grid(True, linestyle=':', alpha=0.4)
        
        plt.suptitle(f'Bootstrap Uncertainty Analysis: {param.upper()}\n'
                    f'(n={bootstrap.n_bootstrap}, {bootstrap.confidence_level*100:.0f}% CI)',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}bootstrap_{param}.jpg"
        fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"   ✅ 已保存: {filename.name}")
        return filename
    
    def plot_model_correlation_matrix(self, param: str = 'vs'):
        """绘制模型相关性矩阵热图"""
        if param not in self.model_correlation:
            self.logger.info(f"   ⏭️ 无相关性数据，跳过")
            return None
        
        self.logger.info(f"\n🎨 绘制模型相关性矩阵: {param.upper()}")
        
        corr_data = self.model_correlation[param]
        matrix = corr_data['matrix']
        names = corr_data['model_names']
        
        fig, ax = plt.subplots(figsize=self.config.visualization['figsize_correlation'])
        
        im = ax.imshow(matrix, cmap=self.config.visualization['cmap_correlation'],
                       vmin=0, vmax=1)
        
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(names, fontsize=11)
        
        # 添加数值标注
        for i in range(len(names)):
            for j in range(len(names)):
                text_color = 'white' if matrix[i, j] < 0.5 else 'black'
                ax.text(j, i, f'{matrix[i, j]:.3f}',
                       ha='center', va='center', color=text_color, fontsize=12)
        
        plt.colorbar(im, ax=ax, label='Correlation', shrink=0.8)
        
        ax.set_title(f'Model Correlation Matrix: {param.upper()}\n'
                    f'(Method: {self.config.uncertainty["model_correlation"]["method"]})',
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}correlation_{param}.jpg"
        fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"   ✅ 已保存: {filename.name}")
        return filename
    
    def plot_comprehensive_summary(self, param: str = 'vs'):
        """绘制综合汇总面板（增强版：添加海岸线）"""
        self.logger.info(f"\n🎨 绘制综合汇总面板: {param.upper()}")
        
        if param not in self.voting_results:
            self.logger.warning(f"   ⚠️ 无投票结果")
            return None
        
        fig = plt.figure(figsize=self.config.visualization['figsize_comprehensive'])
        gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
        
        median_data = self.voting_results[param]['median']
        cv_data = self.consistency_metrics[param]['coefficient_of_variation']
        coverage_data = self.coverage_count[param]
        
        lat_grid = self.unified_grid_coords['lat']
        lon_grid = self.unified_grid_coords['lon']
        depth = self.unified_grid_coords['depth']
        
        comparison_depths = self.config.visualization['comparison_depths']
        
        # Row 1: Median水平切片（带海岸线，自动色标）
        for i, depth_km in enumerate(comparison_depths):
            depth_idx = np.argmin(np.abs(depth - depth_km))
            actual_depth = depth[depth_idx]
            
            ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
            ax.set_extent([lon_grid.min(), lon_grid.max(),
                          lat_grid.min(), lat_grid.max()],
                         crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
            
            median_slice = median_data[:, :, depth_idx]
            lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
            
            # 不设置固定色标范围，让matplotlib自动适应数据范围
            im1 = ax.pcolormesh(lon_2d, lat_2d, median_slice,
                               cmap='jet_r', 
                               shading='auto', transform=ccrs.PlateCarree())
            
            ax.set_title(f'Median @ {actual_depth:.0f} km', fontsize=12, fontweight='bold')
            
            cbar = plt.colorbar(im1, ax=ax, orientation='horizontal', pad=0.08, shrink=0.9)
            cbar.set_label(f'{param.upper()} (km/s)', fontsize=10)
        
        # Row 2: CV不确定性地图（带海岸线）
        for i, depth_km in enumerate(comparison_depths):
            depth_idx = np.argmin(np.abs(depth - depth_km))
            actual_depth = depth[depth_idx]
            
            ax = fig.add_subplot(gs[1, i], projection=ccrs.PlateCarree())
            ax.set_extent([lon_grid.min(), lon_grid.max(),
                          lat_grid.min(), lat_grid.max()],
                         crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
            
            cv_slice = cv_data[:, :, depth_idx]
            lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
            
            vmax_cv = min(np.nanpercentile(cv_slice, 95), 15)
            im2 = ax.pcolormesh(lon_2d, lat_2d, cv_slice,
                               cmap='YlOrRd', vmin=0, vmax=vmax_cv,
                               shading='auto', transform=ccrs.PlateCarree())
            
            ax.set_title(f'CV @ {actual_depth:.0f} km', fontsize=12, fontweight='bold')
            
            cbar = plt.colorbar(im2, ax=ax, orientation='horizontal', pad=0.08, shrink=0.9)
            cbar.set_label('CV (%)', fontsize=10)
        
        # Row 3: 1D剖面和统计
        # 速度剖面
        ax3 = fig.add_subplot(gs[2, 0])
        median_profile = np.nanmean(median_data, axis=(0, 1))
        ax3.plot(median_profile, depth, label='Median', linewidth=2.5, color='red')
        
        for model_key, model_data in self.interpolated_models.items():
            if param in model_data:
                profile = np.nanmean(model_data[param], axis=(0, 1))
                meta = self.config.target_models[model_key]['metadata']
                ax3.plot(profile, depth, label=meta.name,
                        linewidth=1.5, color=meta.color, linestyle='--', alpha=0.7)
        
        ax3.invert_yaxis()
        ax3.set_xlabel(f'{param.upper()} (km/s)', fontsize=11)
        ax3.set_ylabel('Depth (km)', fontsize=11)
        ax3.set_title('1D Profiles', fontsize=12, fontweight='bold')
        ax3.legend(loc='lower left', fontsize=9)
        ax3.grid(True, linestyle=':', alpha=0.4)
        
        # CV深度剖面
        ax4 = fig.add_subplot(gs[2, 1])
        cv_profile = np.nanmean(cv_data, axis=(0, 1))
        ax4.plot(cv_profile, depth, linewidth=2, color='orange')
        ax4.fill_betweenx(depth, 0, cv_profile, alpha=0.3, color='orange')
        ax4.invert_yaxis()
        ax4.set_xlabel('CV (%)', fontsize=11)
        ax4.set_ylabel('Depth (km)', fontsize=11)
        ax4.set_title('Uncertainty vs Depth', fontsize=12, fontweight='bold')
        ax4.grid(True, linestyle=':', alpha=0.4)
        
        # 覆盖度深度剖面
        ax5 = fig.add_subplot(gs[2, 2])
        coverage_profile = np.nanmean(coverage_data, axis=(0, 1))
        ax5.plot(coverage_profile, depth, linewidth=2, color='blue')
        ax5.fill_betweenx(depth, 0, coverage_profile, alpha=0.3, color='blue')
        ax5.invert_yaxis()
        ax5.set_xlabel('Coverage Count', fontsize=11)
        ax5.set_ylabel('Depth (km)', fontsize=11)
        ax5.set_title('Model Coverage vs Depth', fontsize=12, fontweight='bold')
        ax5.grid(True, linestyle=':', alpha=0.4)
        ax5.set_xlim(0, len(self.models) + 0.5)
        
        plt.suptitle(f'Voting Map Comprehensive Summary: {param.upper()}',
                    fontsize=14, fontweight='bold')
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}comprehensive_summary_{param}.jpg"
        fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"   ✅ 已保存: {filename.name}")
        return filename
    
    def plot_uncertainty_maps(self, param: str, metric: str):
        """绘制不确定性地图"""
        if param not in self.consistency_metrics or metric not in self.consistency_metrics[param]:
            return []
        
        uncertainty_data = self.consistency_metrics[param][metric]
        depths = self.config.visualization['horizontal_depths']
        
        saved_files = []
        
        for depth_km in depths:
            depth_idx = np.argmin(np.abs(self.unified_grid_coords['depth'] - depth_km))
            actual_depth = self.unified_grid_coords['depth'][depth_idx]
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            lat_grid = self.unified_grid_coords['lat']
            lon_grid = self.unified_grid_coords['lon']
            
            ax.set_extent([lon_grid.min(), lon_grid.max(),
                          lat_grid.min(), lat_grid.max()],
                         crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
            
            uncertainty_slice = uncertainty_data[:, :, depth_idx]
            lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
            
            if metric == 'model_agreement_index':
                cmap = self.config.visualization['cmap_agreement']
                vmin, vmax = 0, 1
                label = 'MAI (0-1)'
                title_metric = 'Model Agreement Index'
            else:
                cmap = self.config.visualization['cmap_uncertainty']
                vmin = 0
                vmax = min(np.nanpercentile(uncertainty_slice, 95), 15)
                label = 'CV (%)'
                title_metric = 'Coefficient of Variation'
            
            im = ax.pcolormesh(lon_2d, lat_2d, uncertainty_slice,
                              cmap=cmap, vmin=vmin, vmax=vmax,
                              shading='auto', transform=ccrs.PlateCarree())
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                             color='gray', alpha=0.3, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
            cbar.set_label(label, fontsize=11)
            
            ax.set_title(f'{title_metric}\n{param.upper()} @ {actual_depth:.0f} km',
                        fontsize=13, fontweight='bold')
            
            plt.tight_layout()
            
            metric_abbr = 'cv' if 'coefficient' in metric else 'mai'
            filename = self.output_dir / f"{self.config.output['figure_prefix']}uncertainty_{metric_abbr}_{param}_{int(actual_depth)}km.jpg"
            fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
            saved_files.append(filename)
            plt.close(fig)
        
        return saved_files

    # ==================== 模型保存方法 ====================
    
    def save_voting_model_as_netcdf(self, param: str = 'vs', strategy: str = 'median',
                                    save_smoothed: bool = True):
        """保存投票结果为标准NetCDF速度模型"""
        self.logger.info(f"\n💾 保存投票模型为NetCDF: {param.upper()} ({strategy})")
        
        # 检查数据
        if param not in self.voting_results or strategy not in self.voting_results[param]:
            self.logger.warning(f"   ⚠️ 无投票结果: {param}/{strategy}")
            return None
        
        # 创建输出目录
        model_name = f"VotingMap_{strategy}_{param.upper()}"
        output_dir = self.base_config.dirs['data'] / 'fwi-models' / 'processed' / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取数据
        voting_data = self.voting_results[param][strategy]
        lat = self.unified_grid_coords['lat']
        lon = self.unified_grid_coords['lon']
        depth = self.unified_grid_coords['depth']
        
        # 转置为 (lon, lat, depth) - NetCDF标准格式
        data_transposed = np.transpose(voting_data, (1, 0, 2))
        
        # 创建数据变量字典
        data_vars = {
            param: xr.DataArray(
                data_transposed,
                dims=['longitude', 'latitude', 'depth'],
                coords={'longitude': lon, 'latitude': lat, 'depth': depth},
                attrs={
                    'units': 'km/s',
                    'long_name': f'{param.upper()} velocity from voting map',
                    'voting_strategy': strategy
                }
            )
        }
        
        # 添加衍生参数
        if param == 'vs':
            for var_name in ['vs0', 'vsv', 'vsh']:
                data_vars[var_name] = data_vars['vs'].copy()
                data_vars[var_name].attrs['long_name'] = f'{var_name.upper()} velocity'
        
        # 添加不确定性数据
        if param in self.consistency_metrics:
            cv_data = self.consistency_metrics[param]['coefficient_of_variation']
            cv_transposed = np.transpose(cv_data, (1, 0, 2))
            data_vars['cv'] = xr.DataArray(
                cv_transposed.astype(np.float32),
                dims=['longitude', 'latitude', 'depth'],
                coords={'longitude': lon, 'latitude': lat, 'depth': depth},
                attrs={'units': '%', 'long_name': 'Coefficient of Variation'}
            )
        
        # 添加覆盖度数据
        if param in self.coverage_count:
            coverage = self.coverage_count[param]
            coverage_transposed = np.transpose(coverage, (1, 0, 2))
            data_vars['coverage'] = xr.DataArray(
                coverage_transposed.astype(np.int8),
                dims=['longitude', 'latitude', 'depth'],
                coords={'longitude': lon, 'latitude': lat, 'depth': depth},
                attrs={'units': 'count', 'long_name': 'Number of models with valid data'}
            )
        
        # 创建Dataset
        ds = xr.Dataset(data_vars)
        
        # 全局属性
        ds.attrs = {
            'title': f"Voting Map Model: {strategy} - {param.upper()}",
            'source': f"Multi-model voting: {', '.join([m.metadata.name for m in self.models.values()])}",
            'voting_strategy': strategy,
            'creation_date': datetime.now().isoformat(),
            'creator': 'EASTASIA-FWI VotingMapAnalyzer v2.1',
            'n_source_models': len(self.models),
            'spatial_smoothing': str(self.config.spatial_smoothing['enabled'])
        }
        
        # 编码设置
        encoding = {var: {'dtype': 'float32', 'zlib': True, 'complevel': 4} 
                   for var in ds.data_vars if var != 'coverage'}
        if 'coverage' in ds.data_vars:
            encoding['coverage'] = {'dtype': 'int8', 'zlib': True}
        
        # 保存
        nc_file = output_dir / f"{model_name}_original.nc"
        ds.to_netcdf(nc_file, encoding=encoding, format='NETCDF4')
        
        file_size = nc_file.stat().st_size / (1024**2)
        self.logger.info(f"   ✅ NetCDF已保存: {nc_file.name} ({file_size:.2f} MB)")
        
        # 保存平滑版本
        if save_smoothed and self.config.spatial_smoothing['enabled']:
            if param in self.voting_results_smoothed and strategy in self.voting_results_smoothed[param]:
                smoothed_data = self.voting_results_smoothed[param][strategy]
                smoothed_transposed = np.transpose(smoothed_data, (1, 0, 2))
                
                ds_smoothed = ds.copy()
                ds_smoothed[param] = xr.DataArray(
                    smoothed_transposed,
                    dims=['longitude', 'latitude', 'depth'],
                    coords={'longitude': lon, 'latitude': lat, 'depth': depth}
                )
                ds_smoothed.attrs['spatial_smoothing'] = 'True'
                ds_smoothed.attrs['smoothing_sigma_h'] = self.config.spatial_smoothing['sigma_horizontal']
                ds_smoothed.attrs['smoothing_sigma_v'] = self.config.spatial_smoothing['sigma_vertical']
                
                nc_file_smoothed = output_dir / f"{model_name}_smoothed.nc"
                ds_smoothed.to_netcdf(nc_file_smoothed, encoding=encoding, format='NETCDF4')
                self.logger.info(f"   ✅ 平滑版本已保存: {nc_file_smoothed.name}")
        
        ds.close()
        return nc_file
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """生成汇总报告"""
        self.logger.info("\n📋 生成汇总报告...")
        
        report = {
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'analysis_version': 'v2.1-Enhanced',
                'n_models': len(self.models),
                'model_names': [m.metadata.name for m in self.models.values()]
            },
            'config': {
                'spatial_smoothing': self.config.spatial_smoothing,
                'uncertainty': self.config.uncertainty
            },
            'voting_results': {},
            'uncertainty_analysis': {}
        }
        
        # 投票结果统计
        for param, strategies in self.voting_results.items():
            report['voting_results'][param] = {}
            for strategy, data in strategies.items():
                valid = data[~np.isnan(data)]
                report['voting_results'][param][strategy] = {
                    'min': float(np.min(valid)) if len(valid) > 0 else None,
                    'max': float(np.max(valid)) if len(valid) > 0 else None,
                    'mean': float(np.mean(valid)) if len(valid) > 0 else None,
                    'valid_percentage': float(len(valid) / data.size * 100)
                }
        
        # Bootstrap结果
        for param, bootstrap in self.bootstrap_results.items():
            ci_width = bootstrap.ci_width[~np.isnan(bootstrap.ci_width)]
            report['uncertainty_analysis'][param] = {
                'bootstrap': {
                    'n_samples': bootstrap.n_bootstrap,
                    'confidence_level': bootstrap.confidence_level,
                    'mean_ci_width': float(np.mean(ci_width)) if len(ci_width) > 0 else None
                }
            }
        
        # 模型相关性
        for param, corr in self.model_correlation.items():
            if param not in report['uncertainty_analysis']:
                report['uncertainty_analysis'][param] = {}
            report['uncertainty_analysis'][param]['model_correlation'] = {
                'matrix': corr['matrix'].tolist(),
                'model_names': corr['model_names']
            }
        
        # 保存
        report_file = self.results_dir / 'voting_map_report_v21.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"   ✅ 报告已保存: {report_file.name}")
        return report
    
    def _visualize_and_save_results(self, param: str):
        """生成所有可视化结果"""
        self.logger.info(f"\n🎨 生成 {param.upper()} 可视化结果")
        
        # 1. 覆盖度地图
        self.plot_coverage_map(param)
        
        # 2. Median水平切片
        self.plot_median_horizontal_slices(param, use_smoothed=False)
        if self.config.spatial_smoothing['enabled']:
            self.plot_median_horizontal_slices(param, use_smoothed=True)
        
        # 3. 不确定性地图
        for metric in ['coefficient_of_variation', 'model_agreement_index']:
            self.plot_uncertainty_maps(param, metric)
        
        # 4. 垂直剖面
        self.plot_vertical_cross_sections(param)
        
        # 5. 模型差异对比
        self.plot_model_difference_panel(param)
        
        # 6. 1D剖面对比
        self.plot_1d_profile_comparison(param)
        
        # 7. Bootstrap可视化
        self.plot_bootstrap_uncertainty(param)
        
        # 8. 相关性矩阵
        self.plot_model_correlation_matrix(param)
        
        # 9. 综合面板
        self.plot_comprehensive_summary(param)
        
        self.logger.info(f"   ✅ {param.upper()} 可视化完成")
    
    def run_full_analysis(self, params: List[str] = None):
        """运行完整分析流程"""
        if params is None:
            params = self.config.visualization['parameters']
        
        self.logger.info("\n" + "="*80)
        self.logger.info("🚀 开始完整投票图分析 (v2.1 - 增强版)")
        self.logger.info(f"📊 分析参数: {', '.join([p.upper() for p in params])}")
        self.logger.info("="*80)
        
        try:
            # 1. 加载模型
            self.load_models()
            
            # 2. 创建统一网格
            self.create_unified_grid()
            
            # 3-8. 对每个参数处理
            for param in params:
                self.logger.info("\n" + "🔸"*40)
                self.logger.info(f"🔸 处理参数: {param.upper()}")
                self.logger.info("🔸"*40)
                
                # 插值
                self.interpolate_models_to_unified_grid(param)
                
                # 投票计算
                self.compute_voting_results(param)
                
                # 空间平滑
                self.apply_spatial_smoothing(param)
                
                # 一致性指标
                self.compute_consistency_metrics(param)
                
                # Bootstrap分析
                self.compute_bootstrap_confidence_interval(param)
                
                # 模型相关性
                self.compute_model_correlation(param)
                
                # 可视化
                self._visualize_and_save_results(param)
                
                # 保存NetCDF
                if self.config.output['save_voting_models']:
                    self.save_voting_model_as_netcdf(param, 'median')
            
            # 9. 生成报告
            self.generate_summary_report()
            
            self.logger.info("\n" + "="*80)
            self.logger.info("✅ 投票图分析完成!")
            self.logger.info(f"📁 图表: {self.output_dir}")
            self.logger.info(f"📁 数据: {self.results_dir}")
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"\n❌ 分析失败: {e}")
            import traceback
            traceback.print_exc()


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🚀 EASTASIA-FWI 速度模型投票图分析 (v2.1 - 增强版)")
    print("="*80)
    print("\n🔥 v2.1 新增功能:")
    print("   1. 多种投票策略（Median/Mean/Weighted Mean）")
    print("   2. 空间平滑选项（减少模型边界不连续）")
    print("   3. Bootstrap置信区间（不确定性量化）")
    print("   4. 模型间相关性分析（避免重复计票）")
    print("   5. 垂直剖面可视化")
    print("   6. 模型差异对比面板")
    print("   7. 1D剖面独立大图")
    print("\n🗳️ 核心功能:")
    print("   • 使用模型原始数据（保留完整覆盖）")
    print("   • 智能处理部分NaN（np.nanmedian）")
    print("   • 多维度不确定性量化（CV/MAI/Bootstrap）")
    print("="*80 + "\n")
    
    try:
        analyzer = VotingMapAnalyzer()
        
        params_to_analyze = ['vs', 'vp']
        analyzer.run_full_analysis(params=params_to_analyze)
        
        print("\n" + "="*80)
        print("✅ 分析成功完成!")
        print(f"📁 图表输出: {analyzer.output_dir}")
        print(f"📁 数据输出: {analyzer.results_dir}")
        print("\n📊 生成的新增可视化:")
        print("   • 垂直剖面图 (vertical_lon/lat_*.jpg)")
        print("   • 模型差异面板 (model_difference_*.jpg)")
        print("   • 1D剖面对比 (1d_profiles_*.jpg)")
        print("   • Bootstrap不确定性 (bootstrap_*.jpg)")
        print("   • 相关性矩阵 (correlation_*.jpg)")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()