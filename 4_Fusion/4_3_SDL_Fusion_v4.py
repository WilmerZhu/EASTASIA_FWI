"""
EASTASIA-FWI 稀疏字典学习（Sparse Dictionary Learning）速度模型融合模块 v3.4
================================================================================

基于论文《Enhancing Regional Seismic Velocity Models With Higher-Resolution 
Local Results Using Sparse Dictionary Learning》(Zhang & Ben-Zion, 2024) 实现

v3.4 核心更新 ⭐ 移除统一网格方法:
1. ⭐ 移除 PairedPatchExtractor: 只保留 DirectPatchExtractor（论文方法）
2. ⭐ 移除 UnifiedModelFramework 统一网格功能: 直接使用原始模型
3. ⭐ 修复 F01 可视化 bug: original_slice 现使用上采样版本，与坐标维度一致
4. ⭐ Phase 1/1.5 训练/变换一致: 全部使用直接提取方法

v3.3.5 核心更新:
1. Phase 1 所有深度独立训练: 每个深度层有独立的字典对 (D₁, D₂)
2. Phase 1.5 使用对应深度字典: 精确匹配，不复用其他深度的字典
3. 新增 use_all_depths 参数: 可选择论文方法(全部)或快速模式(代表性深度)

v3.3.4 核心更新:
1. 简化网格搜索: 移除 quick/full 区分，统一使用 --grid-search
2. 海岸线可视化: 所有地理坐标图使用 cartopy 添加海岸线
3. 参数配置更新: window=6°, stride=1° (83%重叠率)
4. Phase 1.5 对所有深度变换: 不再限于 target_depths
5. F02 可视化选择代表性深度: 最多显示5层

v3.3 核心功能:
1. Phase H: K × λ 联合网格搜索 (论文方法，不搜索 patch size)
2. 三种运行模式:
   - 模式 A: 默认运行（直接使用代码中配置的参数）
   - 模式 B: --grid-search（运行 K × λ 网格搜索）
   - 模式 C: --use-optimal（使用已有最优配置）
3. HyperparameterSearcher 类: 自动记录最优配置到 optimal_config.json
4. H 组可视化: H01-H05 搜索结果可视化
5. GridSearchResult 数据类: 标准化搜索结果存储
6. 统一 physical_stride 命名 (移除 stride_ratio)

v3.2 核心更新:
1. D₂计算修正: 支持论文的纯最小二乘方法 (d2_regularization=0)
2. DirectPatchExtractor: 直接从原始网格提取配对patches（论文方法，默认）
3. 参数敏感性测试: 支持窗口大小、步长、字典参数的系统测试

v3.1 核心更新:
1. Phase -1 - 数据预处理与Patch提取
2. Phase 0 - 字典表示能力验证
3. Phase 1 - 2D Patch级变换验证
4. Phase 1.5 - 完整2D切片变换
5. 双目标模型支持

科学原理:
- 字典表示: p_j ≈ D* c_j = Σ c_{i,j} d_i （公式1）
- 优化问题: D*, C* = argmin ||P - DC||_2^2 + λ||C||_0 （公式3）
- 耦合字典: P_1 ≈ D_1 C, P_2 ≈ D_2 C （公式4-5）
- 字典学习: D_1, C = argmin ||P_1 - D_1 C||_2^2 + λ||C||_0 （公式6）
- D₂计算: D_2 = argmin ||P_2 - D_2 C||_2^2 （公式7，论文无正则化）
- 网格搜索: Best = argmin_{K,λ} D(P̂ᵛ, P₂ᵛ) （公式10，论文40组合）

模型配置:
- USTClitho2.0: 1°×1°×20km（基础模型 M₁，待增强）
- CSES_VM1.0: 0.25°×0.25°×10km（高分辨率模型 M₂）
- CSES_VM1.0: 0.25°×0.25°×10km（高分辨率模型 M₃）

参考文献:
- Zhang, H., & Ben-Zion, Y. (2024). Enhancing regional seismic velocity models 
  with higher-resolution local results using sparse dictionary learning.
  Journal of Geophysical Research: Solid Earth, 129, e2023JB027016.

作者: EASTASIA-FWI Team
日期: 2026-01-09
版本: v3.4 - 直接提取方法（移除统一网格）
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from dataclasses import dataclass, field
import warnings
import sys
import logging
import json
import argparse
from scipy.stats import pearsonr
from scipy.ndimage import zoom
import time
from tqdm import tqdm

# 稀疏编码库
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import Ridge

# 地理可视化 ⭐ v3.3.4 新增
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig

warnings.filterwarnings('ignore')


# ==============================================================================
# 第一部分: 数据类定义
# ==============================================================================

@dataclass
class ModelResolutionInfo:
    """模型分辨率信息"""
    name: str
    full_name: str
    lat_resolution: float  # 纬度分辨率（度）
    lon_resolution: float  # 经度分辨率（度）
    depth_resolution: float  # 深度分辨率（km）
    color: str
    marker: str
    role: str  # 'base' 或 'highres'
    filename: str = ''  # NetCDF文件名
    short_name: str = ''  # 简称用于变量命名


@dataclass
class ValidationPatchResult:
    """Phase 0: 验证Patch提取结果"""
    patches: np.ndarray           # (N, L) patches矩阵
    coords: np.ndarray            # (N, 2) patch中心坐标 [lat, lon]
    velocity_slice: np.ndarray    # 原始速度切片
    lat_coords: np.ndarray
    lon_coords: np.ndarray
    n_patches: int
    patch_size: int
    depth_km: float


@dataclass
class ValidationResult:
    """Phase 0: 字典验证结果"""
    depth_km: float
    model_name: str
    n_patches: int
    dictionary: np.ndarray        # (K, L) 字典
    coefficients: np.ndarray      # (N, K) 稀疏系数
    patches_original: np.ndarray
    patches_reconstructed: np.ndarray
    coords: np.ndarray
    velocity_slice: np.ndarray
    lat_coords: np.ndarray
    lon_coords: np.ndarray
    error_stats: Dict[str, float]
    passed: bool
    mean_patch: np.ndarray        # 用于重建的均值
    patch_size: int = 0           # Patch边长（格点数）


@dataclass
class PreprocessingInfo:
    """预处理参数保存"""
    method: str               # 'centering' or 'global_standardization'
    mean_P1: np.ndarray       # P1 均值 (L1,)
    mean_P2: np.ndarray       # P2 均值 (L2,)
    std_P1: Optional[np.ndarray] = None
    std_P2: Optional[np.ndarray] = None
    global_mean: Optional[float] = None
    global_std: Optional[float] = None


@dataclass
class PairedPatchResult:
    """Phase 1/2: 配对Patch提取结果"""
    patches_base: np.ndarray      # (N, L₁) 基础模型patches
    patches_highres: np.ndarray   # (N, L₂) 高分辨率patches
    coords_latlon: np.ndarray     # (N, 2) patch中心坐标 [lat, lon]
    correlations: np.ndarray      # (N,) CC值
    valid_mask: np.ndarray        # (N,) CC>阈值的有效标记
    n_total: int
    n_valid: int
    physical_window_size: float   # 物理窗口大小（度）
    base_patch_size: int          # 基础模型patch尺寸
    highres_patch_size: int       # 高分辨率patch尺寸
    base_slice: Optional[np.ndarray] = None
    highres_slice: Optional[np.ndarray] = None
    base_lat: Optional[np.ndarray] = None
    base_lon: Optional[np.ndarray] = None
    highres_lat: Optional[np.ndarray] = None
    highres_lon: Optional[np.ndarray] = None
    depth_km: Optional[float] = None
    # v3.1: 3D支持
    is_3d: bool = False
    depth_range: Optional[Tuple[float, float]] = None


@dataclass
class CoupledDictionaryResult:
    """耦合字典学习结果"""
    D1: np.ndarray                # (K, L₁) 基础模型字典
    D2: np.ndarray                # (K, L₂) 高分辨率字典
    C_train: np.ndarray           # (N_train, K) 训练集稀疏系数
    C_val: np.ndarray             # (N_val, K) 验证集稀疏系数
    train_indices: np.ndarray
    val_indices: np.ndarray
    preprocessing: PreprocessingInfo
    train_error_base: float
    train_error_highres: float
    val_error_base: float
    val_error_highres: float
    transform_error: float        # SDL变换误差
    baseline_error: float         # Baseline上采样误差
    relative_improvement: float   # 相对改善率
    error_distribution: Dict
    dict_learner: DictionaryLearning
    depth_km: Optional[float] = None
    target_model: Optional[str] = None


@dataclass
class TransformResult:
    """Phase 1/2: 变换结果"""
    depth_km: float
    target_model: str
    param: str
    # Patch级别结果
    paired_patches: PairedPatchResult
    coupled_dict: Optional[CoupledDictionaryResult]  # 可能为None（失败时）
    # 变换后的patches
    transformed_patches: np.ndarray  # (N_val, L₂)
    baseline_patches: np.ndarray     # (N_val, L₂)
    # 误差统计
    transform_error: float
    baseline_error: float
    relative_improvement: float
    error_distribution: Dict[str, float]
    # 验收
    passed: bool


@dataclass
class FullSliceTransformResult:
    """Phase 1.5: 完整切片变换结果"""
    depth_km: float
    target_model: str
    param: str
    # 完整切片
    transformed_slice: np.ndarray     # SDL变换后的完整切片
    baseline_slice: np.ndarray        # Baseline上采样切片
    target_slice: np.ndarray          # 高分辨率目标切片 (GT)
    original_slice: np.ndarray        # 原始低分辨率切片
    weight_map: np.ndarray            # 权重分布图
    # 坐标
    lat_coords: np.ndarray
    lon_coords: np.ndarray
    # 误差统计
    sdl_error: float                  # SDL全域误差
    baseline_error: float             # Baseline全域误差
    relative_improvement: float       # 相对改善率
    error_mean: float                 # 误差均值
    error_std: float                  # 误差标准差
    error_distribution: Dict[str, float]
    # 处理统计
    n_patches_processed: int
    n_patches_skipped: int
    coverage_ratio: float             # 覆盖率
    # 验收
    passed: bool
    # 使用的字典
    coupled_dict: Optional[CoupledDictionaryResult] = None
    stitch_method: str = 'average'   # 'gaussian' or 'average'


@dataclass
class FullDomainResult:
    """Phase 3: 全域变换结果"""
    enhanced_model: np.ndarray            # 增强后的模型
    baseline_model: np.ndarray            # Baseline上采样模型
    original_base: np.ndarray             # 原始基础模型（插值到高分辨率）
    original_highres: np.ndarray          # 原始高分辨率模型
    enhancement_mask: np.ndarray          # 增强区域掩码
    transition_mask: np.ndarray           # 过渡区域掩码
    lat_coords: np.ndarray
    lon_coords: np.ndarray
    depth_coords: np.ndarray
    param: str
    target_model: str
    global_improvement: float


@dataclass
class ErrorDistributionStats:
    """误差分布统计"""
    mean: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    shapiro_stat: float
    shapiro_pvalue: float
    is_normal_shapiro: bool
    min_val: float
    max_val: float
    percentile_5: float
    percentile_95: float


@dataclass
class GridSearchResult:
    """
    Phase H: K × λ 网格搜索单次结果 (论文方法)
    
    v3.3 新增，用于记录每种参数组合的性能
    
    ⭐ 论文只搜索 K 和 λ，patch size (window/stride) 由物理约束决定
    """
    config_name: str                   # 配置名称，如 "K20_a0.1"
    window: float                      # 物理窗口大小（度）- 固定值
    stride: float                      # 步长（度）- 固定值
    n_components: int                  # K: 字典原子数 ⭐ 搜索参数
    alpha: float                       # λ: 稀疏性参数 ⭐ 搜索参数
    
    # 性能指标
    transform_error: float             # SDL 变换误差 (%)
    baseline_error: float              # Baseline 上采样误差 (%)
    relative_improvement: float        # 相对改善率 (%)
    error_mean: float                  # 误差均值 (%)
    error_std: float                   # 误差标准差 (%)
    n_patches: int                     # Patches 数量
    
    # 各深度结果
    depth_results: Dict[float, Dict]   # {depth_km: {metrics}}
    
    # 综合评分 (0-1, 越高越好)
    score: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典（用于JSON序列化）"""
        return {
            'config_name': self.config_name,
            'window': self.window,
            'stride': self.stride,
            'n_components': self.n_components,
            'alpha': self.alpha,
            'transform_error': self.transform_error,
            'baseline_error': self.baseline_error,
            'relative_improvement': self.relative_improvement,
            'error_mean': self.error_mean,
            'error_std': self.error_std,
            'n_patches': self.n_patches,
            'depth_results': self.depth_results,
            'score': self.score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GridSearchResult':
        """从字典创建（用于JSON反序列化）"""
        return cls(
            config_name=data['config_name'],
            window=data['window'],
            stride=data['stride'],
            n_components=data['n_components'],
            alpha=data['alpha'],
            transform_error=data['transform_error'],
            baseline_error=data['baseline_error'],
            relative_improvement=data['relative_improvement'],
            error_mean=data['error_mean'],
            error_std=data['error_std'],
            n_patches=data['n_patches'],
            depth_results=data['depth_results'],
            score=data.get('score', 0.0)
        )


# ==============================================================================
# 第二部分: 配置类
# ==============================================================================

# ⭐ v3.4: UnifiedGridConfig 已移除 - 直接使用原始模型分辨率


@dataclass
class Phase0Config:
    """Phase 0: 字典表示能力验证配置"""
    
    # 启用开关
    enabled: bool = True
    
    # 通用字典参数
    n_components: int = 20
    alpha: float = 0.1
    max_iter: int = 100
    transform_algorithm: Literal['lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'] = 'omp'
    n_nonzero_coefs: int = 5
    fit_algorithm: Literal['lars', 'cd'] = 'lars'
    random_state: int = 42
    
    # 验收标准（基于论文）
    # 注：USTClitho2.0 深度分层不规则(5-20km间距)，浅层过渡带存在少量跳变格点，
    #     最大单点误差放宽至 15%（全局/平均误差仍保持 < 3% 的严格标准）
    max_mean_error: float = 3.0     # 平均误差 < 3%（论文: 2.2%）
    max_point_error: float = 15.0   # 最大点误差 < 15%（论文规则网格: 7%，不规则网格适当放宽）
    
    # 测试深度 (USTClitho/CSES 为岩石圈浅部模型，深度 0-120km)
    validation_depths: List[float] = field(
        default_factory=lambda: [20, 40, 60, 100, 120]
    )
    
    # 窗口配置 - 基于物理窗口大小（度）
    physical_window_size: float = 3.0  # 3°×3° 物理窗口 (M₁: 6×6 @0.5°, M₂: 12×12 @0.25°)
    physical_stride: float = 0.5       # 物理步长 0.5° (= 1 格点 @0.5°分辨率，83%重叠)

    def __post_init__(self) -> None:
        """参数自校验，防止无效配置导致 Phase 0 不稳定。"""
        if self.n_components <= 0:
            raise ValueError("Phase0Config.n_components 必须 > 0")
        if self.alpha <= 0:
            raise ValueError("Phase0Config.alpha 必须 > 0")
        if not (0 < self.physical_stride <= self.physical_window_size):
            raise ValueError("Phase0Config 需要满足 0 < physical_stride <= physical_window_size")
        # 去重并排序，保证日志与结果可复现
        self.validation_depths = sorted({float(d) for d in self.validation_depths})


@dataclass
class Phase1Config:
    """
    Phase 1: 2D水平切片变换配置
    
    ⭐ 当前默认参数（SinoScope1.0→FWEA23 4:1 场景）:
    - 物理窗口: 6°×6° (M₁: 6×6 @1.0°, M₂: 24×24 @0.25°)
    - 步长: 1.0° (83% 重叠)
    - D₂正则化: 0 (纯最小二乘，与论文一致) ✅
    - 提取方法: DirectPatchExtractor (与论文一致) ✅
    
    ⭐ v3.4: 只使用直接提取方法 (DirectPatchExtractor)
    """
    
    # 启用开关
    enabled: bool = True
    
    # ⭐ v3.4: 移除 extraction_method，只使用 DirectPatchExtractor
    
    # === 物理窗口配置 ⭐ 测试模型对 SinoScope1.0(1.0°)→FWEA23(0.25°) ===
    physical_window_size: float = 6.0    # 6°×6° 物理窗口 (M₁: 6×6 格点, M₂: 24×24 格点)
    physical_stride: float = 1.0         # 1.0° 步长 (=1 格点 @1.0°，83%重叠)
    # 说明: 东亚子区覆盖较大，6°窗口在稳定样本数与空间局地性之间折中
    
    # === 字典参数 (与论文一致) ===
    n_components: int = 20          # K=20 原子 (论文: 20)
    alpha: float = 0.1              # λ=0.1 稀疏性控制 (论文: 0.1)
    n_nonzero_coefs: int = 5        # OMP每patch最多5个非零系数
    max_iter: int = 100
    fit_algorithm: Literal['lars', 'cd'] = 'lars'
    transform_algorithm: Literal['lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'] = 'omp'
    random_state: int = 42
    
    # === D₂正则化 ⭐ 论文方法 ===
    d2_regularization: float = 0.0  # ⭐ 论文方法: 0 (纯最小二乘)
    # 当 d2_regularization=0 时使用 np.linalg.lstsq (论文方法)
    # 当 d2_regularization>0 时使用 Ridge 回归
    
    # === 训练参数 ===
    train_ratio: float = 0.88      # 88% 训练集（与论文示例 1000/1131 接近）
    min_cc_threshold: float = 0.0   # CC > 0 筛选
    max_pairs: int = 30000           # 最大配对数
    
    # === 验收标准（基于论文）===
    max_transform_error: float = 10.0    # 变换误差 < 10% (论文: 7.6%)
    min_relative_improvement: float = 20.0  # 相对改善 > 20% (论文: ~35%)
    max_error_mean: float = 1.0          # 误差均值 < 1% (论文: 0.4%)
    max_error_std: float = 5.0           # 误差标准差 < 5% (论文: ~3%)
    
    # === 测试深度 (快速模式，需与模型可用深度匹配) ===
    target_depths: List[float] = field(
        default_factory=lambda: [40, 100, 200, 400]
    )
    # 深度匹配容差（km）：用于非均匀深度轴上的“近似同深度”配对
    depth_match_tolerance_km: float = 0.1

    def __post_init__(self) -> None:
        """核心参数自校验，减少训练时隐性失败。"""
        if self.n_components <= 0:
            raise ValueError("Phase1Config.n_components 必须 > 0")
        if self.alpha <= 0:
            raise ValueError("Phase1Config.alpha 必须 > 0")
        if not (0 < self.train_ratio < 1):
            raise ValueError("Phase1Config.train_ratio 必须在 (0, 1) 区间")
        if self.max_pairs <= 0:
            raise ValueError("Phase1Config.max_pairs 必须 > 0")
        if self.depth_match_tolerance_km <= 0:
            raise ValueError("Phase1Config.depth_match_tolerance_km 必须 > 0")
        if not (0 < self.physical_stride <= self.physical_window_size):
            raise ValueError("Phase1Config 需要满足 0 < physical_stride <= physical_window_size")
        self.target_depths = sorted({float(d) for d in self.target_depths})


@dataclass
class Phase2Config:
    """Phase 2: 3D Patch融合配置（默认针对 SinoScope1.0→FWEA23）。"""
    
    # 启用开关
    enabled: bool = False  # 默认不启用
    
    # === 3D物理窗口配置 ===
    physical_window_lat: float = 8.0      # 8°
    physical_window_lon: float = 8.0      # 8°
    physical_window_depth: float = 240.0  # 240 km
    
    physical_stride_lat: float = 2.0
    physical_stride_lon: float = 2.0
    physical_stride_depth: float = 80.0
    
    # === 字典参数 ===
    n_components: int = 20
    alpha: float = 0.1
    n_nonzero_coefs: int = 6      # 3D可能需要稍多系数
    fit_algorithm: Literal['lars', 'cd'] = 'lars'
    transform_algorithm: Literal['lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'] = 'omp'
    
    # === 深度范围 ===
    depth_range: Tuple[float, float] = (0, 800)  # 覆盖壳幔过渡到上地幔主体

    def __post_init__(self) -> None:
        """Phase 2 参数自校验。"""
        if self.n_components <= 0:
            raise ValueError("Phase2Config.n_components 必须 > 0")
        if self.alpha <= 0:
            raise ValueError("Phase2Config.alpha 必须 > 0")
        zmin, zmax = self.depth_range
        if zmin < 0 or zmax <= zmin:
            raise ValueError("Phase2Config.depth_range 必须满足 0 <= min < max")


@dataclass
class Phase3Config:
    """Phase 3: 全域模型增强配置"""
    
    # 启用开关
    enabled: bool = False  # 默认不启用
    
    # === 过渡区域配置 ===
    transition_width: int = 4  # 过渡区域宽度（网格点数）
    
    # === 融合权重 ===
    use_gaussian_weight: bool = True
    gaussian_sigma: float = 2.0
    
    # === 输出配置 ===
    output_format: Literal['netcdf', 'zarr'] = 'netcdf'


@dataclass
class VisualizationConfig:
    """可视化配置"""
    
    # 输出设置
    dpi: int = 300
    formats: List[str] = field(default_factory=lambda: ['jpg'])
    figure_prefix: str = '4-3_v3-4_'
    
    # 配色方案
    velocity_cmap: str = 'jet_r'
    error_cmap: str = 'RdBu_r'
    atom_cmap: str = 'seismic'
    improvement_cmap: str = 'PiYG'
    
    # 模型颜色
    model_colors: Dict[str, str] = field(default_factory=lambda: {
        'SinoScope1.0': '#3498DB',
        'FWEA23': '#E74C3C',
        'SDL_Enhanced': '#9B59B6'
    })
    
    # 字体设置
    font_size: int = 13
    title_size: int = 16


@dataclass
class SDLConfigV31:
    """SDL融合v3.3总配置（保持类名兼容性）"""
    
    def __init__(self):
        # ============ 模型信息 ============
        # ⭐ 默认模型对: 大范围粗分辨率 SinoScope1.0 (M₁) → 高分辨率 FWEA23 (M₂)
        self.model_info: Dict[str, ModelResolutionInfo] = {
            '2022_SinoScope1.0': ModelResolutionInfo(
                name='SinoScope1.0',
                full_name='2022_SinoScope1.0',
                lat_resolution=1.0,
                lon_resolution=1.0,
                depth_resolution=20.0,
                color='#3498DB',
                marker='o',
                role='base',
                filename='2022_SinoScope1.0_original.nc',
                short_name='sino'
            ),
            '2024_FWEA23': ModelResolutionInfo(
                name='FWEA23',
                full_name='2024_FWEA23',
                lat_resolution=0.25,
                lon_resolution=0.25,
                depth_resolution=10.0,
                color='#E74C3C',
                marker='s',
                role='highres',
                filename='2024_FWEA23_original.nc',
                short_name='fwea'
            )
        }
        
        # ============ 核心参数 ============
        self.base_model_key: str = '2022_SinoScope1.0'
        self.highres_model_keys: List[str] = ['2024_FWEA23']
        self.default_target_model: str = '2024_FWEA23'
        self.velocity_params: List[str] = ['vs', 'vp']
        
        # ============ 阶段配置 ============
        # ⭐ v3.4: 移除 unified_grid，直接使用原始模型分辨率
        self.phase0 = Phase0Config()
        self.phase1 = Phase1Config()
        self.phase2 = Phase2Config()
        self.phase3 = Phase3Config()
        self.visualization = VisualizationConfig()
        
        # ============ 日志配置 ============
        self.logging = {
            'level': 'INFO',
            'console_output': True
        }
    
    def get_base_model_info(self) -> ModelResolutionInfo:
        """获取基础模型信息"""
        return self.model_info[self.base_model_key]

    def resolve_model_key(self, model_key_or_alias: str) -> str:
        """将模型完整键名/简称/显示名统一解析为 model_info 键名。"""
        if model_key_or_alias in self.model_info:
            return model_key_or_alias

        query = model_key_or_alias.strip().lower()
        for key, info in self.model_info.items():
            candidates = {
                key.lower(),
                info.full_name.lower(),
                info.name.lower(),
                info.short_name.lower(),
            }
            if query in candidates:
                return key

        available = ", ".join(sorted(self.model_info.keys()))
        raise KeyError(f"未知模型标识 '{model_key_or_alias}'，可用模型: {available}")
    
    def get_highres_model_info(self, model_key: str) -> ModelResolutionInfo:
        """获取高分辨率模型信息"""
        resolved_key = self.resolve_model_key(model_key)
        info = self.model_info[resolved_key]
        if info.role != 'highres':
            raise ValueError(f"模型 '{resolved_key}' 不是高分辨率模型（role={info.role}）")
        return info
    
    def get_resolution_ratio(self) -> int:
        """计算分辨率比值"""
        base_info = self.get_base_model_info()
        highres_info = self.get_highres_model_info(self.default_target_model)
        return int(base_info.lat_resolution / highres_info.lat_resolution)
    
    def compute_patch_sizes(self, physical_window: float) -> Dict[str, int]:
        """
        根据物理窗口大小计算各模型的patch尺寸
        
        Args:
            physical_window: 物理窗口大小（度）
            
        Returns:
            {'base': base_patch_size, 'highres': highres_patch_size}
        """
        base_info = self.get_base_model_info()
        highres_info = self.get_highres_model_info(self.default_target_model)
        
        base_size = int(physical_window / base_info.lat_resolution)
        highres_size = int(physical_window / highres_info.lat_resolution)
        
        return {
            'base': base_size,
            'highres': highres_size,
            'ratio': highres_size // base_size
        }
    
    def get_matched_depths(self, 
                           base_depths: np.ndarray, 
                           highres_depths: np.ndarray,
                           tolerance: float = 0.1) -> List[float]:
        """
        ⭐ v3.4.2: 获取深度精确匹配的层（论文方法）
        
        论文中两个模型的深度是精确对应的 ("at the same depths")。
        这里只返回 USTClitho2.0 和 CSES_VM1.0 在容差范围内匹配的深度。
        
        Args:
            base_depths: 基础模型深度数组（可非均匀）
            highres_depths: 高分辨率模型深度数组（可非均匀）
            tolerance: 深度匹配容差 (km)
            
        Returns:
            匹配的深度列表 (公共深度)
        """
        matched = []
        for bd in base_depths:
            # 找到高分辨率模型中最接近的深度
            closest_idx = np.argmin(np.abs(highres_depths - bd))
            closest_depth = highres_depths[closest_idx]
            
            # 只有深度差异在容差范围内才算匹配
            if abs(closest_depth - bd) <= tolerance:
                matched.append(float(bd))
        
        return sorted(matched)


# ==============================================================================
# 第三部分: 模型加载器 (Phase -1) ⭐ v3.5 重构
# ==============================================================================


class ModelLoader:
    """
    速度模型加载器 (Phase -1) ⭐ v3.5 重构
    
    简化的模型管理类，只负责：
    1. 加载原始模型 (load_models)
    2. 从原始模型获取数据切片 (get_slice_from_model)
    
    ⭐ v3.5: 完全移除统一网格功能，所有操作直接使用原始模型数据
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.models: Dict[str, xr.Dataset] = {}
        
        # 模型简称映射
        self.short_name_map = {
            '2022_SinoScope1.0': 'sino',
            '2024_FWEA23': 'fwea'
        }
        self.key_from_short = {v: k for k, v in self.short_name_map.items()}
    
    def load_models(self, model_paths: Dict[str, Path]) -> Dict[str, xr.Dataset]:
        """
        加载原始模型
        
        Args:
            model_paths: {model_key: path_to_netcdf}
            
        Returns:
            加载的模型字典
        """
        self.logger.info("\n  📂 加载原始速度模型...")
        
        self.models = {}
        for name, path in model_paths.items():
            self.logger.info(f"    加载 {name}...")
            try:
                ds = xr.open_dataset(path)
                self.models[name] = ds
                self._log_model_info(name, ds)
            except Exception as e:
                self.logger.error(f"    ❌ 加载失败: {e}")
                
        self.logger.info(f"\n  ✅ 成功加载 {len(self.models)} 个模型")
        return self.models
    
    def _log_model_info(self, name: str, ds: xr.Dataset) -> None:
        """记录模型信息"""
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        
        lat = ds.coords[lat_name].values
        lon = ds.coords[lon_name].values
        depth = ds.coords['depth'].values
        
        lat_res = np.abs(np.diff(lat)).mean() if len(lat) > 1 else 0
        lon_res = np.abs(np.diff(lon)).mean() if len(lon) > 1 else 0
        depth_res = np.abs(np.diff(depth)).mean() if len(depth) > 1 else 0
        
        self.logger.info(f"      经度: {lon.min():.2f}° - {lon.max():.2f}° ({len(lon)}点, Δ={lon_res:.2f}°)")
        self.logger.info(f"      纬度: {lat.min():.2f}° - {lat.max():.2f}° ({len(lat)}点, Δ={lat_res:.2f}°)")
        self.logger.info(f"      深度: {depth.min():.1f} - {depth.max():.1f}km ({len(depth)}点, Δ={depth_res:.1f}km)")
        
        if 'vs' in ds.data_vars:
            vs_data = ds['vs'].values
            valid_pct = np.sum(~np.isnan(vs_data)) / vs_data.size * 100
            self.logger.info(f"      Vs有效覆盖: {valid_pct:.1f}%")
    
    def get_model(self, model_key_or_short: str) -> Optional[xr.Dataset]:
        """
        获取模型数据集
        
        Args:
            model_key_or_short: 模型键名或简称 (如 '2022_USTClitho2.0' 或 'ust')
            
        Returns:
            xarray Dataset 或 None
        """
        # 先尝试直接查找
        if model_key_or_short in self.models:
            return self.models[model_key_or_short]
        
        # 尝试从简称查找
        full_key = self.key_from_short.get(model_key_or_short)
        if full_key and full_key in self.models:
            return self.models[full_key]
        
        return None
    
    def get_model_key(self, short_name: str) -> Optional[str]:
        """从简称获取完整模型键名"""
        return self.key_from_short.get(short_name)
    
    def get_slice_from_model(self, model_key: str, param: str, 
                             depth_km: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ⭐ v3.5: 直接从原始模型获取深度切片（不插值）
        
        Args:
            model_key: 模型键名 (如 '2022_USTClitho2.0') 或简称 (如 'ust')
            param: 参数名 ('vs' or 'vp')
            depth_km: 深度 (km)
            
        Returns:
            (velocity_slice, lat_coords, lon_coords) 原始分辨率的数据
            velocity_slice: (lat, lon) 形状
        """
        model = self.get_model(model_key)
        if model is None:
            raise ValueError(f"模型 {model_key} 未加载")
        
        if param not in model.data_vars:
            raise ValueError(f"参数 {param} 在模型 {model_key} 中不存在")
        
        # 获取坐标名称
        lat_name = 'latitude' if 'latitude' in model.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in model.coords else 'lon'
        
        lat = model.coords[lat_name].values
        lon = model.coords[lon_name].values
        depth = model.coords['depth'].values
        
        # 找最近深度
        depth_idx = int(np.argmin(np.abs(depth - depth_km)))
        
        # 获取数据
        data = model[param].values
        dims = [str(d) for d in model[param].dims]
        
        # 提取深度切片
        slice_data = self._extract_depth_slice(data, dims, lat_name, lon_name, depth_idx)
        
        return slice_data, lat, lon
    
    def _extract_depth_slice(self, data: np.ndarray, dims: Sequence[str],
                             lat_name: str, lon_name: str, 
                             depth_idx: int) -> np.ndarray:
        """
        从3D数据中提取深度切片
        
        Returns:
            (lat, lon) 形状的2D数组
        """
        # 找到各维度索引
        depth_dim_idx = None
        lat_dim_idx = None
        lon_dim_idx = None
        
        for i, dim in enumerate(dims):
            if 'dep' in dim.lower():
                depth_dim_idx = i
            elif 'lat' in dim.lower():
                lat_dim_idx = i
            elif 'lon' in dim.lower():
                lon_dim_idx = i
        
        if depth_dim_idx is None:
            raise ValueError(f"无法找到深度维度: {dims}")
        
        # 构建索引
        slices: List[Any] = [slice(None)] * len(dims)
        slices[depth_dim_idx] = depth_idx
        
        slice_2d = data[tuple(slices)]
        
        # 确保返回 (lat, lon) 顺序
        if lat_dim_idx is not None and lon_dim_idx is not None:
            if lat_dim_idx > lon_dim_idx:
                slice_2d = slice_2d.T
        
        return slice_2d
    
    def get_model_resolution(self, model_key: str) -> Tuple[float, float, float]:
        """
        获取模型分辨率
        
        Returns:
            (lat_res, lon_res, depth_res)
        """
        model = self.get_model(model_key)
        if model is None:
            raise ValueError(f"模型 {model_key} 未加载")
        
        lat_name = 'latitude' if 'latitude' in model.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in model.coords else 'lon'
        
        lat = model.coords[lat_name].values
        lon = model.coords[lon_name].values
        depth = model.coords['depth'].values
        
        lat_res = np.abs(np.diff(lat)).mean() if len(lat) > 1 else 1.0
        lon_res = np.abs(np.diff(lon)).mean() if len(lon) > 1 else 1.0
        depth_res = np.abs(np.diff(depth)).mean() if len(depth) > 1 else 10.0
        
        return lat_res, lon_res, depth_res
    
    def get_model_depths(self, model_key: str) -> np.ndarray:
        """获取模型的深度坐标"""
        model = self.get_model(model_key)
        if model is None:
            raise ValueError(f"模型 {model_key} 未加载")
        return model.coords['depth'].values
    
    def get_model_bounds(self, model_key: str) -> Dict[str, Tuple[float, float]]:
        """
        获取模型的边界范围
        
        Returns:
            {'lat': (min, max), 'lon': (min, max), 'depth': (min, max)}
        """
        model = self.get_model(model_key)
        if model is None:
            raise ValueError(f"模型 {model_key} 未加载")
        
        lat_name = 'latitude' if 'latitude' in model.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in model.coords else 'lon'
        
        lat = model.coords[lat_name].values
        lon = model.coords[lon_name].values
        depth = model.coords['depth'].values
        
        return {
            'lat': (float(lat.min()), float(lat.max())),
            'lon': (float(lon.min()), float(lon.max())),
            'depth': (float(depth.min()), float(depth.max()))
        }
    
    def get_overlap_region(self, model_key1: str, model_key2: str,
                           param: str = 'vs', depth_km: float = 100) -> Dict:
        """
        ⭐ v3.5: 计算两个模型在指定深度的重叠区域
        
        基于实际数据（非NaN）计算重叠
        
        Returns:
            {'lat_range': (min, max), 'lon_range': (min, max), 
             'overlap_ratio': float, 'bounds1': dict, 'bounds2': dict}
        """
        # 获取两个模型的切片
        slice1, lat1, lon1 = self.get_slice_from_model(model_key1, param, depth_km)
        slice2, lat2, lon2 = self.get_slice_from_model(model_key2, param, depth_km)
        
        # 计算有效数据的经纬度范围
        valid1 = ~np.isnan(slice1)
        valid2 = ~np.isnan(slice2)
        
        # 模型1的有效范围
        lat1_valid = lat1[np.any(valid1, axis=1)]
        lon1_valid = lon1[np.any(valid1, axis=0)]
        
        # 模型2的有效范围
        lat2_valid = lat2[np.any(valid2, axis=1)]
        lon2_valid = lon2[np.any(valid2, axis=0)]
        
        # 计算重叠范围
        lat_overlap = (max(lat1_valid.min(), lat2_valid.min()),
                      min(lat1_valid.max(), lat2_valid.max()))
        lon_overlap = (max(lon1_valid.min(), lon2_valid.min()),
                      min(lon1_valid.max(), lon2_valid.max()))
        
        # 检查是否有重叠
        has_overlap = (lat_overlap[0] < lat_overlap[1] and 
                      lon_overlap[0] < lon_overlap[1])
        
        if has_overlap:
            overlap_area = (lat_overlap[1] - lat_overlap[0]) * (lon_overlap[1] - lon_overlap[0])
            model1_area = (lat1_valid.max() - lat1_valid.min()) * (lon1_valid.max() - lon1_valid.min())
            overlap_ratio = overlap_area / model1_area if model1_area > 0 else 0
        else:
            overlap_ratio = 0
        
        return {
            'lat_range': lat_overlap if has_overlap else None,
            'lon_range': lon_overlap if has_overlap else None,
            'has_overlap': has_overlap,
            'overlap_ratio': overlap_ratio,
            'bounds1': {'lat': (lat1_valid.min(), lat1_valid.max()), 
                       'lon': (lon1_valid.min(), lon1_valid.max())},
            'bounds2': {'lat': (lat2_valid.min(), lat2_valid.max()),
                       'lon': (lon2_valid.min(), lon2_valid.max())}
        }


# ==============================================================================
# 第四部分: Phase 0 - 字典表示能力验证器
# ==============================================================================

class DictionaryValidator:
    """
    Phase 0: 字典表示能力验证器
    
    验证SDL方法是否能够准确重建速度模型
    
    论文参考 (Section 3.1):
    "As the basis of our methodology... we first demonstrate the capability of 
    representing a seismic velocity model using a dictionary with minimal error."
    """
    
    def __init__(self, config: Phase0Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.dict_learner: Optional[DictionaryLearning] = None
        self.mean_patch: Optional[np.ndarray] = None
    
    def extract_patches_from_slice(self,
                                   velocity_slice: np.ndarray,
                                   lat_coords: np.ndarray,
                                   lon_coords: np.ndarray,
                                   window_size: int,
                                   stride: int) -> ValidationPatchResult:
        """
        从2D速度切片提取所有patches
        
        Args:
            velocity_slice: (N_lat, N_lon) 速度切片
            lat_coords: 纬度坐标数组
            lon_coords: 经度坐标数组
            window_size: 窗口大小（格点数）
            stride: 步长（格点数）
            
        Returns:
            ValidationPatchResult
        """
        n_lat, n_lon = velocity_slice.shape
        patches_list = []
        coords_list = []
        
        for i in range(0, n_lat - window_size + 1, stride):
            for j in range(0, n_lon - window_size + 1, stride):
                patch = velocity_slice[i:i+window_size, j:j+window_size]
                
                # 检查NaN
                if np.any(np.isnan(patch)):
                    continue
                
                patches_list.append(patch.ravel())
                
                # 记录中心坐标
                lat_c = (lat_coords[i] + lat_coords[min(i+window_size-1, n_lat-1)]) / 2
                lon_c = (lon_coords[j] + lon_coords[min(j+window_size-1, n_lon-1)]) / 2
                coords_list.append([lat_c, lon_c])
        
        patches = np.array(patches_list) if patches_list else np.array([]).reshape(0, window_size*window_size)
        coords = np.array(coords_list) if coords_list else np.array([]).reshape(0, 2)
        
        return ValidationPatchResult(
            patches=patches,
            coords=coords,
            velocity_slice=velocity_slice,
            lat_coords=lat_coords,
            lon_coords=lon_coords,
            n_patches=len(patches),
            patch_size=window_size,
            depth_km=0.0  # 将在调用时设置
        )
    
    def train_dictionary(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        训练字典（公式3）
        
        Args:
            patches: (N, L) patches矩阵
            
        Returns:
            D: (K, L) 字典矩阵
            C: (N, K) 稀疏系数矩阵
        """
        # 中心化
        self.mean_patch = np.mean(patches, axis=0)
        patches_centered = patches - self.mean_patch
        
        # 字典学习
        self.dict_learner = DictionaryLearning(
            n_components=self.config.n_components,
            alpha=self.config.alpha,
            max_iter=self.config.max_iter,
            transform_algorithm=self.config.transform_algorithm,
            transform_n_nonzero_coefs=self.config.n_nonzero_coefs,
            fit_algorithm=self.config.fit_algorithm,
            random_state=self.config.random_state
        )
        
        self.dict_learner.fit(patches_centered)
        
        D = self.dict_learner.components_  # (K, L)
        C = self.dict_learner.transform(patches_centered)  # (N, K)
        
        return D, C
    
    def reconstruct_patches(self, D: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        使用字典重建patches
        
        Args:
            D: (K, L) 字典
            C: (N, K) 稀疏系数
            
        Returns:
            reconstructed: (N, L) 重建的patches
        """
        if self.mean_patch is None:
            raise ValueError("请先调用train_dictionary()")
        
        reconstructed_centered = C @ D  # (N, L)
        reconstructed = reconstructed_centered + self.mean_patch
        return reconstructed
    
    def compute_reconstruction_error(self,
                                     original: np.ndarray,
                                     reconstructed: np.ndarray) -> Dict[str, float]:
        """
        计算重建误差（公式9）
        
        Returns:
            error_stats: 误差统计字典
        """
        # 全局相对误差（论文公式9）
        global_error = float(np.linalg.norm(original - reconstructed) / np.linalg.norm(original) * 100)
        
        # 逐点相对误差
        original_safe = np.where(np.abs(original) > 1e-6, original, 1e-6)
        point_errors = (reconstructed - original) / original_safe * 100
        point_errors_flat = point_errors.ravel()
        
        return {
            'global_error': global_error,
            'mean_error': float(np.mean(np.abs(point_errors_flat))),
            'std_error': float(np.std(point_errors_flat)),
            'max_error': float(np.max(np.abs(point_errors_flat))),
            'min_error': float(np.min(point_errors_flat)),
            'median_error': float(np.median(np.abs(point_errors_flat))),
            'percentile_5': float(np.percentile(point_errors_flat, 5)),
            'percentile_95': float(np.percentile(point_errors_flat, 95))
        }
    
    def validate_model_at_depth(self,
                                velocity_slice: np.ndarray,
                                lat_coords: np.ndarray,
                                lon_coords: np.ndarray,
                                depth_km: float,
                                model_name: str,
                                resolution: float) -> ValidationResult:
        """
        在指定深度验证单个模型
        
        Args:
            velocity_slice: (lat, lon) 速度切片
            lat_coords: 纬度坐标
            lon_coords: 经度坐标
            depth_km: 深度
            model_name: 模型名称
            resolution: 模型分辨率（度）
            
        Returns:
            ValidationResult
        """
        self.logger.info(f"\n    🔬 验证 {model_name} @ {depth_km}km")
        
        # 计算窗口大小和步长 (格点数)
        window_size = int(self.config.physical_window_size / resolution)
        stride = max(1, int(self.config.physical_stride / resolution))  # 物理步长 → 格点数
        
        self.logger.info(f"      窗口: {window_size}×{window_size} 格点 ({self.config.physical_window_size}°)")
        self.logger.info(f"      步长: {stride} 格点 ({self.config.physical_stride}°)")
        self.logger.info(f"      切片形状: {velocity_slice.shape}")
        self.logger.info(f"      速度范围: [{np.nanmin(velocity_slice):.3f}, {np.nanmax(velocity_slice):.3f}] km/s")
        
        # Step 1: 提取patches
        patch_result = self.extract_patches_from_slice(
            velocity_slice, lat_coords, lon_coords, window_size, stride
        )
        
        if patch_result.n_patches < self.config.n_components:
            self.logger.warning(f"      ⚠️ Patches数量({patch_result.n_patches}) < K({self.config.n_components})")
            return ValidationResult(
                depth_km=depth_km,
                model_name=model_name,
                n_patches=patch_result.n_patches,
                dictionary=np.zeros((0, 0)),
                coefficients=np.zeros((0, 0)),
                patches_original=patch_result.patches,
                patches_reconstructed=np.zeros((0, 0)),
                coords=patch_result.coords,
                velocity_slice=velocity_slice,
                lat_coords=lat_coords,
                lon_coords=lon_coords,
                error_stats={'global_error': np.nan},
                passed=False,
                mean_patch=np.zeros(0),
                patch_size=window_size
            )
        
        self.logger.info(f"      提取patches: {patch_result.n_patches}")
        
        # Step 2: 训练字典
        D, C = self.train_dictionary(patch_result.patches)
        sparsity = float(np.mean(np.abs(C) > 1e-6)) * 100
        self.logger.info(f"      字典形状: {D.shape}")
        self.logger.info(f"      稀疏度: {sparsity:.1f}% 非零")
        
        # Step 3: 重建
        reconstructed = self.reconstruct_patches(D, C)
        
        # Step 4: 计算误差
        error_stats = self.compute_reconstruction_error(patch_result.patches, reconstructed)
        
        self.logger.info(f"      全局误差: {error_stats['global_error']:.2f}%")
        self.logger.info(f"      平均误差: {error_stats['mean_error']:.2f}%")
        self.logger.info(f"      最大误差: {error_stats['max_error']:.2f}%")
        
        # 验收检查
        passed = (
            error_stats['mean_error'] < self.config.max_mean_error and
            error_stats['max_error'] < self.config.max_point_error
        )
        
        self.logger.info(f"      验收: {'✅ 通过' if passed else '❌ 未通过'}")
        
        return ValidationResult(
            depth_km=depth_km,
            model_name=model_name,
            n_patches=patch_result.n_patches,
            dictionary=D,
            coefficients=C,
            patches_original=patch_result.patches,
            patches_reconstructed=reconstructed,
            coords=patch_result.coords,
            velocity_slice=velocity_slice,
            lat_coords=lat_coords,
            lon_coords=lon_coords,
            error_stats=error_stats,
            passed=passed,
            mean_patch=self.mean_patch if self.mean_patch is not None else np.zeros(0),
            patch_size=window_size
        )


# ==============================================================================
# 第五部分: Phase 1 - 2D水平切片变换器
# ==============================================================================

# ⭐ v3.4: 移除 PairedPatchExtractor，只保留 DirectPatchExtractor（论文方法）


class DirectPatchExtractor:
    """
    直接从原始网格提取配对 Patches（论文方法）⭐ v3.4 唯一的提取方法
    
    核心特性:
    - 不需要预先插值到统一网格
    - 直接从 USTClitho2.0 原始 1° 网格提取低分辨率 patches
    - 直接从 CSES 原始 0.25° 网格提取高分辨率 patches
    
    论文方法:
        同一物理窗口 (例如 6° × 6°):
        - M₁ patch: 6×6 格点 @ 1° = 6° 物理窗口  → L₁ = 36
        - M₂ patch: 24×24 格点 @ 0.25° = 6° 物理窗口 → L₂ = 576
    """
    
    def __init__(self, config: Phase1Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def extract_paired_patches_direct(self,
                                      base_ds: xr.Dataset,
                                      highres_ds: xr.Dataset,
                                      param: str,
                                      depth_km: float,
                                      base_resolution: float = 1.0,
                                      highres_resolution: float = 0.25) -> PairedPatchResult:
        """
        直接从原始分辨率网格提取配对 patches
        
        核心思想: 
        - 不对 USTClitho2.0 进行上采样
        - 直接从原始 1° 网格提取 6×6 patches
        - 直接从原始 0.25° 网格提取 24×24 patches
        
        Args:
            base_ds: 基础模型原始 Dataset (USTClitho2.0 @ 1°)
            highres_ds: 高分辨率模型原始 Dataset (CSES @ 0.25°)
            param: 参数名 ('vs' or 'vp')
            depth_km: 深度
            base_resolution: 基础模型分辨率（度）
            highres_resolution: 高分辨率分辨率（度）
            
        Returns:
            PairedPatchResult
        """
        self.logger.info(f"\n    📦 [Direct] 直接从原始网格提取配对 patches...")
        
        physical_window = self.config.physical_window_size  # 6°
        physical_stride = self.config.physical_stride       # 3°
        
        # 计算各自网格上的 patch 尺寸
        base_patch_size = int(physical_window / base_resolution)      # 6
        highres_patch_size = int(physical_window / highres_resolution)  # 24
        
        self.logger.info(f"      物理窗口: {physical_window}° × {physical_window}°")
        self.logger.info(f"      基础模型 patch: {base_patch_size}×{base_patch_size} @ {base_resolution}°")
        self.logger.info(f"      高分辨率 patch: {highres_patch_size}×{highres_patch_size} @ {highres_resolution}°")
        
        # 获取坐标名称
        base_lat_name = 'latitude' if 'latitude' in base_ds.coords else 'lat'
        base_lon_name = 'longitude' if 'longitude' in base_ds.coords else 'lon'
        highres_lat_name = 'latitude' if 'latitude' in highres_ds.coords else 'lat'
        highres_lon_name = 'longitude' if 'longitude' in highres_ds.coords else 'lon'
        
        # 获取坐标
        base_lat = base_ds.coords[base_lat_name].values
        base_lon = base_ds.coords[base_lon_name].values
        highres_lat = highres_ds.coords[highres_lat_name].values
        highres_lon = highres_ds.coords[highres_lon_name].values
        base_depth = base_ds.coords['depth'].values
        highres_depth = highres_ds.coords['depth'].values
        
        # 找到最近的深度索引
        base_depth_idx = int(np.argmin(np.abs(base_depth - depth_km)))
        highres_depth_idx = int(np.argmin(np.abs(highres_depth - depth_km)))
        
        self.logger.info(f"      基础模型深度: {base_depth[base_depth_idx]:.1f}km (idx={base_depth_idx})")
        self.logger.info(f"      高分辨率深度: {highres_depth[highres_depth_idx]:.1f}km (idx={highres_depth_idx})")
        
        # 获取速度切片
        base_data = base_ds[param].values
        highres_data = highres_ds[param].values
        
        # 检查数据维度顺序并调整
        base_dims = [str(d) for d in base_ds[param].dims]
        highres_dims = [str(d) for d in highres_ds[param].dims]
        
        # 提取指定深度的切片 - 需要根据维度顺序处理
        base_slice = self._extract_depth_slice(base_data, base_dims, 
                                                base_lat_name, base_lon_name, 
                                                base_depth_idx)
        highres_slice = self._extract_depth_slice(highres_data, highres_dims,
                                                   highres_lat_name, highres_lon_name,
                                                   highres_depth_idx)
        
        self.logger.info(f"      基础模型切片: {base_slice.shape}")
        self.logger.info(f"      高分辨率切片: {highres_slice.shape}")
        
        # 计算重叠区域
        lat_min_overlap = max(base_lat.min(), highres_lat.min()) + physical_window / 2
        lat_max_overlap = min(base_lat.max(), highres_lat.max()) - physical_window / 2
        lon_min_overlap = max(base_lon.min(), highres_lon.min()) + physical_window / 2
        lon_max_overlap = min(base_lon.max(), highres_lon.max()) - physical_window / 2
        
        self.logger.info(f"      重叠区域: lat [{lat_min_overlap:.1f}, {lat_max_overlap:.1f}], "
                        f"lon [{lon_min_overlap:.1f}, {lon_max_overlap:.1f}]")
        
        # 生成 patch 中心坐标
        lat_centers = np.arange(lat_min_overlap, lat_max_overlap + 0.001, physical_stride)
        lon_centers = np.arange(lon_min_overlap, lon_max_overlap + 0.001, physical_stride)
        
        self.logger.info(f"      网格: {len(lat_centers)} × {len(lon_centers)} = {len(lat_centers)*len(lon_centers)} 候选位置")
        
        patches_base_list = []
        patches_highres_list = []
        coords_list = []
        cc_list = []
        
        half_window = physical_window / 2
        total_positions = len(lat_centers) * len(lon_centers)
        
        # 使用 tqdm 显示进度
        pbar = tqdm(total=total_positions, desc=f"      Patches@{depth_km}km",
                    unit="pos", ncols=100, file=sys.stdout)
        
        for lat_c in lat_centers:
            for lon_c in lon_centers:
                pbar.update(1)
                # 物理坐标边界
                lat_start = lat_c - half_window
                lat_end = lat_c + half_window
                lon_start = lon_c - half_window
                lon_end = lon_c + half_window
                
                # === 从基础模型原始网格直接提取 ===
                base_patch = self._extract_patch_from_original(
                    base_slice, base_lat, base_lon,
                    lat_start, lat_end, lon_start, lon_end,
                    base_patch_size
                )
                
                if base_patch is None:
                    continue
                
                # === 从高分辨率模型原始网格直接提取 ===
                highres_patch = self._extract_patch_from_original(
                    highres_slice, highres_lat, highres_lon,
                    lat_start, lat_end, lon_start, lon_end,
                    highres_patch_size
                )
                
                if highres_patch is None:
                    continue
                
                # 计算 CC（上采样低分辨率 patch 后比较）
                base_upsampled = np.asarray(
                    zoom(base_patch, highres_patch_size / base_patch_size, order=3)
                )
                
                if base_upsampled.shape != highres_patch.shape:
                    # 微调尺寸
                    base_upsampled = np.asarray(
                        zoom(
                            base_patch,
                            (highres_patch.shape[0] / base_patch.shape[0],
                             highres_patch.shape[1] / base_patch.shape[1]),
                            order=3,
                        )
                    )
                
                try:
                    cc, _ = pearsonr(base_upsampled.ravel(), highres_patch.ravel())
                except:
                    cc = 0.0
                
                # 存储
                patches_base_list.append(base_patch.ravel())
                patches_highres_list.append(highres_patch.ravel())
                coords_list.append([lat_c, lon_c])
                cc_list.append(cc)
        
        pbar.close()
        
        # 转换为数组
        n_total = len(patches_base_list)
        
        if n_total == 0:
            self.logger.warning("      ⚠️ [Direct] 未找到有效的配对 patches!")
            return PairedPatchResult(
                patches_base=np.array([]).reshape(0, base_patch_size**2),
                patches_highres=np.array([]).reshape(0, highres_patch_size**2),
                coords_latlon=np.array([]).reshape(0, 2),
                correlations=np.array([]),
                valid_mask=np.array([]),
                n_total=0,
                n_valid=0,
                physical_window_size=physical_window,
                base_patch_size=base_patch_size,
                highres_patch_size=highres_patch_size,
                depth_km=depth_km
            )
        
        patches_base = np.array(patches_base_list)
        patches_highres = np.array(patches_highres_list)
        coords = np.array(coords_list)
        correlations = np.array(cc_list)
        
        # CC 筛选
        valid_mask = correlations > self.config.min_cc_threshold
        n_valid = int(np.sum(valid_mask))
        
        self.logger.info(f"      [Direct] 提取 patches: {n_total}")
        self.logger.info(f"      [Direct] CC > {self.config.min_cc_threshold}: {n_valid} ({100*n_valid/n_total:.1f}%)")
        self.logger.info(f"      [Direct] CC 范围: [{correlations.min():.3f}, {correlations.max():.3f}]")
        self.logger.info(f"      [Direct] CC 均值: {correlations.mean():.3f}")
        
        return PairedPatchResult(
            patches_base=patches_base,
            patches_highres=patches_highres,
            coords_latlon=coords,
            correlations=correlations,
            valid_mask=valid_mask,
            n_total=n_total,
            n_valid=n_valid,
            physical_window_size=physical_window,
            base_patch_size=base_patch_size,
            highres_patch_size=highres_patch_size,
            base_slice=base_slice,
            highres_slice=highres_slice,
            base_lat=base_lat,
            base_lon=base_lon,
            highres_lat=highres_lat,
            highres_lon=highres_lon,
            depth_km=depth_km
        )
    
    def _extract_depth_slice(self, data: np.ndarray, dims: Sequence[str],
                             lat_name: str, lon_name: str,
                             depth_idx: int) -> np.ndarray:
        """
        根据维度顺序提取指定深度的切片
        
        Returns:
            (lat, lon) 形状的切片
        """
        depth_dim_idx = dims.index('depth')
        
        # 提取深度切片 - 使用 np.take 避免类型问题
        slice_data = np.take(data, depth_idx, axis=depth_dim_idx)
        
        # 剩余维度索引（去掉 depth）
        remaining_dims = [d for i, d in enumerate(dims) if i != depth_dim_idx]
        
        # 转置为 (lat, lon) 顺序
        if remaining_dims[0] == lon_name:
            # 当前是 (lon, lat)，需要转置
            slice_data = slice_data.T
        
        return slice_data
    
    def _extract_patch_from_original(self, 
                                     slice_data: np.ndarray,
                                     lat_coords: np.ndarray,
                                     lon_coords: np.ndarray,
                                     lat_start: float, lat_end: float,
                                     lon_start: float, lon_end: float,
                                     expected_size: int) -> Optional[np.ndarray]:
        """
        从原始网格直接提取 patch
        
        Args:
            slice_data: (lat, lon) 形状的切片
            lat_coords, lon_coords: 坐标数组
            lat_start, lat_end, lon_start, lon_end: 物理坐标边界
            expected_size: 期望的 patch 边长
            
        Returns:
            patch 或 None（如果无法提取）
        """
        # 找到最近的网格索引
        lat_idx_start = np.argmin(np.abs(lat_coords - lat_start))
        lat_idx_end = lat_idx_start + expected_size
        lon_idx_start = np.argmin(np.abs(lon_coords - lon_start))
        lon_idx_end = lon_idx_start + expected_size
        
        # 边界检查
        if lat_idx_end > len(lat_coords) or lon_idx_end > len(lon_coords):
            return None
        
        # 提取 patch
        patch = slice_data[lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end]
        
        # 检查尺寸
        if patch.shape != (expected_size, expected_size):
            return None
        
        # 检查 NaN
        if np.any(np.isnan(patch)):
            return None
        
        return patch


class CoupledDictionaryLearner:
    """
    耦合字典学习器
    
    学习低分辨率字典D₁和高分辨率字典D₂之间的对应关系
    
    核心约束: 同一物理位置的patches共享相同的稀疏系数
    """
    
    def __init__(self, config: Phase1Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.dict_learner: Optional[DictionaryLearning] = None
        self.preprocessing: Optional[PreprocessingInfo] = None
    
    def train(self, paired_patches: PairedPatchResult) -> CoupledDictionaryResult:
        """
        训练耦合字典
        
        Args:
            paired_patches: 配对的patches
            
        Returns:
            CoupledDictionaryResult
        """
        self.logger.info(f"\n    📚 训练耦合字典...")
        
        # 获取有效patches
        valid_mask = paired_patches.valid_mask
        P1_all = paired_patches.patches_base[valid_mask]
        P2_all = paired_patches.patches_highres[valid_mask]
        
        n_samples = len(P1_all)
        
        if n_samples < self.config.n_components:
            raise ValueError(f"有效样本数({n_samples}) < K({self.config.n_components})")
        
        # 限制最大样本数
        if n_samples > self.config.max_pairs:
            indices = np.random.choice(n_samples, self.config.max_pairs, replace=False)
            P1_all = P1_all[indices]
            P2_all = P2_all[indices]
            n_samples = self.config.max_pairs
            self.logger.info(f"      限制样本数: {n_samples}")
        
        # 划分训练/验证集
        n_train = int(n_samples * self.config.train_ratio)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        P1_train, P1_val = P1_all[train_idx], P1_all[val_idx]
        P2_train, P2_val = P2_all[train_idx], P2_all[val_idx]
        
        self.logger.info(f"      训练集: {len(P1_train)}")
        self.logger.info(f"      验证集: {len(P1_val)}")
        
        # ⭐ v3.4.2: 移除 centering，与论文原始方法一致
        # 论文公式 (6): D₁, C = argmin ||P₁ᵀ - D₁C||² + λ||C||₀
        # 直接使用原始 patches，不做 centering 预处理
        self.preprocessing = PreprocessingInfo(
            method='none',  # ⭐ 不做 centering
            mean_P1=np.zeros(P1_train.shape[1]),  # 零向量（占位）
            mean_P2=np.zeros(P2_train.shape[1])   # 零向量（占位）
        )
        
        # 直接使用原始数据（与论文一致）
        P1_train_for_dict = P1_train
        P2_train_for_dict = P2_train
        
        self.logger.info(f"      预处理: 无 centering（论文方法）")
        
        # Step 2: 训练D₁和C (公式6)
        self.logger.info(f"      训练D₁...")
        self.dict_learner = DictionaryLearning(
            n_components=self.config.n_components,
            alpha=self.config.alpha,
            max_iter=self.config.max_iter,
            transform_algorithm=self.config.transform_algorithm,
            transform_n_nonzero_coefs=self.config.n_nonzero_coefs,
            fit_algorithm=self.config.fit_algorithm,
            random_state=self.config.random_state
        )
        
        self.dict_learner.fit(P1_train_for_dict)
        D1 = self.dict_learner.components_  # (K, L₁)
        C_train = self.dict_learner.transform(P1_train_for_dict)  # (N_train, K)
        
        self.logger.info(f"      D₁形状: {D1.shape}")
        
        # Step 3: 计算D₂ (公式7)
        # ⭐ 论文使用纯最小二乘（无正则化），当前支持两种方法
        self.logger.info(f"      计算D₂ (正则化={self.config.d2_regularization})...")
        
        if self.config.d2_regularization == 0:
            # 论文方法：纯最小二乘 ⭐ 推荐
            # 公式 (7): D₂ = argmin ||P₂ᵀ - D₂Cᵀ||²
            # lstsq(A, B) 返回 X，使得 A @ X ≈ B
            # C_train: (N_train, K), P2: (N_train, L₂) -> X: (K, L₂)
            D2_T = np.linalg.lstsq(C_train, P2_train_for_dict, rcond=None)[0]  # (K, L₂)
            D2 = D2_T  # 不需要转置！lstsq 直接返回 (K, L₂)
            self.logger.info(f"      使用纯最小二乘（论文方法）")
        else:
            # 带 L2 正则化的岭回归（防止过拟合）
            # Ridge.coef_ 形状是 (n_targets, n_features) = (L₂, K)
            ridge = Ridge(alpha=self.config.d2_regularization, fit_intercept=False)
            ridge.fit(C_train, P2_train_for_dict)
            D2 = ridge.coef_.T  # (L₂, K).T -> (K, L₂)
            self.logger.info(f"      使用岭回归（α={self.config.d2_regularization}）")
        
        self.logger.info(f"      D₂形状: {D2.shape}")
        
        # Step 4: 训练集误差 (⭐ v3.4.2: 无 centering，直接计算)
        P1_train_recon = C_train @ D1  # 直接重建，无需加 mean
        P2_train_recon = C_train @ D2
        
        train_error_base = float(np.linalg.norm(P1_train - P1_train_recon) / 
                                np.linalg.norm(P1_train) * 100)
        train_error_highres = float(np.linalg.norm(P2_train - P2_train_recon) / 
                                   np.linalg.norm(P2_train) * 100)
        
        self.logger.info(f"      训练误差(base): {train_error_base:.2f}%")
        self.logger.info(f"      训练误差(highres): {train_error_highres:.2f}%")
        
        # Step 5: 验证集评估 (⭐ v3.4.2: 无 centering)
        # 公式 (8): C^V = argmin ||P₁^V - D₁C||² + λ||C||₀
        C_val = self.dict_learner.transform(P1_val)  # 直接变换，无需 centering
        
        # SDL变换: P̂^V = D₂ C^V
        P2_val_sdl = C_val @ D2  # 直接重建，无需加 mean
        
        # Baseline上采样
        L1 = P1_val.shape[1]
        L2 = P2_val.shape[1]
        patch_size_base = int(np.sqrt(L1))
        patch_size_highres = int(np.sqrt(L2))
        scale = patch_size_highres / patch_size_base
        
        P2_val_baseline = np.zeros_like(P2_val)
        for i in range(len(P1_val)):
            base_patch = P1_val[i].reshape(patch_size_base, patch_size_base)
            upsampled = np.asarray(zoom(base_patch, scale, order=3))
            P2_val_baseline[i] = upsampled.ravel()
        
        # 计算误差 (⭐ v3.4.2: 无 centering)
        val_error_base = float(np.linalg.norm(P1_val - (C_val @ D1)) /
                              np.linalg.norm(P1_val) * 100)
        val_error_highres = float(np.linalg.norm(P2_val - P2_val_sdl) /
                                 np.linalg.norm(P2_val) * 100)
        
        transform_error = val_error_highres
        baseline_error = float(np.linalg.norm(P2_val - P2_val_baseline) /
                              np.linalg.norm(P2_val) * 100)
        
        relative_improvement = (baseline_error - transform_error) / baseline_error * 100
        
        self.logger.info(f"\n      📊 验证集评估:")
        self.logger.info(f"        SDL变换误差: {transform_error:.2f}%")
        self.logger.info(f"        Baseline误差: {baseline_error:.2f}%")
        self.logger.info(f"        相对改善: {relative_improvement:.1f}%")
        
        # 计算误差分布
        point_errors = (P2_val_sdl - P2_val) / np.where(np.abs(P2_val) > 1e-6, P2_val, 1e-6) * 100
        
        error_distribution = {
            'mean': float(np.mean(point_errors)),
            'std': float(np.std(point_errors)),
            'min': float(np.min(point_errors)),
            'max': float(np.max(point_errors)),
            'percentile_5': float(np.percentile(point_errors, 5)),
            'percentile_95': float(np.percentile(point_errors, 95))
        }
        
        return CoupledDictionaryResult(
            D1=D1,
            D2=D2,
            C_train=C_train,
            C_val=C_val,
            train_indices=train_idx,
            val_indices=val_idx,
            preprocessing=self.preprocessing,
            train_error_base=train_error_base,
            train_error_highres=train_error_highres,
            val_error_base=val_error_base,
            val_error_highres=val_error_highres,
            transform_error=transform_error,
            baseline_error=baseline_error,
            relative_improvement=relative_improvement,
            error_distribution=error_distribution,
            dict_learner=self.dict_learner,
            depth_km=paired_patches.depth_km
        )
    
    def transform(self, P1: np.ndarray) -> np.ndarray:
        """
        使用训练好的字典变换低分辨率patches到高分辨率
        
        Args:
            P1: (N, L₁) 低分辨率patches
            
        Returns:
            P2_transformed: (N, L₂) 变换后的高分辨率patches
        """
        if self.dict_learner is None or self.preprocessing is None:
            raise ValueError("请先调用train()")
        
        P1_centered = P1 - self.preprocessing.mean_P1
        C = self.dict_learner.transform(P1_centered)
        D2 = self.dict_learner.components_  # 需要从训练结果获取
        
        # 这里需要D2，实际使用时从CoupledDictionaryResult获取
        raise NotImplementedError("使用CoupledDictionaryResult.D2进行变换")


class TwoDTransformer:
    """
    2D水平切片变换器 (Phase 1)
    
    在2D水平切片上执行SDL变换
    
    ⭐ v3.4: 只使用 DirectPatchExtractor（论文方法）
    """
    
    def __init__(self, config: Phase1Config, sdl_config: SDLConfigV31, 
                 logger: logging.Logger):
        self.config = config
        self.sdl_config = sdl_config
        self.logger = logger
        
        # ⭐ v3.4: 只使用直接提取方法
        self.patch_extractor = DirectPatchExtractor(config, logger)
        self.dict_learner = CoupledDictionaryLearner(config, logger)
        
        self.logger.info(f"    📦 Patch提取方法: direct (论文方法)")
        
        # 存储训练结果
        self.trained_dicts: Dict[float, CoupledDictionaryResult] = {}
    
    def run(self,
            model_loader: 'ModelLoader',
            param: str,
            depth_km: float,
            target_model: str = 'fwea') -> TransformResult:
        """
        在指定深度执行2D变换
        
        ⭐ v3.5: 使用 ModelLoader 替代 UnifiedModelFramework
        
        Args:
            model_loader: 模型加载器
            param: 参数名 ('vs' or 'vp')
            depth_km: 深度
            target_model: 目标模型简称 ('cses' or 'ust')
            
        Returns:
            TransformResult
        """
        self.logger.info(f"\n  🔄 Phase 1: 2D变换 @ {depth_km}km ({param})")
        self.logger.info(f"    目标模型: {target_model}")
        self.logger.info(f"    提取方法: direct (论文方法)")
        
        # 获取分辨率信息
        base_info = self.sdl_config.get_base_model_info()
        
        # 模型简称到完整名称的映射（新默认 + 兼容旧别名）
        model_key_map = {
            'fwea': '2024_FWEA23',
            'fwea23': '2024_FWEA23',
            'sino': '2022_SinoScope1.0',
            'sinoscope': '2022_SinoScope1.0',
            '2024_fwea23': '2024_FWEA23',
            '2024_FWEA23': '2024_FWEA23',
            '2022_sinoscope1.0': '2022_SinoScope1.0',
            '2022_SinoScope1.0': '2022_SinoScope1.0',
            'cses': '2024_CSES_VM1.0',
            'ust': '2022_USTClitho2.0'
        }
        highres_model_key = model_key_map.get(target_model.lower(), target_model)
        highres_info = self.sdl_config.get_highres_model_info(highres_model_key)
        
        # ⭐ v3.5: 只使用直接提取方法（论文方法）
        self.logger.info(f"    📦 使用直接提取方法（论文方法）")
        
        # ⭐ v3.5: 从 model_loader 获取原始模型
        base_ds = model_loader.get_model(self.sdl_config.base_model_key)
        highres_ds = model_loader.get_model(highres_model_key)
        
        if base_ds is None or highres_ds is None:
            raise ValueError(f"原始模型未加载: base={base_ds is not None}, highres={highres_ds is not None}")
        
        # 直接从原始网格提取配对patches
        paired_patches = self.patch_extractor.extract_paired_patches_direct(
            base_ds=base_ds,
            highres_ds=highres_ds,
            param=param,
            depth_km=depth_km,
            base_resolution=base_info.lat_resolution,
            highres_resolution=highres_info.lat_resolution
        )
        
        if paired_patches.n_valid < self.config.n_components:
            self.logger.error(f"    ❌ 有效patches不足: {paired_patches.n_valid} < {self.config.n_components}")
            return TransformResult(
                depth_km=depth_km,
                target_model=target_model,
                param=param,
                paired_patches=paired_patches,
                coupled_dict=None,
                transformed_patches=np.array([]),
                baseline_patches=np.array([]),
                transform_error=np.nan,
                baseline_error=np.nan,
                relative_improvement=np.nan,
                error_distribution={},
                passed=False
            )
        
        # 训练耦合字典
        coupled_dict = self.dict_learner.train(paired_patches)
        
        # 存储结果
        self.trained_dicts[depth_km] = coupled_dict
        
        # 验收检查
        passed = (
            coupled_dict.transform_error < self.config.max_transform_error and
            coupled_dict.relative_improvement > self.config.min_relative_improvement
        )
        
        self.logger.info(f"\n    🎯 验收: {'✅ 通过' if passed else '❌ 未通过'}")
        
        # 计算变换后的patches用于可视化
        valid_mask = paired_patches.valid_mask
        P1_valid = paired_patches.patches_base[valid_mask]
        
        # 使用验证集的结果
        val_idx = coupled_dict.val_indices
        P2_transformed = coupled_dict.C_val @ coupled_dict.D2 + coupled_dict.preprocessing.mean_P2
        
        # Baseline
        L1 = P1_valid.shape[1]
        L2 = paired_patches.patches_highres.shape[1]
        patch_size_base = int(np.sqrt(L1))
        patch_size_highres = int(np.sqrt(L2))
        scale = patch_size_highres / patch_size_base
        
        P1_val = P1_valid[val_idx]
        P2_baseline = np.zeros((len(val_idx), L2))
        for i in range(len(val_idx)):
            base_patch = P1_val[i].reshape(patch_size_base, patch_size_base)
            upsampled = np.asarray(zoom(base_patch, scale, order=3))
            P2_baseline[i] = upsampled.ravel()
        
        return TransformResult(
            depth_km=depth_km,
            target_model=target_model,
            param=param,
            paired_patches=paired_patches,
            coupled_dict=coupled_dict,
            transformed_patches=P2_transformed,
            baseline_patches=P2_baseline,
            transform_error=coupled_dict.transform_error,
            baseline_error=coupled_dict.baseline_error,
            relative_improvement=coupled_dict.relative_improvement,
            error_distribution=coupled_dict.error_distribution,
            passed=passed
        )
    
    def get_best_dictionaries(self) -> Dict[float, CoupledDictionaryResult]:
        """获取所有深度的训练字典"""
        return self.trained_dicts


# ==============================================================================
# 第五部分(续): Phase 1.5 - 完整2D切片变换器
# ==============================================================================

class FullSliceTransformer:
    """
    完整2D切片变换器 (Phase 1.5)
    
    对USTClitho2.0的完整2D切片应用SDL变换，生成增强后的完整速度切片。
    
    对应论文 Figure 3 的实现。
    
    支持两种拼接方法:
    - 'gaussian': 高斯加权平均（更平滑，推荐）
    - 'average': 简单平均（更接近 Dong 2015 原始方法）
    """
    
    def __init__(self, config: Phase1Config, sdl_config: SDLConfigV31, 
                 logger: logging.Logger):
        self.config = config
        self.sdl_config = sdl_config
        self.logger = logger
        
        # 存储训练好的字典（按深度）
        self.trained_dicts: Dict[float, CoupledDictionaryResult] = {}
    
    def set_trained_dict(self, depth_km: float, coupled_dict: CoupledDictionaryResult) -> None:
        """设置指定深度的训练字典"""
        self.trained_dicts[depth_km] = coupled_dict
    
    def get_nearest_dict(self, depth_km: float) -> Optional[CoupledDictionaryResult]:
        """获取最近深度的训练字典"""
        if not self.trained_dicts:
            return None
        
        available_depths = list(self.trained_dicts.keys())
        nearest_depth = min(available_depths, key=lambda d: abs(d - depth_km))
        return self.trained_dicts[nearest_depth]
    
    def _create_gaussian_weight(self, patch_size: int) -> np.ndarray:
        """
        创建高斯权重矩阵
        
        中心权重高，边缘权重低，用于平滑拼接
        基于 Dong et al. 2015 的 weighted averaging 建议
        """
        y, x = np.ogrid[:patch_size, :patch_size]
        center = (patch_size - 1) / 2
        sigma = patch_size / 3  # 标准差
        weight = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
        return weight
    
    def _create_feather_weight(self, patch_size: int) -> np.ndarray:
        """
        创建羽毛权重矩阵（边界渐变，中心为1）
        
        保留patch中心的原始值，仅在边界进行平滑过渡
        用于减少过度平滑，保留细节
        """
        # 创建从0到1的渐变（从边界到中心）
        y, x = np.ogrid[:patch_size, :patch_size]
        center = (patch_size - 1) / 2
        
        # 计算到中心的距离（归一化到 0-1）
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        max_dist = np.sqrt(2 * center**2)
        normalized_dist = dist / max_dist
        
        # 羽毛函数：平滑从0（边界）到1（中心）
        # 使用三次多项式确保平滑过渡
        weight = 1.0 - (1.0 - np.clip(1.0 - normalized_dist, 0, 1))**2
        return weight
    
    def transform_full_slice(self,
                            base_slice: np.ndarray,
                            target_slice: np.ndarray,
                            lat_coords: np.ndarray,
                            lon_coords: np.ndarray,
                            overlap_mask: np.ndarray,
                            coupled_dict: CoupledDictionaryResult,
                            depth_km: float,
                            target_model: str,
                            param: str,
                            stitch_method: str = 'average') -> FullSliceTransformResult:
        """
        对完整2D切片应用SDL变换
        
        Args:
            base_slice: (N_lat, N_lon) 基础模型切片（已插值到高分辨率网格）
            target_slice: (N_lat, N_lon) 高分辨率目标切片 (GT)
            lat_coords: 纬度坐标
            lon_coords: 经度坐标
            overlap_mask: (N_lat, N_lon) 重叠区域掩码
            coupled_dict: 训练好的耦合字典
            depth_km: 深度
            target_model: 目标模型名称
            param: 参数名
            stitch_method: 拼接方法 ('gaussian' or 'average')
            
        Returns:
            FullSliceTransformResult
        """
        self.logger.info(f"\n    🔄 完整切片变换 @ {depth_km}km ({param})")
        self.logger.info(f"      拼接方法: {stitch_method}")
        
        # 获取参数
        physical_window = self.config.physical_window_size  # 6°
        physical_stride = self.config.physical_stride       # 3°
        
        # 获取分辨率
        base_info = self.sdl_config.get_base_model_info()
        highres_info = self.sdl_config.get_highres_model_info(self.sdl_config.default_target_model)
        
        base_resolution = base_info.lat_resolution          # 1°
        highres_resolution = highres_info.lat_resolution    # 0.25°
        
        # 计算patch尺寸（在高分辨率网格上）
        highres_patch_size = int(physical_window / highres_resolution)  # 24
        base_patch_size = int(physical_window / base_resolution)        # 6
        
        # 初始化输出
        n_lat, n_lon = base_slice.shape
        reconstructed = np.zeros((n_lat, n_lon), dtype=np.float64)
        weight_map = np.zeros((n_lat, n_lon), dtype=np.float64)
        copy_mode = (stitch_method == 'copy')
        
        # 创建权重
        if stitch_method == 'gaussian':
            patch_weight = self._create_gaussian_weight(highres_patch_size)
        elif stitch_method == 'feather':
            patch_weight = self._create_feather_weight(highres_patch_size)
        elif copy_mode:
            patch_weight = np.ones((highres_patch_size, highres_patch_size))
        else:  # 'average' 或其他
            patch_weight = np.ones((highres_patch_size, highres_patch_size))
        
        # 获取字典参数 (⭐ v3.4.2: 无 centering，与论文一致)
        D1 = coupled_dict.D1
        D2 = coupled_dict.D2
        dict_learner = coupled_dict.dict_learner
        # 不再使用 mean_P1/mean_P2，因为移除了 centering
        
        # 遍历范围
        half_window = physical_window / 2
        lat_min_valid = lat_coords.min() + half_window
        lat_max_valid = lat_coords.max() - half_window
        lon_min_valid = lon_coords.min() + half_window
        lon_max_valid = lon_coords.max() - half_window
        
        # 生成patch中心坐标
        lat_centers = np.arange(lat_min_valid, lat_max_valid + 0.001, physical_stride)
        lon_centers = np.arange(lon_min_valid, lon_max_valid + 0.001, physical_stride)
        
        n_patches_processed = 0
        n_patches_skipped = 0
        total_patches = len(lat_centers) * len(lon_centers)
        
        self.logger.info(f"      网格: {len(lat_centers)} × {len(lon_centers)} = {total_patches} patches")
        
        # 使用 tqdm 显示进度
        pbar = tqdm(total=total_patches, desc=f"      Transform@{depth_km}km",
                    unit="patch", ncols=100, file=sys.stdout)
        
        # 遍历所有patch位置
        for lat_c in lat_centers:
            for lon_c in lon_centers:
                pbar.update(1)
                # 计算patch边界（物理坐标）
                lat_start_phys = lat_c - half_window
                lat_end_phys = lat_c + half_window
                lon_start_phys = lon_c - half_window
                lon_end_phys = lon_c + half_window
                
                # 转换为网格索引
                lat_idx_start = np.argmin(np.abs(lat_coords - lat_start_phys))
                lat_idx_end = lat_idx_start + highres_patch_size
                lon_idx_start = np.argmin(np.abs(lon_coords - lon_start_phys))
                lon_idx_end = lon_idx_start + highres_patch_size
                
                # 边界检查
                if lat_idx_end > n_lat or lon_idx_end > n_lon:
                    n_patches_skipped += 1
                    continue
                
                # 提取基础模型patch（高分辨率网格上）
                base_patch_highres = base_slice[lat_idx_start:lat_idx_end, 
                                                lon_idx_start:lon_idx_end]
                
                # 检查NaN
                if np.any(np.isnan(base_patch_highres)):
                    n_patches_skipped += 1
                    continue
                
                # 下采样到基础模型分辨率
                scale_factor = base_patch_size / highres_patch_size
                base_patch_lowres = np.asarray(zoom(base_patch_highres, scale_factor, order=1))
                
                # 确保尺寸正确
                if base_patch_lowres.shape != (base_patch_size, base_patch_size):
                    n_patches_skipped += 1
                    continue
                
                # 编码: p₁ → c (⭐ v3.4.2: 无 centering)
                p1 = base_patch_lowres.ravel()
                
                try:
                    c = dict_learner.transform([p1])  # (1, K) 直接编码，不减均值
                except:
                    n_patches_skipped += 1
                    continue
                
                # 解码: c → p̂₂ (⭐ v3.4.2: 无 centering)
                p2 = (c @ D2).ravel()  # 直接重建，不加均值
                
                # 重塑为patch
                transformed_patch = p2.reshape(highres_patch_size, highres_patch_size)
                
                # 放回输出网格（根据拼接方法选择不同的策略）
                if copy_mode:
                    # 直接覆盖：后面的patch覆盖前面的，无加权
                    reconstructed[lat_idx_start:lat_idx_end, 
                                 lon_idx_start:lon_idx_end] = transformed_patch
                else:
                    # 加权拼接（'average', 'gaussian', 'feather'）
                    reconstructed[lat_idx_start:lat_idx_end, 
                                 lon_idx_start:lon_idx_end] += transformed_patch * patch_weight
                    weight_map[lat_idx_start:lat_idx_end,
                              lon_idx_start:lon_idx_end] += patch_weight
                
                n_patches_processed += 1
        
        pbar.close()
        
        # 归一化（仅对加权拼接方法）
        if not copy_mode:
            valid_mask = weight_map > 0
            reconstructed[valid_mask] = reconstructed[valid_mask] / weight_map[valid_mask]
            reconstructed[~valid_mask] = np.nan
        else:
            # 'copy' 方法：直接使用，检查未被覆盖的区域
            valid_mask = np.isfinite(reconstructed)
            if not np.any(valid_mask):
                valid_mask = np.ones_like(reconstructed, dtype=bool)
        
        # 计算覆盖率
        coverage_ratio = float(np.sum(valid_mask) / valid_mask.size)
        
        self.logger.info(f"      处理patches: {n_patches_processed}")
        self.logger.info(f"      跳过patches: {n_patches_skipped}")
        self.logger.info(f"      覆盖率: {coverage_ratio*100:.1f}%")
        
        # 生成Baseline（双三次上采样）
        # ⭐ 修正：base_slice 已经是插值后的数据，这就是 Baseline
        # 因为 USTClitho2.0 已经被插值到高分辨率网格，本身就是上采样结果
        # 真正的 Baseline = 原始低分辨率 → 插值到高分辨率（已完成于 UnifiedModelFramework）
        baseline_slice = base_slice.copy()
        
        # 注意：这里的 base_slice 来自 unified_framework，已经是插值后的结果
        # 如果需要更精确的 Baseline，应该从原始数据重新上采样
        # 但当前实现中 baseline_slice 已经是合理的 Baseline 参照
        
        # 计算误差（仅在有效区域）
        # 有效区域: SDL有输出 且 目标有数据
        eval_mask = valid_mask & ~np.isnan(target_slice)
        
        if np.sum(eval_mask) > 0:
            # SDL误差
            sdl_diff = reconstructed[eval_mask] - target_slice[eval_mask]
            sdl_error = float(np.sqrt(np.mean(sdl_diff**2)) / 
                            np.sqrt(np.mean(target_slice[eval_mask]**2)) * 100)
            
            # Baseline误差
            baseline_diff = baseline_slice[eval_mask] - target_slice[eval_mask]
            baseline_error = float(np.sqrt(np.mean(baseline_diff**2)) / 
                                  np.sqrt(np.mean(target_slice[eval_mask]**2)) * 100)
            
            # 相对改善
            relative_improvement = (baseline_error - sdl_error) / baseline_error * 100
            
            # 误差分布统计
            normalized_error = (reconstructed[eval_mask] - target_slice[eval_mask]) / \
                              np.where(np.abs(target_slice[eval_mask]) > 1e-6, 
                                      target_slice[eval_mask], 1e-6) * 100
            
            error_mean = float(np.mean(normalized_error))
            error_std = float(np.std(normalized_error))
            
            error_distribution = {
                'mean': error_mean,
                'std': error_std,
                'min': float(np.min(normalized_error)),
                'max': float(np.max(normalized_error)),
                'median': float(np.median(normalized_error)),
                'percentile_5': float(np.percentile(normalized_error, 5)),
                'percentile_95': float(np.percentile(normalized_error, 95))
            }
        else:
            sdl_error = np.nan
            baseline_error = np.nan
            relative_improvement = np.nan
            error_mean = np.nan
            error_std = np.nan
            error_distribution = {}
        
        self.logger.info(f"\n      📊 误差统计:")
        self.logger.info(f"        SDL误差: {sdl_error:.2f}%")
        self.logger.info(f"        Baseline误差: {baseline_error:.2f}%")
        self.logger.info(f"        相对改善: {relative_improvement:.1f}%")
        self.logger.info(f"        误差均值: {error_mean:.3f}%")
        self.logger.info(f"        误差标准差: {error_std:.3f}%")
        
        # 验收检查
        passed = (
            not np.isnan(sdl_error) and
            sdl_error < 10.0 and  # 变换误差 < 10%
            relative_improvement > 20.0 and  # 相对改善 > 20%
            abs(error_mean) < 1.0 and  # 误差均值 < 1%
            error_std < 5.0  # 误差标准差 < 5%
        )
        
        self.logger.info(f"      验收: {'✅ 通过' if passed else '❌ 未通过'}")
        
        return FullSliceTransformResult(
            depth_km=depth_km,
            target_model=target_model,
            param=param,
            transformed_slice=reconstructed,
            baseline_slice=baseline_slice,
            target_slice=target_slice,
            original_slice=base_slice,
            weight_map=weight_map,
            lat_coords=lat_coords,
            lon_coords=lon_coords,
            sdl_error=sdl_error,
            baseline_error=baseline_error,
            relative_improvement=relative_improvement,
            error_mean=error_mean,
            error_std=error_std,
            error_distribution=error_distribution,
            n_patches_processed=n_patches_processed,
            n_patches_skipped=n_patches_skipped,
            coverage_ratio=coverage_ratio,
            passed=passed,
            coupled_dict=coupled_dict,
            stitch_method=stitch_method
        )
    
    def transform_full_slice_direct(self,
                                    base_ds: xr.Dataset,
                                    highres_ds: xr.Dataset,
                                    param: str,
                                    depth_km: float,
                                    coupled_dict: CoupledDictionaryResult,
                                    target_model: str,
                                    stitch_method: str = 'average') -> FullSliceTransformResult:
        """
        直接从原始模型数据进行完整切片变换
        
        ⭐ v3.3.5: 修复训练/变换不一致问题
        - 训练时: 从原始 1° 网格直接提取 6×6 patches
        - 变换时: 也从原始 1° 网格提取，保持一致性
        
        Args:
            base_ds: 原始基础模型 xarray Dataset (1° 分辨率)
            highres_ds: 原始高分辨率模型 xarray Dataset (0.25° 分辨率)
            param: 参数名 ('vs' or 'vp')
            depth_km: 深度
            coupled_dict: 训练好的耦合字典
            target_model: 目标模型名称
            stitch_method: 拼接方法 ('gaussian' or 'average')
            
        Returns:
            FullSliceTransformResult
        """
        self.logger.info(f"\n    🔄 直接变换 @{depth_km}km (论文方法)...")
        
        physical_window = self.config.physical_window_size
        physical_stride = self.config.physical_stride
        
        # 获取分辨率
        base_info = self.sdl_config.get_base_model_info()
        highres_info = self.sdl_config.get_highres_model_info(self.sdl_config.default_target_model)
        
        base_resolution = base_info.lat_resolution      # 1°
        highres_resolution = highres_info.lat_resolution  # 0.25°
        
        # 计算 patch 尺寸
        base_patch_size = int(physical_window / base_resolution)        # 6
        highres_patch_size = int(physical_window / highres_resolution)  # 24
        
        self.logger.info(f"      基础 patch: {base_patch_size}×{base_patch_size} ({base_resolution}°)")
        self.logger.info(f"      高分辨率 patch: {highres_patch_size}×{highres_patch_size} ({highres_resolution}°)")
        
        # 获取坐标
        base_lat_name = 'latitude' if 'latitude' in base_ds.coords else 'lat'
        base_lon_name = 'longitude' if 'longitude' in base_ds.coords else 'lon'
        base_lat = base_ds[base_lat_name].values
        base_lon = base_ds[base_lon_name].values
        
        highres_lat_name = 'latitude' if 'latitude' in highres_ds.coords else 'lat'
        highres_lon_name = 'longitude' if 'longitude' in highres_ds.coords else 'lon'
        highres_lat = highres_ds[highres_lat_name].values
        highres_lon = highres_ds[highres_lon_name].values
        
        # 获取深度索引
        base_depth_name = 'depth' if 'depth' in base_ds.coords else 'dep'
        highres_depth_name = 'depth' if 'depth' in highres_ds.coords else 'dep'
        base_depth = base_ds[base_depth_name].values
        highres_depth = highres_ds[highres_depth_name].values
        
        base_depth_idx = int(np.argmin(np.abs(base_depth - depth_km)))
        highres_depth_idx = int(np.argmin(np.abs(highres_depth - depth_km)))
        
        # 获取速度切片 - 需要处理维度顺序
        base_data = base_ds[param].values
        highres_data = highres_ds[param].values
        base_dims = [str(d) for d in base_ds[param].dims]
        highres_dims = [str(d) for d in highres_ds[param].dims]
        
        # 提取深度切片
        base_slice = self._extract_depth_slice_direct(
            base_data, base_dims, base_lat_name, base_lon_name, base_depth_idx
        )
        highres_slice = self._extract_depth_slice_direct(
            highres_data, highres_dims, highres_lat_name, highres_lon_name, highres_depth_idx
        )
        
        self.logger.info(f"      基础模型切片: {base_slice.shape}")
        self.logger.info(f"      高分辨率切片: {highres_slice.shape}")
        
        # ⭐ v3.4.1 核心修复: 以 USTClitho2.0 的全域范围进行 SDL 变换
        # 论文核心: "enhance the low-resolution model both in the overlapping region as well as outside it"
        half_window = physical_window / 2
        
        # === 创建覆盖 USTClitho2.0 全域的高分辨率输出网格 ===
        out_lat = np.arange(base_lat.min(), base_lat.max() + highres_resolution/2, highres_resolution)
        out_lon = np.arange(base_lon.min(), base_lon.max() + highres_resolution/2, highres_resolution)
        n_lat_out = len(out_lat)
        n_lon_out = len(out_lon)
        
        self.logger.info(f"      输出网格 (USTClitho2.0全域): {n_lat_out} × {n_lon_out} ({highres_resolution}°)")
        self.logger.info(f"      输出范围: lat [{out_lat.min():.2f}, {out_lat.max():.2f}], lon [{out_lon.min():.2f}, {out_lon.max():.2f}]")
        
        # patch 中心遍历 USTClitho2.0 全域（除边缘）
        lat_min_transform = base_lat.min() + half_window
        lat_max_transform = base_lat.max() - half_window
        lon_min_transform = base_lon.min() + half_window
        lon_max_transform = base_lon.max() - half_window
        
        # 计算重叠区域（用于评估）
        lat_min_overlap = max(base_lat.min(), highres_lat.min())
        lat_max_overlap = min(base_lat.max(), highres_lat.max())
        lon_min_overlap = max(base_lon.min(), highres_lon.min())
        lon_max_overlap = min(base_lon.max(), highres_lon.max())
        
        # 生成 patch 中心坐标（覆盖 USTClitho2.0 全域）
        lat_centers = np.arange(lat_min_transform, lat_max_transform + 0.001, physical_stride)
        lon_centers = np.arange(lon_min_transform, lon_max_transform + 0.001, physical_stride)
        
        self.logger.info(f"      变换范围: lat [{lat_min_transform:.2f}, {lat_max_transform:.2f}], lon [{lon_min_transform:.2f}, {lon_max_transform:.2f}]")
        self.logger.info(f"      网格: {len(lat_centers)} × {len(lon_centers)} patches")
        
        # 获取字典参数 (⭐ v3.4.2: 无 centering，与论文一致)
        D1 = coupled_dict.D1
        D2 = coupled_dict.D2
        dict_learner = coupled_dict.dict_learner
        # 不再使用 mean_P1/mean_P2，因为移除了 centering
        
        # 创建输出网格
        reconstructed = np.zeros((n_lat_out, n_lon_out), dtype=np.float64)
        weight_map = np.zeros((n_lat_out, n_lon_out), dtype=np.float64)
        
        # 权重
        if stitch_method == 'gaussian':
            patch_weight = self._create_gaussian_weight(highres_patch_size)
        else:
            patch_weight = np.ones((highres_patch_size, highres_patch_size))
        
        n_patches_processed = 0
        n_patches_skipped = 0
        total_patches = len(lat_centers) * len(lon_centers)
        
        # 使用 tqdm 显示进度
        pbar = tqdm(total=total_patches, desc=f"      Direct@{depth_km}km",
                    unit="patch", ncols=100, file=sys.stdout)
        
        for lat_c in lat_centers:
            for lon_c in lon_centers:
                pbar.update(1)
                
                lat_start = lat_c - half_window
                lat_end = lat_c + half_window
                lon_start = lon_c - half_window
                lon_end = lon_c + half_window
                
                # === 从基础模型原始网格直接提取 ===
                base_lat_idx = np.where((base_lat >= lat_start) & (base_lat < lat_end))[0]
                base_lon_idx = np.where((base_lon >= lon_start) & (base_lon < lon_end))[0]
                
                if len(base_lat_idx) < base_patch_size or len(base_lon_idx) < base_patch_size:
                    n_patches_skipped += 1
                    continue
                
                # 取精确尺寸
                base_lat_idx = base_lat_idx[:base_patch_size]
                base_lon_idx = base_lon_idx[:base_patch_size]
                
                base_patch = base_slice[np.ix_(base_lat_idx, base_lon_idx)]
                
                if base_patch.shape != (base_patch_size, base_patch_size):
                    n_patches_skipped += 1
                    continue
                
                if np.any(np.isnan(base_patch)):
                    n_patches_skipped += 1
                    continue
                
                # === 编码: p₁ → c === (⭐ v3.4.2: 无 centering，与论文一致)
                p1 = base_patch.ravel()
                # 直接编码，不减均值（论文方法）
                
                try:
                    c = dict_learner.transform([p1])  # (1, K)
                except:
                    n_patches_skipped += 1
                    continue
                
                # === 解码: c → p̂₂ === (⭐ v3.4.2: 无 centering)
                p2 = (c @ D2).ravel()  # 直接重建，不加均值
                
                # 重塑为 patch
                transformed_patch = p2.reshape(highres_patch_size, highres_patch_size)
                
                # ⭐ 确定输出位置（在 USTClitho2.0 全域输出网格上）
                out_lat_idx = np.where((out_lat >= lat_start) & (out_lat < lat_end))[0]
                out_lon_idx = np.where((out_lon >= lon_start) & (out_lon < lon_end))[0]
                
                if len(out_lat_idx) < highres_patch_size or len(out_lon_idx) < highres_patch_size:
                    n_patches_skipped += 1
                    continue
                
                out_lat_idx = out_lat_idx[:highres_patch_size]
                out_lon_idx = out_lon_idx[:highres_patch_size]
                
                # 放回输出网格
                lat_slice = slice(out_lat_idx[0], out_lat_idx[-1]+1)
                lon_slice = slice(out_lon_idx[0], out_lon_idx[-1]+1)
                
                if (out_lat_idx[-1] - out_lat_idx[0] + 1) != highres_patch_size or \
                   (out_lon_idx[-1] - out_lon_idx[0] + 1) != highres_patch_size:
                    n_patches_skipped += 1
                    continue
                
                reconstructed[lat_slice, lon_slice] += transformed_patch * patch_weight
                weight_map[lat_slice, lon_slice] += patch_weight
                
                n_patches_processed += 1
        
        pbar.close()
        
        # 归一化
        valid_mask = weight_map > 0
        reconstructed[valid_mask] = reconstructed[valid_mask] / weight_map[valid_mask]
        reconstructed[~valid_mask] = np.nan
        
        coverage_ratio = float(np.sum(valid_mask) / valid_mask.size)
        
        self.logger.info(f"      处理 patches: {n_patches_processed}")
        self.logger.info(f"      跳过 patches: {n_patches_skipped}")
        self.logger.info(f"      覆盖率: {coverage_ratio*100:.1f}%")
        
        # ⭐ v3.4.1 修复: Baseline 在 USTClitho2.0 全域输出网格上生成
        from scipy.interpolate import RegularGridInterpolator
        
        # 创建基于坐标的插值器（双三次插值）
        base_lat_sorted = np.sort(base_lat)
        base_lon_sorted = np.sort(base_lon)
        
        # 如果原始数据坐标是递减的，需要翻转数据
        if base_lat[0] > base_lat[-1]:
            base_slice_for_interp = base_slice[::-1, :]
        else:
            base_slice_for_interp = base_slice
        
        if base_lon[0] > base_lon[-1]:
            base_slice_for_interp = base_slice_for_interp[:, ::-1]
        
        interp_base = RegularGridInterpolator(
            (base_lat_sorted, base_lon_sorted), 
            base_slice_for_interp,
            method='cubic',
            bounds_error=False, 
            fill_value=np.nan
        )
        
        # Baseline: USTClitho2.0 插值到输出网格 (out_lat × out_lon)
        out_lat_grid, out_lon_grid = np.meshgrid(out_lat, out_lon, indexing='ij')
        out_points = np.column_stack([out_lat_grid.ravel(), out_lon_grid.ravel()])
        baseline_slice = interp_base(out_points).reshape(n_lat_out, n_lon_out)
        
        self.logger.info(f"      Baseline生成: USTClitho2.0 → 输出网格 ({n_lat_out}×{n_lon_out})")
        
        # ⭐ 误差评估: 在高分辨率模型的有效数据范围内进行
        # 由于输出网格分辨率 = 高分辨率模型分辨率 (0.25°)，可以直接映射
        target_on_out = np.full((n_lat_out, n_lon_out), np.nan)
        
        # 向量化: 计算高分辨率网格点在输出网格中的索引
        lat_offset = (highres_lat - out_lat[0]) / highres_resolution
        lon_offset = (highres_lon - out_lon[0]) / highres_resolution
        
        # 四舍五入到最近的网格点
        lat_idx = np.round(lat_offset).astype(int)
        lon_idx = np.round(lon_offset).astype(int)
        
        # 创建有效点掩膜 (在输出网格范围内)
        lat_valid = (lat_idx >= 0) & (lat_idx < n_lat_out)
        lon_valid = (lon_idx >= 0) & (lon_idx < n_lon_out)
        
        # 使用 meshgrid 创建索引网格
        hr_lat_idx, hr_lon_idx = np.meshgrid(np.arange(len(highres_lat)), 
                                              np.arange(len(highres_lon)), indexing='ij')
        out_lat_idx, out_lon_idx = np.meshgrid(lat_idx, lon_idx, indexing='ij')
        lat_valid_grid, lon_valid_grid = np.meshgrid(lat_valid, lon_valid, indexing='ij')
        
        # 有效数据掩膜 (在范围内 & 非 NaN)
        valid_data = lat_valid_grid & lon_valid_grid & ~np.isnan(highres_slice)
        
        # 向量化赋值
        valid_hr_lat = hr_lat_idx[valid_data]
        valid_hr_lon = hr_lon_idx[valid_data]
        valid_out_lat = out_lat_idx[valid_data]
        valid_out_lon = out_lon_idx[valid_data]
        
        target_on_out[valid_out_lat, valid_out_lon] = highres_slice[valid_hr_lat, valid_hr_lon]
        
        # 评估掩膜：有效变换区域 & 有高分辨率数据
        eval_mask = valid_mask & ~np.isnan(target_on_out) & ~np.isnan(baseline_slice)
        
        self.logger.info(f"      评估区域: {np.sum(eval_mask)} 点 (重叠区域)")
        
        if np.sum(eval_mask) > 0:
            sdl_diff = reconstructed[eval_mask] - target_on_out[eval_mask]
            sdl_error = float(np.sqrt(np.mean(sdl_diff**2)) / 
                            np.sqrt(np.mean(target_on_out[eval_mask]**2)) * 100)
            
            baseline_diff = baseline_slice[eval_mask] - target_on_out[eval_mask]
            baseline_error = float(np.sqrt(np.mean(baseline_diff**2)) / 
                                  np.sqrt(np.mean(target_on_out[eval_mask]**2)) * 100)
            
            relative_improvement = (baseline_error - sdl_error) / baseline_error * 100
            
            normalized_error = (reconstructed[eval_mask] - target_on_out[eval_mask]) / \
                              np.where(np.abs(target_on_out[eval_mask]) > 1e-6, 
                                      target_on_out[eval_mask], 1e-6) * 100
            
            error_mean = float(np.mean(normalized_error))
            error_std = float(np.std(normalized_error))
            
            error_distribution = {
                'mean': error_mean,
                'std': error_std,
                'min': float(np.min(normalized_error)),
                'max': float(np.max(normalized_error)),
                'median': float(np.median(normalized_error))
            }
        else:
            sdl_error = baseline_error = relative_improvement = error_mean = error_std = np.nan
            error_distribution = {}
        
        self.logger.info(f"\n      📊 误差统计 (重叠区域):")
        self.logger.info(f"        SDL误差: {sdl_error:.2f}%")
        self.logger.info(f"        Baseline误差: {baseline_error:.2f}%")
        self.logger.info(f"        相对改善: {relative_improvement:.1f}%")
        
        passed = (
            not np.isnan(sdl_error) and
            sdl_error < 10.0 and
            relative_improvement > 20.0
        )
        
        self.logger.info(f"      验收: {'✅ 通过' if passed else '❌ 未通过'}")
        
        return FullSliceTransformResult(
            depth_km=depth_km,
            target_model=target_model,
            param=param,
            transformed_slice=reconstructed,
            baseline_slice=baseline_slice,
            target_slice=target_on_out,  # 高分辨率目标（在输出网格上）
            original_slice=baseline_slice,  # Baseline (USTClitho2.0 上采样)
            weight_map=weight_map,
            lat_coords=out_lat,  # ⭐ 使用输出网格坐标（USTClitho2.0 全域）
            lon_coords=out_lon,
            sdl_error=sdl_error,
            baseline_error=baseline_error,
            relative_improvement=relative_improvement,
            error_mean=error_mean,
            error_std=error_std,
            error_distribution=error_distribution,
            n_patches_processed=n_patches_processed,
            n_patches_skipped=n_patches_skipped,
            coverage_ratio=coverage_ratio,
            passed=passed,
            coupled_dict=coupled_dict,
            stitch_method=stitch_method
        )
    
    def _extract_depth_slice_direct(self, data: np.ndarray, dims: Sequence[str],
                                    lat_name: str, lon_name: str,
                                    depth_idx: int) -> np.ndarray:
        """根据维度顺序提取深度切片"""
        # 找到各维度索引
        depth_dim_idx = None
        lat_dim_idx = None
        lon_dim_idx = None
        
        for i, dim in enumerate(dims):
            if 'dep' in dim.lower():
                depth_dim_idx = i
            elif 'lat' in dim.lower():
                lat_dim_idx = i
            elif 'lon' in dim.lower():
                lon_dim_idx = i

        if depth_dim_idx is None:
            raise ValueError(f"无法找到深度维度: {list(dims)}")
        if lat_dim_idx is None or lon_dim_idx is None:
            raise ValueError(f"无法找到经纬度维度: {list(dims)}")
        
        # 构建索引
        slices: List[Any] = [slice(None)] * len(dims)
        slices[depth_dim_idx] = depth_idx
        
        slice_2d = data[tuple(slices)]
        
        # 确保 (lat, lon) 顺序
        if lat_dim_idx > lon_dim_idx:
            slice_2d = slice_2d.T
        
        return slice_2d
    
    def transform_all_depths(self,
                            model_loader: 'ModelLoader',
                            param: str,
                            depths: List[float],
                            target_model: str = 'fwea',
                            stitch_method: str = 'gaussian') -> Dict[float, FullSliceTransformResult]:
        """
        ⭐ v3.5: 对所有深度进行完整切片变换（使用原始模型数据）
        
        Args:
            model_loader: 模型加载器
            param: 参数名 ('vs' or 'vp')
            depths: 要处理的深度列表
            target_model: 目标模型简称
            stitch_method: 拼接方法
            
        Returns:
            {depth: FullSliceTransformResult}
        """
        self.logger.info(f"\n  🔄 对所有深度进行完整切片变换...")
        self.logger.info(f"    深度列表: {depths}")
        self.logger.info(f"    ⭐ v3.5: 使用 transform_full_slice_direct() 直接从原始模型变换")
        
        # 模型简称到完整名称的映射（新默认 + 兼容旧别名）
        model_key_map = {
            'fwea': '2024_FWEA23',
            'fwea23': '2024_FWEA23',
            'sino': '2022_SinoScope1.0',
            'sinoscope': '2022_SinoScope1.0',
            '2024_fwea23': '2024_FWEA23',
            '2024_FWEA23': '2024_FWEA23',
            '2022_sinoscope1.0': '2022_SinoScope1.0',
            '2022_SinoScope1.0': '2022_SinoScope1.0',
            'cses': '2024_CSES_VM1.0',
            'ust': '2022_USTClitho2.0'
        }
        base_model_key = self.sdl_config.base_model_key
        highres_model_key = model_key_map.get(target_model.lower(), target_model)
        
        # 获取原始模型
        base_ds = model_loader.get_model(base_model_key)
        highres_ds = model_loader.get_model(highres_model_key)
        
        if base_ds is None or highres_ds is None:
            raise ValueError(f"模型未加载: base={base_ds is not None}, highres={highres_ds is not None}")
        
        results = {}
        
        for depth_km in depths:
            # 获取最近的训练字典
            coupled_dict = self.get_nearest_dict(depth_km)
            
            if coupled_dict is None:
                self.logger.warning(f"    ⚠️ 深度 {depth_km}km 没有可用的字典，跳过")
                continue
            
            try:
                # ⭐ v3.5: 使用 transform_full_slice_direct() 直接从原始模型变换
                result = self.transform_full_slice_direct(
                    base_ds=base_ds,
                    highres_ds=highres_ds,
                    param=param,
                    depth_km=depth_km,
                    coupled_dict=coupled_dict,
                    target_model=target_model,
                    stitch_method=stitch_method
                )
                
                results[depth_km] = result
                
            except Exception as e:
                self.logger.error(f"    ❌ 深度 {depth_km}km 变换失败: {e}")
                import traceback
                traceback.print_exc()
        
        return results


# ==============================================================================
# 第六部分: 可视化器
# ==============================================================================

class SDLVisualizerV31:
    """
    SDL融合可视化器 v3.3（保持类名兼容性）
    
    按组织结构生成所有可视化图表:
    - U组 (U01-U05): Phase -1 统一框架
    - V组 (V01-V08): Phase 0 字典验证
    - T组 (T01-T11): Phase 1 2D变换
    - F组 (F01-F06): Phase 1.5 完整切片变换
    - H组 (H01-H05): Phase H 超参数搜索 (v3.3 新增)
    - 3D组 (3D01-3D05): Phase 2 3D融合
    - G组 (G01-G05): Phase 3 全域增强
    - S组 (S01-S06): 综合评估
    """
    
    def __init__(self, config: VisualizationConfig, output_dir: Path, 
                 logger: logging.Logger):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
        # 设置matplotlib
        plt.rcParams['font.size'] = self.config.font_size
        plt.rcParams['axes.titlesize'] = self.config.title_size
    
    def save_figure(self, fig, name: str) -> None:
        """保存图表"""
        for fmt in self.config.formats:
            filepath = self.output_dir / f"{self.config.figure_prefix}{name}.{fmt}"
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        plt.close(fig)
    
    def _add_coastlines(self, ax, draw_labels: bool = True) -> None:
        """
        为地理坐标轴添加海岸线和边界 (v3.3.4 新增)
        
        Args:
            ax: matplotlib axes 对象（需要是 GeoAxes）
            draw_labels: 是否绘制经纬度标签
        """
        try:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='k')
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', edgecolor='gray')
            if draw_labels:
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, 
                                  linestyle='--', color='gray')
                gl.top_labels = False
                gl.right_labels = False
        except Exception as e:
            self.logger.debug(f"无法添加海岸线: {e}")
    
    def _create_geo_axes(self, fig, nrows: int = 1, ncols: int = 1,
                         index: int = 1, figsize: Optional[Tuple[int, int]] = None) -> Any:
        """
        创建带有地理投影的子图 (v3.3.4 新增)
        
        Args:
            fig: matplotlib figure 对象
            nrows: 行数
            ncols: 列数
            index: 子图索引 (1-based)
            figsize: 可选的图像尺寸
            
        Returns:
            带有 PlateCarree 投影的 axes
        """
        ax = fig.add_subplot(nrows, ncols, index, projection=ccrs.PlateCarree())
        return ax
    
    # =========================================================================
    # U组: Phase -1 模型概览可视化 ⭐ v3.5 重构
    # =========================================================================
    
    def plot_model_overview(self, model_loader: 'ModelLoader', 
                            param: str = 'vs',
                            depth_km: float = 100) -> None:
        """
        ⭐ v3.5: 直接使用原始模型数据可视化（替代 plot_unified_framework）
        
        U01-U05: 模型概览可视化
        """
        self.logger.info("\n  📊 生成U组可视化 (模型概览)...")
        self.logger.info("    ⭐ v3.5: 直接使用原始模型数据")
        
        # U01: 三模型范围对比
        self._plot_U01_model_coverage_direct(model_loader, param, depth_km)
        
        # U02: 深度覆盖热图
        self._plot_U02_depth_coverage_direct(model_loader, param)
        
        # U04: 三模型速度切片预览
        self._plot_U04_velocity_preview_direct(model_loader, param, depth_km)
        
        # U05: 模型边界和重叠区域
        self._plot_U05_model_bounds(model_loader, param, depth_km)
    
    def _plot_U01_model_coverage_direct(self, model_loader: 'ModelLoader', 
                                        param: str, depth_km: float) -> None:
        """
        ⭐ v3.5: U01 直接从原始模型绘制有效数据范围
        """
        fig = plt.figure(figsize=(12, 5))
        
        models_info = [
            ('2022_SinoScope1.0', 'SinoScope1.0'),
            ('2024_FWEA23', 'FWEA23')
        ]
        
        for idx, (model_key, title) in enumerate(models_info):
            ax = self._create_geo_axes(fig, 1, len(models_info), idx + 1)
            
            try:
                slice_data, lat, lon = model_loader.get_slice_from_model(model_key, param, depth_km)
                
                # 创建有效数据掩码
                valid_mask = (~np.isnan(slice_data)).astype(float)
                valid_pct = np.sum(valid_mask) / valid_mask.size * 100
                
                im = ax.pcolormesh(lon, lat, valid_mask, cmap='Greens', vmin=0, vmax=1,
                                  transform=ccrs.PlateCarree())
                ax.set_title(f'{title}\n{valid_pct:.1f}% valid @ {depth_km}km', fontsize=11)
                ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
                self._add_coastlines(ax)
                
            except Exception as e:
                ax.set_title(f'{title}\nN/A', fontsize=11)
                self.logger.warning(f"      ⚠️ {model_key} 加载失败: {e}")
        
        plt.tight_layout()
        self.save_figure(fig, 'U01_model_coverage')
    
    def _plot_U02_depth_coverage_direct(self, model_loader: 'ModelLoader', 
                                        param: str) -> None:
        """
        ⭐ v3.5: U02 直接从原始模型计算各深度覆盖率
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models_info = [
            ('2022_SinoScope1.0', 'SinoScope1.0', '#3498DB'),
            ('2024_FWEA23', 'FWEA23', '#E74C3C')
        ]
        
        for model_key, title, color in models_info:
            try:
                depths = model_loader.get_model_depths(model_key)
                coverage = []
                
                for depth_km in depths:
                    slice_data, _, _ = model_loader.get_slice_from_model(model_key, param, depth_km)
                    valid_mask = ~np.isnan(slice_data)
                    coverage.append(np.sum(valid_mask) / valid_mask.size * 100)
                
                ax.plot(depths, coverage, '-o', label=title, color=color, 
                       markersize=4, linewidth=2)
                       
            except Exception as e:
                self.logger.warning(f"      ⚠️ {model_key} 覆盖率计算失败: {e}")
        
        ax.set_xlabel('Depth (km)')
        ax.set_ylabel('Valid Data Coverage (%)')
        ax.set_title('Data Coverage vs Depth (Original Resolution)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        self.save_figure(fig, 'U02_depth_coverage')
    
    def _plot_U04_velocity_preview_direct(self, model_loader: 'ModelLoader',
                                          param: str, depth_km: float) -> None:
        """
        ⭐ v3.5: U04 直接从原始模型绘制速度切片预览
        """
        fig = plt.figure(figsize=(12, 5))
        
        models_info = [
            ('2022_SinoScope1.0', 'SinoScope1.0'),
            ('2024_FWEA23', 'FWEA23')
        ]
        
        # 先获取全局速度范围
        vmin, vmax = np.inf, -np.inf
        slices_data = {}
        
        for model_key, title in models_info:
            try:
                slice_data, lat, lon = model_loader.get_slice_from_model(model_key, param, depth_km)
                slices_data[model_key] = (slice_data, lat, lon, title)
                vmin = min(vmin, np.nanmin(slice_data))
                vmax = max(vmax, np.nanmax(slice_data))
            except Exception as e:
                self.logger.warning(f"      ⚠️ {model_key} 加载失败: {e}")
        
        for idx, (model_key, (slice_data, lat, lon, title)) in enumerate(slices_data.items()):
            ax = self._create_geo_axes(fig, 1, len(slices_data), idx + 1)
            
            im = ax.pcolormesh(lon, lat, slice_data, cmap=self.config.velocity_cmap,
                              vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            
            # 获取分辨率
            lat_res, lon_res, _ = model_loader.get_model_resolution(model_key)
            ax.set_title(f'{title} {param.upper()} @ {depth_km}km\n'
                        f'(Δ={lat_res:.2f}° × {lon_res:.2f}°)', fontsize=11)
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
            self._add_coastlines(ax)
            plt.colorbar(im, ax=ax, label=f'{param.upper()} (km/s)', shrink=0.8)
        
        plt.tight_layout()
        self.save_figure(fig, f'U04_velocity_preview_{param}')
    
    def _plot_U05_model_bounds(self, model_loader: 'ModelLoader',
                               param: str, depth_km: float) -> None:
        """
        ⭐ v3.5: U05 显示模型边界和重叠区域
        """
        fig = plt.figure(figsize=(12, 8))
        ax = self._create_geo_axes(fig, 1, 1, 1)
        
        models_info = [
            ('2022_SinoScope1.0', 'SinoScope1.0', '#3498DB', '-'),
            ('2024_FWEA23', 'FWEA23', '#E74C3C', '--')
        ]
        
        # 绘制每个模型的边界
        for model_key, title, color, linestyle in models_info:
            try:
                bounds = model_loader.get_model_bounds(model_key)
                lat_min, lat_max = bounds['lat']
                lon_min, lon_max = bounds['lon']
                
                # 绘制边界矩形
                rect_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
                rect_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
                ax.plot(rect_lons, rect_lats, linestyle=linestyle, color=color, 
                       linewidth=2.5, label=title, transform=ccrs.PlateCarree())
                
            except Exception as e:
                self.logger.warning(f"      ⚠️ {model_key} 边界获取失败: {e}")
        
        # 设置显示范围（所有模型的并集 + 边距）
        ax.set_extent([65, 142, 14, 58], crs=ccrs.PlateCarree())
        self._add_coastlines(ax, draw_labels=True)
        
        ax.set_title(f'Model Coverage Bounds @ {depth_km}km', fontsize=14)
        ax.legend(loc='lower left', fontsize=10)
        
        plt.tight_layout()
        self.save_figure(fig, 'U05_model_bounds')
    
    # =========================================================================
    # V组: Phase 0 字典验证可视化
    # =========================================================================
    
    def plot_validation_results(self, results: Dict[str, Dict[float, ValidationResult]],
                                param: str) -> None:
        """
        V01-V08: 字典验证可视化
        """
        self.logger.info("\n  📊 生成V组可视化 (字典验证)...")
        
        # V01: 原始模型切片
        self._plot_V01_original_slices(results, param)
        
        # V02: 完整2D切片重建对比 (论文Figure 2风格) ⭐ 核心可视化
        self._plot_V02_full_slice_reconstruction(results, param)
        
        # V03: Patch级别重建对比 (辅助)
        self._plot_V03_patch_reconstruction(results, param)
        
        # V05: 字典原子展示
        self._plot_V05_dictionary_atoms(results, param)
        
        # V07: 多深度验证汇总
        self._plot_V07_depth_summary(results, param)
    
    def _plot_V01_original_slices(self, results: Dict, param: str) -> None:
        """
        V01: 各模型原始速度切片
        
        ⭐ v3.4.2: 添加海岸线
        """
        # 找一个通用深度
        depth_km = None
        for model_results in results.values():
            for d in model_results.keys():
                depth_km = d
                break
            if depth_km:
                break
        
        if depth_km is None:
            return
        
        n_models = len(results)
        fig = plt.figure(figsize=(5*n_models, 5))
        
        for idx, (model_name, model_results) in enumerate(results.items(), 1):
            if depth_km in model_results:
                result = model_results[depth_km]
                
                # ⭐ v3.4.2: 使用 cartopy 地理投影 + 海岸线
                ax = self._create_geo_axes(fig, 1, n_models, idx)
                
                lon = result.lon_coords
                lat = result.lat_coords
                
                im = ax.pcolormesh(
                    lon, lat,
                    result.velocity_slice,
                    cmap=self.config.velocity_cmap,
                    transform=ccrs.PlateCarree()
                )
                ax.set_title(f'{model_name}\n{param} @ {depth_km}km')
                ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
                self._add_coastlines(ax)
                plt.colorbar(im, ax=ax, label=f'{param} (km/s)', shrink=0.8)
        
        plt.tight_layout()
        self.save_figure(fig, f'V01_original_slices_{param}')
    
    def _plot_V02_full_slice_reconstruction(self, results: Dict, param: str) -> None:
        """
        V02: 完整2D切片的字典重建对比 (论文Figure 2风格)
        
        ⭐ v3.4.2: 添加海岸线
        
        对每个模型、每个深度生成:
        (a) 原始速度切片
        (b) 字典重建结果
        (c) 归一化误差 = (重建-原始)/原始 × 100%
        """
        for model_name, model_results in results.items():
            for depth_km, result in model_results.items():
                if result.n_patches < 10 or result.dictionary.size == 0:
                    continue
                
                # 重建完整切片
                reconstructed_slice = self._reconstruct_full_slice(result)
                
                if reconstructed_slice is None:
                    continue
                
                lon = result.lon_coords
                lat = result.lat_coords
                original = result.velocity_slice
                
                # 计算共同的速度范围
                vmin = np.nanmin(original)
                vmax = np.nanmax(original)
                
                # ⭐ v3.4.2: 使用 cartopy 地理投影 + 海岸线
                fig = plt.figure(figsize=(16, 5))
                
                # (a) 原始模型
                ax1 = self._create_geo_axes(fig, 1, 3, 1)
                im1 = ax1.pcolormesh(lon, lat, original, 
                                     cmap=self.config.velocity_cmap,
                                     vmin=vmin, vmax=vmax,
                                     transform=ccrs.PlateCarree())
                ax1.set_title(f'(a) {model_name}', fontsize=12)
                ax1.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
                self._add_coastlines(ax1)
                plt.colorbar(im1, ax=ax1, label=f'{param.upper()} (km/s)', shrink=0.8)
                
                # (b) 重建结果
                ax2 = self._create_geo_axes(fig, 1, 3, 2)
                im2 = ax2.pcolormesh(lon, lat, reconstructed_slice,
                                     cmap=self.config.velocity_cmap,
                                     vmin=vmin, vmax=vmax,
                                     transform=ccrs.PlateCarree())
                ax2.set_title(f'(b) Reconstruction', fontsize=12)
                ax2.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
                self._add_coastlines(ax2)
                plt.colorbar(im2, ax=ax2, label=f'{param.upper()} (km/s)', shrink=0.8)
                
                # (c) 归一化误差
                ax3 = self._create_geo_axes(fig, 1, 3, 3)
                # 避免除零
                original_safe = np.where(np.abs(original) > 1e-6, original, np.nan)
                normalized_error = (reconstructed_slice - original) / original_safe * 100
                
                # ⭐ v3.4.2: 误差范围固定为 ±10%
                im3 = ax3.pcolormesh(lon, lat, normalized_error,
                                     cmap=self.config.error_cmap,
                                     vmin=-10, vmax=10,
                                     transform=ccrs.PlateCarree())
                ax3.set_title(f'(c) Normalized Error', fontsize=12)
                ax3.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
                self._add_coastlines(ax3)
                plt.colorbar(im3, ax=ax3, label='Error (%)', shrink=0.8)
                
                # 计算误差统计
                valid_errors = normalized_error[~np.isnan(normalized_error)]
                mean_err = np.mean(np.abs(valid_errors))
                max_err = np.max(np.abs(valid_errors))
                
                fig.suptitle(f'{model_name} @ {depth_km}km - Dictionary Reconstruction\n'
                            f'Mean Error: {mean_err:.2f}%, Max Error: {max_err:.2f}%',
                            fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                self.save_figure(fig, f'V02_full_reconstruction_{model_name}_{int(depth_km)}km_{param}')
    
    def _reconstruct_full_slice(self, result: ValidationResult) -> Optional[np.ndarray]:
        """
        将字典重建的patches拼接回完整的2D切片
        
        使用加权平均处理重叠区域
        """
        if result.n_patches == 0 or result.dictionary.size == 0:
            return None
        
        # 获取原始切片形状和坐标
        n_lat, n_lon = result.velocity_slice.shape
        lat_coords = result.lat_coords
        lon_coords = result.lon_coords
        
        # 计算patch尺寸
        patch_size = result.patch_size
        
        # 初始化输出和权重
        reconstructed = np.zeros((n_lat, n_lon))
        weight_map = np.zeros((n_lat, n_lon))
        
        # 创建高斯权重（中心权重高，边缘权重低）
        y, x = np.ogrid[:patch_size, :patch_size]
        center = (patch_size - 1) / 2
        gaussian_weight = np.exp(-((x - center)**2 + (y - center)**2) / (2 * (patch_size/3)**2))
        
        # 将每个重建的patch放回原位置
        for i in range(result.n_patches):
            # 获取patch中心坐标
            lat_c, lon_c = result.coords[i]
            
            # 找到最近的网格索引
            lat_idx = np.argmin(np.abs(lat_coords - lat_c))
            lon_idx = np.argmin(np.abs(lon_coords - lon_c))
            
            # 计算patch边界
            half = patch_size // 2
            lat_start = lat_idx - half
            lat_end = lat_idx + half + (patch_size % 2)
            lon_start = lon_idx - half
            lon_end = lon_idx + half + (patch_size % 2)
            
            # 边界检查
            if lat_start < 0 or lat_end > n_lat or lon_start < 0 or lon_end > n_lon:
                continue
            
            # 获取重建的patch
            recon_patch = result.patches_reconstructed[i].reshape(patch_size, patch_size)
            
            # 加权累加
            reconstructed[lat_start:lat_end, lon_start:lon_end] += recon_patch * gaussian_weight
            weight_map[lat_start:lat_end, lon_start:lon_end] += gaussian_weight
        
        # 归一化
        reconstructed = np.divide(reconstructed, weight_map, 
                                 where=weight_map > 0,
                                 out=np.full_like(reconstructed, np.nan))
        
        return reconstructed
    
    def _plot_V03_patch_reconstruction(self, results: Dict, param: str) -> None:
        """V03: Patch级别的字典重建对比"""
        # 找一个通用深度
        depth_km = None
        for model_results in results.values():
            for d in model_results.keys():
                depth_km = d
                break
            if depth_km:
                break
        
        if depth_km is None:
            return
        
        n_models = len(results)
        fig, axes = plt.subplots(n_models, 3, figsize=(15, 5*n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for row, (model_name, model_results) in enumerate(results.items()):
            if depth_km not in model_results:
                continue
            
            result = model_results[depth_km]
            
            if result.n_patches < 5:
                continue
            
            # 选择5个patch展示
            n_show = min(5, result.n_patches)
            indices = np.linspace(0, result.n_patches-1, n_show).astype(int)
            
            patch_size = result.dictionary.shape[1]
            side = int(np.sqrt(patch_size))
            
            # 原始
            orig_mosaic = np.hstack([result.patches_original[i].reshape(side, side) 
                                    for i in indices])
            # 重建
            recon_mosaic = np.hstack([result.patches_reconstructed[i].reshape(side, side)
                                     for i in indices])
            # 误差
            error_mosaic = np.hstack([(result.patches_reconstructed[i] - result.patches_original[i]).reshape(side, side)
                                     for i in indices])
            
            vmin = min(orig_mosaic.min(), recon_mosaic.min())
            vmax = max(orig_mosaic.max(), recon_mosaic.max())
            
            axes[row, 0].imshow(orig_mosaic, cmap=self.config.velocity_cmap, vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f'{model_name} - Original')
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(recon_mosaic, cmap=self.config.velocity_cmap, vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f'{model_name} - Reconstructed')
            axes[row, 1].axis('off')
            
            im = axes[row, 2].imshow(error_mosaic, cmap=self.config.error_cmap, 
                                     vmin=-0.1, vmax=0.1)
            axes[row, 2].set_title(f'{model_name} - Error')
            axes[row, 2].axis('off')
            plt.colorbar(im, ax=axes[row, 2], shrink=0.5)
        
        plt.tight_layout()
        self.save_figure(fig, f'V03_patch_reconstruction_{param}')
    
    def _plot_V05_dictionary_atoms(self, results: Dict, param: str) -> None:
        """V05: 字典原子展示"""
        for model_name, model_results in results.items():
            for depth_km, result in model_results.items():
                if result.dictionary.size == 0:
                    continue
                
                K, L = result.dictionary.shape
                side = int(np.sqrt(L))
                
                n_cols = 5
                n_rows = (K + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5*n_rows))
                axes = np.atleast_2d(axes)
                
                for k in range(K):
                    row = k // n_cols
                    col = k % n_cols
                    
                    atom = result.dictionary[k].reshape(side, side)
                    im = axes[row, col].imshow(atom, cmap=self.config.atom_cmap)
                    axes[row, col].set_title(f'Atom {k+1}', fontsize=9)
                    axes[row, col].axis('off')
                
                # 隐藏多余的子图
                for k in range(K, n_rows * n_cols):
                    row = k // n_cols
                    col = k % n_cols
                    axes[row, col].axis('off')
                
                fig.suptitle(f'{model_name} Dictionary @ {depth_km}km ({param})', fontsize=12)
                plt.tight_layout()
                self.save_figure(fig, f'V05_atoms_{model_name}_{int(depth_km)}km_{param}')
                break  # 只绘制第一个深度
    
    def _plot_V07_depth_summary(self, results: Dict, param: str) -> None:
        """V07: 多深度验证汇总"""
        fig, axes = plt.subplots(figsize=(10, 6))
        ax = axes  # 单个axes
        
        colors = ['#3498DB', '#2ECC71', '#E74C3C']
        
        for (model_name, model_results), color in zip(results.items(), colors):
            depths = []
            errors = []
            
            for depth_km, result in sorted(model_results.items()):
                if 'global_error' in result.error_stats and not np.isnan(result.error_stats['global_error']):
                    depths.append(depth_km)
                    errors.append(result.error_stats['global_error'])
            
            if depths:
                ax.plot(depths, errors, '-o', label=model_name, color=color, 
                       markersize=8, linewidth=2)
        
        ax.set_xlabel('Depth (km)')
        ax.set_ylabel('Reconstruction Error (%)')
        ax.set_title(f'Dictionary Reconstruction Error vs Depth ({param})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加验收标准线
        ax.axhline(y=3.0, color='r', linestyle='--', alpha=0.5, label='Max Error (3%)')
        
        plt.tight_layout()
        self.save_figure(fig, f'V07_depth_summary_{param}')
    
    # =========================================================================
    # T组: Phase 1 2D变换可视化
    # =========================================================================
    
    def plot_2d_transform_results(self, results: Dict[float, TransformResult],
                                  param: str, target_model: str) -> None:
        """
        T01-T11: 2D变换可视化
        """
        self.logger.info("\n  📊 生成T组可视化 (2D变换)...")
        
        # T02: 配对Patch样例
        self._plot_T02_paired_patches(results, param)
        
        # T03: CC分布分析
        self._plot_T03_cc_distribution(results, param)
        
        # T04-T05: 字典原子
        self._plot_T04_T05_dictionary_atoms(results, param)
        
        # T08: 验证集重建对比
        self._plot_T08_validation_comparison(results, param)
        
        # T10: 多深度汇总
        self._plot_T10_depth_summary(results, param, target_model)
    
    def _plot_T02_paired_patches(self, results: Dict[float, TransformResult], param: str) -> None:
        """T02: 配对Patch样例"""
        # 选择第一个有结果的深度
        for depth_km, result in results.items():
            if result.paired_patches.n_valid < 4:
                continue
            
            pp = result.paired_patches
            n_show = min(4, pp.n_valid)
            
            valid_indices = np.where(pp.valid_mask)[0][:n_show]
            
            fig, axes = plt.subplots(4, n_show, figsize=(3*n_show, 12))
            
            base_size = pp.base_patch_size
            highres_size = pp.highres_patch_size
            scale = highres_size / base_size
            
            for col, idx in enumerate(valid_indices):
                # Low-res
                base_patch = pp.patches_base[idx].reshape(base_size, base_size)
                axes[0, col].imshow(base_patch, cmap=self.config.velocity_cmap)
                axes[0, col].set_title(f'Low-res ({base_size}x{base_size})', fontsize=9)
                axes[0, col].axis('off')
                
                # Upsampled
                upsampled = zoom(base_patch, scale, order=3)
                axes[1, col].imshow(upsampled, cmap=self.config.velocity_cmap)
                axes[1, col].set_title(f'Upsampled ({highres_size}x{highres_size})', fontsize=9)
                axes[1, col].axis('off')
                
                # High-res GT
                highres_patch = pp.patches_highres[idx].reshape(highres_size, highres_size)
                axes[2, col].imshow(highres_patch, cmap=self.config.velocity_cmap)
                axes[2, col].set_title(f'High-res GT', fontsize=9)
                axes[2, col].axis('off')
                
                # Difference
                diff = highres_patch - upsampled
                im = axes[3, col].imshow(diff, cmap=self.config.error_cmap,
                                        vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
                axes[3, col].set_title(f'Diff (CC={pp.correlations[idx]:.2f})', fontsize=9)
                axes[3, col].axis('off')
            
            fig.suptitle(f'Paired Patches @ {depth_km}km ({param})', fontsize=12)
            plt.tight_layout()
            self.save_figure(fig, f'T02_paired_patches_{param}_{int(depth_km)}km')
            break
    
    def _plot_T03_cc_distribution(self, results: Dict[float, TransformResult], param: str) -> None:
        """T03: CC分布分析"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        all_cc = []
        for depth_km, result in results.items():
            if result.paired_patches.n_total > 0:
                all_cc.extend(result.paired_patches.correlations.tolist())
        
        if not all_cc:
            return
        
        all_cc = np.array(all_cc)
        
        # 直方图
        axes[0].hist(all_cc, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', label='CC=0 threshold')
        axes[0].set_xlabel('Correlation Coefficient')
        axes[0].set_ylabel('Count')
        axes[0].set_title('CC Distribution')
        axes[0].legend()
        
        # CDF
        sorted_cc = np.sort(all_cc)
        cdf = np.arange(1, len(sorted_cc) + 1) / len(sorted_cc)
        axes[1].plot(sorted_cc, cdf, color='steelblue', linewidth=2)
        axes[1].axvline(x=0, color='r', linestyle='--', label='CC=0')
        axes[1].set_xlabel('Correlation Coefficient')
        axes[1].set_ylabel('CDF')
        axes[1].set_title('CC Cumulative Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, f'T03_cc_distribution_{param}')
    
    def _plot_T04_T05_dictionary_atoms(self, results: Dict[float, TransformResult], param: str) -> None:
        """
        T04-T05: D₁和D₂字典原子可视化（优化版）
        
        参考论文 Figure S1 风格：
        - D₁ 和 D₂ 配对显示，展示对应关系
        - 添加 colorbar 和相关性统计
        """
        for depth_km, result in results.items():
            if result.coupled_dict is None:
                continue
            
            cd = result.coupled_dict
            K, L1 = cd.D1.shape
            L2 = cd.D2.shape[1]
            side1 = int(np.sqrt(L1))
            side2 = int(np.sqrt(L2))
            
            # ========== T04: D₁ 字典（优化版）==========
            n_cols = 5
            n_rows = (K + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3*n_rows))
            axes = np.atleast_2d(axes)
            
            # 计算全局颜色范围
            vmax1 = np.abs(cd.D1).max()
            
            for k in range(K):
                row = k // n_cols
                col = k % n_cols
                atom = cd.D1[k].reshape(side1, side1)
                norm = np.linalg.norm(atom)
                im = axes[row, col].imshow(atom, cmap=self.config.atom_cmap, 
                                          vmin=-vmax1, vmax=vmax1)
                axes[row, col].set_title(f'D₁[{k+1}]\n‖·‖={norm:.3f}', fontsize=9)
                axes[row, col].axis('off')
            
            for k in range(K, n_rows * n_cols):
                row = k // n_cols
                col = k % n_cols
                axes[row, col].axis('off')
            
            # 添加 colorbar
            cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Atom Value', fontsize=10)
            
            fig.suptitle(f'D₁ Dictionary (Low-res {side1}×{side1}) @ {depth_km}km\n'
                        f'K={K} atoms, Patch size: {side1}° × {side1}°', fontsize=12)
            plt.tight_layout(rect=(0, 0, 0.9, 0.95))
            self.save_figure(fig, f'T04_D1_atoms_{param}_{int(depth_km)}km')
            
            # ========== T05: D₂ 字典（优化版）==========
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3*n_rows))
            axes = np.atleast_2d(axes)
            
            vmax2 = np.abs(cd.D2).max()
            
            for k in range(K):
                row = k // n_cols
                col = k % n_cols
                atom = cd.D2[k].reshape(side2, side2)
                norm = np.linalg.norm(atom)
                im = axes[row, col].imshow(atom, cmap=self.config.atom_cmap,
                                          vmin=-vmax2, vmax=vmax2)
                axes[row, col].set_title(f'D₂[{k+1}]\n‖·‖={norm:.3f}', fontsize=9)
                axes[row, col].axis('off')
            
            for k in range(K, n_rows * n_cols):
                row = k // n_cols
                col = k % n_cols
                axes[row, col].axis('off')
            
            # 添加 colorbar
            cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Atom Value', fontsize=10)
            
            fig.suptitle(f'D₂ Dictionary (High-res {side2}×{side2}) @ {depth_km}km\n'
                        f'K={K} atoms, Patch size: {side2}×{side2} @ 0.25°', fontsize=12)
            plt.tight_layout(rect=(0, 0, 0.9, 0.95))
            self.save_figure(fig, f'T05_D2_atoms_{param}_{int(depth_km)}km')
            
            # ========== T04b: D₁-D₂ 配对显示（论文 Figure S1 风格）==========
            self._plot_T04b_paired_atoms(cd, depth_km, param, side1, side2)
            
            break  # 只绘制第一个深度
    
    def _plot_T04b_paired_atoms(self, cd: CoupledDictionaryResult, 
                                depth_km: float, param: str,
                                side1: int, side2: int) -> None:
        """
        T04b: D₁-D₂ 配对原子显示（参考论文 Figure S1）
        
        展示每对 (D₁[k], D₂[k]) 的对应关系
        """
        K = cd.D1.shape[0]
        
        # 显示前 10 个原子的配对
        n_show = min(10, K)
        
        fig, axes = plt.subplots(n_show, 3, figsize=(10, 2.5*n_show))
        
        vmax1 = np.abs(cd.D1).max()
        vmax2 = np.abs(cd.D2).max()
        
        for k in range(n_show):
            # D₁ 原子
            atom1 = cd.D1[k].reshape(side1, side1)
            im1 = axes[k, 0].imshow(atom1, cmap=self.config.atom_cmap, 
                                   vmin=-vmax1, vmax=vmax1)
            axes[k, 0].set_title(f'D₁[{k+1}]' if k == 0 else '', fontsize=10)
            axes[k, 0].set_ylabel(f'Atom {k+1}', fontsize=9)
            axes[k, 0].axis('off')
            
            # D₂ 原子
            atom2 = cd.D2[k].reshape(side2, side2)
            im2 = axes[k, 1].imshow(atom2, cmap=self.config.atom_cmap,
                                   vmin=-vmax2, vmax=vmax2)
            axes[k, 1].set_title(f'D₂[{k+1}]' if k == 0 else '', fontsize=10)
            axes[k, 1].axis('off')
            
            # D₂ 上采样后与 D₁ 对比（使用 bicubic 上采样 D₁）
            atom1_up = np.asarray(zoom(atom1, side2 / side1, order=3))
            
            # 计算相关系数
            corr = np.corrcoef(atom1_up.flatten(), atom2.flatten())[0, 1]
            
            # 显示差异
            diff = atom2 - atom1_up
            vmax_diff = max(np.abs(diff).max(), 0.1)
            im3 = axes[k, 2].imshow(diff, cmap='RdBu_r', 
                                   vmin=-vmax_diff, vmax=vmax_diff)
            axes[k, 2].set_title(f'D₂ - D₁↑\nCC={corr:.3f}' if k == 0 
                                else f'CC={corr:.3f}', fontsize=9)
            axes[k, 2].axis('off')
        
        # 添加总标题
        fig.suptitle(f'D₁-D₂ Paired Dictionary Atoms @ {depth_km}km ({param})\n'
                    f'Left: D₁ ({side1}×{side1}), Middle: D₂ ({side2}×{side2}), '
                    f'Right: Difference (D₂ - bicubic(D₁))', fontsize=12)
        
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        self.save_figure(fig, f'T04b_paired_atoms_{param}_{int(depth_km)}km')
    
    def _plot_T08_validation_comparison(self, results: Dict[float, TransformResult], param: str) -> None:
        """T08: 验证集重建对比"""
        for depth_km, result in results.items():
            if result.coupled_dict is None or result.transformed_patches.size == 0:
                continue
            
            cd = result.coupled_dict
            pp = result.paired_patches
            
            n_show = min(5, len(result.transformed_patches))
            
            highres_size = pp.highres_patch_size
            
            # 获取验证集真值
            valid_mask = pp.valid_mask
            P2_all = pp.patches_highres[valid_mask]
            P2_val = P2_all[cd.val_indices]
            
            fig, axes = plt.subplots(4, n_show, figsize=(3*n_show, 12))
            
            for col in range(n_show):
                # GT
                gt = P2_val[col].reshape(highres_size, highres_size)
                vmin, vmax = gt.min(), gt.max()
                
                axes[0, col].imshow(gt, cmap=self.config.velocity_cmap, vmin=vmin, vmax=vmax)
                axes[0, col].set_title('Ground Truth', fontsize=9)
                axes[0, col].axis('off')
                
                # SDL
                sdl = result.transformed_patches[col].reshape(highres_size, highres_size)
                axes[1, col].imshow(sdl, cmap=self.config.velocity_cmap, vmin=vmin, vmax=vmax)
                axes[1, col].set_title('SDL Transform', fontsize=9)
                axes[1, col].axis('off')
                
                # Baseline
                baseline = result.baseline_patches[col].reshape(highres_size, highres_size)
                axes[2, col].imshow(baseline, cmap=self.config.velocity_cmap, vmin=vmin, vmax=vmax)
                axes[2, col].set_title('Baseline', fontsize=9)
                axes[2, col].axis('off')
                
                # SDL - GT
                diff = sdl - gt
                im = axes[3, col].imshow(diff, cmap=self.config.error_cmap,
                                        vmin=-0.1, vmax=0.1)
                axes[3, col].set_title('SDL - GT', fontsize=9)
                axes[3, col].axis('off')
            
            fig.suptitle(f'Validation Set Comparison @ {depth_km}km ({param})\n'
                        f'SDL Error: {result.transform_error:.2f}%, '
                        f'Baseline: {result.baseline_error:.2f}%, '
                        f'Improvement: {result.relative_improvement:.1f}%', fontsize=11)
            plt.tight_layout()
            self.save_figure(fig, f'T08_validation_{param}_{int(depth_km)}km')
            break
    
    def _plot_T10_depth_summary(self, results: Dict[float, TransformResult], 
                                param: str, target_model: str) -> None:
        """T10: 多深度汇总"""
        depths = []
        sdl_errors = []
        baseline_errors = []
        improvements = []
        
        for depth_km, result in sorted(results.items()):
            if not np.isnan(result.transform_error):
                depths.append(depth_km)
                sdl_errors.append(result.transform_error)
                baseline_errors.append(result.baseline_error)
                improvements.append(result.relative_improvement)
        
        if not depths:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 误差曲线
        axes[0].plot(depths, sdl_errors, '-o', label='SDL', color='#2ECC71', linewidth=2)
        axes[0].plot(depths, baseline_errors, '-s', label='Baseline', color='#E74C3C', linewidth=2)
        axes[0].set_xlabel('Depth (km)')
        axes[0].set_ylabel('Transform Error (%)')
        axes[0].set_title(f'Transform Error vs Depth ({param})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 改善率曲线
        axes[1].bar(depths, improvements, color='#3498DB', alpha=0.7)
        axes[1].axhline(y=20, color='r', linestyle='--', label='Target (20%)')
        axes[1].set_xlabel('Depth (km)')
        axes[1].set_ylabel('Relative Improvement (%)')
        axes[1].set_title(f'Improvement vs Depth ({param})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, f'T10_depth_summary_{param}_{target_model}')
    
    # =========================================================================
    # F组: Phase 1.5 完整切片变换可视化
    # =========================================================================
    
    def plot_full_slice_transform_results(self, 
                                          results: Dict[float, FullSliceTransformResult],
                                          param: str, 
                                          target_model: str) -> None:
        """
        F01-F06: 完整切片变换可视化
        """
        self.logger.info("\n  📊 生成F组可视化 (完整切片变换)...")
        
        # F01: 完整切片变换对比（论文Figure 3风格）
        self._plot_F01_full_slice_comparison(results, param)

        # F01b: 论文 Figure 3 复合版（切片 + 字典原子）
        self._plot_F01b_paper_composite(results, param)
        
        # F02: 多深度完整切片
        self._plot_F02_multi_depth_slices(results, param)
        
        # F03: 变换效果热图
        self._plot_F03_improvement_heatmap(results, param)
        
        # F04: 误差分布分析
        self._plot_F04_error_distribution(results, param)
        
        # F05: 权重分布图
        self._plot_F05_weight_map(results, param)
        
        # F06: 多深度汇总
        self._plot_F06_depth_summary(results, param, target_model)

    def _plot_F01b_paper_composite(self,
                                   results: Dict[float, FullSliceTransformResult],
                                   param: str) -> None:
        """
        F01b: 论文 Figure 3 风格复合图

        布局:
        - (a) 左侧 2×2: Target / Original / Reconstruction / Difference
        - (b) 右上 4×5: D1 字典原子
        - (c) 右下 4×5: D2 字典原子
        """
        candidates: List[Tuple[float, FullSliceTransformResult]] = []
        for depth_km, result in sorted(results.items()):
            if result.coupled_dict is None:
                continue
            if np.isnan(result.sdl_error):
                continue
            candidates.append((depth_km, result))

        if not candidates:
            self.logger.info("  ⏭️  跳过 F01b: 无可用耦合字典结果")
            return

        # 优先选择改善率最高的深度，得到一张代表性图
        depth_km, result = max(candidates, key=lambda item: item[1].relative_improvement)
        cd = result.coupled_dict
        if cd is None:
            return

        lon = result.lon_coords
        lat = result.lat_coords

        fig = plt.figure(figsize=(20, 11))
        outer = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.35],
                                 wspace=0.12, hspace=0.20)

        # --------------------------
        # (a) 左侧 2×2 切片图
        # --------------------------
        ax_a1 = fig.add_subplot(outer[0, 0], projection=ccrs.PlateCarree())
        ax_a2 = fig.add_subplot(outer[0, 1], projection=ccrs.PlateCarree())
        ax_a3 = fig.add_subplot(outer[1, 0], projection=ccrs.PlateCarree())
        ax_a4 = fig.add_subplot(outer[1, 1], projection=ccrs.PlateCarree())

        velocity_arrays = [result.target_slice, result.original_slice, result.transformed_slice]
        vmin = np.nanmin([np.nanmin(arr) for arr in velocity_arrays])
        vmax = np.nanmax([np.nanmax(arr) for arr in velocity_arrays])

        im_vel = ax_a1.pcolormesh(lon, lat, result.target_slice,
                                  cmap=self.config.velocity_cmap,
                                  vmin=vmin, vmax=vmax,
                                  transform=ccrs.PlateCarree())
        ax_a1.set_title('Target (High-res GT)', fontsize=11)

        ax_a2.pcolormesh(lon, lat, result.original_slice,
                         cmap=self.config.velocity_cmap,
                         vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree())
        ax_a2.set_title('Original (Base on target grid)', fontsize=11)

        ax_a3.pcolormesh(lon, lat, result.transformed_slice,
                         cmap=self.config.velocity_cmap,
                         vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree())
        ax_a3.set_title('SDL Reconstruction', fontsize=11)

        # 与论文示意一致：差异采用 Reconstruction - Target
        diff = result.transformed_slice - result.target_slice
        diff_max = np.nanpercentile(np.abs(diff), 98) if np.any(np.isfinite(diff)) else 0.1
        diff_max = max(float(diff_max), 0.01)
        im_diff = ax_a4.pcolormesh(lon, lat, diff,
                                   cmap=self.config.error_cmap,
                                   vmin=-diff_max, vmax=diff_max,
                                   transform=ccrs.PlateCarree())
        ax_a4.set_title('Difference (SDL - Target)', fontsize=11)

        map_axes = [ax_a1, ax_a2, ax_a3, ax_a4]
        for ax in map_axes:
            if hasattr(ax, 'set_extent'):
                ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())  # type: ignore[attr-defined]
            self._add_coastlines(ax)

        # 统一色标
        cbar_vel = fig.colorbar(im_vel, ax=[ax_a1, ax_a2, ax_a3],
                                orientation='horizontal', shrink=0.95, pad=0.06)
        cbar_vel.set_label(f'{param.upper()} (km/s)', fontsize=10)

        cbar_diff = fig.colorbar(im_diff, ax=ax_a4, orientation='horizontal',
                                 shrink=0.95, pad=0.06)
        cbar_diff.set_label(f'Δ{param.upper()} (km/s)', fontsize=10)

        ax_a1.annotate('(a)', xy=(-0.15, 1.07), xycoords='axes fraction',
                   fontsize=14, fontweight='bold')

        # --------------------------
        # (b) 右上 D1 字典原子
        # --------------------------
        d1_grid = outer[0, 2].subgridspec(4, 5, wspace=0.05, hspace=0.08)
        d2_grid = outer[1, 2].subgridspec(4, 5, wspace=0.05, hspace=0.08)

        K = min(20, cd.D1.shape[0], cd.D2.shape[0])
        side1 = int(np.sqrt(cd.D1.shape[1]))
        side2 = int(np.sqrt(cd.D2.shape[1]))
        vmax1 = np.max(np.abs(cd.D1)) if cd.D1.size > 0 else 1.0
        vmax2 = np.max(np.abs(cd.D2)) if cd.D2.size > 0 else 1.0

        first_b_ax: Optional[Any] = None
        first_c_ax: Optional[Any] = None

        for idx in range(20):
            row = idx // 5
            col = idx % 5

            ax_b = fig.add_subplot(d1_grid[row, col])
            ax_c = fig.add_subplot(d2_grid[row, col])

            if first_b_ax is None:
                first_b_ax = ax_b
            if first_c_ax is None:
                first_c_ax = ax_c

            if idx < K:
                atom1 = cd.D1[idx].reshape(side1, side1)
                atom2 = cd.D2[idx].reshape(side2, side2)
                ax_b.imshow(atom1, cmap=self.config.atom_cmap, vmin=-vmax1, vmax=vmax1)
                ax_c.imshow(atom2, cmap=self.config.atom_cmap, vmin=-vmax2, vmax=vmax2)
            ax_b.set_xticks([])
            ax_b.set_yticks([])
            ax_c.set_xticks([])
            ax_c.set_yticks([])

        if first_b_ax is not None:
            first_b_ax.text(-0.32, 1.12, '(b)', transform=first_b_ax.transAxes,
                            fontsize=14, fontweight='bold')
            first_b_ax.text(0.0, 1.12, f'D₁ atoms ({side1}×{side1})',
                            transform=first_b_ax.transAxes, fontsize=11, fontweight='bold')

        if first_c_ax is not None:
            first_c_ax.text(-0.32, 1.12, '(c)', transform=first_c_ax.transAxes,
                            fontsize=14, fontweight='bold')
            first_c_ax.text(0.0, 1.12, f'D₂ atoms ({side2}×{side2})',
                            transform=first_c_ax.transAxes, fontsize=11, fontweight='bold')

        fig.suptitle(
            f'Paper-style SDL Composite @ {depth_km} km ({param})\n'
            f'SDL Error: {result.sdl_error:.2f}% | Baseline: {result.baseline_error:.2f}% | '
            f'Improvement: {result.relative_improvement:.1f}% | K={cd.D1.shape[0]}',
            fontsize=14,
            fontweight='bold'
        )

        self.save_figure(fig, f'F01b_paper_composite_{param}_{int(depth_km)}km')
    
    def _plot_F01_full_slice_comparison(self, 
                                        results: Dict[float, FullSliceTransformResult],
                                        param: str) -> None:
        """
        F01: 完整切片变换对比（论文Figure 3a风格）(v3.3.4 添加海岸线)
        
        2×2子图:
        (a) 高分辨率目标 (GT)
        (b) 低分辨率原始
        (c) SDL重建
        (d) 差异 (SDL - 原始)
        """
        for depth_km, result in results.items():
            if np.isnan(result.sdl_error):
                continue
            
            fig = plt.figure(figsize=(14, 12))
            
            lon = result.lon_coords
            lat = result.lat_coords
            
            # 计算共同的速度范围
            all_data = [result.target_slice, result.original_slice, result.transformed_slice]
            vmin = np.nanmin([np.nanmin(d) for d in all_data])
            vmax = np.nanmax([np.nanmax(d) for d in all_data])
            
            # (a) 高分辨率目标 (GT)
            ax1 = self._create_geo_axes(fig, 2, 2, 1)
            im1 = ax1.pcolormesh(lon, lat, result.target_slice,
                                 cmap=self.config.velocity_cmap,
                                 vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            ax1.set_title(f'(a) {result.target_model.upper()} (Target GT)', fontsize=12)
            ax1.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
            self._add_coastlines(ax1)
            plt.colorbar(im1, ax=ax1, label=f'{param.upper()} (km/s)', shrink=0.8)
            
            # (b) 低分辨率原始
            ax2 = self._create_geo_axes(fig, 2, 2, 2)
            im2 = ax2.pcolormesh(lon, lat, result.original_slice,
                                 cmap=self.config.velocity_cmap,
                                 vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            ax2.set_title(f'(b) Base Model (Original)', fontsize=12)
            ax2.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
            self._add_coastlines(ax2)
            plt.colorbar(im2, ax=ax2, label=f'{param.upper()} (km/s)', shrink=0.8)
            
            # (c) SDL重建
            ax3 = self._create_geo_axes(fig, 2, 2, 3)
            im3 = ax3.pcolormesh(lon, lat, result.transformed_slice,
                                 cmap=self.config.velocity_cmap,
                                 vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            ax3.set_title(f'(c) SDL Reconstruction', fontsize=12)
            ax3.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
            self._add_coastlines(ax3)
            plt.colorbar(im3, ax=ax3, label=f'{param.upper()} (km/s)', shrink=0.8)
            
            # (d) 差异 (SDL - Original)
            ax4 = self._create_geo_axes(fig, 2, 2, 4)
            diff = result.transformed_slice - result.original_slice
            diff_max = np.nanpercentile(np.abs(diff), 98)
            diff_max = max(diff_max, 0.01)
            
            im4 = ax4.pcolormesh(lon, lat, diff,
                                 cmap=self.config.error_cmap,
                                 vmin=-diff_max, vmax=diff_max, transform=ccrs.PlateCarree())
            ax4.set_title(f'(d) Difference (SDL - Original)', fontsize=12)
            ax4.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
            self._add_coastlines(ax4)
            plt.colorbar(im4, ax=ax4, label=f'Δ{param.upper()} (km/s)', shrink=0.8)
            
            fig.suptitle(f'Full Slice Transformation @ {depth_km}km ({param})\n'
                        f'SDL Error: {result.sdl_error:.2f}%, '
                        f'Baseline: {result.baseline_error:.2f}%, '
                        f'Improvement: {result.relative_improvement:.1f}%',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            self.save_figure(fig, f'F01_full_slice_{param}_{int(depth_km)}km')
    
    def _plot_F02_multi_depth_slices(self, 
                                     results: Dict[float, FullSliceTransformResult],
                                     param: str) -> None:
        """
        F02: 多深度完整切片（论文Figure 4-5风格）
        
        每行一个深度，每列: GT / Original / SDL / Difference
        
        ⭐ v3.3.4: 选择代表性深度进行可视化，覆盖浅层-中层-深层
        """
        # 选择代表性深度 (最多5个)
        all_depths = sorted(results.keys())
        if len(all_depths) <= 5:
            depths = all_depths
        else:
            # 选择代表性深度：浅层(40-100km), 中层(200-300km), 深层(500-800km)
            representative_targets = [40, 100, 200, 500, 800]
            depths = []
            for target in representative_targets:
                # 找到最接近目标深度的可用深度
                closest = min(all_depths, key=lambda x: abs(x - target))
                if closest not in depths:
                    depths.append(closest)
                if len(depths) >= 5:
                    break
            depths = sorted(depths)
        
        n_depths = len(depths)
        
        if n_depths == 0:
            return
        
        fig, axes = plt.subplots(n_depths, 4, figsize=(16, 4*n_depths))
        if n_depths == 1:
            axes = axes.reshape(1, -1)
        
        for row, depth_km in enumerate(depths):
            result = results[depth_km]
            
            lon = result.lon_coords
            lat = result.lat_coords
            
            # 速度范围
            vmin = np.nanmin([np.nanmin(result.target_slice), 
                             np.nanmin(result.original_slice),
                             np.nanmin(result.transformed_slice)])
            vmax = np.nanmax([np.nanmax(result.target_slice), 
                             np.nanmax(result.original_slice),
                             np.nanmax(result.transformed_slice)])
            
            # GT
            axes[row, 0].pcolormesh(lon, lat, result.target_slice,
                                   cmap=self.config.velocity_cmap, vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f'{result.target_model.upper()} @ {depth_km}km' if row == 0 else f'{depth_km}km')
            axes[row, 0].set_aspect('equal')
            if row == 0:
                axes[row, 0].set_title('Target (GT)')
            
            # Original
            axes[row, 1].pcolormesh(lon, lat, result.original_slice,
                                   cmap=self.config.velocity_cmap, vmin=vmin, vmax=vmax)
            axes[row, 1].set_aspect('equal')
            if row == 0:
                axes[row, 1].set_title('Original')
            
            # SDL
            axes[row, 2].pcolormesh(lon, lat, result.transformed_slice,
                                   cmap=self.config.velocity_cmap, vmin=vmin, vmax=vmax)
            axes[row, 2].set_aspect('equal')
            if row == 0:
                axes[row, 2].set_title('SDL Reconstruction')
            
            # Difference
            diff = result.transformed_slice - result.original_slice
            diff_max = np.nanpercentile(np.abs(diff), 98) if np.any(~np.isnan(diff)) else 0.1
            axes[row, 3].pcolormesh(lon, lat, diff,
                                   cmap=self.config.error_cmap, 
                                   vmin=-diff_max, vmax=diff_max)
            axes[row, 3].set_aspect('equal')
            if row == 0:
                axes[row, 3].set_title('Difference')
            
            # 添加深度标签
            axes[row, 0].set_ylabel(f'{depth_km}km', fontsize=11, fontweight='bold')
        
        fig.suptitle(f'Multi-Depth Full Slice Transformation ({param})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.save_figure(fig, f'F02_multi_depth_{param}')
    
    def _plot_F03_improvement_heatmap(self, 
                                      results: Dict[float, FullSliceTransformResult],
                                      param: str) -> None:
        """
        F03: 变换效果热图 - 显示SDL相对于Baseline的改善率空间分布 (v3.3.4 添加海岸线)
        """
        for depth_km, result in list(results.items())[:1]:  # 只画第一个深度
            if np.isnan(result.sdl_error):
                continue
            
            fig = plt.figure(figsize=(15, 5))
            
            lon = result.lon_coords
            lat = result.lat_coords
            
            # 计算局部误差
            target = result.target_slice
            sdl = result.transformed_slice
            baseline = result.baseline_slice
            
            # 避免除零
            target_safe = np.where(np.abs(target) > 1e-6, target, np.nan)
            
            # SDL误差
            sdl_error = np.abs(sdl - target) / np.abs(target_safe) * 100
            
            # Baseline误差
            baseline_error = np.abs(baseline - target) / np.abs(target_safe) * 100
            
            # 改善率 = (baseline_error - sdl_error) / baseline_error * 100
            improvement = (baseline_error - sdl_error) / np.where(baseline_error > 0.1, baseline_error, np.nan) * 100
            
            # SDL误差图
            ax1 = self._create_geo_axes(fig, 1, 2, 1)
            err_max = np.nanpercentile(sdl_error, 95) if np.any(~np.isnan(sdl_error)) else 10
            im1 = ax1.pcolormesh(lon, lat, sdl_error,
                                 cmap='hot_r', vmin=0, vmax=err_max,
                                 transform=ccrs.PlateCarree())
            ax1.set_title(f'SDL Error (%) @ {depth_km}km', fontsize=12)
            ax1.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
            self._add_coastlines(ax1)
            plt.colorbar(im1, ax=ax1, label='Error (%)', shrink=0.8)
            
            # 改善率图
            ax2 = self._create_geo_axes(fig, 1, 2, 2)
            imp_max = np.nanpercentile(np.abs(improvement), 95) if np.any(~np.isnan(improvement)) else 50
            im2 = ax2.pcolormesh(lon, lat, improvement,
                                 cmap=self.config.improvement_cmap,
                                 vmin=-imp_max, vmax=imp_max,
                                 transform=ccrs.PlateCarree())
            ax2.set_title(f'Improvement (%) @ {depth_km}km', fontsize=12)
            ax2.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
            self._add_coastlines(ax2)
            plt.colorbar(im2, ax=ax2, label='Improvement (%)', shrink=0.8)
            
            plt.tight_layout()
            self.save_figure(fig, f'F03_improvement_heatmap_{param}_{int(depth_km)}km')
    
    def _plot_F04_error_distribution(self, 
                                     results: Dict[float, FullSliceTransformResult],
                                     param: str) -> None:
        """
        F04: 误差分布分析
        
        (a) 直方图
        (b) 空间分布
        (c) 统计信息
        """
        for depth_km, result in list(results.items())[:1]:
            if np.isnan(result.sdl_error) or not result.error_distribution:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 计算误差
            target = result.target_slice
            sdl = result.transformed_slice
            target_safe = np.where(np.abs(target) > 1e-6, target, np.nan)
            normalized_error = (sdl - target) / target_safe * 100
            
            valid_errors = normalized_error[~np.isnan(normalized_error)].ravel()
            
            if len(valid_errors) == 0:
                continue
            
            # (a) 直方图
            axes[0, 0].hist(valid_errors, bins=100, color='steelblue', 
                           edgecolor='white', alpha=0.7, density=True)
            axes[0, 0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
            axes[0, 0].axvline(x=np.mean(valid_errors), color='g', linestyle='-', 
                              linewidth=2, label=f'Mean={np.mean(valid_errors):.2f}%')
            axes[0, 0].set_xlabel('Normalized Error (%)')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('(a) Error Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # (b) Q-Q图
            from scipy import stats
            stats.probplot(valid_errors, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('(b) Q-Q Plot (Normal)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # (c) 空间误差分布
            err_max = np.nanpercentile(np.abs(normalized_error), 95)
            im = axes[1, 0].pcolormesh(result.lon_coords, result.lat_coords,
                                       normalized_error,
                                       cmap=self.config.error_cmap,
                                       vmin=-err_max, vmax=err_max)
            axes[1, 0].set_title('(c) Spatial Error Distribution')
            axes[1, 0].set_xlabel('Longitude (°)')
            axes[1, 0].set_ylabel('Latitude (°)')
            axes[1, 0].set_aspect('equal')
            plt.colorbar(im, ax=axes[1, 0], label='Error (%)', shrink=0.8)
            
            # (d) 统计信息文本
            stats_text = f"Error Distribution Statistics\n"
            stats_text += f"{'='*40}\n\n"
            stats_text += f"N samples: {len(valid_errors):,}\n"
            stats_text += f"Mean: {np.mean(valid_errors):.4f}%\n"
            stats_text += f"Std: {np.std(valid_errors):.4f}%\n"
            stats_text += f"Median: {np.median(valid_errors):.4f}%\n"
            stats_text += f"Min: {np.min(valid_errors):.4f}%\n"
            stats_text += f"Max: {np.max(valid_errors):.4f}%\n"
            stats_text += f"5th percentile: {np.percentile(valid_errors, 5):.4f}%\n"
            stats_text += f"95th percentile: {np.percentile(valid_errors, 95):.4f}%\n"
            stats_text += f"\n{'='*40}\n"
            stats_text += f"SDL Error: {result.sdl_error:.2f}%\n"
            stats_text += f"Baseline Error: {result.baseline_error:.2f}%\n"
            stats_text += f"Improvement: {result.relative_improvement:.1f}%\n"
            
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace')
            axes[1, 1].axis('off')
            axes[1, 1].set_title('(d) Statistics')
            
            fig.suptitle(f'Error Distribution Analysis @ {depth_km}km ({param})', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            self.save_figure(fig, f'F04_error_distribution_{param}_{int(depth_km)}km')
    
    def _plot_F05_weight_map(self, 
                             results: Dict[float, FullSliceTransformResult],
                             param: str) -> None:
        """
        F05: 权重分布图 - 显示patches拼接的权重分布
        """
        for depth_km, result in list(results.items())[:1]:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            lon = result.lon_coords
            lat = result.lat_coords
            
            # 权重分布
            im1 = axes[0].pcolormesh(lon, lat, result.weight_map,
                                    cmap='viridis')
            axes[0].set_title(f'Weight Map @ {depth_km}km', fontsize=12)
            axes[0].set_xlabel('Longitude (°)')
            axes[0].set_ylabel('Latitude (°)')
            axes[0].set_aspect('equal')
            plt.colorbar(im1, ax=axes[0], label='Weight', shrink=0.8)
            
            # 覆盖率掩码
            coverage = (result.weight_map > 0).astype(float)
            im2 = axes[1].pcolormesh(lon, lat, coverage,
                                    cmap='Greens', vmin=0, vmax=1)
            axes[1].set_title(f'Coverage @ {depth_km}km\n'
                             f'{result.coverage_ratio*100:.1f}% covered, '
                             f'{result.n_patches_processed} patches', fontsize=12)
            axes[1].set_xlabel('Longitude (°)')
            axes[1].set_ylabel('Latitude (°)')
            axes[1].set_aspect('equal')
            plt.colorbar(im2, ax=axes[1], label='Coverage', shrink=0.8)
            
            plt.tight_layout()
            self.save_figure(fig, f'F05_weight_map_{param}_{int(depth_km)}km')
    
    def _plot_F06_depth_summary(self, 
                                results: Dict[float, FullSliceTransformResult],
                                param: str,
                                target_model: str) -> None:
        """
        F06: 多深度汇总 - 误差和改善率随深度的变化
        """
        depths = []
        sdl_errors = []
        baseline_errors = []
        improvements = []
        error_means = []
        error_stds = []
        
        for depth_km, result in sorted(results.items()):
            if not np.isnan(result.sdl_error):
                depths.append(depth_km)
                sdl_errors.append(result.sdl_error)
                baseline_errors.append(result.baseline_error)
                improvements.append(result.relative_improvement)
                error_means.append(result.error_mean)
                error_stds.append(result.error_std)
        
        if not depths:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 误差曲线
        axes[0, 0].plot(depths, sdl_errors, '-o', label='SDL', color='#2ECC71', linewidth=2, markersize=8)
        axes[0, 0].plot(depths, baseline_errors, '-s', label='Baseline', color='#E74C3C', linewidth=2, markersize=8)
        axes[0, 0].axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Target (10%)')
        axes[0, 0].set_xlabel('Depth (km)')
        axes[0, 0].set_ylabel('Transform Error (%)')
        axes[0, 0].set_title('(a) Transform Error vs Depth')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 改善率曲线
        colors = ['#2ECC71' if imp > 20 else '#E74C3C' for imp in improvements]
        axes[0, 1].bar(depths, improvements, color=colors, alpha=0.7, width=depths[1]-depths[0] if len(depths) > 1 else 20)
        axes[0, 1].axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='Target (20%)')
        axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
        axes[0, 1].set_xlabel('Depth (km)')
        axes[0, 1].set_ylabel('Relative Improvement (%)')
        axes[0, 1].set_title('(b) Improvement vs Depth')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 误差均值
        axes[1, 0].plot(depths, error_means, '-o', color='#3498DB', linewidth=2, markersize=8)
        axes[1, 0].fill_between(depths, 
                                [m - s for m, s in zip(error_means, error_stds)],
                                [m + s for m, s in zip(error_means, error_stds)],
                                alpha=0.3, color='#3498DB')
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Target (±1%)')
        axes[1, 0].axhline(y=-1, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Depth (km)')
        axes[1, 0].set_ylabel('Error Mean (%)')
        axes[1, 0].set_title('(c) Error Mean ± Std vs Depth')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 误差标准差
        axes[1, 1].plot(depths, error_stds, '-o', color='#9B59B6', linewidth=2, markersize=8)
        axes[1, 1].axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Target (5%)')
        axes[1, 1].set_xlabel('Depth (km)')
        axes[1, 1].set_ylabel('Error Std (%)')
        axes[1, 1].set_title('(d) Error Std vs Depth')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(f'Phase 1.5 Full Slice Transform Summary ({param}, {target_model})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.save_figure(fig, f'F06_depth_summary_{param}_{target_model}')
    
    # =========================================================================
    # H组: 超参数搜索可视化 (v3.3 新增)
    # =========================================================================
    
    def plot_grid_search_results(self, 
                                 results: List['GridSearchResult'],
                                 best_config: Optional['GridSearchResult'] = None) -> None:
        """
        H01-H05: 超参数网格搜索结果可视化
        
        Args:
            results: 网格搜索结果列表
            best_config: 最优配置
        """
        if not results:
            self.logger.warning("      ⚠️ 无网格搜索结果可视化")
            return
        
        self.logger.info("\n  📊 生成H组可视化 (超参数搜索)...")
        
        # H01: K×λ 性能热力图
        self._plot_H01_grid_search_heatmap(results)
        
        # H02: 参数敏感性曲线
        self._plot_H02_param_sensitivity(results)
        
        # H03: 最优配置卡片
        if best_config:
            self._plot_H03_optimal_config_card(best_config, results)
        
        # H04: 搜索结果排序
        self._plot_H04_ranking_chart(results)
    
    def _plot_H01_grid_search_heatmap(self, results: List['GridSearchResult']) -> None:
        """
        H01: K×λ 性能热力图
        
        展示不同 K 和 λ 组合下的性能
        """
        # 提取唯一的 K 和 alpha 值
        K_values = sorted(set(r.n_components for r in results))
        alpha_values = sorted(set(r.alpha for r in results))
        window_values = sorted(set(r.window for r in results))
        
        if len(K_values) < 2 or len(alpha_values) < 2:
            self.logger.info("      ⚠️ K或λ值不足，跳过热力图")
            return
        
        # 为每个窗口大小创建一个热力图
        n_windows = len(window_values)
        fig, axes = plt.subplots(2, n_windows, figsize=(5*n_windows, 10))
        
        if n_windows == 1:
            axes = axes.reshape(-1, 1)
        
        for w_idx, window in enumerate(window_values):
            # 筛选当前窗口的结果
            window_results = [r for r in results if r.window == window]
            
            # 创建改善率矩阵
            improve_matrix = np.full((len(K_values), len(alpha_values)), np.nan)
            error_matrix = np.full((len(K_values), len(alpha_values)), np.nan)
            
            for r in window_results:
                k_idx = K_values.index(r.n_components)
                a_idx = alpha_values.index(r.alpha)
                improve_matrix[k_idx, a_idx] = r.relative_improvement
                error_matrix[k_idx, a_idx] = r.transform_error
            
            # 绘制改善率热力图
            im1 = axes[0, w_idx].imshow(improve_matrix, cmap='RdYlGn', 
                                       aspect='auto', origin='lower')
            axes[0, w_idx].set_xticks(range(len(alpha_values)))
            axes[0, w_idx].set_xticklabels([f'{a:.2f}' for a in alpha_values])
            axes[0, w_idx].set_yticks(range(len(K_values)))
            axes[0, w_idx].set_yticklabels(K_values)
            axes[0, w_idx].set_xlabel('λ (sparsity)')
            axes[0, w_idx].set_ylabel('K (atoms)')
            axes[0, w_idx].set_title(f'Improvement (%) - Window {window}°')
            
            # 添加数值标注
            for i in range(len(K_values)):
                for j in range(len(alpha_values)):
                    if not np.isnan(improve_matrix[i, j]):
                        color = 'white' if improve_matrix[i, j] > 30 else 'black'
                        axes[0, w_idx].text(j, i, f'{improve_matrix[i, j]:.1f}',
                                          ha='center', va='center', fontsize=8, color=color)
            
            fig.colorbar(im1, ax=axes[0, w_idx], shrink=0.8)
            
            # 绘制误差热力图
            im2 = axes[1, w_idx].imshow(error_matrix, cmap='RdYlGn_r',
                                       aspect='auto', origin='lower')
            axes[1, w_idx].set_xticks(range(len(alpha_values)))
            axes[1, w_idx].set_xticklabels([f'{a:.2f}' for a in alpha_values])
            axes[1, w_idx].set_yticks(range(len(K_values)))
            axes[1, w_idx].set_yticklabels(K_values)
            axes[1, w_idx].set_xlabel('λ (sparsity)')
            axes[1, w_idx].set_ylabel('K (atoms)')
            axes[1, w_idx].set_title(f'Transform Error (%) - Window {window}°')
            
            for i in range(len(K_values)):
                for j in range(len(alpha_values)):
                    if not np.isnan(error_matrix[i, j]):
                        color = 'white' if error_matrix[i, j] < 3 else 'black'
                        axes[1, w_idx].text(j, i, f'{error_matrix[i, j]:.1f}',
                                          ha='center', va='center', fontsize=8, color=color)
            
            fig.colorbar(im2, ax=axes[1, w_idx], shrink=0.8)
        
        fig.suptitle('H01: Hyperparameter Grid Search - K × λ Performance', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        self.save_figure(fig, 'H01_grid_search_heatmap')
    
    def _plot_H02_param_sensitivity(self, results: List['GridSearchResult']) -> None:
        """
        H02: 参数敏感性曲线
        
        展示各参数对性能的影响
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 按 K 值分组
        K_values = sorted(set(r.n_components for r in results))
        alpha_values = sorted(set(r.alpha for r in results))
        window_values = sorted(set(r.window for r in results))
        
        # (a) K 敏感性
        k_means = []
        k_stds = []
        for K in K_values:
            K_results = [r.relative_improvement for r in results if r.n_components == K]
            if K_results:
                k_means.append(np.mean(K_results))
                k_stds.append(np.std(K_results))
            else:
                k_means.append(np.nan)
                k_stds.append(np.nan)
        
        axes[0, 0].errorbar(K_values, k_means, yerr=k_stds, 
                           fmt='-o', capsize=5, color='#3498DB', linewidth=2)
        axes[0, 0].axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Target (20%)')
        axes[0, 0].set_xlabel('K (Number of Atoms)')
        axes[0, 0].set_ylabel('Relative Improvement (%)')
        axes[0, 0].set_title('(a) K Sensitivity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # (b) λ 敏感性
        alpha_means = []
        alpha_stds = []
        for alpha in alpha_values:
            a_results = [r.relative_improvement for r in results if r.alpha == alpha]
            if a_results:
                alpha_means.append(np.mean(a_results))
                alpha_stds.append(np.std(a_results))
            else:
                alpha_means.append(np.nan)
                alpha_stds.append(np.nan)
        
        axes[0, 1].errorbar(alpha_values, alpha_means, yerr=alpha_stds,
                           fmt='-s', capsize=5, color='#E74C3C', linewidth=2)
        axes[0, 1].axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Target (20%)')
        axes[0, 1].set_xlabel('λ (Sparsity)')
        axes[0, 1].set_ylabel('Relative Improvement (%)')
        axes[0, 1].set_title('(b) λ Sensitivity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # (c) 窗口大小敏感性
        window_means = []
        window_stds = []
        for window in window_values:
            w_results = [r.relative_improvement for r in results if r.window == window]
            if w_results:
                window_means.append(np.mean(w_results))
                window_stds.append(np.std(w_results))
            else:
                window_means.append(np.nan)
                window_stds.append(np.nan)
        
        axes[1, 0].errorbar(window_values, window_means, yerr=window_stds,
                           fmt='-^', capsize=5, color='#2ECC71', linewidth=2)
        axes[1, 0].axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Target (20%)')
        axes[1, 0].set_xlabel('Window Size (°)')
        axes[1, 0].set_ylabel('Relative Improvement (%)')
        axes[1, 0].set_title('(c) Window Size Sensitivity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # (d) 综合评分分布
        scores = [r.score for r in results]
        improvements = [r.relative_improvement for r in results]
        errors = [r.transform_error for r in results]
        
        sc = axes[1, 1].scatter(errors, improvements, c=scores, 
                               cmap='viridis', s=80, alpha=0.7)
        axes[1, 1].axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Target (20%)')
        axes[1, 1].axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='Error < 10%')
        axes[1, 1].set_xlabel('Transform Error (%)')
        axes[1, 1].set_ylabel('Relative Improvement (%)')
        axes[1, 1].set_title('(d) Error vs Improvement (colored by Score)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        cbar = fig.colorbar(sc, ax=axes[1, 1])
        cbar.set_label('Composite Score')
        
        fig.suptitle('H02: Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        self.save_figure(fig, 'H02_param_sensitivity')
    
    def _plot_H03_optimal_config_card(self, best_config: 'GridSearchResult',
                                       results: List['GridSearchResult']) -> None:
        """
        H03: 最优配置卡片
        
        显示最优配置的详细信息
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # 左侧: 配置信息卡片
        ax = axes[0]
        ax.axis('off')
        
        card_text = f"""
╔══════════════════════════════════════════════════════════╗
║            ⭐ OPTIMAL CONFIGURATION ⭐                    ║
╠══════════════════════════════════════════════════════════╣
║  Config Name:  {best_config.config_name:<40} ║
╠══════════════════════════════════════════════════════════╣
║  PARAMETERS                                              ║
║  ──────────────────────────────────────────────────────  ║
║  • Window Size:     {best_config.window:>6.1f}°                              ║
║  • Stride:          {best_config.stride:>6.2f}°                              ║
║  • K (atoms):       {best_config.n_components:>6d}                               ║
║  • λ (sparsity):    {best_config.alpha:>6.2f}                              ║
╠══════════════════════════════════════════════════════════╣
║  PERFORMANCE                                             ║
║  ──────────────────────────────────────────────────────  ║
║  • Transform Error: {best_config.transform_error:>6.2f}%                             ║
║  • Baseline Error:  {best_config.baseline_error:>6.2f}%                             ║
║  • Improvement:     {best_config.relative_improvement:>6.1f}%                             ║
║  • Composite Score: {best_config.score:>6.4f}                             ║
║  • Total Patches:   {best_config.n_patches:>6d}                               ║
╚══════════════════════════════════════════════════════════╝
"""
        
        ax.text(0.1, 0.5, card_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # 右侧: 深度结果柱状图
        ax2 = axes[1]
        
        if best_config.depth_results:
            depths = list(best_config.depth_results.keys())
            improvements = [best_config.depth_results[d].get('improvement', 0) for d in depths]
            errors = [best_config.depth_results[d].get('transform_error', 0) for d in depths]
            
            x = np.arange(len(depths))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, improvements, width, label='Improvement (%)',
                           color='#2ECC71', alpha=0.7)
            bars2 = ax2.bar(x + width/2, errors, width, label='Error (%)',
                           color='#E74C3C', alpha=0.7)
            
            ax2.axhline(y=20, color='green', linestyle='--', alpha=0.5)
            ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5)
            
            ax2.set_xlabel('Depth (km)')
            ax2.set_ylabel('Value (%)')
            ax2.set_title('Performance by Depth')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'{int(d)}km' for d in depths])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 添加数值标注
            for bar in bars1:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        fig.suptitle('H03: Optimal Configuration Summary', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        self.save_figure(fig, 'H03_optimal_config')
    
    def _plot_H04_ranking_chart(self, results: List['GridSearchResult']) -> None:
        """
        H04: 搜索结果排序图
        
        展示 Top N 配置的对比
        """
        # 按评分排序
        sorted_results = sorted(results, key=lambda x: -x.score)[:15]  # Top 15
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左侧: 条形图
        names = [r.config_name for r in sorted_results]
        scores = [r.score for r in sorted_results]
        improvements = [r.relative_improvement for r in sorted_results]
        
        y_pos = np.arange(len(names))
        
        # 评分条形图
        colors = ['#2ECC71' if s > 0.5 else '#F39C12' if s > 0.3 else '#E74C3C' 
                 for s in scores]
        bars = axes[0].barh(y_pos, scores, color=colors, alpha=0.7)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(names, fontsize=9)
        axes[0].set_xlabel('Composite Score')
        axes[0].set_title('(a) Configuration Ranking by Score')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # 添加数值标注
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            axes[0].annotate(f'{score:.3f}',
                           xy=(width, bar.get_y() + bar.get_height()/2),
                           xytext=(3, 0), textcoords="offset points",
                           ha='left', va='center', fontsize=8)
        
        # 右侧: 改善率柱状图
        colors2 = ['#2ECC71' if imp > 20 else '#F39C12' if imp > 10 else '#E74C3C'
                  for imp in improvements]
        bars2 = axes[1].barh(y_pos, improvements, color=colors2, alpha=0.7)
        axes[1].axvline(x=20, color='green', linestyle='--', alpha=0.5, label='Target (20%)')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(names, fontsize=9)
        axes[1].set_xlabel('Relative Improvement (%)')
        axes[1].set_title('(b) Configuration Ranking by Improvement')
        axes[1].invert_yaxis()
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        for i, (bar, imp) in enumerate(zip(bars2, improvements)):
            width = bar.get_width()
            axes[1].annotate(f'{imp:.1f}%',
                           xy=(width, bar.get_y() + bar.get_height()/2),
                           xytext=(3, 0), textcoords="offset points",
                           ha='left', va='center', fontsize=8)
        
        fig.suptitle('H04: Grid Search Results Ranking (Top 15)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        self.save_figure(fig, 'H04_ranking_chart')
    
    # =========================================================================
    # S组: 综合评估可视化
    # =========================================================================
    
    def plot_summary(self, all_results: Dict) -> None:
        """
        S01-S06: 综合评估可视化
        """
        self.logger.info("\n  📊 生成S组可视化 (综合评估)...")
        
        # S01: 总体误差报告
        self._plot_S01_error_report(all_results)
    
    def _plot_S01_error_report(self, all_results: Dict) -> None:
        """S01: 总体误差报告"""
        fig, axes = plt.subplots(figsize=(12, 8))
        ax = axes  # 单个axes
        
        report_text = "SDL Fusion v3.4 - Summary Report\n"
        report_text += "=" * 50 + "\n\n"
        
        # Phase 0 结果
        if 'phase0' in all_results:
            report_text += "Phase 0: Dictionary Validation\n"
            report_text += "-" * 30 + "\n"
            for model_name, model_results in all_results['phase0'].items():
                for depth_km, result in model_results.items():
                    status = "✓" if result.passed else "✗"
                    report_text += f"  {model_name} @ {depth_km}km: {result.error_stats.get('global_error', 0):.2f}% [{status}]\n"
            report_text += "\n"
        
        # Phase 1 结果
        if 'phase1' in all_results:
            report_text += "Phase 1: 2D Transform\n"
            report_text += "-" * 30 + "\n"
            for depth_km, result in all_results['phase1'].items():
                status = "✓" if result.passed else "✗"
                report_text += f"  {depth_km}km: SDL={result.transform_error:.2f}%, "
                report_text += f"Baseline={result.baseline_error:.2f}%, "
                report_text += f"Improve={result.relative_improvement:.1f}% [{status}]\n"
        
        ax.text(0.05, 0.95, report_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
        
        plt.tight_layout()
        self.save_figure(fig, 'S01_error_report')


# ==============================================================================
# 第七部分: 超参数搜索器 (v3.3 新增)
# ==============================================================================

class HyperparameterSearcher:
    """
    Phase H: 超参数联合网格搜索器 (v3.3 新增)
    
    功能:
    - 联合搜索 K × λ × window × stride 参数组合
    - 自动记录最优配置到 optimal_config.json
    - 生成 H01-H05 搜索结果可视化
    
    论文依据 (Section 3.2):
    > "Out of 40 combinations of hyperparameter configurations, the optimal 
    > number of atoms in D₁ and D₂ is determined to be 20, with an optimal 
    > sparsity control factor of 0.1."
    """
    
    def __init__(self, sdl_fusion: 'SDLFusionV33', logger: logging.Logger):
        """
        初始化超参数搜索器
        
        Args:
            sdl_fusion: SDL融合器实例
            logger: 日志记录器
        """
        self.sdl_fusion = sdl_fusion
        self.logger = logger
        self.results: List[GridSearchResult] = []
        self.best_config: Optional[GridSearchResult] = None
        
        # 输出目录
        self.results_dir = self.sdl_fusion.base_config.dirs['results'] / 'SDL_grid_search'
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def search(self,
               K_values: List[int],
               alpha_values: List[float],
               depths: List[float] = [100, 200, 300],
               param: str = 'vs',
               target_model: str = 'fwea') -> Optional[GridSearchResult]:
        """
        执行 K × λ 联合网格搜索（论文方法）
        
        ⭐ 论文只搜索 K 和 λ，patch size 由物理约束决定，不需要搜索
        > "Out of 40 combinations of hyperparameter configurations, the optimal 
        >  number of atoms in D₁ and D₂ is determined to be 20, with an optimal 
        >  sparsity control factor of 0.1."
        
        Args:
            K_values: 字典原子数列表，如 [15, 20, 25]
            alpha_values: 稀疏性参数列表，如 [0.05, 0.1, 0.15]
            depths: 测试深度列表
            param: 参数名 ('vs' 或 'vp')
            target_model: 目标模型 ('cses' 或 'ust')
            
        Returns:
            最优配置 GridSearchResult
        """
        # 使用固定的窗口和步长（由 Phase1Config 配置）
        window = self.sdl_fusion.config.phase1.physical_window_size
        stride = self.sdl_fusion.config.phase1.physical_stride
        
        total_combinations = len(K_values) * len(alpha_values)
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🔍 Phase H: K × λ 联合网格搜索 (论文方法)")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"  总组合数: {total_combinations}")
        self.logger.info(f"  固定窗口: {window}° (patch size 由物理约束决定)")
        self.logger.info(f"  固定步长: {stride}° ({int((window-stride)/window*100)}% 重叠)")
        self.logger.info(f"  K (原子数): {K_values}")
        self.logger.info(f"  λ (稀疏性): {alpha_values}")
        self.logger.info(f"  测试深度: {depths}")
        
        self.results = []
        current_idx = 0
        
        # 使用 tqdm 显示总体进度
        pbar = tqdm(total=total_combinations, desc="🔍 Grid Search K×λ",
                    unit="config", ncols=100, file=sys.stdout)
        
        for K in K_values:
            for alpha in alpha_values:
                current_idx += 1
                config_name = f"K{K}_a{alpha}"
                pbar.set_postfix_str(f"{config_name}")
                
                self.logger.info(f"\n  [{current_idx}/{total_combinations}] 测试: {config_name}")
                
                try:
                    # 设置参数（窗口和步长保持固定）
                    self.sdl_fusion.config.phase1.n_components = K
                    self.sdl_fusion.config.phase1.alpha = alpha
                    
                    # 运行单次测试
                    result = self._run_single_test(
                        config_name, depths, param, target_model
                    )
                    
                    if result is not None:
                        # 计算综合评分
                        result.score = self._compute_score(result)
                        self.results.append(result)
                        
                        # 更新最优配置
                        if self.best_config is None or result.score > self.best_config.score:
                            self.best_config = result
                            self.logger.info(f"    ⭐ 新最优配置! 评分: {result.score:.4f}")
                        
                        self.logger.info(f"    误差: {result.transform_error:.2f}%, "
                                       f"改善: {result.relative_improvement:.1f}%, "
                                       f"评分: {result.score:.4f}")
                
                except Exception as e:
                    self.logger.warning(f"    ⚠️ 测试失败: {e}")
                    
                    pbar.update(1)
        
        pbar.close()
        
        # 保存结果
        self._save_results()
        
        # 打印摘要
        self._print_summary()
        
        return self.best_config
    
    def _run_single_test(self, config_name: str, depths: List[float],
                         param: str, target_model: str) -> Optional[GridSearchResult]:
        """运行单次参数配置测试"""
        try:
            # 重新初始化 2D 变换器（使用新参数）
            self.sdl_fusion.transformer_2d = TwoDTransformer(
                self.sdl_fusion.config.phase1, 
                self.sdl_fusion.config, 
                self.sdl_fusion.logger
            )
            
            depth_results = {}
            all_errors = []
            all_improvements = []
            all_baseline_errors = []
            total_patches = 0
            
            for depth_km in depths:
                try:
                    # ⭐ v3.5: 使用 model_loader
                    result = self.sdl_fusion.transformer_2d.run(
                        self.sdl_fusion.model_loader,
                        param, depth_km, target_model
                    )
                    
                    if result.coupled_dict is not None:
                        depth_results[depth_km] = {
                            'transform_error': result.transform_error,
                            'baseline_error': result.baseline_error,
                            'improvement': result.relative_improvement,
                            'n_patches': result.paired_patches.n_valid
                        }
                        all_errors.append(result.transform_error)
                        all_improvements.append(result.relative_improvement)
                        all_baseline_errors.append(result.baseline_error)
                        total_patches += result.paired_patches.n_valid
                        
                except Exception as e:
                    self.logger.debug(f"      深度 {depth_km}km 测试失败: {e}")
                    continue
            
            if not depth_results:
                return None
            
            # 创建结果
            return GridSearchResult(
                config_name=config_name,
                window=self.sdl_fusion.config.phase1.physical_window_size,
                stride=self.sdl_fusion.config.phase1.physical_stride,
                n_components=self.sdl_fusion.config.phase1.n_components,
                alpha=self.sdl_fusion.config.phase1.alpha,
                transform_error=float(np.mean(all_errors)),
                baseline_error=float(np.mean(all_baseline_errors)),
                relative_improvement=float(np.mean(all_improvements)),
                error_mean=float(np.mean(all_errors)),
                error_std=float(np.std(all_errors)),
                n_patches=total_patches,
                depth_results=depth_results
            )
            
        except Exception as e:
            self.logger.debug(f"      单次测试异常: {e}")
            return None
    
    def _compute_score(self, result: GridSearchResult) -> float:
        """
        计算综合评估分数
        
        论文目标: 最小化验证集误差 D(P̂ᵛ, P₂ᵛ)
        
        综合评分 = 0.4 × 误差得分 + 0.4 × 改善率得分 + 0.2 × 稳定性得分
        """
        # 误差得分 (越低越好，归一化到 0-1)
        error_score = 1.0 - min(result.transform_error / 20.0, 1.0)
        
        # 改善率得分 (越高越好，归一化到 0-1)
        improve_score = min(result.relative_improvement / 50.0, 1.0)
        
        # 稳定性得分 (误差标准差越低越好)
        std_score = 1.0 - min(result.error_std / 10.0, 1.0)
        
        return 0.4 * error_score + 0.4 * improve_score + 0.2 * std_score
    
    def _save_results(self) -> None:
        """保存搜索结果"""
        if not self.results:
            return
        
        # 保存 CSV
        csv_path = self.results_dir / 'grid_search_results.csv'
        with open(csv_path, 'w') as f:
            f.write("config_name,window,stride,K,alpha,transform_error,baseline_error,"
                   "improvement,error_mean,error_std,n_patches,score\n")
            for r in sorted(self.results, key=lambda x: -x.score):
                f.write(f"{r.config_name},{r.window},{r.stride},{r.n_components},"
                       f"{r.alpha},{r.transform_error:.3f},{r.baseline_error:.3f},"
                       f"{r.relative_improvement:.2f},{r.error_mean:.3f},"
                       f"{r.error_std:.3f},{r.n_patches},{r.score:.4f}\n")
        
        self.logger.info(f"  📊 搜索结果已保存: {csv_path}")
        
        # 保存最优配置 JSON
        if self.best_config:
            optimal_path = self.results_dir / 'optimal_config.json'
            with open(optimal_path, 'w') as f:
                json.dump(self.best_config.to_dict(), f, indent=2)
            self.logger.info(f"  ⭐ 最优配置已保存: {optimal_path}")
        
        # 生成 H 组可视化
        self._generate_visualizations()
    
    def _generate_visualizations(self) -> None:
        """生成 H 组搜索结果可视化"""
        if not self.results:
            return
        
        try:
            # 使用 SDL fusion 的 visualizer
            if hasattr(self.sdl_fusion, 'visualizer'):
                self.logger.info("\n  📊 生成H组可视化...")
                self.sdl_fusion.visualizer.plot_grid_search_results(
                    self.results, self.best_config
                )
            else:
                self.logger.warning("  ⚠️ Visualizer 未初始化，跳过可视化")
        except Exception as e:
            self.logger.warning(f"  ⚠️ H组可视化生成失败: {e}")
    
    def _print_summary(self) -> None:
        """打印搜索摘要"""
        if not self.results:
            self.logger.info("\n  ⚠️ 没有有效的搜索结果")
            return
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 网格搜索结果摘要")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"  有效配置数: {len(self.results)}")
        
        if self.best_config:
            self.logger.info(f"\n  ⭐ 最优配置:")
            self.logger.info(f"    名称: {self.best_config.config_name}")
            self.logger.info(f"    窗口: {self.best_config.window}°, 步长: {self.best_config.stride}°")
            self.logger.info(f"    K: {self.best_config.n_components}, λ: {self.best_config.alpha}")
            self.logger.info(f"    变换误差: {self.best_config.transform_error:.2f}%")
            self.logger.info(f"    相对改善: {self.best_config.relative_improvement:.1f}%")
            self.logger.info(f"    综合评分: {self.best_config.score:.4f}")
        
        # Top 5 配置
        self.logger.info(f"\n  🏆 Top 5 配置:")
        for i, r in enumerate(sorted(self.results, key=lambda x: -x.score)[:5], 1):
            self.logger.info(f"    {i}. {r.config_name}: 误差={r.transform_error:.2f}%, "
                           f"改善={r.relative_improvement:.1f}%, 评分={r.score:.4f}")
    
    def load_optimal_config(self, config_path: Optional[Path] = None) -> Optional[Dict]:
        """
        加载最优配置
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
            
        Returns:
            配置字典
        """
        if config_path is None:
            config_path = self.results_dir / 'optimal_config.json'
        
        if not config_path.exists():
            self.logger.warning(f"⚠️ 最优配置文件不存在: {config_path}")
            return None
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self.logger.info(f"✅ 已加载最优配置: {config_path}")
            self.logger.info(f"   配置: {config_data['config_name']}")
            
            return config_data
            
        except Exception as e:
            self.logger.error(f"❌ 加载配置失败: {e}")
            return None
    
    def apply_optimal_config(self) -> bool:
        """应用最优配置到 SDL 融合器"""
        config_data = self.load_optimal_config()
        
        if config_data is None:
            return False
        
        try:
            self.sdl_fusion.config.phase1.physical_window_size = config_data['window']
            self.sdl_fusion.config.phase1.physical_stride = config_data['stride']
            self.sdl_fusion.config.phase1.n_components = config_data['n_components']
            self.sdl_fusion.config.phase1.alpha = config_data['alpha']
            
            self.logger.info(f"✅ 已应用最优配置:")
            self.logger.info(f"   窗口: {config_data['window']}°, 步长: {config_data['stride']}°")
            self.logger.info(f"   K: {config_data['n_components']}, λ: {config_data['alpha']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 应用配置失败: {e}")
            return False


# ==============================================================================
# 第八部分: SDL融合主类
# ==============================================================================

class SDLFusionV33:
    """
    稀疏字典学习速度模型融合 v3.5
    
    ⭐ v3.5 重构:
    1. 移除 UnifiedModelFramework，使用 ModelLoader
    2. 所有操作直接使用原始模型数据，不再插值
    3. Phase 0/可视化使用原始分辨率数据
    
    主要改进 (v3.3):
    1. ⭐ Phase H: 超参数联合网格搜索
    2. ⭐ 三种运行模式 (代码配置/网格搜索/最优配置)
    3. ⭐ HyperparameterSearcher 类
    4. ⭐ H 组可视化 (H01-H05)
    
    主要改进 (v3.2):
    1. D₂ 纯最小二乘计算
    2. DirectPatchExtractor 直接提取
    3. 参数敏感性测试
    """
    
    def __init__(self, config: Optional[SDLConfigV31] = None,
                 output_dir: Optional[Path] = None):
        """
        初始化SDL融合器
        
        Args:
            config: SDL配置，如果为None则使用默认配置
            output_dir: 输出目录
        """
        # 加载配置
        self.config = config or SDLConfigV31()
        
        # 加载基础配置
        self.base_config = BaseConfig()
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.SDL_v3.5',
            self.config.logging['level']
        )
        
        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.base_config.dirs['figures'] / 'SDL_Fusion_v3.5'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置数据目录
        self.data_dir = self.base_config.dirs['data'] / 'fwi-models' / 'processed'
        
        # 初始化组件
        self._init_components()
        
        # 存储结果
        self.all_results: Dict = {}
        
        # v3.3: 超参数搜索器
        self.hyperparam_searcher: Optional[HyperparameterSearcher] = None
        
        self.logger.info("🎯 SDL Fusion v3.5 初始化完成")
        self._print_config_summary()
    
    def _init_components(self) -> None:
        """初始化各阶段组件"""
        # ⭐ v3.5: 使用 ModelLoader 替代 UnifiedModelFramework
        self.model_loader = ModelLoader(self.logger)
        
        # Phase 0: 字典验证器
        self.validator = DictionaryValidator(
            self.config.phase0, self.logger
        )
        
        # Phase 1: 2D变换器 (Patch级别验证)
        self.transformer_2d = TwoDTransformer(
            self.config.phase1, self.config, self.logger
        )
        
        # Phase 1.5: 完整切片变换器
        self.full_slice_transformer = FullSliceTransformer(
            self.config.phase1, self.config, self.logger
        )
        
        # 可视化器
        self.visualizer = SDLVisualizerV31(
            self.config.visualization, self.output_dir, self.logger
        )
    
    def _print_config_summary(self) -> None:
        """打印配置摘要"""
        self.logger.info("\n" + "="*60)
        self.logger.info("📋 SDL Fusion v3.5 配置摘要")
        self.logger.info("="*60)
        
        self.logger.info(f"\n  📂 数据目录: {self.data_dir}")
        self.logger.info(f"  📁 输出目录: {self.output_dir}")
        
        self.logger.info(f"\n  🎯 模型配置:")
        self.logger.info(f"    基础模型: {self.config.base_model_key}")
        self.logger.info(f"    高分辨率模型: {self.config.highres_model_keys}")
        self.logger.info(f"    分辨率比: 1:{self.config.get_resolution_ratio()}")
        
        self.logger.info(f"\n  ⚙️ Phase 1 参数 (可直接在代码中修改):")
        self.logger.info(f"    物理窗口: {self.config.phase1.physical_window_size}°")
        self.logger.info(f"    步长: {self.config.phase1.physical_stride}°")
        self.logger.info(f"    K (原子数): {self.config.phase1.n_components}")
        self.logger.info(f"    λ (稀疏性): {self.config.phase1.alpha}")
        self.logger.info(f"    提取方法: direct (论文方法) ⭐")
        self.logger.info(f"    D₂正则化: {self.config.phase1.d2_regularization}")
        
        self.logger.info(f"\n  ⭐ v3.5: 直接使用原始模型数据，无插值")
        
        self.logger.info("="*60)
    
    def run_full_pipeline(self,
                          run_phase_minus1: bool = True,
                          run_phase0: bool = True,
                          run_phase1: bool = True,
                          run_phase1_5: bool = False,
                          run_phase2: bool = False,
                          run_phase3: bool = False,
                          params: List[str] = ['vs'],
                          target_model: str = 'fwea',
                          stitch_method: str = 'gaussian',
                          use_all_depths: bool = True) -> Dict:
        """
        运行完整SDL融合流程
        
        ⭐ v3.5: 移除 unified_ds_path 参数，直接使用原始模型
        
        Args:
            run_phase_minus1: 是否运行Phase -1 (加载模型)
            run_phase0: 是否运行Phase 0 (字典验证)
            run_phase1: 是否运行Phase 1 (2D Patch级验证)
            run_phase1_5: 是否运行Phase 1.5 (完整切片变换)
            run_phase2: 是否运行Phase 2 (3D融合)
            run_phase3: 是否运行Phase 3 (全域增强)
            params: 要处理的参数列表
            target_model: 目标高分辨率模型 ('cses' or 'ust')
            stitch_method: 拼接方法 ('gaussian' or 'average')
            use_all_depths: 是否对所有深度训练 (True=论文方法, False=快速模式)
            
        Returns:
            所有阶段的结果字典
        """
        start_time = time.time()
        
        self.logger.info("\n" + "="*70)
        self.logger.info("🚀 开始 SDL Fusion v3.5 完整流程")
        self.logger.info("="*70)
        
        self.all_results = {}
        
        # ========== Phase -1: 加载原始模型 ==========
        if run_phase_minus1:
            self.logger.info("\n" + "="*70)
            self.logger.info("📋 Phase -1: 加载原始模型")
            self.logger.info("="*70)
            
            self.all_results['phase_minus1'] = self.run_phase_minus1()
            
            # 可视化（使用原始模型数据）
            self.visualizer.plot_model_overview(self.model_loader, params[0])
        
        # 检查模型是否已加载
        if not self.model_loader.models:
            self.logger.error("❌ 需要先运行 Phase -1 加载模型")
            return self.all_results
        
        # ========== Phase 0: 字典表示能力验证 ==========
        if run_phase0:
            self.logger.info("\n" + "="*70)
            self.logger.info("📋 Phase 0: 字典表示能力验证")
            self.logger.info("="*70)
            
            self.all_results['phase0'] = {}
            for param in params:
                self.all_results['phase0'].update(
                    self.run_phase0(param)
                )
            
            # 可视化
            self.visualizer.plot_validation_results(self.all_results['phase0'], params[0])
        
        # ========== Phase 1: 2D水平切片变换 (Patch级验证) ==========
        if run_phase1:
            self.logger.info("\n" + "="*70)
            self.logger.info("📋 Phase 1: 2D水平切片变换 (Patch级验证)")
            if use_all_depths:
                self.logger.info("    ⭐ 论文方法: 对所有深度独立训练字典")
            else:
                self.logger.info("    ⚡ 快速模式: 只训练代表性深度")
            self.logger.info("="*70)
            
            for param in params:
                self.all_results['phase1'] = self.run_phase1(
                    param, target_model, use_all_depths=use_all_depths
                )
                
                # 可视化
                self.visualizer.plot_2d_transform_results(
                    self.all_results['phase1'], param, target_model
                )
        
        # ========== Phase 1.5: 完整2D切片变换 ==========
        if run_phase1_5:
            self.logger.info("\n" + "="*70)
            self.logger.info("📋 Phase 1.5: 完整2D切片变换")
            self.logger.info("="*70)
            
            # 需要先有 Phase 1 的字典
            if 'phase1' not in self.all_results:
                self.logger.warning("  ⚠️ Phase 1.5 需要 Phase 1 的训练字典")
                self.logger.info("  🔄 自动运行 Phase 1...")
                for param in params:
                    self.all_results['phase1'] = self.run_phase1(
                        param, target_model, use_all_depths=use_all_depths
                    )
            
            for param in params:
                self.all_results['phase1_5'] = self.run_phase1_5(
                    param, target_model, stitch_method
                )

                saved_path = self._save_phase1_5_enhanced_model(
                    results=self.all_results['phase1_5'],
                    param=param,
                    target_model=target_model,
                    stitch_method=stitch_method
                )
                if saved_path is not None:
                    self.all_results.setdefault('phase1_5_output_files', {})[param] = str(saved_path)
                    self.logger.info(f"  💾 Phase 1.5 增强模型已保存: {saved_path}")
                
                # 可视化
                self.visualizer.plot_full_slice_transform_results(
                    self.all_results['phase1_5'], param, target_model
                )
        
        # ========== 生成综合报告 ==========
        self.visualizer.plot_summary(self.all_results)
        
        # 完成
        elapsed = time.time() - start_time
        self.logger.info("\n" + "="*70)
        self.logger.info(f"✅ SDL Fusion v3.5 完成")
        self.logger.info(f"⏱️  总耗时: {elapsed/60:.1f} 分钟")
        self.logger.info(f"📁 输出目录: {self.output_dir}")
        self.logger.info("="*70)
        
        return self.all_results

    def _save_phase1_5_enhanced_model(self,
                                      results: Dict[float, FullSliceTransformResult],
                                      param: str,
                                      target_model: str,
                                      stitch_method: str) -> Optional[Path]:
        """
        将 Phase 1.5 结果保存为新的高分辨率（例如 0.25°）NetCDF 模型。
        """
        if not results:
            self.logger.warning("  ⚠️ Phase 1.5 结果为空，跳过保存")
            return None

        depths = sorted(results.keys())
        first = results[depths[0]]
        lat = np.asarray(first.lat_coords)
        lon = np.asarray(first.lon_coords)

        n_depth = len(depths)
        n_lat = len(lat)
        n_lon = len(lon)

        transformed_3d = np.full((n_depth, n_lat, n_lon), np.nan, dtype=np.float32)
        baseline_3d = np.full((n_depth, n_lat, n_lon), np.nan, dtype=np.float32)
        target_3d = np.full((n_depth, n_lat, n_lon), np.nan, dtype=np.float32)
        weight_3d = np.full((n_depth, n_lat, n_lon), np.nan, dtype=np.float32)

        for i, depth_km in enumerate(depths):
            r = results[depth_km]
            if r.transformed_slice.shape != (n_lat, n_lon):
                self.logger.warning(
                    f"  ⚠️ 深度 {depth_km}km 输出网格不一致: {r.transformed_slice.shape} != {(n_lat, n_lon)}，该层跳过"
                )
                continue

            transformed_3d[i] = np.asarray(r.transformed_slice, dtype=np.float32)
            baseline_3d[i] = np.asarray(r.baseline_slice, dtype=np.float32)
            target_3d[i] = np.asarray(r.target_slice, dtype=np.float32)
            weight_3d[i] = np.asarray(r.weight_map, dtype=np.float32)

        target_model_key = self.config.resolve_model_key(target_model)
        highres_res = float(self.config.get_highres_model_info(target_model_key).lat_resolution)

        output_ds = xr.Dataset(
            data_vars={
                param: (("depth", "lat", "lon"), transformed_3d),
                f"{param}_baseline": (("depth", "lat", "lon"), baseline_3d),
                f"{param}_target": (("depth", "lat", "lon"), target_3d),
                f"{param}_weight": (("depth", "lat", "lon"), weight_3d),
            },
            coords={
                "depth": np.asarray(depths, dtype=np.float32),
                "lat": lat,
                "lon": lon,
            },
            attrs={
                "title": "SDL enhanced full-domain model",
                "source": "EASTASIA-FWI/4_Fusion/4_3_SDL_Fusion_v4.py",
                "base_model": self.config.base_model_key,
                "target_model": target_model_key,
                "target_model_input": target_model,
                "parameter": param,
                "stitch_method": stitch_method,
                "grid_resolution_degree": highres_res,
                "description": "Coupled-dictionary SDL transformed model on high-resolution output grid",
            },
        )

        output_dir = self.base_config.dirs['results'] / 'SDL_Fusion_v3.5'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"SDL_enhanced_{self.config.base_model_key}_{target_model_key}_{param}_{timestamp}.nc"

        output_ds.to_netcdf(output_path)
        return output_path
    
    def run_grid_search(self,
                        param: str = 'vs',
                        target_model: str = 'fwea',
                        depths: Optional[List[float]] = None) -> Optional[GridSearchResult]:
        """
        Phase H: 运行超参数联合网格搜索 (v3.3.4 简化)
        
        执行 K × λ 联合网格搜索 (20 种组合，接近论文 40 种)
        
        Args:
            param: 参数名 ('vs' 或 'vp')
            target_model: 目标模型 ('cses' 或 'ust')
            depths: 测试深度，默认 [100, 200, 300]
            
        Returns:
            最优配置 GridSearchResult
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("🔍 Phase H: K × λ 联合网格搜索")
        self.logger.info("="*70)
        
        # ⭐ v3.5: 确保模型已加载
        if not self.model_loader.models:
            self.logger.info("  📊 先加载模型...")
            self.run_phase_minus1()
        
        # 初始化超参数搜索器
        self.hyperparam_searcher = HyperparameterSearcher(self, self.logger)
        
        # 设置搜索参数 (论文方法: 只搜索 K × λ)
        if depths is None:
            depths = [30, 60, 100]
        
        # 窗口和步长固定（由物理约束决定，不参与搜索）
        window = self.config.phase1.physical_window_size
        stride = self.config.phase1.physical_stride
        self.logger.info(f"  📐 固定窗口: {window}° | 固定步长: {stride}°")
        
        # K × λ 联合搜索: 5 × 4 = 20 种组合 (接近论文 40 种)
        K_values = [10, 15, 20, 25, 30]
        alpha_values = [0.05, 0.1, 0.15, 0.2]
        self.logger.info(f"  📋 搜索网格: K={K_values} × λ={alpha_values}")
        self.logger.info(f"  📋 总组合数: {len(K_values) * len(alpha_values)} 种")
        
        # 执行搜索 (论文方法: 只搜索 K × λ)
        best_config = self.hyperparam_searcher.search(
            K_values=K_values,
            alpha_values=alpha_values,
            depths=depths,
            param=param,
            target_model=target_model
        )
        
        return best_config
    
    def apply_optimal_config(self) -> bool:
        """
        应用最优配置 (v3.3 新增)
        
        从 results/SDL_grid_search/optimal_config.json 加载并应用最优配置
        
        Returns:
            是否成功应用
        """
        if self.hyperparam_searcher is None:
            self.hyperparam_searcher = HyperparameterSearcher(self, self.logger)
        
        return self.hyperparam_searcher.apply_optimal_config()
    
    def run_phase_minus1(self) -> Dict:
        """
        ⭐ v3.5: Phase -1 简化为仅加载原始模型
        
        不再创建统一数据集，所有操作直接使用原始模型数据
        """
        self.logger.info("\n  📂 加载原始速度模型...")
        
        # 构建模型路径
        model_paths = {}
        for model_key, model_info in self.config.model_info.items():
            # 尝试多个候选路径（兼容服务器扁平目录 与 本地 models/processed/<model>/ 结构）
            candidates = [
                self.data_dir / model_info.filename,
                self.base_config.dirs['models'] / 'processed' / model_key / model_info.filename,
                self.base_config.dirs['models'] / 'processed' / model_info.filename,
            ]
            path = next((p for p in candidates if p.exists()), None)
            if path is not None:
                model_paths[model_key] = path
            else:
                self.logger.warning(f"    ⚠️ 模型文件不存在: {candidates[0]} (已尝试 {len(candidates)} 个候选路径)")
        
        if len(model_paths) < 2:
            raise FileNotFoundError("需要至少两个模型文件")
        
        # 加载模型
        self.model_loader.load_models(model_paths)
        
        # 打印模型边界和重叠信息
        self.logger.info("\n  📊 模型边界信息:")
        for model_key in model_paths.keys():
            bounds = self.model_loader.get_model_bounds(model_key)
            short_name = self.model_loader.short_name_map.get(model_key, model_key)
            self.logger.info(f"    {short_name}: 经度 {bounds['lon'][0]:.1f}°-{bounds['lon'][1]:.1f}°, "
                           f"纬度 {bounds['lat'][0]:.1f}°-{bounds['lat'][1]:.1f}°")
        
        # 计算重叠区域
        base_key = self.config.base_model_key
        for target_key in self.config.highres_model_keys:
            overlap = self.model_loader.get_overlap_region(base_key, target_key)
            if overlap['has_overlap']:
                base_short = self.model_loader.short_name_map.get(base_key, base_key)
                target_short = self.model_loader.short_name_map.get(target_key, target_key)
                self.logger.info(f"\n  🔗 {base_short} ∩ {target_short} 重叠区域:")
                self.logger.info(f"      经度: {overlap['lon_range'][0]:.1f}°-{overlap['lon_range'][1]:.1f}°")
                self.logger.info(f"      纬度: {overlap['lat_range'][0]:.1f}°-{overlap['lat_range'][1]:.1f}°")
                self.logger.info(f"      重叠比例: {overlap['overlap_ratio']*100:.1f}%")
        
        return {
            'models': list(model_paths.keys()),
            'n_models': len(model_paths)
        }
    
    def run_phase0(self, param: str) -> Dict[str, Dict[float, ValidationResult]]:
        """
        ⭐ v3.5: Phase 0 直接使用原始模型数据验证
        
        Args:
            param: 参数名 ('vs' or 'vp')
            
        Returns:
            {model_name: {depth: ValidationResult}}
        
        注意：
            使用各模型的原始分辨率进行验证，不再插值到统一网格
        """
        if not self.model_loader.models:
            raise ValueError("请先运行 Phase -1 加载模型")
        
        self.logger.info(f"\n  🔬 验证字典表示能力 ({param})...")
        self.logger.info("    ⭐ v3.5: 直接使用原始模型数据")
        
        results = {}
        
        # 验证各模型 - 使用原始分辨率
        models_to_validate = [
            ('SinoScope1.0', 'sino', '2022_SinoScope1.0'),
            ('FWEA23', 'fwea', '2024_FWEA23')
        ]
        
        # 计算总任务数
        total_tasks = len(models_to_validate) * len(self.config.phase0.validation_depths)
        pbar = tqdm(total=total_tasks, desc="    Phase 0 验证", 
                    unit="task", ncols=100, file=sys.stdout)
        
        for model_name, short_name, model_key in models_to_validate:
            # 获取模型
            model = self.model_loader.get_model(model_key)
            if model is None or param not in model.data_vars:
                self.logger.warning(f"    ⚠️ 模型 {model_key} 或参数 {param} 不存在，跳过")
                pbar.update(len(self.config.phase0.validation_depths))
                continue
            
            # 获取该模型的原始分辨率
            lat_res, lon_res, _ = self.model_loader.get_model_resolution(model_key)
            model_resolution = (lat_res + lon_res) / 2  # 平均分辨率
            
            results[model_name] = {}
            
            for depth_km in self.config.phase0.validation_depths:
                pbar.set_postfix_str(f"{model_name}@{depth_km}km")
                
                try:
                    # ⭐ v3.5: 直接从原始模型获取切片
                    slice_data, lat, lon = self.model_loader.get_slice_from_model(
                        model_key, param, depth_km
                    )
                    
                    # 验证 - 使用模型原始分辨率
                    result = self.validator.validate_model_at_depth(
                        velocity_slice=slice_data,
                        lat_coords=lat,
                        lon_coords=lon,
                        depth_km=depth_km,
                        model_name=model_name,
                        resolution=model_resolution
                    )
                    
                    results[model_name][depth_km] = result
                    
                except Exception as e:
                    self.logger.warning(f"    ⚠️ {model_name}@{depth_km}km 验证失败: {e}")
                
                pbar.update(1)
        
        pbar.close()
        return results
    
    def run_phase1(self, param: str, target_model: str,
                   use_all_depths: bool = True) -> Dict[float, TransformResult]:
        """
        Phase 1: 2D水平切片变换 - 每个深度独立训练字典
        
        ⭐ v3.3.5 更新: 论文方法要求每个深度层独立训练字典
        
        Args:
            param: 参数名 ('vs' or 'vp')
            target_model: 目标高分辨率模型简称
            use_all_depths: 是否对所有深度训练（True=论文方法，False=只用target_depths）
            
        Returns:
            {depth: TransformResult}
        """
        self.logger.info(f"\n  🔄 Phase 1: 2D水平切片变换 ({param})...")
        
        results = {}
        
        # ⭐ v3.4.2: 论文方法 - 只使用深度精确匹配的层
        # 论文: "the target model M₂ is taken from velocities of the F2019 model at the same depths"
        
        # ⭐ v3.5: 从 model_loader 获取模型
        base_model_key = self.config.base_model_key
        target_model_key_map = {
            'fwea': '2024_FWEA23',
            'fwea23': '2024_FWEA23',
            'sino': '2022_SinoScope1.0',
            'sinoscope': '2022_SinoScope1.0',
            'cses': '2024_CSES_VM1.0',
            'ust': '2022_USTClitho2.0'
        }
        highres_model_key = target_model_key_map.get(target_model.lower(), target_model)
        
        base_ds = self.model_loader.get_model(base_model_key)
        highres_ds = self.model_loader.get_model(highres_model_key)
        
        if base_ds is None or highres_ds is None:
            self.logger.error(f"    ❌ 无法获取模型: base={base_model_key}, highres={highres_model_key}")
            return {}
        
        base_depth_name = 'depth' if 'depth' in base_ds.coords else 'dep'
        highres_depth_name = 'depth' if 'depth' in highres_ds.coords else 'dep'
        base_depths = base_ds[base_depth_name].values
        highres_depths = highres_ds[highres_depth_name].values
        
        # 计算匹配的深度（支持非均匀深度轴）
        depth_tol = self.config.phase1.depth_match_tolerance_km
        matched_depths = self.config.get_matched_depths(base_depths, highres_depths, tolerance=depth_tol)

        # 深度间隔统计（不假设均匀分布）
        base_dz = np.abs(np.diff(base_depths)) if len(base_depths) > 1 else np.array([np.nan])
        highres_dz = np.abs(np.diff(highres_depths)) if len(highres_depths) > 1 else np.array([np.nan])

        base_model_name = self.config.get_base_model_info().name
        self.logger.info(
            f"    {base_model_name} 深度: {len(base_depths)} 层 "
            f"(dz min/med/max = {np.nanmin(base_dz):.2f}/{np.nanmedian(base_dz):.2f}/{np.nanmax(base_dz):.2f} km)"
        )
        self.logger.info(
            f"    {target_model.upper()} 深度: {len(highres_depths)} 层 "
            f"(dz min/med/max = {np.nanmin(highres_dz):.2f}/{np.nanmedian(highres_dz):.2f}/{np.nanmax(highres_dz):.2f} km)"
        )
        self.logger.info(f"    ⭐ 匹配深度: {len(matched_depths)} 层 (容差 = {depth_tol:.2f} km)")
        
        if use_all_depths:
            training_depths = matched_depths
            self.logger.info(f"    将训练所有匹配深度: {len(training_depths)} 层")
        else:
            # 快速模式：从匹配深度中选择代表性深度
            target_depths = self.config.phase1.target_depths
            training_depths = [d for d in target_depths if d in matched_depths]
            self.logger.info(f"    快速模式: 只训练 {len(training_depths)} 个匹配的代表性深度")
        
        # 预估时间
        est_time_per_depth = 70  # 秒/深度（基于之前的测试）
        est_total_time = len(training_depths) * est_time_per_depth / 3600
        self.logger.info(f"    预估总时间: {est_total_time:.1f} 小时")
        
        for depth_km in tqdm(training_depths, desc="    Phase 1 Training",
                             unit="depth", ncols=100, file=sys.stdout):
            try:
                # ⭐ v3.5: 使用 model_loader
                result = self.transformer_2d.run(
                    model_loader=self.model_loader,
                    param=param,
                    depth_km=depth_km,
                    target_model=target_model
                )
                results[depth_km] = result
            except Exception as e:
                self.logger.error(f"    ❌ 深度 {depth_km}km 变换失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 汇总
        self.logger.info(f"\n  📊 Phase 1 汇总:")
        self.logger.info(f"    训练深度: {len(results)}/{len(training_depths)}")
        n_passed = sum(1 for r in results.values() if r.passed)
        self.logger.info(f"    验收通过: {n_passed}/{len(results)}")
        
        return results
    
    def run_phase1_5(self, param: str, target_model: str,
                     stitch_method: str = 'gaussian') -> Dict[float, FullSliceTransformResult]:
        """
        Phase 1.5: 完整2D切片变换
        
        对USTClitho2.0的完整切片应用SDL变换，生成增强后的完整速度切片。
        对应论文 Figure 3 的实现。
        
        Args:
            param: 参数名 ('vs' or 'vp')
            target_model: 目标高分辨率模型简称 ('cses' or 'ust')
            stitch_method: 拼接方法 ('gaussian' or 'average')
            
        Returns:
            {depth: FullSliceTransformResult}
        """
        self.logger.info(f"\n  🔄 Phase 1.5: 完整切片变换 ({param})...")
        self.logger.info(f"    拼接方法: {stitch_method}")
        
        # 从 Phase 1 获取训练好的字典
        trained_dicts = self.transformer_2d.get_best_dictionaries()
        
        if not trained_dicts:
            self.logger.error("    ❌ 没有找到训练好的字典，请先运行 Phase 1")
            return {}
        
        trained_depth_set = set(trained_dicts.keys())
        self.logger.info(f"    可用字典深度: {len(trained_depth_set)} 层")
        
        # 将字典传递给完整切片变换器
        for depth_km, coupled_dict in trained_dicts.items():
            self.full_slice_transformer.set_trained_dict(depth_km, coupled_dict)
        
        results = {}
        
        # ⭐ v3.3.5: 只对有训练字典的深度进行变换（论文方法）
        # 如果 Phase 1 对所有深度训练了字典，这里就是所有深度
        # 如果 Phase 1 只训练了代表性深度，这里就只处理这些深度
        transform_depths = sorted(trained_depth_set)
        self.logger.info(f"    将变换 {len(transform_depths)} 层深度 (与 Phase 1 训练深度一致)")
        
        # 对每个有字典的深度进行完整切片变换
        n_skipped = 0
        for depth_km in tqdm(transform_depths, desc="    Phase 1.5 Transform", 
                             unit="depth", ncols=100, file=sys.stdout):
            # ⭐ v3.3.5: 使用精确匹配的字典（论文方法），不复用其他深度的字典
            coupled_dict = trained_dicts.get(depth_km)
            
            if coupled_dict is None:
                n_skipped += 1
                continue
            
            try:
                # ⭐ v3.5: 使用 model_loader 获取原始模型
                base_model_key = self.config.base_model_key
                
                # 模型键映射
                target_model_key_map = {
                    'fwea': '2024_FWEA23',
                    'fwea23': '2024_FWEA23',
                    'sino': '2022_SinoScope1.0',
                    'sinoscope': '2022_SinoScope1.0',
                    'cses': '2024_CSES_VM1.0',
                    'ust': '2022_USTClitho2.0'
                }
                highres_model_key = target_model_key_map.get(target_model.lower(), target_model)
                
                # ⭐ v3.5: 从 model_loader 获取原始模型
                base_ds = self.model_loader.get_model(base_model_key)
                highres_ds = self.model_loader.get_model(highres_model_key)
                
                if base_ds is None or highres_ds is None:
                    self.logger.error(f"    ❌ 无法获取原始模型: base={base_model_key}, highres={highres_model_key}")
                    continue
                
                # ⭐ v3.5: 直接变换方法（论文方法）
                result = self.full_slice_transformer.transform_full_slice_direct(
                    base_ds=base_ds,
                    highres_ds=highres_ds,
                    param=param,
                    depth_km=depth_km,
                    coupled_dict=coupled_dict,
                    target_model=target_model,
                    stitch_method=stitch_method
                )
                
                results[depth_km] = result
                
            except Exception as e:
                self.logger.error(f"    ❌ 深度 {depth_km}km 变换失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 打印汇总
        self.logger.info(f"\n  📊 Phase 1.5 汇总:")
        n_passed = sum(1 for r in results.values() if r.passed)
        self.logger.info(f"    完成深度: {len(results)}/{len(transform_depths)}")
        self.logger.info(f"    跳过深度: {n_skipped}")
        self.logger.info(f"    通过验收: {n_passed}/{len(results)}")
        
        if results:
            avg_sdl_error = np.mean([r.sdl_error for r in results.values() if not np.isnan(r.sdl_error)])
            avg_improvement = np.mean([r.relative_improvement for r in results.values() if not np.isnan(r.relative_improvement)])
            self.logger.info(f"    平均SDL误差: {avg_sdl_error:.2f}%")
            self.logger.info(f"    平均改善率: {avg_improvement:.1f}%")
        
        return results
    
    # ⭐ v3.4: 移除 run_preprocessing_comparison 方法（不再需要对比）


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """
    SDL Fusion v3.4 主函数
    
    支持三种运行模式:
    - 模式 A (默认): 直接使用代码中配置的参数
    - 模式 B: --grid-search 运行 K × λ 网格搜索
    - 模式 C: --use-optimal 使用已有最优配置
    """
    print("\n" + "="*70)
    print("🎯 EASTASIA-FWI SDL Fusion v3.4")
    print("   稀疏字典学习速度模型融合")
    print("   ⭐ v3.4: 直接提取方法（移除统一网格）")
    print("="*70)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='SDL Fusion v3.4 - 稀疏字典学习速度模型融合 (直接提取方法)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式:
  模式 A (默认): 直接使用代码中 Phase1Config 配置的参数
  模式 B:        运行 K × λ 联合网格搜索 (20 种组合)
  模式 C:        加载并使用已有的最优配置

⭐ v3.4 新增选项:
  --quick        快速模式：只训练 4 个代表性深度（而不是全部 101 层）
  (默认)         论文方法：对所有深度独立训练字典（耗时约 8 小时）

示例:
  python 4_3_SDL_Fusion_v3.4.py                    # 论文方法: 所有深度训练
  python 4_3_SDL_Fusion_v3.4.py --quick            # 快速模式: 4 个代表性深度
  python 4_3_SDL_Fusion_v3.4.py --grid-search      # 模式 B: K × λ 网格搜索
  python 4_3_SDL_Fusion_v3.4.py --use-optimal      # 模式 C: 使用最优配置
        """
    )
    
    parser.add_argument('--grid-search', 
                        action='store_true',
                        help='运行 K × λ 联合网格搜索 (20 种组合)')
    parser.add_argument('--use-optimal', 
                        action='store_true',
                        help='加载并使用 optimal_config.json 中的最优配置')
    parser.add_argument('--quick',
                        action='store_true',
                        help='快速模式: 只训练 4 个代表性深度 (默认: 论文方法, 所有深度)')
    parser.add_argument('--skip-phase0',
                        action='store_true',
                        help='跳过 Phase 0 字典验证')
    parser.add_argument('--skip-phase1',
                        action='store_true',
                        help='跳过 Phase 1 Patch级验证')
    parser.add_argument('--stitch',
                        type=str,
                        choices=['average', 'gaussian', 'feather', 'copy'],
                        default='average',
                        help='Phase 1.5 拼接方法 (默认: average). 新增选项: feather=边界平滑, copy=直接覆盖')
    parser.add_argument('--k',
                        type=int,
                        default=None,
                        help='覆盖 Phase1 n_components (字典原子数 K)')
    parser.add_argument('--stride',
                        type=float,
                        default=None,
                        help='覆盖 Phase1 physical_stride (度)')
    parser.add_argument('--cc-threshold',
                        type=float,
                        default=None,
                        help='覆盖 Phase1 min_cc_threshold')
    parser.add_argument('--depth-tol',
                        type=float,
                        default=None,
                        help='覆盖 Phase1 depth_match_tolerance_km (km)')
    
    args = parser.parse_args()
    
    try:
        # ========== 创建配置 ==========
        config = SDLConfigV31()
        
        # ⭐ 使用论文推荐的 D₂ 纯最小二乘方法（固定）
        config.phase1.d2_regularization = 0.0

        # ========== 命令行参数覆盖（用于快速实验） ==========
        if args.k is not None:
            config.phase1.n_components = args.k
        if args.stride is not None:
            config.phase1.physical_stride = args.stride
        if args.cc_threshold is not None:
            config.phase1.min_cc_threshold = args.cc_threshold
        if args.depth_tol is not None:
            config.phase1.depth_match_tolerance_km = args.depth_tol
        
        # ========== 在此处修改超参数 (模式 A) ==========
        # 如需调整参数，直接修改以下值：
        # config.phase1.physical_window_size = 6.0    # 物理窗口大小 (度)
        # config.phase1.physical_stride = 1.0         # 步长 (度)
        # config.phase1.n_components = 20             # K: 字典原子数
        # config.phase1.alpha = 0.1                   # λ: 稀疏性参数
        # ================================================
        
        # 创建融合器
        sdl_fusion = SDLFusionV33(config)
        
        # ========== 模式 B: 网格搜索 ==========
        if args.grid_search:
            print(f"\n🔍 模式 B: 运行 K × λ 联合网格搜索...")
            
            best_config = sdl_fusion.run_grid_search(
                param='vs',
                target_model='fwea',
                depths=[30, 60, 100]
            )
            
            if best_config:
                print(f"\n⭐ 最优配置: {best_config.config_name}")
                print(f"   评分: {best_config.score:.4f}")
                print(f"   改善率: {best_config.relative_improvement:.1f}%")
            
            print(f"\n✅ 网格搜索完成!")
            print(f"📁 结果目录: {sdl_fusion.base_config.dirs['results'] / 'SDL_grid_search'}")
        
        # ========== 模式 C: 使用最优配置 ==========
        elif args.use_optimal:
            print(f"\n⭐ 模式 C: 加载最优配置...")
            
            # 确定深度训练模式
            use_all_depths = not args.quick
            if use_all_depths:
                print(f"   📚 论文方法: 对所有深度独立训练（耗时约 8 小时）")
            else:
                print(f"   ⚡ 快速模式: 只训练 4 个代表性深度")
            
            if sdl_fusion.apply_optimal_config():
                # 运行完整流程
                results = sdl_fusion.run_full_pipeline(
                    run_phase_minus1=True,
                    run_phase0=not args.skip_phase0,
                    run_phase1=not args.skip_phase1,
                    run_phase1_5=True,
                    run_phase2=False,
                    run_phase3=False,
                    params=['vs'],
                    target_model='fwea',
                    stitch_method=args.stitch,
                    use_all_depths=use_all_depths
                )
                print(f"\n✅ SDL Fusion v3.4 完成 (使用最优配置)!")
            else:
                print(f"\n⚠️ 未找到最优配置，请先运行网格搜索:")
                print(f"   python 4_3_SDL_Fusion_v3.4.py --grid-search")
        
        # ========== 模式 A: 默认运行 ==========
        else:
            # 确定深度训练模式
            use_all_depths = not args.quick
            
            print(f"\n📋 模式 A: 使用代码中配置的参数")
            print(f"   窗口: {config.phase1.physical_window_size}°")
            print(f"   步长: {config.phase1.physical_stride}°")
            print(f"   K: {config.phase1.n_components}")
            print(f"   λ: {config.phase1.alpha}")
            print(f"   CC阈值: {config.phase1.min_cc_threshold}")
            print(f"   深度匹配容差: {config.phase1.depth_match_tolerance_km} km")
            print(f"   拼接方法: {args.stitch}")
            
            if use_all_depths:
                print(f"   📚 深度训练: 论文方法（所有深度独立训练，耗时约 8 小时）")
            else:
                print(f"   ⚡ 深度训练: 快速模式（只训练 4 个代表性深度）")
            
            results = sdl_fusion.run_full_pipeline(
                run_phase_minus1=True,
                run_phase0=not args.skip_phase0,
                run_phase1=not args.skip_phase1,
                run_phase1_5=True,
                run_phase2=False,
                run_phase3=False,
                params=['vs'],
                target_model='fwea',
                stitch_method=args.stitch,
                use_all_depths=use_all_depths
            )
            
            print(f"\n✅ SDL Fusion v3.4 完成!")
        
        print(f"📁 输出目录: {sdl_fusion.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断处理")
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("💡 v3.4 使用提示:")
    print("  模式 A (默认): python 4_3_SDL_Fusion_v3.4.py")
    print("                 直接修改代码中的 Phase1Config 参数")
    print("  模式 B (搜索): python 4_3_SDL_Fusion_v3.4.py --grid-search")
    print("  模式 C (最优): python 4_3_SDL_Fusion_v3.3.py --use-optimal")
    print("="*70)


if __name__ == "__main__":
    main()

