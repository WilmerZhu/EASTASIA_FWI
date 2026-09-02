"""
EASTASIA-FWI 单模型速度相聚类模块 (v7.6)
====================================================================================

科学目标：
- 刻画单模型内部速度相 / 构造分区（体素级 GMM）
- 切片/剖面/三维体叠绘 USGS Slab2 俯冲板片几何（定性对照，不做定量评分）
- 浅部 facies 切片叠绘 USGS 地质省 / CN 地块边界（仅定性对照，不做定量符合度）
- Moho 点数据仅用于 moho_4band 空间分带，不再参与符合度评价
- 不用于模型间代表性筛选（那是 2_2 CW-SSIM + 模型级聚类的职责）

核心流程：
1. 加载各模型原始 NetCDF（经纬度用原生范围；深度统一截到 1000 km）
2. 转为相对水平平均 1D 参考的 δlnV 特征（消除深度主趋势）
3. 深度分层 GMM，支持两种方案：
   - fixed_3band: 浅部(0-410) / 过渡带 / 下地幔
   - moho_4band: 地壳(z<Moho) / 岩石圈(Moho-410) / 过渡带 / 下地幔
4. 按零模型校正的划分重现性选 K（体元数 N 不是独立观测数，argmin(BIC)
   必贴 K 上界且随抽样量漂移；改为逐 K 做多次部分重抽样重拟合，取标签
   ARI，再减去等量单高斯零模型的同一指标，选超额重现率最大的 K。
   同时报告 argmin(BIC) 与 argmin(BIC_eff) 作为诊断上/下界）
5. 可视化：3×3 深度切片（第1排首格基岩地质底图 + 10/40 km facies + 地质线，
   第2–3排叠 Slab2 等深线）、剖面与三维体叠 Slab2 几何（均为定性对照）
6. 聚类剖面图按 δlnVp/δlnVs 特征自动标注 facies 推测解释名

作者：EASTASIA-FWI Team
日期：2026-09-02
版本：v7.6
"""

import sys
import warnings
import importlib.util
import logging
import time
import traceback
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

# 科学计算库
import numpy as np
import pandas as pd
import xarray as xr

# 机器学习库
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


# 数据处理
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig

# 设置警告和随机种子
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



# ==================== 可视化模块（已拆分）====================
# 三个可视化类与模块级绘图辅助函数已移至
# 5_Visualization/5_10_Clustering_visualization.py，此处按项目既有方式加载。
_VIZ_PY = project_root / '5_Visualization' / '5_10_Clustering_visualization.py'
if not _VIZ_PY.exists():
    raise FileNotFoundError(f'找不到聚类可视化模块: {_VIZ_PY}')
_viz_spec = importlib.util.spec_from_file_location(
    'eastasia_clustering_viz', _VIZ_PY
)
if _viz_spec is None or _viz_spec.loader is None:
    raise ImportError(f'无法加载聚类可视化模块: {_VIZ_PY}')
_viz_mod = importlib.util.module_from_spec(_viz_spec)
sys.modules['eastasia_clustering_viz'] = _viz_mod
_viz_spec.loader.exec_module(_viz_mod)

GeologyConcordanceEvaluator = _viz_mod.GeologyConcordanceEvaluator
EnhancedClusteringVisualizer = _viz_mod.EnhancedClusteringVisualizer
BasicClusteringVisualizer = _viz_mod.BasicClusteringVisualizer

# 模块级绘图辅助函数在此再导出，使 2_4_Facies_voting 等下游模块
# 仍可按 `mc.overlay_slab_on_section(...)` 的原有方式调用
apply_clustering_plot_style = _viz_mod.apply_clustering_plot_style
overlay_slab_on_section = _viz_mod.overlay_slab_on_section
perturbation_clim = _viz_mod.perturbation_clim
make_equal_map_subplots = _viz_mod.make_equal_map_subplots
generate_distinct_colors = _viz_mod.generate_distinct_colors
interpret_facies = _viz_mod.interpret_facies

@dataclass
class ClusteringConfig:
    """单模型速度相聚类配置类"""

    # ==================== 数据配置 ====================
    data: Dict[str, Any] = field(default_factory=lambda: {
        # 使用各模型原始 NetCDF（原生 lon/lat；深度另由 preprocessing.depth_range 截断）
        'netcdf_pattern': 'processed/{model_name}/{model_name}_original.nc',
        'models_base_dir': 'models',  # 相对 base_config.dirs['data']
        'target_models': [
            '2022_SinoScope1.0',
            '2024_EARA2024',
            '2024_FWEA23',
        ],
        'all_features': ['vp', 'vs', 'vsv', 'vsh', 'vpv', 'vph'],
        'default_features': ['vp', 'vs'],
    })

    # ==================== 预处理配置 ====================
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        # 统一截到 1000 km（含 SinoScope 原始数据更深部分）
        'depth_range': [0, 1000],
        # 扰动域：相对各深度水平平均 1D 参考，δlnV = ln(V/V_ref)
        'perturbation': {
            'enabled': True,
            'method': 'dln',  # 'dln' | 'relative'=(V-Vref)/Vref
            'reference': 'horizontal_mean',  # 与 2_1 一致
        },
        'normalization': {
            'enabled': True,
            'method': 'standard',
        },
        'sampling': {
            'enabled': True,
            'max_total_points': 300_000,
            'min_points_per_depth': 500,
        },
    })

    # ==================== 深度分层 GMM ====================
    # scheme:
    #   fixed_3band — 浅部(0-410) / 过渡带 / 下地幔
    #   moho_4band  — 地壳(z<Moho) / 岩石圈(Moho→410) / 过渡带 / 下地幔
    depth_stratified: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'scheme': 'fixed_3band',
        'moho_fallback_km': 35.0,  # Moho 无覆盖时回退
        'moho_clip_km': [10.0, 80.0],
        'offset_labels_across_bands': True,
        'schemes': {
            'fixed_3band': {
                'bands': [
                    {
                        'name': 'shallow',
                        'mask': 'depth_range',
                        'depth_range': [0, 410],
                        'min_clusters': 2,
                        'max_clusters': 10,
                    },
                    {
                        'name': 'transition_zone',
                        'mask': 'depth_range',
                        'depth_range': [410, 660],
                        'min_clusters': 2,
                        'max_clusters': 5,
                    },
                    {
                        'name': 'lower_mantle',
                        'mask': 'depth_range',
                        'depth_range': [660, 1000],
                        'min_clusters': 2,
                        'max_clusters': 4,
                    },
                ],
            },
            'moho_4band': {
                # 每带可选 'fixed_k'：基于构造可解释性先验直接固定 K，
                # 跳过 BIC 扫描（论文表述为"K 先验固定，BIC 拐点分析作参考"）；
                # 设为 None 时按 auto_gmm['k_selection'] 自动选 K
                'bands': [
                    {
                        'name': 'crust',
                        'mask': 'above_moho',
                        'depth_cap': 410,
                        'min_clusters': 2,
                        'max_clusters': 10,
                        'fixed_k': None,
                    },
                    {
                        'name': 'lithosphere',
                        'mask': 'moho_to_depth',
                        'depth_range': [None, 410],
                        'min_clusters': 2,
                        'max_clusters': 10,
                        'fixed_k': None,
                    },
                    {
                        'name': 'transition_zone',
                        'mask': 'depth_range',
                        'depth_range': [410, 660],
                        'min_clusters': 2,
                        'max_clusters': 10,
                        'fixed_k': None,
                    },
                    {
                        'name': 'lower_mantle',
                        'mask': 'depth_range',
                        'depth_range': [660, 1000],
                        'min_clusters': 2,
                        'max_clusters': 10,
                        'fixed_k': None,
                    },
                ],
            },
        },
    })

    # ==================== 自动GMM聚类配置（全深度回退路径） ====================
    auto_gmm: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'max_clusters': 12,
        'min_clusters': 2,
        # 选 K 判据：'bic_knee'（默认）| 'bic_min'
        #
        # ── 'bic_knee'：BIC 曲线拐点（max-distance-to-chord，无自由参数）──
        # 把 K 与 BIC 分别归一化到 [0,1]，在首末两点间连一条弦，取曲线偏离
        # 该弦最远的 K（Satopää et al. 2011 的 Kneedle 同族）。无阈值、确定。
        #
        # 稳健性（EARA2024 实测，2 种抽样量 × 2 个随机种子 × 3 个 K 上界 ×
        # 2 种拐点算法 = 12 种组合）：地壳带 K*=4、岩石圈带 K*=4、过渡带
        # K*=3 在全部组合下完全一致；同一批扫描的 argmin(BIC) 却在 5~12 之间
        # 跳变。下地幔带是例外，K* 随 K 上界在 5~9 漂移，须单独说明或固定 K。
        #
        # 代价：拐点位置依赖 K 上界 —— 该上界是分析者按构造可解释性给定的
        # 科学判断，须在方法学中明确声明。
        #
        # 为什么情形 (b) 是常态：体元数 N 不是独立观测数。层析模型经正则化
        # 平滑后空间自相关长度可达数度，BIC 的似然项 ∝ N、惩罚项 ∝ ln(N)，
        # 前者完全支配 → argmin(BIC) 必贴 K 上界，且同一模型换个抽样量就换
        # 个 K（实测 N=2k/20k/200k 分别给出 K=4/6/12，纯粹是 N 的函数）。
        # 理论上这是必然的：Cai, Campbell & Broderick (2021, ICML) 证明模型
        # 设定只要有任意小偏差，有限混合模型的组分数后验即发散，任何有限 K
        # 的后验概率随 N 趋于 0。故 BIC 只用于定位拐点，不用于断言"真实 K"。
        #
        # ── 'bic_min'：原始 argmin(BIC)，仅作诊断 ──
        # 体元级数据下必贴 K 上界（理由同上），不建议作为选 K 依据。
        #
        # ── 已评估并弃用的判据 ──
        # 'bic_eff'（有效样本量修正 BIC）与 'stability'（零模型校正的划分
        # 重现性）已于 2026-09-02 移除，实现存档于
        # backup/2_3_deprecated_k_selection_20260902.py，弃用理由见该文件抬头。
        'k_selection': 'bic_knee',
        # EM 随机重启次数：过少会使 BIC 曲线出现局部最优造成的锯齿，干扰拐点判断。
        # 残余锯齿由 find_optimal_clusters 的对数似然单调修复兜底。
        'n_init': 10,
        # 重抽样稳定性复验（论文附录证据，默认关闭）：对每个带用不同随机种子
        # 重新抽子样本 + 重新拟合（K 固定为已选定值），在全量体素上预测标签，
        # 与参考结果计算 ARI。ARI ≈ 1 即证明结果对拟合子采样不敏感。
        'stability_check': {
            'enabled': False,
            'seeds': [1234, 20240901],  # 每个种子一次独立复验
        },
        # K 扫描稳健性（论文主证据，默认关闭以免拖慢常规运行）：与其论证
        # "K* 是真实簇数"，不如证明结论在一段 K 区间内不变。复用 BIC 扫描已
        # 缓存的各 K 模型，边际开销仅为逐 K 的一次 predict。
        'k_robustness': {
            'enabled': True,
            'neutral_threshold': 0.01,  # 快/慢三分类的中性带半宽（物理 δlnVs）
        },
        # 簇编号排序：GMM 组分索引由 EM 初始化随机决定，未排序时 C0 在不同
        # 模型/深度带间含义不同，配色与跨模型对应都无法直接进行。按簇均值
        # δlnVs 升序重排后 C0 = 最慢、C_{K-1} = 最快，编号与配色即可跨模型比较。
        # 标准化是单调仿射变换，故在标准化域排序等价于在物理 δlnVs 上排序。
        'cluster_order': 'dlnvs_ascending',  # 'dlnvs_ascending'|'dlnvs_descending'|None
        # ── HMRF-GMM 空间正则化 ──
        # 逐体元独立分类忽略了相邻体元的相关性，而层析模型本身经反演正则化
        # 平滑，实际分辨长度远大于网格间距，相邻体元并非独立观测。结果是标签
        # 场出现椒盐噪声：后验接近 0.5/0.5 的体元被噪声推向任意一侧。
        #
        # 采用隐马尔可夫随机场 GMM（Zhang, Brady & Smith 2001, IEEE TMI）：
        # 标签场服从 Potts 先验，能量 U = -log N(x|μ_k,Σ_k) + β·Σ_j w_j·1[l_j≠l_k]。
        # 用均场近似而非硬 ICM，以保留软后验（下游概率图与投票均需要）。
        # 邻域权重 w_j 按邻点物理距离反比加权：经度间距随纬度收缩、深度层间距
        # 与水平间距量级不同，若等权则等价于在物理空间做各向异性平滑。
        #
        # ⚠️ 跨模型投票前提：三个模型必须用同一组 (beta, 邻域定义)，否则有效
        # 平滑量不同，一致性不可比。
        'hmrf': {
            'enabled': True,
            'beta': 1.0,        # Potts 耦合强度（相对于对数似然的量纲）
            'max_iter': 8,      # 均场外迭代上限
            'tol': 1e-3,        # 标签变动比例收敛阈值
            'update_params': True,  # 是否在均场迭代中同步更新 μ/Σ/π（完整 EM）
        },
        'max_iter': 300,
        'covariance_type': 'full',  # 保留 Vp–Vs 相关
        'random_state': RANDOM_SEED,
        'probability_threshold': 0.6,
        'reuse_bic_fit': True,  # BIC 搜索时缓存最优 GMM，避免二次拟合
    })

    # ==================== K-means对比（默认关闭，科学收益低） ====================
    kmeans_comparison: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'n_clusters': [2, 4, 6, 8, 10, 12],
        'use_minibatch': True,
        'batch_size': 500_000,
    })

    # ==================== 结构参考数据（仅定性叠图 + moho_4band 分带） ====================
    geology_concordance: Dict[str, Any] = field(default_factory=lambda: {
        # Moho 仅供 moho_4band 空间分带
        'moho_file': '莫霍面深度数据.txt',
        # Slab2：与 5_1_Basemap 同源（5_Visualization/data_EastAsia/Slab2）
        'slab': {
            'enabled': True,
            'vis_data_subdir': '5_Visualization/data_EastAsia/Slab2',
            'distribute_subdir': 'Slab2Distribute_Mar2018',
            # 东亚及邻区板片（depth grid）
            'regions': [
                'kur', 'ryu', 'izu', 'man', 'phi', 'sum', 'sul',
                'him', 'hin', 'mak', 'cot', 'hal', 'png', 'sol',
            ],
            'grd_glob': '{region}_slab2_dep_*.grd',
            'thk_glob': '{region}_slab2_thk_*.grd',
            # Slab2 z 为负值；叠图用正深度 km
            'use_absolute_depth': True,
            # 三维体对比可视化（dep+thk → 体掩膜）
            'volume_viz': {
                'enabled': True,
                'slice_depths_km': [100.0, 200.0, 400.0],
                'section_latitudes': [20.0, 30.0, 40.0],
            },
        },
        # 地质省 / CN 地块边界线：仅供切片图叠绘
        'surface_units': {
            'geology_subdir': 'geology',
            # 叠图：优先用 5_9 过滤后的线划缓存（三模型公共区）
            'province_line_files': [
                'prv1ec_nosouth_rim_v3_80_150_10_55.gmt',
                'prv3bl_nosouth_north_rim_v2_80_150_10_55.gmt',
            ],
            'block_line_files': [
                'CN-block-L2.gmt',
                'CN-block-L1-deduced.gmt',
                'CN-block-L1.gmt',
            ],
            # 与 5_9 / 三模型 original.nc 公共范围一致；None 则运行时求交
            'overlap_region': [80.0, 150.0, 10.0, 55.0],
            'overlay_depths_km': [40.0, 60.0, 80.0],
        },
    })

    # ==================== 可视化配置 ====================
    visualization: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'plot_types': {
            'depth_slices': True,
            'vertical_sections': True,
            '3d_scatter': False,
            'bic_analysis': True,
            'probability_distribution': True,
            'confidence_analysis': True,
            'cluster_uncertainty': False,
            'gmm_vs_kmeans': False,
            'performance_comparison': False,
            'cluster_profiles': True,
            'feature_distributions': True,
            'cluster_centers': True,
            'velocity_statistics': False,
            'spatial_distribution': False,
            # 三维聚类体 vs Slab2(dep+thk) 体掩膜对比图
            'slab_volume_3d': True,
        },
        # 兼容旧字段；实际 3×3 布局见 slice_layout
        'slice_depths': [40, 60, 100, 150, 200, 500, 600, 700],
        # 3×3：第1排首格=基岩地质底图，其余为浅部 facies+地质线；第2–3排叠 Slab2
        'slice_layout': {
            'n_rows': 3,
            'n_cols': 3,
            # 首格画 USGS 拼合地质图（仿 5_9，无图例）；其后为 10/40 km facies
            'row1_leading_geology_map': True,
            'geology_depths_km': [10.0, 40.0],
            'slab_depths_km': [100.0, 150.0, 200.0, 500.0, 600.0, 700.0],
        },
        'section_positions': {
            'latitudes': [20, 30, 40],
            'longitudes': [100, 110, 120, 130],
        },
        'figsize': {
            'slices': (18, 15),
            'sections': (14, 10),
            '3d': (12, 10),
            'profiles': (16, 10),
            'distributions': (18, 12),
            'metrics': (14, 10),
            'comparison': (16, 12),
            'bic': (14, 8),
            'concordance': (14, 10),
        },
        'dpi': 300,
        'save_formats': ['jpg'],
        'n_colors': 30,
        'alpha': 0.7,
        # 统一字号（所有聚类可视化共用）
        'fonts': {
            'base': 16,
            'tick': 15,
            'label': 17,
            'title': 19,
            'suptitle': 24,
            'legend': 15,
            'annotation': 14,
            'clabel': 13,
        },
    })

    # ==================== 输出配置 ====================
    output: Dict[str, Any] = field(default_factory=lambda: {
        'save_results': True,
        'save_labels': True,
        'save_probabilities': True,
        'save_3d_cube': True,
        'save_cluster_stats': True,
        'result_formats': ['npz', 'nc'],
        'create_report': True,
    })

    # ==================== 性能配置 ====================
    performance: Dict[str, Any] = field(default_factory=lambda: {
        'n_jobs': -1,
        'verbose': True,
        'chunk_size': 990_000,
    })


class NetCDF3DLoader:
    """三维NetCDF速度模型数据加载器（支持全部特征）"""
    
    def __init__(self, config: ClusteringConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.base_config = BaseConfig()
    
    def load_3d_model(
        self, 
        model_name: str,
        features_to_use: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        加载三维速度模型数据立方体
        
        Parameters:
        -----------
        model_name : str
            模型名称
        features_to_use : List[str], optional
            要加载的特征列表，默认使用配置中的default_features
        
        Returns:
        --------
        data_cube : np.ndarray
            形状为 (n_lats, n_lons, n_depths, n_features)
        metadata : Dict
            元数据信息
        """
        try:
            self.logger.info(f"📂 加载三维模型: {model_name}")
            
            # 构建NetCDF文件路径（对齐 1_5 / 2_1）
            models_base = self.config.data.get('models_base_dir', 'models')
            nc_path = (
                self.base_config.dirs['data']
                / models_base
                / self.config.data['netcdf_pattern'].format(model_name=model_name)
            )

            if not nc_path.exists():
                raise FileNotFoundError(f"NetCDF文件不存在: {nc_path}")

            self.logger.info(f"  文件: {nc_path}")

            # 打开NetCDF
            ds = xr.open_dataset(nc_path)
            
            # 提取坐标（经纬度保留模型原生范围，不做研究区裁剪）
            lons = ds['longitude'].values
            lats = ds['latitude'].values
            depths = ds['depth'].values

            # 深度范围：默认使用模型全部深度；可按配置可选裁剪
            depth_range = self.config.preprocessing.get('depth_range')
            if depth_range is None:
                depth_mask = np.ones(len(depths), dtype=bool)
                depth_min = float(depths.min())
                depth_max = float(depths.max())
            else:
                depth_min, depth_max = depth_range
                if depth_max is None:
                    depth_max = float(depths.max())
                depth_mask = (depths >= depth_min) & (depths <= depth_max)
            depths_filtered = depths[depth_mask]

            self.logger.info(
                f"  原生空间范围: lon [{float(lons.min()):.2f}, {float(lons.max()):.2f}], "
                f"lat [{float(lats.min()):.2f}, {float(lats.max()):.2f}]"
            )
            self.logger.info(f"  深度范围: {depth_min:.1f}-{depth_max:.1f} km")
            self.logger.info(f"  网格尺寸: {len(lats)}×{len(lons)}×{len(depths_filtered)}")
            
            # 确定要加载的特征
            if features_to_use is None:
                features_to_use = self.config.data['default_features']
            
            features_to_load = self._get_available_features(ds, features_to_use)
            
            # 创建数据立方体
            n_lats = len(lats)
            n_lons = len(lons)
            n_depths = len(depths_filtered)
            n_features = len(features_to_load)
            
            # 初始化数据立方体
            data_cube = np.full((n_lats, n_lons, n_depths, n_features), np.nan)
            
            # 加载数据
            self.logger.info(f"  加载特征: {', '.join(features_to_load)}")
            
            for i, feature in enumerate(tqdm(features_to_load, desc="  加载特征")):
                if feature in ds:
                    data_3d = ds[feature].values
                    # 确保维度顺序正确 (lat, lon, depth)
                    if data_3d.shape[0] == len(lons) and data_3d.shape[1] == len(lats):
                        data_3d = data_3d.transpose(1, 0, 2)
                    # 应用深度mask
                    data_3d = data_3d[:, :, depth_mask]
                    # 替换无效值
                    data_3d[data_3d == 9999.0] = np.nan
                    data_cube[:, :, :, i] = data_3d
            
            ds.close()
            
            # 构建元数据
            metadata = {
                'model_name': model_name,
                'lats': lats,
                'lons': lons,
                'depths': depths_filtered,
                'features': features_to_load,
                'dims': (n_lats, n_lons, n_depths),
                'spatial_range': {
                    'lat': (float(lats.min()), float(lats.max())),
                    'lon': (float(lons.min()), float(lons.max())),
                    'depth': (float(depths_filtered.min()), float(depths_filtered.max()))
                }
            }
            
            # 统计有效数据点
            valid_points = np.sum(~np.isnan(data_cube[:, :, :, 0]))
            total_points = n_lats * n_lons * n_depths
            
            self.logger.info(f"✅ 数据立方体加载完成")
            self.logger.info(f"  形状: {data_cube.shape}")
            self.logger.info(f"  有效点: {valid_points:,}/{total_points:,} ({valid_points/total_points:.1%})")
            
            return data_cube, metadata
            
        except Exception as e:
            self.logger.error(f"❌ 加载失败: {e}")
            raise
    
    def _get_available_features(self, ds: xr.Dataset, requested_features: List[str]) -> List[str]:
        """获取数据集中可用的特征"""
        available = []
        all_possible = self.config.data['all_features']
        
        for feature in requested_features:
            if feature in all_possible and feature in ds:
                available.append(feature)
            else:
                self.logger.warning(f"  ⚠️ 特征 '{feature}' 在数据集中不可用")
        
        if not available:
            raise ValueError(f"没有可用的特征！请求的特征: {requested_features}")
        
        return available


class DataCube3DProcessor:
    """三维数据立方体预处理器（δlnV 扰动域 + 向量化提取）"""

    def __init__(self, config: ClusteringConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scaler: Optional[Any] = None
        self.reference_1d: Dict[str, np.ndarray] = {}
        # 未标准化的 3D 特征立方体 (lat, lon, depth, feat)，供剖面叠图
        self.last_feature_cube: Optional[np.ndarray] = None

    def prepare_for_clustering(
        self,
        data_cube: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        准备三维聚类数据

        Returns:
            X: 特征矩阵 (n_points, n_features)，已在扰动域并可选标准化
            spatial_indices: (n_points, 3) → [lat_idx, lon_idx, depth_idx]
            prep_info: 预处理元信息
        """
        self.logger.info("🔧 准备三维聚类数据...")

        prep_info: Dict[str, Any] = {
            'original_shape': data_cube.shape,
            'preprocessing_steps': [],
            'feature_space': 'absolute',
            'feature_names': list(metadata.get('features', [])),
        }

        working_cube = data_cube
        pert_cfg = self.config.preprocessing.get('perturbation', {})
        if pert_cfg.get('enabled', False):
            working_cube, ref_1d = self._to_perturbation(
                data_cube, metadata['depths'], metadata['features'], pert_cfg
            )
            self.reference_1d = ref_1d
            method = pert_cfg.get('method', 'dln')
            prep_info['feature_space'] = f'perturbation_{method}'
            prep_info['feature_names'] = [
                f'dln_{f}' if method == 'dln' else f'drel_{f}'
                for f in metadata['features']
            ]
            prep_info['preprocessing_steps'].append(
                f"扰动转换({method}, ref={pert_cfg.get('reference', 'horizontal_mean')})"
            )
            self.logger.info(f"  特征空间: {prep_info['feature_space']}")

        # 缓存未标准化的 3D 特征立方体（供剖面叠图：δlnVp/δlnVs）
        self.last_feature_cube = np.asarray(working_cube, dtype=np.float32)

        # 向量化提取全部有效点（标签重建必须覆盖全网格，采样只用于 BIC/拟合）
        X, spatial_indices = self._extract_valid_points(working_cube)
        prep_info['n_valid_points'] = len(X)
        self.logger.info(f"  有效数据点: {len(X):,}")

        # 标准化（在扰动域上做，避免深度趋势主导）
        if self.config.preprocessing['normalization']['enabled']:
            X = self._normalize_features(X)
            prep_info['preprocessing_steps'].append('特征标准化')

        prep_info['final_shape'] = X.shape
        prep_info['fit_max_points'] = self.config.preprocessing['sampling'].get(
            'max_total_points', 10_000_000
        )
        self.logger.info(
            f"✅ 数据准备完成: {prep_info['original_shape']} → {prep_info['final_shape']}"
        )
        return X, spatial_indices, prep_info

    def _to_perturbation(
        self,
        data_cube: np.ndarray,
        depths: np.ndarray,
        features: List[str],
        pert_cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        将绝对速度转为相对水平平均 1D 参考的扰动

        δlnV = ln(V / V_ref)  或  δV/V = (V - V_ref) / V_ref
        """
        method = pert_cfg.get('method', 'dln')
        n_features = data_cube.shape[-1]
        out = np.full_like(data_cube, np.nan, dtype=np.float64)
        ref_1d: Dict[str, np.ndarray] = {'depth': np.asarray(depths, dtype=float)}

        for f_idx in range(n_features):
            slab = data_cube[:, :, :, f_idx]
            v_ref = np.nanmean(slab, axis=(0, 1))  # (n_depths,)
            v_ref = np.where(np.isfinite(v_ref) & (v_ref > 0), v_ref, np.nan)
            feat_name = features[f_idx] if f_idx < len(features) else f'f{f_idx}'
            ref_1d[feat_name] = v_ref

            with np.errstate(divide='ignore', invalid='ignore'):
                if method == 'relative':
                    pert = (slab - v_ref[np.newaxis, np.newaxis, :]) / v_ref[
                        np.newaxis, np.newaxis, :
                    ]
                else:
                    pert = np.log(slab / v_ref[np.newaxis, np.newaxis, :])
                pert[~np.isfinite(pert)] = np.nan
            out[:, :, :, f_idx] = pert

            finite = pert[np.isfinite(pert)]
            if finite.size:
                self.logger.info(
                    f"    {feat_name}: δ 范围 "
                    f"[{np.nanpercentile(finite, 1):.3f}, {np.nanpercentile(finite, 99):.3f}]"
                )

        return out.astype(np.float32), ref_1d

    def _extract_valid_points(
        self, data_cube: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """向量化提取有效数据点"""
        valid_mask = ~np.any(np.isnan(data_cube), axis=-1)
        spatial_indices = np.argwhere(valid_mask).astype(np.int32)
        X = data_cube[valid_mask].astype(np.float64)
        return X, spatial_indices

    def _apply_3d_sampling(
        self,
        X: np.ndarray,
        spatial_indices: np.ndarray,
        metadata: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """按深度层分层随机采样"""
        max_points = self.config.preprocessing['sampling']['max_total_points']
        if len(X) <= max_points:
            return X, spatial_indices

        unique_depths = np.unique(spatial_indices[:, 2])
        n_depths = max(len(unique_depths), 1)
        samples_per_depth = max(
            max_points // n_depths,
            self.config.preprocessing['sampling']['min_points_per_depth'],
        )

        sampled_indices: List[int] = []
        rng = np.random.default_rng(RANDOM_SEED)
        for depth_idx in unique_depths:
            depth_indices = np.where(spatial_indices[:, 2] == depth_idx)[0]
            n_samples = min(samples_per_depth, len(depth_indices))
            if n_samples > 0:
                selected = rng.choice(depth_indices, n_samples, replace=False)
                sampled_indices.extend(selected.tolist())

        sampled_indices_arr = np.asarray(sampled_indices, dtype=np.int64)
        if len(sampled_indices_arr) > max_points:
            sampled_indices_arr = rng.choice(
                sampled_indices_arr, max_points, replace=False
            )
        return X[sampled_indices_arr], spatial_indices[sampled_indices_arr]

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """特征标准化"""
        method = self.config.preprocessing['normalization']['method']
        self.scaler = StandardScaler() if method == 'standard' else RobustScaler()
        return self.scaler.fit_transform(X)

class GMMAutoClusteringOptimizer:
    """GMM 自动聚类优化器（BIC 选 K + 深度分层 + 模型缓存）"""

    def __init__(self, config: ClusteringConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._cached_gmms: Dict[int, GaussianMixture] = {}
        # 由 pipeline 注入，用于 moho_4band 空间分带
        self.concordance_eval: Optional['GeologyConcordanceEvaluator'] = None
        # 由 pipeline 注入（processor.scaler.inverse_transform），把标准化簇均值
        # 还原为物理 δlnV，供 K 扫描的快/慢三分类使用
        self.feature_inverse_transform: Optional[Any] = None

    @staticmethod
    def _repair_loglik_monotone(
        log_likelihoods: List[float],
        n_params: List[float],
        n_fit: int,
    ) -> Tuple[List[float], List[float], List[int]]:
        """
        对数似然单调修复，并据此重算 BIC。

        K+1 个组分总能复现 K 个组分的解（令多出的组分权重趋于 0），故全局最优
        的对数似然必随 K 单调不减。实测曲线出现下降只可能是该 K 的 EM 收敛到了
        局部最优。若不修复，这类假象会在 BIC 曲线上造成上翘，使拐点判据失稳。

        修复取运行最大值 ℓ*(K) = max_{k≤K} ℓ(k)，再由 BIC = -2ℓ* + p·ln(N)
        重算。这是保守处理：被修复的 K 其真实最优只会更好，不会更差。

        Args:
            log_likelihoods: 各候选 K 的总对数似然（-inf 表示拟合失败）
            n_params: 各候选 K 的自由参数数
            n_fit: 拟合样本量

        Returns:
            (修复后对数似然, 重算后 BIC, 被修复的索引列表)
        """
        ll = np.asarray(log_likelihoods, dtype=float)
        p = np.asarray(n_params, dtype=float)
        finite = np.isfinite(ll)
        if not np.any(finite):
            return list(ll), [float('inf')] * len(ll), []

        ll_mono = ll.copy()
        running = -np.inf
        repaired: List[int] = []
        for i in range(len(ll_mono)):
            if not finite[i]:
                continue
            if ll_mono[i] < running:
                ll_mono[i] = running
                repaired.append(i)
            else:
                running = ll_mono[i]

        log_n = float(np.log(max(n_fit, 2)))
        bics = [
            float(-2.0 * ll_mono[i] + p[i] * log_n)
            if finite[i] and np.isfinite(p[i])
            else float('inf')
            for i in range(len(ll_mono))
        ]
        return list(ll_mono), bics, repaired

    @staticmethod
    def _knee_select(
        bics: List[float],
        valid_indices: List[int],
    ) -> Tuple[Optional[int], List[float]]:
        """
        BIC 曲线拐点判据（max-distance-to-chord，无自由参数）。

        把 K 与 BIC 分别线性归一化到 [0,1]，在首末两点间连一条弦，取曲线
        低于该弦最多的 K（Satopää et al. 2011 的 Kneedle 同族）。判据不含
        任何阈值，输出确定；代价是拐点位置依赖 K 的搜索上界——该上界由
        分析者按构造可解释性给定，须在方法学中声明。

        Args:
            bics: 各候选 K 的 BIC（可能含 inf 表示拟合失败）
            valid_indices: BIC 有限的索引列表（升序）

        Returns:
            (拐点索引或 None, 各索引到弦的归一化距离(nan 表示不适用))
        """
        distances: List[float] = [float('nan')] * len(bics)
        if len(valid_indices) < 3:
            return None, distances

        idx = np.asarray(valid_indices, dtype=int)
        b = np.asarray([bics[i] for i in valid_indices], dtype=float)
        span = float(b.max() - b.min())
        if span <= 0:
            return None, distances

        # K 轴按候选序号归一化（候选 K 等距，与直接用 K 值等价）
        x = (idx - idx[0]) / float(idx[-1] - idx[0])
        y = (b - b.min()) / span
        chord = y[0] + (y[-1] - y[0]) * x
        d = chord - y                      # 曲线低于弦的幅度，越大越像拐点
        for j, i in enumerate(valid_indices):
            distances[i] = float(d[j])
        return int(idx[int(np.argmax(d))]), distances

    def find_optimal_clusters(
        self,
        X: np.ndarray,
        max_clusters: Optional[int] = None,
        min_clusters: int = 2,
        band_name: str = 'full',
        spatial_indices: Optional[np.ndarray] = None,
        dims: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        全区间扫描 K 并选最优，缓存各 K 的 GMM。

        扫描后先对对数似然做单调修复（消除 EM 局部最优造成的 BIC 上翘），
        再按 auto_gmm['k_selection'] 选 K：
        - 'bic_knee': 归一化 BIC 曲线偏离首末弦最远处（默认，无阈值）
        - 'bic_min' : 原始 argmin(BIC)（体元级数据下几乎必贴 K 上界）

        Args:
            X: 特征矩阵
            max_clusters: K 上界（同时决定拐点位置，需在方法学中声明）
            min_clusters: K 下界
            band_name: 深度带名称（仅用于日志/结果标注）
            spatial_indices: 该带各点空间索引，供 N_eff 估计（缺省则退回拐点判据）
            dims: 完整立方体维度

        Returns:
            (optimal_clusters, bic_analysis)
        """
        gmm_config = self.config.auto_gmm
        self._cached_gmms = {}

        if max_clusters is None:
            max_clusters = min(gmm_config['max_clusters'], max(2, len(X) // 50))

        max_clusters = int(min(max(max_clusters, min_clusters), len(X)))
        min_clusters = int(max(2, min_clusters))
        if min_clusters > max_clusters:
            min_clusters = max_clusters

        n_clusters_range = np.arange(min_clusters, max_clusters + 1)

        bics: List[float] = []
        aics: List[float] = []
        log_likelihoods: List[float] = []
        n_params_list: List[float] = []

        # BIC/拟合可用子样本加速；最终 predict 仍在全量 X 上
        X_fit = self._subsample_for_fit(X, spatial_depth_indices=None)
        self.logger.info(
            f"  🔍 [{band_name}] BIC 搜索最优 K ({min_clusters}-{max_clusters}), "
            f"N_fit={len(X_fit):,}/{len(X):,}"
        )

        for n in tqdm(n_clusters_range, desc=f"  BIC[{band_name}]"):
            try:
                gmm = GaussianMixture(
                    n_components=int(n),
                    covariance_type=gmm_config['covariance_type'],
                    n_init=gmm_config['n_init'],
                    max_iter=gmm_config['max_iter'],
                    random_state=gmm_config['random_state'],
                    reg_covar=1e-6,
                )
                gmm.fit(X_fit)
                self._cached_gmms[int(n)] = gmm

                # 仅计算选 K 判据 BIC（AIC/logL 顺带记录，代价为零）；
                # silhouette/DB/CH 等诊断指标只对最终模型算一次，避免逐 K 的高开销
                bic_n = float(gmm.bic(X_fit))
                ll_total = float(gmm.score(X_fit) * len(X_fit))
                bics.append(bic_n)
                aics.append(float(gmm.aic(X_fit)))
                log_likelihoods.append(ll_total)
                # 由 BIC 定义反解自由参数数（避免依赖 sklearn 私有 API）
                n_params_list.append(
                    (bic_n + 2.0 * ll_total) / float(np.log(len(X_fit)))
                )

            except Exception as e:
                self.logger.warning(f"    n={n} 失败: {e}")
                bics.append(float('inf'))
                aics.append(float('inf'))
                log_likelihoods.append(-float('inf'))
                n_params_list.append(float('nan'))

        valid_indices = [i for i, bic in enumerate(bics) if np.isfinite(bic)]
        if not valid_indices:
            raise ValueError(f"[{band_name}] 所有聚类尝试都失败了")

        # ── 对数似然单调修复：消除 EM 局部最优造成的 BIC 上翘 ──────────
        bics_raw = list(bics)
        log_likelihoods, bics, repaired_idx = self._repair_loglik_monotone(
            log_likelihoods, n_params_list, len(X_fit)
        )
        if repaired_idx:
            repaired_ks = [int(n_clusters_range[i]) for i in repaired_idx]
            self.logger.warning(
                f"    ⚠️ [{band_name}] K={repaired_ks} 的 EM 收敛劣于更小的 K，"
                f"已按单调性修复其似然；如频繁出现可增大 auto_gmm['n_init']"
            )

        argmin_idx = valid_indices[int(np.argmin([bics[i] for i in valid_indices]))]
        argmin_clusters = int(n_clusters_range[argmin_idx])
        k_upper = int(n_clusters_range[valid_indices[-1]])

        rule = str(gmm_config.get('k_selection', 'bic_knee'))
        knee_distances: List[float] = [float('nan')] * len(bics)
        knee_idx: Optional[int] = None
        optimal_idx: Optional[int] = None
        selection_rule = rule

        # ── 判据 1：BIC 拐点（默认）──────────────────────────────────
        # 体元数 N 不是独立观测数：层析模型经正则化平滑，δlnV 的空间自相关
        # 长度可达数度。BIC 的似然项 ∝ N、惩罚项 ∝ p·ln(N)，前者完全支配，
        # argmin(BIC) 因而不可用——实测同一带换个随机种子即从 K=5 跳到 K=12。
        # 拐点则对抽样量、种子、K 上界均稳定（见方法学的稳健性检验）。
        #
        # 注意不要因"argmin 落在上界之内"就改用 argmin：曲线走完绝大部分
        # 降幅后往往在噪声级来回摆动（实测 SinoScope 岩石圈带 K≥3 的归一化
        # BIC 全在 0.00–0.04 之间），此时的"内部极小"纯属噪声。真正存在极小
        # 时拐点会自动逼近它（实测两带拐点与 argmin 完全重合），无需特判。
        if rule == 'bic_knee':
            knee_idx, knee_distances = self._knee_select(bics, valid_indices)
            if knee_idx is not None:
                optimal_idx = knee_idx
                selection_rule = 'bic_knee'

        # ── 判据 2：argmin(BIC)（拐点不可用时的回退）────────────
        if optimal_idx is None:
            optimal_idx = argmin_idx
            selection_rule = 'bic_min'
        optimal_clusters = int(n_clusters_range[optimal_idx])

        # 选中的 K 若属于被单调修复者，其缓存模型是坏的局部最优 → 弃用缓存，
        # 让 perform_gmm_clustering 以完整 n_init 重新拟合
        if optimal_idx in repaired_idx:
            self._cached_gmms.pop(optimal_clusters, None)
            self.logger.warning(
                f"    ⚠️ [{band_name}] 选中的 K={optimal_clusters} 属修复项，"
                f"将重新拟合而非复用扫描模型"
            )

        bic_analysis = {
            'band_name': band_name,
            'n_clusters_range': n_clusters_range[: len(bics)].tolist(),
            'bics': bics,                    # 单调修复后（用于选 K 与绘图）
            'bics_raw': bics_raw,            # 修复前原始值（存档备查）
            'aics': aics,
            'log_likelihoods': log_likelihoods,
            'knee_distances': knee_distances,
            'knee_n': int(n_clusters_range[knee_idx]) if knee_idx is not None else None,
            'repaired_ks': [int(n_clusters_range[i]) for i in repaired_idx],
            'selection_rule': selection_rule,
            'k_upper': k_upper,
            'optimal_n': optimal_clusters,
            'optimal_bic': bics[optimal_idx],
            'optimal_aic': aics[optimal_idx],
            'argmin_n': argmin_clusters,
            'argmin_bic': bics[argmin_idx],
            'convergence_info': {
                'n_tested': len(bics),
            },
        }

        if selection_rule == 'bic_knee':
            self.logger.info(
                f"  ✅ [{band_name}] 选定 K={optimal_clusters} (BIC 拐点, "
                f"弦距={knee_distances[optimal_idx]:.3f}, K 上界={k_upper}); "
                f"argmin(BIC)=K{argmin_clusters}（仅作诊断，不稳定）"
            )
        else:
            self.logger.info(
                f"  ✅ [{band_name}] 选定 K={optimal_clusters} "
                f"({selection_rule}), BIC={bics[optimal_idx]:.2f}"
            )
        return optimal_clusters, bic_analysis

    def perform_gmm_clustering(
        self,
        X: np.ndarray,
        n_clusters: Optional[int] = None,
        band_name: str = 'full',
        bic_analysis: Optional[Dict[str, Any]] = None,
        spatial_indices: Optional[np.ndarray] = None,
        dims: Optional[Tuple[int, int, int]] = None,
        geometry: Optional[Dict[str, Any]] = None,
        sort_col: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, GaussianMixture, Dict[str, Any]]:
        """
        执行 GMM 聚类；若 reuse_bic_fit 且缓存命中则复用 BIC 搜索中的模型。

        在标准 GMM 之后可选做两步后处理：
        1. HMRF 空间正则化（均场近似），消除逐体元独立分类产生的椒盐噪声；
        2. 按簇均值 δlnVs 重排簇编号，使编号跨模型、跨深度带含义一致。

        Args:
            X: 该带特征矩阵 (n_points, n_features)
            n_clusters: 簇数；None 时自动选 K
            band_name: 深度带名称（日志与指标标注用）
            bic_analysis: 选 K 诊断信息
            spatial_indices: 该带各点的 (lat_idx, lon_idx, depth_idx)，HMRF 必需
            dims: 全网格形状 (n_lat, n_lon, n_depth)，HMRF 必需
            geometry: 含 lats/lons/depths 的网格几何，用于各向异性邻域权重
            sort_col: 用于排序的特征列索引（δlnVs 所在列）

        Returns:
            (labels, probabilities, gmm, metrics)
        """
        if n_clusters is None:
            n_clusters, bic_analysis = self.find_optimal_clusters(X, band_name=band_name)

        gmm_config = self.config.auto_gmm
        reuse = gmm_config.get('reuse_bic_fit', True) and n_clusters in self._cached_gmms

        self.logger.info(
            f"  📊 [{band_name}] GMM 聚类 K={n_clusters}"
            + (" (复用 BIC 拟合)" if reuse else "")
        )

        t0 = time.time()
        if reuse:
            gmm = self._cached_gmms[n_clusters]
            fit_time = time.time() - t0
        else:
            gmm = GaussianMixture(
                n_components=int(n_clusters),
                covariance_type=gmm_config['covariance_type'],
                n_init=gmm_config['n_init'],
                max_iter=gmm_config['max_iter'],
                random_state=gmm_config['random_state'],
                reg_covar=1e-6,
            )
            X_fit = self._subsample_for_fit(X)
            gmm.fit(X_fit)
            fit_time = time.time() - t0

        # 始终在全量点上预测，保证 3D 标签体完整
        labels = gmm.predict(X)
        probabilities = gmm.predict_proba(X)

        # ── HMRF 空间正则化（均场近似 EM）──
        hmrf_cfg = gmm_config.get('hmrf', {}) or {}
        hmrf_info: Optional[Dict[str, Any]] = None
        if hmrf_cfg.get('enabled', False):
            if spatial_indices is None or dims is None or geometry is None:
                self.logger.warning(
                    f"    ⚠️ [{band_name}] 缺少空间索引/网格几何，跳过 HMRF 正则化"
                )
            else:
                labels, probabilities, hmrf_info = self._hmrf_refine(
                    X, probabilities, gmm, spatial_indices, dims,
                    geometry, hmrf_cfg, band_name,
                )

        # ── 簇编号按均值 δlnVs 重排 ──
        order_rule = gmm_config.get('cluster_order')
        if order_rule and sort_col is not None:
            labels, probabilities = self._order_clusters(
                labels, probabilities, gmm, int(sort_col), str(order_rule)
            )

        metrics = self._calculate_clustering_metrics(
            X, labels, probabilities, gmm, fit_time
        )
        metrics['band_name'] = band_name
        if hmrf_info is not None:
            metrics['hmrf'] = hmrf_info
        if bic_analysis is not None:
            metrics['bic_optimal_n'] = bic_analysis.get('optimal_n')

        # 重抽样稳定性复验（默认关闭；开启后结果随 metrics 写入 JSON）
        stab_cfg = gmm_config.get('stability_check', {})
        if stab_cfg.get('enabled', False):
            metrics['stability'] = self._check_subsample_stability(
                X, int(n_clusters), labels, band_name, stab_cfg
            )

        self.logger.info(
            f"    ✅ [{band_name}] 完成: {metrics['n_clusters']} 簇, "
            f"mean_max_p={metrics['mean_max_probability']:.3f}"
        )
        return labels, probabilities, gmm, metrics

    def _check_subsample_stability(
        self,
        X: np.ndarray,
        n_clusters: int,
        ref_labels: np.ndarray,
        band_name: str,
        stab_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        重抽样稳定性复验：证明结果对"拟合子采样"不敏感。

        对每个复验种子：用该种子重新抽取拟合子样本并重新拟合 GMM
        （K 固定为已选定值，EM 初始化种子同步更换），在全量体素上预测标签，
        与参考标签计算 ARI（ARI 对簇编号置换不变）。ARI ≈ 1 说明子采样
        与随机初始化都不影响分区结果，可直接作为论文附录的稳健性证据。
        若样本量低于抽样上限（未触发抽样），复验退化为仅检验 EM 初始化的
        随机性，结论同样有效。

        Returns:
            {'seeds_ari': {种子: ARI}, 'mean_ari': 平均 ARI, 'n_clusters': K}
        """
        gmm_config = self.config.auto_gmm
        seeds = stab_cfg.get('seeds', [1234, 20240901])
        aris: Dict[str, float] = {}
        for seed in seeds:
            seed = int(seed)
            t0 = time.time()
            X_fit = self._subsample_for_fit(X, seed=seed)
            gmm_r = GaussianMixture(
                n_components=n_clusters,
                covariance_type=gmm_config['covariance_type'],
                n_init=gmm_config['n_init'],
                max_iter=gmm_config['max_iter'],
                random_state=seed,
                reg_covar=1e-6,
            )
            gmm_r.fit(X_fit)
            labels_r = gmm_r.predict(X)
            ari = float(adjusted_rand_score(ref_labels, labels_r))
            aris[str(seed)] = ari
            self.logger.info(
                f"    🔁 [{band_name}] 稳定性复验 seed={seed}: "
                f"ARI={ari:.4f} ({time.time() - t0:.1f}s)"
            )
        mean_ari = float(np.mean(list(aris.values()))) if aris else float('nan')
        verdict = '稳定' if mean_ari >= 0.9 else '不稳定，需检查'
        self.logger.info(
            f"    🔁 [{band_name}] 稳定性复验汇总: mean ARI={mean_ari:.4f} [{verdict}]"
        )
        return {'seeds_ari': aris, 'mean_ari': mean_ari, 'n_clusters': n_clusters}

    # ==================== K 扫描稳健性 ====================

    def scan_k_robustness(
        self,
        X: np.ndarray,
        k_selected: int,
        band_name: str,
        k_range: Optional[List[int]] = None,
        sort_col: int = 1,
        neutral_threshold: float = 0.01,
        inverse_transform: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        K 扫描稳健性：量化科学结论对 K 取值的依赖程度。

        K 无法从体元级数据可靠估计（Cai, Campbell & Broderick 2021, ICML：
        模型设定只要有任意偏差，有限混合模型的组分数后验即发散）。因此与其
        论证"K* 是真实簇数"，不如直接证明"结论在一段 K 区间内不变"——这正是
        Bao et al. (2026, Science) 补充材料中跨 k、跨阈值、跨模型的做法。

        输出两类指标：
        1. 相邻 K 的 ARI：反映分区本身随 K 的变化。K 不同则粒度不同，ARI
           必然小于 1，该曲线只用于定位分区结构趋于稳定的 K 区间。
        2. 快/慢三分类一致度：按簇平均 δlnVs 与 ±neutral_threshold 的关系把
           每个体元归为快/中性/慢，再与 K* 的结果比较。跨模型投票图只依赖这个
           三分类，故该曲线才是"结论是否依赖 K"的直接证据，应作为主要论据。

        Args:
            X: 该带特征矩阵（标准化域）
            k_selected: 已选定的 K*
            band_name: 深度带名称
            k_range: 扫描范围；缺省用已缓存的全部 K
            sort_col: δlnVs 所在特征列
            neutral_threshold: 中性带半宽（物理 δlnVs 单位）
            inverse_transform: 把标准化簇均值还原为物理量的回调

        Returns:
            含 k_values / ari_consecutive / ari_vs_selected / trichotomy_agreement
            的字典
        """
        gmm_config = self.config.auto_gmm
        if k_range is None:
            k_range = sorted(self._cached_gmms.keys()) or [k_selected]
        k_range = [int(k) for k in k_range]
        if k_selected not in k_range:
            k_range = sorted(set(k_range) | {int(k_selected)})

        self.logger.info(
            f"    📐 [{band_name}] K 扫描稳健性: K={k_range[0]}..{k_range[-1]}, "
            f"基准 K*={k_selected}"
        )

        labels_by_k: Dict[int, np.ndarray] = {}
        tri_by_k: Dict[int, np.ndarray] = {}

        for k in k_range:
            gmm = self._cached_gmms.get(k)
            if gmm is None:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=gmm_config['covariance_type'],
                    n_init=gmm_config['n_init'],
                    max_iter=gmm_config['max_iter'],
                    random_state=gmm_config['random_state'],
                    reg_covar=1e-6,
                )
                gmm.fit(self._subsample_for_fit(X))
            lbl = gmm.predict(X)
            labels_by_k[k] = lbl

            # 簇均值还原到物理 δlnVs，据此把体元归为 快(+1)/中性(0)/慢(-1)
            means = np.asarray(gmm.means_, dtype=float)
            phys = (
                np.asarray(inverse_transform(means))
                if inverse_transform is not None
                else means
            )
            vs_mean = phys[:, sort_col] if sort_col < phys.shape[1] else phys[:, 0]
            cluster_tri = np.where(
                vs_mean >= neutral_threshold, 1,
                np.where(vs_mean <= -neutral_threshold, -1, 0),
            ).astype(np.int8)
            tri_by_k[k] = cluster_tri[lbl]

        ref_tri = tri_by_k[int(k_selected)]
        ref_lbl = labels_by_k[int(k_selected)]

        ari_consecutive: Dict[str, float] = {}
        for a, b in zip(k_range[:-1], k_range[1:]):
            ari_consecutive[f"{a}->{b}"] = float(
                adjusted_rand_score(labels_by_k[a], labels_by_k[b])
            )
        ari_vs_selected = {
            str(k): float(adjusted_rand_score(ref_lbl, labels_by_k[k]))
            for k in k_range
        }
        tri_agreement = {
            str(k): float(np.mean(tri_by_k[k] == ref_tri)) for k in k_range
        }

        tri_vals = [v for k, v in tri_agreement.items() if int(k) != int(k_selected)]
        self.logger.info(
            f"    📐 [{band_name}] 三分类一致度: 最低 {min(tri_vals):.3f}, "
            f"均值 {float(np.mean(tri_vals)):.3f}（相对 K*={k_selected}）"
            if tri_vals else
            f"    📐 [{band_name}] K 扫描仅含单个 K，无对比"
        )

        return {
            'band_name': band_name,
            'k_values': k_range,
            'k_selected': int(k_selected),
            'neutral_threshold': float(neutral_threshold),
            'ari_consecutive': ari_consecutive,
            'ari_vs_selected': ari_vs_selected,
            'trichotomy_agreement': tri_agreement,
        }

    # ==================== 簇编号排序 ====================

    @staticmethod
    def _order_clusters(
        labels: np.ndarray,
        probabilities: np.ndarray,
        gmm: GaussianMixture,
        sort_col: int,
        rule: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        按簇均值在指定特征列上的取值重排簇编号，并同步重排 GMM 组分参数。

        GMM 的组分索引由 EM 初始化随机决定，本身不含信息：同一模型换个种子、
        或换个深度带，C0 的含义就完全不同，导致配色无法固定、跨模型的簇也无法
        直接对应。按 δlnVs 升序重排后 C0 = 最慢、C_{K-1} = 最快，编号本身即携带
        物理含义。标准化是单调仿射变换，故在标准化域上排序等价于在物理 δlnVs
        上排序，无需反变换。

        就地重排 gmm 的组分参数，使其与返回的标签、概率保持一致。

        Args:
            labels: 原始标签 (n_points,)
            probabilities: 原始后验概率 (n_points, K)
            gmm: 已拟合的 GMM（就地修改）
            sort_col: 排序依据的特征列索引
            rule: 'dlnvs_ascending' 或 'dlnvs_descending'

        Returns:
            (重排后的标签, 重排后的概率)
        """
        means = np.asarray(gmm.means_)
        if sort_col >= means.shape[1]:
            return labels, probabilities

        order = np.argsort(means[:, sort_col], kind='stable')
        if rule == 'dlnvs_descending':
            order = order[::-1].copy()

        if np.array_equal(order, np.arange(order.size)):
            return labels, probabilities

        # rank[旧编号] = 新编号
        rank = np.empty(order.size, dtype=np.int64)
        rank[order] = np.arange(order.size)

        new_labels = rank[labels].astype(labels.dtype, copy=False)
        new_probs = probabilities[:, order]

        gmm.weights_ = np.asarray(gmm.weights_)[order]
        gmm.means_ = means[order]
        # tied 协方差为全局共享 (d,d)，不随组分重排；其余按首轴重排
        for attr in ('covariances_', 'precisions_', 'precisions_cholesky_'):
            arr = getattr(gmm, attr, None)
            if arr is not None and np.asarray(arr).shape[0] == order.size:
                setattr(gmm, attr, np.asarray(arr)[order])

        return new_labels, new_probs

    # ==================== HMRF-GMM 空间正则化 ====================

    @staticmethod
    def _neighbor_edge_weights(
        geometry: Dict[str, Any],
        dims: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算 6 邻域三个方向的边权，按邻点物理距离反比加权。

        三个方向的格点间距在物理上并不等价：经度方向弧长随纬度按 cos(lat)
        收缩（本区跨 -15°~60°，高低纬相差约 2 倍），深度层间距通常非均匀且
        与水平间距量级不同。若三方向等权，等价于在物理空间中做各向异性平滑，
        高纬与深部会被过度平滑。以三方向特征间距的中位数为参考距离 d_ref，
        边权取 w = d_ref / d，并截断上限以防极端各向异性主导邻域项。

        Returns:
            (w_lat, w_lon, w_dep)，形状分别为 (n_lat-1,1,1)、(n_lat,1,1)、
            (1,1,n_dep-1)，可直接与移位后的三维场广播相乘。
        """
        DEG_KM = 111.19
        lats = np.asarray(geometry['lats'], dtype=float)
        lons = np.asarray(geometry['lons'], dtype=float)
        depths = np.asarray(geometry['depths'], dtype=float)
        n_lat, n_lon, n_dep = dims

        # 纬度方向：逐条边的弧长
        if n_lat > 1 and lats.size == n_lat:
            d_lat = np.abs(np.diff(lats)) * DEG_KM
        else:
            d_lat = np.array([np.inf])

        # 经度方向：弧长随格点纬度收缩，权重是纬度的函数
        if n_lon > 1 and lons.size == n_lon:
            d_lon_eq = float(np.median(np.abs(np.diff(lons)))) * DEG_KM
            # cos 下限防止极区弧长趋零导致权重爆炸
            d_lon = d_lon_eq * np.maximum(np.cos(np.deg2rad(lats)), 0.05)
        else:
            d_lon = np.array([np.inf])

        # 深度方向：层间距常随深度变粗
        if n_dep > 1 and depths.size == n_dep:
            d_dep = np.abs(np.diff(depths))
        else:
            d_dep = np.array([np.inf])

        finite = np.concatenate(
            [a[np.isfinite(a)] for a in (d_lat, d_lon, d_dep)] or [np.array([1.0])]
        )
        d_ref = float(np.median(finite)) if finite.size else 1.0

        def _w(d: np.ndarray) -> np.ndarray:
            return np.clip(d_ref / np.maximum(d, 1e-6), 0.0, 4.0)

        return (
            _w(d_lat).astype(np.float32).reshape(-1, 1, 1),
            _w(d_lon).astype(np.float32).reshape(-1, 1, 1),
            _w(d_dep).astype(np.float32).reshape(1, 1, -1),
        )

    @staticmethod
    def _neighbor_field(
        gamma: np.ndarray,
        li: np.ndarray,
        lj: np.ndarray,
        lk: np.ndarray,
        dims: Tuple[int, int, int],
        w_lat: np.ndarray,
        w_lon: np.ndarray,
        w_dep: np.ndarray,
    ) -> np.ndarray:
        """
        计算邻域一致性场 S_ik = Σ_{j∈N(i)} w_ij · γ_jk。

        逐类散射到三维网格后用移位求和，内存占用为单个网格大小而非 K 倍。
        带外体元在网格中留零，故深度带边界处邻域项自然减弱——各深度带本就
        独立聚类，跨带传播标签信息没有意义。

        Returns:
            (n_points, K) 的邻域一致性场
        """
        n_pts, n_k = gamma.shape
        out = np.empty((n_pts, n_k), dtype=np.float64)
        buf = np.zeros(dims, dtype=np.float32)
        acc = np.zeros(dims, dtype=np.float32)

        for k in range(n_k):
            buf[...] = 0.0
            buf[li, lj, lk] = gamma[:, k].astype(np.float32)
            acc[...] = 0.0
            # 纬度方向
            acc[:-1] += w_lat * buf[1:]
            acc[1:] += w_lat * buf[:-1]
            # 经度方向
            acc[:, :-1] += w_lon * buf[:, 1:]
            acc[:, 1:] += w_lon * buf[:, :-1]
            # 深度方向
            acc[:, :, :-1] += w_dep * buf[:, :, 1:]
            acc[:, :, 1:] += w_dep * buf[:, :, :-1]
            out[:, k] = acc[li, lj, lk]

        return out

    @staticmethod
    def _full_covariances(gmm: GaussianMixture, n_k: int, n_feat: int) -> np.ndarray:
        """把任意 covariance_type 的协方差统一展开为 (K, d, d)"""
        cov = np.asarray(gmm.covariances_)
        ctype = getattr(gmm, 'covariance_type', 'full')
        if ctype == 'full':
            return cov.astype(float).copy()
        if ctype == 'tied':
            return np.repeat(cov.astype(float)[None], n_k, axis=0)
        if ctype == 'diag':
            return np.stack([np.diag(cov[k].astype(float)) for k in range(n_k)])
        if ctype == 'spherical':
            return np.stack([np.eye(n_feat) * float(cov[k]) for k in range(n_k)])
        raise ValueError(f"不支持的协方差类型: {ctype}")

    @staticmethod
    def _gaussian_log_density(
        X: np.ndarray, means: np.ndarray, covs: np.ndarray
    ) -> np.ndarray:
        """各组分的多元正态对数密度 (n_points, K)，经 Cholesky 分解求马氏距离"""
        n_pts, n_feat = X.shape
        n_k = means.shape[0]
        out = np.empty((n_pts, n_k), dtype=np.float64)
        cst = n_feat * np.log(2.0 * np.pi)
        eye = np.eye(n_feat)
        for k in range(n_k):
            chol = np.linalg.cholesky(covs[k] + eye * 1e-10)
            sol = np.linalg.solve(chol, (X - means[k]).T).T
            out[:, k] = -0.5 * (
                cst
                + 2.0 * np.sum(np.log(np.diag(chol)))
                + np.einsum('ij,ij->i', sol, sol)
            )
        return out

    @staticmethod
    def _weighted_gaussian_mstep(
        X: np.ndarray, gamma: np.ndarray, reg: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """以均场后验为权重的 M 步，返回 (means, covs, weights)"""
        n_pts, n_feat = X.shape
        n_k = gamma.shape[1]
        n_eff = gamma.sum(axis=0) + 1e-10
        means = (gamma.T @ X) / n_eff[:, None]
        covs = np.empty((n_k, n_feat, n_feat), dtype=float)
        for k in range(n_k):
            diff = X - means[k]
            covs[k] = (diff * gamma[:, k : k + 1]).T @ diff / n_eff[k]
            covs[k].flat[:: n_feat + 1] += reg
        return means, covs, n_eff / float(n_pts)

    def _hmrf_refine(
        self,
        X: np.ndarray,
        probabilities: np.ndarray,
        gmm: GaussianMixture,
        spatial_indices: np.ndarray,
        dims: Tuple[int, int, int],
        geometry: Dict[str, Any],
        cfg: Dict[str, Any],
        band_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        HMRF-GMM 空间正则化（均场近似 EM）。

        标签场服从 Potts 先验，单体元能量
            U(l_i=k) = -log[π_k · N(x_i | μ_k, Σ_k)] + β·Σ_{j∈N(i)} w_ij·1[l_j ≠ k]
        取均场近似把邻域指示函数换成邻居后验 γ_jk，则罚项为
            β·Σ_j w_ij·(1 - γ_jk) = β·(W_i - S_ik)
        其中 W_i = Σ_j w_ij 与类别 k 无关，在 softmax 中作为逐点常数被约去。
        故均场后验只需在对数似然上加 +β·S_ik，S 即邻域一致性场——形式上是
        "与邻居一致则获得奖励"，且奖励按邻居自身的确信度加权。

        使用均场而非硬 ICM 的原因：下游的概率分布图、最大后验图与跨模型投票
        都依赖软后验，硬指派会把后验退化成 0/1，这些输出随之失效。

        参数更新（update_params=True）使其成为完整的 HMRF-EM：初始 GMM 只在
        子样本上拟合，而均场迭代在全量体元上进行，同步更新 μ/Σ/π 可让组分
        参数与正则化后的分区自洽。收敛后回写 gmm，并重算 precisions_cholesky_
        以保证该对象后续调用 predict/score 时仍然自洽。

        Returns:
            (labels, probabilities, 诊断信息)
        """
        beta = float(cfg.get('beta', 1.0))
        max_iter = int(cfg.get('max_iter', 8))
        tol = float(cfg.get('tol', 1e-3))
        update_params = bool(cfg.get('update_params', True))

        t0 = time.time()
        n_pts, n_feat = X.shape
        n_k = probabilities.shape[1]

        li = np.ascontiguousarray(spatial_indices[:, 0], dtype=np.intp)
        lj = np.ascontiguousarray(spatial_indices[:, 1], dtype=np.intp)
        lk = np.ascontiguousarray(spatial_indices[:, 2], dtype=np.intp)
        w_lat, w_lon, w_dep = self._neighbor_edge_weights(geometry, dims)

        gamma = np.asarray(probabilities, dtype=np.float64).copy()
        labels = np.argmax(gamma, axis=1).astype(np.int32)
        labels_gmm = labels.copy()

        means = np.asarray(gmm.means_, dtype=float).copy()
        covs = self._full_covariances(gmm, n_k, n_feat)
        weights = np.asarray(gmm.weights_, dtype=float).copy()

        self.logger.info(
            f"    🧲 [{band_name}] HMRF 空间正则化: β={beta}, "
            f"最多 {max_iter} 次均场迭代"
        )

        changed_hist: List[float] = []
        for it in range(max_iter):
            field = self._neighbor_field(
                gamma, li, lj, lk, dims, w_lat, w_lon, w_dep
            )
            log_post = self._gaussian_log_density(X, means, covs)
            log_post += np.log(np.maximum(weights, 1e-300))
            log_post += beta * field

            log_post -= log_post.max(axis=1, keepdims=True)
            np.exp(log_post, out=log_post)
            gamma = log_post / log_post.sum(axis=1, keepdims=True)

            new_labels = np.argmax(gamma, axis=1).astype(np.int32)
            changed = float(np.mean(new_labels != labels))
            labels = new_labels
            changed_hist.append(changed)

            if update_params:
                means, covs, weights = self._weighted_gaussian_mstep(X, gamma)

            self.logger.info(
                f"      迭代 {it + 1}/{max_iter}: 标签变动 {changed:.4%}"
            )
            if changed < tol:
                break

        # 回写自洽的组分参数
        if update_params:
            gmm.means_ = means
            gmm.weights_ = weights
            if getattr(gmm, 'covariance_type', 'full') == 'full':
                gmm.covariances_ = covs
                prec_chol = np.empty_like(covs)
                for k in range(n_k):
                    prec_chol[k] = np.linalg.inv(np.linalg.cholesky(covs[k])).T
                gmm.precisions_cholesky_ = prec_chol
                gmm.precisions_ = np.stack(
                    [prec_chol[k] @ prec_chol[k].T for k in range(n_k)]
                )

        moved = float(np.mean(labels != labels_gmm))
        self.logger.info(
            f"    🧲 [{band_name}] HMRF 完成: 相对纯 GMM 改判 {moved:.2%} 体元, "
            f"{time.time() - t0:.1f}s"
        )

        return labels, gamma, {
            'beta': beta,
            'n_iter': len(changed_hist),
            'changed_history': changed_hist,
            'converged': bool(changed_hist and changed_hist[-1] < tol),
            'relabeled_fraction_vs_gmm': moved,
            'update_params': update_params,
        }

    def _active_scheme(self) -> Tuple[str, Dict[str, Any]]:
        """返回当前分层方案名与方案配置"""
        cfg = self.config.depth_stratified
        scheme = str(cfg.get('scheme', 'fixed_3band'))
        schemes = cfg.get('schemes', {})
        if scheme not in schemes:
            raise ValueError(
                f"未知分层方案 '{scheme}'，可选: {list(schemes.keys())}"
            )
        return scheme, schemes[scheme]

    def _moho_depth_per_point(
        self,
        spatial_indices: np.ndarray,
        metadata: Dict[str, Any],
    ) -> np.ndarray:
        """
        为每个体素取 Moho 深度（km）。

        无覆盖处用 moho_fallback_km，并裁剪到 moho_clip_km。
        """
        if self.concordance_eval is None:
            self.concordance_eval = GeologyConcordanceEvaluator(
                self.config, self.logger
            )

        lons = np.asarray(metadata['lons'], dtype=float)
        lats = np.asarray(metadata['lats'], dtype=float)
        moho_grid = self.concordance_eval.get_moho_grid(lons, lats)
        if moho_grid is None:
            raise RuntimeError(
                "moho_4band 需要 Moho 数据，但插值失败；请检查 "
                f"{self.config.geology_concordance.get('moho_file')}"
            )

        cfg = self.config.depth_stratified
        fallback = float(cfg.get('moho_fallback_km', 35.0))
        z_lo, z_hi = cfg.get('moho_clip_km', [10.0, 80.0])

        lat_i = spatial_indices[:, 0].astype(int)
        lon_i = spatial_indices[:, 1].astype(int)
        moho_z = moho_grid[lat_i, lon_i].astype(np.float64)
        n_fallback = int(np.sum(~np.isfinite(moho_z)))
        moho_z[~np.isfinite(moho_z)] = fallback
        moho_z = np.clip(moho_z, float(z_lo), float(z_hi))

        self.logger.info(
            f"  🧭 Moho 分带: 点位={len(moho_z):,}, "
            f"深度[{moho_z.min():.1f}, {moho_z.max():.1f}] km, "
            f"回退填充 {n_fallback:,} 点 → {fallback:.1f} km"
        )
        return moho_z

    def _build_band_mask(
        self,
        band: Dict[str, Any],
        point_depths: np.ndarray,
        moho_z: Optional[np.ndarray],
        is_last: bool,
        model_z_max: float,
    ) -> Tuple[np.ndarray, List[float], str]:
        """
        构造深度带布尔掩膜。

        Returns:
            (mask, depth_range_for_log, description)
        """
        mask_type = band.get('mask', 'depth_range')

        if mask_type == 'above_moho':
            if moho_z is None:
                raise ValueError(f"带 {band['name']} 需要 Moho 场")
            cap = float(band.get('depth_cap', 410))
            mask = (point_depths < moho_z) & (point_depths < cap)
            z_ref = [0.0, float(np.nanmedian(moho_z))]
            desc = f"z < Moho (median={z_ref[1]:.1f} km, cap={cap:.0f})"
            return mask, z_ref, desc

        if mask_type == 'moho_to_depth':
            if moho_z is None:
                raise ValueError(f"带 {band['name']} 需要 Moho 场")
            _, z1 = band['depth_range']
            z1_eff = model_z_max if z1 is None else float(z1)
            mask = (point_depths >= moho_z) & (point_depths < z1_eff)
            if is_last:
                mask = (point_depths >= moho_z) & (point_depths <= z1_eff)
            z_ref = [float(np.nanmedian(moho_z)), z1_eff]
            desc = f"Moho ≤ z < {z1_eff:.0f} km (median Moho={z_ref[0]:.1f})"
            return mask, z_ref, desc

        # 固定深度区间
        z0, z1 = band['depth_range']
        z0_eff = 0.0 if z0 is None else float(z0)
        z1_eff = model_z_max if z1 is None else float(z1)
        mask = (point_depths >= z0_eff) & (point_depths < z1_eff)
        if is_last:
            mask = (point_depths >= z0_eff) & (point_depths <= z1_eff)
        desc = f"{z0_eff:.0f}-{z1_eff:.0f} km"
        return mask, [z0_eff, z1_eff], desc

    def perform_depth_stratified_clustering(
        self,
        X: np.ndarray,
        spatial_indices: np.ndarray,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        按深度带分别做 GMM，再拼回全局标签/概率。

        Returns:
            含 labels / probabilities / labels_3d / per_band / metrics 的结果字典
        """
        depths = np.asarray(metadata['depths'], dtype=float)
        dims = metadata['dims']

        # HMRF 邻域权重所需的网格几何；簇排序所需的 δlnVs 列索引
        geometry = {
            'lats': metadata['lats'],
            'lons': metadata['lons'],
            'depths': depths,
        }
        feature_names = [str(f).lower() for f in metadata.get('features', [])]
        sort_col = feature_names.index('vs') if 'vs' in feature_names else None
        if sort_col is None and feature_names:
            # 无 vs 时退回首个特征，保证簇编号仍然有序可比
            sort_col = 0
            self.logger.warning("  ⚠️ 特征中无 vs，簇排序退回首个特征列")

        scheme, scheme_cfg = self._active_scheme()
        bands = scheme_cfg['bands']
        offset_labels = self.config.depth_stratified.get('offset_labels_across_bands', True)

        self.logger.info(f"  📚 分层方案: {scheme}（{len(bands)} 带）")

        needs_moho = any(
            b.get('mask') in ('above_moho', 'moho_to_depth') for b in bands
        )
        moho_z = (
            self._moho_depth_per_point(spatial_indices, metadata)
            if needs_moho
            else None
        )

        global_labels = np.full(len(X), -1, dtype=np.int32)
        max_k_possible = max(
            int(b.get('fixed_k') or b['max_clusters']) for b in bands
        )
        global_probs = np.zeros((len(X), max_k_possible), dtype=np.float64)

        per_band: Dict[str, Any] = {}
        per_band_krob: Dict[str, Any] = {}
        label_offset = 0
        total_fit_time = 0.0
        point_depths = depths[spatial_indices[:, 2]]
        model_z_max = float(np.nanmax(depths))

        for i_band, band in enumerate(bands):
            name = band['name']
            is_last = i_band == len(bands) - 1
            mask, z_ref, desc = self._build_band_mask(
                band, point_depths, moho_z, is_last, model_z_max
            )

            n_band = int(np.sum(mask))
            self.logger.info(f"  📦 [{name}] {desc}, N={n_band:,}")
            if n_band < max(int(band['min_clusters']) * 10, 50):
                self.logger.warning(
                    f"  ⚠️ 深度带 {name} 有效点过少 ({n_band})，跳过"
                )
                continue

            X_band = X[mask]
            fixed_k = band.get('fixed_k')
            if fixed_k:
                # 先验固定 K：跳过 BIC 扫描（清空缓存避免复用旧带的模型）
                optimal_n = int(fixed_k)
                self._cached_gmms = {}
                bic_analysis = {
                    'band_name': name,
                    'optimal_n': optimal_n,
                    'optimal_bic': float('nan'),
                    'selection_rule': 'fixed_k',
                }
                self.logger.info(f"  📌 [{name}] 先验固定 K={optimal_n}（跳过 BIC 扫描）")
            else:
                optimal_n, bic_analysis = self.find_optimal_clusters(
                    X_band,
                    min_clusters=int(band['min_clusters']),
                    max_clusters=int(band['max_clusters']),
                    band_name=name,
                    spatial_indices=spatial_indices[mask],
                    dims=dims,
                )
            labels_b, probs_b, gmm_b, metrics_b = self.perform_gmm_clustering(
                X_band,
                n_clusters=optimal_n,
                band_name=name,
                bic_analysis=bic_analysis,
                spatial_indices=spatial_indices[mask],
                dims=dims,
                geometry=geometry,
                sort_col=sort_col,
            )
            total_fit_time += float(metrics_b.get('fit_time', 0.0))

            # K 扫描稳健性：在 BIC 缓存仍然有效时进行。各 K 一律用纯 GMM 标签
            # 比较，以隔离 K 本身的影响（若含 HMRF 则平滑效应会与 K 效应混淆）。
            krob_cfg = self.config.auto_gmm.get('k_robustness', {}) or {}
            if krob_cfg.get('enabled', False):
                try:
                    per_band_krob[name] = self.scan_k_robustness(
                        X_band,
                        k_selected=optimal_n,
                        band_name=name,
                        sort_col=int(sort_col) if sort_col is not None else 1,
                        neutral_threshold=float(
                            krob_cfg.get('neutral_threshold', 0.01)
                        ),
                        inverse_transform=self.feature_inverse_transform,
                    )
                except Exception as exc:
                    self.logger.warning(f"  ⚠️ [{name}] K 扫描稳健性失败: {exc}")

            if offset_labels:
                labels_global_b = labels_b + label_offset
                label_offset += optimal_n
            else:
                labels_global_b = labels_b

            global_labels[mask] = labels_global_b
            k_b = probs_b.shape[1]
            idx = np.where(mask)[0]
            global_probs[idx, :k_b] = probs_b

            per_band[name] = {
                'depth_range': [float(z_ref[0]), float(z_ref[1])],
                'mask_type': band.get('mask', 'depth_range'),
                'description': desc,
                'n_points': n_band,
                'n_clusters': optimal_n,
                'labels_local': labels_b,
                'labels_global': labels_global_b,
                'probabilities': probs_b,
                'model': gmm_b,
                'metrics': metrics_b,
                'bic_analysis': bic_analysis,
                'label_offset': label_offset - optimal_n if offset_labels else 0,
                'point_mask': mask,
            }

        if not per_band:
            raise ValueError("所有深度带均无法完成聚类")

        n_global = int(global_labels.max()) + 1 if global_labels.max() >= 0 else 1
        max_prob = np.max(global_probs, axis=1)

        labels_3d = Reconstruction3DHelper.reconstruct_3d_labels(
            global_labels, spatial_indices, dims
        )

        valid = global_labels >= 0
        probs_3d_max = Reconstruction3DHelper.reconstruct_3d_field(
            max_prob[valid], spatial_indices[valid], dims
        )

        overall_metrics = {
            'n_clusters': n_global,
            'n_bands': len(per_band),
            'scheme': scheme,
            'fit_time': total_fit_time,
            'mean_max_probability': float(np.mean(max_prob[valid])) if np.any(valid) else 0.0,
            'high_confidence_ratio': float(
                np.mean(max_prob[valid] > self.config.auto_gmm['probability_threshold'])
            )
            if np.any(valid)
            else 0.0,
            'mean_entropy': float('nan'),
            'band_optimal_k': {k: v['n_clusters'] for k, v in per_band.items()},
            'silhouette_score': float(
                np.nanmean(
                    [
                        v['metrics'].get('silhouette_score', np.nan)
                        for v in per_band.values()
                    ]
                )
            ),
            'converged': all(v['metrics'].get('converged', False) for v in per_band.values()),
            'n_iter': int(
                np.sum([v['metrics'].get('n_iter', 0) for v in per_band.values()])
            ),
            'bic': float(
                np.nansum([v['metrics'].get('bic', np.nan) for v in per_band.values()])
            ),
            'aic': float(
                np.nansum([v['metrics'].get('aic', np.nan) for v in per_band.values()])
            ),
            'davies_bouldin_score': float(
                np.nanmean(
                    [
                        v['metrics'].get('davies_bouldin_score', np.nan)
                        for v in per_band.values()
                    ]
                )
            ),
            'calinski_harabasz_score': float(
                np.nanmean(
                    [
                        v['metrics'].get('calinski_harabasz_score', np.nan)
                        for v in per_band.values()
                    ]
                )
            ),
            'median_max_probability': float(np.median(max_prob[valid])) if np.any(valid) else 0.0,
            'std_max_probability': float(np.std(max_prob[valid])) if np.any(valid) else 0.0,
            'low_confidence_ratio': float(np.mean(max_prob[valid] < 0.5)) if np.any(valid) else 0.0,
            'cluster_sizes': {
                int(k): int(v)
                for k, v in zip(*np.unique(global_labels[valid], return_counts=True))
            },
        }

        first_band = next(iter(per_band.values()))
        return {
            'n_clusters': n_global,
            'labels': global_labels,
            'labels_3d': labels_3d,
            'probabilities': global_probs[:, :1],
            'probabilities_3d': probs_3d_max[..., np.newaxis],
            'max_probability': max_prob,
            'max_probability_3d': probs_3d_max,
            'model': None,
            'metrics': overall_metrics,
            'bic_analysis': first_band['bic_analysis'],
            'per_band': per_band,
            'k_robustness': per_band_krob,
            'mode': 'depth_stratified',
            'scheme': scheme,
        }

    def _subsample_for_fit(
        self,
        X: np.ndarray,
        spatial_depth_indices: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        为 BIC/GMM 拟合构造子样本。

        注意：标签预测始终在全量点上进行，避免 3D 重建出现空洞。

        Args:
            seed: 抽样随机种子；None 时用全局 RANDOM_SEED（稳定性复验时传入不同种子）
        """
        samp = self.config.preprocessing.get('sampling', {})
        if not samp.get('enabled', True):
            return X
        max_points = int(samp.get('max_total_points', 2_000_000))
        if len(X) <= max_points:
            return X

        rng = np.random.default_rng(RANDOM_SEED if seed is None else int(seed))
        if spatial_depth_indices is not None and len(spatial_depth_indices) == len(X):
            unique_depths = np.unique(spatial_depth_indices)
            n_depths = max(len(unique_depths), 1)
            per_depth = max(
                max_points // n_depths, int(samp.get('min_points_per_depth', 500))
            )
            chosen: List[int] = []
            for d in unique_depths:
                idx = np.where(spatial_depth_indices == d)[0]
                n = min(per_depth, len(idx))
                if n > 0:
                    chosen.extend(rng.choice(idx, n, replace=False).tolist())
            chosen_arr = np.asarray(chosen, dtype=np.int64)
            if len(chosen_arr) > max_points:
                chosen_arr = rng.choice(chosen_arr, max_points, replace=False)
            self.logger.info(
                f"    拟合子采样(分层): {len(X):,} → {len(chosen_arr):,}"
            )
            return X[chosen_arr]

        idx = rng.choice(len(X), max_points, replace=False)
        self.logger.info(f"    拟合子采样: {len(X):,} → {max_points:,}")
        return X[idx]

    def _calculate_clustering_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
        gmm: GaussianMixture,
        fit_time: float,
    ) -> Dict[str, Any]:
        """计算详细的聚类指标"""
        n_clusters = len(np.unique(labels))

        metrics: Dict[str, Any] = {
            'n_clusters': n_clusters,
            'fit_time': fit_time,
            'converged': bool(gmm.converged_),
            'n_iter': int(gmm.n_iter_),
            'bic': float(gmm.bic(X)),
            'aic': float(gmm.aic(X)),
            'log_likelihood': float(gmm.score(X) * len(X)),
        }

        max_probs = np.max(probabilities, axis=1)
        metrics['mean_max_probability'] = float(np.mean(max_probs))
        metrics['median_max_probability'] = float(np.median(max_probs))
        metrics['std_max_probability'] = float(np.std(max_probs))

        threshold = self.config.auto_gmm['probability_threshold']
        metrics['high_confidence_ratio'] = float(np.mean(max_probs > threshold))
        metrics['low_confidence_ratio'] = float(np.mean(max_probs < 0.5))

        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        metrics['mean_entropy'] = float(np.mean(entropy))
        metrics['median_entropy'] = float(np.median(entropy))

        if n_clusters > 1 and len(X) > 10:
            metrics['silhouette_score'] = float(
                silhouette_score(X, labels, sample_size=min(5000, len(X)))
            )
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(X, labels))
            metrics['calinski_harabasz_score'] = float(
                calinski_harabasz_score(X, labels)
            )

        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = {
            int(k): int(v) for k, v in zip(unique_labels.tolist(), counts.tolist())
        }
        metrics['cluster_balance'] = float(np.std(counts) / max(np.mean(counts), 1e-12))
        return metrics


class KMeansComparisonAnalyzer:
    """K-means对比分析器"""
    
    def __init__(self, config: ClusteringConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def run_kmeans_comparison(
        self,
        X: np.ndarray,
        gmm_labels: np.ndarray,
        gmm_n_clusters: int
    ) -> Dict[str, Any]:
        """
        运行K-means对比分析
        
        Returns:
        --------
        comparison_results : Dict
            K-means vs GMM对比结果
        """
        self.logger.info(f"  🔹 运行K-means对比分析...")
        
        kmeans_config = self.config.kmeans_comparison
        n_clusters_list = kmeans_config['n_clusters']
        
        # 确保包含GMM的聚类数
        if gmm_n_clusters not in n_clusters_list:
            n_clusters_list = sorted(list(set(n_clusters_list + [gmm_n_clusters])))
        
        results = {
            'n_clusters_tested': n_clusters_list,
            'models': {},
            'metrics': {},
            'best_k': None,
            'best_score': -1,
            'comparison_with_gmm': {}
        }
        
        # 对每个k值运行K-means
        for k in n_clusters_list:
            self.logger.info(f"    测试 k={k}")
            
            # 选择算法
            if kmeans_config['use_minibatch'] and len(X) > 10000:
                kmeans = MiniBatchKMeans(
                    n_clusters=k,
                    batch_size=kmeans_config['batch_size'],
                    random_state=RANDOM_SEED,
                    n_init=10
                )
            else:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=RANDOM_SEED,
                    n_init=10
                )
            
            # 训练
            t0 = time.time()
            kmeans_labels = kmeans.fit_predict(X)
            fit_time = time.time() - t0
            
            # 计算指标
            metrics = self._calculate_kmeans_metrics(X, kmeans_labels, fit_time)
            
            results['models'][k] = kmeans
            results['metrics'][k] = metrics
            
            # 更新最优k
            if metrics['silhouette_score'] > results['best_score']:
                results['best_score'] = metrics['silhouette_score']
                results['best_k'] = k
        
        # 与GMM对比（使用相同的聚类数）
        if gmm_n_clusters in results['metrics']:
            kmeans_metrics = results['metrics'][gmm_n_clusters]
            
            # 计算标签一致性
            kmeans_labels_same_k = results['models'][gmm_n_clusters].labels_
            ari = adjusted_rand_score(gmm_labels, kmeans_labels_same_k)
            nmi = normalized_mutual_info_score(gmm_labels, kmeans_labels_same_k)
            
            results['comparison_with_gmm'] = {
                'n_clusters': gmm_n_clusters,
                'adjusted_rand_index': float(ari),
                'normalized_mutual_info': float(nmi),
                'kmeans_silhouette': kmeans_metrics['silhouette_score'],
                'kmeans_davies_bouldin': kmeans_metrics['davies_bouldin_score'],
                'kmeans_calinski': kmeans_metrics['calinski_harabasz_score'],
                'agreement_ratio': float(np.mean(gmm_labels == kmeans_labels_same_k))
            }
            
            self.logger.info(f"    与GMM一致性: ARI={ari:.4f}, NMI={nmi:.4f}")
        
        self.logger.info(f"    最优 k={results['best_k']}, 轮廓系数={results['best_score']:.4f}")
        
        return results
    
    def _calculate_kmeans_metrics(
        self, 
        X: np.ndarray, 
        labels: np.ndarray, 
        fit_time: float
    ) -> Dict[str, float]:
        """计算K-means聚类指标"""
        n_clusters = len(np.unique(labels))
        
        metrics = {
            'n_clusters': n_clusters,
            'fit_time': fit_time
        }
        
        if n_clusters > 1:
            metrics['silhouette_score'] = silhouette_score(
                X, labels, sample_size=min(5000, len(X))
            )
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        else:
            metrics['silhouette_score'] = 0
            metrics['davies_bouldin_score'] = float('inf')
            metrics['calinski_harabasz_score'] = 0
        
        # 簇大小统计
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
        metrics['cluster_balance'] = float(np.std(counts) / np.mean(counts))
        
        return metrics


class Reconstruction3DHelper:
    """三维重建辅助类（向量化）"""

    @staticmethod
    def reconstruct_3d_labels(
        labels: np.ndarray,
        spatial_indices: np.ndarray,
        dims: Tuple[int, int, int],
    ) -> np.ndarray:
        """将1D标签重建为3D立方体"""
        labels_3d = np.full(dims, -1, dtype=np.int32)
        if len(spatial_indices) == 0:
            return labels_3d
        ii, jj, kk = (
            spatial_indices[:, 0],
            spatial_indices[:, 1],
            spatial_indices[:, 2],
        )
        labels_3d[ii, jj, kk] = labels.astype(np.int32)
        return labels_3d

    @staticmethod
    def reconstruct_3d_probabilities(
        probabilities: np.ndarray,
        spatial_indices: np.ndarray,
        dims: Tuple[int, int, int],
    ) -> np.ndarray:
        """将概率矩阵重建为3D立方体"""
        n_clusters = probabilities.shape[1]
        probs_3d = np.full(dims + (n_clusters,), np.nan, dtype=np.float32)
        if len(spatial_indices) == 0:
            return probs_3d
        ii, jj, kk = (
            spatial_indices[:, 0],
            spatial_indices[:, 1],
            spatial_indices[:, 2],
        )
        probs_3d[ii, jj, kk, :] = probabilities.astype(np.float32)
        return probs_3d

    @staticmethod
    def reconstruct_3d_field(
        values: np.ndarray,
        spatial_indices: np.ndarray,
        dims: Tuple[int, int, int],
    ) -> np.ndarray:
        """将1D标量场重建为3D立方体"""
        field_3d = np.full(dims, np.nan, dtype=np.float32)
        if len(spatial_indices) == 0:
            return field_3d
        ii, jj, kk = (
            spatial_indices[:, 0],
            spatial_indices[:, 1],
            spatial_indices[:, 2],
        )
        field_3d[ii, jj, kk] = values.astype(np.float32)
        return field_3d


class SmartClusteringPipeline:
    """单模型速度相聚类分析管道（δlnV + 分层 GMM + 结构符合度）"""

    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self.base_config = BaseConfig()
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.ModelClustering', 'INFO'
        )

        self.loader = NetCDF3DLoader(self.config, self.logger)
        self.processor = DataCube3DProcessor(self.config, self.logger)
        self.gmm_optimizer = GMMAutoClusteringOptimizer(self.config, self.logger)
        self.concordance_eval = GeologyConcordanceEvaluator(self.config, self.logger)
        self.gmm_optimizer.concordance_eval = self.concordance_eval

        self.enhanced_visualizer = EnhancedClusteringVisualizer(self.config, self.logger)
        self.enhanced_visualizer.set_processor(self.processor)
        self.basic_visualizer = BasicClusteringVisualizer(self.config, self.logger)
        self.basic_visualizer.set_processor(self.processor)

        if self.config.kmeans_comparison['enabled']:
            self.kmeans_analyzer = KMeansComparisonAnalyzer(self.config, self.logger)
        else:
            self.kmeans_analyzer = None

        self.all_results: Dict[str, Any] = {}

    def _scheme_output_root(self) -> Path:
        """数据根目录：results/model_clustering/{scheme}/（标签、概率、NetCDF、统计）"""
        scheme = str(self.config.depth_stratified.get('scheme', 'fixed_3band'))
        return self.base_config.dirs['results'] / 'model_clustering' / scheme

    def _scheme_figure_root(self) -> Path:
        """图件根目录：figures/model_clustering/{scheme}/

        图件与数据分置：图件集中在 figures/ 便于取用与投稿，数据留在 results/。
        """
        scheme = str(self.config.depth_stratified.get('scheme', 'fixed_3band'))
        return self.base_config.dirs['figures'] / 'model_clustering' / scheme

    def _setup_logger(self) -> logging.Logger:
        """回退日志初始化"""
        logger = logging.getLogger('EASTASIA-FWI.ModelClustering')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        log_dir = self.base_config.dirs['logs']
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(log_dir / f'clustering_{timestamp}.log')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        return logger

    def run_analysis(
        self,
        model_names: Optional[List[str]] = None,
        features_to_use: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """运行完整的单模型速度相聚类分析"""
        try:
            self._print_header()

            if model_names is None:
                model_names = self.config.data['target_models']
            if features_to_use is None:
                features_to_use = self.config.data['default_features']

            stratified = self.config.depth_stratified.get('enabled', True)
            scheme = str(self.config.depth_stratified.get('scheme', 'fixed_3band'))
            pert_on = self.config.preprocessing.get('perturbation', {}).get(
                'enabled', True
            )

            self.logger.info("\n" + "=" * 80)
            self.logger.info("🚀 EASTASIA-FWI 单模型速度相聚类 (v7.5)")
            self.logger.info("=" * 80)
            self.logger.info(
                f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            self.logger.info(f"\n📊 分析模型 ({len(model_names)}):")
            for i, model in enumerate(model_names, 1):
                self.logger.info(f"  {i}. {model}")
            self.logger.info(f"\n🔬 特征: {', '.join(features_to_use)}")
            self.logger.info(f"📐 扰动域: {'ON' if pert_on else 'OFF'}")
            self.logger.info(
                f"📚 深度分层: {'ON' if stratified else 'OFF'} ({scheme})"
            )
            self.logger.info(f"📁 输出: {self._scheme_output_root()}")

            success_count = 0
            failed_models: List[Tuple[str, str]] = []

            for model_name in model_names:
                try:
                    self.logger.info(f"\n{'=' * 80}")
                    self.logger.info(f"🔬 分析模型: {model_name}")
                    self.logger.info(f"{'=' * 80}")

                    data_cube, metadata = self.loader.load_3d_model(
                        model_name, features_to_use
                    )
                    X, spatial_indices, prep_info = self.processor.prepare_for_clustering(
                        data_cube, metadata
                    )
                    # 用扰动后的特征名覆盖 metadata 展示名
                    if prep_info.get('feature_names'):
                        metadata = dict(metadata)
                        metadata['feature_display_names'] = prep_info['feature_names']

                    results: Dict[str, Any] = {
                        'model_name': model_name,
                        'data_shape': X.shape,
                        'features': X,
                        'spatial_indices': spatial_indices,
                        'metadata': metadata,
                        'preprocessing_info': prep_info,
                        'gmm_results': None,
                        'kmeans_results': None,
                    }

                    # 注入标准化反变换：K 扫描的快/慢三分类须在物理 δlnV 上判定
                    self.gmm_optimizer.feature_inverse_transform = (
                        self.processor.scaler.inverse_transform
                        if getattr(self.processor, 'scaler', None) is not None
                        else None
                    )

                    self.logger.info("\n🤖 执行 GMM 速度相聚类...")
                    if stratified:
                        gmm_results = self.gmm_optimizer.perform_depth_stratified_clustering(
                            X, spatial_indices, metadata
                        )
                    else:
                        optimal_n, bic_analysis = self.gmm_optimizer.find_optimal_clusters(
                            X
                        )
                        labels, probabilities, gmm_model, gmm_metrics = (
                            self.gmm_optimizer.perform_gmm_clustering(
                                X, n_clusters=optimal_n, bic_analysis=bic_analysis
                            )
                        )
                        labels_3d = Reconstruction3DHelper.reconstruct_3d_labels(
                            labels, spatial_indices, metadata['dims']
                        )
                        probabilities_3d = (
                            Reconstruction3DHelper.reconstruct_3d_probabilities(
                                probabilities, spatial_indices, metadata['dims']
                            )
                        )
                        gmm_results = {
                            'n_clusters': optimal_n,
                            'labels': labels,
                            'labels_3d': labels_3d,
                            'probabilities': probabilities,
                            'probabilities_3d': probabilities_3d,
                            'model': gmm_model,
                            'metrics': gmm_metrics,
                            'bic_analysis': bic_analysis,
                            'per_band': None,
                            'mode': 'full_depth',
                        }

                    results['gmm_results'] = gmm_results

                    # K-means（可选）
                    if self.kmeans_analyzer and gmm_results.get('mode') != 'depth_stratified':
                        self.logger.info("\n📊 执行 K-means 对比...")
                        results['kmeans_results'] = (
                            self.kmeans_analyzer.run_kmeans_comparison(
                                X, gmm_results['labels'], gmm_results['n_clusters']
                            )
                        )

                    self.logger.info("\n📊 生成可视化...")
                    self._visualize_all_results(results, model_name)

                    self.logger.info("\n💾 保存结果...")
                    self._save_results(results, model_name)

                    self.all_results[model_name] = results
                    success_count += 1
                    self.logger.info(f"\n✅ 模型 {model_name} 分析完成")

                except Exception as e:
                    self.logger.error(f"\n❌ 模型 {model_name} 分析失败: {e}")
                    self.logger.error(traceback.format_exc())
                    failed_models.append((model_name, str(e)))

            self._generate_summary_report(success_count, failed_models)
            return self.all_results

        except Exception as e:
            self.logger.error(f"❌ 管道执行失败: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _visualize_all_results(self, results: Dict[str, Any], model_name: str) -> None:
        """生成可视化"""
        viz_dir = self._scheme_figure_root() / model_name
        viz_dir.mkdir(parents=True, exist_ok=True)

        gmm_results = results['gmm_results']
        kmeans_results = results.get('kmeans_results')
        vis_config = self.config.visualization['plot_types']
        stratified = gmm_results.get('mode') == 'depth_stratified'

        # 选 K 判据图：分层时优先合成一张（BIC 拐点模式），否则逐带绘制
        if vis_config.get('bic_analysis'):
            if stratified and gmm_results.get('per_band'):
                merged = self.enhanced_visualizer.plot_k_selection_panels(
                    gmm_results['per_band'], viz_dir, model_name
                )
                if not merged:
                    for band_name, band_res in gmm_results['per_band'].items():
                        self.enhanced_visualizer.plot_bic_analysis(
                            band_res['bic_analysis'],
                            viz_dir,
                            f"{model_name}_{band_name}",
                        )
            else:
                self.enhanced_visualizer.plot_bic_analysis(
                    gmm_results['bic_analysis'], viz_dir, model_name
                )

        # K 扫描稳健性图（k_robustness 开启时才有数据）
        if gmm_results.get('k_robustness'):
            self.enhanced_visualizer.plot_k_robustness(
                gmm_results['k_robustness'], viz_dir, model_name
            )

        # 概率分析：分层时用 max_probability 构造伪概率分布
        if vis_config.get('probability_distribution') or vis_config.get(
            'confidence_analysis'
        ):
            if stratified:
                max_p = gmm_results.get('max_probability')
                if max_p is not None:
                    # 用两列伪概率 [1-p, p] 兼容原绘图接口
                    pseudo = np.column_stack([1.0 - max_p, max_p])
                    self.enhanced_visualizer.plot_probability_distribution(
                        pseudo, gmm_results['labels'], viz_dir, model_name
                    )
            else:
                self.enhanced_visualizer.plot_probability_distribution(
                    gmm_results['probabilities'],
                    gmm_results['labels'],
                    viz_dir,
                    model_name,
                )

        if vis_config.get('gmm_vs_kmeans') and kmeans_results:
            self.enhanced_visualizer.plot_gmm_vs_kmeans_comparison(
                gmm_results, kmeans_results, viz_dir, model_name
            )

        if vis_config.get('depth_slices'):
            # 3×3：第1排地质，第2–3排 Slab2（替代原 2-3-9 / 2-3-10）
            self.basic_visualizer.plot_depth_slices(
                gmm_results['labels_3d'],
                results['metadata'],
                viz_dir,
                'gmm',
                model_name,
                concordance_eval=self.concordance_eval,
            )

        # 未标准化扰动立方体（供 2-3-5 / 2-3-11 叠 δlnVp/δlnVs）
        pert_cube = getattr(self.processor, 'last_feature_cube', None)

        # 三维同维度：聚类体 vs Slab2(dep+thk) 体掩膜（独立于切片 MI/ρ）
        if vis_config.get('slab_volume_3d', True):
            try:
                self.concordance_eval.plot_slab_volume_comparison(
                    gmm_results['labels_3d'],
                    results['metadata'],
                    viz_dir,
                    model_name,
                    pert_cube=pert_cube,
                )
            except Exception as e:
                self.logger.warning(f"  ⚠️ Slab2 三维体对比图失败: {e}")

        if vis_config.get('vertical_sections'):
            self.basic_visualizer.plot_vertical_sections(
                gmm_results['labels_3d'],
                results['metadata'],
                viz_dir,
                'gmm',
                model_name,
                concordance_eval=self.concordance_eval,
                pert_cube=pert_cube,
            )

        if vis_config.get('cluster_profiles'):
            self.basic_visualizer.plot_cluster_profiles(
                results['features'],
                gmm_results['labels'],
                results['spatial_indices'],
                results['metadata'],
                viz_dir,
                'gmm',
                model_name,
                per_band=gmm_results.get('per_band'),
            )

        if vis_config.get('feature_distributions'):
            self.basic_visualizer.plot_feature_distributions(
                results['features'],
                gmm_results['labels'],
                results['metadata'],
                viz_dir,
                'gmm',
                model_name,
                per_band=gmm_results.get('per_band'),
            )

        if vis_config.get('cluster_centers') and gmm_results.get('model') is not None:
            self.basic_visualizer.plot_cluster_centers(
                results['features'],
                gmm_results['labels'],
                results['metadata'],
                gmm_results['model'],
                viz_dir,
                'gmm',
                model_name,
            )

        self.logger.info(f"  ✅ 所有可视化已保存到: {viz_dir}")
    
    def _save_results(self, results: Dict[str, Any], model_name: str) -> None:
        """保存聚类结果"""
        output_dir = self._scheme_output_root() / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        gmm_results = results['gmm_results']
        
        # 1. 保存标签
        if self.config.output['save_labels']:
            np.savez_compressed(
                output_dir / 'gmm_labels.npz',
                labels_1d=gmm_results['labels'],
                labels_3d=gmm_results['labels_3d'],
                spatial_indices=results['spatial_indices']
            )
            self.logger.info(f"  ✅ 保存标签: gmm_labels.npz")
        
        # 2. 保存概率
        if self.config.output['save_probabilities']:
            np.savez_compressed(
                output_dir / 'gmm_probabilities.npz',
                probabilities=gmm_results['probabilities'],
                probabilities_3d=gmm_results['probabilities_3d']
            )
            self.logger.info(f"  ✅ 保存概率: gmm_probabilities.npz")
        
        # 3. 保存NetCDF格式
        if 'nc' in self.config.output['result_formats']:
            self._save_to_netcdf(results, output_dir)
        
        # 4. 保存聚类统计
        if self.config.output['save_cluster_stats']:
            self._save_cluster_statistics(results, output_dir)
        
        # 5. 保存配置和元数据
        def _jsonify(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {str(k): _jsonify(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_jsonify(v) for v in obj]
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (str, bool)) or obj is None:
                return obj
            return str(obj)

        meta_out = {
            'model_name': model_name,
            'version': 'v7.5',
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mode': gmm_results.get('mode', 'full_depth'),
            'n_clusters': int(gmm_results['n_clusters']),
            'features_used': results['metadata']['features'],
            'feature_space': results['preprocessing_info'].get('feature_space'),
            'spatial_range': results['metadata']['spatial_range'],
            'preprocessing': _jsonify(results['preprocessing_info']),
            'gmm_metrics': _jsonify(
                {
                    k: v
                    for k, v in gmm_results['metrics'].items()
                    if not isinstance(v, (GaussianMixture,))
                }
            ),
            'bic_analysis': {
                'optimal_n': int(gmm_results['bic_analysis']['optimal_n']),
                'optimal_bic': float(gmm_results['bic_analysis']['optimal_bic']),
            },
            'band_optimal_k': gmm_results['metrics'].get('band_optimal_k'),
        }

        with open(output_dir / 'analysis_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)

        self.logger.info("  ✅ 保存元数据: analysis_metadata.json")
    
    def _save_to_netcdf(self, results: Dict[str, Any], output_dir: Path) -> None:
        """保存为NetCDF格式"""
        try:
            metadata = results['metadata']
            gmm_results = results['gmm_results']
            probs_3d = gmm_results['probabilities_3d']
            if gmm_results.get('max_probability_3d') is not None:
                max_prob_3d = gmm_results['max_probability_3d']
            else:
                max_prob_3d = np.max(probs_3d, axis=-1)

            ds = xr.Dataset(
                {
                    'cluster_labels': (
                        ['latitude', 'longitude', 'depth'],
                        gmm_results['labels_3d'].astype(np.int16),
                    ),
                    'max_probability': (
                        ['latitude', 'longitude', 'depth'],
                        max_prob_3d.astype(np.float32),
                    ),
                },
                coords={
                    'latitude': metadata['lats'],
                    'longitude': metadata['lons'],
                    'depth': metadata['depths'],
                },
                attrs={
                    'title': f'Velocity Facies GMM — {results["model_name"]}',
                    'n_clusters': int(gmm_results['n_clusters']),
                    'method': gmm_results.get('mode', 'gmm'),
                    'feature_space': results['preprocessing_info'].get(
                        'feature_space', 'absolute'
                    ),
                    'features': ', '.join(metadata['features']),
                    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                },
            )

            # 全深度模式保存各簇概率；分层模式仅保留 max_probability
            if (
                gmm_results.get('mode') != 'depth_stratified'
                and probs_3d.ndim == 4
                and probs_3d.shape[-1] == gmm_results['n_clusters']
            ):
                for i in range(gmm_results['n_clusters']):
                    ds[f'probability_cluster_{i}'] = (
                        ['latitude', 'longitude', 'depth'],
                        probs_3d[:, :, :, i].astype(np.float32),
                    )

            output_file = output_dir / 'gmm_clustering_results.nc'
            ds.to_netcdf(output_file)
            ds.close()
            self.logger.info("  ✅ 保存NetCDF: gmm_clustering_results.nc")

        except Exception as e:
            self.logger.warning(f"  ⚠️ NetCDF保存失败: {e}")
    
    def _save_cluster_statistics(self, results: Dict[str, Any], output_dir: Path) -> None:
        """保存聚类统计信息"""
        gmm_results = results['gmm_results']
        labels = gmm_results['labels']
        X = results['features']
        metadata = results['metadata']
        
        # 计算每个簇的统计
        unique_labels = np.unique(labels[labels >= 0])
        
        stats_data = []
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            cluster_data = X[mask]
            
            # 反归一化
            if self.processor.scaler:
                cluster_data_original = self.processor.scaler.inverse_transform(cluster_data)
            else:
                cluster_data_original = cluster_data
            
            # 计算统计
            stats = {
                'cluster_id': int(cluster_id),
                'n_points': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(labels) * 100)
            }
            
            # 每个特征的统计
            for feat_idx, feat_name in enumerate(metadata['features']):
                feat_data = cluster_data_original[:, feat_idx]
                stats[f'{feat_name}_mean'] = float(np.mean(feat_data))
                stats[f'{feat_name}_std'] = float(np.std(feat_data))
                stats[f'{feat_name}_min'] = float(np.min(feat_data))
                stats[f'{feat_name}_max'] = float(np.max(feat_data))
                stats[f'{feat_name}_median'] = float(np.median(feat_data))
            
            stats_data.append(stats)
        
        # 保存为CSV
        df = pd.DataFrame(stats_data)
        df.to_csv(output_dir / 'cluster_statistics.csv', index=False)
        
        self.logger.info(f"  ✅ 保存统计: cluster_statistics.csv")
    
    def _generate_summary_report(self, success_count: int, failed_models: List[Tuple[str, str]]) -> None:
        """生成总结报告"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📋 分析总结")
        self.logger.info("=" * 80)
        self.logger.info(f"✅ 成功: {success_count} 个模型")
        
        if failed_models:
            self.logger.info(f"❌ 失败: {len(failed_models)} 个模型")
            for model_name, error in failed_models:
                self.logger.info(f"  - {model_name}: {error}")
        
        self.logger.info(f"\n📅 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
        
        # 保存报告
        if self.config.output['create_report'] and self.all_results:
            self._save_summary_report(success_count, failed_models)
    
    def _save_summary_report(self, success_count: int, failed_models: List[Tuple[str, str]]) -> None:
        """保存总结报告"""
        report_dir = self._scheme_output_root()
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f'clustering_summary_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EASTASIA-FWI 单模型速度相聚类总结报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"版本: v7.5\n\n")
            
            f.write(f"成功分析: {success_count} 个模型\n")
            f.write(f"失败分析: {len(failed_models)} 个模型\n\n")
            
            if self.all_results:
                f.write("=" * 80 + "\n")
                f.write("各模型聚类结果\n")
                f.write("=" * 80 + "\n\n")
                
                for model_name, results in self.all_results.items():
                    gmm_results = results['gmm_results']
                    metrics = gmm_results['metrics']
                    
                    f.write(f"\n模型: {model_name}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"聚类数: {gmm_results['n_clusters']}\n")
                    f.write(f"特征: {', '.join(results['metadata']['features'])}\n")
                    f.write(f"数据点数: {len(gmm_results['labels']):,}\n")
                    f.write(f"\n性能指标:\n")
                    f.write(f"  - BIC: {metrics['bic']:.2f}\n")
                    f.write(f"  - 轮廓系数: {metrics.get('silhouette_score', 0):.4f}\n")
                    f.write(f"  - Davies-Bouldin: {metrics.get('davies_bouldin_score', 0):.4f}\n")
                    f.write(f"  - Calinski-Harabasz: {metrics.get('calinski_harabasz_score', 0):.2f}\n")
                    f.write(f"\n概率分析:\n")
                    f.write(f"  - 平均最大概率: {metrics['mean_max_probability']:.4f}\n")
                    f.write(f"  - 高置信度比例: {metrics['high_confidence_ratio']:.2%}\n")
                    f.write(f"  - 平均熵: {metrics['mean_entropy']:.4f}\n")
                    f.write(f"\n计算信息:\n")
                    f.write(f"  - 拟合时间: {metrics['fit_time']:.2f}秒\n")
                    f.write(f"  - 是否收敛: {'是' if metrics['converged'] else '否'}\n")
                    f.write(f"  - 迭代次数: {metrics['n_iter']}\n")
            
            if failed_models:
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("失败的模型\n")
                f.write("=" * 80 + "\n\n")
                for model_name, error in failed_models:
                    f.write(f"- {model_name}\n")
                    f.write(f"  错误: {error}\n\n")
        
        self.logger.info(f"  ✅ 保存总结报告: {report_file}")
    
    def _print_header(self) -> None:
        """打印头部信息"""
        scheme = str(self.config.depth_stratified.get('scheme', 'fixed_3band'))
        bands = (
            self.config.depth_stratified['schemes'][scheme]['bands']
            if scheme in self.config.depth_stratified.get('schemes', {})
            else []
        )
        band_desc = ' / '.join(
            f"{b['name']}(K≤{b.get('fixed_k') or b['max_clusters']}"
            f"{'，先验固定' if b.get('fixed_k') else ''})"
            for b in bands
        ) or '未配置'
        rule = str(self.config.auto_gmm.get('k_selection', 'bic_knee'))
        rule_desc = {
            'bic_knee': 'BIC 拐点（max-distance-to-chord，无阈值）',
            'bic_min': '原始 argmin(BIC)',
        }.get(rule, rule)
        header = f"""
================================================================================
EASTASIA-FWI 单模型速度相聚类  v7.6
================================================================================
  - 数据: 各模型 original.nc（原生 lon/lat，深度 ≤1000 km）
  - 特征: 相对水平平均 1D 参考的 δlnV（消除深度主趋势）
  - 分层方案: {scheme}
  - 分层: {band_desc}
  - 算法: 深度分层 GMM，选 K 判据 = {rule_desc}
  - 结构参考: Slab2 板片 / 地质省 / CN 地块（仅定性叠图）
  - 可视化: 2-3-1 选 K 决策图；2-3-4 为 3×3（第1排地质 / 第2–3排 Slab2）
  - 三维体对比: 2-3-11（聚类体 vs Slab2 dep+thk 体掩膜）
  - 输出: results/model_clustering/{scheme}/
================================================================================
"""
        print(header)


def main() -> int:
    """主函数"""
    try:
        config = ClusteringConfig()

        # ===== 默认科学路径（可按需覆盖）=====
        config.preprocessing['perturbation']['enabled'] = True
        config.depth_stratified['enabled'] = True
        # 测试 Moho 四层：'moho_4band'；固定三层：'fixed_3band'
        config.depth_stratified['scheme'] = 'moho_4band'
        # 选 K 判据：'bic_knee'（BIC 拐点，无阈值，推荐）
        #           'bic_min'（原始 argmin，体元级数据下必贴 K 上界）
        config.auto_gmm['k_selection'] = 'bic_knee'
        # K 搜索上界：拐点位置依赖此上界，须在论文方法学中声明。
        # 四带统一上界便于跨带与跨模型比较。
        for band in config.depth_stratified['schemes']['moho_4band']['bands']:
            band['max_clusters'] = 10
        # 如需基于构造先验直接固定每带 K（跳过 BIC 扫描），取消下面注释：
        # for band, k in zip(
        #     config.depth_stratified['schemes']['moho_4band']['bands'],
        #     [5, 5, 4, 3],  # crust / lithosphere / transition_zone / lower_mantle
        # ):
        #     band['fixed_k'] = k
        config.auto_gmm['covariance_type'] = 'full'
        config.kmeans_comparison['enabled'] = False
        config.visualization['enabled'] = True

        # 速度相默认用各向同性 Vp/Vs
        features_to_use = ['vp', 'vs']

        pipeline = SmartClusteringPipeline(config)
        results = pipeline.run_analysis(features_to_use=features_to_use)

        print(f"\n{'=' * 80}")
        print(f"✅ 分析完成！共处理 {len(results)} 个模型")
        print(f"{'=' * 80}\n")

        for model_name, result in results.items():
            gmm = result['gmm_results']
            print(f"\n模型: {model_name}")
            print(f"  模式: {gmm.get('mode')} / {gmm.get('scheme', gmm['metrics'].get('scheme'))}")
            print(f"  全局簇数: {gmm['n_clusters']}")
            if gmm['metrics'].get('band_optimal_k'):
                print(f"  各带 K: {gmm['metrics']['band_optimal_k']}")
            print(
                f"  平均最大概率: "
                f"{gmm['metrics'].get('mean_max_probability', 0):.4f}"
            )
        return 0

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断执行")
        return 1
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
