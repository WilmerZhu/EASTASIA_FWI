"""
EASTASIA-FWI 单模型速度簇聚类模块 (v8.0)
====================================================================================

科学目标：
- 刻画单模型内部速度簇 / 构造分区（体素级聚类，默认 GMM 正文主路径）
- 切片/剖面/三维体叠绘 USGS Slab2 俯冲板片几何（定性对照，不做定量评分）
- 浅部簇切片叠绘 USGS 地质省 / CN 地块边界（仅定性对照，不做定量符合度）
- Moho 点数据仅用于 moho_4band 空间分带，不参与符合度评价
- 不用于模型间代表性筛选（那是 2_2 CW-SSIM + 模型级聚类的职责）

支持的聚类算法（--algorithm，各算法独立输出目录，互不覆盖）：
- gmm（默认）：高斯混合，软后验，支持投票与融合；K 由 BIC 拐点或分带 fixed_k 先验
- wkmeans：特征加权 k-means（Huang et al., 2005；Hao et al., 2026）；硬指派；
  全量体元拟合，K 由肘部法（Kneedle）定值 + 轮廓系数评估（Nainggolan et al., 2019）
- hdbscan：层次密度聚类（Campello et al., 2013）；K 由密度参数自动决定，不预设
- hierarchical：Ward 凝聚层次聚类；K 沿用分带 fixed_k 先验，子样本拟合 + 全量簇心指派

核心流程：
1. 加载各模型原始 NetCDF（经纬度用原生范围；深度统一截到 1000 km）
2. 特征域（--raw 可切换，两类对照实验）：
   - 默认扰动域：NetCDF 原生 δlnVp、δlnVs = ln(V/V_ref)，V_ref 来自模型自带
     vp0/vs0（EARA2024 的 vp_ref/vs_ref、FWEA23 的 vp0/vs0）；SinoScope1.0 无
     参考场时回退为各深度水平平均 1D 剖面（与 2_1 / 1_5 一致）
   - 原始域（--raw）：标准化 Vp、Vs 绝对值（径向分层 / Hao et al. 对照实验用）
   - 可选坐标增广（--coords）：经度/纬度/深度并入特征（默认关闭，对照实验须声明）
3. 深度分带（两条路径共用 DepthBandPartitioner，保证逐带可比）：
   - fixed_3band: 浅部(0-410) / 过渡带(410-660) / 下地幔(660-1000)
   - moho_4band: 地壳(z<Moho) / 岩石圈(Moho-410) / 过渡带 / 下地幔（默认）
4. 各带独立聚类，带内簇按 Vs（或扰动域 δlnVs）升序编号，带间 label_offset 拼合
5. 质量诊断：
   - GMM：BIC 曲线、划分重现性（k_robustness）、后验概率
   - W-k-means：SSE-轮廓双轴选 K 图、特征权重、可选 β 扫描
   - HDBSCAN / 层次：方法学诊断图（簇规模分布 / 树状图）
   - 全算法：δlnVp-δlnVs（或 Vp-Vs）交会图 + 振幅切分判据 R = σ∥/σ⊥
6. 可视化（5_10_Clustering_visualization.py）：
   2-3-1 GMM 双轴选 K（BIC + silhouette，panel a–d）、3×3 深度切片、
   垂直剖面、簇剖面、Slab2 三维体对照、特征空间交会图等
7. 输出：数据 → results/model_clustering/{scheme|algo_tag}/；
   图件 → figures/model_clustering/{scheme|algo_tag}/

命令行示例：
  python 2_3_Model_clustering.py                                    # GMM 正文路径
  python 2_3_Model_clustering.py --algorithm wkmeans                  # 扰动域 W-k-means
  python 2_3_Model_clustering.py --algorithm wkmeans --raw            # 原始 Vp/Vs
  python 2_3_Model_clustering.py --algorithm hdbscan --models M1 M2   # 密度聚类对照
  python 2_3_Model_clustering.py --algorithm hierarchical               # 层次聚类对照

作者：EASTASIA-FWI Team
日期：2026-09-08
版本：v8.0
"""

import sys
import warnings
import importlib.util
import logging
import argparse
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
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AgglomerativeClustering,
    kmeans_plusplus,
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
import hdbscan
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
principal_axis = _viz_mod.principal_axis
crossplot_diagnostics = _viz_mod.crossplot_diagnostics

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
        # 扰动参考：NetCDF 中 vp→vp0、vs→vs0（1_5 标准化命名）
        'reference_features': {'vp': 'vp0', 'vs': 'vs0'},
    })

    # ==================== 聚类算法选择 ====================
    # 'gmm'           — 高斯混合 + BIC/先验选 K（正文主路径，软后验支持投票与融合）
    # 'wkmeans'       — 特征加权 k-means（Huang et al. 2005；Hao et al. 2026 采用）
    # 'hdbscan'       — 层次密度聚类（Campello et al., 2013）
    # 'hierarchical'  — 凝聚层次聚类 + 树状图切分（Ward linkage）
    # 各路径共用同一套深度分带，结果可逐带对照。
    algorithm: str = 'gmm'

    # ==================== W-k-means 配置 ====================
    wkmeans: Dict[str, Any] = field(default_factory=lambda: {
        'beta': 6.5,           # 与 Hao et al. (2026) 一致；须在 [0,1] 之外
        'n_init': 3,
        'max_iter': 100,
        # None = 用全部体元拟合。W-k-means 每次迭代是 O(n·k·d)，且 _assign
        # 已分块，全量拟合只增加线性时间，不存在内存瓶颈。
        'fit_max_points': None,
        'random_state': RANDOM_SEED,

        # 选 K：肘部法定值 + 轮廓系数评估簇的紧致度与分离度
        # （Nainggolan et al., 2019）
        'k_selection': {
            # 'elbow'            — SSE 曲线 Kneedle 拐点
            # 'elbow_silhouette' — 肘点之后取轮廓系数的局部极大（Hao et al. 2026
            #                      图 S39a 的做法）；本项目数据的轮廓曲线多为单调
            #                      下降，无局部极大时回退到肘点
            # 'sse_threshold'    — 边际 SSE 改善首次低于总降幅的给定比例
            # 'silhouette'       — 轮廓系数全局最大（连续速度场下必压到 K=2，
            #                      仅作诊断，不建议用于选 K）
            # 'fixed'            — 直接指定
            'method': 'elbow',
            # 沿用分带配置的 min/max_clusters，保证与 GMM 路径扫描范围一致
            'use_band_k_range': True,
            'min_clusters': 2,          # use_band_k_range=False 时生效
            'max_clusters': 12,
            # 分带里的 fixed_k 是为 GMM 设的地质省先验，W-k-means 默认忽略
            'respect_fixed_k': False,
            'fixed_k': None,            # method='fixed' 时生效
            # method='sse_threshold' 时生效：边际降幅低于总降幅的该比例即停
            'sse_improvement_threshold': 0.05,
            'scan_max_points': None,     # None = K 扫描也用全部体元
            'scan_n_init': 2,
            # 轮廓系数是 O(n²)，全量计算不可行（本项目单带可达 2.8×10⁶ 体元，
            # 对应 7.8×10¹² 个点对）。改为在多个独立子样本上重复估计，用
            # 估计量的标准差量化抽样不确定度，而非假定子样本有代表性。
            'silhouette_sample_size': 50_000,
            'silhouette_n_repeats': 3,
        },

        # 权重指数 β 的选取：Hao et al. (2026) 图 S39b 以平均轮廓系数扫 β 定值。
        # β→1⁺ 时权重塌缩到单一特征，β 增大权重趋于均分，故存在一个平台区。
        # 默认关闭：扫描开销约等于再跑一遍 K 扫描，只在方法学标定时开启。
        'beta_selection': {
            'enabled': False,
            'beta_min': 1.2,      # β 须在 [0,1] 之外，故从 1.2 起扫
            'beta_max': 10.0,
            'beta_step': 0.4,
            'n_clusters': None,   # None 表示用该带选定的 K
            'scan_max_points': None,
        },

        # 全深度模式（depth_stratified.enabled=False）专用
        'order_by_depth': True,     # 按簇平均深度重排编号，簇序即径向次序
        'radial_analysis': {
            'enabled': True,
            'reference_discontinuities': [410.0, 660.0],
            'section_latitude': 35.0,
        },
    })

    # ==================== HDBSCAN ====================
    hdbscan: Dict[str, Any] = field(default_factory=lambda: {
        # None = max(50, 0.05% × 带内体元数)
        'min_cluster_size': None,
        'min_samples': None,
        'cluster_selection_epsilon': 0.0,
        'metric': 'euclidean',
        # 全带体元过多时在子样本上拟合，再 approximate_predict 赋全量标签
        'fit_max_points': 200_000,
        'assign_noise_to_nearest': True,
        'random_state': RANDOM_SEED,
    })

    # ==================== 层次聚类 ====================
    hierarchical: Dict[str, Any] = field(default_factory=lambda: {
        'linkage': 'ward',          # 'ward' | 'average' | 'complete'
        'n_clusters': None,         # None = 沿用分带 fixed_k
        'fit_max_points': 20_000,   # Ward 为 O(n²)；10 万点/带需 15–30 min，2 万约 1–3 min
        'respect_fixed_k': True,
        'dendrogram_sample': 5000,  # 树状图绘制的子样本规模
        'random_state': RANDOM_SEED,
    })

    # ==================== 预处理配置 ====================
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        # 统一截到 1000 km（含 SinoScope 原始数据更深部分）
        'depth_range': [0, 1000],
        # 扰动域：δlnV = ln(V/V_ref)；优先模型原生 vp0/vs0，缺失时水平平均 1D
        'perturbation': {
            'enabled': True,
            'method': 'dln',  # 'dln' | 'relative'=(V-Vref)/Vref
            'reference': 'model_native',  # 'model_native' | 'horizontal_mean'
        },
        # 坐标增广：把经度/纬度/深度并入特征空间（Hao et al. 2026 的做法）
        #
        # ⚠️ 该选项会改变聚类的性质，默认关闭。加入坐标后距离度量同时包含
        # 空间邻近性，簇被算法强制成空间紧致的块体，此时"簇边界与构造单元
        # 吻合"部分是算法造成的，而非从速度结构中发现的。用于对照实验时
        # 必须在结论中声明这一点。
        'coordinates': {
            'enabled': False,
            # 可选 'longitude' | 'latitude' | 'depth'
            'components': ['longitude', 'latitude', 'depth'],
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
                        'max_clusters': 15,
                        # 地质先验：研究区内 Hasterok et al. (2022) 地质省
                        # prov_type 面积占比 ≥1% 的类别恰为 10 种（火山弧、
                        # 增生杂岩、造山带、洋壳、陆缘/洋内弧后盆地、克拉通、
                        # 地盾、被动陆缘、窄裂谷）
                        'fixed_k': 10,
                    },
                    {
                        'name': 'lithosphere',
                        'mask': 'moho_to_depth',
                        'depth_range': [None, 410],
                        'min_clusters': 2,
                        'max_clusters': 15,
                        # 与地壳带取同一 K，便于跨带比较岩石圈—软流圈过渡。
                        # 注意：该值高于本带分辨率上界（动态范围/模型间 RMS
                        # 差异 = 5.0），属可解释性优先的选择，跨模型稳健的
                        # 类别数仍以约 5 类为准，须在方法学中声明。
                        'fixed_k': 10,
                    },
                    {
                        'name': 'transition_zone',
                        'mask': 'depth_range',
                        'depth_range': [410, 660],
                        'min_clusters': 2,
                        'max_clusters': 10,
                        # 分辨率上界 3.9；取 5 以容纳滞留板片的内部分异
                        'fixed_k': 5,
                    },
                    {
                        'name': 'lower_mantle',
                        'mask': 'depth_range',
                        'depth_range': [660, 1000],
                        'min_clusters': 2,
                        'max_clusters': 10,
                        # 分辨率上界 3.2；取 5 与上覆各带保持可比
                        'fixed_k': 5,
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
        # ── 'bic_knee_silhouette'：BIC 拐点划定下界，其后取轮廓系数局部极大 ──
        # 与 W-k-means 的 elbow_silhouette 同构：左轴 BIC 曲线定复杂度下界，
        # 右轴轮廓系数检验簇分离度；最终 K 供自动流水线使用，图件供人工复核。
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
        # 先验固定 K 时是否仍执行 BIC 扫描。开启后 BIC 曲线仅作诊断进附录，
        # 且其缓存模型可供 K 扫描稳健性复用（k_robustness 开启时本就要拟合
        # 各 K，故此项几乎不增加额外开销）。
        'bic_diagnostic_when_fixed': True,
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
        #
        # ── 默认关闭（2026-09-03）──
        # FWEA23 三设置 A/B 实测（带内相邻体元标签跳变率，相对纯 GMM）：
        #   地壳带   各向同性 横向-21% 垂向-10% ；物理间距 横向-20% 垂向-24%
        #   过渡带   各向同性 横向-17% 垂向-18% ；物理间距 横向-15% 垂向-33%
        # 物理间距加权在横向上并未多去噪，却把垂向平滑放大约一倍，垂/横跳变
        # 比从 6.18 降到 5.84（地壳）、1.75 降到 1.39（过渡带），即压平了模型
        # 真实的成层性。根因是 w = d_ref/d 用采样间距代替了分辨率长度：本区
        # 模型深度采样 10 km、横向约 28 km，而面波主导下垂向分辨率反而更差。
        # 在权重判据确定前默认关闭，保留实现供后续启用。
        'hmrf': {
            'enabled': False,
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
            # 与 5_9 / 三模型 original.nc 公共范围一致；None → 5_10 运行时求交，再退到 BaseConfig 'common'
            'overlap_region': None,
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
            # 簇在 (δlnVs, δlnVp) 平面的交会图 + 振幅切分判据 R
            'cluster_crossplot': True,
            'cluster_crossplot_3d': True,
            # GMM 簇心 Ward 谱系树（附录稳健性 / 分量亲缘解释）
            'centroid_phylogeny': True,
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
        # 特征空间交会图（2-3-12）
        'crossplot': {
            # 参考线斜率随特征域而变，两者含义不同，不可混用：
            #   扰动域 δlnVp = 0.5·δlnVs 为温度主导的热标定关系
            #   原始域 Vp = √3·Vs 对应泊松固体，是岩性判别的经典参考线
            'reference_slope_perturbation': 0.5,
            'reference_slope_raw': 1.732,
            'max_scatter_points': 25_000,
            # 首个面板汇总全部深度带，并标出各带在特征空间中的占位
            'overview_panel': True,
            # 独立图 2-3-13：δlnVs–δlnVp–depth 三维交会
            'perturbation_lim_3d': [-20.0, 20.0],  # 超出范围的体元不绘制
            'max_scatter_points_3d': None,           # None = 范围内全量散点
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

            # 加载扰动参考场（vp0/vs0 等），供 δlnV 计算；不参与聚类特征
            ref_map = self.config.data.get('reference_features', {})
            reference_cube: Dict[str, np.ndarray] = {}
            for feat, ref_name in ref_map.items():
                if ref_name not in ds:
                    continue
                ref_3d = ds[ref_name].values
                if ref_3d.shape[0] == len(lons) and ref_3d.shape[1] == len(lats):
                    ref_3d = ref_3d.transpose(1, 0, 2)
                ref_3d = ref_3d[:, :, depth_mask]
                ref_3d[ref_3d == 9999.0] = np.nan
                reference_cube[feat] = ref_3d.astype(np.float32)
                finite_frac = float(np.isfinite(ref_3d).mean())
                self.logger.info(
                    f"  参考场 {ref_name}→{feat}: 有效占比 {finite_frac:.1%}"
                )
            
            ds.close()
            
            # 构建元数据
            metadata = {
                'model_name': model_name,
                'lats': lats,
                'lons': lons,
                'depths': depths_filtered,
                'features': features_to_load,
                'reference_cube': reference_cube,
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
        self.reference_1d: Dict[str, Any] = {}
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
                data_cube, metadata['depths'], metadata['features'], pert_cfg,
                reference_cube=metadata.get('reference_cube'),
            )
            self.reference_1d = ref_1d
            method = pert_cfg.get('method', 'dln')
            prep_info['feature_space'] = f'perturbation_{method}'
            prep_info['feature_names'] = [
                f'dln_{f}' if method == 'dln' else f'drel_{f}'
                for f in metadata['features']
            ]
            prep_info['preprocessing_steps'].append(
                f"扰动转换({method}, ref={ref_1d.get('_source', pert_cfg.get('reference', 'model_native'))})"
            )
            self.logger.info(f"  特征空间: {prep_info['feature_space']}")

        # 缓存未标准化的 3D 特征立方体（供剖面叠图：δlnVp/δlnVs）
        self.last_feature_cube = np.asarray(working_cube, dtype=np.float32)

        # 向量化提取全部有效点（标签重建必须覆盖全网格，采样只用于 BIC/拟合）
        X, spatial_indices = self._extract_valid_points(working_cube)
        prep_info['n_valid_points'] = len(X)
        self.logger.info(f"  有效数据点: {len(X):,}")

        # 坐标增广（默认关闭；开启后聚类同时受空间邻近性约束）
        coord_cfg = self.config.preprocessing.get('coordinates', {})
        if coord_cfg.get('enabled', False):
            X, coord_names = self._append_coordinates(
                X, spatial_indices, metadata,
                coord_cfg.get('components', ['longitude', 'latitude', 'depth']),
            )
            prep_info['feature_names'] = list(prep_info['feature_names']) + coord_names
            prep_info['preprocessing_steps'].append(
                f"坐标增广({'+'.join(coord_names)})"
            )
            prep_info['coordinates_appended'] = coord_names
            self.logger.info(f"  坐标增广: +{', '.join(coord_names)}")

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
        reference_cube: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        将绝对速度转为 δlnV 或相对扰动。

        参考优先级（reference='model_native' 时）：
        1. NetCDF 原生参考场 vp0/vs0（EARA2024、FWEA23 等）
        2. 各深度水平平均 1D 剖面（SinoScope1.0 等无参考场模型，与 2_1 一致）
        """
        method = pert_cfg.get('method', 'dln')
        ref_mode = pert_cfg.get('reference', 'model_native')
        n_features = data_cube.shape[-1]
        out = np.full_like(data_cube, np.nan, dtype=np.float64)
        ref_1d: Dict[str, Any] = {'depth': np.asarray(depths, dtype=float)}
        reference_cube = reference_cube or {}
        used_native = False
        used_fallback_features: List[str] = []

        for f_idx in range(n_features):
            slab = data_cube[:, :, :, f_idx]
            feat_name = features[f_idx] if f_idx < len(features) else f'f{f_idx}'

            v_ref_3d: Optional[np.ndarray] = None
            if ref_mode == 'model_native' and feat_name in reference_cube:
                candidate = reference_cube[feat_name]
                if np.isfinite(candidate).any():
                    v_ref_3d = candidate
                    used_native = True

            if v_ref_3d is not None:
                v_ref = v_ref_3d
                ref_for_1d = np.nanmean(v_ref, axis=(0, 1))
            else:
                v_ref = np.nanmean(slab, axis=(0, 1))  # (n_depths,)
                v_ref = np.where(np.isfinite(v_ref) & (v_ref > 0), v_ref, np.nan)
                v_ref = np.broadcast_to(
                    v_ref[np.newaxis, np.newaxis, :], slab.shape
                )
                ref_for_1d = v_ref[0, 0, :]
                used_fallback_features.append(feat_name)

            ref_1d[feat_name] = ref_for_1d

            with np.errstate(divide='ignore', invalid='ignore'):
                if method == 'relative':
                    pert = (slab - v_ref) / v_ref
                else:
                    pert = np.log(slab / v_ref)
                pert[~np.isfinite(pert)] = np.nan
            out[:, :, :, f_idx] = pert

            finite = pert[np.isfinite(pert)]
            if finite.size:
                self.logger.info(
                    f"    {feat_name}: δ 范围 "
                    f"[{np.nanpercentile(finite, 1):.3f}, {np.nanpercentile(finite, 99):.3f}]"
                )

        if used_native and not used_fallback_features:
            ref_1d['_source'] = 'model_native'
            self.logger.info("  扰动参考: 模型原生 vp0/vs0")
        elif used_native and used_fallback_features:
            ref_1d['_source'] = 'model_native+horizontal_mean'
            self.logger.info(
                f"  扰动参考: 混合（原生 + 水平平均回退: {', '.join(used_fallback_features)}）"
            )
        else:
            ref_1d['_source'] = 'horizontal_mean'
            self.logger.info("  扰动参考: 各深度水平平均 1D（模型无 vp0/vs0）")

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

    def _append_coordinates(
        self,
        X: np.ndarray,
        spatial_indices: np.ndarray,
        metadata: Dict[str, Any],
        components: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        把空间坐标作为额外特征列拼接到特征矩阵

        坐标取网格实际值（经纬度为度、深度为 km），后续统一标准化，
        因此各坐标的原始量纲差异不会直接转化为权重差异。

        Args:
            X: (n_points, n_features) 特征矩阵
            spatial_indices: (n_points, 3) → [lat_idx, lon_idx, depth_idx]
            metadata: 含 lats / lons / depths 的元数据
            components: 需要拼接的坐标名列表

        Returns:
            (增广后的特征矩阵, 新增列名列表)
        """
        axis_map = {
            'latitude': (np.asarray(metadata['lats'], dtype=float), 0),
            'longitude': (np.asarray(metadata['lons'], dtype=float), 1),
            'depth': (np.asarray(metadata['depths'], dtype=float), 2),
        }
        columns: List[np.ndarray] = []
        names: List[str] = []
        for comp in components:
            key = str(comp).lower()
            if key not in axis_map:
                self.logger.warning(f"    ⚠️ 未知坐标分量 '{comp}'，已跳过")
                continue
            values, idx_col = axis_map[key]
            columns.append(values[spatial_indices[:, idx_col]])
            names.append(key)

        if not columns:
            return X, []
        return np.hstack([X, np.stack(columns, axis=1)]), names

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """特征标准化"""
        method = self.config.preprocessing['normalization']['method']
        self.scaler = StandardScaler() if method == 'standard' else RobustScaler()
        return self.scaler.fit_transform(X)

class WKMeans:
    """
    特征加权 k-means（W-k-means, Huang et al. 2005, IEEE TPAMI 27(5): 657-668）

    在标准 k-means 的基础上，把特征权重 w_j 也作为待优化变量，目标函数为

        P(U, Z, W) = Σ_l Σ_i Σ_j u_il · w_j^β · (x_ij - z_lj)²
        s.t.  Σ_l u_il = 1, u_il ∈ {0,1};  Σ_j w_j = 1, 0 ≤ w_j ≤ 1

    通过交替固定两组变量求另一组的极小值迭代求解，三步均有闭式解，
    目标函数单调不增，因此必然收敛到局部极小（Huang et al. 2005, Thm 1-3）。

    权重闭式解为 w_j ∝ D_j^{-1/(β-1)}，其中 D_j = Σ_l Σ_{i∈l} (x_ij - z_lj)²
    为特征 j 的簇内总离散度。即簇内离散度越大的特征权重越小——该特征对
    当前划分的贡献越弱，越应被抑制。

    参数 β 控制加权强度：β → ∞ 时权重趋于均匀（退化为标准 k-means），
    β → 1⁺ 时权重趋于集中在单一特征。Huang et al. (2005) 证明 β ∈ [0,1]
    时迭代不收敛，故 β 必须取该区间之外的值。

    Args:
        n_clusters: 簇数 K
        beta: 权重调节参数，须满足 beta > 1 或 beta < 0
        max_iter: 最大迭代轮数
        tol: 目标函数相对变化收敛阈值
        n_init: 不同初始化的重复次数，取目标函数最小者
        random_state: 随机种子
        chunk_size: 分配步的分块大小，控制距离矩阵内存占用
    """

    def __init__(
        self,
        n_clusters: int,
        beta: float = 6.5,
        max_iter: int = 100,
        tol: float = 1e-6,
        n_init: int = 5,
        random_state: int = RANDOM_SEED,
        chunk_size: int = 200_000,
    ):
        if 0.0 <= beta <= 1.0:
            raise ValueError(
                f"beta 必须位于 [0, 1] 之外（Huang et al. 2005），当前 beta={beta}"
            )
        self.n_clusters = int(n_clusters)
        self.beta = float(beta)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.random_state = int(random_state)
        self.chunk_size = int(chunk_size)

        self.cluster_centers_: Optional[np.ndarray] = None
        self.feature_weights_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0
        self.converged_: bool = False

    def _assign(
        self, X: np.ndarray, centers: np.ndarray, w_pow: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        分配步：在加权平方欧氏距离下把每个样本指派到最近的簇中心

        Args:
            X: (n_samples, n_features) 特征矩阵
            centers: (n_clusters, n_features) 簇中心
            w_pow: (n_features,) 已取过 β 次幂的特征权重

        Returns:
            (labels, objective) 标签数组与目标函数值
        """
        n = len(X)
        labels = np.empty(n, dtype=np.int32)
        objective = 0.0
        for start in range(0, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            chunk = X[start:stop]
            # (chunk, K)：Σ_j w_j^β (x_ij - z_lj)²
            diff = chunk[:, None, :] - centers[None, :, :]
            dist = np.einsum('nkj,j->nk', diff ** 2, w_pow)
            idx = np.argmin(dist, axis=1)
            labels[start:stop] = idx
            objective += float(dist[np.arange(stop - start), idx].sum())
        return labels, objective

    def _update_centers(
        self, X: np.ndarray, labels: np.ndarray, centers: np.ndarray
    ) -> np.ndarray:
        """更新步：簇中心取簇内均值；空簇保留原中心以免退化"""
        new_centers = centers.copy()
        for l in range(self.n_clusters):
            mask = labels == l
            if np.any(mask):
                new_centers[l] = X[mask].mean(axis=0)
        return new_centers

    def _update_weights(
        self, X: np.ndarray, labels: np.ndarray, centers: np.ndarray
    ) -> np.ndarray:
        """
        权重步：按闭式解 w_j ∝ D_j^{-1/(β-1)} 更新特征权重

        若存在 D_j = 0 的特征（该特征在所有簇内完全无离散），则权重全部
        分配给这些特征，其余置零——这是 Huang et al. (2005) 的边界情形处理。
        """
        n_features = X.shape[1]
        dispersion = np.zeros(n_features, dtype=float)
        for l in range(self.n_clusters):
            mask = labels == l
            if np.any(mask):
                dispersion += ((X[mask] - centers[l]) ** 2).sum(axis=0)

        zero = dispersion <= 0
        if np.any(zero):
            weights = np.zeros(n_features, dtype=float)
            weights[zero] = 1.0 / int(np.count_nonzero(zero))
            return weights

        exponent = 1.0 / (self.beta - 1.0)
        inv = dispersion ** (-exponent)
        return inv / inv.sum()

    def _fit_once(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
        """单次初始化下的完整迭代"""
        n_features = X.shape[1]
        # k-means++ 初始化中心，权重从均匀开始
        centers, _ = kmeans_plusplus(
            X, n_clusters=self.n_clusters,
            random_state=int(rng.integers(0, 2 ** 31 - 1)),
        )
        centers = np.asarray(centers, dtype=float)
        weights = np.full(n_features, 1.0 / n_features, dtype=float)

        prev_obj = np.inf
        labels = np.zeros(len(X), dtype=np.int32)
        converged = False
        n_iter = 0

        for n_iter in range(1, self.max_iter + 1):
            labels, _ = self._assign(X, centers, weights ** self.beta)
            centers = self._update_centers(X, labels, centers)
            weights = self._update_weights(X, labels, centers)
            # 三步更新后重新计算目标函数，保证与最终 (U, Z, W) 一致
            _, obj = self._assign(X, centers, weights ** self.beta)

            if prev_obj < np.inf:
                rel = abs(prev_obj - obj) / max(abs(prev_obj), 1e-300)
                if rel < self.tol:
                    prev_obj = obj
                    converged = True
                    break
            prev_obj = obj

        return labels, centers, weights, float(prev_obj), n_iter, converged

    def fit(self, X: np.ndarray) -> 'WKMeans':
        """
        拟合模型

        Args:
            X: (n_samples, n_features) 特征矩阵，应事先标准化

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X 必须为二维数组，当前 shape={X.shape}")
        if len(X) < self.n_clusters:
            raise ValueError(f"样本数 {len(X)} 少于簇数 {self.n_clusters}")

        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(max(1, self.n_init)):
            result = self._fit_once(X, rng)
            if best is None or result[3] < best[3]:
                best = result

        assert best is not None
        (self.labels_, self.cluster_centers_, self.feature_weights_,
         self.inertia_, self.n_iter_, self.converged_) = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        用已拟合的中心与权重为新样本分配簇标签

        Args:
            X: (n_samples, n_features) 特征矩阵

        Returns:
            (n_samples,) 簇标签
        """
        if self.cluster_centers_ is None or self.feature_weights_ is None:
            raise RuntimeError("WKMeans 尚未拟合，请先调用 fit()")
        labels, _ = self._assign(
            np.asarray(X, dtype=float), self.cluster_centers_,
            self.feature_weights_ ** self.beta,
        )
        return labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """拟合并返回训练样本的簇标签"""
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_


class DepthBandPartitioner:
    """
    深度分带划分器（GMM 与 W-k-means 共用）

    把体元按深度带切分是两条聚类路径共同的前置步骤：绝对速度随深度
    单调增长，不分带时深度趋势会主导整个特征空间，横向结构被淹没。
    带边界可取固定深度，也可取随位置变化的 Moho 面。

    使用者须提供 self.config / self.logger / self.concordance_eval。
    """

    config: 'ClusteringConfig'
    logger: logging.Logger
    concordance_eval: Optional['GeologyConcordanceEvaluator'] = None

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


class WKMeansClusteringOptimizer(DepthBandPartitioner):
    """
    W-k-means 聚类执行器（Huang et al. 2005 特征加权 k-means）

    与 GMM 路径的三点关键差异：

    1. 硬指派，无后验概率。故不产出概率分布图、置信度图与投票权重，
       所有依赖软后验的下游输出在该路径下不可用。
    2. 选 K 用肘部法定值、轮廓系数评估（Nainggolan et al., 2019）。
       k-means 无似然函数，BIC/AIC 无从定义；SSE 随 K 单调下降也不存在
       极值点，只能以曲率拐点定 K，再用簇内紧致度与簇间分离度加以检验。
    3. 特征权重由算法自适应求解并作为结果的一部分输出，权重分布本身即
       "哪个特征在分簇中起作用"的定量答案。

    分带口径与 GMM 路径完全一致（共用 DepthBandPartitioner），保证两条
    路径的结果可以逐带对照，差异只来自算法而非分区方式。
    """

    def __init__(self, config: ClusteringConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.concordance_eval: Optional[GeologyConcordanceEvaluator] = None
        self.feature_inverse_transform: Optional[Any] = None

    # ==================== 选 K：肘部法 + 轮廓系数 ====================

    @staticmethod
    def _elbow_k(k_values: np.ndarray, sse: np.ndarray) -> int:
        """
        肘部法定 K：SSE-K 曲线上离首末端点连线最远的点

        k-means 的 SSE 随 K 单调下降、无极值点，故须以曲率而非极值定 K。
        把 (K, SSE) 归一化到单位方形后取到首末连线的最大垂距（Kneedle
        判据，Satopää et al. 2011），等价于在无量纲曲线上找曲率最大处，
        避免了"下降幅度小于百分之几即停"这类阈值的任意性。

        Args:
            k_values: 升序排列的候选 K
            sse: 对应的簇内平方和

        Returns:
            肘点对应的 K
        """
        k_values = np.asarray(k_values, dtype=float)
        sse = np.asarray(sse, dtype=float)
        if len(k_values) < 3:
            return int(k_values[0])

        span_k = k_values[-1] - k_values[0]
        span_sse = sse[0] - sse[-1]
        if span_k <= 0 or not np.isfinite(span_sse) or span_sse <= 0:
            return int(k_values[0])

        x = (k_values - k_values[0]) / span_k
        y = (sse - sse[-1]) / span_sse
        # 首末点连线为 y = 1 - x，点到该线的距离 ∝ |x + y - 1|
        dist = np.abs(x + y - 1.0) / np.sqrt(2.0)
        return int(k_values[int(np.argmax(dist))])

    def find_optimal_k(
        self,
        X: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        band_name: str = '',
    ) -> Tuple[int, Dict[str, Any]]:
        """
        扫描 K 并选定最优值

        流程遵循 Nainggolan et al. (2019)：先以肘部法在 SSE-K 曲线上定 K，
        再用轮廓系数评估该 K 下簇的紧致度（cohesion）与分离度（separation）。
        轮廓系数作为检验量而非选择量——它在 K 小时系统性偏高，单独用会把
        K 压到 2-3，抹掉需要分辨的结构。两者不一致时按配置的 method 取舍，
        并在日志中记录分歧，供方法学讨论引用。

        Args:
            X: (n_points, n_features) 标准化特征矩阵
            min_clusters: 候选 K 下界
            max_clusters: 候选 K 上界
            band_name: 深度带名（仅用于日志与诊断记录）

        Returns:
            (选定的 K, 诊断字典)
        """
        cfg = self.config.wkmeans
        ksel = cfg['k_selection']
        tag = f"[{band_name}] " if band_name else ""

        k_values = np.arange(int(min_clusters), int(max_clusters) + 1)
        if len(k_values) == 0:
            raise ValueError(f"{tag}K 搜索区间为空")

        X_fit = self._subsample(X, ksel['scan_max_points'], seed_offset=0)

        sse: List[float] = []
        sil: List[float] = []
        sil_sd: List[float] = []
        self.logger.info(
            f"  🔍 {tag}扫描 K={k_values[0]}-{k_values[-1]}"
            f"（拟合点数 {len(X_fit):,}"
            f"{'，全量' if len(X_fit) == len(X) else f'/{len(X):,}'}）"
        )
        t_scan = time.time()
        for k in k_values:
            model = WKMeans(
                n_clusters=int(k),
                beta=float(cfg['beta']),
                n_init=int(ksel['scan_n_init']),
                max_iter=int(cfg['max_iter']),
                random_state=int(cfg['random_state']),
            ).fit(X_fit)
            labels_k = model.predict(X_fit)
            sse.append(float(model.inertia_))
            m, sd = self._silhouette_repeated(X_fit, labels_k)
            sil.append(m)
            sil_sd.append(sd)
            self.logger.info(
                f"    K={k:>2}: SSE={sse[-1]:.5f}  "
                f"silhouette={m:.4f}±{sd:.4f}  ({time.time() - t_scan:.0f}s)"
            )

        sse_arr = np.asarray(sse)
        sil_arr = np.asarray(sil)
        sil_sd_arr = np.asarray(sil_sd)
        k_elbow = self._elbow_k(k_values, sse_arr)
        k_sil = (
            int(k_values[int(np.nanargmax(sil_arr))])
            if np.any(np.isfinite(sil_arr)) else k_elbow
        )

        k_local = self._silhouette_local_maxima(k_values, sil_arr)
        method = str(ksel.get('method', 'elbow')).lower()

        if method == 'silhouette':
            optimal_n = k_sil
        elif method == 'fixed':
            optimal_n = int(ksel.get('fixed_k') or k_elbow)
        elif method == 'elbow_silhouette':
            # Hao et al. (2026) 图 S39a 的做法：肘点划定下界，其后取轮廓系数的
            # 局部极大。连续速度场的轮廓曲线常无局部极大，此时回退到肘点。
            cands = [k for k in k_local if k >= k_elbow]
            optimal_n = (
                max(cands, key=lambda k: sil_arr[k_values == k][0])
                if cands else k_elbow
            )
        elif method == 'sse_threshold':
            optimal_n = self._sse_threshold_k(
                k_values, sse_arr, float(ksel.get('sse_improvement_threshold', 0.05))
            )
        else:
            optimal_n = k_elbow

        sil_at_opt = float(sil_arr[k_values == optimal_n][0])
        agree = k_elbow == k_sil
        self.logger.info(
            f"  📌 {tag}肘部 K={k_elbow}，轮廓全局最优 K={k_sil}"
            f"，轮廓局部极大 K={k_local if k_local else '无'}"
            f" → 采用 K={optimal_n}（{method}），轮廓系数={sil_at_opt:.4f}"
        )

        return optimal_n, {
            'band_name': band_name,
            'k_values': k_values.tolist(),
            'sse': sse_arr.tolist(),
            'silhouette': sil_arr.tolist(),
            'silhouette_sd': sil_sd_arr.tolist(),
            'n_points_scanned': int(len(X_fit)),
            'k_elbow': int(k_elbow),
            'k_silhouette': int(k_sil),
            'k_local_maxima': [int(k) for k in k_local],
            'optimal_n': int(optimal_n),
            'silhouette_at_optimal': sil_at_opt,
            'selection_rule': method,
            'elbow_silhouette_agree': bool(agree),
        }

    @staticmethod
    def _silhouette_local_maxima(
        k_values: np.ndarray, sil: np.ndarray
    ) -> List[int]:
        """
        找出轮廓系数曲线的内部局部极大点

        Hao et al. (2026) 在 SE Tibet 的 5 维特征空间（含经纬度、深度）里，
        轮廓曲线有明确的局部峰，故可据此定 K。空间坐标把数据切成有间隙的
        团块是该峰出现的前提；纯速度特征的连续场点云无此间隙，曲线通常单调
        下降。返回空列表即表明该带不具备这一结构。
        """
        return [
            int(k_values[i])
            for i in range(1, len(sil) - 1)
            if sil[i] > sil[i - 1] and sil[i] > sil[i + 1]
        ]

    @staticmethod
    def _sse_threshold_k(
        k_values: np.ndarray, sse: np.ndarray, threshold: float
    ) -> int:
        """
        取边际 SSE 改善首次低于总降幅给定比例处的 K

        判据为 (SSE[k-1] - SSE[k]) / (SSE[k_min] - SSE[k_max]) < threshold。
        与 Kneedle 相比不依赖曲线整体形状，只看逐步收益，在光滑凸曲线上
        通常给出比拐点更大的 K。
        """
        total = float(sse[0] - sse[-1])
        if total <= 0:
            return int(k_values[0])
        gains = -np.diff(sse) / total
        below = np.where(gains < threshold)[0]
        return int(k_values[below[0] + 1]) if below.size else int(k_values[-1])

    def scan_beta(
        self,
        X: np.ndarray,
        n_clusters: int,
        band_name: str = '',
    ) -> Dict[str, Any]:
        """
        扫描权重指数 β 并按平均轮廓系数定值（Hao et al., 2026 图 S39b）

        β 控制权重分布的锐度：β→1⁺ 时权重塌缩到方差贡献最小的单一特征，
        β 增大则权重趋于均分，中间存在一个轮廓系数的平台区。本项目默认沿用
        原文的 β=6.5，但该值是在 SE Tibet 的 5 维特征空间上标定的，特征数与
        量纲都与本项目不同，直接沿用缺乏依据——本方法即用于自行标定。

        Args:
            X: 标准化特征矩阵
            n_clusters: 扫描时固定的簇数
            band_name: 深度带名（日志用）

        Returns:
            含 beta_values / silhouette / best_beta 的诊断字典
        """
        cfg = self.config.wkmeans
        bcfg = cfg['beta_selection']
        tag = f"[{band_name}] " if band_name else ""

        betas = np.arange(
            float(bcfg['beta_min']),
            float(bcfg['beta_max']) + 1e-9,
            float(bcfg['beta_step']),
        )
        X_fit = self._subsample(X, bcfg['scan_max_points'], seed_offset=1)
        self.logger.info(
            f"  🔍 {tag}扫描 β={betas[0]:.1f}-{betas[-1]:.1f}"
            f"（{len(betas)} 点，K={n_clusters}）"
        )

        sils: List[float] = []
        weights: List[List[float]] = []
        for b in betas:
            try:
                m = WKMeans(
                    n_clusters=int(n_clusters), beta=float(b),
                    n_init=1, max_iter=int(cfg['max_iter']),
                    random_state=int(cfg['random_state']),
                ).fit(X_fit)
                lab = m.predict(X_fit)
                sils.append(self._silhouette_repeated(X_fit, lab)[0])
                weights.append(np.asarray(m.feature_weights_).tolist())
            except Exception as e:
                self.logger.debug(f"    β={b:.1f} 失败: {e}")
                sils.append(float('nan'))
                weights.append([])

        sil_arr = np.asarray(sils)
        best_beta = float(betas[int(np.nanargmax(sil_arr))])
        self.logger.info(
            f"  📌 {tag}最优 β={best_beta:.1f}"
            f"（轮廓系数={np.nanmax(sil_arr):.4f}；当前采用 β={cfg['beta']}）"
        )
        return {
            'band_name': band_name,
            'n_clusters': int(n_clusters),
            'beta_values': betas.tolist(),
            'silhouette': sil_arr.tolist(),
            'feature_weights': weights,
            'best_beta': best_beta,
            'beta_in_use': float(cfg['beta']),
        }

    # ==================== 单次聚类 ====================

    def _subsample(
        self, X: np.ndarray, max_points: Optional[int], seed_offset: int = 0
    ) -> np.ndarray:
        """
        按需抽样以控制拟合开销

        max_points 为 None 或 0 时返回全量数据——这是本项目的默认设置：
        W-k-means 的每次迭代是 O(n·k·d)，全量拟合只增加线性时间。
        """
        if not max_points or len(X) <= int(max_points):
            return X
        rng = np.random.default_rng(
            int(self.config.wkmeans['random_state']) + seed_offset
        )
        return X[rng.choice(len(X), int(max_points), replace=False)]

    def _silhouette_repeated(
        self, X: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, float]:
        """
        在多个独立子样本上重复估计轮廓系数

        轮廓系数需要全部点对距离，复杂度 O(n²)，单带体元数达百万量级时
        无法全量计算。重复抽样给出估计量的均值与标准差：标准差远小于不同
        K 之间的差异时，即可确证子样本规模足以支撑选 K 判断。

        Args:
            X: 特征矩阵（全量）
            labels: 对应的簇标签（全量）

        Returns:
            (均值, 标准差)；无法计算时返回 (nan, nan)
        """
        ksel = self.config.wkmeans['k_selection']
        size = min(int(ksel['silhouette_sample_size']), len(X))
        n_rep = int(ksel.get('silhouette_n_repeats', 1))
        vals: List[float] = []
        for r in range(n_rep):
            try:
                vals.append(float(silhouette_score(
                    X, labels, sample_size=size,
                    random_state=RANDOM_SEED + r,
                )))
            except Exception as e:
                self.logger.debug(f"    轮廓系数第 {r} 次估计失败: {e}")
        if not vals:
            return float('nan'), float('nan')
        return float(np.mean(vals)), float(np.std(vals))

    def perform_clustering(
        self,
        X: np.ndarray,
        n_clusters: int,
        band_name: str = '',
        sort_col: Optional[int] = None,
    ) -> Tuple[np.ndarray, WKMeans, Dict[str, Any]]:
        """
        以给定 K 执行一次 W-k-means

        Args:
            X: 标准化特征矩阵
            n_clusters: 簇数
            band_name: 深度带名（日志用）
            sort_col: 簇编号排序依据的特征列（通常为 Vs 列）；None 则不排序

        Returns:
            (标签, 已拟合模型, 指标字典)
        """
        cfg = self.config.wkmeans
        tag = f"[{band_name}] " if band_name else ""
        t0 = time.time()

        X_fit = self._subsample(X, cfg['fit_max_points'])
        model = WKMeans(
            n_clusters=int(n_clusters),
            beta=float(cfg['beta']),
            n_init=int(cfg['n_init']),
            max_iter=int(cfg['max_iter']),
            random_state=int(cfg['random_state']),
        ).fit(X_fit)
        labels = model.predict(X)

        if sort_col is not None:
            labels = self._order_clusters(labels, model, int(sort_col))

        fit_time = time.time() - t0
        metrics = self._compute_metrics(X, labels, model, fit_time)
        self.logger.info(
            f"  ✅ {tag}K={n_clusters}  N={len(X_fit):,}"
            f"  轮廓系数={metrics['silhouette_score']:.4f}"
            f"±{metrics['silhouette_score_sd']:.4f}"
            f"  收敛={'是' if model.converged_ else '否'}"
            f"  迭代={model.n_iter_}  用时={fit_time:.1f}s"
        )
        return labels, model, metrics

    @staticmethod
    def _order_clusters(
        labels: np.ndarray, model: WKMeans, sort_col: int
    ) -> np.ndarray:
        """
        按簇中心在指定特征列上的取值升序重排簇编号（就地同步模型参数）

        k-means 的簇索引由初始化随机决定，本身不含信息。按 Vs 升序重排后
        C0 = 最慢、C_{K-1} = 最快，编号即携带物理含义，配色可固定，跨带与
        跨模型的簇也能直接对应。标准化是单调仿射变换，故在标准化域上排序
        等价于在物理速度上排序。
        """
        centers = np.asarray(model.cluster_centers_)
        if sort_col >= centers.shape[1]:
            return labels

        order = np.argsort(centers[:, sort_col], kind='stable')
        if np.array_equal(order, np.arange(order.size)):
            return labels

        remap = np.empty(order.size, dtype=labels.dtype)
        remap[order] = np.arange(order.size, dtype=labels.dtype)
        model.cluster_centers_ = centers[order]
        return remap[labels]

    def _compute_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        model: WKMeans,
        fit_time: float,
    ) -> Dict[str, Any]:
        """
        计算聚类质量指标

        Davies-Bouldin 与 Calinski-Harabasz 只需簇心与簇内离散度，复杂度
        O(n·k)，故在全量体元上计算。轮廓系数需全部点对距离，复杂度 O(n²)，
        改用重复子样本估计并同时报告标准差。
        """
        metrics: Dict[str, Any] = {
            'inertia': float(model.inertia_),
            'converged': bool(model.converged_),
            'n_iter': int(model.n_iter_),
            'fit_time': float(fit_time),
            'n_points': int(len(X)),
            'feature_weights': np.asarray(model.feature_weights_).tolist(),
        }
        for name, fn in (
            ('davies_bouldin_score', davies_bouldin_score),
            ('calinski_harabasz_score', calinski_harabasz_score),
        ):
            try:
                metrics[name] = float(fn(X, labels))
            except Exception as e:
                self.logger.warning(f"    ⚠️ {name} 计算失败: {e}")
                metrics[name] = float('nan')

        sil_mean, sil_sd = self._silhouette_repeated(X, labels)
        metrics['silhouette_score'] = sil_mean
        metrics['silhouette_score_sd'] = sil_sd

        _, counts = np.unique(labels[labels >= 0], return_counts=True)
        metrics['cluster_balance'] = (
            float(np.std(counts) / np.mean(counts)) if counts.size else float('nan')
        )
        return metrics

    # ==================== 深度分带 ====================

    def perform_depth_stratified_clustering(
        self,
        X: np.ndarray,
        spatial_indices: np.ndarray,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        按深度带分别做 W-k-means，再拼回全局标签

        分带与 GMM 路径共用同一套带定义与 Moho 面，故两条路径的结果可以
        逐带对照。每带独立选 K（肘部法 + 轮廓系数），带内簇按 Vs 升序编号，
        带间用 label_offset 隔开以保证全局标签唯一。

        Args:
            X: (n_points, n_features) 标准化特征矩阵
            spatial_indices: (n_points, 3) → [lat_idx, lon_idx, depth_idx]
            metadata: 含 lats / lons / depths / dims / features 的元数据

        Returns:
            聚类结果字典（mode='depth_stratified'）
        """
        depths = np.asarray(metadata['depths'], dtype=float)
        dims = metadata['dims']

        feature_names = [str(f).lower() for f in metadata.get('features', [])]
        sort_col = feature_names.index('vs') if 'vs' in feature_names else None
        if sort_col is None and feature_names:
            sort_col = 0
            self.logger.warning("  ⚠️ 特征中无 vs，簇排序退回首个特征列")

        scheme, scheme_cfg = self._active_scheme()
        bands = scheme_cfg['bands']
        offset_labels = self.config.depth_stratified.get(
            'offset_labels_across_bands', True
        )
        ksel = self.config.wkmeans['k_selection']

        self.logger.info(f"  📚 分层方案: {scheme}（{len(bands)} 带）")

        needs_moho = any(
            b.get('mask') in ('above_moho', 'moho_to_depth') for b in bands
        )
        moho_z = (
            self._moho_depth_per_point(spatial_indices, metadata)
            if needs_moho else None
        )

        global_labels = np.full(len(X), -1, dtype=np.int32)
        per_band: Dict[str, Any] = {}
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
                self.logger.warning(f"  ⚠️ 深度带 {name} 有效点过少 ({n_band})，跳过")
                continue

            X_band = X[mask]

            # K 搜索区间：默认沿用分带配置，保证与 GMM 路径的扫描范围一致
            if ksel.get('use_band_k_range', True):
                k_lo = int(band['min_clusters'])
                k_hi = int(band['max_clusters'])
            else:
                k_lo = int(ksel['min_clusters'])
                k_hi = int(ksel['max_clusters'])

            # 分带配置里的 fixed_k 是为 GMM 设的地质省先验，W-k-means 路径
            # 默认忽略它，由肘部法自行定 K；置 respect_fixed_k=True 可沿用。
            fixed_k = band.get('fixed_k') if ksel.get('respect_fixed_k', False) else None
            if fixed_k:
                optimal_n = int(fixed_k)
                k_selection = {
                    'band_name': name,
                    'optimal_n': optimal_n,
                    'selection_rule': 'fixed_k',
                }
                self.logger.info(f"  📌 [{name}] 先验固定 K={optimal_n}")
            else:
                optimal_n, k_selection = self.find_optimal_k(
                    X_band, k_lo, k_hi, band_name=name
                )

            if self.config.wkmeans['beta_selection'].get('enabled', False):
                bsel = self.config.wkmeans['beta_selection']
                k_selection['beta_scan'] = self.scan_beta(
                    X_band,
                    n_clusters=int(bsel.get('n_clusters') or optimal_n),
                    band_name=name,
                )

            labels_b, model_b, metrics_b = self.perform_clustering(
                X_band, n_clusters=optimal_n, band_name=name, sort_col=sort_col
            )
            total_fit_time += float(metrics_b.get('fit_time', 0.0))

            if offset_labels:
                labels_global_b = labels_b + label_offset
                label_offset += optimal_n
            else:
                labels_global_b = labels_b

            global_labels[mask] = labels_global_b

            per_band[name] = {
                'depth_range': [float(z_ref[0]), float(z_ref[1])],
                'mask_type': band.get('mask', 'depth_range'),
                'description': desc,
                'n_points': n_band,
                'n_clusters': optimal_n,
                'labels_local': labels_b,
                'labels_global': labels_global_b,
                'model': model_b,
                'metrics': metrics_b,
                'k_selection': k_selection,
                'feature_weights': np.asarray(model_b.feature_weights_),
                'cluster_centers': self._to_physical(model_b.cluster_centers_),
                'label_offset': label_offset - optimal_n if offset_labels else 0,
                'point_mask': mask,
            }

        if not per_band:
            raise ValueError("所有深度带均无法完成聚类")

        n_global = int(global_labels.max()) + 1 if global_labels.max() >= 0 else 1
        labels_3d = Reconstruction3DHelper.reconstruct_3d_labels(
            global_labels, spatial_indices, dims
        )
        valid = global_labels >= 0

        overall_metrics = {
            'n_clusters': n_global,
            'n_bands': len(per_band),
            'scheme': scheme,
            'fit_time': total_fit_time,
            'band_optimal_k': {k: v['n_clusters'] for k, v in per_band.items()},
            'band_feature_weights': {
                k: v['feature_weights'].tolist() for k, v in per_band.items()
            },
            'converged': all(
                v['metrics'].get('converged', False) for v in per_band.values()
            ),
            'n_iter': int(
                np.sum([v['metrics'].get('n_iter', 0) for v in per_band.values()])
            ),
            'inertia': float(
                np.nansum([v['metrics'].get('inertia', np.nan) for v in per_band.values()])
            ),
            'cluster_sizes': {
                int(k): int(v)
                for k, v in zip(*np.unique(global_labels[valid], return_counts=True))
            },
        }
        for key in ('silhouette_score', 'davies_bouldin_score',
                    'calinski_harabasz_score', 'cluster_balance'):
            overall_metrics[key] = float(np.nanmean(
                [v['metrics'].get(key, np.nan) for v in per_band.values()]
            ))

        self.logger.info(
            f"  📊 全局: {n_global} 簇 / {len(per_band)} 带，"
            f"各带 K={overall_metrics['band_optimal_k']}，"
            f"平均轮廓系数={overall_metrics['silhouette_score']:.4f}"
        )

        return {
            'algorithm': 'wkmeans',
            'n_clusters': n_global,
            'labels': global_labels,
            'labels_3d': labels_3d,
            'probabilities': None,
            'probabilities_3d': None,
            'model': None,
            'metrics': overall_metrics,
            'per_band': per_band,
            'k_selection': {k: v['k_selection'] for k, v in per_band.items()},
            'mode': 'depth_stratified',
            'scheme': scheme,
        }

    # ==================== 全深度（径向分层） ====================

    def perform_full_depth_clustering(
        self,
        X: np.ndarray,
        spatial_indices: np.ndarray,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        0-1000 km 一次性聚类，簇按平均深度重排为径向序列

        不分带时绝对速度的深度趋势主导特征空间，簇必然构成径向壳层。这不是
        缺陷而是另一种用法：层界深度是有量纲的物理量，其横向起伏可直接与
        410/660 km 相变面对照，且不受簇编号任意性影响，跨模型可比。

        Args:
            X: 标准化特征矩阵
            spatial_indices: (n_points, 3) 空间索引
            metadata: 含 depths / dims / features 的元数据

        Returns:
            聚类结果字典（mode='full_depth'），另含 feature_weights 与 radial_stats
        """
        cfg = self.config.wkmeans
        ksel = cfg['k_selection']

        if str(ksel.get('method', 'elbow')).lower() == 'fixed':
            k = int(ksel.get('fixed_k') or cfg['n_clusters'])
            k_selection = {'optimal_n': k, 'selection_rule': 'fixed_k'}
        else:
            k, k_selection = self.find_optimal_k(
                X, int(ksel['min_clusters']), int(ksel['max_clusters'])
            )

        labels, model, metrics = self.perform_clustering(X, n_clusters=k)

        depths = np.asarray(metadata['depths'], dtype=float)
        point_depth = depths[spatial_indices[:, 2]]
        centers = np.asarray(model.cluster_centers_, dtype=float)
        if cfg.get('order_by_depth', True):
            labels, centers = self._order_by_depth(labels, centers, point_depth, k)

        centers_physical = self._to_physical(centers)
        radial_stats = self._radial_stats(
            labels, centers_physical, point_depth, metadata, k
        )
        labels_3d = Reconstruction3DHelper.reconstruct_3d_labels(
            labels, spatial_indices, metadata['dims']
        )

        weights = np.asarray(model.feature_weights_, dtype=float)
        metrics['band_optimal_k'] = None
        self.logger.info(
            "    特征权重: " + '  '.join(
                f'{n}={w:.3f}' for n, w in zip(
                    metadata.get('feature_display_names',
                                 metadata.get('features', [])), weights)
            )
        )

        return {
            'algorithm': 'wkmeans',
            'n_clusters': k,
            'labels': labels,
            'labels_3d': labels_3d,
            'probabilities': None,
            'probabilities_3d': None,
            'model': model,
            'metrics': metrics,
            'per_band': None,
            'k_selection': {'all': k_selection},
            'mode': 'full_depth',
            'feature_weights': weights,
            'cluster_centers': centers_physical,
            'radial_stats': radial_stats,
        }

    def _to_physical(self, centers: Any) -> np.ndarray:
        """把标准化域的簇中心反变换回物理量纲，供解释与作图"""
        centers = np.asarray(centers, dtype=float)
        if self.feature_inverse_transform is None:
            return centers
        try:
            return self.feature_inverse_transform(centers)
        except Exception as e:
            self.logger.warning(f"    ⚠️ 簇中心反标准化失败: {e}")
            return centers

    @staticmethod
    def _order_by_depth(
        labels: np.ndarray,
        centers: np.ndarray,
        point_depth: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """按簇平均深度重排簇编号，0 为最浅"""
        mean_depth = np.array([
            point_depth[labels == l].mean() if np.any(labels == l) else np.inf
            for l in range(k)
        ])
        order = np.argsort(mean_depth)
        remap = np.empty(k, dtype=labels.dtype)
        remap[order] = np.arange(k, dtype=labels.dtype)
        return remap[labels], centers[order]

    def _radial_stats(
        self,
        labels: np.ndarray,
        centers_physical: np.ndarray,
        point_depth: np.ndarray,
        metadata: Dict[str, Any],
        k: int,
    ) -> pd.DataFrame:
        """
        统计各簇的深度占位与物理量纲簇中心

        Returns:
            含 cluster / 各特征中心 / depth_p05 / depth_mean / depth_p95 /
            volume_frac 的数据表，按径向次序排列
        """
        names = list(
            metadata.get('feature_display_names')
            or metadata.get('features', [])
        )
        rows: List[Dict[str, Any]] = []
        for l in range(k):
            m = labels == l
            if not np.any(m):
                continue
            zl = point_depth[m]
            row: Dict[str, Any] = {'cluster': l}
            for j, name in enumerate(names[:centers_physical.shape[1]]):
                row[f'center_{name}'] = float(centers_physical[l, j])
            row.update({
                'depth_p05': float(np.percentile(zl, 5)),
                'depth_mean': float(zl.mean()),
                'depth_p95': float(np.percentile(zl, 95)),
                'volume_frac': float(m.mean()),
            })
            rows.append(row)
        return pd.DataFrame(rows)


# ==================== 密度/层次聚类共用工具 ====================

def _clustering_subsample(
    X: np.ndarray,
    max_points: Optional[int],
    seed: int,
    seed_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按需子采样并返回 (子样本, 在原数组中的索引)

    max_points 为 None 或 0 时返回全量。
    """
    if not max_points or len(X) <= int(max_points):
        return X, np.arange(len(X), dtype=np.int64)
    rng = np.random.default_rng(int(seed) + seed_offset)
    idx = rng.choice(len(X), int(max_points), replace=False)
    return X[idx], idx


def _sort_labels_by_feature(
    X: np.ndarray,
    labels: np.ndarray,
    sort_col: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """按指定特征列的簇均值升序重排标签"""
    valid = labels >= 0
    uniq = np.unique(labels[valid])
    if uniq.size == 0:
        return labels, np.zeros((0, X.shape[1]), dtype=float)
    means = np.array([
        X[labels == c, sort_col].mean() if np.any(labels == c) else np.inf
        for c in uniq
    ])
    order = np.argsort(means)
    remap = {int(old): int(new) for new, old in enumerate(uniq[order])}
    new_labels = labels.copy()
    for old, new in remap.items():
        new_labels[labels == old] = new
    centers = np.array([
        X[new_labels == new].mean(axis=0) for new in range(len(uniq))
    ])
    return new_labels, centers


def _assign_nearest_centroid(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """将每个点指派到最近簇心"""
    if centers.size == 0:
        return np.full(len(X), -1, dtype=np.int32)
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(np.int32)


def compute_cross_algorithm_ari(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    per_band: Dict[str, Any],
) -> Dict[str, Any]:
    """
    逐深度带比较两套硬指派标签的分区一致性（ARI / NMI）。

    标签编号任意，ARI/NMI 对置换不变；不要用原始 label 相等比例代替 ARI。
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f'标签长度不一致: {len(labels_a)} vs {len(labels_b)}'
        )

    results: Dict[str, Any] = {'overall': {}, 'bands': {}}
    valid = (labels_a >= 0) & (labels_b >= 0)
    if np.any(valid):
        results['overall'] = {
            'ari': float(adjusted_rand_score(labels_a[valid], labels_b[valid])),
            'nmi': float(normalized_mutual_info_score(labels_a[valid], labels_b[valid])),
            'n_points': int(np.sum(valid)),
        }

    for bname, binfo in per_band.items():
        pmask = binfo.get('point_mask')
        if pmask is None:
            continue
        pmask = np.asarray(pmask, dtype=bool)
        m = pmask & valid
        if not np.any(m):
            continue
        la, lb = labels_a[m], labels_b[m]
        results['bands'][bname] = {
            'ari': float(adjusted_rand_score(la, lb)),
            'nmi': float(normalized_mutual_info_score(la, lb)),
            'n_points': int(np.sum(m)),
            'n_clusters_a': int(len(np.unique(la))),
            'n_clusters_b': int(len(np.unique(lb))),
        }
    return results


class AlternativeClusteringBase(DepthBandPartitioner):
    """
    HDBSCAN / 层次聚类共用基类

    提供子采样、轮廓系数估计、分带编排与标签拼合，子类只需实现单带聚类逻辑。
    """

    algorithm_name: str = 'alt'

    def __init__(self, config: ClusteringConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.concordance_eval: Optional[GeologyConcordanceEvaluator] = None
        self.feature_inverse_transform: Optional[Any] = None

    def _sil_cfg(self) -> Dict[str, Any]:
        """轮廓系数子样本配置（与 W-k-means 共用 wkmeans.k_selection 项）"""
        return self.config.wkmeans['k_selection']

    def _silhouette_repeated(
        self, X: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, float]:
        ksel = self._sil_cfg()
        valid = labels >= 0
        if int(np.sum(valid)) < 3 or len(np.unique(labels[valid])) < 2:
            return float('nan'), float('nan')
        size = min(int(ksel['silhouette_sample_size']), int(np.sum(valid)))
        n_rep = int(ksel.get('silhouette_n_repeats', 1))
        vals: List[float] = []
        for r in range(n_rep):
            try:
                vals.append(float(silhouette_score(
                    X[valid], labels[valid],
                    sample_size=size,
                    random_state=RANDOM_SEED + r,
                )))
            except Exception as e:
                self.logger.debug(f"    轮廓系数估计失败: {e}")
        if not vals:
            return float('nan'), float('nan')
        return float(np.mean(vals)), float(np.std(vals))

    def _compute_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        fit_time: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            'fit_time': float(fit_time),
            'n_points': int(len(X)),
        }
        valid = labels >= 0
        n_clu = len(np.unique(labels[valid])) if np.any(valid) else 0
        metrics['n_clusters_found'] = int(n_clu)
        metrics['noise_fraction'] = float(np.mean(labels < 0)) if len(labels) else 0.0

        for name, fn in (
            ('davies_bouldin_score', davies_bouldin_score),
            ('calinski_harabasz_score', calinski_harabasz_score),
        ):
            try:
                if n_clu >= 2 and np.sum(valid) >= n_clu + 1:
                    metrics[name] = float(fn(X[valid], labels[valid]))
                else:
                    metrics[name] = float('nan')
            except Exception as e:
                self.logger.debug(f"    {name} 失败: {e}")
                metrics[name] = float('nan')

        sil_m, sil_s = self._silhouette_repeated(X, labels)
        metrics['silhouette_score'] = sil_m
        metrics['silhouette_score_sd'] = sil_s

        if np.any(valid):
            uniq, counts = np.unique(labels[valid], return_counts=True)
            metrics['cluster_balance'] = float(np.std(counts) / np.mean(counts))
            metrics['cluster_sizes'] = {
                int(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist())
            }
        else:
            metrics['cluster_balance'] = float('nan')
            metrics['cluster_sizes'] = {}

        if extra:
            metrics.update(extra)
        return metrics

    def _band_target_k(self, band: Dict[str, Any]) -> Optional[int]:
        """读取分带 fixed_k 先验（若配置允许）"""
        cfg = getattr(self.config, self.algorithm_name, {})
        if not cfg.get('respect_fixed_k', True):
            return None
        fk = band.get('fixed_k')
        return int(fk) if fk else None

    def _cluster_band(
        self,
        X: np.ndarray,
        spatial_indices: np.ndarray,
        metadata: Dict[str, Any],
        band: Dict[str, Any],
        band_name: str,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError

    def perform_depth_stratified_clustering(
        self,
        X: np.ndarray,
        spatial_indices: np.ndarray,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """按深度带分别聚类并拼合全局标签"""
        depths = np.asarray(metadata['depths'], dtype=float)
        dims = metadata['dims']
        feature_names = [str(f).lower() for f in metadata.get('features', [])]
        sort_col = feature_names.index('vs') if 'vs' in feature_names else 0

        scheme, scheme_cfg = self._active_scheme()
        bands = scheme_cfg['bands']
        offset_labels = self.config.depth_stratified.get(
            'offset_labels_across_bands', True
        )

        self.logger.info(f"  📚 分层方案: {scheme}（{len(bands)} 带）")

        needs_moho = any(
            b.get('mask') in ('above_moho', 'moho_to_depth') for b in bands
        )
        moho_z = (
            self._moho_depth_per_point(spatial_indices, metadata)
            if needs_moho else None
        )

        global_labels = np.full(len(X), -1, dtype=np.int32)
        per_band: Dict[str, Any] = {}
        method_diagnostics: Dict[str, Any] = {}
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
            if n_band < max(int(band.get('min_clusters', 2)) * 10, 50):
                self.logger.warning(f"  ⚠️ 深度带 {name} 有效点过少 ({n_band})，跳过")
                continue

            X_band = X[mask]
            si_band = spatial_indices[mask]
            labels_b, diag_b, metrics_b = self._cluster_band(
                X_band, si_band, metadata, band, name
            )
            labels_b, centers_b = _sort_labels_by_feature(X_band, labels_b, sort_col)
            n_clu = int(len(np.unique(labels_b[labels_b >= 0])))
            total_fit_time += float(metrics_b.get('fit_time', 0.0))

            if offset_labels and n_clu > 0:
                lb = labels_b.copy()
                lb[lb >= 0] += label_offset
                label_offset += n_clu
            else:
                lb = labels_b

            global_labels[mask] = lb
            per_band[name] = {
                'depth_range': [float(z_ref[0]), float(z_ref[1])],
                'mask_type': band.get('mask', 'depth_range'),
                'description': desc,
                'n_points': n_band,
                'n_clusters': n_clu,
                'labels_local': labels_b,
                'labels_global': lb,
                'metrics': metrics_b,
                'cluster_centers': centers_b,
                'label_offset': label_offset - n_clu if offset_labels else 0,
                'point_mask': mask,
                'method_diagnostics': diag_b,
            }
            method_diagnostics[name] = diag_b

        if not per_band:
            raise ValueError("所有深度带均无法完成聚类")

        valid = global_labels >= 0
        n_global = int(global_labels[valid].max()) + 1 if np.any(valid) else 0
        labels_3d = Reconstruction3DHelper.reconstruct_3d_labels(
            global_labels, spatial_indices, dims
        )

        overall_metrics = {
            'n_clusters': n_global,
            'n_bands': len(per_band),
            'scheme': scheme,
            'fit_time': total_fit_time,
            'band_n_clusters': {k: v['n_clusters'] for k, v in per_band.items()},
            'cluster_sizes': {
                int(k): int(v)
                for k, v in zip(*np.unique(global_labels[valid], return_counts=True))
            } if np.any(valid) else {},
        }
        for key in (
            'silhouette_score', 'davies_bouldin_score',
            'calinski_harabasz_score', 'cluster_balance', 'noise_fraction',
        ):
            overall_metrics[key] = float(np.nanmean([
                v['metrics'].get(key, np.nan) for v in per_band.values()
            ]))

        self.logger.info(
            f"  📊 全局: {n_global} 簇 / {len(per_band)} 带，"
            f"各带 K={overall_metrics['band_n_clusters']}，"
            f"平均轮廓系数={overall_metrics['silhouette_score']:.4f}"
        )

        return {
            'algorithm': self.algorithm_name,
            'n_clusters': n_global,
            'labels': global_labels,
            'labels_3d': labels_3d,
            'probabilities': None,
            'probabilities_3d': None,
            'model': None,
            'metrics': overall_metrics,
            'per_band': per_band,
            'method_diagnostics': method_diagnostics,
            'mode': 'depth_stratified',
            'scheme': scheme,
        }


class HDBSCANClusteringOptimizer(AlternativeClusteringBase):
    """
    HDBSCAN 密度聚类（Campello et al., 2013）

    自动确定簇数并标记噪声点。大带在子样本上拟合，全量标签用
    approximate_predict 或最近簇心指派。
    """

    algorithm_name = 'hdbscan'

    def _cluster_band(
        self,
        X: np.ndarray,
        spatial_indices: np.ndarray,
        metadata: Dict[str, Any],
        band: Dict[str, Any],
        band_name: str,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        cfg = self.config.hdbscan
        t0 = time.time()
        n_band = len(X)
        mcs = cfg.get('min_cluster_size')
        if mcs is None:
            mcs = max(50, int(round(n_band * 0.0005)))
        ms = cfg.get('min_samples')
        if ms is None:
            ms = max(10, mcs // 5)

        X_fit, fit_idx = _clustering_subsample(
            X, cfg.get('fit_max_points'), int(cfg.get('random_state', RANDOM_SEED))
        )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(mcs),
            min_samples=int(ms),
            metric=str(cfg.get('metric', 'euclidean')),
            cluster_selection_epsilon=float(cfg.get('cluster_selection_epsilon', 0.0)),
            core_dist_n_jobs=1,
        )
        fit_labels = clusterer.fit_predict(X_fit)

        if len(X_fit) < len(X):
            try:
                labels, _ = hdbscan.approximate_predict(clusterer, X)
            except Exception:
                valid = fit_labels >= 0
                if np.any(valid):
                    centers = np.array([
                        X_fit[fit_labels == c].mean(axis=0)
                        for c in np.unique(fit_labels[valid])
                    ])
                    labels = _assign_nearest_centroid(X, centers)
                    # 重映射到 0..k-1
                    uniq = np.unique(labels)
                    remap = {int(o): i for i, o in enumerate(uniq)}
                    labels = np.array([remap[int(l)] for l in labels], dtype=np.int32)
                else:
                    labels = np.zeros(len(X), dtype=np.int32)
        else:
            labels = fit_labels.astype(np.int32)

        n_noise = int(np.sum(labels < 0))
        if n_noise and cfg.get('assign_noise_to_nearest', True):
            valid = labels >= 0
            if np.any(valid):
                centers = np.array([
                    X[labels == c].mean(axis=0) for c in np.unique(labels[valid])
                ])
                noise = labels < 0
                labels[noise] = _assign_nearest_centroid(X[noise], centers)

        # 压缩标签到 0..k-1
        valid = labels >= 0
        if np.any(valid):
            uniq = np.unique(labels[valid])
            remap = {int(o): i for i, o in enumerate(uniq)}
            new_lab = labels.copy()
            for o, n in remap.items():
                new_lab[labels == o] = n
            labels = new_lab

        prob = getattr(clusterer, 'probabilities_', None)
        diag = {
            'min_cluster_size': int(mcs),
            'min_samples': int(ms),
            'n_noise_fit': int(np.sum(fit_labels < 0)),
            'n_clusters_fit': int(len(np.unique(fit_labels[fit_labels >= 0]))),
            'fit_n_points': int(len(X_fit)),
            'probabilities_fit': prob.tolist() if prob is not None else None,
        }
        metrics = self._compute_metrics(
            X, labels, time.time() - t0,
            extra={'min_cluster_size': int(mcs), 'min_samples': int(ms)},
        )
        self.logger.info(
            f"  📌 [{band_name}] HDBSCAN: K={metrics['n_clusters_found']}, "
            f"noise={metrics['noise_fraction']:.2%}, "
            f"min_cluster_size={mcs}"
        )
        return labels, diag, metrics


class HierarchicalClusteringOptimizer(AlternativeClusteringBase):
    """
    凝聚层次聚类（Ward / average / complete linkage）

    在子样本上完成 linkage 与切分，全量体元按最近簇心指派。
    树状图在更小样本上绘制供方法学展示。
    """

    algorithm_name = 'hierarchical'

    def _cluster_band(
        self,
        X: np.ndarray,
        spatial_indices: np.ndarray,
        metadata: Dict[str, Any],
        band: Dict[str, Any],
        band_name: str,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        cfg = self.config.hierarchical
        t0 = time.time()
        target_k = (
            int(cfg['n_clusters']) if cfg.get('n_clusters') else self._band_target_k(band)
        )
        if target_k is None:
            target_k = int(band.get('min_clusters', 2))

        X_fit, _ = _clustering_subsample(
            X, cfg.get('fit_max_points'), int(cfg.get('random_state', RANDOM_SEED))
        )
        linkage_method = str(cfg.get('linkage', 'ward'))
        model = AgglomerativeClustering(
            n_clusters=int(target_k),
            linkage=linkage_method,
        )
        fit_labels = model.fit_predict(X_fit)
        centers = np.array([
            X_fit[fit_labels == c].mean(axis=0) for c in range(target_k)
        ])
        labels = _assign_nearest_centroid(X, centers)

        # 树状图用更小样本
        n_dend = min(int(cfg.get('dendrogram_sample', 5000)), len(X_fit))
        rng = np.random.default_rng(int(cfg.get('random_state', RANDOM_SEED)))
        d_idx = rng.choice(len(X_fit), n_dend, replace=False) if len(X_fit) > n_dend else np.arange(len(X_fit))
        X_d = X_fit[d_idx]
        if linkage_method == 'ward':
            Z = linkage(X_d, method='ward')
        else:
            Z = linkage(pdist(X_d, metric='euclidean'), method=linkage_method)

        diag = {
            'n_clusters': int(target_k),
            'linkage': linkage_method,
            'fit_n_points': int(len(X_fit)),
            'dendrogram_n_points': int(n_dend),
            'linkage_matrix': Z.tolist(),
        }
        metrics = self._compute_metrics(
            X, labels, time.time() - t0,
            extra={'linkage': linkage_method, 'n_clusters_target': int(target_k)},
        )
        self.logger.info(
            f"  📌 [{band_name}] Hierarchical ({linkage_method}): K={target_k}"
        )
        return labels, diag, metrics


class GMMAutoClusteringOptimizer(DepthBandPartitioner):
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

    def _sil_cfg(self) -> Dict[str, Any]:
        """轮廓系数子样本配置（与 W-k-means 共用 wkmeans.k_selection 项）"""
        return self.config.wkmeans['k_selection']

    def _silhouette_repeated(
        self, X: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, float]:
        """在多个独立子样本上重复估计轮廓系数（与 W-k-means 路径一致）"""
        ksel = self._sil_cfg()
        valid = labels >= 0
        if int(np.sum(valid)) < 3 or len(np.unique(labels[valid])) < 2:
            return float('nan'), float('nan')
        size = min(int(ksel['silhouette_sample_size']), int(np.sum(valid)))
        n_rep = int(ksel.get('silhouette_n_repeats', 1))
        vals: List[float] = []
        for r in range(n_rep):
            try:
                vals.append(float(silhouette_score(
                    X[valid], labels[valid],
                    sample_size=size,
                    random_state=RANDOM_SEED + r,
                )))
            except Exception as e:
                self.logger.debug(f"    轮廓系数估计失败: {e}")
        if not vals:
            return float('nan'), float('nan')
        return float(np.mean(vals)), float(np.std(vals))

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
        silhouettes: List[float] = []
        silhouette_sds: List[float] = []

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
                labels_k = gmm.predict(X_fit)
                sil_m, sil_sd = self._silhouette_repeated(X_fit, labels_k)
                silhouettes.append(sil_m)
                silhouette_sds.append(sil_sd)

            except Exception as e:
                self.logger.warning(f"    n={n} 失败: {e}")
                bics.append(float('inf'))
                aics.append(float('inf'))
                log_likelihoods.append(-float('inf'))
                n_params_list.append(float('nan'))
                silhouettes.append(float('nan'))
                silhouette_sds.append(float('nan'))

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
        k_values_arr = np.asarray(n_clusters_range[: len(bics)], dtype=int)
        sil_arr = np.asarray(silhouettes, dtype=float)
        sil_sd_arr = np.asarray(silhouette_sds, dtype=float)
        k_sil = (
            int(k_values_arr[int(np.nanargmax(sil_arr))])
            if np.any(np.isfinite(sil_arr)) else int(k_values_arr[argmin_idx])
        )
        k_local = WKMeansClusteringOptimizer._silhouette_local_maxima(
            k_values_arr, sil_arr
        )
        k_bic_knee = argmin_clusters

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
                k_bic_knee = int(k_values_arr[knee_idx])

        # ── 判据 1b：BIC 拐点 + 轮廓系数局部极大（混合，与 W-k-means 对齐）──
        elif rule == 'bic_knee_silhouette':
            knee_idx, knee_distances = self._knee_select(bics, valid_indices)
            if knee_idx is not None:
                k_bic_knee = int(k_values_arr[knee_idx])
                cands = [k for k in k_local if k >= k_bic_knee]
                if cands:
                    optimal_n_h = max(
                        cands,
                        key=lambda k: sil_arr[k_values_arr == k][0],
                    )
                    optimal_idx = int(np.where(k_values_arr == optimal_n_h)[0][0])
                else:
                    optimal_idx = knee_idx
                selection_rule = 'bic_knee_silhouette'
            else:
                optimal_idx = argmin_idx
                selection_rule = 'bic_min'

        # ── 判据 2：argmin(BIC)（拐点不可用时的回退）────────────
        if optimal_idx is None:
            optimal_idx = argmin_idx
            selection_rule = 'bic_min'
        if knee_idx is not None:
            k_bic_knee = int(k_values_arr[knee_idx])
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
            'knee_n': k_bic_knee if knee_idx is not None else None,
            'repaired_ks': [int(n_clusters_range[i]) for i in repaired_idx],
            'selection_rule': selection_rule,
            'k_upper': k_upper,
            'optimal_n': optimal_clusters,
            'optimal_bic': bics[optimal_idx],
            'optimal_aic': aics[optimal_idx],
            'argmin_n': argmin_clusters,
            'argmin_bic': bics[argmin_idx],
            'silhouette': sil_arr.tolist(),
            'silhouette_sd': sil_sd_arr.tolist(),
            'k_silhouette': int(k_sil),
            'k_local_maxima': [int(k) for k in k_local],
            'n_points_scanned': int(len(X_fit)),
            'convergence_info': {
                'n_tested': len(bics),
            },
        }

        if selection_rule == 'bic_knee':
            self.logger.info(
                f"  ✅ [{band_name}] 建议 K={optimal_clusters} (BIC 拐点, "
                f"弦距={knee_distances[optimal_idx]:.3f}, K 上界={k_upper}); "
                f"轮廓全局最优 K={k_sil}，局部极大 K={k_local or '无'}; "
                f"argmin(BIC)=K{argmin_clusters}（诊断）— 请结合 2-3-1 图复核"
            )
        elif selection_rule == 'bic_knee_silhouette':
            sil_at_opt = float(sil_arr[optimal_idx])
            self.logger.info(
                f"  ✅ [{band_name}] 建议 K={optimal_clusters} "
                f"(BIC 拐点 K={k_bic_knee} + 轮廓局部极大, "
                f"轮廓={sil_at_opt:.4f}); "
                f"轮廓全局最优 K={k_sil}; argmin(BIC)=K{argmin_clusters} "
                f"（诊断）— 请结合 2-3-1 图复核"
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
                optimal_n = int(fixed_k)
                keep_bic = self.config.auto_gmm.get('bic_diagnostic_when_fixed', True)
                if keep_bic:
                    # 仍执行扫描，但只作诊断：BIC 曲线进附录，缓存模型供 K 扫描
                    # 稳健性复用；选 K 由先验决定，不受曲线影响。
                    _knee_n, bic_analysis = self.find_optimal_clusters(
                        X_band,
                        min_clusters=int(band['min_clusters']),
                        max_clusters=int(band['max_clusters']),
                        band_name=name,
                        spatial_indices=spatial_indices[mask],
                        dims=dims,
                    )
                    bic_analysis['knee_n'] = int(_knee_n)
                    bic_analysis['optimal_n'] = optimal_n
                    bic_analysis['selection_rule'] = 'fixed_k'
                    self.logger.info(
                        f"  📌 [{name}] 先验固定 K={optimal_n}"
                        f"（BIC 拐点 K={_knee_n} 仅作诊断，不参与选 K）"
                    )
                else:
                    # 跳过扫描（清空缓存避免复用旧带的模型）
                    self._cached_gmms = {}
                    bic_analysis = {
                        'band_name': name,
                        'optimal_n': optimal_n,
                        'optimal_bic': float('nan'),
                        'selection_rule': 'fixed_k',
                    }
                    self.logger.info(
                        f"  📌 [{name}] 先验固定 K={optimal_n}（跳过 BIC 扫描）"
                    )
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
            'algorithm': 'gmm',
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
        self.wkmeans_optimizer = WKMeansClusteringOptimizer(self.config, self.logger)
        self.hdbscan_optimizer = HDBSCANClusteringOptimizer(self.config, self.logger)
        self.hierarchical_optimizer = HierarchicalClusteringOptimizer(
            self.config, self.logger
        )
        self.concordance_eval = GeologyConcordanceEvaluator(self.config, self.logger)
        self.gmm_optimizer.concordance_eval = self.concordance_eval
        for opt in (
            self.hdbscan_optimizer,
            self.hierarchical_optimizer,
            self.wkmeans_optimizer,
        ):
            opt.concordance_eval = self.concordance_eval

        self.enhanced_visualizer = EnhancedClusteringVisualizer(self.config, self.logger)
        self.enhanced_visualizer.set_processor(self.processor)
        self.basic_visualizer = BasicClusteringVisualizer(self.config, self.logger)
        self.basic_visualizer.set_processor(self.processor)

        if self.config.kmeans_comparison['enabled']:
            self.kmeans_analyzer = KMeansComparisonAnalyzer(self.config, self.logger)
        else:
            self.kmeans_analyzer = None

        self.all_results: Dict[str, Any] = {}

    def _algo_name(self) -> str:
        """当前聚类算法名，用于输出文件名与元数据，避免把 W-k-means 结果标成 GMM"""
        return str(getattr(self.config, 'algorithm', 'gmm')).lower()

    def _scheme_tag(self) -> str:
        """
        输出目录名：GMM 路径用分带方案名；其余算法编码分带方案与特征域。

        非 GMM 算法的目录名形如 {algo}_{dln|raw}_{band}，确保不同设置的
        测试结果互不覆盖，也不会覆盖正文依赖的 GMM 分带结果。
        """
        scheme = str(self.config.depth_stratified.get('scheme', 'fixed_3band'))
        algo = self._algo_name()
        coord_suffix = (
            '_xyz'
            if self.config.preprocessing.get('coordinates', {}).get(
                'enabled', False
            )
            else ''
        )
        if algo == 'gmm':
            domain = (
                'dln'
                if self.config.preprocessing.get('perturbation', {}).get(
                    'enabled', True
                )
                else 'raw'
            )
            if bool(getattr(self.config, 'k_scan', False)):
                return f'gmm_{domain}_{scheme}{coord_suffix}'
            # 正文 dln 固定 K → moho_4band/；raw 对照 → gmm_raw_moho_4band/，互不覆盖
            if domain == 'raw':
                return f'gmm_raw_{scheme}{coord_suffix}'
            return f'{scheme}{coord_suffix}'

        band = scheme if self.config.depth_stratified.get('enabled', True) else 'fulldepth'
        domain = (
            'dln'
            if self.config.preprocessing.get('perturbation', {}).get('enabled', True)
            else 'raw'
        )
        tag = f'{algo}_{domain}_{band}'
        if self.config.preprocessing.get('coordinates', {}).get('enabled', False):
            tag += '_xyz'
        return tag

    def _scheme_output_root(self) -> Path:
        """数据根目录：results/model_clustering/{tag}/（标签、概率、NetCDF、统计）"""
        return self.base_config.dirs['results'] / 'model_clustering' / self._scheme_tag()

    def _scheme_figure_root(self) -> Path:
        """图件根目录：figures/model_clustering/{tag}/

        图件与数据分置：图件集中在 figures/ 便于取用与投稿，数据留在 results/。
        """
        return self.base_config.dirs['figures'] / 'model_clustering' / self._scheme_tag()

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

            algorithm = str(getattr(self.config, 'algorithm', 'gmm')).lower()
            stratified = self.config.depth_stratified.get('enabled', True)
            scheme = str(self.config.depth_stratified.get('scheme', 'fixed_3band'))
            pert_on = self.config.preprocessing.get('perturbation', {}).get(
                'enabled', True
            )
            coord_on = self.config.preprocessing.get('coordinates', {}).get(
                'enabled', False
            )
            algo_label = {
                'gmm': 'GMM',
                'wkmeans': 'W-k-means',
                'hdbscan': 'HDBSCAN',
                'hierarchical': 'Hierarchical',
            }.get(algorithm, algorithm.upper())

            self.logger.info("\n" + "=" * 80)
            self.logger.info("🚀 EASTASIA-FWI 单模型速度簇聚类 (v7.5)")
            self.logger.info("=" * 80)
            self.logger.info(
                f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            self.logger.info(f"\n📊 分析模型 ({len(model_names)}):")
            for i, model in enumerate(model_names, 1):
                self.logger.info(f"  {i}. {model}")
            self.logger.info(f"\n🔬 特征: {', '.join(features_to_use)}")
            self.logger.info(f"🧮 算法: {algo_label}")
            self.logger.info(f"📐 扰动域: {'ON' if pert_on else 'OFF'}")
            self.logger.info(f"🧭 坐标增广: {'ON' if coord_on else 'OFF'}")
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
                        'cluster_results': None,
                        'kmeans_results': None,
                    }

                    inv_transform = (
                        self.processor.scaler.inverse_transform
                        if getattr(self.processor, 'scaler', None) is not None
                        else None
                    )
                    self.gmm_optimizer.feature_inverse_transform = inv_transform

                    self.logger.info(
                        f"\n🤖 执行 {algo_label} 速度簇聚类"
                        f"（{'深度分带' if stratified else '全深度'}）..."
                    )

                    if algorithm == 'wkmeans':
                        self.wkmeans_optimizer.feature_inverse_transform = inv_transform
                        cluster_res = (
                            self.wkmeans_optimizer.perform_depth_stratified_clustering(
                                X, spatial_indices, metadata
                            )
                            if stratified else
                            self.wkmeans_optimizer.perform_full_depth_clustering(
                                X, spatial_indices, metadata
                            )
                        )
                    elif algorithm == 'hdbscan':
                        if not stratified:
                            raise ValueError("HDBSCAN 聚类需启用深度分带")
                        cluster_res = (
                            self.hdbscan_optimizer.perform_depth_stratified_clustering(
                                X, spatial_indices, metadata
                            )
                        )
                    elif algorithm == 'hierarchical':
                        if not stratified:
                            raise ValueError("层次聚类需启用深度分带")
                        cluster_res = (
                            self.hierarchical_optimizer.perform_depth_stratified_clustering(
                                X, spatial_indices, metadata
                            )
                        )
                    elif stratified:
                        cluster_res = self.gmm_optimizer.perform_depth_stratified_clustering(
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
                        cluster_res = {
                            'algorithm': 'gmm',
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

                    results['cluster_results'] = cluster_res

                    # K-means（可选）
                    if self.kmeans_analyzer and cluster_res.get('mode') != 'depth_stratified':
                        self.logger.info("\n📊 执行 K-means 对比...")
                        results['kmeans_results'] = (
                            self.kmeans_analyzer.run_kmeans_comparison(
                                X, cluster_res['labels'], cluster_res['n_clusters']
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

    def _viz_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """可视化用元数据：合并 feature_space / 展示名，区分 raw 与 δln。"""
        meta = dict(results['metadata'])
        prep = results.get('preprocessing_info') or {}
        meta['feature_space'] = prep.get(
            'feature_space', meta.get('feature_space', 'absolute')
        )
        if prep.get('feature_names'):
            meta['feature_display_names'] = prep['feature_names']
        return meta

    def _visualize_all_results(self, results: Dict[str, Any], model_name: str) -> None:
        """生成可视化"""
        viz_dir = self._scheme_figure_root() / model_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        meta_viz = self._viz_metadata(results)

        cluster_res = results['cluster_results']
        kmeans_results = results.get('kmeans_results')
        vis_config = self.config.visualization['plot_types']
        stratified = cluster_res.get('mode') == 'depth_stratified'
        algo = str(cluster_res.get('algorithm', 'gmm'))
        is_wkmeans = algo == 'wkmeans'
        is_gmm = algo == 'gmm'
        is_alt = algo in ('hdbscan', 'hierarchical')

        # W-k-means 选 K 图：肘部法 SSE 曲线 + 轮廓系数曲线（逐带）
        if is_wkmeans and cluster_res.get('k_selection'):
            try:
                self.enhanced_visualizer.plot_wkmeans_k_selection(
                    cluster_res['k_selection'], viz_dir, model_name
                )
                self._save_k_selection(cluster_res['k_selection'], model_name)
            except Exception as e:
                self.logger.warning(f"  ⚠️ W-k-means 选 K 图失败: {e}")

        # W-k-means 特征权重图：各带权重分布是"哪个特征在分簇中起作用"的定量答案
        if is_wkmeans and cluster_res.get('per_band'):
            try:
                self.basic_visualizer.plot_wkmeans_feature_weights(
                    cluster_res['per_band'], results['metadata'],
                    viz_dir, model_name,
                )
            except Exception as e:
                self.logger.warning(f"  ⚠️ W-k-means 特征权重图失败: {e}")

        # W-k-means 全深度模式：径向分层分析（深度占位、层界深度与横向起伏）
        if (
            is_wkmeans and not stratified
            and self.config.wkmeans['radial_analysis'].get('enabled', True)
        ):
            try:
                cluster_res['boundary_depths'] = (
                    self.basic_visualizer.plot_radial_stratification(
                        cluster_res,
                        results['metadata'],
                        viz_dir,
                        model_name,
                        self.config.wkmeans['radial_analysis'],
                    )
                )
            except Exception as e:
                self.logger.warning(f"  ⚠️ 径向分层图失败: {e}")

        # 替代聚类算法的方法学诊断图
        if is_alt and cluster_res.get('method_diagnostics'):
            try:
                self.enhanced_visualizer.plot_alt_clustering_diagnostics(
                    algo, cluster_res['method_diagnostics'],
                    cluster_res.get('per_band'), viz_dir, model_name,
                )
                self._save_method_diagnostics(
                    cluster_res['method_diagnostics'], model_name, algo
                )
            except Exception as e:
                self.logger.warning(f"  ⚠️ {algo} 方法学诊断图失败: {e}")

        # 选 K 判据图：分层时优先合成一张（BIC 拐点模式），否则逐带绘制
        # 非 GMM 算法无 BIC，已由 W-k-means 肘部法图或上方诊断图替代
        if vis_config.get('bic_analysis') and is_gmm:
            if stratified and cluster_res.get('per_band'):
                merged = self.enhanced_visualizer.plot_k_selection_panels(
                    cluster_res['per_band'], viz_dir, model_name
                )
                k_plot_per_band = cluster_res['per_band']
                if not merged:
                    # 固定 K 且跳过 BIC 扫描时，从 k-scan 归档重绘诊断曲线
                    k_overrides = cluster_res.get('metrics', {}).get(
                        'band_optimal_k'
                    )
                    archive_pb = self._per_band_bic_from_kscan_archive(
                        model_name, k_overrides
                    )
                    if archive_pb:
                        merged = self.enhanced_visualizer.plot_k_selection_panels(
                            archive_pb, viz_dir, model_name
                        )
                        if merged:
                            k_plot_per_band = archive_pb
                            self.logger.info(
                                '  📊 选 K 图来自 k-scan 归档'
                                f'（标注固定 K={k_overrides}）'
                            )
                self._save_gmm_k_selection(k_plot_per_band, model_name)
                if not merged:
                    for band_name, band_res in cluster_res['per_band'].items():
                        self.enhanced_visualizer.plot_bic_analysis(
                            band_res['bic_analysis'],
                            viz_dir,
                            f"{model_name}_{band_name}",
                        )
            else:
                self.enhanced_visualizer.plot_bic_analysis(
                    cluster_res['bic_analysis'], viz_dir, model_name
                )

        # K 扫描稳健性图（k_robustness 开启时才有数据）
        if cluster_res.get('k_robustness'):
            self.enhanced_visualizer.plot_k_robustness(
                cluster_res['k_robustness'], viz_dir, model_name
            )

        # 概率分析：分层时用 max_probability 构造伪概率分布
        # 硬指派算法（W-k-means / HDBSCAN / 层次）跳过
        if is_gmm and (
            vis_config.get('probability_distribution')
            or vis_config.get('confidence_analysis')
        ):
            if stratified:
                max_p = cluster_res.get('max_probability')
                if max_p is not None:
                    # 用两列伪概率 [1-p, p] 兼容原绘图接口
                    pseudo = np.column_stack([1.0 - max_p, max_p])
                    self.enhanced_visualizer.plot_probability_distribution(
                        pseudo, cluster_res['labels'], viz_dir, model_name
                    )
            else:
                self.enhanced_visualizer.plot_probability_distribution(
                    cluster_res['probabilities'],
                    cluster_res['labels'],
                    viz_dir,
                    model_name,
                )

        # GMM 与 K-means 的判据对比图依赖 BIC/后验，仅 GMM 路径可用
        if vis_config.get('gmm_vs_kmeans') and kmeans_results and is_gmm:
            self.enhanced_visualizer.plot_gmm_vs_kmeans_comparison(
                cluster_res, kmeans_results, viz_dir, model_name
            )

        if vis_config.get('depth_slices'):
            # 3×3：第1排地质，第2–3排 Slab2（替代原 2-3-9 / 2-3-10）
            self.basic_visualizer.plot_depth_slices(
                cluster_res['labels_3d'],
                meta_viz,
                viz_dir,
                algo,
                model_name,
                concordance_eval=self.concordance_eval,
            )

        # 未标准化特征立方体（δln 或 raw Vp/Vs，供 2-3-5 / 2-3-11）
        pert_cube = getattr(self.processor, 'last_feature_cube', None)

        # 三维同维度：聚类体 vs Slab2(dep+thk) 体掩膜（独立于切片 MI/ρ）
        if vis_config.get('slab_volume_3d', True):
            try:
                self.concordance_eval.plot_slab_volume_comparison(
                    cluster_res['labels_3d'],
                    meta_viz,
                    viz_dir,
                    model_name,
                    pert_cube=pert_cube,
                )
            except Exception as e:
                self.logger.warning(f"  ⚠️ Slab2 三维体对比图失败: {e}")

        if vis_config.get('vertical_sections'):
            self.basic_visualizer.plot_vertical_sections(
                cluster_res['labels_3d'],
                meta_viz,
                viz_dir,
                algo,
                model_name,
                concordance_eval=self.concordance_eval,
                pert_cube=pert_cube,
            )

        if vis_config.get('cluster_profiles'):
            self.basic_visualizer.plot_cluster_profiles(
                results['features'],
                cluster_res['labels'],
                results['spatial_indices'],
                results['metadata'],
                viz_dir,
                algo,
                model_name,
                per_band=cluster_res.get('per_band'),
            )

        # GMM 簇心谱系：Ward on μ_k + 启发式 facies 短标签（附录用）
        if (
            is_gmm
            and stratified
            and vis_config.get('centroid_phylogeny', True)
            and cluster_res.get('per_band')
        ):
            try:
                phylo = self.enhanced_visualizer.plot_centroid_phylogeny(
                    results['features'],
                    cluster_res['labels'],
                    results['metadata'],
                    cluster_res['per_band'],
                    viz_dir,
                    model_name,
                    algorithm_name=algo,
                )
                if phylo:
                    self._save_centroid_phylogeny(phylo, model_name)
            except Exception as e:
                self.logger.warning(f"  ⚠️ 簇心谱系图失败: {e}")

        if vis_config.get('feature_distributions'):
            self.basic_visualizer.plot_feature_distributions(
                results['features'],
                cluster_res['labels'],
                results['metadata'],
                viz_dir,
                algo,
                model_name,
                per_band=cluster_res.get('per_band'),
            )

        if vis_config.get('cluster_centers') and cluster_res.get('model') is not None:
            self.basic_visualizer.plot_cluster_centers(
                results['features'],
                cluster_res['labels'],
                results['metadata'],
                cluster_res['model'],
                viz_dir,
                algo,
                model_name,
            )

        if vis_config.get('cluster_crossplot', True):
            try:
                diagnostics = self.basic_visualizer.plot_cluster_crossplot(
                    results['features'],
                    cluster_res['labels'],
                    meta_viz,
                    viz_dir,
                    algo,
                    model_name,
                    per_band=cluster_res.get('per_band'),
                    spatial_indices=results.get('spatial_indices'),
                )
                cluster_res['crossplot_diagnostics'] = diagnostics
                self._save_crossplot_diagnostics(diagnostics, model_name)
            except Exception as e:
                self.logger.warning(f"  ⚠️ 特征空间交会图失败: {e}")

        if vis_config.get('cluster_crossplot_3d', True):
            try:
                self.basic_visualizer.plot_cluster_crossplot_3d(
                    results['features'],
                    cluster_res['labels'],
                    meta_viz,
                    viz_dir,
                    algo,
                    model_name,
                    per_band=cluster_res.get('per_band'),
                    spatial_indices=results.get('spatial_indices'),
                )
            except Exception as e:
                self.logger.warning(f"  ⚠️ 三维特征空间交会图失败: {e}")

        self.logger.info(f"  ✅ 所有可视化已保存到: {viz_dir}")

    def _save_k_selection(
        self, k_selection: Dict[str, Dict[str, Any]], model_name: str
    ) -> None:
        """
        把 W-k-means 各带的选 K 扫描结果写入 CSV，供方法学章节引用。

        每行是一个 (深度带, K) 组合，记录 SSE 与轮廓系数，以及该带最终采用的
        K 及其来源（肘部法/轮廓系数/先验固定），使选 K 过程完全可复核。

        Args:
            k_selection: {深度带名: 选 K 诊断字典}
            model_name: 模型名
        """
        rows: List[Dict[str, Any]] = []
        for band, diag in k_selection.items():
            k_values = diag.get('k_values')
            if not k_values:
                continue
            sds = diag.get('silhouette_sd') or [float('nan')] * len(k_values)
            for k, sse, sil, sd in zip(
                k_values, diag['sse'], diag['silhouette'], sds
            ):
                rows.append({
                    'model': model_name,
                    'band': band,
                    'k': int(k),
                    'n_points': diag.get('n_points_scanned'),
                    'sse': float(sse),
                    'silhouette': float(sil),
                    'silhouette_sd': float(sd),
                    'k_elbow': diag['k_elbow'],
                    'k_silhouette': diag['k_silhouette'],
                    'k_selected': diag['optimal_n'],
                    'selection_rule': diag['selection_rule'],
                })
        if not rows:
            return

        output_dir = self._scheme_output_root() / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / 'wkmeans_k_selection.csv'
        pd.DataFrame(rows).to_csv(filepath, index=False, float_format='%.6f')
        self.logger.info(f"  💾 选 K 扫描: {filepath.name}")

    def _kscan_archive_root(self) -> Optional[Path]:
        """
        对应域/分带方案的 GMM k-scan 结果目录（固定 K 正文跑批时复用 BIC 曲线）。

        Returns:
            results/model_clustering/gmm_{dln|raw}_{scheme}/；不存在则 None
        """
        if self._algo_name() != 'gmm' or bool(getattr(self.config, 'k_scan', False)):
            return None
        body_tag = self._scheme_tag()
        domain = (
            'dln'
            if self.config.preprocessing.get('perturbation', {}).get(
                'enabled', True
            )
            else 'raw'
        )
        # 正文 moho_4band[_xyz] → k-scan 归档 gmm_dln_moho_4band[_xyz]
        archive_scheme = body_tag
        if not archive_scheme.startswith('gmm_'):
            archive_scheme = f'gmm_{domain}_{body_tag}'
        root = (
            self.base_config.dirs['results']
            / 'model_clustering'
            / archive_scheme
        )
        return root if root.is_dir() else None

    def _per_band_bic_from_kscan_csv(
        self,
        csv_path: Path,
        k_overrides: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        从 gmm_k_selection.csv 重建 per_band['bic_analysis']，供固定 K 跑批重绘 2-3-1。

        Args:
            csv_path: k-scan 输出的 CSV 路径
            k_overrides: 各带最终采用的 K（覆盖 CSV 中的 k_selected）

        Returns:
            {深度带: {'bic_analysis': {...}}}；文件无效则空 dict
        """
        if not csv_path.is_file():
            return {}
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            self.logger.warning(f'  ⚠️ 读取 k-scan CSV 失败: {exc}')
            return {}
        if df.empty or 'band' not in df.columns:
            return {}

        per_band: Dict[str, Any] = {}
        for band, grp in df.groupby('band', sort=False):
            grp = grp.sort_values('k')
            k_sel = int(grp['k_selected'].iloc[0])
            if k_overrides and band in k_overrides:
                k_sel = int(k_overrides[band])
            ba: Dict[str, Any] = {
                'band_name': str(band),
                'n_clusters_range': grp['k'].astype(int).tolist(),
                'bics': grp['bic'].astype(float).tolist(),
                'silhouette': grp['silhouette'].astype(float).tolist(),
                'silhouette_sd': grp['silhouette_sd'].astype(float).tolist(),
                'n_points_scanned': (
                    int(grp['n_points'].iloc[0])
                    if 'n_points' in grp.columns
                    else None
                ),
                'optimal_n': k_sel,
                'selection_rule': (
                    'fixed_k' if k_overrides else str(
                        grp['selection_rule'].iloc[0]
                    )
                ),
            }
            for src, dst in (
                ('k_bic_knee', 'knee_n'),
                ('k_silhouette', 'k_silhouette'),
                ('argmin_bic_k', 'argmin_n'),
            ):
                if src in grp.columns and pd.notna(grp[src].iloc[0]):
                    ba[dst] = int(grp[src].iloc[0])
            per_band[str(band)] = {'bic_analysis': ba}
        return per_band

    def _per_band_bic_from_kscan_archive(
        self,
        model_name: str,
        k_overrides: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        从 k-scan 归档目录加载 BIC 诊断，供固定 K 正文结果补绘选 K 图。

        Args:
            model_name: 模型名
            k_overrides: 各带最终 K

        Returns:
            per_band 字典；归档不存在则空 dict
        """
        root = self._kscan_archive_root()
        if root is None:
            return {}
        csv_path = root / model_name / 'gmm_k_selection.csv'
        return self._per_band_bic_from_kscan_csv(csv_path, k_overrides)

    def _save_gmm_k_selection(
        self, per_band: Dict[str, Any], model_name: str
    ) -> None:
        """
        把 GMM 各带的 BIC + 轮廓系数 K 扫描结果写入 CSV，供方法学复核。

        Args:
            per_band: 分层聚类 per_band 字典
            model_name: 模型名
        """
        rows: List[Dict[str, Any]] = []
        for band, info in per_band.items():
            ba = info.get('bic_analysis') if isinstance(info, dict) else None
            if not isinstance(ba, dict) or 'bics' not in ba:
                continue
            sils = ba.get('silhouette') or [float('nan')] * len(ba['bics'])
            sds = ba.get('silhouette_sd') or [float('nan')] * len(ba['bics'])
            for k, bic, sil, sd in zip(
                ba['n_clusters_range'], ba['bics'], sils, sds
            ):
                rows.append({
                    'model': model_name,
                    'band': band,
                    'k': int(k),
                    'n_points': ba.get('n_points_scanned'),
                    'bic': float(bic),
                    'silhouette': float(sil),
                    'silhouette_sd': float(sd),
                    'k_bic_knee': ba.get('knee_n'),
                    'k_silhouette': ba.get('k_silhouette'),
                    'k_selected': ba.get('optimal_n'),
                    'selection_rule': ba.get('selection_rule'),
                    'argmin_bic_k': ba.get('argmin_n'),
                })
        if not rows:
            return
        output_dir = self._scheme_output_root() / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / 'gmm_k_selection.csv'
        pd.DataFrame(rows).to_csv(filepath, index=False, float_format='%.6f')
        self.logger.info(f"  💾 GMM 选 K 扫描: {filepath.name}")

    def _save_crossplot_diagnostics(
        self, diagnostics: Dict[str, Dict[str, float]], model_name: str
    ) -> None:
        """
        将交会图诊断量写入 CSV，供跨模型汇总与论文引用。

        Args:
            diagnostics: {深度带名: 诊断量字典}
            model_name: 模型名
        """
        if not diagnostics:
            return
        output_dir = self._scheme_output_root() / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame.from_dict(diagnostics, orient='index')
        df.index.name = 'band'
        df.insert(0, 'model', model_name)
        filepath = output_dir / 'crossplot_diagnostics.csv'
        df.to_csv(filepath, float_format='%.4f')
        self.logger.info(f"  💾 交会图诊断量: {filepath.name}")

    def _save_method_diagnostics(
        self,
        diagnostics: Dict[str, Dict[str, Any]],
        model_name: str,
        algorithm: str,
    ) -> None:
        """
        将 HDBSCAN / 层次聚类的方法学诊断写入 JSON

        Args:
            diagnostics: {深度带名: 诊断字典}
            model_name: 模型名
            algorithm: 算法名
        """
        if not diagnostics:
            return
        output_dir = self._scheme_output_root() / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

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

        out = {
            'model': model_name,
            'algorithm': algorithm,
            'bands': _jsonify(diagnostics),
        }
        filepath = output_dir / f'{algorithm}_method_diagnostics.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        self.logger.info(f"  💾 方法学诊断: {filepath.name}")

    def _save_centroid_phylogeny(
        self, phylo: Dict[str, Any], model_name: str
    ) -> None:
        """保存簇心谱系与 facies 解释 JSON"""
        output_dir = self._scheme_output_root() / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / 'gmm_centroid_phylogeny.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(phylo, f, ensure_ascii=False, indent=2)
        self.logger.info(f"  💾 簇心谱系: {filepath.name}")

    def _save_cross_algorithm_ari(
        self, comparison: Dict[str, Any], model_name: str
    ) -> None:
        """保存 GMM vs 对照算法的 ARI/NMI"""
        output_dir = self._scheme_output_root() / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / 'cross_algorithm_ari.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        self.logger.info(f"  💾 算法一致率: {filepath.name}")
    
    def _save_results(self, results: Dict[str, Any], model_name: str) -> None:
        """保存聚类结果"""
        output_dir = self._scheme_output_root() / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cluster_res = results['cluster_results']
        # 文件名带算法前缀：W-k-means 的结果不应被读成 GMM 的
        algo = self._algo_name()
        
        # 1. 保存标签
        if self.config.output['save_labels']:
            fname = f'{algo}_labels.npz'
            np.savez_compressed(
                output_dir / fname,
                labels_1d=cluster_res['labels'],
                labels_3d=cluster_res['labels_3d'],
                spatial_indices=results['spatial_indices']
            )
            self.logger.info(f"  ✅ 保存标签: {fname}")

        # 2. 保存概率（W-k-means 为硬指派，无后验概率，跳过）
        if (
            self.config.output['save_probabilities']
            and cluster_res.get('probabilities_3d') is not None
        ):
            fname = f'{algo}_probabilities.npz'
            np.savez_compressed(
                output_dir / fname,
                probabilities=cluster_res['probabilities'],
                probabilities_3d=cluster_res['probabilities_3d']
            )
            self.logger.info(f"  ✅ 保存概率: {fname}")

        # 2b. W-k-means 专属输出：特征权重、径向统计、层界深度
        if cluster_res.get('per_band') and algo == 'wkmeans':
            names = list(results['metadata'].get('feature_display_names')
                         or results['metadata'].get('features', []))
            rows = [
                {'model': model_name, 'band': b, 'n_clusters': v['n_clusters'],
                 **{f'w_{n}': float(w)
                    for n, w in zip(names, v['feature_weights'])}}
                for b, v in cluster_res['per_band'].items()
            ]
            pd.DataFrame(rows).to_csv(
                output_dir / 'wkmeans_feature_weights.csv',
                index=False, float_format='%.6f',
            )
            self.logger.info("  ✅ 保存特征权重: wkmeans_feature_weights.csv")

        if cluster_res.get('radial_stats') is not None:
            stats_df = cluster_res['radial_stats'].copy()
            stats_df.insert(0, 'model', model_name)
            stats_df.to_csv(output_dir / 'wkmeans_radial_stats.csv',
                            index=False, float_format='%.4f')
            self.logger.info("  ✅ 保存径向统计: wkmeans_radial_stats.csv")
        if cluster_res.get('boundary_depths') is not None:
            cluster_res['boundary_depths'].to_csv(
                output_dir / 'wkmeans_boundary_depths.csv', index=False,
                float_format='%.2f',
            )
            self.logger.info("  ✅ 保存层界深度: wkmeans_boundary_depths.csv")
        
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
            'algorithm': algo,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mode': cluster_res.get('mode', 'full_depth'),
            'n_clusters': int(cluster_res['n_clusters']),
            'features_used': results['metadata']['features'],
            'feature_space': results['preprocessing_info'].get('feature_space'),
            'spatial_range': results['metadata']['spatial_range'],
            'preprocessing': _jsonify(results['preprocessing_info']),
            'cluster_metrics': _jsonify(
                {
                    k: v
                    for k, v in cluster_res['metrics'].items()
                    if not isinstance(v, (GaussianMixture,))
                }
            ),
            'bic_analysis': (
                {
                    'optimal_n': int(cluster_res['bic_analysis']['optimal_n']),
                    'optimal_bic': float(
                        cluster_res['bic_analysis']['optimal_bic']
                    ),
                }
                if cluster_res.get('bic_analysis') else None
            ),
            'band_optimal_k': cluster_res['metrics'].get('band_optimal_k'),
        }
        if algo == 'wkmeans':
            meta_out['wkmeans'] = _jsonify({
                'beta': self.config.wkmeans['beta'],
                'k_selection': self.config.wkmeans['k_selection'],
                'band_k_selection': cluster_res.get('k_selection'),
                'feature_weights': cluster_res.get('feature_weights'),
                'band_feature_weights': cluster_res['metrics'].get(
                    'band_feature_weights'
                ),
                'cluster_centers': cluster_res.get('cluster_centers'),
            })

        with open(output_dir / 'analysis_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)

        self.logger.info("  ✅ 保存元数据: analysis_metadata.json")
    
    def _save_to_netcdf(self, results: Dict[str, Any], output_dir: Path) -> None:
        """保存为NetCDF格式"""
        try:
            metadata = results['metadata']
            cluster_res = results['cluster_results']
            probs_3d = cluster_res.get('probabilities_3d')
            if cluster_res.get('max_probability_3d') is not None:
                max_prob_3d = cluster_res['max_probability_3d']
            elif probs_3d is not None:
                max_prob_3d = np.max(probs_3d, axis=-1)
            else:
                # 硬指派（W-k-means）：有效体元置信度记为 1
                max_prob_3d = (cluster_res['labels_3d'] >= 0).astype(np.float32)

            ds = xr.Dataset(
                {
                    'cluster_labels': (
                        ['latitude', 'longitude', 'depth'],
                        cluster_res['labels_3d'].astype(np.int16),
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
                    'n_clusters': int(cluster_res['n_clusters']),
                    'algorithm': self._algo_name(),
                    'method': cluster_res.get('mode', 'full_depth'),
                    'feature_space': results['preprocessing_info'].get(
                        'feature_space', 'absolute'
                    ),
                    'features': ', '.join(metadata['features']),
                    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                },
            )

            # 全深度模式保存各簇概率；分层与硬指派模式仅保留 max_probability
            if (
                cluster_res.get('mode') != 'depth_stratified'
                and probs_3d is not None
                and probs_3d.ndim == 4
                and probs_3d.shape[-1] == cluster_res['n_clusters']
            ):
                for i in range(cluster_res['n_clusters']):
                    ds[f'probability_cluster_{i}'] = (
                        ['latitude', 'longitude', 'depth'],
                        probs_3d[:, :, :, i].astype(np.float32),
                    )

            output_file = output_dir / f'{self._algo_name()}_clustering_results.nc'
            ds.to_netcdf(output_file)
            ds.close()
            self.logger.info(f"  ✅ 保存NetCDF: {output_file.name}")

        except Exception as e:
            self.logger.warning(f"  ⚠️ NetCDF保存失败: {e}")
    
    def _save_cluster_statistics(self, results: Dict[str, Any], output_dir: Path) -> None:
        """保存聚类统计信息"""
        cluster_res = results['cluster_results']
        labels = cluster_res['labels']
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
                    cluster_res = results['cluster_results']
                    metrics = cluster_res['metrics']
                    
                    f.write(f"\n模型: {model_name}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"聚类数: {cluster_res['n_clusters']}\n")
                    f.write(f"特征: {', '.join(results['metadata']['features'])}\n")
                    f.write(f"数据点数: {len(cluster_res['labels']):,}\n")
                    f.write(f"\n性能指标:\n")
                    if 'bic' in metrics:
                        f.write(f"  - BIC: {metrics['bic']:.2f}\n")
                    if 'inertia' in metrics:
                        f.write(f"  - 目标函数: {metrics['inertia']:.6f}\n")
                    f.write(f"  - 轮廓系数: {metrics.get('silhouette_score', 0):.4f}\n")
                    f.write(f"  - Davies-Bouldin: {metrics.get('davies_bouldin_score', 0):.4f}\n")
                    f.write(f"  - Calinski-Harabasz: {metrics.get('calinski_harabasz_score', 0):.2f}\n")

                    # W-k-means 为硬指派，无后验概率
                    if 'mean_max_probability' in metrics:
                        f.write(f"\n概率分析:\n")
                        f.write(f"  - 平均最大概率: {metrics['mean_max_probability']:.4f}\n")
                        f.write(f"  - 高置信度比例: {metrics['high_confidence_ratio']:.2%}\n")
                        f.write(f"  - 平均熵: {metrics['mean_entropy']:.4f}\n")
                    if metrics.get('feature_weights') is not None:
                        weights = ', '.join(
                            f'{w:.4f}' for w in metrics['feature_weights']
                        )
                        f.write(f"\n特征权重: {weights}\n")

                    f.write(f"\n计算信息:\n")
                    if 'fit_time' in metrics:
                        f.write(f"  - 拟合时间: {metrics['fit_time']:.2f}秒\n")
                    if 'converged' in metrics:
                        f.write(
                            f"  - 是否收敛: {'是' if metrics['converged'] else '否'}\n"
                        )
                    if 'n_iter' in metrics:
                        f.write(f"  - 迭代次数: {metrics['n_iter']}\n")
                    if metrics.get('noise_fraction') is not None:
                        f.write(
                            f"  - 噪声体元比例: {metrics['noise_fraction']:.2%}\n"
                        )
                    if metrics.get('band_n_clusters'):
                        f.write(f"  - 各带簇数: {metrics['band_n_clusters']}\n")
            
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
  - 特征: 默认 δlnVp/δlnVs（模型原生 vp0/vs0；SinoScope 回退水平平均 1D）；--raw 用 Vp/Vs 绝对值
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


def _build_arg_parser() -> argparse.ArgumentParser:
    """命令行参数：用于在不改代码的前提下切换算法与特征空间做对照测试"""
    p = argparse.ArgumentParser(
        description='EASTASIA-FWI 单模型速度聚类分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  # 正文主路径：扰动域 + 分带 GMM\n'
            '  python 2_3_Model_clustering.py\n\n'
            '  # 对照测试：原始速度值 + 全深度 W-k-means（Hao et al. 2026 方案）\n'
            '  python 2_3_Model_clustering.py --algorithm wkmeans --raw -k 8\n\n'
            '  # 再加坐标增广（经度/纬度/深度并入特征空间）\n'
            '  python 2_3_Model_clustering.py --algorithm wkmeans --raw --coords\n'
        ),
    )
    p.add_argument('--algorithm',
                   choices=['gmm', 'wkmeans', 'hdbscan', 'hierarchical'],
                   default='gmm',
                   help='聚类算法（默认 gmm）')
    p.add_argument('--raw', action='store_true',
                   help='用原始速度值而非 δlnV 扰动域')
    p.add_argument('--coords', action='store_true',
                   help='把经度/纬度/深度并入特征空间')
    p.add_argument('--full-depth', action='store_true',
                   help='关闭深度分带，0-1000 km 一次性聚类（径向分层分析）')
    p.add_argument('-k', '--n-clusters', type=int, default=None,
                   help='固定 W-k-means 的簇数 K，跳过自动选 K')
    p.add_argument('--k-method',
                   choices=['elbow', 'elbow_silhouette', 'sse_threshold',
                            'silhouette'],
                   default=None,
                   help='选 K 判据（默认 elbow）')
    p.add_argument('--k-max', type=int, default=None,
                   help='K 搜索上界（moho_4band 四带统一；fixed_3band 见分带参数）')
    p.add_argument(
        '--scheme', choices=['moho_4band', 'fixed_3band'], default=None,
        help='深度分带方案（默认 moho_4band）',
    )
    p.add_argument('--k-scan', action='store_true',
                   help='GMM BIC 拐点自动选 K（清除分带 fixed_k 先验）')
    p.add_argument('--k-max-shallow', type=int, default=None,
                   help='fixed_3band 浅部(0-410 km) K 上界')
    p.add_argument('--k-max-crust', type=int, default=None,
                   help='moho_4band 地壳(z<Moho) K 上界')
    p.add_argument('--k-max-lithosphere', type=int, default=None,
                   help='moho_4band 岩石圈(Moho→410 km) K 上界')
    p.add_argument('--k-max-transition', type=int, default=None,
                   help='过渡带(410-660 km) K 上界')
    p.add_argument('--k-max-lower', type=int, default=None,
                   help='下地幔(660-1000 km) K 上界')
    p.add_argument('--k-crust', type=int, default=None,
                   help='moho_4band 地壳固定 K（跳过 BIC 选 K，直接聚类）')
    p.add_argument('--k-lithosphere', type=int, default=None,
                   help='moho_4band 岩石圈固定 K')
    p.add_argument('--k-transition', type=int, default=None,
                   help='过渡带(410-660 km) 固定 K')
    p.add_argument('--k-lower', type=int, default=None,
                   help='下地幔(660-1000 km) 固定 K')
    p.add_argument('--beta', type=float, default=None,
                   help='W-k-means 的权重调节参数（默认 6.5，须在 [0,1] 之外）')
    p.add_argument('--scan-beta', action='store_true',
                   help='扫描 β 并按平均轮廓系数标定（Hao et al. 2026 图 S39b）')
    p.add_argument('--models', nargs='+', default=None,
                   help='指定模型名，默认使用配置中的三个模型')
    return p


def main() -> int:
    """主函数"""
    args = _build_arg_parser().parse_args()
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
        # 每带 K 已在 ClusteringConfig 中按先验固定为 10/10/5/5：地壳取研究区
        # prov_type 面积占比 ≥1% 的 10 类地质省，岩石圈与之取齐便于跨带比较，
        # 过渡带与下地幔取分辨率上界。BIC 扫描仍执行但仅作诊断，见
        # auto_gmm['bic_diagnostic_when_fixed']。
        # 如需回到自动选 K，把各带 'fixed_k' 置 None：
        # for band in config.depth_stratified['schemes']['moho_4band']['bands']:
        #     band['fixed_k'] = None
        config.auto_gmm['covariance_type'] = 'full'
        config.kmeans_comparison['enabled'] = False
        config.visualization['enabled'] = True

        # 速度相默认用各向同性 Vp/Vs
        features_to_use = ['vp', 'vs']

        # ===== 命令行覆盖：对照测试路径 =====
        config.algorithm = args.algorithm
        if args.scheme is not None:
            config.depth_stratified['scheme'] = args.scheme
        active_scheme = str(config.depth_stratified.get('scheme', 'moho_4band'))
        active_bands = config.depth_stratified['schemes'][active_scheme]['bands']

        if args.algorithm in ('hdbscan', 'hierarchical'):
            # HDBSCAN 完全自动定 K；层次聚类默认 respect_fixed_k=True 对齐先验
            if args.algorithm == 'hdbscan':
                for band in active_bands:
                    band['fixed_k'] = None
        if args.raw:
            config.preprocessing['perturbation']['enabled'] = False
        if args.coords:
            config.preprocessing['coordinates']['enabled'] = True
        if args.full_depth:
            config.depth_stratified['enabled'] = False
        if args.k_method is not None:
            config.wkmeans['k_selection']['method'] = args.k_method
        if args.k_max is not None:
            for band in active_bands:
                band['max_clusters'] = args.k_max
            config.wkmeans['k_selection']['max_clusters'] = args.k_max
        # 分带 K 上界（GMM BIC 扫描 / W-k-means 肘部扫描）
        k_band_map = {
            'shallow': args.k_max_shallow,
            'crust': args.k_max_crust,
            'lithosphere': args.k_max_lithosphere,
            'transition_zone': args.k_max_transition,
            'lower_mantle': args.k_max_lower,
        }
        k_fixed_map = {
            'crust': args.k_crust,
            'lithosphere': args.k_lithosphere,
            'transition_zone': args.k_transition,
            'lower_mantle': args.k_lower,
            'shallow': None,
        }
        if args.k_scan:
            config.k_scan = True
            config.auto_gmm['k_selection'] = 'bic_knee_silhouette'
            for band in active_bands:
                band.pop('fixed_k', None)
            config.auto_gmm['bic_diagnostic_when_fixed'] = False
        elif any(v is not None for v in k_fixed_map.values()):
            # 人工裁定 K：跳过 BIC 扫描，直接按 fixed_k 聚类（快）
            config.auto_gmm['bic_diagnostic_when_fixed'] = False
            for band in active_bands:
                k_fix = k_fixed_map.get(band['name'])
                if k_fix is not None:
                    band['fixed_k'] = int(k_fix)
        for band in active_bands:
            k_hi = k_band_map.get(band['name'])
            if k_hi is not None:
                band['max_clusters'] = int(k_hi)
            if args.algorithm == 'wkmeans' and not config.wkmeans['k_selection'].get(
                'respect_fixed_k', False
            ):
                band.pop('fixed_k', None)
        if args.n_clusters is not None:
            config.wkmeans['k_selection']['method'] = 'fixed'
            config.wkmeans['k_selection']['fixed_k'] = args.n_clusters
        if args.beta is not None:
            config.wkmeans['beta'] = args.beta
        if args.scan_beta:
            config.wkmeans['beta_selection']['enabled'] = True

        pipeline = SmartClusteringPipeline(config)
        results = pipeline.run_analysis(
            model_names=args.models, features_to_use=features_to_use
        )

        print(f"\n{'=' * 80}")
        print(f"✅ 分析完成！共处理 {len(results)} 个模型")
        print(f"{'=' * 80}\n")

        for model_name, result in results.items():
            res = result['cluster_results']
            print(f"\n模型: {model_name}")
            print(f"  模式: {res.get('mode')} / "
                  f"{res.get('scheme', res['metrics'].get('scheme'))}")
            print(f"  全局簇数: {res['n_clusters']}")

            if res['metrics'].get('band_optimal_k'):
                print(f"  各带 K: {res['metrics']['band_optimal_k']}")

            if str(res.get('algorithm')) == 'wkmeans':
                names = result['metadata'].get('feature_display_names', [])
                print(f"  平均轮廓系数: "
                      f"{res['metrics'].get('silhouette_score', float('nan')):.4f}")
                for band, diag in (res.get('k_selection') or {}).items():
                    if 'k_elbow' not in diag:
                        continue
                    print(f"  [{band}] 肘部 K={diag['k_elbow']}, "
                          f"轮廓最优 K={diag['k_silhouette']}, "
                          f"采用 K={diag['optimal_n']} "
                          f"(轮廓系数 {diag['silhouette_at_optimal']:.4f})")
                for band, v in (res.get('per_band') or {}).items():
                    print(f"  [{band}] 特征权重: " + '  '.join(
                        f'{n}={w:.3f}'
                        for n, w in zip(names, v['feature_weights'])
                    ))
                bnd = res.get('boundary_depths')
                if bnd is not None and not bnd.empty:
                    print("  层界深度 (km，中位数 / 横向起伏):")
                    for _, r in bnd.iterrows():
                        print(f"    界{int(r['boundary'])}: "
                              f"{r['depth_median']:>6.1f} / {r['relief']:>5.1f}")
            else:
                print(
                    f"  平均最大概率: "
                    f"{res['metrics'].get('mean_max_probability', 0):.4f}"
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
