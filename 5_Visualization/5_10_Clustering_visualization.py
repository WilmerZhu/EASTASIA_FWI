"""
EASTASIA-FWI 速度相聚类可视化模块
====================================================================

功能描述:
- K 选择诊断图: BIC 曲线、拐点弦距、四带合并面板
- 聚类结果图: 深度切片、垂直剖面、簇剖面、特征分布、簇中心
- 特征空间交会图: 2-3-12 二维 (δlnVs, δlnVp) 与 2-3-13 三维 (δlnVs, δlnVp, depth)
- 结构对照: Slab2 俯冲板片几何、地质省与 CN 地块边界叠绘（均为定性对照）
- GMM 与 KMeans 对比图、后验概率分布图

科学原理:
- facies 解释标注依据 δlnVp/δlnVs 的联合符号与幅值（见 interpret_facies）
- 交会图判据 R = σ∥/σ⊥ 量化簇中心相对点云主轴的走向，R ≫ 1 表明聚类主要
  按扰动幅度切分而非物性差异（见 crossplot_diagnostics）
- Slab2 与地质边界仅作定性对照，不参与任何定量评分

说明:
本模块自 2_3_Model_clustering.py 拆分而来，仅承担可视化职责。配置对象为
2_3 中定义的 ClusteringConfig；为避免循环导入，此处以 Any 标注其类型。

作者: EASTASIA-FWI Team
日期: 2026-09-02
版本: v1.0
"""

import sys
import re
import textwrap
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn as sns

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import BaseConfig

# 与 2_3_Model_clustering 保持一致的随机种子，确保抽样绘图可复现
RANDOM_SEED = 42

# moho_4band 选 K 图固定顺序与发表用层名
MOHO_BAND_ORDER: Tuple[str, ...] = (
    'crust', 'lithosphere', 'transition_zone', 'lower_mantle',
)
MOHO_BAND_TITLES: Dict[str, str] = {
    'crust': 'Crust',
    'lithosphere': 'Lithosphere',
    'transition_zone': 'Transition zone',
    'lower_mantle': 'Lower mantle',
    'shallow': 'Shallow mantle',
    'all': 'Full depth',
}
# 谱系图 facies 表格行高相对基准的缩放（含 GridSpec 表格区高度）
PHYLO_TABLE_ROW_SCALE = 1.5


def _short_model_label(model_name: str) -> str:
    """2022_SinoScope1.0 → SinoScope1.0"""
    return re.sub(r'^\d{4}_', '', str(model_name))


def _sort_bands_for_k_plot(
    bands: Sequence[Tuple[str, Any]],
) -> List[Tuple[str, Any]]:
    """按 moho_4band 固定顺序排列子图 (a–d)"""
    order = {name: i for i, name in enumerate(MOHO_BAND_ORDER)}
    return sorted(bands, key=lambda item: order.get(item[0], 99))



def apply_clustering_plot_style(config: Optional[Any] = None) -> Dict[str, float]:
    """
    统一聚类可视化 Matplotlib 字号（rcParams + 返回字号字典供显式调用）。

    Returns:
        fonts: 含 base/tick/label/title/suptitle/legend/annotation/clabel
    """
    defaults = {
        'base': 16,
        'tick': 15,
        'label': 17,
        'title': 19,
        'suptitle': 24,
        'legend': 15,
        'annotation': 14,
        'clabel': 13,
    }
    fonts = dict(defaults)
    if config is not None:
        fonts.update(config.visualization.get('fonts') or {})
    plt.rcParams.update({
        'font.size': float(fonts['base']),
        'axes.titlesize': float(fonts['title']),
        'axes.labelsize': float(fonts['label']),
        'xtick.labelsize': float(fonts['tick']),
        'ytick.labelsize': float(fonts['tick']),
        'legend.fontsize': float(fonts['legend']),
        'figure.titlesize': float(fonts['suptitle']),
    })
    return fonts


def overlay_slab_on_section(
    ax,
    sec_mask: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    *,
    with_legend: bool = False,
    legend_fontsize: float = 12,
) -> None:
    """
    在经度–深度剖面上叠绘 Slab2(dep+thk) 体：灰罩、轮廓、顶/底线。

    Args:
        sec_mask: (n_lon, n_depth) 布尔掩膜
        lons, depths: 1D 坐标
    """
    if sec_mask is None or not np.any(sec_mask):
        return
    extent = [
        float(np.min(lons)),
        float(np.max(lons)),
        float(np.max(depths)),
        float(np.min(depths)),
    ]
    shade = np.ma.masked_where(~sec_mask.T, np.ones(sec_mask.T.shape))
    ax.imshow(
        shade,
        aspect='auto',
        origin='upper',
        cmap='Greys',
        interpolation='nearest',
        extent=extent,
        vmin=0,
        vmax=1,
        alpha=0.12,
        zorder=3,
    )
    Lon, Dep = np.meshgrid(lons, depths)
    ax.contour(
        Lon,
        Dep,
        sec_mask.T.astype(float),
        levels=[0.5],
        colors='k',
        linewidths=0.7,
        zorder=5,
    )
    has = np.any(sec_mask, axis=1)
    z_top = np.full(len(lons), np.nan)
    z_bot = np.full(len(lons), np.nan)
    for j in range(len(lons)):
        if not has[j]:
            continue
        zs = depths[sec_mask[j, :]]
        z_top[j] = float(np.min(zs))
        z_bot[j] = float(np.max(zs))
    ax.plot(
        lons,
        z_top,
        color='k',
        linewidth=0.7,
        linestyle='-',
        zorder=6,
        label='Slab top',
    )
    ax.plot(
        lons,
        z_bot,
        color='k',
        linewidth=0.7,
        linestyle='--',
        zorder=6,
        label='Slab bottom',
    )
    if with_legend:
        ax.legend(loc='lower left', fontsize=legend_fontsize, framealpha=0.85)


def perturbation_clim(
    cube: np.ndarray, feat_idx: int, pct: float = 98.0
) -> Tuple[float, float]:
    """对称色标：±百分位 |δ|。"""
    vals = cube[:, :, :, feat_idx]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -0.05, 0.05
    m = float(np.nanpercentile(np.abs(vals), pct))
    m = max(m, 1e-4)
    return -m, m


def velocity_clim(
    cube: np.ndarray,
    feat_idx: int,
    pct_lo: float = 2.0,
    pct_hi: float = 98.0,
) -> Tuple[float, float]:
    """绝对速度色标：分位数 min–max（非对称，不含负值）。"""
    vals = cube[:, :, :, feat_idx]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(vals, pct_lo))
    hi = float(np.nanpercentile(vals, pct_hi))
    if hi <= lo:
        hi = lo + 1e-3
    return lo, hi


# ==================== 等大地图子图网格 ====================
def make_equal_map_subplots(
    nrows: int,
    ncols: int,
    ccrs=None,
    *,
    panel_w: float = 6.0,
    panel_h: float = 5.0,
    wspace: float = 0.28,
    hspace: float = 0.38,
    top: float = 0.92,
    bottom: float = 0.06,
    left: float = 0.05,
    right: float = 0.98,
):
    """
    创建等大子图网格（每格 panel_w × panel_h inch）。
    固定 GridSpec，避免 tight_layout 把子图挤成不同大小。
    """
    fig = plt.figure(figsize=(panel_w * ncols, panel_h * nrows))
    gs = fig.add_gridspec(
        nrows,
        ncols,
        wspace=wspace,
        hspace=hspace,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        width_ratios=[1] * ncols,
        height_ratios=[1] * nrows,
    )
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            if ccrs is not None:
                axes[r, c] = fig.add_subplot(
                    gs[r, c], projection=ccrs.PlateCarree()
                )
            else:
                axes[r, c] = fig.add_subplot(gs[r, c])
    return fig, axes


# ==================== 生成30+种颜色 ====================
def generate_distinct_colors(n_colors: int) -> List:
    """生成视觉上可区分的颜色"""
    if n_colors <= 10:
        return plt.cm.tab10(np.linspace(0, 1, n_colors))
    elif n_colors <= 20:
        return plt.cm.tab20(np.linspace(0, 1, n_colors))
    else:
        # 组合多个colormap
        base_colors = []
        base_colors.extend(plt.cm.tab20(np.linspace(0, 1, 20)))
        base_colors.extend(plt.cm.Set3(np.linspace(0, 1, 12)))
        
        if n_colors > 32:
            # 添加更多颜色
            base_colors.extend(plt.cm.Pastel1(np.linspace(0, 1, 9)))
            base_colors.extend(plt.cm.Pastel2(np.linspace(0, 1, 8)))
        
        # 选择前n个
        return base_colors[:n_colors]


# ==================== 扰动特征空间交会图诊断 ====================
# 地幔热标定：温度主导的异常满足 dlnVp ≈ 0.5·dlnVs（Karato 1993；
# Cammarano et al. 2003）。偏离该线的簇才携带独立的 Vp/Vs 信息，
# 即成分或流体差异；沿线排列的簇只反映扰动幅度强弱。
MANTLE_SCALING_SLOPE = 0.5

# 泊松固体的 Vp/Vs = √3，原始速度交会图的岩性判别参考线
# （Christensen 1996；Hao et al. 2026 的 Vs-Vp 交会图即以此线为界）
POISSON_VPVS_RATIO = 1.732


def principal_axis(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    对二维点云做总体最小二乘（PCA）求主轴方向。

    与普通最小二乘不同，总体最小二乘对两个变量对称，适用于两个特征
    均含误差的扰动域交会图。

    Args:
        xy: 形状 (N, 2) 的点云

    Returns:
        (主轴单位向量, 次轴单位向量, 主轴/次轴奇异值之比即点云各向异性)
    """
    centered = np.asarray(xy, dtype=float) - np.mean(xy, axis=0)
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    anisotropy = float(sv[0] / sv[1]) if sv[1] > 0 else float('inf')
    return vt[0], vt[1], anisotropy


def crossplot_diagnostics(
    xy: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """
    计算簇中心在特征空间中的走向诊断量。

    判据 R = σ∥/σ⊥ 为簇中心沿点云主轴方向与垂直方向的标准差之比。
    R 越大说明簇中心越集中排列在主轴上，即聚类主要按扰动幅度切分，
    各簇之间缺乏独立的 Vp/Vs（成分/流体）差异；R 接近 1 则说明簇在
    垂直主轴方向上也有分异，具备物性域含义。

    Args:
        xy: 形状 (N, 2) 的点云，列序为 (横轴特征, 纵轴特征)
        labels: 形状 (N,) 的簇标签，负值视为无效

    Returns:
        含点数、簇数、相关系数、主轴斜率、点云各向异性、σ∥、σ⊥、R 的字典
    """
    xy = np.asarray(xy, dtype=float)
    labels = np.asarray(labels)
    valid = labels >= 0
    valid &= np.all(np.isfinite(xy), axis=1)
    xy, labels = xy[valid], labels[valid]

    empty = {
        'n_points': 0, 'n_clusters': 0, 'corr': np.nan, 'pc1_slope': np.nan,
        'cloud_anisotropy': np.nan, 'sigma_par': np.nan,
        'sigma_perp': np.nan, 'ratio_R': np.nan,
    }
    if len(xy) < 10:
        return empty

    pc1, pc2, anisotropy = principal_axis(xy)
    uniq = np.unique(labels)
    centers = np.stack([xy[labels == u].mean(axis=0) for u in uniq])
    centers_c = centers - centers.mean(axis=0)

    sigma_par = float(np.std(centers_c @ pc1))
    sigma_perp = float(np.std(centers_c @ pc2))

    return {
        'n_points': int(len(xy)),
        'n_clusters': int(len(uniq)),
        'corr': float(np.corrcoef(xy[:, 0], xy[:, 1])[0, 1]),
        'pc1_slope': float(pc1[1] / pc1[0]) if pc1[0] != 0 else float('inf'),
        'cloud_anisotropy': anisotropy,
        'sigma_par': sigma_par,
        'sigma_perp': sigma_perp,
        'ratio_R': (
            sigma_par / sigma_perp if sigma_perp > 0 else float('inf')
        ),
    }


# ==================== facies 自动解释（启发式，供剖面图标注） ====================
# 各深度带 δlnVs 分类阈值 (weak, strong)；数值为无量纲对数扰动。
# 依据：地壳扰动幅度 ~±10%，上地幔 ~±3%，过渡带/下地幔 ~±1–2%。
# 解释仅为基于扰动符号与幅度的推测标签，规则表见 2_3_Model_clustering_methods.md。
FACIES_DVS_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    'crust': (0.03, 0.10),
    'shallow': (0.010, 0.030),
    'lithosphere': (0.010, 0.030),
    'transition_zone': (0.005, 0.015),
    'lower_mantle': (0.004, 0.012),
}


def interpret_facies(band: str, dlnvp: float, dlnvs: float) -> str:
    """
    按深度带与簇平均扰动给出速度相的推测构造解释（英文短标签）。

    主判据为 δlnVs 的符号与幅度（对温度/部分熔融最敏感）；
    若 δlnVp 与 δlnVs 符号相反且幅度均显著，追加 [Vp-Vs decoupled]
    标记（提示成分/流体等非热成因）。

    Args:
        band: 深度带名（crust/shallow/lithosphere/transition_zone/lower_mantle）
        dlnvp: 簇平均 δlnVp
        dlnvs: 簇平均 δlnVs

    Returns:
        推测解释短标签（英文，用于图面标注）
    """
    b = str(band).lower()
    weak, strong = FACIES_DVS_THRESHOLDS.get(b, (0.010, 0.030))
    v = float(dlnvs) if np.isfinite(dlnvs) else 0.0

    if b == 'crust':
        if v <= -strong:
            name = 'Very slow crust (thick sediments / basin)'
        elif v <= -weak:
            name = 'Slow crust (sedimentary / warm)'
        elif v >= strong:
            name = 'Very fast crust (cratonic / mafic)'
        elif v >= weak:
            name = 'Fast crust (crystalline basement)'
        else:
            name = 'Average crust'
    elif b in ('lithosphere', 'shallow', 'all'):
        if v >= strong:
            name = 'Fast anomaly (cratonic root / slab)'
        elif v >= weak:
            name = 'Moderately fast mantle (cold)'
        elif v <= -strong:
            name = 'Slow anomaly (asthenospheric / back-arc)'
        elif v <= -weak:
            name = 'Moderately slow mantle (warm)'
        else:
            name = 'Ambient upper mantle'
    elif b == 'transition_zone':
        if v >= weak:
            name = 'Fast TZ anomaly (stagnant slab)'
        elif v <= -weak:
            name = 'Slow TZ anomaly (warm / hydrated)'
        else:
            name = 'Ambient transition zone'
    elif b == 'lower_mantle':
        if v >= weak:
            name = 'Fast anomaly (slab remnant)'
        elif v <= -weak:
            name = 'Slow anomaly (thermal upwelling)'
        else:
            name = 'Ambient lower mantle'
    else:
        name = 'Unclassified'

    if (
        np.isfinite(dlnvp)
        and abs(v) >= weak
        and abs(float(dlnvp)) >= 0.5 * weak
        and np.sign(float(dlnvp)) != np.sign(v)
    ):
        name += ' [Vp-Vs decoupled]'
    return name


def short_facies_label(full_label: str, max_len: int = 34) -> str:
    """
    压缩 interpret_facies 输出，供表格列显示。

    完整解释仍写在图下方图例；叶节点只保留主类名，避免 90° 旋转后不可读。
    """
    s = str(full_label).split(' [')[0].strip()
    decoupled = ' [Vp-Vs decoupled]' if '[Vp-Vs decoupled]' in str(full_label) else ''
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + '…'
    return s + decoupled


def facies_leaf_color(dlnvs: float, band: str) -> str:
    """按 δlnVs 符号与幅度返回叶节点/表格行配色（蓝=快，红=慢，灰=背景）"""
    b = str(band).lower()
    weak, strong = FACIES_DVS_THRESHOLDS.get(b, (0.010, 0.030))
    v = float(dlnvs) if np.isfinite(dlnvs) else 0.0
    if v >= weak:
        return '#1d4ed8' if v >= strong else '#3b82f6'
    if v <= -weak:
        return '#b91c1c' if v <= -strong else '#ef4444'
    return '#4b5563'


def _wrap_table_text(text: str, width: int = 32) -> str:
    """表格单元格自动换行，避免 facies 列被截断。"""
    s = str(text).strip()
    if not s or s == '—':
        return s
    return '\n'.join(textwrap.wrap(s, width=width, break_long_words=False))


def _phylogeny_grid_layout(
    n_pair_rows: int,
    n_cols: int,
    band_meta: List[Tuple[str, Any, np.ndarray, int]],
    tbl_row_u: float = 0.22,
) -> Tuple[List[float], List[Tuple[str, int]]]:
    """
    构建谱系图 GridSpec：每块 dend | 叶标缓冲 | table，块间 block_gap。

    Returns:
        height_ratios, row_specs（('dend'|'label_gap'|'tbl'|'block', pair_row)）
    """
    dend_h = 1.05
    label_gap = 0.38
    block_gap = 0.72
    height_ratios: List[float] = []
    row_specs: List[Tuple[str, int]] = []
    for pr in range(n_pair_rows):
        if pr > 0:
            height_ratios.append(block_gap)
            row_specs.append(('block', -1))
        i0 = pr * n_cols
        i1 = i0 + 1
        k_pair = band_meta[i0][3]
        if i1 < len(band_meta):
            k_pair = max(k_pair, band_meta[i1][3])
        tbl_h = tbl_row_u * (k_pair + 1)
        height_ratios.extend([dend_h, label_gap, tbl_h])
        row_specs.extend([
            ('dend', pr), ('label_gap', pr), ('tbl', pr),
        ])
    return height_ratios, row_specs


def _phylogeny_row_index(
    row_specs: List[Tuple[str, int]], kind: str, pair_row: int,
) -> int:
    """返回指定块、指定类型的 GridSpec 行号。"""
    for i, (k, pr) in enumerate(row_specs):
        if k == kind and pr == pair_row:
            return i
    raise KeyError(f'row {kind!r} for pair {pair_row}')


def _render_centroid_phylogeny_panel(
    ax_dend: plt.Axes,
    ax_tbl: plt.Axes,
    bname: str,
    binfo: Dict[str, Any],
    band_labels: np.ndarray,
    band_X: np.ndarray,
    cids: List[int],
    vp_idx: int,
    vs_idx: int,
    is_pert: bool,
    F: Dict[str, int],
    panel_label: Optional[str] = None,
    show_ylabel: bool = True,
    table_slot_rows: Optional[int] = None,
    table_row_h: float = 0.118,
    table_scale_y: float = 2.0,
) -> List[Dict[str, Any]]:
    """单深度带：上方 dendrogram（C# 叶标）+ 下方 facies 表；发表版 2×2 子块。"""
    from scipy.cluster.hierarchy import dendrogram, linkage

    centroids: List[np.ndarray] = []
    band_export: List[Dict[str, Any]] = []
    for local_idx, cid in enumerate(cids):
        m = band_labels == cid
        mu = np.nanmean(band_X[m], axis=0)
        centroids.append(mu)
        if is_pert and mu.size > max(vs_idx, vp_idx):
            facies = interpret_facies(
                bname, float(mu[vp_idx]), float(mu[vs_idx]),
            )
            short = short_facies_label(facies, max_len=42)
        else:
            facies = 'Absolute velocity cluster'
            short = '—'
        band_export.append({
            'cluster_id': cid,
            'display_id': local_idx,
            'n_voxels': int(np.sum(m)),
            'dlnvp': float(mu[vp_idx]) if mu.size > vp_idx else None,
            'dlnvs': float(mu[vs_idx]) if mu.size > vs_idx else None,
            'facies_interpretation': facies,
            'facies_short': short,
        })

    C = np.asarray(centroids, dtype=float)
    Z = linkage(C, method='ward')
    leaf_ids = [f'C{r["display_id"]}' for r in band_export]
    # 2×2 发表版每块较宽，K≤10 时水平叶标即可，避免 90° 标签伸入下方表格
    leaf_rot = 90. if len(cids) >= 11 else 0.
    leaf_fs = max(F['tick'] - 3, 7) if len(cids) >= 10 else max(F['tick'] - 2, 8)
    dendrogram(
        Z,
        ax=ax_dend,
        labels=leaf_ids,
        leaf_rotation=leaf_rot,
        leaf_font_size=leaf_fs,
        color_threshold=0,
        above_threshold_color='#94a3b8',
        count_sort='ascending',
    )
    band_title = MOHO_BAND_TITLES.get(bname, bname.replace('_', ' ').title())
    title_prefix = f'{panel_label} ' if panel_label else ''
    ax_dend.set_title(
        f'{title_prefix}{band_title} · $K={len(cids)}$',
        fontsize=F['title'] - 5,
        fontweight='bold',
        pad=4,
    )
    if show_ylabel:
        ax_dend.set_ylabel(
            'Ward distance', fontsize=F['label'] - 2, labelpad=4,
        )
    else:
        ax_dend.set_ylabel('')
    x_pad = 8 if leaf_rot else 6
    ax_dend.tick_params(axis='both', labelsize=F['tick'] - 2)
    ax_dend.tick_params(axis='x', pad=x_pad)
    ax_dend.margins(x=0.02)
    ax_dend.grid(True, axis='y', alpha=0.22, linestyle='-', linewidth=0.5)
    ax_dend.spines['top'].set_visible(False)
    ax_dend.spines['right'].set_visible(False)
    ax_dend.spines['bottom'].set_visible(True)
    ax_dend.spines['bottom'].set_linewidth(0.8)
    ax_dend.spines['bottom'].set_color('#334155')
    ax_dend.spines['left'].set_linewidth(0.8)
    ax_dend.spines['left'].set_color('#334155')

    # 叶节点按 fast/slow/ambient 着色（display_id 对齐）
    id_to_row = {f"C{r['display_id']}": r for r in band_export}
    for tick in ax_dend.get_xmajorticklabels():
        row = id_to_row.get(tick.get_text())
        if row and is_pert and row.get('dlnvs') is not None:
            tick.set_color(facies_leaf_color(row['dlnvs'], bname))
            tick.set_fontweight('bold')

    # 下方 facies 对照表（与叶节点 ID 对齐）
    ax_tbl.axis('off')
    if is_pert and band_export:
        rows_sorted = sorted(band_export, key=lambda r: r['display_id'])
        wrap_w = 34 if len(cids) <= 7 else 28
        table_data = [
            [
                f"C{r['display_id']}",
                f"{r['dlnvs']:+.3f}" if r['dlnvs'] is not None else '—',
                f"{r['dlnvp']:+.3f}" if r['dlnvp'] is not None else '—',
                _wrap_table_text(r['facies_short'], width=wrap_w),
            ]
            for r in rows_sorted
        ]
        col_labels = ['Cluster', r'$\delta\ln V_s$', r'$\delta\ln V_p$', 'Facies hint']
        n_tbl_rows = len(cids) + 1
        slot_rows = max(n_tbl_rows, int(table_slot_rows or n_tbl_rows))
        bbox_h = min(1.0, n_tbl_rows / slot_rows)
        tbl = ax_tbl.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='lower center',
            cellLoc='left',
            colLoc='center',
            bbox=[0.0, 0.0, 1.0, bbox_h],
        )
        tbl.auto_set_font_size(False)
        tbl_fs = max(F['annotation'] - 4, 6) if len(cids) >= 9 else max(F['annotation'] - 3, 7)
        tbl.set_fontsize(tbl_fs)
        tbl.scale(1.0, table_scale_y)
        col_widths = [0.10, 0.13, 0.13, 0.64]
        header_row_h = 0.072 * PHYLO_TABLE_ROW_SCALE
        data_row_h = 0.118 * PHYLO_TABLE_ROW_SCALE
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor('#e2e8f0')
            cell.set_linewidth(0.5)
            if col < len(col_widths):
                cell.set_width(col_widths[col])
            if row == 0:
                cell.set_facecolor('#f1f5f9')
                cell.set_text_props(fontweight='bold', color='#334155')
                cell.set_height(header_row_h)
            elif col == 0 and row > 0:
                r = rows_sorted[row - 1]
                cell.set_text_props(
                    fontweight='bold',
                    color=facies_leaf_color(r['dlnvs'] or 0.0, bname),
                )
                cell.set_height(data_row_h)
            elif row > 0:
                cell.set_facecolor('#ffffff' if row % 2 else '#f8fafc')
                cell.set_height(data_row_h)
                if col == 3:
                    cell.get_text().set_ha('left')
                    cell.get_text().set_va('center')
                    cell.PAD = 0.04

    return band_export


class GeologyConcordanceEvaluator:
    """
    结构参考数据辅助类（仅定性叠图与空间分带，无定量评分）

    - Slab2 俯冲板片（dep/thk，与 5_1_Basemap 同源）：切片等深线、剖面
      体轮廓、三维体掩膜叠绘
    - Moho 点数据：插值成网格，仅供 moho_4band 空间分带
    - USGS 地质省 / CN 地块边界线：浅部切片叠绘
    """

    def __init__(self, config: Any, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.base_config = BaseConfig()
        self.fonts = apply_clustering_plot_style(config)
        self._moho_grid: Optional[np.ndarray] = None
        self._slab_grid: Optional[np.ndarray] = None
        self._moho_meta: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._slab_meta: Optional[Tuple[np.ndarray, np.ndarray]] = None
        # 三维板片体掩膜缓存: (mask_3d, lons, lats, depths)
        self._slab_vol_cache: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = None

    @staticmethod
    def _import_cartopy():
        """尝试导入 cartopy；失败则返回 (None, None)"""
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            return ccrs, cfeature
        except ImportError:
            return None, None

    def _add_coastlines(
        self,
        ax,
        extent: Sequence[float],
        ccrs=None,
        cfeature=None,
        *,
        draw_labels: bool = True,
        left_labels: bool = True,
        bottom_labels: bool = True,
        match_depth_slices: bool = False,
    ) -> None:
        """
        在地图轴上添加海岸线/国界，并标注经纬度。
        extent: [lon_min, lon_max, lat_min, lat_max]
        match_depth_slices: True 时样式对齐 2-3-4（线宽/字号）
        """
        lon0, lon1, lat0, lat1 = [float(x) for x in extent]
        if ccrs is not None and cfeature is not None:
            try:
                if match_depth_slices:
                    ax.coastlines(
                        resolution='50m', color='gray', linewidth=0.45, zorder=3
                    )
                    ax.add_feature(
                        cfeature.BORDERS,
                        linestyle='--',
                        linewidth=0.5,
                        edgecolor='gray',
                        zorder=3,
                    )
                    label_size = 15
                    gl_lw, gl_alpha = 0.5, 0.5
                else:
                    ax.coastlines(
                        resolution='50m', color='gray', linewidth=0.4, zorder=6
                    )
                    ax.add_feature(
                        cfeature.BORDERS,
                        linestyle=':',
                        linewidth=0.4,
                        edgecolor='gray',
                        zorder=5,
                    )
                    label_size = 8
                    gl_lw, gl_alpha = 0.4, 0.45
                ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
                gl = ax.gridlines(
                    crs=ccrs.PlateCarree(),
                    draw_labels=draw_labels,
                    linewidth=gl_lw,
                    color='gray',
                    alpha=gl_alpha,
                    linestyle='--',
                    zorder=4,
                )
                gl.top_labels = False
                gl.right_labels = False
                gl.left_labels = left_labels
                gl.bottom_labels = bottom_labels
                try:
                    from cartopy.mpl.gridliner import (
                        LONGITUDE_FORMATTER,
                        LATITUDE_FORMATTER,
                    )
                    gl.xformatter = LONGITUDE_FORMATTER
                    gl.yformatter = LATITUDE_FORMATTER
                except Exception:
                    pass
                gl.xlabel_style = {'size': label_size}
                gl.ylabel_style = {'size': label_size}
                # 与 2-3-4 一致：填满子图；固定框比例（高/宽=5/6，对应 6×5 inch）
                if match_depth_slices:
                    try:
                        ax.set_aspect('auto')
                        ax.set_box_aspect(5.0 / 6.0)
                    except Exception:
                        pass
                return
            except Exception:
                pass
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(lat0, lat1)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    def _make_map_subplots(self, nrows: int, ncols: int, ccrs=None, **kwargs):
        """创建等大子图网格（委托模块级函数）。"""
        return make_equal_map_subplots(nrows, ncols, ccrs, **kwargs)

    def _plot_facies_slice_imshow(
        self,
        ax,
        label_slice: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        cmap,
        n_clusters: int,
        ccrs=None,
    ) -> None:
        """
        与 2-3-4 深度切片相同的填色方式：
        imshow + aspect='auto'，填满子图，比例一致。
        """
        extent = [
            float(lons.min()),
            float(lons.max()),
            float(lats.min()),
            float(lats.max()),
        ]
        plot_kw: Dict[str, Any] = {}
        if ccrs is not None:
            plot_kw['transform'] = ccrs.PlateCarree()
        ax.imshow(
            label_slice,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            interpolation='nearest',
            extent=extent,
            vmin=-0.5,
            vmax=max(n_clusters - 0.5, 0.5),
            zorder=1,
            **plot_kw,
        )

    def get_moho_grid(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> Optional[np.ndarray]:
        """获取插值到模型网格的 Moho 深度场（仅用于 moho_4band 分带）"""
        self._ensure_moho_grid(lons, lats)
        return self._moho_grid

    def _ensure_moho_grid(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> None:
        """插值 Moho 到模型经纬网格（缓存；不用于符合度）"""
        if (
            self._moho_meta is not None
            and np.array_equal(self._moho_meta[0], lons)
            and np.array_equal(self._moho_meta[1], lats)
            and self._moho_grid is not None
        ):
            return

        geo_cfg = self.config.geology_concordance
        data_dir = self.base_config.dirs['data']
        moho_name = geo_cfg.get('moho_file', '莫霍面深度数据.txt')
        self._moho_grid = self._interpolate_points_to_grid(
            data_dir / moho_name, lons, lats, kind='moho'
        )
        self._moho_meta = (lons, lats)

    def _slab_data_root(self) -> Path:
        slab_cfg = self.config.geology_concordance.get('slab') or {}
        rel = slab_cfg.get(
            'vis_data_subdir', '5_Visualization/data_EastAsia/Slab2'
        )
        return self.base_config.dirs['project_root'] / rel

    def _ensure_slab_grid(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> None:
        """
        拼接东亚 Slab2 depth .grd → 模型网格正深度场 (km)。
        重叠区取最浅板片顶面（nanmin |z|）。
        """
        if (
            self._slab_meta is not None
            and np.array_equal(self._slab_meta[0], lons)
            and np.array_equal(self._slab_meta[1], lats)
            and self._slab_grid is not None
        ):
            return

        slab_cfg = self.config.geology_concordance.get('slab') or {}
        if not slab_cfg.get('enabled', True):
            self._slab_grid = None
            self._slab_meta = (lons, lats)
            return

        root = self._slab_data_root()
        dist = root / slab_cfg.get('distribute_subdir', 'Slab2Distribute_Mar2018')
        regions = list(
            slab_cfg.get(
                'regions',
                [
                    'kur', 'ryu', 'izu', 'man', 'phi', 'sum', 'sul',
                    'him', 'hin', 'mak', 'cot', 'hal', 'png', 'sol',
                ],
            )
        )
        use_abs = bool(slab_cfg.get('use_absolute_depth', True))
        pattern_tmpl = slab_cfg.get('grd_glob', '{region}_slab2_dep_*.grd')

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        composite = np.full(lon_grid.shape, np.nan, dtype=np.float64)
        n_loaded = 0

        for region in regions:
            matches = sorted(dist.glob(pattern_tmpl.format(region=region)))
            if not matches:
                continue
            grd_path = matches[0]
            try:
                with xr.open_dataset(grd_path) as ds:
                    if 'z' in ds:
                        da = ds['z']
                    else:
                        da = ds[list(ds.data_vars)[0]]
                    # 坐标名 x=lon, y=lat
                    xname = 'x' if 'x' in da.dims else da.dims[-1]
                    yname = 'y' if 'y' in da.dims else da.dims[0]
                    # 裁到模型范围附近以加速
                    pad = 2.0
                    da = da.sel(
                        {
                            xname: slice(
                                float(lons.min()) - pad, float(lons.max()) + pad
                            ),
                            yname: slice(
                                float(lats.min()) - pad, float(lats.max()) + pad
                            ),
                        }
                    )
                    if da.size == 0:
                        continue
                    interp = da.interp(
                        {xname: xr.DataArray(lons, dims='lon'),
                         yname: xr.DataArray(lats, dims='lat')},
                        method='linear',
                    )
                    # 结果维度 (lat, lon)
                    arr = np.asarray(interp.transpose('lat', 'lon').values, dtype=np.float64)
                if use_abs:
                    arr = np.abs(arr)
                # 无效值：Slab2 常用极大值/NaN
                arr[~np.isfinite(arr)] = np.nan
                arr[(arr <= 0) | (arr > 800)] = np.nan
                if not np.any(np.isfinite(arr)):
                    continue
                if np.all(np.isnan(composite)):
                    composite = arr
                else:
                    composite = np.where(
                        np.isnan(composite),
                        arr,
                        np.where(np.isnan(arr), composite, np.fmin(composite, arr)),
                    )
                n_loaded += 1
            except Exception as e:
                self.logger.debug(f"  Slab2 {grd_path.name} 读取失败: {e}")
                continue

        if n_loaded == 0 or not np.any(np.isfinite(composite)):
            self.logger.warning(
                f"  ⚠️ 未加载到有效 Slab2 网格（目录: {dist}）"
            )
            self._slab_grid = None
        else:
            self._slab_grid = composite.astype(np.float32)
            self.logger.info(
                f"  ✅ Slab2 深度场: {n_loaded} 个板片 → 网格 {composite.shape}, "
                f"深度 [{np.nanmin(composite):.1f}, {np.nanmax(composite):.1f}] km, "
                f"覆盖 {np.sum(np.isfinite(composite))}/{composite.size}"
            )
        self._slab_meta = (lons, lats)

    def _interp_slab_grd_to_model(
        self,
        grd_path: Path,
        lons: np.ndarray,
        lats: np.ndarray,
        *,
        use_absolute: bool = False,
        valid_range: Optional[Tuple[float, float]] = None,
    ) -> Optional[np.ndarray]:
        """将单个 Slab2 .grd 双线性插值到模型 (lat, lon) 网格"""
        try:
            with xr.open_dataset(grd_path) as ds:
                if 'z' in ds:
                    da = ds['z']
                else:
                    da = ds[list(ds.data_vars)[0]]
                xname = 'x' if 'x' in da.dims else da.dims[-1]
                yname = 'y' if 'y' in da.dims else da.dims[0]
                pad = 2.0
                da = da.sel(
                    {
                        xname: slice(
                            float(lons.min()) - pad, float(lons.max()) + pad
                        ),
                        yname: slice(
                            float(lats.min()) - pad, float(lats.max()) + pad
                        ),
                    }
                )
                if da.size == 0:
                    return None
                interp = da.interp(
                    {
                        xname: xr.DataArray(lons, dims='lon'),
                        yname: xr.DataArray(lats, dims='lat'),
                    },
                    method='linear',
                )
                arr = np.asarray(
                    interp.transpose('lat', 'lon').values, dtype=np.float64
                )
            if use_absolute:
                arr = np.abs(arr)
            arr[~np.isfinite(arr)] = np.nan
            if valid_range is not None:
                lo, hi = valid_range
                arr[(arr < lo) | (arr > hi)] = np.nan
            if not np.any(np.isfinite(arr)):
                return None
            return arr.astype(np.float32)
        except Exception as e:
            self.logger.debug(f"  Slab2 {grd_path.name} 插值失败: {e}")
            return None

    def build_slab_volume_mask(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        depths: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        用 Slab2 dep + thk 构建与聚类同维度的三维板片掩膜。

        对每个板片区域：z_top = |dep|, z_bot = z_top + thk；
        体素满足 z_top ≤ z ≤ z_bot 则记为板片内。多板片取并集。

        Returns:
            mask: (n_lat, n_lon, n_depth) bool，或不可用时 None
        """
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        depths = np.asarray(depths, dtype=float)

        if self._slab_vol_cache is not None:
            m, lo, la, de = self._slab_vol_cache
            if (
                np.array_equal(lo, lons)
                and np.array_equal(la, lats)
                and np.array_equal(de, depths)
            ):
                return m

        slab_cfg = self.config.geology_concordance.get('slab') or {}
        if not slab_cfg.get('enabled', True):
            return None

        root = self._slab_data_root()
        dist = root / slab_cfg.get('distribute_subdir', 'Slab2Distribute_Mar2018')
        regions = list(
            slab_cfg.get(
                'regions',
                [
                    'kur', 'ryu', 'izu', 'man', 'phi', 'sum', 'sul',
                    'him', 'hin', 'mak', 'cot', 'hal', 'png', 'sol',
                ],
            )
        )
        use_abs = bool(slab_cfg.get('use_absolute_depth', True))
        dep_tmpl = slab_cfg.get('grd_glob', '{region}_slab2_dep_*.grd')
        thk_tmpl = slab_cfg.get('thk_glob', '{region}_slab2_thk_*.grd')

        n_lat, n_lon, n_dep = len(lats), len(lons), len(depths)
        mask = np.zeros((n_lat, n_lon, n_dep), dtype=bool)
        # 深度广播: (1,1,nz)
        z_axis = depths.reshape(1, 1, -1)
        n_loaded = 0

        for region in regions:
            dep_matches = sorted(dist.glob(dep_tmpl.format(region=region)))
            thk_matches = sorted(dist.glob(thk_tmpl.format(region=region)))
            if not dep_matches or not thk_matches:
                continue
            dep = self._interp_slab_grd_to_model(
                dep_matches[0],
                lons,
                lats,
                use_absolute=use_abs,
                valid_range=(0.0, 800.0),
            )
            thk = self._interp_slab_grd_to_model(
                thk_matches[0],
                lons,
                lats,
                use_absolute=False,
                valid_range=(1.0, 300.0),
            )
            if dep is None or thk is None:
                continue
            valid = np.isfinite(dep) & np.isfinite(thk) & (thk > 0)
            if not np.any(valid):
                continue
            z_top = np.where(valid, dep, np.nan)[:, :, np.newaxis]
            z_bot = np.where(valid, dep + thk, np.nan)[:, :, np.newaxis]
            local = (
                np.isfinite(z_top)
                & (z_axis >= z_top)
                & (z_axis <= z_bot)
            )
            mask |= local
            n_loaded += 1

        if n_loaded == 0 or not np.any(mask):
            self.logger.warning("  ⚠️ 未能构建 Slab2 三维体掩膜（缺 dep/thk）")
            self._slab_vol_cache = None
            return None

        frac = float(np.mean(mask))
        self.logger.info(
            f"  ✅ Slab2 体掩膜: {n_loaded} 板片, 形状 {mask.shape}, "
            f"占体素 {100.0 * frac:.2f}% ({int(mask.sum())}/{mask.size})"
        )
        self._slab_vol_cache = (mask, lons.copy(), lats.copy(), depths.copy())
        return mask

    def plot_slab_volume_comparison(
        self,
        labels_3d: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Path,
        model_name: str,
        pert_cube: Optional[np.ndarray] = None,
    ) -> None:
        """
        三维同维度对比可视化：聚类体 vs Slab2(dep+thk) 体掩膜。
        输出: 2-3-11_slab_volume_3d
          行0 — 水平切片 + 板片体边界
          行1 — 聚类垂直剖面 + 板片
          行2–3 — δlnVp / δlnVs 垂直剖面 + 板片（若提供 pert_cube）
        不修改现有切片 MI/ρ 符合度逻辑。
        """
        self.fonts = apply_clustering_plot_style(self.config)
        slab_cfg = self.config.geology_concordance.get('slab') or {}
        vol_cfg = slab_cfg.get('volume_viz') or {}
        if not vol_cfg.get('enabled', True):
            return

        lons = np.asarray(metadata['lons'], dtype=float)
        lats = np.asarray(metadata['lats'], dtype=float)
        depths = np.asarray(metadata['depths'], dtype=float)
        mask = self.build_slab_volume_mask(lons, lats, depths)
        if mask is None:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        slice_zs = list(vol_cfg.get('slice_depths_km', [100.0, 200.0, 400.0]))[:3]
        sec_lats = list(vol_cfg.get('section_latitudes', [20.0, 30.0, 40.0]))[:3]
        n_col = max(len(slice_zs), len(sec_lats), 1)

        n_clusters = int(len(np.unique(labels_3d[labels_3d >= 0])))
        colors = generate_distinct_colors(max(n_clusters, 2))
        cmap = mcolors.ListedColormap(colors[: max(n_clusters, 1)])
        map_extent = [
            float(lons.min()),
            float(lons.max()),
            float(lats.min()),
            float(lats.max()),
        ]
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        sec_extent = [
            float(lons.min()),
            float(lons.max()),
            float(depths.max()),
            float(depths.min()),
        ]

        has_pert = (
            pert_cube is not None
            and np.ndim(pert_cube) == 4
            and pert_cube.shape[-1] >= 2
            and pert_cube.shape[:3] == labels_3d.shape
        )
        n_rows = 4 if has_pert else 2

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
            use_cartopy = True
        except ImportError:
            use_cartopy = False
            ccrs = cfeature = LONGITUDE_FORMATTER = LATITUDE_FORMATTER = None

        fig = plt.figure(figsize=(6.2 * n_col, 4.8 * n_rows))
        gs = fig.add_gridspec(
            n_rows,
            n_col,
            wspace=0.16,
            hspace=0.32,
            left=0.05,
            right=0.90 if has_pert else 0.98,
            top=0.93,
            bottom=0.04,
            width_ratios=[1] * n_col,
            height_ratios=[1.0] * n_rows,
        )
        projection = ccrs.PlateCarree() if use_cartopy else None
        plot_tf = {'transform': projection} if use_cartopy else {}

        # —— 上行：水平切片 ——
        for col, z_target in enumerate(slice_zs):
            if use_cartopy:
                ax = fig.add_subplot(gs[0, col], projection=projection)
            else:
                ax = fig.add_subplot(gs[0, col])
            z_idx = int(np.argmin(np.abs(depths - float(z_target))))
            actual_z = float(depths[z_idx])
            lab = np.ma.masked_where(
                labels_3d[:, :, z_idx] < 0, labels_3d[:, :, z_idx]
            )
            slab_slice = mask[:, :, z_idx]
            ax.imshow(
                lab,
                aspect='auto',
                origin='lower',
                cmap=cmap,
                interpolation='nearest',
                extent=map_extent,
                vmin=-0.5,
                vmax=max(n_clusters - 0.5, 0.5),
                zorder=1,
                **plot_tf,
            )
            if np.any(slab_slice):
                # 板片体水平截面：半透明灰罩 + 黑边界
                shade = np.ma.masked_where(~slab_slice, np.ones(slab_slice.shape))
                ax.imshow(
                    shade,
                    aspect='auto',
                    origin='lower',
                    cmap='Greys',
                    interpolation='nearest',
                    extent=map_extent,
                    vmin=0,
                    vmax=1,
                    alpha=0.12,
                    zorder=3,
                    **plot_tf,
                )
                ax.contour(
                    lon_grid,
                    lat_grid,
                    slab_slice.astype(float),
                    levels=[0.5],
                    colors='k',
                    linewidths=0.7,
                    zorder=5,
                    **plot_tf,
                )
            if use_cartopy:
                ax.coastlines(
                    resolution='50m', color='gray', linewidth=0.45, zorder=6
                )
                ax.add_feature(
                    cfeature.BORDERS,
                    linestyle='--',
                    linewidth=0.4,
                    edgecolor='gray',
                    zorder=5,
                )
                ax.set_extent(map_extent, crs=projection)
                try:
                    ax.set_aspect('auto')
                except Exception:
                    pass
                gl = ax.gridlines(
                    crs=projection,
                    draw_labels=True,
                    linewidth=0.4,
                    color='gray',
                    alpha=0.45,
                    linestyle='--',
                )
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 11}
                gl.ylabel_style = {'size': 11}
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
            else:
                ax.set_xlim(map_extent[0], map_extent[1])
                ax.set_ylim(map_extent[2], map_extent[3])
                ax.set_aspect('auto')

            # 与同列下行剖面一一对应：每个子图只画一条纬度切线（同 2-3-5）
            if col < len(sec_lats):
                lat_t = float(sec_lats[col])
                lat_idx = int(np.argmin(np.abs(lats - lat_t)))
                actual_sec_lat = float(lats[lat_idx])
                if use_cartopy:
                    ax.plot(
                        [lons.min(), lons.max()],
                        [actual_sec_lat, actual_sec_lat],
                        color='red',
                        linewidth=1.8,
                        linestyle='-',
                        transform=projection,
                        zorder=7,
                        label='Section Line',
                    )
                    ax.scatter(
                        [lons.min(), lons.max()],
                        [actual_sec_lat, actual_sec_lat],
                        c='red',
                        s=40,
                        marker='o',
                        edgecolors='black',
                        linewidths=0.8,
                        transform=projection,
                        zorder=8,
                    )
                else:
                    ax.axhline(
                        actual_sec_lat,
                        color='red',
                        linewidth=1.8,
                        label='Section Line',
                    )
                ax.legend(loc='upper right', fontsize=self.fonts['legend'], framealpha=0.85)

            ax.set_title(
                f'{actual_z:.0f} km',
                fontsize=self.fonts['title'],
                fontweight='bold',
                pad=4,
            )
            ax.tick_params(labelsize=self.fonts['tick'])

        for col in range(len(slice_zs), n_col):
            fig.add_subplot(gs[0, col]).axis('off')

        # —— 行1：聚类垂直剖面 ——
        for col, lat_t in enumerate(sec_lats):
            ax = fig.add_subplot(gs[1, col])
            lat_idx = int(np.argmin(np.abs(lats - float(lat_t))))
            actual_lat = float(lats[lat_idx])
            sec_lab = labels_3d[lat_idx, :, :]  # (lon, depth)
            sec_mask = mask[lat_idx, :, :]
            sec_plot = np.ma.masked_where(sec_lab < 0, sec_lab)
            ax.imshow(
                sec_plot.T,
                aspect='auto',
                origin='upper',
                cmap=cmap,
                interpolation='nearest',
                extent=sec_extent,
                vmin=-0.5,
                vmax=max(n_clusters - 0.5, 0.5),
                zorder=1,
            )
            overlay_slab_on_section(
                ax,
                sec_mask,
                lons,
                depths,
                with_legend=(col == 0),
                legend_fontsize=self.fonts['legend'],
            )
            F = self.fonts
            ax.set_xlabel('Longitude (°)', fontsize=F['label'])
            ax.set_ylabel('Depth (km)', fontsize=F['label'])
            ax.tick_params(labelsize=F['tick'])
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_title(
                f'Cluster  |  Lat={actual_lat:.1f}°',
                fontsize=F['title'],
                fontweight='bold',
                pad=4,
            )
            ax.set_ylim(float(depths.max()), float(depths.min()))

        for col in range(len(sec_lats), n_col):
            fig.add_subplot(gs[1, col]).axis('off')

        # —— 行2–3：δlnVp / δlnVs 垂直剖面 ——
        if has_pert:
            feat_titles = [r'$\delta\ln V_p$', r'$\delta\ln V_s$']
            for r_off, (fi, ftitle) in enumerate(zip((0, 1), feat_titles)):
                row = 2 + r_off
                vmin, vmax = perturbation_clim(pert_cube, fi)
                last_im = None
                for col, lat_t in enumerate(sec_lats):
                    ax = fig.add_subplot(gs[row, col])
                    lat_idx = int(np.argmin(np.abs(lats - float(lat_t))))
                    actual_lat = float(lats[lat_idx])
                    sec = np.ma.masked_invalid(pert_cube[lat_idx, :, :, fi])
                    last_im = ax.imshow(
                        sec.T,
                        aspect='auto',
                        origin='upper',
                        cmap='RdBu_r',
                        interpolation='nearest',
                        extent=sec_extent,
                        vmin=vmin,
                        vmax=vmax,
                        zorder=1,
                    )
                    overlay_slab_on_section(
                        ax, mask[lat_idx, :, :], lons, depths, with_legend=False
                    )
                    ax.set_title(
                        f'{ftitle}  |  Lat={actual_lat:.1f}°',
                        fontsize=self.fonts['title'],
                        fontweight='bold',
                        pad=4,
                    )
                    ax.set_xlabel('Longitude (°)', fontsize=self.fonts['label'])
                    if col == 0:
                        ax.set_ylabel('Depth (km)', fontsize=self.fonts['label'])
                    ax.tick_params(labelsize=self.fonts['tick'])
                    ax.set_ylim(float(depths.max()), float(depths.min()))
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    if col == len(sec_lats) - 1 and last_im is not None:
                        cbar = fig.colorbar(
                            last_im, ax=ax, fraction=0.046, pad=0.02
                        )
                        cbar.set_label(ftitle, fontsize=self.fonts['label'])
                        cbar.ax.tick_params(labelsize=self.fonts['tick'])
                for col in range(len(sec_lats), n_col):
                    fig.add_subplot(gs[row, col]).axis('off')

        fig.suptitle(
            f'{model_name} — Cluster / dlnV vs Slab2 Volume '
            f'(dep+thk mask; black = slab body)',
            fontsize=self.fonts['suptitle'],
            fontweight='bold',
            y=0.985,
        )

        stem = '2-3-11_slab_volume_3d'
        save_fmts = set(self.config.visualization.get('save_formats', ['jpg']))
        save_fmts.add('png')
        for fmt in sorted(save_fmts):
            fig.savefig(
                output_dir / f'{stem}.{fmt}',
                dpi=self.config.visualization['dpi'],
            )
        plt.close(fig)
        self.logger.info(f"      ✅ 保存: {stem} ({', '.join(sorted(save_fmts))})")

    def _interpolate_points_to_grid(
        self,
        path: Path,
        lons: np.ndarray,
        lats: np.ndarray,
        kind: str,
    ) -> Optional[np.ndarray]:
        """读取散点结构深度并插值到规则网格"""
        if not path.exists():
            self.logger.warning(f"  ⚠️ 结构文件不存在: {path}")
            return None
        try:
            df = pd.read_csv(path, sep=r'\s+', engine='python', encoding='utf-8')
            cols = list(df.columns)
            if len(cols) < 3:
                raise ValueError(f"列数不足: {cols}")
            lon_col, lat_col, depth_col = cols[0], cols[1], cols[2]
            pts = df[[lon_col, lat_col, depth_col]].apply(pd.to_numeric, errors='coerce')
            pts = pts.dropna()
            # 限制到模型范围附近
            lon_min, lon_max = float(lons.min()), float(lons.max())
            lat_min, lat_max = float(lats.min()), float(lats.max())
            pad = 2.0
            pts = pts[
                (pts[lon_col] >= lon_min - pad)
                & (pts[lon_col] <= lon_max + pad)
                & (pts[lat_col] >= lat_min - pad)
                & (pts[lat_col] <= lat_max + pad)
            ]
            if len(pts) < 20:
                self.logger.warning(f"  ⚠️ {kind} 有效点过少 ({len(pts)})")
                return None

            lon_grid, lat_grid = np.meshgrid(lons, lats)
            grid = griddata(
                pts[[lon_col, lat_col]].values,
                pts[depth_col].values,
                (lon_grid, lat_grid),
                method='linear',
            )
            # 线性插值空洞用最近邻填补
            nan_mask = ~np.isfinite(grid)
            if np.any(nan_mask):
                grid_nn = griddata(
                    pts[[lon_col, lat_col]].values,
                    pts[depth_col].values,
                    (lon_grid, lat_grid),
                    method='nearest',
                )
                grid[nan_mask] = grid_nn[nan_mask]

            self.logger.info(
                f"  ✅ {kind} 插值完成: {len(pts)} 点 → "
                f"网格 {grid.shape}, 深度 "
                f"[{np.nanmin(grid):.1f}, {np.nanmax(grid):.1f}] km"
            )
            return grid.astype(np.float32)
        except Exception as e:
            self.logger.warning(f"  ⚠️ 读取/插值 {kind} 失败: {e}")
            return None

    # ------------------------------------------------------------------ #
    # 地质省 / CN 地块边界线叠图工具（供 3×3 深度切片图复用）
    # ------------------------------------------------------------------ #
    def _geology_dir(self) -> Path:
        surf = (self.config.geology_concordance.get('surface_units') or {})
        sub = surf.get('geology_subdir', 'geology')
        return self.base_config.dirs['data'] / sub

    def get_overlap_region(self) -> List[float]:
        """三模型公共制图范围 [lon_min, lon_max, lat_min, lat_max]"""
        surf = self.config.geology_concordance.get('surface_units') or {}
        region = surf.get('overlap_region')
        if region and len(region) == 4:
            return [float(x) for x in region]

        models = self.config.data.get('target_models') or []
        pattern = self.config.data.get(
            'netcdf_pattern',
            'processed/{model_name}/{model_name}_original.nc',
        )
        models_base = self.base_config.dirs['data'] / self.config.data.get(
            'models_base_dir', 'models'
        )
        lon_mins, lon_maxs, lat_mins, lat_maxs = [], [], [], []
        for model_name in models:
            nc = models_base / pattern.format(model_name=model_name)
            if not nc.exists():
                continue
            with xr.open_dataset(nc) as ds:
                lons = ds['longitude'].values
                lats = ds['latitude'].values
                lon_mins.append(float(lons.min()))
                lon_maxs.append(float(lons.max()))
                lat_mins.append(float(lats.min()))
                lat_maxs.append(float(lats.max()))
        if not lon_mins:
            return [80.0, 150.0, 10.0, 55.0]
        return [max(lon_mins), min(lon_maxs), max(lat_mins), min(lat_maxs)]

    @staticmethod
    def _parse_gmt_segments(
        path: Path,
        as_polygons: bool = False,
    ) -> List[Dict[str, Any]]:
        """解析 GMT 多段线/多边形 → [{coords, name, code}, ...]"""
        if not path.exists():
            return []
        text = path.read_text(encoding='utf-8', errors='replace')
        segments: List[List[str]] = []
        cur: Optional[List[str]] = None
        for line in text.splitlines(keepends=True):
            if line.startswith('>'):
                if cur is not None:
                    segments.append(cur)
                cur = [line]
            elif cur is None:
                continue
            else:
                cur.append(line)
        if cur is not None:
            segments.append(cur)

        out: List[Dict[str, Any]] = []
        for seg in segments:
            body = ''.join(seg)
            name_m = re.search(r'# @D[^|]*\|[^|]*\|[^|]*\|"([^"]*)"', body)
            code_m = re.search(
                r'# @D[^|]*\|[^|]*\|(\d+)\|"', body
            )
            name = name_m.group(1) if name_m else ''
            code = int(code_m.group(1)) if code_m else -1
            coords: List[Tuple[float, float]] = []
            for ln in seg:
                s = ln.strip()
                if not s or s.startswith('#') or s == '>':
                    continue
                toks = s.split()
                if len(toks) < 2:
                    continue
                try:
                    coords.append((float(toks[0]), float(toks[1])))
                except ValueError:
                    continue
            min_pts = 3 if as_polygons else 2
            if len(coords) < min_pts:
                continue
            if as_polygons and coords[0] != coords[-1]:
                coords = coords + [coords[0]]
            out.append({'coords': coords, 'name': name, 'code': code})
        return out

    def _load_overlay_lines(self) -> List[Dict[str, Any]]:
        """加载叠图用地质省线 + CN 地块线"""
        surf = self.config.geology_concordance.get('surface_units') or {}
        geo_dir = self._geology_dir()
        lines: List[Dict[str, Any]] = []
        for fname in surf.get('province_line_files', []):
            path = geo_dir / fname
            segs = self._parse_gmt_segments(path, as_polygons=False)
            # 过滤缓存不存在时回退原始多边形边界
            if not segs:
                alt = geo_dir / fname.replace(
                    '_nosouth_rim_v3_80_150_10_55', ''
                ).replace('_nosouth_north_rim_v2_80_150_10_55', '')
                if 'prv1ec' in fname:
                    segs = self._parse_gmt_segments(
                        geo_dir / 'prv1ec.gmt', as_polygons=True
                    )
                elif 'prv3bl' in fname:
                    segs = self._parse_gmt_segments(
                        geo_dir / 'prv3bl.gmt', as_polygons=True
                    )
                # 多边形当线画
            for s in segs:
                lines.append({**s, 'kind': 'province', 'file': fname})
        for fname in surf.get('block_line_files', []):
            path = geo_dir / fname
            for s in self._parse_gmt_segments(path, as_polygons=False):
                lines.append({**s, 'kind': 'block', 'file': fname})
        return lines

    def _draw_geology_lines_on_ax(
        self,
        ax,
        lines: Optional[List[Dict[str, Any]]] = None,
        region: Optional[Sequence[float]] = None,
        transform=None,
    ) -> None:
        """在 matplotlib 轴上叠绘地质省/地块线"""
        lines = lines if lines is not None else getattr(
            self, '_surface_overlay_lines', None
        )
        if not lines:
            lines = self._load_overlay_lines()
        region = list(region) if region is not None else self.get_overlap_region()
        lon0, lon1, lat0, lat1 = region
        plot_kw = {'transform': transform} if transform is not None else {}
        for seg in lines:
            coords = seg['coords']
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            # 段中心不在范围内则跳过
            if (
                max(xs) < lon0
                or min(xs) > lon1
                or max(ys) < lat0
                or min(ys) > lat1
            ):
                continue
            kind = seg.get('kind', 'province')
            if kind == 'block':
                ax.plot(
                    xs, ys, color='red', linewidth=0.55, alpha=0.95, zorder=5,
                    **plot_kw,
                )
            else:
                ax.plot(
                    xs, ys, color='k', linewidth=0.45, alpha=0.85, zorder=4,
                    **plot_kw,
                )


class EnhancedClusteringVisualizer:
    """增强的聚类可视化器（包含BIC分析、概率分布、GMM vs K-means对比）"""
    
    def __init__(self, config: Any, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.base_config = BaseConfig()
        self.fonts = apply_clustering_plot_style(config)
        self.processor = None
    
    def set_processor(self, processor):
        """设置数据处理器引用"""
        self.processor = processor
    
    def plot_wkmeans_k_selection(
        self,
        k_selection: Dict[str, Dict[str, Any]],
        output_dir: Path,
        model_name: str,
    ) -> None:
        """
        绘制 W-k-means 的选 K 判据：肘部法 SSE 曲线 + 轮廓系数曲线。

        每个深度带一列，上排为 SSE-K 曲线（含首末端点连线，肘点即离该线
        最远的点），下排为轮廓系数-K 曲线。两条曲线并置是有意的：SSE 决定
        K，轮廓系数检验该 K 下簇的紧致度与分离度（Nainggolan et al., 2019）。
        二者最优 K 不一致时在标题中标出，这一分歧本身需要在方法学中说明。

        与 GMM 路径的 BIC 图是两套不能混用的判据——k-means 无似然函数，
        BIC/AIC 无从定义；GMM 的 BIC 曲线有极小值，而 SSE 曲线单调下降。

        Args:
            k_selection: {深度带名: 选 K 诊断字典}
            output_dir: 图件输出目录
            model_name: 模型名
        """
        bands = [(b, d) for b, d in k_selection.items() if d.get('k_values')]
        if not bands:
            return
        self.logger.info("    📊 绘制 W-k-means 选 K 判据...")
        F = self.fonts = apply_clustering_plot_style(self.config)

        has_beta = any(d.get('beta_scan') for _, d in bands)
        n = len(bands)
        n_rows = 2 if has_beta else 1
        fig, axes = plt.subplots(n_rows, n, figsize=(5.2 * n, 4.6 * n_rows),
                                 squeeze=False)

        c_sse, c_sil = 'tab:blue', 'tab:red'
        for i, (bname, d) in enumerate(bands):
            k = np.asarray(d['k_values'], dtype=float)
            sse = np.asarray(d['sse'], dtype=float)
            sil = np.asarray(d['silhouette'], dtype=float)
            k_elbow, k_sil, k_sel = d['k_elbow'], d['k_silhouette'], d['optimal_n']
            k_loc = d.get('k_local_maxima', [])

            # SSE 与轮廓系数共用横轴、分列左右纵轴：选 K 须同时读这两条曲线，
            # 分置两图会让读者难以判断二者在同一 K 处的取舍
            ax = axes[0, i]
            ax.plot(k, sse, 'o-', color=c_sse, lw=1.8, ms=5.5)
            ax.set_xlabel('Number of Clusters (K)', fontsize=F['label'] - 2)
            ax.set_ylabel('SSE', color=c_sse, fontsize=F['label'] - 2)
            ax.tick_params(axis='y', labelcolor=c_sse)
            ax.grid(True, alpha=0.3)

            # 轮廓系数是子样本估计，误差棒为多次独立抽样的标准差：
            # 它远小于相邻 K 之间的差异，即证明子样本规模足以支撑选 K
            sd = np.asarray(d.get('silhouette_sd') or np.zeros_like(sil))
            ax2 = ax.twinx()
            ax2.errorbar(k, sil, yerr=sd, fmt='o-', color=c_sil, lw=1.8,
                         ms=5.5, capsize=3, elinewidth=1.2)
            ax2.set_ylabel('Mean Silhouette', color=c_sil,
                           fontsize=F['label'] - 2)
            ax2.tick_params(axis='y', labelcolor=c_sil)

            # 选定的 K 在两条曲线上各圈一次
            j = int(np.argmin(np.abs(k - k_sel)))
            for a, y, col in ((ax, sse, c_sse), (ax2, sil, c_sil)):
                a.plot(k[j], y[j], 'o', ms=15, mfc='none', mec='red', mew=2.0,
                       zorder=5)
            ax2.annotate(f'K={k_sel}', (k[j], sil[j]),
                         textcoords='offset points', xytext=(8, 8),
                         color='red', fontsize=F['annotation'] - 3,
                         fontweight='bold')

            note = f'elbow K={k_elbow}'
            note += (f', silhouette local max K={k_loc}' if k_loc
                     else ', no silhouette local max')
            npt = d.get('n_points_scanned')
            head = f'{bname}' + (f'  (N = {npt:,})' if npt else '')
            ax.set_title(f'{head}\n{note}', fontsize=F['title'] - 5,
                         fontweight='bold')

            # β 扫描：平均轮廓系数随权重指数的变化
            if has_beta:
                axb = axes[1, i]
                bs = d.get('beta_scan')
                if not bs:
                    axb.axis('off')
                    continue
                betas = np.asarray(bs['beta_values'], dtype=float)
                bsil = np.asarray(bs['silhouette'], dtype=float)
                axb.plot(betas, bsil, 'o-', color=c_sse, lw=1.5, ms=4,
                         label='Silhouette Scores')
                jb = int(np.nanargmax(bsil))
                axb.plot(betas[jb], bsil[jb], 'o', ms=14, mfc='none',
                         mec='red', mew=2.0, label='Best Beta')
                axb.annotate(rf'$\beta$ = {bs["best_beta"]:.1f}',
                             (betas[jb], bsil[jb]),
                             textcoords='offset points', xytext=(8, 8),
                             color='red', fontsize=F['annotation'] - 3,
                             fontweight='bold')
                axb.axvline(bs['beta_in_use'], color='0.4', ls='--', lw=1.4,
                            label=f"In use ({bs['beta_in_use']:.1f})")
                axb.set_xlabel('Beta', fontsize=F['label'] - 2)
                axb.set_ylabel('Mean Silhouette', fontsize=F['label'] - 2)
                axb.legend(fontsize=F['legend'] - 4)
                axb.grid(True, alpha=0.3)

        fig.suptitle(
            f'{model_name} — W-k-means parameter selection '
            f'(elbow + silhouette; Nainggolan et al., 2019)',
            fontsize=F['suptitle'] - 4, fontweight='bold',
        )
        fig.tight_layout()

        stem = '2-3-15_wkmeans_k_selection'
        for fmt in self.config.visualization.get('save_formats', ['jpg']):
            fig.savefig(output_dir / f'{stem}.{fmt}',
                        dpi=self.config.visualization['dpi'],
                        bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"      ✅ 保存: {stem}")

    def plot_alt_clustering_diagnostics(
        self,
        algorithm: str,
        method_diagnostics: Dict[str, Dict[str, Any]],
        per_band: Optional[Dict[str, Any]],
        output_dir: Path,
        model_name: str,
    ) -> None:
        """
        HDBSCAN / 层次聚类的方法学诊断图

        - HDBSCAN: 各带簇规模分布与噪声比例
        - Hierarchical: 子样本树状图 + 目标 K 切分线

        Args:
            algorithm: 'hdbscan' | 'hierarchical'
            method_diagnostics: {深度带名: 诊断字典}
            per_band: 分带聚类结果（含 metrics）
            output_dir: 图件输出目录
            model_name: 模型名
        """
        if not method_diagnostics:
            return
        algo = str(algorithm).lower()
        self.logger.info(f"    📊 绘制 {algo.upper()} 方法学诊断图...")
        F = self.fonts = apply_clustering_plot_style(self.config)
        bands = list(method_diagnostics.items())
        n = len(bands)

        if algo == 'hdbscan':
            fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), squeeze=False)
            for i, (bname, diag) in enumerate(bands):
                ax = axes[0, i]
                pb = (per_band or {}).get(bname, {})
                sizes = pb.get('metrics', {}).get('cluster_sizes', {})
                if sizes:
                    ids = sorted(sizes.keys())
                    vals = [sizes[k] for k in ids]
                    ax.bar([str(k) for k in ids], vals, color='tab:purple', alpha=0.85)
                noise = pb.get('metrics', {}).get('noise_fraction', 0.0)
                ax.set_title(
                    f'{bname}\nK={pb.get("n_clusters", "?")}, '
                    f'noise={noise:.1%}\n'
                    f'min_cluster_size={diag.get("min_cluster_size", "?")}',
                    fontsize=F['title'] - 4, fontweight='bold',
                )
                ax.set_xlabel('Cluster ID', fontsize=F['label'] - 2)
                ax.set_ylabel('Voxel count', fontsize=F['label'] - 2)
                ax.grid(True, alpha=0.3, axis='y')
            fig.suptitle(
                f'{model_name} — HDBSCAN cluster size distribution',
                fontsize=F['suptitle'] - 4, fontweight='bold',
            )
            stem = '2-3-15_hdbscan_diagnostics'

        elif algo == 'hierarchical':
            fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5), squeeze=False)
            for i, (bname, diag) in enumerate(bands):
                ax = axes[0, i]
                Z = np.asarray(diag.get('linkage_matrix', []), dtype=float)
                if Z.size:
                    from scipy.cluster.hierarchy import dendrogram
                    dendrogram(
                        Z, ax=ax, truncate_mode='lastp', p=12,
                        leaf_rotation=90., color_threshold=0,
                        above_threshold_color='0.4',
                    )
                    ax.axhline(
                        Z[-(int(diag.get('n_clusters', 2)) - 1), 2],
                        color='red', ls='--', lw=1.5,
                        label=f"Cut K={diag.get('n_clusters')}",
                    )
                pb = (per_band or {}).get(bname, {})
                ax.set_title(
                    f'{bname}\n{diag.get("linkage", "ward")} linkage, '
                    f'K={diag.get("n_clusters", "?")}\n'
                    f'n={diag.get("dendrogram_n_points", "?")} (dendrogram sample)',
                    fontsize=F['title'] - 5, fontweight='bold',
                )
                ax.set_xlabel('Sample index / merged cluster', fontsize=F['label'] - 3)
                ax.set_ylabel('Distance', fontsize=F['label'] - 3)
                if i == 0:
                    ax.legend(fontsize=F['legend'] - 4)
            fig.suptitle(
                f'{model_name} — Hierarchical clustering dendrogram',
                fontsize=F['suptitle'] - 4, fontweight='bold',
            )
            stem = '2-3-15_hierarchical_diagnostics'
        else:
            return

        fig.tight_layout()
        for fmt in self.config.visualization.get('save_formats', ['jpg']):
            fig.savefig(
                output_dir / f'{stem}.{fmt}',
                dpi=self.config.visualization['dpi'],
                bbox_inches='tight',
            )
        plt.close(fig)
        self.logger.info(f"      ✅ 保存: {stem}")

    def plot_centroid_phylogeny(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        metadata: Dict[str, Any],
        per_band: Dict[str, Any],
        output_dir: Path,
        model_name: str,
        algorithm_name: str = 'gmm',
    ) -> Optional[Dict[str, Any]]:
        """
        GMM 簇心在 (δlnVp, δlnVs) 空间的 Ward 谱系树 + facies 对照表（2×2 发表版）。

        四深度带按 (a–d) 排列：上排 crust / lithosphere，下排 TZ / lower mantle。
        每块上方 dendrogram、下方 facies 表；叶节点按 fast/slow 着色。

        Returns:
            各带簇心 δlnV 与 facies 标签，供 JSON 导出
        """
        self.logger.info(f"    📊 绘制簇心谱系 dendrogram ({algorithm_name})...")
        F = apply_clustering_plot_style(self.config)

        labels = np.asarray(labels)
        X = np.asarray(X)
        X_phys = X.copy()
        if (
            self.processor is not None
            and getattr(self.processor, 'scaler', None) is not None
        ):
            try:
                X_phys = self.processor.scaler.inverse_transform(X)
            except Exception:
                X_phys = X

        feature_names = list(
            metadata.get('feature_display_names')
            or metadata.get('features', ['vp', 'vs'])
        )
        vp_idx = next(
            (i for i, nm in enumerate(feature_names) if 'vp' in str(nm).lower()),
            0,
        )
        vs_idx = next(
            (i for i, nm in enumerate(feature_names) if 'vs' in str(nm).lower()),
            min(1, X_phys.shape[1] - 1),
        )
        feat_space = str(metadata.get('feature_space', '') or '')
        is_pert = 'perturbation' in feat_space or any(
            'dln' in str(n).lower() for n in feature_names
        )

        band_items = [
            (bname, binfo)
            for bname, binfo in per_band.items()
            if binfo.get('point_mask') is not None
        ]
        if not band_items:
            return None

        band_items = _sort_bands_for_k_plot(band_items)
        n = len(band_items)
        n_cols = 2 if n > 1 else 1
        n_pair_rows = int(np.ceil(n / n_cols))

        # 预计算各带 K，用于固定 dend 高度 + 同行表格行槽对齐
        band_meta: List[Tuple[str, Any, np.ndarray, int]] = []
        for bname, binfo in band_items:
            pmask = np.asarray(binfo['point_mask'], dtype=bool)
            bl = labels[pmask]
            n_cids = len([
                c for c in np.unique(bl[bl >= 0]) if c >= 0
            ])
            band_meta.append((bname, binfo, pmask, n_cids))

        max_k = max(m[3] for m in band_meta)
        panel_w = max(6.6, 0.40 * max(max_k, 5) + 2.0)
        tbl_row_u = 0.22 * PHYLO_TABLE_ROW_SCALE
        height_ratios, row_specs = _phylogeny_grid_layout(
            n_pair_rows, n_cols, band_meta, tbl_row_u,
        )

        unit_in = 1.22
        fig_h = sum(height_ratios) * unit_in + 2.2
        fig = plt.figure(figsize=(panel_w * n_cols, fig_h))
        gs = GridSpec(
            len(height_ratios), n_cols, figure=fig,
            height_ratios=height_ratios,
            hspace=0.0, wspace=0.28,
            top=0.88, bottom=0.07, left=0.10, right=0.98,
        )
        panel_labels = [f'({chr(ord("a") + i)})' for i in range(n)]
        export: Dict[str, Any] = {
            'model': model_name,
            'algorithm': algorithm_name,
            'bands': {},
        }

        # 块间空白行
        for i, (kind, _) in enumerate(row_specs):
            if kind != 'block':
                continue
            for c in range(n_cols):
                ax_blk = fig.add_subplot(gs[i, c])
                ax_blk.axis('off')

        for idx, (bname, binfo, pmask, n_cids) in enumerate(band_meta):
            pair_row, col = divmod(idx, n_cols)
            dend_row = _phylogeny_row_index(row_specs, 'dend', pair_row)
            gap_row = _phylogeny_row_index(row_specs, 'label_gap', pair_row)
            tbl_row = _phylogeny_row_index(row_specs, 'tbl', pair_row)

            ax_dend = fig.add_subplot(gs[dend_row, col])
            ax_gap = fig.add_subplot(gs[gap_row, col])
            ax_gap.axis('off')
            ax_tbl = fig.add_subplot(gs[tbl_row, col])

            i0 = pair_row * n_cols
            i1 = i0 + 1
            pair_k = band_meta[i0][3]
            if i1 < len(band_meta):
                pair_k = max(pair_k, band_meta[i1][3])
            table_slot_rows = pair_k + 1

            band_labels = labels[pmask]
            band_X = X_phys[pmask]
            cids = sorted(int(c) for c in np.unique(band_labels[band_labels >= 0]))
            if len(cids) < 2:
                ax_dend.text(
                    0.5, 0.5, f'{bname}\n$K={len(cids)}$ (too few for tree)',
                    ha='center', va='center', transform=ax_dend.transAxes,
                )
                ax_dend.set_axis_off()
                ax_tbl.set_axis_off()
                continue

            band_export = _render_centroid_phylogeny_panel(
                ax_dend, ax_tbl, bname, binfo, band_labels, band_X,
                cids, vp_idx, vs_idx, is_pert, F,
                panel_label=panel_labels[idx],
                show_ylabel=(col == 0),
                table_slot_rows=table_slot_rows,
                table_row_h=0.118 * PHYLO_TABLE_ROW_SCALE,
                table_scale_y=1.75 * PHYLO_TABLE_ROW_SCALE,
            )
            export['bands'][bname] = band_export

        short_name = _short_model_label(model_name)
        fig.suptitle(
            f'{short_name} — GMM cluster centroid phylogeny (Ward linkage)',
            fontsize=F['suptitle'] - 5,
            fontweight='bold',
            y=0.97,
        )
        if is_pert:
            fig.text(
                0.5, 0.018,
                'Blue = fast $\\delta\\ln V_s$ · Red = slow · Gray = ambient  |  '
                'Facies hints are heuristic ($\\delta\\ln V$-based), not lithology',
                ha='center', va='bottom',
                fontsize=max(F['annotation'] - 2, 8),
                color='#64748b', style='italic',
            )

        stem = f'2-3-17_{algorithm_name}_centroid_phylogeny'
        for fmt in self.config.visualization.get('save_formats', ['jpg']):
            fig.savefig(
                output_dir / f'{stem}.{fmt}',
                dpi=self.config.visualization['dpi'],
                facecolor='white',
            )
        plt.close(fig)
        self.logger.info(f"      ✅ 保存: {stem}")
        return export

    def plot_k_selection_panels(
        self,
        per_band: Dict[str, Any],
        output_dir: Path,
        model_name: str,
    ) -> bool:
        """
        GMM 选 K 判据图（2×2 发表版，双轴 BIC + 轮廓系数）。

        左轴 BIC、右轴 Mean Silhouette；建议 K 以红圈标出。诊断细节写入
        gmm_k_selection.csv，图面仅保留层名与 panel 标号 (a–d)。

        Args:
            per_band: 分层聚类 per_band 字典
            output_dir: 输出目录
            model_name: 模型名

        Returns:
            True 表示已出图；False 表示不适用
        """
        bands = [
            (name, info)
            for name, info in per_band.items()
            if isinstance(info.get('bic_analysis'), dict)
            and 'bics' in info['bic_analysis']
            and info['bic_analysis'].get('selection_rule') != 'stability'
        ]
        if not bands:
            return False

        self.logger.info("    📊 绘制 GMM 选 K 判据（BIC + silhouette）...")
        F = self.fonts = apply_clustering_plot_style(self.config)
        bands = _sort_bands_for_k_plot(bands)

        n = len(bands)
        n_cols = 2 if n > 1 else 1
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5.6 * n_cols, 4.4 * n_rows),
            squeeze=False,
        )

        c_bic, c_sil = '#2166AC', '#B2182B'
        panel_labels = [f'({chr(ord("a") + i)})' for i in range(n)]

        for idx, (ax, (name, info)) in enumerate(zip(axes.ravel(), bands)):
            ba = info['bic_analysis']
            k = np.asarray(ba['n_clusters_range'], dtype=float)
            bic = np.asarray(ba['bics'], dtype=float)
            finite = np.isfinite(bic)
            if not np.any(finite):
                ax.axis('off')
                continue

            sil = np.asarray(
                ba.get('silhouette') or np.full_like(bic, np.nan), dtype=float
            )
            sd = np.asarray(
                ba.get('silhouette_sd') or np.zeros_like(sil), dtype=float
            )
            k_sel = int(ba['optimal_n'])

            ax.plot(
                k[finite], bic[finite], 'o-', color=c_bic,
                lw=2.0, ms=6, markerfacecolor='white',
                markeredgewidth=1.4, zorder=3,
            )
            ax.set_xlabel('Number of clusters ($K$)', fontsize=F['label'] - 1)
            ax.set_ylabel('BIC', color=c_bic, fontsize=F['label'] - 1)
            ax.tick_params(axis='y', labelcolor=c_bic, labelsize=F['tick'] - 1)
            ax.tick_params(axis='x', labelsize=F['tick'] - 1)
            ax.grid(True, alpha=0.22, ls='-', lw=0.6)
            ax.set_axisbelow(True)

            ax2 = None
            if np.any(np.isfinite(sil)):
                ax2 = ax.twinx()
                ax2.errorbar(
                    k, sil, yerr=sd, fmt='o-', color=c_sil, lw=2.0, ms=6,
                    capsize=2.5, elinewidth=1.0, markerfacecolor='white',
                    markeredgewidth=1.4, zorder=3,
                )
                ax2.set_ylabel(
                    'Mean silhouette', color=c_sil, fontsize=F['label'] - 1,
                )
                ax2.tick_params(
                    axis='y', labelcolor=c_sil, labelsize=F['tick'] - 1,
                )

            j = int(np.argmin(np.abs(k - k_sel)))
            ax.plot(
                k[j], bic[j], 'o', ms=13, mfc='none', mec=c_sil,
                mew=2.2, zorder=5,
            )
            if ax2 is not None and np.isfinite(sil[j]):
                ax2.plot(
                    k[j], sil[j], 'o', ms=13, mfc='none', mec=c_sil,
                    mew=2.2, zorder=5,
                )
                ax2.annotate(
                    f'$K={k_sel}$', (k[j], sil[j]),
                    textcoords='offset points', xytext=(6, 6),
                    color=c_sil, fontsize=F['annotation'] - 1,
                    fontweight='bold',
                )

            band_title = MOHO_BAND_TITLES.get(name, name.replace('_', ' ').title())
            ax.set_title(
                band_title,
                fontsize=F['title'] - 2, fontweight='bold', pad=8,
            )
            ax.text(
                -0.11, 1.06, panel_labels[idx],
                transform=ax.transAxes,
                fontsize=F['title'] - 1, fontweight='bold',
                va='top', ha='left',
            )

        for ax in axes.ravel()[len(bands):]:
            ax.axis('off')

        short_name = _short_model_label(model_name)
        fig.suptitle(
            f'{short_name} — GMM optimal $K$ selection',
            fontsize=F['suptitle'] - 6, fontweight='bold', y=0.98,
        )

        # 统一图例（BIC / Silhouette）
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color=c_bic, marker='o', lw=2, ms=6,
                   markerfacecolor='white', label='BIC'),
            Line2D([0], [0], color=c_sil, marker='o', lw=2, ms=6,
                   markerfacecolor='white', label='Mean silhouette'),
        ]
        fig.legend(
            handles=legend_handles, loc='lower center', ncol=2,
            frameon=False, fontsize=F['legend'] - 2,
            bbox_to_anchor=(0.5, -0.02),
        )

        fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))

        for fmt in self.config.visualization.get('save_formats', ['jpg']):
            fig.savefig(
                output_dir / f'2-3-1_k_selection.{fmt}',
                dpi=self.config.visualization['dpi'],
                bbox_inches='tight',
            )
        plt.close(fig)
        self.logger.info("      ✅ 保存: 2-3-1_k_selection")
        return True

    def plot_k_robustness(
        self,
        k_robustness: Dict[str, Any],
        output_dir: Path,
        model_name: str,
    ) -> bool:
        """
        K 扫描稳健性图：证明结论不依赖 K 的具体取值。

        左格为相邻 K 的 ARI——K 不同则粒度不同，该值必然小于 1，只用于定位
        分区结构趋于稳定的区间，不应解读为"K 选得对不对"。右格为快/慢三分类
        相对 K* 的一致度，跨模型投票只依赖这个三分类，故它才是结论稳健性的
        直接证据；灰色参考线标在 0.9。

        Args:
            k_robustness: perform_depth_stratified_clustering 返回的 k_robustness
            output_dir: 输出目录
            model_name: 模型名

        Returns:
            True 表示已出图，False 表示无可用数据
        """
        bands = [(n, d) for n, d in (k_robustness or {}).items() if d.get('k_values')]
        if not bands:
            return False

        apply_clustering_plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        palette = plt.get_cmap('tab10')

        for i, (name, info) in enumerate(bands):
            color = palette(i % 10)
            label = name.replace('_', ' ').title()
            k_sel = int(info['k_selected'])

            # 左：相邻 K 的 ARI，横坐标取两个 K 的中点
            pairs = info.get('ari_consecutive', {})
            if pairs:
                xs, ys = [], []
                for key, val in pairs.items():
                    a, b = key.split('->')
                    xs.append((int(a) + int(b)) / 2.0)
                    ys.append(val)
                idx = np.argsort(xs)
                axes[0].plot(
                    np.asarray(xs)[idx], np.asarray(ys)[idx],
                    'o-', color=color, label=label, lw=1.8, ms=4.5,
                )

            # 右：快/慢三分类相对 K* 的一致度
            tri = info.get('trichotomy_agreement', {})
            if tri:
                ks = sorted(int(k) for k in tri)
                vs = [tri[str(k)] for k in ks]
                axes[1].plot(ks, vs, 'o-', color=color, label=label, lw=1.8, ms=4.5)
                axes[1].plot(
                    [k_sel], [tri[str(k_sel)]], marker='*', ms=15,
                    color=color, mec='k', mew=0.7, ls='none', zorder=5,
                )

        axes[0].set_xlabel('K (midpoint of adjacent pair)')
        axes[0].set_ylabel('ARI between adjacent K')
        axes[0].set_title('Partition change with K', fontsize=11.5, fontweight='bold')
        axes[0].set_ylim(0, 1.02)

        axes[1].axhline(0.9, color='0.5', ls='--', lw=1.0, zorder=1)
        axes[1].set_xlabel('Number of clusters K')
        axes[1].set_ylabel('Fast / slow agreement with K*')
        axes[1].set_title(
            'Robustness of the fast–slow classification',
            fontsize=11.5, fontweight='bold',
        )
        axes[1].set_ylim(0, 1.02)

        for ax in axes:
            ax.grid(alpha=0.3, ls=':')

        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc='lower center',
            ncol=min(len(labels), 4), frameon=False, fontsize=10,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.suptitle(
            f'{model_name} — Sensitivity of the results to K '
            '(star = selected K)',
            fontsize=13, fontweight='bold', y=0.99,
        )
        fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))

        for fmt in self.config.visualization.get('save_formats', ['jpg']):
            fig.savefig(
                output_dir / f'2-3-3_k_robustness.{fmt}',
                dpi=self.config.visualization['dpi'],
            )
        plt.close(fig)
        self.logger.info("      ✅ 保存: 2-3-3_k_robustness")
        return True

    def plot_bic_analysis(
        self,
        bic_analysis: Dict[str, Any],
        output_dir: Path,
        model_name: str
    ) -> None:
        """
        BIC 选 K 决策图。

        bic_eff 模式：双轴对比未修正 BIC（灰，单调下降至 K 上界）与有效
        样本量修正 BIC_eff（蓝，中等 K 处出现真实极小值），红星标注选定 K，
        并注明去相关长度与 N_eff——直观说明 K 由数据的独立信息量决定，
        而非由体元数或人为阈值决定。
        其余模式：单轴 BIC(K) + 拐点增益内嵌图。
        """
        self.logger.info("    📊 绘制BIC决策图...")

        if 'bics' not in bic_analysis:
            # fixed_k 模式未做 BIC 扫描，无曲线可画
            self.logger.info(
                f"      ⏭️ [{bic_analysis.get('band_name', '?')}] "
                f"K 为先验固定（{bic_analysis.get('optimal_n')}），跳过 BIC 图"
            )
            return

        n_range = np.asarray(bic_analysis['n_clusters_range'], dtype=int)
        bics = np.asarray(bic_analysis['bics'], dtype=float)
        optimal_n = int(bic_analysis['optimal_n'])
        optimal_bic = float(bic_analysis['optimal_bic'])
        argmin_n = int(bic_analysis.get('argmin_n', optimal_n))
        rule = bic_analysis.get('selection_rule', 'bic_min')

        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.plot(
            n_range, bics, 'o-', color='steelblue',
            linewidth=2, markersize=8, label='BIC',
        )
        sel_label = (
            f'Selected K = {optimal_n} (BIC knee)'
            if rule == 'bic_knee' else f'Selected K = {optimal_n} (min BIC)'
        )
        ax.axvline(
            optimal_n, color='red', linestyle='--', linewidth=1.5,
            label=sel_label,
        )
        ax.scatter(
            [optimal_n], [optimal_bic], s=220, c='red', marker='*',
            zorder=10, edgecolors='black', linewidth=1.5,
        )
        if argmin_n != optimal_n:
            ax.scatter(
                [argmin_n], [float(bic_analysis.get('argmin_bic', np.nan))],
                s=140, c='gray', marker='*', zorder=9,
                edgecolors='black', linewidth=1.0,
                label=f'min BIC at K = {argmin_n}',
            )
        ax.set_xlabel('Number of Clusters K')
        ax.set_ylabel('BIC (lower is better)')
        ax.set_xticks(n_range)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f'{model_name} — BIC Model Selection',
            fontweight='bold',
        )

        # 内嵌图：各 K 到首末弦的归一化距离（拐点判据本身的可视化）
        dist = np.asarray(
            bic_analysis.get('knee_distances', [np.nan] * len(bics)), dtype=float
        )
        finite = np.isfinite(dist)
        if np.sum(finite) > 0:
            ax_in = ax.inset_axes([0.42, 0.45, 0.53, 0.4])
            ax_in.plot(
                n_range[finite], dist[finite], 'o-',
                color='gray', linewidth=1.2, markersize=4,
            )
            ax_in.axvline(optimal_n, color='red', linestyle='--', linewidth=1)
            ax_in.set_ylabel('Distance to chord', fontsize=11)
            ax_in.tick_params(labelsize=10)
            ax_in.grid(True, alpha=0.3)

        band = bic_analysis.get('band_name', 'full')
        for fmt in self.config.visualization.get('save_formats', ['jpg']):
            filepath = output_dir / f'2-3-1_bic_analysis_{band}.{fmt}'
            fig.savefig(
                filepath, dpi=self.config.visualization['dpi'], bbox_inches='tight'
            )
        plt.close(fig)
        self.logger.info(f"      ✅ 保存: 2-3-1_bic_analysis_{band}")
    
    def plot_probability_distribution(
        self, 
        probabilities: np.ndarray,
        labels: np.ndarray,
        output_dir: Path,
        model_name: str
    ) -> None:
        """绘制概率分布和置信度分析"""
        self.logger.info("    📊 绘制概率分布分析...")
        
        n_clusters = probabilities.shape[1]
        max_probs = np.max(probabilities, axis=1)
        
        # 计算熵
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        
        # 创建图形
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 最大概率分布直方图
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(max_probs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(max_probs), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean={np.mean(max_probs):.3f}')
        ax1.axvline(np.median(max_probs), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median={np.median(max_probs):.3f}')
        threshold = self.config.auto_gmm['probability_threshold']
        ax1.axvline(threshold, color='green', linestyle='--', 
                   linewidth=2, label=f'Threshold={threshold}')
        ax1.set_xlabel('Maximum Probability', fontsize=16)
        ax1.set_ylabel('Frequency', fontsize=16)
        ax1.set_title('Distribution of Maximum Probabilities', fontsize=17, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 置信度分类
        ax2 = fig.add_subplot(gs[0, 1])
        high_conf = np.sum(max_probs > threshold)
        medium_conf = np.sum((max_probs > 0.5) & (max_probs <= threshold))
        low_conf = np.sum(max_probs <= 0.5)
        
        categories = ['High\n(>{})'.format(threshold), 
                     'Medium\n(0.5-{})'.format(threshold), 
                     'Low\n(<0.5)']
        counts = [high_conf, medium_conf, low_conf]
        colors_conf = ['#2ecc71', '#f39c12', '#e74c3c']
        
        bars = ax2.bar(categories, counts, color=colors_conf, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Number of Samples', fontsize=16)
        ax2.set_title('Confidence Level Distribution', fontsize=17, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加百分比标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}\n({count/len(max_probs):.1%})',
                    ha='center', va='bottom', fontsize=15, fontweight='bold')
        
        # 3. 熵分布
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(entropy, bins=50, color='coral', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(entropy), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean={np.mean(entropy):.3f}')
        ax3.set_xlabel('Entropy (Uncertainty)', fontsize=16)
        ax3.set_ylabel('Frequency', fontsize=16)
        ax3.set_title('Clustering Uncertainty Distribution', fontsize=17, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 每个簇的平均概率
        ax4 = fig.add_subplot(gs[1, :])
        cluster_avg_probs = []
        cluster_ids = []
        
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if np.any(mask):
                cluster_probs = probabilities[mask, cluster_id]
                cluster_avg_probs.append(cluster_probs)
                cluster_ids.append(cluster_id)
        
        # 箱线图
        colors = generate_distinct_colors(len(cluster_ids))
        bp = ax4.boxplot(cluster_avg_probs, positions=cluster_ids, widths=0.6,
                        patch_artist=True, showmeans=True, meanline=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_xlabel('Cluster ID', fontsize=16)
        ax4.set_ylabel('Assignment Probability', fontsize=16)
        ax4.set_title('Probability Distribution by Cluster', fontsize=17, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim([0, 1.05])
        
        # 5. 概率矩阵热图
        ax5 = fig.add_subplot(gs[2, :2])
        
        # 采样（如果样本太多）
        n_samples_to_show = min(1000, len(probabilities))
        sample_indices = np.random.choice(len(probabilities), n_samples_to_show, replace=False)
        sample_indices = np.sort(sample_indices)
        
        # 按照最大概率排序
        sorted_indices = sample_indices[np.argsort(np.argmax(probabilities[sample_indices], axis=1))]
        
        im = ax5.imshow(probabilities[sorted_indices].T, aspect='auto', 
                       cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
        ax5.set_xlabel('Sample Index (sorted by cluster)', fontsize=16)
        ax5.set_ylabel('Cluster ID', fontsize=16)
        ax5.set_title(f'Probability Matrix (showing {n_samples_to_show} samples)', 
                     fontsize=17, fontweight='bold')
        ax5.set_yticks(range(n_clusters))
        plt.colorbar(im, ax=ax5, label='Probability')
        
        # 6. 统计表格
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        stats_data = [
            ['Total Samples', f'{len(probabilities):,}'],
            ['Number of Clusters', f'{n_clusters}'],
            ['', ''],
            ['Mean Max Prob', f'{np.mean(max_probs):.4f}'],
            ['Median Max Prob', f'{np.median(max_probs):.4f}'],
            ['Std Max Prob', f'{np.std(max_probs):.4f}'],
            ['', ''],
            ['High Confidence', f'{high_conf:,} ({high_conf/len(max_probs):.1%})'],
            ['Medium Confidence', f'{medium_conf:,} ({medium_conf/len(max_probs):.1%})'],
            ['Low Confidence', f'{low_conf:,} ({low_conf/len(max_probs):.1%})'],
            ['', ''],
            ['Mean Entropy', f'{np.mean(entropy):.4f}'],
            ['Median Entropy', f'{np.median(entropy):.4f}']
        ]
        
        table = ax6.table(cellText=stats_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in [0, 1]:
            table[(i, 0)].set_facecolor('#3498db')
            table[(i, 1)].set_facecolor('#3498db')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        
        plt.suptitle(f'{model_name} - GMM Probability & Confidence Analysis', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        filepath = output_dir / '2-3-2_probability_analysis.png'
        plt.savefig(filepath, dpi=self.config.visualization['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"      ✅ 保存: 2-3-2_probability_analysis.png")
    
    def plot_gmm_vs_kmeans_comparison(
        self,
        gmm_results: Dict[str, Any],
        kmeans_results: Dict[str, Any],
        output_dir: Path,
        model_name: str
    ) -> None:
        """绘制GMM vs K-means对比分析"""
        self.logger.info("    📊 绘制GMM vs K-means对比...")
        
        # 创建图形
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 轮廓系数对比
        ax1 = fig.add_subplot(gs[0, 0])
        
        kmeans_n_list = kmeans_results['n_clusters_tested']
        kmeans_sil_scores = [kmeans_results['metrics'][k]['silhouette_score'] 
                            for k in kmeans_n_list]
        
        ax1.plot(kmeans_n_list, kmeans_sil_scores, 'o-', 
                color='steelblue', linewidth=2, markersize=8, label='K-means')
        
        # GMM最优点
        gmm_n = gmm_results['n_clusters']
        gmm_sil = gmm_results['metrics']['silhouette_score']
        ax1.scatter([gmm_n], [gmm_sil], s=300, c='red', marker='*', 
                   zorder=10, edgecolors='black', linewidth=2, label='GMM (Optimal)')
        
        ax1.set_xlabel('Number of Clusters', fontsize=16)
        ax1.set_ylabel('Silhouette Score', fontsize=16)
        ax1.set_title('Silhouette Score Comparison', fontsize=17, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # 2. Davies-Bouldin对比
        ax2 = fig.add_subplot(gs[0, 1])
        
        kmeans_db_scores = [kmeans_results['metrics'][k]['davies_bouldin_score'] 
                           for k in kmeans_n_list]
        
        ax2.plot(kmeans_n_list, kmeans_db_scores, 'o-', 
                color='coral', linewidth=2, markersize=8, label='K-means')
        
        gmm_db = gmm_results['metrics']['davies_bouldin_score']
        ax2.scatter([gmm_n], [gmm_db], s=300, c='red', marker='*', 
                   zorder=10, edgecolors='black', linewidth=2, label='GMM (Optimal)')
        
        ax2.set_xlabel('Number of Clusters', fontsize=16)
        ax2.set_ylabel('Davies-Bouldin Score', fontsize=16)
        ax2.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=17, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Calinski-Harabasz对比
        ax3 = fig.add_subplot(gs[0, 2])
        
        kmeans_ch_scores = [kmeans_results['metrics'][k]['calinski_harabasz_score'] 
                           for k in kmeans_n_list]
        
        ax3.plot(kmeans_n_list, kmeans_ch_scores, 'o-', 
                color='forestgreen', linewidth=2, markersize=8, label='K-means')
        
        gmm_ch = gmm_results['metrics']['calinski_harabasz_score']
        ax3.scatter([gmm_n], [gmm_ch], s=300, c='red', marker='*', 
                   zorder=10, edgecolors='black', linewidth=2, label='GMM (Optimal)')
        
        ax3.set_xlabel('Number of Clusters', fontsize=16)
        ax3.set_ylabel('Calinski-Harabasz Score', fontsize=16)
        ax3.set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=17, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 计算时间对比
        ax4 = fig.add_subplot(gs[1, 0])
        
        kmeans_times = [kmeans_results['metrics'][k]['fit_time'] for k in kmeans_n_list]
        
        ax4.plot(kmeans_n_list, kmeans_times, 'o-', 
                color='purple', linewidth=2, markersize=8, label='K-means')
        
        gmm_time = gmm_results['metrics']['fit_time']
        ax4.scatter([gmm_n], [gmm_time], s=300, c='red', marker='*', 
                   zorder=10, edgecolors='black', linewidth=2, label='GMM')
        
        ax4.set_xlabel('Number of Clusters', fontsize=16)
        ax4.set_ylabel('Fit Time (seconds)', fontsize=16)
        ax4.set_title('Computational Time Comparison', fontsize=17, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 聚类平衡性对比
        ax5 = fig.add_subplot(gs[1, 1])
        
        kmeans_balance = [kmeans_results['metrics'][k]['cluster_balance'] 
                         for k in kmeans_n_list]
        
        ax5.plot(kmeans_n_list, kmeans_balance, 'o-', 
                color='brown', linewidth=2, markersize=8, label='K-means')
        
        gmm_balance = gmm_results['metrics']['cluster_balance']
        ax5.scatter([gmm_n], [gmm_balance], s=300, c='red', marker='*', 
                   zorder=10, edgecolors='black', linewidth=2, label='GMM')
        
        ax5.set_xlabel('Number of Clusters', fontsize=16)
        ax5.set_ylabel('Cluster Balance (std/mean)', fontsize=16)
        ax5.set_title('Cluster Balance Comparison (Lower is Better)', 
                     fontsize=17, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 标签一致性分析（如果有）
        ax6 = fig.add_subplot(gs[1, 2])
        
        if 'comparison_with_gmm' in kmeans_results and kmeans_results['comparison_with_gmm']:
            comp = kmeans_results['comparison_with_gmm']
            
            metrics_names = ['ARI', 'NMI', 'Agreement\nRatio']
            metrics_values = [
                comp['adjusted_rand_index'],
                comp['normalized_mutual_info'],
                comp['agreement_ratio']
            ]
            
            bars = ax6.bar(metrics_names, metrics_values, 
                          color=['#3498db', '#e74c3c', '#2ecc71'], 
                          alpha=0.7, edgecolor='black', linewidth=2)
            
            ax6.set_ylabel('Score', fontsize=16)
            ax6.set_title(f'Label Agreement (n={gmm_n})', fontsize=17, fontweight='bold')
            ax6.set_ylim([0, 1])
            ax6.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, metrics_values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', 
                        fontsize=15, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                    fontsize=24, transform=ax6.transAxes)
            ax6.set_title('Label Agreement', fontsize=17, fontweight='bold')
        
        # 7. 对比表格
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # 找出K-means最优结果
        best_k = kmeans_results['best_k']
        kmeans_best = kmeans_results['metrics'][best_k]
        
        table_data = [
            ['Metric', 'GMM (BIC-Optimal)', f'K-means (Best k={best_k})', 'Winner'],
            ['Number of Clusters', f'{gmm_n}', f'{best_k}', '-'],
            ['Silhouette Score ↑', f"{gmm_sil:.4f}", f"{kmeans_best['silhouette_score']:.4f}", 
             'GMM' if gmm_sil > kmeans_best['silhouette_score'] else 'K-means'],
            ['Davies-Bouldin ↓', f"{gmm_db:.4f}", f"{kmeans_best['davies_bouldin_score']:.4f}",
             'GMM' if gmm_db < kmeans_best['davies_bouldin_score'] else 'K-means'],
            ['Calinski-Harabasz ↑', f"{gmm_ch:.2f}", f"{kmeans_best['calinski_harabasz_score']:.2f}",
             'GMM' if gmm_ch > kmeans_best['calinski_harabasz_score'] else 'K-means'],
            ['Fit Time (s)', f"{gmm_time:.2f}", f"{kmeans_best['fit_time']:.2f}",
             'GMM' if gmm_time < kmeans_best['fit_time'] else 'K-means'],
            ['Cluster Balance ↓', f"{gmm_balance:.4f}", f"{kmeans_best['cluster_balance']:.4f}",
             'GMM' if gmm_balance < kmeans_best['cluster_balance'] else 'K-means'],
            ['Probability Support', 'Yes ✓', 'No ✗', 'GMM'],
            ['Soft Clustering', 'Yes ✓', 'No ✗', 'GMM']
        ]
        
        table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # 设置表头样式
        for j in range(4):
            table[(0, j)].set_facecolor('#3498db')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        # 高亮Winner列
        for i in range(1, len(table_data)):
            winner = table_data[i][3]
            if winner == 'GMM':
                table[(i, 3)].set_facecolor('#2ecc71')
                table[(i, 3)].set_text_props(weight='bold')
            elif winner == 'K-means':
                table[(i, 3)].set_facecolor('#f39c12')
                table[(i, 3)].set_text_props(weight='bold')
        
        plt.suptitle(f'{model_name} - GMM vs K-means Comprehensive Comparison', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        filepath = output_dir / '2-3-3_gmm_vs_kmeans_comparison.png'
        plt.savefig(filepath, dpi=self.config.visualization['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"      ✅ 保存: 2-3-3_gmm_vs_kmeans_comparison.png")
        
class BasicClusteringVisualizer:
    """基础聚类可视化器（深度切片、垂直剖面等）- 修正版（含海岸线）"""
    
    def __init__(self, config: Any, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.base_config = BaseConfig()
        self.fonts = apply_clustering_plot_style(config)
        self.processor = None
    
    def set_processor(self, processor):
        """设置数据处理器引用"""
        self.processor = processor

    def _is_perturbation_space(self, metadata: Dict[str, Any]) -> bool:
        """判断当前聚类/绘图是否处于扰动域（δln）而非原始 Vp/Vs。"""
        fs = str(metadata.get('feature_space', '') or '')
        if fs.startswith('perturbation'):
            return True
        if fs == 'absolute':
            return False
        names = list(
            metadata.get('feature_display_names')
            or metadata.get('features', [])
        )
        if any(str(d).lower().startswith(('dln', 'drel')) for d in names[:2]):
            return True
        return bool(
            self.config.preprocessing.get('perturbation', {}).get(
                'enabled', True
            )
        )

    def _physical_feature_matrix(
        self,
        X: np.ndarray,
        spatial_indices: Optional[np.ndarray],
        n_features: int = 2,
    ) -> np.ndarray:
        """
        取绘图用物理量特征矩阵：优先 last_feature_cube，否则反标准化。

        交会图/剖面应展示 km/s 或 δln 原值，而非 StandardScaler 后的 z 分数。
        """
        if spatial_indices is not None and self.processor is not None:
            cube = getattr(self.processor, 'last_feature_cube', None)
            if cube is not None and cube.ndim == 4:
                si = np.asarray(spatial_indices, dtype=int)
                if len(si) == len(X):
                    nf = min(int(cube.shape[-1]), int(n_features))
                    return np.column_stack([
                        cube[si[:, 0], si[:, 1], si[:, 2], j]
                        for j in range(nf)
                    ]).astype(float)

        X_plot = np.asarray(X, dtype=float)
        n_use = min(X_plot.shape[1], int(n_features))
        if (
            self.processor is not None
            and getattr(self.processor, 'scaler', None) is not None
        ):
            try:
                return self.processor.scaler.inverse_transform(
                    X_plot[:, :n_use]
                )
            except Exception:
                pass
        return X_plot[:, :n_use]

    def _ensure_geology_basemap_png(
        self,
        map_extent: Sequence[float],
        cache_dir: Path,
        dpi: int = 200,
    ) -> Path:
        """
        按模型制图范围缓存 USGS 拼合基岩地质底图（仿 5_9，无图例）。

        Args:
            map_extent: [lon_min, lon_max, lat_min, lat_max]
            cache_dir: 缓存目录
            dpi: PNG 分辨率

        Returns:
            PNG 路径
        """
        lon0, lon1, lat0, lat1 = [float(x) for x in map_extent]
        tag = (
            f'{lon0:.1f}_{lon1:.1f}_{lat0:.1f}_{lat1:.1f}'.replace('.', 'p')
        )
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # 文件名带 bathy 标记，避免沿用旧的纯色海水缓存
        # _bathy_otop_：地质后仅洋区重绘水深，避免年代面挡住陆架/洋区
        out_png = cache_dir / f'geology_basemap_{tag}_bathy_otop_dpi{dpi}.png'
        if out_png.exists() and out_png.stat().st_size > 1000:
            return out_png

        import importlib.util

        geology_py = project_root / '5_Visualization' / '5_9_Geology.py'
        if not geology_py.exists():
            raise FileNotFoundError(f'找不到地质图模块: {geology_py}')
        mod_name = 'eastasia_geology_5_9'
        # 始终重载，确保 ocean_bathymetry 等改动生效
        spec = importlib.util.spec_from_file_location(mod_name, geology_py)
        if spec is None or spec.loader is None:
            raise ImportError(f'无法加载地质图模块: {geology_py}')
        geology_mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = geology_mod
        spec.loader.exec_module(geology_mod)
        EastAsiaGeologyMap = geology_mod.EastAsiaGeologyMap

        plotter = EastAsiaGeologyMap(output_dir=str(cache_dir))
        plotter.config.layers['cn_blocks'] = True
        plotter.config.layers['province_overlay'] = True
        plotter.config.layers['ocean_bathymetry'] = True
        plotter.config.layers['legend'] = False
        plotter.config.layers['title'] = False
        plotter.render_composite_png_for_embed(
            region=[lon0, lon1, lat0, lat1],
            out_path=out_png,
            dpi=dpi,
            force_convert=False,
        )
        return out_png

    def plot_depth_slices(
        self,
        labels_3d: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Path,
        algorithm_name: str,
        model_name: str,
        concordance_eval: Optional['GeologyConcordanceEvaluator'] = None,
    ) -> None:
        """
        3×3 深度切片（子图等大）：
        - 第1排：首格 USGS 基岩地质底图（无图例）；其后浅部 facies+地质线（默认 10/40 km）
        - 第2–3排：叠 Slab2 等深线（默认 100/150/200/500/600/700 km）
        子图标题只保留深度与叠图类型（Slab2/地质线均为定性对照）。
        """
        self.logger.info(f"    📊 绘制 3×3 深度切片 ({algorithm_name})...")
        self.fonts = apply_clustering_plot_style(self.config)

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
            use_cartopy = True
        except ImportError:
            self.logger.warning("    ⚠️ cartopy未安装，使用基础版本（无海岸线）")
            use_cartopy = False
            ccrs = cfeature = LONGITUDE_FORMATTER = LATITUDE_FORMATTER = None

        layout = self.config.visualization.get('slice_layout') or {}
        n_rows = int(layout.get('n_rows', 3))
        n_cols = int(layout.get('n_cols', 3))
        lead_geo_map = bool(layout.get('row1_leading_geology_map', True))
        geo_depths = list(
            layout.get('geology_depths_km')
            or (self.config.geology_concordance.get('surface_units') or {}).get(
                'overlay_depths_km', [10.0, 40.0]
            )
        )
        slab_depths = list(
            layout.get('slab_depths_km', [100.0, 150.0, 200.0, 500.0, 600.0, 700.0])
        )
        # 第1排：可选首格地质底图 + 浅部 facies 槽位
        n_facies_geo = n_cols - 1 if lead_geo_map else n_cols
        geo_depths = (geo_depths + [None] * n_facies_geo)[:n_facies_geo]
        n_slab_slots = n_rows * n_cols - n_cols
        slab_depths = (slab_depths + [None] * n_slab_slots)[:n_slab_slots]

        lats = np.asarray(metadata['lats'], dtype=float)
        lons = np.asarray(metadata['lons'], dtype=float)
        depths = np.asarray(metadata['depths'], dtype=float)
        map_extent = [
            float(lons.min()),
            float(lons.max()),
            float(lats.min()),
            float(lats.max()),
        ]
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        n_clusters = int(len(np.unique(labels_3d[labels_3d >= 0])))
        colors = generate_distinct_colors(
            max(n_clusters, self.config.visualization['n_colors'])
        )
        cmap = mcolors.ListedColormap(colors[: max(n_clusters, 1)])

        # Slab2 / 地质线
        slab_grid = None
        geo_region = None
        if concordance_eval is not None:
            try:
                concordance_eval._ensure_slab_grid(lons, lats)
                slab_grid = concordance_eval._slab_grid
            except Exception as e:
                self.logger.warning(f"      ⚠️ Slab2 加载失败: {e}")
            try:
                if not getattr(concordance_eval, '_surface_overlay_lines', None):
                    concordance_eval._surface_overlay_lines = (
                        concordance_eval._load_overlay_lines()
                    )
                geo_region = concordance_eval.get_overlap_region()
            except Exception as e:
                self.logger.warning(f"      ⚠️ 地质线加载失败: {e}")

        slab_levels = [50, 100, 150, 200, 300, 400, 500, 600]
        # 右侧留出色标空间（仅最右列挂簇色标）
        fig, axes = make_equal_map_subplots(
            n_rows,
            n_cols,
            ccrs if use_cartopy else None,
            panel_w=6.4,
            panel_h=5.2,
            wspace=0.12,
            hspace=0.18,
            top=0.90,
            bottom=0.04,
            left=0.035,
            right=0.90,
        )
        projection = ccrs.PlateCarree() if use_cartopy else None
        plot_tf = {'transform': projection} if use_cartopy else {}

        def _style_ax(ax) -> None:
            if use_cartopy:
                ax.coastlines(
                    resolution='50m', color='gray', linewidth=0.45, zorder=6
                )
                ax.add_feature(
                    cfeature.BORDERS,
                    linestyle='--',
                    linewidth=0.4,
                    edgecolor='gray',
                    zorder=5,
                )
                ax.set_extent(map_extent, crs=projection)
                try:
                    ax.set_aspect('auto')
                except Exception:
                    pass
                gl = ax.gridlines(
                    crs=projection,
                    draw_labels=True,
                    linewidth=0.5,
                    color='gray',
                    alpha=0.5,
                    linestyle='--',
                    zorder=4,
                )
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 13}
                gl.ylabel_style = {'size': 13}
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
            else:
                ax.set_xlim(map_extent[0], map_extent[1])
                ax.set_ylim(map_extent[2], map_extent[3])
                ax.set_aspect('auto')
                F = self.fonts
                ax.set_xlabel('Longitude', fontsize=F['label'])
                ax.set_ylabel('Latitude', fontsize=F['label'])
                ax.tick_params(labelsize=F['tick'])
                ax.grid(True, alpha=0.3, linestyle='--')

        def _draw_facies(ax, z_target: float):
            z_idx = int(np.argmin(np.abs(depths - float(z_target))))
            actual_z = float(depths[z_idx])
            label_slice = np.ma.masked_where(
                labels_3d[:, :, z_idx] < 0, labels_3d[:, :, z_idx]
            )
            im = ax.imshow(
                label_slice,
                aspect='auto',
                origin='lower',
                cmap=cmap,
                interpolation='nearest',
                extent=map_extent,
                vmin=-0.5,
                vmax=max(n_clusters - 0.5, 0.5),
                zorder=1,
                **plot_tf,
            )
            return actual_z, im

        def _add_cluster_cbar(ax, im) -> None:
            """最右列子图挂簇色标。"""
            if n_clusters <= 0:
                return
            # 簇多时抽稀刻度，避免文字挤叠
            if n_clusters <= 12:
                ticks = list(range(n_clusters))
            elif n_clusters <= 24:
                ticks = list(range(0, n_clusters, 2))
                if ticks[-1] != n_clusters - 1:
                    ticks.append(n_clusters - 1)
            else:
                ticks = list(range(0, n_clusters, 3))
                if ticks[-1] != n_clusters - 1:
                    ticks.append(n_clusters - 1)
            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.02,
                ticks=ticks,
            )
            cbar.ax.tick_params(labelsize=max(10, int(self.fonts['annotation']) - 2))
            cbar.set_label('Cluster ID', fontsize=self.fonts['label'])

        # —— 第1排：首格基岩地质底图 + 浅部 facies 叠地质线 ——
        facies_start_col = 0
        if lead_geo_map:
            ax0 = axes[0, 0]
            try:
                # output_dir 是叶子图件目录（figures/model_clustering/<scheme>/<model>），
                # 其父目录即 scheme 根，各模型共享同一份地质底图缓存避免重复渲染
                out_res = Path(output_dir).resolve()
                cache_dir = out_res.parent / '_geology_basemap_cache'
                geo_png = self._ensure_geology_basemap_png(
                    map_extent, cache_dir=cache_dir, dpi=220
                )
                img = plt.imread(str(geo_png))
                ax0.imshow(
                    img,
                    extent=map_extent,
                    origin='upper',
                    aspect='auto',
                    zorder=1,
                    **plot_tf,
                )
            except Exception as e:
                self.logger.warning(f'      ⚠️ 地质底图渲染失败: {e}')
                ax0.text(
                    0.5,
                    0.5,
                    'Bedrock geology unavailable',
                    transform=ax0.transAxes,
                    ha='center',
                    va='center',
                    fontsize=self.fonts['annotation'],
                )
            _style_ax(ax0)
            ax0.set_title(
                'Bedrock Geology',
                fontsize=self.fonts['title'],
                fontweight='bold',
                pad=6,
            )
            facies_start_col = 1

        for i, z_target in enumerate(geo_depths):
            col = facies_start_col + i
            if col >= n_cols:
                break
            ax = axes[0, col]
            if z_target is None:
                ax.axis('off')
                continue
            actual_z, im = _draw_facies(ax, float(z_target))
            if concordance_eval is not None and geo_region is not None:
                concordance_eval._draw_geology_lines_on_ax(
                    ax, region=geo_region, transform=projection
                )
            _style_ax(ax)
            ax.set_title(
                f'{actual_z:.0f} km · Geology',
                fontsize=self.fonts['title'],
                fontweight='bold',
                pad=6,
            )
            if col == n_cols - 1:
                _add_cluster_cbar(ax, im)

        # —— 第2–3排：Slab2 ——
        for i, z_target in enumerate(slab_depths):
            row = 1 + i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            if z_target is None:
                ax.axis('off')
                continue
            actual_z, im = _draw_facies(ax, float(z_target))
            if slab_grid is not None and np.any(np.isfinite(slab_grid)):
                levels = [
                    lv for lv in slab_levels
                    if np.nanmin(slab_grid) <= lv <= np.nanmax(slab_grid)
                ]
                if levels:
                    cs = ax.contour(
                        lon_grid,
                        lat_grid,
                        slab_grid,
                        levels=levels,
                        colors='k',
                        linewidths=0.75,
                        alpha=0.9,
                        zorder=4,
                        **plot_tf,
                    )
                    ax.clabel(cs, inline=True, fontsize=self.fonts['clabel'], fmt='%.0f')
            _style_ax(ax)
            ax.set_title(
                f'{actual_z:.0f} km · Slab2',
                fontsize=self.fonts['title'],
                fontweight='bold',
                pad=6,
            )
            if col == n_cols - 1:
                _add_cluster_cbar(ax, im)

        fig.suptitle(
            f'{model_name} — {algorithm_name.upper()} Depth Slices '
            f'(row1 bedrock+cluster; row2–3 Slab2)',
            fontsize=self.fonts['suptitle'],
            fontweight='bold',
            y=0.975,
        )
        # 略增行距；右侧已预留色标空间，勿再把 right 拉满
        fig.subplots_adjust(top=0.90, hspace=0.22, wspace=0.14, right=0.90)

        out_stem = f'2-3-4_{algorithm_name}_depth_slices'
        save_fmts = set(self.config.visualization.get('save_formats', ['jpg']))
        save_fmts.add('png')  # 兼容旧深度切片命名
        for fmt in sorted(save_fmts):
            fig.savefig(
                output_dir / f'{out_stem}.{fmt}',
                dpi=self.config.visualization['dpi'],
            )
        plt.close(fig)
        self.logger.info(f"      ✅ 保存: {out_stem} ({', '.join(sorted(save_fmts))})")

    def plot_vertical_sections(
        self,
        labels_3d: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Path,
        algorithm_name: str,
        model_name: str,
        concordance_eval: Optional['GeologyConcordanceEvaluator'] = None,
        pert_cube: Optional[np.ndarray] = None,
    ) -> None:
        """
        绘制垂直剖面（带地理位置标注）。

        行布局：
          0 — 剖面位置地图
          1 — 聚类 facies + Slab2
          2–3 — δlnVp / δlnVs 原始扰动 + Slab2（若提供 pert_cube）
        """
        self.logger.info(f"    📊 绘制垂直剖面 ({algorithm_name})...")
        self.fonts = apply_clustering_plot_style(self.config)

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            use_cartopy = True
        except ImportError:
            self.logger.warning("    ⚠️ cartopy未安装，使用基础版本")
            use_cartopy = False
            ccrs = cfeature = None

        section_lats = self.config.visualization['section_positions']['latitudes']
        lats = np.asarray(metadata['lats'], dtype=float)
        lons = np.asarray(metadata['lons'], dtype=float)
        depths = np.asarray(metadata['depths'], dtype=float)

        n_sections = min(len(section_lats), 3)
        selected_lats = section_lats[:n_sections]

        n_clusters = len(np.unique(labels_3d[labels_3d >= 0]))
        colors = generate_distinct_colors(
            max(n_clusters, self.config.visualization['n_colors'])
        )
        cmap = mcolors.ListedColormap(colors[:n_clusters])

        # Slab2 三维体掩膜（与聚类同网格）
        slab_mask = None
        if concordance_eval is not None:
            try:
                slab_mask = concordance_eval.build_slab_volume_mask(
                    lons, lats, depths
                )
            except Exception as e:
                self.logger.warning(f"      ⚠️ 剖面 Slab2 体掩膜失败: {e}")

        if pert_cube is None and self.processor is not None:
            pert_cube = getattr(self.processor, 'last_feature_cube', None)
        is_pert = self._is_perturbation_space(metadata)
        has_velocity = (
            pert_cube is not None
            and np.ndim(pert_cube) == 4
            and pert_cube.shape[-1] >= 2
            and pert_cube.shape[:3] == labels_3d.shape
        )
        n_rows = 4 if has_velocity else 2
        right_margin = 0.90 if has_velocity else 0.98

        panel_w, panel_h = 6.0, 4.2
        fig = plt.figure(figsize=(panel_w * n_sections, panel_h * n_rows))
        gs = fig.add_gridspec(
            n_rows,
            n_sections,
            wspace=0.28,
            hspace=0.38,
            left=0.06,
            right=right_margin,
            top=0.93,
            bottom=0.05,
            width_ratios=[1] * n_sections,
            height_ratios=[1.0] * n_rows,
        )
        projection = ccrs.PlateCarree() if use_cartopy else None
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        slab_footprint = (
            np.any(slab_mask, axis=2) if slab_mask is not None else None
        )
        sec_extent = [
            float(lons.min()),
            float(lons.max()),
            float(depths.max()),
            float(depths.min()),
        ]

        # 第一行：剖面位置地图
        for idx in range(n_sections):
            if use_cartopy:
                ax_map = fig.add_subplot(gs[0, idx], projection=projection)
            else:
                ax_map = fig.add_subplot(gs[0, idx])

            target_lat = selected_lats[idx]
            lat_idx = int(np.argmin(np.abs(lats - target_lat)))
            actual_lat = float(lats[lat_idx])

            depth_idx_bg = int(np.argmin(np.abs(depths - 200)))
            slice_bg = labels_3d[:, :, depth_idx_bg]
            slice_masked_bg = np.ma.masked_where(slice_bg < 0, slice_bg)
            extent_map = [
                float(lons.min()),
                float(lons.max()),
                float(lats.min()),
                float(lats.max()),
            ]
            imshow_kw: Dict[str, Any] = {}
            if use_cartopy:
                imshow_kw['transform'] = projection
            ax_map.imshow(
                slice_masked_bg,
                aspect='auto',
                origin='lower',
                cmap=cmap,
                interpolation='nearest',
                extent=extent_map,
                alpha=0.5,
                vmin=0,
                vmax=max(n_clusters - 1, 0),
                **imshow_kw,
            )

            if slab_footprint is not None and np.any(slab_footprint):
                ax_map.contour(
                    lon_grid,
                    lat_grid,
                    slab_footprint.astype(float),
                    levels=[0.5],
                    colors='k',
                    linewidths=0.55,
                    zorder=3,
                    **imshow_kw,
                )

            if use_cartopy:
                ax_map.coastlines(
                    resolution='50m', color='gray', linewidth=0.45, zorder=3
                )
                ax_map.add_feature(
                    cfeature.BORDERS,
                    linestyle='--',
                    linewidth=0.4,
                    edgecolor='gray',
                    zorder=3,
                )
                ax_map.plot(
                    [lons.min(), lons.max()],
                    [actual_lat, actual_lat],
                    'r-',
                    linewidth=3,
                    transform=projection,
                    zorder=4,
                    label='Section Line',
                )
                ax_map.scatter(
                    [lons.min(), lons.max()],
                    [actual_lat, actual_lat],
                    c='red',
                    s=100,
                    marker='o',
                    edgecolors='black',
                    linewidth=2,
                    transform=projection,
                    zorder=5,
                )
                ax_map.set_extent(extent_map, crs=projection)
                try:
                    ax_map.set_aspect('auto')
                except Exception:
                    pass
                gl = ax_map.gridlines(
                    draw_labels=True,
                    linewidth=0.5,
                    color='gray',
                    alpha=0.5,
                    linestyle='--',
                )
                gl.top_labels = False
                gl.right_labels = False
            else:
                ax_map.axhline(actual_lat, color='r', linewidth=2)
                ax_map.set_xlim(extent_map[0], extent_map[1])
                ax_map.set_ylim(extent_map[2], extent_map[3])
                ax_map.set_aspect('auto')

            ax_map.set_title(
                f'Section Location: Lat={actual_lat:.1f}°',
                fontsize=self.fonts['title'],
                fontweight='bold',
            )
            if use_cartopy:
                ax_map.legend(loc='upper right', fontsize=self.fonts['legend'])
            ax_map.tick_params(labelsize=self.fonts['tick'])

        # 第二行：聚类 facies + Slab2
        for idx, target_lat in enumerate(selected_lats):
            ax = fig.add_subplot(gs[1, idx])
            lat_idx = int(np.argmin(np.abs(lats - target_lat)))
            actual_lat = float(lats[lat_idx])
            section_data = labels_3d[lat_idx, :, :]
            section_masked = np.ma.masked_where(section_data < 0, section_data)
            ax.imshow(
                section_masked.T,
                aspect='auto',
                origin='upper',
                cmap=cmap,
                interpolation='nearest',
                extent=sec_extent,
                vmin=0,
                vmax=max(n_clusters - 1, 0),
                zorder=1,
            )
            if slab_mask is not None:
                overlay_slab_on_section(
                    ax,
                    slab_mask[lat_idx, :, :],
                    lons,
                    depths,
                    with_legend=(idx == 0),
                    legend_fontsize=self.fonts['legend'],
                )
            ax.set_title(
                f'Cluster  |  Lat={actual_lat:.1f}°',
                fontsize=self.fonts['title'],
                fontweight='bold',
            )
            ax.set_xlabel('Longitude (°)', fontsize=self.fonts['label'])
            ax.set_ylabel('Depth (km)', fontsize=self.fonts['label'])
            ax.tick_params(labelsize=self.fonts['tick'])
            ax.set_ylim(float(depths.max()), float(depths.min()))
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # 第三/四行：Vp/Vs 或 δlnVp/δlnVs + Slab2
        if has_velocity:
            if is_pert:
                feat_titles = [r'$\delta\ln V_p$', r'$\delta\ln V_s$']
                vel_cmap = 'RdBu_r'
                clim_fn = perturbation_clim
            else:
                feat_titles = [r'$V_p$ (km/s)', r'$V_s$ (km/s)']
                vel_cmap = 'turbo'
                clim_fn = velocity_clim
            for r_off, (fi, ftitle) in enumerate(zip((0, 1), feat_titles)):
                row = 2 + r_off
                vmin, vmax = clim_fn(pert_cube, fi)
                last_im = None
                for idx, target_lat in enumerate(selected_lats):
                    ax = fig.add_subplot(gs[row, idx])
                    lat_idx = int(np.argmin(np.abs(lats - target_lat)))
                    actual_lat = float(lats[lat_idx])
                    sec = np.ma.masked_invalid(pert_cube[lat_idx, :, :, fi])
                    last_im = ax.imshow(
                        sec.T,
                        aspect='auto',
                        origin='upper',
                        cmap=vel_cmap,
                        interpolation='nearest',
                        extent=sec_extent,
                        vmin=vmin,
                        vmax=vmax,
                        zorder=1,
                    )
                    if slab_mask is not None:
                        overlay_slab_on_section(
                            ax,
                            slab_mask[lat_idx, :, :],
                            lons,
                            depths,
                            with_legend=False,
                        )
                    ax.set_title(
                        f'{ftitle}  |  Lat={actual_lat:.1f}°',
                        fontsize=self.fonts['title'],
                        fontweight='bold',
                    )
                    ax.set_xlabel('Longitude (°)', fontsize=self.fonts['label'])
                    if idx == 0:
                        ax.set_ylabel('Depth (km)', fontsize=self.fonts['label'])
                    ax.tick_params(labelsize=self.fonts['tick'])
                    ax.set_ylim(float(depths.max()), float(depths.min()))
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    if idx == n_sections - 1 and last_im is not None:
                        cbar = fig.colorbar(
                            last_im, ax=ax, fraction=0.046, pad=0.02
                        )
                        cbar.set_label(ftitle, fontsize=self.fonts['label'])
                        cbar.ax.tick_params(labelsize=self.fonts['tick'])

        vel_tag = (
            'cluster + dlnV + Slab2' if is_pert
            else 'cluster + Vp/Vs + Slab2'
        )
        fig.suptitle(
            f'{model_name} - {algorithm_name.upper()} Vertical Sections'
            + (f'  |  {vel_tag}' if has_velocity else '')
            + ('  (black: Slab2 body)' if slab_mask is not None else ''),
            fontsize=self.fonts['suptitle'],
            fontweight='bold',
        )

        filepath = output_dir / f'2-3-5_{algorithm_name}_vertical_sections.png'
        fig.savefig(filepath, dpi=self.config.visualization['dpi'])
        # 同步 jpg
        try:
            fig.savefig(
                filepath.with_suffix('.jpg'),
                dpi=self.config.visualization['dpi'],
            )
        except Exception:
            pass
        plt.close(fig)

        self.logger.info(f"      ✅ 保存: 2-3-5_{algorithm_name}_vertical_sections.png")

    def plot_cluster_profiles(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        spatial_indices: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Path,
        algorithm_name: str,
        model_name: str,
        per_band: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        绘制各簇深度–特征剖面（δlnVp 与 δlnVs 同轴叠画）。

        有 per_band 时按深度带分行组图，带内统一深度轴与扰动轴量程，
        便于比较 Vp/Vs 耦合与簇间幅度。
        """
        self.logger.info(f"    📊 绘制聚类速度剖面 ({algorithm_name})...")
        self.fonts = apply_clustering_plot_style(self.config)

        labels = np.asarray(labels)
        X = np.asarray(X)
        spatial_indices = np.asarray(spatial_indices)
        depths_all = np.asarray(metadata['depths'], dtype=float)

        # 反标准化到扰动/原始特征空间
        X_plot = X.copy()
        if (
            self.processor is not None
            and hasattr(self.processor, 'scaler')
            and self.processor.scaler is not None
        ):
            try:
                X_plot = self.processor.scaler.inverse_transform(X)
            except Exception:
                X_plot = X

        feature_names = list(
            metadata.get('feature_display_names')
            or metadata.get('features', [f'F{i}' for i in range(X_plot.shape[1])])
        )
        n_features = min(2, X_plot.shape[1])  # 默认叠画前两特征 (vp/vs)
        feat_space = str(metadata.get('feature_space', '') or '')
        is_pert = 'perturbation' in feat_space or any(
            'dln' in str(n).lower() or 'δ' in str(n) for n in feature_names
        )
        xlabel = r'$\delta\ln V$' if is_pert else 'Velocity (km/s)'

        def _feat_label(name: str) -> str:
            n = str(name).lower()
            if is_pert:
                if 'vp' in n:
                    return r'$\delta\ln V_p$'
                if 'vs' in n:
                    return r'$\delta\ln V_s$'
            return str(name)

        # 特征样式：同轴叠画，线型区分 Vp/Vs，阴影用低透明度避免混色
        feat_styles = [
            {
                'color': '#2563EB',
                'ls': '-',
                'label': _feat_label(feature_names[0] if n_features > 0 else 'vp'),
            },
            {
                'color': '#DC2626',
                'ls': '--',
                'label': _feat_label(feature_names[1] if n_features > 1 else 'vs'),
            },
        ]

        def _depth_profile(
            cluster_id: int,
        ) -> Optional[Dict[str, Any]]:
            """单簇按深度分箱的均值±IQR 剖面。"""
            mask = labels == cluster_id
            n_pts = int(np.sum(mask))
            if n_pts < 5:
                return None
            z = depths_all[spatial_indices[mask, 2]]
            feats = X_plot[mask, :n_features]
            z_min, z_max = float(np.min(z)), float(np.max(z))
            if not np.isfinite(z_min) or z_max <= z_min:
                return None
            # 仅在簇实际深度跨度内分箱（避免全深度空箱）
            n_bins = int(np.clip(np.ceil((z_max - z_min) / 5.0) + 1, 8, 40))
            bins = np.linspace(z_min, z_max, n_bins)
            centers: List[float] = []
            means = [[] for _ in range(n_features)]
            q1s = [[] for _ in range(n_features)]
            q3s = [[] for _ in range(n_features)]
            for j in range(len(bins) - 1):
                right = bins[j + 1] if j < len(bins) - 2 else bins[j + 1] + 1e-9
                dm = (z >= bins[j]) & (z < right)
                if not np.any(dm):
                    continue
                centers.append(0.5 * (bins[j] + bins[j + 1]))
                for fi in range(n_features):
                    vals = feats[dm, fi]
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        means[fi].append(np.nan)
                        q1s[fi].append(np.nan)
                        q3s[fi].append(np.nan)
                    else:
                        means[fi].append(float(np.mean(vals)))
                        q1s[fi].append(float(np.percentile(vals, 25)))
                        q3s[fi].append(float(np.percentile(vals, 75)))
            if not centers:
                return None
            return {
                'n': n_pts,
                'z': np.asarray(centers, dtype=float),
                'mean': [np.asarray(m, dtype=float) for m in means],
                'q1': [np.asarray(q, dtype=float) for q in q1s],
                'q3': [np.asarray(q, dtype=float) for q in q3s],
                'z_span': (z_min, z_max),
                # 簇整体平均（供 facies 解释标注）
                'mu': [
                    float(np.nanmean(feats[:, fi])) for fi in range(n_features)
                ],
            }

        # ---------- 按带组织簇列表 ----------
        band_groups: List[Tuple[str, str, List[int], Tuple[float, float]]] = []
        if per_band:
            for bname, binfo in per_band.items():
                pmask = binfo.get('point_mask')
                if pmask is None:
                    continue
                pmask = np.asarray(pmask, dtype=bool)
                cids = sorted(
                    int(c) for c in np.unique(labels[pmask & (labels >= 0)])
                )
                if not cids:
                    continue
                z0, z1 = binfo.get('depth_range', [np.nan, np.nan])
                k = int(binfo.get('n_clusters', len(cids)))
                subtitle = f'{bname}  ({z0:.0f}–{z1:.0f} km, K={k})'
                band_groups.append((bname, subtitle, cids, (float(z0), float(z1))))
        if not band_groups:
            cids = [int(c) for c in np.unique(labels[labels >= 0])]
            z_lo = float(np.nanmin(depths_all))
            z_hi = float(np.nanmax(depths_all))
            band_groups = [('all', 'all depths', cids, (z_lo, z_hi))]

        max_cid = max((max(g[2]) for g in band_groups if g[2]), default=0)
        colors = generate_distinct_colors(
            max(max_cid + 1, self.config.visualization['n_colors'])
        )

        # 预计算剖面
        profiles: Dict[int, Dict[str, Any]] = {}
        for _, _, cids, _ in band_groups:
            for cid in cids:
                if cid not in profiles:
                    prof = _depth_profile(cid)
                    if prof is not None:
                        profiles[cid] = prof

        # facies 解释所需的 Vp/Vs 特征索引（扰动域才有意义）
        vp_idx = next(
            (
                i for i, nm in enumerate(feature_names[:n_features])
                if 'vp' in str(nm).lower()
            ),
            0,
        )
        vs_idx = next(
            (
                i for i, nm in enumerate(feature_names[:n_features])
                if 'vs' in str(nm).lower()
            ),
            max(n_features - 1, 0),
        )

        # ---------- 布局：每带独立行块，带内统一坐标 ----------
        n_cols = 5
        row_heights: List[float] = []
        band_row_spans: List[
            Tuple[int, int, str, str, List[int], Tuple[float, float]]
        ] = []
        row_cursor = 0
        for bname, subtitle, cids, z_rng in band_groups:
            cids_ok = [c for c in cids if c in profiles]
            if not cids_ok:
                continue
            n_r = (len(cids_ok) + n_cols - 1) // n_cols
            band_row_spans.append((row_cursor, n_r, bname, subtitle, cids_ok, z_rng))
            row_heights.extend([1.0] * n_r)
            row_cursor += n_r

        if not band_row_spans:
            self.logger.warning("      ⚠️ 无可用簇剖面，跳过 2-3-6")
            return

        F = self.fonts
        n_rows_total = len(row_heights)
        fig_w = 3.8 * n_cols
        fig_h = 3.3 * n_rows_total + 0.7 * len(band_row_spans)
        fig = plt.figure(figsize=(fig_w, max(fig_h, 6.0)))
        height_ratios: List[float] = []
        for _start, n_r, _bname, _subtitle, _cids_ok, _z_rng in band_row_spans:
            height_ratios.append(0.38)  # band 标题行
            height_ratios.extend([1.0] * n_r)

        gs = fig.add_gridspec(
            len(height_ratios),
            n_cols,
            height_ratios=height_ratios,
            hspace=0.55,
            wspace=0.32,
            left=0.07,
            right=0.98,
            top=0.93,
            bottom=0.05,
        )

        # 带内共享 xlim：取各簇均值与 IQR 的 2–98 分位
        band_xlim: Dict[str, Tuple[float, float]] = {}
        for _start, _n_r, _bname, subtitle, cids_ok, _z_rng in band_row_spans:
            vals: List[float] = []
            for cid in cids_ok:
                p = profiles[cid]
                for fi in range(n_features):
                    vals.extend(p['mean'][fi][np.isfinite(p['mean'][fi])].tolist())
                    vals.extend(p['q1'][fi][np.isfinite(p['q1'][fi])].tolist())
                    vals.extend(p['q3'][fi][np.isfinite(p['q3'][fi])].tolist())
            if not vals:
                band_xlim[subtitle] = (-0.05, 0.05)
                continue
            lo, hi = np.percentile(vals, [2, 98])
            pad = 0.12 * max(hi - lo, 1e-4)
            if is_pert:
                m = max(abs(lo), abs(hi)) + pad
                band_xlim[subtitle] = (-m, m)
            else:
                band_xlim[subtitle] = (float(lo - pad), float(hi + pad))

        legend_handles: List[Any] = []
        legend_done = False
        h_cursor = 0
        for _start, n_r, bname, subtitle, cids_ok, z_rng in band_row_spans:
            ax_t = fig.add_subplot(gs[h_cursor, :])
            ax_t.axis('off')
            ax_t.text(
                0.0,
                0.3,
                subtitle,
                transform=ax_t.transAxes,
                fontsize=F['title'],
                fontweight='bold',
                color='0.2',
                va='center',
            )
            h_cursor += 1

            xlim = band_xlim.get(subtitle, (-0.05, 0.05))
            n_band = sum(int(profiles[c]['n']) for c in cids_ok)
            n_band = max(n_band, 1)

            for r in range(n_r):
                for c in range(n_cols):
                    idx = r * n_cols + c
                    ax = fig.add_subplot(gs[h_cursor, c])
                    if idx >= len(cids_ok):
                        ax.axis('off')
                        continue
                    cid = cids_ok[idx]
                    p = profiles[cid]
                    cc = colors[cid % len(colors)]
                    z_lo, z_hi = p['z_span']
                    z_pad = 0.03 * max(z_hi - z_lo, 1.0)
                    pct = 100.0 * float(p['n']) / float(n_band)

                    for fi in range(n_features):
                        st = feat_styles[fi % len(feat_styles)]
                        mu = p['mean'][fi]
                        q1 = p['q1'][fi]
                        q3 = p['q3'][fi]
                        zz = p['z']
                        ok = np.isfinite(mu) & np.isfinite(zz)
                        if not np.any(ok):
                            continue
                        ax.fill_betweenx(
                            zz[ok],
                            q1[ok],
                            q3[ok],
                            color=st['color'],
                            alpha=0.12,
                            linewidth=0,
                            zorder=1,
                        )
                        (line,) = ax.plot(
                            mu[ok],
                            zz[ok],
                            linestyle=st['ls'],
                            color=st['color'],
                            linewidth=2.0,
                            label=st['label'],
                            zorder=3,
                        )
                        if not legend_done:
                            legend_handles.append(line)
                    if len(legend_handles) >= n_features:
                        legend_done = True

                    if is_pert:
                        ax.axvline(
                            0.0,
                            color='0.5',
                            linewidth=0.7,
                            linestyle=':',
                            zorder=0,
                        )
                    ax.set_xlim(xlim)
                    ax.set_ylim(z_hi + z_pad, z_lo - z_pad)  # 深度向下
                    ax.spines['left'].set_color(cc)
                    ax.spines['left'].set_linewidth(2.5)
                    for spine in ('top', 'right'):
                        ax.spines[spine].set_visible(False)

                    ax.set_title(
                        f'C{cid}  ({pct:.1f}%)',
                        fontsize=F['title'],
                        fontweight='bold',
                        color=cc,
                        pad=6,
                    )
                    # facies 推测解释（仅扰动域有意义）
                    if is_pert and p.get('mu') is not None:
                        interp = interpret_facies(
                            bname,
                            p['mu'][vp_idx] if vp_idx < len(p['mu']) else np.nan,
                            p['mu'][vs_idx] if vs_idx < len(p['mu']) else np.nan,
                        )
                        # 放在子图标题正下方（轴内顶部）
                        ax.text(
                            0.5,
                            0.985,
                            interp,
                            transform=ax.transAxes,
                            ha='center',
                            va='top',
                            fontsize=max(F['annotation'] - 3, 9),
                            color='0.25',
                            style='italic',
                            wrap=True,
                            bbox=dict(
                                facecolor='white',
                                alpha=0.75,
                                edgecolor='none',
                                pad=1.5,
                            ),
                            zorder=6,
                        )
                    ax.grid(True, axis='both', alpha=0.25, linestyle='--', linewidth=0.6)
                    ax.tick_params(labelsize=F['tick'])

                    if c == 0:
                        ax.set_ylabel('Depth (km)', fontsize=F['label'])
                    else:
                        ax.set_ylabel('')
                    if r == n_r - 1:
                        ax.set_xlabel(xlabel, fontsize=F['label'])
                    else:
                        ax.set_xlabel('')
                h_cursor += 1

        if legend_handles:
            fig.legend(
                legend_handles,
                [h.get_label() for h in legend_handles],
                loc='upper right',
                ncol=n_features,
                fontsize=F['legend'],
                frameon=True,
                fancybox=False,
                edgecolor='0.7',
                bbox_to_anchor=(0.98, 0.995),
            )

        title_x = 'dlnV' if is_pert else 'velocity'
        fig.suptitle(
            f'{model_name} — {algorithm_name.upper()} Cluster Profiles '
            f'(vp & vs on same axis; mean ± IQR; {title_x})',
            fontsize=F['suptitle'],
            fontweight='bold',
            y=0.995,
        )

        filepath = output_dir / f'2-3-6_{algorithm_name}_cluster_profiles.png'
        fig.savefig(
            filepath, dpi=self.config.visualization['dpi'], bbox_inches='tight'
        )
        # 同步 jpg
        try:
            fig.savefig(
                filepath.with_suffix('.jpg'),
                dpi=self.config.visualization['dpi'],
                bbox_inches='tight',
            )
        except Exception:
            pass
        plt.close(fig)
        self.logger.info(f"      ✅ 保存: 2-3-6_{algorithm_name}_cluster_profiles.png")
    
    def plot_feature_distributions(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Path,
        algorithm_name: str,
        model_name: str,
        per_band: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        绘制聚类特征分布图。

        有 per_band 时按深度带分面（行=带，列=特征）；
        每格为小提琴 + 内嵌箱线（无离群点），簇按中位数排序。
        """
        self.logger.info(f"    📊 绘制特征分布 ({algorithm_name})...")
        self.fonts = apply_clustering_plot_style(self.config)

        feature_names = list(
            metadata.get('feature_display_names')
            or metadata.get(
                'features',
                [f'Feature {i + 1}' for i in range(X.shape[1])],
            )
        )
        n_features = min(2, X.shape[1])  # 默认 vp/vs 两列
        labels = np.asarray(labels)
        X = np.asarray(X)

        # 反标准化到扰动/原始特征空间（与训练一致）
        X_plot = X.copy()
        if (
            self.processor is not None
            and hasattr(self.processor, 'scaler')
            and self.processor.scaler is not None
        ):
            try:
                X_plot = self.processor.scaler.inverse_transform(X)
            except Exception:
                X_plot = X

        max_cid = int(labels.max()) if np.any(labels >= 0) else 0
        colors = generate_distinct_colors(
            max(max_cid + 1, self.config.visualization['n_colors'])
        )
        rng = np.random.default_rng(RANDOM_SEED)
        max_sample = 3000

        def _cluster_series(
            point_mask: np.ndarray, feat_idx: int
        ) -> Tuple[List[int], List[np.ndarray], List[int]]:
            """返回按中位数排序的 (cluster_ids, samples, counts)"""
            cids: List[int] = []
            samples: List[np.ndarray] = []
            counts: List[int] = []
            local = labels[point_mask]
            xloc = X_plot[point_mask, feat_idx]
            for cid in np.unique(local[local >= 0]):
                vals = xloc[local == cid]
                vals = vals[np.isfinite(vals)]
                if vals.size < 5:
                    continue
                n_tot = int(vals.size)
                if n_tot > max_sample:
                    vals = rng.choice(vals, size=max_sample, replace=False)
                cids.append(int(cid))
                samples.append(vals.astype(float))
                counts.append(n_tot)
            if not cids:
                return [], [], []
            order = np.argsort([float(np.median(s)) for s in samples])
            return (
                [cids[i] for i in order],
                [samples[i] for i in order],
                [counts[i] for i in order],
            )

        def _draw_violin_panel(
            ax,
            cids: List[int],
            samples: List[np.ndarray],
            counts: List[int],
            feat_name: str,
            panel_title: str,
            ylabel: bool,
        ) -> None:
            F = self.fonts
            if not cids:
                ax.axis('off')
                ax.set_title(f'{panel_title}\n(no data)', fontsize=F['title'])
                return
            # 长表供 seaborn
            rows_v: List[float] = []
            rows_c: List[str] = []
            for cid, samp in zip(cids, samples):
                rows_v.extend(samp.tolist())
                rows_c.extend([f'C{cid}'] * len(samp))
            df = pd.DataFrame({'value': rows_v, 'cluster': rows_c})
            order = [f'C{c}' for c in cids]
            palette = {
                f'C{c}': colors[c % len(colors)] for c in cids
            }
            try:
                sns.violinplot(
                    data=df,
                    x='cluster',
                    y='value',
                    order=order,
                    hue='cluster',
                    hue_order=order,
                    palette=palette,
                    ax=ax,
                    cut=0,
                    inner=None,
                    linewidth=0.6,
                    density_norm='width',
                    legend=False,
                )
            except TypeError:
                # 兼容旧版 seaborn
                sns.violinplot(
                    data=df,
                    x='cluster',
                    y='value',
                    order=order,
                    palette=palette,
                    ax=ax,
                    cut=0,
                    inner=None,
                    linewidth=0.6,
                    scale='width',
                )
            # 内嵌箱线（无 fliers）
            bp = ax.boxplot(
                samples,
                positions=np.arange(len(cids)),
                widths=0.12,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )
            for patch, cid in zip(bp['boxes'], cids):
                patch.set_facecolor('white')
                patch.set_alpha(0.85)
                patch.set_edgecolor(colors[cid % len(colors)])
            for key in ('whiskers', 'caps', 'medians'):
                for line in bp[key]:
                    line.set_color('0.2')
                    line.set_linewidth(0.8)
            # 簇 ID + 带内占比（%）
            n_tot = max(int(sum(counts)), 1)
            ax.set_xticks(np.arange(len(cids)))
            ax.set_xticklabels(
                [
                    f'C{c}\n{100.0 * n / n_tot:.1f}%'
                    for c, n in zip(cids, counts)
                ],
                fontsize=F['tick'],
                linespacing=1.2,
            )
            ax.tick_params(axis='x', pad=2, labelsize=F['tick'])
            ax.tick_params(axis='y', labelsize=F['tick'])
            ax.axhline(0.0, color='gray', linewidth=0.7, linestyle='--', alpha=0.7)
            ax.set_xlabel('')
            ax.set_ylabel(feat_name if ylabel else '', fontsize=F['label'])
            ax.set_title(panel_title, fontsize=F['title'], fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')

        # ---------- 按深度带分面 ----------
        band_items: List[Tuple[str, np.ndarray, str]] = []
        if per_band:
            for bname, binfo in per_band.items():
                pmask = binfo.get('point_mask')
                if pmask is None:
                    continue
                pmask = np.asarray(pmask, dtype=bool)
                if not np.any(pmask):
                    continue
                z0, z1 = binfo.get('depth_range', [np.nan, np.nan])
                k = int(binfo.get('n_clusters', 0))
                subtitle = f'{bname}  ({z0:.0f}–{z1:.0f} km, K={k})'
                band_items.append((bname, pmask, subtitle))

        if band_items:
            n_rows = len(band_items)
            n_cols = n_features
            # 按各带簇数动态加宽，避免浅部 K≈10 时标签挤叠
            max_k = 2
            for bname, pmask, _sub in band_items:
                k_cfg = int(per_band.get(bname, {}).get('n_clusters', 0) or 0)
                k_obs = int(len(np.unique(labels[pmask & (labels >= 0)])))
                max_k = max(max_k, k_cfg, k_obs)
            # 百分比标注较短：略宽于紧凑版，避免小提琴过挤
            panel_w = max(6.0, 0.90 * max_k)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(panel_w * n_cols, 3.8 * n_rows),
                squeeze=False,
            )
            for r, (_bname, pmask, subtitle) in enumerate(band_items):
                for c in range(n_features):
                    feat_name = (
                        feature_names[c]
                        if c < len(feature_names)
                        else f'Feature {c + 1}'
                    )
                    cids, samples, counts = _cluster_series(pmask, c)
                    _draw_violin_panel(
                        axes[r, c],
                        cids,
                        samples,
                        counts,
                        feat_name,
                        panel_title=f'{subtitle}  ·  {feat_name}',
                        ylabel=(c == 0),
                    )
            F = self.fonts
            fig.suptitle(
                f'{model_name} — {algorithm_name.upper()} Feature Distributions '
                f'(by depth band; violin + quartile box)',
                fontsize=F['suptitle'],
                fontweight='bold',
                y=0.998,
            )
            fig.tight_layout(rect=[0, 0.01, 1, 0.96], h_pad=1.0, w_pad=0.6)
        else:
            # 回退：全部簇一行
            n_cols = n_features
            n_all = int(len(np.unique(labels[labels >= 0])))
            n_all = max(n_all, 8)
            fig, axes = plt.subplots(
                1,
                n_cols,
                figsize=(max(6.0, 0.75 * n_all) * n_cols, 4.5),
                squeeze=False,
            )
            all_mask = labels >= 0
            for c in range(n_features):
                feat_name = (
                    feature_names[c]
                    if c < len(feature_names)
                    else f'Feature {c + 1}'
                )
                cids, samples, counts = _cluster_series(all_mask, c)
                _draw_violin_panel(
                    axes[0, c],
                    cids,
                    samples,
                    counts,
                    feat_name,
                    panel_title=feat_name,
                    ylabel=(c == 0),
                )
            F = self.fonts
            fig.suptitle(
                f'{model_name} — {algorithm_name.upper()} Feature Distributions '
                f'(violin + quartile box)',
                fontsize=F['suptitle'],
                fontweight='bold',
            )
            fig.tight_layout(rect=[0, 0.01, 1, 0.94], w_pad=0.6)

        filepath = output_dir / f'2-3-7_{algorithm_name}_feature_distributions.png'
        fig.savefig(
            filepath, dpi=self.config.visualization['dpi'], bbox_inches='tight'
        )
        plt.close(fig)
        self.logger.info(
            f"      ✅ 保存: 2-3-7_{algorithm_name}_feature_distributions.png"
        )

    def plot_cluster_centers(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        metadata: Dict[str, Any],
        gmm_model: Optional[GaussianMixture],
        output_dir: Path,
        algorithm_name: str,
        model_name: str
    ) -> None:
        """绘制聚类中心的速度结构"""
        self.logger.info(f"    📊 绘制聚类中心 ({algorithm_name})...")
        
        unique_labels = np.unique(labels[labels >= 0])
        n_clusters = len(unique_labels)
        
        # 获取聚类中心
        if gmm_model and hasattr(gmm_model, 'means_'):
            centers = gmm_model.means_
        else:
            # 手动计算中心
            centers = []
            for label in unique_labels:
                mask = labels == label
                if np.sum(mask) > 0:
                    centers.append(np.mean(X[mask], axis=0))
            centers = np.array(centers)
        
        # 反归一化
        if self.processor and hasattr(self.processor, 'scaler') and self.processor.scaler:
            centers_original = self.processor.scaler.inverse_transform(centers)
        else:
            centers_original = centers
        
        # 创建图形
        n_features = centers.shape[1]
        feature_names = metadata.get('features', [f'Feature {i+1}' for i in range(n_features)])
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 生成颜色
        colors = generate_distinct_colors(max(n_clusters, self.config.visualization['n_colors']))
        
        # 1. 热力图
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(centers_original.T, aspect='auto', cmap='RdYlBu_r')
        ax1.set_xlabel('Cluster ID', fontsize=15)
        ax1.set_ylabel('Features', fontsize=15)
        ax1.set_title('Cluster Centers Heatmap', fontsize=16, fontweight='bold')
        ax1.set_yticks(range(n_features))
        ax1.set_yticklabels(feature_names[:n_features], fontsize=14)
        ax1.set_xticks(range(n_clusters))
        plt.colorbar(im, ax=ax1, label='Value (km/s or g/cm³)')
        
        # 2. 平行坐标图
        ax2 = fig.add_subplot(gs[0, 1])
        centers_norm = (centers_original - centers_original.min(axis=0)) / \
                      (centers_original.max(axis=0) - centers_original.min(axis=0) + 1e-10)
        
        x = np.arange(n_features)
        for i in range(n_clusters):
            ax2.plot(x, centers_norm[i], 'o-', 
                    color=colors[i % len(colors)], 
                    label=f'C{i}', alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Features', fontsize=15)
        ax2.set_ylabel('Normalized Value', fontsize=15)
        ax2.set_title('Cluster Centers - Parallel Coordinates', fontsize=16, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(feature_names[:n_features], rotation=45, ha='right', fontsize=14)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, ncol=max(1, n_clusters//10))
        ax2.grid(True, alpha=0.3)
        
        # 3. Vp-Vs关系
        ax3 = fig.add_subplot(gs[0, 2])
        if 'vp' in feature_names and 'vs' in feature_names:
            vp_idx = feature_names.index('vp')
            vs_idx = feature_names.index('vs')
            
            for i in range(n_clusters):
                ax3.scatter(centers_original[i, vs_idx], centers_original[i, vp_idx],
                          s=200, c=[colors[i % len(colors)]], edgecolors='black', linewidth=2,
                          label=f'Cluster {i}', zorder=10, alpha=0.8)
            
            vs_range = np.linspace(centers_original[:, vs_idx].min(), 
                                  centers_original[:, vs_idx].max(), 100)
            ax3.plot(vs_range, vs_range * 1.732, 'k--', alpha=0.3, linewidth=2, label='Vp/Vs=1.732')
            
            ax3.set_xlabel('Vs (km/s)', fontsize=15)
            ax3.set_ylabel('Vp (km/s)', fontsize=15)
            ax3.set_title('Cluster Centers - Vp vs Vs', fontsize=16, fontweight='bold')
            ax3.legend(fontsize=14, ncol=2)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Vp/Vs plot\nnot available', 
                    ha='center', va='center', fontsize=16, transform=ax3.transAxes)
            ax3.set_title('Cluster Centers - Vp vs Vs', fontsize=16, fontweight='bold')
        
        # 4. 聚类大小分布
        ax4 = fig.add_subplot(gs[1, 0])
        unique_labels_list, counts = np.unique(labels[labels >= 0], return_counts=True)
        bars = ax4.bar(unique_labels_list, counts, 
                      color=[colors[i % len(colors)] for i in range(len(unique_labels_list))],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('Cluster ID', fontsize=22)
        ax4.set_ylabel('Number of Points', fontsize=22)
        ax4.set_title('Cluster Size Distribution', fontsize=22, fontweight='bold')
        ax4.set_xticks(unique_labels_list)
        ax4.grid(True, alpha=0.3, axis='y')
        
        total = np.sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}\n({count/total:.1%})',
                    ha='center', va='bottom', fontsize=14)
        
        # 5. 特征值表格
        ax5 = fig.add_subplot(gs[1, 1:])
        ax5.axis('off')
        
        # 创建表格数据
        table_data = [['Cluster'] + feature_names[:n_features]]
        for i in range(n_clusters):
            row = [f'C{i}'] + [f'{centers_original[i, j]:.3f}' for j in range(n_features)]
            table_data.append(row)
        
        table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.1] + [0.9/n_features]*n_features)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # 设置表头样式
        for j in range(len(table_data[0])):
            table[(0, j)].set_facecolor('#3498db')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        # 设置聚类行颜色
        for i in range(1, len(table_data)):
            table[(i, 0)].set_facecolor(colors[(i-1) % len(colors)])
            table[(i, 0)].set_text_props(weight='bold')
        
        plt.suptitle(f'{model_name} - {algorithm_name.upper()} Cluster Centers Analysis',
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        filepath = output_dir / f'2-3-8_{algorithm_name}_cluster_centers.png'
        plt.savefig(filepath, dpi=self.config.visualization['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"      ✅ 保存: 2-3-8_{algorithm_name}_cluster_centers.png")

    def plot_wkmeans_feature_weights(
        self,
        per_band: Dict[str, Any],
        metadata: Dict[str, Any],
        output_dir: Path,
        model_name: str,
    ) -> None:
        """
        绘制 W-k-means 各深度带的特征权重与簇中心。

        特征权重是 W-k-means 相对普通 k-means 的核心产出：它直接回答"哪个
        特征在这一带的分簇中起作用"。权重接近均分说明各特征贡献相当，
        某一特征权重显著偏高则说明该带的分簇几乎由它单独决定。

        Args:
            per_band: 分带结果字典，需含 feature_weights / cluster_centers / n_clusters
            metadata: 含 feature_display_names 的元数据
            output_dir: 图件输出目录
            model_name: 模型名
        """
        self.logger.info("    📊 绘制 W-k-means 特征权重...")
        F = self.fonts = apply_clustering_plot_style(self.config)

        bands = [(b, v) for b, v in per_band.items()
                 if v.get('feature_weights') is not None]
        if not bands:
            return

        names = list(metadata.get('feature_display_names')
                     or metadata.get('features', []))
        n_feat = len(np.asarray(bands[0][1]['feature_weights']))
        names = (names + [f'f{i}' for i in range(n_feat)])[:n_feat]

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

        # (a) 各带特征权重的分组柱状图
        ax = axes[0]
        x = np.arange(len(bands))
        width = 0.8 / max(n_feat, 1)
        for j, fname in enumerate(names):
            vals = [float(np.asarray(v['feature_weights'])[j]) for _, v in bands]
            ax.bar(x + (j - (n_feat - 1) / 2) * width, vals, width,
                   label=fname, edgecolor='black', linewidth=0.6)
        ax.axhline(1.0 / n_feat, color='crimson', ls='--', lw=1.5,
                   label=f'Equal ({1.0 / n_feat:.2f})')
        ax.set_xticks(x)
        ax.set_xticklabels([b for b, _ in bands], rotation=15, ha='right')
        ax.set_ylabel('Feature weight', fontsize=F['label'])
        ax.set_title('W-k-means feature weights per band',
                     fontsize=F['title'] - 3, fontweight='bold')
        ax.legend(fontsize=F['legend'] - 2)
        ax.grid(True, axis='y', alpha=0.3)

        # (b) 各带簇中心（物理量纲），点大小随带内簇序递增
        ax = axes[1]
        cmap = plt.get_cmap('tab10', max(len(bands), 3))
        for i, (bname, v) in enumerate(bands):
            cen = v.get('cluster_centers')
            if cen is None:
                continue
            cen = np.asarray(cen, dtype=float)
            if cen.shape[1] < 2:
                continue
            ix = next((j for j, n in enumerate(names)
                       if str(n).lower().endswith('vs')), 0)
            iy = next((j for j, n in enumerate(names)
                       if str(n).lower().endswith('vp')), 1)
            ax.plot(cen[:, ix], cen[:, iy], 'o-', color=cmap(i), ms=8,
                    lw=1.2, alpha=0.85,
                    label=f"{bname} (K={v['n_clusters']})",
                    markeredgecolor='black', markeredgewidth=0.6)
            ax.set_xlabel(names[ix], fontsize=F['label'])
            ax.set_ylabel(names[iy], fontsize=F['label'])
        ax.set_title('Cluster centers by band', fontsize=F['title'] - 3,
                     fontweight='bold')
        ax.legend(fontsize=F['legend'] - 2)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'{model_name} — W-k-means feature weighting',
                     fontsize=F['suptitle'] - 4, fontweight='bold')
        fig.tight_layout()

        stem = '2-3-16_wkmeans_feature_weights'
        for fmt in self.config.visualization.get('save_formats', ['jpg']):
            fig.savefig(output_dir / f'{stem}.{fmt}',
                        dpi=self.config.visualization['dpi'],
                        bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"      ✅ 保存: {stem}")

    def plot_radial_stratification(
        self,
        cluster_results: Dict[str, Any],
        metadata: Dict[str, Any],
        output_dir: Path,
        model_name: str,
        radial_config: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        绘制全深度聚类的径向分层结构。

        原始速度值随深度单调增长，故全深度域一次性聚类得到的簇必然构成
        径向壳层序列。本图从三个角度解读该序列：

        1. 各簇的深度占位区间与参考间断面（410/660 km）的相对位置，
           检验聚类是否复现了地幔相变引起的速度跃变
        2. 层界深度图：层界的横向起伏即构造信号。俯冲板片使 410 km 抬升、
           660 km 下沉（两处相变的 Clapeyron 斜率符号相反）
        3. 层界深度的中位数与起伏幅度，是有量纲的物理量，可跨模型直接比较，
           不受簇编号任意性影响

        Args:
            cluster_results: W-k-means 结果字典，需含 labels_3d / radial_stats /
                feature_weights / cluster_centers
            metadata: 含 lats / lons / depths 的元数据
            output_dir: 图件输出目录
            model_name: 模型名
            radial_config: 含 reference_discontinuities / section_latitude 的配置

        Returns:
            层界深度统计表（boundary / depth_median / depth_p05 / depth_p95 / relief）
        """
        self.logger.info("    📊 绘制径向分层分析...")
        F = self.fonts = apply_clustering_plot_style(self.config)

        lats = np.asarray(metadata['lats'], dtype=float)
        lons = np.asarray(metadata['lons'], dtype=float)
        depths = np.asarray(metadata['depths'], dtype=float)
        labels_3d = np.asarray(cluster_results['labels_3d'])
        stats = cluster_results['radial_stats']
        n_clusters = int(cluster_results['n_clusters'])

        refs = [float(z) for z in radial_config.get(
            'reference_discontinuities', [410.0, 660.0])]
        sec_lat = float(radial_config.get('section_latitude', 35.0))

        bnd = self._boundary_depth_table(labels_3d, depths, n_clusters, model_name)
        maps = self._boundary_depth_maps(labels_3d, depths, n_clusters)

        cmap = plt.get_cmap('tab10', n_clusters)
        fig = plt.figure(figsize=(20, 11))
        gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.26,
                              width_ratios=[1.0, 1.0, 1.5])

        # (a) 簇中心在物理量纲下的分布，按平均深度着色
        ax = fig.add_subplot(gs[0, 0])
        cen_cols = [c for c in stats.columns if c.startswith('center_')]
        if len(cen_cols) >= 2:
            xcol = next((c for c in cen_cols if c.endswith('vs')), cen_cols[1])
            ycol = next((c for c in cen_cols if c.endswith('vp')), cen_cols[0])
            sc = ax.scatter(stats[xcol], stats[ycol], c=stats['depth_mean'],
                            s=190, cmap='viridis', edgecolor='black',
                            linewidth=1.3, zorder=3)
            for _, r in stats.iterrows():
                ax.annotate(f"{int(r['cluster'])}", (r[xcol], r[ycol]),
                            ha='center', va='center', fontsize=8,
                            fontweight='bold', color='white', zorder=4)
            fig.colorbar(sc, ax=ax, label='Mean depth (km)')
            ax.set_xlabel(xcol.replace('center_', ''), fontsize=F['label'])
            ax.set_ylabel(ycol.replace('center_', ''), fontsize=F['label'])
        ax.set_title('Cluster centers', fontsize=F['title'] - 3,
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

        # (b) 各簇的深度占位区间
        ax = fig.add_subplot(gs[0, 1])
        for _, r in stats.iterrows():
            l = int(r['cluster'])
            ax.plot([r['depth_p05'], r['depth_p95']], [l, l], lw=7,
                    color=cmap(l), solid_capstyle='butt')
            ax.plot(r['depth_mean'], l, 'k|', ms=13, mew=2)
        for zd in refs:
            ax.axvline(zd, color='crimson', ls='--', lw=1.5)
            ax.text(zd, -0.7, f'{zd:.0f}', color='crimson',
                    fontsize=F['annotation'] - 4, ha='center')
        ax.set_xlabel('Depth (km)', fontsize=F['label'])
        ax.set_ylabel('Cluster (ordered by depth)', fontsize=F['label'])
        ax.set_title('Depth occupancy (5th-95th pct)',
                     fontsize=F['title'] - 3, fontweight='bold')
        ax.set_yticks(range(n_clusters))
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)

        # (c) 经度—深度剖面
        ilat = int(np.argmin(np.abs(lats - sec_lat)))
        ax = fig.add_subplot(gs[0, 2])
        sec = labels_3d[ilat, :, :].astype(float)
        sec[labels_3d[ilat, :, :] < 0] = np.nan
        im = ax.pcolormesh(lons, depths, sec.T, cmap=cmap, vmin=-0.5,
                           vmax=n_clusters - 0.5, shading='auto')
        for zd in refs:
            ax.axhline(zd, color='crimson', ls='--', lw=1.5)
        ax.invert_yaxis()
        ax.set_xlabel('Longitude (°E)', fontsize=F['label'])
        ax.set_ylabel('Depth (km)', fontsize=F['label'])
        ax.set_title(f'Section at {lats[ilat]:.0f}°N '
                     f'(dashed: {", ".join(f"{z:.0f}" for z in refs)} km)',
                     fontsize=F['title'] - 3, fontweight='bold')
        fig.colorbar(im, ax=ax, ticks=range(n_clusters), label='Cluster',
                     pad=0.01)

        # (d)(e) 最接近参考间断面的两个层界的深度图
        if not bnd.empty:
            for i, zd in enumerate(refs[:2]):
                k = int(bnd.iloc[(bnd['depth_median'] - zd).abs().argmin()]
                        ['boundary'])
                ax = fig.add_subplot(gs[1, i])
                dmap = maps[k]
                med = float(np.nanmedian(dmap))
                span = float(np.nanpercentile(np.abs(dmap - med), 95)) or 1.0
                im = ax.pcolormesh(lons, lats, dmap, cmap='RdBu',
                                   vmin=med - span, vmax=med + span,
                                   shading='auto')
                fig.colorbar(im, ax=ax, label='Depth (km)')
                ax.set_title(f'Boundary {k}: median {med:.0f} km '
                             f'(ref {zd:.0f} km)', fontsize=F['title'] - 4,
                             fontweight='bold')
                ax.set_xlabel('Longitude (°E)', fontsize=F['label'] - 2)
                if i == 0:
                    ax.set_ylabel('Latitude (°N)', fontsize=F['label'] - 2)
                ax.set_aspect('equal')

            # (f) 层界深度与横向起伏汇总
            ax = fig.add_subplot(gs[1, 2])
            ax.errorbar(
                bnd['depth_median'], bnd['boundary'],
                xerr=[bnd['depth_median'] - bnd['depth_p05'],
                      bnd['depth_p95'] - bnd['depth_median']],
                fmt='o', color='navy', ecolor='steelblue', elinewidth=3,
                capsize=4, ms=8,
            )
            for zd in refs:
                ax.axvline(zd, color='crimson', ls='--', lw=1.5)
                ax.text(zd, bnd['boundary'].min() - 0.5, f'{zd:.0f} km',
                        color='crimson', fontsize=F['annotation'] - 4,
                        ha='center')
            ax.set_xlabel('Boundary depth (km)', fontsize=F['label'])
            ax.set_ylabel('Boundary index', fontsize=F['label'])
            ax.set_title('Boundary depth and lateral relief (5th-95th pct)',
                         fontsize=F['title'] - 3, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)

        weights = cluster_results.get('feature_weights')
        wtxt = ''
        if weights is not None:
            names = list(metadata.get('feature_display_names')
                         or metadata.get('features', []))
            wtxt = '   weights: ' + ', '.join(
                f'{n}={w:.3f}' for n, w in zip(names, np.asarray(weights))
            )
        fig.suptitle(
            f'{model_name} — W-k-means radial stratification '
            f'(K={n_clusters}){wtxt}',
            fontsize=F['suptitle'] - 4, fontweight='bold',
        )

        stem = '2-3-14_wkmeans_radial_stratification'
        for fmt in self.config.visualization.get('save_formats', ['jpg']):
            fig.savefig(output_dir / f'{stem}.{fmt}',
                        dpi=self.config.visualization['dpi'],
                        bbox_inches='tight')
        plt.close(fig)

        for _, r in bnd.iterrows():
            self.logger.info(
                f"      界{int(r['boundary'])}  中位 {r['depth_median']:>6.1f} km  "
                f"P5-P95 {r['depth_p05']:>6.1f}-{r['depth_p95']:>6.1f}  "
                f"起伏 {r['relief']:>5.1f} km"
            )
        self.logger.info(f"      ✅ 保存: {stem}")
        return bnd

    @staticmethod
    def _boundary_depth_maps(
        labels_3d: np.ndarray, depths: np.ndarray, n_clusters: int
    ) -> Dict[int, np.ndarray]:
        """
        计算各层界（簇 k 的顶界）在每个地理网格点上的深度

        簇已按平均深度排序，故层界 k 定义为沿深度方向标签首次达到 k 的深度。
        该定义对局部非单调（如板片造成的标签反转）稳健，取首次穿越深度。
        """
        maps: Dict[int, np.ndarray] = {}
        invalid_column = np.all(labels_3d < 0, axis=2)
        for k in range(1, n_clusters):
            reached = labels_3d >= k
            any_reached = np.any(reached, axis=2)
            first_idx = np.argmax(reached, axis=2)
            dmap = np.where(any_reached, depths[first_idx], np.nan)
            dmap[invalid_column] = np.nan
            maps[k] = dmap
        return maps

    def _boundary_depth_table(
        self,
        labels_3d: np.ndarray,
        depths: np.ndarray,
        n_clusters: int,
        model_name: str,
    ) -> pd.DataFrame:
        """汇总各层界的中位深度与横向起伏（P95-P5）"""
        maps = self._boundary_depth_maps(labels_3d, depths, n_clusters)
        rows: List[Dict[str, Any]] = []
        for k, dmap in maps.items():
            valid = dmap[np.isfinite(dmap)]
            if valid.size == 0:
                continue
            p05, p95 = np.percentile(valid, [5, 95])
            rows.append({
                'model': model_name,
                'boundary': k,
                'depth_median': float(np.median(valid)),
                'depth_p05': float(p05),
                'depth_p95': float(p95),
                'relief': float(p95 - p05),
            })
        return pd.DataFrame(rows)

    def _crossplot_overview_panel(
        self,
        ax: Any,
        X_plot: np.ndarray,
        labels: np.ndarray,
        band_items: List[Tuple[str, np.ndarray, str]],
        ix: int,
        iy: int,
        ref_slope: float,
        ref_label: str,
        xlabel: str,
        ylabel: str,
        is_perturbation: bool,
        max_scatter: int,
        rng: np.random.Generator,
        F: Dict[str, int],
    ) -> None:
        """
        绘制交会图的全区域总览面板：全部体元按深度带着色，并框出各带占位。

        各带的单独面板用各自的坐标范围以看清带内结构，代价是读者无法比较
        带与带之间的位置关系。本面板用统一坐标补上这一层：矩形框取每带
        2–98 百分位范围，标注带名，使"哪一带落在特征空间的哪个位置"一目了然。

        Args:
            ax: 目标坐标轴
            X_plot: 已反标准化的特征矩阵
            labels: 簇标签，负值为无效点
            band_items: [(带名, 点掩膜, 子标题)]
            ix, iy: 横轴/纵轴对应的特征列索引
            ref_slope: 参考线斜率
            ref_label: 参考线图例名
            xlabel, ylabel: 轴标签
            is_perturbation: 是否为扰动域
            max_scatter: 散点抽样上限
            rng: 随机数发生器
            F: 字号字典
        """
        # 直接取 tab10 的前 N 色。get_cmap('tab10', N) 在 N<10 时会沿整条色表
        # 重采样，得到的并非前 N 色，且相邻带的配色可能撞色。
        tab10 = plt.get_cmap('tab10').colors
        band_colors = [tab10[i % len(tab10)] for i in range(len(band_items))]
        per_band_quota = max(max_scatter // max(len(band_items), 1), 1)

        all_xy: List[np.ndarray] = []
        boxes: List[Tuple[str, np.ndarray, np.ndarray, Any]] = []

        for i, (bname, pmask, _sub) in enumerate(band_items):
            sel = np.asarray(pmask, dtype=bool) & (labels >= 0)
            if not np.any(sel):
                continue
            xy = np.stack([X_plot[sel, ix], X_plot[sel, iy]], axis=1)
            xy = xy[np.all(np.isfinite(xy), axis=1)]
            if len(xy) < 10:
                continue

            pick = (rng.choice(len(xy), per_band_quota, replace=False)
                    if len(xy) > per_band_quota else np.arange(len(xy)))
            ax.scatter(xy[pick, 0], xy[pick, 1], s=2.0,
                       color=band_colors[i], alpha=0.30, linewidths=0,
                       rasterized=True, label=bname)

            lo = np.percentile(xy, 2, axis=0)
            hi = np.percentile(xy, 98, axis=0)
            boxes.append((bname, lo, hi, band_colors[i]))
            all_xy.append(xy)

        if not all_xy:
            ax.axis('off')
            return

        # 坐标范围取各带方框的并集：地壳的扰动幅度可比地幔大一个量级，若按
        # 全体点的分位数定范围，占比小的地壳会被整个裁到画面之外。
        box_lo = np.min([b[1] for b in boxes], axis=0)
        box_hi = np.max([b[2] for b in boxes], axis=0)
        pad = 0.10 * (box_hi - box_lo)
        if is_perturbation:
            lim = float(np.max(np.abs([box_lo - pad, box_hi + pad])))
            xlim = ylim = (-lim, lim)
            t = np.array([-lim, lim])
            ax.axhline(0, color='0.6', lw=0.6, zorder=1)
            ax.axvline(0, color='0.6', lw=0.6, zorder=1)
        else:
            xlim = (float(box_lo[0] - pad[0]), float(box_hi[0] + pad[0]))
            ylim = (float(box_lo[1] - pad[1]), float(box_hi[1] + pad[1]))
            t = np.array([0.0, xlim[1]])

        # 标注轮流放到方框的不同角上：扰动域下各带方框近似同心嵌套，
        # 统一放同一个角会把四个标签叠在一起
        corners = [(1, 1, 4, 4), (0, 1, -4, 4), (1, 0, 4, -10), (0, 0, -4, -10)]
        for i, (bname, lo, hi, col) in enumerate(boxes):
            ax.add_patch(Rectangle(
                (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                fill=False, edgecolor=col, linewidth=2.0, zorder=6,
            ))
            cx, cy, dx, dy = corners[i % len(corners)]
            ax.annotate(
                bname,
                (hi[0] if cx else lo[0], hi[1] if cy else lo[1]),
                textcoords='offset points', xytext=(dx, dy),
                ha='left' if cx else 'right', color=col,
                fontsize=F['annotation'] - 4, fontweight='bold', zorder=7,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=col,
                          alpha=0.85, lw=0.8),
            )

        ax.plot(t, ref_slope * t, 'k--', lw=1.8, zorder=4,
                label=f'{ref_label} ({ref_slope:.2f})')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if is_perturbation:
            ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=F['tick'])
        ax.set_xlabel(xlabel, fontsize=F['label'])
        ax.set_ylabel(ylabel, fontsize=F['label'])
        ax.set_title(f'All depths  (N = {sum(len(a) for a in all_xy):,})',
                     fontsize=F['title'] - 3, fontweight='bold')

        leg = ax.legend(fontsize=F['legend'] - 3, loc='upper left',
                        framealpha=0.9, markerscale=4)
        for h in leg.legend_handles:
            if hasattr(h, 'set_alpha'):
                h.set_alpha(1.0)

    def _prepare_crossplot_context(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        metadata: Dict[str, Any],
        per_band: Optional[Dict[str, Any]],
        spatial_indices: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """交会图（2D/3D）共用的特征反变换、轴标签与分带组织。"""
        F = apply_clustering_plot_style(self.config)
        labels = np.asarray(labels)

        X_phys = self._physical_feature_matrix(X, spatial_indices, n_features=2)
        features = [str(f).lower() for f in metadata.get('features', [])]

        def _find(tag: str) -> Optional[int]:
            for i, f in enumerate(features):
                if f.endswith(tag):
                    return i
            return None

        ix, iy = _find('vs'), _find('vp')
        if ix is None or iy is None or X_phys.shape[1] < 2:
            return None

        is_perturbation = self._is_perturbation_space(metadata)
        cp_cfg = self.config.visualization.get('crossplot', {})
        X_plot = X_phys
        if is_perturbation:
            X_plot = X_phys * 100.0
            xlabel = r'$\delta\ln V_S$ (%)'
            ylabel = r'$\delta\ln V_P$ (%)'
            ref_label = 'Thermal scaling'
            space_label = 'Perturbation Feature Space'
            ref_slope = float(cp_cfg.get('reference_slope_perturbation',
                                         MANTLE_SCALING_SLOPE))
        else:
            xlabel = r'$V_S$ (km/s)'
            ylabel = r'$V_P$ (km/s)'
            ref_label = r'$V_P/V_S$'
            space_label = 'Velocity Feature Space'
            ref_slope = float(cp_cfg.get('reference_slope_raw',
                                         POISSON_VPVS_RATIO))

        band_items: List[Tuple[str, np.ndarray, str]] = []
        if per_band:
            for bname, binfo in per_band.items():
                pmask = binfo.get('point_mask')
                if pmask is None:
                    continue
                pmask = np.asarray(pmask, dtype=bool)
                if not np.any(pmask):
                    continue
                z0, z1 = binfo.get('depth_range', [np.nan, np.nan])
                k = int(binfo.get('n_clusters', 0))
                subtitle = f'{bname}  ({z0:.0f}–{z1:.0f} km, K={k})'
                band_items.append((bname, pmask, subtitle))
        if not band_items:
            band_items = [('all', labels >= 0, 'All depths')]

        return {
            'F': F,
            'X_plot': X_plot,
            'labels': labels,
            'ix': ix,
            'iy': iy,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'ref_label': ref_label,
            'ref_slope': ref_slope,
            'space_label': space_label,
            'is_perturbation': is_perturbation,
            'band_items': band_items,
            'cp_cfg': cp_cfg,
        }

    def _crossplot_3d_panel(
        self,
        ax: Any,
        X_plot: np.ndarray,
        labels: np.ndarray,
        point_depths: np.ndarray,
        ix: int,
        iy: int,
        colors: List[Any],
        F: Dict[str, int],
        xlabel: str,
        ylabel: str,
        is_perturbation: bool,
        cp_cfg: Dict[str, Any],
    ) -> int:
        """
        三维交会面板：横纵轴为 δlnVs/δlnVp（或 Vp/Vs），竖轴为深度。

        扰动域默认裁剪到 perturbation_lim_3d（±20%），范围内全量散点绘制。

        Returns:
            实际绘制的体元数
        """
        valid = (labels >= 0) & np.all(np.isfinite(X_plot[:, [ix, iy]]), axis=1)
        valid &= np.isfinite(point_depths)
        if not np.any(valid):
            ax.set_axis_off()
            return 0

        xs = X_plot[valid, ix]
        ys = X_plot[valid, iy]
        zs = np.asarray(point_depths[valid], dtype=float)
        lbl = labels[valid]

        xlim = ylim = None
        if is_perturbation:
            lim_cfg = cp_cfg.get('perturbation_lim_3d', [-20.0, 20.0])
            lo, hi = float(lim_cfg[0]), float(lim_cfg[1])
            in_box = (xs >= lo) & (xs <= hi) & (ys >= lo) & (ys <= hi)
            xs, ys, zs, lbl = xs[in_box], ys[in_box], zs[in_box], lbl[in_box]
            xlim = ylim = (lo, hi)
        else:
            xlo, xhi = np.percentile(xs, [1.0, 99.0])
            ylo, yhi = np.percentile(ys, [1.0, 99.0])
            xpad = 0.05 * max(float(xhi - xlo), 1e-3)
            ypad = 0.05 * max(float(yhi - ylo), 1e-3)
            xlim = (max(0.0, float(xlo) - xpad), float(xhi) + xpad)
            ylim = (max(0.0, float(ylo) - ypad), float(yhi) + ypad)

        if len(xs) == 0:
            ax.set_axis_off()
            return 0

        max_scatter = cp_cfg.get('max_scatter_points_3d')
        n = len(xs)
        if max_scatter is not None and int(max_scatter) > 0 and n > int(max_scatter):
            rng = np.random.default_rng(RANDOM_SEED)
            pick = rng.choice(n, int(max_scatter), replace=False)
        else:
            pick = np.arange(n)

        pt_colors = np.array([colors[int(c) % len(colors)] for c in lbl[pick]])
        ax.scatter(
            xs[pick], ys[pick], zs[pick],
            c=pt_colors, s=0.35, alpha=0.18, linewidths=0, depthshade=False,
        )

        # 簇中心（仅在裁剪框内体元上统计）
        for cid in np.unique(lbl):
            m = lbl == cid
            cx, cy, cz = xs[m].mean(), ys[m].mean(), zs[m].mean()
            col = colors[int(cid) % len(colors)]
            ax.scatter(
                [cx], [cy], [cz], s=120, c=[col],
                edgecolors='black', linewidths=1.0, depthshade=False, zorder=5,
            )
            ax.text(
                cx, cy, cz, f'{int(cid)}', fontsize=F['annotation'] - 5,
                ha='center', va='center', fontweight='bold', zorder=6,
            )

        # 410 / 660 km 参考面
        if xlim is not None:
            xx, yy = np.meshgrid(
                np.linspace(xlim[0], xlim[1], 2),
                np.linspace(ylim[0], ylim[1], 2),
            )
        else:
            xspan = float(np.percentile(np.abs(xs), 99.5))
            yspan = float(np.percentile(np.abs(ys), 99.5))
            xx, yy = np.meshgrid(
                np.linspace(-xspan, xspan, 2),
                np.linspace(-yspan, yspan, 2),
            )
        for z_ref in (410.0, 660.0):
            if z_ref <= float(np.nanmax(zs)):
                ax.plot_surface(
                    xx, yy, np.full_like(xx, z_ref),
                    color='0.4', alpha=0.08, linewidth=0, shade=False,
                )

        if xlim is not None:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, fontsize=F['label'] - 2, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=F['label'] - 2, labelpad=8)
        ax.set_zlabel('Depth (km)', fontsize=F['label'] - 2, labelpad=8)
        ax.invert_zaxis()
        ax.view_init(elev=22, azim=-58)
        ax.tick_params(labelsize=F['tick'] - 2)
        ax.grid(True, alpha=0.25)
        return int(len(xs))

    def plot_cluster_crossplot(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Path,
        algorithm_name: str,
        model_name: str,
        per_band: Optional[Dict[str, Any]] = None,
        spatial_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        绘制各深度带簇在扰动特征空间 (δlnVs, δlnVp) 中的交会图。

        每个面板叠绘两条参考线：地幔热标定线 δlnVp = 0.5·δlnVs，以及点云
        实测主轴（总体最小二乘）。若簇中心沿主轴串成一列（R = σ∥/σ⊥ ≫ 1），
        说明聚类主要按扰动幅度切分，各簇缺乏独立的 Vp/Vs 差异；若簇在垂直
        主轴方向上分开，则簇携带成分或流体信息，具备物性域含义。

        对应 Hao et al. (2026, EPSL) 在 Vs–Vp 交会图上以斜率 1.73 参考线判别
        簇物性差异的做法，此处为扰动域下的等价诊断。

        Args:
            X: 形状 (N, n_features) 的标准化特征矩阵（内部反标准化到扰动域）
            labels: 形状 (N,) 的簇标签，负值为无效点
            metadata: 含 features / feature_display_names 的元数据
            output_dir: 图件输出目录
            algorithm_name: 算法名（用于文件名，如 'gmm'）
            model_name: 模型名（用于图题）
            per_band: 分带结果字典，含 point_mask / depth_range / n_clusters

        Returns:
            {深度带名: 诊断量字典}，诊断量见 crossplot_diagnostics
        """
        self.logger.info(f"    📊 绘制特征空间交会图 ({algorithm_name})...")
        ctx = self._prepare_crossplot_context(
            X, labels, metadata, per_band, spatial_indices=spatial_indices,
        )
        if ctx is None:
            self.logger.warning("      ⚠️ 未找到 vp/vs 特征列，跳过交会图")
            return {}

        self.fonts = ctx['F']
        F = ctx['F']
        X_plot = ctx['X_plot']
        labels = ctx['labels']
        ix, iy = ctx['ix'], ctx['iy']
        xlabel, ylabel = ctx['xlabel'], ctx['ylabel']
        ref_label, ref_slope = ctx['ref_label'], ctx['ref_slope']
        space_label = ctx['space_label']
        is_perturbation = ctx['is_perturbation']
        band_items = ctx['band_items']
        cp_cfg = ctx['cp_cfg']
        max_scatter = int(cp_cfg.get('max_scatter_points', 25_000))

        show_overview = bool(cp_cfg.get('overview_panel', True)) and len(band_items) > 1
        n_panels = len(band_items) + (1 if show_overview else 0)
        n_cols = min(5 if show_overview else 4, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5.6 * n_cols, 6.2 * n_rows), squeeze=False
        )
        panel_offset = 1 if show_overview else 0

        max_cid = int(labels.max()) if np.any(labels >= 0) else 0
        colors = generate_distinct_colors(
            max(max_cid + 1, self.config.visualization['n_colors'])
        )
        rng = np.random.default_rng(RANDOM_SEED)
        diagnostics: Dict[str, Dict[str, float]] = {}

        if show_overview:
            self._crossplot_overview_panel(
                axes[0, 0], X_plot, labels, band_items, ix, iy,
                ref_slope, ref_label, xlabel, ylabel,
                is_perturbation, max_scatter, rng, F,
            )

        for idx0, (bname, pmask, subtitle) in enumerate(band_items):
            idx = idx0 + panel_offset
            ax = axes[idx // n_cols, idx % n_cols]
            sel = pmask & (labels >= 0)
            xy = np.stack([X_plot[sel, ix], X_plot[sel, iy]], axis=1)
            band_labels = labels[sel]
            finite = np.all(np.isfinite(xy), axis=1)
            xy, band_labels = xy[finite], band_labels[finite]

            if len(xy) < 10:
                ax.axis('off')
                ax.set_title(f'{subtitle}\n(no data)', fontsize=F['title'])
                continue

            # 诊断量在全量体元上计算；下面的抽样只作用于散点绘制，
            # 因为把数百万个点画成散点会糊成色块且文件体积失控
            diag = crossplot_diagnostics(xy, band_labels)
            diag['n_points'] = float(len(xy))
            diagnostics[bname] = diag

            # 散点抽样（仅影响显示密度，不影响上面的诊断量）
            if len(xy) > max_scatter:
                pick = rng.choice(len(xy), max_scatter, replace=False)
            else:
                pick = np.arange(len(xy))
            pt_colors = np.array(
                [colors[int(c) % len(colors)] for c in band_labels[pick]]
            )
            ax.scatter(
                xy[pick, 0], xy[pick, 1], s=2.0, c=pt_colors,
                alpha=0.35, linewidths=0, rasterized=True,
            )

            # 簇中心（标号按带内顺序，与 2-3-6/2-3-7 的 C 编号一致）
            for cid in np.unique(band_labels):
                c = xy[band_labels == cid].mean(axis=0)
                ax.scatter(
                    c[0], c[1], s=180, facecolor=colors[int(cid) % len(colors)],
                    edgecolor='black', linewidth=1.4, zorder=5,
                )
                ax.annotate(
                    f'{int(cid)}', c, ha='center', va='center',
                    fontsize=F['annotation'] - 4, fontweight='bold', zorder=6,
                )

            # 坐标范围：扰动域以 0 为中心对称，原始域按数据实际范围
            if is_perturbation:
                lim = float(np.percentile(np.abs(xy), 99.5))
                xlim = ylim = (-lim, lim)
                t = np.array([-lim, lim])
                ax.axhline(0, color='0.6', lw=0.6, zorder=1)
                ax.axvline(0, color='0.6', lw=0.6, zorder=1)
            else:
                lo = np.percentile(xy, 0.5, axis=0)
                hi = np.percentile(xy, 99.5, axis=0)
                pad = 0.05 * (hi - lo)
                xlim = (float(lo[0] - pad[0]), float(hi[0] + pad[0]))
                ylim = (float(lo[1] - pad[1]), float(hi[1] + pad[1]))
                t = np.array([0.0, float(hi[0] + pad[0])])

            # 参考线：过原点的标定线 + 点云实测主轴
            ax.plot(
                t, ref_slope * t, 'k--', lw=1.8, zorder=4,
                label=f'{ref_label} ({ref_slope:.2f})',
            )
            pc1, _, _ = principal_axis(xy)
            mean_xy = xy.mean(axis=0)
            span = np.array([-1.0, 1.0]) * float(np.ptp(xlim))
            ax.plot(
                mean_xy[0] + span * pc1[0], mean_xy[1] + span * pc1[1],
                color='crimson', lw=1.8, zorder=4,
                label=f"Observed axis ({diag['pc1_slope']:.2f})",
            )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if is_perturbation:
                ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=F['tick'])
            ax.set_title(
                f"{subtitle}\n"
                f"r = {diag['corr']:.2f},  "
                f"R = $\\sigma_\\parallel/\\sigma_\\perp$ = {diag['ratio_R']:.1f}",
                fontsize=F['title'] - 3, fontweight='bold',
            )
            ax.set_xlabel(xlabel, fontsize=F['label'])
            if idx % n_cols == 0:
                ax.set_ylabel(ylabel, fontsize=F['label'])
            if idx == panel_offset:
                ax.legend(fontsize=F['legend'] - 3, loc='upper left',
                          framealpha=0.9)

        # 关闭多余空面板
        for idx in range(n_panels, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].axis('off')

        fig.suptitle(
            f'{model_name} — {algorithm_name.upper()} Cluster Distribution '
            f'in the {space_label}',
            fontsize=F['suptitle'], fontweight='bold',
        )
        fig.tight_layout(rect=[0, 0.01, 1, 0.95], w_pad=0.8, h_pad=1.0)

        stem = f'2-3-12_{algorithm_name}_cluster_crossplot'
        for fmt in self.config.visualization.get('save_formats', ['jpg']):
            fig.savefig(
                output_dir / f'{stem}.{fmt}',
                dpi=self.config.visualization['dpi'], bbox_inches='tight',
            )
        plt.close(fig)

        for bname, d in diagnostics.items():
            self.logger.info(
                f"      {bname:<16} K={d['n_clusters']:>2}  "
                f"r={d['corr']:.3f}  axis={d['pc1_slope']:.3f}  "
                f"cloud_aniso={d['cloud_anisotropy']:.2f}  R={d['ratio_R']:.2f}"
            )
        self.logger.info(f"      ✅ 保存: {stem}")
        return diagnostics

    def plot_cluster_crossplot_3d(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Path,
        algorithm_name: str,
        model_name: str,
        per_band: Optional[Dict[str, Any]] = None,
        spatial_indices: Optional[np.ndarray] = None,
    ) -> None:
        """
        独立绘制 (δlnVs, δlnVp, depth) 三维特征空间交会图（2-3-13）。

        与 2-3-12 的 2D 分带面板互补：展示四带在扰动–深度体积中的分层占位。
        """
        self.logger.info(f"    📊 绘制三维特征空间交会图 ({algorithm_name})...")
        ctx = self._prepare_crossplot_context(
            X, labels, metadata, per_band, spatial_indices=spatial_indices,
        )
        if ctx is None:
            self.logger.warning("      ⚠️ 未找到 vp/vs 特征列，跳过三维交会图")
            return

        if spatial_indices is None:
            self.logger.warning("      ⚠️ 缺少 spatial_indices，跳过三维交会图")
            return

        depths_arr = np.asarray(metadata.get('depths', []), dtype=float)
        if len(depths_arr) == 0 or len(spatial_indices) != len(ctx['labels']):
            self.logger.warning("      ⚠️ depth 坐标无效，跳过三维交会图")
            return

        point_depths = depths_arr[np.asarray(spatial_indices[:, 2], dtype=int)]
        self.fonts = ctx['F']
        F = ctx['F']
        labels = ctx['labels']
        cp_cfg = ctx['cp_cfg']

        max_cid = int(labels.max()) if np.any(labels >= 0) else 0
        colors = generate_distinct_colors(
            max(max_cid + 1, self.config.visualization['n_colors'])
        )

        figsize = self.config.visualization.get('figsize', {}).get('3d', (12, 10))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        n_plotted = self._crossplot_3d_panel(
            ax, ctx['X_plot'], labels, point_depths,
            ctx['ix'], ctx['iy'], colors, F,
            ctx['xlabel'], ctx['ylabel'],
            ctx['is_perturbation'], cp_cfg,
        )
        if n_plotted == 0:
            plt.close(fig)
            self.logger.warning('      ⚠️ 三维交会图无有效体元，已跳过保存')
            return
        self.logger.info(f'      三维交会散点: {n_plotted:,} 体元')

        title_3d = (
            r'3D feature space  ($\delta\ln V_S$, $\delta\ln V_P$, depth)'
            if ctx['is_perturbation'] else
            r'3D feature space  ($V_S$, $V_P$, depth)'
        )
        fig.suptitle(
            f'{model_name} — {algorithm_name.upper()} {title_3d}',
            fontsize=F['suptitle'], fontweight='bold', y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        stem = f'2-3-13_{algorithm_name}_cluster_crossplot_3d'
        for fmt in self.config.visualization.get('save_formats', ['jpg']):
            fig.savefig(
                output_dir / f'{stem}.{fmt}',
                dpi=self.config.visualization['dpi'], bbox_inches='tight',
            )
        plt.close(fig)
        self.logger.info(f"      ✅ 保存: {stem}")

