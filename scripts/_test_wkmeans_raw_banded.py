#!/usr/bin/env python3
"""
分带原始速度值 W-k-means 聚类测试
==================================

功能描述:
- 在 moho_4band 的每个深度带内，用原始 Vp/Vs 值（不转扰动域）做 W-k-means
- 对照 Hao et al. (2026, EPSL) 的五维方案，同时跑含坐标与不含坐标两组
- 量化簇的深度分层程度与横向信息量，判断原始值在带内是否可用

科学原理:
- 全域 (0-1000 km) 一次性聚类时，绝对速度的压力/相变深度趋势（约 5→11 km/s）
  比横向非均匀性（百分之几）大一到两个量级，簇必然退化为水平壳层。
  分带后带内深度跨度收窄，该趋势被大幅削弱，原始值才可能携带横向信息。
- 判据一：NMI(簇标签, 深度层号)。接近 1 表示簇即深度壳层。
- 判据二：η²(横向) 与簇中心在 (Vs, Vp) 平面的分布，衡量横向可分性。

作者: EASTASIA-FWI Team
日期: 2026-09-07
版本: v1.0
"""

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / '2_Model_space_analysis'))

if '2_3_Model_clustering' in sys.modules:
    del sys.modules['2_3_Model_clustering']
mod = importlib.import_module('2_3_Model_clustering')

MODELS: List[str] = ['2022_SinoScope1.0', '2024_EARA2024', '2024_FWEA23']
BETA = 6.5
FIT_MAX_POINTS = 200_000
RANDOM_SEED = 42
SECTION_LAT = 35.0
SLICE_DEPTHS = [30.0, 150.0, 500.0, 800.0]

OUT_DIR = project_root / 'results' / 'model_clustering' / 'wkmeans_raw_banded'


def eta_squared(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """逐特征方差解释率 η² = 1 - SS_within / SS_total"""
    ss_total = ((X - X.mean(axis=0)) ** 2).sum(axis=0)
    ss_within = np.zeros(X.shape[1], dtype=float)
    for l in np.unique(labels):
        m = labels == l
        ss_within += ((X[m] - X[m].mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_within / np.maximum(ss_total, 1e-300)


def run_model(
    pipeline: Any, model_name: str, use_coordinates: bool
) -> Dict[str, Any]:
    """
    在每个深度带内分别做原始值 W-k-means

    Args:
        pipeline: SmartClusteringPipeline 实例
        model_name: 模型名
        use_coordinates: 是否并入经纬度与深度

    Returns:
        含 labels_3d / metadata / 各带诊断量的字典
    """
    cfg = pipeline.config
    cfg.preprocessing['perturbation']['enabled'] = False   # 原始速度值
    cfg.preprocessing['coordinates']['enabled'] = use_coordinates
    cfg.depth_stratified['enabled'] = True
    cfg.depth_stratified['scheme'] = 'moho_4band'

    data_cube, metadata = pipeline.loader.load_3d_model(model_name, ['vp', 'vs'])
    X, spatial_indices, prep = pipeline.processor.prepare_for_clustering(
        data_cube, metadata
    )
    feature_names = list(prep['feature_names'])

    depths = np.asarray(metadata['depths'], dtype=float)
    point_depths = depths[spatial_indices[:, 2]]
    depth_idx = spatial_indices[:, 2]

    opt = pipeline.gmm_optimizer
    opt.concordance_eval = pipeline.concordance_eval
    moho_z = opt._moho_depth_per_point(spatial_indices, metadata)
    bands = cfg.depth_stratified['schemes']['moho_4band']['bands']
    model_z_max = float(depths.max())

    rng = np.random.default_rng(RANDOM_SEED)
    global_labels = np.full(len(X), -1, dtype=np.int32)
    offset = 0
    records: List[Dict[str, Any]] = []

    for bi, band in enumerate(bands):
        name = band['name']
        k = int(band.get('fixed_k') or band['max_clusters'])
        mask, z_ref, desc = opt._build_band_mask(
            band, point_depths, moho_z, bi == len(bands) - 1, model_z_max
        )
        n_band = int(np.count_nonzero(mask))
        if n_band < k * 10:
            continue

        Xb = X[mask]
        idx = np.where(mask)[0]
        fit_idx = (
            rng.choice(len(Xb), FIT_MAX_POINTS, replace=False)
            if len(Xb) > FIT_MAX_POINTS else np.arange(len(Xb))
        )
        wk = mod.WKMeans(
            n_clusters=k, beta=BETA, n_init=3, max_iter=100,
            random_state=RANDOM_SEED,
        ).fit(Xb[fit_idx])
        lab_b = wk.predict(Xb)
        global_labels[idx] = lab_b + offset
        offset += k

        e2 = eta_squared(Xb[fit_idx], wk.labels_)
        # 簇标签与深度层号的归一化互信息：接近 1 即深度壳层
        nmi_depth = float(normalized_mutual_info_score(
            depth_idx[mask][fit_idx], lab_b[fit_idx]
        ))

        rec: Dict[str, Any] = {
            'model': model_name, 'band': name, 'K': k,
            'n_points': n_band,
            'depth_range': f"{z_ref[0]:.0f}-{z_ref[1]:.0f}",
            'nmi_depth': nmi_depth,
            'converged': wk.converged_,
        }
        for fname, w, e in zip(feature_names, wk.feature_weights_, e2):
            rec[f'w_{fname}'] = float(w)
            rec[f'eta2_{fname}'] = float(e)
        records.append(rec)

        print(f"    {name:<16} K={k:>2} n={n_band:>9,} "
              f"NMI(depth)={nmi_depth:.3f}  "
              + "  ".join(f"η²({f})={e:.3f}"
                          for f, e in zip(feature_names, e2)), flush=True)

    labels_3d = np.full(tuple(data_cube.shape[:3]), -1, dtype=np.int16)
    labels_3d[
        spatial_indices[:, 0], spatial_indices[:, 1], spatial_indices[:, 2]
    ] = global_labels

    return {
        'labels_3d': labels_3d, 'metadata': metadata,
        'records': records, 'n_total_clusters': offset,
    }


def plot_model(res: Dict[str, Any], model_name: str, tag: str, title: str) -> None:
    """绘制分带聚类的空间形态"""
    meta = res['metadata']
    lats = np.asarray(meta['lats'], dtype=float)
    lons = np.asarray(meta['lons'], dtype=float)
    depths = np.asarray(meta['depths'], dtype=float)
    k_tot = res['n_total_clusters']

    lab = res['labels_3d'].astype(float)
    lab[res['labels_3d'] < 0] = np.nan
    cmap = plt.get_cmap('tab20', max(k_tot, 2))

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, len(SLICE_DEPTHS), hspace=0.3, wspace=0.25)

    for i, z in enumerate(SLICE_DEPTHS):
        iz = int(np.argmin(np.abs(depths - z)))
        ax = fig.add_subplot(gs[0, i])
        ax.pcolormesh(lons, lats, lab[:, :, iz], cmap=cmap,
                      vmin=-0.5, vmax=k_tot - 0.5, shading='auto')
        ax.set_title(f'Depth = {depths[iz]:.0f} km', fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('Longitude (°E)', fontsize=10)
        if i == 0:
            ax.set_ylabel('Latitude (°N)', fontsize=10)
        ax.set_aspect('equal')

    ilat = int(np.argmin(np.abs(lats - SECTION_LAT)))
    ax = fig.add_subplot(gs[1, :])
    im = ax.pcolormesh(lons, depths, lab[ilat, :, :].T, cmap=cmap,
                       vmin=-0.5, vmax=k_tot - 0.5, shading='auto')
    ax.invert_yaxis()
    ax.set_xlabel('Longitude (°E)', fontsize=11)
    ax.set_ylabel('Depth (km)', fontsize=11)
    ax.set_title(f'Longitude-depth section at {lats[ilat]:.0f}°N',
                 fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, pad=0.01).set_label('Cluster', fontsize=10)

    fig.suptitle(f'{model_name} — {title}', fontsize=15, fontweight='bold')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f'2-3-14_wkmeans_raw_{tag}_{model_name}.jpg',
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    """主函数"""
    print("🎯 分带原始速度值 W-k-means 聚类测试")
    print("=" * 88)
    print(f"特征为原始 Vp/Vs（不转扰动域），按 moho_4band 分带，K=10/10/5/5，"
          f"beta={BETA}")
    print("NMI(depth) 接近 1 表示簇退化为深度壳层\n")

    config = mod.ClusteringConfig()
    pipeline = mod.SmartClusteringPipeline(config)

    settings = [
        ('vpvs', False, 'W-k-means on raw (Vp, Vs), per depth band'),
        ('5d', True, 'W-k-means on raw (lon, lat, depth, Vp, Vs), per depth band'),
    ]

    all_records: List[Dict[str, Any]] = []
    for model_name in MODELS:
        print(f"\n=== {model_name} ===", flush=True)
        for tag, use_coord, title in settings:
            print(f"  [{tag}] {title}", flush=True)
            res = run_model(pipeline, model_name, use_coord)
            plot_model(res, model_name, tag, title)
            for r in res['records']:
                r['setting'] = tag
                all_records.append(r)

    df = pd.DataFrame(all_records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / 'wkmeans_raw_banded.csv', index=False,
              float_format='%.4f')

    print("\n" + "=" * 88)
    print("汇总：NMI(簇标签, 深度层号)  —— 越接近 1 越是纯深度壳层")
    piv = df.pivot_table(index=['model', 'band'], columns='setting',
                         values='nmi_depth')
    print(piv.to_string(float_format=lambda v: f'{v:.3f}'))

    print("\n汇总：原始 Vp/Vs 两维设置下的 η²")
    sub = df[df['setting'] == 'vpvs'][
        ['model', 'band', 'eta2_vp', 'eta2_vs']
    ]
    print(sub.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    print(f"\n✅ 图件与数据: {OUT_DIR}")
    print("=" * 88)


if __name__ == '__main__':
    main()
