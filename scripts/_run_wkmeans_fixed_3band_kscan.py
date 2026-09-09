"""
W-k-means fixed_3band K 扫描（肘部法 + 轮廓系数诊断）

上界：shallow K≤20，transition_zone K≤10，lower_mantle K≤10
输出：wkmeans_{dln|raw}_fixed_3band/

用法:
  python scripts/_run_wkmeans_fixed_3band_kscan.py --both
  python scripts/_run_wkmeans_fixed_3band_kscan.py --raw --models 2024_EARA2024
  python scripts/_run_wkmeans_fixed_3band_kscan.py --kscan-only --both
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / '2_Model_space_analysis'))

spec = importlib.util.spec_from_file_location(
    'mc', project_root / '2_Model_space_analysis' / '2_3_Model_clustering.py'
)
mc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mc)

K_MAX = {'shallow': 20, 'transition_zone': 10, 'lower_mantle': 10}


def build_config(raw: bool, kscan_only: bool) -> mc.ClusteringConfig:
    config = mc.ClusteringConfig()
    config.algorithm = 'wkmeans'
    config.preprocessing['perturbation']['enabled'] = not raw
    config.depth_stratified['enabled'] = True
    config.depth_stratified['scheme'] = 'fixed_3band'

    for band in config.depth_stratified['schemes']['fixed_3band']['bands']:
        name = band['name']
        if name in K_MAX:
            band['max_clusters'] = K_MAX[name]
            band['min_clusters'] = 2
            band.pop('fixed_k', None)

    config.wkmeans['k_selection']['method'] = 'elbow'
    config.wkmeans['k_selection']['use_band_k_range'] = True
    config.wkmeans['k_selection']['respect_fixed_k'] = False
    config.wkmeans['fit_max_points'] = None
    config.wkmeans['k_selection']['scan_max_points'] = None
    config.kmeans_comparison['enabled'] = False

    if kscan_only:
        # 仅 K 扫描诊断 + 交会图（轻量）
        pt = config.visualization['plot_types']
        for key in pt:
            pt[key] = False
        pt['cluster_crossplot'] = True
        config.visualization['enabled'] = True
    else:
        config.visualization['enabled'] = True

    return config


def run_domain(raw: bool, models: List[str], kscan_only: bool) -> None:
    domain = 'raw' if raw else 'dln'
    config = build_config(raw=raw, kscan_only=kscan_only)
    print(f"\n{'=' * 70}")
    print(f"W-k-means fixed_3band K-scan  domain={domain}  kscan_only={kscan_only}")
    print(f"K 上界: shallow≤{K_MAX['shallow']}, TZ≤{K_MAX['transition_zone']}, "
          f"LM≤{K_MAX['lower_mantle']}")
    print(f"models: {models}")
    print('=' * 70)

    pipeline = mc.SmartClusteringPipeline(config)
    results = pipeline.run_analysis(model_names=models, features_to_use=['vp', 'vs'])

    print(f"\n--- 结果 ({domain}) ---")
    for name, result in results.items():
        res = result['cluster_results']
        ksel = res.get('k_selection') or {}
        band_k = res['metrics'].get('band_optimal_k', {})
        print(f"{name}:")
        print(f"  选定 K: {band_k}  (全局 n_clusters={res['n_clusters']})")
        for bname, diag in ksel.items():
            if isinstance(diag, dict):
                print(
                    f"    [{bname}] elbow={diag.get('k_elbow')}, "
                    f"sil_max={diag.get('k_silhouette')}, "
                    f"selected={diag.get('optimal_n')}, "
                    f"sil@K*={diag.get('silhouette_at_optimal', float('nan')):.3f}"
                )
    print(f"figures: {pipeline._scheme_figure_root()}")
    print(f"results: {pipeline._scheme_output_root()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='W-k-means fixed_3band K 扫描 (上界 20/10/10)'
    )
    parser.add_argument('--raw', action='store_true', help='原始 Vp/Vs 域')
    parser.add_argument('--both', action='store_true', help='dln 与 raw 都跑')
    parser.add_argument(
        '--kscan-only', action='store_true',
        help='关闭深度切片/剖面等重图，仅 K 选择图 + 交会图',
    )
    parser.add_argument('--models', nargs='+', default=None)
    args = parser.parse_args()

    models = args.models or mc.ClusteringConfig().data['target_models']
    domains = [True, False] if args.both else [args.raw]

    for raw in domains:
        run_domain(raw=raw, models=models, kscan_only=args.kscan_only)

    print('\n✅ 全部完成')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
