"""
GMM fixed_3band 测试：浅部 K=20 / 过渡带 K=10 / 下地幔 K=10

用法:
  python scripts/_run_gmm_fixed_3band_k20.py
  python scripts/_run_gmm_fixed_3band_k20.py --models 2024_EARA2024
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / '2_Model_space_analysis'))

spec = importlib.util.spec_from_file_location(
    'mc', project_root / '2_Model_space_analysis' / '2_3_Model_clustering.py'
)
mc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mc)


def build_config() -> mc.ClusteringConfig:
    config = mc.ClusteringConfig()
    config.algorithm = 'gmm'
    config.preprocessing['perturbation']['enabled'] = True
    config.depth_stratified['enabled'] = True
    config.depth_stratified['scheme'] = 'fixed_3band'

    k_map = {'shallow': 20, 'transition_zone': 10, 'lower_mantle': 10}
    for band in config.depth_stratified['schemes']['fixed_3band']['bands']:
        name = band['name']
        if name in k_map:
            band['fixed_k'] = k_map[name]
            band['max_clusters'] = k_map[name]

    config.auto_gmm['bic_diagnostic_when_fixed'] = False
    config.auto_gmm['k_robustness']['enabled'] = False
    config.kmeans_comparison['enabled'] = False
    config.visualization['enabled'] = True
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description='GMM fixed_3band K=20/10/10')
    parser.add_argument('--models', nargs='+', default=None)
    args = parser.parse_args()

    models = args.models or mc.ClusteringConfig().data['target_models']
    config = build_config()

    print('fixed_3band GMM: shallow K=20, transition_zone K=10, lower_mantle K=10')
    print('models:', models)

    pipeline = mc.SmartClusteringPipeline(config)
    results = pipeline.run_analysis(model_names=models, features_to_use=['vp', 'vs'])

    print('\n' + '=' * 60)
    for name, result in results.items():
        res = result['cluster_results']
        print(f'{name}: n_clusters={res["n_clusters"]}, band_K={res["metrics"].get("band_optimal_k")}')
    print('figures:', pipeline._scheme_figure_root())
    print('results:', pipeline._scheme_output_root())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
