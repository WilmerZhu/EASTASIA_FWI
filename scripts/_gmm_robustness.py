"""
EASTASIA-FWI GMM 稳健性工具
============================

功能:
1. GMM vs Hierarchical 逐带 ARI/NMI（需两套缓存标签）
2. GMM 簇心 Ward 谱系 dendrogram（含启发式 facies 短标签）

用法:
  # 对比 GMM 与 Hierarchical（需 results/.../gmm_labels.npz 与 hierarchical_labels.npz）
  python scripts/_gmm_robustness.py --compare-ari --models 2024_EARA2024

  # 仅从 GMM 标签重绘簇心谱系图
  python scripts/_gmm_robustness.py --centroid-phylogeny --models 2024_EARA2024

  # 两者都做
  python scripts/_gmm_robustness.py --compare-ari --centroid-phylogeny

作者: EASTASIA-FWI Team
日期: 2026-09-08
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / '2_Model_space_analysis'))
sys.path.insert(0, str(project_root / '5_Visualization'))

_clu = importlib.import_module('2_3_Model_clustering')

_regen_path = project_root / 'scripts' / '_regen_crossplot.py'
_spec = importlib.util.spec_from_file_location('_regen_crossplot', _regen_path)
_regen = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_regen)


def _gmm_scheme_root(base: Path) -> Path:
    return base / 'model_clustering' / 'moho_4band'


def _hier_scheme_root(base: Path) -> Path:
    return base / 'model_clustering' / 'hierarchical_dln_moho_4band'


def _load_labels(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    data = np.load(path)
    return np.asarray(data['labels_1d'])


def run_compare_ari(
    pipeline: Any,
    model_name: str,
    results_base: Path,
) -> Optional[Dict[str, Any]]:
    """GMM vs Hierarchical 逐带 ARI/NMI"""
    gmm_npz = _gmm_scheme_root(results_base) / model_name / 'gmm_labels.npz'
    hier_npz = _hier_scheme_root(results_base) / model_name / 'hierarchical_labels.npz'

    labels_gmm = _load_labels(gmm_npz)
    labels_hier = _load_labels(hier_npz)
    if labels_gmm is None:
        print(f"  ⏭️  缺 GMM 标签: {gmm_npz}")
        return None
    if labels_hier is None:
        print(f"  ⏭️  缺 Hierarchical 标签: {hier_npz}")
        return None

    config = _clu.ClusteringConfig()
    config.algorithm = 'gmm'
    config.preprocessing['perturbation']['enabled'] = True
    config.depth_stratified['scheme'] = 'moho_4band'
    pipe = _clu.SmartClusteringPipeline(config)

    data_cube, metadata = pipe.loader.load_3d_model(
        model_name, config.data['default_features']
    )
    X, spatial_indices, prep_info = pipe.processor.prepare_for_clustering(
        data_cube, metadata
    )
    if len(labels_gmm) != len(X) or len(labels_hier) != len(X):
        print(
            f"  ⚠️ 标签点数与特征不符 "
            f"(GMM={len(labels_gmm)}, Hier={len(labels_hier)}, X={len(X)})"
        )
        return None

    per_band = _regen.rebuild_per_band(
        pipe, X, spatial_indices, metadata, labels_gmm
    )
    comparison = _clu.compute_cross_algorithm_ari(
        labels_gmm, labels_hier, per_band
    )
    comparison['model'] = model_name
    comparison['algorithms'] = ['gmm', 'hierarchical']

    out_dir = pipe._scheme_output_root() / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'cross_algorithm_ari.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    overall = comparison.get('overall', {})
    print(
        f"  ✅ ARI 全深度: {overall.get('ari', float('nan')):.4f}, "
        f"NMI: {overall.get('nmi', float('nan')):.4f}"
    )
    for bname, row in comparison.get('bands', {}).items():
        print(
            f"     [{bname}] ARI={row['ari']:.4f}, "
            f"NMI={row['nmi']:.4f}, n={row['n_points']:,}"
        )
    print(f"  💾 {out_path}")
    return comparison


def run_centroid_phylogeny(
    pipeline: Any,
    model_name: str,
    results_base: Path,
) -> None:
    """从 GMM 缓存标签重绘簇心谱系 dendrogram"""
    gmm_npz = _gmm_scheme_root(results_base) / model_name / 'gmm_labels.npz'
    labels = _load_labels(gmm_npz)
    if labels is None:
        print(f"  ⏭️  缺 GMM 标签: {gmm_npz}")
        return

    config = _clu.ClusteringConfig()
    config.algorithm = 'gmm'
    config.preprocessing['perturbation']['enabled'] = True
    config.depth_stratified['scheme'] = 'moho_4band'
    pipe = _clu.SmartClusteringPipeline(config)

    data_cube, metadata = pipe.loader.load_3d_model(
        model_name, config.data['default_features']
    )
    X, spatial_indices, prep_info = pipe.processor.prepare_for_clustering(
        data_cube, metadata
    )
    if prep_info.get('feature_names'):
        metadata = dict(metadata)
        metadata['feature_display_names'] = prep_info['feature_names']
        metadata['feature_space'] = prep_info.get('feature_space', '')

    if len(labels) != len(X):
        print(f"  ⚠️ 标签点数 {len(labels)} 与特征 {len(X)} 不符")
        return

    per_band = _regen.rebuild_per_band(
        pipe, X, spatial_indices, metadata, labels
    )
    viz_dir = pipe._scheme_figure_root() / model_name
    viz_dir.mkdir(parents=True, exist_ok=True)

    phylo = pipe.enhanced_visualizer.plot_centroid_phylogeny(
        X, labels, metadata, per_band, viz_dir, model_name, 'gmm'
    )
    if phylo:
        pipe._save_centroid_phylogeny(phylo, model_name)
        print(f"  ✅ 图件: {viz_dir / '2-3-17_gmm_centroid_phylogeny.jpg'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='GMM 稳健性：ARI 对比 + 簇心谱系 dendrogram'
    )
    parser.add_argument(
        '--compare-ari', action='store_true',
        help='GMM vs Hierarchical 逐带 ARI/NMI',
    )
    parser.add_argument(
        '--centroid-phylogeny', action='store_true',
        help='重绘 GMM 簇心 Ward 谱系图',
    )
    parser.add_argument('--models', nargs='+', default=None)
    args = parser.parse_args()

    if not args.compare_ari and not args.centroid_phylogeny:
        parser.error('请指定 --compare-ari 和/或 --centroid-phylogeny')

    models: List[str] = args.models or _clu.ClusteringConfig().data['target_models']
    base_config = _clu.BaseConfig()
    results_base = base_config.dirs['results']

    for model_name in models:
        print(f"\n🔬 {model_name}")
        if args.compare_ari:
            run_compare_ari(None, model_name, results_base)
        if args.centroid_phylogeny:
            run_centroid_phylogeny(None, model_name, results_base)

    print("\n✅ 完成")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
