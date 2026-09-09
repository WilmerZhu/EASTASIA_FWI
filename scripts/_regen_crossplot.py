"""
EASTASIA-FWI 交会图重绘工具
============================

功能描述:
- 从已缓存的聚类标签重绘 2-3-12 特征空间交会图，不重跑聚类
- 支持 GMM 与 W-k-means 两种算法

用途:
- 交会图的绘制逻辑（参考线、总览面板、坐标范围）调整后，无需重复数小时的
  聚类计算即可刷新图件

作者: EASTASIA-FWI Team
日期: 2026-09-07
版本: v1.1
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / '2_Model_space_analysis'))
sys.path.insert(0, str(project_root / '5_Visualization'))

import importlib

_clu = importlib.import_module('2_3_Model_clustering')


def _band_optimizer(pipeline: Any) -> Any:
    """按当前算法返回带分带方法的优化器实例"""
    if pipeline.config.algorithm == 'gmm':
        return pipeline.gmm_optimizer
    return pipeline.wkmeans_optimizer


def rebuild_per_band(
    pipeline: Any,
    X: np.ndarray,
    spatial_indices: np.ndarray,
    metadata: Dict[str, Any],
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    用与聚类时相同的分带规则重建 per_band 结构。

    交会图只需要 point_mask / depth_range / n_clusters 三个字段，故不必恢复
    模型对象与指标，按带掩膜统计簇数即可。

    Args:
        pipeline: 已初始化的聚类管道（提供分带配置与 Moho 面）
        X: 特征矩阵
        spatial_indices: 空间索引
        metadata: 模型元数据
        labels: 缓存的全局簇标签

    Returns:
        per_band 字典
    """
    opt = _band_optimizer(pipeline)
    if opt.concordance_eval is None:
        opt.concordance_eval = pipeline.concordance_eval
    depths = np.asarray(metadata['depths'], dtype=float)
    point_depths = depths[spatial_indices[:, 2]]
    model_z_max = float(np.nanmax(depths))

    _scheme, scheme_cfg = opt._active_scheme()
    bands = scheme_cfg['bands']
    needs_moho = any(
        b.get('mask') in ('above_moho', 'moho_to_depth') for b in bands
    )
    moho_z = (
        opt._moho_depth_per_point(spatial_indices, metadata)
        if needs_moho else None
    )

    per_band: Dict[str, Any] = {}
    for i_band, band in enumerate(bands):
        mask, z_ref, desc = opt._build_band_mask(
            band, point_depths, moho_z, i_band == len(bands) - 1, model_z_max
        )
        if not np.any(mask):
            continue
        k = int(len(np.unique(labels[mask & (labels >= 0)])))
        per_band[band['name']] = {
            'point_mask': mask,
            'depth_range': [float(z_ref[0]), float(z_ref[1])],
            'description': desc,
            'n_clusters': k,
        }
    return per_band


def regenerate(
    algorithm: str,
    raw: bool,
    models: List[str],
) -> None:
    """
    重绘指定算法与特征域下所有模型的交会图

    Args:
        algorithm: 'gmm' 或 'wkmeans'
        raw: True 用原始速度值，False 用 δlnV 扰动域
        models: 模型名列表
    """
    config = _clu.ClusteringConfig()
    config.algorithm = algorithm
    config.preprocessing['perturbation']['enabled'] = not raw
    config.depth_stratified['scheme'] = 'moho_4band'

    pipeline = _clu.SmartClusteringPipeline(config)
    out_root = pipeline._scheme_output_root()
    fig_root = pipeline._scheme_figure_root()
    algo = pipeline._algo_name()

    for model_name in models:
        label_file = out_root / model_name / f'{algo}_labels.npz'
        if not label_file.exists():
            print(f"⏭️  跳过 {model_name}：无缓存标签 {label_file}")
            continue

        print(f"\n🔬 {model_name}  ({algo}, {'raw' if raw else 'dln'})")
        data_cube, metadata = pipeline.loader.load_3d_model(
            model_name, config.data['default_features']
        )
        X, spatial_indices, prep_info = pipeline.processor.prepare_for_clustering(
            data_cube, metadata
        )
        metadata = dict(metadata)
        metadata['feature_space'] = prep_info.get('feature_space', 'absolute')
        if prep_info.get('feature_names'):
            metadata['feature_display_names'] = prep_info['feature_names']

        label_npz = np.load(label_file)
        labels = label_npz['labels_1d']
        if len(labels) != len(X):
            print(f"  ⚠️ 标签点数 {len(labels)} 与特征点数 {len(X)} 不符，跳过")
            continue

        per_band = rebuild_per_band(
            pipeline, X, spatial_indices, metadata, labels
        )
        viz_dir = fig_root / model_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        pert_cube = getattr(pipeline.processor, 'last_feature_cube', None)

        if 'labels_3d' in label_npz:
            pipeline.basic_visualizer.plot_vertical_sections(
                label_npz['labels_3d'],
                metadata,
                viz_dir,
                algo,
                model_name,
                concordance_eval=pipeline.concordance_eval,
                pert_cube=pert_cube,
            )
            print(f"  ✅ 保存: {fig_root / model_name / f'2-3-5_{algo}_vertical_sections.jpg'}")

        diagnostics = pipeline.basic_visualizer.plot_cluster_crossplot(
            X, labels, metadata, viz_dir, algo, model_name,
            per_band=per_band,
            spatial_indices=spatial_indices,
        )
        if diagnostics:
            pipeline._save_crossplot_diagnostics(diagnostics, model_name)
            print(f"  ✅ 保存: {fig_root / model_name / f'2-3-12_{algo}_cluster_crossplot.jpg'}")
        pipeline.basic_visualizer.plot_cluster_crossplot_3d(
            X, labels, metadata, viz_dir, algo, model_name,
            per_band=per_band,
            spatial_indices=spatial_indices,
        )
        print(f"  ✅ 保存: {fig_root / model_name / f'2-3-13_{algo}_cluster_crossplot_3d.jpg'}")


def main() -> int:
    """主函数"""
    import argparse

    p = argparse.ArgumentParser(description='重绘聚类特征空间交会图（GMM / W-k-means）')
    p.add_argument(
        '--algorithm', choices=['gmm', 'wkmeans'], default='wkmeans',
        help='聚类算法（默认 wkmeans）',
    )
    p.add_argument('--raw', action='store_true', help='原始速度值特征域')
    p.add_argument('--both', action='store_true', help='原始域与扰动域都重绘')
    p.add_argument('--models', nargs='+', default=None)
    args = p.parse_args()

    models = args.models or _clu.ClusteringConfig().data['target_models']
    domains = [True, False] if args.both else [args.raw]
    for raw in domains:
        regenerate(args.algorithm, raw, models)

    print("\n✅ 重绘完成")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
