#!/usr/bin/env python3
"""
重绘受「Facies → Cluster」标签修改影响的聚类图件。

对应论文图号（paper/figures/）：
    图6  = 2-3-4_gmm_depth_slices      (EARA2024)   ← suptitle 含 facies
    图7  = 2-3-5_gmm_vertical_sections (EARA2024)   ← 子图标题与 suptitle 含 facies
    fig12/fig13 = 其余两模型的 2-3-4（备选图）
    fig14 = 2-3-11_slab_volume_3d      (EARA2024)   ← suptitle 含 Facies

直接读取 results/model_clustering/moho_4band/<model>/gmm_labels.npz 中已缓存的
聚类标签，不重跑 GMM，因此结果与正文表3、表4完全一致。
"""
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / '2_Model_space_analysis'))

if '2_3_Model_clustering' in sys.modules:
    del sys.modules['2_3_Model_clustering']
mod = importlib.import_module('2_3_Model_clustering')

MODELS = ['2022_SinoScope1.0', '2024_EARA2024', '2024_FWEA23']

config = mod.ClusteringConfig()
config.depth_stratified['scheme'] = 'moho_4band'
config.preprocessing['perturbation']['enabled'] = True
pipeline = mod.SmartClusteringPipeline(config)

for model_name in MODELS:
    print(f"=== {model_name} ===", flush=True)
    out = pipeline._scheme_output_root() / model_name
    viz = out / 'visualizations'

    data_cube, metadata = pipeline.loader.load_3d_model(model_name, ['vp', 'vs'])
    _, _, prep = pipeline.processor.prepare_for_clustering(data_cube, metadata)
    metadata = {**metadata, **prep}
    labels_3d = np.load(out / 'gmm_labels.npz')['labels_3d']
    pert_cube = pipeline.processor.last_feature_cube
    print(f"  labels {labels_3d.shape} | 簇数 {len(np.unique(labels_3d[labels_3d >= 0]))}",
          flush=True)

    pipeline.basic_visualizer.set_processor(pipeline.processor)

    # 图6 / fig12 / fig13：深度切片（第1排地质底图，第2–3排叠 Slab2）
    pipeline.basic_visualizer.plot_depth_slices(
        labels_3d, metadata, viz, 'gmm', model_name,
        concordance_eval=pipeline.concordance_eval,
    )
    print("  ✓ 2-3-4_gmm_depth_slices", flush=True)

    # 图7：经度—深度剖面
    pipeline.basic_visualizer.plot_vertical_sections(
        labels_3d, metadata, viz, 'gmm', model_name,
        concordance_eval=pipeline.concordance_eval,
        pert_cube=pert_cube,
    )
    print("  ✓ 2-3-5_gmm_vertical_sections", flush=True)

    # fig14：Slab2 三维体对比
    try:
        pipeline.concordance_eval.plot_slab_volume_comparison(
            labels_3d, metadata, viz, model_name, pert_cube=pert_cube,
        )
        print("  ✓ 2-3-11_slab_volume_3d", flush=True)
    except Exception as e:
        print(f"  ⚠️ 2-3-11 失败: {e}", flush=True)

print("done", flush=True)
