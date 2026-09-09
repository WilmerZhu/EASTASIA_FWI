#!/usr/bin/env python3
"""临时脚本：重生成带 δlnVp/δlnVs 行的 2-3-5 / 2-3-11。"""
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

models = ['2022_SinoScope1.0', '2024_EARA2024', '2024_FWEA23']
config = mod.ClusteringConfig()
config.depth_stratified['scheme'] = 'moho_4band'
config.preprocessing['perturbation']['enabled'] = True
pipeline = mod.SmartClusteringPipeline(config)

for model_name in models:
    print('===', model_name, flush=True)
    out = pipeline._scheme_output_root() / model_name
    viz = out / 'visualizations'
    data_cube, metadata = pipeline.loader.load_3d_model(model_name, ['vp', 'vs'])
    _, _, prep = pipeline.processor.prepare_for_clustering(data_cube, metadata)
    metadata = {**metadata, **prep}
    labels_3d = np.load(out / 'gmm_labels.npz')['labels_3d']
    pert_cube = pipeline.processor.last_feature_cube
    print(
        '  pert_cube',
        None if pert_cube is None else pert_cube.shape,
        'labels',
        labels_3d.shape,
        flush=True,
    )
    pipeline.basic_visualizer.set_processor(pipeline.processor)
    pipeline.basic_visualizer.plot_vertical_sections(
        labels_3d,
        metadata,
        viz,
        'gmm',
        model_name,
        concordance_eval=pipeline.concordance_eval,
        pert_cube=pert_cube,
    )
    pipeline.concordance_eval.plot_slab_volume_comparison(
        labels_3d, metadata, viz, model_name, pert_cube=pert_cube
    )
print('done', flush=True)
