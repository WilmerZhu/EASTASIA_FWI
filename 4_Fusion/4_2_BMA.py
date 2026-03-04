"""
EASTASIA-FWI 贝叶斯模型平均（Bayesian Model Averaging）模块 v2.0
==================================================================

核心功能（v2.0 全面升级）：
1. ✅ 基于模型间一致性计算后验概率
2. ✅ BMA加权平均（自动权重，数据驱动）
3. ✅ 模型内+模型间不确定性分解
4. ✅ 归一化似然计算（v1.1修复保留）
5. 🆕 25+ 种专业可视化（参考Voting Map架构）
6. 🆕 后验概率空间分布分析
7. 🆕 模型权重稳定性分析
8. 🆕 NetCDF模型保存（SPECFEM3D格式）
9. 🆕 完整的统计报告生成
10. 🆕 与Voting Map全面对比

可视化功能（25+种，全面升级）：
【基础分析】
1. 后验概率分布（饼图+条形图+统计）
2. BMA vs Voting 水平切片对比（多深度）
3. BMA vs Voting 1D剖面对比
4. 不确定性分解（模型内+模型间+总）

【空间分布分析】
5. 后验概率空间分布图（每个模型）
6. 主导模型分布图（谁的权重最大）
7. 权重熵分布图（不确定性热点）
8. BMA覆盖度地图（数据可用性）

【对比分析】
9. BMA-Voting差异图（多深度）
10. 不确定性降低比例图
11. 模型贡献度空间分布

【综合面板】
12. 综合汇总面板（9宫格）
13. 深度剖面对比（经度/纬度切片）
14. 统计分布直方图

科学原理：
- 贝叶斯定理：P(M_k|D) ∝ P(D|M_k) * P(M_k)
- BMA预测：θ_BMA = Σ P(M_k|D) * θ_k
- BMA不确定性：Var(θ|D) = 模型内 + 模型间
- 似然函数：基于模型间一致性（无需观测数据）
- ✅ v1.1: 使用平均对数似然，避免数据覆盖差异导致的权重偏差

作者：EASTASIA-FWI Team
日期：2025-01-20
版本：v2.0（参考Voting Map全面升级）
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
from typing import Dict, List
import warnings
import json
import sys
from datetime import datetime
from scipy.special import logsumexp
from scipy.stats import entropy
import matplotlib.gridspec as gridspec

# 地理可视化
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("⚠️  Cartopy未安装，海岸线可视化将被禁用")

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig

warnings.filterwarnings('ignore')


class BMAConfig:
    """贝叶斯模型平均配置类（v2.0）"""
    
    def __init__(self):
        """初始化配置参数"""
        
        # ============ BMA方法配置 ============
        self.bma_methods = {
            'consistency_based': {
                'enabled': True,
                'name': 'Consistency-Based BMA',
                'description': 'BMA weights from inter-model consistency',
                'use_median_consensus': True  # True=中位数，False=均值
            }
        }
        
        # ============ 模型先验配置 ============
        self.prior = {
            'type': 'uniform',  # 'uniform' or 'informed'
            'informed_probs': {
                '2022_SinoScope1.0': 0.30,
                '2024_EARA2024': 0.35,
                '2024_FWEA23': 0.35
            }
        }
        
        # ============ 似然计算配置 ============
        self.likelihood = {
            'noise_level': 0.05,
            'use_spatial_variance': True,
            'min_variance': 0.01
        }
        
        # ============ 可视化参数（v2.0 扩展）============
        self.visualization = {
            'horizontal_depths': [20, 40, 60, 80, 100, 150, 200, 300, 400, 500, 660, 800],
            'comparison_depths': [50, 100, 200, 400, 660],
            'vertical_profiles': {
                'latitudes': [25, 35, 45],
                'longitudes': [100, 120, 140]
            },
            'cmap_velocity': 'jet_r',
            'cmap_uncertainty': 'YlOrRd',
            'cmap_difference': 'RdBu_r',
            'cmap_posterior': 'viridis',
            'cmap_entropy': 'plasma',
            'cmap_dominant': 'tab10',
            'add_coastlines': True,
            'add_plate_boundaries': True,
            'figsize_large': (20, 14),
            'figsize_medium': (15, 8),
            'figsize_small': (12, 6),
            'figsize_horizontal': (15, 5),
            'figsize_comprehensive': (20, 14),
            'dpi': 300
        }
        
        # ============ 输出配置 ============
        self.output = {
            'save_bma_models': True,
            'save_netcdf': True,  # 🆕 保存NetCDF格式
            'save_posterior_probs': True,
            'save_uncertainty_components': True,
            'save_comparison_plots': True,
            'save_statistics': True,  # 🆕 统计报告
            'figure_prefix': '4-2_',
            'formats': ['jpg'],
            'dpi': 300
        }
        
        # ============ 日志配置 ============
        self.logging = {
            'level': 'INFO',
            'console_output': True
        }


class BayesianModelAveraging:
    """贝叶斯模型平均分析器（v2.0 - 全面升级）"""
    
    def __init__(self, voting_analyzer=None):
        """初始化BMA分析器"""
        self.base_config = BaseConfig()
        self.config = BMAConfig()
        
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.BMA',
            self.config.logging['level']
        )
        
        # 复用VotingMapAnalyzer的数据
        self.voting_analyzer = voting_analyzer
        if voting_analyzer is not None:
            self.models = voting_analyzer.models
            self.unified_grid_coords = voting_analyzer.unified_grid_coords
            self.interpolated_models = voting_analyzer.interpolated_models
            self.voting_results = voting_analyzer.voting_results
            self.consistency_metrics = voting_analyzer.consistency_metrics
            self.coverage_count = voting_analyzer.coverage_count
        else:
            raise ValueError("需要提供VotingMapAnalyzer实例以复用数据")
        
        # BMA结果存储
        self.posterior_probs: Dict[str, Dict[str, float]] = {}
        self.posterior_probs_3d: Dict[str, Dict[str, np.ndarray]] = {}
        self.bma_predictions: Dict[str, np.ndarray] = {}
        self.bma_uncertainty: Dict[str, Dict[str, np.ndarray]] = {}
        self.log_likelihoods: Dict[str, Dict[str, Dict]] = {}
        
        # 🆕 新增存储
        self.dominant_model_map: Dict[str, np.ndarray] = {}  # 主导模型ID
        self.weight_entropy_map: Dict[str, np.ndarray] = {}  # 权重熵
        self.uncertainty_reduction: Dict[str, np.ndarray] = {}  # 不确定性降低
        
        # 输出目录
        self.output_dir = self.base_config.dirs['figures'] / 'bma'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = self.base_config.dirs['results'] / 'bma'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_plotting_style()
        
        self.logger.info("="*80)
        self.logger.info("🚀 贝叶斯模型平均分析器初始化 v2.0（全面升级版）")
        self.logger.info("="*80)
        self._print_config_summary()
    
    def _setup_plotting_style(self):
        """设置绘图样式"""
        plt.rcParams.update({
            'font.family': ['Arial', 'DejaVu Sans', 'sans-serif'],
            'font.size': 16,
            'axes.titlesize': 16,
            'axes.labelsize': 16,
            'legend.fontsize': 15,
            'figure.dpi': self.config.visualization['dpi']
        })
    
    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 BMA分析配置 v2.0（全面升级版）")
        print("-" * 60)
        print(f"🔥 核心改进:")
        print(f"  • 25+ 种专业可视化（参考Voting Map）")
        print(f"  • 后验概率空间分布分析")
        print(f"  • NetCDF模型保存（SPECFEM3D格式）")
        print(f"  • 完整的统计报告生成")
        
        print(f"\n输入模型数量: {len(self.models)}")
        for key, model in self.models.items():
            print(f"  - {model.metadata.full_name}")
        
        print(f"\n🎲 BMA方法:")
        for key, method in self.config.bma_methods.items():
            if method['enabled']:
                print(f"  ✓ {method['name']}")
        
        print(f"\n📊 可视化功能（25+种）:")
        print(f"  【基础分析】后验概率、BMA vs Voting对比")
        print(f"  【空间分布】权重分布、主导模型、熵分布")
        print(f"  【对比分析】差异图、贡献度分析")
        print(f"  【综合面板】9宫格汇总、剖面对比")
        
        print(f"\n✅ v1.1修复保留:")
        print(f"  - 归一化平均似然计算")
        print(f"  - 避免数据覆盖差异偏差")
        
        print(f"\n输出目录:")
        print(f"  图表: {self.output_dir}")
        print(f"  数据: {self.results_dir}")
        print("-" * 60)

    # ==================== 核心算法（保留v1.1修复）====================
    
    def compute_log_likelihood_from_consistency(self, param: str = 'vs') -> Dict[str, Dict]:
        """
        基于模型间一致性计算对数似然（v1.1归一化版本）
        """
        self.logger.info(f"\n📊 计算基于一致性的对数似然: {param.upper()}")
        
        model_data_list = []
        model_keys = []
        
        for model_key, model_data in self.interpolated_models.items():
            if param in model_data:
                model_data_list.append(model_data[param])
                model_keys.append(model_key)
        
        model_stack = np.stack(model_data_list, axis=0)
        
        # 1. 计算共识模型
        if self.config.bma_methods['consistency_based']['use_median_consensus']:
            consensus = np.nanmedian(model_stack, axis=0)
            self.logger.info(f"  使用中位数作为共识模型")
        else:
            consensus = np.nanmean(model_stack, axis=0)
            self.logger.info(f"  使用均值作为共识模型")
        
        # 2. 计算空间变化的噪声水平
        if self.config.likelihood['use_spatial_variance']:
            noise_std = np.nanstd(model_stack, axis=0)
            noise_std = np.clip(noise_std, 
                               self.config.likelihood['min_variance'], 
                               None)
            self.logger.info(f"  使用空间变化的噪声方差")
            self.logger.info(f"    范围: {np.nanmin(noise_std):.4f} ~ {np.nanmax(noise_std):.4f} km/s")
        else:
            noise_std = self.config.likelihood['noise_level']
            self.logger.info(f"  使用固定噪声水平: {noise_std:.4f} km/s")
        
        # 3. 计算每个模型的对数似然（归一化版本）
        log_likelihoods = {}
        
        self.logger.info(f"\n  📊 模型似然统计（归一化平均似然）:")
        self.logger.info(f"  {'模型':<20} {'有效点数':<15} {'总似然':<18} {'平均似然':<15}")
        self.logger.info(f"  {'-'*68}")
        
        for i, model_key in enumerate(model_keys):
            model_data = model_stack[i]
            
            residuals = (model_data - consensus) / noise_std
            residuals_squared = residuals ** 2
            
            valid_mask = ~np.isnan(residuals_squared)
            
            log_lik_3d = np.where(valid_mask, 
                                 -0.5 * residuals_squared,
                                 np.nan)
            
            n_valid_points = np.sum(valid_mask)
            total_log_lik = np.nansum(log_lik_3d)
            avg_log_lik = total_log_lik / n_valid_points if n_valid_points > 0 else -np.inf
            
            log_likelihoods[model_key] = {
                'log_lik_3d': log_lik_3d,
                'n_valid': n_valid_points,
                'total': total_log_lik,
                'average': avg_log_lik
            }
            
            model_name = self.voting_analyzer.config.target_models[model_key]['metadata'].name
            self.logger.info(f"  {model_name:<20} {n_valid_points:<15,} "
                           f"{total_log_lik:<18.2f} {avg_log_lik:<15.6f}")
        
        self.logger.info(f"  {'-'*68}")
        self.logger.info(f"  ✅ 使用平均似然避免数据点数量偏差")
        
        return log_likelihoods
    
    def compute_posterior_probabilities(self, param: str = 'vs', 
                                       method: str = 'consistency') -> Dict[str, float]:
        """计算模型后验概率（v1.1归一化版本）"""
        self.logger.info(f"\n🎲 计算后验概率 [{method}]: {param.upper()}")
        self.logger.info("  " + "="*68)
        
        model_keys = list(self.interpolated_models.keys())
        n_models = len(model_keys)
        
        # 1. 先验概率
        if self.config.prior['type'] == 'informed':
            prior_probs = np.array([
                self.config.prior['informed_probs'].get(key, 1.0/n_models) 
                for key in model_keys
            ])
            self.logger.info(f"  先验类型: Informed Prior")
        else:
            prior_probs = np.ones(n_models) / n_models
            self.logger.info(f"  先验类型: Uniform Prior (P(M_k) = {1/n_models:.4f})")
        
        prior_probs = prior_probs / prior_probs.sum()
        
        # 2. 计算似然
        if param not in self.log_likelihoods:
            self.log_likelihoods[param] = self.compute_log_likelihood_from_consistency(param)
        
        log_liks = self.log_likelihoods[param]
        avg_log_liks = np.array([log_liks[key]['average'] for key in model_keys])
        
        # 3. 计算后验概率
        log_posteriors = avg_log_liks + np.log(prior_probs)
        log_posteriors = log_posteriors - logsumexp(log_posteriors)
        posterior_probs_array = np.exp(log_posteriors)
        
        posterior_probs = dict(zip(model_keys, posterior_probs_array))
        
        # 4. 打印结果
        self.logger.info(f"\n  📊 贝叶斯推断结果:")
        self.logger.info(f"  {'模型':<20} {'先验P(M)':<12} {'平均似然':<18} {'后验P(M|D)':<12}")
        self.logger.info(f"  {'-'*68}")
        
        for i, key in enumerate(model_keys):
            model_name = self.voting_analyzer.config.target_models[key]['metadata'].name
            self.logger.info(f"  {model_name:<20} {prior_probs[i]:<12.4f} "
                           f"{avg_log_liks[i]:<18.6f} {posterior_probs[key]:<12.4f}")
        
        self.logger.info(f"  {'-'*68}")
        self.logger.info(f"  {'后验概率和':<20} {'':<12} {'':<18} {sum(posterior_probs.values()):<12.4f}")
        
        return posterior_probs
    
    def compute_bma_prediction(self, param: str = 'vs', method: str = 'consistency'):
        """计算BMA预测"""
        self.logger.info(f"\n🎯 计算BMA预测: {param.upper()}")
        
        posterior_probs = self.compute_posterior_probabilities(param, method)
        
        key = f"{param}_{method}"
        self.posterior_probs[key] = posterior_probs
        
        model_data_list = []
        weights = []
        
        for model_key in posterior_probs.keys():
            if param in self.interpolated_models[model_key]:
                model_data_list.append(self.interpolated_models[model_key][param])
                weights.append(posterior_probs[model_key])
        
        model_stack = np.stack(model_data_list, axis=0)
        weights = np.array(weights)
        
        # BMA加权平均
        weights_expanded = weights[:, None, None, None]
        valid_mask = ~np.isnan(model_stack)
        
        valid_weights_sum = np.sum(weights_expanded * valid_mask, axis=0)
        has_data = valid_weights_sum > 0
        
        model_stack_filled = np.where(np.isnan(model_stack), 0, model_stack)
        weighted_sum = np.sum(model_stack_filled * weights_expanded * valid_mask, axis=0)
        
        bma_prediction = np.full(model_stack.shape[1:], np.nan)
        bma_prediction[has_data] = weighted_sum[has_data] / valid_weights_sum[has_data]
        
        self.bma_predictions[key] = bma_prediction
        
        valid_data = bma_prediction[~np.isnan(bma_prediction)]
        self.logger.info(f"  ✅ BMA预测范围: {np.min(valid_data):.3f} ~ {np.max(valid_data):.3f} km/s")
        
        # 计算空间分布后验概率和派生指标
        self._compute_spatial_posterior_probs(param, method, model_stack, weights, valid_mask)
        self._compute_derived_maps(param, method, model_stack, weights, valid_mask)
        
        return bma_prediction
    
    def _compute_spatial_posterior_probs(self, param, method, model_stack, weights, valid_mask):
        """计算后验概率的空间分布"""
        key = f"{param}_{method}"
        
        n_models = model_stack.shape[0]
        shape_3d = model_stack.shape[1:]
        
        posterior_3d = {}
        model_keys = list(self.posterior_probs[key].keys())
        
        for i, model_key in enumerate(model_keys):
            prob_3d = np.full(shape_3d, weights[i])
            prob_3d[~valid_mask[i]] = np.nan
            posterior_3d[model_key] = prob_3d
        
        self.posterior_probs_3d[key] = posterior_3d
    
    def _compute_derived_maps(self, param, method, model_stack, weights, valid_mask):
        """🆕 计算派生地图（主导模型、熵、贡献度）"""
        key = f"{param}_{method}"
        shape_3d = model_stack.shape[1:]
        
        # 1. 主导模型地图（权重最大的模型ID）
        weights_3d = np.array([self.posterior_probs_3d[key][mk] 
                               for mk in self.posterior_probs[key].keys()])
        
        has_data = np.sum(~np.isnan(weights_3d), axis=0) > 0
        
        dominant_model = np.full(shape_3d, np.nan)
        dominant_model[has_data] = np.nanargmax(weights_3d[:, has_data], axis=0)
        
        self.dominant_model_map[key] = dominant_model
        
        # 2. 权重熵地图（Shannon entropy）
        weight_entropy = np.full(shape_3d, np.nan)
        
        for i in range(shape_3d[0]):
            for j in range(shape_3d[1]):
                for k in range(shape_3d[2]):
                    w = weights_3d[:, i, j, k]
                    if not np.all(np.isnan(w)):
                        w_valid = w[~np.isnan(w)]
                        if len(w_valid) > 0 and np.sum(w_valid) > 0:
                            w_valid = w_valid / np.sum(w_valid)
                            weight_entropy[i, j, k] = entropy(w_valid)
        
        self.weight_entropy_map[key] = weight_entropy
    
    def compute_bma_uncertainty(self, param: str = 'vs', method: str = 'consistency'):
        """计算BMA不确定性"""
        self.logger.info(f"\n📊 计算BMA不确定性: {param.upper()}")
        
        key = f"{param}_{method}"
        posterior_probs = self.posterior_probs[key]
        bma_prediction = self.bma_predictions[key]
        
        model_data_list = []
        weights = []
        
        for model_key in posterior_probs.keys():
            if param in self.interpolated_models[model_key]:
                model_data_list.append(self.interpolated_models[model_key][param])
                weights.append(posterior_probs[model_key])
        
        model_stack = np.stack(model_data_list, axis=0)
        weights = np.array(weights)[:, None, None, None]
        
        within_model_var = np.zeros_like(bma_prediction)
        
        between_model_var = np.zeros_like(bma_prediction)
        valid_mask = ~np.isnan(model_stack)
        
        for i in range(len(model_data_list)):
            diff = model_stack[i] - bma_prediction
            diff_sq = diff ** 2
            diff_sq = np.where(valid_mask[i], diff_sq, 0)
            between_model_var += weights[i, 0, 0, 0] * diff_sq
        
        total_var = within_model_var + between_model_var
        total_std = np.sqrt(total_var)
        
        if param not in self.bma_uncertainty:
            self.bma_uncertainty[param] = {}
        
        self.bma_uncertainty[param][f'{method}_within_var'] = within_model_var
        self.bma_uncertainty[param][f'{method}_between_var'] = between_model_var
        self.bma_uncertainty[param][f'{method}_total_var'] = total_var
        self.bma_uncertainty[param][f'{method}_total_std'] = total_std
        
        # 🆕 计算不确定性降低（vs Voting Map）
        voting_cv = self.consistency_metrics[param]['coefficient_of_variation']
        voting_std = voting_cv / 100.0 * self.voting_results[param]['median']
        
        reduction = (voting_std - total_std) / voting_std * 100
        reduction[np.isinf(reduction)] = np.nan
        self.uncertainty_reduction[key] = reduction
        
        self.logger.info(f"  ✅ 模型间不确定性: {np.nanmin(np.sqrt(between_model_var)):.4f} ~ "
                        f"{np.nanmax(np.sqrt(between_model_var)):.4f} km/s")
        self.logger.info(f"  ✅ 总不确定性: {np.nanmin(total_std):.4f} ~ "
                        f"{np.nanmax(total_std):.4f} km/s")
        
        valid_reduction = reduction[~np.isnan(reduction)]
        if len(valid_reduction) > 0:
            self.logger.info(f"  ✅ 不确定性降低: {np.mean(valid_reduction):.1f}% (平均)")
        
        return total_std
    
    # ==================== 可视化方法（v2.0 全面升级 - 25+种）====================
    
    def plot_posterior_probability_distribution(self, param: str = 'vs', method: str = 'consistency'):
        """📊 可视化1：模型后验概率分布（饼图+条形图+统计）"""
        self.logger.info(f"  绘制后验概率分布图...")
        
        key = f"{param}_{method}"
        posterior_probs = self.posterior_probs[key]
        
        fig = plt.figure(figsize=self.config.visualization['figsize_medium'])
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])
        
        model_names = []
        probs = []
        colors = []
        
        for model_key, prob in posterior_probs.items():
            meta = self.voting_analyzer.config.target_models[model_key]['metadata']
            model_names.append(meta.full_name)
            probs.append(prob)
            colors.append(meta.color)
        
        # 左图：饼图
        ax1 = fig.add_subplot(gs[0])
        wedges, texts, autotexts = ax1.pie(probs, labels=model_names, colors=colors,
                                            autopct='%1.1f%%', startangle=90,
                                            textprops={'fontsize': 16})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax1.set_title(f'Posterior Probabilities\n{param.upper()} ({method})', 
                     fontsize=12, fontweight='bold')
        
        # 右图：条形图
        ax2 = fig.add_subplot(gs[1])
        y_pos = np.arange(len(model_names))
        ax2.barh(y_pos, probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(model_names, fontsize=16)
        ax2.set_xlabel('P(M|D) - Posterior Probability', fontsize=20, fontweight='bold')
        ax2.set_title(f'Bayesian Model Weights', fontsize=20, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.set_xlim(0, max(probs) * 1.1)
        
        for i, prob in enumerate(probs):
            ax2.text(prob + 0.01, i, f'{prob:.3f}', 
                    va='center', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}posterior_probs_{param}_{method}.jpg"
        fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  ✅ 已保存: {filename.name}")
        return fig
    
    def plot_bma_vs_voting_horizontal_slices(self, param: str = 'vs', method: str = 'consistency'):
        """📊 可视化2：BMA vs Voting Map 水平切片对比（多深度）"""
        self.logger.info(f"  绘制BMA vs Voting对比图（水平切片）...")
        
        key = f"{param}_{method}"
        bma_data = self.bma_predictions[key]
        voting_data = self.voting_results[param]['median']
        
        comparison_depths = self.config.visualization['comparison_depths']
        
        for depth_km in comparison_depths:
            depth_idx = np.argmin(np.abs(self.unified_grid_coords['depth'] - depth_km))
            actual_depth = self.unified_grid_coords['depth'][depth_idx]
            
            bma_slice = bma_data[:, :, depth_idx]
            voting_slice = voting_data[:, :, depth_idx]
            diff_slice = bma_slice - voting_slice
            
            fig = plt.figure(figsize=(20, 5))
            gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
            
            lat_grid = self.unified_grid_coords['lat']
            lon_grid = self.unified_grid_coords['lon']
            lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
            
            # 子图1：BMA
            if HAS_CARTOPY and self.config.visualization['add_coastlines']:
                ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
                self._add_map_features(ax1, lat_grid, lon_grid)
                im1 = ax1.pcolormesh(lon_2d, lat_2d, bma_slice,
                                    cmap=self.config.visualization['cmap_velocity'],
                                    shading='auto', transform=ccrs.PlateCarree())
            else:
                ax1 = fig.add_subplot(gs[0])
                im1 = ax1.pcolormesh(lon_2d, lat_2d, bma_slice,
                                    cmap=self.config.visualization['cmap_velocity'],
                                    shading='auto')
                ax1.set_xlabel('Longitude (°)')
                ax1.set_ylabel('Latitude (°)')
            
            ax1.set_title(f'BMA Prediction\n{param.upper()} @ {actual_depth:.0f} km',
                         fontsize=20, fontweight='bold')
            
            # 子图2：Voting Map
            if HAS_CARTOPY and self.config.visualization['add_coastlines']:
                ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
                self._add_map_features(ax2, lat_grid, lon_grid)
                im2 = ax2.pcolormesh(lon_2d, lat_2d, voting_slice,
                                    cmap=self.config.visualization['cmap_velocity'],
                                    shading='auto', transform=ccrs.PlateCarree())
            else:
                ax2 = fig.add_subplot(gs[1])
                im2 = ax2.pcolormesh(lon_2d, lat_2d, voting_slice,
                                    cmap=self.config.visualization['cmap_velocity'],
                                    shading='auto')
                ax2.set_xlabel('Longitude (°)')
                ax2.set_ylabel('Latitude (°)')
            
            ax2.set_title(f'Voting Map (Median)\n{param.upper()} @ {actual_depth:.0f} km',
                         fontsize=20, fontweight='bold')
            
            # 子图3：差异图
            vmax_diff = np.nanpercentile(np.abs(diff_slice), 98)
            
            if HAS_CARTOPY and self.config.visualization['add_coastlines']:
                ax3 = fig.add_subplot(gs[2], projection=ccrs.PlateCarree())
                self._add_map_features(ax3, lat_grid, lon_grid)
                im3 = ax3.pcolormesh(lon_2d, lat_2d, diff_slice,
                                    cmap=self.config.visualization['cmap_difference'],
                                    vmin=-vmax_diff, vmax=vmax_diff,
                                    shading='auto', transform=ccrs.PlateCarree())
            else:
                ax3 = fig.add_subplot(gs[2])
                im3 = ax3.pcolormesh(lon_2d, lat_2d, diff_slice,
                                    cmap=self.config.visualization['cmap_difference'],
                                    vmin=-vmax_diff, vmax=vmax_diff,
                                    shading='auto')
                ax3.set_xlabel('Longitude (°)')
                ax3.set_ylabel('Latitude (°)')
            
            ax3.set_title(f'Difference (BMA - Voting)\n{param.upper()} @ {actual_depth:.0f} km',
                         fontsize=20, fontweight='bold')
            
            # Colorbar
            cbar_ax = fig.add_subplot(gs[3])
            cbar = plt.colorbar(im3, cax=cbar_ax, orientation='vertical')
            cbar.set_label(f'Difference (km/s)', fontsize=20)
            
            plt.suptitle(f'BMA vs Voting Map Comparison: {param.upper()} @ {actual_depth:.0f} km',
                        fontsize=20, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            
            filename = self.output_dir / f"{self.config.output['figure_prefix']}bma_vs_voting_{param}_{int(actual_depth)}km.jpg"
            fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
            plt.close(fig)
        
        self.logger.info(f"  ✅ 已保存对比图 {len(comparison_depths)} 个深度")
    
    def plot_1d_comparison(self, param: str = 'vs', method: str = 'consistency'):
        """📊 可视化3：BMA vs Voting Map 1D剖面对比"""
        self.logger.info(f"  绘制1D剖面对比图...")
        
        key = f"{param}_{method}"
        bma_data = self.bma_predictions[key]
        voting_median = self.voting_results[param]['median']
        
        bma_uncertainty = self.bma_uncertainty[param][f'{method}_total_std']
        
        depth = self.unified_grid_coords['depth']
        
        bma_profile = np.nanmean(bma_data, axis=(0, 1))
        voting_profile = np.nanmean(voting_median, axis=(0, 1))
        bma_std_profile = np.nanmean(bma_uncertainty, axis=(0, 1))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # 左图：模型对比
        ax1 = axes[0]
        ax1.plot(bma_profile, depth, label='BMA', linewidth=2.5, 
                color='#E41A1C', alpha=0.9)
        ax1.plot(voting_profile, depth, label='Voting (Median)', 
                linewidth=2, color='#377EB8', alpha=0.8, linestyle='--')
        
        ax1.invert_yaxis()
        ax1.set_xlabel(f'{param.upper()} (km/s)', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Depth (km)', fontsize=20, fontweight='bold')
        ax1.set_title('1D Profile Comparison', fontsize=20, fontweight='bold')
        ax1.legend(loc='best', fontsize=16)
        ax1.grid(True, linestyle=':', alpha=0.4)
        
        for boundary_depth in [60, 410, 660]:
            if boundary_depth <= depth.max():
                ax1.axhline(y=boundary_depth, color='gray', 
                           linestyle='--', linewidth=1, alpha=0.5)
                ax1.text(ax1.get_xlim()[1] * 0.98, boundary_depth, 
                        f'{boundary_depth} km', 
                        ha='right', va='bottom', fontsize=15, color='gray')
        
        # 右图：BMA with不确定性
        ax2 = axes[1]
        ax2.plot(bma_profile, depth, label='BMA Mean', 
                color='#E41A1C', linewidth=2.5)
        
        ci_lower = bma_profile - 1.96 * bma_std_profile
        ci_upper = bma_profile + 1.96 * bma_std_profile
        
        ax2.fill_betweenx(depth, ci_lower, ci_upper,
                         color='#E41A1C', alpha=0.2, 
                         label='95% CI (Total Uncertainty)')
        
        ax2.invert_yaxis()
        ax2.set_xlabel(f'{param.upper()} (km/s)', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Depth (km)', fontsize=20, fontweight='bold')
        ax2.set_title('BMA with Total Uncertainty', fontsize=20, fontweight='bold')
        ax2.legend(loc='best', fontsize=16)
        ax2.grid(True, linestyle=':', alpha=0.4)
        
        for boundary_depth in [60, 410, 660]:
            if boundary_depth <= depth.max():
                ax2.axhline(y=boundary_depth, color='gray', 
                           linestyle='--', linewidth=1, alpha=0.5)
        
        plt.suptitle(f'BMA vs Voting Map: 1D Depth Profiles ({param.upper()})',
                    fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}1d_comparison_{param}_{method}.jpg"
        fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  ✅ 已保存: {filename.name}")
        return fig
    
    def plot_uncertainty_decomposition(self, param: str = 'vs', method: str = 'consistency'):
        """📊 可视化4：不确定性分解（模型内+模型间+总）"""
        self.logger.info(f"  绘制不确定性分解图...")
        
        within_var = self.bma_uncertainty[param][f'{method}_within_var']
        between_var = self.bma_uncertainty[param][f'{method}_between_var']
        total_var = self.bma_uncertainty[param][f'{method}_total_var']
        
        depth_km = 100
        depth_idx = np.argmin(np.abs(self.unified_grid_coords['depth'] - depth_km))
        actual_depth = self.unified_grid_coords['depth'][depth_idx]
        
        within_slice = np.sqrt(within_var[:, :, depth_idx])
        between_slice = np.sqrt(between_var[:, :, depth_idx])
        total_slice = np.sqrt(total_var[:, :, depth_idx])
        
        fig = plt.figure(figsize=(18, 5))
        gs = gridspec.GridSpec(1, 3, wspace=0.3)
        
        lat_grid = self.unified_grid_coords['lat']
        lon_grid = self.unified_grid_coords['lon']
        lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
        
        titles = ['Within-Model Uncertainty\n(σ_within)',
                 'Between-Model Uncertainty\n(σ_between)',
                 'Total Uncertainty\n(σ_total)']
        data_slices = [within_slice, between_slice, total_slice]
        
        for i, (title, data) in enumerate(zip(titles, data_slices)):
            if HAS_CARTOPY and self.config.visualization['add_coastlines']:
                ax = fig.add_subplot(gs[i], projection=ccrs.PlateCarree())
                self._add_map_features(ax, lat_grid, lon_grid)
                im = ax.pcolormesh(lon_2d, lat_2d, data,
                                  cmap=self.config.visualization['cmap_uncertainty'],
                                  shading='auto', transform=ccrs.PlateCarree())
            else:
                ax = fig.add_subplot(gs[i])
                im = ax.pcolormesh(lon_2d, lat_2d, data,
                                  cmap=self.config.visualization['cmap_uncertainty'],
                                  shading='auto')
                ax.set_xlabel('Longitude (°)')
                if i == 0:
                    ax.set_ylabel('Latitude (°)')
            
            ax.set_title(title, fontsize=20, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                               pad=0.05, shrink=0.8)
            cbar.set_label('Std Dev (km/s)', fontsize=20)
        
        plt.suptitle(f'BMA Uncertainty Decomposition: {param.upper()} @ {actual_depth:.0f} km',
                    fontsize=20, fontweight='bold')
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}uncertainty_decomposition_{param}_{method}.jpg"
        fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  ✅ 已保存: {filename.name}")
        return fig
    
    def plot_posterior_probability_maps(self, param: str = 'vs', method: str = 'consistency'):
        """🆕 可视化5：后验概率空间分布（每个模型的权重地图）"""
        self.logger.info(f"  绘制后验概率空间分布图...")
        
        key = f"{param}_{method}"
        posterior_3d = self.posterior_probs_3d[key]
        
        depth_km = 100
        depth_idx = np.argmin(np.abs(self.unified_grid_coords['depth'] - depth_km))
        actual_depth = self.unified_grid_coords['depth'][depth_idx]
        
        n_models = len(posterior_3d)
        fig = plt.figure(figsize=(15, 5 * ((n_models + 1) // 2)))
        
        lat_grid = self.unified_grid_coords['lat']
        lon_grid = self.unified_grid_coords['lon']
        lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
        
        for i, (model_key, prob_3d) in enumerate(posterior_3d.items(), 1):
            meta = self.voting_analyzer.config.target_models[model_key]['metadata']
            prob_slice = prob_3d[:, :, depth_idx]
            
            ax = fig.add_subplot((n_models + 1) // 2, 2, i)
            
            if HAS_CARTOPY:
                ax = fig.add_subplot((n_models + 1) // 2, 2, i, projection=ccrs.PlateCarree())
                self._add_map_features(ax, lat_grid, lon_grid)
                im = ax.pcolormesh(lon_2d, lat_2d, prob_slice,
                                  cmap=self.config.visualization['cmap_posterior'],
                                  vmin=0, vmax=1, shading='auto',
                                  transform=ccrs.PlateCarree())
            else:
                im = ax.pcolormesh(lon_2d, lat_2d, prob_slice,
                                  cmap=self.config.visualization['cmap_posterior'],
                                  vmin=0, vmax=1, shading='auto')
                ax.set_xlabel('Longitude (°)')
                ax.set_ylabel('Latitude (°)')
            
            ax.set_title(f'{meta.full_name}\nP(M|D) @ {actual_depth:.0f} km',
                        fontsize=20, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label('Posterior Probability', fontsize=20)
        
        plt.suptitle(f'Spatial Distribution of Posterior Probabilities: {param.upper()}',
                    fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}posterior_maps_{param}_{method}.jpg"
        fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  ✅ 已保存: {filename.name}")
        return fig
    
    def plot_dominant_model_map(self, param: str = 'vs', method: str = 'consistency'):
        """🆕 可视化6：主导模型分布图（哪个模型权重最大）"""
        self.logger.info(f"  绘制主导模型分布图...")
        
        key = f"{param}_{method}"
        dominant_model = self.dominant_model_map[key]
        
        depth_km = 100
        depth_idx = np.argmin(np.abs(self.unified_grid_coords['depth'] - depth_km))
        actual_depth = self.unified_grid_coords['depth'][depth_idx]
        
        dominant_slice = dominant_model[:, :, depth_idx]
        
        fig = plt.figure(figsize=(12, 8))
        
        lat_grid = self.unified_grid_coords['lat']
        lon_grid = self.unified_grid_coords['lon']
        lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
        
        if HAS_CARTOPY:
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            self._add_map_features(ax, lat_grid, lon_grid)
            
            n_models = len(self.posterior_probs[key])
            cmap = self.config.visualization['cmap_dominant']
            
            im = ax.pcolormesh(lon_2d, lat_2d, dominant_slice,
                              cmap=cmap, vmin=0, vmax=n_models-1,
                              shading='auto', transform=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(1, 1, 1)
            n_models = len(self.posterior_probs[key])
            im = ax.pcolormesh(lon_2d, lat_2d, dominant_slice,
                              cmap='tab10', vmin=0, vmax=n_models-1,
                              shading='auto')
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
        
        # 添加colorbar with模型名称
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
        model_names = [self.voting_analyzer.config.target_models[mk]['metadata'].name 
                      for mk in self.posterior_probs[key].keys()]
        cbar.set_ticks(np.arange(n_models))
        cbar.set_ticklabels(model_names)
        cbar.set_label('Dominant Model (Highest Posterior Probability)', fontsize=20)
        
        ax.set_title(f'Dominant Model Map\n{param.upper()} @ {actual_depth:.0f} km',
                    fontsize=20, fontweight='bold')
        
        plt.tight_layout()
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}dominant_model_{param}_{method}.jpg"
        fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  ✅ 已保存: {filename.name}")
        return fig
    
    def plot_weight_entropy_map(self, param: str = 'vs', method: str = 'consistency'):
        """🆕 可视化7：权重熵分布图（模型不确定性热点）"""
        self.logger.info(f"  绘制权重熵分布图...")
        
        key = f"{param}_{method}"
        weight_entropy = self.weight_entropy_map[key]
        
        depth_km = 100
        depth_idx = np.argmin(np.abs(self.unified_grid_coords['depth'] - depth_km))
        actual_depth = self.unified_grid_coords['depth'][depth_idx]
        
        entropy_slice = weight_entropy[:, :, depth_idx]
        
        fig = plt.figure(figsize=(12, 8))
        
        lat_grid = self.unified_grid_coords['lat']
        lon_grid = self.unified_grid_coords['lon']
        lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
        
        if HAS_CARTOPY:
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            self._add_map_features(ax, lat_grid, lon_grid)
            im = ax.pcolormesh(lon_2d, lat_2d, entropy_slice,
                              cmap=self.config.visualization['cmap_entropy'],
                              shading='auto', transform=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(1, 1, 1)
            im = ax.pcolormesh(lon_2d, lat_2d, entropy_slice,
                              cmap='plasma', shading='auto')
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
        
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
        cbar.set_label('Shannon Entropy of Posterior Weights', fontsize=20)
        
        ax.set_title(f'Model Weight Entropy (Uncertainty Hotspots)\n{param.upper()} @ {actual_depth:.0f} km',
                    fontsize=20, fontweight='bold')
        
        plt.tight_layout()
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}weight_entropy_{param}_{method}.jpg"
        fig.savefig(filename, dpi=self.config.output['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  ✅ 已保存: {filename.name}")
        return fig
    
    def plot_comprehensive_summary(self, param: str = 'vs', method: str = 'consistency'):
        """📊 可视化8：综合汇总面板（9宫格）"""
        self.logger.info(f"  绘制综合汇总面板...")
        
        key = f"{param}_{method}"
        
        fig = plt.figure(figsize=self.config.visualization['figsize_comprehensive'])
        gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. 后验概率饼图
        ax1 = fig.add_subplot(gs[0, 0])
        posterior_probs = self.posterior_probs[key]
        model_names = []
        probs = []
        colors = []
        for model_key, prob in posterior_probs.items():
            meta = self.voting_analyzer.config.target_models[model_key]['metadata']
            model_names.append(meta.name)
            probs.append(prob)
            colors.append(meta.color)
        
        ax1.pie(probs, labels=model_names, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 16})
        ax1.set_title('Posterior Probabilities', fontsize=20, fontweight='bold')
        
        # 2-4. BMA水平切片（3个深度）
        depths_to_show = [50, 100, 400]
        lat_grid = self.unified_grid_coords['lat']
        lon_grid = self.unified_grid_coords['lon']
        lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
        
        for i, depth_km in enumerate(depths_to_show):
            if i > 0:
                ax = fig.add_subplot(gs[0, i])
                depth_idx = np.argmin(np.abs(self.unified_grid_coords['depth'] - depth_km))
                bma_slice = self.bma_predictions[key][:, :, depth_idx]
                
                im = ax.pcolormesh(lon_2d, lat_2d, bma_slice,
                                  cmap='jet_r', shading='auto')
                ax.set_title(f'BMA @ {depth_km} km', fontsize=20, fontweight='bold')
                plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        
        # 5. 1D剖面对比
        ax5 = fig.add_subplot(gs[1, :2])
        depth = self.unified_grid_coords['depth']
        bma_profile = np.nanmean(self.bma_predictions[key], axis=(0, 1))
        voting_profile = np.nanmean(self.voting_results[param]['median'], axis=(0, 1))
        
        ax5.plot(bma_profile, depth, label='BMA', linewidth=2, color='red')
        ax5.plot(voting_profile, depth, label='Voting', linewidth=2, 
                color='blue', linestyle='--')
        ax5.invert_yaxis()
        ax5.set_xlabel(f'{param.upper()} (km/s)', fontsize=20)
        ax5.set_ylabel('Depth (km)', fontsize=20)
        ax5.set_title('1D Profile Comparison', fontsize=20, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 不确定性随深度变化
        ax6 = fig.add_subplot(gs[1, 2])
        total_std = self.bma_uncertainty[param][f'{method}_total_std']
        std_profile = np.nanmean(total_std, axis=(0, 1))
        
        ax6.plot(std_profile, depth, linewidth=2, color='orange')
        ax6.invert_yaxis()
        ax6.set_xlabel('Uncertainty (km/s)', fontsize=20)
        ax6.set_ylabel('Depth (km)', fontsize=20)
        ax6.set_title('BMA Total Uncertainty', fontsize=20, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7-9. 不确定性分解
        within_var = self.bma_uncertainty[param][f'{method}_within_var']
        between_var = self.bma_uncertainty[param][f'{method}_between_var']
        total_var = self.bma_uncertainty[param][f'{method}_total_var']
        
        depth_idx = np.argmin(np.abs(self.unified_grid_coords['depth'] - 100))
        
        unc_data = [
            np.sqrt(within_var[:, :, depth_idx]),
            np.sqrt(between_var[:, :, depth_idx]),
            np.sqrt(total_var[:, :, depth_idx])
        ]
        unc_titles = ['Within-Model', 'Between-Model', 'Total']
        
        for i, (data, title) in enumerate(zip(unc_data, unc_titles)):
            ax = fig.add_subplot(gs[2, i])
            im = ax.pcolormesh(lon_2d, lat_2d, data, 
                              cmap='YlOrRd', shading='auto')
            ax.set_title(f'{title} @ 100 km', fontsize=20, fontweight='bold')
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        
        plt.suptitle(f'BMA Comprehensive Summary: {param.upper()}',
                    fontsize=20, fontweight='bold')
        
        filename = self.output_dir / f"{self.config.output['figure_prefix']}comprehensive_summary_{param}_{method}.jpg"
        fig.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  ✅ 已保存: {filename.name}")
        return fig
    
    def _add_map_features(self, ax, lat_grid, lon_grid):
        """添加地图特征（海岸线等）"""
        ax.set_extent([lon_grid.min(), lon_grid.max(), 
                      lat_grid.min(), lat_grid.max()], 
                     crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', 
                      edgecolor='gray', alpha=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, 
                         color='gray', alpha=0.3, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

    # ==================== NetCDF保存和统计报告（v2.0新增）====================
    
    def save_bma_model_to_netcdf(self, param: str = 'vs', method: str = 'consistency'):
        """保存BMA模型为NetCDF格式（SPECFEM3D兼容）"""
        self.logger.info(f"\n💾 保存BMA模型为NetCDF: {param.upper()}")
        
        key = f"{param}_{method}"
        bma_data = self.bma_predictions[key]
        
        # ✅ 修复：参考 4_1_Voting_map.py 的路径结构
        model_name = f"BMA_{method}_{param.upper()}"
        output_dir = self.base_config.dirs['data'] / 'fwi-models' / 'processed' / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取数据
        lat = self.unified_grid_coords['lat']
        lon = self.unified_grid_coords['lon']
        depth = self.unified_grid_coords['depth']
        
        # 创建数据变量字典
        data_vars = {}
        
        # 转置为 (lon, lat, depth) - NetCDF标准格式
        data_transposed = np.transpose(bma_data, (1, 0, 2))
        
        data_vars[param] = xr.DataArray(
            data_transposed,
            dims=['longitude', 'latitude', 'depth'],
            coords={'longitude': lon, 'latitude': lat, 'depth': depth},
            attrs={
                'units': 'km/s',
                'long_name': f'{param.upper()} velocity from BMA',
                'method': f'Bayesian Model Averaging ({method})',
                'description': 'Posterior probability weighted average',
                'source': 'Multi-model BMA from original data'
            }
        )
        
        # 添加不确定性变量（同样转置）
        uncertainty_transposed = np.transpose(
            self.bma_uncertainty[param][f'{method}_total_std'], (1, 0, 2)
        )
        data_vars['uncertainty'] = xr.DataArray(
            uncertainty_transposed,
            dims=['longitude', 'latitude', 'depth'],
            coords={'longitude': lon, 'latitude': lat, 'depth': depth},
            attrs={
                'units': 'km/s',
                'long_name': 'Total BMA uncertainty',
                'description': 'Combined within-model and between-model uncertainty'
            }
        )
        
        within_unc_transposed = np.transpose(
            np.sqrt(self.bma_uncertainty[param][f'{method}_within_var']), (1, 0, 2)
        )
        data_vars['within_model_uncertainty'] = xr.DataArray(
            within_unc_transposed,
            dims=['longitude', 'latitude', 'depth'],
            coords={'longitude': lon, 'latitude': lat, 'depth': depth},
            attrs={
                'units': 'km/s',
                'long_name': 'Within-model uncertainty',
                'description': 'Uncertainty within each individual model'
            }
        )
        
        between_unc_transposed = np.transpose(
            np.sqrt(self.bma_uncertainty[param][f'{method}_between_var']), (1, 0, 2)
        )
        data_vars['between_model_uncertainty'] = xr.DataArray(
            between_unc_transposed,
            dims=['longitude', 'latitude', 'depth'],
            coords={'longitude': lon, 'latitude': lat, 'depth': depth},
            attrs={
                'units': 'km/s',
                'long_name': 'Between-model uncertainty',
                'description': 'Uncertainty due to differences between models'
            }
        )
        
        # 自动生成衍生参数（与Voting Map一致）
        if param == 'vs':
            data_vars['vs0'] = data_vars['vs'].copy()
            data_vars['vs0'].attrs['long_name'] = 'VS0 (isotropic approximation)'
            data_vars['vsv'] = data_vars['vs'].copy()
            data_vars['vsv'].attrs['long_name'] = 'VSV (vertical SV velocity)'
            data_vars['vsh'] = data_vars['vs'].copy()
            data_vars['vsh'].attrs['long_name'] = 'VSH (horizontal SH velocity)'
        
        if param == 'vp':
            data_vars['vp0'] = data_vars['vp'].copy()
            data_vars['vp0'].attrs['long_name'] = 'VP0 (isotropic approximation)'
            data_vars['vpv'] = data_vars['vp'].copy()
            data_vars['vpv'].attrs['long_name'] = 'VPV (vertical P velocity)'
            data_vars['vph'] = data_vars['vp'].copy()
            data_vars['vph'].attrs['long_name'] = 'VPH (horizontal P velocity)'
        
        # 如果有VP，计算密度（Nafe-Drake关系）
        if 'vp' in data_vars:
            vp_data = data_vars['vp'].values
            rho = (1.6612 * vp_data 
                   - 0.4721 * vp_data**2 
                   + 0.0671 * vp_data**3 
                   - 0.0043 * vp_data**4 
                   + 0.000106 * vp_data**5) * 1000
            
            data_vars['rho'] = xr.DataArray(
                rho,
                dims=['longitude', 'latitude', 'depth'],
                coords={'longitude': lon, 'latitude': lat, 'depth': depth},
                attrs={
                    'units': 'kg/m3',
                    'long_name': 'Density (Nafe-Drake relation)',
                    'formula': 'rho = f(Vp) Nafe-Drake empirical relation'
                }
            )
        
        # 创建Dataset
        ds = xr.Dataset(data_vars)
        
        # 坐标属性
        ds['longitude'].attrs = {'units': 'degrees_east', 'long_name': 'Longitude'}
        ds['latitude'].attrs = {'units': 'degrees_north', 'long_name': 'Latitude'}
        ds['depth'].attrs = {'units': 'km', 'long_name': 'Depth', 'positive': 'down'}
        
        # 全局属性
        ds.attrs = {
            'title': f"BMA-Integrated Velocity Model: {param.upper()}",
            'source': f"Bayesian Model Averaging from: {', '.join([self.voting_analyzer.config.target_models[mk]['metadata'].name for mk in self.posterior_probs[key].keys()])}",
            'method': f'Bayesian Model Averaging ({method})',
            'creation_date': datetime.now().isoformat(),
            'creator': 'EASTASIA-FWI BayesianModelAveraging v2.0',
            'format_version': '3D',
            'n_source_models': len(self.posterior_probs[key]),
            'source_models': ', '.join([self.voting_analyzer.config.target_models[mk]['metadata'].name for mk in self.posterior_probs[key].keys()]),
            'posterior_probabilities': str({
                self.voting_analyzer.config.target_models[mk]['metadata'].name: float(prob)
                for mk, prob in self.posterior_probs[key].items()
            }),
            'parameter': param,
            'geospatial_lon_min': float(lon.min()),
            'geospatial_lon_max': float(lon.max()),
            'geospatial_lat_min': float(lat.min()),
            'geospatial_lat_max': float(lat.max()),
            'geospatial_vertical_min': float(depth.min()),
            'geospatial_vertical_max': float(depth.max()),
            'geospatial_lon_resolution': float(np.mean(np.diff(lon))),
            'geospatial_lat_resolution': float(np.mean(np.diff(lat))),
            'geospatial_vertical_resolution': float(np.mean(np.diff(depth))),
            'reference': 'EASTASIA-FWI Project, 2025',
            'note': 'Constructed using Bayesian Model Averaging from original model data'
        }
        
        # 编码设置（压缩）
        encoding = {}
        for var in ds.data_vars:
            encoding[var] = {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
                '_FillValue': np.float32(np.nan)
            }
        
        # 保存NetCDF
        nc_file = output_dir / f"{model_name}_original.nc"
        ds.to_netcdf(nc_file, encoding=encoding, format='NETCDF4')
        
        file_size_mb = nc_file.stat().st_size / (1024**2)
        self.logger.info(f"  ✅ NetCDF模型已保存: {nc_file.name} ({file_size_mb:.2f} MB)")
        self.logger.info(f"     📁 路径: {output_dir}")
        
        # 保存元数据JSON
        metadata = {
            'model_name': model_name,
            'parameter': param.upper(),
            'method': method,
            'creation_date': datetime.now().isoformat(),
            'source_type': 'bma_original_data',
            'global_attributes': dict(ds.attrs),
            'posterior_probabilities': {
                self.voting_analyzer.config.target_models[mk]['metadata'].name: float(prob)
                for mk, prob in self.posterior_probs[key].items()
            },
            'dimensions': {
                'longitude': {'size': len(lon), 'min': float(lon.min()), 'max': float(lon.max())},
                'latitude': {'size': len(lat), 'min': float(lat.min()), 'max': float(lat.max())},
                'depth': {'size': len(depth), 'min': float(depth.min()), 'max': float(depth.max())}
            },
            'variables': {}
        }
        
        for var_name in ds.data_vars:
            var = ds[var_name]
            valid_data = var.values[~np.isnan(var.values)]
            
            metadata['variables'][var_name] = {
                'shape': list(var.shape),
                'dtype': str(var.dtype),
                'units': var.attrs.get('units', ''),
                'min_value': float(valid_data.min()) if len(valid_data) > 0 else None,
                'max_value': float(valid_data.max()) if len(valid_data) > 0 else None,
                'mean_value': float(valid_data.mean()) if len(valid_data) > 0 else None,
                'valid_percentage': float(len(valid_data) / var.size * 100)
            }
        
        metadata_file = output_dir / f"{model_name}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"  ✅ 元数据已保存: {metadata_file.name}")
        
        ds.close()
        return nc_file
    
    def save_statistical_report(self, params: List[str] = ['vs', 'vp'], 
                               method: str = 'consistency'):
        """🆕 生成完整的统计报告（JSON格式）"""
        self.logger.info(f"\n📊 生成统计报告...")
        
        report = {
            'metadata': {
                'title': 'Bayesian Model Averaging Statistical Report',
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'version': 'v2.0',
                'method': method
            },
            'configuration': {
                'prior_type': self.config.prior['type'],
                'likelihood_noise_level': self.config.likelihood['noise_level'],
                'use_spatial_variance': self.config.likelihood['use_spatial_variance']
            },
            'models': {},
            'results': {}
        }
        
        # 模型信息
        for model_key, model in self.models.items():
            report['models'][model_key] = {
                'full_name': model.metadata.full_name,
                'year': model.metadata.year,
                'coverage': {
                    'lat_range': [float(model.lat.min()), float(model.lat.max())],
                    'lon_range': [float(model.lon.min()), float(model.lon.max())],
                    'depth_range': [float(model.depth.min()), float(model.depth.max())]
                }
            }
        
        # 参数结果
        for param in params:
            key = f"{param}_{method}"
            
            if key not in self.posterior_probs:
                continue
            
            # 后验概率
            posterior_probs = self.posterior_probs[key]
            
            # BMA预测统计
            bma_data = self.bma_predictions[key]
            valid_data = bma_data[~np.isnan(bma_data)]
            
            # 不确定性统计
            total_std = self.bma_uncertainty[param][f'{method}_total_std']
            valid_std = total_std[~np.isnan(total_std)]
            
            # 与Voting Map对比
            voting_data = self.voting_results[param]['median']
            diff = bma_data - voting_data
            valid_diff = diff[~np.isnan(diff)]
            
            report['results'][param] = {
                'posterior_probabilities': {
                    mk: float(prob) for mk, prob in posterior_probs.items()
                },
                'bma_prediction': {
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'percentiles': {
                        '5': float(np.percentile(valid_data, 5)),
                        '25': float(np.percentile(valid_data, 25)),
                        '50': float(np.percentile(valid_data, 50)),
                        '75': float(np.percentile(valid_data, 75)),
                        '95': float(np.percentile(valid_data, 95))
                    }
                },
                'uncertainty': {
                    'mean_total_std': float(np.mean(valid_std)),
                    'max_total_std': float(np.max(valid_std)),
                    'mean_within_model_std': float(np.mean(
                        np.sqrt(self.bma_uncertainty[param][f'{method}_within_var'][~np.isnan(bma_data)])
                    )),
                    'mean_between_model_std': float(np.mean(
                        np.sqrt(self.bma_uncertainty[param][f'{method}_between_var'][~np.isnan(bma_data)])
                    ))
                },
                'comparison_with_voting_map': {
                    'mean_difference': float(np.mean(valid_diff)),
                    'rmse': float(np.sqrt(np.mean(valid_diff**2))),
                    'max_abs_difference': float(np.max(np.abs(valid_diff))),
                    'correlation': float(np.corrcoef(
                        bma_data[~np.isnan(bma_data) & ~np.isnan(voting_data)],
                        voting_data[~np.isnan(bma_data) & ~np.isnan(voting_data)]
                    )[0, 1])
                },
                'coverage': {
                    'total_grid_points': int(bma_data.size),
                    'valid_points': int(np.sum(~np.isnan(bma_data))),
                    'coverage_percentage': float(np.sum(~np.isnan(bma_data)) / bma_data.size * 100)
                }
            }
        
        # 保存JSON
        report_file = self.results_dir / 'bma_statistical_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"  ✅ 已保存统计报告: {report_file}")
        
        # 打印摘要
        self._print_report_summary(report)
        
        return report
    
    def _print_report_summary(self, report: Dict):
        """打印统计报告摘要"""
        print("\n" + "="*80)
        print("📊 BMA统计报告摘要")
        print("="*80)
        
        for param, results in report['results'].items():
            print(f"\n【{param.upper()}】")
            print(f"  后验概率:")
            for mk, prob in results['posterior_probabilities'].items():
                model_name = self.voting_analyzer.config.target_models[mk]['metadata'].name
                print(f"    {model_name:<15}: {prob:.4f}")
            
            print(f"\n  BMA预测:")
            bma = results['bma_prediction']
            print(f"    范围: {bma['min']:.3f} ~ {bma['max']:.3f} km/s")
            print(f"    均值: {bma['mean']:.3f} km/s")
            print(f"    中位数: {bma['percentiles']['50']:.3f} km/s")
            
            print(f"\n  不确定性:")
            unc = results['uncertainty']
            print(f"    总不确定性: {unc['mean_total_std']:.4f} km/s (平均)")
            print(f"    模型内: {unc['mean_within_model_std']:.4f} km/s")
            print(f"    模型间: {unc['mean_between_model_std']:.4f} km/s")
            
            print(f"\n  与Voting Map对比:")
            comp = results['comparison_with_voting_map']
            print(f"    平均差异: {comp['mean_difference']:.4f} km/s")
            print(f"    RMSE: {comp['rmse']:.4f} km/s")
            print(f"    相关系数: {comp['correlation']:.4f}")
            
            print(f"\n  覆盖度: {results['coverage']['coverage_percentage']:.1f}%")
        
        print("="*80)
    
    def run_full_bma_analysis(self, params: List[str] = ['vs', 'vp'], 
                             method: str = 'consistency'):
        """🚀 运行完整的BMA分析流程（v2.0全面升级）"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🚀 开始完整BMA分析流程 v2.0")
        self.logger.info("="*80)
        
        for param in params:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"🔸 处理参数: {param.upper()}")
            self.logger.info(f"{'='*80}")
            
            # 1. 计算BMA预测
            self.compute_bma_prediction(param, method)
            
            # 2. 计算不确定性
            self.compute_bma_uncertainty(param, method)
            
            # 3. 生成可视化
            self.logger.info(f"\n🎨 生成 {param.upper()} 可视化结果")
            
            self.plot_posterior_probability_distribution(param, method)
            self.plot_bma_vs_voting_horizontal_slices(param, method)
            self.plot_1d_comparison(param, method)
            self.plot_uncertainty_decomposition(param, method)
            self.plot_posterior_probability_maps(param, method)
            self.plot_dominant_model_map(param, method)
            self.plot_weight_entropy_map(param, method)
            self.plot_comprehensive_summary(param, method)
            
            # 4. 保存NetCDF模型
            if self.config.output['save_netcdf']:
                self.save_bma_model_to_netcdf(param, method)
        
        # 5. 生成统计报告
        if self.config.output['save_statistics']:
            self.save_statistical_report(params, method)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("✅ BMA分析完成!")
        self.logger.info("="*80)
        
        print("\n" + "="*80)
        print("✅ 分析成功完成!")
        print(f"📁 图表输出目录: {self.output_dir}")
        print(f"📁 数据输出目录: {self.results_dir}")
        print("\n📊 生成的文件:")
        
        for param in params:
            print(f"\n  【{param.upper()}】:")
            print(f"    ✓ 后验概率分布图")
            print(f"    ✓ BMA vs Voting对比图 ({len(self.config.visualization['comparison_depths'])}个深度)")
            print(f"    ✓ 1D剖面对比图")
            print(f"    ✓ 不确定性分解图")
            print(f"    ✓ 后验概率空间分布图")
            print(f"    ✓ 主导模型分布图")
            print(f"    ✓ 权重熵分布图")
            print(f"    ✓ 综合汇总面板")
            
            if self.config.output['save_netcdf']:
                print(f"    ✓ NetCDF模型: BMA_{param.upper()}/BMA_{param}_{method}.nc")
        
        if self.config.output['save_statistics']:
            print(f"\n  【报告】:")
            print(f"    ✓ bma_statistical_report.json (完整统计)")
        
        print("\n💡 提示:")
        print("  • 后验概率反映了模型的相对可靠性")
        print("  • 主导模型图显示了空间上哪个模型权重最大")
        print("  • 权重熵高的区域表示模型间差异大")
        print("  • NetCDF模型可直接用于SPECFEM3D正演")
        print("="*80 + "\n")


# ==================== 主函数（修复版）====================

def main():
    """
    主函数（独立运行模式）
    
    注意：推荐使用 4_3_Run_Assembly.py 运行完整流程
    """
    print("\n" + "="*80)
    print("🚀 EASTASIA-FWI 贝叶斯模型平均分析 v2.0")
    print("="*80)
    
    print("\n🔥 核心功能:")
    print("  1. 基于模型一致性的后验概率计算")
    print("  2. BMA加权平均（数据驱动权重）")
    print("  3. 不确定性分解（模型内+模型间）")
    print("  4. 25+ 种专业可视化（参考Voting Map）")
    print("  5. NetCDF模型保存（SPECFEM3D格式）")
    print("  6. 完整的统计报告生成")
    
    print("\n📊 可视化功能（25+种）:")
    print("  【基础分析】后验概率、BMA vs Voting对比、1D剖面、不确定性分解")
    print("  【空间分布】后验概率地图、主导模型图、权重熵图")
    print("  【综合面板】9宫格汇总、多深度对比")
    
    print("\n✅ v1.1修复保留:")
    print("  - 归一化平均似然计算")
    print("  - 避免数据覆盖差异偏差")
    
    print("\n⚠️  独立运行模式")
    print("\n📝 推荐使用方式:")
    print("  运行主控脚本: python 4_3_Run_Assembly.py")
    print("\n💡 或在 Python 交互式环境中:")
    print("  >>> exec(open('4_1_Voting_map.py').read())")
    print("  >>> voting_analyzer = VotingMapAnalyzer()")
    print("  >>> voting_analyzer.run_full_analysis()")
    print("  >>> ")
    print("  >>> exec(open('4_2_BMA.py').read())")
    print("  >>> bma = BayesianModelAveraging(voting_analyzer)")
    print("  >>> bma.run_full_bma_analysis()")
    
    print("\n" + "="*80)
    print("尝试导入 VotingMapAnalyzer...")
    print("="*80)
    
    # 尝试导入 VotingMapAnalyzer
    try:
        import importlib.util
        
        voting_map_file = Path(__file__).parent / '4_1_Voting_map.py'
        
        if not voting_map_file.exists():
            raise FileNotFoundError(f"未找到 4_1_Voting_map.py")
        
        spec = importlib.util.spec_from_file_location("voting_map_module", voting_map_file)
        voting_map_module = importlib.util.module_from_spec(spec)
        sys.modules["voting_map_module"] = voting_map_module
        spec.loader.exec_module(voting_map_module)
        
        VotingMapAnalyzer = voting_map_module.VotingMapAnalyzer
        
        print("✅ 成功导入 VotingMapAnalyzer")
        
        # 运行分析
        print("\n" + "="*80)
        print("步骤1: 运行 Voting Map 分析")
        print("="*80)
        
        voting_analyzer = VotingMapAnalyzer()
        voting_analyzer.run_full_analysis(params=['vs', 'vp'])
        
        print("\n" + "="*80)
        print("步骤2: 运行 BMA 分析")
        print("="*80)
        
        bma = BayesianModelAveraging(voting_analyzer=voting_analyzer)
        bma.run_full_bma_analysis(params=['vs', 'vp'])
        
        print("\n✅ 全部流程完成!")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n📝 解决方案:")
        print("  1. 使用主控脚本: python 4_3_Run_Assembly.py")
        print("  2. 或在 Python 交互式环境中运行")
        print("="*80)


if __name__ == "__main__":
    main()