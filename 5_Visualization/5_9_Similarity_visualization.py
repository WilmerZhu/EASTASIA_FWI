"""
EASTASIA-FWI 模型相似性分析可视化模块
====================================

本模块实现速度模型相似性分析结果的专业可视化：
- 相似性热图矩阵（SSIM、相关系数、RMS）
- 深度分层相似性曲线
- 深度切片对比图
- 统计分布对比图

基于 2_3_Model_similarity.py 的分析结果，生成发表级别的科学图表。

作者：EASTASIA-FWI Team
日期：2025-01
版本：v1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
import warnings
from datetime import datetime
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from config.base_config import BaseConfig

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class SimilarityVisualization:
    """模型相似性分析可视化类"""
    
    def __init__(self):
        """初始化可视化器"""
        self.base_config = BaseConfig()
        self.results_dir = Path('results/model_similarity')
        self.figures_dir = Path('figures/model_similarity')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # 可视化配置
        self.config = {
            'dpi': 300,
            'figsize_heatmap': (12, 10),
            'figsize_depth_curve': (14, 10),
            'figsize_depth_slice': (20, 6),
            'cmap_similarity': 'RdYlGn',
            'cmap_velocity': 'jet_r',
            'cmap_difference': 'RdBu_r',
        }
        
        print("="*80)
        print("EASTASIA-FWI 模型相似性可视化模块")
        print("="*80)
        print(f"结果目录: {self.results_dir}")
        print(f"图表目录: {self.figures_dir}")
        print("="*80)
    
    # ==================== 相似性热图 ====================
    
    def plot_similarity_heatmap(self, feature: str = 'vs', 
                               metric: str = 'ssim') -> Path:
        """
        绘制相似性热图矩阵
        
        Args:
            feature: 特征名（vs, vp）
            metric: 相似性指标（ssim, correlation, rms_difference）
        
        Returns:
            Path: 保存的图表路径
        """
        print(f"\n📊 生成相似性热图: {feature} - {metric}")
        
        # 读取所有对比结果
        feature_dir = self.results_dir / feature
        if not feature_dir.exists():
            raise FileNotFoundError(f"结果目录不存在: {feature_dir}")
        
        # 查找所有metrics文件
        metrics_files = list(feature_dir.glob('metrics_*_vs_*_*.json'))
        if not metrics_files:
            raise FileNotFoundError(f"未找到metrics文件: {feature_dir}")
        
        # 提取模型名称
        model_names = set()
        for file in metrics_files:
            parts = file.stem.replace('metrics_', '').replace(f'_{feature}', '').split('_vs_')
            model_names.update(parts)
        
        model_names = sorted(model_names)
        n_models = len(model_names)
        
        # 创建相似性矩阵
        similarity_matrix = np.eye(n_models)
        model_to_idx = {name: i for i, name in enumerate(model_names)}
        
        # 填充矩阵
        for file in metrics_files:
            with open(file, 'r') as f:
                data = json.load(f)
            
            parts = file.stem.replace('metrics_', '').replace(f'_{feature}', '').split('_vs_')
            model1, model2 = parts[0], parts[1]
            
            value = data['global_metrics'].get(metric, np.nan)
            
            i, j = model_to_idx[model1], model_to_idx[model2]
            similarity_matrix[i, j] = value
            similarity_matrix[j, i] = value
        
        # 绘制热图
        fig, ax = plt.subplots(figsize=self.config['figsize_heatmap'])
        
        # 根据指标类型选择colormap
        if metric == 'rms_difference':
            cmap = 'RdYlGn_r'  # RMS越小越好
            vmin, vmax = 0, np.nanmax(similarity_matrix[~np.eye(n_models, dtype=bool)])
        else:
            cmap = 'RdYlGn'  # SSIM/correlation越大越好
            vmin, vmax = np.nanmin(similarity_matrix[~np.eye(n_models, dtype=bool)]), 1.0
        
        # 绘制heatmap
        im = ax.imshow(similarity_matrix, cmap=cmap, vmin=vmin, vmax=vmax, 
                      aspect='auto', interpolation='nearest')
        
        # 添加数值标注
        for i in range(n_models):
            for j in range(n_models):
                value = similarity_matrix[i, j]
                if not np.isnan(value):
                    color = 'white' if (value - vmin) / (vmax - vmin) < 0.5 else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                           color=color, fontsize=10, fontweight='bold')
        
        # 设置刻度
        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        ax.set_xticklabels([name.replace('2022_', '').replace('2024_', '') 
                           for name in model_names], rotation=45, ha='right')
        ax.set_yticklabels([name.replace('2022_', '').replace('2024_', '') 
                           for name in model_names])
        
        # 标题和颜色条
        metric_names = {
            'ssim': 'SSIM (Structural Similarity Index)',
            'correlation': 'Pearson Correlation',
            'rms_difference': 'RMS Difference (km/s)'
        }
        ax.set_title(f'{metric_names.get(metric, metric)} - {feature.upper()}',
                    fontsize=16, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric_names.get(metric, metric), fontsize=12)
        
        # 网格线
        ax.set_xticks(np.arange(n_models) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_models) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        # 保存
        output_file = self.figures_dir / feature / f'heatmap_{metric}_{feature}.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 保存: {output_file}")
        return output_file
    
    # ==================== 深度分层相似性曲线 ====================
    
    def plot_depth_similarity_curves(self, feature: str = 'vs') -> List[Path]:
        """
        绘制深度分层相似性曲线
        
        Args:
            feature: 特征名
        
        Returns:
            List[Path]: 保存的图表路径列表
        """
        print(f"\n📈 生成深度相似性曲线: {feature}")
        
        feature_dir = self.results_dir / feature
        depth_files = list(feature_dir.glob('depth_metrics_*_vs_*_*.csv'))
        
        if not depth_files:
            print(f"  ⚠️  未找到深度metrics文件")
            return []
        
        output_files = []
        
        for depth_file in depth_files:
            # 读取数据
            df = pd.read_csv(depth_file)
            
            # 提取模型名称
            pair_name = depth_file.stem.replace('depth_metrics_', '').replace(f'_{feature}', '')
            model1, model2 = pair_name.split('_vs_')
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_depth_curve'])
            
            # 1. 相关系数 vs 深度
            ax1 = axes[0, 0]
            ax1.plot(df['correlation'], df['depth'], 'b-', linewidth=2)
            ax1.axvline(0.9, color='g', linestyle='--', alpha=0.5, label='High (0.9)')
            ax1.axvline(0.7, color='orange', linestyle='--', alpha=0.5, label='Medium (0.7)')
            ax1.set_xlabel('Correlation', fontsize=12)
            ax1.set_ylabel('Depth (km)', fontsize=12)
            ax1.set_title('Pearson Correlation', fontsize=14, fontweight='bold')
            ax1.invert_yaxis()
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. RMS差异 vs 深度
            ax2 = axes[0, 1]
            ax2.plot(df['rms_difference'], df['depth'], 'r-', linewidth=2)
            ax2.set_xlabel('RMS Difference (km/s)', fontsize=12)
            ax2.set_ylabel('Depth (km)', fontsize=12)
            ax2.set_title('RMS Difference', fontsize=14, fontweight='bold')
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3)
            
            # 3. 平均差异 ± 标准差
            ax3 = axes[1, 0]
            ax3.plot(df['mean_diff'], df['depth'], 'g-', linewidth=2, label='Mean')
            ax3.fill_betweenx(df['depth'], 
                             df['mean_diff'] - df['std_diff'],
                             df['mean_diff'] + df['std_diff'],
                             alpha=0.3, label='±1σ')
            ax3.axvline(0, color='k', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Velocity Difference (km/s)', fontsize=12)
            ax3.set_ylabel('Depth (km)', fontsize=12)
            ax3.set_title('Mean ± Std Difference', fontsize=14, fontweight='bold')
            ax3.invert_yaxis()
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 4. 有效数据点
            ax4 = axes[1, 1]
            ax4.plot(df['n_valid_points'], df['depth'], 'm-', linewidth=2)
            ax4.set_xlabel('Number of Valid Points', fontsize=12)
            ax4.set_ylabel('Depth (km)', fontsize=12)
            ax4.set_title('Data Coverage', fontsize=14, fontweight='bold')
            ax4.invert_yaxis()
            ax4.grid(True, alpha=0.3)
            
            # 总标题
            fig.suptitle(f'{model1} vs {model2} - {feature.upper()}\nDepth-wise Similarity',
                        fontsize=16, fontweight='bold', y=0.995)
            
            plt.tight_layout()
            
            # 保存
            output_file = self.figures_dir / feature / 'depth_curves' / f'depth_similarity_{pair_name}_{feature}.png'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            output_files.append(output_file)
            print(f"  ✅ 保存: {output_file.name}")
        
        return output_files
    
    # ==================== 深度切片对比图 ====================
    
    def plot_depth_slice_comparison(self, feature: str = 'vs', 
                                    depth_percentiles: List[int] = [10, 50, 90]) -> List[Path]:
        """
        绘制深度切片对比图
        
        Args:
            feature: 特征名
            depth_percentiles: 深度百分位（选择代表性深度）
        
        Returns:
            List[Path]: 保存的图表路径列表
        """
        print(f"\n🗺️  生成深度切片对比图: {feature}")
        
        feature_dir = self.results_dir / feature
        aligned_files = list(feature_dir.glob('aligned_data_*_vs_*_*.pkl'))
        
        if not aligned_files:
            print(f"  ⚠️  未找到aligned_data文件")
            return []
        
        output_files = []
        
        for aligned_file in aligned_files:
            # 加载对齐数据
            with open(aligned_file, 'rb') as f:
                aligned_data = pickle.load(f)
            
            data1 = aligned_data['data1']
            data2 = aligned_data['data2']
            depths = aligned_data['depths']
            
            # 选择深度切片
            depth_indices = [int(len(depths) * p / 100) for p in depth_percentiles]
            selected_depths = depths[depth_indices]
            
            # 提取模型名称
            pair_name = aligned_file.stem.replace('aligned_data_', '').replace(f'_{feature}', '')
            model1, model2 = pair_name.split('_vs_')
            
            # 创建图表
            fig, axes = plt.subplots(len(depth_indices), 3, 
                                    figsize=(self.config['figsize_depth_slice'][0],
                                            4 * len(depth_indices)))
            
            if len(depth_indices) == 1:
                axes = axes.reshape(1, -1)
            
            for i, (depth_idx, depth) in enumerate(zip(depth_indices, selected_depths)):
                slice1 = data1[:, :, depth_idx]
                slice2 = data2[:, :, depth_idx]
                diff = slice1 - slice2
                
                # 计算统计值
                valid_mask = (~np.isnan(slice1)) & (~np.isnan(slice2))
                if np.sum(valid_mask) == 0:
                    continue
                
                vmin = min(np.nanmin(slice1), np.nanmin(slice2))
                vmax = max(np.nanmax(slice1), np.nanmax(slice2))
                
                # 绘制模型1
                im1 = axes[i, 0].imshow(slice1.T, cmap=self.config['cmap_velocity'],
                                       vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
                axes[i, 0].set_title(f'{model1}\nDepth = {depth:.1f} km', 
                                    fontsize=12, fontweight='bold')
                plt.colorbar(im1, ax=axes[i, 0], label='Vs (km/s)')
                
                # 绘制模型2
                im2 = axes[i, 1].imshow(slice2.T, cmap=self.config['cmap_velocity'],
                                       vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
                axes[i, 1].set_title(f'{model2}\nDepth = {depth:.1f} km',
                                    fontsize=12, fontweight='bold')
                plt.colorbar(im2, ax=axes[i, 1], label='Vs (km/s)')
                
                # 绘制差异
                diff_max = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
                im3 = axes[i, 2].imshow(diff.T, cmap=self.config['cmap_difference'],
                                       vmin=-diff_max, vmax=diff_max, origin='lower', aspect='auto')
                axes[i, 2].set_title(f'Difference\nRMS = {np.sqrt(np.nanmean(diff**2)):.3f} km/s',
                                    fontsize=12, fontweight='bold')
                plt.colorbar(im3, ax=axes[i, 2], label='ΔVs (km/s)')
            
            fig.suptitle(f'{model1} vs {model2} - {feature.upper()}\nDepth Slice Comparison',
                        fontsize=16, fontweight='bold', y=0.995)
            
            plt.tight_layout()
            
            # 保存
            output_file = self.figures_dir / feature / 'depth_slices' / f'depth_slices_{pair_name}_{feature}.png'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            output_files.append(output_file)
            print(f"  ✅ 保存: {output_file.name}")
        
        return output_files
    
    # ==================== 综合报告生成 ====================
    
    def generate_complete_report(self, features: List[str] = ['vs', 'vp']):
        """
        生成完整的可视化报告
        
        Args:
            features: 要分析的特征列表
        """
        print("\n" + "="*80)
        print("📊 生成完整可视化报告")
        print("="*80)
        
        all_files = []
        
        for feature in features:
            print(f"\n处理特征: {feature.upper()}")
            print("-" * 80)
            
            # 1. 相似性热图
            for metric in ['ssim', 'correlation', 'rms_difference']:
                try:
                    file = self.plot_similarity_heatmap(feature, metric)
                    all_files.append(file)
                except Exception as e:
                    print(f"  ❌ {metric} 热图生成失败: {str(e)}")
            
            # 2. 深度相似性曲线
            try:
                files = self.plot_depth_similarity_curves(feature)
                all_files.extend(files)
            except Exception as e:
                print(f"  ❌ 深度曲线生成失败: {str(e)}")
            
            # 3. 深度切片对比
            try:
                files = self.plot_depth_slice_comparison(feature)
                all_files.extend(files)
            except Exception as e:
                print(f"  ❌ 深度切片生成失败: {str(e)}")
        
        print("\n" + "="*80)
        print(f"✨ 可视化完成！共生成 {len(all_files)} 张图表")
        print("="*80)
        print(f"\n📁 图表目录: {self.figures_dir}")
        print("\n生成的图表:")
        for file in all_files[:10]:  # 只显示前10个
            print(f"  • {file.relative_to(self.figures_dir)}")
        if len(all_files) > 10:
            print(f"  • ... 还有 {len(all_files) - 10} 个文件")


def main():
    """主函数"""
    print("="*80)
    print("EASTASIA-FWI 模型相似性可视化")
    print("="*80)
    
    # 创建可视化器
    visualizer = SimilarityVisualization()
    
    # 生成完整报告
    visualizer.generate_complete_report(features=['vs', 'vp'])
    
    print("\n" + "="*80)
    print("✅ 可视化任务完成！")
    print("="*80)


if __name__ == '__main__':
    main()