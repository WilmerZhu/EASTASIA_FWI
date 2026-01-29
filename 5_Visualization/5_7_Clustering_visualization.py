"""
EASTASIA-FWI 聚类结果可视化模块（优化版）
为速度模型聚类分析提供全面的可视化功能

主要功能:
- BIC曲线与最优K值可视化
- 轮廓系数分析图
- 聚类分布图（2D/3D空间分布）
- 地理空间分布
- 深度剖面可视化
- 速度特征分布
- 速度-深度剖面（地震学经典图）
- 聚类对比分析
- 交互式3D可视化

可视化类型:
- 静态图表（PNG/PDF）
- 交互式HTML图表
- 综合分析报告图

数据兼容性:
- 兼容2_2_Model_clustering.py的输出格式
- 支持summary.json + labels.npz格式
- 自动处理列名差异（depth vs depth(km)）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import pickle
import json
from datetime import datetime

# 科学可视化
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 3D可视化
from mpl_toolkits.mplot3d import Axes3D

# 交互式可视化
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly未安装，交互式可视化功能将不可用")

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ClusteringVisualizationConfig:
    """聚类可视化配置类"""
    
    def __init__(self):
        """初始化可视化配置"""
        
        # ==================== 基础配置 ====================
        self.style = {
            'figure_size': (14, 10),
            'dpi': 300,
            'font_size': 10,
            'title_size': 14,
            'label_size': 12,
            'legend_size': 9,
            'color_palette': 'husl',  # 'husl', 'Set2', 'tab10', 'viridis'
            'background_color': 'white',
            'grid_alpha': 0.3
        }
        
        # ==================== 图表类型配置 ====================
        self.plot_types = {
            # 评估指标可视化
            'bic_curve': {
                'enabled': True,
                'title': 'BIC Score vs Number of Clusters',
                'xlabel': 'Number of Clusters (K)',
                'ylabel': 'BIC Score (lower is better)',
                'show_optimal': True,
                'figsize': (10, 6)
            },
            
            # 轮廓系数分析
            'silhouette_analysis': {
                'enabled': True,
                'title': 'Silhouette Analysis',
                'show_avg_score': True,
                'show_cluster_sizes': True,
                'figsize': (12, 8)
            },
            
            # 聚类分布图（2D投影）
            'cluster_distribution_2d': {
                'enabled': True,
                'title': '2D Cluster Distribution (PCA Projection)',
                'projection_method': 'pca',  # 'pca', 'tsne', 'umap'
                'show_centers': True,
                'show_hulls': True,
                'figsize': (12, 10)
            },
            
            # 聚类分布图（3D）
            'cluster_distribution_3d': {
                'enabled': True,
                'title': '3D Cluster Distribution',
                'projection_method': 'pca',
                'figsize': (12, 10)
            },
            
            # 地理空间分布
            'spatial_distribution': {
                'enabled': True,
                'title': 'Spatial Distribution of Clusters',
                'view': 'map',  # 'map', 'cross_section'
                'show_boundaries': True,
                'figsize': (14, 10)
            },
            
            # 深度剖面
            'depth_profiles': {
                'enabled': True,
                'title': 'Depth Profiles by Cluster',
                'show_statistics': True,
                'figsize': (14, 8)
            },
            
            # 聚类大小分布
            'cluster_sizes': {
                'enabled': True,
                'title': 'Cluster Size Distribution',
                'plot_type': 'bar',  # 'bar', 'pie'
                'figsize': (10, 6)
            },
            
            # ⭐ 速度特征分布（新增）
            'velocity_distributions': {
                'enabled': True,
                'title': 'Velocity Feature Distributions',
                'figsize': (14, 6)
            },
            
            # ⭐ 速度-深度剖面（新增）
            'velocity_depth_profiles': {
                'enabled': True,
                'title': 'Velocity-Depth Profiles',
                'figsize': (14, 10)
            },
            
            # 交互式3D可视化
            'interactive_3d': {
                'enabled': True and PLOTLY_AVAILABLE,
                'title': 'Interactive 3D Cluster Visualization',
                'output_html': True
            }
        }
        
        # ==================== 输出配置 ====================
        self.output = {
            'formats': ['png'],  # ⭐ 简化为只输出PNG
            'save_individual': True,
            'save_combined': False,  # ⭐ 默认不生成综合图
            'combined_name': 'clustering_analysis_summary',
            'transparent_background': False,
            'bbox_inches': 'tight',
            'pad_inches': 0.1
        }
        
        # ==================== 交互式可视化配置 ====================
        self.interactive = {
            'plotly_template': 'plotly_white',
            'height': 800,
            'width': 1200,
            'show_legend': True,
            'hover_data': ['cluster', 'longitude', 'latitude', 'depth']
        }
        
        # ==================== 日志配置 ====================
        self.logging = {
            'level': 'INFO',
            'console_output': True
        }


class ClusteringResultsLoader:
    """聚类结果加载器（优化版）"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def load_summary(self, summary_file: Path) -> Dict[str, Any]:
        """加载summary.json"""
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            self.logger.info(f"✅ 加载Summary: {summary_file.name}")
            return summary
        except Exception as e:
            self.logger.error(f"❌ 加载Summary失败: {e}")
            raise
    
    def load_labels(self, labels_file: Path) -> Dict[str, np.ndarray]:
        """加载聚类标签（NPZ格式）"""
        try:
            data = np.load(labels_file, allow_pickle=True)
            labels_dict = {key: data[key] for key in data.keys()}
            self.logger.info(f"✅ 加载标签: {len(labels_dict)} 个聚类结果")
            return labels_dict
        except Exception as e:
            self.logger.error(f"❌ 加载标签失败: {e}")
            raise
    
    def load_model_data(self, model_csv: Path, n_samples: int) -> Optional[pd.DataFrame]:
        """
        加载原始模型数据并采样对齐
        
        Args:
            model_csv: 模型CSV文件
            n_samples: 需要的样本数（与聚类标签数量对齐）
        """
        try:
            df = pd.read_csv(model_csv)
            self.logger.info(f"✅ 加载模型数据: {len(df):,} 个点")
            
            # 数据采样对齐
            if len(df) > n_samples:
                self.logger.info(f"  📏 数据对齐: {len(df):,} → {n_samples:,}")
                np.random.seed(42)  # 使用相同的随机种子
                sample_indices = np.random.choice(len(df), n_samples, replace=False)
                df = df.iloc[sample_indices].reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ 加载模型数据失败: {e}")
            return None


class ClusteringVisualizer:
    """聚类结果可视化器（优化版）"""
    
    def __init__(self, config: ClusteringVisualizationConfig, logger):
        self.config = config
        self.logger = logger
        self.figures = {}
    
    def _get_depth_column(self, df: pd.DataFrame) -> str:
        """⭐ 智能获取深度列名"""
        if 'depth' in df.columns:
            return 'depth'
        elif 'depth(km)' in df.columns:
            return 'depth(km)'
        else:
            raise ValueError("数据中未找到depth或depth(km)列")
    
    def plot_bic_curve(
        self,
        bic_scores: List[float],
        n_clusters_tested: List[int],
        optimal_k: int,
        output_dir: Path
    ) -> Path:
        """绘制BIC曲线"""
        self.logger.info("📊 绘制BIC曲线")
        
        cfg = self.config.plot_types['bic_curve']
        fig, ax = plt.subplots(figsize=cfg['figsize'], dpi=self.config.style['dpi'])
        
        # 绘制BIC曲线
        ax.plot(n_clusters_tested, bic_scores, 'o-', linewidth=2, 
                markersize=8, label='BIC Score', color='steelblue')
        
        # 标记最优点
        if cfg['show_optimal']:
            optimal_idx = n_clusters_tested.index(optimal_k)
            ax.plot(optimal_k, bic_scores[optimal_idx], 'r*', 
                   markersize=20, label=f'Optimal K = {optimal_k}')
            ax.axvline(optimal_k, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel(cfg['xlabel'], fontsize=self.config.style['label_size'])
        ax.set_ylabel(cfg['ylabel'], fontsize=self.config.style['label_size'])
        ax.set_title(cfg['title'], fontsize=self.config.style['title_size'], 
                    fontweight='bold')
        ax.legend(fontsize=self.config.style['legend_size'])
        ax.grid(True, alpha=self.config.style['grid_alpha'])
        
        output_file = self._save_figure(fig, '5-7_bic_curve', output_dir)
        plt.close(fig)
        
        return output_file
    
    def plot_silhouette_analysis(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
        output_dir: Path
    ) -> Path:
        """绘制轮廓系数分析图"""
        from sklearn.metrics import silhouette_samples, silhouette_score
        
        self.logger.info("📊 绘制轮廓系数分析")
        
        cfg = self.config.plot_types['silhouette_analysis']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg['figsize'], 
                                       dpi=self.config.style['dpi'])
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(features, labels)
        sample_silhouette_values = silhouette_samples(features, labels)
        
        # 左图：轮廓系数分布
        y_lower = 10
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            cluster_silhouette_values = sample_silhouette_values[labels == i]
            cluster_silhouette_values.sort()
            
            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                             0, cluster_silhouette_values,
                             facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
            
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax1.set_title("Silhouette Plot for Each Cluster", 
                     fontsize=self.config.style['title_size'])
        ax1.set_xlabel("Silhouette Coefficient", 
                      fontsize=self.config.style['label_size'])
        ax1.set_ylabel("Cluster Label", fontsize=self.config.style['label_size'])
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--", 
                   label=f'Average: {silhouette_avg:.3f}')
        ax1.legend()
        
        # 右图：每个聚类的轮廓系数分布（箱线图）
        cluster_silhouettes = [sample_silhouette_values[labels == i] 
                              for i in range(n_clusters)]
        
        bp = ax2.boxplot(cluster_silhouettes, labels=range(n_clusters),
                        patch_artist=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.axhline(y=silhouette_avg, color="red", linestyle="--", 
                   label=f'Average: {silhouette_avg:.3f}')
        ax2.set_title("Silhouette Distribution by Cluster", 
                     fontsize=self.config.style['title_size'])
        ax2.set_xlabel("Cluster Label", fontsize=self.config.style['label_size'])
        ax2.set_ylabel("Silhouette Coefficient", 
                      fontsize=self.config.style['label_size'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self._save_figure(fig, '5-7_silhouette_analysis', output_dir)
        plt.close(fig)
        
        return output_file
    
    def plot_cluster_distribution_2d(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
        output_dir: Path,
        method: str = 'pca'
    ) -> Path:
        """绘制2D聚类分布图"""
        self.logger.info(f"📊 绘制2D聚类分布 (方法: {method})")
        
        cfg = self.config.plot_types['cluster_distribution_2d']
        fig, ax = plt.subplots(figsize=cfg['figsize'], 
                              dpi=self.config.style['dpi'])
        
        # 降维到2D
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            features_2d = reducer.fit_transform(features)
            var_explained = reducer.explained_variance_ratio_
            xlabel = f'PC1 ({var_explained[0]*100:.1f}%)'
            ylabel = f'PC2 ({var_explained[1]*100:.1f}%)'
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = reducer.fit_transform(features)
            xlabel, ylabel = 't-SNE 1', 't-SNE 2'
        
        # 绘制每个聚类
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            cluster_points = features_2d[labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      c=[colors[i]], label=f'Cluster {i}',
                      alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
            
            # 绘制聚类中心
            if cfg['show_centers']:
                center = cluster_points.mean(axis=0)
                ax.scatter(center[0], center[1], c='red', marker='X',
                          s=200, edgecolors='k', linewidth=2)
        
        ax.set_xlabel(xlabel, fontsize=self.config.style['label_size'])
        ax.set_ylabel(ylabel, fontsize=self.config.style['label_size'])
        ax.set_title(f'{cfg["title"]} ({method.upper()})', 
                    fontsize=self.config.style['title_size'], fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize=self.config.style['legend_size'])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self._save_figure(fig, f'5-7_cluster_distribution_2d_{method}', 
                                       output_dir)
        plt.close(fig)
        
        return output_file
    
    def plot_cluster_distribution_3d(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
        output_dir: Path
    ) -> Path:
        """绘制3D聚类分布图"""
        self.logger.info("📊 绘制3D聚类分布")
        
        cfg = self.config.plot_types['cluster_distribution_3d']
        fig = plt.figure(figsize=cfg['figsize'], dpi=self.config.style['dpi'])
        ax = fig.add_subplot(111, projection='3d')
        
        # PCA降维到3D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3, random_state=42)
        features_3d = pca.fit_transform(features)
        var_explained = pca.explained_variance_ratio_
        
        # 绘制每个聚类
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            cluster_points = features_3d[labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      cluster_points[:, 2],
                      c=[colors[i]], label=f'Cluster {i}',
                      alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
        
        ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', 
                     fontsize=self.config.style['label_size'])
        ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', 
                     fontsize=self.config.style['label_size'])
        ax.set_zlabel(f'PC3 ({var_explained[2]*100:.1f}%)', 
                     fontsize=self.config.style['label_size'])
        ax.set_title(cfg['title'], fontsize=self.config.style['title_size'], 
                    fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.15, 1), fontsize=self.config.style['legend_size'])
        
        plt.tight_layout()
        output_file = self._save_figure(fig, '5-7_cluster_distribution_3d', output_dir)
        plt.close(fig)
        
        return output_file
    
    def plot_spatial_distribution(
        self,
        coordinates: pd.DataFrame,
        labels: np.ndarray,
        n_clusters: int,
        output_dir: Path
    ) -> Path:
        """绘制聚类的地理空间分布"""
        self.logger.info("📊 绘制空间分布图")
        
        cfg = self.config.plot_types['spatial_distribution']
        fig = plt.figure(figsize=cfg['figsize'], dpi=self.config.style['dpi'])
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 准备数据
        df = coordinates.copy()
        df['cluster'] = labels
        
        # ⭐ 智能获取深度列名
        depth_col = self._get_depth_column(df)
        
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
        
        # 1. 平面分布图（经度-纬度）
        ax1 = fig.add_subplot(gs[0, :])
        for i in range(n_clusters):
            cluster_data = df[df['cluster'] == i]
            ax1.scatter(cluster_data['longitude'], cluster_data['latitude'],
                       c=[colors[i]], label=f'Cluster {i}',
                       alpha=0.6, s=10, edgecolors='none')
        
        ax1.set_xlabel('Longitude (°)', fontsize=self.config.style['label_size'])
        ax1.set_ylabel('Latitude (°)', fontsize=self.config.style['label_size'])
        ax1.set_title('Horizontal Distribution', 
                     fontsize=self.config.style['title_size'], fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                  fontsize=self.config.style['legend_size'])
        ax1.grid(True, alpha=0.3)
        
        # 2. 深度剖面（经度-深度）
        ax2 = fig.add_subplot(gs[1, 0])
        for i in range(n_clusters):
            cluster_data = df[df['cluster'] == i]
            ax2.scatter(cluster_data['longitude'], cluster_data[depth_col],
                       c=[colors[i]], label=f'Cluster {i}',
                       alpha=0.6, s=10, edgecolors='none')
        
        ax2.set_xlabel('Longitude (°)', fontsize=self.config.style['label_size'])
        ax2.set_ylabel('Depth (km)', fontsize=self.config.style['label_size'])
        ax2.set_title('Longitude-Depth Cross-Section', 
                     fontsize=self.config.style['title_size'])
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        
        # 3. 深度剖面（纬度-深度）
        ax3 = fig.add_subplot(gs[1, 1])
        for i in range(n_clusters):
            cluster_data = df[df['cluster'] == i]
            ax3.scatter(cluster_data['latitude'], cluster_data[depth_col],
                       c=[colors[i]], label=f'Cluster {i}',
                       alpha=0.6, s=10, edgecolors='none')
        
        ax3.set_xlabel('Latitude (°)', fontsize=self.config.style['label_size'])
        ax3.set_ylabel('Depth (km)', fontsize=self.config.style['label_size'])
        ax3.set_title('Latitude-Depth Cross-Section', 
                     fontsize=self.config.style['title_size'])
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Spatial Distribution of Clusters', 
                    fontsize=self.config.style['title_size']+2, 
                    fontweight='bold', y=0.98)
        
        output_file = self._save_figure(fig, '5-7_spatial_distribution', output_dir)
        plt.close(fig)
        
        return output_file
    
    def plot_depth_profiles(
        self,
        coordinates: pd.DataFrame,
        labels: np.ndarray,
        n_clusters: int,
        output_dir: Path
    ) -> Path:
        """绘制每个聚类的深度分布剖面"""
        self.logger.info("📊 绘制深度剖面")
        
        cfg = self.config.plot_types['depth_profiles']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg['figsize'], 
                                       dpi=self.config.style['dpi'])
        
        df = coordinates.copy()
        df['cluster'] = labels
        
        # ⭐ 智能获取深度列名
        depth_col = self._get_depth_column(df)
        
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
        
        # 1. 深度分布直方图
        depth_bins = np.linspace(df[depth_col].min(), df[depth_col].max(), 30)
        
        for i in range(n_clusters):
            cluster_depths = df[df['cluster'] == i][depth_col]
            ax1.hist(cluster_depths, bins=depth_bins, alpha=0.6, 
                    color=colors[i], label=f'Cluster {i}', edgecolor='black')
        
        ax1.set_xlabel('Depth (km)', fontsize=self.config.style['label_size'])
        ax1.set_ylabel('Frequency', fontsize=self.config.style['label_size'])
        ax1.set_title('Depth Distribution by Cluster', 
                     fontsize=self.config.style['title_size'])
        ax1.legend(fontsize=self.config.style['legend_size'])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 深度统计箱线图
        cluster_depths = [df[df['cluster'] == i][depth_col].values 
                         for i in range(n_clusters)]
        
        bp = ax2.boxplot(cluster_depths, labels=range(n_clusters),
                        patch_artist=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Cluster', fontsize=self.config.style['label_size'])
        ax2.set_ylabel('Depth (km)', fontsize=self.config.style['label_size'])
        ax2.set_title('Depth Statistics by Cluster', 
                     fontsize=self.config.style['title_size'])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = self._save_figure(fig, '5-7_depth_profiles', output_dir)
        plt.close(fig)
        
        return output_file
    
    def plot_cluster_sizes(
        self,
        labels: np.ndarray,
        n_clusters: int,
        output_dir: Path
    ) -> Path:
        """绘制聚类大小分布"""
        self.logger.info("📊 绘制聚类大小分布")
        
        cfg = self.config.plot_types['cluster_sizes']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg['figsize'], 
                                       dpi=self.config.style['dpi'])
        
        # 统计每个聚类的大小
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
        
        # 1. 柱状图
        bars = ax1.bar(unique_labels, counts, color=colors, alpha=0.7, 
                      edgecolor='black')
        ax1.set_xlabel('Cluster', fontsize=self.config.style['label_size'])
        ax1.set_ylabel('Number of Points', fontsize=self.config.style['label_size'])
        ax1.set_title('Cluster Size Distribution', 
                     fontsize=self.config.style['title_size'])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count):,}', ha='center', va='bottom', fontsize=8)
        
        # 2. 饼图
        ax2.pie(counts, labels=[f'Cluster {i}' for i in unique_labels],
               colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Cluster Size Distribution (%)', 
                     fontsize=self.config.style['title_size'])
        
        plt.tight_layout()
        output_file = self._save_figure(fig, '5-7_cluster_sizes', output_dir)
        plt.close(fig)
        
        return output_file

    # ==================== ⭐ 新增可视化方法 ====================
    
    def plot_velocity_distributions(
        self,
        features: np.ndarray,
        feature_names: List[str],
        labels: np.ndarray,
        n_clusters: int,
        output_dir: Path
    ) -> Path:
        """
        绘制每个聚类的速度特征分布
        
        适用于地震速度模型：Vp, Vs, 各向异性等
        """
        self.logger.info("📊 绘制速度特征分布")
        
        n_features = len(feature_names)
        fig, axes = plt.subplots(1, n_features, figsize=(6*n_features, 6), 
                                dpi=self.config.style['dpi'])
        
        if n_features == 1:
            axes = [axes]
        
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
        
        for idx, (ax, feature_name) in enumerate(zip(axes, feature_names)):
            # 为每个聚类绘制特征分布
            for i in range(n_clusters):
                cluster_feature = features[labels == i, idx]
                ax.hist(cluster_feature, bins=50, alpha=0.6, 
                       color=colors[i], label=f'Cluster {i}',
                       edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel(f'{feature_name} Value (km/s)', 
                         fontsize=self.config.style['label_size'])
            ax.set_ylabel('Frequency', fontsize=self.config.style['label_size'])
            ax.set_title(f'{feature_name.upper()} Distribution by Cluster', 
                        fontsize=self.config.style['title_size'])
            ax.legend(fontsize=self.config.style['legend_size']-2, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Velocity Feature Distributions', 
                    fontsize=self.config.style['title_size']+2, 
                    fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_file = self._save_figure(fig, '5-7_velocity_distributions', output_dir)
        plt.close(fig)
        
        return output_file
    
    def plot_velocity_depth_profiles(
        self,
        coordinates: pd.DataFrame,
        features: np.ndarray,
        feature_names: List[str],
        labels: np.ndarray,
        n_clusters: int,
        output_dir: Path
    ) -> Path:
        """
        绘制速度-深度剖面（地震学经典图）
        
        每个聚类用不同颜色，展示Vp/Vs随深度的变化
        """
        self.logger.info("📊 绘制速度-深度剖面")
        
        # ⭐ 智能获取深度列名
        depth_col = self._get_depth_column(coordinates)
        
        n_features = len(feature_names)
        fig, axes = plt.subplots(1, n_features, figsize=(7*n_features, 10), 
                                dpi=self.config.style['dpi'])
        
        if n_features == 1:
            axes = [axes]
        
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
        
        for idx, (ax, feature_name) in enumerate(zip(axes, feature_names)):
            for i in range(n_clusters):
                cluster_mask = labels == i
                cluster_depths = coordinates[cluster_mask][depth_col].values
                cluster_values = features[cluster_mask, idx]
                
                # 绘制散点
                ax.scatter(cluster_values, cluster_depths, 
                          c=[colors[i]], alpha=0.3, s=5, 
                          label=f'Cluster {i}', edgecolors='none')
                
                # 计算并绘制平均趋势线（分层统计）
                depth_bins = np.arange(0, min(cluster_depths.max(), 1000), 50)  # 每50km一层
                bin_means = []
                bin_centers = []
                
                for j in range(len(depth_bins)-1):
                    mask = (cluster_depths >= depth_bins[j]) & (cluster_depths < depth_bins[j+1])
                    if np.sum(mask) > 10:  # 至少10个点
                        bin_means.append(np.mean(cluster_values[mask]))
                        bin_centers.append((depth_bins[j] + depth_bins[j+1]) / 2)
                
                if bin_means:
                    ax.plot(bin_means, bin_centers, '-', color=colors[i], 
                           linewidth=2, alpha=0.8)
            
            ax.set_xlabel(f'{feature_name.upper()} (km/s)', 
                         fontsize=self.config.style['label_size'])
            ax.set_ylabel('Depth (km)', fontsize=self.config.style['label_size'])
            ax.set_title(f'{feature_name.upper()} vs Depth', 
                        fontsize=self.config.style['title_size'])
            ax.invert_yaxis()  # ⭐ 深度向下
            ax.legend(fontsize=self.config.style['legend_size']-2, loc='best')
            ax.grid(True, alpha=0.3)
            
            # 添加地质层位参考线（可选）
            crust_depth = 40  # 地壳-地幔界面
            if cluster_depths.max() > crust_depth:
                ax.axhline(y=crust_depth, color='gray', linestyle='--', 
                          alpha=0.5, linewidth=1.5, label='Moho (~40km)')
        
        plt.suptitle('Velocity-Depth Profiles by Cluster', 
                    fontsize=self.config.style['title_size']+2, 
                    fontweight='bold', y=0.99)
        plt.tight_layout()
        
        output_file = self._save_figure(fig, '5-7_velocity_depth_profiles', output_dir)
        plt.close(fig)
        
        return output_file
    
    def create_interactive_3d(
        self,
        features: np.ndarray,
        coordinates: pd.DataFrame,
        labels: np.ndarray,
        n_clusters: int,
        output_dir: Path,
        model_name: str
    ) -> Optional[Path]:
        """创建交互式3D可视化（Plotly）"""
        if not PLOTLY_AVAILABLE:
            self.logger.warning("⚠️  Plotly未安装，跳过交互式可视化")
            return None
        
        self.logger.info("📊 创建交互式3D可视化")
        
        cfg = self.config.interactive
        
        # ⭐ 智能获取深度列名
        depth_col = self._get_depth_column(coordinates)
        
        # PCA降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3, random_state=42)
        features_3d = pca.fit_transform(features)
        
        # 准备数据
        df_plot = pd.DataFrame({
            'PC1': features_3d[:, 0],
            'PC2': features_3d[:, 1],
            'PC3': features_3d[:, 2],
            'cluster': labels,
            'longitude': coordinates['longitude'].values,
            'latitude': coordinates['latitude'].values,
            'depth': coordinates[depth_col].values
        })
        
        # 创建3D散点图
        fig = px.scatter_3d(
            df_plot,
            x='PC1', y='PC2', z='PC3',
            color='cluster',
            hover_data=['longitude', 'latitude', 'depth'],
            title=f'Interactive 3D Cluster Visualization - {model_name}',
            labels={'cluster': 'Cluster ID'},
            color_continuous_scale='viridis',
            template=cfg['plotly_template']
        )
        
        fig.update_traces(marker=dict(size=3, opacity=0.7))
        fig.update_layout(
            height=cfg['height'],
            width=cfg['width'],
            showlegend=cfg['show_legend']
        )
        
        # 保存HTML
        output_file = output_dir / f'5-7_{model_name}_interactive_3d.html'
        fig.write_html(str(output_file))
        
        self.logger.info(f"✅ 交互式可视化已保存: {output_file.name}")
        return output_file
    
    def _save_figure(
        self,
        fig: plt.Figure,
        name: str,
        output_dir: Path
    ) -> Path:
        """保存图表到文件"""
        saved_files = []
        
        for fmt in self.config.output['formats']:
            output_file = output_dir / f"{name}.{fmt}"
            fig.savefig(
                output_file,
                format=fmt,
                dpi=self.config.style['dpi'],
                bbox_inches=self.config.output['bbox_inches'],
                pad_inches=self.config.output['pad_inches'],
                transparent=self.config.output['transparent_background']
            )
            saved_files.append(output_file)
        
        return saved_files[0] if saved_files else None


# ==================== 完整可视化管道（优化版）====================

class ClusteringVisualizationPipeline:
    """聚类可视化完整管道（优化版）"""
    
    def __init__(self):
        """初始化可视化管道"""
        # 加载基础配置
        self.base_config = BaseConfig()
        
        # 加载可视化配置
        self.config = ClusteringVisualizationConfig()
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.ClusteringVisualization',
            self.config.logging['level']
        )
        
        # 初始化组件
        self.loader = ClusteringResultsLoader(self.logger)
        self.visualizer = ClusteringVisualizer(self.config, self.logger)
        
        # 设置输出目录
        self.output_base = self.base_config.get_module_dir('model_clustering') / 'figures'
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("🎨 聚类可视化管道初始化完成")
    
    def visualize_results(
        self,
        summary_file: Path,
        model_data_dir: Path
    ) -> Dict[str, List[Path]]:
        """
        可视化聚类结果（优化版）
        
        Args:
            summary_file: summary.json文件路径
            model_data_dir: 模型数据目录（包含标准化CSV）
            
        Returns:
            生成的图表文件路径字典
        """
        try:
            self.logger.info("=" * 70)
            self.logger.info(f"🎨 开始可视化: {summary_file.name}")
            self.logger.info("=" * 70)
            
            # 1. 加载summary
            summary = self.loader.load_summary(summary_file)
            model_name = summary['model_name']
            
            # 2. ⭐ 加载labels（NPZ格式）
            labels_dir = summary_file.parent.parent / 'labels'
            labels_file = labels_dir / f"{model_name}_labels.npz"
            
            if not labels_file.exists():
                self.logger.error(f"❌ 标签文件不存在: {labels_file}")
                return {}
            
            labels_data = self.loader.load_labels(labels_file)
            
            # 获取最优聚类的标签
            optimal_result = summary.get('optimal_result')
            if not optimal_result:
                self.logger.error("❌ Summary中未找到optimal_result")
                return {}
            
            optimal_k = optimal_result.get('k')
            optimal_algo = optimal_result.get('algorithm', 'gmm_umap')
            
            # 尝试找到对应的标签
            labels_key = f'{optimal_algo}_k{optimal_k}'
            if labels_key not in labels_data:
                # 尝试其他键
                self.logger.warning(f"⚠️  未找到 {labels_key}，使用第一个可用标签")
                labels_key = list(labels_data.keys())[0]
            
            labels = labels_data[labels_key]
            n_clusters = len(np.unique(labels))
            
            self.logger.info(f"📊 聚类算法: {optimal_algo}, K={optimal_k}")
            self.logger.info(f"📊 实际聚类数: {n_clusters}, 样本数: {len(labels)}")
            
            # 3. ⭐ 加载原始模型数据
            model_csv = model_data_dir / f"{model_name}.csv"
            if not model_csv.exists():
                # 尝试其他命名
                model_csv = model_data_dir / f"{model_name}_standardized.csv"
            
            if not model_csv.exists():
                self.logger.warning(f"⚠️  模型数据不存在: {model_csv}")
                self.logger.warning("  仅生成基础可视化")
                df_model = None
            else:
                df_model = self.loader.load_model_data(model_csv, len(labels))
            
            # 创建输出目录
            model_output_dir = self.output_base / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            generated_figures = {}
            
            # 4. ⭐ 基础可视化（不需要原始数据）
            
            # 4.1 聚类大小分布
            if self.config.plot_types['cluster_sizes']['enabled']:
                self.logger.info("  📊 生成聚类大小分布图...")
                fig_file = self.visualizer.plot_cluster_sizes(
                    labels, n_clusters, model_output_dir
                )
                generated_figures['cluster_sizes'] = [fig_file]
            
            # 5. ⭐ 高级可视化（需要原始数据）
            if df_model is not None:
                # 提取坐标
                coordinates = df_model[['longitude', 'latitude', 'depth']].copy()
                
                # 提取特征
                feature_names = summary['data_info'].get('selected_features', ['vp', 'vs'])
                
                # 验证特征是否存在
                available_features = [f for f in feature_names if f in df_model.columns]
                if not available_features:
                    self.logger.error(f"❌ 模型数据中未找到特征列: {feature_names}")
                    return generated_figures
                
                features = df_model[available_features].values
                
                self.logger.info(f"📊 特征: {available_features}, 形状: {features.shape}")
                
                # 5.1 空间分布
                if self.config.plot_types['spatial_distribution']['enabled']:
                    self.logger.info("  📊 生成空间分布图...")
                    fig_file = self.visualizer.plot_spatial_distribution(
                        coordinates, labels, n_clusters, model_output_dir
                    )
                    generated_figures['spatial_distribution'] = [fig_file]
                
                # 5.2 深度剖面
                if self.config.plot_types['depth_profiles']['enabled']:
                    self.logger.info("  📊 生成深度剖面图...")
                    fig_file = self.visualizer.plot_depth_profiles(
                        coordinates, labels, n_clusters, model_output_dir
                    )
                    generated_figures['depth_profiles'] = [fig_file]
                
                # 5.3 ⭐ 速度特征分布（新增）
                if self.config.plot_types['velocity_distributions']['enabled']:
                    self.logger.info("  📊 生成速度特征分布图...")
                    fig_file = self.visualizer.plot_velocity_distributions(
                        features, available_features, labels, n_clusters, model_output_dir
                    )
                    generated_figures['velocity_distributions'] = [fig_file]
                
                # 5.4 ⭐ 速度-深度剖面（新增）
                if self.config.plot_types['velocity_depth_profiles']['enabled']:
                    self.logger.info("  📊 生成速度-深度剖面图...")
                    fig_file = self.visualizer.plot_velocity_depth_profiles(
                        coordinates, features, available_features, labels, 
                        n_clusters, model_output_dir
                    )
                    generated_figures['velocity_depth_profiles'] = [fig_file]
                
                # 5.5 轮廓系数分析
                if self.config.plot_types['silhouette_analysis']['enabled']:
                    self.logger.info("  📊 生成轮廓系数分析图...")
                    fig_file = self.visualizer.plot_silhouette_analysis(
                        features, labels, n_clusters, model_output_dir
                    )
                    generated_figures['silhouette_analysis'] = [fig_file]
                
                # 5.6 2D聚类分布
                if self.config.plot_types['cluster_distribution_2d']['enabled']:
                    self.logger.info("  📊 生成2D聚类分布图...")
                    fig_file = self.visualizer.plot_cluster_distribution_2d(
                        features, labels, n_clusters, model_output_dir, method='pca'
                    )
                    generated_figures['cluster_distribution_2d'] = [fig_file]
                
                # 5.7 3D聚类分布
                if self.config.plot_types['cluster_distribution_3d']['enabled']:
                    self.logger.info("  📊 生成3D聚类分布图...")
                    fig_file = self.visualizer.plot_cluster_distribution_3d(
                        features, labels, n_clusters, model_output_dir
                    )
                    generated_figures['cluster_distribution_3d'] = [fig_file]
                
                # 5.8 交互式3D（可选）
                if self.config.plot_types['interactive_3d']['enabled']:
                    self.logger.info("  📊 生成交互式3D可视化...")
                    fig_file = self.visualizer.create_interactive_3d(
                        features, coordinates, labels, n_clusters, 
                        model_output_dir, model_name
                    )
                    if fig_file:
                        generated_figures['interactive_3d'] = [fig_file]
            
            self.logger.info("=" * 70)
            self.logger.info(f"✅ 可视化完成: 生成 {sum(len(v) for v in generated_figures.values())} 个图表")
            self.logger.info(f"📁 保存位置: {model_output_dir}")
            self.logger.info("=" * 70)
            
            return generated_figures
            
        except Exception as e:
            self.logger.error(f"❌ 可视化失败: {e}")
            import traceback
            traceback.print_exc()
            raise


# ==================== 主函数（直接配置版）====================

def main():
    """
    主函数 - 直接配置版本（优化版）
    
    在此处直接修改配置参数，无需命令行参数
    """
    
    # ============================================================
    # 🎯 配置参数（在此直接修改）
    # ============================================================
    
    # 模型名称（不含扩展名）
    MODEL_NAME = '2022_SinoScope1.0_standardized'
    
    # 自动构建路径
    RESULTS_DIR = Path('results/model_clustering')
    SUMMARY_FILE = RESULTS_DIR / 'clustering_results' / f'{MODEL_NAME}_summary.json'
    MODEL_DATA_DIR = Path('data/fwi-models/processed/2022_SinoScope1.0')
    
    # ============================================================
    
    print("\n" + "=" * 80)
    print("EASTASIA-FWI 聚类结果可视化（优化版）")
    print("=" * 80)
    print(f"\n📊 模型: {MODEL_NAME}")
    print(f"📂 Summary文件: {SUMMARY_FILE}")
    print(f"📂 模型数据目录: {MODEL_DATA_DIR}")
    print(f"\n🎨 可视化内容:")
    print("  ✅ 聚类大小分布")
    print("  ✅ 空间分布图（3视图）")
    print("  ✅ 深度剖面分析")
    print("  ✅ 速度特征分布")
    print("  ✅ 速度-深度剖面（地震学经典图）")
    print("  ✅ 轮廓系数分析")
    print("  ✅ 2D/3D聚类分布")
    print("  ✅ 交互式3D可视化（HTML）\n")
    
    try:
        # 初始化管道
        pipeline = ClusteringVisualizationPipeline()
        
        # 检查文件
        if not SUMMARY_FILE.exists():
            print(f"❌ Summary文件不存在: {SUMMARY_FILE}")
            print(f"\n💡 提示: 请先运行 2_2_Model_clustering.py 生成聚类结果")
            print(f"   预期位置: {SUMMARY_FILE.absolute()}")
            sys.exit(1)
        
        if not MODEL_DATA_DIR.exists():
            print(f"❌ 模型数据目录不存在: {MODEL_DATA_DIR}")
            print(f"   预期位置: {MODEL_DATA_DIR.absolute()}")
            sys.exit(1)
        
        # 执行可视化
        generated = pipeline.visualize_results(SUMMARY_FILE, MODEL_DATA_DIR)
        
        if not generated:
            print(f"\n⚠️  未生成任何图表，请检查日志")
            sys.exit(1)
        
        print(f"\n✅ 可视化完成!")
        print(f"📊 生成图表总数: {sum(len(v) for v in generated.values())}")
        print(f"📁 保存位置: {pipeline.output_base / MODEL_NAME}")
        print(f"\n📋 生成的图表:")
        for plot_type, files in generated.items():
            print(f"  - {plot_type}: {len(files)} 个文件")
            for file in files:
                print(f"    └─ {file.name}")
        print()
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
        print("\n💡 提示:")
        print("   1. 确认已运行 2_2_Model_clustering.py 生成聚类结果")
        print("   2. 检查文件路径是否正确")
        print("   3. 确认模型数据已标准化")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()