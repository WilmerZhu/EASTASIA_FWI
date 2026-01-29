"""
东亚地区台站网络分布和统计可视化模块
基于PyGMT绘制台站网络分布图和统计分析

功能特性：
- 台站网络分布地图（公开台站 + SUSTECH台站）
- 按网络分色显示
- 台站网络统计分析
- 时间序列台站部署分析
- 多维度统计图表
"""
import pygmt
import pandas as pd
from pathlib import Path
import logging
import sys
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
from matplotlib import cm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig


class StationPlottingConfig:
    """台站可视化配置参数"""
    
    def __init__(self):
        # 可视化配置
        self.visualization = {
            'dpi': 600,
            'pdf_dpi': 720,
            'figure_format': ['png', 'pdf'],
            'map_projection': "M15c",
            'max_legend_networks': 25
        }
        
        # 地图配置
        self.map = {
            'resolution': "f",
            'land_color': "lightgray",
            'water_color': "lightblue",
            'shoreline_pen': "0.1p",
            'area_threshold': 1000,
            'border_pen': "1/0.5p,black,-2-2"
        }
        
        # 台站标记配置
        self.markers = {
            'public_stations': {
                'symbol': "t0.1c",  # 三角形
                'pen': "0.1p,black"
            },
            'sustech_stations': {
                'symbol': "s0.05c",  # 方形
                'pen': "0.1p,black",
                'fill': "red"
            }
        }
        
        # 统计图配置
        self.statistics = {
            'figure_size': (16, 12),
            'timeline_figure_size': (16, 6),
            'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        }
        
        # 日志配置
        self.logging = {
            'level': 'INFO',
            'console_output': True
        }


class StationNetworkPlotter:
    """台站网络专业绘图类
    
    专门用于绘制地震台站网络分布图和统计分析图
    遵循EASTASIA-FWI项目的地震学可视化标准
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化台站网络绘图器
        
        Args:
            output_dir: 输出目录
        """
        # 加载基础配置
        self.base_config = BaseConfig()
        
        # 加载模块配置
        self.config = StationPlottingConfig()
        
        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.base_config.dirs['figures']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.StationPlotter',
            self.config.logging['level']
        )
        
        # 设置地图区域 - 使用基础配置中的东亚区域
        region_bounds = self.base_config.get_region_bounds()
        self.region = [
            region_bounds['lon_min'], region_bounds['lon_max'],
            region_bounds['lat_min'], region_bounds['lat_max']
        ]
        
        # 数据目录设置
        self.data_dir = self.base_config.dirs['project_root'] / '5_Visualization' / 'data_EastAsia'
        self.station_data_dir = self.data_dir / 'station'
        
        # 创建数据目录
        self.station_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置PyGMT配置
        self._setup_pygmt_config()
        
        # 生成颜色方案（40种颜色：tab20的20种 + tab20b的20种）
        tab20_colors = [cm.get_cmap('tab20')(i) for i in range(20)]
        tab20b_colors = [cm.get_cmap('tab20b')(i) for i in range(20)]
        all_colors = tab20_colors + tab20b_colors
        
        self.colors_combined = [
            f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            for r, g, b, _ in all_colors
        ]
        
        self.logger.info("📡 台站网络绘图器初始化完成")

    def _setup_pygmt_config(self):
        """设置PyGMT全局配置"""
        pygmt.config(
            MAP_FRAME_TYPE="plain",
            MAP_GRID_PEN_PRIMARY="0.3p,dimgrey",
            MAP_ANNOT_OBLIQUE="30",
            MAP_ANNOT_OFFSET_PRIMARY="5p",
            MAP_ANNOT_OFFSET_SECONDARY="5p",
            FONT_ANNOT_PRIMARY="10p,4",
            FONT_LABEL="10p,28,black",
            MAP_FRAME_WIDTH="2p",
            MAP_FRAME_PEN="0.5p",
            MAP_TICK_LENGTH_PRIMARY="5p",
            MAP_TICK_PEN_PRIMARY="0.5p,black,-",
            MAP_LABEL_OFFSET="5p",
        )

    def load_sustech_station_data(self) -> Optional[pd.DataFrame]:
        """加载SUSTECH台站数据"""
        try:
            sustech_file = self.station_data_dir / "SUSTECH_Datacoverage.xlsx"
            
            if not sustech_file.exists():
                self.logger.warning(f"SUSTECH台站数据文件不存在: {sustech_file}")
                return None
            
            # 读取SUSTECH台站数据
            sustech_data = pd.read_excel(
                sustech_file,
                sheet_name="Sheet1",
                usecols=["net", "station", "lon", "lat", "elev", "start", "end"],
                dtype={"start": str, "end": str}
            )
            
            # 数据处理
            sustech_df = pd.DataFrame({
                "Network": sustech_data["net"],
                "Station": sustech_data["station"],
                "Longitude": sustech_data["lon"],
                "Latitude": sustech_data["lat"],
                "Elevation": sustech_data["elev"],
                "StartTime": sustech_data["start"],
                "EndTime": sustech_data["end"],
                "DataSource": "SUSTECH"  # 标记数据源
            })
            
            # 时间数据处理
            sustech_df['StartTime'] = pd.to_datetime(sustech_df['StartTime'], errors='coerce')
            sustech_df['EndTime'] = pd.to_datetime(sustech_df['EndTime'], errors='coerce')
            sustech_df['StartYear'] = sustech_df['StartTime'].dt.year
            sustech_df['EndYear'] = sustech_df['EndTime'].dt.year
            
            # 地理边界过滤
            sustech_df = sustech_df[
                (sustech_df['Longitude'] >= self.region[0]) & 
                (sustech_df['Longitude'] <= self.region[1]) &
                (sustech_df['Latitude'] >= self.region[2]) & 
                (sustech_df['Latitude'] <= self.region[3])
            ]
            
            self.logger.info(f"📊 加载SUSTECH台站数据: {len(sustech_df)} 个台站")
            self.logger.info(f"   网络数量: {sustech_df['Network'].nunique()}")
            
            return sustech_df
            
        except Exception as e:
            self.logger.error(f"SUSTECH台站数据加载失败: {e}")
            return None

    def prepare_public_station_data(self, station_file: Optional[str] = None) -> pd.DataFrame:
        """准备公开台站数据"""
        try:
            if station_file and Path(station_file).exists():
                # 使用提供的文件
                df = pd.read_csv(station_file)
                self.logger.info(f"📊 加载公开台站数据: {station_file}")
            else:
                # 查找默认数据文件
                stations_dir = self.base_config.dirs['stations']
                possible_files = [
                    stations_dir / f"{self.base_config.region['name']}_stations.csv",
                    stations_dir / "EastAsia_stations.csv",
                    stations_dir / "stations.csv"
                ]
                
                df = None
                for file_path in possible_files:
                    if file_path.exists():
                        try:
                            df = pd.read_csv(file_path)
                            self.logger.info(f"📊 找到并加载公开台站数据: {file_path}")
                            break
                        except Exception as e:
                            self.logger.debug(f"文件 {file_path} 加载失败: {e}")
                            continue
                
                if df is None:
                    raise FileNotFoundError("未找到可用的公开台站数据文件")
            
            # 数据预处理
            self.logger.info(f"📈 原始公开台站数据: {len(df)} 个台站")
            
            # 标准化列名映射
            column_mapping = {
                'net': 'Network',
                'station': 'Station', 
                'lon': 'Longitude',
                'lat': 'Latitude',
                'elev': 'Elevation',
                'start': 'StartTime',
                'end': 'EndTime',
                'Provider': 'Network',  # 如果有Provider列，映射到Network
                'Code': 'Station'       # 如果有Code列，映射到Station
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name not in df.columns:
                    df[new_name] = df[old_name]
            
            # 确保必需列存在
            required_columns = ['Network', 'Station', 'Longitude', 'Latitude']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"缺少必需的列: {missing_columns}")
            
            # 数据类型转换和清理
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            
            # 过滤无效坐标
            df = df.dropna(subset=['Longitude', 'Latitude'])
            
            # 处理时间数据
            for time_col in ['StartTime', 'EndTime']:
                if time_col in df.columns:
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                    if time_col == 'StartTime':
                        df['StartYear'] = df[time_col].dt.year
                    elif time_col == 'EndTime':
                        df['EndYear'] = df[time_col].dt.year
            
            # 地理边界过滤
            df = df[
                (df['Longitude'] >= self.region[0]) & 
                (df['Longitude'] <= self.region[1]) &
                (df['Latitude'] >= self.region[2]) & 
                (df['Latitude'] <= self.region[3])
            ]
            
            self.logger.info(f"✅ 公开台站数据处理完成: {len(df)} 个台站")
            
            return df
            
        except Exception as e:
            self.logger.error(f"公开台站数据准备失败: {e}")
            return pd.DataFrame()

    def prepare_combined_station_data(self, public_station_file: Optional[str] = None) -> pd.DataFrame:
        """准备合并的台站数据（公开台站 + SUSTECH台站）"""
        try:
            # 1. 加载公开台站数据
            public_df = self.prepare_public_station_data(public_station_file)
            
            # 2. 加载SUSTECH台站数据
            sustech_df = self.load_sustech_station_data()
            
            # 3. 合并数据
            if sustech_df is not None and len(sustech_df) > 0:
                # 为公开台站添加数据源标记
                public_df['DataSource'] = 'Public'
                
                # 确保列名一致
                common_columns = ['Network', 'Station', 'Longitude', 'Latitude', 'DataSource']
                
                # 添加可选列
                optional_columns = ['Elevation', 'StartTime', 'EndTime', 'StartYear', 'EndYear']
                for col in optional_columns:
                    if col in public_df.columns and col in sustech_df.columns:
                        common_columns.append(col)
                
                # 选择共同列进行合并
                public_subset = public_df[common_columns].copy()
                sustech_subset = sustech_df[common_columns].copy()
                
                # 合并数据
                combined_df = pd.concat([public_subset, sustech_subset], ignore_index=True)
                
                self.logger.info(f"✅ 台站数据合并完成:")
                self.logger.info(f"   公开台站: {len(public_df)} 个")
                self.logger.info(f"   SUSTECH台站: {len(sustech_df)} 个")
                self.logger.info(f"   合并后总计: {len(combined_df)} 个台站")
                
            else:
                # 如果没有SUSTECH数据，只使用公开台站
                public_df['DataSource'] = 'Public'
                combined_df = public_df
                self.logger.info(f"📊 只使用公开台站数据: {len(combined_df)} 个台站")
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"台站数据合并失败: {e}")
            return pd.DataFrame()

    def generate_network_colors(self, df: pd.DataFrame) -> Tuple[Dict[str, str], pd.DataFrame]:
        """生成网络颜色映射"""
        try:
            # 统计网络台站数量
            station_counts = df['Network'].value_counts()
            
            # 按台站数量排序
            station_counts_sorted = station_counts.sort_values(ascending=False).reset_index()
            station_counts_sorted.columns = ["Network", "StationCount"]
            
            self.logger.info(f"🎨 独立台网数量: {len(station_counts_sorted)}")
            
            # 生成颜色映射
            color_map = {}
            for i, network in enumerate(station_counts_sorted["Network"]):
                color_map[network] = self.colors_combined[i % len(self.colors_combined)]
                self.logger.debug(f"  {i+1}. {network}: {station_counts_sorted.iloc[i]['StationCount']}个台站 | 颜色: {color_map[network]}")
            
            # 添加颜色到原始数据
            df['Color'] = df['Network'].map(color_map)
            
            return color_map, station_counts_sorted
            
        except Exception as e:
            self.logger.error(f"网络颜色生成失败: {e}")
            # 返回默认颜色
            color_map = {net: '#FF4500' for net in df['Network'].unique()}
            df['Color'] = '#FF4500'
            return color_map, pd.DataFrame()

    def create_network_legend(self, df: pd.DataFrame, color_map: Dict[str, str], 
                            station_counts: pd.DataFrame) -> str:
        """创建网络图例文件"""
        try:
            legend_file = self.output_dir / "station_network_legend.txt"
            
            # 准备时间信息
            net_time_info = {}
            if 'StartTime' in df.columns and 'EndTime' in df.columns:
                # 聚合网络时间信息
                time_grouped = df.groupby('Network').agg({
                    'StartTime': 'min',
                    'EndTime': 'max'
                }).reset_index()
                
                for _, row in time_grouped.iterrows():
                    start_str = row['StartTime'].strftime('%Y-%m-%d') if pd.notna(row['StartTime']) else '***'
                    end_str = row['EndTime'].strftime('%Y-%m-%d') if pd.notna(row['EndTime']) else '***'
                    net_time_info[row['Network']] = f"{start_str}~{end_str}"
            
            # 创建图例内容
            legend_content = """H 13.5p,Helvetica-Bold Network Info
N 3
G 0.2c
"""
            
            # 限制显示的网络数量（避免图例过长）
            max_networks = min(self.config.visualization['max_legend_networks'], len(station_counts))
            
            for i in range(max_networks):
                row = station_counts.iloc[i]
                network = row['Network']
                count = row['StationCount']
                color = color_map.get(network, '#FF4500')
                time_range = net_time_info.get(network, '***~***')
                
                legend_content += f"S 0.3c t 0.25c {color} 0.2p,black 0.8c {network} ({count}) {time_range}\n"
            
            # 如果有更多网络，添加说明
            if len(station_counts) > max_networks:
                remaining = len(station_counts) - max_networks
                legend_content += f"T ... and {remaining} more networks\n"
            
            # 保存图例文件
            with open(legend_file, 'w') as f:
                f.write(legend_content)
            
            self.logger.info(f"✅ 网络图例文件创建: {legend_file}")
            
            return str(legend_file)
            
        except Exception as e:
            self.logger.error(f"网络图例创建失败: {e}")
            return ""

    def plot_combined_station_network_map(self, 
                                        combined_data: pd.DataFrame,
                                        color_map: Dict[str, str],
                                        legend_file: str,
                                        save_file: str = "5-3-combined_station_network_distribution.png") -> str:
        """绘制合并的台站网络分布图（公开台站 + SUSTECH台站）"""
        
        self.logger.info(f"📡 开始绘制合并台站网络分布图...")
        
        try:
            # 创建图形
            fig = pygmt.Figure()
            
            # 绘制基础地图
            map_config = self.config.map
            fig.coast(
                resolution=map_config['resolution'],
                land=map_config['land_color'],
                water=map_config['water_color'],
                shorelines=map_config['shoreline_pen'],
                area_thresh=map_config['area_threshold'],
                region=self.region,
                projection=self.config.visualization['map_projection'],
                frame=["a10f1", "WseN"],
                borders=[map_config['border_pen']], 
            )
            
            # 分别绘制公开台站和SUSTECH台站
            public_stations = combined_data[combined_data['DataSource'] == 'Public']
            sustech_stations = combined_data[combined_data['DataSource'] == 'SUSTECH']
            
            # 1. 绘制公开台站（三角形）
            if len(public_stations) > 0:
                marker_config = self.config.markers['public_stations']
                for color, group in public_stations.groupby("Color"):
                    if len(group) > 0:
                        try:
                            fig.plot(
                                x=group["Longitude"],
                                y=group["Latitude"],
                                style=marker_config['symbol'],
                                pen=marker_config['pen'],
                                fill=color,
                            )
                        except Exception as e:
                            self.logger.warning(f"绘制公开台站颜色组 {color} 失败: {e}")
            
            # 2. 绘制SUSTECH台站（方形，红色标记）
            if len(sustech_stations) > 0:
                try:
                    marker_config = self.config.markers['sustech_stations']
                    fig.plot(
                        x=sustech_stations["Longitude"],
                        y=sustech_stations["Latitude"],
                        style=marker_config['symbol'],
                        pen=marker_config['pen'],
                        fill=marker_config['fill'],
                    )
                    self.logger.info(f"✅ 绘制SUSTECH台站: {len(sustech_stations)} 个")
                except Exception as e:
                    self.logger.warning(f"绘制SUSTECH台站失败: {e}")
            
            # 添加图例
            if legend_file and Path(legend_file).exists():
                try:
                    pygmt.config(FONT_ANNOT_PRIMARY="8p,5,black")
                    fig.legend(
                        spec=legend_file,
                        position="JBL+w15c/7c+jTM+o0c/-3.5c+l1.5",
                        box="+gwhite+p1p",
                    )
                    self.logger.info("✅ 网络图例已添加")
                except Exception as e:
                    self.logger.warning(f"图例添加失败: {e}")
            
            # 保存图片
            output_path = self.output_dir / save_file
            viz_config = self.config.visualization
            fig.savefig(str(output_path), dpi=viz_config['dpi'], crop=True)
            
            # 同时保存PDF版本
            if 'pdf' in viz_config['figure_format']:
                pdf_path = output_path.with_suffix('.pdf')
                fig.savefig(str(pdf_path), dpi=viz_config['pdf_dpi'], crop=True)
            
            self.logger.info(f"✅ 合并台站网络分布图保存至: {output_path}")
            self.logger.info(f"   公开台站: {len(public_stations)} 个（三角形）")
            self.logger.info(f"   SUSTECH台站: {len(sustech_stations)} 个（红色方形）")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"合并台站网络分布图绘制失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def plot_station_statistics(self, 
                               station_data: pd.DataFrame,
                               station_counts: pd.DataFrame,
                               save_file: str = "5-3-station_comprehensive_statistics.png") -> str:
        """绘制台站网络综合统计分析图"""
        
        self.logger.info(f"📊 开始绘制台站统计分析图...")
        
        try:
            # 设置matplotlib样式
            plt.style.use('default')
            
            # 创建2x2子图布局
            fig_size = self.config.statistics['figure_size']
            fig, axes = plt.subplots(2, 2, figsize=fig_size)
            fig.suptitle('Seismic Station Network Comprehensive Analysis', 
                        fontsize=18, fontweight='bold')
            
            # 颜色配置
            colors = self.config.statistics['colors']
            
            # 1. 数据源分布（饼图）
            self._plot_data_source_distribution(axes[0, 0], station_data, station_counts, colors)
            
            # 2. 前15个网络台站数量（条形图）
            self._plot_top_networks_bar(axes[0, 1], station_counts, colors)
            
            # 3. 地理分布散点图（按数据源着色）
            self._plot_geographic_distribution(axes[1, 0], station_data, colors)
            
            # 4. 海拔分布直方图或台站部署时间分布
            self._plot_elevation_or_deployment(axes[1, 1], station_data, colors)
            
            plt.tight_layout()
            
            # 保存图片
            output_path = self.output_dir / save_file
            plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ 台站统计分析图保存至: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"台站统计分析图绘制失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _plot_data_source_distribution(self, ax, station_data: pd.DataFrame, 
                                     station_counts: pd.DataFrame, colors: list):
        """绘制数据源分布饼图"""
        try:
            if 'DataSource' in station_data.columns:
                source_counts = station_data['DataSource'].value_counts()
                
                wedges, texts, autotexts = ax.pie(
                    source_counts.values, 
                    labels=source_counts.index,
                    autopct='%1.1f%%', 
                    colors=colors[:len(source_counts)], 
                    startangle=90
                )
                ax.set_title('Station Distribution by Data Source', fontweight='bold', pad=20)
                
                # 调整文字大小
                for autotext in autotexts:
                    autotext.set_fontsize(10)
            else:
                # 如果没有数据源信息，绘制网络分布
                top_networks = station_counts.head(8)
                other_count = station_counts.iloc[8:]['StationCount'].sum() if len(station_counts) > 8 else 0
                
                if other_count > 0:
                    plot_data = list(top_networks['StationCount']) + [other_count]
                    plot_labels = list(top_networks['Network']) + ['Others']
                else:
                    plot_data = list(top_networks['StationCount'])
                    plot_labels = list(top_networks['Network'])
                
                wedges, texts, autotexts = ax.pie(
                    plot_data, 
                    labels=plot_labels,
                    autopct='%1.1f%%', 
                    colors=colors[:len(plot_data)], 
                    startangle=90
                )
                ax.set_title('Station Distribution by Network', fontweight='bold', pad=20)
                
                for autotext in autotexts:
                    autotext.set_fontsize(8)
                    
        except Exception as e:
            self.logger.warning(f"饼图绘制失败: {e}")
            ax.text(0.5, 0.5, 'No Data Available', 
                   transform=ax.transAxes, ha='center')

    def _plot_top_networks_bar(self, ax, station_counts: pd.DataFrame, colors: list):
        """绘制前15个网络的条形图"""
        try:
            top_15_networks = station_counts.head(15)
            bars = ax.barh(
                range(len(top_15_networks)), 
                top_15_networks['StationCount'].values, 
                color=colors[0], 
                alpha=0.8
            )
            ax.set_yticks(range(len(top_15_networks)))
            ax.set_yticklabels(top_15_networks['Network'], fontsize=8)
            ax.set_title('Top 15 Networks by Station Count', fontweight='bold', pad=20)
            ax.set_xlabel('Number of Stations')
            ax.invert_yaxis()
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + max(top_15_networks['StationCount']) * 0.01, 
                       bar.get_y() + bar.get_height()/2, 
                       f'{int(width)}', ha='left', va='center', fontsize=8)
            
        except Exception as e:
            self.logger.warning(f"条形图绘制失败: {e}")
            ax.text(0.5, 0.5, 'No Network Data', 
                   transform=ax.transAxes, ha='center')

    def _plot_geographic_distribution(self, ax, station_data: pd.DataFrame, colors: list):
        """绘制地理分布散点图"""
        try:
            if 'DataSource' in station_data.columns:
                # 按数据源着色
                public_data = station_data[station_data['DataSource'] == 'Public']
                sustech_data = station_data[station_data['DataSource'] == 'SUSTECH']
                
                if len(public_data) > 0:
                    ax.scatter(
                        public_data['Longitude'], 
                        public_data['Latitude'],
                        c=colors[0], 
                        alpha=0.6, 
                        s=20,
                        marker='^',  # 三角形
                        label='Public Stations'
                    )
                
                if len(sustech_data) > 0:
                    ax.scatter(
                        sustech_data['Longitude'], 
                        sustech_data['Latitude'],
                        c='red', 
                        alpha=0.8, 
                        s=30,
                        marker='s',  # 方形
                        label='SUSTECH Stations'
                    )
            else:
                # 如果没有数据源信息，按网络着色
                unique_networks = station_data['Network'].unique()[:10]
                
                for i, network in enumerate(unique_networks):
                    network_data = station_data[station_data['Network'] == network]
                    if len(network_data) > 0:
                        ax.scatter(
                            network_data['Longitude'], 
                            network_data['Latitude'],
                            c=colors[i % len(colors)], 
                            alpha=0.6, 
                            s=20,
                            label=network
                        )
            
            ax.set_title('Geographic Distribution', fontweight='bold', pad=20)
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
        except Exception as e:
            self.logger.warning(f"地理分布图绘制失败: {e}")
            ax.text(0.5, 0.5, 'No Geographic Data', 
                   transform=ax.transAxes, ha='center')

    def _plot_elevation_or_deployment(self, ax, station_data: pd.DataFrame, colors: list):
        """绘制海拔分布或部署时间分布"""
        try:
            if 'Elevation' in station_data.columns:
                elevation_data = station_data['Elevation'].dropna()
                if len(elevation_data) > 0:
                    ax.hist(elevation_data, bins=30, alpha=0.7, 
                           color=colors[2], edgecolor='black', linewidth=0.5)
                    ax.set_title('Elevation Distribution', fontweight='bold', pad=20)
                    ax.set_xlabel('Elevation (m)')
                    ax.set_ylabel('Number of Stations')
                    ax.grid(True, alpha=0.3)
                    
                    # 添加统计信息
                    elev_stats = (
                        f'Mean: {elevation_data.mean():.0f}m\n'
                        f'Median: {elevation_data.median():.0f}m\n'
                        f'Range: {elevation_data.min():.0f}~{elevation_data.max():.0f}m'
                    )
                    ax.text(0.02, 0.98, elev_stats, transform=ax.transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
                    return
                    
        except:
            pass
        
        # 如果没有海拔数据，尝试绘制台站部署时间分布
        try:
            if 'StartYear' in station_data.columns:
                start_years = station_data['StartYear'].dropna()
                if len(start_years) > 0:
                    ax.hist(start_years, bins=20, alpha=0.7, 
                           color=colors[3], edgecolor='black', linewidth=0.5)
                    ax.set_title('Station Deployment Timeline', fontweight='bold', pad=20)
                    ax.set_xlabel('Start Year')
                    ax.set_ylabel('Number of Stations')
                    ax.grid(True, alpha=0.3)
                    return
        except:
            pass
            
        # 如果都没有数据
        ax.text(0.5, 0.5, 'No Additional Data', 
               transform=ax.transAxes, ha='center')

    def plot_station_deployment_timeline(self, 
                                       station_data: pd.DataFrame,
                                       save_file: str = "5-3-station_deployment_timeline.png") -> str:
        """绘制台站部署时间线分析"""
        
        self.logger.info(f"📅 开始绘制台站部署时间线...")
        
        try:
            if 'StartYear' not in station_data.columns:
                self.logger.warning("无台站部署时间数据")
                return ""
            
            # 设置matplotlib样式
            plt.style.use('default')
            
            # 创建1x2子图布局
            fig_size = self.config.statistics['timeline_figure_size']
            fig, axes = plt.subplots(1, 2, figsize=fig_size)
            fig.suptitle('Station Network Deployment Timeline Analysis', 
                        fontsize=16, fontweight='bold')
            
            # 颜色配置
            colors = self.config.statistics['colors']
            
            # 准备时间数据
            time_data = station_data.dropna(subset=['StartYear'])
            
            if len(time_data) == 0:
                self.logger.warning("没有有效的时间数据")
                return ""
            
            # 1. 年度台站部署数量（按数据源分组）
            self._plot_annual_deployment(axes[0], time_data, colors)
            
            # 2. 主要网络的部署时间线
            self._plot_network_timeline(axes[1], time_data, colors)
            
            plt.tight_layout()
            
            # 保存图片
            output_path = self.output_dir / save_file
            plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"✅ 台站部署时间线图保存至: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"台站部署时间线图绘制失败: {e}")
            return ""

    def _plot_annual_deployment(self, ax, time_data: pd.DataFrame, colors: list):
        """绘制年度台站部署图"""
        try:
            if 'DataSource' in time_data.columns:
                # 按数据源和年份分组
                public_data = time_data[time_data['DataSource'] == 'Public']
                sustech_data = time_data[time_data['DataSource'] == 'SUSTECH']
                
                if len(public_data) > 0:
                    public_yearly = public_data.groupby('StartYear').size()
                    ax.bar(public_yearly.index, public_yearly.values, 
                           alpha=0.7, color=colors[0], width=0.8, label='Public Stations')
                
                if len(sustech_data) > 0:
                    sustech_yearly = sustech_data.groupby('StartYear').size()
                    ax.bar(sustech_yearly.index, sustech_yearly.values, 
                           alpha=0.7, color='red', width=0.8, 
                           bottom=public_yearly.reindex(sustech_yearly.index, fill_value=0), 
                           label='SUSTECH Stations')
                    
                ax.legend()
            else:
                # 如果没有数据源信息，绘制总体部署
                yearly_deployment = time_data.groupby('StartYear').size()
                ax.bar(yearly_deployment.index, yearly_deployment.values, 
                       alpha=0.7, color=colors[0], width=0.8)
            
            ax.set_title('Annual Station Deployment', fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of Stations Deployed')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.warning(f"年度部署图绘制失败: {e}")

    def _plot_network_timeline(self, ax, time_data: pd.DataFrame, colors: list):
        """绘制网络部署时间线"""
        try:
            # 选择台站数量最多的前10个网络
            top_networks = time_data['Network'].value_counts().head(10).index
            
            for i, network in enumerate(top_networks):
                network_data = time_data[time_data['Network'] == network]
                yearly_count = network_data.groupby('StartYear').size()
                cumulative_count = yearly_count.cumsum()
                
                ax.plot(cumulative_count.index, cumulative_count.values, 
                       marker='o', markersize=3, linewidth=1.5,
                       color=colors[i % len(colors)], label=network)
            
            ax.set_title('Network Deployment Timeline (Top 10)', fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Cumulative Stations')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
        except Exception as e:
            self.logger.warning(f"网络时间线图绘制失败: {e}")

    def plot_all_station_analysis(self, public_station_file: Optional[str] = None) -> Dict[str, str]:
        """生成完整的台站网络分析图集（公开台站 + SUSTECH台站）"""
        
        self.logger.info("🎨 开始生成完整的台站网络分析图集...")
        
        results = {}
        
        try:
            # 准备合并的台站数据
            combined_data = self.prepare_combined_station_data(public_station_file)
            
            if len(combined_data) == 0:
                self.logger.warning("没有可用的台站数据")
                return results
            
            # 生成颜色映射和统计
            color_map, station_counts = self.generate_network_colors(combined_data)
            
            # 创建图例文件
            legend_file = self.create_network_legend(combined_data, color_map, station_counts)
            
            # 1. 合并台站网络分布图
            self.logger.info("  📡 生成合并台站网络分布图...")
            network_map = self.plot_combined_station_network_map(combined_data, color_map, legend_file)
            if network_map:
                results['combined_network_distribution'] = network_map
            
            # 2. 台站统计分析图
            self.logger.info("  📊 生成台站统计分析图...")
            statistics_plot = self.plot_station_statistics(combined_data, station_counts)
            if statistics_plot:
                results['comprehensive_statistics'] = statistics_plot
            
            # 3. 台站部署时间线图
            self.logger.info("  📅 生成台站部署时间线图...")
            timeline_plot = self.plot_station_deployment_timeline(combined_data)
            if timeline_plot:
                results['deployment_timeline'] = timeline_plot
            
            self.logger.info("✅ 台站网络分析图集生成完成!")
            
        except Exception as e:
            self.logger.error(f"台站网络分析图集生成失败: {e}")
            import traceback
            traceback.print_exc()
        
        return results


def find_station_data_file() -> Optional[str]:
    """查找台站数据文件"""
    try:
        base_config = BaseConfig()
        stations_dir = base_config.dirs['stations']
        
        # 支持多种文件格式和命名
        patterns = [
            f"{base_config.region['name']}_stations.csv",
            "EastAsia_stations.csv",
            "*_stations.csv",
            "stations.csv"
        ]
        
        for pattern in patterns:
            files = list(stations_dir.glob(pattern))
            if files:
                # 返回最新的文件
                latest_file = sorted(files, key=lambda x: x.stat().st_mtime)[-1]
                return str(latest_file)
        
        return None
        
    except Exception as e:
        logging.error(f"查找台站数据文件失败: {e}")
        return None


def main():
    """主函数 - 台站网络分布和统计可视化（公开台站 + SUSTECH台站）"""
    print("📡 EASTASIA-FWI 台站网络分布和统计可视化系统")
    print("   支持公开台站 + SUSTECH台站数据")
    print("="*70)
    
    try:
        # 查找台站数据文件
        public_station_file = find_station_data_file()
        
        print(f"📊 公开台站数据: {Path(public_station_file).name if public_station_file else '未找到'}")
        
        # 创建台站网络绘图器
        station_plotter = StationNetworkPlotter()
        
        # 检查SUSTECH数据文件
        sustech_file = station_plotter.station_data_dir / "SUSTECH_Datacoverage.xlsx"
        print(f"📊 SUSTECH台站数据: {sustech_file.name if sustech_file.exists() else '未找到'}")
        
        print("\n🎨 开始生成台站网络分析图集...")
        print("   公开台站: 三角形标记，按网络着色")
        print("   SUSTECH台站: 红色方形标记")
        
        # 生成完整分析图集
        results = station_plotter.plot_all_station_analysis(public_station_file)
        
        # 结果汇总
        print(f"\n🎉 台站网络分析图集生成完成!")
        print("="*70)
        print(f"📁 所有图片保存在: {station_plotter.output_dir}")
        print(f"\n📊 生成的图表:")
        
        if results:
            for analysis_type, file_path in results.items():
                print(f"  ✅ {analysis_type}: {Path(file_path).name}")
        else:
            print("  ⚠️  没有成功生成图表")
        
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断绘制")
    except Exception as e:
        print(f"❌ 台站网络可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()