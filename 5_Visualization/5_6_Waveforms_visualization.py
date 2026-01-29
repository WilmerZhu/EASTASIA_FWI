"""
波形数据可视化模块 - 选择性事件可视化
允许用户选择特定事件进行详细的波形分析和可视化
支持震源球绘制和东亚地区地图显示
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from typing import List, Dict, Optional
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig

# ObsPy imports
from obspy import read
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel
from obspy.core import UTCDateTime
from obspy.imaging.beachball import beach


# Cartopy imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class WaveformVisualizationConfig:
    """波形可视化配置类"""
    
    def __init__(self):
        """初始化波形可视化配置参数"""
        # ============ 运行参数配置 ============
        self.run_params = {
            # 事件选择配置
            'selected_event_indices': [2],  # 要可视化的事件序号列表，如 [0, 1, 2] 或 [2]
            
            # 波形分量选择: 'Z' (垂直), 'N' (北向), 'E' (东向), 'All' (所有分量)
            'component': 'Z',
            
            # 显示选项配置
            'display_option': {
                'type': 'all',  # 'limit' | 'distance' | 'magnitude' | 'all'
                'max_traces': 30,  # 当 type='limit' 时有效
                'min_dist': 0,  # 当 type='distance' 时有效 (度)
                'max_dist': 180,  # 当 type='distance' 时有效 (度)
                'min_mag': 5.0,  # 当 type='magnitude' 时有效
                'max_mag': 10.0  # 当 type='magnitude' 时有效
            },
            
            # 速度模型选择: 'prem' | 'ak135' | 'iasp91'
            'velocity_model': 'prem',
            
            # 是否绘制地图
            'plot_map': True,
            
            # 是否绘制波形
            'plot_waveforms': True
        }
        
        # ============ 滤波配置 ============
        self.filtering = {
            'enable': True,  # 是否启用滤波
            'type': 'bandpass',  # 'bandpass' | 'highpass' | 'lowpass' | 'none'
            
            # 带通滤波参数
            'bandpass': {
                'freqmin': 0.01,  # 最低频率 (Hz)
                'freqmax': 0.1,   # 最高频率 (Hz)
                'corners': 4,     # 滤波器阶数
                'zerophase': True  # 是否使用零相位滤波
            },
            
            # 高通滤波参数
            'highpass': {
                'freq': 0.01,     # 截止频率 (Hz)
                'corners': 4,
                'zerophase': True
            },
            
            # 低通滤波参数
            'lowpass': {
                'freq': 0.1,      # 截止频率 (Hz)
                'corners': 4,
                'zerophase': True
            },
            
            # 常用预设（方便快速切换）
            'presets': {
                'long_period': {'type': 'bandpass', 'freqmin': 0.01, 'freqmax': 0.05},  # 长周期
                'intermediate': {'type': 'bandpass', 'freqmin': 0.05, 'freqmax': 0.2},   # 中周期
                'body_wave': {'type': 'bandpass', 'freqmin': 0.1, 'freqmax': 1.0},      # 体波
                'surface_wave': {'type': 'bandpass', 'freqmin': 0.02, 'freqmax': 0.1},  # 面波
                'broadband': {'type': 'bandpass', 'freqmin': 0.01, 'freqmax': 2.0}      # 宽频
            }
        }
        
        # 数据查找配置
        self.data_search = {
            'event_file_patterns': [
                "selected_events*.par",
                "selected_events*.csv"
            ],
            'waveform_search_dirs': [
                "data/events/vel_data",
            ],
            'waveform_file_patterns': ["*.SAC", "*.sac"],
            'require_waveforms': True,  # 是否要求事件必须有波形
            'validate_coordinates': True,
            # 台站文件配置
            'station_file_patterns': [
                "EastAsia_stations.csv",
                "*_stations.csv"
            ]
        }
        
        # 速度模型配置
        self.velocity_models = {
            'available_models': ['prem', 'ak135', 'iasp91'],
            'default_model': 'prem',
            'auto_load': True
        }
        
        # 震相配置
        self.seismic_phases = {
            'phase_list': ['P', 'S', 'PP', 'SS', 'PPP', 'SSS', 'PKP', 'SKS'],
            'phase_colors': {
                'P': 'red', 'S': 'green', 'PP': 'orange', 'SS': 'purple',
                'PPP': 'blue', 'SSS': 'brown', 'PKP': 'cyan', 'SKS': 'magenta'
            },
            'labeled_phases': ['P', 'S', 'PP', 'SS', 'PKP', 'SKS', 'PPP', 'SSS'],
            'phase_label_fontsize': 6,
            'phase_linewidth': 1.2,
            'phase_alpha': 0.8,
            'phase_line_height': 0.8
        }
        
        # 波形分量配置
        self.components = {
            'Z': ['BHZ', 'HHZ', 'EHZ', 'SHZ', 'LHZ', 'Z'],
            'N': ['BHN', 'HHN', 'EHN', 'SHN', 'LHN', 'N', 'BH1', 'HH1'],
            'E': ['BHE', 'HHE', 'EHE', 'SHE', 'LHE', 'E', 'BH2', 'HH2']
        }
        
        # 事件匹配配置
        self.event_matching = {
            'max_distance_deg': 1.0,      # 最大位置差（度）
            'max_time_diff_days': 7,      # 最大时间差（天）
            'focal_mechanism_fields': [
                'First_Nodal_Strike', 'First_Nodal_Dip', 'First_Nodal_Rake',
                'Second_Nodal_Strike', 'Second_Nodal_Dip', 'Second_Nodal_Rake',
                'Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp'
            ]
        }
        
        # 可视化配置 - 地图
        self.map_visualization = {
            'figure_size': (16, 12),
            'projection': 'PlateCarree',
            'coastline_linewidth': 1.0,
            'coastline_color': 'black',
            'borders_linewidth': 0.8,
            'borders_color': 'gray',
            'land_color': 'lightgray',
            'land_alpha': 0.4,
            'ocean_color': 'lightblue',
            'ocean_alpha': 0.4,
            'grid_linewidth': 0.5,
            'grid_color': 'gray',
            'grid_alpha': 0.7,
            'grid_label_size': 11,
            # 事件显示
            'background_event_size': 15,
            'background_event_color': 'lightgray',
            'background_event_alpha': 0.6,
            'selected_event_color': 'red',
            'selected_event_edgecolor': 'black',
            'selected_event_linewidth': 1.5,
            'selected_event_alpha': 0.8,
            'focal_mechanism_linewidth': 0.5,
            'focal_mechanism_facecolor': 'red',
            'focal_mechanism_alpha': 0.8,
            'focal_mechanism_size_factor': 0.4,  # 震源球大小 = 震级 * 因子
            'event_label_fontsize': 9,
            'event_label_offset_lon': 1.0,
            'event_label_offset_lat': 1.0,
            # 台站显示
            'station_size': 25,
            'station_color': 'blue',
            'station_marker': '^',
            'station_edgecolor': 'white',
            'station_linewidth': 0.5,
            'station_alpha': 0.8,
            # 图例
            'legend_location': 'lower right',
            'legend_fontsize': 10
        }
        
        # 可视化配置 - 记录段
        self.record_section = {
            'min_figure_height': 8,
            'max_figure_height': 20,
            'height_per_trace': 0.3,
            'figure_width': 30,
            'waveform_linewidth': 0.7,
            'waveform_alpha': 0.8,
            'amplitude_scale_base': 8,
            'amplitude_scale_factor': 0.1,
            'min_amplitude_scale': 2,
            'max_display_time': 3600,  # 最大显示时间（秒）
            'distance_margin': 3,  # 距离轴边距（度）
            'label_fontsize': 7,
            'label_position_factor': 1.01,
            'xlabel_fontsize': 12,
            'ylabel_fontsize': 12,
            'title_fontsize': 12,
            'grid_alpha': 0.3,
            'colormap': 'viridis',
            'progress_interval': 10  # 每读取N个文件显示进度
        }
        
        # 输出配置
        self.output = {
            'map_filename': '5-6-east_asia_events_and_stations.png',
            'record_section_template': '5-6-event_{index}_{event_id}_{component}_{model}_{filter}_record_section.png',
            'dpi': 300,
            'map_dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white'
        }
        
        # Matplotlib配置
        self.matplotlib = {
            'font_size': 10,
            'figure_dpi': 100,
            'unicode_minus': False,
            'font_families': ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        }
        
        # 日志配置
        self.logging = {
            'level': 'INFO',
            'console_output': True
        }


class EventWaveformVisualizer:
    """事件波形可视化类"""
    
    def __init__(self):
        """初始化可视化器"""
        # 加载基础配置
        self.base_config = BaseConfig()
        
        # 加载模块配置
        self.config = WaveformVisualizationConfig()
        
        # 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.WaveformVisualization',
            self.config.logging['level']
        )
        
        # 设置路径
        self._setup_paths()
        
        # 设置matplotlib
        self._setup_matplotlib()
        
        # 加载速度模型
        self._load_velocity_models()
        
        # 加载台站数据库
        self._load_station_database()
        
        self.logger.info("🎨 波形可视化器初始化完成")
        self._print_config_summary()

    def _setup_paths(self):
        """设置路径"""
        self.data_dir = self.base_config.dirs['project_root']
        self.figure_dir = self.base_config.dirs['figures']
        self.events_dir = self.base_config.dirs['events']
        
        # 确保输出目录存在
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📁 数据目录: {self.data_dir}")
        self.logger.info(f"📁 图表目录: {self.figure_dir}")

    def _setup_matplotlib(self):
        """设置matplotlib"""
        plt.rcParams['font.size'] = self.config.matplotlib['font_size']
        plt.rcParams['axes.unicode_minus'] = self.config.matplotlib['unicode_minus']
        plt.rcParams['figure.dpi'] = self.config.matplotlib['figure_dpi']
        plt.rcParams['font.sans-serif'] = self.config.matplotlib['font_families']
        
        self.logger.info("✅ Matplotlib配置完成")
    
    def _load_velocity_models(self):
        """加载地震速度模型"""
        self.models = {}
        model_names = self.config.velocity_models['available_models']
        
        for model_name in model_names:
            try:
                self.models[model_name] = TauPyModel(model=model_name)
                self.logger.info(f"✅ {model_name.upper()}模型加载成功")
            except Exception as e:
                self.logger.warning(f"⚠️ {model_name.upper()}模型加载失败: {e}")
        
        # 设置默认模型
        default_name = self.config.run_params['velocity_model']
        if default_name in self.models:
            self.current_model = self.models[default_name]
            self.current_model_name = default_name
            self.logger.info(f"✅ 使用速度模型: {default_name.upper()}")
        elif self.models:
            default_name = list(self.models.keys())[0]
            self.current_model = self.models[default_name]
            self.current_model_name = default_name
            self.logger.info(f"✅ 使用速度模型: {default_name.upper()}")
        else:
            self.current_model = None
            self.current_model_name = 'none'
            self.logger.warning("⚠️ 未能加载任何速度模型")
    
    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n📋 波形可视化配置摘要")
        print("-" * 60)
        
        # 研究区域配置
        print(f"研究区域: {self.base_config.region['name']}")
        region = self.base_config.region
        print(f"纬度范围: {region['lat_min']}° ~ {region['lat_max']}°")
        print(f"经度范围: {region['lon_min']}° ~ {region['lon_max']}°")
        
        # 速度模型配置
        print(f"可用速度模型: {', '.join(self.models.keys())}")
        print(f"当前使用模型: {self.current_model_name.upper()}")
        
        # 运行参数
        print(f"选择事件序号: {self.config.run_params['selected_event_indices']}")
        print(f"波形分量: {self.config.run_params['component']}")
        
        # 滤波配置
        filter_cfg = self.config.filtering
        if filter_cfg['enable'] and filter_cfg['type'] != 'none':
            print(f"滤波类型: {filter_cfg['type'].upper()}")
            if filter_cfg['type'] == 'bandpass':
                bp = filter_cfg['bandpass']
                print(f"滤波频率: {bp['freqmin']}-{bp['freqmax']} Hz")
            elif filter_cfg['type'] == 'highpass':
                hp = filter_cfg['highpass']
                print(f"滤波频率: >{hp['freq']} Hz")
            elif filter_cfg['type'] == 'lowpass':
                lp = filter_cfg['lowpass']
                print(f"滤波频率: <{lp['freq']} Hz")
        else:
            print("滤波: 关闭")
        
        # 输出配置
        print(f"输出目录: {self.figure_dir}")
        
        print("-" * 60)
    
    def load_events(self) -> pd.DataFrame:
        """加载事件数据"""
        # 优先查找PAR文件
        par_pattern = self.config.data_search['event_file_patterns'][0]
        par_files = list(self.events_dir.glob(par_pattern))
        
        if par_files:
            latest_file = max(par_files, key=lambda x: x.stat().st_mtime)
            par_events = self._load_par_file(latest_file)
            
            # 尝试匹配GCMT震源机制
            csv_pattern = self.config.data_search['event_file_patterns'][1]
            csv_files = list(self.events_dir.glob(csv_pattern))
            
            if csv_files:
                latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                gcmt_events = self._load_gcmt_csv(latest_csv)
                return self._match_par_with_gcmt(par_events, gcmt_events)
            else:
                self.logger.warning("⚠️ 未找到GCMT文件，仅使用PAR事件")
                return par_events
        
        # 备选: 仅GCMT文件
        csv_pattern = self.config.data_search['event_file_patterns'][1]
        csv_files = list(self.events_dir.glob(csv_pattern))
        
        if csv_files:
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            return self._load_gcmt_csv_file(latest_file)
        
        raise FileNotFoundError("未找到事件文件")

    def _load_par_file(self, par_file: Path) -> pd.DataFrame:
        """加载PAR文件"""
        self.logger.info(f"📁 加载事件文件: {par_file}")
        
        events_data = []
        with open(par_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) != 12:
                    continue
                
                try:
                    event_id = parts[0]
                    year_date = parts[1]
                    hour, minute, second = parts[2], parts[3], parts[4]
                    lat, lon, depth = float(parts[5]), float(parts[6]), float(parts[7])
                    mag, mag_type = float(parts[10]), parts[11]
                    
                    # 解析时间
                    year, month, day = year_date[:4], year_date[4:6], year_date[6:8]
                    
                    # 验证数据
                    if self.config.data_search['validate_coordinates']:
                        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                            continue
                    
                    if depth < 0 or depth > 1000:
                        depth = abs(depth) if depth < 0 else min(depth, 700)
                    
                    # 检查是否有波形文件
                    if self.config.data_search['require_waveforms']:
                        if not self._check_waveforms_exist(event_id):
                            continue
                    
                    # 处理时间精度
                    try:
                        second_float = float(second)
                        second_int = int(second_float)
                        microsecond = int((second_float - second_int) * 1000000)
                    except ValueError:
                        second_int, second_float, microsecond = 0, 0.0, 0
                    
                    datetime_str = f"{year}-{month}-{day} {hour.zfill(2)}:{minute.zfill(2)}:{second_int:02d}"
                    
                    events_data.append({
                        'EventID': event_id,
                        'DateTime': datetime_str,
                        'Year': int(year), 'Month': int(month), 'Day': int(day),
                        'Hour': int(hour), 'Minute': int(minute),
                        'Second': second_float, 'Microsecond': microsecond,
                        'Latitude': lat, 'Longitude': lon, 'Depth': depth,
                        'Magnitude': mag, 'MagnitudeType': mag_type,
                        'has_waveforms': True
                    })
                    
                except (ValueError, IndexError):
                    continue
        
        if not events_data:
            raise ValueError("文件中没有有效事件数据")
        
        df = pd.DataFrame(events_data)
        self.logger.info(f"✅ 加载 {len(events_data)} 个有波形的事件")
        return df

    def _check_waveforms_exist(self, event_id: str) -> bool:
        """检查事件是否有波形文件"""
        for search_dir_rel in self.config.data_search['waveform_search_dirs']:
            search_dir = self.data_dir / search_dir_rel / event_id
            if search_dir.exists():
                for pattern in self.config.data_search['waveform_file_patterns']:
                    sac_files = list(search_dir.glob(pattern))
                    if sac_files:
                        return True
        return False
    
    def _load_gcmt_csv(self, csv_file: Path) -> pd.DataFrame:
        """加载GCMT CSV文件"""
        self.logger.info(f"📁 加载GCMT震源机制文件: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # 标准化列名
        if 'Longitude' not in df.columns and 'longitude' in df.columns:
            df['Longitude'] = df['longitude']
        if 'Latitude' not in df.columns and 'latitude' in df.columns:
            df['Latitude'] = df['latitude']
        if 'Depth' not in df.columns and 'depth' in df.columns:
            df['Depth'] = df['depth']
        
        # 处理时间信息
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        
        # 添加事件ID
        if 'Event Name' not in df.columns:
            df['EventID'] = df.index.map(lambda x: f"GCMT_{x:04d}")
        else:
            df['EventID'] = df['Event Name']
        
        self.logger.info(f"✅ 加载 {len(df)} 个GCMT震源机制")
        return df

    def _match_par_with_gcmt(self, par_events: pd.DataFrame, gcmt_events: pd.DataFrame) -> pd.DataFrame:
        """匹配PAR事件与GCMT震源机制"""
        self.logger.info(f"🔗 匹配PAR事件与GCMT震源机制...")
        
        matched_events = []
        max_distance = self.config.event_matching['max_distance_deg']
        max_time_diff = self.config.event_matching['max_time_diff_days'] * 24 * 3600
        
        for _, par_event in par_events.iterrows():
            par_lat, par_lon = par_event['Latitude'], par_event['Longitude']
            par_time = pd.to_datetime(par_event['DateTime'], errors='coerce')
            
            best_match = None
            min_distance = float('inf')
            
            for _, gcmt_event in gcmt_events.iterrows():
                gcmt_lat, gcmt_lon = gcmt_event.get('Latitude', 0), gcmt_event.get('Longitude', 0)
                datetime_value = gcmt_event.get('DateTime')
                gcmt_time = pd.to_datetime(datetime_value, errors='coerce') if datetime_value is not None else pd.NaT
                
                # 计算距离（简单的度数差）
                distance = ((par_lat - gcmt_lat)**2 + (par_lon - gcmt_lon)**2)**0.5
                
                # 计算时间差
                time_diff = float('inf')
                if pd.notna(par_time) and pd.notna(gcmt_time):
                    time_diff = abs((par_time - gcmt_time).total_seconds())
                
                # 匹配条件
                if distance < max_distance and time_diff < max_time_diff and distance < min_distance:
                    min_distance = distance
                    best_match = gcmt_event
            
            # 合并事件信息
            merged_event = par_event.copy()
            if best_match is not None:
                # 添加GCMT震源机制信息
                for field in self.config.event_matching['focal_mechanism_fields']:
                    if field in best_match:
                        merged_event[field] = best_match[field]
                merged_event['has_focal_mechanism'] = True
            else:
                merged_event['has_focal_mechanism'] = False
            
            matched_events.append(merged_event)
        
        result_df = pd.DataFrame(matched_events)
        matched_count = sum(result_df['has_focal_mechanism'])
        self.logger.info(f"✅ 匹配完成: {matched_count}/{len(par_events)} 个事件有震源机制")
        
        return result_df

    def _load_gcmt_csv_file(self, csv_file: Path) -> pd.DataFrame:
        """加载GCMT CSV文件（备选方案）"""
        self.logger.info(f"📁 加载GCMT事件文件: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # 标准化列名
        if 'Longitude' not in df.columns and 'longitude' in df.columns:
            df['Longitude'] = df['longitude']
        if 'Latitude' not in df.columns and 'latitude' in df.columns:
            df['Latitude'] = df['latitude']
        if 'Depth' not in df.columns and 'depth' in df.columns:
            df['Depth'] = df['depth']
        
        # 处理时间信息
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        
        # 处理震级信息
        if 'Magnitude2_Ms' in df.columns:
            df['Magnitude'] = pd.to_numeric(df['Magnitude2_Ms'], errors='coerce').fillna(
                pd.to_numeric(df.get('Magnitude1_mb', 0), errors='coerce'))
        elif 'Magnitude1_mb' in df.columns:
            df['Magnitude'] = pd.to_numeric(df['Magnitude1_mb'], errors='coerce')
        else:
            df['Magnitude'] = 6.0  # 默认震级
        
        # 添加事件ID
        if 'Event Name' not in df.columns:
            df['EventID'] = df.index.map(lambda x: f"GCMT_{x:04d}")
        else:
            df['EventID'] = df['Event Name']
        
        df['has_focal_mechanism'] = True
        df['has_waveforms'] = False  # GCMT数据默认没有波形标记
        
        self.logger.info(f"✅ 加载 {len(df)} 个GCMT事件")
        return df

    def parse_focal_mechanism(self, event_row) -> Optional[Dict]:
        """从事件数据中解析震源机制参数"""
        try:
            if not event_row.get('has_focal_mechanism', False):
                return None
            
            # 尝试获取节面参数
            strike1 = event_row.get('First_Nodal_Strike')
            dip1 = event_row.get('First_Nodal_Dip') 
            rake1 = event_row.get('First_Nodal_Rake')
            
            if all(pd.notna([strike1, dip1, rake1])):
                try:
                    return {
                        'type': 'nodal', 
                        'params': {
                            'strike': float(strike1),
                            'dip': float(dip1), 
                            'rake': float(rake1)
                        }
                    }
                except (ValueError, TypeError):
                    pass
            
            # 尝试获取矩张量参数
            mt_fields = ['Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp']
            mt_values = []
            
            for field in mt_fields:
                value = event_row.get(field)
                if pd.notna(value):
                    try:
                        if isinstance(value, str):
                            value = value.replace('E', 'e').replace('D', 'e')
                            mt_values.append(float(value))
                        else:
                            mt_values.append(float(value))
                    except (ValueError, TypeError):
                        mt_values.append(0.0)
                else:
                    mt_values.append(0.0)
            
            if len(mt_values) == 6 and any(abs(v) > 1e-20 for v in mt_values):
                return {
                    'type': 'mt', 
                    'params': {
                        'mrr': mt_values[0], 'mtt': mt_values[1], 'mpp': mt_values[2],
                        'mrt': mt_values[3], 'mrp': mt_values[4], 'mtp': mt_values[5]
                    }
                }
            
            return None
                
        except Exception as e:
            self.logger.debug(f"解析震源机制失败: {e}")
            return None

    def show_events(self, events_df: pd.DataFrame):
        """显示事件列表"""
        print("\n📋 可用事件列表:")
        print("=" * 100)
        print(f"{'序号':<4} {'事件ID':<15} {'日期':<12} {'纬度':<8} {'经度':<9} {'深度':<7} {'震级':<6} {'震源机制':<10}")
        print("-" * 100)
        
        for idx, row in events_df.iterrows():
            event_id = str(row.get('EventID', f"Event_{idx}"))[:14]
            date_str = str(row.get('DateTime', 'N/A'))[:10]
            lat = f"{row.get('Latitude', 0):.2f}"
            lon = f"{row.get('Longitude', 0):.2f}"
            depth = f"{row.get('Depth', 0):.1f}"
            mag = f"{row.get('Magnitude', 0):.1f}"
            fm_status = "✅" if row.get('has_focal_mechanism', False) else "❌"
            
            print(f"{idx:<4} {event_id:<15} {date_str:<12} {lat:<8} {lon:<9} {depth:<7} {mag:<6} {fm_status:<10}")
        print("-" * 100)

    def extract_stations_from_event_waveforms(self, event_id: str) -> List[Dict]:
        """从事件波形文件的SAC头文件中提取台站信息"""
        stations = []
        
        # 在事件专用文件夹中查找SAC文件
        for search_dir_rel in self.config.data_search['waveform_search_dirs']:
            search_dir = self.data_dir / search_dir_rel / event_id
            
            if not search_dir.exists():
                continue
            
            for pattern in self.config.data_search['waveform_file_patterns']:
                sac_files = list(search_dir.glob(pattern))
                
                for sac_file in sac_files:
                    try:
                        st = read(str(sac_file))
                        tr = st[0]
                        stats = tr.stats
                        sac = getattr(stats, 'sac', {})
                        
                        # 从SAC头文件获取台站信息
                        network = stats.network or "XX"
                        station = stats.station or sac.get('kstnm', 'STA')
                        
                        # 优先从SAC头文件获取坐标
                        station_lat = sac.get('stla')
                        station_lon = sac.get('stlo')
                        
                        # 如果SAC中没有，从台站数据库获取
                        if station_lat is None or station_lon is None:
                            station_key = f"{network}.{station}"
                            if station_key in self.station_db:
                                station_lat = self.station_db[station_key]['latitude']
                                station_lon = self.station_db[station_key]['longitude']
                            else:
                                # 尝试只用台站名查找
                                for key, info in self.station_db.items():
                                    if info['station'] == station:
                                        station_lat = info['latitude']
                                        station_lon = info['longitude']
                                        break
                        
                        if station_lat is not None and station_lon is not None:
                            stations.append({
                                'network': network,
                                'station': station,
                                'latitude': station_lat,
                                'longitude': station_lon,
                                'code': f"{network}.{station}"
                            })
                            
                    except Exception:
                        continue
        
        # 去重
        unique_stations = []
        seen_codes = set()
        for station in stations:
            if station['code'] not in seen_codes:
                unique_stations.append(station)
                seen_codes.add(station['code'])
        
        return unique_stations

    def plot_event_map(self, events_df: pd.DataFrame, selected_indices: List[int]):
        """绘制东亚地区事件地图"""
        if events_df.empty:
            self.logger.error("❌ 没有事件数据")
            return None
        
        # 使用基础配置中的区域定义
        region = self.base_config.region
        map_config = self.config.map_visualization
        
        # 创建地图
        fig_size = map_config['figure_size']
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # 设置地图范围为东亚
        ax.set_extent([region['lon_min'], region['lon_max'], 
                      region['lat_min'], region['lat_max']], 
                     crs=ccrs.PlateCarree())
        
        # 添加地理要素
        ax.add_feature(cfeature.COASTLINE, 
                      linewidth=map_config['coastline_linewidth'], 
                      color=map_config['coastline_color'])
        ax.add_feature(cfeature.BORDERS, 
                      linewidth=map_config['borders_linewidth'], 
                      color=map_config['borders_color'])
        ax.add_feature(cfeature.LAND, 
                      color=map_config['land_color'], 
                      alpha=map_config['land_alpha'])
        ax.add_feature(cfeature.OCEAN, 
                      color=map_config['ocean_color'], 
                      alpha=map_config['ocean_alpha'])
        
        # 添加网格
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                         linewidth=map_config['grid_linewidth'], 
                         color=map_config['grid_color'], 
                         alpha=map_config['grid_alpha'])
        gl.xlabel_style = gl.ylabel_style = {'size': map_config['grid_label_size']}
        
        transform = ccrs.PlateCarree()
        
        # 绘制背景事件（在东亚范围内）
        background_events = events_df.drop(selected_indices)
        if not background_events.empty:
            bg_in_region = background_events[
                (background_events['Latitude'] >= region['lat_min']) &
                (background_events['Latitude'] <= region['lat_max']) &
                (background_events['Longitude'] >= region['lon_min']) &
                (background_events['Longitude'] <= region['lon_max'])
            ]
            if not bg_in_region.empty:
                ax.scatter(bg_in_region['Longitude'], bg_in_region['Latitude'],
                          s=map_config['background_event_size'], 
                          c=map_config['background_event_color'], 
                          alpha=map_config['background_event_alpha'], 
                          transform=transform, 
                          label=f'Other Events ({len(bg_in_region)})', zorder=2)
        
        # 绘制选择的事件和台站
        selected_events = events_df.iloc[selected_indices]
        all_stations = []
        
        for i, (idx, event) in enumerate(selected_events.iterrows()):
            lon, lat = event['Longitude'], event['Latitude']
            mag = event.get('Magnitude', 0)
            event_id = str(event.get('EventID', f'Event_{idx}'))
            
            # 绘制震源球或地震点
            focal_mech = self.parse_focal_mechanism(event)
            
            if focal_mech:
                # 绘制震源球
                size = max(1.0, mag * map_config['focal_mechanism_size_factor'])
                
                try:
                    if focal_mech['type'] == 'mt':
                        mt = focal_mech['params']
                        mt_list = [mt['mrr'], mt['mtt'], mt['mpp'], 
                                  mt['mrt'], mt['mrp'], mt['mtp']]
                        bb = beach(mt_list, xy=(lon, lat), width=size, 
                                  linewidth=map_config['focal_mechanism_linewidth'], 
                                  facecolor=map_config['focal_mechanism_facecolor'], 
                                  alpha=map_config['focal_mechanism_alpha'])
                        ax.add_collection(bb)
                    elif focal_mech['type'] == 'nodal':
                        nodal = focal_mech['params']
                        nodal_list = [nodal['strike'], nodal['dip'], nodal['rake']]
                        bb = beach(nodal_list, xy=(lon, lat), width=size,
                                  linewidth=map_config['focal_mechanism_linewidth'], 
                                  facecolor=map_config['focal_mechanism_facecolor'], 
                                  alpha=map_config['focal_mechanism_alpha'])
                        ax.add_collection(bb)
                    
                    self.logger.info(f"✅ 事件 {i}: 震源球绘制成功")
                except Exception as e:
                    self.logger.warning(f"❌ 事件 {i} 震源球绘制失败: {e}")
                    size = max(50, mag * 20)
                    ax.scatter(lon, lat, s=size, 
                              c=map_config['selected_event_color'], 
                              marker='o',
                              edgecolors=map_config['selected_event_edgecolor'], 
                              linewidth=map_config['selected_event_linewidth'], 
                              transform=transform, 
                              alpha=map_config['selected_event_alpha'], 
                              zorder=4)
            else:
                # 绘制普通地震点
                size = max(50, mag * 20)
                ax.scatter(lon, lat, s=size, 
                          c=map_config['selected_event_color'], 
                          marker='o',
                          edgecolors=map_config['selected_event_edgecolor'], 
                          linewidth=map_config['selected_event_linewidth'], 
                          transform=transform, 
                          alpha=map_config['selected_event_alpha'], 
                          zorder=4)
            
            # 添加事件标签
            fm_status = "🟢" if event.get('has_focal_mechanism', False) else "🔴"
            ax.text(lon + map_config['event_label_offset_lon'], 
                   lat + map_config['event_label_offset_lat'], 
                   f"{i}: {event_id[:8]} {fm_status}\nM{mag:.1f}",
                   fontsize=map_config['event_label_fontsize'], 
                   fontweight='bold', 
                   transform=transform,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   zorder=6)
            
            # 提取台站信息
            self.logger.info(f"🔍 提取事件 {event_id} 的台站信息...")
            stations = self.extract_stations_from_event_waveforms(event_id)
            all_stations.extend(stations)
        
        # 绘制台站（仅在东亚范围内）
        if all_stations:
            stations_in_region = [
                s for s in all_stations 
                if (region['lat_min'] <= s['latitude'] <= region['lat_max'] and
                    region['lon_min'] <= s['longitude'] <= region['lon_max'])
            ]
            
            if stations_in_region:
                station_lons = [s['longitude'] for s in stations_in_region]
                station_lats = [s['latitude'] for s in stations_in_region]
                
                ax.scatter(station_lons, station_lats, 
                          s=map_config['station_size'], 
                          c=map_config['station_color'], 
                          marker=map_config['station_marker'],
                          edgecolors=map_config['station_edgecolor'], 
                          linewidth=map_config['station_linewidth'], 
                          transform=transform,
                          label=f'Stations ({len(stations_in_region)})', 
                          alpha=map_config['station_alpha'], 
                          zorder=3)
                self.logger.info(f"✅ 绘制 {len(stations_in_region)} 个台站")
        
        # 添加图例
        legend_elements = [
            plt.scatter([], [], c='lightgray', s=15, label='Other Events'),
            plt.scatter([], [], c='red', s=80, label='Selected Events'),
            plt.scatter([], [], c='blue', marker='^', s=50, label='Stations') # type: ignore
        ]
        plt.legend(handles=legend_elements, 
                  loc=map_config['legend_location'], 
                  fontsize=map_config['legend_fontsize'])
        
        # 标题
        fm_count = sum(events_df.iloc[selected_indices]['has_focal_mechanism'])
        plt.title(f'East Asia Seismic Events and Stations\n'
                 f'Selected Events: {len(selected_indices)}, With FM: {fm_count}, '
                 f'Stations: {len(all_stations) if all_stations else 0}', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # 保存地图
        save_path = self.figure_dir / self.config.output['map_filename']
        plt.savefig(save_path, 
                   dpi=self.config.output['map_dpi'], 
                   bbox_inches=self.config.output['bbox_inches'], 
                   facecolor=self.config.output['facecolor'])
        self.logger.info(f"📊 东亚地图已保存: {save_path}")
        
        plt.close()
        return fig

    def apply_filter(self, trace, filter_type: str = None):
        """应用滤波到波形数据
        
        Args:
            trace: ObsPy trace对象
            filter_type: 滤波类型 (None表示从配置读取)
            
        Returns:
            滤波后的trace对象
        """
        filter_cfg = self.config.filtering
        
        # 检查是否启用滤波
        if not filter_cfg['enable']:
            return trace
        
        # 确定滤波类型
        if filter_type is None:
            filter_type = filter_cfg['type']
        
        if filter_type == 'none':
            return trace
        
        try:
            # 创建trace副本以避免修改原始数据
            tr_filtered = trace.copy()
            
            # 去除趋势和均值
            tr_filtered.detrend('demean')
            tr_filtered.detrend('linear')
            
            # 应用滤波
            if filter_type == 'bandpass':
                params = filter_cfg['bandpass']
                tr_filtered.filter(
                    'bandpass',
                    freqmin=params['freqmin'],
                    freqmax=params['freqmax'],
                    corners=params['corners'],
                    zerophase=params['zerophase']
                )
                self.logger.debug(f"应用带通滤波: {params['freqmin']}-{params['freqmax']} Hz")
                
            elif filter_type == 'highpass':
                params = filter_cfg['highpass']
                tr_filtered.filter(
                    'highpass',
                    freq=params['freq'],
                    corners=params['corners'],
                    zerophase=params['zerophase']
                )
                self.logger.debug(f"应用高通滤波: >{params['freq']} Hz")
                
            elif filter_type == 'lowpass':
                params = filter_cfg['lowpass']
                tr_filtered.filter(
                    'lowpass',
                    freq=params['freq'],
                    corners=params['corners'],
                    zerophase=params['zerophase']
                )
                self.logger.debug(f"应用低通滤波: <{params['freq']} Hz")
            
            return tr_filtered
            
        except Exception as e:
            self.logger.warning(f"⚠️ 滤波失败: {e}，返回原始波形")
            return trace

    def get_filter_description(self) -> str:
        """获取滤波参数描述字符串"""
        filter_cfg = self.config.filtering
        
        if not filter_cfg['enable'] or filter_cfg['type'] == 'none':
            return 'nofilter'
        
        filter_type = filter_cfg['type']
        
        if filter_type == 'bandpass':
            params = filter_cfg['bandpass']
            return f"bp{params['freqmin']}-{params['freqmax']}Hz"
        elif filter_type == 'highpass':
            params = filter_cfg['highpass']
            return f"hp{params['freq']}Hz"
        elif filter_type == 'lowpass':
            params = filter_cfg['lowpass']
            return f"lp{params['freq']}Hz"
        else:
            return 'filtered'

    def get_filter_title(self) -> str:
        """获取滤波标题描述"""
        filter_cfg = self.config.filtering
        
        if not filter_cfg['enable'] or filter_cfg['type'] == 'none':
            return 'No Filter'
        
        filter_type = filter_cfg['type']
        
        if filter_type == 'bandpass':
            params = filter_cfg['bandpass']
            return f"Bandpass {params['freqmin']}-{params['freqmax']} Hz"
        elif filter_type == 'highpass':
            params = filter_cfg['highpass']
            return f"Highpass >{params['freq']} Hz"
        elif filter_type == 'lowpass':
            params = filter_cfg['lowpass']
            return f"Lowpass <{params['freq']} Hz"
        else:
            return 'Filtered'

    def filter_waveforms_by_component(self, sac_files: List[Path], component: str) -> List[Path]:
        """根据分量过滤波形文件"""
        if component == 'All':
            return sac_files
        
        patterns = self.config.components.get(component, ['Z'])
        filtered_files = []
        
        for sac_file in sac_files:
            try:
                # 检查文件名
                if any(pattern in sac_file.name.upper() for pattern in patterns):
                    filtered_files.append(sac_file)
                    continue
                
                # 检查通道信息
                st = read(str(sac_file))
                channel = st[0].stats.channel.upper() if st[0].stats.channel else ""
                if any(pattern in channel for pattern in patterns):
                    filtered_files.append(sac_file)
                    
            except Exception:
                continue
        
        self.logger.info(f"📊 {component}分量过滤: {len(sac_files)} -> {len(filtered_files)} 个文件")
        return filtered_files

    def find_waveforms(self, event_id: str) -> List[Path]:
        """查找波形文件"""
        found_files = []
        
        for search_dir_rel in self.config.data_search['waveform_search_dirs']:
            search_dir = self.data_dir / search_dir_rel / event_id
            
            if not search_dir.exists():
                continue
            
            for pattern in self.config.data_search['waveform_file_patterns']:
                files = list(search_dir.glob(pattern))
                found_files.extend(files)
        
        found_files = sorted(list(set(found_files)), key=lambda x: x.name)
        self.logger.info(f"📁 找到 {len(found_files)} 个波形文件")
        return found_files

    def plot_waveforms(self, event_info: Dict, event_index: int):
        """绘制波形"""
        event_id = event_info.get('EventID', f"Event_{event_index}")
        self.logger.info(f"\n🔍 处理事件 {event_id}...")
        
        # 从配置读取参数
        component = self.config.run_params['component']
        display_option = self.config.run_params['display_option']
        model_name = self.current_model_name
        
        self.logger.info(f"📊 波形分量: {component}")
        self.logger.info(f"📊 显示选项: {display_option['type']}")
        self.logger.info(f"📊 速度模型: {model_name.upper()}")
        
        # 查找并过滤波形文件
        sac_files = self.find_waveforms(event_id)
        if not sac_files:
            self.logger.error("❌ 未找到波形文件")
            return None
        
        filtered_files = self.filter_waveforms_by_component(sac_files, component)
        if not filtered_files:
            self.logger.error(f"❌ 未找到 {component} 分量的波形文件")
            return None
        
        # 读取并处理波形数据
        waveform_data = self._read_waveform_data(filtered_files, event_info)
        if not waveform_data:
            self.logger.error("❌ 没有有效的波形数据")
            return None
        
        waveform_data.sort(key=lambda x: x['distance_deg'])
        waveform_data = self._filter_waveform_data(waveform_data, display_option)
        
        if not waveform_data:
            self.logger.error("❌ 过滤后没有数据")
            return None
        
        return self._plot_record_section(waveform_data, event_info, event_index, 
                                       component, model_name)

    def _load_station_database(self):
        """加载台站数据库"""
        self.station_db = {}
        
        stations_dir = self.base_config.dirs['stations']
        patterns = self.config.data_search.get('station_file_patterns', ['*_stations.csv'])
        
        for pattern in patterns:
            station_files = list(stations_dir.glob(pattern))
            if station_files:
                # 使用最新的台站文件
                latest_file = max(station_files, key=lambda x: x.stat().st_mtime)
                try:
                    df = pd.read_csv(latest_file)
                    self.logger.info(f"📍 加载台站文件: {latest_file.name}")
                    
                    # 构建台站数据库：key = "Network.Station"
                    for _, row in df.iterrows():
                        network = str(row.get('Network', 'XX'))
                        station = str(row.get('Station', ''))
                        lat = row.get('Latitude')
                        lon = row.get('Longitude')
                        elev = row.get('Elevation', 0.0)
                        
                        if station and pd.notna(lat) and pd.notna(lon):
                            key = f"{network}.{station}"
                            self.station_db[key] = {
                                'latitude': float(lat),
                                'longitude': float(lon),
                                'elevation': float(elev) if pd.notna(elev) else 0.0,
                                'network': network,
                                'station': station
                            }
                    
                    self.logger.info(f"✅ 加载 {len(self.station_db)} 个台站坐标")
                    break
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ 台站文件读取失败: {e}")
        
        if not self.station_db:
            self.logger.warning("⚠️ 未能加载台站数据库，将只使用 SAC 头文件中的坐标")

    def _read_waveform_data(self, sac_files: List[Path], event_info: Dict) -> List[Dict]:
        """读取波形数据（包含滤波）"""
        waveform_data = []
        event_lat, event_lon = event_info.get('Latitude', 0), event_info.get('Longitude', 0)
        event_time = self._get_event_origin_time(event_info)
        
        progress_interval = self.config.record_section['progress_interval']
        self.logger.info(f"📊 读取 {len(sac_files)} 个波形文件...")
        
        # 显示滤波信息
        if self.config.filtering['enable'] and self.config.filtering['type'] != 'none':
            self.logger.info(f"🔧 应用滤波: {self.get_filter_title()}")
        
        # 统计跳过原因
        skip_reasons = {
            'no_station_coords': 0,
            'no_distance': 0,
            'read_error': 0,
            'filter_error': 0,
            'success': 0,
            'success_from_db': 0,
            'success_from_sac': 0
        }
        
        for i, sac_file in enumerate(sac_files):
            if i % progress_interval == 0:
                print(f"   进度: {i+1}/{len(sac_files)}")
            
            try:
                st = read(str(sac_file))
                tr = st[0]
                sac = getattr(tr.stats, 'sac', {})
                
                # 台站信息
                network = tr.stats.network or "XX"
                station = tr.stats.station or sac.get('kstnm', 'STA')
                channel = tr.stats.channel or "BHZ"
                
                # 获取距离信息
                distance_km = sac.get('dist')
                distance_deg = sac.get('gcarc')
                azimuth = sac.get('az')
                
                station_lat = None
                station_lon = None
                
                # 方法1: 从 SAC 头文件获取台站坐标
                if sac.get('stla') is not None and sac.get('stlo') is not None:
                    station_lat = sac.get('stla')
                    station_lon = sac.get('stlo')
                    skip_reasons['success_from_sac'] += 1
                else:
                    # 方法2: 从台站数据库获取坐标
                    station_key = f"{network}.{station}"
                    if station_key in self.station_db:
                        station_lat = self.station_db[station_key]['latitude']
                        station_lon = self.station_db[station_key]['longitude']
                        skip_reasons['success_from_db'] += 1
                    else:
                        # 尝试只用台站名查找
                        for key, info in self.station_db.items():
                            if info['station'] == station:
                                station_lat = info['latitude']
                                station_lon = info['longitude']
                                skip_reasons['success_from_db'] += 1
                                break
                
                # 如果还是没有台站坐标，跳过
                if station_lat is None or station_lon is None:
                    skip_reasons['no_station_coords'] += 1
                    continue
                
                # 如果没有距离信息则计算
                if distance_km is None or distance_deg is None:
                    try:
                        distance_m, az, _ = gps2dist_azimuth(event_lat, event_lon, station_lat, station_lon)
                        distance_km = distance_m / 1000
                        distance_deg = distance_m / 111194.9
                        azimuth = az
                    except Exception as e:
                        skip_reasons['no_distance'] += 1
                        if i < 5:
                            self.logger.debug(f"跳过 {sac_file.name}: 距离计算失败 - {e}")
                        continue
                
                # 应用滤波
                try:
                    tr = self.apply_filter(tr)
                except Exception as e:
                    skip_reasons['filter_error'] += 1
                    if i < 5:
                        self.logger.debug(f"跳过 {sac_file.name}: 滤波失败 - {e}")
                    continue
                
                waveform_data.append({
                    'trace': tr,
                    'distance_deg': distance_deg,
                    'distance_km': distance_km,
                    'azimuth': azimuth,
                    'station_name': f"{network}.{station}.{channel}",
                    'event_origin_time': event_time,
                    'sac_origin_time': self._get_sac_origin_time(sac)
                })
                
                skip_reasons['success'] += 1
                
            except Exception as e:
                skip_reasons['read_error'] += 1
                if i < 5:
                    self.logger.debug(f"跳过 {sac_file.name}: 读取错误 - {e}")
                continue
        
        # 打印跳过统计
        self.logger.info(f"✅ 成功读取 {skip_reasons['success']} 个波形")
        if skip_reasons['success_from_db'] > 0:
            self.logger.info(f"   📍 从台站数据库获取坐标: {skip_reasons['success_from_db']} 个")
        if skip_reasons['success_from_sac'] > 0:
            self.logger.info(f"   📍 从SAC头文件获取坐标: {skip_reasons['success_from_sac']} 个")
        if skip_reasons['no_station_coords'] > 0:
            self.logger.warning(f"⚠️ 跳过 {skip_reasons['no_station_coords']} 个文件：找不到台站坐标")
        if skip_reasons['no_distance'] > 0:
            self.logger.warning(f"⚠️ 跳过 {skip_reasons['no_distance']} 个文件：距离计算失败")
        if skip_reasons['filter_error'] > 0:
            self.logger.warning(f"⚠️ 跳过 {skip_reasons['filter_error']} 个文件：滤波失败")
        if skip_reasons['read_error'] > 0:
            self.logger.warning(f"⚠️ 跳过 {skip_reasons['read_error']} 个文件：读取错误")
        
        return waveform_data

    def _get_event_origin_time(self, event_info: Dict) -> Optional[UTCDateTime]:
        """获取事件发震时间"""
        try:
            fields = ['Year', 'Month', 'Day', 'Hour', 'Minute']
            if all(event_info.get(field) is not None for field in fields):
                return UTCDateTime(
                    event_info['Year'], event_info['Month'], event_info['Day'],
                    event_info['Hour'], event_info['Minute'],
                    event_info.get('Second', 0), event_info.get('Microsecond', 0)
                )
            
            datetime_str = event_info.get('DateTime')
            if datetime_str:
                return UTCDateTime(datetime_str)
        except Exception:
            pass
        return None

    def _get_sac_origin_time(self, sac: Dict) -> Optional[UTCDateTime]:
        """从SAC头文件获取发震时间"""
        try:
            fields = ['nzyear', 'nzjday', 'nzhour', 'nzmin', 'nzsec']
            if all(sac.get(field) is not None for field in fields):
                ref_time = UTCDateTime(
                    year=sac['nzyear'], julday=sac['nzjday'],
                    hour=sac['nzhour'], minute=sac['nzmin'], second=sac['nzsec'],
                    microsecond=sac.get('nzmsec', 0) * 1000
                )
                o_time = sac.get('o')
                return ref_time + o_time if o_time is not None else ref_time
        except Exception:
            pass
        return None

    def _filter_waveform_data(self, waveform_data: List[Dict], display_option: Dict) -> List[Dict]:
        """根据显示选项过滤波形数据"""
        if display_option['type'] == 'limit':
            return waveform_data[:display_option.get('max_traces', 30)]
        elif display_option['type'] == 'distance':
            min_dist = display_option.get('min_dist', 0)
            max_dist = display_option.get('max_dist', 180)
            return [wf for wf in waveform_data if min_dist <= wf['distance_deg'] <= max_dist]
        elif display_option['type'] == 'magnitude':
            min_mag = display_option.get('min_mag', 0)
            max_mag = display_option.get('max_mag', 10)
            return [wf for wf in waveform_data if min_mag <= wf.get('magnitude', 0) <= max_mag]
        else:
            return waveform_data

    def _plot_record_section(self, waveform_data: List[Dict], event_info: Dict, 
                           event_index: int, component: str, model_name: str):
        """绘制记录段"""
        n_traces = len(waveform_data)
        self.logger.info(f"📊 绘制 {n_traces} 个波形的记录段")
        
        rs_config = self.config.record_section
        
        # 创建图形
        fig_height = max(rs_config['min_figure_height'], 
                        min(rs_config['max_figure_height'], 
                            n_traces * rs_config['height_per_trace']))
        fig, ax = plt.subplots(figsize=(rs_config['figure_width'], fig_height))
        
        colors = plt.cm.get_cmap(rs_config['colormap'])(np.linspace(0, 1, n_traces)) # type: ignore
        max_time = 0
        amplitude_scale = max(
            rs_config['min_amplitude_scale'], 
            rs_config['amplitude_scale_base'] - n_traces * rs_config['amplitude_scale_factor']
        )
        
        for i, wf_data in enumerate(waveform_data):
            tr = wf_data['trace']
            distance_deg = wf_data['distance_deg']
            
            # 计算时间轴
            time_axis = self._get_relative_time_axis(tr, wf_data)
            max_time = max(max_time, time_axis[-1])
            
            # 归一化数据
            data_norm = (tr.data / np.max(np.abs(tr.data)) * amplitude_scale 
                        if np.max(np.abs(tr.data)) > 0 else tr.data)
            
            # 绘制波形
            y_offset = distance_deg
            ax.plot(time_axis, data_norm + y_offset, 
                   color=colors[i], 
                   linewidth=rs_config['waveform_linewidth'], 
                   alpha=rs_config['waveform_alpha'])
            
            # 添加理论到时
            if self.current_model:
                self._add_arrivals_to_record_section(ax, event_info, y_offset, distance_deg)
            
            # 标注
            label_text = f"{wf_data['station_name']}\n{distance_deg:.1f}°"
            if wf_data.get('distance_km'):
                label_text += f"\n({wf_data['distance_km']:.0f} km)"
            if wf_data.get('azimuth'):
                label_text += f"\n{wf_data['azimuth']:.0f}°"
            
            ax.text(max_time * rs_config['label_position_factor'], y_offset, label_text, 
                   fontsize=rs_config['label_fontsize'], va='center', ha='left')
        
        # 设置坐标轴
        ax.set_xlabel('Time after Origin (s)', fontsize=rs_config['xlabel_fontsize'])
        ax.set_ylabel('Epicentral Distance (°)', fontsize=rs_config['ylabel_fontsize'])
        
        # 标题包含滤波信息
        filter_title = self.get_filter_title()
        title = (f'Record Section ({component} Component): Event {event_index}\n'
                f'Mag: {event_info.get("Magnitude", "N/A")}, '
                f'Depth: {event_info.get("Depth", "N/A")} km, '
                f'Model: {model_name.upper()}, Filter: {filter_title}, Traces: {n_traces}')
        ax.set_title(title, fontsize=rs_config['title_fontsize'])
        
        # 设置范围
        if waveform_data:
            y_min = min(wf['distance_deg'] for wf in waveform_data) - rs_config['distance_margin']
            y_max = max(wf['distance_deg'] for wf in waveform_data) + rs_config['distance_margin']
            ax.set_ylim(y_min, y_max)
        
        if max_time > 0:
            ax.set_xlim(0, min(max_time, rs_config['max_display_time']))
        
        ax.grid(True, alpha=rs_config['grid_alpha'])
        self._add_phase_legend(ax)
        plt.tight_layout()
        
        # 保存（文件名包含滤波信息）
        event_id = event_info.get('EventID', f"Event_{event_index}")
        filter_tag = self.get_filter_description()
        filename = self.config.output['record_section_template'].format(
            index=event_index,
            event_id=event_id,
            component=component,
            model=model_name,
            filter=filter_tag
        )
        save_path = self.figure_dir / filename
        plt.savefig(save_path, 
                   dpi=self.config.output['dpi'], 
                   bbox_inches=self.config.output['bbox_inches'])
        self.logger.info(f"📊 记录段已保存: {save_path}")
        plt.close()
        return fig

    def _get_relative_time_axis(self, trace, wf_data):
        """获取相对于发震时间的时间轴"""
        try:
            starttime = trace.stats.starttime
            origin_time = wf_data.get('event_origin_time') or wf_data.get('sac_origin_time')
            
            if origin_time:
                time_offset = starttime - origin_time
                return np.arange(trace.stats.npts) * trace.stats.delta + time_offset
            else:
                return np.arange(trace.stats.npts) * trace.stats.delta
        except Exception:
            return np.arange(trace.stats.npts) * trace.stats.delta

    def _add_arrivals_to_record_section(self, ax, event_info: Dict, y_offset: float, distance_deg: float):
        """添加理论到时"""
        try:
            if not self.current_model:
                return
            
            event_depth = event_info.get('Depth', 10)
            phase_config = self.config.seismic_phases
            
            arrivals = self.current_model.get_travel_times(
                source_depth_in_km=event_depth,
                distance_in_degree=distance_deg,
                phase_list=phase_config['phase_list']
            )
            
            phase_colors = phase_config['phase_colors']
            marked_phases = set()
            
            for arrival in arrivals:
                phase_name = arrival.name
                if phase_name in phase_colors and phase_name not in marked_phases:
                    line_height = phase_config['phase_line_height']
                    ax.plot([arrival.time, arrival.time], 
                           [y_offset - line_height, y_offset + line_height], 
                           color=phase_colors[phase_name], 
                           linewidth=phase_config['phase_linewidth'], 
                           alpha=phase_config['phase_alpha'])
                    
                    if phase_name in phase_config['labeled_phases']:
                        ax.text(arrival.time, y_offset + line_height + 0.3, phase_name, 
                               color=phase_colors[phase_name], 
                               fontsize=phase_config['phase_label_fontsize'], 
                               ha='center', va='bottom', fontweight='bold')
                    
                    marked_phases.add(phase_name)
        except Exception:
            pass

    def _add_phase_legend(self, ax):
        """添加震相图例"""
        phase_colors = self.config.seismic_phases['phase_colors']
        legend_elements = [
            Line2D([0], [0], color=phase_colors['P'], linewidth=2, label='P'),
            Line2D([0], [0], color=phase_colors['S'], linewidth=2, label='S'),
            Line2D([0], [0], color=phase_colors['PP'], linewidth=2, label='PP'),
            Line2D([0], [0], color=phase_colors['SS'], linewidth=2, label='SS'),
            Line2D([0], [0], color=phase_colors['PKP'], linewidth=2, label='PKP'),
            Line2D([0], [0], color=phase_colors['SKS'], linewidth=2, label='SKS')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, 
                 bbox_to_anchor=(0.99, 0.99), ncol=2)

    def run(self):
        """主运行函数"""
        print("🎨 东亚地震事件波形与震源机制可视化工具")
        print("=" * 60)
        
        try:
            # 加载事件
            events_df = self.load_events()
            
            if events_df.empty:
                self.logger.error("❌ 没有找到有波形数据的地震事件")
                return
            
            # 显示事件列表
            self.show_events(events_df)
            
            # 从配置获取选择的事件序号
            selected_indices = self.config.run_params['selected_event_indices']
            
            # 验证事件序号
            valid_indices = [idx for idx in selected_indices if 0 <= idx < len(events_df)]
            if not valid_indices:
                self.logger.error(f"❌ 配置的事件序号无效: {selected_indices}")
                return
            
            self.logger.info(f"\n🎯 选择了 {len(valid_indices)} 个事件: {valid_indices}")
            
            # 绘制东亚地图
            if self.config.run_params['plot_map']:
                self.logger.info("\n📊 绘制东亚地区事件地图...")
                self.plot_event_map(events_df, valid_indices)
            
            # 绘制波形分析
            if self.config.run_params['plot_waveforms']:
                self.logger.info("\n📊 开始波形分析...")
                for idx in valid_indices:
                    event_info = events_df.iloc[idx].to_dict()
                    self.plot_waveforms(event_info, idx)
            
            self.logger.info(f"\n✅ 完成！图片保存在: {self.figure_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数 - 执行波形可视化"""
    print("🎨 EASTASIA-FWI 波形可视化系统")
    print("="*60)
    
    try:
        # 初始化可视化器
        visualizer = EventWaveformVisualizer()
        
        # 运行可视化流程
        visualizer.run()
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断可视化")
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()