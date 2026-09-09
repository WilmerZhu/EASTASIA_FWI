"""
SPECFEM3D Global 网格划分参数设置模块
用于东亚地区正演模拟的网格生成和参数配置
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import subprocess
import yaml

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from config.base_config import BaseConfig

logger = logging.getLogger(__name__)


class SPECFEM3DSetup:
    """SPECFEM3D Global 设置类"""
    
    def __init__(self, 
                 specfem_root: Optional[str] = None,
                 simulation_dir: Optional[str] = None,
                 region_config: Optional[Dict] = None):
        """
        初始化SPECFEM3D设置
        
        Args:
            specfem_root: SPECFEM3D安装目录
            simulation_dir: 模拟输出目录  
            region_config: 区域配置参数
        """
        self.base_config = BaseConfig()
        self.specfem_root = Path(specfem_root) if specfem_root else self.base_config.dirs['specfem3d_globe']
        self.simulation_dir = Path(simulation_dir) if simulation_dir else self.base_config.dirs['simulations']
        self.region = region_config or self._region_from_base()
        
        # 确保目录存在
        self.simulation_dir.mkdir(parents=True, exist_ok=True)
        
        # 东亚区域默认参数
        self.default_params = self._get_eastasia_defaults()
    
    def _region_from_base(self) -> Dict:
        """从 base_config 构建 SPECFEM3D 兼容的区域字典"""
        r = self.base_config.region
        return {
            'NAME': r.get('name', 'EastAsia'),
            'LAT_MIN': r['lat_min'],
            'LAT_MAX': r['lat_max'],
            'LON_MIN': r['lon_min'],
            'LON_MAX': r['lon_max'],
        }
        
        # 模板文件路径
        self.templates_dir = Path(__file__).parent / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        
    def _get_eastasia_defaults(self) -> Dict:
        """获取东亚地区默认参数"""
        return {
            # === 区域参数 ===
            'ANGULAR_WIDTH_XI_IN_DEGREES': 110.0,    # 经度跨度 (60°E-170°E)
            'ANGULAR_WIDTH_ETA_IN_DEGREES': 75.0,     # 纬度跨度 (-15°N-60°N)
            'CENTER_LATITUDE_IN_DEGREES': 22.5,       # 中心纬度
            'CENTER_LONGITUDE_IN_DEGREES': 115.0,     # 中心经度
            'GAMMA_ROTATION_AZIMUTH': 0.0,            # 旋转角度
            
            # === 网格参数 ===
            'NPROC_XI': 8,                            # X方向处理器数量
            'NPROC_ETA': 6,                           # Y方向处理器数量
            'NCHUNKS': 1,                             # 区域块数量
            'NEX_XI': 240,                            # X方向单元数
            'NEX_ETA': 240,                           # Y方向单元数
            'NEX_PER_PROC_XI': 30,                    # 每个处理器X方向单元数
            'NEX_PER_PROC_ETA': 40,                   # 每个处理器Y方向单元数
            
            # === 时间参数 ===
            'RECORD_LENGTH_IN_MINUTES': 120.0,        # 记录长度(分钟)
            'DT': 0.05,                               # 时间步长(秒)
            'HDUR_MOVIE': 0.0,                        # 电影输出时长
            
            # === 模拟类型 ===
            'SIMULATION_TYPE': 1,                     # 1=正演, 2=伴随, 3=核函数
            'NOISE_TOMOGRAPHY': 0,                    # 噪声层析成像
            'SAVE_FORWARD': '.false.',                # 保存正演场
            'UNDO_ATTENUATION': '.false.',            # 撤销衰减
            
            # === 输出控制 ===
            'OUTPUT_SEISMOS_ASCII_TEXT': '.true.',    # ASCII地震图输出
            'OUTPUT_SEISMOS_SAC_ALPHANUM': '.true.',  # SAC格式输出
            'OUTPUT_SEISMOS_SAC_BINARY': '.false.',   # SAC二进制输出
            'ROTATE_SEISMOGRAMS_RT': '.false.',       # 旋转到径向-切向
            'WRITE_SEISMOGRAMS_BY_MASTER': '.false.', # 主进程写入
            
            # === 地球模型 ===
            'MODEL': 'EMC_model',                     # EMC NetCDF 模型 (prem 为参考模型)
            'REGIONAL_MESH_CUTOFF': '.true.',         # EMC 区域截断
            'REGIONAL_MESH_CUTOFF_DEPTH': 1000.0,     # 深度 1000 km
            'OCEANS': '.true.',                       # 包含海洋
            'ELLIPTICITY': '.true.',                  # 椭球校正
            'TOPOGRAPHY': '.true.',                   # 地形
            'GRAVITY': '.true.',                      # 重力
            'ROTATION': '.true.',                     # 地球自转
            'ATTENUATION': '.true.',                  # 衰减
            
            # === 数值参数 ===
            'ANGULAR_WIDTH_ETA_IN_DEGREES_HALFSPACE': 0.0,  # 半空间宽度
            'USE_ONE_LAYER_SB': '.false.',            # 使用单层边界
            'THICKNESS_OF_X_PML': 12.7d0,             # PML厚度X
            'THICKNESS_OF_Y_PML': 12.7d0,             # PML厚度Y
            'THICKNESS_OF_Z_PML': 12.7d0,             # PML厚度Z
            
            # === 高级选项 ===
            'USE_LDDRK': '.false.',                   # 低存储Runge-Kutta
            'INCREASE_CFL_FOR_LDDRK': '.false.',      # 增加CFL条件
            'RATIO_BY_WHICH_TO_INCREASE_IT': 1.4,     # CFL增加比率
            'PARTIAL_PHYS_DISPERSION_ONLY': '.false.', # 部分物理频散
            'EXACT_MASS_MATRIX_FOR_ROTATION': '.false.', # 精确质量矩阵
        }
    
    def calculate_mesh_parameters(self, 
                                 target_resolution_km: float = 50.0,
                                 max_period_s: float = 200.0,
                                 nproc_total: int = 48) -> Dict:
        """
        计算网格参数
        
        Args:
            target_resolution_km: 目标分辨率(公里)
            max_period_s: 最大周期(秒)  
            nproc_total: 总处理器数量
            
        Returns:
            计算得到的网格参数
        """
        print(f"🔢 计算网格参数...")
        print(f"  目标分辨率: {target_resolution_km} km")
        print(f"  最大周期: {max_period_s} s")
        print(f"  总处理器数: {nproc_total}")
        
        # 计算区域大小
        lon_span = self.region['LON_MAX'] - self.region['LON_MIN']
        lat_span = self.region['LAT_MAX'] - self.region['LAT_MIN']
        
        # 地球半径(km)
        R_EARTH = 6371.0
        
        # 计算弧度距离
        lon_span_rad = np.radians(lon_span)
        lat_span_rad = np.radians(lat_span)
        
        # 计算实际距离(km)  
        lon_distance_km = R_EARTH * lon_span_rad * np.cos(np.radians(self.region['LAT_MIN'] + lat_span/2))
        lat_distance_km = R_EARTH * lat_span_rad
        
        print(f"  区域跨度: {lon_span:.1f}° × {lat_span:.1f}°")
        print(f"  实际距离: {lon_distance_km:.0f} km × {lat_distance_km:.0f} km")
        
        # 根据分辨率计算单元数
        nex_xi_base = int(np.ceil(lon_distance_km / target_resolution_km))
        nex_eta_base = int(np.ceil(lat_distance_km / target_resolution_km))
        
        # 确保单元数是8的倍数(SPECFEM3D要求)
        nex_xi = ((nex_xi_base + 7) // 8) * 8
        nex_eta = ((nex_eta_base + 7) // 8) * 8
        
        # 计算处理器分布
        nproc_factors = self._factorize_nproc(nproc_total)
        nproc_xi, nproc_eta = self._optimize_processor_distribution(
            nex_xi, nex_eta, nproc_factors
        )
        
        # 确保单元数能被处理器数整除
        nex_per_proc_xi = nex_xi // nproc_xi
        nex_per_proc_eta = nex_eta // nproc_eta
        
        # 调整以确保整除
        nex_xi = nex_per_proc_xi * nproc_xi
        nex_eta = nex_per_proc_eta * nproc_eta
        
        # 重新计算实际分辨率
        actual_res_xi = lon_distance_km / nex_xi
        actual_res_eta = lat_distance_km / nex_eta
        
        # 计算时间步长(基于CFL条件)
        vs_min = 3.0  # 最小S波速度(km/s)
        cfl_factor = 0.3  # CFL安全因子
        dt_max = cfl_factor * min(actual_res_xi, actual_res_eta) / vs_min
        dt_suggested = min(dt_max, 0.05)  # 不超过0.05秒
        
        mesh_params = {
            'NEX_XI': nex_xi,
            'NEX_ETA': nex_eta,
            'NPROC_XI': nproc_xi,
            'NPROC_ETA': nproc_eta,
            'NEX_PER_PROC_XI': nex_per_proc_xi,
            'NEX_PER_PROC_ETA': nex_per_proc_eta,
            'DT': dt_suggested,
            'ANGULAR_WIDTH_XI_IN_DEGREES': lon_span,
            'ANGULAR_WIDTH_ETA_IN_DEGREES': lat_span,
            'CENTER_LONGITUDE_IN_DEGREES': (self.region['LON_MIN'] + self.region['LON_MAX']) / 2,
            'CENTER_LATITUDE_IN_DEGREES': (self.region['LAT_MIN'] + self.region['LAT_MAX']) / 2,
        }
        
        print(f"  计算结果:")
        print(f"    网格单元: {nex_xi} × {nex_eta}")
        print(f"    处理器分布: {nproc_xi} × {nproc_eta} = {nproc_xi * nproc_eta}")
        print(f"    实际分辨率: {actual_res_xi:.1f} × {actual_res_eta:.1f} km")
        print(f"    建议时间步长: {dt_suggested:.3f} s")
        
        return mesh_params
    
    def _factorize_nproc(self, nproc_total: int) -> List[Tuple[int, int]]:
        """分解处理器总数为因子对"""
        factors = []
        for i in range(1, int(np.sqrt(nproc_total)) + 1):
            if nproc_total % i == 0:
                factors.append((i, nproc_total // i))
        return factors
    
    def _optimize_processor_distribution(self, 
                                       nex_xi: int, 
                                       nex_eta: int, 
                                       nproc_factors: List[Tuple[int, int]]) -> Tuple[int, int]:
        """优化处理器分布"""
        best_ratio = float('inf')
        best_nproc = nproc_factors[0]
        
        for nproc_xi, nproc_eta in nproc_factors:
            # 确保网格能被处理器整除
            if nex_xi % nproc_xi == 0 and nex_eta % nproc_eta == 0:
                # 计算负载平衡比
                elements_per_proc_xi = nex_xi // nproc_xi
                elements_per_proc_eta = nex_eta // nproc_eta
                ratio = max(elements_per_proc_xi, elements_per_proc_eta) / min(elements_per_proc_xi, elements_per_proc_eta)
                
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_nproc = (nproc_xi, nproc_eta)
        
        return best_nproc
    
    def create_par_file(self, 
                       output_dir: str,
                       custom_params: Optional[Dict] = None,
                       mesh_params: Optional[Dict] = None) -> str:
        """
        创建Par_file参数文件
        
        Args:
            output_dir: 输出目录
            custom_params: 自定义参数
            mesh_params: 网格参数
            
        Returns:
            Par_file路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 合并参数
        params = self.default_params.copy()
        if mesh_params:
            params.update(mesh_params)
        if custom_params:
            params.update(custom_params)
        
        # 生成Par_file内容
        par_file_content = self._generate_par_file_content(params)
        
        # 写入文件
        par_file_path = output_path / "Par_file"
        with open(par_file_path, 'w') as f:
            f.write(par_file_content)
        
        print(f"✅ Par_file 已创建: {par_file_path}")
        return str(par_file_path)
    
    def _generate_par_file_content(self, params: Dict) -> str:
        """生成Par_file内容"""
        content = f"""#
# SPECFEM3D Global Par_file for East Asia FWI Project
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Region: {self.region['NAME']} ({self.region['LON_MIN']}°E - {self.region['LON_MAX']}°E, {self.region['LAT_MIN']}°N - {self.region['LAT_MAX']}°N)
#

# ====================================
# SIMULATION PARAMETERS
# ====================================

# Simulation type
SIMULATION_TYPE                 = {params['SIMULATION_TYPE']}
NOISE_TOMOGRAPHY               = {params['NOISE_TOMOGRAPHY']}
SAVE_FORWARD                   = {params['SAVE_FORWARD']}
UNDO_ATTENUATION              = {params['UNDO_ATTENUATION']}

# ====================================
# MESH PARAMETERS  
# ====================================

# Number of MPI processors
NPROC_XI                        = {params['NPROC_XI']}
NPROC_ETA                       = {params['NPROC_ETA']}
NCHUNKS                         = {params['NCHUNKS']}

# Number of elements in each direction
NEX_XI                          = {params['NEX_XI']}
NEX_ETA                         = {params['NEX_ETA']}
NEX_PER_PROC_XI                 = {params['NEX_PER_PROC_XI']}
NEX_PER_PROC_ETA                = {params['NEX_PER_PROC_ETA']}

# ====================================
# REGIONAL MESH PARAMETERS
# ====================================

# Angular width of the East Asia region
ANGULAR_WIDTH_XI_IN_DEGREES     = {params['ANGULAR_WIDTH_XI_IN_DEGREES']}
ANGULAR_WIDTH_ETA_IN_DEGREES    = {params['ANGULAR_WIDTH_ETA_IN_DEGREES']}

# Center coordinates
CENTER_LATITUDE_IN_DEGREES      = {params['CENTER_LATITUDE_IN_DEGREES']}
CENTER_LONGITUDE_IN_DEGREES     = {params['CENTER_LONGITUDE_IN_DEGREES']}
GAMMA_ROTATION_AZIMUTH          = {params['GAMMA_ROTATION_AZIMUTH']}

# ====================================
# TIME PARAMETERS
# ====================================

# Record length and time step
RECORD_LENGTH_IN_MINUTES        = {params['RECORD_LENGTH_IN_MINUTES']}
DT                              = {params['DT']}
HDUR_MOVIE                      = {params['HDUR_MOVIE']}

# ====================================
# EARTH MODEL PARAMETERS
# ====================================

# Reference Earth model
MODEL                           = {params['MODEL']}
REGIONAL_MESH_CUTOFF            = {params.get('REGIONAL_MESH_CUTOFF', '.false.')}
REGIONAL_MESH_CUTOFF_DEPTH      = {params.get('REGIONAL_MESH_CUTOFF_DEPTH', 1000.0)}d0
OCEANS                          = {params['OCEANS']}
ELLIPTICITY                     = {params['ELLIPTICITY']}
TOPOGRAPHY                      = {params['TOPOGRAPHY']}
GRAVITY                         = {params['GRAVITY']}
ROTATION                        = {params['ROTATION']}
ATTENUATION                     = {params['ATTENUATION']}

# ====================================
# OUTPUT PARAMETERS
# ====================================

# Seismogram output formats
OUTPUT_SEISMOS_ASCII_TEXT       = {params['OUTPUT_SEISMOS_ASCII_TEXT']}
OUTPUT_SEISMOS_SAC_ALPHANUM     = {params['OUTPUT_SEISMOS_SAC_ALPHANUM']}
OUTPUT_SEISMOS_SAC_BINARY       = {params['OUTPUT_SEISMOS_SAC_BINARY']}
ROTATE_SEISMOGRAMS_RT           = {params['ROTATE_SEISMOGRAMS_RT']}
WRITE_SEISMOGRAMS_BY_MASTER     = {params['WRITE_SEISMOGRAMS_BY_MASTER']}

# ====================================
# ADVANCED PARAMETERS
# ====================================

# Numerical scheme
USE_LDDRK                       = {params['USE_LDDRK']}
INCREASE_CFL_FOR_LDDRK          = {params['INCREASE_CFL_FOR_LDDRK']}
RATIO_BY_WHICH_TO_INCREASE_IT   = {params['RATIO_BY_WHICH_TO_INCREASE_IT']}

# Physical dispersion
PARTIAL_PHYS_DISPERSION_ONLY    = {params['PARTIAL_PHYS_DISPERSION_ONLY']}
EXACT_MASS_MATRIX_FOR_ROTATION  = {params['EXACT_MASS_MATRIX_FOR_ROTATION']}

# PML absorbing boundaries
THICKNESS_OF_X_PML              = {params['THICKNESS_OF_X_PML']}
THICKNESS_OF_Y_PML              = {params['THICKNESS_OF_Y_PML']}
THICKNESS_OF_Z_PML              = {params['THICKNESS_OF_Z_PML']}

# ====================================
# END OF PAR_FILE
# ====================================
"""
        return content
    
    def create_stations_file(self, 
                           stations_data: str,
                           output_dir: str,
                           network_filter: Optional[List[str]] = None) -> str:
        """
        创建STATIONS文件
        
        Args:
            stations_data: 台站数据CSV文件路径
            output_dir: 输出目录
            network_filter: 网络代码过滤器
            
        Returns:
            STATIONS文件路径
        """
        import pandas as pd
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 读取台站数据
        df = pd.read_csv(stations_data)
        
        # 筛选区域内台站
        df_filtered = df[
            (df['Latitude'] >= self.region['LAT_MIN']) &
            (df['Latitude'] <= self.region['LAT_MAX']) &
            (df['Longitude'] >= self.region['LON_MIN']) &
            (df['Longitude'] <= self.region['LON_MAX'])
        ]
        
        # 网络过滤
        if network_filter:
            df_filtered = df_filtered[df_filtered['Network'].isin(network_filter)]
        
        # 生成STATIONS文件
        stations_file = output_path / "STATIONS"
        with open(stations_file, 'w') as f:
            f.write(f"# STATIONS file for East Asia FWI\n")
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total stations: {len(df_filtered)}\n")
            f.write("#\n")
            f.write("# Format: STATION NETWORK LATITUDE LONGITUDE ELEVATION BURIAL\n")
            f.write("#\n")
            
            for _, station in df_filtered.iterrows():
                # SPECFEM3D STATIONS格式
                f.write(f"{station['Station']:<8} {station['Network']:<8} "
                       f"{station['Latitude']:>9.4f} {station['Longitude']:>10.4f} "
                       f"{station.get('Elevation', 0.0):>8.1f} {0.0:>8.1f}\n")
        
        print(f"✅ STATIONS文件已创建: {stations_file}")
        print(f"   包含 {len(df_filtered)} 个台站")
        
        return str(stations_file)
    
    def create_cmtsolution_file(self, 
                              event_data: Dict,
                              output_dir: str) -> str:
        """
        创建CMTSOLUTION文件
        
        Args:
            event_data: 事件数据字典
            output_dir: 输出目录
            
        Returns:
            CMTSOLUTION文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 解析事件时间
        event_time = pd.to_datetime(event_data['Date'])
        
        cmt_file = output_path / "CMTSOLUTION"
        with open(cmt_file, 'w') as f:
            # PDE行
            f.write(f"PDE {event_time.strftime('%Y %m %d %H %M %S.%f')[:-4]} "
                   f"{event_data['Latitude']:7.3f} {event_data['Longitude']:8.3f} "
                   f"{event_data['Depth']:5.1f} {event_data.get('Magnitude1_mb', 5.0):.1f} "
                   f"{event_data.get('Magnitude2_Ms', 5.0):.1f} "
                   f"{event_data.get('Region', 'EAST ASIA')}\n")
            
            # 事件名称
            event_id = event_data.get('EventID', f"{event_time.strftime('%Y%m%d%H%M')}")
            f.write(f"event name:     {event_id}\n")
            
            # CMT参数
            f.write(f"time shift:     {event_data.get('TimeShift', 0.0):8.4f}\n")
            f.write(f"half duration:  {event_data.get('HalfDuration', 1.0):8.4f}\n")
            f.write(f"latitude:       {event_data['Latitude']:8.4f}\n")
            f.write(f"longitude:      {event_data['Longitude']:9.4f}\n")
            f.write(f"depth:          {event_data['Depth']:8.4f}\n")
            
            # 力矩张量分量 (如果有的话)
            if 'Mrr' in event_data:
                f.write(f"Mrr:            {event_data['Mrr']:13.6e}\n")
                f.write(f"Mtt:            {event_data['Mtt']:13.6e}\n")
                f.write(f"Mpp:            {event_data['Mpp']:13.6e}\n")
                f.write(f"Mrt:            {event_data['Mrt']:13.6e}\n")
                f.write(f"Mrp:            {event_data['Mrp']:13.6e}\n")
                f.write(f"Mtp:            {event_data['Mtp']:13.6e}\n")
            else:
                # 使用默认力矩张量(震级相关)
                magnitude = event_data.get('Magnitude1_mb', 5.0)
                moment = 10**(1.5 * magnitude + 9.1)  # Hanks-Kanamori关系
                
                # 简单的走滑断层机制
                f.write(f"Mrr:            {0.0:13.6e}\n")
                f.write(f"Mtt:            {-moment:13.6e}\n")
                f.write(f"Mpp:            {moment:13.6e}\n")
                f.write(f"Mrt:            {0.0:13.6e}\n")
                f.write(f"Mrp:            {0.0:13.6e}\n")
                f.write(f"Mtp:            {0.0:13.6e}\n")
        
        print(f"✅ CMTSOLUTION文件已创建: {cmt_file}")
        return str(cmt_file)
    
    def setup_simulation_directory(self, 
                                 event_id: str,
                                 stations_file: str,
                                 event_data: Dict,
                                 custom_params: Optional[Dict] = None) -> str:
        """
        设置完整的模拟目录
        
        Args:
            event_id: 事件ID
            stations_file: 台站文件路径
            event_data: 事件数据
            custom_params: 自定义参数
            
        Returns:
            模拟目录路径
        """
        print(f"🏗️  设置模拟目录: {event_id}")
        
        # 创建模拟目录
        sim_dir = self.simulation_dir / f"{event_id}_simulation"
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 计算网格参数
        mesh_params = self.calculate_mesh_parameters()
        
        # 2. 创建Par_file
        par_file = self.create_par_file(
            output_dir=str(sim_dir),
            custom_params=custom_params,
            mesh_params=mesh_params
        )
        
        # 3. 复制/创建STATIONS文件
        if Path(stations_file).exists():
            shutil.copy(stations_file, sim_dir / "STATIONS")
        else:
            self.create_stations_file(stations_file, str(sim_dir))
        
        # 4. 创建CMTSOLUTION文件
        cmt_file = self.create_cmtsolution_file(event_data, str(sim_dir))
        
        # 5. 创建必要的目录结构
        (sim_dir / "OUTPUT_FILES").mkdir(exist_ok=True)
        (sim_dir / "DATABASES_MPI").mkdir(exist_ok=True)
        (sim_dir / "bin").mkdir(exist_ok=True)
        
        # 6. 复制SPECFEM3D可执行文件
        self._copy_specfem_executables(sim_dir)
        
        # 7. 创建作业提交脚本
        self._create_job_scripts(sim_dir, mesh_params)
        
        print(f"✅ 模拟目录设置完成: {sim_dir}")
        print(f"   包含文件: Par_file, STATIONS, CMTSOLUTION")
        print(f"   网格配置: {mesh_params['NEX_XI']}×{mesh_params['NEX_ETA']} 单元")
        print(f"   处理器配置: {mesh_params['NPROC_XI']}×{mesh_params['NPROC_ETA']} = {mesh_params['NPROC_XI']*mesh_params['NPROC_ETA']} 核")
        
        return str(sim_dir)
    
    def _copy_specfem_executables(self, sim_dir: Path):
        """复制SPECFEM3D可执行文件"""
        specfem_bin = self.specfem_root / "bin"
        sim_bin = sim_dir / "bin"
        
        # SPECFEM3D Globe 区域模拟: xmeshfem3D + xspecfem3D (无 xgenerate_databases)
        executables = [
            "xmeshfem3D",
            "xspecfem3D"
        ]
        
        for exe in executables:
            src = specfem_bin / exe
            dst = sim_bin / exe
            if src.exists():
                shutil.copy2(src, dst)
                os.chmod(dst, 0o755)
            else:
                logger.warning(f"可执行文件不存在: {src}")
    
    def _create_job_scripts(self, sim_dir: Path, mesh_params: Dict):
        """创建集群作业提交脚本"""
        nproc_total = mesh_params['NPROC_XI'] * mesh_params['NPROC_ETA']
        
        # SLURM作业脚本
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=SPECFEM3D_EastAsia
#SBATCH --nodes={max(1, nproc_total // 48)}
#SBATCH --ntasks={nproc_total}
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=specfem_%j.out
#SBATCH --error=specfem_%j.err

# 设置环境
module load intel/2021.4.0
module load impi/2021.4.0

# 进入工作目录
cd {sim_dir}

echo "======================================"
echo "SPECFEM3D East Asia Simulation"
echo "======================================"
echo "Job started at: $(date)"
echo "Working directory: $(pwd)"
echo "Number of processors: {nproc_total}"
echo "Mesh elements: {mesh_params['NEX_XI']} x {mesh_params['NEX_ETA']}"
echo "======================================"

# 步骤1: 生成网格
echo "Step 1: Generating mesh..."
mpirun -np {nproc_total} ./bin/xmeshfem3D
if [ $? -eq 0 ]; then
    echo "✅ Mesh generation completed successfully"
else
    echo "❌ Mesh generation failed"
    exit 1
fi

# 步骤2: 正演计算 (Globe 区域模拟无 xgenerate_databases)
echo "Step 2: Running forward simulation..."
mpirun -np {nproc_total} ./bin/xspecfem3D
if [ $? -eq 0 ]; then
    echo "✅ Forward simulation completed successfully"
else
    echo "❌ Forward simulation failed"
    exit 1
fi

echo "======================================"
echo "Job completed at: $(date)"
echo "======================================"
"""
        
        # 写入SLURM脚本
        slurm_file = sim_dir / "run_specfem.sh"
        with open(slurm_file, 'w') as f:
            f.write(slurm_script)
        os.chmod(slurm_file, 0o755)
        
        # PBS作业脚本(备用)
        pbs_script = f"""#!/bin/bash
#PBS -N SPECFEM3D_EastAsia
#PBS -l nodes={max(1, nproc_total // 48)}:ppn=48
#PBS -l walltime=24:00:00
#PBS -q normal
#PBS -V

cd {sim_dir}
mpirun -np {nproc_total} ./bin/xmeshfem3D
mpirun -np {nproc_total} ./bin/xspecfem3D
"""
        
        pbs_file = sim_dir / "run_specfem.pbs"
        with open(pbs_file, 'w') as f:
            f.write(pbs_script)
        os.chmod(pbs_file, 0o755)
        
        print(f"✅ 作业脚本已创建: run_specfem.sh, run_specfem.pbs")
    
    def validate_setup(self, sim_dir: str) -> Dict:
        """验证模拟设置"""
        sim_path = Path(sim_dir)
        
        validation_results = {
            'par_file_exists': (sim_path / "Par_file").exists(),
            'stations_exists': (sim_path / "STATIONS").exists(),
            'cmtsolution_exists': (sim_path / "CMTSOLUTION").exists(),
            'executables_exist': all((sim_path / "bin" / exe).exists() 
                                   for exe in ["xmeshfem3D", "xspecfem3D"]),
            'output_dirs_exist': all((sim_path / d).exists() 
                                   for d in ["OUTPUT_FILES", "DATABASES_MPI"]),
            'job_scripts_exist': (sim_path / "run_specfem.sh").exists()
        }
        
        all_valid = all(validation_results.values())
        
        print(f"🔍 模拟设置验证:")
        for check, result in validation_results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {check}")
        
        if all_valid:
            print(f"✅ 模拟设置验证通过，可以开始计算!")
        else:
            print(f"❌ 模拟设置存在问题，请检查!")
        
        return validation_results


def main():
    """主函数 - SPECFEM3D设置示例"""
    print("🔧 SPECFEM3D Global 东亚区域设置")
    print("="*50)
    
    # 初始化设置器
    setup = SPECFEM3DSetup()
    
    # 示例事件数据
    event_data = {
        'EventID': 'TEST_EVENT_001',
        'Date': '2008-05-12T06:28:01.400',
        'Latitude': 31.002,
        'Longitude': 103.322,
        'Depth': 19.0,
        'Magnitude1_mb': 7.9,
        'Region': 'EASTERN SICHUAN, CHINA'
    }
    
    # 设置模拟目录
    stations_file = setup.base_config.dirs['stations'] / f"{setup.region['NAME']}_stations.csv"
    
    if stations_file.exists():
        sim_dir = setup.setup_simulation_directory(
            event_id=event_data['EventID'],
            stations_file=str(stations_file),
            event_data=event_data
        )
        
        # 验证设置
        setup.validate_setup(sim_dir)
        
        print(f"\n🚀 准备运行SPECFEM3D:")
        print(f"   cd {sim_dir}")
        print(f"   sbatch run_specfem.sh")
        
    else:
        print(f"❌ 台站文件不存在: {stations_file}")
        print("   请先运行 1_1_Query_stations.py 生成台站数据")


if __name__ == "__main__":
    main()