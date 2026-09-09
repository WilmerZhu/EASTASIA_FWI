"""
1_5_Process_velocity_models.py: 
速度模型预处理模块
================================================================

功能描述:
----------
速度模型预处理和标准化模块，将多种格式的地震速度模型转换为统一的NetCDF和CSV格式。
支持HDF5、NetCDF、GeoCSV、TXT、Excel等多种输入格式，同时输出共同覆盖区域和原始模型区域两个版本，
为模型空间分析和全波形反演提供标准化的速度模型数据。

核心功能:
----------
1. ✅ 多格式模型读取
   - HDF5格式 (SinoScope1.0)
   - NetCDF格式 (EARA2024, FWEA23)
   - GeoCSV格式 (CSEM_Japan, FWEA18, TP2019等)
   - 文本格式 (USTClitho2.0, SWChinaCVM等)
   - Excel格式 (Cao_SETibet)
   - 多深度文件格式 (Chen_SChinaSea, Wu_NETibet等)

2. ✅ 双区域输出
   - 共同覆盖区域版本 (*_standardized.nc/csv): 用于聚类分析和模型对比
   - 原始模型区域版本 (*_original.nc/csv): 保留完整模型信息
   - 100%有效数据覆盖的共同区域计算
   - 自动裁剪到共同有效区域

3. ✅ 参数标准化
   - 统一参数命名 (vpv, vph, vsv, vsh, rho, vs, vp)
   - 单位标准化 (速度: km/s, 密度: 保留原始单位)
   - 坐标系统统一 (WGS84, latitude/longitude/depth)
   - 深度单位自动转换和验证

4. ✅ 深度平移支持
   - 支持模型深度平移（如SinoScope1.0: 20-2820 km → 0-2800 km）
   - 自动应用配置的深度偏移
   - 深度范围验证和调整

5. ✅ 数据质量控制
   - 坐标范围验证
   - 数据完整性检查
   - 参数值范围验证
   - 缺失值处理

6. ✅ 元数据和报告生成
   - 完整的模型元数据 (ModelMetadata)
   - 参数值范围统计
   - 处理报告和统计信息
   - JSON格式元数据输出

使用方法:
----------
通过修改 config.runtime 参数来控制处理行为：

1. 处理所有模型:
   ```python
   config = VelocityModelConfig()
   config.runtime['model_names'] = None
   config.runtime['compute_common_region'] = False
   processor = VelocityModelProcessor(config)
   processor.process_all_models()
   ```

2. 处理指定模型列表:
   ```python
   config = VelocityModelConfig()
   config.runtime['model_names'] = ['2022_SinoScope1.0', '2024_EARA2024']
   config.runtime['compute_common_region'] = True
   processor = VelocityModelProcessor(config)
   processor.process_all_models()
   ```

3. 处理单个模型:
   ```python
   config = VelocityModelConfig()
   config.runtime['single_model'] = '2022_SinoScope1.0'
   processor = VelocityModelProcessor(config)
   processor.process_all_models()
   ```

4. 列出所有可用模型:
   ```python
   config = VelocityModelConfig()
   config.runtime['list_models'] = True
   processor = VelocityModelProcessor(config)
   # 运行 main() 会自动列出模型
   ```

配置说明:
----------
通过 VelocityModelConfig 类配置处理参数:
- models: 模型清单和格式配置
- standard_parameters: 标准化参数列表
- standard_units: 标准单位定义
- qc: 数据质量控制参数
- processing: 处理选项 (保存格式、压缩等)
- runtime: 运行时参数 (模型选择、共同区域计算)

输出文件:
----------
- {model_name}_original.nc: 原始模型区域的NetCDF文件
- {model_name}_original.csv: 原始模型区域的CSV文件
- {model_name}_standardized.nc: 共同覆盖区域的NetCDF文件
- {model_name}_standardized.csv: 共同覆盖区域的CSV文件
- {model_name}_metadata.json: 模型元数据
- processing_summary.json: 批量处理总结

科学原理:
----------
- 速度模型标准化: 统一不同来源模型的格式和单位，便于模型对比和分析
- 共同覆盖区域: 计算所有模型的交集区域，确保模型对比的公平性和有效性
- 参数映射: 将不同模型的参数名称映射到标准参数集，支持各向异性和各向同性模型
- 深度校正: 处理不同模型的深度定义差异（如海平面vs地表），确保深度一致性
- 数据插值: 在需要时进行网格插值，统一模型分辨率，便于后续分析

作者: EASTASIA-FWI Team
版本: v2.0 (使用config参数)
"""

import numpy as np
import pandas as pd
import xarray as xr
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json
import warnings
from dataclasses import dataclass, asdict
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.base_config import BaseConfig

warnings.filterwarnings('ignore')


@dataclass
class ModelMetadata:
    """速度模型元数据"""
    name: str
    full_name: str
    year: int
    format: str
    reference: str
    region: str
    original_parameters: List[str]
    coordinate_system: str
    depth_unit: str
    velocity_unit: str
    density_unit: str
    data_shape: Optional[Tuple[int, int, int]] = None
    lat_range: Optional[Tuple[float, float]] = None
    lon_range: Optional[Tuple[float, float]] = None
    depth_range: Optional[Tuple[float, float]] = None
    resolution: Optional[Dict[str, float]] = None
    # 🔥 新增：各参数的值范围
    parameter_ranges: Optional[Dict[str, Dict[str, float]]] = None
    standardized_lat_range: Optional[Tuple[float, float]] = None
    standardized_lon_range: Optional[Tuple[float, float]] = None
    standardized_depth_range: Optional[Tuple[float, float]] = None
    cropped_shape: Optional[Tuple[int, int, int]] = None
    depth_shift_applied: bool = False
    valid_coverage_percent: Optional[float] = None
    notes: str = ""


class VelocityModelConfig:
    """速度模型处理配置"""
    
    def __init__(self):
        """初始化配置参数"""
        
        # ============ 路径配置 ============
        self.paths = {
            'input_dir': Path('data/models/metadata'),  # 与 metadata_dir 相同，模型文件存放于此
            'output_dir': Path('data/models/processed'),
            'metadata_dir': Path('data/models/metadata'),
            'log_dir': Path('logs'),
            'results_dir': Path('results/vel_models')
        }
        
        # ============ 模型清单 ============
        # 格式说明:
        # - h5: HDF5格式 (SinoScope1.0)
        # - nc: NetCDF格式 (EARA2024, FWEA23)
        # - geocsv: GeoCSV格式，| 分隔符 (CSEM_Japan, FWEA18, TP2019, Banda, KEA20, SASSY21)
        # - txt: 简单文本格式，空格分隔 (USTClitho2.0, SWChinaCVM等)
        # - txt_3d: 3D文本格式 (CSES_VM1.0)
        # - txt_11col: 11列文本格式 (ASIA2024)
        # - xlsx: Excel格式 (Cao_SETibet)
        # - multi_depth: 多深度文件格式 (Chen_SChinaSea, Wu_NETibet, Kumar_Pamir, CSRM1.0)
        
        self.models = {
            # ========== 已有模型 (H5/NC格式) ==========
            '2022_SinoScope1.0': {
                'file': '2022_SinoScope1.0/FWI_SinoScope_1.0_JMa_ET_AL_2022.h5',
                'format': 'h5',
                'reference': 'Ma et al., 2022, GRL',
                'region': 'China and adjacent regions',
                'depth_shift': -20.0,
                'params': ['vp', 'vsv', 'vsh', 'rho']
            },
            '2024_EARA2024': {
                'file': '2024_EARA2024/EARA2024.r0.0-n4.nc',
                'format': 'nc',
                'reference': 'Xi et al., 2024, GJI',
                'region': 'East Asia and northwestern Pacific',
                'depth_shift': 0.0,
                'params': ['vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho', 'vs', 'vp', 'vs_ref', 'vp_ref']
            },
            '2024_FWEA23': {
                'file': '2024_FWEA23/FWEA23.r0.0.nc',
                'format': 'nc',
                'reference': 'Liu et al., 2024, EPSL',
                'region': 'East Asia FWI model',
                'depth_shift': 0.0,
                'params': ['vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho', 'qmu', 'vs0', 'vp0']
            },
            
            # ========== GeoCSV 格式 (| 分隔符) ==========
            '2018_CSEM_Japan': {
                'file': '2018_CSEM_Japan/csem-japan-2019.12.01.csv',
                'format': 'geocsv',
                'reference': 'Simute et al., 2016, JGR',
                'region': 'Japan and surrounding regions',
                'depth_shift': 0.0,
                'params': ['vp', 'vsv', 'vsh', 'vs', 'rho'],
                'rho_unit': 'g/cm3'  # 需要转换为 kg/m3
            },
            '2018_FWEA18': {
                'file': '2018_FWEA18/FWEA18_kmps.csv',
                'format': 'geocsv',
                'reference': 'Chen et al., 2018, JGR',
                'region': 'East Asia',
                'depth_shift': 0.0,
                'params': ['vpv', 'vph', 'vsv', 'vsh', 'vs', 'rho'],
                'rho_unit': 'g/cm3'
            },
            '2020_TP2019': {
                'file': '2020_TP2019/TP2019.csv',
                'format': 'geocsv',
                'reference': 'Tao et al., 2020, G-cubed',
                'region': 'Tibetan Plateau',
                'depth_shift': 0.0,
                'params': ['vp', 'vsv', 'vsh', 'vs', 'rho'],
                'rho_unit': 'g/cm3'
            },
            '2021_Banda_ANT_CrustVs_2020': {
                'file': '2021_Banda_ANT_CrustVs_2020/Banda-ANT-CrustVs-2020.r0.0.csv',
                'format': 'geocsv',
                'reference': 'Lebedev et al., 2021',
                'region': 'Banda Sea region',
                'depth_shift': 0.0,
                'params': ['vp', 'vsv', 'vsh', 'vs', 'rho'],
                'rho_unit': 'g/cm3'
            },
            '2021_KEA20': {
                'file': '2021_KEA20/KEA20.r0.0.csv',
                'format': 'geocsv',
                'reference': 'Chen et al., 2021, EPSL',
                'region': 'Korea and East Asia',
                'depth_shift': 0.0,
                'params': ['vp', 'vsv', 'vsh', 'vs', 'rho'],
                'rho_unit': 'g/cm3'
            },
            '2022_SASSY21': {
                'file': '2022_SASSY21/SASSY21.r0.0-n4.csv',
                'format': 'geocsv',
                'reference': 'Schaeffer et al., 2022',
                'region': 'Southeast Asia',
                'depth_shift': 0.0,
                'params': ['vp', 'vsv', 'vsh', 'vs', 'rho'],
                'rho_unit': 'g/cm3'
            },
            
            # ========== 简单文本格式 (空格分隔) ==========
            '2022_USTClitho2.0': {
                'file': '2022_USTClitho2.0/USTClitho2.0.txt',
                'format': 'txt',
                'reference': 'Xiao et al., 2022, JGR',
                'region': 'Chinese mainland',
                'depth_shift': 0.0,
                'params': ['vp', 'vs'],
                'columns': ['longitude', 'latitude', 'depth', 'vp', 'vs'],
                'separator': r'\s+'
            },
            '2021_SWChinaCVM-1.0': {
                'file': '2021_SWChinaCVM-1.0/SWChinaCVMv1.0.txt',
                'format': 'txt',
                'reference': 'Yao et al., 2021',
                'region': 'Southwest China',
                'depth_shift': 0.0,
                'params': ['vp', 'vs'],
                'columns': ['longitude', 'latitude', 'depth', 'vp', 'vs'],
                'separator': r'\s+'
            },
            '2023_SWChinaCVM-2.0': {
                'file': '2023_SWChinaCVM-2.0/SWChinaCVMv2.0.txt',
                'format': 'txt',
                'reference': 'Yao et al., 2023',
                'region': 'Southwest China',
                'depth_shift': 0.0,
                'params': ['vp', 'vs'],
                'columns': ['longitude', 'latitude', 'depth', 'vp', 'vs'],
                'separator': r'\s+'
            },
            '2022_Chen_SChina': {
                'file': '2022_Chen_SChina/Vs_model_ESC.txt',
                'format': 'txt',
                'reference': 'Chen et al., 2022',
                'region': 'Eastern South China',
                'depth_shift': 0.0,
                'params': ['vs'],
                'columns': ['longitude', 'latitude', 'depth', 'vs'],
                'separator': r'\s+'
            },
            # '2022_Toyokuni_SEAsia' 暂时禁用：数据为不规则网格（每个点深度不同），
            # 数据量过大（658*596*17776），不适合转换为规则 3D 数组
            # '2022_Toyokuni_SEAsia': {
            #     'file': '2022_Toyokuni_SEAsia/model_global_sea.dat',
            #     'format': 'txt',
            #     'reference': 'Toyokuni et al., 2022',
            #     'region': 'Southeast Asia',
            #     'depth_shift': 0.0,
            #     'params': ['vp'],
            #     'columns': ['latitude', 'longitude', 'depth', 'vp', 'dVp', 'hitcount'],
            #     'separator': r'\s+'
            # },
            '2025_Gao_NEChina': {
                'file': '2025_Gao_NEChina/Vp.xyz',
                'format': 'txt',
                'reference': 'Gao et al., 2025',
                'region': 'Northeast China',
                'depth_shift': 0.0,
                'params': ['vp'],
                'columns': ['longitude', 'latitude', 'depth', 'vp'],
                'separator': r'\s+'
            },
            '2025_Han_NEChina': {
                'file': '2025_Han_NEChina/vel_mod',
                'format': 'txt',
                'reference': 'Han et al., 2025',
                'region': 'Northeast China',
                'depth_shift': 0.0,
                'params': ['vs'],
                'columns': ['longitude', 'latitude', 'depth', 'vs'],
                'separator': r'\s+',
                'skiprows': 1
            },
            '2026_Li_NEChina': {
                'file': '2026_Li_NEChina/Dongbei_ZW.csv',
                'format': 'txt',
                'reference': 'Li et al., 2026',
                'region': 'Northeast China',
                'depth_shift': 0.0,
                'depth_negate': True,  # 深度定义为负向下（0~-62.5 km），取反后得正常深度
                'params': ['vs'],
                'columns': ['longitude', 'latitude', 'depth', 'vs'],
                'separator': ',',
                'skiprows': 1  # 跳过表头行
            },
            '2026_Li_SChina': {
                'file': '2026_Li_SChina/Huanan_CJQ.txt',
                'format': 'txt',
                'reference': 'Li et al., 2026',
                'region': 'South China',
                'depth_shift': 0.0,
                'params': ['vs'],
                'columns': ['longitude', 'latitude', 'depth', 'vs'],
                'separator': r'\s+'
            },
            
            # ========== 特殊文本格式 ==========
            '2024_CSES_VM1.0': {
                'file': '2024_CSES_VM1.0/CSES_VM1.0.txt',
                'format': 'txt_3d',  # 特殊3D格式
                'reference': 'CSES team, 2024',
                'region': 'China',
                'depth_shift': 0.0,
                'params': ['vp', 'vs']
            },
            '2024_ASIA2024': {
                'file': '2024_ASIA2024/fullmodel.txt',
                'format': 'txt_11col',  # 11列格式，速度单位 m/s
                'reference': 'Dou et al., 2024, Earth-Science Reviews',
                'region': 'Asia',
                'depth_shift': 0.0,
                'params': ['vp', 'vs'],
                'velocity_unit': 'm/s'  # 需要转换为 km/s
            },
            
            # ========== Excel 格式 ==========
            '2023_Cao_SETibet': {
                'file': '2023_Cao_SETibet/1-s2.0-S004019512200484X-mmc1.xlsx',
                'format': 'xlsx',
                'reference': 'Cao et al., 2023, Tectonophysics',
                'region': 'Southeast Tibet',
                'depth_shift': 0.0,
                'params': ['vs'],
                'skiprows': 2,
                'usecols': [1, 2, 3, 4],  # lat, lon, depth, vs
                'column_names': ['latitude', 'longitude', 'depth', 'vs']
            },
            
            # ========== 多深度文件格式 ==========
            '2021_Chen_SChinaSea': {
                'file': '2021_Chen_SChinaSea/scsmodel',  # 目录
                'format': 'multi_depth',
                'reference': 'Chen et al., 2021',
                'region': 'South China Sea',
                'depth_shift': 0.0,
                'params': ['vs'],
                'file_pattern': '{depth:03d}km.txt',
                'depth_range': list(range(10, 251, 5)),  # 10-250 km, step 5
                'columns': ['longitude', 'latitude', 'vs'],
                'separator': r'\s+'
            },
            '2023_Wu_NETibet': {
                'file': '2023_Wu_NETibet/Anisotropic_Vs_model',  # 目录
                'format': 'multi_depth',
                'reference': 'Wu et al., 2023',
                'region': 'Northeast Tibet',
                'depth_shift': 0.0,
                'params': ['vs'],
                'file_pattern': 'Vs_{depth}km.dat',
                'depth_range': list(range(5, 51, 5)) + [60, 70, 80],  # 5-50 step 5, then 60/70/80
                'columns': ['longitude', 'latitude', 'vs'],
                'separator': r'\s+'
            },
            '2022_Kumar_Pamir': {
                'file': '2022_Kumar_Pamir/3D_Shear_velociy_model/data',  # 目录
                'format': 'multi_lon',  # 按经度分文件
                'reference': 'Kumar et al., 2022',
                'region': 'Pamir',
                'depth_shift': 0.0,
                'params': ['vs'],
                'file_pattern': '*.rev.xyz',  # 使用通配符扫描目录
                'columns': ['longitude', 'latitude', 'depth', 'vs'],  # 文件内有4列
                'separator': r'\s+'
            },
            '2023_CSRM1.0': {
                'file': '2023_CSRM1.0/CSRM1.0_2024',  # 目录
                'format': 'multi_depth_mod',
                'reference': 'CSRM team, 2023',
                'region': 'China',
                'depth_shift': 0.0,
                'params': ['vp', 'vs'],
                'file_pattern': 'CSRM1.0_{depth}_km.mod',
                'columns': ['longitude', 'latitude', 'vs', 'col3', 'vp', 'col5'],  # 第3/5列为vs/vp
                'separator': r'\s+'
            }
        }
        
        # ============ 标准化参数配置 ============
        self.standard_parameters = [
            'latitude', 'longitude', 'depth',
            'vpv', 'vph', 'vsv', 'vsh',
            'eta',
            'rho', 'qmu',
            'vs0', 'vp0',
            'vs', 'vp'
        ]
        
        self.standard_units = {
            'latitude': 'degrees_north',
            'longitude': 'degrees_east',
            'depth': 'km',
            'vpv': 'km/s',
            'vph': 'km/s',
            'vsv': 'km/s',
            'vsh': 'km/s',
            'rho': 'kg/m3',
            'vs0': 'km/s',
            'vp0': 'km/s',
            'vs': 'km/s',
            'vp': 'km/s'
        }
        
        # ============ 数据质量控制（仅检查坐标范围，保留原始速度参数值） ============
        self.qc = {
            'lat_range': (-90.0, 90.0),
            'lon_range': (-180.0, 360.0),
            'depth_range': (0.0, 3000.0)
            # 注：不再对速度参数进行 QC，保留原始模型数据
        }
        
        # ============ 处理选项 ============
        self.processing = {
            'save_standardized_region': True,      # 🔥 新增：保存共同覆盖区域版本
            'save_original_region': True,    # 🔥 新增：保存原始模型区域版本
            'require_full_coverage': True,
            'keep_original_resolution': True,
            'chunk_size': 100000,
            'save_csv': True,
            'save_netcdf': True,
            'save_netcdf_metadata': True,
            'save_completeness_report': True,
            'csv_sort_order': ['depth', 'latitude', 'longitude'],
            'netcdf_format': '3d',
            'compress_netcdf': True,
            'float_precision': 'float32',
            'generate_report': True
        }
        
        # ============ 共同区域缓存 ============
        self.valid_standardized_region: Optional[Dict[str, Tuple[float, float]]] = None
        
        # ============ 运行时参数配置 ============
        # 这些参数可以通过修改config来设置，替代命令行参数
        # 详细使用说明请参考文件头部文档
        # 
        # 也可以使用 set_runtime_params() 辅助方法来设置参数
        self.runtime = {
            # 'model_names': None,  # None表示处理所有模型，或指定模型名称列表，如 ['2022_SinoScope1.0', '2024_EARA2024']
            # 'compute_common_region': False,  # 是否计算共同覆盖区域（需要多个模型）
            'model_names': ['2026_Li_NEChina', '2026_Li_SChina'],  # None 表示处理所有模型；或指定列表如 ['2022_SinoScope1.0', '2024_EARA2024']
            # 'model_names': ['2022_SinoScope1.0', '2024_EARA2024', '2024_FWEA23'],  # 处理3个模型
            'compute_common_region': False,  # 计算共同覆盖区域（需要多个模型）
            'list_models': False,  # 是否只列出模型列表（不执行处理）
            'single_model': None  # 处理单个模型（优先级高于model_names），如 '2022_SinoScope1.0'
        }
    
    def set_runtime_params(self, model_names: Optional[List[str]] = None,
                          single_model: Optional[str] = None,
                          compute_common_region: bool = False,
                          list_models: bool = False) -> None:
        """
        设置运行时参数的辅助方法
        
        Args:
            model_names: 要处理的模型名称列表，None表示处理所有模型
            single_model: 处理单个模型（优先级高于model_names）
            compute_common_region: 是否计算共同覆盖区域
            list_models: 是否只列出模型列表
        """
        if single_model is not None:
            self.runtime['single_model'] = single_model
            self.runtime['model_names'] = None  # 清除model_names，因为single_model优先级更高
        elif model_names is not None:
            self.runtime['model_names'] = model_names
            self.runtime['single_model'] = None
        
        self.runtime['compute_common_region'] = compute_common_region
        self.runtime['list_models'] = list_models


class VelocityModelProcessor:
    """速度模型预处理主类 (双区域输出)"""
    
    def __init__(self, config: Optional[VelocityModelConfig] = None):
        """初始化处理器"""
        self.config = config or VelocityModelConfig()
        self.base_config = BaseConfig()
        self.logger = self._setup_logger()
        
        for path in self.config.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("="*80)
        self.logger.info("速度模型预处理器初始化完成 (双区域输出)")
        self.logger.info("  • 🔥 同时保存共同覆盖区域和原始模型区域")
        self.logger.info("  • 🔥 计算100%有效数据覆盖的共同区域")
        self.logger.info("  • 🔥 保留原始模型完整信息")
        self.logger.info("  • SinoScope1.0 深度平移：20-2820 km → 0-2800 km")
        self.logger.info("  • CSV 排序：depth(慢) → latitude → longitude(快)")
        self.logger.info("="*80)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        log_file = self.config.paths['log_dir'] / f'VelModelProcessor_{datetime.now():%Y%m%d_%H%M%S}.log'
        
        logger = logging.getLogger('VelocityModelProcessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            fh = logging.FileHandler(log_file, encoding='utf-8')
            ch = logging.StreamHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)
        
        return logger

    # ==================== 工具方法：H5 文件结构探测 ====================
    
    def inspect_h5_structure(self, file_path: Path) -> Dict:
        """深度探测 H5 文件结构（调试用）"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔍 深度探测 H5 文件结构: {file_path.name}")
        self.logger.info("="*80)
        
        structure = {}
        
        def print_attrs(name, obj):
            """递归打印 HDF5 对象属性"""
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                self.logger.info(f"{indent}📊 Dataset: {name}")
                self.logger.info(f"{indent}   Shape: {obj.shape}")
                self.logger.info(f"{indent}   Dtype: {obj.dtype}")
                obj_size = obj.size if hasattr(obj, 'size') and obj.size is not None else 0
                if obj_size < 20:
                    self.logger.info(f"{indent}   Values: {obj[:]}")
                else:
                    self.logger.info(f"{indent}   Range: [{obj[:].min():.2f}, {obj[:].max():.2f}]")
                structure[name] = {
                    'type': 'dataset',
                    'shape': obj.shape,
                    'dtype': str(obj.dtype)
                }
            elif isinstance(obj, h5py.Group):
                self.logger.info(f"{indent}📁 Group: {name}")
                structure[name] = {'type': 'group'}
        
        with h5py.File(file_path, 'r') as f:
            self.logger.info("\n文件根目录:")
            self.logger.info(f"  Keys: {list(f.keys())}\n")
            f.visititems(print_attrs)
        
        self.logger.info("="*80 + "\n")
        return structure

    # ==================== 核心：计算有效覆盖区域（可选） ====================
    
    def compute_valid_standardized_region(self, model_names: Optional[List[str]] = None) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        计算指定模型的共同有效数据区域
        
        Args:
            model_names: 要计算共同区域的模型列表，默认使用所有模型
        
        Returns:
            共同有效区域字典，如果无法计算则返回 None
        """
        if model_names is None:
            model_names = list(self.config.models.keys())
        
        # 如果只有一个模型，跳过共同区域计算
        if len(model_names) < 2:
            self.logger.info("⚠️ 少于2个模型，跳过共同区域计算")
            self.config.valid_standardized_region = None
            return None
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"🔥 计算 {len(model_names)} 个模型的有效数据覆盖区域")
        self.logger.info("="*80)
        
        all_coords = {}
        
        # 第一步：读取所有模型的坐标范围
        for model_name in model_names:
            try:
                model_info = self.config.models[model_name]
                input_file = self.config.paths['input_dir'] / model_info['file']
                
                self.logger.info(f"\n分析 {model_name} 的坐标范围:")
                
                # 读取模型获取坐标信息（简化版，只获取坐标）
                data_3d, coords = self.read_model(model_name)
                
                all_coords[model_name] = {
                    'latitude': np.sort(coords['latitude']),
                    'longitude': np.sort(coords['longitude']),
                    'depth': np.sort(coords['depth'])
                }
                
                self.logger.info(f"  纬度: {coords['latitude'].min():.2f}° ~ {coords['latitude'].max():.2f}°")
                self.logger.info(f"  经度: {coords['longitude'].min():.2f}° ~ {coords['longitude'].max():.2f}°")
                self.logger.info(f"  深度: {coords['depth'].min():.1f} ~ {coords['depth'].max():.1f} km")
                
            except Exception as e:
                self.logger.warning(f"  ⚠️ 无法读取 {model_name}: {e}")
                continue
        
        if len(all_coords) < 2:
            self.logger.warning("⚠️ 有效模型少于2个，跳过共同区域计算")
            self.config.valid_standardized_region = None
            return None
        
        # 第二步：计算坐标系交集范围
        self.logger.info("\n计算坐标系交集范围...")
        
        lat_min = max([coords['latitude'].min() for coords in all_coords.values()])
        lat_max = min([coords['latitude'].max() for coords in all_coords.values()])
        lon_min = max([coords['longitude'].min() for coords in all_coords.values()])
        lon_max = min([coords['longitude'].max() for coords in all_coords.values()])
        depth_min = max([coords['depth'].min() for coords in all_coords.values()])
        depth_max = min([coords['depth'].max() for coords in all_coords.values()])
        
        # 检查是否有有效交集
        if lat_min >= lat_max or lon_min >= lon_max or depth_min >= depth_max:
            self.logger.warning("⚠️ 模型之间没有重叠区域，跳过共同区域计算")
            self.config.valid_standardized_region = None
            return None
        
        self.logger.info(f"  纬度: {lat_min:.2f}° ~ {lat_max:.2f}°")
        self.logger.info(f"  经度: {lon_min:.2f}° ~ {lon_max:.2f}°")
        self.logger.info(f"  深度: {depth_min:.1f} ~ {depth_max:.1f} km")
        
        # 第三步：计算分辨率（取最粗）
        resolutions = {}
        for name, coords in all_coords.items():
            lat_diff = np.diff(coords['latitude'])
            lon_diff = np.diff(coords['longitude'])
            depth_diff = np.diff(coords['depth'])
            
            resolutions[name] = {
                'lat': float(np.mean(lat_diff)) if len(lat_diff) > 0 else 1.0,
                'lon': float(np.mean(lon_diff)) if len(lon_diff) > 0 else 1.0,
                'depth': float(np.mean(depth_diff)) if len(depth_diff) > 0 else 10.0
            }
        
        standardized_res = {
            'lat': max([r['lat'] for r in resolutions.values()]),
            'lon': max([r['lon'] for r in resolutions.values()]),
            'depth': max([r['depth'] for r in resolutions.values()])
        }
        
        self.logger.info(f"\n统一网格分辨率（取最粗）:")
        self.logger.info(f"  纬度: {standardized_res['lat']:.4f}°")
        self.logger.info(f"  经度: {standardized_res['lon']:.4f}°")
        self.logger.info(f"  深度: {standardized_res['depth']:.2f} km")
        
        # 构建共同区域信息（确保所有数值都是Python原生类型，以便JSON序列化）
        valid_standardized_region = {
            'lat': (float(lat_min), float(lat_max)),
            'lon': (float(lon_min), float(lon_max)),
            'depth': (float(depth_min), float(depth_max)),
            'resolution': {
                'lat': float(standardized_res['lat']),
                'lon': float(standardized_res['lon']),
                'depth': float(standardized_res['depth'])
            },
            'grid_points': {
                'lat': int((lat_max - lat_min) / standardized_res['lat']) + 1,
                'lon': int((lon_max - lon_min) / standardized_res['lon']) + 1,
                'depth': int((depth_max - depth_min) / standardized_res['depth']) + 1
            },
            'models_included': list(all_coords.keys())
        }
        
        self.logger.info("\n" + "="*80)
        self.logger.info("✅ 共同覆盖区域:")
        self.logger.info("="*80)
        self.logger.info(f"  纬度: {lat_min:.2f}° ~ {lat_max:.2f}°")
        self.logger.info(f"  经度: {lon_min:.2f}° ~ {lon_max:.2f}°")
        self.logger.info(f"  深度: {depth_min:.1f} ~ {depth_max:.1f} km")
        self.logger.info(f"  包含模型: {', '.join(all_coords.keys())}")
        self.logger.info("="*80 + "\n")
        
        self.config.valid_standardized_region = valid_standardized_region
        
        region_file = self.config.paths['metadata_dir'] / 'valid_standardized_region.json'
        with open(region_file, 'w', encoding='utf-8') as f:
            json.dump(valid_standardized_region, f, indent=2, default=str)
        self.logger.info(f"共同有效区域信息已保存: {region_file}\n")
        
        return valid_standardized_region

    # ==================== 第一步：分析模型结构 ====================
    
    def analyze_model_structure(self, model_name: str) -> ModelMetadata:
        """分析模型结构并创建初步元数据（参数范围在数据处理后更新）"""
        self.logger.info(f"开始分析模型结构: {model_name}")
        
        model_info = self.config.models[model_name]
        input_file = self.config.paths['input_dir'] / model_info['file']
        model_format = model_info['format']
        
        if model_format == 'h5':
            metadata = self._analyze_h5_structure(input_file, model_name)
        elif model_format == 'nc':
            metadata = self._analyze_nc_structure(input_file, model_name)
        else:
            # 对于其他格式，创建通用元数据
            metadata = self._create_generic_metadata(model_name)
        
        # 添加有效覆盖率信息
        if self.config.valid_standardized_region is not None:
            coverage_value = self.config.valid_standardized_region.get('valid_coverage_percent')
            if isinstance(coverage_value, (int, float)):
                metadata.valid_coverage_percent = float(coverage_value)
            else:
                metadata.valid_coverage_percent = None
        
        # 注：metadata 将在 process_model 中数据处理后更新参数范围并保存
        
        self.logger.info(f"模型 {model_name} 结构分析完成")
        return metadata
    
    def _create_generic_metadata(self, model_name: str) -> ModelMetadata:
        """为非 H5/NC 格式创建通用元数据"""
        model_info = self.config.models[model_name]
        
        year = 2020  # 默认年份
        try:
            year = int(model_name.split('_')[0])
        except (ValueError, IndexError):
            pass
        
        metadata = ModelMetadata(
            name=model_name,
            full_name=model_info.get('reference', model_name),
            year=year,
            format=model_info['format'].upper(),
            reference=model_info.get('reference', 'Unknown'),
            region=model_info.get('region', 'East Asia'),
            original_parameters=model_info.get('params', []),
            coordinate_system='WGS84',
            depth_unit='km',
            velocity_unit='km/s',
            density_unit='kg/m3',
            depth_shift_applied=model_info.get('depth_shift', 0.0) != 0.0,
            notes=f"Format: {model_info['format']}"
        )
        
        return metadata
    
    def _analyze_h5_structure(self, file_path: Path, model_name: str) -> ModelMetadata:
        """分析HDF5文件结构"""
        self.logger.info(f"分析HDF5文件: {file_path.name}")
        
        model_info = self.config.models[model_name]
        
        with h5py.File(file_path, 'r') as f:
            lat_dataset: h5py.Dataset = f['model_coordinates/latitude']  # type: ignore
            lon_dataset: h5py.Dataset = f['model_coordinates/longitude']  # type: ignore
            lat = np.array(lat_dataset[:])
            lon = np.array(lon_dataset[:])
            
            if 'model_coordinates/depth_in_meter' in f:
                depth_dataset: h5py.Dataset = f['model_coordinates/depth_in_meter']  # type: ignore
                depth_raw = np.array(depth_dataset[:])
                depth_km = depth_raw / 1000.0
                depth_unit = 'km (converted from meters)'
            elif 'model_coordinates/depth' in f:
                depth_dataset: h5py.Dataset = f['model_coordinates/depth']  # type: ignore
                depth_raw = np.array(depth_dataset[:])
                depth_km = depth_raw / 1000.0 if depth_raw.max() > 5000 else depth_raw
                depth_unit = 'km'
            else:
                vp_dataset: h5py.Dataset = f['model_parameters/vp']  # type: ignore
                vp = np.array(vp_dataset[:])
                depth_km = np.linspace(0, 1000, vp.shape[2])
                depth_unit = 'km (inferred)'
            
            depth_shift = model_info.get('depth_shift', 0.0)
            depth_shift_applied = False
            if depth_shift != 0.0:
                depth_km = depth_km + depth_shift
                depth_shift_applied = True
                self.logger.info(f"  深度平移: {depth_shift:+.1f} km 已应用")
            
            model_params_group: h5py.Group = f['model_parameters']  # type: ignore
            original_params = [str(key) for key in model_params_group.keys()]
            first_param = original_params[0]
            first_param_dataset: h5py.Dataset = f[f'model_parameters/{first_param}']  # type: ignore
            first_param_shape = np.array(first_param_dataset[:]).shape
            data_shape: Tuple[int, int, int] = (int(first_param_shape[0]), int(first_param_shape[1]), int(first_param_shape[2]))
        
        resolution = {
            'latitude': float(np.mean(np.diff(np.sort(lat)))),
            'longitude': float(np.mean(np.diff(np.sort(lon)))),
            'depth': float(np.mean(np.diff(np.sort(depth_km))))
        }
        
        metadata = ModelMetadata(
            name=model_name,
            full_name=model_info.get('reference', model_name),
            year=int(model_name.split('_')[0]),
            format='HDF5',
            reference=model_info['reference'],
            region=model_info['region'],
            original_parameters=original_params,
            coordinate_system='WGS84',
            depth_unit=depth_unit,
            velocity_unit='km/s',
            density_unit='kg/m3',
            data_shape=data_shape,
            lat_range=(float(lat.min()), float(lat.max())),
            lon_range=(float(lon.min()), float(lon.max())),
            depth_range=(float(depth_km.min()), float(depth_km.max())),
            resolution=resolution,
            depth_shift_applied=depth_shift_applied,
            notes=f"Depth shift: {depth_shift:+.1f} km" if depth_shift != 0.0 else ""
        )
        
        return metadata
    
    def _analyze_nc_structure(self, file_path: Path, model_name: str) -> ModelMetadata:
        """分析NetCDF文件结构"""
        self.logger.info(f"分析NetCDF文件: {file_path.name}")
        
        model_info = self.config.models[model_name]
        
        ds = xr.open_dataset(file_path)
        
        lat = ds['latitude'].values
        lon = ds['longitude'].values
        depth = ds['depth'].values
        
        original_params = [str(key) for key in ds.data_vars.keys()]
        first_var = original_params[0]
        first_var_shape = ds[first_var].shape
        data_shape: Tuple[int, int, int] = (int(first_var_shape[0]), int(first_var_shape[1]), int(first_var_shape[2]))
        
        resolution = {
            'latitude': float(np.mean(np.diff(np.sort(lat)))),
            'longitude': float(np.mean(np.diff(np.sort(lon)))),
            'depth': float(np.mean(np.diff(np.sort(depth))))
        }
        
        ds.close()
        
        metadata = ModelMetadata(
            name=model_name,
            full_name=model_info.get('reference', model_name),
            year=int(model_name.split('_')[0]),
            format='NetCDF4',
            reference=model_info['reference'],
            region=model_info['region'],
            original_parameters=original_params,
            coordinate_system='WGS84',
            depth_unit='km',
            velocity_unit='km/s',
            density_unit='g/cm3',
            data_shape=data_shape,
            lat_range=(float(lat.min()), float(lat.max())),
            lon_range=(float(lon.min()), float(lon.max())),
            depth_range=(float(depth.min()), float(depth.max())),
            resolution=resolution,
            depth_shift_applied=False
        )
        
        return metadata

    # ==================== 第二步：读取模型数据 ====================
    
    def read_model(self, model_name: str) -> Tuple[Dict[str, np.ndarray], Dict]:
        """读取速度模型数据（返回原始区域数据）"""
        self.logger.info(f"开始读取模型: {model_name}")
        
        model_info = self.config.models[model_name]
        input_file = self.config.paths['input_dir'] / model_info['file']
        model_format = model_info['format']
        
        # 根据格式调用相应的读取器
        if model_format == 'h5':
            data_3d, coords = self._read_h5_model_3d(input_file, model_name)
        elif model_format == 'nc':
            data_3d, coords = self._read_nc_model_3d(input_file, model_name)
        elif model_format == 'geocsv':
            data_3d, coords = self._read_geocsv_model_3d(input_file, model_name)
        elif model_format == 'txt':
            data_3d, coords = self._read_txt_model_3d(input_file, model_name)
        elif model_format == 'txt_3d':
            data_3d, coords = self._read_txt_3d_model(input_file, model_name)
        elif model_format == 'txt_11col':
            data_3d, coords = self._read_txt_11col_model(input_file, model_name)
        elif model_format == 'xlsx':
            data_3d, coords = self._read_xlsx_model(input_file, model_name)
        elif model_format == 'multi_depth':
            data_3d, coords = self._read_multi_depth_model(input_file, model_name)
        elif model_format == 'multi_lon':
            data_3d, coords = self._read_multi_lon_model(input_file, model_name)
        elif model_format == 'multi_depth_mod':
            data_3d, coords = self._read_multi_depth_mod_model(input_file, model_name)
        else:
            raise ValueError(f"不支持的格式: {model_format}")
        
        self.logger.info(f"模型 {model_name} 读取完成（原始区域）")
        self.logger.info(f"  数据形状: {data_3d[list(data_3d.keys())[0]].shape}")
        
        return data_3d, coords
    
    def _read_h5_model_3d(self, file_path: Path, model_name: str) -> Tuple[Dict, Dict]:
        """读取HDF5格式模型（3D数组）"""
        self.logger.info(f"读取HDF5文件: {file_path.name} (3D数组模式)")
        
        model_info = self.config.models[model_name]
        
        with h5py.File(file_path, 'r') as f:
            lat_dataset: h5py.Dataset = f['model_coordinates/latitude']  # type: ignore
            lon_dataset: h5py.Dataset = f['model_coordinates/longitude']  # type: ignore
            lat = np.array(lat_dataset[:])
            lon = np.array(lon_dataset[:])
            
            if 'model_coordinates/depth_in_meter' in f:
                depth_dataset: h5py.Dataset = f['model_coordinates/depth_in_meter']  # type: ignore
                depth_raw = np.array(depth_dataset[:])
                depth_km = depth_raw / 1000.0
            elif 'model_coordinates/depth_in_km' in f:
                depth_dataset: h5py.Dataset = f['model_coordinates/depth_in_km']  # type: ignore
                depth_km = np.array(depth_dataset[:])
            elif 'model_coordinates/depth' in f:
                depth_dataset: h5py.Dataset = f['model_coordinates/depth']  # type: ignore
                depth_raw = np.array(depth_dataset[:])
                depth_km = depth_raw / 1000.0 if depth_raw.max() > 5000 else depth_raw
            else:
                vp_dataset: h5py.Dataset = f['model_parameters/vp']  # type: ignore
                vp_temp = np.array(vp_dataset[:])
                n_depth = vp_temp.shape[2]
                depth_km = np.linspace(0, 1000, n_depth)
            
            # 🔥 应用深度平移
            depth_shift = model_info.get('depth_shift', 0.0)
            if depth_shift != 0.0:
                depth_km_original = depth_km.copy()
                depth_km = depth_km + depth_shift
                self.logger.info(f"  🔥 深度平移: {depth_shift:+.1f} km")
                self.logger.info(f"     原始: {depth_km_original.min():.1f} ~ {depth_km_original.max():.1f} km")
                self.logger.info(f"     平移后: {depth_km.min():.1f} ~ {depth_km.max():.1f} km")
            
            vp_dataset: h5py.Dataset = f['model_parameters/vp']  # type: ignore
            vsv_dataset: h5py.Dataset = f['model_parameters/vsv']  # type: ignore
            vsh_dataset: h5py.Dataset = f['model_parameters/vsh']  # type: ignore
            rho_dataset: h5py.Dataset = f['model_parameters/rho']  # type: ignore
            vp = np.array(vp_dataset[:]) / 1000.0
            vsv = np.array(vsv_dataset[:]) / 1000.0
            vsh = np.array(vsh_dataset[:]) / 1000.0
            rho = np.array(rho_dataset[:])
            
            self.logger.info(f"  原始数据形状: {vp.shape}")
            self.logger.info(f"  坐标范围:")
            self.logger.info(f"    经度: {lon.min():.2f}° - {lon.max():.2f}° ({len(lon)} 点)")
            self.logger.info(f"    纬度: {lat.min():.2f}° - {lat.max():.2f}° ({len(lat)} 点)")
            self.logger.info(f"    深度: {depth_km.min():.1f} - {depth_km.max():.1f} km ({len(depth_km)} 点)")
        
        data_3d = {
            'vp': vp,
            'vsv': vsv,
            'vsh': vsh,
            'rho': rho
        }
        
        coords = {
            'longitude': lon,
            'latitude': lat,
            'depth': depth_km
        }
        
        return data_3d, coords
    
    def _read_nc_model_3d(self, file_path: Path, model_name: str) -> Tuple[Dict, Dict]:
        """读取NetCDF格式模型（3D数组）"""
        self.logger.info(f"读取NetCDF文件: {file_path.name} (3D数组模式)")
        
        ds = xr.open_dataset(file_path)
        
        lon = ds['longitude'].values
        lat = ds['latitude'].values
        depth = ds['depth'].values
        
        # 🔥 深度网格验证
        self.logger.info(f"  深度网格验证:")
        self.logger.info(f"    深度范围: {depth.min():.1f} ~ {depth.max():.1f} km")
        self.logger.info(f"    前5个深度: {depth[:5]}")
        if 0.0 not in depth:
            self.logger.warning(f"    ⚠️ 深度网格不包含 0 km")
        
        data_3d = {}
        for var in ds.data_vars:
            data = ds[var].values
            data[data == 9999.0] = np.nan
            
            if len(data.shape) == 3:
                data = np.transpose(data, (1, 2, 0))
                self.logger.info(f"    {var}: 维度转换 (depth, lat, lon) -> (lat, lon, depth)")
            
            data_3d[var] = data
        
        coords = {
            'longitude': lon,
            'latitude': lat,
            'depth': depth
        }
        
        ds.close()
        
        self.logger.info(f"  读取到 {len(data_3d)} 个参数: {list(data_3d.keys())}")
        self.logger.info(f"  数据形状（统一后）: {data_3d[list(data_3d.keys())[0]].shape} (lat, lon, depth)")
        
        return data_3d, coords

    # ==================== 新增：GeoCSV 格式读取 ====================
    
    def _read_geocsv_model_3d(self, file_path: Path, model_name: str) -> Tuple[Dict, Dict]:
        """读取 GeoCSV 格式模型（| 分隔符）"""
        self.logger.info(f"读取 GeoCSV 文件: {file_path.name}")
        
        model_info = self.config.models[model_name]
        
        # 读取 CSV 文件
        df = pd.read_csv(file_path, sep='|', comment='#', na_values=['nan', 'NaN', '9999.0'])
        
        # 提取坐标
        lon = np.sort(df['longitude'].unique())
        lat = np.sort(df['latitude'].unique())
        depth = np.sort(df['depth'].unique())
        
        self.logger.info(f"  坐标范围:")
        self.logger.info(f"    经度: {lon.min():.2f}° - {lon.max():.2f}° ({len(lon)} 点)")
        self.logger.info(f"    纬度: {lat.min():.2f}° - {lat.max():.2f}° ({len(lat)} 点)")
        self.logger.info(f"    深度: {depth.min():.1f} - {depth.max():.1f} km ({len(depth)} 点)")
        
        # 创建 3D 数组
        data_3d = {}
        shape = (len(lat), len(lon), len(depth))
        
        # 创建坐标索引映射
        lat_idx = {v: i for i, v in enumerate(lat)}
        lon_idx = {v: i for i, v in enumerate(lon)}
        depth_idx = {v: i for i, v in enumerate(depth)}
        
        # 提取可用参数
        available_params = model_info.get('params', [])
        for param in available_params:
            col_name = param
            if param == 'rho':
                col_name = 'rho'
            if col_name in df.columns:
                data_3d[param] = np.full(shape, np.nan, dtype=np.float32)
        
        # 填充数据
        self.logger.info(f"  填充 3D 数组...")
        for _, row in df.iterrows():
            try:
                i = lat_idx[row['latitude']]
                j = lon_idx[row['longitude']]
                k = depth_idx[row['depth']]
                
                for param in available_params:
                    if param in data_3d and param in df.columns:
                        value = row.get(param, np.nan)
                        if pd.notna(value):
                            data_3d[param][i, j, k] = value
            except (KeyError, ValueError):
                continue
        
        # 密度单位转换 (g/cm3 -> kg/m3)
        if 'rho' in data_3d and model_info.get('rho_unit') == 'g/cm3':
            self.logger.info(f"  密度单位转换: g/cm3 -> kg/m3")
            data_3d['rho'] = data_3d['rho'] * 1000
        
        coords = {
            'longitude': lon,
            'latitude': lat,
            'depth': depth
        }
        
        self.logger.info(f"  读取到 {len(data_3d)} 个参数: {list(data_3d.keys())}")
        return data_3d, coords

    # ==================== 新增：简单文本格式读取 ====================
    
    def _read_txt_model_3d(self, file_path: Path, model_name: str) -> Tuple[Dict, Dict]:
        """读取简单文本格式模型（空格分隔）"""
        self.logger.info(f"读取文本文件: {file_path.name}")
        
        model_info = self.config.models[model_name]
        columns = model_info.get('columns', ['longitude', 'latitude', 'depth', 'vp', 'vs'])
        separator = model_info.get('separator', r'\s+')
        skiprows = model_info.get('skiprows', 0)
        
        # 读取文件
        df = pd.read_csv(
            file_path, 
            sep=separator, 
            comment='#', 
            header=None,
            names=columns,
            skiprows=skiprows,
            na_values=['nan', 'NaN', '9999.0']
        )
        
        # 确保列名标准化
        if 'longitude' not in df.columns:
            df.rename(columns={'lon': 'longitude'}, inplace=True)
        if 'latitude' not in df.columns:
            df.rename(columns={'lat': 'latitude'}, inplace=True)

        # depth_negate: 部分模型深度定义为负向下（如 0~-62.5 km），取反转为正常约定
        if model_info.get('depth_negate', False):
            df['depth'] = -df['depth']

        # 提取坐标
        lon = np.sort(df['longitude'].unique())
        lat = np.sort(df['latitude'].unique())
        depth = np.sort(df['depth'].unique())
        
        self.logger.info(f"  坐标范围:")
        self.logger.info(f"    经度: {lon.min():.2f}° - {lon.max():.2f}° ({len(lon)} 点)")
        self.logger.info(f"    纬度: {lat.min():.2f}° - {lat.max():.2f}° ({len(lat)} 点)")
        self.logger.info(f"    深度: {depth.min():.1f} - {depth.max():.1f} km ({len(depth)} 点)")
        
        # 创建 3D 数组
        data_3d = {}
        shape = (len(lat), len(lon), len(depth))
        
        # 创建坐标索引映射
        lat_idx = {v: i for i, v in enumerate(lat)}
        lon_idx = {v: i for i, v in enumerate(lon)}
        depth_idx = {v: i for i, v in enumerate(depth)}
        
        # 提取可用参数
        available_params = model_info.get('params', [])
        for param in available_params:
            if param in df.columns:
                data_3d[param] = np.full(shape, np.nan, dtype=np.float32)
        
        # 填充数据
        self.logger.info(f"  填充 3D 数组...")
        for _, row in df.iterrows():
            try:
                i = lat_idx[row['latitude']]
                j = lon_idx[row['longitude']]
                k = depth_idx[row['depth']]
                
                for param in available_params:
                    if param in data_3d and param in df.columns:
                        value = row.get(param, np.nan)
                        if pd.notna(value):
                            data_3d[param][i, j, k] = value
            except (KeyError, ValueError):
                continue
        
        coords = {
            'longitude': lon,
            'latitude': lat,
            'depth': depth
        }
        
        self.logger.info(f"  读取到 {len(data_3d)} 个参数: {list(data_3d.keys())}")
        return data_3d, coords

    # ==================== 新增：3D 文本格式读取 (CSES_VM1.0) ====================
    
    def _read_txt_3d_model(self, file_path: Path, model_name: str) -> Tuple[Dict, Dict]:
        """读取 3D 格式文本文件 (如 CSES_VM1.0)"""
        self.logger.info(f"读取 3D 格式文本文件: {file_path.name}")
        
        with open(file_path, 'r') as f:
            # 第一行：nlat, nlon, ndep
            nlat, nlon, ndep = map(int, f.readline().split())
            self.logger.info(f"  网格维度: lat={nlat}, lon={nlon}, depth={ndep}")
            
            def read_coords(n_points):
                """读取坐标数组（每行最多10个值）"""
                lines = (n_points + 9) // 10
                values = []
                for _ in range(lines):
                    values.extend(f.readline().strip().split())
                return np.array(values[:n_points], dtype=float)
            
            alat = read_coords(nlat)
            alon = read_coords(nlon)
            adep = read_coords(ndep)
            
            def read_3d_array(nlat, nlon, ndep):
                """读取 3D 数据数组"""
                total_values = nlat * nlon * ndep
                data = []
                while len(data) < total_values:
                    line = f.readline().strip()
                    data.extend([np.nan if x == 'NaN' else float(x) for x in line.split()])
                return np.array(data[:total_values]).reshape(nlat, nlon, ndep)
            
            vp_data = read_3d_array(nlat, nlon, ndep)
            vs_data = read_3d_array(nlat, nlon, ndep)
        
        # 转换为 (lat, lon, depth) 格式
        # 原始数据格式为 (lat, lon, depth)，与我们的标准一致
        data_3d = {
            'vp': vp_data.astype(np.float32),
            'vs': vs_data.astype(np.float32)
        }
        
        coords = {
            'longitude': np.round(alon, 3),
            'latitude': np.round(alat, 3),
            'depth': np.round(adep, 3)
        }
        
        self.logger.info(f"  坐标范围:")
        self.logger.info(f"    经度: {coords['longitude'].min():.2f}° - {coords['longitude'].max():.2f}°")
        self.logger.info(f"    纬度: {coords['latitude'].min():.2f}° - {coords['latitude'].max():.2f}°")
        self.logger.info(f"    深度: {coords['depth'].min():.1f} - {coords['depth'].max():.1f} km")
        
        return data_3d, coords

    # ==================== 新增：11 列文本格式读取 (ASIA2024) ====================
    
    def _read_txt_11col_model(self, file_path: Path, model_name: str) -> Tuple[Dict, Dict]:
        """读取 11 列格式文本文件 (如 ASIA2024)，速度单位为 m/s"""
        self.logger.info(f"读取 11 列格式文本文件: {file_path.name}")
        
        model_info = self.config.models[model_name]
        
        # 读取数据（11列：lon lat depth Vp_abs Vp_ref dVp Vs_abs Vs_ref dVs col10 col11）
        df = pd.read_csv(
            file_path, 
            sep=r'\s+', 
            header=None,
            names=['lon', 'lat', 'depth', 'Vp_abs', 'Vp_ref', 'dVp', 
                   'Vs_abs', 'Vs_ref', 'dVs', 'col10', 'col11'],
            na_values=['nan', 'NaN'],
            dtype=float
        )
        
        # 提取坐标
        lon = np.sort(df['lon'].unique())
        lat = np.sort(df['lat'].unique())
        depth = np.sort(df['depth'].unique())
        
        self.logger.info(f"  坐标范围:")
        self.logger.info(f"    经度: {lon.min():.2f}° - {lon.max():.2f}° ({len(lon)} 点)")
        self.logger.info(f"    纬度: {lat.min():.2f}° - {lat.max():.2f}° ({len(lat)} 点)")
        self.logger.info(f"    深度: {depth.min():.1f} - {depth.max():.1f} km ({len(depth)} 点)")
        
        # 速度单位转换: m/s -> km/s
        velocity_unit = model_info.get('velocity_unit', 'km/s')
        if velocity_unit == 'm/s':
            self.logger.info(f"  速度单位转换: m/s -> km/s")
            df['vp'] = df['Vp_abs'] / 1000.0
            df['vs'] = df['Vs_abs'] / 1000.0
        else:
            df['vp'] = df['Vp_abs']
            df['vs'] = df['Vs_abs']
        
        # 创建 3D 数组
        shape = (len(lat), len(lon), len(depth))
        data_3d = {
            'vp': np.full(shape, np.nan, dtype=np.float32),
            'vs': np.full(shape, np.nan, dtype=np.float32)
        }
        
        # 创建坐标索引映射
        lat_idx = {v: i for i, v in enumerate(lat)}
        lon_idx = {v: i for i, v in enumerate(lon)}
        depth_idx = {v: i for i, v in enumerate(depth)}
        
        # 填充数据
        self.logger.info(f"  填充 3D 数组...")
        for _, row in df.iterrows():
            try:
                i = lat_idx[row['lat']]
                j = lon_idx[row['lon']]
                k = depth_idx[row['depth']]
                
                if pd.notna(row['vp']):
                    data_3d['vp'][i, j, k] = row['vp']
                if pd.notna(row['vs']):
                    data_3d['vs'][i, j, k] = row['vs']
            except (KeyError, ValueError):
                continue
        
        coords = {
            'longitude': lon,
            'latitude': lat,
            'depth': depth
        }
        
        return data_3d, coords

    # ==================== 新增：Excel 格式读取 ====================
    
    def _read_xlsx_model(self, file_path: Path, model_name: str) -> Tuple[Dict, Dict]:
        """读取 Excel 格式模型"""
        self.logger.info(f"读取 Excel 文件: {file_path.name}")
        
        model_info = self.config.models[model_name]
        skiprows = model_info.get('skiprows', 0)
        usecols = model_info.get('usecols', None)
        column_names = model_info.get('column_names', ['latitude', 'longitude', 'depth', 'vs'])
        
        # 读取 Excel 文件
        df = pd.read_excel(
            file_path,
            skiprows=skiprows,
            usecols=usecols,
            names=column_names
        )
        
        # 提取坐标
        lon = np.sort(df['longitude'].unique())
        lat = np.sort(df['latitude'].unique())
        depth = np.sort(df['depth'].unique())
        
        self.logger.info(f"  坐标范围:")
        self.logger.info(f"    经度: {lon.min():.2f}° - {lon.max():.2f}° ({len(lon)} 点)")
        self.logger.info(f"    纬度: {lat.min():.2f}° - {lat.max():.2f}° ({len(lat)} 点)")
        self.logger.info(f"    深度: {depth.min():.1f} - {depth.max():.1f} km ({len(depth)} 点)")
        
        # 创建 3D 数组
        shape = (len(lat), len(lon), len(depth))
        available_params = model_info.get('params', ['vs'])
        data_3d = {}
        for param in available_params:
            if param in df.columns:
                data_3d[param] = np.full(shape, np.nan, dtype=np.float32)
        
        # 创建坐标索引映射
        lat_idx = {v: i for i, v in enumerate(lat)}
        lon_idx = {v: i for i, v in enumerate(lon)}
        depth_idx = {v: i for i, v in enumerate(depth)}
        
        # 填充数据
        for _, row in df.iterrows():
            try:
                i = lat_idx[row['latitude']]
                j = lon_idx[row['longitude']]
                k = depth_idx[row['depth']]
                
                for param in available_params:
                    if param in data_3d and param in df.columns:
                        value = row.get(param, np.nan)
                        if pd.notna(value):
                            data_3d[param][i, j, k] = value
            except (KeyError, ValueError):
                continue
        
        coords = {
            'longitude': lon,
            'latitude': lat,
            'depth': depth
        }
        
        return data_3d, coords

    # ==================== 新增：多深度文件格式读取 ====================
    
    def _read_multi_depth_model(self, dir_path: Path, model_name: str) -> Tuple[Dict, Dict]:
        """读取多深度文件格式模型"""
        self.logger.info(f"读取多深度文件目录: {dir_path}")
        
        model_info = self.config.models[model_name]
        file_pattern = model_info.get('file_pattern', '{depth}km.txt')
        depth_range = model_info.get('depth_range', list(range(10, 251, 10)))
        columns = model_info.get('columns', ['longitude', 'latitude', 'vs'])
        separator = model_info.get('separator', r'\s+')
        
        all_data = []
        
        for depth in depth_range:
            # 生成文件名
            if '{depth:03d}' in file_pattern:
                filename = file_pattern.format(depth=depth)
            elif '{depth}' in file_pattern:
                filename = file_pattern.format(depth=depth)
            else:
                filename = file_pattern.replace('{depth}', str(depth))
            
            file_path = dir_path / filename
            
            if not file_path.exists():
                self.logger.warning(f"  文件不存在: {filename}")
                continue
            
            df = pd.read_csv(file_path, sep=separator, header=None, names=columns)
            df['depth'] = depth
            all_data.append(df)
        
        if not all_data:
            raise ValueError(f"未找到任何有效的深度文件: {dir_path}")
        
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # 提取坐标
        lon = np.sort(merged_df['longitude'].unique())
        lat = np.sort(merged_df['latitude'].unique())
        depth = np.sort(merged_df['depth'].unique())
        
        self.logger.info(f"  坐标范围:")
        self.logger.info(f"    经度: {lon.min():.2f}° - {lon.max():.2f}° ({len(lon)} 点)")
        self.logger.info(f"    纬度: {lat.min():.2f}° - {lat.max():.2f}° ({len(lat)} 点)")
        self.logger.info(f"    深度: {depth.min():.1f} - {depth.max():.1f} km ({len(depth)} 点)")
        
        # 创建 3D 数组
        shape = (len(lat), len(lon), len(depth))
        available_params = model_info.get('params', ['vs'])
        data_3d = {}
        for param in available_params:
            if param in merged_df.columns:
                data_3d[param] = np.full(shape, np.nan, dtype=np.float32)
        
        # 创建坐标索引映射
        lat_idx = {v: i for i, v in enumerate(lat)}
        lon_idx = {v: i for i, v in enumerate(lon)}
        depth_idx = {v: i for i, v in enumerate(depth)}
        
        # 填充数据
        for _, row in merged_df.iterrows():
            try:
                i = lat_idx[row['latitude']]
                j = lon_idx[row['longitude']]
                k = depth_idx[row['depth']]
                
                for param in available_params:
                    if param in data_3d and param in merged_df.columns:
                        value = row.get(param, np.nan)
                        if pd.notna(value):
                            data_3d[param][i, j, k] = value
            except (KeyError, ValueError):
                continue
        
        coords = {
            'longitude': lon,
            'latitude': lat,
            'depth': depth
        }
        
        return data_3d, coords

    # ==================== 新增：多经度文件格式读取 (Kumar_Pamir) ====================
    
    def _read_multi_lon_model(self, dir_path: Path, model_name: str) -> Tuple[Dict, Dict]:
        """读取多经度文件格式模型 (按经度分文件，支持通配符扫描)"""
        self.logger.info(f"读取多经度文件目录: {dir_path}")
        
        model_info = self.config.models[model_name]
        file_pattern = model_info.get('file_pattern', '*.rev.xyz')
        columns = model_info.get('columns', ['longitude', 'latitude', 'depth', 'vs'])
        separator = model_info.get('separator', r'\s+')
        
        all_data = []
        
        # 使用通配符扫描目录
        if '*' in file_pattern:
            import glob
            file_list = list(dir_path.glob(file_pattern))
            self.logger.info(f"  通过通配符找到 {len(file_list)} 个文件")
            
            for file_path in sorted(file_list):
                try:
                    df = pd.read_csv(file_path, sep=separator, header=None, names=columns)
                    all_data.append(df)
                except Exception as e:
                    self.logger.warning(f"  读取文件 {file_path.name} 失败: {e}")
                    continue
        else:
            # 使用预定义的经度范围
            lon_range = model_info.get('lon_range', np.arange(68.5, 85.01, 0.25))
            for lon in lon_range:
                lon_str = f"{lon:.2f}".rstrip('0').rstrip('.')
                filename = file_pattern.format(lon=lon_str)
                file_path = dir_path / filename
                
                if not file_path.exists():
                    continue
                
                df = pd.read_csv(file_path, sep=separator, header=None, names=columns)
                if 'longitude' not in df.columns:
                    df['longitude'] = np.round(lon, 3)
                all_data.append(df)
        
        if not all_data:
            raise ValueError(f"未找到任何有效的数据文件: {dir_path}")
        
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # 提取坐标
        lon = np.sort(merged_df['longitude'].unique())
        lat = np.sort(merged_df['latitude'].unique())
        depth = np.sort(merged_df['depth'].unique())
        
        self.logger.info(f"  坐标范围:")
        self.logger.info(f"    经度: {lon.min():.2f}° - {lon.max():.2f}° ({len(lon)} 点)")
        self.logger.info(f"    纬度: {lat.min():.2f}° - {lat.max():.2f}° ({len(lat)} 点)")
        self.logger.info(f"    深度: {depth.min():.1f} - {depth.max():.1f} km ({len(depth)} 点)")
        
        # 创建 3D 数组
        shape = (len(lat), len(lon), len(depth))
        available_params = model_info.get('params', ['vs'])
        data_3d = {}
        for param in available_params:
            if param in merged_df.columns:
                data_3d[param] = np.full(shape, np.nan, dtype=np.float32)
        
        # 创建坐标索引映射
        lat_idx = {v: i for i, v in enumerate(lat)}
        lon_idx = {v: i for i, v in enumerate(lon)}
        depth_idx = {v: i for i, v in enumerate(depth)}
        
        # 填充数据
        for _, row in merged_df.iterrows():
            try:
                i = lat_idx[row['latitude']]
                j = lon_idx[row['longitude']]
                k = depth_idx[row['depth']]
                
                for param in available_params:
                    if param in data_3d and param in merged_df.columns:
                        value = row.get(param, np.nan)
                        if pd.notna(value):
                            data_3d[param][i, j, k] = value
            except (KeyError, ValueError):
                continue
        
        coords = {
            'longitude': lon,
            'latitude': lat,
            'depth': depth
        }
        
        return data_3d, coords

    # ==================== 新增：CSRM1.0 多深度 mod 文件格式读取 ====================
    
    def _read_multi_depth_mod_model(self, dir_path: Path, model_name: str) -> Tuple[Dict, Dict]:
        """读取 CSRM1.0 多深度 mod 文件格式"""
        self.logger.info(f"读取 CSRM1.0 多深度 mod 文件目录: {dir_path}")
        
        import glob
        
        model_info = self.config.models[model_name]
        
        # 查找所有 mod 文件
        mod_files = list(dir_path.glob("CSRM1.0_*_km.mod"))
        
        if not mod_files:
            raise ValueError(f"未找到任何 mod 文件: {dir_path}")
        
        # 解析深度并排序
        file_depth_pairs = []
        for f in mod_files:
            try:
                base_name = f.name
                if not base_name.startswith("CSRM1.0_") or not base_name.endswith("_km.mod"):
                    continue
                depth_str = base_name.split("_")[1]
                depth = float(depth_str)
                file_depth_pairs.append((depth, f))
            except Exception:
                continue
        
        file_depth_pairs.sort(key=lambda x: x[0])
        self.logger.info(f"  找到 {len(file_depth_pairs)} 个 mod 文件")
        
        all_data = []
        
        for depth_km, mod_file in file_depth_pairs:
            try:
                with open(mod_file, 'r') as f:
                    for line in f:
                        if line.strip().startswith('#'):
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            try:
                                lon = float(parts[0])
                                lat = float(parts[1])
                                vs = float(parts[2])
                                vp = float(parts[4])
                                all_data.append({
                                    'longitude': lon,
                                    'latitude': lat,
                                    'depth': depth_km,
                                    'vp': vp,
                                    'vs': vs
                                })
                            except ValueError:
                                continue
            except Exception as e:
                self.logger.warning(f"  处理文件 {mod_file.name} 出错: {e}")
                continue
        
        if not all_data:
            raise ValueError(f"未读取到任何有效数据: {dir_path}")
        
        df = pd.DataFrame(all_data)
        
        # 提取坐标
        lon = np.sort(df['longitude'].unique())
        lat = np.sort(df['latitude'].unique())
        depth = np.sort(df['depth'].unique())
        
        self.logger.info(f"  坐标范围:")
        self.logger.info(f"    经度: {lon.min():.2f}° - {lon.max():.2f}° ({len(lon)} 点)")
        self.logger.info(f"    纬度: {lat.min():.2f}° - {lat.max():.2f}° ({len(lat)} 点)")
        self.logger.info(f"    深度: {depth.min():.1f} - {depth.max():.1f} km ({len(depth)} 点)")
        
        # 创建 3D 数组
        shape = (len(lat), len(lon), len(depth))
        data_3d = {
            'vp': np.full(shape, np.nan, dtype=np.float32),
            'vs': np.full(shape, np.nan, dtype=np.float32)
        }
        
        # 创建坐标索引映射
        lat_idx = {v: i for i, v in enumerate(lat)}
        lon_idx = {v: i for i, v in enumerate(lon)}
        depth_idx = {v: i for i, v in enumerate(depth)}
        
        # 填充数据
        for _, row in df.iterrows():
            try:
                i = lat_idx[row['latitude']]
                j = lon_idx[row['longitude']]
                k = depth_idx[row['depth']]
                
                if pd.notna(row['vp']):
                    data_3d['vp'][i, j, k] = row['vp']
                if pd.notna(row['vs']):
                    data_3d['vs'][i, j, k] = row['vs']
            except (KeyError, ValueError):
                continue
        
        coords = {
            'longitude': lon,
            'latitude': lat,
            'depth': depth
        }
        
        return data_3d, coords
    
    def _crop_to_valid_region(self, data_3d: Dict[str, np.ndarray],
                              coords: Dict[str, np.ndarray],
                              model_name: str) -> Tuple[Dict, Dict]:
        """🔥 裁剪到100%有效覆盖的共同区域"""
        self.logger.info(f"  🔥 裁剪到100%有效覆盖区域: {model_name}")
        
        valid_region = self.config.valid_standardized_region
        if valid_region is None:
            raise ValueError("valid_standardized_region 未设置，请先调用 compute_valid_standardized_region()")
        
        lon = coords['longitude']
        lat = coords['latitude']
        depth = coords['depth']
        
        # 🔥 使用精确的浮点数比较（允许1e-6误差）
        lon_mask = (lon >= valid_region['lon'][0] - 1e-6) & (lon <= valid_region['lon'][1] + 1e-6)
        lat_mask = (lat >= valid_region['lat'][0] - 1e-6) & (lat <= valid_region['lat'][1] + 1e-6)
        depth_mask = (depth >= valid_region['depth'][0] - 1e-6) & (depth <= valid_region['depth'][1] + 1e-6)
        
        self.logger.info(f"    裁剪掩码:")
        self.logger.info(f"      经度: {np.sum(lon_mask)}/{len(lon)} 点 保留")
        self.logger.info(f"      纬度: {np.sum(lat_mask)}/{len(lat)} 点 保留")
        self.logger.info(f"      深度: {np.sum(depth_mask)}/{len(depth)} 点 保留")
        
        # 🔥 关键：裁剪坐标数组
        cropped_coords = {
            'longitude': lon[lon_mask],
            'latitude': lat[lat_mask],
            'depth': depth[depth_mask]
        }
        
        # 🔥 关键：裁剪数据数组（使用 np.ix_ 确保正确索引）
        cropped_data = {}
        for param, data in data_3d.items():
            if len(data.shape) != 3:
                cropped_data[param] = data
                continue
            
            # 数据格式是 (lat, lon, depth)
            cropped_data[param] = data[np.ix_(lat_mask, lon_mask, depth_mask)]
        
        original_shape = data_3d[list(data_3d.keys())[0]].shape
        cropped_shape = cropped_data[list(cropped_data.keys())[0]].shape
        
        self.logger.info(f"    原始形状: {original_shape}")
        self.logger.info(f"    裁剪后形状: {cropped_shape}")
        self.logger.info(f"    保留数据: {np.prod(cropped_shape)/np.prod(original_shape)*100:.1f}%")
        
        # 🔥 验证裁剪后的坐标范围
        self.logger.info(f"    裁剪后坐标范围:")
        self.logger.info(f"      经度: {cropped_coords['longitude'].min():.2f}° ~ {cropped_coords['longitude'].max():.2f}°")
        self.logger.info(f"      纬度: {cropped_coords['latitude'].min():.2f}° ~ {cropped_coords['latitude'].max():.2f}°")
        self.logger.info(f"      深度: {cropped_coords['depth'].min():.1f} ~ {cropped_coords['depth'].max():.1f} km")
        
        # 🔥 验证无NaN（核心验证）
        first_param = list(cropped_data.keys())[0]
        nan_count = np.sum(np.isnan(cropped_data[first_param]))
        if nan_count > 0:
            self.logger.warning(f"    ⚠️ 裁剪后仍有 {nan_count:,} 个NaN值")
            # 可选：移除仍包含NaN的点
            if self.config.processing.get('require_full_coverage', True):
                self.logger.info(f"    🔄 移除包含NaN的数据点...")
                valid_mask = ~np.isnan(cropped_data[first_param])
                for param in cropped_data:
                    if len(cropped_data[param].shape) == 3:
                        cropped_data[param] = np.where(valid_mask, cropped_data[param], np.nan)
        else:
            self.logger.info(f"    ✅ 裁剪后数据100%有效，无NaN")
        
        return cropped_data, cropped_coords

    # ==================== 第三步：标准化处理 ====================
    
    def standardize_model_3d(self, data_3d: Dict[str, np.ndarray], 
                            coords: Dict[str, np.ndarray],
                            model_name: str) -> Tuple[Dict[str, np.ndarray], Dict]:
        """标准化模型数据（3D数组版）- 支持所有模型格式"""
        self.logger.info(f"开始标准化模型数据 (3D模式): {model_name}")
        
        shape = data_3d[list(data_3d.keys())[0]].shape
        std_data = {}
        model_info = self.config.models.get(model_name, {})
        
        # ========== 处理各向异性参数 (vpv, vph, vsv, vsh) ==========
        
        # 如果有完整的各向异性参数
        if 'vpv' in data_3d and 'vph' in data_3d:
            std_data['vpv'] = data_3d['vpv']
            std_data['vph'] = data_3d['vph']
        elif 'vp' in data_3d:
            # 只有各向同性 vp，假设各向异性参数相等
            std_data['vpv'] = data_3d['vp'].copy()
            std_data['vph'] = data_3d['vp'].copy()
        else:
            std_data['vpv'] = np.full(shape, np.nan, dtype=np.float32)
            std_data['vph'] = np.full(shape, np.nan, dtype=np.float32)
        
        if 'vsv' in data_3d and 'vsh' in data_3d:
            std_data['vsv'] = data_3d['vsv']
            std_data['vsh'] = data_3d['vsh']
        elif 'vs' in data_3d:
            # 只有各向同性 vs，假设各向异性参数相等
            std_data['vsv'] = data_3d['vs'].copy()
            std_data['vsh'] = data_3d['vs'].copy()
        else:
            std_data['vsv'] = np.full(shape, np.nan, dtype=np.float32)
            std_data['vsh'] = np.full(shape, np.nan, dtype=np.float32)
        
        # ========== 处理密度 ==========
        if 'rho' in data_3d:
            std_data['rho'] = data_3d['rho'].copy()
        else:
            std_data['rho'] = np.full(shape, np.nan, dtype=np.float32)
        
        # ========== 处理 eta 各向异性参数 ==========
        if 'eta' in data_3d:
            std_data['eta'] = data_3d['eta'].copy()
        
        # ========== 处理 Qmu 衰减参数 ==========
        if 'qmu' in data_3d:
            std_data['qmu'] = data_3d['qmu'].copy()
        
        # ========== 处理参考速度 ==========
        if 'vs_ref' in data_3d:
            std_data['vs0'] = data_3d['vs_ref']
        elif 'vs0' in data_3d:
            std_data['vs0'] = data_3d['vs0']
        else:
            std_data['vs0'] = np.full(shape, np.nan, dtype=np.float32)
        
        if 'vp_ref' in data_3d:
            std_data['vp0'] = data_3d['vp_ref']
        elif 'vp0' in data_3d:
            std_data['vp0'] = data_3d['vp0']
        else:
            std_data['vp0'] = np.full(shape, np.nan, dtype=np.float32)
        
        # ========== 计算等效速度 ==========
        vsv = std_data['vsv']
        vsh = std_data['vsh']
        vpv = std_data['vpv']
        vph = std_data['vph']
        
        # 计算 Voigt 平均速度
        # 如果有各向异性参数，使用 Voigt 公式；否则使用原始值
        with np.errstate(invalid='ignore'):
            if not np.isnan(vsv).all() and not np.isnan(vsh).all():
                std_data['vs'] = np.sqrt((2 * vsv**2 + vsh**2) / 3)
            elif 'vs' in data_3d:
                std_data['vs'] = data_3d['vs'].copy()
            else:
                std_data['vs'] = np.full(shape, np.nan, dtype=np.float32)
            
            if not np.isnan(vpv).all() and not np.isnan(vph).all():
                std_data['vp'] = np.sqrt((vpv**2 + 4 * vph**2) / 5)
            elif 'vp' in data_3d:
                std_data['vp'] = data_3d['vp'].copy()
            else:
                std_data['vp'] = np.full(shape, np.nan, dtype=np.float32)
        
        # ========== 密度单位检查 ==========
        # 不再自动转换单位，保留原始值。
        # NC 模型 rho 通常为 g/cm³（1~6），CSV 模型在 load 阶段已根据 rho_unit 配置转换。
        # convert_nc_to_gll.py 会根据实际单位做最终转换。
        rho = std_data['rho']
        if not np.isnan(rho).all():
            rho_mean = np.nanmean(rho)
            if rho_mean < 10:
                self.logger.info(f"  密度单位: g/cm³ (mean={rho_mean:.2f})")
            elif rho_mean < 10000:
                self.logger.info(f"  密度单位: kg/m³ (mean={rho_mean:.1f})")
            else:
                self.logger.warning(f"  密度值异常 (mean={rho_mean:.1f})，请检查")
        
        # ========== 质量控制 ==========
        std_data = self._quality_control_3d(std_data, coords, model_name)
        
        self.logger.info(f"模型 {model_name} 标准化完成")
        self.logger.info(f"  参数列表: {list(std_data.keys())}")
        
        # 统计有效数据
        total_points = np.prod(shape)
        for param, data in std_data.items():
            valid_count = np.sum(~np.isnan(data))
            valid_pct = (valid_count / total_points) * 100
            if valid_pct > 0:
                self.logger.info(f"    {param}: {valid_pct:.2f}% valid ({valid_count:,} 点)")
            else:
                self.logger.info(f"    {param}: 无数据")
        
        return std_data, coords
    
    def _quality_control_3d(self, data_3d: Dict[str, np.ndarray],
                           coords: Dict[str, np.ndarray],
                           model_name: str) -> Dict[str, np.ndarray]:
        """数据质量控制（3D数组版）- 仅检查坐标范围，保留原始速度参数值"""
        self.logger.info(f"执行数据质量控制 (3D模式): {model_name}")
        
        qc = self.config.qc
        
        # 仅检查坐标范围（不修改速度参数值）
        lat = coords.get('latitude', np.array([]))
        lon = coords.get('longitude', np.array([]))
        depth = coords.get('depth', np.array([]))
        
        # 纬度范围检查
        if len(lat) > 0:
            lat_min, lat_max = lat.min(), lat.max()
            if lat_min < qc['lat_range'][0] or lat_max > qc['lat_range'][1]:
                self.logger.warning(f"  纬度范围 [{lat_min:.2f}, {lat_max:.2f}] 超出有效范围 {qc['lat_range']}")
        
        # 经度范围检查
        if len(lon) > 0:
            lon_min, lon_max = lon.min(), lon.max()
            if lon_min < qc['lon_range'][0] or lon_max > qc['lon_range'][1]:
                self.logger.warning(f"  经度范围 [{lon_min:.2f}, {lon_max:.2f}] 超出有效范围 {qc['lon_range']}")
        
        # 深度范围检查
        if len(depth) > 0:
            depth_min, depth_max = depth.min(), depth.max()
            if depth_min < qc['depth_range'][0] or depth_max > qc['depth_range'][1]:
                self.logger.warning(f"  深度范围 [{depth_min:.1f}, {depth_max:.1f}] km 超出有效范围 {qc['depth_range']}")
        
        # 统计速度参数信息（仅记录，不修改）
        for param in ['vp', 'vs', 'vpv', 'vph', 'vsv', 'vsh', 'rho']:
            if param in data_3d:
                data = data_3d[param]
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    self.logger.info(f"  {param}: 范围 [{valid_data.min():.3f}, {valid_data.max():.3f}], "
                                   f"有效点 {len(valid_data)}")
        
        return data_3d

    # ==================== 🔥 新增：双区域保存方法 ====================
    
    def save_model(self, data_3d: Dict[str, np.ndarray], 
                   coords: Dict[str, np.ndarray],
                   model_name: str):
        """
        🔥 保存两个版本的模型数据：
        1. 原始模型区域版本 (*_original.nc, *_original.csv)
        2. 共同覆盖区域版本 (*_standardized.nc, *_standardized.csv)
        """
        self.logger.info(f"保存模型数据: {model_name}")
        
        output_dir = self.config.paths['output_dir'] / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ========== 1. 保存原始模型区域版本 ==========
        if self.config.processing['save_original_region']:
            self.logger.info(f"\n{'='*60}")
            self.logger.info("📦 保存原始模型区域版本")
            self.logger.info("="*60)
            
            if self.config.processing['save_netcdf']:
                self._save_netcdf_3d(
                    data_3d, coords, output_dir, model_name, 
                    suffix='original',
                    region_type='原始模型区域'
                )
            
            if self.config.processing['save_csv']:
                self._save_csv_from_3d(
                    data_3d, coords, output_dir, model_name,
                    suffix='original',
                    region_type='原始模型区域'
                )
        
        # ========== 2. 保存共同覆盖区域版本 ==========
        if self.config.processing['save_standardized_region'] and self.config.valid_standardized_region is not None:
            self.logger.info(f"\n{'='*60}")
            self.logger.info("📦 保存共同覆盖区域版本")
            self.logger.info("="*60)
            
            # 裁剪到共同区域
            standardized_data_3d, standardized_coords = self._crop_to_valid_region(
                data_3d.copy(), coords.copy(), model_name
            )
            
            if self.config.processing['save_netcdf']:
                self._save_netcdf_3d(
                    standardized_data_3d, standardized_coords, output_dir, model_name,
                    suffix='standardized',
                    region_type='共同覆盖区域'
                )
            
            if self.config.processing['save_csv']:
                self._save_csv_from_3d(
                    standardized_data_3d, standardized_coords, output_dir, model_name,
                    suffix='standardized',
                    region_type='共同覆盖区域'
                )
    
    def _save_netcdf_3d(self, data_3d: Dict[str, np.ndarray],
                       coords: Dict[str, np.ndarray],
                       output_dir: Path,
                       model_name: str,
                       suffix: str = 'original',
                       region_type: str = '原始模型区域'):
        """保存为3D NetCDF格式 + 元信息"""
        nc_file = output_dir / f"{model_name}_{suffix}.nc"
        
        self.logger.info(f"  保存3D NetCDF ({region_type}): {nc_file.name}")
        
        lon = coords['longitude']
        lat = coords['latitude']
        depth = coords['depth']
        
        # 创建数据变量
        data_vars = {}
        for var_name in self.config.standard_parameters:
            if var_name in ['latitude', 'longitude', 'depth']:
                continue
            
            if var_name in data_3d:
                data_transposed = np.transpose(data_3d[var_name], (1, 0, 2))
                
                data_vars[var_name] = xr.DataArray(
                    data_transposed,
                    dims=['longitude', 'latitude', 'depth'],
                    coords={
                        'longitude': lon,
                        'latitude': lat,
                        'depth': depth
                    },
                    attrs={
                        'units': self.config.standard_units.get(var_name, ''),
                        'long_name': var_name
                    }
                )
        
        ds = xr.Dataset(data_vars)
        
        ds['longitude'].attrs = {
            'units': 'degrees_east',
            'long_name': 'Longitude',
            'standard_name': 'longitude'
        }
        ds['latitude'].attrs = {
            'units': 'degrees_north',
            'long_name': 'Latitude',
            'standard_name': 'latitude'
        }
        ds['depth'].attrs = {
            'units': 'km',
            'long_name': 'Depth',
            'standard_name': 'depth',
            'positive': 'down'
        }
        
        model_info = self.config.models[model_name]
        depth_shift = model_info.get('depth_shift', 0.0)
        
        ds.attrs = {
            'title': f"Standardized velocity model: {model_name} ({region_type})",
            'source': model_info['reference'],
            'creation_date': datetime.now().isoformat(),
            'creator': 'EASTASIA-FWI VelocityModelProcessor',
            'format_version': '3D',
            'region_type': region_type,
            'dimension_order': 'longitude, latitude, depth (CF convention)',
            'standard_parameters': ', '.join(self.config.standard_parameters),
            'geospatial_lon_min': float(lon.min()),
            'geospatial_lon_max': float(lon.max()),
            'geospatial_lat_min': float(lat.min()),
            'geospatial_lat_max': float(lat.max()),
            'geospatial_vertical_min': float(depth.min()),
            'geospatial_vertical_max': float(depth.max()),
            'geospatial_lon_resolution': float(np.mean(np.diff(np.sort(lon)))),
            'geospatial_lat_resolution': float(np.mean(np.diff(np.sort(lat)))),
            'geospatial_vertical_resolution': float(np.mean(np.diff(np.sort(depth))))
        }
        
        if depth_shift != 0.0:
            ds.attrs['depth_shift_applied'] = f"{depth_shift:+.1f} km"
        
        if suffix == 'standardized' and self.config.valid_standardized_region is not None:
            # 确保所有numpy类型转换为Python原生类型以便JSON序列化
            def convert_numpy_types(obj):
                """递归转换numpy类型为Python原生类型"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return type(obj)(convert_numpy_types(item) for item in obj)
                else:
                    return obj
            
            region_dict = convert_numpy_types(self.config.valid_standardized_region)
            ds.attrs['valid_standardized_region'] = json.dumps(region_dict)
        
        encoding = {}
        if self.config.processing['compress_netcdf']:
            for var in ds.data_vars:
                encoding[var] = {
                    'dtype': self.config.processing['float_precision'],
                    'zlib': True,
                    'complevel': 4,
                    '_FillValue': np.float32(np.nan)
                }
        
        ds.to_netcdf(nc_file, encoding=encoding, format='NETCDF4')
        
        file_size = nc_file.stat().st_size / (1024**2)
        self.logger.info(f"  ✅ 3D NetCDF已保存 ({region_type}): {nc_file.name} ({file_size:.2f} MB)")
        
        if self.config.processing.get('save_netcdf_metadata', True):
            self._save_netcdf_metadata(ds, output_dir, model_name, suffix, region_type)
        
        ds.close()
    
    def _save_netcdf_metadata(self, ds: xr.Dataset, output_dir: Path, 
                             model_name: str, suffix: str, region_type: str):
        """保存NetCDF元信息到JSON文件"""
        metadata_file = output_dir / f"{model_name}_{suffix}_metadata.json"
        
        metadata = {
            'model_name': model_name,
            'region_type': region_type,
            'file_format': 'NetCDF4',
            'creation_date': datetime.now().isoformat(),
            'global_attributes': dict(ds.attrs),
            'dimensions': {
                'longitude': {
                    'size': len(ds['longitude']),
                    'min': float(ds['longitude'].min()),
                    'max': float(ds['longitude'].max()),
                    'resolution': float(np.mean(np.diff(ds['longitude'].values)))
                },
                'latitude': {
                    'size': len(ds['latitude']),
                    'min': float(ds['latitude'].min()),
                    'max': float(ds['latitude'].max()),
                    'resolution': float(np.mean(np.diff(ds['latitude'].values)))
                },
                'depth': {
                    'size': len(ds['depth']),
                    'min': float(ds['depth'].min()),
                    'max': float(ds['depth'].max()),
                    'resolution': float(np.mean(np.diff(ds['depth'].values)))
                }
            },
            'variables': {}
        }
        
        for var_name in ds.data_vars:
            var = ds[var_name]
            valid_data = var.values[~np.isnan(var.values)]
            
            metadata['variables'][var_name] = {
                'dimensions': list(var.dims),
                'shape': list(var.shape),
                'dtype': str(var.dtype),
                'units': var.attrs.get('units', ''),
                'statistics': {
                    'total_points': int(var.size),
                    'valid_points': int(len(valid_data)),
                    'valid_percentage': float((len(valid_data) / var.size) * 100),
                    'min': float(valid_data.min()) if len(valid_data) > 0 else None,
                    'max': float(valid_data.max()) if len(valid_data) > 0 else None,
                    'mean': float(valid_data.mean()) if len(valid_data) > 0 else None,
                    'std': float(valid_data.std()) if len(valid_data) > 0 else None
                }
            }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"  ✅ NetCDF元信息已保存: {metadata_file.name}")

    def _save_csv_from_3d(self, data_3d: Dict[str, np.ndarray],
                         coords: Dict[str, np.ndarray],
                         output_dir: Path,
                         model_name: str,
                         suffix: str = 'original',
                         region_type: str = '原始模型区域'):
        """从3D数组保存为CSV格式"""
        csv_file = output_dir / f"{model_name}_{suffix}.csv"
        
        self.logger.info(f"  保存1D CSV ({region_type}): {csv_file.name}")
        
        lon = np.sort(coords['longitude'])
        lat = np.sort(coords['latitude'])
        depth = np.sort(coords['depth'])
        
        depth_grid, lat_grid, lon_grid = np.meshgrid(
            depth, lat, lon, indexing='ij'
        )
        
        df_dict = {
            'longitude': lon_grid.flatten(),
            'latitude': lat_grid.flatten(),
            'depth': depth_grid.flatten()
        }
        
        for param in self.config.standard_parameters:
            if param in ['latitude', 'longitude', 'depth']:
                continue
            if param in data_3d:
                data_reordered = np.transpose(data_3d[param], (2, 0, 1))
                df_dict[param] = data_reordered.flatten()
        
        df = pd.DataFrame(df_dict)
        
        # 🔥 关键：只保留至少有一个参数有效的行
        value_cols = [c for c in df.columns if c not in ['longitude', 'latitude', 'depth']]
        df = df.dropna(subset=value_cols, how='all')
        
        # 排序
        df = df.sort_values(by=['depth', 'latitude', 'longitude'])
        df = df.reset_index(drop=True)
        
        # 保存
        df.to_csv(csv_file, index=False, float_format='%.6f', na_rep='nan')
        
        file_size = csv_file.stat().st_size / (1024**2)
        self.logger.info(f"  ✅ 1D CSV已保存 ({region_type}): {csv_file.name} ({file_size:.2f} MB)")
        self.logger.info(f"    总行数: {len(df):,}")
        self.logger.info(f"    有效数据点: {df.notna().any(axis=1).sum():,}")

    def _save_metadata(self, metadata: ModelMetadata, model_name: str):
        """保存模型元数据"""
        metadata_file = self.config.paths['metadata_dir'] / f"{model_name}_metadata.json"
        self.config.paths['metadata_dir'].mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"  元数据已保存: {metadata_file.name}")

    # ==================== 报告生成 ====================
    
    def _generate_report_3d(self, model_data_3d: Dict, coords: Dict, 
                           metadata: ModelMetadata, model_name: str):
        """生成处理报告"""
        self.logger.info(f"生成处理报告: {model_name}")
        
        shape = model_data_3d[list(model_data_3d.keys())[0]].shape
        total_points = np.prod(shape)
        
        report = {
            'model_name': model_name,
            'processing_time': datetime.now().isoformat(),
            'format_version': '3D NetCDF',
            'has_standardized_region': self.config.valid_standardized_region is not None,
            'csv_sort_order': 'depth(慢) → latitude → longitude(快)',
            'depth_grid': {
                'min': float(coords['depth'].min()),
                'max': float(coords['depth'].max()),
                'n_points': len(coords['depth']),
                'contains_zero': bool(0.0 in coords['depth'])
            },
            'metadata': asdict(metadata),
            'data_statistics': {}
        }
        
        for param in self.config.standard_parameters:
            if param in ['latitude', 'longitude', 'depth']:
                continue
            if param in model_data_3d:
                data = model_data_3d[param]
                valid_data = data[~np.isnan(data)]
                
                if len(valid_data) > 0:
                    report['data_statistics'][param] = {
                        'total_points': int(total_points),
                        'valid_points': int(len(valid_data)),
                        'valid_percentage': float((len(valid_data) / total_points) * 100),
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data)),
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'q25': float(np.percentile(valid_data, 25)),
                        'q50': float(np.percentile(valid_data, 50)),
                        'q75': float(np.percentile(valid_data, 75))
                    }
        
        report_file = self.config.paths['results_dir'] / f"{model_name}_processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"  处理报告已保存: {report_file.name}")

    # ==================== 主处理流程 ====================
    
    def _compute_parameter_ranges(self, data_3d: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """计算各参数的值范围统计"""
        parameter_ranges = {}
        
        for param, data in data_3d.items():
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                parameter_ranges[param] = {
                    'min': round(float(np.min(valid_data)), 6),
                    'max': round(float(np.max(valid_data)), 6),
                    'mean': round(float(np.mean(valid_data)), 6),
                    'std': round(float(np.std(valid_data)), 6),
                    'valid_count': int(len(valid_data)),
                    'total_count': int(data.size),
                    'valid_percent': round(float(len(valid_data) / data.size * 100), 2)
                }
        
        return parameter_ranges
    
    def _update_metadata_with_data(self, metadata: ModelMetadata, 
                                   data_3d: Dict[str, np.ndarray],
                                   coords: Dict[str, np.ndarray]) -> ModelMetadata:
        """使用实际数据更新 metadata（修复浮点数精度问题）"""
        # 更新坐标范围（四舍五入避免浮点精度问题）
        lat = coords['latitude']
        lon = coords['longitude']
        depth = coords['depth']
        
        metadata.lat_range = (round(float(lat.min()), 6), round(float(lat.max()), 6))
        metadata.lon_range = (round(float(lon.min()), 6), round(float(lon.max()), 6))
        metadata.depth_range = (round(float(depth.min()), 6), round(float(depth.max()), 6))
        
        # 更新数据形状
        first_param = list(data_3d.keys())[0]
        metadata.data_shape = data_3d[first_param].shape
        
        # 更新分辨率
        metadata.resolution = {
            'latitude': round(float(np.mean(np.diff(np.sort(lat)))), 6),
            'longitude': round(float(np.mean(np.diff(np.sort(lon)))), 6),
            'depth': round(float(np.mean(np.diff(np.sort(depth)))), 6)
        }
        
        # 🔥 新增：计算并更新参数值范围
        metadata.parameter_ranges = self._compute_parameter_ranges(data_3d)
        
        return metadata
    
    def process_model(self, model_name: str) -> Tuple[Dict, Dict, ModelMetadata]:
        """完整处理流程"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"开始处理速度模型: {model_name}")
        self.logger.info(f"{'='*80}\n")
        
        try:
            metadata = self.analyze_model_structure(model_name)
            data_3d, coords = self.read_model(model_name)
            std_data_3d, coords = self.standardize_model_3d(data_3d, coords, model_name)
            
            # 🔥 使用实际数据更新 metadata（包括参数范围和修复浮点精度）
            metadata = self._update_metadata_with_data(metadata, std_data_3d, coords)
            self._save_metadata(metadata, model_name)
            
            self.save_model(std_data_3d, coords, model_name)
            
            if self.config.processing['generate_report']:
                self._generate_report_3d(std_data_3d, coords, metadata, model_name)
            
            self.logger.info(f"\n✅ 模型 {model_name} 处理完成！")
            self.logger.info(f"  • 原始区域版本: *_original.nc/csv")
            self.logger.info(f"  • 共同区域版本: *_standardized.nc/csv\n")
            return std_data_3d, coords, metadata
            
        except Exception as e:
            self.logger.error(f"处理模型 {model_name} 时出错: {str(e)}", exc_info=True)
            raise
    
    def process_all_models(self, model_names: Optional[List[str]] = None, 
                          compute_common_region: bool = False):
        """
        批量处理模型
        
        Args:
            model_names: 要处理的模型列表，默认处理所有模型
            compute_common_region: 是否计算共同覆盖区域，默认为 False
        """
        if model_names is None:
            model_names = list(self.config.models.keys())
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"开始批量处理 {len(model_names)} 个模型")
        self.logger.info("="*80 + "\n")
        
        # 可选：计算共同覆盖区域
        if compute_common_region and len(model_names) >= 2:
            try:
                self.compute_valid_standardized_region(model_names)
            except Exception as e:
                self.logger.warning(f"⚠️ 计算共同区域失败: {e}")
                self.config.valid_standardized_region = None
        else:
            self.config.valid_standardized_region = None
        
        results = {}
        success_count = 0
        failed_count = 0
        
        for model_name in model_names:
            try:
                data_3d, coords, metadata = self.process_model(model_name)
                
                results[model_name] = {
                    'status': 'success',
                    'original_region': {
                        'data_shape': list(data_3d[list(data_3d.keys())[0]].shape),
                        'lat_range': (float(coords['latitude'].min()), float(coords['latitude'].max())),
                        'lon_range': (float(coords['longitude'].min()), float(coords['longitude'].max())),
                        'depth_range': (float(coords['depth'].min()), float(coords['depth'].max()))
                    },
                    'parameters': list(data_3d.keys()),
                    'metadata': asdict(metadata)
                }
                
                if self.config.valid_standardized_region:
                    try:
                        standardized_data, standardized_coords = self._crop_to_valid_region(
                            data_3d.copy(), coords.copy(), model_name
                        )
                        results[model_name]['standardized_region'] = {
                            'data_shape': list(standardized_data[list(standardized_data.keys())[0]].shape),
                            'depth_range': (float(standardized_coords['depth'].min()), 
                                          float(standardized_coords['depth'].max()))
                        }
                    except Exception as e:
                        self.logger.warning(f"  裁剪到共同区域失败: {e}")
                
                success_count += 1
                    
            except Exception as e:
                results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                self.logger.error(f"❌ 处理 {model_name} 失败: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                failed_count += 1
        
        # 保存处理总结
        summary_file = self.config.paths['results_dir'] / 'processing_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"✅ 批量处理完成！")
        self.logger.info(f"  成功: {success_count} 个模型")
        self.logger.info(f"  失败: {failed_count} 个模型")
        self.logger.info(f"  总结报告: {summary_file}")
        self.logger.info("="*80 + "\n")
        
        return results


def main():
    """主函数 - 从config读取参数选择要处理的模型"""
    
    print("="*80)
    print("EASTASIA-FWI 速度模型预处理")
    print("支持多种格式：H5, NetCDF, GeoCSV, TXT, Excel, 多深度文件等")
    print("="*80)
    
    # 初始化配置和处理器
    config = VelocityModelConfig()
    processor = VelocityModelProcessor(config)
    
    # 从config读取运行时参数
    runtime_config = config.runtime
    
    # 列出所有模型
    if runtime_config['list_models']:
        print("\n📋 可用模型列表:")
        print("-"*60)
        for i, (name, info) in enumerate(config.models.items(), 1):
            print(f"  {i:2d}. {name}")
            print(f"      格式: {info['format']}")
            print(f"      区域: {info.get('region', 'Unknown')}")
            print(f"      参数: {', '.join(info.get('params', []))}")
            print()
        print(f"共 {len(config.models)} 个模型")
        return
    
    # 确定要处理的模型（优先级：single_model > model_names > None）
    if runtime_config['single_model']:
        model_names = [runtime_config['single_model']]
    elif runtime_config['model_names']:
        model_names = runtime_config['model_names']
    else:
        model_names = None  # 处理所有模型
    
    # 验证模型名称
    if model_names:
        valid_models = []
        for name in model_names:
            if name in config.models:
                valid_models.append(name)
            else:
                print(f"⚠️ 未知模型: {name}")
        model_names = valid_models if valid_models else None
    
    print("\n🔥 核心功能:") 
    print("  ✅ 标准化输出：统一的 NetCDF 3D 和 CSV 1D 格式")
    print("  ✅ 参数标准化：vpv, vph, vsv, vsh, rho, vs, vp 等")
    print("  ✅ 深度平移支持（如 SinoScope1.0）")
    print("  ✅ CSV 排序：depth(慢) → latitude → longitude(快)")
    print("="*80)
    
    if model_names:
        print(f"\n将处理以下 {len(model_names)} 个模型:")
        for name in model_names:
            print(f"  • {name}")
    else:
        print(f"\n将处理所有 {len(config.models)} 个模型")
    
    print()
    
    # 执行处理
    results = processor.process_all_models(
        model_names=model_names,
        compute_common_region=runtime_config['compute_common_region']
    )
    
    # 打印处理结果
    print("\n" + "="*80)
    print("✨ 处理完成！")
    print("="*80)
    print(f"\n📁 输出目录: {config.paths['output_dir']}")
    print(f"📊 结果目录: {config.paths['results_dir']}")
    
    print("\n📊 处理结果:")
    success_models = [k for k, v in results.items() if v.get('status') == 'success']
    failed_models = [k for k, v in results.items() if v.get('status') == 'failed']
    
    if success_models:
        print(f"\n  ✅ 成功 ({len(success_models)}):")
        for name in success_models:
            info = results[name]
            shape = info.get('original_region', {}).get('data_shape', [])
            params = info.get('parameters', [])
            print(f"      {name}: {shape}, 参数: {', '.join(params[:5])}{'...' if len(params) > 5 else ''}")
    
    if failed_models:
        print(f"\n  ❌ 失败 ({len(failed_models)}):")
        for name in failed_models:
            error = results[name].get('error', 'Unknown error')
            print(f"      {name}: {error[:80]}...")
    
    print("\n输出文件格式:")
    print("  • *_original.nc (NetCDF 3D) - 原始模型区域")
    print("  • *_original.csv (CSV 1D) - 原始模型区域")
    print("  • *_original_metadata.json - 元数据")
    if runtime_config['compute_common_region']:
        print("  • *_standardized.nc/csv - 共同覆盖区域")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    main()