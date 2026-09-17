# Custom instructions for Copilot

## 项目背景

EASTASIA-FWI 是基于 SPECFEM3D Globe 的东亚地区全波形成像项目，通过模型空间智能分析（聚类、相似性指数、机器学习算法）和数据空间波形评估，建立东亚地区三维地球结构初始模型，结合大规模波形数据的高精度全波形反演，得到新一代标准东亚岩石圈模型 SEAM 1.0。

**项目别名**: SMART-py (Seismic Model Assessment and Reconciliation Toolbox in Python)

### 研究区域

- **设计区域**: 经度 60°E ~ 170°E，纬度 15°S ~ 60°N，深度 0 ~ 1000 km
- **当前配置 (base_config.py)**: 经度 60°E ~ 150°E，纬度 10°S ~ 60°N（实际跑通的工作子区域）
- **以 `config/base_config.py` 的 `self.region` 为权威值**：生成代码时务必读取该配置，切勿硬编码区域边界

### 核心技术栈

- **SPECFEM3D Globe**: 地震波正演/反演
  - **v8.1.0 (EMC 原生路径)**: `specfem3d_globe/` + `specfem3d_globe-EMC/`，直接读取 EMC NetCDF（`model_EMC.f90`，各向异性 11 参数），当前主推路径
  - **v7.0.0 (Chujie 修改版, GLL 路径)**: `specfem/specfem3d_globe_code_new/`，配合 `convert_nc_to_gll.py` 走 NC→GLL 转换
- **ObsPy/SAC**: 地震学数据处理（含 MassDownloader 波形下载）
- **scikit-learn/scikit-image**: 机器学习、相似性分析（SSIM/CW-SSIM）、字典学习
- **xarray/h5py/scipy**: NetCDF/HDF5 速度模型读写与插值
- **PyGMT/GMT**: 地球物理可视化（首选）
- **Cartopy**: 地理投影和海岸线（切片图）
- **Matplotlib/Seaborn**: 统计图表

> 📌 **完整设计参见**: `.github/instructions/EastAsia_FWI_Development.md`（主开发设计文档 v2.2，含科学目标、23 个速度模型库、双数据库架构、融合策略、波形评估指标体系、里程碑与 Open Issues）。本文件仅作为面向 Copilot 的精简开发约定。

---

## 🎯 核心开发指令

### 基本原则

- **专业性**: 严格遵循地震学和全波形反演科学标准
- **完整性**: 每次提供完整代码文件，包含完整导入、类定义、方法实现
- **一致性**: 遵循项目已建立的代码架构、命名规范和模块接口
- **统一配置**: 所有模块使用 `base_config.py` 作为全局配置，模块内使用独立配置类
- **中文回答**: 代码注释用中文，可视化图表标题/标签用英文
- **保留修改**: 如果用户有自己的修改，保留并在此基础上完善

### 📝 代码质量要求

- **类型提示**: 所有函数使用完整类型注解 (`from typing import ...`)
- **异常处理**: 完整的错误处理和日志记录
- **注释规范**: 使用准确的地震学术语，避免"简化版"、"增强版"等模糊描述
- **性能优化**: 支持大规模地震学数据处理，注意内存管理
- **日志系统**: 使用 `logging` 模块，通过 `self.base_config.setup_logger()` 初始化

### 🎨 可视化策略

- **统一管理**: 大部分可视化功能放在 `5_Visualization/` 模块，部分简单可视化可内嵌在功能模块中
- **统一命名**: 图片命名加模块序号前缀，如 `2-1_clustering.jpg`, `5-4_stations.jpg`
- **首选工具**: PyGMT 用于地图绑制，Matplotlib 用于统计图表，Cartopy 用于切片图的海岸线，还有其他合适的绘图工具
- **多级支持**: 从数据探索到发表质量的多级可视化
- **输出格式**: 同时保存 jpg (300dpi) 和 PDF (300dpi) 格式

### 🚀 响应策略

- **长度评估**: 单文件 > 1200 行或多文件时分批提供
- **保留用户修改**: 更新时保留用户自定义内容
- **算法创新**: 集成现代机器学习方法
- **工程创新**: 支持大规模并行计算

---

## ⚙️ 配置管理架构

### 双层配置模式

项目采用**全局配置 + 模块配置类**的双层架构：

1. **`base_config.py`**: 全局配置（路径、区域参数、通用设置）
2. **模块内配置类**: 每个模块在代码文件内定义自己的 `XxxConfig` 类

### 全局配置 (base_config.py)

```python
class BaseConfig:
    def __init__(self):
        # 研究区域
        self.region = {
            'name': 'EastAsia',
            'lat_min': -15.0, 'lat_max': 60.0,
            'lon_min': 60.0, 'lon_max': 170.0,
            'depth_min': 0, 'depth_max': 1000
        }
        # 项目目录
        self.dirs = {
            'project_root': Path(...),
            'data': Path(...),
            'results': Path(...),
            'figures': Path(...),
            'stations': Path(...),
            # ...
        }

    def setup_logger(self, name: str, level: str) -> logging.Logger:
        """统一日志初始化"""
        ...

    def get_region_bounds(self) -> dict:
        """获取区域边界"""
        ...
```

### 模块配置类模板

```python
class ModuleNameConfig:
    """模块配置类 - 放在模块代码文件内"""

    def __init__(self):
        # 模块特定参数
        self.param1 = {...}
        self.param2 = {...}

        # 可视化配置
        self.visualization = {
            'dpi': 300,
            'pdf_dpi': 300,
            'figure_format': ['jpg', 'pdf'],
        }

        # 日志配置
        self.logging = {
            'level': 'INFO',
            'console_output': True
        }
```

### 模块初始化模板

```python
class ModuleName:
    """模块主类"""

    def __init__(self, output_dir: Optional[str] = None):
        # 1. 加载全局配置
        self.base_config = BaseConfig()

        # 2. 加载模块配置
        self.config = ModuleNameConfig()

        # 3. 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.base_config.dirs['figures']
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 4. 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.ModuleName',
            self.config.logging['level']
        )

        # 5. 设置区域参数
        region_bounds = self.base_config.get_region_bounds()
        self.region = [
            region_bounds['lon_min'], region_bounds['lon_max'],
            region_bounds['lat_min'], region_bounds['lat_max']
        ]

        self.logger.info("🎯 模块初始化完成")
        self._print_config_summary()
```

---

## 📁 项目结构与开发状态 (2026 年 6 月)

```
EASTASIA-FWI/
├── 1_Data_preparation/                 # ✅ 数据准备（基本完成）
│   ├── 1_1_Query_stations.py           # ✅ 台站查询（高质量参考）
│   ├── 1_2_Process_GCMT_catalogs.py    # ✅ GCMT 事件处理
│   ├── 1_3_Download_waveforms.py       # ✅ 波形下载（ObsPy MassDownloader）
│   ├── 1_4_Preprocess_waveforms.py     # ✅ 波形预处理（高质量参考）
│   ├── 1_5_Process_velocity_models.py  # ✅ 速度模型标准化 → NetCDF
│   ├── 1_5b_Extract_subregion.py       # ✅ 模型子区域裁剪
│   └── 1_7_Build_test_database.py      # ✅ 标准测试数据库构建（CMT3D + 固定台站）
│
├── 2_Model_space_analysis/             # ✅ 模型空间分析（核心完成）
│   ├── 2_1_Model_compare.py            # ✅ 速度模型对比（高质量参考）
│   ├── 2_2_Model_similarity.py         # ✅ SSIM/CW-SSIM 相似性分析
│   └── cwssim_index/                   # ✅ CW-SSIM 复小波相似性实现
│
├── 3_Data_space_simulation/            # 🔄 数据空间模拟（开发中）
│   ├── 3_1_Setup_specfem3d_globe.py    # 🔄 SPECFEM3D Globe 算例/参数设置
│   └── templates/                      # ✅ SPECFEM3D 模板（Par_file 等）
│
├── 4_Fusion/                           # 🔄 模型融合（开发中）
│   ├── 4_1_Voting_map.py               # ✅ 模型一致性投票（v1.0）
│   ├── 4_2_BMA.py                      # ✅ 贝叶斯模型平均（v2.0）
│   └── 4_3_SDL_Fusion_v3.4.py          # 🔄 稀疏字典学习融合（v3.4）
│
├── 5_Visualization/                    # ✅ 可视化（基本完成）
│   ├── 5_1_Basemap.py                  # ✅ 基础地图（高质量参考）
│   ├── 5_2_GCMT.py                     # ✅ GCMT 事件可视化
│   ├── 5_3_All_Stations.py             # ✅ 全部台站分布
│   ├── 5_4_Station_Comparison.py       # ✅ 与 FWEA23 台站对比
│   ├── 5_5_Permanent_Stations.py       # ✅ 永久台站分布（高质量参考）
│   ├── 5_6_Event_Sampling.py           # ✅ 事件采样优化
│   ├── 5_7_Moho_LAB.py                 # ✅ Moho/LAB 深度可视化
│   ├── data_EastAsia/                  # ✅ 地理数据（地形/构造/火山/Slab2）
│   └── utils/                          # ✅ 可视化通用工具
│
├── specfem/                            # 🔄 SPECFEM 正演工作区（多模型，GLL 路径）
│   ├── run_forward.bash                #    多模型正演主脚本
│   ├── convert_nc_to_gll.py            #    NC/HDF5/CSV → GLL 转换
│   ├── DATA/ DATABASES_MPI/ OUTPUT_FILES/   #  共享配置/网格/输出
│   ├── models/ GLL/ model_updated/     #    原始模型与 GLL 文件
│   ├── FWEA23/ EARA2024/ SinoScope/    #    各模型独立工作目录
│   ├── specfem3d_globe_code_new/       #    SPECFEM 源码（v7.0.0 修改版）
│   └── sem_config_tibet_v4/            #    区域 chunk 配置（constants.h.in）
│
├── config/base_config.py               # ✅ 全局配置（区域/路径/日志，权威来源）
├── data/                               # ✅ models / catalogs / stations / events
├── docs/                               # ✅ 工作流文档、季度总结、问题与资源清单
├── results/ figures/ logs/ output/     # ✅ 结果 / 图表 / 日志 / 正演输出
├── backup/ temp/                       # ✅ 备份 / 临时
├── specfem3d_globe/                    # 🔄 SPECFEM3D Globe v8.1.0（EMC 原生路径）
├── specfem3d_globe-EMC/                # 🔄 EMC 模型相关修改版
└── specfem_original/                   # ✅ 服务器成功编译的原始代码备份
```

**状态**: ✅ 已完成 | 🔄 开发中 | ⏳ 待开发

> ⚠️ 模块 3/4 为当前关键路径。SPECFEM 正演存在已知阻塞（曙光集群 MPI `face flag` 间歇错误、solver 发散、EMC NetCDF 各向异性遗漏等），相关诊断记录见 `docs/` 与仓库记忆。

---

## 📐 代码风格规范

### 模块文件头模板

```python
"""
EASTASIA-FWI 模块名称
======================

功能描述:
- 功能1
- 功能2

科学原理:
- 原理说明

作者: EASTASIA-FWI Team
日期: 2025-xx-xx
版本: vX.X
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import warnings

import numpy as np
import pandas as pd
# ... 其他导入

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入基础配置
from config.base_config import BaseConfig

warnings.filterwarnings('ignore')
```

### 类定义规范

```python
@dataclass
class DataClass:
    """数据类定义（使用dataclass装饰器）"""
    field1: str
    field2: int
    field3: Optional[float] = None


class ConfigClass:
    """配置类定义（使用__init__初始化）"""

    def __init__(self):
        self.param1 = {...}
        self.param2 = {...}


class MainClass:
    """主功能类"""

    def __init__(self, ...):
        """初始化方法 - 遵循标准初始化流程"""
        ...

    def _private_method(self, ...) -> ReturnType:
        """私有方法（以下划线开头）"""
        ...

    def public_method(self, ...) -> ReturnType:
        """
        公开方法

        Args:
            param1: 参数说明
            param2: 参数说明

        Returns:
            返回值说明
        """
        ...
```

### 主函数模板

```python
def main():
    """主函数"""
    print("🎯 EASTASIA-FWI 模块名称")
    print("="*60)

    try:
        # 初始化
        processor = MainClass()

        # 执行处理
        results = processor.run()

        # 显示结果
        print(f"\n✅ 处理完成!")
        print(f"📁 输出目录: {processor.output_dir}")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断处理")
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

    print("="*60)


if __name__ == "__main__":
    main()
```

---

## 📈 当前开发优先级

### 🔥 本周重点

1. **SPECFEM 正演阻塞排查**: 解决曙光集群 MPI `face flag` 间歇错误 / solver 发散 / EMC NetCDF 各向异性遗漏
2. **多模型正演跑通**: FWEA23 / EARA2024 / SinoScope 三模型，统一事件、台站、频段

### 🚀 短期目标（1-2 周）

1. 三模型合成波形产出（SAC）并完成质检
2. 开发波形评估指标体系（χ^T / χ^A / χ^F / NZCC，参考 Zhou et al. 2021）
3. NC→GLL 与 EMC 原生路径的插值精度对比（量化系统偏差）

### 🎯 中期目标（1 个月）

1. 完成数据空间模拟与波形评估模块
2. 推进融合方法验证（BMA / Voting / 岩石圈叠加），SDL/PGM 服务 Paper I
3. 由波形评分驱动选择 FWI 初始模型

---

## 🛠️ 技术规范

### 数据流架构

```
GCMT/CMT3D 事件 + 台站数据 → 事件采样 → 波形下载 → 预处理
                                      ↓
速度模型库(NetCDF) → 模型对比 → SSIM/CW-SSIM 相似性 → 代表性模型选择
                                      ↓
代表性模型 → NC→GLL / EMC 原生 → SPECFEM3D 正演 → 合成波形
                                      ↓
观测 + 合成波形 → 波形评估(χ^T/χ^A/χ^F/NZCC) → 投票/BMA → FWI 初始模型 → SEAM 1.0
```

### 文件命名规范

- **代码文件**: `模块号_功能描述.py` (如 `2_1_Model_compare.py`)
- **输出图片**: `模块号-子序号_描述.jpg` (如 `2-1_velocity_comparison.jpg`)
- **数据文件**: `描述_日期.csv` (如 `gcmt_events_2000_2024.csv`)
- **配置文件**: `模块名_config.yaml` (如需要外部配置)

### 速度模型数据格式

- **标准化 NetCDF**: 所有速度模型转换为统一的 NetCDF 格式
- **坐标系统**: (latitude, longitude, depth)
- **参数命名**: vs, vp, vsv, vsh, vpv, vph, rho, eta
- **单位**: 速度 km/s, 深度 km, 密度 g/cm³

### 可视化配色规范

- **速度模型**: `jet_r` 或 `seismic` (绝对值), `RdBu_r` (扰动/差异)
- **网络分布**: `tab20` + `tab20b` (Matplotlib)
- **聚类结果**: 离散配色方案

---

## 🚨 特殊开发指令

### 响应策略

- **长度评估**: 代码超过 1200 行主动分批提供
- **优先级**: 当前优先打通 SPECFEM 多模型正演与波形评估（模块 3/4 关键路径）
- **依赖管理**: 确保新模块与现有架构兼容
- **测试驱动**: 每个新模块包含基本功能测试

### 质量保证

- **代码审查**: 确保与现有架构的一致性（参考高质量模块）
- **性能考虑**: 支持大规模数据处理，注意内存管理
- **文档完整**: 每个函数有详细的 docstring
- **错误处理**: 完整的异常处理和日志记录
- **进度保存**: 长时间任务支持断点续传

### 参考高质量模块

- `1_1_Query_stations.py`: 配置类设计、日志系统、错误处理
- `1_4_Preprocess_waveforms.py`: 批处理、进度保存、性能优化
- `2_1_Model_compare.py`: NetCDF 数据处理、多种可视化、报告生成
- `5_1_Basemap.py`: PyGMT 使用、地图绘制、色标管理
- `5_5_Permanent_Stations.py`: 数据筛选、统计分析、时间线可视化

---

**项目当前状态**: 数据准备与模型空间分析完成，正演模拟为当前关键路径（受 SPECFEM MPI/收敛问题阻塞）
**下一里程碑**: 跑通三模型正演 → 建立波形评估框架 → 由评分驱动选择 FWI 初始模型
**更新日期**: 2026 年 6 月 10 日

---

_此指令文件根据项目实际代码状态更新，确保开发方向与科研目标保持一致。_
