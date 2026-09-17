# EASTASIA-FWI — Agent 指令

东亚全波形成像项目（别名 SMART-py）：基于 SPECFEM3D Globe，通过模型空间分析（聚类 / CW-SSIM 相似性）、
数据空间波形评估和多模型融合（Voting / BMA / SDL），构建东亚岩石圏初始模型并推进 FWI，目标产出 SEAM 1.0。

## 沟通与语言

- 用中文回答；代码注释用中文；图表标题、坐标轴、图例用英文。
- 术语遵循地震学规范（如 "Moho"、"LAB"、"radial anisotropy"、"CW-SSIM"），不用"简化版 / 增强版"之类模糊词。
- 保留用户已有修改，在其基础上完善；不要重写用户未提及的部分。
- 单文件超过 1200 行或涉及多文件时分批提供。

## 权威来源（不要硬编码）

- 研究区域、目录路径、日志：只从 `config/base_config.py` 的 `BaseConfig` 读取
  （`self.dirs` / `setup_logger()` / `get_region_bounds(kind)` / `get_gmt_region(kind)` / `is_in_region(..., kind=)`）。
- 研究区域分三层，用 `kind` 选择，不要在模块内另定一份：
  - `'study'`（默认）：总研究区，台站/事件查询、底图。
  - `'common'`：核心三模型 FWEA23 ∩ EARA2024 ∩ SinoScope1.0 交集，用于 standardized 对比、正演事件-台站筛选。
  - 模型名（`'2024_FWEA23'` 或短名 `'FWEA23'`）：单模型原始覆盖范围；未登记模型自动读 `data/models/metadata/<model>_metadata.json`。
  - 核心模型列表取 `base_config.core_models`；多模型交集用 `compute_common_region([...])`。
- 速度模型标准化格式：NetCDF，坐标 `(latitude, longitude, depth)`，
  变量名 `vp vs vpv vph vsv vsh rho eta`，单位 km/s、km、g/cm³。
- 项目当前状态与待办：`PROJECT_STATUS.md`、`README.md`（开发优先级章节）。
- 详细设计文档：`docs/design/`（主设计 v2.2、SDL 设计）；历史计划：`docs/plans/`。
- SPECFEM 正演工作流与已知问题：`specfem/*.md`、`docs/FWEA23_*_workflow_server.md`、`docs/问题与资源清单.md`。

## 目录与命名

```
1_Data_preparation/  2_Model_space_analysis/  3_Data_space_simulation/  4_Fusion/  5_Visualization/
config/  scripts/  tests/  specfem/  docs/  archive/  web/  paper/
data/ results/ figures/ logs/ output/   ← 不入库，禁止由 agent 直接写入或删除
```

- 代码文件：`模块号_功能.py`（如 `2_3_Model_clustering.py`）；新脚本沿用所在模块的序号规则。
- 输出图片：`模块号-子序号_描述.jpg`，同时保存 jpg 与 pdf（300 dpi）。
- 历史版本进 `archive/`，不放在主模块目录里；一次性实验脚本进 `scripts/`。

## 代码架构约定

- 双层配置：`BaseConfig`（全局）+ 模块内 `XxxConfig` 类（模块参数、`visualization`、`logging` 字段）。
- 模块主类初始化顺序：BaseConfig → 模块 Config → output_dir → `self.logger = base_config.setup_logger('EASTASIA-FWI.Xxx', level)` → region。
- 所有函数完整类型注解（`from typing import ...`）；数据结构优先 `@dataclass`。
- 新代码用 `logging`，不要新增 `print()`（`main()` 中的进度提示除外）。
- 大数据处理注意内存（分块读取 NetCDF、避免不必要的 `.values` 全量物化）。
- 长任务支持断点续传 / 进度保存。
- 详细的模块模板、文件头、`main()` 骨架见 `.github/instructions/python-modules.instructions.md`（编辑 `.py` 时自动加载）。

## 工具偏好

- 地图：PyGMT 首选；切片图海岸线：Cartopy；统计图：Matplotlib/Seaborn。
- 配色：绝对速度 `jet_r` / `seismic`，扰动或差值 `RdBu_r`，台网 `tab20`+`tab20b`，聚类用离散色。
- 环境：conda `eastasia_fwi`（Python 3.12）；测试用 `pytest`，格式化用 `black`（仅对新文件与 `tests/` 强制）。

## 安全与边界

- 不要修改、删除或"清理" `data/ results/ figures/ output/ logs/ backup/` 下的任何文件。
- 不要改动 `specfem/`、`specfem3d_globe*/` 下的 Fortran 源码或 Par_file，除非用户明确要求。
- 不要在代码中硬编码区域边界、绝对路径或服务器账号。
- 修改算法参数（聚类 K、SDL 字典大小、阈值等）前先说明预期影响，并优先补/跑 `tests/` 里的回归测试。

## 参考高质量模块

`1_1_Query_stations.py`（配置类/日志/异常）、`1_4_Preprocess_waveforms.py`（批处理/断点续传）、
`2_1_Model_compare.py`（NetCDF 处理/报告）、`5_1_Basemap.py`（PyGMT 地图/色标）、
`5_5_Permanent_Stations.py`（数据筛选/统计可视化）。
