# EASTASIA_FWI

## 🌏 概述

EASTASIA-FWI 东亚地区全波形成像项目，通过模型空间的智能分析（聚类分析和相似性指数分析，或者其他先进算法）和数据空间的波形评估，建立东亚地区三维地球结构初始模型，再结合大规模的高精度全波形反演，得到新一代标准东亚岩石圈模型SEAM 1.0。

**研究区域**: 纬度 -15°至60°，经度 50°至170°  
**核心特色**: 模型聚类分析 + 相似性评估 + 波形模拟评估 + 初始模型融合 + 全波形反演

## 📁 项目结构

```
EASTASIA-FWI/
├── 1_Data_preparation/             # 数据准备模块
│   ├── 1_1_Query_stations.py           # 台站查询与筛选 ⭐
│   ├── 1_2_Process_GCMT_catlogs.py     # GCMT事件处理 ⭐
│   ├── 1_3_Download_wa.py            # 数据下载管理 ⭐
│   ├── 1_4_Preprocessing.py            # 波形预处理 ⭐
│   ├── 1_5_Process_models.py           # 速度模型处理 ⭐
│   ├── 1_6_Quality_control.py          # 数据质量控制
│   └── utils/                          # 数据处理工具
├── 2_Model_space_analysis/         # 模型空间分析
│   ├── 2_1_Model_clustering.py         # 聚类分析 ⭐
│   ├── 2_2_Model_similarity.py         # 相似性分析 ⭐
│   ├── 2_3_Model_fusion.py             # 模型融合 ？
│   ├── 2_4_Representative_selection.py  # 代表性模型选择
│   └── utils/                          # 模型分析工具
├── 3_Data_space_simulation/        # 数据空间模拟
│   ├── 3_1_Setup_specfem.py           # SPECFEM3D参数设置 ⭐
│   ├── 3_2_Model_converter.py         # 模型格式转换
│   ├── 3_3_Run_simulation.py          # 正演计算管理
│   ├── 3_4_Batch_manager.py           # 批量任务管理
│   ├── templates/                     # SPECFEM3D模板文件
│   └── utils/                         # 模拟工具
├── 4_Evaluation/                   # 波形评估模块
│   ├── 4_1_Waveform_metrics.py        # 波形拟合指标
│   ├── 4_2_Model_ranking.py           # 模型评估排序
│   ├── 4_3_Statistical_analysis.py    # 统计分析
│   ├── 4_4_Uncertainty_analysis.py    # 不确定性分析
│   └── utils/                         # 评估工具
├── 5_Visualization/                # 可视化模块
│   ├── 5_1_Basemap_plotting.py        # 地图绘制 ⭐
│   ├── 5_2_GCMT_plotting.py          # GCMT事件可视化 ⭐
│   ├── 5_3_Station_plotting.py        # 台站分布可视化 ⭐
│   ├── 5_4_Event_Station_Selection.py # 事件台站选择可视化 ⭐
│   ├── 5_5_Waveform_plotting.py       # 波形可视化 ⭐
│   ├── 5_4_Clustering_visualization.py # 聚类结果可视化
│   ├── 5_5_Similarity_visualization.py # 相似性分析可视化
│   └── utils/                         # 绘图工具
├── config/                         # 配置管理
│   ├── config.py                      # 全局配置
│   ├── model_configs.py               # 模型配置
│   ├── clustering_config.py           # 聚类分析配置 ⭐
│   └── region_configs.py              # 区域配置
├── data/                           # 数据目录
│   ├── models/                        # 速度模型数据
│   ├── events/                        # 地震事件数据
│   ├── stations/                      # 台站数据
│   └── waveforms/                     # 波形数据
├── results/                        # 结果输出
│   ├── clustering/                    # 聚类分析结果
│   ├── similarity/                    # 相似性分析结果
│   ├── fusion/                        # 模型融合结果
│   └── evaluation/                    # 波形评估结果
├── backup/                         # 开发中的工具
│   ├── Seis_Model_Clustering/         # K-means聚类工具
│   └── Seis_Model_Similarity/         # SSIM相似性工具
├── specfem3d_globe/               # SPECFEM3D Global
├── output/                        # 模拟输出
├── figures/                       # 图表输出
├── logs/                          # 日志文件
└── docs/                          # 项目文档
```

## 🚀 环境要求

# 创建环境: conda env create -f environment.yml

# 激活环境: conda activate eastasia_fwi

## 🌐 Web 展示界面

项目提供**纯静态**展示网站，无需任何依赖：

### 直接打开（推荐本地预览）

```bash
# 在项目根目录，用浏览器打开
open web/index.html
# 或
xdg-open web/index.html   # Linux
```

## 📊 数据格式说明

**支持的模型**:

- global models: ak135, prem, iasp91
- regional models:

```csv
ID,Model name,Model type,"Model scope (regional >25°, or local <25°)",Publication year,Reference,DOI,Min_longitude,Max_longitude,Min_latitude,Max_latitude,Min_depth,Max_depth
1,2018_CSEM_Japan,Vs,regional,2018.0,"Fichtner et al., 2018 GRL","https://doi.org/10.1029/2018GL077338",120.0,150.0,20.0,50.0,0.0,350.0
2,2018_FWEA18,Vp & Vs,regional,2018.0,"Tao et al., 2018 G3","https://doi.org/10.1029/2018GC007460",90.0,140.0,15.0,50.0,0.0,800.0
3,2020_TP2019,Vs,regional,2020.0,"Xiao et al., 2020 JGR","https://doi.org/10.1029/2019JB018344",74.0,108.0,27.0,42.0,25.0,700.0
4,2021_Banda_ANT_CrustVs_2020,Vs,local,2021.0,"Zhang and Miller 2021 GRL","https://doi.org/10.1029/2020GL089632",119.0,127.0,-11.0,-8.0,0.0,50.0
5,2021_Chen_SChinaSea,Vs,regional,2021.0,"Chen et al., 2021 G3","https://doi.org/10.1029/2020GC009356",100.0,120.0,-5.0,25.0,0.0,250.0
6,2021_KEA20,Vs,regional,2021.0,"Witek et al., 2021 JGR","https://doi.org/10.1029/2020JB021201",90.0,135.0,20.0,45.0,25.0,250.0
7,2021_SWChinaCVM-1.0,Vp & Vs,local,2021.0,"Liu et al., 2021 SRL","https://doi.org/10.1785/0220200318",98.0,106.0,22.0,33.0,0.0,50.0
8,2022_Chen_SChina,Vs,local,2022.0,"Chen et al., 2022 JGR","https://doi.org/10.1029/2022JB024776",108.0,119.0,22.0,35.0,0.0,150.0
9,2022_Kumar_Pamir,Vs,local,2022.0,"Kumar et al., 2022 JGR","https://doi.org/10.1029/2021JB022574",68.0,85.0,28.0,42.0,0.0,100.0
10,2022_SASSY21,Vs,regional,2022.0,"Wehner et al., 2022 JGR","https://doi.org/10.1029/2021JB022930",95.0,135.0,-12.0,10.0,150.0,600.0
11,2022_SinoScope1.0,Vp & Vs,regional,2022.0,"Ma et al., 2022 JGR","https://doi.org/10.1029/2022JB024957",50.0,165.0,-10.0,55.0,0.0,2800.0
12,2022_Toyokuni_SEAsia,Vp,regional,2022.0,"Toyokuni et al., 2022 JGR","https://doi.org/10.1029/2022JB024298",90.0,140.0,-20.0,20.0,0.0,2900.0
13,2022_USTClitho2.0,Vp & Vs,regional,2022.0,"Han et al., 2022 SRL","https://doi.org/10.1785/0220210122",72.0,136.0,20.0,50.0,0.0,150.0
14,2023_Cao_SETibet,Vs,local,2023.0,"Cao et al., 2023 Tectonophysics","https://doi.org/10.1016/j.tecto.2022.229690",100.0,104.5,26.5,32.0,0.0,100.0
15,2023_CSRM1.0,Vp & Vs,regional,2023.0,"Wen and Yu 2023 EPP","https://doi.org/10.26464/epp2023078",80.0,120.0,25.0,50.0,0.0,100.0
16,2023_SWChinaCVM-2.0,Vp & Vs,regional,2023.0,"Liu et al., 2023 SCES","https://doi.org/10.1007/s11430-022-1161-7",99.0,106.0,23.0,33.0,0.0,50.0
17,2023_Wu_NETibet,Vs,local,2023.0,"Wu et al., 2023 JGR","https://doi.org/10.1029/2022JB026109",96.0,109.0,32.0,42.0,0.0,75.0
18,2024_CSES_VM1.0,Vp & Vs,regional,2024.0,"Wu et al., 2024 SCES","https://doi.org/10.1007/s11430-023-1293-4",99.0,106.0,23.0,34.0,0.0,75.0
19,2024_EARA2024,Vp & Vs,regional,2024.0,"Xi et al., 2024 GJI","https://doi.org/10.1093/gji/ggae302",80.0,160.0,10.0,60.0,0.0,1000.0
20,2024_FWEA23,Vp & Vs,regional,2024.0,"Liu et al., 2024 EPSL","https://doi.org/10.1016/j.epsl.2024.118764",70.0,150.0,0.0,54.0,0.0,1000.0
21,2025_Gao_NEChina,Vp,regional,2025.0,"Gao et al., 2025 NC","https://doi.org/10.1038/s41467-025-58053-5",110.5,150.5,30.5,50.5,0.0,2500.0
22,2025_Han_NEChina,Vs,regional,2025.0,"Han et al., 2025 GJI","https://doi.org/10.1093/gji/ggaf070",110.0,150.0,30.0,50.0,0.0,2500.0

Collection continuing...
...
```

## 🚀 开发优先级

### 已完成 ✅

- **数据准备** (1_1~1_5): 台站查询、GCMT 处理、波形下载/预处理、速度模型标准化
- **模型对比** (2_1): 双区域 1D 对比、水平切片、垂直剖面
- **相似性分析** (2_2_Model_similarity): CW-SSIM 多尺度结构相似性、热力图、空间分布
- **模型融合** (4_1, 4_2): Voting Map 投票图、BMA 贝叶斯模型平均
- **SPECFEM3D 设置** (3_1): 正演参数配置
- **可视化** (5_1~5_3): 基础地图、GCMT、台站分布
- **Web 展示**: 纯静态展示界面（速度模型对比、Huang2024、相似性分析）

### Phase 1: 模型空间分析 ⭐ (当前重点)

- [ ] 新建 `2_3_Model_clustering.py`：速度模型聚类分析（K-means 等）
- [ ] 聚类结果可视化（5_Visualization 专项模块）
- [ ] 高级融合算法：SDL、PGM 等

### Phase 2: 数据空间模拟与评估

- [ ] 完善 `3_Data_space_simulation`：模型格式转换、正演计算、批量管理
- [ ] 波形拟合指标计算
- [ ] 模型评估与排序系统

### Phase 3: 系统集成

- [ ] 端到端工作流集成
- [ ] 性能优化与并行化

## 📋 近期目标

**本周任务**:

1. 开发 `2_3_Model_clustering.py`，读取 standardized NetCDF 进行聚类
2. 实现聚类结果可视化（热力图、空间分布）
3. 补充 5_Visualization 聚类/速度模型专项图表

**下周任务**:

1. 推进 SPECFEM3D 批量正演流程（3_2~3_4）
2. 探索 SDL/PGM 等高级融合算法
3. 建立模型评估基准

---

**版本**: v1.0.0
**最后更新**: 2026年3月8日
