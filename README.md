# EASTASIA_FWI

## 🌏 概述

EASTASIA-FWI 东亚地区全波形成像项目，通过模型空间的智能分析（聚类分析和相似性指数分析，或者其他先进算法）和数据空间的波形评估，建立东亚地区三维地球结构初始模型，再结合大规模的高精度全波形反演，得到新一代标准东亚岩石圈模型SEAM 1.0。

**研究区域**: 纬度 -15°至60°，经度 50°至170°  
**核心特色**: 模型聚类分析 + 相似性评估 + 波形模拟评估 + 初始模型融合 + 全波形反演

## 📁 项目结构

```
EASTASIA-FWI/
├── 1_Data_preparation/             # 数据准备模块
│   ├── 1_1_Query_stations.py           # 台站查询与筛选
│   ├── 1_2_Process_GCMT_catalogs.py    # GCMT 事件目录处理
│   ├── 1_3_Download_waveforms.py       # 波形数据下载
│   ├── 1_4_Preprocess_waveforms.py     # 波形预处理
│   ├── 1_5_Process_velocity_models.py  # 速度模型标准化
│   ├── 1_5b_Extract_subregion.py       # 子区域提取
│   ├── 1_6_Split_CSRM_resolution.py    # CSRM 模型分辨率拆分
│   ├── 1_7_Build_test_database.py      # 正演测试数据库构建
│   └── 1_8_Select_quality_stations.py  # 高质量台站筛选
├── 2_Model_space_analysis/         # 模型空间分析
│   ├── 2_1_Model_compare.py            # 模型对比（1D 剖面/水平切片/垂直剖面）
│   ├── 2_2_Model_similarity.py         # CW-SSIM 多尺度结构相似性
│   ├── 2_3_Model_clustering.py         # 速度簇聚类（GMM/WKMeans/HDBSCAN/层次）
│   ├── 2_4_Facies_voting.py            # 相投票
│   ├── 2_3_Model_clustering_methods.md # 聚类方法说明
│   └── cwssim_index/                   # CW-SSIM MATLAB 参考实现
├── 3_Data_space_simulation/        # 数据空间模拟
│   ├── 3_1_Setup_specfem3d_globe.py    # SPECFEM3D Globe 参数设置
│   └── templates/                      # Par_file / STATIONS / CMTSOLUTION 模板
├── 4_Fusion/                       # 初始模型融合
│   ├── 4_1_Voting_map.py               # 投票图（Voting Map）
│   ├── 4_2_BMA.py                      # 贝叶斯模型平均（BMA）
│   └── 4_3_SDL_Fusion_v4.py            # SDL 稀疏字典学习融合（当前主线）
├── 5_Visualization/                # 可视化模块
│   ├── 5_1_Basemap.py                  # 底图
│   ├── 5_2_GCMT.py                     # GCMT 事件
│   ├── 5_3_All_Stations.py             # 全部台站
│   ├── 5_4_Station_Comparison.py       # 台站对比
│   ├── 5_5_Permanent_Stations.py       # 固定台站
│   ├── 5_6_Event_Sampling.py           # 事件采样
│   ├── 5_7_Moho_LAB.py                 # Moho / LAB 界面
│   ├── 5_8_Velocity_Slices.py          # 速度切片
│   ├── 5_9_Geology.py                  # 地质背景
│   ├── 5_10_Clustering_visualization.py# 聚类结果可视化
│   └── utils/                          # 绘图工具
├── config/                         # 配置管理
│   └── base_config.py                  # 全局配置与研究区域定义
├── scripts/                        # 实验与工作流脚本（聚类实验/图表重生成/正演）
├── archive/                        # 归档的历史版本（留档，不参与主管线）
├── mermaid/                        # 技术流程图（.mmd / .md / .svg）
├── web/                            # 纯静态项目展示网站
├── paper/                          # 论文文稿源码（仅 .md/.py 入库，二进制不入库）
├── data/                           # 数据目录（不入库）
├── results/                        # 结果输出（不入库）
├── figures/                        # 图表输出（不入库）
├── output/                         # 正演二进制输出（不入库）
├── logs/                           # 日志文件（不入库）
└── specfem3d_globe/                # SPECFEM3D Globe 源码（需单独安装，不入库）
```

## 🚀 环境要求

```bash
# 创建环境
conda env create -f environment.yml
conda activate eastasia_fwi
```

### SPECFEM3D Globe（正演模拟必需）

项目使用 SPECFEM3D Globe v8.1.0 进行地震波正演。**不纳入 Git 仓库**，需单独安装：

```bash
# 克隆源码（v8.1.0）
git clone --depth 1 --branch v8.1.0 https://github.com/SPECFEM/specfem3d_globe.git specfem3d_globe

# 编译（需 MPI + Fortran 环境）
cd specfem3d_globe && ./configure && make -j
```

详见 [docs/SPECFEM3D_globe_GUIDE.md](docs/SPECFEM3D_globe_GUIDE.md)

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

- **数据准备** (1_1~1_8): 台站查询、GCMT 处理、波形下载/预处理、速度模型标准化、子区域提取、CSRM 分辨率拆分、测试数据库构建、质量台站筛选
- **模型对比** (2_1): 双区域 1D 对比、水平切片、垂直剖面
- **相似性分析** (2_2): CW-SSIM 多尺度结构相似性、热力图、空间分布
- **速度簇聚类** (2_3): GMM / WKMeans / HDBSCAN / 层次聚类、Moho 深度分带、HMRF 空间正则化、K 扫描稳健性
- **相投票** (2_4): Facies voting
- **模型融合** (4_1, 4_2, 4_3): Voting Map 投票图、BMA 贝叶斯模型平均、SDL 稀疏字典学习融合（v4 主线）
- **SPECFEM3D 设置** (3_1): SPECFEM3D Globe 正演参数配置
- **可视化** (5_1~5_10): 底图、GCMT、台站、事件采样、Moho/LAB、速度切片、地质背景、聚类结果
- **Web 展示**: 纯静态展示界面（速度模型对比、Huang2024、相似性分析）
- **论文工作流** (paper/): 《东亚地震学模型对比与检验》文稿源码与构建脚本

### Phase 1: 模型空间分析与融合 ⭐ (当前重点)

- [x] `2_3_Model_clustering.py`：速度簇聚类分析（GMM/WKMeans/HDBSCAN/层次）
- [x] 聚类结果可视化（5_10 专项模块）
- [ ] 高级融合算法完善：SDL v4 全深度验证、PGM 等

### Phase 2: 数据空间模拟与评估

- [ ] 完善 `3_Data_space_simulation`：模型格式转换（GLL）、批量正演管理
- [ ] 波形拟合指标计算
- [ ] 模型评估与排序系统

### Phase 3: 系统集成

- [ ] 端到端工作流集成
- [ ] 性能优化与并行化
- [ ] 单元测试与 CI（环境已含 pytest，测试待补）

## 📋 近期目标

1. 推进 SPECFEM3D 批量正演流程（模型格式转换 + 批量管理）
2. SDL v4 融合结果的全深度验证与 Paper I 图表产出
3. 建立模型评估基准（波形拟合指标 + 模型排序）

---

**版本**: v1.0.0
**最后更新**: 2026年9月9日
