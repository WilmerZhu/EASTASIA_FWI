# 稀疏字典学习(SDL)速度模型融合模块 - 完整开发设计文档

> **文档类型**: 主开发文档（记录全部理论方法和实现架构）  
> **创建日期**: 2026-01-08  
> **文档版本**: v3.4.2  
> **当前代码版本**: v3.4.2  
> **配套文档**: `plan-sdlFusion.prompt.md`（迭代开发计划和测试评估）  
> **代码文件**: `4_Assembly/4_3_SDL_Fusion_v3.4.py`

---

## 📚 目录

1. [项目概述](#1-项目概述)
2. [论文方法深度解读](#2-论文方法深度解读)
   - 2.9 [同分辨率 SDL：模型风格迁移](#29-同分辨率-sdl：模型风格迁移)
3. [东亚三模型数据分析](#3-东亚三模型数据分析)
4. [完整处理流程架构](#4-完整处理流程架构)
5. [Phase -1: 数据预处理与 Patch 提取](#5-phase--1-数据预处理与-patch-提取)
6. [Phase 0: 字典表示能力验证](#6-phase-0-字典表示能力验证)
7. [Phase H: 超参数联合网格搜索](#7-phase-h-超参数联合网格搜索)
8. [Phase 1: 2D Patch 级变换验证](#8-phase-1-2d-patch级变换验证)
9. [Phase 1.5: 完整 2D 切片变换](#9-phase-15-完整2d切片变换)
10. [Phase 2: 3D Patch 融合](#10-phase-2-3d-patch融合)
11. [Phase 3: 全域模型增强](#11-phase-3-全域模型增强)
12. [Patches 拼接方法详解](#12-patches拼接方法详解)
13. [全流程可视化体系](#13-全流程可视化体系)
14. [代码架构设计](#14-代码架构设计)
15. [核心数据类定义](#15-核心数据类定义)
16. [配置参数详解](#16-配置参数详解)
17. [公式速查表](#17-公式速查表)
18. [参考文献](#18-参考文献)

---

## 1. 项目概述

### 1.1 目标

使用稀疏字典学习（Sparse Dictionary Learning, SDL）方法，将高分辨率地震速度模型（EARA2024/FWEA23）的特征迁移融合到区域模型（SinoScope1.0），生成增强后的东亚岩石圈速度模型。

### 1.2 核心价值

与简单嵌入方法相比，SDL 方法的优势：

| 方面       | 简单嵌入方法               | SDL 方法                               |
| ---------- | -------------------------- | -------------------------------------- |
| 边界处理   | 依赖人为的平滑参数         | 自动学习过渡                           |
| 信息保留   | 丢失低分辨率模型的有用信息 | 融合两种模型的信息                     |
| 增强范围   | 仅更新重叠区域             | **全域增强（包括重叠区外部）**         |
| 风格迁移   | 无                         | 将高分辨率模型的"风格"迁移到低分辨率域 |
| 参数依赖性 | 高（人工设定）             | 低（数据驱动）                         |

### 1.3 论文核心创新

论文的三个核心贡献：

1. **建立分辨率间的变换关系**：

   > _"we develop a method for merging multi-scale velocity models by establishing a **transformation** between models of different resolutions"_

2. **利用共享稀疏系数约束**：

   - 同一物理位置的 patches 使用相同的稀疏系数 $\mathbf{C}$
   - 这建立了 D₁ 和 D₂ 原子之间的**一一对应关系**

3. **全域增强能力**：
   > _"enhance the low-resolution model **both in the overlapping region as well as outside it**"_
   - 变换在重叠区域学习
   - 但可以应用到 M₁ 的**整个覆盖范围**

### 1.4 论文验证结论

论文通过多个实验验证了方法的有效性：

| 实验          | 关键发现                                  |
| ------------- | ----------------------------------------- |
| 字典重建      | 平均误差 2.2%，证明字典表示能力           |
| 2D 变换       | 改善率 ~34%，重建模型更平滑、与目标更一致 |
| 3D Vp+Vs 变换 | 改善率 ~58%，误差分布正态(μ=0.4%, σ=3%)   |
| 波形验证      | 重建模型的合成波形与目标模型波形更一致    |

> _"The reconstructed model is **smoother** than CVM-S4.26 in the entire domain and **more consistent** with the Z2015 model in the common region."_

### 1.5 处理流程总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SDL Fusion v3.4 完整处理流程                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  原始模型数据:                                                           │
│    ├── SinoScope1.0 (1°×1°×20km) → 基础模型 M₁                          │
│    ├── EARA2024 (0.25°×0.25°×10km) → 高分辨率模型 M₂                    │
│    └── FWEA23 (0.25°×0.25°×10km) → 高分辨率模型 M₃                      │
│                          ↓                                               │
│  Phase -1: 数据预处理与 Patch 提取                                       │
│    └── DirectPatchExtractor: 直接从原始网格提取配对 Patches ⭐           │
│                          ↓                                               │
│  Phase 0: 字典表示能力验证                                               │
│    └── 验证 SDL 能准确重建各模型 (误差<3%)                               │
│                          ↓                                               │
│          ┌───────────────┴───────────────┐                               │
│          │                               │                               │
│          ↓                               ↓                               │
│  ┌─────────────────┐           ┌─────────────────┐                       │
│  │ 模式 A: 代码配置 │           │ 模式 B: 网格搜索 │                       │
│  │ 直接使用代码中   │           │ Phase H 超参数  │                       │
│  │ 配置的参数      │           │ 联合网格搜索    │                       │
│  └────────┬────────┘           └────────┬────────┘                       │
│           │                             │                                │
│           │    ┌────────────────────────┘                                │
│           │    │ 输出: optimal_config.json                               │
│           ↓    ↓                                                         │
│  Phase 1: 2D Patch 级变换验证                                            │
│    └── 训练耦合字典 D₁, D₂（所有深度独立训练）                           │
│                          ↓                                               │
│  Phase 1.5: 完整 2D 切片变换 (核心)                        │
│    └── transform_full_slice_direct(): 直接使用原始模型数据              │
│        (修复训练/变换一致性问题)                                         │
│                          ↓                                               │
│  Phase 2: 3D Patch 融合 (待实现)                                         │
│    └── 利用深度相关性，3D 耦合字典变换                                   │
│                          ↓                                               │
│  Phase 3: 全域模型增强 (待实现)                                          │
│    └── 输出增强后的完整 NetCDF 模型                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**v3.4 运行模式**:

| 模式       | 说明                   | 适用场景           | 运行方式           |
| ---------- | ---------------------- | ------------------ | ------------------ |
| **模式 A** | 直接在代码中指定超参数 | 快速测试、确定参数 | 默认运行（无参数） |
| **模式 B** | K × λ 联合网格搜索     | 探索最优参数组合   | `--grid-search`    |
| **模式 C** | 使用已有最优配置       | 生产运行、重复实验 | `--use-optimal`    |
| **快速**   | 只训练 4 个代表性深度  | 调试、初步验证     | `--quick`          |

> ⭐ v3.4: 移除统一网格方法，只保留直接提取方法（论文方法）

---

## 2. 论文方法深度解读

### 2.1 核心参考论文

**Zhang, H., & Ben-Zion, Y. (2024)**: _Enhancing Regional Seismic Velocity Models With Higher-Resolution Local Results Using Sparse Dictionary Learning_. Journal of Geophysical Research: Solid Earth, 129, e2023JB027016.

### 2.2 研究背景与动机

#### 2.2.1 多尺度速度模型融合的需求

论文 Introduction 指出：

> _"The spatial coverage and resolution of derived velocity models depend on the used data and inversion methods, and models of widely different scales and resolutions are often available in well-studied regions."_

**南加州的多尺度模型示例**：

| 模型类型 | 分辨率 | 数据来源     | 覆盖范围 |
| -------- | ------ | ------------ | -------- |
| 区域模型 | ~5 km  | 走时反演     | 全区域   |
| 断层模型 | 2-3 km | 密集台阵     | 断层周围 |
| 局部模型 | 数十米 | DAS/密集台阵 | 局部区域 |

**融合多尺度模型的价值**：

> _"Developing methods to merge velocity models of different scales and resolutions can provide enhanced multi-scale frameworks that benefit from the complementary strengths of the different results."_

#### 2.2.2 为什么选择字典学习而不是 CNN

论文明确解释了选择字典学习的原因：

> _"Since the available amount of data is limited, we opt for this approach over Convolutional Neural Networks (CNN) that demand a large number of trainable parameters."_

| 方法 | 优势                                   | 劣势                 |
| ---- | -------------------------------------- | -------------------- |
| CNN  | 强大的特征学习能力                     | 需要大量数据和参数   |
| SDL  | **physically interpretable**（可解释） | 特征学习能力相对有限 |
|      | **better generalization**（泛化更好）  |                      |
|      | 数据需求少                             |                      |

**字典学习在地震学中的应用**：

- 信号去噪 (Beckouche & Ma, 2014)
- 地震层析成像 (Bianco & Gerstoft, 2018)

### 2.3 问题定义

> _"The most straightforward approach to combine models is to embed small-scale high-resolution velocity models within regional models and smooth the boundaries. While this approach is simple, the obtained seismic velocities near the boundaries between models depend on ad-hoc choices of the applied smoothing parameters... such procedures disregard useful information from the low-resolution results in the area covered by the high-resolution model."_

**简单嵌入方法的三个问题**:

1. **边界处理**: 依赖 ad-hoc（人为的、临时性的、无理论依据的）平滑参数
2. **信息丢失**: 在高分辨率覆盖区域内丢失低分辨率模型的有用信息
3. **局限性**: 只更新重叠区域，不增强外部区域

**论文方法的核心创新**：

> _"we develop a method for merging multi-scale velocity models by establishing a transformation between models of different resolutions. Specifically, we develop a transformation by comparing high- and low-resolution imaging results in a given region and then utilize this transformation to enhance the low-resolution model **both in the overlapping region as well as outside it**."_

关键点：**变换可以应用到重叠区域之外**，这是与简单嵌入方法的本质区别。

### 2.4 字典表示 (Dictionary Representation)

#### 2.4.1 核心概念 - Patch 提取

论文 Section 2.1 详细描述了 Patch 提取过程：

> _"Given a seismic velocity model M, we use a sliding window which has the same dimension as the model to extract patches from it, resulting in N patches p₁, ..., pₙ that represent values inside subsections of the model."_

**Patch 向量化**：

- 2D 窗口 $3 \times 3$ → $L = 9$
- 2D 窗口 $6 \times 6$ → $L = 36$
- 3D 窗口 $9 \times 9 \times 20$ → $L = 1620$

所有 flattened patches 组成矩阵：$\mathbf{P} \in \mathbb{R}^{L \times N}$

#### 2.4.2 稀疏表示公式

每个速度 patch 可以用字典原子的**稀疏**线性组合表示：

$$\mathbf{p}_j \approx \mathbf{D}^* \mathbf{c}_j = \sum_{i=1}^{K} c_{i,j} \mathbf{d}_i \tag{公式1}$$

其中：

- $\mathbf{p}_j \in \mathbb{R}^L$: 第 j 个 patch（向量化后）
- $\mathbf{D}^* = [\mathbf{d}_1, ..., \mathbf{d}_K] \in \mathbb{R}^{L \times K}$: 字典矩阵（K 个原子）
- $\mathbf{c}_j = [c_{1,j}, ..., c_{K,j}]^T \in \mathbb{R}^K$: 稀疏系数向量（大部分为 0）
- $c_{i,j}$: patch $\mathbf{p}_j$ 在原子 $\mathbf{d}_i$ 上的投影系数

**关键**：字典原子数 $K < N$（通常远小于样本数），且系数向量 $\mathbf{c}_j$ 是**稀疏的**（very few nonzero entries）。

#### 2.4.3 矩阵形式

$$\mathbf{P} \approx \mathbf{D}^* \mathbf{C}^* \tag{公式2}$$

其中 $\mathbf{C}^* = [\mathbf{c}_1, ..., \mathbf{c}_N] \in \mathbb{R}^{K \times N}$ 是稀疏系数矩阵。

#### 2.4.4 优化目标

$$\mathbf{D}^*, \mathbf{C}^* = \arg\min_{\mathbf{D}, \mathbf{C}} \|\mathbf{P} - \mathbf{D}\mathbf{C}\|_2^2 + \lambda\|\mathbf{C}\|_0 \tag{公式3}$$

其中：

- $\|\mathbf{P} - \mathbf{D}\mathbf{C}\|_2^2$: 重建误差（Frobenius 范数）
- $\lambda\|\mathbf{C}\|_0$: 稀疏性惩罚（L0 范数 = 非零元素个数）
- $\lambda$: 稀疏性控制参数，**controls the tradeoff between the sparsity and the minimization error**

#### 2.4.5 稀疏编码算法

论文提到两种常用算法：

| 算法  | 全称                                       | 特点                     |
| ----- | ------------------------------------------ | ------------------------ |
| OMP   | Orthogonal Matching Pursuit                | 贪婪算法，直接控制非零数 |
| LASSO | Least Absolute Shrinkage and Selection Op. | L1 正则化，连续优化      |

> _"Various algorithms, such as Orthogonal Matching Pursuit (OMP) (Mallat & Zhang, 1993) and Least Absolute Shrinkage and Selection Operator (LASSO) (Tibshirani, 1996), can be used to solve this optimization problem."_

**当前实现使用 OMP**（`transform_algorithm='omp'`），更直接地控制稀疏度。

#### 2.4.6 Patch 拼接重建

> _"After obtaining the dictionary D* and corresponding representation matrix C* using the sparse coding algorithm, the reconstructed patches is calculated as P̂ = D*C*. The updated velocity model can be then developed by **stitching the reconstructed patches together in their original spatial order** (Dong et al., 2015)."_

这里引用了 Dong et al. (2015) 的 SRCNN 论文的 patch 拼接方法。

### 2.5 耦合字典学习 (Coupled Dictionary Learning)

#### 2.5.1 问题设定

论文 Section 2.2 详细描述了两个模型之间的变换问题：

> _"Suppose we have an initial model M₁ and a second more local model M₂ with a higher-resolution that is **spatially included in M₁**. Our goal is to transform M₁ to model M̂, with resolution and values in the common region of both models consistent with M₂ and enhanced resolution outside the overlapping region."_

**关键概念**：

- $\mathbf{P}_1^U = [\mathbf{p}_1^1, ..., \mathbf{p}_1^{N_1}] \in \mathbb{R}^{L_1 \times N_1}$: 从**整个** M₁ 提取的 patches
- $\mathbf{P}_1 = [\mathbf{p}_1^1, ..., \mathbf{p}_1^{N_2}] \in \mathbb{R}^{L_1 \times N_2}$: P₁^U 的**子集**，与 P₂ 在空间上对应
- $\mathbf{P}_2 = [\mathbf{p}_2^1, ..., \mathbf{p}_2^{N_2}] \in \mathbb{R}^{L_2 \times N_2}$: 从 M₂ 提取的 patches

**空间对应约束**：

> _"Such pairs of patches, for example, pᵢ¹ and pᵢ², extracted from the **same spatial locations** are aligned by their index in P₁ and P₂."_

#### 2.5.2 核心 - 共享稀疏系数

**同一物理位置的 patches 共享相同的稀疏系数** → 这是变换的核心约束

```
低分辨率模型 M₁:  P₁ ≈ D₁ C    (公式4)
高分辨率模型 M₂:  P₂ ≈ D₂ C    (公式5)
                      ↑
              共享相同的稀疏系数 C
```

> _"The atom d²ᵢ in D₂ have **one-to-one corresponding** to d¹ᵢ in D₁ due to the duality between P₁ and P₂."_

#### 2.5.3 训练步骤

论文将变换分为三步（Section 2.2）：

**第一步**：从整个 M₁ 提取 patches P₁^U，其子集 P₁ 与 P₂ 在空间上对应

**第二步**：将 patches 从 P₁^U 变换到 P̂

**第三步**：将变换后的 patches P̂ 放回原始空间位置，重建变换后的模型 M̂

#### 2.5.3.1 深度处理方式 ⭐ 关键

**论文处理的是 2D 横向速度切片，每个深度层独立处理**：

> _"The method is applied independently to each depth layer"_ > _"the target model M₂ is taken from velocities of the F2019 model **at the same depths**"_

这意味着：

- **每个深度层**都有独立的字典对 (D₁, D₂)
- **不同深度的字典不共享、不复用**
- **⭐ v3.4.2: 两个模型必须有精确匹配的深度**（论文要求 "at the same depths"）

**东亚场景深度匹配**：

| 模型            | 深度分辨率 | 深度层                                  |
| --------------- | ---------- | --------------------------------------- |
| SinoScope1.0    | 20 km      | 0, 20, 40, 60, 80, ...                  |
| FWEA23/EARA2024 | 10 km      | 0, 10, 20, 30, 40, ...                  |
| **匹配深度**    | -          | 0, 20, 40, 60, 80, 100, ... (20km 间隔) |

```python
# v3.4.2: 只使用深度精确匹配的层
matched_depths = get_matched_depths(base_depths, highres_depths, tolerance=0.1)
# 结果: [0, 20, 40, 60, 80, 100, 120, ...]  # 共约 51 层
```

> ✅ **v3.4.2 正确做法**：只使用 SinoScope 和高分辨率模型深度精确匹配的层进行训练和变换。

#### 2.5.4 训练/验证集划分

> _"In practice, we randomly divide the patches P₁ and P₂ into training set P₁ᵀ, P₂ᵀ and validation set P₁ᵛ, P₂ᵛ, and put each pair of patches with the same location in the same data set."_

- 训练集用于学习字典和稀疏系数
- 验证集用于选择最优超参数

#### 2.5.5 训练 D₁ 和 C（公式 6）

$$D_1, C^T = \arg\min_{D_1, C} \|P_1^T - D_1 C\|_2^2 + \lambda\|C\|_0 \tag{公式6}$$

使用 sklearn 的 `DictionaryLearning`:

```python
# ⭐ v3.4.2: 直接使用原始 patches，不做 centering（与论文一致）
dict_learner = DictionaryLearning(
    n_components=K,      # K=20 原子
    alpha=λ,             # λ=0.1 稀疏性
    transform_algorithm='omp',
    transform_n_nonzero_coefs=5
)
dict_learner.fit(P1_train)  # 直接使用原始数据，不减均值
D1 = dict_learner.components_  # (K, L₁)
C_train = dict_learner.transform(P1_train)  # (N, K)
```

> ⭐ **v3.4.2 重要更新**: 移除了 centering 预处理，与论文原始方法一致。论文公式中没有明确的减均值步骤。

#### 2.5.6 计算 D₂（公式 7）⭐ 关键差异

**论文原始公式 (纯最小二乘，无正则化)**:

$$D_2 = \arg\min_{D_2} \|P_2^T - D C^T\|_2^2 \tag{公式7}$$

论文明确使用 **Least Squares Method**：

> _"For the derived sparse representation Cᵀ, we calculate the corresponding dictionary D₂ = argmin ‖P₂ᵀ - DCᵀ‖₂² from P₁ᵀ using the **Least Squares Method**."_

```python
# 论文方法（纯最小二乘）⭐ v3.4.2: 无 centering
D2 = np.linalg.lstsq(C_train, P2_train, rcond=None)[0]  # (K, L₂)
```

**当前可选实现 (带 L2 正则化)**:

```python
# 岭回归版本（可通过 d2_regularization > 0 启用）
ridge = Ridge(alpha=0.1, fit_intercept=False)
ridge.fit(C_train, P2_train_centered)
D2 = ridge.coef_.T  # (K, L₂)
```

> ✅ **v3.2 已修正**: 默认使用纯最小二乘（`d2_regularization=0`），与论文一致。

#### 2.5.7 验证集评估（公式 8-9）

**验证集稀疏编码**：

$$C^V = \arg\min_C \|P_1^V - D_1 C\|_2^2 + \lambda\|C\|_0 \tag{公式8}$$

**变换结果**：$\hat{P}^V = D_2 C^V$

**相对误差（Baseline）**：

$$D(\mathbf{P}, \mathbf{P}_0) = \frac{\|\mathbf{P} - \mathbf{P}_0\|_2}{\|\mathbf{P}_0\|_2} \tag{公式9}$$

#### 2.5.8 超参数优化

> _"The optimal hyper-parameters in the training procedure (e.g., the K number of atoms in a dictionary, the patch size, and the sparsity control parameter λ) are found by minimizing the difference D(P̂ᵛ, P₂ᵛ)."_

论文通过**网格搜索 40 种组合**来确定最优参数。

### 2.6 变换流程

#### 2.6.1 三步变换

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: 提取配对 patches                                             │
├─────────────────────────────────────────────────────────────────────┤
│   从 M₁ 和 M₂ 的重叠区域提取配对 patches                             │
│   物理对应约束: (p₁ᵢ, p₂ᵢ) 覆盖完全相同的物理区域                    │
│   CC 筛选: 只保留 CC > 0 的配对（排除异常值）                        │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: 训练耦合字典 D₁ 和 D₂                                        │
├─────────────────────────────────────────────────────────────────────┤
│   划分: 训练集 80%, 验证集 20%                                       │
│   训练 D₁: 从 P₁ᵀ 学习字典和稀疏系数 C                              │
│   计算 D₂: 使用 C 和 P₂ᵀ 求解                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: 对 M₁ 全域进行变换（核心！）                                  │
├─────────────────────────────────────────────────────────────────────┤
│   对于 M₁ 全域的每个 patch p₁:                                       │
│     1. 编码: c = transform(p₁ - mean_P1) using D₁                   │
│     2. 解码: p̂₂ = D₂ @ c + mean_P2                                  │
│     3. 拼接: 使用加权平均处理重叠区域                                │
└─────────────────────────────────────────────────────────────────────┘
```

#### 2.6.2 训练区域 vs 应用区域

```
┌─────────────────────────────────────────────────────────────────────┐
│                    M₁ 全域（SinoScope1.0 待增强）                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │    ┌──────────────────────┐                                 │    │
│  │    │   M₂ 有数据区域        │  ← 训练数据来源                  │    │
│  │    │   (FWEA23 部分覆盖)   │     从这里提取配对 patches       │    │
│  │    └──────────────────────┘                                 │    │
│  │                                                              │    │
│  │    ← 变换应用到 M₁ 全域（包括 M₂ 没有覆盖的区域！）                    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘

论文原文 (Section 2.2):
  "extract patches P₁^U from the ENTIRE M₁"
  "A SUBSET of P₁^U, P₁, shares spatial locations with patches P₂"

关键点:
- 训练: 只在 M₂ 有数据的区域提取配对 patches
- 应用: 对 M₁ 全域的所有 patches 进行变换
- 核心价值: M₂ 没有覆盖的区域也获得"高分辨率风格"的增强！
```

### 2.7 误差度量

**相对误差（论文公式 9）**:

$$D(\mathbf{P}, \mathbf{P}_0) = \frac{\|\mathbf{P} - \mathbf{P}_0\|_2}{\|\mathbf{P}_0\|_2} \times 100\% \tag{公式9}$$

**相对改善率**:

$$\text{Improvement} = \frac{\text{Baseline Error} - \text{SDL Error}}{\text{Baseline Error}} \times 100\%$$

### 2.8 论文关键参数详解

#### 2.8.1 论文 2D 实验参数 (Section 3.2)

| 参数      | 论文值            | 说明                                           |
| --------- | ----------------- | ---------------------------------------------- |
| M₁ 分辨率 | 0.03°×0.03°       | CVM-S4.26 模型                                 |
| M₂ 分辨率 | 0.01°×0.01°       | Z2015 模型（M₂ 覆盖范围 < M₁）                 |
| 分辨率比  | 3:1               | 0.03/0.01 = 3                                  |
| M₁ 窗口   | 6×6 格点          | 物理大小 = 6×0.03° = **0.18°**                 |
| M₂ 窗口   | 18×18 格点        | 物理大小 = 18×0.01° = 0.18°                    |
| M₁ 步长   | **stride=1 格点** | 物理步长 = 1×0.03° = **0.03°**（非常密集）     |
| M₂ 步长   | **stride=3 格点** | 物理步长 = 3×0.01° = **0.03°**（保持物理对应） |

> ⚠️ **重要**: 论文中 stride 定义为**格点数**（grid points），不是度！
> _"a stride (amount of **grid points** of window's movement over the model in one direction)"_
> | 配对数 | 1,131 对（CC>0 筛选后） | 从 1,212 对中筛选 |
> | 训练/验证 | 1,000 / 131 | 约 88% / 12% |
> | K | 20 | 字典原子数 |
> | λ | 0.1 | 稀疏性参数 |
> | **D₂ 计算** | **纯最小二乘（无正则化）** | $D_2 = \arg\min \|P_2^T - D_2 C^T\|_2^2$ |

**📐 stride 定义辨析 (重要！)**

| 概念                | 论文定义                 | 我们代码          | 换算关系          |
| ------------------- | ------------------------ | ----------------- | ----------------- |
| **stride**          | **格点数** (grid points) | -                 | 论文原义          |
| **physical_stride** | -                        | **物理步长 (度)** | = stride × 分辨率 |

```
论文公式: physical_stride = stride (格点数) × resolution (度/格点)

论文 M₁: stride=1 格点 × 0.03°/格点 = 0.03° 物理步长
论文 M₂: stride=3 格点 × 0.01°/格点 = 0.03° 物理步长 ← 物理步长相同！

东亚 M₁: stride=1 格点 × 1°/格点    = 1° 物理步长
东亚 M₂: stride=4 格点 × 0.25°/格点 = 1° 物理步长 ← 物理步长相同！
```

#### 2.8.2 论文 3D 实验参数 (Section 3.3)

| 参数       | 论文值                  | 说明                                          |
| ---------- | ----------------------- | --------------------------------------------- |
| M₁ 分辨率  | 0.09°×0.09°× 层         | CVM-S4.26（1-10km，10 层）                    |
| M₂ 分辨率  | 0.03°×0.03°× 层         | F2019 模型                                    |
| **预处理** | **双三次插值上采样 M₁** | "employ bicubic interpolation to upsample M₁" |
| 3D 窗口    | **9×9×20**              | 9×9 水平格点，20 层深度                       |
| 配对数     | 980 对                  |                                               |
| 训练       | 800 对                  |                                               |
| K          | 20                      | 每个原子维度 = 9×9×20 = 1620                  |

**论文关键发现**:

> "because velocities at different depths are not entirely independent and are likely to include correlated variations. Utilizing 3D reconstruction ensures continuity and smoothness in depth"

#### 2.8.3 论文实验结果详解

##### Section 3.1: 字典表示能力验证

| 参数             | 值                     |
| ---------------- | ---------------------- |
| 模型             | CVM-S4.26              |
| 深度             | 2 km（$V_P$ 水平切片） |
| 分辨率           | 0.09° × 0.09°          |
| 网格尺寸         | 30 × 24 格点           |
| **窗口大小**     | **3 × 3**              |
| **步长**         | **stride=1**           |
| **提取 Patches** | **616 个**             |
| K                | 20                     |
| λ                | 0.1                    |
| **平均误差**     | **2.2%**               |
| 最大点误差       | 7%                     |

> _"The average error between the reconstruction and the original model is **2.2%**, while the peak error at a single grid is 7%, demonstrating accurate reconstruction using the dictionary representation."_

##### Section 3.2: 2D 变换实验

**CC 筛选机制**：

> _"To have efficient training of dictionaries, we compute the **cross-correlation (CC)** between upsampled low-resolution patches and their corresponding high-resolution patches. We only utilize patch pairs with **CC > 0** for both training and validation. This screening step **excludes outlier model values** and ensures an efficient convergence of the training process."_

**原子纹理一致性**：

> _"Notably, there is a **coherence between the value ranges of atom pairs** in D₁ and D₂. This similarity does not appear explicitly in the loss function, but is rather **implicitly required by the design of the algorithm**. However, textures of atom pairs are **not entirely consistent**, allowing the transformation of the model's style."_

**重建模型特征**：

> _"The reconstructed model is smoother than CVM-S4.26 in the entire domain and more consistent with the Z2015 model in the common region."_

##### Section 3.3: 3D 变换实验 (Vp + Vs)

**3D 窗口设计理由**：

> _"We employ a 3D window because **velocities at different depths are not entirely independent** and are likely to include **correlated variations**. Utilizing 3D reconstruction ensures **continuity and smoothness in depth** for the reconstructed model."_

**视频帧类比**：

> _"This approach is similar to applying super-resolution techniques to videos, where not only individual frames are considered but also **information from neighboring frames** is utilized."_

##### 误差结果汇总

| 实验类型 | 训练误差 | 验证误差 | Baseline | 改善率 |
| -------- | -------- | -------- | -------- | ------ |
| 2D Vs    | 6.1%     | 7.6%     | 11.6%    | ~34%   |
| 3D Vp+Vs | 2.94%    | 3.41%    | 8.12%    | ~58%   |

**误差分布特征** (3D 结果):

> _"The differences follow a **normal distribution** with a **mean of 0.4%** and a **variance of 3%**, indicating that the reconstruction model **effectively and unbiasedly** updates the regional model."_

- 均值: 0.4%
- 方差: 3%
- 分布: 正态（无偏更新）

#### 2.8.4 论文附录图分析

##### Figure S1: 字典原子可视化

论文 Supplementary Material Figure S1 展示了 2D 变换实验中的 D₁ 和 D₂ 字典原子：

| 属性                 | D₁ (低分辨率) | D₂ (高分辨率)    |
| -------------------- | ------------- | ---------------- |
| **形状**             | 6×6 = **36**  | 18×18 = **324**  |
| **分辨率比**         | \-            | 3:1              |
| **原子数 K**         | 20            | 20               |
| **D₁-D₂ 原子相关性** | \-            | **-0.28 ~ 0.83** |

**关键发现**：

- 对应原子的相关性**差异很大**（从负相关到强正相关）
- 这说明 D₁ 和 D₂ 原子**不需要完全一致**
- 低分辨率和高分辨率的纹理模式可以不同 → **允许风格变换**

> _"Notably, there is a coherence between the **value ranges** of atom pairs in D₁ and D₂... However, **textures of atom pairs are not entirely consistent**, allowing the transformation of the model's style."_

##### Figure S2 & S3: 重建效果散点图

论文 Figure S2 和 S3 展示了变换效果的空间分析：

**散点图设计**：

- **X 轴**: M₁ 值 或 (M₂ - M₁) 差异
- **Y 轴**: (Reconstruction - M₁) 重建变化
- **黑点**: 重叠区域内的格点
- **红点**: 重叠区域外的格点

**核心发现** (Figure S2b, S3c, S3d):

> 重建差异 (Reconstruction - M₁) 与模型差异 (M₂ - M₁) **正相关**

这证明了 SDL 的核心功能：**成功地将 M₁ 向 M₂ 方向调整**！

```
关键观察:
- 当 M₂ > M₁ 时，Reconstruction > M₁ （速度被增大）
- 当 M₂ < M₁ 时，Reconstruction < M₁ （速度被减小）
- 重叠区域外（红点）也显示出类似趋势 → 证明全域增强有效
```

##### Figure S4: 误差分布直方图最重要

论文 Figure S4 展示了 3D 变换的误差分布：

| 指标         | SDL 重建 (蓝色) | Baseline (红色) |
| ------------ | --------------- | --------------- |
| **均值**     | **0.4%**        | 偏负 (~-5%)     |
| **标准差**   | **~3%**         | ~15-20%         |
| **分布**     | **正态**        | 宽且偏斜        |
| **峰值位置** | 0 附近          | 负值区域        |

**关键结论**：

> _"The differences follow a **normal distribution** with a **mean of 0.4%** and a **variance of 3%**, indicating that the reconstruction model **effectively and unbiasedly** updates the regional model."_

**误差无偏性的科学意义**：

1. 均值接近 0 → SDL 不会系统性地高估或低估速度
2. 标准差小 → 变换结果稳定可靠
3. 正态分布 → 误差是随机的，不存在系统性偏差

#### 2.8.5 与当前实现的差异

**⭐ 当前测试对: USTClitho2.0 (M₁, 0.5°) → CSES_VM1.0 (M₂, 0.25°)**

| 方面          | 论文场景 (2D)               | USTClitho→CSES (当前 2D)        | 差异与说明                       |
| ------------- | --------------------------- | -------------------------------- | -------------------------------- |
| **分辨率比**  | **3:1**                     | **2:1**                          | 比论文更接近 1:1，超分辨率效果稍弱 |
| **物理窗口**  | **0.18°**                   | **3.0°**                         | 窗口大 17 倍（受覆盖范围限制）   |
| **M₁ stride** | **1 格点** (物理步长 0.03°) | **1 格点** (物理步长 0.5°)       | 重叠率相同 83% ✅                |
| **M₂ stride** | **3 格点** (物理步长 0.03°) | **2 格点** (物理步长 0.5°)       | 物理步长一致 ✅                  |
| **D₂ 正则化** | **无**                      | **无**                           | ✅ 一致                          |
| CC 阈值       | >0                          | >0                               | ✅ 一致                          |
| K, λ          | 20, 0.1                     | 20, 0.1                          | ✅ 一致                          |
| 匹配深度      | 论文均匀 10 层              | **9 层**（不规则间距）           | 深度插值影响 3D SDL              |
| Patches/深度  | ~1,131 对                   | **~220-236 对**                  | ⚠️ 偏少，建议 3D 滑窗扩充        |
| 超参数优化    | 网格搜索 40 组合            | Phase H 支持                     | ✅ 已实现框架                    |

**USTClitho→CSES 测试结果** (2026-06-11):

| 深度  | SDL 误差 | Baseline | 改善率         | 状态 |
| ----- | -------- | -------- | -------------- | ---- |
| 30km  | 1.98%    | 3.59%    | **44.8%** ⭐   | ✅   |
| 60km  | —        | —        | —（待补充）    | ⏳   |
| 100km | —        | —        | —（待补充）    | ⏳   |

**历史参考: SinoScope→EARA2024 测试结果** (2026-01-09):

| 深度     | SDL 误差 | Baseline | 改善率       | 误差均值 |
| -------- | -------- | -------- | ------------ | -------- |
| 40km     | 5.03%    | 8.53%    | **41.1%** ⭐ | -0.1%    |
| 100km    | 2.40%    | 4.07%    | **41.0%** ⭐ | +0.5%    |
| 500km    | ~1.0%    | ~2.2%    | **~54%**     | ~0%      |
| 800km    | ~0.8%    | ~2.0%    | **~61%**     | ~0%      |

---

### 2.9 同分辨率 SDL：模型风格迁移

#### 2.9.1 问题提出

论文方法 (Section 2.2) 默认 M₁ 分辨率低于 M₂（超分辨率场景）。但 SDL 框架对**相同分辨率**的两个模型同样有效，此时方法的含义从「超分辨率增强」转变为「模型风格迁移 / 系统性偏差修正」。

**适用场景**：

| 模型对                       | 分辨率        | 场景描述                              |
| ---------------------------- | ------------- | ------------------------------------- |
| EARA2024 ↔ FWEA23            | 两者均 0.25°  | 两个 FWI 模型间风格迁移               |
| USTClitho2.0 ↔ CSRM1.0       | 均约 0.5°     | 同分辨率岩石圈层析模型间偏差修正      |
| SinoScope1.0 ↔ FWEA18        | 均约 1°       | 跨数据集的系统性速度偏差学习          |

#### 2.9.2 数学框架

当 Δ₁ = Δ₂ = Δ（相同分辨率）时，数学框架完全不变：

$$D_1, C^T = \arg\min_{D_1, C} \|P_1^T - D_1 C\|_F^2 + \lambda\|C\|_0 \tag{公式6}$$

$$D_2 = \arg\min_{D_2} \|P_2^T - D_2 C^T\|_F^2 \tag{公式7}$$

**唯一变化**：L₁ = L₂ = L = n²（patch 尺寸相同），D₁, D₂ ∈ ℝ^{K×L} 形状一致。

#### 2.9.3 与超分辨率 SDL 的核心差异

| 方面            | 超分辨率 SDL（L₁ ≠ L₂）          | 同分辨率 SDL（L₁ = L₂ = L）          |
| --------------- | --------------------------------- | ------------------------------------- |
| **patch 含义**  | M₁ 低分辨率块 → M₂ 高分辨率块    | M₁ 某风格块 → M₂ 另风格块            |
| **D₁, D₂ 形状** | 不同（L₂ > L₁）                  | 相同                                  |
| **输出分辨率**  | 高于 M₁（超分辨率增强）           | 与 M₁ 相同（偏差修正）               |
| **CC 计算**     | 需上采样 M₁ patch 再算 CC         | 直接计算 CC（无需上采样）             |
| **Baseline**    | 双三次插值上采样 M₁               | M₁ 直接使用（无需插值）               |
| **方法本质**    | 超分辨率增强                      | 系统性偏差修正 / 模型风格迁移        |

#### 2.9.4 SDL 在同分辨率场景下学到了什么

1. **系统性速度偏差**：M₂ 相对于 M₁ 的非线性空间速度偏差结构（简单均值校正无法捕捉的空间异质性）
2. **分辨率异质性效应**：两种反演方法对速度异常的不同"风格"表达（即使分辨率相同，因数据不同、正则化策略不同，两模型的速度场结构也不同）
3. **全域外推能力**：在 M₂ 覆盖区域学到的变换，仍可外推到 M₁ 全域（论文核心贡献，同分辨率场景依然成立）

#### 2.9.5 改善率指标重定义

超分辨率 Baseline = 双三次插值，衡量「相对于无信息插值的改善」。

同分辨率 Baseline = M₁ 直接使用，改善率为：

$$\text{Improvement}_{same} = \frac{\|M_1 - M_2\|_2 - \|\hat{M} - M_2\|_2}{\|M_1 - M_2\|_2} \times 100\%$$

其中 $\hat{M}$ 是 SDL 变换后的模型。此指标衡量「SDL 将 M₁ 校正为更接近 M₂ 的程度」。

#### 2.9.6 实现调整

```python
# 同分辨率时 CC 计算无需上采样
if base_resolution == highres_resolution:
    # 直接计算相关系数，不需要 zoom 上采样
    cc = pearsonr(base_patch.ravel(), highres_patch.ravel())[0]
else:
    # 超分辨率：上采样 M₁ patch 再算 CC（论文方法）
    scale = highres_resolution / base_resolution  # < 1，需放大
    base_upsampled = zoom(base_patch, 1.0/scale, order=3)
    cc = pearsonr(base_upsampled.ravel(), highres_patch.ravel())[0]

# Baseline 计算区别
if base_resolution == highres_resolution:
    # 同分辨率：M₁ 直接作为 Baseline（相同坐标插值）
    baseline_patch = base_patch.ravel()
else:
    # 超分辨率：双三次上采样
    baseline_patch = zoom(base_patch, 1.0/scale, order=3).ravel()
```

> ✅ `DirectPatchExtractor` 通过自动检测 `base_resolution == highres_resolution` 来切换模式，代码接口不变，只需传入相同分辨率的两个模型即可。

---

## 3. 东亚三模型数据分析

### 3.1 模型详细信息

| 模型 | 名称         | 分辨率           | 覆盖范围                   | 有效覆盖 | 角色          |
| ---- | ------------ | ---------------- | -------------------------- | -------- | ------------- |
| M₁   | SinoScope1.0 | 1°×1°×20km       | 55°-165°E, -10°-58°N       | 99.99%   | 待增强 (base) |
| M₂   | EARA2024     | 0.25°×0.25°×10km | 不规则覆盖（从元数据提取） | 部分覆盖 | 高分辨率目标  |
| M₃   | FWEA23       | 0.25°×0.25°×10km | 不规则覆盖（从元数据提取） | 部分覆盖 | 高分辨率目标  |

### 3.2 与论文场景的对比

| 方面         | 论文场景                       | 东亚场景                               | 影响        |
| ------------ | ------------------------------ | -------------------------------------- | ----------- |
| **空间关系** | M₂ ⊂ M₁（M₂ 是 M₁ 的局部区域） | M₂/M₃ ⊂ M₁（FWEA23/EARA2024 部分覆盖） | 基本一致 ✅ |
| 分辨率比     | 1:3 (2D) ~ 1:3 (3D)            | 1:4                                    | 接近        |
| 训练样本     | ~1000 对                       | ~10000 对                              | 足够        |
| 目标模型     | 单一（Z2015 或 F2019）         | 双目标（可选 EARA2024 或 FWEA23）      | 可对比验证  |

**论文原文确认** (Section 3.2):

> "the range of **CVM-S4.26** used in the analysis is **larger than** that of **M₂**"

即 **M₁ 覆盖范围 > M₂ 覆盖范围**，这与东亚场景完全一致：

- SinoScope1.0 (M₁) 覆盖全东亚
- FWEA23/EARA2024 (M₂) 只覆盖部分区域

### 3.3 Patch 尺寸配置

| 物理窗口         | SinoScope (1°) | EARA/FWEA (0.25°) | L₁  | L₂     |
| ---------------- | -------------- | ----------------- | --- | ------ |
| 6°×6° (2D)       | 6×6 格点       | 24×24 格点        | 36  | 576    |
| 6°×6°×200km (3D) | 6×6×10 格点    | 24×24×20 格点     | 360 | 11,520 |

---

## 4. 完整处理流程架构

### 4.1 Phase 关系图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SDL Fusion v3.4 架构                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │ SinoScope1.0│     │  EARA2024   │     │   FWEA23    │                   │
│  │ 原始 (1°)   │     │ 原始(0.25°) │     │ 原始(0.25°) │                   │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │
│         │                   │                   │                           │
│         └─────────────┬─────┴─────────────┬─────┘                           │
│                       ↓                   ↓                                 │
│           ┌───────────────────────────────────────┐                         │
│           │  Phase -1: 数据预处理与 Patch 提取      │                         │
│           │  DirectPatchExtractor ⭐              │                         │
│           └───────────────────┬───────────────────┘                         │
│                               ↓                                             │
│           ┌───────────────────────────────────────┐                         │
│           │  Phase 0: 字典表示能力验证             │                         │
│           │  验证: 重建误差 < 3%                  │                         │
│           └───────────────────┬───────────────────┘                         │
│                               │                                             │
│           ┌───────────────────┼───────────────────┐                         │
│           ↓                   ↓                   ↓                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│  │  模式 A: 默认   │  │  模式 B: 搜索   │  │  模式 C: 最优   │                │
│  │  直接使用代码   │  │  --grid-search │  │  --use-optimal │                │
│  │  中配置的参数   │  │  K × λ 搜索     │  │  加载已有配置   │                │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                │
│          │                   ↓                   │                          │
│          │        ┌─────────────────────┐        │                          │
│          │        │  Phase H: 超参数搜索  │        │                          │
│          │        │  K × λ 联合网格搜索  │        │                          │
│          │        └──────────┬──────────┘        │                          │
│          │                   ↓                   │                          │
│          │        ┌─────────────────────┐        │                          │
│          │        │ optimal_config.json │        │                          │
│          │        └──────────┬──────────┘        │                          │
│          │                   │                   │                          │
│          └───────────────────┼───────────────────┘                          │
│                              ↓                                              │
│           ┌───────────────────────────────────────┐                         │
│           │  Phase 1: 2D Patch 级变换验证          │                         │
│           │  训练: DirectPatchExtractor            │                         │
│           │  输出: 各深度独立 D₁, D₂               │                         │
│           └───────────────────┬───────────────────┘                         │
│                               ↓                                             │
│           ┌───────────────────────────────────────┐                         │
│           │  Phase 1.5: 完整 2D 切片变换 ⭐ v3.4   │                         │
│           │  transform_full_slice_direct()        │                         │
│           │  直接从原始模型提取，与训练一致        │                         │
│           └───────────────────┬───────────────────┘                         │
│                               ↓                                             │
│           ┌───────────────────────────────────────┐                         │
│           │  Phase 2: 3D Patch 融合 (待实现)       │                         │
│           └───────────────────┬───────────────────┘                         │
│                               ↓                                             │
│           ┌───────────────────────────────────────┐                         │
│           │  Phase 3: 全域模型增强 (待实现)        │                         │
│           │  输出: Enhanced_SinoScope1.0.nc       │                         │
│           └───────────────────────────────────────┘                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 三种运行模式

| 模式       | 运行方式        | 说明                         | 适用场景             |
| ---------- | --------------- | ---------------------------- | -------------------- |
| **模式 A** | 默认（无参数）  | 直接使用代码中配置的参数     | 快速测试、参数已确定 |
| **模式 B** | `--grid-search` | 运行 Phase H 网格搜索        | 探索最优参数组合     |
| **模式 C** | `--use-optimal` | 加载已有 optimal_config.json | 生产运行、重复实验   |

### 4.3 各 Phase 步骤

| Phase    | 名称             | 输入                   | 输出                    | 验收标准           | 状态      |
| -------- | ---------------- | ---------------------- | ----------------------- | ------------------ | --------- |
| -1       | 数据预处理       | 三个原始 NetCDF        | 配对 Patches            | 提取成功           | ✅ 完成   |
| 0        | 字典验证         | 各模型原始数据         | 各模型字典              | 重建误差 < 3%      | ✅ 完成   |
| **H** ⭐ | **超参数搜索**   | **Patches + 搜索网格** | **optimal_config.json** | **最高评分**       | ⏳ 新增   |
| 1        | Patch 验证       | 配对 Patches + 参数    | **所有深度** D₁, D₂ ⭐  | 部分深度改善 > 20% | 🔄 更新   |
| **1.5**  | **完整切片变换** | **各深度字典**         | **变换后的完整切片**    | **全域误差 < 10%** | ✅ 完成   |
| 2        | 3D 融合          | 3D patches             | 3D 增强结果             | 3D 优于 2D         | 🔄 设计完成 |
| 3        | 全域增强         | 所有深度变换结果       | 增强模型 NetCDF         | 无边界伪影         | ⏳ 待开发 |

> ⭐ **v3.3.5 更新**: Phase 1 现在对**所有目标深度**进行字典训练（论文方法），而不是只训练 4 个代表性深度然后复用。

### 4.4 Phase H 与其他 Phase 的关系

```
            Phase 0 完成后
                  │
       ┌──────────┼──────────┐
       │          │          │
       ↓          ↓          ↓
   [模式 A]   [模式 B]   [模式 C]
   直接使用   运行搜索   加载配置
   默认参数    Phase H
       │          │          │
       │     ┌────┴────┐     │
       │     │  搜索   │     │
       │     │ K×λ×size│     │
       │     └────┬────┘     │
       │          │          │
       │          ↓          │
       │   保存最优配置      │
       │          │          │
       └──────────┼──────────┘
                  │
                  ↓
              Phase 1
           (使用参数执行)
```

---

## 5. Phase -1: 数据预处理与 Patch 提取

### 5.1 目标

从三个不同分辨率的模型中提取配对的 Patches，为后续耦合字典学习做准备。

### 5.2 实现流程（默认：直接提取方法）

**核心理解（基于论文 Section 2.2）**：

> "Suppose we have an initial model M₁ and a second more local model M₂ with a higher-resolution that is **spatially included in M₁**"
> "extract patches P₁^U from the **entire** M₁"
> "A **subset** of P₁^U, P₁, shares spatial locations with patches P₂"

即：

- M₁ (SinoScope1.0) 是**全域基础模型**
- M₂ (FWEA23) 是 M₁ 的**局部高分辨率区域**（M₂ ⊂ M₁）
- **训练**：在 M₂ 有数据的区域提取配对 patches
- **应用**：对 M₁ 全域进行变换

```python
# Step 1: 加载原始模型（保持原始分辨率）
base_model = xr.open_dataset('2022_SinoScope1.0_original.nc')    # M₁: 1°
highres_model = xr.open_dataset('2024_FWEA23_original.nc')       # M₂: 0.25°

# Step 2: 遍历 M₁ 的网格，在 M₂ 有数据的位置提取配对 patches
#         （不需要预先计算"重叠区域"，而是在提取时判断 M₂ 是否有数据）
extractor = DirectPatchExtractor(config, logger)
for depth_km in target_depths:
    paired_patches = extractor.extract_paired_patches_direct(
        base_ds=base_model,        # M₁: SinoScope1.0 原始 1° 分辨率
        highres_ds=highres_model,  # M₂: FWEA23 原始 0.25° 分辨率
        param='vs',
        depth_km=depth_km
    )
    # 提取逻辑：
    # - 遍历物理坐标位置 (lat_c, lon_c)
    # - 从 M₁ 提取 patch: 6×6 @ 1° = 6° 窗口 → L₁ = 36
    # - 从 M₂ 提取 patch: 24×24 @ 0.25° = 6° 窗口 → L₂ = 576
    # - 只有两个模型在该位置都有数据时，才形成配对
```

**关键区别**：

- ✅ 是"在 M₂ 有数据的位置，与 M₁ 形成配对"，不是三模型重叠区域
- ✅ 变换最终应用到 M₁ 的**全域**（包括 M₂ 没有数据的区域）

### 5.3 输出数据结构

**配对 Patches 结果** (`PairedPatchResult`):

```python
@dataclass
class PairedPatchResult:
    patches_base: np.ndarray      # (N, L₁) 基础模型 patches, L₁=36
    patches_highres: np.ndarray   # (N, L₂) 高分辨率 patches, L₂=576
    coords_latlon: np.ndarray     # (N, 2) patch 中心坐标 [lat, lon]
    correlations: np.ndarray      # (N,) CC 值
    valid_mask: np.ndarray        # (N,) CC>阈值的有效标记
    n_total: int
    n_valid: int
    physical_window_size: float   # 物理窗口大小（度）
    base_patch_size: int          # 基础模型 patch 尺寸 (6)
    highres_patch_size: int       # 高分辨率 patch 尺寸 (24)
    depth_km: float
```

### 5.4 关键类

```python
class DirectPatchExtractor:
    """
    直接从原始网格提取配对 Patches（论文方法）

    核心思想：
    - 直接从 SinoScope 原始 1° 网格提取低分辨率 patches
    - 直接从 EARA/FWEA 原始 0.25° 网格提取高分辨率 patches
    - 配对 patches 覆盖相同的物理区域
    - 不需要预先插值，保持原始数据特征
    """

    def extract_paired_patches_direct(self,
                                      base_ds: xr.Dataset,
                                      highres_ds: xr.Dataset,
                                      param: str,
                                      depth_km: float) -> PairedPatchResult
    def _extract_depth_slice(self, ...) -> np.ndarray
    def _extract_patch_from_original(self, ...) -> np.ndarray
```

### 5.5 直接提取方法详解

#### 5.5.1 提取流程

```python
class DirectPatchExtractor:
    """
    直接从原始网格提取配对 Patches（论文方法）

    M₁: 6×6 格点 @ 1° = 6° 窗口  → L₁ = 36
    M₂: 24×24 格点 @ 0.25° = 6° 窗口 → L₂ = 576
    """

    def extract_paired_patches_direct(self,
                                      base_ds: xr.Dataset,      # 原始 1°
                                      highres_ds: xr.Dataset,   # 原始 0.25°
                                      param: str,
                                      depth_km: float):
        # 遍历重叠区域的物理坐标
        for lat_c, lon_c in overlap_centers:
            # 直接从 M₁ 原始网格提取
            p1 = self._extract_patch_from_original(
                base_slice, base_lat, base_lon,
                lat_start, lat_end, lon_start, lon_end,
                expected_size=6  # 直接 6×6
            )

            # 直接从 M₂ 原始网格提取
            p2 = self._extract_patch_from_original(
                highres_slice, highres_lat, highres_lon,
                lat_start, lat_end, lon_start, lon_end,
                expected_size=24  # 直接 24×24
            )

            patches_M1.append(p1.ravel())  # (36,)
            patches_M2.append(p2.ravel())  # (576,)
```

#### 5.5.2 方法优势

| 特性              | 说明                                         |
| ----------------- | -------------------------------------------- |
| **保真度**        | ✅ 直接使用原始数据，无插值损失              |
| **论文一致**      | ✅ 与论文方法完全一致                        |
| **训练/变换一致** | ✅ Phase 1 训练和 Phase 1.5 变换使用相同数据 |
| **物理对应**      | ✅ 配对 patches 覆盖相同物理区域             |

#### 5.5.3 v3.4 关键修复：训练/变换一致性 ⭐

**问题诊断** (v3.3.x 及之前):

```
Phase 1 (训练):  从原始 1° 网格直接提取 6×6 patch
Phase 1.5 (变换): 从插值后 0.25° 网格提取 24×24，再下采样到 6×6 ❌
                 → 插值 → 下采样 ≠ 原始数据，导致变换失真！
```

**v3.4 修复**:

```
Phase 1 (训练):  从原始 1° 网格直接提取 6×6 patch ✅
Phase 1.5 (变换): 从原始 1° 网格直接提取 6×6 patch ✅
                 → 使用 transform_full_slice_direct() 方法
                 → 训练和变换使用相同的数据源，保证一致性！
```

#### 5.5.4 关于原论文场景

> ⚠️ **重要**: 经过仔细分析，原论文的 M₁ 和 M₂ **也不是嵌套关系**，网格也并不严格对齐，情况与东亚场景类似。论文通过物理坐标精确匹配来确保配对 patches 覆盖相同的地理区域

---

## 6. Phase 0: 字典表示能力验证

### 6.1 目标

验证 SDL 方法能够准确重建速度模型（对应论文 Section 3.1, Figure 2）。

### 6.2 实现流程

```python
For each model in [SinoScope1.0, EARA2024, FWEA23]:
    For each depth in [40, 100, 200, 400] km:

        # 1. 提取 2D 速度切片
        slice_data, lat, lon = get_slice_at_depth(param, model, depth)

        # 2. 提取 patches (窗口 16×16 格点 = 4°×4°)
        patches, coords = extract_patches(slice_data, window_size=16, stride=4)

        # 3. 训练字典 D* (K=20, α=0.1)
        mean_patch = np.mean(patches, axis=0)
        patches_centered = patches - mean_patch

        dict_learner = DictionaryLearning(n_components=20, alpha=0.1)
        dict_learner.fit(patches_centered)

        D = dict_learner.components_  # (K, L)
        C = dict_learner.transform(patches_centered)  # (N, K)

        # 4. 重建
        patches_reconstructed = C @ D + mean_patch

        # 5. 计算误差
        error = np.linalg.norm(patches - patches_reconstructed) / np.linalg.norm(patches) * 100

        # 6. 验收: mean_error < 3%, max_error < 10%
```

### 6.3 验收标准

| 指标         | 论文值 | 目标值 |
| ------------ | ------ | ------ |
| 平均重建误差 | 2.2%   | < 3%   |
| 最大单点误差 | 7%     | < 10%  |

### 6.4 关键类

```python
class DictionaryValidator:
    """字典表示能力验证器"""

    def extract_patches_from_slice(self, slice, lat, lon, window_size, stride) -> ValidationPatchResult
    def train_dictionary(self, patches) -> Tuple[np.ndarray, np.ndarray]  # D, C
    def reconstruct_patches(self, D, C) -> np.ndarray
    def compute_reconstruction_error(self, original, reconstructed) -> Dict[str, float]
    def validate_model_at_depth(self, slice, lat, lon, depth, model_name, resolution) -> ValidationResult
```

---

## 7. Phase H: 超参数联合网格搜索

### 7.1 目标

通过联合网格搜索找到最优的超参数组合 (K, λ, window, stride)，最大化 SDL 变换效果。

**论文依据** (Section 3.2):

> 论文通过网格搜索 **~40 种 K × λ 组合** 确定最优参数
>
> ⭐ **重要**: 论文只搜索 K 和 λ，**patch size 不参与搜索** > _"Out of 40 combinations of hyperparameter configurations, the optimal number of atoms in D₁ and D₂ is determined to be 20, with an optimal sparsity control factor of 0.1."_

### 7.2 搜索参数

| 参数类型 | 参数名              | 搜索范围             | 默认值 | 说明                       |
| -------- | ------------------- | -------------------- | ------ | -------------------------- |
| **搜索** | `n_components` (K)  | 10, 15, 20, 25, 30   | 20     | 字典原子数 ⭐ **论文搜索** |
|          | `alpha` (λ)         | 0.05, 0.1, 0.15, 0.2 | 0.1    | 稀疏性参数 ⭐ **论文搜索** |
|          | `physical_window`   | -                    | 6°     | 物理窗口（由物理约束决定） |
|          | `physical_stride`   | -                    | 1°     | 物理步长（83%重叠率）      |
|          | `d2_regularization` | -                    | 0      | 纯最小二乘（论文方法）     |

**网格搜索配置** (论文方法: 只搜索 K × λ):

```python
# K × λ 联合搜索: 5 × 4 = 20 种组合 (接近论文 40 种)
search_grid = {
    'K_values': [10, 15, 20, 25, 30],      # 字典原子数
    'alpha_values': [0.05, 0.1, 0.15, 0.2], # 稀疏性参数
}
# 窗口和步长由 Phase1Config 固定，不参与搜索
# physical_window_size = 6.0°  (固定)
# physical_stride = 1.0°       (固定)
```

> ⭐ v3.3.4 简化: 移除 quick/full 区分，统一使用 `--grid-search`

### 7.3 评估指标

**综合评分函数** (论文目标: 最小化验证集误差):

$$\text{Score} = 0.4 \times \text{ErrorScore} + 0.4 \times \text{ImproveScore} + 0.2 \times \text{StdScore}$$

其中:

```python
def compute_score(result):
    # 变换误差得分 (越低越好)
    error_score = 1.0 - min(result.transform_error / 20.0, 1.0)

    # 改善率得分 (越高越好)
    improve_score = min(result.relative_improvement / 50.0, 1.0)

    # 误差标准差得分 (越低越好)
    std_score = 1.0 - min(result.error_std / 10.0, 1.0)

    return 0.4 * error_score + 0.4 * improve_score + 0.2 * std_score
```

### 7.4 输出文件

```
results/SDL_grid_search/
├── grid_search_results.csv         # 所有搜索结果
├── optimal_config.json             # 最优配置 ⭐
├── depth_optimal_configs.json      # 各深度最优配置
└── search_log.txt                  # 搜索日志

figures/SDL_Fusion_v3.3/
├── H01_grid_search_heatmap.jpg     # K×λ 热力图
├── H02_param_sensitivity.jpg       # 参数敏感性曲线
├── H03_optimal_config.jpg          # 最优配置卡片
├── H04_search_progress.jpg         # 搜索进度
└── H05_depth_optimal.jpg           # 各深度最优分布
```

### 7.5 关键类

```python
@dataclass
class GridSearchResult:
    """单次搜索结果"""
    config_name: str
    window: float
    stride: float
    n_components: int
    alpha: float

    # 性能指标
    transform_error: float
    baseline_error: float
    relative_improvement: float
    error_mean: float
    error_std: float
    n_patches: int

    # 各深度结果
    depth_results: Dict[float, Dict]

    # 综合评分
    score: float


class HyperparameterSearcher:
    """超参数联合网格搜索器"""

    def __init__(self, sdl_fusion: SDLFusionV33):
        self.sdl_fusion = sdl_fusion
        self.results: List[GridSearchResult] = []
        self.best_config: Optional[GridSearchResult] = None

    def search(self,
               window_stride_pairs: List[Tuple[float, float]],
               K_values: List[int],
               alpha_values: List[float],
               depths: List[float],
               param: str = 'vs',
               target_model: str = 'fwea') -> GridSearchResult:
        """执行联合网格搜索"""
        ...

    def save_results(self, output_dir: Path) -> None:
        """保存搜索结果"""
        ...

    def load_optimal_config(self, config_path: Path) -> Dict:
        """加载最优配置"""
        ...
```

### 7.6 命令行接口 (v3.3.4 简化)

```bash
# 模式 A: 直接使用代码中配置的参数（默认）
python 4_3_SDL_Fusion_v3.3.py

# 模式 B: K × λ 联合网格搜索 (20 种组合)
python 4_3_SDL_Fusion_v3.3.py --grid-search

# 模式 C: 使用已有最优配置
python 4_3_SDL_Fusion_v3.3.py --use-optimal

# 可选参数
python 4_3_SDL_Fusion_v3.3.py --skip-phase0      # 跳过字典验证
python 4_3_SDL_Fusion_v3.3.py --skip-phase1      # 跳过Patch级验证
```

> ⭐ v3.3.4: 简化为单一 `--grid-search`，移除 quick/full 区分

**模式 A 参数配置位置** (直接在代码中修改):

```python
# 在 Phase1Config 中配置超参数
@dataclass
class Phase1Config:
    physical_window_size: float = 6.0    # 物理窗口 6°
    physical_stride: float = 1.0         # 步长 1° (83%重叠)
    n_components: int = 20               # K: 字典原子数
    alpha: float = 0.1                   # λ: 稀疏性参数
```

---

## 8. Phase 1: 2D Patch 级变换验证

### 8.1 目标

验证 SDL 变换方法在 Patch 级别的有效性。

### 8.2 实现流程

```
┌────────────────────────────────────────────────────────────────┐
│              Phase 1: Patch 级变换验证                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: 在重叠区域提取配对 patches                             │
│    ├── SinoScope patches: 6×6 (下采样到原始 1° 分辨率)         │
│    ├── EARA patches: 24×24 (原始 0.25° 分辨率)                 │
│    ├── 物理窗口: 6° × 6°                                       │
│    └── CC 筛选: CC > 0                                         │
│                                                                 │
│  Step 2: 划分训练/验证集 (80%/20%)                              │
│                                                                 │
│  Step 3: 训练耦合字典                                           │
│    ├── 中心化: P1_centered = P1 - mean_P1                      │
│    ├── 训练 D₁: DictionaryLearning.fit(P1_train_centered)      │
│    ├── 获取 C: DictionaryLearning.transform(P1_train_centered) │
│    └── 计算 D₂: np.linalg.lstsq(C, P2_centered) ⭐ 纯最小二乘  │
│                                                                 │
│  Step 4: 验证集评估                                             │
│    ├── 编码: C_val = dict_learner.transform(P1_val_centered)   │
│    ├── SDL 变换: P2_sdl = C_val @ D2 + mean_P2                 │
│    ├── Baseline: P2_baseline = bicubic_upsample(P1_val)        │
│    └── 误差对比                                                 │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 8.3 配对 Patch 提取细节（直接提取方法）

```python
def extract_paired_patches_direct(base_ds, highres_ds, param, depth_km):
    """
    直接从原始网格提取配对 patches（论文方法）

    核心约束: 配对 patches 必须覆盖完全相同的物理区域

    M₁ (SinoScope1.0): 1° 分辨率
    M₂ (FWEA23):       0.25° 分辨率
    """
    # 获取原始切片
    base_slice = get_depth_slice(base_ds, param, depth_km)      # M₁ @ 1°
    highres_slice = get_depth_slice(highres_ds, param, depth_km)  # M₂ @ 0.25°

    for lat_c, lon_c in M1_grid_centers:  # 遍历 M₁ 的网格
        # 检查 M₂ 在该位置是否有数据
        if not has_data(highres_ds, lat_c, lon_c, physical_window):
            continue  # M₂ 无数据则跳过

        # 直接从 M₁ 原始 1° 网格提取 (无需下采样)
        base_patch = extract_patch_at(base_slice, lat_c, lon_c,
                                       window_size=6)  # 6×6 @ 1° = 6° 窗口

        # 直接从 M₂ 原始 0.25° 网格提取 (无需上/下采样)
        highres_patch = extract_patch_at(highres_slice, lat_c, lon_c,
                                         window_size=24)  # 24×24 @ 0.25° = 6° 窗口

        # 计算 CC (论文: 上采样 M₁ patch 后计算)
        base_upsampled = zoom(base_patch, scale=4, order=3)  # 6×6 → 24×24
        cc = pearsonr(base_upsampled.ravel(), highres_patch.ravel())[0]

        if cc > 0:  # CC 筛选
            patches_base.append(base_patch.ravel())      # L₁ = 36
            patches_highres.append(highres_patch.ravel())  # L₂ = 576
```

### 8.4 耦合字典训练细节

```python
class CoupledDictionaryLearner:
    """耦合字典学习器"""

    def train(self, paired_patches: PairedPatchResult) -> CoupledDictionaryResult:
        # 获取有效 patches
        P1_all = paired_patches.patches_base[valid_mask]      # (N, 36)
        P2_all = paired_patches.patches_highres[valid_mask]   # (N, 576)

        # 划分训练/验证集
        n_train = int(len(P1_all) * 0.8)
        P1_train, P1_val = P1_all[:n_train], P1_all[n_train:]
        P2_train, P2_val = P2_all[:n_train], P2_all[n_train:]

        # Step 1: 中心化
        mean_P1 = np.mean(P1_train, axis=0)
        mean_P2 = np.mean(P2_train, axis=0)
        P1_train_centered = P1_train - mean_P1
        P2_train_centered = P2_train - mean_P2

        # Step 2: 训练 D₁ 和 C (公式 6)
        dict_learner = DictionaryLearning(
            n_components=20,      # K=20
            alpha=0.1,            # λ=0.1
            transform_algorithm='omp',
            transform_n_nonzero_coefs=5
        )
        dict_learner.fit(P1_train_centered)
        D1 = dict_learner.components_           # (K, L₁) = (20, 36)
        C_train = dict_learner.transform(P1_train_centered)  # (N_train, K)

        # Step 3: 计算 D₂ (公式 7 + 正则化)
        ridge = Ridge(alpha=0.1, fit_intercept=False)
        ridge.fit(C_train, P2_train_centered)
        D2 = ridge.coef_.T                      # (K, L₂) = (20, 576)

        # Step 4: 验证集评估
        P1_val_centered = P1_val - mean_P1
        C_val = dict_learner.transform(P1_val_centered)
        P2_val_sdl = C_val @ D2 + mean_P2       # SDL 变换

        # 计算误差
        transform_error = np.linalg.norm(P2_val - P2_val_sdl) / np.linalg.norm(P2_val) * 100
```

### 8.5 验收标准

| 指标     | 论文值 | 目标值 |
| -------- | ------ | ------ |
| 变换误差 | 7.6%   | < 10%  |
| 相对改善 | ~35%   | > 20%  |

### 8.6 关键类

```python
class DirectPatchExtractor:
    """直接从原始网格提取配对 Patches（论文方法）"""
    def extract_paired_patches_direct(self, base_ds, highres_ds, ...) -> PairedPatchResult

class CoupledDictionaryLearner:
    """耦合字典学习器"""
    def train(self, paired_patches) -> CoupledDictionaryResult

class TwoDTransformer:
    """2D 水平切片变换器"""
    def run(self, base_ds, highres_ds, param, depth_km, target_model) -> TransformResult
    def get_best_dictionaries(self) -> Dict[float, CoupledDictionaryResult]
```

---

## 9. Phase 1.5: 完整 2D 切片变换

### 9.1 目标

**核心功能**: 对 SinoScope1.0 的 **所有深度层** 完整 2D 切片应用 SDL 变换，生成增强后的完整速度切片。

对应论文 **Figure 3** 的实现。

> ⭐ **v3.3.5 更新**:
>
> - Phase 1 对**所有深度**训练独立的字典对 (D₁, D₂)，符合论文方法
> - Phase 1.5 使用**对应深度**的字典进行变换，而不是复用最近深度的字典
> - 可视化时自动选择 **代表性深度** (最多 5 层) 进行展示

### 9.2 与 Phase 1 的区别

| 方面     | Phase 1 (Patch 验证)       | Phase 1.5 (完整切片)           |
| -------- | -------------------------- | ------------------------------ |
| 输入     | 验证集 patches             | SinoScope 完整切片             |
| 输出     | **各深度的 D₁, D₂** ⭐     | 变换后的完整切片               |
| 区域     | 仅重叠区域                 | **M₁ 全域**                    |
| 深度范围 | **所有深度**（独立训练）⭐ | **所有深度**（使用对应字典）⭐ |
| 目的     | 训练耦合字典               | 实际应用变换                   |
| 处理量   | ~3000 个 patches × N 深度  | ~2000+ 个 patches/深度         |

### 9.3 完整切片变换流程（v3.4 直接提取方法）⭐

```python
def transform_full_slice_direct(self, base_ds, highres_ds, param, depth_km,
                                 coupled_dict, target_model, stitch_method='average'):
    """
    直接从原始模型数据进行完整切片变换（v3.4 论文方法）

    ⭐ 关键修复：训练和变换使用相同的原始数据
    - 训练时: 从原始 1° 网格直接提取 6×6 patches
    - 变换时: 也从原始 1° 网格提取，保持一致性
    """
    # 参数
    physical_window = 6.0  # 6°
    physical_stride = 1.0  # 1° (83% 重叠)
    base_patch_size = 6      # 1° × 6 = 6°
    highres_patch_size = 24  # 0.25° × 24 = 6°

    # 获取原始模型的坐标和切片
    base_lat, base_lon = base_ds['latitude'].values, base_ds['longitude'].values
    highres_lat, highres_lon = highres_ds['latitude'].values, highres_ds['longitude'].values

    base_slice = self._extract_depth_slice_direct(base_ds, param, depth_km)
    highres_slice = self._extract_depth_slice_direct(highres_ds, param, depth_km)

    # 初始化输出（高分辨率网格）
    reconstructed = np.zeros((len(highres_lat), len(highres_lon)))
    weight_map = np.zeros_like(reconstructed)

    # 权重（average 或 gaussian）
    if stitch_method == 'gaussian':
        patch_weight = create_gaussian_weight(highres_patch_size)
    else:
        patch_weight = np.ones((highres_patch_size, highres_patch_size))

    # 获取字典参数
    D2 = coupled_dict.D2
    mean_P1 = coupled_dict.preprocessing.mean_P1
    mean_P2 = coupled_dict.preprocessing.mean_P2
    dict_learner = coupled_dict.dict_learner

    # 遍历所有 patch 位置
    for lat_c in lat_centers:
        for lon_c in lon_centers:
            # === 直接从原始 1° 网格提取 ===
            base_lat_idx = np.where((base_lat >= lat_start) & (base_lat < lat_end))[0]
            base_lon_idx = np.where((base_lon >= lon_start) & (base_lon < lon_end))[0]
            base_patch = base_slice[np.ix_(base_lat_idx[:6], base_lon_idx[:6])]  # 6×6

            # 编码: p₁ → c
            p1 = base_patch.ravel()  # (36,)
            p1_centered = p1 - mean_P1
            c = dict_learner.transform([p1_centered])  # (1, K)

            # 解码: c → p̂₂
            p2_centered = c @ D2  # (1, L₂)
            p2 = p2_centered.ravel() + mean_P2
            transformed_patch = p2.reshape(24, 24)

            # 确定输出位置（高分辨率网格）
            highres_lat_idx = np.where((highres_lat >= lat_start) & (highres_lat < lat_end))[0]
            highres_lon_idx = np.where((highres_lon >= lon_start) & (highres_lon < lon_end))[0]

            # 加权拼接
            lat_slice = slice(highres_lat_idx[0], highres_lat_idx[0] + 24)
            lon_slice = slice(highres_lon_idx[0], highres_lon_idx[0] + 24)
            reconstructed[lat_slice, lon_slice] += transformed_patch * patch_weight
            weight_map[lat_slice, lon_slice] += patch_weight

    # 归一化
    valid_mask = weight_map > 0
    reconstructed[valid_mask] /= weight_map[valid_mask]

    return reconstructed, weight_map, highres_lat, highres_lon
```

### 9.4 验收标准

| 指标         | 论文值 | 目标值 |
| ------------ | ------ | ------ |
| 全域变换误差 | 7.6%   | < 10%  |
| 相对改善     | ~35%   | > 20%  |
| 误差均值     | 0.4%   | < 1%   |
| 误差标准差   | ~3%    | < 5%   |

### 9.5 关键类

```python
class FullSliceTransformer:
    """完整 2D 切片变换器（v3.4 直接提取方法）"""

    def set_trained_dict(self, depth_km, coupled_dict) -> None
    def get_dict(self, depth_km) -> CoupledDictionaryResult  # v3.4: 精确匹配，不复用
    def transform_full_slice_direct(self, base_ds, highres_ds, ...) -> FullSliceTransformResult  # v3.4 新增
    def _extract_depth_slice_direct(self, ds, param, depth_km) -> np.ndarray  # v3.4 新增
```

---

## 10. Phase 2: 3D Patch 融合

### 10.1 目标

将 2D 逐深度处理扩展为 **3D Patch**，同时利用水平空间结构与深度方向的相关性，对应论文 Section 3.3（Figure 4-5）。

### 10.2 3D 方法的优势

> _"velocities at different depths are not entirely independent and are likely to include correlated variations. Utilizing 3D reconstruction ensures continuity and smoothness in depth for the reconstructed model."_

| 优势                 | 说明                                                          |
| -------------------- | ------------------------------------------------------------- |
| **利用深度相关性**   | 速度在相邻深度之间存在系统性相关变化，3D patch 天然捕捉这种规律 |
| **深度连续性**       | 避免逐深度独立训练导致的深度方向速度不连续跳变               |
| **多参数联合**       | Vp 和 Vs 可共享稀疏系数 C，物理上强制 Vp/Vs 比合理           |
| **样本扩增**         | 滑动深度窗口将 2D 的 N_spatial 对扩增为 N_spatial × N_window 对 |
| **论文改善率更高**   | 3D 改善率 ~58% > 2D 改善率 ~34%（论文 Section 3.3）          |

### 10.3 USTClitho→CSES 3D Patch 配置详解

#### 10.3.1 模型深度信息

| 模型           | 总深度层 | 匹配深度层 (tolerance=0.1km)            | 深度间距    |
| -------------- | -------- | --------------------------------------- | ----------- |
| USTClitho2.0   | 12 层    | **9 层** [0,10,20,30,40,60,80,100,120]  | 不规则      |
| CSES_VM1.0     | 47 层    | **9 层**（同上）                        | ~2-4km 均匀 |

**深度间距不规则问题**：匹配深度间隔为 [10,10,10,10,20,20,20,20] km，在 3D patch 中视为等间距深度"通道"处理（论文同样如此，只用索引位置，不做归一化）。

#### 10.3.2 两种 3D Patch 策略

**策略 A：全深度堆叠**（简单，样本少）

```
M₁ (USTClitho @ 0.5°):  6 × 6 × 9  → L₁ = 324
M₂ (CSES @ 0.25°):      12 × 12 × 9 → L₂ = 1296
空间训练对: ~236（CSES覆盖域内）
深度"位置": 1（所有匹配深度一次处理）
总 3D 训练对: 236
```

**策略 B：滑动深度窗口**（推荐，样本充足）⭐

```
深度窗口大小: n_d = 5 层
深度步长:     s_d = 1 层
深度窗口数: (9 - 5) / 1 + 1 = 5 个

M₁ patch: 6 × 6 × 5  → L₁ = 180
M₂ patch: 12 × 12 × 5 → L₂ = 720

总 3D 训练对: 5 × 236 = 1,180 ≈ 论文的 980 对 ✅
```

> ⭐ **推荐策略 B**：训练对数量接近论文，且学习到的字典更具深度泛化性。

**策略 A vs B 对比**：

| 指标               | 策略 A (全深度) | 策略 B (滑窗 n=5) |
| ------------------ | -------------- | ----------------- |
| M₁ patch 维度 L₁  | 324            | 180               |
| M₂ patch 维度 L₂  | 1296           | 720               |
| 训练对数量         | 236            | **1,180** ✅      |
| 对应深度组合       | 唯一组合       | 5 种连续组合      |
| 深度泛化性         | 弱             | **强** ✅         |
| 内存占用           | 大             | 中等              |

#### 10.3.3 多参数 Vp+Vs 联合训练

论文 Section 3.3 将 Vp 和 Vs 拼接在同一 patch 向量中，**共享稀疏系数 C**：

```python
# Vp 和 Vs 拼接 → 单一 3D patch
# M₁ patch: [vs_patch, vp_patch] → L₁ = 2 × 6×6×5 = 360
# M₂ patch: [vs_patch, vp_patch] → L₂ = 2 × 12×12×5 = 1440

P1_joint = np.concatenate([P1_vs, P1_vp], axis=1)   # (N, 360)
P2_joint = np.concatenate([P2_vs, P2_vp], axis=1)   # (N, 1440)

# 共享字典训练
D1_joint, C = train_dictionary(P1_joint)              # D1: (K, 360)
D2_joint = lstsq(C, P2_joint)                         # D2: (K, 1440)

# 解码时分离 Vs 和 Vp
P2_hat_joint = C_new @ D2_joint                       # (N, 1440)
P2_hat_vs = P2_hat_joint[:, :720]                     # Vs 部分
P2_hat_vp = P2_hat_joint[:, 720:]                     # Vp 部分
```

> 关键优势：**共享 C** 保证 Vs 和 Vp 的变换方向一致，物理上约束了 Vp/Vs 比的合理性。

### 10.4 论文 3D 参数对比

| 参数          | 论文 (CVM-S4.26 → F2019) | USTClitho→CSES (目标值) | 说明                       |
| ------------- | ------------------------ | ------------------------ | -------------------------- |
| M₁ 水平分辨率 | 0.09°                    | 0.5°                     | 使用双三次插值预处理 M₁ ⭐ |
| M₂ 水平分辨率 | 0.03°                    | 0.25°                    | 分辨率比 2:1               |
| **预处理**    | **双三次插值上采样 M₁**  | **同样需要上采样 M₁**   | 论文 3D 方法特有            |
| 3D 窗口       | 9×9×20 格点              | **6×6×5** (策略B)        | L₁=180 vs 论文 1620        |
| 分辨率比      | 3:1                      | 2:1                      | 接近                       |
| 训练对        | 980 对                   | **~1,180 对** ✅         | 策略 B 接近论文             |
| K (原子数)    | 20                       | 20                       | ✅ 一致                    |
| 参数类型      | Vp + Vs 联合             | Vp + Vs 联合             | ✅ 一致                    |

> ⭐ **论文 3D 方法的特殊预处理**：论文对 M₁ 做了双三次插值上采样到 M₂ 的分辨率，再提取 3D patch。这与 2D 方法（直接使用原始网格）不同，是为了让 D₁ 和 D₂ 的 patch 维度对称（L₁ = L₂）！

#### 两种 3D 预处理方式对比

```
方式 A（论文 3D 方法）: 上采样 M₁ 到 M₂ 分辨率
  M₁ 上采样: 0.5° → 0.25° (zoom 2×)
  M₁ 3D patch: 12×12×5 → L₁ = 720
  M₂ 3D patch: 12×12×5 → L₂ = 720
  L₁ = L₂ = 720  ← 维度对称
  优点: 与论文完全一致，字典原子可直接可视化比较
  缺点: 引入插值误差（与 2D 直接提取精神不完全一致）

方式 B（直接提取，2D 方法延伸）:
  M₁ 3D patch: 6×6×5 → L₁ = 180（原始 0.5° 网格）
  M₂ 3D patch: 12×12×5 → L₂ = 720（原始 0.25° 网格）
  L₁ ≠ L₂  ← 维度不对称（同 2D 直接提取方法）
  优点: 保留 2D 阶段的训练/变换一致性优势
  缺点: 与论文 3D 实现略有差异
```

**推荐**：先实现方式 B（与现有代码架构一致），再评估方式 A 是否显著提升性能。

### 10.5 论文 3D 性能指标

| 指标          | 论文值  | 目标值 (USTClitho→CSES) |
| ------------- | ------- | ----------------------- |
| Vp 变换误差   | 2.92%   | < 5%                    |
| Vs 变换误差   | 2.98%   | < 5%                    |
| Baseline 误差 | 8.12%   | ~3-4%（2D 参考）        |
| **相对改善**  | **~58%** | **> 40%**               |
| 误差分布均值  | 0.4%    | < 1%                    |
| 误差分布标准差| ~3%     | < 5%                    |

### 10.6 Phase2Config（优化版）

```python
@dataclass
class Phase2Config:
    """Phase 2: 3D Patch 融合配置（USTClitho→CSES 优化）"""

    enabled: bool = True  # Phase 1 完成后启用

    # === 水平窗口（与 Phase 1 一致）===
    physical_window_lat: float = 3.0   # 3°（CSES 覆盖域决定上限）
    physical_window_lon: float = 3.0
    physical_stride_lat: float = 0.5   # 1格点@0.5°，83%重叠
    physical_stride_lon: float = 0.5

    # === 深度窗口配置（策略 B：滑动窗口）⭐ ===
    n_depth_window: int = 5   # 滑动窗口大小（匹配深度层数）
    depth_stride: int = 1     # 深度步长（格点，=1 最大扩增效果）

    # === 预处理方式（见 10.4 两种方式）===
    upsample_base: bool = False  # True=论文方式A; False=直接提取方式B

    # === 字典参数 ===
    n_components: int = 20
    alpha: float = 0.1
    # 3D patch 更大 → 允许更多非零系数
    n_nonzero_coefs: int = 8    # 2D 为 5，3D 增至 8
    max_iter: int = 100
    train_ratio: float = 0.8

    # === 多参数配置 ===
    joint_params: List[str] = field(
        default_factory=lambda: ['vs', 'vp']
    )  # 联合训练参数列表，空列表=单独训练

    # === 验收标准（高于 2D）===
    max_transform_error: float = 8.0         # < 8%（严于 2D 的 10%）
    min_relative_improvement: float = 40.0   # > 40%（高于 2D 的 20%）
    max_error_mean: float = 1.0
    max_error_std: float = 5.0
```

### 10.7 关键类设计

```python
class ThreeDPatchExtractor:
    """3D Patch 提取器（策略 B：直接提取 + 滑动深度窗口）"""

    def extract_3d_paired_patches(self,
                                  base_ds: xr.Dataset,
                                  highres_ds: xr.Dataset,
                                  params: List[str],
                                  matched_depths: List[float],
                                  config: Phase2Config) -> List[PairedPatch3DResult]:
        """
        提取所有深度窗口的 3D 配对 patches

        Returns:
            list of PairedPatch3DResult，每个对应一个深度窗口位置
        """

class CoupledDict3DLearner:
    """3D 耦合字典学习器（支持 Vp+Vs 联合）"""

    def train(self,
              patches_base: np.ndarray,   # (N, L₁) 含所有深度窗口
              patches_highres: np.ndarray # (N, L₂)
              ) -> CoupledDictionaryResult:
        """共享稀疏系数 C 训练 D₁ 和 D₂"""


class ThreeDTransformer:
    """Phase 2: 3D 变换器"""

    def run(self,
            model_loader: ModelLoader,
            params: List[str],
            target_model: str,
            config: Phase2Config) -> Dict[str, FullSliceTransformResult]:
        """
        对 M₁ 全域的所有深度窗口进行 3D 变换

        步骤:
        1. 对每个深度窗口提取配对 patches → 训练 (D₁_joint, D₂_joint)
        2. 滑动窗口遍历 M₁ 全域提取 3D patch → 编码 → 解码
        3. 深度方向加权拼接（相邻窗口重叠部分取平均）
        4. 分离 Vs 和 Vp，输出增强后的完整 3D 速度场
        """
```

### 10.8 待实现

⏳ Phase 2 待实现（优先级：Phase 1.5 全深度结果验证完成后）

**实现优先级**：
1. `ThreeDPatchExtractor.extract_3d_paired_patches()` — 在 `DirectPatchExtractor` 基础上增加深度维度
2. `CoupledDict3DLearner.train()` — 在 `CoupledDictionaryLearner` 基础上处理 L₁ ≠ L₂ 的大向量
3. `ThreeDTransformer.run()` — 深度方向滑动 + 拼接逻辑
4. 3D 可视化：3D01-3D06 图组（深度剖面、误差分布、字典原子三维可视化）



---

## 11. Phase 3: 全域模型增强

### 10.1 目标

生成最终的增强模型 NetCDF 文件。

### 10.2 输出内容

```
Enhanced_SinoScope1.0_v31.nc:
├── vs_enhanced: 增强后的 Vs (lon×lat×depth @ 0.25°)
├── vp_enhanced: 增强后的 Vp (lon×lat×depth @ 0.25°)
├── vs_original: 原始 SinoScope Vs (用于对比)
├── vs_baseline: Baseline 上采样 (用于对比)
├── improvement_map: 改善率空间分布
├── confidence_map: 置信度分布
└── metadata: 方法参数、训练信息等
```

### 10.3 待实现

⏳ Phase 3 待实现（优先级：Phase 2 完成后）

---

## 12. Patches 拼接方法详解

### 11.1 论文依据

**Zhang & Ben-Zion (2024)**:

> _"The updated velocity model can be then developed by **stitching the reconstructed patches together** in their original spatial order (Dong et al., 2015)."_

**Dong et al. (2015) SRCNN**:

> _"In the traditional methods, the predicted overlapping high-resolution patches are often **averaged** to produce the final full image."_

### 11.2 两种拼接方法

#### 方法 1: 简单平均（Dong 2015 传统方法）

```python
# 简单平均拼接
reconstructed[lat_start:lat_end, lon_start:lon_end] += patch
count_map[lat_start:lat_end, lon_start:lon_end] += 1
# 最后
reconstructed /= count_map
```

**优点**: 实现简单，更接近论文
**缺点**: 边界可能不够平滑

#### 方法 2: 高斯加权平均（我们的实现）

```python
def create_gaussian_weight(patch_size: int) -> np.ndarray:
    """创建高斯权重矩阵，中心权重高，边缘权重低"""
    y, x = np.ogrid[:patch_size, :patch_size]
    center = (patch_size - 1) / 2
    sigma = patch_size / 3
    weight = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    return weight

# 高斯加权拼接
reconstructed[lat_start:lat_end, lon_start:lon_end] += patch * gaussian_weight
weight_map[lat_start:lat_end, lon_start:lon_end] += gaussian_weight
# 最后
reconstructed /= weight_map
```

**优点**: 过渡更平滑
**缺点**: 需要设置 sigma 参数

### 11.3 重叠率设置

| Patch 尺寸 | 重叠像素 | Stride | 重叠率 | 来源      |
| ---------- | -------- | ------ | ------ | --------- |
| 9×9        | 4        | 5      | 44%    | Dong 2015 |
| 6×6 (当前) | 3        | 3      | 50%    | 当前设置  |

### 11.4 代码实现

代码支持两种方法，通过 `stitch_method` 参数选择：

```python
result = sdl_fusion.run_full_pipeline(
    stitch_method='average'  # 或 'gaussian'
)
```

---

## 13. 全流程可视化体系

### 12.0 地理可视化增强 ⭐ v3.3.4 新增

**所有地理坐标可视化统一添加海岸线数据**，使用 `cartopy` 模块：

```python
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_geo_slice(data, lat, lon, title):
    """带海岸线的地理可视化"""
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # 绑制数据
    im = ax.pcolormesh(lon, lat, data, cmap='jet_r',
                       transform=ccrs.PlateCarree())

    # 添加海岸线 ⭐
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='k')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')

    # 添加经纬度网格
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Vs (km/s)')
```

**适用图类**: U01-U05, V01-V02, F01-F06, S 组所有地理切片图

### 12.1 可视化分组

| 组别      | 图号范围  | 数量   | Phase | 说明              | 状态  |
| --------- | --------- | ------ | ----- | ----------------- | ----- |
| **U 组**  | U01-U05   | 5      | -1    | 数据预处理        | ✅    |
| **V 组**  | V01-V08   | 8      | 0     | 字典验证          | ✅    |
| **T 组**  | T01-T11   | 11     | 1     | Patch 级验证      | ✅    |
| **F 组**  | F01-F06   | 6      | 1.5   | 完整切片变换      | ✅    |
| **H 组**  | H01-H05   | 5      | -     | **超参数搜索** ⭐ | ⏳    |
| **3D 组** | 3D01-3D06 | 6      | 2     | 3D 融合           | ⏳    |
| **G 组**  | G01-G06   | 6      | 3     | 全域增强          | ⏳    |
| **S 组**  | S01-S06   | 6      | -     | 综合评估          | 🔄    |
| **总计**  | -         | **53** | -     | -                 | 28/53 |

#### 新增可视化（超参数搜索 H 组）⭐

| 图号    | 名称                | 说明                                | 状态 |
| ------- | ------------------- | ----------------------------------- | ---- |
| **H01** | grid_search_heatmap | K × λ 性能热力图（变换误差/改善率） | ⏳   |
| **H02** | param_sensitivity   | 单参数敏感性曲线                    | ⏳   |
| **H03** | optimal_config      | 最优配置详情卡片                    | ⏳   |
| **H04** | search_progress     | 搜索过程收敛曲线                    | ⏳   |
| **H05** | depth_optimal       | 各深度最优参数分布                  | ⏳   |

#### 新增可视化（论文附录图风格）

| 图号    | 名称                   | 对应论文     | 说明                                              | 状态 |
| ------- | ---------------------- | ------------ | ------------------------------------------------- | ---- |
| **S02** | reconstruction_scatter | Figure S2/S3 | X=(M₂-M₁), Y=(Recon-M₁) 散点图，验证 SDL 调整方向 | ⏳   |
| **S03** | atom_correlation       | Figure S1    | D₁-D₂ 原子相关性矩阵                              | ⏳   |
| **F04** | error_distribution     | Figure S4    | 误差分布直方图（与 Baseline 对比）                | 🔄   |

### 12.2 核心可视化详细设计

#### F01: 完整切片变换对比（论文 Figure 3 风格）

```
┌─────────────────────────────────────────────────────────────────┐
│                 F01: Full Slice Transformation                   │
│                      @ XXX km, Vs                                │
├───────────────────────────┬─────────────────────────────────────┤
│  (a) EARA2024 (Target GT) │  (b) SinoScope1.0 (Original)        │
│  高分辨率目标模型          │  低分辨率原始模型                    │
├───────────────────────────┼─────────────────────────────────────┤
│  (c) SDL Reconstruction   │  (d) Difference (SDL - Original)    │
│  SDL 变换结果              │  差异图（显示变换效果）              │
└───────────────────────────┴─────────────────────────────────────┘

配色方案:
- (a)(b)(c): jet_r, 统一色标范围
- (d): RdBu_r, 对称色标
```

#### F02: 多深度完整切片（论文 Figure 4-5 风格）

> ⭐ **v3.3.4 更新**: 自动选择 **代表性深度** (最多 5 层) 进行可视化，
> 覆盖浅层 (~40-100km)、中层 (~200-300km)、深层 (~500-800km)

```
       40km    100km    200km    500km    800km  (代表性深度)
Row 1: EARA2024 (Target GT)
Row 2: SinoScope1.0 (Original)
Row 3: SDL Reconstruction
Row 4: Difference (SDL - Original)
```

#### F06: 深度汇总曲线

```
(a) Transform Error vs Depth: SDL 误差 vs Baseline 误差
(b) Improvement vs Depth: 改善率柱状图
(c) Error Mean ± Std vs Depth: 误差统计
(d) Error Std vs Depth: 误差标准差
```

### 12.3 已生成可视化 (v3.3)

```
figures/SDL_Fusion_v3.3/ (共 38 张)
├── U 组 (4 张): 数据预处理
│   ├── 4-3_v3-3_U01_model_coverage.jpg
│   ├── 4-3_v3-3_U02_depth_coverage.jpg
│   ├── 4-3_v3-3_U04_unified_preview.jpg
│   └── 4-3_v3-3_U05_nan_distribution.jpg
│
├── V 组 (17 张): 字典验证
│   ├── 4-3_v3-3_V01_original_slices_vs.jpg
│   ├── 4-3_v3-3_V02_full_reconstruction_{model}_{depth}km_vs.jpg (12 张)
│   ├── 4-3_v3-3_V03_patch_reconstruction_vs.jpg
│   ├── 4-3_v3-3_V05_atoms_{model}_{depth}km_vs.jpg (3 张)
│   └── 4-3_v3-3_V07_depth_summary_vs.jpg
│
├── T 组 (6 张): Patch 级验证
│   ├── 4-3_v3-3_T02_paired_patches_vs_40km.jpg
│   ├── 4-3_v3-3_T03_cc_distribution_vs.jpg
│   ├── 4-3_v3-3_T04_D1_atoms_vs_40km.jpg
│   ├── 4-3_v3-3_T05_D2_atoms_vs_40km.jpg
│   ├── 4-3_v3-3_T08_validation_vs_40km.jpg
│   └── 4-3_v3-3_T10_depth_summary_vs_fwea.jpg
│
├── F 组 (10 张): 完整切片变换 ⭐
│   ├── 4-3_v3-3_F01_full_slice_vs_{depth}km.jpg (4 张: 40,100,500,800km)
│   ├── 4-3_v3-3_F02_multi_depth_vs.jpg
│   ├── 4-3_v3-3_F03_improvement_heatmap_vs_40km.jpg
│   ├── 4-3_v3-3_F04_error_distribution_vs_40km.jpg
│   ├── 4-3_v3-3_F05_weight_map_vs_40km.jpg
│   └── 4-3_v3-3_F06_depth_summary_vs_fwea.jpg
│
├── S 组 (1 张): 综合评估
│   └── 4-3_v3-3_S01_error_report.jpg
│
├── H 组 (待实现): 超参数搜索
│   ├── H01_grid_search_heatmap.jpg ⏳
│   ├── H02_param_sensitivity.jpg ⏳
│   └── H03_optimal_config.jpg ⏳
│
└── unified_models_v3-3.nc (统一数据集)
```

---

## 14. 代码架构设计

### 14.1 类层次结构 (v3.4)

```
SDLFusionV34 (主类)
├── config: SDLConfigV34
│   ├── model_info: Dict[str, ModelResolutionInfo]  # 模型信息
│   ├── phase0: Phase0Config
│   ├── phase1: Phase1Config
│   ├── phase2: Phase2Config
│   ├── phase3: Phase3Config
│   └── visualization: VisualizationConfig
│
├── models: Dict[str, xr.Dataset]  # 原始模型数据 ⭐ v3.4
│   ├── '2022_SinoScope1.0': 原始 1° 模型
│   ├── '2024_EARA2024': 原始 0.25° 模型
│   └── '2024_FWEA23': 原始 0.25° 模型
│
├── direct_extractor: DirectPatchExtractor ✅
│   └── 直接从原始网格提取配对 Patches（论文方法）
│
├── validator: DictionaryValidator ✅
│   └── Phase 0 字典验证
│
├── hyperparam_searcher: HyperparameterSearcher
│   └── Phase H 超参数联合网格搜索
│
├── transformer_2d: TwoDTransformer ✅
│   └── dict_learner: CoupledDictionaryLearner ✅
│
├── full_slice_transformer: FullSliceTransformer ✅
│   └── Phase 1.5 完整切片变换 (transform_full_slice_direct) ⭐ v3.4
│
├── transformer_3d: ThreeDTransformer ⏳
│   └── Phase 2 3D 融合
│
├── enhancer: FullDomainEnhancer ⏳
│   └── Phase 3 全域增强
│
└── visualizer: SDLVisualizerV34 ✅
    └── 全流程可视化（含 cartopy 海岸线）
```

### 14.2 主入口 (v3.4)

```python
def main():
    """v3.4 支持三种运行模式"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-search', action='store_true',
                        help='运行 K × λ 联合网格搜索')
    parser.add_argument('--use-optimal', action='store_true',
                        help='使用最优配置')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式: 只训练 4 个代表性深度')
    args = parser.parse_args()

    config = SDLConfigV34()
    sdl_fusion = SDLFusionV34(config)

    if args.grid_search:
        # 模式 B: 运行网格搜索 (Phase H)
        results = sdl_fusion.run_grid_search()
    elif args.use_optimal:
        # 模式 C: 加载最优配置
        sdl_fusion.load_optimal_config()
        results = sdl_fusion.run_full_pipeline(use_all_depths=not args.quick)
    else:
        # 模式 A: 直接使用代码中配置的参数（默认）
        # 参数在 Phase1Config 中配置，无需命令行传递
        use_all_depths = not args.quick  # 论文方法 vs 快速模式
        results = sdl_fusion.run_full_pipeline(use_all_depths=use_all_depths)
```

### 14.3 完整运行流程

```python
# 运行流程
results = sdl_fusion.run_full_pipeline(
    run_phase_minus1=True,    # 数据预处理（直接提取 Patches）
    run_phase0=True,          # 字典验证
    run_phase_h=False,        # ⭐ 超参数搜索 (可选)
    run_phase1=True,          # 2D Patch 级验证
    run_phase1_5=True,        # ⭐ 完整切片变换
    run_phase2=False,         # 3D 融合（待开发）
    run_phase3=False,         # 全域增强（待开发）
    params=['vs'],            # 处理参数
    target_model='fwea',      # 目标模型
    stitch_method='gaussian'  # 拼接方法
)
```

---

## 15. 核心数据类定义

### 15.1 Phase 0 数据类

```python
@dataclass
class ValidationResult:
    """Phase 0: 字典验证结果"""
    depth_km: float
    model_name: str
    n_patches: int
    dictionary: np.ndarray        # (K, L)
    coefficients: np.ndarray      # (N, K)
    patches_original: np.ndarray
    patches_reconstructed: np.ndarray
    coords: np.ndarray
    velocity_slice: np.ndarray
    lat_coords: np.ndarray
    lon_coords: np.ndarray
    error_stats: Dict[str, float]
    passed: bool
    mean_patch: np.ndarray
    patch_size: int
```

### 15.2 Phase 1 数据类

```python
@dataclass
class PairedPatchResult:
    """配对 Patch 提取结果"""
    patches_base: np.ndarray      # (N, L₁)
    patches_highres: np.ndarray   # (N, L₂)
    coords_latlon: np.ndarray     # (N, 2)
    correlations: np.ndarray      # (N,)
    valid_mask: np.ndarray        # (N,)
    n_total: int
    n_valid: int
    physical_window_size: float
    base_patch_size: int
    highres_patch_size: int
    depth_km: float

@dataclass
class CoupledDictionaryResult:
    """耦合字典学习结果"""
    D1: np.ndarray                # (K, L₁)
    D2: np.ndarray                # (K, L₂)
    C_train: np.ndarray           # (N_train, K)
    C_val: np.ndarray             # (N_val, K)
    preprocessing: PreprocessingInfo
    transform_error: float
    baseline_error: float
    relative_improvement: float
    dict_learner: DictionaryLearning
    depth_km: float
```

### 15.3 Phase 1.5 数据类

```python
@dataclass
class FullSliceTransformResult:
    """Phase 1.5: 完整切片变换结果"""
    depth_km: float
    target_model: str
    param: str
    # 完整切片
    transformed_slice: np.ndarray     # SDL 变换后的完整切片
    baseline_slice: np.ndarray        # Baseline 上采样切片
    target_slice: np.ndarray          # 高分辨率目标切片 (GT)
    original_slice: np.ndarray        # 原始低分辨率切片
    weight_map: np.ndarray            # 权重分布图
    # 坐标
    lat_coords: np.ndarray
    lon_coords: np.ndarray
    # 误差统计
    sdl_error: float                  # SDL 全域误差
    baseline_error: float             # Baseline 全域误差
    relative_improvement: float       # 相对改善率
    error_mean: float                 # 误差均值
    error_std: float                  # 误差标准差
    error_distribution: Dict[str, float]
    # 处理统计
    n_patches_processed: int
    n_patches_skipped: int
    coverage_ratio: float             # 覆盖率
    # 验收
    passed: bool
    stitch_method: str                # 'gaussian' or 'average'
```

---

## 16. 配置参数详解

### 16.1 Phase 0 配置

```python
@dataclass
class Phase0Config:
    # 字典参数
    n_components: int = 20      # K = 20 原子
    alpha: float = 0.1          # λ = 0.1 稀疏性
    n_nonzero_coefs: int = 5    # OMP 非零系数数

    # 窗口配置
    physical_window_size: float = 4.0  # 4°×4° 物理窗口
    stride_ratio: float = 0.25         # 步长比例

    # 验收标准
    max_mean_error: float = 3.0        # 平均误差 < 3%
    max_point_error: float = 10.0      # 最大点误差 < 10%

    # 测试深度
    validation_depths: List[float] = [40, 100, 200, 400]
```

### 16.2 Phase 1/1.5 配置 (v3.4 直接提取方法)

```python
@dataclass
class Phase1Config:
    # 物理窗口配置
    physical_window_size: float = 6.0    # 6°×6° 物理窗口 (M₁: 6×6 格点 @1°)
    physical_stride: float = 1.0         # 物理步长 1° (论文: stride=1 格点)
    # 注: 论文 stride=格点数，我们用 physical_stride=物理步长(度)
    # M₁ stride=1 格点 × 1°/格点 = 1° 物理步长

    # 字典参数 ⭐ 网格搜索参数
    n_components: int = 20               # K = 20 原子 (论文: 20)
    alpha: float = 0.1                   # λ = 0.1 稀疏性 (论文: 0.1)
    n_nonzero_coefs: int = 5             # OMP 非零系数数
    # 搜索范围: K=[10, 15, 20, 25, 30], λ=[0.05, 0.1, 0.15, 0.2]

    # D₂ 计算 ⭐ 固定为论文方法
    d2_regularization: float = 0.0       # 固定为 0，纯最小二乘

    # 训练参数
    train_ratio: float = 0.8             # 80% 训练集
    min_cc_threshold: float = 0.0        # CC > 0 筛选
    max_pairs: int = 30000               # 最大训练样本数

    # 验收标准
    max_transform_error: float = 10.0    # 变换误差 < 10%
    min_relative_improvement: float = 20.0  # 相对改善 > 20%
    max_error_mean: float = 1.0          # 误差均值 < 1%
    max_error_std: float = 5.0           # 误差标准差 < 5%

    # 测试深度（快速模式使用）
    target_depths: List[float] = [40, 100, 500, 800]  # 覆盖浅层到深层
```

**v3.4 参数说明**:

| 参数                  | 值            | 说明                                    |
| --------------------- | ------------- | --------------------------------------- |
| physical_stride       | 1°            | 物理步长 1°（直接从原始 1° 网格提取）   |
| 等效 M₁ stride        | 1 格点        | 与论文一致 (stride=1 格点)              |
| 重叠率                | 5/6 ≈ 83%     | (6-1)/6 = 83%，与论文一致               |
| max_pairs             | 30000         | 限制训练样本，平衡精度与速度            |
| **extraction_method** | **direct** ⭐ | v3.4 只保留直接提取（移除统一网格方法） |

### 16.3 超参数联合网格搜索 ⭐

**搜索参数总览**:

| 参数分类     | 参数                   | 默认值     | 论文值                 | 搜索范围             | 状态      |
| ------------ | ---------------------- | ---------- | ---------------------- | -------------------- | --------- |
| **几何参数** | physical_window        | **4°** ⭐  | 0.18°                  | 2°, 3°, 4°, 6°       | 🔍 搜索   |
|              | physical_stride (物理) | **1°** ⭐  | 0.03° (=1 格点 @0.03°) | 0.5°, 1°, 2°, 3°     | 🔍 搜索   |
|              | _(M₁ stride 格点)_     | _(1 格点)_ | **1 格点** (论文原义)  | _(0.5, 1, 2, 3)_     | -         |
| **字典参数** | n_components (K)       | 20         | 20                     | 10, 15, 20, 25, 30   | 🔍 搜索   |
|              | alpha (λ)              | 0.1        | 0.1                    | 0.05, 0.1, 0.15, 0.2 | 🔍 搜索   |
| **D₂ 计算**  | d2_regularization      | **0** ⭐   | 0 (lstsq)              | **固定为 0** (论文)  | ✅ 已固定 |
| **提取方法** | extraction_method      | direct     | direct                 | 固定为 direct        | ✅ 已固定 |

> ⚠️ `d2_regularization` 已固定为 0（纯最小二乘），与论文保持一致，不参与网格搜索

**联合网格搜索策略** (类似论文 ~40 种组合):

```python
# 推荐搜索配置
search_config = {
    'window_stride_pairs': [
        (4.0, 1.0),   # 当前最优
        (2.0, 0.5),   # 小窗口密集
        (6.0, 1.0),   # 大窗口密集
    ],
    'K_values': [15, 20, 25, 30],         # 4 种
    'alpha_values': [0.05, 0.1, 0.15],    # 3 种
}
# 总组合: 3 × 4 × 3 = 36 种
```

**最优配置选择标准** (论文 Section 3.2):

$$\text{Best} = \arg\min_{K, \lambda} D(\hat{P}^V, P_2^V)$$

代码实现使用加权多目标评分:

```python
def compute_score(result):
    error_score = 1.0 - min(transform_error / 20.0, 1.0)
    improve_score = min(relative_improvement / 50.0, 1.0)
    std_score = 1.0 - min(error_std / 10.0, 1.0)
    return 0.4 * error_score + 0.4 * improve_score + 0.2 * std_score
```

**输出文件**:

```
results/SDL_grid_search/
├── grid_search_results.csv         # 所有搜索结果
├── optimal_config.json             # 最优配置
└── depth_optimal_configs.json      # 各深度最优配置

figures/SDL_Fusion_v3.2/
├── H01_grid_search_heatmap.jpg     # K×λ 热力图
├── H02_param_sensitivity.jpg       # 参数敏感性曲线
└── H03_optimal_config.jpg          # 最优配置详情
```

**详细搜索实现**: 参见 `plan-sdlFusion.prompt.md` > "超参数联合网格搜索"

---

## 17. 公式速查表

| 公式编号 | 含义           | 数学表达式                                    | 代码实现                            |
| -------- | -------------- | --------------------------------------------- | ----------------------------------- |
| (1)      | Patch 稀疏表示 | $p_j \approx D^* c_j$                         | `p = D @ c`                         |
| (3)      | 优化目标       | $\min \|P - DC\|_2^2 + \lambda\|C\|_0$        | `DictionaryLearning()`              |
| (6)      | D₁, C 训练     | $\min \|P_1^T - D_1 C\|_2^2 + \lambda\|C\|_0$ | `dict_learner.fit(P1_train)`        |
| (7)      | D₂ 计算        | $\min \|P_2^T - D_2 C^T\|_2^2$                | `np.linalg.lstsq(C, P2)` ⭐         |
|          |                | ✅ **固定为纯最小二乘，无正则化**             | `d2_regularization=0` (固定)        |
| (8)      | 验证编码       | $C^V = \text{encode}(P_1^V)$                  | `dict_learner.transform(P1_val)`    |
| (9)      | 相对误差       | $D(P, P_0) = \|P - P_0\|_2 / \|P_0\|_2$       | `np.linalg.norm(P - P0) / norm(P0)` |
| 网格搜索 | 最优参数选择   | $\arg\min_{K,\lambda} D(\hat{P}^V, P_2^V)$    | `HyperparameterSearcher.search()`   |

---

## 18. 参考文献

### 17.1 核心参考

1. **Zhang, H., & Ben-Zion, Y. (2024)**. _Enhancing Regional Seismic Velocity Models With Higher-Resolution Local Results Using Sparse Dictionary Learning_. Journal of Geophysical Research: Solid Earth, 129, e2023JB027016.

   - SDL 方法在地震学中的应用
   - 耦合字典学习的核心方法

2. **Dong, C., Loy, C. C., He, K., & Tang, X. (2015)**. _Image Super-Resolution Using Deep Convolutional Networks_. IEEE TPAMI, 38(2), 295-307.

   - Patch 拼接方法参考
   - **关键引用**:
     - Section 3.1.3: "overlapping patches are often **averaged**"
     - Footnote 5: "patches overlapped with 4 pixels at each direction"
   - **注意**: 传统方法使用简单平均，我们选择高斯加权以获得更平滑过渡

3. **Aharon, M., Elad, M., & Bruckstein, A. (2006)**. _K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation_. IEEE TSP, 54(11), 4311-4322.
   - 字典学习的经典算法

### 17.2 论文 Figure 对应表

#### 主文 Figure

| 论文 Figure  | 内容                | 代码可视化   | 状态 |
| ------------ | ------------------- | ------------ | ---- |
| Figure 1     | 字典表示示意图      | V05 (atoms)  | ✅   |
| Figure 2     | 字典重建验证        | V02          | ✅   |
| **Figure 3** | **2D 完整切片变换** | **F01, F02** | ✅   |
| Figure 4     | 3D Vp 多深度        | 3D02         | ⏳   |
| Figure 5     | 3D Vs 多深度        | 3D02         | ⏳   |

#### 附录 Figure (Supplementary Material) ⭐

| 论文 Figure   | 内容               | 关键发现                       | 代码可视化 | 状态 |
| ------------- | ------------------ | ------------------------------ | ---------- | ---- |
| **Figure S1** | **D₁/D₂ 字典原子** | D₁-D₂ 相关性范围 -0.28~0.83    | T04, T05   | ✅   |
| **Figure S2** | **2D 重建散点图**  | 重建差异与模型差异正相关       | ⏳ 待实现  | ⏳   |
| **Figure S3** | **3D 重建散点图**  | 重叠区域外也有增强效果         | ⏳ 待实现  | ⏳   |
| **Figure S4** | **误差分布直方图** | 均值 0.4%, 标准差 3%, 正态分布 | F04 (部分) | 🔄   |

**Figure S2/S3 可视化建议** (待实现):

```python
# 散点图：验证 SDL 是否将 M₁ 向 M₂ 方向调整
fig, ax = plt.subplots()
# X 轴: M₂ - M₁ (目标与原始的差异)
# Y 轴: Reconstruction - M₁ (重建与原始的差异)
ax.scatter(M2 - M1, Reconstruction - M1, c='k', alpha=0.3, label='Overlap')
ax.scatter(M2_outside - M1_outside, Recon_outside - M1_outside, c='r', alpha=0.3, label='Outside')
ax.plot([vmin, vmax], [vmin, vmax], 'r--', label='Perfect transfer')
# 如果点沿对角线分布 → SDL 成功将 M₁ 向 M₂ 方向调整
```

---

**文档版本**: v3.4.2  
**创建日期**: 2026-01-08  
**最后更新**: 2026-01-09  
**作者**: EASTASIA-FWI Team

> ### v3.4.2 核心更新 ⭐ 与论文完全一致！
>
> 1. **移除 Centering**: 不再对 patches 做减均值预处理，与论文原始方法一致
> 2. **深度精确匹配**: 只使用 SinoScope (20km) 和 FWEA (10km) 深度精确匹配的层（论文: "at the same depths"）
> 3. **新增 `get_matched_depths()` 方法**: 自动计算两个模型深度匹配的层
>
> ### v3.4.1 核心更新
>
> 1. **SinoScope1.0 全域变换**: 输出网格覆盖 SinoScope1.0 的完整范围
> 2. **坐标对齐修复**: 使用 `RegularGridInterpolator` 基于坐标插值
> 3. **评估逻辑修正**: 误差评估只在重叠区域进行
>
> ### v3.4 核心更新
>
> 1. **移除统一网格方法**: 只保留 `DirectPatchExtractor`
> 2. **修复训练/变换一致性**: Phase 1.5 使用 `transform_full_slice_direct()`

> 💡 **配套文档**: 具体的测试评估和迭代计划请参见 `plan-sdlFusion.prompt.md`
>
> **历史版本**:
>
> - v3.4.2: 移除 centering，深度精确匹配（与论文完全一致）
> - v3.4.1: SinoScope1.0 全域变换，输出网格覆盖完整范围
> - v3.4: 移除统一网格方法，只保留直接提取方法
> - v3.3.5: 论文深度处理方法（所有深度独立训练）
> - v3.3.4: 简化网格搜索 + cartopy 海岸线
> - v3.3.1~v3.3.3: 40km 达标、stride 定义更正

```
┌─────────────────────────────────────────────────────────────────────┐
│ SDL 变换流程（基于 Figure 1）                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ Step 1: 在重叠区域提取配对 patches                                     │
│ P₁ (from M₁) ←→ P₂ (from M₂)                             │
│ 同一物理位置，不同分辨率
│                                                                     │
│ Step 2: 训练 D₁ 和稀疏系数 C                                           │
│ min ‖P₁ - D₁C‖² + λ‖C‖₀                                              │
│                                                                      │
│ Step 3: 用 C 和 P₂ 计算 D₂（最小二乘）                                 │
│ D₂ = argmin ‖P₂ - D₂C‖²                                              │
│                                                                     │
│ Step 4: 变换！对于 M₁ 任意位置的 patch p₁:                             │
│ 1. 编码: c = encode(p₁) using D₁                                    │
│ 2. 解码: p̂₂ = D₂ @ c ← 变换到高分辨率！                                │
│                                                                     │
│ 核心约束: p₁ 和 p₂ 共享相同的稀疏系数 c                                  │
│ → 建立 D₁ 和 D₂ 原子的一一对应关系                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```
