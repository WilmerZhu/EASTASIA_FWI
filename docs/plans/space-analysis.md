## 任务：开发地震模型空间聚类分析模块

### 目标
开发一个完整的地震速度模型聚类分析系统，包括聚类算法实现和可视化功能。

### 开发策略
1. **先设计后编码**：首先讨论并确定整体架构和实现逻辑
2. **渐进式开发**：从核心聚类功能开始，逐步完善各个模块

### 项目结构与需求

#### 1. 核心聚类模块
- **主文件**：`2_1_Model_clustering.py`
- **配置文件**：`clustering_config.py` 
  - config的处理请参考之前模块的例子 `download_config.py` 和 `base_config.py` 
- **参考代码**：
  - `backup/Seis_Model_Clustering/` - 已有的聚类实现
  - `backup/K-Means/` - K-means 算法参考
- **功能要求**：
  - 对速度模型的多个特征vp(km/s),vsv(km/s),vsh(km/s),vs_voigt(km/s),vs(km/s),density(kg/m3)一起进行聚类分析
  - 支持多种聚类算法
  - 可配置的相似度度量方法
  - 处理缺失值（NaN）的策略
  - 批量处理多个模型文件
  - 可视化要是独立的代码# Plan: 稀疏字典学习(SDL)速度模型融合模块 v3.0

> **更新日期**: 2025-12-09
> **版本**: v3.0 - 基于实际运行结果和论文深度分析更新

## 📋 项目背景

基于论文《Enhancing Regional Seismic Velocity Models With Higher-Resolution Local Results Using Sparse Dictionary Learning》，实现东亚地区三个速度模型的SDL融合：

| 模型 | 分辨率 | 覆盖范围 | 角色 |
|------|--------|----------|------|
| **SinoScope1.0** | 1°×1°×20km | 最广（80°-150°E, 10°-55°N） | 基础模型 M₁（待增强） |
| **EARA2024** | 0.25°×0.25°×10km | 较小但精细 | 高分辨率模型 M₂（参考） |
| **FWEA23** | 0.25°×0.25°×10km | 较小但精细 | 高分辨率模型 M₃（参考） |

**核心目标**：以SinoScope1.0的广覆盖为基础，利用EARA2024/FWEA23的高分辨率细节，通过耦合字典学习实现全区域增强型融合模型。

### 论文实验结果参考

| 实验 | Patch尺寸 | 字典原子K | λ值 | 训练/验证 | 重建误差 |
|------|-----------|----------|------|-----------|----------|
| 2D单层表示 | 3×3 | 20 | 0.1 | 616 patches | 2.2% |
| 2D模型变换 | 6×6 / 18×18 | 20 | 0.1 | 1000/131 | 6.1% |
| 3D模型变换 | 9×9×20 | 20 | - | 800/180 | Vp:2.94%, Vs:3.41% |

---

## ⚠️ 当前实现状态与问题分析

### 已完成功能 ✅

| 模块 | 状态 | 说明 |
|------|------|------|
| 多分辨率网格管理 | ✅ 完成 | `MultiResolutionGrid` 类管理各模型原始网格 |
| 配对Patch提取 | ✅ 完成 | 从原始网格提取，6×6(1°) 对应 24×24(0.25°) |
| 耦合字典学习 | ✅ 完成 | DictionaryLearning + OMP，K=20, λ=0.1 |
| 稀疏性控制 | ✅ 完成 | `n_nonzero_coefs=5`，平均NNZ=5-7.6/20 |
| 可视化系统 | ✅ 完成 | 11张图表（覆盖对比、配对Patch、字典原子等） |
| 统计输出 | ✅ 完成 | JSON统计报告 + NetCDF融合模型 |

### 当前运行结果（2025-12-09）

```json
{
  "vs_100km": {
    "train_error_base": 41.85%,
    "train_error_highres": 48.20%,
    "val_error_base": 90.36%,
    "val_error_highres": 103.81%,  // ⚠️ 远超论文6%目标
    "n_train": 208, "n_val": 53
  },
  "vs_200km": {
    "train_error_base": 45.02%,
    "train_error_highres": 56.87%,
    "val_error_base": 106.75%,
    "val_error_highres": 118.69%,
    "n_train": 193, "n_val": 49
  }
}
```

### 🔴 核心问题诊断

| 问题 | 论文情况 | 我们的情况 | 影响 |
|------|---------|-----------|------|
| **模型关系** | 同一区域不同分辨率成像 | 独立构建的三个模型 | 无内在结构对应 |
| **P1-P2相关性** | 隐含高相关（同源） | CC=0.47-0.61（差异大） | 字典迁移失效 |
| **上采样方法** | 双三次插值M1→M2分辨率 | 块平均下采样P2→P1分辨率 | 可能损失信息 |
| **重建误差** | 6.1%（验证集） | 100+%（验证集） | 方法不适用 |

### 根本原因分析

论文SDL方法的**前提假设**：
> "The dictionary representation technique can facilitate the transformation between two different seismic velocity models by establishing a correspondence between their respective dictionaries."

这要求两个模型在相同位置具有**结构一致性**（相同地质特征的不同分辨率表达）。

我们的三个模型是**独立反演**得到的，在相同位置可能有：
- 不同的速度异常幅度
- 不同的结构边界位置
- 不同的深度分辨率

**结论**：SDL方法更适合"同模型升级"而非"多模型融合"

---

## 📐 论文算法原理（详细）

### 第一步：字典表示 (Section 2.1)

速度模型 → 滑动窗口提取patches → 展平为向量：

$$\mathbf{p}_j \approx \mathbf{D}^* \mathbf{c}_j = \sum_{i=1}^{K} c_{i,j} \mathbf{d}_i \quad \text{(公式1)}$$

优化问题：

$$\mathbf{D}^*, \mathbf{C}^* = \arg\min_{\mathbf{D},\mathbf{C}} \|\mathbf{P} - \mathbf{DC}\|_2^2 + \lambda\|\mathbf{C}\|_0 \quad \text{(公式3)}$$

### 第二步：耦合字典学习 (Section 2.2)

从**相同空间位置**提取配对patches $\mathbf{P}_1^U$（低分辨率）和 $\mathbf{P}_2$（高分辨率）：

$$\mathbf{P}_1 \approx \mathbf{D}_1 \mathbf{C} \quad \text{(公式4)}$$
$$\mathbf{P}_2 \approx \mathbf{D}_2 \mathbf{C} \quad \text{(公式5)}$$

**关键**：两个字典共享稀疏系数 $\mathbf{C}$，建立模型间的结构对应关系。

### 第三步：模型变换 (公式6-7)

1. 用 $\mathbf{D}_1$ 对低分辨率模型全域patches编码：$\mathbf{C}^U = \text{OMP}(\mathbf{P}_1^U, \mathbf{D}_1)$
2. 用 $\mathbf{D}_2$ 重建得到增强patches：$\hat{\mathbf{P}} = \mathbf{D}_2 \mathbf{C}^U$
3. 拼接重建patches恢复增强后的速度模型

### 训练/验证集划分

$$\mathbf{C}^V = \arg\min_{\mathbf{C}} \|\mathbf{P}_1^V - \mathbf{D}_1\mathbf{C}\|_2^2 + \lambda\|\mathbf{C}\|_0 \quad \text{(公式8)}$$

$$D(\mathbf{P}, \mathbf{P}_0) = \frac{\|\mathbf{P} - \mathbf{P}_0\|_2}{\|\mathbf{P}_0\|_2} \quad \text{(公式9)}$$

### 论文关键细节（图片分析补充）

1. **CC>0过滤**: "We only utilize patch pairs with CC > 0 for both training and validation" (Section 3.2)
2. **上采样验证**: "To evaluate the performance of trained dictionaries, we upsample M₁ to the same resolution as M₂ using bicubic interpolation"
3. **3D方法优势**: 9×9×20 patches保持深度连续性，误差仅2.94-3.41%
4. **同源模型**: 论文使用CVM-S4.26 + Fang2019/Zigone2015，都是南加州同区域模型

---

## 🚀 理论改进方案

### 方案A：提高CC阈值过滤（保守方案）

**思路**：只在高相关区域应用SDL，其他区域使用BMA/Voting

```python
# 当前：min_correlation = 0.0
# 改进：
min_correlation_for_sdl = 0.6  # 只保留结构相似的patches
min_correlation_for_bma = 0.3  # 中等相关区域用BMA
# CC < 0.3的区域保持原模型
```

**优点**：
- 简单易实现
- 在有效区域可能降低误差
- 与BMA/Voting形成互补

**缺点**：
- 高CC区域可能很少
- 不能真正发挥SDL的超分辨率能力

### 方案B：多模型平均作为参考（推荐方案）

**思路**：不是M1→M2变换，而是构建"共识参考模型"

```python
# 步骤1：在公共区域，计算三模型的加权平均作为"共识模型"
consensus_model = w1 * M1_interp + w2 * M2 + w3 * M3
# 权重可基于各模型在该区域的数据约束

# 步骤2：学习各模型到共识模型的字典变换
# 这确保了字典迁移有明确的目标

# 步骤3：在只有M1覆盖的区域，用学到的D1→D_consensus变换
```

**优点**：
- 利用多模型信息构建更可靠的参考
- 可能提高结构一致性
- 理论上更适合独立模型融合

**缺点**：
- 需要重新设计算法框架
- 共识模型的权重选择需要研究

### 方案C：分区域差异化策略（实用方案）

**思路**：根据模型间差异程度选择不同融合策略

```python
def select_fusion_method(patch_cc, model_diff):
    if patch_cc > 0.7:
        return 'SDL'  # 高相关：使用SDL超分辨率
    elif patch_cc > 0.4:
        return 'BMA'  # 中相关：使用BMA概率加权
    elif model_diff < threshold:
        return 'MEAN'  # 低相关但差异小：简单平均
    else:
        return 'VOTING'  # 低相关且差异大：投票选择
```

**优点**：
- 充分利用各方法优势
- 自适应处理不同区域
- 与已有BMA/Voting模块衔接

**缺点**：
- 需要设计合理的阈值
- 边界可能需要平滑处理

### 方案D：3D耦合字典学习（论文推荐）

**思路**：利用深度连续性约束，减少逐层独立处理的误差

```python
# 论文3D patch配置
patch_size_3d = (6, 6, 10)  # 6°×6°×200km（假设20km/层）

# 3D字典学习
# 论文显示3D方法误差从6%降到3%
```

**优点**：
- 论文验证有效
- 保持深度结构连续性
- 可能更好地捕获俯冲带等3D结构

**缺点**：
- 实现复杂度高
- patches数量减少
- 需要更多计算资源

### 方案E：迭代精化策略（研究导向）

**思路**：多轮迭代，逐步收敛到一致模型

```python
for iteration in range(max_iter):
    # 轮1：用当前模型训练字典
    D1, D2, C = train_coupled_dict(P1, P2)
    
    # 轮2：变换基础模型
    M1_enhanced = transform(M1, D1, D2)
    
    # 轮3：更新P1为增强后的patches
    P1 = extract_patches(M1_enhanced)
    
    # 检查收敛
    if error < threshold:
        break
```

---

## 📊 改进优先级建议

| 优先级 | 方案 | 工作量 | 预期效果 | 建议 |
|--------|------|--------|----------|------|
| 🔴 P0 | 方案A (CC过滤) | 0.5天 | 中 | **立即实施** |
| 🔴 P0 | 方案C (分区域策略) | 1天 | 高 | **推荐实施** |
| 🟡 P1 | 方案D (3D字典) | 2天 | 高 | 验证2D后实施 |
| 🟢 P2 | 方案B (共识参考) | 3天 | 中-高 | 研究方向 |
| 🟢 P2 | 方案E (迭代精化) | 3天 | 未知 | 探索性研究 |

---

## 🔧 代码实现更新计划

### Step 1: 数据加载与网格系统 ✅ 已完成

- `MultiResolutionGrid` 类管理各模型原始网格
- 保持各模型原始分辨率，不做统一重采样
- 计算公共覆盖区域边界

### Step 2: 配对Patch提取系统 ✅ 已完成

- `PatchExtractor.extract_paired_patches_at_depth()` 
- 从原始网格提取：6×6(1°) 对应 24×24(0.25°)
- CC计算使用下采样匹配
- 支持NaN处理和有效性检查

### Step 3: 耦合字典学习 ✅ 已完成（需优化）

- `CoupledDictionaryLearner.train()` 实现
- 使用 `DictionaryLearning` + OMP算法
- 稀疏性控制：`n_nonzero_coefs=5`

**待优化**：
- [ ] 实现高CC阈值过滤（方案A）
- [ ] 添加上采样验证（论文方法）
- [ ] 实现3D patch提取（方案D）

### Step 4: 3D模型变换与融合 🔄 部分完成

**已完成**：
- `_perform_sdl_fusion()` 基础框架
- 简化版单深度融合

**待实现**：
- [ ] 完整的逐patch变换和拼接
- [ ] 区域自适应重建（公共区域/边界/仅M1区域）
- [ ] 边界平滑过渡

### Step 5: 可视化系统 ✅ 基本完成

已实现11张图表：
| 序号 | 图表 | 状态 |
|------|------|------|
| 1 | 模型覆盖范围对比 | ✅ |
| 2 | 配对Patch演示（100km） | ✅ |
| 3 | 配对Patch演示（200km） | ✅ |
| 4 | 字典原子网格 | ✅ |
| 5 | 耦合字典对应关系 | ✅ |
| 6 | 稀疏系数分析 | ✅ |
| 7 | Patch重建演示（100km） | ✅ |
| 8 | Patch重建演示（200km） | ✅ |

**待实现**：
- [ ] 变换前后对比图
- [ ] 水平/垂直切片对比
- [ ] 融合不确定性空间分布
- [ ] Voting/BMA/SDL三方法对比

### Step 6: 输出与验证 ✅ 基本完成

- JSON统计报告：`SDL_fusion_statistics.json`
- NetCDF融合模型：`SDL_fused_vs.nc`

**待实现**：
- [ ] 与BMA/Voting结果定量对比
- [ ] 生成综合评估报告

---

## ⚙️ 配置参数（当前版本）

```python
@dataclass
class SDLConfig:
    """SDL配置 - v2.2"""
    
    # === 模型分辨率 ===
    base_resolution: float = 1.0       # SinoScope1.0
    highres_resolution: float = 0.25   # EARA2024/FWEA23
    
    # === Patch配置 ===
    patch_size_base: Tuple[int, int] = (6, 6)      # 6×6格点 = 6°×6°
    patch_size_highres: Tuple[int, int] = (24, 24) # 24×24格点 = 6°×6°
    physical_size: float = 6.0  # 物理区域尺寸（度）
    stride_ratio: float = 0.5   # 50%重叠
    
    # === 字典配置 ===
    n_components: int = 20           # 字典原子数K
    alpha: float = 0.1               # 稀疏正则化λ（当前未生效）
    n_nonzero_coefs: int = 5         # OMP非零系数数
    train_ratio: float = 0.8         # 训练集比例
    min_correlation: float = 0.0     # ⚠️ 待提高到0.6
    
    # === 分析深度 ===
    depths: List[float] = [100, 200]
```

---

## 🔬 下一步工作计划

### 短期（本周）

1. **实施方案A**：将 `min_correlation` 提高到 0.6
   - 修改 `SDLDictionaryConfig.min_correlation = 0.6`
   - 重新运行，对比误差变化
   - 记录有效patches数量变化

2. **实施方案C**：设计分区域融合策略
   - 创建 `RegionalFusionSelector` 类
   - 根据CC值选择SDL/BMA/Voting
   - 集成现有BMA模块

3. **完善可视化**：添加CC分布热力图
   - 显示哪些区域CC高，适合SDL
   - 显示哪些区域需要BMA/Voting

### 中期（下周）

1. **实施方案D**：3D字典学习
   - 实现 `extract_paired_patches_3d()` 方法
   - Patch尺寸：6×6×10（6°×6°×200km）
   - 验证是否降低重建误差

2. **完整融合流程**：
   - 实现逐patch变换和拼接
   - 边界平滑过渡
   - 全域融合模型输出

### 长期（研究方向）

1. **共识参考模型**（方案B）
2. **迭代精化**（方案E）
3. **发表质量图表**

---

## 🔍 技术决策更新

### Q1: 为什么当前SDL效果不佳？

**论文前提假设**：
- 两个模型是同一区域的不同分辨率表达
- 相同位置具有结构一致性（CC隐含高）

**我们的情况**：
- 三个模型独立构建，结构差异大
- P1 vs P2_downsampled CC只有0.47-0.61
- 字典迁移失去了结构对应基础

### Q2: 高CC过滤能解决问题吗？

**理论上**：高CC区域确实具有更好的结构一致性，SDL可能有效

**实际问题**：
- 高CC区域（>0.6）可能很少
- 可能只覆盖地质结构简单的区域
- 需要验证过滤后的patches数量是否足够训练

### Q3: 3D方法为什么可能更好？

论文Section 3.3指出：
> "velocities at different depths are not entirely independent and are likely to include correlated variations. Utilizing 3D reconstruction ensures continuity and smoothness in depth"

3D patches（9×9×20）可以：
- 利用深度方向的结构连续性约束
- 减少逐层独立处理的累积误差
- 更好地捕获倾斜结构（如俯冲带）

### Q4: SDL vs BMA vs Voting的适用场景

| 方法 | 最适合场景 | 我们的情况 |
|------|-----------|-----------|
| **SDL** | 同源模型的分辨率增强 | ⚠️ 不完全适用 |
| **BMA** | 模型可信度可量化 | ✅ 适用于多模型融合 |
| **Voting** | 需要离散选择 | ✅ 适用于结构边界 |
| **分区域策略** | 复杂融合场景 | 🌟 最推荐 |

---

## ⚙️ 关键配置参数（基于论文实证）

```python
@dataclass
class SDLConfigV3:
    """SDL配置类 v3.0 - 增加CC过滤优化"""
    
    # === Patch配置（匹配物理区域） ===
    patch_size_base: Tuple[int, int] = (6, 6)         # 基础模型: 6×6格点 = 6°×6°
    patch_size_highres: Tuple[int, int] = (24, 24)    # 高分辨率: 24×24格点 = 6°×6°
    patch_size_3d: Tuple[int, int, int] = (6, 6, 10)  # 3D: 6°×6°×200km
    stride_base: Tuple[int, int] = (3, 3)             # 50%重叠
    
    # === 字典学习配置（论文验证值） ===
    n_components: int = 20           # 字典原子数K
    alpha: float = 0.1               # 稀疏正则化参数λ
    transform_algorithm: str = 'omp' # 稀疏编码算法
    n_nonzero_coefs: int = 5         # OMP非零系数数量
    
    # === CC过滤配置（新增优化） ===
    min_correlation_sdl: float = 0.6   # SDL适用阈值
    min_correlation_bma: float = 0.3   # BMA适用阈值
    # CC < 0.3: 使用Voting或保持原模型
    
    # === 训练/验证配置 ===
    train_val_ratio: float = 0.8
    random_seed: int = 42
```

---

## 📊 预期输出

### 已生成文件 ✅

| 文件 | 说明 |
|------|------|
| `4-3_model_coverage_comparison.jpg/pdf` | 三模型覆盖范围对比 |
| `4-3_paired_patches_demo_vs_100km.jpg/pdf` | 配对Patch演示(100km) |
| `4-3_paired_patches_demo_vs_200km.jpg/pdf` | 配对Patch演示(200km) |
| `4-3_dictionary_atoms_grid_vs.jpg/pdf` | 字典原子网格 |
| `4-3_coupled_dict_correspondence_vs.jpg/pdf` | 耦合字典对应关系 |
| `4-3_sparse_coefficients_analysis_vs.jpg/pdf` | 稀疏系数分析 |
| `4-3_reconstruction_demo_vs_100km.jpg/pdf` | 重建演示(100km) |
| `4-3_reconstruction_demo_vs_200km.jpg/pdf` | 重建演示(200km) |
| `SDL_fusion_statistics.json` | 统计报告 |
| `SDL_fused_vs.nc` | 融合模型NetCDF |

### 待生成文件 📋

- `4-3_cc_distribution_heatmap.jpg` - CC分布热力图
- `4-3_regional_fusion_method_map.jpg` - 区域融合方法选择图
- `4-3_sdl_bma_voting_comparison.jpg` - 三方法对比仪表板
- `4-3_vertical_cross_section.jpg` - 垂直剖面对比

---

## 📅 项目进度总结

| 阶段 | 状态 | 完成度 |
|------|------|--------|
| 数据加载 | ✅ 完成 | 100% |
| Patch提取 | ✅ 完成 | 100% |
| 字典学习 | ✅ 完成 | 90% |
| 模型融合 | 🔄 进行中 | 40% |
| 可视化 | ✅ 基本完成 | 75% |
| 方法优化 | 🔄 进行中 | 20% |

**核心发现**：SDL方法在独立构建模型融合场景效果有限，需要结合BMA/Voting形成分区域策略。

---

*更新日期：2025-12-09*
*版本：v3.0 - 基于实际运行结果和论文深度分析*
*参考论文：Zhang, H., & Ben-Zion, Y. (2024). JGR: Solid Earth, 129, e2023JB027016.*


#### 2. 可视化模块
- **文件**：`5_7_Clustering_visualization.py`
- **功能要求**：
  - 聚类结果的2D/3D可视化
  - 聚类质量评估图表
  - 交互式探索界面

#### 3. 数据格式
**输入数据路径**：`data/models/**.csv`

**CSV 格式规范**：
```
CSV 格式：longitude,latitude,depth(km),vp(km/s),vsv(km/s),vsh(km/s),vs_voigt(km/s),vs(km/s),density(kg/m3)
注意：某些模型可能只有 vp 或者 vs 或者所有参数都有，没有的参数都是 nan