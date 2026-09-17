# SDL 融合核心概念速记 (Cheat Sheet)

## 🎯 融合目标一句话

**将 FWEA23(0.25°) 的高分辨率特征"风格"迁移到 SinoScope1.0(1°) 全域**
- 保持 1° 覆盖范围 ✓
- 获得高分辨率细节 ✓
- 不依赖人工参数 ✓

---

## 📐 关键数字记忆

| 项目 | 数值 | 含义 |
|------|------|------|
| **分辨率比** | 4:1 | 1° vs 0.25° |
| **字典原子数 K** | 20 | 提取 20 个基础特征模式 |
| **稀疏性 λ** | 0.1 | 每个 patch 只用 5 个非零系数 |
| **窗口大小** | 6° × 6° | 物理尺度（对应主要构造特征） |
| **物理步长** | 1° | M₁、M₂ 都是 1° 物理步长 |
| **改善率** | 30-40% | 相对简单插值的改善 |
| **验证误差** | 8-10% | 在新数据上的重建误差 |
| **误差分布** | σ ≈ 3% | 正态分布，无偏 |

---

## 🔄 三步融合框架

```
STEP 1: 提取   → 从重叠区提取配对 patches (6×6 vs 24×24)
STEP 2: 学习   → 训练耦合字典 D₁ ↔ D₂ (共享系数 C)
STEP 3: 变换   → 对 SinoScope 全域进行编码-解码
```

---

## 🧠 核心物理直觉

### 为什么有效？

```
物理现象      → 低分辨率表现   → 高分辨率表现
俯冲带        →  6×6 格点异常  → 24×24 格点细节结构
软流圈        →  扩散边界     → 层状细节
板内弱带      →  低速异常     → 复杂的非均匀结构

学到的 D₁/D₂ 对应关系 = 这些物理特征的多尺度转换规律
```

### 全域增强的秘密

```
在 FWEA23 有数据的地方学到的规律
  ↓
是"普遍的地学规律"（不仅特定于该地区）
  ↓
可外推到 SinoScope 其他无 FWEA23 数据的地方
  ↓
该地区虽无高分辨率参考，但获得相同的"高分辨率风格增强"
```

---

## 🔢 数学公式一览

| 编号 | 公式 | 含义 |
|-----|------|------|
| **(3)** | $\min_D \|P - DC\|_F^2 + \lambda\|C\|_0$ | 字典学习：最小重建误差 + 稀疏约束 |
| **(4-5)** | $P_1≈D_1C, P_2≈D_2C$ | 耦合字典：共享系数 C |
| **(6)** | $D_1,C=\arg\min\|P_1^T-D_1C\|_F^2+λ\|C\|_0$ | 训练低分辨率字典与系数 |
| **(7)** | $D_2=\arg\min\|P_2^T-D_2C^T\|_F^2$ | 计算高分辨率字典 (最小二乘) |
| **(9)** | $D(P,P_0)=\|P-P_0\|_2/\|P_0\|_2\times100\%$ | 相对误差度量 |

---

## 📊 关键参数对比

### 论文参数 (南加州)
```
M₁ 分辨率: 0.03° (CVM-S4.26)
M₂ 分辨率: 0.01° (Z2015) 
分辨率比: 3:1
最优 K: 20
最优 λ: 0.1
改善率: 34% (2D), 58% (3D)
```

### 东亚参数 (我们的场景)
```
M₁ 分辨率: 1.0° (SinoScope1.0)
M₂ 分辨率: 0.25° (FWEA23)
分辨率比: 4:1 (略大，但相似)
推荐 K: 20 (相同)
推荐 λ: 0.1 (相同)
预期改善率: 25-40% (考虑更多样本)
```

---

## 🎬 具体执行流程（伪代码）

```python
# ==================== PHASE 1: 准备 ====================
for depth in matched_depths:  # [0, 20, 40, ..., 100km]
    
    # 提取配对 patches
    base_slice = SinoScope.get_slice(depth)           # (nlat, nlon)
    highres_slice = FWEA23.get_slice(depth)           # (nlat, nlon)
    
    for position in sliding_window(6°×6°, stride=1°):
        patch1 = extract(base_slice, position, 6°, 1°)       # 6×6 格点 → 36 值
        patch2 = extract(highres_slice, position, 6°, 0.25°) # 24×24 格点 → 576 值
        
        cc = correlation(patch1, patch2)
        if cc > 0:
            paired_patches.append((patch1, patch2))

# ==================== PHASE 2: 学习 ====================
# 训练集/验证集: 88% / 12%

# Step A: 学习 D₁ 和 C (OMP算法)
dict_learner = DictionaryLearning(K=20, lambda=0.1)
dict_learner.fit(P1_training)  # P1_training: (7000, 36)
D1 = dict_learner.components_              # (20, 36)
C_training = dict_learner.transform(P1_training)  # (7000, 20)

# Step B: 计算 D₂ (最小二乘)
D2 = np.linalg.lstsq(C_training.T, P2_training.T)[0].T  # (20, 576)

# Step C: 验证
C_valid = encode_with_D1(P1_valid)
error_valid = ||P2_valid - D2 @ C_valid|| / ||P2_valid||
print(f"验证误差: {error_valid*100:.1f}%")  # 期望 <10%

# ==================== PHASE 3: 变换 ====================
for depth in matched_depths:
    base_slice = SinoScope.get_slice(depth)  # 全切片！
    
    result_grid = zeros_like(base_slice)
    weight_grid = zeros_like(base_slice)
    
    for position in sliding_window(6°×6°, stride=1°):
        # 提取原始 patch
        patch_orig = extract(base_slice, position, 6°, 1°)  # (6, 6)
        
        # Step 1: 编码
        c = OMP_encode(patch_orig.flatten(), D1, k=5)  # (20,)
        
        # Step 2: 解码 (高分辨率空间)
        patch_highres = (D2 @ c).reshape(24, 24)  # (24, 24)
        
        # Step 3: 下采样回 1°
        patch_enhanced = zoom(patch_highres, scale=1/4)  # (6, 6)
        
        # Step 4: 累积加权
        weight = gaussian_weight(patch_enhanced)
        result_grid[position] += weight * patch_enhanced
        weight_grid[position] += weight
    
    # Step 5: 归一化
    enhanced_slice = result_grid / weight_grid
    output.write_slice(enhanced_slice, depth)

print("✅ 融合完成！输出: fusion_enhanced_SinoScope.nc")
```

---

## ⚡ 关键决策点

### Q1: 为什么不用 CNN？
**A:** 论文明确说 "CNN 需要大量数据，我们数据有限。SDL 更可解释、泛化更好。"

### Q2: 为什么需要共享系数 C？
**A:** 这是耦合字典的核心约束。同一物理位置，两种分辨率共享相同的"特征强度"，只是表现方式不同。

### Q3: 为什么验证集很重要？
**A:** 防止过度拟合。测试模型在新数据上的真实性能，而非只看训练误差。

### Q4: 如果改善率 < 10% 怎么办？
**A:** 运行 `--grid-search` 找最优 K 和 λ，或检查数据质量/坐标对齐。

### Q5: 输出分辨率真的没变吗？
**A:** 输出仍是 1°（SinoScope 原分辨率），但**内容**更接近高分辨率风格。不是超分辨率！

---

## 🚨 常见错误及纠正

| 错误理解 | 正确理解 |
|---------|---------|
| "SDL 生成高分辨率数据" | SDL 保持原分辨率，只改变特征风格 |
| "所有区域改善相同" | 重叠区改善大(30-40%)，边界区较小(20-25%) |
| "可用任意 K 和 λ" | 需网格搜索找最优值（论文用 40 组合） |
| "深度可随意选择" | 必须两模型都有数据的深度 |
| "D₁ 和 D₂ 原子相同" | 原子不同，但编码相同的物理特征 |

---

## 📈 预期结果指标

### 成功标志
```
✓ 改善率 > 20%
✓ 验证误差 < 12%
✓ 输出无 NaN 或异常值
✓ 误差分布正态，μ ≈ 0
✓ 可视化显示平滑过渡，无界
✓ 与 FWEA23 相关性 r > 0.7
```

### 可视化检查
```
F01: 低分辨率原始字典 D₁（20 个 6×6 原子）
F02: 高分辨率字典 D₂（20 个 24×24 原子）
F03: 原始 patch 分布
F04: 变换前后对比（重叠区）
F05: 改善率空间分布
F06: 误差直方图
```

---

## 🔗 代码关键函数

| 函数 | 作用 | 位置 |
|------|------|------|
| `DirectPatchExtractor.extract_paired_patches_direct()` | 提取配对 patches | Phase 1 |
| `CoupledDictionaryLearner.train()` | 训练 D₁, D₂, C | Phase 2 |
| `FullSliceTransformer.transform_full_slice_direct()` | 完整变换 | Phase 1.5 |
| `HyperparameterSearcher.search_grid()` | 网格搜索 | Phase H |
| `SDLVisualizerV31.visualize_*()` | 可视化结果 | 所有 phase |

---

## 📚 学习路径推荐

### 初学者
1. 读这份 Cheat Sheet ← 你在这
2. 运行 `python 4_3_SDL_Fusion_v4.py` (模式 A)
3. 查看可视化结果 (F01-F06)
4. 阅读 `SDL_FWEA23融合流程详解.md`

### 进阶用户
1. 理解论文数学部分 (公式3-7)
2. 运行 `--grid-search` 找最优参数
3. 修改参数进行敏感性测试
4. 对比不同配置的结果

### 研究开发
1. 阅读完整论文 + 设计文档
2. 实现 3D 耦合字典学习 (Phase 2)
3. 进行波形验证
4. 论文发表或开源贡献

---

## 🎓 论文关键句摘录

> **问题定义**  
> "简单嵌入方法依赖人为平滑参数，且丢失低分辨率模型的有用信息。"

> **核心创新**  
> "我们建立高低分辨率模型间的变换关系，并应用到覆盖范围外的区域。"

> **耦合字典**  
> "同一物理位置的 patches 共享相同的稀疏系数，建立 D₁ 和 D₂ 的一一对应。"

> **验证结论**  
> "重建模型更平滑、更一致，误差分布呈正态，无偏。"

> **全域增强**  
> "学到的变换可外推到高分辨率模型无覆盖的区域。" ⭐ **核心**

---

## 💾 输出文件清单

```
fusion_enhanced_SinoScope.nc
  ├─ 与原始 SinoScope 相同的结构
  ├─ 但所有值都经过 SDL 增强
  └─ 可直接用于 SPECFEM3D 正演

results/<timestamp>/
  ├─ improved_metrics.json (改善率、误差等)
  ├─ F01-F06.png (Phase 1.5 可视化)
  ├─ H01-H05.png (超参数搜索结果，仅 --grid-search)
  └─ optimal_config.json (最优参数，仅 --grid-search)

logs/
  └─ fusion.log (详细执行日志)
```

---

## 最后的话

**SDL 融合的本质**：不是"升级"，而是"风格迁移" 🎨

- 输入: 低分辨率全球模型 (SinoScope)
- 学习: 如何用高分辨率特征的"笔触"表现它
- 输出: 相同分辨率，但包含高分辨率"风格"的增强模型

这正是论文标题所说的：**用稀疏字典学习增强区域模型** ✨
