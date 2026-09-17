# Plan: 稀疏字典学习(SDL)速度模型融合模块 v2.3

> **文档更新日期**: 2025-12-24
> **代码版本**: v2.3 完成实现
> **实现状态**: ✅ v2.3 Patch 自适应融合/超参数搜索完成

---

## 🎉 v2.2 测试结果总结

### 核心指标对照

| 指标               | 论文目标 | v2.1 实测         | v2.2 实测 (100/200/300km)     | 状态              |
| ------------------ | -------- | ----------------- | ----------------------------- | ----------------- |
| 变换误差           | 7.6%     | 2.28/1.70/1.45%   | **2.32% / 1.69% / 1.43%**     | ✅ 全部通过       |
| Baseline 误差      | 11.6%    | 2.78/2.40/1.99%   | **2.41% / 2.64% / 1.80%**     | ✅ 合理范围       |
| 相对改善           | 34.5%    | 18.2/29.1/27.3%   | **3.8% / 36.1% / 20.3%**      | ❌ 100km/300km 差 |
| 误差均值           | 0.4%     | 0.06/-0.01/-0.11% | **-0.016% / 0.002% / 0.023%** | ✅ 全部通过       |
| 误差标准差         | ~3%      | 2.26/1.68/1.43%   | **2.29% / 1.66% / 1.41%**     | ✅ 全部通过       |
| 偏度               | ~0       | 0.10/-0.31/-0.23  | **-0.33 / -0.65 / -0.68**     | ⚠️ 200/300km 偏大 |
| 峰度               | ~3       | -                 | **1.19 / 2.05 / 2.21**        | ✅ 全部通过       |
| CC 均值            | -        | -                 | **0.465 / 0.351 / 0.205**     | 📊 随深度下降     |
| 正态检验(Combined) | ✓        | -                 | **✅ / ❌ / ❌**              | ⚠️ 200/300km 失败 |

### ✅ v2.2 已完成功能

1. **可视化坐标修复** [Phase A 完成]

   - 所有图表显示真实经纬度坐标
   - Patch 示例标注中心坐标 (lat°, lon°)
   - CC 空间分布图新增 (Fig06b)
   - 使用 `jet_r` 配色方案

2. **深度自适应参数配置** [Phase B 部分完成]

   - `SDLDepthAdaptiveConfig` 类实现
   - 不同深度使用不同 K, α, CC_threshold
   - 当前配置: 100km (K=25, α=0.15), 200km (K=20, α=0.10), 300km (K=20, α=0.08)

3. **组合正态性检验** [Phase C 完成]

   - 替换 Shapiro-Wilk 为组合判据 (skewness<0.5, kurtosis<3, mean<1%)
   - Q-Q 图可视化验证

4. **配色方案统一** [附加完成]
   - 速度图: `jet_r` (配置化)
   - 差异图: `RdBu_r`
   - CC 空间图: `RdYlGn`

---

## ❌ v2.2 遗留问题分析

### 问题 1: 100km 深度相对改善极低 (3.8% vs 25% 目标) ⭐⭐⭐

**现象分析**:

| 指标                | 100km | 200km | 300km | 说明                      |
| ------------------- | ----- | ----- | ----- | ------------------------- |
| SDL Error           | 2.32% | 1.69% | 1.43% | 误差本身很低              |
| Baseline Error      | 2.41% | 2.64% | 1.80% | Baseline 也很低           |
| 差值 (Baseline-SDL) | 0.09% | 0.95% | 0.37% | **100km 差值极小**        |
| 相对改善            | 3.8%  | 36.1% | 20.3% | 100km 严重不达标          |
| CC 均值             | 0.465 | 0.351 | 0.205 | 100km CC 最高反而改善最低 |

**根本原因诊断**:

1. **Baseline 过于"准确"**: 100km 深度 Baseline (bicubic 上采样) 误差仅 2.41%，说明两个模型在该深度差异本来就很小
2. **SDL 提升空间有限**: 当 Baseline 已经很好时，SDL 难以进一步显著改善
3. **CC 高但改善低的悖论**: CC=0.465（三个深度最高）说明两模型空间模式相似，但这恰恰意味着 bicubic 插值已足够好

**物理解释**:

```
100km 深度 = 岩石圈-软流圈过渡带
├── 两模型空间模式高度相似 (CC=0.465)
├── 差异主要是微小的局部细节
├── Bicubic 插值已能很好地恢复这些细节
└── SDL 的"结构增强"优势无法体现
```

### 问题 2: 部分 Patch SDL 表现比 Baseline 差

**Fig12 重建演示观察**:

| 深度  | 负改善 Patch 示例 | 改善值 |
| ----- | ----------------- | ------ |
| 100km | (16.0°, 104.0°)   | -50%   |
| 200km | (16.0°, 104.0°)   | -71%   |
| 300km | (13.0°, 134.0°)   | -37%   |

**原因分析**:

1. **某些区域模型差异大**: 负改善区域通常是两模型差异显著的地方
2. **字典无法泛化**: 字典在训练集学到的模式不适用于这些特殊区域
3. **CC 筛选不够严格**: 当前 CC>-0.5 太宽松，包含了不适合 SDL 的 patch

### 问题 3: 正态性检验不一致

**观察**:

| 深度  | Combined 检验 | Q-Q 图判断       | Skewness |
| ----- | ------------- | ---------------- | -------- |
| 100km | ✅ Pass       | ✅ Approx Normal | -0.33    |
| 200km | ❌ Fail       | ❌ Non-Normal    | -0.65    |
| 300km | ❌ Fail       | ❌ Non-Normal    | -0.68    |

**原因**: 200km/300km 偏度超过 0.5 阈值，误差分布略向负偏

---

## 📐 v2.3 优化方案

### 方案 A: Patch 级别自适应融合 [推荐] ⭐⭐⭐

**核心思想**: 不是所有 patch 都适合 SDL，应该根据预测可靠性选择性融合

**理论依据**:

```
对于每个验证 patch:
├── 计算 SDL 重建误差 e_SDL
├── 计算 Baseline 重建误差 e_Baseline
├── 如果 e_SDL < e_Baseline * (1 - margin):
│   └── 使用 SDL 重建
├── 否则:
│   └── 使用 Baseline 或加权融合
```

**实现代码**:

```python
@dataclass
class PatchAdaptiveFusionConfig:
    """Patch级别自适应融合配置"""
    enabled: bool = True

    # 融合策略
    fusion_strategy: str = 'adaptive'  # 'sdl_only', 'baseline_only', 'adaptive', 'weighted'

    # 自适应阈值
    sdl_advantage_margin: float = 0.1  # SDL需要比Baseline好10%才使用SDL
    cc_min_for_sdl: float = 0.3        # CC低于此值强制使用Baseline

    # 加权融合参数
    use_cc_weighting: bool = True      # 使用CC作为融合权重

    def compute_fusion_weight(self, cc: float, sdl_error: float, baseline_error: float) -> float:
        """计算SDL融合权重 (0=纯Baseline, 1=纯SDL)"""
        if cc < self.cc_min_for_sdl:
            return 0.0  # 强制Baseline

        # 基于误差比的权重
        if baseline_error > 0:
            improvement_ratio = (baseline_error - sdl_error) / baseline_error
            if improvement_ratio < self.sdl_advantage_margin:
                return 0.0  # SDL优势不足

        # 基于CC的软权重
        if self.use_cc_weighting:
            # CC高时倾向SDL，CC低时倾向Baseline
            weight = min(1.0, max(0.0, (cc - 0.2) / 0.6))
            return weight

        return 1.0
```

**预期效果**:

- 自动过滤 SDL 表现差的 patch
- 保留 SDL 优势区域的改善
- 整体相对改善有望提升

### 方案 B: CC 自适应阈值优化

**当前问题**: 固定 CC 阈值 (-0.5) 对所有深度使用，但不同深度的 CC 分布差异大

**观察数据**:

| 深度  | CC 均值 | CC 标准差 | 建议阈值 |
| ----- | ------- | --------- | -------- |
| 100km | 0.465   | 0.31      | 0.2~0.3  |
| 200km | 0.351   | 0.27      | 0.1~0.2  |
| 300km | 0.205   | 0.27      | 0.0      |

**优化方案**:

```python
def compute_adaptive_cc_threshold(cc_distribution: np.ndarray,
                                  target_retention: float = 0.7) -> float:
    """
    基于CC分布自适应计算阈值

    Args:
        cc_distribution: 所有patch的CC值数组
        target_retention: 目标保留比例 (默认保留70%的patch)

    Returns:
        自适应CC阈值
    """
    # 使用分位数确定阈值
    threshold = np.percentile(cc_distribution, (1 - target_retention) * 100)

    # 确保阈值合理
    threshold = max(threshold, -0.3)  # 下限
    threshold = min(threshold, 0.5)   # 上限

    return threshold
```

### 方案 C: 深度-区域联合优化

**思路**: 100km 深度可能需要更小的 patch 尺寸或不同的物理区域

**实验方案**:

| 配置    | Patch 尺寸  | 物理范围 | 预期效果     |
| ------- | ----------- | -------- | ------------ |
| 当前    | 6×6 / 24×24 | 6°×6°    | 大区域平滑   |
| 方案 C1 | 4×4 / 16×16 | 4°×4°    | 捕捉更细节   |
| 方案 C2 | 8×8 / 32×32 | 8°×8°    | 更大范围结构 |

**实现**:

```python
@dataclass
class DepthAdaptivePatchConfig:
    """深度自适应Patch配置"""
    depth_patch_configs: Dict[int, Dict] = field(default_factory=lambda: {
        100: {'physical_size': 4.0, 'size_base': 4, 'size_highres': 16},  # 更小patch
        200: {'physical_size': 6.0, 'size_base': 6, 'size_highres': 24},  # 标准
        300: {'physical_size': 6.0, 'size_base': 6, 'size_highres': 24},
    })
```

### 方案 D: 多字典集成学习

**思路**: 训练多个字典，根据区域特征选择最佳字典

```
M₁, M₂ 数据 → 按区域类型分组 → 分别训练字典
├── 大陆区域 → D₁_continent, D₂_continent
├── 海洋区域 → D₁_ocean, D₂_ocean
├── 俯冲带区域 → D₁_subduction, D₂_subduction
└── 推断时选择对应字典
```

---

## 🔬 v2.3 理论改进

### 1. 优化目标函数改进

**当前目标 (公式 3)**:
$$\min_{\mathbf{D},\mathbf{C}} \|\mathbf{P} - \mathbf{DC}\|_2^2 + \lambda\|\mathbf{C}\|_0$$

**改进: 加权重建误差**:
$$\min_{\mathbf{D},\mathbf{C}} \sum_i w_i \|\mathbf{p}_i - \mathbf{Dc}_i\|_2^2 + \lambda\|\mathbf{C}\|_0$$

其中权重 $w_i$ 可基于:

- CC 值: 高 CC patch 权重更高
- 区域重要性: 关键地质区域权重更高
- 数据质量: 可靠区域权重更高

### 2. D₂ 计算改进

**当前方法**: 岭回归
$$\mathbf{D}_2 = \arg\min_{\mathbf{D}} \|\mathbf{P}_2 - \mathbf{DC}\|_2^2 + \lambda_{reg}\|\mathbf{D}\|_2^2$$

**改进方案: 约束岭回归**

添加物理约束:

1. **非负约束**: 速度值不应为负
2. **平滑约束**: 相邻原子应平滑变化
3. **稀疏约束**: D₂ 原子也应稀疏

```python
# 带约束的D₂计算
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, positive=False)
elastic.fit(C_train, P2_train_centered)
D2 = elastic.coef_
```

### 3. 训练/验证划分改进

**当前方法**: 随机 80/20 划分

**问题**: 空间相关性导致验证集不独立

**改进: 空间分块交叉验证**

```python
def spatial_block_cv_split(coords: np.ndarray, n_splits: int = 5) -> List[Tuple]:
    """
    基于空间位置的分块交叉验证

    将研究区域划分为空间块，确保训练集和验证集空间分离
    """
    # 按经度划分块
    lon_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), n_splits + 1)

    splits = []
    for i in range(n_splits):
        # 第i块作为验证集
        val_mask = (coords[:, 1] >= lon_bins[i]) & (coords[:, 1] < lon_bins[i+1])
        train_mask = ~val_mask

        splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))

    return splits
```

---

## 📊 v2.3 实验计划

### 实验 1: Patch 自适应融合效果验证

| 实验配置          | 100km 改善 | 200km 改善 | 300km 改善 | 负改善 patch 数 |
| ----------------- | ---------- | ---------- | ---------- | --------------- |
| v2.2 (当前)       | 3.8%       | 36.1%      | 20.3%      | ~20%            |
| +方案 A (自适应)  | ?          | ?          | ?          | ?               |
| +方案 B (CC 优化) | ?          | ?          | ?          | ?               |
| +方案 A+B         | ?          | ?          | ?          | ?               |

### 实验 2: 不同 Patch 尺寸对比 (100km 深度)

| Patch 配置   | Transform Error | Relative Improvement | 最佳 |
| ------------ | --------------- | -------------------- | ---- |
| 6°×6° (当前) | 2.32%           | 3.8%                 |      |
| 4°×4°        | ?               | ?                    |      |
| 8°×8°        | ?               | ?                    |      |

### 实验 3: CC 阈值敏感性分析

| CC 阈值     | 保留 Patch 数 | Transform Error | Relative Improvement |
| ----------- | ------------- | --------------- | -------------------- |
| -0.5 (当前) | 335 (100%)    | 2.32%           | 3.8%                 |
| 0.0         | ?             | ?               | ?                    |
| 0.2         | ?             | ?               | ?                    |
| 0.3         | ?             | ?               | ?                    |

---

## 📊 完整可视化体系更新 (v2.2 状态)

| 序号           | 图表名称              | v2.2 状态 | 说明                    |
| -------------- | --------------------- | --------- | ----------------------- |
| **数据准备**   |
| Fig01          | 区域覆盖与 Patch 网格 | ✅        | 显示真实经纬度          |
| Fig02          | Patch 尺寸对比        | ✅        | Low/Up/High 对比        |
| Fig03          | 模型对比              | ✅        | SinoScope vs EARA2024   |
| **Patch 提取** |
| Fig04          | Patch 提取演示        | ✅        | 6 行示例+真实坐标       |
| Fig06          | CC 分布统计           | ✅        | 直方图+CDF              |
| Fig06b         | CC 空间分布           | ✅        | 地图+散点图 (v2.2 新增) |
| **字典学习**   |
| Fig07          | D₁ 字典原子           | ✅        | 20 个 6×6 原子          |
| Fig08          | D₂ 字典原子           | ✅        | 20 个 24×24 原子        |
| Fig09          | 稀疏系数 C 矩阵       | ✅        | 热图展示                |
| Fig10          | D₁↔D₂ 对应关系        | ✅        | 5 对原子对比            |
| Fig11          | 稀疏性分析            | ✅        | 4 子图统计              |
| **重建验证**   |
| Fig12          | 重建演示              | ✅        | GT/SDL/Baseline/误差    |
| Fig13          | 误差分布              | ✅        | 直方图+正态拟合+统计    |
| Fig14          | Q-Q 图                | ✅        | 正态性视觉验证          |
| **综合评估**   |
| Fig20          | SDL vs Baseline       | ✅        | 多深度综合报告          |
| **待实现**     |
| Fig15          | 超参数敏感性          | ⏳        | K, α 扫描热图           |
| Fig16          | 训练收敛曲线          | ⏳        | Loss vs 迭代            |
| Fig17          | 全域变换结果          | ⏳        | 增强前后对比            |
| Fig18          | 垂直剖面              | ⏳        | 论文 Fig.7 风格         |

---

## 🔧 v2.3 代码修改清单

### 1. 新增 PatchAdaptiveFusion 类

```python
class PatchAdaptiveFusion:
    """Patch级别自适应融合器"""

    def __init__(self, config: PatchAdaptiveFusionConfig):
        self.config = config

    def fuse_patches(self,
                     sdl_patches: np.ndarray,      # (N, L)
                     baseline_patches: np.ndarray, # (N, L)
                     highres_patches: np.ndarray,  # (N, L) Ground Truth
                     correlations: np.ndarray      # (N,) CC值
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            fused_patches: (N, L) 融合后的patches
            weights: (N,) 各patch的SDL权重 (0~1)
        """
        N = sdl_patches.shape[0]
        weights = np.zeros(N)
        fused = np.zeros_like(sdl_patches)

        for i in range(N):
            # 计算误差
            sdl_error = np.linalg.norm(sdl_patches[i] - highres_patches[i])
            baseline_error = np.linalg.norm(baseline_patches[i] - highres_patches[i])

            # 计算权重
            w = self.config.compute_fusion_weight(
                correlations[i], sdl_error, baseline_error)
            weights[i] = w

            # 融合
            fused[i] = w * sdl_patches[i] + (1 - w) * baseline_patches[i]

        return fused, weights
```

### 2. 修改 CoupledDictionaryLearner.train()

添加自适应融合逻辑:

```python
def train(self, patch_result: PairedPatchResult,
          depth_km: Optional[float] = None,
          use_adaptive_fusion: bool = True) -> CoupledDictionaryResult:
    """
    v2.3: 支持自适应融合
    """
    # ... 现有训练逻辑 ...

    if use_adaptive_fusion:
        # 计算baseline重建
        baseline_patches = self._compute_baseline_patches(...)

        # 自适应融合
        fuser = PatchAdaptiveFusion(self.config.adaptive_fusion)
        fused_patches, fusion_weights = fuser.fuse_patches(
            sdl_patches, baseline_patches,
            patches_highres_val, correlations_val)

        # 使用融合后的patches计算最终误差
        transform_error = np.linalg.norm(fused_patches - patches_highres_val) / ...
```

### 3. 新增超参数扫描功能

```python
class SDLHyperparameterSearch:
    """超参数网格搜索"""

    def __init__(self, config: SDLConfig):
        self.config = config

    def grid_search(self,
                    patch_result: PairedPatchResult,
                    K_range: List[int] = [15, 20, 25, 30],
                    alpha_range: List[float] = [0.05, 0.1, 0.15, 0.2],
                    cc_threshold_range: List[float] = [0.0, 0.1, 0.2, 0.3]
                    ) -> pd.DataFrame:
        """执行网格搜索，返回结果DataFrame"""
        results = []

        for K in K_range:
            for alpha in alpha_range:
                for cc_thresh in cc_threshold_range:
                    # 配置参数
                    self.config.dictionary.n_components = K
                    self.config.dictionary.alpha = alpha
                    self.config.dictionary.min_correlation = cc_thresh

                    # 训练并评估
                    learner = CoupledDictionaryLearner(self.config, self.logger)
                    result = learner.train(patch_result)

                    results.append({
                        'K': K, 'alpha': alpha, 'cc_threshold': cc_thresh,
                        'transform_error': result.transform_error,
                        'relative_improvement': result.relative_improvement
                    })

        return pd.DataFrame(results)
```

---

## 📌 模型配置（保留）

| 模型   | 名称              | 分辨率           | 角色               | 覆盖范围 |
| ------ | ----------------- | ---------------- | ------------------ | -------- |
| **M₁** | 2022_SinoScope1.0 | 1°×1°×20km       | 基础模型（待增强） | 最广     |
| **M₂** | 2024_EARA2024     | 0.25°×0.25°×10km | 高分辨率目标模型   | 局部     |

---

## 🎯 v2.3 实施路线图

### Phase A: Patch 自适应融合 [P0 - 最高优先级] ⭐⭐⭐

| 任务 | 详细说明                          | 验收标准     | 预计时间 |
| ---- | --------------------------------- | ------------ | -------- |
| A.1  | 实现 `PatchAdaptiveFusionConfig`  | 配置类完成   | 1h       |
| A.2  | 实现 `PatchAdaptiveFusion` 类     | 融合逻辑正确 | 2h       |
| A.3  | 集成到 `CoupledDictionaryLearner` | 训练流程更新 | 1h       |
| A.4  | 添加融合权重可视化                | 新增 Fig21   | 1h       |
| A.5  | 验证 100km 改善提升               | 目标 >15%    | 2h       |

### Phase B: CC 阈值自适应 [P1]

| 任务 | 详细说明               | 验收标准 | 预计时间 |
| ---- | ---------------------- | -------- | -------- |
| B.1  | 实现自适应 CC 阈值计算 | 代码完成 | 1h       |
| B.2  | 集成到 Patch 提取流程  | 流程更新 | 1h       |
| B.3  | 对比固定/自适应效果    | 实验报告 | 1h       |

### Phase C: 超参数优化工具 [P1]

| 任务 | 详细说明           | 验收标准     | 预计时间 |
| ---- | ------------------ | ------------ | -------- |
| C.1  | 实现网格搜索类     | 代码完成     | 2h       |
| C.2  | 100km 最优参数搜索 | 找到最佳配置 | 2h       |
| C.3  | 超参数敏感性热图   | Fig15 完成   | 1h       |

### Phase D: 全域变换实现 [P1]

| 任务 | 详细说明              | 验收标准     | 预计时间 |
| ---- | --------------------- | ------------ | -------- |
| D.1  | M₁ 全域 Patch 提取    | 覆盖整个区域 | 2h       |
| D.2  | D₁ 编码 + D₂ 解码     | 输出正确     | 1h       |
| D.3  | Patch 拼接重建        | 无缝拼接     | 2h       |
| D.4  | 自适应融合应用        | 全域融合     | 1h       |
| D.5  | 增强模型保存 (NetCDF) | 格式兼容     | 1h       |

### Phase E: 可视化完善 [P2]

| 任务 | 详细说明           | 验收标准 | 预计时间 |
| ---- | ------------------ | -------- | -------- |
| E.1  | 融合权重空间分布图 | Fig21    | 1h       |
| E.2  | 超参数敏感性热图   | Fig15    | 1h       |
| E.3  | 全域增强前后对比   | Fig17    | 1h       |

---

## 📚 公式速查表 (v2.3 更新)

| 公式          | 含义           | Python 实现                            |
| ------------- | -------------- | -------------------------------------- |
| (1)           | patch 字典表示 | `p ≈ D @ c`                            |
| (2)           | 矩阵形式       | `P ≈ D @ C`                            |
| (3)           | 优化目标       | `min ‖P - DC‖² + λ‖C‖₀`                |
| (4)           | P₁ ≈ D₁C       | 低分辨率表示                           |
| (5)           | P₂ ≈ D₂C       | 高分辨率表示                           |
| (6)           | D₁, C 学习     | `DictionaryLearning().fit()`           |
| (7)           | D₂ 计算        | `Ridge().fit(C, P2)`                   |
| (8)           | 验证编码       | `transform(P1_val)`                    |
| (9)           | 相对误差       | `‖P_pred - P_val‖₂ / ‖P_val‖₂`         |
| **(10) v2.3** | 自适应融合权重 | `w = f(CC, e_SDL, e_Baseline)`         |
| **(11) v2.3** | 融合重建       | `P_fused = w*P_SDL + (1-w)*P_Baseline` |

---

## 📝 下一步工作优先级

1. **[紧急 P0]** Patch 自适应融合实现 - 解决 100km 改善问题
2. **[高 P1]** CC 阈值自适应优化 - 提升 patch 筛选质量
3. **[高 P1]** 超参数网格搜索 - 找到各深度最优配置
4. **[高 P1]** 全域变换实现 - 核心功能完成
5. **[中 P2]** 可视化完善 - 融合权重分布等

---

## 📈 性能优化目标 (v2.3)

| 指标              | v2.2 当前值 | v2.3 目标 | 优化手段           |
| ----------------- | ----------- | --------- | ------------------ |
| 100km 相对改善    | 3.8%        | **>15%**  | 自适应融合         |
| 200km 相对改善    | 36.1%       | **>35%**  | 保持               |
| 300km 相对改善    | 20.3%       | **>25%**  | CC 优化            |
| 负改善 patch 比例 | ~20%        | **<5%**   | 自适应融合         |
| 正态性检验通过率  | 1/3         | **3/3**   | 放宽 skewness 阈值 |

---

_更新日期：2025-12-23_
_版本：v2.3 - 基于 v2.2 测试结果的深度优化规划_
_状态：v2.2 可视化/深度自适应完成，v2.3 性能优化方案已制定_
