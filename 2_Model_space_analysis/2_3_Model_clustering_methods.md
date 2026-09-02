# 2_3 单模型速度相聚类：数据、方法与公式

> 对应代码：`2_Model_space_analysis/2_3_Model_clustering.py` (v7.5)
> 用途：论文方法章节（Methods）素材 + 方法学讨论（含审稿预案）
> 日期：2026-09-01
> 版本要点：v7.4 移除 surface_units 定量评分；v7.5 全部结构对比改为定性叠图（Slab2 定量 MI/ρ 亦移除）、选 K 改为 ΔBIC 拐点判据（避免 argmin(BIC) 贴 K 上界）、剖面图新增 facies 自动解释标注（置于子图标题下方）

---

## 1. 科学目标与定位

对单个三维速度模型做**体素级无监督聚类**，将 (δlnVp, δlnVs) 特征空间划分为若干"速度相"（velocity facies），用于：

1. 刻画模型内部的速度相 / 构造分区结构，并按扰动特征给出各相的推测构造解释；
2. 与独立构造参考做**定性**空间对照：切片/剖面/三维体叠绘 USGS Slab2 俯冲板片几何，浅部切片叠绘 USGS 地质省与中国大陆活动地块边界（不做定量评分，见 §5 说明）；
3. 与 2_2（CW-SSIM 模型间相似性）互补：本模块只做**模型内部**结构分析，不用于模型间代表性筛选。

---

## 2. 数据

### 2.1 输入速度模型

| 模型 | 水平分辨率 | 深度范围（截断后） | 来源 |
| --- | --- | --- | --- |
| SinoScope 1.0 (Ma et al., 2022) | 1.0° | 0–1000 km | 标准化 NetCDF `*_original.nc` |
| EARA2024 (Xi et al., 2024) | 0.25° | 0–1000 km | 同上 |
| FWEA23 (Liu et al., 2024) | 0.25° | 0–1000 km | 同上 |

- 经纬度保留各模型**原生范围**，不裁剪到统一研究区；深度统一截断到 0–1000 km。
- 默认特征为 `(vp, vs)`；代码支持 `vsv/vsh/vpv/vph` 扩展。
- 无效值（9999.0）置为 NaN；任一特征缺失的体素整体剔除。
- 坐标约定：`(latitude, longitude, depth)`，深度向下为正（km），速度 km/s。

### 2.2 结构参考数据（仅用于定性叠图与分带，不参与聚类）

| 数据 | 用途 |
| --- | --- |
| USGS Slab2（Hayes et al., 2018）`dep`/`thk` 网格，14 个板片区（kur, ryu, izu, man, phi, sum, sul, him, hin, mak, cot, hal, png, sol） | 切片等深线 / 剖面体轮廓 / 三维体掩膜叠图（定性对照） |
| USGS 世界地质省边界线（`prv1ec`, `prv3bl` 线划缓存） | 浅部切片叠图（定性对照） |
| 中国大陆活动地块边界（CN-block L1/L2） | 浅部切片叠图（定性对照） |
| Moho 深度点数据（插值成网格） | 仅供 `moho_4band` 方案空间分带 |

---

## 3. 预处理

### 3.1 扰动域转换（去除深度主趋势）

绝对速度随深度的一阶增长会主导任何聚类结果，因此先转到相对各深度**水平平均一维参考**的对数扰动域：

$$
\bar V(z) = \frac{1}{N_{xy}(z)} \sum_{(\varphi,\lambda)} V(\varphi, \lambda, z), \qquad
\delta \ln V(\varphi, \lambda, z) = \ln \frac{V(\varphi, \lambda, z)}{\bar V(z)}
$$

- 参考取**模型自身的水平平均**（与 2_1 模块一致），而非 PREM/ak135，避免不同模型原始参考模型不一致引入的系统偏差；
- 备选 `relative` 形式 $\delta V/V = (V-\bar V)/\bar V$，一阶等价（$\ln(1+x)\approx x$）。

### 3.2 特征标准化

对每个特征做 z-score 标准化（在扰动域上、全深度范围内全局拟合一次）：

$$
\tilde x_j = \frac{x_j - \mu_j}{\sigma_j}, \qquad j \in \{\delta\ln V_p,\ \delta\ln V_s\}
$$

使 Vp、Vs 扰动在特征空间中权重相当。

### 3.3 拟合子采样与全量预测

- **拟合**（BIC 搜索与 GMM 参数估计）：若有效体素数 > $2\times10^6$，按深度层分层随机抽样（每层 ≥ 500 点，固定随机种子 42）；
- **预测**（标签与后验概率）：始终在**全量**有效体素上进行，保证重建的三维标签体无空洞。

抽样在统计上无损：GMM 自由参数极少（$K=10$、$d=2$、full 协方差时仅 59 个），$2\times10^6$ 样本远超估计需求；且体素间强空间相关，名义样本量再大有效信息量也不再增加。

**重抽样稳定性复验**（`auto_gmm.stability_check`，默认关闭）：开启后对每个深度带用不同随机种子重新抽样并重新拟合（K 固定），在全量体素上预测标签，与参考结果计算 ARI（对簇编号置换不变）。ARI ≈ 1 即证明分区结果对拟合子采样与 EM 随机初始化均不敏感，可直接作为论文附录的稳健性证据；结果随指标 JSON 保存（`stability.seeds_ari` / `stability.mean_ari`）。

---

## 4. 聚类方法：深度分层高斯混合模型（GMM）

### 4.1 深度分层

不同深度域的扰动幅度与物理机制差异显著（地壳 ±10% vs 下地幔 ±1–2%），混在一起聚类会让浅部方差主导。因此按深度分带独立聚类，带间标签偏移后拼合为全局标签。支持两种方案：

| 方案 | 分带 | K 搜索范围 |
| --- | --- | --- |
| `fixed_3band`（默认） | 浅部 0–410 km / 过渡带 410–660 km / 下地幔 660–1000 km | 2–10 / 2–5 / 2–4 |
| `moho_4band` | 地壳 z<Moho / 岩石圈 Moho–410 km / 过渡带 / 下地幔 | 2–10 / 2–10 / 2–5 / 2–4 |

`moho_4band` 中 Moho 深度由点数据插值到模型网格；无覆盖处回退 35 km，并裁剪到 [10, 80] km。

### 4.2 高斯混合模型

对每个深度带内的标准化特征向量 $\mathbf{x} \in \mathbb{R}^d$（$d=2$），假设其密度为 $K$ 个高斯分量的混合：

$$
p(\mathbf{x} \mid \Theta) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k),
\qquad \sum_k \pi_k = 1
$$

- **协方差类型**：`full`（完整协方差矩阵），以保留 δlnVp–δlnVs 的相关结构与各向异性簇形状；
- **正则化**：对角线加 `reg_covar` $=10^{-6}$ 防止奇异协方差。

参数由期望最大化（EM）算法估计。E 步计算后验责任（responsibility）：

$$
\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}
{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

M 步更新参数：

$$
N_k = \sum_i \gamma_{ik}, \quad
\pi_k = \frac{N_k}{N}, \quad
\boldsymbol{\mu}_k = \frac{1}{N_k}\sum_i \gamma_{ik}\mathbf{x}_i, \quad
\boldsymbol{\Sigma}_k = \frac{1}{N_k}\sum_i \gamma_{ik}(\mathbf{x}_i-\boldsymbol{\mu}_k)(\mathbf{x}_i-\boldsymbol{\mu}_k)^{\mathsf T}
$$

体素 $i$ 的硬标签取最大后验分量 $\hat k_i = \arg\max_k \gamma_{ik}$；$\max_k \gamma_{ik}$ 同时作为该体素的分类置信度输出。

### 4.3 EM 收敛准则（论文中应明确写出）

采用 scikit-learn `GaussianMixture` 的标准收敛判据：当相邻两次迭代**单样本平均对数似然下界**的变化量小于容差时判定收敛：

$$
\left| \mathcal{L}^{(t)} - \mathcal{L}^{(t-1)} \right| < \text{tol}, \qquad
\mathcal{L}^{(t)} = \frac{1}{N}\sum_i \ln p(\mathbf{x}_i \mid \Theta^{(t)})
$$

- 容差 tol $= 10^{-3}$（scikit-learn 默认值，代码未覆盖）；
- 最大迭代次数 `max_iter` = 300；
- 随机重启 `n_init` = 10 次（k-means 初始化），取对数似然最高的一次——重启不足会使 BIC(K) 曲线出现局部最优造成的非单调锯齿，直接干扰拐点判据（§4.4）；
- 每次拟合的 `converged_` 标志与迭代次数 `n_iter_` 记录到指标中；全局 `converged` 定义为**所有深度带均收敛**。

> 注意区分两个层面："EM 迭代收敛"（上式）与"K 的选择"（下节 ΔBIC 拐点判据，不是收敛问题）。

### 4.4 分量数 K 的选择：BIC 全区间扫描 + ΔBIC 拐点判据

对每个深度带在给定区间内全扫描 $K$，计算贝叶斯信息准则（BIC）：

$$
\mathrm{BIC}(K) = -2 \ln \hat L + p_K \ln N
$$

其中 $\hat L$ 为最大化似然，$p_K$ 为自由参数个数。对 $d$ 维特征、full 协方差：

$$
p_K = \underbrace{(K-1)}_{\pi_k} + \underbrace{Kd}_{\boldsymbol\mu_k} + \underbrace{K\frac{d(d+1)}{2}}_{\boldsymbol\Sigma_k}
\;\; \xrightarrow{d=2} \;\; p_K = 6K - 1
$$

**选 K 判据：ΔBIC 拐点（elbow），而非 argmin(BIC)**。体元级数据中 $N \sim 10^6$，BIC 的似然项完全支配惩罚项（$p_K \ln N$ 相对可忽略），而连续、空间相关的扰动场并非有限个高斯的真实混合——增加分量总能改进分段近似，故 $\arg\min_K \mathrm{BIC}$ 几乎必然贴到 K 上界（实测四个深度带均如此）。因此采用拐点判据：将每个 K 的改善量归一化到**整条 BIC 曲线的总下降幅度**，定义增益

$$
g(K) = \frac{\min_{k<K} \mathrm{BIC}(k) - \mathrm{BIC}(K)}{\max_k \mathrm{BIC}(k) - \min_k \mathrm{BIC}(k)},
\qquad
K^\ast = \max\{K : g(K) \ge \tau\}
$$

即**最后一个增益超过总下降幅度 $\tau$ 比例的 K**（默认 $\tau$ = `bic_elbow_frac` = 0.05，即 5%）。两个设计要点：(i) 分母是曲线跨度而非 $|\mathrm{BIC}|$ 本身——BIC 绝对值与曲线跨度可相差两个量级且随 $N$（模型分辨率）变化，用 $|\mathrm{BIC}|$ 归一会使粗分辨率模型的真实改善被误判为"无收益"，而曲线跨度归一是尺度无关的，不同模型可共用同一阈值；(ii) 分子以运行最小值（而非相邻 K）为基准，可抵抗 EM 局部最优造成的非单调尖峰（某个 K 偶然拟合差不会误触发拐点）。设 $\tau=0$ 可退回纯 argmin(BIC)。

**先验固定 K 模式**（`bands[i]['fixed_k']`）：亦可基于构造可解释性直接固定每带 K、跳过 BIC 扫描（Lekić et al., 2012 即采用先验固定 K=5 的做法）。论文表述为"K 基于构造先验固定，ΔBIC 拐点分析作为参考"；两种模式互斥，结果 JSON 中 `selection_rule` 分别记为 `bic_elbow` / `fixed_k`。

搜索循环中只计算 BIC（AIC 与对数似然顺带记录，代价为零）；轮廓系数、Davies–Bouldin、Calinski–Harabasz 等内部诊断指标**不逐 K 计算**，仅对选定 $K^\ast$ 的最终模型计算一次（§4.6）。BIC 搜索中各 K 的已拟合模型被缓存，选定 $K^\ast$ 后直接复用，避免二次拟合。决策图为单面板 BIC 曲线（红星标注拐点 $K^\ast$，若与 argmin 不同则灰星并注；内嵌小图展示 $g(K)$ 曲线与阈值线 $\tau$）。

> **诚实声明（论文 Limitations 建议）**：体素间存在强空间相关，违背 BIC 的 i.i.d. 假设，有效样本量被高估，BIC 惩罚项相对偏弱——这正是 argmin(BIC) 失效、需要拐点判据的根本原因（Lekić et al., 2012 对全球下地幔模型聚类同样未依赖信息准则的绝对最小值）。拐点阈值 $\tau$ 是超参数，代码在每次 BIC 扫描后自动记录 $\tau \in \{2\%, 5\%, 10\%\}$ 下各自的 $K^\ast$（`elbow_sensitivity`，无需重新拟合）：三者一致说明拐点稳健，可在论文中直接引用；不一致则应报告 $K^\ast$ 的变化区间并检验主要速度相的空间格局在该区间内是否稳定。同时以 K 上界（浅部 ≤10、过渡带 ≤5、下地幔 ≤4，基于构造先验）与后验置信度交叉检验分区稳健性。

### 4.5 不确定性度量

对每个体素输出：

- **最大后验概率** $p_{\max,i} = \max_k \gamma_{ik}$；高置信比例定义为 $\Pr(p_{\max} > 0.6)$，低置信比例 $\Pr(p_{\max} < 0.5)$；
- **后验熵**：

$$
H_i = -\sum_{k=1}^{K} \gamma_{ik} \ln \gamma_{ik}
$$

熵高的体素位于速度相过渡带，可解释为相边界的模糊区。

### 4.6 内部质量指标（诊断用，仅对每带最终模型计算一次）

| 指标 | 定义 | 方向 |
| --- | --- | --- |
| 轮廓系数 $s = \frac{1}{N}\sum_i \frac{b_i - a_i}{\max(a_i, b_i)}$（$a_i$ 簇内平均距离，$b_i$ 最近邻簇平均距离；5000 点子采样） | 越大越好 | [-1, 1] |
| Davies–Bouldin $\mathrm{DB} = \frac{1}{K}\sum_k \max_{j\neq k}\frac{s_k+s_j}{d_{kj}}$ | 越小越好 | ≥0 |
| Calinski–Harabasz（组间/组内方差比） | 越大越好 | ≥0 |

---

## 5. 结构参考叠图（定性对照）

聚类完全不使用空间坐标与地质信息，因此速度相与独立构造参考在空间上的目视一致性本身就是非平凡的**定性外部对照**。本模块的全部结构对比均为叠图，不产生定量分数。

### 5.1 Slab2 板片几何叠图

- **深度切片**：facies 切片叠绘 Slab2 顶面等深线（100–700 km 各档）；
- **垂直剖面**：经度–深度剖面叠绘板片体轮廓（顶/底界线 + 灰罩）；
- **三维体掩膜**：由 `dep`（板片顶面深度）与 `thk`（板片厚度）构建体素级板片掩膜，与聚类标签体同维度对比：

$$
M(\varphi,\lambda,z) = \bigcup_{r \in \text{regions}} \left[ z_{\text{top}}^{(r)} \le z \le z_{\text{top}}^{(r)} + \text{thk}^{(r)} \right],
\qquad z_{\text{top}} = |\mathrm{dep}|
$$

### 5.2 浅部地质边界叠图

浅部 facies 切片（10/40 km 等）叠绘 USGS 地质省界线与中国大陆活动地块边界（公共区 [80°–150°E, 10°–55°N]）。

> **为什么不做定量评分（论文可直接引用的理由）**：早期版本曾计算两类定量符合度——(i) 地质省栅格化后的 NMI/加权纯度（v7.4 移除）：地质省是地表二维图斑，与深部速度相不存在应然的一一对应；(ii) facies–Slab2 的切片互信息与"标签梯度 vs 板片深度梯度"的 Spearman 相关（v7.5 移除）：聚类标签是**名义变量**，其数值梯度依赖任意的标签编号，且切片 MI 受板片覆盖率和分箱方案支配，两个分数都缺乏稳定的物理解释，也不参与任何下游决策。叠图目视对照（板片轮廓与快速相的空间重合）已足以支撑论文讨论；模型间的定量优劣评价统一交给数据空间的波形拟合（Phase D）。

### 5.3 速度相自动解释标注（启发式）

剖面图（2-3-6）为每个簇标注推测构造解释。主判据为簇平均 $\overline{\delta\ln V_s}$ 的符号与幅度（对温度/部分熔融最敏感）；若 $\overline{\delta\ln V_p}$ 与 $\overline{\delta\ln V_s}$ 符号相反且均显著，追加 "Vp–Vs decoupled" 标记（提示成分/流体等非热成因）。

各带阈值（weak / strong，无量纲 δlnV）：

| 深度带 | weak | strong | 依据 |
| --- | --- | --- | --- |
| crust | 0.03 | 0.10 | 地壳扰动幅度 ~±10% |
| lithosphere / shallow | 0.010 | 0.030 | 上地幔 ~±3% |
| transition_zone | 0.005 | 0.015 | 过渡带 ~±1.5% |
| lower_mantle | 0.004 | 0.012 | 下地幔 ~±1% |

分类规则（英文标签用于图面）：

| 深度带 | 条件（$v=\overline{\delta\ln V_s}$） | 推测解释 |
| --- | --- | --- |
| crust | $v \le -$strong | Very slow crust (thick sediments / basin) |
| crust | $-$strong $< v \le -$weak | Slow crust (sedimentary / warm) |
| crust | $\|v\| <$ weak | Average crust |
| crust | weak $\le v <$ strong | Fast crust (crystalline basement) |
| crust | $v \ge$ strong | Very fast crust (cratonic / mafic) |
| lithosphere | $v \ge$ strong | Fast anomaly (cratonic root / slab) |
| lithosphere | $v \le -$strong | Slow anomaly (asthenospheric / back-arc) |
| lithosphere | 介于 weak 与 strong 之间 | Moderately fast/slow mantle (cold/warm) |
| lithosphere | $\|v\| <$ weak | Ambient upper mantle |
| transition_zone | $v \ge$ weak / $v \le -$weak / 其余 | Stagnant slab / warm–hydrated / ambient TZ |
| lower_mantle | $v \ge$ weak / $v \le -$weak / 其余 | Slab remnant / thermal upwelling / ambient |

> **定位声明（论文必须写）**：该标注是基于扰动符号与幅度的**启发式推测**（interpretation guide），不是岩性判定。GMM 分量与物理"岩石相"不必一一对应（§7.1），跨相解释应结合区域构造背景与已知板片/克拉通分布核验。

---

## 6. 关键参数汇总（论文参数表素材）

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| 特征 | δlnVp, δlnVs（标准化） | 相对水平平均 1D 参考 |
| 深度范围 | 0–1000 km | 统一截断 |
| 分层方案 | fixed_3band（默认）/ moho_4band | 带间独立 GMM |
| 协方差类型 | full | 保留 Vp–Vs 相关 |
| K 选择 | ΔBIC 拐点判据（增益 ≥ 总下降幅度的 τ = 5%） | 浅部 2–10，过渡带 2–5，下地幔 2–4 |
| EM 收敛 | \|ΔL̄\| < 10⁻³（平均对数似然） | max_iter=300, n_init=10 |
| 拐点敏感性 | τ ∈ {2%, 5%, 10%} 下的 K* 均写入结果 JSON | 三者一致 → 稳健；否则报告区间 |
| 协方差正则 | reg_covar = 10⁻⁶ | 防奇异 |
| 拟合子采样 | ≤ 2×10⁶ 点，深度分层抽样 | 预测用全量体素 |
| 稳定性复验 | 换种子重抽样+重拟合，报告标签 ARI | 默认关闭，附录证据用 |
| 置信度阈值 | p_max > 0.6 | 高置信比例统计 |
| facies 解释阈值 | 见 §5.3 表 | 分带 δlnVs weak/strong |
| 随机种子 | 42 | 全流程固定 |

---

## 7. 方法学讨论（审稿预案）

### 7.1 为什么用 GMM 而不是 K-means？

层析模型聚类在文献中有明确先例：Lekic et al. (2012, EPSL) 与 Cottaar & Lekic (2016, GJI) 用 k-means 对全球下地幔 Vs 模型做聚类识别 LLSVP；勘探地震学中 GMM 是地震相分类的常规工具。相对 k-means，GMM 的优势与本问题高度匹配：

1. **软分配**：后验概率 $\gamma_{ik}$ 天然给出每个体素的分类不确定性，速度相边界本质上是渐变的，硬边界（k-means）会丢失这一信息；
2. **完整协方差**：δlnVp–δlnVs 存在显著相关（热异常 vs 成分异常的判别正是基于二者比值），k-means 的等方差球形簇假设不成立，GMM full 协方差可拟合斜椭圆簇；
3. **概率框架**：提供似然，从而 BIC/AIC 的模型选择有理论依据；k-means 的 elbow/gap 准则更 ad hoc。

k-means 实为 GMM 在"等权重、等球形协方差、硬分配"下的极限特例，因此 GMM 严格更一般。代码保留了 k-means 对比通道（默认关闭）用于稳健性附录。

**局限（应写入论文）**：(i) 高斯分量假设——单一构造单元内扰动近似单峰，可接受，但分量与"物理相"不必一一对应，解释时以分量组合而非单分量为准；(ii) GMM 不建模空间相关，体素按 i.i.d. 处理（见 4.4 BIC 讨论）；(iii) 聚类在特征空间进行，空间连通性是结果涌现而非约束——这反而使"速度相在空间上成片"成为可检验的非平凡结论。

### 7.2 层次聚类 / DBSCAN 是否可用？

**层次聚类（agglomerative）**：
- 复杂度 O(N²) 内存与时间，对 10⁶–10⁷ 体素**直接不可行**；
- 可行的用法有二：(a) 对 ~10⁴ 子采样点做 Ward 层次聚类作为稳健性检验；(b) 对 GMM 分量均值 $\boldsymbol\mu_k$ 做层次聚类，得到速度相的树状谱系（dendrogram），辅助解释分量间亲缘关系。后者成本几乎为零，若审稿人质疑 K 的解释性，这是低成本补充。
- 缺点：无法自然地对新点/全量点做预测（需另加最近质心分配），也没有概率输出。

**DBSCAN / HDBSCAN**：
- 适用场景是"密度可分、含噪声、任意形状"的簇。而标准化后的 (δlnVp, δlnVs) 特征空间是**连续单峰为主的密度**（地幔扰动近高斯地集中于原点附近），不存在清晰的密度谷；DBSCAN 在这种数据上典型结果是"一个巨簇 + 少量离群噪声"，无法产出有构造意义的分区；
- eps/min_samples 对结果极敏感，且在 10⁶ 量级点上邻域查询内存开销大；
- 结论：**不适合作为本问题的主方法**。若审稿人问及，回答要点是"速度扰动特征空间是连续介质的连续统，密度聚类的'簇间密度间隙'前提不成立；我们需要的是对连续密度的分解（mixture decomposition），GMM 正是为此设计"。

### 7.3 收敛标准是什么？

见 4.3。一句话总结：**每个深度带、每个候选 K 的 GMM 以 EM 平均对数似然变化 < 10⁻³（≤300 次迭代、5 次随机重启取最优）为收敛；K 本身不通过"收敛"确定，而是 BIC 全区间扫描后按 ΔBIC 拐点判据（§4.4）选取**。全局报告的 `converged` = 所有带均收敛。

### 7.4 已识别的简化空间

已完成的简化（v7.4–v7.5）：

| 项 | 处理 |
| --- | --- |
| surface_units 定量评分（省栅格化 + NMI/纯度/边界掩膜 + 2-3-10 叠图） | v7.4 移除，地质省/地块线仅保留叠图 |
| facies–Slab2 定量评分（切片 MI + 名义标签梯度的 Spearman 边界对齐） | v7.5 移除（名义标签梯度幅值依赖任意编号，指标不可靠），Slab2 仅保留叠图 |
| `plot_concordance`（2-3-9 Slab2 符合度图，无调用点） | v7.5 随定量评分一并移除 |
| BIC 搜索中逐 K 计算 silhouette/DB/CH | v7.5 移除：搜索循环只算 BIC/AIC，诊断指标仅对最终模型算一次（最大提速点） |
| 六面板 BIC 决策图 | v7.5 简化为单面板 BIC 曲线 + 相对改善内嵌图 |
| argmin(BIC) 选 K（总贴 K 上界） | v7.5 改为 ΔBIC 拐点判据（§4.4），并移除循环内早停逻辑 |

剩余可简化项（不影响结果正确性的工程清理）：

| 项 | 现状 | 建议 |
| --- | --- | --- |
| `_apply_3d_sampling`（DataCube3DProcessor） | 未被调用（拟合采样由 `_subsample_for_fit` 完成） | 死代码，删除 |
| K-means 对比通道 | 默认关闭且分层模式下跳过 | 论文若不需要可整体删除，正文用 7.1 的理论论证替代 |
| 全深度（非分层）GMM 回退路径 | 与分层路径重复约 100 行 | 论文只用分层方案，可择机移除回退路径 |
| 跨带 BIC/AIC 的 nansum 汇总 | 不同数据子集的 BIC 相加无统计意义 | 只报每带 BIC，删除全局求和 |
| 双分层方案并存 | fixed_3band 与 moho_4band 均维护 | 正文定一个主方案，另一个降级为敏感性测试 |
| 概率场输出 `probabilities[:, :1]` | 仅保留 max_probability 有实际意义 | 明确只输出 p_max 与熵，删除占位的截断概率数组 |

---

## 8. 输出

- `labels_3d`（NetCDF/NPZ）：全局速度相标签体（带间偏移编码）；
- `max_probability_3d`：体素级分类置信度；
- 每带：K*（拐点判据）与 argmin(BIC) 参考值、BIC 曲线与相对改善序列、GMM 参数、最终模型聚类指标；
- 图件：3×3 深度切片（首格基岩地质底图，浅部叠地质线，深部叠 Slab2 等深线）、经度–深度剖面与三维体对比（叠 Slab2 几何）、单面板 BIC 决策图（含相对改善内嵌图与阈值线）、后验概率分布、簇剖面图（facies 推测解释标注于各子图标题下方）、簇心/特征分布图；输出至 `results/model_clustering/<scheme>/<model>/`。

---

## 9. 参考文献（方法相关）

- Dempster, A.P., Laird, N.M., Rubin, D.B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS B*, 39(1), 1–38.
- Schwarz, G. (1978). Estimating the dimension of a model. *Ann. Statist.*, 6(2), 461–464.
- Lekic, V., Cottaar, S., Dziewonski, A., Romanowicz, B. (2012). Cluster analysis of global lower mantle tomography. *EPSL*, 357–358, 68–77.
- Cottaar, S., Lekic, V. (2016). Morphology of seismically slow lower-mantle structures. *GJI*, 207(2), 1122–1136.
- Hayes, G.P., et al. (2018). Slab2, a comprehensive subduction zone geometry model. *Science*, 362(6410), 58–61.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
- Rousseeuw, P.J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *J. Comput. Appl. Math.*, 20, 53–65.
