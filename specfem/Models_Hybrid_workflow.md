# 混合模型正演流程 — CRUST1.0 共用地壳 + 各家地幔

在曙光 Slurm 集群上，让所有候选模型共用同一套 CRUST1.0 地壳，Moho 以下才换成各自的地幔，在 **10 个代表性震源**上做五组严格可比的正演，外加一个单事件的 GLL 往返对照。

本文档取代 `Models_GLL_workflow_server.md` 中 Phase 2–4 的做法。旧文档的编译、MPI、集群配置部分仍然有效。

**配套代码**（均在 `specfem/`）：

| 文件 | 作用 |
| --- | --- |
| `select_test_events.py` | 从 GCMT3D 目录中按构造省选出 10 个测试震源。见 §2.5 |
| `build_hybrid_gll.py` | 构建混合 GLL 模型，含五项自检。规范见 §5 |
| `check_mesh_diff.py` | 比对两套 `DATABASES_MPI` 的坐标段，既是诊断工具也是作业内的放行关卡。见 §6.5 |
| `run_all_test.sh` | 服务器端驱动：校验、编译、建目录、提交 MESH0 + 各组正演。见 §6 |
| `tests/test_depth_datum.py` | 深度基准、底图提取、Moho 对齐的验证。见 §4.5 |
| `tests/test_run_scripts.sh` | 本地校验 `run_all_test.sh` 生成的 Slurm 作业脚本。见 §6.17 |

**全流程在服务器上执行**，包括建模型。插值基准取自将要读取该模型的那次 mesher，保证坐标来源单一；服务器端因此需要可用的 Python（numpy / h5py / netCDF4），由 `PYTHON_BIN` 指向解释器绝对路径，见 §6.8。

**当前状态**：四个脚本已完成。C1–C4、C12 通过。probe 对照实验已定案（§6.16）：mesher 输出逐位确定性，网格几何与 `MODEL` 无关，此前反复出现的 mesher 崩溃根因是故障节点 c08r1，已通过 `BAD_RACKS` 黑名单规避。

2026-08-03 的方案调整分两批。第一批：共用地壳由 FWEA23 改为 CRUST1.0（§2），深度基准由 `R⊕(1−r)` 改为到局部自由表面的径向距离（§4.5），替换上界由固定 75 km 改为跟随 CRUST1.0 的 Moho（§4.2）——三项已用本地一套完整的 144 分片 `DATABASES_MPI` 验证通过。第二批：由单震源扩到 10 震源（§2.5），新增 `S362ANI_ref` 原生对照组（§2.3）。服务器端 MESH0 与各组正演待执行，C10/C11/C13 待验证。

---

## 1. 方案概述

**一句话**：不从 NC 重建整个模型，而是以一个已知正确的 GLL 模型为底，只替换关心的那部分；底图取 mesher 自己算出的 CRUST1.0 + s362ani，让所有候选模型站在同一套地壳上。

旧流程要回答「域外和深部填什么」，为此引入了两次 meshfem、从 `solver_data.bin` 反算弹性模量、扰动插值三套机制。本流程只跑一次额外的 mesher（MESH0），底图直接从它的 `solver_data.bin` 反算，扰动插值机制不再需要。

| | 旧流程 | 本流程 |
| --- | --- | --- |
| 底图来源 | mesher 算 s362ani，再从 κ/μ/ρ 反算速度 | 同样从 MESH0 的 κ/μ/ρ 反算，但**只用作底图，不再拿去做扰动插值** |
| meshfem 次数 | 2（s362ani + GLL） | 1 次 MESH0 定坐标 + 各组自己一次 GLL mesher |
| 扰动插值 | 需要，且参考场不匹配（见 §9.1） | 不需要，直接在绝对速度上做加权混合 |
| 共用地壳 | 无（各模型各自的地壳） | CRUST1.0，与网格的 Moho 拉伸一致 |
| chunk 几何 | 各模型不同，不可比 | 强制统一为 FWEA23 的 chunk |
| 深度基准 | `R⊕(1−r)`，含椭率与地形偏差 | 到局部自由表面的径向距离（§4.5） |

关于对 `solver_data.bin` 的依赖有一点要交代清楚。该文件的记录布局受 `ANISOTROPIC_3D_MANTLE`、`ATTENUATION`、`OCEANS`、`ABSORBING_CONDITIONS` 等开关影响，读到弹性模量记录时任何一个开关变动都会让偏移错位且不报错。早先的设计因此只读文件开头的 `nspec / nglob / x / y / z / ibool` 这几条无条件记录。改用 CRUST1.0 共用地壳后必须重新读到弹性模量记录，这个风险回来了——`build_hybrid_gll.py` 的应对是读完立刻检查六个参数是否落在物理区间（§5.3 的 S3），偏移一旦错位，反算出的"速度"会是天文数字或负数，当场被拦下。`tests/test_depth_datum.py` 的第 2 项也专门核对这一点。

---

## 2. 科学设计：六组正演 × 多震源

### 2.1 受控变量

三个模型如果各自用自己的 chunk、自己的背景、自己的地壳，评分差异里就混着流程差异，不可比。本方案把除「Moho 以下的地幔速度结构」以外的一切都固定下来：

| 变量 | 处理 |
| --- | --- |
| chunk 几何、网格拓扑 | 全部沿用 FWEA23 的 80°×80° / 中心 32°N,106°E / GAMMA=60° / NEX=288 / NSPEC=7236 |
| 地壳（Moho 以上） | 全部用 CRUST1.0，四个 `_c1` run 完全相同 |
| 替换区以外的地幔 | 全部用 s362ani，四个 `_c1` run 完全相同 |
| Q 衰减 | `MODEL=GLL` 根本不读 `qmu.bin`（`model_gll.f90` 里没有该字段），衰减一律来自参考模型，各组自动相同 |
| 震源、台站、频段、时窗 | 完全相同。同一组的 10 个事件共用一份逐位相同的模型与网格（一组只网格化一次，solver 循环全部事件），组内不引入额外变量 |
| **地幔速度（替换区内）** | **唯一变量** |

### 2.2 为什么共用地壳取 CRUST1.0

早先的方案让三个 run 共用 `model_updated` 的 FWEA23 地壳。改成 CRUST1.0 有五条理由，前三条是技术性的、后两条是方法论上的：

1. **网格本来就是按 CRUST1.0 的 Moho 拉伸的。** `moho_stretching_honor_crust`（`moho_stretching.f90:100-145`）用 CRUST1.0 的 Moho 深度变形地壳单元层，让真实 Moho 落在单元边界上。CRUST1.0 因此是唯一一个速度间断面正好压在单元边界上的地壳模型；换成别的地壳，间断面掉进单元内部，SEM 的表示精度下降。

2. **地壳值不经我们插值，深度基准误差不参与。** CRUST1.0 的值直接从 MESH0 的 `solver_data.bin` 反算，是 SPECFEM 自己在球面坐标上赋的值，按定义就是对的。§4.5 那个在地壳里能到 4–9% 的深度映射误差在地壳部分完全不存在。

3. **海域缺值、水层、CFL 三个问题一起消失。** CRUST1.0 全球覆盖，`model_crust_1_0` 自己处理水层，不需要有效数据掩膜也不需要 vs 下限；而 `DT = 0.1016261` 就是这套网格配 CRUST1.0 + s362ani 算出来的，用回它本身没有失稳风险。

4. **消除 FWEA23 的主场优势。** `BASE` 本身就是 FWEA23，若三个候选再共用 FWEA23 的地壳，等于让被测对象之一同时充当公共背景。CRUST1.0 是公开的标准地壳，论文里"所有候选地幔模型嵌入同一个 CRUST1.0 地壳"这句话经得起追问。

5. **SinoScope 终于被对称对待。** 它的 H5 最浅只到 20 km，本来就没有地壳，配一个标准地壳正是它被设计出来的用法。

**代价要如实记账**：CRUST1.0 是 1°×1° 的全球编译模型，比 FWEA23 的 FWI 地壳差不少，各组的绝对波形拟合都会变差，9–20 s 频段尤其明显。但各组同等变差，模型间的排名依然有效。`BASE` 组保留下来，用于报告"目前可得的最好拟合"。

另外注意替换区**以外**的地幔现在是 s362ani 而不是 FWEA23。这同样是各组共享的，不影响可比性，但意味着远场结构比早先的方案差。

### 2.3 六组正演

| Run | 名称 | 地壳 | Moho 以下（替换区内） | 事件数 | 用途 |
| --- | --- | --- | --- | --- | --- |
| 0 | `BASE` | FWEA23 | FWEA23（原生 GLL，不替换） | 10 | 链路校验；目前最好拟合的参照 |
| 1 | `S362ANI_c1` | CRUST1.0 | s362ani（不嵌任何模型） | 10 | 共用地壳下的空对照 |
| 2 | `FWEA23_c1` | CRUST1.0 | FWEA23 NC 插值 | 10 | 兼作插值误差基线 |
| 3 | `EARA2024_c1` | CRUST1.0 | EARA2024 NC 插值 | 10 | 与 2、4 严格可比 |
| 4 | `SinoScope_c1` | CRUST1.0 | SinoScope1.0 HDF5 插值 | 10 | 同上 |
| R | `S362ANI_ref` | CRUST1.0 | s362ani（原生 `MODEL=s362ani`） | 1 | GLL 往返代价的量度 |

四个判据分别由四对差异给出：

- **`S362ANI_ref` vs `S362ANI_c1`（同一事件）** — 「把 s362ani 从 mesher 提取成 GLL 文件，再由 mesher 读回来」这一往返的全部代价。两者的弹性场理论上逐位相同，衰减也同为 1DREF（`s362ani` 的 `REFERENCE_1D_MODEL` 与 `constants.h.in` 里的 `GLL_REFERENCE_1D_MODEL` 恰好都是 `REFERENCE_MODEL_1DREF`），所以波形上任何可见差异都只能来自往返本身。**这个量是一切组间差异的分辨率下限**：小于它的差异不能声称是模型差异。
- **`FWEA23_c1` vs `S362ANI_c1`** — 区域地幔模型相对全球背景带来多少改进。这是"做这件事到底值不值"的量度。
- **`FWEA23_c1` vs `BASE`** — 里面混着两样东西：换掉地壳（FWEA23 FWI 地壳 → CRUST1.0）的代价，以及 NC→GLL 插值误差。它给出共用地壳这个选择本身付出了多少拟合。
- **Run 2/3/4 之间** — 结构上完全对称（地壳相同、替换区外相同、替换区内都是 NC/H5 插值），可直接互比。判据仍是：

> **Run 2/3/4 之间任何小于插值误差本底的 misfit 差异，都不具备统计意义。**

插值误差本底的独立估计仍是 Open Issue，见 §9.4；`S362ANI_ref` 给出的是它的一个下界，不是它本身——往返误差只涵盖「GLL 写盘再读回」，不涵盖「NC 三线性插值到 GLL 点」。

`S362ANI_ref` 不跑自己的 mesher：MESH0 本来就是用 `MODEL=s362ani` 跑出来的，它的 `DATABASES_MPI` 就是原生 s362ani 网格，该组的工作目录把 `DATABASES_MPI` 软链过去直接跑 solver。正演 `SIMULATION_TYPE=1` 且 `SAVE_FORWARD=.false.` 时只读该目录，与其它组并发读安全。

### 2.4 为什么要 10 个震源

单个震源只照亮一条方位的窄条带。用一个事件排出来的名次，很可能只反映"谁在这条路径上碰巧更准"，换个方位就翻盘。10 个震源要同时满足三件事：

1. **深度分档** — 5 浅（≤30 km）/ 2 中（105、245 km）/ 3 深（367、542、601 km）。三个深源分别自北（日本海滞留板片）、东（中日本）、南（苏拉威西海）穿透地幔过渡带，构成 MTZ 的三向交叉照明；浅源负责地壳与岩石圈。
2. **方位覆盖** — 相对 chunk 中心的最大方位角空隙 71°，落在正北到东北（方位角 349°–60°，西伯利亚地台方向）。那里是稳定克拉通，本来就不产生 Mw≥5.5 的地震，这个缺口补不上。
3. **震源可用性** — 全部距 chunk 边界 ≥12°（避免吸收边界反射混入有效时窗），可用台站 130–237 个，自身方位角空隙 19–112°，前后半小时内无其它 Mw≥5.5 事件干扰（保证后续接观测数据时记录是干净的）。

**排名要在多数事件上一致才算稳。** 单个事件上的胜负受震源机制与路径影响很大，不足以下结论。

### 2.5 震源选取

`select_test_events.py` 从 GCMT3D 目录（`data/catalogs/CMT3D/cmt3d.txt`，9382 个事件，1994–2019）中选，流程是先硬筛再按构造省取最优：

```
9382 → 年份≥2000 → 7927 → Mw 5.5–7.0 → 7408 → 距 chunk 边界 ≥12° → 952
     → 可用台站 ≥40 → 952 → 方位角空隙 ≤200° → 941 → ±30 min 无干扰 → 843
```

843 个候选按 10 个构造省分组，每组取评分最高的一个。评分是五项归一化指标的加权和：台站覆盖 0.35、方位角均匀性 0.25、离 chunk 边界的余量 0.15、震级适中程度（以 Mw 6.2 为中心的高斯）0.15、年份新近度 0.10。

选 GCMT3D 而不是 GCMT，是因为前者的质心位置是在三维地球（GLAD-M25）中反演得到的，去除了 1D 定位带来的系统偏移——否则震源侧的定位误差会被误读成模型差异。

| # | 事件 | 位置 | 深度 km | Mw | 机制 | 台站 | 构造省 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `C200604300043A` | 44.5N 102.5E | 9.7 | 5.68 | 正断 | 217 | 蒙古/华北 |
| 2 | `C201208121047A` | 35.9N 82.5E | 8.9 | 6.16 | 逆冲 | 203 | 西昆仑/帕米尔 |
| 3 | `C201301090141A` | 25.1N 94.9E | 104.8 | 5.83 | 正断 | 231 | 缅甸板片 |
| 4 | `C201403211341A` | 7.7N 94.3E | 10.9 | 6.41 | 走滑 | 207 | 安达曼/巽他 |
| 5 | `C201707121948A` | 40.8N 131.6E | 542.1 | 5.91 | 走滑 | 150 | 日本海深震 |
| 6 | `C201811040755A` | 7.8N 123.8E | 600.6 | 6.01 | 逆冲 | 162 | 苏拉威西海深震 |
| 7 | `C201904180501A` | 24.1N 121.5E | 28.1 | 6.17 | 正断 | 158 | 台湾 |
| 8 | `C201904232015A` | 28.3N 94.6E | 14.6 | 5.99 | 斜滑 | 237 | 藏东南 |
| 9 | `C201907130057A` | 29.4N 128.2E | 244.6 | 6.11 | 逆冲 | 162 | 东海中源 |
| 10 | `C201907271831A` | 33.2N 137.2E | 367.1 | 6.33 | 正断 | 130 | 中日本深震 |

机制类型为正断 4、逆冲 3、走滑 2、斜滑 1，辐射花样不集中。半持续时间 1.7–3.9 s，对 9–120 s 的分析频段都远在震源谱拐角之下。

`S362ANI_ref` 用 #8（藏东南，237 个台站、方位角空隙 22.8°，覆盖最好）做往返对照。

输出在 `specfem/events_test/`：`CMTSOLUTION/`（标准格式）、`CMTSOLUTION_ECEF/`（供 `USE_ECEF_CMTSOLUTION=.true.` 的本地改造版使用，由 `change_cmt_to_ECEF.py` 转换）、`events.list`、`selected_test_events.csv`、`3-1_test_events_map.jpg`。整个目录上传到服务器的 `$EVENTS_ROOT`。

**两点要留意**：

- 台站表里有 40 个台落在 chunk 之外（瓦努阿图、所罗门、巴新、北澳、马绍尔、迪戈加西亚、科科斯），SPECFEM 会静默丢弃它们——原作者的参考运行报 "Total number of receivers = 293" 却只写出 256 个台的 SAC。这不影响可比性（各组一致），但统计台站数时要按 253–256 算，不是 293。
- `RECORD_LENGTH_IN_MINUTES = 30` 对 40° 处的长周期面波刚好卡住尾巴（3.0 km/s 群速度走 40° 需 24.7 min，盆地内更慢）。所以选源时的台站覆盖统计把震中距窗口设在 3°–40°，分析时也不要指望更远台站的面波是完整的。改 `RECORD_LENGTH` 会让新结果与已有的 `BASE` 参考波形不可直接比，因此维持 30 min。

### 2.6 结论的正确表述

本方案测的不是「EARA2024 这个模型的波形拟合有多好」，而是「在同一套 CRUST1.0 地壳之上，把地幔换成 EARA2024 之后拟合变好还是变坏」。论文里必须按后者写。

同时要说明一个必然的副作用：因为四个 `_c1` run 共享地壳和替换区外的背景，任何模型都不可能在这些区域拉开差距，最终得分差异会小于「各自独立正演」的情形。这是控制无关变量后的正常结果，不是缺陷，但审稿人会问。

另一方面，这些混合模型本身就是合法的 SEAM 1.0 初始模型候选，所以这一步同时完成了总纲 Phase D（评估）和 Phase E（融合）的一部分。

---

## 3. 前置条件

### 3.1 `model_updated/` 核实结果（已确认）

```
144 个 proc × 8 种参数
参数: vpv, vph, vsv, vsh, eta, rho, qmu, vs
单文件 3,618,008 字节 = 8 + 125 × 7236 × 4  →  NSPEC = 7236，单精度
```

单精度与编译配置一致（`run_forward.bash` 用 `CUSTOM_REAL=SIZE_REAL`），无需 `--double`。

### 3.2 网格与坐标（已定案：mesher 输出是确定性的）

`FWEA23/DATABASES_MPI/` 中已有 144 个 `proc*_reg1_solver_data.bin`，实测 `nspec = 7236`、`nglob = 485401`，与 `model_updated` 匹配。

这一节曾两次改写。2026-08-02 观察到不同 run 之间 `solver_data.bin` 坐标段不逐位相同（56/144、4/144 等），据此判断 mesher 输出不可复现。**2026-08-03 的 probe 对照实验（§6.16）证明那个判断是错的**：在健康节点上连跑四轮 mesher（s362ani、GLL 各两轮），144 个 proc 的坐标段**全部逐位相同**，跨 MODEL 比对同样逐位相同。

| probe 比对（c10r3，144 proc） | 逐位相同 | ibool 有差异 | 坐标相对差 max |
| --- | --- | --- | --- |
| s362ani 轮 1 vs 轮 3 | 144 / 144 | 0 | 0.0 |
| GLL 轮 2 vs 轮 4 | 144 / 144 | 0 | 0.0 |
| s362ani 轮 1 vs GLL 轮 2 | 144 / 144 | 0 | 0.0 |

由此确立三条事实：

1. **mesher 在给定 Par_file 和二进制下是逐位确定性的**，重跑必然复现。
2. **网格几何与 `MODEL` 取值无关**，所以 mesh0 用 s362ani 还是 GLL 建基准坐标都可以，不影响后续插值。
3. 之前观察到的坐标不一致，来自故障节点上跑出的**损坏输出**，不是 mesher 的固有性质。那批数据全部作废（§6.16）。

坐标核对机制因此保留，但定位变了：它不再是"防范不可复现"，而是**故障节点的探测器**。正常情况下必然 144/144 逐位相同；一旦出现差异，说明这次 mesher 跑在了有问题的硬件上，其输出不可信，应换节点重跑而不是调大容差放行。

风险本身依然成立：`nspec` 与 `nglob` 在坐标损坏时仍然相同，SPECFEM 全程不报错。模型按 A 网格坐标插值却交给 B 网格的 solver 读取，速度值会安静地落在错误的空间位置，波形照算，只是不对应任何真实模型。这类错误没有运行时症状，只能靠显式核对发现，所以核对这一步不能省。

> ⚠️ `FWEA23/DATABASES_MPI/` 与 `mesh0/DATABASES_MPI/` 都不能被任何 run 写入——每个 run 的 mesher 只写自己目录下的 `DATABASES_MPI/`。

### 3.3 chunk 真实几何（已实测，144 proc 全量）

`GAMMA_ROTATION_AZIMUTH = 60°` 使 chunk 成为旋转菱形，几何估算严重失真：

| | 天真估算（中心 ± 宽度/2） | **实测** |
| --- | --- | --- |
| lat | −8° ~ 72° | **−16.55° ~ 77.17°** |
| lon | 66° ~ 146° | **54.08° ~ 165.13°** |
| depth | — | −8.4 ~ 2896.5 km |

GLL 总点数 130,248,000，与 144 × 7236 × 125 完全吻合。

由此得到两个结论：

1. **EARA2024 的东界 160°E 落在 chunk 内**（chunk 到 165°E），§9.2 原先的疑问解除。
2. chunk 远大于任何一个模型的覆盖范围——北到北极圈、南到印度洋。这些区域没有任何模型有数据，全部由 `model_updated` 自身的背景承担。这正是本方案相对旧流程（那里要填 1DREF）的主要优势所在。

### 3.4 替换区占比（已实测，144 proc 全量）

现行参数（水平外边界 lat [10,55]、lon [80,150]、taper 5°；上界跟随 CRUST1.0 Moho、下界 1000 km、taper 上 15 / 下 50 km），GLL 总点数 130,248,000：

| 区域 | 点数 | 占比 |
| --- | --- | --- |
| $w > 0$（受影响，含过渡带） | 29,856,339 | 22.92% |
| $w = 1$（完全替换） | 16,723,625 | 12.84% |

即四个 `_c1` run 之间有 77% 的 GLL 点逐位相同，12.8% 被完全替换，其余 10.1% 处于过渡带。上界从固定 75 km 改成跟随 Moho 后，受影响占比从 16.65% 升到 22.92%，多出来的正是 Moho 到 75 km 之间那段岩石圈地幔。

这仍然意味着射线路径不穿过替换区的台站-事件对，各 run 的波形会几乎相同，对排名没有贡献。后处理时必须统计有效数据量，见 §9.3。

按 proc 统计（旧参数下实测），144 个 proc 中有 81 个被修改、63 个完全未动，且被修改的 proc 编号成连续条带（5–9、16–22、27–34…），符合 12×12 proc 网格上一个矩形区域的预期。

### 3.5 模型传递误差基线（旧参数下实测，结论待重做）

> ⚠️ 下表是在**旧方案**（底图 `model_updated`、固定 75 km 上界）下测的，改用 CRUST1.0 共用地壳 + Moho 上界后数值会变，但结论的方向不变。基线本身的定义也需要调整，见 §9.4。

用 `build_hybrid_gll.py --dry-run` 统计各模型嵌入后相对底图的扰动幅度，只在 $w>0$ 的点上计算：

| Run | vsv peak | **vsv rms** | vpv rms | rho rms |
| --- | --- | --- | --- | --- |
| FWEA23（**基线**） | 7.26% | **0.78%** | 0.66% | 0.76% |
| EARA2024 | 25.46% | **1.40%** | 1.00% | 0.73% |
| SinoScope | 13.84% | **2.37%** | 2.21% | 1.87% |

FWEA23 一行是把 FWEA23 自己发布的 NC 重新嵌回它自己的 GLL 底图，理想情况下应当为 0。实测 rms 0.78% 就是纯粹的**模型传递误差**——它同时包含三线性插值误差，以及「发布的 NC 是 `model_updated` 的重采样/平滑版本，或干脆是不同迭代次数的产物」这两种可能。

这个数字**占 EARA2024 真实信号（1.40%）的 56%**。若没有这一组作对照，很容易把全部 1.40% 当成模型间的真实差异，这说明基线不是可选项。

需要注意的是模型空间的 rms 与波形空间的 misfit 不是线性关系：0.78% 的速度扰动在长周期面波上可能只造成很小的走时差，也可能因沿路径累积而放大。最终的基线要以**波形差异**为准，上表只用于事前判断信噪比是否值得跑。

### 3.6 Par_file 关键项（不得改动）

```
NCHUNKS                       = 1
ANGULAR_WIDTH_XI_IN_DEGREES   = 80.d0
ANGULAR_WIDTH_ETA_IN_DEGREES  = 80.d0
CENTER_LATITUDE_IN_DEGREES    = 32.d0
CENTER_LONGITUDE_IN_DEGREES   = 106.d0
GAMMA_ROTATION_AZIMUTH        = 60.d0
NEX_XI / NEX_ETA              = 288
NPROC_XI / NPROC_ETA          = 12        →  144 MPI
```

以上任何一项改动都会让 `model_updated` 失效（NSPEC 不匹配，mesher 报 `requires too much data`）。

---

## 4. 替换规则（核心规范）

### 4.1 参数必须整组替换

SPECFEM 内部最终用的是弹性模量：

$$\mu_v = \rho V_{sv}^2, \quad \mu_h = \rho V_{sh}^2, \quad \kappa_v = \rho\left(V_{pv}^2 - \tfrac{4}{3}V_{sv}^2\right), \quad \kappa_h = \rho\left(V_{ph}^2 - \tfrac{4}{3}V_{sh}^2\right)$$

在同一个 GLL 点上，`vpv / vph / vsv / vsh / eta / rho` 这**六个必须整组来自同一个模型**。只换 Vs 而保留底图的 Vp，会做出一个不属于任何真实模型的 Vp/Vs 比；极端情况下 $V_p^2 < \frac{4}{3}V_s^2$ 使体积模量为负，solver 直接发散。

各模型的参数补全规则：

| 模型 | NC/H5 提供 | 补全方式 |
| --- | --- | --- |
| FWEA23 | vpv, vph, vsv, vsh, eta, rho | 无需补全 |
| EARA2024 | vpv, vph, vsv, vsh, eta, rho | 无需补全 |
| SinoScope1.0 | vp（各向同性）, vsv, vsh, rho | `vpv = vph = vp`；`eta = 1.0`；速度 m/s → km/s |

补全值一律来自模型自身的约定，**不得从底图借用**。

### 4.2 空间替换范围与三维 taper

替换权重 $w \in [0,1]$，输出为

$$V_{\text{out}} = w \cdot V_{\text{model}} + (1-w)\cdot V_{\text{base}}$$

**同一个 $w$ 必须施加到全部六个参数上**，否则 §4.1 的一致性会在过渡带内被破坏。

$w$ 由水平和垂直两个方向的权重相乘得到，两者都用升余弦（raised cosine）过渡，不用硬边界——硬边界会在计算域内部造出速度台阶，散射出假震相。

$$w = w_{\text{horiz}} \cdot w_{\text{depth}} \cdot \mathbb{1}[\text{点在模型数据域内}]$$

**taper 方向是向内的。** `--region` 与 `--depth-range` 给出的是 $w=0$ 的**外边界**，权重从边界向内经 taper 宽度升到 1。这一点在实现时做了修正：最初设计是从核心区向外渐变，但那样过渡带会伸出模型数据域（例如 FWEA23 的 NC 只到 lat 55°，向外 taper 到 60° 的部分没有数据），随后被数据域掩码强行归零，反而在计算域内部制造出硬边界——正是 taper 想要避免的东西。向内 taper 保证过渡完整落在数据域里。

**水平方向**：外边界取三模型 NC 覆盖的交集，lat [10°, 55°]、lon [80°, 150°]。taper 宽度 $L_h$ 默认 5°（约 550 km，覆盖 40–120 s 面波若干波长），因此满权重区为 lat [15°, 50°]、lon [85°, 145°]。

**垂直方向：上界跟随 CRUST1.0 的 Moho**（`--depth-min-mode moho`）。逐点取

$$d_{\text{top}}(\text{lat}, \text{lon}) = d_{\text{Moho}}^{\text{CRUST1.0}}(\text{lat}, \text{lon}) + \texttt{moho-offset}$$

权重从 $d_{\text{top}}$ 起向下经 15 km 升到 1，下界 1000 km 处向上经 50 km 降回 0。实测替换区内 Moho 深度跨度为 6.5 ~ 76.7 km，因此满权重区的上界随地点在 21 ~ 92 km 之间浮动。

早先用的是固定 75 km（CRUST1.0 全球最深 Moho）。改成跟随 Moho 之后，Moho 到 75 km 之间那段岩石圈地幔归各模型自己所有——那正是区域模型差异最大的深度之一，固定上界等于把它让给了共用背景。

Moho 深度从 `DATA/crust1.0/crust1.bnds` 读。基准与 SPECFEM 一致：`model_crust_1_0.f90:181` 把 moho 算成沉积层加三层结晶地壳的**厚度之和**（不含水层与冰层），即地表以下深度而非海平面以下深度，对应 `crust1.bnds` 的第 3 列减第 9 列。

**15 km 的 taper 宽度是有来由的**，它要盖住两处不确定性：我们用双线性读 `crust1.bnds`，而 SPECFEM 查询时做 CAP 平滑（`crust_1_0_CAPsmoothed`），两者在 Moho 起伏剧烈处可差几 km；深度基准本身在地壳内还有约 1 km 残差（§4.5）。取 0 宽度做硬切换在理论上也成立（Moho 本就是间断面且压在单元边界上），但那样结果会直接暴露在上述几 km 的不确定性下。

脚本会在启动时检查替换区是否完整落在模型数据域内，不满足则告警。moho 模式下用的是区内最浅的 Moho（6.5 km）与模型数据域上界比较。

### 4.3 qmu 不参与

`MODEL = GLL` 的读取例程 `model_gll.f90` 里没有 `qmu` 字段——GLL 模型根本不覆盖衰减，`qmu.bin` 放不放都不影响结果，衰减一律来自参考模型。脚本仍会复制一份 `model_updated` 的 `qmu.bin` 到输出目录，纯粹是为了让输出目录内容完整，不产生任何物理效果。

EARA2024 和 SinoScope 都未发布 Q 模型，这一层因此也无需处理。

### 4.4 输出格式

与 `model_updated` 完全一致：Fortran 无格式记录，单精度，每个文件 `8 + 125 × nspec × 4` 字节。直接复用 `convert_nc_to_gll.py` 里的 `write_gll_binary()`。

`vs.bin` 是原作者额外提供的各向同性 Vs，SPECFEM 的 `MODEL = GLL` 不读它，可以不生成。

### 4.5 深度基准：到局部自由表面的径向距离

这是 2026-08-03 修掉的一处系统性错误，也是能做地壳替换的前提。

**问题**。mesher 的调用顺序写在 `compute_element_properties.f90:217-297`：

```
compute_element_GLL_locations   ← 此时网格仍是完美球面
get_model                       ← (a) 在球面坐标上查模型并赋值
add_topography_gll              ← (b) 加地形
get_ellipticity_gll             ← (c) 加椭率
```

也就是说 **SPECFEM 查模型用的是完美球面半径，而写进 `solver_data.bin` 的坐标是加完地形和椭率的**（`save_arrays_solver.f90` 在 (c) 之后才写）。早先直接用 $\text{depth} = R_\oplus(1-r)$ 把后者当成前者，引入的偏差有两部分：

- **椭率**：因子 $A = 1 - \tfrac{2}{3}\,\varepsilon\,P_2(\cos\theta)$ 在赤道让半径涨约 7 km、在 60°N 缩约 9 km，研究区内跨度 16 km，且**随纬度系统性变化**——不是随机噪声，不会在统计中抵消。
- **地形**：另外 ±5 km 量级。

在地幔里这不要紧（速度梯度约 0.002 km/s 每 km，9 km 深度误差只值 0.3%），但在浅部地壳梯度可达 0.02–0.04 km/s 每 km，同样的误差就是 4–9% 的速度误差，p95 能到 38%。这正是早先不敢碰地壳的技术原因之一。

**解法**。地形和椭率都是**纯径向缩放**，所以用同一根径向线上的自由表面半径作参照，两者会一起约掉。记球面深度 $u = 1-r_s$、归一化地形 $e$、椭率因子 $A$，把变形关系两式相减可得

$$\frac{r_{\text{surf}} - r_{\text{def}}}{A} = \begin{cases} u\,(1 + e/u_{220}) & R220 \text{ 以上（地形拉伸区）} \\ u + e & R220 \text{ 以下} \end{cases}$$

两式在 $u = u_{220}$ 处连续。$A$ 由纬度解析给出，$e$ 再从 $r_{\text{surf}}/A - 1$ 反解，于是 $u$ 可逐点求出。

自由表面半径 $r_{\text{surf}}$ **直接从网格自身取**，不去复现 SPECFEM 对 ETOPO 做过的插值与平滑：扫一遍全部分片，按 0.1° 格子统计最大半径。reg1 是横向分域的，每个 slice 都从地表贯通到 CMB，格子里的最大半径就是该处的地表。这样做的关键好处是**自洽**——无论 SPECFEM 对地形做过什么处理，我们都自动跟它一致。

有一个坑：chunk 侧边界上，同一根径向线的浅部与深部点可能被舍入到相邻格子，个别格子里只剩深部点，其"最大半径"根本不是地表（实测出现过 0.723，相当于地下 1764 km）。这类格子必须在最近邻填补**之前**剔除，否则错值会扩散到整片空格，之后任何邻域判据都失效。判据取 1° 窗口的局部上包络减 8 km：真实地形在 1° 内起伏最多约 4 km，而最近的单元层面在 15 km 深，两者分得开。

**验证**（`tests/test_depth_datum.py`，本地一套完整的 144 分片 `DATABASES_MPI`）：

| 检查项 | 结果 |
| --- | --- |
| 自由表面（应为 0 km） | 新定义 −0.02 ~ −0.13 km；旧定义 −7.6 ~ +11.5 km |
| CMB（应为 2891 km） | 新定义 −1.48 ~ +0.97 km；旧定义 −2.8 ~ +5.5 km |
| 底图 Moho 跳变 vs CRUST1.0 Moho | **+0.2 km**（vsv 越过 4.3 km/s 处，档宽 2.5 km） |

最大偏差从 11.5 km 压到 1.5 km。最后一项是端到端判据，同时验证了深度映射、`crust1.bnds` 的读取和底图提取三件事。

**残余误差**只剩一项：把 $\varepsilon(r)$ 当常数。$\varepsilon$ 从地表的 1/299.8 降到 CMB 的约 0.0026，因此这项误差**随深度增长、在地表为零**——CMB 处约 1 km，1000 km 深处约 $0.8\,P_2$ km，地壳内可忽略。恰好在我们最在意的地方最小。

脚本在写出前会核对深度的两端（自由表面与 CMB），偏差超过 3 km 就拒绝写出。

---

## 5. 实现：`build_hybrid_gll.py`

### 5.1 定位

新脚本，与 `convert_nc_to_gll.py` 并列放在 `specfem/`。它**复用**后者已经验证过的这些函数，不重复造轮子：

| 复用函数 | 用途 |
| --- | --- |
| `read_solver_data(path, extract_velocities=...)` | 读 GLL 点坐标；`True` 时继续读到弹性模量记录，反算 CRUST1.0 + s362ani 速度场作底图 |
| `read_gll_binary(path, nspec)` | 读 `model_updated` 的底图（`BASE` 组与 `--base-from-mesh` 关闭时） |
| `write_gll_binary(path, data)` | 写输出 |
| `load_h5_as_dataset(path)` | SinoScope HDF5 → xarray Dataset |
| `_fill_nan_nearest_3d(arr)` | 插值前填补 NC 内部 NaN，避免污染 stencil |

**不复用** `interpolate_from_netcdf()` 的扰动分支和域外填充逻辑——那些是为旧流程设计的，新流程只需要纯绝对值插值 + 域外 $w=0$。

### 5.2 逐 proc 处理流程

```
先做一遍全局前置扫描（只做一次，不在 proc 循环内）:

  A. 扫全部 144 个分片的坐标，按 0.1° 格子建自由表面半径查表（§4.5）
  B. 读 crust1.bnds，建 CRUST1.0 Moho 网格（moho 模式下）

对 proc_id in 0..143:

  1. 读坐标（--base-from-mesh 时同时反算出底图速度）
     nspec, nglob, x, y, z, ibool, ref_vel = read_solver_data(...)
     lat, lon, r = xyz_to_lat_lon_radius(x, y, z)       # 在 nglob 个唯一点上算
     depth = datum.to_depth(lat, lon, r)                # 到局部自由表面的径向距离
     idx = ibool.ravel(order='F') - 1                   # 再按 ibool 展开成 (5,5,5,nspec)

  2. 底图（6 个参数）
     --base-from-mesh:  base[v] = ref_vel[v]            # CRUST1.0 + s362ani
     否则:              base[v] = read_gll_binary(model_updated/...)

  3. 计算替换权重 w = w_horiz(lat, lon) × w_depth(depth) × 1[点在模型域内]
     moho 模式下 w_depth 的上界逐点取 moho.query(lat, lon) + offset
     w 为标量场，对六个参数共用
     未指定模型时 w 恒为 0，输出即底图（S362ANI_c1 组走这条路）

  4. 插值模型值（仅在 w > 0 涉及的唯一点上算）
     RegularGridInterpolator(method='linear', bounds_error=False)
     按 §4.1 补全缺失参数（SinoScope 的 vpv/vph/eta，单位换算）

  5. 混合（只在 mask = w>0 内赋值，w=0 处保持底图的原始比特）
     out[v][mask] = w[mask] * model[v][mask] + (1 - w[mask]) * base[v][mask]

  6. 一致性自检（见 §5.3），任一项失败则该 proc 不写出，并报出失败点数

  7. 写出 6 个参数；qmu 直接复制底图文件（不参与计算，见 §4.3）

全部完成后核对深度两端：自由表面应为 0、CMB 应为 2891 km，偏差 > 3 km 拒绝写出
```

第 1 步的 `ibool` 展开是主要的性能优化：GLL 点共 904,500 个，但唯一全局点只有 485,401 个（元素面上的点被相邻元素共享）。在唯一点上做坐标变换与插值再展开，省掉约 46% 的计算。

144 个 proc 之间完全独立，用 `multiprocessing.Pool` 并行，模型网格在 worker 初始化时加载一次并复用，避免每个 proc 重复读 NC。

### 5.3 自检项（写出前必须全部通过）

| # | 检查 | 判据 | 失败含义 |
| --- | --- | --- | --- |
| S1 | 无 NaN / Inf | `np.isfinite(out[v]).all()` | 插值或补全逻辑有洞 |
| S2 | 体积模量为正 | `vpv > 1.1547 * vsv` 且 `vph > 1.1547 * vsh` | 参数混搭，solver 必发散 |
| S3 | 值域合理 | 只检查 `w>0` 的点，容许区间 = 物理区间 ∪ 底图自身包络 | 单位换算错误 |
| S4 | **底图区逐位一致** | `w == 0` 的点上 `out[v]` 与 `base[v]` 完全相等 | 混合逻辑污染了不该动的区域 |
| S5 | 替换比例合理 | 报告 `w>0`、`w==1` 的点数占比 | 占比为 0 说明替换区设错，chunk 与 NC 没交集 |

S4 是最有价值的一项：它把「各组在替换区外完全相同」这个实验设计前提变成了可执行的断言，而不是靠约定。实现上不用 `w*model + (1-w)*base` 的算式在 $w=0$ 处「自然」得到底图值（浮点上 `0*NaN = NaN`），而是显式只在 `w>0` 的掩码内赋值，让 S4 恒真。

S3 的判据在实测中修正过。原先用固定物理区间 `rho ∈ [1, 6]`、`vpv ∈ [1, 15]` 检查整个数组，结果 26 个 proc 报警——但那些异常点来自**底图自身**：`model_updated` 的近地表含有 rho 低至 0.35 g/cm³、vp 低至 0.75 km/s 的 GLL 节点（海水层与 FWI 更新后的浅部节点）。这些值是原作者模型的既有事实，且该模型能正常跑完正演，不是本流程引入的问题。因此判据改为「只检查被修改的点，且容许区间放宽到物理区间与底图包络的并集」，语义是**混合结果不得比底图本身更离谱**。

### 5.4 CLI

```bash
python build_hybrid_gll.py \
    --mesh-dir        mesh0/DATABASES_MPI \
    --base-gll-dir    ../model_updated \
    --output-dir      DATA/GLL \
    --base-from-mesh \
    --model-nc        ../models/EARA2024.r0.0-n4.nc \
    --region          10 55 80 150 \
    --horiz-taper     5.0 \
    --depth-range     0 1000 \
    --depth-taper     15 50 \
    --depth-min-mode  moho \
    --moho-offset     0 \
    --crust1-bnds     ../specfem3d_globe_code_new/DATA/crust1.0/crust1.bnds \
    --nproc           144
```

| 选项 | 说明 |
| --- | --- |
| `--mesh-dir` | 含 `proc*_reg1_solver_data.bin` 的目录 |
| `--base-gll-dir` | 磁盘底图，即 `model_updated/`。`--base-from-mesh` 时只用来取 `qmu` |
| `--output-dir` | 输出 GLL 目录 |
| `--base-from-mesh` | 弹性参数底图改为从 `--mesh-dir` 的 `solver_data.bin` 反算，即 mesher 当时用的 CRUST1.0 + s362ani。**本流程的四个 `_c1` 组都开启** |
| `--model-nc` / `--model-h5` | 待嵌入的模型（二选一；都省略 = 只把底图落盘，即 `S362ANI_c1` 组） |
| `--region lat0 lat1 lon0 lon1` | 水平**外边界**（该处 $w=0$），默认 `10 55 80 150` |
| `--horiz-taper` | 水平余弦过渡宽度（度，向内），默认 5.0 |
| `--depth-range d0 d1` | 深度**外边界**（km）。moho 模式下上界被忽略，只有下界生效 |
| `--depth-taper wtop wbot` | 上下过渡宽度（km，向内），本流程用 `15 50` |
| `--depth-min-mode` | `fixed` 用 `--depth-range` 的上界；`moho` 逐点跟随 CRUST1.0 Moho |
| `--moho-offset` | moho 模式下相对 Moho 的偏移（km，正为更深），默认 0 |
| `--crust1-bnds` | `crust1.bnds` 路径，省略时在源码树的 `DATA/crust1.0/` 下自动查找 |
| `--surface-bin` | 自由表面查表的格距（度），默认 0.1 |
| `--nproc` | proc 数，默认 144 |
| `--jobs` | 并行进程数，默认 CPU 核数。每 worker 常驻约 200 MB，加载期峰值约 1 GB |
| `--skip-unchanged` | 与底图逐位相同的文件不落盘。为早期的本地建模流程而设，**本流程不使用**（§6.4）；`--base-from-mesh` 时会自动忽略它，因为那份底图根本不在磁盘上 |
| `--dry-run` | 只统计不写文件，输出 S5 占比与相对底图的扰动幅度 |

`--dry-run` 用于覆盖诊断，§3.4 与 §3.5 的数字都由它产生。

运行时间参考：144 proc、8 并行，`--base-from-mesh` + moho 模式全量 dry-run 约 27 秒（含建自由表面查表 1 秒、模型加载 7 秒）。写文件时每组输出 1008 个文件、约 3.1 GB，四个 `_c1` 组共约 12.5 GB。

---

## 6. 执行流程：全流程服务器端

模型在**服务器上、作业内**用 `build_hybrid_gll.py` 生成，插值坐标直接取自当次流程刚跑出来的 mesher 输出。服务器端需要可用的 conda 环境（`eastasia_fwi`，含 numpy / h5py / netCDF4）。

> **本节相对早期版本是一次结构性改写。** 早先的设计是本地建模型、`--skip-unchanged` 压缩上传量、服务器端按 `manifest.txt` 组装并核对网格指纹。当时指纹核对持续失败，被解读为"mesher 输出不可复现、本地坐标不可能等于服务器坐标"，于是整体搬到服务器端。
>
> 后来 probe 实验证明 mesher 其实是确定性的（§6.16），指纹失败的真实原因是那些坐标产自故障节点。**但改到服务器端仍然是对的选择**：本地与服务器的编译器、指令集、`constants.h.in` 都可能不同，"本地坐标等于服务器坐标"从来就不是能免费成立的前提；而且服务器端建模让插值基准与 solver 读取的网格天然同源，比核对两份独立产出的坐标更可靠。这一节不回退。

**命令一览**

```bash
./run_all_test.sh events      # 列出 10 个震源
./run_all_test.sh prepare     # 校验路径/Python/全部震源，编译一次，建目录，写作业脚本
./run_all_test.sh probe       # 可选：mesher 对照实验，排查节点硬件（§6.16）
./run_all_test.sh submit      # 提交 MESH0；基准网格过校验后由它自行提交各组
./run_all_test.sh status      # 队列 + 各组进度 + 逐事件完成矩阵
./run_all_test.sh collect     # 核查 sac/<TAG>/<EVENT>/ 的 SAC 是否齐全、台站数是否一致
./run_all_test.sh resubmit <TAG>   # 单组重投，已完成的事件自动跳过
```

三个强制开关：`FORCE_BUILD=1` 重新编译，`FORCE_MESH=1` 重跑 mesher，
`FORCE_BUILD_GLL=1` 重建模型，`FORCE_SOLVER=1` 重算已完成的事件。平时都不需要——
所有阶段都是幂等的，重投时自动接着上次的进度走。

`status` 给两张表。第一张按组汇总，`事件` 一列是「已完成 / 应完成」，`DT` 一列用来核对
各组时间步是否一致（不一致波形不在同一采样上，misfit 不能直接比）：

```
TAG               GLL solver_data     事件          DT 状态
BASE              144         144       10/10  0.1016261 ✅ 全部完成
S362ANI_c1        144         144        7/10  0.1016261 🔄 正在跑事件
FWEA23_c1         144         144        0/10  0.1016261 🔄 已网格化
S362ANI_ref         -         144         1/1  0.1016261 ✅ 全部完成
```

第二张是逐事件矩阵，一行一事件、一列一组，格子里是该组该事件的 SAC 数，`-` 表示还没做、
`n/a` 表示这一组不跑该事件（只有 `S362ANI_ref` 会出现）。哪一组卡在哪个事件上一眼可见。

### 6.1 总流程

```
       MESH0（mesher, MODEL=s362ani）
                │
                └──> mesh0/DATABASES_MPI   坐标基准 + CRUST1.0/s362ani 底图来源
                │                                        │
   ┌────────┬───┴────────┬─────────────┬──────────────┐  └─> S362ANI_ref
   │        │            │             │              │       （软链复用，
 BASE   S362ANI_c1   FWEA23_c1   EARA2024_c1   SinoScope_c1     直接跑 solver）
   │        │            │             │              │
（DATA/GLL  └──── build_hybrid_gll.py --base-from-mesh ────┘
 -> 底图）         （--mesh-dir = mesh0/DATABASES_MPI）
   │        │            │             │              │
   └──── mesher (MODEL=GLL)，各写各的 DATABASES_MPI ─────┘
                │
        check_mesh_diff.py  核对是否复现 mesh0 的坐标
                │  └── 不通过则中止，不启动 solver
                │
        solver × 10 events   逐事件换 CMTSOLUTION，SAC 收进 sac/<TAG>/<EVENT>/
```

五组待遇完全一致，唯一的差别是 `DATA/GLL` 的来源：BASE 直接软链 `model_updated`，
其余四组由 `build_hybrid_gll.py` 生成。**一组只网格化一次，10 个事件在同一份
`DATABASES_MPI` 上依次跑 solver**——这既省掉九次 mesher，也保证组内各事件用的是逐位
相同的模型。

`S362ANI_ref` 是例外：它的 `DATABASES_MPI` 软链到 mesh0，跳过阶段 1–3，只对
`ROUNDTRIP_EVENT` 跑一次 solver。

各组不用 Slurm 依赖，改由 MESH0 在基准网格通过校验后自行提交，原因见 §6.15。

### 6.2 MESH0：坐标基准兼底图来源

MESH0 用 `MODEL = s362ani`（配置项 `MESH0_MODEL`）跑一次 mesher，产出
`mesh0/DATABASES_MPI`。它现在有两个用途：给各组提供插值坐标，以及**提供 CRUST1.0 +
s362ani 的底图**——四个 `_c1` 组的弹性参数底图就是从它的 `solver_data.bin` 反算出来的。

选 s362ani 而不是 GLL，起初是因为本机只有这条路有干净的运行记录（§6.6）。改用 CRUST1.0
共用地壳之后，这个选择变成了硬要求：底图必须来自一次带 CRUST1.0 地壳的 mesher，而
`MODEL = GLL` 的那次 mesher 会被 GLL 值覆盖掉地壳（`get_model.F90` 里 GLL 的覆盖排在
`meshfem3D_models_get3Dcrust_val` 之后），拿不到干净的 CRUST1.0。
`probe` 的跨 MODEL 比对已经证明两种 MODEL 给出逐位相同的坐标（§6.16），所以用 s362ani
不影响坐标基准的有效性。

**BASE 不复用 mesh0 的网格。** 早期版本让 BASE 直接软链 mesh0/DATABASES_MPI 以省一次
mesher，那是在 mesh0 也用 GLL 建网格的前提下才成立的。现在 mesh0 是 s362ani 建的，
BASE 若复用它，BASE 的网格就与其余组不同源，各组不可比。改为 BASE 也跑自己的
`MODEL = GLL` mesher 并接受同样的坐标核对，各组待遇才真正一致，代价约一分钟。

若 `mesh0/DATABASES_MPI/.mesh_ok` 已存在，`submit` 会跳过 MESH0 直接提交各组。
`FORCE_MESH=1` 可强制重跑。

### 6.3 各组作业的四个阶段

五组（含 BASE）走完全相同的四个阶段，每一步失败都会中止：

| 阶段 | 执行者 | 内容 |
| --- | --- | --- |
| 1 建模型 | `$PYTHON_BIN` | `build_hybrid_gll.py --mesh-dir mesh0/DATABASES_MPI` → `DATA/GLL/` |
| 2 mesher | Intel + IMPI | `MODEL = GLL`，读 `DATA/GLL`，写本组自己的 `DATABASES_MPI` |
| 3 坐标核对 | `$PYTHON_BIN` | `check_mesh_diff.py mesh0/DATABASES_MPI DATABASES_MPI` |
| 4 solver | Intel + IMPI | **循环 10 个事件**，每个出一套 SAC |

BASE 的阶段 1 为空——它的 `DATA/GLL` 在 `prepare` 时就软链到了 `model_updated`，
零改动对照不需要构建任何东西。`S362ANI_c1` 的阶段 1 不嵌任何模型，但仍要跑：它的底图
是从 `solver_data.bin` 反算出来的，磁盘上并不存在，必须落盘一份才能被 `MODEL = GLL` 读。

Python 用解释器绝对路径直接调用，不 `module load` 也不 `activate`（§6.8），所以阶段间
无需切换环境，Intel 模块从阶段 2 起一直保持加载。

阶段 1 和 2 都有幂等跳过：`DATA/GLL` 已完整则不重建（`FORCE_BUILD_GLL=1` 强制），
`DATABASES_MPI` 已带 `.mesh_ok` 标记则不重跑 mesher（`FORCE_MESH=1` 强制）。注意跳过
判据是标记而不是文件数，原因见 §6.15。

`--jobs` 固定为 16 而非默认的满核数。`build_hybrid_gll.py` 的每个 worker 各自加载一份
模型网格，64 个 worker 会驻留 64 份，FWEA23 解压后单份即数 GB，必然 OOM。

阶段 3 通过后会在 `DATABASES_MPI/` 下写 `.coord_ok`；重投时见到该标记就跳过核对，
不必为了续跑再花一两分钟重比 144 个 proc。

#### 阶段 4 的事件循环

```
for ev in $EVENTS_TODO:
    sac/<TAG>/$ev/.done 存在 → 跳过（FORCE_SOLVER=1 可无视）
    剩余墙钟 < 上一个事件耗时 × 1.2 + 10 min → 就地重投自己，退出
    cp events_test/CMTSOLUTION_ECEF/CMTSOLUTION_$ev  DATA/CMTSOLUTION
    rm -f OUTPUT_FILES/*.sac        ← 关键，见下
    mpirun -np 144 ./bin/xspecfem3D
    SAC → sac/<TAG>/$ev/ ，写 .done（含时间、SAC 数、耗时）
```

两条机制保证一次跑不完也不会白跑：

- **逐事件断点**。`.done` 标记落在 `sac/<TAG>/<EVENT>/` 下，与结果同处一地——标记在
  说明结果在，不会出现"标记说完成了但目录是空的"。重投时自动从第一个未完成的事件继续，
  `FORCE_SOLVER=1` 强制全部重算。
- **时限保护**。每个事件开跑前用 `squeue -h -j $SLURM_JOB_ID -o %L` 查剩余墙钟，不足以
  跑完下一个就主动 `sbatch` 重投自己再退出。这样作业永远在事件边界上切分，不会在
  solver 跑到一半时被 Slurm 砍掉留下半套 SAC。第一个事件没有历史耗时可参考，按 2 小时
  估。

每个事件开跑前先 `rm -f OUTPUT_FILES/*.sac`。少了这一句，若本次 solver 中途失败，
上一个事件残留的 SAC 会被当成本次结果收走——文件名不带事件号，收走之后无从分辨。

单事件 solver 实测约 70 分钟（4 × 36 核、30 分钟记录长度），10 个事件串起来约 12 小时，
作业时限 `SLURM_TIME_RUN = 24:00:00` 原则上装得下，但节点故障、抢占、排队都可能打断，
所以重投按常态设计——`status` 里同一组出现多个 JobID 不是异常。

### 6.4 输出为完整模型，不再用 `--skip-unchanged`

服务器端建模后不存在上传量问题，`DATA/GLL` 直接写全量 1008 个文件（144 × 7）、约 3.5 GB / 套。三套合计约 10.5 GB。

`--skip-unchanged` 与 `manifest.txt` 组装逻辑在脚本中已移除。`build_hybrid_gll.py` 仍保留该选项，但本流程不使用。

### 6.5 坐标核对：现在拦的是什么

`check_mesh_diff.py` 逐 proc 比对两套 `DATABASES_MPI` 的坐标段，判据分两级：

- **`ibool` 必须逐位相同。** 它是全局编号，属拓扑，任何差异都意味着网格结构变了。
- **`x / y / z` 允许相对误差 ≤ `COORD_TOL`（默认 1e-6）。** 这个容差用于容纳异构节点上向量化求和顺序不同带来的末位差异；1e-6 折算约 6 米，相对 25 km 量级的网格间距可忽略。要求逐位相同则设 `--tol 0`。

与旧的指纹核对的区别在于**核对对象换了**：旧方案比的是"本地预备的坐标"与"服务器网格"，这两者没有理由相同；新方案比的是"mesh0"与"本组 mesher 的复现"，这两者是同一台机器、同一份配置、同一批二进制的两次运行，本就应当一致，不一致即为需要追查的真问题。

核对不通过时作业直接退出，不启动 solver——避免在错误的模型上浪费 6–12 小时。

单独诊断时可以带 `--verbose` 看逐 proc 明细：

```bash
/work/home/acf11bgjob/.conda/envs/eastasia_fwi/bin/python check_mesh_diff.py \
    mesh0/DATABASES_MPI runs/EARA2024_c1/DATABASES_MPI \
    --nproc 144 --tol 1e-6 --verbose
```

### 6.6 MPI 界面错误：根因是故障节点（已定案）

mesher 的 `face flag` / `ineighbour points differ` / `MAX_NEIGHBOURS` 系列错误，
在本项目里前后被归因过三次，前两次都错了：

| 假设 | 依据 | 结论 |
| --- | --- | --- |
| 跨机架 MPI 通信不一致 | 早期成功案例四节点同机架 | ❌ 同机架照样失败 |
| MPI 运行时变量（`I_MPI_FABRICS=shm:dapl` + `I_MPI_FALLBACK=0`） | 失败脚本有这两项，`run_forward.bash` 没有 | ❌ 改成一致后仍失败（作业 7589247） |
| **故障节点** | probe 对照实验，见下 | ✅ |

probe 实验（§6.16）在两个机架上各跑同样的四轮 mesher，同一份 Par_file、同一套二进制、
同一个 4 × 36 几何：

| 机架 | s362ani | GLL | 坐标复现 |
| --- | --- | --- | --- |
| c08r1 | 两轮全崩 | 两轮全崩 | 无输出 |
| c10r3 | 两轮干净 | 两轮干净 | 144/144 逐位相同 |

决定性的细节是 c08r1 上**四轮报出四种互不相同的错误**：`Error Jacobian rank: 125`、
`ineighbour points differ`（193 点）、`ineighbour points differ`（7838 点）、
`Error interfaces rank: 67`。其中 `Error Jacobian` 是单元雅可比退化，属网格几何层面，
与 MPI 界面完全不是一回事。相同输入产生形态各异的损坏，只能是硬件或内存层面的随机出错；
算法缺陷、模型选择、通信参数都不会有这种表现。

落实为一条配置：`run_all_test.sh` 的 `BAD_RACKS` 变量列出黑名单机架，`pick_rack_nodes()`
跳过其中的节点。当前值 `c08r1`。今后若在别的机架遇到同类随机崩溃，追加进去即可。

同时确认了两件此前不清楚的事：`MODEL = GLL` 的 mesher 本身没有问题（c10r3 上两轮全过），
网格几何与 `MODEL` 取值无关（跨 MODEL 坐标逐位相同）。

MPI 设置保持现状，它们无害也无关：

| 变量 | 取值 | 说明 |
| --- | --- | --- |
| `I_MPI_FABRICS` / `I_MPI_FALLBACK` | 不设置 | 让运行时自选，不把节点间通信钉死在 legacy DAPL 上 |
| `I_MPI_ADJUST_ALLREDUCE` | 5 | 沿用 `run_forward.bash` |
| `I_MPI_ADJUST_ALLTOALL` | 2 | 同上 |
| 节点几何 | 4 × 36 | 有成功记录，取代此前无依据的 3 × 48 |

顺带纠正一处旧结论：`run_forward.bash` 判定 mesher 成功用的是 `solver_data.bin 144/144`，
只数文件、不看退出码也不查告警——与本流程早期犯的是同一个错（§6.15）。那次日志里确实
没有 `flag` 与 `ineighbour`，所以那次成功是真的，但判据本身靠不住，不应照搬。

### 6.7 同机架分配与自动重投

集群 `TopologyPlugin=topology/default`，不支持 `--switches`，只能在提交时显式挑节点。`pick_rack_nodes()` 从 `sinfo` 的空闲列表里按机架分组（节点名 `c08r1n20` 去掉尾部 `n<编号>` 即机架标识），凑出 `SLURM_NODES` 个同机架节点。

挑节点时必须用 `scontrol show node` 二次确认状态。`sinfo -t idle` 只匹配基础状态，被 drain、卡在 completing、或处于预约中的节点基础状态仍是 `IDLE`，照样会被列出来，但 `--nodelist` 指过去只会让作业永远挂在 `ReqNodeNotAvail`——这是实测踩过的坑。

`BAD_RACKS` 里的机架被直接跳过，当前是 `c08r1`（§6.6）。各组提交时轮流分配不同机架；空闲机架不够分时复用已用机架（作业会排队等待，但同机架约束仍然成立，便于把故障范围收敛到一组硬件上）。

mesher 若失败且日志里出现 `face flag` / `MAX_NEIGHBOURS` / `valence` / `ineighbour` / `MPI_Abort`，作业会自动换一批同机架节点重投，最多 `MAX_MESH_ATTEMPT`（默认 5）次。既然根因是故障节点，同节点原地重试必然无效，必须换节点。非 MPI 界面类的错误不自动重投。

一个机架若连续多次触发重投，把它加进 `BAD_RACKS`，别让后续作业继续踩。

手动重投：`./run_all_test.sh resubmit <TAG>`。

### 6.8 配置项

| 变量 | 含义 | 本项目取值 |
| --- | --- | --- |
| `CODE_DIR` | SPECFEM 源码根目录 | `specfem3d_globe_code_new/` |
| `DATA_DIR` | **直接**含 `Par_file`、`STATIONS`、`constants.h.in` | `specfem/DATA/` |
| `EVENTS_ROOT` | 震源目录，含 `events.list` 与 `CMTSOLUTION_ECEF/` | `specfem/events_test/` |
| `CONSTANTS_H_IN` | 编译用 `constants.h.in` 全路径 | `$DATA_DIR/constants.h.in` |
| `BASE_GLL_DIR` | 底图 `model_updated/` | — |
| `MODELS_DIR` | 原始模型文件目录 | `specfem/models/` |
| `PYTHON_BIN` | Python 解释器绝对路径 | `~/.conda/envs/eastasia_fwi/bin/python` |
| `BUILD_JOBS` | 建模型并行度 | 16 |
| `COORD_TOL` | 坐标核对相对容差 | `1e-6` |
| `ROUNDTRIP_EVENT` | `S362ANI_ref` 用哪个事件 | `C201904232015A` |
| `SLURM_*` | 队列、账号、节点数、时限 | 4 × 36，mesher 2 h / solver 24 h |
| `SAME_RACK` / `MAX_MESH_ATTEMPT` | 同机架挑节点 / 自动重投上限 | 1 / 5 |
| `REGION_LATLON` / `HORIZ_TAPER` | 水平替换区与 taper | `10 55 80 150` / `5.0` |
| `DEPTH_MIN_MODE` / `MOHO_OFFSET` | 替换上界模式与偏移 | `moho` / `0.0` |
| `DEPTH_RANGE` / `DEPTH_TAPER` | 深度外边界与 taper | `0 1000` / `15 50` |

各组要嵌入的地幔模型在 `model_opt()` 里映射：

```bash
S362ANI_c1    （空）        不嵌模型，只把 CRUST1.0+s362ani 底图落盘
FWEA23_c1     --model-nc  models/FWEA23.r0.0-n4.nc
EARA2024_c1   --model-nc  models/EARA2024.r0.0-n4.nc
SinoScope_c1  --model-h5  models/FWI_SinoScope_1.0_JMa_ET_AL_2022.h5
```

**`PYTHON_BIN` 为什么写绝对路径。** 早期版本用 `module load anaconda3/2023.09 && source activate eastasia_fwi`，实测这在本集群上会**安静地失败**：命令全部返回 0，但解释器仍是 base 的 `/public/software/apps/anaconda3/2023.09/bin/python`，而 base 没有 netCDF4。真正的环境在 `/work/home/acf11bgjob/.conda/envs/eastasia_fwi`。这个错误不会在提交时暴露，要等 MESH0 跑完二十多分钟、三个混合组同时进入阶段 1 才集体崩掉。

直接指解释器绝对路径消除了整个中间环节，附带好处是阶段间不必再 `module purge` 来回切换，Intel 模块可以从阶段 2 起一直保持加载。`prepare` 的 `check_python()` 会在两种场景下验证依赖可导入——裸环境（阶段 1）与 Intel 模块已加载（阶段 3）——后者是为了排除 conda Python 与 Intel 运行时的 `LD_LIBRARY_PATH` 冲突。

`CONSTANTS_H_IN` 单列一项而不是写死成 `$DATA_DIR/setup/constants.h.in`，是因为它未必与数据同目录——本项目里它现在放在 `specfem/DATA/` 下（与 `sem_config_tibet_v4/setup/` 那份逐位相同），但换个部署可能在别处。显式指定却找不到会直接中止。

替换区参数现在由 `run_all_test.sh` 顶部的 `REGION_LATLON` / `HORIZ_TAPER` / `DEPTH_RANGE` / `DEPTH_TAPER` 统一传给各组，保证各组用的是同一套几何——这是可比的前提之一。

### 6.9 震源配置：一个必须避开的坑

震源不再取自 `DATA_DIR`，而是由 `EVENTS_ROOT` 下的 `events.list` 驱动，作业在每个事件
开跑前把 `CMTSOLUTION_ECEF/CMTSOLUTION_<EVENT>` 拷成 `DATA/CMTSOLUTION`。这样 `DATA_DIR`
里那份 CMTSOLUTION 就只是 mesher 阶段的占位（mesher 也要求文件存在），不参与任何结果。

**格式必须是 ECEF。** `constants.h.in` 里 `USE_ECEF_CMTSOLUTION = .true.`，代码按
`t0(s)` / `x(m)` / `Mxx(N*m)` 解析。标准格式的 CMTSOLUTION（`time shift` / `latitude` /
`Mrr`）字段对不上号，会被**静默**误解析成一个完全不同的震源机制——不报错，只是算出来的
东西全错。`select_test_events.py` 调 `change_cmt_to_ECEF.py` 生成 `CMTSOLUTION_ECEF/`
就是为此。

服务器上另有一套 `sem_config_tibet_v4/DATA/`，里面是 C201102041353A、标准格式、945 个
台站，与本流程完全不兼容，不要误用。两套 Par_file 的网格参数倒是一致的（NCHUNKS=1、
80°、中心 32°N/106°E、GAMMA=60°、NEX 288、NPROC 12×12），所以只有震源和台站表需要留意。

原作者的参考波形只覆盖 C201403131706A（`C201403131706A/output_syn/sac`，4302 个文件），
该事件不在这 10 个之内。零点校准已经在上一轮单事件测试中完成，本轮不再依赖它。

`constants.h.in` 现在也放在 `specfem/DATA/` 下，与 `sem_config_tibet_v4/setup/` 那份逐位相同。其中与本流程相关的关键项：

```
USE_ECEF_CMTSOLUTION      = .true.                    震源按 ECEF 格式解析
REGIONAL_MOHO_MESH        = .true.
RMOHO_STRETCH_ADJUSTMENT  = -20000.d0                 Moho 网格边界下压到 60 km
R80_STRETCH_ADJUSTMENT    = -40000.d0                 r80 下压到 120 km
GLL_REFERENCE_1D_MODEL    = REFERENCE_MODEL_1DREF
```

两个 `*_STRETCH_ADJUSTMENT` 直接决定网格几何。它们一旦与生成 `model_updated` 时不同，GLL 点位置就会偏移。本流程中 mesh0 与各组用的是同一份 `constants.h.in`，所以组间不会因此产生差异；但若相对生成 `model_updated` 时有变动，各组会共同偏移，且 §6.5 的核对发现不了——它比的是组间一致性，不是与底图来源的一致性。

`prepare` 的 `check_sources` 逐个核对 `events.list` 里的全部事件：

- 文件是否存在（缺一个就中止，不要跑到第 7 个事件才发现）
- CMTSOLUTION 内的 `event name` 与文件名是否一致（不一致给告警）
- 格式与 `USE_ECEF_CMTSOLUTION` 是否匹配（不匹配直接中止）
- `ROUNDTRIP_EVENT` 是否在清单内（不在则 `S362ANI_ref` 无组可比，中止）

已反向验证：拿标准格式的 CMTSOLUTION 顶替时，格式检查会检出并中止。

### 6.10 时间步 DT

SPECFEM3D **Globe** 的 `DT` 不在 Par_file 里——它由 mesher 按 `NEX` / `NCHUNKS` 自动确定（`get_timestep_and_layers`），CFL 由构造保证。Cartesian 版那套「比较 Par_file 的 DT 与 Maximum suggested time step」的检查在这里不适用，`output_mesher.txt` 里也没有 suggested 那一行，只有：

```
the time step of the solver will be DT =   0.1016261
```

各 run 会把这个值存进 `OUTPUT_FILES/.dt`，`status` 以一列显示并核对各组是否一致。各组 DT 必须相同，否则波形不在同一时间采样上，misfit 不能直接比较。

### 6.11 目录结构

```
hybrid_test_multi/
├── bin/                        所有作业共用的 xmeshfem3D / xspecfem3D
├── mesh0/
│   ├── DATA/  (Par_file 里 MODEL 被改为 s362ani)
│   ├── DATABASES_MPI/          ★ 坐标基准 + 底图来源，通过校验后带 .mesh_ok
│   └── OUTPUT_FILES/
├── probe/                      mesher 对照实验（§6.16）
│   ├── work/                   四轮共用的工作目录
│   ├── coords_r1..r4/          各轮的坐标段副本（每文件截断到 12 MB）
│   ├── round<N>_<MODEL>.log
│   └── report.txt
├── runs/<TAG>/                 五组结构完全相同
│   ├── DATA/{Par_file,STATIONS,CMTSOLUTION}   CMTSOLUTION 逐事件被覆盖
│   ├── DATA/GLL/               ← 各组之间唯一的变量
│   ├── DATA/{crust1.0,crust2.0,s362ani,QRFSI12,topo_bathy} -> 共用软链
│   ├── DATABASES_MPI/          本组 mesher 输出 + .mesh_ok + .coord_ok
│   └── OUTPUT_FILES/           .dt + 逐事件的 .done_<EVENT>
├── runs/S362ANI_ref/
│   └── DATABASES_MPI -> ../../mesh0/DATABASES_MPI    软链复用，不自己网格化
├── jobs/run_{PROBE,MESH0,<TAG>}.slurm
├── logs/                       build.log、各作业 .out/.err、jobids.txt
└── sac/<TAG>/<EVENT>/          ★ 每个事件一个目录，各约 250 个 SAC
```

根目录名不再带事件号（`hybrid_test_multi`），因为一套目录承载全部 10 个事件。

BASE 的 `DATA/GLL` 是指向 `model_updated` 的软链，其余一切与四个混合组相同。

SAC 按 `<TAG>/<EVENT>/` 两级分目录，是后处理脚本能按事件循环的前提。总量约
51 套 × 250 个文件。

### 6.12 mesher 不能跳过

mesher 读 `DATA/GLL` 里的速度值，转成弹性模量写进本 run 自己的
`DATABASES_MPI/solver_data.bin`，solver 只认这个文件。所以各组都是完整的
「mesher + solver」两阶段，BASE 也不例外。

mesh0 那次 mesher 不算在内——它用的是 s362ani，产物只当坐标用，不会被任何 solver 读取。

跳过逻辑只在 `.mesh_ok` 标记存在时生效，用途是失败重投时不必重跑已通过校验的阶段
（`FORCE_MESH=1` 可强制重跑）。它不会让不同模型共用一份数据库——各组的
`DATABASES_MPI` 是物理隔离的目录。

### 6.13 耗时与风险

| 阶段 | 估计 | 次数 |
| --- | --- | --- |
| `prepare`（含编译） | 10–20 min | 1 |
| MESH0 | 2–20 min | 1 |
| 建模型（每组，16 并行） | 10–30 min | 4 |
| mesher（每组） | 2–20 min | 5 |
| 坐标核对（每组） | 1–2 min | 5 |
| solver（每事件） | ~70 min | 51 |

51 次 solver = 5 组 × 10 事件 + `S362ANI_ref` 1 次。单次 70 分钟 × 144 核 ≈ 168 核时，
合计约 **8600 核时**。五组并行时墙钟约 12 小时（受最慢一组决定），加上前面的准备阶段，
理想情况一天内出全部结果；排队与重投会拉长，按两三天预期比较现实。

磁盘占用：四套 `DATA/GLL` 约 14 GB，五套 `DATABASES_MPI` 加 mesh0 约 25 GB，
51 套 SAC 约 2.8 GB（单套实测 54 MB），总量 40–50 GB。

meshfem 的 MPI face flag 类错误根因已定位为故障节点（§6.6），处理手段是 `BAD_RACKS` 黑名单加自动换机架重投。各组独立提交正是为此；失败的那组 `status` 会显示停在对应阶段，`./run_all_test.sh resubmit <TAG>` 单独重投即可。若某个机架反复触发重投，把它加进 `BAD_RACKS`。

坐标核对是新增的失败模式：如果某组的 mesher 复现不出 mesh0 的坐标，作业会在 solver 之前停下。既然已知健康节点上必然 144/144 逐位相同，出现差异就说明这次跑在了坏硬件上——换节点重跑，不要放宽 `COORD_TOL`。`--verbose` 可以看差异的分布，但不改变处置方式。

---

### 6.14 手动分步流程（排查时参考）

以下是封装之前的手动步骤，仅在需要单独调试某一环时使用。全部在服务器上执行。

#### Step 1 — 基准网格

```bash
cd hybrid_test_multi/mesh0
module purge && module load compiler/intel/2018.5.274 mpi/intelmpi/2018.4.274
ulimit -s unlimited && export I_MPI_ADJUST_ALLREDUCE=5 I_MPI_ADJUST_ALLTOALL=2
mpirun -np 144 ./bin/xmeshfem3D
ls DATABASES_MPI/proc*_reg1_solver_data.bin | wc -l      # 应为 144
```

产出的 `DATABASES_MPI` 即后续所有插值的坐标基准，不要让任何 run 往里写。

#### Step 2 — 覆盖诊断

改动 `--region` 或深度参数后需重新确认替换区占比（结果见 §3.3、§3.4）：

```bash
/work/home/acf11bgjob/.conda/envs/eastasia_fwi/bin/python build_hybrid_gll.py \
    --mesh-dir hybrid_test_multi/mesh0/DATABASES_MPI \
    --base-gll-dir model_updated \
    --model-nc models/EARA2024.r0.0-n4.nc \
    --output-dir /tmp/dryrun --dry-run
```

各组的 `--region`、深度参数与 `--base-from-mesh` 必须**取同一组值**，否则替换区形状不同，对比不成立。`run_all_test.sh` 顶部的 `REGION_LATLON` 等变量统一供给各组，就是为了防止这一点被手滑破坏。

#### Step 3 — 构建一套 GLL

```bash
/work/home/acf11bgjob/.conda/envs/eastasia_fwi/bin/python build_hybrid_gll.py \
    --mesh-dir hybrid_test_multi/mesh0/DATABASES_MPI \
    --base-gll-dir model_updated \
    --output-dir hybrid_test_multi/runs/EARA2024_c1/DATA/GLL \
    --base-from-mesh \
    --model-nc models/EARA2024.r0.0-n4.nc \
    --region 10 55 80 150 --horiz-taper 5.0 \
    --depth-range 0 1000 --depth-taper 15 50 \
    --depth-min-mode moho --moho-offset 0 \
    --crust1-bnds specfem3d_globe_code_new/DATA/crust1.0/crust1.bnds \
    --nproc 144 --jobs 16
```

应产出 144 × 7 个文件（6 参数 + qmu），单文件 3,618,008 字节。收尾日志里的「深度基准自检」一行要落在 ±3 km 内，否则脚本会拒绝写出。

#### Step 4 — mesher + 坐标核对

```bash
cd hybrid_test_multi/runs/EARA2024_c1
module purge && module load compiler/intel/2018.5.274 mpi/intelmpi/2018.4.274
mpirun -np 144 ./bin/xmeshfem3D

/work/home/acf11bgjob/.conda/envs/eastasia_fwi/bin/python ../../../check_mesh_diff.py \
    ../../mesh0/DATABASES_MPI DATABASES_MPI --nproc 144 --tol 1e-6
```

核对不通过就不要往下走，此时 solver 算出来的东西没有意义。

#### Step 5 — solver（单个事件）

```bash
cp ../../../events_test/CMTSOLUTION_ECEF/CMTSOLUTION_C201904232015A DATA/CMTSOLUTION
rm -f OUTPUT_FILES/*.sac
module purge && module load compiler/intel/2018.5.274 mpi/intelmpi/2018.4.274
mpirun -np 144 ./bin/xspecfem3D
mkdir -p ../../sac/EARA2024_c1/C201904232015A
mv OUTPUT_FILES/*.sac ../../sac/EARA2024_c1/C201904232015A/
```

三点容易漏：换事件前要清 `OUTPUT_FILES/*.sac`（文件名不带事件号，混在一起就分不清了）；
CMTSOLUTION 必须取 `CMTSOLUTION_ECEF/` 那一份；mesher 会重写本目录的 `solver_data.bin`，
务必保证 `mesh0/DATABASES_MPI` 不被写入。

---

### 6.15 mesher 的通过判据（🔴 一个已修复的严重缺陷）

早期版本按 `solver_data.bin` 数量判断 mesher 是否成功，这是错的，而且代价惨重。
实测经过（2026-08-03）：

1. 作业 7589247 的 mesher 在 MPI 界面校验阶段 `MPI_Abort(30)` 退出，但**在中止之前
   144 个 `solver_data.bin` 已经全部写完**。
2. 自动重投的 7589253 启动后，跳过逻辑看到"144 个文件齐全"，判定数据库已完整，
   跳过 mesher，**3 秒后以 COMPLETED 退出并写下 `.done` 标记**。
3. 于是 `mesh0/DATABASES_MPI` 里躺着一份在界面校验中崩掉的网格，却被标记为可用。
   各组若不是被 Slurm 连带取消，会在这份网格上建模型、跑 solver，全程不报错。

修正后的判据要求四项全部满足才写 `.mesh_ok`：

```
退出码 = 0
solver_data.bin = NPROC
MPI 界面告警 = 0      (flag: missed / WARNING MPI interface)
MPI 界面错误 = 0      (Error ineighbour / MAX_NEIGHBOURS / valence / MPI_Abort)
```

跳过逻辑只认 `.mesh_ok`。任一项不过就**清空 `DATABASES_MPI` 再换节点重投**，杜绝坏网格
被后续复用。

界面告警在本项目里等同于失败。源码把若干一致性检查降级成了 warning
（`Models_GLL_workflow_server.md` §19.4），放行的网格可能真的有拓扑问题，而这不会在
后续任何环节报错。项目总纲 §7 的"不要把 MPI 一致性检查降级为 warning"说的就是这个。

顺带修掉的另一个缺陷：MESH0 失败时自动重投产生新 JobID，而各组挂的是
`--dependency=afterok:<旧 JobID>`，旧作业已失败，Slurm 判定依赖永不满足，把各组全部
`CANCELLED`（实测）。现改为**不用 dependency**，由 MESH0 在基准网格通过校验后自己调
`run_all_test.sh submit-runs` 提交各组，重投多少次都不受影响。

### 6.16 `probe`：mesher 对照实验（已完成，结论见下）

设计这个实验时有两个问题卡住流程，现有证据都不足以分辨：

| | 问题 | 为什么当时答不出 |
| --- | --- | --- |
| Q1 | `MODEL = GLL` 的 mesher 是否比 `s362ani` 更容易触发 MPI 界面错误？ | 唯一成功案例是 s362ani + c12r4，唯一失败是 GLL + c08r1，两个变量同时不同 |
| Q2 | 同节点、同二进制、同 MODEL 跑两次，坐标是否逐位相同？ | 决定"以 mesh0 坐标建模型"是否成立 |

`probe` 把四轮 mesher 放进**同一个作业**：s362ani、GLL、s362ani、GLL。节点分配与二进制
完全相同，MODEL 成为唯一变量；同 MODEL 的两轮之间比坐标即可分离出可复现性。每轮约
一分钟，总计 5–10 分钟。每轮只保留 `solver_data.bin` 的前 12 MB（坐标段约 9 MB，
`check_mesh_diff.py` 顺序读到 `ibool` 就停，截断不影响比对），四轮合计约 7 GB。

```bash
./run_all_test.sh probe                          # 自动挑机架
./run_all_test.sh probe c10r3n02,c10r3n03,c10r3n04,c10r3n05   # 指定机架
cat hybrid_test_multi/probe/report.txt
```

报告里的每一轮若判为"有问题"，会额外打印首条错误以及出错 rank 所在的节点（按 mpirun
块分配从 rank 号反推），用于区分"某台机器坏了"和"普遍性问题"。

#### 实测结果（2026-08-03）

| 机架 | 轮 1 s362ani | 轮 2 GLL | 轮 3 s362ani | 轮 4 GLL |
| --- | --- | --- | --- | --- |
| c08r1n[20,29,31,37] | `Error Jacobian rank 125` | `ineighbour` 193 点 | `ineighbour` 7838 点 | `Error interfaces rank 67` |
| c10r3n[02-05] | 干净 | 干净 | 干净 | 干净 |

c10r3 的三组坐标比对全部 **144/144 逐位相同**（同 MODEL 两组、跨 MODEL 一组），
`ibool` 零差异，坐标相对差 0.0。

由此得到四条结论，都已写回相应章节：

1. mesher 输出是**逐位确定性**的，§3.2 里"不可复现"的旧判断作废。
2. 网格几何与 `MODEL` 无关，`MESH0_MODEL` 可自由选择。
3. `MODEL = GLL` 的 mesher 没有问题。
4. 所有失败的根因是**故障节点**，处理方式是 `BAD_RACKS` 黑名单（§6.6）。

在 c08r1 上产出过的一切 `DATABASES_MPI` 与由其插值得到的模型全部作废，不得复用。

### 6.17 作业脚本的本地自检

`run_all_test.sh` 用嵌套 heredoc 生成 Slurm 脚本，这类代码有一类特别难发现的错误：
**生成阶段与运行阶段的变量混淆**。`$SAC_DIR` 应当在生成时就展开成路径，`$SLURM_JOB_ID`
则必须留到作业运行时才求值，两者在源码里只差一个反斜杠，写反了不会报错——生成出来的
脚本语法完全合法，只是路径变成空串或者 JobID 永远取不到。等发现时已经烧掉几百核时。

`tests/test_run_scripts.sh` 在本地把六份作业脚本生成出来逐项核查，不需要集群：

| 检查 | 目的 |
| --- | --- |
| `bash -n` | 语法 |
| 事件循环里出现 10 个事件号 | 循环没被截断 |
| `$SAC_DIR` / `$NPROC` 已展开 | 生成时变量确实求了值 |
| `$SLURM_JOB_ID` / `$(date +%s)` 未展开 | 运行时变量确实留到了运行时 |
| `EVENTS_TODO` / `remaining_sec` / `.done` 齐备 | 断点续跑机制没漏 |
| `BASE` 不含 `build_hybrid_gll.py`；`S362ANI_ref` 不含 `xmeshfem3D` | 分组逻辑 |
| 变量后紧跟非 ASCII 字符时是否加了花括号 | 见下 |
| 是否出现 U+FFFD | 多字节字符被截断的信号 |

最后两项针对的是一个实际踩过的坑：`$MESH0_MODEL（` 这样的写法，bash 在解析变量名边界时
会把中文全角括号一并吞进去，`set -u` 下直接报 unbound variable，生成的脚本从那一行起
全部丢失。文档和脚本都是中文注释，这种组合出现得很频繁，只能靠 `${VAR}` 显式定界。
已修 11 处。

另外两项 mock 测试用一棵人造的结果树验证 `status` 与 `collect`：故意让某一组只完成 7 个
事件、某个事件的 SAC 数少几个，确认两个命令能分别报出「部分完成」和「台站数不一致」，
而不是笼统地说一句失败。

```bash
bash tests/test_run_scripts.sh
```

---

## 7. 验证清单

| # | 阶段 | 检查 | 通过判据 | 状态 |
| --- | --- | --- | --- | --- |
| C1 | `prepare` | 网格与底图匹配 | NSPEC = 7236；底图文件 3,618,008 字节 = 8 + 125×7236×4；proc 数 = `NCHUNKS×NPROC_XI×NPROC_ETA` | ✅ 脚本自动校验 |
| C2 | `--dry-run` | 覆盖诊断 | `w>0` 占比合理；各组用同一 `--region` | ✅ 已实测（§3.4） |
| C3 | phase1 | 脚本自检 | §5.3 的 S1–S5 全部通过，尤其 S4 | ✅ 三模型本地全通过 |
| C4 | phase1 | **BASE 等价性** | `BASE/DATA/GLL` 与 `model_updated/` 逐位相同 | ✅ 已验证（见下） |
| C5 | mesher 后 | mesher 读到 GLL | 各组 `DATABASES_MPI/proc*_reg1_solver_data.bin` 与 BASE 逐字节比对，有差异的 proc 数 > 0 | ✅ 三组均 81/144（见下） |
| C6 | mesher 后 | CFL | `output_mesher.txt` 的建议时间步 > Par_file 的 DT | ⏳ phase1 自动打印 |
| C7 | solver 中 | 稳定性 | Max U 不指数增长 | ⏳ 待服务器 |
| C8 | solver 后 | 波形完整 | 同一事件下各组的台站集合完全相同 | ⏳ `collect` 可查 |
| C9 | 后处理 | **波形基线** | 插值误差在波形空间的量级，作为判据阈值 | ⏳ 定义需重做，见 §9.4 |
| C10 | MESH0 后 | 基准网格干净 | 日志中 MPI 界面告警条数为 0 | ✅ 作业 7592372，退出码 0、144/144、零告警零错误 |
| C11 | mesher 后 | **坐标复现** | 本组 `DATABASES_MPI` 与 mesh0：`ibool` 逐位相同、`x/y/z` 相对差 ≤ 1e-6 | ✅ 旧四组均 144/144 逐位相同，相对差 0.0 |
| C12 | 全程 | 替换区一致 | 各组用同一套 `REGION_LATLON` / taper 参数 | ✅ 脚本统一供给 |
| C13 | solver 前 | DT 一致 | 各组 mesher 计算的 DT 相同 | ✅ 旧四组均 DT = 0.1016261 |
| C14 | phase1 | **深度基准** | 自由表面与 CMB 两端偏差 ≤ 3 km；底图 Moho 跳变落在 CRUST1.0 Moho ±5 km 内 | ✅ 本地 144 分片实测 −0.1 / +1.0 / +0.2 km（§4.5） |
| C15 | phase1 | **底图提取正确** | 从 `solver_data.bin` 反算的六个参数全部落在物理区间 | ✅ 本地实测通过（`tests/test_depth_datum.py` 第 2 项） |
| C16 | `prepare` | **震源齐备且格式正确** | 10 个 CMTSOLUTION 全部存在、event name 与文件名一致、为 ECEF 格式 | ⏳ `check_sources` 自动校验（§6.9） |
| C17 | `prepare` | **作业脚本正确** | `tests/test_run_scripts.sh` 全部通过 | ✅ 本地全通过（§6.17） |
| C18 | 后处理 | **GLL 往返代价** | `S362ANI_ref` 与 `S362ANI_c1` 在同一事件上的波形差异，作为组间差异的分辨率下限 | ⏳ 待服务器 |
| C19 | 后处理 | **排名跨事件稳定** | 同一名次关系在 10 个事件中的多数上成立 | ⏳ 待服务器 |

C4 是整条链路的零点校准，本地已在 macOS 上用真实数据验证：不指定模型时，`build_hybrid_gll.py` 产出的 28 个文件（4 proc × 7 参数）与 `model_updated/` 对应文件 `cmp` 全部逐位相同。脚本里 BASE 组的 `DATA/GLL` 直接软链到 `model_updated/`，连拷贝都省了，这一条恒真。

C11 取代了早期版本的"网格指纹核对"。指纹方案比的是本地预备坐标与服务器网格，这两者没有理由相同（§3.2），核对必然失败；现在比的是同一台机器上两次 mesher 的输出，不一致才是真问题。

C5 的判据从"翻 `output_mesher.txt` 找日志行"改成了逐字节比对，后者是直接证据而非间接推断。实测三组与 BASE 的差异 proc 数**都是 81 / 144**，数字相同是必然的——替换区取三模型 NC 的交集（§9.3），区域形状一致，受影响的 proc 集合自然一致。`proc000000` 无差异，说明它整个落在替换区外。反过来说，若某组报 0/144，就意味着 GLL 根本没被读进去，各组算的是同一个模型。

C9 是最终判据，与 C4 的区别要分清：C4 保证模型文件层面没有引入噪声，C9 才回答「波形空间里多大的差异才算真实差异」。§3.5 的模型空间 rms 只是它的事前估计。改用共用地壳后 C9 的定义需要重做，因为 `BASE` 与 `FWEA23_c1` 之间同时变了地壳和插值两件事，见 §9.4。

C18 与 C9 是两个不同的下限，别混。C18 只量「GLL 写盘再读回」这一步的损失，C9 还要加上
「NC 三线性插值到 GLL 点」。所以 C18 是 C9 的下界，报告判据阈值时要用 C9，不能拿 C18
充数。

C19 是多震源方案存在的理由。任何"A 优于 B"的结论都要给出它在 10 个事件中成立了几个；
只在少数事件上成立的名次，写进论文之前要先解释清楚为什么。

C14 和 C15 是随共用地壳方案一起加的两项。C15 尤其要紧：读到 `solver_data.bin` 的弹性模量记录意味着记录偏移不再免疫 Par_file 开关的变动（§1），而偏移一旦错位，反算出的"速度"会离谱到当场被值域检查拦下，不会静默通过。两项都由 `tests/test_depth_datum.py` 覆盖，换网格或改 Par_file 开关后应重跑。

---

## 8. 旧流程中不再需要的部分

以下机制在新流程中全部作废，相关代码可以停用（建议保留文件但不再调用，便于回溯）：

| 机制 | 位置 | 作废原因 |
| --- | --- | --- |
| `--use-3d-fallback` 反算参考速度 | `convert_nc_to_gll.py` `read_solver_data(extract_velocities=True)` | 底图现成，无需反算 |
| `--use-perturbation` 扰动插值 | 同上 `interpolate_from_netcdf()` | 参考场不匹配（§9.1），且新流程不需要 |
| 第二次 meshfem 前的 s362ani 参考 | `run_forward.bash` Phase 2 | 只保留坐标用途 |
| `--horizontal-fill {nearest,ref,blend}` | `convert_nc_to_gll.py` | 由统一的三维 taper 取代 |
| 本地建模 + `--skip-unchanged` + `manifest.txt` 组装 | `build_hybrid_gll.py` / `run_all_test.sh` | 前提不成立：本地坐标与服务器网格无法保证一致（§3.2） |
| 网格指纹核对（`mesh_prefix_fingerprint`） | 同上 | 比对对象错了，由 `check_mesh_diff.py` 的 mesh0 复现核对取代（§6.5） |
| `I_MPI_FABRICS=shm:dapl` + `I_MPI_FALLBACK=0` | 旧作业脚本 | meshfem 的 MPI 界面错误真凶（§6.6） |
| Slurm 3 节点 × 48 | 旧作业脚本 | 改为有成功记录的 4 × 36 |

`convert_nc_to_gll.py` 本身**仍然有用**：它是把任意公开模型转成 GLL 的通用工具，是 SMART-py 工具箱要发表的部分。只是在本轮对比实验中，改由 `build_hybrid_gll.py` 承担。两者可以共享底层函数。

---

## 9. 已知风险与未决问题

### 9.1 旧流程的参考场不匹配（本方案已规避，但需在文档中留痕）

实测结论：FWEA23 的 `vs0` 在 100 km 深度上，82181 个格点只有 **1 个唯一值**（4.436 km/s）；200 km 处横向标准差 0.004 km/s。EARA2024 的 `vsv_ref` 同样，100 km 处 17 个唯一值、std 0.0002。**两个模型的 NC 参考场在地幔里都是严格一维的**，地壳部分才是三维的（CRUST1.0）。

而旧流程 Phase 2 用 `MODEL = s362ani`，给出的是三维地幔参考。代入扰动公式后地幔里会凭空多出整个 S362ANI 的异常场，量级（±2–4%）与 FWEA23 自身异常（200 km 处实测 2.2%）相当。

`Models_GLL_workflow_server.md` §13.6 声称「FWEA23 参考模型 s362ani+CRUST1.0 ✅ 完全匹配」，这一条与实测不符，应予修正。§13.5 的「预期效果」表（预测 410/660 处 <0.1%、深部 ~0%）也没有证据支撑——`figures/gll_compare/report.txt` 实测 VSV 在 410–660 km 是 0.890%、1000–3000 km 是 0.448%，差 5–10 倍。

**本方案不使用扰动插值，因此不受此问题影响。**

### 9.2 尚未确认的事项

| 项 | 说明 | 何时能确认 |
| --- | --- | --- |
| ~~chunk 是否覆盖 EARA2024 东界~~ | ✅ 已确认覆盖，chunk 东界 165.13°E > 160°E（§3.3） | 已完成 |
| ~~网格逐元素一致性~~ | ✅ 已确认，probe 实测健康节点上 144/144 逐位相同（§6.16） | 已完成 |
| SinoScope 的 rho 来源 | 若为经验标定而非反演量，需在论文中说明 | 查原文献 |
| ~~深度基准（地表 vs 海平面 vs 参考球）~~ | ✅ 已定案：GLL 侧统一为地表以下（§4.5），CRUST1.0 Moho 亦为地表以下（§4.2）。仍需核对各 NC 的 depth 属性是否也是地表基准 | 查各模型 NC 元数据 |
| 我们的 Moho 与 SPECFEM CAP 平滑后的 Moho 的实际差距 | 目前靠 15 km taper 兜住，未量化 | 可在 mesher 后从底图反查 |

### 9.3 本方案的固有局限

- **不比较地壳模型。** 四个 `_c1` run 的地壳完全相同（CRUST1.0）。这是刻意的设计，为的是把差异干净地归因到地幔；代价是地壳结构的贡献完全不参与评分。若要比较地壳，需另设一轮让各模型带自己的地壳，但那会撞上三个问题：SinoScope 根本没有地壳数据（最浅 20 km）、FWEA23/EARA2024 在 0 km 处分别有 28% / 22% 的海域缺值、以及各模型自己的 Moho 与网格拉伸所依据的 CRUST1.0 Moho 不一致。
- **共用地壳精度不高。** CRUST1.0 是 1°×1° 全球编译模型，比 FWEA23 的 FWI 地壳差不少，四个 `_c1` run 的绝对拟合都会因此变差，9–20 s 尤其明显。各组同等变差，排名有效，但绝对 misfit 水平不代表这些模型的真实能力。`BASE` 组用于报告最好拟合。
- **替换区外与浅部依赖 s362ani。** 改用 CRUST1.0 底图后，替换区以外的地幔从 FWEA23 换成了 s362ani（度 18 全球模型），远场结构比早先的方案差。四个 `_c1` 组共享，不影响可比性，但会抬高所有组的绝对 misfit。
- **替换区外的敏感核不参与区分。** 实测受影响点占 GLL 总点数的 22.9%、完全替换的 12.8%（moho 上界 + 15/50 taper 下的 dry-run 结果），其余 77% 四个 `_c1` run 逐位一致。射线路径不穿过替换区的台站-事件对波形几乎相同，对排名没有贡献。后处理时必须统计有效数据量，并在选台站-事件对时优先覆盖替换区。
- **替换区取的是三模型 NC 交集，牺牲了各模型自己的外围覆盖。** 技术上每个模型都可以在自己的完整 NC 范围内嵌入，但那样三个 run 的替换区形状不同，对比会被区域形状混淆。交集是受控对比的正确选择。若需要各模型的完整表现，可另跑一组「各自完整范围」的结果作为次要结论，但不能与主结论混用。
- **10 个震源在方位上并不均匀。** 相对 chunk 中心的最大方位角空隙 71°，缺口在正北到东北
  （方位角 349°–60°，西伯利亚地台方向）。那里是稳定克拉通，不产生够大的地震，不是选源没选
  好，但意味着自北进入研究区的路径采样不足，该方向上的模型差异分辨不出来。`select_test_events.py`
  的方位角插图把这个缺口画了出来。
- **CFL 未随模型重算。** `DT = 0.1016261` 是 mesher 按 CRUST1.0 + s362ani 算出来的。地壳既然保持 CRUST1.0 不变，浅部（速度最低、单元最小、最吃 CFL 的地方）没有变化，风险比替换地壳的方案小得多。但替换区内若某处 vp 显著高于 s362ani，仍有余量收窄的可能，`output_solver.txt` 的位移量级需要照例检查。

### 9.4 插值误差基线不再有干净的估计（🔴）

早先的方案里，`BASE` 与 `FWEA23_nc` 只差一件事——FWEA23 是走原生 GLL 还是走 NC 插值——所以两者之差就是纯粹的插值误差基线。§3.5 实测该基线的 vsv rms 为 0.78%，而 EARA2024 只有 1.40%、SinoScope 2.37%，基线占真实信号的 33–56%，本身就偏大。

改用 CRUST1.0 共用地壳之后，`BASE` 与 `FWEA23_c1` 之间**同时**变了两样东西：地壳（FWEA23 FWI 地壳 → CRUST1.0）和地幔的走法（原生 GLL → NC 插值）。这个差值不再是纯插值误差，而是两者之和，无法直接用作判据。

新加的 `S362ANI_ref` 组（§2.3）只解决了其中一小块：它给出「GLL 写盘再读回」的代价，
这是分辨率的绝对下限，但不含 NC 三线性插值那一项，因此不能当作插值误差基线用。

三条可选的补救路线，尚未决定：

1. **加一个 `FWEA23_gll_c1` 组**：CRUST1.0 地壳 + FWEA23 的**原生 GLL** 地幔（从 `model_updated` 而非 NC 取值，用同一个 Moho 上界和 taper）。它与 `FWEA23_c1` 只差插值一项，基线重新变干净。代价是多一次正演。
2. **在模型空间估计**：把 `model_updated` 的 GLL 值与 FWEA23 NC 插值到同一批 GLL 点上直接比，只在替换区内统计。这不需要正演，但给出的是模型空间的误差，不是波形空间的。
3. **接受混合基线**，在论文里明确表述为"换地壳 + 插值"的合计代价。最省事，但削弱了结论强度。

原先列出的三种误差来源仍然适用于第 1、2 条：三线性插值本身的误差；发布的 FWEA23 NC 可能是 `model_updated` 的重采样/平滑版本（0.25° 规则网格 vs GLL 点）；两者可能来自不同的 FWI 迭代次数（查原文献元数据即可确认）。若第三条成立，基线的物理含义要从「插值误差」改成「插值误差 + 版本差异」。

这一条与项目总纲 §9 的 🔴「各模型走 NC→GLL 插值 vs FWEA23 有原生 GLL，存在对比不公平性」是同一个问题。

### 9.5 ~~mesher 坐标输出不可复现~~（已排除，2026-08-03）

这一条曾被列为 🔴 阻塞项，现已证明是误判，保留记录是因为它导致了一轮不必要的流程改造。

原始观察：同一集群、同一 Par_file、同一批二进制，两次 mesher 产出的 `solver_data.bin`
坐标段不逐位相同（56/144、4/144），`nspec` / `nglob` 却完全一致。

probe 对照实验（§6.16）在健康节点上连跑四轮，144 个 proc 全部逐位相同，坐标相对差 0.0。
**mesher 是确定性的**，此前的差异来自故障节点 c08r1 上产出的损坏数据。

尽管前提被推翻，为此建立的两条机制仍然保留，因为它们的价值换了个方向：

- 插值坐标取自当次流程的 mesher 输出（§6.2）。原意是防不可复现，现在是保证来源单一，成本为零。
- solver 前显式核对坐标（§6.5）。原意是拦不可复现，现在是**故障节点探测器**——既然正常必然逐位相同，任何差异都说明这次跑在了坏硬件上。

判读规则相应收紧：

| 核对结果 | 含义 | 应对 |
| --- | --- | --- |
| 144/144 逐位相同 | 正常 | 继续 solver |
| 有差异（无论 `ibool` 还是坐标） | 这次 mesher 的输出不可信 | 换节点重跑；机架反复出问题就加进 `BAD_RACKS` |

**不要因为核对失败就调大 `COORD_TOL` 放行。** 既然已知正常情况下差异恒为零，放宽容差
只会把硬件故障重新变成静默错误。`COORD_TOL` 保留 1e-6 是为了兼容将来可能出现的异构
节点场景，不是给当前的失败开口子。

---

_创建日期：2026-08-03_
_最近修订：2026-08-03（扩展为 10 震源 × 5 组 + `S362ANI_ref` 往返对照；新增 §2.4 / §2.5 震源选取、§6.17 作业脚本自检、C16–C19；§6.1 / §6.3 / §6.8–§6.13 按多事件改写）_
_上一版：2026-08-03（probe 实验定案：mesher 确定性、几何与 MODEL 无关、失败根因为故障节点；§3.2 / §6.6 / §6.16 / §9.2 / §9.5 相应改写）_
_关联文档：`Models_GLL_workflow_server.md`（编译/MPI/集群配置部分仍然有效；其 §19 把跨机架列为根因的结论已被本文 §6.6 取代）_
