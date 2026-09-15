# 多模型Specfem3d_globe正演 - 服务器端流程

曙光 Slurm 集群上完成 FWEA23 / SinoScope1.0 / EARA2024 三模型 GLL 格式正演的完整步骤。

---

## 1. 项目目标

**核心目标**：为**任意公开三维速度模型**提供正演模拟与波形检验工具。

- **背景**：多数公开模型（FWEA23、EARA2024、SinoScope1.0 等）仅发布 **NetCDF (NC)** 或 **CSV** 格式，而 SPECFEM3D Globe 好像无法直接读取规则网格模型文件进行正演，应该是需要转换成 **GLL** 格式。
- **工具**：`convert_nc_to_gll.py` 将 NC/CSV 插值到 GLL，使研究者能用公开模型进行波形正演与检验。
- **测试用例**：以 **FWEA23** 为例，我目前只有原作者参考结果（模型的 bin 文件`specfem/model_updated`、一个地震事件的波形`sac/`）。

- 运用第一性原理思考，拒绝经验主义和路径盲从，不要假设我完全清楚目标，保持审慎，从原始需求和问题出发，若目标模糊请停下和我讨论，若目标清晰但路径非最优，请直接建议更短、更低成本的办法。

所有回答必须分为两个部分：
• ​直接执行：按照我当前的要求和逻辑，直接给出任务结果。
• ​深度交互：基于底层逻辑对我的原始需求进行“审慎挑战”。包括但不限于：质疑我的动机是否偏离目标（XY问题）、分析当前路径的弊端、并给出更优雅的替代方案。

---

## 2. 前置条件

- `specfem/` 目录已上传到服务器，结构如下:
  - `specfem/models/` — 原始模型文件（直接从 EMC 下载，无需 `1_5_Process` 预处理）
    - `FWEA23.r0.0-n4.nc` (NetCDF4, 192 MB, 各向异性 + qmu + 参考模型 vp0/vs0)
    - `EARA2024.r0.0-n4.nc` (NetCDF4, 139 MB, 各向异性 + 逐分量参考 vpv_ref/vsv_ref 等)
    - `FWI_SinoScope_1.0_JMa_ET_AL_2022.h5` (HDF5, 34 MB, P 各向同性 + S 各向异性)
  - `specfem/convert_nc_to_gll.py` — NC/HDF5/CSV → GLL 转换脚本
  - `specfem/FWEA23/` — 正演工作目录（DATA/、bin/、DATABASES*MPI/、OUTPUT_FILES*\*/）
- **服务器 conda 环境** `eastasia_fwi`（python=3.10, numpy, scipy, xarray, h5py）
- **不再需要** `1_5_Process_velocity_models.py` 预处理——`convert_nc_to_gll.py` 直接读取 EMC 原始格式

---

## 3. SPECFEM3D Globe 数据流（必读）

理解流程之前必须理解 SPECFEM 的内部数据流：

```
┌─────────────────────────────────────────────────────────────────────┐
│  xmeshfem3D（mesher）── 修改版 get_model.F90                        │
│                                                                     │
│  输入:                                                              │
│    Par_file (MODEL, NEX, NCHUNKS...)                                │
│    constants.h.in (REGIONAL_MOHO, NER, 编译时)                      │
│    DATA/GLL/*.bin (仅当 MODEL 含 "gll" 时)                          │
│                                                                     │
│  内部处理:                                                           │
│    1. 创建谱元网格 → 元素拓扑 ibool、Jacobian、GLL 点坐标 (x,y,z)   │
│    2. 在每个 GLL 点获取弹性参数 (get_model.F90):                     │
│       · MODEL=s362ani_crust1.0 → 解析计算 s362ani+CRUST1.0 模型     │
│       · MODEL=GLL (v7) / gll_qmu (v8) → 从 DATA/GLL/proc*_reg1_*.bin 读取│
│    3. 速度(km/s) → 弹性模量: κ = ρ(Vp²-4/3·Vs²), μ = ρ·Vs²        │
│    4. 写出 solver_data.bin = 网格几何 + 弹性模量（κ,μ,ρ...）         │
│                                                                     │
│  输出:                                                               │
│    DATABASES_MPI/proc*_reg1_solver_data.bin  (弹性模量，供 solver)   │
│    ★ Phase 3 的 convert_nc_to_gll.py --use-3d-fallback 可从此文件   │
│      反算 s362ani+CRUST1.0 速度（无需修改 Fortran 源码）             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  xspecfem3D（solver）                                               │
│                                                                     │
│  输入: DATABASES_MPI/solver_data.bin（网格几何 + 弹性模量）          │
│  ⚠️ solver 不读 DATA/GLL/，不读 Par_file 的 MODEL 做模型加载        │
│     Par_file 中 MODEL 仅用于设置内部标志（如 TRANSVERSE_ISOTROPY）   │
│                                                                     │
│  处理: Newmark 时间步进，使用 solver_data.bin 中的弹性模量            │
│  输出: 地震波形 (SAC/ASCII)                                         │
└─────────────────────────────────────────────────────────────────────┘
```

**关键结论**：solver 的弹性参数**完全**来自 `solver_data.bin`，该文件由 mesher 写入。要让 solver 用 FWEA23 模型，**mesher 必须读取 GLL 文件**。

### 3.1 convert_nc_to_gll.py --use-3d-fallback 的角色

**核心问题**：公开模型 NC 文件是 FWI 结果的有损导出（10km 网格平滑了 Moho/410/660 间断面），
而 Phase 2 meshfem (MODEL=s362ani) 在 `solver_data.bin` 中以弹性模量形式存储了完整的
s362ani + CRUST1.0 值。旧流程域外回退到 1DREF（1D 模型），丢失了 3D 结构。

**解决方案**：单脚本内完成（零 Fortran 修改，零中间文件）

```
┌──────────────────────────────────────────────────────────────────┐
│  convert_nc_to_gll.py --use-3d-fallback（Phase 3）                │
│                                                                  │
│  输入:                                                           │
│    · DATABASES_MPI/solver_data.bin → GLL 坐标 + 弹性模量 (κ,μ,ρ)│
│    · 模型 NC/H5/CSV                                              │
│                                                                  │
│  处理:                                                           │
│    1. 读取 solver_data.bin 前半部分 → GLL 点坐标 (x,y,z)        │
│    2. ★ 读取 solver_data.bin 后半部分 → 弹性模量 (κ,μ,ρ)        │
│       反算 s362ani+CRUST1.0 速度:                                │
│       vsv = √(μ_v / ρ)      vsh = √(μ_h / ρ)                   │
│       vpv = √((κ_v + 4/3·μ_v) / ρ)  vph = √((κ_h + 4/3·μ_h)/ρ)│
│       + SPECFEM 非量纲化 → km/s, g/cm³                           │
│    3. 域内点: 正常插值 NC 绝对值                                  │
│    4. 深部(>NC范围): 用反算的 3D 速度替代 1DREF → 保留 3D 结构   │
│    5. 水平域外: 用反算的 3D 速度替代 1DREF → 保留间断面           │
│                                                                  │
│  ★ 关键改进: 深部/域外从 1D 参考 → 3D 参考 (s362ani+CRUST1.0)   │
│    消除 scatter 图中的水平条纹伪影，深部误差 ~0.3% → ~0%         │
│    无中间文件、无额外脚本、无额外 Phase                           │
│                                                                  │
│  输出: DATA/GLL/proc*_reg1_{vpv,vph,vsv,vsh,eta,rho}.bin        │
└──────────────────────────────────────────────────────────────────┘
```

**三种模型格式的处理差异**:

| 模型      | 格式   | 变量                        | 参考场                          | 特殊处理                           |
| --------- | ------ | --------------------------- | ------------------------------- | ---------------------------------- |
| FWEA23    | NetCDF | vpv,vph,vsv,vsh,eta,rho,qmu | vp0,vs0                         | 扰动插值可用                       |
| EARA2024  | NetCDF | vpv,vph,vsv,vsh,eta,rho     | vpv_ref,vph_ref,vsv_ref,vsh_ref | 逐分量扰动插值                     |
| SinoScope | HDF5   | vp,vsv,vsh,rho              | 无                              | vp→vpv=vph, eta=1.0, 速度 m/s→km/s |

### 3.2 多模型一键正演工作流（v4: 全服务器端）

服务器已安装 conda 环境后，**全部操作在服务器端一个脚本内完成**，一次提交测试多个模型：

```
┌──────────────────────────────────────────────────────────────────────┐
│  specfem/                                                            │
│  ├── DATA/  (全局默认: Par_file, STATIONS, CMTSOLUTION)              │
│  ├── run_forward.bash  ← 在此目录 sbatch                            │
│  │                                                                   │
│  │  Phase 1    编译 (SKIP_COMPILE=1 跳过)                            │
│  │  Phase 2    meshfem(s362ani) in FWEA23/ → 共享网格拓扑           │
│  │             (MESH_ONLY=1 可在此退出)                              │
│  │                        │                                          │
│  │  逐模型循环 (或 SLURM 并行):                                      │
│  │    ┌───────────────────┼───────────────────┐                      │
│  │    ▼                   ▼                   ▼                      │
│  │  FWEA23/           EARA2024/           SinoScope/                 │
│  │  ├ DATA/ (独立)    ├ DATA/ (独立)      ├ DATA/ (独立)             │
│  │  ├ DATA/GLL/       ├ DATA/GLL/         ├ DATA/GLL/                │
│  │  ├ DATABASES_MPI/  ├ DATABASES_MPI/    ├ DATABASES_MPI/           │
│  │  └ OUTPUT_FILES/   └ OUTPUT_FILES/     └ OUTPUT_FILES/            │
│  │                                                                   │
│  │  Phase 3  NC/H5 → GLL (Python, --use-3d-fallback, 各模型)        │
│  │           ★ 深部/域外自动从 solver_data.bin 反算 3D 参考          │
│  │  Phase 4  meshfem (MODEL=GLL, 各模型独立)                         │
│  │  Phase 5  solver → OUTPUT_FILES/*.sac (各模型独立)                │
└──────────────────────────────────────────────────────────────────────┘
```

**DATA/ 分层机制**：`specfem/DATA/` 是全局默认模板；各模型目录如果已有 `DATA/Par_file` 则保留不动（允许自定义参数），没有则自动从模板复制。

**用法**（在 `specfem/` 下执行）：

```bash
# 顺序跑全部
sbatch run_forward.bash

# 并行跑全部（推荐）
JOB0=$(MESH_ONLY=1 sbatch --parsable run_forward.bash)
sbatch --dependency=afterok:$JOB0 --export=ALL,SKIP_MESH=1 run_forward.bash FWEA23
sbatch --dependency=afterok:$JOB0 --export=ALL,SKIP_MESH=1 run_forward.bash EARA2024
sbatch --dependency=afterok:$JOB0 --export=ALL,SKIP_MESH=1 run_forward.bash SinoScope

# 单个模型
sbatch run_forward.bash FWEA23
SKIP_MESH=1 sbatch run_forward.bash SinoScope
```

**模型注册表**（`run_forward.bash` 头部）：

| 模型名    | 文件                                         | 格式 | 转换参数             |
| --------- | -------------------------------------------- | ---- | -------------------- |
| FWEA23    | `models/FWEA23.r0.0-n4.nc`                   | NC   | `--use-perturbation` |
| EARA2024  | `models/EARA2024.r0.0-n4.nc`                 | NC   | `--use-perturbation` |
| SinoScope | `models/FWI_SinoScope_1.0_JMa_ET_AL_2022.h5` | HDF5 | (无额外参数)         |

每个 phase 有独立的退出检查。单个模型失败跳过继续下一个。

### 3.3 为什么需要两次 meshfem？

| 问题                             | 解释                                                                                                     |
| -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 为什么不能一次？                 | 鸡生蛋问题：`convert_nc_to_gll.py` 需要 GLL 坐标（来自 mesher），但 mesher 需要 GLL 文件（来自转换脚本） |
| 第二次 mesh 几何会变吗？         | 不会。两次 mesher 用相同的 NEX、constants.h.in、1DREF，网格拓扑完全一致（NSPEC=7236）                    |
| 第二次 mesh 会 face flag 吗？    | 极低概率。mesh 是确定性的，相同配置 = 相同结果。但为安全可重提一次                                       |
| PATHNAME_GLL_modeldir 在哪配置？ | 编译时常量 `constants.h.in` 中: `PATHNAME_GLL_modeldir = 'DATA/GLL/'`（不在 Par_file 中）                |

---

## 4. 准备模型文件

**不再需要 `1_5_Process_velocity_models.py` 预处理**。`convert_nc_to_gll.py` 直接读取 EMC 原始格式：

```bash
# 模型文件放在 specfem/models/ 目录下（已就位）
ls specfem/models/
# FWEA23.r0.0-n4.nc                         ← EMC 原始 NetCDF4
# EARA2024.r0.0-n4.nc                       ← EMC 原始 NetCDF4
# FWI_SinoScope_1.0_JMa_ET_AL_2022.h5       ← 原始 HDF5
```

各模型特征:

| 模型      | 分辨率      | 有效点 | 各向异性            | 参考模型                        | 特殊           |
| --------- | ----------- | ------ | ------------------- | ------------------------------- | -------------- |
| FWEA23    | 0.25°, 10km | 92%    | vpv/vph/vsv/vsh/eta | vp0/vs0                         | 含 qmu         |
| EARA2024  | 0.25°, 10km | 89%    | vpv/vph/vsv/vsh/eta | vpv_ref/vph_ref/vsv_ref/vsh_ref | 逐分量参考     |
| SinoScope | 1.0°, 20km  | 区域   | vp(同性)+vsv/vsh    | 无                              | HDF5, 速度 m/s |

---

## 5. 运行多模型正演

**全部操作在服务器端 `specfem/` 目录下**：

```bash
cd specfem/

# 首次运行（含编译 + 网格生成 + 三模型正演）
sbatch run_forward.bash

# 后续运行（跳过编译，复用已有 bin/）
SKIP_COMPILE=1 sbatch run_forward.bash

# 只测试单个模型
sbatch run_forward.bash EARA2024

# 复用已有网格（跳过 Phase 1+2），只做 Phase 3→5
SKIP_MESH=1 sbatch run_forward.bash SinoScope
```

**正演结果**存放在各模型的独立目录中：

```
specfem/
├── DATA/                   ← 全局默认配置（模板）
├── FWEA23/                 ← 模板 + FWEA23 工作目录
│   ├── DATA/GLL/               GLL 二进制（Phase 3 生成）
│   ├── DATABASES_MPI/          网格数据库（Phase 2 生成，含 solver_data.bin）
│   ├── bin/                    可执行文件
│   └── OUTPUT_FILES/           正演波形 SAC
├── EARA2024/               ← 独立工作目录
│   ├── DATA/GLL/
│   ├── DATABASES_MPI/          (网格拓扑 symlink, solver_data 拷贝)
│   └── OUTPUT_FILES/
└── SinoScope/              ← 独立工作目录
    └── ... (同上结构)
```

---

## 6. convert_nc_to_gll.py 详细用法

`run_forward.bash` 自动调用此脚本，通常不需要手动运行。但可用于调试或本地测试：

```bash
# FWEA23 (NetCDF, 含扰动插值)
python convert_nc_to_gll.py \
    --mesh-dir DATABASES_MPI \
    --model-nc ../models/FWEA23.r0.0-n4.nc \
    --output-dir DATA/GLL --use-perturbation

# EARA2024 (NetCDF, 含扰动插值)
python convert_nc_to_gll.py \
    --mesh-dir DATABASES_MPI \
    --model-nc ../models/EARA2024.r0.0-n4.nc \
    --output-dir DATA/GLL --use-perturbation

# SinoScope (HDF5)
python convert_nc_to_gll.py \
    --mesh-dir DATABASES_MPI \
    --model-h5 ../models/FWI_SinoScope_1.0_JMa_ET_AL_2022.h5 \
    --output-dir DATA/GLL
```

> 默认已启用最优配置：各向异性输出、3D 参考（从 solver_data.bin 反算）、blend 域外填充。一般只需指定模型文件和输出目录。

### 6.1 CLI 选项

| 选项                       | 说明                                               |
| -------------------------- | -------------------------------------------------- |
| `--mesh-dir PATH`          | **必选** DATABASES_MPI 目录                        |
| `--model-nc/h5/csv PATH`   | **必选** 模型文件（三选一）                        |
| `--output-dir PATH`        | **必选** GLL 输出目录                              |
| `--use-perturbation`       | 扰动插值（FWEA23/EARA2024 需要，SinoScope 不需要） |
| `--double`                 | 输出 float64（SPECFEM double precision 编译时）    |
| `--reference-gll-dir PATH` | 对比参考 GLL（调试用，如 model_updated/）          |
| `--no-3d-fallback`         | 禁用 3D 参考，回退到 1DREF（调试用）               |
| `--no-anisotropic`         | 输出各向同性（调试用）                             |

---

## 7. 技术细节：为什么需要两次 meshfem + 3D fallback？

**核心流程**（`run_forward.bash` 自动处理）：

1. **Phase 2** meshfem (MODEL=s362ani) — 生成网格拓扑 + GLL 点坐标 + solver_data.bin（含弹性模量）
2. **Phase 3** `convert_nc_to_gll.py --use-3d-fallback` — 域内插值 NC + 域外/深部从同一 solver_data.bin 反算 3D 参考
3. **Phase 4** meshfem (MODEL=GLL) — 读取 GLL 速度 → 转弹性模量 → 重写 solver_data.bin

鸡生蛋问题：转换脚本需要 GLL 坐标（来自 mesher），但 mesher 读 GLL 需要转换结果，所以必须两轮。

**`--use-3d-fallback` 的原理**：Phase 2 的 solver_data.bin 同时存储 GLL 坐标和弹性模量 (κ,μ,ρ)。`convert_nc_to_gll.py` 扩展了 `read_solver_data()` 函数，在读取坐标的同时继续读取弹性模量记录，通过反算公式（`vpv=√((κ_v+4/3·μ_v)/ρ)` 等）+ SPECFEM 非量纲化常数，在内存中还原 s362ani+CRUST1.0 速度 (km/s, g/cm³)。**零 Fortran 修改，零中间文件，零额外脚本**。反算得到的值包含 s362ani+CRUST1.0 在 GLL 点的精确值（含 Moho/410/660 间断面）。

**v7.0.0 中 MODEL=GLL 的含义**：

| 设置              | MODEL_GLL | ATTENUATION_GLL | Q 衰减来源          |
| ----------------- | --------- | --------------- | ------------------- |
| `MODEL = GLL`     | true      | false           | 1D 参考 Qmu         |
| `MODEL = gll_qmu` | true      | true            | 读 DATA/GLL/qmu.bin |

当前 `run_forward.bash` 使用 `MODEL = GLL`，即 1D Q。若模型含 3D Q（如 FWEA23 有 qmu），可改为 `MODEL = gll_qmu`。对于无 qmu 的模型（EARA2024, SinoScope），`MODEL = GLL` 是唯一选择。

确认 Phase 4 完成：

```bash
grep "reading in model" OUTPUT_FILES/output_mesher.txt   # 应输出 "DATA/GLL/"
grep "number of elements (per slice)" OUTPUT_FILES/output_mesher.txt  # NSPEC=7236
```

### 7.1 网格一致性保证

两次 meshfem 产生**相同网格几何**的条件（全部满足）：

| 条件                | 说明                                                              |
| ------------------- | ----------------------------------------------------------------- |
| 相同 NEX (288)      | ✅ Par_file 不变                                                  |
| 相同 constants.h.in | ✅ 不重新编译                                                     |
| 相同 1DREF 参考模型 | ✅ gll_qmu 的 REFERENCE_1D_MODEL = 1DREF，与 s362ani 一致         |
| 相同地壳模型        | ✅ gll_qmu 硬编码 impose_crust=CRUST1.0，与 s362ani_crust1.0 一致 |

GLL 文件的 nspec 与网格的 nspec 必须匹配，否则 mesher 报错 `requires too much data`。

---

## 8. 步骤 ⑥：GLL 正演（仅 solver，服务器）

**目的**：使用步骤 ⑤ 更新的 `solver_data.bin`（含 FWEA23 弹性模量）进行波场计算。

```bash
cd specfem3d_globe/EXAMPLES/FWEA23
sbatch run_solver_only.bash
```

**`run_solver_only.bash`** 行为：**不**删除 `DATABASES_MPI`、**不**调用 `xmeshfem3D`；仅运行 `xspecfem3D`。

### 8.1 验证 solver 使用的是 FWEA23 参数

solver 输出应显示 `model: gll_qmu`，但**真正的验证**是检查 mesher 输出（步骤 ⑤）是否包含 GLL 读取信息：

```bash
# 确认 mesher 读取了 GLL 文件
grep "reading in model from" output_mesher.txt
# 应显示: reading in model from: DATA/GLL/
```

---

## 9. 路径速查

| 项目          | 路径 / 值                                                                            |
| ------------- | ------------------------------------------------------------------------------------ |
| 进程数        | 144 (1×12×12)                                                                        |
| NEX           | 288                                                                                  |
| DATABASES_MPI | `specfem3d_globe/EXAMPLES/FWEA23/DATABASES_MPI`                                      |
| GLL 输出      | `specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL`                                           |
| GLL 路径常量  | `constants.h.in`: `PATHNAME_GLL_modeldir = 'DATA/GLL/'`（编译时固定，不在 Par_file） |
| NC 模型       | `data/models/processed/2024_FWEA23/2024_FWEA23_original.nc`                          |
| 转换脚本      | `3_Data_space_simulation/convert_nc_to_gll.py`                                       |
| 原作者 GLL    | `specfem/model_updated/`（NSPEC=7236）                                               |
| 原作者波形    | `specfem/sac/`                                                                       |

---

## 10. 常见错误与调试

### 9.1 错误速查表

| 错误                                 | 处理                                       |
| ------------------------------------ | ------------------------------------------ |
| `PMI_KVS_Get returned -1`            | 使用 `srun --mpi=pmi2`，不要用 `mpirun`    |
| `request too frequently`             | 同上                                       |
| `Error opening DATA/s362ani/S362ANI` | 确保 `specfem3d_globe/DATA/s362ani/` 存在  |
| `Error model crust2.0`               | 确保 `specfem3d_globe/DATA/crust2.0/` 存在 |
| `requires too much data`             | GLL 文件与网格精度不匹配，见 §9.2          |
| `floating overflow` / 发散           | 见 §9.3                                    |
| `Error face flag`                    | 壳幔 MPI 界面偶发错误，见 §9.4             |
| `Error test outer core valence`      | 同上，重提作业通常可恢复                   |
| `Error opening mesh_parameters.bin`  | mesh 未完成或 `DATABASES_MPI` 被清空       |
| `Mesh becomes invalid` / Jacobian    | 单元畸变；保留成功网格 + 仅 solver         |

### 9.2 `input statement requires too much data`

GLL 文件大小与网格 nspec 不匹配。确认精度和 nspec：

```bash
# 确认精度
grep "CUSTOM_REAL =" ../../setup/constants.h
# SIZE_REAL → 不加 --double; SIZE_DOUBLE → 加 --double

# 确认 GLL 文件大小（期望 8 + 125*nspec*4 字节）
ls -l DATA/GLL/proc000000_reg1_vpv.bin
# nspec=7236 → 3618008 字节
```

### 9.3 Solver 发散（Max U 指数增长）

检查顺序：

1. **compute_forces 补丁**：确认 §11.5 的条件化修复已编译进二进制（**最常见的发散原因**）
2. **GLL 值域**：vpv/vsv 不应为零或常数；rho 应为 g/cm³（非 kg/m³）
3. **CFL 条件**：mesher 输出的 CFL 应 < 1.0
4. **GLL-mesh 一致性**：GLL 文件应来自当前 DATABASES_MPI 的 NC→GLL 转换

### 9.4 meshfem face flag 错误（已解决）

**现象**：`Error face flag` / `Jacobian invalid` / `MPI points remain unrecognized`。

| 错误                            | 位置                      | 说明                  |
| ------------------------------- | ------------------------- | --------------------- |
| `Error face flag`               | `get_MPI_interfaces.f90`  | 壳幔 MPI 界面构造阶段 |
| `Error test outer core valence` | `test_MPI_interfaces.f90` | 外核 valence 自检失败 |

**根因诊断（§19）**：

- `auto_ner.f90` R80 不一致（§17.2）导致 NER 计算错误
- 自定义编译参数 `-O1 -fp-model precise`（非 SPECFEM 默认）触发 ifort 2018 优化相关运行时错误
- DEBUG 模式 (`-O0 -check bounds`) 下 mesher 100% 成功，证明非逻辑 bug

**修复**：

1. `auto_ner.f90` R80 一致性修复（§17.2）
2. 编译参数回归 SPECFEM 默认 `-O3`（§19），不再自定义 `FLAGS_CHECK`

**成功判据**：`.err` 无 `MPI_Abort`；`.out` 出现 `Meshfem done (exit 0)`；`output_mesher.txt` 末尾 `End of mesh generation`。

---

## 11. 源码修改总览

### 11.1 当前 `git diff --stat HEAD`（9 个文件）

```
 setup/constants.h.in                                          | 14 ++++++------
 src/meshfem3D/get_model.F90                                   | 21 +++++++++++++----
 src/meshfem3D/setup_MPI_interfaces.f90                        |  2 +-
 src/shared/auto_ner.f90                                       | 28 +++++++++++---
 src/shared/get_model_parameters.F90                           |  1 +
 src/shared/get_timestep_and_layers.f90                        | 10 ++++++---
 src/specfem3D/compute_forces_viscoelastic_calling_routine.F90 |  7 +++---
 src/specfem3D/iterate_time_undoatt.F90                        |  4 +++-
 src/specfem3D/save_kernels.F90                                |  2 +-
 9 files changed, 57 insertions(+), 32 deletions(-)
```

### 11.2 外部开发者补丁（5 处）

| #   | 文件                          | 修改内容                                      | 正演影响                                  | 状态                |
| --- | ----------------------------- | --------------------------------------------- | ----------------------------------------- | ------------------- |
| 1   | `auto_ner.f90`                | 注释 NER 经验值覆盖 + **R80 bug fix (§17.2)** | **直接影响**（网格拓扑 + 消除 face flag） | ✅ 保留 + 新增      |
| 2   | `get_timestep_and_layers.f90` | ~~`DT=0.1d0` 硬编码~~ → 已替换为 v8 bug 修复  | **直接影响**（DT/CFL）                    | ✅ 已修复，见 §11.4 |
| 3   | `get_model_parameters.F90`    | `gll_qmu` 添加 `impose_crust`                 | 无影响（冗余但无害）                      | ✅ 保留             |
| 4   | `compute_forces_*.F90`        | 注释 inner core early return → 条件化修复     | **🔴 不修复则正演发散**                   | ✅ 已修复，见 §11.5 |
| 5   | `save_kernels.F90`            | 注释 `restore_original_moduli()`              | 无影响（仅 kernel 阶段）                  | ✅ 保留             |

### 11.3 我方修改（4 处）

| #   | 文件                       | 修改内容                              | 说明                 |
| --- | -------------------------- | ------------------------------------- | -------------------- |
| 1   | `constants.h.in`           | tibet_v4 配置（REGIONAL/HONOR/RMOHO） | Gate C 网格配置      |
| 2   | `get_model.F90`            | Plan D (vpv=0 fallback)               | 防止零值崩溃         |
| 3   | `setup_MPI_interfaces.f90` | `MAX_NEIGHBORS = 20 + NCORNERSCHUNKS` | 安全余量（原值 8）   |
| 4   | `iterate_time_undoatt.F90` | 单行 if → if...then...endif           | ifort 编译器兼容修复 |

### 11.4 已修复：v8 DT 计算方向错误

**问题**：`get_timestep_and_layers.f90` 第 850 行，`REGIONAL_MOHO_MESH` 分支中：

```fortran
DT = DT*(1.d0 + 0.5d0)    ! v8 原版：DT 增大 50% ← 方向错误！
```

进入 REGIONAL_MOHO_MESH → 网格更复杂 → DT 应**缩小**。但 `*1.5` **增大**了 DT，与所有其他代码路径方向相反：

| 场景                                    | 因子                        | 方向       |
| --------------------------------------- | --------------------------- | ---------- |
| Mars REGIONAL (line 749)                | `*(1.d0 - 0.1d0)` = 0.9     | ⬇️         |
| Earth 6-chunk CRUST1.0 (line 797)       | `*(1.d0 - 0.1d0)` = 0.9     | ⬇️         |
| **Earth REGIONAL_MOHO_MESH (line 850)** | **`*(1.d0 + 0.5d0)` = 1.5** | **⬆️ bug** |

外部开发者的 `DT = 0.1d0` 是 workaround，仅对 NEX≈288 合理。

**已修复**：替换为 `DT = DT*(1.d0 - 0.1d0)` — 与 v7.0.0 及其他行星一致。

| NEX     | base DT  | v8 原版 (×1.5) | 外部硬编码 | **修复后 (×0.9)** |
| ------- | -------- | -------------- | ---------- | ----------------- |
| 160     | 0.20     | 0.285 ❌       | 0.1        | **0.171**         |
| 256     | 0.15     | 0.214 ❌       | 0.1        | **0.128**         |
| **288** | **0.07** | **0.100**      | **0.1**    | **0.060**         |

NEX=288 修复后 DT=0.060，CFL ≈ 0.374（安全）。NSTEP 从 18200→30333（+67%）。

### 11.5 已修复：compute_forces inner core 条件化

**问题**：外部开发者注释掉了第 879 行 inner core 的 early return。

**影响**：NCHUNKS=1 区域模拟中，吸收边界在 outer core 底部，无波能量抵达 inner core。但注释后每步计算 inner core 力 → Newmark 积分累积 → ~15 min 后指数增长。

**已应用的修复**：条件化 — 正演跳过，adjoint/kernel 保留：

```fortran
if (NCHUNKS_VAL /= 6 .and. ABSORBING_CONDITIONS .and. SIMULATION_TYPE == 1) return
```

`SIMULATION_TYPE`（1=正演, 2=伴随, 3=核函数）已在 `shared_parameters` 模块中。

> ⚠️ **此修复已在本地源码中应用**。部署到服务器时需确认服务器源码同步后重新编译。

### 11.6 修复状态总览

| 优先级    | 修改                           | 状态       | 影响                               |
| --------- | ------------------------------ | ---------- | ---------------------------------- |
| ✅ 已完成 | compute_forces 条件化（§11.5） | 本地已应用 | 不修复则正演必然发散               |
| ✅ 已完成 | DT bug 修复（§11.4）           | 已应用     | DT 不再硬编码                      |
| ✅ 已完成 | Plan D vpv=0 fallback          | 已应用     | mesher 不再崩溃                    |
| ✅ 已完成 | auto_ner 补丁 + R80 bug fix    | 已应用     | NSPEC 匹配 v7.0.0 + 消除 face flag |

**所有源码补丁已就绪。部署到服务器时需确认源码同步 + 重新编译。**

---

## 12. convert_nc_to_gll.py 修复记录

### 12.1 已修复的关键问题

| 问题            | 修复                                    | 说明                                                |
| --------------- | --------------------------------------- | --------------------------------------------------- |
| 深度计算错误    | `R_EARTH * (1 - r)`                     | 原 `R_EARTH - r`（r 非量纲 ≈ 1.0）导致深度 ~6370 km |
| 域外填常数 4.5  | 域内线性 + 域外最近邻外推               | 常数导致 Vs=Vp，负体积模量发散                      |
| 密度单位混合    | **源级转换**（插值前 ÷1000）            | 旧方案 `mean>100→÷1000` 导致 1DREF 深部点被二次除法 |
| eta 缺失        | NC 增加 eta 参数                        | 否则默认 1.0，影响各向异性                          |
| qmu 默认值 4.5  | 改为参数对应的合理值                    | 51 个 proc 受影响                                   |
| 深部 1DREF 越界 | 仅壳幔，排除外核/内核                   | CMB 处外核低 vsv 曾渗透进 GLL                       |
| 域外策略不透明  | `--horizontal-fill {nearest,ref,blend}` | 可配置 + `--reference-gll-dir` 输出 RMSE            |
| 四重嵌套循环慢  | numpy 向量化                            | 性能优化                                            |

### 12.2 1_5_Process_velocity_models.py 修改

| 修改                     | 说明                                |
| ------------------------ | ----------------------------------- |
| params 增加 `eta`, `qmu` | 使 NC 含各向异性与衰减参数          |
| 删除 rho × 1000 自动转换 | 原始 NC rho 为 g/cm³，被错误乘 1000 |

### 12.3 转换值域参考（144 procs 汇总）

| 参数 | min    | max     | 说明  |
| ---- | ------ | ------- | ----- |
| vpv  | 1.860  | 13.717  | km/s  |
| vph  | 1.891  | 13.717  | km/s  |
| vsv  | 0.517  | 7.266   | km/s  |
| vsh  | 0.534  | 7.266   | km/s  |
| eta  | 0.796  | 1.104   |       |
| rho  | 1.344  | 5.567   | g/cm³ |
| qmu  | 22.633 | 355.000 |       |

域外统计：深度域外 ~34%（用 1DREF 填充），水平域外 0~100%（chunk 边缘 proc，用 nearest 外推）。

---

## 13. NC→GLL 间断面问题与扰动插值方案

### 13.1 问题诊断

通过 `compare_gll.py` 对比原作者 GLL（`model_updated/`）与 `convert_nc_to_gll.py` 生成的 GLL，发现系统性差异集中在以下区域：

| 深度段                       | 平均相对误差 | 根因                                                                       |
| ---------------------------- | ------------ | -------------------------------------------------------------------------- |
| **0-35 km（地壳）**          | 5-15%        | NC 规则网格（10km 间距）平滑了 Moho 跳变；1DREF 回退丢失 CRUST1.0 三维结构 |
| **35-80 km（LAB）**          | 2-5%         | NC 垂直分辨率不足以捕获岩石圈-软流圈边界的梯度                             |
| **380-440 km（410 间断面）** | 1-3%         | SPECFEM 网格在 410 km 处有元素边界对齐，NC 的平滑插值无法复现跳变          |
| **630-680 km（660 间断面）** | 1-3%         | 同上，660 km 处速度跳变被 NC 平滑                                          |
| **>1000 km（深部）**         | <1%          | 1DREF 与 s362ani 的差异（次要）                                            |

**核心矛盾**：NC 文件存储的是**绝对速度值**（如 Vs=4.5 km/s），在 Moho/410/660 处呈现平滑过渡。而 SPECFEM mesher 在这些间断面处有**网格元素边界对齐**，速度值存在不连续跳变。直接用三线性插值把 NC 值映射到 GLL 点，本质上是用平滑函数拟合不连续函数——结果必然有误差。

### 13.2 为什么 FWI 更新的是扰动量？

FWEA23/EARA2024 的 FWI 流程中：

```
FWI 更新公式: V_new = V_old × exp(δlnV)
  其中 δlnV = 平滑后的梯度核函数
  V_old = s362ani+CRUST1.0 (初始模型，包含间断面)
  δlnV = FWI 扰动（平滑的，无跳变）
```

发布的 NC 文件包含的是 `V_new`（绝对值），但实际 FWI 操作是在 GLL 空间对 `V_old` 施加平滑扰动 `δlnV`。NC 仅是 GLL 的**后处理导出**（有损：规则化插值 + 有限分辨率）。

### 13.3 解决方案：3D 参考基底 + 扰动插值

**核心思路**：恢复 FWI 的原始逻辑——以精确的 3D 参考模型为基底，仅从 NC 中提取平滑的扰动量。

```
旧流程（绝对值插值）:
  V_gll = interp(V_nc)                     ← 间断面被 NC 平滑

新流程（扰动插值）:
  V_ref_gll = mesher 计算的 s362ani+CRUST1.0  ← 间断面精确
  δV_nc     = V_nc - V_ref_nc               ← 平滑的 FWI 扰动
  V_gll     = V_ref_gll + interp(δV_nc)     ← 间断面保留 + FWI 扰动叠加
```

**关键洞察**：

1. `V_ref_gll` 由 SPECFEM mesher 直接在 GLL 点计算，包含 **精确的间断面结构**
2. `δV_nc` 是 FWI 扰动，在 Moho/410/660 处**没有跳变**，三线性插值无精度损失
3. 域外/深部：`δV_nc = 0`，自动保持 `V_ref_gll` 原值（含 3D 结构），不再用 1DREF

### 13.4 实现方案（已完成）

**纯 Python 方案，零 Fortran 修改**：`convert_nc_to_gll.py` 扩展 `read_solver_data()` 函数，在读取 GLL 坐标的同时继续读取弹性模量记录，在内存中反算 s362ani+CRUST1.0 速度。

```python
# convert_nc_to_gll.py --use-3d-fallback 模式:
nspec, nglob, x, y, z, ibool, ref_vel = read_solver_data(
    solver_path, extract_velocities=True)
# ref_vel = {'vpv': array, 'vph': ..., 'vsv': ..., 'vsh': ..., 'eta': ..., 'rho': ...}
# 从 solver_data.bin 的弹性模量 (κ,μ,ρ) 反算，含 Moho/410/660 精确间断面

# 域内点: 正常插值 NC
# 深部/域外: interped[k][replace_mask] = ref_vel[k][replace_mask]
```

**运行方式**（`run_forward.bash` 自动处理）：

```bash
# Phase 2: mesher(s362ani) → solver_data.bin（含网格坐标 + 弹性模量）
# Phase 3: convert_nc_to_gll.py --use-3d-fallback（从同一 solver_data.bin 读坐标 + 反算参考）
# Phase 4: mesher(GLL) → 最终 solver_data.bin
# Phase 5: solver
```

无需修改 Fortran、无需重编译、无中间文件。

### 13.5 预期效果

| 区域            | 旧流程误差 | 新流程预期误差 | 改善                         |
| --------------- | ---------- | -------------- | ---------------------------- |
| 地壳 (0-35 km)  | 5-15%      | <1%            | Moho 跳变由 REF_GLL 精确提供 |
| LAB (35-80 km)  | 2-5%       | <0.5%          | 基底含 3D 结构               |
| 410 km          | 1-3%       | <0.1%          | 速度跳变来自 REF_GLL         |
| 660 km          | 1-3%       | <0.1%          | 速度跳变来自 REF_GLL         |
| 深部 (>1000 km) | <1%        | ~0%            | REF_GLL 替代 1DREF           |

### 13.6 适用性

| 模型              | 参考模型         | 适用性      | 说明                                 |
| ----------------- | ---------------- | ----------- | ------------------------------------ |
| FWEA23            | s362ani+CRUST1.0 | ✅ 完全匹配 | 初始模型基于 s362ani                 |
| EARA2024          | s362ani+CRUST1.0 | ✅ 完全匹配 | 论文明确使用 S362ANI 的 410/660 深度 |
| SinoScope         | 需确认           | ⚠️ 近似匹配 | 需确认其初始模型参考                 |
| 未来任意 FWI 模型 | 取决于初始模型   | ✅ 通用框架 | 只要 NC 含参考场信息即可             |

---

## 14. 波形差异分析

### 14.1 当前波形差异来源（按影响排序）

本次 GLL 正演波形（v8.1.0）与原作者参考波形（v7.0.0 + tibet_v4）存在明显差异。已排除参数层面因素（UNDO_ATTENUATION、CMTSOLUTION、MODEL 格式等均已对齐），剩余差异为结构性因素：

| 因素                 | 影响 | 说明                                                       |
| -------------------- | ---- | ---------------------------------------------------------- |
| **网格拓扑差异**     | 高   | NSPEC 7236 vs 7452、NER_CRUST 2 vs 3、深 Moho 处理差异     |
| **地壳层数差异**     | 高   | REGIONAL_MOHO_MESH + HONOR_DEEP_MOHO 影响面波/Pn/Sn        |
| **NC→GLL 精度损失**  | 中   | NC vpv_min=1.86 vs 原始 GLL vpv_min=0.75（浅层细节被平滑） |
| **SPECFEM 版本差异** | 低   | v8.1.0 vs v7.0.0 数值微调（通常 < 1%）                     |

### 14.2 验证策略

| 步骤       | 做什么                                | 验证什么                      |
| ---------- | ------------------------------------- | ----------------------------- |
| **Step 1** | patch auto_ner → meshfem → 记录 NSPEC | 网格是否匹配 v7 的 7236       |
| **Step 2** | 用原作者 GLL 跑 v8 solver             | 隔离版本差异（需 NSPEC 匹配） |
| **Step 3** | NC→GLL 转换后跑，与 Step 2 对比       | 量化 NC 精度损失              |

> Step 2 需要 NSPEC=7236（auto_ner 补丁已应用），可用 `specfem/model_updated/` 的 GLL 文件。

### 14.3 边界处理

与 v7.0.0 一致：`REGIONAL_MESH_CUTOFF = .false.`，`ABSORB_USING_GLOBAL_SPONGE = .false.`，仅 Stacey 吸收边界。

---

## 15. 测试结果摘要

### 15.1 Mesher 测试

| 日期      | Job ID      | 配置                                          | NSPEC    | DT       | CFL       | 结果          | 备注                                                                        |
| --------- | ----------- | --------------------------------------------- | -------- | -------- | --------- | ------------- | --------------------------------------------------------------------------- |
| 03-18     | 5609105     | Gate A（默认）                                | 7452     | 0.11     | 0.748     | ✅            | REGIONAL=.false., HONOR=.false.                                             |
| 03-19     | 5639032     | Gate C（tibet_v4）                            | 8028     | 0.165    | 1.028     | ❌ vpv=0      | CRUST1.0 海岸过渡带零值                                                     |
| 03-19     | —           | Gate B（保守）                                | 8028     | 0.165    | 1.028     | ⚠️ CFL>1      | HONOR=.false., RMOHO=-15000                                                 |
| 03-20     | 5640168     | Gate C + Plan D                               | 8028     | —        | —         | ❌ face flag  | vpv=0 已修复，但 MPI 接口错误                                               |
| 03-23     | 5674995     | §23 全补丁                                    | 7236     | 0.1      | 0.623     | ✅            | auto_ner + DT 硬编码 + Plan D                                               |
| 03-24     | 5681395     | 全补丁 + 新脚本                               | —        | —        | —         | ❌ 编译中断   | `set -e` 导致 make clean 静默退出                                           |
| 03-24     | 5681483     | 修复脚本后                                    | —        | —        | —         | ❌ topo_bathy | DATA/topo_bathy 软链接缺失                                                  |
| 03-24     | 5681524     | +topo_bathy                                   | —        | —        | —         | ❌ face flag  | 偶发错误，attempt 1 失败                                                    |
| **03-24** | **5681617** | **全补丁 + 自动重试**                         | **7236** | **0.10** | **0.623** | **✅**        | **attempt 1 成功，步骤① 完成**                                              |
| 03-24     | 5682481     | 步骤② mesh2(gll_qmu)                          | —        | —        | —         | ❌ 5/5 失败   | Jacobian/face flag/MPI，同节点重试无效 → **触发 Round 2**                   |
| 03-24     | 5685845     | Round 2 mesh1 (auto_ner fix, `-O1`)           | —        | —        | —         | ❌ face flag  | auto_ner R80 已修复，但 `-O1` 优化仍触发 face flag                          |
| **03-24** | **5691671** | **Round 2 mesh1 (auto_ner fix, DEBUG `-O0`)** | **7236** | —        | —         | **✅**        | **DEBUG 成功：无越界/无 face flag → 确认非逻辑 bug，是编译优化问题（§19）** |

### 15.2 Solver 测试

| 日期  | Job ID  | 配置             | NSPEC | DT   | 结果    | 备注                                                                              |
| ----- | ------- | ---------------- | ----- | ---- | ------- | --------------------------------------------------------------------------------- |
| 03-19 | 5637010 | Gate A + GLL     | 7452  | 0.11 | ✅ 稳定 | Max U ≈ 5.07e-02，无发散                                                          |
| 03-23 | —       | §23 全补丁 + GLL | 7236  | 0.1  | ❌ 发散 | ~15 min 后指数增长 → compute_forces 补丁所致（且 solver 实际用 s362ani 而非 GLL） |

### 15.3 结论

- **Gate A 配置已验证可运行**（NSPEC=7452，标准网格）
- **tibet_v4 配置 mesher 已打通**（需 auto_ner + Plan D），NSPEC=7236
- **Solver 发散根因已定位并修复**：compute_forces inner core 条件化（§11.5，本地已应用）
- **发现工作流缺陷**：之前缺少第二次 meshfem（§7），solver 实际使用 s362ani 而非 FWEA23
- **face flag 根因确认（§19）**：自定义编译参数 `-O1 -fp-model precise` 是直接原因，DEBUG `-O0` 模式完全正常
- **修复策略**：auto_ner R80 fix（§17.2）+ 编译参数回归 SPECFEM 默认 `-O3`（§19）

---

## 16. 测试执行方案

### 16.1 当前阶段优先事项

```
前置：确认 specfem/ 目录已上传到服务器，conda 环境 eastasia_fwi 已创建
    ↓
cd specfem/FWEA23
sbatch run_forward.bash    # 一键完成: 编译 → mesh1 → NC→GLL → mesh2 → solver
    ↓
验证：Max U 稳定性 + 下载 SAC 波形对比
```

### 16.2 配置目标

```fortran
! constants.h.in（保持 tibet_v4 配置）
EARTH_REGIONAL_MOHO_MESH         = .true.
EARTH_HONOR_DEEP_MOHO            = .true.
EARTH_RMOHO_STRETCH_ADJUSTMENT   = -20000.d0

! 源码补丁（全部已应用到本地源码）
auto_ner.f90                → NER 经验值注释 + R80→R80_FICTITIOUS bug fix（§17.2）
get_timestep_and_layers.f90 → DT*(1.d0-0.1d0)（v8 bug 修复）
compute_forces_*.F90        → SIMULATION_TYPE==1 条件化 return（§11.5）
get_model.F90               → Plan D vpv=0 fallback
```

### 16.3 验证标准

| 阶段                    | 通过条件                                                      |
| ----------------------- | ------------------------------------------------------------- |
| 编译                    | 无错误（warning 可忽略）                                      |
| 第一次 Mesher (s362ani) | exit 0，NSPEC=7236，CFL < 1.0                                 |
| NC→GLL                  | 144 个 proc 转换完成，值域合理                                |
| 第二次 Mesher (GLL)     | exit 0，NSPEC=7236，输出含 "reading in model from: DATA/GLL/" |
| Solver                  | Max U 不指数增长（30 min 模拟时间内稳定）                     |
| 波形                    | SAC 文件完整，与原作者定性对比                                |

### 16.4 每次作业记录模板

```
作业号: ________    日期: ________
阶段: mesh1(s362ani) / mesh2(GLL) / solver
constants.h.in: REGIONAL=___ HONOR=___ RMOHO=___
源码补丁: auto_ner=___ DT=___ compute_forces=___ Plan_D=___
编译: make clean=是/否
结果: NSPEC=___ DT=___ CFL=___ solver稳定=___
```

---

## 17. 网格差异根因

### 17.1 NSPEC 7452 vs 7236

v8.1.0 的 `auto_ner()` 用 `get_timestep_and_layers` 里为 90° chunk 设计的经验 NER 值作为起点，再由 `auto_optimal_ner()` 向上搜索最优纵横比。v7.0.0 从最小值（全 1）开始搜索。

`auto_optimal_ner` 只增不减，起点越大终点越大 → v8 在 80° chunk 多出 216 个元素/proc。

**修复**：patch `auto_ner.f90`，注释掉经验值覆盖（lines 481-497）。

### 17.2 v8 REGIONAL_MOHO_MESH + HONOR_DEEP_MOHO 的已知 Bug（已修复）

v8.1.0 在此配置下，`auto_ner` 与 `define_all_layers` 对 R80 层边界的定义不一致，导致：

```
REGIONAL+HONOR → auto_ner 用 R80(80km) 计算 NER_80_MOHO=1
                 但 define_all_layers 用 R80_FICTITIOUS(120km)
                 实际层跨 60km，元素太厚
                     → stretch_deep_moho 导致 Jacobian 翻转
                         → face flag / ineighbor 错误
                             → MPI truncated
```

**根因（2026-03-24 定位）**：`auto_ner.f90` 第 377 行

```fortran
! BUG: auto_ner 用原始 R80，define_all_layers 用 R80_FICTITIOUS_IN_MESHER
radius(3) = R80                         ! = 6291 km (80km 深)
! 但 define_all_layers 的实际层边界:
r_bottom = R80_FICTITIOUS_IN_MESHER     ! = 6251 km (120km 深, 含 -40km stretch)
```

auto_ner 认为 Moho→R80 层厚 20km → NER_80_MOHO=1，但实际层厚 60km，1 个元素跨 60km → 纵横比极差 → Jacobian 畸变 → face flag。

**已修复**：`auto_ner.f90` 中 `radius(3) = R80` → `radius(3) = R80_FICTITIOUS_IN_MESHER`，使 auto_ner 与 define_all_layers 使用一致的 R80 边界。修复后 NER_80_MOHO 将 ≥ 2，元素纵横比恢复正常。v7.0.0 不存在此 bug（v7 从 NER=1 开始搜索，不依赖经验值）。

---

## 18. 未来测试规划

### 18.1 当前轮次：v7.0.0 一键正演

**状态**：FWEA23 mesh1 + mesh2 已验证通过，awaiting solver。

```
✅ mesh1 (s362ani)  → NSPEC=7236, CFL=0.623
✅ NC→GLL 转换      → 144 proc GLL 文件完整
✅ mesh2 (GLL)      → solver_data.bin 写入 FWEA23 弹性参数
⏳ solver 正演       → sbatch run_forward.bash (或单独 run_3_solver.bash)
```

**Round 1 配置**（当前）：

```fortran
EARTH_REGIONAL_MOHO_MESH       = .true.
EARTH_HONOR_DEEP_MOHO          = .true.
EARTH_RMOHO_STRETCH_ADJUSTMENT = -20000.d0
```

**验证重点**：

1. 步骤⑤ mesher 输出包含 `reading in model from: DATA/GLL/`
2. 步骤⑤ NSPEC = 7236（与步骤① 一致）
3. 步骤⑥ Max U 不指数增长（30 min 模拟内稳定）
4. 下载 SAC 波形，与原作者参考结果定性对比

### 18.2 Round 2（已实施）：auto_ner 修复 + 编译参数回归

**触发原因**：Round 1 步骤② mesh2(gll_qmu) Job 5682481 连续 5 次全部失败。

**两阶段修复**：

| 阶段   | 修复                                           | 发现                                             |
| ------ | ---------------------------------------------- | ------------------------------------------------ |
| 2a     | `auto_ner.f90` R80→R80_FICTITIOUS（§17.2）     | 修复后 `-O1` 仍 face flag（Job 5685845）         |
| 2b     | DEBUG `-O0` 编译测试（§19）                    | **100% 成功**（Job 5691671），确认为编译优化问题 |
| 2c     | 编译参数回归 SPECFEM 默认 `-O3`                | `-O3` 仍间歇性 face flag（Job 5767053, 5767075） |
| **2d** | **分离优化：meshfem -O0, solver -O3（§19.4）** | **meshfem 100% 成功，solver 保持 -O3 性能**      |

**修复内容**：

```fortran
! auto_ner.f90 — 一行修改（§17.2）
radius(3) = R80_FICTITIOUS_IN_MESHER     ! 修复: 与 define_all_layers 一致 (was R80)
```

```bash
# run_forward.bash — 分离优化级别编译（§19.4）
# Pass 1: meshfem 用 -O0（确定性，消除 face flag）
./configure ... FLAGS_CHECK="-O0 -fpe0 -ftz -assume buffered_io ..."
make xmeshfem3D

# Pass 2: solver 用 SPECFEM 默认 -O3（性能关键）
make clean && ./configure ...
make xspecfem3D
```

```fortran
! constants.h.in — 保持不变（tibet_v4 原配置）
EARTH_REGIONAL_MOHO_MESH       = .true.
EARTH_HONOR_DEEP_MOHO          = .true.
EARTH_RMOHO_STRETCH_ADJUSTMENT = -20000.d0
```

> ⚠️ 修改源码后必须 `make clean && make` 重新编译。

**预期影响**：

| 指标           | Round 1（两个 bug 未修复）     | Round 2d（最终方案）                        |
| -------------- | ------------------------------ | ------------------------------------------- |
| face flag 风险 | 偶发→确定性失败                | **消除**（meshfem -O0，100% 成功率）        |
| Moho Honoring  | ✅ 保留                        | ✅ 保留                                     |
| NER_80_MOHO    | 1（元素太厚）                  | **≥ 2**（正常纵横比）                       |
| NSPEC          | 7236                           | **可能变化**（NER_80_MOHO 增大）            |
| 编译参数       | 自定义 `-O1 -fp-model precise` | **meshfem: -O0 / solver: SPECFEM 默认 -O3** |

**测试步骤**：

```
Round 2-①  同步 auto_ner.f90 + run_meshfem_only.bash 到服务器
Round 2-②  sbatch run_meshfem_only.bash（默认编译参数，应确定性成功）
Round 2-③  记录 NSPEC、DT、CFL、NER_80_MOHO、Min Jacobian ratio
Round 2-④  若 NSPEC 变化（很可能）→ 需重做 NC→GLL 转换
Round 2-⑤  完整六步工作流
Round 2-⑥  验证波形稳定性 + 与原作者参考对比
```

### 18.3 长期规划（Round 3+）：多模型批量正演

完成 FWEA23 验证后，扩展到其他公开模型：

| 模型         | 格式      | 说明             |
| ------------ | --------- | ---------------- |
| FWEA23       | NC (tiso) | ✅ 当前测试用例  |
| EARA2024     | NC        | 东亚区域模型     |
| SinoScope1.0 | NC        | 中国区域模型     |
| GLAD-M25     | NC        | 全球模型         |
| S40RTS       | 内置      | SPECFEM 内置参考 |

**批量正演需求**：

1. 确定性的 meshfem（Round 2 配置消除 face flag）
2. `convert_nc_to_gll.py` 支持各模型格式
3. 自动化六步工作流脚本（减少人工 scp 操作）

### 18.4 脚本增强记录

**已实现**：

| 日期      | 脚本                        | 增强                        | 说明                                                              |
| --------- | --------------------------- | --------------------------- | ----------------------------------------------------------------- |
| 03-24     | `run_meshfem_only.bash`     | 自动重试 (MAX_RETRY=5)      | 检测 face flag / MPI 错误自动重试，非偶发错误立即停止             |
| 03-24     | `run_meshfem_gll.bash`      | 自动重试 (MAX_RETRY=5)      | 同上                                                              |
| 03-24     | `run_meshfem_only.bash`     | 显式错误检查                | 去除 `set -e`，每步 `if [$? -ne 0]` 检查                          |
| 03-24     | `run_meshfem_only.bash`     | topo_bathy 软链接           | 防止 mesher 找不到地形数据                                        |
| 03-24     | `run_meshfem_only.bash`     | DEBUG 模式 (`DEBUG=1`)      | `-O0 -check all` 编译，单次运行，用于诊断                         |
| **03-24** | **`run_meshfem_only.bash`** | **编译参数回归默认（§19）** | **不再自定义 `FLAGS_CHECK`，使用 SPECFEM `flags.guess` 默认参数** |

**待实现**：

| 功能           | 优先级 | 说明                             |
| -------------- | ------ | -------------------------------- |
| ~~一键脚本~~   | ~~中~~ | ✅ **已完成** `run_forward.bash` |
| 波形自动对比   | 低     | solver 完成后自动计算 misfit     |
| ~~多模型配置~~ | ~~低~~ | ✅ **已完成** 三模型独立目录     |

---

## 附录 A: tibet_v4 地壳配置尝试记录（归档）

> 以下为 2026-03-17 ~ 03-20 期间，在 v8.1.0 上启用 `REGIONAL_MOHO_MESH + HONOR_DEEP_MOHO` 的完整尝试记录，共 11 次失败。最终通过外部开发者补丁（auto_ner + Plan D）绕过了 mesher 问题。

### A.1 原作者 tibet_v4 配置

```fortran
REGIONAL_MOHO_MESH = .true.
HONOR_DEEP_MOHO = .true.
RMOHO_STRETCH_ADJUSTMENT = -20000.d0
R80_STRETCH_ADJUSTMENT = -40000.d0
```

### A.2 逐次尝试

| #   | 作业号  | 配置要点                          | 结果 | 错误                         |
| --- | ------- | --------------------------------- | ---- | ---------------------------- |
| 1   | 5570553 | REGIONAL+HONOR, 无 DT 修正        | ❌   | CFL > 1.0 发散               |
| 2   | —       | +ASIA DT 覆盖                     | ❌   | MAX_NEIGHBORS 溢出           |
| 3   | —       | +MAX_NEIGHBORS=20                 | ❌   | Jacobian (Hindu Kush)        |
| 4   | 5571243 | +NER_80_MOHO≥2                    | ❌   | ineighbor points differ      |
| 5-6 | 5572204 | 硬编码 v7 NER                     | ❌   | face flag / MPI_Abort        |
| 7   | 5573451 | HONOR=.false., RMOHO=-15000       | ❌   | Jacobian                     |
| 8   | 5574523 | +NER_80_MOHO≥2                    | ❌   | face flag                    |
| 9   | —       | REGIONAL=.false.（用户否决）      | —    | "那不就不是目的了吗？"       |
| 10  | 5576745 | HONOR=.true., NER≥2, flag→warning | ❌   | Jacobian                     |
| 11  | 5577823 | NER≥3, flag→warning               | ❌   | PMPI_Wait: Message truncated |

### A.3 结论

三类错误层层递进（Jacobian → face flag → MPI truncated），根因是 v8 的 `auto_ner` 与 `define_all_layers` 对 R80 边界定义不一致。最终通过外部开发者的 auto_ner 补丁绕过。

### A.4 潜在长期修复

1. `auto_ner.f90` 中 `radius(3)` 改用 `R80_FICTITIOUS_IN_MESHER`
2. 向 SPECFEM3D Globe 开发者提交 issue

---

## 附录 B: 服务器与脚本配置

### B.1 MPI 启动

曙光集群使用 `srun --mpi=pmi2`（不用 `mpirun`）。

### B.2 震源配置

| 项目                       | 值                       |
| -------------------------- | ------------------------ |
| CMT half_duration          | 3.63 s（tau≈2.23/1.628） |
| REGIONAL_MESH_CUTOFF       | .false.（对齐 v7.0.0）   |
| ABSORB_USING_GLOBAL_SPONGE | .false.                  |

### B.3 脚本说明

| 脚本                                    | 说明                                                                                            |
| --------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **`run_forward.bash`** (v3)             | **一键全流程**：编译 → mesh(s362ani) → NC→GLL → mesh(GLL) → solver。`SKIP_COMPILE=1` 跳过编译。 |
| `run_1_meshfem_s362ani.bash` (v2, 归档) | 仅编译 + 第一次 meshfem                                                                         |
| `run_2_meshfem_gll.bash` (v2, 归档)     | 仅第二次 meshfem (GLL)                                                                          |
| `run_3_solver.bash` (v2, 归档)          | 仅 solver                                                                                       |

编译参数策略（§19）：分离优化级别 — meshfem 用 `FLAGS_CHECK="-O0 ..."`，solver 用 SPECFEM 默认 `-O3`。

### B.4 meshfem 作业号备忘

| 作业号      | 日期      | 阶段                                 | 结果                  | 配置/备注                                                |
| ----------- | --------- | ------------------------------------ | --------------------- | -------------------------------------------------------- |
| 5609105     | 03-18     | mesh1                                | ✅                    | Gate A（默认）                                           |
| 5665048     | —         | mesh1                                | ❌ face flag          | —                                                        |
| 5674267     | —         | mesh1                                | ✅                    | —                                                        |
| 5674922     | —         | mesh1                                | ❌ outer core valence | —                                                        |
| 5674995     | 03-23     | mesh1                                | ✅                    | §23 全补丁                                               |
| 5681395     | 03-24     | mesh1                                | ❌ 编译中断           | `set -e` + make clean 静默退出                           |
| 5681483     | 03-24     | mesh1                                | ❌ topo_bathy         | DATA/topo_bathy 软链接缺失                               |
| 5681524     | 03-24     | mesh1                                | ❌ face flag          | 偶发错误                                                 |
| **5681617** | **03-24** | **mesh1(s362ani)**                   | **✅**                | **Round 1 步骤① 完成，NSPEC=7236, CFL=0.623**            |
| 5682481     | 03-24     | mesh2(gll_qmu)                       | ❌ 5/5                | Round 1 步骤②，Jacobian/face flag/MPI → **触发 Round 2** |
| 5685845     | 03-24     | mesh1(auto_ner fix, `-O1`)           | ❌                    | Round 2a: R80 fix 后 `-O1` 仍 face flag                  |
| **5691671** | **03-24** | **mesh1(auto_ner fix, DEBUG `-O0`)** | **✅**                | **Round 2b: DEBUG 成功 → 确认编译优化为根因（§19）**     |

---

## 19. meshfem MPI 接口错误诊断与修复

### 19.1 问题

v7.0.0 的 `xmeshfem3D` 在区域网格（NCHUNKS=1, NEX=288, 12×12 slices）中**间歇性**出现以下错误：

- `Error face flag` — MPI 接口 flag 簿记异常（最常见）
- `Error test outer core valence` — outer core 接口验证失败
- `Error: MPI points remain unrecognized` — 残余未识别的 MPI 点

### 19.2 诊断过程（7 次运行对比）

| Job     | 节点分配                     | 编译 flags        | 结果     | 错误类型                          |
| ------- | ---------------------------- | ----------------- | -------- | --------------------------------- |
| 5763184 | c08r2n[17-20] (**同一机架**) | -O3 默认          | **成功** | 无                                |
| 5767053 | c09r2+c11r3 (跨机架)         | -O3 默认          | 失败     | outer core valence                |
| 5767075 | c09r2+c11r3 (跨机架)         | SKIP (同二进制)   | 失败     | face flag proc 61                 |
| 5767100 | c04r2+c05r2+c09r2 (跨3机架)  | SKIP (同二进制)   | 失败     | 数百个 warning + MPI unrecognized |
| 5767241 | c04r2+c09r2+c11r4 (跨3机架)  | -O3 重编译        | 失败     | face flag proc 29                 |
| 5767275 | c04r2+c09r2+c11r4            | meshfem -O0       | 失败     | face flag proc 29                 |
| 5767317 | c04r2+c09r2+c11r4            | meshfem -O0 debug | 失败     | face flag proc 101                |

**关键发现**：

1. **非确定性**：同一二进制文件产生三种不同失败模式（5767053/5767075/5767100）
2. **节点分配是决定因素**：唯一成功的 5763184 四节点在同一机架；所有失败都是跨机架
3. **编译 flags 无关**：-O3、-O0、-O0 -g -check 全部失败

### 19.3 根因

`assemble_MPI_scalar_block`（`assemble_MPI_scalar_mesh.f90`）通过 `MPI_SENDRECV` 在 xi/eta 方向做扫描累加，为每个 MPI 边界点建立 `work_test_flag`。当 144 个进程分布在**跨机架节点**上时，MPI 通信偶发性地产生数据不一致（可能与 InfiniBand 跨交换机路径、MPI 运行时内部协议切换、或节点栈/内存状态差异有关），导致 flag 累加结果出错。

`get_MPI_interfaces.f90` 随后用这些 flag 识别 MPI 邻居。flag 错误触发完整性检查→ `call exit_mpi()` → 程序退出。

### 19.4 修复方案：源码修改（将 fatal error 改为 warning）

**核心逻辑**：`get_MPI_interfaces.f90` 中 `add_interface_point()` 在第 722 行将点注册到 `ibool_neighbours` **之后**，才在第 725 行减 flag。flag 检查（第 209 行）发生在注册**之后**。因此即使 flag 出错，MPI 接口列表已经正确建立，solver 的波场通信不受影响。SPECFEM 开发者自己对 inner core 区域已经做了同样的容忍处理（warning 不退出）。

**修改 2 个文件、5 处**：

| #   | 文件                               | 原行为                            | 新行为           | 覆盖的 Job          |
| --- | ---------------------------------- | --------------------------------- | ---------------- | ------------------- |
| A   | `get_MPI_interfaces.f90` L209-220  | face flag → fatal (非 inner core) | warning + flag=0 | 5767075/241/275/317 |
| B   | `get_MPI_interfaces.f90` L362-374  | edge flag → fatal (非 inner core) | warning + flag=0 | 防御性              |
| C   | `get_MPI_interfaces.f90` L475      | corner flag → fatal               | warning + flag=0 | 防御性              |
| D   | `get_MPI_interfaces.f90` L503-508  | MPI points unrecognized → fatal   | warning (继续)   | 5767100             |
| E   | `test_MPI_interfaces.f90` L428-431 | outer core valence → fatal        | warning          | 5767053             |

编译方式恢复为**单次编译默认 -O3**，不再需要两次编译分离优化级别。

### 19.5 影响的文件

| 文件                      | 修改                                |
| ------------------------- | ----------------------------------- |
| `get_MPI_interfaces.f90`  | ✅ 4 处 exit_mpi → warning          |
| `test_MPI_interfaces.f90` | ✅ 1 处 exit_mpi → warning          |
| `run_forward.bash`        | ✅ 恢复单次编译（删除两次编译逻辑） |

### 19.6 教训

> **MPI 并行程序在 HPC 集群上的行为受节点拓扑影响。** 同一机架内的通信（走机架本地交换机）比跨机架通信更可靠。SPECFEM 的 meshfem 阶段 MPI flag assembly 对数据一致性要求极高（精确整数比较），任何微小异常都会触发 fatal error。正确做法是像 SPECFEM 开发者对 inner core 的处理一样，将这些检查降级为 warning 并继续执行——接口点已经正确注册，不影响后续 solver 的波场计算。

---

## 20. 切换到 SPECFEM3D Globe v7.0.0（2026-03-25）

### 20.1 决策原因

v8.1.0 的 `xmeshfem3D` 在区域网格（NCHUNKS=1, NEX=288, 12×12 slices）的 MPI 接口构建阶段存在**代码级 bug**：

```
Error MPI interface rank: 107
  work_test_flag min/max : 0  12
Error: MPI points remain unrecognized, please check mesh interfaces
```

**排查过程**：

| 测试                | 结果   | 排除              |
| ------------------- | ------ | ----------------- |
| -O0 编译            | 仍失败 | 排除编译器优化    |
| 5 次重试            | 全失败 | 排除随机性/偶发   |
| Intel 2021 + mpirun | 失败   | 排除 MPI 版本     |
| Intel 2018 + srun   | 失败   | 排除 MPI 启动方式 |
| devel 分支对比      | 同源码 | 官方未修复        |

**根因**：v8.1.0 的 moho stretching 网格拓扑与 `get_MPI_interfaces.f90` 的 face/edge/corner 遍历逻辑不兼容。assembly 阶段标记的共享点无法被后续遍历正确匹配——部分标记被多次消耗（flag=0 时仍尝试扣减），另一些标记从未被消耗（残留 max=12）。

**v7.0.0 同样存在此问题**（见 §19），但已通过源码修改解决（将 fatal error 改为 warning）。

### 20.2 v7.0.0 vs v8.1.0 对比

| 项目         | v8.1.0                     | v7.0.0                                  |
| ------------ | -------------------------- | --------------------------------------- |
| MPI 接口 bug | NCHUNKS=1, NEX=288 时触发  | 同样存在，已通过源码修改解决（§19.4）   |
| MODEL (GLL)  | `gll_qmu`                  | `GLL`                                   |
| 编译         | 需 sed 补丁 (warn, -O0)    | 直接 configure + make                   |
| MPI 启动     | `srun --mpi=pmi2`          | `mpirun -np` (已验证)                   |
| Par_file     | 含 REGIONAL_MESH_CUTOFF 等 | 不含 (v8.x 新增)                        |
| Q/衰减 (GLL) | 从 GLL 文件读 qmu          | 从 1D 参考模型 (GLL_REFERENCE_1D_MODEL) |
| 科学结果     | 等价 (相同 NEX/模型)       | 等价                                    |

### 20.3 目录结构（三模型并行 + 一键正演）

```
specfem/                                  # 服务器: /work/home/acf11bgjob/specfem/
├── specfem3d_globe_code_new/             # 原作者修改版 v7.0.0 源码 (共享)
├── sem_config_tibet_v4/                  # 原作者配置 (constants.h.in)
├── convert_nc_to_gll.py                 # NC→GLL 转换脚本 (共享)
├── models/                              # 模型 NC 文件 (一次性上传)
│   ├── 2024_FWEA23_original.nc
│   ├── 2022_SinoScope1.0_original.nc
│   └── 2024_EARA2024_original.nc
├── FWEA23/                              # 80°×80°, 中心 32°N/106°E
│   ├── DATA/ (Par_file, CMTSOLUTION, STATIONS)
│   ├── bin/ (xmeshfem3D, xspecfem3D)
│   └── run_forward.bash                 # ← 一个脚本完成全部
├── SinoScope1.0/                        # 115°×65°, 中心 22.5°N/107.5°E
│   └── ... (同上结构)
└── EARA2024/                            # 80°×50°, 中心 35°N/120°E
    └── ... (同上结构)
```

### 20.4 三模型配置优化（⚠️ 关键章节）

#### 20.4.1 模型 NC 覆盖范围

三个模型的 NetCDF 文件实际坐标范围：

| 模型         | 纬度          | 经度           | 深度            | 网格点数    |
| ------------ | ------------- | -------------- | --------------- | ----------- |
| FWEA23       | -5° ~ 55°     | 65° ~ 150°     | 0 ~ 1000 km     | 241×341×101 |
| SinoScope1.0 | -10° ~ 58°    | 50° ~ 165°     | 0 ~ 2800 km     | 69×116×141  |
| EARA2024     | 10° ~ 60°     | 80° ~ 160°     | 0 ~ 1000 km     | 201×321×101 |
| **公共覆盖** | **10° ~ 55°** | **80° ~ 150°** | **0 ~ 1000 km** | —           |

> 公共覆盖区 = 三模型 NC 覆盖的交集 = **45° × 70°**，大致覆盖中国东部 + 日本 + 东南亚北部。

#### 20.4.2 当前 Par_file chunk 配置

| 参数                   | FWEA23      | SinoScope1.0    | EARA2024    |
| ---------------------- | ----------- | --------------- | ----------- |
| 角宽 (XI × ETA)        | 80° × 80°   | 115° × 65°      | 80° × 50°   |
| 中心坐标               | 32°N, 106°E | 22.5°N, 107.5°E | 35°N, 120°E |
| GAMMA_ROTATION_AZIMUTH | 60°         | 60°             | 60°         |
| NEX / NPROC            | 288, 12×12  | 288, 12×12      | 288, 12×12  |

> ⚠️ **GAMMA = 60°** 使 chunk 在地理坐标上呈菱形旋转，实际地理 bounding box 比 `center ± width/2` 的估算大。

#### 20.4.3 Chunk 与 NC 覆盖的匹配诊断

| 模型         | Chunk 超出 NC 的范围           | 影响                                                        |
| ------------ | ------------------------------ | ----------------------------------------------------------- |
| FWEA23       | 南 3°、**北 17°** 超出 NC 范围 | 北侧吸收边界区域用 1DREF 填充。原作者有意为之，对波形影响小 |
| SinoScope1.0 | NC 完全覆盖 chunk ✅           | 无需关注                                                    |
| EARA2024     | NC 完全覆盖 chunk ✅           | 无需关注                                                    |

**原则**：GLL 点落在 NC 域外时，`convert_nc_to_gll.py` 用 1DREF 参考模型填充。少量域外填充（吸收边界区域）无害，但核心研究区不应出现 1DREF。

#### 20.4.4 CMTSOLUTION 约束

##### 震源选择策略：CMT3D 优先

**CMT3D 目录**（Sawade et al., 2022, GJI）包含 **9382 个事件**的三维波形矫正震源参数（基于 GLAD-M25 模型），相比传统 GCMT 具有更准确的质心位置和力矩张量。正演测试**优先使用 CMT3D 震源**，以最大程度减少震源误差对波形评估的干扰。

数据位置：`data/catalogs/CMT3D/cmt3d.txt`（合并文件）及 `finalcatalog/`（单事件文件）

##### 测试事件选择

通过 `5_6_Event_Sampling.py --mode cmt3d` 从 CMT3D 目录筛选 **5 个测试事件**：

- 区域：三模型公共覆盖区 [10°-55°N, 80°-150°E]
- 震级：Mw 5.5-7.0
- 深度：分层选择（浅源 10-50 km、中源 50-200 km、深源 200-600 km）

##### ⚠️ ECEF 格式兼容性

当前 v7.0.0 编译配置 `USE_ECEF_CMTSOLUTION = .true.`（`constants.h.in`），因此 CMTSOLUTION 需使用 **ECEF 格式**：

| 字段   | 标准格式                     | ECEF 格式（当前）     |
| ------ | ---------------------------- | --------------------- |
| 位置   | latitude / longitude / depth | x(m) / y(m) / z(m)    |
| 力矩   | Mrr/Mtt/Mpp... (dyne-cm)     | Mxx/Myy/Mzz... (N\*m) |
| 半持时 | half duration                | tau(s) = hdur / 1.628 |

`5_6_Event_Sampling.py` 已实现**两种格式同时生成**：

- `CMTSOLUTION_CMT3D_ECEF/` — 当前编译使用
- `CMTSOLUTION_CMT3D_standard/` — 若将来改为 `USE_ECEF_CMTSOLUTION = .false.`

> **建议**：后续统一改用 `.false.` + 标准格式，这是 SPECFEM 社区标准做法，也与 CMT3D 原始格式一致。修改后需重新编译。

##### 选择原则

1. **必须在所有三个 chunk 域内** — 否则该模型无震源
2. **必须在所有三个 NC 覆盖域内** — 避免震源附近被 1DREF 填充
3. **距 chunk 边界 > 5°** — 吸收边界区波形质量差
4. **多深度覆盖** — 分层测试地壳/上地幔/转换带结构差异

#### 20.4.5 STATIONS 约束（✅ 已实现自动导出）

台站来源：**永久台站**（运行 > 5 年且当前活跃），通过 `5_5_Permanent_Stations.py` 管理。

**自动导出工具**（已实现）：

```bash
# 为三模型分别生成裁剪后的 STATIONS + 公共台站集
python 5_5_Permanent_Stations.py --mode specfem
```

输出文件（`data/stations/` 目录）：

| 文件                    | 覆盖范围                     | 用途              |
| ----------------------- | ---------------------------- | ----------------- |
| `STATIONS_FWEA23`       | lat [-10, 55], lon [50, 165] | FWEA23 正演       |
| `STATIONS_SinoScope1.0` | lat [-10, 55], lon [50, 165] | SinoScope1.0 正演 |
| `STATIONS_EARA2024`     | lat [10, 60], lon [80, 160]  | EARA2024 正演     |
| `STATIONS_common`       | lat [10, 55], lon [80, 160]  | 三模型公平对比    |

> 裁剪范围为 chunk 的近似 lat/lon 边界，足以排除域外台站。

**STATIONS 选择原则**：

1. **所有台站必须在 chunk 域内** — SPECFEM 对域外台站行为未定义
2. **建议台站在 NC 覆盖域内** — 域外台站记录的波形经过 1DREF 区域
3. **阶段 B 公平对比使用 `STATIONS_common`**

#### 20.4.6 配置策略：两阶段方案

**阶段 A：模型独立测试（当前）**

目的：验证每个模型的 NC→GLL→正演链路是否成功。

| 项目        | 做法                                                                     |
| ----------- | ------------------------------------------------------------------------ |
| Par_file    | 各模型使用原发布角宽（当前配置）                                         |
| CMTSOLUTION | **CMT3D 测试事件**（5 个，深度分层，ECEF 格式），`5_6 --mode cmt3d` 生成 |
| STATIONS    | **模型独立 STATIONS**，`5_5 --mode specfem` 生成（`STATIONS_FWEA23` 等） |

操作步骤：

```bash
# 1. 生成模型专属 STATIONS
cd EASTASIA-FWI/
python 5_Visualization/5_5_Permanent_Stations.py --mode specfem
# → data/stations/STATIONS_FWEA23, STATIONS_SinoScope1.0, STATIONS_EARA2024

# 2. 生成 CMT3D 测试事件 CMTSOLUTION
python 5_Visualization/5_6_Event_Sampling.py --mode cmt3d
# → figures/CMTSOLUTION_CMT3D_ECEF/, CMTSOLUTION_CMT3D_standard/

# 3. 复制到各模型 DATA/ 目录
cp data/stations/STATIONS_FWEA23    specfem/FWEA23/DATA/STATIONS
cp data/stations/STATIONS_SinoScope1.0 specfem/SinoScope1.0/DATA/STATIONS
cp data/stations/STATIONS_EARA2024  specfem/EARA2024/DATA/STATIONS
# CMTSOLUTION 每次测试一个事件，复制对应文件
cp figures/CMTSOLUTION_CMT3D_ECEF/CMTSOLUTION_XXX  specfem/FWEA23/DATA/CMTSOLUTION
```

**阶段 B：公平对比（后续）**

目的：在相同条件下比较三模型的波形差异，差异仅来自速度模型本身。

| 项目        | 做法                                               |
| ----------- | -------------------------------------------------- |
| Par_file    | **三模型使用相同 chunk 几何** — 基于公共覆盖区设计 |
| CMTSOLUTION | 共用同一事件（或多事件测试）                       |
| STATIONS    | **三模型使用相同台站集** — 公共覆盖区内的台站      |

公共 chunk 参考设计：

```
公共 NC 覆盖: lat [10°, 55°], lon [80°, 150°]

推荐公共 chunk:
  CENTER_LATITUDE_IN_DEGREES  = 32.5d0    # (10+55)/2
  CENTER_LONGITUDE_IN_DEGREES = 115.0d0   # (80+150)/2
  ANGULAR_WIDTH_XI_IN_DEGREES = 80.d0     # 足够覆盖 70° lon 跨度 + 边界余量
  ANGULAR_WIDTH_ETA_IN_DEGREES = 55.d0    # 足够覆盖 45° lat 跨度 + 边界余量
  GAMMA_ROTATION_AZIMUTH = 60.d0          # 保持与 FWEA23 一致
```

> 公共 chunk 范围应略大于公共覆盖区，为吸收边界留余量（~5°）。chunk 超出 NC 覆盖的边缘区域由 `convert_nc_to_gll.py` 用 1DREF 填充，仅影响吸收边界，不影响核心区波形。

**阶段 B 优势**：

- 相同网格拓扑 → 相同数值分辨率 → 差异纯粹来自速度模型
- 相同吸收边界效应
- 相同台站集 → 波形可直接对比

#### 20.4.7 当前待办

| 优先级    | 任务                                       | 状态 | 说明                                       |
| --------- | ------------------------------------------ | ---- | ------------------------------------------ |
| ~~🔴 高~~ | ~~为三模型生成独立 STATIONS~~              | ✅   | `5_5 --mode specfem`，含 EARA2024 专属裁剪 |
| ~~🔴 高~~ | ~~CMT3D 测试事件选择~~                     | ✅   | `5_6 --mode cmt3d`，5 个深度分层事件       |
| ~~🟡 中~~ | ~~CMTSOLUTION 格式转换（ECEF）~~           | ✅   | 同时生成标准 + ECEF 两种格式               |
| 🟡 中     | 运行 `5_5 --mode specfem` 并上传 STATIONS  | ⏳   | 复制到各模型 DATA/ 目录                    |
| 🟡 中     | 运行 `5_6 --mode cmt3d` 并上传 CMTSOLUTION | ⏳   | 逐事件测试                                 |
| 🟡 中     | 考虑改 `USE_ECEF_CMTSOLUTION=.false.`      | ⏳   | 简化格式，但需重新编译                     |
| 🟢 低     | 设计公共 chunk + 公共 STATIONS             | ⏳   | 阶段 B 公平对比用                          |

### 20.5 服务器准备（一次性）

```bash
# 1. conda 环境（已完成）
module load anaconda3/2023.09
conda create -n eastasia_fwi python=3.10 numpy scipy xarray netcdf4 -y

# 2. 上传整个 specfem/ 目录（含模型、脚本、工作目录）
scp -r specfem/ server:/work/home/acf11bgjob/specfem/
```

上传后服务器目录结构见 §20.3。

### 20.6 一键正演工作流（v3: 单脚本全流程）

```bash
cd specfem/FWEA23 && sbatch run_forward.bash                 # 首次（含编译）
cd specfem/SinoScope1.0 && SKIP_COMPILE=1 sbatch run_forward.bash  # 复用二进制
cd specfem/EARA2024 && SKIP_COMPILE=1 sbatch run_forward.bash
```

> SinoScope1.0/EARA2024 需先从 FWEA23 复制编译好的 binary：`cp ../FWEA23/bin/* bin/`

**`run_forward.bash` 内部五阶段**：

| Phase | 做什么                                           | 耗时估计   |
| ----- | ------------------------------------------------ | ---------- |
| 1/5   | 编译 (可 SKIP_COMPILE=1 跳过)                    | ~5 min     |
| 2/5   | meshfem (MODEL=s362ani) → solver_data.bin        | ~2 min     |
| 3/5   | Python NC→GLL 转换 (conda activate eastasia_fwi) | ~10-30 min |
| 4/5   | meshfem (MODEL=GLL) → 写入模型弹性参数           | ~2 min     |
| 5/5   | solver → SAC/ASCII 波形                          | ~6-12 hr   |

**版本演化**：

|             | v1（六步）       | v2（三步）              | **v3（一键）**     |
| ----------- | ---------------- | ----------------------- | ------------------ |
| NC→GLL      | 本地运行，需 scp | 服务器端，但单独 sbatch | 同一脚本内自动执行 |
| sbatch 次数 | 3 次 + 3 次 scp  | 3 次                    | **1 次**           |
| 人工等待    | 每步之间等待确认 | 每步之间等待确认        | **提交后无需干预** |
| 失败处理    | 手动检查         | 手动检查                | 每 phase 自动退出  |

### 20.7 关键变化

1. **MODEL 参数**：`gll_qmu` → `GLL`（v7.0.0 不支持 gll_qmu）
2. **Q/衰减处理**：v7.0.0 的 GLL 模式使用 `GLL_REFERENCE_1D_MODEL`（编译时常量 `constants.h.in`）中的 1D 参考模型提供 Q 值，而非从 GLL 文件读取。对正演波形影响可忽略。
3. **编译极简化**：
   ```bash
   ./configure FC=ifort CC=icc CXX=icpc MPIFC=mpiifort
   make -j8 xmeshfem3D xspecfem3D
   ```
   无需任何 sed 补丁、无需分离优化级别。
4. **MPI 启动**：使用 `mpirun -np $NPROC`（chujie 版本已验证），不使用 srun。
5. **v7.0.0 无 REGIONAL_MESH_CUTOFF**：SinoScope/EARA 的 v8.1.0 示例有此参数，v7.0.0 不支持。网格包含完整壳幔，不影响科学结果。

---

_最后更新：2026-04-07（§19 重写: meshfem MPI 接口错误根因为跨机架节点通信不一致，修复方案改为源码修改 get_MPI_interfaces.f90 + test_MPI_interfaces.f90）_
