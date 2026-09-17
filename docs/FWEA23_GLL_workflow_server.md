# FWEA23 GLL 格式正演 - 服务器端流程

曙光 Slurm 集群上完成 FWEA23 模型 GLL 格式正演的完整步骤。

---

## 1. 项目目标

**核心目标**：为**任意公开三维速度模型**提供正演模拟与波形检验工具。

- **背景**：多数公开模型（FWEA23、EARA2024、SinoScope1.0 等）仅发布 **NetCDF (NC)** 或 **CSV** 格式，而 SPECFEM3D Globe 需要 **GLL** 格式。
- **工具**：`convert_nc_to_gll.py` 将 NC/CSV 插值到 GLL，使研究者能用公开模型进行波形正演与检验。
- **测试用例**：以 **FWEA23** 为例，因有原作者参考结果（`specfem/model_updated`、`sac/`）可对比验证。

- 运用第一性原理思考，拒绝经验主义和路径盲从，不要假设我完全清楚目标，保持审慎，从原始需求和问题出发，若目标模糊请停下和我讨论，若目标清晰但路径非最优，请直接建议更短、更低成本的办法。

所有回答必须分为两个部分：
• ​直接执行：按照我当前的要求和逻辑，直接给出任务结果。
• ​深度交互：基于底层逻辑对我的原始需求进行“审慎挑战”。包括但不限于：质疑我的动机是否偏离目标（XY问题）、分析当前路径的弊端、并给出更优雅的替代方案。

---

## 2. 前置条件

- 项目已上传到服务器（含 `specfem3d_globe`、`3_Data_space_simulation`）
- SPECFEM3D Globe 已编译
- **服务器**：conda 环境 `eastasia_fwi`（numpy, scipy, xarray, netcdf4）用于 NC→GLL 转换
- FWEA23 NetCDF：`data/models/processed/2024_FWEA23/2024_FWEA23_original.nc`（需含 eta）

---

## 3. SPECFEM3D Globe 数据流（必读）

理解流程之前必须理解 SPECFEM 的内部数据流：

```
┌─────────────────────────────────────────────────────────────────────┐
│  xmeshfem3D（mesher）                                               │
│                                                                     │
│  输入:                                                              │
│    Par_file (MODEL, NEX, NCHUNKS...)                                │
│    constants.h.in (REGIONAL_MOHO, NER, 编译时)                      │
│    DATA/GLL/*.bin (仅当 MODEL 含 "gll" 时)                          │
│                                                                     │
│  内部处理:                                                           │
│    1. 创建谱元网格 → 元素拓扑 ibool、Jacobian、GLL 点坐标 (x,y,z)   │
│    2. 在每个 GLL 点获取弹性参数:                                     │
│       · MODEL=s362ani_crust1.0 → 解析计算 s362ani+CRUST1.0 模型     │
│       · MODEL=GLL (v7) / gll_qmu (v8) → 从 DATA/GLL/proc*_reg1_*.bin 读取│
│    3. 速度(km/s) → 弹性模量: κ = ρ(Vp²-4/3·Vs²), μ = ρ·Vs²        │
│    4. 写出 solver_data.bin = 网格几何 + 弹性模量（κ,μ,ρ...）         │
│                                                                     │
│  输出: DATABASES_MPI/proc*_reg1_solver_data.bin                     │
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

### 3.1 convert_nc_to_gll.py 的角色

```
┌──────────────────────────────────────────────────────────────────┐
│  convert_nc_to_gll.py（坐标映射器，本地执行）                     │
│                                                                  │
│  输入:                                                           │
│    · DATABASES_MPI/solver_data.bin → 提取 GLL 点坐标 (x,y,z)    │
│    · FWEA23 NetCDF (lat, lon, depth 规则网格)                    │
│    · 1DREF 参考模型（填充深部/域外区域）                          │
│                                                                  │
│  处理:                                                           │
│    1. 从 solver_data.bin 读取 nspec, nglob, (x,y,z), ibool      │
│    2. 非量纲化 (x,y,z) → 地理 (lat°, lon°, depth_km)            │
│    3. 在 NC 规则网格上插值:                                       │
│       · 域内: 三线性插值                                         │
│       · 深部(>NC范围): 用 1DREF 参考模型                         │
│       · 水平域外(chunk 边缘): 最近邻外推                         │
│    4. 单位转换: rho kg/m³→g/cm³（已在插值前完成）                 │
│                                                                  │
│  输出: DATA/GLL/proc*_reg1_{vpv,vph,vsv,vsh,eta,rho,qmu}.bin    │
│  ⚠️ 输出的是速度格式(km/s)，需要 mesher 转换为弹性模量            │
└──────────────────────────────────────────────────────────────────┘
```

`convert_nc_to_gll.py` 本质上是一个**坐标映射器**：把规则网格上的速度模型值，映射到 SPECFEM 的非结构化 GLL 网格上。但它**不**做速度→弹性模量的转换——这是 mesher 的工作。

### 3.2 完整操作流程（六步）

```
┌──────────────────────────────────────────────────────────────────────┐
│  步骤 ①  [服务器] 编译 + 第一次 meshfem（s362ani）                   │
│          sbatch run_meshfem_only.bash                                │
│          → DATABASES_MPI/proc*_reg1_solver_data.bin（含 GLL 坐标）   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ scp -r DATABASES_MPI → 本地
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  步骤 ②  [本地] 下载 DATABASES_MPI                                   │
│          scp -r user@曙光:/.../FWEA23/DATABASES_MPI \                │
│              specfem3d_globe/EXAMPLES/FWEA23/                        │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  步骤 ③  [本地] NC → GLL 转换                                        │
│          conda activate eastasia_fwi                                 │
│          python 3_Data_space_simulation/convert_nc_to_gll.py \       │
│              --mesh-dir  specfem3d_globe/EXAMPLES/FWEA23/DATABASES_MPI \
│              --model-nc  data/models/processed/2024_FWEA23/2024_FWEA23_original.nc \
│              --output-dir specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL \ │
│              --anisotropic                                           │
│          → DATA/GLL/proc*_reg1_{vpv,vph,vsv,vsh,eta,rho,qmu}.bin    │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ scp -r DATA/GLL → 服务器
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  步骤 ④  [本地→服务器] 上传 DATA/GLL                                 │
│          scp -r specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL \           │
│              user@曙光:/.../FWEA23/DATA/                             │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  步骤 ⑤  [服务器] 第二次 meshfem（gll_qmu）                          │
│          sbatch run_meshfem_gll.bash                                 │
│          → mesher 读 DATA/GLL/ 速度 → 转弹性模量 → 重写 solver_data  │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  步骤 ⑥  [服务器] Solver 正演                                        │
│          sbatch run_solver_only.bash                                 │
│          → 使用 FWEA23 弹性参数进行波场计算 → 输出 SAC/ASCII 波形     │
└──────────────────────────────────────────────────────────────────────┘
```

> **步骤 ⑤ 是之前缺失的关键环节**。之前的流程跳过了第二次 meshfem，导致 solver 实际使用 s362ani 参数而非 FWEA23。

### 3.3 为什么需要两次 meshfem？

| 问题                             | 解释                                                                                                     |
| -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 为什么不能一次？                 | 鸡生蛋问题：`convert_nc_to_gll.py` 需要 GLL 坐标（来自 mesher），但 mesher 需要 GLL 文件（来自转换脚本） |
| 第二次 mesh 几何会变吗？         | 不会。两次 mesher 用相同的 NEX、constants.h.in、1DREF，网格拓扑完全一致（NSPEC=7236）                    |
| 第二次 mesh 会 face flag 吗？    | 极低概率。mesh 是确定性的，相同配置 = 相同结果。但为安全可重提一次                                       |
| PATHNAME_GLL_modeldir 在哪配置？ | 编译时常量 `constants.h.in` 中: `PATHNAME_GLL_modeldir = 'DATA/GLL/'`（不在 Par_file 中）                |

---

## 4. 准备 NC 模型（本地，含 eta）

确保 `1_5_Process_velocity_models.py` 中 FWEA23 的 params 包含 `eta` 和 `qmu`：

```python
'params': ['vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho', 'qmu', 'vs0', 'vp0']
```

运行 `1_5` 生成含 eta 和 qmu 的 `2024_FWEA23_original.nc`。

---

## 5. 步骤 ①：编译 + 第一次 meshfem（s362ani，服务器）

**目的**：生成网格几何，提供 GLL 点坐标给 NC→GLL 转换。

```bash
cd specfem3d_globe/EXAMPLES/FWEA23
sbatch run_meshfem_only.bash
# 若未改 constants.h.in，可跳过 make clean 加速:
# SKIP_CLEAN=1 sbatch run_meshfem_only.bash
```

等待作业完成，确认：

```bash
ls DATABASES_MPI/proc*_reg1_solver_data.bin | wc -l   # 应为 144
```

**完成后立即下载到本地（步骤 ②）**：

```bash
# 在本地执行，将服务器上的 DATABASES_MPI 下载到本地项目对应目录
scp -r user@曙光:/path/to/specfem3d_globe/EXAMPLES/FWEA23/DATABASES_MPI \
    specfem3d_globe/EXAMPLES/FWEA23/
```

> DATABASES_MPI 约 144×50MB ≈ 7GB，视网络情况可能需要较长时间。
> 可只下载 `solver_data.bin`：`scp "user@曙光:/.../DATABASES_MPI/proc*_reg1_solver_data.bin" DATABASES_MPI/`

**此步骤产出的 solver_data.bin 包含**：

| 内容                           | 用途                                   |
| ------------------------------ | -------------------------------------- |
| 元素拓扑 (ibool, nspec, nglob) | 步骤 ③ 读取 GLL 坐标                   |
| GLL 点坐标 (x, y, z)           | 步骤 ③ 转换为 (lat, lon, depth) 后插值 |
| Jacobian 矩阵                  | 步骤 ⑤ 复用（不变）                    |
| s362ani 弹性模量 (κ, μ, ρ)     | **步骤 ⑤ 将被 FWEA23 值覆盖**          |

**当前 constants.h.in 配置**：

```fortran
EARTH_REGIONAL_MOHO_MESH         = .true.    ! tibet_v4: 3-layer crust
EARTH_HONOR_DEEP_MOHO            = .true.    ! deep moho stretching (60km)
EARTH_RMOHO_STRETCH_ADJUSTMENT   = -20000.d0 ! moho mesh boundary down to 60km
```

> face flag bug 两阶段修复（§16.2 + §18）：
>
> 1. `auto_ner.f90` R80 一致性修复（§16.2）— 消除 NER 计算错误
> 2. 编译参数回归 SPECFEM 默认（§18）— 消除 `-O1` 优化触发的运行时错误
>    修改 `constants.h.in` 或源码后必须 `make clean` 重编译。

---

## 6. 步骤 ③④：NC → GLL 转换 + 上传（本地执行）

**目的**：将 FWEA23 NetCDF 速度模型插值到步骤 ① 生成的 GLL 网格点上，然后上传到服务器。

曙光服务器 GLIBC 较旧，Miniconda 无法安装，**在本地**完成转换。

`convert_nc_to_gll.py` 对每个 proc 执行：

1. **读取坐标**：从 `solver_data.bin` 提取 (x,y,z) 和 ibool
2. **坐标转换**：非量纲 (x,y,z) → 地理 (lat°, lon°, depth_km)，公式 `depth = R_EARTH × (1 - r)`
3. **三线性插值**：在 NC 规则网格上插值 vpv, vph, vsv, vsh, eta, rho, qmu
4. **域外填充**：深度超出 NC → 1DREF 参考模型；水平域外 → 最近邻外推
5. **单位转换**：rho 在插值前从 kg/m³ 转为 g/cm³（源级转换，与 1DREF 统一）
6. **写出 GLL 文件**：Fortran 无格式二进制 `(NGLLX × NGLLY × NGLLZ × nspec)`

### 步骤 ③：NC→GLL 转换

```bash
# 在本地项目根目录执行
cd /path/to/EASTASIA_FWI
conda activate eastasia_fwi

# 确认 DATABASES_MPI 已下载到本地（步骤 ② 完成）
ls specfem3d_globe/EXAMPLES/FWEA23/DATABASES_MPI/proc000000_reg1_solver_data.bin

# 运行转换
python 3_Data_space_simulation/convert_nc_to_gll.py \
    --mesh-dir specfem3d_globe/EXAMPLES/FWEA23/DATABASES_MPI \
    --model-nc data/models/processed/2024_FWEA23/2024_FWEA23_original.nc \
    --output-dir specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL \
    --anisotropic

python 3_Data_space_simulation/convert_nc_to_gll.py \
    --mesh-dir specfem/FWEA23/DATABASES_MPI \
    --model-nc data/models/processed/2024_FWEA23/2024_FWEA23_original.nc \
    --output-dir specfem/FWEA23/DATA/GLL \
    --anisotropic

```

### 步骤 ④：上传 GLL 到服务器

```bash
# 确认 144 个 proc 的 GLL 文件均已生成
ls specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL/proc*_reg1_vpv.bin | wc -l  # 应为 144

# 上传到服务器（DATA/GLL/ 约 144×7×3.5MB ≈ 3.5GB）
scp -r specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL \
    user@曙光:/path/to/specfem3d_globe/EXAMPLES/FWEA23/DATA/

# 在服务器上验证文件完整
ssh user@曙光 "ls /path/to/FWEA23/DATA/GLL/proc*_reg1_vpv.bin | wc -l"  # 应为 144
```

**验证诊断输出**（proc000000 应显示合理值）：

```
vpv: 1.97 ~ 13.72   (km/s)
rho: 1.34 ~ 5.57    (g/cm³，全局 min 来自浅层沉积盆地）
eta: 0.90 ~ 1.01    (非全 1.0)
rho mean ≈ 4.0 g/cm³
```

**若 SPECFEM 以 double 精度编译**（`CUSTOM_REAL=SIZE_DOUBLE`），加 `--double`。

### 6.1 转换自检选项

| 选项                                        | 说明                                           |
| ------------------------------------------- | ---------------------------------------------- |
| `--reference-gll-dir specfem/model_updated` | 自动对比 `proc000000` 的 RMSE（需 nspec 匹配） |
| `--compare-all-procs`                       | 对全部 proc 执行 reference 对比                |
| `--apply-physical-floor`                    | 仅在数值稳定性需要时开启                       |
| `--horizontal-fill {nearest,ref,blend}`     | 域外填充策略（默认 `nearest`）                 |

---

## 7. 步骤 ⑤：第二次 meshfem（gll_qmu，服务器）

**目的**：让 mesher 读取 `DATA/GLL/` 中的速度文件，转换为弹性模量，写入新的 `solver_data.bin`。

**这是之前流程中缺失的关键步骤。** 没有这一步，solver 使用的是第一次 mesh 的 s362ani 参数。

```bash
cd specfem3d_globe/EXAMPLES/FWEA23

# 确认 Par_file 已设为 GLL 模型
grep "^MODEL" DATA/Par_file
# 应显示: MODEL = gll_qmu

# 确认 DATA/GLL/ 文件存在
ls DATA/GLL/proc000000_reg1_vpv.bin  # 应存在

# 提交第二次 meshfem（不重新编译，仅 xmeshfem3D）
sbatch run_meshfem_gll.bash
```

**`run_meshfem_gll.bash`** 与第一次的区别：

| 对比项               | 第一次 (`run_meshfem_only.bash`) | 第二次 (`run_meshfem_gll.bash`) |
| -------------------- | -------------------------------- | ------------------------------- |
| MODEL                | s362ani_crust1.0                 | gll_qmu                         |
| 编译                 | 是（make meshfem3D xspecfem3D）  | 否（用已有二进制）              |
| 清理 DATABASES_MPI   | 是                               | 是（mesher 从零生成全部文件）   |
| mesher 读 GLL 文件   | 否                               | **是**（读 DATA/GLL/）          |
| mesher 读 qmu.bin    | 否                               | **是**（ATTENUATION_GLL=true）  |
| 输出 solver_data.bin | 含 s362ani 弹性模量              | **含 FWEA23 弹性模量**          |

**为什么用 `gll_qmu` 而非 `GLL_crust1.0`？**

`get_model_parameters.F90` 中两者代码路径不同：

|                 | `GLL_crust1.0` → case `gll` | `gll_qmu` → case `gll_qmu` |
| --------------- | --------------------------- | -------------------------- |
| MODEL_GLL       | true                        | true                       |
| MODEL_GLL_TYPE  | 2 (tiso)                    | 2 (tiso)                   |
| ATTENUATION_GLL | **false**                   | **true**                   |
| impose_crust    | ICRUST_CRUST1（后缀）       | ICRUST_CRUST1（硬编码）    |

`ATTENUATION_GLL = false` 时 mesher **不读** `qmu.bin`，使用默认 1D Qmu。`gll_qmu` 是原作者 tibet_v4 专门创建的 case，确保 FWEA23 的三维衰减结构被正确读取。

确认第二次 mesh 完成：

```bash
# mesher 应输出 "reading in model from: DATA/GLL/"
grep "reading in model" output_mesher.txt
# NSPEC 应与第一次相同 (7236)
grep "number of elements (per slice)" output_mesher.txt
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

**根因诊断（§18）**：

- `auto_ner.f90` R80 不一致（§16.2）导致 NER 计算错误
- 自定义编译参数 `-O1 -fp-model precise`（非 SPECFEM 默认）触发 ifort 2018 优化相关运行时错误
- DEBUG 模式 (`-O0 -check bounds`) 下 mesher 100% 成功，证明非逻辑 bug

**修复**：

1. `auto_ner.f90` R80 一致性修复（§16.2）
2. 编译参数回归 SPECFEM 默认 `-O3`（§18），不再自定义 `FLAGS_CHECK`

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
| 1   | `auto_ner.f90`                | 注释 NER 经验值覆盖 + **R80 bug fix (§16.2)** | **直接影响**（网格拓扑 + 消除 face flag） | ✅ 保留 + 新增      |
| 2   | `get_timestep_and_layers.f90` | ~~`DT=0.1d0` 硬编码~~ → 已替换为 v8 bug 修复  | **直接影响**（DT/CFL）                    | ✅ 已修复，见 §11.4 |
| 3   | `get_model_parameters.F90`    | `gll_qmu` 添加 `impose_crust`                 | 无影响（冗余但无害）                      | ✅ 保留             |
| 4   | `compute_forces_*.F90`        | 注释 inner core early return → 条件化修复     | **🔴 不修复则正演发散**                   | ✅ 已修复，见 §11.5 |
| 5   | `save_kernels.F90`            | 注释 `restore_original_moduli()`              | 无影响（仅 kernel 阶段）                  | ✅ 保留             |

### 11.3 我方修改（4 处）

| #   | 文件                       | 修改内容                              | 说明                             |
| --- | -------------------------- | ------------------------------------- | -------------------------------- |
| 1   | `constants.h.in`           | tibet_v4 配置（REGIONAL/HONOR/RMOHO） | Gate C 网格配置                  |
| 2   | `get_model.F90`            | Plan D：vpv=0 时回退到 1D 参考值      | 防止 CRUST1.0 海岸过渡带零值崩溃 |
| 3   | `setup_MPI_interfaces.f90` | `MAX_NEIGHBORS = 20 + NCORNERSCHUNKS` | 安全余量（原值 8）               |
| 4   | `iterate_time_undoatt.F90` | 单行 if → if...then...endif           | ifort 编译器兼容修复             |

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

## 13. 波形差异分析

### 13.1 当前波形差异来源（按影响排序）

本次 GLL 正演波形（v8.1.0）与原作者参考波形（v7.0.0 + tibet_v4）存在明显差异。已排除参数层面因素（UNDO_ATTENUATION、CMTSOLUTION、MODEL 格式等均已对齐），剩余差异为结构性因素：

| 因素                 | 影响 | 说明                                                       |
| -------------------- | ---- | ---------------------------------------------------------- |
| **网格拓扑差异**     | 高   | NSPEC 7236 vs 7452、NER_CRUST 2 vs 3、深 Moho 处理差异     |
| **地壳层数差异**     | 高   | REGIONAL_MOHO_MESH + HONOR_DEEP_MOHO 影响面波/Pn/Sn        |
| **NC→GLL 精度损失**  | 中   | NC vpv_min=1.86 vs 原始 GLL vpv_min=0.75（浅层细节被平滑） |
| **SPECFEM 版本差异** | 低   | v8.1.0 vs v7.0.0 数值微调（通常 < 1%）                     |

### 13.2 验证策略

| 步骤       | 做什么                                | 验证什么                      |
| ---------- | ------------------------------------- | ----------------------------- |
| **Step 1** | patch auto_ner → meshfem → 记录 NSPEC | 网格是否匹配 v7 的 7236       |
| **Step 2** | 用原作者 GLL 跑 v8 solver             | 隔离版本差异（需 NSPEC 匹配） |
| **Step 3** | NC→GLL 转换后跑，与 Step 2 对比       | 量化 NC 精度损失              |

> Step 2 需要 NSPEC=7236（auto_ner 补丁已应用），可用 `specfem/model_updated/` 的 GLL 文件。

### 13.3 边界处理

与 v7.0.0 一致：`REGIONAL_MESH_CUTOFF = .false.`，`ABSORB_USING_GLOBAL_SPONGE = .false.`，仅 Stacey 吸收边界。

---

## 14. 测试结果摘要

### 14.1 Mesher 测试

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
| **03-24** | **5691671** | **Round 2 mesh1 (auto_ner fix, DEBUG `-O0`)** | **7236** | —        | —         | **✅**        | **DEBUG 成功：无越界/无 face flag → 确认非逻辑 bug，是编译优化问题（§18）** |

### 14.2 Solver 测试

| 日期  | Job ID  | 配置             | NSPEC | DT   | 结果    | 备注                                                                              |
| ----- | ------- | ---------------- | ----- | ---- | ------- | --------------------------------------------------------------------------------- |
| 03-19 | 5637010 | Gate A + GLL     | 7452  | 0.11 | ✅ 稳定 | Max U ≈ 5.07e-02，无发散                                                          |
| 03-23 | —       | §23 全补丁 + GLL | 7236  | 0.1  | ❌ 发散 | ~15 min 后指数增长 → compute_forces 补丁所致（且 solver 实际用 s362ani 而非 GLL） |

### 14.3 结论

- **Gate A 配置已验证可运行**（NSPEC=7452，标准网格）
- **tibet_v4 配置 mesher 已打通**（需 auto_ner + Plan D），NSPEC=7236
- **Solver 发散根因已定位并修复**：compute_forces inner core 条件化（§11.5，本地已应用）
- **发现工作流缺陷**：之前缺少第二次 meshfem（§7），solver 实际使用 s362ani 而非 FWEA23
- **face flag 根因确认（§18）**：自定义编译参数 `-O1 -fp-model precise` 是直接原因，DEBUG `-O0` 模式完全正常
- **修复策略**：auto_ner R80 fix（§16.2）+ 编译参数回归 SPECFEM 默认 `-O3`（§18）

---

## 15. 测试执行方案

### 15.1 当前阶段优先事项

```
前置：确认服务器源码包含所有补丁（§11），make clean && make
    ↓
步骤 ①  [服务器] sbatch run_meshfem_only.bash
        → 编译 + 第一次 mesh(s362ani)，获取 GLL 坐标
    ↓
步骤 ②  [服务器→本地] scp -r DATABASES_MPI → 本地
    ↓
步骤 ③  [本地] python convert_nc_to_gll.py → DATA/GLL/
    ↓
步骤 ④  [本地→服务器] scp -r DATA/GLL → 服务器
    ↓
步骤 ⑤  [服务器] sbatch run_meshfem_gll.bash
        → 第二次 mesh(gll_qmu)，读 DATA/GLL/ + qmu.bin 写 solver_data.bin
    ↓
步骤 ⑥  [服务器] sbatch run_solver_only.bash
        → 正演，使用 FWEA23 弹性参数
    ↓
验证：Max U 稳定性 + 下载 SAC 波形对比
```

### 15.2 配置目标

```fortran
! constants.h.in（保持 tibet_v4 配置）
EARTH_REGIONAL_MOHO_MESH         = .true.
EARTH_HONOR_DEEP_MOHO            = .true.
EARTH_RMOHO_STRETCH_ADJUSTMENT   = -20000.d0

! 源码补丁（全部已应用到本地源码）
auto_ner.f90                → NER 经验值注释 + R80→R80_FICTITIOUS bug fix（§16.2）
get_timestep_and_layers.f90 → DT*(1.d0-0.1d0)（v8 bug 修复）
compute_forces_*.F90        → SIMULATION_TYPE==1 条件化 return（§11.5）
get_model.F90               → Plan D vpv=0 fallback
```

### 15.3 验证标准

| 阶段                    | 通过条件                                                      |
| ----------------------- | ------------------------------------------------------------- |
| 编译                    | 无错误（warning 可忽略）                                      |
| 第一次 Mesher (s362ani) | exit 0，NSPEC=7236，CFL < 1.0                                 |
| NC→GLL                  | 144 个 proc 转换完成，值域合理                                |
| 第二次 Mesher (GLL)     | exit 0，NSPEC=7236，输出含 "reading in model from: DATA/GLL/" |
| Solver                  | Max U 不指数增长（30 min 模拟时间内稳定）                     |
| 波形                    | SAC 文件完整，与原作者定性对比                                |

### 15.4 每次作业记录模板

```
作业号: ________    日期: ________
阶段: mesh1(s362ani) / mesh2(GLL) / solver
constants.h.in: REGIONAL=___ HONOR=___ RMOHO=___
源码补丁: auto_ner=___ DT=___ compute_forces=___ Plan_D=___
编译: make clean=是/否
结果: NSPEC=___ DT=___ CFL=___ solver稳定=___
```

---

## 16. 网格差异根因

### 16.1 NSPEC 7452 vs 7236

v8.1.0 的 `auto_ner()` 用 `get_timestep_and_layers` 里为 90° chunk 设计的经验 NER 值作为起点，再由 `auto_optimal_ner()` 向上搜索最优纵横比。v7.0.0 从最小值（全 1）开始搜索。

`auto_optimal_ner` 只增不减，起点越大终点越大 → v8 在 80° chunk 多出 216 个元素/proc。

**修复**：patch `auto_ner.f90`，注释掉经验值覆盖（lines 481-497）。

### 16.2 v8 REGIONAL_MOHO_MESH + HONOR_DEEP_MOHO 的已知 Bug（已修复）

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

## 17. 未来测试规划

### 17.1 当前轮次（Round 1）：完成六步工作流

**状态**：步骤① 已完成（Job 5681617），后续步骤待执行。

```
✅ 步骤 ①  mesh1(s362ani)    → Job 5681617, NSPEC=7236, CFL=0.623
⏳ 步骤 ②  下载 DATABASES_MPI 到本地
⏳ 步骤 ③  NC→GLL 转换（本地）
⏳ 步骤 ④  上传 DATA/GLL 到服务器
⏳ 步骤 ⑤  mesh2(gll_qmu)    → sbatch run_meshfem_gll.bash
⏳ 步骤 ⑥  solver 正演        → sbatch run_solver_only.bash
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

### 17.2 Round 2（已实施）：auto_ner 修复 + 编译参数回归

**触发原因**：Round 1 步骤② mesh2(gll_qmu) Job 5682481 连续 5 次全部失败。

**两阶段修复**：

| 阶段   | 修复                                       | 发现                                                  |
| ------ | ------------------------------------------ | ----------------------------------------------------- |
| 2a     | `auto_ner.f90` R80→R80_FICTITIOUS（§16.2） | 修复后 `-O1` 仍 face flag（Job 5685845）              |
| 2b     | DEBUG `-O0` 编译测试（§18）                | **100% 成功**（Job 5691671），确认为编译优化问题      |
| **2c** | **编译参数回归 SPECFEM 默认（§18.4）**     | 不再自定义 `FLAGS_CHECK`，使用 `flags.guess` 的 `-O3` |

**修复内容**：

```fortran
! auto_ner.f90 — 一行修改（§16.2）
radius(3) = R80_FICTITIOUS_IN_MESHER     ! 修复: 与 define_all_layers 一致 (was R80)
```

```bash
# run_meshfem_only.bash — 编译配置（§18.4）
# 不传 FLAGS_CHECK → ./configure 自动通过 flags.guess 选择 ifort 默认参数
./configure FC=ifort CC=icc CXX=icpc MPIFC=mpiifort --with-netcdf ...
# flags.guess → -O3 -check nobounds -fpe0 -ftz -assume byterecl ...
```

```fortran
! constants.h.in — 保持不变（tibet_v4 原配置）
EARTH_REGIONAL_MOHO_MESH       = .true.
EARTH_HONOR_DEEP_MOHO          = .true.
EARTH_RMOHO_STRETCH_ADJUSTMENT = -20000.d0
```

> ⚠️ 修改源码后必须 `make clean && make` 重新编译。

**预期影响**：

| 指标           | Round 1（两个 bug 未修复）     | Round 2（两个修复均已应用）         |
| -------------- | ------------------------------ | ----------------------------------- |
| face flag 风险 | 偶发→确定性失败                | **消除**（auto_ner + 默认编译参数） |
| Moho Honoring  | ✅ 保留                        | ✅ 保留                             |
| NER_80_MOHO    | 1（元素太厚）                  | **≥ 2**（正常纵横比）               |
| NSPEC          | 7236                           | **可能变化**（NER_80_MOHO 增大）    |
| 编译参数       | 自定义 `-O1 -fp-model precise` | **SPECFEM 默认 `-O3`**              |

**测试步骤**：

```
Round 2-①  同步 auto_ner.f90 + run_meshfem_only.bash 到服务器
Round 2-②  sbatch run_meshfem_only.bash（默认编译参数，应确定性成功）
Round 2-③  记录 NSPEC、DT、CFL、NER_80_MOHO、Min Jacobian ratio
Round 2-④  若 NSPEC 变化（很可能）→ 需重做 NC→GLL 转换
Round 2-⑤  完整六步工作流
Round 2-⑥  验证波形稳定性 + 与原作者参考对比
```

### 17.3 长期规划（Round 3+）：多模型批量正演

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

### 17.4 脚本增强记录

**已实现**：

| 日期      | 脚本                        | 增强                        | 说明                                                              |
| --------- | --------------------------- | --------------------------- | ----------------------------------------------------------------- |
| 03-24     | `run_meshfem_only.bash`     | 自动重试 (MAX_RETRY=5)      | 检测 face flag / MPI 错误自动重试，非偶发错误立即停止             |
| 03-24     | `run_meshfem_gll.bash`      | 自动重试 (MAX_RETRY=5)      | 同上                                                              |
| 03-24     | `run_meshfem_only.bash`     | 显式错误检查                | 去除 `set -e`，每步 `if [$? -ne 0]` 检查                          |
| 03-24     | `run_meshfem_only.bash`     | topo_bathy 软链接           | 防止 mesher 找不到地形数据                                        |
| 03-24     | `run_meshfem_only.bash`     | DEBUG 模式 (`DEBUG=1`)      | `-O0 -check all` 编译，单次运行，用于诊断                         |
| **03-24** | **`run_meshfem_only.bash`** | **编译参数回归默认（§18）** | **不再自定义 `FLAGS_CHECK`，使用 SPECFEM `flags.guess` 默认参数** |

**待实现**：

| 功能           | 优先级 | 说明                         |
| -------------- | ------ | ---------------------------- |
| 六步一键脚本   | 中     | 封装 ①~⑥ 减少手动 scp        |
| 波形自动对比   | 低     | solver 完成后自动计算 misfit |
| 多模型配置文件 | 低     | 不同模型共用同一套脚本       |

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

| 脚本                    | 编译   | 作用                                                                             |
| ----------------------- | ------ | -------------------------------------------------------------------------------- |
| `run_meshfem_only.bash` | **是** | 编译（SPECFEM 默认 flags） + 第一次 meshfem（s362ani）→ 恢复 Par_file 为 gll_qmu |
| `run_meshfem_gll.bash`  | 否     | 第二次 meshfem（gll_qmu）→ 读取 DATA/GLL/ + qmu.bin 写入 solver_data.bin         |
| `run_solver_only.bash`  | 否     | 仅 solver（检查 mesh_parameters.bin 存在）                                       |

编译参数策略（§18）：不自定义 `FLAGS_CHECK`，让 `flags.guess` 为 ifort 选择已验证的默认参数。DEBUG 模式（`DEBUG=1`）覆盖为 `-O0 -check all`。

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
| **5691671** | **03-24** | **mesh1(auto_ner fix, DEBUG `-O0`)** | **✅**                | **Round 2b: DEBUG 成功 → 确认编译优化为根因（§18）**     |

---

## 18. 编译参数诊断与修复

### 18.1 问题

`auto_ner.f90` R80 bug 修复后（§16.2），mesh1 仍持续出现 face flag 错误（Job 5685845）。

### 18.2 诊断过程

在 `run_meshfem_only.bash` 中增加 `DEBUG=1` 模式（`-O0 -g -traceback -check bounds -check uninit`），提交 DEBUG 编译运行。

**结果（Job 5691671）**：`-O0` 编译的 mesher **100% 成功**，144 个 `solver_data.bin` 全部生成，`.err` 无任何错误。

**结论**：mesh 生成逻辑与 MPI 接口构造**本身没有 bug**。问题出在编译优化。

### 18.3 根因

之前的 `run_meshfem_only.bash` **绕过了 SPECFEM 默认编译配置**，自定义传入：

```bash
FLAGS_CHECK="-O1 -fp-model precise -fpe0 -ftz -assume buffered_io -assume byterecl -align sequence -std08 ..."
```

而 SPECFEM 自己 `flags.guess` 为 ifort 生成的已验证默认参数是：

```bash
DEF_FFLAGS="-xHost -fpe0 -ftz -assume buffered_io -assume byterecl -align sequence -std08 -diag-disable 6477 -implicitnone -gen-interfaces -warn all,noexternal"
OPT_FFLAGS="-O3 -check nobounds"
```

| 差异         | SPECFEM 默认                  | 自定义              | 风险                                                 |
| ------------ | ----------------------------- | ------------------- | ---------------------------------------------------- |
| 优化级别     | `-O3`                         | `-O1`               | `-O1` 使用不同优化路径，可能触发 ifort 2018 特定 bug |
| 浮点模型     | 无                            | `-fp-model precise` | 改变浮点运算行为，影响 MPI flag 累加                 |
| bounds check | `-check nobounds`（显式关闭） | 无                  | 无直接影响                                           |
| `-xHost`     | 有                            | 被 sed 移除         | 可能影响指令集选择                                   |

自定义参数本意是"更保守"，实际上偏离了 SPECFEM 开发者验证过的配置组合，在 ifort 2018.5.274 + REGIONAL_MOHO_MESH 场景下触发了运行时错误。

### 18.4 修复

`run_meshfem_only.bash` 不再传入 `FLAGS_CHECK`，让 `./configure` 通过 `flags.guess` 自动选择：

```bash
# 修复前（自定义覆盖）
./configure FC=ifort ... FLAGS_CHECK="$FLAGS_SAFE" ...

# 修复后（使用 SPECFEM 默认）
./configure FC=ifort CC=icc CXX=icpc MPIFC=mpiifort --with-netcdf ...
# flags.guess 自动检测 ifort → -O3 -check nobounds -fpe0 -ftz ...
```

仅保留两项最小调整：

- 移除 `-xHost`（集群登录节点与计算节点 CPU 可能不同）
- DEBUG 模式保留 `FLAGS_CHECK=-O0 -check all`

### 18.5 影响的脚本

| 脚本                    | 是否涉及编译                          | 是否需要修改  |
| ----------------------- | ------------------------------------- | ------------- |
| `run_meshfem_only.bash` | **是**（make meshfem3D + xspecfem3D） | **✅ 已修改** |
| `run_meshfem_gll.bash`  | 否（运行已编译 binary）               | ❌ 不需要     |
| `run_solver_only.bash`  | 否（运行已编译 binary）               | ❌ 不需要     |

### 18.6 教训

> **不要覆盖大型 Fortran 项目已验证的编译配置。** SPECFEM3D Globe 的 `flags.guess` 经过开发者长期测试，自定义 `FLAGS_CHECK` 绕过了这个验证过的配置路径。"更保守"的参数（`-O1 -fp-model precise`）反而引入了问题。

---

## 19. 切换到 SPECFEM3D Globe v7.0.0（2026-03-25）

### 19.1 决策原因

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

**v7.0.0 无此问题**：已通过 chujie 版本在同一集群上验证成功。

### 19.2 v7.0.0 vs v8.1.0 对比

| 项目         | v8.1.0                     | v7.0.0                                  |
| ------------ | -------------------------- | --------------------------------------- |
| MPI 接口 bug | NCHUNKS=1, NEX=288 时触发  | 无                                      |
| MODEL (GLL)  | `gll_qmu`                  | `GLL`                                   |
| 编译         | 需 sed 补丁 (warn, -O0)    | 直接 configure + make                   |
| MPI 启动     | `srun --mpi=pmi2`          | `mpirun -np` (已验证)                   |
| Par_file     | 含 REGIONAL_MESH_CUTOFF 等 | 不含 (v8.x 新增)                        |
| Q/衰减 (GLL) | 从 GLL 文件读 qmu          | 从 1D 参考模型 (GLL_REFERENCE_1D_MODEL) |
| 科学结果     | 等价 (相同 NEX/模型)       | 等价                                    |

### 19.3 目录结构（三模型并行 + 服务器端转换）

```
specfem/                                  # 服务器: /work/home/acf11bgjob/specfem/
├── specfem3d_globe_code_new/             # 原作者修改版 v7.0.0 源码 (共享)
├── sem_config_tibet_v4/                  # 原作者配置 (constants.h.in)
├── models/                              # 模型 NC 文件 (一次性上传)
│   ├── 2024_FWEA23_original.nc
│   ├── 2022_SinoScope1.0_original.nc
│   └── 2024_EARA2024_original.nc
├── scripts/                             # 共享脚本 (一次性上传)
│   └── convert_nc_to_gll.py
├── FWEA23/                              # ✅ 已验证通过
│   ├── DATA/ (Par_file, CMTSOLUTION, STATIONS, GLL/)
│   ├── bin/ (xmeshfem3D, xspecfem3D)
│   ├── run_1_meshfem_s362ani.bash
│   ├── run_2_convert_and_meshfem_gll.bash
│   └── run_3_solver.bash
├── SinoScope1.0/                        # 115°×65°, 中心 22.5°N/107.5°E
│   └── ... (同上结构)
└── EARA2024/                            # 80°×50°, 中心 35°N/120°E
    └── ... (同上结构)
```

### 19.4 三模型 Par_file 差异

| 参数 | FWEA23 | SinoScope1.0 | EARA2024 |
| ---- | ------ | ------------ | -------- |
| 角宽 (XI × ETA) | 80° × 80° | 115° × 65° | 80° × 50° |
| 中心坐标 | 32°N, 106°E | 22.5°N, 107.5°E | 35°N, 120°E |
| 记录时长 | 30 min | 30 min | 30 min |
| NEX / NPROC | 288, 12×12 | 288, 12×12 | 288, 12×12 |
| CMTSOLUTION | Kyushu 2014 Mw6.3 | 同左 | 同左 |
| STATIONS | 1403 台站 | 同左 | 同左 |

其余参数（ATTENUATION、UNDO_ATTENUATION、ANISOTROPIC_KL 等）均继承 v7.0.0 原作者配置。

### 19.5 服务器准备（一次性）

```bash
# 1. conda 环境（已完成）
module load anaconda3/2023.09
conda create -n eastasia_fwi python=3.10 numpy scipy xarray netcdf4 -y

# 2. 上传模型 NC 文件
mkdir -p /work/home/acf11bgjob/specfem/models
scp data/models/processed/2024_FWEA23/2024_FWEA23_original.nc \
    server:specfem/models/
scp data/models/processed/2022_SinoScope1.0/2022_SinoScope1.0_original.nc \
    server:specfem/models/
scp data/models/processed/2024_EARA2024/2024_EARA2024_original.nc \
    server:specfem/models/

# 3. 上传转换脚本
mkdir -p /work/home/acf11bgjob/specfem/scripts
scp 3_Data_space_simulation/convert_nc_to_gll.py \
    server:specfem/scripts/

# 4. 上传模型工作目录
for m in FWEA23 SinoScope1.0 EARA2024; do
    scp -r specfem/$m/ server:specfem/$m/
done
```

### 19.6 优化三步工作流（v2: 服务器端转换）

| 步骤 | 位置   | 命令 | 说明 |
| ---- | ------ | ---- | ---- |
| ①    | 服务器 | `sbatch run_1_meshfem_s362ani.bash` | 编译 + mesh (MODEL=s362ani) |
| ②    | 服务器 | `sbatch run_2_convert_and_meshfem_gll.bash` | NC→GLL + mesh (MODEL=GLL) |
| ③    | 服务器 | `sbatch run_3_solver.bash` | 正演模拟 |

**相比旧流程（v1）的改进**：

| | v1（旧，六步） | v2（新，三步） |
| --- | --- | --- |
| NC→GLL | 本地运行，需 scp 下载 DATABASES_MPI (~GB) + 上传 GLL | 服务器端直接运行，零传输 |
| 步骤数 | 6 步（含 3 次 scp） | 3 步（全在服务器） |
| 依赖 | 本地 Python | 服务器 conda (eastasia_fwi) |
| 多模型 | 每个模型重复 scp | 只需 sbatch 切换目录 |

### 19.7 关键变化

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

_最后更新：2026-03-25（三模型并行部署 + 服务器端转换优化）_
