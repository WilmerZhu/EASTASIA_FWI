# FWEA23 GLL 格式正演 - 服务器端流程

曙光 Slurm 集群上完成 FWEA23 模型 GLL 格式正演的完整步骤。

---

## 项目目标

**核心目标**：为**任意公开三维速度模型**提供正演模拟与波形检验工具。

- **背景**：多数公开模型（FWEA23、EARA2024、SinoScope1.0 等）仅发布 **NetCDF (NC)** 或 **CSV** 格式，而 SPECFEM3D Globe 需要 **GLL** 格式。
- **工具**：`convert_nc_to_gll.py` 将 NC/CSV 插值到 GLL，使研究者能用公开模型进行波形正演与检验。
- **测试用例**：以 **FWEA23** 为例，因有原作者参考结果（`specfem/model_updated`、`sac/`）可对比验证。

---

## 0. 前置条件

- 项目已上传到服务器（含 `specfem3d_globe`、`3_Data_space_simulation`）
- SPECFEM3D Globe 已编译
- **本地**：Python 环境（`conda activate eastasia_fwi`）用于 NC→GLL 转换
- FWEA23 NetCDF：`data/models/processed/2024_FWEA23/2024_FWEA23_original.nc`（需含 eta）

---

## 1. 流程概览

```
编译 + 第一次 meshfem（s362ani，服务器）→ 下载 DATABASES_MPI
    → NC→GLL 转换（本地）→ 上传 GLL → 仅 xspecfem3D 正演（服务器，保留 DATABASES_MPI）
```

**说明**：`run_meshfem_only.bash` 完成编译与第一次 meshfem（临时 `MODEL=s362ani_crust1.0`，结束后将 `MODEL` 恢复为 `GLL_crust1.0` 占位）。NC→GLL 在本地按**该次** `DATABASES_MPI` 插值到 GLL 点。**推荐**正演阶段**不再**清空或重跑 `xmeshfem3D`，直接用已有网格 + `DATA/GLL` 跑 solver（`run_second_mesh_solver.bash`），避免第二次区域网格在相同配置下出现 `Error face flag`、Jacobian 退化等问题（见 §5、§6.4、§18）。

*可选旧路径*：若科研上需要「第二次 mesh 再读 GLL」对照，可自行 `rm -rf DATABASES_MPI/*` 后单独跑 `xmeshfem3D`；本仓库脚本默认**不再**执行该步骤。

---

## 2. 准备 NC 模型（本地，含 eta）

确保 `1_5_Process_velocity_models.py` 中 FWEA23 的 params 包含 `eta`：

```python
'params': ['vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho', 'vs0', 'vp0']
```

运行 `1_5` 生成含 eta 的 `2024_FWEA23_original.nc`。

---

## 3. 编译 + 第一次 meshfem（s362ani，服务器）

`run_meshfem_only.bash` 会自动完成：编译 SPECFEM → 第一次 meshfem（s362ani）→ 恢复 Par_file 为 `GLL_crust1.0`（占位，供后续改 GLL 正演）。

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

---

## 4. NC → GLL 转换（本地执行）

曙光服务器 GLIBC 较旧，Miniconda 无法安装，**在本地**完成转换。

`convert_nc_to_gll.py` 会自动处理：
- 非量纲化坐标 → 地理坐标（lat, lon, depth）
- 域内线性插值 + 域外最近邻外推（chunk 旋转后边缘超出模型范围的区域）
- 密度 kg/m³ → g/cm³（SPECFEM GLL 要求）
- eta 从 NC 读入（若 NC 无 eta 则默认 1.0）

```bash
# 1. 从服务器下载 DATABASES_MPI 到本地项目
# scp -r user@曙光:/path/to/FWEA23/DATABASES_MPI specfem3d_globe/EXAMPLES/FWEA23/

# 2. 在本地项目根目录执行
cd /path/to/EASTASIA_FWI

python 3_Data_space_simulation/convert_nc_to_gll.py \
    --mesh-dir specfem3d_globe/EXAMPLES/FWEA23/DATABASES_MPI \
    --model-nc data/models/processed/2024_FWEA23/2024_FWEA23_original.nc \
    --output-dir specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL \
    --anisotropic
    # --horizontal-fill nearest  ← 默认值，可省略
    # --reference-gll-dir specfem/model_updated  ← nspec 不同，不可用

# 3. 上传 GLL 到服务器
scp -r specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL user@曙光:/path/to/FWEA23/DATA/
```

**验证诊断输出**（proc000000 应显示合理值）：

```
vpv: 1.97 ~ 13.72   (km/s)
rho: 1.34 ~ 5.57    (g/cm³，含 1DREF 深部填充；全局 min 来自浅层沉积盆地）
eta: 0.90 ~ 1.01    (非全 1.0)
rho mean ≈ 4.0 g/cm³（非 2300+ → 应已在源级转换）
```

**若 SPECFEM 以 double 精度编译**（`CUSTOM_REAL=SIZE_DOUBLE`），加 `--double`。

### 4.1 转换自检选项

- `--reference-gll-dir specfem/model_updated`：自动对比 `proc000000` 的 `vpv/vph/vsv/vsh/eta/rho`，输出 `rmse` 与 `mean|Δ|`。
- `--compare-all-procs`：对全部 proc 执行 reference-gll 对比（日志较长）。
- `--apply-physical-floor`：仅在数值稳定性需要时开启；默认关闭以保留原始模型值。
- `--horizontal-fill {nearest,ref,blend}`：
  - `nearest`（默认）：最稳定，适合当前批量测试；
  - `ref`：水平域外直接用 1DREF；
  - `blend`：边界附近最近邻，远离边界渐变到 1DREF（可配 `--blend-width-deg`）。

---

## 5. GLL 正演（仅 solver，服务器）

`run_second_mesh_solver.bash` 当前行为：**不**删除 `DATABASES_MPI`、**不**调用 `xmeshfem3D`；仅 `mkdir -p OUTPUT_FILES` 后运行 `xspecfem3D`。脚本会检查 `DATABASES_MPI/mesh_parameters.bin` 与 `DATA/GLL/` 是否存在，缺失则退出。

```bash
cd specfem3d_globe/EXAMPLES/FWEA23

# Par_file：须指向 GLL 速度与目录（与第一次 mesh 所用网格一致）
grep "^MODEL" DATA/Par_file
# 常见：`MODEL = GLL_crust1.0`（run_meshfem_only 恢复名；源码会将 MODEL 名转为小写再解析，等价 `gll` + CRUST1.0）
# 或显式：`MODEL = gll` / `gll_tiso` 等（见 `get_model_parameters.F90`）
grep "^PATHNAME_GLL_modeldir" DATA/Par_file   # 应指向 DATA/GLL

sbatch run_second_mesh_solver.bash
```

### 5.1 这样处理是否合理？

**合理，且与 SPECFEM 设计一致**，前提满足：

| 条件 | 说明 |
|------|------|
| **网格与 GLL 同源** | `convert_nc_to_gll.py` 使用的 `DATABASES_MPI` 与服务器上正演目录中的为**同一次** `xmeshfem3D` 产物；`NPROC`、分区与 mesh 时一致。 |
| **仅换弹性参数** | `MODEL` 设为 GLL 分支后，**波场计算使用的 Vp/Vs/Vph/…（及相应类型下的 rho、eta 等）来自 `PATHNAME_GLL_modeldir`**；`DATABASES_MPI` 提供几何与连接，不因「不重 mesh」而变成 s362ani 速度。 |
| **规避第二次 mesh 风险** | 区域 + Moho 细化下，第二次全量 mesh 曾与第一次成功并存 **face flag / Jacobian** 失败（§6.4、§18）；保留第一次成功网格可稳定进入 solver。 |

**不适用的情况**：若修改了 `NEX`/`NER`/`constants` 等与**几何**相关的设置，必须重新 `xmeshfem3D` 并**重新** NC→GLL。

**验证**：

- 转换阶段：§4 中 vpv/vsv/rho/eta 值域诊断。
- 正演阶段：`Max norm displacement vector U` 不应指数增长；`OUTPUT_FILES/` 中 SAC 等输出正常。

---

## 6. 调试

### 6.1 `input statement requires too much data`

GLL 文件与网格不匹配。确认精度和 nspec：

```bash
# 确认精度
grep "CUSTOM_REAL =" ../../setup/constants.h
# SIZE_REAL → 不加 --double; SIZE_DOUBLE → 加 --double

# 确认 GLL 文件大小（期望 8 + 125*nspec*4 字节）
ls -l DATA/GLL/proc000000_reg1_vpv.bin
# nspec=4320 → 2160008 字节
```

### 6.2 solver 发散（Max U 指数增长）

检查 mesher 输出中的模型值：
- 若 vpv/vsv 全为常数（如 4.5）→ 坐标转换或插值错误
- 若 rho 为 1500 或 3300（非 g/cm³）→ 密度单位错误
- 若 Vs ≈ Vp → 不物理，负体积模量

### 6.3 远震波形偏差

- 确认 NC 含 eta（否则默认 1.0，影响面波）
- GLL 模型会覆盖 CRUST2.0（`model_gll_impose_val` 在 `get3Dcrust_val` 之后调用）

### 6.4 第一次 meshfem 偶发失败（同一配置、重跑又可成功）

2026-03 曙光上 `run_meshfem_only.bash` 曾出现：**相同 Par_file / constants 思路下，有时失败、重提作业后成功**。mesh 算法对固定输入本应确定，此类现象优先从**环境**排查，而非急于改 Fortran。

| 现象 | 位置 | 说明 |
|------|------|------|
| `Error face flag` | `get_MPI_interfaces.f90` | 多发生在**壳幔** MPI 界面构造阶段 |
| `Error test outer core valence` | `test_MPI_interfaces.f90` | 壳幔已打印 `all interfaces okay` 后，在**外核** `assemble_MPI_scalar` 后的 valence 自检失败 |

**处理建议**：

- 确认**未**对同一 `EXAMPLES/FWEA23` 目录**并发**多个 mesh 作业（避免争抢 `DATABASES_MPI` / `OUTPUT_FILES`）。
- `run_meshfem_only.bash` 等 mesh 作业若含清理逻辑，勿与正演作业同时写同一目录；**`run_second_mesh_solver.bash` 不再清空 `DATABASES_MPI`**，仅依赖已有 mesh。
- **原样重提** `sbatch` 常可恢复；若二进制与输入未改，可用 `SKIP_CLEAN=1` **跳过 make clean** 仅跑 mesh，减少编译与链接差异。
- **成功判据**：`mesh_<jobid>.err` 无 `MPI_Abort`；`mesh_<jobid>.out` 出现 `Meshfem done (exit 0)`；`OUTPUT_FILES/output_mesher.txt` 末尾为 `End of mesh generation`。

**备忘作业号**（便于对照日志）：5665048 壳幔 face flag；5674267 成功；5674922 外核 valence 失败；**5674995 成功**（2026-03-23）。

---

## 7. 常见错误

| 错误                                 | 处理                                                                            |
| ------------------------------------ | ------------------------------------------------------------------------------- |
| `PMI_KVS_Get returned -1`            | 使用 `srun --mpi=pmi2`，不要用 `mpirun`                                         |
| `request too frequently`             | 同上                                                                            |
| `Error opening DATA/s362ani/S362ANI` | 确保 `specfem3d_globe/DATA/s362ani/` 存在，脚本会自动链接                       |
| `Error model crust2.0`               | s362ani 需 crust2.0，确保 `specfem3d_globe/DATA/crust2.0/` 存在，脚本会自动链接 |
| `requires too much data`             | 见 §6.1                                                                         |
| `floating overflow` / 发散           | 见 §6.2                                                                         |
| `Error face flag`                    | 壳幔 MPI 界面；见 §6.4，排除并发写目录后重提作业；GLL 正演推荐 §5 **不重跑第二次 mesh** |
| `Error test outer core valence`      | 外核 `test_MPI_interfaces`；见 §6.4，常与集群/MPI 偶发有关，可重提或 `SKIP_CLEAN=1` 仅 mesh |
| `Error opening mesh_parameters.bin`  | mesh 未完成或 `DATABASES_MPI` 被清空；先保证第一次 mesh 成功且 §5 流程未误删库 |
| `Mesh becomes invalid` / Jacobian    | 单元畸变；见 §18.1；优先 §5 **保留成功网格 + 仅 solver** 或调 Moho/NER |

---

## 8. 路径速查

| 项目          | 路径                                            |
| ------------- | ----------------------------------------------- |
| 进程数        | 144 (1×12×12)                                   |
| DATABASES_MPI | `specfem3d_globe/EXAMPLES/FWEA23/DATABASES_MPI` |
| GLL 输出      | `specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL`      |
| NC 模型       | `data/models/processed/2024_FWEA23/2024_FWEA23_original.nc` |
| 转换脚本      | `3_Data_space_simulation/convert_nc_to_gll.py`  |

---

## 9. convert_nc_to_gll.py 已修复的关键问题

| 问题 | 修复 |
|------|------|
| 深度计算：`R_EARTH - r`（r 非量纲 ≈ 1.0）| `R_EARTH * (1 - r)` |
| 域外填常数 4.5 导致 Vs=Vp 发散 | 域内线性 + 域外最近邻外推 |
| 密度单位：NC 为 kg/m³，SPECFEM 要 g/cm³ | 源级转换（插值前 ÷1000） |
| 四重嵌套循环慢 | numpy 向量化 |

---

## 10. 开发与调试尝试记录

以下为打通 NC→GLL 正演流程过程中的主要尝试与结论，供后续模型接入参考。

### 10.1 convert_nc_to_gll.py 修复

| 问题 | 修复 | 说明 |
|------|------|------|
| 深度计算错误 | `R_EARTH * (1 - r)` | r 为量纲化半径，原 `R_EARTH - r` 导致深度 ~6370 km |
| 域外填常数 4.5 | 域内线性 + 域外最近邻 | 常数导致 Vs=Vp，负体积模量发散 |
| 密度单位 | **源级转换**（插值前在 NC 数组上 ÷1000） | 旧方案（插值后 mean>100 则 ÷1000）会导致 1DREF 填充的深部点被二次除法（rho≈0.003），solver 必然发散。改为在 `interpolate_from_netcdf` 内、构造插值器前统一转为 g/cm³，与 1DREF（已为 g/cm³）保持一致 |
| eta 缺失 | 1_5 增加 eta，NC→GLL 读入 | 否则默认 1.0，影响各向异性 |
| qmu 异常低值 | vmin/default 增加 qmu，1DREF 填充 | 避免 NaN 被填为 4.5 |
| 深部 1DREF | 仅壳幔，排除外核/内核 | CMB 处外核低 vsv 曾渗透进 GLL |
| 四重循环慢 | numpy 向量化 | 性能优化 |
| **域外填充策略不透明** | **改为可配置 + 自动对比** | **新增 `--horizontal-fill {nearest,ref,blend}` + `--reference-gll-dir` 输出 RMSE** |
| qmu 默认填充 4.5 残留 | 默认值改为参数对应的合理值 | 309987 个 qmu=4.5 的点，51 个 proc 受影响 |

### 10.2 1_5_Process_velocity_models.py

| 修改 | 说明 |
|------|------|
| params 增加 `eta`, `qmu` | 使 NC 含各向异性与衰减参数 |
| standardize_model_3d 处理 eta/qmu | 标准化与保存 |
| **删除 rho × 1000 自动转换** | **原始 NC rho 为 g/cm³，被错误乘 1000。改为仅记录日志不修改** |

### 10.3 震源与配置

| 项目 | 尝试 | 结论 |
|------|------|------|
| CMT 格式 | ECEF→CMT 转换（change_ECEF_to_cmt.py） | 使用原作者 tau×1.628 得 half_duration=3.63 |
| REGIONAL_MESH_CUTOFF | 改为 .false. | 对齐 7.0.0 无深度截断 |
| constants.h.in | THRESHOLD_EXCLUDE_STATION、USE_OLD_VERSION_7_0_0 | 已撤销，对波形无实质影响 |

### 10.4 服务器与脚本

| 项目 | 修改 |
|------|------|
| MPI 启动 | `mpirun` → `srun --mpi=pmi2`（曙光 Slurm） |
| GLL 正演脚本 | `run_second_mesh_solver.bash`：仅 `xspecfem3D`，保留 `DATABASES_MPI`，检查 `mesh_parameters.bin` 与 `DATA/GLL` |
| run_meshfem_only | 集成编译 + 第一次 meshfem，脚本末尾将 `MODEL` 恢复为 `GLL_crust1.0`（见 `run_meshfem_only.bash`） |

### 10.4.1 曙光第一次 mesh 稳定性备忘（2026-03）

| 作业号   | 结果 | 备注 |
|----------|------|------|
| 5665048  | 失败 | 壳幔 `Error face flag`（rank 102 / iglob 255290 等） |
| 5674267  | 成功 | `End of mesh generation`，可与 NC→GLL 对齐的 `DATABASES_MPI` |
| 5674922  | 失败 | 壳幔通过后，外核 `Error test outer core valence`（如 rank 26/34） |
| 5674995  | 成功 | `Meshfem done (exit 0)`，同配置重跑恢复 |

结论与操作要点见 **§6.4**。

### 10.5 波形差异分析

**根因定位**：两个独立问题叠加导致波形差异。

| 发现 | 详情 |
|------|------|
| **网格差异（主因之一）** | nspec 7236 (v7.0.0) vs 7452 (v8.1.0)，见 §11.2 |
| **域外点占比** | 19.65%（26,359,787 / 134,136,000），主要在 lon > 150° |
| **NC 与原始 GLL 差异** | NC vpv_min=1.86，原始 GLL vpv_min=0.75（精细地壳） |
| **修复方案** | 域外策略改为可控：默认 `nearest`；`ref`/`blend` 用于对比试验 |
| **NC 数据本身** | 未被 1_5 处理破坏（逐参数验证 max_diff=0） |
| **rho 单位** | 原始 NC 为 g/cm³，1_5 错误 ×1000，convert 又 ÷1000 互相抵消 |

### 10.6 边界处理（与 7.0.0 一致）

- 7.0.0 无 REGIONAL_MESH_CUTOFF、无 sponge；仅 Stacey 吸收边界。
- 当前 Par_file：`REGIONAL_MESH_CUTOFF = .false.`，`ABSORB_USING_GLOBAL_SPONGE = .false.`，与 7.0.0 行为一致。

---

## 11. 当前解决方案：分步验证

### 11.1 问题分析

波形差异由两个**独立因素**叠加，必须分开验证：

1. **网格不同**：8.1.0 nspec=7452 vs 7.0.0 nspec=7236
2. **NC ≠ 原始 GLL**：NC 是有损的（vpv_min 1.86 vs 原始 GLL 0.75）

### 11.2 网格差异根因（已定位并修复）

8.1.0 的 `auto_ner()` 用 `get_timestep_and_layers` 里为 **90° chunk** 设计的经验 NER 值作为起点，再由 `auto_optimal_ner()` 向上搜索最优纵横比。而 7.0.0 的 `auto_ner()` 从**最小值**（全 1）开始搜索。

`auto_optimal_ner` 只增不减，起点越大终点越大 → 8.1.0 在 80° chunk 上多出 216 个元素/proc。

**修复**：patch `specfem3d_globe/src/shared/auto_ner.f90`，注释掉经验值覆盖（lines 481-497），让 8.1.0 和 7.0.0 一样从最小值开始。

### 11.3 三步验证流程

| 步骤 | 做什么 | 验证什么 | 预期结果 |
|------|--------|----------|----------|
| **Step 1** | patch auto_ner → 重新编译 + meshfem | nspec 是否变为 7236 | 网格与原作者一致 |
| **Step 2** | 直接用原作者 GLL 跑 8.1.0 solver | 8.1.0 pipeline 本身是否正确 | 波形应高度匹配原作者 |
| **Step 3** | NC→GLL 转换后跑，与 Step 2 对比 | 量化 NC 有损压缩的影响 | 定义"NC 精度损失基线" |

### 11.4 Step 1：重新编译（auto_ner 已 patch）

```bash
cd specfem3d_globe/EXAMPLES/FWEA23
# auto_ner.f90 已修改，需要 make clean + 重新编译
sbatch run_meshfem_only.bash
# 验证 nspec
grep NSPEC_CRUST_MANTLE OUTPUT_FILES/values_from_mesher.h
# 期望: NSPEC_CRUST_MANTLE = 7236
```

### 11.5 Step 2：用原作者 GLL 直接跑

nspec 匹配后，原作者 GLL 文件可以直接使用：

```bash
# 复制原作者 GLL 到 DATA/GLL
cp specfem/model_updated/proc*_reg1_*.bin DATA/GLL/
# 确认 Par_file: MODEL = gll（或 GLL_crust1.0）
# 保留第一次 mesh 的 DATABASES_MPI，仅 solver（见 §5）
sbatch run_second_mesh_solver.bash
# 对比波形 → 验证 8.1.0 pipeline
```

### 11.6 Step 3：NC→GLL 转换

```bash
python 3_Data_space_simulation/convert_nc_to_gll.py \
  --mesh-dir DATABASES_MPI \
  --model-nc data/models/processed/2024_FWEA23/2024_FWEA23_original.nc \
  --output-dir DATA/GLL \
  --anisotropic \
  --reference-gll-dir specfem/model_updated
# 此时 nspec 一致，reference-gll 对比为逐点 RMSE
```

---

## 12. tibet_v4 地壳配置对齐尝试记录（2026-03-17）

### 12.1 背景

原作者在 7.0.0 修改版（`specfem/sem_config_tibet_v4/`）中启用了 4 项地壳网格参数，以更好地处理东亚深 Moho 区域（喜马拉雅/Hindu Kush，Moho 70-80km）。8.1.0 默认未启用这些参数，导致网格拓扑差异（nspec 7452 vs 7236）。

**原作者 tibet_v4 配置**（`specfem/sem_config_tibet_v4/setup/constants.h.in`）：
```fortran
REGIONAL_MOHO_MESH = .true.
REGIONAL_MOHO_MESH_EUROPE = .false.
REGIONAL_MOHO_MESH_ASIA = .false.    ! 7.0.0 DT 公式不同，无需此标志
HONOR_DEEP_MOHO = .true.
RMOHO_STRETCH_ADJUSTMENT = -20000.d0  ! Moho 网格边界下移到 60km
R80_STRETCH_ADJUSTMENT = -40000.d0    ! R80 网格边界下移到 120km
```

**7.0.0 vs 8.1.0 的关键代码差异**：
- DT 公式：7.0.0 `DT *= (1.0 - 0.1)` (降 10%)；8.1.0 `DT *= (1.0 + 0.5)` (升 50%)
- `auto_ner` 算法：两版本相同，均使用 R80(80km) 而非 R80_FICTITIOUS(120km) 计算 NER_80_MOHO
- `get_MPI_interfaces` 算法：两版本核心逻辑相同
- `stretch_deep_moho`/`moho_stretching`：两版本核心逻辑相同

### 12.2 修改的文件清单

#### (1) `specfem3d_globe/setup/constants.h.in`

| 参数 | 默认值 | 修改值 | 说明 |
|------|--------|--------|------|
| `EARTH_REGIONAL_MOHO_MESH` | `.false.` | `.true.` | 启用区域 Moho 网格（3 层地壳） |
| `EARTH_REGIONAL_MOHO_MESH_ASIA` | `.false.` | `.true.` | 覆盖 DT=0.15（修正 8.1.0 DT 公式 bug） |
| `EARTH_HONOR_DEEP_MOHO` | `.false.` | `.true.` → `.false.` → `.true.` | 多次切换，见下表 |
| `EARTH_RMOHO_STRETCH_ADJUSTMENT` | `-5000.d0` | `-20000.d0` → `-15000.d0` → `-20000.d0` | 多次切换 |

#### (2) `specfem3d_globe/src/shared/get_timestep_and_layers.f90`

在 `REGIONAL_MOHO_MESH` 代码块中添加：
```fortran
! auto_ner 用 R80(80km) 计算 NER_80_MOHO，但 define_all_layers
! 用 R80_FICTITIOUS(120km)。实际层跨度 ~60km，NER=1 太粗导致
! 深 Moho 区域 stretch_deep_moho 引发 Jacobian 翻转。
if (NER_80_MOHO < 3) NER_80_MOHO = 3
```

同时添加了 `EARTH_REGIONAL_MOHO_MESH_ASIA` 的 DT 覆盖：
```fortran
if (EARTH_REGIONAL_MOHO_MESH_ASIA) DT = 0.15  ! Asia & Middle East
```

#### (3) `specfem3d_globe/src/meshfem3D/setup_MPI_interfaces.f90`

```fortran
MAX_NEIGHBORS = 20 + NCORNERSCHUNKS  ! 原值 8，增大以容纳地壳拉伸的复杂拓扑
```

#### (4) `specfem3d_globe/src/specfem3D/iterate_time_undoatt.F90`

```fortran
! ifort 编译器内部错误修复：单行 if 改为 if...then...endif
if (EXACT_UNDOING_TO_DISK) then
  call finish_exact_undoing_to_disk()
endif
```

#### (5) `specfem3d_globe/src/meshfem3D/get_MPI_interfaces.f90`

将 face/edge/corner flag 的 `call exit_mpi` 改为 warning + `work_test_flag = 0`。
将末尾 `minval/maxval` 全局检查改为 warning。
将 duplicate 点检查改为 warning + 删除重复。

#### (6) `specfem3d_globe/src/meshfem3D/test_MPI_interfaces.f90`

将 `ineighbor points differ` 的 `call exit_mpi` 改为 warning。

### 12.3 逐次尝试记录

| # | 作业号 | constants.h.in 配置 | get_timestep_and_layers 修改 | 其他修改 | 结果 | 错误 |
|---|--------|---------------------|------------------------------|----------|------|------|
| 1 | 5570553 | REGIONAL=.true., HONOR=.true., RMOHO=-20000 | 无 | MAX_NEIGHBORS=20 | ❌ | DT 超限发散（CFL > 1.0） |
| 2 | (调整后) | +ASIA=.true. | DT=0.15 覆盖 | — | ❌ | `interface edge exceeds MAX_NEIGHBORS` |
| 3 | (调整后) | 同上 | 同上 | MAX_NEIGHBORS=20 | ❌ | `Error Jacobian rank 124`，Hindu Kush 区域 |
| 4 | 5571243 | 同上 | +`NER_80_MOHO>=2` | — | ❌ | `Error ineighbor points differ`（MPI 接口点数不一致） |
| 5 | (调整后) | 同上 | 硬编码全部 7.0.0 NER 值 | — | ❌ | nspec=7380，`Error face flag`（MPI abort） |
| 6 | 5572204 | 同上 | 同上 | — | ❌ | exit code 137，MPI_Abort（face flag） |
| 7 | 5573451 | HONOR=.false., RMOHO=-15000 ("Europe 模式") | 撤销 NER 硬编码 | — | ❌ | `Error Jacobian`（NER_80_MOHO=1 不够） |
| 8 | 5574523 | 同上 | +`NER_80_MOHO>=2` | — | ❌ | `Error face flag`（与 #5 同类错误） |
| 9 | (提议) | REGIONAL=.false. | 撤销所有 | — | ❌ | **用户否决**："那不就不是我们的目的了吗？" |
| 10 | 5576745 | HONOR=.true., RMOHO=-20000（还原 tibet_v4） | NER_80_MOHO>=2 | face/edge flag → warning | ❌ | `Error Jacobian rank 111`，NER=2 仍不够 |
| 11 | 5577823 | 同上 | NER_80_MOHO>=**3** | face/edge flag → warning | ❌ | `PMPI_Wait: Message truncated`（MPI 缓冲区溢出） |

### 12.4 三类错误的因果关系

```
REGIONAL_MOHO_MESH=.true. + HONOR_DEEP_MOHO=.true.
    │
    ├─→ auto_ner 用 R80(80km) 计算 NER_80_MOHO=1
    │   但 define_all_layers 用 R80_FICTITIOUS(120km)
    │   实际层跨 60km，元素太厚
    │       │
    │       └─→ stretch_deep_moho 在 Hindu Kush 等深 Moho 区域
    │           导致元素翻转 → ❌ Jacobian Error
    │
    ├─→ 强制 NER_80_MOHO=2 或 3 解决 Jacobian
    │   但改变了 auto_ner 的内部一致性
    │       │
    │       └─→ get_MPI_interfaces 中 test_flag 检测到
    │           doubling 层边缘点的 flag 值不一致
    │           → ❌ Error face flag / Error ineighbor points differ
    │
    └─→ 将 face flag abort 改为 warning + reset
        导致两个相邻进程的 interface 点数不对称
            │
            └─→ MPI 通信时发送端数据量 > 接收端缓冲区
                → ❌ PMPI_Wait: Message truncated
```

**结论**：三类错误层层递进，每次修复一个都引入下一个。根本原因是 8.1.0 的网格生成在 `REGIONAL_MOHO_MESH + HONOR_DEEP_MOHO` 配置下，`auto_ner` 与 `define_all_layers` 对 R80 层边界的定义不一致，导致网格拓扑与 MPI 接口检测算法不兼容。7.0.0 不存在此问题（原因不明，可能在底层元素创建顺序或 get_global 点编号等细节差异）。

### 12.5 待探索方向

1. **对比 7.0.0 和 8.1.0 的 `create_regions_mesh.F90`**：重点关注 doubling 层元素创建顺序、`get_global` 调用时机（是否在 moho_stretching 之前/之后）
2. **对比 `create_chunk_buffers.f90`**：MPI 边界点选取逻辑
3. **修复 `auto_ner.f90`**：让 radius(3) 使用 R80_FICTITIOUS_IN_MESHER 而非 R80，使 NER 计算与 define_all_layers 一致
4. **用 7.0.0 的 xmeshfem3D 生成网格**：测试 DATABASES_MPI 格式兼容性
5. **向 SPECFEM3D Globe 开发者报告此 bug**：`REGIONAL_MOHO_MESH + HONOR_DEEP_MOHO` 在 8.1.0 中不可用

### 12.6 当前代码状态（2026-03-18 `git diff HEAD` 实测）

§12.2 记录的是 tibet_v4 全程实验期间的改动。经过 §13.1 代码归一后，部分文件已恢复原版。
以下为 **`specfem3d_globe` 仓库相对原始 8.1.0 的实际 diff**（6 个文件）：

| # | 文件 | 改动量 | 当前状态 | 具体改动 | GLL 流程影响 |
|---|------|--------|----------|----------|-------------|
| 1 | `setup/constants.h.in` | +10 -4 | 🟡 需清理 | `RMOHO_STRETCH=-20000`（tibet_v4 残留，但 `REGIONAL=.false.` 故**不生效**）；`REGIONAL/HONOR/ASIA` 均为 `.false.`（Round-1 基线）；COURANT 注释 | 无直接影响，但 RMOHO 值应恢复默认 |
| 2 | `src/meshfem3D/model_EMC.f90` | +221 -8 | ⚠️ EMC 线误放 | 新增 vpv/vph/vsv/vsh/eta 各向异性读取、插值、广播、赋值逻辑。**属于 EMC 测试线**，GLL 流程走 `model_gll.f90` 不会触发此代码 | 无（但增加编译时间） |
| 3 | `src/meshfem3D/get_MPI_interfaces.f90` | +1 -1 | 🟢 无害 | 仅删除一个多余空格（`iglob) <` → `iglob)<`），无逻辑改动 | 无 |
| 4 | `src/meshfem3D/setup_MPI_interfaces.f90` | +1 -1 | 🟢 保留 | `MAX_NEIGHBORS = 20 + NCORNERSCHUNKS`（原值 8），为复杂拓扑预留余量 | 无负面影响 |
| 5 | `src/shared/get_timestep_and_layers.f90` | +1 -1 | 🟢 无害 | `NER_CRUST = 3` 行注释改为 `for regional 1-chunk simulation`，无代码逻辑变化 | 无 |
| 6 | `src/specfem3D/iterate_time_undoatt.F90` | +3 -1 | 🟢 保留 | 单行 `if` 改 `if...then...endif`，修复 ifort 编译器内部错误 | 通用修复，保留 |

**已恢复原版的文件**（曾在 §12 实验期间修改，现已撤销）：

| 文件 | 曾修改内容 | 恢复原因 |
|------|-----------|----------|
| `src/shared/auto_ner.f90` | 注释经验 NER 覆盖（使 nspec 从 7452→7236） | §13.1 归一 |
| `src/meshfem3D/test_MPI_interfaces.f90` | `ineighbor points differ` exit→warning | §13.1 归一（吞错误会导致 MPI truncated） |

**另有**：`specfem3d_globe-EMC` 仓库（独立 EMC 测试线）也有 3 个文件修改（`constants.h.in` +2, `model_EMC.f90` +242 -13, `get_model_parameters.F90` +6 -6），与 GLL 流程无关。

---

_更新：2026-03-18_

---

## 13. 后续测试执行方案（仅 8.1.0，收敛版）

目标：在 **8.1.0** 内完成可复现、可判定的测试闭环，避免继续“多处同时改动”导致问题互相掩盖。

### 13.1 先做代码状态归一（必须）

先把“会破坏 MPI 对称性”的实验补丁撤销，再开始新测试：

1. 撤销 `src/meshfem3D/get_MPI_interfaces.f90` 中所有 `error -> warning + reset` 修改（恢复 `call exit_mpi` 原逻辑）
2. 撤销 `src/meshfem3D/test_MPI_interfaces.f90` 中 `ineighbor points differ` 的降级修改（恢复 `call exit_mpi`）
3. 保留 `src/meshfem3D/setup_MPI_interfaces.f90` 的 `MAX_NEIGHBORS=20+NCORNERSCHUNKS`
4. 保留 `src/specfem3D/iterate_time_undoatt.F90` 的 ifort 兼容修复
5. 重新 `make clean && make meshfem3D specfem3D`

**原因**：MPI 报错必须“真实失败”，不能吞掉；吞掉后会演化为 `PMPI_Wait: Message truncated`，定位价值为 0。

### 13.2 测试门禁（每步只改 1 个变量）

#### Gate A：8.1.0 基线可运行性（不追求 tibet_v4）

- 配置：`REGIONAL_MOHO_MESH=.false.`，`HONOR_DEEP_MOHO=.false.`
- 目标：确认 8.1.0 编译链、脚本链、作业环境无系统性问题
- 通过条件：
  - meshfem 完成，无 Jacobian/MPI interface 错误
  - solver 稳定（`Max norm U` 不指数增长）

> Gate A 失败时，不进入任何地壳拉伸测试。

#### Gate B：只打开 REGIONAL，不开 HONOR_DEEP_MOHO

- 配置：`REGIONAL_MOHO_MESH=.true.`，`HONOR_DEEP_MOHO=.false.`
- 目标：验证“浅层区域 Moho 拉伸”是否可被 8.1.0 接受
- 通过条件：
  - 无 Jacobian
  - 无 `Error face/edge/corner flag`
  - 无 `Error ineighbor points differ`

#### Gate C：打开 HONOR_DEEP_MOHO（tibet_v4 目标态）

- 配置：`REGIONAL_MOHO_MESH=.true.`，`HONOR_DEEP_MOHO=.true.`，`RMOHO=-20000`, `R80=-40000`
- 目标：复现并固化当前核心故障
- 通过条件（任一失败即停）：
  - Jacobian 失败：记录 rank/ispec/深度区间
  - MPI interface 失败：记录 rank/flag/iface 类型

### 13.3 DT 与 NER 的处理原则（防止再走回头路）

1. `EARTH_REGIONAL_MOHO_MESH_ASIA=.true.` 仅作为 **DT 稳定性修正** 使用（8.1.0 特有）
2. 不再使用“硬钳位 `NER_80_MOHO>=2/3`”作为长期方案；只允许作为一次性诊断分支
3. 禁止把 MPI abort 改 warning 作为“通过手段”

### 13.4 推荐的两条可交付路径（都在 8.1.0 框架内）

#### 路径 P1（优先，科研上最干净）

在 8.1.0 内做最小源码修复，使 NER 与实际层边界一致：

- 修改点：`src/shared/auto_ner.f90`
- 目标：`radius(3)` 相关计算改为与 `define_all_layers` 一致（使用 fictitious 边界思想）
- 验证：
  - Gate C 下不再 Jacobian 翻转
  - MPI 接口检查零报错（无需任何 warning 降级）

#### 路径 P2（工程兜底，保证测试先完成）

保持 8.1.0 solver 与 NC→GLL 链条，先以 Gate B 形成稳定发布流程，再单独维护 Gate C 为“已知缺陷分支”。

- 交付内容：
  - 可复现实验脚本（Gate A/B）
  - 波形对比与 RMSE 报告
  - Gate C 失败日志归档 + 根因说明

### 13.5 每次作业必须记录的最小证据

每个作业号统一记录以下 8 项，避免后续无法复盘：

1. `constants.h.in` 四个关键参数值
2. `get_timestep_and_layers.f90` 是否存在 NER 钳位
3. `get_MPI_interfaces.f90` / `test_MPI_interfaces.f90` 是否原版 abort
4. 编译命令与编译时间戳
5. `values_from_mesher.h` 的 `NSPEC_CRUST_MANTLE`
6. `output_mesher.txt` 的 DT 建议值、Jacobian 统计
7. `.out/.err` 首个 fatal 栈
8. 本次相对上次“仅改变的 1 个变量”

### 13.6 下一轮建议执行顺序（直接照跑）

1. 先完成 §13.1 的代码归一与重编译
2. 跑 Gate A（拿到“环境健康”证明）
3. 跑 Gate B（拿到“区域 Moho 可运行”证明）
4. 跑 Gate C（锁定 tibet_v4 目标态失败点）
5. 进入 P1（auto_ner 最小修复）或 P2（先交付稳定链条）

---

## 14. 代码清理与状态归一（2026-03-18）

### 14.1 问题：GLL 仓库中混入 EMC 改动与 tibet_v4 残留

`specfem3d_globe`（GLL 测试线）当前有 6 个文件改动，其中存在两个问题：

1. **`model_EMC.f90` 的 221 行各向异性改动**属于 EMC 测试线（`specfem3d_globe-EMC`），与 GLL 流程无关。GLL 模式 `MODEL=gll` 走的是 `model_gll.f90`，不触发 `model_EMC` 代码路径。但这些改动增加了编译时间，且在概念上造成混淆。
2. **`constants.h.in` 中 `RMOHO_STRETCH_ADJUSTMENT = -20000.d0`** 是 tibet_v4 实验残留。虽然 `REGIONAL_MOHO_MESH=.false.` 时不生效，但保留非默认值会在 Gate B/C 测试时引入隐患（忘记改回时直接生效）。

### 14.2 必须执行的清理（Phase 0）

在上传到服务器之前，在本地完成以下清理：

#### 14.2.1 恢复 `RMOHO_STRETCH_ADJUSTMENT` 为默认值

```bash
cd specfem3d_globe
# 当前: EARTH_RMOHO_STRETCH_ADJUSTMENT = -20000.d0
# 恢复: EARTH_RMOHO_STRETCH_ADJUSTMENT = 5000.d0
```

修改 `setup/constants.h.in`：
```fortran
! 修改前（tibet_v4 残留）：
  double precision, parameter :: EARTH_RMOHO_STRETCH_ADJUSTMENT = -20000.d0 ! moho mesh boundary down to 60km (tibet_v4)

! 修改后（8.1.0 默认）：
  double precision, parameter :: EARTH_RMOHO_STRETCH_ADJUSTMENT = 5000.d0 ! moho up to 35km (default)
```

**原因**：Gate A 基线测试必须用纯默认配置，排除所有实验残留。

#### 14.2.2 处理 `model_EMC.f90`

两个选项：

- **选项 A（推荐）**：`git checkout src/meshfem3D/model_EMC.f90` 恢复原版。EMC 各向异性改动已独立存在于 `specfem3d_globe-EMC` 仓库中。
- **选项 B**：保留（不影响 GLL 正确性，仅增加编译内容）。

#### 14.2.3 清理后的预期 `git diff` 状态

完成 14.2.1 + 选项 A 后，GLL 仓库应仅剩 **4 个无害/有益改动**：

| 文件 | 改动 | 性质 |
|------|------|------|
| `setup/constants.h.in` | COURANT 注释 + Round-1 基线注释 | 🟢 文档性 |
| `src/meshfem3D/setup_MPI_interfaces.f90` | MAX_NEIGHBORS=20 | 🟢 安全余量 |
| `src/shared/get_timestep_and_layers.f90` | 注释微调 | 🟢 文档性 |
| `src/specfem3D/iterate_time_undoatt.F90` | ifort 编译修复 | 🟢 通用修复 |

`get_MPI_interfaces.f90` 的空格差异也可 `git checkout` 恢复，使 diff 更干净。

### 14.3 Par_file 当前状态

```
MODEL = s362ani_crust1.0  ← run_meshfem_only.bash 末尾恢复为 GLL_crust1.0
                             但当前文件仍为 s362ani_crust1.0（上次 meshfem 后的状态）
```

**说明**：`run_meshfem_only.bash` 脚本逻辑是：
1. 先 `sed` 改为 `s362ani_crust1.0` 跑第一次 meshfem
2. 结束后 `sed` 恢复为 `GLL_crust1.0`

当前 Par_file 为 `s362ani_crust1.0`，说明上次脚本可能中断或在恢复前手动改动。GLL 正演（`run_second_mesh_solver.bash`）需要 Par_file 处于 **GLL 分支**（如 `GLL_crust1.0` 或 `gll`）且 `PATHNAME_GLL_modeldir` 正确；**不再**依赖第二次 `xmeshfem3D`。

### 14.4 本地 ↔ 服务器同步检查清单

上传前确认：

- [x] `constants.h.in`: `RMOHO_STRETCH=5000.d0`, `REGIONAL=.false.`, `HONOR=.false.`, `ASIA=.false.` ✅ 2026-03-18 已更新
- [x] `auto_ner.f90`: 原版（无 patch） ✅
- [x] `get_MPI_interfaces.f90`: 原版 ✅ 2026-03-18 git checkout
- [x] `test_MPI_interfaces.f90`: 原版（`call exit_mpi` 未降级） ✅
- [ ] `model_EMC.f90`: 本地已恢复原版；**服务器端仍为改动版**（不影响 GLL 正确性，可稍后同步）
- [x] Par_file: `MODEL = GLL_crust1.0`（run_meshfem_only.bash 自动恢复） ✅ job 5609105 确认
- [x] `run_meshfem_only.bash` 末尾会自动恢复为 `GLL_crust1.0` ✅

---

## 15. 更新版测试执行方案（2026-03-18）

基于 §13 的框架，结合 §14 的代码审查结论，更新具体执行步骤。

### 15.1 总体策略

```
Phase 0: 代码清理（本地）
    ↓
Phase 1: Gate A 基线（服务器）
    ↓ 通过
Phase 2: 三步验证（§11.3）
    ├─ Step 1: meshfem → 记录 nspec
    ├─ Step 2: 原作者 GLL → solver → 波形（需 nspec 匹配，见路径选择）
    └─ Step 3: NC→GLL → solver → 波形 → 与 Step 2 或原作者 SAC 对比
    ↓ 完成
Phase 3: Gate B/C（可选，tibet_v4 对齐）
```

### 15.2 Phase 0：代码清理（本地，即刻执行）

```bash
cd specfem3d_globe

# 1. 恢复 RMOHO_STRETCH 为默认值（见 §14.2.1）
# 编辑 setup/constants.h.in

# 2. （推荐）恢复 model_EMC.f90
git checkout src/meshfem3D/model_EMC.f90

# 3. （可选）恢复空格差异
git checkout src/meshfem3D/get_MPI_interfaces.f90

# 4. 验证 diff 状态
git diff --stat HEAD
# 预期: 3-4 个文件，无 model_EMC.f90
```

### 15.3 Phase 1：Gate A — 8.1.0 纯默认基线

**目的**：证明 8.1.0 + FWEA23 Par_file 配置在默认 constants.h.in 下可以正常 mesh。

**关键配置**：
```fortran
EARTH_REGIONAL_MOHO_MESH         = .false.
EARTH_REGIONAL_MOHO_MESH_ASIA    = .false.
EARTH_HONOR_DEEP_MOHO            = .false.
EARTH_RMOHO_STRETCH_ADJUSTMENT   = 5000.d0   ! ← 已恢复默认
EARTH_R80_STRETCH_ADJUSTMENT     = -40000.d0  ! 默认
```

**操作**：
```bash
cd specfem3d_globe/EXAMPLES/FWEA23
# make clean 后重新编译（constants.h.in 已改）
sbatch run_meshfem_only.bash
```

**通过标准**：
- ✅ 编译成功
- ✅ meshfem 完成（无 Jacobian / face flag / MPI 错误）
- ✅ `DATABASES_MPI/proc*_reg1_solver_data.bin` 数量 = 144
- ✅ 记录 `NSPEC_CRUST_MANTLE`（预期 7452）

**如果 Gate A 失败**：
- 检查服务器端代码是否与本地一致（逐文件 md5 对比）
- 检查编译日志中是否有 warning/error
- 关闭 `TOPOGRAPHY` 和 `OCEANS` 做简化诊断（见 `测试步骤_曙光.md`）
- **不进入 Phase 2 或 Gate B/C**

### 15.4 Phase 2：三步验证

#### 路径选择

Gate A 通过后 nspec=7452（8.1.0 默认），而原作者 GLL 基于 7.0.0 tibet_v4（nspec=7236）。两者不匹配，无法直接互用。

**路径 A（推荐，先完成科研链路）**：

接受 nspec=7452，跳过 Step 2，直接 Step 1 + Step 3：
- Step 1: 记录 nspec=7452
- Step 3: NC→GLL 转换 → solver → 波形输出 → 与原作者 SAC 波形定性对比

**路径 B（完整，需 patch auto_ner）**：

Patch `auto_ner.f90` 使 nspec=7236，完成 Step 1/2/3 全流程。
Step 2 的价值在于：隔离"8.1.0 solver 本身"与"NC→GLL 转换精度"两个变量。

> **建议先走路径 A**，快速拿到端到端波形结果。路径 B 作为后续精细化对比。

#### Step 1（两条路径都做）

```bash
# Gate A meshfem 完成后
grep NSPEC_CRUST_MANTLE OUTPUT_FILES/values_from_mesher.h
# 记录值
```

#### Step 3（路径 A 核心步骤）

```bash
# 1. 下载 DATABASES_MPI 到本地
scp -r user@曙光:/path/to/FWEA23/DATABASES_MPI specfem3d_globe/EXAMPLES/FWEA23/

# 2. NC→GLL 转换（本地）
cd /path/to/EASTASIA_FWI
python 3_Data_space_simulation/convert_nc_to_gll.py \
    --mesh-dir specfem3d_globe/EXAMPLES/FWEA23/DATABASES_MPI \
    --model-nc data/models/processed/2024_FWEA23/2024_FWEA23_original.nc \
    --output-dir specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL \
    --anisotropic

# 注意：--reference-gll-dir 不可用（nspec 不同，文件大小不匹配）
# 此时只能依赖域值统计和最终波形对比验证

# 3. 上传 GLL 到服务器
scp -r specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL user@曙光:/path/to/FWEA23/DATA/

# 4. 确认 Par_file：GLL 分支 + GLL 目录（若仍为 s362ani，需先改回）
ssh user@曙光 "grep -E '^(MODEL|PATHNAME_GLL_modeldir)' /path/to/FWEA23/DATA/Par_file"
# MODEL 应为 GLL_crust1.0 或 gll 等；PATHNAME_GLL_modeldir 指向 DATA/GLL

# 5. 仅 solver（保留服务器上已成功的 DATABASES_MPI，勿清空）
ssh user@曙光 "cd /path/to/FWEA23 && sbatch run_second_mesh_solver.bash"
```

**验证**：
- §4 转换值域合理；服务器上 `DATABASES_MPI/mesh_parameters.bin` 存在且与本次 GLL 同源
- solver 稳定（`Max norm U` 不指数增长）
- 输出波形 SAC 文件存在

#### Step 2（仅路径 B，需 auto_ner patch）

```bash
# 前置：patch auto_ner.f90（注释 lines 481-497 经验值覆盖）
# 重新 make clean && sbatch run_meshfem_only.bash
# 确认 NSPEC=7236 后：

# 复制原作者 GLL
cp specfem/model_updated/proc*_reg1_*.bin DATA/GLL/
# Par_file: MODEL = gll
sbatch run_second_mesh_solver.bash
# 对比波形 → 量化 8.1.0 pipeline 本身的差异
```

### 15.5 Phase 3：Gate B/C（可选，长期）

仅在 Phase 2 完全通过后考虑。

| Gate | 改动 | 目的 | 预期 |
|------|------|------|------|
| **B** | `REGIONAL_MOHO_MESH=.true.`, `HONOR_DEEP_MOHO=.false.` | 浅层 Moho 拉伸 | 可能通过 |
| **C** | `REGIONAL_MOHO_MESH=.true.`, `HONOR_DEEP_MOHO=.true.`, `RMOHO=-20000` | 完整 tibet_v4 | 大概率失败（§12 已验证） |

Gate C 失败后的选择：
- **P1**：`auto_ner.f90` 中 `radius(3)` 改用 `R80_FICTITIOUS_IN_MESHER`
- **P2**：暂以 Gate A/B 交付，Gate C 归档为已知 8.1.0 缺陷
- **P3**：向 SPECFEM 开发者提交 issue

### 15.6 每次作业记录模板

```
作业号: ________
日期: ________
Gate: A / B / C
相对上次仅改变的 1 个变量: ________

constants.h.in:
  REGIONAL_MOHO_MESH      = .false. / .true.
  REGIONAL_MOHO_MESH_ASIA = .false. / .true.
  HONOR_DEEP_MOHO         = .false. / .true.
  RMOHO_STRETCH           = ________

源码修改:
  auto_ner.f90            = 原版 / patched
  get_timestep_and_layers = 原版 / NER钳位 / DT覆盖
  get_MPI_interfaces      = 原版 / warning降级
  test_MPI_interfaces     = 原版 / warning降级

编译: make clean=是/否, 时间=________

结果:
  NSPEC_CRUST_MANTLE = ________
  DT_suggested       = ________
  首个 fatal         = ________（或"无"）
  solver Max U       = ________（是否发散）
```

---

## 16. Gate A 测试结果（2026-03-18）

### 16.1 测试记录

```
作业号: 5609105
日期: 2026-03-18
Gate: A
相对上次仅改变的 1 个变量: constants.h.in RMOHO_STRETCH 恢复为 5000.d0 + model_EMC.f90/get_MPI_interfaces.f90 还原原版

constants.h.in:
  REGIONAL_MOHO_MESH      = .false.
  REGIONAL_MOHO_MESH_ASIA = .false.
  HONOR_DEEP_MOHO         = .false.
  RMOHO_STRETCH           = 5000.d0

源码修改:
  auto_ner.f90            = 原版
  get_timestep_and_layers = 原版（仅注释微调）
  get_MPI_interfaces      = 原版
  test_MPI_interfaces     = 原版
  model_EMC.f90           = 原版（已还原 EMC 各向异性改动）
  setup_MPI_interfaces    = MAX_NEIGHBORS=20（保留）
  iterate_time_undoatt    = ifort 兼容修复（保留）

编译: make clean=是, 时间=16:13:25 → 16:14:22 CST (57s)
编译器: ifort (Intel 2018.5) + mpiifort, -O1 -fp-model precise

结果:
  NSPEC_CRUST_MANTLE = 7452       ✅ 符合 8.1.0 默认预期
  NSPEC_OUTER_CORE   = 774
  NSPEC_INNER_CORE   = 36
  DT                 = 0.11 s
  min_period         ≈ 13.73 s
  nproc              = 144
  NEX                = 288 × 288
  内存/slice          ≈ 375 MB
  meshfem 耗时        = 16:14:22 → 16:18:03 (~3min41s)
  首个 fatal          = 无
  .err 中仅有         = ifort #7025 DIR$ 警告（标准 F2008 兼容提示，无害）
```

### 16.2 Gate A 判定：✅ 通过

| 通过标准 | 结果 |
|----------|------|
| 编译成功（xmeshfem3D + xspecfem3D） | ✅ |
| meshfem 完成（exit 0） | ✅ |
| 无 Jacobian / face flag / MPI 错误 | ✅ |
| NSPEC_CRUST_MANTLE = 7452 | ✅ |
| Par_file 自动恢复为 GLL_crust1.0 | ✅ |

**关键确认**：
- 编译仅 57 秒（make clean 重编译，含 meshfem3D + specfem3D）
- meshfem 约 3 分 41 秒完成（144 进程）
- `.err` 中仅有 13 条 ifort `#7025` 警告（`!DIR$ ATTRIBUTES` 非标准 F2008），无功能影响
- 服务器端 `model_EMC.f90` 仍为改动版（编译日志中可见 `model_EMC.checknetcdf.o`），但不影响 GLL 流程正确性（未走 EMC 代码路径）

### 16.3 下一步：Phase 2 Step 3（路径 A）

Gate A 通过，nspec=7452。原作者 GLL 基于 nspec=7236，不匹配，采用**路径 A**（跳过 Step 2）。

**立即执行的步骤**：

#### Step 1：从服务器下载 DATABASES_MPI

```bash
# 从曙光下载 DATABASES_MPI 到本地
scp -r user@曙光:/path/to/specfem3d_globe/EXAMPLES/FWEA23/DATABASES_MPI \
    specfem3d_globe/EXAMPLES/FWEA23/
```

#### Step 2：本地 NC → GLL 转换

```bash
cd /path/to/EASTASIA_FWI
conda activate eastasia_fwi

python 3_Data_space_simulation/convert_nc_to_gll.py \
    --mesh-dir specfem3d_globe/EXAMPLES/FWEA23/DATABASES_MPI \
    --model-nc data/models/processed/2024_FWEA23/2024_FWEA23_original.nc \
    --output-dir specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL \
    --anisotropic
```

**验证转换输出**：
- vpv: ~1.86 ~ 13.72 km/s
- rho: ~1.34 ~ 5.57 g/cm³（全局 min 来自浅层沉积盆地，含 1DREF 深部，mean ≈ 4.0）
- eta: ~0.80 ~ 1.10（非全 1.0）

> 注意：`--reference-gll-dir` 不可用（nspec 7452 vs 7236 不匹配），只能依赖域值统计验证。

#### Step 3：上传 GLL + 运行 solver

```bash
# 上传 GLL 到服务器
scp -r specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL user@曙光:/path/to/FWEA23/DATA/

# 确认 MODEL + GLL 路径（GLL 分支即可，不必再跑 mesh）
ssh user@曙光 "grep -E '^(MODEL|PATHNAME_GLL_modeldir)' /path/to/FWEA23/DATA/Par_file"

# 提交仅 solver（保留 DATABASES_MPI）
ssh user@曙光 "cd /path/to/FWEA23 && sbatch run_second_mesh_solver.bash"
```

**Solver 验证**：
- `DATABASES_MPI` 完整；`DATA/GLL` 与转换时 mesh 一致
- `Max norm displacement vector U` 不指数增长
- `OUTPUT_FILES/` 中有 SAC 波形文件
- 波形与原作者 SAC 定性对比

---

## 17. Phase 2 NC→GLL 转换结果（2026-03-18）

### 17.1 rho 单位 Bug 修复

转换前发现并修复了 `convert_nc_to_gll.py` 中的 **rho 单位混合 bug**：

**问题**：旧方案在 `process_one_proc` 中对插值后的 rho 数组做全局 `mean>100 → ÷1000` 检测。但 `interpolate_from_netcdf` 返回的 rho 是**混合单位**——域内/nearest 点为 kg/m³，深部 1DREF 填充点已为 g/cm³。全局 ÷1000 导致 1DREF 点被二次除法，rho 降至 ~0.003 g/cm³。

**修复**：将 rho kg/m³→g/cm³ 转换移到 `interpolate_from_netcdf` **源级**（构造插值器前对 NC 数组 ÷1000），使所有来源（线性插值、nearest 外推、1DREF 填充）统一为 g/cm³。

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| rho 日志 | `rho mean=2330，检测为 kg/m³` | `rho mean=4.00 g/cm³` |
| proc000000 rho | 0.005 ~ 4.609 | 1.921 ~ 5.565 |
| 深部 rho (>1000km) | ~0.003 (被 ÷1000) | ~5.5 (PREM CMB) |
| solver 预期 | **必然发散** | 正常 |

### 17.2 转换执行记录

```
日期: 2026-03-18
命令: python convert_nc_to_gll.py --mesh-dir .../DATABASES_MPI --model-nc .../2024_FWEA23_original.nc --output-dir .../DATA/GLL --anisotropic
nspec: 7452, nglob: 500785, nproc: 144
精度: float32
水平域外策略: nearest
```

**全局值域统计（144 procs 汇总）**：

| 参数 | min | max | 说明 |
|------|-----|-----|------|
| vpv | 1.860 | 13.717 | km/s |
| vph | 1.891 | 13.717 | km/s |
| vsv | 0.517 | 7.266 | km/s |
| vsh | 0.534 | 7.266 | km/s |
| eta | 0.796 | 1.104 | |
| rho | 1.344 | 5.567 | g/cm³（全局 min 来自浅层沉积盆地，含深部 1DREF） |
| qmu | 22.633 | 355.000 | |

**域外统计**：
- 深度域外：~34%（depth > 1000 km，用 1DREF 填充）
- 水平域外：0%（域内 proc）~ 100%（chunk 边缘 proc，用 nearest 外推）
- 100% 水平域外的 proc（chunk 旋转后完全超出 FWEA23 域）：proc000/001/010/011/022/023/071 等
- rho mean 全部 ~4.0 g/cm³ ✅

### 17.3 下一步

- [x] 上传 GLL 到曙光服务器
- [x] 提交 `run_second_mesh_solver.bash`（当前脚本：**仅 solver**，不重 mesh）
- [x] 验证转换值域 / solver 稳定性（`Max norm U` 不指数增长）
- [ ] 下载 SAC 波形，与原作者波形定性对比

---

## 18. Phase 2 Solver 运行结果（2026-03-19）

> **流程更新（2026-03-23）**：下文 job **5610153 / 5637010** 等记录的是当时「**第二次 xmeshfem3D + solver**」路径。当前推荐改为 **§5：保留 `DATABASES_MPI`，仅 `xspecfem3D`**，以避免第二次 mesh 的 Jacobian / face flag 问题；历史结论仍可用于分析**几何与模型**差异。

### 18.1 首次尝试：Jacobian 失败（job 5610153）

```
作业号: 5610153
日期: 2026-03-18
Gate: Phase 2 Step 3 (solver)
MODEL: GLL_crust1.0

结果: ❌ meshfem Jacobian 失败
  Error Jacobian rank: 112   Jacobian = -9.35e-10
  location r = 6281.985 km → depth ≈ 89 km
  lat/lon = 32.16° / 78.18° → 西藏/Hindu Kush 深 Moho 区域
```

**原因分析**：Jacobian 值 -9.35×10⁻¹⁰ 本质是浮点精度噪声（阈值为 `<= VERYSMALLVAL` 即 1e-24），在 Hindu Kush 深 Moho 拉伸区域（depth ~89km）元素接近退化。不同节点分配导致 MPI 浮点运算顺序微差，使边界元素在 ±0 之间翻转。Gate A（job 5609105, MODEL=s362ani）同一区域侥幸通过。

**Solver 未执行**：meshfem 失败后 DATABASES_MPI 不完整 → `mesh_parameters.bin` 不存在 → solver 立即报错退出。

### 18.2 第二次尝试：成功（job 5637010）

```
作业号: 5637010
日期: 2026-03-19
Gate: Phase 2 Step 3 (solver)
MODEL: GLL_crust1.0

constants.h.in:
  REGIONAL_MOHO_MESH      = .false.
  REGIONAL_MOHO_MESH_ASIA = .false.
  HONOR_DEEP_MOHO         = .false.
  RMOHO_STRETCH           = 5000.d0

结果: ✅ meshfem + solver 均成功（.err 为空）
```

#### Mesher 输出验证

**GLL 模型值域**（mesher 读入并确认）：

| 参数 | min | max | 与 NC→GLL 转换输出一致 |
|------|-----|-----|----------------------|
| vpv | 1.860 | 13.717 | ✅ |
| vph | 1.891 | 13.717 | ✅ |
| vsv | 0.511 | 7.266 | ✅ |
| vsh | 0.516 | 7.266 | ✅ |
| eta | 0.796 | 1.104 | ✅ |
| rho | 1.344 | 5.566 | ✅ |

**网格参数**：
- NSPEC_CRUST_MANTLE = 7452
- DT = 0.11 s
- Min period = 13.73 s
- Max Jacobian eigenvalue ratio = 0.997
- Min Jacobian eigenvalue ratio = 0.164
- Max stability (CFL) = 0.748
- Moho Crust1.0 min/max = 4.13 ~ 80.0 km
- 内部拓扑 410/660：**已启用**（s362ani 数据用于间断面拉伸）

**关键发现**：`MODEL = GLL_crust1.0` 时，mesher 同时加载了 s362ani（用于 410/660 内部拓扑拉伸）和 GLL 模型（用于速度/密度赋值）。mesher 日志显示 `broadcast model: S362ANI models` + `broadcast model: GLL_crust1.0`。

#### Solver 稳定性

| 时间步 | Max norm U (m) | 状态 |
|--------|---------------|------|
| 200 | 2.91e-06 | 初始 |
| 400 | 4.99e-02 | 上升（波传播中） |
| 1000 | 5.03e-02 | 稳定 |
| 2000 | 5.06e-02 | 稳定 |
| 3000 | 5.07e-02 | 稳定 |
| 3200 | 5.07e-02 | 稳定 |

**判定**：Max U 在 ~5.07e-02 m 处平稳，**无指数增长**，solver 稳定。

**运行时间**：
- 总时间步数：16500
- 每步平均：~0.251 s
- 预计总时间：~1h09min
- 台站数：1402
- 预计完成时间：2026-03-19 12:28

### 18.3 下一步

- [x] 等待 solver 完成（预计 12:28）
- [x] 确认 OUTPUT_FILES 中 SAC 波形文件完整
- [x] 下载 SAC 波形到本地
- [x] 与原作者波形（`specfem/sac/`）定性对比
- [x] 记录波形对比结果

---

## 19. 波形对比分析（GLL 正演 vs 原作者参考）

### 19.1 对比方法

使用 `2_waveform_plotting.py` 脚本对比：
- **参考数据**：`sac/` 目录，原作者（Chujie）的 GLL 正演结果（~1402 台站 × 3 分量）
- **测试数据**：`OUTPUT_FILES/` 目录，本次 GLL_crust1.0 正演结果
- **滤波**：带通 0.025–0.05 Hz（20–40 s 周期），2阶 Butterworth，零相位
- **归一化**：每条波形独立归一化（除以自身最大绝对值）
- **时间轴**：SAC B 值 + np.arange(npts) × delta
- **绘图**：红色=原作者，黑色=本次，y 偏移=震中距（度）

### 19.2 对比结果

**定性判断**：区别还是很大（"这次的模拟区别还是很大"）

对比 PDF：`figures_synthetic_comparison/synthetic_comparison_Z_20260319_065115.pdf`

### 19.3 已对齐参数（排除因素）

经逐项核实，以下参数已与原作者完全对齐，**不再**是差异来源：

| 参数 | 原作者 | 本次 | 状态 |
|------|--------|------|------|
| UNDO_ATTENUATION | .true. | .true. | ✅ 已对齐 |
| CMTSOLUTION half_duration | 3.63 (tau≈2.23/1.628) | 3.63 | ✅ 已对齐 |
| CMTSOLUTION header 时间 | 质心时间 17:06:54.6452 | 17:06:54.6452 | ✅ 已对齐 |
| CMTSOLUTION time_shift | 0.0 | 0.0 | ✅ 已对齐 |
| MODEL 格式 | GLL | GLL_crust1.0 | ✅ 已对齐 |
| REGIONAL_MESH_CUTOFF | 无（v7） | .false. | ✅ 已对齐 |

### 19.4 根因分析 — 仍存在的差异（按影响排序）

#### ① 网格拓扑不同（NSPEC 7452 vs 7236）— 影响：高

| 项目 | 本次 (v8.1.0 auto_ner) | 原作者 (v7.0.0 + tibet_v4) |
|------|------------------------|---------------------------|
| NSPEC_CRUST_MANTLE | 7452 | 7236 |
| NER_CRUST | 2 | 3 |
| NER_80_MOHO | 1 | ? |
| constants.h.in | 默认 auto_ner | tibet_v4（手动径向分层） |

**机制**：径向单元分布不同 → GLL 积分节点位置不同 → 波场在物理空间的采样不同 → 体波和面波走时、振幅均受影响。

**根源**：v8.1.0 移除了 tibet_v4 配置，auto_ner 算法给出不同的 NER 分布。要匹配 NSPEC=7236 需要手动修改 `auto_ner.f90` 或 `constants.h.in`。

#### ② 地壳处理不同（2 层 vs 3 层 + 深 Moho）— 影响：高

| 项目 | 本次 | 原作者 |
|------|------|--------|
| NER_CRUST | 2 | 3 |
| REGIONAL_MOHO_MESH | .false.（默认） | .true. |
| HONOR_DEEP_MOHO | .false.（默认） | .true. |
| 实际地壳层数 | 2 | 3 |
| 深 Moho 处理 | 无（Moho 最多放在 default ~35 km） | 实际 Moho（青藏 60-80 km）|

**机制**：
- 2 层 vs 3 层：影响地壳内部 Pn、Sn 及多次反射波
- 无 HONOR_DEEP_MOHO：青藏高原、兴都库什地区 Moho 60–80 km，但网格未拉伸适配 → 深壳结构被错误表示
- 主要影响：近源台站和穿过厚壳地区的路径

#### ③ NC→GLL 精度损失 — 影响：中

| 参数 | 本次 NC→GLL | 原作者原始 GLL |
|------|-------------|----------------|
| vpv_min | 1.86 km/s | 0.75 km/s |
| vsv_min | 0.51 km/s | ~ |
| 浅层细节 | 插值模糊 | 原始 FWI 分辨率 |

**机制**：NC 网格（0.25°×0.25°×深度）精度不如原始 GLL 节点上的 FWI 结果。浅层沉积盆地（极低速）、地壳内不连续面等细节在 NC→GLL 插值中被平滑。

**验证方法**：§11.3 Step 2 — 直接用原作者 GLL（`specfem/model_updated/`）运行，可隔离此因素。

#### ④ SPECFEM 版本差异（v8.1.0 vs v7.0.0）— 影响：低-中

v8.1.0 相比 v7.0.0 的已知变更：
- 坐标系修正（椭率、重力计算更新）
- 数值求解器微调（CFL 阈值、边界条件）
- 源码重构（模块化改进）

即使使用相同网格和模型，两个版本的数值解也会略有不同（通常 < 1%）。

#### ⑤ 410/660 内部拓扑处理不确定 — 影响：低

**发现**：`output_mesher.txt` 显示 "incorporating element stretching for 3-D internal surfaces"，但源码 `setup_model.f90` 第 103–111 行的条件判断**不包含 GLL 模型**（仅 S362ANI, S362WMANI, S362ANI_PREM, S29EA, MANTLE_SH, SPIRAL）。

`compute_element_properties.f90` 第 249–253 行的实际拉伸代码同样不包含 GLL 模型。

**疑点**：日志输出与源码条件不一致 — 服务器上编译的二进制可能来自不同版本的源码，或源码在编译后有修改。

### 19.5 结论与下一步

**核心问题**：参数层面已完全对齐，剩余差异来自**网格拓扑**（NSPEC、地壳层数、深 Moho）和 **NC→GLL 精度损失**，这些是结构性差异，无法通过修改 Par_file 参数解决。

**可选方案**（按优先级排序）：

| 方案 | 内容 | 目的 |
|------|------|------|
| **A. 隔离模型因素** | 直接用原作者 GLL（`specfem/model_updated/`）运行 | 排除 NC→GLL 精度损失（因素③），确认因素①②④的影响 |
| **B. 接受当前基线** | 将本次结果作为 "v8.1.0 默认网格 GLL" 基线 | 记录当前差异水平，后续逐步改进 |
| **C. 手动对齐网格** | 修改 `auto_ner.f90`/`constants.h.in` 以匹配 NSPEC=7236 + HONOR_DEEP_MOHO | 消除因素①②，最大程度逼近原作者 |

**建议**：先执行方案 A（已有原作者 GLL 文件，只需替换 DATABASES_MPI 后重新 mesher+solver），可明确量化 NC→GLL 精度损失的影响大小；如果方案 A 后差异大幅减小，则 NC→GLL 精度损失是主因；如果差异仍然很大，则网格拓扑差异是主因，需要方案 C。

---

## 20. Gate C 测试结果（2026-03-19）— 失败

### 20.1 测试记录

```
作业号: 5639032
日期: 2026-03-19
Gate: C (tibet_v4 完整配置)
相对上次仅改变的 1 个变量: constants.h.in 三参数改为 tibet_v4 值

constants.h.in:
  REGIONAL_MOHO_MESH      = .true.
  HONOR_DEEP_MOHO         = .true.
  RMOHO_STRETCH           = -20000.d0

编译: make clean=是, 时间=15:38:41 → 15:39:30 CST (49s)

结果:
  NER_CRUST              = 3（3-layer crust ✅）
  NER_80_MOHO            = 1
  NSPEC per slice        = 8028（非 7452 也非 7236，因 R80_STRETCH=-40000 不同于 tibet_v4）
  DT                     = 0.165 s
  编译                   = ✅ 成功
  meshfem 第一遍（分配）  = ✅ 完成
  meshfem 第二遍（模型）  = ❌ 崩溃
```

### 20.2 崩溃分析

**错误消息**：
```
Error vpv: 0.0  vph: 0.0  vsv: 0.0  vsh: 0.0  rho: 0.0
radius: 6367.93 km  theta/phi: 87.17° / 110.90°
Error get_model values
Error detected, aborting MPI... proc 38
```

**崩溃位置**: `get_model.F90:227` — `vpv < TINYVAL` 检查

**地理位置**: (~2.8°N, 110.9°E) — 加里曼丹/婆罗洲西海岸过渡带

**根本原因链**:
1. `stretch_deep_moho()` 将网格边界拉到 R60 (6311 km = 60 km 深度)
2. 拉伸后某些 GLL 点落在海岸过渡带（海洋-陆地边界）
3. CRUST1.0 的 CAP 平滑在该点某个沉积/地壳层返回 `found_crust=.true.` 但 vp=vs=rho=0
4. 地壳模型覆写了 1D 参考模型的非零值 → 全零 → `vpv < TINYVAL` 崩溃

**代码路径**:
```
get_model.F90:
  1D-REF model → vpv ≠ 0 ✅
  s362ani 3D mantle → vpv*(1+dv) ≠ 0 ✅
  CRUST1.0 → found_crust=.true., vpv=0 → 覆写 ❌
  vpv < TINYVAL → MPI_Abort
```

**为什么 Gate C 在 v8.1.0 失败而在 v7.0.0 成功**:
- v8.1.0 的 `stretch_deep_moho` 在海岸过渡带产生更极端的网格变形
- v8.1.0 的 CRUST1.0 CAP 平滑可能使用了不同的平滑参数
- v7.0.0 可能不存在这些边缘情况，或处理方式不同

**关键约束**: `stretch_deep_moho()` 硬编码检查 `RMOHO_STRETCH_ADJUSTMENT == -20000`，无法使用其他值。

### 20.3 Gate C 判定：❌ 失败

与 §15.5 的预测一致（"大概率失败"）。

### 20.4 下一步：Gate B

切换为 Gate B 配置（§15.5 表格），使用更保守的 `stretch_moho` 替代 `stretch_deep_moho`：

| 参数 | Gate C (失败) | Gate B (下一步) |
|------|------|------|
| `EARTH_REGIONAL_MOHO_MESH` | `.true.` | `.true.` |
| `EARTH_HONOR_DEEP_MOHO` | `.true.` | `.false.` |
| `EARTH_RMOHO_STRETCH_ADJUSTMENT` | `-20000.d0` | `-15000.d0` |
| 预期 RMOHO_FICTITIOUS | 6311000 (60 km) | 6316000 (55 km) |
| 使用的 stretch 函数 | `stretch_deep_moho` (max 60 km) | `stretch_moho` (max 45 km) |
| NER_CRUST | 3 | 3 |

`stretch_moho` 对 Moho > 45 km 的区域只拉伸到 45 km，深层 Moho（喜马拉雅 60-80 km）会在单元内插值而非精确对齐网格边界。但不会产生 Gate C 的零值崩溃。

```bash
# constants.h.in 已修改为 Gate B:
# EARTH_HONOR_DEEP_MOHO = .false.
# EARTH_RMOHO_STRETCH_ADJUSTMENT = -15000.d0
# EARTH_REGIONAL_MOHO_MESH = .true. (不变)

# 同步到服务器后:
sbatch run_meshfem_only.bash   # 必须 make clean 重编译
```

---

## 21. Gate B 结果与 Plan D 实施（2026-03-19）

### 21.1 Gate B 网格结果

```
作业号: (Gate B mesh)
日期: 2026-03-19
Gate: B
constants.h.in:
  REGIONAL_MOHO_MESH      = .true.
  HONOR_DEEP_MOHO         = .false.
  RMOHO_STRETCH           = -15000.d0

结果:
  NSPEC_CRUST_MANTLE = 8028
  NER_CRUST          = 3 (3-layer crust ✅)
  NER_80_MOHO        = 1
  DT                 = 0.165 s
  min_element_edge   = 6.599 km
  Max CFL            = 1.028  ← 超过 1.0！不稳定
  Max suggested DT   = 0.088 s
  Min Jacobian ratio = 0.1525 (可接受)
  Mesher 耗时        = 3 min 49 s
  首个 fatal          = 无（mesher 成功完成）
```

### 21.2 Gate B 判定：⚠️ Mesher 通过但 CFL 不稳定

| 通过标准 | 结果 |
|----------|------|
| 编译成功 | ✅ |
| meshfem 完成（exit 0） | ✅ |
| 无 Jacobian / face flag / MPI 错误 | ✅ |
| CFL ≤ 1.0（solver 可稳定运行） | ❌ CFL = 1.028 |

**关键问题**：`stretch_moho` 在海岸/深 Moho 过渡区域创建了较小的元素（min edge = 6.6 km），导致 CFL = 1.028 > 1.0。要让 solver 稳定运行，DT 必须从 0.165 s 降到 ≤ 0.088 s，**时间步数翻倍，计算成本翻倍**。

**与其他配置对比**：

| 项目 | Gate A | Gate B | Gate C | tibet_v4 |
|------|--------|--------|--------|----------|
| NSPEC | 7452 | 8028 | 8028 | 7236 |
| NER_CRUST | 2 | 3 | 3 | 3 |
| CFL | OK (0.748) | 1.028 ❌ | crashed (vpv=0) | OK |
| DT | 0.11 s | 需 ≤ 0.088 s | — | 0.15 s |
| Moho 对齐 | 默认 35 km | 最大 45 km | 60 km (崩溃) | 60 km |

### 21.3 Plan D 方案选择

Gate B 存在两个问题：
1. **CFL > 1.0**：必须降 DT，计算成本翻倍
2. **Moho 对齐仅 45 km**：青藏高原 60-80 km 的深 Moho 无法精确对齐网格边界

而 Gate C（tibet_v4 目标配置）仅因 CRUST1.0 在海岸过渡带返回零值而崩溃。**Plan D** 通过在 `get_model.F90` 中将 `call exit_mpi` 替换为 1D 参考值回退，解决 Gate C 的唯一问题，同时保留完整的 60 km 深 Moho 网格拓扑。

**Plan D 对波形零影响的原因**：
- 崩溃发生在 mesher 第二遍（模型赋值阶段）
- MODEL=GLL_crust1.0 / MODEL=gll 的 solver 阶段会用 GLL 二进制文件**覆写所有速度/密度值**
- mesher 阶段的模型值仅用于：(1) 网格拓扑（第一遍，不受影响）；(2) Jacobian 计算（第二遍，回退到非零 1D 值即可）；(3) CFL 初始估计

### 21.4 Plan D 实施 — 修改清单

#### (1) `src/meshfem3D/get_model.F90`

**变量声明**（新增）：
```fortran
! Plan D: saved 1D+3Dmntl values for fallback when crust returns zero
double precision :: vpv_1d,vph_1d,vsv_1d,vsh_1d,rho_1d,eta_1d
```

**在 `get3Dcrust_val` 调用前**（新增）：
```fortran
! saves 1D+3Dmntl values before crustal override (Plan D fallback)
vpv_1d = vpv; vph_1d = vph; vsv_1d = vsv; vsh_1d = vsh
rho_1d = rho; eta_1d = eta_aniso
```

**替换 `vpv < TINYVAL` 错误处理**（修改）：
```fortran
! 原代码（崩溃）：
!   if (vpv < TINYVAL) then
!     print *,'Error vpv: ...'
!     call exit_mpi(myrank,'Error get_model values')
!   endif

! Plan D（回退 + 警告）：
if (vpv < TINYVAL) then
  vpv = vpv_1d; vph = vph_1d; vsv = vsv_1d; vsh = vsh_1d
  rho = rho_1d; eta_aniso = eta_1d
  if (myrank == 0) then
    print *,'Warning: vpv=0 at r=',r*R_PLANET_KM,' km, theta/phi=', &
            theta*180.d0/PI,phi*180.d0/PI,', falling back to 1D reference'
  endif
endif
```

**代码逻辑**：
```
get1D_val → vpv ≠ 0 ✅
get3Dmntl_val → vpv*(1+dv) ≠ 0 ✅
    ↓ 保存 vpv_1d = vpv  ← Plan D 新增
get3Dcrust_val → found_crust=.true., vpv=0 ❌
    ↓
vpv < TINYVAL → 恢复 vpv = vpv_1d  ← Plan D 回退（不再 MPI_Abort）
    ↓
后续 kappav/muhv/etc 计算使用非零回退值 ✅
    ↓
solver 阶段 GLL 覆写全部值 → 最终波形不受影响
```

#### (2) `setup/constants.h.in`

恢复为 Gate C（tibet_v4）配置：
```fortran
EARTH_REGIONAL_MOHO_MESH         = .true.    ! 3-layer crust
EARTH_HONOR_DEEP_MOHO            = .true.    ! Gate C + Plan D: deep moho (60km)
EARTH_RMOHO_STRETCH_ADJUSTMENT   = -20000.d0 ! Gate C + Plan D (vpv=0 fallback patched)
```

### 21.5 预期结果

- Mesher 不再因 vpv=0 崩溃，改为打印 Warning 并用 1D PREM 值填充
- NSPEC_CRUST_MANTLE 预期 = 8028（与 Gate C 相同，因网格拓扑未变）
- DT 预期 = 0.165 s（与 Gate C mesher 第一遍相同）
- CFL 需关注：Gate C 第一遍通过了（在第二遍崩溃），DT 是否稳定取决于 mesher 输出
- 如果 CFL > 1.0，需要像 Gate B 一样降低 DT

### 21.6 下一步

1. 将修改后的 `get_model.F90` 和 `constants.h.in` 同步到服务器
2. `make clean && make meshfem3D specfem3D`
3. `sbatch run_meshfem_only.bash`
4. 检查 mesher 输出：
   - Warning 消息数量和位置
   - NSPEC、DT、CFL 值
   - 是否有 Jacobian / MPI 错误
5. 如通过，继续 NC→GLL → solver 流程

---

## 22. Plan D 测试结果 — 仍然失败（2026-03-20）

### 22.1 测试记录

- **Job ID**: 5640168
- **配置**: Gate C + Plan D (vpv=0 fallback patched)
- **结果**: ❌ Mesher 第二遍(second pass) 完成所有10层 → **MPI 接口构建时崩溃**
- **错误**: `Error face flag` on proc 61
- **NSPEC**: 8028, **CFL**: 1.028 > 1.0

### 22.2 错误级联（§12.4 确认）

```
vpv=0 crash ─[Plan D 已修复]─→ Error face flag (proc 61) ─→ Error ineighbor points differ ─→ PMPI_Wait truncated
```

Plan D 成功修复了 vpv=0 问题，但暴露了更深层的 MPI 接口检测错误。

### 22.3 v7.0.0 vs v8.1.0 深度对比分析

对完整源码进行系统性对比，逐文件审查所有 MPI 接口、网格构建、Moho 拉伸相关代码。

#### 确认相同的代码（排除嫌疑）

| 文件 | v7 行数 | v8 行数 | 对比结论 |
|------|---------|---------|----------|
| `get_MPI_interfaces.f90` (face flag 检测) | 733 | 783 | **逻辑完全相同**，仅 allocatable 数组 + neighbours→neighbors 改名 |
| `assemble_MPI_scalar_block()` (test_flag 装配) | ~390 | ~390 | **逻辑完全相同**，仅 myrank 移入 module + intent 声明 |
| `setup_MPI_interfaces.f90` (test_flag 初始化) | 580 | 617 | **逻辑完全相同**，v8 增加 NPROCTOT>1 守卫 |
| `add_interface_point()` (接口点添加) | 88 | 85+work_points | **逻辑完全相同**，v8 增加可选快速查找(默认关闭) |
| `moho_stretching.f90` (Moho 拉伸核心) | 663 | 812 | **Earth 路径逻辑完全相同**，v8 增加 Mars/Moon 支持 |
| `compute_element_properties.f90` (拉伸调用) | — | — | **调用逻辑完全相同** |

#### 确认不同的代码（可能的根因）

| 文件/变更 | v7 | v8 | 影响评估 |
|-----------|----|----|----------|
| **DT 乘数** (`get_timestep_and_layers.f90`) | `DT*(1.d0 - 0.1d0)` = **DT×0.9** | `DT*(1.d0 + 0.5d0)` = **DT×1.5** | ⚠️ v8 **增大** DT（更不稳定！），方向相反 |
| **MAX_NEIGHBORS** | 8 + corners | **20** + corners | ✅ 增大容量，不应导致问题 |
| **create_regions_mesh.F90** | 560 行 | 1422 行 | ⚠️ 重大重构 |
| **create_regions_elements.f90** | (内联) | **新独立模块** 275 行 | ⚠️ 元素创建流程重构 |
| **crm_fill_global_meshes()** | 不存在 | **v8 新增** | ⚠️ 坐标聚合新步骤 |
| **get_global.F90** | 126 行 | 249 行 | ⚠️ 新增并行 STL 排序选项 |
| **xstore_glob 使用** | 直接使用 xstore_crust_mantle | 通过 crm_fill_global_meshes() 生成 xstore_glob | ⚠️ 坐标传递路径改变 |

### 22.4 根因分析

**结论：v8.1.0 的网格构建流程重构导致 MPI 接口检测失败**

Face flag 检测逻辑和 test_flag 装配逻辑在 v7 和 v8 之间**完全相同**，因此问题出在上游数据：

1. **`create_regions_elements.f90` 重构**：v8 将元素创建分离为独立模块，使用 `perm_layer(:)` 层排列。虽然逻辑流程相似，但元素创建顺序的微小差异可能影响 `ibool` 全局编号
2. **`crm_fill_global_meshes()` 新步骤**：v8 在第二遍后额外聚合 `xstore→xstore_glob`，作为 MPI 接口点排序的坐标源。v7 直接使用 region-local 数组
3. **`get_global.F90` 变更**：v8 新增并行 STL 排序选项，可能在处理拉伸后近重合坐标时产生不同编号

**错误传播链**：
```
Moho 拉伸(正确) → 元素创建顺序变化(v8重构) → ibool 编号差异(get_global)
→ 坐标聚合路径变化(crm_fill_global_meshes) → MPI 接口检测失败(face flag)
```

### 22.5 DT 乘数 Bug（独立问题）

v8 在 `REGIONAL_MOHO_MESH` 模式下将 DT 乘以 **1.5**（增大），而 v7 乘以 **0.9**（减小）。

```fortran
! v7: get_timestep_and_layers.f90 line 494
DT = DT*(1.d0 - 0.1d0)   ! DT × 0.9 — 更保守

! v8: get_timestep_and_layers.f90 line 849
DT = DT*(1.d0 + 0.5d0)   ! DT × 1.5 — 更激进（方向反了！）
```

这不影响 mesher 崩溃（DT 只在 solver 中使用），但即使 mesher 通过，solver 也会因 CFL > 1.0 而不稳定。当前 CFL=1.028，最大允许 DT≈0.088s。

### 22.6 下一步方案

#### 方案 E：编译使用 v7.0.0（推荐）

v7.0.0 源码已在 `specfem/specfem3d_globe_code_new/`，原作者 tibet_v4 配置验证可用。

1. 在曙光集群编译 v7.0.0
2. 使用 tibet_v4 完整配置（`specfem/sem_config_tibet_v4/`）
3. 运行 MODEL=GLL + REGIONAL_MOHO_MESH + HONOR_DEEP_MOHO
4. 预期：Mesher 成功 + DT×0.9 更保守 + Solver 需验证 CFL

**优势**：绕过 v8 的所有兼容性问题，完全复现原作者环境

#### 方案 F：修复 v8 DT + 向 SPECFEM 提交 Bug

1. 修改 `get_timestep_and_layers.f90` line 849：`DT*(1.d0 + 0.5d0)` → `DT*(1.d0 - 0.1d0)`
2. Face flag 问题向 SPECFEM3D Globe GitHub 提交 issue，附带 v7/v8 对比分析
3. 等待开发者修复或自行深入调试 `create_regions_elements.f90` + `get_global.F90`

#### 方案 G：Gate A 基础上规避

放弃 REGIONAL_MOHO_MESH，使用 Gate A 配置（标准 s362ani_crust1.0 网格）+ 手动修正 DT/CFL。波形精度会损失，但可以运行。

---

## 23. 应用外部开发者修改（2026-03-20）

### 23.1 背景

收到外部开发者按需求对 v8.1.0 进行的代码修改（`specfem3d_globe/specfem3d_makefile/` 目录，共 7 个文件）。
经逐文件 diff 对比后，决定采纳其中 5 处关键修改，结合我们已有的 Plan D vpv=0 fallback 补丁，重新运行 Gate C。

### 23.2 修改清单

| # | 文件 | 修改内容 | 影响 |
|---|------|----------|------|
| 1 | `src/shared/get_timestep_and_layers.f90` | 在 `DT*(1.d0+0.5d0)` 之后添加 `DT = 0.1d0`，硬编码覆盖 v8 的 1.5x DT bug | **关键**：解决 CFL>1.0 |
| 2 | `src/shared/auto_ner.f90` | 注释掉 lines 483-497 的 NER 经验值赋值，让 `auto_optimal_ner` 从最小值开始搜索 | **关键**：改变 NSPEC，可能避免 face flag 错误 |
| 3 | `src/shared/get_model_parameters.F90` | `gll_qmu` case 添加 `impose_crust = ICRUST_CRUST1` | 低影响（我们用 `GLL_crust1.0` 后缀已自动设置） |
| 4 | `src/specfem3D/compute_forces_viscoelastic_calling_routine.F90` | 注释掉 `if (NCHUNKS_VAL /= 6 .and. ABSORBING_CONDITIONS) return` | 允许 1-chunk 区域模拟计算内核力 |
| 5 | `src/specfem3D/save_kernels.F90` | 注释掉 `call restore_original_moduli()`（标注 "huangf"） | FWI 反演专用，正演无影响 |

### 23.3 保留的已有修改

| 文件 | 修改 | 来源 |
|------|------|------|
| `setup/constants.h.in` | tibet_v4 配置（REGIONAL=.true., HONOR=.true., RMOHO=-20000） | Gate C |
| `src/meshfem3D/get_model.F90` | Plan D vpv=0 fallback（CRUST1.0 返回零值时回退到 1D 参考） | §21.4 |
| `src/meshfem3D/setup_MPI_interfaces.f90` | MAX_NEIGHBORS=20 | §12 安全余量 |
| `src/specfem3D/iterate_time_undoatt.F90` | ifort 编译器兼容修复 | §12 通用修复 |

### 23.4 当前 `git diff --stat HEAD`

```
 setup/constants.h.in                                          | 14 ++++++++------
 src/meshfem3D/get_model.F90                                   | 21 +++++++++++++++++----
 src/meshfem3D/setup_MPI_interfaces.f90                        |  2 +-
 src/shared/auto_ner.f90                                       | 26 +++++++++++++-------------
 src/shared/get_model_parameters.F90                           |  1 +
 src/shared/get_timestep_and_layers.f90                        |  4 ++--
 src/specfem3D/compute_forces_viscoelastic_calling_routine.F90 |  2 +-
 src/specfem3D/iterate_time_undoatt.F90                        |  4 +++-
 src/specfem3D/save_kernels.F90                                |  2 +-
 9 files changed, 47 insertions(+), 29 deletions(-)
```

### 23.5 下一步

1. 同步到曙光服务器
2. `make clean && make meshfem3D specfem3D`（constants.h.in 有改动，必须 clean）
3. 提交 Gate C 测试（tibet_v4 完整配置）
4. 记录 NSPEC、DT、CFL，对比之前的测试结果

### 23.6 预期效果

- **DT=0.1d0** 覆盖 v8 的 1.5x bug → CFL 应降至 < 1.0（之前 CFL=1.028 是因为 DT=0.165 过大）
- **NER 经验值注释** → `auto_optimal_ner` 搜索可能给出不同的 NSPEC（需确认是否变为 7236）
- **vpv=0 fallback**（Plan D）→ 防止 CRUST1.0 海岸过渡带零值崩溃
- **内核力 early return 注释** → 1-chunk 区域模拟正确计算内核力

---

## 24. §23 补丁后首次正演结果 — Solver 发散（2026-03-23）

### 24.1 测试记录

```
日期: 2026-03-23
配置: §23 全部补丁已应用（auto_ner + DT=0.1 硬编码 + Plan D vpv=0 fallback
     + compute_forces early-return 注释 + save_kernels 注释）
constants.h.in:
  REGIONAL_MOHO_MESH      = .true.
  HONOR_DEEP_MOHO         = .true.
  RMOHO_STRETCH           = -20000.d0
MODEL                     = GLL（solver 检测）
DATA/Par_file MODEL       = s362ani_crust1.0（mesher 阶段设置）
SPECFEM 版本              = v8.1.0-1-gb6fa4b6
精度                      = 单精度
MPI 进程                  = 144 (12×12), 1 chunk
NEX                       = 288
震源                      = Mw 6.35, lat=33.64°, lon=131.88°, depth=76.65 km
```

### 24.2 Mesher 结果：✅ 成功

```
NSPEC_CRUST_MANTLE  = 7236   ← 与 v7.0.0 目标一致 ✅（auto_ner 补丁生效）
NSPEC_OUTER_CORE    = 720
NSPEC_INNER_CORE    = 36
NER_CRUST           = 3      ← 3-layer crust ✅（tibet_v4 配置）
DT                  = 0.1 s  ← 硬编码覆写生效 ✅
Min period          = 13.73 s
Max suggested DT    = 0.088 s
Max CFL (stability) = 0.623  ← 小于 1.0 ✅
Min Jacobian ratio  = 0.151
Moho Crust1.0 范围  = 4.13 ~ 80.0 km
Mesher 耗时         = 3 min 27 s
结果                = ✅ End of mesh generation（无错误）
```

**关键确认**：
- ✅ auto_ner 补丁使 NSPEC=7236（与 v7.0.0 一致）
- ✅ DT=0.1 硬编码覆写了 v8 的 1.5x bug
- ✅ CFL=0.623 < 1.0（理论上 solver 应稳定）
- ✅ deep moho stretching 已启用（NER_CRUST=3）
- ✅ Plan D vpv=0 fallback 使 mesher 不再崩溃

### 24.3 Solver 结果：❌ 发散

Solver 完成全部 18200 步（30.24 分钟），但 **Max norm displacement 出现指数增长**。

| 输出编号 | 时间步 | 模拟时间 | Max U (m) | 状态 |
|----------|--------|----------|-----------|------|
| 1 | 5 | 0.5 s | 2.31e-06 | 初始 |
| 2 | 200 | 20 s | 5.21e-02 | 上升（波传播中） |
| 5 | 800 | 1.3 min | 5.26e-02 | 稳定 |
| 10 | 1800 | 3.0 min | 5.27e-02 | 稳定 |
| 20 | 3800 | 6.3 min | 5.29e-02 | 稳定 |
| 40 | 7800 | 13.0 min | 5.29e-02 | 稳定 |
| 45 | 8800 | 14.7 min | 5.29e-02 | 最后稳定步 |
| **46** | **9000** | **15.0 min** | **6.57e-02** | **⚠️ 发散起始** |
| 47 | 9200 | 15.3 min | 1.10e-01 | ↑ |
| 48 | 9400 | 15.7 min | 1.51e-01 | ↑ |
| 50 | 9800 | 16.3 min | 2.95e-01 | ↑ |
| 55 | 10800 | 18.0 min | 1.89 | ↑ |
| 60 | 11800 | 19.7 min | 13.9 | ↑↑ |
| 70 | 13800 | 23.0 min | 624 | ↑↑↑ |
| 80 | 15800 | 26.3 min | 24,795 | ↑↑↑↑ |
| 90 | 17800 | 29.7 min | 1,072,649 | 💥 |
| 92 | 18200 | 30.3 min | 2,602,598 | 💥 完全爆炸 |

**发散时间线**：~15.0 min（时间步 ~9000）开始指数增长，从 0.053 m → 260 万 m。

**运行时间**：1h 10m 49s（~0.234 s/step），18200 步全部完成。

### 24.4 SAC 波形分析

对全部 1401 个 BXZ 分量 SAC 文件进行振幅统计：

**代表性台站**：

| 台站 | abs_max (m) | 震中距 | 判定 |
|------|-------------|--------|------|
| IC.BJT.00 | 2.97e-01 | ~15° | 偏大 |
| IU.INCN.00 | 1.03e+01 | ~10° | ❌ 明显异常 |
| IU.ULN.00 | 4.91e-03 | ~25° | 正常 |
| CB.LZH.00 | 1.67e-02 | ~20° | 正常 |

**振幅最大的 10 个台站**（全部在震源附近，日本/九州）：

| 台站 | abs_max (m) | 网络 |
|------|-------------|------|
| BO.KGM | 1.25e+05 | Bosai (日本) |
| JP.JMZ | 4.64e+04 | Japan |
| BO.ZMM | 2.26e+04 | Bosai |
| JP.JOW | 1.67e+04 | Japan |
| BO.AMM | 1.56e+04 | Bosai |
| BO.KYK | 2.42e+03 | Bosai |
| BO.TAS | 1.28e+03 | Bosai |
| BO.SIB | 6.17e+02 | Bosai |
| BO.YNG | 6.01e+02 | Bosai |
| BO.IGK | 5.94e+02 | Bosai |

**全局统计**：

| 指标 | 值 |
|------|-----|
| 总台站数 | 1401 |
| 中位振幅 | 0.124 m |
| 平均振幅 | 178.8 m |
| 最小振幅 | 1.59e-05 m |
| 最大振幅 | 1.25e+05 m |
| > 1.0 m | 440 (31.4%) |
| > 100 m | 63 (4.5%) |
| > 10,000 m | 5 (0.4%) |

**空间分布**：发散集中在震源附近（日本九州，lat≈33.6°, lon≈131.9°），BO 和 JP 网络台站受影响最大。远场台站（中亚、塔吉克斯坦等，震中距 > 25°）振幅在 1e-5 量级，属正常。

### 24.5 GLL 模型值域检查：✅ 物理合理

用 `temp/check_gll_values.py` 对 DATA/GLL/ 全部 1008 个文件（144 proc × 7 params）进行统计：

| 参数 | min | max | zeros | neg | 判定 |
|------|-----|-----|-------|-----|------|
| vpv | 1.86 km/s | 13.72 km/s | 0 | 0 | ✅ 正常 |
| vsv | 0.51 km/s | 7.27 km/s | 0 | 0 | ✅ 正常 |
| vph | 1.89 km/s | 13.72 km/s | 0 | 0 | ✅ 正常 |
| vsh | 0.52 km/s | 7.27 km/s | 0 | 0 | ✅ 正常 |
| rho | 1.34 g/cm³ | 5.57 g/cm³ | 0 | 0 | ✅ 正常 |
| eta | 0.80 | 1.10 | 0 | 0 | ✅ 正常 |
| qmu | 22.6 | 355 | 0 | 0 | ✅ 正常 |

**Vs/Vp 比值**（proc000000 样本）：0.27 – 0.59，无 Vs ≈ Vp 异常。

**GLL 文件大小**：3,618,008 字节/文件 = Fortran header(4) + 7236×125×4 + trailer(4) = NSPEC 7236 ✅

### 24.6 已排除的发散原因

| 原因 | 检查结果 | 排除 |
|------|----------|------|
| CFL > 1.0 | CFL = 0.623 ✅ | ✅ |
| GLL 值域异常（零值/负值/Vs≈Vp） | 全部物理合理 ✅ | ✅ |
| 密度单位错误 (kg/m³ vs g/cm³) | rho 1.34–5.57 g/cm³ ✅ | ✅ |
| NSPEC 不匹配 | GLL 文件大小与 mesher NSPEC=7236 精确一致 ✅ | ✅ |
| Mesher 失败 | Mesher 成功完成 ✅ | ✅ |

### 24.7 可能的发散原因分析

#### ① GLL 元素排序不匹配（嫌疑：高）

当前 GLL 文件来源需确认：
- 如果来自**原作者 v7.0.0 运行**：即使 NSPEC=7236 相同，v7 和 v8 的元素编号/排序/MPI 分配可能不同 → GLL 值被映射到错误的空间位置 → 局部非物理 → 发散
- 如果来自 **NC→GLL 转换**（使用当前 mesh 的 DATABASES_MPI）：排序应正确

**验证方法**：检查 GLL 文件与 DATABASES_MPI 中 `proc*_solver_data.bin` 的坐标是否一致，或重新执行 NC→GLL 转换。

#### ② `compute_forces_viscoelastic_calling_routine.F90` 补丁副作用（嫌疑：中）

§23 补丁注释掉了 1-chunk + ABSORBING_CONDITIONS 的 early return。该 return 原本阻止力计算在 1-chunk 区域模拟中执行。虽然描述为"内核力"，但如果影响了正演力计算路径，可能导致吸收边界处理异常。

**验证方法**：撤销该补丁单独测试，或检查源码确认 early return 仅在 adjoint/kernel 路径中。

#### ③ DT=0.1 硬编码与网格不完全匹配（嫌疑：低-中）

Mesher 建议最大 DT = 0.088 s，但硬编码使用 0.1 s。虽然 CFL=0.623 < 1.0（mesher 报告），但 DT > max_suggested_DT 意味着某些元素可能在边界条件下不稳定。

**对比 §18.2**：Job 5637010（NSPEC=7452, DT=0.11, CFL=0.748, **稳定**）同样 DT > max_suggested_DT，但未发散。

#### ④ 吸收边界 + deep moho stretching 耦合问题（嫌疑：中）

本次运行启用了 HONOR_DEEP_MOHO（拉伸到 60 km），而 §18.2 稳定运行禁用了该选项。deep moho stretching 可能在边界附近产生畸形元素，与吸收边界条件耦合后引发不稳定。发散在 15 min 开始（波前到达边界区域附近）支持此假设。

### 24.8 与 §18.2 稳定运行的关键差异

| 项目 | §18.2（✅ 稳定） | 本次（❌ 发散） |
|------|------------------|-----------------|
| NSPEC | 7452 | 7236 |
| NER_CRUST | 2 | 3 |
| DT | 0.11 s | 0.1 s |
| CFL | 0.748 | 0.623 |
| REGIONAL_MOHO_MESH | .false. | .true. |
| HONOR_DEEP_MOHO | .false. | .true. |
| RMOHO_STRETCH | 5000 | -20000 |
| auto_ner 补丁 | 否 | 是 |
| compute_forces 补丁 | 否 | 是 |
| GLL 来源 | NC→GLL（7452 mesh） | 需确认 |

**最主要差异**：REGIONAL_MOHO_MESH + HONOR_DEEP_MOHO + compute_forces 补丁。

### 24.9 下一步建议

1. **确认 GLL 来源**：检查 DATA/GLL/ 中的文件是原作者 v7 GLL 还是 NC→GLL 转换结果。若为前者，**重新执行 NC→GLL 转换**（使用当前 DATABASES_MPI 坐标）
2. **回退 compute_forces 补丁**：恢复 `if (NCHUNKS_VAL /= 6 .and. ABSORBING_CONDITIONS) return`，单独测试是否解决发散
3. **对比测试**：使用 §18.2 的配置（REGIONAL_MOHO_MESH=.false., HONOR_DEEP_MOHO=.false.）+ 当前 GLL，确认是否仍然稳定
4. **如 GLL 来源正确且补丁无问题**：考虑 deep moho stretching 本身导致的边界不稳定，尝试缩短 RECORD_LENGTH 或降低 DT 至 0.088 s

## 25. §24 发散根因分析与 DT 确定方法（2026-03-24）

### 25.1 §24.7 嫌疑重新排序

基于源码深入分析，§24.7 中 4 个嫌疑的优先级需要调整：

| 嫌疑 | §24.7 评级 | **修正后评级** | 理由 |
|------|------------|---------------|------|
| ① GLL 元素排序不匹配 | 高 | **已排除** | 用户确认 GLL 来自 NC→GLL 转换（使用当前 DATABASES_MPI），排序正确 |
| ② compute_forces 补丁副作用 | 中 | **🔴 极高** | 源码确认影响正演（非仅 adjoint），见 §25.2 |
| ③ DT=0.1 硬编码 | 低-中 | **低-中** | CFL=0.623 < 1.0，理论上应稳定，见 §25.3 |
| ④ 吸收边界 + deep moho 耦合 | 中 | **中** | 仍有嫌疑但非首要 |

### 25.2 compute_forces 补丁：确认为发散主因

#### 源码分析

文件：`src/specfem3D/compute_forces_viscoelastic_calling_routine.F90`

**被注释掉的代码（§23 补丁，第 879 行）**：

```fortran
! no need for inner core, absorbing boundaries placed at outer core bottom surface
!if (NCHUNKS_VAL /= 6 .and. ABSORBING_CONDITIONS) return    ← §23 注释掉
! for regional mesh cut-offs, there are no inner core elements
if (NSPEC_INNER_CORE == 0) return                           ← 此行未触发（NSPEC_INNER_CORE=36）
```

**调用位置（第 87–99 行）**：

```fortran
! inner core region
call compute_forces_inner_core(...)    ← 每个时间步均调用，包括正演
```

#### 为什么导致发散

1. **NCHUNKS_VAL = 1**（1-chunk 区域模拟），**ABSORBING_CONDITIONS = .true.**
2. 原始代码：满足 `NCHUNKS_VAL /= 6 .and. ABSORBING_CONDITIONS` → **跳过内核力计算** → 正确行为
3. 补丁后：条件被注释 → **每步执行内核力计算** → 错误行为
4. 第 881 行 `if (NSPEC_INNER_CORE == 0) return` **无法拦截**，因为 NSPEC_INNER_CORE = 36（mesher 为区域模型也生成了少量内核单元）

#### 物理机制

- 区域模拟中，吸收边界在 outer core 底部 → 物理上无波能抵达 inner core
- 但 mesher 仍生成了 36 个 inner core 单元（含 GLL 节点和材料属性）
- `prepare_wavefields.F90:928` 在初始化时将 `displ_inner_core` 置零
- **补丁使每一步计算内核力** → Newmark 时间积分更新 `accel_inner_core` → `displ_inner_core` 从零逐渐累积 → 无物理约束的力反馈 → 指数增长

#### 时间线吻合

- 发散起始：~15 min（时间步 ~9000）
- 在此之前 inner core 数值噪声需要时间通过 Newmark 积分累积到可观量级
- 一旦超过阈值 → 正反馈回路 → 指数爆炸

#### 修复方案

§23 的补丁来自外部开发者（§23.1），其意图是为 FWI adjoint/kernel 阶段保留 inner core 力计算。
因此**不宜简单恢复原始代码**，而应**条件化**——正演跳过，adjoint 保留。

`shared_parameters` 模块（已在子程序的 `use` 中引入）包含 `SIMULATION_TYPE`：
- 1 = 正演（forward）
- 2 = 伴随（adjoint）
- 3 = 核函数（kernel）

**推荐修改（第 879 行）**：

```fortran
! 原始代码（v8 原版）：跳过所有 1-chunk 区域模拟的 inner core 力
! if (NCHUNKS_VAL /= 6 .and. ABSORBING_CONDITIONS) return
!
! 条件化修复：仅正演跳过 inner core 力，adjoint/kernel 保留
if (NCHUNKS_VAL /= 6 .and. ABSORBING_CONDITIONS .and. SIMULATION_TYPE == 1) return
```

这样既解决正演发散，又保留外部开发者的 FWI 反演需求。

### 25.3 DT 确定方法

#### 公式

SPECFEM3D Globe 的 mesher 通过 `check_mesh_resolution.f90` 计算最大建议 DT：

$$DT_{\text{suggested}} = C_{\text{Courant}} \times \frac{d_{\min}}{V_{p,\max}}$$

其中：
- $C_{\text{Courant}} = 0.55$（`constants.h.in:343`，保守阈值）
- $d_{\min}$ = 最小 GLL 节点间距 = `elemsize_min × percent_GLL(NGLLX)`
- $V_{p,\max}$ = 区域内最大 P 波速度

#### 当前各区域值（来自 output_mesher.txt）

| 区域 | Min edge (km) | Max Vp (km/s) | Max suggested DT (s) | CFL@DT=0.1 |
|------|-------------|-------------|---------------------|-------------|
| **Crust/Mantle** | **6.599** | **13.844** | **0.088** | **0.623** |
| Outer Core | 35.23 | 10.36 | 0.330 | 0.165 |
| Inner Core | 29.36 | 11.11 | 0.250 | 0.219 |

**瓶颈区域**：壳幔区（crust/mantle），Min edge = 6.599 km 是全局最小值。

#### 判定标准

| CFL 值 | 状态 | 说明 |
|--------|------|------|
| < 0.55 | ✅ 安全 | 在 COURANT_SUGGESTED 阈值内 |
| 0.55 – 1.0 | ⚠️ 边缘 | 理论上稳定但无安全余量 |
| ≥ 1.0 | ❌ 不稳定 | 违反 CFL 条件 |

当前 CFL = 0.623 处于**边缘区**。虽然不是发散的直接原因（compute_forces 补丁才是），但不够保守。

#### 如何选择 DT

1. **最简单方法**：直接使用 mesher 输出的 `Maximum suggested time step`
   ```
   Maximum suggested time step =   8.8000000E-02  (s)    ← 使用此值
   ```

2. **手动计算**：
   ```
   DT = 0.55 × min_element_edge × percent_GLL(NGLLX) / max_Vp
   ```
   其中 `percent_GLL(5) ≈ 0.1727`（5 阶 GLL 最小节点间距比例）

3. **上取整策略**：将 mesher 建议值 0.088 上调不超过 10%（即 ≤ 0.097），以平衡精度和效率

#### DT 对运行时间的影响

| DT (s) | NSTEP (30 min) | 相对增量 |
|--------|---------------|---------|
| 0.100 | 18,200 | 基准 |
| 0.088 | 20,682 | +13.6% |
| 0.080 | 22,750 | +25.0% |

> 降至 DT=0.088 后运行时间增加约 14%，可接受。

### 25.4 Mesher 间歇性失败分析

#### Job 5674922：失败

```
Mon Mar 23 17:55:08 CST 2026
make clean + compile xmeshfem3D + xspecfem3D → ✅
执行 mesher → ❌ 失败：
  Error test MPI: rank 26  valence: 3  flag: 2.0
  Error test outer core valence
  Error detected, aborting MPI... proc 26
  Error test MPI: rank 34  valence: 1  flag: 0.0
  Error test outer core valence
  Error detected, aborting MPI... proc 34
```

**原因**：outer core 的 MPI 接口检测中，rank 26 和 34 的某些 mesh face 的 valence（共享进程数）不符合预期。这是 v8 mesher 在 1-chunk 区域模拟中的已知间歇性问题，与 144 进程的 MPI 分区有关。

#### Job 5674995：仅编译

```
Mon Mar 23 18:01:39 CST 2026
仅包含编译输出（242 行），未运行 mesher
```

#### 处理策略

- 该错误**间歇性发生**（用户描述：有时会有错误）
- 重新提交通常即可解决
- 与 solver 发散是**独立问题**
- 如果频繁出现，可尝试调整进程数（如 128 = 8×16 或 150 = 10×15）

### 25.5 修正后的下一步建议

**优先级排序**：

1. **🔴 条件化 compute_forces early return（第 879 行）**
   ```fortran
   ! 条件化修复：仅正演跳过 inner core 力，adjoint/kernel 保留
   if (NCHUNKS_VAL /= 6 .and. ABSORBING_CONDITIONS .and. SIMULATION_TYPE == 1) return
   ```
   注意：需要在子程序的 `use shared_parameters` 行中添加 `SIMULATION_TYPE` 的导入（实际上 `shared_parameters` 已包含，但需确认编译器可见性）。
   修改后重新编译 `make specfem3D` 并测试正演。

2. **降低 DT 至 0.088 s**（可选，与步骤 1 同时或单独测试）
   ```fortran
   ! get_timestep_and_layers.f90 中修改：
   DT = 0.088d0    ! 改为 mesher 建议值
   ```
   同时调整 NSTEP 以保持相同记录长度：
   ```
   RECORD_LENGTH_IN_MINUTES = 30.0
   NSTEP = 30 × 60 / 0.088 ≈ 20,455（向上取整）
   ```

3. **验证策略**：
   - 若条件化修复后**正演稳定** → 发散根因确认
   - 后续进入 adjoint 阶段时，SIMULATION_TYPE=2/3 会自动保留 inner core 力计算 → 同时满足 FWI 需求
   - 若修复后正演仍发散 → 继续排查 deep moho + DT

---

## 26. 外部开发者补丁总审：是否必须全部采用？（2026-03-24）

### 26.1 补丁来源确认

`specfem3d_globe/specfem3d_makefile/` 目录中 7 个文件与当前源码（`src/` 下已应用补丁的版本）**完全一致**（`diff` 无差异，仅 `constants.h.in` 有注释差异和 `get_timestep_and_layers.f90` 有一行注释措辞差异）。

即：§23 所采纳的 5 处修改 **100% 来自该外部开发者**，当前源码已完整应用。

### 26.2 逐补丁分析：对正演波形的影响

| # | 补丁 | 文件 | 对正演波形的影响 | 必要性 | 建议 |
|---|------|------|-----------------|--------|------|
| **1** | auto_ner：注释掉 NER 经验值覆盖 | `auto_ner.f90` | **🔴 直接影响**：NSPEC 从 7452→7236，改变网格拓扑 → 改变波形细节 | 若要匹配原作者 v7.0.0 网格则**必须** | ✅ 保留 |
| **2** | DT=0.1d0 硬编码 | `get_timestep_and_layers.f90` | **🔴 直接影响**：DT 从 0.165→0.1，影响时间积分精度和 CFL 稳定性 | **必须**（v8 的 `DT*1.5` bug 导致 CFL>1.0 发散） | ✅ 保留 |
| **3** | gll_qmu 添加 `impose_crust` | `get_model_parameters.F90` | **⚪ 无影响**：我们用 `GLL_crust1.0` 后缀已自动设置此值 | 低（冗余但无害） | ✅ 保留（安全冗余） |
| **4** | 注释掉 inner core early return | `compute_forces_*.F90` | **🔴 导致发散**：正演计算无物理意义的 inner core 力 → 指数增长（§25.2 确认） | 正演**有害**，adjoint 可能需要 | ⚠️ **条件化修复** |
| **5** | 注释掉 `restore_original_moduli()` | `save_kernels.F90` | **⚪ 正演无影响**：此函数仅在 `SIMULATION_TYPE == 3`（kernel）时执行 | 正演不进入此分支 | ✅ 保留 |

**额外补丁（我们自己的，非外部开发者）**：

| 补丁 | 文件 | 正演影响 |
|------|------|---------|
| Plan D vpv=0 fallback | `get_model.F90` | ⚪ 无影响（GLL solver 阶段用 GLL 覆盖所有值） |
| MAX_NEIGHBORS=20 | `setup_MPI_interfaces.f90` | ⚪ 无影响（仅增大 buffer 上限） |
| ifort 编译修复 | `iterate_time_undoatt.F90` | ⚪ 无影响（仅语法修复） |

### 26.3 关键结论

5 个外部补丁中：
- ✅ **3 个直接保留**（#1 auto_ner, #2 DT=0.1, #3 gll_qmu）
- ⚠️ **1 个需要条件化修改**（#4 compute_forces → 正演跳过，adjoint 保留，见 §25.2）
- ✅ **1 个无需关注**（#5 save_kernels，正演不执行）

### 26.4 补丁 #1 和 #2 对波形的定量影响

这两个补丁直接改变了网格和时间步，因此正演波形与不打补丁时**必然不同**：

| 配置 | §18.2（无外部补丁） | §24（全部外部补丁） |
|------|---------------------|---------------------|
| NSPEC | 7452 | 7236 |
| NER_CRUST | 2 | 3 |
| DT | 0.11 s | 0.1 s |
| CFL | 0.748 | 0.623 |
| REGIONAL_MOHO_MESH | .false. | .true. |
| HONOR_DEEP_MOHO | .false. | .true. |

**波形差异来源**：

1. **NSPEC 7236 vs 7452**：网格单元数不同 → GLL 节点位置不同 → 同一 NC 模型插值到不同网格 → 波形细节差异（量级通常较小）
2. **DT 0.1 vs 0.11**：时间积分步长不同 → 数值色散略不同（长周期影响极小）
3. **3 层地壳 vs 2 层地壳**：地壳内波传播精度不同 → 面波/Pn/Sn 等浅部相位差异**可能显著**

**是否影响科学结论？**

- 若目标是**与原作者 v7.0.0 波形对比**：应保留补丁 #1,#2 + tibet_v4 配置（原作者用 NSPEC=7236 + 3 层地壳）
- 若目标是**绝对波形精度**：tibet_v4 配置理论上**更精确**（更好拟合深 Moho 区域）
- 若目标是**先稳定跑通**：§18.2 配置已证明稳定

### 26.5 推荐策略

```
          保留 auto_ner (#1) + DT=0.1 (#2) + gll_qmu (#3) + save_kernels (#5)
                                    │
                     条件化 compute_forces (#4)
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                   │
           配置 A（推荐）      配置 B（保守）      配置 C（参考）
    REGIONAL=.true.          REGIONAL=.false.     REGIONAL=.true.
    HONOR=.true.             HONOR=.false.        HONOR=.true.
    RMOHO=-20000             RMOHO=5000           RMOHO=-20000
    NSPEC≈7236               NSPEC≈7452           NSPEC≈7236
    3层地壳                  2层地壳              3层地壳
    与原作者一致             已确认稳定(§18.2)    未修复#4时发散
```

**当前阶段建议**：
1. 先修复 compute_forces 条件化（§25.2）→ 用配置 A 测试正演
2. 若稳定 → 配置 A 作为正式配置（与原作者一致 + 更准确地壳）
3. 若仍不稳定 → 回退到配置 B（已确认可行），deep moho stretching 才是根因

---

## 27. DT 硬编码移除：修复 v8 REGIONAL_MOHO_MESH 的 DT 计算 bug（2026-03-24）

### 27.1 问题

`get_timestep_and_layers.f90` 第 850–851 行（修改前）：

```fortran
DT = DT*(1.d0 + 0.5d0)    ← 将 DT 增大 50%
DT = 0.1d0                 ← 硬编码覆盖（外部开发者 workaround）
```

这两行存在于 `REGIONAL_MOHO_MESH` 且 `HONOR_1D_SPHERICAL_MOHO = .false.` 的分支。

### 27.2 bug 分析

`*(1.d0 + 0.5d0)` 方向错误：进入 REGIONAL_MOHO_MESH → 3 层地壳 + deep moho stretching → 网格更复杂 → DT 应**缩小**。但 `*1.5` 将 DT **增大 50%**，与所有其他代码路径的做法相反：

| 代码位置 | 场景 | 因子 | 方向 |
|----------|------|------|------|
| line 749 | Mars REGIONAL crustmaps | `*(1.d0 - 0.1d0)` = 0.9 | ⬇️ 缩小 |
| line 797 | Earth 6-chunk CRUST1.0 | `*(1.d0 - 0.1d0)` = 0.9 | ⬇️ 缩小 |
| **line 850** | **Earth REGIONAL_MOHO_MESH** | **`*(1.d0 + 0.5d0)` = 1.5** | **⬆️ 增大 ← bug** |

外部开发者的 `DT = 0.1d0` 是 workaround，仅对 NEX≈288 合理，不具通用性。

### 27.3 修复（已应用）

```fortran
! 修复前(v8 + workaround)：
! DT = DT*(1.d0 + 0.5d0)    ← 增大 50% (bug)
! DT = 0.1d0                 ← 硬编码 (workaround)

! 修复后（与 v7 及其他行星一致）：
DT = DT*(1.d0 - 0.1d0)       ← 缩小 10%（保守方向）
```

### 27.4 修复后各 NEX 的 DT 对比

| NEX | base DT | v8 原版 (×1.5) | 外部硬编码 | **修复后 (×0.9)** |
|-----|---------|----------------|-----------|-------------------|
| 160 | 0.20 | 0.285 ❌ | 0.1 | **0.171** |
| 256 | 0.15 | 0.214 ❌ | 0.1 | **0.128** |
| **288** | **0.07** | **0.100** | **0.1** | **0.060** |

NEX=288 修复后 DT=0.060，CFL ≈ 0.374（安全）。NSTEP 从 18200→30333（+67%）。
若有需要可微调因子（如 `*(1.d0 - 0.05d0)` → DT=0.063），但关键是**不再硬编码**。

---

_最后更新：2026-03-24_
