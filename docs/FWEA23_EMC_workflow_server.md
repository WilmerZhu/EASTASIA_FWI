# FWEA23 EMC 模式正演 - 服务器端流程

曙光 Slurm 集群上完成 FWEA23 模型 **EMC 模式**（NetCDF 直读）正演的完整步骤。

---

## 项目目标

**核心目标**：为**任意公开三维速度模型**提供正演模拟与波形检验工具。

- **背景**：多数公开模型（FWEA23、EARA2024、SinoScope1.0 等）仅发布 **NetCDF (NC)** 或 **CSV** 格式，而 SPECFEM3D Globe 需要特定输入格式。
- **方案选择**：使用 **EMC 模式**（`specfem3d_globe-EMC`），meshfem3D **运行时直接读取 NetCDF** 并插值到 GLL 点，无需预计算 GLL 二进制文件。
- **优势**：流程大幅简化（3 步 vs GLL 的 6 步），无 nspec 绑定，换模型只需替换 `model.nc`。
- **测试用例**：以 **FWEA23** 为例，对比原作者参考波形（`sac/`）和观测数据（`processed_observed/`）验证。

### EMC vs GLL 流程对比

```
GLL (6步):
  编译+meshfem(s362ani,服务器) → 下载DATABASES_MPI(到本地)
    → NC→GLL转换(本地) → 上传GLL(到服务器) → 第二次meshfem+solver(服务器)

EMC (3步):
  NC→EMC格式转换(本地) → 编译(含NetCDF)+meshfem+solver(服务器,一步完成) → 波形验证
```

---

## 0. 前置条件

- **服务器**：`specfem3d_globe-EMC/` 代码已上传（含 `model_EMC.f90` 等 EMC 修改）
- **服务器环境**：Intel 2018 + NetCDF 4.4.1（`module load mathlib/netcdf/4.4.1-intel-2017`）
- **本地**：Python 环境（`conda activate eastasia_fwi`），用于 `convert_to_emc_nc.py`
- **FWEA23 NetCDF**：`data/models/processed/2024_FWEA23/2024_FWEA23_original.nc`（需含 vpv/vph/vsv/vsh/eta/rho）

---

## 1. 流程概览

```
本地: NC→EMC格式转换 (convert_to_emc_nc.py)
  → 上传 specfem3d_globe-EMC/ 到服务器
  → 服务器: run_this_example.sh (编译+meshfem+solver 一步完成)
  → 波形验证
```

**与 GLL 流程的关键区别**：

| 步骤 | GLL 模式 | EMC 模式 |
|------|---------|---------|
| 模型输入 | `proc*_reg1_*.bin`（GB 级） | `model.nc`（~100 MB） |
| 预处理 | `convert_nc_to_gll.py` + 两次 meshfem | `convert_to_emc_nc.py`（一次） |
| mesh 绑定 | 强绑定（nspec 必须匹配） | 无绑定（任意 nspec 可用） |
| 域外处理 | 用户配置 `--horizontal-fill` | `model_EMC.f90` 自动 PREM fallback |
| 换模型 | 重跑 convert_nc_to_gll + 匹配 nspec | 替换 `model.nc` 即可 |

---

## 2. 准备 EMC 模型（本地）

### 2.1 确保 NC 含各向异性参数

确保 `1_5_Process_velocity_models.py` 中 FWEA23 的 params 包含 `eta`：

```python
'params': ['vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho', 'vs0', 'vp0']
```

运行 `1_5` 生成含 eta 的 `2024_FWEA23_original.nc`。

### 2.2 转换为 EMC 格式

```bash
cd /path/to/EASTASIA_FWI

python 3_Data_space_simulation/convert_to_emc_nc.py \
    --model 2024_FWEA23 \
    --anisotropic
```

脚本自动完成：
- 从 vpv/vph/vsv/vsh 计算 Voigt 平均 vp/vs
- NaN/Inf 清洗与极小值保护
- 输出 EMC 格式（depth@positive="down", units="km.s-1"）
- **自动复制**到 `specfem3d_globe-EMC/DATA/IRIS_EMC/model.nc`

### 2.3 验证输出

```bash
ncdump -h specfem3d_globe-EMC/DATA/IRIS_EMC/model.nc
```

应显示：
```
dimensions:
    longitude = 341 ;
    latitude = 241 ;
    depth = 101 ;
variables:
    float vp(longitude, latitude, depth) ;    // km.s-1
    float vs(longitude, latitude, depth) ;    // km.s-1
    float rho(longitude, latitude, depth) ;   // kg.m-3
    float vpv(longitude, latitude, depth) ;   // km.s-1 (各向异性)
    float vph(longitude, latitude, depth) ;   // km.s-1
    float vsv(longitude, latitude, depth) ;   // km.s-1
    float vsh(longitude, latitude, depth) ;   // km.s-1
    float eta(longitude, latitude, depth) ;   // 无量纲
```

关键属性：
- `depth:positive = "down"` 且 `depth:units = "km"`
- 速度 `units = "km.s-1"`（EMC 约定，非 `km/s`）
- 密度 `units = "kg.m-3"`

---

## 3. 编译 + 正演（服务器，一步完成）

### 3.1 上传到服务器

```bash
# 上传整个 specfem3d_globe-EMC/ 到服务器（首次）
scp -r specfem3d_globe-EMC/ user@曙光:/path/to/EASTASIA_FWI/

# 后续仅更新 model.nc
scp specfem3d_globe-EMC/DATA/IRIS_EMC/model.nc \
    user@曙光:/path/to/EASTASIA_FWI/specfem3d_globe-EMC/DATA/IRIS_EMC/
```

### 3.2 运行一键脚本

```bash
cd specfem3d_globe-EMC/EXAMPLES/FWEA23
./run_this_example.sh
```

`run_this_example.sh` 自动完成：

1. **检查 EMC 模型**：确认 `../../DATA/IRIS_EMC/model.nc` 存在
2. **加载编译环境**：`module load compiler/intel/2018.5.274` + `mpi/intelmpi/2018.4.274` + `mathlib/netcdf/4.4.1-intel-2017`
3. **Configure**：`./configure --with-netcdf`，自动设置 `NETCDF_INC` 和 `NETCDF_LIBS`
4. **编译**：`FLAGS_SAFE="-O1 -fp-model precise -fpe0 ..."` 避免 `-O3`/`-xHost` 导致的 SIGABRT
5. **链接数据**：`DATA/IRIS_EMC`, `DATA/topo_bathy`, `DATA/crust1.0`
6. **提交 Slurm 作业**：`sbatch go_mesher_solver_slurm.bash`

**增量编译**（仅改 Par_file/STATIONS/CMTSOLUTION 时）：
```bash
SKIP_CLEAN=1 ./run_this_example.sh
```

### 3.3 Slurm 作业配置

`go_mesher_solver_slurm.bash` 关键参数：

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=36
#SBATCH --time=05:00:00
#SBATCH --partition=tyhcnormal
```

作业流程：meshfem3D → specfem3D（用 `srun --mpi=pmi2`，不用 `mpirun`）。

### 3.4 验证 mesher 输出

```bash
grep "vpv new min/max" OUTPUT_FILES/output_mesher.txt
grep "rho new min/max" OUTPUT_FILES/output_mesher.txt
grep "eta new min/max" OUTPUT_FILES/output_mesher.txt
```

**正确输出**（模型已读入）：
```
vpv new min/max:    1.86 ~ 13.72   (km/s，非常数)
rho new min/max:    1.34 ~ 5.57    (g/cm³，范围合理)
eta new min/max:    0.80 ~ 1.10    (非全 1.0，各向异性已读入)
```

**错误输出**（需排查）：
```
vpv new min/max:    4.50 ~ 4.50    (全常数 → model.nc 路径/格式错误)
rho new min/max: 1500.0 ~ 3300.0   (非 g/cm³ → 单位转换问题)
eta new min/max:    1.00 ~ 1.00    (全 1.0 → NC 无 eta 参数)
```

### 3.5 验证 solver 稳定性

```bash
grep "Max norm displacement" OUTPUT_FILES/output_solver.txt | tail -20
```

`Max norm displacement vector U` 不应指数增长。少量震荡后稳定 → 正常。

---

## 4. 波形验证（测试路线）

### 4.1 三级测试

每级依赖前一级通过：

| 级别 | 做什么 | 验证什么 | 通过标准 |
|------|--------|---------|---------|
| **A 正演可行性** | meshfem + solver 跑完 | EMC 读模型正确、solver 稳定 | mesher 值非常数、Max U 不发散、OUTPUT_FILES/ 有 `.sem.sac` |
| **B 合成波形对比** | EMC 合成 vs 原作者合成 (`sac/`) | EMC 模式精度 | CC > 0.9, time shift < 5s（20–100s 周期） |
| **C 观测波形拟合** | EMC 合成 vs 观测 (`processed_observed/`) | 模型预测能力 | 主要震相（P, S, surface waves）可识别，CC 合理 |

### 4.2 Step A：正演可行性

```bash
# 确认输出文件存在
ls OUTPUT_FILES/*.sem.sac | wc -l

# 检查 mesher 模型值
grep "new min/max" OUTPUT_FILES/output_mesher.txt

# 检查 solver 稳定性
grep "Max norm displacement" OUTPUT_FILES/output_solver.txt | tail -5
```

### 4.3 Step B：合成波形对比

使用 `2_waveform_plotting.py`，对比**我们的 EMC 合成**与**原作者合成**（`sac/` 目录，3768 个 SAC 文件）：

```bash
cd specfem3d_globe-EMC/EXAMPLES/FWEA23
python 2_waveform_plotting.py
```

对比内容：
- 原作者合成：`sac/`（FWEA23 GLL 模式，SPECFEM 7.0.0）
- 我们的合成：`OUTPUT_FILES/`（FWEA23 EMC 模式，SPECFEM 8.1.0-EMC）
- 滤波：0.025–0.1 Hz（10–40s 周期）
- 分量：Z（垂直分量）

**预期差异来源**（不影响结论的合理差异）：
- SPECFEM 版本差异（7.0.0 vs 8.1.0）
- 网格差异（nspec 可能不同，auto_ner 算法差异）
- EMC 运行时插值 vs GLL 预计算插值

### 4.4 Step C：观测波形拟合

使用 `waveform_plotting-all.py`，对比 **EMC 合成** vs **观测数据**：

```bash
cd specfem3d_globe-EMC/EXAMPLES/FWEA23
python waveform_plotting-all.py
```

对比内容：
- 观测数据：`processed_observed/`（20 个台站，BHZ 分量）
- 参考合成：`processed_synthetic/`（原作者处理的 20 个台站）
- 我们的合成：`OUTPUT_FILES/`
- 去仪器响应 + 滤波（0.025–0.1 Hz）

---

## 5. 调试

### 5.1 编译与运行问题

| 错误 | 原因 | 解决 |
|------|------|------|
| Configure 未启用 NetCDF | module 未加载或 `NETCDF_ROOT` 未设 | `module show mathlib/netcdf/...` 确认路径，手动 `export NETCDF_INC=...` |
| `PMI_KVS_Get returned -1` | 曙光 MPI 不兼容 `mpirun` | 用 `srun --mpi=pmi2`（脚本已配置） |
| `request too frequently` | 同上 | 同上 |
| solver exit 6 (SIGABRT) | `-O3`/`-xHost` 数值不稳定 | 确认 `FLAGS_SAFE` 用 `-O1`（脚本已配置） |
| ifort internal error | `-check nobounds` 与 `create_regions_elements` 冲突 | 脚本已用 `sed` 移除 |

### 5.2 模型读取问题

| 错误 | 原因 | 解决 |
|------|------|------|
| vpv/vsv 全为常数 | `model.nc` 路径错误或格式不对 | 检查 `DATA/IRIS_EMC/` 链接是否正确：`ls -la DATA/IRIS_EMC/model.nc` |
| rho 值 1500/3300 | 单位为 kg/m³ 但 model_EMC 期望 g/cm³ | `convert_to_emc_nc.py` 已处理；若手动准备 model.nc 需确认单位 |
| eta 全为 1.0 | NC 无 eta 参数 | 重跑 `1_5` 加 eta，重新 `convert_to_emc_nc.py --anisotropic` |
| `Compiled without EMC/NetCDF support` | 编译时未加 `--with-netcdf` | 重新 configure：`./configure --with-netcdf NETCDF_INC=...` |

### 5.3 波形问题

| 现象 | 可能原因 | 排查方法 |
|------|---------|---------|
| solver 发散（Max U 指数增长） | 模型值不物理（Vs ≈ Vp、负体积模量） | 检查 mesher 输出的 min/max 值 |
| 远场台站波形偏差大 | PREM 域外填充（80° chunk 超出 FWEA23 54° 纬度范围） | 检查台站是否在 FWEA23 模型范围内（70–150°E, 0–54°N） |
| 面波拟合差 | eta 缺失或不准确 | 确认 `eta new min/max` 非全 1.0 |
| P 波到时偏移 | 深部模型差异 | EMC 模式域外（>1000 km）自动用 PREM，与 GLL 行为可能不同 |

---

## 6. EMC vs GLL 详细对比

| 方面 | EMC 模式 | GLL 模式 |
|------|---------|---------|
| **代码** | `specfem3d_globe-EMC/`（独立 fork） | `specfem3d_globe/`（主仓 + patch） |
| **Par_file MODEL** | `emc_model_tiso` | `gll` |
| **输入文件** | `model.nc`（~100 MB） | `proc*_reg1_*.bin`（数 GB） |
| **预处理** | `convert_to_emc_nc.py`（秒级） | `convert_nc_to_gll.py` + 两次 meshfem（分钟~小时） |
| **mesh 绑定** | 无（任意 nspec） | 强绑定（nspec 必须匹配） |
| **域外处理** | `model_EMC.f90` 自动 PREM fallback | 用户配置 `--horizontal-fill {nearest,ref,blend}` |
| **meshfem 速度** | 较慢（运行时 trilinear 插值） | 较快（读二进制数组） |
| **浅层精度** | 受 NC 10 km 深度间隔限制 | 同上（NC 来源相同） |
| **灵活性** | 高（换模型只换 `model.nc`） | 低（换模型需重新转换 + 匹配 nspec） |
| **编译依赖** | 必须 `--with-netcdf` | 无额外依赖 |
| **适用场景** | 快速测试多个模型、生产正演 | 需精确控制域外填充、需与 7.0.0 对比 |

### EMC 模式的技术机制

`model_EMC.f90`（2261 行）的关键行为：

1. **坐标转换**：SPECFEM 内部坐标 (theta, phi, r) → 地理坐标 (lat, lon, depth_km)
2. **Trilinear 插值**：在 NetCDF 规则网格上三线性插值到每个 GLL 点
3. **域外 PREM fallback**：`ENFORCE_EMC_MESH_REGION = .false.`，GLL 点在 NC 范围外时自动用 PREM 1D 背景模型
4. **各向异性支持**：`emc_model_tiso` 模式读取 vpv/vph/vsv/vsh/eta
5. **单位自动检测**：从 NetCDF attributes 读取 `depth_unit`, `vp_unit` 等，自动转换

---

## 7. 路径速查

| 项目 | 路径 |
|------|------|
| EMC 代码 | `specfem3d_globe-EMC/` |
| 示例目录 | `specfem3d_globe-EMC/EXAMPLES/FWEA23/` |
| 编译+提交脚本 | `specfem3d_globe-EMC/EXAMPLES/FWEA23/run_this_example.sh` |
| Slurm 作业脚本 | `specfem3d_globe-EMC/EXAMPLES/FWEA23/go_mesher_solver_slurm.bash` |
| Par_file | `specfem3d_globe-EMC/EXAMPLES/FWEA23/DATA/Par_file` |
| EMC 模型 | `specfem3d_globe-EMC/DATA/IRIS_EMC/model.nc` |
| NC 原始模型 | `data/models/processed/2024_FWEA23/2024_FWEA23_original.nc` |
| 转换脚本 | `3_Data_space_simulation/convert_to_emc_nc.py` |
| 原作者合成波形 | `specfem3d_globe-EMC/EXAMPLES/FWEA23/sac/`（3768 files） |
| 处理后观测 | `specfem3d_globe-EMC/EXAMPLES/FWEA23/processed_observed/`（20 files） |
| 处理后合成 | `specfem3d_globe-EMC/EXAMPLES/FWEA23/processed_synthetic/`（20 files） |
| 波形对比脚本 | `specfem3d_globe-EMC/EXAMPLES/FWEA23/2_waveform_plotting.py` |
| 全波形对比脚本 | `specfem3d_globe-EMC/EXAMPLES/FWEA23/waveform_plotting-all.py` |
| 进程数 | 144 (1 × 12 × 12) |

---

## 8. Par_file 关键参数

当前 FWEA23 EMC 的 Par_file（`specfem3d_globe-EMC/EXAMPLES/FWEA23/DATA/Par_file`）：

### 8.1 与 regional_EMC_model 示例对比

| 参数 | regional_EMC_model | FWEA23 EMC | 说明 |
|------|-------------------|------------|------|
| **MODEL** | `EMC_model` | `emc_model_tiso` | FWEA23 为各向异性 |
| **ANGULAR_WIDTH_XI** | 30.d0 | 80.d0 | 东亚大区域 |
| **ANGULAR_WIDTH_ETA** | 25.d0 | 80.d0 | 与原作者 tibet_v4 一致 |
| **CENTER_LAT/LON** | 64°N, -150°E | 32°N, 106°E | 阿拉斯加 vs 东亚 |
| **GAMMA_ROTATION** | 20.d0 | 60.d0 | 网格旋转角 |
| **NEX** | 32 | 288 | 表面单元数 |
| **NPROC** | 2×2 | 12×12 | MPI 进程数 |
| **RECORD_LENGTH** | 2.5 min | 30.0 min | 记录时长 |
| **REGIONAL_MESH_CUTOFF** | `.true.` (200 km) | `.false.` | FWEA23 不截断 |
| **USE_LOCAL_MESH** | `.true.` | 注释掉 | FWEA23 用 auto_ner 自动决定 |

### 8.2 FWEA23 EMC 完整关键参数

```
NCHUNKS                         = 1
ANGULAR_WIDTH_XI_IN_DEGREES     = 80.d0
ANGULAR_WIDTH_ETA_IN_DEGREES    = 80.d0
CENTER_LATITUDE_IN_DEGREES      = 32.0d0
CENTER_LONGITUDE_IN_DEGREES     = 106.0d0
GAMMA_ROTATION_AZIMUTH          = 60.d0
NEX_XI / NEX_ETA                = 288
NPROC_XI / NPROC_ETA            = 12

MODEL                           = emc_model_tiso
RECORD_LENGTH_IN_MINUTES        = 30.0d0
REGIONAL_MESH_CUTOFF            = .false.
REGIONAL_MESH_CUTOFF_DEPTH      = 400.d0

ABSORBING_CONDITIONS            = .true.
ABSORB_USING_GLOBAL_SPONGE      = .false.
ATTENUATION                     = .true.
```

### 8.3 NEX 约束验证

- NEX 需为 16 的倍数：288 / 16 = 18 ✓
- NEX 需为 8 × NPROC 的倍数：288 / (8 × 12) = 3 ✓
- NEX_XI = NEX_ETA：288 = 288 ✓

---

## 9. 震源与台站

### 9.1 震源（CMTSOLUTION）

```
event name:     C201403131706A
time shift:     0.0
half duration:  3.63
latitude:       33.6423
longitude:      131.8784
depth:          76.6527
```

2014-03-13 九州地震（Mw 6.3），深度 76.65 km。

### 9.2 台站（STATIONS）

约 1800+ 台站，覆盖：
- 中国大陆（AH, BJ, CB, GD, SC 等多省台网）
- 日本（MK, JP 等）
- 中亚（KG, KZ, TJ 等）
- 东南亚（MM, MY, IN 等）
- 台湾（TW）、韩国（KS）
- 全球参考台（IC, IU, II, GE 等）

---

## 10. 开发与调试记录

### 10.1 convert_to_emc_nc.py

| 功能 | 说明 |
|------|------|
| Voigt 平均 | `vp = sqrt((2*vpv² + vph²) / 3)`，`vs` 同理 |
| NaN/Inf 清洗 | 替换为 median，并限制极小值（vp > 0.5, vs > 0.2, rho > 1500） |
| 各向异性输出 | `--anisotropic` 额外输出 vpv, vph, vsv, vsh, eta |
| 自动复制 | 同时复制到 `specfem3d_globe/DATA/IRIS_EMC/` 和 `specfem3d_globe-EMC/DATA/IRIS_EMC/` |
| 密度单位 | 保持 kg/m³（EMC 内部自动转换） |

### 10.2 model_EMC.f90 关键行为

| 行为 | 说明 |
|------|------|
| `ENFORCE_EMC_MESH_REGION = .false.` | 域外 GLL 点用 PREM 1D 背景填充 |
| Trilinear 插值 | 在 NetCDF 规则网格上三线性插值 |
| 各向异性检测 | 自动检测 NC 是否含 vpv/vph/vsv/vsh/eta，有则读入 |
| 单位自动检测 | 从 `units` attribute 判断 km/s vs m/s, km vs m 等 |
| 变量名灵活匹配 | 支持 `vp`, `vpfinal`, `VP`, `vpvar` 等多种命名 |

### 10.3 编译注意事项

| 问题 | 处理 |
|------|------|
| `-O3` 导致 solver SIGABRT | 强制用 `-O1 -fp-model precise` |
| `-xHost` 导致浮点异常 | 脚本自动用 `sed` 移除 |
| `-check nobounds` 导致 ifort 内部错误 | 脚本自动移除 |
| `-warn all,noexternal` 不兼容 | 脚本自动替换为 `-warn all` |
| Makefile 注入 `-O3` | `sed` 替换为 `-O1` |

### 10.4 plotting 脚本

| 修改 | 说明 |
|------|------|
| `2_waveform_plotting.py` data_dir | 从 `sem_config_tibet_v4/DATA` 修正为当前 `FWEA23/DATA` 路径 |
| `waveform_plotting-all.py` data_dir | 同上 |

### 10.5 FWEA23 EMC 与 GLL 版本的 Par_file 差异

| 参数 | GLL 版本 | EMC 版本 | 说明 |
|------|---------|---------|------|
| MODEL | `gll` | `emc_model_tiso` | 核心区别 |
| REGIONAL_MESH_CUTOFF | `.false.` | `.false.` | 一致 |
| USE_LOCAL_MESH | 未设 | 注释掉 | 一致（均用 auto_ner） |
| 编译依赖 | 无额外 | `--with-netcdf` | EMC 需 NetCDF |
