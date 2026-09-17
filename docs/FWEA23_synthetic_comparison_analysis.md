# FWEA23 正演结果差异分析

本文档分析您与 FWEA23 原作者（Chujie）正演结果存在差异的可能原因。基于已知的版本、参数和代码修改进行归纳。

---

## 1. 已知差异概览

| 类别 | 您方 | FWEA23 原作者 |
|------|------|---------------|
| SPECFEM3D Globe 版本 | 8.x（含 EMC 支持） | 7.0.0 |
| MODEL | EMC_model（FWEA23） | GLL（FWEA23） |
| Par_file 衰减 | UNDO_ATTENUATION = .false. | **UNDO_ATTENUATION = .true.** |
| CMTSOLUTION 格式 | 标准 geographic | ECEF（x,y,z,Mxx...） |
| DT / CFL | COURANT_SUGGESTED = 0.15 | 默认 0.55 |
| REGIONAL_MESH_CUTOFF | .true. (v8) | 无（v7 无此参数） |

---

## 2. 各因素对波形的影响

### 2.1 Q（衰减）处理 —— 影响最大（已确认）

**参数**：`UNDO_ATTENUATION`、`PARTIAL_PHYS_DISPERSION_ONLY`

- **原作者 Par_file**：`UNDO_ATTENUATION = .true.`，`PARTIAL_PHYS_DISPERSION_ONLY = .false.`
- **您当前 Par_file**：`UNDO_ATTENUATION = .false.`，`PARTIAL_PHYS_DISPERSION_ONLY = .false.`

**含义**：
- `UNDO_ATTENUATION = .true.`：正演中“撤销”衰减，输出合成地震图**几乎无衰减**
- `UNDO_ATTENUATION = .false.`：正演中**完整衰减**，输出包含衰减效应

**对波形的影响**：
- 振幅：您的结果整体会更小（衰减更强），原作者振幅更大
- 相位：衰减会改变相位，尤其在长周期（>20 s）更明显
- 体波与面波：面波对 Q 更敏感，差异会更大

**注意**：`UNDO_ATTENUATION = .true.` 需要较多内存和磁盘空间（用于存储中间波场），需确保 `MEMORY_INSTALLED_PER_CORE_IN_GB` 和 `PERCENT_OF_MEM_TO_USE_PER_CORE` 足够。

---

### 2.2 DT / CFL（时间步长）—— 影响中等

**参数**：`COURANT_SUGGESTED`（在 `constants.h.in`）

- **上游默认**：0.55
- **您当前**：0.15（为 FWEA23 网格稳定性而改）

**含义**：
- 时间步长 dt 与 COURANT 成正比
- 0.15 时 dt 约为 0.55 时的 1/3.7，时间步更小、更保守

**对波形的影响**：
- 数值精度：更小 dt 一般更稳定、精度更高
- 相位：可能带来轻微相位漂移（通常 < 1 个采样点）
- 振幅：影响很小

**结论**：DT 差异通常不是主要差异来源，更多是数值实现上的细微差别。

---

### 2.3 CMTSOLUTION 格式 —— 重大差异（已确认）

FWEA23 原作者使用 **ECEF 格式**，通过 `change_cmt_to_ECEF.py` 将标准格式转换为 ECEF 后供其修改版 SPECFEM 读取。

#### 原作者格式（ECEF，来自 Downloads/CMTSOLUTION）

```
PDEW 2014 03 13 17 06 54.6452 33.68 131.82 79.0 0.0 6.3 KYUSHU, JAPAN
event name:        C201403131706A
t0(s):             +0.00000000E+00
tau(s):            +2.22997142E+00
x(m):              -3.50566942E+06
y(m):              +3.91009303E+06
z(m):              +3.47101823E+06
Mxx(N*m):          +1.37906491E+18
Myy(N*m):          -2.16426373E+18
Mzz(N*m):          +7.96610103E+17
Mxy(N*m):          -4.26653945E+16
Mxz(N*m):          -2.99180324E+18
Myz(N*m):          -1.09080110E+18
```

#### 您当前格式（标准 geographic）

```
PDE 2014/03/13 17:06:50.8  33.6800  131.8200  79.0 0.0 6.3 KYUSHU, JAPAN
event name: 20140313_170650
time shift: 0.0
half duration: 4.5
latitude:  33.6800
longitude:  131.8200
depth:  79.0
Mrr: 1.070000e+25
Mtt: -3.670000e+24
...
```

#### 关键差异

| 项目 | 原作者 (ECEF) | 您方 (标准) |
|------|---------------|-------------|
| **坐标** | x,y,z (m) 地心直角 | lat,lon,dep (km) 地理 |
| **矩张量** | Mxx,Myy,Mzz,Mxy,Mxz,Myz (N·m) | Mrr,Mtt,Mpp,Mrt,Mrp,Mtp (dyne·cm) |
| **时间参数** | t0=0, tau=half_duration/1.628 | time_shift, half_duration |
| **震源时间** | tau≈2.23 s（对应 half_duration≈3.6 s） | half_duration=4.5 s |
| **header 时间** | 质心时间 17:06:54.6452 | 发震时间 17:06:50.8 |
| **header 格式** | 空格分隔 `2014 03 13 17 06 54.6452` | 斜杠 `2014/03/13 17:06:50.8` |

#### 转换脚本要点（change_cmt_to_ECEF.py）

- **tau**：`tau = half_duration / 1.628`（用高斯近似三角形，见 constants.h.in 中 `SOURCE_DECAY_MIMIC_TRIANGLE = 1.628`）
- **质心时间**：header 中写入 `origin + time_shift`，文件中 `t0=0`
- **矩张量**：球坐标 (r,θ,φ) → 直角坐标 (x,y,z)，单位 dyne·cm → N·m（×1e-7）

#### 对波形的影响

1. **half_duration**：原作者 tau≈2.23 对应 half_duration≈3.6 s，您为 4.5 s，震源时间函数更宽，会影响振幅和相位。
2. **质心时间**：若原作者 time_shift≈3.8 s 而您为 0，会引入约 3.8 s 的相位差。
3. **格式兼容**：标准 SPECFEM 只读 geographic 格式；ECEF 需修改源码。您若用标准格式，则与他们的实现路径不同。

**建议**：若要尽量复现原作者结果，可尝试：

1. **在标准格式下对齐参数**（无需改 SPECFEM）：
   - `half_duration` ≈ 3.6 s（对应其 tau≈2.23）
   - `time_shift` ≈ 3.85 s（使其质心时间与 17:06:54.65 一致）

2. **使用其转换脚本**：将您的标准 CMTSOLUTION 作为输入，运行 `change_cmt_to_ECEF.py` 得到 ECEF 格式；但标准 SPECFEM 8.x 不读 ECEF，需确认其 7.0.0 是否打了 ECEF 补丁。

---

### 2.4 SPECFEM3D 版本差异（7.0.0 vs 8.x）

**v7.0.0 特点**（来自 release notes）：
- 地心/地理坐标转换的 bug 修复
- AK135 模型更新
- 椭率和重力因子更新
- 使用二进制地形文件
- **地震图命名**：由 `station.network..` 改为 `network.station..`（IRIS 约定）

**v8.0.0 新增**：
- 区域网格截断（REGIONAL_MESH_CUTOFF）
- 单色震源时间函数、海绵吸收边界
- LDDRK 在 GPU 上的支持
- GLL 模型对方位各向异性和 Q 的支持
- ADIOS2 等

**v8.1.0**：
- 增加 EMC 模型支持（您在使用）

**对波形的影响**：
- 7.0→8.0 的坐标、椭率、重力等修正可能带来小幅相位/振幅变化
- 区域网格截断、吸收边界等会改变边界附近的波形
- 若原作者用 7.0.0 且未启用区域截断，而您用 8.x 的区域截断，差异会更大

---

### 2.5 Par_file 差异（已与原作者 Par_file 对比）

以下为与 FWEA23 原作者 Par_file（Downloads/Par_file）的逐项对比：

| 参数 | 原作者 | 您方 | 影响 |
|------|--------|------|------|
| **MODEL** | GLL（FWEA23） | EMC_model（FWEA23） | 格式不同（GLL 网格 vs NetCDF），底层数据相同 |
| **UNDO_ATTENUATION** | **.true.** | **.false.** | **影响最大**：他们“撤销”衰减，您完整衰减 |
| PARTIAL_PHYS_DISPERSION_ONLY | .false. | .false. | 一致 |
| RECORD_LENGTH_IN_MINUTES | 30.0 | 60.0 | 记录长度不同 |
| EXACT_MASS_MATRIX_FOR_ROTATION | .true. | .false. | 旋转精度不同 |
| PERCENT_OF_MEM_TO_USE_PER_CORE | 90.d0 | 85.d0 | 内存使用比例 |
| REGIONAL_MESH_CUTOFF | 无（v7 无此参数） | .true. | 您使用区域截断 |
| REGIONAL_MESH_CUTOFF_DEPTH | - | 400.d0 | 同上 |
| SAVE_MESH_FILES | .false. | .true. | 仅输出 |
| MOVIE_COARSE | .false. | .true. | 仅输出 |
| OUTPUT_SEISMOS_ASCII_TEXT | .false. | .true. | 输出格式 |
| ANISOTROPIC_KL | .true. | .false. | 反演核（正演无关） |
| APPROXIMATE_HESS_KL | .true. | .false. | 反演核（正演无关） |
| USE_FULL_TISO_CRUST_220 等 | 有（v7 风格） | USE_FULL_TISO_MANTLE | 版本参数差异 |

#### UNDO_ATTENUATION 的影响（最重要）

- **原作者**：`UNDO_ATTENUATION = .true.` → 正演中“撤销”衰减，合成地震图**几乎无衰减**
- **您方**：`UNDO_ATTENUATION = .false.` → 正演中**完整衰减**

因此您的波形振幅会更小、相位更易偏移，尤其在长周期和面波上，与原作者差异会非常明显。

---

### 2.6 速度模型格式（底层数据相同）

- **底层数据**：双方均使用 FWEA23 速度模型
- **格式差异**：原作者用 GLL（SPECFEM 原生网格格式），您用 EMC_model（NetCDF → 插值到网格）
- **可能影响**：EMC 的 NetCDF 插值、各向异性处理（vpv/vph/vsv/vsh/eta）可能与 GLL 直接存储有细微差异，但不应是主要差异来源

---

## 3. 差异来源优先级（估计）

1. **UNDO_ATTENUATION**：原作者 .true.（无衰减输出） vs 您 .false.（完整衰减）→ **振幅与相位差异最大**  
2. **CMTSOLUTION**：ECEF vs 标准、half_duration (3.6 vs 4.5 s)、time_shift (≈3.8 vs 0) → **震源差异显著**  
3. **速度模型格式**：GLL vs EMC（底层均为 FWEA23，格式/插值可能引入小差异）  
4. **版本与网格**：7.0 vs 8.x、区域截断、吸收边界 → 中等影响  
5. **DT/CFL**：时间步长差异 → 影响相对较小  

---

## 4. 建议的验证步骤

1. **对齐 UNDO_ATTENUATION（优先）**  
   将 `UNDO_ATTENUATION` 设为 `.true.`，重新正演。若内存足够，这应能显著缩小与原作者波形的振幅差异。

2. **对齐 CMTSOLUTION 参数**  
   在标准格式下尝试：`half_duration` ≈ 3.6 s，`time_shift` ≈ 3.85 s，或使用其 `change_cmt_to_ECEF.py` 得到 ECEF 格式（需确认其 SPECFEM 是否支持）。

3. **统一 Par_file 其他项**  
   可尝试将 `EXACT_MASS_MATRIX_FOR_ROTATION` 设为 `.true.`，`RECORD_LENGTH_IN_MINUTES` 设为 30.0，以更接近原作者设置。

4. **控制变量对比**  
   - 速度模型已相同（FWEA23），差异主要在格式（GLL vs EMC 插值）
   - 可逐步调整 UNDO_ATTENUATION、CMTSOLUTION、区域截断等，观察差异变化

---

## 5. 参考文献与链接

- [SPECFEM3D_GLOBE v7.0.0 Release](https://github.com/SPECFEM/specfem3d_globe/releases/tag/v7.0.0)
- [SPECFEM3D_GLOBE v8.0.0 Release](https://github.com/SPECFEM/specfem3d_globe/releases/tag/v8.0.0)
- [SPECFEM3D_GLOBE v8.1.0 Release](https://github.com/SPECFEM/specfem3d_globe/releases/tag/v8.1.0)
- [CMTSOLUTION 格式说明 (ObsPy)](https://docs.obspy.org/master/packages/obspy.io.cmtsolution.html)

---

## 6. 源码分析与波形对比脚本（2026-03-12 补充）

### 6.1 SPECFEM get_cmt.f90 要点

- **格式**：仅支持标准 geographic（latitude, longitude, depth, Mrr, Mtt, Mpp...），**不支持 ECEF**
- **tshift_src**：单震源时强制设为 0（第 288–291 行），故 header 中的时间即为有效“发震时间”
- **hdur**：`hdur_Gaussian = hdur / SOURCE_DECAY_MIMIC_TRIANGLE`（1.628），与 tau = half_duration/1.628 一致
- **header 解析**：自由格式，跳过 datasource 后读取 year, month, day, hour, minute, seconds

### 6.2 波形对比脚本潜在问题（已修复）

**2_waveform_plotting.py** 原时间轴计算：
```python
chujie_time = np.arange(npts) * delta  # 错误：从 0 开始
```

SAC 的 **B**（begin）表示首样点相对 origin 的时间。若 Chujie 与您的 B 不同，会导致**相位错位**。

**修复**：使用 `time = B + np.arange(npts) * delta`，并兼容 SAC 未定义值（-12345）。

### 6.3 剩余差异的可能来源

| 来源 | 说明 |
|------|------|
| **GLL vs EMC** | 原作者用 GLL 网格直接存储，您用 NetCDF 插值，插值误差可能引入小差异 |
| **REGIONAL_MESH_CUTOFF** | v8 区域截断改变网格与吸收边界，v7 无此功能 |
| **版本差异** | 7.0→8.x 的坐标、椭率、重力等修正 |
| **ECEF 实现** | 原作者 ECEF 格式需修改 get_cmt，其实现细节未知 |

### 6.4 原作者源码分析（specfem/specfem3d_globe_code_new）

已确认其 7.0.0 修改版实现：

#### ECEF 支持（constants.h.in）

```fortran
logical, parameter :: USE_ECEF_CMTSOLUTION = .true.
```

#### get_cmt.f90 修改要点

| 项目 | 标准格式 | ECEF 格式（USE_ECEF_CMTSOLUTION=.true.） |
|------|----------|----------------------------------------|
| 第 3 行 | time shift | t0(s) |
| 第 4 行 | half duration | tau(s) → 读入后 `hdur = hdur * 1.628` 转为 half_duration |
| 第 5–7 行 | lat, lon, depth (度, 度, km) | x, y, z (m)，存入 lat/long/depth 变量 |
| 第 8–13 行 | Mrr…Mtp (dyne·cm) | Mxx…Myz (N·m)，**不做** ×1e-7 |
| 矩张量缩放 | ×1e-7 再除以 scaleM | 仅除以 scaleM |

#### locate_sources.f90

- **ECEF 时**：`lat,long,depth` 存的是 x,y,z (m)，直接除以 R_EARTH 得到无量纲坐标
- **矩张量**：ECEF 时 `moment_tensor(1:6)` 已是 Mxx,Myy,Mzz,Mxy,Mxz,Myz，直接赋给 Mxx(isource) 等

#### 与您当前实现的差异

1. **您方**：标准 geographic 格式，无 ECEF
2. **若要完全对齐**：需在您的 specfem3d_globe 中移植 `USE_ECEF_CMTSOLUTION` 及 get_cmt/locate_sources 的 ECEF 分支，或用 `change_cmt_to_ECEF.py` 生成 ECEF 文件后，在标准格式下尽量逼近其参数（质心时间、tau/half_duration 等）

### 6.5 NC/CSV → GLL 格式转换（供 8.1.0 使用）

若希望与原作者一样使用 GLL 格式（`MODEL = 1D_isotropic_GLL` 或 `1D_transversely_isotropic_GLL`），需将 NetCDF/CSV 模型转换为 GLL 二进制。

**原理**：GLL 文件与 SPECFEM 网格一一对应，每个 MPI 进程一个文件，形状 `(NGLLX, NGLLY, NGLLZ, nspec)`，数据在 GLL 点上。转换需：

1. **先运行 meshfem** 生成 `DATABASES_MPI`（含 `proc***_reg1_solver_data.bin`）
2. **读取 solver_data.bin** 获取每个 GLL 点的 (x,y,z) 笛卡尔坐标
3. **转换为 (lat, lon, depth)**，从 NetCDF/CSV 插值得到 vp/vs/rho 等
4. **写出** `proc***_reg1_vp.bin`, `vs.bin`, `rho.bin`（各向同性）或 `vpv`, `vph`, `vsv`, `vsh`, `eta`, `rho`（横向各向同性）

**项目脚本**：`3_Data_space_simulation/convert_nc_to_gll.py`

```bash
# 1. 运行 meshfem 生成网格（可用任意模型如 PREM）
cd specfem3d_globe/EXAMPLES/FWEA23
./run_mesher_solver.bash  # 或仅 meshfem 步骤

# 2. 转换 NC → GLL
python 3_Data_space_simulation/convert_nc_to_gll.py \
    --mesh-dir specfem3d_globe/EXAMPLES/FWEA23/DATABASES_MPI \
    --model-nc data/models/processed/2024_FWEA23/2024_FWEA23_original.nc \
    --output-dir specfem3d_globe/EXAMPLES/FWEA23/DATA/GLL \
    --anisotropic
```

**Par_file 配置**：
- `MODEL = 1D_transversely_isotropic_GLL`（若用 --anisotropic）或 `1D_isotropic_GLL`
- `PATHNAME_GLL_modeldir = DATA/GLL/`

**注意**：`xinterpolate_model` 仅用于**已有 GLL 网格之间的插值**，不能直接读 NetCDF。从 NC/CSV 创建 GLL 需使用上述脚本。

### 6.6 建议的进一步排查

1. **检查 SAC B 值**：对比 Chujie 与您输出的 SAC 中 B、O，确认时间参考一致
2. **移植 ECEF 支持**：从 `specfem3d_globe_code_new` 将 ECEF 相关逻辑移植到您的 8.x，或改用其 7.0.0 修改版做对比
3. **控制变量**：在相同 Par_file、CMTSOLUTION 下，仅切换 GLL/EMC 或 REGIONAL_MESH_CUTOFF，观察差异
4. **GLL 格式对比**：用 `convert_nc_to_gll.py` 生成 GLL 后，与 EMC 结果对比，量化插值差异

---

*文档创建日期：2026-03-09，更新：2026-03-12*
