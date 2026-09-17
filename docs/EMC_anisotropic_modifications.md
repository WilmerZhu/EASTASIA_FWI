# EMC 模型各向异性适配修改说明

本文档记录为支持 EMC 模型各向异性参数（vpv, vph, vsv, vsh, eta）所做的代码修改。

---

## 1. `3_Data_space_simulation/convert_to_emc_nc.py`

### 1.1 新增 `--anisotropic` 参数

- **位置**：`argparse` 参数解析
- **作用**：启用时，若源模型含 vpv/vph/vsv/vsh，则额外输出各向异性变量到 NetCDF

```python
parser.add_argument('--anisotropic', action='store_true',
                    help='输出各向异性参数 vpv/vph/vsv/vsh/eta（需源模型含 vpv,vph,vsv,vsh）')
```

### 1.2 `convert_to_emc()` 函数修改

| 修改项 | 说明 |
|--------|------|
| 新增参数 `anisotropic: bool = False` | 控制是否输出各向异性参数 |
| 新增变量 `vpv, vph, vsv, vsh, eta` | 从源 NetCDF 读取各向异性参数 |
| 分支逻辑 | 当源含 vpv/vph/vsv/vsh 且 `anisotropic=True` 时读取并保留 |
| **补充逻辑** | 当源**同时含** vp/vs 和 vpv/vph/vsv/vsh 时，`--anisotropic` 仍会读取各向异性参数（此前会漏读） |
| `sanitize()` | 对各向异性参数做 NaN/Inf 替换及极小值限制 |
| `data_vars` | 输出 vpv, vph, vsv, vsh, eta 到 NetCDF（eta 缺失时用 1.0） |

### 1.3 输出变量

各向异性模式下，除 vp、vs、rho 外，额外输出：

- `vpv`：垂直方向 P 波速度 (km.s-1)
- `vph`：水平方向 P 波速度 (km.s-1)
- `vsv`：垂直方向 S 波速度 (km.s-1)
- `vsh`：水平方向 S 波速度 (km.s-1)
- `eta`：横向各向同性参数（无则填 1.0）

---

## 2. `specfem3d_globe/src/meshfem3D/model_EMC.f90`

### 2.1 `model_emc_par` 模块（约第 49–182 行）

#### 新增变量声明（第 49–52 行）

```fortran
  ! 各向异性参数（可选，若 NetCDF 含 vpv/vph/vsv/vsh/eta 则使用）
  logical :: EMC_has_aniso = .false.
  real(kind=CUSTOM_REAL), dimension(:,:,:), allocatable :: EMC_vpv, EMC_vph, EMC_vsv, EMC_vsh, EMC_eta
```

- **EMC_has_aniso**：逻辑标志，表示 NetCDF 是否包含各向异性参数
- **EMC_vpv, EMC_vph, EMC_vsv, EMC_vsh, EMC_eta**：三维数组，仅在检测到各向异性时分配

#### 新增变量名列表（第 171–181 行）

```fortran
  ! 各向异性变量名（可选）
  character(len=16), dimension(4), parameter :: vpvnames = (/ character(len=16) :: &
  'vpv','vpvvar','vpv_var','vpv-var' /)
  character(len=16), dimension(4), parameter :: vphnames = (/ character(len=16) :: &
  'vph','vphvar','vph_var','vph-var' /)
  character(len=16), dimension(4), parameter :: vsvnames = (/ character(len=16) :: &
  'vsv','vsvvar','vsv_var','vsv-var' /)
  character(len=16), dimension(4), parameter :: vshnames = (/ character(len=16) :: &
  'vsh','vshvar','vsh_var','vsh-var' /)
  character(len=16), dimension(4), parameter :: etanames = (/ character(len=16) :: &
  'eta','etavar','eta_var','eta-var' /)
```

用于在 NetCDF 中匹配各向异性变量名。

---

### 2.2 `check_varnames` 子程序（约第 378–453 行）

#### 修改内容

1. **子程序接口**：增加 5 个输出参数  
   `varid_vpv, varid_vph, varid_vsv, varid_vsh, varid_eta`

2. **初始化**：将上述 varid 初始化为 0

3. **变量遍历**：在原有 vp/vs/rho 判断之后，增加对各向异性变量的识别：
   ```fortran
   else if (any(vpvnames == trim(varname))) then
     varid_vpv = varid
   else if (any(vphnames == trim(varname))) then
     varid_vph = varid
   else if (any(vsvnames == trim(varname))) then
     varid_vsv = varid
   else if (any(vshnames == trim(varname))) then
     varid_vsh = varid
   else if (any(etanames == trim(varname))) then
     varid_eta = varid
   ```

4. **调试输出**：在 VERBOSE 模式下打印各向异性 varid

#### 功能

- 自动识别 NetCDF 中的 vpv、vph、vsv、vsh、eta
- 支持多种命名（如 vpv、vpvvar、vpv_var 等）

---

### 2.3 `read_emc_model` 子程序（约第 988–1578 行）

#### 修改 1：局部变量（约第 1008 行）

```fortran
  integer :: varid_vpv, varid_vph, varid_vsv, varid_vsh, varid_eta
```

#### 修改 2：调用 `check_varnames`（约第 1101–1102 行）

```fortran
  call check_varnames(ncid, varid_vp, varid_vs, varid_rho, varid_lat, varid_lon, varid_dep, &
                      varid_vpv, varid_vph, varid_vsv, varid_vsh, varid_eta)
```

#### 修改 3：各向异性读取（约第 1346–1364 行）

在读取 vp、vs、rho 之后，若 vpv、vph、vsv、vsh 的 varid 均非 0，则：

1. 分配 `EMC_vpv, EMC_vph, EMC_vsv, EMC_vsh, EMC_eta`
2. 调用 `nf90_get_var` 读取上述变量
3. 若 NetCDF 无 eta，则 `EMC_eta` 全部置为 1.0
4. 设置 `EMC_has_aniso = .true.`
5. 输出提示：`anisotropic model (vpv,vph,vsv,vsh,eta) detected and loaded`

#### 修改 4：单位转换（约第 1536–1542 行）

在 vp、vs 的 km/s→m/s 转换之后，增加各向异性数组的单位转换：

```fortran
  ! 各向异性数组单位转换（convert_to_emc_nc 输出 km/s，需转为 m/s）
  if (EMC_has_aniso) then
    EMC_vpv(:,:,:) = EMC_vpv(:,:,:) * 1000.d0
    EMC_vph(:,:,:) = EMC_vph(:,:,:) * 1000.d0
    EMC_vsv(:,:,:) = EMC_vsv(:,:,:) * 1000.d0
    EMC_vsh(:,:,:) = EMC_vsh(:,:,:) * 1000.d0
  endif
```

---

### 2.4 `model_emc_broadcast` 子程序（约第 867–982 行）

#### 新增代码（约第 967–980 行）

在原有 vp、vs、rho、mask 广播之后：

1. **广播标志**：`call bcast_all_singlel(EMC_has_aniso)`

2. **若 `EMC_has_aniso` 为真**：
   - 在 `myrank /= 0` 的进程上分配各向异性数组
   - 广播 `EMC_vpv, EMC_vph, EMC_vsv, EMC_vsh, EMC_eta`

#### 功能

- 保证所有 MPI 进程都能访问各向异性模型数据
- 仅在检测到各向异性时分配和广播，避免多余内存

---

### 2.5 `model_EMC_crustmantle` 子程序（约第 1652–2240 行）

#### 修改 1：局部变量（约第 1677 行）

```fortran
  double precision :: vpv_l,vph_l,vsv_l,vsh_l,eta_l
```

用于存放插值得到的各向异性参数。

#### 修改 2：各向异性插值（约第 2103–2215 行）

当 `EMC_has_aniso` 为真时，对 vpv、vph、vsv、vsh、eta 分别做三线性插值：

1. **vpv**：取 8 个格点值，对 mask 点用 `vp_iso` 替代，再按三线性公式插值
2. **vph**：同上，mask 点用 `vp_iso`
3. **vsv**：mask 点用 `vs_iso`
4. **vsh**：mask 点用 `vs_iso`
5. **eta**：mask 点用 1.d0

插值公式（以 vpv 为例）：
```
vpv_l = Σ val_i × (1-γx)^(1-δx) × (1-γy)^(1-δy) × (1-γz)^(1-δz)
```
其中 γx, γy, γz 为插值权重，δx, δy, δz 为 0 或 1，取决于格点位置。

6. **非维数化**：将 vpv_l、vph_l、vsv_l、vsh_l 乘以 `scaleval_vel`（m/s → 无量纲）

#### 修改 3：赋值逻辑（约第 2221–2236 行）

原逻辑为：`vpv=vph=vpl, vsv=vsh=vsl, eta_aniso=1.d0`（各向同性近似）。

新逻辑：

- **若 `EMC_has_aniso` 为真**：使用插值得到的各向异性值  
  `vpv=vpv_l, vph=vph_l, vsv=vsv_l, vsh=vsh_l, eta_aniso=eta_l`
- **否则**：保持原各向同性赋值

#### 功能

- 在 EMC 覆盖区域内，按各向异性模型给出 vpv、vph、vsv、vsh、eta
- 对缺失值（mask）使用背景模型（PREM）的 Voigt 平均或 eta=1
- 与原有 vp、vs、rho 插值流程一致，仅增加各向异性分支

---

## 3. 使用流程

### 3.1 转换模型（各向异性）

```bash
python 3_Data_space_simulation/convert_to_emc_nc.py --model 2024_FWEA23 --anisotropic
```

- 源模型需含 vpv、vph、vsv、vsh（可同时含 vp、vs）
- 输出会复制到 `specfem3d_globe/DATA/IRIS_EMC/model.nc`

### 3.2 各向同性模式（默认）

```bash
python 3_Data_space_simulation/convert_to_emc_nc.py --model 2024_FWEA23
```

- 仅输出 vp、vs、rho
- model_EMC 自动按各向同性处理（eta=1）

### 3.3 正演运行

- Par_file 中 `MODEL = EMC_model` 无需改动
- 修改 `model_EMC.f90` 后需重新编译 meshfem 和 solver
- 运行后可在 `output_solver.txt` 中查看是否出现：
  ```
  anisotropic model (vpv,vph,vsv,vsh,eta) detected and loaded
  ```

---

## 4. 兼容性

- **向后兼容**：无 vpv/vph/vsv/vsh 的 NetCDF 仍按原逻辑处理
- **可选各向异性**：有各向异性变量时自动启用，无需额外配置
- **混合模型**：同时含 vp/vs 和 vpv/vph/vsv/vsh 时，`--anisotropic` 会正确输出各向异性参数

---

*文档更新日期：2026-03-09*
