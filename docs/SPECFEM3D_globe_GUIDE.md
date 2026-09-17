# SPECFEM3D Globe 正演流程手册

EASTASIA-FWI 使用 SPECFEM3D Globe 进行东亚区域地震波正演。本手册涵盖安装、EMC 模型、FWEA23 示例、曙光 Slurm 流程及常见问题。

---

## 一、安装与配置

### 1.1 克隆源码

```bash
# 在项目根目录执行，克隆 v8.1.0（项目推荐版本）
git clone --depth 1 --branch v8.1.0 https://github.com/SPECFEM/specfem3d_globe.git specfem3d_globe

# 或指定其他版本
git clone --branch v8.0.0 https://github.com/SPECFEM/specfem3d_globe.git specfem3d_globe
cd specfem3d_globe && git checkout v8.0.0
```

`specfem3d_globe/` 已在 `.gitignore` 中排除，不会纳入版本控制。

### 1.2 编译

```bash
cd specfem3d_globe
./configure
make clean
make -j
```

### 1.3 环境要求

- **MPI**: OpenMPI 或 MPICH（曙光超算使用 Intel MPI）
- **Fortran 编译器**: gfortran, ifort 等
- **C 编译器**: gcc, icc 等
- **NetCDF**（EMC 模型必需）: 需 `--with-netcdf` 编译

### 1.4 EMC 模型编译（必需）

使用 EMC NetCDF 模型时，必须启用 NetCDF 支持：

```bash
cd specfem3d_globe
./configure --with-netcdf NETCDF_INC=/path/to/include NETCDF_LIBS="-L/path/to/lib -lnetcdff"
make clean
make -j
```

曙光超算上若 NetCDF 模块可用：

```bash
module load netcdf/xxx  # 根据实际模块名
./configure --with-netcdf NETCDF_INC=$NETCDF_INC NETCDF_LIBS="$NETCDF_LIBS"
make -j
```

### 1.5 可执行文件

SPECFEM3D Globe 区域模拟使用：

| 可执行文件 | 用途 |
|------------|------|
| `xmeshfem3D` | 网格生成 |
| `xspecfem3D` | 正演计算 |

注意：区域模拟**无** `xgenerate_databases` 步骤，流程为 xmeshfem3D → xspecfem3D。

---

## 二、EMC 模型格式

### 2.1 文件路径

SPECFEM3D 从以下路径读取 EMC 模型：

```
DATA/IRIS_EMC/model.nc
```

运行目录下需有 `DATA/IRIS_EMC/`，其中 `model.nc` 为符号链接或实际文件。

### 2.2 维度/坐标名称（支持多种别名）

| 类型 | 支持名称 |
|------|----------|
| 纬度 | latitude, lat, y, ydim, y_dim, y-dim |
| 经度 | longitude, lon, x, xdim, x_dim, x-dim |
| 深度 | depth, dep, z, zdim, z_dim, z-dim |

### 2.3 速度/密度变量

| 变量 | 支持名称 | 说明 |
|------|----------|------|
| Vp | vp, vpfinal, VP | 至少需 vp 或 vs 之一 |
| Vs | vs, vsfinal, VS | 缺失时可用 Brocher 从 Vp 推算 |
| Rho | rho, rhofinal, RHO | 缺失时可用 Brocher 从 Vp 推算 |

### 2.4 我们的 NetCDF 与 EMC 的差异

| 项目 | 我们的输出 | EMC 期望 | 处理方式 |
|------|------------|----------|----------|
| 变量 | vpv, vph, vsv, vsh, rho | vp, vs, rho | 需从各向异性转 Voigt 平均 |
| 维度 | longitude, latitude, depth | 支持 lat/lon/depth | ✅ 兼容 |
| 深度单位 | km | km | ✅ 兼容 |
| 密度单位 | kg/m³ | g/cm³ 或 kg/m³ | ✅ 兼容 |

**Voigt 平均公式**（各向异性 → 等效各向同性）：
- `vp = sqrt((2*vpv² + vph²) / 3)`
- `vs = sqrt((2*vsv² + vsh²) / 3)`

---

## 三、模型转换流程

### 3.1 步骤 1：转换为 EMC 兼容 NetCDF

```bash
# 转换单个模型（如 FWEA23），输出到 processed/
python 3_Data_space_simulation/convert_to_emc_nc.py --model 2024_FWEA23

# 转换所有模型（每个会自动复制到 specfem3d_globe）
python 3_Data_space_simulation/convert_to_emc_nc.py
```

输出：`data/models/processed/{model}_emc.nc`，并自动复制到 `specfem3d_globe/DATA/IRIS_EMC/model.nc`

### 3.2 步骤 2：准备 FWEA23 示例

项目已创建 `specfem3d_globe/EXAMPLES/FWEA23/` 示例目录，包含：
- `DATA/Par_file`：东亚区域 (70–150°E, 0–54°N, 0–1000 km)
- `DATA/CMTSOLUTION`、`DATA/STATIONS`：测试事件和台站
- 模型需先运行 `convert_to_emc_nc.py --model 2024_FWEA23`，会自动复制到 `DATA/IRIS_EMC/model.nc`
- `run_this_example.sh`：编译 + 提交 Slurm 作业
- `go_mesher_solver_slurm.bash`：Slurm 作业脚本

### 3.3 前置：速度模型标准化

```bash
python 1_Data_preparation/1_5_Process_velocity_models.py  # 需指定 model_names=['2024_FWEA23']
python 3_Data_space_simulation/convert_to_emc_nc.py --model 2024_FWEA23
```

---

## 四、FWEA23 示例运行（曙光 Slurm）

本示例**仅支持曙光 Slurm 超算运行**，无本地运行脚本。

### 4.1 运行步骤

```bash
cd specfem3d_globe/EXAMPLES/FWEA23
./run_this_example.sh
```

脚本会：
1. 确保已运行 `convert_to_emc_nc.py --model 2024_FWEA23`，模型会自动复制到 `DATA/IRIS_EMC/model.nc`
2. 设置目录结构
3. 加载 Intel 编译环境并编译 SPECFEM3D（带 NetCDF）
4. 提交 `sbatch go_mesher_solver_slurm.bash`

### 4.2 Slurm 参数可调

在 `go_mesher_solver_slurm.bash` 中可修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--partition` | xahcnormal | 队列名 |
| `--account` | lyuyj | 账户 |
| `--nodes` | 4 | 节点数 |
| `--ntasks-per-node` | 36 | 每节点核数 |
| `--time` | 05:00:00 | 运行时间 |

### 4.3 监控作业

```bash
squeue -u $USER
tail -f OUTPUT_FILES/job_<jobid>.out
scancel <jobid>
```

---

## 五、Par_file 东亚区域参考

```fortran
NCHUNKS                         = 1
ANGULAR_WIDTH_XI_IN_DEGREES     = 80.d0   # 经度跨度 ~70–150°E
ANGULAR_WIDTH_ETA_IN_DEGREES    = 54.d0   # 纬度跨度 0–54°N
CENTER_LATITUDE_IN_DEGREES      = 27.0d0
CENTER_LONGITUDE_IN_DEGREES     = 110.0d0
GAMMA_ROTATION_AZIMUTH          = 0.d0

# EMC 模型与区域截断
MODEL                           = EMC_model
REGIONAL_MESH_CUTOFF            = .true.
REGIONAL_MESH_CUTOFF_DEPTH      = 1000.d0
USE_LOCAL_MESH                  = .true.
NUMBER_OF_LAYERS_CRUST          = 4
NUMBER_OF_LAYERS_MANTLE         = 8
```

---

## 六、3_1_Setup_specfem3d_globe.py 模块

`3_Data_space_simulation/3_1_Setup_specfem3d_globe.py` 用于生成正演目录和参数：

- 默认 `MODEL = 'EMC_model'`，支持 EMC 区域截断
- 生成 Par_file、STATIONS、CMTSOLUTION
- 创建 Slurm 作业脚本（xmeshfem3D + xspecfem3D，无 xgenerate_databases）
- 使用 `config/base_config.py` 中的 `dirs['specfem3d_globe']`、`dirs['simulations']`

---

## 七、参考链接

- [SPECFEM3D Globe GitHub](https://github.com/SPECFEM/specfem3d_globe)
- [Getting Started Wiki](https://github.com/SPECFEM/specfem3d_globe/wiki/02_getting_started)
- [SPECFEM3D EMC 示例](https://github.com/SPECFEM/specfem3d_globe/tree/master/EXAMPLES/regional_EMC_model)
- [IRIS EMC 数据格式](https://github.com/SPECFEM/specfem3d_globe/blob/master/DATA/IRIS_EMC/README.md)
- [Changing the Model 文档](https://specfem3d-globe.readthedocs.io/en/latest/12_changing_the_model/)

---

## 八、常见问题

### Q1: 编译报错 `-warn all,noexternal` 不兼容

在 Makefile 中替换：
```bash
sed -i.bak 's/-warn all,noexternal/-warn all/g' Makefile
```

### Q2: EMC 模型未找到

运行 `python 3_Data_space_simulation/convert_to_emc_nc.py --model 2024_FWEA23`，转换后会自动复制到 `specfem3d_globe/DATA/IRIS_EMC/model.nc`，再上传到服务器。

### Q3: 需要各向异性 (vpv, vph, vsv, vsh)

当前 EMC 接口仅支持各向同性 vp/vs。需修改 specfem3d_globe 源码以支持各向异性变量。

### Q4: 本地测试

若需本地测试，可手动编译并运行：
```bash
cd specfem3d_globe/EXAMPLES/FWEA23
./setup_model_link.sh
# 手动 mpirun -np N ./bin/xmeshfem3D 及 xspecfem3D
```

---

*更新日期：2026 年 3 月*
