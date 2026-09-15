# SPECFEM 多模型正演

东亚三维速度模型 (FWEA23 / EARA2024 / SinoScope) 的正演模拟与波形检验。

---

## 1. 首次使用：准备工作

### 1.1 上传到服务器

将整个 `specfem/` 目录上传到曙光服务器：

```bash
# 本地执行
rsync -avz --progress specfem/ user@sugon:/work/home/.../specfem/
```

确认关键文件到位：

```bash
# 服务器上检查
ls specfem/models/                        # 3 个模型文件
ls specfem/convert_nc_to_gll.py           # 转换脚本
ls specfem/specfem3d_globe_code_new/src/  # SPECFEM 源码
ls specfem/sem_config_tibet_v4/setup/     # constants.h.in
ls specfem/FWEA23/DATA/Par_file           # Par_file
ls specfem/FWEA23/DATA/STATIONS           # 台站列表
ls specfem/FWEA23/DATA/CMTSOLUTION        # 震源
```

### 1.2 确认 conda 环境

```bash
module load anaconda3/2023.09
source activate eastasia_fwi
python -c "import xarray, scipy, h5py; print('OK')"
```

---

## 2. 运行正演

所有命令在 `specfem/` 目录下执行：

```bash
cd /path/to/specfem/
```

### 首次运行（三模型并行，推荐）

```bash
# 首先确保 logs 目录存在（SLURM 输出需要）
mkdir -p logs

# 一次性粘贴这 4 行，全自动
JOB0=$(sbatch --parsable --export=ALL,MESH_ONLY=1 run_forward.bash)
echo "网格作业: $JOB0"
sbatch --dependency=afterok:$JOB0 --export=ALL,SKIP_MESH=1 run_forward.bash FWEA23
sbatch --dependency=afterok:$JOB0 --export=ALL,SKIP_MESH=1 run_forward.bash EARA2024
sbatch --dependency=afterok:$JOB0 --export=ALL,SKIP_MESH=1 run_forward.bash SinoScope
```

执行流程：

1. `JOB0`：编译 + meshfem(s362ani)，生成共享网格（~40 min），完成后自动退出
2. JOB0 成功后，三个模型作业**同时启动**，各用 4 节点 144 核（~3 h）
3. 总墙钟时间：**约 3.5 小时**

### 只跑单个模型

```bash
# 首次（含编译 + 网格 + 正演）
sbatch run_forward.bash FWEA23

# 已有网格，只跑正演
SKIP_MESH=1 sbatch run_forward.bash EARA2024
```

### 顺序跑全部（一个作业）

```bash
sbatch run_forward.bash
```

一个 SLURM 作业内顺序跑三个模型。耗时：~10 小时。

---

## 3. 监控和查看结果

### 作业状态

```bash
# 查看所有作业（PD=排队等待, R=运行中, CD=完成）
squeue -u $(whoami)

# 典型输出:
#  JOBID  NAME        STATE    TIME    DEPENDENCY
#  10001  FWD-multi   RUNNING  00:15   (网格生成中)
#  10002  FWD-multi   PENDING  00:00   afterok:10001  (等待网格)
#  10003  FWD-multi   PENDING  00:00   afterok:10001
#  10004  FWD-multi   PENDING  00:00   afterok:10001
```

### 实时跟踪

```bash
# 网格生成阶段（JOB0）
tail -f logs/forward_10001.out

# 各模型正演阶段（JOB0 完成后自动开始）
tail -f FWEA23/forward.log
tail -f EARA2024/forward.log
tail -f SinoScope/forward.log
```

### 快速检查是否完成

```bash
# 一行看全部状态
for m in FWEA23 EARA2024 SinoScope; do
    n=$(ls $m/OUTPUT_FILES/*.sac 2>/dev/null | wc -l)
    echo "$m: $n SAC files"
done
```

### 下载结果到本地

```bash
# 本地执行
for m in FWEA23 EARA2024 SinoScope; do
    rsync -avz user@sugon:/path/to/specfem/$m/OUTPUT_FILES/*.sac \
        specfem/$m/OUTPUT_FILES/
done
```

---

## 4. 目录结构

运行后的目录结构：

```
specfem/
├── run_forward.bash               # 主脚本（在此目录 sbatch）
├── convert_nc_to_gll.py           # NC/HDF5 → GLL 转换
├── logs/                          # SLURM 作业日志
├── DATA/                          # 全局默认配置（模板）
│   ├── Par_file                   #   新模型自动从这里复制
│   ├── STATIONS                   #   已有 DATA/ 的模型不会被覆盖
│   └── CMTSOLUTION                #   允许各模型自定义参数
├── models/                        # 原始模型文件（不修改）
│   ├── FWEA23.r0.0-n4.nc
│   ├── EARA2024.r0.0-n4.nc
│   └── FWI_SinoScope_1.0_JMa_ET_AL_2022.h5
│
├── FWEA23/                        # 模板 + FWEA23 工作目录
│   ├── DATA/                      #   首次从 specfem/DATA/ 复制；可自定义
│   │   ├── Par_file, STATIONS, CMTSOLUTION
│   │   └── GLL/                   #   FWEA23 的 GLL 二进制（自动生成）
│   ├── DATABASES_MPI/             # 网格数据库（Phase 2 生成）
│   ├── bin/                       # xmeshfem3D, xspecfem3D
│   ├── OUTPUT_FILES/              # 正演波形 (SAC)
│   └── forward.log
│
├── EARA2024/                      # 自动创建，独立工作目录
│   ├── DATA/                      #   无→从 specfem/DATA/ 复制；有→保留不动
│   │   └── GLL/                   #   EARA2024 的 GLL
│   ├── DATABASES_MPI/             #   网格拓扑 symlink, solver_data 独立
│   ├── bin/                       #   symlink → FWEA23/bin/
│   └── OUTPUT_FILES/
│
└── SinoScope/                     # 同上
```

> **关键规则**: 模型目录如果**已有** `DATA/Par_file`，脚本不会覆盖（允许自定义参数）。
> 如果**没有** `DATA/`，自动从 `specfem/DATA/` 复制默认配置。

---

## 5. 添加新模型

在 `run_forward.bash` 头部的模型注册表中添加：

```bash
MODEL_FILE[NewModel]="models/NewModel.nc"
MODEL_TYPE[NewModel]="nc"          # nc / h5 / csv
MODEL_EXTRA[NewModel]="--anisotropic --horizontal-fill blend"

ALL_MODELS=(FWEA23 EARA2024 SinoScope NewModel)
```

将模型文件放入 `models/`，然后：

```bash
sbatch run_forward.bash NewModel
```

---

## 6. 常见问题

| 问题                                 | 解决                                                  |
| ------------------------------------ | ----------------------------------------------------- |
| `模型文件不存在`                     | 检查 `models/` 下文件名是否与注册表一致               |
| `bin/ 中无可执行文件`                | 去掉 `SKIP_COMPILE` 重新跑，或从已有目录拷贝 bin/     |
| `DATABASES_MPI 中无 solver_data.bin` | 去掉 `SKIP_MESH` 重新跑 Phase 1+2                     |
| Phase 3 转换失败                     | 检查 `forward.log`，确认 conda 环境和依赖             |
| Phase 4/5 MPI 失败                   | 检查节点数 × ntasks-per-node >= 144 (NPROC)           |
| 波形全零                             | 检查 CMTSOLUTION 格式（需 ECEF，参见 constants.h.in） |

---

## 7. 参考

- [Models_GLL_workflow_server.md](Models_GLL_workflow_server.md) — 完整技术文档、SPECFEM 数据流、排错指南
- FWEA23: [Liu et al., 2024, EPSL](https://doi.org/10.1016/j.epsl.2024.118764)
- EARA2024: [EMC EARA2024](https://ds.iris.edu/ds/products/emc-earthmodels/)
- SinoScope 1.0: Ma et al., 2022
