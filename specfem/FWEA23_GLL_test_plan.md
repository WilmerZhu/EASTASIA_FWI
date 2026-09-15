# FWEA23 NC→GLL 测试方案

> **范围**：只验证 FWEA23。EARA2024 / SinoScope 等全部延期。  
> **SPECFEM**：v7.0.0（`specfem3d_globe_code_new/` + tibet_v4）  
> **详细排障史**：见 `Models_GLL_workflow_server.md`（归档参考，勿按其中多模型步骤开跑）  
> **更新日期**：2026-08-01

---

## 1. 目标

证明：从公开 NC（`FWEA23.r0.0-n4.nc`）转到 SPECFEM GLL 后，

1. **场量**接近原作者原生 GLL（`model_updated/`）  
2. **正演**稳定，且插值误差可与「作者 GLL 直跑」隔离量化  

未完成下列 T0–T3 前，不扩展其他模型。

---

## 2. 原理（必须理解）

```text
Phase 2  mesher(MODEL=s362ani)
         → solver_data.bin = 网格坐标 + s362ani+CRUST1.0（含精确间断面）

Phase 3  convert_nc_to_gll.py --use-perturbation
         δV    = interp(V_nc − V_ref_nc)     # V_ref_nc = vp0/vs0
         V_gll = V_ref_gll + δV              # V_ref_gll 从 solver_data 反算
         域外 δV → 0 → 保持 V_ref_gll

Phase 4  mesher(MODEL=GLL) → 用新 GLL 重写 solver_data.bin
Phase 5  solver → SAC
```

**硬约束**：Phase 3 读的 `solver_data.bin` 必须来自 **刚完成的 Phase 2（s362ani）**。  
若文件已被 Phase 4（GLL）覆盖，则 `V_ref_gll` 已是终模型，扰动会双重叠加 → 结果错误。

自检（T0 后、T1 前）：

```bash
# solver_data 反算速度 与 model_updated 不应几乎相同（corr 应明显 < 1）
# 若 corr ≈ 1.0 → 网格已污染，必须重跑 MESH_ONLY=1
```

---

## 3. 目录与文件

```text
specfem/                                 # 服务器建议: /work/home/acf11bgjob/specfem/
├── run_forward.bash
├── convert_nc_to_gll.py                 # 须为含 V_ref_gll+δV 的最新版
├── compare_gll.py                       # 默认对比 model_updated/ vs GLL/
├── models/FWEA23.r0.0-n4.nc
├── model_updated/                       # 作者原生 GLL（场量金标准）
├── sac/                                 # 作者波形（定性参考）
├── GLL/                                 # compare_gll.py 读这里 → 用软链指向 FWEA23/DATA/GLL
├── specfem3d_globe_code_new/            # v7.0.0
├── sem_config_tibet_v4/
└── FWEA23/
    ├── DATA/  DATABASES_MPI/  bin/  OUTPUT_FILES/
    └── DATA/GLL/                        # Phase 3 输出
```

环境：`conda activate eastasia_fwi`（需 netCDF4 / xarray / numpy / scipy）

---

## 4. 验收门禁（按序，不可跳）

| 门禁 | 做什么 | 通过标准 |
|------|--------|----------|
| **T0** | 只跑编译 + mesh(s362ani) | exit 0；NSPEC=7236；CFL&lt;1；`MODEL=s362ani`；`solver_data` **未被** GLL 污染 |
| **T1** | NC→GLL + 与 `model_updated` 比 | 日志含 `V_gll = V_ref_gll + δV`；域内 proc mean\|δV\| 不≈0；分深度 RMSE 合理 |
| **T2** | 作者 GLL → mesh(GLL)+solver | Max U 稳定（隔离版本/补丁） |
| **T3** | 我们的 GLL → 同配置 solver | 与 **T2** 波形比；再与 `sac/` 定性对照 |

未过 T1 → 禁止 T2/T3。未过 T3 → 禁止测其他模型。

**对比注意**：

- 勿只用 **proc0**：该 slice 常整片在 NC 域外（δV≈0），RMSE 会虚低。优先看域内 proc（如 57）或 `compare_gll.py` 全域统计。  
- 作者 `sac/` 不是唯一金标准；主对比是 **T3 vs T2**（同一套 SPECFEM）。

---

## 5. 操作步骤（服务器）

### 5.0 上传最新脚本

```bash
# 本机
scp specfem/convert_nc_to_gll.py server:/work/home/acf11bgjob/specfem/
scp specfem/FWEA23_GLL_test_plan.md server:/work/home/acf11bgjob/specfem/
```

### 5.1 T0 — 干净 s362ani 网格

```bash
cd /work/home/acf11bgjob/specfem
conda activate eastasia_fwi
mkdir -p logs

JOB0=$(sbatch --parsable --export=ALL,MESH_ONLY=1 run_forward.bash)
# 等待成功：日志含 "MESH_ONLY=1: Phase 1+2 完成"
# 检查:
#   FWEA23/DATABASES_MPI/ 有 144 个 proc*_reg1_solver_data.bin
#   OUTPUT_FILES/output_mesher.txt: NSPEC=7236, CFL<1
```

### 5.2 T1 — NC→GLL + 场量对比

```bash
cd /work/home/acf11bgjob/specfem/FWEA23

python ../convert_nc_to_gll.py \
  --mesh-dir DATABASES_MPI \
  --model-nc ../models/FWEA23.r0.0-n4.nc \
  --output-dir DATA/GLL \
  --use-perturbation \
  --reference-gll-dir ../model_updated

# 日志必须出现:
#   ★ 正确公式: V_gll = V_ref_gll(mesher) + interp(δV_nc)
#   [PERT] ... mean|δV|=...

# 供 compare_gll.py 使用（其默认读 specfem/GLL/）
cd ..
ln -sfn FWEA23/DATA/GLL GLL
python compare_gll.py --mesh-dir FWEA23/DATABASES_MPI
# 图输出: figures/gll_compare/
```

**T1 通过参考（经验目标，以实际图为准）**：

| 深度段 | 关注 |
|--------|------|
| 地壳 / Moho | 相对旧绝对值插值应明显改善 |
| 410 / 660 | 间断面附近误差应下降 |
| 值域 | 无零速度；rho 为 g/cm³ 量级 |

### 5.3 T2 — 作者 GLL 正演（强烈建议）

```bash
cd /work/home/acf11bgjob/specfem/FWEA23
rm -rf DATA/GLL
mkdir -p DATA/GLL
cp ../model_updated/proc*_reg1_*.bin DATA/GLL/

# 仅 mesh(GLL)+solver：需 DATABASES_MPI 拓扑仍在；
# 若 run_forward 会先覆盖 GLL，可临时改脚本或手工:
sed -i '/^MODEL/s/= .*/= GLL/' DATA/Par_file
rm -f DATABASES_MPI/proc*_reg1_solver_data.bin   # Phase4 会重写；保留其他网格文件
mpirun -np 144 ./bin/xmeshfem3D
# 确认 output_mesher.txt: reading in model from: DATA/GLL/
mpirun -np 144 ./bin/xspecfem3D
# 检查 Max U 稳定；保存 OUTPUT_FILES 为 REF_T2/
```

> 若手工 MPI 不便，也可写一临时脚本只跑 Phase 4+5；原则是 **GLL 内容 = model_updated**。

### 5.4 T3 — 我们的 GLL 正演

```bash
cd /work/home/acf11bgjob/specfem

# 用 T1 的 GLL；网格已存在则跳过 Phase1+2
# 注意: run_forward Phase3 会重新转换并覆盖 DATA/GLL
SKIP_MESH=1 SKIP_COMPILE=1 sbatch --export=ALL,SKIP_MESH=1,SKIP_COMPILE=1 \
  run_forward.bash FWEA23
```

对比：

1. T3 波形 vs T2 波形 → **NC→GLL 误差**  
2. T3 波形 vs `sac/` → 定性（含版本/事件等混杂因素）

---

## 6. 一键脚本与本方案的关系

| 命令 | 用途 |
|------|------|
| `MESH_ONLY=1 sbatch run_forward.bash` | = T0 |
| `python convert_nc_to_gll.py --use-perturbation ...` | = T1 转换 |
| `SKIP_MESH=1 sbatch run_forward.bash FWEA23` | = 自动 Phase3–5（含再转换）；适合 T1 场量已过关后的 T3 |
| `sbatch run_forward.bash`（无参数） | 会跑多模型 → **本阶段禁止** |

---

## 7. 配置要点

```fortran
! constants.h.in (tibet_v4)
EARTH_REGIONAL_MOHO_MESH       = .true.
EARTH_HONOR_DEEP_MOHO          = .true.
EARTH_RMOHO_STRETCH_ADJUSTMENT = -20000.d0
```

```text
Par_file:  Phase2 → MODEL = s362ani
           Phase4 → MODEL = GLL          # v7；不要写 gll_qmu
NPROC = 144 (12×12), NEX = 288
MPI: mpirun -np $NPROC（v7 路径已验证）
```

---

## 8. 作业记录模板

```
日期: ________  作业号: ________  门禁: T0 / T1 / T2 / T3
NSPEC=___  DT=___  CFL=___
solver_data 干净(≠ model_updated): 是 / 否
convert 日志含 V_ref_gll+δV: 是 / 否
域内 mean|δV|: ___
compare_gll 摘要: crust___  410___  660___
solver Max U 稳定: 是 / 否
T3 vs T2 结论: ________
通过 / 失败 — 原因: ________
```

---

## 9. 禁止事项

| 不要做 | 原因 |
|--------|------|
| `SKIP_MESH=1` 却未确认 DATABASES_MPI 来自最新 s362ani | V_ref 污染 |
| 只拿 proc0 验收 T1 | 常在 NC 域外 |
| T1 未过就对作者 sac 下结论 | 无法归因 |
| 同时跑 EARA / SinoScope | 偏离本阶段目标 |
| 把 MPI fatal→warning 当「无问题」 | 可能静默错误（见总文档 §19） |

---

## 10. 通过后的下一步（延期）

FWEA23 T0–T3 全部通过后，再打开：

1. EARA2024（`--use-perturbation`，逐分量 `*_ref`）  
2. SinoScope（无参考场，绝对值 + 域外 3D）  
3. 三模型公平对比（同 chunk / 同事件 / 同台站）

---

_本文件为 FWEA23 测试的唯一操作入口；总流程长文档仅作背景与排障归档。_
