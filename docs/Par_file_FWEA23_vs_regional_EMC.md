# Par_file 差异对比：FWEA23 vs regional_EMC_model

## 一、差异汇总

| 参数 | regional_EMC_model | FWEA23 | 说明 |
|------|-------------------|--------|------|
| **区域参数** | | | |
| ANGULAR_WIDTH_XI | 30.d0 | 80.d0 | 经度跨度，FWEA23 模型 70–150°E |
| ANGULAR_WIDTH_ETA | 25.d0 | 54.d0 | 纬度跨度，FWEA23 模型 0–54°N |
| CENTER_LATITUDE | 64.d0 | 27.0d0 | 中心纬度 |
| CENTER_LONGITUDE | -150.d0 | 110.0d0 | 中心经度 |
| GAMMA_ROTATION_AZIMUTH | 20.d0 | 60.d0 | 网格旋转角 |
| **网格/MPI** | | | |
| NEX_XI | 32 | 288 | 表面单元数 |
| NEX_ETA | 32 | 288 | 表面单元数 |
| NPROC_XI | 2 | 12 | MPI 进程数 |
| NPROC_ETA | 2 | 12 | MPI 进程数 |
| **模型/时间** | | | |
| RECORD_LENGTH_IN_MINUTES | 2.5d0 | 5.0d0 | 记录时长 |
| REGIONAL_MESH_CUTOFF_DEPTH | 200.d0 | 1000.d0 | 区域截断深度，FWEA23 深度 0–1000 km |

## 二、相同的 EMC 相关参数

- `MODEL = EMC_model`
- `REGIONAL_MESH_CUTOFF = .true.`
- `USE_LOCAL_MESH = .true.`
- `NUMBER_OF_LAYERS_CRUST = 4`
- `NUMBER_OF_LAYERS_MANTLE = 8`
- `NDOUBLINGS = 2`, `NZ_DOUBLING_1/2` 等
- `DT = 0.05`
- 其余物理、输出、衰减等参数一致

## 三、处理建议

### 1. 保持当前差异（推荐）

- **区域参数**：regional_EMC 为阿拉斯加小区域（30°×25°），FWEA23 为东亚大区域（80°×54°），差异合理。
- **REGIONAL_MESH_CUTOFF_DEPTH**：regional_EMC 为 200 km，FWEA23 为 1000 km，与 FWEA23 模型深度范围一致。
- **NEX/NPROC**：FWEA23 区域更大，需更高分辨率与更多核，当前设置合理。

### 2. 可补充的 Mesh 段注释

regional_EMC 在 Mesh 段有更细的注释，FWEA23 可补充以保持一致：

- `USE_LOCAL_MESH` 前：`# flag to turn on local mesh layout`
- `NUMBER_OF_LAYERS_CRUST` 前：`# total number of mesh layers for local mesh` 及 moho 说明
- `NDOUBLINGS` 前：`# number of doubling layers`
- `NZ_DOUBLING_1` 前：`# position of doubling layer (counted from top down)`
- `DT` 前：`# time step size`

### 3. 需注意的约束

- NEX 需为 16 的倍数，且为 8×NPROC 的倍数：288/16=18 ✓，288/(8×12)=3 ✓
- NEX_XI = NEX_ETA：288 = 288 ✓

---

## 四、Mesh 层数与截断深度的关系（重要）

### 4.1 当前设置

| 参数 | regional_EMC (200 km) | FWEA23 (1000 km) |
|------|----------------------|------------------|
| REGIONAL_MESH_CUTOFF_DEPTH | 200.d0 | 1000.d0 |
| NUMBER_OF_LAYERS_CRUST | 4 | 4 |
| NUMBER_OF_LAYERS_MANTLE | 8 | 8 |
| NDOUBLINGS | 2 | 2 |
| NZ_DOUBLING_1 / NZ_DOUBLING_2 | 2, 5 | 2, 5 |

### 4.2 潜在影响

- **垂向分辨率**：同一套层数（4 地壳 + 8 地幔）在 200 km 与 1000 km 截断下，垂向单元厚度约差 5 倍。
- **粗略估算**：200 km 时每层约 15–25 km；1000 km 时每层约 80–125 km，深部垂向采样变粗。
- **Par_file 注释**：`REGIONAL_MESH_CUTOFF_DEPTH` 的官方可选值为 `24.4, 80, 220, 400, 600, 670, 771` km，**200 与 1000 均不在其中**，说明两者均为非标准深度。

### 4.3 建议

1. **验证当前设置**：FWEA23 示例可能已用该配置做过验证，建议先按现有参数跑通，再评估波形拟合。
2. **若需提高深部分辨率**：可尝试将 `NUMBER_OF_LAYERS_MANTLE` 增至 12 或 16，并相应调整 `NZ_DOUBLING_2` 等 doubling 位置。
3. **查阅源码**：SPECFEM3D Globe 的 `create_regions_mesh_*` 等模块会约束层数与 doubling 的对应关系，修改前需确认是否满足内部约束。
