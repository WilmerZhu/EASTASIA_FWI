# FWEA23 与 SinoScope1.0 融合：实践操作指南

## 快速开始

### 假设你已有：
- `SinoScope1.0.nc` - 1°×1°×20km，全东亚覆盖
- `FWEA23.nc` - 0.25°×0.25°×10km，部分覆盖

### 三种运行模式

#### 🚀 模式 A：快速测试（推荐入门）

```bash
cd /Users/wilmer/Github/EASTASIA_FWI/4_Fusion

# 运行融合（使用代码中硬编码的参数）
python 4_3_SDL_Fusion_v4.py

# 输出结果：
# ├─ results/ 
# │  └─ 2026-06-11_14-30-00/
# │     ├─ Phase1_2D_transform/ (深度切片可视化)
# │     ├─ Phase1_5_full_slice/ (完整融合结果)
# │     └─ fusion_enhanced_SinoScope.nc (最终模型)
# └─ logs/
#    └─ fusion.log (详细日志)
```

**输出解读**：
- `improved_metrics.json` - 改善率、误差等指标
- 可视化图表 - F01-F06 组 (Phase 1.5 结果)

#### 🔬 模式 B：参数优化（最优结果）

```bash
# 执行 K × λ 联合网格搜索
python 4_3_SDL_Fusion_v4.py --grid-search

# 搜索空间：
# K ∈ [10, 15, 20, 25, 30] 
# λ ∈ [0.01, 0.05, 0.1, 0.15, 0.2]
# 总共 25 种组合

# 输出：
# ├─ search_results.csv (所有组合的评估结果)
# ├─ optimal_config.json (最优参数)
# └─ results/ (最优配置的融合结果)
```

**预期耗时**：
- 单组合耗时：~100 秒/深度
- 51 个深度：~1.4 小时
- 25 个参数组合：~35 小时（可用 GPU 加速）

#### ⚡ 模式 C：生产运行（重复使用最优参数）

```bash
# 假设已从 --grid-search 得到 optimal_config.json

python 4_3_SDL_Fusion_v4.py --use-optimal

# 只运行 1 个参数组合，~1.4 小时完成
```

---

## 关键代码节点解析

### 1. 如何指定输入模型路径

```python
# 编辑 config/base_config.py

BASE_CONFIG = {
    'models': {
        'sinoscope': '/path/to/SinoScope1.0.nc',
        'eara2024': '/path/to/EARA2024.nc',
        'fwea23': '/path/to/FWEA23.nc',
    }
}
```

或使用 CLI 覆盖参数：

```bash
python 4_3_SDL_Fusion_v4.py \
    --base-model /path/to/SinoScope1.0.nc \
    --target-model /path/to/FWEA23.nc
```

### 2. 调整深度匹配容差

```python
# 在 4_3_SDL_Fusion_v4.py 中找到：

DEPTH_TOLERANCE = 0.1  # km，默认 ±0.1km

# 如果你的模型深度刻度不完全对齐，增加容差：
DEPTH_TOLERANCE = 1.0  # 允许 ±1km 偏差
```

### 3. 调整 Patch 参数

```python
# 代码中的配置类

phase1_config = Phase1Config(
    window_size_degrees=6.0,      # 物理窗口大小（度）
    physical_stride_degrees=1.0,  # 物理步长（度）
    dict_atoms_k=20,              # 字典原子数
    sparsity_lambda=0.1,          # 稀疏性参数
    cc_threshold=0,               # 相关系数筛选阈值
)
```

### 4. 调整 Patch 拼接方法

```python
# Phase 1.5 变换时选择拼接方式

result = transformer.transform_full_slice_direct(
    base_ds=base_ds,
    highres_ds=highres_ds,
    param='vs',
    depth_km=100,
    coupled_dict=coupled_dict,
    stitch_method='gaussian',  # 或 'average'
)
```

对比：
| 方法 | 特点 | 结果 |
|------|------|------|
| `gaussian` | 自然过渡，中心权重高 | 更平滑，推荐 |
| `average` | 简单平均 | 可能有界过渡 |

### 5. 监控训练进度

```bash
# 查看实时日志
tail -f logs/fusion.log | grep -E "Phase|depth|error"

# 关键输出：
# [Phase 1 @ depth=20km] Extracted 9823 patch pairs
# [Phase 2 @ depth=20km] Training dict: D1=(20,36), D2=(20,576)
# [Phase 2 @ depth=20km] Validation error: 8.7%
# [Phase 1.5 @ depth=20km] Transform complete: improvement=31.2%
```

---

## 详细工作流：从零开始

### 步骤 1：准备数据

```python
import xarray as xr
import numpy as np

# 加载模型
base = xr.open_dataset('SinoScope1.0.nc')
target = xr.open_dataset('FWEA23.nc')

# 检查坐标系
print("SinoScope 坐标:")
print(f"  lat: {base.lat.min():.2f}° ~ {base.lat.max():.2f}°")
print(f"  lon: {base.lon.min():.2f}° ~ {base.lon.max():.2f}°")
print(f"  depth: {base.depth.values}")

print("\nFWEA23 坐标:")
print(f"  lat: {target.lat.min():.2f}° ~ {target.lat.max():.2f}°")
print(f"  lon: {target.lon.min():.2f}° ~ {target.lon.max():.2f}°")
print(f"  depth: {target.depth.values}")

# 检查变量名
print(f"\nSinoScope 变量: {list(base.data_vars)}")
print(f"FWEA23 变量: {list(target.data_vars)}")
```

### 步骤 2：执行融合

```bash
# 首先测试一个单独的深度（快速验证）
python 4_3_SDL_Fusion_v4.py --test-depth 20

# 如果成功，运行完整融合（所有深度）
python 4_3_SDL_Fusion_v4.py
```

### 步骤 3：评估结果

```python
import json
import matplotlib.pyplot as plt

# 读取评估指标
with open('results/improved_metrics.json') as f:
    metrics = json.load(f)

print(f"平均改善率: {metrics['avg_improvement']:.1%}")
print(f"验证误差: {metrics['validation_error']:.2%}")

# 可视化误差改善
depths = list(metrics['by_depth'].keys())
improvements = [metrics['by_depth'][d]['improvement'] for d in depths]

plt.figure(figsize=(12, 6))
plt.plot(depths, improvements, 'o-', linewidth=2)
plt.xlabel('Depth (km)')
plt.ylabel('Improvement Rate (%)')
plt.title('SDL Fusion: Depth-wise Improvement')
plt.grid(True, alpha=0.3)
plt.savefig('improvement_by_depth.png', dpi=150, bbox_inches='tight')
```

### 步骤 4：生成最终产品

```bash
# 融合结果已包含在：
# results/<timestamp>/fusion_enhanced_SinoScope.nc

# 可视化增强效果
python 5_Visualization/5_1_Basemap.py \
    --model fusion_enhanced_SinoScope.nc \
    --depth 100 \
    --param vs
```

---

## 故障排除

### 问题 1：深度不匹配

**症状**：
```
ERROR: No matched depths found!
Tolerance: 0.1 km
```

**解决**：
```python
# 检查深度精度
import xarray as xr

base = xr.open_dataset('SinoScope1.0.nc')
target = xr.open_dataset('FWEA23.nc')

print("SinoScope depths:", base.depth.values)
print("FWEA23 depths:", target.depth.values)

# 如果只有整数深度，增加容差
DEPTH_TOLERANCE = 1.0  # 从 0.1 改为 1.0
```

### 问题 2：Patch 数过少

**症状**：
```
WARNING: Only 100 patch pairs extracted (expected ~8000)
```

**原因**：
- FWEA23 覆盖范围太小
- CC 阈值设置过高
- 模型坐标系不对齐

**解决**：
```python
# 降低 CC 阈值
cc_threshold = 0.1  # 从 0 改为 0.1，包括更多 patch

# 或检查坐标对齐
# 确保 FWEA23 坐标确实在 SinoScope 范围内
```

### 问题 3：训练误差过高 (> 15%)

**症状**：
```
Validation error: 18.3% (expected < 10%)
```

**原因**：
- K 值过小，表现力不足
- λ 过大，过度稀疏
- 数据质量问题

**解决**：
```bash
# 运行网格搜索找最优参数
python 4_3_SDL_Fusion_v4.py --grid-search

# 或手动调试
python 4_3_SDL_Fusion_v4.py --k 25 --lambda 0.05
```

### 问题 4：输出结果有假干扰

**症状**：
- 边界出现明显的网格状干扰
- 某些深度有异常峰值

**原因**：
- Patch 权重设置不当
- 某个深度的数据有异常

**解决**：
```python
# 使用 Feather 权重而非 Gaussian
result = transformer.transform_full_slice_direct(
    ...,
    stitch_method='feather'  # 改为 feather
)

# 或检查该深度的输入数据
plt.figure()
plt.imshow(base_slice_problematic)
plt.colorbar()
plt.title('Check anomalies')
plt.show()
```

---

## 性能优化

### 1. 并行处理多个深度

```bash
# 使用 GNU parallel 或 xargs
cat depths.txt | parallel -j 4 "python 4_3_SDL_Fusion_v4.py --depth {}"
```

### 2. GPU 加速（如果可用）

```python
# 在 DictionaryLearning 中使用 GPU
# 当前实现基于 sklearn，不支持 GPU
# 可选：用 PyTorch 重新实现以支持 GPU

from torch.nn.functional import cosine_similarity
# 但 sklearn DictionaryLearning 目前无 GPU 版本
```

### 3. 内存优化

```python
# 对于大模型，分块处理
chunk_size = 100  # 每次处理 100 个 patch

for chunk in get_patch_chunks(all_patches, chunk_size):
    # 处理单个 chunk
    pass
```

---

## 验证清单

运行完成后，检查以下要点：

```
☐ 输出文件存在
  - fusion_enhanced_SinoScope.nc (完整模型)
  - F01-F06.png (Phase 1.5 可视化)
  - improved_metrics.json (评估指标)

☐ 数据完整性
  - 输出覆盖全东亚范围
  - 无 NaN 或无限值
  - 物理量单位一致

☐ 质量指标
  - 改善率 > 20%
  - 验证误差 < 12%
  - 误差标准差 < 5%

☐ 可视化检查
  - 无明显的界或跳变
  - 平滑过渡区域
  - 与 FWEA23 在重叠区相关性 > 0.7

☐ 日志无错误
  - 无 Exception 或 Error 消息
  - 所有深度都成功处理
  - 最终输出统计信息完整
```

---

## 常用命令速查

```bash
# 快速测试（单深度）
python 4_3_SDL_Fusion_v4.py --test-depth 100

# 指定参数运行
python 4_3_SDL_Fusion_v4.py --k 20 --lambda 0.1 --stride 1

# 网格搜索
python 4_3_SDL_Fusion_v4.py --grid-search --k-range 15 30 --lambda-range 0.05 0.15

# 使用最优配置
python 4_3_SDL_Fusion_v4.py --use-optimal

# 仅验证模型加载（无融合）
python 4_3_SDL_Fusion_v4.py --check-data

# 生成详细日志
python 4_3_SDL_Fusion_v4.py --loglevel DEBUG

# 指定输出目录
python 4_3_SDL_Fusion_v4.py --output-dir /custom/path/

# 保存所有中间结果（调试用）
python 4_3_SDL_Fusion_v4.py --keep-intermediates
```

---

## 后处理步骤

融合完成后，通常需要：

### 1. 格式转换

```bash
# 从 NetCDF 转换为 GLL 格式（SPECFEM3D Globe）
python 3_Data_space_simulation/convert_nc_to_gll.py \
    --input fusion_enhanced_SinoScope.nc \
    --output fusion_enhanced_SinoScope_GLL/

# 用于 SPECFEM3D 正演
```

### 2. 质量检查

```bash
# 正演合成波形
python specfem/run_forward.bash \
    --model fusion_enhanced_SinoScope_GLL \
    --events catalog.txt

# 对比与 FWEA23、SinoScope 的波形
```

### 3. 论文发表

```bash
# 准备图表
python 5_Visualization/5_1_Basemap.py \
    --compare fusion_enhanced_SinoScope.nc FWEA23.nc SinoScope1.0.nc \
    --depths 20 50 100 150 200

# 输出高质量 EPS 图
```

---

## 引用参考

如果使用本融合方法，请引用原论文：

> Zhang, H., & Ben-Zion, Y. (2024). Enhancing Regional Seismic Velocity Models 
> With Higher-Resolution Local Results Using Sparse Dictionary Learning. 
> *Journal of Geophysical Research: Solid Earth*, 129, e2023JB027016.

以及这个项目的改进版本实现。
