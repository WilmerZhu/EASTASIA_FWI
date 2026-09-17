---
description: "Use when writing or modifying plotting/visualization code in 5_Visualization (PyGMT maps, Cartopy slices, Matplotlib statistics), figure naming, colormaps, dual jpg+pdf export at 300 dpi, English labels."
applyTo: "5_Visualization/**"
---
# 可视化规范

## 工具选择

| 场景 | 工具 |
|---|---|
| 地图（台站、事件、地质、速度水平切片） | PyGMT（首选） |
| 垂直剖面 / 切片带海岸线 | Matplotlib + Cartopy |
| 统计图（直方图、热力图、时间线） | Matplotlib / Seaborn |

## 输出

- 文件名：`模块号-子序号_描述`，如 `5-8_vs_slice_100km`；同一图同时保存 `.jpg` 与 `.pdf`，均 300 dpi。
- 输出目录由 `BaseConfig.dirs['figures']` 派生，子目录按模块/方案分层（如 `figures/model_clustering/{scheme}/`）。
- 保存前 `output_dir.mkdir(parents=True, exist_ok=True)`；保存后 `plt.close(fig)` 释放内存。

## 文字语言

- 标题、坐标轴、色标、图例、注释一律英文；代码注释与日志中文。
- 单位写在标签中：`Vs (km/s)`、`Depth (km)`、`Longitude (°E)`。

## 配色

- 绝对速度：`jet_r` 或 `seismic`；扰动 / 差值 / 残差：`RdBu_r`（以 0 为中心，对称范围）。
- 台网：`tab20` + `tab20b`；聚类 / 相：离散配色（`tab10` 或自定义列表），色序与簇编号一致。
- 相似性指数（SSIM/CW-SSIM）：`viridis` 或 `plasma`，范围固定 0–1 便于跨图对比。
- 色标范围来自数据分位数（如 2%–98%）或 Config 固定值，不要用 min/max。

## 区域与投影

- 地图范围取 `BaseConfig.get_region_bounds()`，不要硬编码经纬度。
- PyGMT 区域投影常用 `M`（Mercator）或 `Q`（等距圆柱）；全域图用 `M15c` 级别宽度。
- 海岸线、国界用 PyGMT `coast(shorelines=..., borders=...)`；地质/构造线数据在 `5_Visualization/data_EastAsia/`。

## 结构

- 每个图形函数只负责一类图，输入为已计算好的数组/DataFrame，不在绑图函数内做数据加载或重算。
- 公共绘图工具放 `5_Visualization/utils/`，模块内不重复实现色标/投影辅助函数。
- 多子图对比时统一色标与范围，避免视觉误导。
