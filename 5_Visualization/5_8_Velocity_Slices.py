"""
5_8_Velocity_Slices.py:
多模型绝对速度切片对比可视化模块（九宫格）
================================================================

功能描述:
----------
基于 PyGMT 绘制东亚区域多个三维速度模型在同一深度的绝对速度（Vs / Vp）水平切片，
采用 3×3 九宫格布局并排对比，便于直观比较不同模型在同一深度的速度结构差异。
对每个目标深度生成一张九宫格图，各子图共用同一色标（按深度自适应），
缺数据的模型（深度超出其覆盖范围）自动留白并标注。

核心功能:
----------
1. ✅ 多模型同深度切片
   - 默认对比 6 个模型：
     CSRM1.0 / ASIA2024 / USTClitho2.0 / FWEA23 / EARA2024 / SinoScope1.0
   - 读取 data/models/processed/{model}/{model}_original.nc 标准化 NetCDF
   - 每个模型占据九宫格中的一个子图

2. ✅ 绝对速度切片绘制
   - 选取最邻近目标深度的切片 (vs/vp)
   - 自动识别规则/非规则网格，必要时插值到规则网格
   - NaN 区域透明，仅显示模型覆盖范围（参考截图样式）

3. ✅ 每个深度共享色标（手动指定 + 自动备用）
   - 同一深度的所有模型子图使用同一色标范围
   - manual_series 手动指定各深度色标范围（'vmin/vmax/step'）
   - 未手动指定的深度自动用合并数据的 1~99 百分位数备用
   - 图底一条共享色标条

4. ✅ 地理底图
   - 海岸线 + 国界
   - 与 5_1/5_7 一致的 PyGMT 配置风格

输入数据:
----------
- data/models/processed/{model}/{model}_original.nc
  (变量: vs, vp, vsv, vsh, vpv, vph, ...; 维度: longitude, latitude, depth)

输出文件:
----------
- 5-8_velocity_slices_vs_{depth}km.jpg/pdf  (每个深度一张九宫格)

科学原理:
----------
- 同深度绝对速度切片可揭示岩石圈—软流圈结构（如克拉通高速根、俯冲板片、
  造山带低速异常）在不同模型间的一致性与分歧。
- 共享色标确保跨模型对比的公平性；按深度自适应避免浅/深部速度量级差异掩盖结构细节。

作者: EASTASIA-FWI Team
日期: 2026-06-10
版本: v1.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import pygmt

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import BaseConfig  # noqa: E402

# 可视化模块内 utils
sys.path.insert(0, str(Path(__file__).parent))
from utils.file_utils import extract_module_prefix, generate_filename  # noqa: E402


class VelocitySlicesConfig:
    """多模型速度切片可视化配置类"""

    def __init__(self):
        # ============ 模型清单（顺序即九宫格排列顺序） ============
        # 键: 处理后数据目录名; 值: 图中显示名
        # 注: CSRM1.0 分辨率特殊处理，拆为 0.2/0.4 两个独立模型条目
        self.models: Dict[str, str] = {
            '2024_FWEA23': 'FWEA23',
            '2024_EARA2024': 'EARA2024',
            '2022_SinoScope1.0': 'SinoScope1.0',
            '2024_ASIA2024': 'ASIA2024',
            '2023_CSRM1.0_res0.4': 'CSRM1.0-0.4°',
            '2023_CSRM1.0_res0.2': 'CSRM1.0-0.2°',
            '2022_USTClitho2.0': 'USTClitho2.0',
            '2024_CSES_VM1.0': 'CSES_VM1.0',
            '2023_SWChinaCVM-2.0': 'SWChinaCVM-2.0',
            '2022_Chen_SChina': 'Chen_SChina',
            '2026_Li_NEChina': 'Li_NEChina',
            # '2026_Li_SChina': 'Li_SChina',
            '2022_SASSY21': 'SASSY21',

        }

        # ============ 绘图参数 ============
        # 速度参数: 'vs'（绝对 S 波速度，对应截图）或 'vp'
        self.parameter = 'vs'

        # 目标深度列表 (km)，每个深度生成一张九宫格
        self.depths: List[float] = [10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 120.0, 150.0]

        # 深度匹配容差 (km)：模型最大深度 + 容差 < 目标深度则该模型留白
        self.depth_tolerance = 5.0

        # 九宫格布局
        self.grid = {
            'nrows': 3,
            'ncols': 4,
            'panel_width': '5.2c',
            'figsize': ('25c', '20c'),
            'margins': ['0.3c', '0.55c'],
        }

        # 色标
        self.colorbar = {
            'cmap': 'seis',            # 红(低速)→蓝(高速)
            'reverse': False,
            # 未手动指定时的备用分位数
            'percentile': (1.0, 99.0),
            # ======================================================
            # 手动指定各深度色标范围: {depth_km: 'vmin/vmax/step'}
            # 未列入的深度将自动计算。
            # ======================================================
            'manual_series': {
                10.0:  '2.2/4.5/0.10',
                20.0:  '3.1/4.5/0.10',
                30.0:  '3.2/4.6/0.10',
                40.0:  '3.4/4.7/0.10',
                60.0:  '3.6/4.8/0.10',
                80.0:  '3.8/4.9/0.10',
                100.0: '3.9/4.9/0.10',
                120.0: '4.0/4.9/0.10',
                150.0: '4.0/4.9/0.10',
            },
            # 图底共享色标条位置
            'shared_cbar_position': 'JBC+o0c/0.8c+w14c/0.35c+h',
        }

        # 子图坐标轴
        # 注: 在 GMT 6.6 中，轴间隔必须放在边框规范(WSen)之前，
        # +t 标题必须和边框规范合并为同一个 -B 参数。
        self.axes = {
            'xaxis': 'xa20f10',
            'yaxis': 'ya15f5',
        }

        # 区域：None 表示使用 base_config 研究区域；否则 [lon_min, lon_max, lat_min, lat_max]
        self.region_override: Optional[List[float]] = None

        # PyGMT 全局配置（与 5_1/5_7 一致）
        self.pygmt = {
            'MAP_FRAME_TYPE': 'plain',
            'MAP_FRAME_PEN': '0.6p,black',
            'MAP_FRAME_WIDTH': '2p',
            'FONT_ANNOT_PRIMARY': '10p,Helvetica,black',
            'FONT_LABEL': '9p,Helvetica,black',
            'FONT_TITLE': '11p,Helvetica-Bold,black',
            'MAP_TITLE_OFFSET': '4p',
            'MAP_TICK_LENGTH_PRIMARY': '3p',
        }

        # 海岸线 / 国界
        self.coast = {
            'shorelines': '0.25p,gray20',   # 细海岸线
            'borders': ['1/0.2p,gray40'],    # 细国境线
            'resolution': 'c',               # crude 精度，足够显示轮廓
        }

        # 输出
        self.output = {
            'dpi_jpg': 300,
            'dpi_pdf': 300,
            'crop': True,
            'save_jpg': True,
            'save_pdf': False,
        }

        self.logging = {'level': 'INFO'}


class VelocitySlicesPlotter:
    """多模型速度切片九宫格绘图类"""

    def __init__(self,
                 config_file: Optional[Path] = None,
                 output_dir: Optional[str] = None,
                 data_dir: Optional[Path] = None):
        # 1. 全局配置
        self.base_config = BaseConfig(config_file)
        # 2. 模块配置
        self.config = VelocitySlicesConfig()

        # 3. 输出目录（与 5_1/5_7 一致：项目根 figures/）
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.base_config.dirs['figures']
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 4. 日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.VelocitySlicesPlotter',
            self.config.logging['level'],
        )

        # 5. 区域参数
        if self.config.region_override is not None:
            self.region = list(self.config.region_override)
        else:
            r = self.base_config.region
            self.region = [r['lon_min'], r['lon_max'], r['lat_min'], r['lat_max']]

        # 6. 速度模型处理数据目录
        self.data_dir = (Path(data_dir) if data_dir
                         else self.base_config.dirs['models'] / 'processed')

        # 7. PyGMT 配置与模块前缀
        self._setup_pygmt_config()
        self.module_prefix = extract_module_prefix(Path(__file__), default="5-8")

        self.logger.info("🗺️ 多模型速度切片绘图器初始化完成")
        self._print_config_summary()

    # ------------------------------------------------------------------ #
    # 配置
    # ------------------------------------------------------------------ #
    def _setup_pygmt_config(self):
        pygmt.config(**self.config.pygmt)

    def _print_config_summary(self):
        print("\n📋 多模型速度切片可视化配置摘要")
        print("-" * 60)
        print(f"研究区域: {self.base_config.region['name']}")
        print(f"绘图区域: lon {self.region[0]}~{self.region[1]}°, "
              f"lat {self.region[2]}~{self.region[3]}°")
        print(f"速度参数: {self.config.parameter}")
        print(f"目标深度: {self.config.depths} km")
        print(f"模型数量: {len(self.config.models)}")
        print(f"数据目录: {self.data_dir}")
        print(f"输出目录: {self.output_dir}")
        print("-" * 60)

    # ------------------------------------------------------------------ #
    # 数据读取与切片
    # ------------------------------------------------------------------ #
    def _model_nc_path(self, model_key: str) -> Path:
        """处理后 NetCDF 文件路径"""
        # 特殊处理 CSRM1.0 分辨率拆分模型
        if model_key == '2023_CSRM1.0_res0.2':
            return self.data_dir / '2023_CSRM1.0' / '2023_CSRM1.0_res0.2.nc'
        elif model_key == '2023_CSRM1.0_res0.4':
            return self.data_dir / '2023_CSRM1.0' / '2023_CSRM1.0_res0.4.nc'
        return self.data_dir / model_key / f"{model_key}_original.nc"



    def _extract_slice(self, model_key: str,
                       target_depth: float) -> Optional[Tuple[xr.DataArray, float]]:
        """
        提取指定模型在最邻近目标深度处的绝对速度切片。

        Args:
            model_key: 模型目录名（如 '2024_FWEA23'，或特殊的拆分模型如 '2023_CSRM1.0_res0.2'）
            target_depth: 目标深度 (km)

        Returns:
            (规则网格的 DataArray(dims=lat,lon), 实际使用深度) 或 None（无数据/深度越界）
        """
        nc_path = self._model_nc_path(model_key)
        if not nc_path.exists():
            self.logger.warning(f"  ⚠️ 缺少 NetCDF: {nc_path}")
            return None

        param = self.config.parameter
        try:
            with xr.open_dataset(nc_path) as ds:
                if param not in ds.data_vars:
                    self.logger.warning(f"  ⚠️ {model_key} 缺少参数 {param}")
                    return None

                depths = ds['depth'].values.astype(float)
                # 深度越界检查
                if (target_depth > float(np.nanmax(depths)) + self.config.depth_tolerance
                        or target_depth < float(np.nanmin(depths)) - self.config.depth_tolerance):
                    self.logger.info(f"  ⏭️ {model_key} 在 {target_depth} km 无覆盖 "
                                     f"(范围 {np.nanmin(depths):.0f}~{np.nanmax(depths):.0f} km)")
                    return None

                # 找最近深度索引，直接用 iloc 切片避免 GMT 非均匀坐标警告
                idx = int(np.argmin(np.abs(depths - target_depth)))
                actual_depth = float(depths[idx])
                da = ds[param].isel(depth=idx)
                # 统一维度顺序为 (latitude, longitude)
                da = da.transpose('latitude', 'longitude').load()
                # 重新赋予标准坐标（去掉 depth 维度的多余信息）
                da = da.drop_vars('depth', errors='ignore')

            preserve_support = (model_key == '2023_CSRM1.0_res0.2')
            grid = self._to_regular_grid(da, preserve_support=preserve_support)
            if grid is None or np.all(np.isnan(grid.values)):
                self.logger.info(f"  ⏭️ {model_key} 在 {actual_depth:.0f} km 切片全为 NaN")
                return None
            self.logger.info(f"  ✅ {model_key}: 取 {actual_depth:.1f} km 切片")
            return grid, actual_depth

        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"  ⚠️ 读取 {model_key} 失败: {e}")
            return None

    def _to_regular_grid(self,
                         da: xr.DataArray,
                         preserve_support: bool = False) -> Optional[xr.DataArray]:
        """
        将切片转换为 PyGMT 可用的规则网格 DataArray(dims=lat, lon)。

        处理策略:
        - 均匀网格: 直接返回（重命名维度即可）
        - 非均匀网格（如 CSRM1.0 混合 0.2°/0.4° 分辨率）: 先提取所有有效点，
          用 scipy.interpolate.griddata 做 linear 插值到规则网格，
          规则网格步长取原始步长中位数（保证不粗于原始最密区域）。
          linear 插值域外的格点设为 NaN（不做 nearest 外推），保持覆盖范围真实。
        """
        from scipy.interpolate import griddata  # 局部导入，避免全局依赖

        lon = da['longitude'].values.astype(float)
        lat = da['latitude'].values.astype(float)
        if lon.size < 2 or lat.size < 2:
            return None

        def _is_uniform(arr: np.ndarray) -> bool:
            d = np.diff(arr)
            return bool(np.allclose(d, d[0], rtol=1e-3, atol=1e-6))

        if _is_uniform(lon) and _is_uniform(lat):
            return da.rename({'latitude': 'lat', 'longitude': 'lon'})

        # ---------- 非均匀网格：用 griddata 重采样 ----------
        # 步长取原始中位数（对混合分辨率网格取最密一侧），但不小于 0.05°
        sp_lon = float(np.median(np.abs(np.diff(lon))))
        sp_lat = float(np.median(np.abs(np.diff(lat))))
        sp = max(min(sp_lon, sp_lat), 0.05)

        lon_reg = np.arange(lon.min(), lon.max() + sp * 0.5, sp)
        lat_reg = np.arange(lat.min(), lat.max() + sp * 0.5, sp)

        # 提取所有有效（非 NaN）散点
        vals = da.values  # (lat, lon)
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        mask = np.isfinite(vals)
        if mask.sum() < 4:
            return None

        pts = np.column_stack([lon_mesh[mask], lat_mesh[mask]])
        zz = vals[mask].astype(float)

        lon_out, lat_out = np.meshgrid(lon_reg, lat_reg)

        # linear 插值（仅在数据凸包内有值，域外自动 NaN）
        result = griddata(pts, zz, (lon_out, lat_out), method='linear')

        if preserve_support:
            # 可选：用原始覆盖掩膜约束插值结果，避免跨 NaN 空洞“连片”产生伪结构。
            # 仅对 CSRM res0.2 开启；res0.4 若开启会出现条纹伪缺口。
            full_pts = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
            full_support = np.isfinite(vals).astype(np.float32).ravel()
            support_out = griddata(full_pts, full_support, (lon_out, lat_out), method='nearest')
            result = np.where(support_out > 0.5, result, np.nan)

        result = result.astype(np.float32)

        da_out = xr.DataArray(
            result,
            dims=['lat', 'lon'],
            coords={'lat': lat_reg, 'lon': lon_reg},
        )
        return da_out

    def _csrm02_mask_on_target_grid(self,
                                    target_depth: float,
                                    lon_target: np.ndarray,
                                    lat_target: np.ndarray) -> Optional[np.ndarray]:
        """
        基于 CSRM res0.2 原始切片构建目标网格上的布尔掩膜（True=0.2 覆盖区）。
        说明：直接用原始数据生成掩膜，避免使用已插值网格导致边界失真。
        """
        from scipy.interpolate import griddata

        nc_02 = self._model_nc_path('2023_CSRM1.0_res0.2')
        if not nc_02.exists():
            return None

        param = self.config.parameter
        try:
            with xr.open_dataset(nc_02) as ds_02:
                if param not in ds_02.data_vars:
                    return None
                depths = ds_02['depth'].values.astype(float)
                idx = int(np.argmin(np.abs(depths - target_depth)))
                da_02 = ds_02[param].isel(depth=idx).transpose('latitude', 'longitude').load()

            lon_02 = da_02['longitude'].values.astype(float)
            lat_02 = da_02['latitude'].values.astype(float)
            vals_02 = da_02.values

            lon_mesh_02, lat_mesh_02 = np.meshgrid(lon_02, lat_02)
            points_02 = np.column_stack([lon_mesh_02.ravel(), lat_mesh_02.ravel()])
            support_02 = np.isfinite(vals_02).astype(np.float32).ravel()

            lon_out, lat_out = np.meshgrid(lon_target.astype(float), lat_target.astype(float))
            support_out = griddata(points_02, support_02, (lon_out, lat_out), method='linear')
            mask = np.isfinite(support_out) & (support_out > 0.5)
            return mask
        except Exception:  # noqa: BLE001
            return None

    def _model_series(self, grid: xr.DataArray) -> str:
        """计算单个模型切片的色标范围字符串 'vmin/vmax/step'（1~99 百分位）"""
        vals = grid.values[np.isfinite(grid.values)]
        if vals.size == 0:
            return "3.0/5.0/0.1"
        lo_p, hi_p = self.config.colorbar['percentile']
        vmin = float(np.floor(np.percentile(vals, lo_p) / 0.1) * 0.1)
        vmax = float(np.ceil(np.percentile(vals, hi_p) / 0.1) * 0.1)
        span = vmax - vmin
        for candidate in [0.05, 0.1, 0.2, 0.5, 1.0]:
            if span / candidate <= 15:
                step = candidate
                break
        else:
            step = 0.5
        return f"{vmin:.2f}/{vmax:.2f}/{step:.2f}"

    def _auto_series(self, slices: List[xr.DataArray],
                     target_depth: float) -> str:
        """
        确定共享色标范围字符串 'vmin/vmax/step'。

        逻辑：手动覆盖 > 所有模型 1~99 百分位数，边界对齐到 0.1 km/s
        """
        cfg = self.config.colorbar

        # 1. 手动覆盖
        if cfg['manual_series'] and target_depth in cfg['manual_series']:
            return cfg['manual_series'][target_depth]

        # 2. 百分位数
        all_vals = np.concatenate([
            g.values[np.isfinite(g.values)].ravel() for g in slices
        ]) if slices else np.array([])

        if all_vals.size == 0:
            return "3.0/5.0/0.1"

        lo_p, hi_p = cfg.get('percentile', (1.0, 99.0))
        vmin = float(np.percentile(all_vals, lo_p))
        vmax = float(np.percentile(all_vals, hi_p))

        # 对齐到 0.1 km/s
        vmin = np.floor(vmin / 0.1) * 0.1
        vmax = np.ceil(vmax / 0.1) * 0.1

        # step 从干净候选值中选最接近 ≤15 等分的最小值
        span = vmax - vmin
        for candidate in [0.05, 0.1, 0.2, 0.5, 1.0]:
            if span / candidate <= 15:
                step = candidate
                break
        else:
            step = 0.5
        return f"{vmin:.2f}/{vmax:.2f}/{step:.2f}"

    # ------------------------------------------------------------------ #
    # 绘图
    # ------------------------------------------------------------------ #
    def plot_depth(self, target_depth: float,
                   save_file: Optional[str] = None) -> Optional[str]:
        """绘制单个深度的九宫格多模型绝对速度切片图"""
        param = self.config.parameter
        param_label = param.upper()
        self.logger.info(f"\n🎨 绘制 {param_label} 切片九宫格 @ {target_depth} km ...")

        # 1. 预读取所有模型切片
        model_keys = list(self.config.models.keys())
        slices: Dict[str, Tuple[xr.DataArray, float]] = {}
        for key in model_keys:
            result = self._extract_slice(key, target_depth)
            if result is not None:
                slices[key] = result

        # CSRM 专项：0.2 区域作为 0.4 的内边界，直接用 0.2 显示掩膜挖空 0.4，避免视觉重叠
        if '2023_CSRM1.0_res0.4' in slices and '2023_CSRM1.0_res0.2' in slices:
            grid_04, depth_04 = slices['2023_CSRM1.0_res0.4']
            mask_02_on_04 = self._csrm02_mask_on_target_grid(
                target_depth=target_depth,
                lon_target=grid_04['lon'].values,
                lat_target=grid_04['lat'].values,
            )
            if mask_02_on_04 is not None:
                slices['2023_CSRM1.0_res0.4'] = (grid_04.where(~mask_02_on_04), depth_04)

        if not slices:
            self.logger.warning(f"⚠️ {target_depth} km 无任何模型数据，跳过")
            return None

        # 2. 共享色标范围（manual_series 手动指定优先，否则合并数据自动计算）
        series = self._auto_series([g for g, _ in slices.values()], target_depth)
        self.logger.info(f"   共享色标范围: {series} ({param_label} km/s)")

        if save_file is None:
            save_file = generate_filename(
                self.module_prefix,
                f"velocity_slices_{param}_{int(round(target_depth))}km",
                "jpg",
            )

        # 3. 建立图件并设置共享色标（makecpt 必须在 Figure() 之后调用）
        proj = f"M{self.config.grid['panel_width']}"
        nrows = self.config.grid['nrows']
        ncols = self.config.grid['ncols']

        fig = pygmt.Figure()
        pygmt.makecpt(
            cmap=self.config.colorbar['cmap'],
            series=series,
            continuous=True,
            reverse=self.config.colorbar['reverse'],
            background=True,
        )
        with fig.subplot(
            nrows=nrows,
            ncols=ncols,
            figsize=self.config.grid['figsize'],
            margins=self.config.grid['margins'],
            title=f"Absolute {param_label} Velocity @ {target_depth:.0f} km",
        ):
            for idx, key in enumerate(model_keys):
                if idx >= nrows * ncols:
                    break
                display_name = self.config.models[key]
                with fig.set_panel(panel=idx):
                    if key in slices:
                        grid, actual_depth = slices[key]
                        self._draw_model_panel(fig, proj, grid, display_name, actual_depth)
                    else:
                        # 无数据的模型显示 "No data"
                        self._draw_empty_panel(fig, proj, display_name)

        # 4. 图底共享色标条
        fig.colorbar(
            position=self.config.colorbar['shared_cbar_position'],
            frame=[f"xaf+f10p+lVs (km/s)"],
        )

        return self._save_figure(fig, save_file)

    def _draw_model_panel(self, fig: pygmt.Figure, projection: str,
                          grid: xr.DataArray, display_name: str,
                          actual_depth: float):
        """绘制单个模型子图（共享色标已在 plot_depth 中初始化）"""
        # GMT 6.6 要求轴间隔在前，边框/标题规范在后，且 +t 必须附在 WSen 上
        fig.basemap(
            region=self.region,
            projection=projection,
            frame=[self.config.axes['xaxis'],
                   self.config.axes['yaxis'],
                   f"WseN+t{display_name}"],
        )
        fig.grdimage(
            grid=grid,
            region=self.region,
            projection=projection,
            cmap=True,
            nan_transparent=True,
        )
        fig.coast(
            region=self.region,
            projection=projection,
            shorelines=self.config.coast['shorelines'],
            borders=self.config.coast['borders'],
            resolution=self.config.coast['resolution'],
        )

    def _draw_empty_panel(self, fig: pygmt.Figure, projection: str,
                          display_name: str):
        """无数据模型的留白子图"""
        fig.basemap(
            region=self.region,
            projection=projection,
            frame=[self.config.axes['xaxis'],
                   self.config.axes['yaxis'],
                   f"WseN+t{display_name}"],
        )
        fig.coast(
            region=self.region,
            projection=projection,
            shorelines=self.config.coast['shorelines'],
            borders=self.config.coast['borders'],
            resolution=self.config.coast['resolution'],
        )
        lon_c = (self.region[0] + self.region[1]) / 2
        lat_c = (self.region[2] + self.region[3]) / 2
        fig.text(
            x=lon_c, y=lat_c, text="No data at this depth",
            font="9p,Helvetica-Oblique,gray30", no_clip=False,
        )

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    def _save_figure(self, fig: pygmt.Figure, save_file: str) -> str:
        out_jpg = self.output_dir / save_file
        if self.config.output['save_jpg']:
            fig.savefig(str(out_jpg),
                        dpi=self.config.output['dpi_jpg'],
                        crop=self.config.output['crop'])
            self.logger.info(f"🖼️ JPG 已保存: {out_jpg}")
        if self.config.output['save_pdf']:
            out_pdf = out_jpg.with_suffix('.pdf')
            fig.savefig(str(out_pdf),
                        dpi=self.config.output['dpi_pdf'],
                        crop=self.config.output['crop'])
            self.logger.info(f"📄 PDF 已保存: {out_pdf}")
        return str(out_jpg)

    # ------------------------------------------------------------------ #
    # 一键全流程
    # ------------------------------------------------------------------ #
    def plot_all(self) -> Dict[str, str]:
        """对配置中的所有深度生成九宫格切片图"""
        outputs: Dict[str, str] = {}
        for depth in self.config.depths:
            out = self.plot_depth(depth)
            if out:
                outputs[f"{int(round(depth))}km"] = out
        return outputs


def main():
    """主函数"""
    print("🗺️ EASTASIA-FWI 多模型绝对速度切片可视化（九宫格）")
    print("=" * 70)
    try:
        plotter = VelocitySlicesPlotter()
        outputs = plotter.plot_all()
        if outputs:
            print("\n✅ 全部绘图完成:")
            for k, v in outputs.items():
                print(f"  - {k}: {v}")
        else:
            print("\n⚠️ 未生成任何图件，请检查数据与深度配置")
        print(f"\n📁 输出目录: {plotter.output_dir}")
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:  # noqa: BLE001
        print(f"\n❌ 出现错误: {e}")
        import traceback
        traceback.print_exc()
    print("=" * 70)


if __name__ == "__main__":
    main()
