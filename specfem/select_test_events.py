"""
EASTASIA-FWI 正演测试震源选取
==============================

功能描述:
- 从 GCMT3D (Sawade et al., 2022) 目录中挑选正演对比测试用的代表性震源
- 按 SPECFEM3D Globe chunk 几何、台站覆盖、方位角空隙、震级与深度多样性打分
- 输出标准 CMTSOLUTION、ECEF 格式 CMTSOLUTION、选源清单 CSV 与分布图

科学原理:
- 模型对比要求射线路径充分覆盖研究区的地壳、岩石圈与地幔过渡带。单一震源
  只照亮一条方位的窄条带，不同模型的差异会被路径采样偏差掩盖。因此按
  「构造省 × 深度档」划分角色，每个角色取一个最优事件，使 10 个震源在
  方位、深度与机制类型上互补。
- 震源必须距 chunk 边界足够远。regional chunk 的四周是吸收边界，紧邻边界
  的震源会把边界反射混入有效时窗。
- 选 GCMT3D 而非 GCMT：GCMT3D 的质心位置是在三维地球 (GLAD-M25) 中反演
  得到的，去除了 1D 定位带来的系统偏移，震源侧误差不会被误读成模型差异。

作者: EASTASIA-FWI Team
日期: 2026-08-03
版本: v1.0
"""

from __future__ import annotations

import argparse
import logging
import math
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

R_EARTH_KM: float = 6371.0
FLATTENING: float = 1.0 / 298.25          # SPECFEM 地理纬度→地心纬度所用扁率


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Event:
    """一个 GCMT3D 震源。矩张量分量单位 dyn·cm，与目录原始单位一致。"""
    name: str
    origin: datetime                       # PDE 发震时刻
    lat: float
    lon: float
    depth: float                           # km
    time_shift: float                      # s
    half_duration: float                   # s
    mrr: float
    mtt: float
    mpp: float
    mrt: float
    mrp: float
    mtp: float
    pde_header: str                        # 原始首行，写回 CMTSOLUTION 用

    # 以下字段由选取流程填充
    mw: float = 0.0
    m0: float = 0.0
    mech: str = ''                         # thrust / normal / strike-slip / oblique
    chunk_xi: float = 0.0
    chunk_eta: float = 0.0
    chunk_margin: float = 0.0              # 距 chunk 边界的度数，负值表示在 chunk 外
    n_station: int = 0                     # 可用台站数
    az_gap: float = 360.0                  # 最大方位角空隙
    dist_med: float = 0.0                  # 台站震中距中位数
    province: str = ''
    score: float = 0.0
    score_terms: Dict[str, float] = field(default_factory=dict)

    @property
    def centroid_time(self) -> datetime:
        """质心时刻 = PDE 时刻 + time shift。"""
        return self.origin + timedelta(seconds=self.time_shift)


@dataclass
class Province:
    """一个目标构造省（选源角色）。"""
    key: str
    label: str                             # 图件与报告用英文名
    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]
    depth_range: Tuple[float, float]
    rationale: str                         # 中文说明：这个角色照亮什么结构

    def contains(self, ev: Event) -> bool:
        return (self.lat_range[0] <= ev.lat <= self.lat_range[1]
                and self.lon_range[0] <= ev.lon <= self.lon_range[1]
                and self.depth_range[0] <= ev.depth <= self.depth_range[1])


# ============================================================================
# 配置
# ============================================================================

class EventSelectionConfig:
    """选源配置。"""

    def __init__(self) -> None:
        # 硬筛条件
        self.filters: Dict[str, Any] = {
            'mw_range': (5.5, 7.0),        # 下限保信噪比，上限保点源近似与短震源时间函数
            'depth_max': 700.0,
            'year_min': 2000,              # 宽频带台网在 2000 年后才足够密
            'chunk_margin_min': 12.0,      # 震源距 chunk 边界的最小度数
            'min_station': 40,             # 可用台站数下限
            'az_gap_max': 200.0,           # 最大方位角空隙上限
            'clean_window_s': 1800.0,      # 前后半小时内不得有其它 Mw≥5.5 事件
        }

        # 台站可用性判据
        self.station: Dict[str, Any] = {
            'margin_min': 2.0,             # 台站也要在 chunk 内且留 2° 余量
            'dist_range': (3.0, 40.0),     # 震中距窗口（度），上限受 RECORD_LENGTH 约束
        }

        # 打分权重（各项已归一化到 0–1）
        self.weights: Dict[str, float] = {
            'coverage': 0.35,              # 可用台站数
            'azgap': 0.25,                 # 方位角均匀性
            'margin': 0.15,                # 离 chunk 边界的余量
            'magnitude': 0.15,             # 震级适中程度
            'recency': 0.10,               # 年份越新，实测数据越可能拿得到
        }
        self.mw_optimum: float = 6.2
        self.mw_sigma: float = 0.5

        # 目标构造省（选源角色）。深度分档是刻意设计的：三个深源分别从北、东、
        # 南三个方位穿透地幔过渡带，是区分各模型 MTZ 结构最有力的组合；浅源
        # 负责地壳与岩石圈。顺序即优先级，某档无候选时按空间散布补齐。
        self.provinces: List[Province] = [
            Province('NEChina_deep', 'NE China / Japan Sea (deep)',
                     (39.0, 47.0), (127.0, 137.0), (450.0, 700.0),
                     '西太平洋滞留板片，自北穿透地幔过渡带，最能区分各模型的 MTZ'),
            Province('Celebes_deep', 'Celebes Sea / Mindanao (deep)',
                     (2.0, 13.0), (118.0, 130.0), (400.0, 700.0),
                     '自南入射的深源，与 NEChina_deep 构成 MTZ 的南北互补路径'),
            Province('Japan_deep', 'Pacific slab beneath Japan (deep)',
                     (27.0, 38.0), (128.0, 142.0), (300.0, 500.0),
                     '自东入射的深源，与南北两个深源一起构成 MTZ 的三向交叉照明'),
            Province('Japan_interm', 'Japan arc (intermediate)',
                     (25.0, 41.0), (126.0, 143.0), (60.0, 300.0),
                     '日本俯冲带 Wadati-Benioff 带，东侧入射的岩石圈路径'),
            Province('Myanmar_Burma', 'Myanmar / Burma arc',
                     (18.0, 28.0), (91.0, 99.0), (30.0, 200.0),
                     '缅甸板片中源，直接照亮青藏东南缘与印支地块'),
            Province('Taiwan_Ryukyu', 'Taiwan / Ryukyu (shallow)',
                     (21.0, 29.0), (119.0, 132.0), (0.0, 70.0),
                     '琉球—台湾碰撞带浅源，东南方位且台站最密'),
            Province('Sunda_Sumatra', 'Sunda / Sumatra / Andaman',
                     (-10.0, 10.0), (90.0, 115.0), (0.0, 100.0),
                     '西南方位唯一的震源，不选则该扇区留下近百度的方位角空洞'),
            Province('Tibet_Himalaya', 'Tibet / Himalaya',
                     (26.0, 37.0), (78.0, 103.0), (0.0, 45.0),
                     '青藏高原地壳震源，对厚地壳与岩石圈最敏感'),
            Province('TienShan_Pamir', 'Tien Shan / Pamir / Hindu Kush',
                     (33.0, 45.0), (66.0, 85.0), (0.0, 300.0),
                     '西侧方位，检验帕米尔—天山下方的岩石圈结构'),
            Province('Mongolia_NChina', 'Mongolia / N China / Baikal',
                     (38.0, 56.0), (95.0, 125.0), (0.0, 60.0),
                     '正北方位，路径穿越华北克拉通与中亚造山带'),
        ]

        self.n_target: int = 10

        self.visualization: Dict[str, Any] = {
            'dpi': 300,
            'figure_format': ['jpg'],
            # 与 5_2_GCMT.plot_test_database_globe 一致：General Perspective
            # 投影中心取 SPECFEM chunk 中心，这样 chunk 边框在球面上接近正方形
            'width_cm': 14.0,              # 地球直径
            'altitude': 2.75,              # 观测高度（地球半径倍数）
            'grid_interval': 45,           # 经纬网间隔（度）
            'title': '{n} test events for the SPECFEM3D Globe model comparison',

            # 底图（半球投影用纯色海陆，与 5_2 地球图一致；地形在球面上只会抢眼）
            'coast_resolution': 'low',
            'water_color': 'white',
            'land_color': 'gray75',
            'border_pen': '1/0.2p,gray60',

            # 震源球与深度色标
            'depth_cmap': 'seis',
            'depth_series': [0, 700, 50],
            'beachball_scale_cm': 0.38,

            # 台站
            'station_color': 'steelblue',

            # 标注避让（在以 chunk 中心为原点的方位等距平面上做）
            'label_font_pt': 7.0,
            'label_radii_deg': (1.5, 3.0, 5.0, 8.0, 12.0),
            'leader_min_deg': 4.0,

            'gmt_config': {
                'MAP_FRAME_TYPE': 'plain',
                'MAP_FRAME_WIDTH': '1.5p',
                'MAP_FRAME_PEN': '0.6p',
                'MAP_GRID_PEN_PRIMARY': '0.25p,gray70',
                'FONT_ANNOT_PRIMARY': '9p,Helvetica,black',
                'FONT_LABEL': '9p,Helvetica,black',
                'FONT_TITLE': '12p,Helvetica-Bold,black',
            },
        }
        self.logging: Dict[str, Any] = {'level': 'INFO', 'console_output': True}


# ============================================================================
# 几何工具
# ============================================================================

def euler_rotation_matrix(center_lon: float, center_lat: float,
                          gamma: float) -> np.ndarray:
    """
    构造 SPECFEM3D Globe 的 chunk 旋转矩阵。

    与 src/shared/euler_angles.f90 中的定义逐项一致：矩阵把 chunk 局部坐标
    映射到全球坐标，局部 +z 轴指向 chunk 中心。

    Args:
        center_lon: chunk 中心经度（度）
        center_lat: chunk 中心纬度（度）
        gamma: 方位角旋转 GAMMA_ROTATION_AZIMUTH（度）

    Returns:
        3×3 旋转矩阵
    """
    a = math.radians(center_lon)
    b = math.radians(90.0 - center_lat)
    g = math.radians(gamma)
    sa, ca = math.sin(a), math.cos(a)
    sb, cb = math.sin(b), math.cos(b)
    sg, cg = math.sin(g), math.cos(g)
    return np.array([
        [cg * cb * ca - sg * sa, -sg * cb * ca - cg * sa, sb * ca],
        [cg * cb * sa + sg * ca, -sg * cb * sa + cg * ca, sb * sa],
        [-cg * sb,                sg * sb,               cb],
    ])


def geographic_to_unit_vector(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """地理经纬度转单位矢量。先按 SPECFEM 的扁率换算成地心纬度。"""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_geoc = np.degrees(np.arctan(np.tan(np.radians(lat))
                                    * (1.0 - FLATTENING) ** 2))
    theta = np.radians(90.0 - lat_geoc)
    phi = np.radians(lon)
    return np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=-1)


class ChunkGeometry:
    """SPECFEM3D Globe 单 chunk 的几何，用于判断点是否落在计算域内。"""

    def __init__(self, center_lat: float, center_lon: float,
                 gamma: float, width_xi: float, width_eta: float) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.gamma = gamma
        self.width_xi = width_xi
        self.width_eta = width_eta
        self._R = euler_rotation_matrix(center_lon, center_lat, gamma)

    @classmethod
    def from_par_file(cls, par_file: Path) -> 'ChunkGeometry':
        """从 Par_file 读取 chunk 几何参数。"""
        keys: Dict[str, Optional[str]] = {
            'CENTER_LATITUDE_IN_DEGREES': None,
            'CENTER_LONGITUDE_IN_DEGREES': None,
            'GAMMA_ROTATION_AZIMUTH': None,
            'ANGULAR_WIDTH_XI_IN_DEGREES': None,
            'ANGULAR_WIDTH_ETA_IN_DEGREES': None,
            'NCHUNKS': None,
        }
        for line in par_file.read_text().splitlines():
            line = line.split('#')[0]
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            if k in keys:
                # Par_file 里的浮点常量带 Fortran 双精度后缀，如 80.d0
                keys[k] = v.strip().replace('d0', '').replace('D0', '').strip()
        missing = [k for k, v in keys.items() if v is None]
        if missing:
            raise ValueError(f'Par_file 缺少参数: {missing}')
        if int(keys['NCHUNKS']) != 1:  # type: ignore[arg-type]
            raise ValueError('本工具只支持 NCHUNKS = 1 的 regional chunk')
        return cls(float(keys['CENTER_LATITUDE_IN_DEGREES']),   # type: ignore[arg-type]
                   float(keys['CENTER_LONGITUDE_IN_DEGREES']),  # type: ignore[arg-type]
                   float(keys['GAMMA_ROTATION_AZIMUTH']),       # type: ignore[arg-type]
                   float(keys['ANGULAR_WIDTH_XI_IN_DEGREES']),  # type: ignore[arg-type]
                   float(keys['ANGULAR_WIDTH_ETA_IN_DEGREES'])) # type: ignore[arg-type]

    def xi_eta(self, lat: np.ndarray, lon: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        """返回点在 chunk 局部坐标下的 (xi, eta)，单位度。"""
        v_local = geographic_to_unit_vector(lat, lon) @ self._R
        xi = np.degrees(np.arctan2(v_local[..., 0], v_local[..., 2]))
        eta = np.degrees(np.arctan2(v_local[..., 1], v_local[..., 2]))
        # 背面的点 (v_local_z < 0) 用 arctan2 会折回来，显式标成远在域外
        behind = v_local[..., 2] <= 0.0
        xi = np.where(behind, 180.0, xi)
        eta = np.where(behind, 180.0, eta)
        return xi, eta

    def margin(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """距 chunk 边界的度数。正值在域内，负值在域外。"""
        xi, eta = self.xi_eta(lat, lon)
        return np.minimum(self.width_xi / 2.0 - np.abs(xi),
                          self.width_eta / 2.0 - np.abs(eta))

    def outline(self, n: int = 200, inset: float = 0.0
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        chunk 边框的地理经纬度，用于绘图。

        Args:
            n: 每条边的采样点数
            inset: 边框向内收缩的度数。取 chunk_margin_min 即得到选源的
                   可用区边界——落在该框以外的震源离吸收边界太近

        Returns:
            (纬度数组, 经度数组)
        """
        hx = self.width_xi / 2.0 - inset
        he = self.width_eta / 2.0 - inset
        t = np.linspace(-1.0, 1.0, n)
        xi = np.concatenate([t * hx, np.full(n, hx), t[::-1] * hx, np.full(n, -hx)])
        eta = np.concatenate([np.full(n, -he), t * he, np.full(n, he), t[::-1] * he])
        x, y = np.tan(np.radians(xi)), np.tan(np.radians(eta))
        g = 1.0 / np.sqrt(1.0 + x * x + y * y)
        v_local = np.stack([x * g, y * g, g], axis=-1)
        v = v_local @ self._R.T
        lat_geoc = np.degrees(np.arcsin(np.clip(v[:, 2], -1.0, 1.0)))
        lat = np.degrees(np.arctan(np.tan(np.radians(lat_geoc))
                                   / (1.0 - FLATTENING) ** 2))
        lon = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        return lat, lon


def dist_azimuth(ev_lat: float, ev_lon: float,
                 st_lat: np.ndarray, st_lon: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    球面震中距与从震源看向台站的方位角。

    Args:
        ev_lat, ev_lon: 震源经纬度（度）
        st_lat, st_lon: 台站经纬度数组（度）

    Returns:
        (震中距[度], 方位角[度, 0-360])
    """
    p1 = math.radians(ev_lat)
    p2 = np.radians(st_lat)
    dl = np.radians(st_lon - ev_lon)
    cos_d = (math.sin(p1) * np.sin(p2)
             + math.cos(p1) * np.cos(p2) * np.cos(dl))
    delta = np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))
    az = np.degrees(np.arctan2(np.sin(dl) * np.cos(p2),
                               math.cos(p1) * np.sin(p2)
                               - math.sin(p1) * np.cos(p2) * np.cos(dl)))
    return delta, np.mod(az, 360.0)


def azimuthal_gap(azimuths: np.ndarray) -> float:
    """最大方位角空隙。台站少于 2 个时返回 360。"""
    if azimuths.size < 2:
        return 360.0
    a = np.sort(np.mod(azimuths, 360.0))
    gaps = np.diff(a)
    return float(max(gaps.max(), 360.0 - (a[-1] - a[0])))


def classify_mechanism(mrr: float, mtt: float, mpp: float,
                       mrt: float, mrp: float, mtp: float) -> str:
    """
    按 Frohlich (1992) 的主轴倾伏角对震源机制分类。

    矩张量以 (r, theta, phi) 给出，r 向上。主轴倾伏角相对水平面计算。
    """
    m = np.array([[mrr, mrt, mrp],
                  [mrt, mtt, mtp],
                  [mrp, mtp, mpp]], dtype=float)
    vals, vecs = np.linalg.eigh(m)
    # eigh 升序：vals[0]=P(最小), vals[2]=T(最大), vals[1]=B
    plunge = np.degrees(np.arcsin(np.clip(np.abs(vecs[0, :]), 0.0, 1.0)))
    p_pl, b_pl, t_pl = plunge[0], plunge[1], plunge[2]
    if t_pl >= 50.0:
        return 'normal'
    if p_pl >= 50.0:
        return 'thrust'
    if b_pl >= 60.0:
        return 'strike-slip'
    return 'oblique'


# ============================================================================
# 主类
# ============================================================================

class TestEventSelector:
    """从 GCMT3D 目录中挑选正演测试震源。"""

    def __init__(self,
                 catalog: Path,
                 par_file: Path,
                 stations: Path,
                 output_dir: Path,
                 config: Optional[EventSelectionConfig] = None) -> None:
        self.config = config or EventSelectionConfig()
        self.catalog_path = Path(catalog)
        self.par_file = Path(par_file)
        self.stations_path = Path(stations)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logger()

        self.chunk = ChunkGeometry.from_par_file(self.par_file)
        self.sta_lat, self.sta_lon, self.sta_name = self._load_stations()
        self.events: List[Event] = []
        self.selected: List[Event] = []

        self._print_config_summary()

    # ---------------- 初始化辅助 ----------------

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('EASTASIA-FWI.TestEventSelector')
        logger.setLevel(getattr(logging, self.config.logging['level']))
        if not logger.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                             datefmt='%H:%M:%S'))
            logger.addHandler(h)
        return logger

    def _print_config_summary(self) -> None:
        c = self.chunk
        self.logger.info('🎯 正演测试震源选取')
        self.logger.info(f'   chunk 中心 ({c.center_lat:.1f}°N, {c.center_lon:.1f}°E)  '
                         f'张角 {c.width_xi:.0f}°×{c.width_eta:.0f}°  '
                         f'方位旋转 {c.gamma:.0f}°')
        m = self.chunk.margin(self.sta_lat, self.sta_lon)
        keep = m >= self.config.station['margin_min']
        self.logger.info(f'   台站 {len(self.sta_lat)} 个，'
                         f'chunk 内可用 {int(keep.sum())} 个，'
                         f'域外剔除 {int((~keep).sum())} 个')
        self.sta_usable = keep

    def _load_stations(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        lat, lon, name = [], [], []
        for line in self.stations_path.read_text().splitlines():
            f = line.split()
            if len(f) < 4:
                continue
            name.append(f'{f[1]}.{f[0]}')
            lat.append(float(f[2]))
            lon.append(float(f[3]))
        return np.array(lat), np.array(lon), name

    # ---------------- 目录读取 ----------------

    def load_catalog(self) -> None:
        """读取 GCMT3D 目录（13 行一个事件的 CMTSOLUTION 串联格式）。"""
        lines = self.catalog_path.read_text().splitlines()
        if len(lines) % 13 != 0:
            self.logger.warning(f'目录行数 {len(lines)} 不是 13 的整数倍，'
                                f'按整块解析，忽略末尾 {len(lines) % 13} 行')

        events: List[Event] = []
        for i in range(0, len(lines) - 12, 13):
            try:
                events.append(self._parse_block(lines[i:i + 13]))
            except (ValueError, IndexError) as exc:
                self.logger.debug(f'第 {i + 1} 行起的事件块解析失败: {exc}')
        self.events = events
        self.logger.info(f'📖 读入 {len(events)} 个 GCMT3D 事件  '
                         f'({min(e.origin.year for e in events)}–'
                         f'{max(e.origin.year for e in events)})')

    @staticmethod
    def _parse_block(block: Sequence[str]) -> Event:
        h = block[0].split()
        sec = float(h[6])
        origin = (datetime(int(h[1]), int(h[2]), int(h[3]),
                           int(h[4]), int(h[5]))
                  + timedelta(seconds=sec))

        def val(idx: int) -> float:
            return float(block[idx].split(':', 1)[1])

        return Event(
            name=block[1].split(':', 1)[1].strip(),
            origin=origin,
            time_shift=val(2), half_duration=val(3),
            lat=val(4), lon=val(5), depth=val(6),
            mrr=val(7), mtt=val(8), mpp=val(9),
            mrt=val(10), mrp=val(11), mtp=val(12),
            pde_header=block[0],
        )

    # ---------------- 指标计算 ----------------

    def compute_metrics(self) -> None:
        """为每个事件计算震级、机制、chunk 余量。"""
        lat = np.array([e.lat for e in self.events])
        lon = np.array([e.lon for e in self.events])
        margin = self.chunk.margin(lat, lon)
        xi, eta = self.chunk.xi_eta(lat, lon)

        for k, ev in enumerate(self.events):
            # 标量矩：Silver & Jordan 定义 M0 = sqrt(M:M/2)
            m2 = (ev.mrr ** 2 + ev.mtt ** 2 + ev.mpp ** 2
                  + 2.0 * (ev.mrt ** 2 + ev.mrp ** 2 + ev.mtp ** 2))
            ev.m0 = math.sqrt(0.5 * m2)
            ev.mw = (math.log10(ev.m0) - 16.1) / 1.5 if ev.m0 > 0 else 0.0
            ev.mech = classify_mechanism(ev.mrr, ev.mtt, ev.mpp,
                                         ev.mrt, ev.mrp, ev.mtp)
            ev.chunk_margin = float(margin[k])
            ev.chunk_xi = float(xi[k])
            ev.chunk_eta = float(eta[k])

    def _station_metrics(self, ev: Event) -> Tuple[int, float, float]:
        """事件的可用台站数、方位角空隙、震中距中位数。"""
        d, az = dist_azimuth(ev.lat, ev.lon, self.sta_lat, self.sta_lon)
        lo, hi = self.config.station['dist_range']
        ok = self.sta_usable & (d >= lo) & (d <= hi)
        n = int(ok.sum())
        if n == 0:
            return 0, 360.0, 0.0
        return n, azimuthal_gap(az[ok]), float(np.median(d[ok]))

    # ---------------- 筛选与打分 ----------------

    def filter_candidates(self) -> List[Event]:
        """按硬条件筛出候选池。"""
        f = self.config.filters
        mw_lo, mw_hi = f['mw_range']

        stage = {'总数': len(self.events)}
        pool = [e for e in self.events if e.origin.year >= f['year_min']]
        stage[f"年份 ≥{f['year_min']}"] = len(pool)
        pool = [e for e in pool if mw_lo <= e.mw <= mw_hi]
        stage[f'Mw {mw_lo}–{mw_hi}'] = len(pool)
        pool = [e for e in pool if e.depth <= f['depth_max']]
        stage[f"深度 ≤{f['depth_max']:.0f} km"] = len(pool)
        pool = [e for e in pool if e.chunk_margin >= f['chunk_margin_min']]
        stage[f"距 chunk 边界 ≥{f['chunk_margin_min']:.0f}°"] = len(pool)

        for ev in pool:
            ev.n_station, ev.az_gap, ev.dist_med = self._station_metrics(ev)
        pool = [e for e in pool if e.n_station >= f['min_station']]
        stage[f"可用台站 ≥{f['min_station']}"] = len(pool)
        pool = [e for e in pool if e.az_gap <= f['az_gap_max']]
        stage[f"方位角空隙 ≤{f['az_gap_max']:.0f}°"] = len(pool)

        pool = self._filter_interference(pool, f['clean_window_s'])
        stage[f"±{f['clean_window_s'] / 60:.0f} min 内无干扰事件"] = len(pool)

        self.logger.info('🔎 逐级筛选:')
        for k, v in stage.items():
            self.logger.info(f'   {k:<28s} {v:6d}')
        return pool

    def _filter_interference(self, pool: List[Event],
                             window_s: float) -> List[Event]:
        """
        剔除时窗内存在其它 Mw≥5.5 事件的震源。

        相邻大事件的尾波会叠加到目标事件的记录上，实测波形不可用；即便只比
        合成波形，选这类事件也会让后续接观测数据时白做一遍。
        """
        others = sorted((e for e in self.events if e.mw >= 5.5),
                        key=lambda e: e.origin)
        times = np.array([e.origin.timestamp() for e in others])
        keep = []
        for ev in pool:
            t = ev.origin.timestamp()
            lo = np.searchsorted(times, t - window_s)
            hi = np.searchsorted(times, t + window_s)
            neighbours = [others[i] for i in range(lo, hi)
                          if others[i].name != ev.name]
            if not neighbours:
                keep.append(ev)
        return keep

    def score(self, pool: List[Event]) -> None:
        """给候选事件打分。各分项归一化到 0–1 后加权求和。"""
        w = self.config.weights
        n_max = max((e.n_station for e in pool), default=1) or 1
        year_lo = self.config.filters['year_min']
        year_hi = max((e.origin.year for e in pool), default=year_lo)
        year_span = max(year_hi - year_lo, 1)

        for ev in pool:
            terms = {
                'coverage': ev.n_station / n_max,
                'azgap': max(0.0, (360.0 - ev.az_gap) / 360.0),
                'margin': min(ev.chunk_margin / 25.0, 1.0),
                'magnitude': math.exp(-0.5 * ((ev.mw - self.config.mw_optimum)
                                              / self.config.mw_sigma) ** 2),
                'recency': (ev.origin.year - year_lo) / year_span,
            }
            ev.score_terms = terms
            ev.score = sum(w[k] * v for k, v in terms.items())

    # ---------------- 选取 ----------------

    def select(self, pool: List[Event]) -> List[Event]:
        """
        按构造省逐个取最优事件，凑不满目标数时用空间散布补齐。

        Args:
            pool: 通过硬筛并已打分的候选事件

        Returns:
            选中的事件列表，按构造省顺序排列
        """
        chosen: List[Event] = []
        used: set = set()

        for prov in self.config.provinces:
            cands = [e for e in pool
                     if prov.contains(e) and e.name not in used]
            if not cands:
                self.logger.warning(f'⚠️  构造省 {prov.key} 无合格候选，留待补齐')
                continue
            best = max(cands, key=lambda e: e.score)
            best.province = prov.key
            chosen.append(best)
            used.add(best.name)
            self.logger.info(f'   {prov.key:<18s} → {best.name}  '
                             f'Mw{best.mw:.1f}  {best.depth:5.1f} km  '
                             f'score {best.score:.3f}  '
                             f'（{len(cands)} 个候选）')

        # 补齐：每次挑与已选事件球面距离最远的高分事件，避免扎堆
        while len(chosen) < self.config.n_target:
            rest = [e for e in pool if e.name not in used]
            if not rest:
                self.logger.warning('候选池耗尽，选出数量少于目标数')
                break
            best, best_val = None, -np.inf
            for ev in rest:
                d, _ = dist_azimuth(ev.lat, ev.lon,
                                    np.array([c.lat for c in chosen]),
                                    np.array([c.lon for c in chosen]))
                val = ev.score + 0.02 * float(d.min())
                if val > best_val:
                    best, best_val = ev, val
            assert best is not None
            best.province = 'fill'
            chosen.append(best)
            used.add(best.name)
            self.logger.info(f'   {"补齐":<18s} → {best.name}  '
                             f'Mw{best.mw:.1f}  {best.depth:5.1f} km  '
                             f'score {best.score:.3f}')

        chosen.sort(key=lambda e: e.origin)
        self.selected = chosen
        return chosen

    # ---------------- 输出 ----------------

    def write_cmtsolutions(self, ecef_script: Optional[Path] = None) -> Path:
        """写出标准 CMTSOLUTION，并在可能时一并生成 ECEF 版本。"""
        cmt_dir = self.output_dir / 'CMTSOLUTION'
        cmt_dir.mkdir(parents=True, exist_ok=True)
        ecef_dir = self.output_dir / 'CMTSOLUTION_ECEF'

        for ev in self.selected:
            path = cmt_dir / f'CMTSOLUTION_{ev.name}'
            path.write_text(
                f'{ev.pde_header}\n'
                f'event name:     {ev.name}\n'
                f'time shift:     {ev.time_shift:.4f}\n'
                f'half duration:  {ev.half_duration:.4f}\n'
                f'latitude:       {ev.lat:.4f}\n'
                f'longitude:      {ev.lon:.4f}\n'
                f'depth:          {ev.depth:.4f}\n'
                f'Mrr:            {ev.mrr:.6e}\n'
                f'Mtt:            {ev.mtt:.6e}\n'
                f'Mpp:            {ev.mpp:.6e}\n'
                f'Mrt:            {ev.mrt:.6e}\n'
                f'Mrp:            {ev.mrp:.6e}\n'
                f'Mtp:            {ev.mtp:.6e}\n')
        self.logger.info(f'💾 CMTSOLUTION × {len(self.selected)} → {cmt_dir}')

        if ecef_script is not None and Path(ecef_script).is_file():
            ecef_dir.mkdir(parents=True, exist_ok=True)
            n_ok = 0
            for ev in self.selected:
                src = cmt_dir / f'CMTSOLUTION_{ev.name}'
                dst = ecef_dir / f'CMTSOLUTION_{ev.name}'
                try:
                    subprocess.run([sys.executable, str(ecef_script),
                                    str(src), str(dst)],
                                   check=True, capture_output=True)
                    n_ok += 1
                except subprocess.CalledProcessError as exc:
                    self.logger.error(
                        f'{ev.name} ECEF 转换失败: '
                        f'{exc.stderr.decode(errors="replace")[:300]}')
            self.logger.info(f'💾 CMTSOLUTION_ECEF × {n_ok} → {ecef_dir}')
        return cmt_dir

    def write_summary(self) -> Path:
        """写出选源清单 CSV。"""
        rows = []
        prov_label = {p.key: p.label for p in self.config.provinces}
        for idx, ev in enumerate(self.selected, start=1):
            # 从 chunk 中心看向震源的方位角，用于核对 10 个震源的方位是否均匀
            _, az_c = dist_azimuth(self.chunk.center_lat, self.chunk.center_lon,
                                   np.array([ev.lat]), np.array([ev.lon]))
            rows.append({
                'id': idx,
                'event': ev.name,
                'province': ev.province,
                'province_label': prov_label.get(ev.province, 'fill-in'),
                'origin_utc': ev.origin.strftime('%Y-%m-%dT%H:%M:%S'),
                'latitude': round(ev.lat, 4),
                'longitude': round(ev.lon, 4),
                'depth_km': round(ev.depth, 2),
                'Mw': round(ev.mw, 2),
                'M0_dyncm': f'{ev.m0:.4e}',
                'mechanism': ev.mech,
                'half_duration_s': round(ev.half_duration, 2),
                'time_shift_s': round(ev.time_shift, 3),
                'n_station': ev.n_station,
                'az_gap_deg': round(ev.az_gap, 1),
                'dist_median_deg': round(ev.dist_med, 1),
                'chunk_margin_deg': round(ev.chunk_margin, 1),
                'az_from_center_deg': round(float(az_c[0]), 1),
                'score': round(ev.score, 4),
            })
        df = pd.DataFrame(rows)
        csv = self.output_dir / 'selected_test_events.csv'
        df.to_csv(csv, index=False)
        self.logger.info(f'💾 选源清单 → {csv}')

        # 供 run_all_test.sh 直接 source 的事件名清单
        lst = self.output_dir / 'events.list'
        lst.write_text('\n'.join(e.name for e in self.selected) + '\n')
        self.logger.info(f'💾 事件名清单 → {lst}')

        print('\n' + '=' * 124)
        print(df[['id', 'event', 'origin_utc', 'latitude', 'longitude',
                  'depth_km', 'Mw', 'mechanism', 'half_duration_s',
                  'n_station', 'az_gap_deg', 'az_from_center_deg',
                  'province']].to_string(index=False))
        print('=' * 124)

        az = np.sort(df['az_from_center_deg'].to_numpy())
        gap = max(float(np.diff(az).max()), 360.0 - (az[-1] - az[0]))
        self.logger.info(f'📐 10 个震源相对 chunk 中心的最大方位角空隙: {gap:.0f}°')
        dep = df['depth_km'].to_numpy()
        self.logger.info(
            f'📐 深度分档: <50 km {(dep < 50).sum()} 个 | '
            f'50–300 km {((dep >= 50) & (dep < 300)).sum()} 个 | '
            f'≥300 km {(dep >= 300).sum()} 个')
        return csv

    # ---------------- 绘图 ----------------

    def _projection_string(self) -> str:
        """
        General Perspective 投影字符串。

        与 5_2_GCMT.plot_test_database_globe 同一套写法：
        G<lon>/<lat>/<altitude>/<azimuth>/<tilt>/<twist>/<W>/<H>/<width>
        中心取 chunk 中心，这样 SPECFEM 的 80°×80° chunk 在球面上接近正方形。
        """
        vis = self.config.visualization
        return (f"G{self.chunk.center_lon}/{self.chunk.center_lat}/"
                f"{vis['altitude']}/0/0/0/0/0/{vis['width_cm']}c")

    def _aeqd_xy(self, lon: Sequence[float] | np.ndarray,
                 lat: Sequence[float] | np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """以 chunk 中心为原点的方位等距平面坐标（度）。东为正 x，北为正 y。"""
        d, az = dist_azimuth(self.chunk.center_lat, self.chunk.center_lon,
                             np.asarray(lat, dtype=float),
                             np.asarray(lon, dtype=float))
        rad = np.radians(az)
        return d * np.sin(rad), d * np.cos(rad)

    def _aeqd_to_ll(self, x: float, y: float) -> Tuple[float, float]:
        """方位等距平面坐标 → 地理经纬度。"""
        dist = math.hypot(x, y)
        if dist < 1e-9:
            return self.chunk.center_lon, self.chunk.center_lat
        az = math.degrees(math.atan2(x, y)) % 360.0
        # 球面终点公式（Vincenty 球面形式）
        lat1 = math.radians(self.chunk.center_lat)
        lon1 = math.radians(self.chunk.center_lon)
        ang = math.radians(dist)
        brg = math.radians(az)
        lat2 = math.asin(math.sin(lat1) * math.cos(ang)
                         + math.cos(lat1) * math.sin(ang) * math.cos(brg))
        lon2 = lon1 + math.atan2(
            math.sin(brg) * math.sin(ang) * math.cos(lat1),
            math.cos(ang) - math.sin(lat1) * math.sin(lat2))
        return math.degrees(lon2), math.degrees(lat2)

    def _place_labels(self, texts: Sequence[str]
                      ) -> List[Tuple[float, float]]:
        """
        在方位等距平面上为标注挑不打架的位置，再换回经纬度。

        半球投影下墨卡托坐标无意义；以 chunk 中心为原点的 AEQD 平面与
        General Perspective（高空俯视）的图面排布近似一致，够用。
        优先把标注放在「自中心向外」的方向，避免压到 chunk 内部密集区。
        """
        vis = self.config.visualization
        ex, ey = self._aeqd_xy([e.lon for e in self.selected],
                               [e.lat for e in self.selected])
        # 文本半宽/半高折合成度：半球上 1° ≈ 图面 width/180 cm
        deg_per_cm = 180.0 / float(vis['width_cm'])
        char_deg = vis['label_font_pt'] * 0.0176 * deg_per_cm
        half_h = char_deg * 1.1
        sym_r = vis['beachball_scale_cm'] * 0.75 * deg_per_cm

        boxes: List[Tuple[float, float, float, float]] = []
        out: List[Tuple[float, float]] = []

        for i, txt in enumerate(texts):
            half_w = max(1.2, 0.5 * len(txt) * char_deg)
            # 偏好方向：自中心指向该事件的方位角（向外）
            prefer = math.atan2(ex[i], ey[i])   # 注意 atan2(x,y) → 自北的方位
            best: Optional[Tuple[float, Tuple[float, float]]] = None
            for radius in vis['label_radii_deg']:
                for dang in range(0, 360, 20):
                    a = prefer + math.radians(dang)
                    cx = ex[i] + (sym_r + radius + half_w) * math.sin(a)
                    cy = ey[i] + (sym_r + radius + half_h) * math.cos(a)
                    pen = 0.0
                    # 压住任何一个震源球
                    for j in range(len(self.selected)):
                        pen += 40.0 * self._overlap(
                            cx, cy, half_w, half_h, ex[j], ey[j], sym_r, sym_r)
                    for bx, by, bw, bh in boxes:
                        pen += 60.0 * self._overlap(cx, cy, half_w, half_h,
                                                    bx, by, bw, bh)
                    pen += 1.2 * radius
                    # 偏离「向外」方向要付代价
                    pen += 0.8 * (1.0 - math.cos(a - prefer))
                    # 落在背面半球（距中心 > 90°）判死
                    if math.hypot(cx, cy) > 85.0:
                        pen += 500.0
                    if best is None or pen < best[0]:
                        best = (pen, (cx, cy))
            assert best is not None
            out.append(best[1])
            boxes.append((best[1][0], best[1][1], half_w, half_h))
        return out

    @staticmethod
    def _overlap(ax: float, ay: float, aw: float, ah: float,
                 bx: float, by: float, bw: float, bh: float) -> float:
        """两个轴对齐矩形的重叠面积。"""
        dx = min(ax + aw, bx + bw) - max(ax - aw, bx - bw)
        dy = min(ay + ah, by + bh) - max(ay - ah, by - bh)
        return max(0.0, dx) * max(0.0, dy)

    def _draw_basemap(self, fig: Any, projection: str) -> None:
        """General Perspective 半球底图，风格对齐 5_2_GCMT 地球图。"""
        vis = self.config.visualization
        fig.coast(
            region='g',
            projection=projection,
            frame=f"g{vis['grid_interval']}",
            land=vis['land_color'],
            water=vis['water_color'],
            borders=vis['border_pen'],
            shorelines='1/0.2p,gray50',
            resolution=vis['coast_resolution'],
        )
        title = vis['title'].format(n=len(self.selected))
        fig.text(position='TC', text=title,
                 font='12p,Helvetica-Bold,black',
                 offset='0c/0.55c', no_clip=True)

    def _draw_domain(self, fig: Any) -> None:
        """chunk 边框，以及震源必须落在其内的 12° 余量框。"""
        lat_o, lon_o = self.chunk.outline()
        fig.plot(x=lon_o, y=lat_o, pen='1.6p,black')
        inset = self.config.filters['chunk_margin_min']
        lat_i, lon_i = self.chunk.outline(inset=inset)
        fig.plot(x=lon_i, y=lat_i, pen='0.9p,black,4_2')
        fig.plot(x=[self.chunk.center_lon], y=[self.chunk.center_lat],
                 style='x0.45c', pen='1.3p,black')

    def _draw_stations(self, fig: Any) -> None:
        """台站。域外台站单独标出——SPECFEM 会静默丢弃它们。"""
        vis = self.config.visualization
        fig.plot(x=self.sta_lon[self.sta_usable], y=self.sta_lat[self.sta_usable],
                 style='i0.14c', fill=vis['station_color'], pen='0.12p,gray20',
                 transparency=15)
        if (~self.sta_usable).any():
            fig.plot(x=self.sta_lon[~self.sta_usable],
                     y=self.sta_lat[~self.sta_usable],
                     style='i0.14c', fill='white', pen='0.45p,firebrick')

    def _draw_events(self, fig: Any) -> None:
        """震源球（按质心深度着色）与避让后的标注。"""
        import pygmt

        vis = self.config.visualization
        pygmt.makecpt(cmap=vis['depth_cmap'], series=vis['depth_series'],
                      output=str(self._cpt_depth))

        rows = []
        for ev in self.selected:
            comps = np.array([ev.mrr, ev.mtt, ev.mpp, ev.mrt, ev.mrp, ev.mtp])
            expo = int(np.floor(np.log10(np.abs(comps).max())))
            m = comps / 10.0 ** expo
            rows.append({'longitude': ev.lon, 'latitude': ev.lat,
                         'depth': ev.depth,
                         'mrr': m[0], 'mtt': m[1], 'mff': m[2],
                         'mrt': m[3], 'mrf': m[4], 'mtf': m[5],
                         'exponent': expo})
        fig.meca(spec=pd.DataFrame(rows), convention='mt',
                 scale=f"{vis['beachball_scale_cm']}c+f0p",
                 cmap=str(self._cpt_depth),
                 extensionfill='white', pen='0.3p,gray20')

        texts = [f'{k}  Mw{ev.mw:.1f}  {ev.depth:.0f} km'
                 for k, ev in enumerate(self.selected, start=1)]
        pos = self._place_labels(texts)
        ex, ey = self._aeqd_xy([e.lon for e in self.selected],
                               [e.lat for e in self.selected])

        for i, (cx, cy) in enumerate(pos):
            if math.hypot(cx - ex[i], cy - ey[i]) < vis['leader_min_deg']:
                continue
            lon_a, lat_a = self._aeqd_to_ll(cx, cy)
            fig.plot(x=[self.selected[i].lon, lon_a],
                     y=[self.selected[i].lat, lat_a], pen='0.4p,gray30')
        for i, (cx, cy) in enumerate(pos):
            lon_a, lat_a = self._aeqd_to_ll(cx, cy)
            fig.text(x=lon_a, y=lat_a, text=texts[i],
                     font=f"{vis['label_font_pt']}p,Helvetica-Bold,black",
                     justify='CM', fill='white@20', pen='0.25p,gray55',
                     clearance='0.06c/0.05c+tO')

    def _draw_legend(self, fig: Any) -> None:
        """图例放在球体正下方，不压球面；深度信息已写在各事件标注上，不再单独给色标。"""
        vis = self.config.visualization
        n_in = int(self.sta_usable.sum())
        n_out = int((~self.sta_usable).sum())
        spec = self.output_dir / '.legend_events.txt'
        # 横向排布，贴在球下方
        lines = [
            'N 3',
            'S 0.25c c 0.26c white 0.3p,gray20 0.55c '
            'Test event (colour = depth)',
            f'S 0.25c i 0.14c {vis["station_color"]} 0.12p,gray20 0.55c '
            f'Station inside ({n_in})',
            f'S 0.25c i 0.14c white 0.45p,firebrick 0.55c '
            f'Station outside, dropped ({n_out})',
            'S 0.25c - 0.50c - 1.6p,black 0.55c SPECFEM chunk',
            f'S 0.25c - 0.50c - 0.9p,black,4_2 0.55c '
            f'Source-eligible ({self.config.filters["chunk_margin_min"]:.0f}'
            f'@. margin)',
            'S 0.25c x 0.35c - 1.3p,black 0.55c Chunk centre',
        ]
        spec.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        fig.legend(spec=str(spec),
                   position=f'JBC+w{0.92 * vis["width_cm"]:.2f}c+jTC+o0c/0.55c',
                   box='+gwhite@10+p0.6p,gray40')
        spec.unlink(missing_ok=True)

    def plot_map(self) -> Optional[Path]:
        """
        绘制震源与台站分布图。

        投影采用与 5_2_GCMT.plot_test_database_globe 相同的 General
        Perspective（G），中心在 SPECFEM chunk 中心。墨卡托会把 80°×80°
        的旋转 chunk 拉成菱形，掩盖它在球面上接近正方形这一事实。

        图面：半球底图 + chunk / 选源可用区 + 台站 + 震源球 + 避让标注。

        Returns:
            主图路径；未安装 pygmt 时返回 None
        """
        try:
            import pygmt
        except ImportError:
            self.logger.warning('未安装 pygmt，跳过绘图')
            return None

        vis = self.config.visualization
        projection = self._projection_string()
        self.logger.info(
            f'🌐 投影: General Perspective  '
            f'中心 ({self.chunk.center_lon:.1f}°E, {self.chunk.center_lat:.1f}°N)  '
            f'altitude={vis["altitude"]}'
        )

        with tempfile.TemporaryDirectory() as tmp:
            self._cpt_depth = Path(tmp) / 'depth.cpt'

            pygmt.config(**vis['gmt_config'])
            fig = pygmt.Figure()
            self._draw_basemap(fig, projection)
            self._draw_domain(fig)
            self._draw_stations(fig)
            self._draw_events(fig)
            self._draw_legend(fig)

            out: Optional[Path] = None
            for ext in vis['figure_format']:
                p = self.output_dir / f'3-1_test_events_map.{ext}'
                fig.savefig(str(p), dpi=vis['dpi'], crop=True)
                self.logger.info(f'🖼️  分布图 → {p}')
                out = out or p
        return out

    # ---------------- 编排 ----------------

    def run(self, ecef_script: Optional[Path] = None,
            make_plot: bool = True) -> List[Event]:
        self.load_catalog()
        self.compute_metrics()
        pool = self.filter_candidates()
        if not pool:
            raise RuntimeError('候选池为空，请放宽 EventSelectionConfig.filters')
        self.score(pool)
        self.logger.info(f'🎯 按 {len(self.config.provinces)} 个构造省选取:')
        self.select(pool)
        self.write_cmtsolutions(ecef_script)
        self.write_summary()
        if make_plot:
            self.plot_map()
        return self.selected


# ============================================================================
# 命令行
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(
        description='从 GCMT3D 目录中挑选 SPECFEM 正演对比测试用的代表性震源',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--catalog', type=Path,
                   default=here.parent / 'data/catalogs/CMT3D/cmt3d.txt',
                   help='GCMT3D 目录文件')
    p.add_argument('--par-file', type=Path, default=here / 'DATA/Par_file',
                   help='SPECFEM Par_file，用于读取 chunk 几何')
    p.add_argument('--stations', type=Path, default=here / 'DATA/STATIONS',
                   help='台站列表')
    p.add_argument('--output-dir', type=Path, default=here / 'events_test',
                   help='输出目录')
    p.add_argument('--n-events', type=int, default=10, help='目标震源数')
    p.add_argument('--mw-range', type=float, nargs=2, metavar=('LO', 'HI'),
                   default=None, help='震级范围，覆盖默认的 5.5 7.0')
    p.add_argument('--chunk-margin', type=float, default=None,
                   help='震源距 chunk 边界的最小度数')
    p.add_argument('--dist-range', type=float, nargs=2, metavar=('LO', 'HI'),
                   default=None, help='统计台站覆盖时的震中距窗口（度）')
    p.add_argument('--no-plot', action='store_true', help='不绘制分布图')
    p.add_argument('--no-ecef', action='store_true', help='不生成 ECEF 格式')
    return p


def main() -> None:
    """主函数"""
    print('🎯 EASTASIA-FWI 正演测试震源选取')
    print('=' * 60)

    args = build_parser().parse_args()
    try:
        cfg = EventSelectionConfig()
        cfg.n_target = args.n_events
        if args.mw_range:
            cfg.filters['mw_range'] = tuple(args.mw_range)
        if args.chunk_margin is not None:
            cfg.filters['chunk_margin_min'] = args.chunk_margin
        if args.dist_range:
            cfg.station['dist_range'] = tuple(args.dist_range)

        selector = TestEventSelector(
            catalog=args.catalog, par_file=args.par_file,
            stations=args.stations, output_dir=args.output_dir, config=cfg)

        ecef = None if args.no_ecef else Path(__file__).parent / 'change_cmt_to_ECEF.py'
        selector.run(ecef_script=ecef, make_plot=not args.no_plot)

        print(f'\n✅ 处理完成!')
        print(f'📁 输出目录: {selector.output_dir}')

    except KeyboardInterrupt:
        print('\n⚠️ 用户中断处理')
    except Exception as exc:
        print(f'\n❌ 处理失败: {exc}')
        import traceback
        traceback.print_exc()

    print('=' * 60)


if __name__ == '__main__':
    main()
