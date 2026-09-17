"""
EASTASIA-FWI 3_4 波形评估（数据空间：测量与评分）
================================================

功能描述:
- 30 事件 × 多模型：合成 vs 已去响应观测。测量、CSV、评分报告都在本文件内完成
- 出图见 5_Visualization/5_13_Waveform_assessment.py
- 只读 specfem/waveform/sac_30 的 SAC；不 import specfem 里的评估脚本
- specfem/waveform 下的旧脚本原样保留，互不影响

科学原理:
- Zhou et al. (2021, GJI)：dT、dlnA、NZCC、相对 L2
- BASE = 原作者 FWEA23 原生 GLL（model_updated），不是 1-D

作者: EASTASIA-FWI Team
日期: 2026-09-17
版本: v1.1
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import re
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import correlate
from obspy import Trace, UTCDateTime, read, read_inventory
from obspy.geodetics import gps2dist_azimuth, locations2degrees, degrees2kilometers
from obspy.signal.rotate import rotate_ne_rt
from obspy.taup import TauPyModel
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SAC_ROOT_DEFAULT = PROJECT_ROOT / 'specfem' / 'waveform'
sys.path.insert(0, str(PROJECT_ROOT))
VIS_DATA = PROJECT_ROOT / '5_Visualization' / 'data_EastAsia'

from config.base_config import BaseConfig  # noqa: E402

warnings.filterwarnings('ignore')

# ── 测量核（自 compare_waveforms 并入）──────────────────────────

SAC_UNDEF = -12345.0

OBSERVED_LABEL = 'Observed'

# 线型分层：
# - 链路图：Observed / AuthorRef / BASE 三色要能一眼分开（勿再用灰阶互盖）
# - 模型对比图：待检模型用蓝/橙/红；观测仍用黑色垫底
MODEL_STYLE: Dict[str, Dict[str, Any]] = {
    OBSERVED_LABEL: {'color': '#000000', 'lw': 1.0, 'zorder': 1, 'alpha': 1.0},  # 黑
    'AuthorRef':    {'color': '#d62728', 'lw': 1.4, 'zorder': 2, 'alpha': 0.9},  # 红
    'BASE':         {'color': '#1f77b4', 'lw': 1.1, 'zorder': 3, 'alpha': 0.95},  # 蓝
    # 旧 hybrid 命名（本地 sac/）
    'FWEA23_nc':    {'color': '#1f77b4', 'lw': 0.8, 'zorder': 4, 'alpha': 0.9},
    'EARA2024_hy':  {'color': '#ff7f0e', 'lw': 0.8, 'zorder': 5, 'alpha': 0.9},
    'SinoScope_hy': {'color': '#d62728', 'lw': 0.8, 'zorder': 6, 'alpha': 0.9},
    # 新 CRUST1.0+_c1 命名（run_all_test.sh / 曙光）
    'S362ANI_ref':  {'color': '#1f77b4', 'lw': 1.0, 'zorder': 3, 'alpha': 0.95},  # 原生 s362ani
    'S362ANI_c1':   {'color': '#9467bd', 'lw': 0.8, 'zorder': 4, 'alpha': 0.9},
    'FWEA23_c1':    {'color': '#2ca02c', 'lw': 0.8, 'zorder': 4, 'alpha': 0.9},
    'EARA2024_c1':  {'color': '#ff7f0e', 'lw': 0.8, 'zorder': 5, 'alpha': 0.9},
    'SinoScope_c1': {'color': '#d62728', 'lw': 0.8, 'zorder': 6, 'alpha': 0.9},
    'SinoScope_sem_c1': {'color': '#17becf', 'lw': 0.8, 'zorder': 7, 'alpha': 0.9},
}
FALLBACK_STYLE: Dict[str, Any] = {
    'color': '#8c564b', 'lw': 0.7, 'zorder': 2, 'alpha': 0.9,
}

# 待检模型别名：每组按优先级取第一个存在的目录（_c1 优先于旧 _hy/_nc）
# 排名时另加 BASE（见 resolve_rank_model_dirs），与观测一起比
TEST_MODEL_ALIASES: Tuple[Tuple[str, ...], ...] = (
    ('S362ANI_c1', 'S362ANI'),
    ('FWEA23_c1', 'FWEA23_nc', 'FWEA23_hy'),
    ('EARA2024_c1', 'EARA2024_hy'),
    ('SinoScope_c1', 'SinoScope_hy'),
    ('SinoScope_sem_c1',),
)


def resolve_test_model_dirs(sac_root: Path) -> Dict[str, Path]:
    """
    在 sac_root 下解析待检模型目录。

    兼容本地旧名（FWEA23_nc / EARA2024_hy / SinoScope_hy）与
    曙光 hybrid 新名（*_c1）。每组只取第一个存在的目录。
    """
    out: Dict[str, Path] = {}
    root = Path(sac_root)
    for aliases in TEST_MODEL_ALIASES:
        found = None
        for tag in aliases:
            d = root / tag
            if d.is_dir():
                found = (tag, d)
                break
        if found is None:
            print(f"⚠️  缺少目录，跳过: {' / '.join(str(root / a) for a in aliases)}")
            continue
        tag, d = found
        out[tag] = d
    return out


def resolve_rank_model_dirs(sac_root: Path) -> Dict[str, Path]:
    """
    参与「合成 vs 观测」排名的模型目录：BASE + 区域待检模型。

    Returns:
        有序 dict：标签 → SAC 目录
    """
    root = Path(sac_root)
    out: Dict[str, Path] = {}
    base = root / 'BASE'
    if base.is_dir():
        out['BASE'] = base
    else:
        print(f'⚠️  缺少 BASE，排名将不含该组: {base}')
    out.update(resolve_test_model_dirs(root))
    return out

# 观测通道优先级。BH/HH 是宽频带主力，SH/EH 短周期只在没别的可选时用。
# BH1/BH2 这类非正北取向的通道需要先做旋转才等价于 N/E，这里不收。
OBS_CHANNELS: Dict[str, List[str]] = {
    'Z': ['BHZ', 'HHZ', 'LHZ', 'MHZ', 'SHZ', 'EHZ'],
    'N': ['BHN', 'HHN', 'LHN', 'MHN', 'SHN', 'EHN'],
    'E': ['BHE', 'HHE', 'LHE', 'MHE', 'SHE', 'EHE'],
}


# ── 共同台站（正演 ∩ chujie ∩ 观测）─────────────────────────────────────────


@dataclass
class StationCoord:
    """台站坐标"""
    network: str
    station: str
    lat: float
    lon: float
    elev: float = 0.0


@dataclass
class CommonStations:
    """三方台站交集及其统计"""
    keys: Set[Tuple[str, str]]
    coords: Dict[Tuple[str, str], StationCoord]
    ours: Set[Tuple[str, str]]
    chujie: Set[Tuple[str, str]]
    observed: Set[Tuple[str, str]]

    @property
    def n_common(self) -> int:
        return len(self.keys)

    def sorted_keys(self) -> List[Tuple[str, str]]:
        """按台网、台站名排序。"""
        return sorted(self.keys)


def read_stations_file(path: Path) -> Dict[Tuple[str, str], StationCoord]:
    """
    读 SPECFEM STATIONS 文件。

    格式: NETWORK STATION LATITUDE LONGITUDE ELEVATION BURIAL

    Args:
        path: STATIONS 路径

    Returns:
        (network, station) -> StationCoord
    """
    out: Dict[Tuple[str, str], StationCoord] = {}
    path = Path(path)
    if not path.is_file():
        return out
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            net, sta = parts[0].upper(), parts[1].upper()
            try:
                lat, lon = float(parts[2]), float(parts[3])
                elev = float(parts[4]) if len(parts) >= 5 else 0.0
            except ValueError:
                continue
            out[(net, sta)] = StationCoord(net, sta, lat, lon, elev)
    return out


def station_keys_from_sac(directory: Path) -> Set[Tuple[str, str]]:
    """从合成 SAC 目录提取 (network, station)，跳过台网码缺失的道。"""
    return {(e.network, e.station) for e in build_entries(directory)
            if e.has_network}


def station_keys_from_mseed(directory: Path) -> Set[Tuple[str, str]]:
    """
    从观测 mseed 目录提取台站键。

    文件名约定: NET.STA.LOC.CHAN__....mseed
    """
    out: Set[Tuple[str, str]] = set()
    directory = Path(directory)
    if not directory.is_dir():
        return out
    for path in directory.glob('*.mseed'):
        parts = path.name.split('.')
        if len(parts) < 2:
            continue
        out.add((parts[0].upper(), parts[1].upper()))
    return out


def _parse_preprocessed_sac_name(name: str) -> Optional[Tuple[str, str, str]]:
    """
    解析 1_4 预处理 SAC 文件名 → (NET, STA, CHAN)。

    命名: YYYY.JJJ.H.M.S.NET.STA..CHAN.SAC
    """
    upper = name.upper()
    if not (upper.endswith('.SAC') or upper.endswith('.SAC.GZ')):
        return None
    stem = name[:-4] if upper.endswith('.SAC') else name
    if '..' not in stem:
        return None
    left, right = stem.rsplit('..', 1)
    chan = right.split('.')[0].upper()
    parts = left.split('.')
    if len(parts) < 2 or not chan:
        return None
    return parts[-2].upper(), parts[-1].upper(), chan


def station_keys_from_preprocessed_sac(directory: Path) -> Set[Tuple[str, str]]:
    """
    从 1_4 disp_data SAC 目录提取台站键。

    Args:
        directory: 含 *.SAC 的事件目录

    Returns:
        {(network, station), ...}
    """
    out: Set[Tuple[str, str]] = set()
    directory = Path(directory)
    if not directory.is_dir():
        return out
    for path in list(directory.glob('*.SAC')) + list(directory.glob('*.sac')):
        parsed = _parse_preprocessed_sac_name(path.name)
        if parsed is None:
            continue
        out.add((parsed[0], parsed[1]))
    return out


def station_keys_from_observed_dir(directory: Path) -> Set[Tuple[str, str]]:
    """观测目录台站键：优先预处理 SAC，否则 mseed。"""
    directory = Path(directory)
    keys = station_keys_from_preprocessed_sac(directory)
    if keys:
        return keys
    return station_keys_from_mseed(directory)


def build_common_stations(
    our_sac_dir: Path,
    chujie_dir: Optional[Path],
    observed_dir: Optional[Path],
    stations_file: Optional[Path] = None,
) -> CommonStations:
    """
    构建「正演 ∩ chujie ∩ 观测」共同台站集合。

    坐标优先取自 SPECFEM STATIONS；文件里没有的台站仍保留键，图上会跳过无坐标点。

    Args:
        our_sac_dir: 本项目正演波形目录（通常 sac/BASE）
        chujie_dir: 原作者参考波形目录
        observed_dir: 观测 mseed 目录
        stations_file: STATIONS 文件，用于取经纬度

    Returns:
        CommonStations
    """
    ours = station_keys_from_sac(Path(our_sac_dir))
    chujie = station_keys_from_sac(Path(chujie_dir)) if chujie_dir and \
        Path(chujie_dir).is_dir() else set()
    observed = station_keys_from_observed_dir(Path(observed_dir)) if observed_dir and \
        Path(observed_dir).is_dir() else set()

    keys = set(ours)
    if chujie:
        keys &= chujie
    if observed:
        keys &= observed

    coords_all = read_stations_file(stations_file) if stations_file else {}
    coords = {k: coords_all[k] for k in keys if k in coords_all}

    return CommonStations(
        keys=keys, coords=coords, ours=ours, chujie=chujie, observed=observed,
    )


def write_common_stations_csv(common: CommonStations, path: Path) -> Path:
    """
    写出共同台站清单 CSV。

    Args:
        common: 共同台站结果
        path: 输出路径

    Returns:
        写入的路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('network,station,latitude,longitude,elevation,'
                'in_ours,in_chujie,in_observed\n')
        # 一并列出三方并集中「几乎共同」的诊断行：只写交集键，避免 CSV 膨胀
        for net, sta in common.sorted_keys():
            c = common.coords.get((net, sta))
            lat = f'{c.lat:.4f}' if c else ''
            lon = f'{c.lon:.4f}' if c else ''
            elev = f'{c.elev:.1f}' if c else ''
            f.write(
                f'{net},{sta},{lat},{lon},{elev},'
                f'{int((net, sta) in common.ours)},'
                f'{int((net, sta) in common.chujie)},'
                f'{int((net, sta) in common.observed)}\n'
            )
    return path


def pair_in_allowlist(e_ref: 'TraceEntry', e_tst: 'TraceEntry',
                      allow: Set[Tuple[str, str]]) -> bool:
    """配对的一道是否属于共同台站集合。"""
    for e in (e_tst, e_ref):
        if e.has_network and (e.network, e.station) in allow:
            return True
    # 台网码缺失时退回台站名（原作者个别道）
    allow_sta = {s for _, s in allow}
    for e in (e_tst, e_ref):
        if not e.has_network and e.station in allow_sta:
            return True
    return False


# ── 配置 ────────────────────────────────────────────────────────────────────


class CompareConfig:
    """模块配置类"""

    def __init__(self):
        # 频段划分：岩石圈 / 上地幔（暂不跑 12-20s 地壳频段）
        # (标签, 最短周期 s, 最长周期 s)
        self.bands: List[Tuple[str, float, float]] = [
            ('20-40s', 20.0, 40.0),
            ('40-120s', 40.0, 120.0),
        ]

        self.filt = {
            'corners': 2,
            'zerophase': True,
            'taper_pct': 0.05,
        }

        # 互相关最大搜索时移，按频段最长周期的比例给，避免跨周期误配对
        self.max_lag_ratio = 0.5

        # 测量窗（相对首样点的秒数）。None 表示用整道。
        self.window: Optional[Tuple[float, float]] = None

        # 逐道 QC：能量为零、平直道、样点过少的道不参与统计
        self.qc = {
            'min_samples': 200,
            'flat_std_ratio': 1e-3,
        }

        self.obs = {
            'water_level': 60,       # 去响应的水位，压制低频发散
            # 观测/合成峰值比；绘图选台超出则整台丢弃
            # 下限略严：挡住 TM.NAYO 这类观测远大于合成的坏道
            'amp_ratio_limits': (0.25, 5.0),
            # 绘图选台：面波窗 NZCC 过低不画
            'plot_min_nzcc': 0.5,
            'plot_skip_bad_obs': True,
        }

        # Zhou et al. (2021) Fig.10：NZCC 超过该阈值计为「可预测」
        self.nzcc_predict_threshold = 0.7
        # Fig.9/10 高亮前 N 名（最好 = misfit 最低 / 可预测性最高）
        self.summary_highlight_top = 3

        self.visualization = {
            'dpi': 300,
            # 只出 jpg。这一批是排查用的中间产物，波形图动辄几十道叠在一起，
            # 矢量 PDF 会有几十万条路径，体积比 jpg 还大且打开很慢
            'figure_format': ['jpg', 'pdf'],
            'comp_colors': {
                'Z': '#1f77b4', 'R': '#2ca02c', 'T': '#d62728',
                'N': '#9467bd', 'E': '#8c564b',
            },
        }

        # 相位时窗：Zhou et al. (2021) — 体波 AK135 初至，面波 Table 2 群速度
        # Z/R → P, SV, Rayleigh；T → SH, Love
        # 体波：到时前 10 s、后 60 s；面波：群到时前 10 s、后 120 s
        # 长周期再按最长周期加宽，避免 40–120 s 把波形截断
        self.phase_windows = {
            'enabled': True,
            'taup_model': 'ak135',
            'body': {
                'P':  {'pre': 10.0, 'post': 60.0},
                'SV': {'pre': 10.0, 'post': 60.0},
                'SH': {'pre': 10.0, 'post': 60.0},
            },
            # 面波群速度 km/s（Zhou 2021 Table 2）
            'surface_velocity': {
                '20-40s':  {'Rayleigh': 3.3, 'Love': 3.9},
                '40-120s': {'Rayleigh': 3.5, 'Love': 4.2},
            },
            'surface_pre': 10.0,
            'surface_post': 120.0,
            'comp_phases': {
                'Z': ['P', 'SV', 'Rayleigh'],
                'R': ['P', 'SV', 'Rayleigh'],
                'T': ['SH', 'Love'],
            },
        }


@dataclass
class Measurement:
    """单道、单频段、单震相时窗的测量结果"""
    network: str
    station: str
    component: str
    band: str
    phase: str           # P / SV / SH / Rayleigh / Love / full
    dist_deg: float
    azimuth: float
    t_pred: float        # 预测到时 (s)，整道模式为 nan
    win_start: float
    win_end: float
    dt_shift: float      # 正值表示 test 相对 ref 走时偏晚
    ccmax: float
    nzcc: float
    dln_a: float         # ln(A_test / A_ref)
    rel_l2: float        # ||test - ref||^2 / ||ref||^2
    snr: float = float('nan')  # 观测窗 RMS / P 前噪声 RMS；算不出则为 NaN，评分拒窗


@dataclass
class PairResult:
    """一组「参考 vs 待检」的全部测量"""
    ref_label: str
    test_label: str
    measurements: List[Measurement] = field(default_factory=list)
    n_matched: int = 0
    n_ref_only: int = 0
    n_test_only: int = 0


# ── SAC 文件名解析 ──────────────────────────────────────────────────────────


def detect_sac_naming(directory: Path) -> str:
    """
    检测目录内 SAC 的命名格式，逻辑与 waveform_plotting.py 一致。

    Args:
        directory: 波形目录

    Returns:
        'chujie'  -> NET.STA[.LOC].CHAN.sem.sac
        'specfem' -> STA[.LOC].NET.CHAN.sem.sac
    """
    votes_specfem = votes_chujie = count = 0
    for p in directory.glob('*.sem.sac'):
        parts = p.name.split('.')
        if len(parts) < 5:
            continue
        field0 = parts[0]
        if field0 in ('-12345', ''):
            continue
        if len(field0) >= 3:
            votes_specfem += 1
        else:
            votes_chujie += 1
        count += 1
        if count >= 20:
            break
    return 'specfem' if votes_specfem > votes_chujie else 'chujie'


def parse_sac_filename(parts: List[str], fmt: str) -> Tuple[str, str, str, str]:
    """
    按格式解析文件名，返回 (network, station, location, channel)。

    两种长度都要支持：6 段含 location（可能为空串或 -12345），5 段不含。
    通道名恒为 .sem.sac 之前的最后一段，与格式无关。
    """
    n = len(parts)
    if fmt == 'specfem':
        if n >= 6:
            return parts[2], parts[0], parts[1], parts[3]
        return parts[1], parts[0], '', parts[2]
    if n >= 6:
        return parts[0], parts[1], parts[2], parts[3]
    return parts[0], parts[1], '', parts[2]


@dataclass
class TraceEntry:
    """一个波形文件的标识信息"""
    path: Path
    network: str
    station: str
    component: str
    channel: str

    @property
    def has_network(self) -> bool:
        """台网码是否可用。原作者的部分文件写成了 -12345 或空串。"""
        return self.network not in ('', '-12345')


def build_entries(directory: Path) -> List[TraceEntry]:
    """
    扫描目录，解析出全部波形文件的标识信息。

    分量只取通道名末位的 Z/N/E。SPECFEM 的通道前缀（MX / BX）由 DT 决定，
    两次正演 DT 不同时前缀就不同，只有末位是稳定的。

    Args:
        directory: 波形目录
    """
    fmt = detect_sac_naming(directory)
    out: List[TraceEntry] = []

    for path in sorted(directory.glob('*.sem.sac')):
        parts = path.name.split('.')
        if len(parts) < 5:
            continue
        network, station, _loc, channel = parse_sac_filename(parts, fmt)
        if not channel:
            continue
        comp = channel[-1].upper()
        if comp not in ('Z', 'N', 'E'):
            continue
        out.append(TraceEntry(path=path, network=network.upper(),
                              station=station.upper(), component=comp,
                              channel=channel))
    return out


def match_entries(ref: List[TraceEntry], test: List[TraceEntry]
                  ) -> Tuple[List[Tuple[TraceEntry, TraceEntry]], int, int]:
    """
    两个目录逐道配对，分两轮。

    先按 (台网, 台站, 分量) 严格配，跨台网重名的台站靠这一轮区分开——本项目实测
    768 道里有 15 个这样的重名。剩下的再按 (台站, 分量) 松配，用于原作者参考那种
    台网码缺失的情形；松配只在两边都唯一时才认，避免把不同台网的同名台站配错。

    Args:
        ref: 参考目录的条目
        test: 待检目录的条目

    Returns:
        (配对列表, 仅参考有的道数, 仅待检有的道数)
    """
    def strict(e: TraceEntry) -> Optional[Tuple[str, str, str]]:
        return (e.network, e.station, e.component) if e.has_network else None

    def loose(e: TraceEntry) -> Tuple[str, str]:
        return (e.station, e.component)

    pairs: List[Tuple[TraceEntry, TraceEntry]] = []
    used_ref, used_test = set(), set()

    idx_ref_strict: Dict[Tuple[str, str, str], TraceEntry] = {}
    for e in ref:
        k = strict(e)
        if k is not None and k not in idx_ref_strict:
            idx_ref_strict[k] = e

    for e in test:
        k = strict(e)
        if k is not None and k in idx_ref_strict:
            r = idx_ref_strict[k]
            pairs.append((r, e))
            used_ref.add(id(r))
            used_test.add(id(e))

    # 第二轮只处理没配上的，且要求台站分量在各自剩余集合里唯一
    rest_ref = [e for e in ref if id(e) not in used_ref]
    rest_test = [e for e in test if id(e) not in used_test]

    def unique_map(items: List[TraceEntry]) -> Dict[Tuple[str, str], TraceEntry]:
        counts: Dict[Tuple[str, str], int] = {}
        for e in items:
            counts[loose(e)] = counts.get(loose(e), 0) + 1
        return {loose(e): e for e in items if counts[loose(e)] == 1}

    m_ref, m_test = unique_map(rest_ref), unique_map(rest_test)
    for k, e in m_test.items():
        if k in m_ref:
            pairs.append((m_ref[k], e))
            used_ref.add(id(m_ref[k]))
            used_test.add(id(e))

    n_ref_only = sum(1 for e in ref if id(e) not in used_ref)
    n_test_only = sum(1 for e in test if id(e) not in used_test)
    return pairs, n_ref_only, n_test_only


class DirIndex:
    """
    一个波形目录的可查索引，绘图时按台站取道用。

    同时维护严格键与宽松键，查找规则与 match_entries 一致：先用台网区分重名台站，
    取不到再退回台站加分量。宽松键遇到重名时留空，宁可少画一道也不画错。
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        entries = build_entries(self.directory)
        self.strict: Dict[Tuple[str, str, str], TraceEntry] = {}
        counts: Dict[Tuple[str, str], int] = {}
        for e in entries:
            if e.has_network:
                self.strict.setdefault((e.network, e.station, e.component), e)
            k = (e.station, e.component)
            counts[k] = counts.get(k, 0) + 1
        self.loose: Dict[Tuple[str, str], TraceEntry] = {
            (e.station, e.component): e
            for e in entries if counts[(e.station, e.component)] == 1
        }

    def get(self, network: str, station: str,
            component: str) -> Optional[TraceEntry]:
        """按台网台站分量取道，取不到返回 None。"""
        hit = self.strict.get((network.upper(), station.upper(),
                               component.upper()))
        if hit is not None:
            return hit
        return self.loose.get((station.upper(), component.upper()))


# ── 波形读取与处理 ──────────────────────────────────────────────────────────


def ecef_to_geodetic(x: float, y: float, z: float
                     ) -> Tuple[float, float, float]:
    """
    WGS84 直角坐标转大地坐标。

    `USE_ECEF_CMTSOLUTION = .true.` 时 SPECFEM 把震源的 ECEF 分量原样写进 SAC 的
    evla / evlo / evdp，数值是米级，直接当经纬度用会越界。这里用 Bowring 闭式解
    换回来；地心纬度与大地纬度在中纬度能差近 0.2°，对震中距分档不可忽略，所以不
    用球近似。

    Args:
        x, y, z: ECEF 坐标 (m)

    Returns:
        (纬度 deg, 经度 deg, 高程 m)
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2 = f * (2.0 - f)
    ep2 = (a * a - b * b) / (b * b)

    p = np.hypot(x, y)
    if p == 0.0:
        lat = np.pi / 2 * np.sign(z)
        return float(np.degrees(lat)), 0.0, float(abs(z) - b)

    theta = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(theta) ** 3,
                     p - e2 * a * np.cos(theta) ** 3)
    lon = np.arctan2(y, x)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)


def normalize_event_header(hdr: Dict[str, float]) -> Dict[str, float]:
    """
    把头段里的震源位置统一成经纬度。

    判据是取值越界：纬度超出 ±90 或经度超出 ±180，就认定 evla/evlo/evdp 装的是
    ECEF 三分量。正常的经纬度不可能触发这个条件，所以不会误判。
    """
    try:
        la = float(hdr['evla'])
        lo = float(hdr['evlo'])
        dp = float(hdr['evdp'])
    except (KeyError, TypeError, ValueError):
        return hdr
    if any(np.isnan(v) for v in (la, lo, dp)):
        return hdr
    if abs(la) <= 90.0 and abs(lo) <= 180.0:
        return hdr

    lat, lon, alt = ecef_to_geodetic(la, lo, dp)
    out = dict(hdr)
    out['evla'], out['evlo'] = lat, lon
    out['evdp'] = -alt / 1000.0        # 高程转深度 km
    return out


@dataclass
class RawTrace:
    """预处理完成、尚未滤波的一道"""
    trace: object          # obspy Trace
    time: np.ndarray       # 相对发震时刻的秒数
    header: Dict[str, float]


def differentiate_raw(raw: RawTrace) -> Optional[RawTrace]:
    """
    将 RawTrace 从位移微分到速度（SPECFEM 合成默认位移）。

    Args:
        raw: 位移道

    Returns:
        速度道；失败返回 None
    """
    try:
        tr = raw.trace.copy()
        tr.differentiate()
    except Exception:
        return None
    data = np.asarray(tr.data, dtype=float)
    if data.size < 2 or not np.all(np.isfinite(data)):
        return None
    if len(data) != len(raw.time):
        # ObsPy 默认保持点数；若缩短则截齐时间轴
        n = min(len(data), len(raw.time))
        return RawTrace(
            trace=tr, time=np.asarray(raw.time[:n], dtype=float),
            header=dict(raw.header),
        )
    return RawTrace(trace=tr, time=raw.time, header=dict(raw.header))


def parse_origin_time(cmt_file: Path) -> Optional[UTCDateTime]:
    """
    从 CMTSOLUTION 首行的 PDE 记录取发震时刻。

    观测数据的时间轴是绝对 UTC，合成波形的时间轴以发震时刻为零点，两者叠图前必须
    用同一个零点对齐。本项目的 CMTSOLUTION 是 ECEF 格式，没有 `time shift:` 字段，
    质心时移写在 `t0(s):` 行，为零时质心时刻即 PDE 时刻。

    Args:
        cmt_file: CMTSOLUTION 路径

    Returns:
        UTCDateTime；解析失败返回 None
    """
    try:
        with open(cmt_file, 'r') as fh:
            first = fh.readline().split()
            rest = fh.read()
    except OSError:
        return None

    if len(first) < 7:
        return None
    try:
        y, mo, d, h, mi = (int(first[i]) for i in range(1, 6))
        sec = float(first[6])
        origin = UTCDateTime(y, mo, d, h, mi) + sec
    except (ValueError, TypeError):
        return None

    for line in rest.splitlines():
        s = line.strip()
        if s.startswith('t0(s):') or s.startswith('time shift:'):
            try:
                origin += float(s.split(':', 1)[1])
            except (ValueError, IndexError):
                pass
            break
    return origin


class ObservedIndex:
    """
    观测波形目录索引。

    支持两种布局：
    1. 预处理 SAC（1_4 → disp_data）：已是物理位移，无需 StationXML
    2. 原始 mseed + StationXML：现场去响应到位移（旧路径）

    mseed 文件名形如 NET.STA.LOC.CHAN__起__止.mseed；预处理 SAC 形如
    YYYY.JJJ.H.M.S.NET.STA..CHAN.SAC。
    """

    def __init__(self, waveform_dir: Path, response_dir: Optional[Path] = None,
                 preprocessed: Optional[bool] = None):
        self.waveform_dir = Path(waveform_dir)
        self.response_dir = Path(response_dir) if response_dir else None
        self._inv_cache: Dict[str, Any] = {}
        self.files: Dict[Tuple[str, str, str], Path] = {}

        sac_paths = sorted(
            list(self.waveform_dir.glob('*.SAC'))
            + list(self.waveform_dir.glob('*.sac'))
        )
        mseed_paths = sorted(self.waveform_dir.glob('*.mseed'))

        if preprocessed is None:
            preprocessed = bool(sac_paths) and not bool(mseed_paths)
            if sac_paths and mseed_paths:
                # 同目录两者皆有时优先预处理 SAC
                preprocessed = True
        self.is_preprocessed = bool(preprocessed)

        if self.is_preprocessed:
            for path in sac_paths:
                parsed = _parse_preprocessed_sac_name(path.name)
                if parsed is None:
                    continue
                net, sta, chan = parsed
                self.files[(net, sta, chan)] = path
        else:
            for path in mseed_paths:
                parts = path.name.split('.')
                if len(parts) < 4:
                    continue
                net, sta = parts[0].upper(), parts[1].upper()
                chan = parts[3].split('__')[0].upper()
                self.files[(net, sta, chan)] = path

        self.stations = {(n, s) for n, s, _ in self.files}

    def has(self, network: str, station: str, component: str) -> bool:
        """该台站分量是否有可用观测。"""
        return self.pick_channel(network, station, component) is not None

    def pick_channel(self, network: str, station: str,
                      component: str) -> Optional[Tuple[str, Path]]:
        net, sta = network.upper(), station.upper()
        for chan in OBS_CHANNELS.get(component.upper(), []):
            path = self.files.get((net, sta, chan))
            if path is not None:
                return chan, path
        return None

    def inventory(self, network: str, station: str):
        """按 NET.STA.xml 取响应，取不到返回 None。逐台站缓存，避免重复解析。"""
        if self.is_preprocessed or self.response_dir is None:
            return None
        key = f"{network.upper()}.{station.upper()}"
        if key in self._inv_cache:
            return self._inv_cache[key]
        path = self.response_dir / f"{key}.xml"
        inv = None
        if path.is_file():
            try:
                inv = read_inventory(str(path))
            except Exception:
                inv = None
        self._inv_cache[key] = inv
        return inv


def read_preprocessed_observed(
    idx: ObservedIndex, network: str, station: str,
    component: str, origin: UTCDateTime,
    cfg: CompareConfig,
) -> Optional[RawTrace]:
    """
    读入 1_4 预处理 DISP SAC，时间轴对齐到发震时刻。

    不再去仪器响应；头段 O 可能缺失，一律用 CMT origin 与 starttime 对齐。

    Args:
        idx: 预处理观测索引
        network, station, component: 台站标识
        origin: 发震时刻
        cfg: 模块配置

    Returns:
        RawTrace（位移）；缺数据或未过 QC 时返回 None
    """
    hit = idx.pick_channel(network, station, component)
    if hit is None:
        return None
    _chan, path = hit

    try:
        tr = read(str(path))[0]
    except Exception:
        return None

    data = np.asarray(tr.data, dtype=float)
    if data.size < cfg.qc['min_samples'] or not np.all(np.isfinite(data)):
        return None
    if np.max(np.abs(data)) <= 0:
        return None
    tr.data = data

    try:
        tr.detrend('demean')
        tr.detrend('linear')
        tr.taper(max_percentage=cfg.filt['taper_pct'], type='cosine')
    except Exception:
        return None

    d = np.asarray(tr.data, dtype=float)
    if not np.all(np.isfinite(d)) or np.max(np.abs(d)) <= 0:
        return None

    sac = getattr(tr.stats, 'sac', None)
    hdr: Dict[str, float] = {}
    for k in ('b', 'stla', 'stlo', 'evla', 'evlo', 'evdp', 'gcarc', 'az', 'baz'):
        v = getattr(sac, k, SAC_UNDEF) if sac is not None else SAC_UNDEF
        hdr[k] = float(v) if v is not None and float(v) != SAC_UNDEF else np.nan
    hdr = normalize_event_header(hdr)

    t0 = float(tr.stats.starttime - origin)
    t = t0 + np.arange(tr.stats.npts) * float(tr.stats.delta)
    return RawTrace(trace=tr, time=t, header=hdr)


def read_observed(idx: ObservedIndex, network: str, station: str,
                  component: str, origin: UTCDateTime,
                  cfg: CompareConfig) -> Optional[RawTrace]:
    """
    读入一道观测，时间轴换算到相对发震时刻。

    - 预处理 SAC（is_preprocessed）：直接读位移，不去响应
    - 原始 mseed：去仪器响应到位移；缺响应则放弃该道

    Args:
        idx: 观测索引
        network, station, component: 台站标识
        origin: 发震时刻
        cfg: 模块配置

    Returns:
        RawTrace；缺数据、缺响应或未过 QC 时返回 None
    """
    if getattr(idx, 'is_preprocessed', False):
        return read_preprocessed_observed(
            idx, network, station, component, origin, cfg)

    hit = idx.pick_channel(network, station, component)
    if hit is None:
        return None
    _chan, path = hit

    inv = idx.inventory(network, station)
    if inv is None:
        return None

    try:
        tr = read(str(path))[0]
    except Exception:
        return None

    data = np.asarray(tr.data, dtype=float)
    if data.size < cfg.qc['min_samples'] or not np.all(np.isfinite(data)):
        return None
    tr.data = data

    try:
        tr.detrend('demean')
        tr.detrend('linear')
        tr.taper(max_percentage=cfg.filt['taper_pct'], type='cosine')
        tr.attach_response(inv)
        # evalresp 的 sensitivity/FIR 提示刷屏且不影响继续执行，压掉 stderr
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stderr(devnull):
                tr.remove_response(output='DISP',
                                   water_level=cfg.obs['water_level'])
    except Exception:
        return None

    d = np.asarray(tr.data, dtype=float)
    if not np.all(np.isfinite(d)) or np.max(np.abs(d)) <= 0:
        return None

    t0 = float(tr.stats.starttime - origin)
    t = t0 + np.arange(tr.stats.npts) * float(tr.stats.delta)
    return RawTrace(trace=tr, time=t, header={})


def read_raw(
    path: Path,
    cfg: CompareConfig,
    origin: Optional[UTCDateTime] = None,
) -> Optional[RawTrace]:
    """
    读入一道 SAC，去均值去线性趋势加余弦窗，但不滤波。

    滤波前的处理与频段无关，所以只做一次；三个频段各自从这一份拷贝出去滤。

    Args:
        path: SAC 文件
        cfg: 模块配置
        origin: 若给定，时间轴用 starttime−origin（与观测一致）；
            否则退回 SAC 头段 b + n·Δt

    Returns:
        RawTrace；读取失败或未通过基本 QC 时返回 None
    """
    try:
        tr = read(str(path))[0]
    except Exception as exc:
        print(f"   ❌ 读取失败 {path.name}: {exc}")
        return None

    data = np.asarray(tr.data, dtype=float)
    if data.size < cfg.qc['min_samples'] or not np.all(np.isfinite(data)):
        return None
    if np.max(np.abs(data)) <= 0:
        return None

    tr.data = data
    tr.detrend('demean')
    tr.detrend('linear')
    tr.taper(max_percentage=cfg.filt['taper_pct'], type='cosine')

    sac = getattr(tr.stats, 'sac', None)
    hdr: Dict[str, float] = {}
    for k in ('b', 'stla', 'stlo', 'evla', 'evlo', 'evdp', 'gcarc', 'az', 'baz'):
        v = getattr(sac, k, SAC_UNDEF) if sac is not None else SAC_UNDEF
        hdr[k] = float(v) if v is not None and float(v) != SAC_UNDEF else np.nan

    hdr = normalize_event_header(hdr)
    # 缺 baz 时用震源—台站几何补算（旋转 NE→RT 必需）
    if np.isnan(hdr.get('baz', np.nan)):
        try:
            if all(np.isfinite(hdr.get(k, np.nan))
                   for k in ('evla', 'evlo', 'stla', 'stlo')):
                _, az, baz = gps2dist_azimuth(
                    hdr['evla'], hdr['evlo'], hdr['stla'], hdr['stlo'])
                if np.isnan(hdr.get('az', np.nan)):
                    hdr['az'] = float(az)
                hdr['baz'] = float(baz)
        except (ValueError, TypeError):
            pass

    if origin is not None:
        t = np.asarray(tr.times(reftime=origin), dtype=float)
    else:
        b = 0.0 if np.isnan(hdr['b']) else hdr['b']
        t = b + np.arange(tr.stats.npts) * float(tr.stats.delta)
    return RawTrace(trace=tr, time=t, header=hdr)


def filter_band(tr, tmin_period: float, tmax_period: float,
                cfg: CompareConfig) -> np.ndarray:
    """
    从已预处理的道拷贝一份做带通滤波，返回数据数组。

    Args:
        tr: read_raw 返回的 Trace
        tmin_period: 频段最短周期 (s)
        tmax_period: 频段最长周期 (s)
        cfg: 模块配置
    """
    w = tr.copy()
    w.filter('bandpass',
             freqmin=1.0 / tmax_period,
             freqmax=1.0 / tmin_period,
             corners=cfg.filt['corners'],
             zerophase=cfg.filt['zerophase'])
    return np.asarray(w.data, dtype=float)


def to_common_axis(t_ref: np.ndarray, d_ref: np.ndarray,
                   t_tst: np.ndarray, d_tst: np.ndarray
                   ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    把两道重采样到公共时间轴上。

    四组自己之间 DT 完全一致，这一步是恒等变换；但与原作者参考比对时两边 DT 不同
    （0.1016 vs 0.11），不重采样就没法逐点相减。步长取两者中较粗的一个，避免插值
    出实际不存在的高频。滤波已在此之前完成，所以线性插值不会引入频段内失真。

    Returns:
        (公共时间轴, ref 数据, test 数据)；重叠段过短时返回 None
    """
    dt = max(t_ref[1] - t_ref[0], t_tst[1] - t_tst[0])
    t0 = max(t_ref[0], t_tst[0])
    t1 = min(t_ref[-1], t_tst[-1])
    if t1 - t0 < 50.0:
        return None
    t = np.arange(t0, t1, dt)
    return t, np.interp(t, t_ref, d_ref), np.interp(t, t_tst, d_tst)


def measure_pair(ref: np.ndarray, tst: np.ndarray, dt: float,
                 max_lag_s: float) -> Optional[Dict[str, float]]:
    """
    计算一对波形的四项指标。

    走时差用互相关峰值位置，并做抛物线插值细化到亚采样点。符号约定：
    dt_shift > 0 表示 test 相对 ref 走时偏晚（波形滞后），与「模型偏慢」对应。

    Args:
        ref: 参考波形
        tst: 待检波形
        dt: 采样间隔 (s)
        max_lag_s: 互相关搜索的最大时移 (s)

    Returns:
        dict，含 dt_shift / ccmax / nzcc / dln_a / rel_l2；无效时返回 None
    """
    n = len(ref)
    e_ref = float(np.dot(ref, ref))
    e_tst = float(np.dot(tst, tst))
    if e_ref <= 0 or e_tst <= 0:
        return None

    norm = np.sqrt(e_ref * e_tst)
    cc = correlate(tst, ref, mode='full', method='fft') / norm

    zero = n - 1
    max_lag = min(int(round(max_lag_s / dt)), n - 1)
    lo, hi = zero - max_lag, zero + max_lag + 1
    seg = cc[lo:hi]
    i = int(np.argmax(seg))
    idx = lo + i

    # 抛物线插值。峰值落在搜索窗边界时跳过，此时本就不该信任这个测量。
    sub = 0.0
    if 0 < i < len(seg) - 1:
        y0, y1, y2 = seg[i - 1], seg[i], seg[i + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-30:
            sub = 0.5 * (y0 - y2) / denom

    # correlate(tst, ref) 在 lag = L 处取峰，意味着 tst[n+L] ≈ ref[n]，
    # 即 tst 比 ref 提前 L 个样点。走时差取相反数才符合「滞后为正」的约定。
    lag_samples = (idx - zero) + sub
    dt_shift = -lag_samples * dt

    rms_ref = np.sqrt(e_ref / n)
    rms_tst = np.sqrt(e_tst / n)

    return {
        'dt_shift': float(dt_shift),
        'ccmax': float(seg[i]),
        'nzcc': float(cc[zero]),
        'dln_a': float(np.log(rms_tst / rms_ref)),
        'rel_l2': float(np.sum((tst - ref) ** 2) / e_ref),
    }


# ── 相位时窗 ─────────────────────────────────────────────────────────────────


_TAUP_CACHE: Dict[str, TauPyModel] = {}


def get_taup_model(name: str) -> TauPyModel:
    """缓存 TauPyModel，避免每个台站重复读模型文件。"""
    if name not in _TAUP_CACHE:
        _TAUP_CACHE[name] = TauPyModel(model=name)
    return _TAUP_CACHE[name]


def _normalize_source_depth(depth_km: float) -> float:
    """规范化震源深度 (km)；异常大值按米→千米换算。"""
    if not np.isfinite(depth_km) or depth_km < 0:
        return 0.0
    depth = float(depth_km)
    # TauP 震源深度上限约 800 km；异常大值多半是头段单位未归一
    if depth > 800.0:
        depth = depth / 1000.0
    return depth


def taup_arrivals(dist_deg: float, depth_km: float, model_name: str = 'ak135'
                  ) -> Dict[str, float]:
    """
    用 TauP 取 P、S 初至（相对发震时刻，秒）。

    每个震中距取上出 p/s 与下行 P/S 中最早的一支。深源近台初至是 p/s，
    只查大写 P/S 会整段没有体波窗。

    Returns:
        {'P': t_p, 'S': t_s}；算不出的键不出现
    """
    out: Dict[str, float] = {}
    if not np.isfinite(dist_deg) or dist_deg <= 0:
        return out
    depth = _normalize_source_depth(depth_km)
    try:
        model = get_taup_model(model_name)
    except Exception:
        return out
    for key, names in (('P', ('p', 'P')), ('S', ('s', 'S'))):
        times: List[float] = []
        for name in names:
            try:
                arr = model.get_travel_times(
                    source_depth_in_km=depth,
                    distance_in_degree=float(dist_deg),
                    phase_list=[name],
                )
            except Exception:
                continue
            if arr:
                times.append(float(arr[0].time))
        if times:
            out[key] = min(times)
    return out


def travel_time_curve(
    phase: str,
    dist_min: float,
    dist_max: float,
    depth_km: float,
    model_name: str = 'ak135',
    n_samples: int = 80,
    time_limit: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在震中距区间上采样单震相走时曲线（对齐 waveform_plotting-all 画法）。

    Returns:
        (times_s, distances_deg)；点数不足时返回空数组
    """
    if not np.isfinite(dist_min) or not np.isfinite(dist_max):
        return np.array([]), np.array([])
    if dist_max <= dist_min:
        dist_max = dist_min + 0.5
    depth = _normalize_source_depth(depth_km)
    model = get_taup_model(model_name)
    dists = np.linspace(float(dist_min), float(dist_max), max(int(n_samples), 5))
    times: List[float] = []
    keep: List[float] = []
    for dist in dists:
        try:
            arr = model.get_travel_times(
                source_depth_in_km=depth,
                distance_in_degree=float(dist),
                phase_list=[phase],
            )
            if not arr:
                continue
            t = float(arr[0].time)
            if time_limit is not None and t > float(time_limit):
                continue
            times.append(t)
            keep.append(float(dist))
        except Exception:
            continue
    if len(times) < 3:
        return np.array([]), np.array([])
    return np.asarray(times), np.asarray(keep)


def phase_time_window(
    phase: str,
    band: str,
    tmin_period: float,
    tmax_period: float,
    dist_deg: float,
    arrivals: Dict[str, float],
    cfg: Dict[str, Any],
) -> Optional[Tuple[float, float, float]]:
    """
    计算震相测量时窗。

    Args:
        phase: P / SV / SH / Rayleigh / Love
        band: 频段标签
        tmin_period, tmax_period: 频段周期范围
        dist_deg: 震中距
        arrivals: TauP 结果
        cfg: config.phase_windows

    Returns:
        (win_start, win_end, t_pred)；无法构造时返回 None
    """
    if phase in ('P', 'SV', 'SH'):
        key = 'P' if phase == 'P' else 'S'
        t_pred = arrivals.get(key)
        if t_pred is None or not np.isfinite(t_pred):
            return None
        body = cfg['body'][phase]
        # Zhou: 到时前 10 s、后 60 s；长周期再按 Tmax 加宽
        pre = max(float(body['pre']), 0.5 * tmin_period)
        post = max(float(body['post']), 1.0 * tmax_period)
        return t_pred - pre, t_pred + post, t_pred

    if phase in ('Rayleigh', 'Love'):
        vel_tab = cfg['surface_velocity'].get(band)
        if not vel_tab or phase not in vel_tab:
            return None
        v = float(vel_tab[phase])
        if v <= 0 or not np.isfinite(dist_deg):
            return None
        dist_km = float(degrees2kilometers(dist_deg))
        t_pred = dist_km / v
        # Zhou: 群到时前 10 s、后 120 s；长周期至少覆盖 2 个 Tmax
        pre = max(float(cfg.get('surface_pre', 10.0)), 0.5 * tmax_period)
        post = max(float(cfg.get('surface_post', 120.0)), 1.0 * tmax_period)
        w0 = max(0.0, t_pred - pre)
        w1 = t_pred + post
        min_dur = 2.0 * float(tmax_period)
        if np.isfinite(min_dur) and min_dur > 0.0 and (w1 - w0) < min_dur:
            w0 = max(0.0, t_pred - float(tmax_period))
            w1 = t_pred + float(tmax_period)
        return w0, w1, t_pred

    return None


def obs_window_snr(
    t: np.ndarray,
    data: np.ndarray,
    signal_mask: np.ndarray,
    t_first: float,
    tmax_period: float,
    min_samples: int = 50,
    pad_floor: float = 5.0,
) -> float:
    """
    观测窗 RMS / P 前噪声窗 RMS（同一滤波道）。

    噪声窗取 t < t_P − max(5 s, 0.5 Tmax)，避开零相位滤波把 P 能量抹到初至前。
    噪声段不够长则返回 NaN，由接受准则拒窗。

    Args:
        t: 公共时间轴（相对发震时刻，秒）
        data: 与 t 等长的观测振幅
        signal_mask: 信号窗布尔掩膜
        t_first: P/p 初至（秒）
        tmax_period: 频段最长周期（秒）
        min_samples: 噪声窗最少点数
        pad_floor: 初至前最少留白（秒）

    Returns:
        信噪比；无法计算时为 NaN
    """
    if not np.isfinite(t_first):
        return np.nan
    pad = max(float(pad_floor), 0.5 * float(tmax_period))
    nmask = t < (float(t_first) - pad)
    if int(nmask.sum()) < int(min_samples):
        return np.nan
    nrms = float(np.sqrt(np.mean(np.asarray(data[nmask], dtype=float) ** 2)))
    if nrms <= 0.0:
        return np.nan
    srms = float(np.sqrt(np.mean(np.asarray(data[signal_mask], dtype=float) ** 2)))
    if srms <= 0.0:
        return np.nan
    return srms / nrms


def align_two_traces(
    t_a: np.ndarray, d_a: np.ndarray,
    t_b: np.ndarray, d_b: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """把两道对齐到公共时间轴（用于旋转前对齐 N/E）。"""
    return to_common_axis(t_a, d_a, t_b, d_b)


def build_zrt_components(
    raws: Dict[str, RawTrace],
) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    从 Z/N/E 原始道构造 Z/R/T。

    Returns:
        {'Z': (t, d), 'R': (t, d), 'T': (t, d)}；缺 baz 或水平分量时 R/T 可缺
    """
    if 'Z' not in raws:
        return None
    z = raws['Z']
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        'Z': (z.time, np.asarray(z.trace.data, dtype=float)),
    }
    if 'N' not in raws or 'E' not in raws:
        return out

    baz = z.header.get('baz', np.nan)
    if not np.isfinite(baz):
        for comp in ('N', 'E'):
            baz = raws[comp].header.get('baz', np.nan)
            if np.isfinite(baz):
                break
    if not np.isfinite(baz):
        return out

    aligned = align_two_traces(
        raws['N'].time, np.asarray(raws['N'].trace.data, dtype=float),
        raws['E'].time, np.asarray(raws['E'].trace.data, dtype=float),
    )
    if aligned is None:
        return out
    t, n_d, e_d = aligned
    r_d, t_d = rotate_ne_rt(n_d, e_d, float(baz))
    out['R'] = (t, np.asarray(r_d, dtype=float))
    out['T'] = (t, np.asarray(t_d, dtype=float))
    return out


# ── 多事件评分（自 multi_event_assessment 并入）────────────────

# 默认待评模型（可按需用 CLI 覆盖）
# BASE             : 原作者 FWEA23 原生 GLL（model_updated），地壳+地幔都不替换
#                    → 目前可得的最好配置，不是 1-D / s362ani
# S362ANI_c1       : CRUST1.0 + s362ani，共用地壳下的空对照
# FWEA23_c1        : CRUST1.0 + FWEA23 地幔（NC→GLL）
# EARA2024_c1      : CRUST1.0 + EARA2024 地幔
# SinoScope_c1     : 公开 1° HDF5 → mesh0 GLL
# SinoScope_sem_c1 : 作者 Salvus mesh.h5 → mesh0 GLL（原生 SEM 点云）
DEFAULT_MODELS: Tuple[str, ...] = (
    'BASE', 'S362ANI_c1', 'FWEA23_c1', 'EARA2024_c1',
    'SinoScope_c1', 'SinoScope_sem_c1',
)

MODEL_COLORS: Dict[str, str] = {
    'BASE': '#1f77b4',
    'S362ANI_c1': '#9467bd',
    'FWEA23_c1': '#2ca02c',
    'EARA2024_c1': '#ff7f0e',
    'SinoScope_c1': '#d62728',
    'SinoScope_sem_c1': '#17becf',
}


# ── 事件与路径 ──────────────────────────────────────────────────────────────


@dataclass
class EventSpec:
    """单个测试事件的路径与元数据"""
    event_id: str
    origin: UTCDateTime
    cmt_file: Path
    lat: float = np.nan
    lon: float = np.nan
    depth_km: float = np.nan
    mw: float = np.nan
    obs_dir: Optional[Path] = None
    resp_dir: Optional[Path] = None

    @property
    def time_tag(self) -> str:
        """观测目录常用时间戳，如 20060430.00.43"""
        o = self.origin
        return (f'{o.year:04d}{o.month:02d}{o.day:02d}.'
                f'{o.hour:02d}.{o.minute:02d}')

    @property
    def ready(self) -> bool:
        return self.obs_dir is not None and self.obs_dir.is_dir()


@dataclass
class EventModelResult:
    """单事件、单模型相对观测的全部测量"""
    event_id: str
    model: str
    measurements: List[Measurement] = field(default_factory=list)
    n_stations: int = 0
    n_skip: int = 0


def load_event_list(
    events_list: Path,
    events_csv: Optional[Path],
    cmt_dir: Path,
) -> List[EventSpec]:
    """
    从 events.list + selected_test_events.csv + CMTSOLUTION 构建事件表。

    Args:
        events_list: 每行一个事件名
        events_csv: 含 origin_utc / lat / lon / depth / Mw 的 CSV（可选）
        cmt_dir: CMTSOLUTION_<EVENT> 所在目录
    """
    meta: Dict[str, Dict[str, Any]] = {}
    if events_csv and Path(events_csv).is_file():
        with open(events_csv, 'r', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                eid = (row.get('event') or '').strip()
                if not eid:
                    continue
                try:
                    meta[eid] = {
                        'lat': float(row.get('latitude', 'nan')),
                        'lon': float(row.get('longitude', 'nan')),
                        'depth_km': float(row.get('depth_km', 'nan')),
                        'mw': float(row.get('Mw', row.get('mw', 'nan'))),
                        'origin_utc': (row.get('origin_utc') or '').strip(),
                    }
                except (TypeError, ValueError):
                    continue

    out: List[EventSpec] = []
    with open(events_list, 'r', encoding='utf-8') as fh:
        for line in fh:
            eid = line.strip()
            if not eid or eid.startswith('#'):
                continue
            cmt = Path(cmt_dir) / f'CMTSOLUTION_{eid}'
            if not cmt.is_file():
                alt = Path(cmt_dir) / eid
                cmt = alt if alt.is_file() else cmt
            origin = parse_origin_time(cmt) if cmt.is_file() else None
            m = meta.get(eid, {})
            if origin is None and m.get('origin_utc'):
                try:
                    origin = UTCDateTime(str(m['origin_utc']))
                except Exception:
                    origin = None
            if origin is None:
                print(f'⚠️  跳过事件 {eid}：无法解析发震时刻（缺 CMT）')
                continue
            out.append(EventSpec(
                event_id=eid,
                origin=origin,
                cmt_file=cmt,
                lat=float(m.get('lat', np.nan)),
                lon=float(m.get('lon', np.nan)),
                depth_km=float(m.get('depth_km', np.nan)),
                mw=float(m.get('mw', np.nan)),
            ))
    return out


def _dir_has_sac(directory: Path) -> bool:
    """目录是否含预处理 SAC。"""
    return any(directory.glob('*.SAC')) or any(directory.glob('*.sac'))


def _dir_has_mseed(directory: Path) -> bool:
    """目录是否含原始 mseed。"""
    return any(directory.glob('*.mseed'))


def resolve_obs_dirs(
    ev: EventSpec,
    obs_root: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    解析观测波形目录（及可选响应目录）。

    优先 1_4 disp_data 风格的预处理 SAC；找不到再回退 mseed+StationXML。

    Returns:
        (waveforms_dir, responses_dir)；SAC 模式下 responses_dir 为 None；
        找不到时 (None, None)
    """
    root = Path(obs_root)
    tag = ev.time_tag

    # 预处理 DISP SAC（无需响应）
    sac_candidates = [
        root / tag,
        root / ev.event_id,
        root / 'disp_data' / tag,
        PROJECT_ROOT / 'data' / 'events' / 'disp_data' / tag,
    ]
    for wdir in sac_candidates:
        if wdir.is_dir() and _dir_has_sac(wdir):
            return wdir, None

    # 旧布局：mseed + StationXML
    mseed_candidates: List[Tuple[Path, Optional[Path]]] = [
        (root / ev.event_id / 'waveforms', root / ev.event_id / 'responses'),
        (root / ev.event_id, root / ev.event_id / 'responses'),
        (root / 'waveforms' / tag, root / 'responses' / tag),
        (root / tag, root / 'responses' / tag),
        (root / tag / 'waveforms', root / tag / 'responses'),
    ]
    for wdir, rdir in mseed_candidates:
        if not wdir.is_dir() or not _dir_has_mseed(wdir):
            continue
        resp = rdir if rdir is not None and rdir.is_dir() else None
        if resp is None and any(wdir.glob('*.xml')):
            resp = wdir
        return wdir, resp
    return None, None


def syn_event_dir(sac_root: Path, model: str, event_id: str) -> Path:
    """合成波形目录 sac/<TAG>/<EVENT>/"""
    return Path(sac_root) / model / event_id


def list_available_models(sac_root: Path, models: Sequence[str],
                          event_id: str) -> List[str]:
    """某事件下实际存在且含 SAC 的模型标签。"""
    out: List[str] = []
    for m in models:
        d = syn_event_dir(sac_root, m, event_id)
        if d.is_dir() and any(d.glob('*.sem.sac')):
            out.append(m)
    return out


def parse_harvard_mt(cmt_file: Path) -> Optional[Dict[str, float]]:
    """
    从 Harvard 格式 CMTSOLUTION 读取矩张量（Mrr/Mtt/...）。

    Args:
        cmt_file: CMTSOLUTION 路径

    Returns:
        含 mrr/mtt/mpp/mrt/mrp/mtp 的字典；失败返回 None
    """
    if not Path(cmt_file).is_file():
        return None
    keys = {
        'mrr': 'mrr', 'mtt': 'mtt', 'mpp': 'mpp',
        'mrt': 'mrt', 'mrp': 'mrp', 'mtp': 'mtp',
    }
    vals: Dict[str, float] = {}
    for line in Path(cmt_file).read_text(encoding='utf-8',
                                         errors='replace').splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        tag = parts[0].rstrip(':').lower()
        if tag in keys:
            try:
                vals[keys[tag]] = float(parts[1])
            except ValueError:
                continue
    if len(vals) < 6:
        return None
    return vals


class MultiEventConfig(CompareConfig):
    """多事件评估配置（继承频段/时窗/QC，补充接受准则与加权）"""

    def __init__(self) -> None:
        super().__init__()
        # 观测峰值与合成差太多时只比形态（同 compare_waveforms）
        self.obs['amp_ratio_limits'] = (0.25, 5.0)
        self.obs['plot_min_nzcc'] = 0.5
        self.obs['plot_skip_bad_obs'] = True
        self.nzcc_predict_threshold = 0.7
        self.summary_highlight_top = 3

        # 测量接受准则（Zhou et al. 2021: 窗口 SNR>4 且 CC>0.7 才可信）
        # dT / dlnA 统计仅收同时满足以下条件的窗口：
        #   ccmax >= cc_min            互相关峰值（防 cycle-skipping）
        #   |dT| < edge_lag_frac×max_lag  峰未贴搜索边界（贴边=测量饱和）
        #   snr >= snr_min             观测窗信噪比（算不出 SNR 则拒窗）
        self.accept = {
            'cc_min': 0.7,
            'snr_min': 4.0,
            'edge_lag_frac': 0.95,
        }

        # W_r 地理加权：台站按 (方位角扇区 × 震中距带) 分格，
        # 权重 = 1/格内台站数，抑制日本/台湾等密集台阵主导统计
        self.weighting = {
            'az_bins': 12,
            'dist_edges': (0.0, 10.0, 20.0, 30.0, 180.0),
        }


class MultiEventAssessment:
    """评分 / CSV / 接受准则（自 multi_event_assessment 并入）。"""

    """多事件 × 多模型：合成 vs 观测 评估"""

    def _dist_deg(hdr: Dict[str, float]) -> float:
        if np.isfinite(hdr.get('gcarc', np.nan)):
            return float(hdr['gcarc'])
        for k in ('stla', 'stlo', 'evla', 'evlo'):
            if not np.isfinite(hdr.get(k, np.nan)):
                return np.nan
        try:
            return float(locations2degrees(
                hdr['evla'], hdr['evlo'], hdr['stla'], hdr['stlo']))
        except (ValueError, TypeError):
            return np.nan


    def _azimuth(hdr: Dict[str, float]) -> float:
        if np.isfinite(hdr.get('az', np.nan)):
            return float(hdr['az'])
        for k in ('stla', 'stlo', 'evla', 'evlo'):
            if not np.isfinite(hdr.get(k, np.nan)):
                return np.nan
        try:
            return float(gps2dist_azimuth(
                hdr['evla'], hdr['evlo'], hdr['stla'], hdr['stlo'])[1])
        except (ValueError, TypeError):
            return np.nan


    def _write_event_model_csv(self, r: EventModelResult) -> Path:
        path = self.output_dir / r.event_id / f'meas_{r.model}_vs_obs.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('event,model,network,station,component,band,phase,'
                     'dist_deg,azimuth,t_pred,win_start,win_end,'
                     'dt_shift,ccmax,nzcc,dln_a,rel_l2,snr\n')
            for m in r.measurements:
                fh.write(
                    f'{r.event_id},{r.model},{m.network},{m.station},'
                    f'{m.component},{m.band},{m.phase},'
                    f'{m.dist_deg:.4f},{m.azimuth:.2f},{m.t_pred:.3f},'
                    f'{m.win_start:.3f},{m.win_end:.3f},'
                    f'{m.dt_shift:.6f},{m.ccmax:.6f},{m.nzcc:.6f},'
                    f'{m.dln_a:.6f},{m.rel_l2:.6f},{m.snr:.4f}\n'
                )
        return path


    def _stats(vals: Sequence[float]) -> Dict[str, float]:
        a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        if a.size == 0:
            return {'n': 0, 'median': np.nan, 'median_abs': np.nan,
                    'mean': np.nan, 'p95_abs': np.nan}
        return {
            'n': int(a.size),
            'median': float(np.median(a)),
            'median_abs': float(np.median(np.abs(a))),
            'mean': float(np.mean(a)),
            'p95_abs': float(np.percentile(np.abs(a), 95)),
        }


    _WKey = Tuple[str, str, str, str, str]  # (net, sta, comp, band, phase)


    def _wkey(m: Measurement) -> Tuple[str, str, str, str, str]:
        return (m.network, m.station, m.component, m.band, m.phase)


    def _phase_class(m: Measurement) -> str:
        """Zhou 六分类：体波/面波 × 分量（full 模式退化为按分量三类）。"""
        if m.phase in ('P', 'SV', 'SH'):
            return f'body-{m.component}'
        if m.phase in ('Rayleigh', 'Love'):
            return f'surf-{m.component}'
        return f'full-{m.component}'


    def _band_tmax(self, band: str) -> float:
        for b, _tmin, tmax in self.config.bands:
            if b == band:
                return float(tmax)
        return np.nan


    def _dt_reliable(self, m: Measurement) -> bool:
        """dT/dlnA 测量是否可信（接受准则见 MultiEventConfig.accept）。"""
        acc = self.config.accept
        if not (np.isfinite(m.dt_shift) and np.isfinite(m.ccmax)):
            return False
        if m.ccmax < acc['cc_min']:
            return False
        tmax = self._band_tmax(m.band)
        win_len = m.win_end - m.win_start
        if np.isfinite(tmax) and np.isfinite(win_len) and win_len > 0:
            max_lag = min(self.config.max_lag_ratio * tmax, 0.25 * win_len)
            if abs(m.dt_shift) >= acc['edge_lag_frac'] * max_lag:
                return False
        if not np.isfinite(m.snr) or m.snr < acc['snr_min']:
            return False
        return True


    def _accepted_keys(
        self,
        mp: Dict[_WKey, Measurement],
        band: Optional[str] = None,
    ) -> List[_WKey]:
        """单个模型通过 CC/SNR 接受准则的窗口（Zhou 精选窗）。"""
        out: List[_WKey] = []
        for k, m in mp.items():
            if band is not None and k[3] != band:
                continue
            if self._dt_reliable(m):
                out.append(k)
        return out


    def _common_by_event(
        self,
    ) -> Dict[str, Dict[str, Dict[_WKey, Measurement]]]:
        """事件 → 模型 → {窗口键: 测量}，窗口键取该事件所有模型的交集。"""
        by_event: Dict[str, Dict[str, Dict[Any, Measurement]]] = {}
        for r in self.results:
            mp = by_event.setdefault(r.event_id, {}).setdefault(r.model, {})
            for m in r.measurements:
                mp[self._wkey(m)] = m
        out: Dict[str, Dict[str, Dict[Any, Measurement]]] = {}
        for eid, models_map in by_event.items():
            if not models_map:
                continue
            keys = set.intersection(
                *(set(mp.keys()) for mp in models_map.values())
            )
            if not keys:
                continue
            out[eid] = {
                model: {k: mp[k] for k in keys}
                for model, mp in models_map.items()
            }
        return out


    def _reliable_dt_keys(
        self, models_map: Dict[str, Dict[_WKey, Measurement]],
    ) -> Set[_WKey]:
        """所有模型都通过 dT 接受准则的公共窗口（保证 dT 统计样本一致）。"""
        keys = next(iter(models_map.values())).keys()
        return {
            k for k in keys
            if all(self._dt_reliable(mp[k]) for mp in models_map.values())
        }


    def _wr_weights(
        self, any_map: Dict[_WKey, Measurement],
    ) -> Dict[_WKey, float]:
        """W_r：窗口权重 = 1/该台站所在 (方位角×距离) 格内的台站数。"""
        az_bins = int(self.config.weighting['az_bins'])
        edges = list(self.config.weighting['dist_edges'])
        cell_of_station: Dict[Tuple[str, str], Tuple[int, int]] = {}
        n_in_cell: Dict[Tuple[int, int], int] = {}
        for m in any_map.values():
            sta = (m.network, m.station)
            if sta in cell_of_station:
                continue
            az = m.azimuth if np.isfinite(m.azimuth) else 0.0
            dd = m.dist_deg if np.isfinite(m.dist_deg) else 0.0
            ia = int((az % 360.0) // (360.0 / az_bins))
            ib = int(np.searchsorted(edges, dd, side='right')) - 1
            ib = min(max(ib, 0), len(edges) - 2)
            cell_of_station[sta] = (ia, ib)
            n_in_cell[(ia, ib)] = n_in_cell.get((ia, ib), 0) + 1
        return {
            k: 1.0 / n_in_cell[cell_of_station[(m.network, m.station)]]
            for k, m in any_map.items()
        }


    def _wmedian(vals: Sequence[float], wts: Sequence[float]) -> float:
        """加权中位数（累计权重过半处的取值）。"""
        a = np.asarray(vals, dtype=float)
        w = np.asarray(wts, dtype=float)
        ok = np.isfinite(a) & np.isfinite(w) & (w > 0)
        if not np.any(ok):
            return np.nan
        a, w = a[ok], w[ok]
        order = np.argsort(a)
        a, w = a[order], w[order]
        cw = np.cumsum(w)
        return float(a[int(np.searchsorted(cw, 0.5 * cw[-1]))])


    def _balanced_score(
        self,
        mp: Dict[_WKey, Measurement],
        keys: Sequence[_WKey],
        weights: Dict[_WKey, float],
        value_fn: Any,
    ) -> float:
        """W_c：类内 W_r 加权中位数 → 类间等权平均。"""
        by_class: Dict[str, Tuple[List[float], List[float]]] = {}
        for k in keys:
            m = mp[k]
            v = value_fn(m)
            if not np.isfinite(v):
                continue
            cls = self._phase_class(m)
            vals, wts = by_class.setdefault(cls, ([], []))
            vals.append(float(v))
            wts.append(weights.get(k, 1.0))
        scores = [self._wmedian(v, w) for v, w in by_class.values() if v]
        return float(np.mean(scores)) if scores else np.nan


    def aggregate_table(self) -> List[Dict[str, Any]]:
        """
        汇总表：每 (event, model, band) 一行，全部建立在跨模型公共窗口上。

        misfit 约定（越小越好）:
          dt_abs_score / dlnA_abs_score / one_minus_nzcc_score / rel_l2_score
        score = W_r 加权类内中位数 → 六类等权平均。
        四个 misfit 都只在该模型自己通过 CC/SNR 的公共存在窗上计（n_dt_ok）。
        predict_frac 仍用全部公共存在窗（Zhou Fig.10：所有可能窗中 NZCC>0.7 的比例）。
        """
        common = self._common_by_event()
        rows: List[Dict[str, Any]] = []
        bands = [b[0] for b in self.config.bands]
        thr = self.config.nzcc_predict_threshold
        for ev in self.events:
            eid = ev.event_id
            if eid not in common:
                continue
            models_map = common[eid]
            any_map = next(iter(models_map.values()))
            weights = self._wr_weights(any_map)
            for model in sorted(models_map):
                mp = models_map[model]
                for band in bands:
                    keys_b = [k for k in mp if k[3] == band]
                    if not keys_b:
                        continue
                    dt_keys_b = self._accepted_keys(mp, band)
                    n_pred = sum(1 for k in keys_b if mp[k].nzcc > thr)
                    rows.append({
                        'event': eid,
                        'model': model,
                        'band': band,
                        'n_common': len(keys_b),
                        'n_dt_ok': len(dt_keys_b),
                        'dt_abs_score': self._balanced_score(
                            mp, dt_keys_b, weights,
                            lambda m: abs(m.dt_shift)),
                        'dlnA_abs_score': self._balanced_score(
                            mp, dt_keys_b, weights,
                            lambda m: abs(m.dln_a)),
                        'one_minus_nzcc_score': self._balanced_score(
                            mp, dt_keys_b, weights,
                            lambda m: 1.0 - m.nzcc),
                        'rel_l2_score': self._balanced_score(
                            mp, dt_keys_b, weights,
                            lambda m: m.rel_l2 if (
                                np.isfinite(m.rel_l2) and 0.0 <= m.rel_l2 < 20.0
                            ) else np.nan),
                        'predict_frac': n_pred / max(len(keys_b), 1),
                    })
        return rows


    def write_aggregate_csv(self) -> Path:
        rows = self.aggregate_table()
        path = self.output_dir / 'aggregate_by_event_model_band.csv'
        if not rows:
            print('⚠️  无汇总行可写')
            return path
        keys = list(rows[0].keys())
        with open(path, 'w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'📄 {path}')
        return path


    def _rows_index(
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
        """rows → band → event → model → row"""
        idx: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        for row in rows:
            idx.setdefault(row['band'], {}) \
               .setdefault(row['event'], {})[row['model']] = row
        return idx


    def write_report(self) -> Path:
        rows = self.aggregate_table()
        idx = self._rows_index(rows)
        acc = self.config.accept
        lines = [
            '=' * 78,
            'EASTASIA-FWI 多事件多模型波形评估报告',
            '参考 = 观测波形；指标 = 合成相对观测',
            '=' * 78,
            f'SAC: {self.sac_root}',
            f'观测: {self.obs_root}',
            f'模型: {", ".join(self.models)}',
            f'事件: {", ".join(e.event_id for e in self.events)}',
            '',
            '统计口径（Zhou et al. 2021 / Open Issue 6.3）:',
            '  * 只统计所有模型都测到的窗口（公共存在窗）',
            f'  * χT/χA/χF/χN 按各模型自己的 ccmax>={acc["cc_min"]}、'
            f'峰未贴边、有限 SNR>={acc["snr_min"]} 精选；'
            'predict_frac 仍用全部公共存在窗',
            '  * W_r: ' + (
                '1° geographic cells（1/格内台站数）'
                if str(self.config.weighting.get('mode', '')) == 'geo'
                else '台站按方位角×震中距分格加权（1/格内台站数）'
            ),
            '  * W_c: 体波/面波 × Z/R/T 六类等权',
            '  * 主结论 = 逐事件评分的事件等权平均与平均排名，非 pooled',
            '',
            '模型角色:',
            '  BASE            原作者 FWEA23 原生 GLL（model_updated），不换地壳、不插值',
            '                  是目前最好拟合的上界，不是 1-D 参考',
            '  S362ANI_c1      CRUST1.0 + s362ani，共用地壳下的空对照',
            '  FWEA23_c1       CRUST1.0 + FWEA23 地幔（NC→GLL）；相对 BASE = 换地壳+插值',
            '  EARA2024_c1     CRUST1.0 + EARA2024 地幔',
            '  SinoScope_c1    CRUST1.0 + SinoScope 1° HDF5',
            '  SinoScope_sem_c1  CRUST1.0 + SinoScope 作者 Salvus mesh',
            '  公平比地幔：四个 *_c1 互比。BASE 用来报最好拟合，并量 FWEA23_c1 掉了多少。',
            '',
            '符号: dT>0 合成偏晚；dlnA>0 合成偏大；NZCC→1 越好；rel_L2→0 越好',
            '',
        ]
        models = sorted({row['model'] for row in rows})
        bands = [b[0] for b in self.config.bands]

        for band in bands:
            ev_map = idx.get(band, {})
            if not ev_map:
                continue
            # 逐事件排名 → 平均排名（主结论）
            rank_sum: Dict[str, float] = {m: 0.0 for m in models}
            rank_cnt: Dict[str, int] = {m: 0 for m in models}
            for eid, model_rows in ev_map.items():
                scored = [
                    (m, row['dt_abs_score'])
                    for m, row in model_rows.items()
                    if np.isfinite(row['dt_abs_score'])
                ]
                scored.sort(key=lambda x: x[1])
                for rank, (m, _v) in enumerate(scored, 1):
                    rank_sum[m] += rank
                    rank_cnt[m] += 1

            lines.append(f'--- {band}  事件等权汇总（{len(ev_map)} 个事件）---')
            lines.append(
                f'{"模型":<16}{"平均排名":>8}{"dT_score":>10}{"relL2":>12}'
                f'{"1-NZCC":>10}{"pred%":>8}{"n_dt":>7}{"n_win":>7}'
            )

            def _event_mean(model: str, key: str) -> float:
                vals = [
                    row[key] for er in ev_map.values()
                    for m, row in er.items()
                    if m == model and np.isfinite(row[key])
                ]
                return float(np.mean(vals)) if vals else np.nan

            order = sorted(
                models,
                key=lambda m: rank_sum[m] / rank_cnt[m]
                if rank_cnt[m] else np.inf,
            )
            for model in order:
                mean_rank = rank_sum[model] / rank_cnt[model] \
                    if rank_cnt[model] else np.nan
                n_dt = int(sum(
                    row['n_dt_ok'] for er in ev_map.values()
                    for m, row in er.items() if m == model
                ))
                n_win = int(sum(
                    row['n_common'] for er in ev_map.values()
                    for m, row in er.items() if m == model
                ))
                dt_s = _event_mean(model, 'dt_abs_score')
                l2_s = _event_mean(model, 'rel_l2_score')
                nz_s = _event_mean(model, 'one_minus_nzcc_score')
                pr_s = 100.0 * _event_mean(model, 'predict_frac')
                lines.append(
                    f'{model:<16}{mean_rank:8.2f}'
                    f'{dt_s:10.3f}{l2_s:12.4g}{nz_s:10.4f}'
                    f'{pr_s:7.1f}%{n_dt:7d}{n_win:7d}'
                )
            lines.append('')

        # 逐事件排名明细
        lines.append('--- 逐事件排名（dT_score，1=最好）---')
        for band in bands:
            ev_map = idx.get(band, {})
            if not ev_map:
                continue
            lines.append(f'[{band}]')
            for ev in self.events:
                model_rows = ev_map.get(ev.event_id)
                if not model_rows:
                    continue
                scored = [
                    (m, row['dt_abs_score'], row['n_dt_ok'])
                    for m, row in model_rows.items()
                    if np.isfinite(row['dt_abs_score'])
                ]
                scored.sort(key=lambda x: x[1])
                rank_str = '  '.join(
                    f'{i+1}.{m}({v:.3f})' for i, (m, v, _n) in enumerate(scored)
                )
                n_dt = scored[0][2] if scored else 0
                lines.append(f'  {ev.event_id} [n_dt={n_dt}]: {rank_str}')
            lines.append('')

        path = self.output_dir / 'multi_event_report.txt'
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'📄 {path}')
        print('\n'.join(lines[-60:]))
        return path


    def _load_pair_from_csv(
        self, event_id: str, model: str,
    ) -> Optional[PairResult]:
        """从已写出的 meas_*_vs_obs.csv 恢复 PairResult，供选台与标注。"""
        path = self.output_dir / event_id / f'meas_{model}_vs_obs.csv'
        if not path.is_file():
            return None
        ms: List[Measurement] = []
        with open(path, 'r', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                try:
                    ms.append(Measurement(
                        network=row['network'],
                        station=row['station'],
                        component=row['component'],
                        band=row['band'],
                        phase=row.get('phase', 'full'),
                        dist_deg=float(row['dist_deg']),
                        azimuth=float(row.get('azimuth', 'nan')),
                        t_pred=float(row.get('t_pred', 'nan')),
                        win_start=float(row.get('win_start', 'nan')),
                        win_end=float(row.get('win_end', 'nan')),
                        dt_shift=float(row['dt_shift']),
                        ccmax=float(row['ccmax']),
                        nzcc=float(row['nzcc']),
                        dln_a=float(row['dln_a']),
                        rel_l2=float(row['rel_l2']),
                        snr=float(row.get('snr', 'nan')),
                    ))
                except (KeyError, ValueError):
                    continue
        if not ms:
            return None
        return PairResult(
            ref_label=OBSERVED_LABEL,
            test_label=model,
            measurements=ms,
            n_matched=len({(m.network, m.station) for m in ms}),
        )




# ── EastAsia 30 事件测量 ───────────────────────────────────────

_NET_SUFFIX = re.compile(r'^(.+)_(\d{2})$')

# 中国省级台网（CNSN 主体）；Hinet 在本库中台网码为 N
CNSN_NETS: Set[str] = {
    'AH', 'BJ', 'CQ', 'FJ', 'GD', 'GS', 'GX', 'GZ', 'HA', 'HB', 'HE', 'HI',
    'HL', 'HN', 'JL', 'JS', 'JX', 'LN', 'NM', 'NX', 'QH', 'SC', 'SD', 'SH',
    'SN', 'SX', 'TJ', 'XJ', 'XZ', 'YN', 'ZJ',
}
IRIS_NETS: Set[str] = {
    'IU', 'II', 'IC', 'G', 'GE', 'AU', 'MY', 'JP', 'MM', 'TW', 'KR', 'PS',
    'KN', 'HK', 'KG', 'KZ', 'RM', 'CB', 'TM', 'AD',
}
PROVIDER_COLORS: Dict[str, str] = {
    'CNSN': '#c0392b',
    'Hinet': '#2980b9',
    'IRIS': '#27ae60',
    'Other': '#7f8c8d',
}

# 抽查用的浅 / 中 / 深代表事件（默认已改为全部共同区事件）
DEFAULT_PLOT_EVENTS: Tuple[str, ...] = (
    'C201505120705A',   # Nepal M7.2 shallow  (27.67°N, 86.08°E)
    'C201403131706A',   # Kyushu intermediate (33.62°N, 131.82°E)
    'C201601020422A',   # NE China deep       (44.83°N, 129.75°E)
)

# 三模型共同区域由 BaseConfig 统一维护（优先 1_5 写出的 valid_standardized_region.json，否则用 metadata 交集）
COMMON_REGION_JSON = (
    PROJECT_ROOT / 'data' / 'models' / 'metadata' / 'valid_standardized_region.json'
)


# ── 小工具 ──────────────────────────────────────────────────────────────────


def strip_net_suffix(network: str) -> str:
    """去掉 SPECFEM 重名台站的 _02 / _03 后缀。"""
    m = _NET_SUFFIX.match(network.upper())
    return m.group(1) if m else network.upper()


def short_model(name: str) -> str:
    """图轴用的短模型名。"""
    aliases = {
        'BASE': 'BASE',
        'S362ANI_c1': 'S362',
        'FWEA23_c1': 'FWEA23',
        'EARA2024_c1': 'EARA24',
        'SinoScope_c1': 'SinoSc',
        'SinoScope_sem_c1': 'SinoSEM',
    }
    if name in aliases:
        return aliases[name]
    return name.replace('_c1', '')


def load_three_model_common_region(
    path: Optional[Path] = None,
    base_config: Optional[BaseConfig] = None,
) -> Dict[str, float]:
    """
    三模型共同覆盖区域（FWEA23 ∩ EARA2024 ∩ SinoScope1.0，与 2_1 standardized 一致）。

    若显式给出非默认 JSON 路径则从该文件读取，否则统一返回 ``BaseConfig.get_region_bounds('common')``。
    """
    if path is not None and Path(path) != COMMON_REGION_JSON:
        json_path = Path(path)
        if not json_path.is_file():
            raise FileNotFoundError(f'共同区域 JSON 不存在: {json_path}')
        with open(json_path, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
        lat, lon, dep = raw['lat'], raw['lon'], raw.get('depth', [0.0, 1000.0])
        return {
            'lon_min': float(lon[0]), 'lon_max': float(lon[1]),
            'lat_min': float(lat[0]), 'lat_max': float(lat[1]),
            'depth_min': float(dep[0]), 'depth_max': float(dep[1]),
        }
    cfg = base_config if base_config is not None else BaseConfig()
    return cfg.get_region_bounds('common')


def in_lonlat_box(lat: float, lon: float, box: Dict[str, float]) -> bool:
    """点是否落在共同区域经纬度框内。缺坐标视为框外。"""
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return False
    return (box['lat_min'] <= float(lat) <= box['lat_max']
            and box['lon_min'] <= float(lon) <= box['lon_max'])


def provider_of(network: str, catalog: Optional[Dict[Tuple[str, str], str]] = None,
                station: str = '') -> str:
    """台站数据来源族：CNSN / Hinet / IRIS / Other。"""
    net = network.upper()
    if catalog and station:
        hit = catalog.get((net, station.upper()))
        if hit:
            key = hit.upper()
            if 'CNSN' in key:
                return 'CNSN'
            if 'HINET' in key:
                return 'Hinet'
            if 'IRIS' in key:
                return 'IRIS'
    if net in CNSN_NETS:
        return 'CNSN'
    if net == 'N':
        return 'Hinet'
    if net in IRIS_NETS:
        return 'IRIS'
    return 'Other'


def model_color(name: str) -> str:
    if name == 'Chujie':
        return '#d62728'  # 与 2_waveform_plotting 参考正演一致
    return MODEL_COLORS.get(name, '#8c564b')


CHUJIE_EVENT = 'C201403131706A'
CHUJIE_DIR = HERE / 'OUTPUT_FILES-chujie'
PLOT_COMPONENTS: Tuple[str, ...] = ('Z', 'R', 'T')


# ── I/O ─────────────────────────────────────────────────────────────────────


def parse_eastasia_obs_name(name: str) -> Optional[Tuple[str, str, str, str]]:
    """
    解析 EastAsia_model 观测文件名。

    例: AH.ANQ.00CNSN.MXZ.sac → (AH, ANQ, 00CNSN, MXZ)
    """
    upper = name.upper()
    if not (upper.endswith('.SAC') or upper.endswith('.SAC.GZ')):
        return None
    stem = name[:-7] if upper.endswith('.SAC.GZ') else name[:-4]
    parts = stem.split('.')
    if len(parts) < 4:
        return None
    net, sta, loc, chan = parts[0], parts[1], parts[2], parts[3]
    if not net or not sta or not chan:
        return None
    return net.upper(), sta.upper(), loc, chan.upper()


def read_globe_stations(path: Optional[Path]) -> Dict[Tuple[str, str], StationCoord]:
    """
    读 SPECFEM3D Globe STATIONS。

    格式: STATION NETWORK LATITUDE LONGITUDE ELEVATION BURIAL
    """
    out: Dict[Tuple[str, str], StationCoord] = {}
    if path is None or not Path(path).is_file():
        return out
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            sta, net = parts[0].upper(), parts[1].upper()
            try:
                lat, lon = float(parts[2]), float(parts[3])
                elev = float(parts[4]) if len(parts) >= 5 else 0.0
            except ValueError:
                continue
            out[(strip_net_suffix(net), sta)] = StationCoord(
                strip_net_suffix(net), sta, lat, lon, elev)
    return out


def load_provider_catalog(path: Optional[Path]) -> Dict[Tuple[str, str], str]:
    """读 stations_catalog.csv 的 Provider 列。"""
    out: Dict[Tuple[str, str], str] = {}
    if path is None or not Path(path).is_file():
        return out
    with open(path, 'r', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            net = (row.get('Network') or '').strip().upper()
            sta = (row.get('Station') or '').strip().upper()
            prov = (row.get('Provider') or '').strip()
            if net and sta and prov:
                out[(net, sta)] = prov
    return out


def load_eastasia_events(
    events_list: Path,
    events_csv: Path,
    cmt_dir: Path,
    obs_root: Path,
) -> List[EventSpec]:
    """从 events.list + 30 事件 CSV + Harvard CMT 构建事件表。"""
    meta: Dict[str, Dict[str, Any]] = {}
    if Path(events_csv).is_file():
        with open(events_csv, 'r', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                eid = (row.get('event') or row.get('Event_ID') or '').strip()
                if not eid:
                    continue
                try:
                    meta[eid] = {
                        'lat': float(row.get('lat', row.get('latitude', 'nan'))),
                        'lon': float(row.get('lon', row.get('longitude', 'nan'))),
                        'depth_km': float(row.get('depth_km', 'nan')),
                        'mw': float(row.get('Mw', row.get('mw', 'nan'))),
                        'origin_utc': (row.get('origin_pde')
                                       or row.get('origin_utc') or '').strip(),
                    }
                except (TypeError, ValueError):
                    continue

    out: List[EventSpec] = []
    with open(events_list, 'r', encoding='utf-8') as fh:
        for line in fh:
            eid = line.strip()
            if not eid or eid.startswith('#'):
                continue
            cmt = Path(cmt_dir) / f'CMTSOLUTION_{eid}'
            if not cmt.is_file():
                alt = Path(cmt_dir) / eid
                cmt = alt if alt.is_file() else cmt
            origin = parse_origin_time(cmt) if cmt.is_file() else None
            m = meta.get(eid, {})
            if origin is None and m.get('origin_utc'):
                try:
                    origin = UTCDateTime(str(m['origin_utc']))
                except Exception:
                    origin = None
            if origin is None:
                print(f'⚠️  跳过 {eid}：无法解析发震时刻')
                continue
            obs_dir = Path(obs_root) / eid
            ev = EventSpec(
                event_id=eid,
                origin=origin,
                cmt_file=cmt,
                lat=float(m.get('lat', np.nan)),
                lon=float(m.get('lon', np.nan)),
                depth_km=float(m.get('depth_km', np.nan)),
                mw=float(m.get('mw', np.nan)),
                obs_dir=obs_dir if obs_dir.is_dir() else None,
                resp_dir=None,
            )
            out.append(ev)
    return out


class EastAsiaObsIndex:
    """
    EastAsia_model 观测索引（与 ObservedIndex 同接口，供 read_observed 使用）。

    通道优先级: MX*（已处理位移，R/T 已旋转）> BH/HH。
    """

    CHANNELS: Dict[str, Tuple[str, ...]] = {
        'Z': ('MXZ', 'BHZ', 'HHZ', 'LHZ'),
        'N': ('MXN', 'BHN', 'HHN', 'LHN'),
        'E': ('MXE', 'BHE', 'HHE', 'LHE'),
        'R': ('MXR',),
        'T': ('MXT',),
    }

    def __init__(self, waveform_dir: Path) -> None:
        self.waveform_dir = Path(waveform_dir)
        self.response_dir = None
        self.is_preprocessed = True
        self.files: Dict[Tuple[str, str, str], Path] = {}
        if not self.waveform_dir.is_dir():
            self.stations: Set[Tuple[str, str]] = set()
            return
        for path in self.waveform_dir.iterdir():
            if not path.is_file():
                continue
            parsed = parse_eastasia_obs_name(path.name)
            if parsed is None:
                continue
            net, sta, _loc, chan = parsed
            key = (net, sta, chan)
            # MX 优先；同通道后写不覆盖先写的 MX
            if key not in self.files or chan.startswith('MX'):
                self.files[key] = path
        self.stations = {(n, s) for n, s, _ in self.files}

    def has(self, network: str, station: str, component: str) -> bool:
        return self.pick_channel(network, station, component) is not None

    def pick_channel(self, network: str, station: str,
                     component: str) -> Optional[Tuple[str, Path]]:
        net, sta = network.upper(), station.upper()
        for chan in self.CHANNELS.get(component.upper(), ()):
            path = self.files.get((net, sta, chan))
            if path is not None:
                return chan, path
        return None

    def z_stations(self) -> Set[Tuple[str, str]]:
        """有观测 Z 文件的台。没有 Z 的台不参与评估。"""
        return {(n, s) for n, s in self.stations if self.has(n, s, 'Z')}


class SynDirIndex(DirIndex):
    """
    合成目录索引。

    只走 (台网, 台站, 分量) 严格键。4671 台里同名台站很多
    （AH.ANQ 与 SD.ANQ），父类的 (台站, 分量) 宽松匹配会配错台。
    """

    def __init__(self, directory: Path) -> None:
        super().__init__(directory)
        extra: Dict[Tuple[str, str, str], Any] = {}
        for (net, sta, comp), entry in self.strict.items():
            stripped = strip_net_suffix(net)
            if stripped != net:
                extra.setdefault((stripped, sta, comp), entry)
        self.strict.update(extra)
        # 禁用宽松键，避免跨台网同名台站互配
        self.loose = {}

    def get(self, network: str, station: str,
            component: str) -> Optional[Any]:
        return self.strict.get(
            (network.upper(), station.upper(), component.upper()))

    def z_stations(self) -> Set[Tuple[str, str]]:
        out: Set[Tuple[str, str]] = set()
        for (net, sta, comp) in self.strict:
            if comp == 'Z':
                out.add((strip_net_suffix(net), sta))
        return out


def syn_dir_ready(directory: Path) -> bool:
    """合成目录是否已有波形（碰到第一个 *.sem.sac 即返回）。"""
    if not directory.is_dir():
        return False
    return any(p.name.endswith('.sem.sac') for p in directory.iterdir())


_TAUP_LOCK = threading.Lock()


# ── 配置 ────────────────────────────────────────────────────────────────────


class EastAsiaConfig(MultiEventConfig):
    """30 事件评估配置：沿用 Zhou 接受准则，W_r 改为 1° 地理格。"""

    def __init__(self) -> None:
        super().__init__()
        self.weighting.update({
            'mode': 'geo',          # geo | azdist
            'geo_cell_deg': 1.0,
            'az_bins': 16,
            'dist_edges': (3.0, 8.0, 16.0, 24.0, 32.0, 180.0),
        })
        self.plot = {
            'n_gallery': 6,
            'n_section': 50,
            'time_max': 1800.0,
            'az_bins': 8,
            'dist_bins': 3,
            'pad_periods': 1.5,
            'min_nzcc': 0.3,
            'vg': {                 # 面波群速度窗 (vmin, vmax) km/s
                '20-40s': (2.8, 4.2),
                '40-120s': (3.2, 4.5),
            },
            'v_red': {              # 剖面约化速度 km/s
                '20-40s': 3.5,
                '40-120s': 3.8,
            },
        }
        self.min_syn_files = 50
        self.n_workers = min(8, os.cpu_count() or 4)
        self.visualization['figure_format'] = ['jpg']
        self.visualization['waveform_format'] = ['jpg']
        # 只保留 FWEA23 ∩ EARA2024 ∩ SinoScope 共同区域内的事件/台站
        self.use_common_region = True
        self.common_region_json = COMMON_REGION_JSON
        # 近震源 Δ<3° 的窗不参与评分和作图（磁盘 meas_*.csv 仍保留全文）
        self.min_dist_deg = 3.0
        # Zhou et al. 2021：Δ<10° 时 S 并入面波列，不单独开 SV/SH 窗
        self.min_s_dist_deg = 10.0
        self.chujie_dir = CHUJIE_DIR
        self.chujie_event = CHUJIE_EVENT


class WaveformAssessment(MultiEventAssessment):
    """EastAsia_model 30 事件：合成 vs 观测的测量与评分。"""

    def __init__(
        self,
        sac_root: Path,
        obs_root: Path,
        events: Sequence[EventSpec],
        models: Sequence[str] = DEFAULT_MODELS,
        output_dir: Optional[Path] = None,
        stations_file: Optional[Path] = None,
        stations_catalog: Optional[Path] = None,
    ) -> None:
        self.base_config = BaseConfig()
        self.config = EastAsiaConfig()
        self.sac_root = Path(sac_root)
        self.obs_root = Path(obs_root)
        self.models = list(models)
        self.events = list(events)
        self.stations_file = Path(stations_file) if stations_file else None
        self.output_dir = Path(output_dir) if output_dir else \
            SAC_ROOT_DEFAULT / 'figures_eastasia_30'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.WaveformAssessment', 'INFO',
        )

        self.station_coords = read_globe_stations(self.stations_file)
        self.providers = load_provider_catalog(stations_catalog)
        self.n_forward_stations = len(self.station_coords)
        self.n_stations_listed = 0
        if self.stations_file and self.stations_file.is_file():
            with open(self.stations_file, 'r', encoding='utf-8', errors='replace') as fh:
                self.n_stations_listed = sum(
                    1 for line in fh if len(line.split()) >= 4)
        self.results: List[EventModelResult] = []
        self._pbar: Optional[Any] = None
        self._obs_cache: Dict[str, EastAsiaObsIndex] = {}
        self._syn_cache: Dict[Tuple[str, str], SynDirIndex] = {}
        self._z_cache: Dict[Tuple[str, str, str, str, str],
                            Tuple[np.ndarray, np.ndarray]] = {}
        self._zrt_cache: Dict[Tuple[str, str, str, str, str],
                              Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        self._chujie_idx: Optional[SynDirIndex] = None
        self.common_region = load_three_model_common_region(
            self.config.common_region_json, base_config=self.base_config)
        self.events_all = list(self.events)
        if self.config.use_common_region:
            kept, dropped = [], []
            for ev in self.events_all:
                if in_lonlat_box(ev.lat, ev.lon, self.common_region):
                    kept.append(ev)
                else:
                    dropped.append(ev)
            self.events = kept
            self._dropped_events = dropped
        else:
            self._dropped_events = []
        self._setup_style()
        self._print_banner()


    def _setup_style(self) -> None:
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 10,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 0.8,
            'axes.unicode_minus': False,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'legend.frameon': False,
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.bbox': 'tight',
        })


    def _print_banner(self) -> None:
        print('🎯 EASTASIA-FWI 3_4 波形评估（测量与评分）')
        print(f'   观测:     {self.obs_root}  （已去响应 DISP，含 Z/R/T）')
        print(f'   合成:     {self.sac_root}')
        print(f'   输出:     {self.output_dir}')
        print(f'   模型:     {", ".join(self.models)}')
        print('   角色:     BASE = 原作者 FWEA23 原生 GLL（model_updated），'
              '不是 1-D；*_c1 才是共用地壳下的地幔对比')
        print(f'   正演台站: {self.n_stations_listed} 行 / '
              f'{self.n_forward_stations} 个唯一 (NET,STA)')
        print(f'   频段:     {", ".join(b[0] for b in self.config.bands)}')
        print(f'   W_r:      1° geographic cells（抑制 CNSN/Hinet 过密）')
        box = self.common_region
        if self.config.use_common_region:
            print(f'   共同区域: {box["lat_min"]:.1f}–{box["lat_max"]:.1f}°N, '
                  f'{box["lon_min"]:.1f}–{box["lon_max"]:.1f}°E'
                  f'  （FWEA23 ∩ EARA2024 ∩ SinoScope，与 2_1 一致）')
            print(f'   事件:     {len(self.events)}/{len(self.events_all)} 个在共同区域内'
                  f'，去掉 {len(self._dropped_events)} 个框外事件')
            if self._dropped_events:
                print('            框外: ' + ', '.join(
                    e.event_id for e in self._dropped_events))
        else:
            print(f'   事件:     {len(self.events)}  （未裁共同区域）')
        min_d = float(self.config.min_dist_deg)
        if min_d > 0:
            print(f'   震中距:   丢掉 Δ < {min_d:g}° 的台/窗（计算与作图）')
        min_s = float(getattr(self.config, 'min_s_dist_deg', 0.0) or 0.0)
        if min_s > 0:
            print(f'   体波 S:   Δ < {min_s:g}° 不评 SV/SH（与 Zhou 2021 一致）')


    def _obs_index(self, ev: EventSpec) -> Optional[EastAsiaObsIndex]:
        if ev.obs_dir is None:
            return None
        key = ev.event_id
        if key not in self._obs_cache:
            self._obs_cache[key] = EastAsiaObsIndex(ev.obs_dir)
        return self._obs_cache[key]


    def check_readiness(self) -> None:
        """打印各事件观测 Z 台数（合成已齐，不再统计下载百分比）。"""
        print(f'\n{"=" * 72}')
        print('事件清单   评估样本 = 观测Z ∩ 合成Z ∩ 三模型共同区域 ∩ Δ≥3°')
        print(f'{"=" * 72}')
        print(f'{"事件":<16}{"Mw":>6}{"深度":>8}{"观测Z":>8}{"可用Z":>10}')
        print('-' * 72)
        n_obs = n_in = 0
        for ev in self.events:
            obs = self._obs_index(ev)
            n_z = len(obs.z_stations()) if obs else 0
            n_keep = len(self._filter_stations(sorted(obs.z_stations()), ev)) if obs else 0
            n_obs += n_z
            n_in += n_keep
            mw = f'{ev.mw:.1f}' if np.isfinite(ev.mw) else '?'
            dep = f'{ev.depth_km:.0f}' if np.isfinite(ev.depth_km) else '?'
            print(f'{ev.event_id:<16}{mw:>6}{dep:>7} km{n_z:>8}{n_keep:>10}')
        print('-' * 72)
        print(f'{len(self.events)} 个共同区域内事件；观测 Z {n_obs} → 可用 {n_in}')
        if self._dropped_events:
            print('框外事件已去掉: ' + ', '.join(
                e.event_id for e in self._dropped_events))
        print('没有可用观测 Z 的台整台不用。')


    def _compare_model_to_obs(
        self,
        ev: EventSpec,
        model: str,
        syn_dir: Path,
        obs_idx: EastAsiaObsIndex,
    ) -> EventModelResult:
        syn_idx = SynDirIndex(syn_dir)
        stations = sorted(syn_idx.z_stations() & obs_idx.z_stations())
        n_raw = len(stations)
        stations = self._filter_stations(stations, ev)
        print(f'   [{model}] 共同台站（合成Z ∩ 观测Z）: {n_raw}'
              f' → 共同区域且 Δ≥{self.config.min_dist_deg:g}° {len(stations)}')
        result = EventModelResult(event_id=ev.event_id, model=model)
        n_ok = n_skip = 0
        pbar = self._pbar
        n_workers = max(1, int(self.config.n_workers))

        def _consume(got: List[Measurement]) -> None:
            nonlocal n_ok, n_skip
            if got:
                result.measurements.extend(got)
                n_ok += 1
            else:
                n_skip += 1
            if pbar is not None:
                pbar.set_postfix(
                    event=ev.event_id, model=model, refresh=False)
                pbar.update(1)

        if n_workers <= 1 or len(stations) < 4:
            for net, sta in stations:
                _consume(self._measure_station_vs_obs(
                    net, sta, syn_idx, obs_idx, ev))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futs = [
                    pool.submit(self._measure_station_vs_obs,
                                net, sta, syn_idx, obs_idx, ev)
                    for net, sta in stations
                ]
                for fut in as_completed(futs):
                    _consume(fut.result())
        result.n_stations = n_ok
        result.n_skip = n_skip
        print(f'   [{model}] ✅ {n_ok} 台有效，跳过 {n_skip}，'
              f'{len(result.measurements)} 条测量')
        return result


    def _station_coord(self, network: str, station: str) -> Optional[StationCoord]:
        return self.station_coords.get((network.upper(), station.upper()))


    def _in_common_region(self, lat: float, lon: float) -> bool:
        if not self.config.use_common_region:
            return True
        return in_lonlat_box(lat, lon, self.common_region)


    def _station_in_common(self, network: str, station: str) -> bool:
        """无坐标的台视为框外，不参与共同区域评估。"""
        if not self.config.use_common_region:
            return True
        coord = self._station_coord(network, station)
        if coord is None:
            return False
        return in_lonlat_box(coord.lat, coord.lon, self.common_region)


    def _event_spec(self, event_id: str) -> Optional[EventSpec]:
        for ev in self.events_all:
            if ev.event_id == event_id:
                return ev
        return None


    def _station_dist_deg(
        self, ev: Optional[EventSpec], network: str, station: str,
    ) -> float:
        """事件–台站大圆弧距离（度）；缺坐标则为 NaN。"""
        if ev is None or not np.isfinite(ev.lat) or not np.isfinite(ev.lon):
            return np.nan
        coord = self._station_coord(network, station)
        if coord is None:
            return np.nan
        try:
            return float(locations2degrees(ev.lat, ev.lon, coord.lat, coord.lon))
        except (ValueError, TypeError):
            return np.nan


    def _dist_ok(
        self,
        dist_deg: float,
        ev: Optional[EventSpec] = None,
        network: str = '',
        station: str = '',
    ) -> bool:
        """True = 震中距达到门槛（默认 Δ≥3°）。无法确定距离则丢掉。"""
        min_d = float(getattr(self.config, 'min_dist_deg', 0.0) or 0.0)
        if min_d <= 0.0:
            return True
        d = float(dist_deg) if np.isfinite(dist_deg) else np.nan
        if not np.isfinite(d):
            d = self._station_dist_deg(ev, network, station)
        if not np.isfinite(d):
            return False
        return d >= min_d


    def _keep_measurement(
        self, m: Measurement, ev: Optional[EventSpec] = None,
    ) -> bool:
        if not self._station_in_common(m.network, m.station):
            return False
        if not self._dist_ok(m.dist_deg, ev, m.network, m.station):
            return False
        min_s = float(getattr(self.config, 'min_s_dist_deg', 0.0) or 0.0)
        if min_s > 0.0 and m.phase in ('SV', 'SH'):
            d = float(m.dist_deg) if np.isfinite(m.dist_deg) else np.nan
            if not np.isfinite(d):
                d = self._station_dist_deg(ev, m.network, m.station)
            if np.isfinite(d) and d < min_s:
                return False
        return True


    def _filter_stations(
        self,
        stations: Sequence[Tuple[str, str]],
        ev: Optional[EventSpec] = None,
    ) -> List[Tuple[str, str]]:
        kept: List[Tuple[str, str]] = []
        for n, s in stations:
            if not self._station_in_common(n, s):
                continue
            if ev is not None and not self._dist_ok(np.nan, ev, n, s):
                continue
            kept.append((n, s))
        return kept


    def _filter_measurements(
        self,
        measurements: Sequence[Measurement],
        ev: Optional[EventSpec] = None,
    ) -> List[Measurement]:
        return [m for m in measurements if self._keep_measurement(m, ev)]


    def _restrict_result(self, r: EventModelResult) -> EventModelResult:
        """丢掉共同区域外、Δ<3°、或近震 SV/SH 的测量；不改磁盘 meas_*.csv。"""
        ev = self._event_spec(r.event_id)
        kept = self._filter_measurements(r.measurements, ev)
        if len(kept) == len(r.measurements):
            return r
        r.measurements = kept
        r.n_stations = len({(m.network, m.station) for m in kept})
        return r


    def _fill_geometry(
        self,
        raw: RawTrace,
        ev: EventSpec,
        network: str,
        station: str,
    ) -> None:
        """用事件目录 + STATIONS 覆盖 ECEF / 缺失头段。"""
        coord = self._station_coord(network, station)
        if np.isfinite(ev.lat):
            raw.header['evla'] = float(ev.lat)
            raw.header['evlo'] = float(ev.lon)
            raw.header['evdp'] = float(ev.depth_km)
        if coord is not None:
            raw.header['stla'] = coord.lat
            raw.header['stlo'] = coord.lon
        need = ('stla', 'stlo', 'evla', 'evlo')
        if all(np.isfinite(raw.header.get(k, np.nan)) for k in need):
            try:
                dist_m, az, baz = gps2dist_azimuth(
                    raw.header['evla'], raw.header['evlo'],
                    raw.header['stla'], raw.header['stlo'])
                raw.header['az'] = float(az)
                raw.header['baz'] = float(baz)
                raw.header['gcarc'] = float(locations2degrees(
                    raw.header['evla'], raw.header['evlo'],
                    raw.header['stla'], raw.header['stlo']))
                _ = dist_m
            except (ValueError, TypeError):
                pass


    def _measure_station_vs_obs(
        self,
        network: str,
        station: str,
        syn_idx: SynDirIndex,
        obs_idx: EastAsiaObsIndex,
        ev: EventSpec,
    ) -> List[Measurement]:
        """单台：观测 vs 合成。观测或合成缺可用 Z 则整台不用。"""
        out: List[Measurement] = []
        pw = self.config.phase_windows
        origin = ev.origin

        _ = obs_idx
        raw_syn: Dict[str, RawTrace] = {}
        for comp in ('Z', 'N', 'E'):
            entry = syn_idx.get(network, station, comp)
            if entry is None:
                continue
            rs = read_raw(entry.path, self.config, origin=origin)
            if rs is None:
                continue
            self._fill_geometry(rs, ev, network, station)
            raw_syn[comp] = rs
        zrt_syn = self._raws_to_zrt(raw_syn)
        if zrt_syn is None or 'Z' not in zrt_syn:
            return out

        raw_obs = self._load_station_raws(ev, network, station, None)
        zrt_obs = self._raws_to_zrt(raw_obs) if raw_obs else None
        if zrt_obs is None or 'Z' not in zrt_obs:
            return out

        hdr = raw_syn['Z'].header
        dist = self._dist_deg(hdr)
        if not self._dist_ok(dist, ev, network, station):
            return out
        az = self._azimuth(hdr)
        depth = hdr.get('evdp', ev.depth_km)
        with _TAUP_LOCK:
            arrivals = taup_arrivals(
                dist, depth, pw.get('taup_model', 'ak135'))
        t_first = arrivals.get('P', min(arrivals.values()) if arrivals else np.nan)

        filt_cache: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]] = {}

        def filtered(side: str, comp: str, band: str,
                     tmin: float, tmax: float
                     ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            key = (side, comp, band)
            if key in filt_cache:
                return filt_cache[key]
            zrt = zrt_obs if side == 'obs' else zrt_syn
            if comp not in zrt:
                return None
            t0, d0 = zrt[comp]
            tr = Trace(data=np.asarray(d0, dtype=float))
            tr.stats.delta = float(t0[1] - t0[0]) if len(t0) > 1 else 0.1
            d_f = filter_band(tr, tmin, tmax, self.config)
            if len(d_f) != len(t0):
                return None
            filt_cache[key] = (t0, d_f)
            return filt_cache[key]

        use_phases = bool(pw.get('enabled', True))
        for band, tmin, tmax in self.config.bands:
            comps_phases = pw['comp_phases'] if use_phases else {
                'Z': ['full'], 'R': ['full'], 'T': ['full'],
            }
            for comp, phases in comps_phases.items():
                fo = filtered('obs', comp, band, tmin, tmax)
                fs = filtered('syn', comp, band, tmin, tmax)
                if fo is None or fs is None:
                    continue
                common = to_common_axis(fo[0], fo[1], fs[0], fs[1])
                if common is None:
                    continue
                t, d_o, d_s = common
                for phase in phases:
                    min_s = float(getattr(self.config, 'min_s_dist_deg', 0.0) or 0.0)
                    if (min_s > 0.0 and phase in ('SV', 'SH')
                            and np.isfinite(dist) and dist < min_s):
                        continue
                    if phase == 'full':
                        w0, w1, t_pred = float(t[0]), float(t[-1]), float('nan')
                    else:
                        win = phase_time_window(
                            phase, band, tmin, tmax, dist, arrivals, pw)
                        if win is None:
                            continue
                        w0, w1, t_pred = win
                    mask = (t >= w0) & (t <= w1)
                    if int(mask.sum()) < self.config.qc['min_samples']:
                        continue
                    seg_o, seg_s = d_o[mask], d_s[mask]
                    amax = float(np.max(np.abs(seg_o)))
                    if amax <= 0 or (np.std(seg_o) / amax
                                     < self.config.qc['flat_std_ratio']):
                        continue
                    dt = float(t[1] - t[0])
                    win_len = max(w1 - w0, dt)
                    max_lag = min(self.config.max_lag_ratio * tmax,
                                  0.25 * win_len)
                    snr = obs_window_snr(t, d_o, mask, t_first, tmax)
                    met = measure_pair(seg_o, seg_s, dt, max_lag)
                    if met is None:
                        continue
                    out.append(Measurement(
                        network=network, station=station, component=comp,
                        band=band, phase=phase, dist_deg=dist, azimuth=az,
                        t_pred=t_pred, win_start=w0, win_end=w1, snr=snr,
                        **met,
                    ))
        return out


    def assess_event(
        self,
        ev: EventSpec,
        resume: bool = True,
        force: bool = False,
    ) -> List[EventModelResult]:
        """评估单事件；已有 CSV 且 resume 时直接读回。"""
        if not self._in_common_region(ev.lat, ev.lon):
            print(f'\n⚠️  {ev.event_id}: 震源在三模型共同区域外，跳过')
            return []
        obs_idx = self._obs_index(ev)
        if obs_idx is None or not obs_idx.stations:
            print(f'\n⚠️  {ev.event_id}: 无观测，跳过')
            return []

        avail = [
            m for m in self.models
            if syn_dir_ready(syn_event_dir(self.sac_root, m, ev.event_id))
        ]
        if not avail:
            print(f'\n⚠️  {ev.event_id}: 无合成波形，跳过')
            return []

        print(f'\n{"=" * 70}')
        print(f'事件 {ev.event_id}  |  {ev.origin}  |  '
              f'Mw={ev.mw if np.isfinite(ev.mw) else "?"}  |  '
              f'观测Z {len(obs_idx.z_stations())}')
        print(f'模型: {", ".join(avail)}')
        print(f'{"=" * 70}')

        event_results: List[EventModelResult] = []
        for model in avail:
            csv_path = self.output_dir / ev.event_id / f'meas_{model}_vs_obs.csv'
            if resume and not force and csv_path.is_file():
                pr = self._load_pair_from_csv(ev.event_id, model)
                if pr is not None and pr.measurements:
                    r = self._restrict_result(EventModelResult(
                        event_id=ev.event_id, model=model,
                        measurements=pr.measurements,
                        n_stations=pr.n_matched,
                    ))
                    event_results.append(r)
                    self.results.append(r)
                    print(f'   [{model}] ↩  读取已有 {len(r.measurements)} 条'
                          f'（共同区域且 Δ≥{self.config.min_dist_deg:g}°）')
                    if self._pbar is not None:
                        # 恢复时无法精确回退步数，只刷新后缀
                        self._pbar.set_postfix(
                            event=ev.event_id, model=f'{model} (cached)',
                            refresh=False)
                    continue
            syn_dir = syn_event_dir(self.sac_root, model, ev.event_id)
            r = self._compare_model_to_obs(ev, model, syn_dir, obs_idx)
            event_results.append(r)
            self.results.append(r)
            self._write_event_model_csv(r)
        return event_results


    def _count_work_units(self, todo: Sequence[EventSpec]) -> int:
        total = 0
        for ev in todo:
            obs = self._obs_index(ev)
            if obs is None:
                continue
            n_obs_z = len(self._filter_stations(sorted(obs.z_stations()), ev))
            for model in self.models:
                d = syn_event_dir(self.sac_root, model, ev.event_id)
                if not syn_dir_ready(d):
                    continue
                csv_path = self.output_dir / ev.event_id / f'meas_{model}_vs_obs.csv'
                if csv_path.is_file():
                    continue
                total += n_obs_z
        return total


    def _wr_weights(
        self, any_map: Dict[Tuple[str, str, str, str, str], Measurement],
    ) -> Dict[Tuple[str, str, str, str, str], float]:
        """
        W_r：默认按 1°×1° 格，权重 = 1/格内台站数。

        华北 / 川滇 / Hinet 在同一震中距带里仍会挤在一个 az×dist 格子里；
        地理格能把这块过密压下去。azdist 模式保留作对照。
        """
        mode = str(self.config.weighting.get('mode', 'geo'))
        if mode != 'geo':
            return super()._wr_weights(any_map)

        cell_deg = float(self.config.weighting.get('geo_cell_deg', 1.0))
        cell_of: Dict[Tuple[str, str], Tuple[int, int]] = {}
        n_in: Dict[Tuple[int, int], int] = {}
        for m in any_map.values():
            sta = (m.network, m.station)
            if sta in cell_of:
                continue
            coord = self._station_coord(m.network, m.station)
            if coord is None:
                # 无坐标：退回方位角×距离，避免整台丢权
                az = m.azimuth if np.isfinite(m.azimuth) else 0.0
                dd = m.dist_deg if np.isfinite(m.dist_deg) else 0.0
                cell = (int(az // 30.0), int(dd // 5.0))
            else:
                cell = (int(np.floor(coord.lon / cell_deg)),
                        int(np.floor(coord.lat / cell_deg)))
            cell_of[sta] = cell
            n_in[cell] = n_in.get(cell, 0) + 1
        return {
            k: 1.0 / n_in[cell_of[(m.network, m.station)]]
            for k, m in any_map.items()
            if (m.network, m.station) in cell_of
        }


    def _ensure_results_from_csv(
        self,
        event_ids: Optional[Sequence[str]] = None,
    ) -> None:
        """出图-only 时从 meas_*.csv 灌回 self.results。"""
        if self.results:
            return
        want = set(event_ids) if event_ids else None
        for ev in self.events:
            if want is not None and ev.event_id not in want:
                continue
            for model in self.models:
                pr = self._load_pair_from_csv(ev.event_id, model)
                if pr is None or not pr.measurements:
                    continue
                self.results.append(self._restrict_result(EventModelResult(
                    event_id=ev.event_id,
                    model=model,
                    measurements=pr.measurements,
                    n_stations=pr.n_matched,
                )))


    def _syn_index(self, ev: EventSpec, model: str) -> SynDirIndex:
        if model == 'Chujie':
            idx = self._chujie_index()
            if idx is None:
                return SynDirIndex(Path('/__missing_chujie__'))
            return idx
        key = (ev.event_id, model)
        hit = self._syn_cache.get(key)
        if hit is None:
            hit = SynDirIndex(syn_event_dir(self.sac_root, model, ev.event_id))
            self._syn_cache[key] = hit
        return hit


    def _load_station_raws(
        self,
        ev: EventSpec,
        net: str,
        sta: str,
        model: Optional[str],
    ) -> Optional[Dict[str, RawTrace]]:
        """读一台原始道。观测优先 Z/R/T（已旋好）；合成读 Z/N/E。"""
        raws: Dict[str, RawTrace] = {}
        if model is None:
            obs = self._obs_index(ev)
            if obs is None:
                return None
            for comp in ('Z', 'N', 'E', 'R', 'T'):
                raw = read_observed(obs, net, sta, comp, ev.origin, self.config)
                if raw is None:
                    continue
                self._fill_geometry(raw, ev, net, sta)
                raws[comp] = raw
        else:
            idx = self._syn_index(ev, model)
            for comp in ('Z', 'N', 'E'):
                entry = idx.get(net, sta, comp)
                if entry is None:
                    continue
                raw = read_raw(entry.path, self.config, origin=ev.origin)
                if raw is None:
                    continue
                self._fill_geometry(raw, ev, net, sta)
                raws[comp] = raw
        return raws if 'Z' in raws else None


    def _raws_to_zrt(
        self, raws: Dict[str, RawTrace],
    ) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """观测已有 R/T 则直接用；否则用 baz 把 N/E 旋到 R/T。"""
        if 'Z' not in raws:
            return None
        if 'R' in raws and 'T' in raws:
            out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for comp in ('Z', 'R', 'T'):
                raw = raws[comp]
                out[comp] = (raw.time, np.asarray(raw.trace.data, dtype=float))
            return out
        return build_zrt_components(raws)


    def run(
        self,
        skip_missing_obs: bool = True,
        resume: bool = True,
        force: bool = False,
        visualize: bool = False,
        plot_waveforms: bool = True,
        waveform_event_ids: Optional[Sequence[str]] = None,
        n_stations: int = 6,
        n_section: int = 20,
    ) -> None:
        """测量全部就绪事件，写 CSV 与评分报告。出图请走 5_13 或 visualize=True。"""
        todo = [e for e in self.events if e.ready] if skip_missing_obs \
            else list(self.events)
        if not todo:
            print('\n❌ 没有具备观测的事件')
            return
        print(f'\n将评估 {len(todo)}/{len(self.events)} 个事件  '
              f'（resume={resume}, workers={self.config.n_workers}）')
        print('⏳ 预统计待测台站×模型（已有 CSV 不计）…')
        n_total = self._count_work_units(todo)
        print(f'   新工作量: {n_total} 台站×模型')

        for name in ('obspy', 'obspy.core', 'obspy.io', 'obspy.signal'):
            logging.getLogger(name).setLevel(logging.ERROR)

        with tqdm(
            total=max(n_total, 1), desc='Assessment', unit='sta',
            dynamic_ncols=True, mininterval=0.5,
        ) as pbar:
            self._pbar = pbar
            for ev in todo:
                self.assess_event(ev, resume=resume, force=force)
            self._pbar = None

        if not self.results:
            print('❌ 无有效测量')
            return
        self.write_aggregate_csv()
        self.write_report()
        print(f'\n✅ 测量完成 → {self.output_dir}')
        if visualize:
            self.visualize(
                plot_waveforms=plot_waveforms,
                waveform_event_ids=waveform_event_ids,
                n_stations=n_stations,
                n_section=n_section,
            )

    def visualize(
        self,
        plot_waveforms: bool = True,
        waveform_event_ids: Optional[Sequence[str]] = None,
        n_stations: int = 6,
        n_section: int = 20,
        paper_dir: Optional[Path] = None,
    ) -> None:
        """延迟加载 5_13，避免计算模块在无 GMT 环境里被出图拖死。"""
        viz_py = PROJECT_ROOT / '5_Visualization' / '5_13_Waveform_assessment.py'
        if not viz_py.is_file():
            raise FileNotFoundError(f'找不到波形评估可视化模块: {viz_py}')
        name = 'eastasia_waveform_assess_viz'
        if name in sys.modules:
            mod = sys.modules[name]
        else:
            spec = importlib.util.spec_from_file_location(name, viz_py)
            if spec is None or spec.loader is None:
                raise ImportError(f'无法加载: {viz_py}')
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        viz = self
        viz.__class__ = mod.WaveformAssessmentViz
        viz.plot_all(
            plot_waveforms=plot_waveforms,
            waveform_event_ids=waveform_event_ids,
            n_stations=n_stations,
            n_section=n_section,
        )
        if paper_dir is not None:
            viz.plot_paper_figures(paper_dir=paper_dir)


EastAsiaModelAssessment = WaveformAssessment


def default_paths() -> Dict[str, Path]:
    """默认路径：合成仍在 specfem/waveform，结果也写回那里以便接续已有 CSV。"""
    wf = PROJECT_ROOT / 'specfem' / 'waveform'
    return {
        'sac_root': wf / 'sac_30',
        'obs_root': PROJECT_ROOT / 'data' / 'events' / 'EastAsia_model' / 'data_obs',
        'events_list': PROJECT_ROOT / 'specfem' / 'events_eastasia_model_30' / 'events.list',
        'events_csv': (PROJECT_ROOT / 'data' / 'events' / 'eastasia_model_30events'
                       / 'eastasia_model_30events_id_to_dir.csv'),
        'cmt_dir': PROJECT_ROOT / 'data' / 'events' / 'EastAsia_model' / 'src_rec',
        'stations': PROJECT_ROOT / 'specfem' / 'events_eastasia_model_30' / 'STATIONS',
        'stations_catalog': (PROJECT_ROOT / 'data' / 'events' / 'EastAsia_model'
                             / 'catalog' / 'stations_catalog.csv'),
        'output': wf / 'figures_eastasia_30',
    }


def build_assessment(args: argparse.Namespace) -> 'WaveformAssessment':
    """按 CLI 参数构造评估对象。"""
    events = load_eastasia_events(
        args.events_list, args.events_csv, args.cmt_dir, args.obs_root)
    if args.events:
        want = set(args.events)
        events = [e for e in events if e.event_id in want]
    if not events:
        raise SystemExit('事件列表为空')
    assess = WaveformAssessment(
        sac_root=args.sac_root,
        obs_root=args.obs_root,
        events=events,
        models=args.models,
        output_dir=args.output_dir,
        stations_file=args.stations_file,
        stations_catalog=args.stations_catalog,
    )
    assess.config.weighting['mode'] = args.wr_mode
    if args.no_common_region:
        assess.config.use_common_region = False
        assess.events = list(assess.events_all)
        assess._dropped_events = []
    assess.config.min_dist_deg = float(args.min_dist)
    assess.config.weighting['dist_edges'] = (
        max(float(args.min_dist), 0.0), 8.0, 16.0, 24.0, 32.0, 180.0)
    if args.workers is not None:
        assess.config.n_workers = max(1, int(args.workers))
    if args.no_phase_windows:
        assess.config.phase_windows['enabled'] = False
    if args.bands:
        assess.config.bands = [
            b for b in assess.config.bands if b[0] in args.bands
        ]
    return assess


def add_io_args(ap: argparse.ArgumentParser) -> None:
    """合成 / 观测 / 事件路径（3_4 与 5_13 共用）。"""
    dft = default_paths()
    ap.add_argument('--sac-root', type=Path, default=dft['sac_root'])
    ap.add_argument('--obs-root', type=Path, default=dft['obs_root'])
    ap.add_argument('--events-list', type=Path, default=dft['events_list'])
    ap.add_argument('--events-csv', type=Path, default=dft['events_csv'])
    ap.add_argument('--cmt-dir', type=Path, default=dft['cmt_dir'])
    ap.add_argument('--stations-file', type=Path, default=dft['stations'])
    ap.add_argument('--stations-catalog', type=Path, default=dft['stations_catalog'])
    ap.add_argument('--output-dir', type=Path, default=dft['output'])
    ap.add_argument('--events', nargs='+', default=None)
    ap.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS))
    ap.add_argument('--wr-mode', choices=('geo', 'azdist'), default='geo')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--no-phase-windows', action='store_true')
    ap.add_argument('--bands', nargs='+', default=None)
    ap.add_argument('--no-common-region', action='store_true')
    ap.add_argument('--min-dist', type=float, default=3.0)


def main() -> None:
    print('🎯 EASTASIA-FWI 3_4 波形评估（测量）')
    print('=' * 70)
    ap = argparse.ArgumentParser(
        description='数据空间：合成 vs 观测的测量与评分（出图见 5_13）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 3_4_Waveform_assessment.py
  python 3_4_Waveform_assessment.py --force
  python 3_4_Waveform_assessment.py --events C201505120705A --force
  python 3_4_Waveform_assessment.py --visualize   # 测完后调用 5_13
        """,
    )
    add_io_args(ap)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--force', action='store_true', help='忽略已有 CSV，重测')
    ap.add_argument('--no-resume', action='store_true')
    ap.add_argument('--visualize', action='store_true',
                    help='测量结束后调用 5_13 出图')
    ap.add_argument('--no-waveforms', action='store_true')
    ap.add_argument('--waveform-stations', type=int, default=6)
    ap.add_argument('--waveform-section', type=int, default=50)
    args = ap.parse_args()
    try:
        assess = build_assessment(args)
        if args.check:
            assess.check_readiness()
        else:
            assess.run(
                skip_missing_obs=True,
                resume=not args.no_resume,
                force=args.force,
                visualize=args.visualize,
                plot_waveforms=not args.no_waveforms,
                waveform_event_ids=args.events,
                n_stations=args.waveform_stations,
                n_section=args.waveform_section,
            )
    except KeyboardInterrupt:
        print('\n⚠️ 用户中断')
        sys.exit(130)
    except Exception as exc:
        print(f'\n❌ 失败: {exc}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print('=' * 70)


if __name__ == '__main__':
    main()
