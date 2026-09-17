"""
EASTASIA-FWI 5_13 波形评估可视化
================================

功能描述:
- 读取 3_4 写出的 meas_*.csv，画 ea-* 与 Zhou 2021 论文图
- 论文图函数在本文件内；不 import specfem 里的评估 / 出图脚本
- 只读 specfem/waveform 下的 SAC 与 CSV

作者: EASTASIA-FWI Team
日期: 2026-09-17
版本: v1.1
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pygmt
from obspy import Trace
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SAC_ROOT_DEFAULT = PROJECT_ROOT / 'specfem' / 'waveform'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from config.base_config import BaseConfig  # noqa: E402

warnings.filterwarnings('ignore')


def _load_assess_mod() -> Any:
    """加载 3_4（文件名以数字开头）。若 3_4 作为 __main__ 已加载则复用。"""
    main = sys.modules.get('__main__')
    main_file = Path(getattr(main, '__file__', '') or '')
    if (main is not None and main_file.name == '3_4_Waveform_assessment.py'
            and hasattr(main, 'WaveformAssessment')):
        return main
    path = PROJECT_ROOT / '3_Data_space_simulation' / '3_4_Waveform_assessment.py'
    if not path.is_file():
        raise FileNotFoundError(f'找不到计算模块: {path}')
    name = 'eastasia_waveform_assess_compute'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'无法加载: {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_assess = _load_assess_mod()
WaveformAssessment = _assess.WaveformAssessment
EastAsiaConfig = _assess.EastAsiaConfig
short_model = _assess.short_model
model_color = _assess.model_color
provider_of = _assess.provider_of
strip_net_suffix = _assess.strip_net_suffix
PLOT_COMPONENTS = _assess.PLOT_COMPONENTS
CHUJIE_EVENT = _assess.CHUJIE_EVENT
CHUJIE_DIR = _assess.CHUJIE_DIR
DEFAULT_PLOT_EVENTS = _assess.DEFAULT_PLOT_EVENTS
VIS_DATA = _assess.VIS_DATA
SynDirIndex = _assess.SynDirIndex
default_paths = _assess.default_paths
load_eastasia_events = _assess.load_eastasia_events
OBSERVED_LABEL = _assess.OBSERVED_LABEL
phase_time_window = _assess.phase_time_window
taup_arrivals = _assess.taup_arrivals
MODEL_COLORS = _assess.MODEL_COLORS
filter_band = _assess.filter_band
read_raw = _assess.read_raw
read_observed = _assess.read_observed
get_taup_model = _assess.get_taup_model
to_common_axis = _assess.to_common_axis
EventSpec = _assess.EventSpec
Measurement = _assess.Measurement


# ── Zhou 2021 论文图（自 plot_zhou_paper_figs 并入）────────────

TOP3_REDS: Tuple[str, str, str] = ('#7B1113', '#C0392B', '#E38B8B')
BAR_OTHER = '#D0D0D0'
BAR_EDGE = '#4A4A4A'

PAPER_LABELS: Dict[str, str] = {
    'BASE': 'BASE',
    'S362ANI_c1': 'S362',
    'FWEA23_c1': 'FWEA23',
    'EARA2024_c1': 'EARA24',
    'SinoScope_c1': 'SinoSc',
    'SinoScope_sem_c1': 'SinoSEM',
}

ZHOU_CATS: Tuple[Tuple[str, str], ...] = (
    ('body-Z', 'P–SV Z'),
    ('body-R', 'P–SV R'),
    ('body-T', 'SH T'),
    ('surf-Z', 'Rayleigh Z'),
    ('surf-R', 'Rayleigh R'),
    ('surf-T', 'Love T'),
)

# 代表台：浅源尼泊尔事件，20–40 s / 40–120 s 各一张
PAPER_ZRT: Tuple[Tuple[str, str, str, str], ...] = (
    ('C201505120705A', 'IN', 'PBA', '20-40s'),
    ('C201505120705A', 'CB', 'GOM', '40-120s'),
)

DIST_EDGES: Tuple[float, ...] = (3.0, 10.0, 20.0, 30.0, 45.0, 90.0)

SHEAR_PHASES = frozenset({'SV', 'SH', 'Rayleigh', 'Love'})


def paper_label(model: str) -> str:
    """论文轴上的短模型名。"""
    return PAPER_LABELS.get(model, model.replace('_c1', ''))


def paper_tick(model: str) -> str:
    """窄面板用的更短刻度名。"""
    return {
        'BASE': 'BASE',
        'S362ANI_c1': 'S362',
        'FWEA23_c1': 'FWEA',
        'EARA2024_c1': 'EARA',
        'SinoScope_c1': 'Sino',
        'SinoScope_sem_c1': 'SEM',
    }.get(model, paper_label(model))


def band_tex(band: str) -> str:
    """20-40s → 20–40 s。"""
    core = band[:-1] if band.endswith('s') else band
    return core.replace('-', '–') + ' s'


def model_color(_assess: Any, name: str) -> str:
    """沿用评估脚本的模型色。"""
    if name == 'Chujie':
        return '#d62728'
    return MODEL_COLORS.get(name, '#8c564b')


@contextmanager
def paper_rc() -> Iterable[None]:
    """GJI 风格：Helvetica、内刻度、完整边框。"""
    rc = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'axes.linewidth': 0.7,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'legend.frameon': False,
        'legend.fontsize': 8,
        'mathtext.fontset': 'dejavusans',
        'axes.unicode_minus': False,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
    }
    with plt.rc_context(rc):
        yield


def save_paper(fig: Any, paper_dir: Path, stem: str) -> Path:
    """论文图只出 jpg，300 dpi，白底。"""
    paper_dir.mkdir(parents=True, exist_ok=True)
    path = paper_dir / f'{stem}.jpg'
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white',
                pad_inches=0.06)
    plt.close(fig)
    print(f'   🖼️  paper_figs/{path.name}')
    return path


def save_paper_gmt(fig: Any, paper_dir: Path, stem: str) -> Path:
    """PyGMT 论文图。"""
    paper_dir.mkdir(parents=True, exist_ok=True)
    path = paper_dir / f'{stem}.jpg'
    fig.savefig(str(path), dpi=300, crop=True)
    print(f'   🖼️  paper_figs/{path.name}')
    return path


def panel_letter(ax: Any, letter: str, x: float = -0.18, y: float = 1.06) -> None:
    """子图角标 (a)、(b)…"""
    ax.text(
        x, y, f'({letter})', transform=ax.transAxes,
        fontsize=11, fontweight='bold', va='bottom', ha='left',
        clip_on=False,
    )


def letters(n: int) -> List[str]:
    return [chr(ord('a') + i) for i in range(n)]


def top3_colors(vals: Sequence[float], higher_better: bool) -> List[str]:
    """该面板内前三名用红，其余灰。"""
    colors = [BAR_OTHER] * len(vals)
    ranked = [(i, float(v)) for i, v in enumerate(vals) if np.isfinite(v)]
    ranked.sort(key=lambda iv: iv[1], reverse=higher_better)
    for rank, (i, _) in enumerate(ranked[:3]):
        colors[i] = TOP3_REDS[rank]
    return colors


def top3_legend() -> List[Patch]:
    return [
        Patch(facecolor=TOP3_REDS[0], edgecolor=BAR_EDGE, linewidth=0.4,
              label='1st'),
        Patch(facecolor=TOP3_REDS[1], edgecolor=BAR_EDGE, linewidth=0.4,
              label='2nd'),
        Patch(facecolor=TOP3_REDS[2], edgecolor=BAR_EDGE, linewidth=0.4,
              label='3rd'),
        Patch(facecolor=BAR_OTHER, edgecolor=BAR_EDGE, linewidth=0.4,
              label='Other'),
    ]


def paper_models(assess: Any, common: Optional[Dict[str, Any]] = None) -> List[str]:
    if common is None:
        common = assess._common_by_event()
    return [m for m in assess.models
            if any(m in mmap for mmap in common.values())]


def ygrid(ax: Any) -> None:
    ax.grid(True, axis='y', linestyle=':', linewidth=0.45, color='0.75',
            zorder=0)
    ax.set_axisbelow(True)


def event_equal_metric(
    idx: Dict[str, Any],
    band: str,
    model: str,
    key: str,
) -> float:
    """事件等权均值（与 ea-1 同一口径）。"""
    ev_map = idx.get(band, {})
    vals = [row[key] for er in ev_map.values()
            for m, row in er.items()
            if m == model and np.isfinite(row[key])]
    return float(np.mean(vals)) if vals else np.nan


def category_frac(
    assess: Any,
    common: Dict[str, Any],
    model: str,
    band: str,
    cls: str,
    thr: float,
) -> float:
    """事件等权的 NZCC>thr 占比。"""
    ev_frac: List[float] = []
    for mmap in common.values():
        if model not in mmap:
            continue
        rows = [m for key, m in mmap[model].items()
                if key[3] == band
                and assess._phase_class(m) == cls
                and np.isfinite(m.nzcc)]
        if rows:
            ev_frac.append(sum(1 for m in rows if m.nzcc > thr) / len(rows))
    return float(np.mean(ev_frac)) if ev_frac else np.nan


def dist_frac(
    assess: Any,
    common: Dict[str, Any],
    model: str,
    band: str,
    cls: str,
    d0: float,
    d1: float,
    thr: float,
) -> float:
    """该震中距箱内 NZCC>thr 占比（窗数合计，非事件等权）。"""
    n_ok = n_all = 0
    for mmap in common.values():
        if model not in mmap:
            continue
        for key, m in mmap[model].items():
            if key[3] != band or assess._phase_class(m) != cls:
                continue
            if not (np.isfinite(m.dist_deg) and np.isfinite(m.nzcc)):
                continue
            if not (d0 <= float(m.dist_deg) < d1):
                continue
            n_all += 1
            if m.nzcc > thr:
                n_ok += 1
    return n_ok / n_all if n_all else np.nan


def station_mean_dt(
    assess: Any,
    common: Dict[str, Any],
    model: str,
    band: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    台站平均剪切波走时残差（Zhou Fig.8）。

    各模型自己通过 CC/SNR 的窗；相位取 SV/SH/Rayleigh/Love。
    """
    buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for mmap in common.values():
        if model not in mmap:
            continue
        for k in assess._accepted_keys(mmap[model], band):
            m = mmap[model][k]
            if m.phase not in SHEAR_PHASES:
                continue
            if not np.isfinite(m.dt_shift):
                continue
            buckets[(m.network, m.station)].append(float(m.dt_shift))
    lons, lats, vals = [], [], []
    for (net, sta), dts in buckets.items():
        coord = assess._station_coord(net, sta)
        if coord is None or not assess._in_common_region(coord.lat, coord.lon):
            continue
        lons.append(coord.lon)
        lats.append(coord.lat)
        vals.append(float(np.mean(dts)))
    return (np.asarray(lons, dtype=float),
            np.asarray(lats, dtype=float),
            np.asarray(vals, dtype=float))


def render_paper_figures(
    assess: Any,
    paper_dir: Optional[Path] = None,
) -> Path:
    """从已加载的 assess.results 画出全部论文图。"""
    out = Path(paper_dir) if paper_dir is not None else DEFAULT_PAPER_DIR
    out.mkdir(parents=True, exist_ok=True)
    common = assess._common_by_event()
    if not common:
        print('❌ 无公共窗口，无法出论文图')
        return out
    models = paper_models(assess, common)
    print(f'\n{"=" * 64}')
    print(f'论文图  →  {out}')
    print(f'  {len(common)} events × {len(models)} models   Zhou 2021 layout')
    print('=' * 64)
    write_readme(out)
    jobs = [
        ('W1', lambda: plot_w1_geometry(assess, out)),
        ('W2', lambda: plot_w2_misfit_bars(assess, out, common, models)),
        ('W3', lambda: plot_w3_dt_histograms(assess, out, common, models)),
        ('W4', lambda: plot_w4_dt_maps(assess, out, common, models)),
        ('W5', lambda: plot_w5_categories(assess, out, common, models)),
        ('W6', lambda: plot_w6_predict_vs_dist(assess, out, common, models)),
        ('W7', lambda: plot_w7_zrt(assess, out, models)),
        ('W8', lambda: plot_w8_rank(assess, out, models)),
    ]
    with paper_rc():
        for name, fn in jobs:
            try:
                fn()
            except Exception as exc:
                print(f'   ⚠️  {name} 失败: {exc}')
    print(f'\n✅ 论文图完成 → {out}')
    return out


def write_readme(paper_dir: Path) -> None:
    """对照 Zhou 2021 图号，方便写图注。"""
    text = """# Waveform-assessment paper figures

Zhou et al. (2021, GJI 228, 1392–1414) layout, East Asia 22-event set.
Same windows as the scoring: common region, Δ ≥ 3°, SV/SH only if Δ ≥ 10°,
CC ≥ 0.7 and SNR ≥ 4 for dT. JPG, 300 dpi. No remasurement.

| File | Zhou analogue | Content |
| --- | --- | --- |
| `W1_geometry.jpg` | Fig. 3 | Events (CMT) + stations used in scoring (22 events; caption the station count) |
| `W2_misfit_bars.jpg` | Fig. 9 | χT, χA, χF, χNZCC; top-3 in red |
| `W3_dt_histograms.jpg` | Fig. 7 | Traveltime-shift histograms |
| `W4_dt_maps.jpg` | Fig. 8 | Station-mean shear-wave dT, 20–40 s |
| `W5_category_predictability.jpg` | Fig. 10 | NZCC > 0.7 in six categories |
| `W6_predict_vs_distance.jpg` | Fig. 11 | Predictability versus epicentral distance |
| `W7a_zrt_20-40s_IN.PBA.jpg` | Fig. 6 | Single-station ZRT, Nepal event |
| `W7b_zrt_40-120s_CB.GOM.jpg` | Fig. 6 | Single-station ZRT, longer period |
| `W8_rank_heatmap.jpg` | — | Per-event \\|dT\\| rank (our extra plate) |

Red bars are the three best models **in that panel**, not a fixed model colour.
"""
    path = paper_dir / 'README.md'
    path.write_text(text, encoding='utf-8')
    print(f'   📄 paper_figs/{path.name}')


def plot_w1_geometry(assess: Any, paper_dir: Path) -> None:
    """Zhou Fig.3：共同区地形 + 评分台站 + CMT，不加长标题。"""
    region = assess._map_region()
    assess._setup_pygmt()
    pygmt.config(
        FONT_ANNOT_PRIMARY='10p,Helvetica,black',
        FONT_LABEL='11p,Helvetica,black',
        MAP_FRAME_PEN='0.7p,black',
        MAP_TICK_LENGTH_PRIMARY='5p',
    )
    fig = pygmt.Figure()
    proj = 'M18c'
    assess._draw_basemap(
        fig, region, proj,
        frame=assess._map_frame(None),
        topo=True, geology=False,
    )
    lons, lats = assess._used_station_xy()
    if lons:
        fig.plot(
            x=lons, y=lats, style='c0.07c',
            fill='107/174/214', transparency=20, pen='0.03p,64/64/64',
        )
    depth_cpt = assess._depth_cpt()
    for ev in assess.events:
        assess._plot_event_meca(fig, ev, depth_cpt)
    fig.colorbar(
        cmap=depth_cpt,
        position='JMR+o0.50c/0c+w8.0c/0.34c',
        frame='xa100f50+lDepth (km)',
    )
    save_paper_gmt(fig, paper_dir, 'W1_geometry')


def plot_w2_misfit_bars(
    assess: Any,
    paper_dir: Path,
    common: Dict[str, Any],
    models: Sequence[str],
) -> None:
    """Zhou Fig.9：两频段 × 四类 misfit，前三名红色。"""
    _ = common
    rows = assess.aggregate_table()
    if not rows:
        return
    idx = assess._rows_index(rows)
    bands = [b[0] for b in assess.config.bands]
    metrics = [
        (r'Traveltime  $\chi^{T}$', 'dt_abs_score'),
        (r'Amplitude  $\chi^{A}$', 'dlnA_abs_score'),
        (r'Waveform  $\chi^{F}$', 'rel_l2_score'),
        (r'$1-\mathrm{NZCC}$  $\chi^{N}$', 'one_minus_nzcc_score'),
    ]
    fig, axes = plt.subplots(
        len(bands), len(metrics),
        figsize=(12.0, 5.6),
        sharex=True, squeeze=False,
    )
    x = np.arange(len(models))
    labs = letters(len(bands) * len(metrics))
    k = 0
    for i, band in enumerate(bands):
        for j, (title, key) in enumerate(metrics):
            ax = axes[i, j]
            vals = [event_equal_metric(idx, band, m, key) for m in models]
            colors = top3_colors(vals, higher_better=False)
            ax.bar(
                x, vals, width=0.68, color=colors,
                edgecolor=BAR_EDGE, linewidth=0.45, zorder=3,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [paper_label(m) for m in models],
                rotation=32, ha='right', fontsize=7.5,
            )
            if i == 0:
                ax.set_title(title, fontsize=10, pad=6)
            if j == 0:
                ax.set_ylabel(f'{band_tex(band)}\nMisfit')
            ygrid(ax)
            panel_letter(ax, labs[k], x=-0.16, y=1.04)
            k += 1
            ymax = np.nanmax(vals) if np.any(np.isfinite(vals)) else 1.0
            ax.set_ylim(0.0, float(ymax) * 1.18 if np.isfinite(ymax) else 1.0)
    fig.legend(
        handles=top3_legend(), loc='lower center', ncol=4,
        bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=8,
    )
    fig.tight_layout()
    save_paper(fig, paper_dir, 'W2_misfit_bars')


def plot_w3_dt_histograms(
    assess: Any,
    paper_dir: Path,
    common: Dict[str, Any],
    models: Sequence[str],
) -> None:
    """Zhou Fig.7：密度直方图，避免计数轴被 BASE 尖峰压扁。"""
    bands = [b[0] for b in assess.config.bands]
    fig, axes = plt.subplots(
        len(bands), len(models),
        figsize=(11.4, 5.4),
        sharex=True, sharey=True, squeeze=False,
    )
    bins = np.linspace(-10.0, 10.0, 41)
    labs = letters(len(bands) * len(models))
    k = 0
    for i, band in enumerate(bands):
        for j, model in enumerate(models):
            ax = axes[i, j]
            vals: List[float] = []
            for mmap in common.values():
                if model not in mmap:
                    continue
                for key in assess._accepted_keys(mmap[model], band):
                    dt = mmap[model][key].dt_shift
                    if np.isfinite(dt):
                        vals.append(float(dt))
            if vals:
                ax.hist(
                    vals, bins=bins, range=(-10.0, 10.0), density=True,
                    color='#6B7B8C', edgecolor='white', linewidth=0.25,
                    zorder=3,
                )
                med = float(np.median(vals))
                ax.axvline(med, color='#7B1113', lw=1.05, ls='--', zorder=4)
                ax.text(
                    0.96, 0.93,
                    f'n = {len(vals):,}\nmed = {med:+.2f} s',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=7, linespacing=1.35,
                    bbox=dict(boxstyle='round,pad=0.18', fc='white',
                              ec='0.82', lw=0.4, alpha=0.92),
                )
            ax.axvline(0.0, color='0.45', lw=0.55, zorder=2)
            if i == 0:
                ax.set_title(paper_label(model), fontsize=10, pad=5)
            if j == 0:
                ax.set_ylabel(f'{band_tex(band)}\nDensity')
            if i == len(bands) - 1:
                ax.set_xlabel(r'$\Delta T$ (s)')
            ax.set_xlim(-10.0, 10.0)
            ygrid(ax)
            panel_letter(ax, labs[k], x=-0.08, y=1.05)
            k += 1
    fig.tight_layout()
    save_paper(fig, paper_dir, 'W3_dt_histograms')


def plot_w4_dt_maps(
    assess: Any,
    paper_dir: Path,
    common: Dict[str, Any],
    models: Sequence[str],
) -> None:
    """Zhou Fig.8：20–40 s 台站平均剪切波 dT。"""
    band = '20-40s'
    packed = [(m, station_mean_dt(assess, common, m, band)) for m in models]
    all_v = np.concatenate([v for _m, (_x, _y, v) in packed if v.size] or [np.array([1.0])])
    vmax = float(np.nanpercentile(np.abs(all_v), 95)) if all_v.size else 4.0
    vmax = float(min(max(vmax, 2.0), 6.0))
    region = assess._map_region()
    assess._setup_pygmt()
    pygmt.config(
        FONT_ANNOT_PRIMARY='8p,Helvetica,black',
        FONT_LABEL='9p,Helvetica,black',
        FONT_TITLE='11p,Helvetica-Bold,black',
        MAP_FRAME_PEN='0.6p,black',
        MAP_TITLE_OFFSET='5p',
    )
    cpt = paper_dir / '_tmp_dt.cpt'
    pygmt.makecpt(
        cmap='polar', series=[-vmax, vmax, max(vmax / 20.0, 0.1)],
        continuous=True, output=str(cpt),
    )
    ncols = 3
    nrows = int(np.ceil(len(packed) / float(ncols)))
    panel_w = 7.2
    fig = pygmt.Figure()
    with fig.subplot(
        nrows=nrows, ncols=ncols,
        figsize=(f'{panel_w * ncols + 1.2:.1f}c', f'{7.0 * nrows + 1.2:.1f}c'),
        margins=['0.32c', '0.42c'],
    ):
        n_sta = 0
        for i, (model, (lons, lats, vals)) in enumerate(packed):
            row, col = divmod(i, ncols)
            with fig.set_panel(panel=i):
                proj = f'M{panel_w}c'
                assess._draw_basemap(
                    fig, region, proj,
                    frame=assess._map_frame(
                        paper_label(model),
                        left=(col == 0),
                        bottom=(row == nrows - 1),
                    ),
                    topo=False, geology=False,
                )
                if lons.size:
                    fig.plot(
                        data=np.column_stack([lons, lats, vals]),
                        style='c0.055c', cmap=str(cpt), transparency=25,
                    )
                    n_sta = max(n_sta, int(lons.size))
    fig.colorbar(
        cmap=str(cpt),
        position='JBC+o0c/0.55c+w16c/0.32c+h',
        frame='xa2f1+ldT(s)',
    )
    save_paper_gmt(fig, paper_dir, 'W4_dt_maps')
    if cpt.is_file():
        cpt.unlink()


def plot_w5_categories(
    assess: Any,
    paper_dir: Path,
    common: Dict[str, Any],
    models: Sequence[str],
) -> None:
    """Zhou Fig.10：2 频段 × 6 类，每格独立标前三名。"""
    bands = [b[0] for b in assess.config.bands]
    thr = float(assess.config.nzcc_predict_threshold)
    fig, axes = plt.subplots(
        len(bands), len(ZHOU_CATS),
        figsize=(13.2, 5.7),
        sharex=True, sharey=True, squeeze=False,
    )
    x = np.arange(len(models))
    labs = letters(len(bands) * len(ZHOU_CATS))
    k = 0
    for i, band in enumerate(bands):
        for j, (cls, clab) in enumerate(ZHOU_CATS):
            ax = axes[i, j]
            vals = [category_frac(assess, common, m, band, cls, thr)
                    for m in models]
            colors = top3_colors(vals, higher_better=True)
            ax.bar(
                x, vals, width=0.72, color=colors,
                edgecolor=BAR_EDGE, linewidth=0.4, zorder=3,
            )
            ax.set_ylim(0.0, 1.05)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [paper_tick(m) for m in models],
                rotation=90, ha='center', va='top', fontsize=7,
            )
            if i == 0:
                ax.set_title(clab, fontsize=9, pad=5)
            if j == 0:
                ax.set_ylabel(f'{band_tex(band)}\nNZCC > {thr:g}')
            ygrid(ax)
            panel_letter(ax, labs[k], x=-0.10, y=1.05)
            k += 1
    fig.legend(
        handles=top3_legend(), loc='lower center', ncol=4,
        bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=8,
    )
    fig.tight_layout()
    save_paper(fig, paper_dir, 'W5_category_predictability')


def plot_w6_predict_vs_dist(
    assess: Any,
    paper_dir: Path,
    common: Dict[str, Any],
    models: Sequence[str],
) -> None:
    """Zhou Fig.11：可预测性随震中距。"""
    bands = [b[0] for b in assess.config.bands]
    edges = DIST_EDGES
    labels = [f'{edges[i]:.0f}–{edges[i + 1]:.0f}' for i in range(len(edges) - 1)]
    xc = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
    thr = float(assess.config.nzcc_predict_threshold)
    markers = ('o', 's', 'D', '^', 'v', 'P')
    fig, axes = plt.subplots(
        len(bands), len(ZHOU_CATS),
        figsize=(12.0, 5.6),
        sharex=True, sharey=True, squeeze=False,
    )
    labs = letters(len(bands) * len(ZHOU_CATS))
    k = 0
    for i, band in enumerate(bands):
        for j, (cls, clab) in enumerate(ZHOU_CATS):
            ax = axes[i, j]
            for im, model in enumerate(models):
                fracs = [dist_frac(assess, common, model, band, cls,
                                   edges[b], edges[b + 1], thr)
                         for b in range(len(edges) - 1)]
                ax.plot(
                    xc, fracs, color=model_color(assess, model),
                    lw=1.45, marker=markers[im % len(markers)],
                    ms=4.2, mew=0.4, mec='white',
                    label=paper_label(model), zorder=3,
                )
            ax.set_ylim(0.0, 1.05)
            ax.set_xlim(xc[0] - 3.0, xc[-1] + 8.0)
            ax.grid(True, linestyle=':', linewidth=0.45, color='0.75', zorder=0)
            if i == 0:
                ax.set_title(clab, fontsize=9, pad=5)
            if j == 0:
                ax.set_ylabel(f'{band_tex(band)}\nNZCC > {thr:g}')
            if i == len(bands) - 1:
                ax.set_xticks(xc)
                ax.set_xticklabels(labels, fontsize=6.5, rotation=35)
                ax.set_xlabel('Δ (°)')
            panel_letter(ax, labs[k], x=-0.08, y=1.05)
            k += 1
    handles, labels_h = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles, labels_h, loc='lower center', ncol=len(models),
        bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=8,
        handlelength=1.6,
    )
    fig.tight_layout()
    save_paper(fig, paper_dir, 'W6_predict_vs_distance')


def _zrt_xlim(dist_deg: float, band: str, vg: Dict[str, Any]) -> Tuple[float, float]:
    """按群速度裁到有效能量，避免 1800 s 空轴。"""
    vmin, vmax = vg.get(band, (2.8, 4.2))
    dist_km = float(dist_deg) * 111.195
    t0 = max(0.0, dist_km / (float(vmax) + 1.5) - 90.0)
    t1 = dist_km / max(float(vmin) - 0.25, 1.6) + 160.0
    return t0, min(t1, 1400.0)


def plot_w7_zrt(
    assess: Any,
    paper_dir: Path,
    models: Sequence[str],
) -> None:
    """Zhou Fig.6：单台 ZRT，观测黑、合成彩。"""
    by_id = {e.event_id: e for e in assess.events}
    for eid, net, sta, band in PAPER_ZRT:
        ev = by_id.get(eid)
        if ev is None:
            print(f'   ⚠️  W7 跳过 {eid}（不在共同区事件表）')
            continue
        try:
            _draw_paper_zrt(assess, ev, net, sta, band, models, paper_dir)
        except Exception as exc:
            print(f'   ⚠️  W7 {net}.{sta} {band} 失败: {exc}')


def _draw_paper_zrt(
    assess: Any,
    ev: Any,
    net: str,
    sta: str,
    band: str,
    models: Sequence[str],
    paper_dir: Path,
) -> None:
    tmin, tmax_p = assess._band_limits(band)
    pw = assess.config.phase_windows
    obs = assess._load_zrt_filtered(ev, net, sta, None, band, tmin, tmax_p)
    if obs is None:
        print(f'   ⚠️  W7 无观测 {ev.event_id} {net}.{sta}')
        return
    row_labs: List[str] = []
    filtered: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
        OBSERVED_LABEL: obs,
    }
    for model in models:
        syn = assess._load_zrt_filtered(ev, net, sta, model, band, tmin, tmax_p)
        if syn is None:
            continue
        filtered[model] = syn
        row_labs.append(model)
    if not row_labs:
        return

    comps = ('Z', 'R', 'T')
    phase_title = {
        'Z': 'P / SV / Rayleigh',
        'R': 'P / SV / Rayleigh',
        'T': 'SH / Love',
    }
    phase_list = {
        'Z': ['P', 'SV', 'Rayleigh'],
        'R': ['P', 'SV', 'Rayleigh'],
        'T': ['SH', 'Love'],
    }
    phase_color = {
        'P': '#F8E0E0', 'SV': '#DCE8F5', 'SH': '#DCE8F5',
        'Rayleigh': '#DCEFDC', 'Love': '#F7EBD3',
    }
    dist = assess._station_dist_deg(ev, net, sta)
    depth = float(ev.depth_km) if np.isfinite(ev.depth_km) else np.nan
    if np.isfinite(depth) and depth > 800.0:
        depth = depth / 1000.0
    arrivals = taup_arrivals(dist, depth, pw.get('taup_model', 'ak135')) \
        if pw.get('enabled', True) else {}

    col_norm: Dict[str, float] = {}
    for comp in comps:
        series = [obs[comp][1]] if comp in obs else []
        for lab in row_labs:
            syn = filtered[lab].get(comp)
            if syn is not None:
                series.append(syn[1])
        col_norm[comp] = assess._shared_peak_norm(*series)

    n_row = len(row_labs)
    fig, axes = plt.subplots(
        n_row, 3, figsize=(11.0, 1.28 * n_row + 1.15),
        sharex=True, squeeze=False,
    )
    x0, x1 = _zrt_xlim(dist if np.isfinite(dist) else 15.0, band,
                       assess.config.plot['vg'])
    for i, lab in enumerate(row_labs):
        for j, comp in enumerate(comps):
            ax = axes[i, j]
            syn = filtered[lab].get(comp)
            if syn is None:
                ax.set_visible(False)
                continue
            if pw.get('enabled', True):
                for phase in phase_list[comp]:
                    win = phase_time_window(
                        phase, band, tmin, tmax_p, dist, arrivals, pw)
                    if win is None:
                        continue
                    w0, w1, _tp = win
                    ax.axvspan(
                        w0, w1, color=phase_color.get(phase, '#f0f0f0'),
                        alpha=0.55, zorder=0, linewidth=0)
            t_s, d_s = syn
            norm = col_norm[comp]
            obs_c = obs.get(comp)
            if obs_c is not None:
                t_o, d_o = obs_c
                ax.plot(t_o, d_o / norm, color='k', lw=0.95, zorder=3)
            ax.plot(t_s, d_s / norm, color=model_color(assess, lab),
                    lw=0.85, alpha=0.95, zorder=4)
            ax.axhline(0.0, color='0.65', lw=0.35, zorder=1)
            ax.set_yticks([])
            ax.set_xlim(x0, x1)
            ax.grid(True, axis='x', linestyle=':', linewidth=0.4, color='0.8')
            if j == 0:
                ax.set_ylabel(
                    paper_label(lab), fontsize=9, rotation=0,
                    labelpad=26, va='center',
                )
            if i == 0:
                ax.set_title(f'{comp}    {phase_title[comp]}', fontsize=10)
                if j == 0:
                    panel_letter(ax, chr(ord('a') + j), x=-0.02, y=1.12)
                else:
                    panel_letter(ax, chr(ord('a') + j), x=-0.02, y=1.12)
            if i == n_row - 1:
                ax.set_xlabel('Time after origin (s)')
    mw = f'Mw {ev.mw:.1f}' if np.isfinite(ev.mw) else ''
    dep = f'{depth:.0f} km' if np.isfinite(depth) else ''
    dtxt = f'Δ = {dist:.1f}°' if np.isfinite(dist) else ''
    fig.suptitle(
        f'{net}.{sta}    {ev.event_id}    {band_tex(band)}    '
        f'{mw}    {dep}    {dtxt}',
        fontsize=11, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    tag = f'W7{"a" if band.startswith("20") else "b"}_zrt_{band}_{net}.{sta}'
    save_paper(fig, paper_dir, tag)
    assess._zrt_cache.clear()
    assess._z_cache.clear()


def plot_w8_rank(
    assess: Any,
    paper_dir: Path,
    models: Sequence[str],
) -> None:
    """逐事件 |dT| 排名，离散色（一名一色）。"""
    rows = assess.aggregate_table()
    if not rows:
        return
    idx = assess._rows_index(rows)
    events = [e.event_id for e in assess.events
              if any(r['event'] == e.event_id for r in rows)]
    bands = [b[0] for b in assess.config.bands]
    cmap = ListedColormap(
        ['#1a9850', '#66bd63', '#a6d96a', '#fee08b', '#fc8d59', '#d73027',
         '#8e0152'][:max(len(models), 2)]
    )
    bounds = np.arange(0.5, len(models) + 1.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    fig, axes = plt.subplots(
        1, len(bands),
        figsize=(8.8, 0.32 * len(events) + 1.8),
        squeeze=False,
    )
    for j, band in enumerate(bands):
        ax = axes[0, j]
        ev_map = idx.get(band, {})
        rank_mat = np.full((len(events), len(models)), np.nan)
        for i, eid in enumerate(events):
            scored = []
            for k, model in enumerate(models):
                row = ev_map.get(eid, {}).get(model)
                v = row['dt_abs_score'] if row is not None else np.nan
                scored.append((k, v))
            valid = [(k, v) for k, v in scored if np.isfinite(v)]
            valid.sort(key=lambda x: x[1])
            for rank, (k, _) in enumerate(valid, 1):
                rank_mat[i, k] = rank
        im = ax.imshow(rank_mat, aspect='auto', cmap=cmap, norm=norm)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([paper_label(m) for m in models],
                           rotation=32, ha='right', fontsize=8)
        ax.set_yticks(range(len(events)))
        ax.set_yticklabels(events, fontsize=6.5)
        ax.set_title(band_tex(band), fontsize=11, pad=6)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        for i in range(len(events)):
            for k in range(len(models)):
                v = rank_mat[i, k]
                if np.isfinite(v):
                    ax.text(
                        k, i, f'{int(v)}', ha='center', va='center',
                        fontsize=7, color='0.12' if v <= 3 else '0.08',
                    )
        panel_letter(ax, chr(ord('a') + j), x=-0.18, y=1.02)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=range(1, len(models) + 1))
        cbar.set_label('Rank (1 = best)', fontsize=8)
        cbar.ax.tick_params(labelsize=7, length=2)
    fig.tight_layout()
    save_paper(fig, paper_dir, 'W8_rank_heatmap')


class WaveformAssessmentViz(WaveformAssessment):
    """3_4 测量结果的可视化：ea-* 图 + Zhou 2021 论文图。"""

    def plot_summary(self) -> None:
        """频段 × 指标：分组柱状图（事件等权均值）。"""
        rows = self.aggregate_table()
        if not rows:
            return
        idx = self._rows_index(rows)
        models = [m for m in self.models if any(r['model'] == m for r in rows)]
        bands = [b[0] for b in self.config.bands]
        metrics = [
            ('|dT| score (s)', 'dt_abs_score', False),
            ('|dlnA| score', 'dlnA_abs_score', False),
            ('rel L2 score', 'rel_l2_score', False),
            ('1 − NZCC score', 'one_minus_nzcc_score', False),
        ]
        fig, axes = plt.subplots(
            len(bands), len(metrics),
            figsize=(3.4 * len(metrics), 2.9 * len(bands)),
            squeeze=False,
        )
        x = np.arange(len(models))
        for i, band in enumerate(bands):
            ev_map = idx.get(band, {})
            for j, (title, key, higher) in enumerate(metrics):
                ax = axes[i, j]
                vals = []
                for model in models:
                    pe = [row[key] for er in ev_map.values()
                          for m, row in er.items()
                          if m == model and np.isfinite(row[key])]
                    vals.append(float(np.mean(pe)) if pe else np.nan)
                colors = [model_color(m) for m in models]
                ax.bar(x, vals, color=colors, width=0.72, edgecolor='white', linewidth=0.4)
                if np.any(np.isfinite(vals)):
                    order = np.argsort(vals)
                    best = order[-1] if higher else order[0]
                    if np.isfinite(vals[int(best)]):
                        ax.bar([int(best)], [vals[int(best)]], width=0.72,
                               facecolor='none', edgecolor='k', linewidth=1.2)
                ax.set_xticks(x)
                ax.set_xticklabels([short_model(m) for m in models],
                                   rotation=30, ha='right', fontsize=8)
                if i == 0:
                    ax.set_title(title, fontsize=10)
                if j == 0:
                    ax.set_ylabel(band, fontsize=10)
                ax.grid(True, axis='y', alpha=0.25)
        fig.suptitle(
            'EastAsia_model assessment vs Observed  '
            f'(common region {self.common_region["lat_min"]:.0f}–'
            f'{self.common_region["lat_max"]:.0f}N, '
            f'{self.common_region["lon_min"]:.0f}–'
            f'{self.common_region["lon_max"]:.0f}E, '
            f'Δ≥{self.config.min_dist_deg:g}°, ZRT)\n'
            'common windows · CC/SNR accepted · 1° Wr + Wc · event-equal mean',
            fontsize=11, fontweight='bold',
        )
        fig.tight_layout()
        self._save(fig, 'ea-1_model_summary')


    def plot_rank_heatmap(self) -> None:
        """逐事件 |dT| 排名热图。"""
        rows = self.aggregate_table()
        if not rows:
            return
        idx = self._rows_index(rows)
        models = [m for m in self.models if any(r['model'] == m for r in rows)]
        events = [e.event_id for e in self.events
                  if any(r['event'] == e.event_id for r in rows)]
        if not models or not events:
            return
        bands = [b[0] for b in self.config.bands]
        fig, axes = plt.subplots(
            1, len(bands),
            figsize=(1.7 * len(models) + 2.4, 0.34 * len(events) + 2.0),
            squeeze=False,
        )
        for j, band in enumerate(bands):
            ax = axes[0, j]
            ev_map = idx.get(band, {})
            rank_mat = np.full((len(events), len(models)), np.nan)
            for i, eid in enumerate(events):
                scored = []
                for k, model in enumerate(models):
                    row = ev_map.get(eid, {}).get(model)
                    v = row['dt_abs_score'] if row is not None else np.nan
                    scored.append((k, v))
                valid = [(k, v) for k, v in scored if np.isfinite(v)]
                valid.sort(key=lambda x: x[1])
                for rank, (k, _) in enumerate(valid, 1):
                    rank_mat[i, k] = rank
            im = ax.imshow(rank_mat, aspect='auto', cmap='RdYlGn_r',
                           vmin=1, vmax=max(len(models), 2))
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([short_model(m) for m in models],
                               rotation=35, ha='right', fontsize=8)
            ax.set_yticks(range(len(events)))
            ax.set_yticklabels(events, fontsize=7)
            ax.set_title(f'|dT| rank   {band}', fontsize=10)
            for i in range(len(events)):
                for k in range(len(models)):
                    v = rank_mat[i, k]
                    if np.isfinite(v):
                        ax.text(k, i, f'{int(v)}', ha='center', va='center',
                                fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='rank (1=best)')
        fig.suptitle('Per-event model rank by balanced |dT| vs Observed',
                     fontsize=12, fontweight='bold')
        fig.tight_layout()
        self._save(fig, 'ea-2_rank_heatmap')


    def plot_network_groups(self) -> None:
        """按 CNSN / Hinet / IRIS / Other 拆开的 1−NZCC（事件等权，Z/R/T）。"""
        common = self._common_by_event()
        if not common:
            return
        bands = [b[0] for b in self.config.bands]
        groups = ('CNSN', 'Hinet', 'IRIS', 'Other')
        comps = [(c, self._preferred_phase(c)) for c in PLOT_COMPONENTS]
        fig, axes = plt.subplots(
            len(bands), len(comps),
            figsize=(4.4 * len(comps), 3.6 * len(bands)),
            squeeze=False)
        for i, band in enumerate(bands):
            for j, (comp, pref) in enumerate(comps):
                ax = axes[i, j]
                bucket: Dict[str, Dict[str, List[float]]] = {
                    m: {g: [] for g in groups} for m in self.models
                }
                for eid, mmap in common.items():
                    for model, mp in mmap.items():
                        by_g: Dict[str, List[float]] = {g: [] for g in groups}
                        for m in mp.values():
                            if m.band != band or m.component != comp:
                                continue
                            if m.phase not in (pref, 'full'):
                                continue
                            if not np.isfinite(m.nzcc):
                                continue
                            g = provider_of(m.network, self.providers, m.station)
                            by_g[g].append(1.0 - float(m.nzcc))
                        if model not in bucket:
                            continue
                        for g, vals in by_g.items():
                            if vals:
                                bucket[model][g].append(float(np.median(vals)))
                x = np.arange(len(groups))
                width = 0.12
                shown = [m for m in self.models if m in bucket]
                for k, model in enumerate(shown):
                    means = [float(np.mean(bucket[model][g])) if bucket[model][g]
                             else np.nan for g in groups]
                    ax.bar(x + (k - 0.5 * (len(shown) - 1)) * width, means,
                           width=width, color=model_color(model),
                           label=short_model(model), edgecolor='white',
                           linewidth=0.3)
                ax.set_xticks(x)
                ax.set_xticklabels(groups, fontsize=8)
                if j == 0:
                    ax.set_ylabel(f'{band}\n1 − NZCC', fontsize=9)
                ax.set_title(f'{comp} / {pref}', fontsize=10)
                ax.grid(True, axis='y', alpha=0.25)
                if i == 0 and j == len(comps) - 1:
                    ax.legend(fontsize=7, loc='upper right')
        fig.suptitle(
            'Network-family breakdown  Z/R/T  '
            '(dense CNSN/Hinet must not hide IRIS)',
            fontsize=12, fontweight='bold')
        fig.tight_layout()
        self._save(fig, 'ea-5_network_groups')


    def plot_misfit_maps(
        self,
        band: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> None:
        """代表事件：各模型 NZCC 1° 均值图（Z/R/T）。"""
        _ = phase
        bands = [band] if band else [b[0] for b in self.config.bands]
        for name in bands:
            for comp in PLOT_COMPONENTS:
                self._plot_misfit_map_band(
                    name, self._preferred_phase(comp), component=comp)


    def _plot_misfit_map_band(
        self, band: str, phase: str, component: str = 'Z',
    ) -> None:
        common = self._common_by_event()
        if not common:
            return
        eid = max(common, key=lambda k: len(next(iter(common[k].values()))))
        mmap = common[eid]
        models = [m for m in self.models if m in mmap]
        if not models:
            return
        ev = next((e for e in self.events if e.event_id == eid), None)
        region = self._map_region()
        ncols = 3
        nrows = int(np.ceil(len(models) / float(ncols)))
        panel_w = 7.0
        fig = pygmt.Figure()
        self._setup_pygmt()
        pygmt.makecpt(cmap='viridis', series=[0.3, 1.0, 0.05], continuous=True)
        with fig.subplot(
            nrows=nrows, ncols=ncols,
            figsize=(f'{panel_w * ncols + 1.4:.1f}c', f'{7.2 * nrows + 1.0:.1f}c'),
            margins=['0.28c', '0.48c'],
            title=self._gmt_text(f'{eid}  {band}  {component}  {phase}  NZCC'),
        ):
            n_win = 0
            for i, model in enumerate(models):
                lons, lats, vals = [], [], []
                for m in mmap[model].values():
                    if (m.band != band or m.phase != phase
                            or m.component != component):
                        continue
                    coord = self._station_coord(m.network, m.station)
                    if coord is None or not np.isfinite(m.nzcc):
                        continue
                    if not self._in_common_region(coord.lat, coord.lon):
                        continue
                    lons.append(coord.lon)
                    lats.append(coord.lat)
                    vals.append(float(m.nzcc))
                row, col = divmod(i, ncols)
                with fig.set_panel(panel=i):
                    proj = f'M{panel_w}c'
                    self._draw_basemap(
                        fig, region, proj,
                        frame=self._map_frame(
                            short_model(model),
                            left=(col == 0),
                            bottom=(row == nrows - 1),
                        ),
                        topo=False, geology=False,
                    )
                    grid = self._nzcc_grid(lons, lats, vals, region)
                    if grid is not None:
                        fig.grdimage(
                            grid=grid, cmap=True, nan_transparent=True,
                        )
                    elif lons:
                        fig.plot(
                            data=np.column_stack([lons, lats, vals]),
                            style='c0.12c', cmap=True, pen='0.04p,gray20',
                        )
                    fig.coast(
                        region=region, projection=proj,
                        shorelines='0.3p,gray20',
                        borders=['1/0.25p,gray40'],
                        resolution='i',
                    )
                    if ev is not None and np.isfinite(ev.lon):
                        fig.plot(
                            x=[ev.lon], y=[ev.lat],
                            style='a0.35c', fill='red', pen='0.3p,black',
                        )
                    n_win = max(n_win, len(vals))
        fig.colorbar(
            position='JBC+o0c/0.55c+w12c/0.32c+h',
            frame='xa0.2f0.1+lNZCC',
        )
        self._save_gmt(fig, f'ea-4_misfit_map_{eid}_{band}_{component}')


    def plot_zhou_diagnostics(self) -> None:
        """Zhou et al. 2021 Fig.7 / 10 / 11 同口径诊断图，不重测。"""
        self.plot_dt_histograms()
        self.plot_category_predictability()
        self.plot_predict_vs_distance()


    def plot_dt_histograms(self) -> None:
        """公共可信窗的 dT 直方图（Zhou Fig.7）：看系统偏早/偏晚。"""
        common = self._common_by_event()
        if not common:
            return
        bands = [b[0] for b in self.config.bands]
        models = [m for m in self.models
                  if any(m in mmap for mmap in common.values())]
        if not models:
            return
        fig, axes = plt.subplots(
            len(bands), len(models),
            figsize=(2.35 * len(models), 2.55 * len(bands)),
            sharex=True, sharey=True, squeeze=False,
        )
        bins = np.linspace(-12.0, 12.0, 37)
        for i, band in enumerate(bands):
            for j, model in enumerate(models):
                ax = axes[i, j]
                vals: List[float] = []
                for mmap in common.values():
                    if model not in mmap:
                        continue
                    for k in self._accepted_keys(mmap[model], band):
                        m = mmap[model][k]
                        if np.isfinite(m.dt_shift):
                            vals.append(float(m.dt_shift))
                if vals:
                    ax.hist(vals, bins=bins, color=model_color(model),
                            edgecolor='white', linewidth=0.2)
                    med = float(np.median(vals))
                    ax.axvline(med, color='k', lw=0.8, ls='--')
                    ax.text(0.97, 0.92, f'n={len(vals)}\nmed={med:.2f}s',
                            transform=ax.transAxes, ha='right', va='top',
                            fontsize=7)
                ax.axvline(0.0, color='gray', lw=0.5)
                if i == 0:
                    ax.set_title(short_model(model), fontsize=10,
                                 color=model_color(model))
                if j == 0:
                    ax.set_ylabel(f'{band}\ncount', fontsize=9)
                if i == len(bands) - 1:
                    ax.set_xlabel('dT (s)', fontsize=9)
                ax.set_xlim(-12.0, 12.0)
                ax.grid(True, axis='y', alpha=0.25)
        fig.suptitle(
            'Traveltime-shift histograms  (common windows, CC/SNR accepted)\n'
            'dT>0 synthetic late   dashed = median',
            fontsize=12, fontweight='bold',
        )
        fig.tight_layout()
        self._save(fig, 'ea-3_dt_histograms')


    def plot_category_predictability(self) -> None:
        """六类 NZCC>0.7 占比（Zhou Fig.10）。"""
        common = self._common_by_event()
        if not common:
            return
        bands = [b[0] for b in self.config.bands]
        cats = [
            ('body-Z', 'P–SV Z'), ('body-R', 'P–SV R'), ('body-T', 'SH T'),
            ('surf-Z', 'Rayleigh Z'), ('surf-R', 'Rayleigh R'),
            ('surf-T', 'Love T'),
        ]
        models = [m for m in self.models
                  if any(m in mmap for mmap in common.values())]
        thr = float(self.config.nzcc_predict_threshold)
        fig, axes = plt.subplots(
            len(bands), 1, figsize=(10.4, 3.2 * len(bands)), squeeze=False)
        x = np.arange(len(cats))
        width = 0.12
        for i, band in enumerate(bands):
            ax = axes[i, 0]
            for k, model in enumerate(models):
                means = []
                for cls, _lab in cats:
                    ev_frac: List[float] = []
                    for mmap in common.values():
                        if model not in mmap:
                            continue
                        rows = [m for key, m in mmap[model].items()
                                if key[3] == band
                                and self._phase_class(m) == cls
                                and np.isfinite(m.nzcc)]
                        if rows:
                            ev_frac.append(
                                sum(1 for m in rows if m.nzcc > thr) / len(rows))
                    means.append(float(np.mean(ev_frac)) if ev_frac else np.nan)
                ax.bar(x + (k - 0.5 * (len(models) - 1)) * width, means,
                       width=width, color=model_color(model),
                       label=short_model(model), edgecolor='white', linewidth=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels([lab for _c, lab in cats], fontsize=8)
            ax.set_ylabel('event-equal  NZCC>0.7')
            ax.set_ylim(0.0, 1.05)
            ax.set_title(band, fontsize=11)
            ax.grid(True, axis='y', alpha=0.25)
            if i == 0:
                ax.legend(fontsize=8, ncol=len(models), loc='upper right')
        fig.suptitle(
            'Predictability by category  (Zhou et al. 2021 Fig. 10)',
            fontsize=12, fontweight='bold',
        )
        fig.tight_layout()
        self._save(fig, 'ea-10_predict_categories')


    def plot_predict_vs_distance(self) -> None:
        """六类 NZCC>0.7 随震中距变化（Zhou Fig.11）。"""
        common = self._common_by_event()
        if not common:
            return
        bands = [b[0] for b in self.config.bands]
        cats = [
            ('body-Z', 'P–SV Z'), ('body-R', 'P–SV R'), ('body-T', 'SH T'),
            ('surf-Z', 'Rayleigh Z'), ('surf-R', 'Rayleigh R'),
            ('surf-T', 'Love T'),
        ]
        edges = (3.0, 10.0, 20.0, 30.0, 45.0, 90.0)
        labels = [f'{edges[i]:.0f}–{edges[i+1]:.0f}' for i in range(len(edges) - 1)]
        models = [m for m in self.models
                  if any(m in mmap for mmap in common.values())]
        thr = float(self.config.nzcc_predict_threshold)
        fig, axes = plt.subplots(
            len(bands), len(cats),
            figsize=(2.15 * len(cats), 2.55 * len(bands)),
            sharex=True, sharey=True, squeeze=False,
        )
        xc = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
        for i, band in enumerate(bands):
            for j, (cls, clab) in enumerate(cats):
                ax = axes[i, j]
                for model in models:
                    fracs = []
                    for i_bin in range(len(edges) - 1):
                        d0, d1 = edges[i_bin], edges[i_bin + 1]
                        n_ok = n_all = 0
                        for mmap in common.values():
                            if model not in mmap:
                                continue
                            for key, m in mmap[model].items():
                                if key[3] != band or self._phase_class(m) != cls:
                                    continue
                                if not (np.isfinite(m.dist_deg) and np.isfinite(m.nzcc)):
                                    continue
                                if not (d0 <= float(m.dist_deg) < d1):
                                    continue
                                n_all += 1
                                if m.nzcc > thr:
                                    n_ok += 1
                        fracs.append(n_ok / n_all if n_all else np.nan)
                    ax.plot(xc, fracs, color=model_color(model), lw=1.3,
                            marker='o', ms=3.5, label=short_model(model))
                ax.set_ylim(0.0, 1.05)
                ax.grid(True, alpha=0.25)
                if i == 0:
                    ax.set_title(clab, fontsize=9)
                if j == 0:
                    ax.set_ylabel(f'{band}\nNZCC>0.7', fontsize=8)
                if i == len(bands) - 1:
                    ax.set_xticks(xc)
                    ax.set_xticklabels(labels, fontsize=6.5, rotation=35)
                    ax.set_xlabel('Δ (°)', fontsize=8)
                if i == 0 and j == len(cats) - 1:
                    ax.legend(fontsize=6, loc='lower left')
        fig.suptitle(
            'Predictability vs epicentral distance  (Zhou et al. 2021 Fig. 11)',
            fontsize=12, fontweight='bold',
        )
        fig.tight_layout()
        self._save(fig, 'ea-11_predict_vs_dist')


    def _preferred_phase(self, component: str) -> str:
        if component == 'T':
            return 'Love'
        return 'Rayleigh'


    def _pick_spatial_stations(
        self,
        measurements: Sequence[Measurement],
        n_az: int,
        n_dist: int,
        band: str,
        component: str = 'Z',
    ) -> List[Measurement]:
        """方位角 × 震中距分格，每格取 SNR（其次 NZCC）最高的一台。"""
        phase = self._preferred_phase(component)
        pool = [m for m in measurements
                if m.band == band and m.component == component
                and m.phase in (phase, 'full')
                and np.isfinite(m.dist_deg) and np.isfinite(m.azimuth)
                and self._dist_ok(m.dist_deg)]
        if not pool:
            return []
        # 台站去重，保留更高 SNR
        best: Dict[Tuple[str, str], Measurement] = {}
        for m in pool:
            k = (m.network, m.station)
            old = best.get(k)
            score = m.snr if np.isfinite(m.snr) else m.nzcc
            if old is None:
                best[k] = m
                continue
            old_s = old.snr if np.isfinite(old.snr) else old.nzcc
            if np.isfinite(score) and (not np.isfinite(old_s) or score > old_s):
                best[k] = m
        uniq = list(best.values())
        dmin = min(m.dist_deg for m in uniq)
        dmax = max(m.dist_deg for m in uniq)
        if dmax <= dmin:
            dmax = dmin + 1.0
        cells: Dict[Tuple[int, int], List[Measurement]] = defaultdict(list)
        for m in uniq:
            ia = int((m.azimuth % 360.0) / (360.0 / n_az))
            ib = int((m.dist_deg - dmin) / (dmax - dmin) * n_dist)
            ib = min(max(ib, 0), n_dist - 1)
            cells[(ia, ib)].append(m)

        min_nzcc = float(self.config.plot.get('min_nzcc', 0.3))

        def cell_score(m: Measurement) -> float:
            nz = float(m.nzcc) if np.isfinite(m.nzcc) else -1.0
            sn = float(m.snr) if np.isfinite(m.snr) else nz
            bonus = 20.0 if nz >= min_nzcc else 0.0
            return bonus + sn

        picked = [max(vs, key=cell_score) for vs in cells.values() if vs]
        picked.sort(key=lambda m: m.dist_deg)
        return picked


    def _station_pick_score(self, m: Measurement) -> float:
        """选台分数：优先 NZCC 过阈值，再比 SNR。"""
        min_nzcc = float(self.config.plot.get('min_nzcc', 0.3))
        nz = float(m.nzcc) if np.isfinite(m.nzcc) else -1.0
        sn = float(m.snr) if np.isfinite(m.snr) else nz
        bonus = 20.0 if nz >= min_nzcc else 0.0
        return bonus + sn


    def _band_station_pool(
        self,
        measurements: Sequence[Measurement],
        band: str,
        component: str = 'Z',
    ) -> List[Measurement]:
        """该频段、该分量去重后的台站池。"""
        phase = self._preferred_phase(component)
        pool = [m for m in measurements
                if m.band == band and m.component == component
                and m.phase in (phase, 'full')
                and np.isfinite(m.dist_deg)
                and self._dist_ok(m.dist_deg)]
        best: Dict[Tuple[str, str], Measurement] = {}
        for m in pool:
            k = (m.network, m.station)
            old = best.get(k)
            if old is None or self._station_pick_score(m) > self._station_pick_score(old):
                best[k] = m
        return list(best.values())


    def _pick_distance_stations(
        self,
        measurements: Sequence[Measurement],
        n: int,
        band: str,
        component: str = 'Z',
    ) -> List[Measurement]:
        """
        按震中距均匀抽 n 台（对齐 2_waveform_plotting 的 50 台剖面）。

        在 linspace(dmin, dmax, n) 的每个目标距离附近取分数最高的一台。
        """
        uniq = self._band_station_pool(measurements, band, component)
        if not uniq:
            return []
        uniq.sort(key=lambda m: float(m.dist_deg))
        n = max(int(n), 1)
        if len(uniq) <= n:
            return uniq
        dists = np.asarray([float(m.dist_deg) for m in uniq], dtype=float)
        dmin, dmax = float(dists[0]), float(dists[-1])
        if dmax <= dmin:
            return uniq[:n]
        targets = np.linspace(dmin, dmax, n)
        bin_w = (dmax - dmin) / float(n)
        used: Set[int] = set()
        picked: List[Measurement] = []
        for t in targets:
            cand = [i for i in range(len(uniq))
                    if i not in used
                    and abs(dists[i] - t) <= max(0.55 * bin_w, 0.35)]
            if not cand:
                rest = [i for i in range(len(uniq)) if i not in used]
                if not rest:
                    break
                cand = [min(rest, key=lambda i: abs(dists[i] - t))]
            i_best = max(cand, key=lambda i: self._station_pick_score(uniq[i]))
            used.add(i_best)
            picked.append(uniq[i_best])
        picked.sort(key=lambda m: float(m.dist_deg))
        return picked


    def plot_typical_waveforms(
        self,
        event_ids: Optional[Sequence[str]] = None,
        n_stations: int = 6,
        n_section: int = 50,
        time_max: Optional[float] = 1800.0,
    ) -> None:
        """
        共同区内每个事件出波形图。无测量或无合成则跳过。

        每事件每频段:
          ea-6  抽样台站 Z/R/T 三窗（行=模型，列=ZRT，合成由 Z/N/E 旋转）
          ea-7  震中距均匀 50 台记录剖面（Z/R/T 各一张）
          ea-7_*_BASE  同台站：观测 vs BASE（Z/R/T）
          ea-8  分模型对照（行=台，列=模型；Z/R/T 各一张）
          ea-9  抽样台站位置（PyGMT）
          ea-9_obs_stations  该事件全部有观测 Z 的台（共同区域且 Δ≥3°）
        """
        by_id = {e.event_id: e for e in self.events}
        if event_ids:
            want = []
            for eid in event_ids:
                ev = by_id.get(eid)
                if ev is None:
                    print(f'⚠️  {eid} 不在事件表，跳过')
                    continue
                if not self._in_common_region(ev.lat, ev.lon):
                    print(f'⚠️  {eid} 在三模型共同区域外，跳过波形')
                    continue
                want.append(eid)
        else:
            want = [e.event_id for e in self.events]
        self._ensure_results_from_csv(want)
        tmax = self.config.plot['time_max'] if time_max is None else time_max
        print(f'\n{"=" * 70}')
        print(f'波形图（{len(self.models)} 模型 × {len(want)} 个事件，'
              f'Δ≥{self.config.min_dist_deg:g}°）')
        print(f'{"=" * 70}')
        for eid in tqdm(want, desc='Waveform plots', unit='evt',
                       dynamic_ncols=True):
            ev = by_id.get(eid)
            if ev is None or not ev.ready:
                print(f'⚠️  跳过 {eid}')
                continue
            meas_by_model = {
                r.model: r for r in self.results if r.event_id == eid
            }
            if not meas_by_model:
                print(f'⚠️  {eid}: 无测量，先跑评估')
                continue
            self._plot_obs_station_map(ev)
            avail = [m for m in self.models if m in meas_by_model]
            # 用测量条数最多的模型做选台骨架
            skeleton = max(avail, key=lambda m: len(meas_by_model[m].measurements))
            for band, _a, _b in self.config.bands:
                picks = self._pick_spatial_stations(
                    meas_by_model[skeleton].measurements,
                    n_az=self.config.plot['az_bins'],
                    n_dist=self.config.plot['dist_bins'],
                    band=band,
                )
                if not picks:
                    continue
                gallery = picks[::max(1, len(picks) // n_stations)][:n_stations]
                self._plot_overlay_gallery(
                    ev, avail, band, gallery, tmax, meas_by_model)
                for comp in PLOT_COMPONENTS:
                    self._plot_model_columns(
                        ev, avail, band, gallery, meas_by_model,
                        component=comp)
                n_sec = int(n_section or self.config.plot.get('n_section', 50))
                section = self._pick_distance_stations(
                    meas_by_model[skeleton].measurements, n_sec, band)
                if section:
                    for comp in PLOT_COMPONENTS:
                        self._plot_record_section(
                            ev, avail, band, section, tmax, skeleton,
                            component=comp)
                    base_models = self._base_compare_models(ev, avail)
                    if base_models:
                        section_b = self._base_section_picks(
                            ev, meas_by_model[skeleton].measurements,
                            n_sec, band, section)
                        for comp in PLOT_COMPONENTS:
                            self._plot_record_section(
                                ev, base_models, band, section_b, tmax,
                                base_models[0], stem_suffix='_BASE',
                                component=comp)
                        base_gal = section_b[::max(1, len(section_b) // 8)][:8]
                        self._plot_overlay_gallery(
                            ev, base_models, band, base_gal, tmax,
                            meas_by_model, stem_suffix='_BASE')
                        for comp in PLOT_COMPONENTS:
                            self._plot_model_columns(
                                ev, base_models, band, base_gal,
                                meas_by_model, stem_suffix='_BASE',
                                component=comp)
                        self._plot_pick_map(
                            ev, section_b, band,
                            stem=f'ea-9_{band}_ZRT_section50')
                self._plot_pick_map(ev, gallery, band)
            # 逐事件释放滤波缓存，避免 22×5×台站把内存堆满
            self._z_cache.clear()
            self._zrt_cache.clear()
            self._syn_cache.clear()


    def _chujie_index(self) -> Optional[SynDirIndex]:
        """原作者 Chujie 正演（仅九州事件目录）。"""
        path = Path(self.config.chujie_dir)
        if not path.is_dir():
            return None
        if self._chujie_idx is None:
            self._chujie_idx = SynDirIndex(path)
            self.logger.info(
                'Chujie 正演: %s  (%d 个 Z 台)',
                path, len(self._chujie_idx.z_stations()),
            )
        return self._chujie_idx


    def _base_compare_models(
        self,
        ev: EventSpec,
        avail: Sequence[str],
    ) -> List[str]:
        """BASE 对照模型列表；九州事件追加 Chujie。"""
        if 'BASE' not in avail:
            return []
        out = ['BASE']
        if ev.event_id == self.config.chujie_event and self._chujie_index() is not None:
            out.append('Chujie')
        return out


    def _base_section_picks(
        self,
        ev: EventSpec,
        measurements: Sequence[Measurement],
        n: int,
        band: str,
        fallback: Sequence[Measurement],
    ) -> List[Measurement]:
        """
        BASE 剖面选台。九州事件优先抽 Chujie 也有 Z 的台，便于三条线对齐。
        """
        if ev.event_id != self.config.chujie_event:
            return list(fallback)
        idx = self._chujie_index()
        if idx is None:
            return list(fallback)
        keys = idx.z_stations()
        if not keys:
            return list(fallback)
        overlap = [m for m in measurements
                   if (m.network, m.station) in keys
                   or (strip_net_suffix(m.network), m.station) in keys]
        picked = self._pick_distance_stations(overlap, n, band)
        if len(picked) >= 15:
            return picked
        return list(fallback)


    def _load_z_filtered(
        self,
        ev: EventSpec,
        net: str,
        sta: str,
        model: Optional[str],
        band: str,
        tmin: float,
        tmax_p: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """读一台 Z，滤波后返回 (t, d)。model=None 表示观测。"""
        cache_key = (ev.event_id, model or 'OBS', net, sta, band)
        cached = self._z_cache.get(cache_key)
        if cached is not None:
            return cached
        if model is None:
            obs = self._obs_index(ev)
            if obs is None:
                return None
            raw = read_observed(obs, net, sta, 'Z', ev.origin, self.config)
        else:
            entry = self._syn_index(ev, model).get(net, sta, 'Z')
            if entry is None:
                return None
            raw = read_raw(entry.path, self.config, origin=ev.origin)
        if raw is None:
            return None
        self._fill_geometry(raw, ev, net, sta)
        tr = Trace(data=np.asarray(raw.trace.data, dtype=float))
        tr.stats.delta = float(raw.time[1] - raw.time[0]) if len(raw.time) > 1 else 0.1
        d = filter_band(tr, tmin, tmax_p, self.config)
        if len(d) != len(raw.time):
            return None
        out = (raw.time, d)
        self._z_cache[cache_key] = out
        return out


    def _load_zrt_filtered(
        self,
        ev: EventSpec,
        net: str,
        sta: str,
        model: Optional[str],
        band: str,
        tmin: float,
        tmax_p: float,
    ) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """读一台 ZRT，滤波后返回 {Z/R/T: (t, d)}。model=None 表示观测。"""
        cache_key = (ev.event_id, model or 'OBS', net, sta, band)
        cached = self._zrt_cache.get(cache_key)
        if cached is not None:
            return cached
        raws = self._load_station_raws(ev, net, sta, model)
        if raws is None:
            return None
        zrt = self._raws_to_zrt(raws)
        if not zrt:
            return None
        filtered: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for comp, (t0, d0) in zrt.items():
            tr = Trace(data=np.asarray(d0, dtype=float))
            tr.stats.delta = float(t0[1] - t0[0]) if len(t0) > 1 else 0.1
            d = filter_band(tr, tmin, tmax_p, self.config)
            if len(d) != len(t0):
                continue
            filtered[comp] = (t0, d)
        if 'Z' not in filtered:
            return None
        self._zrt_cache[cache_key] = filtered
        z_only = filtered.get('Z')
        if z_only is not None:
            self._z_cache.setdefault(cache_key, z_only)
        return filtered


    def _shared_peak_norm(*arrays: np.ndarray) -> float:
        """多道共用峰值，保留观测与合成的相对振幅。"""
        peak = 0.0
        for arr in arrays:
            if arr is None or len(arr) == 0:
                continue
            val = float(np.max(np.abs(arr)))
            if np.isfinite(val):
                peak = max(peak, val)
        return peak if peak > 0.0 else 1.0


    def _band_limits(self, band: str) -> Tuple[float, float]:
        for name, tmin, tmax in self.config.bands:
            if name == band:
                return float(tmin), float(tmax)
        return 20.0, 40.0


    def _display_window(self, m: Measurement, band: str) -> Tuple[float, float]:
        """震相窗两侧各留 pad_periods 个最长周期；缺失则用群速度窗。"""
        _tmin_p, tmax_p = self._band_limits(band)
        pad = float(self.config.plot.get('pad_periods', 1.5)) * tmax_p
        if (np.isfinite(m.win_start) and np.isfinite(m.win_end)
                and m.win_end > m.win_start):
            return max(0.0, float(m.win_start) - pad), float(m.win_end) + pad
        vmin, vmax = self.config.plot['vg'].get(band, (2.8, 4.2))
        dist_km = float(m.dist_deg) * 111.195
        return max(0.0, dist_km / vmax - pad), dist_km / vmin + pad


    def _station_meas(
        self,
        measurements: Sequence[Measurement],
        network: str,
        station: str,
        band: str,
        component: str = 'Z',
    ) -> Optional[Measurement]:
        phase = self._preferred_phase(component)
        for m in measurements:
            if (m.network == network and m.station == station
                    and m.band == band and m.component == component
                    and m.phase in (phase, 'full')):
                return m
        return None


    def _plot_pair(
        self,
        ax: Any,
        t_o: np.ndarray,
        d_o: np.ndarray,
        syn: Optional[Tuple[np.ndarray, np.ndarray]],
        t0: float,
        t1: float,
        color: str,
        label: Optional[str] = None,
        lw_obs: float = 1.15,
        lw_syn: float = 0.9,
    ) -> None:
        """同一观测尺度下画 Obs + 一条合成，并缩到 [t0, t1]。"""
        mask = (t_o >= t0) & (t_o <= t1)
        if not np.any(mask):
            return
        scale = float(np.max(np.abs(d_o[mask]))) or 1.0
        if syn is not None:
            common = to_common_axis(t_o, d_o / scale, syn[0], syn[1] / scale)
            if common is not None:
                t, _, ds = common
                keep = (t >= t0) & (t <= t1)
                if np.any(keep):
                    ax.plot(t[keep], ds[keep], color=color, lw=lw_syn,
                            alpha=0.9, zorder=2, label=label)
        ax.plot(t_o[mask], d_o[mask] / scale, color='k', lw=lw_obs,
                zorder=3, label='Obs' if label else None)


    def _plot_overlay_gallery(
        self,
        ev: EventSpec,
        models: Sequence[str],
        band: str,
        picks: Sequence[Measurement],
        time_max: float,
        meas_by_model: Optional[Dict[str, EventModelResult]] = None,
        stem_suffix: str = '',
    ) -> None:
        """
        抽样台站的 Z/R/T 三窗图（对齐 compare_waveforms.plot_zrt_station / wf-6）。

        每台一张：行=模型，列=Z|R|T；观测黑线叠在每行上。
        """
        _ = meas_by_model
        picks = [m for m in picks if self._keep_measurement(m, ev)]
        if not picks:
            return
        for m in picks:
            self._plot_zrt_station(
                ev, models, band, m, time_max, stem_suffix)


    def _plot_zrt_station(
        self,
        ev: EventSpec,
        models: Sequence[str],
        band: str,
        pick: Measurement,
        time_max: float,
        stem_suffix: str = '',
    ) -> None:
        """单台 Z/R/T：合成由 Z/N/E 旋转，观测优先用已旋好的 R/T。"""
        tmin, tmax_p = self._band_limits(band)
        pw = self.config.phase_windows
        net, sta = pick.network, pick.station
        obs = self._load_zrt_filtered(ev, net, sta, None, band, tmin, tmax_p)
        if obs is None:
            return
        filtered: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
            OBSERVED_LABEL: obs,
        }
        row_labs: List[str] = []
        for model in models:
            syn = self._load_zrt_filtered(
                ev, net, sta, model, band, tmin, tmax_p)
            if syn is None:
                continue
            filtered[model] = syn
            row_labs.append(model)
        if not row_labs:
            return

        comps = ('Z', 'R', 'T')
        phase_title = {
            'Z': 'P / SV / Rayleigh',
            'R': 'P / SV / Rayleigh',
            'T': 'SH / Love',
        }
        phase_list = {
            'Z': ['P', 'SV', 'Rayleigh'],
            'R': ['P', 'SV', 'Rayleigh'],
            'T': ['SH', 'Love'],
        }
        phase_color = {
            'P': '#ffe6e6', 'SV': '#e6f2ff', 'SH': '#e6f2ff',
            'Rayleigh': '#e6ffe6', 'Love': '#fff5e6',
        }

        dist = float(pick.dist_deg) if np.isfinite(pick.dist_deg) else np.nan
        if not np.isfinite(dist):
            dist = self._station_dist_deg(ev, net, sta)
        depth = float(ev.depth_km) if np.isfinite(ev.depth_km) else np.nan
        if np.isfinite(depth) and depth > 800.0:
            depth = depth / 1000.0
        arrivals = taup_arrivals(dist, depth, pw.get('taup_model', 'ak135')) \
            if pw.get('enabled', True) else {}

        col_norm: Dict[str, float] = {}
        for comp in comps:
            series = []
            if comp in obs:
                series.append(obs[comp][1])
            for lab in row_labs:
                syn = filtered[lab].get(comp)
                if syn is not None:
                    series.append(syn[1])
            col_norm[comp] = self._shared_peak_norm(*series)

        n_row = len(row_labs)
        fig, axes = plt.subplots(
            n_row, 3, figsize=(12.5, 1.55 * n_row + 1.2),
            sharex=True, squeeze=False,
        )
        syn_tmax = 0.0
        for i, lab in enumerate(row_labs):
            for j, comp in enumerate(comps):
                ax = axes[i, j]
                syn = filtered[lab].get(comp)
                if syn is None:
                    ax.set_visible(False)
                    continue
                if pw.get('enabled', True):
                    for phase in phase_list[comp]:
                        win = phase_time_window(
                            phase, band, tmin, tmax_p, dist, arrivals, pw)
                        if win is None:
                            continue
                        w0, w1, _tp = win
                        ax.axvspan(
                            w0, w1, color=phase_color.get(phase, '#f0f0f0'),
                            alpha=0.45, zorder=0, linewidth=0)
                t_s, d_s = syn
                norm = col_norm[comp]
                obs_c = obs.get(comp)
                if obs_c is not None:
                    t_o, d_o = obs_c
                    ax.plot(t_o, d_o / norm, color='k', lw=1.0, zorder=2)
                    syn_tmax = max(syn_tmax, float(t_o[-1]))
                ax.plot(t_s, d_s / norm, color=model_color(lab),
                        lw=0.95, alpha=0.95, zorder=3)
                syn_tmax = max(syn_tmax, float(t_s[-1]))
                ax.axhline(0.0, color='gray', lw=0.4, zorder=1)
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(
                        lab, fontsize=9, rotation=0, labelpad=28, va='center')
                if i == 0:
                    ax.set_title(f'{comp}\n{phase_title[comp]}', fontsize=11)
                if i == n_row - 1:
                    ax.set_xlabel('time (s)')

        x0, x1 = 0.0, float(time_max)
        if syn_tmax:
            x1 = min(x1, float(syn_tmax))
        if x1 <= x0:
            x1 = x0 + 60.0
        for ax in axes.ravel():
            if ax.get_visible():
                ax.set_xlim(x0, x1)
                ax.grid(True, axis='x', alpha=0.25)

        fig.suptitle(
            f'{net}.{sta}  |  {band}  |  Δ = {dist:.1f}°  |  '
            f'Observed (black) + synthetics',
            fontsize=12, y=1.01,
        )
        fig.tight_layout()
        self._save_wave(
            fig, ev.event_id,
            f'ea-6_{band}_ZRT_{net}.{sta}{stem_suffix}',
        )


    def _plot_model_columns(
        self,
        ev: EventSpec,
        models: Sequence[str],
        band: str,
        picks: Sequence[Measurement],
        meas_by_model: Dict[str, EventModelResult],
        stem_suffix: str = '',
        component: str = 'Z',
    ) -> None:
        """行=台、列=模型：每格观测黑 + 该模型色（指定 Z/R/T 分量）。"""
        picks = [m for m in picks if self._keep_measurement(m, ev)]
        if not picks or not models:
            return
        tmin, tmax_p = self._band_limits(band)
        n, n_m = len(picks), len(models)
        fig, axes = plt.subplots(
            n, n_m, figsize=(2.55 * n_m + 1.4, 1.12 * n + 1.5),
            sharex=False, sharey=True, squeeze=False)
        for i, pk in enumerate(picks):
            win_src = self._station_meas(
                next(iter(meas_by_model.values())).measurements,
                pk.network, pk.station, band, component) if meas_by_model else None
            t0, t1 = self._display_window(win_src or pk, band)
            obs_zrt = self._load_zrt_filtered(
                ev, pk.network, pk.station, None, band, tmin, tmax_p)
            obs = None if obs_zrt is None else obs_zrt.get(component)
            for j, model in enumerate(models):
                ax = axes[i, j]
                if obs is None:
                    ax.set_axis_off()
                    continue
                syn_zrt = self._load_zrt_filtered(
                    ev, pk.network, pk.station, model, band, tmin, tmax_p)
                syn = None if syn_zrt is None else syn_zrt.get(component)
                hit = None
                if model in meas_by_model:
                    hit = self._station_meas(
                        meas_by_model[model].measurements,
                        pk.network, pk.station, band, component)
                if hit is not None and np.isfinite(hit.win_start):
                    ax.axvspan(hit.win_start, hit.win_end, color='#f5e6c8',
                               alpha=0.40, zorder=0)
                elif np.isfinite(pk.win_start) and component == 'Z':
                    ax.axvspan(pk.win_start, pk.win_end, color='#f5e6c8',
                               alpha=0.40, zorder=0)
                self._plot_pair(ax, obs[0], obs[1], syn, t0, t1,
                                model_color(model),
                                label=None, lw_obs=1.05, lw_syn=0.95)
                ax.set_xlim(t0, t1)
                ax.set_ylim(-1.35, 1.35)
                ax.set_yticks([])
                if hit is not None and np.isfinite(hit.nzcc):
                    ax.text(0.98, 0.86, f'{hit.nzcc:.2f}',
                            transform=ax.transAxes, ha='right', va='center',
                            fontsize=7, color=model_color(model))
                if i == 0:
                    ax.set_title(short_model(model), fontsize=10,
                                 color=model_color(model), fontweight='bold')
                if j == 0:
                    ax.set_ylabel(
                        f'{pk.network}.{pk.station}\n{pk.dist_deg:.1f}°',
                        fontsize=7.5, rotation=0, ha='right', va='center',
                    )
                if i == n - 1:
                    ax.set_xlabel('t (s)', fontsize=8)
        fig.suptitle(
            f'{ev.event_id}   {band}   {component}   '
            f'Obs (black) + each model  (Mw={ev.mw:.1f})',
            fontsize=11, fontweight='bold',
        )
        fig.tight_layout()
        self._save_wave(
            fig, ev.event_id, f'ea-8_{band}_{component}_models{stem_suffix}')


    def _phase_first_arrivals(
        self,
        phase_names: Sequence[str],
        dist_min: float,
        dist_max: float,
        depth_km: float,
        time_limit: float,
        n_samples: int = 120,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        每个震中距取 phase_names 中最早的一支。

        深源近台初至是上出 p/s，下行 P/S 要到更大 Δ 才出现；
        只查大写 P/S 时曲线会从十几度才开始。
        """
        depth = float(depth_km)
        if (not np.isfinite(depth)) or depth < 0.0:
            depth = 0.0
        if depth > 800.0:
            depth = depth / 1000.0
        if dist_max <= dist_min:
            dist_max = dist_min + 0.5
        model = get_taup_model(
            str(self.config.phase_windows.get('taup_model', 'ak135')))
        dists = np.linspace(float(dist_min), float(dist_max), max(int(n_samples), 5))
        times: List[float] = []
        keep: List[float] = []
        for dist in dists:
            cands: List[float] = []
            for name in phase_names:
                try:
                    arr = model.get_travel_times(
                        source_depth_in_km=depth,
                        distance_in_degree=float(dist),
                        phase_list=[name],
                    )
                except Exception:
                    continue
                if not arr:
                    continue
                t = float(arr[0].time)
                if t <= float(time_limit):
                    cands.append(t)
            if cands:
                times.append(min(cands))
                keep.append(float(dist))
        if len(times) < 3:
            return np.array([]), np.array([])
        return np.asarray(times), np.asarray(keep)


    def _add_section_phases(
        self,
        ax: Any,
        ev: EventSpec,
        dist_min: float,
        dist_max: float,
        time_limit: float,
    ) -> None:
        """AK135 理论走时。P/S 含上出 p/s，避免深源近台空白。"""
        depth = float(ev.depth_km) if np.isfinite(ev.depth_km) else 10.0
        groups = [
            ('P', ('p', 'P'), '#e74c3c'),
            ('S', ('s', 'S'), '#27ae60'),
            ('pP', ('pP',), '#f39c12'),
            ('sS', ('sS',), '#3498db'),
            ('PP', ('PP',), '#e67e22'),
            ('SS', ('SS',), '#16a085'),
        ]
        handles: List[Any] = []
        labels: List[str] = []
        for label, names, color in groups:
            times, dists = self._phase_first_arrivals(
                names, dist_min, dist_max, depth, time_limit)
            if times.size < 3:
                continue
            line, = ax.plot(
                times, dists, color=color, linestyle='--',
                linewidth=1.4, alpha=0.75, zorder=1,
            )
            # 标在近台一侧约 1/4 处，不要钉在曲线中点（中点常是远台）
            i = max(0, min(len(times) - 1, len(times) // 4))
            ax.text(
                times[i], dists[i], f' {label}',
                color=color, fontsize=16, fontweight='bold',
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.85, edgecolor=color, linewidth=0.8),
                zorder=4,
            )
            handles.append(line)
            labels.append(label)
        if handles:
            phase_leg = ax.legend(
                handles, labels, loc='upper left', fontsize=14,
                framealpha=0.95, ncol=2, title='Theoretical Phases (IASP91)',
            )
            phase_leg.get_title().set_fontweight('bold')
            ax.add_artist(phase_leg)


    def _plot_record_section(
        self,
        ev: EventSpec,
        models: Sequence[str],
        band: str,
        picks: Sequence[Measurement],
        time_max: float,
        highlight: str,
        stem_suffix: str = '',
        component: str = 'Z',
    ) -> None:
        """
        记录剖面：横轴发震后时间、纵轴震中距。

        对齐 2_waveform_plotting / waveform_plotting-all：
        观测黑线压在上面，各模型彩色，IASP91 走时曲线，右侧台站标签。
        合成由 Z/N/E 旋到 ZRT 后再取指定分量。
        """
        _ = highlight
        picks = [m for m in picks if self._keep_measurement(m, ev)]
        if not picks:
            return
        tmin, tmax_p = self._band_limits(band)
        t_hi = float(time_max)
        rows: List[Tuple[Measurement, Tuple[np.ndarray, np.ndarray],
                         Dict[str, Tuple[np.ndarray, np.ndarray]]]] = []
        for pk in picks:
            if not np.isfinite(pk.dist_deg):
                continue
            obs_zrt = self._load_zrt_filtered(
                ev, pk.network, pk.station, None, band, tmin, tmax_p)
            obs = None if obs_zrt is None else obs_zrt.get(component)
            if obs is None:
                continue
            syns: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for model in models:
                syn_zrt = self._load_zrt_filtered(
                    ev, pk.network, pk.station, model, band, tmin, tmax_p)
                syn = None if syn_zrt is None else syn_zrt.get(component)
                if syn is not None:
                    syns[model] = syn
            rows.append((pk, obs, syns))
        if not rows:
            return

        rows.sort(key=lambda item: float(item[0].dist_deg))
        dists = [float(pk.dist_deg) for pk, _, _ in rows]
        gaps = np.diff(np.sort(np.asarray(dists)))
        gaps = gaps[gaps > 1.0e-3]
        med_gap = float(np.median(gaps)) if gaps.size else 1.0
        amp = float(np.clip(0.40 * med_gap, 0.12, 1.20))

        n = len(rows)
        fig, ax = plt.subplots(figsize=(20.0, max(24.0, 0.48 * n)))
        single = len(models) <= 2
        lw_syn = 1.15 if single else 0.65
        lw_obs = 1.55 if single else 1.25
        label_fs = 9 if n >= 30 else 12
        drawn_obs = False
        drawn_model = {m: False for m in models}
        prev_y = -1.0e9
        stagger = False

        for pk, obs, syns in rows:
            y = float(pk.dist_deg)
            t_o, d_o = obs
            keep = (t_o >= 0.0) & (t_o <= t_hi)
            if not np.any(keep):
                continue
            abs_d = np.abs(d_o[keep])
            scale = float(np.percentile(abs_d, 99.5)) if abs_d.size else 1.0
            if (not np.isfinite(scale)) or scale <= 0.0:
                scale = float(np.max(abs_d)) or 1.0
            for model in models:
                syn = syns.get(model)
                if syn is None:
                    continue
                common = to_common_axis(
                    t_o, d_o / scale, syn[0], syn[1] / scale)
                if common is None:
                    continue
                t, _, ds = common
                kk = (t >= 0.0) & (t <= t_hi)
                if not np.any(kk):
                    continue
                ax.plot(
                    t[kk], ds[kk] * amp + y,
                    color=model_color(model), linewidth=lw_syn, alpha=0.88,
                    zorder=2,
                    label=short_model(model) if not drawn_model[model] else '',
                )
                drawn_model[model] = True
            ax.plot(
                t_o[keep], d_o[keep] / scale * amp + y,
                color='k', linewidth=lw_obs, alpha=0.82, zorder=3,
                label='Observed' if not drawn_obs else '',
            )
            drawn_obs = True
            if abs(y - prev_y) < 1.15:
                stagger = not stagger
            else:
                stagger = False
            prev_y = y
            label_x = t_hi * (0.86 if stagger else 0.97)
            ax.text(
                label_x, y,
                f'{pk.network}.{pk.station}\n{y:.1f}°',
                fontsize=label_fs, va='center', ha='right', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          alpha=0.92, edgecolor='gray', linewidth=0.5),
                zorder=5,
            )

        floor = float(getattr(self.config, 'min_dist_deg', 0.0) or 0.0)
        d_lo = max(floor, min(dists))
        d_hi = max(dists)
        self._add_section_phases(ax, ev, d_lo, d_hi, t_hi)
        ax.set_xlabel('Time after Origin (s)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Epicentral Distance (°)', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)
        ax.axvline(0.0, color='k', linestyle='--', linewidth=0.75,
                   alpha=0.8, zorder=1)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlim(0.0, t_hi)
        pad = max(1.2, 0.08 * (d_hi - d_lo + 1.0))
        ax.set_ylim(max(0.0, d_lo - 0.25), d_hi + pad)

        wave_handles = [Line2D([0], [0], color='k', lw=2.2, label='Observed')]
        for model in models:
            if drawn_model[model]:
                wave_handles.append(Line2D(
                    [0], [0], color=model_color(model), lw=2.2,
                    label=short_model(model),
                ))
        wave_leg = ax.legend(
            handles=wave_handles, loc='upper right', fontsize=14,
            framealpha=0.95, title='Waveforms',
        )
        wave_leg.get_title().set_fontweight('bold')
        ax.add_artist(wave_leg)

        mw = f'Mw={ev.mw:.1f}' if np.isfinite(ev.mw) else ''
        depth = float(ev.depth_km) if np.isfinite(ev.depth_km) else np.nan
        if np.isfinite(depth) and depth > 800.0:
            depth = depth / 1000.0
        dep = f'{depth:.0f} km' if np.isfinite(depth) else ''
        if 'Chujie' in models and 'BASE' in models:
            cmp = 'Observed vs BASE + Chujie'
        elif single and models[0] == 'BASE':
            cmp = 'Observed vs BASE'
        else:
            cmp = f'{len(models)} models'
        ax.set_title(
            f'Record Section ({component})   {ev.event_id}   {band}   {cmp}\n'
            f'{mw}   {dep}   |   Stations: {n}   |   Filtered: {band}',
            fontsize=18, fontweight='bold', pad=14,
        )
        fig.tight_layout()
        self._save_wave(
            fig, ev.event_id, f'ea-7_{band}_{component}_section{stem_suffix}',
            formats=['jpg'],
        )


    def _save(self, fig: Any, stem: str) -> None:
        for fmt in self.config.visualization.get('figure_format', ['jpg']):
            path = self.output_dir / f'{stem}.{fmt}'
            fig.savefig(path, dpi=self.config.visualization['dpi'])
            print(f'   🖼️  {path.name}')
        plt.close(fig)


    def _save_wave(
        self,
        fig: Any,
        event_id: str,
        stem: str,
        formats: Optional[Sequence[str]] = None,
    ) -> None:
        out = self.output_dir / event_id
        out.mkdir(parents=True, exist_ok=True)
        fmts = list(formats) if formats else list(
            self.config.visualization.get('waveform_format', ['jpg']))
        for fmt in fmts:
            path = out / f'{stem}.{fmt}'
            fig.savefig(path, dpi=self.config.visualization['dpi'],
                        bbox_inches='tight', facecolor='white')
            print(f'   🖼️  {event_id}/{path.name}')
        plt.close(fig)


    def _save_gmt(self, fig: Any, stem: str, event_id: Optional[str] = None) -> None:
        """保存 PyGMT 图（jpg + pdf，与 5_2 / 5_12 一致）。"""
        out = (self.output_dir / event_id) if event_id else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        dpi = int(self.config.visualization.get('dpi', 300))
        for fmt in self.config.visualization.get('figure_format', ['jpg']):
            path = out / f'{stem}.{fmt}'
            fig.savefig(str(path), dpi=dpi, crop=True)
            tag = f'{event_id}/{path.name}' if event_id else path.name
            print(f'   🖼️  {tag}')


    def _setup_pygmt(self) -> None:
        pygmt.config(
            MAP_FRAME_TYPE='plain',
            MAP_FRAME_PEN='0.6p,black',
            FONT_ANNOT_PRIMARY='9p,Helvetica,black',
            FONT_LABEL='10p,Helvetica,black',
            FONT_TITLE='11p,Helvetica-Bold,black',
            MAP_TITLE_OFFSET='6p',
            MAP_TICK_LENGTH_PRIMARY='4p',
            MAP_TICK_PEN_PRIMARY='0.5p,black',
        )


    def _map_region(self) -> List[float]:
        box = self.common_region if self.config.use_common_region \
            else self.base_config.get_region_bounds()
        return [box['lon_min'], box['lon_max'], box['lat_min'], box['lat_max']]


    def _gmt_text(text: str) -> str:
        """GMT +t / subplot 标题：去 °，粘连 km，空格改 \\040（否则会被吃掉）。"""
        cleaned = str(text).replace('°', '')
        cleaned = re.sub(r'(\d)\s*km\b', r'\1km', cleaned, flags=re.IGNORECASE)
        return cleaned.replace(' ', r'\040')


    def _map_frame(
        self,
        title: Optional[str] = None,
        left: bool = True,
        bottom: bool = True,
    ) -> List[str]:
        """对齐 5_1 / 5_12：无空格短标题才走 +t，长句交给 fig.text。"""
        wesn = f"{'W' if left else 'w'}{'S' if bottom else 's'}en"
        if title:
            return ['xa10f5', 'ya10f5', f'{wesn}+t{self._gmt_text(title)}']
        return ['xa10f5', 'ya10f5', wesn]


    def _map_title_xy(
        self,
        fig: Any,
        region: Sequence[float],
        title: str,
        font: str = '12p,Helvetica-Bold,black',
    ) -> None:
        """用地理坐标写图题，避免 GMT +t 吞空格。"""
        lon = 0.5 * (float(region[0]) + float(region[1]))
        lat = float(region[3]) + 0.035 * (float(region[3]) - float(region[2]))
        fig.text(
            x=[lon], y=[lat], text=[title],
            font=font, no_clip=True, justify='CB',
        )


    def _nzcc_grid(
        self,
        lons: Sequence[float],
        lats: Sequence[float],
        vals: Sequence[float],
        region: Sequence[float],
        spacing: str = '1d',
    ) -> Optional[Any]:
        """1° blockmean，与评估用的 W_r 地理格一致。"""
        if len(lons) < 5:
            return None
        data = np.column_stack([
            np.asarray(lons, dtype=float),
            np.asarray(lats, dtype=float),
            np.asarray(vals, dtype=float),
        ])
        try:
            meaned = pygmt.blockmean(data=data, region=list(region), spacing=spacing)
            if meaned is None or len(meaned) == 0:
                return None
            return pygmt.xyz2grd(
                data=meaned, region=list(region), spacing=spacing,
            )
        except Exception as exc:
            self.logger.warning('NZCC blockmean 失败，回退散点: %s', exc)
            return None


    def _elevation_grid(self) -> str:
        local = VIS_DATA / 'elevation' / 'EastAsia.grd'
        if local.is_file():
            return str(local)
        return '@earth_relief_01m'


    def _topo_cmap(self) -> str:
        cpt = VIS_DATA / 'elevation' / 'EastAsiaTopo.cpt'
        return str(cpt) if cpt.is_file() else 'geo'


    def _depth_cpt(self) -> str:
        path = self.output_dir / '_tmp_event_depth.cpt'
        pygmt.makecpt(cmap='seis', series=[0, 700, 50], output=str(path))
        return str(path)


    def _draw_basemap(
        self,
        fig: Any,
        region: Sequence[float],
        projection: str,
        frame: Sequence[str],
        topo: bool = True,
        geology: bool = False,
    ) -> None:
        if topo:
            fig.grdimage(
                grid=self._elevation_grid(),
                region=region,
                projection=projection,
                frame=list(frame),
                cmap=self._topo_cmap(),
                shading=True,
            )
        else:
            fig.basemap(region=region, projection=projection, frame=list(frame))
            fig.coast(
                region=region, projection=projection,
                land='gray92', water='white',
            )
        fig.coast(
            region=region,
            projection=projection,
            shorelines='0.3p,gray20',
            borders=['1/0.25p,gray40'],
            resolution='i',
        )
        if geology:
            self._add_geology(fig)


    def _add_geology(self, fig: Any) -> None:
        """板块边界 / 活动断层 / 火山 / Slab2，缺文件则跳过。"""
        boundaries = VIS_DATA / 'tectonics' / 'boundaries.gmt'
        if boundaries.is_file():
            fig.plot(data=str(boundaries), pen='0.5p,black')
        faults = (VIS_DATA / 'gem-global-active-faults-master'
                  / 'gem_active_faults.txt')
        if faults.is_file():
            fig.plot(data=str(faults), pen='0.4p,black')
        volcano = VIS_DATA / 'volcano' / 'EastAsia_volcanoes.csv'
        if volcano.is_file():
            fig.plot(data=str(volcano), style='t0.14c',
                     fill='red', pen='0.1p,black')
        slab_dir = VIS_DATA / 'Slab2' / 'Slab2_CONTOURS'
        # .in 已是 GMT 多段（> 头），直接交给 plot；不要把分段收成 NaN
        # 再走 x/y 数组，否则 GMT 重采样会报 row contains NaNs
        eastasia_slabs = (
            'kur', 'izu', 'ryu', 'man', 'phi', 'sum', 'sul', 'van',
            'png', 'him', 'hin', 'pam', 'mak', 'alu',
        )
        if slab_dir.is_dir():
            for path in sorted(slab_dir.glob('*_slab2_dep_*.in')):
                if path.parent.name == 'ARC_CONTOURS':
                    continue
                tag = path.name.split('_', 1)[0].lower()
                if tag not in eastasia_slabs:
                    continue
                try:
                    fig.plot(data=str(path), pen='0.3p,pink')
                except Exception:
                    continue


    def _event_meca_spec(self, ev: EventSpec) -> Optional[Dict[str, float]]:
        mt = parse_harvard_mt(ev.cmt_file)
        if mt is None:
            return None
        comps = [mt['mrr'], mt['mtt'], mt['mpp'],
                 mt['mrt'], mt['mrp'], mt['mtp']]
        amp = max(abs(c) for c in comps)
        if amp <= 0:
            return None
        exp = int(np.floor(np.log10(amp)))
        fac = 10.0 ** exp
        return {
            'mrr': mt['mrr'] / fac, 'mtt': mt['mtt'] / fac,
            'mff': mt['mpp'] / fac, 'mrt': mt['mrt'] / fac,
            'mrf': mt['mrp'] / fac, 'mtf': mt['mtp'] / fac,
            'exponent': exp,
        }


    def _meca_scale(self, mw: float) -> str:
        if not np.isfinite(mw):
            mw = 6.0
        size = 0.22 + 0.07 * (float(mw) - 5.5)
        size = min(max(size, 0.20), 0.46)
        return f'{size:.2f}c'


    def _plot_event_meca(self, fig: Any, ev: EventSpec, depth_cpt: str) -> None:
        if not (np.isfinite(ev.lon) and np.isfinite(ev.lat)):
            return
        spec = self._event_meca_spec(ev)
        depth = float(ev.depth_km) if np.isfinite(ev.depth_km) else 10.0
        if spec is None:
            fig.plot(x=[ev.lon], y=[ev.lat], style='a0.28c',
                     fill='red', pen='0.25p,black')
            return
        try:
            fig.meca(
                spec=spec,
                scale=self._meca_scale(ev.mw),
                convention='mt',
                longitude=float(ev.lon),
                latitude=float(ev.lat),
                depth=depth,
                cmap=depth_cpt,
                extension_fill='cornsilk',
                pen='0.2p,gray30',
            )
        except Exception:
            fig.plot(x=[ev.lon], y=[ev.lat], style='a0.28c',
                     fill='red', pen='0.25p,black')


    def _used_station_xy(self) -> Tuple[List[float], List[float]]:
        keys: Set[Tuple[str, str]] = set()
        for r in self.results:
            for m in r.measurements:
                keys.add((m.network, m.station))
        if not keys:
            keys = {(n, s) for (n, s), c in self.station_coords.items()
                    if self._in_common_region(c.lat, c.lon)}
        lons, lats = [], []
        for net, sta in keys:
            c = self._station_coord(net, sta)
            if c is None or not self._in_common_region(c.lat, c.lon):
                continue
            lons.append(c.lon)
            lats.append(c.lat)
        return lons, lats


    def plot_geometry_map(self, **kwargs: Any) -> Optional[Path]:
        """共同区域：地形 + 构造 + 评估台站 + CMT（PyGMT，对齐 5_2）。"""
        _ = kwargs
        region = self._map_region()
        self._setup_pygmt()
        fig = pygmt.Figure()
        proj = 'M16c'
        box = self.common_region
        title = (
            f'Common region  {len(self.events)} events  '
            f'{box["lat_min"]:.0f}-{box["lat_max"]:.0f}N  '
            f'{box["lon_min"]:.0f}-{box["lon_max"]:.0f}E  '
            f'Delta>={self.config.min_dist_deg:g}'
        )
        self._draw_basemap(
            fig, region, proj,
            frame=self._map_frame(title),
            topo=True, geology=False,
        )
        lons, lats = self._used_station_xy()
        if lons:
            fig.plot(x=lons, y=lats, style='c0.08c',
                     fill='dodgerblue', transparency=35, pen=None)
        depth_cpt = self._depth_cpt()
        for ev in self.events:
            self._plot_event_meca(fig, ev, depth_cpt)
        fig.colorbar(
            cmap=depth_cpt,
            position='JMR+o0.45c/0c+w7.2c/0.32c',
            frame='xa100f50+lEvent depth (km)',
        )
        self._save_gmt(fig, 'ea-0b_geometry')
        # 覆盖旧 hex 文件名，方便对照刷新
        self._save_gmt(fig, 'ea-0b_geometry_hex')
        return self.output_dir / 'ea-0b_geometry.jpg'


    def _plot_pick_map(
        self,
        ev: EventSpec,
        picks: Sequence[Measurement],
        band: str,
        stem: Optional[str] = None,
    ) -> None:
        """代表波形所用台站的位置图（PyGMT）。"""
        picks = [m for m in picks if self._keep_measurement(m, ev)]
        if not picks:
            return
        region = self._map_region()
        self._setup_pygmt()
        fig = pygmt.Figure()
        proj = 'M14c'
        mw = f'Mw={ev.mw:.1f}' if np.isfinite(ev.mw) else ''
        title = f'{ev.event_id}  {band}  waveform stations  {mw}'.strip()
        self._draw_basemap(
            fig, region, proj,
            frame=self._map_frame(title),
            topo=True, geology=False,
        )
        lons, lats, labels = [], [], []
        for m in picks:
            c = self._station_coord(m.network, m.station)
            if c is None:
                continue
            lons.append(c.lon)
            lats.append(c.lat)
            labels.append(f'{m.network}.{m.station}')
        if lons:
            fig.plot(x=lons, y=lats, style='t0.22c',
                     fill='dodgerblue', pen='0.2p,black')
            mid_lon = 0.5 * (float(region[0]) + float(region[1]))
            for lon, lat, lab in zip(lons, lats, labels):
                inward = lon <= mid_lon
                fig.text(
                    x=lon, y=lat, text=lab,
                    font='8p,Helvetica,black',
                    justify='LM' if inward else 'RM',
                    offset='0.16c/0.10c' if inward else '-0.16c/0.10c',
                    fill='white@30',
                )
        self._plot_event_meca(fig, ev, self._depth_cpt())
        self._save_gmt(
            fig, stem or f'ea-9_{band}_Z_stations', event_id=ev.event_id)


    def _plot_obs_station_map(self, ev: EventSpec) -> None:
        """
        该事件实际有观测 Z 的台站分布。

        与评估同一套裁剪：共同区域且 Δ≥3°。台数多，只画点不标台名。
        """
        obs = self._obs_index(ev)
        if obs is None:
            return
        stations = self._filter_stations(sorted(obs.z_stations()), ev)
        lons, lats = [], []
        for net, sta in stations:
            coord = self._station_coord(net, sta)
            if coord is None:
                continue
            lons.append(coord.lon)
            lats.append(coord.lat)
        if not lons:
            return
        region = self._map_region()
        self._setup_pygmt()
        fig = pygmt.Figure()
        proj = 'M14c'
        mw = f'Mw={ev.mw:.1f}' if np.isfinite(ev.mw) else ''
        title = (
            f'{ev.event_id}  observed stations  n={len(lons)}  {mw}'
        ).strip()
        self._draw_basemap(
            fig, region, proj,
            frame=self._map_frame(title),
            topo=True, geology=False,
        )
        fig.plot(
            x=lons, y=lats, style='c0.10c',
            fill='dodgerblue', transparency=25, pen='0.05p,gray20',
        )
        self._plot_event_meca(fig, ev, self._depth_cpt())
        self._save_gmt(fig, 'ea-9_obs_stations', event_id=ev.event_id)


    def plot_all(
        self,
        plot_waveforms: bool = True,
        waveform_event_ids: Optional[Sequence[str]] = None,
        n_stations: int = 6,
        n_section: int = 20,
    ) -> None:
        """从已有 meas_*.csv 画全部评估图（含论文图）。"""
        self._ensure_results_from_csv()
        if not self.results:
            print('❌ 没有 meas_*.csv，请先跑 3_4_Waveform_assessment.py')
            return
        n_ev = len({r.event_id for r in self.results})
        n_mo = len({r.model for r in self.results})
        extra = []
        if self.config.use_common_region:
            extra.append('已裁三模型共同区域')
        if float(self.config.min_dist_deg) > 0:
            extra.append(f'Δ≥{self.config.min_dist_deg:g}°')
        if float(getattr(self.config, 'min_s_dist_deg', 0.0) or 0.0) > 0:
            extra.append(f'S 仅 Δ≥{self.config.min_s_dist_deg:g}°')
        extra_s = f'（{"，".join(extra)}）' if extra else ''
        print(f'从 CSV 出图：{n_ev} 个事件 × {n_mo} 个模型，'
              f'{sum(len(r.measurements) for r in self.results)} 条测量{extra_s}')
        self.write_aggregate_csv()
        self.write_report()
        self.plot_geometry_map()
        self.plot_summary()
        self.plot_rank_heatmap()
        self.plot_network_groups()
        self.plot_misfit_maps()
        self.plot_zhou_diagnostics()
        self.plot_paper_figures()
        if plot_waveforms:
            self.plot_typical_waveforms(
                event_ids=waveform_event_ids,
                n_stations=n_stations,
                n_section=n_section,
            )
        print(f'\n✅ 图已更新 → {self.output_dir}')

    def plot_from_csv(
        self,
        waveform_event_ids: Optional[Sequence[str]] = None,
        n_stations: int = 6,
        n_section: int = 20,
        plot_waveforms: bool = True,
    ) -> None:
        """兼容旧名：等同 plot_all。"""
        self.plot_all(
            plot_waveforms=plot_waveforms,
            waveform_event_ids=waveform_event_ids,
            n_stations=n_stations,
            n_section=n_section,
        )

    def plot_paper_figures(self, paper_dir: Optional[Path] = None) -> Path:
        """Zhou et al. 2021 版式论文图 → paper_figs/。"""
        self._ensure_results_from_csv()
        out = Path(paper_dir) if paper_dir else SAC_ROOT_DEFAULT / 'paper_figs'
        if not self.results:
            print('❌ 没有 meas_*.csv，请先跑完 3_4')
            return out
        return render_paper_figures(self, out)


EastAsiaModelAssessment = WaveformAssessmentViz


def main() -> None:
    print('🎯 EASTASIA-FWI 5_13 波形评估可视化')
    print('=' * 70)
    ap = argparse.ArgumentParser(
        description='从 3_4 的 meas_*.csv 出评估图与论文图，不重测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 5_13_Waveform_assessment.py
  python 5_13_Waveform_assessment.py --no-waveforms
  python 5_13_Waveform_assessment.py --paper-figs
  python 5_13_Waveform_assessment.py --events C201505120705A
        """,
    )
    _assess.add_io_args(ap)
    ap.add_argument('--paper-figs', action='store_true')
    ap.add_argument('--paper-dir', type=Path, default=SAC_ROOT_DEFAULT / 'paper_figs')
    ap.add_argument('--no-waveforms', action='store_true')
    ap.add_argument('--waveform-stations', type=int, default=6)
    ap.add_argument('--waveform-section', type=int, default=50)
    ap.add_argument('--plot-map', action='store_true')
    args = ap.parse_args()
    try:
        waveform_ids = args.events
        args.events = None
        base = _assess.build_assessment(args)
        viz = WaveformAssessmentViz.__new__(WaveformAssessmentViz)
        viz.__dict__.update(base.__dict__)
        viz.__class__ = WaveformAssessmentViz
        if args.paper_figs:
            viz.plot_paper_figures(paper_dir=args.paper_dir)
        elif args.plot_map:
            viz.plot_geometry_map()
        else:
            viz.plot_all(
                plot_waveforms=not args.no_waveforms,
                waveform_event_ids=waveform_ids,
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
