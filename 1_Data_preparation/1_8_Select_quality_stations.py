"""
EASTASIA-FWI 高质量台站筛选
============================

功能描述:
- 在 1_3 全网下载、1_4 预处理之后，按**实际波形**而不是永久台清单筛选正演台站
- 单事件：三分量完整性、震前噪声 SNR、位移振幅合理性
- 跨事件：通过次数、通过率、SNR 中位数
- 可选空间抽稀，避免密集台阵主导后续 misfit
- 输出 quality CSV + SPECFEM STATIONS，供正演使用

科学原理:
- 永久台清单只保证长期在网，不保证该事件上波形可用（间隙、响应残缺、噪声、方位错）
- Zhou et al. (2021) 窗口接受：SNR>4 才进入走时/振幅统计
- 正演台站应在多数事件上稳定可用，否则模型对比会被坏道主导

作者: EASTASIA-FWI Team
日期: 2026-08-23
版本: v1.0
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read
from obspy.geodetics import locations2degrees

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import BaseConfig

warnings.filterwarnings('ignore')


class StationQualityConfig:
    """波形质量筛选配置"""

    def __init__(self) -> None:
        self.snr = {
            'period_min': 20.0,
            'period_max': 100.0,
            'noise_end_s': -10.0,
            'signal_start_s': 30.0,
            'signal_end_s': 1800.0,
            'min_noise_samples': 50,
            'min_signal_samples': 100,
            'threshold': 4.0,
        }
        self.accept = {
            'min_components': 3,
            'min_length_s': 1800.0,
            'disp_amp_min_m': 1.0e-12,
            'disp_amp_max_m': 1.0e-1,
        }
        self.aggregate = {
            'min_events_pass': 4,
            'min_pass_frac': 0.30,
            'min_median_snr': 4.0,
        }
        self.thinning = {
            'min_spacing_km': 50.0,
        }
        self.visualization = {
            'dpi': 300,
            'figure_format': ['jpg', 'pdf'],
        }
        self.logging = {
            'level': 'INFO',
        }


@dataclass
class EventMeta:
    """目录事件"""
    name: str
    origin: UTCDateTime
    lat: float
    lon: float
    depth_km: float


@dataclass
class StationEventScore:
    """单台单事件评分"""
    event: str
    network: str
    station: str
    n_comp: int
    length_s: float
    snr: float
    peak_m: float
    lat: float
    lon: float
    elev: float
    passed: bool
    reason: str


def _rms(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a * a)))


def _parse_par_line(line: str) -> Optional[EventMeta]:
    parts = line.split()
    if len(parts) < 8:
        return None
    try:
        if len(parts[1]) == 8 and parts[1].isdigit():
            ymd = parts[1]
            y, mo, d = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8])
            h, mi = int(parts[2]), int(parts[3])
            sec = float(parts[4])
            lat, lon, dep = float(parts[5]), float(parts[6]), float(parts[7])
            name = parts[0]
        elif len(parts) >= 11 and parts[0].startswith('C'):
            lat, lon, dep = float(parts[1]), float(parts[2]), float(parts[3])
            y, mo, d = int(parts[5]), int(parts[6]), int(parts[7])
            h, mi = int(parts[8]), int(parts[9])
            sec = float(parts[10])
            name = parts[0]
        else:
            return None
        sec_i = int(sec)
        micro = int(round((sec - sec_i) * 1e6))
        origin = UTCDateTime(year=y, month=mo, day=d, hour=h, minute=mi,
                             second=sec_i, microsecond=max(micro, 0))
        return EventMeta(name=name, origin=origin, lat=lat, lon=lon,
                         depth_km=dep)
    except (ValueError, TypeError, IndexError):
        return None


def load_par_catalog(path: Path) -> List[EventMeta]:
    events: List[EventMeta] = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        ev = _parse_par_line(line)
        if ev is not None:
            events.append(ev)
    return events


class StationQualitySelector:
    """按实际波形筛选正演台站"""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.base_config = BaseConfig()
        self.config = StationQualityConfig()
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.base_config.dirs['stations']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = self.base_config.dirs['results'] / 'station_quality'
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = self.base_config.dirs['figures']
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.StationQuality',
            self.config.logging['level'],
        )
        self.coord_lookup = self._load_coord_lookup()
        self.logger.info('🎯 台站质量筛选器初始化完成')

    def _load_coord_lookup(self) -> Dict[str, Tuple[float, float, float]]:
        """NET.STA → (lat, lon, elev)"""
        out: Dict[str, Tuple[float, float, float]] = {}
        stations_dir = self.base_config.dirs['stations']
        for name in (
            'EastAsia_all_stations_cleaned.csv',
            'EastAsia_permanent_stations_filtered.csv',
            'EastAsia_stations.csv',
        ):
            path = stations_dir / name
            if not path.is_file():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            for _, row in df.iterrows():
                net = str(row.get('Network', '')).strip()
                sta = str(row.get('Station', '')).strip()
                if not net or not sta:
                    continue
                try:
                    lat = float(row.get('Latitude', np.nan))
                    lon = float(row.get('Longitude', np.nan))
                    elev = float(row.get('Elevation', 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(lat) and np.isfinite(lon):
                    out[f'{net}.{sta}'] = (lat, lon, elev)
        return out

    def _event_waveform_dir(
        self, ev: EventMeta, data_root: Path, kind: str,
    ) -> Optional[Path]:
        """kind=sac → disp_data/<name>；kind=mseed → session/waveforms/<name>。"""
        candidates = [data_root / ev.name]
        if kind == 'sac':
            candidates.append(data_root / ev.name)
        for cand in candidates:
            if cand.is_dir():
                return cand
        return None

    def _iter_station_files(
        self, event_dir: Path,
    ) -> Dict[str, List[Path]]:
        """NET.STA → 该台文件列表。"""
        groups: Dict[str, List[Path]] = {}
        files = list(event_dir.glob('*.SAC')) + list(event_dir.glob('*.sac'))
        files += list(event_dir.glob('*.mseed'))
        for path in files:
            key = self._netsta_from_name(path.name)
            if key is None:
                continue
            groups.setdefault(key, []).append(path)
        return groups

    @staticmethod
    def _netsta_from_name(name: str) -> Optional[str]:
        # MassDownloader: NET.STA.LOC.CHN__start__end.mseed
        if '__' in name:
            parts = name.split('__')[0].split('.')
            if len(parts) >= 2:
                return f'{parts[0]}.{parts[1]}'
        # 1_4 SAC: year.jday.hh.mm.ss.NET.STA..CHN.SAC
        if '..' in name:
            left = name.split('..')[0]
            parts = left.split('.')
            if len(parts) >= 2:
                return f'{parts[-2]}.{parts[-1]}'
        return None

    def _score_station_event(
        self, ev: EventMeta, netsta: str, paths: Sequence[Path],
    ) -> StationEventScore:
        network, station = netsta.split('.', 1)
        lat, lon, elev = self.coord_lookup.get(netsta, (np.nan, np.nan, 0.0))
        acc = self.config.accept
        snr_cfg = self.config.snr

        comps: Dict[str, Any] = {}
        peak = 0.0
        length_s = 0.0
        for path in paths:
            try:
                tr = read(str(path))[0]
            except Exception:
                continue
            chan = str(getattr(tr.stats, 'channel', '') or '')
            comp = chan[-1].upper() if chan else ''
            if comp not in ('Z', 'N', 'E', '1', '2', 'R', 'T'):
                continue
            data = np.asarray(tr.data, dtype=float)
            if data.size < 20 or not np.any(np.isfinite(data)):
                continue
            t0 = float(tr.stats.starttime - ev.origin)
            dt = float(tr.stats.delta)
            t = t0 + np.arange(data.size) * dt
            length_s = max(length_s, float(t[-1] - t[0]) if t.size else 0.0)
            finite = data[np.isfinite(data)]
            if finite.size:
                peak = max(peak, float(np.max(np.abs(finite))))
            sac = getattr(tr.stats, 'sac', None)
            if sac is not None:
                sla = getattr(sac, 'stla', None)
                slo = getattr(sac, 'stlo', None)
                if sla is not None and slo is not None:
                    if np.isfinite(float(sla)) and np.isfinite(float(slo)):
                        lat, lon = float(sla), float(slo)
                stel = getattr(sac, 'stel', None)
                if stel is not None and np.isfinite(float(stel)):
                    elev = float(stel)
            comps[comp] = (tr, t, data)

        n_comp = len({c if c not in ('1', '2') else
                      ('N' if c == '1' else 'E') for c in comps})
        # 优先用 Z 算 SNR
        snr = np.nan
        zkey = 'Z' if 'Z' in comps else (next(iter(comps)) if comps else None)
        if zkey is not None:
            tr, t, data = comps[zkey]
            try:
                w = tr.copy()
                w.detrend('demean')
                w.detrend('linear')
                w.taper(max_percentage=0.05, type='cosine')
                w.filter(
                    'bandpass',
                    freqmin=1.0 / snr_cfg['period_max'],
                    freqmax=1.0 / snr_cfg['period_min'],
                    corners=2,
                    zerophase=True,
                )
                d_f = np.asarray(w.data, dtype=float)
                if d_f.size == t.size:
                    nmask = t <= float(snr_cfg['noise_end_s'])
                    smask = (
                        (t >= float(snr_cfg['signal_start_s']))
                        & (t <= float(snr_cfg['signal_end_s']))
                    )
                    if (int(nmask.sum()) >= snr_cfg['min_noise_samples']
                            and int(smask.sum()) >= snr_cfg['min_signal_samples']):
                        nrms = _rms(d_f[nmask])
                        srms = _rms(d_f[smask])
                        if nrms > 0:
                            snr = srms / nrms
            except Exception:
                snr = np.nan

        reasons: List[str] = []
        if n_comp < int(acc['min_components']):
            reasons.append(f'comp={n_comp}')
        if length_s < float(acc['min_length_s']):
            reasons.append(f'len={length_s:.0f}s')
        if np.isfinite(peak) and peak > 0:
            if peak < float(acc['disp_amp_min_m']) or peak > float(acc['disp_amp_max_m']):
                # mseed 仍是 counts，振幅门限只在量级像位移时启用
                if peak < 1.0:
                    reasons.append(f'amp={peak:.2e}')
        if not np.isfinite(snr) or snr < float(snr_cfg['threshold']):
            reasons.append(f'snr={snr:.2f}' if np.isfinite(snr) else 'snr=nan')
        passed = len(reasons) == 0
        return StationEventScore(
            event=ev.name,
            network=network,
            station=station,
            n_comp=n_comp,
            length_s=length_s,
            snr=float(snr) if np.isfinite(snr) else np.nan,
            peak_m=peak,
            lat=lat,
            lon=lon,
            elev=elev,
            passed=passed,
            reason='ok' if passed else ','.join(reasons),
        )

    def score_events(
        self, events: Sequence[EventMeta], data_root: Path,
    ) -> List[StationEventScore]:
        scores: List[StationEventScore] = []
        for ev in events:
            event_dir = self._event_waveform_dir(ev, data_root, 'auto')
            if event_dir is None:
                self.logger.warning(f'跳过无波形目录的事件: {ev.name}')
                continue
            groups = self._iter_station_files(event_dir)
            self.logger.info(f'  {ev.name}: {len(groups)} 台')
            for netsta, paths in groups.items():
                scores.append(self._score_station_event(ev, netsta, paths))
        return scores

    def aggregate(
        self, scores: Sequence[StationEventScore], n_events: int,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        by_sta: Dict[str, List[StationEventScore]] = {}
        for s in scores:
            by_sta.setdefault(f'{s.network}.{s.station}', []).append(s)
        agg = self.config.aggregate
        for sid, items in sorted(by_sta.items()):
            n_avail = len(items)
            n_pass = sum(1 for x in items if x.passed)
            snrs = [x.snr for x in items if np.isfinite(x.snr)]
            med_snr = float(np.median(snrs)) if snrs else np.nan
            frac = n_pass / max(n_avail, 1)
            lat = next((x.lat for x in items if np.isfinite(x.lat)), np.nan)
            lon = next((x.lon for x in items if np.isfinite(x.lon)), np.nan)
            elev = next((x.elev for x in items if np.isfinite(x.elev)), 0.0)
            net, sta = sid.split('.', 1)
            selected = (
                n_pass >= int(agg['min_events_pass'])
                and frac >= float(agg['min_pass_frac'])
                and np.isfinite(med_snr)
                and med_snr >= float(agg['min_median_snr'])
                and np.isfinite(lat) and np.isfinite(lon)
            )
            rows.append({
                'Network': net,
                'Station': sta,
                'StationID': sid,
                'Latitude': lat,
                'Longitude': lon,
                'Elevation': elev,
                'n_events_available': n_avail,
                'n_events_pass': n_pass,
                'n_events_catalog': n_events,
                'pass_frac': frac,
                'median_snr': med_snr,
                'selected': selected,
            })
        return pd.DataFrame(rows)

    def thin_spatially(self, df: pd.DataFrame, min_km: float) -> pd.DataFrame:
        """按 median_snr 从高到低贪心抽稀。"""
        if df.empty or min_km <= 0:
            return df
        work = df.loc[df['selected']].copy()
        work = work.sort_values('median_snr', ascending=False)
        keep: List[int] = []
        lats = work['Latitude'].to_numpy()
        lons = work['Longitude'].to_numpy()
        idxs = list(work.index)
        for i, idx in enumerate(idxs):
            ok = True
            for j in keep:
                jpos = idxs.index(j)
                dist_deg = locations2degrees(lats[i], lons[i], lats[jpos], lons[jpos])
                if dist_deg * 111.2 < min_km:
                    ok = False
                    break
            if ok:
                keep.append(idx)
        out = df.copy()
        out.loc[out['selected'] & ~out.index.isin(keep), 'selected'] = False
        out.loc[out.index.isin(keep), 'selected'] = True
        return out

    def write_stations(self, df: pd.DataFrame, path: Path) -> Path:
        sel = df.loc[df['selected']].sort_values(['Network', 'Station'])
        lines = []
        for _, row in sel.iterrows():
            lines.append(
                f"{str(row['Station']):<8s} {str(row['Network']):<6s} "
                f"{float(row['Latitude']):12.4f} {float(row['Longitude']):12.4f} "
                f"{float(row['Elevation']):8.1f} {0.0:6.1f}"
            )
        path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        return path

    def write_report(
        self, df: pd.DataFrame, n_events: int, scores: Sequence[StationEventScore],
    ) -> Path:
        n_sta = len(df)
        n_sel = int(df['selected'].sum()) if not df.empty else 0
        n_pass_meas = sum(1 for s in scores if s.passed)
        lines = [
            '=' * 70,
            'EASTASIA-FWI 台站波形质量筛选报告',
            '=' * 70,
            f'事件数: {n_events}',
            f'台站-事件测量: {len(scores)}  (通过 {n_pass_meas})',
            f'独立台站: {n_sta}',
            f'入选正演台站: {n_sel}',
            '',
            '单事件门槛: 三分量 + 时长>=1800s + 20-100s SNR>=4',
            '跨事件门槛: 通过次数 / 通过率 / SNR 中位数（见 StationQualityConfig）',
            '',
            '入选台网统计:',
        ]
        if n_sel:
            vc = df.loc[df['selected'], 'Network'].value_counts()
            for net, cnt in vc.items():
                lines.append(f'  {net}: {cnt}')
        path = self.report_dir / 'station_quality_report.txt'
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return path

    def plot_map(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))
        other = df.loc[~df['selected']]
        sel = df.loc[df['selected']]
        if not other.empty:
            ax.scatter(other['Longitude'], other['Latitude'],
                       s=12, c='#bbbbbb', label='rejected', zorder=2)
        if not sel.empty:
            sc = ax.scatter(
                sel['Longitude'], sel['Latitude'],
                s=28, c=sel['median_snr'], cmap='viridis',
                vmin=4, vmax=20, label='selected', zorder=3, edgecolors='k',
                linewidths=0.3,
            )
            fig.colorbar(sc, ax=ax, label='Median SNR (20-100 s)')
        bounds = self.base_config.region
        ax.set_xlim(bounds['lon_min'], bounds['lon_max'])
        ax.set_ylim(bounds['lat_min'], bounds['lat_max'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Quality-selected stations for forward modeling')
        ax.legend(loc='lower left')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for fmt in self.config.visualization['figure_format']:
            out = self.fig_dir / f'1-8_quality_stations.{fmt}'
            fig.savefig(out, dpi=self.config.visualization['dpi'],
                        bbox_inches='tight', facecolor='white')
            self.logger.info(f'🖼️  {out}')
        plt.close(fig)

    def run(
        self,
        catalog: Path,
        data_root: Path,
        min_spacing_km: Optional[float] = None,
    ) -> pd.DataFrame:
        events = load_par_catalog(catalog)
        if not events:
            raise FileNotFoundError(f'目录为空或无法解析: {catalog}')
        self.logger.info(f'目录事件: {len(events)}  波形根目录: {data_root}')
        scores = self.score_events(events, data_root)
        df = self.aggregate(scores, len(events))
        spacing = (min_spacing_km if min_spacing_km is not None
                   else float(self.config.thinning['min_spacing_km']))
        n_before = int(df['selected'].sum()) if not df.empty else 0
        df = self.thin_spatially(df, spacing)
        n_after = int(df['selected'].sum()) if not df.empty else 0
        self.logger.info(
            f'入选 {n_before} 台 → 间距 {spacing:.0f} km 抽稀后 {n_after} 台'
        )

        stamp = datetime.now().strftime('%Y%m%d')
        csv_path = self.output_dir / f'quality_stations_{stamp}.csv'
        df.to_csv(csv_path, index=False)
        sta_path = self.output_dir / 'STATIONS_quality'
        self.write_stations(df, sta_path)
        report = self.write_report(df, len(events), scores)
        meas_path = self.report_dir / 'station_event_scores.csv'
        pd.DataFrame([s.__dict__ for s in scores]).to_csv(meas_path, index=False)
        self.plot_map(df)
        self.logger.info(f'📄 {csv_path}')
        self.logger.info(f'📄 {sta_path}')
        self.logger.info(f'📄 {report}')
        return df


def _resolve_data_root(raw: Optional[str], base: BaseConfig) -> Path:
    if raw:
        p = Path(raw).expanduser().resolve()
        if (p / 'waveforms').is_dir():
            return p / 'waveforms'
        return p
    disp = base.dirs['data'] / 'events' / 'disp_data'
    if disp.is_dir() and any(disp.iterdir()):
        return disp
    events = base.dirs['data'] / 'events'
    sessions = sorted(
        [p for p in events.iterdir()
         if p.is_dir() and p.name.split('_')[0].isdigit() and 'download' in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if sessions and (sessions[0] / 'waveforms').is_dir():
        return sessions[0] / 'waveforms'
    raise FileNotFoundError('找不到 disp_data 或下载会话 waveforms/')


def main() -> None:
    print('🎯 EASTASIA-FWI 高质量台站筛选')
    print('=' * 60)
    base = BaseConfig()
    parser = argparse.ArgumentParser(
        description='按实际波形 SNR/完整性筛选正演台站',
    )
    parser.add_argument(
        '--catalog', type=str,
        default=str(base.dirs['data'] / 'events'
                    / 'download_catalog_hybrid_30events.par'),
        help='事件目录 .par',
    )
    parser.add_argument(
        '--data-root', type=str, default=None,
        help='disp_data 或 YYYYMMDD_download_NN[/waveforms]',
    )
    parser.add_argument(
        '--min-events-pass', type=int, default=None,
        help='至少通过的事件数',
    )
    parser.add_argument(
        '--min-snr', type=float, default=None,
        help='单事件与中位数 SNR 门槛',
    )
    parser.add_argument(
        '--min-spacing-km', type=float, default=None,
        help='入选台站最小间距（km），0=不抽稀',
    )
    args = parser.parse_args()

    try:
        selector = StationQualitySelector()
        if args.min_events_pass is not None:
            selector.config.aggregate['min_events_pass'] = args.min_events_pass
        if args.min_snr is not None:
            selector.config.snr['threshold'] = args.min_snr
            selector.config.aggregate['min_median_snr'] = args.min_snr
        data_root = _resolve_data_root(args.data_root, base)
        df = selector.run(
            catalog=Path(args.catalog),
            data_root=data_root,
            min_spacing_km=args.min_spacing_km,
        )
        n_sel = int(df['selected'].sum()) if not df.empty else 0
        print(f'\n✅ 筛选完成: 入选 {n_sel} / {len(df)} 台')
        print(f'📁 STATIONS: {selector.output_dir / "STATIONS_quality"}')
    except KeyboardInterrupt:
        print('\n⚠️ 用户中断')
    except Exception as exc:
        print(f'\n❌ 失败: {exc}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print('=' * 60)


if __name__ == '__main__':
    main()
