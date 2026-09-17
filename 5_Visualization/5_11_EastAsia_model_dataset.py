"""
5_11_EastAsia_model_dataset.py
东亚 FWI 波形数据集梳理与可视化模块
================================================================

功能描述:
- 扫描 data/events/EastAsia_model 目录下的 CMTSOLUTION、台站坐标、SAC 波形
- 生成标准化事件目录、台站目录、事件-波形覆盖统计 CSV
- 绘制事件分布、震源机制、台站网络、覆盖统计等可视化图

数据来源:
- src_rec/CMTSOLUTION_*          : 385 个地震事件震源参数
- src_rec/stations_by_loc/*.dat    : 台站坐标库（按数据提供者分组）
- data_obs/{EventID}/*.sac       : 观测波形 SAC 文件

作者: EASTASIA-FWI Team
日期: 2026-09-13
版本: v1.0
"""
import re
import sys
import glob
import logging
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from config.base_config import BaseConfig
from utils.file_utils import extract_module_prefix, generate_filename

# data_obs / CMTSOLUTION 事件 ID 命名规则（共 5 类，385 个事件）
# 1. specfem_datetime  (121): 202001021823A     = YYYYMMDDHHMM + 后缀
# 2. gcmt_c_full       (248): C201712151647A    = C + YYYY + MMDD + HHMM + 后缀
# 3. gcmt_b_full         (3): B201309020251A    = B + YYYY + MMDD + HHMM + 后缀
# 4. gcmt_c_short        (9): C052603D          = C + MMDDYY + 后缀（年份需读 PDE 行）
# 5. gcmt_b_short        (4): B070203D          = B + MMDDYY + 后缀（年份需读 PDE 行）
_EVENT_ID_PATTERNS: Dict[str, re.Pattern] = {
    "specfem_datetime": re.compile(
        r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})(?P<minute>\d{2})[A-Za-z]$"
    ),
    "gcmt_c_full": re.compile(
        r"^C(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})(?P<minute>\d{2})[A-Za-z]$"
    ),
    "gcmt_b_full": re.compile(
        r"^B(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})(?P<minute>\d{2})[A-Za-z]$"
    ),
    "gcmt_c_short": re.compile(r"^C(?P<month>\d{2})(?P<day>\d{2})(?P<yy>\d{2})[A-Za-z]$"),
    "gcmt_b_short": re.compile(r"^B(?P<month>\d{2})(?P<day>\d{2})(?P<yy>\d{2})[A-Za-z]$"),
}


def classify_event_id(event_id: str) -> str:
    """
    识别事件 ID / data_obs 目录的命名类型。

    Returns:
        命名类型字符串，未知则返回 'unknown'
    """
    event_id = str(event_id).strip()
    for name, pattern in _EVENT_ID_PATTERNS.items():
        if pattern.match(event_id):
            return name
    return "unknown"


def parse_event_year_from_id(event_id: str) -> Optional[int]:
    """
    从 event_name / data_obs 目录名解析年份。

    完整 datetime 格式可直接解析；短格式 GCMT ID 不含四位年份，返回 None。
    """
    event_id = str(event_id).strip()
    for name in ("specfem_datetime", "gcmt_c_full", "gcmt_b_full"):
        match = _EVENT_ID_PATTERNS[name].match(event_id)
        if match:
            year = int(match.group("year"))
            if 1970 <= year <= 2035:
                return year
    return None


def parse_event_id_metadata(event_id: str) -> Dict[str, Any]:
    """
    解析事件 ID 的结构化元数据。

    Returns:
        含 pattern、year、month、day、hour、minute 的字典
    """
    event_id = str(event_id).strip()
    pattern_name = classify_event_id(event_id)
    meta: Dict[str, Any] = {"Event_ID": event_id, "ID_Pattern": pattern_name}

    if pattern_name in ("specfem_datetime", "gcmt_c_full", "gcmt_b_full"):
        match = _EVENT_ID_PATTERNS[pattern_name].match(event_id)
        if match:
            meta.update({
                "Year": int(match.group("year")),
                "Month": int(match.group("month")),
                "Day": int(match.group("day")),
                "Hour": int(match.group("hour")),
                "Minute": int(match.group("minute")),
            })
    elif pattern_name in ("gcmt_c_short", "gcmt_b_short"):
        match = _EVENT_ID_PATTERNS[pattern_name].match(event_id)
        if match:
            meta.update({
                "Month": int(match.group("month")),
                "Day": int(match.group("day")),
                "Year_2digit": int(match.group("yy")),
            })

    return meta


def parse_pde_header_line(pde_line: str) -> Optional[Dict[str, Any]]:
    """
    解析 CMTSOLUTION 首行（PDE / PDEW / PDEQ 等变体）。

    格式: PDE[w|q] Y M D H Min Sec Lat Lon Depth mb Mag Location...
    """
    parts = pde_line.strip().split()
    if len(parts) < 12 or not parts[0].startswith("PDE"):
        return None

    try:
        record: Dict[str, Any] = {
            "Year": int(parts[1]),
            "Month": int(parts[2]),
            "Day": int(parts[3]),
            "Hour": int(parts[4]),
            "Minute": int(parts[5]),
            "Second": float(parts[6]),
            "PDE_Lat": float(parts[7]),
            "PDE_Lon": float(parts[8]),
            "PDE_Depth": float(parts[9]),
            "mb": float(parts[10]) if parts[10] != "None" else 0.0,
            "PDE_Type": parts[0],
        }
        try:
            record["MainMagnitude"] = float(parts[11])
            record["Location"] = " ".join(parts[12:])
        except ValueError:
            record["MainMagnitude"] = float(parts[12])
            record["Location"] = " ".join(parts[13:])
        return record
    except (ValueError, IndexError):
        return None


def compute_yearly_active_stations(stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每年处于运行状态的台站数量。

    Args:
        stations_df: 含 StartYear/EndYear 的台站 DataFrame

    Returns:
        年度活跃台站统计 DataFrame
    """
    if "StartYear" not in stations_df.columns:
        return pd.DataFrame()

    dated = stations_df.dropna(subset=["StartYear"]).copy()
    if dated.empty:
        return pd.DataFrame()

    year_min = int(dated["StartYear"].min())
    year_max = int(dated["EndYear"].max())
    records: List[Dict[str, Any]] = []

    for year in range(year_min, year_max + 1):
        active = dated[(dated["StartYear"] <= year) & (dated["EndYear"] >= year)]
        records.append({
            "Year": year,
            "N_Active_Stations": len(active),
            "N_New_Deployments": int((dated["StartYear"] == year).sum()),
        })

    return pd.DataFrame(records)


def compute_yearly_event_observation(
    events_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    计算每年事件的实际波形观测台站数统计。

    Args:
        events_df: 事件目录
        coverage_df: 波形覆盖统计

    Returns:
        年度观测能力 DataFrame
    """
    if events_df.empty or coverage_df.empty:
        return pd.DataFrame()

    merged = events_df.merge(coverage_df, on="Event_ID", how="inner")
    if merged.empty:
        return pd.DataFrame()

    if "Year" not in merged.columns:
        if "Date" in merged.columns:
            merged["Year"] = pd.to_datetime(merged["Date"], errors="coerce").dt.year
        else:
            merged["Year"] = np.nan

    missing_year = merged["Year"].isna()
    if missing_year.any():
        merged.loc[missing_year, "Year"] = merged.loc[missing_year, "Event_ID"].map(
            parse_event_year_from_id
        )

    merged = merged.dropna(subset=["Year"])
    merged["Year"] = merged["Year"].astype(int)

    yearly = merged.groupby("Year").agg(
        N_Events=("Event_ID", "count"),
        Mean_Stations=("N_Stations", "mean"),
        Max_Stations=("N_Stations", "max"),
        Total_SAC=("N_SAC", "sum"),
    ).reset_index()
    return yearly


def compute_optimal_observation_window(
    observation_yearly: pd.DataFrame,
    window_years: int = 5,
    min_events_in_window: int = 30,
) -> Dict[str, Any]:
    """
    滑动窗口搜索平均观测台站数最多的时间段。

    Args:
        observation_yearly: compute_yearly_event_observation 的输出
        window_years: 窗口长度（年）
        min_events_in_window: 窗口内最少事件数

    Returns:
        最优窗口信息字典
    """
    if observation_yearly.empty:
        return {}

    years = sorted(observation_yearly["Year"].astype(int).tolist())
    best: Optional[Dict[str, Any]] = None

    for start in range(years[0], years[-1] - window_years + 2):
        end = start + window_years - 1
        sub = observation_yearly[
            (observation_yearly["Year"] >= start) &
            (observation_yearly["Year"] <= end)
        ]
        n_events = int(sub["N_Events"].sum())
        if n_events < min_events_in_window:
            continue
        mean_sta = float(sub["Mean_Stations"].mean())
        if best is None or mean_sta > best["mean_stations_per_event"]:
            best = {
                "start_year": start,
                "end_year": end,
                "window_years": window_years,
                "n_events": n_events,
                "mean_stations_per_event": round(mean_sta, 1),
                "max_stations_per_event": int(sub["Max_Stations"].max()),
                "selection_criterion": "max_mean_stations_per_event",
            }

    return best or {}


class EastAsiaModelConfig:
    """EastAsia_model 数据集配置类"""

    def __init__(self):
        self.dataset = {
            "name": "EastAsia_model",
            "relative_path": "events/EastAsia_model",
        }

        self.visualization = {
            "dpi": 300,
            "pdf_dpi": 300,
            "figure_format": ["jpg", "pdf"],
            "map_projection": "M15c",
            "max_legend_networks": 20,
            "min_magnitude_for_meca": 5.5,
        }

        self.map = {
            "resolution": "full",
            "land_color": "lightgray",
            "water_color": "lightblue",
            "shoreline_pen": "0.1p",
            "area_threshold": 1000,
            "border_pen": "1/0.5p,black",
        }

        self.colors = {
            "topo_cmap": "geo",
            "depth_cmap": "seis",
            "depth_series": "0/700",
            "event_cmap": "plasma",
            "histogram_fill": "steelblue",
        }

        self.statistics = {
            "figure_size": (16, 12),
            "coverage_figure_size": (14, 10),
            "timeline_figure_size": (16, 10),
            "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"],
        }

        self.timeline = {
            "reference_stations_file": "EastAsia_stations.csv",
            "window_years": 5,
            "min_events_in_window": 30,
        }

        self.logging = {
            "level": "INFO",
            "console_output": True,
        }


class EastAsiaModelInventory:
    """EastAsia_model 数据集目录梳理类"""

    SAC_PATTERN = re.compile(
        r"^(?P<net>[A-Z0-9]+)\.(?P<sta>[A-Z0-9]+)\.(?P<locprov>[A-Z0-9]+)\.(?P<chan>[A-Z0-9]+)\.sac$",
        re.IGNORECASE,
    )

    def __init__(self, dataset_dir: Path, logger: logging.Logger):
        self.dataset_dir = Path(dataset_dir)
        self.src_rec_dir = self.dataset_dir / "src_rec"
        self.data_obs_dir = self.dataset_dir / "data_obs"
        self.catalog_dir = self.dataset_dir / "catalog"
        self.logger = logger

    def run_inventory(
        self,
        scan_waveforms: bool = True,
        build_pairs: bool = False,
    ) -> Dict[str, Path]:
        """
        执行完整数据梳理

        Args:
            scan_waveforms: 是否扫描 SAC 波形覆盖（385 个事件，约 200 万 SAC）
            build_pairs: 是否生成事件-台站配对 CSV（约 44 万行，默认关闭）

        Returns:
            生成的目录文件路径字典
        """
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        outputs: Dict[str, Path] = {}

        self.logger.info("📂 开始梳理 EastAsia_model 数据集...")

        self.scan_event_id_patterns()

        events_df = self.parse_events()
        outputs["events"] = self.catalog_dir / "events_catalog.csv"
        events_df.to_csv(outputs["events"], index=False)
        self.logger.info(f"  ✅ 事件目录: {len(events_df)} 个事件 → {outputs['events'].name}")
        if "ID_Pattern" in events_df.columns:
            for pat, cnt in events_df["ID_Pattern"].value_counts().items():
                self.logger.info(f"     {pat}: {cnt}")

        stations_df = self.parse_stations()
        outputs["stations"] = self.catalog_dir / "stations_catalog.csv"
        stations_df.to_csv(outputs["stations"], index=False)
        self.logger.info(
            f"  ✅ 台站目录: {len(stations_df)} 条记录, "
            f"{stations_df['Name'].nunique()} 个唯一台站 → {outputs['stations'].name}"
        )

        if scan_waveforms:
            coverage_df = self.scan_waveform_coverage()
            outputs["coverage"] = self.catalog_dir / "event_waveform_coverage.csv"
            coverage_df.to_csv(outputs["coverage"], index=False)
            self.logger.info(f"  ✅ 波形覆盖: {len(coverage_df)} 个事件 → {outputs['coverage'].name}")

            if build_pairs:
                pair_df = self.build_event_station_pairs(coverage_df, events_df, stations_df)
                outputs["pairs"] = self.catalog_dir / "event_station_pairs.csv"
                pair_df.to_csv(outputs["pairs"], index=False)
                self.logger.info(f"  ✅ 事件-台站对: {len(pair_df)} 条 → {outputs['pairs'].name}")
        else:
            coverage_df = pd.DataFrame()

        summary_path = self.write_summary_report(events_df, stations_df, coverage_df)
        outputs["summary"] = summary_path

        return outputs

    def parse_cmtsolution(self, cmt_path: Path) -> Dict[str, Any]:
        """
        解析 SPECFEM3D CMTSOLUTION 文件

        PDE 行格式: PDE Y M D H Min Sec Lat Lon Depth mb Mag Location...
        其中 mb 常为 0.0，Mag 位于 index 11
        """
        lines = cmt_path.read_text(encoding="utf-8", errors="replace").splitlines()
        rec: Dict[str, Any] = {"Event_ID": cmt_path.name.replace("CMTSOLUTION_", "")}

        if lines:
            pde_info = parse_pde_header_line(lines[0])
            if pde_info:
                rec.update(pde_info)
                rec["Date"] = datetime(
                    rec["Year"], rec["Month"], rec["Day"]
                ).strftime("%Y-%m-%d")

        if "Year" not in rec or pd.isna(rec.get("Year")):
            year_from_id = parse_event_year_from_id(rec["Event_ID"])
            if year_from_id:
                rec["Year"] = year_from_id

        for line in lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            try:
                rec[key] = float(value)
            except ValueError:
                rec[key] = value

        # 统一坐标字段（避免与 lowercase 字段重复）
        rec["Latitude"] = rec.get("latitude", rec.get("PDE_Lat"))
        rec["Longitude"] = rec.get("longitude", rec.get("PDE_Lon"))
        rec["Depth"] = rec.get("depth", rec.get("PDE_Depth"))
        for dup in ("latitude", "longitude", "depth"):
            rec.pop(dup, None)

        return rec

    def parse_events(self) -> pd.DataFrame:
        """从 CMTSOLUTION 文件构建事件目录"""
        cmt_files = sorted(self.src_rec_dir.glob("CMTSOLUTION_*"))
        if not cmt_files:
            raise FileNotFoundError(f"未找到 CMTSOLUTION 文件: {self.src_rec_dir}")

        events = [self.parse_cmtsolution(f) for f in cmt_files]
        df = pd.DataFrame(events)

        for col in ["Latitude", "Longitude", "Depth", "MainMagnitude"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        if "Year" not in df.columns:
            df["Year"] = np.nan
        df["ID_Pattern"] = df["Event_ID"].map(classify_event_id)

        missing = df["Year"].isna()
        if missing.any():
            df.loc[missing, "Year"] = df.loc[missing, "Event_ID"].map(parse_event_year_from_id)

        df = df.sort_values(["Year", "Event_ID"], na_position="last").reset_index(drop=True)
        return df

    def scan_event_id_patterns(self) -> pd.DataFrame:
        """统计 data_obs 目录下全部事件 ID 命名类型。"""
        records = []
        for ev_dir in sorted(self.data_obs_dir.iterdir()):
            if not ev_dir.is_dir():
                continue
            meta = parse_event_id_metadata(ev_dir.name)
            records.append(meta)

        df = pd.DataFrame(records)
        summary = df["ID_Pattern"].value_counts()
        self.logger.info("📋 事件 ID 命名类型统计:")
        for pat, cnt in summary.items():
            self.logger.info(f"  {pat}: {cnt}")

        out_path = self.catalog_dir / "event_id_patterns.csv"
        df.to_csv(out_path, index=False)
        return df

    def parse_stations(self) -> pd.DataFrame:
        """从 stations_by_loc/*.dat 构建台站目录"""
        stations_dir = self.src_rec_dir / "stations_by_loc"
        records: List[Dict[str, Any]] = []

        for dat_file in sorted(stations_dir.glob("stations_loc_*.dat")):
            provider = dat_file.stem.replace("stations_loc_", "")
            for line in dat_file.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue

                name = parts[0]
                lon, lat, elev = float(parts[1]), float(parts[2]), float(parts[3])
                if "." in name:
                    net, sta = name.split(".", 1)
                else:
                    net, sta = provider, name

                records.append({
                    "Name": name,
                    "Network": net,
                    "Station": sta,
                    "Longitude": lon,
                    "Latitude": lat,
                    "Elevation": elev,
                    "Provider": provider,
                })

        df = pd.DataFrame(records)
        if df.empty:
            raise FileNotFoundError(f"未找到台站坐标文件: {stations_dir}")

        df = df.drop_duplicates(subset=["Name"], keep="first")
        df = df.sort_values(["Network", "Station"]).reset_index(drop=True)
        df = self.enrich_stations_with_dates(df)
        return df

    def enrich_stations_with_dates(self, stations_df: pd.DataFrame) -> pd.DataFrame:
        """
        与全局台站库合并 StartDate/EndDate，补充 EastAsia_model 台站时间信息。

        Args:
            stations_df: 台站目录 DataFrame

        Returns:
            附加时间字段后的 DataFrame
        """
        ref_name = EastAsiaModelConfig().timeline["reference_stations_file"]
        ref_path = self.dataset_dir.parent.parent / "stations" / ref_name
        if not ref_path.exists():
            self.logger.warning(f"未找到参考台站时间文件: {ref_name}")
            return stations_df

        ref_df = pd.read_csv(ref_path)
        ref_df["key"] = ref_df["Network"].astype(str) + "." + ref_df["Station"].astype(str)
        stations_df = stations_df.copy()
        stations_df["key"] = stations_df["Network"].astype(str) + "." + stations_df["Station"].astype(str)

        merged = stations_df.merge(
            ref_df[["key", "StartDate", "EndDate"]],
            on="key",
            how="left",
        )
        merged["StartDate"] = pd.to_datetime(merged["StartDate"], errors="coerce")
        merged["EndDate"] = pd.to_datetime(merged["EndDate"], errors="coerce")
        merged["StartYear"] = merged["StartDate"].dt.year
        merged["EndYear"] = merged["EndDate"].dt.year.fillna(datetime.now().year)
        matched = merged["StartDate"].notna().sum()
        self.logger.info(f"  台站时间匹配: {matched}/{len(merged)}")
        return merged.drop(columns=["key"])

    @classmethod
    def parse_sac_filename(cls, filename: str) -> Optional[Dict[str, str]]:
        """从 SAC 文件名解析台站、通道、数据提供者信息"""
        match = cls.SAC_PATTERN.match(filename)
        if not match:
            return None

        locprov = match.group("locprov")
        loc_match = re.match(r"(?P<loc>\d*)(?P<provider>[A-Z].*)", locprov)
        if loc_match:
            location = loc_match.group("loc") or "00"
            provider = loc_match.group("provider")
        else:
            location, provider = locprov, "UNKNOWN"

        net = match.group("net")
        sta = match.group("sta")
        return {
            "Network": net,
            "Station": sta,
            "Name": f"{net}.{sta}",
            "Location": location,
            "Channel": match.group("chan"),
            "Provider": provider,
        }

    def scan_waveform_coverage(self) -> pd.DataFrame:
        """扫描 data_obs 下各事件的 SAC 波形覆盖统计"""
        if not self.data_obs_dir.exists():
            raise FileNotFoundError(f"波形目录不存在: {self.data_obs_dir}")

        event_dirs = sorted([d for d in self.data_obs_dir.iterdir() if d.is_dir()])
        records: List[Dict[str, Any]] = []

        for i, event_dir in enumerate(event_dirs, 1):
            event_id = event_dir.name
            n_sac = 0
            stations: set = set()
            networks: Counter = Counter()
            channels: Counter = Counter()
            providers: Counter = Counter()
            components: Counter = Counter()

            for sac_path in event_dir.glob("*.sac"):
                parsed = self.parse_sac_filename(sac_path.name)
                if parsed is None:
                    continue
                n_sac += 1
                stations.add(parsed["Name"])
                networks[parsed["Network"]] += 1
                channels[parsed["Channel"]] += 1
                providers[parsed["Provider"]] += 1
                comp = parsed["Channel"][-1] if parsed["Channel"] else "?"
                components[comp] += 1

            top_nets = networks.most_common(5)
            top_providers = providers.most_common(5)

            records.append({
                "Event_ID": event_id,
                "N_SAC": n_sac,
                "N_Stations": len(stations),
                "N_Networks": len(networks),
                "N_Channels": len(channels),
                "N_Providers": len(providers),
                "Top_Networks": ";".join(f"{n}({c})" for n, c in top_nets),
                "Top_Providers": ";".join(f"{p}({c})" for p, c in top_providers),
                "Components": ";".join(f"{c}({n})" for c, n in components.most_common()),
            })

            if i % 50 == 0 or i == len(event_dirs):
                self.logger.info(f"    波形扫描进度: {i}/{len(event_dirs)} 事件")

        return pd.DataFrame(records)

    def build_event_station_pairs(
        self,
        coverage_df: pd.DataFrame,
        events_df: pd.DataFrame,
        stations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """构建事件-台站配对表（基于 SAC 文件名，不含射线路径）"""
        station_lookup = stations_df.set_index("Name")
        event_lookup = events_df.set_index("Event_ID")
        pairs: List[Dict[str, Any]] = []

        for event_dir in sorted(self.data_obs_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            event_id = event_dir.name
            if event_id not in event_lookup.index:
                continue

            event_stations: set = set()
            for sac_path in event_dir.glob("*.sac"):
                parsed = self.parse_sac_filename(sac_path.name)
                if parsed:
                    event_stations.add(parsed["Name"])

            ev = event_lookup.loc[event_id]
            for sta_name in sorted(event_stations):
                rec = {
                    "Event_ID": event_id,
                    "Station_Name": sta_name,
                    "Event_Lat": ev.get("Latitude"),
                    "Event_Lon": ev.get("Longitude"),
                    "Event_Depth": ev.get("Depth"),
                    "Event_Mag": ev.get("MainMagnitude"),
                }
                if sta_name in station_lookup.index:
                    st = station_lookup.loc[sta_name]
                    rec.update({
                        "Network": st["Network"],
                        "Station": st["Station"],
                        "Station_Lat": st["Latitude"],
                        "Station_Lon": st["Longitude"],
                        "Station_Elev": st["Elevation"],
                    })
                pairs.append(rec)

        return pd.DataFrame(pairs)

    def write_summary_report(
        self,
        events_df: pd.DataFrame,
        stations_df: pd.DataFrame,
        coverage_df: pd.DataFrame,
    ) -> Path:
        """生成文本摘要报告"""
        report_path = self.catalog_dir / "inventory_summary.txt"
        lines = [
            "EastAsia_model Dataset Inventory Summary",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "EVENT ID NAMING (data_obs / CMTSOLUTION)",
            "-" * 40,
            "  specfem_datetime  (121): 202001021823A",
            "  gcmt_c_full       (248): C201712151647A",
            "  gcmt_b_full         (3): B201309020251A",
            "  gcmt_c_short        (9): C052603D  (year from PDE line)",
            "  gcmt_b_short        (4): B070203D  (year from PDE line)",
            "",
            "EVENTS",
            "-" * 40,
            f"  Total events:     {len(events_df)}",
        ]
        if "ID_Pattern" in events_df.columns:
            lines.append("  ID pattern counts:")
            for pat, cnt in events_df["ID_Pattern"].value_counts().items():
                lines.append(f"    {pat}: {cnt}")

        if "Date" in events_df.columns and events_df["Date"].notna().any():
            lines.append(f"  Time range:       {events_df['Date'].min()} — {events_df['Date'].max()}")
        if "MainMagnitude" in events_df.columns:
            mag = events_df["MainMagnitude"].dropna()
            lines.append(f"  Magnitude range:  {mag.min():.1f} — {mag.max():.1f}")
        if "Depth" in events_df.columns:
            dep = events_df["Depth"].dropna()
            lines.append(f"  Depth range:      {dep.min():.1f} — {dep.max():.1f} km")

        lines.extend([
            "",
            "STATIONS",
            "-" * 40,
            f"  Unique stations:  {stations_df['Name'].nunique()}",
            f"  Networks:         {stations_df['Network'].nunique()}",
            f"  Providers:        {', '.join(sorted(stations_df['Provider'].unique()))}",
        ])

        top_nets = stations_df["Network"].value_counts().head(10)
        lines.append("  Top 10 networks:")
        for net, cnt in top_nets.items():
            lines.append(f"    {net}: {cnt}")

        if not coverage_df.empty:
            lines.extend([
                "",
                "WAVEFORM COVERAGE",
                "-" * 40,
                f"  Events with SAC:  {len(coverage_df)}",
                f"  Total SAC files:  {coverage_df['N_SAC'].sum():,.0f}",
                f"  Avg SAC/event:    {coverage_df['N_SAC'].mean():.0f}",
                f"  Avg sta/event:    {coverage_df['N_Stations'].mean():.0f}",
                f"  Min sta/event:    {coverage_df['N_Stations'].min()}",
                f"  Max sta/event:    {coverage_df['N_Stations'].max()}",
            ])

        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"  ✅ 摘要报告 → {report_path.name}")
        return report_path


class EastAsiaModelPlotter:
    """EastAsia_model 数据集可视化类"""

    def __init__(self, output_dir: Optional[str] = None):
        self.base_config = BaseConfig()
        self.config = EastAsiaModelConfig()

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.base_config.dirs["figures"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self.base_config.setup_logger(
            "EASTASIA-FWI.EastAsiaModelPlotter",
            self.config.logging["level"],
        )

        bounds = self.base_config.get_region_bounds()
        self.region = [
            bounds["lon_min"], bounds["lon_max"],
            bounds["lat_min"], bounds["lat_max"],
        ]

        self.dataset_dir = self.base_config.dirs["data"] / self.config.dataset["relative_path"]
        self.catalog_dir = self.dataset_dir / "catalog"
        self.data_dirs = self._setup_geo_data_dirs()
        self.module_prefix = extract_module_prefix(Path(__file__), default="5-11")

        tab20 = matplotlib.colormaps["tab20"]
        tab20b = matplotlib.colormaps["tab20b"]
        self.network_colors = [
            f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            for r, g, b, _ in [tab20(i) for i in range(20)] + [tab20b(i) for i in range(20)]
        ]

        self._setup_pygmt_config()
        self.logger.info("🎯 EastAsia_model 数据集可视化器初始化完成")

    def _setup_geo_data_dirs(self) -> Dict[str, Path]:
        vis_root = self.base_config.dirs["project_root"] / "5_Visualization"
        data_base = vis_root / "data_EastAsia"
        return {
            "base": data_base,
            "elevation": data_base / "elevation",
            "volcano": data_base / "volcano",
            "tectonics": data_base / "tectonics",
            "faults": data_base / "gem-global-active-faults-master",
            "slab": data_base / "Slab2",
        }

    def _setup_pygmt_config(self) -> None:
        pygmt.config(
            MAP_FRAME_TYPE="plain",
            MAP_GRID_PEN_PRIMARY="0.3p,dimgrey",
            FONT_ANNOT_PRIMARY="10p,4",
            FONT_LABEL="10p,28,black",
            MAP_FRAME_WIDTH="2p",
            MAP_FRAME_PEN="0.5p",
        )

    def _save_figure(self, fig: pygmt.Figure, description: str) -> str:
        """保存 PyGMT 图为 jpg + pdf"""
        jpg_name = generate_filename(self.module_prefix, description, "jpg")
        jpg_path = self.output_dir / jpg_name
        fig.savefig(str(jpg_path), dpi=self.config.visualization["dpi"], crop=True)

        if "pdf" in self.config.visualization["figure_format"]:
            pdf_path = jpg_path.with_suffix(".pdf")
            fig.savefig(str(pdf_path), dpi=self.config.visualization["pdf_dpi"], crop=True)

        return str(jpg_path)

    def _save_matplotlib(self, description: str) -> str:
        """保存 matplotlib 图为 jpg + pdf"""
        jpg_name = generate_filename(self.module_prefix, description, "jpg")
        jpg_path = self.output_dir / jpg_name
        plt.savefig(str(jpg_path), dpi=300, bbox_inches="tight", facecolor="white")

        if "pdf" in self.config.visualization["figure_format"]:
            pdf_path = jpg_path.with_suffix(".pdf")
            plt.savefig(str(pdf_path), dpi=300, bbox_inches="tight", facecolor="white")

        plt.close()
        return str(jpg_path)

    def _prepare_elevation_grid(self) -> str:
        elevation_file = self.data_dirs["elevation"] / "EastAsia.grd"
        if not elevation_file.exists():
            elevation_file.parent.mkdir(parents=True, exist_ok=True)
            pygmt.grdcut(
                grid="@earth_relief_01m",
                region=self.region,
                outgrid=str(elevation_file),
            )
        return str(elevation_file)

    def _add_geological_features(self, fig: pygmt.Figure) -> None:
        """添加火山、断层、板块边界等地质要素"""
        volcano_file = self.data_dirs["volcano"] / "VHolo.csv"
        if volcano_file.exists():
            fig.plot(data=str(volcano_file), style="kvolcano/0.25c", fill="red", pen="0.1p,black")

        faults_file = self.data_dirs["faults"] / "gem_active_faults.txt"
        if faults_file.exists():
            fig.plot(data=str(faults_file), pen="0.5p,black")

        boundaries_file = self.data_dirs["tectonics"] / "boundaries.gmt"
        if boundaries_file.exists():
            fig.plot(data=str(boundaries_file), pen="0.5p,black")

    def _get_network_color_map(self, networks: pd.Series) -> Dict[str, str]:
        counts = networks.value_counts()
        return {net: self.network_colors[i % len(self.network_colors)] for i, net in enumerate(counts.index)}

    def load_catalogs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载已生成的目录 CSV，若不存在则先执行 inventory"""
        events_path = self.catalog_dir / "events_catalog.csv"
        stations_path = self.catalog_dir / "stations_catalog.csv"
        coverage_path = self.catalog_dir / "event_waveform_coverage.csv"

        if not events_path.exists() or not stations_path.exists():
            self.logger.info("目录文件不存在，先执行数据梳理...")
            inventory = EastAsiaModelInventory(self.dataset_dir, self.logger)
            inventory.run_inventory(scan_waveforms=True)

        events_df = pd.read_csv(events_path, parse_dates=["Date"])
        stations_df = pd.read_csv(stations_path)
        coverage_df = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()

        return events_df, stations_df, coverage_df

    def plot_event_distribution_map(self, events_df: pd.DataFrame) -> str:
        """绘制事件空间分布图（按深度着色）"""
        self.logger.info("🗺️ 绘制事件分布图...")

        elevation = self._prepare_elevation_grid()
        fig = pygmt.Figure()

        pygmt.makecpt(cmap=self.config.colors["depth_cmap"], series=self.config.colors["depth_series"])

        fig.grdimage(
            grid=elevation,
            region=self.region,
            projection=self.config.visualization["map_projection"],
            frame=["a10f1", "WSen"],
            cmap=self.config.colors["topo_cmap"],
        )
        self._add_geological_features(fig)

        fig.plot(
            x=events_df["Longitude"],
            y=events_df["Latitude"],
            fill=events_df["Depth"],
            cmap=True,
            style="c0.15c",
            pen="0.2p,black",
        )

        fig.coast(resolution="full", shorelines="1/0.01p")
        fig.colorbar(frame=["a100", "x+lDepth (km)"], position="JMR+o0.5c/0c+w8c")

        fig.text(
            x=(self.region[0] + self.region[1]) / 2,
            y=self.region[3] + 1.5,
            text=f"EastAsia_model Events (N={len(events_df)})",
            font="14p,Helvetica-Bold,black",
            justify="BC",
        )

        return self._save_figure(fig, "event_distribution_map")

    def plot_focal_mechanisms_map(self, events_df: pd.DataFrame) -> str:
        """绘制震源机制解分布图（基于矩张量分量）"""
        min_mag = self.config.visualization["min_magnitude_for_meca"]
        self.logger.info(f"🎯 绘制震源机制解分布图 (M≥{min_mag})...")

        elevation = self._prepare_elevation_grid()
        fig = pygmt.Figure()

        pygmt.makecpt(cmap=self.config.colors["depth_cmap"], series=self.config.colors["depth_series"])

        fig.grdimage(
            grid=elevation,
            region=self.region,
            projection=self.config.visualization["map_projection"],
            frame=["a10f1", "WSen"],
            cmap=self.config.colors["topo_cmap"],
        )
        self._add_geological_features(fig)

        # PyGMT "mt" 约定: mff/mrf/mtf 对应 GCMT 的 mpp/mrp/mtp
        mt_cols = ["mrr", "mtt", "mpp", "mrt", "mrp", "mtp"]
        large_events = events_df[events_df["MainMagnitude"] >= min_mag]
        meca_count = 0

        for _, ev in large_events.iterrows():
            if not all(c in ev.index and pd.notna(ev[c]) for c in mt_cols):
                continue
            try:
                mrr = float(ev["mrr"])
                exp = int(np.log10(abs(mrr)))
                norm = 10 ** exp
                spec = {
                    "mrr": mrr / norm,
                    "mtt": float(ev["mtt"]) / norm,
                    "mff": float(ev["mpp"]) / norm,
                    "mrt": float(ev["mrt"]) / norm,
                    "mrf": float(ev["mrp"]) / norm,
                    "mtf": float(ev["mtp"]) / norm,
                    "exponent": exp,
                }
                scale = 0.15 + (ev["MainMagnitude"] - min_mag) * 0.04
                fig.meca(
                    spec=spec,
                    scale=f"{min(max(scale, 0.1), 0.5):.2f}c",
                    longitude=float(ev["Longitude"]),
                    latitude=float(ev["Latitude"]),
                    depth=float(ev["Depth"]),
                    cmap=True,
                    extension_fill="cornsilk",
                    pen="0.2p,gray30,solid",
                )
                meca_count += 1
            except Exception as exc:
                self.logger.warning(f"震源机制绘制失败 {ev.get('Event_ID')}: {exc}")

        fig.coast(resolution="full", shorelines="1/0.01p")
        fig.text(
            x=(self.region[0] + self.region[1]) / 2,
            y=self.region[3] + 1.5,
            text=f"Focal Mechanisms (N={meca_count}, M≥{min_mag})",
            font="14p,Helvetica-Bold,black",
            justify="BC",
        )

        self.logger.info(f"  绘制 {meca_count} 个震源机制解")
        return self._save_figure(fig, "focal_mechanisms_map")

    def plot_station_network_map(self, stations_df: pd.DataFrame) -> str:
        """绘制台站网络分布图"""
        self.logger.info("📡 绘制台站网络分布图...")

        color_map = self._get_network_color_map(stations_df["Network"])
        stations_df = stations_df.copy()
        stations_df["Color"] = stations_df["Network"].map(color_map)

        fig = pygmt.Figure()
        map_cfg = self.config.map
        fig.coast(
            resolution=map_cfg["resolution"],
            land=map_cfg["land_color"],
            water=map_cfg["water_color"],
            shorelines=map_cfg["shoreline_pen"],
            area_thresh=map_cfg["area_threshold"],
            region=self.region,
            projection=self.config.visualization["map_projection"],
            frame=["a10f1", "WseN"],
            borders=map_cfg["border_pen"],
        )

        for color, group in stations_df.groupby("Color"):
            fig.plot(
                x=group["Longitude"],
                y=group["Latitude"],
                style="t0.08c",
                fill=color,
                pen="0.05p,black",
            )

        fig.text(
            x=(self.region[0] + self.region[1]) / 2,
            y=self.region[3] + 1.5,
            text=f"Station Network (N={len(stations_df)}, {stations_df['Network'].nunique()} networks)",
            font="14p,Helvetica-Bold,black",
            justify="BC",
        )

        return self._save_figure(fig, "station_network_map")

    def plot_event_station_overview(
        self,
        events_df: pd.DataFrame,
        stations_df: pd.DataFrame,
    ) -> str:
        """绘制事件+台站综合分布图"""
        self.logger.info("🌏 绘制事件-台站综合分布图...")

        fig = pygmt.Figure()
        fig.coast(
            region=self.region,
            projection=self.config.visualization["map_projection"],
            frame=["a10f1", "WSen"],
            land="gray90",
            water="lightblue",
            shorelines="0.5p",
            borders="1/0.5p,black",
        )

        fig.plot(
            x=stations_df["Longitude"],
            y=stations_df["Latitude"],
            style="t0.06c",
            fill="gray70",
            pen="0.02p,gray50",
        )

        fig.plot(
            x=events_df["Longitude"],
            y=events_df["Latitude"],
            fill=events_df["MainMagnitude"],
            cmap="hot",
            style="c0.2c",
            pen="0.3p,black",
        )

        fig.text(
            x=(self.region[0] + self.region[1]) / 2,
            y=self.region[3] + 1.5,
            text=f"Events (N={len(events_df)}) + Stations (N={len(stations_df)})",
            font="14p,Helvetica-Bold,black",
            justify="BC",
        )

        return self._save_figure(fig, "event_station_overview")

    def plot_event_statistics(self, events_df: pd.DataFrame) -> str:
        """绘制事件统计图（震级、深度、时间）"""
        self.logger.info("📊 绘制事件统计图...")

        fig, axes = plt.subplots(2, 2, figsize=self.config.statistics["figure_size"])
        fig.suptitle("EastAsia_model Event Statistics", fontsize=16, fontweight="bold")
        colors = self.config.statistics["colors"]

        mag = events_df["MainMagnitude"].dropna()
        axes[0, 0].hist(mag, bins=20, color=colors[0], edgecolor="black", alpha=0.8)
        axes[0, 0].set_xlabel("Magnitude")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Magnitude Distribution")
        axes[0, 0].grid(True, alpha=0.3)

        dep = events_df["Depth"].dropna()
        axes[0, 1].hist(dep, bins=30, color=colors[1], edgecolor="black", alpha=0.8)
        axes[0, 1].set_xlabel("Depth (km)")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Depth Distribution")
        axes[0, 1].grid(True, alpha=0.3)

        if "Date" in events_df.columns:
            yearly = events_df.groupby(events_df["Date"].dt.year).size()
            axes[1, 0].bar(yearly.index, yearly.values, color=colors[2], edgecolor="black")
            axes[1, 0].set_xlabel("Year")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].set_title("Events per Year")
            axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].scatter(
            events_df["Longitude"], events_df["Latitude"],
            c=events_df["Depth"], cmap="plasma_r", s=20, alpha=0.7, edgecolors="black", linewidths=0.3,
        )
        axes[1, 1].set_xlim(self.region[0], self.region[1])
        axes[1, 1].set_ylim(self.region[2], self.region[3])
        axes[1, 1].set_xlabel("Longitude")
        axes[1, 1].set_ylabel("Latitude")
        axes[1, 1].set_title("Event Locations (colored by depth)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save_matplotlib("event_statistics")

    def plot_station_statistics(self, stations_df: pd.DataFrame) -> str:
        """绘制台站统计图"""
        self.logger.info("📊 绘制台站统计图...")

        fig, axes = plt.subplots(2, 2, figsize=self.config.statistics["figure_size"])
        fig.suptitle("EastAsia_model Station Statistics", fontsize=16, fontweight="bold")
        colors = self.config.statistics["colors"]

        net_counts = stations_df["Network"].value_counts()
        top15 = net_counts.head(15)
        axes[0, 0].barh(range(len(top15)), top15.values, color=colors[0], alpha=0.8)
        axes[0, 0].set_yticks(range(len(top15)))
        axes[0, 0].set_yticklabels(top15.index, fontsize=8)
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_xlabel("Station Count")
        axes[0, 0].set_title("Top 15 Networks")
        axes[0, 0].grid(True, alpha=0.3)

        prov_counts = stations_df["Provider"].value_counts()
        axes[0, 1].pie(
            prov_counts.values, labels=prov_counts.index,
            autopct="%1.1f%%", colors=colors[:len(prov_counts)], startangle=90,
        )
        axes[0, 1].set_title("Stations by Data Provider")

        axes[1, 0].scatter(
            stations_df["Longitude"], stations_df["Latitude"],
            c=stations_df["Elevation"], cmap="terrain", s=8, alpha=0.6,
        )
        axes[1, 0].set_xlim(self.region[0], self.region[1])
        axes[1, 0].set_ylim(self.region[2], self.region[3])
        axes[1, 0].set_xlabel("Longitude")
        axes[1, 0].set_ylabel("Latitude")
        axes[1, 0].set_title("Station Locations (colored by elevation)")
        axes[1, 0].grid(True, alpha=0.3)

        elev = stations_df["Elevation"].dropna()
        axes[1, 1].hist(elev, bins=40, color=colors[3], edgecolor="black", alpha=0.8)
        axes[1, 1].set_xlabel("Elevation (m)")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Elevation Distribution")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save_matplotlib("station_statistics")

    def build_timeline_analysis(
        self,
        stations_df: pd.DataFrame,
        events_df: pd.DataFrame,
        coverage_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        构建台站部署与波形观测时间线分析。

        Returns:
            (部署年度统计, 观测年度统计, 最优事件窗口)
        """
        deployment_yearly = compute_yearly_active_stations(stations_df)
        observation_yearly = compute_yearly_event_observation(events_df, coverage_df)
        optimal = compute_optimal_observation_window(
            observation_yearly,
            window_years=self.config.timeline["window_years"],
            min_events_in_window=self.config.timeline["min_events_in_window"],
        )

        if optimal:
            self.logger.info(
                f"📅 最优观测窗口: {optimal['start_year']}–{optimal['end_year']} "
                f"(均值 {optimal['mean_stations_per_event']:.0f} 台站/事件, "
                f"{optimal['n_events']} 个事件)"
            )

        deploy_path = self.catalog_dir / "station_deployment_yearly.csv"
        obs_path = self.catalog_dir / "event_observation_yearly.csv"
        window_path = self.catalog_dir / "optimal_event_window.json"

        if not deployment_yearly.empty:
            deployment_yearly.to_csv(deploy_path, index=False)
        if not observation_yearly.empty:
            observation_yearly.to_csv(obs_path, index=False)
        if optimal:
            import json
            window_path.write_text(
                json.dumps(optimal, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        return deployment_yearly, observation_yearly, optimal

    def plot_station_timeline(
        self,
        stations_df: pd.DataFrame,
        events_df: pd.DataFrame,
        coverage_df: pd.DataFrame,
        optimal_window: Optional[Dict[str, Any]] = None,
    ) -> str:
        """绘制台站部署与波形观测时间线综合图。"""
        self.logger.info("📅 绘制台站观测 Timeline...")

        deployment_yearly, observation_yearly, optimal = self.build_timeline_analysis(
            stations_df, events_df, coverage_df,
        )
        if optimal_window is None:
            optimal_window = optimal

        colors = self.config.statistics["colors"]
        fig, axes = plt.subplots(2, 2, figsize=self.config.statistics["timeline_figure_size"])
        fig.suptitle(
            "EastAsia_model Station Observation Timeline",
            fontsize=16, fontweight="bold",
        )

        # 1. 每年活跃台站数（部署时间线）
        ax = axes[0, 0]
        if not deployment_yearly.empty:
            ax.fill_between(
                deployment_yearly["Year"],
                deployment_yearly["N_Active_Stations"],
                alpha=0.25, color=colors[0],
            )
            ax.plot(
                deployment_yearly["Year"],
                deployment_yearly["N_Active_Stations"],
                "o-", color=colors[0], linewidth=2, markersize=4,
                label="Active deployed stations",
            )
            peak_row = deployment_yearly.loc[
                deployment_yearly["N_Active_Stations"].idxmax()
            ]
            ax.axvline(peak_row["Year"], color="gray", linestyle=":", alpha=0.7)
            ax.annotate(
                f"Peak: {int(peak_row['N_Active_Stations'])} ({int(peak_row['Year'])})",
                xy=(peak_row["Year"], peak_row["N_Active_Stations"]),
                xytext=(8, -15), textcoords="offset points", fontsize=8,
            )
        ax.set_title("Active Stations by Year (Deployment)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Active Stations")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # 2. 每年新增部署（按 Provider）
        ax = axes[0, 1]
        if "StartYear" in stations_df.columns and "Provider" in stations_df.columns:
            dated = stations_df.dropna(subset=["StartYear"])
            top_providers = dated["Provider"].value_counts().head(6).index
            x_years = sorted(dated["StartYear"].astype(int).unique())
            bottom = np.zeros(len(x_years))
            for i, prov in enumerate(top_providers):
                prov_data = dated[dated["Provider"] == prov]
                yearly = prov_data.groupby("StartYear").size().reindex(x_years, fill_value=0)
                ax.bar(
                    x_years, yearly.values, bottom=bottom,
                    label=prov, color=colors[i % len(colors)], alpha=0.85, width=0.8,
                )
                bottom = bottom + yearly.values.astype(float)
            ax.set_title("New Station Deployments by Provider")
            ax.set_xlabel("Year")
            ax.set_ylabel("New Deployments")
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No deployment dates", ha="center", va="center")

        # 3. 每年事件观测台站数（实际波形覆盖）
        ax = axes[1, 0]
        if not observation_yearly.empty:
            ax.bar(
                observation_yearly["Year"],
                observation_yearly["Mean_Stations"],
                color=colors[2], alpha=0.7, label="Mean stations/event",
            )
            ax.plot(
                observation_yearly["Year"],
                observation_yearly["Max_Stations"],
                "s--", color=colors[1], linewidth=1.5, markersize=4,
                label="Max stations/event",
            )
            if optimal_window:
                ax.axvspan(
                    optimal_window["start_year"] - 0.5,
                    optimal_window["end_year"] + 0.5,
                    alpha=0.15, color="gold",
                    label=(
                        f"Optimal window "
                        f"({optimal_window['start_year']}–{optimal_window['end_year']})"
                    ),
                )
        ax.set_title("Waveform Observation Capacity by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Stations per Event")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 4. 最优窗口内事件数 vs 台站覆盖
        ax = axes[1, 1]
        if not observation_yearly.empty and optimal_window:
            in_win = observation_yearly[
                (observation_yearly["Year"] >= optimal_window["start_year"]) &
                (observation_yearly["Year"] <= optimal_window["end_year"])
            ]
            out_win = observation_yearly[
                (observation_yearly["Year"] < optimal_window["start_year"]) |
                (observation_yearly["Year"] > optimal_window["end_year"])
            ]
            labels = [
                f"In window\n({optimal_window['start_year']}–{optimal_window['end_year']})",
                "Outside window",
            ]
            event_counts = [int(in_win["N_Events"].sum()), int(out_win["N_Events"].sum())]
            mean_stations = [
                float(in_win["Mean_Stations"].mean()) if len(in_win) else 0,
                float(out_win["Mean_Stations"].mean()) if len(out_win) else 0,
            ]
            x = np.arange(2)
            w = 0.35
            ax.bar(x - w / 2, event_counts, w, color=colors[3], alpha=0.8, label="N events")
            ax2 = ax.twinx()
            ax2.bar(x + w / 2, mean_stations, w, color=colors[4], alpha=0.8, label="Mean sta/event")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Number of Events")
            ax2.set_ylabel("Mean Stations per Event")
            ax.set_title("Optimal Window vs Outside")
            lines1, lab1 = ax.get_legend_handles_labels()
            lines2, lab2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")

        plt.tight_layout()
        return self._save_matplotlib("station_observation_timeline")

    def plot_waveform_coverage(self, coverage_df: pd.DataFrame) -> str:
        """绘制波形覆盖统计图"""
        if coverage_df.empty:
            self.logger.warning("无波形覆盖数据，跳过")
            return ""

        self.logger.info("📈 绘制波形覆盖统计图...")

        fig, axes = plt.subplots(2, 2, figsize=self.config.statistics["coverage_figure_size"])
        fig.suptitle("EastAsia_model Waveform Coverage Statistics", fontsize=16, fontweight="bold")
        colors = self.config.statistics["colors"]

        axes[0, 0].hist(coverage_df["N_SAC"], bins=30, color=colors[0], edgecolor="black", alpha=0.8)
        axes[0, 0].set_xlabel("SAC files per event")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title(f"SAC Count Distribution (total={coverage_df['N_SAC'].sum():,.0f})")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(coverage_df["N_Stations"], bins=30, color=colors[1], edgecolor="black", alpha=0.8)
        axes[0, 1].set_xlabel("Stations per event")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title(f"Station Count Distribution (mean={coverage_df['N_Stations'].mean():.0f})")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].scatter(
            coverage_df["N_Stations"], coverage_df["N_SAC"],
            alpha=0.5, c=colors[2], edgecolors="black", linewidths=0.3,
        )
        axes[1, 0].set_xlabel("Stations per event")
        axes[1, 0].set_ylabel("SAC files per event")
        axes[1, 0].set_title("SAC vs Station Count")
        axes[1, 0].grid(True, alpha=0.3)

        sorted_cov = coverage_df.sort_values("N_Stations", ascending=False).head(30)
        axes[1, 1].barh(
            range(len(sorted_cov)), sorted_cov["N_Stations"].values,
            color=colors[3], alpha=0.8,
        )
        axes[1, 1].set_yticks(range(len(sorted_cov)))
        axes[1, 1].set_yticklabels(sorted_cov["Event_ID"], fontsize=6)
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_xlabel("Station Count")
        axes[1, 1].set_title("Top 30 Events by Station Count")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save_matplotlib("waveform_coverage_statistics")

    def plot_all(
        self,
        scan_waveforms: bool = True,
        build_pairs: bool = False,
    ) -> Dict[str, str]:
        """
        执行完整的数据梳理与可视化流程

        Args:
            scan_waveforms: 是否扫描 SAC 波形
            build_pairs: 是否生成事件-台站配对 CSV（约 44 万行）

        Returns:
            生成的图表和目录文件路径字典
        """
        self.logger.info("🎨 开始 EastAsia_model 完整梳理与可视化...")

        inventory = EastAsiaModelInventory(self.dataset_dir, self.logger)
        catalog_files = inventory.run_inventory(
            scan_waveforms=scan_waveforms,
            build_pairs=build_pairs,
        )

        events_df, stations_df, coverage_df = self.load_catalogs()
        results: Dict[str, str] = {k: str(v) for k, v in catalog_files.items()}

        plotters = [
            ("event_distribution_map", lambda: self.plot_event_distribution_map(events_df)),
            ("focal_mechanisms_map", lambda: self.plot_focal_mechanisms_map(events_df)),
            ("station_network_map", lambda: self.plot_station_network_map(stations_df)),
            ("event_station_overview", lambda: self.plot_event_station_overview(events_df, stations_df)),
            ("event_statistics", lambda: self.plot_event_statistics(events_df)),
            ("station_statistics", lambda: self.plot_station_statistics(stations_df)),
            ("station_timeline", lambda: self.plot_station_timeline(
                stations_df, events_df, coverage_df,
            )),
            ("waveform_coverage", lambda: self.plot_waveform_coverage(coverage_df)),
        ]

        for name, plot_fn in plotters:
            try:
                path = plot_fn()
                if path:
                    results[name] = path
            except Exception as exc:
                self.logger.error(f"  ❌ {name} 绘制失败: {exc}")

        self.logger.info("✅ EastAsia_model 梳理与可视化完成!")
        return results


def main() -> None:
    """主函数"""
    print("🎯 EASTASIA-FWI EastAsia_model 数据集梳理与可视化")
    print("=" * 70)

    try:
        plotter = EastAsiaModelPlotter()
        print(f"📂 数据目录: {plotter.dataset_dir}")
        print(f"📁 输出目录: {plotter.output_dir}")
        print(f"📁 目录输出: {plotter.catalog_dir}")
        print()

        results = plotter.plot_all(scan_waveforms=True, build_pairs=False)

        print("\n🎉 处理完成!")
        print("=" * 70)
        print("📊 生成的目录文件:")
        for key in ["events", "stations", "coverage", "summary"]:
            if key in results:
                print(f"  ✅ {key}: {Path(results[key]).name}")

        print("\n📈 生成的图表:")
        for key, path in results.items():
            if key not in ("events", "stations", "coverage", "pairs", "summary"):
                print(f"  ✅ {key}: {Path(path).name}")

        print("=" * 70)

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as exc:
        print(f"\n❌ 处理失败: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
