"""
5_4_Station_Comparison.py:
台站数据对比分析与可视化模块
================================================================

功能描述:
----------
对比我们的台站数据与参考台站数据（别人的STATIONS文件），
分析相同台网和不同台网的分布差异，并生成专业的对比可视化图。

核心功能:
----------
1. ✅ 台网集合对比（共同台网、仅我方台网、仅参考方台网）
2. ✅ 台站级别对比（按 network.station 键匹配）
3. ✅ PyGMT 地图可视化（三色标注共有/独有台站）
4. ✅ 台网统计对比柱状图
5. ✅ 对比分析摘要打印

数据源:
----------
- 我们的台站数据: data/stations/EastAsia_all_stations_cleaned.csv
- 参考台站数据: specfem/C201403131706A/DATA/STATIONS

注意:
----------
参考 STATIONS 文件的列顺序为 NETWORK STATION LAT LON ELEV BURIAL（非标准），
我们的 STATIONS 文件为标准 SPECFEM 格式: STATION NETWORK LAT LON ELEV BURIAL。

输出文件:
----------
- 5-4_station_comparison_map.jpg/pdf: 台站对比分布图
- 5-4_network_comparison_bar.jpg/pdf: 台网对比柱状图
- 所有图片保存在 figures/ 目录

作者: EASTASIA-FWI Team
日期: 2025
"""
import pygmt
import pandas as pd
from pathlib import Path
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Optional, Dict, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import BaseConfig

sys.path.insert(0, str(Path(__file__).parent))
from utils.file_utils import extract_module_prefix, generate_filename


class StationComparer:
    """台站数据对比分析类"""

    def __init__(self, output_dir: Optional[str] = None):
        self.base_config = BaseConfig()

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.base_config.dirs['figures']
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.StationComparer', 'INFO'
        )

        region_bounds = self.base_config.get_region_bounds()
        self.region = [
            region_bounds['lon_min'], region_bounds['lon_max'],
            region_bounds['lat_min'], region_bounds['lat_max']
        ]

        self.module_prefix = extract_module_prefix(Path(__file__), default="5-4")

        # 配色方案
        self.colors = {
            'common': '#2ecc71',      # 绿色 - 两者共有
            'ours_only': '#3498db',    # 蓝色 - 仅我方
            'ref_only': '#e74c3c',     # 红色 - 仅参考方
        }

        self.logger.info("📡 台站对比分析器初始化完成")

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def load_reference_stations(self, path: Optional[str] = None) -> pd.DataFrame:
        """加载参考 STATIONS 文件（NETWORK STATION LAT LON ELEV BURIAL）"""
        if path is None:
            path = (self.base_config.dirs['project_root']
                    / 'specfem' / 'C201403131706A' / 'DATA' / 'STATIONS')
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"参考 STATIONS 文件不存在: {path}")

        rows = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    # 参考文件列顺序: NETWORK STATION LAT LON [ELEV] [BURIAL]
                    network = parts[0]
                    station = parts[1].rstrip('.')  # 去除尾部句点
                    lat = float(parts[2])
                    lon = float(parts[3])
                    elev = float(parts[4]) if len(parts) > 4 else 0.0
                    rows.append({
                        'Network': network,
                        'Station': station,
                        'Latitude': lat,
                        'Longitude': lon,
                        'Elevation': elev,
                    })

        df = pd.DataFrame(rows)
        self.logger.info(f"📂 加载参考台站: {len(df)} 个, {df['Network'].nunique()} 个台网")
        return df

    def load_our_stations(self, path: Optional[str] = None) -> pd.DataFrame:
        """加载我们的台站 CSV 数据"""
        if path is None:
            stations_dir = self.base_config.dirs['stations']
            candidates = sorted(stations_dir.glob('*_all_stations_cleaned.csv'))
            if not candidates:
                candidates = sorted(stations_dir.glob('*_stations*.csv'))
            if not candidates:
                raise FileNotFoundError(f"未找到台站 CSV: {stations_dir}")
            path = candidates[-1]
        path = Path(path)

        df = pd.read_csv(path)
        # 确保列名一致
        for col in ['Network', 'Station', 'Latitude', 'Longitude']:
            if col not in df.columns:
                raise ValueError(f"CSV 缺少列: {col}")
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df = df.dropna(subset=['Latitude', 'Longitude'])

        self.logger.info(f"📂 加载我方台站: {len(df)} 个, {df['Network'].nunique()} 个台网")
        return df

    # ------------------------------------------------------------------
    # 对比分析
    # ------------------------------------------------------------------
    def compare(self, ours: pd.DataFrame, ref: pd.DataFrame) -> Dict:
        """对比两份台站数据，返回分析结果"""
        our_nets = set(ours['Network'].unique())
        ref_nets = set(ref['Network'].unique())

        common_nets = our_nets & ref_nets
        ours_only_nets = our_nets - ref_nets
        ref_only_nets = ref_nets - our_nets

        # 台站级别对比 (network.station 作为键)
        ours_keys = set(ours['Network'] + '.' + ours['Station'])
        ref_keys = set(ref['Network'] + '.' + ref['Station'])

        common_keys = ours_keys & ref_keys
        ours_only_keys = ours_keys - ref_keys
        ref_only_keys = ref_keys - ours_keys

        # 按分类标注 DataFrame
        ref_tagged = ref.copy()
        ref_tagged['Key'] = ref_tagged['Network'] + '.' + ref_tagged['Station']
        ref_tagged['Category'] = ref_tagged['Key'].apply(
            lambda k: 'common' if k in common_keys else 'ref_only'
        )

        ours_tagged = ours.copy()
        ours_tagged['Key'] = ours_tagged['Network'] + '.' + ours_tagged['Station']
        ours_tagged['Category'] = ours_tagged['Key'].apply(
            lambda k: 'common' if k in common_keys else 'ours_only'
        )

        # 每台网的台站数量对比
        our_net_counts = ours.groupby('Network').size().rename('ours')
        ref_net_counts = ref.groupby('Network').size().rename('ref')
        net_compare = pd.concat([our_net_counts, ref_net_counts], axis=1).fillna(0).astype(int)
        net_compare = net_compare.sort_values(by=['ref', 'ours'], ascending=False)

        result = {
            'common_networks': sorted(common_nets),
            'ours_only_networks': sorted(ours_only_nets),
            'ref_only_networks': sorted(ref_only_nets),
            'common_stations': common_keys,
            'ours_only_stations': ours_only_keys,
            'ref_only_stations': ref_only_keys,
            'ref_tagged': ref_tagged,
            'ours_tagged': ours_tagged,
            'net_compare': net_compare,
        }

        self._print_summary(result)
        return result

    def _print_summary(self, result: Dict):
        """打印对比分析摘要"""
        n_common = len(result['common_networks'])
        n_ours = len(result['ours_only_networks'])
        n_ref = len(result['ref_only_networks'])
        self.logger.info("=" * 60)
        self.logger.info("台站数据对比分析摘要")
        self.logger.info("=" * 60)
        self.logger.info(f"台网对比:")
        self.logger.info(f"  共同台网: {n_common} 个  {result['common_networks']}")
        self.logger.info(f"  仅我方:   {n_ours} 个  {result['ours_only_networks']}")
        self.logger.info(f"  仅参考方: {n_ref} 个  {result['ref_only_networks']}")
        self.logger.info(f"台站对比:")
        self.logger.info(f"  共同台站: {len(result['common_stations'])} 个")
        self.logger.info(f"  仅我方:   {len(result['ours_only_stations'])} 个")
        self.logger.info(f"  仅参考方: {len(result['ref_only_stations'])} 个")
        self.logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    def plot_comparison_map(self, result: Dict, save_file: Optional[str] = None) -> str:
        """绘制台站对比分布地图（PyGMT）

        三色标注:
        - 绿色: 两者共有的台站
        - 蓝色: 仅我方拥有
        - 红色: 仅参考方拥有
        """
        if save_file is None:
            save_file = generate_filename(self.module_prefix, "station_comparison_map")

        self.logger.info("🗺️  开始绘制台站对比分布图...")

        ref_tagged = result['ref_tagged']
        ours_tagged = result['ours_tagged']

        fig = pygmt.Figure()

        pygmt.config(
            MAP_FRAME_TYPE="plain",
            FONT_ANNOT_PRIMARY="10p,4",
            MAP_FRAME_WIDTH="2p",
            MAP_FRAME_PEN="0.5p",
        )

        fig.coast(
            resolution="low",
            land="lightgray",
            water="lightblue",
            shorelines="0.1p",
            area_thresh=1000,
            region=self.region,
            projection="M15c",
            frame=["a10f5", "WseN"],
            borders="1/0.5p,black",
        )

        # ① 仅参考方（红）
        ref_only = ref_tagged[ref_tagged['Category'] == 'ref_only']
        if len(ref_only) > 0:
            fig.plot(
                x=ref_only['Longitude'], y=ref_only['Latitude'],
                style="t0.08c", fill=self.colors['ref_only'],
                pen="0.1p,black",
                label=f"Ref only ({len(ref_only)})",
            )

        # ② 仅我方（蓝）
        ours_only = ours_tagged[ours_tagged['Category'] == 'ours_only']
        if len(ours_only) > 0:
            fig.plot(
                x=ours_only['Longitude'], y=ours_only['Latitude'],
                style="i0.08c", fill=self.colors['ours_only'],
                pen="0.1p,black",
                label=f"Ours only ({len(ours_only)})",
            )

        # ③ 共有（绿, 画在最上层）
        common_ref = ref_tagged[ref_tagged['Category'] == 'common']
        if len(common_ref) > 0:
            fig.plot(
                x=common_ref['Longitude'], y=common_ref['Latitude'],
                style="c0.08c", fill=self.colors['common'],
                pen="0.2p,black",
                label=f"Common ({len(common_ref)})",
            )

        # 图例
        fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p0.5p")

        # 标题
        n_common_net = len(result['common_networks'])
        n_total_net = n_common_net + len(result['ours_only_networks']) + len(result['ref_only_networks'])
        fig.text(
            position="TC", offset="0/0.8c",
            text=f"Station Comparison: {n_common_net}/{n_total_net} common networks",
            font="12p,Helvetica-Bold,black",
            no_clip=True,
        )

        output_path = self.output_dir / save_file
        fig.savefig(str(output_path), dpi=300, crop=True)
        pdf_path = output_path.with_suffix('.pdf')
        fig.savefig(str(pdf_path), dpi=300, crop=True)

        self.logger.info(f"✅ 台站对比分布图保存至: {output_path}")
        return str(output_path)

    def plot_network_comparison_bar(self, result: Dict, save_file: Optional[str] = None) -> str:
        """绘制台网对比柱状图（matplotlib）

        按台网显示两方台站数量的并列柱状图，
        标注共同台网、仅我方台网、仅参考方台网。
        """
        if save_file is None:
            save_file = generate_filename(self.module_prefix, "network_comparison_bar")

        self.logger.info("📊 开始绘制台网对比柱状图...")

        net_compare = result['net_compare']
        common_nets = set(result['common_networks'])
        ours_only_nets = set(result['ours_only_networks'])
        ref_only_nets = set(result['ref_only_networks'])

        # 只显示台站数 >= 1 的台网，按参考方台站数排序
        df = net_compare[(net_compare['ours'] > 0) | (net_compare['ref'] > 0)].copy()
        df = df.sort_values(by='ref', ascending=True)

        # 如果台网太多，只显示前 60 个（按参考方台站数最大的）
        if len(df) > 60:
            # 优先保留共同台网和参考方台网
            priority = df.index.map(lambda n: 0 if n in common_nets else (1 if n in ref_only_nets else 2))
            df['_priority'] = priority.values
            df = df.sort_values(by=['_priority', 'ref'], ascending=[True, False]).head(60)
            df = df.drop(columns=['_priority']).sort_values(by='ref', ascending=True)

        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.28)))

        y = np.arange(len(df))
        bar_h = 0.35

        # 参考方柱
        bars_ref = ax.barh(y + bar_h / 2, df['ref'], bar_h, label='Reference', color='#e74c3c', alpha=0.85)
        # 我方柱
        bars_ours = ax.barh(y - bar_h / 2, df['ours'], bar_h, label='Ours', color='#3498db', alpha=0.85)

        ax.set_yticks(y)
        # 台网名加颜色标注
        labels = []
        label_colors = []
        for net in df.index:
            if net in common_nets:
                labels.append(f"★ {net}")
                label_colors.append(self.colors['common'])
            elif net in ours_only_nets:
                labels.append(f"● {net}")
                label_colors.append(self.colors['ours_only'])
            else:
                labels.append(f"▲ {net}")
                label_colors.append(self.colors['ref_only'])

        ax.set_yticklabels(labels, fontsize=7)
        for tick_label, color in zip(ax.get_yticklabels(), label_colors):
            tick_label.set_color(color)

        ax.set_xlabel('Number of Stations', fontsize=11)
        ax.set_title('Network Station Count Comparison', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)

        # 补充图例说明
        legend_text = (
            f"★ Common networks ({len(common_nets)})    "
            f"● Ours only ({len(ours_only_nets)})    "
            f"▲ Ref only ({len(ref_only_nets)})"
        )
        ax.text(0.5, -0.06, legend_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=9, style='italic')

        plt.tight_layout()

        output_path = self.output_dir / save_file
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        pdf_path = output_path.with_suffix('.pdf')
        fig.savefig(str(pdf_path), dpi=300, bbox_inches='tight')
        plt.close(fig)

        self.logger.info(f"✅ 台网对比柱状图保存至: {output_path}")
        return str(output_path)

    def plot_network_venn_summary(self, result: Dict, save_file: Optional[str] = None) -> str:
        """绘制台网 Venn 式摘要饼图 + 台站统计表"""
        if save_file is None:
            save_file = generate_filename(self.module_prefix, "network_venn_summary")

        self.logger.info("📊 开始绘制台网对比摘要图...")

        n_common = len(result['common_networks'])
        n_ours = len(result['ours_only_networks'])
        n_ref = len(result['ref_only_networks'])

        n_sta_common = len(result['common_stations'])
        n_sta_ours = len(result['ours_only_stations'])
        n_sta_ref = len(result['ref_only_stations'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左侧: 台网 Venn 式饼图
        ax = axes[0]
        sizes = [n_common, n_ours, n_ref]
        labels_pie = [
            f'Common\n{n_common}',
            f'Ours only\n{n_ours}',
            f'Ref only\n{n_ref}',
        ]
        colors_pie = [self.colors['common'], self.colors['ours_only'], self.colors['ref_only']]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels_pie, colors=colors_pie,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight('bold')
        ax.set_title('Networks', fontsize=13, fontweight='bold')

        # 右侧: 台站饼图
        ax = axes[1]
        sizes_sta = [n_sta_common, n_sta_ours, n_sta_ref]
        labels_sta = [
            f'Common\n{n_sta_common}',
            f'Ours only\n{n_sta_ours}',
            f'Ref only\n{n_sta_ref}',
        ]
        wedges2, texts2, autotexts2 = ax.pie(
            sizes_sta, labels=labels_sta, colors=colors_pie,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
        )
        for at in autotexts2:
            at.set_fontsize(10)
            at.set_fontweight('bold')
        ax.set_title('Stations', fontsize=13, fontweight='bold')

        fig.suptitle('Station Data Comparison Summary', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / save_file
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        pdf_path = output_path.with_suffix('.pdf')
        fig.savefig(str(pdf_path), dpi=300, bbox_inches='tight')
        plt.close(fig)

        self.logger.info(f"✅ 台网对比摘要图保存至: {output_path}")
        return str(output_path)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run_full_comparison(
        self,
        our_station_file: Optional[str] = None,
        ref_station_file: Optional[str] = None,
    ) -> Dict[str, str]:
        """执行完整的台站对比分析流程"""
        self.logger.info("🚀 开始台站对比分析...")

        ours = self.load_our_stations(our_station_file)
        ref = self.load_reference_stations(ref_station_file)

        result = self.compare(ours, ref)

        outputs = {}

        # 1. 地图
        map_path = self.plot_comparison_map(result)
        if map_path:
            outputs['comparison_map'] = map_path

        # 2. 柱状图
        bar_path = self.plot_network_comparison_bar(result)
        if bar_path:
            outputs['network_bar'] = bar_path

        # 3. 摘要饼图
        venn_path = self.plot_network_venn_summary(result)
        if venn_path:
            outputs['venn_summary'] = venn_path

        self.logger.info("✅ 台站对比分析全部完成!")
        return outputs


def main():
    """主函数"""
    print("📡 EASTASIA-FWI 台站数据对比分析")
    print("=" * 60)

    comparer = StationComparer()
    outputs = comparer.run_full_comparison()

    print(f"\n🎉 分析完成! 图表保存在: {comparer.output_dir}")
    for name, path in outputs.items():
        print(f"  ✅ {name}: {Path(path).name}")


if __name__ == "__main__":
    main()
