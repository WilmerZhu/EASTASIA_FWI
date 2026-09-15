"""
EASTASIA-FWI GLL 模型逐点对比工具
=================================

逐点对比两套 GLL 二进制文件：
  A = model_updated/  (原作者 FWEA23)
  B = GLL/            (convert_nc_to_gll 生成)

无需 solver_data.bin 即可运行基本对比；
若提供 DATABASES_MPI 路径则额外输出地理坐标地图。

用法（服务器 conda eastasia_fwi 环境）:
    conda activate eastasia_fwi
    cd specfem
    python compare_gll.py
    python compare_gll.py --mesh-dir FWEA23/DATABASES_MPI   # 加地图

作者: EASTASIA-FWI Team
日期: 2026-04
"""

import sys
import struct
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import argparse
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ─── 常量 ────────────────────────────────────────────────────
NGLLX, NGLLY, NGLLZ = 5, 5, 5
NGLL3 = NGLLX * NGLLY * NGLLZ  # 125
R_EARTH_KM = 6371.0

SCRIPT_DIR = Path(__file__).parent
DIR_A = SCRIPT_DIR / 'model_updated'
DIR_B = SCRIPT_DIR / 'GLL'
OUTPUT_DIR = SCRIPT_DIR / 'figures' / 'gll_compare'
LABEL_A = 'Author (model_updated)'
LABEL_B = 'Ours (convert_nc_to_gll)'

PARAMS = ['vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho']
PARAM_UNITS = {
    'vpv': 'km/s', 'vph': 'km/s', 'vsv': 'km/s', 'vsh': 'km/s',
    'eta': '', 'rho': 'g/cm³', 'qmu': '',
}

DEPTH_SLICES_KM = [10, 50, 100, 200, 410, 660]


# ─── 读取工具 ────────────────────────────────────────────────
def _read_fortran_record(f) -> bytes:
    rec_len = struct.unpack('i', f.read(4))[0]
    data = f.read(rec_len)
    tail = struct.unpack('i', f.read(4))[0]
    if tail != rec_len:
        raise ValueError(f"Fortran record marker mismatch: {rec_len} vs {tail}")
    return data


def read_gll_flat(path: Path) -> np.ndarray:
    """读取 GLL bin → 1D float32 数组（NGLL3 * nspec 个值）"""
    with open(path, 'rb') as f:
        rec_len = struct.unpack('i', f.read(4))[0]
        payload = f.read(rec_len)
        f.read(4)
    return np.frombuffer(payload, dtype=np.float32).copy()


def read_solver_data(solver_path: Path):
    """读取 solver_data.bin → nspec, nglob, x, y, z, ibool"""
    with open(solver_path, 'rb') as f:
        nspec = struct.unpack('i', _read_fortran_record(f))[0]
        nglob = struct.unpack('i', _read_fortran_record(f))[0]
        x_buf = _read_fortran_record(f)
        y_buf = _read_fortran_record(f)
        z_buf = _read_fortran_record(f)
        nbytes = len(x_buf)
        dtype_xyz = np.float32 if nbytes == nglob * 4 else np.float64
        x = np.frombuffer(x_buf, dtype=dtype_xyz)
        y = np.frombuffer(y_buf, dtype=dtype_xyz)
        z = np.frombuffer(z_buf, dtype=dtype_xyz)
        ibool_buf = _read_fortran_record(f)
        ibool = np.frombuffer(ibool_buf, dtype=np.int32)
        ibool = ibool.reshape((NGLLX, NGLLY, NGLLZ, nspec), order='F')
    return nspec, nglob, x, y, z, ibool


def xyz_to_geo(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-6)
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    depth = R_EARTH_KM * (1.0 - r)
    return lat, lon, depth


# ─── 数据收集 ────────────────────────────────────────────────
def find_common_procs(dir_a: Path, dir_b: Path) -> List[int]:
    """找到两个目录共有的 proc 编号"""
    procs_a = {int(p.name[4:10]) for p in dir_a.glob('proc*_reg1_vpv.bin')}
    procs_b = {int(p.name[4:10]) for p in dir_b.glob('proc*_reg1_vpv.bin')}
    common = sorted(procs_a & procs_b)
    return common


def find_common_params(dir_a: Path, dir_b: Path) -> List[str]:
    """找到两个目录共有的参数（排除 vs）"""
    pa = {p.name.split('_reg1_')[1].replace('.bin', '')
          for p in dir_a.glob('proc000000_reg1_*.bin')}
    pb = {p.name.split('_reg1_')[1].replace('.bin', '')
          for p in dir_b.glob('proc000000_reg1_*.bin')}
    common = sorted(pa & pb)
    return [p for p in common if p != 'vs']


def collect_data(dir_a: Path, dir_b: Path, params: List[str],
                 mesh_dir: Optional[Path] = None):
    """
    逐 proc 收集所有 GLL 点的参数值。

    返回:
        vals_a, vals_b: dict[param] → 1D array (全部 GLL 点拼接)
        coords: dict with lat/lon/depth (仅当 mesh_dir 可用时)
        proc_ids: 1D int array
    """
    procs = find_common_procs(dir_a, dir_b)
    print(f"  共有 {len(procs)} 个 proc")

    all_a = {p: [] for p in params}
    all_b = {p: [] for p in params}
    all_proc = []
    all_lat, all_lon, all_depth = [], [], []
    has_coords = False

    for idx, pid in enumerate(procs):
        for param in params:
            fa = dir_a / f"proc{pid:06d}_reg1_{param}.bin"
            fb = dir_b / f"proc{pid:06d}_reg1_{param}.bin"
            all_a[param].append(read_gll_flat(fa))
            all_b[param].append(read_gll_flat(fb))

        n_pts = len(all_a[params[0]][-1])
        all_proc.append(np.full(n_pts, pid, dtype=np.int32))

        if mesh_dir is not None:
            solver = mesh_dir / f"proc{pid:06d}_reg1_solver_data.bin"
            if solver.exists():
                nspec, nglob, x, y, z, ibool = read_solver_data(solver)
                lat, lon, depth = xyz_to_geo(x, y, z)
                # 展开 ibool 映射到 GLL 元素点
                ibool_flat = ibool.ravel(order='F') - 1
                all_lat.append(lat[ibool_flat])
                all_lon.append(lon[ibool_flat])
                all_depth.append(depth[ibool_flat])
                has_coords = True

        if (idx + 1) % 24 == 0 or idx == len(procs) - 1:
            print(f"    {idx + 1}/{len(procs)} procs", flush=True)

    vals_a = {p: np.concatenate(all_a[p]) for p in params}
    vals_b = {p: np.concatenate(all_b[p]) for p in params}
    proc_ids = np.concatenate(all_proc)

    coords = None
    if has_coords and all_lat:
        coords = {
            'lat': np.concatenate(all_lat),
            'lon': np.concatenate(all_lon),
            'depth': np.concatenate(all_depth),
        }

    return vals_a, vals_b, proc_ids, coords


# ─── 统计 ────────────────────────────────────────────────────
def compute_and_print_stats(vals_a, vals_b, params, coords=None):
    """计算并打印统计摘要"""
    print("\n" + "=" * 80)
    print(f"  GLL 逐点对比报告")
    print(f"  A: {LABEL_A}")
    print(f"  B: {LABEL_B}")
    print(f"  总 GLL 点数: {len(vals_a[params[0]]):,}")
    print("=" * 80)

    stats = {}
    for param in params:
        a = vals_a[param]
        b = vals_b[param]
        diff = b - a
        abs_diff = np.abs(diff)
        rel_pct = np.abs(diff) / np.maximum(np.abs(a), 1e-10) * 100

        # 完全相同的点
        n_exact = int(np.sum(diff == 0))
        pct_exact = n_exact / len(a) * 100

        unit = PARAM_UNITS.get(param, '')

        s = {
            'mean_a': np.mean(a), 'mean_b': np.mean(b),
            'mean_diff': np.mean(diff), 'std_diff': np.std(diff),
            'mean_abs': np.mean(abs_diff), 'median_abs': np.median(abs_diff),
            'p95_abs': np.percentile(abs_diff, 95),
            'p99_abs': np.percentile(abs_diff, 99),
            'max_abs': np.max(abs_diff),
            'mean_rel': np.mean(rel_pct), 'median_rel': np.median(rel_pct),
            'p95_rel': np.percentile(rel_pct, 95),
            'rmse': np.sqrt(np.mean(diff**2)),
            'corr': np.corrcoef(a, b)[0, 1],
            'n_exact': n_exact, 'pct_exact': pct_exact,
        }
        stats[param] = s

        print(f"\n{'─' * 60}")
        print(f"  {param.upper():<5} {'(' + unit + ')' if unit else '':<10}  |  {len(a):,} 点")
        print(f"{'─' * 60}")
        print(f"  均值   A={s['mean_a']:.4f}  B={s['mean_b']:.4f}")
        print(f"  完全相同: {n_exact:,} / {len(a):,} ({pct_exact:.1f}%)")
        print(f"  差异 (B−A):  mean={s['mean_diff']:+.6f}  std={s['std_diff']:.6f}")
        print(f"  |B−A|:       mean={s['mean_abs']:.6f}  median={s['median_abs']:.6f}")
        print(f"               P95={s['p95_abs']:.6f}  P99={s['p99_abs']:.6f}  max={s['max_abs']:.6f}")
        print(f"  相对误差(%): mean={s['mean_rel']:.3f}  median={s['median_rel']:.3f}  P95={s['p95_rel']:.3f}")
        print(f"  RMSE={s['rmse']:.6f}  R={s['corr']:.8f}")

        # 逐深度段（如有坐标）
        if coords is not None:
            depth = coords['depth']
            bins = [(0, 35), (35, 100), (100, 220), (220, 410), (410, 660), (660, 1000), (1000, 3000)]
            print(f"\n  {'深度段':<14} {'点数':>10} {'mean|B−A|':>12} {'rel%':>8}")
            for d0, d1 in bins:
                m = (depth >= d0) & (depth < d1)
                if m.sum() > 0:
                    dd = abs_diff[m]
                    rr = rel_pct[m]
                    print(f"  {d0:>4}-{d1:<5} km {m.sum():>10,} {dd.mean():>12.6f} {rr.mean():>7.3f}%")

    print("\n" + "=" * 80)
    return stats


# ─── 可视化 ──────────────────────────────────────────────────
def plot_all(vals_a, vals_b, params, output_dir, coords=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 差异直方图 ---
    n = len(params)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).ravel()

    for i, param in enumerate(params):
        ax = axes[i]
        diff = vals_b[param] - vals_a[param]
        p01, p99 = np.percentile(diff, [0.5, 99.5])
        clip = diff[(diff >= p01) & (diff <= p99)]
        ax.hist(clip, bins=200, color='steelblue', alpha=0.85, edgecolor='none')
        ax.axvline(0, color='red', lw=1, ls='--')
        unit = PARAM_UNITS.get(param, '')
        ax.set_title(f'{param.upper()} (B − A)', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'Diff{" (" + unit + ")" if unit else ""}', fontsize=9)
        ax.text(0.97, 0.95, f'μ={diff.mean():.5f}\nσ={diff.std():.5f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=7,
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f'GLL Difference Histograms', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, output_dir / 'gll_diff_histograms')

    # --- 2. A vs B 散点图 ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).ravel()
    from matplotlib.colors import LogNorm

    for i, param in enumerate(params):
        ax = axes[i]
        a, b = vals_a[param], vals_b[param]
        mask = (a > 0)
        a_m, b_m = a[mask], b[mask]
        if len(a_m) > 500000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(a_m), 500000, replace=False)
            a_m, b_m = a_m[idx], b_m[idx]
        ax.hist2d(a_m, b_m, bins=300, norm=LogNorm(), cmap='hot_r')
        lim = [min(a_m.min(), b_m.min()), max(a_m.max(), b_m.max())]
        ax.plot(lim, lim, 'b-', lw=0.8, alpha=0.6)
        ax.set_xlabel(LABEL_A.split('(')[0].strip(), fontsize=8)
        ax.set_ylabel(LABEL_B.split('(')[0].strip(), fontsize=8)
        ax.set_title(f'{param.upper()}', fontsize=11, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        corr = np.corrcoef(a_m, b_m)[0, 1]
        ax.text(0.03, 0.97, f'R={corr:.6f}', transform=ax.transAxes,
                ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('GLL Point Scatter', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, output_dir / 'gll_scatter')

    # --- 3. 相对误差 CDF ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for param in params:
        a = vals_a[param]
        valid = np.abs(a) > 1e-6
        rel = np.abs((vals_b[param][valid] - a[valid]) / a[valid]) * 100
        sorted_r = np.sort(rel)
        cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        if len(sorted_r) > 10000:
            idx = np.linspace(0, len(sorted_r) - 1, 10000, dtype=int)
            sorted_r, cdf = sorted_r[idx], cdf[idx]
        ax.plot(sorted_r, cdf, lw=1.5, label=param.upper())
    ax.set_xlim(0, min(10, ax.get_xlim()[1]))
    ax.set_xlabel('Relative Error (%)', fontsize=11)
    ax.set_ylabel('CDF', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    for pct in [0.9, 0.95, 0.99]:
        ax.axhline(pct, color='gray', lw=0.5, ls=':')
    ax.set_title('Relative Error CDF', fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, output_dir / 'gll_rel_error_cdf')

    # --- 4. 逐深度差异剖面（需坐标）---
    if coords is not None:
        depth = coords['depth']
        vel_params = [p for p in params if p in ('vpv', 'vph', 'vsv', 'vsh', 'rho', 'eta')]

        # 4a: A/B 均值对比
        depth_bins = np.arange(0, 1001, 10)
        dc = (depth_bins[:-1] + depth_bins[1:]) / 2
        fig, axes = plt.subplots(1, len(vel_params), figsize=(3.5 * len(vel_params), 8), sharey=True)
        if len(vel_params) == 1:
            axes = [axes]
        for i, param in enumerate(vel_params):
            ax = axes[i]
            a, b = vals_a[param], vals_b[param]
            ma, mb = [], []
            for j in range(len(depth_bins) - 1):
                m = (depth >= depth_bins[j]) & (depth < depth_bins[j + 1])
                ma.append(np.mean(a[m]) if m.sum() > 10 else np.nan)
                mb.append(np.mean(b[m]) if m.sum() > 10 else np.nan)
            ax.plot(ma, dc, 'b-', lw=1.2, label='Author')
            ax.plot(mb, dc, 'r--', lw=1.2, label='Ours')
            ax.invert_yaxis()
            unit = PARAM_UNITS.get(param, '')
            ax.set_xlabel(f'{param.upper()}{" (" + unit + ")" if unit else ""}')
            if i == 0:
                ax.set_ylabel('Depth (km)')
            ax.legend(fontsize=7, loc='lower left')
            for d in [35, 220, 410, 660]:
                ax.axhline(d, color='gray', lw=0.5, ls=':')
        fig.suptitle('Depth Profiles: Author vs Ours', fontsize=13, fontweight='bold')
        plt.tight_layout()
        _save(fig, output_dir / 'gll_depth_profiles')

        # 4b: 差异 ±σ
        fig, axes = plt.subplots(1, len(vel_params), figsize=(3.5 * len(vel_params), 8), sharey=True)
        if len(vel_params) == 1:
            axes = [axes]
        for i, param in enumerate(vel_params):
            ax = axes[i]
            diff = vals_b[param] - vals_a[param]
            md, sd = [], []
            for j in range(len(depth_bins) - 1):
                m = (depth >= depth_bins[j]) & (depth < depth_bins[j + 1])
                if m.sum() > 10:
                    md.append(np.mean(diff[m]))
                    sd.append(np.std(diff[m]))
                else:
                    md.append(np.nan); sd.append(np.nan)
            md, sd = np.array(md), np.array(sd)
            ax.fill_betweenx(dc, md - 2 * sd, md + 2 * sd, alpha=0.15, color='steelblue', label='±2σ')
            ax.fill_betweenx(dc, md - sd, md + sd, alpha=0.3, color='steelblue', label='±1σ')
            ax.plot(md, dc, 'b-', lw=1.2, label='mean(B−A)')
            ax.axvline(0, color='red', lw=0.8, ls='--')
            ax.invert_yaxis()
            unit = PARAM_UNITS.get(param, '')
            ax.set_xlabel(f'Δ{param.upper()}{" (" + unit + ")" if unit else ""}')
            if i == 0:
                ax.set_ylabel('Depth (km)')
            ax.legend(fontsize=6, loc='lower left')
            for d in [35, 220, 410, 660]:
                ax.axhline(d, color='gray', lw=0.5, ls=':')
        fig.suptitle('Depth Difference: Ours − Author', fontsize=13, fontweight='bold')
        plt.tight_layout()
        _save(fig, output_dir / 'gll_depth_diff')

        # --- 5. 地图切片 ---
        for param in [p for p in params if p in ('vsv', 'vpv')]:
            _plot_map_slices(vals_a[param], vals_b[param], coords, param, output_dir)

    print(f"\n  📁 所有图片 → {output_dir}/")


def _plot_map_slices(a, b, coords, param, output_dir):
    import matplotlib.pyplot as plt
    depth = coords['depth']
    lat, lon = coords['lat'], coords['lon']
    diff = b - a

    n_slices = len(DEPTH_SLICES_KM)
    fig, axes = plt.subplots(2, n_slices, figsize=(4 * n_slices, 7))

    for j, d_target in enumerate(DEPTH_SLICES_KM):
        tol = max(5.0, d_target * 0.1)
        m = np.abs(depth - d_target) < tol

        for row in range(2):
            ax = axes[row, j]
            if m.sum() < 50:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                ax.set_title(f'{d_target} km')
                continue

            if row == 0:
                vals = diff[m]
                vmax = np.percentile(np.abs(vals), 98)
                sc = ax.scatter(lon[m], lat[m], c=vals, s=0.2, cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, rasterized=True)
                ax.set_title(f'{d_target} km', fontsize=10, fontweight='bold')
                plt.colorbar(sc, ax=ax, shrink=0.7,
                             label=f'Δ{param} ({PARAM_UNITS.get(param, "")})')
            else:
                rel = diff[m] / np.maximum(np.abs(a[m]), 1e-10) * 100
                vmax_r = np.percentile(np.abs(rel), 98)
                sc = ax.scatter(lon[m], lat[m], c=rel, s=0.2, cmap='RdBu_r',
                                vmin=-vmax_r, vmax=vmax_r, rasterized=True)
                ax.set_xlabel('Lon (°)', fontsize=8)
                plt.colorbar(sc, ax=ax, shrink=0.7, label=f'Δ{param} (%)')

            if j == 0:
                ax.set_ylabel('Lat (°)', fontsize=8)

    axes[0, 0].text(-0.2, 0.5, 'Absolute', transform=axes[0, 0].transAxes,
                    ha='center', va='center', rotation=90, fontsize=10, fontweight='bold')
    axes[1, 0].text(-0.2, 0.5, 'Relative %', transform=axes[1, 0].transAxes,
                    ha='center', va='center', rotation=90, fontsize=10, fontweight='bold')
    fig.suptitle(f'{param.upper()}: Ours − Author', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, output_dir / f'gll_map_{param}')


def _save(fig, stem: Path):
    import matplotlib.pyplot as plt
    fig.savefig(str(stem) + '.jpg', dpi=300, bbox_inches='tight')
    fig.savefig(str(stem) + '.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 {stem.name}.jpg/.pdf")


# ─── 从 npz 重画图 ──────────────────────────────────────────
def plot_from_npz(npz_path: Path, output_dir: Path):
    """从已保存的 npz 文件加载数据并生成可视化"""
    import matplotlib
    matplotlib.use('Agg')

    print(f"📂 加载 {npz_path} ...")
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.keys())
    params = sorted({k[2:] for k in keys if k.startswith('a_')})
    vals_a = {p: d[f'a_{p}'] for p in params}
    vals_b = {p: d[f'b_{p}'] for p in params}
    coords = None
    if 'lat' in keys:
        coords = {k: d[k] for k in ('lat', 'lon', 'depth', 'r')}
    print(f"  参数: {params}")
    print(f"  GLL 点: {len(vals_a[params[0]]):,}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_all(vals_a, vals_b, params, output_dir, coords)
    print(f"\n✅ 图片 → {output_dir}")


# ─── 主函数 ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='GLL 逐点对比')
    parser.add_argument('--mesh-dir', default=None,
                        help='DATABASES_MPI 路径（可选，提供后输出地图切片）')
    parser.add_argument('--from-npz', default=None,
                        help='从已保存的 npz 文件重新生成可视化（跳过数据读取）')
    args = parser.parse_args()

    if args.from_npz:
        plot_from_npz(Path(args.from_npz), OUTPUT_DIR)
        return

    mesh_dir = Path(args.mesh_dir) if args.mesh_dir else None

    print("🎯 GLL 模型逐点对比")
    print("=" * 60)
    print(f"  A (参考): {DIR_A}")
    print(f"  B (对比): {DIR_B}")
    if mesh_dir:
        print(f"  网格坐标: {mesh_dir}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print()

    # 检测共有参数
    params = find_common_params(DIR_A, DIR_B)
    # 对 qmu 也做对比（如果都有）
    if 'qmu' in find_common_params(DIR_A, DIR_B):
        if 'qmu' not in params:
            params.append('qmu')
    print(f"  对比参数: {params}")

    # 收集数据
    print("\n📦 读取全部 GLL 点...")
    vals_a, vals_b, proc_ids, coords = collect_data(DIR_A, DIR_B, params, mesh_dir)
    n_total = len(vals_a[params[0]])
    print(f"\n  总 GLL 点: {n_total:,}  ({len(set(proc_ids))} procs × {NGLL3} × nspec)")

    if coords is not None:
        print(f"  深度: {coords['depth'].min():.1f} ~ {coords['depth'].max():.1f} km")
        print(f"  纬度: {coords['lat'].min():.1f}° ~ {coords['lat'].max():.1f}°")
        print(f"  经度: {coords['lon'].min():.1f}° ~ {coords['lon'].max():.1f}°")

    # 统计
    stats = compute_and_print_stats(vals_a, vals_b, params, coords)

    # 保存报告
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import io
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    compute_and_print_stats(vals_a, vals_b, params, coords)
    sys.stdout = old_stdout
    (OUTPUT_DIR / 'report.txt').write_text(buf.getvalue())
    print(f"\n  📄 报告 → {OUTPUT_DIR / 'report.txt'}")

    # 保存 npz（先存数据，再画图——即使画图失败数据也保留）
    data = {'proc_ids': proc_ids}
    for p in params:
        data[f'a_{p}'] = vals_a[p]
        data[f'b_{p}'] = vals_b[p]
    if coords is not None:
        data.update(coords)
    np.savez_compressed(OUTPUT_DIR / 'gll_compare.npz', **data)
    print(f"  💾 数据 → {OUTPUT_DIR / 'gll_compare.npz'}")

    # 可视化（matplotlib 不可用时跳过，可本地用 npz 重画）
    try:
        import matplotlib
        print("\n🎨 生成可视化...")
        plot_all(vals_a, vals_b, params, OUTPUT_DIR, coords)
    except ImportError:
        print("\n⚠️  matplotlib 未安装，跳过可视化")
        print("    可将 gll_compare.npz 拷回本地后运行:")
        print(f"    python compare_gll.py --from-npz {OUTPUT_DIR / 'gll_compare.npz'}")

    print("\n✅ 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
