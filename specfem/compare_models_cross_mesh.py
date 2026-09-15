"""
跨网格模型参数对比：原作者 GLL (NSPEC=7236) vs 新 EMC (NSPEC=8028)
====================================================================

两套 mesh 的 NSPEC 不同，不能逐点对比。本脚本：
1. 分别从 solver_data.bin 读取 GLL 坐标
2. 从独立 .bin 文件读取速度参数
3. 在物理空间 (lat, lon, depth) 做统计对比
4. 提取台站位置下方的 1D 剖面对比

用法:
    conda activate eastasia_fwi
    cd specfem
    python compare_models_cross_mesh.py
    python compare_models_cross_mesh.py --depth-map    # 额外输出深度切片地图
    python compare_models_cross_mesh.py --profiles      # 台站下方 1D 剖面

作者: EASTASIA-FWI Team
日期: 2026-04
"""

import sys
import struct
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── 常量 ────────────────────────────────────────────────────
NGLLX, NGLLY, NGLLZ = 5, 5, 5
NGLL3 = NGLLX * NGLLY * NGLLZ  # 125
R_EARTH_KM = 6371.0
R_EARTH_M = 6371000.0  # SPECFEM non-dimensionalization

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / 'figures' / 'model_compare'

# 原作者参考文件
REF_GLL_DIR = SCRIPT_DIR / 'model_updated'

# 新 EMC 文件
EMC_DB_DIR = SCRIPT_DIR / '..' / 'specfem3d_globe_new' / 'FWEA23' / 'DATABASES_MPI'

# 旧方案 DATABASES_MPI (含 solver_data.bin 用于坐标)
OLD_DB_DIR = SCRIPT_DIR / 'DATABASES_MPI'

PARAMS = ['vpv', 'vph', 'vsv', 'vsh', 'eta', 'rho']
PARAM_UNITS = {
    'vpv': 'km/s', 'vph': 'km/s', 'vsv': 'km/s', 'vsh': 'km/s',
    'eta': '', 'rho': 'g/cm³',
}

DEPTH_BINS = [
    (0, 35, '地壳 (0-35 km)'),
    (35, 100, '上地幔浅部 (35-100 km)'),
    (100, 300, '上地幔 (100-300 km)'),
    (300, 660, '过渡带 (300-660 km)'),
    (660, 1000, '下地幔浅部 (660-1000 km)'),
]


# ─── 读取工具 ────────────────────────────────────────────────
def _read_fortran_record(f) -> bytes:
    rec_len = struct.unpack('i', f.read(4))[0]
    data = f.read(rec_len)
    tail = struct.unpack('i', f.read(4))[0]
    if tail != rec_len:
        raise ValueError(f"Fortran record mismatch: {rec_len} vs {tail}")
    return data


def read_gll_flat(path: Path) -> np.ndarray:
    with open(path, 'rb') as f:
        rec_len = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(rec_len), dtype=np.float32).copy()
        f.read(4)
    return data


def read_solver_data_coords(solver_path: Path):
    """读取 solver_data.bin → nspec, nglob, x, y, z, ibool"""
    with open(solver_path, 'rb') as f:
        nspec = struct.unpack('i', _read_fortran_record(f))[0]
        nglob = struct.unpack('i', _read_fortran_record(f))[0]
        x_buf = _read_fortran_record(f)
        y_buf = _read_fortran_record(f)
        z_buf = _read_fortran_record(f)
        nbytes = len(x_buf)
        dtype_xyz = np.float32 if nbytes == nglob * 4 else np.float64
        x = np.frombuffer(x_buf, dtype=dtype_xyz).copy()
        y = np.frombuffer(y_buf, dtype=dtype_xyz).copy()
        z = np.frombuffer(z_buf, dtype=dtype_xyz).copy()
        ibool_buf = _read_fortran_record(f)
        ibool = np.frombuffer(ibool_buf, dtype=np.int32).copy()
        ibool = ibool.reshape((NGLLX, NGLLY, NGLLZ, nspec), order='F')
    return nspec, nglob, x, y, z, ibool


def xyz_to_geo(x, y, z):
    """SPECFEM 非量纲化 xyz → 地理坐标 (lat, lon, depth_km)"""
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-10)
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    depth = R_EARTH_KM * (1.0 - r)
    return lat, lon, depth


# ─── 数据加载 ────────────────────────────────────────────────
def load_model_data(gll_dir: Path, db_dir: Path, label: str,
                    max_procs: int = 144, subsample: int = 1):
    """
    加载模型数据 — GLL 参数 + 坐标。

    gll_dir: 参数 .bin 文件目录 (vpv, vsv, ...)
    db_dir:  solver_data.bin 目录 (含坐标)
    label:   标签名
    subsample: 每 N 个 proc 取 1 个 (减少内存)
    """
    procs = sorted({int(p.name[4:10]) for p in gll_dir.glob('proc*_reg1_vpv.bin')})
    if max_procs:
        procs = procs[:max_procs]
    if subsample > 1:
        procs = procs[::subsample]

    print(f"\n加载 {label}: {len(procs)} procs from {gll_dir.name}")

    all_vals = {p: [] for p in PARAMS}
    all_lat, all_lon, all_depth = [], [], []

    for idx, pid in enumerate(procs):
        # 读参数
        for param in PARAMS:
            f = gll_dir / f"proc{pid:06d}_reg1_{param}.bin"
            if f.exists():
                all_vals[param].append(read_gll_flat(f))
            else:
                break

        # 读坐标
        solver = db_dir / f"proc{pid:06d}_reg1_solver_data.bin"
        if solver.exists():
            nspec, nglob, x, y, z, ibool = read_solver_data_coords(solver)
            lat, lon, depth = xyz_to_geo(x, y, z)
            ibool_flat = ibool.ravel(order='F') - 1
            all_lat.append(lat[ibool_flat])
            all_lon.append(lon[ibool_flat])
            all_depth.append(depth[ibool_flat])

        if (idx + 1) % 24 == 0 or idx == len(procs) - 1:
            print(f"  {idx+1}/{len(procs)} procs", flush=True)

    vals = {p: np.concatenate(all_vals[p]) for p in PARAMS if all_vals[p]}
    coords = {
        'lat': np.concatenate(all_lat),
        'lon': np.concatenate(all_lon),
        'depth': np.concatenate(all_depth),
    }
    print(f"  总 GLL 点: {len(coords['lat']):,}")
    return vals, coords


# ─── 分析 ────────────────────────────────────────────────────
def depth_binned_stats(vals, coords, label):
    """按深度分层统计"""
    depth = coords['depth']
    print(f"\n{'='*70}")
    print(f"  {label} — 按深度分层统计")
    print(f"{'='*70}")
    print(f"{'深度层':>25s} | {'N':>10s} | {'vpv':>12s} | {'vsv':>12s} | {'rho':>12s}")
    print('-' * 80)

    for d_min, d_max, name in DEPTH_BINS:
        mask = (depth >= d_min) & (depth < d_max)
        n = mask.sum()
        if n == 0:
            continue
        vpv_str = f"{vals['vpv'][mask].mean():.4f}±{vals['vpv'][mask].std():.4f}"
        vsv_str = f"{vals['vsv'][mask].mean():.4f}±{vals['vsv'][mask].std():.4f}"
        rho_str = f"{vals['rho'][mask].mean():.4f}±{vals['rho'][mask].std():.4f}"
        print(f"{name:>25s} | {n:>10,} | {vpv_str:>12s} | {vsv_str:>12s} | {rho_str:>12s}")


def cross_model_comparison(ref_vals, ref_coords, emc_vals, emc_coords):
    """
    跨网格统计对比 — 在相同深度层对比分布特征。
    由于 NSPEC 不同，只能对比统计量（均值、标准差、分位数）。
    """
    print(f"\n{'='*80}")
    print("  跨网格对比: 原作者 GLL vs 新 EMC — 按深度分层")
    print(f"{'='*80}")

    for param in ['vpv', 'vsv', 'vph', 'vsh', 'rho', 'eta']:
        if param not in ref_vals or param not in emc_vals:
            continue
        unit = PARAM_UNITS.get(param, '')
        print(f"\n--- {param} ({unit}) ---")
        print(f"{'深度层':>25s} | {'Ref 均值':>10s} {'Ref σ':>8s} | {'EMC 均值':>10s} {'EMC σ':>8s} | {'Δ均值':>8s} {'Δ%':>6s}")
        print('-' * 95)

        for d_min, d_max, name in DEPTH_BINS:
            ref_mask = (ref_coords['depth'] >= d_min) & (ref_coords['depth'] < d_max)
            emc_mask = (emc_coords['depth'] >= d_min) & (emc_coords['depth'] < d_max)

            if ref_mask.sum() == 0 or emc_mask.sum() == 0:
                continue

            ref_mean = ref_vals[param][ref_mask].mean()
            ref_std = ref_vals[param][ref_mask].std()
            emc_mean = emc_vals[param][emc_mask].mean()
            emc_std = emc_vals[param][emc_mask].std()
            d_mean = emc_mean - ref_mean
            d_pct = 100 * d_mean / ref_mean if abs(ref_mean) > 1e-6 else 0

            print(f"{name:>25s} | {ref_mean:>10.4f} {ref_std:>8.4f} | "
                  f"{emc_mean:>10.4f} {emc_std:>8.4f} | {d_mean:>+8.4f} {d_pct:>+6.2f}%")


def make_depth_histograms(ref_vals, ref_coords, emc_vals, emc_coords):
    """每个深度层对比 vpv/vsv 直方图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib 不可用，跳过直方图")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for param in ['vpv', 'vsv']:
        fig, axes = plt.subplots(len(DEPTH_BINS), 1, figsize=(10, 3*len(DEPTH_BINS)))
        for i, (d_min, d_max, name) in enumerate(DEPTH_BINS):
            ax = axes[i]
            ref_mask = (ref_coords['depth'] >= d_min) & (ref_coords['depth'] < d_max)
            emc_mask = (emc_coords['depth'] >= d_min) & (emc_coords['depth'] < d_max)

            if ref_mask.sum() > 0:
                ax.hist(ref_vals[param][ref_mask], bins=100, alpha=0.6,
                        label=f'Chujie (N={ref_mask.sum():,})', color='red', density=True)
            if emc_mask.sum() > 0:
                ax.hist(emc_vals[param][emc_mask], bins=100, alpha=0.6,
                        label=f'EMC (N={emc_mask.sum():,})', color='black', density=True)

            ax.set_title(f'{name}')
            ax.set_xlabel(f'{param} ({PARAM_UNITS[param]})')
            ax.legend(fontsize=8)

        fig.suptitle(f'{param} Distribution Comparison: Chujie vs EMC', fontsize=14)
        plt.tight_layout()
        out_path = OUTPUT_DIR / f'hist_{param}_depth_compare.jpg'
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  保存: {out_path}")
        plt.close(fig)


def make_1d_profiles(ref_vals, ref_coords, emc_vals, emc_coords,
                     profile_locations=None):
    """在指定地理位置提取 1D 剖面对比"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib 不可用，跳过剖面图")
        return

    if profile_locations is None:
        # 默认: 波形图中匹配较差的台站方位 + 模型中心
        profile_locations = [
            (33.6, 131.9, 'Source Region (Kyushu)'),
            (35.0, 106.0, 'Model Center'),
            (30.0, 90.0, 'Tibet'),
            (40.0, 116.0, 'North China'),
            (25.0, 120.0, 'Taiwan'),
            (10.0, 105.0, 'South (near edge)'),
        ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    search_radius = 2.0  # degrees
    depth_bins_fine = np.arange(0, 1001, 10)

    fig, axes = plt.subplots(2, 3, figsize=(18, 14))
    axes = axes.flatten()

    for idx, (lat0, lon0, name) in enumerate(profile_locations):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # 提取该位置附近的 GLL 点
        for vals, coords, color, label in [
            (ref_vals, ref_coords, 'red', 'Chujie'),
            (emc_vals, emc_coords, 'black', 'EMC'),
        ]:
            dist = np.sqrt((coords['lat'] - lat0)**2 + (coords['lon'] - lon0)**2)
            near = dist < search_radius
            if near.sum() < 10:
                continue

            dep = coords['depth'][near]
            vpv = vals['vpv'][near]
            vsv = vals['vsv'][near]

            # 按深度 bin 取中位数
            dep_centers = []
            vpv_med, vsv_med = [], []
            for j in range(len(depth_bins_fine) - 1):
                d_mask = (dep >= depth_bins_fine[j]) & (dep < depth_bins_fine[j+1])
                if d_mask.sum() > 0:
                    dep_centers.append((depth_bins_fine[j] + depth_bins_fine[j+1]) / 2)
                    vpv_med.append(np.median(vpv[d_mask]))
                    vsv_med.append(np.median(vsv[d_mask]))

            if dep_centers:
                ax.plot(vpv_med, dep_centers, color=color, linewidth=1.5,
                        label=f'{label} Vpv', linestyle='-')
                ax.plot(vsv_med, dep_centers, color=color, linewidth=1.5,
                        label=f'{label} Vsv', linestyle='--')

        ax.set_ylim(800, 0)
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Depth (km)')
        ax.set_title(f'{name}\n({lat0:.1f}°N, {lon0:.1f}°E)')
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.axhline(35, color='gray', linestyle=':', alpha=0.5, label='Moho')
        ax.axhline(410, color='blue', linestyle=':', alpha=0.3)
        ax.axhline(660, color='blue', linestyle=':', alpha=0.3)

    fig.suptitle('1D Velocity Profile Comparison: Chujie (GLL) vs EMC (NetCDF)', fontsize=14)
    plt.tight_layout()
    out_path = OUTPUT_DIR / 'profiles_1d_compare.jpg'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"  保存: {out_path}")
    plt.close(fig)


# ─── 主程序 ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='跨网格模型参数对比')
    parser.add_argument('--ref-gll', type=Path, default=REF_GLL_DIR,
                        help='原作者 GLL 目录')
    parser.add_argument('--ref-db', type=Path, default=OLD_DB_DIR,
                        help='原作者 DATABASES_MPI (含 solver_data.bin)')
    parser.add_argument('--emc-db', type=Path, default=EMC_DB_DIR,
                        help='新 EMC DATABASES_MPI (含参数 .bin + solver_data.bin)')
    parser.add_argument('--subsample', type=int, default=1,
                        help='每 N 个 proc 取 1 个 (减少内存)')
    parser.add_argument('--depth-map', action='store_true',
                        help='输出深度切片直方图')
    parser.add_argument('--profiles', action='store_true',
                        help='输出 1D 剖面对比')
    parser.add_argument('--max-procs', type=int, default=144)
    args = parser.parse_args()

    # 验证路径
    for name, path in [('ref-gll', args.ref_gll), ('ref-db', args.ref_db), ('emc-db', args.emc_db)]:
        if not path.exists():
            print(f"❌ {name} 不存在: {path}")
            sys.exit(1)

    # 加载原作者数据
    ref_vals, ref_coords = load_model_data(
        gll_dir=args.ref_gll,
        db_dir=args.ref_db,
        label='原作者 (Chujie)',
        max_procs=args.max_procs,
        subsample=args.subsample,
    )

    # 加载新 EMC 数据
    emc_vals, emc_coords = load_model_data(
        gll_dir=args.emc_db,  # EMC 的 vpv.bin 也在 DATABASES_MPI 中
        db_dir=args.emc_db,
        label='新 EMC',
        max_procs=args.max_procs,
        subsample=args.subsample,
    )

    # 分层统计
    depth_binned_stats(ref_vals, ref_coords, '原作者 (Chujie)')
    depth_binned_stats(emc_vals, emc_coords, '新 EMC')

    # 跨网格对比
    cross_model_comparison(ref_vals, ref_coords, emc_vals, emc_coords)

    # 可选输出
    if args.depth_map:
        print("\n生成深度切片直方图...")
        make_depth_histograms(ref_vals, ref_coords, emc_vals, emc_coords)

    if args.profiles:
        print("\n生成 1D 剖面对比...")
        make_1d_profiles(ref_vals, ref_coords, emc_vals, emc_coords)

    print("\n✅ 完成")


if __name__ == '__main__':
    main()
