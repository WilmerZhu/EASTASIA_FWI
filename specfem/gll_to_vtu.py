"""
EASTASIA-FWI GLL 模型对比 → VTK (ParaView)
============================================

将两套 GLL 二进制文件的差异导出为 VTK 格式，用 ParaView 三维可视化。

采用 VTK legacy binary 格式（与 SPECFEM xcombine_vol_data_vtk 同族），
兼容所有版本 ParaView。

模式:
  低分辨率 (默认): 每元素 1 hex (8 角点)，全部 proc 合并到单文件 (~50-80 MB)
  高分辨率 (--high-res): 每元素 64 hex (全部 GLL 点)，逐 proc 输出

用法 (服务器 conda eastasia_fwi):
    cd specfem
    python gll_to_vtu.py --mesh-dir FWEA23/DATABASES_MPI

    # 仅差异字段，更小:
    python gll_to_vtu.py --mesh-dir FWEA23/DATABASES_MPI --diff-only

    # 指定参数:
    python gll_to_vtu.py --mesh-dir FWEA23/DATABASES_MPI --params vsv,vpv

    # ParaView 打开:
    File → Open → vtu_compare/gll_compare.vtk

作者: EASTASIA-FWI Team
日期: 2026-04
"""

import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')

# ─── 常量 ────────────────────────────────────────────────────
NGLLX = NGLLY = NGLLZ = 5
NGLL3 = NGLLX * NGLLY * NGLLZ  # 125
R_EARTH_KM = 6371.0
VTK_HEXAHEDRON = 12

SCRIPT_DIR = Path(__file__).parent
DIR_A = SCRIPT_DIR / 'model_updated'
DIR_B = SCRIPT_DIR / 'GLL'
OUTPUT_DIR = SCRIPT_DIR / 'vtu_compare'

# VTK hex 8 角点在 (i,j,k) GLL 空间中的坐标
_CORNER_IJK = [
    (0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0),  # bottom
    (0, 0, 4), (4, 0, 4), (4, 4, 4), (0, 4, 4),   # top
]
# 对应 Fortran-order flat index: i + j*5 + k*25
_CORNER_FLAT = [i + j * 5 + k * 25 for i, j, k in _CORNER_IJK]

# 高分辨率: 4×4×4 = 64 sub-hexes 的顶点偏移
_HI_DI = [0, 1, 1, 0, 0, 1, 1, 0]
_HI_DJ = [0, 0, 1, 1, 0, 0, 1, 1]
_HI_DK = [0, 0, 0, 0, 1, 1, 1, 1]


# ─── I/O 工具 ────────────────────────────────────────────────
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
        payload = f.read(rec_len)
        f.read(4)
    return np.frombuffer(payload, dtype=np.float32).copy()


def read_solver_data(solver_path: Path):
    with open(solver_path, 'rb') as f:
        nspec = struct.unpack('i', _read_fortran_record(f))[0]
        nglob = struct.unpack('i', _read_fortran_record(f))[0]
        x_buf = _read_fortran_record(f)
        y_buf = _read_fortran_record(f)
        z_buf = _read_fortran_record(f)
        dtype_xyz = np.float32 if len(x_buf) == nglob * 4 else np.float64
        x = np.frombuffer(x_buf, dtype=dtype_xyz).copy()
        y = np.frombuffer(y_buf, dtype=dtype_xyz).copy()
        z = np.frombuffer(z_buf, dtype=dtype_xyz).copy()
        ibool_buf = _read_fortran_record(f)
        ibool = np.frombuffer(ibool_buf, dtype=np.int32).copy()
        ibool = ibool.reshape((NGLLX, NGLLY, NGLLZ, nspec), order='F')
    return nspec, nglob, x, y, z, ibool


def xyz_to_geo(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.maximum(r, 1e-10)
    lat = np.degrees(np.arcsin(np.clip(z / r_safe, -1, 1)))
    lon = np.degrees(np.arctan2(y, x))
    depth = R_EARTH_KM * (1.0 - r)
    return lat.astype(np.float32), lon.astype(np.float32), depth.astype(np.float32)


# ─── 网格提取 ────────────────────────────────────────────────
def extract_lowres(ibool, nspec, nglob, x, y, z,
                   gll_dict_a, gll_dict_b, params, diff_only):
    """低分辨率: 每元素 1 个 hex (8 角点)。

    Returns:
        points (N,3), conn (M,8), point_data dict
    """
    # 角点全局 ID (1-based)
    corner_glob = np.stack([
        ibool[ci, cj, ck, :] for ci, cj, ck in _CORNER_IJK
    ], axis=1)  # (nspec, 8)

    # 去重 + 重编号
    unique_glob, inverse = np.unique(corner_glob.ravel(), return_inverse=True)
    n_unique = len(unique_glob)
    conn = inverse.reshape(nspec, 8).astype(np.int32)

    # 坐标 (km)
    idx = unique_glob - 1  # 0-based
    points = (np.column_stack([x[idx], y[idx], z[idx]]) * R_EARTH_KM).astype(np.float32)

    # 地理坐标
    lat, lon, depth = xyz_to_geo(x[idx], y[idx], z[idx])

    # GLL 角点值
    elem_off = np.arange(nspec, dtype=np.int64) * NGLL3
    corner_indices = np.stack([elem_off + cf for cf in _CORNER_FLAT], axis=1)  # (nspec, 8)

    point_data: Dict[str, np.ndarray] = {
        'depth_km': depth,
        'latitude': lat,
        'longitude': lon,
    }

    for param in params:
        gll_a = gll_dict_a[param]
        gll_b = gll_dict_b[param]
        val_a_corners = gll_a[corner_indices]  # (nspec, 8)
        val_b_corners = gll_b[corner_indices]

        # 映射到去重后的全局点（共享角点取最后赋值，连续场值相同）
        a_global = np.zeros(n_unique, dtype=np.float32)
        b_global = np.zeros(n_unique, dtype=np.float32)
        a_global[conn.ravel()] = val_a_corners.ravel()
        b_global[conn.ravel()] = val_b_corners.ravel()

        if not diff_only:
            point_data[f'{param}_A'] = a_global
            point_data[f'{param}_B'] = b_global
        point_data[f'{param}_diff'] = b_global - a_global

    return points, conn, point_data


def extract_highres(ibool, nspec, nglob, x, y, z,
                    gll_dict_a, gll_dict_b, params, diff_only):
    """高分辨率: 每元素 64 个 hex (全部 GLL 点)。

    Returns:
        points (N,3), conn (M,8), point_data dict
    """
    # 全部 GLL 点
    ibool_flat = ibool.ravel(order='F') - 1  # 0-based

    # 去重 + 重编号（比 nglob 保险）
    unique_glob, inverse = np.unique(ibool_flat, return_inverse=True)
    n_unique = len(unique_glob)
    ibool_renum = inverse.reshape(NGLLX, NGLLY, NGLLZ, nspec)

    # 坐标
    points = (np.column_stack([x[unique_glob], y[unique_glob], z[unique_glob]])
              * R_EARTH_KM).astype(np.float32)

    # 64 sub-hex 连接表
    blocks = []
    for kk in range(4):
        for jj in range(4):
            for ii in range(4):
                hex_verts = np.stack([
                    ibool_renum[ii + _HI_DI[v], jj + _HI_DJ[v], kk + _HI_DK[v], :]
                    for v in range(8)
                ], axis=1)
                blocks.append(hex_verts)
    conn = np.concatenate(blocks, axis=0).astype(np.int32)

    # 地理坐标
    lat, lon, depth = xyz_to_geo(x[unique_glob], y[unique_glob], z[unique_glob])

    point_data: Dict[str, np.ndarray] = {
        'depth_km': depth,
        'latitude': lat,
        'longitude': lon,
    }

    for param in params:
        gll_a = gll_dict_a[param]
        gll_b = gll_dict_b[param]
        a_global = np.zeros(n_unique, dtype=np.float32)
        b_global = np.zeros(n_unique, dtype=np.float32)
        a_global[inverse] = gll_a
        b_global[inverse] = gll_b

        if not diff_only:
            point_data[f'{param}_A'] = a_global
            point_data[f'{param}_B'] = b_global
        point_data[f'{param}_diff'] = b_global - a_global

    return points, conn, point_data


# ─── VTK 写入 ────────────────────────────────────────────────
def write_vtk_binary(filepath: Path, points: np.ndarray,
                     connectivity: np.ndarray,
                     point_data: Dict[str, np.ndarray]):
    """写入 VTK legacy binary 格式 (.vtk)。

    与 SPECFEM xcombine_vol_data_vtk 输出同族格式。
    数据段用 big-endian（VTK legacy 标准）。

    points:       (N, 3) float32
    connectivity: (M, 8) int32 (0-based)
    point_data:   {name: (N,) float32}
    """
    n_pts = points.shape[0]
    n_cells = connectivity.shape[0]
    n_cell_ints = n_cells * 9  # 每个 hex = 1(count) + 8(indices)

    with open(filepath, 'wb') as f:
        f.write(b'# vtk DataFile Version 3.0\n')
        f.write(b'GLL model comparison - EASTASIA-FWI\n')
        f.write(b'BINARY\n')
        f.write(b'DATASET UNSTRUCTURED_GRID\n')

        # Points
        f.write(f'POINTS {n_pts} float\n'.encode())
        points.astype('>f4').tofile(f)

        # Cells
        f.write(f'\nCELLS {n_cells} {n_cell_ints}\n'.encode())
        cell_buf = np.empty((n_cells, 9), dtype=np.int32)
        cell_buf[:, 0] = 8
        cell_buf[:, 1:] = connectivity
        cell_buf.astype('>i4').tofile(f)

        # Cell types
        f.write(f'\nCELL_TYPES {n_cells}\n'.encode())
        np.full(n_cells, VTK_HEXAHEDRON, dtype='>i4').tofile(f)

        # Point data
        f.write(f'\nPOINT_DATA {n_pts}\n'.encode())
        for name, vals in point_data.items():
            f.write(f'SCALARS {name} float 1\n'.encode())
            f.write(b'LOOKUP_TABLE default\n')
            vals.astype('>f4').tofile(f)


# ─── 主函数 ──────────────────────────────────────────────────
def find_common(dir_a, dir_b):
    procs_a = {int(p.name[4:10]) for p in dir_a.glob('proc*_reg1_vpv.bin')}
    procs_b = {int(p.name[4:10]) for p in dir_b.glob('proc*_reg1_vpv.bin')}
    procs = sorted(procs_a & procs_b)

    pa = {p.name.split('_reg1_')[1].replace('.bin', '')
          for p in dir_a.glob('proc000000_reg1_*.bin')}
    pb = {p.name.split('_reg1_')[1].replace('.bin', '')
          for p in dir_b.glob('proc000000_reg1_*.bin')}
    params = sorted((pa & pb) - {'vs'})
    return procs, params


def main():
    parser = argparse.ArgumentParser(
        description='GLL 模型对比 → VTK (ParaView)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例:\n'
               '  python gll_to_vtu.py --mesh-dir FWEA23/DATABASES_MPI\n'
               '  python gll_to_vtu.py --mesh-dir DATABASES_MPI --diff-only --params vsv,vpv')
    parser.add_argument('--mesh-dir', required=True,
                        help='DATABASES_MPI 路径')
    parser.add_argument('--dir-a', default=None,
                        help=f'目录 A (默认: {DIR_A})')
    parser.add_argument('--dir-b', default=None,
                        help=f'目录 B (默认: {DIR_B})')
    parser.add_argument('--output-dir', default=None,
                        help=f'输出目录 (默认: {OUTPUT_DIR})')
    parser.add_argument('--params', default=None,
                        help='参数列表，逗号分隔 (默认: 全部共有参数)')
    parser.add_argument('--diff-only', action='store_true',
                        help='仅输出 diff 字段（不含 A/B），文件更小')
    parser.add_argument('--high-res', action='store_true',
                        help='高分辨率 (64 hex/元素，逐 proc 输出，文件很大)')
    parser.add_argument('--depth-max', type=float, default=None,
                        help='仅导出浅于此深度的元素 (km)，大幅减小文件')
    args = parser.parse_args()

    mesh_dir = Path(args.mesh_dir)
    dir_a = Path(args.dir_a) if args.dir_a else DIR_A
    dir_b = Path(args.dir_b) if args.dir_b else DIR_B
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = 'high-res' if args.high_res else 'low-res'
    print("🎯 GLL → VTK 对比可视化")
    print("=" * 60)
    print(f"  A (参考): {dir_a}")
    print(f"  B (对比): {dir_b}")
    print(f"  网格:     {mesh_dir}")
    print(f"  输出:     {output_dir}")
    print(f"  分辨率:   {mode} ({'1 hex/元素' if mode == 'low-res' else '64 hex/元素'})")
    if args.diff_only:
        print(f"  模式:     仅差异字段")
    if args.depth_max:
        print(f"  深度限制: < {args.depth_max} km")

    procs, all_params = find_common(dir_a, dir_b)
    if args.params:
        params = [p.strip() for p in args.params.split(',') if p.strip() in all_params]
    else:
        params = all_params

    print(f"  Procs:    {len(procs)}")
    print(f"  参数:     {params}")
    print()

    if not procs or not params:
        print("❌ 未找到共有 proc 或参数")
        return

    t0 = time.time()

    if not args.high_res:
        # ─── 低分辨率: 全部 proc 合并到一个文件 ───
        all_pts, all_conn, all_pd = [], [], {
            'depth_km': [], 'latitude': [], 'longitude': []}
        for p in params:
            if not args.diff_only:
                all_pd[f'{p}_A'] = []
                all_pd[f'{p}_B'] = []
            all_pd[f'{p}_diff'] = []

        running_np = 0

        for idx, pid in enumerate(procs):
            solver = mesh_dir / f"proc{pid:06d}_reg1_solver_data.bin"
            if not solver.exists():
                continue

            nspec, nglob, x, y, z, ibool = read_solver_data(solver)

            # 读取 GLL 参数
            gll_a, gll_b = {}, {}
            for p in params:
                gll_a[p] = read_gll_flat(dir_a / f"proc{pid:06d}_reg1_{p}.bin")
                gll_b[p] = read_gll_flat(dir_b / f"proc{pid:06d}_reg1_{p}.bin")

            pts, conn, pd = extract_lowres(
                ibool, nspec, nglob, x, y, z,
                gll_a, gll_b, params, args.diff_only)

            # 深度筛选（按元素：元素 8 角点最小深度 < depth_max 则保留）
            if args.depth_max is not None:
                corner_depth = pd['depth_km'][conn]  # (nspec, 8)
                keep_elem = corner_depth.min(axis=1) < args.depth_max
                if keep_elem.sum() == 0:
                    continue
                conn = conn[keep_elem]
                used_pts = np.unique(conn.ravel())
                pt_map = np.full(pts.shape[0], -1, dtype=np.int32)
                pt_map[used_pts] = np.arange(len(used_pts), dtype=np.int32)
                conn = pt_map[conn]
                pts = pts[used_pts]
                for k in pd:
                    pd[k] = pd[k][used_pts]

            all_pts.append(pts)
            all_conn.append(conn + running_np)
            for k in pd:
                all_pd[k].append(pd[k])
            running_np += pts.shape[0]

            if (idx + 1) % 24 == 0 or idx == len(procs) - 1:
                print(f"  {idx + 1}/{len(procs)} procs  "
                      f"({time.time() - t0:.0f}s)", flush=True)

        total_pts = np.concatenate(all_pts)
        total_conn = np.concatenate(all_conn)
        total_pd = {k: np.concatenate(all_pd[k]) for k in all_pd}

        vtk_path = output_dir / 'gll_compare.vtk'
        print(f"\n  写入 {vtk_path.name} ...")
        write_vtk_binary(vtk_path, total_pts, total_conn, total_pd)

        mb = vtk_path.stat().st_size / 1024**2
        elapsed = time.time() - t0
        print(f"\n✅ 完成!  ({elapsed:.0f}s)")
        print(f"  📁 {vtk_path}")
        print(f"  📊 {total_pts.shape[0]:,} 点, {total_conn.shape[0]:,} hex")
        print(f"  💾 {mb:.1f} MB")
        print(f"\n  🖥️  ParaView 打开:")
        print(f"     File → Open → {vtk_path}")

    else:
        # ─── 高分辨率: 逐 proc 输出 ───
        vtk_files = []
        for idx, pid in enumerate(procs):
            solver = mesh_dir / f"proc{pid:06d}_reg1_solver_data.bin"
            if not solver.exists():
                continue

            nspec, nglob, x, y, z, ibool = read_solver_data(solver)

            gll_a, gll_b = {}, {}
            for p in params:
                gll_a[p] = read_gll_flat(dir_a / f"proc{pid:06d}_reg1_{p}.bin")
                gll_b[p] = read_gll_flat(dir_b / f"proc{pid:06d}_reg1_{p}.bin")

            pts, conn, pd = extract_highres(
                ibool, nspec, nglob, x, y, z,
                gll_a, gll_b, params, args.diff_only)

            # 深度筛选
            if args.depth_max is not None:
                corner_depth = pd['depth_km'][conn]
                keep_elem = corner_depth.min(axis=1) < args.depth_max
                if keep_elem.sum() == 0:
                    continue
                conn = conn[keep_elem]
                used_pts = np.unique(conn.ravel())
                pt_map = np.full(pts.shape[0], -1, dtype=np.int32)
                pt_map[used_pts] = np.arange(len(used_pts), dtype=np.int32)
                conn = pt_map[conn]
                pts = pts[used_pts]
                for k in pd:
                    pd[k] = pd[k][used_pts]

            vtk_name = f'proc{pid:06d}.vtk'
            write_vtk_binary(output_dir / vtk_name, pts, conn, pd)
            vtk_files.append(vtk_name)

            if (idx + 1) % 12 == 0 or idx == len(procs) - 1:
                print(f"  {idx + 1}/{len(procs)} procs  "
                      f"({time.time() - t0:.0f}s)", flush=True)

        elapsed = time.time() - t0
        total_mb = sum((output_dir / f).stat().st_size for f in vtk_files) / 1024**2
        print(f"\n✅ 完成!  ({elapsed:.0f}s)")
        print(f"  📁 {len(vtk_files)} 个文件 → {output_dir}/")
        print(f"  💾 总大小: {total_mb:.0f} MB")
        print(f"\n  🖥️  ParaView 打开:")
        print(f"     File → Open → 选中全部 proc*.vtk → OK")

    print(f"\n  推荐操作:")
    print(f"     1. 选 'vsv_diff' / 'vpv_diff', Representation → Surface")
    print(f"     2. Filters → Threshold → depth_km [0, 200] 看浅部")
    print(f"     3. Filters → Slice 切截面")
    print(f"     4. Color Map → Cool to Warm (diverging)")
    print("=" * 60)


if __name__ == "__main__":
    main()
