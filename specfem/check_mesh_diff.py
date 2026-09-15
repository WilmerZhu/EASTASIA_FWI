#!/usr/bin/env python3
"""
EASTASIA-FWI solver_data.bin 网格坐标一致性核对
================================================

功能描述:
- 逐条比对两套 DATABASES_MPI 的坐标段（nspec / nglob / x / y / z / ibool）
- 支持单个 proc 详查与全部 proc 批量核对两种模式
- 以退出码表明一致与否，可直接嵌入 Slurm 作业作为放行关卡

科学原理:
- solver_data.bin 的前 6 条记录完全决定 GLL 点的物理位置，模型插值即以此为基准。
  两次 mesher 若在这段内容上不一致，同一份速度模型会落在不同的空间位置上，
  程序不会报错，但跨模型波形对比会失去可比性。
- 判据分两级：ibool 属拓扑，必须逐位相同，任何差异都意味着全局编号变了；
  x/y/z 允许一个相对容差，用于容纳异构节点上向量化求和顺序不同带来的末位差异。
  默认容差 0 表示要求逐位相同。

作者: EASTASIA-FWI Team
日期: 2026-08-03
版本: v2.0
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

NGLLX = NGLLY = NGLLZ = 5


# ── 二进制读取 ──────────────────────────────────────────────────────────────


def _read_record(f) -> bytes:
    """读一条 Fortran 无格式记录（首尾各有一个 4 字节长度标记）。"""
    head = f.read(4)
    if len(head) < 4:
        raise EOFError("文件提前结束")
    nbytes = struct.unpack('i', head)[0]
    body = f.read(nbytes)
    tail = struct.unpack('i', f.read(4))[0]
    if tail != nbytes:
        raise ValueError(f"记录首尾标记不一致: {nbytes} vs {tail}")
    return body


def read_coord_records(path: Path) -> Dict[str, object]:
    """
    读取 solver_data.bin 的坐标段。

    Args:
        path: proc***_reg1_solver_data.bin 路径

    Returns:
        dict，含 nspec / nglob / x / y / z / ibool / prefix_bytes / dtype
    """
    with open(path, 'rb') as f:
        nspec = struct.unpack('i', _read_record(f))[0]
        nglob = struct.unpack('i', _read_record(f))[0]
        xb, yb, zb = _read_record(f), _read_record(f), _read_record(f)
        dt = np.float32 if len(xb) == nglob * 4 else np.float64
        ib = _read_record(f)
        prefix_bytes = f.tell()

    return {
        'nspec': nspec,
        'nglob': nglob,
        'x': np.frombuffer(xb, dtype=dt),
        'y': np.frombuffer(yb, dtype=dt),
        'z': np.frombuffer(zb, dtype=dt),
        'ibool': np.frombuffer(ib, dtype=np.int32),
        'prefix_bytes': prefix_bytes,
        'dtype': np.dtype(dt).name,
    }


# ── 比对 ────────────────────────────────────────────────────────────────────


class ProcResult:
    """单个 proc 的比对结果。"""

    def __init__(self, pid: str):
        self.pid = pid
        self.ok = True
        self.exact = True
        self.max_rel = 0.0
        self.max_abs = 0.0
        self.n_coord_diff = 0
        self.ibool_diff = 0
        self.notes: List[str] = []

    def fail(self, msg: str) -> None:
        self.ok = False
        self.exact = False
        self.notes.append(msg)


def compare_proc(path_a: Path, path_b: Path, tol: float) -> ProcResult:
    """
    比对一个 proc 的坐标段。

    Args:
        path_a: 参考侧 solver_data.bin
        path_b: 待检侧 solver_data.bin
        tol:    x/y/z 允许的相对误差；0 表示要求逐位相同

    Returns:
        ProcResult
    """
    pid = path_a.name[4:10]
    res = ProcResult(pid)

    ra, rb = read_coord_records(path_a), read_coord_records(path_b)
    res.prefix_a = ra['prefix_bytes']
    res.prefix_b = rb['prefix_bytes']
    res.nglob_a, res.nglob_b = ra['nglob'], rb['nglob']
    res.nspec_a, res.nspec_b = ra['nspec'], rb['nspec']

    if ra['nspec'] != rb['nspec']:
        res.fail(f"nspec 不同 {ra['nspec']} vs {rb['nspec']}")
        return res
    if ra['nglob'] != rb['nglob']:
        res.fail(f"nglob 不同 {ra['nglob']} vs {rb['nglob']}")
        return res

    # ibool 属拓扑，不容任何差异
    ia, ib = ra['ibool'], rb['ibool']
    if not np.array_equal(ia, ib):
        res.ibool_diff = int((ia != ib).sum())
        res.fail(f"ibool 有 {res.ibool_diff}/{ia.size} 个值不同（全局编号变了）")

    for name in ('x', 'y', 'z'):
        a = ra[name].astype(np.float64)
        b = rb[name].astype(np.float64)
        if np.array_equal(a, b):
            continue
        res.exact = False
        d = np.abs(a - b)
        bad = d > 0
        res.n_coord_diff += int(bad.sum())
        rel = d[bad] / np.maximum(np.abs(a[bad]), 1e-30)
        res.max_abs = max(res.max_abs, float(d.max()))
        res.max_rel = max(res.max_rel, float(rel.max()))
        if res.max_rel > tol:
            res.fail(f"{name} 相对差 {res.max_rel:.3e} 超过容差 {tol:.1e}")

    return res


def collect_pids(dir_a: Path, dir_b: Path, nproc: Optional[int],
                 explicit: List[str]) -> List[str]:
    """确定要比对的 proc 编号列表。"""
    if explicit:
        return explicit
    if nproc:
        return [f"{i:06d}" for i in range(nproc)]
    pids = sorted(p.name[4:10] for p in dir_a.glob('proc*_reg1_solver_data.bin'))
    return pids


# ── 主流程 ──────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description='比对两套 DATABASES_MPI 的网格坐标段',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 批量核对全部 144 个 proc，允许 1e-6 相对误差
  python check_mesh_diff.py mesh0/DATABASES_MPI runs/EARA2024_hy/DATABASES_MPI \\
      --nproc 144 --tol 1e-6

  # 详查几个 proc，要求逐位相同
  python check_mesh_diff.py runs/BASE/DATABASES_MPI runs/SinoScope_hy/DATABASES_MPI \\
      --pids 000074 000075 000000 --verbose

退出码: 0 = 全部在容差内, 1 = 存在超差, 2 = 读取失败
""")
    p.add_argument('dir_a', help='参考侧 DATABASES_MPI 目录')
    p.add_argument('dir_b', help='待检侧 DATABASES_MPI 目录')
    p.add_argument('--pids', nargs='+', default=[], help='指定 proc 编号（如 000074）')
    p.add_argument('--nproc', type=int, default=0, help='比对 0..N-1 全部 proc')
    p.add_argument('--tol', type=float, default=0.0,
                   help='x/y/z 允许的相对误差，默认 0（逐位相同）')
    p.add_argument('--verbose', action='store_true', help='逐 proc 打印明细')
    return p.parse_args(argv)


def main() -> None:
    """主函数"""
    args = parse_args()
    dir_a, dir_b = Path(args.dir_a), Path(args.dir_b)

    print("🎯 EASTASIA-FWI 网格坐标一致性核对")
    print("=" * 70)
    print(f"参考: {dir_a}")
    print(f"待检: {dir_b}")
    print(f"容差: ibool 逐位相同, x/y/z 相对误差 <= {args.tol:.1e}")
    print("=" * 70)

    pids = collect_pids(dir_a, dir_b, args.nproc, args.pids)
    if not pids:
        print("❌ 没有找到可比对的 proc")
        sys.exit(2)

    results: List[ProcResult] = []
    missing = 0
    for pid in pids:
        fn = f"proc{pid}_reg1_solver_data.bin"
        pa, pb = dir_a / fn, dir_b / fn
        if not pa.exists() or not pb.exists():
            missing += 1
            if missing <= 5:
                print(f"⚠️  缺失: {pa if not pa.exists() else pb}")
            continue
        try:
            r = compare_proc(pa, pb, args.tol)
        except Exception as exc:
            print(f"❌ proc{pid} 读取失败: {exc}")
            sys.exit(2)
        results.append(r)
        if args.verbose or not r.ok:
            flag = "✅" if r.ok else "❌"
            print(f"{flag} proc{pid}  nspec={r.nspec_a} nglob={r.nglob_a} "
                  f"坐标段={r.prefix_a} 字节")
            if r.prefix_a != r.prefix_b:
                print(f"     坐标段长度不同: {r.prefix_a} vs {r.prefix_b}")
            if not r.exact:
                print(f"     坐标 {r.n_coord_diff} 个点不同  "
                      f"绝对差 max={r.max_abs:.3e}  相对差 max={r.max_rel:.3e}")
            for n in r.notes:
                print(f"     {n}")

    if not results:
        print("❌ 没有成功比对任何 proc")
        sys.exit(2)

    n_exact = sum(1 for r in results if r.exact)
    n_ok = sum(1 for r in results if r.ok)
    n_bad = len(results) - n_ok
    max_rel = max((r.max_rel for r in results), default=0.0)
    max_abs = max((r.max_abs for r in results), default=0.0)
    n_ibool = sum(1 for r in results if r.ibool_diff)

    print("=" * 70)
    print(f"比对 {len(results)} 个 proc" + (f"（{missing} 个缺失）" if missing else ""))
    print(f"  逐位相同:        {n_exact}")
    print(f"  容差内:          {n_ok}")
    print(f"  超差:            {n_bad}")
    print(f"  ibool 有差异:    {n_ibool}")
    print(f"  坐标相对差 max:  {max_rel:.3e}   绝对差 max（无量纲半径）: {max_abs:.3e}")
    if max_abs > 0:
        print(f"  折算距离约:      {max_abs * 6371.0 * 1000:.3f} 米")

    if n_bad == 0:
        print("✅ 网格坐标一致，模型插值基准有效")
        print("=" * 70)
        sys.exit(0)

    print("❌ 网格坐标不一致，GLL 值会落在错误的位置上，结果不可用")
    print("=" * 70)
    sys.exit(1)


if __name__ == "__main__":
    main()
