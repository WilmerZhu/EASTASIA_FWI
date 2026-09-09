"""
EASTASIA-FWI W-k-means 规模基准
================================

功能描述:
- 实测 W-k-means 拟合耗时随体元数的增长
- 实测轮廓系数在不同子样本规模下的耗时与稳定性
- 实测 Davies-Bouldin / Calinski-Harabasz 在全量点上的耗时

用途:
- 本项目单个深度带的体元数最多约 2.8×10⁶。拟合复杂度 O(n·k·d) 允许全量
  计算，而轮廓系数是 O(n²)，全量对应约 7.8×10¹² 个点对，不可行。本脚本
  给出定量依据，用于确定各环节是否需要抽样、抽样多少。

作者: EASTASIA-FWI Team
日期: 2026-09-07
版本: v1.0
"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / '2_Model_space_analysis'))

import importlib

_clu = importlib.import_module('2_3_Model_clustering')
WKMeans = _clu.WKMeans

# 本项目实际的单带体元数：SinoScope 最小 1.6×10⁴，FWEA23 岩石圈最大 2.8×10⁶
BAND_SIZES = (150_000, 500_000, 1_000_000, 2_000_000, 2_800_000)
SIL_SAMPLE_SIZES = (20_000, 50_000, 100_000, 200_000)
N_ITER_PROBE = 10


def make_clustered(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    构造带簇结构的二维样本

    用高斯混合而非纯噪声，使迭代次数与收敛行为接近真实速度场数据；
    纯噪声上 k-means 的迭代会异常增多，高估耗时。
    """
    centers = rng.normal(scale=3.0, size=(k, 2))
    assign = rng.integers(0, k, n)
    return centers[assign] + rng.normal(scale=1.0, size=(n, 2))


def bench_fit(rng: np.random.Generator) -> None:
    """拟合耗时随体元数的增长"""
    print("=" * 78)
    print(f"W-k-means 拟合耗时（K=10, d=2, 固定 {N_ITER_PROBE} 次迭代）")
    print("=" * 78)
    print(f"{'体元数':>12s} {'总耗时':>9s} {'每迭代':>10s} "
          f"{'外推 100迭代×3init':>20s}")
    for n in BAND_SIZES:
        X = make_clustered(n, 10, rng)
        t0 = time.time()
        m = WKMeans(n_clusters=10, beta=6.5, n_init=1,
                    max_iter=N_ITER_PROBE, tol=0.0, random_state=42).fit(X)
        dt = time.time() - t0
        per = dt / max(m.n_iter_, 1)
        print(f"{n:>12,} {dt:>8.1f}s {per * 1000:>9.0f}ms "
              f"{per * 100 * 3 / 60:>17.1f} 分钟", flush=True)
        del X


def bench_silhouette(rng: np.random.Generator) -> None:
    """轮廓系数：耗时与重复估计的稳定性"""
    print("\n" + "=" * 78)
    print("轮廓系数 O(n²)：子样本规模 vs 耗时与稳定性（全量 2.8×10⁶ 体元）")
    print("=" * 78)
    n = 2_800_000
    X = make_clustered(n, 10, rng)
    labels = WKMeans(n_clusters=10, beta=6.5, n_init=1, max_iter=20,
                     random_state=42).fit(X).predict(X)

    print(f"{'子样本':>10s} {'单次耗时':>10s} {'3 次重复均值':>14s} "
          f"{'标准差':>10s} {'相对标准差':>12s}")
    for s in SIL_SAMPLE_SIZES:
        vals: List[float] = []
        t0 = time.time()
        for r in range(3):
            vals.append(float(silhouette_score(
                X, labels, sample_size=s, random_state=100 + r)))
        dt = (time.time() - t0) / 3
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        print(f"{s:>10,} {dt:>9.1f}s {mu:>14.5f} {sd:>10.5f} "
              f"{sd / abs(mu) * 100 if mu else float('nan'):>11.2f}%", flush=True)

    print("\n" + "=" * 78)
    print("Davies-Bouldin / Calinski-Harabasz：全量计算（O(n·k)，无需抽样）")
    print("=" * 78)
    t0 = time.time()
    db = davies_bouldin_score(X, labels)
    print(f"Davies-Bouldin   {time.time() - t0:>7.1f}s   值={db:.4f}", flush=True)
    t0 = time.time()
    ch = calinski_harabasz_score(X, labels)
    print(f"Calinski-Harab.  {time.time() - t0:>7.1f}s   值={ch:.1f}", flush=True)


def main() -> int:
    """主函数"""
    rng = np.random.default_rng(0)
    try:
        bench_fit(rng)
        bench_silhouette(rng)
        print("\n✅ 基准测试完成")
        return 0
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
