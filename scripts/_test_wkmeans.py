#!/usr/bin/env python3
"""
WKMeans 正确性验证（合成数据）
==============================

验证三项性质:
1. 目标函数在迭代中单调不增（Huang et al. 2005 收敛性的必要条件）
2. 权重能抑制噪声特征：informative 特征应获得显著高于 noise 特征的权重
3. 在含噪声特征的数据上，聚类精度应优于等权 k-means
4. beta ∈ [0,1] 时按论文要求拒绝执行

作者: EASTASIA-FWI Team
日期: 2026-09-07
"""

import importlib
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / '2_Model_space_analysis'))
mod = importlib.import_module('2_3_Model_clustering')
WKMeans = mod.WKMeans


def make_data(seed: int = 0):
    """3 个有信息特征 + 3 个纯噪声特征，4 个簇"""
    rng = np.random.default_rng(seed)
    n_per, k = 500, 4
    centers = rng.normal(0, 4.0, size=(k, 3))
    Xi, y = [], []
    for l in range(k):
        Xi.append(rng.normal(centers[l], 1.0, size=(n_per, 3)))
        y.append(np.full(n_per, l))
    Xi = np.vstack(Xi)
    y = np.concatenate(y)
    noise = rng.normal(0, 4.0, size=(len(Xi), 3))  # 与簇结构无关，方差相当
    X = np.hstack([Xi, noise])
    # 标准化，避免尺度差异冒充权重效应
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def main() -> None:
    print("🎯 WKMeans 合成数据验证")
    print("=" * 68)
    X, y_true = make_data()
    print(f"数据: {X.shape}，前 3 列有信息，后 3 列为噪声，真实簇数 4\n")

    # ---- 1. 目标函数单调性 ----
    print("[1] 目标函数单调性（单次初始化，逐轮记录）")
    wk = WKMeans(n_clusters=4, beta=6.5, n_init=1, max_iter=30, tol=0.0)
    rng = np.random.default_rng(0)
    n_feat = X.shape[1]
    from sklearn.cluster import kmeans_plusplus
    centers, _ = kmeans_plusplus(X, n_clusters=4, random_state=0)
    centers = np.asarray(centers, dtype=float)
    weights = np.full(n_feat, 1.0 / n_feat)
    objs = []
    for _ in range(15):
        labels, _ = wk._assign(X, centers, weights ** wk.beta)
        centers = wk._update_centers(X, labels, centers)
        weights = wk._update_weights(X, labels, centers)
        _, obj = wk._assign(X, centers, weights ** wk.beta)
        objs.append(obj)
    diffs = np.diff(objs)
    monotone = bool(np.all(diffs <= 1e-9))
    print(f"    目标函数序列前 6 项: {[f'{o:.2f}' for o in objs[:6]]}")
    print(f"    最大正向增量: {diffs.max():.3e}")
    print(f"    单调不增: {'✅ 通过' if monotone else '❌ 失败'}\n")

    # ---- 2. 权重是否抑制噪声特征 ----
    print("[2] 特征权重（β=6.5，n_init=5）")
    wk = WKMeans(n_clusters=4, beta=6.5, n_init=5, random_state=42).fit(X)
    w = wk.feature_weights_
    print(f"    informative: {np.array2string(w[:3], precision=4)}")
    print(f"    noise      : {np.array2string(w[3:], precision=4)}")
    ratio = float(w[:3].mean() / w[3:].mean())
    # 距离中真正生效的是 w^β，故以有效权重比作为判据
    eff = ratio ** wk.beta
    print(f"    权重比 (info/noise) = {ratio:.2f}")
    print(f"    有效权重比 w^β      = {eff:.1f}（距离中实际生效的倍数）")
    print(f"    抑制噪声: {'✅ 通过' if eff > 3.0 else '❌ 失败'}")
    print(f"    权重和 = {w.sum():.6f}（应为 1）")
    print(f"    收敛: {wk.converged_}, 迭代 {wk.n_iter_} 轮\n")

    # ---- 3. 与等权 k-means 的精度对比 ----
    print("[3] 聚类精度对比（Adjusted Rand Index，越接近 1 越好）")
    ari_wk = adjusted_rand_score(y_true, wk.labels_)
    km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
    ari_km = adjusted_rand_score(y_true, km.labels_)
    print(f"    W-k-means : ARI = {ari_wk:.4f}")
    print(f"    k-means   : ARI = {ari_km:.4f}")
    print(f"    W-k-means 更优: {'✅ 通过' if ari_wk > ari_km else '❌ 失败'}\n")

    # ---- 4. beta 合法性检查 ----
    print("[4] beta ∈ [0,1] 应被拒绝")
    try:
        WKMeans(n_clusters=4, beta=0.5)
        print("    ❌ 失败：未拒绝 beta=0.5")
    except ValueError as e:
        print(f"    ✅ 通过：{e}\n")

    # ---- 5. β 的影响 ----
    print("[5] β 对权重集中度的影响（同一数据）")
    print(f"    {'beta':>6}{'info 均值':>12}{'noise 均值':>12}{'比值':>8}{'ARI':>8}")
    for beta in (1.5, 2.0, 4.0, 6.5, 10.0, 20.0):
        m = WKMeans(n_clusters=4, beta=beta, n_init=3, random_state=42).fit(X)
        wi, wn = m.feature_weights_[:3].mean(), m.feature_weights_[3:].mean()
        print(f"    {beta:>6.1f}{wi:>12.4f}{wn:>12.4f}{wi / wn:>8.2f}"
              f"{adjusted_rand_score(y_true, m.labels_):>8.4f}")

    print("=" * 68)


if __name__ == '__main__':
    main()
