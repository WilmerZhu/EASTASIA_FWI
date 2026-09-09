"""
可分辨速度类别数上界估计

一个深度带能区分多少个速度类别，上界由 δlnVs 的动态范围除以速度不确定度
决定。层析模型不提供逐体元不确定度，此处以**模型间差异**作经验代理。

注意该代理偏保守：模型间差异同时包含真实不确定度、分辨率差异（SinoScope
为 1.0° 需插值到 0.25°）、参数化与参考模型差异。故所得 N_res 是可分辨类别
数的下界，适用于"跨模型稳健类别数"的论断；单模型自身的可分辨数应更高。
"""
import warnings

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

MODELS = ["2022_SinoScope1.0", "2024_EARA2024", "2024_FWEA23"]
BANDS = [
    ("crust", 0, 50),
    ("lithosphere", 50, 410),
    ("transition_zone", 410, 660),
    ("lower_mantle", 660, 1000),
]
DIMS = ("longitude", "latitude", "depth")


def load_dlnvs(model: str, ref_grid) -> np.ndarray:
    """载入 vs → 相对各深度层水平平均的 δlnVs，插值到公共网格，返回 (lon,lat,dep)"""
    ds = xr.open_dataset(f"data/models/processed/{model}/{model}_standardized.nc")
    vs = ds["vs"].where(ds["vs"] != 9999.0)
    vs = vs.interp(**{d: ref_grid[d] for d in DIMS})
    ref = vs.mean(dim=["longitude", "latitude"], skipna=True)
    out = ((vs - ref) / ref).transpose(*DIMS).values
    ds.close()
    return out


base = xr.open_dataset(
    "data/models/processed/2024_EARA2024/2024_EARA2024_standardized.nc"
)
grid = {d: base[d] for d in DIMS}
depths = base["depth"].values
base.close()

# 全部在同一轴序 (lon,lat,depth) 下计算，避免掩膜与数据轴序错位
arrs = {m: load_dlnvs(m, grid) for m in MODELS}
valid = np.ones_like(arrs[MODELS[0]], dtype=bool)
for m in MODELS:
    valid &= np.isfinite(arrs[m])
print(f"公共有效体元: {valid.sum():,} / {valid.size:,} ({valid.sum()/valid.size:.1%})")

# ── 先看模型间分歧随深度的分布，判断地壳的高 RMS 是否集中在最浅几层 ──
print("\n模型间 RMS 差异随深度：")
print(f"{'深度(km)':>9}{'RMS 差异':>11}{'三模型平均 std':>16}{'差异/信号':>11}")
print("-" * 48)
for i in range(0, len(depths), max(1, len(depths) // 16)):
    bm = valid[:, :, i]
    if bm.sum() < 50:
        continue
    ds_ = [np.nanstd(arrs[m][:, :, i][bm]) for m in MODELS]
    pair = [
        np.sqrt(np.nanmean((arrs[a][:, :, i][bm] - arrs[b][:, :, i][bm]) ** 2))
        for k, a in enumerate(MODELS) for b in MODELS[k + 1:]
    ]
    r, s = float(np.mean(pair)), float(np.mean(ds_))
    print(f"{depths[i]:>9.0f}{r*100:>10.2f}%{s*100:>15.2f}%{r/max(s,1e-9):>11.2f}")

print("\n" + "=" * 88)
print(f"{'深度带':<17}{'模型':<20}{'δlnVs 动态范围':>16}{'模型间 RMS':>13}{'N_res':>9}")
print("-" * 88)

summary = {}
for bname, z0, z1 in BANDS:
    zm = (depths >= z0) & (depths < z1)
    bm = valid & zm[None, None, :]
    if bm.sum() < 100:
        continue
    pair = [
        np.sqrt(np.nanmean((arrs[a][bm] - arrs[b][bm]) ** 2))
        for k, a in enumerate(MODELS) for b in MODELS[k + 1:]
    ]
    rms = float(np.mean(pair))
    for m in MODELS:
        x = arrs[m][bm]
        rng = float(np.nanpercentile(x, 97.5) - np.nanpercentile(x, 2.5))
        n = rng / rms
        print(f"{bname if m == MODELS[0] else '':<17}{m:<20}"
              f"{rng*100:>14.2f}%{rms*100:>12.2f}%{n:>9.1f}")
        summary.setdefault(bname, []).append(n)
    print("-" * 88)

print("=" * 88)
print("\n可分辨类别数（三模型平均）与拟用 K 对照：")
print(f"{'深度带':<20}{'N_res':>9}{'\u62df\u7528 K':>10}{'\u7ed3\u8bba':>16}")
print("-" * 58)
proposed = {"crust": 10, "lithosphere": 5, "transition_zone": 5, "lower_mantle": 5}
for bname, vals in summary.items():
    n, k = float(np.mean(vals)), proposed[bname]
    verdict = "✅ 在界内" if k <= n else "❌ 超出界"
    print(f"{bname:<20}{n:>9.1f}{k:>10}{verdict:>16}")
print("-" * 58 + "\n")
