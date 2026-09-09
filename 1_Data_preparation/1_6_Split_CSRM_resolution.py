"""
1_6_Split_CSRM_resolution.py
============================

将 CSRM1.0 混合分辨率模型拆分为两个部分：
- 0.2° 区域
- 0.4° 区域

判别规则（自动）：
- 若某有效网格点在四邻域中存在相距 0.2° 的有效点，则归为 0.2° 区域。
- 其余有效点归为 0.4° 区域。

输出文件默认保存到：
- data/models/processed/2023_CSRM1.0/2023_CSRM1.0_res0.2.nc
- data/models/processed/2023_CSRM1.0/2023_CSRM1.0_res0.4.nc
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr


def _find_coord_names(ds: xr.Dataset) -> Tuple[str, str, str]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    depth_name = "depth" if "depth" in ds.coords else "dep"
    return lat_name, lon_name, depth_name


def _find_reference_var(ds: xr.Dataset) -> str:
    for candidate in ("vs", "vp", "vsv", "vpv"):
        if candidate in ds.data_vars:
            return candidate
    return list(ds.data_vars.keys())[0]


def _build_neighbor_index(values: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
    val_to_idx = {round(float(v), 6): i for i, v in enumerate(values)}
    plus = np.full(len(values), -1, dtype=int)
    minus = np.full(len(values), -1, dtype=int)

    for i, v in enumerate(values):
        plus_val = round(float(v + step), 6)
        minus_val = round(float(v - step), 6)
        if plus_val in val_to_idx:
            plus[i] = val_to_idx[plus_val]
        if minus_val in val_to_idx:
            minus[i] = val_to_idx[minus_val]

    return plus, minus


def _classify_resolution_masks(valid_2d: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat_plus, lat_minus = _build_neighbor_index(lat, 0.2)
    lon_plus, lon_minus = _build_neighbor_index(lon, 0.2)

    n_lat, n_lon = valid_2d.shape
    hi_mask = np.zeros((n_lat, n_lon), dtype=bool)

    for i in range(n_lat):
        for j in range(n_lon):
            if not valid_2d[i, j]:
                continue

            is_hi = False

            ip = lat_plus[i]
            if ip >= 0 and valid_2d[ip, j]:
                is_hi = True

            im = lat_minus[i]
            if im >= 0 and valid_2d[im, j]:
                is_hi = True

            jp = lon_plus[j]
            if jp >= 0 and valid_2d[i, jp]:
                is_hi = True

            jm = lon_minus[j]
            if jm >= 0 and valid_2d[i, jm]:
                is_hi = True

            hi_mask[i, j] = is_hi

    lo_mask = valid_2d & (~hi_mask)
    return hi_mask, lo_mask


def split_csrm_resolution(
    input_nc: Path,
    out_nc_02: Path,
    out_nc_04: Path,
    summary_json: Path,
) -> None:
    ds = xr.open_dataset(input_nc)

    lat_name, lon_name, depth_name = _find_coord_names(ds)
    ref_var = _find_reference_var(ds)

    ref = ds[ref_var]
    if {lat_name, lon_name, depth_name}.issubset(set(ref.dims)):
        valid_2d = np.isfinite(ref).any(dim=depth_name).transpose(lat_name, lon_name).values
    else:
        raise ValueError(f"参考变量 {ref_var} 缺少经纬深三维坐标，dims={ref.dims}")

    lat = ds[lat_name].values
    lon = ds[lon_name].values

    mask_02, mask_04 = _classify_resolution_masks(valid_2d, lat, lon)

    mask_02_da = xr.DataArray(mask_02, dims=(lat_name, lon_name), coords={lat_name: lat, lon_name: lon})
    mask_04_da = xr.DataArray(mask_04, dims=(lat_name, lon_name), coords={lat_name: lat, lon_name: lon})

    ds_02 = ds.copy(deep=True)
    ds_04 = ds.copy(deep=True)

    for var_name in ds.data_vars:
        var = ds[var_name]
        if lat_name in var.dims and lon_name in var.dims:
            # res0.2 保留 hi_mask（0.2° 区域）的数据，其他设为 NaN
            ds_02[var_name] = var.where(mask_02_da)
            # res0.4 保留 ~hi_mask（0.4° 区域）的数据，其他设为 NaN
            # 关键修正：直接用 mask_02_da 的反面，确保两个文件互不重叠
            ds_04[var_name] = var.where(~mask_02_da)

    n_valid = int(valid_2d.sum())
    n_02 = int(mask_02.sum())
    n_04 = int((~mask_02).sum())

    common_attrs = {
        "split_source": str(input_nc),
        "split_method": "neighbor_0.2deg_detection",
        "split_description": "point is 0.2deg if any N/S/E/W valid neighbor at 0.2deg; otherwise 0.4deg",
        "reference_variable": ref_var,
        "n_valid_points_2d": n_valid,
        "n_points_0p2_2d": n_02,
        "n_points_0p4_2d": n_04,
        "ratio_0p2_percent": round(100.0 * n_02 / n_valid, 2) if n_valid else 0.0,
        "ratio_0p4_percent": round(100.0 * n_04 / n_valid, 2) if n_valid else 0.0,
    }

    ds_02.attrs.update(common_attrs)
    ds_02.attrs["region_resolution"] = "0.2_degree"
    ds_04.attrs.update(common_attrs)
    ds_04.attrs["region_resolution"] = "0.4_degree"

    out_nc_02.parent.mkdir(parents=True, exist_ok=True)
    out_nc_04.parent.mkdir(parents=True, exist_ok=True)

    comp = {name: {"zlib": True, "complevel": 4} for name in ds_02.data_vars}
    ds_02.to_netcdf(out_nc_02, encoding=comp)
    ds_04.to_netcdf(out_nc_04, encoding=comp)

    summary = {
        "input_nc": str(input_nc),
        "output_0p2_nc": str(out_nc_02),
        "output_0p4_nc": str(out_nc_04),
        "reference_variable": ref_var,
        "n_valid_points_2d": n_valid,
        "n_points_0p2_2d": n_02,
        "n_points_0p4_2d": n_04,
        "ratio_0p2_percent": round(100.0 * n_02 / n_valid, 2) if n_valid else 0.0,
        "ratio_0p4_percent": round(100.0 * n_04 / n_valid, 2) if n_valid else 0.0,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> None:
    base_dir = Path("data/models/processed/2023_CSRM1.0")
    input_nc = base_dir / "2023_CSRM1.0_original.nc"
    out_nc_02 = base_dir / "2023_CSRM1.0_res0.2.nc"
    out_nc_04 = base_dir / "2023_CSRM1.0_res0.4.nc"
    summary_json = base_dir / "2023_CSRM1.0_resolution_split_summary.json"

    if not input_nc.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_nc}")

    split_csrm_resolution(input_nc, out_nc_02, out_nc_04, summary_json)

    print("Split done:")
    print(f"  - {out_nc_02}")
    print(f"  - {out_nc_04}")
    print(f"  - {summary_json}")


if __name__ == "__main__":
    main()
