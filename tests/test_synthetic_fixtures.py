"""合成模型 fixture 自检 + 标准化 NetCDF 约定（坐标名、单位量级）。"""

from __future__ import annotations

import numpy as np
import pytest


def test_synthetic_model_follows_netcdf_convention(synthetic_model) -> None:
    assert tuple(synthetic_model["vs"].dims) == ("latitude", "longitude", "depth")
    assert {"vs", "vp", "rho"} <= set(synthetic_model.data_vars)
    # 单位量级：km/s 与 g/cm³
    assert 2.0 < float(synthetic_model["vs"].min()) < float(synthetic_model["vs"].max()) < 6.0
    assert 2.0 < float(synthetic_model["rho"].min()) < 4.0
    # 坐标单调递增，深度为 km
    for c in ("latitude", "longitude", "depth"):
        assert np.all(np.diff(synthetic_model[c].values) > 0), c
    assert float(synthetic_model["depth"].max()) <= 1000.0


def test_vp_vs_ratio_is_physical(synthetic_model) -> None:
    ratio = (synthetic_model["vp"] / synthetic_model["vs"]).values
    # 体积模量为正要求 Vp/Vs > sqrt(4/3) ≈ 1.1547
    assert np.all(ratio > 1.1547)
    assert np.allclose(ratio, 1.75)


def test_rng_fixture_is_reproducible(rng: np.random.Generator) -> None:
    first = rng.standard_normal(3)
    expected = np.random.default_rng(20260917).standard_normal(3)
    np.testing.assert_allclose(first, expected)
    pytest.importorskip("xarray")
