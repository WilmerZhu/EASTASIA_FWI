"""BaseConfig 回归：研究区域、目录派生、日志器行为。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from config.base_config import BaseConfig, ConfigValidator


@pytest.fixture
def cfg() -> BaseConfig:
    return BaseConfig()


def test_region_bounds_are_consistent(cfg: BaseConfig) -> None:
    b = cfg.get_region_bounds()
    assert set(b) == {"lat_min", "lat_max", "lon_min", "lon_max", "depth_min", "depth_max"}
    assert b["lat_min"] < b["lat_max"]
    assert b["lon_min"] < b["lon_max"]
    assert 0.0 <= b["depth_min"] < b["depth_max"]
    assert ConfigValidator.validate_region(cfg.region) == []


def test_region_covers_east_asia_core(cfg: BaseConfig) -> None:
    # 青藏高原、华北、日本海沟均应落在研究区域内
    assert cfg.is_in_region(32.0, 90.0)
    assert cfg.is_in_region(40.0, 116.0, 100.0)
    assert cfg.is_in_region(36.0, 142.0, 500.0)
    assert not cfg.is_in_region(0.0, 0.0)


def test_dirs_derive_from_project_root(cfg: BaseConfig) -> None:
    root: Path = cfg.dirs["project_root"]
    assert (root / "config" / "base_config.py").exists()
    for key in ("data", "results", "figures", "logs", "models", "stations", "events"):
        assert cfg.dirs[key].is_relative_to(root), key
    assert cfg.dirs["models"] == cfg.dirs["data"] / "models"


def test_setup_logger_is_idempotent(cfg: BaseConfig) -> None:
    name = "EASTASIA-FWI.TestModule"
    logger1 = cfg.setup_logger(name, "DEBUG")
    n_handlers = len(logger1.handlers)
    logger2 = cfg.setup_logger(name, "DEBUG")
    assert logger1 is logger2
    assert len(logger2.handlers) == n_handlers
    assert logger1.level == logging.DEBUG


def test_load_region_override(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    override = tmp_path / "region.yaml"
    override.write_text(yaml.safe_dump({"region": {"lat_min": 0.0, "lat_max": 30.0}}), encoding="utf-8")
    cfg = BaseConfig(config_file=override)
    assert cfg.region["lat_min"] == 0.0
    assert cfg.region["lat_max"] == 30.0
    assert cfg.region["lon_min"] == 60.0  # 未覆盖的键保持默认


# ---------------------------------------------------------------- 三层区域


def test_common_region_is_intersection_of_core_models(cfg: BaseConfig) -> None:
    common = cfg.get_region_bounds("common")
    cores = [cfg.get_region_bounds(m) for m in cfg.core_models]
    assert common["lat_min"] == pytest.approx(max(r["lat_min"] for r in cores))
    assert common["lat_max"] == pytest.approx(min(r["lat_max"] for r in cores))
    assert common["lon_min"] == pytest.approx(max(r["lon_min"] for r in cores))
    assert common["lon_max"] == pytest.approx(min(r["lon_max"] for r in cores))
    assert common["depth_max"] == pytest.approx(min(r["depth_max"] for r in cores))
    assert ConfigValidator.validate_region(common) == []


def test_common_region_inside_study_region(cfg: BaseConfig) -> None:
    study, common = cfg.get_region_bounds("study"), cfg.get_region_bounds("common")
    assert study["lat_min"] <= common["lat_min"] < common["lat_max"] <= study["lat_max"]
    assert study["lon_min"] <= common["lon_min"] < common["lon_max"] <= study["lon_max"]


def test_model_region_resolves_short_names(cfg: BaseConfig) -> None:
    assert cfg.resolve_model_name("FWEA23") == "2024_FWEA23"
    assert cfg.resolve_model_name("SinoScope") == "2022_SinoScope1.0"
    assert cfg.get_region_bounds("EARA2024") == cfg.get_region_bounds("2024_EARA2024")
    with pytest.raises(KeyError):
        cfg.get_region_bounds("NoSuchModel_XYZ")


def test_gmt_region_order_and_kind(cfg: BaseConfig) -> None:
    for kind in ("study", "common", "FWEA23"):
        b = cfg.get_region_bounds(kind)
        assert cfg.get_gmt_region(kind) == [b["lon_min"], b["lon_max"], b["lat_min"], b["lat_max"]]


def test_is_in_region_respects_kind(cfg: BaseConfig) -> None:
    # 南海 (5°N, 110°E) 在 study 与 SinoScope 内，但不在 common（lat ≥ 10°）内
    assert cfg.is_in_region(5.0, 110.0)
    assert cfg.is_in_region(5.0, 110.0, kind="SinoScope")
    assert not cfg.is_in_region(5.0, 110.0, kind="common")


def test_compute_common_region_rejects_disjoint() -> None:
    cfg = BaseConfig()
    cfg.model_regions["_far"] = {
        "lat_min": -80.0,
        "lat_max": -70.0,
        "lon_min": 0.0,
        "lon_max": 10.0,
        "depth_min": 0.0,
        "depth_max": 100.0,
    }
    with pytest.raises(ValueError):
        cfg.compute_common_region(["2024_FWEA23", "_far"])


def test_load_model_regions_override_recomputes_common(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    override = tmp_path / "models.yaml"
    override.write_text(yaml.safe_dump({"model_regions": {"2024_EARA2024": {"lat_min": 20.0}}}), encoding="utf-8")
    cfg = BaseConfig(config_file=override)
    assert cfg.get_region_bounds("EARA2024")["lat_min"] == 20.0
    assert cfg.get_region_bounds("common")["lat_min"] == 20.0
