"""pytest 公共 fixture：项目根路径、合成速度模型、模块加载器。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_module(rel_path: str) -> ModuleType:
    """按路径加载数字开头的管线模块（如 '2_Model_space_analysis/2_3_Model_clustering.py'）。"""
    path = PROJECT_ROOT / rel_path
    name = "ea_" + path.stem.replace("-", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260917)


@pytest.fixture
def synthetic_model(rng: np.random.Generator):
    """小型合成速度模型 (lat 5 × lon 6 × depth 4)，单位 km/s、g/cm³。"""
    xr = pytest.importorskip("xarray")
    lat = np.linspace(20.0, 40.0, 5)
    lon = np.linspace(100.0, 125.0, 6)
    depth = np.array([10.0, 50.0, 100.0, 200.0])
    base_vs = 3.5 + 0.004 * depth[None, None, :]
    vs = base_vs + 0.05 * rng.standard_normal((5, 6, 4))
    vp = vs * 1.75
    rho = 2.7 + 0.001 * depth[None, None, :] + np.zeros_like(vs)
    return xr.Dataset(
        {
            "vs": (("latitude", "longitude", "depth"), vs),
            "vp": (("latitude", "longitude", "depth"), vp),
            "rho": (("latitude", "longitude", "depth"), rho),
        },
        coords={"latitude": lat, "longitude": lon, "depth": depth},
    )
