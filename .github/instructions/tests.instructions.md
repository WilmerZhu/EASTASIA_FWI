---
description: "Use when writing pytest tests under tests/: synthetic velocity model fixtures, golden-file regression for clustering and SDL fusion, tolerances, avoiding real data/ dependencies."
applyTo: "tests/**"
---
# 测试规范

## 原则

- 测试不依赖 `data/`、`results/`、`figures/` 等不入库目录；所有输入用合成数据或 `tests/fixtures/` 内的小文件（< 1 MB）。
- 不进行网络访问（IRIS、GCMT 下载等）；需要时用 `monkeypatch` 替换客户端。
- 不生成图片；绘图函数只测数据准备逻辑，或用 `matplotlib.use('Agg')` 并断言返回对象。
- 每个测试函数一个断言主题；命名 `test_<被测行为>_<条件>`。

## 合成速度模型

- 用 `xarray.Dataset` 构造小网格（如 5×5×4），坐标 `latitude / longitude / depth`，变量 `vs vp rho`，单位 km/s、g/cm³。
- 通过 `numpy.random.default_rng(seed)` 固定随机种子；把构造函数放在 `tests/conftest.py` 作为 fixture 复用。

## Golden-file 回归

- 金标准结果存 `tests/golden/<模块>_<用例>.npz` 或 `.json`，由脚本一次性生成并入库。
- 比较用 `numpy.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)`；聚类标签比较用 ARI（`sklearn.metrics.adjusted_rand_score`）≥ 0.99 而不是逐元素相等（标签可置换）。
- 有意更改算法导致金标准变化时，在提交信息中说明并重新生成文件，不要放宽容差来掩盖差异。

## 模块导入

- `tests/conftest.py` 已把项目根目录加入 `sys.path`；数字开头的模块文件用 `importlib.import_module` 或 `importlib.util.spec_from_file_location` 加载，例如：

```python
import importlib.util
from pathlib import Path

def load_module(rel_path: str):
    path = Path(__file__).resolve().parents[1] / rel_path
    spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
```

- 重型依赖（PyGMT、ObsPy、SPECFEM 工具）用 `pytest.importorskip` 保护，保证 CI 轻量环境可跑。

## 运行

```bash
pytest -q            # 全部
pytest -q -k sdl     # 按关键词
```
