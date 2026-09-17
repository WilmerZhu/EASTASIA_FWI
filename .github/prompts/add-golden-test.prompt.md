---
description: "为指定管线模块生成基于合成数据的 pytest 回归测试（golden-file），用于聚类、SDL 融合、模型对比等参数敏感算法"
argument-hint: "模块路径，如 4_Fusion/4_3_SDL_Fusion_v4.py，可附带要覆盖的函数名"
agent: "agent"
---
为 ${input:module:模块相对路径，如 2_Model_space_analysis/2_3_Model_clustering.py} 生成回归测试。

要求：
1. 先阅读该模块，找出核心纯计算函数/方法（不做 I/O、不绑图的部分），列出 2–4 个最值得保护的入口。
2. 在 `tests/` 下新建 `test_<模块简名>.py`，遵循 [tests.instructions.md](../instructions/tests.instructions.md)：
   - 用 `tests/conftest.py` 中的合成模型 fixture 或新增小型 fixture；
   - 数字开头的模块用 `load_module()` 方式导入；
   - 重型依赖用 `pytest.importorskip`。
3. 首次运行时生成金标准到 `tests/golden/<模块简名>_<用例>.npz`，测试中用 `assert_allclose`（聚类标签用 ARI ≥ 0.99）。
4. 运行 `pytest -q tests/test_<模块简名>.py` 并确保通过；把运行结果贴出。
5. 不修改被测模块本身；若发现无法脱离 `data/` 测试的入口，列出并说明需要怎样重构才可测。

输出：新增的测试文件路径、金标准文件列表、pytest 结果摘要。
