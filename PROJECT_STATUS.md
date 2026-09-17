# EASTASIA-FWI 项目文件状态

> 最后更新：2026-09-17

## ✅ 已具备的工程文件

| 文件 | 状态 | 说明 |
|------|------|------|
| README.md | ✅ | 项目说明、结构树、开发优先级 |
| CHANGELOG.md | ✅ | Keep a Changelog 格式 |
| LICENSE | ✅ | 开源许可证 |
| CITATION.cff | ✅ | 学术引用信息 |
| AUTHORS.md | ✅ | 作者信息 |
| CONTRIBUTING.md | ✅ | 贡献指南 |
| GIT_COMMIT_GUIDE.md | ✅ | 提交规范 |
| environment.yml | ✅ | Conda 完整环境（Python 3.12） |
| requirements.txt | ✅ | pip 依赖列表 |
| .gitignore | ✅ | 已覆盖 data/results/figures/output/specfem*/paper 二进制等 |
| config/base_config.py | ✅ | 全局配置，路径基于 `Path(__file__)` 推导 |
| paper/（源码） | ✅ | 论文 .md 文稿与 .py 构建脚本入库；docx/pdf/图片不入库 |
| scripts/ | ✅ | 实验与工作流脚本入库 |
| archive/ | ✅ | 历史版本归档（如 SDL v3.4），不参与主管线 |
| AGENTS.md | ✅ | 面向 AI agent 的常驻指令（≤ 100 行：语言/配置权威源/命名/边界） |
| .github/instructions/ | ✅ | 按 `applyTo` 拆分：`python-modules`（管线 .py）、`visualization`（5_Visualization）、`tests`（tests/） |
| .github/prompts/ | ✅ | `/add-golden-test` `/review-changes` `/sync-docs` 三个可参数化任务 |
| .github/agents/ | ✅ | `reviewer`（只读审查）、`seismo-planner`（只做计划） |
| .github/hooks/ | ✅ | `guard.json`：拒绝 agent 写入 data/results/figures/output/logs/backup；编辑 .py 后自动语法检查 |
| .github/workflows/ci.yml | ✅ | compileall + black（tests/）+ pytest（轻量依赖，重型库 importorskip） |
| tests/ | ✅ 骨架 | `conftest.py`（合成模型 fixture + `load_module()`）、`test_base_config.py`、`test_synthetic_fixtures.py`；待补算法回归 |
| docs/design/ docs/plans/ | ✅ | 设计文档（主设计 v2.2、SDL 设计、旧版指令归档）与历史计划从 .github/ 迁入；docs/ 仅忽略 PDF 等二进制 |

## ❗ 待补充（按优先级）

### 🔴 高优先级

1. **算法回归测试** — tests/ 骨架与 CI 已就位，但核心算法尚无覆盖。
   用 `/add-golden-test 2_Model_space_analysis/2_3_Model_clustering.py` 与
   `/add-golden-test 4_Fusion/4_3_SDL_Fusion_v4.py` 建立合成数据 golden-file 测试，
   防止参数调整后数值结果悄悄变化。

2. **未提交的大体量新文件** — `3_4_Waveform_assessment.py`、`5_11/5_12/5_13` 合计约 9000 行尚未入库，
   建议先 `/review-changes` 审查后提交，并用 `/sync-docs` 同步 README。

### 🟡 中优先级

3. **超大文件拆分** — 以下单文件超 5000 行，建议按"配置 / 数据加载 / 算法 / 输出"分层：
   - `4_Fusion/4_3_SDL_Fusion_v4.py`（约 6000 行，v3.4 已归档）
   - `2_Model_space_analysis/2_3_Model_clustering.py`（约 5600 行）
   - `5_Visualization/5_10_Clustering_visualization.py`（约 5300 行）

4. **重复脚本收敛** — `netCDF_2_GeoCSV_3D.py`（约 410 行）在
   `data/models/metadata/*/` 等处有 9 份相同拷贝，建议抽为公共工具模块。
   （data/ 不入库，仅影响本地工作区。）

5. **日志规范化** — 管线中仍有约 1000 处 `print()`，16 个模块已用 logging；
   建议新代码统一使用 logging，便于批量正演时的日志留存（logs/ 已不入库）。

### 🟢 低优先级

6. **docs/ 二进制** — .md 已入库，PDF/图片仍忽略；参考论文 PDF 建议移到 `docs/refs/`（不入库）以免与设计文档混放。

7. **版本号管理** — CHANGELOG 的 [Unreleased] 已积累较多内容，
   建议在 SDL v4 验证节点发布 v1.1.0 并补记变更。

---

## 版本说明

- 语义化版本（Semantic Versioning）
- [Unreleased] 内容见各模块 commit 历史
