# EASTASIA-FWI 项目文件状态

> 最后更新：2026-09-09

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

## ❗ 待补充（按优先级）

### 🔴 高优先级

1. **tests/ 单元测试** — 环境已含 pytest，但目前无回归测试。
   建议优先为 2_3 聚类、4_3 SDL 融合用小型合成数据（synthetic）建立 golden-file 测试，
   防止参数调整后数值结果悄悄变化。

2. **CI 配置** — `.github/workflows/` 暂无自动化。
   建议加：`pytest` 测试 + `black --check` 风格检查 + 依赖解析检查。
   注意：当前 `.github/` 整体在 .gitignore 中（含 AI 指令文件），
   如需启用 CI，需把 workflows 单独加入版本控制（如 `!.github/workflows/`）。

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

6. **docs/ 文档** — 当前 docs/ 整体不入库（含参考论文 PDF）；
   如需协作，可把其中 .md 设计文档改为入库、仅忽略 PDF。

7. **版本号管理** — CHANGELOG 的 [Unreleased] 已积累较多内容，
   建议在 SDL v4 验证节点发布 v1.1.0 并补记变更。

---

## 版本说明

- 语义化版本（Semantic Versioning）
- [Unreleased] 内容见各模块 commit 历史
