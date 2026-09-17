---
description: "按地震学科学正确性 + 项目代码规范审查当前工作区的未提交改动（git diff），输出分级问题清单"
argument-hint: "可选：限定路径或文件，如 4_Fusion/"
agent: "reviewer"
---
审查 ${input:scope:当前全部未提交改动（留空）或指定路径} 的改动。

执行：
1. `git status --short` 与 `git diff` / `git diff --stat`（含未跟踪的新文件，用 `git diff --no-index /dev/null <file>` 或直接阅读）。
2. 对每个改动文件按以下维度检查：
   - **科学正确性**：单位（km/s、km、g/cm³）、坐标顺序（lat, lon, depth）、区域边界是否来自 `BaseConfig`、各向异性参数（vpv/vph/vsv/vsh/eta/rho）是否整组处理、插值/掩膜边界处理。
   - **数值稳定性**：随机种子、除零、NaN 传播、容差硬编码。
   - **项目规范**：[AGENTS.md](../../AGENTS.md) 与 `.github/instructions/` 中的约定（配置双层、logging、类型注解、命名、jpg+pdf 输出）。
   - **可维护性**：单文件是否超 2000 行仍在增长、重复实现已有 utils 的功能、未使用的导入。
   - **安全**：是否写入/删除 `data/ results/ figures/`，是否硬编码路径或凭据。
3. 不修改任何文件。

输出格式：
```
## 阻塞（必须修）
- 文件:行 — 问题 — 建议
## 建议（应修）
...
## 备注（可选）
...
## 总结
一句话：可否提交。
```
