---
description: "扫描各模块目录，把 README.md 结构树、开发优先级和 PROJECT_STATUS.md 与实际文件同步"
agent: "agent"
---
同步项目文档与实际代码状态。

步骤：
1. 列出 `1_Data_preparation/ 2_Model_space_analysis/ 3_Data_space_simulation/ 4_Fusion/ 5_Visualization/ config/ tests/ scripts/` 下的 `.py` 文件（忽略 `__pycache__`）。
2. 对比 [README.md](../../README.md) 的“项目结构”树与“开发优先级”章节，找出：新增但未列出的文件、已列出但已归档/删除的文件、描述与文件头 docstring 不一致的条目。
3. 对比 [PROJECT_STATUS.md](../../PROJECT_STATUS.md) 的“待补充”列表，标记已经完成的项（如 tests/ 已存在、CI 已配置）。
4. 用 `git log --oneline -15` 与 `git status --short` 补充“最近变更”事实。
5. 直接修改这两个文件：只改动有差异的行，保持原有格式与中文说明；更新“最后更新”日期为今天。
6. 不要新建其他 markdown 文件；不要修改 `CHANGELOG.md`（除非用户要求）。

输出：修改摘要（新增/删除/修正了哪些条目）。
