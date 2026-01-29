# Git 提交步骤指南

## 📋 当前状态

根据 `git status`，你有：

### 修改的文件（Modified）

- `5_Visualization/5_1_Basemap.py` - 已修改
- `README.md` - 已修改

### 新文件（Untracked）

- `1_Data_preparation/1_3_Download_waveforms.py`
- `5_Visualization/5_2_GCMT.py`
- `5_Visualization/5_3_All_Stations.py`
- `5_Visualization/utils/`（目录）
- `AUTHORS.md`
- `CHANGELOG.md`
- `CITATION.cff`
- `CONTRIBUTING.md`
- `LICENSE`
- `PROJECT_STATUS.md`
- `PUBLICATION_GUIDE.md`
- `requirements.txt`

---

## 🚀 Git 提交步骤

### 步骤 1: 查看修改内容（可选但推荐）

在提交前，先查看具体修改了什么：

```bash
# 查看某个文件的修改内容
git diff 5_Visualization/5_1_Basemap.py

# 查看所有修改的文件
git diff

# 查看新文件（预览）
ls -la 5_Visualization/utils/
```

**作用**: 确认修改是否正确，避免提交错误

---

### 步骤 2: 暂存文件（Stage Files）

将文件添加到暂存区（准备提交的文件列表）：

#### 方式 1: 逐个添加文件（推荐，更安全）

```bash
# 添加修改的文件
git add 5_Visualization/5_1_Basemap.py
git add README.md

# 添加新文件
git add 1_Data_preparation/1_3_Download_waveforms.py
git add 5_Visualization/5_2_GCMT.py
git add 5_Visualization/5_3_All_Stations.py
git add 5_Visualization/utils/

# 添加文档文件
git add AUTHORS.md
git add CHANGELOG.md
git add CITATION.cff
git add CONTRIBUTING.md
git add LICENSE
git add PROJECT_STATUS.md
git add PUBLICATION_GUIDE.md
git add requirements.txt
```

#### 方式 2: 添加所有文件（快速但需谨慎）

```bash
# 添加所有修改和新文件
git add .

# 或者只添加特定目录
git add 5_Visualization/
git add 1_Data_preparation/
```

**注意**: `git add .` 会添加所有文件，包括临时文件（如果有的话）

---

### 步骤 3: 检查暂存区状态

确认哪些文件已暂存：

```bash
git status
```

**预期输出**:

- 已暂存的文件会显示在 "Changes to be committed:" 下（绿色）
- 未暂存的文件会显示在 "Changes not staged for commit:" 下（红色）

---

### 步骤 4: 提交更改（Commit）

将暂存的文件提交到本地仓库：

```bash
git commit -m "描述你的更改"
```

#### 提交信息示例：

```bash
# 示例 1: 简洁描述
git commit -m "Add visualization modules and project documentation"

# 示例 2: 详细描述（推荐）
git commit -m "Add: 可视化模块优化和项目文档

- 优化 5_1_Basemap.py，使用共享工具函数
- 修复 5_2_GCMT.py 和 5_3_All_Stations.py 的错误
- 添加项目文档（LICENSE, README, CONTRIBUTING等）
- 创建可视化工具模块（utils/file_utils.py）
- 添加投稿指南（PUBLICATION_GUIDE.md）"

# 示例 3: 多行提交信息
git commit -m "Add: 可视化模块和项目文档

主要更改:
- 修复 5_3_All_Stations.py 的 Matplotlib 弃用警告
- 修复 PyGMT border_pen 格式错误
- 添加自动文件名生成功能
- 创建完整的项目文档结构"
```

**提交信息规范**:

- `Add:` - 添加新功能
- `Fix:` - 修复 bug
- `Update:` - 更新功能
- `Refactor:` - 代码重构
- `Docs:` - 文档更新

---

### 步骤 5: 推送到远程仓库（Push）

将本地提交推送到 GitHub：

```bash
git push origin main
```

**说明**:

- `origin` - 远程仓库的默认名称
- `main` - 分支名称

**如果是第一次推送**，可能需要设置上游分支：

```bash
git push -u origin main
```

---

## 📝 完整示例流程

```bash
# 1. 查看状态
git status

# 2. 查看修改内容（可选）
git diff 5_Visualization/5_1_Basemap.py

# 3. 暂存所有文件
git add .

# 4. 再次确认状态
git status

# 5. 提交
git commit -m "Add: 可视化模块优化和项目文档

- 优化可视化模块，使用共享工具函数
- 修复代码错误和警告
- 添加完整的项目文档结构"

# 6. 推送到远程
git push origin main
```

---

## ⚠️ 常见问题

### 1. 如果提交信息写错了怎么办？

```bash
# 修改最后一次提交信息（还未推送）
git commit --amend -m "新的提交信息"

# 如果已经推送了，需要强制推送（谨慎使用）
git push --force origin main
```

### 2. 如果暂存了错误的文件怎么办？

```bash
# 取消暂存某个文件（但保留修改）
git restore --staged 文件名

# 取消暂存所有文件
git restore --staged .
```

### 3. 如果想撤销修改怎么办？

```bash
# 撤销某个文件的修改（危险：会丢失修改）
git restore 文件名

# 撤销所有修改（危险：会丢失所有修改）
git restore .
```

### 4. 如果推送失败怎么办？

```bash
# 先拉取远程更改
git pull origin main

# 解决冲突后再次推送
git push origin main
```

---

## 🎯 本次提交建议

基于当前的修改，建议的提交信息：

```bash
git add .

git commit -m "Add: 可视化模块优化和项目文档完善

主要更改:
- 优化 5_1_Basemap.py，移除冗余方法，使用共享工具函数
- 修复 5_2_GCMT.py 的类型错误和 PyGMT 参数问题
- 修复 5_3_All_Stations.py 的 Matplotlib 弃用警告和 PyGMT 格式错误
- 创建可视化工具模块（5_Visualization/utils/file_utils.py）
- 添加项目文档：LICENSE, README, CHANGELOG, CONTRIBUTING等
- 添加投稿指南（PUBLICATION_GUIDE.md）
- 添加 Python 依赖列表（requirements.txt）"

git push origin main
```

---

## 📚 有用的 Git 命令

```bash
# 查看提交历史
git log

# 查看简洁的提交历史
git log --oneline

# 查看某个文件的修改历史
git log 文件名

# 查看远程仓库信息
git remote -v

# 查看分支
git branch

# 创建新分支
git branch 新分支名

# 切换分支
git checkout 分支名
```

---

**提示**: 每次提交前都要仔细检查 `git status`，确保只提交需要的文件！
