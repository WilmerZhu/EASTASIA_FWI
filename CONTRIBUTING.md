# Contributing to EASTASIA-FWI

感谢您对 EASTASIA-FWI 项目的关注！我们欢迎所有形式的贡献。

## 如何贡献

### 报告问题

如果您发现了 bug 或有功能建议，请通过 GitHub Issues 提交。

### 提交代码

1. **Fork 项目** - 在 GitHub 上 Fork 本项目
2. **创建分支** - 从 `main` 分支创建新分支
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **编写代码** - 遵循项目代码规范
4. **提交更改** - 使用清晰的提交信息
   ```bash
   git commit -m "Add: 描述你的更改"
   ```
5. **推送分支** - 推送到你的 Fork
   ```bash
   git push origin feature/your-feature-name
   ```
6. **创建 Pull Request** - 在 GitHub 上创建 PR

## 代码规范

### Python 代码风格

- 遵循 PEP 8 规范
- 使用类型提示（Type Hints）
- 函数和类必须有文档字符串（docstring）
- 使用中文注释和文档字符串

### 命名规范

- **模块文件**: `模块号_功能描述.py`（如 `5_1_Basemap.py`）
- **类名**: 大驼峰命名（如 `EastAsiaBasemap`）
- **函数名**: 小写下划线（如 `plot_comprehensive_basemap`）
- **常量**: 全大写下划线（如 `DEFAULT_REGION`）

### 代码结构

每个模块应包含：

- 模块文档字符串（文件头部）
- 配置类（如 `ModuleNameConfig`）
- 主功能类（如 `ModuleName`）
- `main()` 函数（用于直接运行）

### 示例代码结构

```python
"""
EASTASIA-FWI 模块名称
======================

功能描述:
- 功能1
- 功能2

作者: EASTASIA-FWI Team
日期: 2025-xx-xx
版本: vX.X
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import BaseConfig


class ModuleConfig:
    """模块配置类"""
    def __init__(self):
        self.param1 = {...}


class ModuleName:
    """主功能类"""
    def __init__(self):
        self.base_config = BaseConfig()
        self.config = ModuleConfig()
        self.logger = self.base_config.setup_logger(...)

    def public_method(self):
        """公开方法"""
        pass


def main():
    """主函数"""
    pass


if __name__ == "__main__":
    main()
```

## 测试

- 新功能应包含基本测试
- 确保代码通过 linter 检查（无错误）
- 测试主要功能是否正常工作

## 文档

- 更新相关文档（README.md、CHANGELOG.md）
- 添加代码注释和文档字符串
- 更新 API 文档（如有）

## 提交信息规范

提交信息应清晰描述更改内容：

- `Add: 添加新功能`
- `Fix: 修复 bug`
- `Update: 更新功能`
- `Refactor: 代码重构`
- `Docs: 文档更新`
- `Style: 代码格式调整`

## 问题反馈

如有任何问题，请通过以下方式联系：

- GitHub Issues
- 项目维护者邮箱

再次感谢您的贡献！🎉
