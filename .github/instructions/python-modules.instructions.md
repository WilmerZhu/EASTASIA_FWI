---
description: "Use when creating or editing Python pipeline modules (1_Data_preparation, 2_Model_space_analysis, 3_Data_space_simulation, 4_Fusion, 5_Visualization, config, scripts). Covers module file header, BaseConfig + module Config two-layer pattern, class initialization order, logging, main() skeleton."
applyTo: "1_Data_preparation/**/*.py, 2_Model_space_analysis/**/*.py, 3_Data_space_simulation/**/*.py, 4_Fusion/**/*.py, 5_Visualization/**/*.py, config/**/*.py, scripts/**/*.py"
---
# Python 模块规范

## 文件头与导入

```python
"""
EASTASIA-FWI <模块名>
======================

功能:
- ...

科学原理:
- ...

版本: vX.Y   日期: YYYY-MM-DD
"""
import sys
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from config.base_config import BaseConfig  # noqa: E402
```

- 标准库 → 第三方 → 项目内，三段分开；不要 `import *`。
- `warnings.filterwarnings('ignore')` 只在确有噪音时使用，并注明来源。

## 配置类

```python
class XxxConfig:
    """模块配置（放在模块文件内）"""

    def __init__(self) -> None:
        self.params: Dict[str, Any] = {...}
        self.visualization = {'dpi': 300, 'pdf_dpi': 300, 'figure_format': ['jpg', 'pdf']}
        self.logging = {'level': 'INFO', 'console_output': True}
```

- 区域边界、路径、深度范围一律取自 `BaseConfig`，不得在模块 Config 中重复定义。
- 可调科学参数（K、阈值、窗长等）集中在 Config，并写明单位与默认值来源。

## 主类初始化顺序

```python
class Xxx:
    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.base_config = BaseConfig()
        self.config = XxxConfig()
        self.output_dir = Path(output_dir) if output_dir else self.base_config.dirs['figures']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self.base_config.setup_logger('EASTASIA-FWI.Xxx', self.config.logging['level'])
        # kind: 'study'（默认）| 'common'（三模型交集）| 模型名（'FWEA23'）
        self.region = self.base_config.get_gmt_region('study')  # [lon_min, lon_max, lat_min, lat_max]
```

- 涉及公平对比 / 正演筛选的模块用 `'common'`；单模型裁剪、邀剪、STATIONS 用模型名；不要在 Config 中写 `[80, 150, 10, 55]` 这类字面量。

## 方法与类型

- 公开方法有 Google 风格 docstring（Args / Returns / Raises）；私有方法以 `_` 开头。
- 全部参数与返回值带类型注解；数组用 `np.ndarray`，DataFrame 用 `pd.DataFrame`。
- 数据容器用 `@dataclass`；不要用裸 dict 在多个方法间传递复杂结构。
- 用 `self.logger.info/warning/error`，不要新增 `print()`。
- I/O 边界（文件不存在、NetCDF 变量缺失、区域越界）做显式检查并抛出带上下文的异常；内部逻辑不要包一层 `try/except: pass`。

## NetCDF / 大数据

- 用 `xarray.open_dataset(..., chunks=...)` 或按深度分块处理，避免一次性 `.values` 物化整个模型。
- 深度/经纬度插值前先检查坐标单调性与单位（km vs m，g/cm³ vs kg/m³）。
- 长耗时循环支持断点续传：结果按事件/模型分片写出，重跑时跳过已有文件。

## `main()` 骨架

```python
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    try:
        proc = Xxx(output_dir=args.output_dir)
        proc.run()
    except KeyboardInterrupt:
        print('用户中断')
    except Exception as exc:
        logging.getLogger('EASTASIA-FWI').exception('处理失败: %s', exc)
        raise


if __name__ == '__main__':
    main()
```

- 可调参数通过 `argparse` 暴露（参考 `4_3_SDL_Fusion_v4.py` 的 `--k --stride --cc-threshold`），便于不改代码做实验。
- 新增或修改算法后，在 `tests/` 补对应的合成数据回归测试。
