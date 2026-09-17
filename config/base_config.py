"""
EASTASIA-FWI 基础配置类
为所有模块提供通用的配置基础和研究区域定义

研究区域分三层（均为 lat/lon/depth 边界字典，单位 ° 与 km）：
- ``region``          总研究区域（study）：台站/事件查询、底图、模型库询读的最大范围
- ``model_regions``   单个模型的原始覆盖范围（来自 1_5 标准化时写出的 metadata）
- ``common_region``   核心三模型（FWEA23 ∩ EARA2024 ∩ SinoScope1.0）的交集，
                      即 2_1 “standardized” 公平对比区、正演/波形评估的事件-台站筛选区
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import re
import yaml
import json
import logging
from datetime import datetime

RegionDict = Dict[str, float]

_REGION_KEYS = ('lat_min', 'lat_max', 'lon_min', 'lon_max', 'depth_min', 'depth_max')
_YEAR_PREFIX = re.compile(r'^\d{4}_')


def strip_model_year(name: str) -> str:
    """'2024_FWEA23' → 'FWEA23'；无年份前缀原样返回。"""
    return _YEAR_PREFIX.sub('', name)


class BaseConfig:
    """项目基础配置类 - 包含所有模块共享的基础参数"""

    # 核心三模型：它们的交集定义 common_region
    CORE_MODELS: Sequence[str] = ('2024_FWEA23', '2024_EARA2024', '2022_SinoScope1.0')

    # 核心模型原始覆盖范围（与 data/models/metadata/<model>_metadata.json 一致）
    _DEFAULT_MODEL_REGIONS: Dict[str, RegionDict] = {
        '2024_FWEA23': {
            'lat_min': -5.0, 'lat_max': 55.0, 'lon_min': 65.0, 'lon_max': 150.0,
            'depth_min': 0.0, 'depth_max': 1000.0,
        },
        '2024_EARA2024': {
            'lat_min': 10.0, 'lat_max': 60.0, 'lon_min': 80.0, 'lon_max': 160.0,
            'depth_min': 0.0, 'depth_max': 1000.0,
        },
        '2022_SinoScope1.0': {
            'lat_min': -10.0, 'lat_max': 58.0, 'lon_min': 50.0, 'lon_max': 165.0,
            'depth_min': 0.0, 'depth_max': 2800.0,
        },
    }

    def __init__(self, config_file: Optional[Path] = None):
        """
        初始化基础配置

        Args:
            config_file: 配置文件路径，支持 .yaml 和 .json 格式
        """
        # 项目基础路径
        self.project_root = Path(__file__).parent.parent
        self.config_file = config_file

        # 总研究区域（study）
        self.region: Dict[str, object] = {
            'lat_min': -10.0,
            'lat_max': 60.0,
            'lon_min': 60.0,
            'lon_max': 150.0,
            'depth_min': 0.0,  # km
            'depth_max': 1000.0,  # km
            'name': 'EastAsia'
        }

        # 单模型覆盖范围 + 核心三模型交集
        self.core_models: List[str] = list(self.CORE_MODELS)
        self.model_regions: Dict[str, RegionDict] = {
            k: dict(v) for k, v in self._DEFAULT_MODEL_REGIONS.items()
        }
        self.common_region: RegionDict = self.compute_common_region(self.core_models)

        # 标准化目录结构
        self.dirs = self._setup_directories()

        # 确保目录存在
        self._ensure_directories()

        # 1_5 实际网格求交的结果（如存在）优先于静态默认值
        self._load_common_region_metadata()

        # 加载外部配置文件（如果提供）
        if config_file and config_file.exists():
            self.load_from_file(config_file)
    
    def _setup_directories(self) -> Dict[str, Path]:
        """设置标准化目录结构"""
        return {
            'project_root': self.project_root,
            'data': self.project_root / 'data',
            'models': self.project_root / 'data' / 'models',
            'catalogs': self.project_root / 'data' / 'catalogs', 
            'stations': self.project_root / 'data' / 'stations',
            'events': self.project_root / 'data' / 'events',
            'results': self.project_root / 'results',
            'figures': self.project_root / 'figures',
            'logs': self.project_root / 'logs',
            'temp': self.project_root / 'temp',
            'config': self.project_root / 'config',
            'specfem3d_globe': self.project_root / 'specfem3d_globe',  # SPECFEM3D 安装目录（需手动 git clone）
            'simulations': self.project_root / 'output' / 'simulations'  # 正演模拟输出
        }
    
    def _ensure_directories(self):
        """确保所有目录存在"""
        skip_dirs = {'project_root', 'specfem3d_globe'}  # specfem3d_globe 需手动 git clone
        for dir_name, dir_path in self.dirs.items():
            if dir_name not in skip_dirs:
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_from_file(self, config_file: Path):
        """从文件加载配置（覆盖默认值）"""
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        elif config_file.suffix.lower() == '.json':
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_file.suffix}")
        
        # 更新配置
        if 'region' in config_data:
            self.region.update(config_data['region'])
        if 'core_models' in config_data:
            self.core_models = list(config_data['core_models'])
        if 'model_regions' in config_data:
            for name, bounds in config_data['model_regions'].items():
                self.model_regions.setdefault(name, {}).update(bounds)
        if 'common_region' in config_data:
            self.common_region.update(config_data['common_region'])
        elif 'core_models' in config_data or 'model_regions' in config_data:
            self.common_region = self.compute_common_region(self.core_models)
    
    def save_to_file(self, config_file: Path):
        """保存配置到文件"""
        config_dict = {
            'region': self.region,
            'core_models': self.core_models,
            'model_regions': self.model_regions,
            'common_region': self.common_region,
            'dirs': {k: str(v) for k, v in self.dirs.items()}
        }
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        elif config_file.suffix.lower() == '.json':
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    # ------------------------------------------------------------------
    # 研究区域：study / common / 单模型
    # ------------------------------------------------------------------

    def _load_common_region_metadata(self) -> None:
        """若 1_5 已写出 valid_standardized_region.json 且模型集与 core_models 一致，用实际网格交集覆盖默认值。"""
        path = self.dirs['models'] / 'metadata' / 'valid_standardized_region.json'
        if not path.is_file():
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        included = raw.get('models_included')
        if included is not None and set(included) != set(self.core_models):
            return
        try:
            self.common_region = {
                'lat_min': round(float(raw['lat'][0]), 6), 'lat_max': round(float(raw['lat'][1]), 6),
                'lon_min': round(float(raw['lon'][0]), 6), 'lon_max': round(float(raw['lon'][1]), 6),
                'depth_min': round(float(raw['depth'][0]), 6), 'depth_max': round(float(raw['depth'][1]), 6),
            }
        except (KeyError, IndexError, TypeError, ValueError):
            return

    def resolve_model_name(self, name: str) -> str:
        """
        将短名（'FWEA23' / 'SinoScope' / 'EARA2024'）解析为 model_regions 中的完整模型名。

        Raises:
            KeyError: 无法唯一匹配。
        """
        if name in self.model_regions:
            return name
        stem = strip_model_year(name).lower()
        hits = [
            k for k in self.model_regions
            if strip_model_year(k).lower().startswith(stem) or stem.startswith(strip_model_year(k).lower())
        ]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(
            f"无法解析模型名 '{name}'（候选: {hits or list(self.model_regions)}）"
        )

    def get_model_region(self, model_name: str) -> RegionDict:
        """
        单个模型的原始覆盖范围。

        优先使用 model_regions 内置/配置值；未登记的模型尝试读取
        data/models/metadata/<model>_metadata.json 的 lat_range/lon_range/depth_range 并缓存。

        Raises:
            KeyError: 模型未登记且 metadata 不存在。
        """
        try:
            return dict(self.model_regions[self.resolve_model_name(model_name)])
        except KeyError:
            pass
        meta = self.dirs['models'] / 'metadata' / f'{model_name}_metadata.json'
        if not meta.is_file():
            raise KeyError(f"模型 '{model_name}' 未在 model_regions 登记，且找不到 {meta}")
        with open(meta, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        try:
            bounds: RegionDict = {
                'lat_min': float(raw['lat_range'][0]), 'lat_max': float(raw['lat_range'][1]),
                'lon_min': float(raw['lon_range'][0]), 'lon_max': float(raw['lon_range'][1]),
                'depth_min': float(raw['depth_range'][0]), 'depth_max': float(raw['depth_range'][1]),
            }
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise KeyError(f"{meta} 缺少 lat_range/lon_range/depth_range: {exc}") from exc
        self.model_regions[model_name] = bounds
        return dict(bounds)

    def compute_common_region(self, model_names: Iterable[str]) -> RegionDict:
        """
        多个模型覆盖范围的交集（lat/lon/depth 均取交）。

        Raises:
            ValueError: 模型列表为空或交集为空。
        """
        names = list(model_names)
        if not names:
            raise ValueError('compute_common_region: 模型列表为空')
        regs = [self.get_model_region(n) for n in names]
        common: RegionDict = {
            'lat_min': max(r['lat_min'] for r in regs),
            'lat_max': min(r['lat_max'] for r in regs),
            'lon_min': max(r['lon_min'] for r in regs),
            'lon_max': min(r['lon_max'] for r in regs),
            'depth_min': max(r['depth_min'] for r in regs),
            'depth_max': min(r['depth_max'] for r in regs),
        }
        if (common['lat_min'] >= common['lat_max'] or common['lon_min'] >= common['lon_max']
                or common['depth_min'] >= common['depth_max']):
            raise ValueError(f'模型 {names} 无有效公共范围: {common}')
        return common

    def get_region_bounds(self, kind: str = 'study') -> RegionDict:
        """
        获取研究区域边界。

        Args:
            kind: 'study'（总研究区域，默认）、'common'（核心三模型交集），
                  或模型名（完整名或短名，如 '2024_FWEA23' / 'FWEA23'）。

        Returns:
            含 lat_min/lat_max/lon_min/lon_max/depth_min/depth_max 的字典。
        """
        if kind == 'study':
            src = self.region
        elif kind == 'common':
            src = self.common_region
        else:
            src = self.get_model_region(kind)
        return {k: float(src[k]) for k in _REGION_KEYS}

    def get_gmt_region(self, kind: str = 'study') -> List[float]:
        """PyGMT/Cartopy 用 [lon_min, lon_max, lat_min, lat_max]；kind 同 get_region_bounds。"""
        b = self.get_region_bounds(kind)
        return [b['lon_min'], b['lon_max'], b['lat_min'], b['lat_max']]

    def is_in_region(self, lat: float, lon: float, depth: float = 0.0, kind: str = 'study') -> bool:
        """检查坐标是否在指定研究区域内；kind 同 get_region_bounds。"""
        b = self.get_region_bounds(kind)
        return (
            b['lat_min'] <= lat <= b['lat_max'] and
            b['lon_min'] <= lon <= b['lon_max'] and
            b['depth_min'] <= depth <= b['depth_max']
        )
    
    def get_module_dir(self, module_name: str) -> Path:
        """获取模块专用目录"""
        module_dir = self.dirs['results'] / module_name
        module_dir.mkdir(parents=True, exist_ok=True)
        return module_dir
    
    def get_log_file(self, module_name: str) -> Path:
        """获取模块日志文件路径"""
        log_file = self.dirs['logs'] / f"{module_name}_{datetime.now().strftime('%Y%m%d')}.log"
        return log_file
    
    def setup_logger(self, name: str, level: str = 'INFO') -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(name)
        
        # 避免重复添加handler
        if logger.handlers:
            return logger
            
        logger.setLevel(getattr(logging, level.upper()))
        
        # 文件处理器
        log_file = self.get_log_file(name.split('.')[-1])
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def print_summary(self):
        """打印配置摘要"""
        def _fmt(b: RegionDict) -> str:
            return (f"lat {b['lat_min']:g}°~{b['lat_max']:g}°, lon {b['lon_min']:g}°~{b['lon_max']:g}°, "
                    f"depth {b['depth_min']:g}~{b['depth_max']:g} km")

        print(f"\n{'='*60}")
        print(f"EASTASIA-FWI 基础配置")
        print(f"{'='*60}")
        print(f"📁 项目根目录: {self.project_root}")
        print(f"🌍 总研究区域 [{self.region['name']}]: {_fmt(self.get_region_bounds('study'))}")
        print(f"🔗 核心三模型交集 [common]:   {_fmt(self.common_region)}")
        for name in self.core_models:
            print(f"   • {name:<20s} {_fmt(self.get_model_region(name))}")
        print(f"{'='*60}\n")


# 配置验证工具类
class ConfigValidator:
    """配置验证工具"""
    
    @staticmethod
    def validate_region(region: Dict[str, float]) -> list[str]:
        """验证区域配置"""
        errors = []
        
        if not (-90 <= region.get('lat_min', 0) <= 90):
            errors.append("lat_min 必须在 -90° 到 90° 之间")
        if not (-90 <= region.get('lat_max', 0) <= 90):
            errors.append("lat_max 必须在 -90° 到 90° 之间")
        if not (-180 <= region.get('lon_min', 0) <= 180):
            errors.append("lon_min 必须在 -180° 到 180° 之间")
        if not (-180 <= region.get('lon_max', 0) <= 180):
            errors.append("lon_max 必须在 -180° 到 180° 之间")
        
        if region.get('lat_min', 0) >= region.get('lat_max', 0):
            errors.append("lat_min 必须小于 lat_max")
        if region.get('lon_min', 0) >= region.get('lon_max', 0):
            errors.append("lon_min 必须小于 lon_max")
        if region.get('depth_max', 0) <= region.get('depth_min', 0):
            errors.append("depth_max 必须大于 depth_min")
            
        return errors


if __name__ == "__main__":
    # 测试基础配置
    config = BaseConfig()
    config.print_summary()
    
    # 测试区域验证
    validator = ConfigValidator()
    errors = validator.validate_region(config.region)
    errors += [f"[common] {e}" for e in validator.validate_region(config.common_region)]
    for name in config.core_models:
        errors += [f"[{name}] {e}" for e in validator.validate_region(config.get_model_region(name))]
    
    if errors:
        print("区域配置错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ 区域配置验证通过")