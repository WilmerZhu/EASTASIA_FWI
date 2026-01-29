"""
EASTASIA-FWI 基础配置类
为所有模块提供通用的配置基础和研究区域定义
"""
from pathlib import Path
from typing import Dict, Optional
import yaml
import json
import logging
from datetime import datetime

class BaseConfig:
    """项目基础配置类 - 包含所有模块共享的基础参数"""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        初始化基础配置
        
        Args:
            config_file: 配置文件路径，支持 .yaml 和 .json 格式
        """
        # 项目基础路径
        self.project_root = Path(__file__).parent.parent
        self.config_file = config_file
        
        # 共用的研究区域定义
        self.region = {
            'lat_min': -15.0,
            'lat_max': 60.0, 
            'lon_min': 60.0,
            'lon_max': 170.0,
            'depth_min': 0.0,  # km
            'depth_max': 1000.0,  # km
            'name': 'EastAsia'
        }
        
        # 标准化目录结构
        self.dirs = self._setup_directories()
        
        # 确保目录存在
        self._ensure_directories()
        
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
            'config': self.project_root / 'config'
        }
    
    def _ensure_directories(self):
        """确保所有目录存在"""
        for dir_name, dir_path in self.dirs.items():
            if dir_name != 'project_root':
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
    
    def save_to_file(self, config_file: Path):
        """保存配置到文件"""
        config_dict = {
            'region': self.region,
            'dirs': {k: str(v) for k, v in self.dirs.items()}
        }
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        elif config_file.suffix.lower() == '.json':
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def get_region_bounds(self) -> Dict[str, float]:
        """获取研究区域边界"""
        return {
            'lat_min': self.region['lat_min'],
            'lat_max': self.region['lat_max'],
            'lon_min': self.region['lon_min'],
            'lon_max': self.region['lon_max'],
            'depth_min': self.region['depth_min'],
            'depth_max': self.region['depth_max']
        }
    
    def is_in_region(self, lat: float, lon: float, depth: float = 0.0) -> bool:
        """检查坐标是否在研究区域内"""
        return (
            self.region['lat_min'] <= lat <= self.region['lat_max'] and
            self.region['lon_min'] <= lon <= self.region['lon_max'] and
            self.region['depth_min'] <= depth <= self.region['depth_max']
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
        print(f"\n{'='*60}")
        print(f"EASTASIA-FWI 基础配置")
        print(f"{'='*60}")
        print(f"📁 项目根目录: {self.project_root}")
        print(f"🌍 研究区域: {self.region['name']}")
        print(f"   纬度: {self.region['lat_min']}° ~ {self.region['lat_max']}°")
        print(f"   经度: {self.region['lon_min']}° ~ {self.region['lon_max']}°")
        print(f"   深度: {self.region['depth_min']} ~ {self.region['depth_max']} km")
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
    
    if errors:
        print("区域配置错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ 区域配置验证通过")