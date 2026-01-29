"""
可视化模块文件工具函数
提供文件名生成等功能
"""
import re
from pathlib import Path
from typing import Optional


def extract_module_prefix(file_path: Optional[Path] = None, default: str = "5-1") -> str:
    """
    从文件名自动提取模块前缀
    
    从文件名如 '5_1_Basemap.py' 提取 '5-1'
    
    Args:
        file_path: 文件路径（如果为None，使用调用者的__file__）
        default: 默认值（如果无法提取）
        
    Returns:
        模块前缀字符串，如 '5-1'
    """
    try:
        # 获取当前文件路径
        if file_path is None:
            import inspect
            frame = inspect.currentframe()
            if frame and frame.f_back:
                file_path = Path(frame.f_back.f_code.co_filename)
            else:
                return default
        
        # 提取文件名（不含扩展名）
        filename = file_path.stem  # 例如: '5_1_Basemap'
        
        # 使用正则表达式提取模块号（格式：数字_数字）
        match = re.match(r'^(\d+)_(\d+)_', filename)
        if match:
            module_num = match.group(1)
            sub_num = match.group(2)
            return f"{module_num}-{sub_num}"
        else:
            return default
    except Exception:
        return default


def generate_filename(module_prefix: str, description: str, extension: str = "jpg") -> str:
    """
    自动生成符合规范的图片文件名
    
    Args:
        module_prefix: 模块前缀（如 '5-1'）
        description: 图片描述（如 'comprehensive_basemap'）
        extension: 文件扩展名（默认 'jpg'）
        
    Returns:
        符合规范的完整文件名，如 '5-1_comprehensive_basemap.jpg'
    """
    return f"{module_prefix}_{description}.{extension}"
