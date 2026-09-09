"""
EASTASIA-FWI 子区域模型提取
================================

功能描述:
- 从已处理的标准化速度模型中裁剪指定区域的子模型
- 保存为 NetCDF 和 CSV 格式

使用方法:
- 直接运行: python 1_5b_Extract_subregion.py
- 修改 SubregionConfig 中的参数来调整提取区域和模型列表

作者: EASTASIA-FWI Team
日期: 2026-04-15
版本: v1.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import BaseConfig

warnings.filterwarnings('ignore')


class SubregionConfig:
    """子区域提取配置"""

    def __init__(self):
        # ============ 提取区域 ============
        self.subregion = {
            'name': 'XinJiang',
            'lat_min': 35.0,
            'lat_max': 43.0,
            'lon_min': 75.0,
            'lon_max': 92.0,
        }

        # ============ 模型映射 ============
        # key: 源模型目录名, value: 输出名称
        self.model_map = {
            '2024_FWEA23': 'XinJiang_FWEA23',
            '2024_EARA2024': 'XinJiang_EARA2024',
            '2022_SinoScope1.0': 'XinJiang_SinoScope',
        }

        # ============ 输入/输出 ============
        # 使用 original.nc 作为源数据（保留完整参数）
        self.source_suffix = 'original'
        self.save_netcdf = True
        self.save_csv = True

        # ============ 日志 ============
        self.logging = {
            'level': 'INFO',
            'console_output': True,
        }


class SubregionExtractor:
    """子区域模型提取器"""

    def __init__(self, config: Optional[SubregionConfig] = None):
        # 1. 加载全局配置
        self.base_config = BaseConfig()

        # 2. 加载模块配置
        self.config = config or SubregionConfig()

        # 3. 路径设置
        self.processed_dir = self.base_config.dirs['models'] / 'processed'
        self.output_dir = self.base_config.dirs['models'] / 'processed'

        # 4. 设置日志
        self.logger = self.base_config.setup_logger(
            'EASTASIA-FWI.SubregionExtractor',
            self.config.logging['level']
        )

        self.logger.info("🎯 子区域模型提取器初始化完成")
        self._print_config_summary()

    def _print_config_summary(self):
        """打印配置摘要"""
        sr = self.config.subregion
        self.logger.info(f"  区域名称: {sr['name']}")
        self.logger.info(f"  纬度范围: {sr['lat_min']}° ~ {sr['lat_max']}°N")
        self.logger.info(f"  经度范围: {sr['lon_min']}° ~ {sr['lon_max']}°E")
        self.logger.info(f"  模型数量: {len(self.config.model_map)}")

    def extract_subregion(self, src_name: str, dst_name: str) -> Optional[xr.Dataset]:
        """
        从源模型中提取子区域

        Args:
            src_name: 源模型目录名 (如 '2024_FWEA23')
            dst_name: 输出名称 (如 'XinJiang_FWEA23')

        Returns:
            裁剪后的 xr.Dataset，失败返回 None
        """
        sr = self.config.subregion
        suffix = self.config.source_suffix
        nc_path = self.processed_dir / src_name / f"{src_name}_{suffix}.nc"

        if not nc_path.exists():
            self.logger.error(f"源文件不存在: {nc_path}")
            return None

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📂 读取模型: {src_name}")
        self.logger.info(f"   文件: {nc_path.name}")

        ds = xr.open_dataset(nc_path)

        # 原始范围
        lat_orig = (float(ds.latitude.min()), float(ds.latitude.max()))
        lon_orig = (float(ds.longitude.min()), float(ds.longitude.max()))
        self.logger.info(f"   原始纬度: {lat_orig[0]:.2f}° ~ {lat_orig[1]:.2f}°")
        self.logger.info(f"   原始经度: {lon_orig[0]:.2f}° ~ {lon_orig[1]:.2f}°")
        self.logger.info(f"   原始深度: {float(ds.depth.min()):.1f} ~ {float(ds.depth.max()):.1f} km")
        self.logger.info(f"   变量列表: {list(ds.data_vars)}")

        # 检查覆盖情况
        actual_lon_min = max(sr['lon_min'], lon_orig[0])
        actual_lon_max = min(sr['lon_max'], lon_orig[1])
        actual_lat_min = max(sr['lat_min'], lat_orig[0])
        actual_lat_max = min(sr['lat_max'], lat_orig[1])

        if actual_lon_min >= actual_lon_max or actual_lat_min >= actual_lat_max:
            self.logger.error(f"   ❌ 模型不覆盖目标区域，跳过")
            ds.close()
            return None

        if actual_lon_min > sr['lon_min'] or actual_lon_max < sr['lon_max'] or \
           actual_lat_min > sr['lat_min'] or actual_lat_max < sr['lat_max']:
            self.logger.warning(
                f"   ⚠️ 模型部分覆盖目标区域，实际范围: "
                f"lat [{actual_lat_min:.2f}, {actual_lat_max:.2f}], "
                f"lon [{actual_lon_min:.2f}, {actual_lon_max:.2f}]"
            )

        # 裁剪子区域（xarray sel + slice 自动取交集）
        ds_sub = ds.sel(
            latitude=slice(sr['lat_min'], sr['lat_max']),
            longitude=slice(sr['lon_min'], sr['lon_max']),
        )

        self.logger.info(f"   裁剪后纬度: {float(ds_sub.latitude.min()):.2f}° ~ {float(ds_sub.latitude.max()):.2f}°")
        self.logger.info(f"   裁剪后经度: {float(ds_sub.longitude.min()):.2f}° ~ {float(ds_sub.longitude.max()):.2f}°")
        self.logger.info(f"   裁剪后形状: lat={ds_sub.sizes['latitude']}, lon={ds_sub.sizes['longitude']}, depth={ds_sub.sizes['depth']}")

        ds.close()
        return ds_sub

    def save_subregion(self, ds_sub: xr.Dataset, dst_name: str):
        """
        保存子区域模型为 NetCDF 和 CSV

        Args:
            ds_sub: 裁剪后的 Dataset
            dst_name: 输出名称
        """
        out_dir = self.output_dir / dst_name
        out_dir.mkdir(parents=True, exist_ok=True)

        sr = self.config.subregion

        # 更新全局属性
        ds_sub.attrs.update({
            'title': f"Subregion velocity model: {dst_name}",
            'subregion_name': sr['name'],
            'subregion_lat_min': sr['lat_min'],
            'subregion_lat_max': sr['lat_max'],
            'subregion_lon_min': sr['lon_min'],
            'subregion_lon_max': sr['lon_max'],
            'extraction_date': datetime.now().isoformat(),
            'creator': 'EASTASIA-FWI SubregionExtractor',
        })

        # ======== 保存 NetCDF ========
        if self.config.save_netcdf:
            nc_file = out_dir / f"{dst_name}.nc"
            ds_sub.to_netcdf(nc_file, engine='netcdf4')
            self.logger.info(f"   ✅ NetCDF 已保存: {nc_file.name}")

        # ======== 保存 CSV ========
        if self.config.save_csv:
            csv_file = out_dir / f"{dst_name}.csv"
            df = ds_sub.to_dataframe().reset_index()
            # 排序: depth(慢) → latitude → longitude(快)
            df = df.sort_values(['depth', 'latitude', 'longitude']).reset_index(drop=True)
            # 去除全NaN行
            param_cols = [c for c in df.columns if c not in ('latitude', 'longitude', 'depth')]
            df = df.dropna(subset=param_cols, how='all')
            df.to_csv(csv_file, index=False, float_format='%.6f')
            self.logger.info(f"   ✅ CSV 已保存: {csv_file.name} ({len(df)} 行)")

        self.logger.info(f"   📁 输出目录: {out_dir}")

    def run(self) -> Dict[str, str]:
        """
        执行所有模型的子区域提取

        Returns:
            {模型名: 'success'|'failed'} 结果字典
        """
        results = {}

        for src_name, dst_name in self.config.model_map.items():
            try:
                ds_sub = self.extract_subregion(src_name, dst_name)
                if ds_sub is not None:
                    self.save_subregion(ds_sub, dst_name)
                    results[dst_name] = 'success'
                else:
                    results[dst_name] = 'failed'
            except Exception as e:
                self.logger.error(f"   ❌ 处理 {src_name} 失败: {e}")
                import traceback
                traceback.print_exc()
                results[dst_name] = 'failed'

        return results


def main():
    """主函数"""
    print("🎯 EASTASIA-FWI 子区域模型提取")
    print("=" * 60)

    try:
        extractor = SubregionExtractor()
        results = extractor.run()

        print(f"\n{'='*60}")
        print("✨ 处理完成！")
        print("=" * 60)

        success = [k for k, v in results.items() if v == 'success']
        failed = [k for k, v in results.items() if v == 'failed']

        if success:
            print(f"\n✅ 成功 ({len(success)}):")
            for name in success:
                print(f"   • {name}")

        if failed:
            print(f"\n❌ 失败 ({len(failed)}):")
            for name in failed:
                print(f"   • {name}")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断处理")
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)


if __name__ == "__main__":
    main()
