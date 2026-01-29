# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 项目初始版本
- 数据准备模块（1_Data_preparation）
  - 台站查询与筛选（1_1_Query_stations.py）
  - GCMT事件处理（1_2_Process_GCMT_catalogs.py）
  - 波形数据下载（1_3_Download_waveforms.py）
- 可视化模块（5_Visualization）
  - 基础地图绘制（5_1_Basemap.py）
  - GCMT事件可视化（5_2_GCMT.py）
  - 台站分布可视化（5_3_All_Stations.py）
- 配置管理模块（config/base_config.py）
- Conda 环境配置文件（environment.yml）
- Python 依赖列表（requirements.txt）
- 项目文档（README.md）

### Changed

- 优化了代码结构，每个模块独立维护 utils 目录
- 统一了文件命名规范

### Fixed

- 修复了 GCMT 命名大小写问题
- 修复了类型注解错误
- 修复了 PyGMT resolution 参数问题

## [1.0.0-alpha] - 2025-01-29

### Added

- 初始版本发布
- 基础功能模块

---

## 版本说明

- **Unreleased**: 开发中的功能
- **1.0.0-alpha**: Alpha 版本，功能基本完成但可能不稳定

## 版本号规则

- **主版本号**: 不兼容的 API 修改
- **次版本号**: 向下兼容的功能性新增
- **修订号**: 向下兼容的问题修正
