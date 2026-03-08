# EASTASIA-FWI Web 展示界面

纯静态 HTML 展示网站，**无需任何依赖**，可直接在浏览器中打开或通过简单 HTTP 服务访问。

## 自动生成（推荐）

**无需手动改 HTML**。新增或删除 `figures/` 中的图表后，运行：

```bash
python web/generate.py
```

脚本会扫描 `figures/` 目录并重新生成 `index.html`。

## 功能

- **项目概览**: 研究区域、技术栈、模块说明
- **基础地图与数据**: 底图、GCMT 事件、台站分布
- **速度模型对比**: 1D 对比、水平切片（可调深度）、垂直剖面、各向异性
- **模型相似性**: Vs/Vp 相似性热力图与空间分布

## 使用方式

### 方式 1：直接打开（推荐本地预览）

```bash
# 在项目根目录，用浏览器打开
open web/index.html
# 或
xdg-open web/index.html   # Linux
```

> 注意：直接打开时，部分浏览器可能因安全策略限制图片加载。若图片不显示，请用方式 2。

### 方式 2：部署到 GitHub Pages

将 `web/` 目录和 `figures/` 目录部署到 GitHub Pages，即可在线访问。

## 目录结构

```
web/
├── index.html          # 静态展示页面（单文件，含 CSS/JS）
├── .streamlit/         # Streamlit 配置（若使用 app.py）
├── app.py              # Streamlit 应用（可选，需 pip install streamlit）
└── README.md           # 本文件
```

## 图片路径说明

页面中的图片使用相对路径 `../figures/`，需确保从项目根目录启动 HTTP 服务，或 `figures/` 与 `web/` 保持正确的相对位置。
