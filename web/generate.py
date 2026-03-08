"""
生成 web/index.html
================================================================
扫描 figures/ 目录，自动生成静态展示页面。
新增或删除图表后，运行本脚本即可更新网站。

用法: python web/generate.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES = PROJECT_ROOT / "figures"
OUTPUT = Path(__file__).parent / "index.html"


def scan_figures() -> dict:
    """扫描 figures 目录，按分类整理"""
    result = {
        "basemap": [],
        "gcmt": [],
        "station": [],
        "model_compare": [],
        "huang2024": [],
        "model_similarity": [],
    }
    if not FIGURES.exists():
        return result

    IMG_PREFIX = "../figures/"
    for f in FIGURES.iterdir():
        if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            rel = f.relative_to(FIGURES)
            if rel.name.startswith("5-1"):
                result["basemap"].append((IMG_PREFIX + str(rel), rel.stem))
            elif rel.name.startswith("5-2"):
                result["gcmt"].append((IMG_PREFIX + str(rel), rel.stem))
            elif rel.name.startswith("5-3"):
                result["station"].append((IMG_PREFIX + str(rel), rel.stem))

    mc_dir = FIGURES / "model_compare"
    if mc_dir.exists():
        for f in sorted(mc_dir.iterdir()):
            if f.suffix.lower() in (".png", ".jpg"):
                result["model_compare"].append((IMG_PREFIX + "model_compare/" + f.name, f.stem))

    huang_dir = FIGURES / "Huang2024_reproduction"
    if huang_dir.exists():
        for f in sorted(huang_dir.iterdir()):
            if f.suffix.lower() in (".png", ".jpg"):
                result["huang2024"].append((IMG_PREFIX + "Huang2024_reproduction/" + f.name, f.stem))

    sim_dir = FIGURES / "model_similarity"
    if sim_dir.exists():
        for f in sorted(sim_dir.iterdir()):
            if f.suffix.lower() in (".png", ".jpg"):
                result["model_similarity"].append((IMG_PREFIX + "model_similarity/" + f.name, f.stem))

    return result


def get_depths() -> list:
    """从 model_compare 中提取深度列表"""
    mc = FIGURES / "model_compare"
    if not mc.exists():
        return [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    depths = set()
    for f in mc.iterdir():
        if "absolute_vs_" in f.name or "difference_vs_" in f.name:
            try:
                d = int(f.stem.split("_")[-1].replace("km", ""))
                depths.add(d)
            except ValueError:
                pass
    return sorted(depths) if depths else [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900]


def get_profile_nums() -> list:
    """从 model_compare 中提取剖面编号"""
    mc = FIGURES / "model_compare"
    if not mc.exists():
        return list(range(1, 13))
    nums = set()
    for f in mc.iterdir():
        if "vertical_profile_" in f.name:
            try:
                n = int(f.stem.split("_")[-1])
                nums.add(n)
            except ValueError:
                pass
    return sorted(nums) if nums else list(range(1, 13))


def generate_html(figs: dict) -> str:
    depths = get_depths()
    depth_opts = "".join(f'<option value="{d}"{" selected" if d==100 else ""}>{d}</option>' for d in depths)
    profile_nums = get_profile_nums()
    profile_opts = "".join(f'<option value="{n}"{" selected" if n==1 else ""}>{n}</option>' for n in profile_nums)

    basemap_imgs = "".join(
        f'<figure class="figure"><img src="{p}" alt="{t}" onerror="this.parentElement.style.display=\'none\'"><figcaption>{t}</figcaption></figure>'
        for p, t in figs["basemap"]
    )
    gcmt_imgs = "".join(
        f'<figure class="figure"><img src="{p}" alt="{t}"><figcaption>{t}</figcaption></figure>'
        for p, t in figs["gcmt"]
    )
    station_imgs = "".join(
        f'<figure class="figure"><img src="{p}" alt="{t}"><figcaption>{t}</figcaption></figure>'
        for p, t in figs["station"]
    )

    # 模型对比：提取关键图
    mc = figs["model_compare"]
    mc_1d = next((p for p, t in mc if "velocity_comparison" in t), "../figures/model_compare/2-1_velocity_comparison.png")
    mc_abs = [p for p, t in mc if "absolute_vs_" in t]
    mc_diff = [p for p, t in mc if "difference_vs_" in t]
    mc_profiles = [(p, t) for p, t in mc if "profile_" in t and "vertical" not in t]
    mc_vertical = [(p, t) for p, t in mc if "vertical_profile_" in t]
    mc_ani = [(p, t) for p, t in mc if "anisotropy" in t]

    huang_imgs = "".join(
        f'<figure class="figure"><img src="{p}" alt="{t}"><figcaption>{t}</figcaption></figure>'
        for p, t in figs["huang2024"]
    )

    sim_vs = [(p, t) for p, t in figs["model_similarity"] if "_vs" in t or "vs.png" in p]
    sim_vp = [(p, t) for p, t in figs["model_similarity"] if "_vp" in t or "vp.png" in p]

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EASTASIA-FWI | 东亚全波形成像</title>
  <style>
    :root {{ --primary: #1a5276; --primary-light: #2980b9; --bg: #f8f9fa; --card-bg: #fff; --text: #2c3e50; --text-muted: #7f8c8d; --border: #e0e0e0; }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }}
    .header {{ background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%); color: #fff; padding: 1.5rem 2rem; }}
    .header h1 {{ font-size: 1.5rem; font-weight: 600; }}
    .header p {{ opacity: 0.9; font-size: 0.9rem; margin-top: 0.25rem; }}
    .nav {{ display: flex; gap: 0; background: var(--card-bg); border-bottom: 1px solid var(--border); padding: 0 2rem; overflow-x: auto; }}
    .nav a {{ padding: 1rem 1.25rem; text-decoration: none; color: var(--text); font-weight: 500; border-bottom: 3px solid transparent; white-space: nowrap; }}
    .nav a:hover {{ color: var(--primary); }}
    .nav a.active {{ color: var(--primary); border-bottom-color: var(--primary); }}
    .main {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
    .section {{ display: none; }}
    .section.active {{ display: block; }}
    .section h2 {{ color: var(--primary); margin-bottom: 1rem; font-size: 1.25rem; }}
    .section h3 {{ margin: 1.5rem 0 0.75rem; font-size: 1.1rem; }}
    .section p {{ margin-bottom: 1rem; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
    .metric {{ background: var(--card-bg); padding: 1rem; border-radius: 8px; border: 1px solid var(--border); text-align: center; }}
    .metric .value {{ font-size: 1.1rem; font-weight: 600; color: var(--primary); }}
    .metric .label {{ font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem; }}
    .figure img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border); margin: 1rem 0; }}
    .figure figcaption {{ font-size: 0.85rem; color: var(--text-muted); margin-top: 0.5rem; }}
    .tabs {{ display: flex; gap: 0; margin-bottom: 1rem; border-bottom: 1px solid var(--border); }}
    .tab-btn {{ padding: 0.75rem 1rem; border: none; background: none; cursor: pointer; font-size: 0.9rem; color: var(--text-muted); margin-bottom: -1px; }}
    .tab-btn:hover {{ color: var(--primary); }}
    .tab-btn.active {{ color: var(--primary); border-bottom: 2px solid var(--primary); font-weight: 500; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .depth-selector {{ margin: 1rem 0; }}
    .depth-selector label {{ margin-right: 0.5rem; }}
    .depth-selector select {{ padding: 0.5rem 1rem; border: 1px solid var(--border); border-radius: 6px; font-size: 0.9rem; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
    th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ background: var(--bg); font-weight: 600; }}
    ul {{ margin: 0.5rem 0 1rem 1.5rem; }}
    .footer {{ text-align: center; padding: 2rem; color: var(--text-muted); font-size: 0.85rem; }}
    .gen-note {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; }}
  </style>
</head>
<body>
  <header class="header">
    <h1>🌏 EASTASIA-FWI</h1>
    <p>东亚地区全波形成像项目 · 模型空间智能分析 · 全波形反演</p>
  </header>
  <nav class="nav">
    <a href="#" data-section="overview" class="active">项目概览</a>
    <a href="#" data-section="basemap">基础地图与数据</a>
    <a href="#" data-section="model-compare">速度模型对比</a>
    <a href="#" data-section="huang2024">Huang2024 复现</a>
    <a href="#" data-section="similarity">模型相似性</a>
  </nav>
  <main class="main">
    <section id="overview" class="section active">
      <h2>项目概览</h2>
      <p><strong>EASTASIA-FWI</strong> 通过模型空间的智能分析（聚类、相似性指数、机器学习）和数据空间的波形评估，建立东亚地区三维地球结构初始模型，结合大规模波形数据的高精度全波形反演，得到新一代标准东亚岩石圈模型 <strong>SEAM 1.0</strong>。</p>
      <div class="metrics">
        <div class="metric"><span class="value">-15° ~ 60°</span><div class="label">纬度</div></div>
        <div class="metric"><span class="value">60° ~ 170°</span><div class="label">经度</div></div>
        <div class="metric"><span class="value">0 ~ 1000 km</span><div class="label">深度</div></div>
      </div>
      <h3>核心技术栈</h3>
      <ul>
        <li><strong>SPECFEM3D Globe</strong>: 地震波正演/反演</li>
        <li><strong>ObsPy/SAC</strong>: 地震学数据处理</li>
        <li><strong>scikit-learn</strong>: 聚类与机器学习</li>
        <li><strong>PyGMT/Cartopy</strong>: 地球物理可视化</li>
      </ul>
      <h3>项目模块</h3>
      <table>
        <tr><th>模块</th><th>功能</th></tr>
        <tr><td>1_Data_preparation</td><td>台站查询、GCMT 处理、波形下载与预处理、速度模型标准化</td></tr>
        <tr><td>2_Model_space_analysis</td><td>模型对比、Huang2024 复现、相似性分析</td></tr>
        <tr><td>3_Data_space_simulation</td><td>SPECFEM3D 正演设置</td></tr>
        <tr><td>4_Fusion</td><td>Voting Map、BMA 模型融合</td></tr>
        <tr><td>5_Visualization</td><td>地图、GCMT、台站分布可视化</td></tr>
      </table>
    </section>
    <section id="basemap" class="section">
      <h2>基础地图与数据分布</h2>
      <div class="tabs">
        <button class="tab-btn active" data-tab="basemap">综合底图</button>
        <button class="tab-btn" data-tab="gcmt">GCMT 事件</button>
        <button class="tab-btn" data-tab="station">台站网络</button>
      </div>
      <div id="basemap-panel" class="tab-panel active"><h3>东亚研究区域</h3>{basemap_imgs}</div>
      <div id="gcmt-panel" class="tab-panel"><h3>GCMT 震源机制与事件</h3>{gcmt_imgs}</div>
      <div id="station-panel" class="tab-panel"><h3>台站网络分布</h3>{station_imgs}</div>
    </section>
    <section id="model-compare" class="section">
      <h2>速度模型对比</h2>
      <div class="tabs">
        <button class="tab-btn active" data-tab="mc-1d">1D 速度对比</button>
        <button class="tab-btn" data-tab="mc-abs">水平切片 (Vs)</button>
        <button class="tab-btn" data-tab="mc-diff">水平切片 (差异)</button>
        <button class="tab-btn" data-tab="mc-profile">垂直剖面</button>
        <button class="tab-btn" data-tab="mc-ani">各向异性</button>
      </div>
      <div id="mc-1d-panel" class="tab-panel active">
        <figure class="figure"><img src="{mc_1d}" alt="1D速度对比"><figcaption>2-1 速度模型 1D 对比</figcaption></figure>
      </div>
      <div id="mc-abs-panel" class="tab-panel">
        <div class="depth-selector"><label>深度 (km):</label><select id="depth-abs">{depth_opts}</select></div>
        <figure class="figure"><img id="img-abs" src="{mc_abs[0] if mc_abs else '../figures/model_compare/2-1_absolute_vs_100km.png'}" alt="Vs"><figcaption>2-1 Vs 绝对速度水平切片</figcaption></figure>
      </div>
      <div id="mc-diff-panel" class="tab-panel">
        <div class="depth-selector"><label>深度 (km):</label><select id="depth-diff">{depth_opts}</select></div>
        <figure class="figure"><img id="img-diff" src="{mc_diff[0] if mc_diff else '../figures/model_compare/2-1_difference_vs_100km.png'}" alt="Vs差异"><figcaption>2-1 Vs 差异水平切片</figcaption></figure>
      </div>
      <div id="mc-profile-panel" class="tab-panel">
        <h3>切片位置</h3>
        <figure class="figure"><img src="figures/model_compare/2-1_slice_locations.png" alt="切片位置"><figcaption>2-1 剖面切片位置</figcaption></figure>
        <h3>单模型垂直剖面</h3>
        {"".join(f'<figure class="figure"><img src="{p}" alt="{t}"><figcaption>{t}</figcaption></figure>' for p, t in mc_profiles[:5])}
        <h3>多模型垂直剖面</h3>
        <div class="depth-selector"><label>剖面编号:</label><select id="profile-num">{profile_opts}</select></div>
        <figure class="figure"><img id="img-profile" src="../figures/model_compare/2-1_vertical_profile_{profile_nums[0] if profile_nums else 1}.png" alt="垂直剖面"><figcaption>2-1 垂直剖面</figcaption></figure>
      </div>
      <div id="mc-ani-panel" class="tab-panel">
        {"".join(f'<figure class="figure"><img src="{p}" alt="{t}"><figcaption>{t}</figcaption></figure>' for p, t in mc_ani)}
      </div>
    </section>
    <section id="huang2024" class="section">
      <h2>Huang2024 CWSSIM 复现</h2>
      {huang_imgs or '<p>暂无图表，请先运行 2_2_Huang2024_reproduction.py</p>'}
    </section>
    <section id="similarity" class="section">
      <h2>模型相似性分析 (CW-SSIM)</h2>
      <div class="tabs">
        <button class="tab-btn active" data-tab="sim-vs">Vs 相似性</button>
        <button class="tab-btn" data-tab="sim-vp">Vp 相似性</button>
      </div>
      <div id="sim-vs-panel" class="tab-panel active">
        {"".join(f'<figure class="figure"><img src="{p}" alt="{t}"><figcaption>{t}</figcaption></figure>' for p, t in sim_vs)}
      </div>
      <div id="sim-vp-panel" class="tab-panel">
        {"".join(f'<figure class="figure"><img src="{p}" alt="{t}"><figcaption>{t}</figcaption></figure>' for p, t in sim_vp)}
      </div>
    </section>
  </main>
  <footer class="footer">
    EASTASIA-FWI · 东亚全波形成像项目 · 由 <code>web/generate.py</code> 自动生成
  </footer>
  <script>
    document.querySelectorAll('.nav a').forEach(a => {{
      a.addEventListener('click', e => {{
        e.preventDefault();
        const id = a.dataset.section;
        document.querySelectorAll('.nav a').forEach(x => x.classList.remove('active'));
        document.querySelectorAll('.section').forEach(x => x.classList.remove('active'));
        a.classList.add('active');
        document.getElementById(id).classList.add('active');
      }});
    }});
    function initTabs(container) {{
      container.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{
          const tab = btn.dataset.tab;
          const parent = btn.closest('.section');
          parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
          parent.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
          btn.classList.add('active');
          const panel = parent.querySelector('#' + tab + '-panel');
          if (panel) panel.classList.add('active');
        }});
      }});
    }}
    document.querySelectorAll('.section').forEach(s => initTabs(s));
    const abs = document.getElementById('depth-abs');
    const diff = document.getElementById('depth-diff');
    const profile = document.getElementById('profile-num');
    if (abs) abs.addEventListener('change', () => {{ document.getElementById('img-abs').src = '../figures/model_compare/2-1_absolute_vs_' + abs.value + 'km.png'; }});
    if (diff) diff.addEventListener('change', () => {{ document.getElementById('img-diff').src = '../figures/model_compare/2-1_difference_vs_' + diff.value + 'km.png'; }});
    if (profile) profile.addEventListener('change', () => {{ document.getElementById('img-profile').src = '../figures/model_compare/2-1_vertical_profile_' + profile.value + '.png'; }});
  </script>
</body>
</html>"""


def main():
    figs = scan_figures()
    mc = figs["model_compare"]
    html = generate_html(figs)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"✅ 已生成 {OUTPUT}")
    print(f"   发现: 底图 {len(figs['basemap'])} 张, GCMT {len(figs['gcmt'])} 张, 台站 {len(figs['station'])} 张")
    print(f"   模型对比 {len(mc)} 张, Huang2024 {len(figs['huang2024'])} 张, 相似性 {len(figs['model_similarity'])} 张")


if __name__ == "__main__":
    main()
