"""统计研究区内地质省类别数，作为地壳 K 的地质先验（临时脚本）"""
import warnings
from collections import defaultdict

import shapefile  # pyshp
from shapely.geometry import box, shape

warnings.filterwarnings("ignore")

SHP = ("5_Visualization/data_EastAsia/global_tectonics-main/"
       "plates&provinces/shp/global_gprv.shp")
REGION = box(80, 10, 150, 55)          # 三模型公共范围

sf = shapefile.Reader(SHP)
fields = [f[0] for f in sf.fields[1:]]
print(f"字段: {fields}\n")

area_by = {c: defaultdict(float) for c in ("prov_type", "prov_group", "lastorogen")}
n_poly = 0
n_missing = {c: 0 for c in area_by}

for rec in sf.iterShapeRecords():
    try:
        geom = shape(rec.shape.__geo_interface__)
    except Exception:
        continue
    if not geom.is_valid:
        geom = geom.buffer(0)
    if not geom.intersects(REGION):
        continue
    clipped = geom.intersection(REGION)
    if clipped.is_empty:
        continue
    n_poly += 1
    a = clipped.area
    d = rec.record.as_dict()
    for col in area_by:
        if col not in fields:
            continue
        v = d.get(col)
        if v is None or str(v).strip() in ("", "None", "nan"):
            n_missing[col] += 1
            continue
        area_by[col][str(v).strip()] += a

print(f"研究区内地质省多边形: {n_poly} 个\n")
for col, dd in area_by.items():
    if not dd:
        continue
    tot = sum(dd.values())
    items = sorted(dd.items(), key=lambda kv: -kv[1])
    n_1pct = sum(1 for _, v in items if v / tot >= 0.01)
    print(f"── {col} ──  种类数={len(items)}  面积占比≥1%的种类={n_1pct}  "
          f"缺失多边形={n_missing[col]}/{n_poly}")
    for k, v in items[:12]:
        print(f"     {k[:44]:<46}{v / tot:>7.1%}")
    print()
