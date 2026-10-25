"""
run_fullmap.py — full-map multiprocessed graph-heal + GCN for a city.

Processes the ENTIRE map as non-overlapping 512x512 tiles over an
autodetected multiprocessing pool, then assembles:
    - GeoJSON  (roads/bridges as LineStrings, GCN gatekeepers as Points, EPSG:32644)
    - HTML     (self-contained Leaflet map, reprojected to WGS84, with legend)
    - PNG      (downscaled full-extent preview)
    - report.json (aggregate metrics + wall-clock)

Note: tiles are healed independently => the image/GeoJSON are complete per
tile but edges are NOT stitched across tile boundaries into one connected graph.

Run in the `terrain-torch` env:
    conda run -n terrain-torch python tests/graph-tests/run_fullmap.py
      [--city hyderabad] [--tile-size 512] [--fidelity simplified|full]
      [--workers 0(autodetect)] [--out outputs/graph-output-tests]
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT and not os.path.exists(os.path.join(_ROOT, "config.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import config

# ─── worker globals (set via pool initializer) ───────────────────────────────
_W_GEO = None          # list of shapely geometries (projected CRS)
_W_SINDEX = None       # STRtree over _W_GEO
_W_MODEL = None        # loaded RoadGCN
_W_CHECKPOINT = None
_W_CPATH = None


def _init_worker(geos, checkpoint, cpath):
    global _W_GEO, _W_SINDEX, _W_MODEL, _W_CHECKPOINT, _W_CPATH
    _W_GEO, _W_CHECKPOINT, _W_CPATH = geos, checkpoint, cpath
    from shapely.strtree import STRtree
    _W_SINDEX = STRtree(list(_W_GEO))
    import torch
    from criticality.gcn_model import RoadGCN
    _W_MODEL = RoadGCN()
    _W_MODEL.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    _W_MODEL.eval()


def _process_tile(tile):
    """One 512 tile: rasterize -> skeleton -> build -> heal -> GCN.

    tile = (c0, r0, w, h, subtransform)
    Returns a dict (empty-fg tiles are tiny) of healed edges + gatekeepers in
    both global-pixel and projected-geo coordinates.
    """
    c0, r0, w, h, subtransform = tile
    from shapely.geometry import box
    from rasterio.features import rasterize
    from skimage.morphology import dilation, disk
    from graph.skeletonize import skeletonize_mask
    from graph.build_graph import build_graph
    from graph.heal import heal_graph
    from criticality.graph_adapter import graph_to_tensor
    from criticality.gcn_evaluate import rank_gatekeepers

    bounds = (subtransform.c, subtransform.f + subtransform.e * h,
              subtransform.c + subtransform.a * w, subtransform.f)
    cand = _W_SINDEX.query(box(*bounds))
    if len(cand) == 0:
        return {"fg": 0, "c0": c0, "r0": r0}
    mask = np.zeros((h, w), dtype=np.uint8)
    shapes = ((_W_GEO[int(i)], 255) for i in cand)
    rasterize(shapes, out_shape=(h, w), transform=subtransform, fill=0, out=mask)
    fg = int((mask > 0).sum())
    if fg == 0:
        return {"fg": 0, "c0": c0, "r0": r0}

    road = dilation((mask > 0).astype(np.uint8), disk(3))
    sk, nodes, edges = skeletonize_mask(road)
    G = build_graph(sk, nodes, edges, transform=subtransform)
    Gh = heal_graph(
        G,
        max_distance=config.MAX_HEAL_DISTANCE,
        lambda_weight=config.LAMBDA_ANGULAR,
        max_gap_m=config.HEAL_MAX_GAP_M,
    )

    inv = ~subtransform
    edges_px, edges_geo = [], []
    gaps = []
    for u, v, d in Gh.edges(data=True):
        pu, pv = Gh.nodes[u].get("pos"), Gh.nodes[v].get("pos")
        if pu is None or pv is None:
            continue
        healing = 1 if d.get("healing", False) else 0
        if healing:
            gaps.append(float(d.get("length", 0.0)))
        edges_geo.append([pu[0], pu[1], pv[0], pv[1], healing])
        c1, r1 = inv * (pu[0], pu[1])
        c2, r2 = inv * (pv[0], pv[1])
        edges_px.append([int(c0 + round(c1)), int(r0 + round(r1)),
                         int(c0 + round(c2)), int(r0 + round(r2)), healing])

    gates_px, gates_geo = [], []
    try:
        x, edge_index, _, node_ids = graph_to_tensor(Gh, centrality=None)
        idx = rank_gatekeepers(_W_MODEL, x, edge_index)
        for i in idx.tolist():
            n = node_ids[i]
            p = Gh.nodes[n].get("pos")
            if p is None:
                continue
            gates_geo.append([p[0], p[1]])
            cg, rg = inv * (p[0], p[1])
            gates_px.append([int(c0 + round(cg)), int(r0 + round(rg))])
    except Exception:
        gates_px, gates_geo = [], []

    return {
        "fg": fg, "c0": c0, "r0": r0,
        "edges_px": edges_px, "edges_geo": edges_geo,
        "gates_px": gates_px, "gates_geo": gates_geo,
        "n_edges": len(edges_px),
        "n_bridges": sum(1 for e in edges_px if e[4]),
        "gap_max": float(max(gaps)) if gaps else 0.0,
        "gap_sum": float(sum(gaps)), "gap_n": len(gaps),
    }


# ─── assembly ────────────────────────────────────────────────────────────────
def _to_wgs84(x, y, xform):
    return xform(x, y)


def main():
    ap = argparse.ArgumentParser(description="Full-map graph-heal + GCN")
    ap.add_argument("--city", default="hyderabad")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--fidelity", default="simplified", choices=["full", "simplified"])
    ap.add_argument("--road-step", type=int, default=4,
                    help="keep 1-in-N road (non-bridge) edges when simplified")
    ap.add_argument("--workers", type=int, default=0, help="0 = autodetect")
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "graph-output-tests"))
    ap.add_argument("--checkpoint", default=config.MODEL_SAVE_PATH)
    args = ap.parse_args()

    import rasterio
    import geopandas as gpd
    from rasterio.windows import Window
    from rasterio.features import rasterize

    import harness as H
    cfg = H.CITIES[args.city]

    t0 = time.perf_counter()
    os.makedirs(args.out, exist_ok=True)

    # full map extent
    with rasterio.open(cfg["tif"]) as src:
        W, HH = src.width, src.height
        full_transform, full_crs = src.transform, src.crs

    # load road geometries once
    gdf = gpd.read_file(cfg["gpkg"], layer=cfg["layer"])
    geos = [g for g in gdf.geometry if g is not None and not g.is_empty]
    print(f"[fullmap] {args.city}: map {W}x{HH} px, {len(geos)} road geometries")

    # build non-overlapping tile grid
    ts = args.tile_size
    tiles = []
    for r0 in range(0, HH, ts):
        for c0 in range(0, W, ts):
            w = min(ts, W - c0)
            h = min(ts, HH - r0)
            window = Window(c0, r0, w, h)
            subtransform = rasterio.windows.transform(window, full_transform)
            tiles.append((c0, r0, w, h, subtransform))
    print(f"[fullmap] tiles: {len(tiles)} ({len(tiles) // 1000}K)")

    workers = args.workers or max(1, mp.cpu_count() - 1)
    print(f"[fullmap] workers: {workers}")

    results = []
    with mp.Pool(processes=workers, initializer=_init_worker,
                 initargs=(geos, args.checkpoint, os.path.join(_ROOT, "criticality"))) as pool:
        it = pool.imap_unordered(_process_tile, tiles, chunksize=32)
        for i, res in enumerate(it, 1):
            results.append(res)
            if i % 500 == 0:
                el = time.perf_counter() - t0
                print(f"  {i}/{len(tiles)} tiles  {el:.0f}s  "
                      f"({i / max(el, 1e-6):.0f} tiles/s)")
    pool.join()

    t_graph = time.perf_counter() - t0
    # aggregate
    nonempty = [r for r in results if r.get("fg", 0) > 0]
    e_px = [e for r in nonempty for e in r["edges_px"]]
    e_geo = [e for r in nonempty for e in r["edges_geo"]]
    g_px = [p for r in nonempty for p in r["gates_px"]]
    g_geo = [p for r in nonempty for p in r["gates_geo"]]
    n_bridges = sum(r["n_bridges"] for r in nonempty)
    gap_max = max((r["gap_max"] for r in nonempty), default=0.0)
    gap_sum = sum(r["gap_sum"] for r in nonempty)
    gap_n = sum(r["gap_n"] for r in nonempty)
    gap_mean = gap_sum / gap_n if gap_n else 0.0

    print(f"[fullmap] graph done {t_graph:.0f}s | nonempty tiles={len(nonempty)} "
          f"| edges={len(e_geo)} | bridges={n_bridges} | gatekeepers={len(g_geo)}")

    # ---- GeoJSON (projected CRS) ----
    feats = []
    road_n = 0  # decimation counter for simplified fidelity
    for e in e_geo:
        h = e[4]
        if args.fidelity == "simplified" and h == 0:
            road_n += 1
            if road_n % args.road_step != 0:
                continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[e[0], e[1]], [e[2], e[3]]]},
            "properties": {"kind": "bridge" if h else "road", "healing": h},
        })
    for p in g_geo:
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p[0], p[1]]},
            "properties": {"kind": "gatekeeper"},
        })
    geojson = {"type": "FeatureCollection", "features": feats}
    gjson_path = os.path.join(args.out, f"{args.city}_fullmap.geojson")
    with open(gjson_path, "w") as f:
        json.dump(geojson, f)
    print(f"[fullmap] GeoJSON: {gjson_path} ({os.path.getsize(gjson_path)/1e6:.1f} MB, {len(feats)} features)")

    # ---- PNG preview (downscaled, global pixel) ----
    _render_png(e_px, g_px, W, HH, os.path.join(args.out, f"{args.city}_fullmap.png"))

    # ---- Leaflet HTML (WGS84) ----
    try:
        from pyproj import Transformer
        xf = Transformer.from_crs(str(full_crs), "EPSG:4326", always_xy=True)
        out_geo = []
        for e in e_geo:
            x1, y1 = xf.transform(e[0], e[1]); x2, y2 = xf.transform(e[2], e[3])
            out_geo.append([[y1, x1], [y2, x2], e[4]])
        g_wgs = []
        for p in g_geo:
            x, y = xf.transform(p[0], p[1])
            g_wgs.append([y, x])
        _write_html(out_geo, g_wgs, os.path.join(args.out, f"{args.city}_fullmap.html"),
                    center=[xf.transform(full_transform.c + W / 2 * full_transform.a,
                                        full_transform.f + HH / 2 * full_transform.e)])
    except Exception as exc:
        print(f"[fullmap] HTML skipped ({type(exc).__name__}: {exc})")

    report = {
        "city": args.city,
        "map_size": [W, HH],
        "crs": str(full_crs),
        "tile_size": ts,
        "workers": workers,
        "fidelity": args.fidelity,
        "tiles_total": len(tiles),
        "tiles_with_roads": len(nonempty),
        "edges_total": len(e_geo),
        "bridges_total": n_bridges,
        "gatekeepers_total": len(g_geo),
        "gap_max_m": round(gap_max, 2),
        "gap_mean_m": round(gap_mean, 2),
        "gap_cap_m": config.HEAL_MAX_GAP_M,
        "wall_clock_s": round(t_graph, 1),
        "outputs": {
            "geojson": gjson_path, "png": f"{args.city}_fullmap.png",
            "html": f"{args.city}_fullmap.html",
        },
        "note": "tiles healed independently; edges not stitched across tile boundaries",
    }
    rpath = os.path.join(args.out, f"{args.city}_fullmap_report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[fullmap] report: {rpath}")
    print(json.dumps({k: v for k, v in report.items() if k != "outputs"}, indent=2))


# ─── rendering ───────────────────────────────────────────────────────────────
def _render_png(edges_px, gates_px, W, HH, path, target_w=10000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    scale = target_w / W
    tw, th = int(W * scale), int(HH * scale)
    fig = plt.figure(figsize=(tw / 100, th / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    road = [[(e[0] * scale, e[1] * scale), (e[2] * scale, e[3] * scale)] for e in edges_px if e[4] == 0]
    br = [[(e[0] * scale, e[1] * scale), (e[2] * scale, e[3] * scale)] for e in edges_px if e[4] == 1]
    if road:
        ax.add_collection(LineCollection(road, colors="#3b7fdd", linewidths=0.4))
    if br:
        ax.add_collection(LineCollection(br, colors="#e03131", linewidths=0.7))
    if gates_px:
        gx = [p[0] * scale for p in gates_px]
        gy = [p[1] * scale for p in gates_px]
        ax.scatter(gx, gy, s=3, c="#c2185b", marker="o", linewidths=0)
    ax.set_xlim(0, tw); ax.set_ylim(th, 0)
    ax.legend(handles=[
        Line2D([0], [0], color="#3b7fdd", lw=2, label="Roads"),
        Line2D([0], [0], color="#e03131", lw=2, label="Healed bridges"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c2185b", markersize=6, label="GCN gatekeepers"),
    ], loc="lower right", frameon=True)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"[fullmap] PNG: {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def _write_html(edges, gates, path, center):
    js_edges = json.dumps(edges)
    js_gates = json.dumps(gates)
    cx, cy = center[0]
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>TERRAIN fullmap</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#m{{margin:0;height:100%}}</style></head>
<body><div id="m"></div>
<script>
var m = L.map('m').setView([{cx:.5f},{cy:.5f}], 13);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom:19, attribution:'© OpenStreetMap'}}).addTo(m);
var edges = {js_edges};
var bridges = L.layerGroup(); var roads = L.layerGroup();
edges.forEach(function(e){{
  var p = L.polyline(e.slice(0,2), {{color: e[2]==1?'#e03131':'#3b7fdd', weight:2}});
  (e[2]==1?bridges:roads).addLayer(p);
}});
roads.addTo(m); bridges.addTo(m);
var gates = {js_gates};
gates.forEach(function(g){{ L.circleMarker(g,{{radius:4,color:'#c2185b',fillOpacity:1}}).addTo(m); }});
L.control.layers({{'Roads':roads,'Bridges':bridges}}, {{'Gatekeepers':L.layerGroup()}}).addTo(m);
var lg = L.control({{position:'bottomright'}}); lg.onAdd=function(){{
 var d=L.DomUtil.create('div'); d.innerHTML=
 '<div style="background:#fff;padding:6px 8px;border:1px solid #ccc;font:12px sans-serif">'+
 '<i style="background:#3b7fdd;display:inline-block;width:18px;height:3px"></i> Roads<br>'+
 '<i style="background:#e03131;display:inline-block;width:18px;height:3px"></i> Healed bridges<br>'+
 '<i style="background:#c2185b;display:inline-block;width:8px;height:8px;border-radius:50%"></i> GCN gatekeepers</div>';
 return d;}}; lg.addTo(m);
</script></body></html>"""
    with open(path, "w") as f:
        f.write(html)
    print(f"[fullmap] HTML: {path} ({os.path.getsize(path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
