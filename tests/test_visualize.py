import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT and not os.path.exists(os.path.join(_ROOT, "config.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import numpy as np
import networkx as nx
from skimage.morphology import dilation, disk
from graph.skeletonize import skeletonize_mask
from graph.build_graph import build_graph
from graph.heal import heal_graph
from graph.visualize import plot_graph_comparison, plot_healing_result

MASK = os.path.join(_HERE, "graph-tests", "data", "bengaluru_mask_window.npy")
mask = np.load(MASK)
road = dilation((mask > 0).astype(np.uint8), disk(3))

sk, nodes, edges = skeletonize_mask(road)
G = build_graph(sk, nodes, edges)
Gh = heal_graph(G, max_distance=200.0, lambda_weight=0.3, max_gap_m=300.0)

outdir = os.path.join(_ROOT, "outputs", "graph-output-tests")
if not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok=True)

plot_graph_comparison(G, Gh, save=True, save_path=os.path.join(outdir, "healing_comparison.png"))
plot_healing_result(Gh, save=True, save_path=os.path.join(outdir, "healing_result.png"))
print("plots written to", outdir)
for f in os.listdir(outdir):
    if f.endswith(".png"):
        print(" -", f, os.path.getsize(os.path.join(outdir, f)), "bytes")