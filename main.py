import argparse
import os

import networkx as nx
import torch

import config


def _fallback_graph() -> nx.Graph:
    """Placeholder graph when the segmentation/graph pipeline isn't ready."""
    from experiments.synthetic_graphs import sprawl_city
    return sprawl_city()


def _build_graph(tile_path: str) -> nx.Graph:
    """Segmentation -> skeleton -> build -> heal; synthetic fallback for now."""
    try:
        from graph.build_graph import build_graph
        from graph.heal import heal_graph
        from graph.skeletonize import skeletonize_mask
        from segmentation.inference import infer_mask

        mask = infer_mask(tile_path)
        skeleton, nodes, edges = skeletonize_mask(mask)
        return heal_graph(build_graph(skeleton, nodes, edges))
    except (ImportError, NotImplementedError) as exc:
        print(f"[main] segmentation/graph not ready ({type(exc).__name__}); using synthetic fallback")
        return _fallback_graph()


def _criticality(G: nx.Graph):
    """Centrality -> GCN predictions -> resilience index."""
    from criticality.centrality import compute_centrality, detect_gatekeepers
    from criticality.gcn_evaluate import rank_gatekeepers
    from criticality.gcn_model import RoadGCN
    from criticality.graph_adapter import graph_to_tensor
    from criticality.resilience import compute_resilience_index

    centrality = compute_centrality(G, k=config.BETWEENNESS_K)
    x, edge_index, _, node_ids = graph_to_tensor(G, centrality)

    if os.path.exists(config.MODEL_SAVE_PATH):
        try:
            model = RoadGCN()
            model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location="cpu"))
            model.eval()
            idx = rank_gatekeepers(model, x, edge_index)
            predictions = [node_ids[i] for i in idx.tolist()]
        except Exception as exc:
            print(f"[main] model load failed ({type(exc).__name__}); using betweenness rankings")
            predictions = detect_gatekeepers(centrality)
    else:
        print(f"[main] no model at {config.MODEL_SAVE_PATH}; using betweenness rankings")
        predictions = detect_gatekeepers(centrality)

    resilience_index, removal_log = compute_resilience_index(G, centrality)
    return centrality, predictions, resilience_index, removal_log


def _launch_dashboard(G: nx.Graph, centrality: dict, predictions: list, resilience_index: float):
    try:
        from dashboard.app import launch_dashboard
        launch_dashboard(G, centrality, predictions, resilience_index)
    except (ImportError, NotImplementedError) as exc:
        print(f"[main] dashboard not ready ({type(exc).__name__}); skipping")


def run_terrain_pipeline(satellite_tile_path: str, skip_dashboard: bool = False) -> dict:
    """Segmentation -> graph -> criticality -> dashboard. Returns pipeline outputs."""
    print(f"[main] tile: {satellite_tile_path}")
    G = _build_graph(satellite_tile_path)
    print(f"[main] graph: n={G.number_of_nodes()} m={G.number_of_edges()}")
    centrality, predictions, resilience_index, removal_log = _criticality(G)
    print(f"[main] gatekeepers={len(predictions)} resilience_index={resilience_index:.3f}")
    if not skip_dashboard:
        _launch_dashboard(G, centrality, predictions, resilience_index)
    return {
        "graph": G,
        "centrality": centrality,
        "predictions": predictions,
        "resilience_index": resilience_index,
        "removal_log": removal_log,
    }


def _train():
    """Multi-city inductive GCN training over OSM cities (needs internet + osmnx)."""
    from criticality.centrality import compute_centrality
    from criticality.gcn_train import train_gcn_multicity

    cities = {}
    for name, place in {**config.OSM_TRAIN_CITIES, **config.OSM_HOLDOUT_CITIES}.items():
        G = _fetch_osm_graph(place)
        cities[name] = (G, compute_centrality(G, k=config.BETWEENNESS_K))
        print(f"[train] {name}: n={G.number_of_nodes()} m={G.number_of_edges()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loss = train_gcn_multicity(cities, list(config.OSM_TRAIN_CITIES.keys()), device=device)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"[train] saved {config.MODEL_SAVE_PATH} loss={loss.item():.4f}")


def _fetch_osm_graph(place: str) -> nx.Graph:
    """OSM place -> graph normalized to the contract (lat/lng + length)."""
    import osmnx as ox

    ox.settings.use_cache = True
    G = ox.graph_from_place(place, network_type="drive")
    G = ox.convert.to_undirected(G)
    G = nx.Graph(G)
    for _, d in G.nodes(data=True):
        lat, lng = d.get("y", 0.0), d.get("x", 0.0)
        d.clear()
        d["lat"], d["lng"] = lat, lng
    for _, _, d in G.edges(data=True):
        d["length"] = float(d.get("length", 0.0))
    return nx.convert_node_labels_to_integers(G)


def parse_args():
    parser = argparse.ArgumentParser(description="TERRAIN — End-to-End Pipeline")
    parser.add_argument("--tile", type=str, default="data/input_tile.tif")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--train", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        _train()
    else:
        run_terrain_pipeline(args.tile, skip_dashboard=args.skip_dashboard)
