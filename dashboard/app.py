"""TERRAIN Streamlit dashboard for exploring processed road-network GeoPackages."""

from __future__ import annotations

import json
from pathlib import Path

import folium
import geopandas as gpd
import networkx as nx
import pandas as pd
import pydeck as pdk
import pyogrio
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from data_adapter import AdapterError, discover_dataset, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics.json"

CITY_CONFIG = {
    "Bengaluru": {
        "file": PROCESSED_DIR / "Bengaluru_mask.gpkg",
        "sensor": "Sentinel-2 MSI",
        "resolution": "10 m",
        "projection": "EPSG:32643",
    },
    "Delhi": {
        "file": PROCESSED_DIR / "Delhi_mask.gpkg",
        "sensor": "Resourcesat-2 LISS-4",
        "resolution": "5.8 m",
        "projection": "EPSG:32644",
    },
    "Hyderabad": {
        "file": PROCESSED_DIR / "Hyderabad_mask.gpkg",
        "sensor": "Cartosat-3 MX",
        "resolution": "~1 m",
        "projection": "EPSG:32643",
    },
}

DEFAULT_MAP_EDGES = 5_000
MAX_MAP_EDGES = 300_000
DEFAULT_MAP_NODES = 2_000
MAX_MAP_NODES = 250_000

DATASET_BY_CITY = {
    "Bengaluru": "Sentinel-2 MSI",
    "Delhi": "Resourcesat-2 LISS-4",
    "Hyderabad": "Cartosat-3 MX",
}


@st.cache_data(show_spinner=False)
def load_model_metrics(path_string: str, modified_ns: int) -> dict:
    """Load and validate the lightweight GCN evaluation artifact."""
    del modified_ns
    path = Path(path_string)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    normalized = {}
    for city_key, values in payload.items():
        if not isinstance(values, dict):
            continue
        missing = {"coverage_at_k", "auc", "spearman", "topk"} - values.keys()
        if missing:
            raise ValueError(f"{city_key} is missing: {', '.join(sorted(missing))}")
        normalized[city_key.lower()] = {
            "coverage_at_k": float(values["coverage_at_k"]),
            "auc": float(values["auc"]),
            "spearman": float(values["spearman"]),
            "topk": [int(node) for node in values["topk"]],
        }
    if not normalized:
        raise ValueError("No valid city metrics were found.")
    return normalized


@st.cache_data(show_spinner=False)
def load_ranked_node_locations(
    path_string: str,
    modified_ns: int,
    node_ids: tuple[int, ...],
) -> list[list[float]]:
    """Resolve ranked model IDs against real GeoPackage node geometries."""
    del modified_ns
    path = Path(path_string)
    contract = discover_dataset(path)
    node_layer = contract["layers"]["nodes"]
    node_id_field = contract["fields"]["nodes"]["node_id"]
    nodes = pyogrio.read_dataframe(path, layer=node_layer, columns=[node_id_field])
    if nodes.crs is None:
        raise AdapterError("The nodes layer has no CRS.")
    nodes[node_id_field] = pd.to_numeric(nodes[node_id_field], errors="coerce")
    nodes = nodes[nodes[node_id_field].isin(node_ids)]
    nodes = nodes[nodes.geometry.notna() & ~nodes.geometry.is_empty].to_crs(4326)
    return [[float(point.y), float(point.x), 1.0] for point in nodes.geometry]


def render_model_performance() -> None:
    """Render model evaluation without importing the Torch/GCN runtime."""
    st.markdown('<div class="terrain-kicker">GCN EVALUATION / HOLDOUT CITIES</div>', unsafe_allow_html=True)
    st.markdown('<div class="terrain-title">Measure what the model<br/><em>actually finds.</em></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="terrain-copy">Compare gatekeeper coverage, ranking quality, and top-ranked junctions across the three supplied sensor datasets.</div>',
        unsafe_allow_html=True,
    )

    if not MODEL_METRICS_PATH.is_file():
        st.warning("Model metrics are unavailable. Expected outputs/metrics.json.")
        return
    try:
        metrics = load_model_metrics(str(MODEL_METRICS_PATH), MODEL_METRICS_PATH.stat().st_mtime_ns)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        st.error(f"Model metrics could not be loaded: {error}")
        return

    available_cities = [city for city in DATASET_BY_CITY if city.lower() in metrics]
    if not available_cities:
        st.warning("The metrics file contains no supported city names.")
        return

    if st.session_state.get("performance_city") not in available_cities:
        st.session_state.performance_city = available_cities[0]

    st.divider()
    st.selectbox(
        "Study city",
        available_cities,
        key="performance_city",
    )

    selected_city = st.session_state.performance_city
    selected = metrics[selected_city.lower()]
    st.caption(f"Satellite dataset: {DATASET_BY_CITY[selected_city]}")
    coverage_card, auc_card, spearman_card = st.columns(3)
    coverage_card.metric(
        "Coverage@20%",
        f'{selected["coverage_at_k"] * 100:.2f}%',
        help="Fraction of ground-truth critical junctions recovered in the model's top 20% predictions.",
    )
    auc_card.metric(
        "ROC AUC",
        f'{selected["auc"]:.3f}',
        help="Ranking discrimination across critical and non-critical junctions; higher is better.",
    )
    spearman_card.metric(
        "Spearman correlation",
        f'{selected["spearman"]:.3f}',
        help="Rank correlation between GCN scores and centrality targets; higher is better.",
    )

    st.divider()
    comparison_column, ranking_column = st.columns([1.45, 1], gap="large")
    with comparison_column:
        st.markdown('<div class="terrain-kicker">CROSS-DATASET COMPARISON</div>', unsafe_allow_html=True)
        st.subheader("Holdout performance")
        metric_labels = {
            "Coverage@20%": "coverage_at_k",
            "ROC AUC": "auc",
            "Spearman": "spearman",
        }
        comparison_metric = st.segmented_control(
            "Comparison metric",
            list(metric_labels),
            default="Coverage@20%",
            label_visibility="collapsed",
        ) or "Coverage@20%"
        metric_key = metric_labels[comparison_metric]
        comparison = pd.DataFrame(
            {
                "City / dataset": [f"{city} · {DATASET_BY_CITY[city]}" for city in available_cities],
                comparison_metric: [metrics[city.lower()][metric_key] for city in available_cities],
            }
        ).set_index("City / dataset")
        st.bar_chart(comparison, horizontal=True, color="#087f6d", height=310)
        st.caption("Evaluation artifact: outputs/metrics.json")

    with ranking_column:
        st.markdown('<div class="terrain-kicker">MODEL RANKING</div>', unsafe_allow_html=True)
        st.subheader(f"{selected_city} top junctions")
        ranking = pd.DataFrame({
            "Rank": range(1, len(selected["topk"]) + 1),
            "Node ID": selected["topk"],
        })
        st.dataframe(ranking, hide_index=True, use_container_width=True, height=310)
        st.caption("Node IDs are mapped only when they match real GeoPackage node_id values.")

    st.divider()
    st.markdown('<div class="terrain-kicker">SPATIAL CONCENTRATION</div>', unsafe_allow_html=True)
    st.subheader(f"{selected_city} top-ranked junction heatmap")
    city_path = CITY_CONFIG[selected_city]["file"]
    if not city_path.is_file():
        st.info(f"Heatmap unavailable: expected {city_path.relative_to(PROJECT_ROOT)}.")
        return
    try:
        with st.spinner("Matching ranked nodes to GeoPackage geometry…", show_time=True):
            heat_points = load_ranked_node_locations(
                str(city_path),
                city_path.stat().st_mtime_ns,
                tuple(selected["topk"]),
            )
    except (AdapterError, OSError, ValueError) as error:
        st.warning(f"Heatmap unavailable: {error}")
        return
    if not heat_points:
        st.warning("None of the ranked model IDs match this GeoPackage's node_id values, so no heatmap was generated.")
        return

    center_latitude = sum(point[0] for point in heat_points) / len(heat_points)
    center_longitude = sum(point[1] for point in heat_points) / len(heat_points)
    heatmap = folium.Map(
        location=[center_latitude, center_longitude],
        tiles="CartoDB positron",
        zoom_start=11,
        control_scale=True,
    )
    HeatMap(
        heat_points,
        radius=28,
        blur=20,
        min_opacity=0.35,
        gradient={0.25: "#58e1c1", 0.6: "#f4ba5f", 1.0: "#d64f3b"},
    ).add_to(heatmap)
    heatmap.fit_bounds([[point[0], point[1]] for point in heat_points], padding=(24, 24))
    st_folium(
        heatmap,
        use_container_width=True,
        height=500,
        returned_objects=[],
        key=f"performance-heatmap-{selected_city}",
    )
    st.caption(f"Showing {len(heat_points)} matched ranked junctions from the real {selected_city} GeoPackage.")


def road_color(highway: object) -> list[int]:
    value = str(highway).lower()
    if any(road_class in value for road_class in ("motorway", "trunk", "primary")):
        return [255, 118, 95, 220]
    if any(road_class in value for road_class in ("secondary", "tertiary")):
        return [244, 186, 95, 210]
    return [88, 225, 193, 175]


@st.cache_resource(show_spinner=False)
def load_graph_snapshot(
    path_string: str,
    modified_ns: int,
    max_map_edges: int,
    max_map_nodes: int,
) -> dict:
    """Read graph metadata and a bounded display sample from a GeoPackage."""
    del modified_ns  # Included in the cache key so replacing the file invalidates the cache.
    path = Path(path_string)
    adapted = load_dataset(path, max_map_edges=max_map_edges, max_map_nodes=max_map_nodes)
    edges = adapted["display_edges"].copy()
    edges["path"] = edges.geometry.apply(lambda geometry: [list(point) for point in geometry.coords])
    edges["color"] = edges["highway"].apply(road_color) if "highway" in edges else [[88, 225, 193, 175]] * len(edges)
    edges["road_name"] = edges["name"].fillna("Unnamed road").astype(str) if "name" in edges else "Unnamed road"
    edges["road_class"] = edges["highway"].fillna("unknown").astype(str) if "highway" in edges else "unknown"
    edges["length_label"] = pd.to_numeric(edges["length"], errors="coerce").map(lambda value: f"{value:.1f} m" if pd.notna(value) else "Unknown")
    junctions = adapted["display_nodes"].copy()
    junctions["street_count"] = junctions["degree"]

    west, south, east, north = edges.total_bounds
    return {
        "source": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_path": str(path),
        "modified_ns": path.stat().st_mtime_ns,
        "crs": adapted["source_crs"],
        "nodes": adapted["node_count"],
        "edges": adapted["edge_count"],
        "junctions": adapted["junction_count"],
        "displayed_edges": len(edges),
        "edges_frame": edges.drop(columns="geometry"),
        "junctions_frame": junctions,
        "routing_edges_frame": adapted["routing_edges"],
        "routing_nodes_frame": adapted["routing_nodes"],
        "capabilities": adapted["contract"]["capabilities"],
        "resilience": adapted["resilience"],
        "center": {"longitude": (west + east) / 2, "latitude": (south + north) / 2},
    }


def render_map(snapshot: dict, edge_limit: int = DEFAULT_MAP_EDGES) -> None:
    visible_edges = snapshot["edges_frame"].head(edge_limit)
    edge_layer = pdk.Layer(
        "PathLayer",
        visible_edges,
        get_path="path",
        get_color="color",
        get_width=2.2,
        width_min_pixels=1,
        width_max_pixels=5,
        pickable=True,
        auto_highlight=True,
    )
    junction_layer = pdk.Layer(
        "ScatterplotLayer",
        snapshot["junctions_frame"],
        get_position="[longitude, latitude]",
        get_fill_color=[255, 118, 95, 220],
        get_line_color=[9, 17, 15, 255],
        get_radius=20,
        radius_min_pixels=3,
        radius_max_pixels=9,
        line_width_min_pixels=1,
        stroked=True,
        pickable=True,
    )
    deck = pdk.Deck(
        layers=[edge_layer, junction_layer],
        initial_view_state=pdk.ViewState(**snapshot["center"], zoom=10.3, pitch=0),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={
            "html": "<b>{road_name}</b><br/>Class: {road_class}<br/>Length: {length_label}<br/>Streets: {street_count}",
            "style": {"backgroundColor": "#ffffff", "color": "#17211e", "fontSize": "12px"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True, height=570)


@st.cache_resource(show_spinner=False)
def build_simulation_graph(
    path_string: str,
    modified_ns: int,
    _edges: pd.DataFrame,
    _nodes: pd.DataFrame,
) -> nx.Graph:
    """Build the full routing graph once per source-file revision."""
    del path_string, modified_ns
    graph = nx.Graph()
    positions = {
        int(row.node_id): [float(row.longitude), float(row.latitude)]
        for row in _nodes.itertuples(index=False)
        if pd.notna(row.node_id)
    }
    for edge in _edges.itertuples(index=False):
        if pd.isna(edge.u) or pd.isna(edge.v):
            continue
        u, v = int(edge.u), int(edge.v)
        if u not in positions or v not in positions:
            continue
        weight = float(edge.length) if pd.notna(edge.length) and float(edge.length) > 0 else 1.0
        graph.add_edge(
            u,
            v,
            weight=weight,
            path=[positions[u], positions[v]],
        )
        graph.nodes[u]["position"] = positions[u]
        graph.nodes[v]["position"] = positions[v]
    if graph.number_of_nodes() == 0:
        return graph
    largest_component = max(nx.connected_components(graph), key=len)
    simulation_graph = graph.subgraph(largest_component).copy()
    source_node_count = max(len(_nodes), graph.number_of_nodes())
    simulation_graph.graph["source_node_count"] = source_node_count
    simulation_graph.graph["largest_component_nodes"] = len(largest_component)
    simulation_graph.graph["connected_fraction"] = len(largest_component) / source_node_count
    return simulation_graph


@st.cache_data(show_spinner=False)
def selectable_node_frame(
    path_string: str,
    modified_ns: int,
    _graph: nx.Graph,
) -> pd.DataFrame:
    """Prepare the static degree-filtered node deck once per graph revision."""
    del path_string, modified_ns
    records = []
    for node, attributes in _graph.nodes(data=True):
        degree = _graph.degree(node)
        position = attributes.get("position")
        if degree <= 4 or not position:
            continue
        records.append({
            "node_id": int(node),
            "longitude": position[0],
            "latitude": position[1],
            "degree": degree,
        })
    return pd.DataFrame.from_records(records)


def route_coordinates(graph: nx.Graph, route: list[int]) -> list[list[float]]:
    coordinates: list[list[float]] = []
    for start, end in zip(route, route[1:]):
        edge = graph.edges[start, end]
        path = edge["path"]
        start_position = graph.nodes[start]["position"]
        if path and path[0] != start_position:
            path = list(reversed(path))
        if coordinates and path and coordinates[-1] == path[0]:
            coordinates.extend(path[1:])
        else:
            coordinates.extend(path)
    return coordinates


def render_failure_simulator(snapshot: dict) -> None:
    graph = build_simulation_graph(
        snapshot["source_path"],
        snapshot["modified_ns"],
        snapshot["routing_edges_frame"],
        snapshot["routing_nodes_frame"],
    )
    if graph.number_of_nodes() < 3:
        st.warning("The loaded edge sample is too small for failure simulation.")
        return

    state_defaults = {
        "sim_city": f'{snapshot["source"]}|degree>4',
        "sim_mode": "origin",
        "sim_origin": None,
        "sim_destination": None,
        "sim_via": [],
        "sim_blocked": [],
        "sim_history": [],
        "sim_chart_epoch": 0,
    }
    if st.session_state.get("sim_city") != f'{snapshot["source"]}|degree>4':
        for key, value in state_defaults.items():
            st.session_state[key] = value.copy() if isinstance(value, list) else value
    else:
        for key, value in state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value.copy() if isinstance(value, list) else value

    def remember_state() -> None:
        st.session_state.sim_history.append(
            {
                "origin": st.session_state.sim_origin,
                "destination": st.session_state.sim_destination,
                "via": list(st.session_state.sim_via),
                "blocked": list(st.session_state.sim_blocked),
            }
        )
        st.session_state.sim_history = st.session_state.sim_history[-30:]

    def restore_state(state: dict) -> None:
        st.session_state.sim_origin = state["origin"]
        st.session_state.sim_destination = state["destination"]
        st.session_state.sim_via = list(state["via"])
        st.session_state.sim_blocked = list(state["blocked"])

    st.markdown('<div class="sim-section-label">1 · BUILD YOUR SCENARIO</div>', unsafe_allow_html=True)
    controls = st.container(border=True)
    origin_button, destination_button, via_button, block_button = controls.columns(4)
    if origin_button.button("① Select origin", type="primary" if st.session_state.sim_mode == "origin" else "secondary", use_container_width=True):
        st.session_state.sim_mode = "origin"
    if destination_button.button("② Select destination", type="primary" if st.session_state.sim_mode == "destination" else "secondary", use_container_width=True):
        st.session_state.sim_mode = "destination"
    if via_button.button("＋ Toggle waypoint", type="primary" if st.session_state.sim_mode == "via" else "secondary", use_container_width=True):
        st.session_state.sim_mode = "via"
    if block_button.button("⊘ Toggle closure", type="primary" if st.session_state.sim_mode == "blocked" else "secondary", use_container_width=True):
        st.session_state.sim_mode = "blocked"

    undo_button, clear_via_button, clear_blocked_button, reset_button = controls.columns(4)
    if undo_button.button("↶ Undo last", disabled=not st.session_state.sim_history, use_container_width=True):
        restore_state(st.session_state.sim_history.pop())
        st.session_state.sim_chart_epoch += 1
        st.rerun()
    if clear_via_button.button("Clear waypoints", disabled=not st.session_state.sim_via, use_container_width=True):
        remember_state()
        st.session_state.sim_via = []
        st.session_state.sim_chart_epoch += 1
        st.rerun()
    if clear_blocked_button.button("Clear closures", disabled=not st.session_state.sim_blocked, use_container_width=True):
        remember_state()
        st.session_state.sim_blocked = []
        st.session_state.sim_chart_epoch += 1
        st.rerun()
    if reset_button.button("Reset all", use_container_width=True):
        remember_state()
        restore_state({"origin": None, "destination": None, "via": [], "blocked": []})
        st.session_state.sim_mode = "origin"
        st.session_state.sim_chart_epoch += 1
        st.rerun()

    mode_labels = {
        "origin": ("ORIGIN", "Click one junction to set the route start", "#44e68f"),
        "destination": ("DESTINATION", "Click one junction to set the route end", "#5399ff"),
        "via": ("WAYPOINTS", "Click junctions to add or remove ordered stops", "#ad70ff"),
        "blocked": ("CLOSURES", "Click junctions to block or restore them", "#ff765f"),
    }
    mode_title, mode_instruction, mode_color = mode_labels[st.session_state.sim_mode]
    controls.markdown(
        f'<div class="mode-banner" style="--mode-color:{mode_color}"><span>ACTIVE MODE</span><strong>{mode_title}</strong><p>{mode_instruction}</p></div>',
        unsafe_allow_html=True,
    )

    origin = st.session_state.sim_origin
    destination = st.session_state.sim_destination
    via_nodes = list(st.session_state.sim_via)
    blocked_nodes = list(st.session_state.sim_blocked)

    def node_summary(node: int | None) -> str:
        return f"Node {node}" if node is not None else "Not selected"

    waypoint_summary = "None" if not via_nodes else f"{len(via_nodes)} selected"
    closure_summary = "None" if not blocked_nodes else f"{len(blocked_nodes)} blocked"
    st.markdown(
        f"""
        <div class="selection-strip">
          <div class="selection-item origin"><span>ORIGIN</span><strong>{node_summary(origin)}</strong></div>
          <div class="selection-arrow">→</div>
          <div class="selection-item via"><span>WAYPOINTS</span><strong>{waypoint_summary}</strong></div>
          <div class="selection-arrow">→</div>
          <div class="selection-item destination"><span>DESTINATION</span><strong>{node_summary(destination)}</strong></div>
          <div class="selection-item blocked"><span>CLOSURES</span><strong>{closure_summary}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def calculate_route(route_graph: nx.Graph, stops: list[int]) -> tuple[list[int], float] | None:
        complete_route: list[int] = []
        total_distance = 0.0
        try:
            for start, end in zip(stops, stops[1:]):
                segment_distance, segment = nx.bidirectional_dijkstra(
                    route_graph,
                    source=start,
                    target=end,
                    weight="weight",
                )
                total_distance += segment_distance
                complete_route.extend(segment if not complete_route else segment[1:])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return complete_route, total_distance

    baseline_result = calculate_route(graph, [origin, destination]) if origin is not None and destination is not None else None
    blocked_set = frozenset(blocked_nodes)
    disrupted_graph = nx.subgraph_view(graph, filter_node=lambda node: node not in blocked_set)
    scenario_result = calculate_route(disrupted_graph, [origin, *via_nodes, destination]) if origin is not None and destination is not None else None

    if origin is None or destination is None:
        missing = "origin and destination" if origin is None and destination is None else "origin" if origin is None else "destination"
        st.markdown(f'<div class="route-state waiting"><strong>ROUTE WAITING</strong><span>Select the {missing} on the map to begin calculation.</span></div>', unsafe_allow_html=True)
    elif baseline_result is None:
        st.markdown('<div class="route-state failed"><strong>NO BASELINE ROUTE</strong><span>The selected endpoints are disconnected. The previous route has been cleared.</span></div>', unsafe_allow_html=True)
    elif scenario_result is None:
        baseline_metric, disrupted_metric, impact_metric = st.columns(3)
        baseline_metric.metric("Baseline route", f"{baseline_result[1] / 1000:.2f} km")
        disrupted_metric.metric("After scenario", "Disconnected")
        impact_metric.metric("Travel impact", "Network split")
        st.markdown('<div class="route-state failed"><strong>NETWORK DISCONNECTED</strong><span>No viable route remains after applying the selected waypoints and closures.</span></div>', unsafe_allow_html=True)
    else:
        impact = ((scenario_result[1] / baseline_result[1]) - 1) * 100 if baseline_result[1] else 0
        baseline_metric, disrupted_metric, impact_metric = st.columns(3)
        baseline_metric.metric("Baseline route", f"{baseline_result[1] / 1000:.2f} km")
        disrupted_metric.metric("Scenario route", f"{scenario_result[1] / 1000:.2f} km")
        impact_metric.metric("Travel impact", f"{impact:+.1f}%")

    base_layer = pdk.Layer(
        "PathLayer",
        snapshot["edges_frame"].head(DEFAULT_MAP_EDGES),
        id="road-background",
        get_path="path",
        get_color=[77, 112, 103, 70],
        get_width=1,
        width_min_pixels=1,
        pickable=True,
    )
    layers = [base_layer]
    if blocked_nodes:
        incident_edges = []
        for blocked_node in blocked_nodes:
            if blocked_node not in graph:
                continue
            for neighbor in graph.neighbors(blocked_node):
                incident_edges.append({
                    "path": route_coordinates(graph, [blocked_node, neighbor]),
                    "blocked_node": blocked_node,
                    "neighbor": neighbor,
                })
        if incident_edges:
            incident_frame = pd.DataFrame.from_records(incident_edges)
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    incident_frame,
                    id="blocked-edge-halo",
                    get_path="path",
                    get_color=[214, 79, 59, 80],
                    get_width=11,
                    width_min_pixels=7,
                    pickable=False,
                )
            )
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    incident_frame,
                    id="blocked-incident-edges",
                    get_path="path",
                    get_color=[182, 56, 38, 245],
                    get_width=5,
                    width_min_pixels=3,
                    pickable=False,
                )
            )
    if baseline_result:
        baseline_frame = pd.DataFrame([{"path": route_coordinates(graph, baseline_result[0])}])
        layers.append(pdk.Layer("PathLayer", baseline_frame, id="baseline-route", get_path="path", get_color=[88, 225, 193, 230], get_width=5, width_min_pixels=3))
    if scenario_result:
        scenario_frame = pd.DataFrame([{"path": route_coordinates(disrupted_graph, scenario_result[0])}])
        layers.append(pdk.Layer("PathLayer", scenario_frame, id="scenario-route", get_path="path", get_color=[244, 186, 95, 245], get_width=6, width_min_pixels=4))

    role_colors = {
        "origin": [68, 230, 143, 255],
        "destination": [83, 153, 255, 255],
        "via": [173, 112, 255, 255],
        "blocked": [255, 118, 95, 255],
        "available": [207, 220, 212, 165],
    }
    node_frame = selectable_node_frame(
        snapshot["source_path"],
        snapshot["modified_ns"],
        graph,
    ).copy()
    node_frame["role"] = "available"
    node_frame.loc[node_frame["node_id"] == origin, "role"] = "origin"
    node_frame.loc[node_frame["node_id"] == destination, "role"] = "destination"
    node_frame.loc[node_frame["node_id"].isin(via_nodes), "role"] = "via"
    node_frame.loc[node_frame["node_id"].isin(blocked_nodes), "role"] = "blocked"
    node_frame["order"] = node_frame["node_id"].map({node: index + 1 for index, node in enumerate(via_nodes)})
    node_frame["color"] = node_frame["role"].map(role_colors)
    node_frame["radius"] = 15 + node_frame["degree"].clip(upper=6) * 2
    node_frame.loc[node_frame["role"] != "available", "radius"] = 42
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            node_frame,
            id="junction-selector",
            get_position="[longitude, latitude]",
            get_fill_color="color",
            get_line_color=[9, 17, 15, 255],
            get_radius="radius",
            radius_min_pixels=3,
            radius_max_pixels=11,
            line_width_min_pixels=1,
            stroked=True,
            pickable=True,
            auto_highlight=True,
        )
    )
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(**snapshot["center"], zoom=11, pitch=0),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"html": "<b>Junction {node_id}</b><br/>Degree: {degree}<br/>Role: {role}", "style": {"backgroundColor": "#ffffff", "color": "#17211e"}},
    )
    st.markdown('<div class="sim-section-label map-label">2 · INSPECT THE IMPACT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="map-legend"><span><i class="origin"></i>Origin</span><span><i class="destination"></i>Destination</span><span><i class="via"></i>Waypoint</span><span><i class="blocked"></i>Closure</span><span><b class="incident"></b>Blocked connection</span><span><b class="baseline"></b>Baseline</span><span><b class="scenario"></b>Scenario</span></div>',
        unsafe_allow_html=True,
    )
    event = st.pydeck_chart(
        deck,
        use_container_width=True,
        height=570,
        on_select="rerun",
        selection_mode="single-object",
        key=f"failure-map-{st.session_state.sim_chart_epoch}",
    )

    selected_objects = event.selection.objects.get("junction-selector", []) if event and event.selection else []
    if selected_objects:
        selected_node = int(selected_objects[0]["node_id"])
        mode = st.session_state.sim_mode
        remember_state()
        if mode == "origin":
            if selected_node == st.session_state.sim_destination:
                st.session_state.sim_destination = None
            st.session_state.sim_origin = selected_node
            st.session_state.sim_via = [node for node in st.session_state.sim_via if node != selected_node]
            st.session_state.sim_blocked = [node for node in st.session_state.sim_blocked if node != selected_node]
        elif mode == "destination":
            if selected_node == st.session_state.sim_origin:
                st.session_state.sim_origin = None
            st.session_state.sim_destination = selected_node
            st.session_state.sim_via = [node for node in st.session_state.sim_via if node != selected_node]
            st.session_state.sim_blocked = [node for node in st.session_state.sim_blocked if node != selected_node]
        elif mode == "via":
            if selected_node not in {st.session_state.sim_origin, st.session_state.sim_destination}:
                st.session_state.sim_blocked = [node for node in st.session_state.sim_blocked if node != selected_node]
                if selected_node in st.session_state.sim_via:
                    st.session_state.sim_via.remove(selected_node)
                else:
                    st.session_state.sim_via.append(selected_node)
        elif mode == "blocked":
            protected = {st.session_state.sim_origin, st.session_state.sim_destination, *st.session_state.sim_via}
            if selected_node not in protected:
                if selected_node in st.session_state.sim_blocked:
                    st.session_state.sim_blocked.remove(selected_node)
                else:
                    st.session_state.sim_blocked.append(selected_node)
        st.session_state.sim_chart_epoch += 1
        st.rerun()

    st.caption("Selectable junctions are temporarily filtered to degree > 4. Click any visible junction while a selection mode is active.")


st.set_page_config(page_title="TERRAIN — Urban Network Explorer", page_icon="🛰️", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f7f8f5; color: #17211e; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #edf1ed; border-right: 1px solid #d5ddd8; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #dce3de; border-top: 2px solid #087f6d; padding: 1.15rem 1.25rem; box-shadow: 0 5px 18px rgba(23,33,30,.05); }
    [data-testid="stMetric"]:nth-child(2) { border-top-color: #f4ba5f; }
    [data-testid="stMetric"]:nth-child(3) { border-top-color: #ff765f; }
    [data-testid="stMetricLabel"] { color: #607069; }
    .terrain-kicker { color: #087f6d; font-size: .72rem; font-weight: 800; letter-spacing: .16em; }
    .terrain-title { color: #17211e; font-size: clamp(2.8rem, 6vw, 5.8rem); line-height: .94; letter-spacing: -.055em; font-weight: 650; margin: .65rem 0 1rem; }
    .terrain-title em { color: #d64f3b; font-style: normal; }
    .terrain-copy { color: #607069; max-width: 690px; line-height: 1.7; }
    .source-box { border-left: 2px solid #087f6d; background: #ffffff; padding: .9rem 1rem; color: #607069; font-size: .78rem; box-shadow: 0 3px 14px rgba(23,33,30,.05); }
    .source-box code { color: #24433a; }
    .unavailable-banner { border: 1px solid rgba(177,117,18,.35); background: #fff8e8; padding: .75rem 1rem; color: #835a12; }
    .sim-section-label { color: #087f6d; font-size: .68rem; font-weight: 800; letter-spacing: .16em; margin: .35rem 0 .7rem; }
    .sim-section-label.map-label { margin-top: 1.2rem; }
    .mode-banner { display: grid; grid-template-columns: auto auto 1fr; align-items: center; gap: .75rem; margin-top: .75rem; padding: .8rem 1rem; border-left: 3px solid var(--mode-color); background: color-mix(in srgb, var(--mode-color) 8%, #ffffff); }
    .mode-banner span { color: #697870; font-size: .6rem; letter-spacing: .12em; }
    .mode-banner strong { color: #17211e; font-size: .78rem; letter-spacing: .08em; }
    .mode-banner p { color: #526159; font-size: .78rem; margin: 0; }
    .selection-strip { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr 1fr; gap: .55rem; align-items: stretch; margin: .8rem 0 1rem; }
    .selection-item { padding: .8rem .9rem; background: #ffffff; border: 1px solid #dce3de; border-top: 2px solid #7b8982; box-shadow: 0 3px 12px rgba(23,33,30,.04); }
    .selection-item.origin { border-top-color: #44e68f; }.selection-item.destination { border-top-color: #5399ff; }.selection-item.via { border-top-color: #ad70ff; }.selection-item.blocked { border-top-color: #ff765f; }
    .selection-item span { display: block; color: #697870; font-size: .58rem; letter-spacing: .12em; margin-bottom: .35rem; }
    .selection-item strong { color: #17211e; font-size: .82rem; }
    .selection-arrow { color: #89968f; display: grid; place-items: center; }
    .route-state { display: flex; gap: 1rem; align-items: center; padding: .8rem 1rem; margin: .7rem 0; border: 1px solid #dce3de; background: #ffffff; }
    .route-state strong { font-size: .66rem; letter-spacing: .12em; }.route-state span { color: #526159; font-size: .78rem; }
    .route-state.waiting { border-left: 3px solid #b17512; }.route-state.waiting strong { color: #8b5b0b; }
    .route-state.failed { border-left: 3px solid #d64f3b; }.route-state.failed strong { color: #b63826; }
    .map-legend { display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; padding: .55rem .75rem; margin-bottom: .35rem; background: #ffffff; border: 1px solid #dce3de; }
    .map-legend span { display: flex; align-items: center; gap: .38rem; color: #526159; font-size: .68rem; }
    .map-legend i { width: .55rem; height: .55rem; border-radius: 50%; display: inline-block; }.map-legend i.origin { background:#44e68f; }.map-legend i.destination { background:#5399ff; }.map-legend i.via { background:#ad70ff; }.map-legend i.blocked { background:#ff765f; }
    .map-legend b { width: 1rem; height: 3px; display: inline-block; }.map-legend b.incident { background:#b63826; }.map-legend b.baseline { background:#58e1c1; }.map-legend b.scenario { background:#f4ba5f; }
    @media (max-width: 900px) { .selection-strip { grid-template-columns: 1fr 1fr; }.selection-arrow { display:none; }.mode-banner { grid-template-columns: 1fr; gap:.3rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

menu = st.sidebar.radio(
    "TERRAIN workspace",
    ["Network Explorer", "Model Performance", "Failure Simulator"],
    captions=[
        "Inspect available GeoPackages",
        "Compare GCN evaluation outputs",
        "Disable/Reroute through a node",
    ],
)

if menu == "Model Performance":
    render_model_performance()
    st.stop()

if "applied_edge_limit" not in st.session_state:
    st.session_state.applied_edge_limit = DEFAULT_MAP_EDGES
if "pending_edge_limit" not in st.session_state:
    st.session_state.pending_edge_limit = DEFAULT_MAP_EDGES
if "applied_node_limit" not in st.session_state:
    st.session_state.applied_node_limit = DEFAULT_MAP_NODES
if "pending_node_limit" not in st.session_state:
    st.session_state.pending_node_limit = DEFAULT_MAP_NODES

if menu == "Network Explorer":
    st.sidebar.markdown("#### Map density")
    st.sidebar.slider(
        "Rendered road edges",
        min_value=1_000,
        max_value=MAX_MAP_EDGES,
        step=1_000,
        key="pending_edge_limit",
        help="Choose the geometry count, then confirm. Full-graph routing is unaffected.",
    )
    st.sidebar.caption(f'Pending: {st.session_state.pending_edge_limit:,} · Active: {st.session_state.applied_edge_limit:,}')
    if st.session_state.pending_edge_limit > 50_000:
        st.sidebar.warning("High edge counts can take substantial memory and may make browser interaction slow.")
    if st.sidebar.button(
        "Apply edge count",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.pending_edge_limit == st.session_state.applied_edge_limit,
    ):
        st.session_state.applied_edge_limit = int(st.session_state.pending_edge_limit)
        load_graph_snapshot.clear()
        st.rerun()
    rendered_edge_limit = int(st.session_state.applied_edge_limit)

    st.sidebar.slider(
        "Rendered junction nodes",
        min_value=100,
        max_value=MAX_MAP_NODES,
        step=500,
        key="pending_node_limit",
        help="Choose the visible junction count, then confirm. Full-graph routing is unaffected.",
    )
    st.sidebar.caption(f'Pending: {st.session_state.pending_node_limit:,} · Active: {st.session_state.applied_node_limit:,}')
    if st.session_state.pending_node_limit > 20_000:
        st.sidebar.warning("High node counts can overwhelm WebGL picking and make hover interaction slow.")
    if st.sidebar.button(
        "Apply node count",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.pending_node_limit == st.session_state.applied_node_limit,
    ):
        st.session_state.applied_node_limit = int(st.session_state.pending_node_limit)
        load_graph_snapshot.clear()
        st.rerun()
    rendered_node_limit = int(st.session_state.applied_node_limit)
else:
    rendered_edge_limit = DEFAULT_MAP_EDGES
    rendered_node_limit = DEFAULT_MAP_NODES

top_left, top_right = st.columns([3, 1], vertical_alignment="bottom")
with top_left:
    st.markdown('<div class="terrain-kicker">NETWORK INTELLIGENCE / PS4</div>', unsafe_allow_html=True)
    st.markdown('<div class="terrain-title">See where a city<br/>could <em>break.</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="terrain-copy">Explore road continuity, inspect the supplied GeoPackage, and prepare the interface for healed-network criticality and resilience outputs.</div>', unsafe_allow_html=True)
with top_right:
    selected_city = st.selectbox("Study area", list(CITY_CONFIG), index=0)
    selected_config = CITY_CONFIG[selected_city]
    st.caption(f'{selected_config["sensor"]} · {selected_config["resolution"]} · {selected_config["projection"]}')

selected_path: Path = selected_config["file"]
available = selected_path.is_file()
snapshot = None
graph_metrics = None

if available:
    with st.spinner(f"Loading {selected_city} GeoPackage…", show_time=True):
        try:
            snapshot = load_graph_snapshot(
                str(selected_path),
                selected_path.stat().st_mtime_ns,
                rendered_edge_limit,
                rendered_node_limit,
            )
        except Exception as error:
            st.error(f"The GeoPackage was found but could not be read: {error}")
    if snapshot:
        with st.spinner(f"Calculating {selected_city} graph metrics…", show_time=True):
            metrics_graph = build_simulation_graph(
                snapshot["source_path"],
                snapshot["modified_ns"],
                snapshot["routing_edges_frame"],
                snapshot["routing_nodes_frame"],
            )
            graph_metrics = {
                "connected_percent": float(metrics_graph.graph.get("connected_fraction", 0.0)) * 100.0,
                "resilience_index": snapshot["resilience"].get("resilience_index") if isinstance(snapshot["resilience"], dict) else None,
                "source": "pipeline" if isinstance(snapshot["resilience"], dict) else None,
            }
else:
    expected_source = selected_path.relative_to(PROJECT_ROOT)
    st.markdown(
        f'<div class="unavailable-banner">No processed GeoPackage found for {selected_city}. Expected <code>{expected_source}</code>. No substitute data is being displayed.</div>',
        unsafe_allow_html=True,
    )

if snapshot:
    metric_one, metric_two, metric_three = st.columns(3)
    connected_value = graph_metrics["connected_percent"] if graph_metrics else None
    metric_one.metric(
        "Connected road network",
        f"{connected_value:.2f}%" if isinstance(connected_value, (int, float)) else "Not calculated",
        help="Percentage of supplied graph nodes contained in the largest connected component.",
    )
    metric_two.metric("Junction candidates", f'{snapshot["junctions"]:,}', help="Temporary filter: graph degree greater than 4.")
    resilience_value = graph_metrics["resilience_index"] if graph_metrics else None
    resilience_help = "Read from the teammate pipeline's city resilience JSON sidecar; no dashboard approximation is used."
    metric_three.metric("Resilience index", f"{resilience_value:.3f}" if isinstance(resilience_value, (int, float)) else "Not calculated", help=resilience_help)

st.divider()
if menu == "Network Explorer":
    map_column, summary_column = st.columns([3.2, 1], gap="large")
    with map_column:
        st.markdown('<div class="terrain-kicker">MAIN GRAPH DECK</div>', unsafe_allow_html=True)
        st.subheader(f"{selected_city} road explorer")
        if snapshot:
            render_map(snapshot, rendered_edge_limit)
            shown_edges = min(rendered_edge_limit, snapshot["displayed_edges"])
            shown_nodes = min(rendered_node_limit, len(snapshot["junctions_frame"]))
            st.caption(f'Interactive sample: {shown_edges:,} of {snapshot["edges"]:,} edges and {shown_nodes:,} degree-filtered junctions. Adjust both render limits in the sidebar.')
        else:
            st.info("Add the city's GeoPackage to data/processed to activate the interactive explorer.")

    with summary_column:
        st.markdown('<div class="terrain-kicker">GRAPH SUMMARY</div>', unsafe_allow_html=True)
        st.subheader("Network anatomy")
        if snapshot:
            st.metric("Nodes", f'{snapshot["nodes"]:,}')
            st.metric("Edges", f'{snapshot["edges"]:,}')
            st.metric("Junction candidates · degree > 4", f'{snapshot["junctions"]:,}')
            st.markdown(f'<div class="source-box">SOURCE<br/><code>{snapshot["source"]}</code><br/><br/>CRS<br/><code>{snapshot["crs"]}</code></div>', unsafe_allow_html=True)
        else:
            st.caption("Network statistics are unavailable until the GeoPackage is present.")
            st.markdown(f'<div class="source-box">EXPECTED SOURCE<br/><code>{selected_path.relative_to(PROJECT_ROOT)}</code></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="terrain-kicker">WHAT-IF ANALYSIS</div>', unsafe_allow_html=True)
    st.subheader(f"{selected_city} junction failure simulator")
    if snapshot:
        render_failure_simulator(snapshot)
    else:
        st.warning("Failure simulation requires a real GeoPackage. Add this city's file to data/processed and reload.")
