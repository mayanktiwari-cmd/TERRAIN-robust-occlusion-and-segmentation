"""Schema-tolerant data adapter for TERRAIN graph and resilience outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd
import pyogrio


LAYER_ALIASES = {
    "nodes": ("nodes", "junction_nodes", "critical_nodes"),
    "edges": ("edges", "road_edges", "healed_graph"),
    "healed_edges": ("healed_edges", "bridges", "healing_edges"),
}

COLUMN_ALIASES = {
    "node_id": ("node_id", "osmid", "id", "fid"),
    "u": ("u", "from", "source", "source_id"),
    "v": ("v", "to", "target", "target_id"),
    "length": ("length", "length_m", "distance", "weight"),
    "travel_time": ("travel_time", "travel_time_s", "time_seconds"),
    "highway": ("highway", "road_class", "class"),
    "name": ("name", "road_name", "street_name"),
    "lanes": ("lanes", "lane_count"),
    "oneway": ("oneway", "one_way"),
    "street_count": ("street_count", "degree", "valence"),
    "healed": ("healed", "is_healed", "bridge_added"),
    "heal_cost": ("heal_cost", "healing_cost", "bridge_cost"),
    "confidence": ("confidence", "segmentation_confidence", "road_confidence"),
    "betweenness": ("betweenness", "betweenness_centrality", "bc"),
    "criticality_class": ("criticality_class", "criticality", "risk_class"),
    "gcn_probability": ("gcn_probability", "gatekeeper_probability", "gcn_score"),
    "is_gatekeeper": ("is_gatekeeper", "gatekeeper", "critical_node"),
}

OPTIONAL_NODE_FIELDS = ("street_count", "betweenness", "criticality_class", "gcn_probability", "is_gatekeeper")
OPTIONAL_EDGE_FIELDS = ("travel_time", "highway", "name", "lanes", "oneway", "healed", "heal_cost", "confidence")


class AdapterError(ValueError):
    """Raised when an input cannot satisfy the dashboard graph contract."""


def _available_fields(path: Path, layer: str) -> list[str]:
    return [str(field) for field in pyogrio.read_info(path, layer=layer)["fields"]]


def _resolve_field(fields: list[str], canonical: str, required: bool = False) -> str | None:
    lookup = {field.lower(): field for field in fields}
    for candidate in COLUMN_ALIASES[canonical]:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise AdapterError(f"Required field '{canonical}' was not found. Accepted aliases: {', '.join(COLUMN_ALIASES[canonical])}.")
    return None


def _resolve_layer(layer_names: set[str], canonical: str, required: bool = False) -> str | None:
    lookup = {name.lower(): name for name in layer_names}
    for candidate in LAYER_ALIASES[canonical]:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise AdapterError(f"Required layer '{canonical}' was not found. Accepted aliases: {', '.join(LAYER_ALIASES[canonical])}.")
    return None


def discover_dataset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AdapterError(f"GeoPackage not found: {path}")

    layer_names = {str(row[0]) for row in pyogrio.list_layers(path)}
    node_layer = _resolve_layer(layer_names, "nodes", required=True)
    edge_layer = _resolve_layer(layer_names, "edges", required=True)
    healed_layer = _resolve_layer(layer_names, "healed_edges")
    node_fields = _available_fields(path, node_layer)
    edge_fields = _available_fields(path, edge_layer)

    node_map = {"node_id": _resolve_field(node_fields, "node_id", required=True)}
    node_map.update({field: _resolve_field(node_fields, field) for field in OPTIONAL_NODE_FIELDS})
    edge_map = {
        "u": _resolve_field(edge_fields, "u", required=True),
        "v": _resolve_field(edge_fields, "v", required=True),
        "length": _resolve_field(edge_fields, "length", required=True),
    }
    edge_map.update({field: _resolve_field(edge_fields, field) for field in OPTIONAL_EDGE_FIELDS})

    capabilities = {
        "healing": bool(healed_layer or edge_map["healed"]),
        "confidence": bool(edge_map["confidence"]),
        "criticality": bool(node_map["betweenness"] or node_map["criticality_class"]),
        "gcn": bool(node_map["gcn_probability"] or node_map["is_gatekeeper"]),
    }
    return {
        "layers": {"nodes": node_layer, "edges": edge_layer, "healed_edges": healed_layer},
        "fields": {"nodes": node_map, "edges": edge_map},
        "capabilities": capabilities,
    }


def _read_normalized(path: Path, layer: str, field_map: dict[str, str | None], **kwargs: Any):
    selected = [source for source in field_map.values() if source]
    frame = pyogrio.read_dataframe(path, layer=layer, columns=list(dict.fromkeys(selected)), **kwargs)
    rename = {source: canonical for canonical, source in field_map.items() if source}
    return frame.rename(columns=rename)


def load_dataset(path: Path, max_map_edges: int = 15_000, max_map_nodes: int = 2_000) -> dict[str, Any]:
    contract = discover_dataset(path)
    node_layer = contract["layers"]["nodes"]
    edge_layer = contract["layers"]["edges"]
    node_map = contract["fields"]["nodes"]
    edge_map = contract["fields"]["edges"]
    node_info = pyogrio.read_info(path, layer=node_layer)
    edge_info = pyogrio.read_info(path, layer=edge_layer)

    node_attributes = _read_normalized(path, node_layer, node_map, read_geometry=False)
    routing_edges = _read_normalized(
        path,
        edge_layer,
        {key: edge_map[key] for key in ("u", "v", "length", "travel_time")},
        read_geometry=False,
    )
    routing_edges["u"] = pd.to_numeric(routing_edges["u"], errors="coerce")
    routing_edges["v"] = pd.to_numeric(routing_edges["v"], errors="coerce")
    routing_edges["length"] = pd.to_numeric(routing_edges["length"], errors="coerce")
    routing_edges = routing_edges.dropna(subset=["u", "v", "length"])

    routing_nodes = _read_normalized(path, node_layer, {"node_id": node_map["node_id"]})
    if routing_nodes.crs is None:
        raise AdapterError("The nodes layer has no CRS.")
    routing_nodes = routing_nodes.to_crs(4326)
    routing_nodes = routing_nodes[routing_nodes.geometry.notna() & ~routing_nodes.geometry.is_empty].copy()
    routing_nodes["node_id"] = pd.to_numeric(routing_nodes["node_id"], errors="coerce")
    routing_nodes = routing_nodes.dropna(subset=["node_id"])
    routing_nodes["node_id"] = routing_nodes["node_id"].astype("int64")
    routing_nodes["longitude"] = routing_nodes.geometry.x
    routing_nodes["latitude"] = routing_nodes.geometry.y
    routing_nodes = routing_nodes[["node_id", "longitude", "latitude"]]
    degree_counts = pd.concat([routing_edges["u"], routing_edges["v"]]).astype("int64").value_counts()
    routing_nodes["degree"] = routing_nodes["node_id"].map(degree_counts).fillna(0).astype(int)

    # Temporary frontend contract for unlabelled graphs: expose major junctions only.
    # Replace this predicate with teammate-provided is_gatekeeper labels when available.
    candidate_nodes = routing_nodes[routing_nodes["degree"] > 4].copy()
    junction_count = len(candidate_nodes)

    display_edges = _read_normalized(path, edge_layer, edge_map, max_features=max_map_edges)
    if display_edges.crs is None:
        raise AdapterError("The edges layer has no CRS.")
    display_edges = display_edges.to_crs(4326)
    display_edges = display_edges[display_edges.geometry.notna() & ~display_edges.geometry.is_empty].explode(index_parts=False)
    display_edges["geometry"] = display_edges.geometry.simplify(0.00001, preserve_topology=True)
    display_edges["path"] = display_edges.geometry.apply(lambda geometry: [list(point) for point in geometry.coords])

    display_nodes = candidate_nodes.head(max_map_nodes).copy()
    city_stem = path.stem
    for suffix in ("_terrain", "_mask", "_network", "_graph"):
        if city_stem.lower().endswith(suffix):
            city_stem = city_stem[: -len(suffix)]
            break
    resilience_candidates = [
        path.with_name(f"{city_stem}_resilience.json"),
        path.with_name(f"{path.stem}_resilience.json"),
    ]
    resilience_path = next((candidate for candidate in resilience_candidates if candidate.is_file()), resilience_candidates[0])
    resilience = None
    if resilience_path.is_file():
        with resilience_path.open("r", encoding="utf-8") as file:
            resilience = json.load(file)

    return {
        "contract": contract,
        "source_crs": str(edge_info["crs"]),
        "node_count": int(node_info["features"]),
        "edge_count": int(edge_info["features"]),
        "junction_count": junction_count,
        "node_attributes": node_attributes,
        "routing_edges": routing_edges,
        "routing_nodes": routing_nodes,
        "display_edges": display_edges,
        "display_nodes": display_nodes,
        "resilience": resilience,
        "resilience_path": resilience_path if resilience_path.is_file() else None,
    }
