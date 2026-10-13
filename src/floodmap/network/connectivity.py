"""Settlement-to-hub connectivity before and after the predicted flood.

Scientific responsibility
-------------------------
For each settlement and each hub group (e.g. hospital, town), compare the
shortest road route to the nearest hub before the event with the route after
removing flood-disrupted road pieces, and classify the change
(architecture.md §9, AGENTS.md §9).

Method
------
* **Removal rule.** A piece is removed when its flooded length reaches the
  threshold, or the whole piece if it is shorter (same rule as exposure).
* **Unobserved pixels.** The flooded graph is evaluated twice: nodata pieces
  passable ("optimistic") and nodata pieces removed ("pessimistic"). Only an
  outcome that holds in both is reported as such; otherwise UNKNOWN.
* **Routing.** One Dijkstra run per scenario from a virtual source joined to
  every hub node at zero cost, weighted by length. That yields every node's
  distance to its nearest hub and, through the predecessor map, which hub.
* **Disconnected components.** A settlement with no route to any hub before
  the event is INSUFFICIENT_DATA, never POTENTIALLY_CUT_OFF: the flood did not
  cause that, and many Himalayan hamlets have no mapped road at all.

These are inferences about modelled road accessibility, not confirmed human
isolation (architecture.md §18).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import MultiLineString
from shapely.ops import linemerge

from floodmap.network.graph import RoadNetwork, Snap
from floodmap.utils.config import load_config

CONNECTED = "CONNECTED"
POTENTIALLY_DISRUPTED = "POTENTIALLY_DISRUPTED"
POTENTIALLY_CUT_OFF = "POTENTIALLY_CUT_OFF"
UNKNOWN = "UNKNOWN"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
STATUSES = (CONNECTED, POTENTIALLY_DISRUPTED, POTENTIALLY_CUT_OFF, UNKNOWN, INSUFFICIENT_DATA)

REASON_NO_ROAD = "no mapped road within snap distance"
REASON_NO_BASELINE_ROUTE = "no road route to any hub before the event"
REASON_NO_HUBS = "no hub of this group is near a mapped road"

_SOURCE = -1  # virtual node joined to every hub


@dataclass(frozen=True)
class ConnectivityRules:
    snap_max_m: float = 500.0
    default_disruption_m: float = 10.0
    sensitivity_disruption_m: Tuple[float, ...] = (10.0, 30.0, 100.0)
    detour_ratio: float = 1.5

    @classmethod
    def from_config(cls, impact_config: Optional[Mapping[str, Any]] = None) -> "ConnectivityRules":
        block = (impact_config or load_config("impact"))["network"]
        return cls(
            snap_max_m=float(block["snap_max_m"]),
            default_disruption_m=float(block["default_disruption_m"]),
            sensitivity_disruption_m=tuple(float(t) for t in block["sensitivity_disruption_m"]),
            detour_ratio=float(block["detour_ratio"]),
        )


# ---------------------------------------------------------------------------
# Disruption and routing
# ---------------------------------------------------------------------------


def removed_pieces(pieces: pd.DataFrame, threshold_m: float, block_unknown: bool) -> frozenset:
    """Piece ids removed under a threshold.

    ``pieces`` needs ``piece_id``, ``length_m``, ``flooded_m`` and ``unknown_m``.
    """
    length = pieces["length_m"].to_numpy()
    threshold = np.minimum(threshold_m, length) - 1e-6
    hit = pieces["flooded_m"].to_numpy() >= threshold
    if block_unknown:
        hit |= pieces["unknown_m"].to_numpy() >= threshold
    hit &= length > 0
    return frozenset(pieces["piece_id"].to_numpy()[hit].tolist())


def _weight(removed: frozenset) -> Callable:
    def weight(u, v, keyed: Mapping[int, Mapping[str, float]]):
        best = None
        for key, attrs in keyed.items():
            if key in removed:
                continue
            if best is None or attrs["length_m"] < best:
                best = attrs["length_m"]
        return best  # None hides the edge from Dijkstra

    return weight


@dataclass
class Reach:
    """Distances from the nearest hub, with predecessors for tracing routes."""

    distance: Dict[int, float]
    predecessor: Dict[int, List[int]]
    removed: frozenset

    def reachable(self, node: int) -> bool:
        return node in self.distance

    def path_from_hub(self, node: int) -> List[int]:
        """Nodes from the nearest hub to ``node`` (first element is the hub node)."""
        path = [node]
        while True:
            previous = self.predecessor[path[-1]]
            if previous == [_SOURCE]:
                break
            path.append(previous[0])
        return path[::-1]


def reach_from_hubs(graph: nx.MultiGraph, hub_nodes: Iterable[int], removed: frozenset) -> Reach:
    """Multi-source shortest paths from every hub node at once."""
    hubs = sorted({int(n) for n in hub_nodes if n >= 0})
    if not hubs:
        return Reach(distance={}, predecessor={}, removed=removed)
    graph.add_edges_from((_SOURCE, hub, -1, {"length_m": 0.0}) for hub in hubs)
    try:
        predecessor, distance = nx.dijkstra_predecessor_and_distance(
            graph, _SOURCE, weight=_weight(removed)
        )
    finally:
        graph.remove_node(_SOURCE)
    distance.pop(_SOURCE, None)
    predecessor.pop(_SOURCE, None)
    return Reach(distance=distance, predecessor=predecessor, removed=removed)


def route_geometry(network: RoadNetwork, reach: Reach, node: int):
    """Line geometry of the route from the nearest hub to ``node``."""
    path = reach.path_from_hub(node)
    if len(path) < 2:
        return None
    pieces = network.pieces
    segments = []
    for a, b in zip(path[:-1], path[1:]):
        candidates = [
            (attrs["length_m"], key)
            for key, attrs in network.graph[a][b].items()
            if key not in reach.removed and key >= 0
        ]
        _, key = min(candidates)
        segments.append(pieces.geometry.iloc[key])
    return linemerge(MultiLineString(segments))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(
    node: int,
    baseline: Reach,
    optimistic: Reach,
    pessimistic: Reach,
    detour_ratio: float,
) -> Tuple[str, Optional[str]]:
    """Status and reason for one settlement node in one hub group."""
    if node < 0:
        return INSUFFICIENT_DATA, REASON_NO_ROAD
    if not baseline.reachable(node):
        return INSUFFICIENT_DATA, REASON_NO_BASELINE_ROUTE
    if not optimistic.reachable(node):
        return POTENTIALLY_CUT_OFF, None
    if not pessimistic.reachable(node):
        return UNKNOWN, "cut off only if unobserved pixels hide flood"
    before = baseline.distance[node]
    after = optimistic.distance[node]
    if before > 0 and after / before >= detour_ratio:
        return POTENTIALLY_DISRUPTED, None
    return CONNECTED, None


def _text(value: Any) -> Optional[str]:
    """Missing names arrive as NaN (pandas string dtype), which is not valid JSON."""
    return value if isinstance(value, str) else None


def _hub_label(hub_names: Mapping[int, str], reach: Reach, node: int) -> Optional[str]:
    if node < 0 or not reach.reachable(node):
        return None
    return hub_names.get(reach.path_from_hub(node)[0])


def _km(reach: Reach, node: int) -> Optional[float]:
    if node < 0 or not reach.reachable(node):
        return None
    return round(reach.distance[node] / 1000.0, 3)


def assess_connectivity(
    network: RoadNetwork,
    pieces_measured: pd.DataFrame,
    settlements: gpd.GeoDataFrame,
    settlement_snap: Snap,
    hubs: gpd.GeoDataFrame,
    hub_snap: Snap,
    rules: ConnectivityRules,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict[str, Any]]:
    """Classify every settlement against every hub group.

    Returns per-settlement results (point geometry, EPSG:4326), route lines
    (EPSG:4326) and a summary dictionary.
    """
    graph = network.graph
    groups = list(dict.fromkeys(hubs["hub_group"])) if len(hubs) else []
    nodes = settlement_snap.node

    results = gpd.GeoDataFrame(
        {
            "osm_id": settlements["osm_id"].to_numpy(),
            "name": settlements["name"].to_numpy(),
            "place": settlements["place"].to_numpy(),
            "snap_distance_m": np.round(settlement_snap.distance_m, 1),
        },
        geometry=settlements.geometry.representative_point().to_crs("EPSG:4326").values,
        crs="EPSG:4326",
    )
    routes: List[Dict[str, Any]] = []
    route_geoms = []
    group_summary: Dict[str, Any] = {}
    sensitivity: Dict[str, Any] = {}

    no_removal = frozenset()
    default_opt = removed_pieces(pieces_measured, rules.default_disruption_m, False)
    default_pes = removed_pieces(pieces_measured, rules.default_disruption_m, True)

    for group in groups:
        in_group = (hubs["hub_group"] == group).to_numpy()
        group_nodes = hub_snap.node[in_group]
        names = hubs["name"].to_numpy()[in_group]
        hub_names: Dict[int, str] = {}
        for node, name in zip(group_nodes, names):
            if node >= 0 and (int(node) not in hub_names or hub_names[int(node)] is None):
                hub_names[int(node)] = _text(name)

        baseline = reach_from_hubs(graph, group_nodes, no_removal)
        optimistic = reach_from_hubs(graph, group_nodes, default_opt)
        pessimistic = reach_from_hubs(graph, group_nodes, default_pes)

        statuses, reasons = [], []
        for node in nodes:
            status, reason = classify(
                int(node), baseline, optimistic, pessimistic, rules.detour_ratio
            )
            if status == INSUFFICIENT_DATA and not hub_names:
                reason = REASON_NO_HUBS
            statuses.append(status)
            reasons.append(reason)

        results[f"{group}_status"] = statuses
        results[f"{group}_reason"] = reasons
        results[f"{group}_hub_before"] = [_hub_label(hub_names, baseline, int(n)) for n in nodes]
        results[f"{group}_km_before"] = [_km(baseline, int(n)) for n in nodes]
        results[f"{group}_hub_after"] = [_hub_label(hub_names, optimistic, int(n)) for n in nodes]
        results[f"{group}_km_after"] = [_km(optimistic, int(n)) for n in nodes]
        results[f"{group}_km_after_pessimistic"] = [_km(pessimistic, int(n)) for n in nodes]

        for i, node in enumerate(nodes):
            node = int(node)
            for scenario, reach in (("before", baseline), ("after", optimistic)):
                if node < 0 or not reach.reachable(node):
                    continue
                geometry = route_geometry(network, reach, node)
                if geometry is None:
                    continue
                routes.append(
                    {
                        "settlement_osm_id": results.loc[i, "osm_id"],
                        "settlement_name": _text(results.loc[i, "name"]),
                        "hub_group": group,
                        "scenario": scenario,
                        "hub_name": _hub_label(hub_names, reach, node),
                        "length_km": _km(reach, node),
                    }
                )
                route_geoms.append(geometry)

        counts = pd.Series(statuses).value_counts()
        group_summary[group] = {
            "hubs_total": int(in_group.sum()),
            "hubs_on_network": int((group_nodes >= 0).sum()),
            "status_counts": {s: int(counts.get(s, 0)) for s in STATUSES},
            "potentially_cut_off": [
                {"osm_id": results.loc[i, "osm_id"], "name": _text(results.loc[i, "name"])}
                for i, s in enumerate(statuses)
                if s == POTENTIALLY_CUT_OFF
            ],
        }

        per_threshold = {}
        for threshold in rules.sensitivity_disruption_m:
            opt = reach_from_hubs(
                graph, group_nodes, removed_pieces(pieces_measured, threshold, False)
            )
            pes = reach_from_hubs(
                graph, group_nodes, removed_pieces(pieces_measured, threshold, True)
            )
            tally = pd.Series(
                [classify(int(n), baseline, opt, pes, rules.detour_ratio)[0] for n in nodes]
            ).value_counts()
            per_threshold[f"{threshold:g}"] = {
                POTENTIALLY_CUT_OFF: int(tally.get(POTENTIALLY_CUT_OFF, 0)),
                UNKNOWN: int(tally.get(UNKNOWN, 0)),
                POTENTIALLY_DISRUPTED: int(tally.get(POTENTIALLY_DISRUPTED, 0)),
            }
        sensitivity[group] = per_threshold

    routes_frame = gpd.GeoDataFrame(routes, geometry=route_geoms, crs=network.crs).to_crs(
        "EPSG:4326"
    )

    pieces_len = pieces_measured.set_index("piece_id")["length_m"]
    summary = {
        "graph": {
            "nodes": network.graph.number_of_nodes(),
            "edges": network.graph.number_of_edges(),
            "road_km": round(float(pieces_len.sum()) / 1000.0, 3),
            "connected_components": nx.number_connected_components(network.graph),
            "removed_pieces_optimistic": len(default_opt),
            "removed_km_optimistic": round(
                float(pieces_len.loc[list(default_opt)].sum()) / 1000.0, 3
            ),
            "removed_pieces_pessimistic": len(default_pes),
        },
        "settlements": {
            "total": int(len(settlements)),
            "on_network": int((nodes >= 0).sum()),
        },
        "groups": group_summary,
        "sensitivity_by_disruption_m": sensitivity,
        "rules": asdict(rules),
        "assumptions": [
            "Road pieces run between junctions: way ends, OSM nodes shared by ways, and "
            "the road vertex each settlement or hub attaches to.",
            "A piece is impassable when its flooded length reaches the threshold "
            "(or all of it, if shorter).",
            "Graph is undirected and weighted by length; oneway tags are ignored.",
            "Settlements are OSM place=village/hamlet; polygons use a representative point; "
            "each attaches to the nearest road vertex within snap_max_m.",
            "Hubs are the OSM features in configs/impact.yaml osm.hubs, searched in the AOI "
            "plus buffer.",
            "Routes are shortest paths by length to the nearest hub of each group.",
            "Settlements with no route to a hub before the event are INSUFFICIENT_DATA, "
            "not cut off.",
            "Roads outside the flood mask are assumed passable.",
            "Nodata pixels are evaluated both passable and impassable; outcomes that differ "
            "are UNKNOWN.",
            "Footpaths and footbridges are not in the graph, so foot access is not modelled.",
        ],
        "interpretation": (
            "Statuses describe modelled road accessibility, not confirmed human isolation."
        ),
    }
    return results, routes_frame, summary


__all__ = [
    "CONNECTED",
    "ConnectivityRules",
    "INSUFFICIENT_DATA",
    "POTENTIALLY_CUT_OFF",
    "POTENTIALLY_DISRUPTED",
    "Reach",
    "STATUSES",
    "UNKNOWN",
    "assess_connectivity",
    "classify",
    "reach_from_hubs",
    "removed_pieces",
    "route_geometry",
]
