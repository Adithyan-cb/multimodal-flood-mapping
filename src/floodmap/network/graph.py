"""Road graph construction from pre-event OSM roads.

Scientific responsibility
-------------------------
Turn OSM road geometries into a routable graph whose edges are the units the
disruption rule acts on (AGENTS.md §9: road segmentation and graph
construction must be explicit).

Construction
------------
* Coordinates are projected to a metric CRS and rounded to 1 cm. ohsome
  returns geometries without node ids, so two ways connect exactly when they
  share a vertex coordinate, i.e. an OSM node. A bridge passing over a road
  does not share a node with it and stays unconnected, as in OSM.
* A vertex is a **junction** when it is a way end, appears more than once
  (shared by ways, or a way touching itself), or is the nearest road vertex to
  a settlement or hub. Each way is cut at its junctions into **pieces**.
* Each piece is one graph edge (key = ``piece_id``) weighted by its length.
  The graph is an undirected multigraph: two ways joining the same pair of
  junctions stay separate edges, so flooding one leaves the other usable.

Pieces rather than vertex-to-vertex segments, because the disruption rule asks
"how much of the road between two junctions is flooded". With per-segment
edges, 10 m of flood split across two short segments would never reach a 10 m
threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import shapely
from pyproj import CRS
from shapely.geometry import LineString

#: Rounding for node identity in metric coordinates (1 cm).
_NODE_DECIMALS = 2


@dataclass
class Snap:
    """Where a set of points attached to the network."""

    node: np.ndarray  # node id per point, -1 when nothing within reach
    distance_m: np.ndarray  # NaN when not snapped


@dataclass
class RoadNetwork:
    crs: CRS
    pieces: gpd.GeoDataFrame  # piece_id, u, v, length_m, osm_id, highway, geometry
    node_xy: np.ndarray
    graph: nx.MultiGraph = field(repr=False)

    @property
    def n_nodes(self) -> int:
        return int(self.node_xy.shape[0])


def _road_lines(roads: gpd.GeoDataFrame, crs: CRS) -> Tuple[np.ndarray, np.ndarray]:
    """Single-part line geometries in ``crs`` and the road row each came from."""
    geoms = roads.to_crs(crs).geometry.values
    geoms = np.asarray(geoms, dtype=object)
    polygons = np.isin(shapely.get_type_id(geoms), (3, 6))
    if polygons.any():
        geoms = geoms.copy()
        geoms[polygons] = shapely.boundary(geoms[polygons])
    parts, row = shapely.get_parts(geoms, return_index=True)
    keep = np.isin(shapely.get_type_id(parts), (1, 2)) & ~shapely.is_empty(parts)
    return parts[keep], row[keep]


def build_road_network(
    roads: gpd.GeoDataFrame,
    crs: CRS,
    snap_points: Sequence[gpd.GeoSeries] = (),
    snap_max_m: float = 500.0,
) -> Tuple[RoadNetwork, List[Snap]]:
    """Build the road graph and attach each set of points to it.

    ``snap_points`` must already be in ``crs``. Each point attaches to its
    nearest road vertex within ``snap_max_m``; that vertex becomes a junction
    so routing starts exactly there.
    """
    crs = CRS.from_user_input(crs)
    lines, road_row = _road_lines(roads, crs)
    coords, part = shapely.get_coordinates(lines, return_index=True)
    keys = np.round(coords, _NODE_DECIMALS)
    node_xy, vertex_node = np.unique(keys, axis=0, return_inverse=True)
    vertex_node = vertex_node.ravel()

    snaps: List[Snap] = []
    snapped_nodes = np.zeros(len(node_xy), dtype=bool)
    if len(node_xy):
        tree = shapely.STRtree(shapely.points(node_xy))
    for points in snap_points:
        node = np.full(len(points), -1, dtype=np.int64)
        distance = np.full(len(points), np.nan)
        if len(points) and len(node_xy):
            (src, dst), dist = tree.query_nearest(
                points.values, max_distance=snap_max_m, return_distance=True, all_matches=False
            )
            node[src] = dst
            distance[src] = dist
            snapped_nodes[dst] = True
        snaps.append(Snap(node=node, distance_m=distance))

    new_part = np.r_[True, part[1:] != part[:-1]] if len(part) else np.zeros(0, bool)
    last_of_part = np.r_[part[1:] != part[:-1], True] if len(part) else np.zeros(0, bool)
    occurrences = np.bincount(vertex_node, minlength=len(node_xy))
    junction = new_part | last_of_part | (occurrences[vertex_node] > 1) | snapped_nodes[vertex_node]

    positions = np.flatnonzero(junction)
    starts, ends = positions[:-1], positions[1:]
    same_part = part[starts] == part[ends]
    starts, ends = starts[same_part], ends[same_part]

    geometries = [LineString(coords[a : b + 1]) for a, b in zip(starts, ends)]
    source_rows = road_row[part[starts]]
    pieces = gpd.GeoDataFrame(
        {
            "piece_id": np.arange(len(starts), dtype=np.int64),
            "u": vertex_node[starts],
            "v": vertex_node[ends],
            "osm_id": roads["osm_id"].to_numpy()[source_rows] if len(starts) else [],
            "highway": roads["highway"].to_numpy()[source_rows] if len(starts) else [],
        },
        geometry=geometries,
        crs=crs,
    )
    pieces["length_m"] = pieces.geometry.length

    graph = nx.MultiGraph()
    routable = pieces[(pieces["u"] != pieces["v"]) & (pieces["length_m"] > 0)]
    graph.add_edges_from(
        (int(u), int(v), int(pid), {"length_m": float(length)})
        for u, v, pid, length in zip(
            routable["u"], routable["v"], routable["piece_id"], routable["length_m"]
        )
    )
    for snap in snaps:
        graph.add_nodes_from(int(n) for n in snap.node if n >= 0)

    return RoadNetwork(crs=crs, pieces=pieces, node_xy=node_xy, graph=graph), snaps


__all__ = ["RoadNetwork", "Snap", "build_road_network"]
