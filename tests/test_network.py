"""Road graph and settlement connectivity tests (offline).

Every scenario is a few straight roads in UTM 45N metres around (X0, Y0), with
flood assigned directly to road pieces, so each expected status and distance
can be checked by hand.
"""

from __future__ import annotations

import math

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from floodmap.network.connectivity import (
    CONNECTED,
    INSUFFICIENT_DATA,
    POTENTIALLY_CUT_OFF,
    POTENTIALLY_DISRUPTED,
    UNKNOWN,
    ConnectivityRules,
    assess_connectivity,
    removed_pieces,
)
from floodmap.network.graph import build_road_network

UTM = "EPSG:32645"
X0, Y0 = 300_000.0, 3_100_000.0
RULES = ConnectivityRules(
    snap_max_m=500.0,
    default_disruption_m=10.0,
    sensitivity_disruption_m=(10.0, 30.0, 100.0),
    detour_ratio=1.5,
)


def _line(*points) -> LineString:
    return LineString([(X0 + x, Y0 + y) for x, y in points])


def _pt(x, y) -> Point:
    return Point(X0 + x, Y0 + y)


def _roads(lines) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"osm_id": [f"way/{i}" for i in range(len(lines))], "highway": ["track"] * len(lines)},
        geometry=list(lines),
        crs=UTM,
    )


def _settlements(*points) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "osm_id": [f"node/s{i}" for i in range(len(points))],
            "name": [f"S{i}" for i in range(len(points))],
            "place": ["hamlet"] * len(points),
        },
        geometry=list(points),
        crs=UTM,
    )


def _hubs(*entries) -> gpd.GeoDataFrame:
    """entries: (point, group, name)"""
    return gpd.GeoDataFrame(
        {
            "hub_group": [e[1] for e in entries],
            "osm_id": [f"node/h{i}" for i in range(len(entries))],
            "name": [e[2] for e in entries],
        },
        geometry=[e[0] for e in entries],
        crs=UTM,
    )


def _run(lines, settlements, hubs, flooded=None, unknown=None, piece_flood=None, rules=RULES):
    """Flood is given per road (osm_id -> metres, capped at piece length) or per piece."""
    roads = _roads(lines)
    network, (s_snap, h_snap) = build_road_network(
        roads, UTM, [settlements.geometry, hubs.geometry], rules.snap_max_m
    )
    pieces = network.pieces
    measured = pieces[["piece_id", "length_m"]].copy()
    measured["flooded_m"] = [
        min((flooded or {}).get(way, 0.0), length)
        for way, length in zip(pieces["osm_id"], pieces["length_m"])
    ]
    measured["unknown_m"] = [
        min((unknown or {}).get(way, 0.0), length)
        for way, length in zip(pieces["osm_id"], pieces["length_m"])
    ]
    if piece_flood is not None:
        measured["flooded_m"] = [
            piece_flood(geom, length) for geom, length in zip(pieces.geometry, pieces["length_m"])
        ]
    results, routes, summary = assess_connectivity(
        network, measured, settlements, s_snap, hubs, h_snap, rules
    )
    return network, results, routes, summary


ROAD_A = _line((0, 0), (3000, 0))  # hub to settlement, 3 km
DETOUR_B = _line((0, 0), (0, 2000), (3000, 2000), (3000, 0))  # 7 km, ratio 2.33
SHORT_C = _line((0, 0), (1500, 500), (3000, 0))  # 3.16 km, ratio 1.05
HUB = (_pt(0, -30), "hospital", "Hub One")
SETTLEMENT = _pt(3000, 50)
ALL = float("inf")


class TestGraph:
    def test_roads_sharing_a_node_connect(self):
        network, _ = build_road_network(
            _roads([_line((0, 0), (100, 0)), _line((100, 0), (100, 100))]), UTM
        )
        assert network.graph.number_of_nodes() == 3
        assert network.graph.number_of_edges() == 2

    def test_crossing_without_shared_node_stays_unconnected(self):
        # A bridge over a road: geometries cross, OSM nodes are not shared.
        crossing = [_line((-500, 0), (500, 0)), _line((0, -500), (0, 500))]
        network, _ = build_road_network(_roads(crossing), UTM)
        assert nx.number_connected_components(network.graph) == 2

    def test_shared_interior_node_splits_ways_into_pieces(self):
        crossing = [_line((-500, 0), (0, 0), (500, 0)), _line((0, -500), (0, 0), (0, 500))]
        network, _ = build_road_network(_roads(crossing), UTM)
        assert len(network.pieces) == 4
        assert network.graph.degree[int(network.pieces["v"].iloc[0])] == 4

    def test_pieces_partition_road_length(self):
        lines = [ROAD_A, DETOUR_B, SHORT_C]
        network, _ = build_road_network(_roads(lines), UTM, [gpd.GeoSeries([_pt(1500, 480)])])
        assert network.pieces["length_m"].sum() == pytest.approx(sum(l.length for l in lines))

    def test_parallel_ways_stay_separate_edges(self):
        network, _ = build_road_network(_roads([ROAD_A, SHORT_C]), UTM)
        u, v = network.pieces["u"].iloc[0], network.pieces["v"].iloc[0]
        assert network.graph.number_of_edges(int(u), int(v)) == 2

    def test_closed_way_returned_as_polygon_is_routable(self):
        loop = gpd.GeoDataFrame(
            {"osm_id": ["way/9"], "highway": ["service"]},
            geometry=[box(X0, Y0, X0 + 100, Y0 + 100)],
            crs=UTM,
        )
        # Attach near (0, 100) so the ring splits into two pieces between distinct nodes.
        network, _ = build_road_network(loop, UTM, [gpd.GeoSeries([_pt(5, 90)])])
        assert network.pieces["length_m"].sum() == pytest.approx(400)
        assert network.graph.number_of_edges() >= 1

    def test_snap_distance_and_reach(self):
        points = gpd.GeoSeries([_pt(3000, 50), _pt(3000, 900)])
        _, (snap,) = build_road_network(_roads([ROAD_A]), UTM, [points], snap_max_m=500)
        assert snap.node[0] >= 0 and snap.distance_m[0] == pytest.approx(50)
        assert snap.node[1] == -1 and math.isnan(snap.distance_m[1])

    def test_snapped_vertex_becomes_a_junction(self):
        road = _line((0, 0), (1500, 0), (3000, 0))
        network, _ = build_road_network(_roads([road]), UTM, [gpd.GeoSeries([_pt(1500, 40)])])
        assert len(network.pieces) == 2


class TestRemovalRule:
    def _pieces(self, length, flooded, unknown=0.0):
        return pd.DataFrame(
            {"piece_id": [0], "length_m": [length], "flooded_m": [flooded], "unknown_m": [unknown]}
        )

    def test_threshold(self):
        assert removed_pieces(self._pieces(500, 10), 10, False) == {0}
        assert removed_pieces(self._pieces(500, 9), 10, False) == frozenset()

    def test_short_piece_flooded_end_to_end_is_removed(self):
        assert removed_pieces(self._pieces(6, 6), 10, False) == {0}

    def test_unknown_only_blocks_in_pessimistic_run(self):
        pieces = self._pieces(500, 0, unknown=50)
        assert removed_pieces(pieces, 10, False) == frozenset()
        assert removed_pieces(pieces, 10, True) == {0}


class TestStatuses:
    def test_connected_when_dry(self):
        _, results, _, _ = _run([ROAD_A], _settlements(SETTLEMENT), _hubs(HUB))
        row = results.iloc[0]
        assert row["hospital_status"] == CONNECTED
        assert row["hospital_km_before"] == pytest.approx(3.0)
        assert row["hospital_hub_before"] == "Hub One"

    def test_only_road_flooded_is_cut_off(self):
        _, results, _, summary = _run(
            [ROAD_A], _settlements(SETTLEMENT), _hubs(HUB), flooded={"way/0": 40}
        )
        row = results.iloc[0]
        assert row["hospital_status"] == POTENTIALLY_CUT_OFF
        assert row["hospital_km_after"] is None or np.isnan(row["hospital_km_after"])
        assert summary["groups"]["hospital"]["potentially_cut_off"][0]["name"] == "S0"

    def test_long_detour_is_disrupted(self):
        _, results, _, _ = _run(
            [ROAD_A, DETOUR_B], _settlements(SETTLEMENT), _hubs(HUB), flooded={"way/0": ALL}
        )
        assert results.iloc[0]["hospital_status"] == POTENTIALLY_DISRUPTED
        assert results.iloc[0]["hospital_km_after"] == pytest.approx(7.0)

    def test_short_detour_is_connected(self):
        _, results, _, _ = _run(
            [ROAD_A, SHORT_C], _settlements(SETTLEMENT), _hubs(HUB), flooded={"way/0": ALL}
        )
        assert results.iloc[0]["hospital_status"] == CONNECTED
        assert results.iloc[0]["hospital_km_after"] == pytest.approx(
            SHORT_C.length / 1000, abs=1e-3
        )

    def test_nodata_on_only_road_is_unknown_not_safe(self):
        _, results, _, _ = _run(
            [ROAD_A], _settlements(SETTLEMENT), _hubs(HUB), unknown={"way/0": 200}
        )
        row = results.iloc[0]
        assert row["hospital_status"] == UNKNOWN
        assert row["hospital_km_after"] == pytest.approx(3.0)

    def test_no_route_before_the_flood_is_insufficient_data_not_cut_off(self):
        island = _line((10000, 0), (10500, 0), (11000, 0))
        _, results, _, _ = _run(
            [ROAD_A, island], _settlements(_pt(10500, 20)), _hubs(HUB), flooded={"way/1": ALL}
        )
        assert results.iloc[0]["hospital_status"] == INSUFFICIENT_DATA
        assert results.iloc[0]["hospital_reason"] == "no road route to any hub before the event"

    def test_no_mapped_road_nearby(self):
        _, results, _, _ = _run([ROAD_A], _settlements(_pt(5000, 5000)), _hubs(HUB))
        assert results.iloc[0]["hospital_status"] == INSUFFICIENT_DATA
        assert results.iloc[0]["hospital_reason"] == "no mapped road within snap distance"

    def test_no_hub_near_a_road(self):
        far_hub = (_pt(50000, 50000), "hospital", "Far")
        _, results, _, summary = _run([ROAD_A], _settlements(SETTLEMENT), _hubs(far_hub))
        assert results.iloc[0]["hospital_reason"] == "no hub of this group is near a mapped road"
        assert summary["groups"]["hospital"]["hubs_on_network"] == 0

    def test_nearest_hub_changes_when_the_route_floods(self):
        hubs = _hubs(HUB, (_pt(10000, 30), "hospital", "Hub Two"))
        roads = [ROAD_A, _line((3000, 0), (10000, 0))]
        _, results, _, _ = _run(roads, _settlements(SETTLEMENT), hubs, flooded={"way/0": ALL})
        row = results.iloc[0]
        assert (row["hospital_hub_before"], row["hospital_hub_after"]) == ("Hub One", "Hub Two")
        assert row["hospital_km_after"] == pytest.approx(7.0)
        assert row["hospital_status"] == POTENTIALLY_DISRUPTED

    def test_flood_beyond_the_settlement_does_not_cut_it_off(self):
        # Settlement attaches mid-road; flood only on the far side of it.
        road = _line((0, 0), (1500, 0), (3000, 0))
        _, results, _, _ = _run(
            [road],
            _settlements(_pt(1500, 40)),
            _hubs(HUB),
            piece_flood=lambda geom, length: length if geom.centroid.x > X0 + 1500 else 0.0,
        )
        assert results.iloc[0]["hospital_status"] == CONNECTED

    def test_groups_are_classified_separately(self):
        hubs = _hubs(HUB, (_pt(3000, -2030), "town", "Town"))
        roads = [ROAD_A, _line((3000, 0), (3000, -2000))]
        _, results, _, _ = _run(roads, _settlements(SETTLEMENT), hubs, flooded={"way/0": ALL})
        assert results.iloc[0]["hospital_status"] == POTENTIALLY_CUT_OFF
        assert results.iloc[0]["town_status"] == CONNECTED


class TestOutputs:
    def test_sensitivity_counts_follow_the_threshold(self):
        _, _, _, summary = _run(
            [ROAD_A], _settlements(SETTLEMENT), _hubs(HUB), flooded={"way/0": 50}
        )
        per = summary["sensitivity_by_disruption_m"]["hospital"]
        assert [per[t][POTENTIALLY_CUT_OFF] for t in ("10", "30", "100")] == [1, 1, 0]

    def test_routes_match_reported_distances(self):
        _, results, routes, _ = _run(
            [ROAD_A, DETOUR_B], _settlements(SETTLEMENT), _hubs(HUB), flooded={"way/0": ALL}
        )
        assert routes.crs.to_epsg() == 4326
        lengths = routes.set_index("scenario").to_crs(UTM).length / 1000
        assert lengths["before"] == pytest.approx(results.iloc[0]["hospital_km_before"], abs=1e-3)
        assert lengths["after"] == pytest.approx(results.iloc[0]["hospital_km_after"], abs=1e-3)

    def test_summary_states_assumptions_and_wording(self):
        _, _, _, summary = _run([ROAD_A], _settlements(SETTLEMENT), _hubs(HUB))
        assert len(summary["assumptions"]) >= 8
        assert "not confirmed human isolation" in summary["interpretation"]
        assert summary["graph"]["road_km"] == pytest.approx(3.0)

    def test_network_config_loads(self):
        assert ConnectivityRules.from_config() == RULES
