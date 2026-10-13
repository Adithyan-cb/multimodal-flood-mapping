"""Pre-event OSM retrieval tests (offline).

The ohsome API is replaced by a fake HTTP object, so these tests check our
request construction, snapshot rule, tiling, version filtering and caching,
not ohsome itself.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from floodmap.acquisition.aoi import BoundingBox
from floodmap.infrastructure.osm import (
    OSM_ATTRIBUTION,
    OhsomeClient,
    OsmError,
    buffer_bbox,
    features_to_frame,
    fetch_pre_event_osm,
    parse_timestamp,
    resolve_snapshot,
    tile_bbox,
)
from floodmap.utils.config import load_config
from floodmap.utils.provenance import REQUIRED_ATTRIBUTIONS

LATEST = parse_timestamp("2026-07-27T09:00Z")
SNAPSHOT_TEXT = "2026-07-27T09:00:00Z"


def _feature(osm_id, valid_to=SNAPSHOT_TEXT, geometry=None, **tags):
    return {
        "type": "Feature",
        "geometry": geometry or {"type": "Point", "coordinates": [85.3, 28.0]},
        "properties": {
            "@osmId": osm_id,
            "@validFrom": "2026-07-26T09:00:00Z",
            "@validTo": valid_to,
            "@version": 1,
            **tags,
        },
    }


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeHttp:
    """Answers /metadata and records every POST."""

    def __init__(self, features=(), status_code=200):
        self.features = list(features)
        self.status_code = status_code
        self.posts = []

    def get(self, url, timeout):
        assert url.endswith("/metadata")
        return _Response(
            {"extractRegion": {"temporalExtent": {"toTimestamp": "2026-07-27T09:00Z"}}}
        )

    def post(self, url, data, timeout):
        self.posts.append((url, data))
        return _Response({"type": "FeatureCollection", "features": self.features}, self.status_code)


def _client(http, cache_dir=None):
    return OhsomeClient("https://ohsome.test/v1", http=http, cache_dir=cache_dir, retry_wait_s=0)


class TestSnapshotRule:
    def test_default_is_capped_at_latest_ohsome_data(self):
        snapshot = resolve_snapshot(date(2026, 8, 26), LATEST)
        assert snapshot == LATEST

    def test_default_is_day_before_event_when_data_is_newer(self):
        snapshot = resolve_snapshot(date(2024, 9, 27), LATEST)
        assert snapshot == datetime(2024, 9, 26, tzinfo=timezone.utc)

    def test_snapshot_on_event_day_is_rejected(self):
        with pytest.raises(OsmError, match="validation-only"):
            resolve_snapshot(
                date(2026, 7, 27), LATEST, datetime(2026, 7, 27, 1, tzinfo=timezone.utc)
            )

    def test_snapshot_after_latest_data_is_rejected(self):
        with pytest.raises(OsmError, match="later than the latest"):
            resolve_snapshot(date(2026, 8, 26), LATEST, datetime(2026, 8, 1, tzinfo=timezone.utc))

    def test_event_before_any_data_cannot_slip_through_the_cap(self):
        # Latest data after the event: the cap must not move the snapshot past it.
        with pytest.raises(OsmError):
            resolve_snapshot(date(2026, 7, 27), LATEST, LATEST)


class TestExtent:
    def test_tiles_cover_bbox_and_respect_size(self):
        bbox = BoundingBox(min_lon=85.25, min_lat=27.85, max_lon=85.40, max_lat=28.20)
        tiles = tile_bbox(bbox, 0.1)
        assert len(tiles) == 2 * 4
        assert min(t[0] for t in tiles) == bbox.min_lon
        assert max(t[2] for t in tiles) == bbox.max_lon
        assert max(t[3] for t in tiles) == bbox.max_lat
        assert all(t[2] - t[0] <= 0.1 + 1e-12 and t[3] - t[1] <= 0.1 + 1e-12 for t in tiles)

    def test_small_bbox_is_one_tile(self):
        bbox = BoundingBox(min_lon=85.0, min_lat=28.0, max_lon=85.01, max_lat=28.01)
        assert tile_bbox(bbox, 0.1) == [(85.0, 28.0, 85.01, 28.01)]

    def test_buffer_is_a_true_distance(self):
        bbox = BoundingBox(min_lon=85.25, min_lat=27.85, max_lon=85.40, max_lat=28.20)
        grown = buffer_bbox(bbox, 25)
        # 25 km is about 0.225 degrees of latitude anywhere.
        assert grown.max_lat - bbox.max_lat == pytest.approx(25 / 111.2, abs=0.01)
        # Longitude degrees are shorter by cos(latitude), so the lon buffer is wider.
        assert bbox.min_lon - grown.min_lon > grown.max_lat - bbox.max_lat

    def test_zero_buffer_is_identity(self):
        bbox = BoundingBox(min_lon=85.0, min_lat=28.0, max_lon=85.1, max_lat=28.1)
        assert buffer_bbox(bbox, 0) == bbox


class TestFeatureConversion:
    def test_keeps_only_versions_valid_at_snapshot(self):
        frame = features_to_frame(
            [
                _feature("node/1", place="hamlet", name="Kept"),
                _feature("node/2", valid_to="2026-07-26T20:00:00Z", place="hamlet"),
            ],
            LATEST,
        )
        assert list(frame["osm_id"]) == ["node/1"]
        assert frame.loc[0, "place"] == "hamlet"
        assert frame.loc[0, "name"] == "Kept"

    def test_features_spanning_tiles_are_deduplicated(self):
        frame = features_to_frame([_feature("way/9"), _feature("way/9")], LATEST)
        assert len(frame) == 1

    def test_english_name_is_preferred_and_tags_survive(self):
        frame = features_to_frame(
            [_feature("node/3", name="धुन्चे", **{"name:en": "Dhunche"})], LATEST
        )
        assert frame.loc[0, "name"] == "Dhunche"
        assert json.loads(frame.loc[0, "tags"])["name"] == "धुन्चे"

    def test_empty_response_gives_empty_frame_with_crs(self):
        frame = features_to_frame([], LATEST)
        assert len(frame) == 0
        assert frame.crs.to_epsg() == 4326


class TestClient:
    def test_request_asks_for_unclipped_history_ending_at_snapshot(self):
        http = FakeHttp([_feature("node/1")])
        bbox = BoundingBox(min_lon=85.0, min_lat=28.0, max_lon=85.05, max_lat=28.05)
        _client(http).fetch_layer("amenity=hospital", bbox, LATEST, 0.1)
        url, data = http.posts[0]
        assert url.endswith("/elementsFullHistory/geometry")
        assert data["clipGeometry"] == "false"
        assert data["time"] == "2026-07-26T09:00:00Z,2026-07-27T09:00:00Z"
        assert data["filter"] == "amenity=hospital"

    def test_cache_prevents_second_request(self, tmp_path):
        http = FakeHttp([_feature("node/1")])
        bbox = BoundingBox(min_lon=85.0, min_lat=28.0, max_lon=85.05, max_lat=28.05)
        client = _client(http, cache_dir=tmp_path)
        first, _, hits_first = client.fetch_layer("amenity=hospital", bbox, LATEST, 0.1)
        second, _, hits_second = client.fetch_layer("amenity=hospital", bbox, LATEST, 0.1)
        assert len(http.posts) == 1
        assert (hits_first, hits_second) == (0, 1)
        assert list(first["osm_id"]) == list(second["osm_id"])

    def test_permanent_http_error_is_not_retried(self):
        http = FakeHttp(status_code=403)
        bbox = BoundingBox(min_lon=85.0, min_lat=28.0, max_lon=85.05, max_lat=28.05)
        with pytest.raises(OsmError, match="HTTP 403"):
            _client(http).fetch_layer("amenity=hospital", bbox, LATEST, 0.1)
        assert len(http.posts) == 1

    def test_transient_http_error_is_retried_then_reported(self):
        http = FakeHttp(status_code=503)
        bbox = BoundingBox(min_lon=85.0, min_lat=28.0, max_lon=85.05, max_lat=28.05)
        with pytest.raises(OsmError, match="after 4 attempts"):
            _client(http).fetch_layer("amenity=hospital", bbox, LATEST, 0.1)
        assert len(http.posts) == 4


class TestFetchPreEventOsm:
    def test_layers_extents_and_provenance(self):
        http = FakeHttp([_feature("node/1", place="town", name="Town")])
        aoi = BoundingBox(min_lon=85.25, min_lat=27.85, max_lon=85.30, max_lat=27.90)
        osm = fetch_pre_event_osm(aoi, date(2026, 8, 26), buffer_km=5, client=_client(http))

        assert set(osm.layers) == {"roads", "bridges", "buildings", "settlements", "hubs"}
        assert set(osm.layers["hubs"]["hub_group"]) == {"hospital", "town"}
        assert osm.provenance["snapshot"] == SNAPSHOT_TEXT
        assert osm.provenance["layers"]["buildings"]["extent"] == "aoi"
        assert osm.provenance["layers"]["roads_buffer"]["extent"] == "aoi+buffer"
        assert osm.provenance["attribution"] == OSM_ATTRIBUTION
        # Every request in the run used the pre-event window.
        assert {data["time"] for _, data in http.posts} == {
            "2026-07-26T09:00:00Z,2026-07-27T09:00:00Z"
        }

    def test_event_inside_osm_history_still_gets_a_pre_event_snapshot(self):
        http = FakeHttp()
        aoi = BoundingBox(min_lon=85.25, min_lat=27.85, max_lon=85.30, max_lat=27.90)
        osm = fetch_pre_event_osm(aoi, date(2024, 9, 27), buffer_km=0, client=_client(http))
        assert osm.provenance["snapshot"] == "2024-09-26T00:00:00Z"


class TestConfig:
    def test_osm_attribution_is_the_required_string(self):
        assert OSM_ATTRIBUTION in REQUIRED_ATTRIBUTIONS

    def test_road_filters_select_ways_and_exclude_footpaths(self):
        filters = load_config("impact")["osm"]["filters"]
        for key in ("roads", "roads_buffer"):
            assert "type:way" in filters[key]
            for footpath in ("footway", "path", "steps"):
                assert footpath not in filters[key]

    def test_hub_groups_are_configured(self):
        hubs = load_config("impact")["osm"]["hubs"]
        assert set(hubs) >= {"hospital", "town"}
