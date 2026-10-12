"""Pre-event OpenStreetMap retrieval from the ohsome API.

Scientific responsibility
-------------------------
Return OSM roads, bridges, buildings, settlements and reference hubs exactly as
they stood at a snapshot strictly before the event (AGENTS.md §3). Post-event
edits are validation-only, so the snapshot rule is enforced here rather than
left to the caller.

How the snapshot is obtained
----------------------------
ohsome's ``/elements/geometry`` endpoint returned HTTP 403 when tested on
2026-10-12, while ``/elementsFullHistory/geometry`` worked. This module asks the
history endpoint for a one-day window ending at the snapshot and keeps only the
element versions still valid at the window's end. That is the same state a
snapshot query would return.

Geometries are requested with ``clipGeometry=false``. ohsome otherwise clips to
the request box, which would cut roads at the AOI edge and split buildings
across tiles. Unclipped features that span several tiles are deduplicated by
OSM id.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box, shape

from floodmap.acquisition.aoi import BoundingBox
from floodmap.utils.config import load_config, project_root
from floodmap.utils.provenance import REQUIRED_ATTRIBUTIONS

HISTORY_PATH = "/elementsFullHistory/geometry"
METADATA_PATH = "/metadata"

#: Taken from the enforcing copy so the two cannot drift apart.
OSM_ATTRIBUTION = next(a for a in REQUIRED_ATTRIBUTIONS if "OpenStreetMap" in a)
OSM_LICENSE = "ODbL 1.0"

#: Tags promoted to their own columns for readability on a map. Every other tag
#: is kept in the ``tags`` column as JSON.
PROMOTED_TAGS = ("highway", "bridge", "building", "place", "amenity")

#: HTTP statuses worth retrying. 403 and other 4xx are permanent.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

Tile = Tuple[float, float, float, float]


class OsmError(RuntimeError):
    """OSM retrieval failed or a request would break the pre-event rule."""


# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------


def parse_timestamp(text: str) -> datetime:
    """Parse an ohsome timestamp (``2026-07-27T09:00Z`` or with seconds) as UTC."""
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_snapshot(
    event_date: date,
    latest_available: datetime,
    requested: Optional[datetime] = None,
) -> datetime:
    """Pick the OSM snapshot for an event.

    Default: 00:00 UTC on the day before the event, capped at the latest data
    ohsome holds. The event's time of day is unknown, so the whole event day is
    treated as post-event.

    Raises
    ------
    OsmError
        If the snapshot is not strictly before the event day, or is later than
        the data ohsome holds.
    """
    event_start = datetime(event_date.year, event_date.month, event_date.day, tzinfo=timezone.utc)
    if requested is None:
        snapshot = min(event_start - timedelta(days=1), latest_available)
    else:
        snapshot = requested if requested.tzinfo else requested.replace(tzinfo=timezone.utc)
        if snapshot > latest_available:
            raise OsmError(
                f"requested OSM snapshot {format_timestamp(snapshot)} is later than the "
                f"latest ohsome data ({format_timestamp(latest_available)})"
            )
    if snapshot >= event_start:
        raise OsmError(
            f"OSM snapshot {format_timestamp(snapshot)} is not before the event day "
            f"{event_date.isoformat()}; post-event OSM is validation-only (AGENTS.md §3)"
        )
    return snapshot


# ---------------------------------------------------------------------------
# Extent handling
# ---------------------------------------------------------------------------


def buffer_bbox(bbox: BoundingBox, buffer_km: float) -> BoundingBox:
    """Grow a bbox by ``buffer_km`` on every side, measured in metres.

    The buffer is applied in the AOI's own UTM zone, so it is a true distance
    at any latitude, then converted back to an EPSG:4326 extent.
    """
    if buffer_km <= 0:
        return bbox
    aoi = gpd.GeoSeries([box(*bbox.as_list())], crs="EPSG:4326")
    utm = aoi.estimate_utm_crs()
    min_lon, min_lat, max_lon, max_lat = (
        aoi.to_crs(utm).buffer(buffer_km * 1000.0).to_crs("EPSG:4326").total_bounds
    )
    return BoundingBox(
        min_lon=max(-180.0, float(min_lon)),
        min_lat=max(-90.0, float(min_lat)),
        max_lon=min(180.0, float(max_lon)),
        max_lat=min(90.0, float(max_lat)),
    )


def tile_bbox(bbox: BoundingBox, tile_deg: float) -> List[Tile]:
    """Split a bbox into equal tiles no larger than ``tile_deg`` on a side."""
    if tile_deg <= 0:
        raise ValueError(f"tile_deg must be positive, got {tile_deg}")
    width = bbox.max_lon - bbox.min_lon
    height = bbox.max_lat - bbox.min_lat
    n_x = max(1, math.ceil(width / tile_deg - 1e-9))
    n_y = max(1, math.ceil(height / tile_deg - 1e-9))
    step_x, step_y = width / n_x, height / n_y
    tiles = []
    for j in range(n_y):
        for i in range(n_x):
            tiles.append(
                (
                    bbox.min_lon + i * step_x,
                    bbox.min_lat + j * step_y,
                    bbox.max_lon if i == n_x - 1 else bbox.min_lon + (i + 1) * step_x,
                    bbox.max_lat if j == n_y - 1 else bbox.min_lat + (j + 1) * step_y,
                )
            )
    return tiles


# ---------------------------------------------------------------------------
# Feature conversion
# ---------------------------------------------------------------------------


def features_to_frame(features: Sequence[Mapping[str, Any]], snapshot: datetime) -> gpd.GeoDataFrame:
    """Keep versions valid at ``snapshot``, deduplicate by OSM id, build a frame.

    A history response holds every version of an element inside the window. A
    version whose ``@validTo`` is earlier than the window end was replaced or
    deleted before the snapshot and is dropped.
    """
    rows: List[Dict[str, Any]] = []
    geometries = []
    seen = set()
    for feature in features:
        props = feature.get("properties") or {}
        if feature.get("geometry") is None or "@validTo" not in props:
            continue
        if parse_timestamp(props["@validTo"]) != snapshot:
            continue
        osm_id = props["@osmId"]
        if osm_id in seen:
            continue
        seen.add(osm_id)
        tags = {k: v for k, v in props.items() if not k.startswith("@")}
        row = {
            "osm_id": osm_id,
            "name": tags.get("name:en") or tags.get("name"),
            "osm_version": props.get("@version"),
        }
        for tag in PROMOTED_TAGS:
            row[tag] = tags.get(tag)
        row["tags"] = json.dumps(tags, ensure_ascii=False, sort_keys=True)
        rows.append(row)
        geometries.append(shape(feature["geometry"]))

    columns = ["osm_id", "name", "osm_version", *PROMOTED_TAGS, "tags"]
    return gpd.GeoDataFrame(rows, columns=columns, geometry=geometries, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# ohsome client
# ---------------------------------------------------------------------------


@dataclass
class LayerReport:
    """What one layer request did, for provenance."""

    filter: str
    extent: str
    bbox: List[float]
    tiles: int
    cache_hits: int
    features: int


class OhsomeClient:
    """Small ohsome client with tiling, retries and an on-disk response cache.

    ``http`` is anything with ``get`` and ``post`` matching ``requests``. Tests
    pass a fake so no test touches the network.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 300,
        max_retries: int = 3,
        max_workers: int = 4,
        cache_dir: Optional[Path] = None,
        http: Any = None,
        retry_wait_s: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.cache_dir = cache_dir
        self.http = http if http is not None else requests
        self.retry_wait_s = retry_wait_s

    @classmethod
    def from_config(cls, osm_config: Mapping[str, Any], **overrides: Any) -> "OhsomeClient":
        cache_dir = osm_config.get("cache_dir")
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            if not cache_dir.is_absolute():
                cache_dir = project_root() / cache_dir
        kwargs = dict(
            base_url=osm_config["base_url"],
            timeout_s=osm_config.get("timeout_s", 300),
            max_retries=osm_config.get("max_retries", 3),
            max_workers=osm_config.get("max_workers", 4),
            cache_dir=cache_dir,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    # -- metadata ----------------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        response = self.http.get(self.base_url + METADATA_PATH, timeout=60)
        if response.status_code != 200:
            raise OsmError(f"ohsome metadata request failed: HTTP {response.status_code}")
        return response.json()

    def latest_timestamp(self) -> datetime:
        meta = self.metadata()
        return parse_timestamp(meta["extractRegion"]["temporalExtent"]["toTimestamp"])

    # -- layers ------------------------------------------------------------

    def fetch_layer(
        self,
        osm_filter: str,
        bbox: BoundingBox,
        snapshot: datetime,
        tile_deg: float,
    ) -> Tuple[gpd.GeoDataFrame, int, int]:
        """Fetch one filter over a bbox at a snapshot.

        Returns the frame, the number of tiles and the number of cache hits.
        """
        window = f"{format_timestamp(snapshot - timedelta(days=1))},{format_timestamp(snapshot)}"
        requests_params = [
            {
                "bboxes": ",".join(f"{v:.7f}" for v in tile),
                "filter": osm_filter,
                "time": window,
                "properties": "tags,metadata",
                "clipGeometry": "false",
            }
            for tile in tile_bbox(bbox, tile_deg)
        ]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(self._fetch_tile, requests_params))

        features = [f for payload, _ in results for f in payload.get("features", [])]
        cache_hits = sum(1 for _, hit in results if hit)
        return features_to_frame(features, snapshot), len(requests_params), cache_hits

    def _cache_path(self, params: Mapping[str, str]) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = json.dumps([self.base_url, HISTORY_PATH, dict(params)], sort_keys=True)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{digest}.json"

    def _fetch_tile(self, params: Mapping[str, str]) -> Tuple[Dict[str, Any], bool]:
        cache_path = self._cache_path(params)
        if cache_path is not None and cache_path.is_file():
            return json.loads(cache_path.read_text(encoding="utf-8")), True

        payload = self._post(HISTORY_PATH, params)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(cache_path)
        return payload, False

    def _post(self, path: str, params: Mapping[str, str]) -> Dict[str, Any]:
        url = self.base_url + path
        last_error = ""
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self.retry_wait_s * 2 ** (attempt - 1))
            try:
                response = self.http.post(url, data=dict(params), timeout=self.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    # ohsome streams its response, so a server-side timeout can
                    # end a 200 response mid-JSON.
                    last_error = "truncated JSON response (server-side timeout?)"
                    continue
            if response.status_code in _RETRYABLE_STATUS:
                last_error = f"HTTP {response.status_code}"
                continue
            raise OsmError(
                f"ohsome request failed: HTTP {response.status_code} for {url} "
                f"bbox={params.get('bboxes')}: {response.text[:300]}"
            )
        raise OsmError(
            f"ohsome request failed after {self.max_retries + 1} attempts for {url} "
            f"bbox={params.get('bboxes')}: {last_error}. If this keeps happening, "
            "lower the tile size in configs/impact.yaml."
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class OsmLayers:
    """Pre-event OSM layers for one AOI and event, with provenance."""

    snapshot: datetime
    aoi: BoundingBox
    fetch_extent: BoundingBox
    layers: Dict[str, gpd.GeoDataFrame]
    provenance: Dict[str, Any] = field(default_factory=dict)


def fetch_pre_event_osm(
    aoi: BoundingBox,
    event_date: date,
    snapshot: Optional[datetime] = None,
    buffer_km: Optional[float] = None,
    client: Optional[OhsomeClient] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> OsmLayers:
    """Fetch every OSM layer the impact analysis needs.

    Extents:

    * ``buildings``, ``bridges``, ``settlements``: the AOI only, because flood
      observations exist only there.
    * ``roads`` and ``hubs``: the AOI plus ``buffer_km``, so routes that leave
      the AOI and hubs outside it are still in the graph.
    """
    osm_config = (config or load_config("impact"))["osm"]
    client = client or OhsomeClient.from_config(osm_config)
    filters = osm_config["filters"]
    hub_filters: Mapping[str, str] = osm_config["hubs"]
    buffer_km = osm_config["buffer_km"] if buffer_km is None else buffer_km

    latest = client.latest_timestamp()
    snapshot = resolve_snapshot(event_date, latest, snapshot)
    extent = buffer_bbox(aoi, buffer_km)
    tile_deg = osm_config["tile_deg"]

    reports: Dict[str, LayerReport] = {}

    def fetch(name: str, osm_filter: str, bbox: BoundingBox, label: str, deg: float):
        frame, tiles, hits = client.fetch_layer(osm_filter, bbox, snapshot, deg)
        reports[name] = LayerReport(
            filter=osm_filter,
            extent=label,
            bbox=bbox.as_list(),
            tiles=tiles,
            cache_hits=hits,
            features=len(frame),
        )
        return frame

    layers: Dict[str, gpd.GeoDataFrame] = {}

    roads_aoi = fetch("roads_aoi", filters["roads"], aoi, "aoi", tile_deg)
    roads_ring = fetch("roads_buffer", filters["roads_buffer"], extent, "aoi+buffer", tile_deg)
    roads = pd.concat([roads_aoi, roads_ring], ignore_index=True)
    layers["roads"] = gpd.GeoDataFrame(
        roads.drop_duplicates("osm_id").reset_index(drop=True), crs="EPSG:4326"
    )

    layers["bridges"] = fetch("bridges", filters["bridges"], aoi, "aoi", tile_deg)
    layers["buildings"] = fetch(
        "buildings", filters["buildings"], aoi, "aoi", osm_config["building_tile_deg"]
    )
    point_tile_deg = osm_config.get("point_tile_deg", tile_deg)
    layers["settlements"] = fetch(
        "settlements", filters["settlements"], aoi, "aoi", point_tile_deg
    )

    hub_frames = []
    for group, hub_filter in hub_filters.items():
        frame = fetch(f"hubs_{group}", hub_filter, extent, "aoi+buffer", point_tile_deg)
        frame.insert(0, "hub_group", group)
        hub_frames.append(frame)
    layers["hubs"] = gpd.GeoDataFrame(
        pd.concat(hub_frames, ignore_index=True), crs="EPSG:4326"
    )

    provenance = {
        "source": "OpenStreetMap via the ohsome API",
        "endpoint": client.base_url + HISTORY_PATH,
        "snapshot": format_timestamp(snapshot),
        "ohsome_latest_available": format_timestamp(latest),
        "event_date": event_date.isoformat(),
        "aoi_bbox": aoi.as_list(),
        "buffer_km": buffer_km,
        "fetch_bbox": extent.as_list(),
        "layers": {name: vars(report) for name, report in reports.items()},
        "retrieved_at": format_timestamp(datetime.now(timezone.utc)),
        "attribution": OSM_ATTRIBUTION,
        "license": OSM_LICENSE,
    }
    return OsmLayers(
        snapshot=snapshot, aoi=aoi, fetch_extent=extent, layers=layers, provenance=provenance
    )


__all__ = [
    "OSM_ATTRIBUTION",
    "OhsomeClient",
    "OsmError",
    "OsmLayers",
    "buffer_bbox",
    "features_to_frame",
    "fetch_pre_event_osm",
    "parse_timestamp",
    "resolve_snapshot",
    "tile_bbox",
]
