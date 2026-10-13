"""Deterministic generator for a synthetic dashboard demo bundle.

Why this exists
---------------
The dashboard has nothing real to show until the pipeline produces artifacts.
Rather than leave the UI empty or invent numbers at render time, this module
builds a small, fully deterministic placeholder bundle — rasters, vectors and
provenance records — so the app and its integration tests exercise the exact
contract that real pipeline output will one day satisfy (``AGENTS.md`` §16,
``architecture.md`` §15).

Every artifact produced here is explicitly synthetic:
:class:`~floodmap.utils.provenance.ArtifactProvenance` notes and the manifest
``is_synthetic`` flag both say so, and the dashboard renders an unmissable
banner whenever that flag is set. Nothing in this module is or claims to be a
real measurement: it is plumbing for development, demonstration and tests.

Where the output goes
---------------------
The generator writes a self-contained AOI bundle under ``output_dir`` (the
application default is ``artifacts/dashboard_demo/``, which is gitignored). The
bundle layout matches the contract in ``src/floodmap/dashboard/contract.py``:

.. code-block:: text

    <output_dir>/
        dashboard_manifest.json
        layers/
            <stem>.tif | .geojson
            <stem>_provenance.yaml

No committed binary is involved; the descriptor is regenerable from the fixed
seed. ``data/samples/dashboard_demo/`` holds only this generator's contract,
not rendered pixels (``AGENTS.md`` §15, and ``*.tif`` is globally gitignored).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from numpy.random import Generator, default_rng

from ..utils.provenance import (
    REQUIRED_ATTRIBUTIONS,
    ArtifactProvenance,
    ArtifactType,
    ProductionInput,
    Unknown,
    UnknownReason,
)
from .contract import (
    MANIFEST_SCHEMA,
    DashboardLayer,
    DashboardManifest,
    EvidenceTier,
    LayerFormat,
)

__all__ = ["build_demo_bundle"]

#: Default landing directory for the synthetic bundle. Kept under ``artifacts/``
#: so git never sees the rasters.
DEFAULT_OUTPUT_DIR = Path("artifacts/dashboard_demo")

#: Preferred grid size. Small enough to be instant, large enough that the
#: synthetic river and extent are visible on a 10 m pixel.
GRID = 128
RESOLUTION_M = 10.0
UTM_EPSG = "EPSG:32645"
#: Approximate location of the Trishuli valley, used only to place the demo in
#: a plausible UTM 45N footprint. Easting/northing of the grid origin.
ORIGIN_X, ORIGIN_Y = 325_000.0, 3_100_000.0
#: Approximate lon/lat of the grid origin, used to render GeoJSON in WGS84.
LON0, LAT0 = 85.1, 28.0

SYNTHETIC_NOTE = "SYNTHETIC PLACEHOLDER — not a real measurement or result."
SYNTHETIC_LIMITATION = (
    "Synthetic placeholder data for development/testing; carries no scientific meaning."
)


# ---------------------------------------------------------------------------
# Synthetic scene synthesis
# ---------------------------------------------------------------------------


def _river_mask(shape: Sequence[int], rng: Generator) -> np.ndarray:
    """A sinuous river band down the middle third of the grid."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    centre_x = width / 2.0 + 14.0 * np.sin((yy / height) * 3 * np.pi)
    channel = np.abs(xx - centre_x) <= 1.5
    valley = np.abs(xx - centre_x) <= 9.0
    flooded = valley & ~channel
    return channel.astype("float32") * 1.0 + flooded.astype("float32") * 2.0


def _s1_amplitude(channel_map: np.ndarray, rng: Generator) -> np.ndarray:
    """VV- and VH-like amplitude with speckle, water visibly darker.

    channel_map: 0=land, 1=permanent water, 2=flooded water. Values are
    synthetic but honour the SAR reading that open water backscatters low while
    urban/rough terrain is bright.
    """
    height, width = channel_map.shape
    land = rng.gamma(1.6, 0.18, size=(height, width)).astype("float32")
    water = rng.gamma(0.7, 0.045, size=(height, width)).astype("float32")
    vv = np.where(channel_map > 0, water, land)
    vh = rng.gamma(0.55, 0.03, size=(height, width)).astype("float32")
    # Flooded valley water is darker still in VH, giving the extent contrast.
    vh = np.where(channel_map > 1, vh * 0.7, vh)
    return np.stack([vv, vh], axis=0)


def _optics(channel_map: np.ndarray) -> np.ndarray:
    """Three synthetic 'optical-like' bands for contrast only."""
    height, width = channel_map.shape
    land = np.full((height, width), 0.55, dtype="float32")
    water = np.full((height, width), 0.12, dtype="float32")
    base = np.where(channel_map > 0, water, land)
    red = base * 1.3
    green = base
    blue = base * 1.6
    return np.stack([red, green, blue], axis=0)


def _flood_water_extent(channel_map: np.ndarray) -> np.ndarray:
    """Predicted flood water mask: 1 where flooded, 0 elsewhere."""
    return (channel_map > 1).astype("uint8")


def _confidence(channel_map: np.ndarray) -> np.ndarray:
    """High confidence in the flooded core, tapering toward the extent boundary.

    Uses a few morphological clears from the boundary edge, so pixels at the
    rim read ~0.6 while the interior reads ~0.95. No third-party morphology
    dependency: eroded with 4-neighbour shifts, which is plenty for a demo.
    """
    flood = channel_map > 1

    def _erode4(bin_: np.ndarray) -> np.ndarray:
        return (
            bin_
            & np.roll(bin_, 1, axis=0)
            & np.roll(bin_, -1, axis=0)
            & np.roll(bin_, 1, axis=1)
            & np.roll(bin_, -1, axis=1)
        )

    conf = np.where(flood, 0.6, 0.0).astype("float32")
    body = flood.copy()
    for step in range(1, 5):
        body = _erode4(body)
        conf[body] = 0.6 + 0.35 * (step / 4.0)
    return conf


# ---------------------------------------------------------------------------
# GeoJSON helpers
# ---------------------------------------------------------------------------


def _to_lonlat(x: float, y: float) -> List[float]:
    """Approximate grid (px) -> WGS84. Fine for a synthetic demo."""
    dx = (x - GRID / 2.0) * RESOLUTION_M
    dy = (y - GRID / 2.0) * RESOLUTION_M
    lat = LAT0 - dy / 110_500.0
    lon = LON0 + dx / (111_320.0 * np.cos(np.deg2rad(LAT0)))
    return [round(lon, 6), round(lat, 6)]


def _feature(geom_type: str, coordinates: Any, properties: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": geom_type, "coordinates": coordinates},
        "properties": properties,
    }


def _buildings(channel_map: np.ndarray, rng: Generator) -> Dict[str, Any]:
    ys, xs = np.where(channel_map == 0)
    features = []
    for idx in range(14):
        pick = int(rng.integers(0, len(xs)))
        px, py = float(xs[pick]), float(ys[pick])
        half = float(rng.integers(4, 9)) / 2.0  # 4-9 px buildings
        corners = [
            [px - half, py - half],
            [px + half, py - half],
            [px + half, py + half],
            [px - half, py + half],
            [px - half, py - half],
        ]
        ny0, ny1 = max(0, int(py - 3)), min(int(py + 3), GRID)
        nx0, nx1 = max(0, int(px - 3)), min(int(px + 3), GRID)
        near_water = bool(np.any(channel_map[ny0:ny1, nx0:nx1] > 1))
        features.append(
            _feature(
                "Polygon",
                [[_to_lonlat(x, y) for x, y in corners]],
                {
                    "id": f"b{idx:03d}",
                    "intersects_flood_extent": near_water,
                },
            )
        )
    return {"type": "FeatureCollection", "features": features}


def _roads(channel_map: np.ndarray) -> Dict[str, Any]:
    height, width = channel_map.shape
    centre = width / 2.0 + 14.0 * np.sin((np.arange(height) / height) * 3 * np.pi)
    left_bank = [(float(i), centre[i] - 6) for i in range(0, height, 4)]
    right_bank = [(float(i), centre[i] + 6) for i in range(0, height, 4)]
    features = [
        _feature(
            "LineString",
            [_to_lonlat(x, y) for y, x in left_bank],
            {"id": "r-l", "highway": "secondary", "potentially_disrupted": True},
        ),
        _feature(
            "LineString",
            [_to_lonlat(x, y) for y, x in right_bank],
            {"id": "r-r", "highway": "secondary", "potentially_disrupted": True},
        ),
    ]
    # One bridge crossing at mid-grid (vertical line of columns).
    mid = height // 2
    bridge = [[float(mid), centre[mid] - 3], [float(mid), centre[mid] + 3]]
    features.append(
        _feature(
            "LineString",
            [_to_lonlat(x, y) for y, x in bridge],
            {"id": "r-b", "highway": "bridge", "potentially_disrupted": False},
        )
    )
    return {"type": "FeatureCollection", "features": features}


def _settlements(channel_map: np.ndarray, rng: Generator) -> Dict[str, Any]:
    names = ["Sano Dovan", "Thulo Dovan", "Ghyalchok"]
    ys, xs = np.where(channel_map == 0)
    features = []
    for idx, name in enumerate(names):
        pick = int(rng.integers(0, len(xs)))
        y, x = float(ys[pick]), float(xs[pick])
        within = np.any(
            channel_map[max(0, int(y - 5)) : int(y + 5), max(0, int(x - 5)) : int(x + 5)] > 1
        )
        features.append(
            _feature(
                "Point",
                _to_lonlat(x, y),
                {"name": name, "potentially_cut_off": bool(within), "settlement_type": "village"},
            )
        )
    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Provenance + manifest
# ---------------------------------------------------------------------------


def _provenance(
    *,
    artifact_type: ArtifactType,
    layer_id_suffix: str,
    production_inputs: Sequence[ProductionInput],
    limitations: Sequence[str],
) -> ArtifactProvenance:
    """A synthetic provenance record, unambiguously labelled as such."""
    return ArtifactProvenance(
        artifact_type=artifact_type,
        artifact_id=f"demo-{layer_id_suffix}",
        aoi="trishuli-demo (synthetic)",
        aoi_crs=UTM_EPSG,
        event_date="2026-08-15",
        production_inputs=list(production_inputs),
        limitations=list(limitations) + [SYNTHETIC_LIMITATION],
        notes=SYNTHETIC_NOTE,
    )


# ---------------------------------------------------------------------------
# Bundle writer
# ---------------------------------------------------------------------------


def _write_geotiff(
    path: Path, data: np.ndarray, *, band_names: Sequence[str], nodata: float | None = None
) -> None:
    import rasterio

    height, width = data.shape[1:]
    transform = rasterio.transform.from_origin(ORIGIN_X, ORIGIN_Y, RESOLUTION_M, RESOLUTION_M)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": int(data.shape[0]),
        "dtype": str(data.dtype),
        "crs": UTM_EPSG,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for idx, name in enumerate(band_names, start=1):
            dst.set_band_description(idx, name)
        dst.update_tags(provenance_note=SYNTHETIC_NOTE)


def _write_geojson(path: Path, collection: Dict[str, Any]) -> None:
    path.write_text(json.dumps(collection, indent=2) + "\n", encoding="utf-8")


def build_demo_bundle(output_dir: Path = DEFAULT_OUTPUT_DIR, *, seed: int = 2026) -> Path:
    """Generate a synthetic dashboard bundle and return the manifest path.

    Deterministic for a given seed. The bundle is a valid contract bundle:
    ``load_manifest`` in :mod:`floodmap.dashboard` can read it directly.
    """
    rng = default_rng(seed)
    output_dir = Path(output_dir)
    layers_dir = output_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    river = _river_mask((GRID, GRID), rng)
    s1 = _s1_amplitude(river, rng)
    s2 = _optics(river)
    flood_mask = _flood_water_extent(river)
    confidence = _confidence(river)

    stems: Dict[str, Any] = {
        "s1_pre": {"data": s1, "bands": ["VV", "VH"]},
        "s1_post": {"data": s1, "bands": ["VV", "VH"]},
        "s2_post": {"data": s2, "bands": ["R", "G", "B"]},
        "flood_water_extent": {"data": flood_mask[np.newaxis, ...], "bands": ["flood_water"]},
        "flood_water_confidence": {"data": confidence[np.newaxis, ...], "bands": ["confidence"]},
    }
    for stem, spec in stems.items():
        _write_geotiff(layers_dir / f"{stem}.tif", spec["data"], band_names=spec["bands"])

    buildings = _buildings(river, rng)
    roads = _roads(river)
    settlements = _settlements(river, rng)
    _write_geojson(layers_dir / "buildings_exposed.geojson", buildings)
    _write_geojson(layers_dir / "roads.geojson", roads)
    _write_geojson(layers_dir / "settlements.geojson", settlements)

    layer_specs: List[Dict[str, Any]] = [
        {
            "id": "s1_pre",
            "title": "Pre-event Sentinel-1 amplitude (synthetic)",
            "artifact_type": ArtifactType.PREPROCESSED_RASTER,
            "tier": EvidenceTier.OBSERVED_DERIVED,
            "fmt": LayerFormat.GEOTIFF,
            "path": "layers/s1_pre.tif",
            "inputs": [ProductionInput.SENTINEL1],
            "limitations": ["Synthetic scene; no real acquisition timestamp."],
        },
        {
            "id": "s1_post",
            "title": "Post-event Sentinel-1 amplitude (synthetic)",
            "artifact_type": ArtifactType.PREPROCESSED_RASTER,
            "tier": EvidenceTier.OBSERVED_DERIVED,
            "fmt": LayerFormat.GEOTIFF,
            "path": "layers/s1_post.tif",
            "inputs": [ProductionInput.SENTINEL1],
            "limitations": ["Synthetic scene; no real acquisition timestamp."],
        },
        {
            "id": "s2_post",
            "title": "Post-event optical composite (synthetic)",
            "artifact_type": ArtifactType.PREPROCESSED_RASTER,
            "tier": EvidenceTier.OBSERVED_DERIVED,
            "fmt": LayerFormat.GEOTIFF,
            "path": "layers/s2_post.tif",
            "inputs": [ProductionInput.SENTINEL2],
            "limitations": ["Synthetic scene; no real acquisition timestamp."],
        },
        {
            "id": "flood_water_extent",
            "title": "Predicted flood water extent",
            "artifact_type": ArtifactType.SEGMENTATION_PREDICTION,
            "tier": EvidenceTier.MODEL_PREDICTION,
            "fmt": LayerFormat.GEOTIFF,
            "path": "layers/flood_water_extent.tif",
            "inputs": [ProductionInput.SENTINEL1],
            "limitations": [
                "Model prediction, not an observation; carries model error.",
                "Labels flood water only; debris/sediment is not labelled (docs/m4-architecture-decision.md §6).",
            ],
        },
        {
            "id": "flood_water_confidence",
            "title": "Flood water prediction confidence",
            "artifact_type": ArtifactType.SEGMENTATION_PREDICTION,
            "tier": EvidenceTier.MODEL_PREDICTION,
            "fmt": LayerFormat.GEOTIFF,
            "path": "layers/flood_water_confidence.tif",
            "inputs": [ProductionInput.SENTINEL1],
            "limitations": ["Model-derived; lower confidence near the extent edge."],
        },
        {
            "id": "buildings_exposed",
            "title": "Buildings intersected by predicted extent",
            "artifact_type": ArtifactType.INFRASTRUCTURE_EXPOSURE,
            "tier": EvidenceTier.SPATIAL_INFERENCE,
            "fmt": LayerFormat.GEOJSON,
            "path": "layers/buildings_exposed.geojson",
            "inputs": [ProductionInput.SENTINEL1, ProductionInput.OSM_PRE_EVENT],
            "limitations": [
                "Spatial intersection only; intersection is exposure, not damage.",
                "Per-building statements carry low confidence at ~10 m resolution.",
            ],
        },
        {
            "id": "roads",
            "title": "Road segments intersecting predicted extent",
            "artifact_type": ArtifactType.ROAD_DISRUPTION,
            "tier": EvidenceTier.SPATIAL_INFERENCE,
            "fmt": LayerFormat.GEOJSON,
            "path": "layers/roads.geojson",
            "inputs": [ProductionInput.SENTINEL1, ProductionInput.OSM_PRE_EVENT],
            "limitations": [
                "Intersection is evidence of exposure, not proof of physical damage.",
                "Depends on OSM pre-event completeness.",
            ],
        },
        {
            "id": "settlements",
            "title": "Settlements potentially cut off",
            "artifact_type": ArtifactType.SETTLEMENT_CONNECTIVITY,
            "tier": EvidenceTier.NETWORK_INFERENCE,
            "fmt": LayerFormat.GEOJSON,
            "path": "layers/settlements.geojson",
            "inputs": [ProductionInput.SENTINEL1, ProductionInput.OSM_PRE_EVENT],
            "limitations": [
                "Graph-based inference; depends on segmentation, OSM completeness and the disruption rule.",
                "'Potentially cut off' is the strongest claim this layer supports.",
            ],
        },
    ]

    prov_by_id = {
        spec["id"]: _provenance(
            artifact_type=spec["artifact_type"],
            layer_id_suffix=spec["id"],
            production_inputs=spec["inputs"],
            limitations=spec["limitations"],
        )
        for spec in layer_specs
    }
    for layer_id, prov in prov_by_id.items():
        (layers_dir / f"{layer_id}_provenance.yaml").write_text(prov.to_yaml(), encoding="utf-8")

    # One explicitly-absent layer: the DEM flood-path trace has not been run for
    # the demo, and the contract requires that absence to be visible with a reason.
    unavailable = DashboardLayer(
        layer_id="flood_path_trace",
        title="DEM-derived flood path trace",
        artifact_type=ArtifactType.FLOOD_PATH_TRACE,
        evidence_tier=EvidenceTier.OBSERVED_DERIVED,
        format=LayerFormat.GEOJSON,
        available=False,
        unavailable_reason=Unknown(
            reason=UnknownReason.NOT_AVAILABLE_FROM_SOURCE,
            note="Flood-path tracing not implemented for the synthetic demo AOI.",
        ),
    )

    layers = []
    for spec in layer_specs:
        is_geotiff = spec["fmt"] is LayerFormat.GEOTIFF
        layers.append(
            DashboardLayer(
                layer_id=spec["id"],
                title=spec["title"],
                artifact_type=spec["artifact_type"],
                evidence_tier=spec["tier"],
                format=spec["fmt"],
                path=spec["path"],
                crs=UTM_EPSG if is_geotiff else "EPSG:4326",
                bbox=_bbox_for_layer() if is_geotiff else _bbox_wgs84(),
                provenance_path=f"layers/{spec['id']}_provenance.yaml",
            )
        )
    layers.append(unavailable)

    manifest = DashboardManifest(
        schema=MANIFEST_SCHEMA,
        aoi_id="trishuli-demo",
        event_name="Trishuli Valley flood event — SYNTHETIC DEMO",
        event_date="2026-08-15",
        is_synthetic=True,
        attribution=list(REQUIRED_ATTRIBUTIONS),
        layers=layers,
    )
    manifest_path = output_dir / "dashboard_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(by_alias=True, mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _bbox_for_layer() -> List[float]:
    """Grid footprint in UTM easting/northing."""
    x0, y0 = ORIGIN_X, ORIGIN_Y - GRID * RESOLUTION_M
    x1, y1 = ORIGIN_X + GRID * RESOLUTION_M, ORIGIN_Y
    return [x0, y0, x1, y1]


def _bbox_wgs84() -> List[float]:
    """Grid footprint in approximate WGS84 lon/lat for the GeoJSON layers."""
    minx, miny = _to_lonlat(0.0, GRID - 1.0)
    maxx, maxy = _to_lonlat(GRID - 1.0, 0.0)
    return [minx, miny, maxx, maxy]
