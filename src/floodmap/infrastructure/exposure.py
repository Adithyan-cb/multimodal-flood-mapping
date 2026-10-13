"""Feature-level flood exposure of pre-event OSM infrastructure.

Scientific responsibility
-------------------------
For each building, road and bridge, measure how much of it lies on flood,
permanent-water and not-observed pixels of the class mask, and assign one of:

* ``EXPOSED``: enough of the feature lies on flood pixels (rules below).
* ``UNKNOWN``: not exposed on observed pixels, but enough of it lies on
  nodata pixels that a flood could be hidden there.
* ``NOT_EXPOSED``: the observed part shows no flood.
* ``NOT_OBSERVED``: entirely outside the mask.

"Exposed" means spatial overlap with the predicted flood class. It is not
evidence of physical damage (AGENTS.md §8).

Rules (configs/impact.yaml, ``exposure``)
-----------------------------------------
* Lines are sampled every ``line_sample_step_m`` metres. Each sample carries
  an equal share of the line's length and takes the class of the pixel it
  falls in. A line is EXPOSED when its flooded length reaches
  ``line_min_flooded_m``, or the whole line if it is shorter than that, so a
  fully flooded 6 m bridge still counts.
* Buildings are rasterised onto the mask grid with ``all_touched=True``, so
  any pixel the footprint touches counts. A building is EXPOSED when at least
  ``building_min_flood_px`` flood pixels touch it.
* UNKNOWN uses the same thresholds applied to nodata pixels.
* Only the flood class counts. Permanent water is measured and reported but
  never makes a feature exposed: every bridge crosses permanent water.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import box

from floodmap.infrastructure.mask import FloodMask
from floodmap.utils.config import load_config
from floodmap.utils.provenance import REQUIRED_ATTRIBUTIONS

EXPOSED = "EXPOSED"
UNKNOWN = "UNKNOWN"
NOT_EXPOSED = "NOT_EXPOSED"
NOT_OBSERVED = "NOT_OBSERVED"
STATUSES = (EXPOSED, UNKNOWN, NOT_EXPOSED, NOT_OBSERVED)

#: Sample value for points that fall outside the mask grid.
_OUTSIDE = -1

LINE_LAYERS = ("roads", "bridges")
POLYGON_LAYERS = ("buildings",)


@dataclass(frozen=True)
class ExposureRules:
    line_sample_step_m: float = 5.0
    line_min_flooded_m: float = 10.0
    building_min_flood_px: int = 1

    @classmethod
    def from_config(cls, impact_config: Optional[Mapping[str, Any]] = None) -> "ExposureRules":
        block = (impact_config or load_config("impact"))["exposure"]
        return cls(
            line_sample_step_m=float(block["line_sample_step_m"]),
            line_min_flooded_m=float(block["line_min_flooded_m"]),
            building_min_flood_px=int(block["building_min_flood_px"]),
        )


# ---------------------------------------------------------------------------
# Pixel lookup
# ---------------------------------------------------------------------------


def sample_classes(mask: FloodMask, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Class value of the pixel containing each point, ``-1`` outside the grid.

    Coordinates must already be in the mask CRS.
    """
    cols, rows = ~mask.transform @ (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
    cols = np.floor(cols).astype(np.int64)
    rows = np.floor(rows).astype(np.int64)
    height, width = mask.shape
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    out = np.full(cols.shape, _OUTSIDE, dtype=np.int16)
    out[inside] = mask.classes[rows[inside], cols[inside]]
    return out


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------


def _as_lines(geoms: np.ndarray) -> np.ndarray:
    """Polygons (closed ways returned as areas) become their outline."""
    geoms = np.asarray(geoms, dtype=object)
    is_polygon = np.isin(shapely.get_type_id(geoms), (3, 6))  # Polygon, MultiPolygon
    if is_polygon.any():
        geoms = geoms.copy()
        geoms[is_polygon] = shapely.boundary(geoms[is_polygon])
    return geoms


def measure_lines(geoms: gpd.GeoSeries, mask: FloodMask, step_m: float) -> pd.DataFrame:
    """Length of each line on each class, in metres.

    ``geoms`` must be in the mask CRS. Columns: ``length_m``, ``flooded_m``,
    ``permanent_water_m``, ``unknown_m`` (nodata) and ``outside_m``.
    """
    arr = _as_lines(geoms.values)
    lengths = np.nan_to_num(shapely.length(arr), nan=0.0)
    n_lines = len(arr)
    result = pd.DataFrame(
        {
            "length_m": lengths,
            "flooded_m": 0.0,
            "permanent_water_m": 0.0,
            "unknown_m": 0.0,
            "outside_m": lengths.copy(),
        },
        index=geoms.index,
    )
    if n_lines == 0:
        return result

    # Only sample lines that can touch the grid; the rest stay fully outside.
    grid = box(*mask.bounds)
    candidates = np.flatnonzero(shapely.intersects(arr, grid) & (lengths > 0))
    if candidates.size == 0:
        return result

    cand_len = lengths[candidates]
    n_samples = np.maximum(1, np.ceil(cand_len / step_m)).astype(np.int64)
    owner = np.repeat(np.arange(candidates.size), n_samples)
    first = np.cumsum(n_samples) - n_samples
    position = np.arange(n_samples.sum()) - np.repeat(first, n_samples)
    share = np.repeat(cand_len / n_samples, n_samples)
    distance = (position + 0.5) * share

    points = shapely.line_interpolate_point(arr[candidates][owner], distance)
    xy = shapely.get_coordinates(points)
    classes = sample_classes(mask, xy[:, 0], xy[:, 1])

    s = mask.semantics

    def total(selector: np.ndarray) -> np.ndarray:
        return np.bincount(owner, weights=share * selector, minlength=candidates.size)

    cols = result.columns
    result.iloc[candidates, cols.get_loc("flooded_m")] = total(classes == s.flood)
    result.iloc[candidates, cols.get_loc("permanent_water_m")] = total(classes == s.permanent_water)
    result.iloc[candidates, cols.get_loc("unknown_m")] = total(classes == s.nodata)
    result.iloc[candidates, cols.get_loc("outside_m")] = total(classes == _OUTSIDE)
    return result


def line_status(measured: pd.DataFrame, rules: ExposureRules) -> np.ndarray:
    length = measured["length_m"].to_numpy()
    # A line shorter than the threshold counts if it is flooded end to end.
    threshold = np.minimum(rules.line_min_flooded_m, length) - 1e-6
    fully_outside = measured["outside_m"].to_numpy() >= length - 1e-6
    return np.select(
        [
            fully_outside,
            measured["flooded_m"].to_numpy() >= threshold,
            measured["unknown_m"].to_numpy() >= threshold,
        ],
        [NOT_OBSERVED, EXPOSED, UNKNOWN],
        default=NOT_EXPOSED,
    )


# ---------------------------------------------------------------------------
# Polygons
# ---------------------------------------------------------------------------


def measure_polygons(geoms: gpd.GeoSeries, mask: FloodMask) -> pd.DataFrame:
    """Pixels touched by each polygon, per class.

    ``geoms`` must be in the mask CRS. Each polygon is rasterised on its own
    small window, so adjacent buildings sharing a pixel both see it.
    """
    s = mask.semantics
    height, width = mask.shape
    inverse = ~mask.transform
    rows_out = []
    for geom in geoms.values:
        counts = {"pixels": 0, "flood_px": 0, "permanent_water_px": 0, "unknown_px": 0}
        if geom is None or geom.is_empty:
            rows_out.append(counts)
            continue
        minx, miny, maxx, maxy = geom.bounds
        c_a, r_a = inverse @ (minx, maxy)
        c_b, r_b = inverse @ (maxx, miny)
        c0 = max(0, math.floor(min(c_a, c_b)))
        c1 = min(width, math.ceil(max(c_a, c_b)) + 1)
        r0 = max(0, math.floor(min(r_a, r_b)))
        r1 = min(height, math.ceil(max(r_a, r_b)) + 1)
        if c1 <= c0 or r1 <= r0:
            rows_out.append(counts)
            continue
        window_transform = mask.transform @ Affine.translation(c0, r0)
        footprint = rasterize(
            [(geom, 1)],
            out_shape=(r1 - r0, c1 - c0),
            transform=window_transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        ).astype(bool)
        values = mask.classes[r0:r1, c0:c1][footprint]
        counts["pixels"] = int(values.size)
        counts["flood_px"] = int(np.count_nonzero(values == s.flood))
        counts["permanent_water_px"] = int(np.count_nonzero(values == s.permanent_water))
        counts["unknown_px"] = int(np.count_nonzero(values == s.nodata))
        rows_out.append(counts)
    return pd.DataFrame(rows_out, index=geoms.index)


def polygon_status(measured: pd.DataFrame, rules: ExposureRules) -> np.ndarray:
    return np.select(
        [
            measured["pixels"].to_numpy() == 0,
            measured["flood_px"].to_numpy() >= rules.building_min_flood_px,
            measured["unknown_px"].to_numpy() >= rules.building_min_flood_px,
        ],
        [NOT_OBSERVED, EXPOSED, UNKNOWN],
        default=NOT_EXPOSED,
    )


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def assess_layer(
    frame: gpd.GeoDataFrame, mask: FloodMask, kind: str, rules: ExposureRules
) -> gpd.GeoDataFrame:
    """Add measurement columns and ``exposure_status`` to one layer.

    The returned frame keeps the input CRS and geometry; measurement happens on
    a copy reprojected to the mask CRS.
    """
    projected = frame.geometry.to_crs(mask.crs)
    if kind == "line":
        measured = measure_lines(projected, mask, rules.line_sample_step_m)
        status = line_status(measured, rules)
        measured = measured.round(1)
    elif kind == "polygon":
        measured = measure_polygons(projected, mask)
        status = polygon_status(measured, rules)
    else:
        raise ValueError(f"unknown layer kind {kind!r}")
    out = frame.copy()
    for column in measured.columns:
        out[column] = measured[column].to_numpy()
    out["exposure_status"] = status
    return out


def assess_exposure(
    layers: Mapping[str, gpd.GeoDataFrame], mask: FloodMask, rules: ExposureRules
) -> Dict[str, gpd.GeoDataFrame]:
    """Exposure for every infrastructure layer present in ``layers``."""
    assessed = {}
    for name in LINE_LAYERS:
        if name in layers:
            assessed[name] = assess_layer(layers[name], mask, "line", rules)
    for name in POLYGON_LAYERS:
        if name in layers:
            assessed[name] = assess_layer(layers[name], mask, "polygon", rules)
    return assessed


def summarize_exposure(
    assessed: Mapping[str, gpd.GeoDataFrame], mask: FloodMask, rules: ExposureRules
) -> Dict[str, Any]:
    """Counts and lengths per status, with the rules and mask they came from."""
    layers: Dict[str, Any] = {}
    for name, frame in assessed.items():
        counts = frame["exposure_status"].value_counts()
        entry: Dict[str, Any] = {
            "count": {status: int(counts.get(status, 0)) for status in STATUSES}
        }
        if "length_m" in frame:
            km = frame.groupby("exposure_status")["length_m"].sum() / 1000.0
            entry["length_km"] = {
                status: round(float(km.get(status, 0.0)), 3) for status in STATUSES
            }
            entry["flooded_length_km"] = round(float(frame["flooded_m"].sum()) / 1000.0, 3)
            if "highway" in frame:
                exposed = frame[frame["exposure_status"] == EXPOSED]
                by_class = exposed.groupby("highway")["flooded_m"].sum() / 1000.0
                entry["flooded_km_by_highway"] = {
                    str(k): round(float(v), 3)
                    for k, v in by_class.sort_values(ascending=False).items()
                }
        if name == "bridges":
            exposed = frame[frame["exposure_status"] == EXPOSED]
            entry["exposed"] = [
                {
                    "osm_id": row.osm_id,
                    "name": row.name if isinstance(row.name, str) else None,
                    "highway": row.highway,
                }
                for row in exposed.itertuples()
            ]
        layers[name] = entry

    return {
        "mask": mask.describe(),
        "rules": asdict(rules),
        "statuses": {
            EXPOSED: "overlaps the predicted flood class by at least the rule threshold",
            UNKNOWN: "no flood on observed pixels, but enough nodata that a flood could be hidden",
            NOT_EXPOSED: "observed pixels show no flood",
            NOT_OBSERVED: "outside the flood mask",
        },
        "interpretation": (
            "Exposure is spatial overlap with the predicted flood class, not evidence "
            "of physical damage (AGENTS.md §8)."
        ),
        "layers": layers,
        "attribution": list(REQUIRED_ATTRIBUTIONS),
    }


__all__ = [
    "EXPOSED",
    "ExposureRules",
    "NOT_EXPOSED",
    "NOT_OBSERVED",
    "STATUSES",
    "UNKNOWN",
    "assess_exposure",
    "assess_layer",
    "line_status",
    "measure_lines",
    "measure_polygons",
    "polygon_status",
    "sample_classes",
    "summarize_exposure",
]
