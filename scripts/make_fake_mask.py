#!/usr/bin/env python3
"""Build a SYNTHETIC flood class mask for developing the impact layer.

Development tool only. The mask is drawn from pre-event OSM river centrelines,
not from satellite data, and is tagged ``FLOODMAP_SYNTHETIC=true`` so every
result computed from it is labelled as synthetic. It must never be reported.

It follows the frozen M4 output contract so the real segmentation output can
replace it without code changes: single-band uint8, the AOI's UTM zone, 10 m
pixels, class ids from configs/segmentation.yaml, nodata 255.

Drawn classes:

* permanent water within ``--river-half-width-m`` of a river centreline;
* flood from there out to ``--flood-width-m``;
* one square nodata patch on the main river, so "unknown" gets exercised.

Example
-------
``python scripts/make_fake_mask.py --aoi-id trishuli-dev \
    --aoi-bbox 85.25 27.85 85.40 28.20 --event-date 2026-08-26``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Optional

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

from floodmap.acquisition.aoi import BoundingBox
from floodmap.infrastructure.mask import SYNTHETIC_TAG, MaskSemantics
from floodmap.infrastructure.osm import OhsomeClient, format_timestamp, resolve_snapshot
from floodmap.utils.config import load_config

RIVER_FILTER = "type:way and waterway=river"
PIXEL_M = 10.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi-id", required=True)
    parser.add_argument(
        "--aoi-bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
    )
    parser.add_argument("--event-date", type=date.fromisoformat, required=True)
    parser.add_argument("--river-half-width-m", type=float, default=15.0)
    parser.add_argument("--flood-width-m", type=float, default=120.0)
    parser.add_argument(
        "--nodata-patch-m",
        type=float,
        default=1500.0,
        help="side of the square nodata patch; 0 disables it",
    )
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    aoi = BoundingBox(
        min_lon=args.aoi_bbox[0],
        min_lat=args.aoi_bbox[1],
        max_lon=args.aoi_bbox[2],
        max_lat=args.aoi_bbox[3],
    )
    out = args.out or (
        _REPOSITORY_ROOT / "data" / "interim" / f"{args.aoi_id}_flood_mask_SYNTHETIC.tif"
    )
    semantics = MaskSemantics.from_config()

    osm_config = load_config("impact")["osm"]
    client = OhsomeClient.from_config(osm_config)
    snapshot = resolve_snapshot(args.event_date, client.latest_timestamp())
    rivers, _, _ = client.fetch_layer(RIVER_FILTER, aoi, snapshot, osm_config["tile_deg"])
    if rivers.empty:
        print(
            "error: no OSM rivers in this AOI; nothing to draw a fake flood along", file=sys.stderr
        )
        return 1

    # Grid: the AOI's UTM zone, 10 m pixels, edges snapped to whole pixels.
    aoi_series = gpd.GeoSeries([box(*aoi.as_list())], crs="EPSG:4326")
    utm = aoi_series.estimate_utm_crs()
    minx, miny, maxx, maxy = aoi_series.to_crs(utm).total_bounds
    minx, miny = math.floor(minx / PIXEL_M) * PIXEL_M, math.floor(miny / PIXEL_M) * PIXEL_M
    maxx, maxy = math.ceil(maxx / PIXEL_M) * PIXEL_M, math.ceil(maxy / PIXEL_M) * PIXEL_M
    width, height = int((maxx - minx) / PIXEL_M), int((maxy - miny) / PIXEL_M)
    transform = from_origin(minx, maxy, PIXEL_M, PIXEL_M)

    lines = rivers.to_crs(utm).geometry
    classes = np.full((height, width), semantics.no_water, dtype="uint8")
    for zone, value in (
        (lines.buffer(args.flood_width_m).union_all(), semantics.flood),
        (lines.buffer(args.river_half_width_m).union_all(), semantics.permanent_water),
    ):
        burned = rasterize(
            [(zone, 1)], out_shape=classes.shape, transform=transform, fill=0, dtype="uint8"
        ).astype(bool)
        classes[burned] = value

    patch = None
    if args.nodata_patch_m > 0:
        # Ways come back unclipped, so measure only the part on the grid.
        on_grid = lines.intersection(box(minx, miny, maxx, maxy))
        main_river = on_grid.iloc[int(np.argmax(on_grid.length.to_numpy()))]
        centre = main_river.interpolate(0.3, normalized=True)
        half = args.nodata_patch_m / 2
        patch = box(centre.x - half, centre.y - half, centre.x + half, centre.y + half)
        burned = rasterize(
            [(patch, 1)], out_shape=classes.shape, transform=transform, fill=0, dtype="uint8"
        ).astype(bool)
        classes[burned] = semantics.nodata

    out.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
        crs=utm,
        transform=transform,
        nodata=semantics.nodata,
        compress="deflate",
    )
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(classes, 1)
        dst.update_tags(
            **{
                SYNTHETIC_TAG: "true",
                "FLOODMAP_DESCRIPTION": (
                    "SYNTHETIC development mask drawn from pre-event OSM rivers. "
                    "Not derived from satellite data. Never report results from it."
                ),
                "FLOODMAP_RIVER_SOURCE": f"OSM via ohsome, filter '{RIVER_FILTER}', "
                f"snapshot {format_timestamp(snapshot)}",
                "FLOODMAP_PARAMS": json.dumps(
                    {
                        "river_half_width_m": args.river_half_width_m,
                        "flood_width_m": args.flood_width_m,
                        "nodata_patch_m": args.nodata_patch_m,
                        "aoi_bbox": aoi.as_list(),
                    }
                ),
            }
        )

    counts = np.bincount(classes.ravel(), minlength=256)
    km2 = PIXEL_M * PIXEL_M / 1e6
    print(f"SYNTHETIC mask written to {out}")
    print(f"  CRS {utm.to_string()}, {width} x {height} px at {PIXEL_M:g} m")
    print(f"  rivers used: {len(rivers)} OSM ways")
    print(
        f"  flood {counts[semantics.flood] * km2:.2f} km2, "
        f"permanent water {counts[semantics.permanent_water] * km2:.2f} km2, "
        f"nodata {counts[semantics.nodata] * km2:.2f} km2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
