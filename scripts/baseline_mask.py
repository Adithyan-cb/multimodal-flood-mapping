#!/usr/bin/env python3
"""Build a Sentinel-1 change-detection flood mask for an AOI and event date.

Finds a same-track pre/post Sentinel-1 RTC pair on Microsoft Planetary
Computer, reads it onto the AOI's UTM 10 m grid with the Copernicus DEM, and
applies the rule in ``floodmap.baseline.change_detection``. The output is a
class mask following the M4 contract, ready for ``scripts/impact.py --mask``.

Example
-------
``python scripts/baseline_mask.py --aoi-id trishuli \
    --aoi-bbox 85.25 27.85 85.40 28.20 --event-date 2026-08-26``

Writes, next to ``--out`` (default ``data/interim/<aoi-id>_flood_mask_s1baseline.tif``):

* the class mask GeoTIFF;
* ``*_layers.tif``: pre dB, post dB, change dB and slope, float32, nodata -9999;
* ``*_preview.png``: quick look (flood cyan, permanent water navy, nodata dark);
* ``*.json``: provenance (scenes, dates, orbit, thresholds, class areas).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin
from shapely.geometry import box

from floodmap.acquisition.aoi import BoundingBox
from floodmap.acquisition.planetary import (
    SOURCE_ATTRIBUTION,
    PlanetaryClient,
    PlanetaryError,
    TargetGrid,
    group_acquisitions,
    select_pair,
)
from floodmap.infrastructure.mask import SYNTHETIC_TAG, MaskSemantics
from floodmap.baseline.change_detection import (
    METHOD,
    BaselineRules,
    classify,
    rules_dict,
    slope_degrees,
)
from floodmap.utils.config import load_config
from floodmap.utils.provenance import REQUIRED_ATTRIBUTIONS

CONTINUOUS_NODATA = -9999.0  # C9


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
    parser.add_argument("--out", type=Path, help="output class mask path")
    return parser


def aoi_grid(aoi: BoundingBox, pixel_m: float) -> TargetGrid:
    """The AOI's UTM zone (C1), edges snapped to whole pixels (C2)."""
    series = gpd.GeoSeries([box(*aoi.as_list())], crs="EPSG:4326")
    utm = series.estimate_utm_crs()
    minx, miny, maxx, maxy = series.to_crs(utm).total_bounds
    minx, miny = math.floor(minx / pixel_m) * pixel_m, math.floor(miny / pixel_m) * pixel_m
    maxx, maxy = math.ceil(maxx / pixel_m) * pixel_m, math.ceil(maxy / pixel_m) * pixel_m
    return TargetGrid(
        crs=rasterio.crs.CRS.from_user_input(utm),
        transform=from_origin(minx, maxy, pixel_m, pixel_m),
        width=int(round((maxx - minx) / pixel_m)),
        height=int(round((maxy - miny) / pixel_m)),
    )


def _preview(classes, post_db, semantics, path: Path) -> None:
    stride = max(1, int(math.ceil(max(classes.shape) / 2000)))
    c = classes[::stride, ::stride]
    db = post_db[::stride, ::stride]
    gray = np.clip((np.nan_to_num(db, nan=-25.0) + 25.0) / 25.0 * 255.0, 0, 255).astype("uint8")
    rgb = np.stack([gray, gray, gray])
    for value, colour in (
        (semantics.flood, (0, 210, 255)),
        (semantics.permanent_water, (0, 40, 160)),
    ):
        hit = c == value
        for band in range(3):
            rgb[band][hit] = colour[band]
    dark = c == semantics.nodata
    for band, tint in enumerate((40, 0, 40)):
        rgb[band][dark] = (rgb[band][dark] * 0.3 + tint).astype("uint8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(
            path, "w", driver="PNG", width=c.shape[1], height=c.shape[0], count=3, dtype="uint8"
        ) as dst:
            dst.write(rgb)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    aoi = BoundingBox(
        min_lon=args.aoi_bbox[0],
        min_lat=args.aoi_bbox[1],
        max_lon=args.aoi_bbox[2],
        max_lat=args.aoi_bbox[3],
    )
    out = args.out or (
        _REPOSITORY_ROOT / "data" / "interim" / f"{args.aoi_id}_flood_mask_s1baseline.tif"
    )
    config = load_config("baseline")
    source, pair_cfg = config["source"], config["pair"]
    rules = BaselineRules.from_config(config)
    semantics = MaskSemantics.from_config()
    client = PlanetaryClient.from_config(source)
    aoi_geom = box(*aoi.as_list())

    start = args.event_date - timedelta(days=pair_cfg["max_pre_days"])
    end = args.event_date + timedelta(days=pair_cfg["max_post_days"] + 1)
    try:
        items = client.search(
            source["s1_collection"], aoi_geom, f"{start.isoformat()}/{end.isoformat()}"
        )
        pair = select_pair(
            group_acquisitions(items, aoi_geom),
            args.event_date,
            pair_cfg["max_post_days"],
            pair_cfg["max_pre_days"],
            pair_cfg["min_aoi_coverage"],
        )
    except PlanetaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Pair: {pair.pre.start:%Y-%m-%d %H:%M} -> {pair.post.start:%Y-%m-%d %H:%M} UTC, "
        f"{pair.post.orbit_state} relative orbit {pair.post.relative_orbit}, "
        f"{pair.days_apart} days apart"
    )

    grid = aoi_grid(aoi, float(config["grid"]["pixel_m"]))
    asset = rules.polarisation.lower()
    print(f"Grid: {grid.crs.to_string()}, {grid.width} x {grid.height} px. Reading scenes...")
    pre = client.read_mosaic(pair.pre.items, asset, source["s1_collection"], grid, True)
    post = client.read_mosaic(pair.post.items, asset, source["s1_collection"], grid, True)
    dem_items = client.search(source["dem_collection"], aoi_geom)
    if not dem_items:
        print("error: no Copernicus DEM tiles for this AOI", file=sys.stderr)
        return 1
    elevation = client.read_mosaic(dem_items, "data", source["dem_collection"], grid, False)
    slope = slope_degrees(elevation, float(config["grid"]["pixel_m"]))

    classes, layers = classify(pre, post, slope, rules, semantics)

    counts = np.bincount(classes.ravel(), minlength=256)
    km2 = float(config["grid"]["pixel_m"]) ** 2 / 1e6
    stats = {
        "flood_km2": round(counts[semantics.flood] * km2, 4),
        "permanent_water_km2": round(counts[semantics.permanent_water] * km2, 4),
        "nodata_fraction": round(counts[semantics.nodata] / classes.size, 6),
        "nodata_from_steep_slope_fraction": round(
            float(np.mean(np.nan_to_num(slope, nan=0.0) > rules.max_slope_deg)), 6
        ),
    }
    provenance = {
        "method": METHOD,
        "synthetic": False,
        "aoi_id": args.aoi_id,
        "aoi_bbox": aoi.as_list(),
        "event_date": args.event_date.isoformat(),
        "pair": pair.describe(),
        "polarisation": rules.polarisation,
        "rules": rules_dict(rules),
        "grid": grid.describe(),
        "dem_items": [i["id"] for i in dem_items],
        "source": {
            "stac_url": source["stac_url"],
            "s1_collection": source["s1_collection"],
            "dem_collection": source["dem_collection"],
            "attribution": SOURCE_ATTRIBUTION,
        },
        "stats": stats,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "attribution": list(REQUIRED_ATTRIBUTIONS) + [SOURCE_ATTRIBUTION],
        "limitations": [
            "Debris and mud can raise backscatter, so debris-covered ground is missed.",
            "Flooded vegetation and built-up areas can brighten (double bounce) and be missed.",
            "Wind-roughened water can look like land.",
            "Slopes above max_slope_deg are nodata, a proxy for radar shadow and layover.",
            "The map shows water at the post acquisition time, not the flood peak.",
        ],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        width=grid.width,
        height=grid.height,
        crs=grid.crs,
        transform=grid.transform,
        compress="deflate",
    )
    with rasterio.open(out, "w", count=1, dtype="uint8", nodata=semantics.nodata, **profile) as dst:
        dst.write(classes, 1)
        dst.update_tags(
            **{
                SYNTHETIC_TAG: "false",
                "FLOODMAP_METHOD": METHOD,
                "FLOODMAP_PRE": json.dumps(pair.pre.describe()),
                "FLOODMAP_POST": json.dumps(pair.post.describe()),
                "FLOODMAP_RULES": json.dumps(rules_dict(rules)),
            }
        )

    stem = out.with_suffix("")
    stack = [layers["pre_db"], layers["post_db"], layers["change_db"], slope]
    names = ["pre_vv_db", "post_vv_db", "change_vv_db", "slope_deg"]
    with rasterio.open(
        f"{stem}_layers.tif",
        "w",
        count=len(stack),
        dtype="float32",
        nodata=CONTINUOUS_NODATA,
        predictor=3,
        **profile,
    ) as dst:
        for band, (layer, name) in enumerate(zip(stack, names), start=1):
            dst.write(
                np.where(np.isfinite(layer), layer, CONTINUOUS_NODATA).astype("float32"), band
            )
            dst.set_band_description(band, name)
    _preview(classes, layers["post_db"], semantics, Path(f"{stem}_preview.png"))
    Path(f"{stem}.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"Flood mask written to {out}")
    print(
        f"  flood {stats['flood_km2']} km2, permanent water {stats['permanent_water_km2']} km2, "
        f"nodata {stats['nodata_fraction']:.1%} (steep slopes {stats['nodata_from_steep_slope_fraction']:.1%})"
    )
    print(f"  preview: {stem}_preview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
