#!/usr/bin/env python3
"""Infrastructure exposure and road connectivity for an AOI and event date.

Stages:

* (a) fetch pre-event OSM layers and write them as GeoJSON;
* (b) with ``--mask``, measure exposure of buildings, roads and bridges to the
  flood class;
* (c) cut-off settlement analysis (not implemented yet).

Example
-------
``python scripts/impact.py --aoi-id trishuli-dev \
    --aoi-bbox 85.25 27.85 85.40 28.20 --event-date 2026-08-26 \
    --mask data/interim/trishuli-dev_flood_mask_SYNTHETIC.tif``

Outputs go to ``artifacts/impact/<aoi-id>/`` (git-ignored). GeoJSON is written
in EPSG:4326 so it opens directly in geojson.io or QGIS.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# Support running directly from a source checkout without requiring an
# editable install first. Installed users still resolve the same package.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPOSITORY_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import geopandas as gpd
from shapely.geometry import box

from floodmap.acquisition.aoi import BoundingBox
from floodmap.infrastructure.exposure import (
    NOT_OBSERVED,
    ExposureRules,
    assess_exposure,
    summarize_exposure,
)
from floodmap.infrastructure.mask import FloodMask, MaskError, load_flood_mask
from floodmap.infrastructure.osm import OsmError, OsmLayers, fetch_pre_event_osm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi-id", required=True, help="stable AOI identifier, used in paths")
    parser.add_argument(
        "--aoi-bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="AOI bounds in EPSG:4326",
    )
    parser.add_argument("--event-date", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--osm-snapshot",
        type=lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")),
        help="explicit OSM snapshot (UTC); default is the day before the event",
    )
    parser.add_argument("--buffer-km", type=float, help="override osm.buffer_km from config")
    parser.add_argument("--mask", type=Path, help="flood class mask GeoTIFF (enables stage b)")
    parser.add_argument("--out", type=Path, help="output directory")
    return parser


def write_osm_layers(osm: OsmLayers, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in osm.layers.items():
        (out_dir / f"osm_{name}.geojson").write_text(frame.to_json(drop_id=True), encoding="utf-8")
    (out_dir / "osm_provenance.json").write_text(
        json.dumps(osm.provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _check_mask_covers_aoi(mask: FloodMask, aoi: BoundingBox) -> None:
    footprint = gpd.GeoSeries([box(*mask.bounds)], crs=mask.crs).to_crs("EPSG:4326").iloc[0]
    if not footprint.intersects(box(*aoi.as_list())):
        raise MaskError(f"flood mask {mask.path} does not overlap the AOI {aoi.as_list()}")


def write_exposure(assessed, summary, out_dir: Path) -> None:
    for name, frame in assessed.items():
        observed = frame[frame["exposure_status"] != NOT_OBSERVED]
        (out_dir / f"exposure_{name}.geojson").write_text(
            observed.to_json(drop_id=True), encoding="utf-8"
        )
    (out_dir / "exposure_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _print_exposure(summary: dict) -> None:
    mask = summary["mask"]
    print()
    if mask["synthetic"]:
        print("!! SYNTHETIC MASK: development output only, never report these numbers !!")
    print(f"Flood mask: {mask['path']}")
    print(f"  flood {mask['flood_area_km2']} km2, nodata fraction {mask['nodata_fraction']}")
    print()
    print(f"  {'layer':<10} {'EXPOSED':>9} {'UNKNOWN':>9} {'NOT_EXP':>9} {'NOT_OBS':>9}")
    for name, entry in summary["layers"].items():
        c = entry["count"]
        print(
            f"  {name:<10} {c['EXPOSED']:>9} {c['UNKNOWN']:>9} "
            f"{c['NOT_EXPOSED']:>9} {c['NOT_OBSERVED']:>9}"
        )
    roads = summary["layers"].get("roads")
    if roads:
        print(f"\n  road length on flood pixels: {roads['flooded_length_km']} km")
        for highway, km in list(roads.get("flooded_km_by_highway", {}).items())[:6]:
            print(f"    {highway:<14} {km} km")


def _print_summary(osm: OsmLayers, out_dir: Path) -> None:
    print(f"OSM snapshot: {osm.provenance['snapshot']} (event {osm.provenance['event_date']})")
    print(f"AOI bbox:     {osm.aoi.as_list()}")
    print(
        f"Fetch bbox:   {[round(v, 4) for v in osm.fetch_extent.as_list()]} "
        f"(buffer {osm.provenance['buffer_km']} km)"
    )
    print()
    for name, report in osm.provenance["layers"].items():
        print(
            f"  {name:<16} {report['features']:>7} features  "
            f"[{report['extent']}, {report['tiles']} tiles, {report['cache_hits']} cached]"
        )
    hubs = osm.layers["hubs"]
    if len(hubs):
        print()
        for group, count in hubs["hub_group"].value_counts().items():
            print(f"  hubs/{group:<11} {count:>7}")
    print(f"\nWritten to {out_dir}")


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    aoi = BoundingBox(
        min_lon=args.aoi_bbox[0],
        min_lat=args.aoi_bbox[1],
        max_lon=args.aoi_bbox[2],
        max_lat=args.aoi_bbox[3],
    )
    out_dir = args.out or (_REPOSITORY_ROOT / "artifacts" / "impact" / args.aoi_id)

    try:
        osm = fetch_pre_event_osm(
            aoi, args.event_date, snapshot=args.osm_snapshot, buffer_km=args.buffer_km
        )
    except OsmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_osm_layers(osm, out_dir)
    _print_summary(osm, out_dir)

    if args.mask is not None:
        try:
            mask = load_flood_mask(args.mask)
            _check_mask_covers_aoi(mask, aoi)
        except MaskError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        rules = ExposureRules.from_config()
        assessed = assess_exposure(osm.layers, mask, rules)
        summary = summarize_exposure(assessed, mask, rules)
        summary["osm"] = {k: osm.provenance[k] for k in ("snapshot", "event_date", "aoi_bbox")}
        write_exposure(assessed, summary, out_dir)
        _print_exposure(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
