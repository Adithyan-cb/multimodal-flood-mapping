#!/usr/bin/env python3
"""Infrastructure exposure and road connectivity for an AOI and event date.

Stage (a), implemented: fetch pre-event OSM layers and write them as GeoJSON.
Later stages add flood-mask exposure and cut-off settlement analysis.

Example
-------
``python scripts/impact.py --aoi-id trishuli-dev \
    --aoi-bbox 85.25 27.85 85.40 28.20 --event-date 2026-08-26``

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

from floodmap.acquisition.aoi import BoundingBox
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
    parser.add_argument("--out", type=Path, help="output directory")
    return parser


def write_osm_layers(osm: OsmLayers, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in osm.layers.items():
        (out_dir / f"osm_{name}.geojson").write_text(frame.to_json(drop_id=True), encoding="utf-8")
    (out_dir / "osm_provenance.json").write_text(
        json.dumps(osm.provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _print_summary(osm: OsmLayers, out_dir: Path) -> None:
    print(f"OSM snapshot: {osm.provenance['snapshot']} (event {osm.provenance['event_date']})")
    print(f"AOI bbox:     {osm.aoi.as_list()}")
    print(f"Fetch bbox:   {[round(v, 4) for v in osm.fetch_extent.as_list()]} "
          f"(buffer {osm.provenance['buffer_km']} km)")
    print()
    for name, report in osm.provenance["layers"].items():
        print(f"  {name:<16} {report['features']:>7} features  "
              f"[{report['extent']}, {report['tiles']} tiles, {report['cache_hits']} cached]")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
