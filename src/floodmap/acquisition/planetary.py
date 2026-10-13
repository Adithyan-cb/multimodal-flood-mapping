"""Sentinel-1 RTC and Copernicus DEM access through Microsoft Planetary Computer.

Scientific responsibility
-------------------------
Find a pre/post Sentinel-1 pair for any AOI and event date on the SAME
relative orbit and pass direction (organizer guidance, AGENTS.md §4), and read
it onto the analysis grid with provenance.

Source
------
``sentinel-1-rtc``: Sentinel-1 IW GRD, radiometrically terrain corrected to
gamma0 (linear power) by Catalyst with the Copernicus DEM, 10 m, UTM. Licence
CC-BY-4.0 (Catalyst, Microsoft); it is modified Copernicus Sentinel data.
``cop-dem-glo-30``: Copernicus DEM GLO-30. Both are read anonymously with a
short-lived SAS token from the Planetary Computer token endpoint.

Pair rule
---------
* The event day itself is excluded from both sides, because the event's time
  of day is unknown.
* Post: the earliest acquisition after the event day, within
  ``max_post_days``, covering at least ``min_aoi_coverage`` of the AOI.
* Pre: the latest acquisition before the event day, within ``max_pre_days``,
  on the same relative orbit and pass direction as the post image, with the
  same coverage requirement.
* An acquisition is every slice of one pass on one day, mosaicked, because an
  AOI can straddle two slices.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from floodmap.utils.config import project_root

SOURCE_ATTRIBUTION = (
    "Sentinel-1 RTC: Catalyst and Microsoft Planetary Computer, CC-BY-4.0, "
    "derived from Copernicus Sentinel-1 data."
)


class PlanetaryError(RuntimeError):
    """Catalog access failed or no usable image pair exists."""


# ---------------------------------------------------------------------------
# Target grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetGrid:
    crs: rasterio.crs.CRS
    transform: rasterio.Affine
    width: int
    height: int

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        left, top = self.transform.c, self.transform.f
        return (
            left,
            top + self.transform.e * self.height,
            left + self.transform.a * self.width,
            top,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "crs": self.crs.to_string(),
            "transform": list(self.transform)[:6],
            "width": self.width,
            "height": self.height,
        }


# ---------------------------------------------------------------------------
# Acquisitions and pair selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Acquisition:
    """All slices of one Sentinel-1 pass over the AOI on one day."""

    platform: str
    relative_orbit: int
    orbit_state: str
    day: date
    start: datetime
    items: Tuple[Mapping[str, Any], ...]
    coverage: float

    @property
    def item_ids(self) -> List[str]:
        return [item["id"] for item in self.items]

    def describe(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "relative_orbit": self.relative_orbit,
            "orbit_state": self.orbit_state,
            "start": self.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "aoi_coverage": round(self.coverage, 4),
            "item_ids": self.item_ids,
        }


@dataclass(frozen=True)
class S1Pair:
    pre: Acquisition
    post: Acquisition

    @property
    def days_apart(self) -> float:
        return round((self.post.start - self.pre.start).total_seconds() / 86400.0, 2)

    def describe(self) -> Dict[str, Any]:
        return {
            "pre": self.pre.describe(),
            "post": self.post.describe(),
            "days_apart": self.days_apart,
        }


def _parse_dt(text: str) -> datetime:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def group_acquisitions(items: Sequence[Mapping[str, Any]], aoi) -> List[Acquisition]:
    """Group STAC items into passes and measure how much of the AOI each covers."""
    groups: Dict[Tuple, List[Mapping[str, Any]]] = {}
    for item in items:
        p = item["properties"]
        start = _parse_dt(p["datetime"])
        key = (p.get("platform"), int(p["sat:relative_orbit"]), p["sat:orbit_state"], start.date())
        groups.setdefault(key, []).append(item)

    acquisitions = []
    for (platform, orbit, state, day), members in groups.items():
        footprint = unary_union([shape(m["geometry"]) for m in members])
        coverage = footprint.intersection(aoi).area / aoi.area if aoi.area else 0.0
        acquisitions.append(
            Acquisition(
                platform=platform,
                relative_orbit=orbit,
                orbit_state=state,
                day=day,
                start=min(_parse_dt(m["properties"]["datetime"]) for m in members),
                items=tuple(sorted(members, key=lambda m: m["id"])),
                coverage=coverage,
            )
        )
    return sorted(acquisitions, key=lambda a: a.start)


def select_pair(
    acquisitions: Sequence[Acquisition],
    event_date: date,
    max_post_days: int,
    max_pre_days: int,
    min_coverage: float,
) -> S1Pair:
    """Apply the pair rule in the module docstring.

    Raises
    ------
    PlanetaryError
        With a list of what was available, so a judge-chosen AOI/date that has
        no usable pair fails with an explanation rather than a wrong map.
    """
    usable = [a for a in acquisitions if a.coverage >= min_coverage]
    posts = [a for a in usable if event_date < a.day <= event_date + timedelta(days=max_post_days)]
    for post in posts:
        pres = [
            a
            for a in usable
            if a.relative_orbit == post.relative_orbit
            and a.orbit_state == post.orbit_state
            and event_date - timedelta(days=max_pre_days) <= a.day < event_date
        ]
        if pres:
            return S1Pair(pre=max(pres, key=lambda a: a.start), post=post)

    seen = ", ".join(
        f"{a.day} {a.orbit_state[:4]} rel{a.relative_orbit} cov{a.coverage:.2f}"
        for a in acquisitions
    )
    raise PlanetaryError(
        f"no same-track Sentinel-1 pair around {event_date} (post within {max_post_days} d, "
        f"pre within {max_pre_days} d, AOI coverage >= {min_coverage}). Available: "
        f"{seen or 'none'}"
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PlanetaryClient:
    """STAC search, SAS signing and windowed reads, with an on-disk cache."""

    def __init__(
        self,
        stac_url: str,
        token_url: str,
        timeout_s: float = 120,
        cache_dir: Optional[Path] = None,
        http: Any = None,
    ) -> None:
        self.stac_url = stac_url.rstrip("/")
        self.token_url = token_url.rstrip("/")
        self.timeout_s = timeout_s
        self.cache_dir = cache_dir
        self.http = http if http is not None else requests
        self._tokens: Dict[str, Tuple[str, float]] = {}

    @classmethod
    def from_config(cls, source: Mapping[str, Any]) -> "PlanetaryClient":
        cache_dir = source.get("cache_dir")
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            if not cache_dir.is_absolute():
                cache_dir = project_root() / cache_dir
        return cls(
            stac_url=source["stac_url"],
            token_url=source["token_url"],
            timeout_s=source.get("timeout_s", 120),
            cache_dir=cache_dir,
        )

    def search(
        self, collection: str, geometry, datetime_range: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {
            "collections": [collection],
            "intersects": mapping(geometry),
            "limit": limit,
        }
        if datetime_range:
            body["datetime"] = datetime_range
        url = self.stac_url + "/search"
        features: List[Dict[str, Any]] = []
        for _ in range(20):  # page cap
            response = self.http.post(url, json=body, timeout=self.timeout_s)
            if response.status_code != 200:
                raise PlanetaryError(
                    f"STAC search failed: HTTP {response.status_code}: {response.text[:300]}"
                )
            payload = response.json()
            features.extend(payload.get("features", []))
            nxt = next((l for l in payload.get("links", []) if l.get("rel") == "next"), None)
            if not nxt or not nxt.get("body"):
                break
            body = nxt["body"]
        return features

    def token(self, href: str) -> str:
        """SAS token for the storage account and container that hold ``href``.

        Tokens are scoped to storage, not to STAC collections: the DEM's
        collection token was refused (HTTP 403) for its own files.
        """
        host, container = href.split("://", 1)[1].split("/")[:2]
        account = host.split(".")[0]
        key = f"{account}/{container}"
        cached = self._tokens.get(key)
        if cached and cached[1] - time.time() > 300:
            return cached[0]
        response = self.http.get(f"{self.token_url}/{key}", timeout=self.timeout_s)
        if response.status_code != 200:
            raise PlanetaryError(f"token request for {key} failed: HTTP {response.status_code}")
        payload = response.json()
        expiry = _parse_dt(payload["msft:expiry"]).timestamp()
        self._tokens[key] = (payload["token"], expiry)
        return payload["token"]

    def sign(self, href: str, collection: str = "") -> str:
        return f"{href}?{self.token(href)}"

    # -- reading -----------------------------------------------------------

    def read_mosaic(
        self,
        items: Sequence[Mapping[str, Any]],
        asset: str,
        collection: str,
        grid: TargetGrid,
        positive_only: bool,
    ) -> np.ndarray:
        """Resample every item's asset onto ``grid`` (bilinear) and mosaic, NaN where empty.

        ``positive_only`` treats values <= 0 as invalid, which is right for
        backscatter in linear power and wrong for elevation.

        Each item is read with one windowed request over the grid's footprint
        (GDAL fetches the COG tiles in parallel), then reprojected in memory.
        Bilinear on linear power, before any dB conversion (C3).
        """
        cache_path = self._cache_path(items, asset, grid)
        if cache_path is not None and cache_path.is_file():
            with rasterio.open(cache_path) as cached:
                return cached.read(1)

        out = np.full((grid.height, grid.width), np.nan, dtype="float32")
        for item in items:
            data = self._read_item(item["assets"][asset]["href"], collection, grid)
            valid = np.isfinite(data)
            if positive_only:
                valid &= data > 0
            fill = np.isnan(out) & valid
            out[fill] = data[fill]

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp.tif")
            profile = dict(
                driver="GTiff",
                width=grid.width,
                height=grid.height,
                count=1,
                dtype="float32",
                crs=grid.crs,
                transform=grid.transform,
                nodata=np.nan,
                compress="deflate",
                predictor=3,
            )
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(out, 1)
            tmp.replace(cache_path)
        return out

    #: GDAL HTTP settings. Without a timeout a stalled connection hangs forever
    #: (seen on the first Trishuli run); low-speed abort plus retries recover.
    _GDAL_ENV = dict(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_HTTP_TIMEOUT="60",
        GDAL_HTTP_CONNECTTIMEOUT="20",
        GDAL_HTTP_LOW_SPEED_LIMIT="1024",
        GDAL_HTTP_LOW_SPEED_TIME="30",
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="2",
        VSI_CACHE="TRUE",
        CPL_VSIL_CURL_CHUNK_SIZE="2097152",
    )

    def _read_item(self, href: str, collection: str, grid: TargetGrid, attempts: int = 3):
        destination = np.full((grid.height, grid.width), np.nan, dtype="float32")
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                with (
                    rasterio.Env(**self._GDAL_ENV),
                    rasterio.open(self.sign(href, collection)) as src,
                ):
                    left, bottom, right, top = transform_bounds(
                        grid.crs, src.crs, *grid.bounds, densify_pts=21
                    )
                    window = from_bounds(left, bottom, right, top, src.transform)
                    # Pad by two source pixels so bilinear has neighbours at the edge.
                    window = (
                        Window(
                            window.col_off - 2,
                            window.row_off - 2,
                            window.width + 4,
                            window.height + 4,
                        )
                        .round_offsets()
                        .round_lengths()
                    )
                    window = window.intersection(Window(0, 0, src.width, src.height))
                    source = src.read(1, window=window).astype("float32")
                    reproject(
                        source=source,
                        destination=destination,
                        src_transform=src.window_transform(window),
                        src_crs=src.crs,
                        src_nodata=src.nodata,
                        dst_transform=grid.transform,
                        dst_crs=grid.crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                    )
                return destination
            except RasterioIOError as exc:
                last_error = exc
                time.sleep(2 * (attempt + 1))
            except Exception as exc:  # window outside the raster: nothing to read
                if "intersect" in str(exc).lower():
                    return destination
                raise
        raise PlanetaryError(
            f"reading {href.split('?')[0]} failed after {attempts} attempts: {last_error}"
        )

    def _cache_path(self, items, asset: str, grid: TargetGrid) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = json.dumps(
            {"items": sorted(i["id"] for i in items), "asset": asset, "grid": grid.describe()},
            sort_keys=True,
        )
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.tif"


__all__ = [
    "Acquisition",
    "PlanetaryClient",
    "PlanetaryError",
    "S1Pair",
    "SOURCE_ATTRIBUTION",
    "TargetGrid",
    "group_acquisitions",
    "select_pair",
]
