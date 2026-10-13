"""Flood class mask loading and validation.

Scientific responsibility
-------------------------
Read the segmentation output that every downstream impact number depends on,
and refuse anything that does not match the frozen M4 output contract
(docs/m4-architecture-decision.md §10, C1, C2, C6, C9):

* single-band ``uint8`` GeoTIFF;
* projected CRS in metres (C1: UTM);
* values only from the semantic classes plus nodata.

Class ids are read from ``configs/segmentation.yaml`` (``classes.names`` and
``output.class_mask_nodata``) rather than restated here, so the impact layer
follows the model contract if it ever changes.

Nodata (255) means "not observed" (radar shadow, layover, outside the swath).
It is never treated as "not flooded" (docs/scientific-assumptions.md §8
distinguishes "not observed" from "observed, not flooded").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import shapes
from shapely.geometry import shape

from floodmap.utils.config import load_config

#: GeoTIFF tag written by scripts/make_fake_mask.py. Results computed from a
#: mask carrying it are development output and must never be reported.
SYNTHETIC_TAG = "FLOODMAP_SYNTHETIC"

_FLOOD_NAME = "floods"
_PERMANENT_NAME = "permanent waters"
_NO_WATER_NAME = "no water"


class MaskError(ValueError):
    """The flood mask does not satisfy the output contract."""


@dataclass(frozen=True)
class MaskSemantics:
    """Which raster value means what."""

    no_water: int
    permanent_water: int
    flood: int
    nodata: int

    @property
    def allowed_values(self) -> frozenset:
        return frozenset({self.no_water, self.permanent_water, self.flood, self.nodata})

    @classmethod
    def from_config(
        cls, segmentation_config: Optional[Mapping[str, Any]] = None
    ) -> "MaskSemantics":
        config = segmentation_config or load_config("segmentation")
        names = [str(name).strip().lower() for name in config["classes"]["names"]]
        values = list(config["classes"]["semantic_values"])
        lookup = dict(zip(names, values))
        try:
            semantics = cls(
                no_water=int(lookup[_NO_WATER_NAME]),
                permanent_water=int(lookup[_PERMANENT_NAME]),
                flood=int(lookup[_FLOOD_NAME]),
                nodata=int(config["output"]["class_mask_nodata"]),
            )
        except KeyError as exc:
            raise MaskError(
                f"configs/segmentation.yaml no longer names the class {exc}; the impact "
                "layer cannot tell which value means flood"
            ) from exc
        if semantics.nodata in (semantics.no_water, semantics.permanent_water, semantics.flood):
            raise MaskError("class_mask_nodata collides with a semantic class id")
        return semantics


@dataclass
class FloodMask:
    """A validated class mask held in memory."""

    path: Path
    classes: np.ndarray
    transform: Affine
    crs: CRS
    semantics: MaskSemantics
    synthetic: bool
    tags: Dict[str, str]

    @property
    def shape(self):
        return self.classes.shape

    @property
    def pixel_size_m(self):
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def bounds(self):
        height, width = self.classes.shape
        left, top = self.transform @ (0, 0)
        right, bottom = self.transform @ (width, height)
        return min(left, right), min(top, bottom), max(left, right), max(top, bottom)

    def class_pixel_counts(self) -> Dict[str, int]:
        s = self.semantics
        counts = np.bincount(self.classes.ravel(), minlength=256)
        return {
            "no_water": int(counts[s.no_water]),
            "permanent_water": int(counts[s.permanent_water]),
            "flood": int(counts[s.flood]),
            "nodata": int(counts[s.nodata]),
        }

    def describe(self) -> Dict[str, Any]:
        counts = self.class_pixel_counts()
        px_w, px_h = self.pixel_size_m
        pixel_km2 = px_w * px_h / 1e6
        return {
            "path": str(self.path),
            "synthetic": self.synthetic,
            "crs": self.crs.to_string(),
            "pixel_size_m": [px_w, px_h],
            "shape": list(self.shape),
            "bounds": list(self.bounds),
            "class_values": {
                "no_water": self.semantics.no_water,
                "permanent_water": self.semantics.permanent_water,
                "flood": self.semantics.flood,
                "nodata": self.semantics.nodata,
            },
            "pixel_counts": counts,
            "flood_area_km2": round(counts["flood"] * pixel_km2, 4),
            "nodata_fraction": round(counts["nodata"] / max(1, self.classes.size), 6),
        }

    def to_polygons(self, simplify_m: float = 0.0) -> gpd.GeoDataFrame:
        """Flood, permanent-water and nodata areas as EPSG:4326 polygons.

        For map display only. Analysis always uses the raster. ``simplify_m``
        smooths pixel staircases; keep it below the pixel size.
        """
        s = self.semantics
        labels = {s.flood: "flood", s.permanent_water: "permanent_water", s.nodata: "nodata"}
        keep = np.isin(self.classes, list(labels))
        rows = [
            {"mask_class": labels[int(value)], "geometry": shape(geom)}
            for geom, value in shapes(self.classes, mask=keep, transform=self.transform)
        ]
        frame = gpd.GeoDataFrame(rows, columns=["mask_class", "geometry"], crs=self.crs)
        if simplify_m > 0 and len(frame):
            frame["geometry"] = frame.geometry.simplify(simplify_m, preserve_topology=True)
        frame = frame.dissolve(by="mask_class", as_index=False) if len(frame) else frame
        frame["area_km2"] = (frame.geometry.area / 1e6).round(4)
        frame["synthetic"] = self.synthetic
        return frame.to_crs("EPSG:4326")


def load_flood_mask(path: Path, semantics: Optional[MaskSemantics] = None) -> FloodMask:
    """Open and validate a flood class mask.

    Raises
    ------
    MaskError
        On any contract violation. A wrong mask is never coerced, because
        every exposure and connectivity number downstream would inherit the
        error silently.
    """
    semantics = semantics or MaskSemantics.from_config()
    path = Path(path)
    if not path.is_file():
        raise MaskError(f"flood mask not found: {path}")

    with rasterio.open(path) as src:
        if src.count != 1:
            raise MaskError(f"flood mask must have exactly one band, got {src.count}")
        if src.dtypes[0] != "uint8":
            raise MaskError(f"flood mask must be uint8, got {src.dtypes[0]}")
        if src.crs is None:
            raise MaskError("flood mask has no CRS")
        if not src.crs.is_projected:
            raise MaskError(
                f"flood mask CRS {src.crs} is geographic; the contract requires a "
                "projected CRS in metres (C1)"
            )
        units = (src.crs.linear_units or "").lower()
        if units not in ("metre", "meter", "metres", "meters", "m"):
            raise MaskError(f"flood mask CRS units must be metres, got {units!r}")
        if src.transform.b != 0 or src.transform.d != 0:
            raise MaskError("rotated or sheared rasters are not supported")
        if src.nodata is not None and int(src.nodata) != semantics.nodata:
            raise MaskError(
                f"flood mask declares nodata={src.nodata}, contract says {semantics.nodata} (C9)"
            )
        classes = src.read(1)
        transform = src.transform
        crs = src.crs
        tags = src.tags()

    present = set(np.unique(classes).tolist())
    unexpected = present - semantics.allowed_values
    if unexpected:
        raise MaskError(
            f"flood mask contains values {sorted(unexpected)} outside the contract "
            f"{sorted(semantics.allowed_values)}; is this a probability raster or an "
            "unmapped label?"
        )

    return FloodMask(
        path=path,
        classes=classes,
        transform=transform,
        crs=crs,
        semantics=semantics,
        synthetic=str(tags.get(SYNTHETIC_TAG, "")).lower() == "true",
        tags=tags,
    )


__all__ = ["FloodMask", "MaskError", "MaskSemantics", "SYNTHETIC_TAG", "load_flood_mask"]
