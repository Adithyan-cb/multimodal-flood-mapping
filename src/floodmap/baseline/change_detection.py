"""Sentinel-1 change-detection flood baseline (deterministic, not learned).

Scientific responsibility
-------------------------
Produce a flood class mask from a same-track pre/post Sentinel-1 pair using
explicit, cited rules, so that

1. the impact layer has a real flood map for any AOI and date even without a
   trained model, and
2. the learned model has a baseline it must beat (AGENTS.md §5).

Output follows the frozen M4 class-mask contract: uint8, the AOI's UTM grid
at 10 m, 0 no water / 1 permanent water / 2 flood / 255 nodata.

Rule (thresholds in configs/baseline.yaml)
------------------------------------------
On VV gamma0 (terrain-corrected backscatter):

1. Boxcar mean in linear power over ``speckle_window_px`` to suppress speckle,
   then convert to dB. Invalid pixels are excluded from the mean and stay
   invalid; they are never filled from neighbours.
2. ``water`` = backscatter below ``water_db`` (calm water reflects the radar
   pulse away from the sensor, so it is dark).
3. FLOOD = water after, not water before, and darker by at least
   ``change_db`` (log-ratio change).
4. PERMANENT WATER = water both before and after.
5. NODATA = no valid observation on either date, or slope above
   ``max_slope_deg`` (radar shadow and layover concentrate on steep slopes,
   and standing flood water there is implausible).
6. Flood patches smaller than ``min_flood_px`` are removed.

Known blind spots (state them in the report): debris and mud can raise
backscatter rather than lower it, so debris-covered ground is missed; flooded
vegetation and urban areas can brighten through double bounce; wind-roughened
water can look like land; and the map shows the water present at the post
acquisition time, not the flood peak.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from rasterio.features import sieve

from floodmap.infrastructure.mask import MaskSemantics
from floodmap.utils.config import load_config

METHOD = "s1_change_detection_baseline"


@dataclass(frozen=True)
class BaselineRules:
    polarisation: str = "VV"
    speckle_window_px: int = 5
    water_db: float = -18.0
    change_db: float = -3.0
    max_slope_deg: float = 20.0
    min_flood_px: int = 10

    @classmethod
    def from_config(cls, baseline_config: Optional[Mapping[str, Any]] = None) -> "BaselineRules":
        block = (baseline_config or load_config("baseline"))["classification"]
        rules = cls(
            polarisation=str(block["polarisation"]).upper(),
            speckle_window_px=int(block["speckle_window_px"]),
            water_db=float(block["water_db"]),
            change_db=float(block["change_db"]),
            max_slope_deg=float(block["max_slope_deg"]),
            min_flood_px=int(block["min_flood_px"]),
        )
        if rules.speckle_window_px < 1 or rules.speckle_window_px % 2 == 0:
            raise ValueError("speckle_window_px must be a positive odd number")
        if rules.change_db >= 0:
            raise ValueError("change_db must be negative: flooding darkens the image")
        return rules


def boxcar_mean(values: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware moving mean over a ``window`` x ``window`` box.

    Uses integral images, so it is O(pixels) for any window. Windows are
    truncated at the raster edge. A pixel that is NaN on input stays NaN.
    """
    if window <= 1:
        return values.astype("float32", copy=True)
    r = window // 2
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0).astype("float64")

    def box_sum(x: np.ndarray) -> np.ndarray:
        padded = np.pad(x, ((r + 1, r), (r + 1, r)))
        c = padded.cumsum(0).cumsum(1)
        return (
            c[window:, window:]
            - c[:-window, window:]
            - c[window:, :-window]
            + c[:-window, :-window]
        )

    total = box_sum(filled)
    count = box_sum(valid.astype("float64"))
    out = np.full(values.shape, np.nan, dtype="float32")
    ok = valid & (count > 0)
    out[ok] = (total[ok] / count[ok]).astype("float32")
    return out


def to_db(linear: np.ndarray) -> np.ndarray:
    """10 log10 of linear power; non-positive and NaN become NaN."""
    out = np.full(linear.shape, np.nan, dtype="float32")
    ok = np.isfinite(linear) & (linear > 0)
    out[ok] = 10.0 * np.log10(linear[ok])
    return out


def slope_degrees(elevation: np.ndarray, pixel_m: float) -> np.ndarray:
    """Terrain slope from an elevation grid in metres with square pixels."""
    dz_dy, dz_dx = np.gradient(elevation.astype("float64"), pixel_m)
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype("float32")


def classify(
    pre_linear: np.ndarray,
    post_linear: np.ndarray,
    slope_deg: np.ndarray,
    rules: BaselineRules,
    semantics: MaskSemantics,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Apply the rule in the module docstring.

    Returns the uint8 class mask and the intermediate dB layers
    (``pre_db``, ``post_db``, ``change_db``) for inspection.
    """
    if not (pre_linear.shape == post_linear.shape == slope_deg.shape):
        raise ValueError("pre, post and slope grids must have the same shape")

    pre_db = to_db(boxcar_mean(pre_linear, rules.speckle_window_px))
    post_db = to_db(boxcar_mean(post_linear, rules.speckle_window_px))
    change_db = post_db - pre_db

    with np.errstate(invalid="ignore"):
        observed = np.isfinite(pre_db) & np.isfinite(post_db) & np.isfinite(slope_deg)
        observed &= slope_deg <= rules.max_slope_deg
        water_pre = pre_db < rules.water_db
        water_post = post_db < rules.water_db
        flood = observed & water_post & ~water_pre & (change_db <= rules.change_db)
        permanent = observed & water_pre & water_post

    if rules.min_flood_px > 1 and flood.any():
        # Sieve merges small regions into their largest neighbour: small flood
        # specks vanish, and small gaps inside flood fill in. Re-applying the
        # observed/permanent masks afterwards keeps the sieve from creating
        # flood on nodata or permanent-water pixels.
        flood = sieve(flood.astype("uint8"), size=rules.min_flood_px, connectivity=8).astype(bool)
        flood &= observed & ~permanent

    classes = np.full(pre_linear.shape, semantics.no_water, dtype="uint8")
    classes[permanent] = semantics.permanent_water
    classes[flood] = semantics.flood
    classes[~observed] = semantics.nodata
    return classes, {"pre_db": pre_db, "post_db": post_db, "change_db": change_db}


def rules_dict(rules: BaselineRules) -> Dict[str, Any]:
    return asdict(rules)


__all__ = [
    "METHOD",
    "BaselineRules",
    "boxcar_mean",
    "classify",
    "rules_dict",
    "slope_degrees",
    "to_db",
]
