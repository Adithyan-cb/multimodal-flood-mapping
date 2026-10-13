"""Sentinel-1 change-detection baseline tests (offline).

Classification is tested on tiny synthetic backscatter grids; pair selection
on hand-made STAC items. No test touches the network.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from shapely.geometry import box, mapping

from floodmap.acquisition.planetary import (
    PlanetaryError,
    group_acquisitions,
    select_pair,
)
from floodmap.infrastructure.mask import MaskSemantics
from floodmap.baseline.change_detection import (
    BaselineRules,
    boxcar_mean,
    classify,
    slope_degrees,
    to_db,
)

SEM = MaskSemantics(no_water=0, permanent_water=1, flood=2, nodata=255)
NO_FILTER = BaselineRules(
    speckle_window_px=1, water_db=-18.0, change_db=-3.0, max_slope_deg=20.0, min_flood_px=1
)


def _lin(db):
    return (10.0 ** (np.asarray(db, dtype="float64") / 10.0)).astype("float32")


class TestFilters:
    def test_db_conversion(self):
        out = to_db(np.array([1.0, 0.1, 0.0, np.nan], dtype="float32"))
        assert out[:2].tolist() == pytest.approx([0.0, -10.0])
        assert np.isnan(out[2:]).all()

    def test_boxcar_matches_brute_force_with_nans(self):
        rng = np.random.default_rng(1)
        a = rng.uniform(0.01, 1.0, size=(9, 11)).astype("float32")
        a[3, 4] = np.nan
        a[0, 0] = np.nan
        out = boxcar_mean(a, 3)
        for i in range(9):
            for j in range(11):
                if np.isnan(a[i, j]):
                    assert np.isnan(out[i, j])  # never filled from neighbours
                    continue
                window = a[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2]
                assert out[i, j] == pytest.approx(np.nanmean(window), rel=1e-5)

    def test_window_one_is_identity(self):
        a = np.array([[0.5, np.nan]], dtype="float32")
        out = boxcar_mean(a, 1)
        assert out[0, 0] == 0.5 and np.isnan(out[0, 1])

    def test_slope_of_a_plane(self):
        # Rises 10 m per 10 m pixel eastwards: 45 degrees.
        elevation = np.tile(np.arange(6, dtype="float64") * 10.0, (5, 1))
        assert slope_degrees(elevation, 10.0) == pytest.approx(np.full((5, 6), 45.0))


class TestClassify:
    def _grids(self):
        pre = np.full((4, 4), -8.0)  # land
        post = np.full((4, 4), -8.0)
        slope = np.zeros((4, 4))
        pre[0, 0], post[0, 0] = -8.0, -22.0  # new water: flood
        pre[0, 1], post[0, 1] = -21.0, -23.0  # water both times: permanent
        pre[0, 2], post[0, 2] = -16.0, -18.5  # dark, but only 2.5 dB darker: not flood
        pre[0, 3], post[0, 3] = -22.0, -9.0  # water receded: not water
        slope[1, 0] = 35.0  # steep: nodata
        pre[1, 1] = np.nan  # unobserved before: nodata
        pre[1, 2], post[1, 2] = -8.0, -15.0  # darkened 7 dB but not dark enough: land
        return _lin(pre), _lin(post), slope.astype("float32")

    def test_rule_cases(self):
        pre, post, slope = self._grids()
        classes, layers = classify(pre, post, slope, NO_FILTER, SEM)
        assert classes[0].tolist() == [2, 1, 0, 0]
        assert classes[1].tolist() == [255, 255, 0, 0]
        assert layers["change_db"][0, 0] == pytest.approx(-14.0, abs=1e-4)

    def test_permanent_water_is_never_flood(self):
        pre = _lin(np.full((3, 3), -25.0))
        post = _lin(np.full((3, 3), -30.0))  # darker still, but was water already
        classes, _ = classify(pre, post, np.zeros((3, 3), "float32"), NO_FILTER, SEM)
        assert (classes == 1).all()

    def test_output_obeys_the_contract(self):
        pre, post, slope = self._grids()
        classes, _ = classify(pre, post, slope, NO_FILTER, SEM)
        assert classes.dtype == np.uint8
        assert set(np.unique(classes)) <= {0, 1, 2, 255}

    def test_sieve_removes_specks_but_keeps_patches(self):
        pre = np.full((20, 20), -8.0)
        post = np.full((20, 20), -8.0)
        post[2, 2] = -22.0  # 1-pixel speck
        post[10:14, 10:14] = -22.0  # 16-pixel patch
        rules = BaselineRules(speckle_window_px=1, min_flood_px=10)
        classes, _ = classify(_lin(pre), _lin(post), np.zeros((20, 20), "float32"), rules, SEM)
        assert classes[2, 2] == 0
        assert (classes[10:14, 10:14] == 2).all()

    def test_sieve_never_creates_flood_on_nodata(self):
        pre = np.full((10, 10), -8.0)
        post = np.full((10, 10), -22.0)  # everything floods...
        slope = np.zeros((10, 10), "float32")
        slope[5, 5] = 40.0  # ...except one steep pixel inside
        rules = BaselineRules(speckle_window_px=1, min_flood_px=5)
        classes, _ = classify(_lin(pre), _lin(post), slope, rules, SEM)
        assert classes[5, 5] == 255

    def test_rules_load_from_config(self):
        rules = BaselineRules.from_config()
        assert rules.polarisation == "VV"
        assert rules.change_db < 0 and rules.speckle_window_px % 2 == 1

    def test_positive_change_threshold_is_rejected(self):
        bad = {
            "classification": {
                "polarisation": "VV",
                "speckle_window_px": 5,
                "water_db": -18,
                "change_db": 3,
                "max_slope_deg": 20,
                "min_flood_px": 10,
            }
        }
        with pytest.raises(ValueError, match="negative"):
            BaselineRules.from_config(bad)


AOI = box(85.25, 27.85, 85.40, 28.20)


def _item(item_id, when, orbit, state="ascending", geom=AOI.buffer(0.5)):
    return {
        "id": item_id,
        "geometry": mapping(geom),
        "properties": {
            "datetime": when,
            "platform": "sentinel-1d",
            "sat:relative_orbit": orbit,
            "sat:orbit_state": state,
        },
    }


class TestPairSelection:
    def _pair(self, items, event=date(2026, 8, 26)):
        return select_pair(group_acquisitions(items, AOI), event, 12, 36, 0.9)

    def test_trishuli_like_catalog_picks_same_track_twelve_days_apart(self):
        items = [
            _item("a", "2026-08-16T12:21:41Z", 85),
            _item("b", "2026-08-24T00:18:44Z", 19, "descending"),
            _item("c", "2026-08-28T12:21:41Z", 85),
            _item("d", "2026-09-05T00:18:44Z", 19, "descending"),
        ]
        pair = self._pair(items)
        assert (pair.pre.item_ids, pair.post.item_ids) == (["a"], ["c"])
        assert pair.days_apart == 12.0

    def test_never_mixes_tracks(self):
        items = [
            _item("pre", "2026-08-20T00:18:44Z", 19, "descending"),
            _item("post", "2026-08-28T12:21:41Z", 85),
        ]
        with pytest.raises(PlanetaryError, match="no same-track"):
            self._pair(items)

    def test_event_day_is_excluded_from_both_sides(self):
        items = [
            _item("pre", "2026-08-14T12:21:41Z", 85),
            _item("same-day", "2026-08-26T12:21:41Z", 85),
            _item("post", "2026-09-07T12:21:41Z", 85),
        ]
        pair = self._pair(items)
        assert pair.pre.item_ids == ["pre"] and pair.post.item_ids == ["post"]

    def test_partial_coverage_post_is_skipped(self):
        sliver = box(85.25, 27.85, 85.27, 27.90)
        items = [
            _item("pre", "2026-08-16T12:21:41Z", 85),
            _item("partial", "2026-08-27T00:10:00Z", 121, "descending", sliver),
            _item("post", "2026-08-28T12:21:41Z", 85),
        ]
        assert self._pair(items).post.item_ids == ["post"]

    def test_two_slices_of_one_pass_are_mosaicked(self):
        south = box(85.0, 27.0, 86.0, 28.0)
        north = box(85.0, 28.0, 86.0, 29.0)
        items = [
            _item("pre-s", "2026-08-16T12:21:41Z", 85, geom=south),
            _item("pre-n", "2026-08-16T12:22:06Z", 85, geom=north),
            _item("post-s", "2026-08-28T12:21:41Z", 85, geom=south),
            _item("post-n", "2026-08-28T12:22:06Z", 85, geom=north),
        ]
        pair = self._pair(items)
        assert pair.pre.item_ids == ["pre-n", "pre-s"]
        assert pair.post.coverage == pytest.approx(1.0)

    def test_no_imagery_explains_what_was_available(self):
        with pytest.raises(PlanetaryError, match="Available: none"):
            self._pair([])
