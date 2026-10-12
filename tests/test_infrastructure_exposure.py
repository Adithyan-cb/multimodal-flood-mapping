"""Flood mask validation and infrastructure exposure tests (offline).

Masks are tiny synthetic grids in UTM 45N with 10 m pixels, so every expected
length and pixel count can be worked out by hand.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon, box

from floodmap.infrastructure.exposure import (
    EXPOSED,
    NOT_EXPOSED,
    NOT_OBSERVED,
    UNKNOWN,
    ExposureRules,
    assess_exposure,
    assess_layer,
    measure_lines,
    measure_polygons,
    sample_classes,
    summarize_exposure,
)
from floodmap.infrastructure.mask import (
    SYNTHETIC_TAG,
    FloodMask,
    MaskError,
    MaskSemantics,
    load_flood_mask,
)
from floodmap.utils.provenance import REQUIRED_ATTRIBUTIONS

UTM = CRS.from_epsg(32645)
X0, Y0 = 300_000.0, 3_100_000.0  # top-left corner
TRANSFORM = from_origin(X0, Y0, 10.0, 10.0)
SEM = MaskSemantics(no_water=0, permanent_water=1, flood=2, nodata=255)
RULES = ExposureRules(line_sample_step_m=5.0, line_min_flooded_m=10.0, building_min_flood_px=1)


def _mask(classes: np.ndarray, synthetic: bool = False) -> FloodMask:
    return FloodMask(
        path="memory",
        classes=classes.astype("uint8"),
        transform=TRANSFORM,
        crs=UTM,
        semantics=SEM,
        synthetic=synthetic,
        tags={},
    )


def _grid() -> np.ndarray:
    """10 x 10 grid (100 m x 100 m).

    Row 2 columns 2-4 flood, row 5 columns 0-9 permanent water,
    row 8 columns 2-4 nodata. Everything else no water.
    """
    g = np.zeros((10, 10), dtype="uint8")
    g[2, 2:5] = 2
    g[5, :] = 1
    g[8, 2:5] = 255
    return g


def _row_line(row: int, x_from: float, x_to: float) -> LineString:
    """Horizontal line through the centre of a pixel row, x in metres from the left edge."""
    y = Y0 - (row + 0.5) * 10.0
    return LineString([(X0 + x_from, y), (X0 + x_to, y)])


def _series(*geoms) -> gpd.GeoSeries:
    return gpd.GeoSeries(list(geoms), crs=UTM)


class TestSemantics:
    def test_class_ids_come_from_the_segmentation_contract(self):
        assert MaskSemantics.from_config() == SEM

    def test_renamed_flood_class_fails_loudly(self):
        config = {
            "classes": {
                "names": ["No water", "Permanent Waters", "Inundation"],
                "semantic_values": [0, 1, 2],
            },
            "output": {"class_mask_nodata": 255},
        }
        with pytest.raises(MaskError, match="floods"):
            MaskSemantics.from_config(config)


class TestLoadFloodMask:
    def _write(self, path, data, crs=UTM, nodata=255, tags=None, count=1):
        profile = dict(
            driver="GTiff",
            width=data.shape[-1],
            height=data.shape[-2],
            count=count,
            dtype=data.dtype,
            crs=crs,
            transform=TRANSFORM,
            nodata=nodata,
        )
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data if data.ndim == 3 else data[None])
            if tags:
                dst.update_tags(**tags)
        return path

    def test_valid_mask_loads_with_synthetic_flag(self, tmp_path):
        path = self._write(tmp_path / "m.tif", _grid(), tags={SYNTHETIC_TAG: "true"})
        mask = load_flood_mask(path, SEM)
        assert mask.synthetic
        assert mask.class_pixel_counts() == {
            "no_water": 84,
            "permanent_water": 10,
            "flood": 3,
            "nodata": 3,
        }
        assert mask.describe()["flood_area_km2"] == pytest.approx(0.0003)

    def test_untagged_mask_is_not_synthetic(self, tmp_path):
        assert not load_flood_mask(self._write(tmp_path / "m.tif", _grid()), SEM).synthetic

    def test_geographic_crs_is_rejected(self, tmp_path):
        path = self._write(tmp_path / "m.tif", _grid(), crs=CRS.from_epsg(4326))
        with pytest.raises(MaskError, match="geographic"):
            load_flood_mask(path, SEM)

    def test_probability_raster_is_rejected(self, tmp_path):
        path = self._write(tmp_path / "m.tif", np.full((4, 4), 0.7, dtype="float32"), nodata=None)
        with pytest.raises(MaskError, match="uint8"):
            load_flood_mask(path, SEM)

    def test_unmapped_label_value_is_rejected(self, tmp_path):
        data = _grid()
        data[0, 0] = 3  # the Kuro Siwo input sentinel, not an output class
        with pytest.raises(MaskError, match=r"\[3\]"):
            load_flood_mask(self._write(tmp_path / "m.tif", data), SEM)

    def test_wrong_declared_nodata_is_rejected(self, tmp_path):
        path = self._write(tmp_path / "m.tif", _grid(), nodata=0)
        with pytest.raises(MaskError, match="nodata"):
            load_flood_mask(path, SEM)

    def test_multiband_is_rejected(self, tmp_path):
        path = self._write(tmp_path / "m.tif", np.stack([_grid(), _grid()]), count=2)
        with pytest.raises(MaskError, match="one band"):
            load_flood_mask(path, SEM)


class TestSampleClasses:
    def test_points_take_their_pixel_and_outside_is_minus_one(self):
        mask = _mask(_grid())
        xs = np.array([X0 + 25, X0 + 5, X0 - 1])
        ys = np.array([Y0 - 25, Y0 - 55, Y0 - 5])
        assert sample_classes(mask, xs, ys).tolist() == [2, 1, -1]


class TestLines:
    def test_flooded_length_is_measured_in_metres(self):
        measured = measure_lines(_series(_row_line(2, 0, 100)), _mask(_grid()), 5.0)
        assert measured.loc[0, "length_m"] == pytest.approx(100)
        assert measured.loc[0, "flooded_m"] == pytest.approx(30)
        assert measured.loc[0, "outside_m"] == pytest.approx(0)

    def test_statuses(self):
        mask = _mask(_grid())
        frame = gpd.GeoDataFrame(
            {"highway": ["track"] * 5},
            geometry=[
                _row_line(2, 0, 100),  # crosses flood
                _row_line(5, 0, 100),  # river only (a bridge, say)
                _row_line(8, 0, 100),  # crosses nodata
                _row_line(0, 0, 100),  # dry
                LineString([(X0 - 500, Y0 + 500), (X0 - 400, Y0 + 500)]),  # off the grid
            ],
            crs=UTM,
        )
        out = assess_layer(frame, mask, "line", RULES)
        assert out["exposure_status"].tolist() == [
            EXPOSED,
            NOT_EXPOSED,
            UNKNOWN,
            NOT_EXPOSED,
            NOT_OBSERVED,
        ]
        assert out.loc[1, "permanent_water_m"] == pytest.approx(100)

    def test_permanent_water_never_makes_a_line_exposed(self):
        grid = np.ones((10, 10), dtype="uint8")
        out = assess_layer(
            gpd.GeoDataFrame(geometry=[_row_line(3, 0, 100)], crs=UTM), _mask(grid), "line", RULES
        )
        assert out.loc[0, "exposure_status"] == NOT_EXPOSED

    def test_short_line_flooded_end_to_end_is_exposed(self):
        # 6 m bridge, shorter than the 10 m threshold, entirely on flood.
        out = assess_layer(
            gpd.GeoDataFrame(geometry=[_row_line(2, 22, 28)], crs=UTM),
            _mask(_grid()),
            "line",
            RULES,
        )
        assert out.loc[0, "exposure_status"] == EXPOSED

    def test_line_brushing_flood_below_threshold_is_not_exposed(self):
        grid = np.zeros((10, 10), dtype="uint8")
        grid[2, 5] = 2
        line = _row_line(2, 54, 100)  # 6 m of its 46 m on the flood pixel
        out = assess_layer(gpd.GeoDataFrame(geometry=[line], crs=UTM), _mask(grid), "line", RULES)
        assert out.loc[0, "flooded_m"] == pytest.approx(6, abs=2.5)
        assert out.loc[0, "exposure_status"] == NOT_EXPOSED

    def test_part_outside_the_grid_is_reported_not_guessed(self):
        measured = measure_lines(_series(_row_line(0, 50, 150)), _mask(_grid()), 5.0)
        assert measured.loc[0, "outside_m"] == pytest.approx(50)

    def test_closed_way_returned_as_polygon_is_measured_as_its_outline(self):
        ring = box(X0 + 20, Y0 - 30, X0 + 50, Y0 - 20)  # 30 m x 10 m around row 2 flood
        measured = measure_lines(_series(ring), _mask(_grid()), 5.0)
        assert measured.loc[0, "length_m"] == pytest.approx(80)
        assert measured.loc[0, "flooded_m"] > 0

    def test_sampling_matches_exact_intersection(self):
        rng = np.random.default_rng(0)
        grid = rng.choice([0, 2], size=(10, 10), p=[0.6, 0.4]).astype("uint8")
        mask = _mask(grid)
        flood = shapely.union_all(
            [
                box(X0 + c * 10, Y0 - (r + 1) * 10, X0 + (c + 1) * 10, Y0 - r * 10)
                for r, c in zip(*np.nonzero(grid == 2))
            ]
        )
        lines = [
            LineString(rng.uniform([X0, Y0 - 100], [X0 + 100, Y0], size=(2, 2))) for _ in range(30)
        ]
        measured = measure_lines(_series(*lines), mask, 1.0)
        exact = np.array([line.intersection(flood).length for line in lines])
        assert measured["flooded_m"].sum() == pytest.approx(exact.sum(), rel=0.02)
        assert np.abs(measured["flooded_m"].to_numpy() - exact).max() < 3.0


class TestPolygons:
    def test_building_touching_one_flood_pixel_is_exposed(self):
        building = box(X0 + 28, Y0 - 38, X0 + 36, Y0 - 29)  # touches row 2, col 2-3
        out = assess_layer(
            gpd.GeoDataFrame(geometry=[building], crs=UTM), _mask(_grid()), "polygon", RULES
        )
        assert out.loc[0, "flood_px"] >= 1
        assert out.loc[0, "exposure_status"] == EXPOSED

    def test_statuses(self):
        frame = gpd.GeoDataFrame(
            geometry=[
                box(X0 + 61, Y0 - 9, X0 + 69, Y0 - 1),  # dry
                box(X0 + 21, Y0 - 89, X0 + 29, Y0 - 81),  # nodata only
                box(X0 + 61, Y0 - 59, X0 + 69, Y0 - 51),  # river only
                box(X0 + 500, Y0 + 500, X0 + 510, Y0 + 510),  # off the grid
            ],
            crs=UTM,
        )
        out = assess_layer(frame, _mask(_grid()), "polygon", RULES)
        assert out["exposure_status"].tolist() == [NOT_EXPOSED, UNKNOWN, NOT_EXPOSED, NOT_OBSERVED]

    def test_adjacent_buildings_sharing_a_pixel_both_see_it(self):
        left = box(X0 + 21, Y0 - 29, X0 + 25, Y0 - 21)
        right = box(X0 + 25, Y0 - 29, X0 + 29, Y0 - 21)
        measured = measure_polygons(_series(left, right), _mask(_grid()))
        assert measured["flood_px"].tolist() == [1, 1]


class TestLayersAndSummary:
    def test_output_keeps_input_crs_and_geometry(self):
        frame = gpd.GeoDataFrame(
            {"osm_id": ["way/1"], "highway": ["track"]}, geometry=[_row_line(2, 0, 100)], crs=UTM
        ).to_crs("EPSG:4326")
        out = assess_exposure({"roads": frame}, _mask(_grid()), RULES)["roads"]
        assert out.crs.to_epsg() == 4326
        assert out.geometry.iloc[0].equals(frame.geometry.iloc[0])

    def test_summary_counts_rules_attribution_and_synthetic_flag(self):
        roads = gpd.GeoDataFrame(
            {"osm_id": ["way/1", "way/2"], "highway": ["track", "primary"]},
            geometry=[_row_line(2, 0, 100), _row_line(0, 0, 100)],
            crs=UTM,
        )
        bridges = gpd.GeoDataFrame(
            {"osm_id": ["way/3"], "name": ["Pul"], "highway": ["path"]},
            geometry=[_row_line(2, 20, 50)],
            crs=UTM,
        )
        mask = _mask(_grid(), synthetic=True)
        assessed = assess_exposure({"roads": roads, "bridges": bridges}, mask, RULES)
        summary = summarize_exposure(assessed, mask, RULES)

        assert summary["mask"]["synthetic"] is True
        assert summary["layers"]["roads"]["count"][EXPOSED] == 1
        assert summary["layers"]["roads"]["flooded_km_by_highway"] == {"track": 0.03}
        assert summary["layers"]["bridges"]["exposed"][0]["name"] == "Pul"
        assert summary["rules"]["line_min_flooded_m"] == 10.0
        assert summary["attribution"] == list(REQUIRED_ATTRIBUTIONS)
        assert "not evidence of physical damage" in summary["interpretation"]

    def test_exposure_config_loads(self):
        assert ExposureRules.from_config() == RULES
