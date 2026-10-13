"""Synthetic demo bundle tests.

The demo bundle exists so the dashboard has something honest to display before
real pipeline artifacts exist. These tests pin the properties that make it
*honest*:

* deterministic — regenerating it yields identical bytes, so it cannot drift
  from the contract it demonstrates;
* contract-valid — ``load_manifest`` accepts it, so what the UI exercises is
  exactly what production bundles will produce;
* labelled — ``is_synthetic`` is set and provenance notes say so, so a demo can
  never be mistaken for a result;
* explicit absence — the DEM flood-path layer is declared unavailable *with a
  reason*, matching ``architecture.md`` §17.
"""

from __future__ import annotations

import json
from pathlib import Path

import rasterio

from floodmap.dashboard import build_demo_bundle, load_manifest
from floodmap.dashboard.contract import EvidenceTier
from floodmap.utils.provenance import REQUIRED_ATTRIBUTIONS

DEMO_SEED = 2026


def _build(tmp_path: Path) -> Path:
    return build_demo_bundle(tmp_path / "bundle", seed=DEMO_SEED)


class TestDeterminism:
    def test_regeneration_produces_identical_manifest(self, tmp_path: Path):
        first = _build(tmp_path)
        second = build_demo_bundle(tmp_path / "again", seed=DEMO_SEED)
        assert first.read_bytes() == second.read_bytes()

    def test_regeneration_produces_identical_rasters(self, tmp_path: Path):
        _build(tmp_path)
        again = build_demo_bundle(tmp_path / "again", seed=DEMO_SEED)
        for relative in ["layers/flood_water_extent.tif", "layers/s1_post.tif"]:
            left = (tmp_path / "bundle" / relative).read_bytes()
            right = (again.parent / relative).read_bytes()
            assert left == right, relative

    def test_different_seed_changes_output(self, tmp_path: Path):
        build_demo_bundle(tmp_path / "bundle", seed=DEMO_SEED)
        build_demo_bundle(tmp_path / "bundle2", seed=DEMO_SEED + 1)
        assert (tmp_path / "bundle" / "layers/s1_post.tif").read_bytes() != (
            tmp_path / "bundle2" / "layers/s1_post.tif"
        ).read_bytes()


class TestContractValidity:
    def test_manifest_loads_through_the_contract(self, tmp_path: Path):
        loaded = load_manifest(_build(tmp_path))
        assert loaded.manifest.aoi_id

    def test_layers_cover_all_evidence_tiers(self, tmp_path: Path):
        loaded = load_manifest(_build(tmp_path))
        covered = {layer.evidence_tier for layer in loaded.manifest.layers}
        assert {
            EvidenceTier.OBSERVED_DERIVED,
            EvidenceTier.MODEL_PREDICTION,
            EvidenceTier.SPATIAL_INFERENCE,
            EvidenceTier.NETWORK_INFERENCE,
        } <= covered

    def test_required_attributions_present(self, tmp_path: Path):
        loaded = load_manifest(_build(tmp_path))
        for required in REQUIRED_ATTRIBUTIONS:
            assert required in loaded.manifest.attribution

    def test_every_available_layer_resolves(self, tmp_path: Path):
        loaded = load_manifest(_build(tmp_path))
        for layer in loaded.manifest.layers:
            if layer.available:
                assert loaded.resolve_path(layer).exists(), layer.layer_id
                assert loaded.provenance[layer.layer_id] is not None, layer.layer_id

    def test_geotiffs_carry_band_names_and_crs(self, tmp_path: Path):
        loaded = load_manifest(_build(tmp_path))
        for layer in loaded.manifest.layers:
            if layer.format.value != "geotiff":
                continue
            with rasterio.open(loaded.resolve_path(layer)) as src:
                assert src.descriptions, layer.layer_id
                assert src.crs.to_string() == "EPSG:32645", layer.layer_id


class TestExplicitSyntheticLabelling:
    def test_manifest_flags_synthetic_data(self, tmp_path: Path):
        loaded = load_manifest(_build(tmp_path))
        assert loaded.manifest.is_synthetic is True

    def test_every_provenance_note_denies_reality(self, tmp_path: Path):
        loaded = load_manifest(_build(tmp_path))
        for layer in loaded.manifest.layers:
            if not layer.available:
                continue
            provenance = loaded.provenance[layer.layer_id]
            assert provenance is not None
            assert "SYNTHETIC" in provenance.notes, layer.layer_id

    def test_event_name_contains_synthetic_marker(self, tmp_path: Path):
        loaded = load_manifest(_build(tmp_path))
        assert "SYNTHETIC" in loaded.manifest.event_name

    def test_unavailable_layer_is_declared_with_reason(self, tmp_path: Path):
        loaded = load_manifest(_build(tmp_path))
        flood_path = next(
            layer for layer in loaded.manifest.layers if layer.layer_id == "flood_path_trace"
        )
        assert flood_path.available is False
        assert flood_path.unavailable_reason is not None
        assert loaded.resolve_path(flood_path) is None

    def test_json_is_complete_and_parseable(self, tmp_path: Path):
        manifest_path = _build(tmp_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["aoi_id"] == "trishuli-demo"
        assert payload["is_synthetic"] is True
