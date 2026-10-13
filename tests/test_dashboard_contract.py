"""Dashboard artifact contract tests.

Verify the two rules this package exists to enforce:

1. An available layer must resolve to a real artifact AND a readable provenance
   record, whose ``artifact_type`` agrees with the manifest entry.
2. Absence is explicit: an unavailable layer must say why, and a production
   layer must never link to a provenance record naming a validation-only source
   (AGENTS.md §3).

These test software behaviour. No real pipeline artifact exists yet, so every
fixture here is constructed directly under tmp_path.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from floodmap.dashboard import (
    DashboardContractError,
    DashboardLayer,
    DashboardManifest,
    EvidenceTier,
    LayerFormat,
    load_manifest,
)
from floodmap.utils.provenance import (
    REQUIRED_ATTRIBUTIONS,
    ArtifactProvenance,
    ArtifactType,
    ProductionInput,
    Unknown,
    UnknownReason,
)

SCHEMA = "floodmap.dashboard_manifest/1"


def _layer(**overrides) -> DashboardLayer:
    values = dict(
        layer_id="flood_water_extent",
        title="Predicted flood water extent",
        artifact_type=ArtifactType.SEGMENTATION_PREDICTION,
        evidence_tier=EvidenceTier.MODEL_PREDICTION,
        format=LayerFormat.GEOTIFF,
        path="layers/flood_water_extent.tif",
        crs="EPSG:32645",
        bbox=[100.0, 10.0, 101.0, 11.0],
        provenance_path="layers/flood_water_extent_provenance.yaml",
        available=True,
    )
    values.update(overrides)
    if not values.get("available", True):
        values.setdefault("path", None)
        values.setdefault("provenance_path", None)
    return DashboardLayer.model_validate(values)


def _manifest(**overrides) -> DashboardManifest:
    values = dict(
        schema=SCHEMA,
        aoi_id="trishuli-demo",
        event_name="Trishuli flood event (DEMO)",
        attribution=list(REQUIRED_ATTRIBUTIONS),
        layers=[_layer()],
    )
    values.update(overrides)
    return DashboardManifest.model_validate(values)


def _provenance(
    artifact_type: ArtifactType = ArtifactType.SEGMENTATION_PREDICTION,
    **overrides,
) -> ArtifactProvenance:
    values = dict(
        artifact_type=artifact_type,
        artifact_id="pred-demo-0001",
        aoi="trishuli-demo",
        event_date="2026-08-15",
        production_inputs=[ProductionInput.SENTINEL1],
    )
    values.update(overrides)
    return ArtifactProvenance.model_validate(values)


def _write_bundle(root: Path, *, manifest: DashboardManifest, provenance_by_id: dict) -> None:
    """Materialize a manifest plus provenance files and dummy artifacts.

    Every available layer gets a placeholder artifact file so the loader's
    existence checks have something to find. The contents are irrelevant; the
    loader validates the contract, not the science.
    """
    (root / "layers").mkdir(parents=True, exist_ok=True)
    for layer in manifest.layers:
        if not layer.available:
            continue
        (root / str(layer.path)).write_bytes(b"placeholder")
        (root / str(layer.provenance_path)).write_text(
            provenance_by_id[layer.layer_id].to_yaml(), encoding="utf-8"
        )
    (root / "dashboard_manifest.json").write_text(
        json.dumps(manifest.model_dump(by_alias=True, mode="json"), indent=2),
        encoding="utf-8",
    )


class TestManifestValidation:
    def test_minimal_manifest_is_valid(self):
        manifest = _manifest()
        assert manifest.aoi_id == "trishuli-demo"
        assert manifest.is_synthetic is False

    def test_unknown_schema_version_is_rejected(self):
        with pytest.raises(ValidationError):
            _manifest(schema="floodmap.dashboard_manifest/0")

    def test_missing_required_attribution_is_rejected(self):
        with pytest.raises(ValidationError):
            _manifest(attribution=REQUIRED_ATTRIBUTIONS[1:])

    def test_full_attribution_list_is_accepted(self):
        _manifest(attribution=list(REQUIRED_ATTRIBUTIONS))

    def test_duplicate_layer_ids_are_rejected(self):
        with pytest.raises(ValidationError):
            _manifest(layers=[_layer(), _layer()])

    def test_unexpected_manifest_field_is_rejected(self):
        with pytest.raises(ValidationError):
            DashboardManifest.model_validate(
                _manifest().model_dump(by_alias=True, mode="json") | {"surprise": 1}
            )


class TestLayerAvailability:
    """An absent layer must be explicit; a present one must be traceable."""

    def test_available_layer_without_path_is_rejected(self):
        with pytest.raises(ValidationError):
            _layer(path=None)

    def test_available_layer_without_provenance_is_rejected(self):
        with pytest.raises(ValidationError):
            _layer(provenance_path=None)

    def test_unavailable_layer_requires_a_reason(self):
        with pytest.raises(ValidationError):
            _layer(available=False)

    def test_unavailable_layer_must_not_keep_a_path(self):
        with pytest.raises(ValidationError):
            _layer(available=False, unavailable_reason="no scenes")

    def test_unavailable_layer_is_explicit(self):
        layer = _layer(
            available=False,
            path=None,
            provenance_path=None,
            unavailable_reason=Unknown(
                reason=UnknownReason.ACQUISITION_FAILED, note="no valid scene"
            ),
        )
        assert layer.unavailable_reason.note == "no valid scene"

    def test_bbox_must_have_four_values(self):
        with pytest.raises(ValidationError):
            _layer(bbox=[1.0, 2.0, 3.0])

    @pytest.mark.parametrize(
        "tier",
        [
            EvidenceTier.OBSERVED_DERIVED,
            EvidenceTier.MODEL_PREDICTION,
            EvidenceTier.SPATIAL_INFERENCE,
            EvidenceTier.NETWORK_INFERENCE,
            EvidenceTier.VALIDATION_REFERENCE,
        ],
    )
    def test_every_evidence_tier_is_storable(self, tier):
        assert _layer(evidence_tier=tier).evidence_tier is tier


class TestEvidenceTierGrouping:
    def test_validation_references_are_separated(self):
        manifest = _manifest(
            layers=[
                _layer(layer_id="flood_water_extent"),
                _layer(
                    layer_id="emsr_comparison",
                    title="Reference flood extent (comparison only)",
                    evidence_tier=EvidenceTier.VALIDATION_REFERENCE,
                    artifact_type=ArtifactType.EVALUATION_RESULT,
                ),
            ]
        )
        assert manifest.layers_by_tier()[EvidenceTier.VALIDATION_REFERENCE] == [manifest.layers[1]]
        assert manifest.layers[1].is_validation_reference is True
        assert manifest.layers[0].is_validation_reference is False

    def test_grouping_preserves_declaration_order(self):
        manifest = _manifest(
            layers=[
                _layer(layer_id="a"),
                _layer(layer_id="b"),
                _layer(layer_id="c"),
            ]
        )
        assert [l.layer_id for l in manifest.layers_by_tier()[EvidenceTier.MODEL_PREDICTION]] == [
            "a",
            "b",
            "c",
        ]


class TestLoadManifest:
    @pytest.fixture()
    def bundle(self, tmp_path: Path):
        manifest = _manifest()
        _write_bundle(
            tmp_path,
            manifest=manifest,
            provenance_by_id={"flood_water_extent": _provenance()},
        )
        return tmp_path, manifest

    def test_clean_bundle_loads(self, bundle):
        root, manifest = bundle
        loaded = load_manifest(root / "dashboard_manifest.json")
        assert loaded.manifest == manifest
        assert loaded.base_dir == root

    def test_layer_paths_resolve_relative_to_manifest(self, bundle):
        root, _ = bundle
        loaded = load_manifest(root / "dashboard_manifest.json")
        layer = loaded.manifest.layers[0]
        assert loaded.resolve_path(layer) == root / "layers/flood_water_extent.tif"

    def test_provenance_is_parsed_and_linked(self, bundle):
        root, _ = bundle
        loaded = load_manifest(root / "dashboard_manifest.json")
        layer = loaded.manifest.layers[0]
        assert loaded.provenance[layer.layer_id].artifact_id == "pred-demo-0001"
        assert ProductionInput.SENTINEL1 in loaded.provenance[layer.layer_id].production_inputs

    def test_loaded_dashboard_separates_production_and_validation_layers(self, bundle):
        root, manifest = bundle
        loaded = load_manifest(root / "dashboard_manifest.json")
        assert loaded.production_layers() == [manifest.layers[0]]
        assert loaded.validation_layers() == []

    def test_missing_provenance_file_raises(self, tmp_path: Path):
        manifest = _manifest()
        (tmp_path / "layers").mkdir(parents=True, exist_ok=True)
        (tmp_path / "layers/flood_water_extent.tif").write_bytes(b"placeholder")
        (tmp_path / "dashboard_manifest.json").write_text(
            json.dumps(manifest.model_dump(by_alias=True, mode="json")), encoding="utf-8"
        )
        with pytest.raises(DashboardContractError, match="missing provenance"):
            load_manifest(tmp_path / "dashboard_manifest.json")

    def test_missing_artifact_file_raises(self, tmp_path: Path):
        _write_bundle(
            tmp_path, manifest=_manifest(), provenance_by_id={"flood_water_extent": _provenance()}
        )
        (tmp_path / "layers/flood_water_extent.tif").unlink()
        with pytest.raises(DashboardContractError, match="does not exist"):
            load_manifest(tmp_path / "dashboard_manifest.json")

    def test_unreadable_provenance_raises(self, tmp_path: Path):
        manifest = _manifest()
        (tmp_path / "layers").mkdir(parents=True, exist_ok=True)
        (tmp_path / "layers/flood_water_extent.tif").write_bytes(b"placeholder")
        (tmp_path / "layers/flood_water_extent_provenance.yaml").write_text(
            ": not: valid: yaml::", encoding="utf-8"
        )
        (tmp_path / "dashboard_manifest.json").write_text(
            json.dumps(manifest.model_dump(by_alias=True, mode="json")), encoding="utf-8"
        )
        with pytest.raises(DashboardContractError, match="unreadable provenance"):
            load_manifest(tmp_path / "dashboard_manifest.json")

    def test_artifact_type_mismatch_raises(self, tmp_path: Path):
        manifest = _manifest()
        _write_bundle(
            tmp_path,
            manifest=manifest,
            provenance_by_id={
                "flood_water_extent": _provenance(artifact_type=ArtifactType.FEATURE_STACK)
            },
        )
        with pytest.raises(DashboardContractError, match="artifact_type"):
            load_manifest(tmp_path / "dashboard_manifest.json")

    def test_manifest_file_missing_raises(self, tmp_path: Path):
        with pytest.raises(DashboardContractError, match="not found"):
            load_manifest(tmp_path / "dashboard_manifest.json")

    def test_invalid_json_raises(self, tmp_path: Path):
        (tmp_path / "dashboard_manifest.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(DashboardContractError, match="not valid JSON"):
            load_manifest(tmp_path / "dashboard_manifest.json")

    def test_production_layer_cannot_link_validation_only_provenance(self, tmp_path: Path):
        """AGENTS.md §3 — a provenance naming a forbidden source must never load."""
        manifest = _manifest()
        (tmp_path / "layers").mkdir(parents=True, exist_ok=True)
        (tmp_path / "layers/flood_water_extent.tif").write_bytes(b"placeholder")
        (tmp_path / "layers/flood_water_extent_provenance.yaml").write_text(
            "artifact_type: segmentation_prediction\n" "production_inputs:\n" "  - emsr927\n",
            encoding="utf-8",
        )
        (tmp_path / "dashboard_manifest.json").write_text(
            json.dumps(manifest.model_dump(by_alias=True, mode="json")), encoding="utf-8"
        )
        with pytest.raises(DashboardContractError, match="provenance"):
            load_manifest(tmp_path / "dashboard_manifest.json")


class TestDashboardIsPresentationOnly:
    """AGENTS.md §16 — the dashboard consumes artifacts, it never runs science.

    A text-level guard mirroring ``test_data_boundary.py``'s approach: the
    dashboard contract package may import provenance types and nothing else from
    the science pipeline. Any future import of acquisition, preprocessing,
    feature, segmentation, network, hydrology or evaluation logic here is a
    coupling defect this test exists to catch.
    """

    #: Modules under floodmap.* that a presentation layer must never import.
    SCIENCE_MODULES = (
        "floodmap.acquisition",
        "floodmap.datasets",
        "floodmap.preprocessing",
        "floodmap.features",
        "floodmap.segmentation",
        "floodmap.network",
        "floodmap.hydrology",
        "floodmap.evaluation",
        "floodmap.infrastructure",
        "floodmap.reporting",
    )

    @staticmethod
    def _science_imports(path: Path) -> list[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for candidate in TestDashboardIsPresentationOnly.SCIENCE_MODULES:
                    if node.module.startswith(candidate):
                        offenders.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for candidate in TestDashboardIsPresentationOnly.SCIENCE_MODULES:
                        if alias.name.startswith(candidate):
                            offenders.append(alias.name)
        return sorted(set(offenders))

    def test_no_science_module_is_imported(self):
        from tests.conftest import REPO_ROOT

        package = REPO_ROOT / "src" / "floodmap" / "dashboard"
        offenders: dict[str, list[str]] = {}
        for path in sorted(package.rglob("*.py")):
            imports = self._science_imports(path)
            if imports:
                offenders[path.relative_to(REPO_ROOT).as_posix()] = imports
        assert not offenders, (
            "the dashboard presentation layer must not import production science; "
            f"offenders: {offenders}"
        )

    def test_guard_actually_scans_files(self):
        from tests.conftest import REPO_ROOT

        package = REPO_ROOT / "src" / "floodmap" / "dashboard"
        files = list(package.rglob("*.py"))
        assert len(files) >= 3, "the presentation-only guard scanned too few files"

    def test_unavailable_layers_are_explicitly_absent(self, tmp_path: Path):
        manifest = _manifest(
            layers=[
                _layer(),
                _layer(
                    layer_id="road_connectivity",
                    title="Potential road disruption",
                    artifact_type=ArtifactType.ROAD_DISRUPTION,
                    evidence_tier=EvidenceTier.NETWORK_INFERENCE,
                    available=False,
                    unavailable_reason="road extraction not run for demo AOI",
                    path=None,
                    provenance_path=None,
                ),
            ]
        )
        _write_bundle(
            tmp_path,
            manifest=manifest,
            provenance_by_id={"flood_water_extent": _provenance()},
        )
        loaded = load_manifest(tmp_path / "dashboard_manifest.json")
        missing_layer = loaded.manifest.layers[1]
        assert missing_layer.available is False
        assert loaded.resolve_path(missing_layer) is None
        assert loaded.provenance["road_connectivity"] is None
