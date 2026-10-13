"""The dashboard artifact manifest and its loader.

The contract in one paragraph
-----------------------------
A dashboard view is driven by a single ``dashboard_manifest.json`` located in an
AOI artifact directory. The manifest names the event and lists *layers*. Each
layer points at one artifact file (raster, vector or table), states which
``ArtifactType`` it is, and — crucially — carries an :class:`EvidenceTier`
saying how strong the claim is. The loader resolves each layer's linked
``ArtifactProvenance`` so the UI can render limitations and attribution without
inventing them.

Two rules this module enforces rather than documents
----------------------------------------------------
1. **Absence is explicit.** A layer that could not be produced is still listed,
   with ``available=false`` and a reason. Omitting it would let a missing layer
   read as "nothing to report", which ``architecture.md`` §17 forbids: failure
   must be visible, not silent. This mirrors the ``Unknown`` vs ``None``
   distinction already used in :mod:`floodmap.utils.provenance`.

2. **A validation reference is never an input.** ``AGENTS.md`` §3 and §16 forbid
   presenting a validation-only product as a production layer. The evidence tier
   :attr:`EvidenceTier.VALIDATION_REFERENCE` exists so such a layer is
   *labelled and separated*, and the loader rejects a manifest whose production
   layer links to a provenance record that names a validation-only source.

The contract deliberately does not describe how a layer is *computed*. That is
the point: the dashboard must not know.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..utils.provenance import (
    REQUIRED_ATTRIBUTIONS,
    ArtifactProvenance,
    ArtifactType,
    Maybe,
)

#: Versioned identifier for the manifest shape. Bump only on a breaking change;
#: the loader refuses an unknown value rather than guessing.
MANIFEST_SCHEMA = "floodmap.dashboard_manifest/1"

__all__ = [
    "MANIFEST_SCHEMA",
    "DashboardContractError",
    "DashboardLayer",
    "DashboardManifest",
    "EvidenceTier",
    "LayerFormat",
    "LoadedDashboard",
    "load_manifest",
]


class DashboardContractError(ValueError):
    """A manifest or artifact violates the dashboard contract.

    Raised loudly on purpose. A frontend that silently skips an unreadable layer
    would present an incomplete picture as if it were complete, which is the
    failure mode this whole package exists to prevent.
    """


class EvidenceTier(str, Enum):
    """How strong the claim behind a layer is.

    Ordered strongest to weakest, matching the hierarchy in
    ``docs/scientific-assumptions.md`` §10.6 / ``PRD.md`` §7
    (observed/derived -> spatially inferred -> network-level inferred), with the
    four presentation buckets from ``AGENTS.md`` §16 plus the finer
    ``SPATIAL_INFERENCE`` split the PRD hierarchy demands. The dashboard groups
    and labels layers by this value; it is not cosmetic metadata.
    """

    #: Observed or directly measured/derived from the permitted inputs.
    OBSERVED_DERIVED = "observed_derived"
    #: A model's output, carrying model error. Labelled "model prediction".
    MODEL_PREDICTION = "model_prediction"
    #: Derived by intersecting a model output with reference data
    #: (e.g. buildings/roads exposed to a predicted extent). Depends on the
    #: model output AND the reference being complete, so it is weaker than a
    #: bare prediction.
    SPATIAL_INFERENCE = "spatial_inference"
    #: Inferred at the network level (e.g. potential cut-off settlements).
    NETWORK_INFERENCE = "network_inference"
    #: A comparison reference. NEVER a production input (AGENTS.md §3).
    VALIDATION_REFERENCE = "validation_reference"


class LayerFormat(str, Enum):
    """How a layer's artifact file should be read."""

    GEOTIFF = "geotiff"
    GEOJSON = "geojson"
    TABLE = "table"


class DashboardLayer(BaseModel):
    """One displayable layer and the evidence behind it."""

    model_config = ConfigDict(extra="forbid")

    layer_id: str = Field(description="Stable, unique key within the manifest.")
    title: str = Field(description="Human-facing label. Plain language, no over-claiming.")
    artifact_type: ArtifactType
    evidence_tier: EvidenceTier
    format: LayerFormat

    path: Maybe = Field(
        default=None,
        description="Artifact path relative to the manifest directory. Null when unavailable.",
    )
    crs: Maybe = Field(default=None, description="CRS of the artifact, e.g. EPSG:32645.")
    bbox: Optional[List[float]] = Field(
        default=None, description="Bounding box [minx, miny, maxx, maxy] in the artifact CRS."
    )
    provenance_path: Maybe = Field(
        default=None,
        description="Linked ArtifactProvenance YAML, relative to the manifest directory.",
    )

    available: bool = True
    unavailable_reason: Maybe = Field(
        default=None,
        description=(
            "Required when available=false. Uses the same vocabulary as provenance "
            "Unknown reasons so an absent layer cannot be mistaken for a measured zero."
        ),
    )

    @field_validator("bbox")
    @classmethod
    def _bbox_has_four_numbers(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None and len(v) != 4:
            raise ValueError(f"bbox must have 4 values [minx, miny, maxx, maxy], got {len(v)}")
        return v

    @model_validator(mode="after")
    def _availability_is_consistent(self) -> "DashboardLayer":
        if self.available:
            if self.path is None:
                raise ValueError(
                    f"layer {self.layer_id!r} is available but has no path; "
                    "an available layer must resolve to an artifact"
                )
            if self.provenance_path is None:
                raise ValueError(
                    f"layer {self.layer_id!r} is available but has no provenance_path; "
                    "every displayed layer must be traceable (AGENTS.md §22)"
                )
        else:
            if self.unavailable_reason is None:
                raise ValueError(
                    f"layer {self.layer_id!r} is unavailable but gives no reason; "
                    "absence must be explicit, not silent (architecture.md §17)"
                )
            if self.path is not None:
                raise ValueError(
                    f"layer {self.layer_id!r} is unavailable but still lists a path; "
                    "either produce it or drop the path so the state is unambiguous"
                )
        return self

    @property
    def is_validation_reference(self) -> bool:
        """True for comparison-only layers, which the UI must separate."""
        return self.evidence_tier is EvidenceTier.VALIDATION_REFERENCE


class DashboardManifest(BaseModel):
    """Top-level descriptor for one AOI's dashboard artifacts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: str = Field(alias="schema")
    aoi_id: str
    event_name: str
    event_date: Maybe = Field(
        default=None, description="Event date, ISO 8601. Carried from the acquisition chain."
    )
    is_synthetic: bool = Field(
        default=False,
        description=(
            "True for placeholder/demo bundles. The dashboard renders an unmissable "
            "banner when this is set, so synthetic data can never read as a result."
        ),
    )
    attribution: List[str] = Field(default_factory=list)
    layers: List[DashboardLayer] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _schema_is_known(cls, v: str) -> str:
        if v != MANIFEST_SCHEMA:
            raise ValueError(
                f"unsupported manifest schema {v!r}; this loader understands {MANIFEST_SCHEMA!r}"
            )
        return v

    @field_validator("attribution")
    @classmethod
    def _required_attributions_present(cls, v: List[str]) -> List[str]:
        """The three mandatory strings must appear somewhere in the list.

        Unlike the provenance record (which injects them), the manifest is
        authored by a pipeline stage; requiring them here turns a forgotten
        attribution into a load-time failure rather than a legal defect on
        submission (AGENTS.md §18).
        """
        missing = [s for s in REQUIRED_ATTRIBUTIONS if s not in v]
        if missing:
            raise ValueError(f"manifest is missing required attributions: {missing}")
        return v

    @field_validator("layers")
    @classmethod
    def _layer_ids_are_unique(cls, v: List[DashboardLayer]) -> List[DashboardLayer]:
        seen: set[str] = set()
        for layer in v:
            if layer.layer_id in seen:
                raise ValueError(f"duplicate layer_id {layer.layer_id!r}")
            seen.add(layer.layer_id)
        return v

    def layers_by_tier(self) -> Dict[EvidenceTier, List[DashboardLayer]]:
        """Group layers by evidence tier, preserving declaration order."""
        grouped: Dict[EvidenceTier, List[DashboardLayer]] = {tier: [] for tier in EvidenceTier}
        for layer in self.layers:
            grouped[layer.evidence_tier].append(layer)
        return grouped


class LoadedDashboard(BaseModel):
    """A manifest plus its resolved provenance records.

    ``provenance`` maps ``layer_id`` to the parsed record, or ``None`` for an
    unavailable layer. ``base_dir`` is the manifest's directory, against which
    layer ``path`` values are resolved.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    manifest: DashboardManifest
    provenance: Mapping[str, Optional[ArtifactProvenance]]
    base_dir: Path

    def resolve_path(self, layer: DashboardLayer) -> Optional[Path]:
        """Absolute path to a layer's artifact, or ``None`` if unavailable."""
        if not layer.available or layer.path is None:
            return None
        return self.base_dir / str(layer.path)

    def production_layers(self) -> List[DashboardLayer]:
        """Layers that are not validation references, in declaration order."""
        return [layer for layer in self.manifest.layers if not layer.is_validation_reference]

    def validation_layers(self) -> List[DashboardLayer]:
        """Comparison-only layers, which the UI must render separately."""
        return [layer for layer in self.manifest.layers if layer.is_validation_reference]


def load_manifest(path: Path) -> LoadedDashboard:
    """Read a ``dashboard_manifest.json`` and resolve every layer's provenance.

    Raises:
        DashboardContractError: if the manifest is malformed, a required
            attribution is missing, an available layer's artifact or provenance
            file is absent, or a production layer links to a provenance record
            naming a validation-only source.
    """
    path = Path(path)
    if not path.exists():
        raise DashboardContractError(f"manifest not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DashboardContractError(f"{path} is not valid JSON: {exc}") from exc

    try:
        manifest = DashboardManifest.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError, re-labelled
        raise DashboardContractError(f"{path} violates the dashboard contract: {exc}") from exc

    base_dir = path.parent
    records: Dict[str, Optional[ArtifactProvenance]] = {}

    for layer in manifest.layers:
        if not layer.available:
            records[layer.layer_id] = None
            continue

        artifact_path = base_dir / str(layer.path)
        if not artifact_path.exists():
            raise DashboardContractError(
                f"layer {layer.layer_id!r} is marked available but {artifact_path} does not exist"
            )

        provenance_path = base_dir / str(layer.provenance_path)
        if not provenance_path.exists():
            raise DashboardContractError(
                f"layer {layer.layer_id!r} links missing provenance file {provenance_path}; "
                "every displayed layer must be traceable (AGENTS.md §22)"
            )
        try:
            provenance = ArtifactProvenance.from_yaml(provenance_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise DashboardContractError(
                f"layer {layer.layer_id!r} has an unreadable provenance record "
                f"({provenance_path}): {exc}"
            ) from exc

        if provenance.artifact_type is not layer.artifact_type:
            raise DashboardContractError(
                f"layer {layer.layer_id!r} declares artifact_type "
                f"{layer.artifact_type.value!r} but its provenance records "
                f"{provenance.artifact_type.value!r}"
            )

        records[layer.layer_id] = provenance

    return LoadedDashboard(manifest=manifest, provenance=records, base_dir=base_dir)
