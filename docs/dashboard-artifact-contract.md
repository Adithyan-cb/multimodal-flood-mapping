# Dashboard artifact contract

Status: **implemented** (`src/floodmap/dashboard/`), exercised end-to-end by the
synthetic demo bundle (`src/floodmap/dashboard/demobundle.py`) and the Streamlit
app (`dashboard/app.py`).

## Purpose

The dashboard is a visualization and decision-support layer only
(`architecture.md` §15). It must never run acquisition, preprocessing,
segmentation, network or hydrology logic, and it must keep the evidence
strength of every claim visible (`AGENTS.md` §16, `docs/scientific-assumptions.md` §10.6).

Both rules are enforceable only if the frontend reads a **declared, tested
contract** rather than guessing file locations or computing values. This
document describes that contract. The authoritative implementation is
`src/floodmap/dashboard/contract.py`; this document mirrors it and cannot drift
silently, because `tests/test_dashboard_contract.py` pins the behaviour.

## Bundle layout

One AOI's generated dashboard inputs live in one directory:

```text
<aoi_dir>/
    dashboard_manifest.json     # top-level descriptor (this contract)
    layers/
        <stem>.tif              # raster artifact (UTM CRS, band names preserved)
        <stem>.geojson          # vector artifact (WGS84)
        <stem>_provenance.yaml  # ArtifactProvenance for that layer
```

`schema` is `floodmap.dashboard_manifest/1`. The loader rejects any other value
so a future breaking change is loud, not silent.

## Manifest fields

| Field | Meaning |
|---|---|
| `schema` | Contract version. Must equal `floodmap.dashboard_manifest/1`. |
| `aoi_id` | AOI identifier, carried from the acquisition chain. |
| `event_name`, `event_date` | Human-readable event identity. |
| `is_synthetic` | `true` only for placeholder/demo bundles. The UI shows an unmissable banner when set. |
| `attribution` | Must contain all three required strings (`AGENTS.md` §18); the manifest author cannot forget them. |
| `layers` | One `DashboardLayer` per displayable artifact, including unavailable ones. |

## Layer fields

| Field | Meaning |
|---|---|
| `layer_id` | Stable unique key within the manifest. |
| `title` | Plain-language label. No over-claiming (no "destroyed", no "isolated"). |
| `artifact_type` | Reuses `floodmap.utils.provenance.ArtifactType`. |
| `evidence_tier` | Closed enum below. This is the scientific honesty control. |
| `format` | `geotiff`, `geojson`, or `table`. |
| `path`, `crs`, `bbox` | Artifact location (relative to the manifest) and georeference. |
| `provenance_path` | Relative path to the linked `ArtifactProvenance` YAML. Mandatory for available layers. |
| `available` | Whether the layer could be produced. |
| `unavailable_reason` | Mandatory when `available` is `false`. Uses the same `Unknown`-style vocabulary as provenance. |

## Evidence tiers (strongest → weakest)

```
observed_derived   Observed or directly derived from permitted inputs
        ↓
model_prediction   A model's output; carries model error ("Predicted flood water extent")
        ↓
spatial_inference  Intersection of a prediction with reference data
                   (e.g. buildings/roads exposed). Depends on model AND reference.
        ↓
network_inference  Graph-based, e.g. "potentially cut off" settlements
        ↓
validation_reference  Comparison-only. NEVER a production input.
```

This refines the four presentation buckets of `AGENTS.md` §16 with the
`spatial_inference` split demanded by `PRD.md` §7 and
`docs/scientific-assumptions.md` §10.6. The UI renders each tier as a separate,
labelled toggle group.

## Rules enforced here rather than documented

1. **An available layer must resolve.** `path` and `provenance_path` are
   mandatory for available layers, and `load_manifest` re-checks both files
   exist and `provenance.artifact_type` agrees with the manifest entry.
2. **Absence is explicit.** An unavailable layer is *listed* with a reason
   (`architecture.md` §17). Omitting it would let "not produced" read as
   "nothing to report". If it is marked unavailable it must not keep a path.
3. **Validation references stay separated.** `validation_reference` layers are
   grouped apart in the UI, and a provenance record naming a validation-only
   source is rejected at parse time (`AGENTS.md` §3), so a comparison product
   can never appear as a production layer.
4. **The presentation layer does not run science.** A guard test
   (`TestDashboardIsPresentationOnly`) fails if `src/floodmap/dashboard/`
   imports any acquisition/preprocessing/features/segmentation/network/
   hydrology/evaluation logic.

## Synthetic demo bundle

`src/floodmap/dashboard/demobundle.py` generates a small, deterministic
(`seed=2026`) bundle under `artifacts/dashboard_demo/` so the viewer and its
tests exercise the real contract before pipeline artifacts exist. It is
unambiguously labelled: `is_synthetic=true`, every provenance note contains
`SYNTHETIC PLACEHOLDER`, and the app banners it. It includes one deliberately
unavailable layer (`flood_path_trace`) so the "explicit absence" rule is always
visible in the demo. No binaries are committed (`*.tif` is globally gitignored).

## Attribution

The manifest must carry the three required strings
(`floodmap.utils.provenance.REQUIRED_ATTRIBUTIONS`); the loader refuses a
manifest that omits any. Provenance records attach them automatically.