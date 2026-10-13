# Multimodal AI Hackathon 2026 --- Track B

## Mapping Flood Damage from Space

A research-grade geospatial AI prototype for mapping flood/debris
impact, estimating infrastructure disruption, and identifying
settlements that may be cut off from road networks using permitted
satellite, terrain, and pre-event OpenStreetMap data.

> **Case study:** August 2026 Trishuli flood, Nepal\
> **Primary AI track:** Flood/debris segmentation with evaluation on
> unseen Himalayan scenes.

## 1. Problem

The challenge asks us to build a system that can take an area and flood
date and answer three operational questions:

1.  Where did the flood hit?
2.  What infrastructure was affected?
3.  Which settlements may have lost road connectivity to the nearest
    town or hospital?

The official challenge describes the August 2026 Trishuli disaster as
involving an avalanche of ice and rock followed by flood water, mud, and
debris along the Bhote Koshi--Trishuli corridor. The challenge
emphasizes rapid, trustworthy answers when roads, bridges, sensors, and
communications may fail.

## 2. System objective

The system will implement the following end-to-end pipeline:

```text
Area + event date
        |
        v
Satellite / terrain / OSM acquisition
        |
        v
Preprocessing + co-registration
        |
        v
Pre/post-event multimodal features
        |
        v
Flood/debris segmentation
        |
        +-------------------+
        |                   |
        v                   v
Infrastructure impact   Road-network analysis
        |                   |
        |                   v
        |              Connectivity graph
        |                   |
        +----------+--------+
                   |
                   v
          Potentially cut-off
             settlements
                   |
                   v
       Map dashboard + reports
```

A bonus hydrology component may trace a downstream flood path from an
upstream point using elevation data and list settlements along the path.

## 3. Scientific design principles

The project is designed around five principles:

- **Data legality:** only permitted inputs enter the production
  pipeline.
- **No reference-map leakage:** EMSR927 and other published damage
  maps are used only for validation/comparison.
- **Reproducibility:** acquisition, preprocessing, model inference,
  network analysis, and evaluation are versioned.
- **Independent evaluation:** the segmentation model must be evaluated
  on Himalayan scenes not used for training.
- **Honest uncertainty:** the system must distinguish what is directly
  observed from what is inferred.

The system must not claim that a flooded road is necessarily physically
destroyed. Where evidence supports only accessibility disruption, the
output should be described as potential flood-induced road disruption.

## 4. Allowed and prohibited data

### Allowed inputs

- Sentinel-1
- Sentinel-2
- Copernicus DEM / WorldDEM-30 as specified by the challenge
- OpenStreetMap using a pre-event snapshot
- Listed permitted training datasets

The challenge specifically recommends comparing Sentinel-1 acquisitions
from the same orbit track for change detection because different tracks
view terrain from different angles.

Milestones 1 and 2 implement the metadata acquisition and analysis-ready
preprocessing boundaries for these inputs.
`scripts/acquire.py` accepts a caller-supplied EPSG:4326 AOI and explicit
temporal windows, discovers CDSE Sentinel-1/Sentinel-2 metadata, applies
deterministic selection, and writes an auditable JSON/YAML manifest. Discovery
does not download products; downloads require the explicit `--download` flag.
M2 consumes that manifest plus explicit local source-product paths and writes
validated, aligned rasters, valid-observation masks, QA metadata and linked
provenance. It does not invent the real AOI or run production preprocessing
while the target grid and source products remain unresolved.

M3 consumes those M2 artifacts and writes model-ready features (§5.1).

### Validation-only sources

- Copernicus EMS / EMSR927
- Other published damage maps
- Post-event OSM edits

These may be used to check results but must never become model or
production-system inputs.

## 5. AI component

### 5.1 Feature generation (implemented)

Milestone 3 converts M2 analysis-ready products into model-ready features. It
is the last stage before the model, and it deliberately stops there: **M3
applies no threshold, produces no class label, and makes no flood/debris
decision.** A feature is evidence for the segmentation model, not a finding
about the surface.

Every feature is declared in a machine-readable registry written beside the
raster, carrying its physical meaning, source sensor, inputs, exact formula,
units, dtype, declared range, nodata policy, version and scientific rationale.
A later model can therefore inspect `feature_registry.json` and know exactly
what each input channel represents. Full specification:
`docs/data-contract.md` §0.6.

| Family | Features | What it represents |
|---|---|---|
| Sentinel-1 level | `s1_<pol>_pre`, `s1_<pol>_post` | Absolute backscatter on each date. Retained so a change value can be read relative to its starting point: a given decrease means something different over permanent water than over a vegetated slope. |
| Sentinel-1 change | `s1_<pol>_change_db` | Log-ratio change in dB. Speckle is multiplicative, so a ratio makes it additive with approximately terrain-independent statistics, and the dB form is symmetric for reciprocal changes. |
| Sentinel-1 polarimetric | `s1_vv_vh_ratio_db_pre`, `_post` | Co/cross-polarised ratio, which responds to scattering mechanism and separates dark-but-rough from dark-and-smooth surfaces. |
| Sentinel-2 reflectance | `s2_<band>_pre`, `_post`, `_change` | Spectral state and per-band change. |
| Sentinel-2 indices | NDWI, MNDWI, NDVI — pre, post and change | Water- and vegetation-sensitive indices. The **change** variants are what separate new inundation from permanent water: a river channel scores high on both dates and changes near zero. |
| Terrain | `dem_elevation`, `dem_slope_degrees` | Hydrological plausibility and SAR-geometry error stratification. Radar shadow in steep terrain is a systematic false-positive mechanism for water detection. |

Four properties are worth stating because they are where this kind of pipeline
usually goes quietly wrong:

- **Validity propagates and is never repaired.** A feature pixel is valid only
  where every contributing input pixel is valid. No invalid input is replaced
  by a plausible value, and no epsilon regularises a zero denominator —
  flooring a zero backscatter at a small positive number would manufacture a
  water-like value out of a non-observation. Masks are written per feature,
  because a SAR change feature can be valid where an optical index is
  cloud-masked.
- **Band identity is by name, never by position.** Features bind logical
  *roles* (`green`, `nir`, `vv`) to the band descriptions M2 actually wrote, so
  an unverified band set fails by name instead of resolving to a guess.
- **Backscatter representation is an explicit gate.** In dB the log-ratio *is*
  the difference; in linear power it is `10·log10(post/pre)`. Applying a
  logarithm to dB data produces a plausible-looking, physically meaningless
  raster, so the representation must be declared.
- **M3 never resamples.** Spatial normalisation belongs to M2, so a grid
  disagreement is an explicit failure rather than a second interpolation.

Normalisation is off: M3 emits physical quantities. When the segmentation
milestone enables it, M3 will *apply* training-split statistics and never *fit*
them — the only data M3 holds is the scene being processed, and fitting on it
would be test-time leakage.

**No production feature set has been generated.** The engine is tested against
synthetic M2-compatible artifacts only, and a production run fails explicitly
until the delivered band/polarisation bindings and target grid are supplied.

### 5.2 Segmentation model (not implemented)

The primary AI component is a flood/debris segmentation model trained
using the permitted training datasets.

The benchmark must report performance on unseen Himalayan scenes.
Planned metrics include:

- Intersection over Union (IoU)
- Dice / F1
- Precision
- Recall
- Class-specific performance where labels permit
- Error analysis by terrain, cloud/visibility conditions, and scene
  characteristics where feasible

The exact model architecture will be selected after the data contract
and baseline are established rather than being hard-coded prematurely.

## 6. Impact analysis

### Infrastructure

Pre-event OSM features will be overlaid with the predicted flood/debris
footprint.

Candidate feature classes include:

- buildings
- roads
- bridges

The result should express spatial exposure and inferred impact
separately.

### Road connectivity

The road network will be represented as a graph.

```text
roads -> graph
          |
          v
flood/disruption mask
          |
          v
remove or mark affected segments
          |
          v
recompute connectivity
          |
          v
settlements without viable
connection to reference hubs
```

The output is a **potentially cut-off settlement** assessment, not a
claim that residents are definitively inaccessible.

## 7. Dashboard

The dashboard should expose evidence rather than hide it.

Core layers:

- pre-event imagery
- post-event imagery
- predicted flood/debris extent
- confidence/uncertainty
- buildings affected/exposed
- affected road segments
- bridges
- road-network connectivity
- potentially cut-off settlements
- optional DEM/flood-path analysis

The UI should allow a judge to move from a regional overview to the
evidence supporting an individual impact.

## 8. Situation report

The system will generate a one-page situation report in:

- English
- Nepali

Every numerical claim must originate from system-generated map/analysis
outputs. The language model must not invent statistics.

## 9. Case study

The August 2026 Trishuli flood is the primary demonstration case.

EMSR927 will be used for post-hoc comparison and validation only.

The case-study workflow is:

```text
Permitted raw data
       |
       v
our pipeline
       |
       v
our prediction
       |
       +-------> EMSR927 comparison
       |
       v
quantitative + qualitative evaluation
```

## 10. Repository structure

Actual current state. Directories marked *(empty)* exist as documented
namespaces awaiting their milestone; `(planned)` entries do not exist yet.

```text
.
├── README.md
├── PRD.md
├── architecture.md
├── AGENTS.md
├── pyproject.toml
├── .gitignore
├── configs/                      # pipeline configuration (placeholders, many TODO)
│   ├── data.yaml
│   ├── preprocessing.yaml
│   ├── features.yaml
│   ├── segmentation.yaml
│   └── evaluation.yaml
├── docs/
│   ├── data-contract.md          # expected interface per data source
│   ├── dataset-registry.md       # permitted vs validation-only datasets
│   ├── scientific-assumptions.md # what the evidence does and does not support
│   ├── evaluation-protocol.md    # splits, metrics, leakage prevention
│   ├── m4-architecture-decision.md # audited datasets -> segmentation design
│   ├── m4-dataset-adapters.md    # training-corpus adapter boundary (M4)
│   └── dashboard-artifact-contract.md # dashboard reads artifacts, not science
├── data/                         # contents git-ignored (AGENTS.md §15)
│   ├── raw/                      (empty)
│   ├── interim/                  (empty)
│   ├── processed/                (empty)
│   └── samples/                  (empty; small licensed examples may be committed)
├── src/
│   └── floodmap/
│       ├── __init__.py
│       ├── acquisition/          # CDSE provider, selection and manifests (M1)
│       ├── preprocessing/        # validation, alignment and M2 artifacts
│       ├── features/             # feature registry, transforms and M3 artifacts
│       ├── datasets/             # permitted training-corpus adapters (M4)
│       ├── segmentation/         (empty — namespace only)
│       ├── infrastructure/       (empty — namespace only)
│       ├── network/              (empty — namespace only)
│       ├── hydrology/            (empty — namespace only)
│       ├── evaluation/           (empty — namespace only)
│       ├── reporting/            (empty — namespace only)
│       ├── dashboard/            # manifest contract + synthetic demo generator
│       └── utils/
│           ├── provenance.py     # artifact provenance schema
│           └── config.py         # YAML configuration loader
├── tests/                        # M0 + M1 + M2 + M3 + M4 + dashboard offline tests
│   ├── conftest.py
│   ├── test_package_structure.py
│   ├── test_configs.py
│   ├── test_provenance.py
│   ├── test_data_boundary.py     # enforces the AGENTS.md §3 data rule
│   ├── test_acquisition_m1.py    # provider, selection and manifest tests
│   ├── test_preprocessing_m2.py  # tiny synthetic raster preprocessing tests
│   ├── test_features_m3.py       # feature formulas, masking, registry, leakage
│   ├── test_contract_freeze.py   # frozen C1-C14 decision schema
│   ├── test_dataset_adapters.py  # M4 adapter behaviour (offline + external roots)
│   ├── test_dashboard_contract.py # manifest contract + presentation-only guard
│   ├── test_dashboard_demo.py    # synthetic bundle determinism and labelling
│   └── test_dashboard_app.py     # Streamlit smoke tests via AppTest
├── models/                       (empty; git-ignored)
├── artifacts/                    (empty; git-ignored)
├── notebooks/                    (empty)
├── scripts/
│   └── acquire.py                # metadata discovery / explicit download CLI
├── dashboard/
│   └── app.py                    # Streamlit viewer (presentation layer only)
└── reports/                      (planned — not created)
```

Raw and large derived datasets must remain outside Git history unless
explicitly small enough and legally appropriate.

## 11. Deliverables

**None of the following exist yet.** This list is the target scope.

- Interactive map dashboard
- One-page system-generated situation report
- GitHub repository with reproducible setup/run instructions
- Maximum 6-page technical report
- 3-minute demo video
- Required data attributions and training-dataset citations
- Trishuli case-study results compared with EMSR927

## 12. Required attribution

Every submission must include the challenge-required attribution:

> Contains modified Copernicus Sentinel data 2026.

> Produced using Copernicus WorldDEM-30 © DLR e.V. 2010--2014 and ©
> Airbus Defence and Space GmbH 2014--2018 provided under COPERNICUS by
> the European Union and ESA; all rights reserved.

> © OpenStreetMap contributors.

Training datasets must also be cited according to their respective
requirements.

## 13. Limitations

This is an educational research prototype, not an operational
emergency-response system.

Important limitations include satellite revisit frequency,
cloud/visibility constraints for optical imagery, SAR interpretation
limitations, spatial resolution, uncertainty in damage inference, OSM
completeness, temporal mismatch between imagery and the event, and the
distinction between observed inundation and actual structural failure.

The challenge explicitly notes that Sentinel satellites cannot provide a
warning minutes before a sudden glacier collapse.

## 14. Development status

Current phase: **Milestone 4 complete — training-corpus adapters, and the
dashboard artifact contract + viewer scaffolded on synthetic data.**

### What exists

| Component | Status |
|---|---|
| Repository scaffold, build config, data-exclusion rules | **Done** |
| `docs/data-contract.md` — per-source expected interface | **Done** (with TODO/UNKNOWN markers) |
| `docs/dataset-registry.md` — permitted vs validation-only | **Done** · training corpora **audited against primary sources**; no data acquired |
| `docs/m4-architecture-decision.md` — segmentation design | **Done** (decided, not implemented) |
| `docs/scientific-assumptions.md` | **Done** |
| `docs/evaluation-protocol.md` | **Done** (no metric targets, no model chosen) |
| `configs/*.yaml` — machine-readable configuration | **Done** (structural placeholders) |
| `floodmap.utils.provenance` — artifact provenance schema | **Done** |
| `floodmap.utils.config` — configuration loader | **Done** |
| Milestone 1 EO acquisition provider, selection and manifest | **Done** (offline; AOI/windows remain operator inputs) |
| Milestone 2 raster validation, masking, alignment and artifacts | **Done** (synthetic fixtures; production inputs/grid remain unresolved) |
| Milestone 3 feature registry, transforms, masking and artifacts | **Done** (synthetic fixtures; band/polarisation bindings remain unresolved) |
| Milestone 4 training-corpus adapters (`datasets/`, frozen contract) | **Done** (Kuro Siwo + Sen1Floods11, `docs/m4-dataset-adapters.md`) |
| Dashboard artifact contract + synthetic demo bundle | **Done** (`src/floodmap/dashboard/`, `docs/dashboard-artifact-contract.md`) |
| Streamlit dashboard viewer | **Done** (presentation layer; consumes artifacts, no science) |
| Test infrastructure incl. data-boundary guard | **Done** |

### What does NOT exist

The following later stages are not implemented. Their corresponding
`src/floodmap/` subpackages remain documented namespaces with no logic:

- Flood/debris segmentation model (**no architecture selected** — `AGENTS.md` §5)
- Training-corpus *training run* (the M4 adapter boundary is implemented; no
  model has been trained and no labels have been fit)
- Infrastructure exposure analysis
- Road-network graph and disruption analysis
- Settlement connectivity analysis
- DEM flood-path tracing (bonus)
- Presentation of real (non-synthetic) pipeline results in the dashboard
- Situation report generation

**No model has been trained and no performance has been measured.** Any
performance figure anywhere in this repository would be fabricated.

### Resolved by the official specification

The official Track B challenge specification has been supplied. It fixes, and
this repository now records as VERIFIED(spec):

| Item | Value |
|---|---|
| Event date | 2026-08-26 (Trishuli flood, Nepal) |
| Pre-event OSM snapshot | 2026-07-27, via the ohsome API |
| DEM source | Copernicus WorldDEM-30 |
| Permitted training datasets | Kuro Siwo (required), Sen1Floods11 (optional) |
| EMSR927 | Validation only; prohibited as a production input |
| Attribution | Three mandatory strings, enforced in code |

This clears the previously reported blocking gap on the training dataset list.

### Remaining unresolved items

Still recorded as TODO/UNKNOWN rather than guessed: AOI geometry, acquisition
search window, target grid CRS/resolution, DEM delivered grid and vertical
datum, Sentinel-1 product type and polarisations, Sentinel-2 processing level
and band set, and the reference hub definition.

Milestone 3 adds no guesses to that list. Feature generation is defined over
band *roles* rather than band names, so the unresolved polarisation and band
sets propagate as explicit, named configuration failures rather than as
defaults: `configs/features.yaml` leaves `polarisation_roles`,
`spectral_band_roles`, `band_roles`, `backscatter_representation`,
`terrain.elevation_unit` and `numerics.output_nodata` null, and a production
run fails on them.

### Resolved by the training-dataset audit

Both permitted corpora have now been audited against their own repositories,
LICENSE files and papers (`docs/dataset-registry.md` §1.5). Four findings
changed the plan:

| Finding | Consequence |
|---|---|
| **Neither corpus labels debris or sediment.** Kuro Siwo is `{No water, Permanent Waters, Floods, Invalid}`; Sen1Floods11 is `{No Data, Not Water, Water}`. | **The product claim narrows from "flood/debris" to flood water.** Supervised debris segmentation is impossible within the closed permitted list. |
| **Kuro Siwo separates permanent water from flood water.** | This is what stops the Trishuli river being reported as flood on every run — the main reason the required corpus is usable here. |
| **The two corpora disagree on SAR representation** (Kuro Siwo linear σ⁰, Sen1Floods11 dB) and on speckle filtering, and Sen1Floods11 has **no pre-event image**. | A naive union is forbidden. The M3 per-corpus representation gate becomes load-bearing, and change features cannot be supervised by Sen1Floods11 at all. |
| **Neither corpus is documented as containing Himalayan terrain.** Kuro Siwo's only Nepal-labelled activation is tropical, lowland and in its *upstream test* split. | There is **no labelled Himalayan test set** in the permitted data. The unseen-Himalaya requirement splits into a labelled proxy plus an unlabelled EMSR927 agreement, reported separately. |

Licensing is **worse than the specification implies** and is recorded
conservatively rather than asserted: Kuro Siwo's `LICENSE` says MIT while its
README says CC BY, and **Sen1Floods11 has no licence file at all**. Neither may
be redistributed from this repository.

The resulting segmentation design is recorded in
`docs/m4-architecture-decision.md`: a **SAR-learned core** with Sentinel-2 and
DEM as deterministic downstream evidence rather than unsupervised model inputs.

See `docs/dataset-registry.md` §5 for the remaining blockers.

### Progress order

1.  Repository and architecture — **done**
2.  Data provenance and legality checks — **partially done** (contract and
    registry written; licenses still require verification)
3.  Acquisition — **done (Milestone 1: metadata discovery and manifest)**
4.  Preprocessing — **done (Milestone 2: analysis-ready raster boundary)**
5.  Feature generation — **done (Milestone 3: registry-driven feature
    artifacts)**
6.  Segmentation baseline — not started (**unblocked**: permitted datasets are
    now specified and the M4 adapters are implemented)
7.  Unseen-Himalaya evaluation — blocked
8.  Infrastructure impact — not started
9.  Road-network connectivity — not started
10. Optional DEM flood-path analysis — not started
11. Trishuli case study — not started
12. Dashboard — **done (contract, synthetic demo bundle, Streamlit viewer);
    real results pending**
13. Situation report — not started
14. Final evaluation and limitations — not started
15. Demo and submission packaging — not started

## 15. Development setup

`pyproject.toml` is the single source of dependency truth. A `uv.lock` pins
the resolved tree.

```bash
# Create .venv/, install the package editable + dev + dashboard extras
uv sync --extra dev --extra dashboard

# Run the test suite
uv run pytest

# Formatting
uv run black src scripts tests

# Dashboard viewer (presentation layer; runs on the synthetic demo bundle and
# banners it as such until real pipeline artifacts exist)
uv run streamlit run dashboard/app.py
```

Without `uv`, the equivalent is `pip install -e ".[dev,dashboard]"` and
`python3 -m pytest`. The heavier `geo` extra (geopandas/osmnx/networkx) stays
opt-in: `uv sync --extra dev --extra dashboard --extra geo`.

### Acquisition quick start

Build a manifest and the exact CDSE queries without network access:

```bash
python3 scripts/acquire.py \
  --dry-run \
  --aoi-id synthetic-demo \
  --aoi-bbox 0.0 0.0 0.1 0.1 \
  --before-days 12 \
  --after-days 12 \
  --manifest artifacts/acquisition_manifest.json
```

Remove `--dry-run` for metadata discovery. Add `--download` only when product
downloads are intentionally enabled and CDSE credentials are available through
`CDSE_USERNAME` and `CDSE_PASSWORD`. The manifest records discovery failures,
empty results, rejected candidates, and download `NOT_ATTEMPTED` states
separately.

Rasterio is a runtime dependency for M2 raster validation, masking, alignment
and GeoTIFF artifact writing. The remaining geospatial stack
(`rioxarray`, `xarray`, `geopandas`, `shapely`, `pyproj`, `osmnx`, `networkx`)
remains under the optional `geo` extra and is not required by the M2 engine.
