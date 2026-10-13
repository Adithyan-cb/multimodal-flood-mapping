"""Flood impact dashboard (Multimodal AI Hackathon 2026, Track B).

This is a visualization and decision-support layer only. ``AGENTS.md`` §16 and
``architecture.md`` §15: it consumes generated artifacts through the dashboard
contract and runs NO acquisition, preprocessing, segmentation, network or
hydrology logic. Every number shown is read from an artifact; nothing is
computed here.

Evidence tiers are rendered as separate, explicitly labelled toggle groups
(``AGENTS.md`` §16, ``docs/scientific-assumptions.md`` §10.6). A validation
reference can never be presented as an input. When the manifest is synthetic,
an unmissable banner is shown.

Run with:
    uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
from folium.utilities import image_to_url
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium

from floodmap.dashboard import (
    DashboardContractError,
    EvidenceTier,
    LayerFormat,
    build_demo_bundle,
    load_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "dashboard_demo" / "dashboard_manifest.json"

TIER_LABELS: dict[EvidenceTier, str] = {
    EvidenceTier.OBSERVED_DERIVED: "Observed / derived",
    EvidenceTier.MODEL_PREDICTION: "Model prediction",
    EvidenceTier.SPATIAL_INFERENCE: "Spatially inferred",
    EvidenceTier.NETWORK_INFERENCE: "Network inference",
    EvidenceTier.VALIDATION_REFERENCE: "Validation reference",
}

st.set_page_config(page_title="Flood impact dashboard", layout="wide")


def _load_bundle():
    """Return the loaded bundle, materializing the synthetic demo if needed."""
    if not DEFAULT_MANIFEST.exists():
        build_demo_bundle(DEFAULT_MANIFEST.parent)
    try:
        return load_manifest(DEFAULT_MANIFEST)
    except DashboardContractError as exc:
        st.error(f"Manifest contract failure: {exc}")
        st.stop()


def _normalize(band: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Percentile-normalize a band to 0..1 for display (presentation only)."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype="uint8")
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        return np.zeros_like(band, dtype="uint8")
    out = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out.astype("float32")


def _flood_mask_rgba(data: np.ndarray) -> np.ndarray:
    """Flood water mask -> translucent blue overlay."""
    height, width = data.shape
    rgba = np.zeros((height, width, 4), dtype="uint8")
    flooded = data > 0
    rgba[flooded] = np.array([30, 144, 255, 170], dtype="uint8")
    return rgba


def _confidence_rgba(data: np.ndarray) -> np.ndarray:
    """Confidence 0..1 -> orange-red heat overlay, more opaque when confident."""
    height, width = data.shape
    rgba = np.zeros((height, width, 4), dtype="uint8")
    conf = np.clip(data.astype("float32"), 0.0, 1.0)
    low = np.array([59, 76, 192], dtype="float32")
    mid = np.array([221, 130, 58], dtype="float32")
    high_c = np.array([240, 59, 32], dtype="float32")
    for y in range(height):
        for x in range(width):
            c = conf[y, x]
            if c <= 0.0:
                continue
            if c < 0.5:
                colour = low + (mid - low) * (c / 0.5)
            else:
                colour = mid + (high_c - mid) * ((c - 0.5) / 0.5)
            rgba[y, x, :3] = colour.astype("uint8")
            rgba[y, x, 3] = int(40 + 200 * c)
    return rgba


def _gray_rgba(band: np.ndarray) -> np.ndarray:
    gray = (np.clip(band, 0.0, 1.0) * 255.0).astype("uint8")
    return np.dstack([gray, gray, gray, np.full_like(gray, 255)])


def _rgb_rgba(bands: np.ndarray) -> np.ndarray:
    height, width = bands.shape[1:]
    stacked = [(_normalize(bands[i]) * 255.0).astype("uint8") for i in range(bands.shape[0])]
    alpha = np.full((height, width), 255, dtype="uint8")
    return np.dstack([*stacked, alpha])


def _layer_rgba(path: Path, layer_id: str) -> np.ndarray:
    """Read a geotiff and turn it into an RGBA display array (presentation only)."""
    with rasterio.open(path) as src:
        count = src.count
        if count == 1:
            if layer_id == "flood_water_extent":
                return _flood_mask_rgba(src.read(1))
            if layer_id == "flood_water_confidence":
                return _confidence_rgba(src.read(1))
            return _gray_rgba(_normalize(src.read(1)))
        if count == 3:
            return _rgb_rgba(src.read())
        return _gray_rgba(_normalize(src.read(1)))


def _wgs84_bounds(path: Path):
    with rasterio.open(path) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return south, west, north, east


def _center_bounds(bundle):
    """Center the map on the first available geotiff layer's footprint."""
    for layer in bundle.manifest.layers:
        if layer.available and layer.format is LayerFormat.GEOTIFF:
            south, west, north, east = _wgs84_bounds(bundle.resolve_path(layer))
            return [(south + north) / 2.0, (west + east) / 2.0]
    return [28.0, 85.1]


def _colour_for_feature(layer_id: str, props: dict) -> str:
    """Traffic-light styling for vector overlays. Presentational only."""
    if layer_id == "buildings_exposed" and props.get("intersects_flood_extent"):
        return "red"
    if layer_id == "roads":
        return "red" if props.get("potentially_disrupted") else "gray"
    if layer_id == "settlements":
        return "red" if props.get("potentially_cut_off") else "green"
    return "blue"


def _add_layers(folium_map, bundle, enabled: set[str]) -> None:
    for layer in bundle.manifest.layers:
        if not layer.available or layer.layer_id not in enabled:
            continue
        path = bundle.resolve_path(layer)
        label = f"{layer.title} ({TIER_LABELS[layer.evidence_tier]})"
        if layer.format is LayerFormat.GEOTIFF:
            try:
                south, west, north, east = _wgs84_bounds(path)
                rgba = _layer_rgba(path, layer.layer_id)
                url = image_to_url(rgba)
                overlay = folium.raster_layers.ImageOverlay(
                    image=url, bounds=[[south, west], [north, east]], name=label, opacity=1.0
                )
            except Exception:
                continue
        elif layer.format is LayerFormat.GEOJSON:
            overlay = folium.GeoJson(
                str(path),
                name=label,
                style_function=lambda feature, l_id=layer.layer_id: {
                    "color": _colour_for_feature(l_id, feature.get("properties", {}))
                },
            )
        else:
            continue
        overlay.add_to(folium_map)
    folium.LayerControl(collapsed=True).add_to(folium_map)


def _render_sidebar(bundle) -> dict:
    st.sidebar.header("Layers")
    st.sidebar.caption(
        "Grouped by evidence strength (strongest first). Each layer's provenance "
        "and limitations are shown under the map."
    )
    selected: dict[str, bool] = {}
    for tier in (
        EvidenceTier.OBSERVED_DERIVED,
        EvidenceTier.MODEL_PREDICTION,
        EvidenceTier.SPATIAL_INFERENCE,
        EvidenceTier.NETWORK_INFERENCE,
        EvidenceTier.VALIDATION_REFERENCE,
    ):
        tiers = bundle.manifest.layers_by_tier()[tier]
        available = [layer for layer in tiers if layer.available]
        if not available:
            continue
        st.sidebar.markdown(f"**{TIER_LABELS[tier]}**")
        for layer in available:
            default = layer.layer_id in {
                "s1_pre",
                "s1_post",
                "flood_water_extent",
                "flood_water_confidence",
                "buildings_exposed",
                "roads",
                "settlements",
            }
            selected[layer.layer_id] = st.sidebar.checkbox(
                layer.title, key=f"layer-{layer.layer_id}", value=default
            )
    st.sidebar.markdown("---")
    # Explicit absence (architecture.md §17): an unavailable layer is shown,
    # not omitted.
    unavailable = [layer for layer in bundle.manifest.layers if not layer.available]
    if unavailable:
        with st.sidebar.expander("Unavailable layers"):
            for layer in unavailable:
                reason = layer.unavailable_reason
                st.caption(f"**{layer.title}** unavailable.")
                st.caption(f"Reason: {getattr(reason, 'note', reason)}")
    st.sidebar.markdown("---")
    st.sidebar.caption("Attribution")
    for line in bundle.manifest.attribution:
        st.sidebar.caption(line)
    return selected


def _render_provenance(bundle, enabled: dict[str, bool]) -> None:
    st.markdown("## Evidence & provenance")
    st.caption(
        "Every displayed layer is traceable to its source chain (AGENTS.md §22): "
        "source data → processing → model → metric/rule → derived result → uncertainty."
    )
    shown = 0
    for layer in bundle.manifest.layers:
        if not layer.available or not enabled[layer.layer_id]:
            continue
        shown += 1
        provenance = bundle.provenance[layer.layer_id]
        tier = TIER_LABELS[layer.evidence_tier]
        with st.expander(f"{layer.title} — {tier}", expanded=False):
            st.markdown(
                f"**Production inputs:** {', '.join(p for p in provenance.production_inputs)}"
            )
            st.markdown("**Known limitations:**")
            for limitation in provenance.limitations:
                st.markdown(f"- {limitation}")
            if provenance.notes:
                st.markdown(f"**Notes:** {provenance.notes}")
            if provenance.model_version or provenance.preprocessing_version:
                st.markdown(
                    f"**Versions:** model={provenance.model_version or '—'}, "
                    f"preprocessing={provenance.preprocessing_version or '—'}"
                )
            st.markdown("**Attribution:**")
            for line in provenance.attribution:
                st.markdown(f"- {line}")
    st.markdown("---")
    if shown == 0:
        st.caption("No layers selected. Use the sidebar to enable layers.")


def main() -> None:
    bundle = _load_bundle()

    if bundle.manifest.is_synthetic:
        st.warning(
            "DEMO — this dashboard is showing SYNTHETIC PLACEHOLDER data, not a "
            "real measurement or result. It exists to exercise the viewer and the "
            "artifact contract, and must never be presented as a flood map."
        )

    st.title("Flood impact dashboard")
    st.markdown(f"**AOI:** {bundle.manifest.aoi_id} — **Event:** {bundle.manifest.event_name}")
    if bundle.manifest.event_date:
        st.caption(f"Event date: {bundle.manifest.event_date}")

    st.caption(
        "Layer groups (strongest → weakest): Observed/derived · Model prediction · "
        "Spatially inferred · Network inference · Validation reference."
    )

    enabled = _render_sidebar(bundle)
    centre = _center_bounds(bundle)
    overview = folium.Map(location=centre, zoom_start=13, tiles="OpenStreetMap")
    _add_layers(overview, bundle, {k for k, v in enabled.items() if v})
    st_folium(overview, width="100%", height=560, key="main-map")

    _render_provenance(bundle, enabled)

    st.caption("Attribution (full): " + " ".join(bundle.manifest.attribution))


if __name__ == "__main__":
    main()
