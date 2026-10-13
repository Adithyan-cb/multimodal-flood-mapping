"""Streamlit app smoke tests via streamlit's AppTest harness.

These do not judge the layout; they assert the app runs end to end against the
synthetic bundle without raising, that the synthetic banner is shown, that
evidence tiers are labelled in the sidebar, and that every mutual-guarantee of
the contract surfaces in the UI (unavailable layers declared, attribution
present).
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.conftest import REPO_ROOT

APP_PATH = REPO_ROOT / "dashboard" / "app.py"


def _run_app() -> AppTest:
    return AppTest.from_file(str(APP_PATH)).run(timeout=30)


class TestDashboardAppSmoke:
    def test_app_runs_without_exception(self):
        at = _run_app()
        assert not at.exception, at.exception

    def test_synthetic_banner_is_shown(self):
        at = _run_app()
        warnings = " ".join(w.value for w in at.warning)
        assert "SYNTHETIC PLACEHOLDER" in warnings

    def test_evidence_groups_are_labelled(self):
        at = _run_app()
        markdown = [m.value for m in at.markdown]
        body = " ".join(markdown + [c.value for c in at.caption])
        assert "Model prediction" in body
        assert "Network inference" in body

    def test_unavailable_layer_is_declared(self):
        at = _run_app()
        text = " ".join(c.value for c in at.caption)
        assert "Unavailable layers" in text or "unavailable" in text

    def test_attribution_is_rendered(self):
        at = _run_app()
        text = " ".join(c.value for c in at.caption)
        assert "OpenStreetMap" in text
        assert "Copernicus Sentinel" in text
