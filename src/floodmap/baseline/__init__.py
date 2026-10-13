"""Non-learned flood baselines.

Scientific responsibility
-------------------------
Deterministic, rule-based flood masks that follow the same M4 class-mask
contract as the learned model. They give the impact layer a real flood map
when no trained model is available, and they are the baseline any learned
model must beat (AGENTS.md §5). Kept apart from ``floodmap.segmentation`` so
the learned model's package stays exclusively the model's.

Modules
-------
* ``change_detection``: Sentinel-1 same-track pre/post change detection.
"""
