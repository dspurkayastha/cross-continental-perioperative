"""Phase 1 inferential statistics for the JAMIA revision.

Every module in this package reads the canonical predictions parquet at
``artifacts/predictions/external_validation_predictions.parquet`` and
emits a CSV under ``results/tables/``. The master orchestrator
``run_phase1.py`` also emits ``results/paper_numbers.csv``, the single
source of truth for numerical claims in the manuscript.

Reproducibility: every random draw uses a deterministic seed derived
from the per-module SEED constant. Run ``run_phase1.py`` to regenerate
every output in one pass.
"""
