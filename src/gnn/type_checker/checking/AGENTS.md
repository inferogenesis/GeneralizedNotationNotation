# Type Checker Checking Agent

## Overview
This directory owns the core structural type-checking rules for `src/gnn/type_checker/`.

## Purpose
- Implement dimension and rule checks used by Step 5.
- `sections.py` owns the single section-scoped content extraction shared by the checker and the estimator; `summary.py` owns the `ValidationSummary` aggregation.
- Keep public exports in `__init__.py` aligned with `core.py`, `sections.py`, `summary.py`, `dimensions.py`, `rules.py`, and `continuous.py`.
- `continuous.py` owns the continuous-model `R_x_family` / `R_x_params` check (`[GNN-E007]`): the family must be in `render.continuous_common.SENSOR_FAMILIES` and the parameter vector must have that family's arity.
- Keep tests in `tests/type_checker/` focused on real checker behavior.

## Verification
Run `uv run --extra dev python -m pytest tests/type_checker/ -q`.
