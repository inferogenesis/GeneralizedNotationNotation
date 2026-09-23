# cpomdp Analysis Sub-module

## Overview

Framework-specific analysis module for cpomdp simulation outputs. Reads `simulation_results.json` produced by the cpomdp runner (always `model_kind == "continuous"`) and generates belief-mean trajectory, per-policy expected free energy (EFE), and epistemic/pragmatic split plots through the shared `flat_payload_analyzer` engine.

## Architecture

```
cpomdp/
├── __init__.py      # Package exports
└── analyzer.py      # FlatPayloadSpec binding for cpomdp results
```

## Key Functions

- **`generate_analysis_from_logs(results_dir, output_dir=None, verbose=False)`** — Main entry point; reads cpomdp JSON results and produces `cpomdp_analysis.json` plus matplotlib visualizations.
- **Continuous metrics** — `rmse_vs_true`, `mean_efe`, `mean_epistemic`, `mean_pragmatic`, `control_mode`, `n_policies` (the categorical entropy/confidence metrics are skipped for Gaussian means).
- **EFE panel** — per-policy `efe_history` plus the epistemic/pragmatic terms per step.

## Dependencies

- `numpy`, `matplotlib` (required)
- cpomdp (runtime, for producing the results analyzed here)

## Parent Module

See [analysis/AGENTS.md](../AGENTS.md) for the overall analysis architecture.

## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
