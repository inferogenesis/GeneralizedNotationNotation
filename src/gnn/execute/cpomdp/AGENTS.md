# cpomdp Execution Sub-module

## Overview

Discovers and runs cpomdp-generated continuous active inference scripts via subprocess. Includes dependency checking, syntax validation, a run-length cap, log persistence, and execution timing. Mirrors the NumPyro runner.

## Architecture

```
cpomdp/
├── __init__.py           # Package exports
└── cpomdp_runner.py      # cpomdp script discovery, cap, and execution
```

## Key Functions

- **`run_cpomdp_scripts(rendered_simulators_dir, execution_output_dir)`** — Discovers and executes all cpomdp scripts under `<render_dir>/cpomdp/`.
- **`execute_cpomdp_script(script_path, verbose, output_dir, timeout)`** — Runs one script through `execute_script_safely` with `CPOMDP_OUTPUT_DIR` set to `output_dir`.
- **`refuse_long_run(script_text)`** — Returns the refusal reason when the script's `NUM_TIMESTEPS` literal exceeds `MAX_TIMESTEPS_WITHOUT_OPT_IN` (60) and `CPOMDP_ALLOW_LONG` is not set.
- **`is_cpomdp_available()`** — Import gate on `cpomdp` and `jax`.

## Integration points

- `execute.executor` registers the runner as `ExecutorFrameworkSpec(framework_dir_key="cpomdp", result_key="cpomdp_executions")`.
- `execute.processor` (the Step 12 pipeline path) sets `CPOMDP_OUTPUT_DIR` for `cpomdp` scripts and gates them on the `cpomdp` import through `utils.runtime_safety.framework_availability`; missing dependency → the script is **skipped**, not failed.
- `execute.data_extractors` reads `simulation_results.json` (plus `cpomdp_outputs/*.json`) for Step 16.

## Dependencies

- `cpomdp`, `jax` (runtime, checked before execution; `uv sync --extra cpomdp`)

## Parent Module

See [execute/AGENTS.md](../AGENTS.md) for the overall execution architecture.

## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
