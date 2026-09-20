# cpomdp Execution — Technical Specification

## Execution Model

- Subprocess execution of the rendered script (`execute_script_safely`)
- Pre-flight: dependency check (`cpomdp`, `jax`, `numpy`), syntax check, run-length cap
- Cap: `NUM_TIMESTEPS > 60` is refused unless `CPOMDP_ALLOW_LONG=1`; the refusal is a failed execution with an explicit message, never a silent skip
- Timeout: 600 s per script by default (the Step 12 pipeline path inherits the step timeout)

## Input

- cpomdp scripts from `output/11_render_output/<model>/cpomdp/*_cpomdp.py`

## Output

- `simulation_results.json` under `CPOMDP_OUTPUT_DIR` — continuous schema plus `efe_history`, `epistemic_term`, `pragmatic_term`, `selected_policy_index`, `n_policies`, `search_warrant`, `sensor_kind`, `R_x_family`
- Execution logs (`stdout.txt`, `stderr.txt`, `execution_log.json`)
- Result key in `execution_summary.json`: `cpomdp_executions`

## Dependencies

- `cpomdp >= 0.4.4, < 0.5`, `jax` (optional `cpomdp` extra)
