# cpomdp Runner

Discovers and executes cpomdp-generated continuous active inference scripts via subprocess.

## Usage

```python
from gnn.execute.cpomdp import run_cpomdp_scripts

ok = run_cpomdp_scripts(
    rendered_simulators_dir="output/11_render_output", execution_output_dir="output/12_execute_output"
)
```

## Features

- Dependency validation (checks `cpomdp` + `jax` availability; install hint `uv sync --extra cpomdp`)
- Syntax pre-validation before execution
- Run-length cap: scripts declaring `NUM_TIMESTEPS > 60` are refused unless `CPOMDP_ALLOW_LONG=1` (the receding-horizon search re-plans every step)
- Log persistence (stdout/stderr capture, `execution_log.json`)
- Wall-clock execution timing
- Results routed through `CPOMDP_OUTPUT_DIR` (`simulation_results.json`)

## See Also

- [Parent: execute/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Architecture documentation
- [SPEC.md](SPEC.md) — Technical specification
