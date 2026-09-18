---
name: gnn-cpomdp-render
description: Render continuous (linear-Gaussian) GNN models to cpomdp continuous active inference scripts. Use when a GNN file declares F/H/Q/R and you want expected-free-energy action selection rather than the proportional controller of the other continuous backends.
---

# cpomdp Render Backend (Step 11)

## Purpose

Emit a standalone cpomdp script for a continuous GNN model: exact Kalman filtering plus an exhaustive expected-free-energy search over a declared finite action set. Discrete specs report `unsupported`.

## Key Commands

```bash
uv sync --extra cpomdp
python src/gnn/11_render.py --target-dir input/gnn_files/continuous --frameworks cpomdp --verbose
python src/gnn/main.py --only-steps "3,11,12,16" --frameworks cpomdp --target-dir input/gnn_files/continuous --output-dir output --verbose
```

## API

```python
from gnn.render.cpomdp import render_gnn_to_cpomdp

ok, message, artifacts = render_gnn_to_cpomdp(spec, out_path, {"control_mode": "parity"})
```

Options: `control_mode` (`efe` | `parity`), `action_scale`, `horizon`, `goal_precision`; also readable from `ModelParameters` as `cpomdp_<option>`.

## Output

- `output/11_render_output/<model>/cpomdp/<model>_cpomdp.py`
- executed by Step 12 into `simulation_results.json` (continuous schema plus `efe_history`, `epistemic_term`, `pragmatic_term`, `selected_policy_index`)

## Troubleshooting

| Issue | Solution |
| ----- | -------- |
| `cpomdp not installed` when the script runs | `uv sync --extra cpomdp` |
| Every discrete file reports `unsupported` | Expected: cpomdp renders continuous models only |
| Long runs refused at Step 12 | Set `CPOMDP_ALLOW_LONG=1` for `num_timesteps > 60` |

## MCP Tools

The cpomdp backend is reached through the render module's MCP surface (`src/gnn/render/mcp.py`); it registers no tools of its own:

- `list_render_frameworks` — reports `cpomdp` with its availability and description
- `render_gnn_to_format` / `render_spec_to_format` — accept `cpomdp` as the target (continuous specs only; discrete specs report `unsupported`)
- `process_render` — includes `cpomdp` in `--frameworks all`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification
