# cpomdp Renderer

`src/gnn/render/cpomdp/` renders a **continuous (linear-Gaussian)** GNN model into a standalone [cpomdp](https://github.com/inferogenesis/cpomdp) simulation script: exact Kalman filtering plus an exhaustive expected-free-energy (EFE) search over a declared finite action set. Discrete POMDP specs are refused with the render status `unsupported`, the mirror image of PyMDP facing a continuous model.

## Usage

```python
from pathlib import Path
from gnn.render.cpomdp import render_gnn_to_cpomdp

success, msg, artifacts = render_gnn_to_cpomdp(
    gnn_spec=parsed_spec,
    output_path=Path("output/11_render_output/model/cpomdp/model_cpomdp.py"),
    options={"control_mode": "efe", "horizon": 1, "action_scale": 0.5},
)
```

## Options

| Option | `ModelParameters` key | Default | Meaning |
|---|---|---|---|
| `control_mode` | `cpomdp_control_mode` | `efe` | `efe`: first action of the EFE-minimising policy. `parity`: GNN's `u_t = control_gain * (goal_mean - mu_t)` on cpomdp's filter (same seed, same numbers as the `jax` backend). |
| `action_scale` | `cpomdp_action_scale` | `0.5` | Magnitude of the compass-rose actions (8 directions plus null for 2-D control; ± per axis plus null otherwise). |
| `horizon` | `cpomdp_horizon` | `1` | Policy length H; the search enumerates all `|A|^H` sequences and re-plans every step. |
| `goal_precision` | `cpomdp_goal_precision` | `1.0` | Scale of the identity precision on the goal (state space for the objective, observation space for the EFE preference). |

Explicit `options` win over the `ModelParameters` hints. Specs without `goal_mean`/`control_gain` render as passive trackers (no control matrix, no EFE search).

## Output

One `.py` file. When executed it writes `simulation_results.json` under `CPOMDP_OUTPUT_DIR` (default `.`) with the continuous result schema (`beliefs`, `posterior_cov`, `true_states_continuous`, `observations_continuous`, `controls`, `rmse_vs_true`, `validation`) plus `efe_history` (per step, per policy), `epistemic_term`, `pragmatic_term`, `selected_policy_index`, `n_policies`, `search_warrant`, `sensor_kind`, `R_x_family` and `cpomdp_version`. The script exits 1 when `validation.all_valid` is false.

## Dependencies

- render-time: `numpy` (no cpomdp import; the renderer is glue only)
- generated script run-time: `cpomdp`, `jax`, `numpy` (`uv sync --extra cpomdp`)

## See Also

- [AGENTS.md](AGENTS.md) — Architecture documentation
- [SPEC.md](SPEC.md) — Technical specification
- [SKILL.md](SKILL.md) — Capability API
- [docs/gnn/implementations/cpomdp.md](../../../../docs/gnn/implementations/cpomdp.md) — Framework guide
