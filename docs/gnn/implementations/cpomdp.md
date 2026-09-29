# cpomdp Framework Implementation

> **GNN Integration Layer**: Python / JAX-based continuous active inference
> **Framework Base**: `cpomdp >= 0.4.4, < 0.5` (the continuous-state sibling of pymdp)
> **Simulation Architecture**: Exact Kalman filtering plus enumerated expected-free-energy (EFE) action search
> **Documentation Version**: 1.1.0

## Overview

[cpomdp](https://github.com/inferogenesis/cpomdp) provides a **continuous-state
active inference** backend for GNN's linear-Gaussian models. Where the other
continuous backends (JAX, NumPyro, PyTorch, Stan, RxInfer.jl) filter the
declared `F`/`H`/`Q`/`R` model and, at most, close the loop with GNN's
proportional controller, cpomdp evaluates expected free energy over a declared
finite action set and takes the first action of the policy that minimises it.
Under a fixed linear-Gaussian sensor the epistemic term of EFE is the same for
every policy (the well-known collapse), so with the current continuous contract
the value-add is the pragmatic search plus the full per-policy EFE record;
state-dependent observation noise `R(x)`, which is what lets the epistemic
term move, is the subject of the language extension described below.

The renderer and runner are glue only: no cpomdp source is vendored into GNN.
The emitted script imports the PyPI package.

## Architecture

| Stage | Module | Description |
|---|---|---|
| Rendering (Step 11) | `src/gnn/render/cpomdp/cpomdp_renderer.py` + `script_template.py` | Continuous GNN spec → standalone cpomdp program |
| Execution (Step 12) | `src/gnn/execute/cpomdp/cpomdp_runner.py` | Subprocess execution, dependency gate, 60-step cap, log persistence |
| Analysis (Step 16) | `src/gnn/analysis/cpomdp/analyzer.py` | Continuous metrics (RMSE, epistemic/pragmatic split) and EFE panels |

Registry entry: `src/gnn/render/framework_registry.py` → `cpomdp`
(`supports_continuous: True`, `supports_discrete: False`, `supports_execution: True`).

## What it renders, what it refuses

- **Renders**: every spec `detect_model_kind` classifies as CONTINUOUS — `F`
  (n×n), `H` (m×n), `Q` (n×n), `R` (m×m), `prior_mean`, `prior_cov`, optional
  `goal_mean` + `control_gain`. Specs without a goal become passive trackers
  (no control matrix, no EFE search).
- **Refuses**: discrete POMDPs. They report the render status `unsupported`
  (reason `discrete POMDP: cpomdp renders continuous (linear-Gaussian) models
  only`), recorded under `unsupported_framework_renderings` and never handed to
  Step 12 — the mirror image of PyMDP facing a continuous model. Structural
  wrapper specs keep their own `structural-spec` message.

`detect_model_kind()` and the continuous trigger (`F/H/Q/R`) are unchanged.

## The emitted program

Built in this order, all from public cpomdp API:

1. `LinearGaussianModel(dynamics=F, sensor_model=H, dynamics_noise=Q, sensor_noise=R, prior=Belief(prior_mean, prior_cov), control=I_n)` — GNN adds `u` directly to the state, so the control matrix is the identity.
2. `StateGoal(goal_mean, precision=goal_precision·I)` under a fixed `R`.
3. A compass-rose `FiniteActionSet` — 8 unit directions scaled by `action_scale` plus the null action for 2-D control (`2p + 1` axis actions otherwise), versioned `gnn-compass-rose-v1`.
4. `EnumeratedEfeSearch(model, action_set, horizon=H)` wrapped in a `RecedingHorizonSelector` (`EFESelector` is 1-D only and is not used).
5. `Agent(model, objective, selector=...)` — cpomdp's goal/sensor pairing check.

The loop drives `KalmanBackend.infer_states` directly so it can record the full
`G` vector every step (`jax.jit` of `search.evaluate`). The prior sits on
`x_1`, as in every GNN continuous script: the first observation is folded in
without a predict step. JAX is seeded from `RANDOM_SEED` with x64 enabled and
the sampling order matches `render/continuous_script.py`.

## State-dependent sensing (`R_x_family`)

The v1.2 language extension lets a continuous file declare a state-dependent
observation noise on top of the nominal `R` (see
[`gnn_syntax.md` § v1.2 Extension](../gnn_syntax.md#v12-extension--state-dependent-observation-noise-rx)):

```gnn
R_x_family=quadratic_beacon
R_x_params={(2.0, 2.0, 0.05, 1.0)}   # [cx, cy, r_min, k]
```

When the family is set, the emitted script builds a `CallableSensor` from a
small in-script library — `constant`, `quadratic_beacon`, `blind_spot`,
`beacon_and_blind_spot`, each scaling the nominal `R` by a positive factor of
the planar position — and passes it to `LinearGaussianModel` via
`observation=` (never `sensor_noise=`). The goal type follows the sensor
type: `ObservationGoal(H @ goal_mean)` under the callable sensor,
`StateGoal(goal_mean)` under a fixed `R`; cpomdp refuses the other pairing
when the `Agent` is built, and the renderer tests pin both. True
observations are sampled with `R(x_true)`, the filter linearises at the
predicted mean, and the EFE search sees the state-dependent noise. That is
what makes the epistemic term differ across policies: the results carry
`epistemic_by_policy_t0` / `pragmatic_by_policy_t0` (every enumerated
policy's split at the first step), and `sensor_kind` reads
`"state_dependent"` with `R_x_family` / `R_x_params` echoed. Under
`constant` (or any fixed `R`) the epistemic column is identical for every
policy — the collapse the extension exists to escape.

The other continuous backends keep the nominal `R` and append
`state-dependent R declared, this backend uses nominal R` to their render
message.

## Control modes

| `control_mode` | Action | Purpose |
|---|---|---|
| `efe` (default) | first action of the EFE-minimising policy, re-planned each step | the value-add |
| `parity` | `u_t = control_gain * (goal_mean - mu_t)` on cpomdp's filter; the EFE search still runs for diagnostics | Step 16 comparison against the `jax` backend on the same seed — trajectories, beliefs and controls agree to floating point (`tests/analysis/test_cpomdp_parity.py`) |

Options (`control_mode`, `action_scale` = 0.5, `horizon` = 1, `goal_precision`
= 1.0) are passed as render `options` or declared in `ModelParameters` as
`cpomdp_<option>`; explicit options win.

## Output schema additions

`simulation_results.json` is the continuous schema (`model_name`, `framework:
"cpomdp"`, `model_kind: "continuous"`, `num_timesteps`, `num_states`,
`num_observations`, `beliefs`, `posterior_cov`, `true_states_continuous`,
`observations_continuous`, `controls`, `control_mode`, `rmse_vs_true`,
`observations: []`, `actions: []`, `validation`, `execution_time_seconds`) plus:

| Key | Meaning |
|---|---|
| `efe_history` | per step, per enumerated policy `G` (T × |A|^H; `[]` for passive models) |
| `epistemic_term`, `pragmatic_term` | the split of the selected policy's `G`, per step |
| `selected_policy_index` | argmin of `G` per step |
| `horizon`, `action_scale`, `n_policies`, `search_warrant` | search configuration and cpomdp's completeness certificate (`PROVED …`) |
| `sensor_kind`, `R_x_family`, `R_x_params` | `"fixed"` / `null` / `null`, or `"state_dependent"` with the declared family and parameters |
| `epistemic_by_policy_t0`, `pragmatic_by_policy_t0` | every enumerated policy's split at the first step (constant across policies under a fixed `R`) |
| `cpomdp_version`, `jax_version` | runtime versions |

`validation` carries `means_finite`, `posterior_cov_psd`, `rmse_finite`,
`controls_finite`, `efe_finite`, `all_valid`; the script exits 1 when
`all_valid` is false.

## Execution notes

- Step 12 sets `CPOMDP_OUTPUT_DIR` and skips cpomdp scripts (never fails
  them) when the extra is not installed.
- The runner refuses `NUM_TIMESTEPS > 60` unless `CPOMDP_ALLOW_LONG=1`
  (receding-horizon re-planning scores every policy on every step).
- The Phase 1 exit criterion `main.py --only-steps "3,11,12,16" --frameworks
  cpomdp --target-dir input/gnn_files/continuous` runs all three exemplars
  (three `success` entries in `execution_summary.json`) and returns the same
  exit code as the `jax` backend on the same command — `2`, the pipeline's
  `SUCCESS_WITH_WARNINGS` from the Step 12 sandbox receipt, not a cpomdp
  failure. The same command on `input/gnn_files/discrete` reports every file
  `unsupported` and exits 0.
- The cross-framework reliability gate (`pipeline/cross_framework_reliability.py`)
  still profiles its original seven frameworks: its `MAINTAINED_FRAMEWORKS`
  tuple is bound to the manuscript token map, so adding cpomdp there is a
  manuscript regeneration, not a backend change.

## Dependency window

Recorded from `uv add --optional cpomdp "cpomdp>=0.4.4,<0.5"` followed by
`uv sync --extra cpomdp` on a checkout at GNN 3.3.0:

| Item | Value |
|---|---|
| cpomdp | 0.4.4 (latest 0.4.x on PyPI at scoping) |
| cpomdp `requires-python` | `>= 3.10` (GNN pins `>= 3.11, < 3.15`) |
| cpomdp runtime deps | `jax >= 0.4`, `jaxtyping >= 0.3.7`, `numpy >= 1.24` |
| Resolved `jax` after adding the extra | 0.9.2 (unchanged; GNN pins `jax[cpu] >= 0.7.0, < 0.11`) |
| Resolved `jaxtyping` | 0.3.9 |
| Lockfile churn | one new package block, no version moves elsewhere |

No conflict: the extra resolves inside GNN's existing JAX window, so `uv sync`
(core) and `uv sync --extra cpomdp` share one `jax`.

## Licence

cpomdp is distributed under the **MIT** licence (`License-Expression: MIT` in
its wheel metadata). The `dependency-review.yml` workflow denies only the AGPL
family (`AGPL-1.0-only`, `AGPL-1.0-or-later`, `AGPL-3.0-only`,
`AGPL-3.0-or-later`), so the extra passes the licence gate. cpomdp's own
dependency set (`jax`, `jaxtyping`, `numpy`) is Apache-2.0 / MIT / BSD-3.

GNN itself stays CC-BY-NC-SA-4.0; the integration never copies cpomdp code
into `src/`.

## Installation

```bash
uv sync --extra cpomdp
```

Check availability:

```bash
uv run python -c "import cpomdp; print(cpomdp.__version__)"
PYTHONPATH=src uv run python -c "from gnn.execute.cpomdp import is_cpomdp_available; print(is_cpomdp_available())"
```

## Run

```bash
# Render continuous models to cpomdp scripts
python src/gnn/11_render.py --target-dir input/gnn_files/continuous --frameworks cpomdp

# Execute and analyse
python src/gnn/main.py --only-steps "3,11,12,16" --frameworks cpomdp --target-dir input/gnn_files/continuous --output-dir output --verbose

# Parity against the jax backend (same seed)
uv run pytest tests/analysis/test_cpomdp_parity.py -m "" -q
```

## Comparison to the other continuous backends

| Feature | JAX / NumPyro / PyTorch | Stan | cpomdp |
|---|---|---|---|
| Filter | Joseph-form Kalman (shared template) | Kalman marginal likelihood (NUTS) | cpomdp `KalmanBackend` (exact) |
| Control | proportional `control_gain * (goal_mean - mu_t)` | none | enumerated EFE search (`efe`) or the same proportional law (`parity`) |
| EFE record | none | none | per step, per policy, with the epistemic/pragmatic split |
| Discrete models | rendered | rendered | `unsupported` |

## Source Code Connections

| Stage | Module | Key Function |
|---|---|---|
| Rendering | [cpomdp_renderer.py](../../../src/gnn/render/cpomdp/cpomdp_renderer.py) | `render_gnn_to_cpomdp()` |
| Template | [script_template.py](../../../src/gnn/render/cpomdp/script_template.py) | `generate_cpomdp_script()` |
| Execution | [cpomdp_runner.py](../../../src/gnn/execute/cpomdp/cpomdp_runner.py) | `execute_cpomdp_script()` |
| Analysis | [analyzer.py](../../../src/gnn/analysis/cpomdp/analyzer.py) | `generate_analysis_from_logs()` |

## See Also / Next Steps

- **[Cross-Framework Methodology](../integration/framework_integration_guide.md)**
- **[GNN Syntax Reference](../reference/gnn_syntax.md)** — the continuous parameter families
- **[GNN Implementations Index](README.md)**
- **[Back to GNN START_HERE](../../START_HERE.md)**
