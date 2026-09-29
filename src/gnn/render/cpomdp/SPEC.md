# cpomdp Render Backend Specification

## Overview

The cpomdp render backend generates **standalone continuous active inference scripts** from continuous (linear-Gaussian) GNN specifications. It is the continuous-state sibling of the PyMDP backend: discrete specs are reported `unsupported`.

## Public API

`render.cpomdp` must export:

- `render_gnn_to_cpomdp(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`
- `UNSUPPORTED_MESSAGE: str`

## Inputs

- `gnn_spec`: the continuous spec shape `render.pomdp_processor` emits for `model_kind == "continuous"` — `initialparameterization` with `F` (n×n), `H` (m×n), `Q` (n×n), `R` (m×m), `prior_mean` (n), `prior_cov` (n×n), optional `goal_mean` (n) + `control_gain` (scalar), optional `R_x_family` (one of `render.continuous_common.SENSOR_FAMILIES`) + `R_x_params` (that family's arity; anything else fails the render with a clear message); `model_parameters` with `num_timesteps`, `dt`, `random_seed`.
- `output_path`: file path for the generated script.
- `options`: `control_mode` ∈ {`efe`, `parity`} (default `efe`), `action_scale` > 0 (default 0.5), `horizon` ≥ 1 (default 1), `goal_precision` > 0 (default 1.0). Each may also arrive as `ModelParameters.cpomdp_<option>`; explicit options win. Invalid values fail the render with a clear message.

## Emitted program

Constructed in this order: `LinearGaussianModel(F, H, Q, R, prior; control = I_n if goal; observation = CallableSensor(H, sensor_noise, {R, R_x_params}) if R_x_family)`, `StateGoal(goal_mean)` under a fixed `R` or `ObservationGoal(H @ goal_mean, action_bounds=(-action_scale, action_scale))` under the callable sensor, `FiniteActionSet` compass rose (9 actions for 2-D control, `2p + 1` otherwise, scaled by `action_scale`), `EnumeratedEfeSearch(horizon)` + `RecedingHorizonSelector`, `Agent(model, objective, selector=...)` as the pairing check. The loop drives `KalmanBackend.infer_states` and scores every policy each step (`jax.jit` of `search.evaluate`). JAX is seeded from `RANDOM_SEED` with x64 enabled; the sampling order matches the shared continuous template so `parity` reproduces the `jax` backend on the same seed.

## Outputs

- exactly one `.py` file at `output_path`.
- when run, the script writes `simulation_results.json` under `CPOMDP_OUTPUT_DIR` with the continuous schema (`model_name`, `framework: "cpomdp"`, `model_kind: "continuous"`, `num_timesteps`, `num_states`, `num_observations`, `beliefs`, `posterior_cov`, `true_states_continuous`, `observations_continuous`, `controls`, `control_mode`, `rmse_vs_true`, `observations: []`, `actions: []`, `validation`, `execution_time_seconds`) plus `efe_history` (T × |A|^H, empty for passive models), `epistemic_term`, `pragmatic_term`, `selected_policy_index`, `epistemic_by_policy_t0`, `pragmatic_by_policy_t0` (every enumerated policy's split at the first step), `horizon`, `action_scale`, `n_policies`, `search_warrant`, `sensor_kind` (`"fixed"` | `"state_dependent"`), `R_x_family`, `R_x_params`, `jax_version`, `cpomdp_version`.
- `validation` keys: `means_finite`, `posterior_cov_psd`, `rmse_finite`, `controls_finite`, `efe_finite`, `all_valid`. The script exits 1 when `all_valid` is false.

## Dependencies

- render-time: `numpy`
- generated script run-time: `cpomdp` (`>= 0.4.4, < 0.5`, optional extra), `jax`, `numpy`

## Success criteria

- continuous spec: file written and `(True, message, [str(output_path)])` returned.
- discrete spec: `(False, UNSUPPORTED_MESSAGE, [])` and no file written.
