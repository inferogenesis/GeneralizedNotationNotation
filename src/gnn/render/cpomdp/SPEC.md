# cpomdp Render Backend Specification

## Public API

`gnn.render.cpomdp` exports `render_gnn_to_cpomdp(gnn_spec, output_path, options=None) -> Tuple[bool, str, List[str]]` and `UNSUPPORTED_MESSAGE`.

## Inputs

- `gnn_spec`: the continuous spec (`model_kind == "continuous"`): `F`, `H`, `Q`, `R`, `prior_mean`, `prior_cov`, optionally `goal_mean` + `control_gain`; `model_parameters` with `num_timesteps` and `random_seed`.
- `options`: see [README.md](README.md). Invalid values fail the render with a message naming the option; unknown keys are ignored.

## Outputs

- Continuous spec: one `.py` file at `output_path`; returns `(True, message, [str(output_path)])`.
- Discrete spec: returns `(False, UNSUPPORTED_MESSAGE, [])` and writes nothing.

When run, the script writes `simulation_results.json` under `CPOMDP_OUTPUT_DIR` (default `.`):

- the shared continuous schema: `model_name`, `framework` (`"cpomdp"`), `model_kind` (`"continuous"`), `num_timesteps`, `num_states`, `num_observations`, `beliefs`, `posterior_cov`, `true_states_continuous`, `observations_continuous`, `controls`, `control_mode`, `rmse_vs_true`, `observations` (`[]`), `actions` (`[]`), `validation`, `execution_time_seconds`;
- cpomdp additions: `efe_history` (one row of `|A|^H` values per step, empty when passive), `epistemic_term`, `pragmatic_term`, `selected_policy_index`, `horizon`, `action_scale`, `n_policies`, `search_warrant`, `jax_version`, `cpomdp_version`.

`validation` holds `means_finite`, `posterior_cov_psd`, `rmse_finite`, `controls_finite`, `efe_finite` and `all_valid`; the script exits 1 when `all_valid` is false.
