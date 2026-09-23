# cpomdp Analysis — Technical Specification

## Input Format

- `simulation_results.json` from the cpomdp execution step (searched recursively under `cpomdp/` paths or `cpomdp_simulation_results.json`)
- Fields: continuous schema (`beliefs` = posterior means, `posterior_cov`, `true_states_continuous`, `controls`, `rmse_vs_true`, `validation`, `model_name`, `model_kind: "continuous"`) plus `efe_history` (T × |A|^H), `epistemic_term`, `pragmatic_term`, `selected_policy_index`, `control_mode`, `n_policies`

## Output Format

- PNG plots: `belief_trajectory.png`, `efe_history.png` (per policy), `efe_terms.png` (epistemic vs pragmatic)
- JSON summary: `{model}/cpomdp_analysis.json` (framework, model_kind, metrics, plots_generated)

## Processing Requirements

- Continuous branch of `flat_payload_analyzer`: no categorical entropy/confidence on Gaussian means; `num_timesteps` read from the payload
- Passive results (empty `efe_history`) produce the trajectory plot only
- Graceful degradation when matplotlib unavailable (logs warnings, skips plots)

## Error Handling

- Missing results file → returns empty analysis with warning
- Malformed JSON → logs parse error, continues with available data
