# cpomdp Analysis

Framework-specific analysis module for cpomdp simulation outputs (continuous active inference).

## Usage

```python
from gnn.analysis.cpomdp.analyzer import generate_analysis_from_logs

generate_analysis_from_logs(
    results_dir="output/12_execute_output", output_dir="output/16_analysis_output"
)
```

## Outputs

- Belief-mean trajectory plot: `belief_trajectory.png`
- Per-policy expected free energy plot: `efe_history.png`
- Epistemic / pragmatic split plot: `efe_terms.png`
- JSON summary: `cpomdp_analysis.json` (`rmse_vs_true`, `mean_efe`, `mean_epistemic`, `mean_pragmatic`, `control_mode`, `n_policies`)

## Dependencies

- `numpy`, `matplotlib` (required)
- `cpomdp` (runtime, for producing the results analyzed here)

## See Also

- [Parent: analysis/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Architecture documentation
- [SPEC.md](SPEC.md) — Technical specification
