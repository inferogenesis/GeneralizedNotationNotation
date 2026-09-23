# cpomdp Render Backend - Agent Scaffolding

## Module Overview

**Purpose**: Render continuous (linear-Gaussian) GNN specifications to standalone cpomdp simulation scripts.

**Parent**: `src/gnn/render/` (Step 11: Render)

**Primary entrypoint**: `render_gnn_to_cpomdp` in `cpomdp_renderer.py`, re-exported by `__init__.py` with `UNSUPPORTED_MESSAGE`.

## Files

- `cpomdp_renderer.py`: refuses non-continuous specs, parses the spec with `render.continuous_common.extract_continuous_spec`, validates the options, writes the script atomically.
- `script_template.py`: the emitted program. It reuses `continuous_common.literal_block` for the matrix literals but not `continuous_script`, whose template hard-codes GNN's proportional controller.

## Design notes

- The emitted script builds a `LinearGaussianModel` with `control_matrix = I_n` when a goal is declared, because the GNN contract adds `u_t` straight to the state.
- The prior is over `x_1`, so the first observation is folded in without a predict step, as in the other continuous scripts. With the same seed and draw order, `parity` mode reproduces the `jax` backend.
- Actions come from `cpomdp.enumeration.EnumeratedEfeSearch` over a versioned compass-rose `FiniteActionSet`. The loop calls `search.evaluate` directly so it can record every policy's `G`. `EFESelector` is not used because it only searches 1-D actions.
- No cpomdp source is vendored; only the generated script imports cpomdp.

## Documentation

- **[README](README.md)**: usage and options
- **[SPEC](SPEC.md)**: inputs and output schema
