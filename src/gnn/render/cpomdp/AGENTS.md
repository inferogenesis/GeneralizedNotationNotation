# cpomdp Render Backend - Agent Scaffolding

## Module Overview

**Purpose**: Render continuous (linear-Gaussian) GNN specifications to **standalone cpomdp simulation scripts** — exact Kalman filtering plus an exhaustive expected-free-energy (EFE) search over a declared finite action set.

**Parent**: `src/gnn/render/` (Step 11: Render)

**Primary entrypoint**: `render_gnn_to_cpomdp` in `cpomdp_renderer.py` (re-exported by `__init__.py`).

---

## Public API

From `src/gnn/render/cpomdp/__init__.py`:

- `render_gnn_to_cpomdp(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`
- `UNSUPPORTED_MESSAGE` — the message returned for discrete specs.

**Contract**:
- continuous spec: writes exactly one Python file to `output_path` and returns `(True, message, [output_path])`
- discrete spec: returns `(False, UNSUPPORTED_MESSAGE, [])`; `render.pomdp_processor` records the framework under `frameworks_unsupported` (never `frameworks_failed`)

---

## Implementation notes

- `cpomdp_renderer.py` builds the `ContinuousSpec` through `render.continuous_common` (the same parser every continuous backend uses) and resolves the render options (`control_mode`, `action_scale`, `horizon`, `goal_precision`) from `options` first, then `ModelParameters` (`cpomdp_<option>` keys).
- `script_template.py` owns the emitted program. It does **not** route through `render.continuous_script.generate_continuous_script` (that template hard-codes a Kalman filter plus proportional control); it reuses `continuous_common.literal_block` for the matrix literals only.
- The emitted script constructs, in order: `LinearGaussianModel` (`control = I_n` when a goal is declared, since GNN adds `u` directly to the state), `StateGoal(goal_mean)` under a fixed `R`, a compass-rose `FiniteActionSet`, `EnumeratedEfeSearch` + `RecedingHorizonSelector`, and an `Agent` whose construction is cpomdp's goal/sensor pairing check. The simulation loop drives `KalmanBackend.infer_states` directly so it can record the full per-policy `G` vector each step. `EFESelector` is 1-D only and is not used.
- The first observation is folded into the prior without a predict step: the GNN continuous contract places the prior on `x_1`, as the JAX/NumPyro/PyTorch scripts do. In `parity` mode the emitted script reproduces the `jax` backend bit-for-bit on the same seed.
- The renderer never imports `cpomdp`; only the emitted script does. No cpomdp source is vendored into GNN.

---

## Integration points

- **Registry**: `render.framework_registry` entry `cpomdp` (`supports_continuous: True`, `supports_discrete: False`).
- **Called by**: `render.pomdp_processor.POMDPRenderProcessor` (route `RENDERER_ROUTES["cpomdp"]`) and `render.processor.render_gnn_spec` (continuous dispatch table).
- **Consumed by**: `src/gnn/execute/cpomdp/` runner (Step 12) and `src/gnn/analysis/cpomdp/` analyzer (Step 16).

---

## Testing

- `tests/render/test_continuous_renderers.py` — schema test (efe/parity/passive) and the discrete refusal
- `tests/render/test_framework_availability.py` — registry contract
- `tests/render/test_render_cli_targets.py` — dispatch guard
- `tests/analysis/test_cpomdp_parity.py` — `parity` mode versus the `jax` backend on the same seed

End-to-end execution tests skip when `cpomdp` is not importable.

---

## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
