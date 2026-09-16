# cpomdp Framework Implementation

> **GNN Integration Layer**: Python / JAX-based continuous active inference
> **Framework Base**: `cpomdp >= 0.4.4, < 0.5` (the continuous-state sibling of pymdp)
> **Simulation Architecture**: Exact Kalman filtering plus enumerated expected-free-energy (EFE) action search
> **Documentation Version**: 0.1.0 (dependency window recorded; backend wiring follows)

## Overview

[cpomdp](https://github.com/inferogenesis/cpomdp) provides a **continuous-state
active inference** backend for GNN's linear-Gaussian models. Where the other
continuous backends (JAX, NumPyro, PyTorch, Stan, RxInfer.jl) filter the
declared `F`/`H`/`Q`/`R` model and, at most, close the loop with GNN's
proportional controller, cpomdp evaluates expected free energy over a declared
finite action set and returns the policy that minimises it. Its epistemic term
only moves under a state-dependent observation noise `R(x)`, which is what a
later phase of this integration exposes at the GNN language level.

The renderer and runner are glue only: no cpomdp source is vendored into GNN.
The emitted script imports the PyPI package.

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
```

## See Also / Next Steps

- **[Cross-Framework Methodology](../integration/framework_integration_guide.md)**
- **[GNN Implementations Index](README.md)**
- **[Back to GNN START_HERE](../../START_HERE.md)**
