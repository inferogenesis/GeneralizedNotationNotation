"""Shared helpers for rendering continuous-state (linear-Gaussian) GNN models.

Every framework renderer consumes the same ``gnn_spec`` produced by
``render.pomdp_processor`` for ``model_kind == "continuous"``:

``initialparameterization`` holds ``F`` (n×n), ``H`` (m×n), ``Q`` (n×n),
``R`` (m×m), ``prior_mean`` (n), ``prior_cov`` (n×n) and optionally
``goal_mean`` (n) + ``control_gain`` (scalar). ``model_parameters`` holds
``num_timesteps``, ``dt`` and ``random_seed``.

Two further optional keys declare a **state-dependent observation noise**
``R(x)`` on top of the nominal ``R``: ``R_x_family`` (one of
:data:`SENSOR_FAMILIES`) and ``R_x_params`` (a vector whose length is the
family's arity). ``R`` stays required and is the nominal matrix, so model-kind
detection is unchanged. Only the cpomdp backend realises ``R(x)``; every other
continuous renderer keeps the nominal ``R`` and says so in its render message
(:func:`nominal_sensor_note`).

The generative model each generated script simulates and filters:

    x_1 ~ N(prior_mean, prior_cov)
    x_t = F x_{t-1} + u_{t-1} + N(0, Q)
    y_t = H x_t + N(0, R)
    u_t = control_gain * (goal_mean - mu_t)   (mu_t = filtered mean; else 0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_KEYS = ("F", "H", "Q", "R", "prior_mean", "prior_cov")

#: Closed vocabulary for ``R_x_family`` → arity of ``R_x_params``. No
#: expression parsing: each family is a fixed, documented closure.
#:
#: - ``constant``: ``R(x) = R`` (arity 0; exercises the callable path).
#: - ``quadratic_beacon`` ``[cx, cy, r_min, k]``: ``R(x) = R · (r_min + k·d²)``,
#:   ``d = ‖x[:2] − (cx, cy)‖`` — sharp at the beacon, noise grows with distance.
#: - ``blind_spot`` ``[cx, cy, radius, r_dead]``:
#:   ``R(x) = R · (1 + (r_dead − 1)·exp(−d²/radius²))`` — noise saturates at
#:   ``r_dead × R`` inside the disc and returns to ``R`` outside it.
#: - ``beacon_and_blind_spot``: the two parameter vectors concatenated
#:   (beacon first); the factors multiply.
SENSOR_FAMILIES: Dict[str, int] = {
    "constant": 0,
    "quadratic_beacon": 4,
    "blind_spot": 4,
    "beacon_and_blind_spot": 8,
}


def is_continuous_spec(gnn_spec: Dict[str, Any]) -> bool:
    """True when the spec is a continuous linear-Gaussian model."""
    if gnn_spec.get("model_kind") == "continuous":
        return True
    try:
        from gnn.render.pomdp_contract import ModelKind, detect_model_kind

        return detect_model_kind(gnn_spec) == ModelKind.CONTINUOUS
    except (ImportError, ValueError) as e:
        # detect_model_kind only raises ValueError (malformed
        # initialparameterization) and the import itself can fail; log rather
        # than silently misclassifying a continuous spec as discrete.
        logger.warning(
            "is_continuous_spec: model-kind detection failed for spec "
            "(model_kind=%r): %s; treating as non-continuous",
            gnn_spec.get("model_kind"),
            e,
        )
        return False


@dataclass
class ContinuousSpec:
    """Validated numeric view of a continuous GNN spec."""

    model_name: str
    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    prior_mean: np.ndarray
    prior_cov: np.ndarray
    goal_mean: Optional[np.ndarray]
    control_gain: Optional[float]
    num_timesteps: int
    dt: float
    random_seed: int
    R_x_family: Optional[str] = None
    R_x_params: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return int(self.F.shape[0])

    @property
    def m(self) -> int:
        return int(self.H.shape[0])

    @property
    def has_control(self) -> bool:
        return self.goal_mean is not None and self.control_gain is not None

    @property
    def has_state_dependent_sensor(self) -> bool:
        return self.R_x_family is not None


def _scalar(value: Any) -> float:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return float(value)


def extract_continuous_spec(gnn_spec: Dict[str, Any]) -> ContinuousSpec:
    """Parse and shape-check the continuous parameter block."""
    initial = gnn_spec.get("initialparameterization") or gnn_spec.get(
        "initial_parameterization"
    )
    if not isinstance(initial, dict):
        raise ValueError("continuous spec requires an initialparameterization mapping")
    missing = [key for key in REQUIRED_KEYS if key not in initial]
    if missing:
        raise ValueError(f"continuous spec is missing {missing}")

    F = np.asarray(initial["F"], dtype=float)
    H = np.asarray(initial["H"], dtype=float)
    Q = np.asarray(initial["Q"], dtype=float)
    R = np.asarray(initial["R"], dtype=float)
    prior_mean = np.asarray(initial["prior_mean"], dtype=float).reshape(-1)
    prior_cov = np.asarray(initial["prior_cov"], dtype=float)
    n = F.shape[0]
    if F.shape != (n, n):
        raise ValueError(f"F must be square, got {F.shape}")
    m = H.shape[0]
    if H.shape != (m, n):
        raise ValueError(f"H must be [m, n]={m, n}, got {H.shape}")
    if Q.shape != (n, n) or R.shape != (m, m) or prior_cov.shape != (n, n):
        raise ValueError(
            f"covariance shapes mismatch: Q{Q.shape} R{R.shape} prior_cov{prior_cov.shape}"
        )
    if prior_mean.shape != (n,):
        raise ValueError(f"prior_mean must have {n} entries, got {prior_mean.shape}")

    goal_mean: Optional[np.ndarray] = None
    control_gain: Optional[float] = None
    if "goal_mean" in initial and "control_gain" in initial:
        goal_mean = np.asarray(initial["goal_mean"], dtype=float).reshape(-1)
        if goal_mean.shape != (n,):
            raise ValueError(f"goal_mean must have {n} entries, got {goal_mean.shape}")
        control_gain = _scalar(initial["control_gain"])

    R_x_family, R_x_params = extract_sensor_family(initial, n)

    params = gnn_spec.get("model_parameters") or {}
    num_timesteps = int(params.get("num_timesteps", 20))
    dt = float(params.get("dt", 1.0))
    seed = int(params.get("random_seed", params.get("seed", 42)))
    name = str(gnn_spec.get("model_name") or gnn_spec.get("name") or "continuous_model")
    return ContinuousSpec(
        model_name=name,
        F=F,
        H=H,
        Q=Q,
        R=R,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        goal_mean=goal_mean,
        control_gain=control_gain,
        num_timesteps=num_timesteps,
        dt=dt,
        random_seed=seed,
        R_x_family=R_x_family,
        R_x_params=R_x_params,
    )


def extract_sensor_family(
    initial: Dict[str, Any], n: int
) -> tuple[Optional[str], Optional[np.ndarray]]:
    """Validate the optional ``R_x_family``/``R_x_params`` pair.

    Returns ``(None, None)`` when no family is declared. Unknown families, a
    missing or ragged parameter vector, and a state too small for the planar
    families are render-time errors with a clear message (no expression
    parsing, no silent fallback to the nominal ``R``).
    """
    raw_family = initial.get("R_x_family")
    raw_params = initial.get("R_x_params")
    if raw_family is None and raw_params is None:
        return None, None
    if raw_family is None:
        raise ValueError("R_x_params given without R_x_family")
    if isinstance(raw_family, (list, tuple)):
        while isinstance(raw_family, (list, tuple)) and len(raw_family) == 1:
            raw_family = raw_family[0]
    family = str(raw_family).strip()
    if family not in SENSOR_FAMILIES:
        raise ValueError(
            f"unknown R_x_family {family!r}; expected one of {sorted(SENSOR_FAMILIES)}"
        )
    arity = SENSOR_FAMILIES[family]
    params = np.asarray([] if raw_params is None else raw_params, dtype=float)
    params = params.reshape(-1)
    if params.shape[0] != arity:
        raise ValueError(
            f"R_x_family {family!r} takes {arity} parameter(s) in R_x_params, "
            f"got {params.shape[0]}"
        )
    if family != "constant" and n < 2:
        raise ValueError(
            f"R_x_family {family!r} reads a planar position from the first two "
            f"state dimensions; the state is {n}-dimensional"
        )
    if family in ("quadratic_beacon", "beacon_and_blind_spot"):
        r_min, k = float(params[2]), float(params[3])
        if r_min <= 0.0 or k < 0.0:
            raise ValueError(
                f"quadratic_beacon needs r_min > 0 and k >= 0, got r_min={r_min}, k={k}"
            )
    if family in ("blind_spot", "beacon_and_blind_spot"):
        offset = 4 if family == "beacon_and_blind_spot" else 0
        radius, r_dead = float(params[offset + 2]), float(params[offset + 3])
        if radius <= 0.0 or r_dead <= 0.0:
            raise ValueError(
                f"blind_spot needs radius > 0 and r_dead > 0, got radius={radius}, "
                f"r_dead={r_dead}"
            )
    return family, params


NOMINAL_SENSOR_NOTE = "state-dependent R declared, this backend uses nominal R"


def nominal_sensor_note(spec: ContinuousSpec) -> str:
    """Suffix for the render message of backends that ignore ``R_x_family``."""
    return f" ({NOMINAL_SENSOR_NOTE})" if spec.has_state_dependent_sensor else ""


def py_literal(arr: np.ndarray) -> str:
    """Render an array as a nested Python list literal with full precision."""
    return repr(np.asarray(arr, dtype=float).tolist())


def literal_block(spec: ContinuousSpec) -> Dict[str, str]:
    """Literals for every parameter, ready to splice into a template."""
    goal = py_literal(spec.goal_mean) if spec.goal_mean is not None else "None"
    gain = repr(float(spec.control_gain)) if spec.control_gain is not None else "None"
    return {
        "F": py_literal(spec.F),
        "H": py_literal(spec.H),
        "Q": py_literal(spec.Q),
        "R": py_literal(spec.R),
        "prior_mean": py_literal(spec.prior_mean),
        "prior_cov": py_literal(spec.prior_cov),
        "goal_mean": goal,
        "control_gain": gain,
        "R_x_family": repr(spec.R_x_family) if spec.R_x_family else "None",
        "R_x_params": (
            py_literal(spec.R_x_params) if spec.R_x_params is not None else "None"
        ),
    }


RESULT_KEYS: List[str] = [
    "model_name",
    "framework",
    "model_kind",
    "num_timesteps",
    "num_states",
    "num_observations",
    "beliefs",
    "posterior_cov",
    "true_states_continuous",
    "observations_continuous",
    "controls",
    "rmse_vs_true",
    "validation",
]
