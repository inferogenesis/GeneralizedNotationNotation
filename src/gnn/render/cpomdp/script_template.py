"""Standalone cpomdp simulation script generator (continuous branch only).

``render.continuous_script`` hard-codes a Kalman filter plus GNN's
proportional controller. cpomdp brings its own filter (``KalmanBackend``)
and chooses actions by an exhaustive expected-free-energy (EFE) search over
a declared finite action set, so it gets its own template. The matrix
literals come from :func:`render.continuous_common.literal_block` so every
continuous backend splices the same numbers.

The emitted script keeps the GNN continuous contract exactly:

    x_1 ~ N(prior_mean, prior_cov)
    x_t = F x_{t-1} + u_{t-1} + N(0, Q)
    y_t = H x_t + N(0, R)

with the control input added directly to the state (cpomdp ``control`` is the
identity), and the prior placed on ``x_1`` (the first observation is folded
in without a predict step, as the other continuous scripts do). Two control
modes exist when the spec declares ``goal_mean``/``control_gain``:

- ``efe`` (default): the first action of the EFE-minimising policy.
- ``parity``: GNN's ``u_t = control_gain * (goal_mean - mu_t)`` on top of
  cpomdp's filter, so Step 16 can compare against the ``jax`` backend on the
  same seed. The EFE search still runs for diagnostics.

Specs without a goal render as passive trackers (no control matrix, no EFE).

When the spec declares ``R_x_family``/``R_x_params`` the script builds a
``CallableSensor`` from a small family library (JAX-jittable closures that
scale the nominal ``R`` by a positive, state-dependent factor) and steers
with an ``ObservationGoal``; under a fixed ``R`` it keeps ``StateGoal``. The
goal type follows the sensor type because cpomdp refuses the other pairing.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from gnn.render.continuous_common import ContinuousSpec, literal_block

OUTPUT_ENV = "CPOMDP_OUTPUT_DIR"
CONTROL_MODES = ("efe", "parity")
DEFAULT_ACTION_SCALE = 0.5
DEFAULT_HORIZON = 1
DEFAULT_GOAL_PRECISION = 1.0
ACTION_SET_VERSION = "gnn-compass-rose-v1"

#: Render options the cpomdp backend understands, with the ``ModelParameters``
#: key that can carry each one from the GNN file (``options`` wins).
OPTION_KEYS: Dict[str, str] = {
    "control_mode": "cpomdp_control_mode",
    "action_scale": "cpomdp_action_scale",
    "horizon": "cpomdp_horizon",
    "goal_precision": "cpomdp_goal_precision",
}


def resolve_render_options(
    model_parameters: Optional[Mapping[str, Any]],
    options: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Merge explicit ``options`` over ``ModelParameters`` hints and validate."""
    params = dict(model_parameters or {})
    opts = dict(options or {})
    resolved: Dict[str, Any] = {}
    for key, param_key in OPTION_KEYS.items():
        if key in opts and opts[key] is not None:
            resolved[key] = opts[key]
        elif param_key in params and params[param_key] is not None:
            resolved[key] = params[param_key]

    control_mode = str(resolved.get("control_mode", "efe")).strip().lower()
    if control_mode not in CONTROL_MODES:
        raise ValueError(
            f"cpomdp control_mode must be one of {CONTROL_MODES}, got {control_mode!r}"
        )
    action_scale = float(resolved.get("action_scale", DEFAULT_ACTION_SCALE))
    if not action_scale > 0.0:
        raise ValueError(f"cpomdp action_scale must be > 0, got {action_scale}")
    horizon = int(resolved.get("horizon", DEFAULT_HORIZON))
    if horizon < 1:
        raise ValueError(f"cpomdp horizon must be >= 1, got {horizon}")
    goal_precision = float(resolved.get("goal_precision", DEFAULT_GOAL_PRECISION))
    if not goal_precision > 0.0:
        raise ValueError(f"cpomdp goal_precision must be > 0, got {goal_precision}")
    return {
        "control_mode": control_mode,
        "action_scale": action_scale,
        "horizon": horizon,
        "goal_precision": goal_precision,
    }


_IMPORTS = """
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("ERROR: JAX not installed. Install with: uv sync")
    sys.exit(1)

try:
    import cpomdp
    from cpomdp import (
        Agent,
        Belief,
        CallableSensor,
        KalmanBackend,
        LinearGaussianModel,
        ObservationGoal,
        StateGoal,
    )
    from cpomdp.efe import policy_efe
    from cpomdp.enumeration import (
        EnumeratedEfeSearch,
        FiniteActionSet,
        RecedingHorizonSelector,
    )
    from cpomdp.selection import Preference
except ImportError:
    print("ERROR: cpomdp not installed. Install with: uv sync --extra cpomdp")
    sys.exit(1)

# cpomdp enables x64 on import; repeat it so the RNG stream matches the other
# continuous backends even if the import order changes.
jax.config.update("jax_enable_x64", True)

FRAMEWORK = "cpomdp"
FRAMEWORK_VERSION = {"jax_version": jax.__version__, "cpomdp_version": cpomdp.__version__}
_KEY = [jax.random.PRNGKey(RANDOM_SEED)]


def arr(x):
    return jnp.asarray(x, dtype=jnp.float64)


def mvn_sample(mean, cov):
    _KEY[0], sub = jax.random.split(_KEY[0])
    return jax.random.multivariate_normal(sub, mean, cov)


def to_list(x):
    return np.asarray(x, dtype=float).tolist()


def is_psd(cov):
    return bool(jnp.all(jnp.linalg.eigvalsh((cov + cov.T) / 2.0) > -1e-9))
"""

_MODEL = '''

def compass_rose(p, scale):
    """The declared, versioned finite action set the EFE search enumerates.

    p == 2: eight unit directions scaled to ``scale`` plus the null action.
    Otherwise: +/- ``scale`` along each axis plus the null action (2p + 1).
    """
    if p == 2:
        diag = scale / np.sqrt(2.0)
        actions = [
            [0.0, 0.0],
            [scale, 0.0],
            [diag, diag],
            [0.0, scale],
            [-diag, diag],
            [-scale, 0.0],
            [-diag, -diag],
            [0.0, -scale],
            [diag, -diag],
        ]
    else:
        actions = [[0.0] * p]
        for axis in range(p):
            for sign in (1.0, -1.0):
                action = [0.0] * p
                action[axis] = sign * scale
                actions.append(action)
    return FiniteActionSet(actions, version=ACTION_SET_VERSION)


def sensor_at(model, x):
    """Local ``(C, R)`` of the sensor at state ``x`` (fixed R is its own linearisation)."""
    if model.observation is None:
        return model.sensor_model, model.sensor_noise
    return model.observation.linearize(x)


def sample_observation(model, x):
    C, R = sensor_at(model, x)
    return C @ x + mvn_sample(jnp.zeros(C.shape[0]), R)


def first_update(model, y):
    """Fold y_1 into the prior without a predict step: the prior is over x_1."""
    mu, P = model.prior.mean, model.prior.cov
    C, R = sensor_at(model, mu)
    S = C @ P @ C.T + R
    K = jnp.linalg.solve(S.T, (P @ C.T).T).T  # P C^T S^{-1}
    mu_new = mu + K @ (y - C @ mu)
    I_KC = jnp.eye(P.shape[0]) - K @ C
    P_new = I_KC @ P @ I_KC.T + K @ R @ K.T  # Joseph form (stays PSD)
    return Belief(mu_new, P_new)


def build_model():
    F, H, Q, R = arr(F_RAW), arr(H_RAW), arr(Q_RAW), arr(R_RAW)
    n = F.shape[0]
    # GNN adds the control input straight to the state: control = I_n.
    control = jnp.eye(n) if GOAL_MEAN_RAW is not None else None
    return LinearGaussianModel(
        dynamics=F,
        sensor_model=H,
        dynamics_noise=Q,
        sensor_noise=R,
        prior=Belief(arr(PRIOR_MEAN_RAW), arr(PRIOR_COV_RAW)),
        control=control,
        observation=build_sensor(H),
    )


def build_objective(model, goal):
    """Goal type follows sensor type.

    A fixed R takes a ``StateGoal`` (the LQR regime); a state-dependent sensor
    takes an ``ObservationGoal`` reading the goal through C. cpomdp raises on
    the other pairing when the Agent is built.
    """
    if model.observation is None:
        return StateGoal(goal, precision=GOAL_PRECISION * jnp.eye(model.n_states))
    return ObservationGoal(
        model.sensor_model @ goal,
        action_bounds=(-ACTION_SCALE, ACTION_SCALE),
        precision=GOAL_PRECISION * jnp.eye(model.n_observations),
        horizon=HORIZON,
    )


def build_planner(model, goal):
    """Objective, exhaustive EFE search, and the observation-space preference it scores."""
    objective = build_objective(model, goal)
    action_set = compass_rose(model.n_controls, ACTION_SCALE)
    search = EnumeratedEfeSearch(model, action_set, horizon=HORIZON)
    selector = RecedingHorizonSelector(search)
    # cpomdp validates the goal/sensor pairing when the Agent is built (a
    # StateGoal on a state-dependent sensor raises). The loop below drives the
    # same backend directly because it also needs the per-policy G vector.
    Agent(model, objective, selector=selector)
    C = model.sensor_model
    preference = Preference(C @ goal, GOAL_PRECISION * jnp.eye(model.n_observations))
    return search, selector, preference
'''

_RUN = """

def run_simulation():
    start = time.time()
    model = build_model()
    backend = KalmanBackend(model)
    F, Q = model.dynamics, model.dynamics_noise
    n, m, T = model.n_states, model.n_observations, NUM_TIMESTEPS
    goal = arr(GOAL_MEAN_RAW) if GOAL_MEAN_RAW is not None else None
    zero_u = jnp.zeros(n)

    if goal is not None:
        search, selector, preference = build_planner(model, goal)
        evaluate = jax.jit(search.evaluate)
        best_policy_terms = jax.jit(
            lambda belief, policy: policy_efe(model, belief, policy, preference)[1]
        )
        all_policy_terms = jax.jit(
            lambda belief: jax.vmap(
                lambda pol: policy_efe(model, belief, pol, preference)[1]
            )(search.policies)
        )
        control_mode = CONTROL_MODE
        n_policies = search.n_policies
        warrant = str(search.certificate)
    else:
        search = selector = preference = evaluate = best_policy_terms = None
        all_policy_terms = None
        control_mode = "passive"
        n_policies = 0
        warrant = None

    true_states, observations, beliefs, covs, controls = [], [], [], [], []
    efe_history, epistemic, pragmatic, selected = [], [], [], []
    terms_t0 = {"epistemic": [], "pragmatic": []}

    def choose(belief):
        if goal is None:
            return zero_u
        result = evaluate(belief, preference)
        g = result.g
        best = int(jnp.argmin(jnp.where(jnp.isnan(g), jnp.inf, g)))
        terms = best_policy_terms(belief, result.best_policy)
        efe_history.append(to_list(g))
        selected.append(best)
        epistemic.append(float(terms["epistemic"]))
        pragmatic.append(float(terms["pragmatic"]))
        if control_mode == "parity":
            return CONTROL_GAIN * (goal - belief.mean)
        return result.best_policy[0]  # == selector.select(belief, preference)

    def record(x, y, belief, u):
        for buf, val in (
            (true_states, x),
            (observations, y),
            (beliefs, belief.mean),
            (covs, belief.cov),
            (controls, u),
        ):
            buf.append(to_list(val))

    x = mvn_sample(model.prior.mean, model.prior.cov)
    y = sample_observation(model, x)
    belief = first_update(model, y)
    if goal is not None:
        # Every enumerated policy's split at t=0: under a fixed R the
        # epistemic column is constant across policies (the linear-Gaussian
        # collapse); a state-dependent R(x) is what makes it vary.
        split = all_policy_terms(belief)
        terms_t0 = {k: to_list(split[k]) for k in ("epistemic", "pragmatic")}
    u = choose(belief)
    record(x, y, belief, u)

    for _t in range(1, T):
        x = F @ x + u + mvn_sample(zero_u, Q)
        y = sample_observation(model, x)
        belief = backend.infer_states(y, belief, u if goal is not None else None)
        u = choose(belief)
        record(x, y, belief, u)

    beliefs_np = np.asarray(beliefs)
    truth_np = np.asarray(true_states)
    rmse = float(np.sqrt(np.mean((beliefs_np - truth_np) ** 2)))
    psd = all(is_psd(arr(c)) for c in covs)
    efe_np = np.asarray(efe_history, dtype=float)
    validation = {
        "means_finite": bool(np.all(np.isfinite(beliefs_np))),
        "posterior_cov_psd": bool(psd),
        "rmse_finite": bool(np.isfinite(rmse)),
        "controls_finite": bool(np.all(np.isfinite(np.asarray(controls)))),
        "efe_finite": bool(np.all(np.isfinite(efe_np))),
    }
    validation["all_valid"] = all(validation.values())
    results = {
        "model_name": MODEL_NAME,
        "framework": FRAMEWORK,
        "model_kind": "continuous",
        "num_timesteps": T,
        "num_states": n,
        "num_observations": m,
        "beliefs": beliefs,
        "posterior_cov": covs,
        "true_states_continuous": true_states,
        "observations_continuous": observations,
        "controls": controls,
        "control_mode": control_mode,
        "rmse_vs_true": rmse,
        # Discrete-schema slots stay empty: nothing categorical is defined here.
        "observations": [],
        "actions": [],
        "validation": validation,
        "execution_time_seconds": round(time.time() - start, 4),
        # cpomdp additions: per-step, per-policy EFE and the chosen policy.
        "efe_history": efe_history,
        "epistemic_term": epistemic,
        "pragmatic_term": pragmatic,
        "selected_policy_index": selected,
        "horizon": HORIZON,
        "action_scale": ACTION_SCALE,
        "n_policies": n_policies,
        "search_warrant": warrant,
        "epistemic_by_policy_t0": terms_t0["epistemic"],
        "pragmatic_by_policy_t0": terms_t0["pragmatic"],
        "sensor_kind": SENSOR_KIND,
        "R_x_family": R_X_FAMILY,
        "R_x_params": R_X_PARAMS,
    }
    results.update(FRAMEWORK_VERSION)

    output_dir = Path(os.environ.get(OUTPUT_ENV, "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "simulation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(
        f"{FRAMEWORK} continuous active inference simulation complete: {T} steps, "
        f"control_mode={control_mode}, rmse_vs_true={rmse:.4f}"
    )
    print(f"Results saved to: {out}")
    print(f"Validation: {validation}")
    return results


if __name__ == "__main__":
    res = run_simulation()
    sys.exit(0 if res["validation"]["all_valid"] else 1)
"""


_FIXED_SENSOR = '''
SENSOR_KIND = "fixed"
R_X_FAMILY = None
R_X_PARAMS = None


def build_sensor(H):
    """Fixed R: the plain linear-Gaussian model, no state-dependent noise."""
    return None
'''

_STATE_DEPENDENT_SENSOR = '''
SENSOR_KIND = "state_dependent"
R_X_FAMILY = {family!r}
R_X_PARAMS = {params}


# --- sensor family library ---------------------------------------------------
# Each family scales the nominal R by a positive factor s(x) that depends on
# the planar position read from the first two state dimensions. They are plain
# jnp closures so cpomdp can trace them under jit/vmap; ``sensor_noise`` is the
# module-level ``noise_fn`` a CallableSensor requires, and every tunable lives
# in ``noise_params`` (a pytree leaf), never in a closure.


def _planar_d2(x, cx, cy):
    return (x[0] - cx) ** 2 + (x[1] - cy) ** 2


def _factor_constant(x, p):
    return 1.0


def _factor_quadratic_beacon(x, p):
    # p = [cx, cy, r_min, k]: sharp at the beacon, noise grows with distance.
    return p[2] + p[3] * _planar_d2(x, p[0], p[1])


def _factor_blind_spot(x, p):
    # p = [cx, cy, radius, r_dead]: noise saturates at r_dead * R inside the disc.
    return 1.0 + (p[3] - 1.0) * jnp.exp(-_planar_d2(x, p[0], p[1]) / p[2] ** 2)


def _factor_beacon_and_blind_spot(x, p):
    return _factor_quadratic_beacon(x, p[:4]) * _factor_blind_spot(x, p[4:])


SENSOR_FAMILY_LIBRARY = {{
    "constant": _factor_constant,
    "quadratic_beacon": _factor_quadratic_beacon,
    "blind_spot": _factor_blind_spot,
    "beacon_and_blind_spot": _factor_beacon_and_blind_spot,
}}


def sensor_noise(x, params):
    """R(x) = R_nominal * s(x) — the CallableSensor ``noise_fn``."""
    return params["R"] * SENSOR_FAMILY_LIBRARY[R_X_FAMILY](x, params["p"])


def build_sensor(H):
    """State-dependent R(x): a CallableSensor passed via ``observation=``."""
    params = {{"R": arr(R_RAW), "p": arr(R_X_PARAMS)}}
    return CallableSensor(H, sensor_noise, params)
'''


def _sensor_block(spec: ContinuousSpec) -> str:
    """Emit ``build_sensor`` plus the sensor constants for ``spec``."""
    if not spec.has_state_dependent_sensor:
        return _FIXED_SENSOR
    lits = literal_block(spec)
    return _STATE_DEPENDENT_SENSOR.format(
        family=spec.R_x_family, params=lits["R_x_params"]
    )


def generate_cpomdp_script(
    spec: ContinuousSpec, options: Optional[Mapping[str, Any]] = None
) -> str:
    """Return the full standalone cpomdp script for ``spec``."""
    opts = resolve_render_options(None, options)
    lits = literal_block(spec)
    if not spec.has_control:
        control_desc = (
            "passive tracker (no goal_mean/control_gain: no control matrix, no EFE)"
        )
    elif opts["control_mode"] == "parity":
        control_desc = (
            f"parity: u_t = {spec.control_gain} * (goal_mean - mu_t) on cpomdp's "
            "filter (EFE search runs for diagnostics only)"
        )
    else:
        control_desc = (
            f"efe: first action of the EFE-minimising policy over a "
            f"{opts['horizon']}-step horizon (receding-horizon re-plan each step)"
        )
    sensor_desc = (
        f"state-dependent R(x), family {spec.R_x_family!r} (CallableSensor, "
        "ObservationGoal)"
        if spec.has_state_dependent_sensor
        else "fixed R (StateGoal)"
    )
    header = f'''#!/usr/bin/env python3
"""
cpomdp continuous active inference simulation: {spec.model_name}

Generated by the GNN pipeline — cpomdp renderer, continuous branch.
Generative model (GNN continuous contract):
    x_1 ~ N(prior_mean, prior_cov)
    x_t = F x_(t-1) + u_(t-1) + N(0, Q)      ({spec.n}-dim latent state)
    y_t = H x_t + N(0, R(x_t))               ({spec.m}-dim observation)
Sensor: {sensor_desc}
Control: {control_desc}
Inference: cpomdp KalmanBackend (exact Kalman filter). Actions come from an
exhaustive expected-free-energy search over a declared finite action set.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_NAME = {spec.model_name!r}
RANDOM_SEED = {spec.random_seed}
NUM_TIMESTEPS = {spec.num_timesteps}
DT = {spec.dt}
OUTPUT_ENV = {OUTPUT_ENV!r}
CONTROL_MODE = {opts["control_mode"]!r}
ACTION_SCALE = {opts["action_scale"]!r}
HORIZON = {opts["horizon"]}
GOAL_PRECISION = {opts["goal_precision"]!r}
ACTION_SET_VERSION = {ACTION_SET_VERSION!r}

F_RAW = {lits["F"]}
H_RAW = {lits["H"]}
Q_RAW = {lits["Q"]}
R_RAW = {lits["R"]}
PRIOR_MEAN_RAW = {lits["prior_mean"]}
PRIOR_COV_RAW = {lits["prior_cov"]}
GOAL_MEAN_RAW = {lits["goal_mean"]}
CONTROL_GAIN = {lits["control_gain"]}
'''
    return "".join([header, _IMPORTS, _sensor_block(spec), _MODEL, _RUN])
