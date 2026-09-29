"""Continuous (linear-Gaussian) branch of the JAX / NumPyro / PyTorch / Stan / cpomdp renderers.

Builds the continuous ``gnn_spec`` by hand (the shape ``render.pomdp_processor``
emits for ``model_kind == "continuous"``), renders each backend, and executes
the generated scripts. Optional-backend execution is gated by the registered
``needs_torch``/``needs_cmdstan`` markers (see tests/helpers/toolchain_probes.py),
not by in-file skips.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, cast

import pytest

from gnn.render.continuous_common import extract_continuous_spec, is_continuous_spec
from gnn.render.cpomdp.cpomdp_renderer import UNSUPPORTED_MESSAGE, render_gnn_to_cpomdp
from gnn.render.jax.jax_renderer import render_gnn_to_jax
from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro
from gnn.render.pytorch.pytorch_renderer import render_gnn_to_pytorch
from gnn.render.stan.stan_renderer import render_gnn_to_stan

T = 8


def _spec(with_control: bool) -> Dict[str, Any]:
    initial: Dict[str, Any] = {
        "F": [[1.0, 0.1], [0.0, 0.9]],
        "H": [[1.0, 0.0], [0.0, 1.0]],
        "Q": [[0.05, 0.0], [0.0, 0.05]],
        "R": [[0.1, 0.0], [0.0, 0.1]],
        "prior_mean": [0.0, 0.0],
        "prior_cov": [[0.5, 0.0], [0.0, 0.5]],
    }
    if with_control:
        initial["goal_mean"] = [1.0, 0.0]
        initial["control_gain"] = 0.3
    return {
        "name": "Test Continuous",
        "model_name": "Test Continuous",
        "gnn_section": "ActInfContinuous",
        "model_kind": "continuous",
        "initialparameterization": initial,
        "model_parameters": {"num_timesteps": T, "dt": 0.1, "random_seed": 7},
    }


def _run(
    script: Path, env_var: str, out: Path, python: str = sys.executable
) -> Dict[str, Any]:
    env = dict(os.environ, **{env_var: str(out)})
    proc = subprocess.run(
        [python, str(script)], capture_output=True, text=True, env=env, timeout=600
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return cast(
        Dict[str, Any], json.loads((out / "simulation_results.json").read_text())
    )


def _assert_schema(res: Dict[str, Any], framework: str, with_control: bool) -> None:
    assert res["framework"] == framework
    assert res["model_kind"] == "continuous"
    assert len(res["beliefs"]) == T and len(res["beliefs"][0]) == 2
    assert len(res["posterior_cov"]) == T and len(res["posterior_cov"][0]) == 2
    assert len(res["controls"]) == T
    assert res["actions"] == [] and res["observations"] == []
    assert res["validation"]["all_valid"] is True
    if with_control:
        assert any(abs(c) > 0 for row in res["controls"] for c in row)
    else:
        assert all(c == 0.0 for row in res["controls"] for c in row)


def _sensor_spec(family: str | None, params: list[float] | None) -> Dict[str, Any]:
    spec = _spec(True)
    initial = spec["initialparameterization"]
    if family is not None:
        initial["R_x_family"] = family
    if params is not None:
        initial["R_x_params"] = params
    return spec


def test_sensor_family_extraction_and_validation() -> None:
    from gnn.render.continuous_common import SENSOR_FAMILIES, extract_sensor_family

    assert SENSOR_FAMILIES == {
        "constant": 0,
        "quadratic_beacon": 4,
        "blind_spot": 4,
        "beacon_and_blind_spot": 8,
    }
    plain = extract_continuous_spec(_spec(True))
    assert plain.R_x_family is None and plain.R_x_params is None
    assert not plain.has_state_dependent_sensor

    beacon = extract_continuous_spec(
        _sensor_spec("quadratic_beacon", [1.0, 0.0, 0.05, 1.0])
    )
    assert beacon.R_x_family == "quadratic_beacon"
    assert beacon.has_state_dependent_sensor
    assert beacon.R_x_params is not None and beacon.R_x_params.tolist() == [
        1.0,
        0.0,
        0.05,
        1.0,
    ]
    both = extract_continuous_spec(
        _sensor_spec("beacon_and_blind_spot", [1, 0, 0.05, 1, -1, 0, 0.5, 20])
    )
    assert both.R_x_params is not None and both.R_x_params.shape == (8,)
    constant = extract_continuous_spec(_sensor_spec("constant", []))
    assert constant.R_x_family == "constant"
    # The extractor writes ``{quadratic_beacon}`` as a one-element list.
    assert (
        extract_sensor_family(
            {"R_x_family": ["blind_spot"], "R_x_params": [0.0, 0.0, 0.5, 20.0]}, 2
        )[0]
        == "blind_spot"
    )

    with pytest.raises(ValueError, match="unknown R_x_family"):
        extract_continuous_spec(_sensor_spec("magic", [1.0]))
    with pytest.raises(ValueError, match="takes 4 parameter"):
        extract_continuous_spec(_sensor_spec("quadratic_beacon", [1.0, 0.0]))
    with pytest.raises(ValueError, match="without R_x_family"):
        extract_continuous_spec(_sensor_spec(None, [1.0, 0.0, 0.05, 1.0]))
    with pytest.raises(ValueError, match="r_min > 0"):
        extract_continuous_spec(_sensor_spec("quadratic_beacon", [1.0, 0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="radius > 0"):
        extract_continuous_spec(_sensor_spec("blind_spot", [1.0, 0.0, 0.0, 20.0]))
    with pytest.raises(ValueError, match="planar position"):
        extract_sensor_family(
            {"R_x_family": "quadratic_beacon", "R_x_params": [1, 0, 0.05, 1]}, 1
        )


def test_other_continuous_backends_keep_nominal_r_and_say_so(tmp_path: Path) -> None:
    """A declared R(x) is a render-message note on every non-cpomdp backend."""
    from gnn.render.continuous_common import NOMINAL_SENSOR_NOTE

    spec = _sensor_spec("quadratic_beacon", [1.0, 0.0, 0.05, 1.0])
    for fn, name in (
        (render_gnn_to_jax, "s_jax.py"),
        (render_gnn_to_numpyro, "s_numpyro.py"),
        (render_gnn_to_pytorch, "s_pytorch.py"),
        (render_gnn_to_stan, "s_stan.py"),
    ):
        ok, msg, _ = fn(spec, tmp_path / name)
        assert ok, f"{name}: {msg}"
        assert NOMINAL_SENSOR_NOTE in msg, (name, msg)
    ok, msg, _ = render_gnn_to_jax(_spec(True), tmp_path / "plain_jax.py")
    assert ok and NOMINAL_SENSOR_NOTE not in msg


def test_detection_and_extraction() -> None:
    spec = extract_continuous_spec(_spec(True))
    assert is_continuous_spec(_spec(False))
    assert spec.n == 2 and spec.m == 2 and spec.has_control
    assert not extract_continuous_spec(_spec(False)).has_control


@pytest.mark.parametrize("with_control", [True, False])
def test_jax_continuous_renders_and_runs(tmp_path: Path, with_control: bool) -> None:
    ok, msg, arts = render_gnn_to_jax(_spec(with_control), tmp_path / "m_jax.py")
    assert ok, msg
    res = _run(Path(arts[0]), "GNN_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "jax", with_control)
    assert "jax_version" in res


def test_numpyro_continuous_renders_and_runs_nuts(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_numpyro(_spec(True), tmp_path / "m_numpyro.py")
    assert ok, msg
    res = _run(Path(arts[0]), "NUMPYRO_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "numpyro", True)
    assert len(res["mcmc_posterior_means"]) == T
    assert res["mcmc_r_hat_max"] < 1.2
    assert res["validation"]["mcmc_finite"] is True


@pytest.mark.needs_cpomdp
@pytest.mark.parametrize(
    ("with_control", "control_mode"),
    [(True, "efe"), (True, "parity"), (False, None)],
)
def test_cpomdp_continuous_renders_and_runs(
    tmp_path: Path, with_control: bool, control_mode: str | None
) -> None:
    options = {"control_mode": control_mode} if control_mode else None
    ok, msg, arts = render_gnn_to_cpomdp(
        _spec(with_control), tmp_path / "m_cpomdp.py", options
    )
    assert ok, msg
    code = Path(arts[0]).read_text()
    assert "LinearGaussianModel(" in code and "EnumeratedEfeSearch(" in code
    res = _run(Path(arts[0]), "CPOMDP_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "cpomdp", with_control)
    assert res["control_mode"] == (control_mode or "passive")
    assert res["validation"]["efe_finite"] is True
    assert "cpomdp_version" in res and "jax_version" in res
    if with_control:
        assert len(res["efe_history"]) == T
        assert all(len(row) == res["n_policies"] == 9 for row in res["efe_history"])
        assert len(res["epistemic_term"]) == len(res["pragmatic_term"]) == T
        assert len(res["selected_policy_index"]) == T
        assert res["search_warrant"].startswith("PROVED")
    else:
        assert res["efe_history"] == [] and res["n_policies"] == 0


@pytest.mark.needs_cpomdp
@pytest.mark.parametrize(
    ("family", "params"),
    [
        ("constant", []),
        ("quadratic_beacon", [1.0, 0.0, 0.05, 1.0]),
        ("blind_spot", [1.0, 0.0, 0.5, 20.0]),
        ("beacon_and_blind_spot", [1.0, 0.0, 0.05, 1.0, -1.0, 0.0, 0.5, 20.0]),
    ],
)
def test_cpomdp_state_dependent_sensor_renders_and_runs(
    tmp_path: Path, family: str, params: list[float]
) -> None:
    """R(x) families emit a CallableSensor + ObservationGoal and run end to end."""
    ok, msg, arts = render_gnn_to_cpomdp(
        _sensor_spec(family, params), tmp_path / f"{family}_cpomdp.py"
    )
    assert ok, msg
    assert f"R(x) family {family}" in msg
    code = Path(arts[0]).read_text()
    assert "CallableSensor(" in code and "ObservationGoal(" in code
    assert f"R_X_FAMILY = {family!r}" in code
    assert (
        "StateGoal(goal"
        not in code.split("def build_objective")[1].split(
            "if model.observation is None"
        )[0]
    )
    res = _run(Path(arts[0]), "CPOMDP_OUTPUT_DIR", tmp_path / f"out_{family}")
    _assert_schema(res, "cpomdp", True)
    assert res["sensor_kind"] == "state_dependent"
    assert res["R_x_family"] == family and res["R_x_params"] == params
    assert res["validation"]["efe_finite"] is True
    assert len(res["efe_history"]) == T
    t0 = res["epistemic_by_policy_t0"]
    assert len(t0) == res["n_policies"] == 9
    assert all(v > 0 for v in t0)
    spread = max(t0) - min(t0)
    if family == "constant":
        # R(x) = R: the linear-Gaussian collapse — every policy gains the same.
        assert spread < 1e-9
    else:
        assert spread > 1e-6


def test_cpomdp_fixed_sensor_keeps_state_goal(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_cpomdp(_spec(True), tmp_path / "fixed_cpomdp.py")
    assert ok and "fixed R" in msg
    code = Path(arts[0]).read_text()
    assert 'SENSOR_KIND = "fixed"' in code and "R_X_FAMILY = None" in code
    assert "SENSOR_FAMILY_LIBRARY" not in code
    assert "StateGoal(goal" in code


@pytest.mark.needs_cpomdp
def test_cpomdp_refuses_state_goal_on_state_dependent_sensor() -> None:
    """The pairing the renderer never emits is refused by cpomdp itself."""
    import jax.numpy as jnp
    from cpomdp import (
        Agent,
        Belief,
        CallableSensor,
        LinearGaussianModel,
        ObservationGoal,
        StateGoal,
    )

    def noise(x, params):
        return params * (1.0 + x[0] ** 2)

    sensor = CallableSensor(jnp.eye(2), noise, jnp.eye(2) * 0.1)
    model = LinearGaussianModel(
        dynamics=jnp.eye(2),
        sensor_model=jnp.eye(2),
        dynamics_noise=jnp.eye(2) * 0.05,
        sensor_noise=jnp.eye(2) * 0.1,
        prior=Belief([0.0, 0.0], jnp.eye(2)),
        control=jnp.array([[1.0], [0.0]]),
        observation=sensor,
    )
    with pytest.raises(ValueError, match="StateGoal needs a fixed sensor"):
        Agent(model, StateGoal([1.0, 0.0]))
    Agent(model, ObservationGoal([1.0, 0.0], action_bounds=(-0.5, 0.5)))


def test_cpomdp_rejects_bad_options(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_cpomdp(
        _spec(True), tmp_path / "bad.py", {"control_mode": "lqr"}
    )
    assert not ok and not arts and "control_mode" in msg


@pytest.mark.needs_torch
def test_pytorch_continuous_renders(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_pytorch(_spec(True), tmp_path / "m_pytorch.py")
    assert ok, msg
    code = Path(arts[0]).read_text()
    assert "torch.distributions.MultivariateNormal" in code
    assert "GOAL_MEAN_RAW = [1.0, 0.0]" in code
    res = _run(Path(arts[0]), "PYTORCH_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "pytorch", True)


@pytest.mark.needs_cmdstan
def test_stan_continuous_program_and_driver(tmp_path: Path) -> None:
    ok, msg, arts = render_gnn_to_stan(_spec(True), tmp_path / "m_stan.py")
    assert ok, msg
    driver, program = Path(arts[0]), Path(arts[1])
    assert program.suffix == ".stan" and driver.suffix == ".py"
    text = program.read_text()
    assert "multi_normal_lpdf" in text and "obs_noise_scale" in text
    res = _run(driver, "STAN_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "stan", True)
    assert res["validation"]["rhat_ok"] is True


def test_discrete_regression_still_renders(tmp_path: Path) -> None:
    spec = {
        "name": "Disc",
        "model_name": "Disc",
        "initialparameterization": {
            "A": [[0.9, 0.1], [0.1, 0.9]],
            "B": [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            "C": [0.0, 1.0],
            "D": [0.5, 0.5],
        },
        "model_parameters": {
            "num_hidden_states": 2,
            "num_obs": 2,
            "num_actions": 2,
            "num_timesteps": 3,
        },
    }
    assert not is_continuous_spec(spec)
    for fn, name in (
        (render_gnn_to_jax, "d_jax.py"),
        (render_gnn_to_numpyro, "d_numpyro.py"),
        (render_gnn_to_pytorch, "d_pytorch.py"),
        (render_gnn_to_stan, "d_stan.py"),
    ):
        ok, msg, _ = fn(spec, tmp_path / name)
        assert ok, f"{name}: {msg}"
    # cpomdp is continuous-only: a discrete spec is refused, not rendered.
    ok, msg, arts = render_gnn_to_cpomdp(spec, tmp_path / "d_cpomdp.py")
    assert not ok and msg == UNSUPPORTED_MESSAGE and arts == []
    assert not (tmp_path / "d_cpomdp.py").exists()
