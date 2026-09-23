"""Continuous (linear-Gaussian) branch of the JAX/NumPyro/PyTorch/Stan/cpomdp renderers.

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

import numpy as np
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
    res = _run(Path(arts[0]), "CPOMDP_OUTPUT_DIR", tmp_path / "out")
    _assert_schema(res, "cpomdp", with_control)
    assert res["control_mode"] == (control_mode or "passive")
    assert res["validation"]["efe_finite"] is True
    assert "cpomdp_version" in res
    if with_control:
        # 9 compass-rose actions at horizon 1: 9 policies, 9 step-evals a cycle.
        assert res["n_policies"] == 9
        assert res["cost_per_cycle"] == 9
        assert all(len(row) == 9 for row in res["efe_history"])
        assert len(res["efe_history"]) == T
        assert len(res["epistemic_term"]) == T
        assert len(res["pragmatic_term"]) == T
        assert len(res["selected_policy_index"]) == T
        assert res["search_warrant"].startswith("PROVED")
    else:
        assert res["efe_history"] == []
        assert res["n_policies"] == 0
        assert res["cost_per_cycle"] == 0


@pytest.mark.needs_cpomdp
@pytest.mark.parametrize("with_control", [True, False])
def test_cpomdp_parity_matches_jax_oracle(tmp_path: Path, with_control: bool) -> None:
    """The jax script's Kalman filter is the exact oracle for cpomdp's filter."""
    options = {"control_mode": "parity"}
    ok, msg, arts = render_gnn_to_cpomdp(
        _spec(with_control), tmp_path / "c_cpomdp.py", options
    )
    assert ok, msg
    cpomdp_res = _run(Path(arts[0]), "CPOMDP_OUTPUT_DIR", tmp_path / "cpomdp")
    ok, msg, arts = render_gnn_to_jax(_spec(with_control), tmp_path / "j_jax.py")
    assert ok, msg
    jax_res = _run(Path(arts[0]), "GNN_OUTPUT_DIR", tmp_path / "jax")
    for key in ("true_states_continuous", "beliefs", "posterior_cov", "controls"):
        np.testing.assert_allclose(cpomdp_res[key], jax_res[key], atol=1e-8)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("control_mode", "lqr"),
        ("action_scale", 0.0),
        ("horizon", 0),
        ("goal_precision", -1.0),
    ],
)
def test_cpomdp_rejects_bad_options(tmp_path: Path, option: str, value: Any) -> None:
    ok, msg, arts = render_gnn_to_cpomdp(
        _spec(True), tmp_path / "bad.py", {option: value}
    )
    assert not ok
    assert arts == []
    assert option in msg


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
    assert not ok
    assert msg == UNSUPPORTED_MESSAGE
    assert arts == []
    assert not (tmp_path / "d_cpomdp.py").exists()
