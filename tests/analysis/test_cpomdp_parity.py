"""cpomdp ``parity`` mode reproduces the jax continuous backend on the same seed.

Renders ``continuous_navigation.md`` (and the other continuous exemplars) on
``jax`` and on ``cpomdp`` with ``control_mode="parity"`` and compares the
filtered RMSE against the true trajectory. The generative model, the RNG
stream and GNN's proportional controller are identical; only the filter
implementation differs (cpomdp's KalmanBackend versus the shared Joseph-form
template), so the two agree to floating-point precision.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gnn import parse_gnn_file
from gnn.render.processor import render_gnn_spec

REPO = Path(__file__).resolve().parents[2]
CONTINUOUS_DIR = REPO / "input" / "gnn_files" / "continuous"
FILES = sorted(CONTINUOUS_DIR.glob("*.md"))

pytestmark = [pytest.mark.slow, pytest.mark.needs_cpomdp]


def _run(script: Path, env_var: str, out: Path) -> dict[str, Any]:
    env = dict(os.environ, **{env_var: str(out)})
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads((out / "simulation_results.json").read_text())


def _render_and_run(
    source: Path, target: str, tmp_path: Path, options: dict[str, Any] | None
) -> dict[str, Any]:
    spec = parse_gnn_file(source)
    out_dir = tmp_path / target
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, message, artifacts = render_gnn_spec(spec, target, out_dir, options)
    assert ok, message
    env_var = {"jax": "GNN_OUTPUT_DIR", "cpomdp": "CPOMDP_OUTPUT_DIR"}[target]
    return _run(Path(artifacts[0]), env_var, out_dir / "results")


@pytest.mark.parametrize("source", FILES, ids=[p.stem for p in FILES])
def test_parity_mode_matches_jax_rmse(source: Path, tmp_path: Path) -> None:
    jax_res = _render_and_run(source, "jax", tmp_path, None)
    cpomdp_res = _render_and_run(source, "cpomdp", tmp_path, {"control_mode": "parity"})
    assert jax_res["num_timesteps"] == cpomdp_res["num_timesteps"]
    assert abs(jax_res["rmse_vs_true"] - cpomdp_res["rmse_vs_true"]) < 0.05
    # Same seed, same generative model: the sampled trajectories coincide.
    assert np.allclose(
        jax_res["true_states_continuous"],
        cpomdp_res["true_states_continuous"],
        atol=1e-9,
    )
    assert np.allclose(jax_res["beliefs"], cpomdp_res["beliefs"], atol=1e-6)
    expected_mode = "parity" if source.stem == "continuous_navigation" else "passive"
    assert cpomdp_res["control_mode"] == expected_mode
    if expected_mode == "parity":
        assert np.allclose(jax_res["controls"], cpomdp_res["controls"], atol=1e-6)
        assert len(cpomdp_res["efe_history"]) == cpomdp_res["num_timesteps"]
