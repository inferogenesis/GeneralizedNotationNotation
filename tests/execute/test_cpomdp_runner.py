"""Step 12 cpomdp runner: renders and executes every continuous exemplar.

Execution needs the optional ``cpomdp`` extra (``needs_cpomdp`` marker, see
tests/helpers/toolchain_probes.py); the cap and discovery tests run anywhere.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from gnn.execute.cpomdp import (
    ALLOW_LONG_ENV,
    MAX_TIMESTEPS_WITHOUT_OPT_IN,
    execute_cpomdp_script,
    find_cpomdp_scripts,
    refuse_long_run,
    run_cpomdp_scripts,
)
from gnn.render.processor import process_render

REPO = Path(__file__).resolve().parents[2]
CONTINUOUS_DIR = REPO / "input" / "gnn_files" / "continuous"
FILES = sorted(CONTINUOUS_DIR.glob("*.md"))


def _render_all(tmp_path: Path) -> Path:
    render_dir = tmp_path / "11_render_output"
    assert process_render(
        target_dir=CONTINUOUS_DIR,
        output_dir=render_dir,
        frameworks=["cpomdp"],
        verbose=False,
    )
    return render_dir


def test_refuse_long_run_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOW_LONG_ENV, raising=False)
    short = f"NUM_TIMESTEPS = {MAX_TIMESTEPS_WITHOUT_OPT_IN}\n"
    long = f"NUM_TIMESTEPS = {MAX_TIMESTEPS_WITHOUT_OPT_IN + 1}\n"
    assert refuse_long_run(short) is None
    assert refuse_long_run("# no literal\n") is None
    reason = refuse_long_run(long)
    assert reason is not None and ALLOW_LONG_ENV in reason
    monkeypatch.setenv(ALLOW_LONG_ENV, "1")
    assert refuse_long_run(long) is None


@pytest.mark.needs_cpomdp
def test_execute_refuses_long_script_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv(ALLOW_LONG_ENV, raising=False)
    script = tmp_path / "long_cpomdp.py"
    script.write_text(
        f"NUM_TIMESTEPS = {MAX_TIMESTEPS_WITHOUT_OPT_IN + 40}\n"
        "raise SystemExit('must not run')\n"
    )
    with caplog.at_level(logging.ERROR):
        assert execute_cpomdp_script(script, output_dir=tmp_path / "out") is False
    assert "exceeds" in caplog.text
    assert not (tmp_path / "out" / "execution_log.json").exists()


@pytest.mark.needs_cpomdp
def test_renders_and_executes_every_continuous_exemplar(tmp_path: Path) -> None:
    render_dir = _render_all(tmp_path)
    scripts = find_cpomdp_scripts(render_dir)
    assert {s.parent.parent.name for s in scripts} == {p.stem for p in FILES}
    exec_dir = tmp_path / "12_execute_output"
    assert run_cpomdp_scripts(render_dir, exec_dir, verbose=False) is True
    for script in scripts:
        result = json.loads(
            (exec_dir / script.stem / "simulation_results.json").read_text()
        )
        assert result["framework"] == "cpomdp"
        assert result["model_kind"] == "continuous"
        assert result["validation"]["all_valid"] is True
        steps = result["num_timesteps"]
        assert len(result["beliefs"]) == steps
        if result["control_mode"] == "passive":
            assert result["efe_history"] == []
            assert all(c == 0.0 for row in result["controls"] for c in row)
        else:
            assert result["control_mode"] == "efe"
            assert len(result["efe_history"]) == steps
            assert all(
                len(row) == result["n_policies"] for row in result["efe_history"]
            )
            assert len(result["selected_policy_index"]) == steps
    modes = {}
    for script in scripts:
        result = json.loads(
            (exec_dir / script.stem / "simulation_results.json").read_text()
        )
        modes[script.parent.parent.name] = result["control_mode"]
    assert modes["continuous_navigation"] == "efe"
    assert modes["predictive_coding_agent"] == "passive"
    assert modes["stochastic_dynamics"] == "passive"


@pytest.mark.needs_cpomdp
def test_runner_routes_results_to_requested_directory(tmp_path: Path) -> None:
    render_dir = _render_all(tmp_path)
    script = next(s for s in find_cpomdp_scripts(render_dir) if "navigation" in str(s))
    out = tmp_path / "routed"
    assert execute_cpomdp_script(script, output_dir=out) is True
    assert (out / "simulation_results.json").exists()
    assert (out / "execution_log.json").exists()
    assert not (script.parent / "simulation_results.json").exists()
