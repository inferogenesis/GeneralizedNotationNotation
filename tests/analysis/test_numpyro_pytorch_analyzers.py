"""Tests for the numpyro, pytorch and cpomdp framework analyzers (Step 16).

Covers ``analysis.numpyro.generate_analysis_from_logs`` and
``analysis.pytorch.generate_analysis_from_logs`` end-to-end on realistic
``simulation_results.json`` payloads plus their documented graceful-degradation
paths (no matplotlib backend, malformed JSON, empty/missing input).

The two modules are near-identical; the parametrized tests exercise each
framework independently so behavioural guarantees hold for both.
"""

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

FRAMEWORKS = ["numpyro", "pytorch", "cpomdp"]


def _read_analyzer(framework: str) -> Any:
    return importlib.import_module(f"gnn.analysis.{framework}.analyzer")


def _realistic_payload(model_name: str = "model_a") -> dict[str, Any]:
    """A payload shaped like what the numpyro/pytorch runners write:
    beliefs as a per-timestep state distribution, actions as a flat list,
    and efe_history as a per-timestep per-action vector (2D).
    """
    return {
        "model_name": model_name,
        "num_timesteps": 3,
        "beliefs": [
            [0.8, 0.2, 0.0],
            [0.9, 0.1, 0.0],
            [1.0, 0.0, 0.0],
        ],
        "actions": [1, 2, 1],
        "efe_history": [
            [1.0, 0.5],
            [0.8, 0.4],
            [0.6, 0.3],
        ],
        "validation": {"all_valid": True},
    }


def _write_results(root: Path, framework: str, payload: dict[str, Any]) -> Path:
    sim_dir = root / "model_a" / framework / "simulation_data"
    sim_dir.mkdir(parents=True)
    sim_file = sim_dir / "simulation_results.json"
    sim_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sim_file


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_module_importable(framework: str) -> None:
    mod = _read_analyzer(framework)
    assert callable(mod.generate_analysis_from_logs)
    assert callable(mod._generate_plots)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_generate_analysis_end_to_end(framework: str, tmp_path: Path) -> None:
    """Realistic inputs yield the documented analysis JSON structure."""
    mod = _read_analyzer(framework)
    exec_dir = tmp_path / "12_execute_output"
    _write_results(exec_dir, framework, _realistic_payload("model_a"))
    out_dir = tmp_path / "16_analysis_output" / framework
    out_dir.mkdir(parents=True)

    generated = mod.generate_analysis_from_logs(exec_dir, out_dir)
    assert len(generated) == 1, generated
    analysis_file = Path(generated[0])
    assert analysis_file.exists()
    assert analysis_file.suffix == ".json"

    analysis = json.loads(analysis_file.read_text(encoding="utf-8"))
    assert analysis["framework"] == framework
    assert analysis["model_name"] == "model_a"
    assert analysis["num_timesteps"] == 3
    assert analysis["num_states"] == 3
    assert analysis["validation"] == {"all_valid": True}

    metrics = analysis["metrics"]
    # Pinned values for the deterministic payload above.
    assert metrics["mean_confidence"] == pytest.approx(0.9)
    assert metrics["final_confidence"] == pytest.approx(1.0)
    assert metrics["action_distribution"] == {"1": 2, "2": 1}
    assert metrics["mean_efe"] == pytest.approx(0.6)
    assert "mean_belief_entropy" in metrics
    assert "final_belief_entropy" in metrics

    # When matplotlib is present in the environment, plots must be marked
    # generated and the plot artifacts must actually exist (regression for the
    # plots_generated flag that previously lied when the backend was missing).
    try:
        import matplotlib  # noqa: F401

        have_matplotlib = True
    except ImportError:
        have_matplotlib = False
    if have_matplotlib:
        model_out = out_dir / "model_a"
        assert analysis["plots_generated"] is True
        assert (model_out / "belief_trajectory.png").exists()
        assert (model_out / "action_distribution.png").exists()
        assert (model_out / "efe_history.png").exists()


def _continuous_payload(
    model_name: str = "model_c", passive: bool = False
) -> dict[str, Any]:
    """A payload shaped like the cpomdp runner writes for a continuous model."""
    payload: dict[str, Any] = {
        "model_name": model_name,
        "framework": "cpomdp",
        "model_kind": "continuous",
        "num_timesteps": 3,
        "num_states": 2,
        "num_observations": 2,
        "beliefs": [[0.1, -0.2], [0.4, -0.1], [0.9, 0.0]],
        "posterior_cov": [[[0.5, 0.0], [0.0, 0.5]]] * 3,
        "true_states_continuous": [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        "observations_continuous": [[0.1, 0.0], [0.5, 0.1], [1.0, -0.1]],
        "controls": [[0.0, 0.0]] * 3,
        "control_mode": "passive" if passive else "efe",
        "rmse_vs_true": 0.123,
        "observations": [],
        "actions": [],
        "validation": {"all_valid": True},
        "efe_history": []
        if passive
        else [[1.0, 0.5, 0.7], [0.8, 0.4, 0.6], [0.6, 0.3, 0.5]],
        "epistemic_term": [] if passive else [0.9, 0.8, 0.7],
        "pragmatic_term": [] if passive else [1.4, 1.2, 1.0],
        "selected_policy_index": [] if passive else [1, 1, 2],
        "n_policies": 0 if passive else 3,
    }
    return payload


@pytest.mark.parametrize("passive", [False, True])
def test_cpomdp_continuous_payload_takes_continuous_branch(
    tmp_path: Path, passive: bool
) -> None:
    """Continuous payloads report RMSE and the EFE split, never categorical entropy."""
    mod = _read_analyzer("cpomdp")
    exec_dir = tmp_path / "12_execute_output"
    _write_results(exec_dir, "cpomdp", _continuous_payload("model_c", passive))
    out_dir = tmp_path / "16_analysis_output" / "cpomdp"
    generated = mod.generate_analysis_from_logs(exec_dir, out_dir)
    assert len(generated) == 1
    analysis = json.loads(Path(generated[0]).read_text(encoding="utf-8"))
    assert analysis["framework"] == "cpomdp"
    assert analysis["model_kind"] == "continuous"
    assert analysis["num_timesteps"] == 3 and analysis["num_states"] == 2
    metrics = analysis["metrics"]
    assert metrics["rmse_vs_true"] == pytest.approx(0.123)
    assert "mean_belief_entropy" not in metrics and "mean_confidence" not in metrics
    assert metrics["final_posterior_trace"] == pytest.approx(1.0)
    if passive:
        assert metrics["control_mode"] == "passive"
        assert "mean_efe" not in metrics and "mean_epistemic" not in metrics
    else:
        assert metrics["control_mode"] == "efe" and metrics["n_policies"] == 3
        assert metrics["mean_efe"] == pytest.approx(0.6)
        assert metrics["mean_epistemic"] == pytest.approx(0.8)
        assert metrics["mean_pragmatic"] == pytest.approx(1.2)
        assert metrics["policy_distribution"] == {"1": 2, "2": 1}
        assert metrics["min_efe_per_step"] == pytest.approx([0.5, 0.4, 0.3])
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    model_out = out_dir / "model_a"
    assert analysis["plots_generated"] is True
    assert (model_out / "belief_trajectory.png").exists()
    assert (model_out / "efe_history.png").exists() is (not passive)
    assert (model_out / "efe_terms.png").exists() is (not passive)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_generate_analysis_missing_dir(framework: str, tmp_path: Path) -> None:
    mod = _read_analyzer(framework)
    result = mod.generate_analysis_from_logs(tmp_path / "nonexistent", tmp_path / "out")
    assert isinstance(result, list)
    assert result == []


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_generate_analysis_empty_dir(framework: str, tmp_path: Path) -> None:
    mod = _read_analyzer(framework)
    result = mod.generate_analysis_from_logs(tmp_path, tmp_path / "out")
    assert isinstance(result, list)
    assert result == []


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_generate_analysis_malformed_json_skipped(
    framework: str, tmp_path: Path
) -> None:
    mod = _read_analyzer(framework)
    sim_dir = tmp_path / "model_a" / framework / "simulation_data"
    sim_dir.mkdir(parents=True)
    (sim_dir / "simulation_results.json").write_text("{ not json", encoding="utf-8")
    result = mod.generate_analysis_from_logs(tmp_path, tmp_path / "out")
    assert result == []


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_analyzer_ignores_other_framework_results(
    framework: str, tmp_path: Path
) -> None:
    """Each analyzer only consumes its own framework's simulation results."""
    mod = _read_analyzer(framework)
    other = "pytorch" if framework == "numpyro" else "numpyro"
    _write_results(tmp_path, other, _realistic_payload("model_b"))
    result = mod.generate_analysis_from_logs(tmp_path, tmp_path / "out")
    assert result == []


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_generate_analysis_root_level_recovery(framework: str, tmp_path: Path) -> None:
    """A root-level simulation_results.json is picked up as a fallback."""
    mod = _read_analyzer(framework)
    (tmp_path / "simulation_results.json").write_text(
        json.dumps(_realistic_payload("rootmodel")), encoding="utf-8"
    )
    result = mod.generate_analysis_from_logs(tmp_path, tmp_path / "out")
    assert len(result) == 1
    analysis = json.loads(Path(result[0]).read_text(encoding="utf-8"))
    assert analysis["model_name"] == "rootmodel"


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_generate_analysis_default_output_dir(framework: str, tmp_path: Path) -> None:
    """output_dir defaults to results_dir when omitted."""
    mod = _read_analyzer(framework)
    _write_results(tmp_path, framework, _realistic_payload())
    result = mod.generate_analysis_from_logs(tmp_path)
    assert len(result) == 1
    assert (tmp_path / "model_a" / f"{framework}_analysis.json").exists()


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_graceful_degradation_without_matplotlib(
    framework: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When matplotlib is unavailable the analyzer still completes analysis."""
    mod = _read_analyzer(framework)
    _write_results(tmp_path, framework, _realistic_payload())
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)

    # `import matplotlib` inside _generate_plots raises ImportError when the
    # module map entry is None, exercising the documented degraded path.
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    generated = mod.generate_analysis_from_logs(tmp_path, out_dir)
    assert generated
    analysis = json.loads(Path(generated[0]).read_text(encoding="utf-8"))
    assert analysis["framework"] == framework
    assert analysis["plots_generated"] is False
