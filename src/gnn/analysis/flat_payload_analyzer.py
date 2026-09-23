"""Shared analyzer for flat-payload simulation results.

PyMDP-agnostic framework analyzers (PyTorch, NumPyro, cpomdp) share an identical
analysis pipeline: discover ``simulation_results.json`` files, compute belief
entropy / confidence / action distribution / EFE metrics, write an
``<framework>_analysis.json`` per model, and render belief/action/EFE plots.

This module is the single implementation; each framework's ``analyzer.py``
re-exports ``generate_analysis_from_logs`` and ``_generate_plots`` bound to a
``FlatPayloadSpec`` so the public call sites (and test pins) remain unchanged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np

from .viz_base import MATPLOTLIB_AVAILABLE, plt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlatPayloadSpec:
    """Per-framework configuration for the shared flat-payload analyzer."""

    framework: str
    # rglob patterns for discovering simulation_results.json files.
    file_patterns: tuple[str, ...]
    # Output JSON filename written under each model's output directory.
    analysis_filename: str
    # Plotting labels.
    title_prefix: str
    bar_color: str
    # Log-line label (e.g. "PyTorch", "NumPyro") — keeps existing log format.
    log_label: str


def discover_result_files(results_dir: Path, spec: FlatPayloadSpec) -> list[Path]:
    """Find all simulation_results.json files matching ``spec`` under ``results_dir``.

    Includes a root-level ``simulation_results.json`` recovery fallback.
    Deduplicates by path.
    """
    results_dir = Path(results_dir)
    found: list[Path] = []
    for pattern in spec.file_patterns:
        found.extend(results_dir.rglob(pattern))
    root_result = results_dir / "simulation_results.json"
    if root_result.exists() and root_result not in found:
        found.append(root_result)
    return found


def compute_flat_payload_metrics(
    beliefs: np.ndarray, actions: list[Any], efe: np.ndarray
) -> dict[str, Any]:
    """Compute the standard flat-payload metrics dict.

    Pure function — no I/O. Beliefs are a 2-D ``(timesteps, states)`` array.
    """
    metrics: dict[str, Any] = {}
    if beliefs.ndim == 2 and beliefs.shape[0] > 0:
        entropy = -np.sum(beliefs * np.log(beliefs + 1e-16), axis=1)
        metrics["mean_belief_entropy"] = float(np.mean(entropy))
        metrics["final_belief_entropy"] = float(entropy[-1])
        confidence = np.max(beliefs, axis=1)
        metrics["mean_confidence"] = float(np.mean(confidence))
        metrics["final_confidence"] = float(confidence[-1])
    if actions:
        unique, counts = np.unique(np.asarray(actions), return_counts=True)
        metrics["action_distribution"] = {
            int(a): int(c) for a, c in zip(unique, counts)
        }
    if efe.ndim == 2 and efe.shape[0] > 0:
        metrics["mean_efe"] = float(np.mean(efe))
    return metrics


def compute_continuous_payload_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Metrics for a ``model_kind == "continuous"`` payload.

    Pure function — no I/O. Beliefs are Gaussian posterior means, so the
    categorical entropy/confidence metrics do not apply; the continuous
    schema carries its own error measure (``rmse_vs_true``) and, for the
    cpomdp backend, the per-step epistemic/pragmatic split of the chosen
    policy plus the per-policy ``efe_history``.
    """
    metrics: dict[str, Any] = {}
    rmse = results.get("rmse_vs_true")
    if isinstance(rmse, (int, float)):
        metrics["rmse_vs_true"] = float(rmse)
    if results.get("control_mode") is not None:
        metrics["control_mode"] = str(results["control_mode"])
    if results.get("n_policies") is not None:
        metrics["n_policies"] = int(results["n_policies"])
    efe = np.asarray(results.get("efe_history") or [], dtype=float)
    if efe.ndim == 2 and efe.shape[0] > 0:
        metrics["mean_efe"] = float(np.mean(efe))
        metrics["min_efe_per_step"] = np.min(efe, axis=1).tolist()
    for key, name in (
        ("epistemic_term", "mean_epistemic"),
        ("pragmatic_term", "mean_pragmatic"),
    ):
        series = np.asarray(results.get(key) or [], dtype=float)
        if series.ndim == 1 and series.shape[0] > 0:
            metrics[name] = float(np.mean(series))
    selected = results.get("selected_policy_index") or []
    if selected:
        unique, counts = np.unique(np.asarray(selected), return_counts=True)
        metrics["policy_distribution"] = {
            int(a): int(c) for a, c in zip(unique, counts)
        }
    covs = np.asarray(results.get("posterior_cov") or [], dtype=float)
    if covs.ndim == 3 and covs.shape[0] > 0:
        metrics["final_posterior_trace"] = float(np.trace(covs[-1]))
    return metrics


def _generate_continuous_term_plot(
    spec: FlatPayloadSpec, results: dict[str, Any], output_dir: Path
) -> bool:
    """The epistemic/pragmatic split panel for continuous EFE payloads."""
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return False
    epistemic = np.asarray(results.get("epistemic_term") or [], dtype=float)
    pragmatic = np.asarray(results.get("pragmatic_term") or [], dtype=float)
    if epistemic.ndim != 1 or epistemic.shape[0] == 0:
        return False
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epistemic, label="epistemic (info gain)", linewidth=1.5)
    if pragmatic.shape == epistemic.shape:
        ax.plot(pragmatic, label="pragmatic (goal)", linewidth=1.5)
        ax.plot(
            pragmatic - epistemic,
            label="G = pragmatic − epistemic",
            linewidth=1.0,
            linestyle="--",
        )
    ax.set_xlabel("Timestep", fontsize=14)
    ax.set_ylabel("nats", fontsize=14)
    ax.set_title(f"{spec.title_prefix} — EFE terms of the selected policy", fontsize=16)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "efe_terms.png", dpi=150)
    plt.close(fig)
    return True


def _generate_plots(
    spec: FlatPayloadSpec,
    beliefs: np.ndarray,
    actions: list[Any],
    observations: list[Any],
    efe: np.ndarray,
    output_dir: Path,
    series_label: str = "Action",
) -> bool:
    """Generate the standard three plots (belief, action, EFE).

    ``series_label`` names the EFE columns (``"Action"`` for categorical
    payloads, ``"Policy"`` for the continuous enumerated-policy panel).
    Returns True if at least one plot was written; False if matplotlib is
    unavailable or no plottable data.
    """
    if not MATPLOTLIB_AVAILABLE or plt is None:
        logger.warning("matplotlib not available — skipping plots")
        return False
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = False

    if beliefs.ndim == 2 and beliefs.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        for s in range(beliefs.shape[1]):
            ax.plot(beliefs[:, s], label=f"State {s}", linewidth=1.5)
        ax.set_xlabel("Timestep", fontsize=14)
        ax.set_ylabel("Belief", fontsize=14)
        ax.set_title(f"{spec.title_prefix} — Belief Trajectory", fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / "belief_trajectory.png", dpi=150)
        plt.close(fig)
        saved = True

    if actions:
        fig, ax = plt.subplots(figsize=(6, 4))
        unique, counts = np.unique(np.asarray(actions), return_counts=True)
        ax.bar(unique.astype(str), counts, color=spec.bar_color)
        ax.set_xlabel("Action", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.set_title(f"{spec.title_prefix} — Action Distribution", fontsize=16)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / "action_distribution.png", dpi=150)
        plt.close(fig)
        saved = True

    if efe.ndim == 2 and efe.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        for a_idx in range(efe.shape[1]):
            ax.plot(efe[:, a_idx], label=f"{series_label} {a_idx}", linewidth=1.5)
        ax.set_xlabel("Timestep", fontsize=14)
        ax.set_ylabel("EFE", fontsize=14)
        ax.set_title(f"{spec.title_prefix} — Expected Free Energy", fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / "efe_history.png", dpi=150)
        plt.close(fig)
        saved = True

    if saved:
        logger.info(f"✅ {spec.log_label} analysis plots saved to: {output_dir}")
    return saved


def generate_analysis_from_logs(
    spec: FlatPayloadSpec,
    results_dir: Path,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> List[str]:
    """Generate analysis from flat-payload simulation results.

    Discovers ``simulation_results.json`` files matching ``spec.file_patterns``,
    computes metrics, renders plots, and writes ``<framework>_analysis.json``
    per model. Returns the list of generated JSON file paths.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[str] = []
    result_files = discover_result_files(results_dir, spec)

    if not result_files:
        if verbose:
            logger.debug(
                f"No {spec.log_label} simulation_results.json found under {results_dir}"
            )
        return generated_files

    for results_file in result_files:
        try:
            with open(results_file) as f:
                results = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read results: {e}")
            continue

        # Determine model name from path: the segment preceding the framework.
        path_parts = results_file.parts
        model_name = results.get("model_name", "unknown")
        for i, part in enumerate(path_parts):
            if part == spec.framework and i >= 1:
                model_name = path_parts[i - 1]
                break

        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        beliefs = np.array(results.get("beliefs", []))
        actions = results.get("actions", [])
        observations = results.get("observations", [])
        efe = np.array(results.get("efe_history", []))
        validation = results.get("validation", {})

        continuous = results.get("model_kind") == "continuous"
        analysis: dict[str, Any] = {
            "framework": spec.framework,
            "model_name": model_name,
            "num_timesteps": (
                int(results.get("num_timesteps") or len(beliefs))
                if continuous
                else len(actions)
            ),
            "num_states": beliefs.shape[1] if beliefs.ndim == 2 else 0,
            "validation": validation,
            "metrics": (
                compute_continuous_payload_metrics(results)
                if continuous
                else compute_flat_payload_metrics(beliefs, actions, efe)
            ),
        }
        if continuous:
            analysis["model_kind"] = "continuous"

        try:
            plots_ok = _generate_plots(
                spec,
                beliefs,
                actions,
                observations,
                efe,
                model_output_dir,
                series_label="Policy" if continuous else "Action",
            )
            if continuous:
                plots_ok = (
                    _generate_continuous_term_plot(spec, results, model_output_dir)
                    or plots_ok
                )
            analysis["plots_generated"] = bool(plots_ok)
        except Exception as e:
            logger.warning(f"Plot generation failed for {model_name}: {e}")
            analysis["plots_generated"] = False

        analysis_file = model_output_dir / spec.analysis_filename
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        generated_files.append(str(analysis_file))
        logger.info(f"✅ {spec.log_label} analysis saved: {model_name}")

    return generated_files


__all__ = [
    "FlatPayloadSpec",
    "compute_continuous_payload_metrics",
    "compute_flat_payload_metrics",
    "discover_result_files",
    "generate_analysis_from_logs",
]
