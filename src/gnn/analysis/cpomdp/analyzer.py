#!/usr/bin/env python3
"""
cpomdp Analyzer for GNN Pipeline

Reads simulation_results.json produced by the cpomdp runner and generates
belief-mean trajectory and expected-free-energy analysis plots.

The analysis pipeline is shared with the PyTorch and NumPyro analyzers via
``analysis.flat_payload_analyzer``; this module binds it to the cpomdp spec
(framework name, file patterns, plot labels). cpomdp results are always
continuous (``model_kind == "continuous"``), so the shared engine's
continuous branch applies: RMSE against the true trajectory, the per-step
epistemic/pragmatic split, and the per-policy EFE panel.

@Web: https://github.com/inferogenesis/cpomdp
"""

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ..flat_payload_analyzer import (
    FlatPayloadSpec,
)
from ..flat_payload_analyzer import (
    generate_analysis_from_logs as _generate_from_spec,
)

# Public spec: cpomdp-prefixed result files, cpomdp bar color, cpomdp labels.
CPOMDP_SPEC = FlatPayloadSpec(
    framework="cpomdp",
    file_patterns=(
        "**/cpomdp/**/simulation_results.json",
        "**/cpomdp_simulation_results.json",
    ),
    analysis_filename="cpomdp_analysis.json",
    title_prefix="cpomdp",
    bar_color="#0B7285",
    log_label="cpomdp",
)


def generate_analysis_from_logs(
    results_dir: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> List[str]:
    """Generate analysis from cpomdp simulation results.

    Searches recursively for simulation_results.json files under results_dir
    (including model/cpomdp/simulation_data subdirectories).

    Args:
        results_dir: Root directory to search for results (e.g. 12_execute_output).
        output_dir: Directory for analysis artifacts. Defaults to results_dir.
        verbose: Enable verbose logging.

    Returns:
        List of generated output file paths.
    """
    return _generate_from_spec(CPOMDP_SPEC, results_dir, output_dir, verbose)


def _generate_plots(
    beliefs: np.ndarray,
    actions: list[Any],
    observations: list[Any],
    efe: np.ndarray,
    output_dir: Path,
) -> bool:
    """Generate analysis plots using matplotlib.

    Returns True if at least one plot artifact was written, False otherwise
    (e.g. matplotlib unavailable or no plottable data).
    """
    from ..flat_payload_analyzer import _generate_plots as _shared_plots

    return _shared_plots(CPOMDP_SPEC, beliefs, actions, observations, efe, output_dir)


__all__ = [
    "CPOMDP_SPEC",
    "generate_analysis_from_logs",
]
