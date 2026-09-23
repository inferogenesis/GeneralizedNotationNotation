#!/usr/bin/env python3
"""
cpomdp Renderer for GNN Specifications

Renders continuous (linear-Gaussian) GNN models to standalone cpomdp
simulation scripts: exact Kalman filtering plus an exhaustive
expected-free-energy search over a declared finite action set. Discrete
POMDP specs are refused with the ``unsupported`` status — cpomdp is the
continuous-state sibling of pymdp, not a categorical backend.

@Web: https://github.com/inferogenesis/cpomdp
@Web: https://cpomdp.inferogenesis.com/
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gnn.render.naming import atomic_write_text

logger = logging.getLogger(__name__)

UNSUPPORTED_MESSAGE = "cpomdp renders continuous (linear-Gaussian) models only"


def render_gnn_to_cpomdp(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """Render a continuous GNN specification to a cpomdp simulation script.

    Args:
        gnn_spec: Parsed GNN model specification (continuous branch).
        output_path: Path to write the generated script.
        options: Optional rendering options — ``control_mode`` (``"efe"`` |
            ``"parity"``), ``action_scale``, ``horizon``, ``goal_precision``.

    Returns:
        Tuple of (success: bool, message: str, artifact_paths: List[str]).
        A discrete spec returns ``(False, UNSUPPORTED_MESSAGE, [])`` so the
        processor records it under ``unsupported_framework_renderings``.
    """
    from gnn.render.continuous_common import (
        extract_continuous_spec,
        is_continuous_spec,
    )

    if not is_continuous_spec(gnn_spec):
        return False, UNSUPPORTED_MESSAGE, []

    try:
        from gnn.render.cpomdp.script_template import (
            generate_cpomdp_script,
            resolve_render_options,
        )

        spec = extract_continuous_spec(gnn_spec)
        resolved = resolve_render_options(options)
        code = generate_cpomdp_script(spec, resolved)
        output_path = Path(output_path)
        atomic_write_text(output_path, code)
        mode = resolved["control_mode"] if spec.has_control else "passive"
        logger.info(f"✅ cpomdp continuous script written to: {output_path}")
        return (
            True,
            f"cpomdp continuous script generated ({mode}): {output_path}",
            [str(output_path)],
        )
    except Exception as e:
        logger.error(f"❌ cpomdp rendering failed: {e}")
        return False, f"cpomdp rendering failed: {e}", []
