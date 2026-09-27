"""Continuous-model checks: the ``R_x_family`` / ``R_x_params`` sensor pair.

A continuous (linear-Gaussian) GNN file may declare a state-dependent
observation noise ``R(x)`` on top of the nominal ``R`` through two optional
``InitialParameterization`` keys. The family vocabulary and each family's
arity live in ``render.continuous_common.SENSOR_FAMILIES``; this module
re-checks them at Step 5 so an unknown family or a wrong-length parameter
vector is reported before any backend renders the file.
"""

from __future__ import annotations

import re
from typing import Any

from gnn.render.continuous_common import SENSOR_FAMILIES

from .sections import extract_markdown_section

_FAMILY_RE = re.compile(
    r"^\s*R_x_family\s*=\s*\{?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}?\s*(?:#.*)?$"
)
_PARAMS_RE = re.compile(r"^\s*R_x_params\s*=\s*(.+?)\s*(?:#.*)?$")


def _count_numbers(literal: str) -> int | None:
    """Number of numeric tokens in a brace/paren literal, or ``None`` if junk."""
    body = literal.strip().strip("{}").strip().strip("()").strip()
    if not body:
        return 0
    tokens = [tok.strip() for tok in body.split(",") if tok.strip()]
    for tok in tokens:
        try:
            float(tok)
        except ValueError:
            return None
    return len(tokens)


def validate_sensor_family(content: str) -> dict[str, Any]:
    """Check the optional state-dependent sensor declaration of a GNN file.

    Returns ``{"declared": bool, "family": str | None, "arity": int | None,
    "errors": [...]}``. Files without either key are ``declared: False`` with
    no errors; the check never touches discrete files.
    """
    section = extract_markdown_section(content, "InitialParameterization")
    result: dict[str, Any] = {
        "declared": False,
        "family": None,
        "arity": None,
        "errors": [],
    }
    if not section:
        return result
    family: str | None = None
    params_literal: str | None = None
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _FAMILY_RE.match(stripped)
        if m:
            family = m.group(1)
            continue
        m = _PARAMS_RE.match(stripped)
        if m:
            params_literal = m.group(1)
    if family is None and params_literal is None:
        return result
    result["declared"] = True
    result["family"] = family
    if family is None:
        result["errors"].append(
            "[GNN-E007] R_x_params declared without R_x_family "
            "(state-dependent observation noise needs a family name)"
        )
        return result
    if family not in SENSOR_FAMILIES:
        result["errors"].append(
            f"[GNN-E007] Unknown R_x_family {family!r}; expected one of "
            f"{sorted(SENSOR_FAMILIES)}"
        )
        return result
    arity = SENSOR_FAMILIES[family]
    result["arity"] = arity
    count = 0 if params_literal is None else _count_numbers(params_literal)
    if count is None:
        result["errors"].append(
            f"[GNN-E007] R_x_params for {family!r} must be a numeric vector, "
            f"got {params_literal!r}"
        )
    elif count != arity:
        result["errors"].append(
            f"[GNN-E007] R_x_family {family!r} takes {arity} parameter(s) in "
            f"R_x_params, got {count}"
        )
    return result


__all__ = ["validate_sensor_family"]
