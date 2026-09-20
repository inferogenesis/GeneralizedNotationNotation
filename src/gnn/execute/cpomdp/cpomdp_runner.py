#!/usr/bin/env python3
"""
cpomdp Runner for Executing Rendered cpomdp Continuous Scripts

Discovers and runs cpomdp-generated scripts via subprocess with dependency
checking, syntax validation, a run-length cap, log persistence, and
execution timing. Mirrors the NumPyro runner; the only cpomdp-specific
behaviour is the cap: ``RecedingHorizonSelector`` re-plans every step, so
the runner refuses ``NUM_TIMESTEPS > 60`` unless ``CPOMDP_ALLOW_LONG=1``.

@Web: https://github.com/inferogenesis/cpomdp
"""

import json as json_mod
import logging
import os
import re
import sys
import tempfile
import time as time_mod
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)

#: Longest run executed without the explicit opt-in (receding-horizon
#: re-planning scores every policy on every step).
MAX_TIMESTEPS_WITHOUT_OPT_IN = 60
ALLOW_LONG_ENV = "CPOMDP_ALLOW_LONG"
OUTPUT_DIR_ENV = "CPOMDP_OUTPUT_DIR"

_NUM_TIMESTEPS_RE = re.compile(r"^NUM_TIMESTEPS\s*=\s*(\d+)\s*$", re.MULTILINE)


def is_cpomdp_available() -> bool:
    """Check if cpomdp (and its JAX backend) is importable."""
    try:
        import cpomdp
        import jax

        logger.info(f"cpomdp version: {cpomdp.__version__} (JAX {jax.__version__})")
        return True
    except ImportError as e:
        logger.error(f"cpomdp not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking cpomdp: {e}")
        return False


def find_cpomdp_scripts(
    base_dir: Union[str, Path], recursive: bool = True
) -> List[Path]:
    """Find cpomdp scripts in the specified directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Directory not found: {base_path}")
        return []
    pattern = "**/*.py" if recursive else "*.py"
    return [
        f
        for f in base_path.glob(pattern)
        if "cpomdp" in f.name.lower() or f.parent.name == "cpomdp"
    ]


def declared_timesteps(script_text: str) -> Optional[int]:
    """Read the ``NUM_TIMESTEPS`` literal the renderer wrote into a script."""
    match = _NUM_TIMESTEPS_RE.search(script_text)
    return int(match.group(1)) if match else None


def long_run_allowed() -> bool:
    """Whether ``CPOMDP_ALLOW_LONG`` opts into runs past the cap."""
    return os.environ.get(ALLOW_LONG_ENV, "").strip().lower() in ("1", "true", "yes")


def refuse_long_run(script_text: str) -> Optional[str]:
    """Return the refusal reason when the script exceeds the cap without opt-in."""
    steps = declared_timesteps(script_text)
    if steps is None or steps <= MAX_TIMESTEPS_WITHOUT_OPT_IN or long_run_allowed():
        return None
    return (
        f"refusing cpomdp run: NUM_TIMESTEPS={steps} exceeds the "
        f"{MAX_TIMESTEPS_WITHOUT_OPT_IN}-step cap (the receding-horizon search "
        f"re-plans every step); set {ALLOW_LONG_ENV}=1 to run it anyway"
    )


def execute_cpomdp_script(
    script_path: Path,
    verbose: bool = False,
    output_dir: Optional[Path] = None,
    timeout: int = 600,
) -> bool:
    """Execute a single cpomdp script with log persistence.

    Args:
        script_path: Path to the cpomdp script.
        verbose: Enable verbose output.
        output_dir: Directory for execution logs and ``simulation_results.json``.
        timeout: Execution timeout in seconds.
    """
    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        return False

    logger.info(f"Executing cpomdp script: {script_path}")

    # Dependency check
    for dep in ("jax", "cpomdp", "numpy"):
        try:
            __import__(dep)
            logger.debug(f"✅ Dependency available: {dep}")
        except ImportError:
            logger.error(f"❌ Missing required dependency: {dep}")
            logger.error("Install with: uv sync --extra cpomdp")
            return False

    # Syntax validation and the run-length cap
    try:
        content = script_path.read_text()
        compile(content, script_path.name, "exec")
        logger.debug(f"✅ Script syntax valid: {script_path.name}")
    except SyntaxError as e:
        logger.error(f"❌ Syntax error in {script_path.name}: {e}")
        return False
    refusal = refuse_long_run(content)
    if refusal is not None:
        logger.error(f"❌ {script_path.name}: {refusal}")
        return False

    env = os.environ.copy()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        env[OUTPUT_DIR_ENV] = str(output_dir)

    # Delegate the subprocess envelope (run + timing + error normalization) to
    # the canonical safe executor. Function-local import: executor.py imports
    # this runner at module scope, so a module-level import would be circular.
    from gnn.execute.executor import execute_script_safely

    abs_path = script_path.resolve()
    envelope = execute_script_safely(
        abs_path,
        timeout=timeout,
        cwd=abs_path.parent,
        env=env,
    )

    if envelope["return_code"] == -1 and "error" in envelope:
        logger.error(f"❌ Error executing {script_path.name}: {envelope['error']}")
        return False

    elapsed = envelope["duration_seconds"]
    success: bool = bool(envelope["success"])
    result_stdout = envelope["stdout"]
    result_stderr = envelope["stderr"]
    return_code = envelope["return_code"]

    if success:
        logger.info(f"✅ Script executed: {script_path.name} ({elapsed:.1f}s)")
        if verbose and result_stdout.strip():
            logger.debug(f"Output:\n{result_stdout}")
    else:
        logger.error(f"❌ Script failed: {script_path.name}")
        logger.error(f"Return code: {return_code}")
        if result_stderr.strip():
            logger.error(f"Error:\n{result_stderr}")

    # Save execution logs
    log_dir = output_dir if output_dir else abs_path.parent
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.txt"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=stdout_path.parent, delete=False
        ) as tmp_f:
            tmp_f.write(result_stdout)
        os.replace(tmp_f.name, str(stdout_path))
        stderr_path = log_dir / "stderr.txt"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=stderr_path.parent, delete=False
        ) as tmp_f:
            tmp_f.write(result_stderr)
        os.replace(tmp_f.name, str(stderr_path))
        execution_log: dict[str, Any] = {
            "script": str(abs_path),
            "return_code": return_code,
            "success": success,
            "elapsed_seconds": round(elapsed, 2),
            "timeout": timeout,
            "timestamp": time_mod.strftime("%Y-%m-%d %H:%M:%S"),
        }
        exec_log_path = log_dir / "execution_log.json"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=exec_log_path.parent, delete=False
        ) as tmp_f:
            tmp_f.write(json_mod.dumps(execution_log, indent=2))
        os.replace(tmp_f.name, str(exec_log_path))
        logger.debug(f"Execution logs saved to: {log_dir}")
    except Exception as log_err:
        logger.warning(f"Could not save execution logs: {log_err}")

    return success


def run_cpomdp_scripts(
    rendered_simulators_dir: Union[str, Path],
    execution_output_dir: Optional[Union[str, Path]] = None,
    recursive_search: bool = True,
    verbose: bool = False,
) -> bool:
    """Find and run cpomdp scripts on rendered models."""
    if not is_cpomdp_available():
        logger.error("cpomdp not available, cannot execute cpomdp scripts")
        return False

    if execution_output_dir:
        exec_out = Path(execution_output_dir)
        exec_out.mkdir(parents=True, exist_ok=True)
        logger.info(f"cpomdp execution outputs → {exec_out}")

    # Step 11 writes ``<render_dir>/<file_stem>/cpomdp/<model>_cpomdp.py``;
    # a flat ``<render_dir>/cpomdp/`` layout is found by the same walk.
    render_dir = Path(rendered_simulators_dir)
    logger.info(f"Looking for cpomdp scripts under: {render_dir}")
    scripts = sorted(find_cpomdp_scripts(render_dir, recursive_search))
    if not scripts:
        logger.info("No cpomdp scripts found")
        return True

    success_count = 0
    failure_count = 0
    for script in scripts:
        out = Path(execution_output_dir) / script.stem if execution_output_dir else None
        if execute_cpomdp_script(script, verbose, out):
            success_count += 1
        else:
            failure_count += 1

    logger.info(
        f"cpomdp execution summary: {success_count} succeeded, "
        f"{failure_count} failed, {success_count + failure_count} total"
    )
    return failure_count == 0


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description="Execute cpomdp scripts generated by GNN rendering step"
    )
    parser.add_argument("--output-dir", type=Path, default="../output")
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    ok = run_cpomdp_scripts(
        rendered_simulators_dir=args.output_dir,
        recursive_search=args.recursive,
        verbose=args.verbose,
    )
    sys.exit(0 if ok else 1)
