"""Every example script must run end to end (headless)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = sorted(p.name for p in EXAMPLES.glob("*.py") if not p.name.startswith("_"))
EXTRA_ARGS = {"large_point_clouds.py": ["--points", "2000"]}


@pytest.fixture(scope="session")
def mpl_config(tmp_path_factory) -> Path:
    """One matplotlib config/cache directory for all runs (the font cache is built once)."""
    return tmp_path_factory.mktemp("matplotlib")


def _run(script: str, *args: str, config: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(config)}
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), "--device", "cpu", "--no-show", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs(script, mpl_config):
    result = _run(script, *EXTRA_ARGS.get(script, []), config=mpl_config)
    assert result.returncode == 0, result.stderr


@pytest.mark.slow
def test_example_saves_animation(tmp_path, mpl_config):
    gif = tmp_path / "rigid.gif"
    result = _run("fish_rigid_2D.py", "--save", str(gif), config=mpl_config)
    assert result.returncode == 0, result.stderr
    assert gif.stat().st_size > 1000


@pytest.mark.slow
def test_quickstart_notebook_runs(tmp_path, mpl_config):
    """Execute the code cells of the notebook as one script."""
    notebook = json.loads((EXAMPLES / "quickstart.ipynb").read_text(encoding="utf-8"))
    code = "\n\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    script = tmp_path / "quickstart.py"
    script.write_text(code, encoding="utf-8")
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(mpl_config)}
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "correct matches: 1.0" in result.stdout
