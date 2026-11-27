"""Real-solver smoke test: the released-policy example runs two actions through the real m-AIA (MPMD).

Marked ``solver``. It needs an m-AIA binary (``maia`` on the PATH or ``MAIA``), ``mpirun`` and the Hugging Face dataset
(network or a populated cache), so it only runs when ``HYDROGYM_SOLVER_TESTS=1`` is set (the images' CI step does) and
skips otherwise. Other launchers: ``MPIRUN="mpirun --bind-to none --oversubscribe"``.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(os.environ.get("HYDROGYM_ROOT", Path(__file__).resolve().parents[1]))
RUN_EXAMPLE = ROOT / "examples" / "maia" / "released_policies" / "run_example.sh"

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(os.environ.get("HYDROGYM_SOLVER_TESTS") != "1", reason="set HYDROGYM_SOLVER_TESTS=1 to run"),
    pytest.mark.skipif(shutil.which(os.environ.get("MAIA", "maia")) is None, reason="no m-AIA binary"),
    pytest.mark.skipif(shutil.which("mpirun") is None and not os.environ.get("MPIRUN"), reason="no mpirun"),
]


def test_cylinder_re200_policy_two_actions(tmp_path):
    env = dict(os.environ, PLOT="0", MPIRUN=os.environ.get("MPIRUN", "mpirun --bind-to none --oversubscribe"))
    out = subprocess.run(
        ["bash", str(RUN_EXAMPLE), "cylinder_2d_re200/ppo/seed1", str(tmp_path / "run"), "1", "--actions", "2"],
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    run = np.load(tmp_path / "run" / "run.npz")
    assert len(run.files) > 0
    for name in run.files:
        arr = np.asarray(run[name])
        if arr.dtype.kind == "f":
            assert np.all(np.isfinite(arr[~np.isnan(arr)])), name
