"""The .rea writer of the v17 Nek5000 cases maps simulation.frame to p092 and simulation.power_mode to p096."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("mpi4py")  # hydrogym.nek.env (imported by nek_utils) needs mpi4py and pandas
pytest.importorskip("omegaconf")
pytest.importorskip("pandas")
from omegaconf import OmegaConf  # noqa: E402

from hydrogym.nek.nek_lib.nek_utils import (  # noqa: E402
    FRAME_CODES,
    NEK_INIT,
    power_mode,
    rea_mismatches,
    write_rea_from_config,
)

REA = Path(__file__).resolve().parents[1] / "third_party/nek5000/cases/naca0012_200k/naca_wing.rea"


def _params(path):
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[1].startswith("p") and parts[1][1:].isdigit():
            out[parts[1]] = float(parts[0])
    return out


@pytest.mark.parametrize("frame,power_mode", [("hg_fortran", 0), ("wall", 1), ("wall", 2)])
def test_rea_frame_power(tmp_path, frame, power_mode):
    shutil.copy(REA, tmp_path / "naca_wing.rea")
    before = _params(tmp_path / "naca_wing.rea")
    sim = OmegaConf.create({"CASENAME": "naca_wing", "frame": frame, "power_mode": power_mode, "dt": -4e-5})
    init = NEK_INIT(nek=sim, drl=OmegaConf.create({}), rank_folder=str(tmp_path))
    assert init.rewrite_REA_v17()
    after = _params(tmp_path / "naca_wing.rea")
    assert after["p092"] == FRAME_CODES[frame] and after["p096"] == power_mode
    assert after["p012"] == -4e-5
    # entries the configuration does not set keep the values of the file
    for key in ("p002", "p011", "p089", "p090"):
        assert after[key] == before[key]


def test_rea_frame_rejects_unknown(tmp_path):
    shutil.copy(REA, tmp_path / "naca_wing.rea")
    sim = OmegaConf.create({"CASENAME": "naca_wing", "frame": "global"})
    with pytest.raises(ValueError, match="frame"):
        NEK_INIT(nek=sim, drl=OmegaConf.create({}), rank_folder=str(tmp_path)).rewrite_REA_v17()


@pytest.mark.parametrize(
    "power,mode",
    [(None, 0), ({"power_sampling": "end"}, 2), ({"reference": {"span": 0.1}}, 1), ({"enabled": False}, 0)],
)
def test_write_rea_from_config_before_launch(tmp_path, power, mode):
    """prepare_workspace.py path: the cached .rea is a read-only symbolic link; the workspace gets its own copy."""
    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.copy(REA, cache / "naca_wing.rea")
    (cache / "naca_wing.rea").chmod(0o444)
    work = tmp_path / "work"
    work.mkdir()
    (work / "naca_wing.rea").symlink_to(cache / "naca_wing.rea")
    conf = OmegaConf.create({"simulation": {"CASENAME": "naca_wing", "frame": "wall", "dt": -4e-5}, "runner": {}})
    if power is not None:
        conf.power = power
    path = write_rea_from_config(conf, str(work))
    assert not Path(path).is_symlink()
    after = _params(Path(path))
    assert after["p092"] == 1.0 and after["p096"] == mode and after["p012"] == -4e-5
    assert _params(cache / "naca_wing.rea") == _params(REA)  # the cache is unchanged
    assert power_mode(conf) == mode
    assert rea_mismatches(conf.simulation, path) == []
    conf.simulation.frame = "hg_fortran"  # a value written after the launch would not reach the solver
    assert [m[0] for m in rea_mismatches(conf.simulation, path)] == ["p092"]


def test_frame_default_is_the_wall_frame():
    """Without simulation.frame in the environment configuration the .rea gets the wall-tangent frame (p092 = 1)."""
    from hydrogym.nek.configs import Config
    from hydrogym.nek.nek_lib.nek_utils import _rea_value

    conf = OmegaConf.merge(OmegaConf.structured(Config()), {"simulation": {"CASENAME": "naca_wing"}})
    assert conf.simulation.frame == "wall"
    assert _rea_value(conf.simulation, "frame") == 1.0
