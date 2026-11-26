"""3D open cavity (``Cavity{,3Jet}_3D_Re{4200,7500}_lz024``): the paths of the cavity envs that only the 3D cases use.

* ``from_hf`` routes the ``_lz024`` names to the 2D classes (the type is the name's first token);
* the baseline lookup works with 3D probe triples (nearest point of ``baseline_state.feather``, u then v);
* ``step()`` reads u, v from the 3D probe layout of m-AIA (u, v, w, rho, p per probe).

No solver: the m-AIA interface is a fake and the env objects are built without ``__init__``.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("omegaconf")

import hydrogym.maia as maia  # noqa: E402  (importing the package registers the env classes)
from hydrogym.maia import env_core  # noqa: E402
from hydrogym.maia.envs.cavity import Cavity, Cavity3Jet, CavityBase  # noqa: E402

NAMES = {
    "Cavity_3D_Re4200_lz024": Cavity,
    "Cavity_3D_Re7500_lz024": Cavity,
    "Cavity3Jet_3D_Re4200_lz024": Cavity3Jet,
    "Cavity3Jet_3D_Re7500_lz024": Cavity3Jet,
}


@pytest.mark.parametrize("name", sorted(NAMES))
def test_from_hf_routes_lz024_names(monkeypatch, name):
    seen = {}

    class Capture:
        def __init__(self, env_config):
            seen.update(env_config)

    cls = NAMES[name]
    key = next(k for k, v in env_core._ENVIRONMENT_REGISTRY.items() if v is cls)
    monkeypatch.setitem(env_core._ENVIRONMENT_REGISTRY, key, Capture)
    maia.from_hf(name, local_fallback_dir="/nonexistent", probe_locations=[0.5, -0.1, 0.0])
    assert seen["environment_name"] == name
    assert seen["probe_locations"] == [0.5, -0.1, 0.0]


def _plane_points():
    # 3 spanwise planes x 4 cells, physical coordinates, as the 3D baseline (Points:0..2 + u, v, w, rho)
    x, y, z = np.meshgrid([0.1, 0.2], [-0.1, 0.1], [-0.1, 0.0, 0.1], indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], 1)


def test_baseline_lookup_with_3d_probes(tmp_path):
    pytest.importorskip("pyarrow")  # pandas feather I/O
    pts = _plane_points()
    u = 0.007 + 0.001 * pts[:, 0]
    v = -0.002 * pts[:, 1] + 0.0001 * pts[:, 2]
    pd.DataFrame(
        {
            "Points:0": pts[:, 0],
            "Points:1": pts[:, 1],
            "Points:2": pts[:, 2],
            "rho": np.ones(len(pts)),
            "u": u,
            "v": v,
            "w": np.zeros(len(pts)),
        }
    ).to_feather(tmp_path / "baseline_state.feather")
    env = CavityBase.__new__(CavityBase)
    env.env_data_path = str(tmp_path)
    env.environment_name = "Cavity3Jet_3D_Re7500_lz024"
    env.nDim = 3
    env.observation_type = ["u", "v"]
    probes = pts[[5, 0, 11]] + 1e-4  # next to three of the baseline points
    env.probe_locations = probes.ravel().tolist()
    env.num_outputs = 2 * len(probes)
    env.reward_strategy = "baseline_mean"
    env._env_config = {}
    env._baseline_loaded = False
    env._ensure_baseline_loaded()
    np.testing.assert_array_equal(env.baseline_values, np.concatenate([u[[5, 0, 11]], v[[5, 0, 11]]]))


class Probe3DInterface:
    """m-AIA 3D probe layout: per probe (u, v, w, rho, p)."""

    def __init__(self, n):
        self.n = n
        self.controls = []

    def runTimeSteps(self, n):
        pass

    def setControlProperties(self, props):
        self.controls.append(list(np.atleast_1d(props)))

    def continueRun(self):
        pass

    def getProbeData(self, locations):
        k = np.arange(self.n, dtype=float)
        return np.stack([1 + k, 10 + k, 100 + k, np.ones(self.n), np.full(self.n, 1 / 3)], 1).ravel()


def test_step_reads_u_v_from_3d_probe_layout():
    n = 4
    env = env_core.MaiaFlowEnv.__new__(env_core.MaiaFlowEnv)
    env.MAX_CONTROL = 0.005
    env.num_substeps_per_iteration = 527
    env.num_inputs = 3
    env.maiaInterface = Probe3DInterface(n)
    env.probe_locations = np.zeros(3 * n).tolist()
    env.noProbes = n
    env.observation_type = ["u", "v"]
    env.nDim = 3
    env.bcId = []
    env.convert_action = lambda action=None: list(action)
    env.get_reward = lambda: (0.0, {})
    env.obs_loc = np.zeros(2 * n)
    env.obs_scale = np.ones(2 * n)
    env.iter = 0
    env.max_episode_steps = 200
    obs, _, terminated, truncated, _ = env.step(np.array([1.0, -1.0, 0.5]))
    np.testing.assert_array_equal(obs, np.r_[1 + np.arange(n), 10 + np.arange(n)])
    assert env.maiaInterface.controls[-1] == pytest.approx([0.005, -0.005, 0.0025])
    assert not terminated and not truncated
