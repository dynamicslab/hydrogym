"""m-AIA env path of the wall-jet power: info keys, equality with the stand-alone model, reward charge."""

import numpy as np
import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("omegaconf")

from hydrogym.maia import wall_jets as wj  # noqa: E402
from hydrogym.maia.env_core import MaiaFlowEnv  # noqa: E402
from hydrogym.power import PowerConfig  # noqa: E402

ORIGIN = (-16.0, -25.6)


def field(pts):
    """Analytical 2D flow at the probe points, m-AIA layout (u, v, rho, p)."""
    x, y = pts[:, 0], pts[:, 1]
    return np.stack([0.01 * y, 0.002 * x, np.ones_like(x), 1 / 3 + 0.01 * x + 0.004 * y**2], 1)


class FieldInterface:
    def runTimeSteps(self, n):
        pass

    def setControlProperties(self, props):
        pass

    def continueRun(self):
        pass

    def getProbeData(self, locations):
        return field(np.asarray(locations, dtype=float).reshape(-1, 2)).ravel()


def _env(weight=0.0):
    props = dict(
        nDim=2,
        lbNoJets=2,
        lbJetCenter=[0, 0, 0, 0],
        lbJetRadius=[0.5, 0.5],
        lbJetAngles=[np.pi / 2, 3 * np.pi / 2],
        lbJetRanges=[0.2, 0.2],
    )
    jets = wj.Bc2007Jets.from_properties(props)
    env = MaiaFlowEnv.__new__(MaiaFlowEnv)
    env.MAX_CONTROL = 0.03
    env.num_substeps_per_iteration = 3
    env.num_inputs = 2
    env.maiaInterface = FieldInterface()
    env.probe_locations = [2.0, 0.0]
    env.noProbes = 1
    env.observation_type = ["u", "v"]
    env.nDim = 2
    env.bcId = []
    env.convert_action = lambda action=None: list(action)
    env.get_reward = lambda: (-1.0, {"forces": []})
    env.obs_loc = np.zeros(2)
    env.obs_scale = np.ones(2)
    env.iter = 0
    env.max_episode_steps = 10
    env.power_cfg = PowerConfig(weight=weight, cost="abs")
    env._power_kind = "wall_jets"
    env._power_model = wj.WallJetPower(wj.flat_wall_jets(jets, 0.05, ORIGIN), 2, ORIGIN, 0.05, 0.1, 20.0)
    return env


def test_info_equals_standalone_model():
    env = _env()
    _, r, _, _, info = env.step(np.array([1.0, -0.5]))
    ref = env._power_model.evaluate(field(env._power_model.probe_coordinates.reshape(-1, 2)).ravel(), [0.03, -0.015])
    assert info["actuation_power"] == pytest.approx(ref["forms"]["signed"], rel=1e-12)
    assert info["actuation_power_abs"] == pytest.approx(ref["forms"]["abs"], rel=1e-12)
    np.testing.assert_allclose(info["actuation_power_per_actuator"], ref["per_actuator"], rtol=1e-12)
    assert info["power_model"] == "wall_jets" and np.isfinite(info["actuation_power_model_spread"])
    assert r == -1.0  # weight 0: reward unchanged


def test_reset_reports_zero_and_weight_charges_abs():
    env = _env(weight=2.0)
    env.reset = None  # (reset needs a solver; check its power info directly)
    info0 = MaiaFlowEnv._power_info(env, None, None)
    np.testing.assert_array_equal(info0["actuation_power_per_actuator"], [0.0, 0.0])
    _, r, _, _, info = env.step(np.array([1.0, 1.0]))
    assert r == pytest.approx(-1.0 - 2.0 * info["actuation_power_abs"], rel=1e-12)


class RecordingInterface(FieldInterface):
    def __init__(self):
        self.requests = []

    def getProbeData(self, locations):
        self.requests.append(list(locations))
        return super().getProbeData(locations)


def test_one_fixed_probe_request_per_step():
    """Observation probes and power-model points go to m-AIA as one list, identical every step (m-AIA caches the
    cell ids of the last list); the values equal those of separate requests."""
    env = _env()
    rec = RecordingInterface()
    env.maiaInterface = rec
    infos, obs = [], []
    for a in ([1.0, -0.5], [0.2, 0.7], [-1.0, 1.0]):
        o, _, _, _, info = env.step(np.array(a))
        obs.append(o)
        infos.append(info)
    assert len(rec.requests) == 3 and all(r == rec.requests[0] for r in rec.requests)
    expected = list(env.probe_locations) + env._power_model.probe_coordinates.tolist()
    assert rec.requests[0] == expected
    np.testing.assert_array_equal(obs[0], field(np.array([[2.0, 0.0]]))[0, :2])
    cmd = [a * env.MAX_CONTROL for a in (0.2, 0.7)]
    ref = env._power_model.evaluate(field(env._power_model.probe_coordinates.reshape(-1, 2)).ravel(), cmd)
    assert infos[1]["actuation_power"] == ref["forms"]["signed"]  # bit-identical to a separate request
