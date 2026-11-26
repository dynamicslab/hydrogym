"""m-AIA env path of the solver jet power (structured finite-volume jets, bc1008): info keys and reward charge."""

import numpy as np
import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("omegaconf")

from hydrogym.maia.env_core import MaiaFlowEnv  # noqa: E402
from hydrogym.power import PowerConfig  # noqa: E402

# [P, P_pos, P_abs, P_0, P_1, Q_0, Q_1] as m-AIA returns it for two jets
REPLY = np.array([0.3, 0.5, 0.9, 0.4, -0.1, 0.02, -0.01])


class JetPowerInterface:
    def __init__(self, reply=REPLY):
        self.reply = np.asarray(reply, dtype=float)
        self.power_requests = 0

    def runTimeSteps(self, n):
        pass

    def setControlProperties(self, props):
        pass

    def continueRun(self):
        pass

    def getProbeData(self, locations):
        return np.zeros(5 * (len(locations) // 3))

    def getActuationPower(self):
        self.power_requests += 1
        return self.reply.copy()


def _env(weight=0.0, cost="abs", reply=REPLY):
    env = MaiaFlowEnv.__new__(MaiaFlowEnv)
    env.MAX_CONTROL = 0.05
    env.num_substeps_per_iteration = 3
    env.num_inputs = 2
    env.maiaInterface = JetPowerInterface(reply)
    env.probe_locations = [0.4, 0.1, 0.05]
    env.noProbes = 1
    env.observation_type = ["u", "v"]
    env.nDim = 3
    env.bcId = []
    env.convert_action = lambda action=None: list(action)
    env.get_reward = lambda: (-1.0, {"forces": [], "C_D": 1.0, "C_P": 0.0})
    env.drag_coefficient_from_objective = lambda obj: obj["C_D"]
    env.obs_loc = np.zeros(2)
    env.obs_scale = np.ones(2)
    env.iter = 0
    env.max_episode_steps = 10
    env.power_cfg = PowerConfig(weight=weight, cost=cost)
    env._power_kind = "solver_jets"
    env._power_model = "solver_jets"
    return env


def test_info_from_solver_reply():
    env = _env()
    _, r, _, _, info = env.step(np.array([1.0, -0.5]))
    assert env.maiaInterface.power_requests == 1
    assert info["power_model"] == "solver_jets" and info["actuation_power_valid"]
    assert info["actuation_power"] == 0.3
    assert info["actuation_power_positive"] == 0.5
    assert info["actuation_power_abs"] == 0.9
    np.testing.assert_array_equal(info["actuation_power_per_actuator"], [0.4, -0.1])
    assert info["drag_coefficient"] == 1.0
    assert r == -1.0  # weight 0: reward unchanged (the C_P of the force vector is not used)


@pytest.mark.parametrize("cost,value", [("signed", 0.3), ("positive", 0.5), ("abs", 0.9)])
def test_weight_charges_the_selected_form(cost, value):
    env = _env(weight=2.0, cost=cost)
    _, r, _, _, info = env.step(np.array([1.0, 1.0]))
    assert r == pytest.approx(-1.0 - 2.0 * value, rel=1e-12)


def test_reset_reports_zero_without_request():
    env = _env()
    info = MaiaFlowEnv._power_info(env, None, None)
    np.testing.assert_array_equal(info["actuation_power_per_actuator"], [0.0, 0.0])
    assert env.maiaInterface.power_requests == 0


def test_reply_of_wrong_length_is_an_error():
    env = _env(reply=REPLY[:5])  # e.g. one jet in the solver for two actions
    with pytest.raises(RuntimeError, match="returned 5 values for 2 jets"):
        env.step(np.array([1.0, 1.0]))
