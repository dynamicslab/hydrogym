"""MAIA MPMD call order: step(a) must return the state at the END of the interval in which a acts.

m-AIA answers control / probe requests only at the start of a substep segment (LbSolver::callMPMDRoutines), so the
wrapper keeps m-AIA waiting there between calls: step = set control, continue (N substeps with the new action), open
the next segment, read the probes; reset = [continue], open a segment, reinit, zero control, probes (no continue);
close = [continue], finish.  The previous order (run, set, probe, continue) returned the state before the action acted.
"""

import numpy as np
import pytest

pytest.importorskip("mpi4py")

from hydrogym.maia.env_core import MaiaFlowEnv  # noqa: E402


class OrderedFakeInterface:
    def __init__(self):
        self.calls = []

    def runTimeSteps(self, n):
        self.calls.append(("run", n))

    def setControlProperties(self, props):
        self.calls.append(("control", tuple(np.atleast_1d(props).tolist())))

    def getProbeData(self, locations):
        self.calls.append(("probe",))
        return np.array([1.0, 2.0])

    def continueRun(self):
        self.calls.append(("continue",))

    def reinit(self):
        self.calls.append(("reinit",))

    def finishRun(self):
        self.calls.append(("finish",))


def _env():
    env = MaiaFlowEnv.__new__(MaiaFlowEnv)
    env.MAX_CONTROL = 1.0
    env.num_substeps_per_iteration = 7
    env.num_inputs = 1
    env.maiaInterface = OrderedFakeInterface()
    env.probe_locations = None
    env.noProbes = 1
    env.observation_type = ["u", "v"]
    env.nDim = 2
    env.bcId = []
    env.convert_action = lambda action=None: action
    env.get_reward = lambda: (0.0, {})
    env.obs_loc = np.zeros(2)
    env.obs_scale = np.ones(2)
    env.iter = 0
    env.max_episode_steps = 10
    return env


def test_reset_step_reset_close_order():
    env = _env()
    env.reset()
    env.step(np.array([0.5]))
    env.reset()
    env.close()
    assert env.maiaInterface.calls == [
        ("run", 7), ("reinit",), ("control", (0.0,)), ("probe",),                     # reset: stays in request loop
        ("control", (0.5,)), ("continue",), ("run", 7), ("probe",),                   # step: act, advance, observe
        ("continue",), ("run", 7), ("reinit",), ("control", (0.0,)), ("probe",),      # reset after a step
        ("continue",), ("finish",),                                                    # close
    ]


def test_close_without_reset_sends_finish_only():
    env = _env()
    env.close()
    assert env.maiaInterface.calls == [("finish",)]
