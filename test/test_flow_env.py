import numpy as np
import pytest

from hydrogym import FlowEnv, PDEBase, TransientSolver


class CounterFlow(PDEBase):
    """Minimal scalar PDE that counts how often observations are taken."""

    def __init__(self, **config):
        self.num_observations = 0
        super().__init__(**config)

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1

    def load_mesh(self, name):
        return None

    def initialize_state(self):
        self.q = 0.0

    def init_bcs(self):
        pass

    def copy_state(self, deepcopy=True):
        return self.q

    def save_checkpoint(self, filename):
        pass

    def load_checkpoint(self, filename):
        pass

    def get_observations(self):
        self.num_observations += 1
        return self.q

    def evaluate_objective(self, q=None):
        return self.q

    def render(self, **kwargs):
        pass


class RecordingSolver(TransientSolver):
    """Records the iteration index and control passed to each step."""

    def __init__(self, flow, dt=0.1):
        super().__init__(flow, dt)
        self.iters = []
        self.controls = []

    def step(self, iter, control=None):
        self.iters.append(iter)
        self.controls.append(control)
        self.flow.q += 1.0
        return self.flow

    def reset(self):
        self.iters = []
        self.controls = []


def make_env(num_substeps, reward_aggregation="mean"):
    return FlowEnv(
        {
            "flow": CounterFlow,
            "solver": RecordingSolver,
            "actuation_config": {
                "num_substeps": num_substeps,
                "reward_aggregation": reward_aggregation,
            },
        }
    )


@pytest.mark.parametrize("num_substeps", [1, 3])
def test_iteration_count_continues_across_steps(num_substeps):
    env = make_env(num_substeps)
    env.reset()

    for _ in range(3):
        env.step(np.array([0.5]))

    assert env.solver.iters == list(range(3 * num_substeps))
    assert env.iter == 3 * num_substeps

    # Resetting the environment restarts the iteration count
    env.reset()
    env.step(np.array([0.5]))
    assert env.solver.iters == list(range(num_substeps))


def test_substeps_hold_action_and_observe_once():
    env = make_env(num_substeps=4)
    env.reset()
    env.flow.num_observations = 0

    action = np.array([0.25])
    obs, _, _, _, _ = env.step(action)

    assert len(env.solver.controls) == 4
    assert all(np.array_equal(u, action) for u in env.solver.controls)
    assert env.flow.num_observations == 1
    np.testing.assert_allclose(obs, [4.0])


@pytest.mark.parametrize(
    "reward_aggregation, expected",
    [("mean", 2.5), ("sum", 10.0), ("median", 2.5)],
)
def test_substep_reward_aggregation(reward_aggregation, expected):
    env = make_env(num_substeps=4, reward_aggregation=reward_aggregation)
    env.reset()

    _, reward, _, _, _ = env.step(np.array([0.0]))

    # Objectives after each substep are 1, 2, 3, 4
    assert reward == pytest.approx(-env.solver.dt * expected)
