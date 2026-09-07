"""Dependency-free unit tests of `hydrogym.core`.

Exercises the shared `PDEBase` / `TransientSolver` / `FlowEnv` abstraction
directly with trivial mock implementations (no real physics), so these tests
run in a bare Python environment with only `gymnasium` + `numpy` installed --
no Firedrake, MPI, or GPU required.

(Established as Task 0.2 of the Phase 0 "safety net" in
HYDROGYM_ENGINEERING_AUDIT_v2.md.)
"""

import numpy as np
import pytest

from hydrogym import CallbackBase, FlowEnv, PDEBase, TransientSolver
from hydrogym.core import ActuatorBase

# --------------------------------------------------------------------------
# Mock implementations of the core contract
# --------------------------------------------------------------------------


class MockActuator(ActuatorBase):
    """Zeroth-order-hold actuator (the base class's `step` is abstract)."""

    def step(self, u: float, dt: float):
        self.x = u


class MockFlow(PDEBase):
    """1-D scalar "flow" with a trivial closed-form state.

    State is a single scalar `self.q`. The "solver" advances it as
    `q <- q + dt * u`, where `u` is the (scalar) control input.
    The objective is `q**2`, so reward is `-dt * q**2`.
    """

    MAX_CONTROL = np.inf
    DEFAULT_DT = 0.1
    DEFAULT_MESH = "default.mesh"

    CHECKPOINTS = {"ckpt_a": 10.0, "ckpt_b": 20.0}

    def __init__(self, initial_condition: float = 1.0, **config):
        self.initial_condition = initial_condition
        self.loaded_meshes = []
        self.bcs_initialized = 0
        self.saved_checkpoints = {}
        self.loaded_checkpoints = []
        self.render_calls = []
        super().__init__(**config)

    # -- abstract method implementations -----------------------------------

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 2

    def load_mesh(self, name: str):
        self.loaded_meshes.append(name)
        return name

    def initialize_state(self):
        self.q = self.initial_condition
        self.mesh = self.mesh  # set by load_mesh in PDEBase.__init__

    def init_bcs(self):
        self.bcs_initialized += 1

    def copy_state(self, deepcopy=True):
        return float(self.q)

    def save_checkpoint(self, filename: str):
        self.saved_checkpoints[filename] = float(self.q)

    def load_checkpoint(self, filename: str):
        self.loaded_checkpoints.append(filename)
        self.q = self.CHECKPOINTS[filename]

    def get_observations(self):
        return [self.q, self.q**2]

    def evaluate_objective(self, q=None):
        state = self.q if q is None else q
        return state**2

    def render(self, **kwargs):
        self.render_calls.append(kwargs)

    # -- overrides ----------------------------------------------------------

    def reset_controls(self):
        self.actuators = [MockActuator() for _ in range(self.num_inputs)]
        self.init_bcs()


class MockSolver(TransientSolver):
    """Advances MockFlow by `q <- q + dt * u`."""

    def step(self, iter: int, control=None, **kwargs):
        if control is None:
            u = 0.0
        else:
            arr = np.asarray(control, dtype=float).reshape(-1)
            u = float(arr[0]) if arr.size else 0.0
        self.flow.q = self.flow.q + self.dt * u
        self.flow.advance_time(self.dt, [u])
        return self.flow


class RecordingCallback(CallbackBase):
    def __init__(self, interval: int = 1):
        super().__init__(interval)
        self.calls = []
        self.closed = False

    def __call__(self, iter: int, t: float, flow: PDEBase) -> bool:
        # Model a "real" callback: it only acts when the interval check fires
        # (the solver invokes every callback every step and ignores the return)
        if super().__call__(iter, t, flow):
            self.calls.append((iter, t, float(flow.q)))
            return True
        return False

    def close(self):
        self.closed = True


def make_env(max_steps: int = 100, **actuation_config) -> FlowEnv:
    """Build a FlowEnv over the mock flow/solver with the given actuation config."""
    env_config = {
        "flow": MockFlow,
        "flow_config": {},
        "solver": MockSolver,
        "solver_config": {"dt": 0.1},
        "actuation_config": actuation_config,
        "max_steps": max_steps,
    }
    return FlowEnv(env_config)


# --------------------------------------------------------------------------
# PDEBase construction / reset / controls
# --------------------------------------------------------------------------


class TestPDEBaseConstruction:
    def test_construction_calls_load_mesh_initialize_state_reset(self):
        flow = MockFlow(mesh="foo.mesh")
        assert flow.loaded_meshes == ["foo.mesh"]
        assert flow.q == 1.0  # initialize_state applied
        assert flow.t == 0.0  # reset applied
        assert flow.bcs_initialized == 1  # reset -> reset_controls -> init_bcs

    def test_default_mesh_used_when_not_given(self):
        flow = MockFlow()
        assert flow.loaded_meshes == [MockFlow.DEFAULT_MESH]

    def test_actuators_created_per_num_inputs(self):
        flow = MockFlow()
        assert len(flow.actuators) == flow.num_inputs
        assert flow.control_state == [0.0]

    def test_restart_string_loaded_immediately(self):
        flow = MockFlow(restart="ckpt_a")
        assert flow.q == 10.0
        assert flow.loaded_checkpoints == ["ckpt_a"]

    def test_restart_list_loads_first_checkpoint(self):
        flow = MockFlow(restart=["ckpt_a", "ckpt_b"])
        assert flow.q == 10.0
        assert flow.loaded_checkpoints == ["ckpt_a"]

    def test_invalid_restart_type_raises(self):
        with pytest.raises(ValueError, match="restart must be a string or list"):
            MockFlow(restart=5)

    # -- unknown-key warning (audit Task 2.1) --------------------------------

    def test_unknown_config_key_warns(self):
        """Keys no subclass consumed reach PDEBase and must warn, not be
        silently dropped (the audit's Finding: misspelled options vanished)."""
        with pytest.warns(UserWarning, match="unrecognized_option") as record:
            MockFlow(unrecognized_option=3)
        assert any("no effect" in str(w.message) for w in record)

    def test_valid_config_keys_raise_no_warning(self):
        """All consumed keys (subclass-level, mesh, restart) must stay
        warning-free -- this must never become a hard error for valid
        Firedrake flow configs."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            flow = MockFlow(initial_condition=5.0, mesh="foo.mesh", restart="ckpt_a")
        assert flow.q == 10.0

    def test_unknown_key_does_not_break_construction(self):
        """Warning, not error: construction still succeeds."""
        with pytest.warns(UserWarning):
            flow = MockFlow(Re=100)  # e.g. a key a backend doesn't consume
        assert flow.q == 1.0

    def test_reset_sets_state_and_time(self):
        flow = MockFlow()
        flow.q = 42.0
        flow.t = 9.0
        flow.reset(q0=7.0, t=2.5)
        assert flow.q == 7.0
        assert flow.t == 2.5

    def test_reset_clears_controls(self):
        flow = MockFlow()
        flow.set_control(0.5)
        assert flow.control_state == [0.5]
        flow.reset()
        assert flow.control_state == [0.0]

    def test_reset_controls_reinitializes_actuators_and_bcs(self):
        flow = MockFlow()
        n_before = flow.bcs_initialized
        flow.reset_controls()
        assert flow.bcs_initialized == n_before + 1
        assert isinstance(flow.actuators[0], MockActuator)


class TestControls:
    def test_set_control_scalar(self):
        flow = MockFlow()
        flow.set_control(0.5)
        assert flow.control_state == [0.5]

    def test_set_control_none_resets_to_zero(self):
        flow = MockFlow()
        flow.set_control(0.5)
        flow.set_control(None)
        assert flow.control_state == [0.0]

    def test_set_control_list(self):
        flow = MockFlow()
        flow.set_control([0.25])
        assert flow.control_state == [0.25]

    def test_advance_time_updates_time_and_actuators(self):
        flow = MockFlow()
        state = flow.advance_time(0.1, [0.7])
        assert flow.t == pytest.approx(0.1)
        assert state == [0.7]
        assert flow.control_state == [0.7]

    def test_advance_time_defaults_to_current_control_state(self):
        flow = MockFlow()
        flow.set_control(0.3)
        state = flow.advance_time(0.1)
        assert state == [0.3]
        assert flow.t == pytest.approx(0.1)

    def test_base_actuator_step_is_abstract(self):
        with pytest.raises(NotImplementedError):
            ActuatorBase().step(0.1, 0.01)


# --------------------------------------------------------------------------
# TransientSolver
# --------------------------------------------------------------------------


class TestTransientSolver:
    def test_default_dt_from_flow(self):
        flow = MockFlow()
        solver = MockSolver(flow)
        assert solver.dt == MockFlow.DEFAULT_DT

    def test_solve_requires_exactly_one_mode(self):
        solver = MockSolver(MockFlow())
        with pytest.raises(ValueError, match="exactly one"):
            solver.solve()
        with pytest.raises(ValueError, match="exactly one"):
            solver.solve(t_span=(0.0, 0.2), num_steps=2)

    def test_solve_num_steps_mode(self):
        flow = MockFlow()
        solver = MockSolver(flow, dt=0.1)
        result = solver.solve(num_steps=3, controller=lambda t, y: 1.0)
        assert result is flow
        assert flow.q == pytest.approx(1.0 + 3 * 0.1)
        assert flow.t == pytest.approx(0.3)

    def test_solve_t_span_mode(self):
        flow = MockFlow()
        solver = MockSolver(flow, dt=0.1)
        result = solver.solve(t_span=(0.0, 0.3))
        assert result is flow
        # np.arange(0, 0.3, 0.1) -> 3 iterations
        assert flow.t == pytest.approx(0.3)

    def test_solve_collect_rewards(self):
        flow = MockFlow()
        solver = MockSolver(flow, dt=0.1)
        result, rewards = solver.solve(num_steps=3, collect_rewards=True)
        assert result is flow
        assert isinstance(rewards, np.ndarray)
        # No control -> q stays at 1.0 -> objective q**2 = 1.0 each step
        assert rewards.shape == (3,)
        assert np.allclose(rewards, 1.0)

    def test_solve_callbacks_called_per_step_and_closed(self):
        flow = MockFlow()
        solver = MockSolver(flow, dt=0.1)
        cb = RecordingCallback()
        solver.solve(num_steps=3, callbacks=[cb])
        assert len(cb.calls) == 3
        assert [c[0] for c in cb.calls] == [0, 1, 2]
        assert cb.closed

    def test_solve_callback_interval(self):
        flow = MockFlow()
        solver = MockSolver(flow, dt=0.1)
        cb = RecordingCallback(interval=2)
        solver.solve(num_steps=4, callbacks=[cb])
        # CallbackBase.__call__ fires on iter % interval == 0
        assert [c[0] for c in cb.calls] == [0, 2]

    def test_solver_reset_is_a_noop_hook(self):
        flow = MockFlow()
        solver = MockSolver(flow, dt=0.1)
        solver.solve(num_steps=2)
        solver.reset()  # must not raise
        assert solver.flow is flow


# --------------------------------------------------------------------------
# FlowEnv
# --------------------------------------------------------------------------


class TestFlowEnvConstruction:
    def test_construction(self):
        env = make_env()
        assert isinstance(env.flow, MockFlow)
        assert isinstance(env.solver, MockSolver)
        assert env.num_substeps == 1
        assert env.reward_aggregation == "mean"
        assert env.max_steps == 100
        assert env.iter == 0

    def test_spaces_match_flow_dimensions(self):
        env = make_env()
        assert env.observation_space.shape == (2,)  # MockFlow.num_outputs
        assert env.action_space.shape == (1,)  # MockFlow.num_inputs

    def test_num_substeps_config(self):
        assert make_env(num_substeps=3).num_substeps == 3

    def test_invalid_num_substeps_raises(self):
        with pytest.raises(ValueError, match="num_substeps must be >= 1"):
            make_env(num_substeps=0)

    def test_invalid_reward_aggregation_raises(self):
        with pytest.raises(ValueError, match="reward_aggregation must be"):
            make_env(reward_aggregation="max")


class TestFlowEnvDeprecations:
    def test_deprecated_num_sim_substeps_per_actuation(self):
        with pytest.warns(DeprecationWarning, match="num_sim_substeps_per_actuation"):
            env = make_env(num_sim_substeps_per_actuation=2)
        assert env.num_substeps == 2

    def test_deprecated_reward_aggreation_rule(self):
        with pytest.warns(DeprecationWarning, match="reward_aggreation_rule"):
            env = make_env(reward_aggreation_rule="sum")
        assert env.reward_aggregation == "sum"

    def test_current_names_raise_no_warning(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            env = make_env(num_substeps=2, reward_aggregation="sum")
        assert env.num_substeps == 2
        assert env.reward_aggregation == "sum"


class TestFlowEnvStepReset:
    def test_reset_returns_obs_and_info(self):
        env = make_env()
        obs, info = env.reset()
        assert isinstance(info, dict)
        assert obs.shape == (2,)
        assert np.allclose(obs, [1.0, 1.0])
        assert env.observation_space.contains(obs)

    def test_single_step_reward_and_obs(self):
        env = make_env()
        env.reset()
        obs, reward, terminated, truncated, info = env.step([2.0])
        # q = 1.0 + 0.1 * 2.0 = 1.2; reward = -dt * q**2 = -0.1 * 1.44
        assert np.allclose(obs, [1.2, 1.44])
        assert reward == pytest.approx(-0.1 * 1.44)
        assert terminated is False
        assert truncated is False
        assert info == {}

    def test_step_without_action_leaves_state_unchanged(self):
        env = make_env()
        env.reset()
        _, reward, *_ = env.step(None)
        assert reward == pytest.approx(-0.1 * 1.0)

    def test_truncated_at_max_steps(self):
        env = make_env(max_steps=2)
        env.reset()
        _, _, _, truncated, _ = env.step(None)
        assert truncated is False  # iter == 1
        _, _, _, truncated, _ = env.step(None)
        assert truncated is False  # iter == 2 (not > max_steps)
        _, _, _, truncated, _ = env.step(None)
        assert truncated is True  # iter == 3 > 2

    def test_terminated_is_always_false(self):
        env = make_env(max_steps=1)
        env.reset()
        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(None)
            assert terminated is False

    def test_reset_restores_initial_state_and_iter(self):
        env = make_env()
        env.reset()
        env.step([5.0])
        env.step([5.0])
        obs, _ = env.reset()
        assert np.allclose(obs, [1.0, 1.0])
        assert env.iter == 0

    def test_reset_seed_reproducibility(self):
        env = make_env()
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        assert np.allclose(obs1, obs2)


class TestFlowEnvMultiSubstep:
    """Multi-substep mode: constant controller held over `num_substeps` steps."""

    # q: 1.0 -> 1.1 -> 1.2 -> 1.3 for u=1.0, dt=0.1
    # per-substep objectives q**2: 1.21, 1.44, 1.69
    REWARDS = np.array([1.21, 1.44, 1.69])

    def _make(self, aggregation):
        return make_env(num_substeps=3, reward_aggregation=aggregation)

    def test_mean_aggregation(self):
        env = self._make("mean")
        env.reset()
        _, reward, *_ = env.step([1.0])
        assert reward == pytest.approx(-0.1 * np.mean(self.REWARDS))
        assert env.iter == 3

    def test_sum_aggregation(self):
        env = self._make("sum")
        env.reset()
        _, reward, *_ = env.step([1.0])
        assert reward == pytest.approx(-0.1 * np.sum(self.REWARDS))
        assert env.iter == 3

    def test_median_aggregation(self):
        env = self._make("median")
        env.reset()
        _, reward, *_ = env.step([1.0])
        assert reward == pytest.approx(-0.1 * np.median(self.REWARDS))
        assert env.iter == 3

    def test_state_evolution_identical_across_aggregations(self):
        # Aggregation must change only the reported reward, not the physics
        final_obs = []
        for agg in ("mean", "sum", "median"):
            env = self._make(agg)
            env.reset()
            obs, *_ = env.step([1.0])
            final_obs.append(obs)
        assert np.allclose(final_obs[0], final_obs[1])
        assert np.allclose(final_obs[1], final_obs[2])
        assert np.allclose(final_obs[0], [1.3, 1.69])

    def test_truncated_counts_substeps(self):
        env = make_env(max_steps=2, num_substeps=3)
        env.reset()
        _, _, _, truncated, _ = env.step(None)
        assert truncated is True  # iter == 3 > 2


class TestFlowEnvCheckpoints:
    def test_single_checkpoint_string(self):
        env_config = {
            "flow": MockFlow,
            "flow_config": {"restart": "ckpt_a"},
            "solver": MockSolver,
            "solver_config": {"dt": 0.1},
        }
        env = FlowEnv(env_config)
        obs, info = env.reset()
        assert np.allclose(obs, [10.0, 100.0])
        assert env.restart_checkpoints == ["ckpt_a"]
        assert len(env.initial_states) == 1

    def test_multiple_checkpoints_preloaded_and_first_selected(self):
        env_config = {
            "flow": MockFlow,
            "flow_config": {"restart": ["ckpt_a", "ckpt_b"]},
            "solver": MockSolver,
            "solver_config": {"dt": 0.1},
        }
        env = FlowEnv(env_config)
        states = [float(s) for s in env.initial_states]
        assert states == [10.0, 20.0]
        # Reset to first checkpoint state at construction
        assert env.flow.q == 10.0

    def test_reset_selects_random_checkpoint_with_index_in_info(self):
        env_config = {
            "flow": MockFlow,
            "flow_config": {"restart": ["ckpt_a", "ckpt_b"]},
            "solver": MockSolver,
            "solver_config": {"dt": 0.1},
        }
        env = FlowEnv(env_config)
        for seed in range(5):
            np.random.seed(seed)
            expected_idx = np.random.randint(0, 2)
            obs, info = env.reset(seed=seed)
            assert info["checkpoint_index"] == expected_idx
            expected_q = MockFlow.CHECKPOINTS[f"ckpt_{'ab'[expected_idx]}"]
            assert np.allclose(obs, [expected_q, expected_q**2])


class TestFlowEnvCallbacksAndClose:
    def test_callbacks_invoked_each_step(self):
        env = make_env()
        cb = RecordingCallback()
        env.set_callbacks([cb])
        env.reset()
        env.step(None)
        env.step(None)
        assert len(cb.calls) == 2
        # t = iter * solver.dt
        assert cb.calls[0][:2] == (1, pytest.approx(0.1))
        assert cb.calls[1][:2] == (2, pytest.approx(0.2))

    def test_close_calls_callback_close(self):
        env = make_env()
        cb = RecordingCallback()
        env.set_callbacks([cb])
        env.close()
        assert cb.closed

    def test_render_delegates_to_flow(self):
        env = make_env()
        env.render(mode="human", foo="bar")
        assert env.flow.render_calls == [{"mode": "human", "foo": "bar"}]
