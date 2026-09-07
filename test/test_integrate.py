"""Unit tests for audit Task 2.4: `integrate()` forwards `collect_rewards`.

`hydrogym.firedrake.solvers.integrate.integrate()` previously had no way to
request reward collection even though the underlying
`core.TransientSolver.solve` (which both METHODS entries inherit -- neither
SemiImplicitBDF nor LinearizedBDF overrides `solve`) already supported
`collect_rewards`. These tests pin the forwarding.

The tests intentionally do NOT construct real Firedrake flows: flow
construction currently fails in this repo's pinned Firedrake environment
(dual_evaluation version skew, see test/README.md and the Task 0.3
baseline). Instead, a fake solver is registered into the module-level
METHODS dict, which is the extension point `integrate()` itself dispatches
through -- the forwarding logic under test needs nothing more.

pytest.importorskip guards: hydrogym.firedrake pulls in firedrake.
"""

import importlib
import inspect

import numpy as np
import pytest

pytest.importorskip("firedrake")

from hydrogym import core  # noqa: E402
from hydrogym.firedrake.solvers import bdf_ext  # noqa: E402
from hydrogym.firedrake.solvers.bdf_ext import LinearizedBDF, SemiImplicitBDF  # noqa: E402

# The `solvers` package re-exports the `integrate` FUNCTION under the same
# name as its submodule, so `from ... import integrate` would bind the
# function -- import the module explicitly.
integrate_mod = importlib.import_module("hydrogym.firedrake.solvers.integrate")  # noqa: E402
integrate_fn = integrate_mod.integrate  # noqa: E402


class FakeFlow:
    """Minimal stand-in for a PDEBase: enough for core.TransientSolver.solve."""

    DEFAULT_DT = 0.1

    def __init__(self):
        self.t = 0.0

    def evaluate_objective(self):
        return self.t


class FakeSolver(core.TransientSolver):
    def step(self, iter, control=None, **kwargs):
        self.flow.t += self.dt
        return self.flow


@pytest.fixture
def fake_method(monkeypatch):
    monkeypatch.setitem(integrate_mod.METHODS, "FakeBDF", FakeSolver)
    return FakeFlow()


def n_steps(t_span, dt):
    return len(np.arange(*t_span, dt))


class TestCollectRewardsForwarding:
    def test_default_returns_flow_only(self, fake_method):
        result = integrate_fn(fake_method, t_span=(0.0, 1.0), dt=0.25, method="FakeBDF")
        assert result is fake_method  # not a (flow, rewards) tuple

    def test_collect_rewards_returns_tuple(self, fake_method):
        flow, rewards = integrate_fn(fake_method, t_span=(0.0, 1.0), dt=0.25, collect_rewards=True, method="FakeBDF")
        assert flow is fake_method
        assert isinstance(rewards, np.ndarray)
        assert rewards.shape == (n_steps((0.0, 1.0), 0.25),)
        # core.solve evaluates the objective AFTER each step
        np.testing.assert_allclose(rewards, [0.25, 0.5, 0.75, 1.0])

    def test_rewards_match_num_steps(self, fake_method):
        t_span = (0.0, 0.5)
        _, rewards = integrate_fn(fake_method, t_span=t_span, dt=0.1, collect_rewards=True, method="FakeBDF")
        assert len(rewards) == n_steps(t_span, 0.1)


class TestMethodValidation:
    def test_invalid_method_still_rejected(self, fake_method):
        with pytest.raises(ValueError, match="must be one of"):
            integrate_fn(fake_method, t_span=(0.0, 1.0), dt=0.25, method="RK45")


class TestRealMethodsSupportKwarg:
    """Guard against drift: the real METHODS entries must keep accepting
    collect_rewards (they inherit solve from core.TransientSolver; if either
    class grows its own solve() without the kwarg, integrate()'s forwarding
    would break at runtime)."""

    @pytest.mark.parametrize("solver_cls", [SemiImplicitBDF, LinearizedBDF])
    def test_solve_signature_has_collect_rewards(self, solver_cls):
        assert "collect_rewards" in inspect.signature(solver_cls.solve).parameters


class TestNoiseKwargsRemoved:
    """Task 2.3: NavierStokesTransientSolver's eta/max_noise_iter/noise_cutoff
    accepted a white-noise body forcing whose implementation was removed
    upstream (a067781, "Clean up old forcing code", 2024-03); the kwargs were
    silently ignored ever since -- even by test_step.py, which passed eta=1.0
    expecting forcing that never happened. They are removed; this pins the
    removal so they can't silently reappear as dead parameters."""

    @pytest.mark.parametrize("kwarg", ["eta", "max_noise_iter", "noise_cutoff"])
    @pytest.mark.parametrize("solver_cls", [SemiImplicitBDF, LinearizedBDF])
    def test_dead_noise_kwargs_gone(self, solver_cls, kwarg):
        assert kwarg not in inspect.signature(solver_cls.__init__).parameters
