"""Unit tests for audit Task 2.7: NekEnv's dead ``**kwargs`` now warns.

``NekEnv.__init__`` documented ``**kwargs`` as "for backward compatibility"
but never referenced it -- any extra kwarg was silently dropped, so e.g.
``NekEnv(env_config=..., WALLTIME=100)`` looked like it configured a walltime
and did nothing. The audit's safer option (keep the parameter for API
stability, warn when non-empty) is implemented here.

Same stubbing approach as test_nek_reward_aggregation.py: no MPI workers or
HF download needed; NekEnv is a plain gym.Env until wired to a solver.

pytest.importorskip guards: mpi4py/pandas via hydrogym.nek.env.
"""

import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("pandas")

from hydrogym.nek.env import NekEnv  # noqa: E402


@pytest.fixture
def stubbed_env_init():
    from unittest.mock import patch

    with patch.object(NekEnv, "_init_from_hf", lambda self, *a, **k: None), patch.object(
        NekEnv, "_init_from_legacy", lambda self, *a, **k: None
    ):
        yield


class TestDeadKwargsWarning:
    def test_unknown_kwarg_warns_and_names_it(self, stubbed_env_init):
        with pytest.warns(UserWarning, match="WALLTIME") as record:
            NekEnv(env_config={"environment_name": "x", "nproc": 1}, WALLTIME=100)
        assert any("no effect" in str(w.message) for w in record)

    def test_multiple_unknown_kwargs_all_listed(self, stubbed_env_init):
        with pytest.warns(UserWarning, match=r"\['A', 'B'\]"):
            NekEnv(env_config={"environment_name": "x", "nproc": 1}, B=2, A=1)

    def test_no_kwargs_no_warning(self, stubbed_env_init):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            NekEnv(env_config={"environment_name": "x", "nproc": 1})

    def test_supported_kwargs_still_quiet(self, stubbed_env_init):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            NekEnv(env_config={"environment_name": "x", "nproc": 1}, reward_agg="median")
