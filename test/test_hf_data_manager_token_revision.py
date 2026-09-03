"""Unit tests for audit Task 2.6: `token`/`revision` on HFDataManager.

Previously there was no HydroGym API surface for private/gated HF repos or
for pinning a revision -- only ambient auth (HF_TOKEN / huggingface-cli
login) and the repo's default branch worked. Now:

- HFDataManager.__init__ takes optional token=/revision= and threads them
  into every huggingface_hub call (3x snapshot_download, 2x list_repo_files).
- Every backend that constructs its own HFDataManager exposes
  hf_token/hf_revision env_config keys (maia/jax/jaxfluids/nek env_core,
  maia workspace, jax channel initial-field loader) or flow_config keys
  (firedrake checkpoint resolver) and forwards them.

Omitting the new kwargs must preserve ambient-auth behavior exactly:
huggingface_hub treats token=None as "use ambient credentials", so
forwarding None is byte-identical to not forwarding.

pytest.importorskip guards: huggingface_hub (data_manager), mpi4py+pandas
(NekEnv behavioral test), firedrake / backend modules (heavier imports).
"""

import inspect

import pytest

pytest.importorskip("huggingface_hub")

from hydrogym import core_external as core_external_mod  # noqa: E402
from hydrogym import data_manager as dm_mod  # noqa: E402
from hydrogym.data_manager import HFDataManager  # noqa: E402

ENV_NAME = "TestEnv"


@pytest.fixture
def recording_hf(monkeypatch, tmp_path):
    """Mock both HF entry points used by data_manager; record their kwargs.

    snapshot_download materializes a minimal JAXFLUIDS-profile env (sentinel
    only, no required files) under the returned snapshot dir; list_repo_files
    is recorded and returns just the sentinel path so profile detection works
    without network.
    """
    snapshot_calls = []
    api_calls = []

    snapshot_dir = tmp_path / "hf" / "datasets--t--r" / "snapshots" / "abc123"
    env_dir = snapshot_dir / ENV_NAME
    env_dir.mkdir(parents=True)
    (env_dir / ".JAXFLUIDS").write_text("")

    def fake_snapshot_download(**kwargs):
        snapshot_calls.append(kwargs)
        return str(snapshot_dir)

    class FakeApi:
        def __init__(self, **kwargs):
            api_calls.append(("HfApi", kwargs))

        def list_repo_files(self, repo_id, repo_type=None, revision=None):
            api_calls.append(("list_repo_files", {"repo_id": repo_id, "repo_type": repo_type, "revision": revision}))
            return [f"{ENV_NAME}/.JAXFLUIDS"]

    monkeypatch.setattr(dm_mod, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(dm_mod, "HfApi", FakeApi)
    return snapshot_calls, api_calls


class TestDataManagerThreading:
    def test_snapshot_download_receives_token_and_revision(self, recording_hf, tmp_path):
        snapshot_calls, _ = recording_hf
        dm = HFDataManager(
            repo_id="t/r",
            cache_dir=str(tmp_path / "clean"),
            use_clean_cache=False,
            fallback_profile="JAXFLUIDS",
            token="hf_tok_123",
            revision="v1.2.3",
        )
        dm.get_environment_path(ENV_NAME, force_download=True)

        assert len(snapshot_calls) == 1
        assert snapshot_calls[0]["token"] == "hf_tok_123"
        assert snapshot_calls[0]["revision"] == "v1.2.3"

    @pytest.mark.parametrize("use_clean_cache", [True, "copy", False])
    def test_all_three_download_strategies_thread(self, recording_hf, tmp_path, use_clean_cache):
        snapshot_calls, _ = recording_hf
        dm = HFDataManager(
            repo_id="t/r",
            cache_dir=str(tmp_path / "clean"),
            use_clean_cache=use_clean_cache,
            fallback_profile="JAXFLUIDS",
            token="tok",
            revision="rev",
        )
        dm.get_environment_path(ENV_NAME, force_download=True)
        assert snapshot_calls[0]["token"] == "tok"
        assert snapshot_calls[0]["revision"] == "rev"

    def test_list_repo_files_receives_token_and_revision(self, recording_hf):
        """Profile detection goes through HfApi(token=...).list_repo_files(
        repo_id, repo_type, revision=...)."""
        _, api_calls = recording_hf
        dm = HFDataManager(
            repo_id="t/r", use_clean_cache=False, fallback_profile="JAXFLUIDS", token="tok", revision="rev"
        )
        profile = dm._detect_solver_profile(ENV_NAME)
        assert profile == "JAXFLUIDS"

        api_init = [k for name, k in api_calls if name == "HfApi"]
        assert api_init and api_init[0].get("token") == "tok"
        listings = [k for name, k in api_calls if name == "list_repo_files"]
        assert listings and listings[0]["revision"] == "rev"

    def test_get_available_environments_receives_token_and_revision(self, recording_hf):
        _, api_calls = recording_hf
        dm = HFDataManager(
            repo_id="t/r", use_clean_cache=False, fallback_profile="JAXFLUIDS", token="tok", revision="rev"
        )
        dm.get_available_environments()

        listings = [k for name, k in api_calls if name == "list_repo_files"]
        assert listings and listings[0]["revision"] == "rev"
        api_init = [k for name, k in api_calls if name == "HfApi"]
        assert api_init[0].get("token") == "tok"

    def test_omitted_preserves_ambient_behavior(self, recording_hf, tmp_path):
        """token=None/revision=None must be forwarded (huggingface_hub treats
        None as ambient auth) -- i.e. exactly the pre-change effective call."""
        snapshot_calls, api_calls = recording_hf
        dm = HFDataManager(
            repo_id="t/r", cache_dir=str(tmp_path / "c"), use_clean_cache=False, fallback_profile="JAXFLUIDS"
        )
        dm.get_environment_path(ENV_NAME, force_download=True)
        assert snapshot_calls[0]["token"] is None
        assert snapshot_calls[0]["revision"] is None

        dm._detect_solver_profile(ENV_NAME)
        api_init = [k for name, k in api_calls if name == "HfApi"]
        assert api_init[0].get("token") is None
        listings = [k for name, k in api_calls if name == "list_repo_files"]
        assert listings[0]["revision"] is None


class TestNekEnvConfigThreading:
    """Behavioral check for one backend's env_config -> HFDataManager path
    (Nek's _init_from_hf, stubbed the same way as test_nek_reward_aggregation).
    The remaining backends share the identical constructor pattern and are
    covered by the source-drift guard below."""

    @pytest.fixture
    def nek_env_ctx(self, monkeypatch, tmp_path):
        pytest.importorskip("mpi4py")
        pytest.importorskip("pandas")
        from hydrogym.nek import env as nek_env_mod

        recorded = {}

        class RecordingDataManager:
            def __init__(self, *args, **kwargs):
                recorded.update(kwargs)

        (tmp_path / "config.yaml").write_text("env: {}\n")
        monkeypatch.setattr(nek_env_mod, "HFDataManager", RecordingDataManager)
        monkeypatch.setattr(nek_env_mod.NekEnv, "_setup_environment_data", lambda self: str(tmp_path))
        monkeypatch.setattr(
            nek_env_mod.NekEnv, "_resolve_configuration_file", lambda self, x: str(tmp_path / "config.yaml")
        )
        monkeypatch.setattr(nek_env_mod.NekEnv, "_update_configuration_paths", lambda self: None)
        monkeypatch.setattr(nek_env_mod.NekEnv, "_apply_runtime_overrides", lambda self, ec: None)
        monkeypatch.setattr(nek_env_mod.NekEnv, "_create_session_file_early", lambda self: None)
        monkeypatch.setattr(core_external_mod, "mpi_split", lambda comm, nproc=None, **kw: None)
        monkeypatch.setattr(nek_env_mod.NekEnv, "_initialize", lambda self: None)
        monkeypatch.chdir(tmp_path)
        return {"recorded": recorded, "NekEnv": nek_env_mod.NekEnv}

    def test_hf_token_and_revision_forwarded(self, nek_env_ctx):
        env = nek_env_ctx["NekEnv"](
            env_config={"environment_name": "x", "nproc": 1, "hf_token": "tok", "hf_revision": "rev"}
        )
        assert env.hf_token == "tok"
        assert env.hf_revision == "rev"
        assert nek_env_ctx["recorded"]["token"] == "tok"
        assert nek_env_ctx["recorded"]["revision"] == "rev"

    def test_omitted_keeps_defaults(self, nek_env_ctx):
        env = nek_env_ctx["NekEnv"](env_config={"environment_name": "x", "nproc": 1})
        assert env.hf_token is None
        assert env.hf_revision is None
        assert nek_env_ctx["recorded"]["token"] is None
        assert nek_env_ctx["recorded"]["revision"] is None


class TestBackendThreadingDriftGuard:
    """Source-level guard: every backend env_core that builds its own
    HFDataManager must (a) read hf_token/hf_revision from env_config and
    (b) forward token=/revision= into the HFDataManager call. Behavioral
    coverage exists for the data_manager layer and for NekEnv; this pins the
    copy-paste family so a future new backend (or an accidental revert) can't
    silently drop the keys. Modules whose heavy deps (jax, jaxfluids, mpi4py)
    are absent simply skip."""

    @pytest.mark.parametrize(
        "module",
        ["hydrogym.maia.env_core", "hydrogym.jax.env_core", "hydrogym.jaxfluids.env_core"],
    )
    def test_env_core_forwards_token_revision(self, module):
        pytest.importorskip(module)
        src = inspect.getsource(__import__(module, fromlist=["*"]))
        assert 'env_config.get("hf_token"' in src, f"{module} does not read hf_token from env_config"
        assert 'env_config.get("hf_revision"' in src, f"{module} does not read hf_revision from env_config"
        assert "token=self.hf_token" in src, f"{module} does not forward token to HFDataManager"
        assert "revision=self.hf_revision" in src, f"{module} does not forward revision to HFDataManager"


class TestFiredrakeCheckpointResolver:
    """flow_config hf_token/hf_revision must reach the checkpoint resolver's
    HFDataManager. Called on a bare instance (no firedrake flow construction --
    that path is broken at baseline) since _resolve_single_checkpoint only
    uses its arguments."""

    @staticmethod
    def _bare_flow_cls():
        """Minimal concrete FlowConfig: object.__new__ refuses abstract
        classes, but the resolver methods only touch their arguments."""
        from hydrogym.firedrake.flow import FlowConfig

        class _StubFlow(FlowConfig):
            def evaluate_objective(self):
                pass

            def init_bcs(self):
                pass

            def num_inputs(self):
                return 0

            def render(self):
                pass

        return _StubFlow

    @staticmethod
    def _install_recorder(monkeypatch, tmp_path, recorded):
        class RecordingDataManager:
            def __init__(self, *args, **kwargs):
                recorded.update(kwargs)

            def get_environment_path(self, name, force_download=False):
                d = tmp_path / "env"
                d.mkdir(exist_ok=True)
                (d / "checkpoint0.h5").write_text("")  # happy path: resolver finds a ckpt
                return str(d)

        monkeypatch.setattr(dm_mod, "HFDataManager", RecordingDataManager)

    def test_flow_config_keys_reach_hf_data_manager(self, monkeypatch, tmp_path):
        pytest.importorskip("firedrake")
        from hydrogym.firedrake.flow import FlowConfig

        recorded = {}
        self._install_recorder(monkeypatch, tmp_path, recorded)
        flow = object.__new__(self._bare_flow_cls())
        flow._resolve_single_checkpoint("SomeEnv", cache_dir=None, local_dir=None, hf_token="tok", hf_revision="rev")
        assert recorded["token"] == "tok"
        assert recorded["revision"] == "rev"

    def test_omitted_keeps_ambient_behavior(self, monkeypatch, tmp_path):
        pytest.importorskip("firedrake")
        from hydrogym.firedrake.flow import FlowConfig

        recorded = {}
        self._install_recorder(monkeypatch, tmp_path, recorded)
        flow = object.__new__(self._bare_flow_cls())
        flow._resolve_single_checkpoint("SomeEnv")
        assert recorded["token"] is None
        assert recorded["revision"] is None
