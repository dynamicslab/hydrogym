"""Unit tests for audit Task 3.1: `HFEnvConfigMixin`.

The four HF-backed backends (jaxfluids, jax, maia, nek) each carried a
private copy of `_setup_environment_data` / `_resolve_configuration_file` /
`_find_configuration_file`, differing only in the ~/.cache namespace. The
mixin centralizes them; Task 3.1 migrates JAX-Fluids onto it first.

The mixin tests below run on a trivial host class with a stubbed
`data_manager` -- no JAX-Fluids install, no HF download. The JAX-Fluids
migration is pinned by a source/subclass test (guarded by importorskip so
it skips wherever jaxfluids isn't installed).
"""

import pytest

pytest.importorskip("huggingface_hub")

from hydrogym.hf_env_mixin import ConfigError, HFEnvConfigMixin  # noqa: E402


class _Host(HFEnvConfigMixin):
    """Trivial consumer of the mixin: just the attributes the mixin needs."""

    HF_CACHE_NAMESPACE = "testgym"

    def __init__(self, environment_name, env_data_path, data_manager):
        self.environment_name = environment_name
        self.env_data_path = env_data_path
        self.data_manager = data_manager


class _StubDataManager:
    def __init__(self, env_path=None, error=None):
        self.env_path = env_path
        self.error = error
        self.calls = []

    def get_environment_path(self, environment_name):
        self.calls.append(environment_name)
        if self.error is not None:
            raise self.error
        return self.env_path


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    """Point `Path.home()` inside the mixin module at a tmp dir."""
    import pathlib

    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: tmp_path))
    return tmp_path


class TestSetupEnvironmentData:
    def test_cache_hit_short_circuits_data_manager(self, fake_home):
        cache = fake_home / ".cache" / "testgym" / "MyEnv"
        cache.mkdir(parents=True)
        dm = _StubDataManager(env_path="/should/not/be/used")

        host = _Host("MyEnv", None, dm)
        assert host._setup_environment_data() == str(cache)
        assert dm.calls == []  # never consulted

    def test_cache_miss_falls_back_to_data_manager(self, fake_home):
        dm = _StubDataManager(env_path="/hf/snapshot/MyEnv")
        host = _Host("MyEnv", None, dm)
        assert host._setup_environment_data() == "/hf/snapshot/MyEnv"
        assert dm.calls == ["MyEnv"]

    def test_data_manager_failure_wraps_in_config_error(self, fake_home):
        dm = _StubDataManager(error=RuntimeError("hub down"))
        host = _Host("MyEnv", None, dm)
        with pytest.raises(ConfigError, match="Failed to setup environment data for MyEnv"):
            host._setup_environment_data()

    def test_namespace_is_parameterized(self, fake_home):
        class _OtherHost(_Host):
            HF_CACHE_NAMESPACE = "othergym"

        cache = fake_home / ".cache" / "othergym" / "MyEnv"
        cache.mkdir(parents=True)
        dm = _StubDataManager(env_path="/hf/MyEnv")

        # _OtherHost hits its own namespace; _Host with the same tree misses
        assert _OtherHost("MyEnv", None, dm)._setup_environment_data() == str(cache)
        assert _Host("MyEnv", None, dm)._setup_environment_data() == "/hf/MyEnv"


class TestResolveConfigurationFile:
    def _make(self, tmp_path, env_files=()):
        env_dir = tmp_path / "env"
        env_dir.mkdir(exist_ok=True)
        for f in env_files:
            (env_dir / f).write_text("env: {}")
        return _Host("MyEnv", str(env_dir), _StubDataManager())

    def test_none_triggers_autodetect(self, tmp_path):
        host = self._make(tmp_path, env_files=["config.yaml"])
        assert host._resolve_configuration_file(None) == str(tmp_path / "env" / "config.yaml")

    def test_absolute_path_exists(self, tmp_path):
        cfg = tmp_path / "elsewhere.yaml"
        cfg.write_text("env: {}")
        assert self._make(tmp_path)._resolve_configuration_file(str(cfg)) == str(cfg)

    def test_absolute_path_missing_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="Configuration file not found"):
            self._make(tmp_path)._resolve_configuration_file(str(tmp_path / "nope.yaml"))

    def test_dot_relative_path(self, tmp_path, monkeypatch):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("env: {}")
        monkeypatch.chdir(tmp_path)
        assert self._make(tmp_path)._resolve_configuration_file("./cfg.yaml") == str(cfg)

    def test_dot_relative_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ConfigError, match="Configuration file not found"):
            self._make(tmp_path)._resolve_configuration_file("./nope.yaml")

    def test_bare_filename_in_cwd(self, tmp_path, monkeypatch):
        (tmp_path / "cfg.yaml").write_text("env: {}")
        monkeypatch.chdir(tmp_path)
        assert self._make(tmp_path)._resolve_configuration_file("cfg.yaml") == str(tmp_path / "cfg.yaml")

    def test_bare_filename_in_env_dir(self, tmp_path):
        host = self._make(tmp_path, env_files=["custom.yaml"])
        assert host._resolve_configuration_file("custom.yaml") == str(tmp_path / "env" / "custom.yaml")

    def test_bare_filename_nowhere_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ConfigError, match="not found in"):
            self._make(tmp_path)._resolve_configuration_file("custom.yaml")


class TestFindConfigurationFile:
    def _make(self, tmp_path, env_files=()):
        env_dir = tmp_path / "env"
        env_dir.mkdir(exist_ok=True)
        for f in env_files:
            (env_dir / f).write_text("env: {}")
        return _Host("MyEnv", str(env_dir), _StubDataManager()), env_dir

    def test_exact_name_priority(self, tmp_path):
        host, env_dir = self._make(tmp_path, env_files=["environment.yaml", "config.yaml"])
        # config.yaml is first in the priority list
        assert host._find_configuration_file() == str(env_dir / "config.yaml")

    def test_environment_name_yaml_fallback(self, tmp_path):
        host, env_dir = self._make(tmp_path, env_files=["MyEnv.yaml"])
        assert host._find_configuration_file() == str(env_dir / "MyEnv.yaml")

    def test_pattern_glob(self, tmp_path):
        host, env_dir = self._make(tmp_path, env_files=["config_timestep10.yaml"])
        assert host._find_configuration_file() == str(env_dir / "config_timestep10.yaml")

    def test_nothing_found_returns_none(self, tmp_path):
        host, _ = self._make(tmp_path, env_files=["readme.md"])
        assert host._find_configuration_file() is None


class TestJaxfluidsMigration:
    """Pin the Task 3.1 migration: JAXFluidsFlowEnv uses the mixin's methods
    (its local copies are deleted), and its namespace/profile survive."""

    def test_uses_mixin(self):
        mod = pytest.importorskip("hydrogym.jaxfluids.env_core")
        assert issubclass(mod.JAXFluidsFlowEnv, HFEnvConfigMixin)
        # The three methods must be inherited, not overridden locally
        for name in ("_setup_environment_data", "_resolve_configuration_file", "_find_configuration_file"):
            assert name not in vars(mod.JAXFluidsFlowEnv), f"{name} still overridden locally"
        assert mod.JAXFluidsFlowEnv.HF_CACHE_NAMESPACE == "jaxfluidsgym"
        assert mod.JAXFluidsFlowEnv.SOLVER_TYPE == "JAXFLUIDS"

    def test_config_error_is_shared_type(self):
        mod = pytest.importorskip("hydrogym.jaxfluids.env_core")
        # Same exception object: existing `except ConfigError` sites keep working
        assert mod.ConfigError is ConfigError
