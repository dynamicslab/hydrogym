"""Unit tests for audit Task 2.12: substep-count naming on MAIA and JAX.

`num_substeps` must be accepted as the primary name in the environments'
YAML config sections (matching Firedrake's non-deprecated name and
core.py's actuation_config), with the old
`num_sim_substeps_per_actuation` still working but emitting a
DeprecationWarning -- mirroring core.py:388-396's existing pattern.

Each backend module is importorskip-guarded independently: the MAIA module
pulls mpi4py/einops/gymnasium, the JAX module pulls jax, so a container
with only some backends installed runs the subset it can.

The helpers are static methods precisely so these tests can exercise the
resolution logic without constructing a full environment (MaiaFlowEnv
needs a live m-AIA solver over MPI; JAXFlowEnv is unconstructable -- see
the caveat in test_jax_env_core.py).
"""

import omegaconf
import pytest

try:
    from hydrogym.maia.env_core import ConfigError as MaiaConfigError
    from hydrogym.maia.env_core import MaiaFlowEnv as _MaiaFlowEnv

    _resolve_maia = _MaiaFlowEnv._resolve_num_substeps
    HAS_MAIA = True
except ImportError:
    HAS_MAIA = False

try:
    from hydrogym.jax.env_core import ConfigError as JaxConfigError
    from hydrogym.jax.env_core import JAXFlowEnv as _JAXFlowEnv

    _resolve_jax = _JAXFlowEnv._resolve_num_substeps
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

PARAMS = []
if HAS_MAIA:
    PARAMS.append(pytest.param(_resolve_maia, MaiaConfigError, "maia", id="maia"))
if HAS_JAX:
    PARAMS.append(pytest.param(_resolve_jax, JaxConfigError, "jax", id="jax"))


@pytest.mark.parametrize("resolve,config_error,backend", PARAMS)
class TestSubstepNaming:
    def test_new_name_works_without_warning(self, resolve, config_error, backend):
        import warnings

        cfg = omegaconf.OmegaConf.create({"num_substeps": 7})
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = resolve(cfg)
        assert result == 7

    def test_old_name_still_works_and_warns(self, resolve, config_error, backend):
        cfg = omegaconf.OmegaConf.create({"num_sim_substeps_per_actuation": 87})
        with pytest.warns(DeprecationWarning, match="num_substeps"):
            result = resolve(cfg)
        assert result == 87

    def test_both_names_produce_identical_state(self, resolve, config_error, backend):
        from_new = resolve(omegaconf.OmegaConf.create({"num_substeps": 42}))
        with pytest.warns(DeprecationWarning):
            from_old = resolve(omegaconf.OmegaConf.create({"num_sim_substeps_per_actuation": 42}))
        assert from_new == from_old

    def test_missing_keys_raise_config_error(self, resolve, config_error, backend):
        with pytest.raises(config_error, match="num_substeps"):
            resolve(omegaconf.OmegaConf.create({"Re": 200}))
