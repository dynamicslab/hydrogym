"""hydrogym.jaxfluids.envs re-exports its env classes.

Like every other backend's envs package (e.g. hydrogym.firedrake.envs
re-exports Cylinder, Cavity, Pinball, Step, RotaryCylinder), so that
`getattr(jxf_envs, "Nozzle2D")` in hydrogym/registration.py's
_jaxfluids_make resolves `gym.make("hydrogym-jaxfluids/Nozzle2D-v0")`.
"""

import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxfluids")


def test_nozzle_classes_are_reexported():
    from hydrogym.jaxfluids import envs

    assert hasattr(envs, "Nozzle2D")
    assert hasattr(envs, "Nozzle3D")
