"""Regression test: hydrogym.jaxfluids.envs must re-export its env classes.

Found via `gym.make("hydrogym-jaxfluids/Nozzle2D-v0")` failing with
`AttributeError: module 'hydrogym.jaxfluids.envs' has no attribute
'Nozzle2D'` -- the package's __init__.py was empty, unlike every other
backend's envs package (e.g. hydrogym.firedrake.envs re-exports Cylinder,
Cavity, Pinball, Step, RotaryCylinder), so `getattr(jxf_envs,
"Nozzle2D")` in hydrogym/registration.py's _jaxfluids_make had nothing to
find even though `hydrogym.jaxfluids.envs.nozzle.Nozzle2D` existed all
along.
"""

import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxfluids")


def test_nozzle_classes_are_reexported():
    from hydrogym.jaxfluids import envs

    assert hasattr(envs, "Nozzle2D")
    assert hasattr(envs, "Nozzle3D")
