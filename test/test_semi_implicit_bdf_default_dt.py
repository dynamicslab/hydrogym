"""SemiImplicitBDF's dt is optional.

Like its parent classes (TransientSolver, NavierStokesTransientSolver),
SemiImplicitBDF defaults dt to flow.DEFAULT_DT if omitted, so
`gym.make("hydrogym/Cylinder-v0")` (and every other registered Firedrake ID)
and env_config dicts without a dt work.

A signature check (not a full solve) is enough: SemiImplicitBDF
never uses `dt` before forwarding it to `super().__init__(flow, dt, ...)`,
which resolves `None`, so there's no solver-numerical behavior at stake
here, only the constructor's default.
"""

import inspect

import pytest

pytest.importorskip("firedrake")

from hydrogym.firedrake.solvers.bdf_ext import SemiImplicitBDF  # noqa: E402


def test_dt_parameter_is_optional():
    sig = inspect.signature(SemiImplicitBDF.__init__)
    assert sig.parameters["dt"].default is None, (
        "SemiImplicitBDF's dt must default to None (resolved to flow.DEFAULT_DT by "
        "the parent TransientSolver/NavierStokesTransientSolver classes) -- making it "
        "a required positional argument breaks every caller that omits dt, including "
        "hydrogym.registration's gym.make() factories."
    )
