"""Regression test: SemiImplicitBDF's dt must be optional.

Found via `gym.make("hydrogym/Cylinder-v0")` (and every other registered
Firedrake ID) failing unconditionally with `TypeError: SemiImplicitBDF.
__init__() missing 1 required positional argument: 'dt'` -- the parent
classes (TransientSolver, NavierStokesTransientSolver) both document and
implement "dt defaults to flow.DEFAULT_DT if omitted", but SemiImplicitBDF
required it positionally, breaking that inherited contract for every
caller that didn't pass dt explicitly (this silently affected
hydrogym/registration.py's Firedrake factories and at least two examples'
own env_config dicts -- see the Verification Addendum for the full list).

A signature check (not a full solve) is enough to pin this: SemiImplicitBDF
never uses `dt` before forwarding it to `super().__init__(flow, dt, ...)`,
which already resolves `None` correctly, so there's no solver-numerical
behavior at stake here, only the constructor's default.
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
