import firedrake_adjoint as fda
from ufl import sin
import pytest

import hydrogym.firedrake as hgym


def test_grad():
    flow = hgym.Cavity(Re=50, mesh="medium")

    c = fda.AdjFloat(0.0)
    flow.set_control(c)

    solver = hgym.NewtonSolver(flow)
    solver.solve()

    (y,) = flow.get_observations()

    dy = fda.compute_gradient(y, fda.Control(c))

    print(dy)
    assert abs(dy) > 0