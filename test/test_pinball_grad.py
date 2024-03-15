import firedrake_adjoint as fda
import numpy as np
import pyadjoint

import hydrogym.firedrake as hgym


def test_steady_grad():
    flow = hgym.Pinball(Re=30, mesh="coarse")
    n_cyl = len(flow.CYLINDER)

    # Option 1: List of AdjFloats
    # omega = [fda.AdjFloat(0.5) for _ in range(n_cyl)]
    # control = [fda.Control(omg) for omg in omega]

    # Option 2: List of Constants
    # omega = [fd.Constant(0.5) for i in range(n_cyl)]
    # control = [fda.Control(omg) for omg in omega]

    # # Option 3: Overloaded array with numpy_adjoint
    omega = pyadjoint.create_overloaded_object(np.zeros(n_cyl))
    control = fda.Control(omega)

    flow.set_control(omega)

    solver = hgym.NewtonSolver(flow)
    solver.solve()

    J = flow.evaluate_objective()

    # fda.compute_gradient(sum(CD), control)
    dJ = fda.compute_gradient(J, control)
    assert np.all(np.abs(dJ) > 0)