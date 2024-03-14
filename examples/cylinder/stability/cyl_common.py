import os

import firedrake as fd
from ufl import dx, inner

import hydrogym.firedrake as hgym

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

base_checkpoint = "output/base.h5"
evec_checkpoint = "output/evecs.h5"

velocity_order = 2
stabilization = "none"

flow = hgym.Cylinder(
    Re=100,
    mesh="medium",
    velocity_order=velocity_order,
)


class FlowPropagator:
    def __init__(self, flow, qB, dt, tau):
        self.flow = flow
        self.tau = tau
        self.solver = hgym.LinearizedBDF(
            flow,
            dt,
            qB=qB,
            stabilization=stabilization,
        )

    def __matmul__(self, q):
        self.flow.q.assign(q)

        # Assign the current solution to all `u_prev`
        # TODO: Move to solver reset function?
        for u in self.solver.u_prev:
            u.assign(q.subfunctions[0])

        self.solver.solve((0.0, self.tau))
        return self.flow.q.copy(deepcopy=True)  # TODO: do we need deepcopy?


def inner_product(q1, q2):
    u1 = q1.subfunctions[0]
    u2 = q2.subfunctions[0]
    return fd.assemble(inner(u1, u2) * dx)
