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

Re = 7500
flow = hgym.Cavity(
    Re=100,
    mesh="fine",
    velocity_order=velocity_order,
)


def inner_product(q1, q2):
    u1 = q1.subfunctions[0]
    u2 = q2.subfunctions[0]
    return fd.assemble(inner(u1, u2) * dx)
