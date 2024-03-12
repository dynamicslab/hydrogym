import os

import firedrake as fd

import hydrogym.firedrake as hgym

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

base_checkpoint = "output/base.h5"

velocity_order = 2
stabilization = "none"

flow = hgym.Cylinder(
    Re=100,
    mesh="medium",
    velocity_order=velocity_order,
)
