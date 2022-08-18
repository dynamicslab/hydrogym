import os

from firedrake import Mesh

mesh_dir = os.path.abspath(f"{__file__}/..")

FLUID = 1
INLET = 2
FREESTREAM = 3
OUTLET = 4
CYLINDER = (5, 6, 7)

rad = 0.5
x0 = [0.0, rad * 1.5 * 1.866, rad * 1.5 * 1.866]
y0 = [0.0, 0.5 * rad, -0.5 * rad]


def load_mesh(name="coarse"):
    return Mesh(f"{mesh_dir}/{name}.msh", name="mesh")
