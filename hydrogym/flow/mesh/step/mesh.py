import os

from firedrake import Mesh

mesh_dir = os.path.abspath(f"{__file__}/..")

FLUID = 1
INLET = 2
OUTLET = 3
WALL = 4
CONTROL = 5
SENSOR = 6


def load_mesh(name="fine"):
    return Mesh(f"{mesh_dir}/{name}.msh", name="mesh")
