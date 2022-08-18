import os

from firedrake import Mesh

mesh_dir = os.path.abspath(f"{__file__}/..")

FLUID = 1
INLET = 2
FREESTREAM = 3
OUTLET = 4
WALL = 5
CONTROL = 6


def load_mesh(name="step"):
    return Mesh(f"{mesh_dir}/{name}.msh", name="mesh")
