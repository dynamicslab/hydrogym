import os

from firedrake import Mesh

mesh_dir = os.path.abspath(f"{__file__}/..")

FLUID = 1
INLET = 2
FREESTREAM = 3
OUTLET = 4
SLIP = 5
WALL = (6, 8)
CONTROL = 7
SENSOR = 8


def load_mesh(name="fine"):
    return Mesh(f"{mesh_dir}/{name}.msh", name="mesh")
