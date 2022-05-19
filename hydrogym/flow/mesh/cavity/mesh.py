from firedrake import Mesh

import os
mesh_dir = os.path.abspath(f'{__file__}/..')

FLUID = 1
INLET = 2
FREESTREAM = 3
OUTLET = 4
SLIP = 5
WALL = 6
CONTROL = 7

def load_mesh(name='fine'):
    return Mesh(f'{mesh_dir}/{name}.msh', name='mesh')