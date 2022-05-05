from firedrake import Mesh

import os
mesh_dir = os.path.abspath(f'{__file__}/..')

FLUID = 1
INLET = 2
FREESTREAM = 3
OUTLET = 4
WALL = 5

def load_mesh(name='step'):
    return Mesh(f'{mesh_dir}/{name}.msh', name='mesh')