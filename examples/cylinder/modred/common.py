import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

import hydrogym as gym

output_dir = 'output'
pvd_out = f"{output_dir}/solution.pvd"
restart = f"../restart.h5"
snap_file = f'{output_dir}/snapshots.h5'
save_prefix=f'{output_dir}/snapshots/'
pod_file = f'{output_dir}/pod.h5'
pod_prefix = f'{output_dir}/pod'

flow = gym.flow.Cylinder(Re=100, h5_file=restart)