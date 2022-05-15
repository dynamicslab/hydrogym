# Load snapshots as firedrake Functions and save as numpy arrays in individual files
# Idea is to avoid PETSc's parallelization in favor of modred's for modal analysis
import numpy as np
import firedrake as fd
from firedrake.petsc import PETSc
assert fd.COMM_WORLD.size == 1, "Run in serial"

import hydrogym as gym

output_dir = 'output'
snap_file = 'output/snapshots.h5'
save_prefix=f'{output_dir}/snapshots/'

gym.utils.snapshots_to_numpy(snap_file, save_prefix, m=100)

flow = gym.flow.Cylinder(mesh_name='noack')
flow.save_mass_matrix(f'{output_dir}/mass_matrix')