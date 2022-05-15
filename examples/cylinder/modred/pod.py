import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np
from scipy import sparse

from firedrake import logging
logging.set_level(logging.DEBUG)

import hydrogym as gym

output_dir = './output'
pvd_out = f"{output_dir}/solution.pvd"
chk_out = f"{output_dir}/checkpoint.h5"

flow = gym.flow.Cylinder(Re=100, h5_file=chk_out)

r = 8
coeffs, mode_handles = gym.linalg.pod(flow,
    snapshot_prefix = f'{output_dir}/snapshots/',
    decomp_indices = range(100),
    mass_matrix = f'{output_dir}/mass_matrix',
    output_dir = './output',
    remove_mean = True,
    r = r
)