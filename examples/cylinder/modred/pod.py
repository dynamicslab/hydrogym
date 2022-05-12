import firedrake as fd
from firedrake.petsc import PETSc
from ufl import curl
import numpy as np

import hydrogym as gym

output_dir = './output'
pvd_out = f"{output_dir}/solution.pvd"
chk_out = f"{output_dir}/checkpoint.h5"

flow = gym.flow.Cylinder(Re=100, h5_file=chk_out)

snap = gym.linalg.Snapshot('output/snapshots', flow, idx=0)

r = 8
coeffs, mode_handles = gym.linalg.pod(flow,
    snapshot_file = f'{output_dir}/snapshots',
    decomp_indices = range(100),
    output_dir = './output',
    remove_mean = True,
    r = r
)