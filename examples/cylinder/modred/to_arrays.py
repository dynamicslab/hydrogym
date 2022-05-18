# Load snapshots as firedrake Functions and save as numpy arrays in individual files
# Idea is to avoid PETSc's parallelization in favor of modred's for modal analysis
from common import *
assert fd.COMM_WORLD.size == 1, "Run in serial"

gym.utils.snapshots_to_numpy(snap_file, save_prefix, m=100)
flow.save_mass_matrix(f'{output_dir}/mass_matrix')