# Load snapshots as firedrake Functions and save as numpy arrays in individual files
# Idea is to avoid PETSc's parallelization in favor of modred's for modal analysis
from common import *
assert fd.COMM_WORLD.size == 1, "Run in serial"

from firedrake import logging
logging.set_level(logging.DEBUG)

# This will convert them to the mesh defined in the common.restart file
gym.utils.snapshots_to_numpy(flow, snap_file, snap_prefix, m=100)
gym.utils.snapshots_to_numpy(flow, transient_file, transient_prefix, m=3000)

flow.save_mass_matrix(f'{output_dir}/mass_matrix')