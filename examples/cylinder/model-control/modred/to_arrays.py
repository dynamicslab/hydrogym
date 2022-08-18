# Load snapshots as firedrake Functions and save as numpy arrays in individual files
# Idea is to avoid PETSc's parallelization in favor of modred's for modal analysis
import firedrake as fd
from common import (
    flow,
    output_dir,
    snap_file,
    snap_prefix,
    transient_file,
    transient_prefix,
)
from firedrake import logging

import hydrogym as gym

assert fd.COMM_WORLD.size == 1, "Run in serial"

logging.set_level(logging.DEBUG)

# This will convert them to the mesh defined in the common.restart file
gym.utils.snapshots_to_numpy(flow, snap_file, snap_prefix, m=100)
gym.utils.snapshots_to_numpy(flow, transient_file, transient_prefix, m=3000)

flow.save_mass_matrix(f"{output_dir}/mass_matrix")
