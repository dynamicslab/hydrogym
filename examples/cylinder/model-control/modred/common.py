import hydrogym as gym

output_dir = "output"
pvd_out = f"{output_dir}/solution.pvd"
restart = "../transient/checkpoint.h5"  # Restart file for unsteady simulation
checkpoint = f"{output_dir}/checkpoint.h5"

# Snapshots used for computing POD modes
snap_file = f"{output_dir}/snapshots.h5"
snap_prefix = f"{output_dir}/snapshots/"

# Snapshots that will be used to compute coefficients for transient wake
transient_file = "../transient/output/snapshots.h5"
transient_prefix = "../transient/output/snapshots/"

# Where the POD modes will end up
pod_file = f"{output_dir}/pod.h5"
pod_prefix = f"{output_dir}/pod"

flow = gym.flow.Cylinder(Re=100, h5_file=restart)
