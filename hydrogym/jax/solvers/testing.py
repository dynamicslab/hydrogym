from base import RK4CNSolver 
from hydrogym.jax.flow import FlowConfig
from hydrogym.jax.utils import io as io

print_fmt = (
    "t: {0:0.3f}\t\tCFL: {1:0.3f}\t\t KE: {2:0.3e}\t\t TKE: {3:0.3e}\t\t Mem: {4:0.1f}"
)

def log_postprocess(flow, trajectory):
  obs = flow.get_observations(trajectory)
  print("made it here")
  return [obs[1], obs[2], obs[3], obs[4]]

output_dir = 'kolmo'




flow = FlowConfig()
solver = RK4CNSolver(flow, 0.001, 1)

callbacks = [
    # hgym.io.ParaviewCallback(interval=100, filename=pvd_out, postprocess=compute_vort),
    # hgym.io.CheckpointCallback(interval=1000, filename=checkpoint),
    io.LogCallback(
        postprocess=log_postprocess(flow, trajectory),
        nvals=4,
        interval=1,
        filename=f"{output_dir}/kolmogorov.dat",
        print_fmt=print_fmt,
    ),
]

final, trajectory = solver.solve(0.001, flow, (0,10), callbacks)
print(trajectory.shape)
obs = flow.get_observations(trajectory)
print(obs)

