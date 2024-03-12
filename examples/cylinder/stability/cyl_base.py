"""Solve the steady-state problem for the cylinder"""

from cyl_common import base_checkpoint, flow, stabilization

import hydrogym.firedrake as hgym

steady_solver = hgym.NewtonSolver(
    flow,
    stabilization=stabilization,
)
qB = steady_solver.solve()

# Check lift/drag
hgym.print(flow.compute_forces(qB))

# Save checkpoint
flow.save_checkpoint(qB, base_checkpoint)
