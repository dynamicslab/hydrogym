import os
import firedrake as fd
import numpy as np

import hydrogym.firedrake as hgym

from step_response import control_vec, LinearBDFSolver

Re = 100
eig_dir = f"../re{Re}_med_eig_output"

#
# Steady base flow
#
flow = hgym.RotaryCylinder(
    Re=Re,
    velocity_order=2,
    restart=f"{eig_dir}/base.h5"
)

qB = flow.q.copy(deepcopy=True)

#
# Derive flow field associated with actuation BC
#
qC = control_vec(flow)


#
# Determine unstable subspace
#

evals = np.load(f"{eig_dir}/evals.npy")

# Load the set of eigenvectors
r = len(evals)
tol = 1e-10
V = []
with fd.CheckpointFile(f"{eig_dir}/evecs.h5", "r") as chk:
    for (i, w) in enumerate(evals[:r]):
        q = chk.load_function(flow.mesh, f"evec_{i}")
        V.append(q)

W = []
with fd.CheckpointFile(f"{eig_dir}/adj_evecs.h5", "r") as chk:
    for (i, w) in enumerate(evals[:r]):
        q = chk.load_function(flow.mesh, f"evec_{i}")
        W.append(q)


# Sort by real part
sort_idx = np.argsort(-evals.real)
evals = evals[sort_idx]

V = [V[i] for i in sort_idx]
W = [W[i] for i in sort_idx]

# Unstable subspace
unstable_idx = np.where(evals.real > 0)[0]
Vu = [V[i] for i in unstable_idx]
Wu = [W[i] for i in unstable_idx]

def orthogonal_project(q, V, W):
    """Orthogonal projection of q onto the subspace spanned by Vu"""
    for i in range(len(V)):
        q_dot_w = flow.inner_product(q, W[i])
        # q_dot_w = flow.inner_product(q, V[i])
        norm_w = 1
        alpha = q_dot_w / norm_w
        # print(i, alpha)
        q.assign(q - V[i] * alpha)
    return q


#
# Step response
#

# Initial condition: stable projection of qC
qCs = qC.copy(deepcopy=True)
orthogonal_project(qCs, Vu, Wu)

# Second-order BDF time-stepping
lin_flow = flow.linearize(qB)
fn_space = lin_flow.function_space
bcs = lin_flow.bcs

q_trial = fd.TrialFunction(fn_space)
q_test = fd.TestFunction(fn_space)
(u, p) = fd.split(q_trial)
(v, s) = fd.split(q_test)

J = lin_flow.J

# qs = q_trial - sum([Vu[i] * ufl.inner(u, Wu[i].subfunctions[0]) for i in range(len(Vu))])
# Js = ufl.replace(J, {q_trial: qs})

dt = 0.01  # Time step
solver = LinearBDFSolver(fn_space, J, order=2, dt=dt, bcs=bcs, q0=qCs)

tf = 300
n_steps = int(tf // dt)
CL = np.zeros(n_steps+1)
CD = np.zeros(n_steps+1)

CL[0], CD[0] = map(np.real, flow.get_observations())

for i in range(n_steps):
    q = solver.step()
    orthogonal_project(q, Vu, Wu)
    flow.q.assign(q)
    CL[i+1], CD[i+1] = map(np.real, flow.get_observations())

    if i % 10 == 0:
        hgym.print(f"t={(i+1)*dt:.2f}, CL={CL[i+1]:.4f}, CD={CD[i+1]:.4f}")

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data = np.column_stack((np.arange(n_steps+1) * dt, CL, CD))
np.save(f"{output_dir}/re{Re}_subspace_response.npy", data)