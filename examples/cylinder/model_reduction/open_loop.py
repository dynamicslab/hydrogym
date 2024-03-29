"""For debugging: open-loop DNS of the linearized system with and without acutation"""
import os
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import ufl
import control

import hydrogym.firedrake as hgym
from lti_system import control_vec
from step_response import LinearBDFSolver

eig_dir = "./re100_med_eig_output"
output_dir = './re100_open_loop_output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

flow = hgym.RotaryCylinder(
    Re=100,
    velocity_order=2,
    restart=f"{eig_dir}/base.h5"
)

qB = flow.q.copy(deepcopy=True)

qC = control_vec(flow)


# Load eigenmodes for projection
    
evals = np.load(f"{eig_dir}/evals.npy")
r = len(evals)

V = []
with fd.CheckpointFile(f"{eig_dir}/evecs.h5", "r") as chk:
    for (i, w) in enumerate(evals[:r]):
        q = chk.load_function(flow.mesh, f"evec_{i}")
        V.append(q)

W = []
with fd.CheckpointFile(f"{eig_dir}/adj_evecs.h5", "r") as chk:
    # mesh = chk.load_mesh("mesh")
    for (i, w) in enumerate(evals[:r]):
        q = chk.load_function(flow.mesh, f"evec_{i}")
        W.append(q)

# Sort by real part
sort_idx = np.argsort(-evals.real)
evals = evals[sort_idx]

V = [V[i] for i in sort_idx]
W = [W[i] for i in sort_idx]


#
# Set up timestepping (BDF2)
#
lin_flow = flow.linearize(qB)
fn_space = lin_flow.function_space
bcs = lin_flow.bcs

J = lin_flow.J

c = fd.Constant(0.0)
f = c * qC.subfunctions[0].copy(deepcopy=True)

q0 = fd.project(ufl.real(V[0]), fn_space)
q0.assign(q0 / flow.inner_product(q0, q0))

dt = 0.01

solver = LinearBDFSolver(fn_space, J, order=2, dt=dt, bcs=bcs, q0=q0, f=f)


#
# Natural flow (no actuation)
#
tf = 50
n_steps = int(tf // dt)

t = dt * np.arange(n_steps)
x = np.zeros((n_steps, r), dtype=complex)  # Kalman filter state
u = np.zeros((n_steps))  # Control signal
y = np.zeros((n_steps))  # Measurement signal

flow.q.assign(solver.q)
hgym.print("\n**************************")
hgym.print("*** DNS (no actuation) ***")
hgym.print("**************************\n")
for i in range(n_steps):
    CL, CD = map(np.real, flow.get_observations())
    y[i] = CL
    x[i] = np.array([flow.inner_product(flow.q, W[j]) for j in range(r)])

    if i % 10 == 0:
        hgym.print(f"t={t[i]:.2f}, y={y[i]:.4f}, u={u[i]:.4f}")

    q = solver.step()
    flow.q.assign(q)

np.savez(f"{output_dir}/no_actuation.npz", t=t, y=y, u=u, x=x)

#
# Sinusoidal actuation (explicit RHS forcing)
#
q0 = fd.project(ufl.real(v0), fn_space)
q0.assign(q0 / flow.inner_product(q0, q0))
solver.q.assign(q0)

tf = 50
n_steps = int(tf // dt)
m = 1
p = 1

t = dt * np.arange(n_steps)
x = np.zeros((n_steps, r))  # Kalman filter state
u = np.zeros((n_steps))  # Control signal
y = np.zeros((n_steps))  # Measurement signal

# Integrated control signal (this would be the BC value)
rho = np.zeros(n_steps)

flow.q.assign(solver.q)
hgym.print("\n*****************************************")
hgym.print("*** Sine actuation (explicit forcing) ***")
hgym.print("*****************************************\n")
for i in range(n_steps):
    CL, CD = map(np.real, flow.get_observations())
    y[i] = CL
    x[i] = np.array([flow.inner_product(flow.q, W[j]) for j in range(r)])
    # u[i] = np.cos(np.pi * t[i])
    u[i] = 0.1

    c.assign(u[i])

    if i > 0:
        rho[i] = rho[i-1] - dt * u[i]

    if i % 10 == 0:
        hgym.print(f"t={t[i]:.2f}, y={y[i]:.4f}, u={u[i]:.4f}")

    q = solver.step()
    flow.q.assign(q)

np.savez(f"{output_dir}/sine_actuation_ex.npz", t=t, y=y, u=u, x=x, rho=rho)


#
# Sine actuation (time-varying BC forcing)
#
q0 = fd.project(ufl.real(v0), fn_space)
q0.assign(q0 / flow.inner_product(q0, q0))
solver.q.assign(q0)

tf = 50
n_steps = int(tf // dt)
m = 1
p = 1

t = dt * np.arange(n_steps)
x = np.zeros((n_steps, r))  # Kalman filter state
u = np.zeros((n_steps))  # Control signal
y = np.zeros((n_steps))  # Measurement signal

# Integrated control signal (this would be the BC value)
rho = np.zeros(n_steps)

flow.q.assign(solver.q)
c.assign(0.0)  # Clear explicit RHS forcing
hgym.print("\n****************************************")
hgym.print("*** Sine actuation (time-varying BC) ***")
hgym.print("****************************************\n")
for i in range(n_steps):
    CL, CD = map(np.real, flow.get_observations())
    y[i] = CL
    x[i] = np.array([flow.inner_product(flow.q, W[j]) for j in range(r)])
    # u[i] = np.cos(np.pi * t[i])
    u[i] = 0.1

    # The Dirichlet boundary condition is the integrated negative control signal
    # c(t) = -drho/dt
    if i > 0:
        rho[i] = rho[i-1] - dt * u[i]

    flow.bcu_actuation[0].set_scale(rho[i])

    if i % 10 == 0:
        hgym.print(f"t={t[i]:.2f}, y={y[i]:.4f}, u={u[i]:.4f}")

    q = solver.step()
    flow.q.assign(q)

np.savez(f"{output_dir}/sine_actuation_bc.npz", t=t, y=y, u=u, x=x, rho=rho)
