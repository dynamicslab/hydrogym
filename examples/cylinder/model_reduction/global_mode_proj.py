import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import ufl

import hydrogym.firedrake as hgym

output_dir = "../eig_output"
output_dir = "../re40_med_eig_output"
# output_dir = "../re40_fine_eig_output"

# 1. Base flow
flow = hgym.RotaryCylinder(
    Re=40,
    mesh="medium",
    velocity_order=2,
    restart=f"{output_dir}/base.h5"
)

qB = flow.q.copy(deepcopy=True)

# 2. Derive flow field associated with actuation BC
# See Barbagallo et al. (2009) for details on the "lifting" procedure
F = flow.residual(fd.split(qB))  # Nonlinear variational form
J = fd.derivative(F, qB)  # Jacobian with automatic differentiation

flow.linearize_bcs()
flow.set_control([1.0])
bcs = flow.collect_bcs()

# Solve steady, inhomogeneous problem
qC = fd.Function(flow.mixed_space, name="qC")
v, s = fd.TestFunctions(flow.mixed_space)
zero = ufl.inner(fd.Constant((0.0, 0.0)), v) * ufl.dx
fd.solve(J == zero, qC, bcs=bcs)


# 3. Global stability modes
evals = np.load(f"{output_dir}/evals.npy")

# Load the set of eigenvectors
r = len(evals)
tol = 1e-10
V = []
with fd.CheckpointFile(f"{output_dir}/evecs.h5", "r") as chk:
    for (i, w) in enumerate(evals[:r]):
        q = chk.load_function(flow.mesh, f"evec_{i}")
        V.append(q)


W = []
with fd.CheckpointFile(f"{output_dir}/adj_evecs.h5", "r") as chk:
    for (i, w) in enumerate(evals[:r]):
        q = chk.load_function(flow.mesh, f"evec_{i}")
        W.append(q)

# Sort by real part
sort_idx = np.argsort(-evals.real)
evals = evals[sort_idx]

V = [V[i] for i in sort_idx]
W = [W[i] for i in sort_idx]


#
# 4. Projection onto global modes
#

r = 12  # Number of global modes for projection
Ar = np.zeros((r, r), dtype=np.complex128)
Br = np.zeros((r, 1), dtype=np.complex128)
Cr = np.zeros((1, r), dtype=np.complex128)

A = flow.linearize(qB)
A.copy_output = True

def meas(q):
    flow.q.assign(q)
    CL, _CD = flow.get_observations()
    return CL

def real_part(q):
    return

for i in range(r):
    for j in range(r):
        Ar[j, i] = flow.inner_product(A @ V[i], W[j])

    Br[i, 0] = flow.inner_product(qC, W[i])
    Cr[0, i] = meas(V[i])

# Finally the feedthrough term
Dr = meas(qC)


# 5. Transfer function of the reduced-order model
def H(s):
    return Cr @ np.linalg.inv(Ar - s * np.eye(r)) @ Br + Dr

omega = 1j * np.linspace(0.01, 4.0, 1000)
H_omega = np.array([H(s).ravel() for s in omega])

fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
ax[0].semilogy(omega.imag, np.abs(H_omega))
# ax[0].set_xlim(0, 2)
ax[1].plot(omega.imag, np.angle(H_omega))
plt.show()
