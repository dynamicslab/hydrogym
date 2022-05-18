import firedrake as fd
from firedrake.petsc import PETSc
from firedrake import dx, ds, lhs, rhs
import ufl
from ufl import inner, dot, nabla_grad, div
import numpy as np

assert (PETSc.ScalarType == np.float64), "Run in real mode"
assert (fd.COMM_WORLD.size == 1), "Run on single core"

from scipy import linalg
import scipy.io as sio

import hydrogym as gym

import control

mesh = 'noack'
evec_dir = 'global-modes'
output_dir = 'continuous'

flow = gym.flow.Cylinder(Re=50, h5_file=f'{evec_dir}/steady.h5')
qB = flow.q.copy(deepcopy=True)

qC = flow.linearize_control(qB)
with qC.dat.vec_ro as vec:
    B = vec.array

# Construct modal basis from real/imag parts
flow.load_checkpoint(f'{evec_dir}/direct_real.h5', idx=0, read_mesh=False)
vr = gym.utils.get_array(flow.q)
    
flow.load_checkpoint(f'{evec_dir}/direct_imag.h5', idx=0, read_mesh=False)
vi = gym.utils.get_array(flow.q)
    
flow.load_checkpoint(f'{evec_dir}/adjoint_real.h5', idx=0, read_mesh=False)
wr = gym.utils.get_array(flow.q)

flow.load_checkpoint(f'{evec_dir}/adjoint_imag.h5', idx=0, read_mesh=False)
wi = gym.utils.get_array(flow.q)

# Construct basis from real/imaginary parts
V = np.vstack([vr + 1j*vi, vr - 1j*vi]).T
W = np.vstack([wr - 1j*wi, wr + 1j*wi]).T

M = flow.mass_matrix(backend='scipy')

# Rescale so bi-orthonormal
scale = W[:, 0].T.conj() @ M @ V[:, 0]
W[:, 0] /= scale.conj()
W[:, 1] /= scale

# Convert to real-valued
T = np.array([[1, 1j], [1, -1j]])/np.sqrt(2)
V = np.real(V @ T)
W = np.real(W @ T)

# Save adjoint vectors for projection
for idx in range(2):
    gym.utils.set_from_array(flow.q, W[:, idx])
    flow.save_checkpoint(f'{output_dir}/w{idx}.h5', write_mesh=False)

    gym.utils.set_from_array(flow.q, V[:, idx])
    flow.save_checkpoint(f'{output_dir}/v{idx}.h5', write_mesh=False)

# Project linearized dynamics onto modal basis
flow.linearize_bcs()
bcs = flow.collect_bcs()
uB, _ = qB.split()

r = 2
y_dim = 2
Ar = np.zeros((r, r))
Cr = np.zeros((y_dim, r))

qH = fd.Function(flow.mixed_space)
v, _ = qH.split()
for i in range(r):
    for j in range(r):
        gym.utils.set_from_array(flow.q, V[:, i])
        gym.utils.set_from_array(qH, W[:, j])
        
        Ar[i, j] = fd.assemble( fd.inner(
            -dot(uB, nabla_grad(flow.u)) + -dot(flow.u, nabla_grad(uB)) + div(flow.sigma(flow.u, flow.p)), v
        )*dx )
        Cr[:, i] = flow.collect_observations()

ct_evals, _ = linalg.eig(Ar)
print(ct_evals)
        
P = (W.T @ M)  # Now `linalg.inv(W.T.conj() @ M @ V) = I`
Br = P @ B[:, None]

### LQR design
Q = np.eye(r)
R = 1e8
K, _, ctrl_evals = control.lqr(Ar, Br, Q, R)

print(ctrl_evals)

### Kalman design
QN = 1e-6*np.eye(r)
RN = 1e-6*np.eye(y_dim)
GN = np.eye(r)

L, _, _ = control.lqe(Ar, GN, Cr, QN, RN)

sio.savemat(f'{output_dir}/controller.mat',
    {'A': Ar, 'B': Br, 'C': Cr, 'P': P, 'K': K, 'L': L, 'V': V, 'W': W})