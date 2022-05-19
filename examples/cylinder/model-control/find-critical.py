import firedrake as fd
from firedrake.petsc import PETSc
from slepc4py import SLEPc

import numpy as np
assert PETSc.ScalarType == np.complex128, "Complex PETSc configuration required for stability analysis"

import hydrogym as gym
# mesh = 'noack'
mesh = 'sipp-lebedev'

def least_stable(Re):
    cyl = gym.flow.Cylinder(Re=Re, mesh_name=mesh)
    qB = cyl.solve_steady()
    A, M = cyl.linearize(qB)

    evals, vr, vi = gym.linalg.eig(A, M, num_eigenvalues=20, sigma=0.8j)
    max_idx = np.argmax(np.real(evals))

    gym.print(f'Re={Re}:\t\tlargest: {evals[max_idx]}')
    return evals[max_idx]

# Bisection search
#  Noack: Re=45.7, omega=0.853
#  Sipp & Lebedev: Re=46.52, omega=0.744
Re_lo = 40
Re_hi = 50
sigma_hi = np.real(least_stable(Re_hi))
sigma_lo = np.real(least_stable(Re_lo))
while (Re_hi - Re_lo) > 0.05:
    Re_mid = 0.5*(Re_hi + Re_lo)
    gym.print((Re_lo, Re_mid, Re_hi))
    sigma_mid = np.real(least_stable(Re_mid))
    if sigma_mid > 0:
        Re_hi = Re_mid
    else:
        Re_lo = Re_mid
    
# least_stable(40)
# least_stable(50)
# least_stable(100)
