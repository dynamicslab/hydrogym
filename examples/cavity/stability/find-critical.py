import firedrake as fd
from firedrake.petsc import PETSc
from slepc4py import SLEPc
from ufl import real

import numpy as np
assert PETSc.ScalarType == np.complex128, "Complex PETSc configuration required for stability analysis"

import hydrogym as gym


## First we have to ramp up the Reynolds number to get the steady state
Re_init = [500, 1000, 2000, 4000]
flow = gym.flow.Cavity(Re=Re_init[0])
gym.print(f"Steady solve at Re={Re_init[0]}")
qB = flow.solve_steady(solver_parameters={"snes_monitor": None})

for (i, Re) in enumerate(Re_init[1:]):
    flow.Re.assign(real(Re))
    gym.print(f"Steady solve at Re={Re_init[i+1]}")
    qB = flow.solve_steady(solver_parameters={"snes_monitor": None})

def least_stable(flow, Re):
    flow.Re.assign(real(Re))
    qB = flow.solve_steady(solver_parameters={"snes_monitor": None})
    A, M = flow.linearize(qB)

    evals, es = gym.linalg.eig(A, M, num_eigenvalues=20, sigma=7.5j)
    max_idx = np.argmax(np.real(evals))

    gym.print(f'Re={Re}:\t\tlargest: {evals[max_idx]}')
    return evals[max_idx]

# Bisection search
#  Sipp & Lebedev: Re=4140, omega=7.5
#  Bengana et al:  Re=4126
#  Meliga:         Re=4114
#  Hydrogym:       Re=4134, omega=7.49
Re_lo = 4000
Re_hi = 4200
sigma = np.real(least_stable(flow,  0.5*(Re_hi + Re_lo)))
while abs(sigma) > 1e-6:
    Re_mid = 0.5*(Re_hi + Re_lo)
    gym.print((Re_lo, Re_mid, Re_hi))
    sigma = np.real(least_stable(flow, Re_mid))
    if sigma > 0:
        Re_hi = Re_mid
    else:
        Re_lo = Re_mid