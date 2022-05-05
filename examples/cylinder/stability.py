import firedrake as fd
from firedrake.petsc import PETSc
from slepc4py import SLEPc

import numpy as np
assert PETSc.ScalarType == np.complex128, "Complex PETSc configuration required for stability analysis"

def print(s):
    PETSc.Sys.Print(s)

import hydrogym as gym
# mesh = 'noack'
mesh = 'sipp-lebedev'

def least_stable(Re):
    cyl = gym.flow.Cylinder(Re=Re, mesh_name=mesh)
    qB = cyl.solve_steady()
    A, B = cyl.linearize(qB)

    ### SLEPc
    opts = PETSc.Options()
    # opts.setValue("eps_monitor_all", None)
    opts.setValue("eps_gen_non_hermitian", None)
    opts.setValue("eps_target", "0.8i") 
    opts.setValue("eps_type", "krylovschur")
    opts.setValue("eps_largest_real", True)
    opts.setValue("st_type", "sinvert")
    # opts.setValue("st_ksp_type", "gmres")
    # opts.setValue("st_ksp_view", None)
    # opts.setValue("st_ksp_monitor_true_residual", None)
    # opts.setValue("st_pc_type", "bjacobi")
    opts.setValue("st_pc_factor_shift_type", "NONZERO")
    opts.setValue("eps_tol", 1e-10)

    num_eigenvalues = 20
    es = SLEPc.EPS().create(comm=fd.COMM_WORLD)
    es.setDimensions(num_eigenvalues)
    es.setOperators(A, B)
    es.setFromOptions()
    es.solve()

    nconv = es.getConverged()
    vr, vi = A.getVecs()

    evals = np.array([es.getEigenpair(i, vr, vi) for i in range(nconv)])
    max_idx = np.argmax(np.real(evals))

    print(f'Re={Re}:\t\t{nconv} converged, largest: {evals[max_idx]}')
    return evals[max_idx]


# Bisection search
#  Noack: Re=45.7, omega=0.853
#  Sipp & Lebedev: Re=46.6, omega=0.744
Re_lo = 40
Re_hi = 50
sigma_hi = np.real(least_stable(Re_hi))
sigma_lo = np.real(least_stable(Re_lo))
while (Re_hi - Re_lo) > 0.05:
    Re_mid = 0.5*(Re_hi + Re_lo)
    print((Re_lo, Re_mid, Re_hi))
    sigma_mid = np.real(least_stable(Re_mid))
    if sigma_mid > 0:
        Re_hi = Re_mid
    else:
        Re_lo = Re_mid
    
# least_stable(40)
# least_stable(50)
# least_stable(100)
