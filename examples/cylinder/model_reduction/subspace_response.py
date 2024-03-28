import os
import firedrake as fd
import numpy as np

import hydrogym.firedrake as hgym

from lti_system import control_vec, measurement_matrix, LinearBDFSolver


def orthogonal_project(flow, q, V, W, adjoint=False):
    """Orthogonal projection of q onto the subspace spanned by the modes V"""
    if adjoint:
        V, W = W, V
    for i in range(len(V)):
        alpha = flow.inner_product(q, W[i])
        q.assign(q - V[i] * alpha)
    return q


def load_eig(flow, eig_dir):
    """Load eigenvalues and (direct, adjoint) eigenvectors."""
    evals = np.load(f"{eig_dir}/evals.npy")

    # Load the set of eigenvectors
    r = len(evals)
    V = []
    with fd.CheckpointFile(f"{eig_dir}/evecs.h5", "r") as chk:
        # mesh = chk.load_mesh("mesh")
        for (i, w) in enumerate(evals[:r]):
            # q = chk.load_function(mesh, f"evec_{i}")
            # V.append(fd.interpolate(q, flow.mixed_space))
            q = chk.load_function(flow.mesh, f"evec_{i}")
            V.append(q)

    W = []
    with fd.CheckpointFile(f"{eig_dir}/adj_evecs.h5", "r") as chk:
        # mesh = chk.load_mesh("mesh")
        for (i, w) in enumerate(evals[:r]):
            # q = chk.load_function(mesh, f"evec_{i}")
            # V.append(fd.interpolate(q, flow.mixed_space))
            q = chk.load_function(flow.mesh, f"evec_{i}")
            W.append(q)


    # Sort by real part
    sort_idx = np.argsort(-evals.real)
    evals = evals[sort_idx]

    V = [V[i] for i in sort_idx]
    W = [W[i] for i in sort_idx]

    return evals, V, W


def impulse_response(
    flow, qB, q0, Vu, Wu, dt=0.01, tf=10.0, adjoint=False, save_interval=10, save_snapshots=True
):
    lin_flow = flow.linearize(qB, adjoint=adjoint)

    # Initial condition via stable projection
    q0s = q0.copy(deepcopy=True)
    orthogonal_project(flow, q0s, Vu, Wu, adjoint=adjoint)

    fn_space = lin_flow.function_space
    bcs = lin_flow.bcs
    J = lin_flow.J

    # Project the initial condition onto the boundary conditions
    q0s = fd.project(q0s, fn_space, bcs=bcs)

    solver = LinearBDFSolver(fn_space, J, order=2, dt=dt, bcs=bcs, q0=q0s)

    n_steps = int(tf // dt)
    CL = np.zeros(n_steps)
    CD = np.zeros(n_steps)
    flow.q.assign(solver.q)
    CL[0], CD[0] = map(np.real, flow.get_observations())


    snapshots = []
    for i in range(n_steps-1):
        if i % save_interval == 0:
            hgym.print(f"t={i*dt:.2f}, CL={CL[i]:.4f}, CD={CD[i]:.4f}")
            if save_snapshots:
                snapshots.append(flow.q.copy(deepcopy=True))

        q = solver.step()
        orthogonal_project(flow, q, Vu, Wu, adjoint=adjoint)
        flow.q.assign(q)
        CL[i+1], CD[i+1] = flow.get_observations()


    data = np.column_stack((np.arange(n_steps) * dt, CL, CD))
    return data, snapshots



if __name__ == "__main__":
    Re = 100

    eig_dir = f"./re{Re}_med_eig_output"
    output_dir = f"./re{Re}_impulse_output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    #
    # Steady base flow
    #
    flow = hgym.RotaryCylinder(
        Re=Re,
        velocity_order=2,
        restart=f"{eig_dir}/base.h5"
    )

    qB = flow.q.copy(deepcopy=True)


    # Unstable subspace
    evals, V, W = load_eig(flow, eig_dir)
    unstable_idx = np.where(evals.real > 0)[0]
    Vu = [V[i] for i in unstable_idx]
    Wu = [W[i] for i in unstable_idx]

    # Derive flow field associated with actuation BC
    qC = control_vec(flow)

    # Derive flow field associated with measurement matrix
    qM = measurement_matrix(flow)

    dt = 0.01
    tf = 10.0

    # Direct response (controllable modes)
    flow.q.assign(qB)
    data, snapshots = impulse_response(
        flow,
        qB,
        qC,
        Vu,
        Wu,
        adjoint=False,
        dt=dt,
        tf=tf,
    )

    with fd.CheckpointFile(f"{output_dir}/dir_snapshots.h5", "w") as chk:
        chk.save_mesh(flow.mesh)
        for (i, q) in enumerate(snapshots):
            q.rename(f"q_{i}")
            chk.save_function(q)

    np.save(f"{output_dir}/dir_response.npy", data)

    # Adjoint response (observable modes)

    flow.q.assign(qB)
    data, snapshots = impulse_response(
        flow,
        qB,
        qM,
        Vu,
        Wu,
        adjoint=True,
        dt=dt,
        tf=tf,
    )

    with fd.CheckpointFile(f"{output_dir}/adj_snapshots.h5", "w") as chk:
        chk.save_mesh(flow.mesh)
        for (i, q) in enumerate(snapshots):
            q.rename(f"q_{i}")
            chk.save_function(q)

    np.save(f"{output_dir}/adj_response.npy", data)


    # Long simulation to derive transfer function
    # Note that running for much longer than this will eventually
    # lead to instability as a result of small errors in the estimate
    # of the unstable subspace compared to the time-stepping.
    tf = 300.0
    dt = 0.01
    data, _snapshots = impulse_response(
        flow,
        qB,
        qC,
        Vu,
        Wu,
        adjoint=False,
        dt=dt,
        tf=tf,
        save_snapshots=False,
    )

    np.save(f"{output_dir}/long_response.npy", data)