"""Impulse response of the stable subspace of the cylinder wake."""

import argparse
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


def load_eig(flow, eig_dir, unstable_only=True):
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

    if unstable_only:
        unstable_idx = np.where(evals.real > 0)[0]
        V = [V[i] for i in unstable_idx]
        W = [W[i] for i in unstable_idx]
        evals = evals[unstable_idx]

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



parser = argparse.ArgumentParser(
    description="Impulse response of the stable subspace of the cylinder wake.")
parser.add_argument(
    "--mesh",
    default="medium",
    type=str,
    help='Identifier for the mesh resolution. Options: ["medium", "fine"]',
)
parser.add_argument(
    "--reynolds",
    default=100.0,
    type=float,
    help="Reynolds number of the flow",
)
parser.add_argument(
    "--dt",
    default=1e-2,
    type=float,
    help="Time step for the impulse response simulation.",
)
parser.add_argument(
    "--tf",
    default=50.0,
    type=float,
    help="End time for the impulse response simulation.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="impulse_output",
    help="Directory in which output files will be stored.",
)
parser.add_argument(
    "--eig-dir",
    type=str,
    default="eig_output",
    help="Directory with results of stability analysis.",
)
parser.add_argument(
    "--no-adjoint",
    action="store_true",
    default=False,
    help="Skip computing the adjoint modes.",
)
parser.add_argument(
    "--no-snapshots",
    action="store_true",
    default=False,
    help="Skip saving snapshots.",
)
if __name__ == "__main__":
    args = parser.parse_args()

    mesh = args.mesh
    Re = args.reynolds
    eig_dir = args.eig_dir
    output_dir = args.output_dir
    dt = args.dt
    tf = args.tf

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #
    # Steady base flow
    #
    flow = hgym.RotaryCylinder(
        Re=Re,
        velocity_order=2,
        mesh=mesh,
        restart=f"{eig_dir}/base.h5"
    )

    qB = flow.q.copy(deepcopy=True)

    # Derive flow field associated with actuation BC
    qC = control_vec(flow)

    # Derive flow field associated with measurement matrix
    qM = measurement_matrix(flow)

    #
    # Load unstable subspace - direct and adjoint global modes
    #
    evals, Vu, Wu = load_eig(flow, eig_dir, unstable_only=True)

    #
    # Direct response (controllable modes)
    #
    flow.q.assign(qB)
    save_snapshots = not args.no_snapshots
    data, X = impulse_response(
        flow,
        qB,
        qC,
        Vu,
        Wu,
        adjoint=False,
        dt=dt,
        tf=tf,
        save_snapshots=save_snapshots,
    )

    if save_snapshots:
        with fd.CheckpointFile(f"{output_dir}/dir_snapshots.h5", "w") as chk:
            chk.save_mesh(flow.mesh)
            for (i, q) in enumerate(X):
                q.rename(f"q_{i}")
                chk.save_function(q)

    np.save(f"{output_dir}/dir_response.npy", data)

    #
    # Adjoint response (observable modes)
    #
    if args.no_adjoint:
        exit()
    
    flow.q.assign(qB)
    data, Y = impulse_response(
        flow,
        qB,
        qM,
        Vu,
        Wu,
        adjoint=True,
        dt=dt,
        tf=tf,
        save_snapshots=save_snapshots,
    )

    if save_snapshots:
        with fd.CheckpointFile(f"{output_dir}/adj_snapshots.h5", "w") as chk:
            chk.save_mesh(flow.mesh)
            for (i, q) in enumerate(Y):
                q.rename(f"q_{i}")
                chk.save_function(q)

    np.save(f"{output_dir}/adj_response.npy", data)
