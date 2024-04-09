"""Compute the BPOD modes of the cylinder wake flow."""

import argparse
import os
import numpy as np
import firedrake as fd
import hydrogym.firedrake as hgym
from scipy import linalg

def bpod(X, Y, inner_product, r=None):
    """Balanced proper orthogonal decomposition with the method of snapshots.
    
    Args:
        X: List of direct snapshots.
        Y: List of adjoint snapshots.
        r: Number of BPOD modes to compute for stable subspace.
        
    Returns:
        V: List of direct BPOD modes.
        W: List of adjoint BPOD modes.
        S: Singular values of the correlation matrix.
    """
    m_d = len(X)  # Number of direct snapshots
    m_a = len(Y)  # Number of adjoint snapshots

    if r is None:
        r = min(m_d, m_a)

    R = np.zeros((m_a, m_d))  # Correlation matrix
    for i in range(m_a):
        for j in range(m_d):
            R[i, j] = inner_product(X[j], Y[i]).real

    U, S, T = linalg.svd(R)

    V = []   # direct modes: X @ T.T @ S ** (-1/2)
    W = []   # adjoint modes: Y @ U @ S ** (-1/2)

    r = 64  # Number of BPOD modes to compute for stable subspace
    for i in range(r):
        q = fd.Function(flow.mixed_space)
        for j in range(m_d):
            q.assign(q + X[j] * T[i, j] / np.sqrt(S[i]))
        V.append(q)

        q = fd.Function(flow.mixed_space)
        for j in range(m_a):
            q.assign(q + Y[j] * U[j, i] / np.sqrt(S[i]))
        W.append(q)

    return V, W, S


parser = argparse.ArgumentParser(
    description="Balanced POD for the cylinder wake.")
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
    "--eig-dir",
    type=str,
    default="eig_output",
    help="Directory with results of stability analysis.",
)
parser.add_argument(
    "--snapshot-dir",
    type=str,
    default="impulse_output",
    help="Directory in which the DNS snapshots are stored.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="bpod_output",
    help="Directory in which output files will be stored.",
)
parser.add_argument(
    "--num-modes",
    type=int,
    default=64,
    help="Number of BPOD modes to compute (default 64).",
)
parser.add_argument(
    "--num-dir-snapshots",
    type=int,
    default=500,
    help="Number of direct snapshots in impulse response.",
)
parser.add_argument(
    "--num-adj-snapshots",
    type=int,
    default=500,
    help="Number of adjoint snapshots in impulse response.",
)
if __name__ == "__main__":
    args = parser.parse_args()

    mesh = args.mesh
    Re = args.reynolds
    output_dir = args.output_dir
    eig_dir = args.eig_dir
    snapshot_dir = args.snapshot_dir
    r = args.num_modes

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    flow = hgym.RotaryCylinder(
        Re=Re,
        velocity_order=2,
        mesh=mesh,
        restart=f"{eig_dir}/base.h5",
    )

    qB = flow.q.copy(deepcopy=True)


    #
    # Load impulse response data
    #

    # Direct impulse response solution
    X = []
    m_d = args.num_dir_snapshots  # Number of direct snapshots
    with fd.CheckpointFile(f"{snapshot_dir}/dir_snapshots.h5", "r") as chk:
        for i in range(m_d):
            q = chk.load_function(flow.mesh, f"q_{i}")
            X.append(q)

    # Adjoint impulse response solution
    Y = []
    m_a = args.num_adj_snapshots  # Number of adjoint snapshots
    with fd.CheckpointFile(f"{snapshot_dir}/adj_snapshots.h5", "r") as chk:
        for i in range(m_a):
            q = chk.load_function(flow.mesh, f"q_{i}")
            Y.append(q)

    #
    # BPOD: Method of snapshots
    #
    V_bpod, W_bpod, S = bpod(X, Y, flow.inner_product, r=r)

    #
    # Save the BPOD modes and Hankel singular values
    #
    with fd.CheckpointFile(f"{output_dir}/bpod_modes.h5", "w") as chk:
        chk.save_mesh(flow.mesh)
        for i, v in enumerate(V_bpod):
            v.rename(f"v_{i}")
            chk.save_function(v)

        for i, w in enumerate(W_bpod):
            w.rename(f"w_{i}")
            chk.save_function(w)

    np.save(f"{output_dir}/hankel_svs.npy", S)