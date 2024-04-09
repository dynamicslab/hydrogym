import argparse
import os

import firedrake as fd
import ufl
import numpy as np

import hydrogym.firedrake as hgym


def duplicate_complex_conjugates(evals, evecs, tol=1e-10):
  """Duplicate eigenvectors for complex conjugate pairs."""
  V = []
  all_evals = []
  for (i, w) in enumerate(evals):
    V.append(evecs[i])
    all_evals.append(w)

    # Double for complex conjugates
    if abs(w.imag) < tol:
      evals[i] = w.real
      continue

    q_conj = fd.Function(flow.mixed_space)
    for (u1, u2) in zip(q_conj.subfunctions, evecs[i].subfunctions):
      u1.interpolate(ufl.conj(u2))

    V.append(q_conj)
    all_evals.append(w.conjugate())
  return all_evals, V


def normalize(V, inner_product):
  """Normalize the eigenvectors against themselves."""
  for i in range(len(V)):
    # First normalize the direct eigenvector
    alpha = np.sqrt(inner_product(V[i], V[i]))
    V[i].assign(V[i] / alpha)
  return V


def binormalize(V, W, inner_product):
  """Normalize the adjoint eigenvectors against the direct eigenvectors."""
  print(len(V), len(W))
  for i in range(len(V)):
    alpha = inner_product(V[i], W[i])

    # The adjoint eigenvectors can be the conjugate of the ones
    # that are actually bi-orthogonal, so swap if necessary
    if np.isclose(abs(alpha), 0, atol=tol) and i < len(V) - 1:
        # Swap adjoint vectors so that they are not orthonormal
        W[i], W[i+1] = W[i+1], W[i]
        alpha = flow.inner_product(V[i], W[i])
      
    W[i].assign(W[i] / alpha.conj())
  return W


parser = argparse.ArgumentParser(
    description="Stability analysis of the Re=100 cylinder wake.")
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
    "--n",
    default=16,
    type=int,
    help="Number of eigenvalues to compute.",
)
parser.add_argument(
    "--krylov-dim",
    default=100,
    type=int,
    help="Dimension of the Krylov subspace (number of Arnoldi vectors)",
)
parser.add_argument(
    "--tol",
    default=1e-10,
    type=float,
    help="Tolerance to use for determining converged eigenvalues.",
)
parser.add_argument(
    "--no-adjoint",
    action="store_true",
    default=False,
    help="Skip computing the adjoint modes.",
)
parser.add_argument(
    "--sigma",
    default=0.0,
    type=complex,
    help="Shift for the shift-invert Arnoldi method.",
)
parser.add_argument(
    "--base-flow",
    type=str,
    help="Path to the HDF5 checkpoint containing the base flow.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="eig_output",
    help="Directory in which output files will be stored.",
)
if __name__ == "__main__":
  args = parser.parse_args()

  mesh = args.mesh
  sigma = args.sigma
  tol = args.tol
  Re = args.reynolds

  output_dir = args.output_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  velocity_order = 2
  stabilization = "none"

  flow = hgym.Cylinder(
      Re=Re,
      mesh=mesh,
      velocity_order=velocity_order,
  )

  hgym.print("|------------------------------------------------|")
  hgym.print("| Linear stability analysis of the cylinder wake |")
  hgym.print("|------------------------------------------------|")
  hgym.print(f"Reynolds number:       {Re}")
  hgym.print(f"Number of eigenvalues: {args.n}")
  hgym.print(f"Spectral shift:        {sigma}")
  hgym.print(f"Include adjoint modes: {not args.no_adjoint}")
  hgym.print(f"Krylov dimension:      {args.krylov_dim}")
  hgym.print("")

  if args.base_flow:
    hgym.print(f"Loading base flow from checkpoint {args.base_flow}")
    flow.load_checkpoint(args.base_flow)

  else:
    hgym.print("Solving the steady-state problem for the cylinder base flow")

    steady_solver = hgym.NewtonSolver(
        flow,
        stabilization=stabilization,
        solver_parameters={"snes_monitor": None})
    if Re > 50:
      hgym.print("\tSolving steady-state problem at Re=50...")
      flow.Re.assign(50)
      steady_solver.solve()

    hgym.print(
        f"\tSolving steady-state problem at target Reynolds number Re={Re}...")
    flow.Re.assign(Re)
    steady_solver.solve()
    CL, CD = map(np.real, flow.compute_forces())
    hgym.print(f"Lift: {CL:0.3e}, Drag: {CD:0.3e}")
    flow.save_checkpoint(f"{args.output_dir}/base.h5")

  hgym.print("\nComputing direct modes...")
  qB = flow.q.copy(deepcopy=True)
  A = flow.linearize(qB)

  evals, V = hgym.utils.linalg.eig(
      A, n=args.n, sigma=sigma, tol=tol, krylov_dim=args.krylov_dim)

  hgym.print("\n--- Eigenvalues ---")
  for i, w in enumerate(evals):
    hgym.print(f"Eigenvalue {i}: {w.real:0.6f} + {w.imag:0.6f}i")

  # Duplicate eigenvectors for complex conjugate pairs
  evals, V = duplicate_complex_conjugates(evals, V, tol=tol)

  np.save(f"{output_dir}/evals", evals)

  # Keep only the unstable modes
  keep_idx = np.nonzero(np.real(evals) > 0)[0]
  evals = evals[keep_idx]
  V = [V[i] for i in keep_idx]

  # Normalize the direct modes against themselves (not really necessary)
  V = normalize(V, flow.inner_product)

  # Save direct modes as checkpoints
  with fd.CheckpointFile(f"{output_dir}/evecs.h5", "w") as chk:
    chk.save_mesh(flow.mesh)
    for i in range(len(evals)):
      V[i].rename(f"evec_{i}")
      chk.save_function(V[i])

  if not args.no_adjoint:
    hgym.print("\nComputing adjoint modes...")
    adj_evals, W = hgym.utils.linalg.eig(
        A.T, n=args.n, sigma=sigma, tol=tol, krylov_dim=args.krylov_dim)
    
    hgym.print("\n--- Adjoint eigenvalues ---")
    for i, ev in enumerate(adj_evals):
      hgym.print(f"Eigenvalue {i}: {ev.real:0.6f} + {ev.imag:0.6f}i")

    # Duplicate eigenvectors for complex conjugate pairs
    adj_evals, W = duplicate_complex_conjugates(adj_evals, W, tol=tol)

    # Keep only the unstable modes
    keep_idx = np.nonzero(np.real(adj_evals) > 0)[0]
    adj_evals = adj_evals[keep_idx]
    W = [W[i] for i in keep_idx]

    # The adjoint basis is already bi-orthogonal with the direct basis, but
    # we can still normalize it.
    W = binormalize(V, W, flow.inner_product)

    # Save adjoint modes as checkpoints in the same file
    with fd.CheckpointFile(f"{output_dir}/adj_evecs.h5", "w") as chk:
      chk.save_mesh(flow.mesh)
      for i in range(len(evals)):
        W[i].rename(f"evec_{i}")
        chk.save_function(W[i])

  hgym.print(
      "NOTE: If there is a warning following this, ignore it.  It is raised by PETSc "
      "CLI argument handling and not argparse. It does not indicate that any CLI "
      "arguments are ignored by this script.")
