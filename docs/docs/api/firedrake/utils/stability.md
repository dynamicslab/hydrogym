---
sidebar_label: stability
title: hydrogym.firedrake.utils.stability
---

#### stability\_analysis

```python
def stability_analysis(flow,
                       sigma=0.0,
                       krylov_dim=100,
                       tol=1e-6,
                       adjoint=False,
                       schur_restart=False,
                       schur_delta=0.1,
                       n_evals=12)
```

Linear stability analysis of the flow.

**Arguments**:

- `flow` - The FlowConfiguration to analyze.
  sigma:
  The shift for the shift-invert Arnoldi method. The algorithm will converge
  most quickly if the shift is close to the eigenvalues of interest.
- `m` - The dimension of the Krylov subspace (number of Arnoldi vectors).
- `tol` - Tolerance to use for determining converged eigenvalues.
- `adjoint` - If True, compute the adjoint modes along with the direct modes.
  schur_restart:
  If True, use Krylov-Schur iteration to restart the Arnoldi process.
  schur_delta:
  The stability margin to use when determining which Schur eigenvalues
  to keep in Krylov-Schur iteration. Ignored if `schur_restart` is False.
  n_evals:
  The number of eigenvalues to converge in order to terminate Krylov-Schur
  iteration. Ignored if `schur_restart` is False.

