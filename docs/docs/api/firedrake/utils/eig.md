---
sidebar_label: eig
title: hydrogym.firedrake.utils.eig
---

#### eig

```python
def eig(flow,
        v0=None,
        sigma=0.0,
        adjoint=False,
        schur_restart=False,
        n_evals=10,
        krylov_dim=100,
        sort=None,
        tol=1e-10,
        delta=0.1,
        maxiter=None,
        rng_seed=None)
```

Eigenvalue decomposition using a shift-invert Arnoldi method.

