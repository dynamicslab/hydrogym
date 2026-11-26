---
sidebar_label: integrate
title: hydrogym.firedrake.solvers.integrate
---

#### integrate

```python
def integrate(flow,
              t_span,
              dt,
              method="BDF",
              callbacks=[],
              controller=None,
              collect_rewards=False,
              **options)
```

Integrate the flow forward in time with the chosen transient method.

All transient methods share `core.TransientSolver.solve`, which supports
optional reward collection: pass `collect_rewards=True` to get back a
`(flow, rewards)` tuple instead of just the final flow state.

