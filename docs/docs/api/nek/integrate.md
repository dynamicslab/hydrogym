---
sidebar_label: integrate
title: hydrogym.nek.integrate
---

#### integrate

```python
def integrate(env,
              t_span: Tuple[float, float],
              dt: Optional[float] = None,
              callbacks: Iterable[CallbackBase] = [],
              controller: Optional[Callable] = None,
              max_steps: Optional[int] = None)
```

Integrate a Nek environment through time.

**Arguments**:

- `env` - Nek environment (NekEnv, NekParallelEnv, or NekPettingZooEnv)
- `t_span` - Tuple of (start_time, end_time)
- `dt` - Time step (optional, uses env&#x27;s default if not provided)
- `callbacks` - List of callbacks to evaluate throughout the solve
- `controller` - Controller object or function. Supports multiple formats:
  - SB3-style object: Object with `predict()` method (e.g.,
  `model.predict(obs, state=..., episode_start=..., deterministic=True)`)
  - Legacy function: `action = controller(t, obs, env)` or
  `action = controller(t, obs)`
- `max_steps` - Maximum number of steps (optional)
  

**Returns**:

  The environment after integration

