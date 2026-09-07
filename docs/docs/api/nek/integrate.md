---
sidebar_label: integrate
title: hydrogym.nek.integrate
---

#### integrate

```python
def integrate(env,
              t_span: Optional[Tuple[float, float]] = None,
              dt: Optional[float] = None,
              callbacks: Iterable[CallbackBase] = [],
              controller: Optional[Callable] = None,
              max_steps: Optional[int] = None,
              num_steps: Optional[int] = None)
```

Integrate a Nek environment through time.

The loop runs until one of the following stops it first: the end of
``t_span`` is reached, the step budget (``max_steps``/``num_steps``) is
exhausted, or the environment reports the episode is over
(``terminated or truncated``).

**Arguments**:

- `env` - Nek environment (NekEnv, NekParallelEnv, or NekPettingZooEnv)
- `t_span` - Tuple of (start_time, end_time). Optional: when omitted the
  loop is bounded only by the step budget (and episode termination),
  and simulated time is reported relative to t=0.
- ``0 - Time step (optional, uses env&#x27;s default if not provided)
- ``1 - List of callbacks to evaluate throughout the solve
- ``2 - Controller object or function. Supports multiple formats:
  - SB3-style object: Object with ``3 method (e.g.,
  ``4)
  - Legacy function: ``5 or
  ``6
- ``7 - Maximum number of steps (optional). If not given, derived
  from t_span (number of dt intervals + 1).
- ``8 - Alternative step bound, taking precedence over max_steps
  when both are given. Provided for callers that think in interaction
  counts rather than a time span (e.g.
  ``integrate(env, controller=model, num_steps=1000)``); it is the
  same kind of limit as max_steps, not an additional one.
  

**Returns**:

  The environment after integration
  

**Raises**:

- ``1 - if no stop condition is specified (neither t_span nor a
  step budget via max_steps/num_steps).

