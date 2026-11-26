---
sidebar_label: power
title: hydrogym.jax.utils.power
---

Actuation-power `info` entries for the jit-compiled (gymnax) JAX environments.

The gymnax `step` is `jax.jit`-compiled, so the `info` dict may only contain arrays (pytree leaves). This
module builds the entries of `hydrogym.power.power_info` from traced arrays; the `power_sampling` string is
carried as a `StaticStr`, a `str` subclass registered as a leaf-less pytree node, so it passes through
`jit`/`vmap`/`scan` as static data and still compares equal to `"mean"`/`"end"`.

## StaticStr Objects

```python
class StaticStr(str)
```

A string that JAX treats as static pytree data (no leaves).

#### power\_info\_jax

```python
def power_info_jax(per_actuator,
                   *,
                   sampling: str,
                   forms: Optional[Mapping[str, Any]] = None,
                   drag_coefficient=None,
                   objective: Optional[Mapping[str, Any]] = None) -> dict
```

Traceable counterpart of `hydrogym.power.power_info` (same keys, array values).

