"""Actuation-power ``info`` entries for the jit-compiled (gymnax) JAX environments.

The gymnax ``step`` is ``jax.jit``-compiled, so the ``info`` dict may only contain arrays (pytree leaves). This
module builds the entries of ``hydrogym.power.power_info`` from traced arrays; the ``power_sampling`` string is
carried as a ``StaticStr``, a ``str`` subclass registered as a leaf-less pytree node, so it passes through
``jit``/``vmap``/``scan`` as static data and still compares equal to ``"mean"``/``"end"``.
"""

from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp

from hydrogym.power import POWER_SAMPLINGS, power_forms


class StaticStr(str):
    """A string that JAX treats as static pytree data (no leaves)."""


jax.tree_util.register_pytree_node(StaticStr, lambda s: ((), str(s)), lambda aux, _: StaticStr(aux))


def power_info_jax(
    per_actuator,
    *,
    sampling: str,
    forms: Optional[Mapping[str, Any]] = None,
    drag_coefficient=None,
    objective: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Traceable counterpart of ``hydrogym.power.power_info`` (same keys, array values)."""
    if sampling not in POWER_SAMPLINGS:
        raise ValueError(f"sampling must be one of {POWER_SAMPLINGS}, got {sampling!r}")
    f = forms if forms is not None else power_forms(per_actuator, xp=jnp)
    info = {
        "actuation_power": jnp.asarray(f["signed"]),
        "actuation_power_positive": jnp.asarray(f["positive"]),
        "actuation_power_abs": jnp.asarray(f["abs"]),
        "actuation_power_per_actuator": jnp.ravel(jnp.asarray(per_actuator)),
        "power_sampling": StaticStr(sampling),
    }
    if drag_coefficient is not None:
        info["drag_coefficient"] = jnp.asarray(drag_coefficient)
    if objective is not None:
        info["objective"] = dict(objective)
    return info
