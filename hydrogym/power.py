"""Actuation power: definitions shared by all HydroGym backends.

The controlled environments report the power their actuators put into the flow, in the units of their drag
coefficient, and can optionally charge that power in the reward. With the default ``power_weight = 0`` the reward is
unchanged. The Nek5000 backend (``hydrogym.nek``) reports it for its wing cases only, not for the channels.

Power put into the fluid by actuator ``i`` (one rotating body, one jet, one forcing mode or one wall patch):

* rotating body / moving wall: ``P_i = Omega_i M_i`` (shaft power), equivalently ``-sum_links F_link . u_w``;
* wall transpiration / jets: ``P_i = int v_n (p - p_ref + rho |u|^2 / 2) dA`` (Bewley, Moin & Temam 2001,
  J. Fluid Mech. 447; Hasegawa & Kasagi 2011, J. Fluid Mech. 683); ``p_ref`` drops out for zero net mass flux;
* volume forcing: ``P_i = int f . u dV``.

Each environment normalises ``P_i`` exactly as it normalises its drag force into ``C_D``
(``C_P = P / (rho U_ref^3 L_ref / 2)``, times the span in 3D), so that ``C_D`` is the drag power coefficient and the
net saving against an uncontrolled reference is ``S = 1 - (C_D + C_P) / C_D0``.

Three cost forms are formed per actuator before summing:

* ``signed``: ``sum_i P_i`` (power recovered by an actuator is credited);
* ``positive``: ``sum_i max(P_i, 0)`` (Hasegawa & Kasagi 2011, App. A);
* ``abs``: ``sum_i |P_i|`` (Bewley et al. 2001, conservative estimate).

Which unit the forms are taken over (the same rule in every backend):

* a rigid moving body (rotating cylinder) is one actuator with one motor: ``P_i = Omega_i M_i`` per body
  (``power_forms``);
* wall transpiration and jets are resolved point-wise on the actuated surface, with the pressure and kinetic terms
  separate (``transpiration_forms``); the per-actuator value is the signed sum over the actuator's points;
* ``p_ref`` is the instantaneous mean wall pressure over each zero-net-mass-flux group of actuators, so the signed
  total does not depend on the pressure gauge.

Backends pass point-wise totals to ``power_info`` via ``forms``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

POWER_COSTS = ("signed", "positive", "abs")
POWER_SAMPLINGS = ("end", "mean")


def power_forms(per_actuator: Sequence[float], xp: Any = None) -> Dict[str, Any]:
    """Signed, positive-part and absolute totals of per-actuator powers.

    Args:
        per_actuator: Signed power coefficient of each actuator (positive = put into the fluid).
        xp: Optional array namespace (e.g. ``jax.numpy``). By default the totals are computed with NumPy and
            returned as Python floats; with ``xp`` they are computed with ``xp`` and returned as 0-d arrays, so the
            function can be traced (``jax.jit``).

    Returns:
        Dictionary with the totals of the three forms under the keys ``signed``, ``positive`` and ``abs``.
    """
    if xp is not None:
        p = xp.ravel(xp.asarray(per_actuator))
        return {
            "signed": xp.sum(p),
            "positive": xp.sum(xp.clip(p, 0.0, None)),
            "abs": xp.sum(xp.abs(p)),
        }
    p = np.asarray(per_actuator, dtype=float).ravel()
    return {
        "signed": float(p.sum()),
        "positive": float(np.clip(p, 0.0, None).sum()),
        "abs": float(np.abs(p).sum()),
    }


def transpiration_forms(
    pi_p: Sequence[float], pi_k: Sequence[float], w: Sequence[float], xp: Any = None
) -> Dict[str, Any]:
    """Cost forms of wall transpiration (jets), formed per surface (quadrature) point.

    The pressure term ``pi_p = v_n (p - p_ref)`` and the kinetic term ``pi_k = v_n |u_w|^2 / 2`` are kept separate
    (Bewley, Moin & Temam 2001, conservative estimate); ``v_n > 0`` is blowing (into the fluid) and ``w`` are the
    surface quadrature weights:

    * ``signed = sum w (pi_p + pi_k)``
    * ``positive = sum w (max(pi_p, 0) + max(pi_k, 0))`` (Hasegawa & Kasagi 2011, App. A)
    * ``abs = sum w (|pi_p| + |pi_k|)``

    ``p_ref`` is the instantaneous mean wall pressure over the zero-net-mass-flux group of actuators, so that all
    three forms are independent of the pressure gauge.

    Args:
        pi_p, pi_k, w: Arrays of equal length (one entry per surface point).
        xp: Optional array namespace (e.g. ``jax.numpy``), see ``power_forms``.

    Returns:
        Dictionary with the totals of the three forms under the keys ``signed``, ``positive`` and ``abs``.
    """
    m = np if xp is None else xp
    pi_p, pi_k, w = (m.ravel(m.asarray(a)) for a in (pi_p, pi_k, w))
    if not (pi_p.shape == pi_k.shape == w.shape):
        raise ValueError("pi_p, pi_k and w must have one entry per surface point")
    out = {
        "signed": m.sum(w * (pi_p + pi_k)),
        "positive": m.sum(w * (m.clip(pi_p, 0.0, None) + m.clip(pi_k, 0.0, None))),
        "abs": m.sum(w * (m.abs(pi_p) + m.abs(pi_k))),
    }
    return {k: float(v) for k, v in out.items()} if xp is None else out


@dataclass(frozen=True)
class PowerConfig:
    """Reward settings for the actuation power.

    Attributes:
        weight: Reward penalty weight; the reward becomes ``reward - weight * C_P[cost]``. ``0`` disables it.
        cost: Cost form charged in the reward, one of ``signed``, ``positive``, ``abs``.
        sampling: ``"end"`` (instantaneous at the end of the action interval) or ``"mean"`` (mean over the solver
            steps of the interval); ``None`` lets the backend choose its default.
    """

    weight: float = 0.0
    cost: str = "abs"
    sampling: Optional[str] = None

    def __post_init__(self):
        if self.cost not in POWER_COSTS:
            raise ValueError(f"power_cost must be one of {POWER_COSTS}, got {self.cost!r}")
        if self.sampling is not None and self.sampling not in POWER_SAMPLINGS:
            raise ValueError(f"power_sampling must be one of {POWER_SAMPLINGS}, got {self.sampling!r}")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError(f"power_weight must be a finite non-negative number, got {self.weight!r}")

    @classmethod
    def from_config(cls, *sources: Optional[Mapping[str, Any]], **overrides: Any) -> "PowerConfig":
        """Build from configuration mappings (later sources win) and keyword overrides.

        Reads the keys ``power_weight``, ``power_cost`` and ``power_sampling``; missing keys keep their defaults.
        Overrides use the same names (``power_weight=...``); ``None`` overrides are ignored.
        """
        vals: Dict[str, Any] = {}
        for src in sources:
            if not src:
                continue
            for key in ("power_weight", "power_cost", "power_sampling"):
                if key in src and src[key] is not None:
                    vals[key] = src[key]
        for key, val in overrides.items():
            if key not in ("power_weight", "power_cost", "power_sampling"):
                raise TypeError(f"unknown power option {key!r}")
            if val is not None:
                vals[key] = val
        return cls(
            weight=float(vals.get("power_weight", 0.0)),
            cost=str(vals.get("power_cost", "abs")),
            sampling=vals.get("power_sampling"),
        )

    def resolve_sampling(self, default: str) -> str:
        """Sampling actually used: the configured one, or the backend's ``default``."""
        return self.sampling if self.sampling is not None else default

    def apply(self, reward: float, forms: Mapping[str, float]) -> float:
        """Charge the configured cost form; returns ``reward`` itself (same object value) when the weight is 0."""
        if self.weight == 0.0:
            return reward
        cost = forms[self.cost]
        if isinstance(cost, (int, float, np.generic)):
            cost = float(cost)
        return reward - self.weight * cost  # also works for traced (jax) arrays


def power_info(
    per_actuator: Sequence[float],
    *,
    sampling: str,
    drag_coefficient: Optional[float] = None,
    objective: Optional[Mapping[str, Any]] = None,
    forms: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """The ``info`` entries every controlled environment returns.

    Args:
        per_actuator: Signed power coefficient per actuator.
        sampling: ``"end"`` or ``"mean"``.
        drag_coefficient: Drag coefficient of the controlled body/bodies, if the environment has one.
        objective: The environment's objective / force dictionary.
        forms: Totals formed point-wise by the backend; defaults to ``power_forms`` of ``per_actuator``.

    Returns:
        Dictionary with ``actuation_power``, ``actuation_power_positive``, ``actuation_power_abs``,
        ``actuation_power_per_actuator``, ``power_sampling`` and, if given, ``drag_coefficient`` and ``objective``.
    """
    if sampling not in POWER_SAMPLINGS:
        raise ValueError(f"sampling must be one of {POWER_SAMPLINGS}, got {sampling!r}")
    f = dict(forms) if forms is not None else power_forms(per_actuator)
    info: Dict[str, Any] = {
        "actuation_power": float(f["signed"]),
        "actuation_power_positive": float(f["positive"]),
        "actuation_power_abs": float(f["abs"]),
        "actuation_power_per_actuator": np.asarray(per_actuator, dtype=float).ravel().copy(),
        "power_sampling": sampling,
    }
    if drag_coefficient is not None:
        info["drag_coefficient"] = float(drag_coefficient)
    if objective is not None:
        info["objective"] = dict(objective)
    return info


def forms_from_info(info: Mapping[str, Any]) -> Dict[str, float]:
    """The three cost totals from an ``info`` dict made by ``power_info``."""
    return {
        "signed": float(info["actuation_power"]),
        "positive": float(info["actuation_power_positive"]),
        "abs": float(info["actuation_power_abs"]),
    }


def net_saving(drag_coefficient: float, actuation_power: float, drag_coefficient_uncontrolled: float) -> float:
    """Net power saving ``S = 1 - (C_D + C_P) / C_D0`` against an uncontrolled reference."""
    return 1.0 - (drag_coefficient + actuation_power) / drag_coefficient_uncontrolled
