"""
Actuator power of wall transpiration in NekRS, mapped onto the HydroGym power interface (hydrogym.power).

The udf returns, at every GLL node of the actuated boundaries, the pressure and the wall velocity imposed by the
actuators (WALLS with the wall-velocity flag; exact for every footprint, direction and zero-net-flux mode). At
each node the power put into the fluid is (Bewley, Moin & Temam 2001, J. Fluid Mech. 447; Hasegawa & Kasagi 2011,
J. Fluid Mech. 683, App. A)

    pi = v_n (p - p_ref) + v_n |u_w|^2 / 2,      v_n = u_w . n (n into the fluid, v_n > 0 = blowing),

with p_ref the instantaneous mean of the wall pressure over the nodes that share the coordinates `pressure_mean`
(default x and y: the spanwise mean at each chordwise station, as in the HydroGym Nek5000 cases). For zero net
mass flux on every such group p_ref drops out of the signed power.
The cost forms are taken per node before any spatial or temporal sum:

    signed    sum_n w_n (pi_p + pi_k)
    positive  sum_n w_n (max(pi_p, 0) + max(pi_k, 0))     (Hasegawa & Kasagi 2011, conservative)
    abs       sum_n w_n (|pi_p| + |pi_k|)                  (Bewley et al. 2001, conservative estimate)

with pi_p = v_n (p - p_ref), pi_k = v_n |u_w|^2 / 2 and w_n the surface quadrature weights (one entry per face copy,
so the sum is the wall integral). All three are divided by the reference power 0.5 rho U^3 A of the env (A = length x
span, or an area), i.e. the normalisation that turns its drag force into C_D, which gives C_P.

Per actuator: the signed node power is attributed to the nearest actuator centre (exact for point actuators on the
wall nodes; the Voronoi cell of the centres for extended footprints). The per-actuator values sum to the signed total.
"""

from typing import Dict, Optional, Sequence

import numpy as np

from hydrogym.power import transpiration_forms

from . import layout


class WallPower:
    """Actuator power on a set of boundary nodes (geometry of NekRSInterface.wall_data(..., geometry=True))."""

    def __init__(
        self,
        xyz: np.ndarray,
        w: np.ndarray,
        normal: np.ndarray,
        actuator_xyz: np.ndarray,
        reference_power: float,
        pressure_mean: Optional[Sequence[str]] = ("x", "y"),
        group_tol: Optional[float] = None,
    ):
        """
        Args:
            xyz, w, normal: node positions [n, 3], surface quadrature weights [n], unit normals into the fluid [n, 3].
            actuator_xyz: actuator centres [m, 3] (attribution of the node power to the actuators).
            reference_power: 0.5 rho U^3 A; the powers are divided by it (C_P).
            pressure_mean: coordinates whose shared values define the groups of the reference pressure p_ref
                (default x, y: spanwise mean per station); None or [] uses the full pressure (p_ref = 0).
            group_tol: tolerance of the grouping (layout.group_ids).
        """
        self.xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        self.w = np.asarray(w, dtype=float).ravel()
        self.normal = np.asarray(normal, dtype=float).reshape(-1, 3)
        if not (len(self.w) == len(self.xyz) == len(self.normal)):
            raise ValueError("xyz, w and normal must have one entry per node")
        if not np.isfinite(reference_power) or reference_power <= 0:
            raise ValueError(f"reference power must be positive, got {reference_power!r}")
        self.reference_power = float(reference_power)
        act = np.asarray(actuator_xyz, dtype=float).reshape(-1, 3)
        self.n_actuators = len(act)
        if self.n_actuators == 0:
            raise ValueError("no actuators")
        from scipy.spatial import cKDTree

        self.owner = cKDTree(act).query(self.xyz)[1].astype(np.int64)
        self.groups = None
        if pressure_mean:
            self.groups = layout.group_ids(self.xyz, list(pressure_mean), group_tol)
            ng = int(self.groups.max()) + 1
            self._wg = np.bincount(self.groups, self.w, ng)
            self._ng = ng

    def node_terms(self, p: np.ndarray, u_wall: np.ndarray):
        """Pressure and kinetic power per node: (pi_p with p - p_ref, pi_k, pi_p with the full pressure)."""
        p = np.asarray(p, dtype=float).ravel()
        u = np.asarray(u_wall, dtype=float).reshape(-1, 3)
        vn = np.einsum("ij,ij->i", u, self.normal)
        if self.groups is not None:
            pref = np.bincount(self.groups, self.w * p, self._ng) / self._wg
            pp = p - pref[self.groups]
        else:
            pp = p
        return pp * vn, 0.5 * vn * np.einsum("ij,ij->i", u, u), p * vn

    def evaluate(self, p: np.ndarray, u_wall: np.ndarray) -> Dict[str, np.ndarray]:
        """C_P of one instant: signed / positive / abs totals, the signed full-pressure total and per actuator."""
        pip, pik, pfull = self.node_terms(p, u_wall)
        w, ref = self.w, self.reference_power
        forms = transpiration_forms(pip, pik, w)  # the shared point-wise rule of hydrogym.power
        return {
            "signed": forms["signed"] / ref,
            "positive": forms["positive"] / ref,
            "abs": forms["abs"] / ref,
            "signed_full_pressure": float(np.sum(w * (pfull + pik))) / ref,
            "per_actuator": np.bincount(self.owner, w * (pip + pik), self.n_actuators) / ref,
        }


class PowerAverage:
    """Time-step-weighted mean of WallPower.evaluate results over the solver steps of one action."""

    def __init__(self):
        self._sum: Optional[Dict[str, np.ndarray]] = None
        self._wsum = 0.0
        self.count = 0

    def add(self, values: Dict[str, np.ndarray], weight: float = 1.0) -> None:
        if self._sum is None:
            self._sum = {k: weight * np.asarray(v, dtype=float) for k, v in values.items()}
        else:
            for k, v in values.items():
                self._sum[k] = self._sum[k] + weight * np.asarray(v, dtype=float)
        self._wsum += weight
        self.count += 1

    def mean(self) -> Dict[str, np.ndarray]:
        if self._sum is None or self._wsum <= 0:
            raise ValueError("no samples")
        out = {k: v / self._wsum for k, v in self._sum.items()}
        return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in out.items()}


def reference_power(rho: float, spec: Optional[Dict], boundary_area: float) -> float:
    """0.5 rho U^3 A from ``{velocity, length, span}`` or ``{velocity, area}``; defaults U = 1, A = boundary area."""
    spec = dict(spec or {})
    u = float(spec.get("velocity", 1.0))
    if "area" in spec and spec["area"] not in (None, "auto"):
        area = float(spec["area"])
    elif "length" in spec or "span" in spec:
        area = float(spec.get("length", 1.0)) * float(spec.get("span", 1.0))
    else:
        area = float(boundary_area)
    return 0.5 * float(rho) * u**3 * area
