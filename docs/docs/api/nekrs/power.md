---
sidebar_label: power
title: hydrogym.nekrs.power
---

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

## WallPower Objects

```python
class WallPower()
```

Actuator power on a set of boundary nodes (geometry of NekRSInterface.wall_data(..., geometry=True)).

#### \_\_init\_\_

```python
def __init__(xyz: np.ndarray,
             w: np.ndarray,
             normal: np.ndarray,
             actuator_xyz: np.ndarray,
             reference_power: float,
             pressure_mean: Optional[Sequence[str]] = ("x", "y"),
             group_tol: Optional[float] = None)
```

**Arguments**:

  xyz, w, normal: node positions [n, 3], surface quadrature weights [n], unit normals into the fluid [n, 3].
- `actuator_xyz` - actuator centres [m, 3] (attribution of the node power to the actuators).
- `reference_power` - 0.5 rho U^3 A; the powers are divided by it (C_P).
- `pressure_mean` - coordinates whose shared values define the groups of the reference pressure p_ref
  (default x, y: spanwise mean per station); None or [] uses the full pressure (p_ref = 0).
- `group_tol` - tolerance of the grouping (layout.group_ids).

#### node\_terms

```python
def node_terms(p: np.ndarray, u_wall: np.ndarray)
```

Pressure and kinetic power per node: (pi_p with p - p_ref, pi_k, pi_p with the full pressure).

#### evaluate

```python
def evaluate(p: np.ndarray, u_wall: np.ndarray) -> Dict[str, np.ndarray]
```

C_P of one instant: signed / positive / abs totals, the signed full-pressure total and per actuator.

## PowerAverage Objects

```python
class PowerAverage()
```

Time-step-weighted mean of WallPower.evaluate results over the solver steps of one action.

#### reference\_power

```python
def reference_power(rho: float, spec: Optional[Dict],
                    boundary_area: float) -> float
```

0.5 rho U^3 A from `{velocity, length, span}` or `{velocity, area}`; defaults U = 1, A = boundary area.

