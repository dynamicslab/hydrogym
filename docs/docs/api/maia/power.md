---
sidebar_label: power
title: hydrogym.maia.power
---

Actuation power of the m-AIA environments, evaluated from probe data (no solver change needed).

Two actuator models, both in lattice units of the finest level and normalised like the drag of the environment
(`C_P = 2 P / (rho_0 U^3 L_proj)` with `U = Ma / sqrt(3)` and `L_proj = referenceLength / dX * zLength / dX`,
the projection length of `MaiaFlowEnv.compute_nondim_coefficients`), so `C_D` is the drag power coefficient.

* `RingTorquePower` - rotating circular bodies (boundary condition 2008, `lbControlActions` = angular
velocity command). The torque of the fluid on each body comes from the angular-momentum balance of the fluid
between the body and circles in an annulus at 1.3-1.6 body radii, evaluated from the cell-centre values of the
finest level (m-AIA samples probes by nearest neighbour, so the model probes the cell centres and interpolates
itself); the shaft power is `P = -T Omega` with `Omega = command * dX` per lattice time step (the wall velocity
of BC 2008 is `command * (-r_y, r_x)` with `r` in geometry units). The momentum exchange of the rotating wall
links is not used because it does not resolve the small tangential stress reliably.
* `SlotJetPower` - slot jets on circular bodies (boundary condition 2007, object class of the cylinder
environments): radial wall velocity `v_n(s) = a (1 - s^2 / h^2)` across the slot of half width `h` (the
parabolic profile of m-AIA). The power is the flux of pressure work and kinetic energy through the slot
(Bewley, Moin & Temam 2001): `P = int v_n (p - p_ref + rho v_n^2 / 2) ds`, with the wall pressure extrapolated
from the cell-centre values 1.5-4.5 cells out along the wall normal and `p_ref` the mean wall pressure over all
slots of a plane (the zero-net-mass-flux group). `p_ref` does not change the signed power; it keeps the absolute
lattice pressure out of the positive and absolute forms, which are formed point-wise across the slots with the
pressure and kinetic terms separate (`hydrogym.power.transpiration_forms`).

In 3D both models sample several spanwise planes (rotation axis / jets along z) and scale the per-plane value with
the span. Both sample the flow once, at the end of the action interval (`power_sampling = "end"`).

Accuracy (m-AIA runs of the released cases):

* rotation, pinball Re 100 (D/20): saturated rotation C_P 3.132-3.138 for five annuli between 1.3 and 1.7 radii
(0.2 %); weak rotation with vortex shedding: time means 0.2874-0.2885 (0.4 %). In unsteady flow a single sample
also contains the change of the angular momentum of the annulus fluid, so per-step values scatter while their
means converge (`actuation_power_model_spread` in the info: spread over the circles);
* jets, cylinder Re 1000 (D/64, slot 13 cells): signed C_P 0.191-0.192 for five sampling stencils (0.8 %), absolute
form within 4 %; at Re 200 (D/20, slot 4 cells) the exit pressure is under-resolved and the signed power varies by
about +-10 % with the stencil (`actuation_power_model_spread`: linear vs quadratic wall extrapolation).

## \_ProbeModel Objects

```python
class _ProbeModel()
```

Common geometry and normalisation.

#### coefficient

```python
def coefficient(p_lattice: np.ndarray) -> np.ndarray
```

Lattice power (per unit span in 2D, total in 3D) -> C_P.

## RingTorquePower Objects

```python
class RingTorquePower(_ProbeModel)
```

Shaft power of rotating circular bodies from the flow around them (see module docstring).

m-AIA returns probe values by nearest-neighbour sampling (its default `interpolationOrder`), so a probe gives the
value of the cell centre nearest to it. The model therefore probes exactly the cell centres of the finest level
in an annulus around each body, takes the velocity gradients on that lattice (second-order central differences),
interpolates bilinearly to circles and evaluates on each circle

T(r) = r^2 oint [tau_rt - rho u_r u_t] dtheta,
tau_rt = rho nu [(d_y u + d_x v) cos 2theta + (d_y v - d_x u) sin 2theta],

the torque of the fluid outside the circle on the fluid inside (the pressure drops out on a circle). For a
statistically steady flow T does not depend on r; the torque is the mean over `n_radii` circles in the annulus
and its spread over the circles is kept in `last_spread` (relative) as a check. The annulus must lie on the
finest grid level and must not contain another body.

**Arguments**:

- `centers` - Centres of the rotating bodies, geometry units, shape (n_bodies, 2).
- `radii` - Their radii, geometry units.
- `lattice_origin` - (x0, y0) with the finest-level cell centres at `x0 + (i + 1/2) dX` (from the grid file:
  `centerOfGravity - lengthLevel0 / 2`).
- `axis_signs` - +1 / -1 per body for rotation about +z / -z (3D); 2D: +1 (counter-clockwise positive).
- `annulus` - Inner and outer circle radius in body radii.
  n_radii, n_theta: Circles in the annulus and points per circle.
- `other_bodies` - Further (centre x, centre y, radius) whose cells are never probed (e.g. non-rotating bodies).
  other arguments: see `_ProbeModel` (from the environment's properties).

#### per\_body

```python
def per_body(probe_data: np.ndarray, commands: Sequence[float]) -> np.ndarray
```

C_P of each body from the probe data at `probe_coordinates` and the angular-velocity commands.

#### evaluate

```python
def evaluate(probe_data: np.ndarray, commands: Sequence[float]) -> dict
```

Per-body C_P and the cost forms (one motor per rigid body: forms over the bodies).

## SlotJetPower Objects

```python
class SlotJetPower(_ProbeModel)
```

Power of slot jets on circular bodies (see module docstring).

The exit pressure is taken at the wall: the model probes the finest-level cell centres around each slot (m-AIA
samples probes by nearest neighbour), interpolates the pressure bilinearly along the wall normal at
`wall_distances` (default 1.5-4.5 cells) from each Gauss point of the slot and extrapolates it to the wall with a
quadratic least-squares fit. The relative change of the signed power between the quadratic and a linear
extrapolation is kept in `last_extrapolation_spread` as an error estimate.

**Arguments**:

- `body_centers` - Centre of the body each jet sits on, geometry units, shape (n_jets, 2).
- `body_radii` - Radius of that body, geometry units, shape (n_jets,).
- `angles` - Jet angle (radians from +x), shape (n_jets,).
- `widths` - Slot width `lbJetRanges` (geometry units), shape (n_jets,).
- `lattice_origin` - (x0, y0) of the finest-level cell-centre lattice (see `lattice_origin_from_grid`).
- `n_slot` - Gauss-Legendre points across each slot.
- `wall_distances` - Distances of the pressure samples from the wall along its normal, in finest cells.
  other arguments: see `_ProbeModel`.

#### wall\_state

```python
def wall_state(probe_data: np.ndarray, degree: int = 2)
```

Wall pressure and density at the Gauss points, shapes (planes, jets, points), by extrapolation.

#### point\_terms

```python
def point_terms(probe_data: np.ndarray, commands: Sequence[float])
```

Point-wise pressure and kinetic power densities and surface weights, shapes (planes, jets, points).

`sum(w * (pi_p + pi_k))` is the lattice power of all jets (per unit span in 2D).

#### per\_jet

```python
def per_jet(probe_data: np.ndarray, commands: Sequence[float]) -> np.ndarray
```

Signed C_P of each jet from the probe data and the jet velocity commands (lattice units).

#### evaluate

```python
def evaluate(probe_data: np.ndarray, commands: Sequence[float]) -> dict
```

Per-jet signed C_P and the cost forms, formed point-wise with separate pressure and kinetic terms.

#### read\_cdf\_global\_attributes

```python
def read_cdf_global_attributes(path: str) -> dict
```

Global attributes of a classic-format NetCDF file (CDF-1, CDF-2 or CDF-5, as written by m-AIA's PnetCDF).

Only the header is read; no NetCDF library is needed.

#### lattice\_origin\_from\_grid

```python
def lattice_origin_from_grid(grid_file: str, dX: float,
                             n_dim: int) -> np.ndarray
```

Origin of the finest-level cell-centre lattice (cell centres at `origin + (i + 1/2) dX`).

The root cell of the m-AIA grid is the cube of side `lengthLevel0` centred at `centerOfGravity`.

#### grid\_finest\_cell\_size

```python
def grid_finest_cell_size(grid_file: str) -> float
```

Finest cell size of an m-AIA grid file: `lengthLevel0 / 2**maxLevel` (from the header only).

#### check\_cell\_size

```python
def check_cell_size(grid_file: str,
                    dX: float,
                    source: str = "the property file",
                    rtol: float = 1e-6) -> float
```

Raise ValueError if the grid's finest cell size differs from `dX`; return the grid's cell size.

HydroGym computes `dX = reductionFactor * domainLength / 2**maxRfnmntLvl` from the property file and uses it
to normalise forces, power and probe positions. A property file whose `domainLength` does not match the grid
would otherwise give silently wrong coefficients.

#### jets\_on\_bodies

```python
def jets_on_bodies(props: dict, n_jets: int) -> Optional[tuple]
```

Jet geometry of a jet-actuated circular-body environment from its properties, or None if not applicable.

m-AIA lists `lbJetCenter` per jet as the centre of the body the jet sits on (it adds `lbJetRadius` along
`lbJetAngles` itself), with `lbJetRanges` the slot width.

#### rotating\_bodies

```python
def rotating_bodies(props: dict, n_bodies: int) -> Optional[tuple]
```

(in-plane centres, radii, centre z, axis sign) of the rotating bodies, or None if not applicable.

In 2D, BC 2008 rotates counter-clockwise for a positive command; in 3D about `lbRotationAxis` (z only here).

#### rotation\_change

```python
def rotation_change(previous: Optional[Sequence[float]],
                    commands: Sequence[float], bound: float) -> float
```

Largest change of the rotation command over one action, relative to the actuation bound.

The ring torque of `RingTorquePower` assumes that the angular momentum of the fluid in the annulus is
steady at the sampling instant. That holds for steady or slowly varying rotation; when the command changes by a
sizeable fraction of the bound within one action (e.g. bang-bang reversal), the end-of-action sample also contains
the change of the annulus angular momentum and can even become negative. Values above `ROTATION_STEADY_TOL`
mark such samples as not valid (`info["actuation_power_valid"] = False`).

#### zero\_power

```python
def zero_power(n: int) -> List[float]
```

Per-actuator zeros (environment without actuation at reset).

