---
sidebar_label: nozzle
title: hydrogym.jaxfluids.utils.nozzle
---

## NozzleGeometry Objects

```python
@dataclass(frozen=True, slots=True)
class NozzleGeometry()
```

Fixed nozzle geometry based on the publication
Das et al. 2025 AIAA

#### area\_ratio\_inlet

```python
def area_ratio_inlet(dim: int) -> float
```

Computes the ratio of the inlet area
to the throat area.

## InjectorGeometry Objects

```python
@dataclass(frozen=True, slots=True)
class InjectorGeometry()
```

#### X

position

#### IW

width

#### N

count

## PressureRatios Objects

```python
@dataclass(frozen=True, slots=True)
class PressureRatios()
```

#### NPR

nozzle pressure ratio

#### SPR

secondary pressure ratio

#### compute\_thrust

```python
def compute_thrust(
        primitives: Array, p_infty: float, apertures_x: Array,
        cell_centers: tuple[Array, ...],
        cell_sizes: tuple[Array, ...]) -> tuple[Array, Array, Array]
```

Computes the thrust of the nozzle.

F_x = mdot_e * u_e + (p_e - p_infty) * A_e
F_y = mdot_e * v_e
F_z = mdot_e * w_e

#### primary\_reference\_power

```python
def primary_reference_power(spec: "TVCSpec") -> float
```

Power normalisation of the nozzle: exergy flow rate of the ideal choked primary jet.

`P_ref = mdot* R T0 ln(NPR)` with
`mdot* = A* p0 sqrt(gamma / (R T0)) (2 / (gamma + 1))^((gamma + 1) / (2 (gamma - 1)))`, `p0 = NPR p_inf`,
`T0 = T_inf` (the inflow total temperature of the released cases), `A* = 2 R_throat` per unit span in 2D
and `pi R_throat^2` in 3D.

#### initialize\_injector\_power\_fn

```python
def initialize_injector_power_fn(
    injector_geometry: InjectorGeometry, pressure_ratios: PressureRatios,
    p_infty: float, T_infty: float, specific_heat_ratio: float,
    specific_gas_constant: float, sim_manager: SimulationManager
) -> Callable[[Array, Array, Array, Array, Array], tuple[Array, Array]]
```

Per-cell injection power, consistent with `initialize_injector_flux_fn`.

For every interface cell of injector `i` the injected mass flux is the one of the flux function,
`mdot = rho_inj u_n A_if`, and its power is the exergy flow of gas supplied from a reservoir at
`T_infty` and the injector total pressure `p_t`, relative to the ambient state
`(p_infty, T_infty)`: `P = mdot R T_infty ln(p_t / p_infty)` (the minimum, isothermal, compression
work; low-Mach limit `Q (p_t - p_infty)`, the incompressible transpiration power).

**Returns**:

  `fn(primitives, interface_length, normal, mesh_grid, actuator) -> (mdot, power)`, both of shape
  `(num_injectors, N)` for flattened cell arrays of length `N`.

