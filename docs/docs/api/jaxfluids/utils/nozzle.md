---
sidebar_label: nozzle
title: hydrogym.jaxfluids.utils.nozzle
---

Nozzle Utility Dataclasses and Functions
=========================================

This module provides dataclasses, named tuples, type aliases, and functions used
by the JAXFluids nozzle environments for geometry, physics, and visualisation.

## Type Aliases

```python
TargetThrustAngle = Array | float
TargetThrustAngleFn = Callable[[Array | float], TargetThrustAngle]
```

## NozzleGeometry Objects

```python
@dataclass(frozen=True, slots=True)
class NozzleGeometry
```

Fixed nozzle geometry based on Das et al. 2025 AIAA.

Stores key vertex coordinates (A–H) and the throat radius R as class-level
defaults. The dataclass is frozen and therefore hashable.

**Fields**:

- `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H` (`tuple[float, float]`) - Nozzle wall vertices.
- `R` (`float`) - Throat radius.
- `UPPER_EDGE` (`float`) - Upper domain edge.

#### area\_ratio\_inlet

```python
def area_ratio_inlet(dim: int) -> float
```

Compute the ratio of the inlet area to the throat area.

**Arguments**:

- `dim` - Dimensionality: `2` for planar, `3` for axisymmetric.

**Returns**:

  Area ratio (linear for 2-D, squared for 3-D).

**Raises**:

- `ValueError` - If `dim` is not 2 or 3.

#### D\_exit

```python
@property
def D_exit() -> float
```

Exit diameter `2 * H[1]`.

#### D\_throat

```python
@property
def D_throat() -> float
```

Throat diameter `2 * R`.

## InjectorGeometry Objects

```python
@dataclass(frozen=True, slots=True)
class InjectorGeometry
```

Injector position and size parameters.

**Fields**:

- `X` (`float`) - Relative axial position of the injector along the nozzle wall.
- `IW` (`float`) - Injector width (metres).
- `N` (`int`) - Number of injectors.

## InjectorPlaneParameters Objects

```python
@dataclass(frozen=True, slots=True)
class InjectorPlaneParameters
```

Pre-computed local coordinate frames for each injector.

**Fields**:

- `positions` (`Array`) - Shape `(N, 3)`. Injector centre coordinates.
- `tangents` (`Array`) - Shape `(N, 3)`. Nozzle-wall tangent at each injector.
- `normals` (`Array`) - Shape `(N, 3)`. Inward surface normal at each injector.

## PressureRatios Objects

```python
@dataclass(frozen=True, slots=True)
class PressureRatios
```

Nozzle operating pressure ratios.

**Fields**:

- `NPR` (`float`) - Nozzle pressure ratio (total-to-ambient).
- `SPR` (`float`) - Secondary (injector) pressure ratio.

## ObsData Objects

```python
class ObsData(NamedTuple)
```

Structured observation returned by `NozzleBase.compute_obs`.

**Fields**:

- `thrust_angle` (`Array`) - Current thrust angle(s) in radians.
- `target_angle` (`Array`) - Target thrust angle(s) in radians.
- `pressure_probes` (`Array | None`) - Wall-pressure probe values, or `None`.

## TVCSpec Objects

```python
@dataclass(frozen=True, slots=True)
class TVCSpec
```

Specification for a Thrust Vector Control environment.

**Fields**:

- `dim` (`int`) - Problem dimensionality (2 or 3).
- `grid_resolutions` (`tuple[str, ...]`) - Allowed resolution strings. Default `('coarse', 'fine')`.
- `fixed_num_actuators` (`int | None`) - Fixed actuator count (2-D nozzle). Mutually exclusive with min/max.
- `min_num_actuators` (`int | None`) - Minimum actuator count (3-D nozzle).
- `max_num_actuators` (`int | None`) - Maximum actuator count (3-D nozzle).
- `ambient_pressure` (`float`) - Ambient static pressure in Pa. Default `1e5`.
- `ambient_temperature` (`float`) - Ambient temperature in K. Default `300.0`.
- `specific_gas_constant` (`float`) - Specific gas constant in J/(kg·K). Default `287.14`.
- `specific_heat_ratio` (`float`) - Ratio of specific heats. Default `1.4`.
- `nozzle_pressure_ratio` (`float`) - Total-to-ambient pressure ratio. Default `4.6`.
- `nozzle_geometry` (`NozzleGeometry`) - Nozzle wall geometry.
- `injector_x` (`float`) - Relative axial injector position. Default `0.789`.
- `injector_width` (`float`) - Injector slot width in metres. Default `0.002032`.
- `t_end` (`float`) - Episode duration in seconds. Default `1e-2`.

#### p0

```python
@property
def p0() -> float
```

Total reservoir pressure: `nozzle_pressure_ratio * ambient_pressure`.

## TVCEnvOptions Objects

```python
@dataclass(frozen=True, slots=True)
class TVCEnvOptions
```

Parsed and validated environment options produced by `build_tvc_env_options`.

**Fields**:

- `num_actuators` (`int`)
- `secondary_pressure_ratio` (`float`)
- `resolution` (`str`)
- `ngpus` (`int`)
- `is_pressure_probes` (`bool`)
- `is_scale_observations` (`bool`)
- `target_fn` (`TargetThrustAngleFn`)

## TVCRuntimeSetup Objects

```python
@dataclass(frozen=True, slots=True)
class TVCRuntimeSetup
```

Runtime file paths and pre-loaded setup dictionaries for the JAXFluids simulation.

**Fields**:

- `env_name` (`str`) - Constructed environment name string (e.g. `'Nozzle2D_fine'`).
- `env_dir` (`Path`) - Directory containing all environment files.
- `case_setup_dict` (`dict`) - Loaded `jxf_case_setup.json` (with GPU decomposition applied).
- `numerical_setup_dict` (`dict`) - Loaded `jxf_numerical_setup.json`.
- `restart_file_path` (`Path`) - Path to `restart.h5`.

## Functions

#### build\_tvc\_env\_options

```python
def build_tvc_env_options(*,
                          env_config: dict,
                          spec: TVCSpec,
                          target_fns: dict[str, TargetThrustAngleFn],
                          cls_name: str) -> TVCEnvOptions
```

Parse and validate environment configuration against a `TVCSpec`.

**Arguments**:

- `env_config` - Raw environment configuration dictionary.
- `spec` - `TVCSpec` describing the environment constraints.
- `target_fns` - Available target-angle functions keyed by name.
- `cls_name` - Class name used in error messages.

**Returns**:

  Validated `TVCEnvOptions`.

**Raises**:

- `ValueError` - If any configuration value is outside the allowed range.

#### build\_tvc\_runtime\_setup

```python
def build_tvc_runtime_setup(*,
                            base_path: Path,
                            dim: int,
                            resolution: str,
                            ngpus: int) -> TVCRuntimeSetup
```

Load JAXFluids case and numerical setup files and return a `TVCRuntimeSetup`.

**Arguments**:

- `base_path` - Root directory that contains the environment files.
- `dim` - Dimensionality (2 or 3).
- `resolution` - Grid resolution string (e.g. `'fine'`).
- `ngpus` - Number of GPUs; sets the x-axis decomposition in `case_setup_dict`.

**Returns**:

  `TVCRuntimeSetup` with all paths validated and dicts loaded.

**Raises**:

- `FileNotFoundError` - If any required file (`jxf_case_setup.json`, `jxf_numerical_setup.json`, `restart.h5`) is missing.

#### compute\_thrust

```python
def compute_thrust(primitives: Array,
                   p_infty: float,
                   apertures_x: Array,
                   cell_centers: tuple[Array, ...],
                   cell_sizes: tuple[Array, ...]) -> tuple[Array, Array, Array]
```

Compute integrated nozzle thrust from the primitive flow state.

Uses the momentum-flux and pressure-area formulation at the nozzle exit plane:

```
F_x = mdot_e * u_e + (p_e - p_infty) * A_e
F_y = mdot_e * v_e
F_z = mdot_e * w_e
```

**Arguments**:

- `primitives` - Primitive variables array, shape `(Np, Nx, Ny, Nz)`.
- `p_infty` - Ambient static pressure (Pa).
- `apertures_x` - X-direction aperture fractions from the level-set solver.
- `cell_centers` - Tuple of 1-D coordinate arrays `(x, y, z)`.
- `cell_sizes` - Tuple of 1-D cell-size arrays `(dx, dy, dz)`.

**Returns**:

  `(thrust, mdot_throat, mdot_exit)` — thrust vector `(3,)` and scalar mass-flow rates.

#### initialize\_injector\_flux\_fn

```python
def initialize_injector_flux_fn(injector_geometry: InjectorGeometry,
                                pressure_ratios: PressureRatios,
                                p_infty: float,
                                T_infty: float,
                                specific_heat_ratio: float,
                                specific_gas_constant: float,
                                sim_manager: SimulationManager) -> Callable
```

Build the interface-flux callable for secondary injection.

Closes over the injector geometry and thermodynamic constants to produce a
JAX-compatible callable that computes mass, momentum, and energy fluxes for
each injector given the current actuator commands (values in [0, 1]).

**Arguments**:

- `injector_geometry` - Injector position, width, and count.
- `pressure_ratios` - Nozzle and secondary pressure ratios.
- `p_infty` - Ambient static pressure (Pa).
- `T_infty` - Ambient temperature (K).
- `specific_heat_ratio` - Ratio of specific heats γ.
- `specific_gas_constant` - Specific gas constant R in J/(kg·K).
- `sim_manager` - Active JAXFluids `SimulationManager` (provides domain info).

**Returns**:

  `compute_interface_flux(primitives, interface_length, normal, mesh_grid, actuator)`
  callable compatible with JAXFluids `InterfaceFluxCallablesSetup`.

#### plot\_flowfield\_3d

```python
def plot_flowfield_3d(primitives: ndarray,
                      levelset: ndarray,
                      cell_centers: tuple[ndarray, ...],
                      cell_sizes: tuple[ndarray, ...],
                      nozzle_pressure: float,
                      injector_geometry: InjectorGeometry,
                      specific_heat_ratio: float) -> pv.Plotter
```

Render a 3-D nozzle flow field using PyVista.

Produces an off-screen `pv.Plotter` with Mach-number slices, a pressure-
coloured nozzle surface, and Schlieren contours.

**Arguments**:

- `primitives` - Primitive variables array (NumPy), shape `(5, Nx, Ny, Nz)`.
- `levelset` - Level-set field (NumPy), shape `(Nx, Ny, Nz)`.
- `cell_centers` - Tuple `(x, y, z)` of 1-D coordinate arrays.
- `cell_sizes` - Tuple `(dx, dy, dz)` of 1-D size arrays.
- `nozzle_pressure` - Total nozzle pressure used for normalisation.
- `injector_geometry` - Injector position, width, and count.
- `specific_heat_ratio` - Ratio of specific heats γ.

**Returns**:

  Configured `pv.Plotter` instance (off-screen, 3000×2200 px).

#### clip\_grids

```python
def clip_grids(grid, grid_levelset)
```

Clip two `pv.RectilinearGrid` objects to the nozzle interior region.

#### plot\_slice

```python
def plot_slice(grid_levelset, grid) -> pv.Plotter
```

Compose the final PyVista scene from clipped grids.
