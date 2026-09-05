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

