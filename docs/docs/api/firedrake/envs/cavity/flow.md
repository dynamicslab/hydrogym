---
sidebar_label: flow
title: hydrogym.firedrake.envs.cavity.flow
---

## Cavity Objects

```python
class Cavity(FlowConfig)
```

#### FUNCTIONS

This flow needs a base flow to compute fluctuation KE

#### TAU

Time constant for controller damping (0.01*instability frequency)

#### wall\_stress\_sensor

```python
def wall_stress_sensor(q=None)
```

Integral of wall-normal shear stress (see Barbagallo et al, 2009)

