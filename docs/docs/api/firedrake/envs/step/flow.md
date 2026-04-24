---
sidebar_label: flow
title: hydrogym.firedrake.envs.step.flow
---

## Step Objects

```python
class Step(FlowConfig)
```

Backwards-facing step

Notes on meshes:
- &quot;coarse&quot;: outlet at L=15 with &quot;medium&quot; resolution (81k elements)
This mesh is much faster to run, but has differences of up to ~5% in
the separation and reattachment points.  It should not be considered
&quot;validated&quot; but can be used for testing and hyperparameter tuning.
- &quot;medium&quot; - outlet at L=25 (110k elements)
- &quot;fine&quot; - outlet at L=25 (223k elements)
This is the closest to the mesh used by the reference paper
(Boujo &amp; Gallaire 2015, DOI:10.1017/jfm.2014.656)

#### FUNCTIONS

This flow needs a base flow to compute fluctuation KE

#### MAX\_CONTROL

Arbitrary... should tune this

#### TAU

Time constant for controller damping (0.01*instability frequency)

#### wall\_stress\_sensor

```python
def wall_stress_sensor(q=None)
```

Integral of wall-normal shear stress (see Barbagallo et al, 2009)

