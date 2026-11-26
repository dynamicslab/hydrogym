---
sidebar_label: layout
title: hydrogym.nekrs.layout
---

Probe and actuator layouts from the YAML description.

A position block is a mapping with a `source`:


and an optional `sort: [x, z, ...]` (stable lexicographic sort by those coordinates, first key primary).
```text
  list             points: [[x, y, z], ...]
  file             path: positions.npy | .csv (columns x, y, z) | whitespace text with three columns |
                   .npz (array xyz [n, 3]; further [n, ...] arrays, e.g. normal, become extra columns)
  grid             x, y, z: a value, a list, or {start, stop, num}; order: [slowest, ..., fastest] (default x, y, z)
  boundary_nodes   boundary: bid, nodes: all | facindf | unique   (GLL nodes of the boundary, from the solver;
                   'all' keeps the copies of shared edge/corner nodes, as the released HydroGym channel)
  actuator_offset  offset: [dx, dy, dz] and/or normal_distance: d   (probes only: actuator centre + offset; d is a
                   value or one value per actuator, e.g. a wall-unit distance resolved by the environment)
```

#### positions

```python
def positions(spec: Dict,
              iface=None,
              actuators: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]
```

Return (xyz[n, 3], extra columns) for a position block.

#### footprint\_arrays

```python
def footprint_arrays(spec, n: int) -> Tuple[np.ndarray, np.ndarray]
```

Footprint code and widths (sigma or half width per axis, 0 = unbounded) for n actuators.

#### direction\_arrays

```python
def direction_arrays(spec,
                     n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Direction code, vector and angle [rad]: `'normal' | {vector: [..]} | {angle: deg, towards: [..]}`.

#### group\_ids

```python
def group_ids(xyz: np.ndarray,
              axes,
              tol: Optional[float] = None) -> np.ndarray
```

Label per point such that points whose coordinates along `axes` agree within tol share a label (e.g. the
spanwise lines of an extruded mesh for axes [x, y]). Groups are split axis by axis at gaps larger than tol,
so round-off between element copies of the same GLL position does not separate them. Default tol: 1e-6 of
the largest coordinate range.

#### group\_mean

```python
def group_mean(values: np.ndarray,
               labels: np.ndarray,
               weights: Optional[np.ndarray] = None) -> np.ndarray
```

Weighted mean of values[n] or values[n, k] over each label, returned per point.

