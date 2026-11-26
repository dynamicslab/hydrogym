---
sidebar_label: io
title: hydrogym.firedrake.utils.io
---

## ParaviewCallback Objects

```python
class ParaviewCallback(CallbackBase)
```

Write fields to a ParaView (`.pvd`) time series.

#### \_\_init\_\_

```python
def __init__(interval: Optional[int] = 1,
             filename: Optional[str] = "output/solution.pvd",
             postprocess: Optional[Callable] = None)
```

**Arguments**:

- `interval` - Write every `interval` iterations.
- `filename` - Output `.pvd` file.
- `postprocess` - `postprocess(flow)` returns the functions to write (default: velocity and pressure),
  e.g. to add the vorticity.

## CheckpointCallback Objects

```python
class CheckpointCallback(CallbackBase)
```

Save the flow state with `flow.save_checkpoint`.

#### \_\_init\_\_

```python
def __init__(interval: Optional[int] = 1,
             filename: Optional[str] = "output/checkpoint.h5",
             write_mesh=True,
             write_timeseries=False)
```

**Arguments**:

- `interval` - Save every `interval` iterations.
- `filename` - Checkpoint file (HDF5).
- `write_mesh` - Also store the mesh in the file.
- `write_timeseries` - Keep every saved state (indexed by the iteration); otherwise each save overwrites the
  previous one.

## LogCallback Objects

```python
class LogCallback(CallbackBase)
```

Record scalar quantities of the flow as a time series (and optionally print and save them).

#### \_\_init\_\_

```python
def __init__(postprocess: Callable,
             nvals,
             interval: Optional[int] = 1,
             filename: Optional[str] = None,
             print_fmt: Optional[str] = None)
```

**Arguments**:

- `postprocess` - `postprocess(flow)` returns the `nvals` values to log, e.g. the force coefficients.
- `nvals` - Number of values `postprocess` returns.
- `interval` - Log every `interval` iterations.
- `filename` - Text file the time series (time and values) is written to on rank 0; `None`: not saved.
- `print_fmt` - Format string for printing `(t, *values)`; `None`: not printed.

## SnapshotCallback Objects

```python
class SnapshotCallback(CallbackBase)
```

Save the state at every `interval` iterations into one checkpoint file, e.g. for modal analysis.

#### \_\_init\_\_

```python
def __init__(interval: Optional[int] = 1,
             filename: Optional[str] = "snapshots")
```

Save snapshots as checkpoints for modal analysis.

Note that this slows down the simulation.

## GenericCallback Objects

```python
class GenericCallback(CallbackBase)
```

Call a user function `callback(iter, t, flow)` every `interval` iterations.

