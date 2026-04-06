---
sidebar_label: modeling
title: hydrogym.firedrake.utils.modeling
---

#### petsc\_to\_scipy

```python
def petsc_to_scipy(petsc_mat)
```

Convert the PETSc matrix to a scipy CSR matrix

#### system\_to\_scipy

```python
def system_to_scipy(sys)
```

Convert the LTI system tuple (A, M, B) to scipy/numpy arrays

#### snapshots\_to\_numpy

```python
@ignore_deprecation_warnings
def snapshots_to_numpy(flow, filename, save_prefix, m)
```

Load from CheckpointFile in `filename` and project to the mesh in `flow`

