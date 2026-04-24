---
sidebar_label: writer_int_pos
title: hydrogym.nek.nek_lib.writer_int_pos
---

## point Objects

```python
class point()
```

class defining point variables

## pset Objects

```python
class pset()
```

class containing data of the point collection

#### set\_pnt\_pos

```python
def set_pnt_pos(data, il, lpos)
```

set position of the single point

#### write\_int\_pos

```python
def write_int_pos(fname, wdsize, emode, data)
```

write point positions to the file

#### write\_channel

```python
def write_channel(path, Ret, yplus, Lx, Lz, Nx, Nz, lx1)
```

Function to write the interpolation sensing plane
Ret : Retau
yplus: wall unit distance
Lx, Lz: Domain size
Nx, Nz: Number of elements
lx1: Poly order

