---
sidebar_label: nek_utils
title: hydrogym.nek.nek_lib.nek_utils
---

Collection of NEK5000 usage

#### power\_mode

```python
def power_mode(conf)
```

p096 from the `power` section of a configuration: 0 without it (or enabled: false), else the sampling.

#### read\_rea\_params

```python
def read_rea_params(path)
```

\{pkey: value\} of the mapped entries (V17_PARAM_MAPPING) of a .rea file.

#### rea\_mismatches

```python
def rea_mismatches(nek, path)
```

Mapped entries whose value in the .rea file differs from the configuration: [(pkey, attr, file, config)].

#### write\_rea\_from\_config

```python
def write_rea_from_config(conf, folder)
```

Write the configuration into the .rea file of a v17 workspace before Nek5000 is launched (Nek5000 reads the file
once at start-up). A symbolic link (to the read-only cache) is replaced by a copy first. Sets
simulation.power_mode from the power section. Returns the path of the .rea file.

## NEK\_INIT Objects

```python
class NEK_INIT()
```

#### \_\_init\_\_

```python
def __init__(nek: nek, drl: drl, rank_folder)
```

A class for initialization of NEK Dependencies
nek:[dataclass]Simulation config
drl:[dataclass]DRL config
rank_folder:[str]target folders to run drl

#### get\_Case\_Files

```python
def get_Case_Files()
```

Get required case files for running simulation
IF it is complusory, it will be rewritten no matter if the file exists
IF it is optional, it will NOT be covered if it Exist.

#### write\_SESSION\_NAME

```python
def write_SESSION_NAME()
```

Write the session name and where the code should be executed

#### rewrite\_REA\_v17

```python
def rewrite_REA_v17()
```

Re-Write parameter files for NEK version &lt;= 17.
For the controllable params, please see config.

#### rewrite\_REA\_v19

```python
def rewrite_REA_v19()
```

Write parameter files for NEK version >= 19

#### init\_restart

```python
def init_restart()
```

Copy the restart file to the target folder only if RSTART NOT EXIST

#### write\_timeSeries

```python
def write_timeSeries()
```

Write the int_pos file for the case file

