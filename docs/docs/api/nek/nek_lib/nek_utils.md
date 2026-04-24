---
sidebar_label: nek_utils
title: hydrogym.nek.nek_lib.nek_utils
---

Collection of NEK5000 usage

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

Write parameter files for NEK version &gt;= 19

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

