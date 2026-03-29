<p align="center">
<a rel="nofollow"><img alt="HydroGym Logo" src="docs/_static/imgs/logo.svg"></a>
</p>

<p align="center">
<a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
<a href="https://python.org/"><img alt="Language: Python" src="https://img.shields.io/badge/language-Python-orange.svg"></a>
<a href="https://spdx.org/licenses/MIT.html"><img alt="License HydroGym" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
<a href="https://join.slack.com/t/hydrogym/shared_invite/zt-27u914dfn-UFq3CkaxiLs8dwZ_fDkBuA"><img alt="Slack" src="https://img.shields.io/badge/slack-hydrogym-brightgreen.svg?logo=slack"></a>
</p>



# About this Package

__IMPORTANT NOTE: This package is still ahead of an official public release, so consider anything here as an early beta. In other words, we're not guaranteeing any of this is working or correct yet. Use at your own risk__

HydroGym is an open-source library of challenge problems in data-driven modeling and control of fluid dynamics.
It is roughly designed as an abstract interface for control of PDEs that is compatible with typical reinforcement learning APIs
(in particular Ray/RLLib and Gymnasium) along with specific numerical solver implementations for some canonical flow control problems.
Environments are implemented using two solver backends: [Firedrake](https://www.firedrakeproject.org/) (finite element) and m-AIA (MPI-based).

## Features
* __Hierarchical:__ Designed for analysis and controller design **from a high-level black-box interface to low-level operator access**
    - High-level: `hydrogym.FlowEnv` classes implement the Gymnasium `Env` interface
    - Intermediate: Typical CFD interface with `hydrogym.FlowConfig` and `hydrogym.TransientSolver` classes
    - Low-level: Access to linearized operators and sparse scipy or PETSc CSR matrices
* __Modeling and analysis tools:__ Global stability analysis (via SLEPc) and modal decompositions (via modred)
* __Scalable:__ Individual environments parallelized with MPI with a **highly scalable [Ray](https://github.com/ray-project/ray) backend reinforcement learning training**.

# Installation

HydroGym supports two CFD solver backends: **Firedrake** (finite element) and **Maia** (m-AIA with MPI). Both backends are fully integrated and ready to use.

## Quick Installation

### Unified Installation (Recommended)

HydroGym now includes all dependencies by default. Simply install the system MPI library and HydroGym:

```bash
# Install system MPI library
sudo apt-get install libopenmpi-dev openmpi-bin  # Ubuntu/Debian
# OR: brew install open-mpi  # macOS

# Install HydroGym (includes Maia support)
pip install hydrogym
```

### With Firedrake Backend

To use Firedrake environments, install Firedrake first using its native [Python installation](https://www.firedrakeproject.org/install.html#installing-firedrake-using-pip):

```bash
# Follow Firedrake pre-install instructions in the pip guide above

# Install Firedrake
pip install firedrake

# Install HydroGym
pip install hydrogym
```

## Development Installation

To install from source:

```bash
git clone https://github.com/dynamicslab/hydrogym.git
cd hydrogym

# Install git-lfs for mesh files
git lfs install && git lfs fetch --all

# Install HydroGym
pip install .
# OR for development: poetry install
```

At which point you are ready to run HydroGym locally.

# Quickstart Guide

 Having installed Hydrogym into our virtual environment experimenting with Hydrogym is as easy as starting the Python interpreter
 
 ```bash
 python
 ```
 
 and then setting up a first Hydrogym environment instance
 
```python
import hydrogym.firedrake as hgym
env = hgym.FlowEnv({"flow": hgym.Cylinder}) # Cylinder wake flow configuration
for i in range(num_steps):
    action = 0.0   # Put your control law here
    (lift, drag), reward, done, info = env.step(action)
```

To test that you can run individual environment instances in a multithreaded fashion, run the steady-state Newton solver on the cylinder wake with 4 processors:

```bash
cd /path/to/hydrogym/examples/cylinder
mpiexec -np 4 python pd-control.py
```

For more detail, check out:

* A quick tour of features in `notebooks/overview.ipynb`
* Example codes for various simulation, modeling, and control tasks in `examples`
* The [ReadTheDocs](https://hydrogym.readthedocs.io/en/latest/)

