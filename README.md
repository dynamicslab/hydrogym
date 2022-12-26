<p align="center">
	<a rel="nofollow">	
		<img src="docs/source/_static/imgs/logo.svg" />
	</a>
</p>

# About this Package

HydroGym is an open-source library of challenge problems in data-driven modeling and control of fluid dynamics.

## Features
* __Hierarchical:__ Designed for analysis and controller design **from a high-level black-box interface to low-level operator access**
    - High-level: `hydrogym.env.FlowEnv` classes implement the OpenAI `gym.Env` interface
    - Intermediate: Typical CFD interface with `hydrogym.FlowConfig` and `hydrogym.TransientSolver` classes
    - Low-level: Access to linearized operators and sparse scipy or PETSc CSR matrices
* __Modeling and anlysis tools:__ Global stability analysis (via SLEPc) and modal decompositions (via modred)
* __Scalable:__ Individual environments parallelized with MPI with a **highly scalable [Ray](https://github.com/ray-project/ray) backend reinforcement learning training**.

# Installation

To begin using Hydrogym we can install its latest release via [PyPI](https://pypi.org/project/hydrogym/) with pip

```bash
pip install hydrogym
```

which provides the core functionality, and is able to launch reinforcement learning training on a Ray-cluster without an underlying Firedrake install. If you want to play around with Hydrogym locally on e.g. your laptop, we recommend a local Firedrake install. The instructions for which can be found in the [Installation Docs](https://hydrogym.readthedocs.io/en/latest/installation.html).

# Quickstart Guide

 Having installed Hydrogym into our virtual environment experimenting with Hydrogym is as easy as starting the Python interpreter
 
 ```bash
 python
 ```
 
 and then setting up a Hydrogym environment instance
 
```python
import hydrogym as hgym
env = hgym.env.CylEnv(Re=100) # Cylinder wake flow configuration
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

# Flow configurations

There are currently a number of main flow configurations, the most prominent of which are:

- Periodic cyclinder wake at Re=100
- Chaotic pinball at Re=130
- Open cavity at Re=7500

with visualizations of the flow configurations available in the [docs](docs/FlowConfigurations.md).
