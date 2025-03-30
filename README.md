<p align="center">
<a rel="nofollow"><img alt="HydroGym Logo" src="docs/_static/imgs/logo.svg"></a>
</p>

<p align="center">
<a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
<a href="https://python.org/"><img alt="Language: Python" src="https://img.shields.io/badge/language-Python-orange.svg"></a>
<a href="https://spdx.org/licenses/MIT.html"><img alt="License WarpX" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
<a href="https://join.slack.com/t/hydrogym/shared_invite/zt-27u914dfn-UFq3CkaxiLs8dwZ_fDkBuA"><img alt="Slack" src="https://img.shields.io/badge/slack-hydrogym-brightgreen.svg?logo=slack"></a>
</p>



# About this Package

__IMPORTANT NOTE: This package is still ahead of an official public release, so consider anything here as an early beta. In other words, we're not guaranteeing any of this is working or correct yet. Use at your own risk__

HydroGym is an open-source library of challenge problems in data-driven modeling and control of fluid dynamics.
It is roughly designed as an abstract interface for control of PDEs that is compatible with typical reinforcement learning APIs
(in particular Ray/RLLib and OpenAI Gym) along with specific numerical solver implementations for some canonical flow control problems.
Currently these "environments" are all implemented using the [Firedrake](https://www.firedrakeproject.org/) finite element library.

## Features
* __Hierarchical:__ Designed for analysis and controller design **from a high-level black-box interface to low-level operator access**
    - High-level: `hydrogym.env.FlowEnv` classes implement the OpenAI `gym.Env` interface
    - Intermediate: Typical CFD interface with `hydrogym.FlowConfig` and `hydrogym.TransientSolver` classes
    - Low-level: Access to linearized operators and sparse scipy or PETSc CSR matrices
* __Modeling and analysis tools:__ Global stability analysis (via SLEPc) and modal decompositions (via modred)
* __Scalable:__ Individual environments parallelized with MPI with a **highly scalable [Ray](https://github.com/ray-project/ray) backend reinforcement learning training**.

# Reinforcement Learning Framework Support

HydroGym aims to support a commensurate variety of reinforcement learning frameworks for all use-cases. As such, we currently have the following support in place:

| Algorithm    | Stable Baselines 3 | CleanRL | RLLib | Acme | TorchRL | LeanRL |
| -------- | ------- | ------- | ------ | ------- | ----- | ---- |
| PPO    |   |   | $\checkmark$ |   |   |   |
| A2C    |   |   |   |   |   |   |
| DQN    |   |   |   |   |   |   |
| SAC    |   |   |   |   |   |   |
| DDPG   |   |   |   |   |   |   |
| TD3    |   |   |   |   |   |   |


# Installation

By design, the core components of Hydrogym are independent of the underlying solvers in order to avoid custom or complex
third-party library installations.
This means that the latest release of Hydrogym can be simply installed via [PyPI](https://pypi.org/project/hydrogym/):

```bash
pip install hydrogym
```

> BEWARE: The pip-package is currently behind the main repository, and we strongly urge users to build HydroGym
>         directly from the source code. Once we've stabilized the package, we will update the pip package in turn.

However, the package assumes that the solver backend is available, so in order to run simulations locally you will
need to _separately_ ensure the solver backend is installed (again, currently all the environments are implemented with Firedrake).
Alternatively (and this is important for large-scale RL training), the core Hydrogym package can (or will soon be able to) launch reinforcement learning training on a Ray-cluster without an underlying Firedrake install.
For more information and suggested approaches see the [Installation Docs](https://hydrogym.readthedocs.io/en/latest/installation.html).

To add HydroGym to an existing Firedrake installation, and install from the repository, run:

```bash
git clone https://github.com/dynamicslab/hydrogym.git
cd hydrogym
pip install .
```

As the mesh files are stored in [git large file storage](https://git-lfs.github.com/), you will need to install git-lfs
to download the mesh files.

```bash
git lfs install && git lfs fetch --all
```

At which point you are ready to run HydroGym locally.

# Quickstart Guide

Having installed Hydrogym into our virtual environment experimenting with Hydrogym is as easy as starting the Python interpreter
 
```bash
python
```

where we have to begin by defining a logging Callback

```python
import hydrogym
log = hydrogym.firedrake.utils.io.LogCallback(
    postprocess=lambda flow: flow.get_observations(),
    nvals=2,
    interval=1,
    print_fmt="t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}",
    filename=None,
)
```
 
at which point we are able to define our first flow control environment through a dictionary

```python
flow_dict = {
    "flow": hydrogym.firedrake.Cylinder,
    "flow_config": {
        "Re": 100,
        "mesh": "medium",
    },
    "solver": hydrogym.firedrake.SemiImplicitBDF,
    "solver_config": {
        "dt": 1e-2,
    },
    "callbacks": [log],
    "max_steps": 10000,
}
```

and are able to subsequently define the [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) algorithm for which we import [RLLib](https://docs.ray.io/en/latest/rllib/index.html) and configure a simple proximal policy optimization (PPO) agent:

```python
from hydrogym.core import FlowEnv
from ray.rllib.algorithms.ppo import PPOConfig
config = (
    PPOConfig()
    .environment(
        FlowEnv,
        env_config=flow_dict,
    )
    .env_runners(
        num_env_runners=2,
        sample_timeout_s = 300.0
    )
    .training(
        lr=0.0002,
        train_batch_size_per_learner=2000,
        num_epochs=10,
    )
)
```

which then has to be constructed, and can then be trained

```python
from pprint import pprint
ppo = config.build_algo()

for _ in range(4):
    pprint(ppo.train())
```

To continue on from this very first example, and dive into more details, check out:

* A quick tour of features in `colabs/overview.ipynb`
* Example codes for various simulation, modeling, and control tasks in `examples`
* The [Docs](https://hydrogym.readthedocs.io/en/latest/)

# Flow configurations

There are currently a number of main flow configurations, the most prominent of which are:

- Periodic cylinder wake at Re=100
- Chaotic pinball at Re=130
- Open cavity at Re=7500
- Backwards-facing step at Re=600

with visualizations of the flow configurations available in the [docs](docs/FlowConfigurations.md).
