# HydroGym
Challenge problems in data-driven modeling and control of fluid dynamics

### Features
* __Hierarchical:__ Designed for analysis and controller design from a high-level black-box interface to low-level operator access
    - High-level: `hydrogym.env.FlowEnv` classes implement the OpenAI `gym.Env` interface
    - Intermediate: Typical CFD interface with `hydrogym.FlowConfig` and `hydrogym.TransientSolver` classes
    - Low-level: Access to linearized operators and sparse scipy or PETSc CSR matrices
* __Differentiable:__ Adjoint-enabled for PDE-constrained optimization via pyadjoint (extensible to PyTorch, Jax, etc... planned for future)
* __Modeling and anlysis tools:__ Global stability analysis (via SLEPc) and modal decompositions (via modred)
* __Scalable:__ Since linear algebra backend is PETSc, fully parallelized with MPI (including pyadjoint, SLEPc, modred)

# Quick Start

The easiest way to get started is with the Docker container.  Assuming you have [Docker installed](https://docs.docker.com/get-docker/), you can build and launch the image easily with the scripts in the `docker` folder:

```
cd docker && ./build.sh && ./launch.sh
```

Note that building the image will take a while the first time, but you shouldn't have to do it again unless you change the configuration.

Once you're in the docker container, the first thing to do is activate the virtual environment where all of the important packages are installed with

```
source $VENV/bin/activate
```

If you try to run something and get an error like "python: command not found" you probably missed this step.

Then you can get running in the interpreter as easy as:


```
import hydrogym as gym
env = gym.env.CylEnv(Re=100) # Cylinder wake flow configuration
for i in range(num_steps):
	action = 0.0   # Put your control law here
    (lift, drag), reward, done, info = env.step(action)
```

For more detail, check out:

* A quick tour of features in `notebooks/overview.ipynb`
* Example codes for various simulation, modeling, and control tasks in `examples`

# Flow configurations

### Periodic cylinder wake (Re=100)

![](doc/cylinder.png)

### Chaotic pinball (Re=130)

![](doc/pinball.png)


### Open cavity (Re=7500)

![](doc/cavity.png)

For the time being the cylinder wake is the most well-developed flow configuration, although the pinball should also be pretty reliable.  The cavity is in development (the boundary conditions are a little iffy and there's no actuation implemented yet) and the backwards-facing step is still planned.

# Status (5/18/22)
## What works
* Overview of features in Jupyter notebook (`notebooks/overview.ipynb`)
* Feedback control of cylinder and pinball flows (both `gym.Env` and solver interfaces)
* Newton-Krylov fixed point solver
* Time integration with a projection scheme
* Direct and adjoint global stability analysis with SLEPc
* Conversion to discrete-time LTI system
* Interface to Modred for modal analysis
* Adjoint-based optimization with Pyadjoint
* Basic test suite

## What doesn't (see issues)
* Adjoint operator construction in discrete time
* LQR control for cylinder (control design works, but blows up in DNS)

## What needs to be tested
* Cavity flow