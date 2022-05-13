# HydroGym
Challenge problems in data-driven modeling and control of fluid dynamics

### Quick hits
* __Hierarchical:__ Designed for analysis and controller design from a high-level black-box interface to low-level operator access
    - High-level: `hydrogym.env.FlowEnv` classes implement the OpenAI `gym.Env` interface
    - Intermediate: Typical CFD interface with `hydrogym.FlowConfig` and `hydrogym.TransientSolver` classes
    - Low-level: Access to linearized operators and sparse scipy or PETSc CSR matrices
* __Differentiable:__ Adjoint-enabled for PDE-constrained optimization via pyadjoint (extensible to PyTorch, Jax, etc... planned for future)
* __Modeling and anlysis tools:__ Global stability analysis (via SLEPc) and modal decompositions (via modred)
* __Scalable:__ Since linear algebra backend is PETSc, fully parallelized with MPI (including pyadjoint, SLEPc, modred)

# Status (5/13/22)
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
* Modred interface in parallel

## What needs to be tested
* Cavity flow