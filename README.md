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

See `notebooks/overview.ipynb` for a quick tour of the interface