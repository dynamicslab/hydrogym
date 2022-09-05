Hydrogym Reference Documentation
===================================

**Hydrogym** is an open-source library of challenge problems in data-driven modeling and
control of fluid dynamics. For that purposes it exposes a number of APIs at multiple levels
to allow every user to interact with the environment in the way most befitting of their
algorithmic approach.


Core Features
-------------

* **Hierarchical:** Designed for analysis and controller design from a high-level black-box interface
  to low-level operator access.
   * High-level: `hydrogym.env.FlowEnv` classes implement the OpenAI `gym.env` interface.
   * Intermediate-level: Provides a typical CFD interface with `hydrogym.FlowConfig`, and `hydrogym.TransientSolver`.
   * Low-level: Enables access to linearized operators, and sparse scipy or PETSc CSR matrices.
* **Differentiable:** Adjoint-enabled for PDE-constrained optimization via pyadjoint, which extensible to PyTorch, JAX, and beyond.
* **Modeling and Analysis Tools:** Provides global stability analysis (via SLEPc) and modal decomposition (via modred).
* **Scalable:** Since the linear algebra backend is PETSc, which is fully parallelized with MPI (including pyadjoint, SLEPc, modred),
   and the Reinforcement learning backend is programmed to automatically utilize all available resources for training. 


.. note::

   This project is under heavy active development for the time being.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation


.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api


.. toctree::
   :maxdepth: 1
   :caption: Notes

   distributed_backend

