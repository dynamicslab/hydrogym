Hydrogym Reference Documentation
===================================

**Hydrogym** is an open-source library of challenge problems in data-driven modeling and
control of fluid dynamics for state-of-the-art reinforcement learning algorithms. For that purposes it exposes a number of APIs at multiple levels
to allow every user to interact with the environment in the way most befitting of their
algorithmic approach. While the current


Core Features
-------------

* **Hierarchical:** Designed for analysis and controller design from a high-level black-box interface
  to low-level operator access.
   * High-level: `hydrogym.env.FlowEnv` classes implement the OpenAI `gym.env` interface.
   * Intermediate-level: Provides a typical CFD interface with `hydrogym.FlowConfig`, and `hydrogym.TransientSolver`.
   * Low-level: Enables access to linearized operators, and sparse scipy or PETSc CSR matrices.
* **Modeling and Analysis Tools:** Provides global stability analysis (via SLEPc) and modal decomposition (via modred).
* **Scalable:** Individual environments parallelized with MPI with a **highly scalable** `Ray <https://github.com/ray-project/ray>`_ **backend reinforcement learning training.**


.. note::

   While there exists a pip-install, this project is still under heavy active development with breaking changes happening with little to no lead time.


.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   quickstart
   basics
   user_docs/index
   integrations/index
   glossary
   dev_notes/index
   api
