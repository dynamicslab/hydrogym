Hydrogym Reference Documentation
===================================

**Hydrogym** is an open-source library of challenge problems in data-driven modeling and
control of fluid dynamics for state-of-the-art reinforcement learning algorithms. For that purposes it exposes a number of APIs at multiple levels
to allow every user to interact with the environment in the way most befitting of their
algorithmic approach. While the current


Core Features
^^^^^^^^^^^^^

.. grid::

   .. grid-item-card::
      :columns: 12 12 12 6

      .. card:: Hierarchical Design
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Hierarchically designed for clean, industry-standard APIs with the expanded `hydrogym.env.FlowEnv` class
            extending the `gymnasium.Env` interface.

   .. grid-item-card::
      :columns: 12 12 12 6

      .. card:: Scalability
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Readily scalable experimentation infrastructure with `Ray <https://ray.io/>`_
            to support the rapid experimentation at cluster-scale.

   .. grid-item-card::
      :columns: 12 12 12 6

      .. card:: 4 Environments
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Provision of four environments for flow control with reinforcement learning
            with varying computational requirements.

   .. grid-item-card::
      :columns: 12 12 12 6

      .. card:: RL Interfaces
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Reinforcement learning interfaces to a broad set of the most popular RL frameworks
            used today.


.. note::

   While there exists a pip-install, this project is still under heavy active development with breaking changes happening with little to no lead time.


.. toctree::
   :hidden:
   :maxdepth: 3

   installation
   quickstart
   basics
   user_docs/index
   integrations/index
   examples/index
   glossary
   dev_notes/index
   changelog
   API Reference <_autosummary/hydrogym>
