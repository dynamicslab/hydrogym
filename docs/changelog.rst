Changelog
=========

Version 0.1.3.0
---------------

* Updates the build system from poetry to uv, in the process of which rewriting the `pyproject.toml`
* Adds group based dependencies to the pyproject.toml, most notably of which for `rllib` i.e.

.. code-block:: bash

    pip install "hydrogym[rllib]"

installs the required additional dependencies now to run `rllib` with HydroGym out of the box.

* Major rewrite of the Devcontainer configuration, and the way container-packaging is done for HydroGym. The custom-built containers are fully deprecated in favor of a thin user-layer shim over the official Firedrake container, which gets pulled by the devcontainer configuration. In line with modern devcontainer convention HydroGym is **not** auto-installed into the environment anymore, but needs to be installed through either

.. code-block:: bash

    pip install -e .

or 
`Link text <https://domain.invalid/>`_

.. code-block:: bash

    pip install ".[rllib]"

* The core abstraction has been updated to convert from the old legacy gym interface to the modern `gymnasium <https://gymnasium.farama.org>`_ interface. This introduces the _truncation_ property, which we do not utilize at the current time as we are handling time limits through the reset interface, but might look to expand upon in the future.
* The rllib integration has been updated to the modern rllib interfaces, and is working seamlessly again with provided examples for a PPO training in rllib, and the tuning of a RL agent with Ray Tune.
* Firedrake imports are now wrapped in a `try` construct to gracefully return an error to the user in case Firedrake is not found in the current environment, but needed by HydroGym. To give an example of the construct:

.. code-block:: python

    try:
        import firedrake as fd
    except ImportError as e:
        raise DependencyNotInstalled(
            "Firedrake is not installed, consult `https://www.firedrakeproject.org/install.html` for installation instructions."
        ) from e

* Most examples have been deprecated as they are out of date now, these will be brought back step by step, making sure that they work as intended.
* The Jupyter notebooks have been converted into a `colab` folder, in which they can still be launched with Jupyter, but going beyond the previous state, can be auto-launched into `Google Colab <https://colab.research.google.com>`_ using the groundwork laid by `FEM-on-Colab <https://fem-on-colab.github.io>`_.
* 2 new GitHub actions have been added, one to test-build the docs whenever they see updates, and one converting the yapf-formatting to ruff-formatting s.t. we only depend on one tool for lifting, as well as formatting now (ruff).
* Extending documentation
    * Fixing the Quickstart in the Readme
    * Addition of a RL framework support matrix to the Readme
    * Addition of a changelog to the docs
    * Extension of the readthedocs build with the examples
