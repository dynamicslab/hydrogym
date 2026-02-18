Installing Hydrogym
===================

.. _installation:

Installation
------------

**NOTE:** We are still pre-release and several components (in particular the distributed workflow) have not been fully merged.

You can install Hydrogym via `pip` locally, as well as on the cluster of your choice:

.. code-block:: console

   $ pip install hydrogym

in the packaging of Hydrogym there are a number of inherent trade-offs related to the fact that powerful simulation
backends (e.g. `Firedrake <https://www.firedrakeproject.org>`_) often require more complex configuration or builds
than typical Python packages, making them difficult to install with typical approaches to dependencies.
We avoid this by splitting Hydrogym into a "core" component with the basic API and minimal dependencies, and 
specific environment dependencies which assume that the backend is available.

There are basically two routes to making this work: *local* and *distributed*.  For a local installation (e.g. on a
laptop or workstation) **you have to install the Firedrake into the virtual environment in addition to Hydrogym**.
For a distributed workflow (e.g. RL training on a cluster), we don't assume that the virtual environment has access
to a Firedrake installation, and instead the distributed backend spawns environment instances with every RL-instance.
For development purposes it may often be preferable to deactivate this behaviour with

.. code-block:: python

    config.update("hydrogym_local_development", True)

which switches the behaviour to look for a Firedrake installation inside of the local virtual environment. and hence
allows for debugging on any laptop. All other dependencies are handled as usual by `pip <https://pip.pypa.io/en/stable/>`_.

Solver Backends
---------------

Hydrogym now supports two CFD solver backends:

- **Firedrake**: Finite element solver (traditional backend)
- **Maia**: m-AIA solver with MPI support (new in v0.2+)

You can install either or both depending on your needs.

Firedrake Installation
----------------------

As of 2025, Firedrake can be installed via pip, making the installation process much simpler than before!

**Method 1: Pip Installation (Recommended for 2025+)**

.. code-block:: console

    # Install system dependencies first
    # Ubuntu/Debian:
    $ sudo apt-get update
    $ sudo apt-get install build-essential python3-dev libopenmpi-dev openmpi-bin

    # macOS:
    $ brew install open-mpi gcc

    # Create a virtual environment
    $ python3 -m venv hydrogym-env
    $ source hydrogym-env/bin/activate

    # Install Firedrake
    $ pip install firedrake

    # Configure environment variables
    $ firedrake-configure

    # Install Hydrogym with Firedrake support
    $ pip install hydrogym

**Method 2: Traditional Installation Script**

For a more controlled installation or if pip method fails, use the traditional script:

.. code-block:: console

    $ git clone --recursive https://github.com/dynamicslab/hydrogym.git
    $ cd third_party/firedrake/scripts
    $ python3 firedrake-install --venv-name=path/to/venv
    $ source path/to/venv/bin/activate
    $ cd ../../.. && pip install .

.. note::

    **Important:** After installing Firedrake, run ``export OMP_NUM_THREADS=1`` to accelerate solves.
    Where a module system is available (e.g., on a cluster), prefer using a pre-installed MPI-enabled Firedrake module.

Maia Solver (Included by Default)
----------------------------------

The Maia solver backend with HuggingFace Hub integration is now included by default in HydroGym!

All you need to do is install the system MPI library:

.. code-block:: console

    # Install system MPI library (required for mpi4py)
    # Ubuntu/Debian:
    $ sudo apt-get install libopenmpi-dev openmpi-bin

    # macOS:
    $ brew install open-mpi

    # Install Hydrogym (Maia support included)
    $ pip install hydrogym

**Included Dependencies:**

- ``gymnasium`` - Modern Gym API (replaces old gym)
- ``mpi4py`` - MPI communication with m-AIA solver
- ``omegaconf`` - Configuration management
- ``einops`` - Tensor operations
- ``huggingface-hub`` - Environment data management
- ``pandas``, ``scipy`` - Data handling

Quick Installation Guide
------------------------

**For Maia users (simplest):**

.. code-block:: console

    $ sudo apt-get install libopenmpi-dev  # or brew install open-mpi on macOS
    $ pip install hydrogym

**For Firedrake users:**

.. code-block:: console

    $ pip install firedrake
    $ firedrake-configure
    $ pip install hydrogym

**For both Firedrake and Maia:**

.. code-block:: console

    $ sudo apt-get install libopenmpi-dev openmpi-bin
    $ pip install firedrake
    $ firedrake-configure
    $ pip install hydrogym

Development Setup
-----------------

For developing Hydrogym we use `Poetry <https://python-poetry.org>`_ for cleaner dependency management. Following Poetry's installation instructions:

.. code-block:: console

   $ curl -sSL https://install.python-poetry.org | python3 -

and follow the instructions to have Poetry on our system. We can then perform a development install of Hydrogym by running

.. code-block:: console

   $ poetry install

to get a virtual environment for testing or debugging we then have to run:

.. code-block:: console

   $ poetry shell

to test a build of the package we then run a build-process inside of Poetry's isolated environment

.. code-block:: console

    $ poetry build

with the development often requiring debugging work with notebooks, and their implied dependencies, these package requirements have been moved out to their own "dev" dependency list and can be installed with

.. code-block:: console

    $ poetry install --with dev

If you feel unsure about the structure of the dependencies, you can always inspect the dependencies with

.. code-block:: console

    $ poetry show --tree
