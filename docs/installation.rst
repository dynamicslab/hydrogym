Installing Hydrogym
===================

.. _installation:

Installation of HydroGym
------------------------

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

Firedrake (local installation)
------------------------------

For a local installation Firedrake and Hydrogym must be installed independently into the same virtual environment.
This could be done by either following the `Firedrake installation instructions <https://www.firedrakeproject.org/download.html>`_
and then pip-installing Hydrogym into the Firedrake virtual environment, or by cloning the Hydrogym source and installing from there.
These instructions describe the latter route.

Hydrogym comes packaged with a tested Firedrake version, for which you have to clone the repository recursively

.. code-block:: console

    $ git clone --recursive https://github.com/dynamicslab/hydrogym.git

or in case you already have a repository copy

.. code-block:: console

    $ git submodule update --init --recursive

Optionally, you could get a newer version of Firedrake by running `git pull origin` within the Firedrake submodule.

The next step is to install Firedrake and let it create its virtual environment into which you can install Hydrogym:

.. code-block:: console

    $ cd third_party/firedrake/scripts
    $ python3 firedrake-install --venv-name=path/to/venv
    $ source path/to/venv/bin/activate

where you should be sure that you are building with `MPI <https://www.open-mpi.org>`_ being enabled. If you do not enable MPI the
environment scheduler will automatically utilize the spare resources and launch the requisite number of more environments,
while each environment instance runs considerably slower.

into which we can then change directories back to the Hydrogym root and install as usual

.. code-block:: console

    $ cd ../../.. && pip install .

(Or use `pip install -e .` to make the installation editable).  From then on Hydrogym works
with the virtual environment-provided version of Firedrake.  As suggested by the Firedrake warning
you should also run `export OMP_NUM_THREADS=1` which will considerably accelerate the solves.

.. note::

    Where a module system is available such as on a cluster, and the module system provides a MPI-enabled version of Firedrake this should **strictly** be preferred.

Development Setup (deprecated)
------------------------------

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
