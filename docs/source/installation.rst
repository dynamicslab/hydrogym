Installing Hydrogym
===================

.. _installation:

Installation
------------

You can install a fully functional version of Hydrogym via `pip` locally, as well as on the cluster of your choice:

.. code-block:: console

   $ pip install hydrogym

in the packaging of Hydrogym there are a number of inherent trade-offs, which have been made. In the technical approach we take
to our distributed backend, we have the default assumption that the virtual environment has no access to a native
`Firedrake <https://www.firedrakeproject.org>`_ installation, the default simulation engine powering the reinforcement
learning (RL) environments, and instead the distributed backend spawns environment instances with every RL-instance. For
development purposes it may often be preferable to deactivate this behaviour with

.. code-block:: python

    config.update("hydrogym_local_development", True)

which switches the behaviour to look for a Firedrake installation inside of the local virtual environment. amd hence
allows for debugging on any laptop. All other dependencies are universally handled by `pip <https://pip.pypa.io/en/stable/>`_ with
its auto-resolving dependency management.

Firedrake
---------

Linking to a tested Firedrake version Hydrogym comes packaged with a Firedrake version, for which you have to clone the
repository recursively

.. code-block:: console

    $ git clone --recursively https://github.com/dynamicslab/hydrogym.git

or in case you already have a repository copy

.. code-block:: console

    $ git submodule update --init --recursive

going into Firedrake's sub-directory one should then **begin** by letting Firedrake setting up its virtual environment
into which you then later on install Hydrogym. Following Firedrake's
`provided instructions <https://www.firedrakeproject.org/download.html>`_ we can then initiate the installation

.. code-block:: console

    $ python3 firedrake-install

where you should ascertain yourself that you are building with `MPI <https://www.open-mpi.org>`_ being enabled. If you do not enable MPI the
environment scheduler will automatically utilize the spare resources and launch the requisite number of more environments,
while each environment instance runs considerably slower. Completing the installation we have to make sure that we are
in the virtual environment instigated by Firedrake's installer

.. code-block:: console

    $ source firedrake/bin/activate

into which we can then install Hydrogym

.. code-block:: console

    $ pip install hydrogym

from which point on Hydrogym works as usual utilizing the virtual environment-provided version of Firedrake to
power its environments.

.. note::

    Where a module system is available such as on a cluster, and the module system provides a MPI-enabled version of Firedrake this should **strictly** be preferred.

Development Setup
-----------------

To use Hydrogym in development mode, we first need to install `Poetry <https://python-poetry.org>`_ onto our system. Following Poetry's installation instructions:

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


On Supercomputers/HPC-Clusters
------------------------------

On supercomputers, or hpc-clusters you will most often find a module system which already contains some of the dependencies. In that case we need to pass a number of extra arguments to our `build.py` script such that it can use those modules as dependencies for the Firedrake install. For example:

.. code-block:: console

   $ python setup.py build_ext 

The following options are available to point to system-wide installations of individual dependencies, and not have Firedrake reinstall everything:

#. Option 1:

   .. code-block:: console

      $ python setup.py build_ext --option1=/../../..

#. Option 2:
