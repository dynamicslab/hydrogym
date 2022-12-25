Installation Instructions
=========================

.. _installation:

Installation
------------

You can install Hydrogym via `pip` for local development without its development dependencies:

.. code-block:: console

   $ pip install hydrogym


Development
-----------

To use Hydrogym in development mode, we first need to install [Poetry](https://python-poetry.org) onto our system. Following Poetry's installation instructions:

.. code-block:: console

   $ curl -sSL https://install.python-poetry.org | python3 -

and follow the instructions to have Poetry on our system. We can then perform a development install of Hydrogym by running

.. code-block:: console

   $ poetry install

to get a virtual environment for testing or debugging we then have to run:

.. code-block:: console

   $ poetry shell

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
