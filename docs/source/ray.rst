Setup of Ray Cluster
====================

To utilize the distributed backend to Hydrogym, wherein Hydrogym utilizes an underlying cluster
to launch as many agents as possible at the same time for faster Reinforcement/Evolutionary learning,
we require a `Ray cluster <https://docs.ray.io/en/latest/ray-core/configure.html>`_ for the framework
to parallelize over.

.. note::

    The way hydrogym is constructed at the very moment it relies heavily on CPU-resources, and
    hence prefers clusters with many available CPU cores.

Below we describe a basic setup, but there exist a great many more configuration options, such
as setting up a Ray cluster on top of a HPC-typical `SLURM <https://slurm.schedmd.com>`_ scheduler on a supercomputer. As
this is heavily being worked on inside of Ray, we refer to Ray's documentation for
`Ray cluster on Slurm <https://docs.ray.io/en/master/cluster/vms/user-guides/community/slurm.html>`_.

Before starting up the Python environment, we then need to perform a number of configuration
steps on all machines involed in our cluster.

1. All machines that are meant to be used need to contain the same Pythone environment, libraries,
    and dependencies.
2. We need to initialize the Ray controller on the head-node:

.. code-block:: console

   (.venv) $ ray start --head

3. We need to connect all worker nodes to the head-node by initiating the connection. Beware of
potential networking traffic restrictions, or blocks which could interfere with your cluster.

.. code-block:: console

    (.venv) $ ray start --address=IP_ADDRESS_OF_HEAD_NODE

Once all the preparatory steps have been performed we can initialize the Ray controller
inside of our Python script s.t. all ray-instances will use our cluster, and fully utilize
the underlying resources.

.. code-block:: python

    import ray
    ray.init(address=IP_ADDRESS_OF_HEAD_NODE)