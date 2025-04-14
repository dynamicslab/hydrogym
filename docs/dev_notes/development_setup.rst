Development Setup
===========================

.. note::
   
   This note is heavily outdated and to be replaced in the coming days.

   MyPy to already be weaved into this developer's note.

We strongly recommend development in the provided Docker container. The docker container
is available from `DockerHub <https://hub.docker.com/repository/docker/lpaehler/hydrogym/general>`_
, and can be pulled with

.. code-block:: console

   $ docker pull lpaehler/hydrogym-env:stable

or directly run if one seeks to develop inside of it:

.. code-block:: console

   $ docker run lpaehler/hydrogym-env:stable

If `VSCode <https://code.visualstudio.com>`_ is used for development, a .devcontainer file is provided
and allows VSCode to immediately spin up a `devcontainer <https://containers.dev>`_ with all dependencies
installed. To make full use of this you have to have the
`devcontainer extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_
installed, and upon pointing VSCode at the HydroGym directory allow it to launch into the devcontainer.
VSCode will then take care of the rest. The VSCode-devcontainer uses a specially-built Docker container
which adds a number of extra settings for VSCode. It can be examined under

.. code-block:: console

   $ docker pull lpaehler/hydrogym-devpod:stable

.. note::

    The first start of the devcontainer can take a minute depending on your internet connection as it needs to pull
    the docker container from the server before being able to launch the IDE into the container.

Depending on your internet connection, and the level on which you want to work on, you should consider which one of
the containers is the right one for your development use-case.

.. list-table:: Size of the Docker Containers
   :widths: 40 40
   :header-rows: 1

   * - Name of the Container
     - Size of the Container
   * - `hydrogym-firedrake-env <https://hub.docker.com/repository/docker/lpaehler/hydrogym-firedrake-env/general>`_
     - 6.22GB
   * - `hydrogym-env <https://hub.docker.com/repository/docker/lpaehler/hydrogym-env/general>`_
     - 9.78GB
   * - `hydrogym-devpod <https://hub.docker.com/repository/docker/lpaehler/hydrogym-devpod/general>`_
     - 18.3GB
   * - `hydrogym <https://hub.docker.com/repository/docker/lpaehler/hydrogym/general>`_
     - 22.5GB

.. note::

   The startup of the dev container can take up to 20 minutes the first time, but will cache the container after which the
   startup will take less than a minute.

Currently the dev container suffers from a slight misconfiguration in the post-deployment setting, as such one needs to run
the following commands before developing inside of the container:

.. code-block:: console

   $ source /home/firedrake/firedrake/bin/activate
   $ pip install -e .

For more information regarding the way devcontainer works please see
`Microsoft's documentation <https://code.visualstudio.com/docs/devcontainers/containers>`_.
