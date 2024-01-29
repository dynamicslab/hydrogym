Container-based Development
===========================

We strongly recommend development in the provided Docker container. The docker container
is available from `DockerHub <https://hub.docker.com/repository/docker/lpaehler/hydrogym/general>`_
, and can be pulled with

.. code-block:: console

   $ docker pull lpaehler/hydrogym:stable

or directly run if one seeks to develop inside of it:

.. code-block:: console

   $ docker run lpaehler/hydrogym:stable

If `VSCode <https://code.visualstudio.com>`_ is used for development, a .devcontainer file is provided
and allows VSCode to immediately spin up a `devcontainer <https://containers.dev>`_ with all dependencies
installed. To make full use of this you have to have the
`devcontainer extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_
installed, and upon pointing VSCode at the HydroGym directory allow it to launch into the devcontainer.
VSCode will then take care of the rest.

.. note::

    The first start of the devcontainer can take a minute depending on your internet connection as it needs to pull
    the docker container from the server before being able to launch the IDE into the container.

For more information regarding the way devcontainer works please see
`Microsoft's documentation <https://code.visualstudio.com/docs/devcontainers/containers>`_.
