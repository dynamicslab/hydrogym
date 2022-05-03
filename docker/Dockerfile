FROM firedrakeproject/firedrake:latest AS hydrogym

USER root
RUN apt-get upgrade && apt-get update && \
        apt-get install -y libhdf5-dev libxcursor-dev libxinerama1

USER firedrake

WORKDIR /home/firedrake

ENV VENV=/home/firedrake/firedrake
ENV PATH="$VENV/bin:/home/firedrake/.local/bin:$PATH"

# Install dependencies
# COPY requirements.txt /src/requirements.txt
# RUN pip3 install --no-cache-dir -r /src/requirements.txt
RUN pip3 install --upgrade pip setuptools && \
    pip3 install --no-cache-dir --no-binary=h5py h5py && \
	pip3 install gmsh meshio dmsuite jupyter

# RUN sudo pip3 git+https://github.com/barkm/torch-fenics.git@master

# # # Install hydrogym package
COPY --chown=firedrake . /home/hydrogym
RUN bash -c ". $VENV/bin/activate && pip install -e /home/hydrogym"

# Install an iPython kernel for firedrake (https://github.com/firedrakeproject/firedrake/blob/master/docker/Dockerfile.jupyter)
RUN bash -c ". $VENV/bin/activate && pip install jupyterhub ipykernel notebook ipywidgets mpltools nbformat nbconvert"
RUN bash -c ". $VENV/bin/activate && jupyter nbextension enable --py widgetsnbextension --sys-prefix"

# Final configuration
ENV OMP_NUM_THREADS=1
WORKDIR /home
CMD ["/bin/bash"]