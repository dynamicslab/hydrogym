FROM quay.io/dolfinadjoint/pyadjoint:latest

ENV PYTHONPATH=/home/fenics/cfd_gym

# Install dependencies
# COPY requirements.txt /src/requirements.txt
# RUN pip3 install --no-cache-dir -r /src/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir --no-binary=h5py h5py && \
	pip3 install gmsh meshio

# RUN sudo pip3 git+https://github.com/barkm/torch-fenics.git@master

# Install cfd_gym package
RUN mkdir cfd_gym
COPY setup.py /home/fenics/cfd_gym
COPY cfd_gym /home/fenics/cfd_gym/cfd_gym

# RUN cd cfd_gym && \
# 	pip3 install -e . && \
# 	cd ..
    
RUN cd /usr/local/lib/python3.6/dist-packages && \
    python3 /home/fenics/cfd_gym/setup.py develop && \
    cd /home/fenics
