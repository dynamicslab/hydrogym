FROM quay.io/dolfinadjoint/pyadjoint:latest

# Install cfd_gym package
RUN mkdir cfd_gym
COPY setup.py /home/fenics/cfd_gym
COPY cfd_gym /home/fenics/cfd_gym/cfd_gym

RUN cd cfd_gym && \
    pip3 install --upgrade pip && \
	pip3 install -e . && \
	cd ..