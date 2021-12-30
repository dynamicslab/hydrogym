FROM quay.io/dolfinadjoint/pyadjoint:latest

ENV PYTHONPATH=/home/fenics/hydrogym

# Install dependencies
# COPY requirements.txt /src/requirements.txt
# RUN pip3 install --no-cache-dir -r /src/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir --no-binary=h5py h5py && \
	pip3 install gmsh meshio

# RUN sudo pip3 git+https://github.com/barkm/torch-fenics.git@master

# Install hydrogym package
RUN mkdir hydrogym
COPY setup.py /home/fenics/hydrogym
COPY hydrogym /home/fenics/hydrogym/hydrogym

# RUN cd hydrogym && \
# 	pip3 install -e . && \
# 	cd ..
    
RUN cd /usr/local/lib/python3.6/dist-packages && \
    python3 /home/fenics/hydrogym/setup.py develop && \
    cd /home/fenics
