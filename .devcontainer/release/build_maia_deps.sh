#!/bin/bash
# Parallel-NetCDF 1.14.0, parallel HDF5 1.14.5 and FFTW 3.3.10 (MPI + threads)
# for m-AIA, installed below MAIA_LIB_HOME. Used by the release images
# (Dockerfile.hydrogym-cuda, Dockerfile.hydrogym-cpu).
#
# Environment:
#   MAIA_LIB_HOME  install prefix (default /opt/maia_deps)
#   MPI_CC, MPI_CXX, MPI_FC  MPI compiler wrappers to build with

set -euo pipefail

: "${MPI_CC:?}" "${MPI_CXX:?}" "${MPI_FC:?}"
MAIA_LIB_HOME="${MAIA_LIB_HOME:-/opt/maia_deps}"
jobs="${JOBS:-$(compute_parallelism.sh 2>/dev/null || nproc)}"
work="$(mktemp -d)"
cd "${work}"

export CC="${MPI_CC}" CXX="${MPI_CXX}" FC="${MPI_FC}" F77="${MPI_FC}" MPICC="${MPI_CC}" MPICXX="${MPI_CXX}"
export CFLAGS="-O3 -fPIC" CXXFLAGS="-O3 -fPIC" FCFLAGS="-O3 -fPIC" FFLAGS="-O3 -fPIC"

wget -q https://parallel-netcdf.github.io/Release/pnetcdf-1.14.0.tar.gz
tar -xzf pnetcdf-1.14.0.tar.gz
(cd pnetcdf-1.14.0 && \
    ./configure --prefix="${MAIA_LIB_HOME}/parallel-netcdf-1.14.0" --enable-shared --disable-fortran && \
    make -j"${jobs}" && make install)

wget -q https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_1.14.5.tar.gz -O hdf5-1.14.5.tar.gz
tar -xzf hdf5-1.14.5.tar.gz
(cd hdf5-hdf5_1.14.5 && \
    ./configure --prefix="${MAIA_LIB_HOME}/hdf5-1.14.5" --enable-parallel --enable-shared --disable-fortran && \
    make -j"${jobs}" && make install)

wget -q https://www.fftw.org/fftw-3.3.10.tar.gz
tar -xzf fftw-3.3.10.tar.gz
(cd fftw-3.3.10 && \
    ./configure --prefix="${MAIA_LIB_HOME}/fftw-3.3.10" --enable-shared --enable-mpi --enable-openmp --enable-threads && \
    make -j"${jobs}" && make install)

cd / && rm -rf "${work}"
echo "=== m-AIA libraries installed in ${MAIA_LIB_HOME} ==="
