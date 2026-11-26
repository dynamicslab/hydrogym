# m-AIA host file for the CPU release image (Dockerfile.hydrogym-cpu),
# selected via MAIA_HOST_FILE (see cmake/Configure.cmake). GCC with the
# image's Open MPI (Ubuntu packages), libraries from build_maia_deps.sh in
# /opt/maia_deps.
set(MAIA_HOST "MAIACONTAINER")
set(HOST "MAIACONTAINER")
set(HOST_SUPPORTED_COMPILERS "GNU")
set(HOST_DEFAULT_COMPILER "GNU")

set(HOST_COMPILER_EXE_GNU "/usr/bin/g++")

set(INCLUDE_DIRS
    "/opt/maia_deps/parallel-netcdf-1.14.0/include"
    "/opt/maia_deps/fftw-3.3.10/include"
    "/usr/lib/x86_64-linux-gnu/openmpi/include"
    "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi"
)

set(LIBRARY_DIRS
    "/opt/maia_deps/parallel-netcdf-1.14.0/lib"
    "/opt/maia_deps/fftw-3.3.10/lib"
    "/usr/lib/x86_64-linux-gnu/openmpi/lib"
)

set(LIBRARY_NAMES "pnetcdf" "fftw3_mpi" "fftw3" "mpi_cxx" "mpi")

if(WITH_HDF5)
  set(INCLUDE_DIRS ${INCLUDE_DIRS} "/opt/maia_deps/hdf5-1.14.5/include")
  set(LIBRARY_DIRS ${LIBRARY_DIRS} "/opt/maia_deps/hdf5-1.14.5/lib")
  set(LIBRARY_NAMES ${LIBRARY_NAMES} "hdf5_hl" "hdf5" "z" "dl")
endif()

set(HOST_CONFIGURE "export LC_ALL=C" "export LANG=C")
set(HOST_DEFAULT_OPTIONS "WITH_HDF5")
set(HOST_PRESERVE_ENV "=")
