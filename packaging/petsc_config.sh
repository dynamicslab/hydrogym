source $VENV/bin/activate
cd /home/firedrake/firedrake/src/petsc

python ./configure PETSC_ARCH=linux-gnu-real-32 --with-scalar-type=real
make PETSC_DIR=/home/firedrake/firedrake/src/petsc PETSC_ARCH=linux-gnu-real-32 ${MAKEFLAGS} all

python ./configure PETSC_ARCH=linux-gnu-complex-32 --with-scalar-type=complex
make PETSC_DIR=/home/firedrake/firedrake/src/petsc PETSC_ARCH=linux-gnu-complex-32 ${MAKEFLAGS} all

python ./configure PETSC_ARCH=linux-gnu-real-64 --with-scalar-type=real
make PETSC_DIR=/home/firedrake/firedrake/src/petsc PETSC_ARCH=linux-gnu-real-64 ${MAKEFLAGS} all

python ./configure PETSC_ARCH=linux-gnu-complex-64 --with-scalar-type=complex
make PETSC_DIR=/home/firedrake/firedrake/src/petsc PETSC_ARCH=linux-gnu-complex-64 ${MAKEFLAGS} all

cd src/binding/petsc4py
PETSC_DIR=/home/firedrake/firedrake/src/petsc \
    PETSC_ARCH=default:linux-gnu-real-32:linux-gnu-complex-32:linux-gnu-real-64:linux-gnu-complex-64 \
    pip install -U --no-cache-dir .