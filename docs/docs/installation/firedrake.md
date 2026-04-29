---
sidebar_position: 1
---

# Firedrake

[Firedrake](https://www.firedrakeproject.org/) is an automated system for the solution of partial differential equations using the finite element method (FEM). HydroGym's Firedrake backend provides the five 2-D canonical flow-control environments (Cylinder, RotaryCylinder, Pinball, Cavity, Step) and is the recommended starting point for users new to HydroGym, because it can run on a single workstation without any GPU or large cluster.

## Option 1: Docker (recommended)

The fastest way to get a working Firedrake + HydroGym environment is to pull the pre-built Docker image:

```bash
docker pull lpaehler/hydrogym-env:stable
docker run -it lpaehler/hydrogym-env:stable
```

Inside the container, activate the Firedrake virtual environment and install HydroGym:

```bash
. /home/firedrake/firedrake/bin/activate
pip install hydrogym[firedrake]
```

The official Firedrake images (without HydroGym) are also available if you prefer to build on top of them yourself:

```bash
docker pull firedrakeproject/firedrake-vanilla-default:latest
docker run -it firedrakeproject/firedrake-vanilla-default:latest
```

Developer images tracking the latest commits are available as `firedrake-vanilla-default:dev-main` and `firedrake-vanilla-default:dev-release`.

## Option 2: Native installation

### System requirements

| | |
|---|---|
| **Supported OS** | Ubuntu (officially supported), ARM macOS, any Linux distribution |
| **Not supported** | Intel macOS (dropped), native Windows (use WSL2) |
| **Python** | 3.10 or newer |
| **macOS extras** | Homebrew and Xcode must be up to date; use the Homebrew-managed Python, not the system Python |

### Step 1 — Install system packages

Download the `firedrake-configure` helper script:

```bash
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/release/scripts/firedrake-configure
```

Use it to install the exact set of system packages required on your platform:

```bash
# Ubuntu / Debian
sudo apt install $(python3 firedrake-configure --show-system-packages)

# macOS (Homebrew)
brew install $(python3 firedrake-configure --show-system-packages)
```

On Ubuntu the list includes `build-essential`, `gfortran`, `libopenmpi-dev`, `libopenblas-dev`, `libhdf5-mpi-dev`, `cmake`, and a number of other numerical libraries. On macOS it includes the Homebrew equivalents (`openmpi`, `openblas`, `hdf5-mpi`, etc.).

### Step 2 — Build PETSc

Firedrake requires a specific PETSc revision. The `firedrake-configure` script prints the exact version and configure options:

```bash
git clone --branch $(python3 firedrake-configure --show-petsc-version) \
    https://gitlab.com/petsc/petsc.git
cd petsc
python3 ../firedrake-configure --show-petsc-configure-options | xargs -L1 ./configure
make PETSC_DIR=$(pwd) PETSC_ARCH=arch-firedrake-default all
make check
cd ..
```

To include SLEPc (required for the stability analysis scripts in the cylinder, cavity, and step examples), add `--download-slepc` to the configure step:

```bash
python3 ../firedrake-configure --show-petsc-configure-options | \
    xargs -L1 ./configure --download-slepc
export SLEPC_DIR=$PETSC_DIR/$PETSC_ARCH
```

### Step 3 — Create a virtual environment and install Firedrake

```bash
python3 -m venv venv-firedrake
. venv-firedrake/bin/activate

# Set PETSC_DIR and related variables
export $(python3 firedrake-configure --show-env)

pip cache purge
pip install --no-binary h5py 'firedrake[check]'

# Verify the installation
firedrake-check
```

### Step 4 — Install HydroGym

With the Firedrake virtual environment still active:

```bash
pip install hydrogym[firedrake]
```

This installs `gmsh` (for mesh generation) and `psutil` alongside the core HydroGym package.

### Persistent environment variables

Add the following to your `.bashrc` or `.zshrc` so that PETSC_DIR and related variables are set automatically in every new shell session:

```bash
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-firedrake-default
export HDF5_MPI=ON
```

## Optional Firedrake extras

| Extra | Install command | Adds |
|-------|----------------|------|
| SLEPc | `pip install 'firedrake[check,slepc]'` | Eigenvalue solvers for stability analysis |
| VTK | `pip install 'firedrake[check,vtk]'` | Paraview export support |
| PyTorch | `pip install 'firedrake[check,torch]' --extra-index-url https://download.pytorch.org/whl/cpu` | PyTorch integration |
| JAX | `pip install 'firedrake[check,jax]'` | JAX integration |

Always pass `--no-binary h5py` when installing Firedrake extras to avoid binary HDF5 conflicts with the MPI-enabled HDF5 built by PETSc.

## Troubleshooting

**`petscvariables` file not found after install**

The `PETSC_DIR` and `PETSC_ARCH` environment variables were not exported before `pip install`. Run:

```bash
pip cache purge
export $(python3 firedrake-configure --show-env)
pip install --no-binary h5py 'firedrake[check]'
```

**Missing symbols or import errors**

Deactivate and recreate the virtual environment from scratch, then reinstall:

```bash
deactivate
rm -r venv-firedrake
pip cache purge
python3 -m venv venv-firedrake
. venv-firedrake/bin/activate
export $(python3 firedrake-configure --show-env)
pip install --no-binary h5py 'firedrake[check]'
```

**macOS: PETSc configure fails with "Cannot use scalapack without Fortran"**

Your Homebrew or Xcode installation is out of date. Run `brew doctor` and follow the suggested remediation steps before retrying.

**macOS + Python 3.14: VTK incompatibility**

Use Python 3.13 or earlier until VTK publishes a 3.14-compatible wheel.

## Updating

```bash
# Activate the environment, then:
export $(python3 firedrake-configure --show-env)
pip install --upgrade firedrake
firedrake-clean
```

To update PETSc to the version expected by the new Firedrake release:

```bash
cd /path/to/petsc
git fetch
git checkout -b $(python3 /path/to/firedrake-configure --show-petsc-version)
make
```
