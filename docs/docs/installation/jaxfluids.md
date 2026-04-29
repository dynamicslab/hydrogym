---
sidebar_position: 5
---

# JAX-Fluids

[JAX-Fluids](https://github.com/tumaer/JAXFLUIDS) is a fully differentiable compressible CFD solver built on JAX, developed at the Technical University of Munich. It supports high-order spatial reconstruction (WENO, TENO), multi-stage time integration, two-phase flow via level-set and diffuse-interface methods, and immersed boundary methods. Because the entire solver is written in JAX, it supports automatic differentiation through the simulation and scales to 512+ NVIDIA A100 GPUs and 2 048+ TPU-v3 cores. HydroGym's JAX-Fluids backend provides compressible jet engine control environments in 2-D and 3-D.

## Prerequisites

JAX must be installed before JAX-Fluids. Choose the installation that matches your hardware — see the [JAX installation page](./jax) for full details.

```bash
# CPU only
pip install --upgrade "jax[cpu]"

# NVIDIA GPU (CUDA 12)
pip install --upgrade "jax[cuda12]"
```

## Installing JAX-Fluids

JAX-Fluids is not published on PyPI and must be installed from source:

```bash
git clone https://github.com/tumaer/JAXFLUIDS.git
cd JAXFLUIDS
pip install .
```

For development or if you intend to modify JAX-Fluids itself, use an editable install:

```bash
pip install -e .
```

:::note[Apple Silicon (M-series Macs)]
`jaxlib` wheel availability for macOS changes frequently. If the `pip install "jax[cpu]"` step fails or produces an incompatible `jaxlib`, consult the [JAX GitHub discussion thread for M-series Macs](https://github.com/google/jax/discussions/14866) for the latest workarounds.
:::

## Installing the HydroGym Python package

After JAX-Fluids is installed, install the HydroGym JAX-Fluids extras:

```bash
pip install hydrogym[jaxfluids]
```

This adds `jax`, `jaxlib`, `flax`, `gitpython`, `h5py`, `optax`, and `pyvista` alongside the core package.

## Full installation sequence

```bash
# 1. Install JAX with the appropriate hardware target
pip install --upgrade "jax[cuda12]"   # or jax[cpu], jax[cuda13], etc.

# 2. Install JAX-Fluids from source
git clone https://github.com/tumaer/JAXFLUIDS.git
cd JAXFLUIDS && pip install . && cd ..

# 3. Install HydroGym with JAX-Fluids extras
pip install hydrogym[jaxfluids]
```

## Verify the installation

```python
import jax
import jaxfluids

print(jax.devices())
```

## Quick start

```bash
cd examples/jaxfluids/
python test_jaxfluids_env.py
```

Like the JAX backend, JAX-Fluids environments run in a single Python process — no MPI or MPMD launch required.
