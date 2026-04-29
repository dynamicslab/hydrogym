---
sidebar_position: 4
---

# JAX

[JAX](https://jax.readthedocs.io/) is a high-performance array computing library from Google that combines NumPy-compatible array operations with automatic differentiation, JIT compilation via XLA, and native support for CPUs, GPUs, and TPUs. HydroGym's JAX backend provides fully differentiable spectral solvers for 2-D Kolmogorov turbulence and 3-D turbulent channel flow — examples that are feasible to run on a laptop without any GPU.

## Installing JAX

### CPU (all platforms)

```bash
pip install -U jax
```

This is sufficient for running the Kolmogorov and channel flow examples and for training small agents on a workstation.

### NVIDIA GPU

JAX provides pre-built CUDA wheels for Linux x86_64 and Linux aarch64. Choose the wheel that matches your installed CUDA toolkit:

```bash
# CUDA 12 (driver ≥ 525 on Linux)
pip install -U "jax[cuda12]"

# CUDA 13 (driver ≥ 580 on Linux)
pip install -U "jax[cuda13]"
```

The CUDA 12 wheel is built against CUDA 12.3 and is compatible with any CUDA ≥ 12.1 installation. The CUDA 13 wheel requires CUDA ≥ 13.0. JAX installs its own CUDA and cuDNN libraries via pip, so a matching system CUDA toolkit is not strictly required — the NVIDIA driver is the only system-level dependency.

:::note
CUDA-enabled JAX wheels are available for Linux only. macOS does not support CUDA; use the CPU installation on Apple hardware.
:::

### AMD GPU (ROCm)

```bash
pip install -U "jax[rocm7-local]"
```

ROCm support on Windows WSL2 is experimental. See the [JAX installation documentation](https://docs.jax.dev/en/latest/installation.html) for the current compatibility matrix.

### Google TPU

```bash
pip install -U "jax[tpu]"
```

## Installing the HydroGym Python package

After installing JAX, install the HydroGym JAX extras:

```bash
pip install hydrogym[jax]
```

This adds `jax`, `jaxlib`, `chex`, `navix`, `gymnax`, `tree-math`, `flax`, `omegaconf`, and `toml` alongside the core package. If JAX is already installed with GPU support, the `jax` and `jaxlib` pins in `hydrogym[jax]` will not downgrade it.

## Platform support

| Platform | CPU | CUDA GPU | ROCm GPU | TPU |
|---|---|---|---|---|
| Linux x86_64 | ✅ | ✅ | ✅ | ✅ |
| Linux aarch64 | ✅ | ✅ | — | — |
| macOS (Apple ARM) | ✅ | — | — | — |
| Windows x86_64 | ✅ | Experimental | Experimental | — |

Windows users may need the [Microsoft Visual Studio 2019 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) for the CPU build.

## Verify the installation

```python
import jax
import jax.numpy as jnp

print(jax.devices())            # lists available devices
x = jnp.ones((3, 3))
print(jnp.linalg.norm(x))      # basic computation check
```

## Quick start

```bash
cd examples/jax/getting_started/1_kolmogorov/
python test_kolmogorov_env.py
```

No MPI or MPMD launch is required — JAX environments run in a single Python process.
