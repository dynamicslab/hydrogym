---
sidebar_position: 1
---

# Introduction

![HydroGym overview — six solver backends, environments ranging from 2-D cylinder wakes to 3-D turbulent boundary layers](/img/HydroGymOverview.png)

HydroGym is a reinforcement learning platform for active flow control in fluid dynamics. It provides a unified, [Gymnasium](https://gymnasium.farama.org/)-compatible interface across six solver backends. The Hugging Face dataset publishes named solver case data; its directory count is an input-data inventory, not a count of runtime-validated environments or backend coverage.

The central design goal is to make it straightforward to apply standard RL tooling and techniques — [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), [TorchRL](https://github.com/pytorch/rl), [RLlib](https://docs.ray.io/en/latest/rllib/index.html), [PettingZoo](https://pettingzoo.farama.org/), or your own training loop — to flow control problems without having to write a custom interface. Successfully instantiated environments expose the Gymnasium `env.reset()` / `env.step()` API; solver setup and optional runtime requirements remain backend-specific.

Pre-configured case data is distributed through [Hugging Face Hub](https://huggingface.co/datasets/dynamicslab/HydroGym-environments). A directory being present in the dataset does not by itself establish that it instantiates with a particular backend version. Pre-built Docker images are available for solver backends; follow the backend-specific installation and verification instructions before training.

The six backend integrations are:

| Backend | Method | Dimensions |
|---------|--------|---|
| [Firedrake](./installation/firedrake) | Finite element (FEM) | 2-D |
| [MAIA LBM](./installation/maia) | Lattice Boltzmann | 2-D, 3-D |
| [MAIA Structured FV](./installation/maia) | Finite volume | 3-D |
| [NEK5000](./installation/nek5000) | Spectral element (SEM) | 3-D |
| [JAX](./installation/jax) | Spectral / finite-difference, fully differentiable | 2-D, 3-D |
| [JAX-Fluids](./installation/jaxfluids) | Compressible finite volume, fully differentiable | 2-D, 3-D |

To get a solver running immediately, proceed to the [Quickstart](./quickstart).

