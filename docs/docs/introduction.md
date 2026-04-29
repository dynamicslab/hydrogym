---
sidebar_position: 1
---

# Introduction

![HydroGym overview — six solver backends, environments ranging from 2-D cylinder wakes to 3-D turbulent boundary layers](/img/HydroGymOverview.png)

HydroGym is a reinforcement learning platform for active flow control in fluid dynamics. It provides a unified, [Gymnasium](https://gymnasium.farama.org/)-compatible interface to 88 pre-configured CFD environments that span six solver backends, two to three spatial dimensions, and Reynolds numbers from laminar benchmarks to wall-bounded turbulence above Re = 400 000.

The central design goal is to make it straightforward to apply standard RL tooling and techniques — [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), [TorchRL](https://github.com/pytorch/rl), [RLlib](https://docs.ray.io/en/latest/rllib/index.html), [PettingZoo](https://pettingzoo.farama.org/), or your own training loop — to a broad set of physically meaningful flow control problems without having to write a custom environment or solver interface. Every environment exposes the same `env.reset()` / `env.step()` API; solver-specific setup is handled behind the scenes.

Environment configurations and initial-condition checkpoints are distributed through [HuggingFace Hub](https://huggingface.co/datasets/dynamicslab/HydroGym-environments) as a dedicated dataset and are downloaded automatically the first time an environment is created. Pre-built Docker images are available for all solver backends, making it possible to go from zero to a running training loop in a single `docker pull` command.

The six presently supported solver backends are:

| Backend | Method | Environments | Dimensions |
|---------|--------|:---:|---|
| [Firedrake](./installation/firedrake) | Finite element (FEM) | 20 | 2-D |
| [MAIA LBM](./installation/maia) | Lattice Boltzmann | 55 | 2-D, 3-D |
| [MAIA Structured FV](./installation/maia) | Finite volume | 8 | 3-D |
| [NEK5000](./installation/nek5000) | Spectral element (SEM) | 1 | 3-D |
| [JAX](./installation/jax) | Spectral / finite-difference, fully differentiable | 2 | 2-D, 3-D |
| [JAX-Fluids](./installation/jaxfluids) | Compressible finite volume, fully differentiable | 2 | 2-D, 3-D |

To get a solver running immediately, proceed to the [Quickstart](./quickstart).

