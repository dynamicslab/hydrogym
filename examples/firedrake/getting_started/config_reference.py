#!/usr/bin/env python3
"""
Firedrake FlowEnv Configuration Reference
==========================================

This file contains comprehensive, copy-pasteable configuration examples
for all Firedrake flow environments in HydroGym.

Each example demonstrates different configuration options and use cases.
Copy and modify these examples for your specific needs.

Author: HydroGym Team
License: MIT
"""

import numpy as np
import hydrogym.firedrake as hgym
from hydrogym import FlowEnv
from hydrogym.firedrake.utils.io import (
    CheckpointCallback,
    ParaviewCallback,
    LogCallback,
    SnapshotCallback,
)


# =============================================================================
# EXAMPLE 1: Minimal Configuration (Defaults)
# =============================================================================
# Simplest possible configuration using all defaults


def example_minimal():
    """Minimal configuration - uses all defaults."""
    env_config = {
        "flow": hgym.Cylinder,
        "flow_config": {},
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {
            "dt": 1e-2,  # Only required parameter
        },
    }
    return FlowEnv(env_config)


# =============================================================================
# EXAMPLE 2: Cylinder with Velocity Probes
# =============================================================================
# Demonstrates probe-based observations in wake region


def example_cylinder_probes():
    """Cylinder with velocity probes in wake."""
    # Define wake probe locations (16x4 grid)
    wake_probes = [
        (x, y)
        for x in np.linspace(1.0, 10.0, 16)  # Downstream positions
        for y in np.linspace(-2.0, 2.0, 4)  # Vertical positions
    ]

    env_config = {
        "flow": hgym.Cylinder,
        "flow_config": {
            "mesh": "medium",  # Options: 'medium', 'fine'
            "Re": 100,  # Reynolds number
            "observation_type": "velocity_probes",  # Use velocity probes
            "probes": wake_probes,  # Probe locations
        },
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {
            "dt": 1e-2,
            "order": 3,  # BDF order (1, 2, or 3)
            "stabilization": "supg",  # SUPG stabilization
        },
        "max_steps": 10000,
    }
    return FlowEnv(env_config)


# =============================================================================
# EXAMPLE 3: Rotary Cylinder with Default Observations
# =============================================================================
# Rotary actuation with lift/drag observations


def example_rotary_cylinder():
    """Rotary cylinder with lift/drag observations."""
    env_config = {
        "flow": hgym.RotaryCylinder,
        "flow_config": {
            "mesh": "fine",
            "Re": 100,
            "observation_type": "lift_drag",  # Default: returns (CL, CD)
        },
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {
            "dt": 1e-2,
            "order": 2,
        },
        "max_steps": 50000,
    }
    return FlowEnv(env_config)


# =============================================================================
# EXAMPLE 4: Cavity with Multi-Substep and Callbacks
# =============================================================================
# Demonstrates multi-substep simulation and callback usage


def example_cavity_multistep():
    """Cavity with multi-substep and comprehensive callbacks."""
    env_config = {
        "flow": hgym.Cavity,
        "flow_config": {
            "mesh": "fine",
            "Re": 7500,
            "observation_type": "stress_sensor",  # Wall shear stress
        },
        'solver': hgym.SemiImplicitBDF,
        'solver_config': {
            'dt': 1e-4,  # Small timestep for stiff cavity flow
            'order': 3,
            'stabilization': 'none',
        },
        "actuation_config": {
            "num_substeps": 5,  # 5 solver steps per action
            "reward_aggregation": "mean",  # Average reward over substeps
        },
        "callbacks": [
            CheckpointCallback(
                interval=1000,
                filename="cavity_checkpoint.h5",
                write_mesh=True,
                write_timeseries=False,
            ),
            ParaviewCallback(
                interval=100,
                filename="cavity_viz.pvd",
                postprocess=lambda flow: (flow.u, flow.p),
            ),
            LogCallback(
                postprocess=lambda flow: flow.get_observations(),
                nvals=1,
                interval=10,
                filename="cavity_log.txt",
                print_fmt="t={:.6f} stress={:.6f}",
            ),
        ],
        "max_steps": 50000,
    }
    return FlowEnv(env_config)


# =============================================================================
# EXAMPLE 5: Pinball with Multiple Checkpoints
# =============================================================================
# Demonstrates curriculum learning with multiple restart checkpoints


def example_pinball_checkpoints():
    """Pinball with multiple checkpoints for curriculum learning."""
    env_config = {
        "flow": hgym.Pinball,
        "flow_config": {
            "mesh": "fine",
            "Re": 30,
            "observation_type": "lift_drag",  # Returns 6 values (3 CL + 3 CD)
            "restart": [
                # List of checkpoints - random selection on each reset()
                "pinball_checkpoint_t0.h5",
                "pinball_checkpoint_t100.h5",
                "pinball_checkpoint_t200.h5",
            ],
        },
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {
            "dt": 1e-2,
            "order": 3,
        },
        "actuation_config": {
            "num_substeps": 2,
            "reward_aggregation": "sum",  # Sum rewards over substeps
        },
        "max_steps": 100000,
    }
    return FlowEnv(env_config)


# =============================================================================
# EXAMPLE 6: Step with Noise Forcing
# =============================================================================
# Step flow with random forcing for exploration


def example_step_noise():
    """Step flow with random noise forcing."""
    env_config = {
        "flow": hgym.Step,
        "flow_config": {
            "mesh": "medium",  # Options: 'coarse', 'medium', 'fine'
            "Re": 600,
            "observation_type": "stress_sensor",
            "noise_amplitude": 1.0,  # Random forcing strength
            "noise_time_constant": 0.05,  # Low-pass filter timescale
            "noise_seed": 42,  # RNG seed for reproducibility
        },
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {
            "dt": 1e-2,
            "order": 2,
            "stabilization": "supg",
        },
        "max_steps": 100000,
    }
    return FlowEnv(env_config)


# =============================================================================
# EXAMPLE 7: Cylinder with Pressure Probes and Single Checkpoint
# =============================================================================
# Load from checkpoint and use pressure observations


def example_cylinder_restart():
    """Cylinder restarting from checkpoint with pressure probes."""
    # Pressure probes around cylinder surface
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    surface_probes = [(0.5 * np.cos(t), 0.5 * np.sin(t)) for t in theta]

    env_config = {
        "flow": hgym.Cylinder,
        "flow_config": {
            "mesh": "fine",
            "Re": 100,
            "observation_type": "pressure_probes",
            "probes": surface_probes,
            "restart": "cylinder_checkpoint.h5",  # Single checkpoint
        },
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {
            "dt": 1e-2,
            "order": 3,
        },
        "callbacks": [
            SnapshotCallback(
                interval=50,
                filename="cylinder_snapshots.h5",
            ),
        ],
        "max_steps": 20000,
    }
    return FlowEnv(env_config)


# =============================================================================
# EXAMPLE 8: Advanced Multi-Substep with Custom Aggregation
# =============================================================================
# Demonstrates all multi-substep aggregation options


def example_advanced_multistep():
    """Compare different reward aggregation strategies."""
    configs = {}

    # Mean aggregation (default)
    configs["mean"] = {
        "flow": hgym.Cylinder,
        "flow_config": {"mesh": "medium", "Re": 100},
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {"dt": 1e-2},
        "actuation_config": {
            "num_substeps": 10,
            "reward_aggregation": "mean",
        },
    }

    # Sum aggregation (useful for episodic return)
    configs["sum"] = {
        "flow": hgym.Cylinder,
        "flow_config": {"mesh": "medium", "Re": 100},
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {"dt": 1e-2},
        "actuation_config": {
            "num_substeps": 10,
            "reward_aggregation": "sum",
        },
    }

    # Median aggregation (robust to outliers)
    configs["median"] = {
        "flow": hgym.Cylinder,
        "flow_config": {"mesh": "medium", "Re": 100},
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {"dt": 1e-2},
        "actuation_config": {
            "num_substeps": 10,
            "reward_aggregation": "median",
        },
    }

    return {key: FlowEnv(cfg) for key, cfg in configs.items()}


# =============================================================================
# EXAMPLE 9: All Observation Types for Cylinder
# =============================================================================
# Demonstrates all available observation types


def example_all_observation_types():
    """Create environments with all observation types."""
    base_config = {
        "flow": hgym.Cylinder,
        "flow_config": {
            "mesh": "medium",
            "Re": 100,
        },
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {"dt": 1e-2},
    }

    # Define probe locations
    probes = [(x, 0.0) for x in np.linspace(1.0, 5.0, 10)]

    configs = {}

    # Lift/drag forces
    cfg = base_config.copy()
    cfg["flow_config"] = {**base_config["flow_config"], "observation_type": "lift_drag"}
    configs["lift_drag"] = FlowEnv(cfg)

    # Velocity probes
    cfg = base_config.copy()
    cfg["flow_config"] = {
        **base_config["flow_config"],
        "observation_type": "velocity_probes",
        "probes": probes,
    }
    configs["velocity"] = FlowEnv(cfg)

    # Pressure probes
    cfg = base_config.copy()
    cfg["flow_config"] = {
        **base_config["flow_config"],
        "observation_type": "pressure_probes",
        "probes": probes,
    }
    configs["pressure"] = FlowEnv(cfg)

    # Vorticity probes
    cfg = base_config.copy()
    cfg["flow_config"] = {
        **base_config["flow_config"],
        "observation_type": "vorticity_probes",
        "probes": probes,
    }
    configs["vorticity"] = FlowEnv(cfg)

    return configs


# =============================================================================
# EXAMPLE 10: Production RL Training Configuration
# =============================================================================
# Recommended configuration for RL training


def example_production_rl():
    """Production-ready configuration for RL training."""
    env_config = {
        "flow": hgym.Cylinder,
        "flow_config": {
            "mesh": "medium",  # Balance accuracy/speed
            "Re": 100,
            "observation_type": "lift_drag",
            "restart": [  # Multiple initial conditions
                "cylinder_ckpt_0.h5",
                "cylinder_ckpt_100.h5",
                "cylinder_ckpt_200.h5",
            ],
        },
        "solver": hgym.SemiImplicitBDF,
        "solver_config": {
            "dt": 1e-2,
            "order": 3,
            "stabilization": "supg",
            "rtol": 1e-6,
        },
        "actuation_config": {
            "num_substeps": 5,  # 5x sample efficiency
            "reward_aggregation": "mean",
        },
        "callbacks": [
            CheckpointCallback(
                interval=5000,  # Save every 5k steps
                filename="training_checkpoint.h5",
                write_timeseries=False,
            ),
            LogCallback(
                postprocess=lambda flow: flow.compute_forces(),
                nvals=2,
                interval=100,
                filename="training_log.txt",
            ),
        ],
        "max_steps": 200,  # Episode length
    }
    return FlowEnv(env_config)


# =============================================================================
# CONFIGURATION SUMMARY TABLE
# =============================================================================

CONFIGURATION_SUMMARY = """
╔══════════════════════════════════════════════════════════════════════════╗
║                    FIREDRAKE FLOWENV CONFIGURATION SUMMARY               ║
╚══════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────┐
│ ENVIRONMENT TYPES                                                        │
├──────────────┬───────────┬──────────────┬───────────────┬────────────────┤
│ Environment  │ Inputs    │ MAX_CONTROL  │ Default Obs   │ Available Mesh │
├──────────────┼───────────┼──────────────┼───────────────┼────────────────┤
│ Cylinder     │ 1         │ 0.1          │ lift_drag     │ medium, fine   │
│ RotaryCyl    │ 1         │ 0.5π         │ lift_drag     │ medium, fine   │
│ Pinball      │ 3         │ 10.0         │ lift_drag     │ medium, fine   │
│ Cavity       │ 1         │ 0.1          │ stress_sensor │ medium, fine   │
│ Step         │ 1         │ 0.1          │ stress_sensor │ coarse,med,fine│
└──────────────┴───────────┴──────────────┴───────────────┴────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ OBSERVATION TYPES                                                        │
├──────────────────┬───────────────────────────────────────────────────────┤
│ Type             │ Description                                           │
├──────────────────┼───────────────────────────────────────────────────────┤
│ lift_drag        │ Lift/drag coefficients (CL, CD)                       │
│ stress_sensor    │ Wall shear stress                                     │
│ velocity_probes  │ Velocity (u, v) at probe locations                    │
│ pressure_probes  │ Pressure at probe locations                           │
│ vorticity_probes │ Vorticity at probe locations                          │
└──────────────────┴───────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ SOLVER OPTIONS                                                           │
├──────────────────┬───────────────────────────────────────────────────────┤
│ Solver           │ Use Case                                              │
├──────────────────┼───────────────────────────────────────────────────────┤
│ NewtonSolver     │ Steady-state solutions                                │
│ SemiImplicitBDF  │ Transient simulation (recommended)                    │
│ LinearizedBDF    │ Perturbation analysis around base flow                │
└──────────────────┴───────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ RECOMMENDED TIME STEPS (dt)                                              │
├──────────────┬───────────────────────────────────────────────────────────┤
│ Environment  │ Recommended dt                                            │
├──────────────┼───────────────────────────────────────────────────────────┤
│ Cylinder     │ 1e-2                                                      │
│ RotaryCyl    │ 1e-2                                                      │
│ Pinball      │ 1e-2                                                      │
│ Cavity       │ 1e-4  (stiff! requires small timestep)                    │
│ Step         │ 1e-2 to 1e-3                                              │
└──────────────┴───────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ MULTI-SUBSTEP CONFIGURATION                                              │
├──────────────────┬───────────────────────────────────────────────────────┤
│ Parameter        │ Options / Description                                 │
├──────────────────┼───────────────────────────────────────────────────────┤
│ num_substeps     │ Number of solver steps per action (default: 1)       │
│ reward_aggreg.   │ 'mean', 'sum', 'median'                               │
└──────────────────┴───────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ AVAILABLE CALLBACKS                                                      │
├────────────────────┬─────────────────────────────────────────────────────┤
│ Callback           │ Purpose                                             │
├────────────────────┼─────────────────────────────────────────────────────┤
│ CheckpointCallback │ Save HDF5 checkpoints for restart                   │
│ ParaviewCallback   │ Export .pvd files for Paraview visualization        │
│ LogCallback        │ Log observations/forces to text file                │
│ SnapshotCallback   │ Save snapshots for POD/DMD modal analysis           │
│ GenericCallback    │ Custom user-defined callback                        │
└────────────────────┴─────────────────────────────────────────────────────┘

For detailed examples, see the functions in this file:
  - example_minimal()              - Simplest configuration
  - example_cylinder_probes()      - Probe-based observations
  - example_cavity_multistep()     - Multi-substep with callbacks
  - example_pinball_checkpoints()  - Multiple restart checkpoints
  - example_production_rl()        - Production RL training setup
"""


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print(CONFIGURATION_SUMMARY)
    print("\n" + "=" * 76)
    print("Creating example environments...")
    print("=" * 76 + "\n")

    # Example 1: Minimal
    print("1. Creating minimal environment...")
    env = example_minimal()
    print(f"   ✓ Observation space: {env.observation_space}")
    print(f"   ✓ Action space: {env.action_space}\n")

    # Example 2: Cylinder with probes
    print("2. Creating cylinder with velocity probes...")
    env = example_cylinder_probes()
    print(f"   ✓ Observation space: {env.observation_space}")
    # For velocity_probes: obs_dim = 2 * num_probes (u and v components)
    num_probes = env.observation_space.shape[0] // 2
    print(f"   ✓ Number of probes: {num_probes}\n")

    # Example 3: Multi-substep
    print("3. Creating multi-substep cavity environment...")
    env = example_cavity_multistep()
    print(f"   ✓ Substeps per action: {env.num_substeps}")
    print(f"   ✓ Reward aggregation: {env.reward_aggregation}")
    print(f"   ✓ Number of callbacks: {len(env.callbacks)}\n")

    print("=" * 76)
    print("All examples created successfully!")
    print("=" * 76)
