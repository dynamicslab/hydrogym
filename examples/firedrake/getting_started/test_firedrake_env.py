#!/usr/bin/env python3
"""
Firedrake Environment Test Script
==================================

Test script for Firedrake-based CFD environments.

Unlike MAIA environments, Firedrake runs directly in Python (no MPMD coupling).
This makes testing simpler - just run: python test_firedrake_env.py

Usage:
    # Single process
    python test_firedrake_env.py --environment cylinder

    # MPI parallel (for larger simulations)
    mpirun -np 4 python test_firedrake_env.py --environment cylinder --num-steps 50

Available environments:
    - cylinder: Flow around a circular cylinder with jet actuation
    - rotary_cylinder: Flow around a rotating cylinder
    - pinball: Flow around three cylinders (pinball configuration)
    - cavity: Open cavity flow
    - step: Backward-facing step flow

Arguments:
    --environment: Environment type (required)
    --num-steps: Number of simulation steps (default: 10)
    --num-episodes: Number of episodes (default: 1)
    --reynolds: Reynolds number (default: environment-specific)
    --mesh-resolution: Mesh resolution - 'coarse', 'medium', or 'fine' (default: environment-specific)
    --seed: Random seed for reproducibility (optional)
    --verbose: Enable verbose output

Configuration Reference:
=======================

This script demonstrates all available FlowEnv configuration options.
See inline comments in run_firedrake_test() for comprehensive examples.

Key Configuration Categories:
-----------------------------

1. FLOW CONFIGURATION (flow_config):
   - mesh: Mesh resolution ('coarse', 'medium', 'fine')
   - Re: Reynolds number (flow-dependent defaults)
   - observation_type: How to compute observations
     * 'lift_drag' - Lift/drag coefficients (cylinder, pinball)
     * 'stress_sensor' - Wall shear stress (cavity, step)
     * 'velocity_probes' - Velocity at probe locations
     * 'pressure_probes' - Pressure at probe locations
     * 'vorticity_probes' - Vorticity at probe locations
   - probes: List of (x, y) coordinates for probe-based observations
   - restart: Checkpoint configuration (flexible!)
     * None - Auto-loads from HF Hub based on flow config
     * 'path/to/file.h5' - Explicit checkpoint path
     * 'Cylinder_2D_Re100_medium_FD' - HF Hub environment name
     * ['ckpt1.h5', 'ckpt2.h5'] - Multiple checkpoints (random selection)
   - local_dir: Local checkpoint directory (for offline/testing)
   - cache_dir: Custom HF cache directory
   - velocity_order: FEM element order (default: 2 for P2-P1)
   - noise_amplitude: Random forcing strength (Step only)
   - noise_time_constant: Noise filter timescale (Step only)
   - noise_seed: RNG seed for noise (Step only)

2. SOLVER CONFIGURATION (solver_config):
   - dt: Time step (REQUIRED for transient solvers)
   - order: BDF order (1, 2, or 3; default: 3)
   - stabilization: 'none', 'supg', 'gls' (default varies)
   - rtol: Krylov solver relative tolerance (default: 1e-6)

3. ACTUATION CONFIGURATION (actuation_config):
   - num_substeps: Number of solver steps per action (default: 1)
   - reward_aggregation: How to aggregate rewards ('mean', 'sum', 'median')

4. CALLBACKS (callbacks):
   - CheckpointCallback: Save simulation state to HDF5
   - ParaviewCallback: Export for visualization
   - LogCallback: Log observations/forces to file
   - SnapshotCallback: Save snapshots for modal analysis
   - GenericCallback: Custom user-defined callbacks

5. ENVIRONMENT SETTINGS:
   - max_steps: Maximum steps per episode (default: 1e6)

Example Configurations:
----------------------

Cylinder with velocity probes:
    flow_config = {
        'mesh': 'medium',
        'Re': 100,
        'observation_type': 'velocity_probes',
        'probes': [(x, y) for x in [1,2,3] for y in [-1,0,1]],
    }

Cavity with multi-substep:
    actuation_config = {
        'num_substeps': 5,
        'reward_aggregation': 'mean',
    }

Automatic checkpoint loading:
    # No restart specified - auto-loads 'Cylinder_2D_Re100_medium_FD' from HF Hub
    flow_config = {
        'mesh': 'medium',
        'Re': 100,
    }

Local checkpoint directory:
    flow_config = {
        'local_dir': '/workspace/my_checkpoints',
    }

Multiple checkpoints (curriculum learning):
    flow_config = {
        'restart': ['ckpt1.h5', 'ckpt2.h5', 'ckpt3.h5'],
    }

See full examples in the script below.
"""

import argparse
import logging
import sys
from typing import Optional

import numpy as np


# Setup logging before importing heavy libraries
def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout
    )
    return logging.getLogger(__name__)


def run_firedrake_test(
    env_type: str,
    num_steps: int,
    num_episodes: int,
    reynolds: Optional[float],
    mesh_resolution: Optional[int],
    seed: Optional[int],
    logger: logging.Logger,
    **kwargs,
) -> None:
    """
    Run Firedrake environment test.

    Args:
        env_type: Environment type (cylinder, pinball, cavity, step)
        num_steps: Number of steps per episode
        num_episodes: Number of episodes
        reynolds: Reynolds number (optional)
        mesh_resolution: Mesh resolution (optional)
        seed: Random seed (optional)
        logger: Logger instance
        **kwargs: Additional environment parameters
    """
    logger.info("=" * 70)
    logger.info(f"Firedrake Environment Test: {env_type}")
    logger.info("=" * 70)

    # Import hydrogym (FlowEnv wrapper + firedrake)
    logger.info("Importing hydrogym...")
    try:
        import hydrogym.firedrake as firedrake
        from hydrogym import FlowEnv
    except ImportError as e:
        logger.error(f"✗ Failed to import: {e}")
        logger.error("Make sure Firedrake is installed: https://www.firedrakeproject.org/download.html")
        sys.exit(1)

    # Map environment names to flow classes and default solvers
    env_map = {
        "cylinder": (firedrake.Cylinder, firedrake.SemiImplicitBDF),
        "rotary_cylinder": (firedrake.RotaryCylinder, firedrake.SemiImplicitBDF),
        "pinball": (firedrake.Pinball, firedrake.SemiImplicitBDF),
        "cavity": (firedrake.Cavity, firedrake.SemiImplicitBDF),
        "step": (firedrake.Step, firedrake.SemiImplicitBDF),
    }

    if env_type not in env_map:
        logger.error(f"✗ Unknown environment: {env_type}")
        logger.error(f"Available: {list(env_map.keys())}")
        sys.exit(1)

    flow_class, solver_class = env_map[env_type]

    # =========================================================================
    # COMPREHENSIVE FLOWENV CONFIGURATION EXAMPLE
    # =========================================================================
    # This section demonstrates ALL available configuration options.
    # Uncomment and modify sections as needed for your specific use case.
    # =========================================================================

    # Build flow configuration
    flow_config = kwargs.copy()
    if reynolds is not None:
        flow_config["Re"] = reynolds
    if mesh_resolution is not None:
        flow_config["mesh"] = mesh_resolution

    # --- OBSERVATION CONFIGURATION ---
    # Specify how observations are computed
    # Options: "lift_drag", "stress_sensor", "velocity_probes", "pressure_probes", "vorticity_probes"
    # flow_config['observation_type'] = 'lift_drag'

    # --- PROBE CONFIGURATION ---
    # Define probe locations for probe-based observations
    # Example: Wake probes for cylinder

    wake_probes = [
        (x, y)
        for x in np.linspace(1.0, 10.0, 16)  # 16 points in x
        for y in np.linspace(-2.0, 2.0, 4)  # 4 points in y
    ]
    flow_config["probes"] = wake_probes
    flow_config["observation_type"] = "velocity_probes"

    # --- CHECKPOINT/RESTART CONFIGURATION ---

    # Option 1: Automatic checkpoint inference
    # Just specify the local directory - auto-loads 'Cylinder_2D_Re100_medium_FD'
    # flow_config['local_dir'] = '/workspace/firedrake_checkpoints'
    # The checkpoint will be auto-loaded based on: {FlowClass}_2D_Re{Re}_{mesh}_FD

    # Option 2: Explicit environment name
    # flow_config['restart'] = 'Cylinder_2D_Re100_medium_FD'
    # flow_config['local_dir'] = '/workspace/firedrake_checkpoints'

    # Option 3: Custom cache directory (downloads from HF Hub to custom location)
    # flow_config['cache_dir'] = '/workspace/firedrake_checkpoints_local'

    # Option 4: Explicit checkpoint path
    # flow_config['restart'] = (
    #     '/workspace/firedrake_checkpoints/Cylinder_2D_Re100_medium_FD/'
    #     'cylinder_Re-100_Mesh-medium_DT-0.01_00000570.ckpt'
    # )

    # Multiple checkpoints: randomly selects one on each reset()
    # flow_config['restart'] = [
    #     'checkpoint_t0.h5',
    #     'checkpoint_t100.h5',
    #     'checkpoint_t200.h5',
    # ]

    # --- STEP-SPECIFIC NOISE CONFIGURATION ---
    # Only for Step environment - random forcing parameters
    # flow_config['noise_amplitude'] = 1.0           # Noise strength
    # flow_config['noise_time_constant'] = 0.05      # Low-pass filter timescale
    # flow_config['noise_seed'] = 42                 # RNG seed for reproducibility

    # --- ADVANCED FEM CONFIGURATION ---
    # flow_config['velocity_order'] = 2  # Velocity element order (default: 2 for P2-P1 Taylor-Hood)

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        logger.info(f"Random seed set to: {seed}")

    # Create environment using FlowEnv wrapper
    logger.info(f"Creating {env_type} environment...")
    logger.info(f"Flow config: {flow_config}")

    # --- SOLVER CONFIGURATION ---
    # Time-stepping parameters for transient solvers
    solver_config = {
        "dt": 1e-2,  # Time step (REQUIRED for transient)
        "order": 3,  # BDF order: 1, 2, or 3 (default: 3)
        "stabilization": "none",  # Options: 'none', 'supg', 'gls'
        # 'rtol': 1e-6,                # Krylov solver relative tolerance
    }

    # Recommended dt by environment:
    # - Cylinder: 1e-2
    # - Cavity: 1e-4 (very stiff!)
    # - Pinball: 1e-2
    # - Step: 1e-2 to 1e-3

    # --- ACTUATION CONFIGURATION ---
    # Multi-substep simulation with reward aggregation
    actuation_config = {
        "num_substeps": 2,  # Number of solver steps per action (default: 1)
        "reward_aggregation": "mean",  # How to aggregate: 'mean', 'sum', 'median'
    }

    # Example multi-substep: Run 5 simulation steps per action
    # actuation_config = {
    #     'num_substeps': 5,
    #     'reward_aggregation': 'mean',  # Average reward over 5 substeps
    # }

    # --- CALLBACK CONFIGURATION ---
    # List of callbacks for logging, checkpointing, visualization
    from hydrogym.firedrake.utils.io import CheckpointCallback, LogCallback, ParaviewCallback

    callbacks = [
        # 1. Paraview visualization - save every 5 steps
        ParaviewCallback(
            interval=5,
            filename="output/solution.pvd",
            postprocess=lambda flow: (flow.u, flow.p),
        ),
        # 2. Checkpoint saving - save every 100 steps
        CheckpointCallback(
            interval=2,
            filename="output/checkpoint.h5",
            write_mesh=True,
            write_timeseries=False,
        ),
        # 3. Log observations - log every step
        LogCallback(
            postprocess=lambda flow: flow.get_observations()[:4],  # Log first 4 probe velocities
            nvals=4,
            interval=1,
            filename="output/observations.txt",
            print_fmt="t={:.4f} obs=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]",
        ),
    ]

    # --- COMPLETE ENV_CONFIG ---
    env_config = {
        "flow": flow_class,
        "flow_config": flow_config,
        "solver": solver_class,
        "solver_config": solver_config,
        "actuation_config": actuation_config,  # Multi-substep configuration
        "callbacks": callbacks,  # Callback list
        "max_steps": int(1e6),  # Maximum steps per episode
    }

    # =========================================================================
    # ENVIRONMENT-SPECIFIC CONFIGURATION REFERENCE
    # =========================================================================
    #
    # CYLINDER (Blowing/Suction):
    #   - flow: firedrake.Cylinder
    #   - num_inputs: 1
    #   - MAX_CONTROL: 0.1
    #   - Default observation: 'lift_drag' (2 values: CL, CD)
    #   - Available meshes: 'medium', 'fine'
    #   - Default Re: 100
    #
    # ROTARY CYLINDER (Rotation):
    #   - flow: firedrake.RotaryCylinder
    #   - num_inputs: 1
    #   - MAX_CONTROL: 0.5*π (radians)
    #   - Default observation: 'lift_drag' (2 values: CL, CD)
    #   - Available meshes: 'medium', 'fine'
    #   - Default Re: 100
    #
    # PINBALL (Three Rotating Cylinders):
    #   - flow: firedrake.Pinball
    #   - num_inputs: 3 (one per cylinder)
    #   - MAX_CONTROL: 10.0
    #   - Default observation: 'lift_drag' (6 values: 3 CL + 3 CD)
    #   - Available meshes: 'medium', 'fine'
    #   - Default Re: 30
    #
    # CAVITY (Lid-Driven Cavity):
    #   - flow: firedrake.Cavity
    #   - num_inputs: 1
    #   - MAX_CONTROL: 0.1
    #   - Default observation: 'stress_sensor' (1 value: wall shear stress)
    #   - Available meshes: 'medium', 'fine'
    #   - Default Re: 7500
    #   - Recommended dt: 1e-4 (stiff!)
    #
    # STEP (Backward-Facing Step):
    #   - flow: firedrake.Step
    #   - num_inputs: 1
    #   - MAX_CONTROL: 0.1
    #   - Default observation: 'stress_sensor' (1 value: wall shear stress)
    #   - Available meshes: 'coarse', 'medium', 'fine'
    #   - Default Re: 600
    #   - Has noise forcing (configurable amplitude/timescale/seed)
    #
    # =========================================================================

    try:
        env = FlowEnv(env_config)
        logger.info("✓ Environment created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create environment: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        sys.exit(1)

    try:
        # Log environment information
        logger.info("-" * 70)
        logger.info("Environment Information:")
        logger.info(f"  Observation space: {env.observation_space}")
        logger.info(f"  Action space: {env.action_space}")
        if hasattr(env, "flow"):
            logger.info(f"  Reynolds number: {float(env.flow.Re)}")

            # Verify checkpoint was loaded (check resolved path from flow object)
            if hasattr(env.flow, "checkpoint_path") and env.flow.checkpoint_path is not None:
                import firedrake as fd

                u_norm = fd.norm(env.flow.u)
                p_norm = fd.norm(env.flow.p)
                logger.info("\n  Checkpoint verification:")
                logger.info(f"    Checkpoint path: {env.flow.checkpoint_path}")
                logger.info(f"    Velocity L2 norm: {u_norm:.6e}")
                logger.info(f"    Pressure L2 norm: {p_norm:.6e}")
                if u_norm < 1e-10 and p_norm < 1e-10:
                    logger.warning("    ⚠ Fields are zero - checkpoint may not have loaded!")
                else:
                    logger.info("    ✓ Non-zero fields detected - checkpoint loaded successfully")
            else:
                logger.info("\n  No checkpoint loaded - starting from zero initial conditions")
        logger.info("-" * 70)

        # Run episodes
        total_steps = 0
        total_reward = 0.0

        for episode in range(num_episodes):
            logger.info(f"\nEpisode {episode + 1}/{num_episodes}")
            logger.info("-" * 70)

            # Reset environment
            try:
                obs, _ = env.reset(seed=seed)
                logger.info("✓ Environment reset")
                logger.info(f"  Initial observation shape: {obs.shape}")
                logger.info(f"  Initial observation values: {obs}")
            except Exception as e:
                logger.error(f"✗ Reset failed: {e}")
                raise

            episode_reward = 0.0
            episode_steps = 0

            # Run steps
            for step in range(num_steps):
                try:
                    # Sample random action
                    action = env.action_space.sample()

                    # Execute step
                    obs, reward, terminated, truncated, _ = env.step(action)

                    episode_reward += reward
                    episode_steps += 1
                    total_steps += 1

                    # Log progress
                    logger.info(
                        f"  Step {step + 1}/{num_steps}: "
                        f"reward={reward:.6f}, "
                        f"terminated={terminated}, "
                        f"truncated={truncated}"
                    )

                    # Handle episode termination
                    if terminated or truncated:
                        reason = "terminated" if terminated else "truncated"
                        logger.info(f"  Episode ended ({reason}) after {episode_steps} steps")
                        break

                except Exception as e:
                    logger.error(f"✗ Step {step + 1} failed: {e}")
                    raise

            total_reward += episode_reward
            logger.info(f"Episode {episode + 1} summary:")
            logger.info(f"  Steps: {episode_steps}")
            logger.info(f"  Total reward: {episode_reward:.6f}")
            logger.info(f"  Average reward: {episode_reward / episode_steps:.6f}")

        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("Test Summary:")
        logger.info(f"  Episodes completed: {num_episodes}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Total reward: {total_reward:.6f}")
        logger.info(f"  Average reward per step: {total_reward / total_steps:.6f}")
        logger.info("=" * 70)

    finally:
        # Cleanup
        logger.info("\nClosing environment...")
        try:
            if hasattr(env, "close"):
                env.close()
            logger.info("✓ Environment closed successfully")
        except Exception as e:
            logger.warning(f"⚠ Error during cleanup: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Firedrake CFD environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--environment",
        required=True,
        choices=["cylinder", "rotary_cylinder", "pinball", "cavity", "step"],
        help="Environment type",
    )
    parser.add_argument("--num-steps", type=int, default=10, help="Number of steps per episode (default: 10)")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes (default: 1)")
    parser.add_argument("--reynolds", type=float, default=None, help="Reynolds number (default: environment-specific)")
    parser.add_argument(
        "--mesh-resolution",
        type=str,
        choices=["coarse", "medium", "fine"],
        default=None,
        help="Mesh resolution: 'coarse', 'medium', or 'fine' (default: environment-specific)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(verbose=args.verbose)

    # Run test
    try:
        run_firedrake_test(
            env_type=args.environment,
            num_steps=args.num_steps,
            num_episodes=args.num_episodes,
            reynolds=args.reynolds,
            mesh_resolution=args.mesh_resolution,
            seed=args.seed,
            logger=logger,
        )
        logger.info("\n✓ Test completed successfully!")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\n⚠ Test interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
