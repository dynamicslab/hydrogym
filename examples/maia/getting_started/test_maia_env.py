#!/usr/bin/env python3
"""
MAIA Environment Test Script
=============================

Test script for MAIA-based CFD environments with MPMD coupling.

Unlike Firedrake, MAIA requires Multi-Program Multiple Data (MPMD) execution:
one MPI process runs Python (this script), another runs the m-AIA CFD solver.

Usage:
    # MPMD execution with mpirun (required for MAIA)
    mpirun -np 1 python test_maia_env.py --environment Cylinder_2D_Re200 : -np 1 maia properties.toml

    # Note: The ':' separator indicates two separate programs:
    #   - Process 0: Python script (this file)
    #   - Process 1: MAIA solver with properties.toml configuration

Available environments:
    2D & 3D Flows:
    - Cylinder_{2D/3D}_Re{Re}: Flow around circular cylinder with jet actuation
    - RotaryCylinder_{2D/3D}_Re{Re}: Flow around rotating cylinder
    - Pinball_{2D/3D}_Re{Re}: Flow around three cylinders (pinball configuration)
    - JetPinball_{2D/3D}_Re{Re}: Pinball with jet actuation
    - SquareCylinder_{2D/3D}_Re{Re}: Flow around square cylinder
    - Cavity_{2D/3D}_Re{Re}: Open cavity flow
    - Cavity3Jet_{2D/3D}_Re{Re}: Open cavity with 3-jet actuation
    - NACA0012_{2D/3D}_Re{Re}: Airfoil flow
    - NACA0012Gust_{2D/3D}_Re{Re}: Airfoil with gust forcing

    3D Flows:
    - Cube_3D_Re{Re}: Flow around cube
    - Sphere_3D_Re{Re}: Flow around sphere
    - Cylinder_3D_Re{Re}: 3D cylinder flow

    Common Reynolds numbers: 100, 200, 1000, etc. (environment-specific)

Arguments:
    --environment: Name of the environment to test (required)
    --num-steps: Number of actuation steps to run (default: 10)
    --num-episodes: Number of episodes to run (default: 1)
    --probe-x-range: X-axis range for probes (default: 1.0,8.0,8)
    --probe-y-range: Y-axis range for probes (default: -1.0,1.0,5)
    --seed: Random seed for reproducibility (optional)
    --obs-norm: Observation normalization strategy (default: U_inf)
    --verbose: Enable verbose output (default: False)

Configuration Reference:
=======================

This script demonstrates all available MAIA environment configuration options.
See inline comments in run_environment_test() for comprehensive examples.

Key Configuration Categories:
-----------------------------

1. ENVIRONMENT SELECTION:
   - environment_name: Required. Format: 'TYPE_DIMENSION_Re{Reynolds}'
     Examples: 'Cylinder_2D_Re200', 'Sphere_3D_Re300'
   - hf_repo_id: Hugging Face repository (default: 'dynamicslab/HydroGym-environments')

2. DATA MANAGEMENT:
   - use_clean_cache: Use clean cache directory (default: True)
     * True - Creates fresh workspace copy (recommended for production)
     * False - Uses cached workspace (faster for development/testing)
   - local_fallback_dir: Local directory for offline usage
   - configuration_file: Custom path to MAIA config.yaml (optional)

3. OBSERVATION CONFIGURATION:
   - probe_locations: List of (x,y) or (x,y,z) coordinates for flow field sampling
     Format: [x0, y0, x1, y1, ...] for 2D (flattened list) or [x0, y0, z0, x1, y1, z1, ...] for 3D
   - obs_normalization_strategy: How to normalize observations
     * 'U_inf' - Normalize by freestream velocity (default, recommended)
     * 'probewise_mean_std' - Per-probe mean/std normalization (has to be computed)
     * 'none' - Raw observation values (no normalization)
     * 'customized' - User-defined normalization (requires obs_loc/obs_scale)
   - obs_loc: Offset for customized normalization (array of means)
   - obs_scale: Scale for customized normalization (array of std devs)

4. TESTING/DEBUGGING:
   - is_testing: Enable testing mode (default: False)
   - seed: Random seed for reproducibility

Observation Normalization Strategies:
------------------------------------

1. 'U_inf' (Recommended):
   - Normalizes observations by the freestream velocity
   - Best for learning: observations are O(1) and comparable across Re
   - obs_normalized = obs_raw / U_inf

2. 'probewise_mean_std':
   - Normalizes each probe independently using its own mean/std
   - Computed from initial observations
   - obs_normalized = (obs_raw - mean) / std
   - Useful for handling large spatial variations

3. 'none':
   - Raw observation values (no normalization)
   - Use when you want to handle normalization in your RL algorithm

4. 'customized':
   - User-provided normalization parameters
   - Requires both obs_loc (offsets) and obs_scale (scales)
   - obs_normalized = (obs_raw - obs_loc) / obs_scale

Probe Configuration:
-------------------

Probes sample the flow field at specified locations. The observation vector
contains velocity components at each probe location.

Available Probe Types in MAIA:
    - u: Velocity component in x-direction (always available)
    - v: Velocity component in y-direction (always available)
    - w: Velocity component in z-direction (3D flows only)
    - rho: Density (always available)
    - p: Pressure (always available)
    - forces: Integrated forces on bodies (environment-specific)

    Note: By default, HydroGym environments return velocity probes (u, v, [w]).
    Additional probe types may require custom configuration in environment_config.yaml.

For 2D flows (x, y coordinates):
    probe_locations = [x0, y0, x1, y1, x2, y2, ...]
    -> Observation dimension: 2 * num_probes (u and v at each probe)

For 3D flows (x, y, z coordinates):
    probe_locations = [x0, y0, z0, x1, y1, z1, ...]
    -> Observation dimension: 3 * num_probes (u, v, w at each probe)

Example probe grids:
    # Wake sampling for cylinder (8x5 grid)
    probe_x = np.linspace(1.0, 8.0, 8)   # 8 points downstream
    probe_y = np.linspace(-1.0, 1.0, 5)  # 5 points in crossflow
    X, Y = np.meshgrid(probe_x, probe_y)
    probe_locations = [coord for x, y in zip(X.flat, Y.flat) for coord in (x, y)]

    # Sparse sampling (4 strategic probes)
    probe_locations = [2.0, 0.0,    # Wake centerline
                       4.0, 0.5,    # Upper wake
                       4.0, -0.5,   # Lower wake
                       6.0, 0.0]    # Far wake

Example Configurations:
----------------------

Cylinder with default settings:
    env = maia.from_hf('Cylinder_2D_Re200')

Cylinder with custom probes:
    probe_locations = [x for x in np.linspace(1, 8, 16)
                         for y in np.linspace(-2, 2, 8)
                         for coord in (x, y)]
    env = maia.from_hf(
        'Cylinder_2D_Re200',
        probe_locations=probe_locations,
        obs_normalization_strategy='U_inf',
    )

Rotary cylinder with probewise normalization:
    env = maia.from_hf(
        'RotaryCylinder_2D_Re1000',
        probe_locations=probe_locations,
        obs_normalization_strategy='probewise_mean_std',
    )

Using local/offline data:
    env = maia.from_hf(
        'SquareCylinder_3D_Re200',
        local_fallback_dir='/path/to/local/environments',
        use_clean_cache=False,
    )

Customized normalization:
    env = maia.from_hf(
        'Pinball_2D_Re150',
        probe_locations=probe_locations,
        obs_normalization_strategy='customized',
        obs_loc=np.zeros(80),        # 40 probes × 2 components
        obs_scale=np.ones(80) * 0.1, # Scale by 0.1
    )

See full examples in the script below.
"""

import argparse
import logging
import sys
from typing import List, Optional, Tuple

import numpy as np

# Force unbuffered output for MPMD mode
sys.stdout = open(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), "w", buffering=1)

# Import hydrogym (lazy import prevents firedrake loading in MPMD mode)
import hydrogym.maia as maia  # noqa: E402


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure logging with timestamps and appropriate level.

    Args:
        verbose: If True, set level to DEBUG, otherwise INFO.

    Returns:
        Configured logger instance.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout
    )
    return logging.getLogger(__name__)


def parse_range_arg(arg_str: str) -> Tuple[float, float, int]:
    """
    Parse range argument in format 'min,max,num'.

    Args:
        arg_str: String in format 'min,max,num' (e.g., '1.0,8.0,8')

    Returns:
        Tuple of (min, max, num_points)
    """
    parts = arg_str.split(",")
    if len(parts) != 3:
        raise ValueError(f"Range must be in format 'min,max,num', got: {arg_str}")
    return float(parts[0]), float(parts[1]), int(parts[2])


def create_probe_locations(
    x_range: Tuple[float, float, int], y_range: Tuple[float, float, int], logger: logging.Logger
) -> List[float]:
    """
    Create probe location grid for 2D flow field sampling.

    Args:
        x_range: Tuple of (x_min, x_max, num_x_probes)
        y_range: Tuple of (y_min, y_max, num_y_probes)
        logger: Logger instance

    Returns:
        Flattened list of probe coordinates [x0, y0, x1, y1, ...]
    """
    x_min, x_max, num_x = x_range
    y_min, y_max, num_y = y_range

    logger.info("Creating probe grid:")
    logger.info(f"  X: [{x_min}, {x_max}] with {num_x} probes")
    logger.info(f"  Y: [{y_min}, {y_max}] with {num_y} probes")
    logger.info(f"  Total probes: {num_x * num_y}")

    xp = np.linspace(x_min, x_max, num_x)
    yp = np.linspace(y_min, y_max, num_y)
    X, Y = np.meshgrid(xp, yp)

    probe_list = [(x, y) for x, y in zip(X.ravel(), Y.ravel())]
    probe_locations = [coord for point in probe_list for coord in point]

    return probe_locations


def run_environment_test(
    environment_name: str,
    num_steps: int,
    num_episodes: int,
    probe_locations: List[float],
    obs_normalization: str,
    seed: Optional[int],
    logger: logging.Logger,
) -> None:
    """
    Run MAIA environment test with specified parameters.

    Args:
        environment_name: Name of the environment
        num_steps: Number of steps per episode
        num_episodes: Number of episodes to run
        probe_locations: List of probe coordinates
        obs_normalization: Normalization strategy
        seed: Random seed (optional)
        logger: Logger instance

    Raises:
        RuntimeError: If environment creation or execution fails
    """
    logger.info("=" * 70)
    logger.info(f"MAIA Environment Test: {environment_name}")
    logger.info("=" * 70)

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        logger.info(f"Random seed set to: {seed}")

    # =========================================================================
    # COMPREHENSIVE MAIA ENVIRONMENT CONFIGURATION EXAMPLE
    # =========================================================================
    # This section demonstrates ALL available configuration options.
    # Uncomment and modify sections as needed for your specific use case.
    # =========================================================================

    # --- BASIC CONFIGURATION ---
    # Required: environment name
    # Optional: Hugging Face repository ID
    # hf_repo_id = 'dynamicslab/HydroGym-environments'  # default

    # --- DATA MANAGEMENT ---
    # Control caching behavior and offline usage
    # use_clean_cache = True   # True: fresh workspace copy (recommended for production)
    #                          # False: reuse cached workspace (faster for development)
    # local_fallback_dir = '/path/to/local/environments'  # For offline usage

    # --- PROBE CONFIGURATION ---
    # Define probe locations for flow field sampling
    # Format: [x0, y0, x1, y1, ...] for 2D or [x0, y0, z0, x1, y1, z1, ...] for 3D

    # Example 1: Default grid (from command-line arguments)
    # probe_locations = probe_locations  # Already computed from args

    # Example 2: Custom wake sampling grid
    # probe_x = np.linspace(1.0, 10.0, 16)  # 16 points downstream
    # probe_y = np.linspace(-2.0, 2.0, 8)   # 8 points in crossflow
    # X, Y = np.meshgrid(probe_x, probe_y)
    # probe_locations = [coord for x, y in zip(X.flat, Y.flat) for coord in (x, y)]

    # Example 3: Sparse strategic probes
    # probe_locations = [
    #     2.0, 0.0,    # Near wake, centerline
    #     2.0, 0.5,    # Near wake, upper
    #     2.0, -0.5,   # Near wake, lower
    #     4.0, 0.0,    # Mid wake, centerline
    #     6.0, 0.0,    # Far wake, centerline
    # ]

    # Example 4: 3D probe grid for cube/sphere
    # probe_x = np.linspace(1.0, 5.0, 8)
    # probe_y = np.linspace(-1.0, 1.0, 4)
    # probe_z = np.linspace(-1.0, 1.0, 4)
    # X, Y, Z = np.meshgrid(probe_x, probe_y, probe_z)
    # probe_locations = [coord for x, y, z in zip(X.flat, Y.flat, Z.flat)
    #                          for coord in (x, y, z)]

    # --- OBSERVATION NORMALIZATION ---
    # Strategy for normalizing observation values

    # Option 1: 'U_inf' normalization (RECOMMENDED)
    # Normalizes by freestream velocity - best for learning
    # obs_normalization_strategy = 'U_inf'  # default

    # Option 2: Probewise mean/std normalization
    # Each probe normalized independently by its own statistics
    # obs_normalization_strategy = 'probewise_mean_std'

    # Option 3: No normalization (raw values)
    # Use when handling normalization in your RL algorithm
    # obs_normalization_strategy = 'none'

    # Option 4: Customized normalization
    # Provide your own offset and scale arrays
    # obs_normalization_strategy = 'customized'
    # obs_loc = np.zeros(num_probes * num_components)      # Offset (mean)
    # obs_scale = np.ones(num_probes * num_components)     # Scale (std)

    # --- ADVANCED CONFIGURATION ---
    # Custom configuration file (if not using HF Hub default)
    # configuration_file = '/path/to/custom/config.yaml'

    # Testing mode (enables additional checks/logging)
    # is_testing = True

    # Create environment
    logger.info("Creating environment from Hugging Face Hub...")
    logger.info("Configuration:")
    logger.info(f"  Environment: {environment_name}")
    logger.info(f"  Probe locations: {len(probe_locations) // 2} probes (2D)")
    logger.info(f"  Observation normalization: {obs_normalization}")

    try:
        # ========== BASIC CONFIGURATION ==========
        # Minimal configuration - just environment name
        # env = maia.from_hf(environment_name)

        # ========== STANDARD CONFIGURATION ==========
        # Production configuration with custom probes and normalization
        env = maia.from_hf(
            environment_name,
            use_clean_cache=False,  # Use cached workspace (already prepared)
            probe_locations=probe_locations,
            obs_normalization_strategy=obs_normalization,
        )

        # ========== ADVANCED CONFIGURATION ==========
        # All available options (uncomment as needed)
        # env = maia.from_hf(
        #     environment_name,
        #     # Data management
        #     hf_repo_id='dynamicslab/HydroGym-environments',
        #     use_clean_cache=False,
        #     local_fallback_dir=None,
        #     # Observation configuration
        #     probe_locations=probe_locations,
        #     obs_normalization_strategy='U_inf',
        #     # obs_loc=custom_offset,         # For 'customized' strategy
        #     # obs_scale=custom_scale,        # For 'customized' strategy
        #     # Configuration override
        #     # configuration_file='/path/to/config.yaml',
        #     # Testing
        #     # is_testing=False,
        # )

        logger.info("✓ Environment created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create environment: {e}")
        raise RuntimeError(f"Environment creation failed: {e}") from e

    # =========================================================================
    # ENVIRONMENT-SPECIFIC CONFIGURATION REFERENCE
    # =========================================================================
    #
    # CYLINDER (Blowing/Suction Jets) [2D/3D]:
    #   - environment: Cylinder_2D_Re{Re}
    #   - Available Re: 200, 1000, etc.
    #   - num_inputs: Depends on jet configuration (typically 1-2)
    #   - Actuation: Synthetic jet actuators on cylinder surface
    #   - Reward: -(|C_D| + omega * |C_L|)
    #   - Observation: Probe velocities in wake region
    #   - Typical probes: Wake sampling grid (x: 1-8, y: -2 to 2)
    #
    # ROTARY CYLINDER (Rotation Control) [2D/3D]:
    #   - environment: RotaryCylinder_2D_Re{Re}
    #   - Available Re: 200, 1000, etc.
    #   - num_inputs: 1 (rotation rate)
    #   - Actuation: Cylinder rotation speed
    #   - Reward: -(|C_D| + omega * |C_L|)
    #   - Observation: Probe velocities in wake region
    #
    # PINBALL (Three Rotating Cylinders) [2D/3D]:
    #   - environment: Pinball_2D_Re{Re}
    #   - Available Re: 30, 75, 100,  etc.
    #   - num_inputs: 3 (one rotation rate per cylinder)
    #   - Actuation: Independent rotation of 3 cylinders
    #   - Reward: -(|C_D| + omega * |C_L|) summed over all cylinders
    #   - Observation: Probe velocities in wake region
    #   - Classic configuration: Equilateral triangle arrangement
    #
    # JET PINBALL (Three Cylinders with Jets) [2D/3D]:
    #   - environment: JetPinball_2D_Re{Re}
    #   - num_inputs: Depends on jet configuration
    #   - Actuation: Synthetic jets on cylinder surfaces
    #   - Reward: Similar to Pinball
    #
    # SQUARE CYLINDER [2D/3D]:
    #   - environment: SquareCylinder_2D_Re{Re}
    #   - Available Re: 200, 1000, etc.
    #   - num_inputs: Depends on actuation type
    #   - Actuation: Jets or other control mechanisms
    #   - Reward: -(|C_D| + omega * |C_L|)
    #
    # CAVITY (Open Cavity) [2D/3D]:
    #   - environment: Cavity_2D_Re{Re}
    #   - Available Re: 4140, 7500
    #   - num_inputs: 1
    #   - Actuation: 1 Jet
    #   - Reward: Problem-specific (mixing, vortex control, etc.)
    #
    # CAVITY3JET (Cavity with 3 Jets) [2D/3D]:
    #   - environment: Cavity3Jet_2D_Re{Re}
    #   - num_inputs: 3 (one per jet)
    #   - Actuation: Three wall jets for flow control
    #   - Reward: Enhanced mixing or flow stabilization
    #
    # NACA0012 (Airfoil) [2D/3D]:
    #   - environment: NACA0012_2D_Re{Re}
    #   - Available Re: 100, 1000, etc.
    #   - num_inputs: 3
    #   - Actuation: Jets
    #   - Reward: -(|C_D| + omega * |C_L|) with target lift
    #   - Observation: Pressure/velocity probes near airfoil
    #
    # NACA0012GUST (Airfoil with Gust) [2D/3D]:
    #   - environment: NACA0012Gust_2D_Re{Re}
    #   - num_inputs: Similar to NACA0012
    #   - Reward: Minimize lift fluctuations due to gust
    #   - Challenge: Unsteady incoming flow
    #
    # CUBE (3D Bluff Body):
    #   - environment: Cube_3D_Re{Re}
    #   - Available Re: 300, 3700, etc.
    #   - num_inputs: Depends on actuation type
    #   - Actuation: Surface jets or other 3D control
    #   - Reward: -(|C_D| + omega * |C_L| + omega * |C_S|)  # Side force
    #   - Observation: 3D probe grid (x, y, z coordinates)
    #
    # SPHERE (3D Bluff Body):
    #   - environment: Sphere_3D_Re{Re}
    #   - Available Re: 300, 3700, etc.
    #   - num_inputs: Depends on actuation type
    #   - Actuation: Surface jets or suction
    #   - Reward: -(|C_D|)
    #   - Observation: 3D probe grid in wake region

    try:
        # Log environment information
        logger.info("-" * 70)
        logger.info("Environment Information:")
        logger.info(f"  Observation space: {env.observation_space}")
        logger.info(f"  Action space: {env.action_space}")
        logger.info(f"  Max episode steps: {env.max_episode_steps}")
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
        # Always close environment
        logger.info("\nClosing environment and signaling MAIA to finish...")
        try:
            env.close()
            logger.info("✓ Environment closed successfully")
        except Exception as e:
            logger.warning(f"⚠ Error during cleanup: {e}")


def main():
    """Main entry point for the environment test script."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Test MAIA CFD environment with MPMD coupling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--environment",
        required=True,
        help=(
            "Environment name (e.g., Cylinder_2D_Re200, Sphere_3D_Re100). "
            "Format: TYPE_DIMENSION_Re{Reynolds}. "
            "Available types: Cylinder, RotaryCylinder, Pinball, JetPinball, "
            "SquareCylinder, Cavity, Cavity3Jet, NACA0012, NACA0012Gust, Cube, Sphere"
        ),
    )
    parser.add_argument(
        "--num-steps", type=int, default=10, help="Number of simulation steps per episode (default: 10)"
    )
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run (default: 1)")
    parser.add_argument(
        "--probe-x-range",
        type=str,
        default="1.0,8.0,8",
        help=(
            "X-axis probe range as 'min,max,num' (default: 1.0,8.0,8). "
            "Creates num equally-spaced probes from min to max in x-direction. "
            "For wake sampling: start > 1.0 (downstream of body)"
        ),
    )
    parser.add_argument(
        "--probe-y-range",
        type=str,
        default="-1.0,1.0,5",
        help=(
            "Y-axis probe range as 'min,max,num' (default: -1.0,1.0,5). "
            "Creates num equally-spaced probes from min to max in y-direction. "
            "For wake sampling: span crossflow region of interest"
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    parser.add_argument(
        "--obs-norm",
        type=str,
        default="U_inf",
        choices=["U_inf", "probewise_mean_std", "none", "customized"],
        help=(
            "Observation normalization strategy (default: U_inf). "
            "U_inf: normalize by freestream velocity (recommended for RL). "
            "probewise_mean_std: per-probe normalization by mean/std. "
            "none: raw observation values. "
            "customized: user-defined (requires code modification)"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (DEBUG level)")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(verbose=args.verbose)

    # Parse probe ranges
    try:
        x_range = parse_range_arg(args.probe_x_range)
        y_range = parse_range_arg(args.probe_y_range)
    except ValueError as e:
        logger.error(f"Invalid probe range: {e}")
        sys.exit(1)

    # Create probe locations
    try:
        probe_locations = create_probe_locations(x_range, y_range, logger)
    except Exception as e:
        logger.error(f"Failed to create probe locations: {e}")
        sys.exit(1)

    # Run environment test
    try:
        run_environment_test(
            environment_name=args.environment,
            num_steps=args.num_steps,
            num_episodes=args.num_episodes,
            probe_locations=probe_locations,
            obs_normalization=args.obs_norm,
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
