#!/usr/bin/env python3
"""
MAIA Environment Test Script
=============================

Production-ready script for testing MAIA environments with MPMD coupling.

Usage:
    mpirun -np 1 python env_test.py --environment Cylinder_2D_Re200 : -np 1 maia properties.toml

Arguments:
    --environment: Name of the environment to test (required)
    --num-steps: Number of simulation steps to run (default: 10)
    --num-episodes: Number of episodes to run (default: 1)
    --probe-x-range: X-axis range for probes (default: 1.0,8.0,8)
    --probe-y-range: Y-axis range for probes (default: -1.0,1.0,5)
    --seed: Random seed for reproducibility (optional)
    --obs-norm: Observation normalization strategy (default: U_inf)
    --verbose: Enable verbose output (default: False)
"""

import sys
import argparse
import logging
from typing import List, Tuple, Optional

import numpy as np

# Force unbuffered output for MPMD mode
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), 'w', buffering=1)

# Import hydrogym (lazy import prevents firedrake loading in MPMD mode)
import hydrogym.maia as maia


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
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
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
    parts = arg_str.split(',')
    if len(parts) != 3:
        raise ValueError(f"Range must be in format 'min,max,num', got: {arg_str}")
    return float(parts[0]), float(parts[1]), int(parts[2])


def create_probe_locations(
    x_range: Tuple[float, float, int],
    y_range: Tuple[float, float, int],
    logger: logging.Logger
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

    logger.info(f"Creating probe grid:")
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
    logger: logging.Logger
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

    # Create environment
    logger.info("Creating environment from Hugging Face Hub...")
    try:
        env = maia.from_hf(
            environment_name,
            use_clean_cache=False,
            probe_locations=probe_locations,
            obs_normalization_strategy=obs_normalization,
        )
        logger.info("✓ Environment created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create environment: {e}")
        raise RuntimeError(f"Environment creation failed: {e}") from e

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
                logger.info(f"✓ Environment reset")
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
        epilog=__doc__
    )

    parser.add_argument(
        "--environment",
        required=True,
        help="Environment name (e.g., Cylinder_2D_Re200)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of steps per episode (default: 10)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)"
    )
    parser.add_argument(
        "--probe-x-range",
        type=str,
        default="1.0,8.0,8",
        help="X-axis probe range as 'min,max,num' (default: 1.0,8.0,8)"
    )
    parser.add_argument(
        "--probe-y-range",
        type=str,
        default="-1.0,1.0,5",
        help="Y-axis probe range as 'min,max,num' (default: -1.0,1.0,5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    parser.add_argument(
        "--obs-norm",
        type=str,
        default="U_inf",
        choices=["U_inf", "probewise_mean_std", "none", "customized"],
        help="Observation normalization strategy (default: U_inf)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

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
            logger=logger
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