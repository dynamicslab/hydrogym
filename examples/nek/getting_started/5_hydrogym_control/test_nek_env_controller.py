#!/usr/bin/env python3
"""
NEK5000 Environment Test Script (MPMD Mode)
===========================================

Test NEK5000 environments with MPMD coupling (Python + Nek5000).

Usage:
    # Using MAIA pattern (recommended)
    mpirun -np 1 python test_nek_env_controller.py --env MiniChannel_Re180 --nproc 10 : -np 10 nek5000

    # Using config file (legacy)
    mpirun -np 1 python test_nek_env_controller.py --config test_config.yml --nproc 10 : -np 10 nek5000

Arguments:
    --env: Environment name from HuggingFace (e.g., MiniChannel_Re180)
    --nproc: Number of Nek5000 processes (required)
    --config: Path to YAML configuration file (optional, for legacy usage)
    --steps: Number of simulation steps (optional, overrides config)
    --num-episodes: Number of episodes to run (default: 1)
    --controller: Controller type (OC, BL, SIN, ZERO, default: OC)
    --verbose: Enable verbose output
    --local-dir: Local fallback directory for environments
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from hydrogym.nek import NekEnv
from hydrogym.nek.nek_lib.nek_utils import NEK_INIT

# Force unbuffered output for MPMD mode
sys.stdout = open(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), "w", buffering=1)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with timestamps."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout
    )
    return logging.getLogger(__name__)


def create_controller(env, controller_type: str, logger: logging.Logger):
    """Create controller based on type."""
    logger.info(f"Creating {controller_type} controller")

    if controller_type.upper() == "OC":
        # Opposition control
        def controller(t, obs, env):
            # Oppose the velocity at sensor location (obs is flattened 1D array)
            # For opposition control, use negative of observation values
            return -obs[1::2]  # YW: Again note that we are opposing the wall-normal velocity (index = 1)
    elif controller_type.upper() == "BL":
        # Constant blowing
        def controller(t, obs, env):
            return np.ones(env.action_space.shape, dtype=np.float32) * 0.5
    elif controller_type.upper() == "SIN":
        # Sinusoidal wave
        def controller(t, obs, env):
            return np.sin(2 * np.pi * t) * 0.5
    elif controller_type.upper() == "ZERO":
        # No control
        def controller(t, obs, env):
            return np.zeros(env.action_space.shape, dtype=np.float32)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    return controller


def run_nek_test(
    env_name: Optional[str],
    nproc: int,
    config_path: Optional[str],
    num_steps: Optional[int],
    num_episodes: int,
    controller_type: str,
    local_dir: Optional[str],
    logger: logging.Logger,
) -> None:
    """
    Run NEK5000 environment test.

    Args:
        env_name: Environment name for MAIA pattern (e.g., 'MiniChannel_Re180')
        nproc: Number of Nek5000 processes
        config_path: Path to YAML configuration (legacy, optional)
        num_steps: Number of steps per episode (None = use config)
        num_episodes: Number of episodes
        controller_type: Type of controller to use
        local_dir: Local fallback directory for environments
        logger: Logger instance
    """
    logger.info("=" * 70)
    logger.info("NEK5000 Environment Test (MPMD Mode)")
    logger.info("=" * 70)

    # Create environment using modern pattern
    logger.info("Creating NEK5000 environment...")
    try:
        if env_name:
            # MAIA pattern (recommended)
            logger.info(f"Using MAIA pattern: environment={env_name}, nproc={nproc}")
            env = NekEnv.from_hf(
                env_name,
                nproc=nproc,
                use_clean_cache=False,
                local_fallback_dir=local_dir,
                nb_interactions=num_steps,  # Override if provided
            )
        elif config_path:
            # Legacy pattern with config file
            logger.info(f"Using legacy pattern with config file: {config_path}")
            from omegaconf import OmegaConf

            from hydrogym.nek.configs import Config

            # Load config
            config = OmegaConf.merge(
                OmegaConf.structured(Config()),
                OmegaConf.load(config_path),
            )

            # Override steps if provided
            if num_steps is not None:
                if hasattr(config, "episode"):
                    config.episode.max_interactions = num_steps
                elif hasattr(config, "runner"):
                    config.runner.nb_interactions = num_steps
                logger.info(f"Overriding steps to: {num_steps}")

            env = NekEnv(conf=config, reward_agg="mean")
        else:
            raise ValueError("Must provide either --env (MAIA pattern) or --config (legacy pattern)")

        # Rewrite the par file to ensure the simulation configuration is correct
        nek_init = NEK_INIT(nek=env.conf.simulation, drl=env.conf.runner, rank_folder=env.run_folder)
        nek_init.rewrite_REA_v19()

        logger.info("✓ Environment created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create environment: {e}")
        raise RuntimeError(f"Environment creation failed: {e}") from e

    try:
        # Log environment info
        logger.info("-" * 70)
        logger.info("Environment Information:")
        logger.info(f"  Observation space: {env.observation_space}")
        logger.info(f"  Action space: {env.action_space}")
        if hasattr(env, "max_episode_steps"):
            logger.info(f"  Max episode steps: {env.max_episode_steps}")
        logger.info(f"  Number of actuators: {env.action_space.shape}")
        logger.info("-" * 70)

        # Create controller
        controller = create_controller(env, controller_type, logger)

        # Run episodes
        total_reward = 0.0

        for episode in range(num_episodes):
            logger.info(f"\nEpisode {episode + 1}/{num_episodes}")
            logger.info("-" * 70)

            # Reset environment
            try:
                obs, info = env.reset()
                logger.info("✓ Environment reset")
                logger.info(f"  Initial observation shape: {obs.shape}")
            except Exception as e:
                logger.error(f"✗ Reset failed: {e}")
                raise

            episode_reward = 0.0
            episode_steps = 0

            # Standard RL loop
            logger.info(f"Running episode with {controller_type} controller...")

            try:
                # Get max steps from environment
                max_steps = env.nb_interactions

                # Standard RL loop: step through the environment
                for step in range(max_steps):
                    # Compute action from controller
                    # Get dt from config
                    dt = env.conf.simulation.dt if hasattr(env.conf.simulation, "dt") else 0.001
                    t = step * abs(dt)
                    action = controller(t, obs, env)

                    # Step the environment
                    obs, reward, terminated, truncated, info = env.step(action)

                    episode_reward += reward
                    episode_steps += 1

                    if step % 10 == 0:  # Log every 10 steps
                        logger.debug(f"  Step {step}/{max_steps}, reward: {reward:.6f}")

                    # Check if episode is done
                    if terminated or truncated:
                        logger.info(f"  Episode terminated at step {step}")
                        break

                total_reward += episode_reward
                logger.info("✓ Episode completed")
                logger.info(f"Episode {episode + 1} summary:")
                logger.info(f"  Steps: {episode_steps}")
                logger.info(f"  Total reward: {episode_reward:.6f}")
                if episode_steps > 0:
                    logger.info(f"  Average reward: {episode_reward / episode_steps:.6f}")

            except Exception as e:
                logger.error(f"✗ Episode failed: {e}")
                raise

        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("Test Summary:")
        logger.info(f"  Episodes completed: {num_episodes}")
        if total_reward != 0:
            logger.info(f"  Total reward: {total_reward:.6f}")
        logger.info("=" * 70)

    finally:
        # Clean up
        logger.info("\nClosing environment and signaling Nek5000 to finish...")
        try:
            env.close()
            logger.info("✓ Environment closed successfully")
        except Exception as e:
            logger.warning(f"⚠ Error during cleanup: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test NEK5000 environment with MPMD coupling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Environment specification (either --env or --config)
    parser.add_argument(
        "--env", type=str, default=None, help="Environment name from HuggingFace (e.g., MiniChannel_Re180)"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file (legacy, optional)")
    parser.add_argument("--nproc", type=int, required=True, help="Number of Nek5000 processes (required)")
    parser.add_argument("--local-dir", type=str, default=None, help="Local fallback directory for environments")

    # Simulation parameters
    parser.add_argument("--steps", type=int, default=None, help="Number of steps per episode (overrides config)")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run (default: 1)")

    # Controller parameters
    parser.add_argument(
        "--controller",
        type=str,
        default="OC",
        choices=["OC", "BL", "SIN", "ZERO"],
        help="Controller type (default: OC)",
    )

    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Validate arguments
    if not args.env and not args.config:
        parser.error("Must provide either --env (MAIA pattern) or --config (legacy pattern)")

    # Setup logging
    logger = setup_logging(verbose=args.verbose)

    # Run test
    try:
        run_nek_test(
            env_name=args.env,
            nproc=args.nproc,
            config_path=args.config,
            num_steps=args.steps,
            num_episodes=args.num_episodes,
            controller_type=args.controller,
            local_dir=args.local_dir,
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
