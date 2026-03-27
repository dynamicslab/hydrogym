#!/usr/bin/env python3
"""
NEK5000 PettingZoo Multi-Agent Environment Test
================================================

Minimal example showing how to:
1. Create a Nek environment with direct instantiation
2. Wrap it with NekPettingZooEnv for PettingZoo compatibility
3. Run a simple control loop using PettingZoo's API
4. Test PettingZoo-compatible multi-agent control

Usage:
    # Run with default steps
    mpirun -np 1 python test_nek_pettingzoo.py : -np 10 nek5000

    # Override number of steps
    mpirun -np 1 python test_nek_pettingzoo.py --steps 100 : -np 10 nek5000

Note:
    - Requires PettingZoo: pip install pettingzoo
    - NekPettingZooEnv wraps NekParallelEnv with PettingZoo API
    - Compatible with PettingZoo-specific tools and libraries
    - Actions and observations are dicts with agent names as keys
"""

import sys
import argparse
from pathlib import Path
import numpy as np

from hydrogym.nek import NekEnv
from hydrogym.nek.pettingzoo_env import make_pettingzoo_env


def main():
    parser = argparse.ArgumentParser(description="Test Nek5000 PettingZoo multi-agent environment")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("--env", type=str, default="MiniChannel_Re180", help="Environment name")
    parser.add_argument("--nproc", type=int, default=10, help="Number of Nek5000 processes")
    parser.add_argument("--local-dir", type=str, default=None, help="Local fallback directory for environments")
    parser.add_argument("--config-file", type=str, default=None, help="Config file (None = auto-detect)")
    args = parser.parse_args()

    # Check if PettingZoo is available
    try:
        import pettingzoo

        print(f"PettingZoo version: {pettingzoo.__version__}")
    except ImportError:
        print("ERROR: PettingZoo is not installed.")
        print("Install it with: pip install pettingzoo")
        sys.exit(1)

    # Create base environment
    print(f"\nCreating Nek5000 environment: {args.env}")
    print("Method: Direct NekEnv + PettingZoo wrapper")

    env_config = {
        "environment_name": args.env,
        "nproc": args.nproc,
        "hostfile": "",
        "hf_repo_id": "dynamicslab/HydroGym-environments",
        "use_clean_cache": False,
        "local_fallback_dir": args.local_dir,
        "configuration_file": args.config_file,
    }

    base_env = NekEnv(env_config=env_config)

    # Wrap with PettingZoo multi-agent environment
    env = make_pettingzoo_env(base_env)

    print("\nPettingZoo Multi-Agent Environment info:")
    print("=" * 80)
    print(f"  Environment name: {env.metadata.get('name', 'nek_v1')}")
    print(f"  Number of possible agents: {len(env.possible_agents)}")
    print(f"  Number of active agents: {len(env.agents)}")
    print("\n  Agent details (first 3):")
    for i, agent_name in enumerate(env.possible_agents[:3]):
        obs_space = env.observation_space(agent_name)
        act_space = env.action_space(agent_name)
        print(f"    {i}: {agent_name}")
        print(f"       obs_shape={obs_space.shape}, act_shape={act_space.shape}")
        print(f"       act_bounds=[{act_space.low[0]:.6f}, {act_space.high[0]:.6f}]")
    if len(env.possible_agents) > 3:
        print(f"    ... and {len(env.possible_agents) - 3} more agents")
    print("=" * 80)

    # Reset environment
    print("\nResetting environment...")
    obs_dict, info = env.reset()
    print(f"  Received observations for {len(obs_dict)} agents")
    first_agent = env.agents[0]
    print(f"  First agent '{first_agent[:40]}...' obs shape: {obs_dict[first_agent].shape}")

    # Run simulation with multi-agent control
    max_steps = args.steps
    print(f"\nRunning {max_steps} steps with PettingZoo API...")
    print("  Strategy: All agents apply zero control (baseline)")

    total_reward = {agent: 0.0 for agent in env.agents}
    episode_length = 0

    for step in range(max_steps):
        # Define actions for each agent
        # Strategy 1: Zero control (baseline)
        actions = {agent: np.zeros(1, dtype=np.float32) for agent in env.agents}

        # Strategy 2: Random actions (uncomment to test)
        # actions = {
        #     agent: env.action_space(agent).sample()
        #     for agent in env.agents
        # }

    # Strategy 3: Cooperative strategy - all agents use same signal
    # signal = np.sin(step * 0.1) * 0.01
    # actions = {agent: np.array([signal], dtype=np.float32) for agent in env.agents}

    # [YW-MOD] Add Opposition Control Strategy, oppose to the wall-normal velocity (-1)
    actions = {agent: -1.0 * obs_dict[agent][:-1] for agent in env.agents}
    # [YW-MOD] End

        # Step environment
        obs_dict, rewards_dict, terminated_dict, truncated_dict, infos_dict = env.step(actions)

        # Accumulate rewards
        for agent in env.agents:
            total_reward[agent] += rewards_dict[agent]

        episode_length += 1

        if step % 10 == 0:
            # Print stats for first agent
            first_agent = env.agents[0]
            agent_reward = rewards_dict[first_agent]
            agent_total = total_reward[first_agent]
            print(f"  Step {step:3d}: agent[0] reward = {agent_reward:8.6f}, total = {agent_total:10.6f}")

        # Check if episode ended (PettingZoo style)
        if any(terminated_dict.values()) or any(truncated_dict.values()):
            print(f"  Episode terminated at step {step}")
            break

    # Summary
    print("\n" + "=" * 80)
    print("PettingZoo Test completed!")
    print(f"  Total steps: {episode_length}")

    # Calculate statistics across all agents
    rewards_list = [total_reward[agent] for agent in env.agents]
    avg_total_reward = np.mean(rewards_list)
    min_total_reward = np.min(rewards_list)
    max_total_reward = np.max(rewards_list)

    print(f"  Average total reward (across agents): {avg_total_reward:.6f}")
    print(f"  Min total reward: {min_total_reward:.6f}")
    print(f"  Max total reward: {max_total_reward:.6f}")
    print(f"  Average reward per step: {avg_total_reward / episode_length:.6f}")
    print("\n  Per-agent total rewards (first 3):")
    for i, agent in enumerate(env.agents[:3]):
        print(f"    {agent[:40]}...: {total_reward[agent]:.6f}")
    if len(env.agents) > 3:
        print(f"    ... and {len(env.agents) - 3} more agents")
    print("=" * 80)

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
