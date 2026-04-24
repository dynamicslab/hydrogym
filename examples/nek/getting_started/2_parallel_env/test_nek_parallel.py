#!/usr/bin/env python3
"""
NEK5000 Parallel Multi-Agent Environment Test
==============================================

Minimal example showing how to:
1. Create a Nek environment with direct instantiation
2. Wrap it with NekParallelEnv for multi-agent control
3. Run a simple control loop with dict-based actions
4. Test coordinated multi-agent control

Usage:
    # Run with default steps
    mpirun -np 1 python test_nek_parallel.py : -np 10 nek5000

    # Override number of steps
    mpirun -np 1 python test_nek_parallel.py --steps 100 : -np 10 nek5000

Note:
    - NekParallelEnv treats each actuator as a separate agent
    - Actions and observations are dicts with agent names as keys
    - Each agent controls one actuator (scalar action)
    - Each agent receives observations from sensors near its actuator
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from hydrogym.nek import NekEnv
from hydrogym.nek.nek_lib.nek_utils import NEK_INIT
from hydrogym.nek.parallel_env import NekParallelEnv


def main():
    parser = argparse.ArgumentParser(description="Test Nek5000 parallel multi-agent environment")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("--env", type=str, default="TCFmini_3D_Re180", help="Environment name")
    parser.add_argument("--nproc", type=int, default=10, help="Number of Nek5000 processes")
    parser.add_argument("--local-dir", type=str, default=None, help="Local fallback directory for environments")
    parser.add_argument("--config-file", type=str, default=None, help="Config file (None = auto-detect)")
    args = parser.parse_args()

    # Create base environment
    print(f"\nCreating Nek5000 environment: {args.env}")
    print("Method: Direct NekEnv instantiation + NekParallelEnv wrapper")

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
    nek_init = NEK_INIT(nek=base_env.conf.simulation, drl=base_env.conf.runner, rank_folder=base_env.run_folder)

    # Rewrite the par file, v19 corresponds to the new Nek5000 format
    nek_init.rewrite_REA_v19()

    # Wrap with parallel multi-agent environment
    env = NekParallelEnv(base_env)

    print("\nMulti-Agent Environment info:")
    print("=" * 80)
    print(f"  Number of agents: {len(env.agents)}")
    print(f"  Observations per agent: {env.obs_per_agent}")
    print(f"  Actions per agent: {env.act_per_agent}")
    print("\n  Agent names (first 3):")
    for i, agent_name in enumerate(env.agents[:3]):
        obs_space = env.observation_space(agent_name)
        act_space = env.action_space(agent_name)
        print(f"    {i}: {agent_name}")
        print(f"       obs_shape={obs_space.shape}, act_shape={act_space.shape}")
        print(f"       act_bounds=[{act_space.low[0]:.6f}, {act_space.high[0]:.6f}]")
    if len(env.agents) > 3:
        print(f"    ... and {len(env.agents) - 3} more agents")
    print("=" * 80)

    # Reset environment
    print("\nResetting environment...")
    obs_dict, info = env.reset()
    print(f"  Received observations for {len(obs_dict)} agents")
    first_agent = env.agents[0]
    print(f"  First agent '{first_agent[:40]}...' obs shape: {obs_dict[first_agent].shape}")

    # Run simulation with multi-agent control
    max_steps = args.steps
    print(f"\nRunning {max_steps} steps with coordinated multi-agent control...")
    print("  Strategy: All agents apply zero control (baseline)")

    total_reward = {agent: 0.0 for agent in env.agents}

    for step in range(max_steps):
        # Define actions for each agent
        # Strategy 1: Zero control (baseline)
        actions = {agent: np.zeros(1, dtype=np.float32) for agent in env.agents}

        # Strategy 2: Uniform blowing (uncomment to test)
        # actions = {agent: np.ones(1, dtype=np.float32) * 0.01 for agent in env.agents}

        # Strategy 3: Opposition control per agent, opposing the wall-normal velocity in this case
        actions = {agent: -obs_dict[agent][1:] for agent in env.agents}

        # Step environment
        obs_dict, rewards_dict, terminated_dict, truncated_dict, infos_dict = env.step(actions)

        # Accumulate rewards
        for agent in env.agents:
            total_reward[agent] += rewards_dict[agent]

        if step % 10 == 0:
            # Print stats for first agent
            first_agent = env.agents[0]
            agent_reward = rewards_dict[first_agent]
            agent_total = total_reward[first_agent]
            print(f"  Step {step:3d}: agent[0] reward = {agent_reward:8.6f}, total = {agent_total:10.6f}")

        # Check if any agent terminated
        if any(terminated_dict.values()) or any(truncated_dict.values()):
            print(f"  Episode terminated at step {step}")
            break

    # Summary
    print("\n" + "=" * 80)
    print("Test completed!")
    print(f"  Total steps: {step + 1}")

    # Calculate average rewards across all agents
    avg_total_reward = np.mean([total_reward[agent] for agent in env.agents])
    avg_per_step = avg_total_reward / (step + 1)

    print(f"  Average total reward (across agents): {avg_total_reward:.6f}")
    print(f"  Average reward per step: {avg_per_step:.6f}")
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
