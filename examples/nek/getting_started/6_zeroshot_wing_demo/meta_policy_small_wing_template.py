#!/usr/bin/env python3
"""
Template configuration for zero-shot multi-policy deployment on the small wing.

The structure mirrors the old MetaPolicy-style configuration, but is lightweight
and directly consumable by ``test_nek_pettingzoo.py`` in this folder.
"""

# Default environment used in this chapter
ENV_NAME = "NACA4412_3D_Re75000_AOA5"

# Number of Nek5000 worker processes for this case
NPROC = 12

# Total rollout steps for demonstration
NUM_STEPS = 3000

# Root directory containing legacy policy run folders.
# Example expected layout for RL entries:
#   <POLICY_ROOT>/<agent_run_name>/logs/<agent_run_name>-<policy>
POLICY_ROOT = "./legacy_runs"

# Per-policy deployment specification.
# Required fields:
#   - name: policy label
#   - x_range: [x_min, x_max]
#   - side: "SS" (y>0) or "PS" (y<0)
#   - algorithm: "PPO"/"TD3"/"DDPG" or baseline "BL"/"OC"/"ZERO"
#   - drl_step: action refresh interval (action is held between updates)
#   - action_bounds: [min, max]
# Optional fields (for RL algorithms):
#   - agent_run_name
#   - policy
# Optional scaling fields:
#   - u_tau
#   - baseline_dudy
POLICY_SPECS = [
    {
        "name": "CTRL000",
        "x_range": [0.23, 0.40],
        "side": "SS",
        "algorithm": "BL",  # This is a constant action policy, no model needed
        "agent_run_name": "601031",  # This is the agent run name, it is the name of the folder that contains the model
        "policy": "rl_model_749700000_steps",  # This is the policy name, it is the name of the policy
        "drl_step": 4,  # This is the action refresh interval, it is the number of steps between each action refresh
        "u_tau": 1.0,  # This is the u_tau, it is the u_tau of the environment, it is used to normalize the observation
        "baseline_dudy": 1135.83,  # This is the baseline_dudy, it is the baseline_dudy of the environment
        "action_bounds": [-1.0, 1.0],  # This is the action bounds, it is the action bounds of the environment
    },
    {
        "name": "CTRL001",
        "x_range": [0.40, 0.50],
        "side": "SS",
        "algorithm": "BL",
        "agent_run_name": "602031",
        "policy": "rl_model_716625000_steps",
        "drl_step": 5,
        "u_tau": 1.0,
        "baseline_dudy": 940.90,
        "action_bounds": [-1.0, 1.0],
    },
    {
        "name": "CTRL002",
        "x_range": [0.50, 0.70],
        "side": "SS",
        "algorithm": "BL",
        "agent_run_name": "603031",
        "policy": "rl_model_217800000_steps",
        "drl_step": 6,
        "u_tau": 1.0,
        "baseline_dudy": 759.46,
        "action_bounds": [-1.0, 1.0],
    },
    {
        "name": "CTRL003",
        "x_range": [0.70, 0.88],
        "side": "SS",
        "algorithm": "BL",
        "agent_run_name": "604031",
        "policy": "rl_model_2914890000_steps",
        "drl_step": 9,
        "u_tau": 1.0,
        "baseline_dudy": 485.32,
        "action_bounds": [-1.0, 1.0],
    },
    {
        "name": "CTRL004",
        "x_range": [0.23, 0.88],
        "side": "PS",
        "algorithm": "BL",
        "agent_run_name": "601031",
        "policy": "rl_model_749700000_steps",
        "drl_step": 4,
        "u_tau": 1.0,
        "baseline_dudy": 417.136,
        "action_bounds": [-1.0, 1.0],
    },
]
