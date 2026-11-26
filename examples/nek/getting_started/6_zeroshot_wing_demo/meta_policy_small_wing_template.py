#!/usr/bin/env python3
"""
Template configuration for zero-shot multi-policy deployment on the small wing.

A template for multi-policy deployment: one policy group per wall segment,
directly consumable by ``zeroshot_demo_pettingzoo.py`` in this folder.
"""

# Default environment used in this chapter
ENV_NAME = "NACA4412_3D_Re75000_AOA5"

# Number of Nek5000 worker processes for this case
NPROC = 12

# Total rollout steps for demonstration
NUM_STEPS = 3000

# Root directory containing the policy run folders.
# Example expected layout for RL entries:
#   <POLICY_ROOT>/<agent_run_name>/logs/<agent_run_name>-<policy>
POLICY_ROOT = "./legacy_runs"

# Per-policy deployment specification.
# Required fields:
#   - name: policy label
#   - x_range: [x_min, x_max]
#   - side: "SS" (y>0) or "PS" (y<0)
#   - algorithm: "PPO"/"TD3"/"DDPG" (SB3 policies) or the baselines "BL" (constant blowing at the upper
#     action bound) and "ZERO"
#   - drl_step: action refresh interval (action is held between updates)
#   - action_bounds: [min, max]
# Optional fields (for RL algorithms):
#   - agent_run_name
#   - policy
# Optional scaling fields:
#   - u_tau: extra scaling of the observations (obs / u_tau) for policies trained with another
#     normalisation; 1 keeps the scaling of the Nek5000 side
#   - baseline_dudy: reference wall-normal velocity gradient of the uncontrolled flow on the segment
POLICY_SPECS = [
    {
        "name": "CTRL000",
        "x_range": [0.23, 0.40],
        "side": "SS",
        "algorithm": "BL",  # constant blowing, no model needed
        "agent_run_name": "my_run",  # folder under POLICY_ROOT that holds the model (RL algorithms)
        "policy": "rl_model_<steps>_steps",  # model file name in <agent_run_name>/logs/ (RL algorithms)
        "drl_step": 4,  # actions between two refreshes of this group's action
        "u_tau": 1.0,  # extra observation scaling (1: as sent by Nek5000)
        "baseline_dudy": 1135.83,  # uncontrolled wall-normal velocity gradient on this segment
        "action_bounds": [-1.0, 1.0],  # clip range of the actions
    },
    {
        "name": "CTRL001",
        "x_range": [0.40, 0.50],
        "side": "SS",
        "algorithm": "BL",
        "agent_run_name": "my_run",
        "policy": "rl_model_<steps>_steps",
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
        "agent_run_name": "my_run",
        "policy": "rl_model_<steps>_steps",
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
        "agent_run_name": "my_run",
        "policy": "rl_model_<steps>_steps",
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
        "agent_run_name": "my_run",
        "policy": "rl_model_<steps>_steps",
        "drl_step": 4,
        "u_tau": 1.0,
        "baseline_dudy": 417.136,
        "action_bounds": [-1.0, 1.0],
    },
]
