from typing import Callable, Iterable, Optional, Tuple, Dict
import os
import time

import numpy as np
import scipy.io as sio

from hydrogym.core import CallbackBase

from .nek_lib.nek_utils import show_title, show_end


def integrate(
    env,
    t_span: Tuple[float, float],
    dt: Optional[float] = None,
    callbacks: Iterable[CallbackBase] = [],
    controller: Optional[Callable] = None,
    max_steps: Optional[int] = None,
):
  """
  Integrate a Nek environment through time.

  Args:
    env: Nek environment (NekEnv, NekParallelEnv, or NekPettingZooEnv)
    t_span: Tuple of (start_time, end_time)
    dt: Time step (optional, uses env's default if not provided)
    callbacks: List of callbacks to evaluate throughout the solve
    controller: Controller object or function. Supports multiple formats:
      - SB3-style object: Object with `predict()` method (e.g.,
        `model.predict(obs, state=..., episode_start=..., deterministic=True)`)
      - Legacy function: `action = controller(t, obs, env)` or
        `action = controller(t, obs)`
    max_steps: Maximum number of steps (optional)

  Returns:
    The environment after integration
  """
  show_title()
  t_start, t_end = t_span
  iter = 0
  t = t_start

  # Reset environment
  if hasattr(env, 'reset'):
    obs = env.reset()
  else:
    obs = None

  # Get time step from environment config if not provided
  if dt is None:
    if hasattr(env, 'conf') and hasattr(env.conf, 'simulation'):
      # Try to get dt from config
      dt = getattr(env.conf.simulation, 'dt', 0.01)
    else:
      dt = 0.01  # Default fallback

  # Calculate max steps if not provided
  if max_steps is None:
    max_steps = int((t_end - t_start) / dt) + 1

  # Track last reward for callbacks
  last_reward = 0.0

  # Initialize controller state for SB3-style controllers
  controller_states = None
  episode_start = np.ones((1,), dtype=bool)  # True on first step

  # Initialize history recording if enabled
  vars_record = False
  vars_record_freq = 1
  agent_dict = {}
  if hasattr(env, 'conf') and hasattr(env.conf, 'runner'):
    vars_record = getattr(env.conf.runner, 'vars_record', False)
    vars_record_freq = getattr(env.conf.runner, 'vars_record_freq', 1)

  if vars_record:
    # Get agent list based on environment type
    if hasattr(env, 'possible_agents'):
      # NekParallelEnv or NekPettingZooEnv (multi-agent)
      agents_list = list(env.possible_agents)
    elif hasattr(env, 'env') and hasattr(env.env, 'possible_agents'):
      # Wrapped parallel env
      agents_list = list(env.env.possible_agents)
    elif hasattr(env, 'pz_env'):
      # Legacy: old NekMARLGymWrapper
      agents_list = list(env.pz_env.possible_agents)
    else:
      # NekEnv (single-agent base)
      agents_list = ['agent_0']

    # Initialize dictionary for collecting data
    agent_dict = {
        agent: {
            "obs_rec": [],
            "act_rec": [],
            "rew_rec": [],
        } for agent in agents_list
    }

  # Main integration loop
  while t < t_end and iter < max_steps:
    # Get controller action if provided
    if controller is not None:
      # Try SB3-style controller: controller.predict(obs, state=states, episode_start=episode_starts)
      # This pattern matches stable_baselines3 model.predict() signature
      if hasattr(controller, 'predict'):
        # Controller is an object with predict() method (e.g., SB3 model)
        action, controller_states = controller.predict(
            obs,
            state=controller_states,
            episode_start=episode_start,
            deterministic=True)
      else:
        # Simple fallback: controller is a callable function with (t, obs, env) or (t, obs) signature
        try:
          action = controller(t, obs, env)
        except TypeError:
          action = controller(t, obs)
    else:
      # Default: zero action
      if hasattr(env, 'action_space'):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
      else:
        # For parallel_env, need dict of actions
        action = {
            agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
            for agent in env.possible_agents
        }

    # Step the environment
    info = {}
    if hasattr(env, 'step'):
      result = env.step(action)
      if isinstance(result, tuple) and len(result) >= 2:
        obs, reward, done, info = result[:4]
        last_reward = reward
        # Store reward in env for callbacks
        env.last_reward = reward
        if done:
          break
      else:
        obs = result
        info = {}

    # Update time
    t = t_start + (iter + 1) * dt
    iter += 1

    # Update episode_start flag (False after first step)
    episode_start = np.zeros((1,), dtype=bool)

    # Record history if enabled
    if vars_record and (iter % vars_record_freq == 0):
      # Dict-based environments (NekParallelEnv, NekPettingZooEnv)
      if isinstance(obs, dict) and isinstance(action, dict):
        for agent in agents_list:
          if agent in obs and agent in action:
            agent_dict[agent]['obs_rec'].append(obs[agent].reshape(1, -1))
            agent_dict[agent]['act_rec'].append(action[agent].reshape(1, -1))
            # Get reward from info
            if isinstance(info, dict) and 'reward_per_agent' in info:
              agent_rew = np.array([[info['reward_per_agent'][agent]]])
            else:
              agent_rew = np.array([[last_reward / len(agents_list)]])
            agent_dict[agent]['rew_rec'].append(agent_rew)
      # Array-based environment (NekEnv)
      elif isinstance(obs, np.ndarray) and isinstance(action, np.ndarray):
        # Check if it's a single-agent env or wrapped multi-agent with arrays
        if len(agents_list) == 1:
          # Single agent case (NekEnv used directly)
          agent = agents_list[0]
          agent_dict[agent]['obs_rec'].append(obs.reshape(1, -1))
          agent_dict[agent]['act_rec'].append(action.reshape(1, -1))
          agent_dict[agent]['rew_rec'].append(np.array([[last_reward]]))
        else:
          # Legacy: concatenated arrays for multi-agent (old wrapper)
          # Try to get per-agent sizes from env
          obs_per_agent = getattr(env, 'per_agent_obs_size', None) or getattr(
              env, 'obs_per_actuator', None)
          act_per_agent = getattr(env, 'per_agent_act_size', None) or 1

          if obs_per_agent is not None:
            for i, agent in enumerate(agents_list):
              start_obs = i * obs_per_agent
              end_obs = (i + 1) * obs_per_agent
              start_act = i * act_per_agent
              end_act = (i + 1) * act_per_agent

              agent_obs = obs[start_obs:end_obs].reshape(1, -1)
              agent_act = action[start_act:end_act].reshape(1, -1)

              # Get per-agent reward
              if isinstance(info, dict) and 'reward_per_actuator' in info:
                agent_rew = np.array([[info['reward_per_actuator'][i]]])
              elif isinstance(info, dict) and 'reward_per_agent' in info:
                agent_rew = np.array([[info['reward_per_agent'][agent]]])
              else:
                agent_rew = np.array([[last_reward / len(agents_list)]])

              agent_dict[agent]['obs_rec'].append(agent_obs)
              agent_dict[agent]['act_rec'].append(agent_act)
              agent_dict[agent]['rew_rec'].append(agent_rew)

      if iter % (vars_record_freq * 10) == 0:  # Print every 10 records
        print(
            f"[INTEGRATE] AT {iter}/{max_steps} SAVE Trajectories", flush=True)

    # Call callbacks
    for cb in callbacks:
      cb(iter, t, env)

  # Close callbacks
  for cb in callbacks:
    if hasattr(cb, 'close'):
      cb.close()

  # Save recorded history if enabled
  if vars_record and len(agent_dict) > 0:
    # Concatenate all recorded data
    for agent in agent_dict.keys():
      for key in agent_dict[agent].keys():
        if len(agent_dict[agent][key]) > 0:
          agent_dict[agent][key] = np.concatenate(
              agent_dict[agent][key], axis=0)
        else:
          agent_dict[agent][key] = np.array([])

    # Determine save path - prioritize RUN_PATH file (source of truth from nek_initial.py)
    # This matches the folder created by nek_initial.py
    save_path = None

    # First, try to read from RUN_PATH file using agent_run_name (like exec-script does)
    # This is the most reliable since it's created by nek_initial.py
    if hasattr(env, 'conf') and hasattr(env.conf, 'runner'):
      agent_run_name = getattr(env.conf.runner, 'agent_run_name', None)
      if agent_run_name is not None:
        dir_files_path = f"dir-files/RUN_PATH_{agent_run_name}.txt"
        if os.path.exists(dir_files_path):
          with open(dir_files_path, 'r') as f:
            save_path = f.readline().strip()
            # Handle relative paths (like "./runs/1998/env_006")
            if save_path.startswith('./'):
              save_path = save_path[2:]

    # Fallback: try to get the folder from the environment
    if save_path is None:
      # Try NekEnv directly
      if hasattr(env, 'run_folder'):
        save_path = str(env.run_folder)
      # Try wrapped env (NekParallelEnv, NekPettingZooEnv)
      elif hasattr(env, 'env') and hasattr(env.env, 'run_folder'):
        save_path = str(env.env.run_folder)
      # Legacy: old pz_env
      elif hasattr(env, 'pz_env') and hasattr(env.pz_env, 'folder'):
        save_path = str(env.pz_env.folder)
      elif hasattr(env, 'folder'):
        save_path = str(env.folder)

    # Last resort: construct from config if RUN_PATH file doesn't exist
    if save_path is None and hasattr(env, 'conf') and hasattr(
        env.conf, 'runner'):
      run_name = getattr(env.conf.logging, 'run_name', None)
      if run_name:
        # Determine rank folder based on evaluation mode (matching nek_initial.py)
        if getattr(env.conf.runner, 'evaluation', False):
          rank = getattr(env.conf.runner, 'rank', 0)
          save_path = f"runs/{run_name}/env_{rank:03d}"
        else:
          save_path = f"runs/{run_name}/train"

    if save_path is None:
      save_path = "."

    # Ensure directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save as .mat file
    filename = os.path.join(save_path, f'vars_record_{int(time.time())}.mat')
    sio.savemat(filename, agent_dict)
    print(f"[INTEGRATE] SAVED RECORD to {filename}", flush=True)

  show_end()
  return env
