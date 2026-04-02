#!/usr/bin/env python3
"""
Zero-shot multi-policy deployment demo on the small wing case.
NOTE: Only DRL policies are supported due to the precomplied Nek5000 executable.
NOTE: This case is ONLY used for demonstration purpose, one should not expect any physical results.

This chapter demonstrates deployment (not training): multiple policy groups are
assigned to actuator subsets, each policy can update at its own DRL interval,
and all groups interact simultaneously with the same environment.

Usage:
    mpirun -np 1 python test_nek_pettingzoo.py : -np 12 nek5000

    mpirun -np 1 python test_nek_pettingzoo.py \
        --policy-template ./meta_policy_small_wing_template.py \
        --policy-root /path/to/legacy_runs \
        --steps 3000 \
        : -np 12 nek5000
"""

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from hydrogym.nek import NekEnv
from hydrogym.nek.pettingzoo_env import make_pettingzoo_env


@dataclass
class PolicySpec:
    """Specification for one control-group policy."""

    name: str
    x_min: float
    x_max: float
    side: str
    algorithm: str
    drl_step: int
    action_min: float
    action_max: float
    u_tau: float = 1.0
    baseline_dudy: float = 1.0
    agent_run_name: str = ""
    policy: str = ""
    model_path: Optional[str] = None


class ZeroController:
    """Always returns zero action."""

    def predict(self, obs_batch, deterministic=True):
        n = int(obs_batch.shape[0])
        return np.zeros((n, 1), dtype=np.float32), None


class BlowingController:
    """Constant blowing/suction."""

    def __init__(self, amplitude: float):
        self.amplitude = float(amplitude)

    def predict(self, obs_batch, deterministic=True):
        n = int(obs_batch.shape[0])
        acts = np.ones((n, 1), dtype=np.float32) * self.amplitude
        return acts, None


class MultiPolicyDeployer:
    """Assign and deploy multiple policies over PettingZoo actuator agents."""

    def __init__(self, env, specs: List[PolicySpec], policy_root: Optional[str]):
        self.env = env
        self.specs = specs
        self.policy_root = Path(policy_root).expanduser().resolve() if policy_root else None
        self._base_env = self._unwrap_base_env(env)

        self.agents = list(env.possible_agents)
        self.agent_to_idx = {agent: i for i, agent in enumerate(self.agents)}

        self._validate_actuator_info()
        self.groups = self._build_groups()

    @staticmethod
    def _unwrap_base_env(env):
        """Unwrap NekPettingZooEnv -> NekParallelEnv -> NekEnv."""
        base = env
        for _ in range(3):
            if hasattr(base, "env"):
                base = base.env
            else:
                break
        return base

    def _validate_actuator_info(self):
        if not hasattr(self._base_env, "actuator_info"):
            raise RuntimeError("Base environment does not expose actuator_info; cannot assign policy regions.")
        info = self._base_env.actuator_info
        for key in ("x", "y"):
            if key not in info:
                raise RuntimeError(f"actuator_info is missing key '{key}'")

    def _agents_in_region(self, spec: PolicySpec) -> List[str]:
        x = np.asarray(self._base_env.actuator_info["x"])
        y = np.asarray(self._base_env.actuator_info["y"])

        x_min = min(spec.x_min, spec.x_max)
        x_max = max(spec.x_min, spec.x_max)
        side = spec.side.upper()

        if side == "SS":
            mask = (x >= x_min) & (x <= x_max) & (y > 0)
        elif side == "PS":
            mask = (x >= x_min) & (x <= x_max) & (y < 0)
        else:
            raise ValueError(f"Unsupported side '{spec.side}', use SS or PS")

        idx = np.where(mask)[0].tolist()
        return [self.agents[i] for i in idx]

    def _load_controller(self, spec: PolicySpec):
        algo = spec.algorithm.upper()

        if algo == "ZERO":
            return ZeroController()
        if algo == "BL":
            return BlowingController(spec.action_max)

        if algo not in ("PPO", "TD3", "DDPG"):
            raise ValueError(f"Unsupported algorithm '{spec.algorithm}' in {spec.name}")

        try:
            if algo == "PPO":
                from stable_baselines3 import PPO as RLAlgorithm
            elif algo == "TD3":
                from stable_baselines3 import TD3 as RLAlgorithm
            else:
                from stable_baselines3 import DDPG as RLAlgorithm
        except Exception as exc:
            raise RuntimeError(
                f"stable_baselines3 import failed for {algo}. Install SB3 or change algorithm in policy template."
            ) from exc

        model_path = spec.model_path
        if not model_path:
            if not self.policy_root:
                raise ValueError(f"Policy '{spec.name}' requires --policy-root or explicit model_path")
            model_path = self.policy_root / spec.agent_run_name / "logs" / (f"{spec.agent_run_name}-{spec.policy}")

        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for {spec.name}: {model_path}")

        return RLAlgorithm.load(str(model_path))

    def _build_groups(self):
        groups = []
        assigned = set()

        for spec in self.specs:
            agents = self._agents_in_region(spec)
            idx = np.array([self.agent_to_idx[a] for a in agents], dtype=int)
            controller = self._load_controller(spec)

            overlap = [a for a in agents if a in assigned]
            if overlap:
                print(
                    f"[WARN] {spec.name} overlaps {len(overlap)} agents with previous groups. "
                    "Last-assigned policy takes precedence.",
                    flush=True,
                )

            assigned.update(agents)
            groups.append(
                {
                    "spec": spec,
                    "agents": agents,
                    "idx": idx,
                    "controller": controller,
                    "cached_actions": np.zeros((len(agents), 1), dtype=np.float32),
                    "step_counter": 0,
                    "reward_sum": 0.0,
                }
            )

            print(
                f"[META] {spec.name:<10s} | alg={spec.algorithm:<4s} | "
                f"x=[{spec.x_min:.3f},{spec.x_max:.3f}] side={spec.side} | "
                f"agents={len(agents)} | drl_step={spec.drl_step}",
                flush=True,
            )

        unassigned = len(self.agents) - len(assigned)
        if unassigned > 0:
            print(
                f"[META] {unassigned} agents are not covered by any policy. They will receive zero action.", flush=True
            )

        return groups

    def compute_actions(self, obs_dict: Dict[str, np.ndarray], step: int):
        """
        Compute the actions for the group.
        The action is updated when the step is 0 or the step counter is divisible by the DRL step.

        NOTE: observation and actions are normalized by the u_tau on Nek5000 end.
        Therefore, the action/observation are NOT normalized by the u_tau on the DRL side.

        Args:
          obs_dict: The observation dictionary for the group.
          step: The current step.
        Returns:
          actions: The actions for the group.
        """
        actions = {agent: np.zeros(1, dtype=np.float32) for agent in self.agents}

        for group in self.groups:
            spec = group["spec"]
            n_agents = len(group["agents"])
            if n_agents == 0:
                continue

            # When the step is 0 or the step counter is divisible by the DRL step, update the action
            refresh = (step == 0) or ((group["step_counter"] % max(spec.drl_step, 1)) == 0)

            if refresh:
                print(f"[ACTION] Refreshing at step={step} for {spec.name} (drl_step={spec.drl_step})", flush=True)
                obs_batch = np.vstack(
                    [np.asarray(obs_dict[agent], dtype=np.float32).reshape(1, -1) for agent in group["agents"]]
                )

                obs_batch = obs_batch / max(spec.u_tau, 1e-12)

                pred = group["controller"].predict(obs_batch, deterministic=True)
                if isinstance(pred, tuple):
                    raw_actions = pred[0]
                else:
                    raw_actions = pred

                raw_actions = np.asarray(raw_actions, dtype=np.float32).reshape(n_agents, -1)
                # Clip the action to the action bounds, note that the action is NOT normalized by the u_tau yet
                clipped = np.clip(raw_actions[:, 0], spec.action_min, spec.action_max)
                group["cached_actions"] = clipped.reshape(n_agents, 1)

            for i, agent in enumerate(group["agents"]):
                actions[agent] = np.array([group["cached_actions"][i, 0]], dtype=np.float32)

            group["step_counter"] += 1

        return actions

    def update_rewards(self, rewards_dict: Dict[str, float]):
        """
        Update the rewards for the groups.
        The reward is inverted as the reward was scaled in the Environment.
        Subsequently, the reward is scaled by the mean of the dUdy for the group.
        Args:
          rewards_dict: The rewards dictionary for the groups.
        """
        for group in self.groups:
            if len(group["agents"]) == 0:
                continue
            # [YW-MOD] invert the reward as the reward was scaled in the Environment
            invert_dUdy = 1.0 - np.array([rewards_dict[a] for a in group["agents"]], dtype=np.float32)
            # Calculate the mean of the dUdy for the group
            mean_dUdy = float(np.mean(invert_dUdy))
            # Scale the reward by the baseline_dudy
            scaled = 1.0 - mean_dUdy / max(group["spec"].baseline_dudy, 1e-12)
            # Add the scaled reward to the reward sum
            group["reward_sum"] += scaled

    def print_reward_table(self, step: int):
        print("\n" + "-" * 56, flush=True)
        print(f"| Step={step:<6d} | Policy Reward Summary", flush=True)
        print("-" * 56, flush=True)
        for group in self.groups:
            spec = group["spec"]
            avg = group["reward_sum"] / max(group["step_counter"], 1)
            print(f"| {spec.name:<10s} | alg={spec.algorithm:<4s} | avg_scaled_R={avg:>10.6f} |", flush=True)
        print("-" * 56, flush=True)


def load_template_module(path: Path):
    """Load a python template module by file path."""
    spec = importlib.util.spec_from_file_location("meta_policy_template", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load template module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_policy_specs(raw_specs: List[dict]) -> List[PolicySpec]:
    """Parse dict list into PolicySpec objects."""
    specs = []
    for i, item in enumerate(raw_specs):
        name = item.get("name", f"CTRL{i:03d}")
        x_range = item.get("x_range")
        if not isinstance(x_range, (list, tuple)) or len(x_range) != 2:
            raise ValueError(f"{name}: x_range must be [x_min, x_max]")

        bnds = item.get("action_bounds", [-1.0, 1.0])
        if not isinstance(bnds, (list, tuple)) or len(bnds) != 2:
            raise ValueError(f"{name}: action_bounds must be [min, max]")

        specs.append(
            PolicySpec(
                name=name,
                x_min=float(x_range[0]),
                x_max=float(x_range[1]),
                side=str(item.get("side", "SS")).upper(),
                algorithm=str(item.get("algorithm", "ZERO")).upper(),
                drl_step=int(item.get("drl_step", 1)),
                action_min=float(bnds[0]),
                action_max=float(bnds[1]),
                u_tau=float(item.get("u_tau", 1.0)),
                baseline_dudy=float(item.get("baseline_dudy", 1.0)),
                agent_run_name=str(item.get("agent_run_name", "")),
                policy=str(item.get("policy", "")),
                model_path=item.get("model_path"),
            )
        )
    return specs


def main():
    parser = argparse.ArgumentParser(description="Zero-shot multi-policy wing deployment demo")
    parser.add_argument(
        "--policy-template",
        type=str,
        default=str(Path(__file__).with_name("meta_policy_small_wing_template.py")),
        help="Python template path defining ENV_NAME/NPROC/NUM_STEPS/POLICY_SPECS",
    )
    parser.add_argument("--env", type=str, default=None, help="Override environment name")
    parser.add_argument("--nproc", type=int, default=None, help="Override number of Nek workers")
    parser.add_argument("--steps", type=int, default=None, help="Override rollout steps")
    parser.add_argument("--policy-root", type=str, default=None, help="Override policy root folder for RL policies")
    parser.add_argument("--local-dir", type=str, default=None, help="Local fallback directory for environment packages")
    parser.add_argument("--log-every", type=int, default=100, help="Print reward summary every N steps")
    args = parser.parse_args()

    # Validate PettingZoo installation early with user-friendly error.
    try:
        import pettingzoo

        print(f"PettingZoo version: {pettingzoo.__version__}")
    except ImportError:
        print("ERROR: PettingZoo is not installed. Install with: pip install pettingzoo")
        sys.exit(1)

    template_path = Path(args.policy_template).expanduser().resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Policy template not found: {template_path}")

    template = load_template_module(template_path)
    env_name = args.env or getattr(template, "ENV_NAME", "NACA4412_3D_Re75000_AOA5")
    nproc = args.nproc or int(getattr(template, "NPROC", 12))
    num_steps = args.steps or int(getattr(template, "NUM_STEPS", 3000))
    policy_root = args.policy_root or getattr(template, "POLICY_ROOT", None)

    raw_specs = getattr(template, "POLICY_SPECS", None)
    if not raw_specs:
        raise ValueError("Template must define non-empty POLICY_SPECS list")
    policy_specs = parse_policy_specs(raw_specs)

    print("\n=== Zero-Shot Wing Demo (PettingZoo) ===")
    print(f"Environment: {env_name}")
    print(f"Nek5000 processes: {nproc}")
    print(f"Rollout steps: {num_steps}")
    print(f"Policy groups: {len(policy_specs)}")

    base_env = NekEnv.from_hf(
        env_name,
        nproc=nproc,
        use_clean_cache=False,
        local_fallback_dir=args.local_dir,
    )
    # Set the baseline_dudy as 1 as the reward varies with streamwise position
    base_env.baseline_dudy = 1.0

    # Keep behavior aligned with existing getting_started scripts.
    from hydrogym.nek.nek_lib.nek_utils import NEK_INIT

    nek_init = NEK_INIT(nek=base_env.conf.simulation, drl=base_env.conf.runner, rank_folder=base_env.run_folder)
    nek_init.rewrite_REA_v17()

    env = make_pettingzoo_env(base_env)

    deployer = MultiPolicyDeployer(env=env, specs=policy_specs, policy_root=policy_root)

    obs_dict, info = env.reset()
    total_reward = {agent: 0.0 for agent in env.agents}

    for step in range(num_steps):
        actions = deployer.compute_actions(obs_dict, step)
        obs_dict, rewards_dict, terminated_dict, truncated_dict, infos_dict = env.step(actions)

        for agent in env.agents:
            total_reward[agent] += float(rewards_dict[agent])

        deployer.update_rewards(rewards_dict)

        if (step % max(args.log_every, 1)) == 0:
            first_agent = env.agents[0]
            print(
                f"Step {step:6d} | first-agent reward={rewards_dict[first_agent]: .6e} | "
                f"total={total_reward[first_agent]: .6e}",
                flush=True,
            )
            deployer.print_reward_table(step)

        if any(terminated_dict.values()) or any(truncated_dict.values()):
            print(f"Episode terminated at step {step}", flush=True)
            break

    rewards = np.array([total_reward[a] for a in env.agents], dtype=np.float32)
    print("\n" + "=" * 72)
    print("Zero-shot deployment summary")
    print(f"  Agents: {len(env.agents)}")
    print(f"  Mean total reward: {float(np.mean(rewards)):.6f}")
    print(f"  Min total reward:  {float(np.min(rewards)):.6f}")
    print(f"  Max total reward:  {float(np.max(rewards)):.6f}")
    print("=" * 72)

    env.close()


if __name__ == "__main__":
    main()
