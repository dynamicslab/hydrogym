#!/usr/bin/env python3
"""
Train a shared multi-agent policy with Stable-Baselines3 on a Nek wall-actuation environment.

Every wall actuator is one agent; all agents share one policy and every agent's transition enters the replay
buffer (the multi-agent setting of the HydroGym channel, e.g. ``TCFlarge_3D_Re207``).

All algorithm settings are read from the ``runner`` section of the environment configuration (replay buffer, batch
size, seed, TD3 noise and update settings, training and gradient steps, episode length and count); command-line
options override them only when given. Network defaults (hidden layers): actor [8], critic [16, 16, 64]. The resolved
settings are written to ``<log_dir>/resolved_settings.json``.

Two wrappers are available:

* ``--wrapper shared`` (default): :func:`hydrogym.nek.shared_policy.make_shared_policy_vec_env`, array-based, one
  agent per physically distinct wall point (duplicated GLL nodes on element edges and periodic ends removed);
* ``--wrapper supersuit``: the PettingZoo + SuperSuit route (one agent per GLL node, dictionary based, about 50 times
  slower for the channel).

Run (MPMD, one Python rank + the Nek ranks):
    mpirun -np 1 python train_sb3_pettingzoo.py --env TCFlarge_3D_Re207 --algo TD3 : -np 512 nek5000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from hydrogym.nek import NekEnv, make_pettingzoo_env
from hydrogym.nek.shared_policy import make_shared_policy_vec_env

# (runner key, argparse dest, type, fallback when neither config nor CLI give a value)
RUNNER_SETTINGS = [
    ("learning_rate", "learning_rate", float, 1e-3),
    ("batch_size", "batch_size", int, 256),
    ("buffer_size", "buffer_size", int, 1_000_000),
    ("learning_starts", "learning_starts", int, 100),
    ("train_steps", "train_freq", int, 300),
    ("gradient_steps", "gradient_steps", int, 64),
    ("tau", "tau", float, 0.005),
    ("policy_delay", "policy_delay", int, 2),
    ("target_policy_noise", "target_policy_noise", float, 0.2),
    ("target_noise_clip", "target_noise_clip", float, 0.5),
    ("action_noise", "action_noise", float, 0.1),
    ("seed", "seed", int, None),
    ("nb_interactions", "episode_length", int, None),
    ("nb_episodes", "episodes", int, None),
    ("ckpt_int", "ckpt_episodes", int, 5),
]


def resolve_settings(base_env, args):
    """Config ``runner`` values, overridden by command-line options that were given explicitly."""
    resolved, source = {}, {}
    for key, dest, typ, fallback in RUNNER_SETTINGS:
        cli = getattr(args, dest)
        cfg = base_env._get_config_value(("runner", key), default=None)
        if cli is not None:
            resolved[dest], source[dest] = typ(cli), "command line"
        elif cfg is not None:
            resolved[dest], source[dest] = typ(cfg), f"config runner.{key}"
        else:
            resolved[dest], source[dest] = (typ(fallback) if fallback is not None else None), "default"
    resolved["gamma"], source["gamma"] = args.gamma, "command line/default"
    resolved["actor_arch"], source["actor_arch"] = list(args.actor_arch), "command line/default"
    resolved["critic_arch"], source["critic_arch"] = list(args.critic_arch), "command line/default"
    return resolved, source


def build_supersuit_vec_env(base_env):
    """The PettingZoo + SuperSuit route (one agent per GLL node)."""
    import numpy as np
    from stable_baselines3.common.vec_env import VecEnvWrapper
    from supersuit import black_death_v3, pad_action_space_v0, pad_observations_v0, pettingzoo_env_to_vec_env_v1

    env = make_pettingzoo_env(base_env)
    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    env = black_death_v3(env)
    env = pettingzoo_env_to_vec_env_v1(env)

    class VecEnvCompatWrapper(VecEnvWrapper):
        """Gymnasium (obs, info) / 5-tuple API to the 4-tuple VecEnv API."""

        def reset(self, **kwargs):
            result = self.venv.reset(**kwargs)
            return result[0] if isinstance(result, tuple) else result

        def step_wait(self):
            result = self.venv.step_wait()
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                return obs, reward, np.logical_or(terminated, truncated), info
            return result

    return VecEnvCompatWrapper(env)


def train(args):
    """Build the environment and the agent from the configuration, then train."""
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import VecNormalize

    if args.algo == "PPO":
        from stable_baselines3 import PPO as Algorithm
    elif args.algo == "TD3":
        from stable_baselines3 import TD3 as Algorithm
    else:
        from stable_baselines3 import SAC as Algorithm

    env_config = {
        "environment_name": args.env,
        "nproc": args.nproc,
        "use_clean_cache": False,
        "local_fallback_dir": args.local_dir,
        "configuration_file": args.config_file,
        "rescale_actions": True,  # actions in [-1, 1], rescaled by the maximum amplitude
        "normalize_input": "utau",
    }
    # Amplitude bounds are config-tree entries under "runner" (hydrogym/nek/configs.py::Runner); override them only
    # when asked, by dotted path.
    if args.ctrl_max_amp is not None:
        env_config["runner.ctrl_min_amp"] = -abs(args.ctrl_max_amp)
        env_config["runner.ctrl_max_amp"] = abs(args.ctrl_max_amp)
    base_env = NekEnv(env_config=env_config)

    settings, source = resolve_settings(base_env, args)
    seed = settings["seed"]
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if args.wrapper == "shared":
        env = make_shared_policy_vec_env(base_env, deduplicate=not args.keep_duplicates)
        print(f"Shared-policy VecEnv: {env.num_envs} agents from {base_env.n_actuators} actuators")
    else:
        env = build_supersuit_vec_env(base_env)
        print(f"SuperSuit VecEnv: {env.num_envs} agents")
    if seed is not None:
        env.seed(seed)
    env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_obs=0.0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"{args.algo}_{args.wrapper}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": settings["learning_rate"],
        "gamma": settings["gamma"],
        "seed": seed,
        "verbose": 1,
        "tensorboard_log": str(log_dir),
    }
    if args.algo == "PPO":
        # PPO is not used in the literature for this case and is not guaranteed to work.
        model_kwargs["policy_kwargs"] = {"net_arch": {"pi": settings["actor_arch"], "vf": settings["critic_arch"]}}
        model_kwargs["n_steps"] = args.n_steps
        model_kwargs["batch_size"] = settings["batch_size"]
    else:
        model_kwargs["policy_kwargs"] = {"net_arch": {"pi": settings["actor_arch"], "qf": settings["critic_arch"]}}
        model_kwargs.update(
            batch_size=settings["batch_size"],
            buffer_size=settings["buffer_size"],
            learning_starts=settings["learning_starts"],
            train_freq=(settings["train_freq"], "step"),
            gradient_steps=settings["gradient_steps"],
            tau=settings["tau"],
            action_noise=NormalActionNoise(
                mean=np.zeros(env.action_space.shape[0]),
                sigma=settings["action_noise"] * np.ones(env.action_space.shape[0]),
            ),
        )
        if args.algo == "TD3":
            model_kwargs.update(
                policy_delay=settings["policy_delay"],
                target_policy_noise=settings["target_policy_noise"],
                target_noise_clip=settings["target_noise_clip"],
            )

    record = {
        "env": args.env,
        "algo": args.algo,
        "wrapper": args.wrapper,
        "agents": env.num_envs,
        "actuators": int(base_env.n_actuators),
        "settings": settings,
        "source": source,
    }
    (log_dir / "resolved_settings.json").write_text(json.dumps(record, indent=2))
    for k, v in settings.items():
        print(f"  {k:20s} = {v!r:24s} ({source[k]})")

    model = Algorithm(**model_kwargs)

    # Total vectorised steps: episodes x episode length unless --total-timesteps is given. SB3 counts transitions
    # (num_envs per vectorised step).
    if args.total_timesteps is not None:
        total = args.total_timesteps
    elif settings["episodes"] and settings["episode_length"]:
        total = settings["episodes"] * settings["episode_length"] * env.num_envs
    else:
        raise SystemExit("Give --total-timesteps or set runner.nb_episodes and runner.nb_interactions.")

    save_every = max(1, (settings["ckpt_episodes"] or 1) * (settings["episode_length"] or 1))  # vectorised steps

    class SaveVecNormalizeCallback(CheckpointCallback):
        """Also store the VecNormalize statistics with every checkpoint."""

        def _on_step(self) -> bool:
            result = super()._on_step()
            if self.n_calls % self.save_freq == 0:
                self.training_env.save(str(Path(self.save_path) / f"vec_normalize_{self.num_timesteps}_steps.pkl"))
            return result

    callback = SaveVecNormalizeCallback(save_freq=save_every, save_path=str(log_dir), name_prefix="model")
    print(f"Log directory: {log_dir}; total transitions {total}; checkpoint every {save_every} vectorised steps")

    try:
        model.learn(total_timesteps=total, callback=[callback], tb_log_name=f"{args.algo}_run")
        model.save(log_dir / "model_final.zip")
        env.save(str(log_dir / "vec_normalize_final.pkl"))
        print(f"Saved {log_dir / 'model_final.zip'}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--env", required=True, help="Environment name, e.g. TCFlarge_3D_Re207")
    parser.add_argument("--local-dir", default=None, help="Local env directory")
    parser.add_argument("--nproc", type=int, default=10, help="Nek5000 processes")
    parser.add_argument("--config-file", type=str, default=None, help="Config file (None = auto-detect)")
    parser.add_argument("--wrapper", choices=["shared", "supersuit"], default="shared")
    parser.add_argument("--keep-duplicates", action="store_true", help="shared wrapper: one agent per GLL node")
    parser.add_argument("--ctrl-max-amp", type=float, default=None, help="override runner.ctrl_max_amp (and min)")

    parser.add_argument("--algo", default="TD3", choices=["PPO", "TD3", "SAC"])
    parser.add_argument("--total-timesteps", type=int, default=None, help="default: episodes x length x agents")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout length")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor-arch", type=int, nargs="+", default=[8], help="hidden sizes of the actor")
    parser.add_argument("--critic-arch", type=int, nargs="+", default=[16, 16, 64], help="hidden sizes of the critic")
    # Settings normally taken from the config "runner" section; given here they override it.
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--train-freq", type=int, default=None, help="runner.train_steps")
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--policy-delay", type=int, default=None)
    parser.add_argument("--target-policy-noise", type=float, default=None)
    parser.add_argument("--target-noise-clip", type=float, default=None)
    parser.add_argument("--action-noise", type=float, default=None, help="std of the exploration noise")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=None, help="runner.nb_interactions")
    parser.add_argument("--episodes", type=int, default=None, help="runner.nb_episodes")
    parser.add_argument("--ckpt-episodes", type=int, default=None, help="runner.ckpt_int")

    parser.add_argument("--log-dir", default="./logs")
    args = parser.parse_args()

    try:
        train(args)
    except Exception as e:  # report and exit non-zero (MPMD jobs otherwise hang on the Nek side)
        print(f"\nTraining failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
