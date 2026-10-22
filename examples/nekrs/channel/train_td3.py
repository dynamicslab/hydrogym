"""
TD3 on the NekRS channel through the multi-agent wrapper (one agent per actuator, shared policy), following
hydrogym/examples/nek/getting_started/3_pettingzoo/train_sb3_pettingzoo.py and the runner block of
pettingzoo_tcflarge_re207.yml (train every 300 actions, 64 gradient steps, lr 1e-3, tau 0.005, policy delay 2,
target noise 0.2 / clip 0.5, action noise 0.1, nets qf [16, 16, 64], pi [8]).
Episodes start from the snapshot library of the YAML (initial_conditions). Launch:
    GPU=3 NP=1 RUN=runs/td3 SCRIPT=train_td3.py ./run.sh --episodes 10
"""

import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise

from hydrogym.nekrs import NekRSFlowEnv
from hydrogym.nekrs.multiagent import ParallelToVecEnv, make_parallel_env

ap = argparse.ArgumentParser()
ap.add_argument("--conf", default="channel_td3.yml")
ap.add_argument("--nproc", type=int)
ap.add_argument("--run-root", default="runs/td3")
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--seed", type=int, default=20010603)
ap.add_argument("--train-freq", type=int, default=300)
ap.add_argument("--gradient-steps", type=int, default=64)
ap.add_argument("--batch-size", type=int, default=2048)
ap.add_argument("--buffer-size", type=int, default=5_000_000)
ap.add_argument("--learning-rate", type=float, default=1e-3)
ap.add_argument("--learning-starts", type=int, default=100)
ap.add_argument("--save-every", type=int, default=1, help="save the policy every n episodes (0 = only at the end)")
ap.add_argument("--eval-actions", type=int, default=300, help="deterministic episode from snapshot 0 (0 = skip)")
args = ap.parse_args()

env = NekRSFlowEnv(args.conf, run_root=args.run_root, nproc=args.nproc)
par = make_parallel_env(env)
vec = ParallelToVecEnv(par)
print(f"[TD3] {vec.num_envs} agents ({type(par).__module__}.{type(par).__name__}), obs {vec.observation_space.shape}", flush=True)


class EpisodeLog(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.f = open(path, "w")
        self.r, self.d, self.t0 = [], [], time.time()

    def _on_step(self):
        self.r.append(float(np.mean(self.locals["rewards"])))
        self.d.append(env._last_dudy)
        if self.locals["dones"][0]:
            n = len(self.r)
            rec = {
                "episode": env.episode - 1,
                "index": vec.episode_info.get("index"),
                "actions": n,
                "mean_reward": float(np.mean(self.r)),
                "last_third_reward": float(np.mean(self.r[2 * n // 3 :])),
                "last_third_dUdy": float(np.mean(self.d[2 * n // 3 :])),
                "timesteps": int(self.num_timesteps),
                "wall_s": round(time.time() - self.t0, 1),
            }
            print("[TD3] " + json.dumps(rec), flush=True)
            self.f.write(json.dumps(rec) + "\n")
            self.f.flush()
            self.r, self.d = [], []
            if args.save_every and (rec["episode"] + 1) % args.save_every == 0:
                self.model.save(os.path.join(args.run_root, "checkpoints", f"td3_ep{rec['episode']:03d}"))
        return True


# dUdy of the last action for the log (the per-agent infos only carry the time)
_step = env.step


def _logged_step(a):
    out = _step(a)
    env._last_dudy = out[4]["dUdy"]
    return out


env.step = _logged_step

try:
    model = TD3(
        "MlpPolicy",
        vec,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=(args.train_freq, "step"),
        gradient_steps=args.gradient_steps,
        # independent noise per agent (a plain NormalActionNoise would broadcast one sample to all agents)
        action_noise=VectorizedActionNoise(NormalActionNoise(np.zeros(1), 0.1 * np.ones(1)), vec.num_envs),
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        policy_kwargs={"net_arch": {"qf": [16, 16, 64], "pi": [8]}},
        seed=args.seed,
        verbose=0,
        device="cuda",
    )
    total = args.episodes * env.episode_actions * vec.num_envs
    model.learn(total_timesteps=total, callback=EpisodeLog(os.path.join(args.run_root, "td3_episodes.jsonl")))
    model.save(os.path.join(args.run_root, "td3_final"))

    def evaluate(policy, name):
        """One episode from snapshot 0: the trained policy (deterministic) or zero actuation."""
        obs, info = par.reset(options={"index": 0})
        o = np.stack([obs[a] for a in par.possible_agents])
        r, d = [], []
        for k in range(args.eval_actions):
            act = policy(o)
            obs, rew, term, trunc, _ = par.step({a: act[i] for i, a in enumerate(par.possible_agents)})
            o = np.stack([obs[a] for a in par.possible_agents])
            r.append(np.mean(list(rew.values())))
            d.append(env._last_dudy)
            if any(term.values()):
                break
        n = len(r)
        return {"policy": name, "index": 0, "actions": n, "mean_reward": float(np.mean(r)),
                "last_third_reward": float(np.mean(r[2 * n // 3 :])), "last_third_dUdy": float(np.mean(d[2 * n // 3 :]))}

    if args.eval_actions:
        res = [evaluate(lambda o: model.predict(o, deterministic=True)[0], "td3"),
               evaluate(lambda o: np.zeros((len(o), 1), dtype=np.float32), "zero")]
        for rec in res:
            print("[TD3-EVAL] " + json.dumps(rec), flush=True)
        json.dump(res, open(os.path.join(args.run_root, "td3_eval.json"), "w"), indent=1)
finally:
    vec.close()
