"""PPO / DDPG / TD3 on the fluidic pinball (HydroGym paper, Exp. 1 2D Re 100 and Exp. 2 3D Re 150).

Hyperparameters follow the paper's Methods where given; values it only gives as ranges are picked
from within the range. Uses Stable-Baselines3 instead of TorchRL.

Probes: an 8 x 5 wake grid (x in [1, 8], y in [-1, 1]); in 3D repeated on --z-planes spanwise
planes of the periodic span z in [-2, 2] (the paper does not state its layout). Observations are
u, v per probe, normalised by U_inf.

Outputs in --out: episodes.csv (per-episode mean reward and sum C_D), model_epN.zip every
--ckpt-every episodes, and a deterministic 200-step test rollout (eval_rollout.json).

Resumable for wall-time-limited jobs: with --max-hours the run stops at the first episode end
past the limit, saves model_latest.zip (+ replay buffer for DDPG/TD3) and state.json, and exits
with code 3; started again with the same --out it continues from there (each episode starts from
the uncontrolled restart file, so nothing else carries over between jobs).

Launch (MPMD, from a workspace made by ../getting_started/prepare_workspace.py):
    mpirun -np 1 python train_pinball.py --env Pinball_2D_Re100 --algo TD3 --out runs/td3_s0 \
        : -np 1 maia properties_run.toml
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import hydrogym.maia as maia
from stable_baselines3 import DDPG, PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

p = argparse.ArgumentParser()
p.add_argument("--env", default="Pinball_2D_Re100", choices=["Pinball_2D_Re100", "Pinball_3D_Re150"])
p.add_argument("--algo", required=True, choices=["PPO", "DDPG", "TD3"])
p.add_argument("--episodes", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
p.add_argument("--z-planes", type=int, default=3, help="3D only: number of spanwise probe planes")
p.add_argument("--ckpt-every", type=int, default=50)
p.add_argument("--max-hours", type=float, default=None, help="stop (resumably) after this wall time")
p.add_argument("--device", default="auto", help="torch device for the networks (auto, cpu, cuda)")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
t_start = time.time()

X, Y = np.meshgrid(np.linspace(1.0, 8.0, 8), np.linspace(-1.0, 1.0, 5))
if args.env.startswith("Pinball_3D"):
    # cell-centred planes, e.g. 3 planes -> z = -4/3, 0, 4/3
    Z = (np.arange(args.z_planes) + 0.5) * 4.0 / args.z_planes - 2.0
    probes = [c for z in Z for x, y in zip(X.flat, Y.flat) for c in (x, y, z)]
else:
    probes = [c for x, y in zip(X.flat, Y.flat) for c in (x, y)]
env = maia.from_hf(args.env, probe_locations=probes, obs_normalization_strategy="U_inf")
print(f"{args.env}: obs dim {env.observation_space.shape}, actions {env.action_space.shape}", flush=True)

# Capture force coefficients computed inside step() (a second query after step() deadlocks MPMD).
base = env.unwrapped
_orig_get_reward = base.get_reward
last = {}


def _get_reward_capture():
    r, od = _orig_get_reward()
    last["forces"] = np.asarray(od["forces"])
    return r, od


base.get_reward = _get_reward_capture
EP_LEN = 200
STATE = f"{args.out}/state.json"
LATEST, BUFFER = f"{args.out}/model_latest", f"{args.out}/replay_buffer_latest"


class EpisodeLogger(BaseCallback):
    def __init__(self, path, ckpt_every, ep0):
        super().__init__()
        new = ep0 == 0 or not os.path.exists(path)
        self.fh = open(path, "w" if new else "a", newline="")
        self.w = csv.writer(self.fh)
        if new:
            self.w.writerow(["episode", "steps", "mean_reward", "mean_sumCD", "last20_sumCD", "wall_s"])
        self.rew, self.cd, self.ep, self.t0, self.ckpt_every = [], [], ep0, time.time(), ckpt_every
        self.timed_out = False

    def _on_step(self):
        self.rew.append(float(self.locals["rewards"][0]))
        self.cd.append(float(last["forces"][:, 0].sum()))
        if self.locals["dones"][0]:
            self.ep += 1
            row = [self.ep, len(self.rew), np.mean(self.rew), np.mean(self.cd), np.mean(self.cd[-20:]),
                   round(time.time() - self.t0, 1)]
            self.w.writerow(row)
            self.fh.flush()
            print("EPISODE", *[f"{v:.4f}" if isinstance(v, float) else v for v in row], flush=True)
            self.rew, self.cd = [], []
            if self.ep % self.ckpt_every == 0:
                self.model.save(f"{args.out}/model_ep{self.ep}")
            if args.max_hours and time.time() - t_start > args.max_hours * 3600 and self.ep < args.episodes:
                self.timed_out = True
                return False
        return True


n_act = env.action_space.shape[0]
algo_cls = {"PPO": PPO, "DDPG": DDPG, "TD3": TD3}[args.algo]
ep0 = json.load(open(STATE))["episode"] if os.path.exists(STATE) else 0
if ep0:
    model = algo_cls.load(LATEST, env=env, device=args.device)
    if args.algo != "PPO":
        model.load_replay_buffer(BUFFER)
    print(f"RESUME from episode {ep0} ({model.num_timesteps} steps)", flush=True)
else:
    common = dict(seed=args.seed, device=args.device, policy_kwargs=dict(net_arch=[256, 256]), verbose=0)
    if args.algo == "PPO":
        # paper: lr 5e-5..1e-3, batch 16..48, GAE 0.95, clip 0.2
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2 * EP_LEN, batch_size=40, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, **common)
    elif args.algo == "DDPG":
        # paper: OU noise (theta 0.15, sigma 0.2), tau 0.005, buffer 1e4..1e5, batch 256
        model = DDPG("MlpPolicy", env, learning_rate=1e-3, buffer_size=100_000, batch_size=256, tau=0.005,
                     gamma=0.99, learning_starts=EP_LEN, train_freq=1, gradient_steps=1,
                     action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(n_act), 0.2 * np.ones(n_act), theta=0.15),
                     **common)
    else:
        # paper: target noise 0.2, noise clip 0.5, policy delay 2
        model = TD3("MlpPolicy", env, learning_rate=1e-3, buffer_size=100_000, batch_size=256, tau=0.005,
                    gamma=0.99, learning_starts=EP_LEN, train_freq=1, gradient_steps=1,
                    policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5,
                    action_noise=NormalActionNoise(np.zeros(n_act), 0.1 * np.ones(n_act)), **common)

# Episodes run 201 steps (env truncates after iter > 200), so budget slightly above episodes * EP_LEN.
logger = EpisodeLogger(f"{args.out}/episodes.csv", args.ckpt_every, ep0)
if ep0 < args.episodes:
    model.learn(total_timesteps=(args.episodes - ep0) * (EP_LEN + 1), callback=logger,
                reset_num_timesteps=ep0 == 0)
model.save(LATEST)
if args.algo != "PPO":
    model.save_replay_buffer(BUFFER)
with open(STATE, "w") as fh:
    json.dump({"episode": logger.ep, "timesteps": int(model.num_timesteps)}, fh)
if logger.timed_out:
    print(f"TIMEOUT after episode {logger.ep} - resubmit with the same --out to continue", flush=True)
    env.close()
    sys.exit(3)
model.save(f"{args.out}/model_final")

# Deterministic test rollout from the uncontrolled restart state
obs, _ = env.reset()
cd, cl, acts = [], [], []
for _ in range(EP_LEN):
    a, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = env.step(a)
    cd.append(float(last["forces"][:, 0].sum()))
    cl.append(float(last["forces"][:, 1].sum()))
    acts.append(np.asarray(a).tolist())
    if term or trunc:
        break
cd_a, acts_a = np.array(cd), np.array(acts)
print(f"EVAL sumCD last50 mean={cd_a[-50:].mean():.4f} min={cd_a.min():.4f}  sumCL last50 mean={np.mean(cl[-50:]):.4f}",
      flush=True)
print("EVAL mean action last50 [front, up, down] =", np.round(acts_a[-50:].mean(0), 3), flush=True)
with open(f"{args.out}/eval_rollout.json", "w") as fh:
    json.dump({"cd": cd, "cl": cl, "actions": acts}, fh)
env.close()
print("DONE", flush=True)
