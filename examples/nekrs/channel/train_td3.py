"""
TD3 on the NekRS channel, one agent per actuator with a shared policy, following
examples/nek/getting_started/3_pettingzoo/train_sb3_pettingzoo.py and the runner block of
pettingzoo_tcflarge_re207.yml (train every 300 actions, 64 gradient steps, lr 1e-3, tau 0.005, policy delay 2,
target noise 0.2 / clip 0.5, action noise 0.1, nets qf [16, 16, 64], pi [8]).
Episodes start from the snapshot library of the YAML (initial_conditions). Launch:
    GPU=0 NP=1 RUN=runs/td3 SCRIPT=train_td3.py ./run.sh --episodes 10
    GPU=0,1 NSIM=2 RUN=runs/td3_x2 SCRIPT=train_td3.py ./run.sh --episodes 10     (2 simulations, one policy)

With --sims M (NSIM in run.sh) M simulations of the case run concurrently and feed the same replay buffer; each
draws its own initial conditions. --episodes counts the episodes of each simulation. The replay buffer size and
the gradient steps per update are totals: keep buffer_size / M and gradient_steps the same to have the same
history per simulation and M times fewer updates per transition, or scale them.

Resume: at the end of every episode the model, the replay buffer and train_state.json are saved to
RUN/checkpoints/latest; a launch with the same RUN continues from there (from fresh episodes). --max-hours stops
after the last episode that fits into the time budget, for batch jobs with a time limit (train_slurm.sbatch).
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise

from hydrogym.nekrs.multisim import MultiSimVecEnv, make_envs

T_START = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("--conf", default="channel_td3.yml")
ap.add_argument("--nproc", type=int, default=1, help="NekRS ranks per simulation")
ap.add_argument("--sims", type=int, default=1, help="concurrent simulations feeding the policy")
ap.add_argument("--run-root", default="runs/td3")
ap.add_argument("--episodes", type=int, default=10, help="episodes per simulation")
ap.add_argument("--seed", type=int, default=20010603)
ap.add_argument("--train-freq", type=int, default=300)
ap.add_argument("--gradient-steps", type=int, default=64)
ap.add_argument("--batch-size", type=int, default=2048)
ap.add_argument("--buffer-size", type=int, default=5_000_000)
ap.add_argument("--learning-rate", type=float, default=1e-3)
ap.add_argument("--learning-starts", type=int, default=100)
ap.add_argument("--save-every", type=int, default=1, help="keep the policy of every n-th episode (0 = only the last)")
ap.add_argument("--eval-actions", type=int, default=300, help="deterministic episode from snapshot 0 (0 = skip)")
ap.add_argument("--max-hours", type=float, default=0, help="wall time budget of this launch (0 = none)")
args = ap.parse_args()

CKPT = os.path.join(args.run_root, "checkpoints")
LATEST = os.path.join(CKPT, "latest")
state = None
if os.path.exists(os.path.join(LATEST, "train_state.json")):
    with open(os.path.join(LATEST, "train_state.json")) as f:
        state = json.load(f)
launch = state["launches"] if state else 0

# env logs of every launch apart (the env files are rewritten at start)
envs = make_envs(args.conf, args.sims, args.nproc, os.path.join(args.run_root, f"launch{launch:02d}"))
vec = MultiSimVecEnv(envs)
episode_actions = envs[0].episode_actions
target_actions = args.episodes * episode_actions
print(
    f"[TD3] {args.sims} simulation(s) x {vec.sizes[0]} agents = {vec.num_envs} sub-envs, "
    f"obs {vec.observation_space.shape}"
    + (
        f", resuming launch {launch} at {state['actions_per_sim']} of {target_actions} actions per simulation"
        if state
        else ""
    ),
    flush=True,
)


def save_latest(model, actions_per_sim, episodes, done=False):
    """Model, replay buffer and counters, written to a new directory and then swapped in (a job may be killed)."""
    tmp = LATEST + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    model.save(os.path.join(tmp, "model"))
    model.save_replay_buffer(os.path.join(tmp, "replay_buffer"))
    st = {
        "launches": launch + 1,
        "actions_per_sim": int(actions_per_sim),
        "episodes": episodes,
        "num_timesteps": int(model.num_timesteps),
        "sims": args.sims,
        "conf": args.conf,
        "target_actions": target_actions,
        "done": done,
    }
    with open(os.path.join(tmp, "train_state.json"), "w") as f:
        json.dump(st, f, indent=1)
    shutil.rmtree(LATEST, ignore_errors=True)
    os.replace(tmp, LATEST)


class EpisodeLog(BaseCallback):
    """Logs every finished episode of every simulation, checkpoints, and stops before the time budget runs out."""

    def __init__(self, path, episodes):
        super().__init__()
        self.f = open(path, "a")
        self.episodes = episodes
        self.t_ep = time.time()
        self.stopped = False

    def _on_step(self):
        if not vec.finished:
            return True
        for ep in vec.finished:
            r, d = np.asarray(ep["rewards"]), np.asarray(ep["dUdy"])
            n = len(r)
            rec = {
                "episode": self.episodes,
                "sim": ep["sim"],
                "sim_episode": ep["episode"],
                "index": ep["index"],
                "actions": n,
                "terminated": ep["terminated"],
                "mean_reward": float(r.mean()),
                "last_third_reward": float(r[2 * n // 3 :].mean()),
                "last_third_dUdy": float(d[2 * n // 3 :].mean()),
                "timesteps": int(self.num_timesteps),
                "wall_s": round(time.time() - T_START, 1),
                "launch": launch,
            }
            print("[TD3] " + json.dumps(rec), flush=True)
            self.f.write(json.dumps(rec) + "\n")
            self.f.flush()
            self.episodes += 1
            if args.save_every and self.episodes % (args.save_every * args.sims) == 0:
                self.model.save(os.path.join(CKPT, f"td3_ep{self.episodes - 1:03d}"))
        vec.finished.clear()
        save_latest(self.model, self.num_timesteps // vec.num_envs, self.episodes)
        now = time.time()
        dt_ep, self.t_ep = now - self.t_ep, now
        if args.max_hours and now - T_START + 1.15 * dt_ep > args.max_hours * 3600:
            print(
                f"[TD3] time budget of {args.max_hours} h: stopping after this episode "
                f"({(now - T_START) / 3600:.2f} h, last episode {dt_ep / 60:.1f} min)",
                flush=True,
            )
            self.stopped = True
            return False
        return True


try:
    if state:
        for k in ("sims", "conf"):
            if state[k] != getattr(args, k):
                raise SystemExit(f"{args.run_root} was started with --{k} {state[k]}, not {getattr(args, k)}")
    if state and state["actions_per_sim"] >= target_actions:
        print(
            f"[TD3] {args.run_root}: {state['actions_per_sim']} of {target_actions} actions per simulation done",
            flush=True,
        )
        model = TD3.load(os.path.join(LATEST, "model"), device="cuda")
    else:
        noise = VectorizedActionNoise(NormalActionNoise(np.zeros(1), 0.1 * np.ones(1)), vec.num_envs)
        if state:
            model = TD3.load(os.path.join(LATEST, "model"), env=vec, device="cuda")
            model.load_replay_buffer(os.path.join(LATEST, "replay_buffer"))
            model._last_obs = None  # saved with the model: learn() must reset the simulations (new episodes)
            model.action_noise = noise
            # new random streams in every launch (action noise, initial condition draws)
            model.set_random_seed(args.seed + 7919 * launch)
        else:
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
                action_noise=noise,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                policy_kwargs={"net_arch": {"qf": [16, 16, 64], "pi": [8]}},
                seed=args.seed,
                verbose=0,
                device="cuda",
            )
        done_actions = state["actions_per_sim"] if state else 0
        log = EpisodeLog(os.path.join(args.run_root, "td3_episodes.jsonl"), state["episodes"] if state else 0)
        model.learn(
            total_timesteps=(target_actions - done_actions) * vec.num_envs, callback=log, reset_num_timesteps=not state
        )
        if log.stopped:
            raise SystemExit(0)
        save_latest(model, model.num_timesteps // vec.num_envs, log.episodes, done=True)
        model.save(os.path.join(args.run_root, "td3_final"))

    def evaluate(policy, name):
        """One episode of simulation 0 from snapshot 0: the trained policy (deterministic) or zero actuation."""
        env = envs[0]
        obs, _ = env.reset(options={"index": 0})
        r, d = [], []
        for _ in range(args.eval_actions):
            act = policy(obs.reshape(env.n_actuators, -1))
            obs, rew, term, trunc, info = env.step(np.asarray(act).reshape(-1))
            r.append(rew)
            d.append(info["dUdy"])
            if term:
                break
        n = len(r)
        return {
            "policy": name,
            "index": 0,
            "actions": n,
            "mean_reward": float(np.mean(r)),
            "last_third_reward": float(np.mean(r[2 * n // 3 :])),
            "last_third_dUdy": float(np.mean(d[2 * n // 3 :])),
        }

    if args.eval_actions and not os.path.exists(os.path.join(args.run_root, "td3_eval.json")):
        res = [
            evaluate(lambda o: model.predict(o, deterministic=True)[0], "td3"),
            evaluate(lambda o: np.zeros((len(o), 1), dtype=np.float32), "zero"),
        ]
        for rec in res:
            print("[TD3-EVAL] " + json.dumps(rec), flush=True)
        json.dump(res, open(os.path.join(args.run_root, "td3_eval.json"), "w"), indent=1)
finally:
    vec.close()
