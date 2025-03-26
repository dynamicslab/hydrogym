import os

os.environ["OMP_NUM_THREADS"] = "1"

from ray.rllib.algorithms.ppo import PPOConfig
from pprint import pprint

from firedrake import logging
import numpy as np
from tap import Tap
from typing import Iterable, Tuple, Union
from numpy.typing import ArrayLike
import gymnasium as gym

import hydrogym
from hydrogym.core import PDEBase, TransientSolver, CallbackBase

logging.set_log_level(logging.DEBUG)


class ArgumentParser(Tap):
    reynolds_number: int = 100  # Reynolds number
    mesh_resolution: str = "medium"  # Mesh resolution
    time_step: float = 1e-2  # Time step
    learning_rate: float = 0.0002  # Learning rate
    batch_size_per_learner: int = 2000  # Batch size per learner
    number_of_epochs: int = 10  # Number of epochs


# Read in the command-line arguments
args = ArgumentParser().parse_args()


class FlowEnv(gym.Env):
    def __init__(self, env_config: dict):
        self.flow: PDEBase = env_config.get("flow")(**env_config.get("flow_config", {}))
        self.solver: TransientSolver = env_config.get("solver")(self.flow, **env_config.get("solver_config", {}))
        self.callbacks: Iterable[CallbackBase] = env_config.get("callbacks", [])
        self.max_steps: int = env_config.get("max_steps", int(1e6))
        self.iter: int = 0
        self.q0: self.flow.StateType = self.flow.copy_state()

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.flow.num_outputs,),
            dtype=float,
        )
        self.action_space = gym.spaces.Box(
            low=-self.flow.MAX_CONTROL,
            high=self.flow.MAX_CONTROL,
            shape=(self.flow.num_inputs,),
            dtype=float,
        )

    def set_callbacks(self, callbacks: Iterable[CallbackBase]):
        self.callbacks = callbacks

    def step(self, action: Iterable[ArrayLike] = None) -> Tuple[ArrayLike, float, bool, dict]:
        """Advance the state of the environment.  See gymnasium.Env documentation

        Args:
            action (Iterable[ArrayLike], optional): Control inputs. Defaults to None.

        Returns:
            Tuple[ArrayLike, float, bool, dict]: obs, reward, done, info
        """
        self.solver.step(self.iter, control=action)
        self.iter += 1
        t = self.iter * self.solver.dt
        for cb in self.callbacks:
            cb(self.iter, t, self.flow)
        obs = self.flow.get_observations()

        reward = self.get_reward()
        done = self.check_complete()
        info = {}

        # Hacking around it for now
        truncated = False

        obs = self.stack_observations(obs)

        return obs, reward, done, truncated, info

    def stack_observations(self, obs):
        return obs

    def get_reward(self):
        return -self.solver.dt * self.flow.evaluate_objective()

    def check_complete(self):
        return self.iter > self.max_steps

    def reset(self, seed=None, t=0.0, options=None) -> Union[ArrayLike, Tuple[ArrayLike, dict]]:
        super().reset(seed=seed)
        self.iter = 0
        self.flow.reset(q0=self.q0, t=t)
        self.solver.reset()

        info = {}

        return self.flow.get_observations(), info

    def render(self, mode="human", **kwargs):
        self.flow.render(mode=mode, **kwargs)

    def close(self):
        for cb in self.callbacks:
            cb.close()


# Define the logging callback
log = hydrogym.firedrake.utils.io.LogCallback(
    postprocess=lambda flow: flow.get_observations(),
    nvals=2,
    interval=1,
    print_fmt="t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}",
    filename=None,
)

# Create a config instance for the PPO algorithm.
config = (
    PPOConfig()
    .environment(
        FlowEnv,
        env_config={
            "flow": hydrogym.firedrake.Cylinder,
            "flow_config": {
                "Re": args.reynolds_number,
                "mesh": args.mesh_resolution,
            },
            "solver": hydrogym.firedrake.SemiImplicitBDF,
            "solver_config": {
                "dt": args.time_step,
            },
            # "callbacks": [log],
            "max_steps": 10000,  # --> This should be going into the `training`
        },
    )
    .env_runners(num_env_runners=2)
    .training(
        lr=args.learning_rate,
        train_batch_size_per_learner=args.batch_size_per_learner,
        num_epochs=args.number_of_epochs,
    )
)

# Construct the actual algorithm
ppo = config.build_algo()

for _ in range(4):
    pprint(ppo.train())

# Store the trained algorithm
checkpoint_path = ppo.save_to_path("/Users/lpaehler/Work/ReinforcementLearning/hydrogym-issue-fix/ppo_checkpoint")
