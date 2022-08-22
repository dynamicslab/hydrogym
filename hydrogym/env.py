import firedrake as fd
import gym
import matplotlib.pyplot as plt
import numpy as np
from firedrake import logging


class FlowEnv(gym.Env):
    from typing import Callable, Iterable, Optional, Tuple, TypeVar, Union

    from .core import FlowConfig
    from .ts import TransientSolver

    ObsType = TypeVar("ObsType")
    ActType = TypeVar("ActType")

    def __init__(self, env_config: dict):
        self.flow = env_config.get("flow")
        self.solver = env_config.get("solver")
        self.callbacks = env_config.get("callbacks", [])
        self.max_steps = env_config.get("max_steps", int(1e6))
        self.iter = 0
        self.q0 = self.flow.q.copy(
            deepcopy=True
        )  # Save the initial state for resetting

    def set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def step(
        self, action: Optional[ActType] = None
    ) -> Tuple[ObsType, float, bool, dict]:
        self.solver.step(self.iter, control=action)
        self.iter += 1
        t = self.iter * self.solver.dt
        for cb in self.callbacks:
            cb(self.iter, t, self.flow)
        obs = self.flow.get_observations()

        reward = self.get_reward()
        done = self.check_complete()
        logging.log(
            logging.DEBUG, f"iter: {self.iter}\t reward: {reward}\t done: {done}"
        )
        info = {}

        obs = self.stack_observations(obs)

        # PETSc.Sys.Print(obs)
        return obs, reward, done, info

    # TODO: Use this to allow for arbitrary returns from collect_observations
    #  That are then converted to a list/tuple/ndarray here
    def stack_observations(self, obs):
        return obs

    def get_reward(self):
        return 1 / self.flow.evaluate_objective()

    def check_complete(self):
        return self.iter > self.max_steps

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.flow.q.assign(self.q0)
        self.flow.reset_control()
        self.solver.initialize_operators()

        return self.flow.collect_observations()

    def render(self, mode="human"):
        pass

    def close(self):
        for cb in self.callbacks:
            cb.close()


class CylEnv(FlowEnv):
    def __init__(self, env_config: dict):
        from .flow import Cylinder

        if env_config.get("differentiable"):
            from .ts import IPCS_diff as IPCS
        else:
            from .ts import IPCS

        env_config["flow"] = Cylinder(
            h5_file=env_config.get("checkpoint", None),
            Re=env_config.get("Re", 100),
            mesh=env_config.get("mesh", "medium"),
        )
        env_config["solver"] = IPCS(env_config["flow"], dt=env_config.get("dt", 1e-2))
        super().__init__(env_config)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=fd.utils.ScalarType
        )
        self.action_space = gym.spaces.Box(
            low=-self.flow.MAX_CONTROL,
            high=self.flow.MAX_CONTROL,
            shape=(1,),
            dtype=fd.utils.ScalarType,
        )

    def render(self, mode="human", clim=None, levels=None, cmap="RdBu", **kwargs):
        if clim is None:
            clim = (-2, 2)
        if levels is None:
            levels = np.linspace(*clim, 10)
        vort = fd.project(fd.curl(self.flow.u), self.flow.pressure_space)
        im = fd.tricontourf(
            vort,
            cmap=cmap,
            levels=levels,
            vmin=clim[0],
            vmax=clim[1],
            extend="both",
            **kwargs,
        )

        cyl = plt.Circle((0, 0), 0.5, edgecolor="k", facecolor="gray")
        im.axes.add_artist(cyl)


class PinballEnv(FlowEnv):
    def __init__(self, env_config: dict):
        from .flow import Pinball

        if env_config.get("differentiable"):
            from .ts import IPCS_diff as IPCS
        else:
            from .ts import IPCS
        env_config["flow"] = Pinball(
            h5_file=env_config.get("checkpoint", None),
            Re=env_config.get("Re", 100),
            mesh=env_config.get("mesh", "fine"),
        )
        env_config["solver"] = IPCS(env_config["flow"], dt=env_config.get("dt", 1e-2))
        super().__init__(env_config)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=fd.utils.ScalarType
        )
        self.action_space = gym.spaces.Box(
            low=-self.flow.MAX_CONTROL,
            high=self.flow.MAX_CONTROL,
            shape=(1,),
            dtype=fd.utils.ScalarType,
        )

    def render(self, mode="human", clim=None, levels=None, cmap="RdBu", **kwargs):
        if clim is None:
            clim = (-2, 2)
        if levels is None:
            levels = np.linspace(*clim, 10)
        vort = fd.project(fd.curl(self.flow.u), self.flow.pressure_space)
        im = fd.tricontourf(
            vort,
            cmap=cmap,
            levels=levels,
            vmin=clim[0],
            vmax=clim[1],
            extend="both",
            **kwargs,
        )

        for (x0, y0) in zip(self.flow.x0, self.flow.y0):
            cyl = plt.Circle((x0, y0), self.flow.rad, edgecolor="k", facecolor="gray")
            im.axes.add_artist(cyl)


class CavityEnv(FlowEnv):
    def __init__(self, env_config: dict):
        from .flow import Cavity

        if env_config.get("differentiable"):
            from .ts import IPCS_diff as IPCS
        else:
            from .ts import IPCS

        env_config["flow"] = Cavity(
            h5_file=env_config.get("checkpoint", None),
            Re=env_config.get("Re", 7500),
            mesh=env_config.get("mesh", "fine"),
        )
        env_config["solver"] = IPCS(env_config["flow"], dt=env_config.get("dt", 1e-4))
        super().__init__(env_config)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=fd.utils.ScalarType
        )
        self.action_space = gym.spaces.Box(
            low=-self.flow.MAX_CONTROL,
            high=self.flow.MAX_CONTROL,
            shape=(1,),
            dtype=fd.utils.ScalarType,
        )
