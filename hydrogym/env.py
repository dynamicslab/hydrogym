from time import time
import gym
import firedrake as fd
import numpy as np

class FlowEnv(gym.Env):
    from .core import FlowConfig
    from .ts import TransientSolver
    from typing import Optional, Tuple, TypeVar, Union
    ObsType = TypeVar("ObsType")
    ActType = TypeVar("ActType")

    def __init__(self, flow: FlowConfig, solver: TransientSolver):
        self.flow = flow
        self.solver = solver
        self.iter = 0
        self.q0 = self.flow.q.copy(deepcopy=True)  # Save the initial state for resetting

    def set_callbacks(self, callbacks):
        self.solver.callbacks = callbacks
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self.flow.set_control(action)
        self.solver.step(self.iter)
        self.iter += 1
        obs = self.flow.collect_observations()

        reward = self.get_reward(obs)
        done = self.check_complete()
        info = {}
        return obs, reward, done, info

    def get_reward(self, obs):
        return False

    def check_complete(self):
        pass

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.flow.q.assign(self.q0)
        self.flow.reset_control()
        self.solver.initialize_operators()

        return self.flow.collect_observations()

    def render(self, mode="human"):
        pass

class CylEnv(FlowEnv):
    def __init__(self, checkpoint=None, callbacks=[]):
        from .flow import Cylinder
        from .ts import IPCSSolver
        dt = 1e-2
        flow = Cylinder(h5_file=checkpoint)
        solver = IPCSSolver(flow, dt=dt, callbacks=callbacks, time_varying_bc=True)
        super().__init__(flow, solver)

    def get_reward(self, obs):
        CL, CD = obs
        return -CD

    def render(self, mode="human", clim=None, levels=None, cmap='RdBu'):
        if clim is None: clim = (-5, 5)
        if levels is None: levels=np.linspace(*clim, 10)
        vort = fd.project(fd.curl(self.flow.u), self.flow.pressure_space)
        fd.tricontourf(vort, cmap=cmap, levels=levels, vmin=clim[0], vmax=clim[1], extend='both')