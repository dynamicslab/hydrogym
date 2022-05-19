from time import time
import gym
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt

class FlowEnv(gym.Env):
    from .core import FlowConfig
    from .ts import TransientSolver
    from typing import Optional, Tuple, TypeVar, Union, Iterable, Callable
    ObsType = TypeVar("ObsType")
    ActType = TypeVar("ActType")

    def __init__(self, flow: FlowConfig, solver: TransientSolver, callbacks: Optional[Iterable[Callable]] = []):
        self.flow = flow
        self.solver = solver
        self.callbacks = callbacks
        self.iter = 0
        self.q0 = self.flow.q.copy(deepcopy=True)  # Save the initial state for resetting

    def set_callbacks(self, callbacks):
        self.callbacks = callbacks
    
    def step(self, action: Optional[ActType] = None) -> Tuple[ObsType, float, bool, dict]:
        self.solver.step(self.iter, control=action)
        self.iter += 1
        t = self.iter*self.solver.dt
        for cb in self.callbacks:
            cb(self.iter, t, self.flow)
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

    def close(self):
        for cb in self.callbacks:
            cb.close()

class CylEnv(FlowEnv):
    def __init__(self, checkpoint=None, callbacks=[], differentiable=False, dt=1e-2, Re=100, mesh='medium'):
        from .flow import Cylinder
        if differentiable:
            from .ts import IPCS_diff as IPCS
        else:
            from .ts import IPCS
        flow = Cylinder(h5_file=checkpoint, Re=Re, mesh=mesh)
        solver = IPCS(flow, dt=dt)
        super().__init__(flow, solver, callbacks)

    def get_reward(self, obs):
        CL, CD = obs
        return -CD

    def render(self, mode="human", clim=None, levels=None, cmap='RdBu', **kwargs):
        if clim is None: clim = (-2, 2)
        if levels is None: levels=np.linspace(*clim, 10)
        vort = fd.project(fd.curl(self.flow.u), self.flow.pressure_space)
        im = fd.tricontourf(vort, cmap=cmap, levels=levels, vmin=clim[0], vmax=clim[1], extend='both', **kwargs)

        cyl = plt.Circle((0, 0), 0.5, edgecolor='k', facecolor='gray')
        im.axes.add_artist(cyl)

class PinballEnv(FlowEnv):
    def __init__(self, checkpoint=None, callbacks=[], differentiable=False, dt=1e-2, Re=100, mesh='fine'):
        from .flow import Pinball
        if differentiable:
            from .ts import IPCS_diff as IPCS
        else:
            from .ts import IPCS
        flow = Pinball(h5_file=checkpoint, Re=Re, mesh=mesh)
        solver = IPCS(flow, dt=dt)
        super().__init__(flow, solver, callbacks)

    def get_reward(self, obs):
        CL, CD = obs
        return -sum(CD)

    def render(self, mode="human", clim=None, levels=None, cmap='RdBu', **kwargs):
        if clim is None: clim = (-2, 2)
        if levels is None: levels=np.linspace(*clim, 10)
        vort = fd.project(fd.curl(self.flow.u), self.flow.pressure_space)
        im = fd.tricontourf(vort, cmap=cmap, levels=levels, vmin=clim[0], vmax=clim[1], extend='both', **kwargs)

        for (x0, y0) in zip(self.flow.x0, self.flow.y0):
            cyl = plt.Circle((x0, y0), self.flow.rad, edgecolor='k', facecolor='gray')
            im.axes.add_artist(cyl)