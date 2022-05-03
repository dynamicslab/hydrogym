import gym
import hydrogym

from typing import Optional, Tuple, TypeVar, Union
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class FlowEnv(gym.Env):
    def __init__(self, flow: hydrogym.Flow, solver: hydrogym.TransientSolver):
        self.flow = flow
        self.solver = solver
        self.iter = 0
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self.flow.update_control(action)
        self.solver.step(self.iter)
        self.iter += 1
        obs = self.flow.observation()

        reward = 0
        done = False
        info = {}
        return obs, reward, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        pass

    def render(self, mode="human"):
        pass