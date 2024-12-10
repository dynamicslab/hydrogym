import abc
from typing import Any, Callable, Iterable, Tuple, TypeVar, Union

import gym
# import gymnasium as gym
import numpy as np
import torch
from numpy.typing import ArrayLike

class OneDimEnv(gym.Env):

  def __init__(self, env_config: dict):
    self.backend = env_config.get("backend", {"torch"})


    self.solver: PDESolverBase1D = env_config.get("flow")(
        **env_config.get("flow_config", {}))
    
    self.set_seed(env_config.get("seed"))

    self.observation_space = gym.spaces.Box(
      low=-np.inf,
      high=np.inf,
      shape=(self.solver.num_outputs,),
      dtype=float,
    )

    self.action_space = gym.spaces.Box(
        low=self.solver.MAX_CONTROL_LOW,
        high=self.solver.MAX_CONTROL_UP,
        shape=(self.solver.num_inputs,),
        dtype=float,
    )

    self.max_steps = env_config.get("max_steps", 1e6)
    self.dt = env_config.get("flow_config", {}).get("dt", None)
    assert self.dt is not None, f"Error: Solver timestep dt ({self.dt}) must not be None"
  
  def set_seed(self, seed):
    np.random.seed(seed)
    self.solver.set_seed(seed)

  def constant_action_controller(self, t, y):
        return self.action

  def step(
        self, action: Iterable[ArrayLike] = None
    ) -> Tuple[ArrayLike, float, bool, dict]:
        """Advance the state of the environment.  See gym.Env documentation

        Args:
            action (Iterable[ActType], optional): Control inputs. Defaults to None.

        Returns:
            Tuple[ObsType, float, bool, dict]: obs, reward, done, info
        """

        self.solver.step(control=action)
        reward = self.get_reward()
        obs = self.solver.get_observations()
        if reward == -torch.inf:
           done = True
        else:
          done = self.check_complete()
        info = {}

        self.solver.iter += 1

        return obs, reward, done, info
  
  def get_reward(self):    
        return -self.solver.evaluate_objective()

  def check_complete(self):
    return False if self.solver.iter < self.max_steps else True

  def reset(self) -> Union[ArrayLike, Tuple[ArrayLike, dict]]:

    self.solver.reset()
    info = {}

    return self.solver.get_observations(), info

  def render(self, mode="human", **kwargs):
    self.solver.render(mode=mode, **kwargs)


class PDESolverBase1D(metaclass=abc.ABCMeta):
  """
    Basic configuration of the 1D PDE Solver

  """

  MAX_CONTROL = np.inf
  DEFAULT_DT = np.inf

  def __init__(self, **config):


    self.initialize_state()

    self.reset()

    if config.get("restart"):
      self.load_checkpoint(config["restart"][0])

  @property
  @abc.abstractmethod
  def num_inputs(self) -> int:
    """Length of the control vector (number of actuators)"""
    pass

  @property
  @abc.abstractmethod
  def num_outputs(self) -> int:
    """Number of scalar observed variables"""
    pass

  def reset(self):
    """Reset the PDE to an initial state"""
    pass

  @abc.abstractmethod
  def get_observations(self) -> Iterable[ArrayLike]:
    """Return the set of measurements/observations"""
    pass

  @abc.abstractmethod
  def render(self, **kwargs):
    """Plot the current PDE state (called by `gym.Env`)"""
    pass
  
  @abc.abstractmethod
  def evaluate_objective(self) -> ArrayLike:
    """Return the objective function to be minimized"""
    pass

if __name__ == '__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    from tqdm import tqdm
    import torch

    torch.manual_seed(0)
    np.random.seed(0)

    config = {
            "flow": Kuramoto_Sivashinsky,
            "flow_config": {
                            "dt": 0.001,
                            "restart": 'ks_init.tensor',
                            "num_sim_substeps_per_actuation": 250,
                            "device": 'cpu'   
                        }, 
            "max_steps": 100,
        }
    
    # ks_init = torch.load('/net/beegfs-hpc/work/lagemannk/container/hydrogym_dev2/home/firedrake/hydrogym_rllib_multiEnv/hydrogym/hydrogym/torch_env/ks_init.tensor')
    env = OneDimEnv(config)

    action_space = env.action_space 
    observation_space = env.observation_space

    env.reset()

    action = torch.zeros(action_space.shape)

    results = []
    for _ in tqdm(range(2000)):
        obs, reward, done, info = env.step(action)
        results.append(obs.cpu().numpy())
        if done:
           print('envrionment should be resetted')
           env.reset()
    
    results = np.stack(results)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16,8))
    plt.imshow(np.transpose(results), cmap='bwr', vmin=-2.0, vmax=2.0)
    plt.xlabel('step')
    plt.ylabel('observation')
    plt.colorbar()
    plt.show()
    plt.savefig('/net/beegfs-hpc/work/lagemannk/container/workspace_christian_ext/logs/result.png', bbox_inches='tight')
    plt.close()

    env.reset()



    
  