import os
import math
import torch
import numpy as np

from hydrogym.core_1DEnvs import PDESolverBase1D

class Kuramoto_Sivashinsky(PDESolverBase1D):
    MAX_CONTROL_LOW = -0.5
    MAX_CONTROL_UP = 0.5
    DEFAULT_DT = 0.001
  
    def __init__(self, **env_config):
        TENSOR_DIR = os.path.abspath(f"{__file__}/..")

        device = env_config.get("device", 'cpu')
        self.iter = 0
        l = 8*math.pi

        self.meshx = 64
        dx = l/self.meshx
        self.maxt = 400
        self.dx = l/self.meshx
        self.dx2 = self.dx**2
        self.dx4 = self.dx**4
        self.dt = env_config.get("dt", None)

        x = torch.linspace(0, l-dx, self.meshx).to(device)
        self.x = x
        self.actuator_position = [0.00, 0.25, 0.5, 0.75]
        self.f0 = (torch.exp(-(x - self.actuator_position[0]*l)**2/2)/math.sqrt(2*math.pi)).to(device)
        self.f1 = (torch.exp(-(x - self.actuator_position[1]*l)**2/2) /
                   math.sqrt(2*math.pi)).to(device)
        self.f2 = (torch.exp(-(x - self.actuator_position[2]*l)**2/2) /
                   math.sqrt(2*math.pi)).to(device)
        self.f3 = (torch.exp(-(x - self.actuator_position[3]*l)**2/2) /
                   math.sqrt(2*math.pi)).to(device)
        
        init_file = env_config.get("restart", 'ks_init.tensor')
        self.init = torch.load(os.path.join(TENSOR_DIR, init_file), map_location=device)

        # self.num_steps = 250
        self.num_sim_substeps_per_actuation = env_config.get("num_sim_substeps_per_actuation", None)

        self.lapa_c = FD_Central_CoefficientMatrix(
            [1/90, -3/20, 3/2, -49/18], self.meshx, periodic=True)
        self.lapa2_c = FD_Central_CoefficientMatrix(
            [7/240, -2/5, 169/60, -122/15, 91/8], self.meshx, periodic=True)
        self.gradient_fc, self.gradient_bc = FD_upwind_CoefficientMatrix(
            [1/4, -4/3, 3, -4, 25/12], self.meshx, periodic=True)
        self.numpy = dict(f=torch.cat(
            (self.f0.unsqueeze(0), self.f1.unsqueeze(0),
             self.f2.unsqueeze(0), self.f3.unsqueeze(0)), dim=0).cpu().numpy(),
            init=self.init.cpu().numpy(),
            lapa_c=self.lapa_c.cpu().numpy(),
            lapa2_c=self.lapa2_c.cpu().numpy(),
            gradient_fc=self.gradient_fc.cpu().numpy(),
            gradient_bc=self.gradient_bc.cpu().numpy())
    
    def set_seed(self, seed):
        torch.manual_seed(seed)

    def step(self, control):
        f = np.matmul(control, self.numpy['f'])
        f = np.expand_dims(f, axis=-1)
        self.current_reward = torch.tensor(0.0)
        
        self.state, self.current_reward = solve_next_state(self.state, 
                                            control, 
                                            self.lapa_c, self.lapa2_c,
                                            self.gradient_fc, self.gradient_bc,
                                            self.dt, 
                                            self.dx, self.dx2, self.dx4, 
                                            self.f0, self.f1, self.f2, self.f3, 
                                            self.current_reward, 
                                            self.num_sim_substeps_per_actuation)

    def evaluate_objective(self):
       return self.current_reward/self.num_sim_substeps_per_actuation
    
    def get_observations(self):
       return self.state

    def step_p(self, state, act):
        state, rew = self.__calculate__(state, act)
        return state, rew

    def reset(self, shape=()):
        self.state = self.init[torch.randint(0, 200, size=shape)]
    
    @property
    def num_inputs(self) -> int:
        return len(self.actuator_position)

    @property
    def num_outputs(self) -> int:
        return int(*self.x.shape)

    def test_reset(self, num_test=5):
        # return self.init[torch.multinomial(torch.ones(200), num_samples=num_test,
        #     replacement=False)].unsqueeze(2).cpu().numpy()
        return self.init[torch.linspace(0,199,num_test,dtype=torch.long)].unsqueeze(2).cpu().numpy()

    def render(self, mode="human", clim=None, levels=None, cmap="RdBu", **kwargs):
        pass

@torch.jit.script
def RHS(u, lapa_c, lapa2_c, gradient_fc, gradient_bc, dx: float, dx2: float, dx4: float, f):
    u2 = u*u
    lapa = torch.matmul(lapa_c, u)
    lapa2 = torch.matmul(lapa2_c, u)
    gradient = torch.matmul(gradient_fc, u2)*(u < 0)\
        + torch.matmul(gradient_bc, u2)*(u >= 0)
    return -lapa2/dx4 - lapa/dx2 - gradient/2./dx + f, lapa, gradient


@torch.jit.script
def solve_next_state(state, action, lapa_c, lapa2_c, gradient_fc, gradient_bc,
                  dt: float, dx: float, dx2: float, dx4: float, f0, f1, f2, f3, r, num_steps: int):

    f = action[0]*f0 + action[1]*f1 + action[2]*f2 + action[3]*f3
    for _ in range(num_steps):
        k1, lapa, gradient = RHS(
            state, lapa_c, lapa2_c, gradient_fc, gradient_bc, dx, dx2, dx4, f)
        k2, _, _ = RHS(state + dt*k1/2, lapa_c, lapa2_c,
                       gradient_fc, gradient_bc, dx, dx2, dx4, f)
        k3, _, _ = RHS(state + dt*k2/2, lapa_c, lapa2_c,
                       gradient_fc, gradient_bc, dx, dx2, dx4, f)
        k4, _, _ = RHS(state + dt*k3, lapa_c, lapa2_c,
                       gradient_fc, gradient_bc, dx, dx2, dx4, f)
        r += (lapa*lapa).mean() + (gradient*gradient).mean() + (state*f).mean()
        state = state + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    return state, r


def FD_Central_CoefficientMatrix(c:list,meshx:int,periodic:bool=False):
    '''
    c is list of FD coefficient
    e.g. for 1st derivative with 2nd accuracy central difference:
    c=[-0.5,0] 
    '''
    if 2*len(c)-1>=meshx: raise ValueError
    acc = len(c)   
    
    tmp=[]
    c.reverse()
    for i in range(acc):
        x = torch.cat((torch.cat((torch.zeros((i,meshx-i)),
                                    c[i]*torch.eye(meshx-i)),dim=0),
                                    torch.zeros((meshx,i))
                                    ),dim=1)
        tmp.append(x)
    re=tmp[0]
    for k in tmp[1:]:
        re+=k+k.T

    if periodic:
        re[:acc,-acc:]=re[acc:2*acc,:acc]
        re[-acc:,:acc]=re[:acc,acc:2*acc]
    return re

def FD_upwind_CoefficientMatrix(c:list,meshx:int,periodic:bool=False):
    '''
    c is list of Backward FD coefficient
    e.g. for 1st derivative with 1st accuracy:
    c=[-1,1] 
    '''
    if len(c)>=meshx: raise ValueError
    acc = len(c)   
    
    tmp=[]
    
    c.reverse()
    for i in range(acc):
        x = torch.cat((torch.cat((torch.zeros((i,meshx-i)),
                                    c[i]*torch.eye(meshx-i)),dim=0),
                                    torch.zeros((meshx,i))
                                    ),dim=1)
        tmp.append(x)

    bre=tmp[0]
    fre=-tmp[0]
    
    for k in tmp[1:]:
        fre+=-k.T
        bre+=k

    if periodic:
        fre[-acc:,:acc]=fre[:acc,acc:2*acc]
        bre[:acc,-acc:]=bre[acc:2*acc,:acc]
    return fre,bre