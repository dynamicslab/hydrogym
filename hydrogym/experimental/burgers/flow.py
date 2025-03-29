import os
import math
import torch
import numpy as np

from hydrogym.core_1DEnvs import PDESolverBase1D

class Burgers(PDESolverBase1D):
    MAX_CONTROL_LOW = -0.025
    MAX_CONTROL_UP = 0.075
    DEFAULT_DT = 0.001
  
    def __init__(self, **env_config):

        device = env_config.get("device", 'cpu')
        l = 2
        self.iter = 0

        self.meshx = 150
        dx = l/self.meshx
        self.maxt = 60
        self.dx = 1/self.meshx
        self.dt = env_config.get("dt", None)
        self.nu = 0.01

        x = torch.linspace(0, l-dx, self.meshx)
        self.x = x
        self.actuator_position = [0.25, 0.75]
        self.f1 = torch.exp(-225*(x/l-self.actuator_position[0])*(x/l-self.actuator_position[0]))
        self.f2 = torch.exp(-225*(x/l-self.actuator_position[1])*(x/l-self.actuator_position[1]))
        self.init1 = 0.2*torch.exp(-25*(x/l-0.5)*(x/l-0.5))
        self.init2 = 0.2*torch.sin(4*math.pi*x/l)

        # self.num_steps = 500
        self.num_sim_substeps_per_actuation = env_config.get("num_sim_substeps_per_actuation", None)

        ref = torch.arange(0, 30.5, self.dt*self.num_sim_substeps_per_actuation)
        self.ref = (0.05*torch.sin(np.pi/15*ref) +
                    0.5).reshape(-1, 1).repeat([1, self.meshx])

        self.act_limit = torch.tensor([[-.025, .075], [-.025, .075]])
    
    def set_seed(self, seed):
        torch.manual_seed(seed)

    def step(self, control):
        self.pdestate = torch.cat(
            (self.pdestate[-2:], self.pdestate, self.pdestate[:2]))
        lapa = -1/12*self.pdestate[:-4]+4/3*self.pdestate[1:-3]-5/2 * \
                self.pdestate[2:-2]+4/3 * \
                self.pdestate[3:-1]-1/12*self.pdestate[4:]
        state2 = self.pdestate**2/2
        gradient = 0.5*state2[:-4]-2*state2[1:-3]+1.5*state2[2:-2]

        self.pdestate = self.pdestate[2:-2] + self.dt * (
                        self.nu * lapa / self.dx**2 - gradient / self.dx
                        + control[0]*self.f1 + control[1]*self.f2)

        self.state = self.pdestate-self.ref[self.iter]

    def evaluate_objective(self):
        if torch.any(self.state.isnan()) == True:
            print('Nan detected - resetting episode!', flush=True)
            return -torch.inf
        else:
            return 10*((self.state**2).mean())
    
    def get_observations(self):
       return self.state

    def step_p(self, state, act):
        state, rew = self.__calculate__(state, act)
        return state, rew

    def reset(self, shape=()):
        a = torch.rand(1)
        self.pdestate = a*self.init1 + (1-a)*self.init2 + 0.2
        self.state = self.pdestate - self.ref[0]
        self.iter = 0
    
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