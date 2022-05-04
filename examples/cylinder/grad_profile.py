import numpy as np
import firedrake as fd
import firedrake_adjoint as fda
from firedrake.petsc import PETSc
import hydrogym as gym

from memory_profiler import memory_usage

def print(s):
    PETSc.Sys.print(s)

# Simple opposition control on lift
def g(y):
    CL, CD = y
    return 0.1*CL

def episode(num_steps):
    env = gym.env.CylEnv()
    u = fd.Constant(0.0)
    control = fda.Control(u)
    for _ in range(num_steps):
        y, reward, done, info = env.step(u)
        u.assign(g(y))

    CL, CD = y
    dJdu = fda.compute_gradient(CD, control)

num_steps = np.unique(np.floor(np.logspace(0, 4)))
data = np.zeros((len(num_steps), 2))
for (i, N) in enumerate(num_steps):
    mem_usage = max(memory_usage((episode, [N])))
    print(f'{N} steps, {mem_usage} Mb')
    data[i, :] = [N, mem_usage]
    np.savetxt('memory.dat', data)