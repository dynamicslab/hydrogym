import firedrake as fd
import firedrake_adjoint as fda
import hydrogym as gym
import numpy as np
import pyadjoint
from ufl import sin

def test_import():
    flow = gym.flow.Cavity(mesh='medium')
    return flow

def test_import2():
    flow = gym.flow.Cavity(mesh='fine')
    return flow

def test_steady(tol=1e-3):
    flow = gym.flow.Cavity(Re=500, mesh='medium')
    flow.solve_steady()

    y = flow.collect_observations()
    assert(abs(y - 2.2122) < tol)  # Re = 500


# def test_steady(tol=1e-3):
#     flow = gym.flow.Cylinder(Re=100, mesh='medium')
#     flow.solve_steady()

#     CL, CD = flow.compute_forces()
#     assert(abs(CL) < tol)
#     assert(abs(CD - 1.2840) < tol)  # Re = 100


def test_actuation():
    flow = gym.flow.Cavity(Re=500, mesh='medium')
    flow.set_control(1.0)
    flow.solve_steady()


def test_step():
    flow = gym.flow.Cavity(Re=500, mesh='medium')
    dt = 1e-4

    solver = gym.ts.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        flow = solver.step(iter)

def test_integrate():
    flow = gym.flow.Cavity(Re=500, mesh='medium')
    dt = 1e-4

    gym.integrate(flow,
        t_span=(0, 10*dt),
        dt=dt,
        method='IPCS'
    )

def test_control():
    flow = gym.flow.Cavity(Re=500, mesh='medium')
    dt = 1e-4

    solver = gym.ts.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        y = flow.collect_observations()
        flow = solver.step(iter, control=0.1*sin(solver.t))

def test_env():
    env_config = {'mesh': 'coarse'}
    env = gym.env.CavityEnv(env_config)

    for _ in range(10):
        y, reward, done, info = env.step(0.1*sin(env.solver.t))

# def test_grad():
#     flow = gym.flow.Cylinder(mesh='coarse')

#     omega = fd.Constant(0.0)
#     flow.set_control(omega)

#     flow.solve_steady()
#     CL, CD = flow.compute_forces()

#     dJdu = fda.compute_gradient(CD, fda.Control(omega))

# def test_env_grad():
#     env_config = {'differentiable': True, 'mesh': 'coarse'}
#     env = gym.env.CylEnv(env_config)
#     y = env.reset()
#     K = fd.Constant(0.0)
#     J = fda.AdjFloat(0.0)
#     for _ in range(10):
#         y, reward, done, info = env.step(feedback_ctrl(y, K=K))
#         J = J - reward
#     dJdm = fda.compute_gradient(J, fda.Control(K))
#     print(dJdm)