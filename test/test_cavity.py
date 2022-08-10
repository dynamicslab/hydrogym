import firedrake as fd
import firedrake_adjoint as fda
import hydrogym as gym
import numpy as np
import pyadjoint

def test_import():
    flow = gym.flow.Cavity(mesh='medium')
    return flow

def test_import2():
    flow = gym.flow.Cavity(mesh='fine')
    return flow

def test_steady():
    flow = gym.flow.Cavity(Re=500, mesh='medium')
    flow.solve_steady()


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