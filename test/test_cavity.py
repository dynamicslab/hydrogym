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