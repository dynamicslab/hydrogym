import firedrake as fd
from firedrake.petsc import PETSc
from slepc4py import SLEPc

import numpy as np

import hydrogym as gym
temp_dir = 'tmp'
output_dir = 'global-modes'
mesh = 'medium'

Re = 100
flow = gym.flow.Cylinder(Re=Re, mesh=mesh)