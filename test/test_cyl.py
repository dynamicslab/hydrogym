import pytest
import firedrake as fd
import numpy as np

import hydrogym.firedrake as hgym
from hydrogym.firedrake.utils.pd import PDController


def test_import_medium():
  hgym.Cylinder(mesh="medium")


def test_import_fine():
  hgym.Cylinder(mesh="fine")


def test_steady(tol=1e-3):
  flow = hgym.Cylinder(Re=100, mesh="medium")
  solver = hgym.NewtonSolver(flow)
  solver.solve()

  CL, CD = flow.compute_forces()
  assert abs(CL) < tol
  assert abs(CD - 1.2840) < tol  # Re = 100


def test_steady_rotation(tol=1e-3):
  Tf = 4.0
  dt = 0.1

  flow = hgym.RotaryCylinder(Re=100, mesh="medium")
  flow.set_control(0.1)

  solver = hgym.SemiImplicitBDF(flow, dt=dt)

  for iter in range(int(Tf / dt)):
    solver.step(iter)

  # Lift/drag on cylinder
  CL, CD = flow.compute_forces()
  assert abs(CL + 0.06032) < tol
  assert abs(CD - 1.49) < tol  # Re = 100


@pytest.mark.parametrize("k", [1, 3])
def test_integrate(k):
  flow = hgym.Cylinder(mesh="medium")
  dt = 1e-2
  hgym.integrate(flow, t_span=(0, 10 * dt), dt=dt, order=k)


# Simple opposition control on lift
def feedback_ctrl(y, K=0.1):
  CL, CD = y
  return K * CL


def test_control():
  k = 2.0
  theta = 4.0
  tf = 10.0
  dt = 1e-2

  # Initialize the flow
  flow = hgym.RotaryCylinder(Re=100, mesh="medium", velocity_order=1)

  # Construct the PDController
  pd_controller = PDController(
      k * np.cos(theta),
      k * np.sin(theta),
      1e-2,
      int(10.0 // 1e-2) + 2,
      filter_type="bilinear",
      N=2,
  )

  def controller(t, obs):
    if t < tf / 2:
      return 0.0
    return pd_controller(t, obs)

  hgym.integrate(flow, t_span=(0, tf), dt=dt, controller=controller)


def test_env():
  env_config = {
      "flow": hgym.Cylinder,
      "flow_config": {
          "mesh": "medium",
      },
      "solver": hgym.SemiImplicitBDF,
      "solver_config": {
          "dt": 1e-2,
      },
  }
  env = hgym.FlowEnv(env_config)

  u = 0.0
  for _ in range(10):
    y, reward, done, info = env.step(u)
    u = feedback_ctrl(y)

  env.reset()


def test_linearize():
  flow = hgym.Cylinder(mesh="medium")

  solver = hgym.NewtonSolver(flow)
  qB = solver.solve()

  A, M = hgym.modeling.linearize(flow, qB, backend="scipy")
  A_adj, M = hgym.modeling.linearize(flow, qB, adjoint=True, backend="scipy")


def test_act_implicit_no_damp():
  flow = hgym.Cylinder(mesh="medium", actuator_integration="implicit")
  # dt = 1e-2
  solver = hgym.NewtonSolver(flow)

  # Since this feature is still experimental, modify actuator attributes *after*=
  flow.actuators[0].k = 0
  solver.solve()


def test_act_implicit_fixed_torque():
  dt = 1e-4

  # Define the flow
  flow = hgym.Cylinder(mesh="medium", actuator_integration="implicit")

  # Set up the solver
  solver = hgym.SemiImplicitBDF(flow, dt=dt)

  # Obtain a torque value for which the system converges to a steady state angular velocity
  tf = 0.1 * flow.TAU
  omega = 1.0

  # Torque to reach steady-state value of `omega`
  torque = omega / flow.TAU

  # Run sim
  num_steps = 10 * int(tf / dt)
  for iter in range(num_steps):
    flow = solver.step(iter, control=torque)
