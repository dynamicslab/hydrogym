from ufl import sin
import pytest

import hydrogym.firedrake as hgym


def test_import_medium():
  hgym.Cavity(Re=500, mesh="medium")


def test_import_fine():
  hgym.Cavity(mesh="fine")


def test_steady():
  flow = hgym.Cavity(Re=50, mesh="medium")
  solver = hgym.NewtonSolver(flow)
  solver.solve()


def test_steady_actuation():
  flow = hgym.Cavity(Re=50, mesh="medium")
  flow.set_control(1.0)
  solver = hgym.NewtonSolver(flow)
  solver.solve()


def test_integrate():
  flow = hgym.Cavity(Re=50, mesh="medium")
  dt = 1e-4

  hgym.integrate(
      flow,
      t_span=(0, 2 * dt),
      dt=dt,
      # stabilization="gls"
  )


def test_control():
  flow = hgym.Cavity(Re=50, mesh="medium")
  dt = 1e-4

  solver = hgym.SemiImplicitBDF(flow, dt=dt)

  num_steps = 10
  for iter in range(num_steps):
    flow.get_observations()
    flow = solver.step(iter, control=0.1 * sin(iter * solver.dt))


def test_env():
  env_config = {
      "flow": hgym.Cavity,
      "flow_config": {
          "mesh": "medium",
          "Re": 10
      },
      "solver": hgym.SemiImplicitBDF,
  }
  env = hgym.FlowEnv(env_config)

  for i in range(10):
    y, reward, done, info = env.step(0.1 * sin(i * env.solver.dt))
