import firedrake_adjoint as fda
from ufl import sin

import hydrogym.firedrake as hgym


def test_import_coarse():
  hgym.Step(mesh="coarse")


def test_import_medium():
  hgym.Step(mesh="medium")


def test_import_fine():
  hgym.Step(mesh="fine")


def test_steady():
  flow = hgym.Step(Re=100, mesh="coarse")

  solver = hgym.NewtonSolver(flow)
  solver.solve()


def test_steady_actuation():
  flow = hgym.Step(Re=100, mesh="coarse")
  flow.set_control(1.0)

  solver = hgym.NewtonSolver(flow)
  solver.solve()


def test_integrate():
  flow = hgym.Step(Re=100, mesh="coarse")
  dt = 1e-3

  hgym.integrate(flow, t_span=(0, 10 * dt), dt=dt, method="IPCS")


def test_integrate_noise():
  flow = hgym.Step(Re=100, mesh="coarse")
  dt = 1e-3

  hgym.integrate(flow, t_span=(0, 10 * dt), dt=dt, method="IPCS", eta=1.0)


def test_control():
  flow = hgym.Step(Re=100, mesh="coarse")
  dt = 1e-3

  solver = hgym.IPCS(flow, dt=dt)

  num_steps = 10
  for iter in range(num_steps):
    flow.get_observations()
    flow = solver.step(iter, control=0.1 * sin(solver.t))


def test_env():
  env_config = {
      "flow": hgym.Step,
      "flow_config": {
          "mesh": "coarse",
          "Re": 100
      },
      "solver": hgym.IPCS,
  }
  env = hgym.FlowEnv(env_config)

  for _ in range(10):
    y, reward, done, info = env.step(0.1 * sin(env.solver.t))


def test_grad():
  flow = hgym.Step(Re=100, mesh="coarse")

  c = fda.AdjFloat(0.0)
  flow.set_control(c)

  solver = hgym.NewtonSolver(flow)
  solver.solve()

  (y,) = flow.get_observations()

  dy = fda.compute_gradient(y, fda.Control(c))

  print(dy)
  assert abs(dy) > 0
