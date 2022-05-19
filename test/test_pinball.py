import firedrake as fd
import firedrake_adjoint as fda
import hydrogym as gym
import numpy as np
import pyadjoint

def test_import():
    flow = gym.flow.Pinball(mesh='coarse')
    return flow

def test_import2():
    flow = gym.flow.Pinball(mesh='fine')
    return flow

def test_steady(tol=1e-2):
    flow = gym.flow.Pinball(Re=30, mesh='coarse')
    flow.solve_steady()

    CL_target = (0.0, 0.520, -0.517)  # Slight asymmetry in mesh
    CD_target = (1.4367, 1.553, 1.554)

    CL, CD = flow.compute_forces()
    for i in range(len(CL)):
        assert(abs(CL[i] - CL_target[i]) < tol)
        assert(abs(CD[i] - CD_target[i]) < tol)

def test_rotation(tol=1e-2):
    flow = gym.flow.Pinball(Re=30, mesh='coarse')
    flow.set_control((0.5, 0.5, 0.5))
    flow.solve_steady()

    CL_target = (0.2718, 0.5263, -0.6146)  # Slight asymmetry in mesh
    CD_target = (1.4027, 1.5166, 1.5696)

    CL, CD = flow.compute_forces()
    for i in range(len(CL)):
        assert(abs(CL[i] - CL_target[i]) < tol)
        assert(abs(CD[i] - CD_target[i]) < tol)

def test_unsteady():
    flow = gym.flow.Pinball(mesh='coarse')
    dt = 1e-2
    gym.ts.integrate(flow, t_span=(0, 10*dt), dt=dt)
    

def test_control():
    flow = gym.flow.Pinball(mesh='coarse')
    dt = 1e-2

    # Simple opposition control on lift
    def feedback_ctrl(y, K=None):
        if K is None: K = -0.1*np.ones((3, 3)) # [Inputs x outputs]
        CL, CD = y
        return K @ CL

    solver = gym.ts.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        y = flow.collect_observations()
        flow = solver.step(iter, control=feedback_ctrl(y))

def test_env():
    env = gym.env.PinballEnv(Re=30, mesh='coarse')

    # Simple opposition control on lift
    def feedback_ctrl(y, K=None):
        if K is None: K = -0.1*np.ones((3, 3)) # [Inputs x outputs]
        CL, CD = y
        return K @ CL

    u = 0.0
    for _ in range(10):
        y, reward, done, info = env.step(u)
        print(y)
        u = feedback_ctrl(y)

def test_grad():
    flow = gym.flow.Pinball(Re=30, mesh='coarse')
    n_cyl = len(flow.CYLINDER)

    # Option 1: List of AdjFloats
    # omega = [fda.AdjFloat(0.1*i) for i in range(n_cyl)]

    # Option 2: List of Constants
    # omega = [fd.Constant(0.1*i) for i in range(n_cyl)]
    # control = [fda.Control(omg) for omg in omega]

    # Option 3: Overloaded array with numpy_adjoint 
    omega = pyadjoint.create_overloaded_object(np.zeros(n_cyl))
    control = fda.Control(omega)

    flow.set_control(omega)
    flow.solve_steady()
    CL, CD = flow.compute_forces()

    dJdu = fda.compute_gradient(sum(CD), control)

def test_env_grad():
    # Simple feedback control on lift
    def feedback_ctrl(y, K):
        CL, CD = y
        return K @ CL
        
    env = gym.env.PinballEnv(Re=30, differentiable=True, mesh='coarse')
    y = env.reset()
    n_cyl = 3
    K = pyadjoint.create_overloaded_object( -0.1*np.ones((n_cyl, n_cyl)) )
    J = 0.0
    for _ in range(10):
        y, reward, done, info = env.step(feedback_ctrl(y, K))
        J = J - reward
    dJdm = fda.compute_gradient(J, fda.Control(K))

# def test_lti():
#     flow = gym.flow.Cylinder()
#     qB = flow.solve_steady()
#     A, M = flow.linearize(qB, backend='scipy')
#     A_adj, M = flow.linearize(qB, adjoint=True, backend='scipy')