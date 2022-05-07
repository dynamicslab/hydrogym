import firedrake as fd
import firedrake_adjoint as fda
import hydrogym as gym

def test_import():
    flow = gym.flow.Cylinder(mesh_name='noack')
    return flow

def test_import2():
    flow = gym.flow.Cylinder(mesh_name='sipp-lebedev')
    return flow

def test_steady():
    flow = gym.flow.Cylinder()
    q = flow.solve_steady()
    tol=1e-3

    # Lift/drag on cylinder
    CL, CD = flow.compute_forces()
    assert(abs(CL) < tol)
    # assert(abs(CD - 1.8083279145880582) < tol)  # Re = 40
    assert(abs(CD - 1.2840) < tol)  # Re = 100

def test_unsteady():
    flow = gym.flow.Cylinder(mesh_name='noack')
    dt = 1e-2
    gym.ts.integrate(flow, t_span=(0, 10*dt), dt=dt)
    
# Simple opposition control on lift
def feedback_ctrl(y, K=0.1):
    CL, CD = y
    return K*CL

def test_control():
    flow = gym.flow.Cylinder(mesh_name='noack')
    dt = 1e-2

    solver = gym.ts.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        y = flow.collect_observations()
        flow.set_control(feedback_ctrl(y))
        flow = solver.step(iter)

def test_env():
    env = gym.env.CylEnv()

    u = 0.0
    for _ in range(10):
        y, reward, done, info = env.step(u)
        print(y)
        u = feedback_ctrl(y)

def test_grad():
    flow = gym.flow.Cylinder()

    omega = fd.Constant(0.0)
    flow.set_control(omega)

    flow.solve_steady()
    CL, CD = flow.compute_forces()

    dJdu = fda.compute_gradient(CD, fda.Control(omega))

def test_env_grad():
    env = gym.env.CylEnv(differentiable=True)
    y = env.reset()
    K = fda.AdjFloat(0.0)
    m = fda.Control(K)
    J = 0.0
    for _ in range(10):
        y, reward, done, info = env.step(feedback_ctrl(y, K=K))
        J = J - reward
    dJdm = fda.compute_gradient(J, m)

def test_lti():
    flow = gym.flow.Cylinder()
    qB = flow.solve_steady()
    M, A, B = flow.linearize(qB, control=True, backend='scipy')

if __name__=='__main__':
    test_import()