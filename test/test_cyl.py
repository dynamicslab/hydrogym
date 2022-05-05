import firedrake as fd
import firedrake_adjoint as fda
import hydrogym as gym

def test_import():
    cyl = gym.flow.Cylinder(mesh_name='noack')
    return cyl

def test_import2():
    cyl = gym.flow.Cylinder(mesh_name='sipp-lebedev')
    return cyl

def test_steady(tol=1e-3):
    cyl = gym.flow.Cylinder()
    q = cyl.solve_steady()

    # Lift/drag on cylinder
    CL, CD = cyl.compute_forces(cyl.u, cyl.p)
    assert(abs(CL) < tol)
    # assert(abs(CD - 1.8083279145880582) < tol)  # Re = 40
    assert(abs(CD - 1.294685958568012) < tol)  # Re = 100

def test_unsteady():
    cyl = gym.flow.Cylinder(mesh_name='noack')
    dt = 1e-2
    solver = gym.ts.IPCSSolver(cyl, dt=dt)
    solver.solve(10*dt)
    
# Simple opposition control on lift
def feedback_ctrl(y, K=0.1):
    CL, CD = y
    return K*CL

def test_control():
    cyl = gym.flow.Cylinder(mesh_name='noack')
    dt = 1e-2

    solver = gym.ts.IPCSSolver(cyl, dt=dt, callbacks=[], time_varying_bc=True)

    num_steps = 10
    for iter in range(num_steps):
        y = cyl.collect_observations()
        cyl.set_control(feedback_ctrl(y))
        solver.step(iter)

def test_env():
    env = gym.env.CylEnv()

    u = 0.0
    for _ in range(10):
        y, reward, done, info = env.step(u)
        print(y)
        u = feedback_ctrl(y)

def test_grad():
    cyl = gym.flow.Cylinder()

    omega = fd.Constant(0.0)
    cyl.set_control(omega)

    cyl.solve_steady()
    CL, CD = cyl.compute_forces(cyl.u, cyl.p)

    dJdu = fda.compute_gradient(CD, fda.Control(omega))

def test_env_grad():
    env = gym.env.CylEnv()
    y = env.reset()
    K = fda.AdjFloat(0.0)
    m = fda.Control(K)
    J = 0.0
    for _ in range(10):
        y, reward, done, info = env.step(feedback_ctrl(y, K=K))
        J = J - reward
    dJdm = fda.compute_gradient(J, m)

if __name__=='__main__':
    test_import()