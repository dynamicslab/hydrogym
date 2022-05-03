import hydrogym as gym

def test_import():
    cyl = gym.flows.Cylinder(mesh_name='noack')
    return cyl

def test_import2():
    cyl = gym.flows.Cylinder(mesh_name='sipp-lebedev')
    return cyl

def test_steady():
    cyl = gym.flows.Cylinder()
    w = cyl.solve_steady()

def test_forces(tol=1e-3):
    cyl = gym.flows.Cylinder(mesh_name='noack')
    w = cyl.solve_steady()
    u, p = w.split()

    # Lift/drag on cylinder
    CL, CD = cyl.compute_forces(u, p)
    assert(abs(CL) < tol)
    # assert(abs(CD - 1.8083279145880582) < tol)  # Re = 40
    assert(abs(CD - 1.294685958568012) < tol)  # Re = 100

def test_unsteady():
    cyl = gym.flows.Cylinder(mesh_name='noack')
    dt = 1e-2
    cyl.integrate(dt=dt, Tf=10*dt, output_dir=None)

if __name__=='__main__':
    test_import()