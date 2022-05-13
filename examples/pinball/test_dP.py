import firedrake as fd
import firedrake_adjoint as fda
import hydrogym as gym

flow = gym.flow.Pinball(Re=30, mesh_name='coarse')

# omega = fd.Constant((0.0, 0.0, 0.0))
n_cyl = 3
# Option 1: List of AdjFloats
# omega = [fda.AdjFloat(0.1*i) for i in range(n_cyl)]

# Option 2: List of Constants
# omega = [fd.Constant(0.1*i) for i in range(n_cyl)]
# control = [fda.Control(omg) for omg in omega]

# Option 3: Overloaded array with numpy_adjoint 
import numpy as np
import pyadjoint
omega = pyadjoint.create_overloaded_object(np.array([0.1, 0.2, 0.3]))
control = fda.Control(omega)

flow.set_control(omega)

flow.solve_steady()
CL, CD = flow.compute_forces()

dJdu = fda.compute_gradient(sum(CD), control)
print(dJdu)