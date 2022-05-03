from fenics import *
import hydrogym as gym

cyl = gym.flows.Cylinder()

# First initialize with constant freestream
# u, p = split(cyl.sol)
# u = interpolate(cyl.U_inf, cyl.function_space.sub(0).collapse())

# Time step
dt = 1e-2
Tf = 1.0
cyl.integrate(dt, Tf, output_dir='output')