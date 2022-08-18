import hydrogym as gym

temp_dir = "tmp"
output_dir = "global-modes"
mesh = "fine"

Re = 7500
flow = gym.flow.Cavity(Re=Re, mesh=mesh)
