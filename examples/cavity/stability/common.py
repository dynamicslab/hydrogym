import hydrogym as gym

temp_dir = "tmp"
output_dir = "global-modes"
mesh = "fine"

Re = 5000
flow = gym.flow.Cavity(Re=Re, mesh=mesh)
