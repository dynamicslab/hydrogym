import hydrogym as hgym

temp_dir = "tmp"
output_dir = "global-modes"
mesh = "fine"

Re = 7500
flow = hgym.flow.Cavity(Re=Re, mesh=mesh)
