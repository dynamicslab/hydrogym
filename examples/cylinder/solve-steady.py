import hydrogym as gym

flow = gym.flow.Cylinder(Re=100)
flow.solve_steady(solver_parameters={"snes_monitor": None})

lift, drag = flow.compute_forces()
gym.print(f"Lift:{lift:08f} \t\tDrag:{drag:08f}")
