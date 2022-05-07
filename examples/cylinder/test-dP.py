import firedrake as fd
import firedrake_adjoint as fda
import hydrogym as gym

# Simple opposition control on lift
def feedback_ctrl(y, K=0.1):
    CL, CD = y
    return K*CL

env = gym.env.CylEnv(differentiable=True)
y = env.reset()
K = fda.AdjFloat(0.0)
m = fda.Control(K)
J = 0.0
for _ in range(10):
    y, reward, done, info = env.step(feedback_ctrl(y, K=K))
    J = J - reward
dJdm = fda.compute_gradient(J, m)
print(dJdm)