import numpy as np
import scipy.io as sio


def deriv_filter(filter_type, N, dt):
  # u[i] = (b[0]*y[i] + b[1]*y[i-1] - a[1]*u[i-1]) / a[0]
  if filter_type == "none":
    a = [dt, 0]
    b = [1, -1]
  elif filter_type == "forward":
    a = [1, N * dt - 1]
    b = [N, -N]
  elif filter_type == "backward":
    a = [N * dt + 1, -1]
    b = [N, -N]
  elif filter_type == "bilinear":
    a = [N * dt + 2, N * dt - 2]
    b = [2 * N, -2 * N]
  else:
    raise ValueError(f"Unknown filter type: {filter_type}")
  return a, b


class PDController:

  def __init__(self,
               kp,
               kd,
               dt,
               n_steps,
               filter_type="none",
               N=20,
               debug_file=None):
    self.kp = kp
    self.kd = kd
    self.N = N
    self.dt = dt

    self.a, self.b = deriv_filter(filter_type, N, dt)

    self.i = 0
    self.t = np.zeros(n_steps)
    self.u = np.zeros(n_steps)
    self.y = np.zeros(n_steps)  # Filtered lift coefficient
    self.dy = np.zeros(n_steps)  # Filtered derivative of lift coefficient

    self.debug_file = debug_file

  def __call__(self, t, obs):
    self.i += 1

    i, dt = self.i, self.dt
    u, y, dy = self.u, self.y, self.dy
    a, b, N = self.a, self.b, self.N

    u[i] = -self.kp * y[i - 1] - self.kd * dy[i - 1]

    CL, CD = obs

    # Low-pass filter and estimate derivative
    y[i] = y[i - 1] + (1 / N) * (CL - y[i - 1])
    # Use the filtered measurement to avoid direct feedthrough
    dy[i] = (b[0] * y[i] + b[1] * y[i - 1] - a[1] * dy[i - 1]) / a[0]

    if self.debug_file is not None and i % 100 == 0:
      data = {
          "y": y[:i],
          "dy": dy[:i],
          "u": u[:i],
          "t": np.arange(0, i * dt, dt),
      }
      sio.savemat(self.debug_file, data)

    return u[i]
