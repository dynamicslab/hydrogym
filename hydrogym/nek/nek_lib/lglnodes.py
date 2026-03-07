"""
A function for identify the lglnodes by order N
@yuningw
"""
import numpy as np


def lglnodes(N):
  # Truncation + 1
  N1 = N + 1

  # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
  x = np.cos(np.pi * np.arange(N + 1) / N).astype(np.float64)

  # The Legendre Vandermonde Matrix
  P = np.zeros((N1, N1)).astype(np.float64)

  # Compute P_(N) using the recursion relation
  # Compute its first and second derivatives and
  # update x using the Newton-Raphson method.
  xold = 2

  while np.max(np.abs(x - xold)) > np.finfo(float).eps:
    xold = x.copy()

    P[:, 0] = 1
    P[:, 1] = x

    for k in range(2, N1):
      P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k

    x = xold - (x * P[:, N1 - 1] - P[:, N - 1]) / (N1 * P[:, N1 - 1])

  w = 2 / (N * N1 * P[:, N1 - 1]**2)

  return x, w, P


class gll_Mapper:

  def __init__(self, poly_ord=5):
    self.N = poly_ord
    self.n, self.w, self.P = self.lglnodes()

  def cartisan_2_gll(self, e_max, e_min):
    """
        Use the upper and lower bound of the domain to interpolate the GLL points 

        """
    h = e_max - e_min
    gll_nodes = np.linspace(e_min, e_max, self.N + 1).astype(np.float64)

    gll_nodes = ((1 + self.n) / 2) * h + e_min
    print(f"Gll_nodes = {gll_nodes}")

    return gll_nodes

  def lglnodes(self):
    N = self.N
    # Truncation + 1
    N1 = N + 1

    # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
    x = np.cos(np.pi * np.arange(N + 1) / N).astype(np.float64)

    # The Legendre Vandermonde Matrix
    P = np.zeros((N1, N1)).astype(np.float64)

    # Compute P_(N) using the recursion relation
    # Compute its first and second derivatives and
    # update x using the Newton-Raphson method.
    xold = 2

    while np.max(np.abs(x - xold)) > np.finfo(float).eps:
      xold = x.copy()

      P[:, 0] = 1
      P[:, 1] = x

      for k in range(2, N1):
        P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k

      x = xold - (x * P[:, N1 - 1] - P[:, N - 1]) / (N1 * P[:, N1 - 1])

    w = 2 / (N * N1 * P[:, N1 - 1]**2)

    return x, w, P


if __name__ == '__main__':
  """
    A very simple check based on 1D domain [0,2] of GLL points
    """

  import matplotlib.pyplot as plt
  plt.rc("font", family="serif")
  plt.rc("font", size=22)
  plt.rc("axes", labelsize=16, linewidth=2)
  plt.rc("legend", fontsize=12, handletextpad=0.3)
  plt.rc("xtick", labelsize=18)
  plt.rc("ytick", labelsize=18)

  class colorplate:
    red = "#D23918"  # luoshenzhu
    blue = "#2E59A7"  # qunqing
    yellow = "#E5A84B"  # huanghe liuli
    cyan = "#5DA39D"  # er lv
    black = "#151D29"  # lanjian
    gray = "#DFE0D9"  # ermuyu
    grays = "#6B6C6E"  # ermuyu

  domain = np.array([0, 1], dtype=np.float64)
  N = 7
  mapper = gll_Mapper(N)

  gll_nodes = mapper.cartisan_2_gll(e_max=domain.max(), e_min=domain.min())
  x, w, P = mapper.lglnodes()
  print(f"x = {x}")
  print(f"w = {w}")
  print(f"P = {P}")
  with open(f'GLL_{N}.dat', 'w') as f:
    for i, gi in enumerate(gll_nodes):
      f.write(f"{i}\t{gi}\n")
  f.close()

  fig, axs = plt.subplots(1, 1, figsize=(7, 2))

  y_domain = np.zeros_like(domain)
  y_gll = np.zeros_like(gll_nodes)
  axs.plot(
      domain,
      y_domain,
      's-',
      markersize=10,
      c=colorplate.red,
      label='Element Nodes')
  axs.plot(
      gll_nodes, y_gll, 'o', markersize=8, c=colorplate.blue, label='GLL Nodes')
  axs.grid()
  axs.set_ylabel('Value')
  axs.set_xlabel('Grids')
  axs.set_title(
      f'Polynomial Order = {N}\n Domain [{domain.min()}, {domain.max()}]')
  axs.legend(loc='upper right')
  # fig.savefig('Figs/Validation.jpg',bbox_inches='tight',dpi=300)
  plt.show()
