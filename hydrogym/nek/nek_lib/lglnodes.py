"""
Gauss-Lobatto-Legendre (GLL) nodes and weights of order N (author: Yuning Wang).
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

    w = 2 / (N * N1 * P[:, N1 - 1] ** 2)

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

        w = 2 / (N * N1 * P[:, N1 - 1] ** 2)

        return x, w, P
