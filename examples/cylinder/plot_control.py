import numpy as np
import matplotlib.pyplot as plt

n_data = np.loadtxt('output/coeffs.dat')           # Natural
a_data = np.loadtxt('controlled/coeffs.dat')  # Controlled

plt.figure()
plt.plot(n_data[:, 0], n_data[:, 2])
plt.plot(a_data[:, 0], a_data[:, 2])
plt.show()