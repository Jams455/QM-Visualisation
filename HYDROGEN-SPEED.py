import matplotlib.pyplot as plt
from scipy import special as special
import numpy as np



laguerre_coeffs = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, -4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [6, -18, 9, -1, 0, 0, 0, 0, 0, 0, 0],
    [24, -96, 72, -16, 1, 0, 0, 0, 0, 0, 0],
    [120, -600, 600, -200, 25, -1, 0, 0, 0, 0, 0],
    [720, -4320, 5400, -2400, 450, -36, 1, 0, 0, 0, 0],
    [5040, -35280, 52920, -29400, 7350, -882, 49, -1, 0, 0, 0],
    [40320, -322560, 564480, -376320, 117600, -18816, 1568, -64, 1, 0, 0],
    [362880, -3265920, 6531840, -5080320, 1905120, -381024, 42336, -2592, 81, -1, 0],
    [3628800, -36288000, 81648000, -72576000, 31752000, -7620480, 1058400, -86400, 4050, -100, 1]
])

x = np.linspace(-5, 20, 100)

# [(x_i)^q -- q][x_i -- i]
x_arr = np.array([np.pow(x, i)  for i in range(0, len(laguerre_coeffs))])

# [L_q -- q][L_q(x_i) -- i]
result = np.dot(laguerre_coeffs, x_arr)

fig, ax = plt.subplots()

for i in range(6):
    result[i] /= special.factorial(i)

    ax.plot(x, result[i])

ax.set_ylim(-10, 20)

plt.show()
