import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def integrate(x, y, z):
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    
    total = 0

    for i in range(len(x)):
        for j in range(len(y)):
            total += z[i][j] * hx * hy

    return total

def integrate(x, y):
    h = (x[-1] - x[0]) / (len(x) - 1)
    
    y_tot = 2 * np.sum(y) - y[0] - y[-1]

    return 0.5 * h * y_tot

def Psi_x_t(x, t):
    psi_x_t = 0+0j

    for n in range(1, len(c)+1):
        s = np.sin(n * np.pi * x / L)

        e = np.exp(-1j * hbar * (n**2) * (np.pi**2) * t / (2*m) )

        psi_x_t += c[n-1] * s * e

    return psi_x_t

c = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1])

L = 1
hbar = 0.6
m = 1






xes = np.linspace(0, L, 1000)

p = np.abs(Psi_x_t(xes, 0))**2
A = integrate(xes, p)
p /= A

MAXY = 0

fig, ax = plt.subplots()

ax.set_ylim(0, 10)

line, = ax.plot(xes, p)

def Update(frame):
    p = np.abs(Psi_x_t(xes, frame/10000))**2
    A = integrate(xes, p)
    p /= A

    temp_max = np.max(p)

    global MAXY
    if temp_max > MAXY:
        MAXY = temp_max

    ax.set_ylim(0, MAXY)

    line.set_ydata(p)

    return line


ani = animation.FuncAnimation(fig=fig, func=Update, frames=40000, interval=10)
plt.show()

# Can be optimised by calculating Psi(t) from Psi(t = 0)
