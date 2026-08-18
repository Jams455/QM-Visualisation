import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def Psi_x_y_t(x, y, t):
    global Lx
    global Ly
    global hbar
    global m
    global c

    psi_x_y_t = 0+0j

    for i in range(1, len(c)+1):
        for j in range(1, len(c[0])+1):
            sx = np.sin(i * np.pi * x / Lx) / np.sqrt(Lx)
            sy = np.sin(j * np.pi * y / Ly) / np.sqrt(Ly)

            E = (hbar**2) * (np.pi**2) * (i**2 / (Lx**2) + j**2 / (Ly**2)) / (2*m)
            e = np.exp(-1j * E * t / hbar)

            psi_x_y_t += c[i-1][j-1] * sx * sy * e * 2

    return psi_x_y_t

c = np.array([
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
    ], dtype=np.float64)

c /= np.sqrt(np.sum(np.abs(c)**2))

Lx = 1
Ly = 1
hbar = 0.6
m = 1


xes = np.linspace(0, 1, 600)
yes = np.linspace(0, 1, 600)

p = []
for x in xes:
    temp = np.abs(Psi_x_y_t(x, yes, 0))**2

    p.append(temp)





fig, ax = plt.subplots()

img = ax.imshow(p, cmap='inferno')

def Update(frame):
    p = []
    for x in xes:
        temp = np.abs(Psi_x_y_t(x, yes, frame/200))**2

        p.append(temp)

    img.set_data(p)
    return [img]


ani = animation.FuncAnimation(fig=fig, func=Update, frames=4000, interval=20)
plt.show()

# Can be optimised by calculating Psi(t) from Psi(t = 0)
