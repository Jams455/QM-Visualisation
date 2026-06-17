import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


def foo(x, t):
    return np.sin(x * 2 * np.pi + 2 * np.pi * t)


xes = np.linspace(0, 1, 100)

p = foo(xes, 0)



fig, ax = plt.subplots()

line, = ax.plot(xes, p)

def Update(frame):
    p = foo(xes, frame / 100)

    line.set_ydata(p)

    return line

ani = animation.FuncAnimation(fig=fig, func=Update, frames=4000, interval=10)
plt.show()


