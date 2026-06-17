import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


def foo(x, y, t):
    return np.sin(2 * np.pi * x + t) + np.sin(2 * np.pi * y - t)




xes = np.linspace(0, 1, 100)
yes = np.linspace(0, 1, 100)

p = []
for x in xes:
    temp = foo(x, yes, 0)

    p.append(temp)

p = np.array(p)



fig, ax = plt.subplots()

img = ax.imshow(p, cmap='plasma')

def Update(frame):
    p = []
    for x in xes:
        temp = foo(x, yes, frame/10)

        p.append(temp)

    p = np.array(p)

    ax.imshow(p, cmap='plasma')

    return img

ani = animation.FuncAnimation(fig=fig, func=Update, frames=4000, interval=1)
plt.show()


