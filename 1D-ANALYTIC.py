import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def integrate(x, y):
    h = (x[-1] - x[0]) / (len(x) - 1)
    
    y_tot = 2 * np.sum(y) - y[0] - y[-1]

    return 0.5 * h * y_tot

def Psi_n_x_0(n, x, l, c):
    if n < 1:
        raise ValueError("n out of bounds. Should satisfy n >= 1.")
    if n != int(n):
        raise ValueError("n must be an integer.")

    psi_n_x_0 = c * np.sin(n * np.pi * x / l) * np.sqrt(2 / l)

    return psi_n_x_0

# ------------------------------------------------------------------ #
# ----------------------------- Inputs ----------------------------- #
# ------------------------------------------------------------------ #

L = 1               # Length of box
M = 500             # Particle mass
HBAR = 1            # Reduced Planck constant
RES = 1000          # Number of points plotted
TIME_STEP = 10      # Time in ms between animation frames

c_n = {}            # Coefficient of each stationary state in superposition (dict key = n, value = coeff)

c_n[1] = 1.0 + 0.0j
c_n[3] = 1.0 + 0.0j
c_n[4] = 1.0 - 1.0j
c_n[5] = 0.0 + 1.0j
c_n[7] = 1.0 + 0.0j
c_n[9] = 1.0 + 0.0j

# ------------------------------------------------------------------ #
# --------------------- Normalise Coefficients --------------------- #
# ------------------------------------------------------------------ #

c_n_magnitude = 0.0

for z in c_n.values():
    c_n_magnitude += np.abs(z)**2

if c_n_magnitude <= 0:
    raise ValueError("Coefficients not valid")

c_n_magnitude = np.sqrt(c_n_magnitude)

c_n = {key: value / c_n_magnitude for key, value in c_n.items()}

# ------------------------------------------------------------------ #
# ------------- Calc Wavefuncs and Check Normalisation ------------- #
# ------------------------------------------------------------------ #

X = np.linspace(0, L, RES)

wavefunctions = {}

print("\nNormalisation check:")
for key, c in c_n.items():
    wfn = Psi_n_x_0(key, X, L, c)

    wfn_integral = integrate(X, np.abs(wfn/c)**2)

    print(f"|Psi_n={key}|**2 integral over 0<=x<=L: {wfn_integral}")

    wavefunctions[key] = wfn
print()

psi_x_0 = sum(wavefunctions.values())
prob_density = np.abs(psi_x_0) ** 2
MAXY = np.max(prob_density)

# ------------------------------------------------------------------ #
# -------------------------- Create Plot --------------------------- #
# ------------------------------------------------------------------ #

fig, ax = plt.subplots()

ax.set_ylim(0, MAXY)

line, = ax.plot(X, prob_density)

def Update(frame):
    for n, c in c_n.items():
        E_n = n**2 * np.pi**2 * HBAR**2 / ( 2 * M * L**2 )

        time_evolution_factor = np.exp(-1j * E_n * (TIME_STEP/1000) / HBAR)

        wavefunctions[key] *= time_evolution_factor

    psi_x_t = sum(wavefunctions.values())
    prob_density = np.abs(psi_x_t) ** 2

    temp_max = np.max(prob_density)

    global MAXY
    if temp_max > MAXY:
        MAXY = temp_max

    ax.set_ylim(0, MAXY)

    line.set_ydata(prob_density)

    return line


ani = animation.FuncAnimation(fig=fig, func=Update, frames=40000, interval=TIME_STEP)
plt.show()
