import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def integrate(x, y, z):
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    
    total = 0

    for i in range(len(x)):
        for j in range(len(y)):
            total += z[i][j] * hx * hy

    return total

def Psi_nx_ny_x_y_0(nx, ny, x, y, lx, ly, c_nx_ny):
    if nx < 1 or ny < 1:
        raise ValueError("nx/ny out of bounds. Should satisfy n >= 1.")
    if nx != int(nx) or ny != int(ny):
        raise ValueError("nx/ny must be an integer.")

    psi_nx_ny_x_y_0 = (c_nx_ny 
                 * np.sin(nx * np.pi * x / lx) 
                 * np.sin(ny * np.pi * y / ly)
                 * np.sqrt(4 / (lx * ly)))

    return psi_nx_ny_x_y_0

# ------------------------------------------------------------------ #
# ----------------------------- Inputs ----------------------------- #
# ------------------------------------------------------------------ #

LX = 1              # Length of box in x-direction
LY = 1              # Length of box in y-direction
M = 300             # Particle mass
HBAR = 1            # Reduced Planck constant
RES = 250           # Number of points plotted along each axis
TIME_STEP = 10      # Time in ms between animation frames

c_nx_ny = {}        # Coefficient of each stationary state in superposition (dict key = (n_x, n_y), value = coeff)

c_nx_ny[(6, 3)] =  1.0 + 0.0j
c_nx_ny[(3, 6)] =  1.0 - 5.0j
# c_nx_ny[(4, 3)] =  1.0 - 1.0j
# c_nx_ny[(5, 2)] =  0.0 + 1.0j
# c_nx_ny[(7, 9)] =  1.0 + 0.0j
# c_nx_ny[(9, 3)] =  1.0 + 0.0j

# ------------------------------------------------------------------ #
# --------------------- Normalise Coefficients --------------------- #
# ------------------------------------------------------------------ #

c_nx_ny_magnitude = 0

for c in c_nx_ny.values():
    c_nx_ny_magnitude += np.abs(c)**2

if c_nx_ny_magnitude <= 0:
    raise ValueError("Coefficients not valid")

c_nx_ny_magnitude = np.sqrt(c_nx_ny_magnitude)

c_nx_ny = {key: value / c_nx_ny_magnitude for key, value in c_nx_ny.items()}

# ------------------------------------------------------------------ #
# ------------- Calc Wavefuncs and Check Normalisation ------------- #
# ------------------------------------------------------------------ #

x = np.linspace(0, LX, RES)
y = np.linspace(0, LY, RES)

Y, X = np.meshgrid(x, y)

wavefunctions = {}

print("\nNormalisation check:")
for key, c in c_nx_ny.items():
    wfn = Psi_nx_ny_x_y_0(key[0], key[1], X, Y, LX, LY, c)
    
    wfn_integral = integrate(x, y, np.abs(wfn/c)**2)
    
    print(f"|Psi_n={key}|**2 integral over 0<=x<=L: {wfn_integral}")

    wavefunctions[key] = wfn
print()

psi_x_y_0 = sum(wavefunctions.values())
prob_density = np.abs(psi_x_y_0) ** 2
MAXZ = np.max(prob_density)

# ------------------------------------------------------------------ #
# -------------------------- Create Plot --------------------------- #
# ------------------------------------------------------------------ #

fig, ax = plt.subplots()

img = ax.imshow(prob_density, cmap='inferno', origin='lower')
img.set_clim(vmin=0, vmax=MAXZ)

def Update(frame):
    for (nx, ny) in c_nx_ny.keys():
        E_nx_ny = np.pi**2 * HBAR**2 * ((nx / LX)**2 + (ny / LY)**2) / ( 2 * M )

        time_evolution_factor = np.exp(-1j * E_nx_ny * (TIME_STEP/1000) / HBAR)

        wavefunctions[key] *= time_evolution_factor

    psi_x_y_0 = sum(wavefunctions.values())
    prob_density = np.abs(psi_x_y_0) ** 2

    temp_max = np.max(prob_density)

    global MAXZ
    if temp_max > MAXZ:
        MAXZ = temp_max

    img.set_data(prob_density)
    img.set_clim(vmin=0, vmax=MAXZ)

    return [img]


ani = animation.FuncAnimation(fig=fig, func=Update, frames=40000, interval=TIME_STEP)
plt.show()
