import matplotlib.animation as animation
from scipy import special as special
import matplotlib.pyplot as plt
import numpy as np

LEGENDRE_COEFFS = np.array([
    np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) / 1,
    np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) / 1,
    np.array([-1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0]) / 2,
    np.array([0, -3, 0, 5, 0, 0, 0, 0, 0, 0, 0]) / 2,
    np.array([3, 0, -30, 0, 35, 0, 0, 0, 0, 0, 0]) / 8,
    np.array([0, 15, 0, -70, 0, 63, 0, 0, 0, 0, 0]) / 8,
    np.array([-5, 0, 105, 0, -315, 0, 231, 0, 0, 0, 0]) / 16,
    np.array([0, -35, 0, 315, 0, -693, 0, 429, 0, 0, 0]) / 16,
    np.array([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435, 0, 0]) / 128,
    np.array([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155, 0]) / 128,
    np.array([-63, 0, 3465, 0, -30030, 0, 90090, 0, -109395, 0, 46189]) / 256
])

LAGUERRE_COEFFS = np.array([
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

DIFFERENTIATE_MATRIX = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
])

def L_q(q, x: np.ndarray):
    result = 0.0

    for i, Laguerre_coeffs in enumerate(LAGUERRE_COEFFS[q]):
        result += np.pow(x, i) * Laguerre_coeffs

    return result

def L_pq(p, q, x: np.ndarray):
    if p + q >= len(LAGUERRE_COEFFS):
        raise ValueError("p + q exceeds available Laguerre polynomial order")

    d_dx_matrix = np.linalg.matrix_power(DIFFERENTIATE_MATRIX, q)

    prefactor_arr = np.dot(LAGUERRE_COEFFS[p+q], d_dx_matrix)

    result = 0.0

    for i, prefactor_coeffs in enumerate(prefactor_arr):
        result += np.pow(x, i) * prefactor_coeffs

    return (-1)**q * result

def P_l(l, x: np.ndarray):
    result = 0.0

    for i, legendre_coeffs in enumerate(LEGENDRE_COEFFS[l]):
        result += np.pow(x, i) * legendre_coeffs

    return result

def P_lm(l, m, x: np.ndarray):
    m_abs = np.abs(m)

    diff_mat = np.linalg.matrix_power(DIFFERENTIATE_MATRIX, m_abs)


    prefactor_vec = np.dot(LEGENDRE_COEFFS[l], diff_mat)

    result = 0.0

    for i, prefactor_val in enumerate(prefactor_vec):
        result += np.pow(x, i) * prefactor_val



    prefactor = np.pow(-1, m_abs)

    if m < 0:
        prefactor = 1

    prefactor_func = prefactor * np.pow(np.maximum(0.0, 1-np.pow(x, 2)), m_abs / 2)

    return prefactor_func * result

def Theta_lm(l, m, theta: np.ndarray):
    if abs(m) > l:
        raise ValueError("m out of bounds. Should satisfy |m| <= l") 

    m_abs = np.abs(m)

    A = np.sqrt(
        ( 2*l + 1) / 2 
        * special.factorial( (l - m_abs) ) 
        / special.factorial( (l + m_abs) )
    )

    cos_theta = np.cos(theta)

    p_lm = P_lm(l, m, cos_theta)
    
    return A * p_lm

def Phi_m(m, phi: np.ndarray):
    return np.exp(1j * m * phi) / np.sqrt(2 * np.pi)

def R_nl(n, l, r: np.ndarray, a_0):
    if n < 1:
        raise ValueError("n out of bounds. Should satisfy n >= 1")
    
    if l < 0 or l >= n:
        raise ValueError("l out of bounds. Should satisfy 0 <= l < n")

    A = np.sqrt(
        np.pow(2 / (n * a_0), 3 ) *
        special.factorial( n - l - 1 ) / 
        ( 2 * n * np.pow( special.factorial(n + l), 3 ) )
    )

    p = n - l - 1
    q = 2 * l + 1

    rho = 2 * r / (n * a_0)
    
    r_nl = L_pq(p, q, rho)

    r_nl *= np.exp( - rho / 2 )

    r_nl *= np.pow( rho, l )

    r_nl *= A

    return r_nl

# ------------------------------------------------------------------ #
# ----------------------------- Inputs ----------------------------- #
# ------------------------------------------------------------------ #

a_0 = 1                     # Bohr radius

RES = 250                   # Number of points to plot along each axis
AXIS_LIM = 10               # Extent of each axis
FRAMES = 4000               # Number of frames to animate
TIME_STEP = 10              # Time in ms between animation frames

(n, l, m) = (2, 1, 0)      # Quantum numbers of orbital to plot

# ------------------------------------------------------------------ #
# ------------------------ Coord Conversions ----------------------- #
# ------------------------------------------------------------------ #

x = np.linspace(-AXIS_LIM, AXIS_LIM, RES)
y = np.linspace(-AXIS_LIM, AXIS_LIM, RES)
z = np.linspace(-AXIS_LIM, AXIS_LIM, RES)

Y, X, Z = np.meshgrid(x, y, z)

R = np.sqrt(np.pow(X, 2) + np.pow(Y, 2) + np.pow(Z, 2))

THETA = np.zeros_like(R)
mask = R > 0
THETA[mask] = np.arccos(Z[mask] / R[mask])

PHI = np.mod(np.atan2(Y, X), 2*np.pi)

# ------------------------------------------------------------------ #
# ----------------------- Wavefunction Calcs ----------------------- #
# ------------------------------------------------------------------ #

Psi = R_nl(n, l, R, a_0) * Theta_lm(l, m, THETA) * Phi_m(m, PHI)

prob_density = np.abs(Psi)**2
max_prob_density = np.max(prob_density)

prob_density_2d = prob_density[:, 0, :]

# ------------------------------------------------------------------ #
# ---------------------------- Plotting ---------------------------- #
# ------------------------------------------------------------------ #

fig, ax = plt.subplots()

img = ax.imshow(prob_density_2d, cmap='inferno', extent=[-AXIS_LIM, AXIS_LIM, -AXIS_LIM, AXIS_LIM], origin='lower')
img.set_clim(vmin=0, vmax=max_prob_density)

ax.set_xticklabels([])
ax.set_xticks([])

ax.set_yticklabels([])
ax.set_yticks([])

# ------------------------------------------------------------------ #
# --------------------------- Animating ---------------------------- #
# ------------------------------------------------------------------ #

def Update(frame):
    res = RES - 1
    curr_slice = 0

    lower = np.mod(frame, res)
    upper = np.mod(frame, res*2)

    if lower == upper:
        curr_slice = lower
    else:
        curr_slice = res - lower

    prob_density_2d = prob_density[:, curr_slice, :]
    
    img.set_data(prob_density_2d)
    return [img]


ani = animation.FuncAnimation(fig=fig, func=Update, frames=FRAMES, interval=TIME_STEP)
plt.show()
