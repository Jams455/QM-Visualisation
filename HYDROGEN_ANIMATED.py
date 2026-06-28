import matplotlib.pyplot as plt
from scipy import special as special
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
    x_arr = np.array([np.pow(x, i)  for i in range(0, len(LAGUERRE_COEFFS))])

    return np.dot(LAGUERRE_COEFFS[q], x_arr)

def L_pq(p, q, x: np.ndarray):
    x_arr = np.array([np.pow(x, i)  for i in range(0, len(LAGUERRE_COEFFS))])

    diff_n_mat = np.linalg.matrix_power(DIFFERENTIATE_MATRIX, q)

    dxarr_dx = np.dot(diff_n_mat, x_arr)

    assert p + q <= len(LAGUERRE_COEFFS)

    return np.multiply(np.pow(-1, q), np.dot(LAGUERRE_COEFFS[p+q], dxarr_dx))

def P_l(l, x: np.ndarray):
    x_arr = np.array([np.pow(x, i)  for i in range(0, len(LEGENDRE_COEFFS))])

    return np.dot(LEGENDRE_COEFFS[l], x_arr)

def P_lm(l, m, x: np.ndarray):
    m_abs = np.abs(m)

    x_arr = np.array([np.pow(x, i)  for i in range(0, len(LEGENDRE_COEFFS))])

    diff_mat = np.linalg.matrix_power(DIFFERENTIATE_MATRIX, m_abs)

    dxarr_dx = np.dot(diff_mat, x_arr)

    dmPl_dxm = np.dot(LEGENDRE_COEFFS[l], dxarr_dx)

    prefactor = np.pow(-1, m_abs)

    if m < 0:
        prefactor = 1

    p_lm = np.multiply(prefactor, np.pow(1-np.pow(x, 2), m_abs / 2))

    p_lm = np.multiply(p_lm, dmPl_dxm)

    return p_lm


def Theta_lm(l, m, theta):
    m_abs = np.abs(m)

    A = np.sqrt(
        ( 2*l + 1) / 2 
        * special.factorial( (l - m_abs) ) 
        / special.factorial( (l + m_abs) )
    )

    cos_theta = np.cos(theta)

    p_lm = P_lm(l, m, cos_theta)
    
    return A * p_lm

def Phi_m(m, phi):
    return np.exp(1j * m * phi) / np.sqrt(2 * np.pi)

def R_nl(n, l, r):
    global a
    A = np.sqrt(
        np.pow(2 / (n * a), 3 ) *
        special.factorial( n - l - 1 ) / 
        ( 2 * n * np.pow( special.factorial(n + l), 3 ) )
    )

    p = n - l - 1
    q = 2 * l + 1

    rho = 2 * r / (n * a)
    
    r_nl = L_pq(p, q, rho)

    r_nl *= np.exp( - rho / 2 )

    r_nl *= np.pow( rho, l )

    r_nl *= A

    return r_nl




xz_max = 40

r_min = 0
r_max = int(np.sqrt(3 * xz_max**2))
r_steps = 5000
r = np.linspace(r_min, r_max, r_steps)

theta_min = 0
theta_max = np.pi
theta_steps = 5000
theta = np.linspace(theta_min, theta_max, theta_steps)

phi_min = 0
phi_max = 2 * np.pi
phi_steps = 1000
phi = np.linspace(phi_min, phi_max, phi_steps)

res = 500
res = int((res / 2)) * 2



a = 1

n1 = 4
l1 = 3
m1 = 1

n2 = 4
l2 = 3
m2 = 1

r_n1_l1 = R_nl(n1, l1, r)
theta_l1_m1 = Theta_lm(l1, m1, theta)
phi_m1 = Phi_m(m1, phi)

r_n2_l2 = R_nl(n2, l2, r)
theta_l2_m2 = Theta_lm(l2, m2, theta)
phi_m2 = Phi_m(m2, phi)

WFN1_0 = []
WFN2_0 = []

for z in np.linspace(-xz_max, xz_max, res):
    WFN1_0_temp = []
    WFN2_0_temp = []

    for x in np.linspace(-xz_max, xz_max, res):
        y = 0

        r_true_val = np.sqrt(np.pow(x, 2) + np.pow(y, 2) + np.pow(z, 2))
        r_ind = int( r_true_val * r_steps / np.max(r) )
        
        theta_true_val = np.arccos(z / r_true_val)
        theta_ind = int( theta_true_val * theta_steps / np.max(theta) )
        
        phi_true_val = np.arccos(x / np.abs(x))
        phi_ind = int( phi_true_val * phi_steps / np.max(phi) )
        


        R_1 = r_n1_l1[r_ind]
        Theta_1 = theta_l1_m1[theta_ind]
        Phi_1 = phi_m1[phi_ind]

        R_2 = r_n2_l2[r_ind]
        Theta_2 = theta_l2_m2[theta_ind]
        Phi_2 = phi_m2[phi_ind]
        
        WFN1_0_temp.append(R_1 * Theta_1 * Phi_1)
        WFN2_0_temp.append(R_2 * Theta_2 * Phi_2)
    
    WFN1_0.append(WFN1_0_temp)
    WFN2_0.append(WFN2_0_temp)

fix, ax = plt.subplots()

ax.imshow(np.abs(WFN1_0), origin='lower', cmap='inferno')

#ax.set_xlabel("x")
#ax.set_ylabel("z")

ax.set_xticklabels([])
ax.set_xticks([])

ax.set_yticklabels([])
ax.set_yticks([])

plt.show()
