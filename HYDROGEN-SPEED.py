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







def Theta(l, m, theta):
    m_abs = np.abs(m)

    A = np.sqrt(
        ( 2*l + 1) / 2 
        * special.factorial( (l - m_abs) ) 
        / special.factorial( (l + m_abs) )
    )

    cos_theta = np.cos(theta)

    p_lm = P_lm(l, m, cos_theta)
    
    return A * p_lm

def Phi(m, phi):
    return np.exp(1j * m * phi) / np.sqrt(2 * np.pi)

def Y_lm(l, m, theta, phi):
    return Theta(l, m, theta) * Phi(m, phi)

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

def Psi(n, l, m, r, theta, phi):
    assert n >= 1
    assert 0 <= l < n
    assert np.abs(m) <= l

    return Y_lm(l, m, theta, phi) * R_nl(n, l, r)




a = 1

# result = integrate.quad(lambda x: R_nl(2, 1, x) * np.pow(x, 2), 0, 300)

# print(result)




n = 2
l = 1
m = -1


fig, ax = plt.subplots()



Psi_vals = []

xes = np.linspace(-10, 10, 200)

for y in xes:
    psi_vals = []
    for z in xes:
        r = np.sqrt(np.pow(z, 2) + np.pow(y, 2))
        
        phi = np.pi / 2

        theta = np.pi / 2

        if z > 0:
            theta = np.arctan(y / z)
        if z < 0 and y >= 0:
            theta = np.arctan(y / z) + np.pi
        elif z < 0 and y < 0:
            theta = np.arctan(y / z) - np.pi
        
        psi_vals.append(np.abs(Psi(n, l, m, r, theta, phi))**2)

    Psi_vals.append(psi_vals)

Psi_vals /= np.max(Psi_vals)

ax.imshow(Psi_vals, cmap='plasma')

plt.show()



