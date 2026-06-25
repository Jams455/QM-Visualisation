import matplotlib.animation as animation
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy.special as special
import scipy.stats as stats
from autograd import grad
import numpy as np


def P_inner(l, x):
    return np.pow(np.pow(x,2) - 1, l)

def P_l(l, x: np.ndarray):
    p_l = P_inner

    for _ in range(l):
        p_l = grad(p_l, 1)

    return_list = []

    for x_i in x:
        return_list.append( p_l(l, x_i) / (np.pow(2, n) * special.factorial(l)))

    return np.array(return_list)

def P_lm(l, m, x: np.ndarray):
    m_abs = np.abs(m)

    p_lm = P_inner

    for _ in range(l + m_abs):
        p_lm = grad(p_lm, 1)

    if type(x) == np.ndarray:
        return_list = []

        for x_i in x:
            val = p_lm(l, x_i)

            val *= np.pow( 1 - np.pow(x_i, 2) , m_abs / 2 )

            val /= np.pow(2, l) * special.factorial(l)

            return_list.append(val)

        return_list = np.array(return_list)

        if m >= 0:
            return_list *= np.pow(-1, m)
        
        return return_list
    else:
        val = p_lm(l, x)

        val *= np.pow( 1 - np.pow(x, 2) , m_abs / 2 )

        val /= np.pow(2, l) * special.factorial(l)

        if m >= 0:
            val *= np.pow(-1, m)

        return val

def L_inner(q, x):
    return np.pow(np.e, -1 * x) * np.pow(x, q)

def L_q(q, x: np.ndarray):
    l_q = L_inner

    for _ in range(q):
        l_q = grad(l_q, 1)

    if type(x) == np.ndarray:
        return_list = []

        for x_i in x:
            return_list.append( np.exp(x_i) * l_q(q, x_i) )

        return np.array(return_list)
    else:
        return np.pow(np.e, x) * l_q(q, x)

def L_pq(p, q, x):
    l_pq = L_q

    for _ in range(q):
        l_pq = grad(l_pq, 1)

    if type(x) == np.ndarray:
        return_list = []

        for x_i in x:
            val = l_pq(p+q, x_i)

            val *= np.pow( -1 , q )

            return_list.append(val)

        return_list = np.array(return_list)

        return return_list
    else:
        val = l_pq(p+q, x)

        val *= np.pow( -1 , q )
        
        return val

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

xes = np.linspace(-10, 10, 25)

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
