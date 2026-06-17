import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate
import scipy.special as special


def P_lm(l, m, x):
    ...

def Theta(l, m, theta):
    c_lm = np.sqrt(
        ( 2*l + 1) / 2 
        * special.factorial( (l - np.abs(m)) ) 
        / special.factorial( (l + np.abs(m)) )
    )

    if m >= 0:
        c_lm *= (-1)**m

    cos_theta = np.cos(theta)
    p_lm = P_lm(l, m, cos_theta)

    return c_lm * p_lm

def Phi(m, phi):
    return np.exp(1j * m * phi) / np.sqrt(2 * np.pi)

def Y(l, m, theta, phi):
    return Theta(l, m, theta) * Phi(m, phi)

def R(n, l, r):
    ...

def Psi(n, l, m, r, theta, phi):
    assert n >= 1
    assert 0 <= l < n
    assert np.abs(m) <= l

    return R(n, l, r) * Y(l, m, theta, phi)

result = integrate.quad(lambda x: Phi(0, x), 0, 2*np.pi)

print(result)
