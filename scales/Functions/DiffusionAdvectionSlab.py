import math
import numpy as np
from numpy import exp, sqrt
from scipy.special import erfc
import matplotlib.pyplot as plt


def Profile(tau, zeta, Pe: float):

    t = None
    x = None
    if isinstance(tau, float) or isinstance(tau, int):
        t = np.full((1, 1), fill_value=tau, dtype=float)
    elif isinstance(tau, np.ndarray):
        t = tau.reshape((-1, 1))
    else:
        raise TypeError('The provided tau cannot be used')

    if isinstance(zeta, float) or isinstance(zeta, int):
        x = np.full((1, 1), fill_value=zeta, dtype=float)
    elif isinstance(zeta, np.ndarray):
        x = zeta.reshape((1, -1))
    else:
        raise TypeError('The provided zeta cannot be used')

    if Pe <= 0.:
        raise ValueError('Pe <=0? Are you stupid?')

    phi = -erfc((Pe * t - x) / (2. * sqrt(t)))
    phi += exp(Pe * x) * erfc((Pe * t + x) / (2. * sqrt(t)))
    phi += 2.
    phi *= 0.5
    return phi


def PulseResponse(tau, zeta, Pe: float):
    t = None
    x = None
    if isinstance(tau, float) or isinstance(tau, int):
        t = np.full((1, 1), fill_value=tau, dtype=float)
    elif isinstance(tau, np.ndarray):
        t = tau.reshape((-1, 1))
    else:
        raise TypeError('The provided tau cannot be used')

    if isinstance(zeta, float) or isinstance(zeta, int):
        x = np.full((1, 1), fill_value=zeta, dtype=float)
    elif isinstance(zeta, np.ndarray):
        x = zeta.reshape((1, -1))
    else:
        raise TypeError('The provided zeta cannot be used')

    if Pe <= 0.:
        raise ValueError('Pe <=0? Are you stupid?')

    phi_r = 0.5 * x * exp((-(x-Pe*t)**2)/(4*t))/(sqrt(math.pi) * t**(3./2.))
    return phi_r


if __name__ == '__main__':
    x = 1
    t = np.linspace(0.01, 1., num=1000, endpoint=True)
    for Pe in range(1, 100, 10):
        phi = Profile(tau=t, zeta=x, Pe=Pe)
        phi = PulseResponse(tau=t, zeta=x, Pe=Pe)
        plt.plot(t.reshape((-1)), phi.reshape((-1)))

    # tau_peak = np.linspace(0.07, 0.166, num=100)
    # Pe_peak = sqrt(1/tau_peak**2-6/tau_peak)
    # plt.plot(tau_peak, Pe_peak)
    plt.pause(0)
    print('finished')
