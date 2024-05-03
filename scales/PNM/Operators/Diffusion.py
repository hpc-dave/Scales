import openpnm as op
import scipy.linalg
import scipy.sparse
import spheres_and_cylinders as geo_model
import numpy as np
import scipy
import math
from OP_operators import construct_grad, construct_div, construct_ddt, EnforcePrescribed

Nx = 10
Ny = 1
Nz = 1
spacing = 1./Nx

# get network
network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)

# add geometry

geo = geo_model.spheres_and_cylinders
network.add_model_collection(geo, domain='all')
network.regenerate_models()

c = np.zeros((network.Np, 1))
c_old = c.copy()
bc = {}
bc['left'] = {'prescribed': 1.}
# bc['right'] = {'prescribed': 0.}

x = np.ndarray.flatten(c).reshape((c.size, 1))
dx = np.zeros_like(x)

dt = 0.01
tsteps = range(1, int(1./dt))
sol = np.zeros_like(c)
sol = np.tile(sol, reps=len(tsteps)+1)
pos = 0
tol = 1e-6
max_iter = 10
time = dt

# need to provide div with some weights, namely an area
# that the flux act on
A_flux = np.zeros((network.Nt, 1), dtype=float) + network['pore.volume'][0]/spacing

grad = construct_grad(network=network)
div = construct_div(network=network, weights=A_flux)
ddt = construct_ddt(network=network, dt=dt)

D = np.ones((network.Nt, 1), dtype=float)
J = ddt - div(D, grad)


def AnalyticalSolution(t, zeta):
    s_inf = 1
    s_tr = np.zeros_like(c).ravel()
    for n in range(100):
        npi = (n + 0.5) * math.pi
        s_tr += 2 * np.cos(n * math.pi)/npi * np.exp(-(npi)**2 * t) * np.cos(npi * zeta)
    return (s_inf - s_tr).reshape(c.shape)


zeta = np.asarray([network['pore.coords'][i][0] for i in range(network.Np)])
zeta = zeta - zeta[0]
zeta = zeta / (zeta[-1]+0.5*spacing)

for t in tsteps:
    x_old = x.copy()
    pos += 1

    G = J * x - ddt * x_old
    J, G = EnforcePrescribed(network=network, bc=bc, A=J, x=x, b=G, type='Jacobian')
    for i in range(max_iter):
        last_iter = i
        dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
        x = x + dx
        G = J * x - ddt * x_old
        J, G = EnforcePrescribed(network=network, bc=bc, A=J, x=x, b=G, type='Jacobian')
        G_norm = np.linalg.norm(np.abs(G), ord=2)
        if G_norm < tol:
            break
    if last_iter == max_iter - 1:
        print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

    ana_sol = AnalyticalSolution(time, 1-zeta)
    err = ana_sol - x
    print(f'{t}/{len(tsteps)} - {time}: {last_iter + 1} it [{G_norm}] err [{np.max(np.abs(err))}]')
    sol[:, pos] = x.ravel()
    time += dt

print('finished')
