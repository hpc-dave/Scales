import openpnm as op
import scipy.linalg
import scipy.sparse
import spheres_and_cylinders as geo_model
import numpy as np
import scipy
from OP_operators import construct_grad, construct_div, construct_ddt, EnforcePrescribed, construct_upwind

Nx = 100
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
v = 0.1

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
# that the flux acts on
A_flux = np.zeros((network.Nt, 1), dtype=float) + network['pore.volume'][0]/spacing

grad = construct_grad(network=network)
c_up = construct_upwind(network=network, fluxes=v)
div = construct_div(network=network, weights=A_flux)
ddt = construct_ddt(network=network, dt=dt)

D = np.zeros((network.Nt, 1), dtype=float) + 1e-3
J = ddt - div(D, grad) + div(v, c_up)

J = EnforcePrescribed(network=network, bc=bc, A=J)
for t in tsteps:
    x_old = x.copy()
    pos += 1

    G = J * x - ddt * x_old
    G = EnforcePrescribed(network=network, bc=bc, x=x, b=G, type='Defect')
    for i in range(max_iter):
        last_iter = i
        dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
        x = x + dx
        G = J * x - ddt * x_old
        G = EnforcePrescribed(network=network, bc=bc, x=x, b=G, type='Defect')
        G_norm = np.linalg.norm(np.abs(G), ord=2)
        if G_norm < tol:
            break
    if last_iter == max_iter - 1:
        print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

    print(f'{t}/{len(tsteps)} - {time}: {last_iter + 1} it [{G_norm}]')
    sol[:, pos] = x.ravel()
    time += dt

print('finished')
