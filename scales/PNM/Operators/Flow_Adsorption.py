import openpnm as op
import scipy.linalg
import scipy.sparse
import spheres_and_cylinders as geo_model
import numpy as np
import scipy
import math
from matplotlib import pyplot as plt
from OP_operators import MulticomponentTools

Nx = 10
Ny = 2
Nz = 1
Nc = 2  # fluid concentration + adsorbed species
spacing = 1./Nx
a_v = 1  # m^2/m^3


def SphereSurface(target: any, diameter: str):
    return math.pi * target[diameter]/6


def SphereSpecificSurface(target: any, diameter: str):
    return 6. / target[diameter]


# get network
network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)

# add geometry
geo = geo_model.spheres_and_cylinders
geo['pore.a_V'] = {'model': SphereSpecificSurface, 'diameter': 'pore.diameter'}
network.add_model_collection(geo, domain='all')
network.regenerate_models()

# flow properties
water = op.phase.Water(network=network)
water.add_model(propname='throat.hydraulic_conductance',
                model=op.models.physics.hydraulic_conductance.generic_hydraulic)

sf = op.algorithms.StokesFlow(network=network, phase=water)
sf.set_value_BC(pores=network.pores('left'), values=1.1e5)
sf.set_value_BC(pores=network.pores('right'), values=1e5)
sf.run()

c = np.zeros((network.Np, Nc))
c_old = c.copy()
bc_0 = {}
bc_0['left'] = {'prescribed': 1.}
bc_0['right'] = {'outflow': None}
bc_1 = {}
bc = [bc_0, bc_1]

mt = MulticomponentTools(network=network, bc=bc, num_components=Nc)

x = np.ndarray.flatten(c).reshape((c.size, 1))
dx = np.zeros_like(x)

dt = 0.01
tsteps = [t for t in range(1, int(1./dt))]
tol = 1e-6
max_iter = 10
time = dt

# need to provide div with some weights, namely an area
# that the flux acts on
A_flux = np.full((network.Nt, 1), dtype=float, fill_value=network['pore.volume'][0]/spacing)
A_pore = np.full((network.Np, 1), fill_value=a_v)
fluid_flux = sf.rate(throats=network.throats('all'), mode='single')*0.01
grad = mt.Gradient(include=0)
c_up = mt.Upwind(fluxes=fluid_flux, include=0)
div = mt.Divergence(include=0)
ddt = mt.DDT(dt=dt)

k_reac = 1.
k_ads = 0.1
r_ads_0 = np.zeros((network.Np, 2), dtype=float)
r_ads_1 = np.zeros_like(r_ads_0)
r_ads_0[:, 0], r_ads_0[:, 1] = k_reac * k_ads * network['pore.a_V'], -k_reac * network['pore.a_V']
r_ads_1[:, 0], r_ads_1[:, 1] = -k_reac * k_ads, k_reac
r_ads_0 *= network['pore.volume'].reshape((-1, 1))
r_ads_1 *= network['pore.volume'].reshape((-1, 1))
row = np.arange(0, network.Np * Nc, Nc, dtype=int).reshape((-1, 1))
col = row.copy()
row = np.hstack((row, row, row+1, row+1)).flatten()
col = np.hstack((col, col+1, col, col+1)).flatten()
data = np.hstack((r_ads_0, r_ads_1)).flatten()
r_ads = scipy.sparse.coo_matrix((data, (row, col)), shape=(network.Np*Nc, network.Np*Nc))
r_ads = r_ads.tocsr()

D = np.full((network.Nt, 2), fill_value=1e-6, dtype=float)
D[:, 1] = 0.

conv = mt.Fluxes(fluid_flux, c_up)
diff = mt.Fluxes(D, A_flux, grad)
J = ddt + div(conv) - div(diff) + r_ads
J = mt.ApplyBC(A=J)

flux_in = 0
flux_out = 0

signal = np.zeros((len(tsteps)+1), dtype=float)
response = np.zeros((len(tsteps)+1), dtype=float)


inlet_pores = network.pores('left')
outlet_pores = network.pores('right')
pores_internal = network.pores(labels=['left', 'right'], mode='nor')
throats_in = network.find_neighbor_throats(pores=inlet_pores, mode='xor')
throats_out = network.find_neighbor_throats(pores=outlet_pores, mode='xor')
signal[0] = np.average(c[inlet_pores, 0])
response[0] = np.average(c[outlet_pores, 0])

for t in tsteps:
    if t == 5:
        bc[0]['left']['prescribed'] = 0.

    x_old = x.copy()
    G = J * x - ddt * x_old
    G = mt.ApplyBC(x=x, b=G, type='Defect')
    for i in range(max_iter):
        last_iter = i
        dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
        x = x + dx
        c = x.reshape(c.shape)
        G = J * x - ddt * x_old
        G = mt.ApplyBC(x=x, b=G, type='Defect')
        G_norm = np.linalg.norm(np.abs(G), ord=2)
        if G_norm < tol:
            break
    if last_iter == max_iter - 1:
        print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

    c = x.reshape((-1, Nc))
    flux_conv = conv * x    # mol/s
    flux_diff = -diff * x    # mol/s
    flux_tot = (flux_conv + flux_diff).reshape((-1, Nc))  # mol/s
    flux_in += np.sum(flux_tot[throats_in, :]) * dt    # mol
    flux_out += np.sum(flux_tot[throats_out, :]) * dt  # mol
    mass = c * network['pore.volume'].reshape((-1, 1))
    mass[:, 1] *= network['pore.a_V']
    mass_err = (flux_in - flux_out) / (np.sum(mass[pores_internal, :]))-1
    print(f'{t}/{len(tsteps)} - {time}: {last_iter + 1} it [Defect:{G_norm:1.2e} mass: {mass_err:1.2e}]')
    signal[t] = np.average(c[inlet_pores, 0])
    response[t] = np.average(c[outlet_pores, 0])
    time += dt

plt.plot(signal, '-', response, '.')
plt.show()
print('finished')
