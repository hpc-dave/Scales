import openpnm as op
import scipy.linalg
import scipy.sparse
import spheres_and_cylinders as geo_model
from calibrated_conductance import GetConductanceObject
import numpy as np
import scipy
import math
from matplotlib import pyplot as plt
from OP_operators import MulticomponentTools

Nx = 10
Ny = 10
Nz = 1
Nc = 1
spacing = 1./Nx
Pin = 1.1e5
Pout = 1e5


def SphereSpecificSurface(target: any, diameter: str):
    return 6. / target[diameter]


# get network
network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)

# add geometry
geo = geo_model.spheres_and_cylinders
network.add_model_collection(geo, domain='all')
network.regenerate_models()

# flow properties
water = op.phase.Water(network=network)
water.add_model(propname='throat.hydraulic_conductance',
                model=op.models.physics.hydraulic_conductance.generic_hydraulic)


sf = op.algorithms.StokesFlow(network=network, phase=water)
sf.set_value_BC(pores=network.pores('left'), values=Pin)
sf.set_value_BC(pores=network.pores('right'), values=Pout)
sf.run()

c = np.zeros((network.Np, Nc))
c_old = c.copy()
bc = {}
bc['left'] = {'prescribed': Pin}
bc['right'] = {'prescribed': Pout}

mt = MulticomponentTools(network=network, bc=bc, num_components=Nc)

x = np.ndarray.flatten(c).reshape((c.size, 1))
dx = np.zeros_like(x)

tol = 1e-20
max_iter = 10

# need to provide div with some weights, namely an area
# that the flux acts on
grad = mt.Gradient()
div = mt.Divergence()
Conductance = GetConductanceObject(network=network, F=1., m=1., n=1.)

mu = water.get_conduit_data('throat.viscosity')[:, 1]
rho = water.get_conduit_data('throat.density')[:, 1]
g = Conductance(throat_density=rho, throat_viscosity=mu)
# g = water.get_conduit_data('throat.hydraulic_conductance')[:, 1]
flux = mt.Fluxes(g, grad)
J = -div(flux)
J = mt.ApplyBC(A=J)

G = J * x
G = mt.ApplyBC(x=x, b=G, type='Defect')
for i in range(max_iter):
    last_iter = i
    dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
    x = x + dx
    c = x.reshape(c.shape)
    rate_prev = g * (grad * x)
    g = Conductance(throat_density=rho, throat_viscosity=mu, rate_prev=rate_prev)
    flux = mt.Fluxes(g, grad)
    J = -div(flux)
    J = mt.ApplyBC(A=J)
    G = J * x
    G = mt.ApplyBC(x=x, b=G, type='Defect')
    G_norm = np.linalg.norm(np.abs(G), ord=2)
    if G_norm < tol:
        break
if last_iter == max_iter - 1:
    print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

c = x.reshape((-1, Nc))

c_comp = sf.x.reshape((-1, Nc))

err = np.max(np.abs((c-c_comp)/Pin))

print(f'{last_iter + 1} it [Defect:{G_norm:1.2e} error:{err:1.2e}]')

print('finished')
