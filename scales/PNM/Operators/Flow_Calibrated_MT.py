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
Ny = 1
Nz = 1
Nc = 1
spacing = 1./Nx
Pin = 1.1e5
Pout = 1e5

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

P = np.zeros((network.Np, Nc))
P_old = P.copy()
bc = {}
bc['left'] = {'prescribed': Pin}
bc['right'] = {'prescribed': Pout}

flow_tools = MulticomponentTools(network=network, bc=bc, num_components=Nc)

x = np.ndarray.flatten(P).reshape((-1, 1))
dx = np.zeros_like(x)

tol = 1e-6
max_iter = 10

###################
# SOLVE FLOWFIELD #
###################

# need to provide div with some weights, namely an area
# that the flux acts on
grad = flow_tools.Gradient()
div = flow_tools.Divergence()
Conductance = GetConductanceObject(network=network, F=1., m=1., n=1.)

mu = water.get_conduit_data('throat.viscosity')[:, 1]
rho = water.get_conduit_data('throat.density')[:, 1]
g = Conductance(throat_density=rho, throat_viscosity=mu)
flux = flow_tools.Fluxes(g, grad)
J = -div(flux)
J = flow_tools.ApplyBC(A=J)

G = J * x
G = flow_tools.ApplyBC(x=x, b=G, type='Defect')
for i in range(max_iter):
    last_iter = i
    dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
    x = x + dx
    P = x.reshape(P.shape)
    rate_prev = g * (grad * x)
    g = Conductance(throat_density=rho, throat_viscosity=mu, rate_prev=rate_prev)
    flux = flow_tools.Fluxes(g, grad)
    J = -div(flux)
    J = flow_tools.ApplyBC(A=J)
    G = J * x
    G = flow_tools.ApplyBC(x=x, b=G, type='Defect')
    G_norm = np.linalg.norm(np.abs(G), ord=2)
    if G_norm < tol:
        break
if last_iter == max_iter - 1:
    print(f'WARNING: the maximum iterations ({max_iter}) were reached!')

P = x.reshape((-1, Nc))
J_h = -g * (grad * x) * 0

print(f'{last_iter + 1} it [Defect:{G_norm:1.2e}]')

########################
# SOLVE MASS TRANSPORT #
########################

c_in = 1     # mol/m^3
c_init = 0   # mol/m^3
D_bin = 1e-6 # m^2/s
dt = 1e-0    # s
t_end = 1000 # s
Nc = 1
bc = {}
bc['left'] = {'prescribed': c_in}
bc['right'] = {'outflow': None}

c = np.full((network.Np, Nc), fill_value=c_init, dtype=float)

mt = MulticomponentTools(network=network, bc=bc)
grad = mt.Gradient()
c_up = mt.Upwind(fluxes=J_h)
div = mt.Divergence()
ddt = mt.DDT(dt=dt)
D = np.full((network.Nt, 1), fill_value=D_bin)
A_eff = math.pi * network['throat.diameter']**2 * 0.25

flux = mt.Fluxes(g, grad)
J = ddt + div(g, c_up) - div(A_eff, D, grad)
J = mt.ApplyBC(A=J)
x = c.reshape((-1, 1))

num_tsteps = int(t_end / dt)
for n in range(1, num_tsteps):
    x_old = x.copy()
    G = J * x - ddt * x_old
    G = mt.ApplyBC(x=x, b=G, type='Defect')

    for i in range(10):
        dx = scipy.sparse.linalg.spsolve(J, -G).reshape((-1, 1))
        x += dx
        G = J * x - ddt * x_old
        G = mt.ApplyBC(x=x, b=G, type='Defect')
        G_norm = np.linalg.norm(np.abs(G), ord=2)
        if G_norm < tol:
            break

    c = x.reshape((-1, Nc))

    print(f'tstep: {n} @ {n*dt} s - G [{G_norm:1.2e}]')



print('finished')
