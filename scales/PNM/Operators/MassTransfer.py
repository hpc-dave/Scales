
import openpnm as op
import scipy.linalg
import scipy.sparse
import spheres_and_cylinders as geo_model
from matplotlib import pyplot as plt
import numpy as np
import scipy
from OP_operators import construct_grad, construct_div, construct_ddt, ApplyBC, EnforcePrescribed
import time

Nx = 10
Ny = 100000
Nz = 1
dx = 1./Nx

# get network
network = op.network.Cubic([Nx, Ny, Nz], spacing=dx)

# add geometry

geo = geo_model.spheres_and_cylinders
network.add_model_collection(geo, domain='all')
network.regenerate_models()

c = np.zeros((network.Np, 1))
c_old = c.copy()
bc = {}
bc['left'] = {'prescribed': 1.}
bc['right'] = {'prescribed': 0.}

x = np.ndarray.flatten(c).reshape((c.size, 1))
dx = np.zeros_like(x)

grad = construct_grad(network=network)
div = construct_div(network=network)
ddt = construct_ddt(network=network, dt=0.01)
D = np.ones((network.Nt, 1), dtype=float)

J = ddt - div(D, grad)
for i in range(10):
    x_old = x.copy()
    J, b = ApplyBC(network=network, bc=bc, A=J)

    G = J * x - ddt * x_old - b

    tic = time.perf_counter_ns()
    J, G = EnforcePrescribed(network=network, bc=bc, A=J, x=x, b=G)
    toc = time.perf_counter_ns()
    print(f'elapsed time: {toc-tic} ns')
    dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
    x = x + dx

# define a phase
phase = op.phase.Air(network=network)

# add physics model
phys = op.models.collections.physics.basic
del phys['throat.entry_pressure']
phase.add_model_collection(phys)
phase.regenerate_models()

# define algorithm
# alg = op.algorithms.FickianDiffusion(network=network, phase=phase)
alg = op.algorithms.TransientFickianDiffusion(network=network, phase=phase)

# define BC
inlet = network.pores('left')
outlet = network.pores('right')
c_in, c_out = [1, 0]
alg.set_value_BC(pores=inlet, values=c_in)
alg.set_value_BC(pores=outlet, values=c_out)

x0 = np.zeros_like(network['pore.diameter'])
alg.run(x0=x0, tspan=[0, 1])

c_pore = alg['pore.concentration']
c_throat = alg.interpolate_data(propname='throat.concentration')
d = network['pore.diameter']
fix, ax = plt.subplots(figsize=[5, 5])
op.visualization.plot_coordinates(network=network, color_by=c_pore, size_by=d, markersize=400, ax=ax)

print('finished')
