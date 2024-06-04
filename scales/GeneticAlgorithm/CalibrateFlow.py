from pnm_mctools import MulticomponentTools
from pnm_mctools.calibrated_conductance import GetConductanceObject
from pnm_mctools import spheres_and_cylinders as geo_model
import openpnm as op
import numpy as np
import random
import pygad
import scipy
import math

#################
# User settings #
#################

# GA settings
num_generations = 100
population_size = 50
num_parents_mating = 2
mutation_probability = 0.1

# dummy peak
peak_exp = {}
peak_exp['time'] = np.arange(0, 1000., dtype=float)
peak_exp['values'] = np.ones_like(peak_exp['time'])
peak_exp['values'][0:-50] = 0

flow_rate = 0.001  # in m^3/s
c_peak = 1.        # concentration in mol/m^3
D_bin = 1e-6       # binary diffusion coefficient in m^2/s

# define the porous network for computing the peak
Nx = 10
Ny = 2
Nz = 2
Nc = 1
spacing = 0.001/Nx

# generate the network
network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)
network.add_model_collection(geo_model, domain='all')
network.regenerate_models()

# define boundary conditions for the flow
bc_flow = {}
bc_flow['left'] = {'rate': flow_rate}
bc_flow['right'] = {'prescribed': 1e5}

# boundary conditions for species
bc_mass = {}
bc_mass['left'] = {'prescribed': c_peak}
bc_mass['right'] = {'outflow'}


# numerical details
tol = 1e-6
max_iter = 10


def ComputeFlow(F, m, n):
    mt_f = MulticomponentTools(network=network, bc=bc_flow, num_components=1)
    phase = op.phase.Water(network=network)

    P = np.zeros((network.Np, 1))
    x = np.ndarray.flatten(P).reshape((-1, 1))
    dx = np.zeros_like(x)

    # compute the flow field with the calibrated conductance
    grad = mt_f.Gradient()
    div = mt_f.Divergence()
    Conductance = GetConductanceObject(network=network, F=F, m=m, n=n)
    mu = phase.get_conduit_data('throat.viscosity')[:, 1]
    rho = phase.get_conduit_data('throat.density')[:, 1]
    g = Conductance(throat_density=rho, throat_viscosity=mu)
    if np.any(np.isnan(g)):
        return np.full_like(peak_exp['values'], fill_value=-1.)
    flux = mt_f.Fluxes(g, grad)
    J = -div(flux)
    J = mt_f.ApplyBC(A=J)

    G = J * x
    G = mt_f.ApplyBC(x=x, b=G, type='Defect')
    for i in range(max_iter):
        last_iter = i
        dx[:] = scipy.sparse.linalg.spsolve(J, -G).reshape(dx.shape)
        x = x + dx
        P = x.reshape(P.shape)
        rate_prev = g * (grad * x)
        g = Conductance(throat_density=rho, throat_viscosity=mu, rate_prev=rate_prev)
        flux = mt_f.Fluxes(g, grad)
        J = -div(flux)
        J = mt_f.ApplyBC(A=J)
        G = J * x
        G = mt_f.ApplyBC(x=x, b=G, type='Defect')
        G_norm = np.linalg.norm(np.abs(G), ord=2)
        if G_norm < tol:
            break
    if last_iter == max_iter - 1:
        return np.full_like(peak_exp, fill_value=-1.)

    P = x.reshape((-1, 1))  # pressure at each pore
    J_h = -g * (grad * P)    # hyrdaulic rate at each throat
    return J_h, P


def ComputeMassTransport(F, J_h):
    # compute mass transport based on the flow field
    c_init = 0   # mol/m^3
    t_end = peak_exp['time'][-1]  # s
    dt = t_end / peak_exp['time'].size

    c = np.full((network.Np, Nc), fill_value=c_init, dtype=float)

    mt_m = MulticomponentTools(network=network, bc=bc_mass, num_components=Nc)
    grad = mt_m.Gradient()
    c_up = mt_m.Upwind(fluxes=J_h)
    div = mt_m.Divergence()
    ddt = mt_m.DDT(dt=dt)
    D = np.full((network.Nt, 1), fill_value=D_bin)
    A_eff = math.pi * (network['throat.diameter'] * F)**2 * 0.25

    J = ddt + div(J_h, c_up) - div(A_eff, D, grad)
    J = mt_m.ApplyBC(A=J)
    x = c.reshape((-1, 1))
    num_tsteps = int(t_end / dt)
    pores_out = network.pores('right')
    peak_num = np.zeros_like(peak_exp['values'])
    for n in range(1, num_tsteps):
        x_old = x.copy()
        G = J * x - ddt * x_old
        G = mt_m.ApplyBC(x=x, b=G, type='Defect')

        for i in range(10):
            dx = scipy.sparse.linalg.spsolve(J, -G).reshape((-1, 1))
            x += dx
            G = J * x - ddt * x_old
            G = mt_m.ApplyBC(x=x, b=G, type='Defect')
            G_norm = np.linalg.norm(np.abs(G), ord=2)
            if G_norm < tol:
                break

        c = x.reshape((-1, Nc))
        peak_num[n] = np.average(c[pores_out, 0])

    return peak_num, c


def ComputePeak(parameters):
    F, m, n = parameters
    J_h, P = ComputeFlow(F, m, n)
    peak, C = ComputeMassTransport(F, J_h)
    return peak


def fitness_func(ga, solution, solution_idx):
    try:
        peak_num = ComputePeak(solution)
    except scipy.linalg.LinAlgError:
        # the computation failed, so fitness is set to 0
        return 0.
    err = (peak_num - peak_exp['values'])**2
    fitness = 1.0 / np.sum(err)
    return fitness

# F, m, n
parameters = [1., 1., 1.]
num_genes = len(parameters)
gene_space = [[0.5, 2.5], [0.5, 2.5], [0.5, 5.]]
initial_population = [[random.uniform(x[0], 2. if x[1] is None else x[1] ) for x in gene_space] for _ in range(population_size)]
ga_instance = pygad.GA(initial_population=initial_population,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       mutation_probability=mutation_probability,
                       gene_type=float,
                       gene_space=gene_space,
                       allow_duplicate_genes=False,
                       suppress_warnings=True)
ga_instance.run()

best_solution, best_solution_fitness, best_index = ga_instance.best_solution()

print("Best solution:", best_solution)
print("Best solution fitness:", best_solution_fitness)

print('finished')
