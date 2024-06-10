from pnm_mctools import MulticomponentTools
from pnm_mctools.calibrated_conductance import GetConductanceObject
from pnm_mctools import spheres_and_cylinders as geo_model
import openpnm as op
import numpy as np
import random
import pygad
import scipy
import math
import matplotlib.pyplot as plt

#################
# User settings #
#################

# GA settings
num_generations = 1000
population_size = 50
num_parents_mating = 4
mutation_probability = 0.05

F_range = [0.5, 2.]
m_range = [0.5, 2.]
n_range = [0.5, 2.]

# dummy peak
peak_exp = {}
peak_exp['time'] = np.linspace(0, 500., 100)

# flow_rate = 0.001  # in m^3/s
delta_P = 250      # Pa
c_peak = 1.        # concentration in mol/m^3
D_bin = 1e-6       # binary diffusion coefficient in m^2/s

# define the porous network for computing the peak
Nx = 10
Ny = 10
Nz = 10
Nc = 1
spacing = 1./Nx

# generate the network
network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)
network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
network.regenerate_models()

network['throat.diameter'] /= F_range[1]

t_full = peak_exp['time'][-1]
dt = t_full/float(peak_exp['time'].size)

# define boundary conditions for the flow
P_out = 1e5
rate_in = 1e-3
bc_flow = {}
bc_flow['left'] = {'rate': rate_in}
bc_flow['right'] = {'prescribed': P_out}

# boundary conditions for species
bc_mass = {}
bc_mass['left'] = {'prescribed': c_peak}
bc_mass['right'] = {'outflow'}


# numerical details
tol_flow = 1e-8
tol_mass = 1e-6
max_iter = 100


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
    # Conductance = GetConductanceObject(network=network, F=F, m=0., n=0., C_0=0., E_0=0., gamma=0.)
    mu = phase.get_conduit_data('throat.viscosity')[:, 1]
    rho = phase.get_conduit_data('throat.density')[:, 1]
    g = Conductance(throat_density=rho, throat_viscosity=mu)
    if np.any(np.isnan(g)):
        return np.full_like(peak_exp['values'], fill_value=-1.), 0.
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
        if G_norm < tol_flow:
            break

    P = x.reshape((-1, 1))  # pressure at each pore
    J_h = -g * (grad * P)    # hyrdaulic rate at each throat
    if last_iter == max_iter - 1:
        J_h[:] = -1.
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
    peak_num = np.zeros_like(peak_exp['time'], dtype=float)
    for n in range(1, num_tsteps):
        x_old = x.copy()
        if n == 10:
            mt_m.SetBC(label='left', bc={'rate': 0.})
        G = J * x - ddt * x_old
        G = mt_m.ApplyBC(x=x, b=G, type='Defect')

        for i in range(10):
            dx = scipy.sparse.linalg.spsolve(J, -G).reshape((-1, 1))
            x += dx
            G = J * x - ddt * x_old
            G = mt_m.ApplyBC(x=x, b=G, type='Defect')
            G_norm = np.linalg.norm(np.abs(G), ord=2)
            if G_norm < tol_mass:
                break

        c = x.reshape((-1, Nc))
        peak_num[n] = np.average(c[pores_out, 0])

    return peak_num, c


def ComputePeak(parameters):
    F, m, n = parameters[0], 1., 1.
    J_h, P = ComputeFlow(F, m, n)
    peak, C = ComputeMassTransport(F, J_h)
    return peak


# test configuration
analytical_solution = [1., 1., 1.]
J_h, P = ComputeFlow(*analytical_solution)
peak_analytical, c_final = ComputeMassTransport(analytical_solution[0], J_h)
# peak_analytical = ComputePeak(analytical_solution)

peak_exp['values'] = peak_analytical

plt.plot(peak_analytical)
plt.show()
plt.pause(1)
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
num_genes = len(analytical_solution)
gene_space = [F_range, m_range, n_range]
initial_population = [[random.uniform(x[0], 2. if x[1] is None else x[1]) for x in gene_space] for _ in range(population_size)]

ga_instance = pygad.GA(initial_population=initial_population,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       mutation_probability=mutation_probability,
                       gene_type=float,
                       gene_space=gene_space,
                       allow_duplicate_genes=False,
                       suppress_warnings=True,
                       keep_elitism=25)
ga_instance.run()

best_solution, best_solution_fitness, best_index = ga_instance.best_solution()

J_h_best, P_best = ComputeFlow(*best_solution)

p_inlet = network.pores('left')
P_err = np.sqrt(np.sum((P[p_inlet] - P_best[p_inlet])**2)/p_inlet.size)

print("Best solution:", best_solution)
print("Best solution fitness:", best_solution_fitness)
print(f'Error in dP: {P_err} Pa')

print('finished')
