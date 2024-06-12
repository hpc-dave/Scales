from pnm_mctools import MulticomponentTools
from pnm_mctools.calibrated_conductance import GetConductanceObject
import openpnm as op
import numpy as np
import random
import pygad
import scipy
import math
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.helpers import cpu_count
import time

#################
# User settings #
#################

parser = argparse.ArgumentParser(description='Constant rate with multiple peaks')
parser.add_argument('-g', '--num_generations', type=int, help='Number of generations (int)')
parser.add_argument('-p', '--population_size', type=int, help='Population size (int)')
parser.add_argument('-m', '--num_parents_mating', type=int, help='Number of mating parents (int)')
parser.add_argument('-b', '--mutation_probability', type=float, help='mutation probability')
parser.add_argument('--parallel', type=int, help='number of parallel processes')

# GA settings
num_generations = 100
population_size = 500
num_parents_mating = 4
mutation_probability = 0.05

parallel_processing = None

if __name__ == '__main__':
    args = parser.parse_args()
    num_generations = args.num_generations if args.num_generations else num_generations
    population_size = args.population_size if args.population_size else population_size
    num_parents_mating = args.num_parents_mating if args.num_parents_mating else num_parents_mating
    mutation_probability = args.mutation_probability if args.mutation_probability else mutation_probability
    if args.parallel:
        if args.parallel < 1:
            raise ValueError(f'Parallelization specifier has to be > 0, received {args.parallel}!')
        elif args.parallel > 1:
            parallel_processing = ['process', args.parallel]

# print arguments
print(f'Generations:              {num_generations}')
print(f'Population size:          {population_size}')
print(f'Number of mating parents: {num_parents_mating}')
print(f'Mutation probability:     {mutation_probability}')
print(f'Parallel processing:      {0 if parallel_processing is None else parallel_processing[1]}')

F_range = [0.5, 1.5]
m_range = [0.5, 1.5]
n_range = [0.5, 1.5]

F_res = 0.01
m_res = 0.1
n_res = 0.1

# dummy peak
flow_rates = [1e-5, 5e-5, 1e-4]     # in m^3/s
run_times = [1e4, 5e3, 1e3]         # in s
num_tsteps = 100

# flow_rate = 0.001  # in m^3/s
delta_P = 250      # Pa
c_peak = 1.        # concentration in mol/m^3
D_bin = 1e-6       # binary diffusion coefficient in m^2/s

# define the porous network for computing the peak
Nx = 100
Ny = 3
Nz = 3
Nc = 1
spacing = 10./Nx

# generate the network
network = op.network.Cubic([Nx, Ny, Nz], spacing=spacing)
network.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
network.regenerate_models()

network['throat.diameter'] /= F_range[1]

# numerical details
tol_flow = 1e-8
tol_mass = 1e-6
max_iter = 100


def ComputeFlow(rate, F, m, n):

    # define boundary conditions for the flow
    P_out = 1e5
    bc_flow = {}
    bc_flow['left'] = {'rate': rate}
    bc_flow['right'] = {'prescribed': P_out}
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
        return np.full_like(peak_exp[0], fill_value=-1.), 0.
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


def ComputeMassTransport(props, F, J_h):

    # boundary conditions for species
    bc_mass = {}
    bc_mass['left'] = {'prescribed': c_peak}
    bc_mass['right'] = {'outflow'}

    # compute mass transport based on the flow field
    c_init = 0   # mol/m^3
    t_end = props['t_end']
    dt = props['dt']
    t_switch = props['t_switch']

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
    num_tsteps_l = int(t_end / dt)
    pores_out = network.pores('right')
    peak_num = np.zeros((num_tsteps_l), dtype=float)
    current_time = 0.
    is_switched = False
    for n in range(1, num_tsteps_l):
        current_time += dt
        x_old = x.copy()

        if current_time > t_switch and not is_switched:
            mt_m.SetBC(label='left', bc={'rate': 0.})
            J = ddt + div(J_h, c_up) - div(A_eff, D, grad)
            J = mt_m.ApplyBC(A=J)
            is_switched = True

        G = J * x - ddt * x_old
        G = mt_m.ApplyBC(x=x, b=G, type='Defect')

        for _ in range(max_iter):
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


def ComputePeak(props, parameters):
    F, m, n = parameters
    J_h, P = ComputeFlow(props['rate'], F, m, n)
    peak, C = ComputeMassTransport(props, F, J_h)
    return peak


# test configuration
num_flow_rates = len(flow_rates)
time_exp = {}
peak_exp = {}

analytical_solution = [1., 1., 1.]


def ComputeAllPeaks(parameters):
    peak_num = [None]*num_flow_rates
    for i in range(num_flow_rates):
        props = {}
        props['rate'] = flow_rates[i]
        props['t_end'] = run_times[i]
        props['dt'] = props['t_end'] / num_tsteps
        props['t_switch'] = 10 * props['dt']
        peak_num[i] = (ComputePeak(props, parameters))
    return peak_num


peak_exp = ComputeAllPeaks(analytical_solution)

# J_h, P = ComputeFlow(*analytical_solution)
# peak_analytical, c_final = ComputeMassTransport(analytical_solution[0], J_h)

# plt.plot(peak_analytical)
# plt.show()
# plt.pause(1)

def fitness_func(ga, solution, solution_idx):
    fitness = 1.
    try:
        peaks = ComputeAllPeaks(solution)
        for i in range(num_flow_rates):
            err = (peaks[i] - peak_exp[i])**2
            fitness *= 1.0 / np.sum(err)
    except scipy.linalg.LinAlgError:
        # the computation failed, so fitness is set to 0
        return 0.
    return fitness


def show_progress(ga):
    best_solution, best_solution_fitness, _ = ga.best_solution()
    best_sol = ''.join(f'{entry:1.3f} ' for entry in best_solution)
    best_sol_fit = f'{best_solution_fitness:1.2e}'
    print(f'Generation {ga.generations_completed}/{ga.num_generations} - best solution: {best_sol} ({best_sol_fit})')


num_Fval = int((F_range[1]-F_range[0])/F_res)
num_mval = int((m_range[1]-m_range[0])/m_res)
num_nval = int((n_range[1]-n_range[0])/n_res)
best_solution = [-1., -1., -1.]
best_solution_fitness = 0.

num_procs = parallel_processing[1] if parallel_processing is not None else cpu_count()
p_range = list(range(0, num_Fval, int(num_Fval/num_procs)))
p_range.append(num_Fval)
print(f'Generating worker pool with {num_procs} workers')
pool = Pool(num_procs)

def inner_loop(i: int):
    best_solution_l = [-1., -1., -1.]
    best_solution_fitness_l = 0.
    disable_tqdm = i != 0
    if not disable_tqdm:
        print('The progress bar only provides an indicator of the progress based on process 0!')
    num_Fval_l = p_range[i+1] - p_range[i]
    Fm_range = num_mval * num_Fval_l
    for Fm_i in tqdm(range(Fm_range), disable=disable_tqdm):
        F_i = int(Fm_i/num_mval)
        m_i = int(Fm_i - F_i * num_mval)
        F = (F_i + p_range[i]) * F_res + F_range[0]
        m = m_i * m_res + m_range[0]
        # print(f'F_i {F_i} -> F {F} | m_i {m_i} -> m {m}')
        for n_i in range(num_nval):
            n = n_i * n_res + n_range[0]
            fit_l = fitness_func(0, [F, m, n], 0)
            if fit_l > best_solution_fitness_l:
                best_solution_l = [F, m, n]
                best_solution_fitness_l = fit_l
    # for F_i in tqdm(range(p_range[i], p_range[i+1]), disable=disable_tqdm):
    #     F = F_i * F_res + F_range[0]
    #     for m_i in range(num_mval):
    #         m = m_i * m_res + m_range[0]
    #         for n_i in range(num_nval):
    #             n = n_i * n_res + n_range[0]
    #             fit_l = fitness_func(0, [F, m, n], 0)
    #             if fit_l > best_solution_fitness_l:
    #                 best_solution_l = [F, m, n]
    #                 best_solution_fitness_l = fit_l
    return (best_solution_l, best_solution_fitness_l)

tic = time.perf_counter()
result = pool.map(inner_loop, range(num_procs))
toc = time.perf_counter()

for r in result:
    if r[1] > best_solution_fitness:
        best_solution_fitness = r[1]
        best_solution = r[0]

print(f'best solution: {best_solution} with {best_solution_fitness} after {toc-tic} s')
# for F_i in tqdm(range(num_Fval)):
#     F = F_i * F_res + F_range[0]
#     is_new = False
#     for m_i in range(num_mval):
#         m = m_i * m_res + m_range[0]
#         for n_i in range(num_nval):
#             n = n_i * n_res + n_range[0]
#             fit_l = fitness_func(0, [F, m, n], 0)
#             if fit_l > best_solution_fitness:
#                 best_solution = [F, m, n]
#                 best_solution_fitness = fit_l
#                 is_new = True
#     print(f'Best parameter set: {best_solution} with fitness {best_solution_fitness} ' + 'New' if is_new else '')

# # F, m, n
# num_genes = len(analytical_solution)
# gene_space = [F_range, m_range, n_range]
# initial_population = [[random.uniform(x[0], 2. if x[1] is None else x[1]) for x in gene_space] for _ in range(population_size)]

# ga_instance = pygad.GA(initial_population=initial_population,
#                        num_generations=num_generations,
#                        num_parents_mating=num_parents_mating,
#                        fitness_func=fitness_func,
#                        mutation_probability=mutation_probability,
#                        gene_type=float,
#                        gene_space=gene_space,
#                        allow_duplicate_genes=False,
#                        suppress_warnings=True,
#                        parent_selection_type='rank',
#                        parallel_processing=parallel_processing,
#                        on_generation=show_progress,
#                        keep_elitism=25)
# ga_instance.run()

# # best_solution = ga_instance.best_solutions[-1]
# # best_solution_fitness = ga_instance.best_solutions_fitness[-1]
# best_solution, best_solution_fitness, best_index = ga_instance.best_solution()

P_best = [None] * num_flow_rates
J_h_best = [None] * num_flow_rates
p_inlet = network.pores('left')
P_err = [None] * num_flow_rates
for i in range(num_flow_rates):
    J_h_best, P_best = ComputeFlow(flow_rates[i], *best_solution)
    J_h_best, P_ana = ComputeFlow(flow_rates[i], *analytical_solution)
    P_err[i] = np.sqrt(np.sum((P_ana[p_inlet] - P_best[p_inlet])**2)/p_inlet.size)

print("Best solution:", best_solution)
print("Best solution fitness:", best_solution_fitness)
print(f'Error in dP: {P_err} Pa')

print('finished')
