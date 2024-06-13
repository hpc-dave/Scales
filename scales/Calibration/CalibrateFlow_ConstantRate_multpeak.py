from pnm_mctools import MulticomponentTools
from pnm_mctools.calibrated_conductance import GetConductanceObject
import openpnm as op
import numpy as np
from BruteForceSweep import Sweeper
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
parser.add_argument('-n', '--num_proc', type=int, help='number of parallel workers')
parser.add_argument('-p', '--parallel', action='store_true', help='activates parallelism and set number of workers automatically')
parser.add_argument('-s', '--samples', type=int, help='number of samples to use')
parser.add_argument('-d', '--deviation', type=float, help='standard deviation of the noise')

parallel_processing = None

args = parser.parse_args()
num_proc = args.num_proc
parallel_processing = args.parallel
num_samples = args.samples
if num_proc is not None:
    if num_proc < 1:
        raise ValueError(f'Parallelization specifier has to be > 0, received {args.parallel}!')
elif parallel_processing:
    num_proc = cpu_count()
else:
    num_proc = 1

F_range = [0.9, 1.1]
m_range = [0.9, 1.1]
n_range = [0.9, 1.1]

F_res = 0.01
m_res = 0.01
n_res = 0.01

# dummy peak
flow_rates_list = [1e-5, 1e-4, 5e-5, 2.5e-5, 7.5e-5]     # in m^3/s
run_times_list  = [1e04, 1e03, 5e03, 7.5e03, 2.5e03]         # in s    # noqa: E221

if num_samples is None:
    num_samples = len(flow_rates_list)
else:
    if num_samples > len(flow_rates_list):
        raise ValueError('the number of samples is too large')
    elif num_samples < 1:
        raise ValueError('at least one sample has to be included')

flow_rates = [flow_rates_list[n] for n in range(num_samples)]
run_times = [run_times_list[n] for n in range(num_samples)]

num_tsteps = 1000

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

# test configuration
num_flow_rates = len(flow_rates)
time_exp = {}
peak_exp = {}

analytical_solution = [1., 1., 1.]


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
if args.deviation is not None:
    for e in peak_exp:
        e += np.random.normal(0, 1e-5, e.size)
        e *= np.random.normal(1, args.deviation, e.size)
else:
    print('No standard deviation specified')


def fitness_func(sw, solution):
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


sw = Sweeper(parameter_range=[F_range, m_range, n_range],
             resolution=[F_res, m_res, n_res],
             fitness_func=fitness_func,
             parallelism=num_proc)

sw.run()

best_fit, best_fitness = sw.BestFit()

# evaluate and print result
print(f'best solution: {best_fit} with {best_fitness} after {sw.ElapsedTime():1.1f} s')

P_best = [None] * num_flow_rates
J_h_best = [None] * num_flow_rates
p_inlet = network.pores('left')
P_err = [None] * num_flow_rates
P_err_rel = [None] * num_flow_rates
for i in range(num_flow_rates):
    J_h_best, P_best = ComputeFlow(flow_rates[i], *best_fit)
    J_h_best, P_ana = ComputeFlow(flow_rates[i], *analytical_solution)
    P_ana = np.average(P_ana)
    P_best = np.average(P_best)
    P_err[i] = np.sqrt((P_ana-P_best)**2)
    P_err_rel[i] = P_err[i]/P_ana

print("Best solution:", best_fit)
print("Best solution fitness:", best_fitness)
print(f'Error in dP: {P_err} Pa - {P_err_rel}')

print('finished')
