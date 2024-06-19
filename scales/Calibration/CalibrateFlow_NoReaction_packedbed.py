from pnm_mctools import MulticomponentTools
from pnm_mctools.calibrated_conductance import GetConductanceObject
import openpnm as op
import numpy as np
from BruteForceSweep import Sweeper
from PrepareBed import ExtractPackingAlongAxis as EPAA
from pnm_mctools.IO import network_to_vtk
from ScaleNetwork import ScaleNetwork
import scipy
import math
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.helpers import cpu_count
import pandas as pd

#################
# User settings #
#################

parser = argparse.ArgumentParser(description='Constant rate with multiple peaks')
parser.add_argument('-n', '--num_proc', type=int, help='number of parallel workers')
parser.add_argument('-p', '--parallel', action='store_true', help='activates parallelism and set number of workers automatically')
parser.add_argument('-s', '--samples', type=int, help='number of samples to use')
parser.add_argument('-d', '--deviation', type=float, help='standard deviation of the noise')
parser.add_argument('-f', '--file', type=str, help='filename of the packing')
parser.add_argument('--file_type', type=str, help='type of the file with the packing, default is csv')
parser.add_argument('--file_summary', type=str, help='if specified, a CSV file of the same name will be provided with an overview')

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
filename = args.file
# filename='scales/Calibration/network_MB_R4.csv'
file_type = args.file_type
if filename is None:
    raise ValueError('No filename provided for reading in the packed bed')
if file_type is None:
    file_type = 'csv'
else:
    file_type = file_type.lower()
file_summary = args.file_summary


F_range = [0.9, 1.1]
m_range = [0.9, 1.1]
n_range = [0.9, 1.1]

F_res = 0.1
m_res = 0.01
n_res = 0.01

# dummy peak
flow_rates_list = [1e-8, 1e-4, 5e-5, 2.5e-5, 7.5e-5]     # in m^3/s
run_times_list  = [600, 1e02, 5e02, 7.5e02, 2.5e02]         # in s    # noqa: E221

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


# read in network
if file_type == 'csv':
    network = op.io.network_from_csv(filename)
else:
    raise ValueError(f'Unknown file type: {file_type}')

if 'throat.diameter' not in network:
    network['throat.diameter'] = network['throat.radius'] * 2.
if 'pore.diameter' not in network:
    network['pore.diameter'] = network['pore.radius'] * 2.

network = ScaleNetwork(network=network, scale=4e-5)

range_low = 1e-3
range_high = 0.2
network.set_label(label='inlets', mode='purge')
network.set_label(label='outlets', mode='purge')
network = EPAA(network=network, axis=0, labels=['inlet_bed', 'outlet_bed'], range=[range_low, range_high])

# we want to get te following network
# P0 - T0 - P1 - T1 - P2 - T2 - Bed
# with:
# P0: pore where tracer is injected, representing half of upstream tube
# T0: throat representing the flux in the upstream tube
# P1: pore representing the one half of the glass wool and half of tube
# T1: throat representing the transport through the wool
# P2: reservoir pore representing second half of glass wool
# T2: connecting the glass wool to the bed (should pose extremely low resistance)

r_tube = 0.002
l_wool = 0.01
V_wool = r_tube**2 * math.pi * l_wool
l_tube = 0.1
V_tube = r_tube**2 * math.pi * l_tube
inlet_throats = {}
mu_f = 0.001  # viscosity water in Pas
g_tube = math.pi * r_tube**4/(8. * mu_f)
# add P2 and T2
coord_P = np.average(network['pore.coords'], axis=0).reshape((1, -1))
coord_P[0, 0] = 0.
p_label = network.Np
conns = np.zeros((network.pores('inlet_bed').size, 2), dtype=int)
conns[:, 0] = network.pores('inlet_bed')
conns[:, 1] = p_label
op.topotools.extend(network=network, coords=coord_P, conns=conns, labels=['outlet_wool'])
t_label = network.find_neighbor_throats(pores=[p_label])
network['pore.volume'][p_label] = V_wool*0.5
network['pore.diameter'][p_label] = r_tube * 2
network['throat.diameter'][t_label] = r_tube * 0.1  # should be a bit smaller, otherwise the conductances turn to shit
network['throat.length'][t_label] = 1e-6
inlet_throats['wool_to_bed'] = t_label

# add P1 and T1
coord_P[0, 0] = -l_wool
p_label = network.Np
conns = np.zeros((network.pores('outlet_wool').size, 2), dtype=int)
conns[:, 0] = network.pores('outlet_wool')
conns[:, 1] = p_label
op.topotools.extend(network=network, coords=coord_P, conns=conns, labels=['inlet_wool'])
t_label = network.find_neighbor_throats(pores=[p_label])
network['pore.volume'][p_label] = V_wool*0.5 + V_tube * 0.5
network['pore.diameter'][p_label] = r_tube * 2
network['throat.diameter'][t_label] = r_tube * 2
network['throat.length'][t_label] = l_wool
inlet_throats['wool'] = t_label


# add P0 and T0
coord_P[0, 0] = -l_wool - l_tube
p_label = network.Np
conns = np.zeros((network.pores('inlet_wool').size, 2), dtype=int)
conns[:, 0] = network.pores('inlet_wool')
conns[:, 1] = p_label
op.topotools.extend(network=network, coords=coord_P, conns=conns, labels=['inlet_tracer'])
t_label = network.find_neighbor_throats(pores=[p_label])
network['pore.volume'][p_label] = V_tube * 0.5
network['pore.diameter'][p_label] = r_tube * 2
network['throat.diameter'][t_label] = r_tube * 2
network['throat.length'][t_label] = l_tube
inlet_throats['tube'] = t_label

Nc = 1

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
    bc_flow['inlet_tracer'] = {'rate': rate}
    bc_flow['outlet_bed'] = {'prescribed': P_out}
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
    # g[inlet_throats['wool_to_bed']] = 1.  # extremely high conductance, just for distribution
    g[inlet_throats['tube']] = g_tube
    g[inlet_throats['wool']] = g_tube * 1e-2

    if np.any(np.isnan(g)):
        return np.full_like(g, fill_value=-1.), 0., False
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
        # g[inlet_throats['wool_to_bed']] = 1.  # extremely high conductance, just for distribution
        g[inlet_throats['tube']] = g_tube
        g[inlet_throats['wool']] = g_tube * 1e-2
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
    # network_to_vtk(network, filename='P_orig.vtk', additional_data={'P': P, 'J_h': J_h})
    return J_h, P, last_iter < max_iter


def ComputeMassTransport(props, F, J_h):

    # boundary conditions for species
    bc_mass = {}
    # bc_mass['inlet'] = {'prescribed': c_peak}
    bc_mass['outlet_bed'] = {'outflow'}

    # compute mass transport based on the flow field
    c_init = 0   # mol/m^3
    t_end = props['t_end']
    dt = props['dt']
    # t_switch = props['t_switch']

    c = np.full((network.Np, Nc), fill_value=c_init, dtype=float)
    c[network.pores('inlet_tracer'), 0] = 1.

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
    pores_out = network.pores('outlet_bed')
    peak_num = np.zeros((num_tsteps_l), dtype=float)
    current_time = 0.

    for n in range(1, num_tsteps_l):
        current_time += dt
        x_old = x.copy()

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
        # if n % 10 == 0:
        #     network_to_vtk(network, filename='C_orig_' + str(n) + '.vtk', additional_data={'C': c}, fill_nans=-1.)
        peak_num[n] = np.average(c[pores_out, 0])

    return peak_num, c


def ComputePeak(props, parameters):
    F, m, n = parameters
    J_h, P, success = ComputeFlow(props['rate'], F, m, n)
    if not success:
        return np.full_like(peak_exp[0], fill_value=math.inf)
    peak, C = ComputeMassTransport(props, F, J_h)
    return peak


def ComputeAllPeaks(parameters):
    peak_num = [None]*num_flow_rates
    for i in range(num_flow_rates):
        props = {}
        props['rate'] = flow_rates[i]
        props['t_end'] = run_times[i]
        props['dt'] = props['t_end'] / num_tsteps
        peak_num[i] = (ComputePeak(props, parameters))
    return peak_num


peak_exp = ComputeAllPeaks(analytical_solution)
if args.deviation is not None:
    for e in peak_exp:
        e += np.random.normal(0, 1e-5, e.size)
        e *= np.random.normal(1, args.deviation, e.size)
else:
    print('No standard deviation specified')
for e in peak_exp:
    e += np.random.normal(0, 1e-5, e.size)
    e *= np.random.normal(1, 1e-2, e.size)


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
    return fitness, peaks[0]


sw = Sweeper(parameter_range=[F_range, m_range, n_range],
             resolution=[F_res, m_res, n_res],
             fitness_func=fitness_func,
             parallelism=num_proc,
             save_all_results=file_summary is not None)

sw.run()

best_fit, best_fitness = sw.BestFit()

if file_summary is not None:
    all_results = sw.AllResults()
    data_base = pd.DataFrame(columns=['F', 'm', 'n', 'fitness'])
    data_base['F'] = [r[0][0] for r in all_results]
    data_base['m'] = [r[0][1] for r in all_results]
    data_base['n'] = [r[0][2] for r in all_results]
    data_base['fitness'] = [r[1] for r in all_results]
    data_peak = [r[2][0] for r in all_results]
    for n in range(1, num_samples):
        data_peak += [r[2][n] for r in all_results]
    data_peak = pd.DataFrame(data=data_peak)

    data = pd.concat([data_base, data_peak], ignore_index=False, axis=1)
    data.to_csv(file_summary)

# evaluate and print result
print(f'best solution: {best_fit} with {best_fitness} after {sw.ElapsedTime():1.1f} s')

P_best = [None] * num_flow_rates
J_h_best = [None] * num_flow_rates
p_inlet = network.pores('inlet_tracer')
P_err = [None] * num_flow_rates
P_err_rel = [None] * num_flow_rates
for i in range(num_flow_rates):
    J_h_best, P_best, success = ComputeFlow(flow_rates[i], *best_fit)
    J_h_best, P_ana, success = ComputeFlow(flow_rates[i], *analytical_solution)
    P_ana = np.average(P_ana)
    P_best = np.average(P_best)
    P_err[i] = np.sqrt((P_ana-P_best)**2)
    P_err_rel[i] = P_err[i]/P_ana

print("Best solution:", best_fit)
print("Best solution fitness:", best_fitness)
print(f'Error in dP: {P_err} Pa - {P_err_rel}')

print('finished')
