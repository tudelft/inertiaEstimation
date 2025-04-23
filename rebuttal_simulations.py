import time
tic = time.time()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
plt.close('all')
import pandas as pd
from tqdm import tqdm

import lib


#%% general setup, lengths/number of parallel bodies, real-world conditions

# random seed, used 42 for all sims in the paper
np.random.seed(42)

# number of parallel simulations. 1000 --> 2GB RAM or something like that, increase with care
N = 1000

# the following results in M = 20000 timesteps for the simulation, and 4000 for evaluation.
# do not increase, otherwise itll blow up your RAM
dt_sim  = 0.00005   # 20kHz, just for simulation
eval_every = 5      # 4kHz filtering/algo evaluation, just like on embedded
T = 1.0             # seconds simulation time

# real-world flywheel inertia (kgm^2)
iR_real = 1.2e-6

# observed reaction wheel speed measurement noise std: 5.7 rad/s @ 4000Hz
# this means  std / sqrt(fs) = 0.09103 rad/s / sqrt(Hz) noise density
wR_noise_density_real = 0.09103

# observed w measurement noise std: 0.0153 rad/s @ 4000Hz, so std / sqrt(fs) = 0.243e-3 rad/s / sqrt(Hz) noise density
# this was for x, for y and z it was higher somehow, up to 1e-3 rad/s / sqrt(Hz). Use mean 0.75e-3
# compare 0.75e-3 rad/s / sqrt(Hz) = 40mdps/sqrt(Hz) with datasheet 2.8mdps / sqrt(Hz)
w_noise_density_real = 0.75e-3

# choice of lowpass filter break frquency (Hz)
lp_real = 20.


#%% plot setup

from argparse import ArgumentParser
parser = ArgumentParser(description="Throw 2 inertia simulations")
parser.add_argument("plot_type", choices=[
    'initial_condition',
    'filtering',
    'body_type_1',
    'body_type_2',
], help="The type of plot to output")
parser.add_argument("--num", type=int, default=N, help="Number of simulations (5000 reasonable max)")
args = parser.parse_args()

resimulate = False

plot_type = args.plot_type

if plot_type == 'initial_condition':
    # plot_series = ['w0_x rad/s', 'w0_y rad/s', 'w0_z rad/s', '$\epsilon\ (\%)$', '$\Psi\ (°)$']
    # plot_series = ['w0_x rad/s', 'w0_y rad/s', 'w0_z rad/s', '$\epsilon\ (\%)$']
    plot_series = ['$\\omega_0\ (\\text{rad}\ s^{-1})$', '$\Psi\ (°)$', '$\epsilon\ (\%)$']
    # plot_series = ['w0 rad/s', '$\epsilon\ (\%)$']
    # plot_series = ['w0 rad/s', 'w0 intermed rad/s', '$\epsilon\ (\%)$', '$\Psi\ (°)$']
elif plot_type == 'filtering':
    # plot_series = ['$I_{R_{zz}}\ (kg\ mm^2)$', 'trace(I)', '$N_\\omega\ (\\text{mrad}\ s^{-1}\ \\text{Hz}^{-0.5})$', '$N_{\\omega_R}\ (\\text{rad}\ s^{-1}\ \\text{Hz}^{-0.5})$', '$f_{\\text{low-pass}}\ (Hz)$', '$\epsilon\ (\%)$']
    plot_series = ['$I_{R_{zz}}\ (kg\ mm^2)$', '$N_\\omega\ (\\text{mrad}\ s^{-1}\ \\text{Hz}^{-0.5})$', '$N_{\\omega_R}\ (\\text{rad}\ s^{-1}\ \\text{Hz}^{-0.5})$', '$f_{\\text{low-pass}}\ (Hz)$', '$\epsilon\ (\%)$']
elif plot_type == 'body_type_1':
    # plot_series = ['$\\min\\ \\Delta\\bar\\sigma\ (\%)$', '$\kappa(I)$', '$\epsilon\ (\%)$', '$\Psi\ (°)$']
    plot_series = ['$\\min\\ \\Delta\\bar\\sigma\ (\%)$', '$\epsilon\ (\%)$', '$\Psi\ (°)$']
elif plot_type == 'body_type_2':
    plot_series = ['$\kappa(I)$', '$\Psi\ (°)$', '$\epsilon\ (\%)$']

N = args.num


#%% conditions for throw/body

import scipy

iR = np.repeat(iR_real, N)
lp = np.repeat(lp_real, N)
axs = scipy.stats.special_ortho_group.rvs(dim=3, size=N)[:,np.newaxis,0]
w0_norm = np.random.uniform(low=2*np.pi, high=+6*np.pi, size=(N,1,1))    # initial w, or low=5
w0 = axs * w0_norm
w_noise_density = np.repeat(w_noise_density_real, N)
wR_noise_density = np.repeat(wR_noise_density_real, N)

tensor = np.zeros((N,3,3))
itensor = np.zeros((N,3,3))
for i in range(N):
    tensor[i] = lib.buildVeryPhysicalTensorTraceFixed(2e-3, 1., 5.)
    itensor[i] = np.linalg.inv(tensor[i])


# override defaults based on plot type
if plot_type == 'initial_condition':
    axs = scipy.stats.special_ortho_group.rvs(dim=3, size=N)[:,np.newaxis,0]
    w0_norm = np.random.uniform(low=0., high=+6*np.pi, size=(N,1,1))    # initial w, or low=5
    w0 = axs * w0_norm
elif plot_type == 'filtering':
    for i in range(N):
        tensor[i] = lib.buildVeryPhysicalTensorTraceFixed(2e-3, 2., 2.)
        itensor[i] = np.linalg.inv(tensor[i])
    iR = np.random.uniform(low=iR_real * 0.25, high=iR_real * 1.75, size=N)           # flywheel
    lp = np.random.uniform(low=10., high=20., size=N)           # flywheel
    w_noise_density = np.random.uniform(low=0., high=w_noise_density_real*2., size=N)
    wR_noise_density = np.random.uniform(low=0., high=wR_noise_density_real*2., size=N)
elif plot_type == 'body_type_1' or plot_type == 'body_type_2':
    for i in range(N):
        # tensor[i] = lib.buildVeryPhysicalTensor(-4, -3)
        tensor[i] = lib.buildVeryPhysicalTensorTraceFixed(2e-3, 1., 10.)
        itensor[i] = np.linalg.inv(tensor[i])




#%%

############################
### START OF SIMULATIONS ###
############################


#%% run motor simulation to get wR and wRdot timeseries for motors (same for all simulations)

t_sim = np.arange(0, T+dt_sim, dt_sim)
wR_sim = np.zeros_like(t_sim)
wRdot_sim = np.zeros_like(t_sim)
wRcs = lib.motorProfile(t_sim)  # commanded flywheel velocities

tau = 0.025 # motor time constant
for i, wRc in enumerate(wRcs[1:]):
    wRdot_sim[i] = lib.motorModel(wR_sim[i], wRc, tau)
    wR_sim[i+1] = wR_sim[i] + wRdot_sim[i] * dt_sim

M = len(t_sim)

toc = time.time()
print("")
print(f"# Finished setup in {toc - tic:.2f} seconds")
print()


#%% simulate 

tic = time.time()

omega_sim = np.zeros((M,N,1,3))
omega_sim[0] = w0
omega_dot_sim = np.zeros_like(omega_sim)
lib.simulateVectorized(tensor, itensor, iR, omega_sim, omega_dot_sim, wR_sim, wRdot_sim, dt_sim, M)

toc = time.time()
print(f"# Finished {N} vectorized simulation in {toc - tic:.2f} seconds (avg {(toc - tic) / (M*N) * 1e9:.0f}ns per timestep)")
print()


#%% run our algorithm (only for I, not for imu offset)

tic = time.time()

# get the samples to actually use
M_eval = int(np.ceil(M / eval_every))
dt = dt_sim * eval_every
t = t_sim[::eval_every].copy()
w = wR_sim[::eval_every].copy()
wdot = wRdot_sim[::eval_every].copy()
omega = omega_sim[::eval_every].copy()
omega_dot = omega_dot_sim[::eval_every].copy()

w_noisy = w.copy()
omega_noisy = omega.copy()
w_noisy = w_noisy[:,np.newaxis] + np.random.normal(0, wR_noise_density, size=(M_eval, N)) / np.sqrt(dt)
omega_noisy[:,:,0,0] += np.random.normal(0, w_noise_density, size=(M_eval, N)) / np.sqrt(dt)
omega_noisy[:,:,0,1] += np.random.normal(0, w_noise_density, size=(M_eval, N)) / np.sqrt(dt)
omega_noisy[:,:,0,2] += np.random.normal(0, w_noise_density, size=(M_eval, N)) / np.sqrt(dt)

tensor_est = np.zeros_like(tensor)
itensor_est = np.zeros_like(tensor)

eps = np.zeros((N,))
Psi = np.zeros((N,))
eval_err = np.zeros((N,1,3))
euler_err = np.zeros((N,1,3))
fit_errors = np.empty((M_eval,N,1,3))
fit_errors[:] = np.nan

doPrint = False
skipEdges = 1000 # around 200ms on each side
for i in tqdm(range(N), desc="Inertia estimation: "):
    filteredOmegas = lib.filterVectorSignalButterworth(omega_noisy[:,i,0], lp[i], dt)
    omegaDots = lib.differentiateVectorSignal(filteredOmegas, dt)
    flywheelOmegas = np.zeros_like(filteredOmegas)
    flywheelOmegas[:, 2] = w_noisy[:,i]
    filteredFlywheelOmegas = lib.filterVectorSignalButterworth(flywheelOmegas, lp[i], dt)
    flywheelOmegaDots = lib.differentiateVectorSignal(filteredFlywheelOmegas, dt)

    lib.Jflywheel = iR[i]

    I_test, residuals, fit_errors[skipEdges:-skipEdges,i,0,:] = lib.computeIVectorized(
                                     filteredOmegas[skipEdges:-skipEdges,:],
                                     omegaDots[skipEdges:-skipEdges,:],
                                     filteredFlywheelOmegas[skipEdges:-skipEdges,:],
                                     flywheelOmegaDots[skipEdges:-skipEdges,:])

    if doPrint:
        print("Ground T\n", tensor[i])
        print("Estimate\n", I_test)
        # print((tensor[i] - I_test) / tensor[i] * 100)
    eps[i], Psi[i], eval_err[i], euler_err[i] = lib.computeError(I_test, tensor[i], doPrint)

    tensor_est[i] = I_test.copy()
    itensor_est[i] = np.linalg.inv(I_test)


toc = time.time()
print(f"# Finished analysis in {toc - tic:.2f} seconds")

print()
print(f"# average error        {100*np.mean(eps):.2f}%. Max error {100*eps.max():.2f}%")
print(f"# average misalignment {180/np.pi*np.mean(Psi):.2f}°. Max error {180/np.pi*Psi.max():.2f}°")


#%% build pandas df for plotting/analysis

eigs, evecs = np.linalg.eig(tensor)
traceI = np.linalg.trace(tensor)
min_rel_sep = np.min(np.abs(eigs - eigs[:, [1,2,0]]) / np.maximum(eigs, eigs[:, [1,2,0]]), axis=1)
cond = np.linalg.cond(tensor)

U, S, VT = np.linalg.svd(tensor)

df = pd.DataFrame({
    '$I_{R_{zz}}\ (kg\ mm^2)$': iR * 1e6,
    '$f_{\\text{low-pass}}\ (Hz)$': lp,
    'w0_x rad/s': w0[:,0,0],
    'w0_y rad/s': w0[:,0,1],
    'w0_z rad/s': w0[:,0,2],
    '$\\omega_0\ (\\text{rad}\ s^{-1})$': w0_norm.squeeze(),
    'w0 intermed rad/s': (w0 @ U)[:, 0, 1],
    '$N_\\omega\ (\\text{mrad}\ s^{-1}\ \\text{Hz}^{-0.5})$': 1e3*w_noise_density,
    '$N_{\\omega_R}\ (\\text{rad}\ s^{-1}\ \\text{Hz}^{-0.5})$': wR_noise_density,
    'trace(I)': traceI,
    '$\\min\\ \\Delta\\bar\\sigma\ (\%)$': 100*min_rel_sep,
    '$\kappa(I)$': cond,
    '$\epsilon\ (\%)$': 100. * eps,
    '$\Psi\ (°)$': 180. / np.pi * Psi,
})

#%% pairplots for now

pg = sbs.pairplot(df[plot_series],
             corner=False,   # True: plots only lower triangle
             diag_kind='hist',  # histogram on diagonals
             plot_kws={"s": 6}, # markersize
             height=1.3, aspect=10/6
)

plt.suptitle("Pairplot of Parameters and Accuracy")
plt.tight_layout()
plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=None, hspace=None)
plt.grid()
#plt.show()

if plot_type == "initial_condition":
    p99 = df['$\epsilon\ (\%)$'][df['$\omega_0\ (\\text{rad}\ s^{-1})$'] > 2*np.pi].quantile(0.99)
    pg.axes[2,0].annotate(f"$P_{{99}}(\\epsilon \\mid \\omega_0 > 2\\pi) = {p99:.1f} \%$",
                          (5, 60),
                          fontsize=7)
elif plot_type == "filtering":
    toplim = 11
    # pg.axes[4,0].set_ylim(bottom=0, top=toplim)
    p99 = df['$\epsilon\ (\%)$'][df['$N_\\omega\ (\\text{mrad}\ s^{-1}\ \\text{Hz}^{-0.5})$'] < 1e3*w_noise_density_real].quantile(0.99)
    pg.axes[4,1].annotate(f"$P_{{99}}(\\epsilon \\mid N_\\omega < {1e3*w_noise_density_real}) = {p99:.1f} \%$",
                          (0, 9),
                          fontsize=7)
    pg.axes[4,0].plot((1e6*iR_real, 1e6*iR_real), (0, toplim), 'k--', lw=1.)
    pg.axes[4,1].plot((1e3*w_noise_density_real, 1e3*w_noise_density_real), (0, 7), 'k--', lw=1.)
    pg.axes[4,2].plot((wR_noise_density_real, wR_noise_density_real), (0, toplim), 'k--', lw=1.)
    pg.axes[4,3].plot((lp_real, lp_real), (0, toplim), 'k--', lw=1.)
elif plot_type == "body_type_1":
    p99 = df['$\Psi\ (°)$'][df['$\\min\\ \\Delta\\bar\\sigma\ (\%)$'] > 2.].quantile(0.99)
    pg.axes[2,0].annotate(f"$P_{{99}}(\\Psi \\mid \\min\\ \\Delta\\bar\\sigma\ > 2\%) = {p99:.1f}°$",
                          (5, 60),
                          fontsize=7)
elif plot_type == "body_type_2":
    p99 = df['$\epsilon\ (\%)$'][df['$\kappa(I)$'] < 5.].quantile(0.99)
    pg.axes[2,0].annotate(f"$P_{{99}}(\\epsilon \\mid \\text{{cond}}(I) < 5) = {p99:.1f} \%$",
                          (1.5, 40.),
                          fontsize=7)

plt.savefig(f"output/{args.plot_type}.pdf")


#%% Re-simulate with estimated tensor and plot

if resimulate:
    omega_sim_est = np.zeros_like(omega_sim)
    omega_dot_sim_est = np.zeros_like(omega_dot_sim)
    omega_sim_est[0, :] = w0
    lib.simulateVectorized(tensor_est, itensor_est, iR, omega_sim_est, omega_dot_sim_est, wR_sim, wRdot_sim, dt_sim, M)

    omega_est = omega_sim_est[::eval_every].copy()
    omega_dot_est = omega_dot_sim_est[::eval_every].copy()

    for i in tqdm(range(min(N, 10)), desc="Plotting: "):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex="col", gridspec_kw={'height_ratios': [2, 1, 1]})
        lib.timePlotVector(t, omega[:,i,0], ax=ax1, label="Ground truth", ylabel=r"${\omega}$ (rad s$^{-1}$)", linestyle="dashed", alpha=0.5)
        lib.timePlotVector(t, omega_noisy[:,i,0], ax=ax1, label="Noisy", ylabel=r"${\omega}$ (rad s$^{-1}$)", alpha=0.4)
        lib.timePlotVector(t, omega_est[:,i,0], ax=ax1, label="Estimated", alpha=1)

        ax2.plot(t * 1000, w, label=r"${\omega}_f$ (s$^{-1}$)", color="tab:blue")
        ax2.plot(t * 1000, w_noisy[:,i], label="Noisy", color="tab:blue", alpha=0.4)

        lib.timePlotVector(t, fit_errors[:,i], ax=ax3, label="Fit errors", ylabel=r"per-axis fit errors", alpha=0.5)

        ax1.grid()
        ax2.grid()
        #ax2.get_legend().remove()
        ax2.invert_yaxis()
        plt.tight_layout()
        # plt.show()
