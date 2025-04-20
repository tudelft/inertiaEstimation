import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm
plt.close('all')

tic = time.time()

#%% general setup, lengths/number of parallel bodies

# random seed, used 42 for all sims in the paper
np.random.seed(42)

# number of parallel simulations. 5000 --> 8GB RAM or something like that, increase with care
N = 5000

# the following results in M = 14001 timesteps for the simulation, and 2801 for evaluation.
# do not increase, otherwise itll blow up your RAM
dt_sim  = 0.00005   # 20kHz, just for simulation
eval_every = 5      # 4kHz filtering/algo evaluation, just like on embedded
T = 0.7             # seconds simulation time


#%% randomize conditions for throw/body

j = np.random.uniform(low=1e-6, high=1e-6, size=N)           # flywheel
lp = np.random.uniform(low=10., high=40., size=N)          # low pass
w0 = np.random.uniform(low=-10., high=+10., size=(N,1,3))    # initial w
wR_noise = np.random.uniform(low=0., high=50., size=N)
W_noise = np.random.uniform(low=0., high=0.02, size=N)
tensor = np.zeros((N,3,3))
itensor = np.zeros((N,3,3))
for i in range(N):
    tensor[i] = lib.buildVeryPhysicalTensor(-4, -2.5)
    itensor[i] = np.linalg.inv(tensor[i])

#%% run motor simulation to get wR and wRdot timeseries for motors (same for all simulations)

t_sim = np.arange(0, 0.7+dt_sim, dt_sim)
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


#%% simulate 

tic = time.time()

omega_sim = np.zeros((M,N,1,3))
omega_sim[0] = w0
omega_dot_sim = np.zeros_like(omega_sim)
lib.simulateVectorized(tensor, itensor, j, omega_sim, omega_dot_sim, wR_sim, wRdot_sim, dt_sim, M)

toc = time.time()
print(f"# Finished vectorized simulation in {toc - tic:.2f} seconds (avg {(toc - tic) / (M*N) * 1e9:.0f}ns per timestep)")


#%% run our algorithm (only for I, not for imu offset)

# get the once to use in 
M_eval = int(np.ceil(M / eval_every))
dt = dt_sim * eval_every
t = t_sim[::eval_every].copy()
w = wR_sim[::eval_every].copy()
wdot = wRdot_sim[::eval_every].copy()
omega = omega_sim[::eval_every].copy()
omega_dot = omega_dot_sim[::eval_every].copy()

w_noisy = w.copy()
omega_noisy = omega.copy()
w_noisy = w_noisy[:,np.newaxis] + np.random.normal(0, wR_noise, size=(M_eval, N))
omega_noisy[:,:,0,0] += np.random.normal(0, W_noise, size=(M_eval, N))
omega_noisy[:,:,0,1] += np.random.normal(0, W_noise, size=(M_eval, N))
omega_noisy[:,:,0,2] += np.random.normal(0, W_noise, size=(M_eval, N))

tensor_est = np.zeros_like(tensor)
itensor_est = np.zeros_like(tensor)

eps = np.zeros((N,))
Psi = np.zeros((N,))
eval_err = np.zeros((N,1,3))
euler_err = np.zeros((N,1,3))
fit_errors = np.empty((M_eval,N,1,3))
fit_errors[:] = np.nan

doPrint = False
skipEdges = 500 # around 125ms on each side
for i in tqdm(range(N), desc="Inertia estimation: "):
    filteredOmegas = lib.filterVectorSignalButterworth(omega_noisy[:,i,0], lp[0], dt)
    # filteredOmegas = omega[:,i,0].copy()
    omegaDots = lib.differentiateVectorSignal(filteredOmegas, dt)
    flywheelOmegas = np.zeros_like(filteredOmegas)
    flywheelOmegas[:, 2] = w_noisy[:,i]
    filteredFlywheelOmegas = lib.filterVectorSignalButterworth(flywheelOmegas, lp[0], dt)
    # filteredFlywheelOmegas = flywheelOmegas.copy()
    flywheelOmegaDots = lib.differentiateVectorSignal(filteredFlywheelOmegas, dt)

    lib.Jflywheel = j[i]

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

print(f"average error        {100*np.mean(eps):.2f}%. Max error {100*eps.max():.2f}%")
print(f"average misalignment {180/np.pi*np.mean(Psi):.2f}°. Max error {180/np.pi*Psi.max():.2f}°")


#%% build pandas df for plotting/analysis

import seaborn as sbs

df = pd.DataFrame({
    'j': j,
    'low pass Hz': lp,
    'w0x': w0[:,0,0],
    'w0y': w0[:,0,1],
    'w0z': w0[:,0,2],
    'wR_noise': wR_noise,
    'w_noise': W_noise,
    'trI': np.linalg.trace(tensor),
    'eps': eps,
    'Psi': Psi,
})

sbs.pairplot(df[['wR_noise', 'w_noise', 'low pass Hz', 'w0x', 'w0y', 'w0z', 'trI', 'eps']],
             corner=False,   # plots only lower triangle
             diag_kind='hist',)  # kernel density on diagonals
             #plot_kws={'alpha': 0.3, 's': 10})  # tweak appearance

plt.suptitle("Pairplot of Parameters and Accuracy", y=1.02)
plt.tight_layout()
plt.show()

import sys
sys.exit()

#%% Re-simulate with estimated tensor and plot

omega_sim_est = np.zeros_like(omega_sim)
omega_dot_sim_est = np.zeros_like(omega_dot_sim)
omega_sim_est[0, :] = w0
lib.simulateVectorized(tensor_est, itensor_est, j, omega_sim_est, omega_dot_sim_est, wR_sim, wRdot_sim, dt_sim, M)

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
