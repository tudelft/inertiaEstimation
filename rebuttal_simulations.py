import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm
plt.close('all')

#%% simulation code

def motorModel(w, wc, tau):
    return (wc - w) / tau

def motorProfile(t):
    # simplistic off-idle-full-idle profile
    wc = np.zeros_like(t)
    wc[t > 0.1] = -250.
    wc[(t > 0.2) & (t < 0.4)] = -1500

    return wc

def simulateVectorized(I, iI, j, w, w_dot, wR, wR_dot, dt, steps):
    # w: body
    # wR: flywheel
    # 1. calculate I wdot = -w x (I w)  -  IR wRdot  -  w x (IR wR) 
    # 2. integrate wdot
    for i in tqdm(range(steps-1), desc="Simulation: "):
        Iw = w[i] @ I
        wxIw = np.cross(w[i], Iw)
        IRwR = j * wR[i]
        wxIRwR_x = w[i,:,0,1] * IRwR   # cross product optimzed for knowing xy = 0 fopr the flywheel
        wxIRwR_y = -w[i,:,0,0] * IRwR
        IRwRdot_z = j * wR_dot[i]
        rhs = -wxIw.copy()
        rhs[:,0,0] -= wxIRwR_x
        rhs[:,0,1] -= wxIRwR_y
        rhs[:,0,2] -= IRwRdot_z
        w_dot[i] = rhs @ iI
        w[i+1] = w[i] + dt * w_dot[i]


#%% run motor simulation to get wR and wRdot timeseries for motors (same for all simulations)

tic = time.time()

dt_sim  = 0.00005   # 20kHz, just for simulation
tau = 0.025 # motor time constant

t_sim = np.arange(0, 0.7+dt_sim, dt_sim)
wR_sim = np.zeros_like(t_sim)
wRdot_sim = np.zeros_like(t_sim)
wRcs = motorProfile(t_sim)  # commanded flywheel velocities
for i, wRc in enumerate(wRcs[1:]):
    wRdot_sim[i] = motorModel(wR_sim[i], wRc, tau)
    wR_sim[i+1] = wR_sim[i] + wRdot_sim[i] * dt_sim


#%% randomize conditions for throw/body

# number of parallel simulations. 1001 --> 2GB RAM or something like that
N = 1001
np.random.seed(42)

j = np.random.uniform(low=1e-6, high=1e-6, size=N)           # flywheel
lp = np.random.uniform(low=200., high=200., size=N)          # low pass
w0 = np.random.uniform(low=-10., high=+10., size=(N,1,3))    # initial w
w_noise = np.random.uniform(low=0., high=0., size=N)
omega_noise = np.random.uniform(low=0., high=0., size=N)
tensor = np.zeros((N,3,3))
itensor = np.zeros((N,3,3))
for i in range(N):
    tensor[i] = lib.buildVeryPhysicalTensor(-4, -2.5)
    itensor[i] = np.linalg.inv(tensor[i])

toc = time.time()

M = len(t_sim)
print("")
print(f"# Finished setup in {toc - tic:.2f} seconds")


#%% simulate 

tic = time.time()

omega_sim = np.zeros((M,N,1,3))
omega_sim[0] = w0
omega_dot_sim = np.zeros_like(omega_sim)
simulateVectorized(tensor, itensor, j, omega_sim, omega_dot_sim, wR_sim, wRdot_sim, dt_sim, M)

toc = time.time()
print(f"# Finished vectorized simulation in {toc - tic:.2f} seconds (avg {(toc - tic) / (M*N) * 1e9:.0f}ns per timestep)")


#%% run our algorithm (only for I, not for imu offset)

# get the once to use in 
eval_every = 5 # 4kHz just like on drone
dt = dt_sim * eval_every
t = t_sim[::eval_every].copy()
w = wR_sim[::eval_every].copy()
wdot = wRdot_sim[::eval_every].copy()
omega = omega_sim[::eval_every].copy()
omega_dot = omega_dot_sim[::eval_every].copy()

w_noisy = w.copy()
omega_noisy = omega.copy()
w_noisy = w_noisy[:,np.newaxis] + np.random.normal(0, w_noise)
omega_noisy[:,:,0,0] += np.random.normal(0, omega_noise)
omega_noisy[:,:,0,1] += np.random.normal(0, omega_noise)
omega_noisy[:,:,0,2] += np.random.normal(0, omega_noise)

tensor_est = np.zeros_like(tensor)
itensor_est = np.zeros_like(tensor)

doPrint = False
for i in tqdm(range(N), desc="Inertia estimation: "):
    # filteredOmegas = lib.filterVectorSignalButterworth(omega[:,i,0], lp[0], dt)
    filteredOmegas = omega[:,i,0].copy()
    omegaDots = lib.differentiateVectorSignal(filteredOmegas, dt)
    flywheelOmegas = np.zeros_like(filteredOmegas)
    flywheelOmegas[:, 2] = w
    # filteredFlywheelOmegas = lib.filterVectorSignalButterworth(flywheelOmegas, lp[0], dt)
    filteredFlywheelOmegas = flywheelOmegas.copy()
    flywheelOmegaDots = lib.differentiateVectorSignal(filteredFlywheelOmegas, dt)

    lib.Jflywheel = j[i]

    I_test, residuals = lib.computeI(filteredOmegas,
                                     omegaDots,
                                     filteredFlywheelOmegas,
                                     flywheelOmegaDots)

    if doPrint:
        print("Ground T\n", tensor[i])
        print("Estimate\n", I_test)
        # print((tensor[i] - I_test) / tensor[i] * 100)
    lib.computeError(I_test, tensor[i], doPrint)

    tensor_est[i] = I_test.copy()
    itensor_est[i] = np.linalg.inv(I_test)


#%% Re-simulate with estimated tensor
omega_sim_est = np.zeros_like(omega_sim)
omega_dot_sim_est = np.zeros_like(omega_dot_sim)
omega_sim_est[0, :] = w0
simulateVectorized(tensor_est, itensor_est, j, omega_sim_est, omega_dot_sim_est, wR_sim, wRdot_sim, dt_sim, M)

omega_est = omega_sim_est[::eval_every].copy()
omega_dot_est = omega_dot_sim_est[::eval_every].copy()

for i in tqdm(range(min(N, 10)), desc="Plotting: "):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col", gridspec_kw={'height_ratios': [2, 1]})
    lib.timePlotVector(t, omega[:,i,0], ax=ax1, label="Ground truth", ylabel=r"${\omega}$ (rad s$^{-1}$)", linestyle="dashed", alpha=0.5)
    lib.timePlotVector(t, omega_noisy[:,i,0], ax=ax1, label="Noisy", ylabel=r"${\omega}$ (rad s$^{-1}$)", alpha=0.4)
    lib.timePlotVector(t, omega_est[:,i,0], ax=ax1, label="Estimated", alpha=1)

    ax2.plot(t * 1000, w, label=r"${\omega}_f$ (s$^{-1}$)", color="tab:blue")
    ax2.plot(t * 1000, w_noisy[:,i], label="Noisy", color="tab:blue", alpha=0.4)

    ax1.grid()
    ax2.grid()
    #ax2.get_legend().remove()
    ax2.invert_yaxis()
    plt.tight_layout()
    # plt.show()
