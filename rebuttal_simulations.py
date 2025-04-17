import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 10
inertia_min = -1.e-3  # [kg m^2]
inertia_max = 1.e-3  # [kg m^2]
random_vars = np.random.random((N, 6))
inertia_vectors = inertia_min + (inertia_max - inertia_min) * random_vars

lib.Jflywheel = 1.e-6
dt = 1e-3
LP_CUTOFF = 20

omega_0 = np.array([-1, 1, 0])
df, omega, a, time, flywheelOmegas = lib.importDatafile("input/final_tests/config_a/LOG00133.BFL")

df = df.loc[df["omegaFlywheel"] > 0]
times = df["time"].values
dt = (time[-1] - time[0])/len(time)

flywheelOmegas = lib.filterSignalButterworth(-df["omegaFlywheel"].values, 30, dt)
#flywheelOmegas = lib.filterSignalButterworth(-df["omegaFlywheel"].values, 100, dt)
#flywheelOmegas = -df["omegaFlywheel"].values
flywheelOmegasVec = np.empty((times.shape[0], 3))
flywheelOmegasVec[:, 2] = flywheelOmegas
flywheelOmegaDotsVec = lib.differentiateVectorSignal(flywheelOmegasVec, dt)
noise_magnitude = 0.

np.random.seed(42)

for t in inertia_vectors:
    #inertia_tensor = lib.buildPhysicalTensor(t)
    inertia_tensor = lib.buildVeryPhysicalTensor(-4, -2.5) # 10^-4 to 10^-2.5 log random space
    print("Ground truth:\n", inertia_tensor)

    omega = lib.simulateThrow(inertia_tensor, times, omega_0, flywheelOmegasVec, flywheelOmegaDotsVec)
    omegaNoise = np.random.normal(0, noise_magnitude, omega.shape)
    flywheelNoise = np.random.normal(0, noise_magnitude, flywheelOmegasVec.shape)

    noisyOmega = omega + omegaNoise
    noisyFlywheelOmega = flywheelOmegasVec + flywheelNoise

    # Calculate
    filteredOmegas = lib.filterVectorSignalButterworth(noisyOmega, LP_CUTOFF, dt)
    omegaDots = lib.differentiateVectorSignal(filteredOmegas, dt)
    filteredFlywheelOmegas = lib.filterVectorSignalButterworth(noisyFlywheelOmega, LP_CUTOFF, dt)
    flywheelOmegaDots = lib.differentiateVectorSignal(filteredFlywheelOmegas, dt)

    I_test, residuals = lib.computeI(filteredOmegas,
                                     omegaDots,
                                     filteredFlywheelOmegas,
                                     flywheelOmegaDots)
    print("Estimate\n", I_test)
    # print((inertia_tensor - I_test) / inertia_tensor * 100)
    lib.computeError(I_test, inertia_tensor, True)

    # Re-simulate with estimated tensor
    omega_after = lib.simulateThrow(I_test, times, omega_0, flywheelOmegasVec, flywheelOmegaDotsVec)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col", gridspec_kw={'height_ratios': [2, 1]})
    lib.timePlotVector(times[1:], omega, ax=ax1, label="Ground truth", ylabel=r"${\omega}$ (s$^{-1}$)", linestyle="dashed", alpha=0.5)
    lib.timePlotVector(times[1:], noisyOmega, ax=ax1, label="Noisy", alpha=0.4)
    lib.timePlotVector(times[1:], omega_after, ax=ax1, label="Estimated", alpha=1)

    lib.timePlotVector(times, flywheelOmegasVec, ax=ax2, label="Ground truth", ylabel=r"${\omega}_f$ (s$^{-1}$)", linestyle="dashed", alpha=0.8)
    lib.timePlotVector(times, noisyFlywheelOmega, ax=ax2, label="Noisy", alpha=0.4)

    ax1.grid()
    ax2.grid()
    ax2.get_legend().remove()
    ax2.invert_yaxis()
    plt.tight_layout()
    plt.show()