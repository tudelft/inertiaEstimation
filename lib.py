import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy

import sys
import os

SUPPORTED_IMPORTERS = ["csv", "bfl"]

def importDatafile(path, poles = 12, importer=None):
    # importer can be "csv" or "bfl" of None. 
    # if importer=None, then it goes by file extension

    if importer is None:
        # try to guess which importer to use by file extension
        extension = os.path.splitext(path)[-1][1:].lower()
        if extension not in SUPPORTED_IMPORTERS:
            raise ValueError(f"Could not determine file importer for file {path}: not in {SUPPORTED_IMPORTERS}")
        importer = extension
    else:
        # force use of that importer, if available
        if importer not in SUPPORTED_IMPORTERS:
            raise ValueError(f"Importer {importer} not supported. Must be one of {SUPPORTED_IMPORTERS}")

    if importer == "csv":
        df = pd.read_csv(path)
    elif importer == "bfl":
        # load module on demand, as it may not be available
        sys.path.append(os.path.join(os.path.dirname(__file__), "ext/indiflightSupport/LogAnalysis"))
        from indiflightLogTools import IndiflightLog
        log = IndiflightLog(path)
        df = log.raw.copy()

    # Angular velocities in rad/s
    df["gyroADC[0]"] = -df["gyroADC[0]"] * math.pi / 180 / 16.384
    df["gyroADC[1]"] = -df["gyroADC[1]"] * math.pi / 180 / 16.384
    df["gyroADC[2]"] =  df["gyroADC[2]"] * math.pi / 180 / 16.384

    # Linear acceleration in m/s/s
    df["accSmooth[0]"] = -df["accSmooth[0]"] * 9.81 / 2048
    df["accSmooth[1]"] = -df["accSmooth[1]"] * 9.81 / 2048
    df["accSmooth[2]"] =  df["accSmooth[2]"] * 9.81 / 2048

    # Time axis in s
    df["time"] = (df["time"] - df["time"].min()) / 1e6

    # Flywheel angular velocity in rad/s
    df["omegaFlywheel"] = df["erpm[0]"].values * 100 / (poles / 2) * 2 * math.pi / 60
    flywheelOmega = np.array([np.zeros(len(df)), np.zeros(len(df)), -df["omegaFlywheel"].values]).T

    omega = np.array([df["gyroADC[0]"].values,
                      df["gyroADC[1]"].values,
                      df["gyroADC[2]"].values]).T
    a = np.array([df["accSmooth[0]"].values,
                  df["accSmooth[1]"].values,
                  df["accSmooth[2]"].values]).T

    return df, omega, a, df["time"].values, flywheelOmega

# Arbitrarily define the locations of the coefficients.
# Note that it is always symmetric due to the definition of the product moments of inertia.
def buildTensor(x):
    return np.array([[x[0], x[1], x[3]],
                     [x[1], x[2], x[4]],
                     [x[3], x[4], x[5]]])

def buildVector(t):
    return np.array([t[0,0], t[1,0], t[1,1], t[2,0], t[2,1], t[2,2]])

def formatTicks(major=1000, minor=100):
    pythonstfu = True
    # loc = ticker.MultipleLocator(base=minor or 100)  # this locator puts ticks at regular intervals
    # plt.gca().xaxis.set_minor_locator(loc)
    # loc = ticker.MultipleLocator(base=major or 1000)  # this locator puts ticks at regular intervals
    # plt.gca().xaxis.set_major_locator(loc)

def timePlot(t, var, ylabel="", minor=None, major=None, ax=plt, **kwargs):
    ax.plot(t * 1000, var, **kwargs)

    plt.xlabel("Time (ms)")

    # formatTicks(major, minor)

    if len(ylabel) > 0:
        ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")

def timePlotVector(t, var, label="", ylabel="", minor=None, major=None, ax=plt, **kwargs):
    ax.plot(t * 1000, (var.T[0]).T, label=f"{label} (x)", color="tab:blue", **kwargs)
    ax.plot(t * 1000, (var.T[1]).T, label=f"{label} (y)", color="tab:orange", **kwargs)
    ax.plot(t * 1000, (var.T[2]).T, label=f"{label} (z)", color="tab:green", **kwargs)
    plt.xlabel("Time (ms)")

    # formatTicks(major, minor)

    if len(ylabel) > 0:
        ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")

global_filter_cutoff = 1000
filter_coefs = None
def filterSignal(signal, cutoff_freq=global_filter_cutoff):
    return scipy.signal.lfilter(filter_coefs[0], filter_coefs[1], signal)

def recomputeFilterCoefficients(filter_cutoff, dt):
    # global filter_coefs
    return scipy.signal.butter(2, filter_cutoff, output="ba", btype="lowpass", fs=1/dt)

def filterVectorSignal(signal, **kwargs):
    res = []
    for i in range(signal.shape[1]):
        res.append(filterSignal(signal[:, i].flatten(), **kwargs).flatten())
    return np.array(res).T

def derivativeCoefficients(n, f):
    T = np.zeros(n * n).reshape(n, n)
    res = np.zeros(n)
    res[1] = 1

    for y in range(n):
        for x in range(n):
            if y == 0:
                T[y, x] = 1
            elif x == 0:
                T[y, x] = 0
            else:
                T[y, x] = (-x) ** y / math.factorial(y)
    res = np.flip(np.linalg.solve(T, res))

    deriv_coefs_kernel = np.zeros(len(res) * f)
    for i, c in enumerate(res):
        deriv_coefs_kernel[i * f] = c
    return deriv_coefs_kernel * m / ((m + 1) * f)

m = 4 # Order of accuracy + 1
f = 1 #
deriv_coefs = derivativeCoefficients(m, f).reshape(-1, 1)
# def differentiateSignal(signal, dt):
#     global deriv_coefs
#     h = dt
#
#     signal = np.array(signal)
#     sig = signal.reshape(-1)
#     deriv_coefs = deriv_coefs.flatten()
#
#     new_signal = np.convolve(sig, np.flip(deriv_coefs), "same") / h
#     new_signal[0] = new_signal[1]
#     new_signal[-1] = new_signal[-2]
#     return new_signal

def differentiateSignal(signal, dt, window_length=86):
    return scipy.signal.savgol_filter(signal, window_length=window_length, polyorder=1, delta=dt, deriv=1)

def delaySavGolFilterVectorSignal(signal, *args, **kwargs):
    res = []
    for i in range(signal.shape[1]):
        res.append(delaySavGolFilterSignal(signal.T[i], *args, **kwargs))
    return np.array(res).T

def delaySavGolFilterSignal(signal, window_length=86):
    return scipy.signal.savgol_filter(signal, window_length=window_length, polyorder=1)


def differentiateVectorSignal(signal, dt, *args, **kwargs):
    res = []
    for i in range(signal.shape[1]):
        res.append(differentiateSignal(signal.T[i], dt, *args, **kwargs))
    return np.array(res).T

def computeI(angular_velocities, angular_accelerations, flywheel_angular_velocities, flywheel_angular_accelerations):
    A = []
    B = []
    ATA = np.zeros(36).reshape(6, 6)
    for i in range(len(angular_velocities)):
        omega = angular_velocities[i].flatten()
        omega_dot = angular_accelerations[i]
        flywheel_omega = flywheel_angular_velocities[i] + omega
        flywheel_omega_dot = flywheel_angular_accelerations[i] + omega_dot

        zeta_X = [omega_dot[0], -omega[2] * omega[0] + omega_dot[1], -omega[2] * omega[1],
                  omega[1] * omega[0] + omega_dot[2], omega[1] ** 2 - omega[2] ** 2,
                  omega[1] * omega[2]]
        A.append(zeta_X)
        zeta_Y = [omega[2] * omega[0], omega[2] * omega[1] + omega_dot[0], omega_dot[1],
                  omega[2] ** 2 - omega[0] ** 2, -omega[0] * omega[1] + omega_dot[2],
                  -omega[0] * omega[2]]
        A.append(zeta_Y)
        zeta_Z = [-omega[1] * omega[0], omega[0] ** 2 - omega[1] ** 2, omega[0] * omega[1],
                  -omega[1] * omega[2] + omega_dot[0], omega[0] * omega[2] + omega_dot[1],
                  omega_dot[2]]
        A.append(zeta_Z)
        zeta = np.matrix([zeta_X, zeta_Y, zeta_Z]).reshape((3, 6))

        global Jflywheel
        beta = -np.cross(omega, Jflywheel * flywheel_omega) - Jflywheel * flywheel_omega_dot
        B.extend(beta)

        ATA += np.matmul(zeta.T, zeta)
    inertiaCoefficients, res, rank, s = np.linalg.lstsq(A, B, rcond=None)
    return buildTensor(inertiaCoefficients)

def computeX(angular_velocities, angular_accelerations, linear_accelerations):
    A = []
    B = []
    for i in range(len(linear_accelerations)):
        a = linear_accelerations[i].reshape(-1, 1)
        omega = angular_velocities[i].flatten()
        omega_dot = angular_accelerations[i]

        # Matrix to find the distance between IMU and CG
        cg_submatrix_X = [-omega[1] ** 2 - omega[2] ** 2, omega[0] * omega[1] - omega_dot[2],
                          omega[0] * omega[2] + omega_dot[1]]
        cg_submatrix_Y = [omega[0] * omega[1] + omega_dot[2], -omega[0] ** 2 - omega[2] ** 2,
                          omega[1] * omega[2] - omega_dot[0]]
        cg_submatrix_Z = [omega[0] * omega[2] - omega_dot[1],
                          omega[1] * omega[2] + omega_dot[0], -omega[0] ** 2 - omega[1] ** 2]
        cg_submatrix = [cg_submatrix_X, cg_submatrix_Y, cg_submatrix_Z]
        A.extend(cg_submatrix)

        a_cg = np.zeros((3, 1))  # CG linear acceleration, assumed zero
        a_difference = a_cg - a
        B.extend(a_difference)

    a = np.array(A).reshape(-1, 3)
    b = np.array(B).reshape(-1, 1)
    x, res, rank, s = np.linalg.lstsq(a, b, rcond=None)
    return x

def computeError(I, I_true):
    # singular value decomposition gives guarantees on right-handedness of rotation matrix, i think
    R, lambdas, _ = np.linalg.svd(I)
    R_true, lambdas_true, _ = np.linalg.svd(I_true)

    # sort ascending, so x is smallest, z is largest
    lambdas[:]      = lambdas[[2,1,0]]
    lambdas_true[:] = lambdas_true[[2,1,0]]
    R[:, :]         = R[:, [2,1,0]]
    R_true[:, :]    = R_true[:, [2,1,0]]

    eigval_error_abs = np.linalg.norm(lambdas - lambdas_true)
    inertial_norm = np.linalg.norm(lambdas_true)
    eigval_error = eigval_error_abs / inertial_norm

    rotation = scipy.spatial.transform.Rotation.from_matrix(R)
    R_error = np.linalg.inv(R_true) @ R
    psi = np.arccos((np.linalg.trace(R_error) - 1.) / 2.)

    rotation_euler = rotation.as_euler('zyx')  # [rad]

    print(f"Absolute inertial error:   {eigval_error_abs:0.2e} kgm²")
    print(f"Inertial norm:             {inertial_norm:0.2e} kgm²")
    print(f"Inertial error:            {eigval_error * 100:0.2f}%")
    print(f"Alignment error:           {psi * 180 / math.pi:0.2f}°")
    print(f"  Euler around z:          {rotation_euler[0] * 180/math.pi:0.2f}°")
    print(f"  Euler around y:          {rotation_euler[1] * 180/math.pi:0.2f}°")
    print(f"  Euler around x:          {rotation_euler[2] * 180/math.pi:0.2f}°")

    return eigval_error, psi

def kroneckerDelta(i, j):
    return int(i == j)

def parallelAxisTheorem(m, r):
    res = np.zeros((3, 3))
    r_norm_squared = np.dot(r.T, r)
    for i in range(len(r)):
        for j in range(len(r)):
            res[i, j] = m * (r_norm_squared * kroneckerDelta(i, j) - r[i] * r[j])
    return res

def translateI(I_test, I_dev, m_obj, m_dev, x_dev, x_test):
    r = (m_dev / m_obj) * (x_dev - x_test)
    s = x_test - x_dev
    return I_test - parallelAxisTheorem(m_dev, s) - I_dev - parallelAxisTheorem(m_obj, r)

### Throw Detection ###
start_threshold = 25
start_threshold_jerk = 15
stop_threshold = 15
stop_threshold_jerk = 20
betweenPeaks = False
min_omega = 2

def startsTumbling(times, absolute_omegas, absolute_accelerations, absolute_jerks, flywheel_omegas):
    if abs(flywheel_omegas[-1][2]) > 0:
        return True
    else:
        return False
    tumbling = False

    jerk_old = absolute_jerks[-2]
    jerk_new = absolute_jerks[-1]

    linacc_old = absolute_accelerations[-2]
    linacc_new = absolute_accelerations[-1]

    global betweenPeaks
    global startpeak
    global endpeak
    currentTime = times[-1]

    if linacc_new > start_threshold and linacc_old < start_threshold:
        startpeak = currentTime
    if linacc_new < start_threshold and linacc_old > start_threshold:
        endpeak = currentTime

        if 0.1 <= (endpeak - startpeak) <= 0.3 and absolute_omegas[-1] > min_omega:
            betweenPeaks = True

    if betweenPeaks and abs(jerk_new) < start_threshold_jerk and not abs(jerk_old) < start_threshold_jerk:
        tumbling = True

    return tumbling

def stopsTumbling(times, absolute_omegas, absolute_accelerations, absolute_jerks):
    return False
    stops_tumbling = False

    jerk_old = absolute_jerks[-2]
    jerk_new = absolute_jerks[-1]

    linacc_old = absolute_accelerations[-2]
    linacc_new = absolute_accelerations[-1]

    global betweenPeaks
    if abs(jerk_new) > stop_threshold_jerk and abs(jerk_old) < stop_threshold_jerk:
        stops_tumbling = True
        betweenPeaks = False
    if linacc_new > stop_threshold and linacc_old < stop_threshold:
        stops_tumbling = True
        betweenPeaks = False

    return stops_tumbling

def detectThrow(times, absolute_omegas, absolute_accelerations, absolute_jerks, flywheel_omegas):
    start_indices = []
    end_indices = []
    wasTumbling = False
    for i in range(len(absolute_omegas) - 1):
        if not wasTumbling:
            if startsTumbling(times[i:i+2], absolute_omegas[i:i+2], absolute_accelerations[i:i+2], absolute_jerks[i:i+2], flywheel_omegas[i:i+2]):
                wasTumbling = True
                start_indices.append(i)
            else:
                continue
        else:
            if stopsTumbling(times[i:i+2], absolute_omegas[i:i+2], absolute_accelerations[i:i+2], absolute_jerks[i:i+2]) and i - start_indices[-1] > 50:
                wasTumbling = False
                end_indices.append(i)
                continue
    return start_indices, end_indices


def simulateThrow(inertiaTensor, times, omega_0, flywheel_omegas, flywheel_omega_dots):
    omega = omega_0
    omegas = []

    for i in range(len(times) - 1):
        w_times = times[i:i+2]
        flywheel_angular_momentum = flywheel_omegas[i+1] * Jflywheel
        flywheel_angular_momentum_dot = flywheel_omega_dots[i+1] * Jflywheel

        # Initialise begin and end times for interval
        t_1 = w_times[0]
        t_2 = w_times[1]
        dt = t_2 - t_1
        ddt = dt / 100 # Iterate N times between datapoints

        inv = np.linalg.inv(inertiaTensor)

        for t in np.arange(t_1, t_2, ddt):
            # Simulate by solving the Euler rotation equation for the angular acceleration and using it
            # to numerically integrate the angular velocity
            omega_dot = np.matmul(inv, -np.cross(omega, np.matmul(inertiaTensor, omega)) -
                                  np.cross(omega, flywheel_angular_momentum) - flywheel_angular_momentum_dot)
            omega = omega + omega_dot * ddt
        omegas.append(omega)
    omegas = np.array(omegas)
    return omegas