import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy

import sys
import os

SUPPORTED_IMPORTERS = ["csv", "bfl"]

def importDatafile(path, importer=None, new_motor=False):
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
    wraw = df[[f"gyroADC[{i}]" for i in range(3)]].to_numpy() * math.pi / 180 / 16.384

    # Linear acceleration in m/s/s
    araw = df[[f"accSmooth[{i}]" for i in range(3)]].to_numpy() * 9.81 / 2048

    if new_motor:
        # because of cables with new bigger motor, FC rotated -90deg around z
        df["gyroADC[0]"] = -wraw[:, 1]
        df["gyroADC[1]"] = +wraw[:, 0]
        df["gyroADC[2]"] = +wraw[:, 2]
        df["accSmooth[0]"] = -araw[:, 1]
        df["accSmooth[1]"] = +araw[:, 0]
        df["accSmooth[2]"] = +araw[:, 2]
    else:
        df["gyroADC[0]"] = -wraw[:, 0]
        df["gyroADC[1]"] = -wraw[:, 1]
        df["gyroADC[2]"] = +wraw[:, 2]
        df["accSmooth[0]"] = -araw[:, 0]
        df["accSmooth[1]"] = -araw[:, 1]
        df["accSmooth[2]"] = +araw[:, 2]

    # Time axis in s
    df["time"] = (df["time"] - df["time"].min()) / 1e6

    # Flywheel angular velocity in rad/s
    poles = 14 if new_motor else 12
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

# Builds a physically possible tensor from only positive inputs.
def buildPhysicalTensor(x):
    diagonals = [abs(x[0]), abs(x[2]), abs(x[5])]
    sorted_diagonals = sorted(diagonals)
    return np.array([[sorted_diagonals[0], x[1], x[3]],
                     [x[1], sorted_diagonals[1], x[4]],
                     [x[3], x[4], sorted_diagonals[2]]])

# Builds a physically possible tensor from only positive inputs.
def buildPhysicalDiagonalTensor(x):
    diagonals = [abs(x[0]), abs(x[2]), abs(x[5])]
    sorted_diagonals = sorted(diagonals)
    return np.array([[sorted_diagonals[0], 0, 0],
                     [0, sorted_diagonals[1], 0],
                     [0, 0, sorted_diagonals[2]]])

def buildVector(t):
    return np.array([t[0,0], t[1,0], t[1,1], t[2,0], t[2,1], t[2,2]])

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
        pass
        ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(0.5, 1.02), ncol=3, loc="lower center")

global_filter_cutoff = 1000
filter_coefs = None
def filterSignalButterworth(signal, filter_cutoff, dt):
    filter_coefs = scipy.signal.butter(4, filter_cutoff, output="ba", btype="lowpass", fs=1/dt)
    return scipy.signal.filtfilt(filter_coefs[0], filter_coefs[1], signal)

def filterSignal(signal, filter_coefficients):
    return scipy.signal.filtfilt(filter_coefficients[0], filter_coefficients[1], signal)

# def recomputeFilterCoefficients(filter_cutoff, dt):
#     return scipy.signal.butter(4, filter_cutoff, output="ba", btype="lowpass", fs=1/dt)

def computeNotchFilterCoefficients(filter_cutoff, dt, bandwidth):
    Q = filter_cutoff / bandwidth
    return scipy.signal.iirnotch(filter_cutoff, Q, 1 / dt)

def filterNotchFrequencies(signal, frequencies, dt, **kwargs):
    for f in frequencies:
        filter_coefs = computeNotchFilterCoefficients(f, dt, **kwargs)
        signal = filterVectorSignal(signal, filter_coefs)
    return signal

def filterVectorSignal(signal, filter_coefs):
    res = []
    for i in range(signal.shape[1]):
        res.append(filterSignal(signal[:, i].flatten(), filter_coefs).flatten())
    return np.array(res).T

def filterVectorSignalButterworth(signal, filter_cutoff, dt):
    res = []
    for i in range(signal.shape[1]):
        res.append(filterSignalButterworth(signal[:, i].flatten(), filter_cutoff, dt).flatten())
    return np.array(res).T

def filterVectorDynamicNotch(signal, frequencies, bandwidth, dt):
    assert len(signal) == len(frequencies)
    res = []
    for i in range(signal.shape[1]):
        res.append(applySignalDynamicNotch(signal[:, i].flatten(), frequencies, bandwidth, dt).flatten())
    return np.array(res).T

def applySignalDynamicNotch(signal, frequencies, bandwidth, dt):
    nyquist_freq = 0.5 / dt  # Nyquist frequency
    result = np.zeros_like(signal)

    for i in range(len(signal)):
        flywheel_frequency = abs(frequencies[i])
        if flywheel_frequency < 20:
            result[i] = signal[i]
            continue

        # Compute filter parameters
        w0 = flywheel_frequency / nyquist_freq
        Q = w0 / (bandwidth / nyquist_freq)

        # Design the notch filter
        b, a = scipy.signal.iirnotch(flywheel_frequency, Q, 1 / dt)

        # Apply zero-phase filtering to the entire signal
        result[i] = scipy.signal.filtfilt(b, a, signal, method="pad")[i]

    return result

WINDOW_LENGTH = 50 # used to be 100
def differentiateSignal(signal, dt):
    return np.gradient(signal, dt)
    # return scipy.signal.savgol_filter(signal, window_length=WINDOW_LENGTH, polyorder=1, delta=dt, deriv=1)

def delaySavGolFilterVectorSignal(signal, *args, **kwargs):
    res = []
    for i in range(signal.shape[1]):
        res.append(delaySavGolFilterSignal(signal.T[i], *args, **kwargs))
    return np.array(res).T

def delaySavGolFilterSignal(signal, window_length=86):
    # raise Exception("Deprecated")
    return scipy.signal.savgol_filter(signal, window_length=window_length, polyorder=1)


def differentiateVectorSignal(signal, dt, *args, **kwargs):
    res = []
    for i in range(signal.shape[1]):
        res.append(differentiateSignal(signal.T[i], dt, *args, **kwargs))
    return np.array(res).T

def signalChain(accelerations, omegas, flywheel_omegas, times, LP_CUTOFF):
    # Prepare discrete filter coefficients
    dt = (times[-1] - times[0]) / len(times)

    filtered_accelerations = filterVectorSignalButterworth(accelerations, LP_CUTOFF, dt)
    filtered_omegas = filterVectorSignalButterworth(omegas, LP_CUTOFF, dt)
    filtered_flywheel_omegas = filterVectorSignalButterworth(flywheel_omegas, LP_CUTOFF, dt)
    #filtered_accelerations = filterVectorDynamicNotch(filtered_accelerations,
    #                                                  filtered_flywheel_omegas[:, 2] / (2 * math.pi),
    #                                                  10,
    #                                                  dt)

    # Numerically differentiate filtered signals
    jerks = differentiateVectorSignal(accelerations, dt)
    omega_dots = differentiateVectorSignal(omegas, dt)
    flywheel_omega_dots = differentiateVectorSignal(flywheel_omegas, dt)

    omega_dots = filterVectorSignalButterworth(omega_dots, LP_CUTOFF, dt)
    flywheel_omega_dots = filterVectorSignalButterworth(flywheel_omega_dots, LP_CUTOFF, dt)

    # Find lengths of filtered values
    absolute_accelerations = np.sqrt(accelerations[:,0] ** 2 +
                                     accelerations[:,1] ** 2 +
                                     accelerations[:,2] ** 2)
    absolute_omegas = np.sqrt(omegas[:,0] ** 2 +
                              omegas[:,1] ** 2 +
                              omegas[:,2] ** 2)
    absolute_jerks = np.sqrt(jerks[:,0] ** 2 +
                             jerks[:,1] ** 2 +
                             jerks[:,2] ** 2)

    return filtered_accelerations, filtered_omegas, filtered_flywheel_omegas, \
        jerks, omega_dots, flywheel_omega_dots, \
        absolute_accelerations, absolute_omegas, absolute_jerks


# Severely reduces RAM usage at the cost of slightly increased computation time, due to an
# increase in matrix multiplications, even though the resulting matrices are smaller
OPTIMISATION = False

def computeI(angular_velocities, angular_accelerations, flywheel_angular_velocities, flywheel_angular_accelerations):
    if OPTIMISATION:
        ATA = np.zeros(36).reshape(6, 6)
        ATB = np.zeros(6).reshape(1, 6)
    else:
        A = []
        B = []
    for i in range(len(angular_velocities)):
        omega = angular_velocities[i].flatten()
        omega_dot = angular_accelerations[i]
        flywheel_omega = flywheel_angular_velocities[i] + omega
        flywheel_omega_dot = flywheel_angular_accelerations[i] + omega_dot

        zeta_X = [omega_dot[0], -omega[2] * omega[0] + omega_dot[1], -omega[2] * omega[1],
                  omega[1] * omega[0] + omega_dot[2], omega[1] ** 2 - omega[2] ** 2,
                  omega[1] * omega[2]]
        zeta_Y = [omega[2] * omega[0], omega[2] * omega[1] + omega_dot[0], omega_dot[1],
                  omega[2] ** 2 - omega[0] ** 2, -omega[0] * omega[1] + omega_dot[2],
                  -omega[0] * omega[2]]
        zeta_Z = [-omega[1] * omega[0], omega[0] ** 2 - omega[1] ** 2, omega[0] * omega[1],
                  -omega[1] * omega[2] + omega_dot[0], omega[0] * omega[2] + omega_dot[1],
                  omega_dot[2]]
        zeta = np.matrix([zeta_X, zeta_Y, zeta_Z]).reshape((3, 6))

        global Jflywheel
        mu = -np.cross(omega, Jflywheel * (flywheel_omega + omega)) - Jflywheel * flywheel_omega_dot

        if OPTIMISATION:
            ATA += zeta.T @ zeta
            ATB += zeta.T @ mu
        else:
            A.append(zeta_X)
            A.append(zeta_Y)
            A.append(zeta_Z)
            B.extend(mu)
    if OPTIMISATION:
        inertiaCoefficients, residuals, rank, s = np.linalg.lstsq(ATA, ATB.reshape(6), rcond=None)
    else:
        inertiaCoefficients, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

    return buildTensor(inertiaCoefficients), residuals

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
    x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)
    return x, residuals

def computeError(I, I_true, doPrint=True):
    # singular value decomposition gives guarantees on right-handedness of rotation matrix, i think. oops, it's not...
    R, lambdas, _ = np.linalg.svd(I)
    R_true, lambdas_true, _ = np.linalg.svd(I_true)

    # sort ascending, so x is smallest, z is largest
    lambdas[:]      = lambdas[[2,1,0]]
    lambdas_true[:] = lambdas_true[[2,1,0]]
    R[:, :]         = R[:, [2,1,0]]
    R_true[:, :]    = R_true[:, [2,1,0]]

    R /= np.linalg.det(R)
    R_true /= np.linalg.det(R_true)

    eigval_error_abs = np.linalg.norm(lambdas - lambdas_true)
    inertial_norm = np.linalg.norm(lambdas_true)
    eigval_error = eigval_error_abs / inertial_norm

    # run through equivalent rotations (180deg rotations around each vector)
    max_error = +np.inf
    for perm in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]:
        R_try = R.copy()
        R_try[:, 0] *= perm[0]
        R_try[:, 1] *= perm[1]
        R_try[:, 2] *= perm[2]

        R_error_try = np.linalg.inv(R_true) @ R_try
        psi_try = np.arccos((np.trace(R_error_try) - 1.) / 2.)

        if psi_try < max_error:
            # log new best orientation
            max_error = psi_try
            psi = psi_try
            R_error = R_error_try.copy()
            rotation_error = scipy.spatial.transform.Rotation.from_matrix(R_error)
            euler_error = rotation_error.as_euler('zyx')  # [rad]

    if doPrint:
        print(f"Alignment error:           {psi * 180 / math.pi: 0.2f}°")
        print(f"  Euler around z:          {euler_error[0] * 180/math.pi: 0.2f}°")
        print(f"  Euler around y:          {euler_error[1] * 180/math.pi: 0.2f}°")
        print(f"  Euler around x:          {euler_error[2] * 180/math.pi: 0.2f}°")
        print(f"Absolute inertial error:   {eigval_error_abs: 0.2e} kgm²")
        print(f"Inertial norm:             {inertial_norm: 0.2e} kgm²")
        print(f"Inertial error:            {eigval_error * 100: 0.2f}%")

    return eigval_error, psi, lambdas_true - lambdas, euler_error

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
    """
    Returns the object inertia based on combined (test) inertia, device inertia,
    then masses and cog estimates
    """
    r = (m_dev / m_obj) * (x_dev - x_test)
    s = x_test - x_dev
    return I_test - parallelAxisTheorem(m_dev, s) - I_dev - parallelAxisTheorem(m_obj, r)

def translateX(m_obj, m_dev, x_dev, x_test):
    # x_test  =  (  m_obj * x_obj + m_dev * x_dev   )  /  (m_obj + m_dev)
    return ( x_test * (m_obj + m_dev)  -  m_dev * x_dev )  /  m_obj

### Throw Detection ###
start_threshold = 25
start_threshold_jerk = 15
stop_threshold = 15
stop_threshold_jerk = 20
betweenPeaks = False
min_omega = 2

def startsTumbling(times, absolute_omegas, absolute_accelerations, absolute_jerks, flywheel_omegas):
    if abs(flywheel_omegas[-1][2]) > 10:
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


def calcGridObject(grid=np.zeros((4, 8), dtype=bool)):
    # coordinate system for CoG and Inertia is
    #
    # fixed at the center of the bottom surface of the grid
    # xy plane is the bottom surface, x axis to the shorter side, y axis to the longest side
    # z down

    #%% center of mass

    M = np.zeros(3, dtype=float)
    _m = 0.

    # base
    mb = 178.5e-3
    cgb = np.array([0., 0., -18.9e-3])  # cad
    Mb = cgb * mb

    _m += mb
    M += Mb

    # blocks
    mbl = 70.1e-3
    dbl = np.array([15e-3, 15e-3, 40e-3], dtype=float)
    cg_bl = lambda i, j: 1e-3 * np.array([9 + (i-2)*18, 9 + (j-4)*18, -22], dtype=float)
    for i, row in enumerate(grid):
        for j, element in enumerate(row):
            if not element:
                continue
            _m += mbl
            M += cg_bl(i, j) * mbl

    cg = M / _m

    #%% inertia
    def getIOfCuboid(dim, mass):
        return 1/12 * mass \
            * np.diag([dim[1]**2 + dim[2]**2, dim[0]**2 + dim[2]**2, dim[0]**2 + dim[1]**2])

    # total inertia
    _I = np.zeros((3,3), dtype=float)

    # from cad
    Ib_cad = np.diag([2.994e10, 1.004e10, 3.506e10]) * 1e-9 * 1e-2  # cad was using 10mm per mm somehow
    m_cad = 145.2 # kg...
    Ib = mb / m_cad * Ib_cad
    Ib += parallelAxisTheorem(mb, cgb - cg)

    _I += Ib

    Ibl = getIOfCuboid(dbl, mbl)
    for i, row in enumerate(grid):
        for j, element in enumerate(row):
            if not element:
                continue
            _I += Ibl + parallelAxisTheorem(mbl, cg_bl(i, j) - cg)

    return _m, M / _m, _I


if __name__=="__main__":
    grid = np.zeros((4, 8), dtype=bool)
    grid[0, 0] = grid[0, 7] = grid[3, 0] = grid[3, 7] = True

    print(grid)
    print(calcGridObject(grid))
