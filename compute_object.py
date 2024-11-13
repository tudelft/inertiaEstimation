from fontTools.mtiLib import build
import scipy.spatial

from lib import *
import lib
import os

LOGFILE_PATH = "input/block_experiment"
dirlist = ["device", "test"]

global_I = None
global_I_true = None

def optim(optimisation_variables):
    Is = []
    xs = []

    for dir in dirlist:
        l_filtered_omegas = []
        l_omega_dots = []
        l_filtered_flywheel_omegas = []
        l_flywheel_omega_dots = []
        l_filtered_accelerations = []
        print(os.path.join(LOGFILE_PATH, dir))
        for (dirpath, dirnames, files) in os.walk(os.path.join(LOGFILE_PATH, dir)):
            for f in files:
                print(f"== [ {f} ] ==")

                df, omegas, accelerations, times, flywheel_omegas \
                    = importDatafile(os.path.join(dirpath, f))

                # Prepare discrete filter coefficients
                filter_cutoff = optimisation_variables[0]  # [Hz]
                dt = (times[-1] - times[0]) / len(times)
                lib.filter_coefs = recomputeFilterCoefficients(filter_cutoff, dt)

                # Apply filter to data
                filtered_omegas = filterVectorSignal(omegas)
                filtered_flywheel_omegas = filterVectorSignal(flywheel_omegas)
                filtered_accelerations = filterVectorSignal(accelerations)

                # Numerically differentiate filtered signals
                jerks = differentiateVectorSignal(filtered_accelerations, dt)
                omega_dots = differentiateVectorSignal(filtered_omegas, dt)
                flywheel_omega_dots = differentiateVectorSignal(filtered_flywheel_omegas, dt)

                # Find lengths of filtered values
                absolute_accelerations = np.sqrt(accelerations[:, 0] ** 2 +
                                                 accelerations[:, 1] ** 2 +
                                                 accelerations[:, 2] ** 2)
                absolute_omegas = np.sqrt(omegas[:, 0] ** 2 +
                                          omegas[:, 1] ** 2 +
                                          omegas[:, 2] ** 2)
                absolute_jerks = np.sqrt(jerks[:, 0] ** 2 +
                                         jerks[:, 1] ** 2 +
                                         jerks[:, 2] ** 2)

                starts, ends = detectThrow(times, absolute_omegas, absolute_accelerations, absolute_jerks, flywheel_omegas)

                if len(starts) == 0:
                    print("No throws detected")
                    continue
                # lib.Jflywheel = 8.430e-08  # kg*m^2
                lib.Jflywheel = optimisation_variables[1]

                throw_offset = 400

                l_filtered_omegas.extend(filtered_omegas[starts[0] + throw_offset:])
                l_omega_dots.extend(omega_dots[starts[0] + throw_offset:])
                l_filtered_flywheel_omegas.extend(filtered_flywheel_omegas[starts[0] + throw_offset:])
                l_flywheel_omega_dots.extend(flywheel_omega_dots[starts[0] + throw_offset:])
                l_filtered_accelerations.extend(filtered_accelerations[starts[0] + throw_offset:])

            # Compute inertia tensor with filtered data
            I = computeI(l_filtered_omegas,
                         l_omega_dots,
                         l_filtered_flywheel_omegas,
                         l_flywheel_omega_dots)
            x = computeX(l_filtered_omegas,
                         l_omega_dots,
                         l_filtered_accelerations)
            Is.append(I)
            xs.append(x)
            break

    # global I, I_true
    m_dev = 0.073 # [kg]
    m_obj = 0.346 # [kg]

    x_dev = xs[0]
    x_test = xs[1]
    I_dev = Is[0]
    I_test = Is[1]

    I = translateI(I_test, I_dev, m_obj, m_dev, x_dev, x_test)
    global global_I
    global_I = I

    np.set_printoptions(formatter={'float': lambda x: format(x, '6.8e')})

    Ixx = 1/12 * m_obj * (0.0302 ** 2 + 0.0700 ** 2)
    Iyy = 1/12 * m_obj * (0.0302 ** 2 + 0.0600 ** 2)
    Izz = 1/12 * m_obj * (0.0600 ** 2 + 0.0700 ** 2)
    global global_I_true
    I_true = np.matrix([[ Ixx, 0, 0 ],
                        [ 0, Iyy, 0 ],
                        [ 0, 0, Izz ]])
    global_I_true = I_true

    # print(I - I_true)
    # print(I)
    # print(I_true)
    error = np.linalg.norm(buildVector(I) - buildVector(I_true)) / np.linalg.norm(I_true)
    # error = ((I - I_true)**2).mean() / ((I + I_true)**2).mean()
    # error = ((I - I_true) @ np.linalg.inv(I + I_true)).sum()
    print(f"* Error:    {error:6.15e}")
    print(f"* Flywheel inertia: {optimisation_variables[1]:6.15e}")

    computeError(I, I_true)

    return error
print("========/[ Optimisation ]\========")
x = scipy.optimize.minimize(optim, [100, 8.46746037e-08], tol=1e-16, method="Nelder-Mead")
# x = scipy.optimize.minimize(optim, [0.173, 2701 * 0.070 * 0.060 * 0.030], tol=1e-16)
print(x)
print(x.x)

#print(optim([100, 8.467e-08]))

print("========\[ Optimisation ]/========\n")

computeError(global_I, global_I_true)
