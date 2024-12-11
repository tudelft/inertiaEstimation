from input.new_motor.calibration import calibration_groundtruth
from lib import *
import calibrate
import lib
import os

LOGFILE_PATH = "input/cyberzoo_tests"
dirlist = ["device", "config_a"]

sys.path.append(LOGFILE_PATH)

global_I = None
global_I_true = None

filter_cutoff = 10

def iter(iteration_params):
    j, epsilon, phi = calibrate.calibrateFlywheel("cyberzoo_tests", dirlist=["device", "calibration_copy"], GROUNDTRUTH_PATH="calibration_copy", filter_cutoff=iteration_params[0])

    Is = []
    xs = []

    for dir in dirlist:
        l_filtered_omegas = []
        l_omega_dots = []
        l_filtered_flywheel_omegas = []
        l_flywheel_omega_dots = []
        l_filtered_accelerations = []
        for (dirpath, dirnames, files) in os.walk(os.path.join(LOGFILE_PATH, dir)):
            for f in files:
                if ".py" in f:
                    continue
                print(f"== [ {f} ] ==")

                df, omegas, accelerations, times, flywheel_omegas \
                    = importDatafile(os.path.join(dirpath, f))

                # Prepare discrete filter coefficients
                dt = (times[-1] - times[0]) / len(times)
                lib.filter_coefs = recomputeFilterCoefficients(iteration_params[0], dt)

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
                lib.Jflywheel = j

                throw_offset = 100

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

    x_dev = xs[0]
    x_test = xs[1]
    I_dev = Is[0]
    I_test = Is[1]

    sys.path.append(os.path.join(LOGFILE_PATH, dirlist[1]))
    import groundtruth

    print(I_test)
    I = translateI(I_test, I_dev, groundtruth.m_obj, groundtruth.m_dev, x_dev, x_test)

    np.set_printoptions(formatter={'float': lambda x: format(x, '.8e')})
    i, psi = computeError(I, groundtruth.trueInertia)
    return epsilon

optim = scipy.optimize.minimize(iter, x0=[10], bounds=[(0., None)], method="L-BFGS-B")
print(optim)
print(optim.x[0])