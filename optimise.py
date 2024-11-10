from lib import *
import lib

LOGFILE_PATH = "input/archive"

files = ["bonus_test.csv", "more_spin_with_flywheel.csv", "counters_rotation.csv"]

# x[0] will be the filter cutoff threshold, and x[1] the flywheel inertia
def function_to_optimise(x):
    file_inertias = []
    for f in files:
        df, omegas, accelerations, times, flywheel_omegas \
            = importDatafile(os.path.join(LOGFILE_PATH, f))

        # Prepare discrete filter coefficients
        filter_cutoff = x[0]
        dt = (times[-1] - times[0])/len(times)
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
        absolute_accelerations = np.sqrt(accelerations[:,0] ** 2 +
                                         accelerations[:,1] ** 2 +
                                         accelerations[:,2] ** 2)
        absolute_omegas = np.sqrt(omegas[:,0] ** 2 +
                                  omegas[:,1] ** 2 +
                                  omegas[:,2] ** 2)
        absolute_jerks = np.sqrt(jerks[:,0] ** 2 +
                                 jerks[:,1] ** 2 +
                                 jerks[:,2] ** 2)

        starts, ends = detectThrow(times, absolute_omegas, absolute_accelerations, absolute_jerks)

        # Set flywheel inertia
        # lib.Jflywheel = x[0] # kg*m^2
        lib.Jflywheel = 9.909e-08

        # Compute inertia tensor with filtered data
        I = computeI(filtered_omegas[starts[0]:],
                     omega_dots[starts[0]:],
                     filtered_flywheel_omegas[starts[0]:],
                     flywheel_omega_dots[starts[0]:])
        file_inertias.append(I)

    error = 0
    for i in range(len(files)):
        for j in range(len(files)):
            if i == j:
                continue
            error += ((file_inertias[i] - file_inertias[j])**2).mean() / ((file_inertias[i] + file_inertias[j])**2).mean()
    print(file_inertias[1])
    print(error)
    return error

x = scipy.optimize.minimize(function_to_optimise, np.array([70]), tol=1e-16)
print(x)
print(x.x)