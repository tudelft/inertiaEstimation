from lib import *
import lib

LOGFILES_ROOT = "input"
LOGFILE_PATH = "cyberzoo_test_the_second"
dirlist = ["device", "test"]

LP_CUTOFF = 100

# window_length = 250
throw_offset = 300
default_filter_cutoff = 150

def calibrateFlywheel(LOGFILE_PATH, LOGFILES_ROOT = "input", dirlist = ["device", "test"], GROUNDTRUTH_PATH="", filter_cutoff=default_filter_cutoff, new_motor=False):
    print("==== CALIBRATING FLYWHEEL INERTIA ====")

    # if GROUNDTRUTH_PATH:
    #     sys.path.append(os.path.join(LOGFILES_ROOT, GROUNDTRUTH_PATH))
    # else:
    sys.path.append(os.path.join(LOGFILES_ROOT, LOGFILE_PATH, GROUNDTRUTH_PATH))

    Is = []
    xs = []
    # iterate first over device-only datafiles, then over device-plus-calibration
    # assuming j = 1 kgm^2
    for dir in dirlist:
        l_filtered_omegas = []
        l_omega_dots = []
        l_filtered_flywheel_omegas = []
        l_flywheel_omega_dots = []
        l_filtered_accelerations = []
        print(os.path.join(LOGFILE_PATH, dir))
        for (dirpath, dirnames, files) in os.walk(os.path.join(LOGFILES_ROOT, LOGFILE_PATH, dir)):
            for f in files:
                if ".py" in f:
                    continue
                df, omegas, accelerations, times, flywheel_omegas \
                    = importDatafile(os.path.join(LOGFILES_ROOT, LOGFILE_PATH, dir, f),
                                     new_motor=new_motor)

                # Prepare discrete filter coefficients
                dt = (times[-1] - times[0]) / len(times)

                filtered_accelerations = filterVectorSignalButterworth(accelerations, 100, dt)
                # Apply filter to data
                filtered_omegas = omegas
                # filtered_flywheel_omegas = flywheel_omegas
                filtered_flywheel_omegas = filterVectorSignalButterworth(flywheel_omegas, 100, dt)
                filtered_accelerations = filterVectorDynamicNotch(filtered_accelerations,
                                                                  filtered_flywheel_omegas[:, 2] / (2 * math.pi),
                                                                  10,
                                                                  dt)
                # filtered_accelerations = filterVectorDynamicNotch(filtered_accelerations,
                #                                                   filtered_flywheel_omegas[:, 2] / (math.pi),
                #                                                   50,
                #                                                   dt)
                filtered_omegas = filterVectorSignalButterworth(omegas, 100, dt)
                filtered_flywheel_omegas = filterVectorSignalButterworth(flywheel_omegas, 100, dt)

                # Numerically differentiate filtered signals
                jerks = differentiateVectorSignal(accelerations, dt)
                omega_dots = differentiateVectorSignal(omegas, dt)
                flywheel_omega_dots = differentiateVectorSignal(flywheel_omegas, dt)

                omega_dots = filterVectorSignalButterworth(omega_dots, 100, dt)
                flywheel_omega_dots = filterVectorSignalButterworth(flywheel_omega_dots, 100, dt)

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

                # Set flywheel inertia
                lib.Jflywheel = 1

                if len(starts) == 0:
                    continue

                l_filtered_omegas.extend(filtered_omegas[starts[0] + throw_offset:-lib.WINDOW_LENGTH])
                l_omega_dots.extend(omega_dots[starts[0] + throw_offset:-lib.WINDOW_LENGTH])
                l_filtered_flywheel_omegas.extend(filtered_flywheel_omegas[starts[0] + throw_offset:-lib.WINDOW_LENGTH])
                l_flywheel_omega_dots.extend(flywheel_omega_dots[starts[0] + throw_offset:-lib.WINDOW_LENGTH])
                l_filtered_accelerations.extend(filtered_accelerations[starts[0] + throw_offset:-lib.WINDOW_LENGTH])
            # Compute inertia tensor with filtered data
            I, residuals = computeI(l_filtered_omegas,
                         l_omega_dots,
                         l_filtered_flywheel_omegas,
                         l_flywheel_omega_dots)
            x, resx = computeX(l_filtered_omegas,
                         l_omega_dots,
                         l_filtered_accelerations)
            Is.append(I)
            xs.append(x)
            break

    # assign inertias and cog found above
    I_dev_j1 = Is[0]
    x_dev_j1 = xs[0]
    I_dev_plus_cal_j1 = Is[1]
    x_dev_plus_cal_j1 = xs[1]

    # with known calibration body inertia tensor, calibrate flywheel inertia 
    # and device inertia
    try:
        import calibration_groundtruth
    except ImportError:
        import groundtruth as calibration_groundtruth

    s = x_dev_j1 - x_dev_plus_cal_j1
    r = (calibration_groundtruth.m_dev / calibration_groundtruth.m_obj) * s

    translated_true_inertia = (calibration_groundtruth.trueInertia +
                               parallelAxisTheorem(calibration_groundtruth.m_obj, r) +
                               parallelAxisTheorem(calibration_groundtruth.m_dev, s))
    right_side_matrix = I_dev_plus_cal_j1 - I_dev_j1

    left_side_vector = buildVector(translated_true_inertia)
    right_side_vector = buildVector(right_side_matrix)
    j = np.dot(right_side_vector, left_side_vector) / np.linalg.norm(right_side_vector) ** 2

    I_dev = I_dev_j1 * j

    e = j * right_side_vector - left_side_vector

    # phi = np.linalg.lstsq(right_side_matrix, left_side_vector)

    print(f"\nOrthogonal projection flywheel inertia:  {j:.4e} kgm^2")
    print(f"OPF inertial error:                      {np.linalg.norm(e):.4e} kgm^2")

    print("\n== ERROR MATRIX ==")
    print(buildTensor(e))
    print("\n== GROUND TRUTH MATRIX ==")
    print(calibration_groundtruth.trueInertia)

    print("\n== RELATIVE ERROR PERCENTAGES ==")
    # Calibration error wrt true assumed inertia
    with np.errstate(divide='ignore'):
        print(f"{100 * buildTensor(e / buildVector(calibration_groundtruth.trueInertia))} %")
    print("\n== ESTIMATE INERTIAL AND ALIGNMENT ERROR ==")
    epsilon, psi = computeError(calibration_groundtruth.trueInertia + buildTensor(e), calibration_groundtruth.trueInertia)

    del calibration_groundtruth
    sys.path.remove(os.path.join(LOGFILES_ROOT, LOGFILE_PATH, GROUNDTRUTH_PATH))

    return j, epsilon, psi, I_dev, x_dev_j1

if __name__=="__main__":
    calibrateFlywheel("cyberzoo_tests_the_second",
                                dirlist=["device", "calibration"],
                                GROUNDTRUTH_PATH="calibration",
                                new_motor=True)