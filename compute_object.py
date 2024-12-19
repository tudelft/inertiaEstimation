from lib import *
import calibrate
import lib
import os

LOGFILE_PATH = "input/cyberzoo_tests_the_second"
dirlist = ["device", "config_c"]

sys.path.append(LOGFILE_PATH)

global_I = None
global_I_true = None

LP_CUTOFF = 100

throw_offset = 300
new_motor = True

j, _, __, I_dev, x_dev = calibrate.calibrateFlywheel("cyberzoo_tests_the_second",
                                dirlist=["device", "calibration"],
                                GROUNDTRUTH_PATH="calibration",
                                new_motor=new_motor)

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
                = importDatafile(os.path.join(dirpath, f), new_motor=new_motor)

            # Prepare discrete filter coefficients
            dt = (times[-1] - times[0]) / len(times)

            filtered_accelerations = filterVectorSignalButterworth(accelerations, LP_CUTOFF, dt)
            # Apply filter to data
            # filtered_omegas = omegas
            # filtered_flywheel_omegas = flywheel_omegas
            filtered_flywheel_omegas = filterVectorSignalButterworth(flywheel_omegas, LP_CUTOFF, dt)
            filtered_accelerations = filterVectorDynamicNotch(filtered_accelerations,
                                                              filtered_flywheel_omegas[:, 2] / (2 * math.pi),
                                                              10,  # 2.27
                                                              dt)
            # filtered_accelerations = filterVectorDynamicNotch(filtered_accelerations,
            #                                                   filtered_flywheel_omegas[:, 2] / (math.pi),
            #                                                   50,
            #                                                   dt)
            filtered_omegas = filterVectorSignalButterworth(omegas, LP_CUTOFF, dt)
            filtered_flywheel_omegas = filterVectorSignalButterworth(flywheel_omegas, LP_CUTOFF, dt)

            # Numerically differentiate filtered signals
            jerks = differentiateVectorSignal(accelerations, dt)
            omega_dots = differentiateVectorSignal(omegas, dt)
            flywheel_omega_dots = differentiateVectorSignal(flywheel_omegas, dt)

            omega_dots = filterVectorSignalButterworth(omega_dots, LP_CUTOFF, dt)
            flywheel_omega_dots = filterVectorSignalButterworth(flywheel_omega_dots, LP_CUTOFF, dt)

            # Numerically differentiate filtered signals
            jerks = differentiateVectorSignal(accelerations, dt)
            omega_dots = differentiateVectorSignal(omegas, dt)
            flywheel_omega_dots = differentiateVectorSignal(flywheel_omegas, dt)

            omega_dots = filterVectorSignalButterworth(omega_dots, LP_CUTOFF, dt)
            flywheel_omega_dots = filterVectorSignalButterworth(flywheel_omega_dots, LP_CUTOFF, dt)

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

print(groundtruth.trueInertia - I)
e = buildVector(groundtruth.trueInertia - I)
with np.errstate(divide='ignore'):
    print(f"{100 * buildTensor(e / buildVector(groundtruth.trueInertia))} %")
