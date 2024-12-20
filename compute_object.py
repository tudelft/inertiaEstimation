from lib import *
import calibrate
import lib
import os

LOGFILE_PATH = "input/cyberzoo_tests_the_second"
dir = "config_c"

sys.path.append(LOGFILE_PATH)

global_I = None
global_I_true = None

LP_CUTOFF = 20
throw_offset = 300

new_motor = True

j, _, __, I_dev, x_dev = calibrate.calibrateFlywheel(
                            "cyberzoo_tests_the_second",
                            dirlist=["device", "calibration"],
                            GROUNDTRUTH_PATH="calibration",
                            new_motor=new_motor,
                            filter_cutoff=LP_CUTOFF,
                            )

Is = []
xs = []

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

        filtered_accelerations, filtered_omegas, filtered_flywheel_omegas, \
            jerks, omega_dots, flywheel_omega_dots, \
            absolute_accelerations, absolute_omegas, absolute_jerks \
            = signalChain(accelerations, omegas, flywheel_omegas, times, LP_CUTOFF)

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

#x_dev = xs[0]
x_test = xs[0]
#I_dev = Is[0]
I_test = Is[0]

sys.path.append(os.path.join(LOGFILE_PATH, dir))
import groundtruth

print(I_test)
I = translateI(I_test, I_dev, groundtruth.m_obj, groundtruth.m_dev, x_dev, x_test)

np.set_printoptions(formatter={'float': lambda x: format(x, '.8e')})
i, psi = computeError(I, groundtruth.trueInertia)

print(groundtruth.trueInertia - I)
e = buildVector(groundtruth.trueInertia - I)
with np.errstate(divide='ignore'):
    print(f"{100 * buildTensor(e / buildVector(groundtruth.trueInertia))} %")
