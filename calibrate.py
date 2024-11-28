from input.block_experiment.groundtruth import trueInertia
from lib import *
import lib

LOGFILES_ROOT = "input"
LOGFILE_PATH = "block_experiment"
dirlist = ["device", "test"]

sys.path.append(os.path.join(LOGFILES_ROOT, LOGFILE_PATH))
# walk_output = os.walk(os.path.join(LOGFILES_ROOT, LOGFILE_PATH))

Is = []
xs = []
for dir in dirlist:
    l_filtered_omegas = []
    l_omega_dots = []
    l_filtered_flywheel_omegas = []
    l_flywheel_omega_dots = []
    l_filtered_accelerations = []
    print(os.path.join(LOGFILE_PATH, dir))
    for (dirpath, dirnames, files) in os.walk(os.path.join(LOGFILES_ROOT, LOGFILE_PATH, dir)):
        for f in files:
            df, omegas, accelerations, times, flywheel_omegas \
                = importDatafile(os.path.join(LOGFILES_ROOT, LOGFILE_PATH, dir, f))

            # Prepare discrete filter coefficients
            filter_cutoff = 85
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

            # Set flywheel inertia
            lib.Jflywheel = 1

            if len(starts) == 0:
                continue

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

from groundtruth import *

r = (m_dev / m_obj) * (xs[0] - xs[1])
s = xs[0] - xs[1]

left_side_matrix = trueInertia + parallelAxisTheorem(m_obj, r) + parallelAxisTheorem(m_dev, s)
right_side_matrix = Is[1] - Is[0]

left_side_vector = buildVector(left_side_matrix)
right_side_vector = buildVector(right_side_matrix)
j = np.dot(right_side_vector, left_side_vector) / np.linalg.norm(right_side_vector) ** 2
e = j * right_side_vector - left_side_vector

print(f"\nOrthogonal projection flywheel inertia:  {j:.4e} kgm^2")
print(f"OPF inertial error:                      {np.linalg.norm(e):.4e} kgm^2")

print("\n== ERROR MATRIX ==")
print(buildTensor(e))
print("\n== GROUND TRUTH MATRIX ==")
print(trueInertia)

print("\n== RELATIVE ERROR PERCENTAGES ==")
# Calibration error wrt true assumed inertia
with np.errstate(divide='ignore'):
    print(f"OPF relative error:                     \n{100 * buildTensor(e / buildVector(trueInertia))} %")
print("\n== ESTIMATE INERTIAL AND ALIGNMENT ERROR ==")
computeError(trueInertia + buildTensor(e), trueInertia)