from fontTools.mtiLib import build

from lib import *
import lib
import os

device_files = ["LOG00030.BFL.csv", "LOG00033.BFL.csv", "LOG00034.BFL.csv"]
object_files = ["LOG00035.BFL.csv", "LOG00036.BFL.csv", "LOG00037.BFL.csv", "LOG00038.BFL.csv"]
filelists = [device_files, object_files]

Is = []
xs = []

for files in filelists:
    l_filtered_omegas = []
    l_omega_dots = []
    l_filtered_flywheel_omegas = []
    l_flywheel_omega_dots = []
    l_filtered_accelerations = []
    for f in files:
        print(f"== [ {f} ] ==")
        df, omegas, accelerations, times, flywheel_omegas = importDatafile(f)

        # Prepare discrete filter coefficients
        filter_cutoff = 86
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
        lib.Jflywheel = 8.430e-08 # kg*m^2

        l_filtered_omegas.extend(filtered_omegas[starts[0]:])
        l_omega_dots.extend(omega_dots[starts[0]:])
        l_filtered_flywheel_omegas.extend(filtered_flywheel_omegas[starts[0]:])
        l_flywheel_omega_dots.extend(flywheel_omega_dots[starts[0]:])
        l_filtered_accelerations.extend(filtered_accelerations[starts[0]:])

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

I = None
I_true = None

def optim(x):
    global I, I_true
    # m_dev = 0.173
    m_dev = x[0]
    # m_dev = 1.98933907e-01
    # print(m_dev)
    # m_obj = 2701 * 0.070 * 0.060 * 0.030 # 0.340326 [kg]
    m_obj = x[1]
    # m_obj = 3.81938958e-01
    x_dev = xs[0]
    x_test = xs[1]
    I_dev = Is[0]
    I_test = Is[1]

    r = (m_dev / m_obj) * (x_dev - x_test)
    I = translateI(I_test, I_dev, m_obj, r)

    np.set_printoptions(formatter={'float': lambda x: format(x, '6.8e')})

    Ixx = 1/12 * m_obj * (0.030 ** 2 + 0.070 ** 2)
    Iyy = 1/12 * m_obj * (0.030 ** 2 + 0.060 ** 2)
    Izz = 1/12 * m_obj * (0.060 ** 2 + 0.070 ** 2)
    I_true = np.matrix([[ Ixx, 0, 0 ],
                        [ 0, Iyy, 0 ],
                        [ 0, 0, Izz ]])

    # print(I - I_true)
    # print(I)
    # print(I_true)
    error = np.linalg.norm(buildVector(I_true) - buildVector(I)) / np.linalg.norm(buildVector(I_true))
    # error = ((I - I_true)**2).mean() / ((I + I_true)**2).mean()
    # error = ((I - I_true) @ np.linalg.inv(I + I_true)).sum()
    print(f"* Error:    {error:6.15e}")
    return error
print("======== [ Optimisation ] ========")
x = scipy.optimize.minimize(optim, [0.173, 2701 * 0.070 * 0.060 * 0.030], tol=1e-16)
print(x)
print(x.x)

# optim([0])

lambdas, eigenvectors = np.linalg.eigh(I)
principal_axis_order = np.diag(I).argsort()
P = eigenvectors[:, principal_axis_order]
lambdas = lambdas[principal_axis_order]
eigval_error = np.linalg.norm(np.array(lambdas) - np.diag(I_true)) / np.linalg.norm(np.diag(I_true))

principal_transform = np.linalg.inv(P) @ I @ P @ np.linalg.inv(I)
theta_x = math.atan2(-principal_transform[1, 2], principal_transform[2, 2])
theta_y = math.asin(principal_transform[0, 2])
theta_z = math.atan2(-principal_transform[0, 1], principal_transform[0, 0])

rotation = scipy.spatial.transform.Rotation.from_matrix(principal_transform)
rotation_euler = rotation.as_euler('zyx') # [rad]
total_rotation = np.linalg.norm(rotation_euler)

print(rotation_euler * 180/math.pi)
print(f"Inertial error:   {eigval_error * 100:0.2f}%")
print(f"Alignment error:  {total_rotation * 180 / math.pi:0.2f}°")
