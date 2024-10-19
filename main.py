from lib import *
import lib

df, omegas, accelerations, times, flywheel_omegas = importDatafile("gyro_spinning.csv")

filter_cutoff = 10
recomputeFilterCoefficients(filter_cutoff)
dt = (times[-1] - times[0])/len(times)
recomputeFilterCoefficients(filter_cutoff, dt)
filtered_omegas = filterVectorSignal(omegas)
filtered_flywheel_omegas = filterVectorSignal(flywheel_omegas)
filtered_accelerations = filterVectorSignal(accelerations)


jerks = differentiateVectorSignal(filtered_accelerations, dt)

absolute_accelerations = np.sqrt(accelerations[:,0] ** 2 + accelerations[:,1] ** 2 + accelerations[:,2] ** 2)
absolute_omegas = np.sqrt(omegas[:,0] ** 2 + omegas[:,1] ** 2 + omegas[:,2] ** 2)
absolute_jerks = np.sqrt(jerks[:,0] ** 2 + jerks[:,1] ** 2 + jerks[:,2] ** 2)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col", gridspec_kw={'height_ratios': [3, 1]})
# timePlotVector(times, omegas, ax=ax1, label="Angular velocity", ylabel="Angular velocity (rad/s)")
timePlotVector(times, filtered_omegas, ax=ax1, label="Filtered angular velocity", ylabel="Angular velocity (rad/s)")
omega_dots = differentiateVectorSignal(filtered_omegas, dt)
# timePlotVector(times, omega_dots, ax=ax1, label="Angular acceleration", linestyle="dashed", alpha=0.7)

timePlotVector(times, filtered_flywheel_omegas, ax=ax2, label="Flywheel angular velocity", ylabel="Flywheel angular velocity (rad/s)")
flywheel_omega_dots = differentiateVectorSignal(filtered_flywheel_omegas, dt)
# timePlotVector(times, flywheel_omega_dots, ax=ax2, label="Flywheel angular acceleration", linestyle="dashed", alpha=0.7)

lib.Jflywheel = 9.42e-8 # kg*m^2

starts, ends = detectThrow(times, absolute_omegas, absolute_accelerations, absolute_jerks)
ax1.axvline([times[starts[0]] * 1000], linestyle="dashed", color="gray")

I = computeI(filtered_omegas[starts[0]:], omega_dots[starts[0]:], filtered_flywheel_omegas[starts[0]:], flywheel_omega_dots[starts[0]:])
print(I)

plt.show()