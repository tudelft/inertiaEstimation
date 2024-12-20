from lib import *
import lib
import os
import pathlib
import calibrate

LOGFILE_PATH = "cyberzoo_tests_the_second/config_a"
LOGFILES_ROOT = "input"
SAVE_FOR_PUBLICATION = False
doPlotting = False

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

eigval_errs = []
for (dirpath, dirnames, filenames) in os.walk(os.path.join(LOGFILES_ROOT, LOGFILE_PATH)):
    for f in filenames:
        if ".py" in f:
            continue
        print(f)
        df, omegas, accelerations, times, flywheel_omegas \
            = importDatafile(os.path.join(LOGFILES_ROOT, LOGFILE_PATH, f), new_motor=new_motor)

        filtered_accelerations, filtered_omegas, filtered_flywheel_omegas, \
            jerks, omega_dots, flywheel_omega_dots, \
            absolute_accelerations, absolute_omegas, absolute_jerks \
            = signalChain(accelerations, omegas, flywheel_omegas, times, LP_CUTOFF)

        if doPlotting:
            # # Initialise plot
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex="col", gridspec_kw={'height_ratios': [2, 1, 1]})
            timePlotVector(times, omegas, ax=ax1, label="Measured", ylabel=r"${\omega}$ (s$^{-1}$)", alpha=0.4)
            timePlotVector(times, filtered_omegas, ax=ax1, label="Filtered")
            # timePlotVector(times, omega_dots, ax=ax1, label="Angular acceleration", linestyle="dashed", alpha=0.8)

            timePlotVector(times, accelerations, ax=ax2, label="Measured", ylabel="Acceleration (ms$^{-1}$)", alpha=0.4)
            timePlotVector(times, filtered_accelerations, ax=ax2, label="Filtered")

            timePlotVector(times, flywheel_omegas, ax=ax3, label="Measured", ylabel=r"${\omega}_f$ (s$^{-1}$)", alpha=0.4)
            timePlotVector(times, filtered_flywheel_omegas, ax=ax3, label="Filtered", ylabel=r"${\omega}_f$ (s$^{-1}$)")
            # timePlotVector(times, filtered_flywheel_omegas / (2 * math.pi), ax=ax3, label="Filtered", ylabel=r"${\omega}_f$ (s$^{-1}$)")
            timePlotVector(times, flywheel_omega_dots, ax=ax3, label="Flywheel angular acceleration", linestyle="dashed", alpha=0.8)
            ax3.invert_yaxis()

        starts, ends = detectThrow(times, absolute_omegas, absolute_accelerations, absolute_jerks, flywheel_omegas)

        if len(starts) == 0:
             print("No throws detected")
             continue
             # plt.show()
             # sys.exit()

        # Set flywheel inertia
        lib.Jflywheel = j # kg*m^2
        # lib.Jflywheel = 1

        # Compute inertia tensor with filtered data
        I_test, residuals = computeI(filtered_omegas[starts[0]+throw_offset:],
                     omega_dots[starts[0]+throw_offset:],
                     filtered_flywheel_omegas[starts[0]+throw_offset:],
                     flywheel_omega_dots[starts[0]+throw_offset:])
        x_test, resx = computeX(filtered_omegas[starts[0]+throw_offset:],
                     omega_dots[starts[0]+throw_offset:],
                     filtered_accelerations[starts[0]+throw_offset:])


        sys.path.append(os.path.join(LOGFILES_ROOT, LOGFILE_PATH))
        import groundtruth

        I_obj = translateI(I_test, I_dev, groundtruth.m_obj, groundtruth.m_dev, x_dev, x_test)
        print(I_test)
        print(I_obj)

        eigval_error, psi = computeError(I_obj, groundtruth.trueInertia)
        eigval_errs.append(eigval_error)
        print(groundtruth.trueInertia)
        
        del groundtruth
        sys.path.remove(os.path.join(LOGFILES_ROOT, LOGFILE_PATH))

        if doPlotting:
            simulation_omegas = simulateThrow(I_test,
                                              times[starts[0]+throw_offset:],
                                              filtered_omegas[starts[0]+throw_offset],
                                              filtered_flywheel_omegas[starts[0]+throw_offset:],
                                              flywheel_omega_dots[starts[0]+throw_offset:])
            timePlotVector(times[starts[0]+throw_offset+1:], simulation_omegas, label="Simulated", ax=ax1, linestyle="dashed", alpha=0.8)

            ax2.get_legend().remove()
            ax3.get_legend().remove()

            for s in starts:
                ax1.axvline([times[s + throw_offset] * 1e3], linestyle="dashed", color="gray")
            for e in ends:
                ax1.axvline([times[e + throw_offset] * 1e3], linestyle="dotted", color="darkgray")

            ax1.set_xlim([times[starts[0]] * 1e3, times[-1] * 1e3])

            ax1.grid()
            ax2.grid()

            filename = os.path.splitext(os.path.join("output", LOGFILE_PATH, f))[0] + "-simulation.pdf"
            pathlib.Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)

            plt.savefig(filename, transparent=True, dpi=300, format="pdf", bbox_inches="tight")
            if SAVE_FOR_PUBLICATION:
                filename = os.path.splitext(os.path.join("output", LOGFILE_PATH, f))[0] + "-simulation_publication.pdf"
                fig.set_size_inches(10, 2.5)
                plt.savefig(filename, transparent=True, dpi=300, format="pdf", bbox_inches="tight")
            else:
                # formatTicks(100, 20)
                plt.tight_layout(pad=0.1)
                # plt.subplots_adjust(left=0.06, right=0.88, top=0.95, bottom=0.05)
                #manager = plt.get_current_fig_manager()
                #manager.window.move(-1680, 0)
                #manager.window.showMaximized()

                def on_resize(event):
                    fig.tight_layout()
                    fig.canvas.draw()
                cid = fig.canvas.mpl_connect('resize_event', on_resize)
                plt.show()
    break

print()
eigval_errs = np.asarray(eigval_errs)
print(f"mean eigval error: {100*eigval_errs.mean()}%")
print(f"max eigval error: {100*eigval_errs.max()}%")
