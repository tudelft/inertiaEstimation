#!/usr/bin/env python3

# sys libraries
import os
import sys
import pathlib
import importlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError

# pretty printing
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

# actual number crunching
from matplotlib import pyplot as plt
import pandas as pd
import lib

COLUMNS = [
    "id", "config",
    "w_init",
    "mgt_obj", "xgt_obj", "Igt_obj",
    "x_test", "x_obj", "I_test", "I_obj",
    "Psi", "dphi", "dtheta", "dpsi", "eps", "dlx", "dly", "dlz"
]

def analyse_single_file(filename, config, calib, throw_offset=300, filter_cutoff=20, new_motor=False, save_plots_to=None):
    res_row = dict.fromkeys(COLUMNS) # initualize all columns to NONE
    res_row['id'] = filename

    # check config
    if not hasattr(config, "m_obj") or not hasattr(config, "name"):
        logger.error(f"'name' and 'm_obj' must be set in config for {filename}")
        return res_row

    # check if groundtruth
    has_groundtruth = hasattr(config, "I_obj")
    if has_groundtruth and not hasattr(config, "x_obj"):
        config.x_obj = None

    if not has_groundtruth:
        logger.info(f"Config for {filename} does not define groundtruth. Continuing without error metrics")

    res_row['config'] = config.name
    res_row['mgt_obj'] = config.m_obj

    # import datafile
    df, omegas, accelerations, times, flywheel_omegas \
        = lib.importDatafile(filename, new_motor=new_motor)

    # do filtering
    filtered_accelerations, filtered_omegas, filtered_flywheel_omegas, \
        jerks, omega_dots, flywheel_omega_dots, \
        absolute_accelerations, absolute_omegas, absolute_jerks \
            = lib.signalChain(accelerations, omegas, flywheel_omegas, times, filter_cutoff)

    # plot raw and filtered data, if asked
    if save_plots_to is not None:
        logger.info(f"plotting raw and filtered input data for {filename}")
        # # Initialise plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex="col", gridspec_kw={'height_ratios': [2, 1, 1]})

        lib.timePlotVector(times, omegas, ax=ax1, label="Measured", ylabel=r"${\omega}$ (s$^{-1}$)", alpha=0.4)
        lib.timePlotVector(times, filtered_omegas, ax=ax1, label="Filtered")
        # lib.timePlotVector(times, omega_dots, ax=ax1, label="Angular acceleration", linestyle="dashed", alpha=0.8)

        lib.timePlotVector(times, accelerations, ax=ax2, label="Measured", ylabel="Acceleration (ms$^{-1}$)", alpha=0.4)
        lib.timePlotVector(times, filtered_accelerations, ax=ax2, label="Filtered")

        lib.timePlotVector(times, flywheel_omegas, ax=ax3, label="Measured", ylabel=r"${\omega}_f$ (s$^{-1}$)", alpha=0.4)
        lib.timePlotVector(times, filtered_flywheel_omegas, ax=ax3, label="Filtered", ylabel=r"${\omega}_f$ (s$^{-1}$)")
        # timePlotVector(times, filtered_flywheel_omegas / (2 * math.pi), ax=ax3, label="Filtered", ylabel=r"${\omega}_f$ (s$^{-1}$)")
        lib.timePlotVector(times, flywheel_omega_dots, ax=ax3, label="Flywheel angular acceleration", linestyle="dashed", alpha=0.8)
        ax3.invert_yaxis()

    # detect throws
    starts, ends = lib.detectThrow(times, absolute_omegas, absolute_accelerations, absolute_jerks, flywheel_omegas)
    if len(starts) == 0:
        logger.warning(f"File {filename}: no throws detected. Aborting its analusis.")
        return res_row

    # Set flywheel inertia
    lib.Jflywheel = calib.j # kg*m^2

    # Compute inertia tensor of object+device with filtered data
    start_idx = starts[0] + throw_offset
    I_test, residuals = lib.computeI(filtered_omegas[start_idx:],
                                     omega_dots[start_idx:],
                                     filtered_flywheel_omegas[start_idx:],
                                     flywheel_omega_dots[start_idx:])
    x_test, resx = lib.computeX(filtered_omegas[start_idx:],
                                omega_dots[start_idx:],
                                filtered_accelerations[start_idx:])

    # extract just object I and x
    I_obj = lib.translateI(I_test, calib.I_dev, config.m_obj, calib.m_dev, calib.x_dev, x_test)
    x_obj = lib.translateX(config.m_obj, calib.m_dev, calib.x_dev, x_test)

    logger.info(f"""
====================================================================
====== Output for Object "{config.name}" using datafile "{filename}"
====================================================================
""")
    logger.info(f"Mass was given as (kg)         : {config.m_obj}")
    logger.info(f"Center of gravity found at (mm): {1e3*x_obj.squeeze()}")
    logger.info(f"Inertia matrix found as (kgmm2):\n{1e6*I_obj}\n")

    # logging
    res_row['w_init'] = filtered_omegas[start_idx].squeeze().copy()
    res_row['x_test'] = x_test.squeeze().copy()
    res_row['x_obj'] = x_obj.squeeze().copy()
    res_row['I_test'] = I_test
    res_row['I_obj'] = I_obj

    # compute and log errors, if we have a groundtruth. Maybe simulate and plot
    if has_groundtruth is None:
        logger.debug(f"""
====================================================================
====== Error Report for Object "{config.name}" unavailable; no groundtruth inertia configured
====================================================================
""")
    else:
        logger.debug(f"""
====================================================================
====== Error Report for Object "{config.name}" using datafile "{filename}"
====================================================================
""")

        eigval_error, psi, eigval_error_vector, euler_errors \
            = lib.computeError(I_obj, config.I_obj, doPrint=(logger.level <= logging.DEBUG))

        # logging
        res_row['xgt_obj'] = config.x_obj
        res_row['Igt_obj'] = config.I_obj

        res_row['Psi'] = psi
        res_row['dphi'] = euler_errors[2]
        res_row['dtheta'] = euler_errors[1]
        res_row['dpsi'] = euler_errors[0]
        res_row['eps'] = eigval_error
        res_row['dlx'] = eigval_error_vector[0]
        res_row['dly'] = eigval_error_vector[1]
        res_row['dlz'] = eigval_error_vector[2]

    if save_plots_to is not None:
        simulation_omegas = lib.simulateThrow(I_test,
                                              times[start_idx:],
                                              filtered_omegas[start_idx],
                                              filtered_flywheel_omegas[start_idx:],
                                              flywheel_omega_dots[start_idx:])
        lib.timePlotVector(times[start_idx+1:], simulation_omegas, label="Simulated", ax=ax1, linestyle="dashed", alpha=0.8)

        ax2.get_legend().remove()
        ax3.get_legend().remove()

        for s in starts:
            ax1.axvline([times[s + throw_offset] * 1e3], linestyle="dashed", color="gray")
        for e in ends:
            ax1.axvline([times[e + throw_offset] * 1e3], linestyle="dotted", color="darkgray")

        ax1.set_xlim([times[starts[0]] * 1e3, times[-1] * 1e3])

        ax1.grid()
        ax2.grid()

        outfilename = os.path.splitext(os.path.join(save_plots_to, f))[0] + "-simulation.pdf"
        pathlib.Path(os.path.dirname(outfilename)).mkdir(parents=True, exist_ok=True)

        fig.set_size_inches(10, 2.5)
        plt.savefig(outfilename, transparent=True, dpi=300, format="pdf", bbox_inches="tight")

    return res_row


if __name__=="__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("data", type=str, metavar="DATA_PATH", help="Folder contraining object-under-test data BFL files, config.py and optionally groundtruth.py")
    parser.add_argument("calibration", type=str, metavar="CALIBRATION", help="Calibration output python file from calibrate.py")
    parser.add_argument("--output", type=str, metavar="PATH", default=None, help="If passed, will write output dataframe pickle to this folder")
    parser.add_argument("--offset", type=int, default=300, metavar="SAMPLES", help="After detection of throw, skip SAMPLES")
    parser.add_argument("--cutoff", type=float, default=20, help="Lowpass filter cutoff in Hz")
    parser.add_argument("--plots", action="store_true", help="Give plot output. Requires --output to be set")
    parser.add_argument("-v", action="count", help="Increase verbosity (can be passed up to 2 times)")
    args = parser.parse_args()

    # check options
    if args.plots and not args.output:
        raise ArgumentError("Must also specify --output, if you use --plots")

    # global log config for loggers in modules
    logging.basicConfig(level=logging.ERROR)

    # config for loggers in this file
    if args.v is None:
        logger.setLevel(logging.WARNING)
    elif args.v == 1:
        logger.setLevel(logging.INFO)
    elif args.v == 2:
        logger.setLevel(logging.DEBUG)
    else:
        raise ArgumentError("-v must be passed between 0 and 2 times")

    # import calibration
    spec = importlib.util.spec_from_file_location("", args.calibration)
    calib = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(calib)
    except:
        logger.error(f"'{args.calibration}' could not be imported.")
        sys.exit(1)

    # import config
    args.data = args.data.rstrip("/").rstrip("\\") # remove problematic trailing slashes
    config_module = os.path.join(args.data, "config.py")
    spec = importlib.util.spec_from_file_location("", config_module)
    config = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(config)
    except:
        logger.error(f"'{config_module}' could not be imported.")
        sys.exit(1)


    # finally start doing stuff
    plotpath = os.path.join(args.output, f"{os.path.basename(args.data)}_plots")
    res = pd.DataFrame(columns=COLUMNS)
    for f in tqdm(os.listdir(args.data), desc="Analysing files"):
        if not (f.lower().endswith(".bfl") or f.lower().endswith(".csv")):
            continue

        row = analyse_single_file(os.path.join(args.data, f),
                                  config,
                                  calib,
                                  throw_offset=args.offset,
                                  filter_cutoff=args.cutoff,
                                  new_motor=True,
                                  save_plots_to=plotpath if args.plots else None)
        res.loc[len(res)] = row


    # output final results table for later analysis
    if args.output is not None:
        outfilename = os.path.join(args.output, f"{os.path.basename(args.data)}.pkl")
        print(f"Writing output pickle to {outfilename}")
        pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
        res.to_pickle(outfilename)
