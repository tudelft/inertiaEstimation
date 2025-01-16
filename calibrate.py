#!/usr/bin/env python3

# sys libraries
import os
import sys
import pathlib
import importlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError

# pretty printing
from tqdm import tqdm
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

# actual number crunching
import numpy as np
import lib


def calibrate(calibration_path, output, throw_offset=300, filter_cutoff=20, new_motor=False):

    # load calibration configuration
    config_module = os.path.join(calibration_path, "config.py")
    spec = importlib.util.spec_from_file_location("", config_module)
    cfg = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(cfg)
    except:
        logger.error(f"'{config_module}' could not be imported.")
        sys.exit(1)


######################################################
################## Load and process data #############
######################################################

    # iterate first over device-only datafiles, then over device-plus-calibration
    # assuming j = 1 kgm^2
    device_only_path = os.path.join(calibration_path, "device_only")
    proof_body_path = os.path.join(calibration_path, "proof_body")

    Is, xs = [], []
    for direc in [device_only_path, proof_body_path]:
        l_filtered_omegas = []
        l_omega_dots = []
        l_filtered_flywheel_omegas = []
        l_flywheel_omega_dots = []
        l_filtered_accelerations = []
        for f in tqdm(os.listdir(direc), desc=f"Load {direc}"):
            if ".py" in f:
                continue
            df, omegas, accelerations, times, flywheel_omegas \
                = lib.importDatafile(os.path.join(direc, f),
                                 new_motor=new_motor)

            filtered_accelerations, filtered_omegas, filtered_flywheel_omegas, \
                jerks, omega_dots, flywheel_omega_dots, \
                absolute_accelerations, absolute_omegas, absolute_jerks \
                = lib.signalChain(accelerations, omegas, flywheel_omegas, times, filter_cutoff)

            starts, ends = lib.detectThrow(times, absolute_omegas, absolute_accelerations, absolute_jerks, flywheel_omegas)
            if len(starts) == 0:
                logger.warning(f"File {f}: no throws detected. Continuing with the next.")
                continue

            # Set flywheel inertia
            lib.Jflywheel = 1

            l_filtered_omegas.extend(filtered_omegas[starts[0] + throw_offset:-lib.WINDOW_LENGTH])
            l_omega_dots.extend(omega_dots[starts[0] + throw_offset:-lib.WINDOW_LENGTH])
            l_filtered_flywheel_omegas.extend(filtered_flywheel_omegas[starts[0] + throw_offset:-lib.WINDOW_LENGTH])
            l_flywheel_omega_dots.extend(flywheel_omega_dots[starts[0] + throw_offset:-lib.WINDOW_LENGTH])
            l_filtered_accelerations.extend(filtered_accelerations[starts[0] + throw_offset:-lib.WINDOW_LENGTH])

        # Compute inertia tensor with filtered data
        I, _ = lib.computeI(l_filtered_omegas,
                        l_omega_dots,
                        l_filtered_flywheel_omegas,
                        l_flywheel_omega_dots)
        x, _ = lib.computeX(l_filtered_omegas,
                        l_omega_dots,
                        l_filtered_accelerations)

        Is.append(I)
        xs.append(x)

    # assign inertias and cog found above
    I_dev_j1 = Is[0]
    x_dev_j1 = xs[0]
    I_dev_plus_cal_j1 = Is[1]
    x_dev_plus_cal_j1 = xs[1]


###############################################################
################## Perform the actual calibration #############
###############################################################

    # with known calibration body inertia tensor, calibrate flywheel inertia 
    logger.info(f"Data processing complete, performing calibration.")

    # calibration
    s = x_dev_j1 - x_dev_plus_cal_j1
    r = (cfg.m_dev / cfg.m_obj) * s

    translated_true_inertia = (cfg.I_obj +
                               lib.parallelAxisTheorem(cfg.m_obj, r) +
                               lib.parallelAxisTheorem(cfg.m_dev, s))
    right_side_matrix = I_dev_plus_cal_j1 - I_dev_j1

    left_side_vector = lib.buildVector(translated_true_inertia)
    right_side_vector = lib.buildVector(right_side_matrix)

    # final flywheel inertia estimate
    j = np.dot(right_side_vector, left_side_vector) / np.linalg.norm(right_side_vector) ** 2

    # final device inertia estimate
    I_dev = I_dev_j1 * j
    x_dev = x_dev_j1


##############################################
################## Debug outputs #############
##############################################

    logger.debug("""
=====================================================
====== Calibration Report for Flywheel Inertia ======
=====================================================
""")

    logger.info (f"Orthogonal projection flywheel inertia:  {j:.4e} kgm^2")

    e = j * right_side_vector - left_side_vector
    logger.debug(f"OPF inertial error:                      {np.linalg.norm(e):.4e} kgm^2")

    logger.debug("== ERROR MATRIX ==\n" + str(lib.buildTensor(e)))
    logger.debug("== GROUND TRUTH MATRIX ==\n" + str(cfg.I_obj))
    with np.errstate(divide='ignore'):
        # Calibration error wrt true assumed inertia
        logger.debug("== RELATIVE ERROR PERCENTAGES ==\n" + f"{100 * lib.buildTensor(e / lib.buildVector(cfg.I_obj))} %")

    logger.debug("== ESTIMATE INERTIAL AND ALIGNMENT ERROR ==")
    epsilon, psi, _, _ = lib.computeError(cfg.I_obj + lib.buildTensor(e), cfg.I_obj, doPrint=(logger.level <= logging.DEBUG))
    logger.info(f"Proof mass reproduction accuracy: {100*epsilon:.2f}% scaling, {180/np.pi*psi:.2f}° alignment")


    logger.debug("""
===================================================
====== Calibration Report for Device Inertia ======
===================================================
""")
    logger.info("Device Inertia Matrix found:\n" + str(I_dev))


###############################################################
################## Output to calibration file #################
###############################################################

    print(f"Saving calibration to {output}")

    pathlib.Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True)
    with open(output, "w") as outfile:
        timestamp = datetime.now().isoformat()

        outfile.write("###### Flywheel and Device inertia calibration ######\n")
        outfile.write("from datetime import datetime\n")
        outfile.write("from numpy import array\n")
        outfile.write(f'date = datetime.fromisoformat("{timestamp}")\n')
        outfile.write(f"j = {j}  # flywheel inertia in kgm^2\n")
        outfile.write(f"epsilon = {epsilon}  # calibration scale reproduction accuracy\n")
        outfile.write(f"psi = {psi}  # calibration alignment reproduction accuracy\n")

        np.set_printoptions(formatter={'all':np.format_float_scientific})
        outfile.write(f'm_dev = {cfg.m_dev}\n')
        outfile.write(f'x_dev = eval("""{repr(x_dev)}""")\n')
        outfile.write(f'I_dev = eval("""{repr(I_dev)}""")\n')

    return j, epsilon, psi, I_dev, x_dev


if __name__=="__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentError

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("data", type=str, metavar="DATA_PATH", help="Path to search for throw data with and without attached proof mass")
    parser.add_argument("--output", type=str, default="./output/calibration.py", help="Output calibration file")
    parser.add_argument("--offset", type=int, default=300, metavar="SAMPLES", help="After detection of throw, skip SAMPLES")
    parser.add_argument("--cutoff", type=float, default=20, help="Lowpass filter cutoff in Hz")
    parser.add_argument("-v", action="count", help="Increase verbosity (can be passed up to 2 times)")
    args = parser.parse_args()

    # global log config for loggers in modules
    logging.basicConfig(level=logging.ERROR)

    # config for loggers in this file
    # config for loggers in this file
    if args.v is None:
        logger.setLevel(logging.WARNING)
    elif args.v == 1:
        logger.setLevel(logging.INFO)
    elif args.v == 2:
        logger.setLevel(logging.DEBUG)
    else:
        raise ArgumentError("-v must be passed between 0 and 2 times")

    if not args.output.endswith(".py"):
        raise ArgumentError("Output file must be a python file")

    calibrate(args.data,
              args.output,
              throw_offset=args.offset,
              filter_cutoff=args.cutoff,
              new_motor=True)
