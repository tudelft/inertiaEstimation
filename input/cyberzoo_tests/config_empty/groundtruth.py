import numpy as np
import lib

# CONFIGURATION 02: heavy flywheel, larger motor

m_dev = 0.09953  # [kg]

grid = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=bool)
m, _, trueInertia = lib.calcGridObject(grid)
m_obj = m
