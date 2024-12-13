import numpy as np
import lib

# CONFIGURATION 03: proper heavy flywheel, larger motor

m_dev = 0.10067  # [kg]

grid = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=bool)
m, _, trueInertia = lib.calcGridObject(grid)
m_obj = m

