import numpy as np
import lib

# name and m_obj must be set, the rest can be None or unset
# to compute error metrics, set I_obj, and optinally x_obj

name = "mavlab B"

grid = np.array([
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0]
], dtype=bool)
m_obj, x_obj, I_obj = lib.calcGridObject(grid)

