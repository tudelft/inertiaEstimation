import numpy as np

# CONFIGURATION 01: light flywheel

m_dev = 0.073  # [kg]
m_obj = 0.346  # [kg]

Ixx = 1 / 12 * m_obj * (0.0302 ** 2 + 0.0700 ** 2)
Iyy = 1 / 12 * m_obj * (0.0302 ** 2 + 0.0600 ** 2)
Izz = 1 / 12 * m_obj * (0.0600 ** 2 + 0.0700 ** 2)
trueInertia = np.matrix([[Ixx, 0, 0],
                         [0, Iyy, 0],
                         [0, 0, Izz]])