import sys
import numpy as np
from lib import *
import matplotlib.pyplot as plt

N = 1000
dt = 1/N
t = np.linspace(0, 1, N)
y = np.array([np.sin(t * 2 * math.pi), # 1 Hz
              np.sin(10 * t * 2 * math.pi), # 2 Hz
              np.sin(100 * t * 2 * math.pi)]).T # 4 Hz
f = 10 + 0 * t

bandwidth = 0.5
filtered_accelerations = filterVectorDynamicNotch(y, f, bandwidth, dt)
ground_truth = filterNotchFrequencies(y, [10], dt, bandwidth=bandwidth)

# # Initialise plot
fig, ax1 = plt.subplots(1, 1, sharex="col")

timePlotVector(t, y, ax=ax1, label="Original", ylabel="Acceleration (ms$^{-1}$)", alpha=0.2)
timePlotVector(t, filtered_accelerations, ax=ax1, label="Filtered")
timePlotVector(t, ground_truth, ax=ax1, label="Ground truth", linestyle="dashed", alpha=0.4)

def on_resize(event):
    fig.tight_layout()
    fig.canvas.draw()
cid = fig.canvas.mpl_connect('resize_event', on_resize)
plt.show()