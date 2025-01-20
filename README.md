# Find inertia matrix from IMU data of a tumbling body

## flight controller firmware and running calibrations/tests

Todo

Data doesn't need to be in BFL format (betaflight/indiflight binary logs). Can also be CSV with the following columns:
```
"time": integer, time in microseconds
"erpm[0]": integer, electrical rpm of the flywheel, such that    omega (rad/s) = erpm[0] * 100 / (n_poles / 2) * 2 * pi / 60
"gyroADC[0]": integer, unfiltered gyro x, such that   rotation (rad/s) = gyroADC[0] * pi/180 / 16.384
"gyroADC[1]": integer, unfiltered gyro y
"gyroADC[2]": integer, unfiltered gyro z
"accSmooth[0]": integer, unfiltered accelerometer x, such that   specific force (m/s/s) = accSmooth[0] * 9.81 / 2048.
"accSmooth[1]": integer, unfiltered acc y
"accSmooth[2]": integer, unfiltered acc z
```


## Running data analysis

### Environment setup

Tested on Ubuntu 22.04. Use `.venv` to keep python dependencies neat:
1. `git submodule init && git submodule sync && git submodule update`
2. `apt install python3-venv python3-pip`
3. `python3 -m venv ~/.inertia-venv`
4. `source ~/.inertia-venv/bin/activate`
5. `pip install -r requirements.txt`


### Calibration

1. Obtain mass of measurement device and proof body by weighing
2. Obtain inertia matrix of proof body, e.g. from geometry
3. Obtain IMU data from throwing measurement device alone, and also with the attached proof mass.
    - **NB: make sure axes of IMU align with your definition of the inertia matrix in step 2**
    - **NB2: make sure proof body has 3 different principal moments of inertia, and is heavier than the measurement device**
4. Organize the resulting data like so:
```
input
└── my_calibration
    ├── config.py
    ├── device_only
    │   ├── LOG00119.BFL
    │   ├── LOG00131.BFL
    │   └── ...
    └── proof_body
        ├── LOG00122.BFL
        ├── LOG00125.BFL
        └── ...
```
5. `config.py` should look like this. `m_dev`, `m_obj` and `I_obj` have to be defined.
```python
import numpy as np

# device mass
m_dev = 0.10067  # [kg]

# proof body mass and inertia
m_obj = 0.346  # [kg]

Ixx = 1 / 12 * m_obj * (0.0302 ** 2 + 0.0700 ** 2)
Iyy = 1 / 12 * m_obj * (0.0302 ** 2 + 0.0600 ** 2)
Izz = 1 / 12 * m_obj * (0.0600 ** 2 + 0.0700 ** 2)
I_obj = np.matrix([[Ixx, 0, 0],
                    [0, Iyy, 0],
                    [0, 0, Izz]])
```
6. run calibration with `python3 calibrate.py -v --output my_calibration.py input/my_calibration`
   - now there is a file called `my_calibration.py` with the calibration data
   - use `-vv` to get more output on screen


### Computing objects

1. Obtain the mass of your object e.g. by weighing on a scale.
2. Attach the device to the object and obtain throw data.
   - **NB: we compute everything in the IMU frame, so note the position of the IMU with respect to your object.**
3. Organize the resulting data like so:
```
input
├── my_calibration
│   └── ...
└── my_object
    ├── config.py
    ├── LOG00180.BFL
    ├── LOG00181.BFL
    └── ...
```
4. `config.py` should look like this. `name` and `m_obj` must be set. To compare the algorithms output with a ground-truth, set `I_obj` as well
```python
name = "my_object"
m_obj = 1.234 # object mass in [kg]

# optional, if you have a ground truth or estimate you want to compare with:
# from numpy import array
# x_obj = 1e-3*array([1., 2., 3.]) # object cog in [m]. Currently unused
# I_obj = 1e-3*array([
#    [1., 0., 0.],
#    [0., 1., 0.],
#    [0., 0., 1.],
#]) # object inertia in [kgm^2]
```
4. run calculations `python3 analyse_individually.py -v input/my_object my_calibration.py`
5. use `-vv` for more output
6. use `--output ./` to get the entire output in a pickled pandas dataframe, which you can import like this
```python
import pandas as pd
df = pd.read_pickle("./my_object.pkl")
print(df)
```

## Reproduce paper results

If you're running linux, you can reproduce our paper's results. See the `for_publication` branch.
