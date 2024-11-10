from sympy import *

def symprint(var, name):
    print(f"----- {name} -----")
    print()
    print(var.__repr__())
    print()
    print("LaTeX: ", latex(var))

wx, wy, wz, dwx, dwy, dwz = symbols("wx wy wz dwx dwy dwz")
Ixx, Ixy, Iyy, Ixz, Iyz, Izz = symbols("Ixx Ixy Iyy Ixz Iyz Izz")

w = Matrix([wx, wy, wz])
dw = Matrix([dwx, dwy, dwz])
i = Matrix([Ixx, Ixy, Iyy, Ixz, Iyz, Izz])
I = Matrix([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

# torque-free equations of motion, left hand side
lhs = I @ dw  +  w.cross(I @ w)

# problem matrix
zetai = lhs.jacobian(i)
symprint(zetai, "zeta_i")

