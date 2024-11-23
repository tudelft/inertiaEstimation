from sympy import *

def symprint(var, name):
    print(f"----- {name} -----")
    print()
    print(var.__repr__())
    print()
    print("LaTeX: ", latex(var))

wx, wy, wz, dwx, dwy, dwz = symbols("wx wy wz dwx dwy dwz")
Ixx, Ixy, Iyy, Ixz, Iyz, Izz = symbols("Ixx Ixy Iyy Ixz Iyz Izz")
Jxx, Jxy, Jyy, Jxz, Jyz, Jzz = symbols("Jxx Jxy Jyy Jxz Jyz Jzz")

w = Matrix([wx, wy, wz])
dw = Matrix([dwx, dwy, dwz])
i = Matrix([Ixx, Ixy, Iyy, Ixz, Iyz, Izz])
I = Matrix([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

wfx, wfy, wfz, dwfx, dwfy, dwfz = symbols("wfx wfy wfz dwfx dwfy dwfz")
wf = Matrix([wfx, wfy, wfz])
dwf = Matrix([dwfx, dwfy, dwfz])
J = Matrix([[Jxx, Jxy, Jxz], [Jxy, Jyy, Jyz], [Jxz, Jyz, Jzz]])
j = Matrix([Jxx, Jxy, Jyy, Jxz, Jyz, Jzz])

# torque-free equations of motion, left hand side
lhs = I @ dw  +  w.cross(I @ w)

# problem matrix
zetai = lhs.jacobian(i)
symprint(zetai, "zeta_i")


etai = (-J @ dwf - w.cross(J @ wf)).jacobian(j)
symprint(etai, "eom")

flywheel_inertia = -zetai @ i @ etai.inv()
symprint(flywheel_inertia, "flywheel_inertia")
