"""
Finding EOM for ball_CMG

This formulation used epsilon as a generalized coordinate, and that broke after 180deg of rotation. Instead, do it like they did it here: https://link.springer.com/article/10.1007/s11012-018-0904-5#Sec5.

"""
import numpy as np
import sympy as sp
from sympy.algebras.quaternion import Quaternion
from sympy.functions.elementary.complexes import conjugate
import scipy.integrate as spi
import pickle

## Setup
Rs, m, Is = sp.symbols("Rs, m, Is") # Radius, mass, Inertia
I_s__s = sp.eye(3) * Is
# w, x, y, z = sp.symbols("w, x, y, z")
# wd, xd, yd, zd = sp.symbols("wd, xd, yd, zd")
t = sp.symbols("t")
ex = sp.Function("ex")(t)
ey = sp.Function("ey")(t)
ez = sp.Function("ez")(t)
nu = sp.sqrt(1 - ex**2 - ey**2 - ez**2)
exd = sp.diff(ex, t)
eyd = sp.diff(ey, t)
ezd = sp.diff(ez, t)
nud = sp.diff(nu, t)
q = Quaternion(nu, ex, ey, ez)
qdot = Quaternion(nud, exd, eyd, ezd)
# Generalized forces applid in the quaternion directions...
F_ex, F_ey, F_ez = sp.symbols("F_ex, F_ey, F_ez")

# Angular velocities
omegaQ_s__0 = (2*qdot*conjugate(q)).vector_part()
omega_s__0 = sp.Matrix([[omegaQ_s__0.b], [omegaQ_s__0.c], [omegaQ_s__0.d]])
omegaQ_s__s = (conjugate(q) * omegaQ_s__0 * q).simplify().vector_part()
omega_s__s = sp.Matrix([[omegaQ_s__s.b], [omegaQ_s__s.c], [omegaQ_s__s.d]])
# Linear velocities
rxd__0 = Rs*omega_s__0[1,0] #These come from (-Rk) x Omega
ryd__0 = -Rs*omega_s__0[0,0]
# [Z-velocity rzd__0 = 0]

## Assume on a flat surface

# Kinetic Energy
T = 1/2*m*(rxd__0**2 + ryd__0**2)
T += (1/2*omega_s__s.T * I_s__s * omega_s__s)[0,0]
T = T.expand()
# print(T)
L = T # V = 0
#print("L: ", L)

## Lagrangian Derivatives
print("Finding EOM w/ Lagrangian Mechanics")
# q1 = ex
dL_dexd = sp.diff(L, exd)
d_dt_dL_dexd = sp.diff(dL_dexd, t)
dL_dex = sp.diff(L, ex)
EOM1 = d_dt_dL_dexd - dL_dex - F_ex
EOM1 = EOM1.expand()
# q2 = ey
dL_deyd = sp.diff(L, eyd)
d_dt_dL_deyd = sp.diff(dL_deyd, t)
dL_dey = sp.diff(L, ey)
EOM2 = d_dt_dL_deyd - dL_dey - F_ey
EOM2 = EOM2.expand()
# q3 = ez
dL_dezd = sp.diff(L, ezd)
d_dt_dL_dezd = sp.diff(dL_dezd, t)
dL_dez = sp.diff(L, ez)
EOM3 = d_dt_dL_dezd - dL_dez - F_ez
EOM3 = EOM3.expand()

# Convert 3 equations to matrix form.

print("Writing in matrix form")
exdd = sp.diff(ex, (t,2))
eydd = sp.diff(ey, (t,2))
ezdd = sp.diff(ez, (t,2))

M = sp.zeros(3)
M[0,0] = EOM1.coeff(exdd)
M[0,1] = EOM1.coeff(eydd)
M[0,2] = EOM1.coeff(ezdd)
EOM1 -= (M[0,0]*exdd + M[0,1]*eydd + M[0,2]*ezdd).expand()
#EOM1 = EOM1.expand().simplify()
M[1,0] = EOM2.coeff(exdd)
M[1,1] = EOM2.coeff(eydd)
M[1,2] = EOM2.coeff(ezdd)
EOM2 -= (M[1,0]*exdd + M[1,1]*eydd + M[1,2]*ezdd).expand()
#EOM2 = EOM2.expand().simplify()
M[2,0] = EOM3.coeff(exdd)
M[2,1] = EOM3.coeff(eydd)
M[2,2] = EOM3.coeff(ezdd)
EOM3 -= (M[2,0]*exdd + M[2,1]*eydd + M[2,2]*ezdd).expand()
#EOM3 = EOM3.expand().simplify()
#print("M: ", M)

# SHORTCUT explained: I know that C & K are zero

C = sp.zeros(3)
# C[0,0] = EOM1.coeff(exd) # SHORTCUT
# C[0,1] = EOM1.coeff(eyd)
# C[0,2] = EOM1.coeff(ezd)
# EOM1 -= (C[0,0]*exd + C[0,1]*eyd + C[0,2]*ezd).expand()
# EOM1 = EOM1.expand().simplify()
# C[1,0] = EOM2.coeff(exd)
# C[1,1] = EOM2.coeff(eyd)
# C[1,2] = EOM2.coeff(ezd)
# EOM2 -= (C[1,0]*exd + C[1,1]*eyd + C[1,2]*ezd).expand()
# EOM2 = EOM2.expand().simplify()
# C[2,0] = EOM3.coeff(exd)
# C[2,1] = EOM3.coeff(eyd)
# C[2,2] = EOM3.coeff(ezd)
# EOM3 -= (C[2,0]*exd + C[2,1]*eyd + C[2,2]*ezd).expand()
# EOM3 = EOM3.expand().simplify()
#print("C: ", C)

K = sp.zeros(3)
# K[0,0] = EOM1.coeff(ex) # SHORTCUT
# K[0,1] = EOM1.coeff(ey)
# K[0,2] = EOM1.coeff(ez)
# K[1,0] = EOM2.coeff(ex)
# K[1,1] = EOM2.coeff(ey)
# K[1,2] = EOM2.coeff(ez)
# K[2,0] = EOM3.coeff(ex)
# K[2,1] = EOM3.coeff(ey)
# K[2,2] = EOM3.coeff(ez)
#print("K: ", K)

# Convert to state form. xdot = [q; qdot]
print("Lambdifying EOM matrices")
# See https://docs.sympy.org/latest/modules/utilities/lambdify.html
# And https://docs.sympy.org/latest/modules/numeric-computation.html

# State variables x
consts = {Is: .001, m: 1, Rs: 0.05}
M = M.subs(consts)
C = C.subs(consts)
K = K.subs(consts)
# print("M: ", M)
# print("C: ", C)
# print("K: ", K)
xs = (ex, ey, ez, sp.diff(ex, t), sp.diff(ey, t), sp.diff(ez, t)) # s for symbolic
Mf = sp.lambdify(xs, M, "numpy")
Cf = sp.lambdify(xs, C, "numpy")
Kf = sp.lambdify(xs, K, "numpy") # Note: K=0
# print("Kf(0): ", Kf(*np.zeros(6)))

# Input "Torques" (they're generalized forces in the directions of the quaternions)
F_ex = lambda t: 1 if (t<.01) else 0
F_ey = lambda t: -1 if (t>.1 and t<.11) else 0
F_ez = lambda t: 0

def xdot(t, x):
  """ State variable EOM
  x is a (6,) numpy array of the state variables
  x[0:3] = q1,q2,q3
  x[3:6] = q1d, q2d, q3d
  """
  print(t)
  xd = np.zeros(6)
  xd[0:3] = x[3:6] # Equations 1-3
  M = Mf(*x)
  C = Cf(*x)
  K = Kf(*x)
  F = np.array([F_ex(t), F_ey(t), F_ez(t)])
  b = F.T - np.dot(C, x[3:6].T) - np.dot(K, x[0:3].T)
  # Using linalg.solve instead of inverting M
  xd[3:6], *_ = np.linalg.lstsq(M, b)
  return xd

print("xdot(0,0): ", xdot(0,np.zeros(6)))

print("Solving IVP")
x0 = np.zeros(6)
sol = spi.solve_ivp(xdot, [0,.195], x0)

with open("sol.pickle","wb") as file:
  pickle.dump(sol, file)
  
print("DONE")
