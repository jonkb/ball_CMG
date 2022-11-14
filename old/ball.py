"""
Finding EOM for a ball on a flat surface
"""
import numpy as np
import sympy as sp
from sympy.algebras.quaternion import Quaternion
from sympy.functions.elementary.complexes import conjugate
import scipy.integrate as spi
import pickle

## Utility functions
def sharp(v):
  """ 
  "Sharp" operator in Putkaradze paper
  Takes an R3 vector and turns it into a quaternion
  """
  return Quaternion(0, v[0], v[1], v[2])
def flat(q):
  """ 
  "Flat" operator in Putkaradze paper
  Takes a quaternion and turns it into an R3 vector
  """
  return sp.Matrix([[q.b], [q.c], [q.d]])

## Setup
print("Setting up symbolic variables")
Rs, m, Is = sp.symbols("Rs, m, Is") # Radius, mass, Inertia
I_s__s = sp.eye(3) * Is
# w, x, y, z = sp.symbols("w, x, y, z")
# wd, xd, yd, zd = sp.symbols("wd, xd, yd, zd")
t = sp.symbols("t")
rx__0 = sp.Function("rx__0")(t)
ry__0 = sp.Function("ry__0")(t)
# Use omega_x, y, & z as generalized coordinates. Assumes flat surface.
omega_x = sp.Function("omega_x")(t)
omega_y = sp.Function("omega_y")(t)
omega_z = sp.Function("omega_z")(t)
omega_xd = sp.diff(omega_x, t)
omega_yd = sp.diff(omega_y, t)
omega_zd = sp.diff(omega_z, t)
omega_xdd = sp.diff(omega_x, (t,2))
omega_ydd = sp.diff(omega_y, (t,2))
omega_zdd = sp.diff(omega_z, (t,2))
# Generalized forces: body-fixed torques
Tau_x, Tau_y, Tau_z = sp.symbols("Tau_x, Tau_y, Tau_z")
# Define orientation with a quaternion
nu = sp.Function("nu")(t)
ex = sp.Function("ex")(t)
ey = sp.Function("ey")(t)
ez = sp.Function("ez")(t)
q = Quaternion(nu, ex, ey, ez)
# Angular velocity
#_s: of sphere; __s: in sphere frame; __0 in global frame
omega_s__s = sp.Matrix([[omega_x], [omega_y], [omega_z]])
# Rotate omega_s into global frame
omega_s__0 = flat((q * sharp(omega_s__s) * conjugate(q)).expand())
# Linear velocities
rxd__0 = Rs*omega_s__0[1,0] #These come from (-Rk) x Omega
ryd__0 = -Rs*omega_s__0[0,0]
# [Z-velocity rzd__0 = 0]

## Write Lagrangian
# Kinetic Energy
T = 1/2*m*(rxd__0**2 + ryd__0**2)
T += (1/2*omega_s__s.T * I_s__s * omega_s__s)[0,0]
T = T.expand()
# print(T)
L = T # V = 0
#print("L: ", L)

## Lagrangian Derivatives
print("Finding EOM w/ Lagrangian Mechanics")
""" This would be if omega were a generalized coordinate
# q1-dot = omega_x
dL_domega_xd = sp.diff(L, omega_xd)
d_dt_dL_domega_xd = sp.diff(dL_domega_xd, t)
dL_domega_x = sp.diff(L, omega_x)
EOM1 = d_dt_dL_domega_xd - dL_domega_x - Tau_x
EOM1 = EOM1.expand()
# q2 = omega_y
dL_domega_yd = sp.diff(L, omega_yd)
d_dt_dL_domega_yd = sp.diff(dL_domega_yd, t)
dL_domega_y = sp.diff(L, omega_y)
EOM2 = d_dt_dL_domega_yd - dL_domega_y - Tau_y
EOM2 = EOM2.expand()
# q3 = omega_z
dL_domega_zd = sp.diff(L, omega_zd)
d_dt_dL_domega_zd = sp.diff(dL_domega_zd, t)
dL_domega_z = sp.diff(L, omega_z)
EOM3 = d_dt_dL_domega_zd - dL_domega_z - Tau_z
EOM3 = EOM3.expand()
"""

# q1d = omega_x
dL_dq1d = sp.diff(L, omega_x)
d_dt_dL_dq1d = sp.diff(dL_dq1d, t)
dL_dq1 = 0 #sp.diff(L, int omega_x) #ASSUME
EOM1 = d_dt_dL_dq1d - dL_dq1 - Tau_x
# q2d = omega_y
dL_dq2d = sp.diff(L, omega_y)
d_dt_dL_dq2d = sp.diff(dL_dq2d, t)
dL_dq2 = 0 #sp.diff(L, int omega_y) #ASSUME
EOM2 = d_dt_dL_dq2d - dL_dq2 - Tau_y
# q3d = omega_z
dL_dq3d = sp.diff(L, omega_z)
d_dt_dL_dq3d = sp.diff(dL_dq3d, t)
dL_dq3 = 0 #sp.diff(L, int omega_z) #ASSUME
EOM3 = d_dt_dL_dq3d - dL_dq3 - Tau_z

# Substitute q-dot in terms of omega
qdot = sharp(omega_s__0) * q / 2
qdsubs = {sp.diff(nu,t): qdot.a, sp.diff(ex,t): qdot.b, 
  sp.diff(ey,t): qdot.c, sp.diff(ez,t): qdot.d}
EOM1 = EOM1.subs(qdsubs).expand()
EOM2 = EOM2.subs(qdsubs).expand()
EOM3 = EOM3.subs(qdsubs).expand()
# print("EOM1: ", EOM1)
# print("EOM2: ", EOM2)
# print("EOM3: ", EOM3)
# quit()

# Convert 3 equations to matrix form.

print("Writing in matrix form")

M = sp.zeros(3)
M[0,0] = EOM1.coeff(omega_xd) # omega_xd = q1dd
M[0,1] = EOM1.coeff(omega_yd)
M[0,2] = EOM1.coeff(omega_zd)
EOM1 -= (M[0,0]*omega_xd + M[0,1]*omega_yd + M[0,2]*omega_zd).expand()
#EOM1 = EOM1.expand().simplify()
M[1,0] = EOM2.coeff(omega_xd)
M[1,1] = EOM2.coeff(omega_yd)
M[1,2] = EOM2.coeff(omega_zd)
EOM2 -= (M[1,0]*omega_xd + M[1,1]*omega_yd + M[1,2]*omega_zd).expand()
#EOM2 = EOM2.expand().simplify()
M[2,0] = EOM3.coeff(omega_xd)
M[2,1] = EOM3.coeff(omega_yd)
M[2,2] = EOM3.coeff(omega_zd)
EOM3 -= (M[2,0]*omega_xd + M[2,1]*omega_yd + M[2,2]*omega_zd).expand()
#EOM3 = EOM3.expand().simplify()
M = M.applyfunc(lambda x: x.simplify())
# print("M: ", M)

C = sp.zeros(3)
C[0,0] = EOM1.coeff(omega_x) # SHORTCUT
C[0,1] = EOM1.coeff(omega_y)
C[0,2] = EOM1.coeff(omega_z)
EOM1 -= (C[0,0]*omega_x + C[0,1]*omega_y + C[0,2]*omega_z).expand()
EOM1 = EOM1.expand().simplify()
C[1,0] = EOM2.coeff(omega_x)
C[1,1] = EOM2.coeff(omega_y)
C[1,2] = EOM2.coeff(omega_z)
EOM2 -= (C[1,0]*omega_x + C[1,1]*omega_y + C[1,2]*omega_z).expand()
EOM2 = EOM2.expand().simplify()
C[2,0] = EOM3.coeff(omega_x)
C[2,1] = EOM3.coeff(omega_y)
C[2,2] = EOM3.coeff(omega_z)
EOM3 -= (C[2,0]*omega_x + C[2,1]*omega_y + C[2,2]*omega_z).expand()
EOM3 = EOM3.expand().simplify()
C = C.applyfunc(lambda x: x.simplify())
# print("C: ", C)

# SHORTCUT explained: I know that K is zero
K = sp.zeros(3) # SHORTCUT
# K[0,0] = EOM1.coeff(omega_x)
# K[0,1] = EOM1.coeff(omega_y)
# K[0,2] = EOM1.coeff(omega_z)
# K[1,0] = EOM2.coeff(omega_x)
# K[1,1] = EOM2.coeff(omega_y)
# K[1,2] = EOM2.coeff(omega_z)
# K[2,0] = EOM3.coeff(omega_x)
# K[2,1] = EOM3.coeff(omega_y)
# K[2,2] = EOM3.coeff(omega_z)
# print("K: ", K)

# Convert to state form.
print("Lambdifying EOM matrices")
# See https://docs.sympy.org/latest/modules/utilities/lambdify.html
# And https://docs.sympy.org/latest/modules/numeric-computation.html

# State variables x
consts = {Is: .001, m: 1, Rs: 0.05}
Rs = 0.05 # Used in xdot function
M = M.subs(consts)
C = C.subs(consts)
K = K.subs(consts)
# print("M: ", M)
# print("C: ", C)
# print("K: ", K)
# quit()

xs = (nu, ex, ey, ez, omega_x, omega_y, omega_z, rx__0, ry__0) # -s in xs for symbolic
Mf = sp.lambdify(xs, M, "numpy")
Cf = sp.lambdify(xs, C, "numpy")
Kf = sp.lambdify(xs, K, "numpy") # Note: K=0
# print("Mf(0): ", Mf(*np.zeros(9)))
# print("Cf(0): ", Cf(*np.zeros(9)))
# print("Kf(0): ", Kf(*np.zeros(9)))
# quit()

# Input Torques
Tau_x = lambda t: .1 if (t<.5) else 0
Tau_y = lambda t: -.1 if (t>1 and t<1.5) else 0
Tau_z = lambda t: 0

def xdot(t, x):
  """ State variable EOM
  x is a (9,) numpy array of the state variables
  x = (nu, ex, ey, ez, omega_x, omega_y, omega_z, rx__0, ry__0)
  """
  
  xd = np.zeros(9)
  
  # Equations 1-4: Orientation quaternion:
  q = Quaternion(x[0], x[1], x[2], x[3]) # NOTE: Maybe this should be switched to numpy
  omega_s__s = [x[4], x[5], x[6]]
  omega_s__0 = flat((q * sharp(omega_s__s) * conjugate(q)).expand())
  qdot = sharp(omega_s__0) * q / 2
  xd[0] = float(qdot.a)
  xd[1] = float(qdot.b)
  xd[2] = float(qdot.c)
  xd[3] = float(qdot.d)
  
  # Equations 5-7: omega-dot from EOM. TODO: Check is this right?
  M = Mf(*x)
  C = Cf(*x)
  K = Kf(*x)
  F = np.array([Tau_x(t), Tau_y(t), Tau_z(t)])
  b = F.T - np.dot(C, x[4:7].T) # NO K: - np.dot(K, x[0:3].T)
  # Using linalg.solve instead of inverting M
  xd[4:7], *_ = np.linalg.lstsq(M, b)
  
  # Equations 8-9: rx, ry
  xd[7] = Rs*omega_s__0[1,0] #These come from (-Rk) x Omega
  xd[8] = -Rs*omega_s__0[0,0]
  
  # print("t:",t)
  # print("x:",x)
  # print("xd:",xd)
  return xd


print("Solving IVP")
x0 = [1,0,0,0,0,0,0,0,0]
print("xdot(0): ", xdot(0,np.zeros(9)))
sol = spi.solve_ivp(xdot, [0,2], x0, dense_output=True, rtol=1e-4, atol=1e-7)

with open("sol.pickle","wb") as file:
  pickle.dump(sol, file)
  
print("DONE")
