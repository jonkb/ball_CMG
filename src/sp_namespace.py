"""
Initializes a bunch of sympy objects used in the dynamics of the CMG ball.
To be imported as follows:
  import sp_namespase as spn
"""

import sympy as sp
from sympy.algebras.quaternion import Quaternion
from sympy.functions.elementary.complexes import conjugate
from util import sharp, flat, printv

printv(1, "Setting up symbolic variables")

g, rax, ray, raz = sp.symbols("g, rax, ray, raz") # gravity, accel position in a-frame
ra = sp.Matrix([[rax], [ray], [raz]])

Rs, m, Is, Ig1, Ig2 = sp.symbols("Rs, m, Is, Ig1, Ig2") # Radius, mass, Inertia
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
omega_xdd = sp.diff(omega_x, (t,2)) # Irrelevant
omega_ydd = sp.diff(omega_y, (t,2))
omega_zdd = sp.diff(omega_z, (t,2))
# Define orientation with a quaternion
eta = sp.Function("eta")(t)
ex = sp.Function("ex")(t)
ey = sp.Function("ey")(t)
ez = sp.Function("ez")(t)
etad = sp.diff(eta,t)
exd = sp.diff(ex,t)
eyd = sp.diff(ey,t)
ezd = sp.diff(ez,t)
# q: active rotation from 0 to s or passive rotation from s to 0
q = Quaternion(eta, ex, ey, ez)
# Angular velocity
#_s: of sphere; __s: in sphere frame; __0 in global frame
omega_s__s = sp.Matrix([[omega_x], [omega_y], [omega_z]])
# Rotate omega_s into global frame
omega_s__0 = flat((q * sharp(omega_s__s) * conjugate(q)).expand())
# Linear velocities of sphere COM. 
rxd__0 = Rs*omega_s__0[1,0] #These come from (-Rk) x Omega
ryd__0 = -Rs*omega_s__0[0,0]
# [Z-velocity rzd__0 = 0]

# Gyroscope setup (inertia, speed, angle, rotations)
I_g__g = sp.diag(Ig1, Ig2, Ig2)
Omega_g = sp.symbols("Omega_g") # Constant angular velocity of gyro about x_g
alpha = sp.Function("alpha")(t) # Angle btw x_s and x_g, rotating about z_s
alphad = sp.diff(alpha, t)
alphadd = sp.diff(alphad, t)
omega_g__s = omega_s__s + sp.Matrix([[Omega_g*sp.cos(alpha)], [Omega_g*sp.sin(alpha)], [alphad]])
omega_g__0 = flat((q * sharp(omega_g__s) * conjugate(q)).expand())

# q_sa: Passive rotation from s to a
#   from_rotation_vector uses active convention
q_sa = Quaternion.from_axis_angle((0,0,1), -alpha)
# OLD: Rotate from sphere frame to gyro frame (active?)
# q_g__s = Quaternion.from_axis_angle((0,0,1), alpha)
# NOTE: This is the a-frame, not really the g-frame
omega_g__g = flat((q_sa * sharp(omega_g__s) * conjugate(q_sa)).expand())

# Symbolic state vector (-s in xs for "symbolic")
xs = (eta, ex, ey, ez, omega_x, omega_y, omega_z, rx__0, ry__0, alpha, alphad, 
  alphadd)
# Substitution vector for ax & ay
s_axay = (eta, ex, ey, ez, omega_x, omega_y, omega_z, etad, exd, eyd, ezd, 
  omega_xd, omega_yd, omega_zd)