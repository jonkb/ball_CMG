""" Dynamics math
"""

import sympy as sp
from sympy.algebras.quaternion import Quaternion
from sympy.functions.elementary.complexes import conjugate

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

t = sp.symbols("t")
# Define orientation with a quaternion
eta, ex, ey, ez = sp.symbols("eta, ex, ey, ez")
etad, exd, eyd, ezd = sp.symbols("etad, exd, eyd, ezd")
if False:
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
qd = Quaternion(etad, exd, eyd, ezd)
#qd = sp.diff(q,t)

# -- Calculate the angular velocity as a function of Q & Qd --

#omega_s__s = 2*flat(qd * conjugate(q)) # OLD
omega_s__s = 2*flat(conjugate(q) * qd)
# Rotate omega_s into global frame
omega_s__0 = flat((q * sharp(omega_s__s) * conjugate(q)).expand())

print(22, omega_s__s)
#print(40, omega_s__0)
#sp.pprint(omega_s__0)
#omega_s__0.simplify()
#omega_s__0 = omega_s__0.applyfunc(lambda x: x.expand().simplify())
#print(45, omega_s__0)

ws0x = sp.collect(omega_s__0[0,0], [etad, exd, eyd, ezd])
print("omega_s__0 (x) = ")
#sp.pprint(omega_s__0[0,0])
print(ws0x)
ws0y = sp.collect(omega_s__0[1,0], [etad, exd, eyd, ezd])
print("omega_s__0 (y) = ")
print(ws0y)
