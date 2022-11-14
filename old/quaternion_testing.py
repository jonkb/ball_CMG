"""
Trying to learn how to use quaternions for dynamics
Run this script in Ubuntu because of the VizScene
"""

import time
import sympy as sp
from sympy.algebras.quaternion import Quaternion
from sympy.functions.elementary.complexes import conjugate
import numpy as np
import quaternion as qtr
# These packages are from ME 537
#from visualization import VizScene
import transforms as tr

sp.init_printing()

symbolic = True
if symbolic:
  w, x, y, z, theta = sp.symbols("w, x, y, z, theta")
  wd, xd, yd, zd = sp.symbols("wd, xd, yd, zd")
  q = Quaternion(w,x,y,z)
  q_rotz = Quaternion.from_axis_angle((0,0,1), theta)
  sp.pprint(q_rotz, use_unicode=False)
  sp.pprint(q_rotz.to_rotation_matrix().applyfunc(lambda x: x.simplify()), use_unicode=False)
  
  
  quit()
  qdot = Quaternion(wd,xd,yd,zd)
  #t = sp.symbols("t")
  #w = sp.Function("w")
  #x = sp.Function("x")
  #y = sp.Function("y")
  #z = sp.Function("z")
  #q = Quaternion(w(t),x(t),y(t),z(t))
  #qdot = sp.diff(q, t)
  omega = (2*qdot*conjugate(q)).vector_part()
  print(q)
  print(qdot)#, use_unicode=False)
  print("omega_s__0", omega)#, use_unicode=False)
  omega_s__s = conjugate(q) * omega * q
  omega_s__s = omega_s__s.simplify().vector_part()
  print("omega_s__s", omega_s__s)

misc = False
if misc:
  # Testing
  q0 = np.quaternion(1,0,0,0)
  rotx90 = tr.rotx(np.pi/2) # Counter-rotates the frame about x by 90 (or rotates a vector by 90)
  qrx90 = tr.R2q(rotx90)
  q1 = np.quaternion(*qrx90)
  print("Rot_x(90) R:\n", rotx90)
  print("Rot_x(90) Q:", qrx90)
  print("R @ y_1__1:", rotx90 @ np.array([0,1,0]))

  y_1__1 = np.quaternion(0,0,1,0) # y1 in the 1 frame
  # https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
  y_1__0 = q1 * y_1__1 * q1.inverse() # "Passive rotation"
  print("y_1__0", y_1__0)

  # Rot_x(90)
  #q1 = np.quaternion(np.sqrt(2), np.sqrt(2), 0, 0)
  print("q0:",q0)
  print("q1:",q1)
  print("q0*q1:",q0*q1)
  print("q1*q0:",q1*q0)
  print("q1 conj:", np.conjugate(q1))
  print("q1 conj:", q1.conjugate())

spin = False
if spin:
  # Rotate slowly about x1
  N = 200
  fps = 30
  dt = 1/fps
  th = np.vstack(np.linspace(0, np.pi/2, N))
  print("omega_x = ", np.pi/2 / (N*dt))
  #th *= th # Acclerate
  # Animate
  viz = VizScene()
  viz.add_frame(np.eye(4))
  q_im1 = np.quaternion(1,0,0,0) # q_{i-1}
  for i in range(N):
    t = dt*i
    R_i = tr.rotz(np.pi/2) @ tr.rotx(th[i])
    q_i = qtr.from_rotation_matrix(R_i)
    if i % 10 == 1:
      # Calculate Omega according to
      #   Omega = 2*qdot*q.conjugate
      qdot = (q_i - q_im1) / dt
      omega = qtr.as_vector_part( 2*qdot*q_i.conjugate() )
      print("omega ~=", omega)
    q_im1 = q_i # Update previous
    # Update display
    T = tr.se3(R=R_i)
    viz.update(As=[T])
    time.sleep(dt)
  viz.close_viz()


  quit() # OLD
  # Rotation vector (axis-angle, with norm = th)
  vrot = np.hstack([th, np.zeros((N,1)), np.zeros((N,1))])
  qs = qtr.from_rotation_vector(vrot)
  print("omega_x = ", np.pi/2 / (N*dt))

  # Animate
  viz = VizScene()
  viz.add_frame(np.eye(4))
  for i, q_i in enumerate(qs):
    if i % 10 == 1:
      # Calculate Omega according to
      #   Omega = 2*qdot*q.conjugate
      qdot = (q_i - qs[i-1]) / dt
      omega = qtr.as_vector_part( 2*qdot*q_i.conjugate() )
      print("omega ~=", omega)
    # Update display
    R = qtr.as_rotation_matrix(q_i) 
    T = tr.se3(R=R)
    viz.update(As=[T])
    time.sleep(dt)

  viz.close_viz()

