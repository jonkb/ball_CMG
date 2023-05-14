"""
Derive the equations of motion for the robot. It's a sphere with a Control Moment Gyroscope (CMG) mounted inside.

The strategy behind the dynamics was mostly adapted from the following paper by Putkaradze & Rogers:
https://link.springer.com/article/10.1007/s11012-018-0904-5#Sec11

TODO: Change the folder where things are saved by default... maybe.
"""

import sympy as sp
from sympy.algebras.quaternion import Quaternion
from sympy.functions.elementary.complexes import conjugate
from util import sharp, flat, printv

def derive_EOM(save=True):
  """ Derive the equations of motion and save them to file.
  
  The equations of motion are in the format
    M@qdd = F
    where qdd = d/dt([omega_x, omega_y, omega_z])
  
  Parameters
  ----------
  save (bool): Whether to save M & F to file (default True)
  """
  
  ## Setup (symbolic variables, rotations, velocities)
  printv(1, "Setting up symbolic variables")
  import sp_namespace as spn
  
  ## Write Lagrangian
  # Kinetic Energy
  T = 1/2*spn.m*(spn.rxd__0**2 + spn.ryd__0**2)
  T += (1/2*spn.omega_s__s.T * spn.I_s__s * spn.omega_s__s)[0,0]
  T += (1/2*spn.omega_g__g.T * spn.I_g__g * spn.omega_g__g)[0,0]
  T = T.expand()
  printv(2, "T: ", T)
  L = T # V = 0 (flat surface)
  printv(2, "L: ", L)

  ## Lagrangian Derivatives
  printv(1, "Finding EOM w/ Lagrangian Mechanics")

  # q1d = omega_x
  dL_dq1d = sp.diff(L, spn.omega_x)
  d_dt_dL_dq1d = sp.diff(dL_dq1d, spn.t)
  # ASSUME no change in T or V with position or orientation
  dL_dq1 = 0 #sp.diff(L, int omega_x)
  EOM1 = d_dt_dL_dq1d - dL_dq1# - Tau_x
  # q2d = omega_y
  dL_dq2d = sp.diff(L, spn.omega_y)
  d_dt_dL_dq2d = sp.diff(dL_dq2d, spn.t)
  dL_dq2 = 0 #sp.diff(L, int omega_y) #ASSUME - I'm not so sure about this...
  EOM2 = d_dt_dL_dq2d - dL_dq2# - Tau_y
  # q3d = omega_z
  dL_dq3d = sp.diff(L, spn.omega_z)
  d_dt_dL_dq3d = sp.diff(dL_dq3d, spn.t)
  dL_dq3 = 0 #sp.diff(L, int omega_z) #ASSUME
  EOM3 = d_dt_dL_dq3d - dL_dq3# - Tau_z
  # Should I have an equation for alpha? Probably not; it's an input...

  """ This (OLD) version would be if omega were a generalized coordinate 
    instead of a generalized velocity. It also uses body-fixed torque inputs.
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

  # Substitute q-dot (time derivative of quaternion) in terms of omega
  qdot = sharp(spn.omega_s__0) * spn.q / 2
  qdsubs = {spn.etad: qdot.a, spn.exd: qdot.b, spn.eyd: qdot.c, 
    spn.ezd: qdot.d}
  EOM1 = EOM1.subs(qdsubs).expand()
  EOM2 = EOM2.subs(qdsubs).expand()
  EOM3 = EOM3.subs(qdsubs).expand()
  printv(2, "EOM1: ", EOM1)
  printv(2, "EOM2: ", EOM2)
  printv(2, "EOM3: ", EOM3)
  # quit()

  ## Convert 3 equations to matrix form.
  printv(1, "Writing in matrix form: [M]{xdot} = {F}")

  printv(1, "\tBuilding M matrix")
  M = sp.zeros(3)
  M[0,0] = EOM1.coeff(spn.omega_xd) # omega_xd = q1dd
  M[0,1] = EOM1.coeff(spn.omega_yd)
  M[0,2] = EOM1.coeff(spn.omega_zd)
  EOM1 -= (M[0,0]*spn.omega_xd + M[0,1]*spn.omega_yd + 
    M[0,2]*spn.omega_zd).expand()
  M[1,0] = EOM2.coeff(spn.omega_xd)
  M[1,1] = EOM2.coeff(spn.omega_yd)
  M[1,2] = EOM2.coeff(spn.omega_zd)
  EOM2 -= (M[1,0]*spn.omega_xd + M[1,1]*spn.omega_yd + 
    M[1,2]*spn.omega_zd).expand()
  M[2,0] = EOM3.coeff(spn.omega_xd)
  M[2,1] = EOM3.coeff(spn.omega_yd)
  M[2,2] = EOM3.coeff(spn.omega_zd)
  EOM3 -= (M[2,0]*spn.omega_xd + M[2,1]*spn.omega_yd + 
    M[2,2]*spn.omega_zd).expand()
  printv(1, "\t\tSimplifying")
  M = M.applyfunc(lambda x: x.expand().simplify())
  printv(2, "M: ", M)
  
  printv(1, "\tBuilding F vector")
  F = sp.zeros(3,1)
  F[0] = -EOM1
  F[1] = -EOM2
  F[2] = -EOM3
  printv(1, "\t\tSimplifying")
  F = F.applyfunc(lambda x: x.simplify())
  
  if save:
    ## Save equations to file (plain-text sympy srepr format).
    ##   Saving with pickle or dill didn't work
    printv(1, "Saving to file")
    with open("M.srepr","w") as file:
      M_str = sp.srepr(M)
      file.write(M_str)
    with open("F.srepr","w") as file:
      F_str = sp.srepr(F)
      file.write(F_str)
    printv(1, "M & F saved to file")
  
  return M, F

def lambdify_MF(M, F, ball):
  """ Lambdify M & F
  The numerical arguments are constants to be substituted into M & F
    before converting them to numpy lambdified functions.
  See sp_namespace.xs for the input vector of the returned functions.
  
  See https://docs.sympy.org/latest/modules/utilities/lambdify.html
  And https://docs.sympy.org/latest/modules/numeric-computation.html
  
  NOTE: This should maybe be in the CBGBall class, but that'd mean importing
  sympy and sp_namespace there as well...
  """

  printv(1, "Lambdifying EOM")
  import sp_namespace as spn

  # Make the substitutions for everything that's not a state variable
  consts = {spn.Is: ball.Is, spn.Ig1: ball.Ig1, spn.Ig2: ball.Ig2, 
    spn.m: ball.m, spn.Rs: ball.Rs, spn.Omega_g: ball.Omega_g}
  M = M.subs(consts)
  F = F.subs(consts)

  # Lambdified functions for M and F
  Mf = sp.lambdify(spn.xs, M, "numpy") # spn.xs: symbolic state vector
  Ff = sp.lambdify(spn.xs, F, "numpy")

  return Mf, Ff

def load_MF():
  """ Load M & F from M.srepr, F.srepr
  Assumes those files exist
  """
  
  with open("M.srepr", "r") as file:
    M_str = file.read()
    M = sp.sympify(M_str)
  with open("F.srepr", "r") as file:
    F_str = file.read()
    F = sp.sympify(F_str)
  return M, F

def find_axay(M, F, save=True):
  """ Find cartesian acceleration in the world frame.
  rx-dot = Rs*omega_s__0[1,0]
  ry-dot = -Rs*omega_s__0[0,0]
  """
  
  printv(1, "Solving for acceleration")
  import sp_namespace as spn
  # Differentiate
  ax = sp.diff(spn.rxd__0, spn.t)
  ay = sp.diff(spn.ryd__0, spn.t)
  
  if save:
    printv(1, "Saving to file")
    with open("ax.srepr","w") as file:
      file.write(sp.srepr(ax))
    with open("ay.srepr","w") as file:
      file.write(sp.srepr(ay))
    printv(1, "ax & ay saved to file")
  
  return ax, ay

def load_axay():
  """ Load ax & ay from ax.srepr, ay.srepr
  Assumes those files exist
  """
  
  with open("ax.srepr", "r") as file:
    ax = sp.sympify(file.read())
  with open("ay.srepr", "r") as file:
    ay = sp.sympify(file.read())
  return ax, ay

def lambdify_axay(ax, ay, ball):
  printv(1, "Lambdifying accelerations")
  import sp_namespace as spn

  # Make the substitutions for everything that's not a state variable
  consts = {spn.Rs: ball.Rs}
  ax = ax.subs(consts)
  ay = ay.subs(consts)

  # Lambdified functions for M and F
  axf = sp.lambdify(spn.s_axay, ax, "numpy") # spn.xs: symbolic state vector
  ayf = sp.lambdify(spn.s_axay, ay, "numpy")

  return axf, ayf

if __name__ == "__main__":
  sp.init_printing(use_unicode=False, num_columns=180) 
  import sp_namespace as spn
  M, F = load_MF()
  
  # Investigate some things about them
  # print(f"M:\n\tSymbols: {M.atoms(sp.Symbol)}")
  # print(f"\tFree symbols: {M.free_symbols}")
  # print(f"\tFunctions: {M.atoms(sp.Function)}")
  # print(f"\tDerivatives: {M.has(sp.core.function.Derivative)}")
  #print(f"\tVariables: {M.variables}")
  # print(f"F:\n\tSymbols: {F.atoms(sp.Symbol)}")
  # print(f"\tFree symbols: {F.free_symbols}")
  # print(f"\tFunctions: {F.atoms(sp.Function)}")
  # print(f"\tDerivatives: {M.has(sp.core.function.Derivative)}")
  
  # See which of spn.[] variables are in each
  svars_in_M = []
  svars_in_F = []
  for svar in dir(spn):
    var = getattr(spn, svar)
    if M.has(var):
      svars_in_M.append(svar)
    if F.has(var):
      svars_in_F.append(svar)
  print(f"M has: {', '.join(svars_in_M)}")
  print(f"F has: {', '.join(svars_in_F)}")
  
  # Investigate equilibrium
  print("At equilibrium, xd=xdd=0 & M@qdd=0")
  eqsubs = {
    spn.omega_x : 0,
    spn.omega_y : 0,
    spn.omega_z : 0,
    spn.omega_xd : 0,
    spn.omega_yd : 0,
    spn.omega_zd : 0
  }
  EOM3 = F[1,0].subs(eqsubs).simplify()
  print("EOM3 at equilibrium:")
  sp.pprint(EOM3)
  sp.pprint(sp.solve(EOM3, spn.alphad)[0])
  # Conclusion: alphad_e = 0
  # EOM1=0 ALSO if alpha = 0, pi, -pi, ...
  # EOM2=EOM3=0 ALSO if alpha = pi/2, -pi/2, ...
  # THAT'S VALUABLE INFO. It says the ways that we can't accelerate, if we're starting from a stop.
  #   E.g. if alpha=0 & qd=0, then omega_xd=0. Wait. Is that trivial?
  #   Logic: qd=0 --> alphad=0 OR alpha=0,pi,-pi
  #     !(alphad=0 OR alpha=0,pi,-pi) --> qd != 0
  #     (alphad != 0 AND alpha != 0,pi,-pi) --> qd != 0
  #     I don't think I can say that (alpha = 0,pi,-pi) --> Anything
  #     But in some cases, at least... I've observed that alpha=0 --> no acceleration of omega_x (if we're starting at rest). To show that, I need the mapping omegad = f(alpha,alphad)
  #     Okay, I guess I can say that (alpha=0) implies that (alpha-dot *might* not be zero at equilibrium).
  
  # For control: state the reference position in the sphere-fixed frame.
  # There's a clear transform from omega to position (in sphere-fixed frame). Er... that's actually tricky bc the sphere rolls. Well, there's an easy transform from omega to velocity in sphere-fixed frame, so at least instantaneously we can do that.
  # Maybe I can linearize the transform from alpha-dot to omega-dot and do a long, fancy cascade. alpha-dot -> omega-dot -> omega -> v__s -> r_0
  
  
  
  # Jacobian Linearization
  # First question: can I invert the mass matrix
  #Mi = M.inv()
  #print(Mi)
  # Ran for ~5min w/o finishing
  # omega_dot = M.gauss_jordan_solve(F) # SLOW
  omega_dot = M.LUsolve(F)
  # OH, that solved nicely
  print(351)
  omega_dot = omega_dot.applyfunc(lambda x: x.expand())
  print("EOM1: omega_x-dot = ")
  sp.pprint(omega_dot[0,0])
  # print(omega_dot)
  with open("tmp_omega_dot.srepr","w") as file:
    omega_dot_str = sp.srepr(omega_dot)
    file.write(omega_dot_str)
  
  
  svars_in_od = []
  for svar in dir(spn):
    var = getattr(spn, svar)
    if omega_dot.has(var):
      svars_in_od.append(svar)
  print(f"omega_dot has: {', '.join(svars_in_M)}")
  
  # Now we have omega_dot (3x1) as a function of (Ig1, Ig2, Is, Rs, alpha, eta, ex, ey, ez, m, t)
  quit()
  
  # Build Jacobian for linearization
  J = sp.Matrix([
    []
  ])
  
  