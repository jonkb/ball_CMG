"""
Finding EOM for ball_CMG
"""
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import sympy as sp
from sympy.algebras.quaternion import Quaternion
from sympy.functions.elementary.complexes import conjugate
import scipy.integrate as spi
from scipy.misc import derivative as spd
from scipy import signal
import scipy.optimize as spo
import dill

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

def derive_EOM():
  """ Derive the equations of motion and save them to file
  """

  ## Setup
  print("Setting up symbolic variables")
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
  omega_xdd = sp.diff(omega_x, (t,2))
  omega_ydd = sp.diff(omega_y, (t,2))
  omega_zdd = sp.diff(omega_z, (t,2))
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
  # Gyroscope
  I_g__g = sp.diag(Ig1, Ig2, Ig2)
  Omega_g = sp.symbols("Omega_g") # Constant angular velocity of gyro about x_g
  alpha = sp.Function("alpha")(t) # Angle btw x_s and x_g, rotating about z_s
  alphad = sp.diff(alpha, t)
  alphadd = sp.diff(alphad, t)
  omega_g__s = omega_s__s + sp.Matrix([[Omega_g*sp.cos(alpha)], [Omega_g*sp.sin(alpha)], [alphad]])
  omega_g__0 = flat((q * sharp(omega_g__s) * conjugate(q)).expand())
  # Rotate from sphere frame to gyro frame
  q_g__s = Quaternion.from_axis_angle((0,0,1), alpha)
  omega_g__g = flat((q_g__s * sharp(omega_g__s) * conjugate(q_g__s)).expand())


  ## Write Lagrangian
  # Kinetic Energy
  T = 1/2*m*(rxd__0**2 + ryd__0**2)
  T += (1/2*omega_s__s.T * I_s__s * omega_s__s)[0,0]
  T += (1/2*omega_g__g.T * I_g__g * omega_g__g)[0,0]
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
  EOM1 = d_dt_dL_dq1d - dL_dq1# - Tau_x
  # q2d = omega_y
  dL_dq2d = sp.diff(L, omega_y)
  d_dt_dL_dq2d = sp.diff(dL_dq2d, t)
  dL_dq2 = 0 #sp.diff(L, int omega_y) #ASSUME
  EOM2 = d_dt_dL_dq2d - dL_dq2# - Tau_y
  # q3d = omega_z
  dL_dq3d = sp.diff(L, omega_z)
  d_dt_dL_dq3d = sp.diff(dL_dq3d, t)
  dL_dq3 = 0 #sp.diff(L, int omega_z) #ASSUME
  EOM3 = d_dt_dL_dq3d - dL_dq3# - Tau_z

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

  print("Writing in matrix form ([M]{xdot} = {F}")

  print("\tBuilding M matrix")
  M = sp.zeros(3)
  M[0,0] = EOM1.coeff(omega_xd) # omega_xd = q1dd
  M[0,1] = EOM1.coeff(omega_yd)
  M[0,2] = EOM1.coeff(omega_zd)
  EOM1 -= (M[0,0]*omega_xd + M[0,1]*omega_yd + M[0,2]*omega_zd).expand()
  M[1,0] = EOM2.coeff(omega_xd)
  M[1,1] = EOM2.coeff(omega_yd)
  M[1,2] = EOM2.coeff(omega_zd)
  EOM2 -= (M[1,0]*omega_xd + M[1,1]*omega_yd + M[1,2]*omega_zd).expand()
  M[2,0] = EOM3.coeff(omega_xd)
  M[2,1] = EOM3.coeff(omega_yd)
  M[2,2] = EOM3.coeff(omega_zd)
  EOM3 -= (M[2,0]*omega_xd + M[2,1]*omega_yd + M[2,2]*omega_zd)
  EOM3 = EOM3.expand()
  print("\t\tSimplifying")
  M = M.applyfunc(lambda x: x.expand())
  M = M.applyfunc(lambda x: x.simplify())
  # print("M: ", M)

  print("\tBuilding F vector")
  F = sp.zeros(3,1)
  F[0] = -EOM1
  F[1] = -EOM2
  F[2] = -EOM3
  print("\t\tSimplifying")
  F = F.applyfunc(lambda x: x.simplify())
  
  print("Saving to file")
  # with open("M.dill","wb") as file:
    # dill.dump(M, file)
  # with open("F.dill","wb") as file:
    # dill.dump(F, file)
  with open("M.srepr","w") as file:
    M_str = sp.srepr(M)
    file.write(M_str)
  with open("F.srepr","w") as file:
    F_str = sp.srepr(F)
    file.write(F_str)
  print("M & F saved to file")
  
  return M, F

def lambdify_MF(M, F, R_sphere=0.05, Omega_gyro=600):
  # Convert to state form.
  print("Lambdifying EOM")
  # See https://docs.sympy.org/latest/modules/utilities/lambdify.html
  # And https://docs.sympy.org/latest/modules/numeric-computation.html


  ## Setup (Copied from derive_EOM) TODO
  print("Setting up symbolic variables")
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
  omega_xdd = sp.diff(omega_x, (t,2))
  omega_ydd = sp.diff(omega_y, (t,2))
  omega_zdd = sp.diff(omega_z, (t,2))
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
  # Gyroscope
  I_g__g = sp.diag(Ig1, Ig2, Ig2)
  Omega_g = sp.symbols("Omega_g") # Constant angular velocity of gyro about x_g
  alpha = sp.Function("alpha")(t) # Angle btw x_s and x_g, rotating about z_s
  alphad = sp.diff(alpha, t)
  alphadd = sp.diff(alphad, t)
  omega_g__s = omega_s__s + sp.Matrix([[Omega_g*sp.cos(alpha)], [Omega_g*sp.sin(alpha)], [alphad]])
  omega_g__0 = flat((q * sharp(omega_g__s) * conjugate(q)).expand())
  # Rotate from sphere frame to gyro frame
  q_g__s = Quaternion.from_axis_angle((0,0,1), alpha)
  omega_g__g = flat((q_g__s * sharp(omega_g__s) * conjugate(q_g__s)).expand())


  # State variables x
  consts = {Is: .001, Ig1: .001, Ig2: .001, m: 1, Rs: R_sphere, 
    Omega_g: Omega_gyro}
  M = M.subs(consts)
  F = F.subs(consts)
  #print("M: ", M)
  #print("F: ", F)

  # Symbolic substitutions vector
  xs = (nu, ex, ey, ez, omega_x, omega_y, omega_z, rx__0, ry__0, alpha, alphad, alphadd) # -s in xs for symbolic
  # Lambdified functions for M and F
  Mfs = sp.lambdify(xs, M, "numpy")
  Ffs = sp.lambdify(xs, F, "numpy")

  return Mfs, Ffs

def load_MF():
  """ Load M & F from M.srepr, F.srepr
  Assumes those files exist
  """
  # with open("M.dill", "rb") as file:
    # M = dill.load(file)
  # with open("F.dill", "rb") as file:
    # F = dill.load(file)
  with open("M.srepr", "r") as file:
    M_str = file.read()
    M = sp.sympify(M_str)
  with open("F.srepr", "r") as file:
    F_str = file.read()
    F = sp.sympify(F_str)
  return M, F

def simulate(Mfs, Ffs, alphadf, t_max=2, R_sphere=0.05):
  """ Simulate the CMG ball
  Parameters:
    Mfs: Lambdified, symbolic mass matrix
    Ffs: Lambdified, symbolic force vector
  
  Defined below: (Should probably be parameters) TODO
    Initial conditions: x0
    Gyro angle alpha: alphaf, alphadf, alphaddf
  """

  print("Solving IVP")
  # Initial conditions
  x0 = np.array([1,0,0,0,0,0,0,0,0])
  
  # Input alpha function int & der
  alphaddf = lambda t: spd(alphadf, t, dx=1e-6)
  alphaf = lambda t: spi.quad(alphadf, 0, t, limit=100)[0]
  #print("alphaddf(1):", alphaddf(1))
  #print("alphadf(1):", alphadf(1))
  #print("alphaf(1):", alphaf(1))

  # Convert input vector to symbolic substitutions
  # in: xin = (t, nu, ex, ey, ez, omega_x, omega_y, omega_z, rx__0, ry__0)
  # out: xs = (xi[1], xi[2], xi[3], xi[4], xi[5], xi[6], xi[7], rx__0, ry__0, alphaf(xi[0]), alphadf(xi[0]), alphaddf(xi[0]))
  xin_to_xs = lambda xin: tuple([*xin[1:], alphaf(xin[0]), alphadf(xin[0]), 
    alphaddf(xin[0])])
  # Matrices from an input vector
  Mf = lambda xi: Mfs(*xin_to_xs(xi))
  Ff = lambda xi: Ffs(*xin_to_xs(xi))

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
    xi = [t, *x]
    M = Mf(xi)
    F = Ff(xi)
    # [M]{qddot} = {F}
    sol = np.linalg.lstsq(M, F, rcond=None)
    # d/dt([omega_x, omega_y, omega_z) = qddot = sol[0]
    # (This is because the omegas are generalized velocities)
    xd[4:7] = sol[0][:,0]
    
    # Equations 8-9: rx, ry
    # These come from The constraint equation: (-Rk) x Omega
    xd[7] = R_sphere*omega_s__0[1,0]
    xd[8] = -R_sphere*omega_s__0[0,0]
    
    # print("t:",t)
    # print("x:",x)
    # print("xd:",xd)
    return xd

  # print("xdot(0): ", xdot(0,x0))
  # print("xdot(0.1): ", xdot(0.1,x0))

  sol = spi.solve_ivp(xdot, [0,t_max], x0, dense_output=True, rtol=1e-4, 
    atol=1e-7)

  with open("sol.dill","wb") as file:
    dill.dump(sol, file)

  print("IVP solution saved to sol.dill")

  return sol

def plot_sol(alphadf, t_max=2, px=None, py=None):
  """ Plots sol.dill
    TODO: Make sol a parameter
  """
  with open("sol.dill", "rb") as file:
    sol = dill.load(file)

  #print(f"dir(sol): {dir(sol)}")

  print("t_min, t_max: ", min(sol.t), max(sol.t))

  t = np.linspace(0, t_max, 200)
  x = sol.sol(t)
  
  # Int & diff alphad
  alphaddf = lambda t: spd(alphadf, t, dx=1e-6)
  alphaf = lambda t: spi.quad(alphadf, 0, t, limit=100)[0]

  fig, axs = plt.subplots(3,2, sharex=True)
  alpha = [alphaf(ti) for ti in t]
  axs[0,0].plot(t, alpha)
  axs[0,0].set_title(r"Gyro angle $\alpha$ (rad)")
  axs[0,1].set_title(r"Nothing (for now)")
  axs[1,0].plot(t, x[0,:], label="nu")
  axs[1,0].plot(t, x[1,:], label="ex")
  axs[1,0].plot(t, x[2,:], label="ey")
  axs[1,0].plot(t, x[3,:], label="ez")
  qnorm = np.linalg.norm(x[0:4,:], axis=0)
  axs[1,0].plot(t, qnorm, label="|q|")
  axs[1,0].legend()
  axs[1,0].set_title("Orientation Q")
  axs[1,1].plot(t, x[4,:], label="$\omega_x$")
  axs[1,1].plot(t, x[5,:], label="$\omega_y$")
  axs[1,1].plot(t, x[6,:], label="$\omega_z$")
  wnorm = np.linalg.norm(x[4:7,:], axis=0)
  axs[1,1].plot(t, wnorm, label="|$\omega$|")
  axs[1,1].legend()
  axs[1,1].set_title("Angular Velocity $\omega$")
  axs[2,0].plot(t, x[7,:])
  axs[2,0].set_xlabel("Time t")
  axs[2,0].set_title("X-Position $r_x$")
  axs[2,1].plot(t, x[8,:])
  axs[2,1].set_xlabel("Time t")
  axs[2,1].set_title("Y-Position $ry$")
  fig.show()
  
  # Infer x-bounds
  xmin = np.min(x[7,:])
  xmax = np.max(x[7,:])
  xspan = xmax-xmin
  ymin = np.min(x[8,:])
  ymax = np.max(x[8,:])
  yspan = ymax-ymin
  xmin -= .1*xspan # Add 10% margins
  xmax += .1*xspan
  ymin -= .1*yspan
  ymax += .1*yspan
  
  # Animate
  # https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
  fig = plt.figure()
  #ax = plt.axes(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
  ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
  ax.set_aspect("equal")
  ax.grid()
  circle, = ax.plot([0], [0], marker="o")
  if (px is not None) and (py is not None):
    p, = ax.plot(px, py)

  # initialization function: plot the background of each frame
  def init():
      circle.set_data([], [])
      return circle,

  # animation function.  This is called sequentially
  def animate(i):
      rx = x[7,i]
      ry = x[8,i]
      circle.set_data([rx], [ry])
      return circle,

  # call the animator.  blit=True means only re-draw the parts that have changed.
  anim = animation.FuncAnimation(fig, animate, init_func=init,
    frames=200, interval=20, blit=True)

  fig.show()
  
  input("PRESS ANY KEY TO QUIT")

def valphadf(t, v):
  # Input: alpha = p0 + p1*t + sum(ai*cos(wi+phii))
  # v0 = np.array([p0, p1, a1, a2, w1, w2, phi1, phi2])
  return v[0] + v[1]*t + v[1]*np.cos(v[3]*t+v[5]) + v[2]*np.cos(v[4]*t+v[6])

def optimize_alpha(Mfs, Ffs, tv, rx_goal, ry_goal, fname="opt_res.dill"):
  """ Currently, the target path is defined here
  TODO: pass target path as argument
  """

  def err(rx, ry):
    # Error between a given path and the goal path
    # Future: could weight the end position heavier than the path
    # [x_err; y_err]
    v_err = np.vstack([rx-rx_goal, ry-ry_goal])
    return np.linalg.norm(v_err)
  
  # Input: alpha = p0 + p1*t + sum(ai*cos(wi+phii))
  # v0 = np.array([p0, p1, a1, a2, w1, w2, phi1, phi2])
  bounds = [(0,2), (0,2), (0,2), (0,2), (1e-4,10), (1e-4,10), (0,np.pi), 
    (0,np.pi)]

  def cost(v):
    # Evaluate the cost of the given input vector
    alphadf = lambda t: valphadf(t, v)
    
    sol = simulate(Mfs, Ffs, alphadf, t_max=tv[-1])
    # Evaluate the path at 
    x = sol.sol(tv)
    
    c = err(x[7,:], x[8,:])
    print("Cost: ", c)
    
    return c
  
  # Optimize
  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html#scipy.optimize.shgo
  res = spo.shgo(cost, bounds, n=16, sampling_method='sobol',
    minimizer_kwargs={"options":{"tol":.1}})
  
  if fname is not None:
    # Save to file
    with open(fname,"wb") as file:
      dill.dump(res, file)
    print(f"Optimization result saved to {fname}")
  
  return res

if __name__ == "__main__":
  # Derive the equations of motion and save them to file
  
  derive = False
  sim = False
  plot = False
  optimize = True
  plot_opt = False
  
  t_max = 1
  
  # Target path (for optimize)
  tv = np.linspace(0,1,100)
  #rx_goal = tv/4
  #rx_goal = 1/(1+np.exp(-(10*tv-5))) - 1/(1+np.exp(5))
  rx_goal = tv*np.cos(2*np.pi*tv)
  #ry_goal = np.zeros(tv.shape)
  #ry_goal = 2/(1+np.exp(-(10*tv-5))) - 2/(1+np.exp(5))
  ry_goal = tv*np.sin(2*np.pi*tv)
  
  # Input alpha function (for sim & plot)
  def alphadf(t):
    # OLD:
    # alphaddf = lambda t: 1 if (t < 0.5) else 0
    # alphadf = lambda t: spi.quad(alphaddf, 0, t, limit=100)[0]
    # alphaf = lambda t: spi.quad(alphadf, 0, t, limit=100)[0]
    #alphadf = lambda t: 1 if (t < 0.5) else 0
    #alphadf = lambda t: (((1 if (t < 0.5) else 0) if (t<1.5) else -1) if (t<2) else 0)
    #alphadf = lambda t: np.sin(t*np.pi - np.pi/2)
    #alphadf = lambda t: signal.square(2*np.pi*t)
    # if t < 1:
      # return 1
    # elif t < 1.5:
      # return 2
    # elif t < 2:
      # return -1
    # # elif t < 2.5:
      # # return 0
    # # elif t < 3:
      # # return -1
    # else:
      # return 0
    
    return -np.exp(.1*t) + 4*np.exp(-t)
  
  if derive:
    print(" -- Deriving the EOM -- ")
    M, F = derive_EOM()
  
  if sim:
    if not derive:
      print(" -- Loading the EOM from file -- ")
      M, F = load_MF()
    print(" -- Simulating -- ")
    Mfs, Ffs = lambdify_MF(M, F, Omega_gyro=1000)
    sol = simulate(Mfs, Ffs, alphadf, t_max=t_max)
  
  if plot:
    print(" -- Plotting -- ")
    plot_sol(alphadf, t_max=t_max)
  
  if optimize:
    if not derive:
      print(" -- Loading the EOM from file -- ")
      M, F = load_MF()
    Mfs, Ffs = lambdify_MF(M, F, Omega_gyro=1000)
    print(" -- Optimize input -- ")
    res = optimize_alpha(Mfs, Ffs, tv, rx_goal, ry_goal)
    print(res)
    
  if plot_opt:
    with open("opt_res.dill", "rb") as file:
      res = dill.load(file)
    alphadf = lambda t: valphadf(t, res.x)
    plot_sol(alphadf, t_max=t_max, px=rx_goal, py=ry_goal)
    
  