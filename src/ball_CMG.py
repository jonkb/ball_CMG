"""
Simulating a robot composed of a ball with a Control Moment Gyroscope (CMG) mounted inside.

Capabilities:
1. Derive the EOM (dynamics.py)
2. Simulate the path of the robot
3. Plot simulation results (plot.py)
4. Optimize the input angle (alpha) to attempt to trace a given path

Currently, the user specifies which of those to perform by setting the booleans at the beginning of the main block at the end of the file.

TODO
* Try feedforward control
  * Solve dynamics for alphad... probably numerically
* Idea: characterize the relationship between the state of the robot 
  and the effect of applying an alphadd.
  Learn the mapping (x[0:7], xd[0:7]) => (ax, ay) with ML?
"""

import numpy as np
from sympy.algebras.quaternion import Quaternion
from sympy.functions.elementary.complexes import conjugate
from util import sharp, flat, tic, toc
#from scipy import signal
import scipy.integrate as spi
from scipy.misc import derivative as spd
import scipy.optimize as spo
import dill

import plot
import dynamics as dyn

def alphaddf(t):
  """ Input alpha function (for sim & plot)
  """

  """ OLD
  alphaddf = lambda t: 1 if (t < 0.5) else 0
  alphadf = lambda t: spi.quad(alphaddf, 0, t, limit=100)[0]
  alphaf = lambda t: spi.quad(alphadf, 0, t, limit=100)[0]
  alphadf = lambda t: 1 if (t < 0.5) else 0
  alphadf = lambda t: (((1 if (t < 0.5) else 0) if (t<1.5) else -1) if (t<2) else 0)
  alphadf = lambda t: np.sin(t*np.pi - np.pi/2)
  alphadf = lambda t: signal.square(2*np.pi*t)
  if t < 1:
    return 1
  elif t < 1.5:
    return 2
  elif t < 2:
    return -1
  elif t < 2.5:
    return 0
  elif t < 3:
    return -1
  else:
    return 0
  """
  
  #return -np.exp(.1*t) + 4*np.exp(-t)
  #return np.cos(4*np.pi*t) + 8*t/2 - 4*t**2
  #return np.sin(4*np.pi*t) + 2*(t-1)
  #return 5 * np.ones(np.array(t).shape) # Keep the shape right
  #return 4 - 2*t
  return 2 - 2*(t>1)

def alphadd_FF(Mf, Ff, axf, ayf, x, a_des):
  """ Calculate the correct alphadd to produce the desired acceleration
    The FF stands for feedforward control because this is using the
    system model, solving it for the correct input.
  
  a_des: [ax_des, ay_des]
  
  [M(x, alpha)]@{xdd} = {F(x, xd, alpha, alphad, alphadd)}
  """
  
  def err(alphadd):
    # See how close this alphadd is to giving a_des
    xd = eom(Mf, Ff, x, alphadd, R_sphere=0.05) # TODO R_sphere
    
    #s_axay = (eta, ex, ey, ez, omega_x, omega_y, omega_z, etad, exd, eyd, ezd, omega_xd, omega_yd, omega_zd)
    s_axay = (*x[0:7], *xd[0:7])
    a_vec = np.array([axf(*s_axay), ayf(*s_axay)])
    # L2 error
    return np.sum(np.square(a_des - a_vec))
  
  alphadd_bounds = (-np.pi, np.pi)
  res = spo.minimize_scalar(err, method="bounded", bounds=alphadd_bounds, 
    options={"maxiter":16})
  #print(f"Error: {res.fun}, nit: {res.nit}, res.x: {res.x}")
  #print(f"Error: {res.fun}, nfev: {res.nfev}, res.x: {res.x}")
  # TODO: Decide to do MPC instead if error is too large
  return res.x

def alphadf(t):
  """ OLD: When I was doing alphad control
  """
  return t**2/4

def valphaddf(t, v):
  """ Input alphadd function, parameterized for optimization
  """
  
  # Input: alpha = p0 + p1*t + sum(ai*cos(wi+phii))
  # v0 = np.array([p0, p1, a1, a2, w1, w2, phi1, phi2])
  return v[0] + v[1]*t + v[1]*np.cos(v[3]*t+v[5]) + v[2]*np.cos(v[4]*t+v[6])

def p_desf(t):
  """ Desired point function
  """
  # L = (1-np.exp(-t))
  # cx = np.cos((L-.5)*np.pi)
  # cy = 1+np.sin((L-.5)*np.pi)
  # return np.array([cx,cy])
  if np.size(t) > 1:
    return np.array([[1],[2]]) * t
  return t*np.array([1,2])

def a_desf(t, x):
  """ Desired acceleration function.
  Returns [ax_des, ay_des]
  """
  
  #return np.array([.01*t,-.1+.1*t])
  kp = .1
  to_goal = p_desf(t) - np.array([x[7], x[8]])
  return kp*to_goal

def eom(Mf, Ff, x, alphaddf, aldd_args=(), R_sphere=0.05):
  """ State variable EOM
  x is a (11,) numpy array of the state variables
  x = (eta, ex, ey, ez, omega_x, omega_y, omega_z, rx, ry, alpha, alphad)
  
  alphaddf: function that returns alphadd or a float alphadd
  aldd_args: tuple of arguments for alphaddf
  """
  
  xd = np.zeros(11)
  if isinstance(alphaddf, float):
    alphadd = np.copy(alphaddf)
  else:
    # Assume alphaddf is a function
    alphadd = alphaddf(*aldd_args)
  
  # Equations 1-4: Orientation quaternion:
  # NOTE: Maybe this part should be switched to use numpy quaternions
  q = Quaternion(x[0], x[1], x[2], x[3])
  omega_s__s = [x[4], x[5], x[6]]
  omega_s__0 = flat((q * sharp(omega_s__s) * conjugate(q)).expand())
  qdot = sharp(omega_s__0) * q / 2
  xd[0] = float(qdot.a)
  xd[1] = float(qdot.b)
  xd[2] = float(qdot.c)
  xd[3] = float(qdot.d)
  
  # Equations 5-7: omega-dot from EOM.
  M = Mf(*x, alphadd)
  F = Ff(*x, alphadd)
  # [M]{qddot} = {F}
  sol = np.linalg.lstsq(M, F, rcond=None)
  # d/dt([omega_x, omega_y, omega_z) = qddot = sol[0]
  # (This is because the omegas are generalized velocities)
  xd[4:7] = sol[0][:,0]
  
  # Equations 8-9: rx, ry
  # These come from the constraint equation: (-Rk) x Omega
  xd[7] = R_sphere*omega_s__0[1,0]
  xd[8] = -R_sphere*omega_s__0[0,0]
  
  # Equations 10-11: alpha, alphad
  #   Integrate the provided alphaddf input function
  xd[9] = x[10]
  xd[10] = alphadd
  
  # print("t:",t)
  # print("x:",x)
  # print("xd:",xd)
  return xd

def simulate(Mf, Ff, alphaddf, axf=None, ayf=None, a_desf=None, t_max=2, 
    x0=None, R_sphere=0.05, fname="sol.dill"):
  """ Simulate the CMG ball
  Parameters:
    Mf: Lambdified, symbolic mass matrix
    Ff: Lambdified, symbolic force vector
    alphaddf: Acceleration of alpha as a function of time
  """

  print("Solving IVP")
  # Initial conditions
  if x0 is None:
    x0 = np.zeros(11)
    x0[0] = 1 # Real part of quaternion starts at 1
  
  xdot = lambda t,x: eom(Mf, Ff, x, alphaddf, aldd_args=(Mf, Ff, axf, ayf, x, 
    a_desf(t, x)), R_sphere=0.05)

  sol = spi.solve_ivp(xdot, [0,t_max], x0, dense_output=True, rtol=1e-4, 
    atol=1e-7)

  if fname is not None:
    # Save to file
    with open(fname,"wb") as file:
      dill.dump(sol, file)
    print(f"IVP solution saved to {fname}")

  return sol

def simulate_old(Mf, Ff, alphadf, t_max=2, x0=None, R_sphere=0.05, 
    fname="sol.dill"):
  """ Simulate the CMG ball
  Parameters:
    Mf: Lambdified, symbolic mass matrix
    Ff: Lambdified, symbolic force vector
    alphadf: alpha-dot function
  
  """

  print("Solving IVP")
  # Initial conditions
  if x0 is None:
    x0 = np.array([1,0,0,0,0,0,0,0,0])
  
  # NOTE: Should I include alpha & derivatives in the state vector equation
  # Input alpha function int & der
  alphaddf = lambda t: spd(alphadf, t, dx=1e-6)
  alphaf = lambda t: spi.quad(alphadf, 0, t, limit=100)[0]
  #print("alphaddf(1):", alphaddf(1))
  #print("alphadf(1):", alphadf(1))
  #print("alphaf(1):", alphaf(1))

  # Convert input vector to symbolic substitutions
  # in: xin = (t, eta, ex, ey, ez, omega_x, omega_y, omega_z, rx__0, ry__0)
  # out: xs = (xi[1], xi[2], xi[3], xi[4], xi[5], xi[6], xi[7], rx__0, ry__0, alphaf(xi[0]), alphadf(xi[0]), alphaddf(xi[0]))
  xin_to_xs = lambda xin: tuple([*xin[1:], alphaf(xin[0]), alphadf(xin[0]), 
    alphaddf(xin[0])])
  # Matrices from an input vector
  Mff = lambda xi: Mf(*xin_to_xs(xi))
  Fff = lambda xi: Ff(*xin_to_xs(xi))

  def xdot(t, x):
    """ State variable EOM
    x is a (9,) numpy array of the state variables
    x = (eta, ex, ey, ez, omega_x, omega_y, omega_z, rx__0, ry__0)
    """
    
    xd = np.zeros(9)
    
    # Equations 1-4: Orientation quaternion:
    # NOTE: Maybe this part should be switched to use numpy quaternions
    q = Quaternion(x[0], x[1], x[2], x[3])
    omega_s__s = [x[4], x[5], x[6]]
    omega_s__0 = flat((q * sharp(omega_s__s) * conjugate(q)).expand())
    qdot = sharp(omega_s__0) * q / 2
    xd[0] = float(qdot.a)
    xd[1] = float(qdot.b)
    xd[2] = float(qdot.c)
    xd[3] = float(qdot.d)
    
    # Equations 5-7: omega-dot from EOM. TODO: Check is this right?
    xi = [t, *x]
    M = Mff(xi)
    F = Fff(xi)
    # [M]{qddot} = {F}
    sol = np.linalg.lstsq(M, F, rcond=None)
    # d/dt([omega_x, omega_y, omega_z) = qddot = sol[0]
    # (This is because the omegas are generalized velocities)
    xd[4:7] = sol[0][:,0]
    
    # Equations 8-9: rx, ry
    # These come from the constraint equation: (-Rk) x Omega
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

  if fname is not None:
    # Save to file
    with open(fname,"wb") as file:
      dill.dump(sol, file)
    print(f"IVP solution saved to {fname}")

  return sol

def optimize_pos(Mf, Ff, tmax, x_goal, y_goal, fname="opt_pos_res.dill"):
  """ Like optimize_path, but only care about a target position
  For now, the goal is to be stopped at the target
  """
  # TODO
  pass

def optimize_path(Mf, Ff, tv, rx_goal, ry_goal, n=32, fname="opt_path_res.dill"):
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
  bounds = [(-2,2), (-4,4), (0,4), (0,4), (1e-4,10), (1e-4,10), (0,np.pi), 
    (0,np.pi)]

  def cost(v):
    # Evaluate the cost of the given input vector
    alphadf = lambda t: valphadf(t, v)
    
    sol = simulate(Mf, Ff, alphadf, t_max=tv[-1])
    # Evaluate the path at 
    x = sol.sol(tv)
    
    c = err(x[7,:], x[8,:])
    print("Cost: ", c)
    
    return c
  
  # Optimize. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html#scipy.optimize.shgo
  res = spo.shgo(cost, bounds, n=n, sampling_method='sobol',
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
  # Simulate the response of the system to the input alphaddf
  sim = False
  # Plot the solution path
  plot_sol = True
  sol_fname = "sol.dill"
  # Optimize the input angle (alpha) to attempt to trace a given path
  optimize = False
  # Plot sol.dill and the alpha from opt_res.dill
  plot_opt = False
  
  t_max = 2.5
  tv = np.linspace(0,t_max,100)
  # Target path (for optimize)
  #rx_goal = tv/4
  #rx_goal = 1/(1+np.exp(-(10*tv-5))) - 1/(1+np.exp(5))
  rx_goal = tv*np.cos(2*np.pi*tv)
  #ry_goal = np.zeros(tv.shape)
  #ry_goal = 2/(1+np.exp(-(10*tv-5))) - 2/(1+np.exp(5))
  ry_goal = tv*np.sin(2*np.pi*tv)
  
  # Start timing the execution
  times = tic()
  
  if derive:
    print(" -- Deriving the EOM -- ")
    toc(times)
    M, F = dyn.derive_EOM()
    toc(times, "EOM derivation")
  
  if sim:
    if not derive: # Then load from file
      print(" -- Loading the EOM from file -- ")
      M, F = dyn.load_MF()
    print(" -- Simulating -- ")
    Mf, Ff = dyn.lambdify_MF(M, F, Omega_g=1000)
    # Get the ax & ay functions
    ax, ay = dyn.load_axay()
    axf, ayf = dyn.lambdify_axay(ax, ay)
    toc(times)
    sol = simulate(Mf, Ff, alphadd_FF, axf=axf, ayf=ayf, a_desf=a_desf, 
      t_max=t_max, fname=sol_fname)
    #sol = simulate_old(Mf, Ff, alphadf, t_max=t_max, fname=sol_fname)
    toc(times, "Simulation")
  
  if plot_sol:
    if not sim: # Then load from file
      with open(sol_fname, "rb") as file:
        sol = dill.load(file)
    print(" -- Plotting -- ")
    p_des = p_desf(tv)
    #print(p_des)
    px = p_des[0,:]
    py = p_des[1,:]
    plot.plot_sol(sol, tv, alphaddf=None, px=px, py=py)
  
  if optimize:
    if not derive:
      print(" -- Loading the EOM from file -- ")
      M, F = dyn.load_MF()
    Mf, Ff = dyn.lambdify_MF(M, F, Omega_g=1000)
    print(" -- Optimize input -- ")
    toc(times)
    res = optimize_path(Mf, Ff, tv, rx_goal, ry_goal, n=128)
    #print(res)
    toc(times, "Simulation")
  
  if plot_opt:
    print(" -- Plotting -- ")
    with open("opt_res.dill", "rb") as file:
      res = dill.load(file)
    alphaddf = lambda t: valphaddf(t, res.x)
    plot.plot_sol(sol, tv, alphaddf, px=rx_goal, py=ry_goal)
  
  print(" -- DONE -- ")
  toc(times, "Total execution", total=True)
  
