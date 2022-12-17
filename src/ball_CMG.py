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
  # Input alpha function (for sim & plot)

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
  
  #return -np.exp(.1*t) + 4*np.exp(-t)
  
  #return np.cos(4*np.pi*t) + 8*t/2 - 4*t**2
  
  #return np.sin(4*np.pi*t) + 2*(t-1)
  return .5*t

def valphaddf(t, v):
  """ Input alphadd function, parameterized for optimization
  """
  
  # Input: alpha = p0 + p1*t + sum(ai*cos(wi+phii))
  # v0 = np.array([p0, p1, a1, a2, w1, w2, phi1, phi2])
  return v[0] + v[1]*t + v[1]*np.cos(v[3]*t+v[5]) + v[2]*np.cos(v[4]*t+v[6])

def simulate(Mfs, Ffs, alphaddf, t_max=2, x0=None, R_sphere=0.05, 
    fname="sol.dill"):
  """ Simulate the CMG ball
  Parameters:
    Mfs: Lambdified, symbolic mass matrix
    Ffs: Lambdified, symbolic force vector
    alphaddf: Acceleration of alpha as a function of time
  """

  print("Solving IVP")
  # Initial conditions
  if x0 is None:
    x0 = np.zeros(11)
    x0[0] = 1 # Real part of quaternion starts at 1
  
  def xdot(t, x):
    """ State variable EOM
    x is a (9,) numpy array of the state variables
    x = (nu, ex, ey, ez, omega_x, omega_y, omega_z, rx, ry, alpha, alphad)
    """
    
    xd = np.zeros(11)
    alphadd = alphaddf(t)
    
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
    M = Mfs(*x, alphadd)
    F = Ffs(*x, alphadd)
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

def simulate_old(Mfs, Ffs, alphadf, t_max=2, R_sphere=0.05, fname="sol.dill"):
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
  
  # NOTE: Should I include alpha & derivatives in the state vector equation
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
    M = Mf(xi)
    F = Ff(xi)
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

def optimize_pos(Mfs, Ffs, tmax, x_goal, y_goal, fname="opt_pos_res.dill"):
  """ Like optimize_path, but only care about a target position
  For now, the goal is to be stopped at the target
  """
  # TODO
  pass

def optimize_path(Mfs, Ffs, tv, rx_goal, ry_goal, n=32, fname="opt_path_res.dill"):
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
    
    sol = simulate(Mfs, Ffs, alphadf, t_max=tv[-1])
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
  sim = True
  # Plot the solution path
  plot_sol = True
  sol_fname = "sol.dill"
  # Optimize the input angle (alpha) to attempt to trace a given path
  optimize = False
  # Plot sol.dill and the alpha from opt_res.dill
  plot_opt = False
  
  t_max = 1
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
    Mfs, Ffs = dyn.lambdify_MF(M, F, Omega_g=1000)
    toc(times)
    sol = simulate(Mfs, Ffs, alphaddf, t_max=t_max, fname=sol_fname)
    toc(times, "Simulation")
  
  if plot_sol:
    if not sim: # Then load from file
      with open(sol_fname, "rb") as file:
        sol = dill.load(file)
    print(" -- Plotting -- ")
    plot.plot_sol(sol, tv, alphaddf)
  
  if optimize:
    if not derive:
      print(" -- Loading the EOM from file -- ")
      M, F = dyn.load_MF()
    Mfs, Ffs = dyn.lambdify_MF(M, F, Omega_g=1000)
    print(" -- Optimize input -- ")
    toc(times)
    res = optimize_path(Mfs, Ffs, tv, rx_goal, ry_goal, n=128)
    print(res)
    toc(times, "Simulation")
  
  if plot_opt:
    print(" -- Plotting -- ")
    with open("opt_res.dill", "rb") as file:
      res = dill.load(file)
    alphaddf = lambda t: valphaddf(t, res.x)
    plot.plot_sol(sol, tv, alphaddf, px=rx_goal, py=ry_goal)
  
  print(" -- DONE -- ")
  toc(times, "Total execution", total=True)
  
