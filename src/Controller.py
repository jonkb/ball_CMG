"""
Classes for the control schemes
"""
import numpy as np
import quaternion
import scipy.optimize as spo
import scipy.integrate as spi
from copy import copy
import dill

class Controller:
  """
  Parent class
  """
  
  name = "Parent controller"
  ball = None
  ref = None
  ref_type="p"
  dt_cnt = 0.1
  
  def __str__(self):
    s = self.name + "\n"
    # Cut off final \n and indent the whole block
    sball = str(self.ball)[0:-1].replace("\n", "\n\t")
    s += f"\tball: {sball}\n"
    return s
  
  def serializable(self):
    scnt = copy(self)
    scnt.ball = scnt.ball.serializable()
    # Replace any np arrays
    for name in dir(scnt):
      i = getattr(scnt, name)
      if isinstance(i, np.ndarray):
        setattr(scnt, name, dill.dumps(i))
    return scnt
  
  def from_serializable(self):
    cnt = copy(self)
    cnt.ball = cnt.ball.to_ball()
    # Replace any dill.dumps strings
    for name in dir(cnt):
      i = getattr(cnt, name)
      if isinstance(i, bytes):
        setattr(cnt, name, dill.loads(i))
    return cnt

class PreSet(Controller):
  """
  Simplest controller where alphadd is pre-set as a constant or as a function 
    of time.
  """
  
  name = "Preset alphadd controller"
  
  def __init__(self, ball, alphadd):
    self.ball = ball
    self.alphadd = alphadd
  
  def update(self, t, x):
    if callable(self.alphadd):
      alphadd = self.alphadd(t)
    else:
      alphadd = self.alphadd
    return self.ball.aa2pwm(alphadd)[1]

class FF(Controller):
  """ Feedforward control
  Calculate u to cause the desired acceleration
  
  This may be named poorly
  """
  
  name = "Feedforward controller"
  
  def __init__(self, ball, ref, ref_type="v"):
    """
    ref: function of time, returns (2,) np array
    ref_type: "p", "v", or "a"
    """
    self.ball = ball
    # NOTE: This assumes we know the whole reference signal from the start,
    #   as a function of time. That's probably fine for now.
    self.ref = ref
    self.ref_type = ref_type
    self.kp_ref = 1
    self.kd = 0.1
    self.opt_maxiter = 10
    self.u_bounds = (-1,1)
  
  def a_des(self, t, p, v):
    """ Outer loop: ref --> a_des
    """
    if self.ref_type == "a":
      return self.ref(t)
    elif self.ref_type == "v":
      v_err = self.ref(t) - v
      return self.kp_ref * v_err
    elif self.ref_type == "p":
      p_err = self.ref(t) - p
      # TODO: This should really be PID, not plain P
      # Ok, now it's PD
      return self.kp_ref * p_err - self.kd * v
    else:
      raise Exception("Invalid ref_type")
  
  def update(self, t, x):
    """ Return the next pwm control input
    """
    
    # Get a_des
    p = np.array([x[7], x[8]])
    v = self.ball.x2v(x) # TODO
    a_des = self.a_des(t, p, v)
    
    ## Inner loop: a_des --> u
    
    # Optimize to invert the mapping from u to acceleration
    def err(u, disp=False):
      # See how close this u is to giving a_des
      xd = self.ball.eom(x, u)
      # Calculate ax & ay for this x & xdot
      #s_axay = (eta, ex, ey, ez, omega_x, omega_y, omega_z, etad, exd, eyd, ezd, omega_xd, omega_yd, omega_zd)
      s_axay = (*x[0:7], *xd[0:7])
      a_vec = np.array([self.ball.axf(*s_axay), self.ball.ayf(*s_axay)])
      if disp:
        print(f"\ta_des: {a_des}, a_vec: {a_vec}")
      # L2 error
      return np.sum(np.square(a_des - a_vec))
    
    res = spo.minimize_scalar(err, method="bounded", bounds=self.u_bounds, 
      options={"maxiter":self.opt_maxiter})
    
    # Print every once in a while (lazy)
    # if abs( (t*100) % 1 ) < 0.01:
    # if np.random.rand() < 0.01:
    # Display how close we got
    print(f"FF141 t: {t}, error: {res.fun}, nfev: {res.nfev}, res.x: {res.x}")
    err(res.x, disp=True)
    
    #Return the optimized input
    u_opt = res.x
    return u_opt

class MPC(Controller):
  """ Model Predictive Control
  Simulate ahead by some time window, then choose the optimal next input.
  
  TODO: weight the cost by place in window. I.e. prioritize being there at the
    end and having low velocity at the end.
    Also maybe penalize high omegas & high u.
  """
  
  name = "MPC controller"
  # MPC options, settable through constructor
  opt_names = ("N_window", "ftol_opt", "maxit_opt", "v0_penalty", "w0_penalty", "w0_max")
  N_window = 4
  ftol_opt = 1e-3 # ftol for MPC optimization
  maxit_opt = 10
  v0_penalty = 0.0
  w0_penalty = 0.0
  w0_max = None
  
  def __init__(self, ball, ref, ref_type="v", dt_cnt=0.2, options={}):
    """
    ref: function of time, returns (2,) np array, corresponding to a vector 
      with 0 z-component, expressed in the 0-frame.
    ref_type: "p", "v", or "a"
    
    options: a dictionary containing MPC options
      "N_window"
      "ftol_opt"
      "v0_penalty" - how much to penalize high speeds
      "w0_penalty" - how much to penalize high angular speeds
      "w0_max" - If w0_penalty > 0, attempt to keep w0 below w0_max 
    """
    
    self.ball = ball
    # NOTE: This assumes we know the whole reference signal from the start,
    #   as a function of time. That's probably fine for now.
    self.ref = ref
    self.ref_type = ref_type
    self.dt_cnt = dt_cnt
    # MPC options
    for opt_name, opt_val in options.items():
      if opt_name in self.opt_names:
        setattr(self, opt_name, opt_val)
      else:
        print(f"Unrecognized MPC option: {opt_name}")
    
    # This variable will hold the result from MPC, to be used as a warm start
    #   at the next timestep.
    self.u_window = np.zeros(self.N_window)
    # Assume u is a signed pwm duty cycle (-1,1)
    self.u_bounds = [(-1,1)]*self.N_window
  
  def update(self, t, x):
    """ Return the next pwm control input
    
    The next N_window optimal inputs are calculated and stored in self.u_window
    Then u_window[0] is returned as the current u
    Next timestep, u_window will be rolled backward and used as a warm start 
      for the optimization
    
    TODO: can I get a jacobian for the cost function?
    
    NOTE / IDEA: Optimize a polynomial control function, like I was doing
      before. Issue: this requires a distinction btw control rate (dt_cnt) and
      the rate at which the control inputs are updated.
    """
    
    # Initial guess for u0 comes from the previous optimization, shifted by
    #   one timestep (dt_cnt)
    u0 = np.roll(self.u_window, -1)
    u0[-1] = u0[-2] # Repeat the last command
    
    # Times in the upcoming window
    t_window = t + np.arange(self.N_window) * self.dt_cnt
    # Simulate up to dt_cnt past the last ti in t_window.
    #   That lets the effect of the last control input be seen.
    t_end = t + self.dt_cnt * self.N_window
    
    def cost(u_w):
      # Cost function to optimize: how close does this u_window get us to the
      #   reference trajectory?
      
      # Reference trajectory in the window
      ref_w = np.array([self.ref(ti) for ti in t_window])
      
      # Simulate the inputs u_wi in the window
      #   NOTE: u_w[ t_w<=ti ][-1] returns the most recent control input 
      #   before (or at) ti
      xdot = lambda ti,xi: self.ball.eom(xi, u_w[t_window<=ti][-1])
      sol = spi.solve_ivp(xdot, [t, t_end], y0=x, method="RK23",
        t_eval=t_window, dense_output=False, rtol=1e-4, atol=1e-7)
        # TODO: Make these tols parameters
      
      # Calculate the error, depending on what type of reference ref is
      if self.ref_type == "p":
        # Position is x[7:9]
        ref_sim = sol.y.T[:,7:9]
      elif self.ref_type == "v":
        # Calculate velocity using self.ball.x2v
        ref_sim = np.array([self.ball.x2v(xi) for xi in sol.y.T])
      elif self.ref_type == "a":
        # Calculate acceleration using self.ball.x2a
        ref_sim = np.array([self.ball.x2v(xi) for xi in sol.y.T])
      
      err = ref_w - ref_sim
      #weighted_err = err*np.linspace(.1,1,N_eval) # Weight later points more?
      # L2 Error
      l2 = np.sum(np.square(err))
      if self.v0_penalty > 0:
        v = np.array([self.ball.x2v(xi) for xi in sol.y.T])
        l2 += self.v0_penalty * np.sum(np.square(v))
      if self.w0_penalty > 0:
        w = np.array([xi[4:7] for xi in sol.y.T])
        l2 += self.w0_penalty * np.sum(np.square(w))
        if self.w0_max is not None:
          # External barrier penalty method
          wn = np.linalg.norm(w, axis=0)
          wover = (wn-self.w0_max) * (wn > self.w0_max)
          l2 += 1000 * self.w0_penalty * np.sum(np.square(wover))
      return l2
    
    # NOTE: Technically, I could provide an analytical jacobian of cost.
    #   I think it'll take much longer than FD though, based on how big the
    #   Jacobian of the EOM is.
    res = spo.minimize(cost, u0, bounds=self.u_bounds, method="L-BFGS-B", 
      tol=self.ftol_opt, options={"maxiter":self.maxit_opt})
    # Save this whole window as a warm start for nexxt
    self.u_window = res.x
    # Return the next optimized input
    u_opt = res.x[0]
    print(f"t: {t}, res.success: {res.success}, cost: {res.fun}"
      f", nit: {res.nit}")
    return u_opt

if __name__ == "__main__":
  
  from CMGBall import CMGBall
  ball = CMGBall(ra=np.array([0.02, 0, 0]))
  # v_ref = lambda t: np.array([2,1]) * np.cos(t)
  # cnt = FF(ball, v_ref)
  
  # State: Initial state, but with an alphad
  x0 = np.zeros(11)
  # q = quaternion.from_rotation_vector([.001,.002,.003])
  q = quaternion.from_rotation_vector([0,0,0])
  # x0[0] = 1 # Real part of Q
  x0[0:4] = [q.w, q.x, q.y, q.z]
  x0[4] = 0.0 # Omega-x
  x0[5] = 0.0 # Omega-y
  x0[6] = 0.0 # Omega-z
  x0[9] = 0.0 #90 * np.pi/180 # alpha
  x0[10] = 0.0 #5 * np.pi/180 # alphad
  u = 0.01 # pwm input for alphadd
  alphadd = ball.pwm2aa(u)
  