"""
Classes for the control schemes
"""
import numpy as np
import scipy.optimize as spo
from copy import copy

class Controller:
  """
  Parent class
  """
  
  name = "Parent Controller"
  ball = None
  
  def __str__(self):
    s = self.name + "\n"
    # Cut off final \n and indent the whole block
    sball = str(self.ball)[0:-1].replace("\n", "\n\t")
    s += f"\tball: {sball}\n"
    return s
  
  def serializable(self):
    scnt = copy(self)
    scnt.ball = scnt.ball.serializable()
    return scnt
  
  def from_serializable(self):
    cnt = copy(self)
    cnt.ball = cnt.ball.to_ball()
    return cnt

class PreSet(Controller):
  """
  Simplest controller where alphadd is pre-set as a constant or as a function 
    of time.
  """
  
  name = "Preset alphadd Controller"
  
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
  Calculate alphadd to cause the desired acceleration
  """
  
  name = "Feedforward Controller"
  
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
    self.kp_ref = 10
    self.opt_maxiter = 10
  
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
      return self.kp_ref * p_err
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
    
    # Optimize to invert the mapping from alphadd to acceleration
    def err(alphadd, disp=False):
      # See how close this alphadd is to giving a_des
      u = self.ball.aa2pwm(alphadd)[1]
      xd = self.ball.eom(x, u)
      #s_axay = (eta, ex, ey, ez, omega_x, omega_y, omega_z, etad, exd, eyd, ezd, omega_xd, omega_yd, omega_zd)
      s_axay = (*x[0:7], *xd[0:7])
      a_vec = np.array([self.ball.axf(*s_axay), self.ball.ayf(*s_axay)])
      if disp:
        print(f"\ta_des: {a_des}, a_vec: {a_vec}")
      # L2 error
      return np.sum(np.square(a_des - a_vec))
    
    alphadd_bounds = (-self.ball.alphadd_max, self.ball.alphadd_max)
    res = spo.minimize_scalar(err, method="bounded", bounds=alphadd_bounds, 
      options={"maxiter":self.opt_maxiter})
    # if abs( (t*100) % 1 ) < 0.01:
    if np.random.rand() < 0.01:
      # Display how close we got
      print(f"t: {t}, error: {res.fun}, nfev: {res.nfev}, res.x: {res.x}")
      err(res.x, disp=True)
    u_opt = self.ball.aa2pwm(res.x)[1]
    return u_opt


  
if False: # Copied from Simulation

  # Parameters for MPC
  MPCprms = {
    "t_window": .25,
    "N_vpoly": 3,
    "N_sobol": 32, # Should be a power of 2
    "N_eval": 5,
    "ratemax": 200, #Hz
    "vweight": 0.001 # Weight on v_err relative to x_err
  }
  t_MPChist = [] # List of each time MPC optimization is run
  v_MPChist = [] # List of v vectors from each time MPC is run
  
  for key, val in MPCprms.items():
    self.MPCprms[key] = val
    
  s += f"\tMPCprms: {self.MPCprms}\n"
  
  def alphadd_v(self, t, v):
    """ Parameterized acceleration function to be optimized in MPC
    v is formatted as a polynomial
    f = v[0] + v[1]*t + v[2]*t**2 + ... + v[n]*t**n
    """
    
    # Equivalent, but possibly slower method:
    # for n,vi in enumerate(v):
      # a += vi*t**n
    # return a
    
    t__n = np.power(t*np.ones_like(v), np.arange(len(v)))
    
    return np.sum(np.multiply(v,t__n))
  
  def alphadd_MPC(self, t, x):
    """ Return the acceleration calculated by MPC
    Procedure:
    1. Check if we've already optimized the input for a time point very close 
      to this one.
    2. If not, simulate ahead for some amount of time
    3. Score the resultant path
    4. Optimize 2 & 3
    """
    
    # Unpack MPCprms
    t_window = self.MPCprms["t_window"]
    N_vpoly = self.MPCprms["N_vpoly"]
    N_sobol = self.MPCprms["N_sobol"]
    N_eval = self.MPCprms["N_eval"]
    ratemax = self.MPCprms["ratemax"]
    vweight = self.MPCprms["vweight"]
    
    if len(self.t_MPChist) > 0:
      t_diff = np.abs(np.array(self.t_MPChist) - t)
      diff_min = np.min(t_diff)
      # Are we close enough to one that's already been optimized?
      if diff_min < 1.0/ratemax:
        # Then use solution from that MPC run
        # Index of closest solution to current time
        i_closest = np.where(t_diff == diff_min)[0][0]
        t_closest = self.t_MPChist[i_closest]
        v_closest = self.v_MPChist[i_closest]
        alphadd = self.alphadd_v(t-t_closest, v_closest)
        return alphadd
      # Otherwise, re-optimize v for alphadd_v(t,v) with MPC
    
    if self.p_des is None:
      assert False, "Must supply desired path to use MPC control"
    
    def cost(v):
      # Simulate using the given input vector v and score the path
      xdot = lambda t,x: dyn.eom(self.Mf, self.Ff, x, self.alphadd_v(t,v), 
        ball=self.ball)
      # NOTE: this time is relative to the current time, t
      t_eval = np.linspace(0, t_window, N_eval)
      sol = spi.solve_ivp(xdot, [0,t_window], y0=x, method="RK23",
        t_eval=t_eval, dense_output=False, rtol=1e-4, atol=1e-7) # TODO: tol
      
      # Compare to desired path
      p_xy = [self.p_des(ti) for ti in t_eval]
      p_x = np.array([p[0] for p in p_xy])
      p_y = np.array([p[1] for p in p_xy])
      v_x = np.gradient(p_x, t_eval)
      v_y = np.gradient(p_y, t_eval)
      vsol_x = np.gradient(sol.y[7], t_eval)
      vsol_y = np.gradient(sol.y[8], t_eval)
      # (4xN_MPCeval)
      err = np.array([p_x-sol.y[7], p_y-sol.y[8], v_x-vsol_x, v_y-vsol_y])
      weighted_err = err*np.linspace(.1,1,N_eval)
      weighted_err[2:4,:] *= vweight
      return np.sum(np.square(weighted_err))
    
    # NOTE: Does it make sense to bound higher order terms the same way?
    v_bounds = (-self.ball.alphadd_max, self.ball.alphadd_max)
    bounds = [v_bounds]*N_vpoly
    # Optimize. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html#scipy.optimize.shgo
    # NOTE: Could switch from Sobol to LHC. Probably wouldn't change much
    res = spo.shgo(cost, bounds, n=N_sobol, sampling_method='sobol',
      minimizer_kwargs={"options":{"tol":.1}})
    
    v = res.x
    print(f"t: {t}, cost: {res.fun}, nfev: {res.nfev}, v_opt: {v}")
    
    # Store this result to be reused
    self.t_MPChist.append(t)
    self.v_MPChist.append(v)
    
    return self.alphadd_v(0, v)


if __name__ == "__main__":
  from CMGBall import CMGBall
  ball = CMGBall()
  v_ref = lambda t: np.array([2,1]) * np.cos(t)
  cnt = FF(ball, v_ref)