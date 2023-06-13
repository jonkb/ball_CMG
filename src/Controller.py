"""
Classes for the control schemes
"""
import numpy as np
import quaternion
import scipy.optimize as spo
import scipy.integrate as spi
from copy import copy
import dill
import control

from diff import rk4
from util import cleanup_versor

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

class Observer:
  """ Luenberger observer. To be used by the controllers.
  Maybe move to a different file?
  """
  
  dt_obs = 0.001
  nx = 11
  x_subset = np.array([0,1,2,3,4,5,6,9,10]) # Focus on q, omega, and alpha
  # x_subset = np.array([0,1,2,3,4,5,6]) # Not estimating alpha
  # x_subset = np.arange(11)
  # 
  # wmin_L
  # Limits on quantities in xhat
  # xhat_max = np.array([1.,1.,1.,1., 15.,15.,15., np.inf,12.])
  xhat_max = np.array([1.,1.,1.,1., 20.,20.,20., np.inf,120.])
  # Limits on xhat_dot
  xhd_max = np.array([10.,10.,10.,10., 20.,20.,20., 12.,10.])
  # L = np.array(
    # [[ 1.18993274e-15,  1.54015810e-06,  2.26394918e-06,
      # -4.52789720e-10, -6.94786078e-16,  9.60169444e-16,
       # 6.99997781e-02,  6.99962781e+00,  1.50943070e-05],
     # [ 1.00456848e-13,  1.35957733e-04, -5.84986302e+01,
       # 5.11699726e-01, -8.68141881e-15,  1.02196029e-13,
       # 5.73286575e+00,  5.73257911e+02, -5.23774477e+00],
     # [ 1.33281348e-13,  1.80116481e-04, -7.79981753e+01,
       # 1.55996351e-02,  5.00000000e-01,  1.15241150e-13,
       # 7.64382117e+00,  7.64343898e+02,  3.84966227e+00],
     # [-2.01589833e+00,  7.86428062e+09,  1.15600689e+10,
      # -2.31200658e+06, -3.57948194e-04,  4.99164646e-01,
      # -1.13288675e+09, -1.13283011e+11,  7.70738187e+10],
     # [ 1.87188284e+00, -8.08262366e+09, -1.18810214e+10,
       # 2.37621388e+06,  3.27312783e-04,  7.71595165e-04,
       # 1.16434009e+09,  1.16428188e+11, -7.92136879e+10],
     # [ 2.00033997e+01, -2.18436539e+07, -3.21119342e+07,
       # 6.42238683e+03,  8.80000093e+00,  6.60800142e+00,
       # 3.14696957e+06,  3.14681222e+08, -2.14078554e+08],
     # [-1.13364454e-15,  2.11486499e-06,  2.49988109e-01,
      # -4.99976218e-05, -6.60000000e+00,  8.80000000e+00,
      # -2.44988347e-02, -2.44976097e+00,  1.66677105e+00],
     # [ 2.30989354e+00, -1.31071344e+10, -1.92667814e+10,
       # 3.85335629e+06,  5.80477733e-04,  9.64334235e-04,
       # 1.88814458e+09,  1.88805017e+11, -1.28456365e+11],
     # [-3.71980091e+00,  9.83035071e+09,  1.44500862e+10,
      # -2.89001725e+06, -4.44758985e-04, -1.52444746e-03,
      # -1.41610845e+09, -1.41603764e+11,  9.63422728e+10]])
  L = np.array([[-1.96782382e-02, -2.95423728e-01,  4.56029889e+00,
        -9.55464814e-01, -1.61235679e-02, -1.17187726e-03],
       [-2.34002057e-01,  7.59921549e-01, -2.55709900e-02,
         5.42576278e-01, -2.85777501e-02, -1.77335084e+00],
       [-5.85427297e-01, -1.89900178e-01, -2.38664853e-02,
         2.09238500e-02,  1.15043187e+00, -4.19882100e+00],
       [ 3.48522734e-01,  1.00578908e+02, -1.28408187e+03,
         2.65420908e+01,  3.44931848e-01, -3.65202625e+01],
       [-3.21830073e-02, -5.64180029e-03,  4.33590897e-02,
         1.98019637e+01, -1.40785397e-01, -4.37241577e-02],
       [ 2.11817132e+00,  2.96896122e-02,  2.51870227e-02,
        -1.50230369e-01,  2.87785422e+01,  3.93295238e-01],
       [-1.62683415e-01, -1.89098555e-01, -1.51966845e-02,
        -2.19582208e-02, -1.38088782e-02,  1.57384449e+01],
       [ 9.43618128e-02, -6.52619788e-02, -4.90192719e-02,
         8.56244977e+02,  1.30250591e+01, -1.50019842e+00],
       [-1.95072785e-01, -6.74336317e-02, -6.24021377e-03,
         2.58119226e-02, -1.91773086e+00, -1.39421669e+00]])
  
  def __init__(self, ball, x0):
    self.ball = ball
    # Initialize state estimate
    self.x_hat = np.copy(x0)
    self.x_hat = self.x_hat[self.x_subset]
    # Settings (TODO: make prms)
    # desired observer poles
    zeta_obs1 = 0.9
    wn_obs1 = 10
    zeta_obs2 = 0.9
    wn_obs2 = 12
    zeta_obs3 = 0.9
    wn_obs3 = 14
    zeta_obs4 = 0.9
    wn_obs4 = 16
    p_obs5 = -18
    des_obsv_char_poly = np.convolve(
      np.convolve(
        np.convolve(
          np.convolve(
            [1, 2*zeta_obs1*wn_obs1, wn_obs1**2],
            [1, 2*zeta_obs2*wn_obs2, wn_obs2**2]),
          [1, 2*zeta_obs3*wn_obs3, wn_obs3**2]),
        [1, 2*zeta_obs4*wn_obs4, wn_obs4**2]),
      [1, -p_obs5])
    self.des_obsv_poles = np.roots(des_obsv_char_poly)
  
  def update(self, y_m, u):
    """ Update the state estimation
    
    y_m - measurement
    u - last control input
    """
    # In the future, maybe only update the linearization every once in a while
    self.update_ABCDL(self.x_hat, u)
    
    # TEMP DEBUGGING
    # self.xhdot(self.x_hat, y_m, u)
    # quit()
    
    # Integrate the observer ODE
    raw_x_hat = rk4(self.xhdot, self.x_hat, self.dt_obs, (y_m,u))
    # print(380, raw_x_hat)
    # q = cleanup_versor(np.quaternion(*raw_x_hat[0:4]))
    # print(382, q.w, q.x, q.y, q.z)
    # raw_x_hat[0:4] = [q.w, q.x, q.y, q.z]
    # print(384, raw_x_hat)
    # print(385, np.maximum(np.minimum(raw_x_hat, self.xhat_max), -self.xhat_max))
    self.x_hat = self.cleanup_xhat(raw_x_hat)
    # print(374, self.x_hat, self.xhat_max)
    return self.augment_xhat(self.x_hat)
  
  def cleanup_xhat(self, raw_x_hat):
    """ Take a raw x_hat vector and clean it up to counter numerical drift.
    
    Does not augment
    """
    
    x_hat = np.copy(raw_x_hat)
    # Normalize the versor to counter numerical drift
    q = np.quaternion(*x_hat[0:4])
    q = cleanup_versor(q)
    x_hat[0:4] = [q.w, q.x, q.y, q.z]
    # Enforce limits
    x_hat = np.maximum(np.minimum(x_hat, self.xhat_max), -self.xhat_max)
    return x_hat
  
  def augment_xhat(self, x_hat):
    """ Convert a subset x-vector x_hat to a full state vector
    """
    full_x = np.zeros(self.nx)
    full_x[self.x_subset] = x_hat
    return full_x
  
  def update_ABCDL(self, x, u):
    """ Calculate ABCD matrices @ observer gains L
    
    Assumes that x is an xhat, defining self.x_subset indices of the full
      state vector
    """
    
    # Augment the given xhat vector so it's the right size
    if x.size < self.nx:
      full_x = self.augment_xhat(x)
    else:
      full_x = x
    
    # Linearization
    A, B = self.ball.JABf(full_x, u=u)
    # print(A!=0)
    C, D = self.ball.mJCDf(full_x, u)
    # Store the most recent linearization point
    self.xe = np.copy(full_x)
    self.ue = np.copy(u)
    self.xde = self.ball.eom(full_x, u) # TODO: Repeated evaluation...
    self.yme = self.ball.measure(full_x, u)
    
    # Focus on q, omega, and alpha
    self.A = A[np.ix_(self.x_subset, self.x_subset)]
    self.B = B[self.x_subset]
    self.C = C[:,self.x_subset]
    self.D = D
    # print(self.A.shape, self.B.shape, self.C.shape, self.D.shape)
    #   (9, 9) (9,) (3, 9) (3,)
    
    # Observability matrix
    O = control.ctrb(self.A.T, self.C.T)
    self.cndO = np.linalg.cond(O)
    # print(O.shape, np.linalg.matrix_rank(O), np.linalg.cond(O))
    #   (9, 27) 9
    #   (9, 27) 7 -- if u is 0
    
    # TEMP: For testing
    # return np.linalg.matrix_rank(O)
    
    if np.linalg.matrix_rank(O) == self.x_subset.size:
      # System is observable. Calculate observer gains.
      self.L = control.place(self.A.T, self.C.T, self.des_obsv_poles).T
      # print(309, f"New L={np.array_repr(self.L)}")
      # print(309, f"Calculated new L. rank(O)={self.x_subset.size}")
    else:
      # System is not currently observable
      print(312, "System is not observable in the current configuration:"
        f"\n\tx={x}, u={u}"
        f"\n\trank(O)={np.linalg.matrix_rank(O)} / {self.x_subset.size}")
      # print(314, "Gains are being left as follows:"
        # f"\n\tL={self.L}")
    
    print(476, self.scale_L())
  
  def scale_L(self):
    """ Factor by which to attenuate the L gains.
    If cnd(O) is too large, don't try to use corrector term
    """
    return np.exp(-self.cndO/1e5)
  
  def xhdot(self, x_hat, y_m, u):
    # xhatdot = A*xhat + B*u + L(y - C*xhat - D*u)
    #   Note: y_pred = C*xhat - D*u
    #   Note: B*u needs to change to B@u if u becomes an array
    
    # DEBUGGING:
    # print(388, self.L.shape, self.A.shape, self.B.shape, self.C.shape, self.D.shape)
    # print(389, x_hat.shape, u, y_m.shape)
    # xh = self.augment_xhat(x_hat)
    # print(459, x_hat, y_m, u)
    # xt = x_hat - self.xe[self.x_subset]
    # ut = u - self.ue
    # print(
      # f"xhdot_AB: self.xde + self.A @ xt + self.B @ ut",
      # f"\n\t={self.xde[self.x_subset]} + {self.A @ xt} + {self.B * ut}",
      # f"\n\t={self.xde[self.x_subset] + self.A @ xt + self.B * ut}")
    # print(
      # f"xhdot_eom: {self.ball.eom(xh, u)}"
    # )
    
    # Difference from linearization point
    xh_tilde = x_hat - self.xe[self.x_subset]
    u_tilde = u - self.ue
    
    # xhd_pred = self.xde[self.x_subset] +  self.A @ xh_tilde + self.B * u_tilde
    # ym_pred = self.yme + self.C @ xh_tilde + self.D * u_tilde
    
    xhd_nlpred = self.ball.eom(self.augment_xhat(x_hat), u)[self.x_subset]
    ym_nlpred = self.ball.measure(self.augment_xhat(x_hat), u)
    
    # err_nl_xhd = np.linalg.norm(xhd_nlpred - xhd_pred)
    # err_nl_ym = np.linalg.norm(ym_nlpred - ym_pred)
    # print("Nonlinearity errors",
      # f"\n\txhd %err_nl = {100*err_nl_xhd / np.linalg.norm(xhd_nlpred)}",
      # f"\n\tym %err_nl = {100*err_nl_ym / np.linalg.norm(ym_nlpred)}")
    # print("Observer xhdot", f"\n\tPredictor: {xhd_nlpred}",
      # f"\n\tCorrector: {self.L @ (y_m - ym_nlpred)}",
      # f"\n\tym: {y_m}",
      # f"\n\tym_pred: {ym_nlpred}")
      
    # xhd_nlpred *= 1.0 + (np.random.rand(*xhd_nlpred.shape)*2-1)*1e-3 # TEMP, TESTING
    
    # print(517, y_m - ym_nlpred, self.L @ (y_m - ym_nlpred))
    
    # xhat_dot = xhd_pred + self.L @ (y_m - ym_pred)
    xhat_dot = xhd_nlpred + self.scale_L()*self.L @ (y_m - ym_nlpred)
    
    # print(f"%Innovation = {100*np.mean((self.L @ (y_m - ym_nlpred)) / xhat_dot):.2f}")
    
    # xhat_dot = self.A @ x_hat \
      # + self.B * u \
      # + self.L @ (y_m - self.C @ x_hat - self.D * u)
    
    # Saturate
    xhat_dot = np.maximum(np.minimum(
      xhat_dot, self.xhd_max), -self.xhd_max)
    
    return xhat_dot


class Obs_harm:
  """ Luenberger observer for a harmonic oscillator. For testing
  """
  
  dt_obs = 0.001 # May break if this is different from dt_dyn
  nx = 2
  # Parameters
  m = 1
  b = .1
  k = 1
  # Constant ABCD
  A = np.array([
    [0, 1],
    [-k/m, -b/m] # xdd = -k/m *x - b/m *xd
  ])
  B = np.array([
    0,
    1/m
  ])
  C = np.array([
    [1, 0]
  ])
  D = np.array([0])
  
  def __init__(self, x0):
    # Initialize state estimate
    self.x_hat = np.copy(x0)
    # desired observer poles
    zeta_obs1 = 0.9
    wn_obs1 = 4
    des_obsv_char_poly = [1, 2*zeta_obs1*wn_obs1, wn_obs1**2]
    self.des_obsv_poles = np.roots(des_obsv_char_poly)
    
    self.find_L()
  
  def update(self, y_m, u):
    """ Update the state estimation
    
    y_m - measurement
    u - last control input
    """
    
    
    # Integrate the observer ODE
    self.x_hat = rk4(self.xhdot, self.x_hat, self.dt_obs, (y_m,u))
    
    return self.x_hat
  
  
  def find_L(self):
    """ Calculate observer gains L
    
    Assumes that x is an xhat, defining self.x_subset indices of the full
      state vector
    """
    
    
    # Observability matrix
    O = control.ctrb(self.A.T, self.C.T)
    
    
    if np.linalg.matrix_rank(O) == self.nx:
      # System is observable. Calculate observer gains.
      self.L = control.place(self.A.T, self.C.T, self.des_obsv_poles).T
      # print(309, f"New L={np.array_repr(self.L)}")
      # print(309, f"Calculated new L. rank(O)={self.x_subset.size}")
    else:
      # System is not currently observable
      print(312, "System is not observable in the current configuration:"
        f"\n\tx={x}, u={u}"
        f"\n\trank(O)={np.linalg.matrix_rank(O)} / {self.nx}")
      # print(314, "Gains are being left as follows:"
        # f"\n\tL={self.L}")
      
    print(608, self.L)
  
  def xhdot(self, x_hat, y_m, u):
    
    # print(612, x_hat, u, y_m)
  
    xhd_pred = self.A @ x_hat + self.B * u
    ym_pred = self.C @ x_hat + self.D * u
    
    xhat_dot = xhd_pred + self.L @ (y_m - ym_pred)
    
    
    return xhat_dot

def obs_harm_test():
  
  def eom(x, u):
    xd = obs.A @ x + obs.B * u
    return xd
  
  def measure(x, u):
    return obs.C @ x + obs.D * u + (np.random.rand()*2-1)*1e-1
  
  def control(t, x):
    return np.sin(t)*.01 + 1
  
  
  dt_dyn = 0.001
  u = 0.0
  x = np.array([0.0,0.0])
  obs = Obs_harm(np.copy(x)+(np.random.rand(2)*2-1)*1e-2)
  t_max = 8
  # Full time vector
  v_t = np.arange(0, t_max, dt_dyn)
  # Store everything
  v_x = np.empty((v_t.size, 2))
  v_u = np.empty((v_t.size, 1))
  v_ym = np.empty((v_t.size, 1))
  v_xhat = np.empty((v_t.size, 2))
  
  ## Simulation loop
  for i,t in enumerate(v_t):
    # Update dynamics
    x = rk4(eom, x, dt_dyn, (u,))
    # Record simulated measurement data
    ym = measure(x, u)
    
    # Update observer
    x_hat = obs.update(ym, u)
    
    # Store state, input, and measurement at every timestep
    v_x[i] = x
    v_u[i] = u
    v_ym[i] = ym
    v_xhat[i] = x_hat
    
    # Update control
    u = control(t, x_hat)
  
  # Plot
  from matplotlib import pyplot as plt
  fig, axs = plt.subplots(2,1)
  axs[0].plot(v_t, v_x[:,0], color="red", linestyle="-")
  axs[0].plot(v_t, v_x[:,1], color="blue", linestyle="-")
  axs[0].plot(v_t, v_xhat[:,0], color="red", linestyle="--")
  axs[0].plot(v_t, v_xhat[:,1], color="blue", linestyle="--")
  # axs[1].plot(v_t, v_u)
  axs[1].plot(v_t, v_x[:,0] - v_xhat[:,0], color="red", linestyle="-")
  axs[1].plot(v_t, v_x[:,1] - v_xhat[:,1], color="blue", linestyle="-")
  
  fig.show()
  input("PAUSE")


if False: # OLD MPC - Copied from Simulation

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

def obsv_test(ball):
  # Observability test: How often is it observable?
  
  N = 1000
  span = np.array([1, 1, 1, 1, 30, 30, 30, 0, 0, 100, 12])
  x0_rand = np.random.rand(N, 11)*span-span/2
  # Cleanup Q
  for i, x0i in enumerate(x0_rand):
    q = quaternion.from_rotation_vector(x0i[0:3])
    x0_rand[i,0:4] = [q.w, q.x, q.y, q.z]
  u_rand = np.random.rand(N)*2-1
  
  successses = 0
  failures = 0
  
  for x0, u in zip(x0_rand, u_rand):
    obs = Observer(ball, x0)
    rnk = obs.update_ABCDL(x0, u)
    if rnk == 9:
      successses += 1
    else:
      failures += 1
  
  print(f"successes: {successses}/{N}; failures: {failures}/{N}")

if __name__ == "__main__":
  obs_harm_test()
  quit()
  
  
  from CMGBall import CMGBall
  ball = CMGBall(ra=np.array([0.02, 0, 0]))
  # v_ref = lambda t: np.array([2,1]) * np.cos(t)
  # cnt = FF(ball, v_ref)
  
  obsv_test(ball)
  quit()
  
  
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
  
  obs = Observer(ball, x0)
  obs.update_ABCDL(x0, u)
