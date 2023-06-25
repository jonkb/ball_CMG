"""
Observers
"""

import numpy as np
import quaternion
from sklearn import linear_model, neural_network
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import qmc
import control

from diff import rk4
from util import cleanup_versor


class Observer:
  """ Parent class, not to be instantiated
  """
  
  dt_obs = 0.001
  nx = 11
  x_subset = np.array([0,1,2,3,4,5,6,9,10]) # Focus on q, omega, and alpha
  n_xhat = x_subset.size
  # Limits on quantities in xhat
  xhat_max = np.array([1.,1.,1.,1., 20.,20.,20., np.inf,120.])
  
  def augment_xhat(self, x_hat):
    """ Convert a subset x-vector x_hat to a full state vector
    """
    full_x = np.zeros(self.nx)
    full_x[self.x_subset] = x_hat
    return full_x
    
  def cleanup_xhat(self, raw_x_hat, clip=True):
    """ Take a raw x_hat vector and clean it up to counter numerical drift.
    
    Does not augment
    """
    
    x_hat = np.copy(raw_x_hat)
    # Normalize the versor to counter numerical drift
    q = np.quaternion(*x_hat[0:4])
    q = cleanup_versor(q)
    x_hat[0:4] = [q.w, q.x, q.y, q.z]
    if clip:
      # Enforce limits
      x_hat = np.maximum(np.minimum(x_hat, self.xhat_max), -self.xhat_max)
    return x_hat

class Luenberger(Observer):
  """ Luenberger observer. To be used by the controllers.
  """
  
  dt_obs = 0.001
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
    self.nx = ball.n_x
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
    """ Update the state estimate
    
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
    self.yme = self.ball.measure(full_x, u, add_noise=False)
    
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
    ym_nlpred = self.ball.measure(self.augment_xhat(x_hat), u, add_noise=False)
    
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

class ObsML(Observer):
  """ Optimal controller implemented with Machine Learning
  """
  
  dt_obs = 0.001
  # How many points to sample in 19-D (x, xhat, u) space. 
  #   72 would be every 5 deg if each of x were an euler angle
  n_state_sample = 10*360*19
  # Bounds for that sample
  omega_max = 6*np.pi
  u_max = 1
  xhat_ub = np.array([1, 1, 1, 1, omega_max, omega_max, omega_max, 
    2*np.pi, 4*np.pi])
  state_ub = np.concatenate([xhat_ub, xhat_ub, [u_max]])
  state_lb = -state_ub
  # Attempt to create error dynamics with this pole
  obs_pole = -50
  # NN Parameters
  hidden_layer_sizes = (4,)
  
  def __init__(self, ball, x0):
    self.x_hat = np.copy(x0)
    self.x_hat = self.x_hat[self.x_subset]
    self.ball = ball
    self.nx = ball.n_x
    # Train & store model
    self.setup()
  
  def setup(self):
    """ Train NN for the dynamics of the given ball
    
    xhat-dot = NN(xhat, u, ym)
      -> For a given state estimate, input, & measurement, what's the optimal
        observer function?
    """
    
    # Generate data to train NN on
    #   This step is pretty fast: ~0.5s
    print(315)
    features, f_des = self.gen_opt_data()
    # Augment the feature space
    poly = PolynomialFeatures(2, interaction_only=True)
    features = poly.fit_transform(features)
    print(features.shape)
    print(319)
    
    # Train the model
    # self.reg = linear_model.LinearRegression()
    self.reg = linear_model.Ridge()
    # self.reg = neural_network.MLPRegressor(max_iter=5000,
      # hidden_layer_sizes=self.hidden_layer_sizes)
    self.model = self.reg.fit(features, f_des)
    
    if True:
      print(323)
      # Evaluate the model performance
      y_pred = self.model.predict(features)
      residuals = y_pred - f_des
      r2 = 1 - np.var(residuals) / np.var(f_des - f_des.mean())
      print(f"r^2: {r2}")
      
      cv_scores = cross_validate(self.reg, features, f_des, cv=5, 
        scoring="r2", return_train_score=True)
      # print(cv_scores)
      print(cv_scores["train_score"].mean())
      print(cv_scores["test_score"].mean())
      
      # For optimization
      return cv_scores["test_score"].mean()
  
  def prm_sweep(self):
    # (130,) : -0.1 is as good as it gets for one layer
    # (50, X, 50): X=5: -.07
    # (25, 5, 25): -.15
    v_hl = np.arange(2, 7, 1)
    v_ts = np.zeros_like(v_hl, dtype=float)
    for i, hl in enumerate(v_hl):
      self.hidden_layer_sizes = (hl,)# 5, 30)
      v_ts[i] = self.setup()
    
    import matplotlib.pyplot as plt
    plt.plot(v_hl, v_ts)
    plt.show()
  
  def gen_opt_data(self):
    """ Generate the dataset used to train the NN observer
    
    Feature space: xhat, u, ym_err
      ym_err = ym - ymhat = ym - fm(xhat, u)
    Output: xhat-dot
    """
    
    # Sample the space (x, xhat, u)
    #   NOTE: I'm going to set the unobserved variables r_x to zero
    dim = 2*self.n_xhat + self.ball.n_u
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n=self.n_state_sample)
    states = qmc.scale(sample, self.state_lb, self.state_ub)
    
    # Observer function
    #   xhat-dot desired = f_eom + f_des
    f_des = np.empty((self.n_state_sample, self.n_xhat))
    ym_err = np.empty((self.n_state_sample, self.ball.n_ym))
    for i, state in enumerate(states):
      # Unpack
      x = state[0:self.n_xhat]
      x = self.cleanup_xhat(x, clip=False)
      x_hat = state[self.n_xhat:2*self.n_xhat]
      x_hat = self.cleanup_xhat(x_hat, clip=False)
      # ONLY SUPPORTS 1 input for now TODO
      u = state[2*self.n_xhat:][0]
      # Run through EOM
      ym = self.ball.measure(self.augment_xhat(x), u, 
        add_noise=False) # "actual" measurement
      ym_hat = self.ball.measure(self.augment_xhat(x_hat), u, 
        add_noise=False) # observer-predicted measurement
      ym_err[i,:] = ym - ym_hat
      # This should make the error dynamics stable if obs_pole < 0
      f_des[i,:] = - self.obs_pole*(x-x_hat)
    
    # Feature space: xhat, u, ym
    features = np.hstack([
      states[:, self.n_xhat:2*self.n_xhat], #xhat
      np.vstack(states[:, 2*self.n_xhat]), #u
      ym_err])
    
    print(407, features.shape, f_des.shape)
    
    return features, f_des
  
  def update(self, y_m, u):
    """ Update the state estimate
    
    y_m - measurement
    u - last control input
    """
    
    # Integrate the observer ODE
    raw_x_hat = rk4(self.xhdot, self.x_hat, self.dt_obs, (y_m,u))
    self.x_hat = self.cleanup_xhat(raw_x_hat)
    return self.augment_xhat(self.x_hat)
  
  def xhdot(self, x_hat, y_m, u):
    """ Returns x-hat-dot
    TODO NOTE: This would probably be better if the model learned the corrector
      without the predicter. Then we're not re-learning the EOM. This may be
      faster to evaluate though
    """
    
    f_eom = self.ball.eom(self.augment_xhat(x_hat), u)
    ym_nlpred = self.ball.measure(self.augment_xhat(x_hat), u, xd=f_eom, 
      add_noise=False)
    ym_err = y_m - ym_nlpred
    
    # print(435, x_hat.shape, y_m.shape)
    # Use the trained model to get x-hat-dot
    x_in = np.concatenate([x_hat, [u], ym_err]).reshape(1,-1)
    # Augment the feature space
    poly = PolynomialFeatures(2, interaction_only=True)
    features = poly.fit_transform(x_in)
    f_obs = self.model.predict(features)[0]
    # print(439, xhd)
    xhd = f_eom[self.x_subset] + f_obs
    # print(xhd.shape)
    xhd = self.cleanup_xhat(xhd)
    return xhd

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
    obs = Luenberger(ball, x0)
    rnk = obs.update_ABCDL(x0, u)
    if rnk == 9:
      successses += 1
    else:
      failures += 1
  
  print(f"successes: {successses}/{N}; failures: {failures}/{N}")

if __name__ == "__main__":
  from CMGBall import CMGBall
  ball = CMGBall(ra=np.array([0.02, 0, 0]))
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
  
  obs = ObsML(ball, x0)
  # obs.prm_sweep()
  quit()
  
  obs_harm_test()
  
  obsv_test(ball)
  
  obs = Luenberger(ball, x0)
  obs.update_ABCDL(x0, u)
  
