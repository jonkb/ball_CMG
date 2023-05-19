"""
CMGBall class to hold parameters of the robot
"""
import numpy as np
import quaternion
from util import sharpn, flatn, ISOnow
import dynamics as dyn
from diff import FD # Finite differencing

import dill

class CMGBall:
  """
  Stores the parameters of the CMG ball robot. Also stores the dynamics, as
    loaded from file.
  """
  # Constants
  n_x = 11 # Length of state vector
  n_u = 1 # Number of inputs. Currently only n_u=1 is supported
  n_ym = 3 # Number of measurements.
  g = 9.8 # Gravity
  
  def __init__(self, Is=.001, Ig1=.001, Ig2=.001, m=1, Rs=0.05, Omega_g=600, 
      km=20, ra=np.array([0, 0, 0]), lams=None):
    """ All parameters are floats:
    Is: Moment of inertia for the sphere
      Translates to [Is]=diag([Is, Is, Is]) because of symmetry
    Ig1, Ig2: Moments of inertia for the Gyroscope
      Translates to [Ig]=diag([Ig1, Ig2, Ig2])
      Ig1 is the moment of inertia about the axis of rotation
    m: Mass of the sphere + mass of the gyro
    Rs: Radius of the sphere
    Omega_g: Constant angular velocity of the gyroscope
      Presumably this is maintained by a motor spinning at a constant speed
    km: Motor constant, mapping pwm to alphadd. This is also equal to
      alphadd_max: Max alpha-double-dot (rad/s^2)
    ra: Position vector of accelerometer, relative to the center of the sphere,
      expressed in the accel-fixed "a"-frame.
    """
    # Constants
    self.Is = Is
    self.Ig1 = Ig1
    self.Ig2 = Ig2
    self.m = m
    self.Rs = Rs
    self.Omega_g = Omega_g
    self.alphadd_max = km
    """
    The physical system maps pwm to voltage to alpha motor torque and from
    torque to acceleration.
    This km variable attempts to represent that conversion.
    TODO: Maybe represent that this mapping is more complex & nonlinear, 
    including the effects of the gearbox.
    """
    self.km = km
    self.ra = ra
    
    if lams is None:
      # Load dynamics equations
      M, F = dyn.load_MF() #OLD
      self.Mf, self.Ff = dyn.lambdify_MF(M, F, self)
      ax, ay = dyn.load_axay()
      self.axf, self.ayf = dyn.lambdify_axay(ax, ay, self)
      # NOTE: EOMf lets you directly calculate omega_dot, but it's
      #   10x slower than the M\F method.
      #   Same story with the analytical Jacobians JAf & JBf
      # EOM = dyn.load_EOM()
      # self.EOMf = dyn.lambdify_EOM(EOM, self)
      # Load Jacobian-linearized A & B matrices
      # JA, JB = dyn.load_JAB()
      # self.JAf, self.JBf = dyn.lambdify_JAB(JA, JB, self)
    else:
      # Read in the lambdified equations from the lams dictionary
      self.axf = lams["axf"]
      self.ayf = lams["ayf"]
      self.Mf = lams["Mf"]
      self.Ff = lams["Ff"]
      # It's cool that I could make these, but they seem to be much slower
      #   than numerical manipulation after calling Mf & Ff
      # self.EOMf = lams["EOMf"]
      # self.JAf = lams["JAf"]
      # self.JBf = lams["JBf"]
  
  def __str__(self):
    s = "CMG Ball Object\n"
    s += f"\tIs={self.Is}\n"
    s += f"\tIg1={self.Ig1}\n"
    s += f"\tIg2=Ig3={self.Ig2}\n"
    s += f"\tm={self.m}\n"
    s += f"\tRs={self.Rs}\n"
    s += f"\tOmega_g={self.Omega_g}\n"
    s += f"\talphadd_max={self.alphadd_max}\n"
    return s
  
  def serializable(self):
    """ Return a SerializableBall version of this CMGBall
    """
    return SerializableBall(self)
  
  def aa2pwm(self, alphadd):
    """
    Map desired alphadd to the corresponding pwm
    
    Returns
    -------
    pwm
    pwmsat (btw -1 & 1)
    """
    pwm = alphadd / self.km
    pwmsat = min(1, max(-1, pwm))
    return pwm, pwmsat
  
  def pwm2aa(self, pwm):
    """
    Map pwm to alphadd (saturating pwm @ +-100%)
    
    Returns
    -------
    alphadd
    """
    pwmsat = min(1, max(-1, pwm))
    alphadd = pwmsat * self.km
    return alphadd
  
  def x2v(self, x):
    """ Pull out the linear velocity from the state vector x
    
    NOTE: These lines are repeated in eom, but this allows them to be 
      calculated without the rest of the EOM
    """
    Rs = self.Rs
    # Frame conversion (i.e. passive rotation)
    q = np.quaternion(x[0], x[1], x[2], x[3])
    omega_s__s = [x[4], x[5], x[6]]
    omega_s__0 = flatn(q * sharpn(omega_s__s) * q.conjugate())
    vx = Rs*omega_s__0[1]
    vy = -Rs*omega_s__0[0]
    return np.array([vx, vy])
  
  def x2a(self, x, u, xd=None, proper=False):
    """ Calculate the linear acceleration from the state vector x
    (In the 0-frame)
    
    if proper == True, return "proper" acceleration, including gravity.
      Otherwise, don't include gravity
      
    NOTE: This is repeated as part of measure
    """
    
    if xd is None:
      # Acceleration depends on x-dot
      xd = self.eom(x, u)
    
    # Extract info from x & xd
    # q: active rotation from 0 to s or passive rotation from s to 0
    q_s0 = np.quaternion(x[0], x[1], x[2], x[3])
    omegad_s__s = xd[4:7]
    
    # Rotations
    #   p' = q p q* (normal conjugation)
    omegad_s__0 = flatn(q_s0 * sharpn(omegad_s__s) * q_s0.conjugate())
    
    # Linear acceleration of sphere
    rdd_x = Rs*omegad_s__0[1]
    rdd_y = -Rs*omegad_s__0[0]
    #   If proper acceleration, include gravity
    rdd_s__0 = np.array([rdd_x, rdd_y, (self.g if proper else 0)])
    return rdd_s__0
  
  def mJCDf(self, x, u):
    """ Return the C & D matrices of the linearized system
    These come from the jacobian of the measure function.
    Currently, this is using finite differencing, because the measure
      function is so gross that its jacobian is going to be a nightmare.
    """
    
    # Redefine measure with augmented xa vector, xa=[x,u]
    ymf = lambda xa: ball.measure(xa[0:11], xa[11])[0]
    x0a = np.concatenate((x, [u]))
    mJ = FD(ymf, x0a)
    mJC = mJ[:,0:11]
    mJD = mJ[:,11]
    
    return mJC, mJD
  
  def JABf(self, x, u=None, alphadd=None, analytical=False):
    """ Return the A & B matrices of the linearized system
    These come from the jacobian of the eom function.
    """
    
    if analytical:
      # Analytical jacobian
      alphadd = self.pwm2aa(u) if (alphadd is None) else alphadd
      A = self.JAf(*x, alphadd)
      B = self.JBf(*x, alphadd)
      return A, B
    else:
      # Finite differencing of self.eom
      u = self.aa2pwm(u) if (u is None) else u
      # Redefine eom with augmented xa vector, xa=[x,u]
      eomf = lambda xa: ball.eom(xa[0:11], xa[11])
      x0a = np.concatenate((x, [u]))
      J = FD(eomf, x0a)
      JA = J[:,0:11]
      JB = J[:,11]
      return JA, JB
  
  def omegadot(self, x, u=None, alphadd=None):
    """ Find omega-dot from the EOM
    """
    
    alphadd = self.pwm2aa(u) if (alphadd is None) else alphadd
    
    # solve [M]{qddot} = {F}
    #   d/dt([omega_x, omega_y, omega_z]) = qddot = sol[0]
    #   (This is because the omegas are generalized velocities)
    M = self.Mf(*x, alphadd)
    F = self.Ff(*x, alphadd)
    sol = np.linalg.solve(M, F)
    
    # Direct method, using pre-solved symbolic EOM
    #   Ended up 10x slower
    # omega_dot = self.EOMf(*x, alphadd).flatten() # Direct method. 10x slower
    # return omega_dot
    
    # Least-squares solution of [M]{qddot} = {F} 
    # sol = np.linalg.lstsq(M, F, rcond=None)
    # return sol[0][:,0]
    
    return sol[:,0]
  
  def eom(self, x, u):
    """ Evaluate state-variable EOM
    
    Parameters
    ----------
    x is a (11,) numpy array of the state variables
    x = (eta, ex, ey, ez, omega_x, omega_y, omega_z, rx, ry, alpha, alphad)
    
    u (float): pwm input to alpha motor
    
    Returns
    -------
    xd (11,) numpy array: time derivative of state vector
    """
    
    xd = np.zeros(11)
    # TODO: Could add Omega-dot as an input, then u would be a vector
    alphadd = self.pwm2aa(u)
    Rs = self.Rs
    
    # Equations 1-4: Orientation quaternion:
    # q: active rotation from 0 to s or passive rotation from s to 0
    q = np.quaternion(x[0], x[1], x[2], x[3])
    omega_s__s = [x[4], x[5], x[6]]
    omega_s__0 = flatn(q * sharpn(omega_s__s) * q.conjugate())
    # I believe this formula assumes that q is an active rotation
    #   By that I mean that p_rot__0 = q p_initial__0 q*
    qdot = sharpn(omega_s__0) * q / 2
    xd[0] = qdot.w
    xd[1] = qdot.x
    xd[2] = qdot.y
    xd[3] = qdot.z
    
    # Equations 5-7: omega-dot from EOM.
    xd[4:7] = self.omegadot(x, alphadd=alphadd)
    
    # Equations 8-9: rx, ry
    # These come from the constraint equation: (-Rk) x Omega
    xd[7] = Rs*omega_s__0[1]
    xd[8] = -Rs*omega_s__0[0]
    
    # Equations 10-11: alpha, alphad
    #   Integrate the provided alphadd input
    xd[9] = x[10]
    xd[10] = alphadd
    
    return xd
  
  def measure(self, x, u, xd=None, sensors=["accel"]):
    """ Simulate a sensor measurement at the given state
    
    Accelerometer
      rdd_a = rdd_s + rdd_{a/s}
      rdd_{a/s} = wa x (wa x r_a/s)  (in a-frame)
    
    TODO: Gyro, Magnetometer, GPS, +Noise (optional)
    """
    
    outputs = []
    
    # Constants
    khat = np.array([0,0,1])
    Rs = self.Rs
    
    if xd is None:
      # Acceleration measurements depend on x-dot
      xd = self.eom(x, u)
    
    # Extract info from x & xd
    # q: active rotation from 0 to s or passive rotation from s to 0
    q_s0 = np.quaternion(x[0], x[1], x[2], x[3])
    omega_s__s = x[4:7]
    omegad_s__s = xd[4:7]
    alpha = x[9]
    alphad = x[10]
    
    # Rotations
    #   p' = q p q* (normal conjugation)
    omega_s__0 = flatn(q_s0 * sharpn(omega_s__s) * q_s0.conjugate())
    omegad_s__0 = flatn(q_s0 * sharpn(omegad_s__s) * q_s0.conjugate())
    # q_sa: Passive rotation from s to a
    #   from_rotation_vector uses active convention
    q_sa = quaternion.from_rotation_vector([0,0,-alpha])
    omega_s__a = flatn(q_sa * sharpn(omega_s__s) * q_sa.conjugate())
    
    # Linear acceleration of sphere
    rdd_x = Rs*omegad_s__0[1]
    rdd_y = -Rs*omegad_s__0[0]
    # NOTE: The accelerometer measures gravity as well
    rdd_s__0 = np.array([rdd_x, rdd_y, self.g])
    # Vector addition for acceleration of accel
    #   The following are all in the a-frame
    rdd_s__a = flatn(q_sa * q_s0.conjugate() * 
      sharpn(rdd_s__0) * q_s0 * q_sa.conjugate())
    omega_a__a = omega_s__a + alphad*khat
    # Acceleration of accel relative to sphere. \ddot{r}_{a/s}
    rdd_ars = np.cross(omega_a__a, np.cross(omega_a__a, self.ra))
    rdd_a = rdd_s__a + rdd_ars
    
    for sensor in sensors:
      if sensor == "accel":
        outputs.append(rdd_a)
    
    return outputs

class SerializableBall:
  def __init__(self, ball):
    """
    Convert a CMGBall object (ball) to a serializable form.
    This allows it to be saved to file.
    The lambdified sympy functions have a hard time being serialized, so leave 
      them out.
    """
    
    # Constants
    self.Is = ball.Is
    self.Ig1 = ball.Ig1
    self.Ig2 = ball.Ig2
    self.m = ball.m
    self.Rs = ball.Rs
    self.Omega_g = ball.Omega_g
    self.alphadd_max = ball.km
    self.km = ball.km
    self.ra = ball.ra
    # Current timestamp
    self.dtstr = ISOnow()
    self.fname = f"ball_{self.dtstr}.dill"
    # Save lambdified functions
    self.save_lams(ball)
  
  def save_lams(self, ball):
    """ Save the lambdified functions to file
    """
    
    # Filenames in which the lambdified functions will be saved
    
    self.lams_fnames = {
      "axf": f"axf_{self.dtstr}.dill",
      "ayf": f"ayf_{self.dtstr}.dill",
      "Mf": f"Mf_{self.dtstr}.dill",
      "Ff": f"Ff_{self.dtstr}.dill",
      # "EOMf": f"EOMf_{self.dtstr}.dill",
      # "JAf": f"JAf_{self.dtstr}.dill",
      # "JBf": f"JBf_{self.dtstr}.dill"
    }
    dill.settings['recurse'] = True
    # Save them
    for lfstr, fname in self.lams_fnames.items():
      with open(fname,"wb") as file:
        lf = getattr(ball, lfstr)
        dill.dump(lf, file)
    print("Lambdified functions saved to the following files:")
    print(self.lams_fnames)
  
  def load_lams(self):
    """ Load the lambdified functions from file
    """
    lams = self.lams_fnames.copy()
    
    for lfstr, fname in self.lams_fnames.items():
      with open(fname,"rb") as file:
        lams[lfstr] = dill.load(file)
    
    return lams
  
  def to_ball(self):
    """
    Return a CMGBall object
    """
    lams = self.load_lams()
    ball = CMGBall(Is=self.Is, Ig1=self.Ig1, Ig2=self.Ig2, m=self.m, 
      Rs=self.Rs, Omega_g=self.Omega_g, km=self.km, lams=lams)
    return ball
  
  #TODO: recursive dill whole SerializableBall
  
  # @classmethod
  # def save_recursive(ball):
  # """ Save a ball object, along with the lambdified functions and all.
  
  # Downsides: BIG file (~5MB), depends on all the other python files in this
    # folder --> may break if versions change.
  # Upside: No need to reload the dynamics & re-lambdify the EOMs (~3.5 min process)
  # """
  # fname = f"ball_{ISOnow()}.dill"
  # with open(fname,"wb") as file:
    # dill.dump(ball, file)

def timing_test_JAB(ball, x):
  """ Time many executions of JAB
  """
  
  from util import tic, toc
  times = tic()
  
  N = 200
  v_u = np.random.rand(N)*2-1
  
  JA_a = np.zeros((N,))
  JA_n = np.zeros((N,))
  
  for i, u in enumerate(v_u):
    A, B = ball.JABf(x, u=u, analytical=True)
    JA_a[i] = np.linalg.norm(A)
  toc(times, f"{N}x JA (analytical)")
  
  for i, u in enumerate(v_u):
    A, B = ball.JABf(x, u=u, analytical=False)
    JA_n[i] = np.linalg.norm(A)
  toc(times, f"{N}x JA (numerical)")
  
  print("Max difference btw a & n", 
    np.max(np.abs(JA_a - JA_n)))
  print("DONE")

def timing_test(Mf, Ff, EOMf, x):
  """ Compare old EOM to new EOM
  
  Results
    All methods are essentially equivalent, in the results they produce.
    EOMf is ~10x slower
    lstsq is maybe ~15-20% slower than solve. It may be more robust though?
      Well, I think M should always be invertible.
  """
  
  from util import tic, toc
  times = tic()
  
  N = 1000
  v_add = np.random.rand(N)*10-5
  # wd_EOM = np.zeros((N,3))
  wd_MFl = np.zeros((N,3))
  wd_MF = np.zeros((N,3))
  
  # for i, alphadd in enumerate(v_add):
    # omega_dot = EOMf(*x, alphadd).flatten()
    # wd_EOM[i] = omega_dot
  # toc(times, f"{N}x EOMf")
  
  for i, alphadd in enumerate(v_add):
    M = Mf(*x, alphadd)
    F = Ff(*x, alphadd)
    sol = np.linalg.lstsq(M, F, rcond=None)
    wd_MFl[i] = sol[0][:,0]
  toc(times, f"{N}x M\F (lstsq)")
  
  for i, alphadd in enumerate(v_add):
    M = Mf(*x, alphadd)
    F = Ff(*x, alphadd)
    sol = np.linalg.solve(M, F)
    wd_MF[i] = sol[:,0]
  toc(times, f"{N}x M\F")
  
  print("Max difference btw lstsq & solve", 
    np.max(np.abs(wd_MFl - wd_MF)))
  
  print("DONE")

if __name__ == "__main__":

  ## Setup
  from util import tic, toc
  times = tic()
  # Load a pre-generated ball object
  fname = "ball_20230518T223742.dill"
  with open(fname,"rb") as file:
    sball = dill.load(file)
    ball = sball.to_ball()
  
  # ball = CMGBall(ra=np.array([0.1, 0, 0]))
  toc(times, "Initializing CMGBall")
  
  # State: Initial state, but with an alphad
  x0 = np.zeros(11)
  x0[0] = 1
  x0[9] = 90 * np.pi/180 # alpha
  x0[10] = 5 * np.pi/180 # alphad
  u = 0.0 # pwm input for alphadd
  alphadd = ball.pwm2aa(u)
  
  
  ## Speed test
  # Load dynamics equations
  # M, F = dyn.load_MF()
  # Mf, Ff = dyn.lambdify_MF(M, F, ball)
  timing_test_JAB(ball, x0)
  quit()
  
  
  ## Test differentiation of ball.measure
  # Test whether measure works with complex arguments -- nah
  # y_m = ball.measure(x0, u)
  # print("y_m: ", y_m)
  # xi = x0 + np.ones_like(x0)*(.01j)
  # ui = u + .01j
  # y_mi = ball.measure(xi, ui)
  # print("y_mi: ", y_mi)
  from diff import FD
  # Redefine measure with augmented xa vector, xa=[x,u]
  mf = lambda xa: ball.measure(xa[0:11], xa[11])[0]
  x0a = np.concatenate((x0, [u]))
  print(x0a.shape)
  mJ = FD(mf, x0a)
  print(mJ.shape)
  mJC = mJ[:,0:11]
  print(mJC.shape)
  mJD = mJ[:,11]
  print(mJD.shape)
  toc(times, "FD Jacobian of measure")
  
  quit()
  
  
  
  # Save this object
  # import dill
  # dill.settings['recurse'] = True
  # fname = "ball_20230518.dill"
  # with open(fname,"wb") as file:
    # dill.dump(ball, file)
  # ball.save_recursive()
  sball = SerializableBall(ball)
  with open(sball.fname,"wb") as file:
    dill.dump(sball, file)
  toc(times, "Saving ball to file")
  quit()
  
  
  ## Test Jacobian linearization of A & B matrices
  print("Evaluating Linearized A & B matrices")
  A = ball.JAf(*x0, alphadd)
  toc(times, "Evaluating JA(x0)")
  B = ball.JBf(*x0, alphadd)
  toc(times, "Evaluating JB(x0)")
  print(309, A)
  print(310, B)
  toc(times, "Total", total=True)
  
  
  ## Test dynamics
  # Load sympy omegadot
  import sp_namespace as spn
  import sympy as sp
  times = tic()
  with open("tmp_omega_dot.srepr", "r") as file:
    wd_str = file.read()
    wd = sp.sympify(wd_str)
  # Make the substitutions for everything that's not a state variable
  consts = {spn.Is: ball.Is, spn.Ig1: ball.Ig1, spn.Ig2: ball.Ig2, 
    spn.m: ball.m, spn.Rs: ball.Rs, spn.Omega_g: ball.Omega_g}
  wd = wd.subs(consts)
  wdf = sp.lambdify(spn.xs, wd, "numpy")
  wd_sp = wdf(*x0, alphadd)
  # Using current EOM
  xd = ball.eom(x0, u)
  print(f"wdot (eom): {xd[4:7]}")
  print(f"wdot (sp): {wd_sp}")
  # They're the same! Good!
  
  # Test measure
  y_m = ball.measure(x0, u)
  
  print("y_m: ", y_m)
  
  print("DONE")
  