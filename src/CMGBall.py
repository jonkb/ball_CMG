"""
CMGBall class to hold parameters of the robot
"""
import numpy as np
import quaternion
from util import sharpn, flatn
import dynamics as dyn

class CMGBall:
  """
  Stores the parameters of the CMG ball robot. Also stores the dynamics, as
    loaded from file.
  
  TODO: Use sensor measurements for control instead of full state vector. 
    (I.e. observer-based control)
  """
  
  def __init__(self, Is=.001, Ig1=.001, Ig2=.001, m=1, Rs=0.05, Omega_g=600, 
      km=20, ra=np.array([0, 0, 0])):
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
    This variable attempts to represent that conversion.
    TODO: Maybe represent that this mapping is more complex & nonlinear, 
    including the effects of the gearbox.
    """
    self.km = km
    self.ra = ra
    
    # Load dynamics equations
    M, F = dyn.load_MF()
    self.Mf, self.Ff = dyn.lambdify_MF(M, F, self)
    ax, ay = dyn.load_axay()
    self.axf, self.ayf = dyn.lambdify_axay(ax, ay, self)
  
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
    M = self.Mf(*x, alphadd)
    F = self.Ff(*x, alphadd)
    # [M]{qddot} = {F}
    # TODO: The mass matrix could be pre-inverted to speed this up
    sol = np.linalg.lstsq(M, F, rcond=None)
    # d/dt([omega_x, omega_y, omega_z]) = qddot = sol[0]
    # (This is because the omegas are generalized velocities)
    xd[4:7] = sol[0][:,0]
    
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
    g = 9.8
    
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
    rdd_s__0 = np.array([rdd_x, rdd_y, g])
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
    
  def to_ball(self):
    """
    Return a CMGBall object
    """
    ball = CMGBall(Is=self.Is, Ig1=self.Ig1, Ig2=self.Ig2, m=self.m, 
      Rs=self.Rs, Omega_g=self.Omega_g, km=self.km)
    return ball
  

if __name__ == "__main__":
  ball = CMGBall(ra=np.array([100, 0, 0]))
  
  # State: Initial state, but with an alphad
  x0 = np.zeros(11)
  x0[0] = 1
  x0[9] = 90 * np.pi/180 # alpha
  x0[10] = 5 * np.pi/180 # alphad
  u = 0.0 # pwm input for alphadd
  y_m = ball.measure(x0, u)
  
  print("y_m: ", y_m)
  
  print("DONE")
  