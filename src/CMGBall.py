"""
CMGBall class to hold parameters of the robot
"""
import numpy as np
import quaternion
from util import sharpn, flatn
import dynamics as dyn

class CMGBall:
  """
  Stores the parameters of the CMG ball robot.
  NOTE: Could also organize this such that the equations of motion are stored
    in this object as well, but I think I'd rather keep them separate.
  """
  def __init__(self, Is=.001, Ig1=.001, Ig2=.001, m=1, Rs=0.05, Omega_g=600, 
      km=20):
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
    q = np.quaternion(x[0], x[1], x[2], x[3])
    omega_s__s = [x[4], x[5], x[6]]
    omega_s__0 = flatn(q * sharpn(omega_s__s) * q.conjugate())
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


if __name__ == "__main__":
  ball = CMGBall()
  print("DONE")
  