"""
CMGBall class to hold parameters of the robot
"""

class CMGBall:
  """
  Stores the parameters of the CMG ball robot.
  NOTE: Could also organize this such that the equations of motion are stored
    in this object as well, but I think I'd rather keep them separate.
  """
  def __init__(Is=.001, Ig1=.001, Ig2=.001, m=1, Rs=0.05, Omega_g=600):
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
    """
    self.Is = Is
    self.Ig1 = Ig1
    self.Ig2 = Ig2
    self.m = m
    self.Rs = Rs
    self.Omega_g = Omega_g
  