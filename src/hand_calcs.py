""" Hand calculations for torque & stress &ct
"""

import numpy as np
pi = np.pi

def get_prnt(disp):
  # Conditional print wrapper
  def prnt(*s):
    if disp:
      print(*s)
  return prnt
  
class Ball:
  ## Container for parameters
  disp = True
  g = 9.8
  # Uncertain
  mu = 0.01
  # Variable
  R_G = 20e-3
  t_G = 15e-3
  d_bearings = 0.028
  # Locked in
  R_s = 2*1e-3
  rho_G = 7.85e3
  omega_NL_G = 24000*2*pi/60
  T_stall_G = 2e-3
  S_y = 350e6 # Steel
  omega_NL_alpha = 100*2*pi/60
  T_stall_alpha = 7e-2
  N = 157/54 # alpha wheel to shell gear ratio
  I_ball = 7e-4
  KE_baseball = 1/2*0.145*40**2
  Fmax_bearing = 100 # Rated axial load for bearings
  
  def prnt(self, *s):
  # Conditional print wrapper
    if self.disp:
      print(*s)
  
  def run_all(self):
    self.run_general()
    self.run_omega()
    self.run_alpha()
    self.run_shaft_stress()
  
  def run_general(self):
    ## Run some general calculations
    # self.prnt = get_prnt(disp)
    self.m = pi*self.R_G**2 * self.t_G * self.rho_G
    self.prnt(f"Mass m = {self.m:.6f} kg")
    self.I_G = 1/2*self.m * self.R_G**2
    self.prnt(f"Flywheel inertia I_G = {self.I_G:.6e} kg*m^2")
  
  def run_omega(self):
    ## Run calculations involving omega motor
    # Note, this is only during startup. Maybe consider inertial forces too
    self.F_N = self.m * self.g
    self.prnt(f"Bearing load due to weight: {self.F_N:.6f} N")
    self.T_f = self.F_N * self.mu * self.R_s
    # self.prnt(f"Frictional torque = {self.T_f:.6f}")
    self.omega = self.omega_NL_G * (1 - self.T_f / self.T_stall_G)
    self.prnt(f"Equilibrium flywheel speed omega = {self.omega:.6f} rad/s"
      f", which is {100*self.omega/self.omega_NL_G:.3f}% of the no-load speed")
    # Very rough constant acceleration assumption
    t_spinup = self.omega / ((self.T_stall_G-self.T_f)/self.I_G)
    self.prnt(f"Rough spinup time estimate: {t_spinup:0.2f} s")
    self.H = self.I_G * self.omega
    # self.prnt(f"Angular momentum H = {self.H:.6f}")
    self.KE = 1/2*self.I_G*self.omega**2
    self.prnt(f"The KE is {self.KE:.6f} J, which is "
      f"{100*self.KE/self.KE_baseball:.2f}% of that of a 90mph fastball")
  
  def run_alpha(self):
    ## Run calculations involving alpha motor
    self.alphad_NL = self.omega_NL_alpha / self.N
    self.alphad = self.alphad_NL/(1+self.alphad_NL*self.H/(self.N*
      self.T_stall_alpha))
    self.prnt(f"Alpha-dot = {self.alphad:.6f} rad/s"
      f", which is {100*self.alphad/self.alphad_NL:.3f}% of the no-load speed")
    self.Hd = self.H * self.alphad
    self.prnt(f"H-dot = M = {self.Hd:.6f} N-m")
    self.wdot = self.Hd / self.I_ball
    self.prnt(f"Approx. omega_ball-dot = {self.wdot:.6f}")
  
  def run_shaft_stress(self):
    ## Run calculations for shaft bending stress
    self.F_bearings = self.Hd / self.d_bearings
    self.prnt(f"Bearing load due to H-dot: {self.F_bearings:.6f} N")
    # Second moment of area
    self.I2_shaft = pi/4 * self.R_s**4
    self.sigma_B = self.Hd * self.R_s / self.I2_shaft
    self.eta_B = self.S_y / self.sigma_B
    self.prnt(f"Bending safety factor eta_B = {self.eta_B}")
    self.eta_bF = self.Fmax_bearing / self.F_bearings
    self.prnt(f"Bearings safety factor eta_bF = {self.eta_bF}")
  

if __name__ == "__main__":
  ball = Ball()
  ball.R_G = 19e-3
  ball.t_G = 16e-3
  ball.run_general()
  ball.run_omega()
  ball.run_alpha()
  ball.run_shaft_stress()
  
  
  if False:

    ball_list = []
    # v_mu = np.linspace(0.001, 0.1, 100)
    v_x = np.linspace(10e-3, 35e-3, 100)
    for xi in v_x:
      ball = Ball()
      ball.disp = False
      # ball.mu = xi
      ball.R_G = xi
      ball.run_all()
      ball_list.append(ball)

    labels = ["KE", "wdot", "eta_B", "eta_bF"] # , "Hd"
    outputs = [[getattr(ball, label) for label in labels] 
      for ball in ball_list]

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.grid()
    lines = ax.plot(v_x, outputs)
    fig.legend(lines, labels)
    fig.show()
    input("DONE")