"""
Make an animation from a solved simulation.

Requires PyQt5
"""

import numpy as np
from Simulation import Simulation
from visualization import VizScene
import transforms as tr
import time

def animate3d(sim):
  """ Plot a 3d animation of the rolling ball
  """
  
  Rs = sim.ball.Rs # Constant vertical offset

  # Evaluate the solution and the input function at discrete time steps
  fps = 30
  slow = 4 # Factor for speeding up or down
  t_eval = np.linspace(0, sim.t_max, fps*slow)
  t, x, alphadd = sim.xeval(t=t_eval)
  
  # Set up the VizScene (from John's robotics visualization code)
  viz = VizScene()
  time_to_run = sim.t_max
  
  ball_frm = tr.se3()
  gyro_frm = tr.se3()
  viz.add_frame(ball_frm, label="ball")
  viz.add_frame(gyro_frm, label="gyro")

  for i in range(int(fps*slow * time_to_run)):
    q = x[0:4,i]
    R_ball = tr.q2R(q)
    R_gyro = R_ball @ tr.rotz(x[9,i]) # Is this backwards?
    p_com = np.array([x[7,i], x[8,i], Rs])
    ball_frm = tr.se3(R=R_ball, p=p_com)
    gyro_frm = tr.se3(R=R_gyro, p=p_com)
    viz.update(As=[ball_frm, gyro_frm])
    time.sleep(1.0/fps)
  
  viz.hold()

if __name__ == "__main__":
  sim = Simulation.load("tmp.dill")
  animate3d(sim)
