"""
Launching point
"""

import numpy as np
from Simulation import Simulation
from CMGBall import CMGBall
from Controller import PreSet, FF, MPC
from util import tic, toc

import dill

def setup():
  """ Derive equations of motion
  This should only need to be run once because it saves them to file.
  """
  
  import dynamics as dyn
  print(" -- Deriving the EOM -- ")
  M, F = dyn.derive_EOM()
  EOM = dyn.solve_EOM(M, F)
  ax, ay = dyn.find_axay()
  return M, F, EOM, ax, ay

def dt_test():
  alphadd = 0.2
  ball = CMGBall(ra=np.array([0.02, 0, 0]))
  cnt = PreSet(ball, alphadd)
  sim = Simulation(cnt, t_max=4.0)
  
  sim.run_dt()#fname="tmpp.dill")
  return sim

def simple_test():
  """ Simple preset control input test
  """
  
  alphadd = 0.2
  #alphaddf = lambda t: 10*np.exp(t)
  # alphaddf = lambda t: alphadd*np.ones_like(t)
  #alphaddf = lambda t: 24*np.cos(2*np.pi*t)
  ball = CMGBall(ra=np.array([0.02, 0, 0]))
  cnt = PreSet(ball, alphadd)

  sim = Simulation(cnt, t_max=0.75)
  
  sim.run(fname="tmp.dill")
  return sim

def FF_test(tag):
  # v_des = np.array([.2,.1])
  # ref = lambda t: v_des
  ref = lambda t: (1-np.exp(-4*t))*np.array([2.0,0.5])
  ball = CMGBall(ra=np.array([0.02, 0, 0]))
  
  cnt = FF(ball, ref, ref_type="p")
  sim = Simulation(cnt, t_max=5.0)
  fname=f"FF_test{tag}.dill"
  # sim.run(fname)
  sim.run_dt(plotting=True, fname=fname)
  return sim

def MPC_test(tag, ball=None):
  
  p_ref = lambda t: np.array([-1.5,2.0])#*(1-np.exp(-4*t))
  if ball is None:
    ball = CMGBall()
  
  MPCprms = {
    "N_window": 7,
    "ftol_opt": 0.01,
    "maxit_opt": 4,
    "v0_penalty": 0.0,
    "w0_penalty": 0.0001
  }
  cnt = MPC(ball, p_ref, ref_type="p", dt_cnt=0.30, 
    options=MPCprms)
  sim = Simulation(cnt, t_max=4.0)
  fname = f"MPC_{tag}.dill"
  sim.run_dt(plotting=True, fname=fname)
  return sim

def prmvar(template_sim, tag):
  """ Parameter variation sensitivity
  Re-run the given simulation with the same input but with a slight
    variation to the robot model parameters.
  """
  
  # Pull out the input (alphadd) from template_sim
  t, x, alphadd = template_sim.xeval()
  # Turn alphadd into a function
  alphaddf = lambda ti: np.interp(ti, t, alphadd)
  
  # List of variant robots to simulate
  robots = [
    CMGBall(m=.99),
    CMGBall(m=1.01),
    CMGBall(Rs=0.0495),
    CMGBall(Rs=0.0505)
  ]

  # function to run a variation of template_sim
  def vary_sim(robot, lbl):
    varsim = template_sim.copy(unsolve=True)
    # Set up the control for this sim to use the saved input alphadd
    varsim.control_mode = None # Otherwise, it may try to redo MPC
    varsim.alphaddf = alphaddf
    # Change out the robot parameters
    varsim.ball = robot
    varsim.Mf, varsim.Ff = varsim.load_MfFf()
    # Run simulation
    #varsim.run(t_eval=t, fname=f"{tag}_var{lbl}")
    varsim.run(t_eval=t, fname=None)
    return varsim
  
  # Simulate for several different robots
  xs = [x]
  for i, robot in enumerate(robots):
    varsim = vary_sim(robot, i)
    _, xi, _ = varsim.xeval(t)
    xs.append(xi)
  
  # Plot everything on top of each other
  Simulation.plot_anims(t, xs, template_sim.p_des, template_sim.a_des)

def load_and_plot(fname="tmp.dill"):
  """ Load a sim from file & plot it
  """
  #fname = "MPC_testCAEDM1.dill" # Hits it exactly
  print(f"Loading simulation from file: {fname}")
  sim = Simulation.load(fname)
  print(sim)
  sim.plot()

if __name__ == "__main__":
  # Derive the equations of motion and save them to file
  derive = False
  
  # Start timing the execution
  times = tic()
  
  if derive:
    print(" -- Deriving the EOM -- ")
    toc(times)
    M, F, EOM, ax, ay = setup()
    toc(times, "EOM derivation")
    
  # Load a pre-generated ball object
  # fname = "ball_20230518T223742.dill"
  # with open(fname,"rb") as file:
    # sball = dill.load(file)
    # ball = sball.to_ball()
  # toc(times, "Loading ball")
  
  # Load an existing sim from file
  # load_and_plot("MPC_0517_1.dill")
  
  # Run a new simulation
  # sim = simple_test()
  sim = FF_test("0522_0")
  # sim = dt_test()
  # sim = MPC_test("0519_1")
  
  toc(times, "Simulation")
  
  input("PRESS ANY KEY TO CONTINUE")
  
  # Test variants of sim
  #prmvar(sim, "prmvartest")
  
  print(" -- DONE -- ")
  toc(times, "Total execution", total=True)
  
