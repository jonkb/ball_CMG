"""
Launching point
"""

import numpy as np
from Simulation import Simulation
from CMGBall import CMGBall
from util import tic, toc

def setup():
  """ Derive equations of motion
  This should only need to be run once because it saves them to file.
  """
  
  print(" -- Deriving the EOM -- ")
  M, F = dyn.derive_EOM()
  ax, ay = find_axay(M, F)
  return M, F, ax, ay

def simple_test():
  #alphaddf = lambda t: 10*np.exp(t)
  alphaddf = lambda t: 10*np.ones_like(t)
  #alphaddf = lambda t: 24*np.cos(2*np.pi*t)
  sim = Simulation(alphaddf, t_max=1.5)
  sim.run(fname="tmp.dill")
  return sim

def load_test():
  sim = Simulation.load("tmp.dill")
  return sim

def FF_test(tag):
  p_desf = np.array([1,1])
  sim = Simulation("FF", p_des=p_desf, t_max=1)
  sim.run(fname=f"FF_test{tag}.dill")
  return sim

def MPC_test(tag):
  MPCprms = {
    "t_window": .25,
    "N_vpoly": 3,
    "N_sobol": 32, # Should be a power of 2
    "N_eval": 4,
    "ratemax": 150, #Hz
    "vweight": 0#.005
  }
  #p_desf = np.array([1,3])
  p_desf = lambda t: (1-np.exp(-3*t))*np.array([1,2])
  sim = Simulation("MPC", p_des=p_desf, t_max=3, MPCprms=MPCprms)
  fname = f"MPC_{tag}.dill"
  sim.save(fname)
  sim.run(fname=fname)
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

if __name__ == "__main__":
  # Derive the equations of motion and save them to file
  derive = False
  
  # Start timing the execution
  times = tic()
  
  if derive:
    print(" -- Deriving the EOM -- ")
    toc(times)
    M, F, ax, ay = setup()
    toc(times, "EOM derivation")
  
  # Load an existing sim from file
  #fname = "MPC_testCAEDM1.dill" # Hits it exactly
  #fname = "MPC_test4.dill"
  #sim = Simulation.load(fname)
  #print(f"Loaded simulation from file: {fname}")
  
  # Run a new simulation
  sim = simple_test()
  #sim = load_test()
  #sim = FF_test()
  #sim = MPC_test("CAEDM14")
  #sim = FF_test("CAEDMFF1")
  #toc(times, "Simulation")
  
  print(sim)
  sim.plot()
  
  # Test variants of sim
  #prmvar(sim, "prmvartest")
  
  print(" -- DONE -- ")
  toc(times, "Total execution", total=True)
  
