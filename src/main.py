"""
Launching point
"""
import numpy as np
from Simulation import Simulation
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
  #alphaddf = lambda t: 0.01*np.ones_like(t)
  alphaddf = lambda t: 20*np.cos(2*np.pi*t)
  sim = Simulation(alphaddf, t_max=2)
  sim.run("tmp.dill")
  return sim

def load_test():
  sim = Simulation.load("tmp.dill")
  return sim

def FF_test():
  p_desf = np.array([1,1])
  sim = Simulation("FF", p_des=p_desf, t_max=1)
  sim.run("FF_test.dill")
  return sim

if __name__ == "__main__":
  # Derive the equations of motion and save them to file
  derive = False
  sol_fname = "sol.dill"
  
  # Start timing the execution
  times = tic()
  
  if derive:
    print(" -- Deriving the EOM -- ")
    toc(times)
    M, F, ax, ay = setup()
    toc(times, "EOM derivation")
  
  #sim = simple_test()
  #sim = load_test()
  sim = FF_test()
  toc(times, "Simulation")
  
  sim.plot()
  
  print(" -- DONE -- ")
  toc(times, "Total execution", total=True)
  