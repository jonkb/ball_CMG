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
  #alphaddf = lambda t: 10*np.exp(t)
  alphaddf = lambda t: 10*np.ones_like(t)
  #alphaddf = lambda t: 24*np.cos(2*np.pi*t)
  sim = Simulation(alphaddf, t_max=1.5)
  sim.run("tmp.dill")
  return sim

def load_test():
  sim = Simulation.load("tmp.dill")
  return sim

def FF_test(tag):
  p_desf = np.array([1,1])
  sim = Simulation("FF", p_des=p_desf, t_max=1)
  sim.run(f"FF_test{tag}.dill")
  return sim

def MPC_test(tag):
  MPCprms = {
    "t_window": .15,
    "N_vpoly": 3,
    "N_sobol": 32, # Should be a power of 2
    "N_eval": 4,
    "ratemax": 150 #Hz
  }
  #p_desf = np.array([1,3])
  p_desf = lambda t: (1-np.exp(-3*t))*np.array([1,2])
  sim = Simulation("MPC", p_des=p_desf, t_max=3)
  fname = f"MPC_{tag}.dill"
  sim.save(fname)
  sim.run(fname)
  return sim

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
  
  #sim = simple_test()
  #sim = load_test()
  #sim = FF_test()
  #sim = MPC_test("CAEDM1")
  sim = MPC_test("CAEDM7")
  #sim = FF_test("CAEDMFF1")
  #sim = Simulation.load("MPC_test.dill")
  toc(times, "Simulation")
  
  #sim.plot()
  
  print(" -- DONE -- ")
  toc(times, "Total execution", total=True)
  
