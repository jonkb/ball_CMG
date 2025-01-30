"""
Simulation class to hold the parameters of a simulation

TODO
DONE 1. Rework this so it can be stepped through interactively
OK 2. Get a full-state control scheme working decently
  Might need to be MPC. If I can get it running real-time, that'd be awesome.
    If not, then maybe pre-optimize a number of trajectories, then pick from
    that set, augmenting with PID / FF.
  Maybe make a surrogate for the dynamics
3. Work on making decent state estimation (observer)
4. Make it so that the controller is using the simulated measurements instead
  of the full state. (i.e. observer-based control)
5. Add noise & disturbances.
"""

import numpy as np
import scipy.integrate as spi
# from scipy.misc import derivative as spd
import dill
from Plotter import PlotterX, AnimatorXR, plot_anim, plot_tym
from diff import rk4
from Observer import Luenberger, ObsML, ObsSim

class Simulation:
  status = "unsolved"
  sol = None
  # Sample times for dynamics, controls, and plotting
  #   NOTE: See what dt_dyn needs to be to converge to something realistic
  # dt_dyn must be the smallest time interval.
  #   For best performance, the other 'dt's should be integer multiples 
  #   of dt_dyn.
  dt_dyn = 0.0005
  dt_plt = 0.01
  # x0 uncertainty
  x0unc = 1e-6
  
  def __init__(self, cnt, ball=None, t_max=1, x0=None):
    """
    cnt: Controller object that calculates control inputs
    t_max: Simulation end time
    x0: Initial conditions (see dynamics.eom for state vector description)
    """
    
    self.cnt = cnt
    self.ball = cnt.ball if ball is None else ball
    self.t_max = t_max
    
    if x0 is None:
      x0 = np.zeros(11)
      x0[0] = 1 # Real part of quaternion starts at 1
    self.x0 = x0
    
    # Initialize observer
    # Add uncertainty to observer x0
    x0u = x0 + (np.random.rand(x0.size)*2-1) * self.x0unc
    # self.obs = Luenberger(cnt.ball, x0u)
    # self.obs = ObsML(cnt.ball, x0u)
    self.obs = ObsSim(cnt.ball, x0u)
  
  def __str__(self):
    s = "Simulation Object\n"
    s += f"\tstatus: {self.status}\n"
    # Cut off final \n and indent the whole block
    scnt = str(self.cnt)[0:-1].replace("\n", "\n\t")
    s += f"\tcontroller: {scnt}\n"
    s += f"\tt_max: {self.t_max}\n"
    s += f"\tx0: {self.x0}\n"
    return s
  
  def run_dt(self, plotting=True, fname="sim.dill"):
    """ Run in timestepping mode
    I.e. timestep instead of using solve_ivp
    
    This lets us simulate control schemes that update at a given control rate
    """
    
    ## Constants TODO: Make some of these constructor parameters
    # dt_dyn must be the smallest time interval.
    assert self.dt_dyn <= min(self.cnt.dt_cnt, self.dt_plt), ("dt_dyn "
      "should be the smallest dt")
    # Set status
    self.status = "running"
    print("Starting simulation")
    
    # Full time vector
    v_t = np.arange(0, self.t_max, self.dt_dyn)
    
    # Variables that are updated during the simulation
    if plotting:
      plotter = PlotterX(x0=self.x0)
      animator = AnimatorXR(ref=self.cnt.ref, ref_type=self.cnt.ref_type)
    t_next_obs = v_t[0]
    t_next_cnt = v_t[0]
    t_next_plt = v_t[0]
    # u = 0.001 #0.0 # Current control input NOTE: May be vector later
    u = 0.0
    x = np.copy(self.x0)
    # Store everything
    v_x = np.empty((v_t.size, self.ball.n_x))
    v_u = np.empty((v_t.size, self.ball.n_u))
    v_ym = np.empty((v_t.size, self.ball.n_ym))
    v_xhat = np.empty((v_t.size, self.ball.n_x))
    
    ## Simulation loop
    for i,t in enumerate(v_t):
      # Update dynamics
      x = rk4(self.ball.eom, x, self.dt_dyn, (u,))
      # Record simulated measurement data
      ym = self.ball.measure(x, u)
      
      if t >= t_next_obs:
        # Update observer
        x_hat = self.obs.update(ym, u)
        # Don't update observer again until dt_obs time has passed
        t_next_obs += self.obs.dt_obs
        
        # print(114, x_hat)
      
      # Store state, input, and measurement at every timestep
      v_x[i] = x
      v_u[i] = u
      v_ym[i] = ym
      v_xhat[i] = x_hat
      
      if t >= t_next_cnt:
        # Update control
        u = self.cnt.update(t, x_hat)
        # TEMP: CHEAT to work with MPC p-ref
        # u = self.cnt.update(t, x)#x_hat)
        # Don't update control again until dt_cnt time has passed
        t_next_cnt += self.cnt.dt_cnt
      
      # print(126, x[0:4], x_hat[0:4], np.sum(np.square(x[0:4] - x_hat[0:4])))
      # print(127, x[4:7], x_hat[4:7], np.sum(np.square(x[4:7] - x_hat[4:7])))
      # print(128, x[9:11], x_hat[9:11], np.sum(np.square(x[9:11] - x_hat[9:11])))
      
      if plotting and (t >= t_next_plt):
        plotter.update_interactive(v_t[0:i+1], v_x[0:i+1], v_u[0:i+1], 
          v_ym=v_ym[0:i+1], v_xhat=v_xhat[0:i+1])
        animator.update(v_t[0:i+1], v_x[0:i+1])
        # Don't update plots again until dt_plt time has passed
        t_next_plt += self.dt_plt
    
    # Save result
    self.sol = SolV(v_t, v_x, v_u, v_ym, v_xhat)
    self.status = "solved"
    if fname is not None:
      self.save(fname)
  
  def run(self, t_eval=None, fname="sim.dill"):
    """ Run the simulation and store the result
    fname (str or None): If str, then save the Simulation object to the given
      filename upon completion.
    """
    
    xdot = lambda t,x: self.ball.eom(x, self.cnt.update(t, x))
    
    # Solving the IVP
    self.status = "running"
    print("Solving IVP")
    """ Notes about the tolerance. For a test with a=10, t_max=1.5:
    rtol=1e-7, atol=1e-10 was indistinguishable from rtol=1e-5, atol=1e-8. 
    rtol=1e-4, atol=1e-7 gave a similar first loop, but a different second loop.
    rtol=1e-5, atol=1e-7 seemed good.
    """
    if t_eval is None:
      sol = spi.solve_ivp(xdot, [0,self.t_max], self.x0, dense_output=True, 
        rtol=1e-5, atol=1e-7) # TODO: tol parameters
    else:
      sol = spi.solve_ivp(xdot, [0,self.t_max], self.x0, t_eval=t_eval,
        dense_output=False, rtol=1e-5, atol=1e-7)
    self.status = "solved"
    
    # Save result
    self.sol = sol
    if fname is not None:
      self.save(fname)

    return sol
  
  def save(self, fname="sim.dill"):
    # Save to file
    
    # with open(fname,"wb") as file:
      # dill.dump(self, file)
    # print(f"Simulation saved to {fname}")
    
    ssim = SerializableSim(self)
    with open(fname,"wb") as file:
      dill.dump(ssim, file)
    print(f"Simulation saved to {fname}")
  
  def xeval(self, t=None):
    """ Evaluate the solved simulation at discrete time steps, assuming that
      self.sol is a scipy.integrate.solve_ivp solution. Otherwise, pull out 
      the relevant vectors.
    
    Returns
    -------
      t - time vector
      x - state at each time step
      u - input at each time step
      ym - measured output at each time step
    Each of these is an array of the same length
    """
    
    # Error checking
    if self.status != "solved":
      print("This simulation still needs to be run")
      return
    assert self.sol is not None, "Error, self.sol undefined"
    
    # Check if self.sol is a SolV object
    if isinstance(self.sol, SolV):
      t = self.sol.v_t
      x = self.sol.v_x # TODO: Make these agree about x.T
      u = self.sol.v_u
      ym = self.sol.v_ym
    else:
      if t is None:
        # Make time vector
        t = np.linspace(0, self.t_max, 200)
        # Evaluate solution at desired time values
        x = self.sol.sol(t).T
      
      # Check bounds on solution
      tmin = np.min(self.sol.t)
      tmax = np.max(self.sol.t)
      if tmin > 0 or tmax < np.max(t):
        print("Warning: solution object was not solved over whole t vector.")
        print(f"\tsol.t = [{tmin}, {tmax}]")
        print(f"\tt = [{min(t)}, {max(t)}]")
      
      # Evaluate solution at desired time values
      if np.all(self.sol.t == t):
        x = self.sol.y.T
      else:
        x = self.sol.sol(t).T
      
      ## Evaluate alphadd @ t
      #alphadd = [self.cnt.update(ti, xi) for ti, xi in zip(t, x)]
      
      # Differentiate alphad to estimate alphadd
      u = np.gradient(x[:,10], t)
      
      # TODO
      ym = None
    
    return t, x, u, ym
  
  def copy(self, unsolve=False):
    """
    Return a copy of this Simulation object
    This only keeps the things which are serialized in SerializableSim
    """
    ssim1 = SerializableSim(self)
    strsim = dill.dumps(ssim1)
    ssim2 = dill.loads(strsim)
    sim2 = ssim2.to_sim()
    if unsolve:
      # Remove the solution and reset the simulation
      sim2.status = "unsolved"
      sim2.sol = None
    return sim2
  
  @staticmethod
  def load(fname="sim.dill"):
    with open(fname, "rb") as file:
      ssim = dill.load(file)
    sim = ssim.to_sim()
    print(f"Simulation loaded from {fname}. Status={sim.status}")
    return sim
  
  def plot(self, t_eval=None, show=True):
    """ Plots the solution results
    show (bool): Whether to show the plots now or simply return the figs
    """
    
    # Evaluate the solution and the input function at discrete time steps
    t, x, u, ym = self.xeval(t=t_eval)
    
    if ym is None:
      ym = [self.ball.measure(xi, ui) for (xi, ui) 
        in zip(x, u)] # NOTE: x.T ?

    # Plot state vector as a function of time
    plotterx = PlotterX(interactive=False)
    plotterx.plot_all(t, x, u)
    fig1 = plotterx.fig
    axs1 = plotterx.axs
    
    # Plot measured values as a function of time
    fig2, ax2 = plot_tym(t, ym)
    
    if show:
      fig1.show()
      fig2.show()
    
    # Animation
    fig3, ax3 = plot_anim(t, x)#, self.p_des, self.a_des)
    
    return fig1, fig2, fig3

class SolV:
  def __init__(self, v_t, v_x, v_u, v_ym, v_xhat):
    """ Pack the solution vectors in a single solution object
    
    The "V" is for vector, because it stores a bunch of vectors instead of
      a sol object like spi.integrate
    """
    
    # print(129, v_t, 131.1, v_x, 131.2, v_u, 131.3, v_ym)
    self.v_t = v_t
    self.v_x = v_x
    self.v_u = v_u
    self.v_ym = v_ym
    self.v_xhat = v_xhat
  

class SerializableSim:
  def __init__(self, sim):
    """
    Convert a Simulation object (sim) to a serializable form.
    This allows it to be saved to file.
    The lambdified sympy functions have a hard time being serialized, so leave 
      them out.
    """
    
    self.cnt = sim.cnt
    self.t_max = sim.t_max
    self.x0 = sim.x0
    # The following are not in the Simulation constructor
    self.status = sim.status
    self.sol = dill.dumps(sim.sol)
    # Make cnt.ball serializable
    self.cnt = self.cnt.serializable()
    # self.cnt.ball = sim.cnt.ball.serializable()
    # Make sim.ball serializable
    self.ball = sim.ball.serializable()
  
  def to_sim(self):
    """
    Convert this to a normal Simulation object
    """
    
    # Convert the SerializableBall in controller back to a CMGBall
    self.cnt = self.cnt.from_serializable()
    # self.cnt.ball = self.cnt.ball.to_ball()
    self.ball = self.ball.to_ball()
    
    sim = Simulation(self.cnt, t_max=self.t_max, x0=self.x0)
    sim.status = self.status
    sim.sol = dill.loads(self.sol)
    
    return sim
  

if __name__ == "__main__":
  # TEST Serialization
  from CMGBall import CMGBall
  from Controller import MPC
  ball = CMGBall()
  p_ref = lambda t: (1-np.exp(-4*t))*np.array([2,1])
  MPCprms = {
    "ftol_opt": 1e-1,
    "maxit_opt": 1
  }
  cnt = MPC(ball, p_ref, ref_type="p", dt_cnt=0.1, 
    options=MPCprms)
  sim = Simulation(cnt, t_max=0.4)
  fname ="testinggggg.dill"
  # sim.save(fname)
  sim.dt_dyn=.01
  print(355)
  sim.run_dt(fname=None)
  print(359)
  # print(335, dir(sim.cnt.N_window))
  # print(335, dir(sim.cnt.ftol_opt))
  # print(335, dir(sim.cnt.ref))
  # print(335, dir(sim.cnt.u_window))
  print(447, SerializableSim(sim))
  print(448, len(SerializableSim(sim).sol))
  
  # with open("tmppp.d","wb") as file:
    # dill.dump(SerializableSim(sim).sol, file)
  
  sim.save(fname)
  # print(449, sim.cnt)
  # print(450, sim.cnt.ball)
  siml = Simulation.load(fname)
  print(453, siml)