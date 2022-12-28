"""
Simulation class to hold the parameters of a simulation

TODO: MPC control
"""

import numpy as np
import scipy.integrate as spi
# from scipy.misc import derivative as spd
import scipy.optimize as spo
from matplotlib import pyplot as plt
from matplotlib import animation
import dill

import dynamics as dyn
from CMGBall import CMGBall

class Simulation:
  status = "unsolved"
  alphaddf = None
  control_mode = None
  p_des = None
  a_des = None
  sol = None
  # Parameters for MPC
  MPCprms = {
    "t_window": .25,
    "N_vpoly": 3,
    "N_sobol": 32, # Should be a power of 2
    "N_eval": 5,
    "ratemax": 200 #Hz
  }
  t_MPChist = [] # List of each time MPC optimization is run
  v_MPChist = [] # List of v vectors from each time MPC is run
  
  def __init__(self, alphadd, p_des=None, a_des=None, ball=CMGBall(), t_max=1, 
      x0=None, Mf=None, Ff=None, axf=None, ayf=None, MPCprms={}):
    """
    alphadd: Gyro acceleration alpha-double-dot
      Either pass a scalar, a function of time, or one of the following:
        "FF": Feedforward control
      If one of those strings is passed, then p_des, v_des, or a_des must be 
        passed as well.
      p_des: Desired position. Either a function of time or a 2x0 point.
      a_des: Desired acceleration. Either a function of time or a 2x0 vector.
      ball: A CMGBall object
      t_max: Simulation end time
      x0: Initial conditions (see dyn.eom for state vector description)
      Mf, Ff, axf, ayf: Lambdified dynamics functions. If not provided, they
        will be loaded from file.
    """
    
    ## Input handling
    if callable(alphadd):
      self.alphaddf = alphadd
    elif isinstance(alphadd, float):
      self.alphaddf = lambda t: alphadd
    elif alphadd == "FF":
      self.control_mode = "FF"
    elif alphadd == "MPC":
      self.control_mode = "MPC"
    else:
      assert False, "Unknown alphadd"
    
    if self.control_mode is not None:
      if p_des is not None:
        if callable(p_des):
          self.p_des = p_des
        else:
          self.p_des = lambda t: p_des
      elif a_des is not None:
        if callable(a_des):
          self.a_des = a_des
        else:
          self.a_des = lambda t: a_des
      else:
        assert False, ("Must provide a desired path or acceleration when "
          "using a control mode")
    
    self.ball = ball
    self.t_max = t_max
    
    if x0 is None:
      x0 = np.zeros(11)
      x0[0] = 1 # Real part of quaternion starts at 1
    self.x0 = x0
    
    if Mf is None or Ff is None:
      self.Mf, self.Ff = self.load_MfFf()
    if axf is None or ayf is None:
      self.axf, self.ayf = self.load_axfayf()
    
    for key, val in MPCprms.items():
      self.MPCprms[key] = val
  
  def __str__(self):
    s = "Simulation Object\n"
    s += f"\tstatus: {self.status}\n"
    s += f"\tcontrol_mode: {self.control_mode}\n"
    s += f"\tt_max: {self.t_max}\n"
    s += f"\tx0: {self.x0}\n"
    # Cut off final \n and indent the whole block
    sball = str(self.ball)[0:-1].replace("\n", "\n\t")
    s += f"\tball: {sball}\n"
    s += f"\tMPCprms: {self.MPCprms}\n"
    return s
  
  def load_MfFf(self):
    M, F = dyn.load_MF()
    return dyn.lambdify_MF(M, F, ball=self.ball)
  
  def load_axfayf(self):
    ax, ay = dyn.load_axay()
    return dyn.lambdify_axay(ax, ay, ball=self.ball)
  
  def alphadd_FF(self, t, x):
    """ Return the acceleration calculated by feedforward control
    """
    
    # Get a_des
    if self.p_des is not None:
      kp = 10 # TODO: Make this a parameter
      to_goal = self.p_des(t) - np.array([x[7], x[8]])
      a_des = kp*to_goal
    elif self.a_des is not None:
      a_des = self.a_des(t)
    else:
      assert False, "Must supply desired path to use FF control"
    
    def err(alphadd):
      # See how close this alphadd is to giving a_des
      xd = dyn.eom(self.Mf, self.Ff, x, alphadd, ball=self.ball)
      #s_axay = (eta, ex, ey, ez, omega_x, omega_y, omega_z, etad, exd, eyd, ezd, omega_xd, omega_yd, omega_zd)
      s_axay = (*x[0:7], *xd[0:7])
      a_vec = np.array([self.axf(*s_axay), self.ayf(*s_axay)])
      # L2 error
      return np.sum(np.square(a_des - a_vec))
    
    alphadd_bounds = (-self.ball.alphadd_max, self.ball.alphadd_max)
    res = spo.minimize_scalar(err, method="bounded", bounds=alphadd_bounds, 
      options={"maxiter":10}) # TODO: maxiter parameter
    #print(f"Error: {res.fun}, nit: {res.nit}, res.x: {res.x}")
    print(f"t: {t}, error: {res.fun}, nfev: {res.nfev}, res.x: {res.x}")
    # TODO: Decide adaptively to do MPC instead if error is too large
    return res.x
  
  def alphadd_v(self, t, v):
    """ Parameterized acceleration function to be optimized in MPC
    v is formatted as a polynomial
    f = v[0] + v[1]*t + v[2]*t**2 + ... + v[n]*t**n
    """
    
    # Equivalent, but possibly slower method:
    # for n,vi in enumerate(v):
      # a += vi*t**n
    # return a
    
    t__n = np.power(t*np.ones_like(v), np.arange(len(v)))
    
    return np.sum(np.multiply(v,t__n))
  
  def alphadd_MPC(self, t, x):
    """ Return the acceleration calculated by MPC
    Procedure:
    1. Check if we've already optimized the input for a time point very close 
      to this one.
    2. If not, simulate ahead for some amount of time
    3. Score the resultant path
    4. Optimize 2 & 3
    """
    
    # Unpack MPCprms
    t_window = self.MPCprms["t_window"]
    N_vpoly = self.MPCprms["N_vpoly"]
    N_sobol = self.MPCprms["N_sobol"]
    N_eval = self.MPCprms["N_eval"]
    ratemax = self.MPCprms["ratemax"]
    vweight = self.MPCprms["vweight"] if "vweight" in self.MPCprms else 0.1
    
    if len(self.t_MPChist) > 0:
      t_diff = np.abs(np.array(self.t_MPChist) - t)
      diff_min = np.min(t_diff)
      # Are we close enough to one that's already been optimized?
      if diff_min < 1.0/ratemax:
        # Then use solution from that MPC run
        # Index of closest solution to current time
        i_closest = np.where(t_diff == diff_min)[0][0]
        t_closest = self.t_MPChist[i_closest]
        v_closest = self.v_MPChist[i_closest]
        alphadd = self.alphadd_v(t-t_closest, v_closest)
        return alphadd
      # Otherwise, re-optimize v for alphadd_v(t,v) with MPC
    
    if self.p_des is None:
      assert False, "Must supply desired path to use MPC control"
    
    def cost(v):
      # Simulate using the given input vector v and score the path
      xdot = lambda t,x: dyn.eom(self.Mf, self.Ff, x, self.alphadd_v, 
        aldd_args=(t,v), ball=self.ball)
      # NOTE: this time is relative to the current time, t
      t_eval = np.linspace(0, t_window, N_eval)
      sol = spi.solve_ivp(xdot, [0,t_window], y0=x, method="RK23",
        t_eval=t_eval, dense_output=False, rtol=1e-4, atol=1e-7) # TODO: tol
      
      # Compare to desired path
      p_xy = [self.p_des(ti) for ti in t_eval]
      p_x = np.array([p[0] for p in p_xy])
      p_y = np.array([p[1] for p in p_xy])
      v_x = np.gradient(p_x, t_eval)
      v_y = np.gradient(p_y, t_eval)
      vsol_x = np.gradient(sol.y[7], t_eval)
      vsol_y = np.gradient(sol.y[8], t_eval)
      # (4xN_MPCeval)
      err = np.array([p_x-sol.y[7], p_y-sol.y[8], v_x-vsol_x, v_y-vsol_y])
      weighted_err = err*np.linspace(.1,1,N_eval)
      weighted_err[2:4,:] *= vweight
      return np.sum(np.square(weighted_err))
    
    # NOTE: Does it make sense to bound higher order terms the same way?
    v_bounds = (-self.ball.alphadd_max, self.ball.alphadd_max)
    bounds = [v_bounds]*N_vpoly
    # Optimize. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html#scipy.optimize.shgo
    # NOTE: Could switch from Sobol to LHC. Probably wouldn't change much
    res = spo.shgo(cost, bounds, n=N_sobol, sampling_method='sobol',
      minimizer_kwargs={"options":{"tol":.1}})
    
    v = res.x
    print(f"t: {t}, cost: {res.fun}, nfev: {res.nfev}, v_opt: {v}")
    
    # Store this result to be reused
    self.t_MPChist.append(t)
    self.v_MPChist.append(v)
    
    return self.alphadd_v(0, v)
  
  def run(self, fname="sim.dill"):
    """ Run the simulation and store the result
    fname (str or None): If str, then save the Simulation object to the given
      filename upon completion.
    """
    
    if self.control_mode == "FF":
      xdot = lambda t,x: dyn.eom(self.Mf, self.Ff, x, self.alphadd_FF, 
        aldd_args=(t, x), ball=self.ball)
    elif self.control_mode == "MPC":
      xdot = lambda t,x: dyn.eom(self.Mf, self.Ff, x, self.alphadd_MPC, 
        aldd_args=(t, x), ball=self.ball)
    else:
      xdot = lambda t,x: dyn.eom(self.Mf, self.Ff, x, self.alphaddf, 
        aldd_args=(t,), ball=self.ball)

    # Solving the IVP
    self.status = "running"
    print("Solving IVP")
    """ Notes about the tolerance. For a test with a=10, t_max=1.5:
    rtol=1e-7, atol=1e-10 was indistinguishable from rtol=1e-5, atol=1e-8. 
    rtol=1e-4, atol=1e-7 gave a similar first loop, but a different second loop.
    rtol=1e-5, atol=1e-7 seemed good.
    """
    sol = spi.solve_ivp(xdot, [0,self.t_max], self.x0, dense_output=True, 
      rtol=1e-5, atol=1e-7) # TODO: tol parameters
    self.status = "solved"
    
    # Save result
    self.sol = sol
    if fname is not None:
      self.save(fname)

    return sol
  
  def save(self, fname="sim.dill"):
    # Save to file
    ssim = SerializableSim(self)
    with open(fname,"wb") as file:
      dill.dump(ssim, file)
    print(f"Simulation saved to {fname}")
  
  @staticmethod
  def load(fname="sim.dill"):
    with open(fname, "rb") as file:
      ssim = dill.load(file)
    sim = ssim.to_sim()
    print(f"Simulation loaded from {fname}. Status={sim.status}")
    return sim
  
  def plot(self, show=True):
    """ Plots the solution results
    show (bool): Whether to show the plots now or simply return the figs
    """
    
    # Error checking
    if self.status != "solved":
      print("This simulation still needs to be run")
      return
    assert self.sol is not None, "Error, self.sol undefined"
    
    # Make time vector
    t = np.linspace(0, self.t_max, 200)
    # Check bounds on solution
    tmin = min(self.sol.t)
    tmax = max(self.sol.t)
    if tmin > 0 or tmax < self.t_max:
      print("Warning: solution object was not solved over whole t vector.")
      print(f"\tsol.t = [{tmin}, {tmax}]")
      print(f"\tt = [{min(t)}, {max(t)}]")
    
    # Evaluate solution & input at desired time values
    x = self.sol.sol(t)
    if self.alphaddf is None:
      # Differentiate alphad
      alphadd = np.gradient(x[10,:], t)
    else:
      alphadd = self.alphaddf(t)

    ## Plot
    fig1, axs = plt.subplots(3,2, sharex=True)
    # 0,0 - Input alpha acceleration
    axs[0,0].plot(t, alphadd)
    axs[0,0].set_title(r"Input gyro acceleration $\ddot{\alpha}$")
    # 0,1 
    axs[0,1].plot(t, x[9,:], label=r"$\alpha$")
    axs[0,1].plot(t, x[10,:], label=r"$\dot{\alpha}$")
    axs[0,1].legend()
    axs[0,1].set_title(r"Gyro angle & velocity $\alpha$, $\dot{\alpha}$")
    # 1,0
    axs[1,0].plot(t, x[0,:], label=r"$\eta$")
    axs[1,0].plot(t, x[1,:], label=r"$\varepsilon_x$")
    axs[1,0].plot(t, x[2,:], label=r"$\varepsilon_y$")
    axs[1,0].plot(t, x[3,:], label=r"$\varepsilon_z$")
    qnorm = np.linalg.norm(x[0:4,:], axis=0)
    axs[1,0].plot(t, qnorm, label="|$q$|")
    axs[1,0].legend()
    axs[1,0].set_title("Orientation $q$")
    # 1,1
    axs[1,1].plot(t, x[4,:], label="$\omega_x$")
    axs[1,1].plot(t, x[5,:], label="$\omega_y$")
    axs[1,1].plot(t, x[6,:], label="$\omega_z$")
    wnorm = np.linalg.norm(x[4:7,:], axis=0)
    axs[1,1].plot(t, wnorm, label="|$\omega$|")
    axs[1,1].legend()
    axs[1,1].set_title("Angular Velocity $\omega$")
    # 2,0
    axs[2,0].plot(t, x[7,:])
    axs[2,0].set_xlabel("Time t")
    axs[2,0].set_title("X-Position $r_x$")
    # 2,1
    axs[2,1].plot(t, x[8,:])
    axs[2,1].set_xlabel("Time t")
    axs[2,1].set_title("Y-Position $ry$")
    
    # Infer x-bounds for animation
    xmin = np.min(x[7,:])
    xmax = np.max(x[7,:])
    xspan = xmax-xmin
    ymin = np.min(x[8,:])
    ymax = np.max(x[8,:])
    yspan = ymax-ymin
    margin = .1*max(xspan, yspan) # Add 10% margins
    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin
    
    # Build px, py vectors
    px = None
    py = None
    if self.p_des is not None:
      px = np.zeros(t.shape)
      py = np.zeros(t.shape)
      for i, ti in enumerate(t):
        pxi, pyi = self.p_des(ti)
        px[i] = pxi
        py[i] = pyi
    elif self.a_des is not None:
      px = np.zeros(t.shape)
      py = np.zeros(t.shape)
      v_xy = lambda ti: spi.quad_vec(a_desf, min(t), ti)[0]
      p_xy = lambda ti: spi.quad_vec(v_xy, min(t), ti)[0]
      for i, ti in enumerate(t):
        pxi, pyi = p_xy(ti)
        px[i] = pxi
        py[i] = pyi
    
    # Animate
    # https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
    fig2 = plt.figure()
    #ax = plt.axes(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_aspect("equal")
    ax.grid()
    if (px is not None) and (py is not None):
      #print(px, py)
      ph, = ax.plot(px, py, linestyle="-", color="g", marker=".")
    rh = ax.scatter([], [], s=5, color="b", marker=".")
    circle, = ax.plot([0], [0], marker="o", markerfacecolor="b")

    # initialization function: plot the background of each frame
    def init_back():
        circle.set_data([], [])
        rh.set_offsets(np.array((0,2)))
        return circle, rh

    # animation function.  This is called sequentially
    def animate(i):
        rx = x[7,i]
        ry = x[8,i]
        circle.set_data([rx], [ry])
        rh.set_offsets(x[7:9,0:i].T)
        return circle, rh

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig2, animate, init_func=init_back,
      frames=t.size, interval=20, repeat_delay=5000, blit=True)

    if show:
      fig1.show()
      fig2.show()
      input("PRESS ANY KEY TO QUIT")
    
    return fig1, fig2

class SerializableSim:
  def __init__(self, sim):
    """
    Convert a Simulation object (sim) to a serializable form.
    This allows it to be saved to file.
    The lambdified functions have a hard time being serialized, so leave them
      out.
    """
    
    self.alphaddf = sim.alphaddf
    self.p_des = sim.p_des
    self.a_des = sim.a_des
    self.ball = sim.ball
    self.t_max = sim.t_max
    self.x0 = sim.x0
    self.MPCprms = sim.MPCprms
    # The following are not in the Simulation constructor
    self.status = sim.status
    self.control_mode = sim.control_mode
    self.sol = sim.sol
    self.t_MPChist = sim.t_MPChist
    self.v_MPChist = sim.v_MPChist
  
  def to_sim(self, Mf=None, Ff=None, axf=None, ayf=None):
    """
    Convert this to a normal Simulation object
    """
    if "MPCprms" not in dir(self):
      # Older versions may not have this
      self.MPCprms = {}
    
    if self.alphaddf is None:
      self.alphaddf = self.control_mode
    
    sim = Simulation(self.alphaddf, p_des=self.p_des, a_des=self.a_des, 
      ball=self.ball, t_max=self.t_max, x0=self.x0, Mf=Mf, Ff=Ff, axf=axf, 
      ayf=ayf, MPCprms=self.MPCprms)
    sim.status = self.status
    sim.control_mode = self.control_mode
    sim.sol = self.sol
    
    if "t_MPChist" in dir(self):
      # Older versions may not have these
      sim.t_MPChist = self.t_MPChist
      sim.v_MPChist = self.v_MPChist
    
    return sim
  
