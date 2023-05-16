"""
Simulation class to hold the parameters of a simulation
"""

import numpy as np
import scipy.integrate as spi
# from scipy.misc import derivative as spd
from matplotlib import pyplot as plt
from matplotlib import animation
import dill

class Simulation:
  status = "unsolved"
  sol = None
  
  def __init__(self, controller, t_max=1, x0=None):
    """
    controller: Controller object that calculates control inputs
    t_max: Simulation end time
    x0: Initial conditions (see dynamics.eom for state vector description)
    """
    
    self.controller = controller
    self.t_max = t_max
    
    if x0 is None:
      x0 = np.zeros(11)
      x0[0] = 1 # Real part of quaternion starts at 1
    self.x0 = x0
  
  def __str__(self):
    s = "Simulation Object\n"
    s += f"\tstatus: {self.status}\n"
    # Cut off final \n and indent the whole block
    scnt = str(self.controller)[0:-1].replace("\n", "\n\t")
    s += f"\tcontroller: {scnt}\n"
    s += f"\tt_max: {self.t_max}\n"
    s += f"\tx0: {self.x0}\n"
    return s
  
  @property
  def ball(self):
    return self.controller.ball
  
  def run(self, t_eval=None, fname="sim.dill"):
    """ Run the simulation and store the result
    fname (str or None): If str, then save the Simulation object to the given
      filename upon completion.
    """
    
    xdot = lambda t,x: self.ball.eom(x, self.controller.update(t, x))
    
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
    """ Evaluate the solved simulation at discrete time steps
    """
    # Error checking
    if self.status != "solved":
      print("This simulation still needs to be run")
      return
    assert self.sol is not None, "Error, self.sol undefined"

    if t is None:
      # Make time vector
      t = np.linspace(0, self.t_max, 200)
      # Evaluate solution at desired time values
      x = self.sol.sol(t)
    
    # Check bounds on solution
    tmin = np.min(self.sol.t)
    tmax = np.max(self.sol.t)
    if tmin > 0 or tmax < np.max(t):
      print("Warning: solution object was not solved over whole t vector.")
      print(f"\tsol.t = [{tmin}, {tmax}]")
      print(f"\tt = [{min(t)}, {max(t)}]")
    
    # Evaluate solution at desired time values
    if np.all(self.sol.t == t):
      x = self.sol.y
    else:
      x = self.sol.sol(t)
    
    ## Evaluate alphadd @ t
    #alphadd = [self.controller.update(ti, xi) for ti, xi in zip(t, x)]
    
    # Differentiate alphad
    alphadd = np.gradient(x[10,:], t)
    
    return t, x, alphadd
  
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
  
  @staticmethod
  def plot_tx(t, x, alphadd, fig=None, axs=None):
    """ Plot the solution results x(t) and alphadd(t)
    Returns fig, axs (axs is a 3x2 grid)
    """
    
    if fig is None and axs is None:
      fig, axs = plt.subplots(3,2, sharex=True)
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
    
    return fig, axs
  
  def plot_tym(self, t, ym=None, x=None, alphadd=None):
    """ Plot the measured output from simulation results x(t) and input 
      alphadd(t)
      
    Either pass ym or x & alphadd
    
    Returns fig, ax
    """
    
    if ym is None:
      accels = [self.controller.ball.measure(xi, alphaddi)[0] for (xi,alphaddi) 
        in zip(x.T, alphadd)]
    else:
      accels = ym[0,:]
    rddx = [accel[0] for accel in accels]
    rddy = [accel[1] for accel in accels]
    rddz = [accel[2] for accel in accels]
    fig, ax = plt.subplots()
    ax.plot(t, rddx, label="$\ddot{r}_x$")
    ax.plot(t, rddy, label="$\ddot{r}_y$")
    ax.plot(t, rddz, label="$\ddot{r}_z$")
    fig.legend()
    
    return fig, ax
  
  @staticmethod
  def plot_anim(t, x, p_des=None, a_des=None, fig=None, ax=None):
    """ Plot an animation of the ball rolling
    """
    
    if fig is None or ax is None:
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
      # Make figure
      fig = plt.figure()
      #ax = plt.axes(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
      ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
      ax.set_aspect("equal")
      ax.grid()
    
    # Build px, py vectors (desired path) #TODO
    if False:
      px = None
      py = None
      if p_des is not None:
        px = np.zeros(t.shape)
        py = np.zeros(t.shape)
        for i, ti in enumerate(t):
          pxi, pyi = p_des(ti)
          px[i] = pxi
          py[i] = pyi
      elif a_des is not None:
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
    
    # Set up animation
    ph = None
    # Desired path line
    if False:# (px is not None):
      #print(px, py)
      #ph, = ax.plot(px, py, linestyle="-", color="g", marker=".")
      ph, = ax.plot([px[0]], [py[0]], linestyle="-", color="g", marker=".")
    # Ball path dotted line
    rh = ax.scatter([], [], s=5, color="b", marker=".")
    # Ball position marker
    circle, = ax.plot([0], [0], marker="o", markerfacecolor="b")

    # initialization function: plot the background of each frame
    def init_back():
      circle.set_data([], [])
      rh.set_offsets(np.array((0,2)))
      if False:# (px is not None):
        ph.set_data([], [])
        return circle, rh, ph
      return circle, rh

    # animation function. This is called sequentially
    def animate(i):
      rx = x[7,i]
      ry = x[8,i]
      circle.set_data([rx], [ry])
      rh.set_offsets(x[7:9,0:i].T)
      if False: # px is not None:
        #print(px[0:i])
        ph.set_data(px[0:i], py[0:i])
        return circle, rh, ph
      return circle, rh

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init_back,
      frames=t.size, interval=20, repeat_delay=5000, blit=True)
    
    # This has to be here, or else something important gets garbage collected
    #   at the end of the function.
    fig.show()
    input("PRESS ANY KEY TO QUIT")
    
    return fig, ax
  
  @staticmethod
  def plot_anims(t, xs, p_des, a_des, fig=None, ax=None):
    """ Plot several simultaneous animation of the ball rolling
    """
    
    if fig is None or ax is None:
      # Infer x-bounds for animation
      xmin=0
      xmax=0
      ymin=0
      ymax=0
      for x in xs:
        xmin = min(xmin, np.min(x[7,:]))
        xmax = max(xmax, np.max(x[7,:]))
        ymin = min(ymin, np.min(x[8,:]))
        ymax = max(ymax, np.max(x[8,:]))
      xspan = xmax-xmin
      yspan = ymax-ymin
      margin = .1*max(xspan, yspan) # Add 10% margins
      # Make figure
      fig = plt.figure()
      ax = plt.axes(xlim=(xmin-margin, xmax+margin), 
        ylim=(ymin-margin, ymax+margin))
      ax.set_aspect("equal")
      ax.grid()
    
    # Build px, py vectors (desired path)
    px = None
    py = None
    if p_des is not None:
      px = np.zeros(t.shape)
      py = np.zeros(t.shape)
      for i, ti in enumerate(t):
        pxi, pyi = p_des(ti)
        px[i] = pxi
        py[i] = pyi
    elif a_des is not None:
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
    
    # Set up animation
    ph = None
    # Desired path line
    if (px is not None):
      #print(px, py)
      #ph, = ax.plot(px, py, linestyle="-", color="g", marker=".")
      ph, = ax.plot([px[0]], [py[0]], linestyle="-", color="g", marker=".")
    # Ball path dotted lines
    rhs = []
    chs = []
    for x in xs:
      rhs.append(ax.scatter([], [], s=5, color="b", marker="."))
      # Ball position marker
      circle, = ax.plot([0], [0], marker="o", markerfacecolor="b")
      chs.append(circle)

    # initialization function: plot the background of each frame
    def init_back():
      for circle in chs:
        circle.set_data([], [])
      for rh in rhs:
        rh.set_offsets(np.array((0,2)))
      if (px is not None):
        ph.set_data([], [])
        return ph, *rhs, *chs
      return *rhs, *chs

    # animation function. This is called sequentially
    def animate(i):
      for x, rh, ch in zip(xs, rhs, chs):
        rx = x[7,i]
        ry = x[8,i]
        circle.set_data([rx], [ry])
        rh.set_offsets(x[7:9,0:i].T)
      if (px is not None):
        #print(px[0:i])
        ph.set_data(px[0:i], py[0:i])
        return ph, *rhs, *chs
      return *rhs, *chs

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init_back,
      frames=t.size, interval=20, repeat_delay=5000, blit=True)
    
    # This has to be here, or else something important gets garbage collected
    #   at the end of the function.
    fig.show()
    input("PRESS ANY KEY TO QUIT")
    
    return fig, ax
  
  def plot(self, t_eval=None, show=True):
    """ Plots the solution results
    show (bool): Whether to show the plots now or simply return the figs
    """
    
    # Evaluate the solution and the input function at discrete time steps
    t, x, alphadd = self.xeval(t=t_eval)

    # Plot state vector as a function of time
    fig1, axs1 = self.plot_tx(t, x, alphadd)
    
    # Plot measured values as a function of time
    fig2, ax2 = self.plot_tym(t, x=x, alphadd=alphadd)
    
    if show:
      fig1.show()
      fig2.show()
    
    # Animation
    fig3, ax3 = self.plot_anim(t, x)#, self.p_des, self.a_des)
    
    return fig1, fig2, fig3

class SerializableSim:
  def __init__(self, sim):
    """
    Convert a Simulation object (sim) to a serializable form.
    This allows it to be saved to file.
    The lambdified sympy functions have a hard time being serialized, so leave 
      them out.
    """
    
    self.controller = sim.controller
    self.t_max = sim.t_max
    self.x0 = sim.x0
    # The following are not in the Simulation constructor
    self.status = sim.status
    self.sol = sim.sol
    # Make controller.ball serializable
    self.controller = self.controller.serializable()
    # self.controller.ball = sim.controller.ball.serializable()
  
  def to_sim(self):
    """
    Convert this to a normal Simulation object
    """
    
    # Convert the SerializableBall in controller back to a CMGBall
    self.controller = self.controller.from_serializable()
    # self.controller.ball = self.controller.ball.to_ball()
    
    sim = Simulation(self.controller, t_max=self.t_max, x0=self.x0)
    sim.status = self.status
    sim.sol = self.sol
    
    return sim
  

if __name__ == "__main__":
  # TEST Serialization
  from CMGBall import CMGBall
  from Controller import PreSet
  ball = CMGBall()
  cnt = PreSet(ball, 1.0)
  sim = Simulation(cnt)
  print(447, sim)
  # print(448, dir(sim))
  # print(449, sim.controller)
  # print(450, sim.controller.ball)
  fname ="testinggggg.dill"
  sim.save(fname)
  siml = Simulation.load(fname)
  print(453, siml)