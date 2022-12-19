"""
Simulation class to hold the parameters of a simulation
"""

import numpy as np
# import scipy.integrate as spi
# from scipy.misc import derivative as spd
from matplotlib import pyplot as plt
from matplotlib import animation
import dill

from CMGBall import CMGBall

class Simulation:
  status = "unsolved"
  alphaddf = None
  control_mode = None
  
  def __init__(self, alphadd, p_des=None, a_des=None, ball=CMGBall(), t_max=1, 
      Mf=None, Ff=None):
    """
    alphadd: Gyro acceleration alpha-double-dot
      Either pass a scalar, a function of time, or one of the following:
        "FF": Feedforward control
      If one of those strings is passed, then p_des, v_des, or a_des must be 
        passed as well.
      p_des: Desired position. Either a function of time or a 2x0 point.
      a_des: Desired acceleration. Either a function of time or a 2x0 vector.
      ball: A CMGBall object
    """
    
    ## Input handling
    if isinstance(alphadd, float):
      self.alphaddf = lambda t: alphadd
    elif callable(alphadd):
      self.alphaddf = alphadd
    elif alphadd == "FF":
      self.control_mode = "FF"
    
    if self.control_mode is not None:
      assert (p_des is not None) or (a_des is not None), ("Must provide a "
        "desired path or acceleration when using a control mode")
    
    self.ball = ball
    
    self.t_max = t_max
    
    if Mf is None or Ff is None:
      self.Mf, self.Ff = self.load_MfFf()
  
  def load_MfFf(self):
    M, F = dyn.load_MF()
    return dyn.lambdify_MF(M, F, ball=self.ball)
  
  def save(self, fname="sim.dill"):
    # Save to file
    with open(fname,"wb") as file:
      dill.dump(self, file)
    print(f"Simulation saved to {fname}")
  
  def load(fname="sim.dill")
    with open(fname, "rb") as file:
      sim = dill.load(file)
    print(f"Simulation loaded from {fname}. Status={sim.status}")
    return sim
  
  def plot(self, animate=True):
    """ Plots the solution results
    
    Parameters
    ----------
    alphaddf: input (alpha-double-dot) function used to create simulation
    show (bool): Whether to show the plots now or just return the fig objects
    animate (bool): Whether to plot animation
    px, py: target path (px,py), plotted for reference behind animation
    """
    
    # Error checking
    if self.status != "solved":
      print("This simulation still needs to be run")
      return
    assert self.sol is not None, "Error, self.sol undefined"
    
    # Make time vector
    t = np.linspace(0, self.t_max, 100)
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
    axs[1,0].plot(t, qnorm, label="|q|")
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
    fig1.show()
    
    # Infer x-bounds for animation
    xmin = np.min(x[7,:])
    xmax = np.max(x[7,:])
    xspan = xmax-xmin
    ymin = np.min(x[8,:])
    ymax = np.max(x[8,:])
    yspan = ymax-ymin
    xmin -= .1*xspan # Add 10% margins
    xmax += .1*xspan
    ymin -= .1*yspan
    ymax += .1*yspan
    
    # if a_desf is not None:
      # # Convert a_desf to px, py
      # v_xy = lambda ti: spi.quad_vec(a_desf, min(t), ti)[0]
      # p_xy = lambda ti: spi.quad_vec(v_xy, min(t), ti)[0]
      # pxy = np.vstack([p_xy(ti) for ti in t])
      # px = pxy[:,0]
      # py = pxy[:,1]
    
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
    def init():
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
    anim = animation.FuncAnimation(fig2, animate, init_func=init,
      frames=t.size, interval=20, repeat_delay=2000, blit=True)

    if show:
      fig2.show()
      input("PRESS ANY KEY TO QUIT")
    
    return fig1, fig2

