""" Plotting functions
"""

import numpy as np
import scipy.integrate as spi
from scipy.misc import derivative as spd
from matplotlib import pyplot as plt
from matplotlib import animation

def plot_sol(sol, t, alphadf, show=True, animate=True, px=None, py=None):
  """ Plots sol.dill
    TODO: Make sol a parameter
  
  Parameters
  ----------
  sol: spi.solve_ivp solution path to plot
    If None, attempt to load from sol.dill
  t: time vector
    If None, then use sol.t
  alphadf: input (alpha-dot) function used to create simulation
  show (bool): Whether to show the plots now or just return the fig objects
  animate (bool): Whether to plot animation
  px, py: target path (px,py), plotted for reference behind animation
  """
  
  ## Input checking
  
  if sol is None:
    with open("sol.dill", "rb") as file:
      sol = dill.load(file)
  #print(f"dir(sol): {dir(sol)}")

  if t is None:
    t = sol.t
  else:
    # Check bounds on solution
    tmin = min(sol.t)
    tmax = max(sol.t)
    if tmin > min(t) or tmax < max(t):
      print("Warning: solution object was not solved over whole t vector.")
      print(f"\tsol.t = [{tmin}, {tmax}]")
      print(f"\tt = [{min(t)}, {max(t)}]")
  
  # Evaluate solution at desired time values
  x = sol.sol(t)
  
  # Integrate & differentiate alphad
  alphaddf = lambda t: spd(alphadf, t, dx=1e-6)
  alphaf = lambda t: spi.quad(alphadf, 0, t, limit=100)[0]

  # Plot
  fig1, axs = plt.subplots(3,2, sharex=True)
  alphad = [alphadf(ti) for ti in t]
  axs[0,0].plot(t, alphad)
  axs[0,0].set_title(r"Gyro angle rate $\dot{\alpha}$ (rad/s)")
  alpha = [alphaf(ti) for ti in t]
  axs[0,1].plot(t, alpha)
  axs[0,1].set_title(r"Gyro angle $\alpha$ (rad)")
  axs[1,0].plot(t, x[0,:], label="nu")
  axs[1,0].plot(t, x[1,:], label="ex")
  axs[1,0].plot(t, x[2,:], label="ey")
  axs[1,0].plot(t, x[3,:], label="ez")
  qnorm = np.linalg.norm(x[0:4,:], axis=0)
  axs[1,0].plot(t, qnorm, label="|q|")
  axs[1,0].legend()
  axs[1,0].set_title("Orientation Q")
  axs[1,1].plot(t, x[4,:], label="$\omega_x$")
  axs[1,1].plot(t, x[5,:], label="$\omega_y$")
  axs[1,1].plot(t, x[6,:], label="$\omega_z$")
  wnorm = np.linalg.norm(x[4:7,:], axis=0)
  axs[1,1].plot(t, wnorm, label="|$\omega$|")
  axs[1,1].legend()
  axs[1,1].set_title("Angular Velocity $\omega$")
  axs[2,0].plot(t, x[7,:])
  axs[2,0].set_xlabel("Time t")
  axs[2,0].set_title("X-Position $r_x$")
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
  
  # Animate
  # https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
  fig2 = plt.figure()
  #ax = plt.axes(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
  ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
  ax.set_aspect("equal")
  ax.grid()
  if (px is not None) and (py is not None):
    ph, = ax.plot(px, py, linestyle="-", color="g")
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
