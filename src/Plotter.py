""" Plotter classes with methods for plotting

TODO: Sensor Plotter

TODO: xhat (observer)
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class PlotterX:
  """ Plot the solution results, x(t) and alphadd(t)
  
  Can be used interactively or plotted all at once
  """
  
  def __init__(self, interactive=True, obs=True, t0=0, x0=None, u0=0):
    
    self.obs = obs
    
    self.setup_fig()
    
    if interactive:
      plt.ion()
      if x0 is None:
        x0 = np.zeros(11)
        x0[0] = 1
      self.init_interactive(t0, x0, u0)
      self.fig.show()
  
  def setup_fig(self):
    """ Create the figure and set up the labels
    Everything but the data
    Returns fig, axs (axs is a 3x2 grid)
    """
    # Create figure with 3x2 grid
    self.fig, self.axs = plt.subplots(3,2, sharex=True)
    # Move & resize
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(25,50,1450,900)
    self.fig.subplots_adjust(left=0.05, right=0.98, bottom=0.09, top=0.96,
      wspace=.13, hspace=.19)
    # 0,0 - Input alpha acceleration
    # self.axs[0,0].set_title(r"Input gyro acceleration $\ddot{\alpha}$")
    self.axs[0,0].set_title(r"Input gyro pwm $u$")
    # 0,1 
    self.axs[0,1].set_title(r"Gyro angle & velocity $\alpha$, $\dot{\alpha}$")
    # 1,0
    self.axs[1,0].set_title("Orientation $q$")
    # 1,1
    self.axs[1,1].set_title("Angular Velocity $\omega$")
    # 2,0
    self.axs[2,0].set_xlabel("Time t")
    # self.axs[2,0].set_title("X-Position $r_x$")
    self.axs[2,0].set_title("Position $r_x$, $r_y$")
    # 2,1
    self.axs[2,1].set_xlabel("Time t")
    self.axs[2,1].set_title("Accelerometer Data $y_m$")
    
    # TODO: Put X&Y together, and put sensor data in the other
    #   Or add a row if I want more than accel data. Or remove rx & ry
  
  def plot_all(self, t, x, u):
    """ Plot entire data vectors at once (non-interactive)
    
    x: (n_t, n_x) numpy array of solution
    """
    
    # alphadd / u
    self.axs[0,0].plot(t, u)
    # alpha & alpha-dot
    self.axs[0,1].plot(t, x[:,9], label=r"$\alpha$")
    self.axs[0,1].plot(t, x[:,10], label=r"$\dot{\alpha}$")
    self.axs[0,1].legend()
    # q
    self.axs[1,0].plot(t, x[:,0], label=r"$\eta$")
    self.axs[1,0].plot(t, x[:,1], label=r"$\varepsilon_x$")
    self.axs[1,0].plot(t, x[:,2], label=r"$\varepsilon_y$")
    self.axs[1,0].plot(t, x[:,3], label=r"$\varepsilon_z$")
    qnorm = np.linalg.norm(x[:,0:4], axis=1)
    self.axs[1,0].plot(t, qnorm, label="|$q$|")
    self.axs[1,0].legend()
    # omega_s
    self.axs[1,1].plot(t, x[:,4], label="$\omega_x$")
    self.axs[1,1].plot(t, x[:,5], label="$\omega_y$")
    self.axs[1,1].plot(t, x[:,6], label="$\omega_z$")
    wnorm = np.linalg.norm(x[:,4:7], axis=1)
    self.axs[1,1].plot(t, wnorm, label="|$\omega$|")
    self.axs[1,1].legend()
    # r_x & r_y
    self.axs[2,0].plot(t, x[:,7])
    self.axs[2,1].plot(t, x[:,8])
  
  def init_interactive(self, t0, x0, u0):
    # This dictionary holds all the plotter handles
    self.handles = {}
    
    # alphadd / u
    self.handles["alphadd"] = self.axs[0,0].plot(t0, u0, 
      color="tab:blue")[0]
    # alpha & alpha-dot
    self.handles["alpha"] = self.axs[0,1].plot(t0, x0[9], label=r"$\alpha$", 
      color="tab:blue")[0]
    self.handles["alphad"] = self.axs[0,1].plot(t0, x0[10], 
      label=r"$\dot{\alpha}$", color="tab:orange")[0]
    # q
    self.handles["q0"] = self.axs[1,0].plot(t0, x0[0], 
      label=r"$\eta$", color="tab:blue")[0]
    self.handles["q1"] = self.axs[1,0].plot(t0, x0[1], 
      label=r"$\varepsilon_x$", color="tab:orange")[0]
    self.handles["q2"] = self.axs[1,0].plot(t0, x0[2], 
      label=r"$\varepsilon_y$", color="tab:green")[0]
    self.handles["q3"] = self.axs[1,0].plot(t0, x0[3], 
      label=r"$\varepsilon_z$", color="tab:red")[0]
    qnorm = np.linalg.norm(x0[0:4])
    self.handles["qn"] = self.axs[1,0].plot(t0, qnorm, 
      label="|$q$|", color="tab:purple")[0]
    # omega_s
    self.handles["wx"] = self.axs[1,1].plot(t0, x0[4], 
      label="$\omega_x$", color="tab:blue")[0]
    self.handles["wy"] = self.axs[1,1].plot(t0, x0[5], 
      label="$\omega_y$", color="tab:orange")[0]
    self.handles["wz"] = self.axs[1,1].plot(t0, x0[6], 
      label="$\omega_z$", color="tab:green")[0]
    wnorm = np.linalg.norm(x0[4:7])
    self.handles["wn"] = self.axs[1,1].plot(t0, wnorm, 
      label="|$\omega$|", color="tab:purple")[0]
    # r_x & r_y
    self.handles["rx"] = self.axs[2,0].plot(t0, x0[7], 
      label="$r_x$")[0]
    self.handles["ry"] = self.axs[2,0].plot(t0, x0[8], 
      label="$r_y$")[0]
    # y_m
    self.handles["ym0"] = self.axs[2,1].plot(t0, 0, color="tab:blue",
      label="$\ddot{r}_{ax}$")[0]
    self.handles["ym1"] = self.axs[2,1].plot(t0, 0, color="tab:orange",
      label="$\ddot{r}_{ay}$")[0]
    self.handles["ym2"] = self.axs[2,1].plot(t0, 0, color="tab:green",
      label="$\ddot{r}_{az}$")[0]
    
    if self.obs:
      ## Repeat, but for x-hat (assuming accurate xhat0)
      # alpha & alpha-dot
      self.handles["alphah"] = self.axs[0,1].plot(t0, x0[9], 
        label=r"$\hat{\alpha}$", linestyle="dashed", color="tab:blue")[0]
      self.handles["alphadh"] = self.axs[0,1].plot(t0, x0[10], 
        label=r"$\hat{\dot{\alpha}}$", linestyle="dashed", 
        color="tab:orange")[0]
      # q
      self.handles["q0h"] = self.axs[1,0].plot(t0, x0[0],
        label=r"$\hat{\eta}$", linestyle="dashed", color="tab:blue")[0]
      self.handles["q1h"] = self.axs[1,0].plot(t0, x0[1], 
        label=r"$\hat{\varepsilon}_x$", linestyle="dashed", 
        color="tab:orange")[0]
      self.handles["q2h"] = self.axs[1,0].plot(t0, x0[2], 
        label=r"$\hat{\varepsilon}_y$", linestyle="dashed", 
        color="tab:green")[0]
      self.handles["q3h"] = self.axs[1,0].plot(t0, x0[3], 
        label=r"$\hat{\varepsilon}_z$", linestyle="dashed", 
        color="tab:red")[0]
      self.handles["qnh"] = self.axs[1,0].plot(t0, qnorm, 
        label="|$\hat{q}$|", linestyle="dashed", color="tab:purple")[0]
      # omega_s
      self.handles["wxh"] = self.axs[1,1].plot(t0, x0[4], 
        label="$\hat{\omega}_x$", linestyle="dashed", color="tab:blue")[0]
      self.handles["wyh"] = self.axs[1,1].plot(t0, x0[5], 
        label="$\hat{\omega}_y$", linestyle="dashed", color="tab:orange")[0]
      self.handles["wzh"] = self.axs[1,1].plot(t0, x0[6], 
        label="$\hat{\omega}_z$", linestyle="dashed", color="tab:green")[0]
      self.handles["wnh"] = self.axs[1,1].plot(t0, wnorm, 
        label="|$\hat{\omega}$|", linestyle="dashed", color="tab:purple")[0]
      # r_x & r_y
      # self.handles["rxh"] = self.axs[2,0].plot(t0, x0[7], 
        # label="$\hat{x}$", linestyle="dashed")[0]
      # self.handles["ryh"] = self.axs[2,0].plot(t0, x0[8], 
        # label="$\hat{y}$", linestyle="dashed")[0]
    
    # Make legends
    self.axs[0,1].legend(loc='upper left')
    self.axs[1,0].legend(loc='upper left')
    self.axs[1,1].legend(loc='upper left')
    self.axs[2,0].legend(loc='upper left')
    self.axs[2,1].legend(loc='upper left')
    
  def update_interactive(self, v_t, v_x, v_u, v_ym, v_xhat=None):
    
    for handle in self.handles.values():
      # They all have the same xdata
      handle.set_xdata(v_t)
    
    # alphadd / u
    self.handles["alphadd"].set_ydata(v_u)
    # alpha & alpha-dot
    self.handles["alpha"].set_ydata(v_x[:,9])
    self.handles["alphad"].set_ydata(v_x[:,10])
    # q
    self.handles["q0"].set_ydata(v_x[:,0])
    self.handles["q1"].set_ydata(v_x[:,1])
    self.handles["q2"].set_ydata(v_x[:,2])
    self.handles["q3"].set_ydata(v_x[:,3])
    qnorm = np.linalg.norm(v_x[:,0:4], axis=1)
    self.handles["qn"].set_ydata(qnorm)
    # omega_s
    self.handles["wx"].set_ydata(v_x[:,4])
    self.handles["wy"].set_ydata(v_x[:,5])
    self.handles["wz"].set_ydata(v_x[:,6])
    wnorm = np.linalg.norm(v_x[:,4:7], axis=1)
    self.handles["wn"].set_ydata(wnorm)
    # r_x & r_y
    self.handles["rx"].set_ydata(v_x[:,7])
    self.handles["ry"].set_ydata(v_x[:,8])
    # accelerometer
    self.handles["ym0"].set_ydata(v_ym[:,0])
    self.handles["ym1"].set_ydata(v_ym[:,1])
    self.handles["ym2"].set_ydata(v_ym[:,2])
    
    if self.obs:
      ## Repeat, but for x-hat
      # alpha & alpha-dot
      self.handles["alphah"].set_ydata(v_xhat[:,9])
      self.handles["alphadh"].set_ydata(v_xhat[:,10])
      # q
      self.handles["q0h"].set_ydata(v_xhat[:,0])
      self.handles["q1h"].set_ydata(v_xhat[:,1])
      self.handles["q2h"].set_ydata(v_xhat[:,2])
      self.handles["q3h"].set_ydata(v_xhat[:,3])
      qnorm = np.linalg.norm(v_xhat[:,0:4], axis=1)
      self.handles["qnh"].set_ydata(qnorm)
      # omega_s
      self.handles["wxh"].set_ydata(v_xhat[:,4])
      self.handles["wyh"].set_ydata(v_xhat[:,5])
      self.handles["wzh"].set_ydata(v_xhat[:,6])
      wnorm = np.linalg.norm(v_xhat[:,4:7], axis=1)
      self.handles["wnh"].set_ydata(wnorm)
      # r_x & r_y
      # self.handles["rxh"].set_ydata(v_xhat[:,7])
      # self.handles["ryh"].set_ydata(v_xhat[:,8])
    
    # Adjust limits
    for ax in self.axs.flatten():
      ax.relim()
      ax.autoscale()
    # Pause so it can update the display
    # self.fig.canvas.draw_idle()
    plt.pause(0.002)

class AnimatorXR:
  """ Animator for position and reference
  """
  
  autoscale = False
  
  def __init__(self, ref=None, ref_type="p", lims=[(-3,3),(-3,3)]):
    
    self.ref = ref
    self.ref_type = ref_type
    self.lims = lims
    
    self.setup_fig()
    plt.ion()
    self.fig.show()
  
  def setup_fig(self):
    # Open figure & set location & size
    self.fig = plt.figure()
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(1500,100,375,400)
    # Make axis
    self.ax = plt.axes(xlim=self.lims[0], ylim=self.lims[1])
    self.ax.set_aspect("equal")
    self.ax.grid()
    # Ball path dotted line
    self.h_path = self.ax.scatter([], [], s=4, color="#00f5", marker=".")
    # Ball position marker
    self.h_ball, = self.ax.plot([], [], marker="o", markersize=16, 
      markerfacecolor="#00f8")
    # Reference
    if self.ref_type == "p":
      self.h_ref, = self.ax.plot([], [], linestyle="-", color="#0b05", marker=".", markersize=4)
    else:
      print("ref_type besides 'p' not currently supported for animation")
      # TODO: show v-ref & a-refs as an arrow
  
  def update(self, v_t, v_x):
    """ Update the drawing
    """
    
    # Update ball position
    rx = v_x[-1,7]
    ry = v_x[-1,8]
    self.h_ball.set_data([rx], [ry])
    # Update ball path
    self.h_path.set_offsets(v_x[:,7:9])
    # Update reference
    if self.ref is not None:
      v_ref = np.array([self.ref(ti) for ti in v_t])
      if self.ref_type == "p":
        self.h_ref.set_data(v_ref[:,0], v_ref[:,1])
    
    if self.autoscale:
      # Adjust limits
      self.ax.relim()
      self.ax.autoscale()
    # Pause so it can update the display
    # self.fig.canvas.draw_idle()
    plt.pause(0.002)
  

# TODO: Make classes?
def plot_anim(t, x, p_des=None, a_des=None, fig=None, ax=None):
  """ Plot an animation of the ball rolling
  """
  
  if fig is None or ax is None:
    # Infer x-bounds for animation
    xmin = np.min(x[:,7])
    xmax = np.max(x[:,7])
    xspan = xmax-xmin
    ymin = np.min(x[:,8])
    ymax = np.max(x[:,8])
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
    rx = x[i,7]
    ry = x[i,8]
    circle.set_data([rx], [ry])
    rh.set_offsets(x[0:i,7:9])
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
  input("PRESS ANY KEY TO CONTINUE")
  
  return fig, ax

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

def plot_tym(t, ym):
  """ Plot the measured output from simulation results x(t) and input u(t)
  
  Returns fig, ax
  """
  
  rddx = [accel[0] for accel in ym]
  rddy = [accel[1] for accel in ym]
  rddz = [accel[2] for accel in ym]
  fig, ax = plt.subplots()
  ax.plot(t, rddx, label="$\ddot{r}_x$")
  ax.plot(t, rddy, label="$\ddot{r}_y$")
  ax.plot(t, rddz, label="$\ddot{r}_z$")
  fig.suptitle("Simulated sensor measurements")
  fig.legend()
  
  return fig, ax
