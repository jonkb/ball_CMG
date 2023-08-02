""" GUI.py

Classes for the GUI for controlling the robot
"""

import os
import tkinter as tk
import tkinter.ttk as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

ico_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../img/phys.ico")

class ToolTip:
  def __init__(self, widget, text):
    self.widget = widget
    self.text = text
    self.tip_window = None

  def show_tip(self):
    if self.tip_window or not self.text:
      return
    x, y, _cx, cy = self.widget.bbox("insert")
    x = x + self.widget.winfo_rootx() + 25
    y = y + cy + self.widget.winfo_rooty() + 25
    self.tip_window = tw = tk.Toplevel(self.widget)
    tw.wm_overrideredirect(True)
    tw.wm_geometry(f"+{x}+{y}")
    label = tk.Label(tw, text=self.text, justify=tk.LEFT,
      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
      font=("tahoma", "8", "normal"))
    label.pack(ipadx=1)

  def hide_tip(self):
    tw = self.tip_window
    self.tip_window = None
    if tw:
        tw.destroy()

class CtrlFrm(ttk.Frame):
  # Keep track of current motor commands
  u = np.array([0.0, 0.0])
  # How much to change u per button press
  u_inc = 0.2
  font_H18 = ("Helvetica", 18)
  
  def __init__(self, root, queue_u):
    super().__init__(root)
    self.queue_u = queue_u
    
    s = ttk.Style()
    s.configure("H18.TButton", font=self.font_H18)
    
    # Display current motor inputs
    self.lbl_m0 = ttk.Label(self, text="Omega motor:")
    self.lbl_m0.grid(row=0, column=0, padx=5, pady=5)
    self.lbl_u0 = ttk.Label(self, text=str(self.u[0]))
    self.lbl_u0.grid(row=0, column=1, padx=5, pady=5)
    self.lbl_m1 = ttk.Label(self, text="Alpha motor:")
    self.lbl_m1.grid(row=1, column=0, padx=5, pady=5)
    self.lbl_u1 = ttk.Label(self, text=str(self.u[1]))
    self.lbl_u1.grid(row=1, column=1, padx=5, pady=5)
    
    # Display buttons
    # Omega speed down
    self.btn_m0d = ttk.Button(self, text=u"\u21E9", style="H18.TButton",
      command=lambda: self.inc_speed(m=0, inc=False))
    self.btn_m0d.grid(row=0, column=2, padx=5, pady=5)
    self.add_tooltip(self.btn_m0d, "Decrease Omega motor speed")
    # Omega speed up
    self.btn_m0u = ttk.Button(self, text=u"\u21E7", style="H18.TButton",
      command=lambda: self.inc_speed(m=0, inc=True))
    self.btn_m0u.grid(row=0, column=3, padx=5, pady=5)
    self.add_tooltip(self.btn_m0u, "Increase Omega motor speed")
    # Alpha speed down
    self.btn_m1u = ttk.Button(self, text=u"\u21E6", style="H18.TButton",
      command=lambda: self.inc_speed(m=1, inc=False))
    self.btn_m1u.grid(row=1, column=2, padx=5, pady=5)
    self.add_tooltip(self.btn_m1u, "Decrease Alpha motor speed")
    # Alpha speed up
    self.btn_m1d = ttk.Button(self, text=u"\u21E8", style="H18.TButton",
      command=lambda: self.inc_speed(m=1, inc=True))
    self.btn_m1d.grid(row=1, column=3, padx=5, pady=5)
    self.add_tooltip(self.btn_m1d, "Increase Alpha motor speed")
    
    # Add keybindings
    self.master.bind("<Down>", lambda e: self.inc_speed(m=0, inc=False))
    self.master.bind("<Up>", lambda e: self.inc_speed(m=0, inc=True))
    self.master.bind("<Left>", lambda e: self.inc_speed(m=1, inc=False))
    self.master.bind("<Right>", lambda e: self.inc_speed(m=1, inc=True))
  
  def add_tooltip(self, widget, text):
    tip = ToolTip(widget, text)
    widget.bind("<Enter>", lambda event: tip.show_tip())
    widget.bind("<Leave>", lambda event: tip.hide_tip())
  
  def inc_speed(self, m=0, inc=True):
    # Increment (inc=True) or decrement (inc=False) the given motor speed
    if inc:
      self.u[m] = min(1.0, self.u[m] + self.u_inc)
    else:
      self.u[m] = max(-1.0, self.u[m] - self.u_inc)
    # Update appropriate label
    if m == 0:
      self.lbl_u0.config(text=f"{self.u[0]:.3f}")
    elif m == 1:
      self.lbl_u1.config(text=f"{self.u[1]:.3f}")
    # Push this update to the motor speeds queue
    self.queue_u.put(self.u)

class PltWin(tk.Toplevel):
  def __init__(self, root, fig):
    super().__init__(root)
    # General setup: Title & icon
    self.title("CMGBall Data Plotter")
    try:
      self.iconbitmap(ico_path)
    except:
      print("Error loading icon ("+ico_path+")")
    
    # Canvas to hold figure
    # https://www.geeksforgeeks.org/how-to-embed-matplotlib-charts-in-tkinter-gui/
    canvas = FigureCanvasTkAgg(fig, master=self)
    canvas.draw()
    
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()

class GUI(tk.Tk):
  def __init__(self, queue_u, fig):
    super().__init__()
    # General setup: Title & icon
    self.title("CMGBall Controls")
    try:
      self.iconbitmap(ico_path)
    except:
      print(f"Error loading icon: {ico_path}")
    
    # Controller frame
    self.ctrl_frm = CtrlFrm(self, queue_u)
    self.ctrl_frm.pack()
    
    # Secondary window for plots
    self.plt_win = PltWin(self, fig)

if __name__ == "__main__":
  from queue import Queue
  import matplotlib.pyplot as plt
  # For testing
  queue_u = Queue()
  fig = plt.figure()
  gui = GUI(queue_u, fig)
  gui.protocol("WM_DELETE_WINDOW", cleanup)
  gui.mainloop()
  