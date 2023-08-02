""" HTTP server to communicate with robot

Premise:
* Robot periodically reports its measurements and expects motor commands
* Main loop runs on laptop, updating simulation & plots at a constant dt
* Separate thread is running server. When a GET request comes in, send those
  measurements to update the observer and return the current motor commands
  from the controller.

Message passing between threads:
* ym queue: server posts messages to a queue that the Observer is listening to
* u queue: Controller posts control updates to this queue. The HTTP server 
  checks this queue for updates before sending motor commands to the ESP32

References
  https://docs.python.org/3/library/http.server.html
  https://www.geeksforgeeks.org/python-communicating-between-threads-set-1/

[Not used currently]
Repeating version of threading.Timer class  
  https://stackoverflow.com/questions/12435211/
  https://docs.python.org/3/library/threading.html
"""

import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from queue import Queue
from threading import Thread
import numpy as np
import time

from Observer import ObsSim
from CMGBall import CMGBall
from Plotter import PlotterX
from GUI import GUI

hostName = ""
serverPort = 8888

# class Msg_ym:
  # """ Container object to hold a measurement and the time it was sent
  # Note: In the future, these timestamps could be based on ESP32 time, and
    # maybe that'd be better, but it sounds much harder.
  # """
  
  # def __init__(self, ym):
    # self.ym = ym
    # self.ts = time.time()

def run_server(obs, queue_u):
  """ Start the HTTP server
  
  obs: Observer object
  # queue_ym: Queue to which the server posts measurements
  queue_u: Queue from which the server reads motor commands
  """
  
  # Current motor commands
  u = np.zeros(2)
  
  class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
      urlp = urlparse(self.path)
      if urlp.path == "/accel":
        query = parse_qs(urlp.query)
        self.get_accel_update(query)
      else:
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
    
    def send_u(self, u):
      # The ESP32 converts this to 0-256 int, so .3f should be plenty
      res = f"{u[0]:.3f},{u[1]:.3f}"
      # Send res
      self.send_response(200)
      self.send_header("Content-type", "text/plain")
      self.end_headers()
      self.wfile.write(bytes(res, "utf-8"))
    
    def get_accel_update(self, query):
      """ Handler for GET /accel?ax=...
      """
      
      nonlocal u
      
      try:
        ax = query['ax'][0] # str
        ay = query['ay'][0]
        az = query['az'][0]
      except KeyError:
        print("Error: Required key missing from query string")
      else:
        ym = np.array([ax, ay, az], dtype=np.float32)
        # Update the observer
        obs.update(ym, u) # Intentionally use the previous motor commands
      
      # Send ym to the Observer via queue_ym
      # queue_ym.put(ym)
      # Load any updates to motor commands from queue_u
      while not queue_u.empty():
        # FIFO --> Only the most recent one will matter
        u = queue_u.get()
        if isinstance(u, str) and u == "KILL":
          # TODO: This isn't working
          self.send_u([0,0])
          server.shutdown()
      
      self.send_u(u)
    
    def log_message(self, format, *args):
      # Override default logging method
      print("Request path: ", self.path)
  
  server = HTTPServer((hostName, serverPort), Handler)
  print("Server started at PORT ", serverPort)
  server.serve_forever()

class ObsPlot:
  """ Observer & plotter
  """
  
  t_window = 30
  N_window = int(t_window / 0.01)
  plotting = True
  
  def __init__(self, ball):
    self.ball = ball
    self.t0 = time.time()
    # Assume starting position of xyz = ENU
    self.x0 = np.zeros(11)
    self.x0[0] = 1 # Real part of Q starts at 1
    self.obs = ObsSim(ball, self.x0)
    if self.plotting:
      # History arrays for plotting
      self.v_t = np.zeros(self.N_window)
      self.v_u = np.zeros((self.N_window, self.ball.n_u))
      self.v_ym = np.zeros((self.N_window, self.ball.n_ym))
      self.v_xhat = np.array(self.N_window * [self.x0])
      # Set up plotter
      self.plotter = PlotterX(x0=self.x0, show=False)
      self.plotter.axs[2,0].set_xlim(-self.t_window,0)
      # TODO: Animation -- observer doesn't observe rx, ry so it seems weird
      # self.animator = AnimatorXR(ref=self.cnt.ref, ref_type=self.cnt.ref_type)
  
  def update(self, ym, u):
    
    # Roll v_t & add current time
    self.v_t[1:] = self.v_t[0:-1]
    self.v_t[0] = time.time() - self.t0
    dt = self.v_t[0] - self.v_t[1]
    
    # Update observer
    x_hat = self.obs.update(ym, u[1], dt)
    # TODO: len of u: 2 vs 1
    
    # Update controler - TODO
    
    if self.plotting:
      # Roll arrays besides v_t
      self.v_u[1:] = self.v_u[0:-1]
      self.v_ym[1:] = self.v_ym[0:-1]
      self.v_xhat[1:] = self.v_xhat[0:-1]
      # Store current values
      self.v_u[0] = u[1] # Previous? TODO
      self.v_ym[0] = ym
      self.v_xhat[0] = x_hat
      # Update plot
      t_neg = self.v_t - self.v_t[0]
      self.plotter.update_interactive(t_neg, v_u=self.v_u, v_ym=self.v_ym, 
        v_xhat=self.v_xhat)
      # self.animator.update(v_t[0:i+1], v_x[0:i+1])
      self.plotter.axs[2,0].set_xlim(-self.t_window,0)

if __name__ == "__main__":
  
  # Parameters for simulation -- TODO: Update these
  ball = CMGBall(ra=np.array([0.02, 0, 0]))
  
  # x0 = np.zeros(11)
  # x0[0] = 1 # Real part of Q starts at 1
  # print(ball.eom(x0, 0.01))
  # quit()
  
  # Set up observer & plotter
  obsp = ObsPlot(ball)
  # Create shared queue
  queue_u = Queue()
  # Create & start server thread
  thread_srv = Thread(target=run_server, args =(obsp, queue_u))
  thread_srv.start()
  
  
  ## Set up GUI
  gui = GUI(queue_u, obsp.plotter.fig)
  gui.mainloop()
  