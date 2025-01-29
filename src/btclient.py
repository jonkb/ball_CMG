""" Bluetooth client for communication with robot

TODO: Switch over to Bluetooth & async framework

Premise:
* The ESP32 is a Bluetooth server, and it publishes two "characteristics"
  * ACCEL: The robot posts its accel measurements here
  * SPEED: This one is writable, so the laptop can send motor commands
* Main loop runs on laptop, updating simulation & plots at a constant dt
* When a notification comes with new ACCEL measurements, update the observer.
* When the user changes the speeds via the GUI, write to SPEED

Message passing between threads:
* ym queue: server posts messages to a queue that the Observer is listening to
(OLD)
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

from queue import Queue
from threading import Thread
import numpy as np

from CMGBall import CMGBall
from GUI import GUI, ObsPlot




# TODO: Copy Bluetooth code & replace HTTP server with Bluetooth client




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
  thread_srv = Thread(target=run_server, args=(obsp, queue_u))
  thread_srv.start()
  
  ## Set up GUI
  gui = GUI(queue_u, obsp.plotter.fig)
  gui.mainloop()
