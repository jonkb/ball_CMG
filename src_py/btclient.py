""" Bluetooth client for communication with robot

Premise:
* The ESP32 is a Bluetooth server, and it publishes two "characteristics"
  * ACCEL: The robot posts its accel measurements here
  * SPEED: This one is writable, so the laptop can send motor commands
* Main loop runs on laptop, updating simulation & plots at a constant dt
* When a notification comes with new ACCEL measurements, update the observer.
* When the user changes the speeds via the GUI, write to SPEED

TODO: Update this documentation

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

# Standard imports
from queue import Queue
from threading import Thread
import asyncio  # Should / could I move the treading things to asyncio?
import numpy as np
from bleak import BleakClient

# Custom imports
from CMGBall import CMGBall
from GUI import GUI, ObsPlot


## Global variables (TMP)
# Most recent accelerometer measurement
accel = np.zeros(3)

## Constants
# UUIDs for the BLE service and characteristics (must match ESP32's UUIDs)
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
ACCEL_CHARACTERISTIC_UUID = "abcdef01-1234-5678-1234-56789abcdef0"
SPEED_CHARACTERISTIC_UUID = "abcdef02-1234-5678-1234-56789abcdef0"

# Address of the ESP32
#   Found with esp32_testing/getbtaddress
ESP32_ADDRESS = "7c:9e:bd:37:e1:8a"  # Replace with your ESP32's MAC address


async def send_u(client, u):
  """ Send motor commands to the ESP32
  """
  omega_speed, alpha_speed = u
  print(f"New motor speeds: [{omega_speed:.3f}, {alpha_speed:.3f}]")
  # Send updated motor speeds as a byte array
  speeds_bytes = np.array(u, dtype=np.float32).tobytes()
  # print(f"New motor speed bytes: {list(speeds_bytes)}")
  await client.write_gatt_char(SPEED_CHARACTERISTIC_UUID, speeds_bytes)
  print(f"Sent motor speed bytes: {list(speeds_bytes)}")

async def accel_handler(sender, data):
  """ Callback for handling accel update notifications.
  """
  #print(list(data))
  accel[:] = np.frombuffer(data, dtype=np.float32)
  # print(f"New accel values: {accel}")

  # TODO: Pass this info to the observer somehow
  # obs.update(ym, u)
  # Send ym to the Observer via queue_ym
  # queue_ym.put(ym)


async def main(queue_u, obsp):
  # Set up Bluetooth connection
  print(f"Searching for ESP32 at {ESP32_ADDRESS}")
  async with BleakClient(ESP32_ADDRESS) as client:
    print("Connected to ESP32")

    # Enable notifications on the sensor characteristic
    await client.start_notify(ACCEL_CHARACTERISTIC_UUID, accel_handler)

    # # Blocking u loop (for testing)
    # Nmain = 100
    # Tmain = 5
    # # Main control loop
    # for i in range(Nmain):
    #   # Send updated motor speeds as a byte array
    #   omega_speed = np.random.rand()
    #   alpha_speed = np.random.rand()
    #   print(f"New motor speeds: [{omega_speed:.3f}, {alpha_speed:.3f}]")
    #   #speeds_bytes = struct.pack('2f', omega_speed, alpha_speed)
    #   speeds_bytes = np.array([omega_speed, alpha_speed], 
    #           dtype=np.float32).tobytes()
    #   await client.write_gatt_char(SPEED_CHARACTERISTIC_UUID, speeds_bytes)
    #   print(f"Sent updated motor speeds: {list(speeds_bytes)}")
    #   await asyncio.sleep(Tmain)

    # Create a task to handle sending motor commands
    async def listen_u():
      while True:
        u = await queue_u.get()
        await send_u(client, u)

        # Bypass send_u for testing
        # omega_speed, alpha_speed = u
        # print(f"New motor speeds: [{omega_speed:.3f}, {alpha_speed:.3f}]")
        # # Send updated motor speeds as a byte array
        # speeds_bytes = np.array(u, dtype=np.float32).tobytes()
        # await client.write_gatt_char(SPEED_CHARACTERISTIC_UUID, speeds_bytes)
        # print(f"Sent updated motor speeds: {list(speeds_bytes)}")

    # Start motor command listener task
    listen_task = asyncio.create_task(listen_u())

    # # OLD - Set up GUI & run tkinter main loop
    # gui = GUI(queue_u, obsp.plotter.fig)
    # gui.mainloop()

    # Function to run the Tkinter main loop in a separate thread
    def run_gui(asyncloop):
      gui = GUI(queue_u, obsp.plotter.fig, asyncloop)
      gui.mainloop()

    # Start the Tkinter main loop in a separate thread
    asyncloop = asyncio.get_event_loop()
    gui_thread = Thread(target=run_gui, args=(asyncloop,))
    gui_thread.start()

    # Keep the asyncio event loop running
    while gui_thread.is_alive():
      await asyncio.sleep(1)
    
    # Wait for the GUI thread to finish, if somehow it's still running
    gui_thread.join()

    # AFTER GUI CLOSES

    # Stop notifications
    await client.stop_notify(ACCEL_CHARACTERISTIC_UUID)
    print("Stopped notifications")

    # Cancel the send task
    listen_task.cancel()
    try:
      await listen_task
    except asyncio.CancelledError:
      pass


if __name__ == "__main__":
  
  # Parameters for simulation -- TODO: Update these
  ball = CMGBall(ra=np.array([0.02, 0, 0]))
  
  # Set up observer & plotter
  obsp = ObsPlot(ball)

  # Create shared queue for motor commands
  queue_u = asyncio.Queue()

  # Run the main event loop
  asyncio.run(main(queue_u, obsp))

