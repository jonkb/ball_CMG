"""

OBSOLETE: see ball_CMG

Read and plot the sol object from spi.solve_ivp
"""

import matplotlib.pyplot as plt
import numpy as np
import dill

file = open("sol.dill", "rb")
sol = dill.load(file)
file.close()

print(f"dir(sol): {dir(sol)}")

print("t_min, t_max: ", min(sol.t), max(sol.t))

t = np.linspace(0, 2, 200)
x = sol.sol(t)

fig, axs = plt.subplots(2,2, sharex=True)
axs[0,0].plot(t, x[0,:], label="nu")
axs[0,0].plot(t, x[1,:], label="ex")
axs[0,0].plot(t, x[2,:], label="ey")
axs[0,0].plot(t, x[3,:], label="ez")
qnorm = np.linalg.norm(x[0:4,:], axis=0)
axs[0,0].plot(t, qnorm, label="|q|")
axs[0,0].legend()
axs[0,0].set_title("Orientation Q")
axs[0,1].plot(t, x[4,:], label="$\omega_x$")
axs[0,1].plot(t, x[5,:], label="$\omega_y$")
axs[0,1].plot(t, x[6,:], label="$\omega_z$")
axs[0,1].legend()
axs[0,1].set_title("Angular Velocity $\omega$")
axs[1,0].plot(t, x[7,:])
axs[1,0].set_xlabel("Time t")
axs[1,0].set_title("X-Position $r_x$")
axs[1,1].plot(t, x[8,:])
axs[1,1].set_xlabel("Time t")
axs[1,1].set_title("Y-Position $ry$")
fig.show()
input("PRESS ANY KEY")
