""" Misc. functions for plotting
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from jnp_quat import Quat

def plot_states(sol, ball):
    """ Generate plots of the ball state over time
    """
    Nt = sol.ts.size
    Nq = ball.Nq
    fig, axs = plt.subplots(3,2, sharex=True, figsize=(14.0,9.0))

    # Set up subplots
    axs[0,0].set_title(r"Position $r_x$, $r_y$")
    axs[0,1].set_title(r"Velocity $\dot{r}_x$, $\dot{r}_y$")
    axs[1,0].set_title(r"Orientation $Q$")
    axs[1,1].set_title(r"Quaternion rates $\dot{Q}$")
    axs[2,0].set_title(r"Constraint violation $a\dot{q}$")
    axs[2,1].set_title(r"Kinetic Energy $T$")
    axs[2,0].set_xlabel(r"Time $t$")
    axs[2,1].set_xlabel(r"Time $t$")

    # Plot: Position
    axs[0,0].plot(sol.ts, sol.ys[:,0], label=r"$x$")
    axs[0,0].plot(sol.ts, sol.ys[:,1], label=r"$y$")
    #axs[0,0].set_ylabel(r"m")
    axs[0,0].legend()
    
    # Plot: Velocity
    axs[0,1].plot(sol.ts, sol.ys[:,Nq], label=r"$\dot{x}$")
    axs[0,1].plot(sol.ts, sol.ys[:,Nq+1], label=r"$\dot{y}$")
    axs[0,1].legend()

    # Plot: Orientation
    axs[1,0].plot(sol.ts, sol.ys[:,2], label=r"$\eta$")
    axs[1,0].plot(sol.ts, sol.ys[:,3], label=r"$\epsilon_x$")
    axs[1,0].plot(sol.ts, sol.ys[:,4], label=r"$\epsilon_y$")
    axs[1,0].plot(sol.ts, sol.ys[:,5], label=r"$\epsilon_z$")
    axs[1,0].legend()

    # Plot: Quaternion rates
    axs[1,1].plot(sol.ts, sol.ys[:,Nq+2], label=r"$\dot{\eta}$")
    axs[1,1].plot(sol.ts, sol.ys[:,Nq+3], label=r"$\dot{\epsilon}_x$")
    axs[1,1].plot(sol.ts, sol.ys[:,Nq+4], label=r"$\dot{\epsilon}_y$")
    axs[1,1].plot(sol.ts, sol.ys[:,Nq+5], label=r"$\dot{\epsilon}_z$")
    axs[1,1].legend()

    # Plot: Constraint violation
    cviol = jnp.array([ball.a(sol.ts[ix], sol.ys[ix,0:ball.Nq])@sol.ys[ix,ball.Nq:] 
        for ix in range(Nt)])
    axs[2,0].plot(sol.ts, cviol[:,0], label=r"$\|Q\|$")
    axs[2,0].plot(sol.ts, cviol[:,1], label=r"non-slip ($x$)")
    axs[2,0].plot(sol.ts, cviol[:,2], label=r"non-slip ($y$)")
    axs[2,0].legend()

    # Plot: Kinetic energy
    T = jnp.array([ball._T(sol.ys[ix,0:ball.Nq], sol.ys[ix,ball.Nq:])
        for ix in range(Nt)])
    axs[2,1].plot(sol.ts, T, label=r"$T$")
    

    return fig, axs


def plot_coords(ax, Q, r, arrow_len=0.1, Q1=None):
    """ Plot arrows for a coordinate system for the given Quaternion
    ax: pyplot axes
    Q jnp.quat.Quat: Orientation of coordinate system to plot
        This rotation should be such that the following:
        v0 = Q * v1 * Q.conj()
        rotates vectors from the rotated frame back to the global frame.
    Q1 jnp.quat.Quat: If given, a second rotation defining a 4th vector
        to be plotted within the coord system.
        v4__1 = Q1 * [1,0,0] * Q1.conj()
    """

    # Unpack position
    rx, ry = r
    # Unpack Quat.Q
    #eta, ex, ey, ez = Q.Q # Unnecessary
    
    # Rotate the unit vectors according to Q
    #   from frame s to frame 0
    E__0 = jnp.eye(3)
    e0 = (Q * Quat.sharp(E__0[:,0]) * Q.conj()).flat() * arrow_len
    e1 = (Q * Quat.sharp(E__0[:,1]) * Q.conj()).flat() * arrow_len
    e2 = (Q * Quat.sharp(E__0[:,2]) * Q.conj()).flat() * arrow_len

    # Plot x & y components of arrows (project down)
    ah1 = ax.arrow(rx, ry, e0[0], e0[1], color="red")
    ah2 = ax.arrow(rx, ry, e1[0], e1[1], color="green")
    ah3 = ax.arrow(rx, ry, e2[0], e2[1], color="blue")

    # If passed, generate & plot the 4th vector
    if Q1 is not None:
        v4 = (Q * Q1 * Quat.sharp(E__0[:,0]) * Q1.conj() * 
            Q.conj()).flat() * arrow_len
        ah4 = ax.arrow(rx, ry, v4[0], v4[1], color="yellow", width=0.0005)
        return ah1, ah2, ah3, ah4

    return ah1, ah2, ah3


def animate_ball(sol, ball, save_pname=None, slow_factor=1):
    """ Generate an animation of a CMGBall
    """

    # Constant parameters
    Rs = ball.Rs
    figsize = (10,10)
    xmin = jnp.min(sol.ys[:,0])
    xmax = jnp.max(sol.ys[:,0])
    ymin = jnp.min(sol.ys[:,1])
    ymax = jnp.max(sol.ys[:,1])
    xbuf = max(Rs, (xmax-xmin) * 0.05)
    ybuf = max(Rs, (ymax-ymin) * 0.05)
    # Auto-scale fig to fit path
    lims = jnp.array([
        [xmin-xbuf, xmax+xbuf],
        [ymin-ybuf, ymax+ybuf]
    ])
    p_txt = (lims[0,0]+xbuf/2, lims[1,1]-ybuf/2)
    c_bg = (1,1,1,0.7) # Text background color
    fontfam = "monospace" # Font for text
    spadj = (.1,.1,.9,.9) # Subplots_adjust params
    # Unpack gen coords
    q1 = sol.ys[:,0]
    q2 = sol.ys[:,1]
    q3 = sol.ys[:,2]
    q4 = sol.ys[:,3]
    q5 = sol.ys[:,4]
    q6 = sol.ys[:,5]
    q7 = sol.ys[:,6]
    Nfrm = len(q1)
    to_blit = save_pname is None

    fig, ax = plt.subplots(figsize=figsize)
    #fig.subplots_adjust(*spadj)
    #lh1, *_ = ax.plot([], [], "o-", lw=5, color="r")
    #lh2, *_ = ax.plot([], [], "-", lw=3, color="b")
    #lh3, *_ = ax.plot([], [], "-", lw=3, color="b")
    #lh4, *_ = ax.plot([], [], "-", lw=3, color="b")
    Q0 = Quat(jnp.array([q3[0], q4[0], q5[0], q6[0]]))
    r0 = jnp.array([q1[0], q2[0]])
    #alpha0 = ball.alpha(sol.ts[0])
    alpha0 = q7[0]
    Qas = Quat.from_axisangle(alpha0, jnp.array([0,0,1]))
    
    ah1, ah2, ah3, ah4 = plot_coords(ax, Q0, r0, arrow_len=Rs, Q1=Qas)
    fcol = [0, 0, 0, 0.2]
    ecol = [0, 0, 0, 0.8]
    cirh = plt.Circle((0, 0), Rs, fc=fcol, ec=ecol, lw=3)

    ax.set_aspect("equal")
    ax.grid()
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    #ax.set_xticks(jnp.arange(lims[0,0], lims[0,1]+1))
    #ax.set_yticks(jnp.arange(lims[1,0], lims[1,1]+1))

    def _init():
        frm_lbl = f"Frame 0000/{Nfrm}: t=0/{sol.ts[-1]:.3f}"
        tth = ax.text(x=p_txt[0], y=p_txt[1], s=frm_lbl, fontfamily=fontfam,
            backgroundcolor=c_bg)
        #tth = ax.set_title(frm_lbl) # The title cannot be updated with blit on
        ax.add_patch(cirh)
        return (ah1, ah2, ah3, ah4, cirh, tth)

    def _animate(ifrm):
        ri = jnp.array([q1[ifrm], q2[ifrm]])
        if to_blit:
            # Move center
            cirh.center = ri
            cirh1 = cirh
        else:
            ax.cla()
            ax.set_aspect("equal")
            ax.grid()
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
            cirh1 = plt.Circle(ri, Rs, fc=fcol, ec=ecol, lw=3)
            ax.add_patch(cirh1)
        
        ti = sol.ts[ifrm]
        frm_lbl = f"Frame {ifrm:04}/{Nfrm}: t={ti:.3f}/{sol.ts[-1]:.3f}"
        tth = ax.text(x=p_txt[0], y=p_txt[1], s=frm_lbl, fontfamily=fontfam,
            backgroundcolor=c_bg)
        
        # Move coordinate frame arrows
        Qi = Quat(jnp.array([q3[ifrm], q4[ifrm], q5[ifrm], q6[ifrm]]))
        #alphai = ball.alpha(ti)
        alphai = q7[ifrm]
        Qas = Quat.from_axisangle(alphai, jnp.array([0,0,1]))
        ah1, ah2, ah3, ah4 = plot_coords(ax, Qi, ri, arrow_len=Rs, Q1=Qas)
        return (ah1, ah2, ah3, ah4, cirh1, tth)

    ani = FuncAnimation(
        fig,
        _animate,
        init_func=_init,
        frames=Nfrm,
        interval=1,
        blit=to_blit,
    )

    if save_pname is not None:
        fps = Nfrm / sol.ts[-1] / slow_factor
        prog_cb = lambda i, n: print(f'Saving frame {i}/{n}')
        #ani.save(save_pname, writer="pillow", fps=30,
        #    progress_callback=prog_cb)
        ani.save(save_pname, writer="ffmpeg", fps=fps,
            progress_callback=prog_cb)
        print(f"Video saved to {save_pname}")
    else:
        plt.show()


if __name__ == "__main__":
    # Testing plot_coords
    fig, ax = plt.subplots()
    Q = Quat.from_axisangle(10*jnp.pi/180, [1,1,1])
    r = jnp.array([2,3])
    plot_coords(ax, Q, r)
    ax.set_aspect("equal")
    fig.show()
    input("DONE")