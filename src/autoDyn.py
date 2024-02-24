""" Implementation of Lagrangian mechanics numerically with Algorithmic 
differentiation instead of symbolic
"""

from typing import Callable
from jax import jacfwd, hessian
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfrx
import matplotlib.pyplot as plt

class AutoELCnstr(eqx.Module):
    """ Automatically implement the Euler-Lagrange equation for the given 
        Lagrangian function L
    This version supports constraints of the form: a @ qd + b = 0
    
    d/dt (dL/d{qd}) = d/d{qd} (dL/d{qd}) * qdd + d/d{q} (dL/d{qd}) * qd
        --> M = d/d{qd} (dL/d{qd})      Mass matrix
        --> C = d/d{q} (dL/d{qd})       Coriolis matrix
    M@qdd + C@qd + fk = Qi
        fk = dL/d{q}
    """

    L: Callable
    a: Callable
    b: Callable
    M: Callable
    C: Callable
    fk: Callable
    # Partial derivatives
    pdadt: Callable
    pdadq: Callable
    pdbdt: Callable
    pdbdq: Callable
    pddLdqdt: Callable
    # Whether of not the system is autonomous (time-invarient)
    autonomous: bool

    def __init__(self, L, a, b, autonomous=True):
        """ 
        L: Lagrangian L = T-V
            L = L(q, qd)
        a: Constraint Jacobian
            a = a(t, q)
        b: Remainder of constraint function
            b = b(t, q)
        Constraints are defined such that a @ qd + b = 0
        autonomous: Whether the dynamics are time-invarient
            if false, L = L(t, q, qd)
        """
        self.L = L
        self.a = a
        self.b = b
        self.autonomous = autonomous
        ix_a = not autonomous # If non-autonomous, shift indices by 1

        # Set up dynamics
        self.M = hessian(L, 1+ix_a)
        # Note: Technically, this C is only part of the Coriolis matrix.
        #   The dL/dq term has qdot terms too in general
        self.C = jacfwd(jacfwd(L, 1+ix_a), 0+ix_a)
        if not self.autonomous:
            # partial (partial L/partial qd) /partial t
            self.pddLdqdt = jacfwd(jacfwd(L, 1+ix_a), 0)
        self.fk = jacfwd(L, 0+ix_a) # dL/dq
        # Partial derivatives of constraint equation terms
        self.pdadt = jacfwd(self.a, 0)
        self.pdadq = jacfwd(self.a, 1)
        self.pdbdt = jacfwd(self.b, 0)
        self.pdbdq = jacfwd(self.b, 1)

    def _dynamics(self, t, x, params):
        # Equation of motion
        Nq = int(len(x)/2)
        q = x[0:Nq]
        qd = x[Nq:]

        if self.autonomous:
            Mi = self.M(q, qd)
            Ci = self.C(q, qd)
            fki = self.fk(q, qd)
            Fi = -Ci@qd + fki
        else:
            Mi = self.M(t, q, qd)
            Ci = self.C(t, q, qd)
            fki = self.fk(t, q, qd)
            Fi = -Ci@qd - self.pddLdqdt(t, q, qd) + fki

        # Constraint equation terms
        ai = self.a(t, q)
        bi = self.b(t, q)
        # Total derivatives of constraint equation terms
        dadt = self.pdadt(t, q) + self.pdadq(t, q)@qd
        dbdt = self.pdbdt(t, q) + self.pdbdq(t, q)@qd

        #print(90, Mi.shape, ai.shape)
        Nlam = ai.shape[0]
        # Set up matrix equation to solve for qdd by the "augmented method"
        #   Dy = E
        #   y = {qdd, lambda}
        D = jnp.block([
            [Mi, -ai.T],
            [-ai, jnp.zeros((Nlam, Nlam))]
        ])
        lmbd_eqs = dadt@qd + dbdt
        E = jnp.concatenate([Fi, lmbd_eqs])
        #print(101)
        #print(D)
        #print(E)
        yi = jnp.linalg.solve(D, E).flatten()
        qdd = yi[0:Nq]

        return jnp.hstack([qd, qdd])

    def __call__(self, t, x, params):
        # Return dx/dt
        return self._dynamics(t, x, params)

class AutoEL(eqx.Module):
    """ Automatically implement the Euler-Lagrange equation for the given 
        Lagrangian function L
    
    d/dt (dL/d{qd}) = d/d{qd} (dL/d{qd}) * qdd + d/d{q} (dL/d{qd}) * qd
        --> M = d/d{qd} (dL/d{qd})      Mass matrix
        --> C = d/d{q} (dL/d{qd})       Coriolis matrix
    M@qdd + C@qd + fk = Qi
        fk = dL/d{q}
    """

    L: Callable
    M: Callable
    C: Callable
    fk: Callable

    def __init__(self, L):
        # Assumes L = L(q, qd)
        self.L = L

        # Set up dynamics
        self.M = hessian(L, 1)
        self.C = jacfwd(jacfwd(L, 1), 0)
        self.fk = jacfwd(L, 0)

    def _dynamics(self, t, x, params):
        # Equation of motion
        Nq = int(len(x)/2)
        q = x[0:Nq]
        qd = x[Nq:]

        Mi = self.M(q, qd)
        Ci = self.C(q, qd)
        fki = self.fk(q, qd)

        qdd = jnp.linalg.solve(Mi, -Ci@qd + fki).flatten()

        return jnp.hstack([qd, qdd])

    def __call__(self, t, x, params):
        # Return dx/dt
        return self._dynamics(t, x, params)


def simulate(mod, ts, x0, tol=1e-5, max_steps=4096):
    """ Simulate the given Equinox module object over the given vector ts, 
        with given initial condition x0
    """

    dt0 = 0.01
    sol = dfrx.diffeqsolve(
        terms=dfrx.ODETerm(mod),
        solver=dfrx.Dopri5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=dt0,
        y0=x0,
        saveat=dfrx.SaveAt(ts=ts),
        stepsize_controller=dfrx.PIDController(rtol=tol, atol=tol),
        max_steps=max_steps
    )
    return sol

def plot_states(sol, state_lbls, title=None, save_pname=None, show=True, 
        plot_trike_energy=None, T_fun=None):
    """ Plots the solution of all state variables
    """

    N_states = len(state_lbls)
    if plot_trike_energy is not None:
        fig, axs = plt.subplots(N_states+2, 1, sharex=True)
    elif T_fun is not None:
        fig, axs = plt.subplots(N_states+1, 1, sharex=True)
    else:
        fig, axs = plt.subplots(N_states, 1, sharex=True)
    for ii in range(N_states):
        axs[ii].plot(sol.ts, sol.ys[:, ii])
        axs[ii].set_ylabel(state_lbls[ii])
    axs[-1].set_xlabel("time")
    if T_fun is not None:
        T = T_fun(sol)
        axs[N_states].plot(sol.ts, T)
        axs[N_states].set_ylabel(r"T")
    elif plot_trike_energy is not None:
        trike = plot_trike_energy
        m = trike.m
        Izz = trike.Izz
        Lm = trike.Lm
        # Speed
        speed = jnp.linalg.norm(sol.ys[:,3:5], axis=1)
        axs[N_states].plot(sol.ts, speed)
        axs[N_states].set_ylabel(r"speed")
        # Kinetic Energy
        x, y, th, xd, yd, thd = sol.ys.T
        rdm2 = (xd+Lm*jnp.sin(th)*thd)**2 + (yd-Lm*jnp.cos(th)*thd)**2
        T = m/2*rdm2 + Izz/2*thd**2
        axs[N_states+1].plot(sol.ts, T)
        axs[N_states+1].set_ylabel(r"T")
    if title is not None:
        fig.suptitle(title)

    if save_pname is not None:
        #Path(fig_name).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_pname)
        print(f"Figure saved to: {save_pname}")
    elif show:
        plt.show()

    return fig, axs


if __name__ == "__main__":
    def L(q, qd):
        # Example Lagrangian T-V
        r, th = q
        rd, thd = qd
        return m/2 * (rd**2 + r**2*thd**2) - k/2 * r**2

