""" Implementation of Lagrangian mechanics numerically with Algorithmic 
differentiation instead of symbolic

TODO: If possible, try implementing automatic I-O feedback linearization for 
    nonlinear dynamics
"""

from functools import partial
from typing import Callable
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfrx
import sympy as sym
import matplotlib.pyplot as plt

class AutoELSym(eqx.Module):
    """ Automatically derive the Euler-Lagrange equations of motion
    for the given Lagrangian function L
    Uses SymPy to derive EOMs

    Doesn't seem to work with diffrax diffeqsolve
    """

    Nq: int
    L: sym.Expr
    t: sym.Symbol
    # The following 3 are Nx1 column vectors
    q: sym.Matrix
    qd: sym.Matrix
    qdd: sym.Matrix
    Qnc: sym.Matrix
    # Lambdified EOM
    EOM: Callable

    def __init__(self, L, t, q, Qnc=None):
        """
        Qnc (Nx1 sym.Matrix): non-conservative forces
        """
        self.L = L
        self.t = t
        self.q = sym.Matrix(q)
        self.Nq = len(self.q)
        if Qnc is None:
            Qnc = sym.zeros(self.Nq, 1)
        self.Qnc = Qnc
        self.qd = sym.diff(self.q, t)
        self.qdd = self._derive_EOM()
        self.EOM = sym.lambdify([self.q, self.qd], self.qdd, "jax")

    def _derive_EOM(self):

        M = sym.hessian(self.L, self.qd)

        # Convert to Matrix for Matrix.jacobian
        L_mat = sym.Matrix([self.L])
        C = L_mat.jacobian(self.qd).jacobian(self.q)
        fk = L_mat.jacobian(self.q).T

        # Solve for \ddot{q}
        F = -C@self.qd + fk
        if self.Qnc is not None:
            F += self.Qnc
        qdd = M.LUsolve(F)
        return qdd
    
    def __call__(self, t, x, params):
        """ State space equations
        x = [q, qd]
        returns xdot = [qd, qdd]
        """

        q = x[0:self.Nq]
        qd = x[self.Nq:]
        qdd = self.EOM(q, qd).flatten()

        xd = jnp.concatenate([qd, qdd])
        return xd

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
    Qnc: Callable
    # Partial derivatives
    pdadt: Callable
    pdadq: Callable
    pdbdt: Callable
    pdbdq: Callable
    pddLdqdt: Callable
    # Whether of not the system is autonomous (time-invarient)
    autonomous: bool

    def __init__(self, L, a, b, autonomous=True, Qnc=None):
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
        self.Qnc = Qnc
        ix_a = not autonomous # If non-autonomous, shift indices by 1

        # Set up dynamics
        self.M = jax.hessian(L, 1+ix_a)
        # Note: Technically, this C is only part of the Coriolis matrix.
        #   The dL/dq term has qdot terms too in general
        self.C = jax.jacfwd(jax.jacfwd(L, 1+ix_a), 0+ix_a)
        if self.autonomous:
            self.pddLdqdt = lambda t, q, qd: 0.0
        if not self.autonomous:
            # partial (partial L/partial qd) /partial t
            self.pddLdqdt = jax.jacfwd(jax.jacfwd(L, 1+ix_a), 0)
        self.fk = jax.jacfwd(L, 0+ix_a) # dL/dq
        # Partial derivatives of constraint equation terms
        self.pdadt = jax.jacfwd(self.a, 0)
        self.pdadq = jax.jacfwd(self.a, 1)
        self.pdbdt = jax.jacfwd(self.b, 0)
        self.pdbdq = jax.jacfwd(self.b, 1)

    def _dynamics(self, t, x, params):
        # Equation of motion
        Nq = int(len(x)/2)
        q = x[0:Nq]
        qd = x[Nq:]

        Mi = self.M(q, qd)
        Ci = self.C(q, qd)
        fki = self.fk(q, qd)
        Fi = -Ci@qd + fki + (0.0 if self.Qnc is None else self.Qnc(t, x))

        #if self.autonomous:
        #    Mi = self.M(q, qd)
        #    Ci = self.C(q, qd)
        #    fki = self.fk(q, qd)
        #    Fi = -Ci@qd + fki
        #else:
        #    Mi = self.M(t, q, qd)
        #    Ci = self.C(t, q, qd)
        #    fki = self.fk(t, q, qd)
        #    Fi = -Ci@qd - self.pddLdqdt(t, q, qd) + fki

        # Constraint equation terms
        ai = self.a(t, q)
        bi = self.b(t, q) # NOTE: b itself isn't important. dbdt is
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

    #@partial(jax.jit, static_argnums=0)
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
    Qnc: Callable

    def __init__(self, L, Qnc=None):
        # Assumes L = L(q, qd)
        self.L = L
        self.Qnc = Qnc

        # Set up dynamics
        self.M = jax.hessian(L, 1)
        self.C = jax.jacfwd(jax.jacfwd(L, 1), 0)
        self.fk = jax.jacfwd(L, 0)

    def _dynamics(self, t, x, params):
        # Equation of motion
        Nq = int(len(x)/2)
        q = x[0:Nq]
        qd = x[Nq:]

        Mi = self.M(q, qd)
        Ci = self.C(q, qd)
        fki = self.fk(q, qd)

        # Solve for \ddot{q}
        Fi = -Ci@qd + fki + (0.0 if self.Qnc is None else self.Qnc(t, x))
        qdd = jnp.linalg.solve(Mi, Fi).flatten()

        return jnp.hstack([qd, qdd])

    #@jax.jit
    #@partial(jax.jit, static_argnums=0)
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

def sim_Euler(mod, ts, x0):
    """ Simulate the given Equinox module object over the given vector ts, 
        with given initial condition x0
    Runs Euler instead of adaptive integration
    For debugging
    """

    # Jank solution container object
    sol = lambda x: 0.0
    sol.ts = ts

    Nt = ts.size
    xs = jnp.zeros([Nt, x0.size])
    xs = xs.at[0,:].set(x0)
    for i in range(Nt-1):
        ti = ts[i]
        dt = ts[i+1] - ts[i]
        xi = xs[i,:]
        xdot = mod(ti, xi, None)
        x_next = xi + xdot*dt
        xs = xs.at[i+1,:].set(x_next)
    
    sol.ys = xs
    return sol

def plot_states(sol, state_lbls, title=None, save_pname=None, show=True, 
        more_lines = []):
    """ Plots the solution of all state variables

    more_lines (list of tuples): [(lbl, yfun), ...]
        lbl (str): y axis label
        yfun (callable): ys = yfun(sol)
        Additional functions to plot
    """

    # Constant parameters
    figsize = (8,8)
    N_states = len(state_lbls)
    N_plots = N_states + len(more_lines)
    fig, axs = plt.subplots(N_plots, 1, sharex=True, figsize=figsize)
    axs[-1].set_xlabel("time")
    for ix in range(N_states):
        axs[ix].plot(sol.ts, sol.ys[:, ix], color="tab:blue")
        axs[ix].set_ylabel(state_lbls[ix])
        axs[ix].grid()
    for ix, (lbl, yfun) in enumerate(more_lines):
        ys = yfun(sol)
        axs[N_states+ix].plot(sol.ts, ys, color="tab:green")
        axs[N_states+ix].set_ylabel(lbl)
        axs[N_states+ix].grid()
    if title is not None:
        fig.suptitle(title)

    if save_pname is not None:
        #Path(fig_name).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_pname)
        print(f"Figure saved to: {save_pname}")
    elif show:
        plt.show()

    return fig, axs



def plot_states_old(sol, state_lbls, title=None, save_pname=None, show=True, 
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
    m = 1
    k = 1

    # Test AutoELSym
    t = sym.symbols('t')
    r = sym.Function('r')(t)
    th = sym.Function('th')(t)
    rd = sym.diff(r, t)
    thd = sym.diff(th, t)
    q = (r, th)
    L = m/2 * (rd**2 + r**2*thd**2) - k/2 * r**2
    print(f"L: {L}")
    mod = AutoELSym(L, t, q)
    print(f"EOM: ")
    sym.pprint(mod.qdd)
    # Test
    x0 = jnp.array([1.0, 0.0, 0.0, 2.0])
    xd = mod(0.0, x0)
    print(306, xd)

    def L(q, qd):
        # Example Lagrangian T-V
        r, th = q
        rd, thd = qd
        return m/2 * (rd**2 + r**2*thd**2) - k/2 * r**2
