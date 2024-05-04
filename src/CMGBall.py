""" CMGBall class to hold parameters of the robot

Generalized coordinates q=[rx, ry, eta, eps_x, eps_y, eps_z]
State vector x=[q, qd]

TODO: Study gyroscopic procession

TODO: Include a-frame mass & I (innards besides gyro)

TODO: Consider re-formulating such that alpha is a coordinate with an applied
    non-conservative force Q. Advantage --> System becomes autonomous again...
    I think. However, if I make Omega_g variable, it'll be non-autonomous again,
    unless I include gyro motor dynamics too.
"""

from jax import jacfwd
import jax.numpy as jnp
from jnp_quat import Quat
from autoDyn import AutoELCnstr, simulate#, plot_states
from plotting import animate_ball, plot_states

class CMGBall:
    """ Set of generalized coordinates q: [rx, ry, Q1, Q2, Q3, Q4, alpha]
    """
    # Constants
    Nx = 11 # Length of state vector # OLD
    # Number of gen coords
    Nq = 7
    Nu = 1  # Number of inputs
    Nym = 3 # Number of measurements
    g = 9.8

    def __init__(self, Is=6e-4, Ig1=2.6e-5, Ig2=1.7e-5, m=0.400, Rs=0.08, 
            Omega_g=2000.0, km=10.0, ra=jnp.array([0,0,0]), tau=None):
        """ All parameters are floats unless otherwise noted:
        Is: Moment of inertia for the sphere
            Translates to [Is]=diag([Is, Is, Is]) because of symmetry
        Ig1, Ig2: Moments of inertia for the Gyroscope
            Translates to [Ig]=diag([Ig1, Ig2, Ig2])
            Ig1 is the moment of inertia about the axis of rotation
        m: Mass of the sphere + mass of the gyro
        Rs: Radius of the sphere
        Omega_g: Constant angular velocity of the gyroscope
            Presumably this is maintained by a motor spinning at a constant 
            speed
        km: Motor constant, mapping pwm to torque. This is also equal to
            max torque
            The physical system maps pwm to voltage to alpha motor torque 
            This km variable attempts to represent that conversion.
            TODO: Maybe represent that this mapping is more complex & nonlinear,
            including the effects of the gearbox.
        ra: Position vector of accelerometer, relative to the center of the 
            sphere, expressed in the accel-fixed "a"-frame.
        tau (Callable): Input torque as a function of time. Should be JAX 
            traceable
        """
        # Constants
        self.Is = Is * jnp.eye(3)
        self.Ig = jnp.diag(jnp.array([Ig1, Ig2, Ig2]))
        self.m = m
        self.Rs = Rs
        self.Omega_g = Omega_g
        self.km = km
        self.tau_max = self.km
        self.ra = ra
        if tau is None:
            self.tau = lambda t: 0.0 * t
        else:
            self.tau = tau

        # Non-conservative force vector (same shape as q)
        Qnc = lambda t,x: jnp.concatenate([jnp.zeros(self.Nq-1), 
                jnp.array([self.tau(t)])])
        # Equinox module for this CMGBall
        self.mod = AutoELCnstr(self.L, self.a, self.b, Qnc=Qnc)

    def __str__(self):
        s = "CMG Ball Object\n"
        s += f"\tIs={self.Is}\n"
        s += f"\tIg={self.Ig}\n"
        #s += f"\tIg1={self.Ig1}\n"
        #s += f"\tIg2=Ig3={self.Ig2}\n"
        s += f"\tm={self.m}\n"
        s += f"\tRs={self.Rs}\n"
        s += f"\tOmega_g={self.Omega_g}\n"
        s += f"\ttau_max={self.tau_max}\n"
        return s

    def aa2pwm(self, alphadd):
        """
        Map desired alphadd to the corresponding pwm
        TODO: Switch aa for tau
        
        Returns
        -------
        pwm
        pwmsat (btw -1 & 1)
        """
        pwm = alphadd / self.km
        pwmsat = jnp.min(1, jnp.max(-1, pwm))
        return pwm, pwmsat
    
    def pwm2aa(self, pwm):
        """
        Map pwm to alphadd (saturating pwm @ +-100%)
        
        Returns
        -------
        alphadd
        """
        pwmsat = jnp.min(1, jnp.max(-1, pwm))
        alphadd = pwmsat * self.km
        return alphadd

    def _T(self, q, qd):
        """ Kinetic Energy
        """
        rx, ry, eta, eps_x, eps_y, eps_z, alpha = q
        rxd, ryd, etad, eps_xd, eps_yd, eps_zd, alphad = qd

        # Qs: passive rotation from s-frame to 0-frame as follows:
        #   v^0 = Qs * v^s * Qs.conj()
        Qs = Quat(jnp.array([eta, eps_x, eps_y, eps_z]))
        Qsd = Quat(jnp.array([etad, eps_xd, eps_yd, eps_zd]))
        # Qas rotates vectors from a-frame to s-frame by
        #   v^s = Qas * v^a * Qas.conj()
        Qas = Quat.from_axisangle(alpha, jnp.array([0,0,1]))

        # Get omegas from Quaternion
        #   See Putkaradze D.21, solved for omega
        omega_s__s = 2*(Qs.conj() * Qsd).flat()
        omega_g__s = omega_s__s + jnp.array([
            self.Omega_g * jnp.cos(alpha),
            self.Omega_g * jnp.sin(alpha),
            alphad
        ])
        # Rotate omega_g__s into the a-frame
        #   NOTE: This is using Qas^{-1}
        omega_g__a = (Qas.conj() * Quat.sharp(omega_g__s) * Qas).flat()

        # Calculate the three contributions to kinetic energy
        T_trn = self.m/2 * (rxd**2 + ryd**2)
        T_rot_S = omega_s__s @ self.Is @ omega_s__s
        T_rot_G = omega_g__a @ self.Ig @ omega_g__a

        return T_trn + T_rot_S + T_rot_G

    def _V(self, q):
        """ Potential Energy
        """
        return 0.0

    def L(self, q, qd):
        """ Lagrangian L = T - V
        """
        T = self._T(q, qd)
        V = self._V(q)
        return T - V
    
    def a(self, t, q):
        """ Constraint Jacobian
        a@qd + b = 0
        """

        R = self.Rs
        rx, ry, eta, ex, ey, ez, alpha = q

        # Calculate the no-slip constraint
        # Partial of  omega_s__0 WRT Q (which appears linearly)
        #   See dynamics.py
        dwxdQ = jnp.array([
            -2*eta**2*ex - 2*ex**3 - 2*ex*ey**2 - 2*ex*ez**2,
            2*eta**3 + 2*eta*ex**2 + 2*eta*ey**2 + 2*eta*ez**2,
            -2*eta**2*ez - 2*ex**2*ez - 2*ey**2*ez - 2*ez**3,
            2*eta**2*ey + 2*ex**2*ey + 2*ey**3 + 2*ey*ez**2
        ])
        dwydQ = jnp.array([
            -2*eta**2*ey - 2*ex**2*ey - 2*ey**3 - 2*ey*ez**2,
            2*eta**2*ez + 2*ex**2*ez + 2*ey**2*ez + 2*ez**3,
            2*eta**3 + 2*eta*ex**2 + 2*eta*ey**2 + 2*eta*ez**2,
            -2*eta**2*ex - 2*ex**3 - 2*ex*ey**2 - 2*ex*ez**2
        ])
        
        return jnp.array([
            # Quaternion norm constraint
            [0, 0, 2*eta, 2*ex, 2*ey, 2*ez, 0],
            # Non-slip constraint (rxd)
            [1, 0, -R*dwydQ[0], -R*dwydQ[1], -R*dwydQ[2], -R*dwydQ[3], 0],
            # Non-slip constraint (ryd)
            [0, 1, R*dwxdQ[0], R*dwxdQ[1], R*dwxdQ[2], R*dwxdQ[3], 0]
        ])

    def b(self, t, q):
        """ Rest of constraint
        a@qd + b = 0
        """

        # None of the constraints in this case have a 'b' term
        return jnp.array([
            0, 0, 0
        ])

class CMGBall_nonauton:
    # Constants
    Nx = 11 # Length of state vector
    Nu = 1  # Number of inputs
    Nym = 3 # Number of measurements
    g = 9.8

    def __init__(self, Is=6e-4, Ig1=2.6e-5, Ig2=1.7e-5, m=0.400, Rs=0.08, 
            Omega_g=2000.0, km=10.0, ra=jnp.array([0,0,0]), alpha=None):
        """ All parameters are floats unless otherwise noted:
        Is: Moment of inertia for the sphere
            Translates to [Is]=diag([Is, Is, Is]) because of symmetry
        Ig1, Ig2: Moments of inertia for the Gyroscope
            Translates to [Ig]=diag([Ig1, Ig2, Ig2])
            Ig1 is the moment of inertia about the axis of rotation
        m: Mass of the sphere + mass of the gyro
        Rs: Radius of the sphere
        Omega_g: Constant angular velocity of the gyroscope
            Presumably this is maintained by a motor spinning at a constant 
            speed
        km: Motor constant, mapping pwm to alphadd. This is also equal to
            alphadd_max: Max alpha-double-dot (rad/s^2)
            The physical system maps pwm to voltage to alpha motor torque and 
            from torque to acceleration.
            This km variable attempts to represent that conversion.
            TODO: Maybe represent that this mapping is more complex & nonlinear,
            including the effects of the gearbox.
        ra: Position vector of accelerometer, relative to the center of the 
            sphere, expressed in the accel-fixed "a"-frame.
        alpha (Callable): Input angle as a function of time. Should be JAX 
            traceable
        """
        # Constants
        self.Is = Is * jnp.eye(3)
        self.Ig = jnp.diag(jnp.array([Ig1, Ig2, Ig2]))
        self.m = m
        self.Rs = Rs
        self.Omega_g = Omega_g
        self.km = km
        self.alphadd_max = self.km
        self.ra = ra
        if alpha is None:
            self.alpha = lambda t: 0.0 * t
        else:
            self.alpha = alpha
        self.alphad = jacfwd(self.alpha)

        # Equinox module for this CMGBall
        self.mod = AutoELCnstr(self.L, self.a, self.b, autonomous=False)

    def __str__(self):
        s = "CMG Ball Object\n"
        s += f"\tIs={self.Is}\n"
        s += f"\tIg1={self.Ig1}\n"
        s += f"\tIg2=Ig3={self.Ig2}\n"
        s += f"\tm={self.m}\n"
        s += f"\tRs={self.Rs}\n"
        s += f"\tOmega_g={self.Omega_g}\n"
        s += f"\talphadd_max={self.alphadd_max}\n"
        return s

    def aa2pwm(self, alphadd):
        """
        Map desired alphadd to the corresponding pwm
        
        Returns
        -------
        pwm
        pwmsat (btw -1 & 1)
        """
        pwm = alphadd / self.km
        pwmsat = jnp.min(1, jnp.max(-1, pwm))
        return pwm, pwmsat
    
    def pwm2aa(self, pwm):
        """
        Map pwm to alphadd (saturating pwm @ +-100%)
        
        Returns
        -------
        alphadd
        """
        pwmsat = jnp.min(1, jnp.max(-1, pwm))
        alphadd = pwmsat * self.km
        return alphadd

    def _T(self, t, q, qd):
        """ Kinetic Energy
        """
        rx, ry, eta, eps_x, eps_y, eps_z = q
        rxd, ryd, etad, eps_xd, eps_yd, eps_zd = qd

        alpha = self.alpha(t)
        alphad = self.alphad(t)

        # Qs: passive rotation from s-frame to 0-frame as follows:
        #   v^0 = Qs * v^s * Qs.conj()
        Qs = Quat(jnp.array([eta, eps_x, eps_y, eps_z]))
        Qsd = Quat(jnp.array([etad, eps_xd, eps_yd, eps_zd]))
        # Qas rotates vectors from a-frame to s-frame by
        #   v^s = Qas * v^a * Qas.conj()
        Qas = Quat.from_axisangle(alpha, jnp.array([0,0,1]))

        # Get omegas from Quaternion
        #   See Putkaradze D.21, solved for omega
        omega_s__s = 2*(Qs.conj() * Qsd).flat()
        omega_g__s = omega_s__s + jnp.array([
            self.Omega_g * jnp.cos(alpha),
            self.Omega_g * jnp.sin(alpha),
            alphad
        ])
        # Rotate omega_g__s into the a-frame
        #   NOTE: This is using Qas^{-1}
        omega_g__a = (Qas.conj() * Quat.sharp(omega_g__s) * Qas).flat()

        # Calculate the three contributions to kinetic energy
        T_trn = self.m/2 * (rxd**2 + ryd**2)
        T_rot_S = omega_s__s @ self.Is @ omega_s__s
        T_rot_G = omega_g__a @ self.Ig @ omega_g__a

        return T_trn + T_rot_S + T_rot_G

    def _V(self, t, q):
        """ Potential Energy
        """
        return 0.0

    def L(self, t, q, qd):
        """ Lagrangian L = T - V
        """
        T = self._T(t, q, qd)
        V = self._V(t, q)
        return T - V
    
    def a(self, t, q):
        """ Constraint Jacobian
        a@qd + b = 0
        """

        R = self.Rs
        rx, ry, eta, ex, ey, ez = q

        # Calculate the no-slip constraint
        # Partial of  omega_s__0 WRT Q (which appears linearly)
        #   See dynamics.py
        dwxdQ = jnp.array([
            -2*eta**2*ex - 2*ex**3 - 2*ex*ey**2 - 2*ex*ez**2,
            2*eta**3 + 2*eta*ex**2 + 2*eta*ey**2 + 2*eta*ez**2,
            -2*eta**2*ez - 2*ex**2*ez - 2*ey**2*ez - 2*ez**3,
            2*eta**2*ey + 2*ex**2*ey + 2*ey**3 + 2*ey*ez**2
        ])
        dwydQ = jnp.array([
            -2*eta**2*ey - 2*ex**2*ey - 2*ey**3 - 2*ey*ez**2,
            2*eta**2*ez + 2*ex**2*ez + 2*ey**2*ez + 2*ez**3,
            2*eta**3 + 2*eta*ex**2 + 2*eta*ey**2 + 2*eta*ez**2,
            -2*eta**2*ex - 2*ex**3 - 2*ex*ey**2 - 2*ex*ez**2
        ])
        
        return jnp.array([
            # Quaternion norm constraint
            [0, 0, 2*eta, 2*ex, 2*ey, 2*ez],
            # Non-slip constraint (rxd)
            [1, 0, -R*dwydQ[0], -R*dwydQ[1], -R*dwydQ[2], -R*dwydQ[3]],
            # Non-slip constraint (ryd)
            [0, 1, R*dwxdQ[0], R*dwxdQ[1], R*dwxdQ[2], R*dwxdQ[3]]
        ])

    def b(self, t, q):
        """ Rest of constraint
        a@qd + b = 0
        """

        # None of the constraints in this case have a 'b' term
        return jnp.array([
            0, 0, 0
        ])


if __name__ == "__main__":

    #A_in = jnp.pi/180
    A_in = jnp.pi/2000
    #f_in = 1.1
    f_in = 1/0.132 / 4
    #alpha = lambda t: A_in * jnp.sin(2*jnp.pi*t*f_in) * (t<1/f_in)
    #alpha = lambda t: A_in * jnp.sin(2*jnp.pi*t*f_in)
    #alpha = lambda t: A_in * t**2
    #alpha = lambda t: A_in * t
    #alpha = lambda t: 0.0 * t + A_in # CONST
    #t_mid = 0.5
    #alpha = lambda t: A_in * (t * (t<t_mid) + (t_mid-(t-t_mid))*(t>=t_mid))
    tau = lambda t: A_in
    #tau = lambda t: A_in * t**2
    ball = CMGBall(tau=tau)

    t1 = 6
    slow_factor = 4
    anim_pname = None #"out.mp4"

    # Define initial conditions
    r0 = jnp.array([0.0, 0.0])
    Q0 = Quat.from_axisangle(jnp.pi/2, jnp.array([1, 0, 0]))
    alpha0 = 0.0
    #Q0 = Quat.eye()
    #q0 = jnp.array([0, 0, 1, 0, 0, 0], dtype=jnp.float32)
    q0 = jnp.concatenate([r0, Q0.Q, jnp.array([alpha0])])
    # Some of this assumes Q0 = Quat.eye
    rxd = 0.0
    ryd = 0.0
    wy = rxd / ball.Rs
    wx = -ryd / ball.Rs
    eyd = wy / 2
    exd = wx / 2
    qd0 = jnp.array([rxd, ryd, 0, exd, eyd, 0, 0], dtype=jnp.float32)
    x0 = jnp.concatenate([q0, qd0])
    #xd = ball.mod(0.0, x0, [])
    # TODO: Try a couple of basic cases (orientations & alpha) and see if it
    #   behaves as expected, according to \sum{M} = \dot{H}

    # Simulate
    Nfrm = int(jnp.ceil(t1 * 30)*slow_factor)
    ts = jnp.linspace(0, t1, Nfrm)
    print("Simulating")
    sol = simulate(ball.mod, ts, x0, tol=1e-6, max_steps=2**14)
    #state_lbls = (r"$x$", r"$y$", r"$\eta$", r"$\epsilon_x$", r"$\epsilon_y$", 
    #    r"$\epsilon_z$",
    #    r"$\dot{x}$", r"$\dot{y}$", r"$\dot{\eta}$", r"$\dot{\epsilon}_x$", 
    #    r"$\dot{\epsilon}_y$", r"$\dot{\epsilon}_z$")
    #fig, axs = plot_states(sol, state_lbls, show=False) #, T_fun=T_fun)
    fig, axs = plot_states(sol, ball)
    #fig.show()
    #input("Press any key to continue")
    
    print("Generating animation")
    animate_ball(sol, ball, save_pname=anim_pname, slow_factor=slow_factor)
    if anim_pname is not None:
        # If anim_pname is None, it already has shown
        fig.show()
        input("Press any key to continue")
    print("DONE")

