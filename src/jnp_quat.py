""" Implement quaternions with jax.numpy
"""

import jax.numpy as jnp

class Quat:
    """ Quaternion
    The attribute Q holds the 4 components of the quaternion
    Q = [a, b, c, d]
    Represents the quaternion Q = a + bi + cj + dk
    """

    def __init__(self, Q):
        self.Q = Q

    def __mul__(self, Q2):
        """ Quaternion multiplication (Hamilton product)

        Q.__mul__(Q2) returns Q * Q2
        """
        
        a1, b1, c1, d1 = self.Q
        a2, b2, c2, d2 = Q2.Q
        Q3 = jnp.array([
            a1*a2 - b1*b2 - c1*c2 - d1*d2,
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2
        ])
        return Quat(Q3)

    def conj(self):
        """ Conjugate of self
        """
        a, b, c, d = self.Q
        Qc = jnp.array([a, -b, -c, -d])
        return Quat(Qc)

    def flat(self):
        """ 'Flat' operator: return vector part of self as a 3-vector
        """
        a, b, c, d = self.Q
        return jnp.array([b, c, d])
    
    @classmethod
    def sharp(cls, v):
        """ 'Sharp' operator: return quaternion for which the vector part
        is the given 3-vector v
        """
        b, c, d = v
        return cls(jnp.array([0, b, c, d]))
    
    @classmethod
    def eye(cls):
        """ Generate an identity quaternion
        """
        Q = jnp.array([1, 0, 0, 0])
        return cls(Q)

    @classmethod
    def from_axisangle(cls, angle, axis):
        """ Convert axis-angle representation to Quat
        """
        x, y, z = axis
        Q = jnp.array([
            jnp.cos(angle/2),
            x*jnp.sin(angle/2),
            y*jnp.sin(angle/2),
            z*jnp.sin(angle/2),
        ])
        return cls(Q)

if __name__ == "__main__":
    # Testing
    q = Quat.from_axisangle(jnp.pi/4, [0,0,1])
    print(f"q = {q.Q}")
    e0 = jnp.array([1, 0, 0])

    # Using q to rotate e0
    #  Euler-Rodrigues formula
    e0_rot = (q * Quat.sharp(e0) * q.conj()).flat()
    print("q * e0 * q.conj = ")
    print(e0_rot)


    # qd = 1/2 * q * sharp(omega)
    # omega = flat(2*q.conj()*qdot)
