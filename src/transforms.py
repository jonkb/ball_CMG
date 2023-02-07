"""
Transforms Module - Contains code for to learn about rotations
and eventually homogenous transforms. 

Empty outline derived from code written by John Morrell. 
"""

import sympy as sp
import numpy as np
from numpy.linalg import norm

## 2D Rotations
def rot2(th, sym=False):
    """
    R = rot2(theta)
    Parameters
        theta: float or int, angle of rotation
        sym (bool): Whether to use sympy
    Returns
        R: 2 x 2 numpy array representing rotation in 2D by theta
    """

    if sym:
        R = sp.Matrix([
            [sp.cos(th), -sp.sin(th)],
            [sp.sin(th), sp.cos(th)]
        ])
        return R
    else:
        R = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th), np.cos(th)]
        ])
        return R

## 3D Transformations
def rotx(th, sym=False):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about x-axis by amount theta
    """
    
    if sym:
        R = sp.Matrix([
            [1, 0, 0],
            [0, sp.cos(th), -sp.sin(th)],
            [0, sp.sin(th), sp.cos(th)]
        ])
        return R
    else:
        R = np.array([
            [1, 0, 0],
            [0, np.cos(th), -np.sin(th)],
            [0, np.sin(th), np.cos(th)]
        ])
        return R

def roty(th, sym=False):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about y-axis by amount theta
    """
    
    if sym:
        R = sp.Matrix([
            [sp.cos(th), 0, sp.sin(th)],
            [0, 1, 0],
            [-sp.sin(th), 0, sp.cos(th)]
        ])
        return R
    else:
        R = np.array([
            [np.cos(th), 0, np.sin(th)],
            [0, 1, 0],
            [-np.sin(th), 0, np.cos(th)]
        ])
        return R

def rotz(th, sym=False):
    """
    R = rotx(th)
    Parameters
        th: float or int, angle of rotation
    Returns
        R: 3 x 3 numpy array representing rotation about z-axis by amount theta
    """
    
    if sym:
        R = sp.Matrix([
            [sp.cos(th), -sp.sin(th), 0],
            [sp.sin(th), sp.cos(th), 0],
            [0, 0, 1]
        ])
        return R
    else:
        R = np.array([
            [np.cos(th), -np.sin(th), 0],
            [np.sin(th), np.cos(th), 0],
            [0, 0, 1]
        ])
        return R

# inverse of rotation matrix 
def rot_inv(R, sym=False):
    '''
    R = rot_inv(R)
    Parameters
        R: 2x2 or 3x3 numpy array representing a proper rotation matrix
    Returns
        R: 2x2 or 3x3 inverse of the input rotation matrix
    '''
    #if sym:
    #    return sp.transpose(R)
    #else:
    #    return R.T
    return R.T

def trn(R=np.eye(3), t=np.zeros((3,))):
    '''
    T = trn(R, t)

    Use se3 instead
    '''
    T = np.zeros((4,4))
    T[0:3,0:3] = R
    T[0:3,3] = t
    T[3,3] = 1
    return T

def se3(R=np.eye(3), p=np.array([0, 0, 0]), sym=False):
    """
        T = se3(R, p)
        Description:
            Given a numpy 3x3 array for R, and a 1x3 or 3x1 array for p, 
            this function constructs a 4x4 homogeneous transformation 
            matrix "T". 

        Parameters:
        R - 3x3 numpy array representing orientation, defaults to identity
        p = 3x1 numpy array representing position, defaults to [0, 0, 0]

        Returns:
        T - 4x4 numpy array
    """
    if sym:
        T = sp.eye(4)
    else:
        T = np.eye(4)
    
    T[0:3,0:3] = R
    T[0:3,3] = p

    return T

def dh(theta, d, a, alpha, sym=False):
    """
        Parameters:
        theta - Rotz
        d - Trnz
        a - Trnx
        alpha - Rotx
        
        Returns:
        T - The transformation matrix for the given DH parameters
    """
    
    T = se3(R=rotz(theta, sym=sym), p=[0,0,d], sym=sym) 
    T = T @ se3(R=rotx(alpha, sym=sym), p=[a,0,0], sym=sym)
    
    return T

def inv(T):
    """
        Tinv = inv(T)
        Description:
        Returns the inverse transform to T

        Parameters:
        T

        Returns:
        Tinv - 4x4 numpy array that is the inverse to T so that T @ Tinv = I
    """
    
    #TODO - fill this out 
    R = T[0:3,0:3]
    p = T[0:3,3]
    R_inv = R.T # works for sp.Matrix too
    p_inv = -R.T @ p
    T_inv = se3(R_inv, p_inv)

    return T_inv

def R2rpy(R):
    """
    rpy = R2rpy(R)
    Description:
    Returns the roll-pitch-yaw representation of the SO3 rotation matrix

    Parameters:
    R - 3 x 3 Numpy array for any rotation

    Returns:
    rpy - 1 x 3 Numpy Matrix, containing <roll pitch yaw> coordinates (in radians)
    """

    # See Siciliano 2.19

    roll = np.arctan2(R[1,0], R[0,0])
    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    yaw = np.arctan2(R[2,1], R[2,2])

    return np.array([roll, pitch, yaw])

def R2axis(R):
    """
    axis_angle = R2axis(R)
    Description:
    Returns an axis angle representation of a SO(3) rotation matrix

    Parameters:
    R

    Returns:
    axis_angle - 1 x 4 numpy matrix, containing  the axis angle representation
    in the form: <angle, rx, ry, rz>
    """

    # see equation (2.27) and (2.28) on pg. 54, using functions like "np.acos," "np.sin," etc. 
    ang = np.arccos( (R[0,0] + R[1,1] + R[2,2] - 1) / 2 )
    axis_angle = np.array([ang,
                           (R[2,1] - R[1,2])/(2*np.sin(ang)),
                           (R[0,2] - R[2,0])/(2*np.sin(ang)),
                           (R[1,0] - R[0,1])/(2*np.sin(ang))])

    return axis_angle

def axis2R(ang, v):
    """
    R = axis2R(angle, rx, ry, rz, radians=True)
    Description:
    Returns an SO3 object of the rotation specified by the axis-angle

    Parameters:
    angle - float, the angle to rotate about the axis in radians
    v = [rx, ry, rz] - components of the unit axis about which to rotate as 3x1 numpy array

    Returns:
    R - 3x3 numpy array
    """

    rx = v[0]
    ry = v[1]
    rz = v[2]
    ct = np.cos(ang)
    st = np.sin(ang)

    R = np.array([
        [rx**2*(1-ct)+ct, rx*ry*(1-ct)-rz*st, rx*rz*(1-ct)+ry*st],
        [rx*ry*(1-ct)+rz*st, ry**2*(1-ct)+ct, ry*rz*(1-ct)-rx*st],
        [rx*rz*(1-ct)-ry*st, ry*rz*(1-ct)+rx*st, rz**2*(1-ct)+ct]
    ])
    return R

def R2q(R):
    """
    quaternion = R2q(R)
    Description:
    Returns a quaternion representation of pose

    Parameters:
    R

    Returns:
    quaternion - 1 x 4 numpy matrix, quaternion representation of pose in the 
    format [nu, ex, ey, ez]
    """
    # TODO, see equation (2.34) and (2.35) on pg. 55, using functions like "sp.sqrt," and "sp.sign"

    return np.array([
        np.sqrt(R[0,0]+R[1,1]+R[2,2]+1)/2,
        np.sign(R[2,1]-R[1,2])*np.sqrt(R[0,0]-R[1,1]-R[2,2]+1)/2,
        np.sign(R[0,2]-R[2,0])*np.sqrt(R[1,1]-R[2,2]-R[0,0]+1)/2,
        np.sign(R[1,0]-R[0,1])*np.sqrt(R[2,2]-R[0,0]-R[1,1]+1)/2
    ])
                    
def q2R(q):
    """
    R = q2R(q)
    Description:
    Returns a 3x3 rotation matrix

    Parameters:
    q - 4x1 numpy array, [nu, ex, ey, ez ] - defining the quaternion
    
    Returns:
    R - a 3x3 numpy array 
    """
    # TODO, extract the entries of q below, and then calculate R
    nu = q[0]
    ex = q[1]
    ey = q[2]
    ez = q[3]
    R = np.array([
        [2*(nu**2+ex**2)-1, 2*(ex*ey-nu*ez), 2*(ex*ez+nu*ey)],
        [2*(ex*ey+nu*ez), 2*(nu**2+ey**2)-1, 2*(ey*ez-nu*ex)],
        [2*(ex*ez-nu*ey), 2*(ey*ez+nu*ex), 2*(nu**2+ez**2)-1]
    ])
    return R

def euler2R(th1, th2, th3, order='xyz'):
    """
    R = euler2R(th1, th2, th3, order='xyz')
    Description:
    Returns a 3x3 rotation matrix as specified by the euler angles, we assume in all cases
    that these are defined about the "current axis," which is why there are only 12 versions 
    (instead of the 24 possiblities noted in the course slides). 

    Parameters:
    th1, th2, th3 - float, angles of rotation
    order - string, specifies the euler rotation to use, for example 'xyx', 'zyz', etc.
    
    Returns:
    R - 3x3 numpy matrix
    """

    # TODO - fill out each expression for R based on the condition 
    # (hint: use your rotx, roty, rotz functions)
    
    valid_orders = ['xyx', 'xyz', 'xzx', 'xzy', 
                    'yxy', 'yxz', 'yzx', 'yzy', 
                    'zxy', 'zxz', 'zyx', 'zyz']
    
    if order not in valid_orders:
        print("Invalid Order!")
        return
    
    def rot_fun(char):
        if char == 'x':
            return rotx
        if char == 'y':
            return roty
        if char == 'z':
            return rotz
    
    rot1 = rot_fun(order[0])
    rot2 = rot_fun(order[1])
    rot3 = rot_fun(order[2])
    
    R = rot1(th1) @ rot2(th2) @ rot3(th3)

    return R
