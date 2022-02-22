"""Some common functions for manipulating quaternions.

VERSION HISTORY
Aug 14, 2014 - initialy created (Felix Berkenkamp)
"""
from __future__ import print_function, division, absolute_import

import math
import numpy as np

from .transformations import (vector_norm,
                              quaternion_multiply,
                              quaternion_conjugate,
                              quaternion_matrix,
                              quaternion_about_axis)

__all__ = ['omega_from_quat_quat', 'apply_omega_to_quat', 'global_to_body',
           'body_to_global']


def omega_from_quat_quat(q1, q2, dt):
    """
    Convert two quaternions and the time difference to angular velocity.

    Parameters:
    -----------
    q1: quaternion
        The old quaternion
    q2: quaternion
        The new quaternion
    dt: float
        The time difference

    Returns:
    --------
    omega_g: ndarray
        The angular velocity in global coordinates
    """
    if vector_norm(q1 - q2) < 1e-8:
        # linearly interpolate
        # the quaternion does not stay on unit sphere -> only for very small
        # rotations!

        # dq/dt
        dq = (q2 - q1) / dt

        # From Diebel: Representing Atitude, 6.6, quaternions are defined
        # differently there: [w, x, y, z] instead of [x, y, z, w]!
        omega = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Equivalent, but slower
        # w = np.array([[q2[3], -q2[2], q2[1], -q2[0]],
        #               [q2[2], q2[3], -q2[0], -q2[1]],
        #               [-q2[1], q2[0], q2[3], -q2[2]]], dtype=np.float64)
        #
        # omega = 2 * w.dot(dq)

        omega[0] = 2.0 * (
            q2[3] * dq[0] - q2[2] * dq[1] + q2[1] * dq[2] - q2[0] * dq[3])
        omega[1] = 2.0 * (
            q2[2] * dq[0] + q2[3] * dq[1] - q2[0] * dq[2] - q2[1] * dq[3])
        omega[2] = 2.0 * (
            -q2[1] * dq[0] + q2[0] * dq[1] + q2[3] * dq[2] - q2[2] * dq[3])

        return omega
    else:
        # This function becomes numerically unstable for q1-q2 --> 0

        # Find rotation from q1 to q2
        # unit quaternion -> conjugate is the same as inverse
        # q2 = r * q1 --> r = q2 * inv(q1)
        r = quaternion_multiply(q2, quaternion_conjugate(q1))

        # Angle of rotation
        angle = 2.0 * math.acos(r[3])

        # acos gives value in [0,pi], ensure that we take the short path
        # (e.g. rotate by -pi/2 rather than 3pi/2)
        if angle > math.pi:
            angle -= 2.0 * math.pi

        # angular velocity = angle / dt
        # axis of rotation corresponds to r[:3]
        return angle / dt * r[:3] / vector_norm(r[:3])


def apply_omega_to_quat(q, omega, dt):
    """
    Convert a quaternion q and apply the angular velocity omega to it over dt.

    Parameters:
    -----------
    q: quaternion
    omega: ndarray
        angular velocity
    dt: float
        time difference

    Returns:
    --------
    quaternion
        The quaternion of the orientation after rotation with omega for dt
        seconds.
    """
    # rotation angle around each axis
    w = omega * dt

    # only rotate if the angle we rotate through is actually significant
    if vector_norm(w) < np.finfo(float).eps * 4.0:
        return q

    # quaternion corresponding to this rotation
    # w = 0 is not a problem because numpy is awesome
    r = quaternion_about_axis(vector_norm(w), w)

    # return the rotated quaternion closest to original
    return quaternion_multiply(r, q)


def global_to_body(q, vec):
    """
    Convert a vector from global to body coordinates.

    Parameters:
    -----------
    q: quaternion
        The rotation quaternion
    vec: ndarray
        The vector in global coordinates

    Returns:
    vec: ndarray
        The vector in body coordinates
    """
    # quaternion_matrix(q)[:3,:3] is a homogenous rotation matrix that
    # rotates a vector by q
    # quaternion_matrix(q)[:3,:3] is rot. matrix from body to global frame
    # its transpose is the trafo matrix from global to body
    # that matrix is multiplied by omega
    return np.dot(quaternion_matrix(q)[:3, :3].transpose(), vec)


def body_to_global(q, vec):
    """Convert a vector from global to body coordinates.

    Parameters:
    -----------
    q: quaternion
        The rotation quaternion
    vec: ndarray
        The vector in body coordinates

    Returns:
    vec: ndarray
        The vector in global coordinates
    """
    # quaternion_matrix(q)[:3,:3] is a homogenous rotation matrix that
    # rotates a vector by q
    # quaternion_matrix(q)[:3,:3] is the matrix from body to global frame
    # that matrix is multiplied by omega
    return np.dot(quaternion_matrix(q)[:3, :3], vec)
