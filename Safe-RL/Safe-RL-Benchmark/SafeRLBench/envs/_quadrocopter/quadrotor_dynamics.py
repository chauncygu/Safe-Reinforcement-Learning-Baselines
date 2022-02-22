"""Quadrotor Dynamics."""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

from .quadrocopter_classes import State, Parameters

__all__ = ['QuadrotorDynamics', 'wind_creator', 'random_disturbance_creator']


class QuadrotorDynamics(object):
    """Implement the quadrotor dynamics and states (independent of gazebo).

    Attributes
    ----------
    pos: 3d array
        Initial position of quadrotor
    vel: 3d array
        Initial velocity of quadrotor
    acc: 3d array
        Initial acceleration of quadrotor
    R: 3x3 array
        Initial rotation matrix
    external_forces: list
        a list of callables that take the state as input and return forces on
        the quadrotor in global coordinates.

    Notes
    -----
    There seems to be an instability where the acceleration overflows and then
    causes issues in the controller.
    """

    def __init__(self, pos=None, vel=None, acc=None, R=None,
                 external_forces=None):
        """Initialize quadrocopter dynamics.

        Parameters
        ----------
        pos: 3d array
            Initial position of quadrotor
        vel: 3d array
            Initial velocity of quadrotor
        acc: 3d array
            Initial acceleration of quadrotor
        R: 3x3 array
            Initial rotation matrix
        external_forces: list
            a list of callables that take the state as input and return forces
            on the quadrotor in global coordinates.
        """
        self.state = State()
        self.params = Parameters()

        if external_forces is None:
            self.external_forces = ()
        else:
            self.external_forces = external_forces

        if pos is not None:
            self.state.pos = pos.copy()
        if vel is not None:
            self.state.vel = vel.copy()
        if acc is not None:
            self.state.acc = acc.copy()
        if R is not None:
            self.state.R = R.copy()

    def dynamics_derivative(self, pitch, roll, z_vel, yaw_vel):
        """Return the state derivatives for the current state and input."""
        rates = self._inputs_to_desired_rates(pitch, roll, z_vel, yaw_vel)

        forces = self._determine_forces(*rates)

        return self._forces_to_derivatives(forces)

    def update_position(self, inputs):
        """Compute the derivatives and integrate them based on inputs."""
        pitch, roll, z_vel, yaw_vel = inputs
        derivatives = self.dynamics_derivative(pitch, roll, z_vel, yaw_vel)
        self._integrate_derivatives(derivatives,
                                    self.params.inner_loop_cycle * 1e-6)

    def _inputs_to_desired_rates(self, pitch, roll, z_vel, yaw_vel):
        """Convert inputs to desired angular rates and thrust."""
        # Current roll, and yaw angles
        roll_cur, _, yaw_cur = self.state.rpy

        # r_des is simply the commanded yaw rate
        r_des = yaw_vel

        # calculate the commanded acceleration in the z direction,
        # (z_dot_des - z_dot) / tau_z
        z_ddot_des = (z_vel - self.state.vel[2]) / self.params.tau_Iz

        # And from this we may find the commanded thrust, (g + z_ddot_cmd)/R33
        c_des = (self.params.g + z_ddot_des) / self.state.R[2, 2]

        # Calculate the commanded yaw angle from:
        yaw_des = yaw_vel * self.params.tau_Iyaw + yaw_cur

        # R13_des = sin(yaw_des) * sin(roll_cmd)
        #         + cos(yaw_des) * cos(roll_cmd) * sin(pitch_cmd)
        r_13_des = (np.sin(yaw_des) * np.sin(roll) +
                    np.cos(yaw_des) * np.cos(roll) * np.sin(pitch))

        # R23_des = cos(roll_cmd) * sin(yaw_des) * sin(pitch_cmd)
        #         - cos(yaw_des) * sin(roll_cmd)
        r_23_des = (np.cos(roll) * np.sin(yaw_des) * np.sin(pitch) -
                    np.cos(yaw_des) * np.sin(roll))

        # p_des = (R21*(R13_des-R13) - R11*(R23_des-R23))/(R33*tau_rp)
        p_des = (self.state.R[1, 0] * (r_13_des - self.state.R[0, 2]) -
                 self.state.R[0, 0] * (r_23_des - self.state.R[1, 2]))
        p_des /= self.state.R[2, 2] * self.params.tau_rp

        # q_des = (R22*(R13_des-R13) - R12*(R23_des-R23))/(R33*tau_rp)
        q_des = (self.state.R[1, 1] * (r_13_des - self.state.R[0, 2]) -
                 self.state.R[0, 1] * (r_23_des - self.state.R[1, 2]))
        q_des /= self.state.R[2, 2] * self.params.tau_rp

        # Return everything!
        return p_des, q_des, r_des, c_des

    def _determine_forces(self, p_des, q_des, r_des, c_des):
        """Convert desired angular rates and thrust to rotor forces."""
        L = self.params.L
        K = self.params.K
        m = self.params.m

        a = np.array(((0, L, 0, -L),
                      (-L, 0, L, 0),
                      (K, -K, K, -K),
                      (1 / m, 1 / m, 1 / m, 1 / m)),
                     dtype=np.float64)

        # The inertial matrix
        j = np.diag((self.params.Ix, self.params.Iy, self.params.Iz))

        # The current angular velocity vector
        omega = self.state.omega

        # The rate vector (our approximation of omega_dot)
        rate_vector = np.array(
            (((1 / self.params.tau_p) * (p_des - self.state.omega[0])),
             ((1 / self.params.tau_q) * (q_des - self.state.omega[1])),
             ((1 / self.params.tau_r) * (r_des - self.state.omega[2])))).T

        b = j.dot(rate_vector) + np.cross(omega, j.dot(omega))

        # Add c_des to the bottom of the row vector
        b = np.concatenate((b, [c_des]))

        # Return the four rotor forces
        return np.linalg.solve(a, b)

    def _forces_to_derivatives(self, forces):
        """Compute the state derivatives based on applied forces."""
        # Update position
        derivatives = State()

        derivatives.pos[:] = self.state.vel

        drag = self._compute_drag()

        # Update accelerations
        derivatives.acc = np.sum(forces) * self.state.R[:, 2] - drag

        # Add external forces
        for force in self.external_forces:
            derivatives.acc += force(self.state)

        # Normalize with mass and add gravity
        derivatives.acc /= self.params.m
        derivatives.acc[2] -= self.params.g

        # Update velocities
        derivatives.vel[:] = self.state.acc

        p, q, r = self.state.omega
        derivatives.R = self.state.R.dot(np.array([[0, -r, q],
                                                   [r, 0, -p],
                                                   [-q, p, 0]]))

        # Angular velocity changes
        f1, f2, f3, f4 = forces

        # p' = (1/Ix)*(L*(f2-f4) + (Iy-Iz)*r*q)
        p_dot = (self.params.L * (f2 - f4) +
                 (self.params.Iy - self.params.Iz) *
                 self.state.omega[2] * self.state.omega[1]) / self.params.Ix

        # q' = (1/Iy)*(L*(f3-f1) + (Iz-Ix)*r*p)
        q_dot = (self.params.L * (f3 - f1) +
                 (self.params.Iz - self.params.Ix) *
                 self.state.omega[2] * self.state.omega[0]) / self.params.Iy

        # r' = (1/Iz)*(K*(f1-f2+f3-f4) + (Ix-Iy)*p*q)
        r_dot = (self.params.K * (f1 - f2 + f3 - f4) +
                 (self.params.Ix - self.params.Iy) *
                 self.state.omega[0] * self.state.omega[1]) / self.params.Iz

        derivatives.omega = np.array([p_dot, q_dot, r_dot])

        return derivatives

    def _integrate_derivatives(self, derivatives, dt):
        """Simple euler integration to determine new states."""
        self.state.pos += dt * derivatives.pos
        self.state.vel += dt * derivatives.vel
        self.state.acc[:] = derivatives.acc

        self.state.R += dt * derivatives.R
        self.state.omega += dt * derivatives.omega

    def _compute_drag(self):
        """Compute velocities and applies linear drag model.

        Inverts the current rotation matrix and solves for the components of
        quadrocopter velocities in the body coordinates. Then a simple linear
        drag model equation is applied. This is done because the quadrocopter
        platform areas don't change in this refernce frame. The drag forces are
        returned in global coordinates.
        """
        v_b = np.linalg.solve(self.state.R, self.state.vel)

        drag_model = np.array((self.params.CD_bx,
                               self.params.CD_by,
                               self.params.CD_bz)) * v_b

        return self.state.R.dot(drag_model)


def wind_creator(direction, strength):
    """
    Return callable that computes the wind force on the quadrotor.

    Parameters:
    direction: 3d-array
        Direction vector for the wind.
    strength: float
        Strength of the wind in N / m^2
    """
    direction = np.asarray(direction, dtype=np.float).squeeze()
    direction /= np.linalg.norm(direction)

    quadrotor_length = 0.3
    quadrotor_height = 0.05

    norm_area = np.array((quadrotor_length * quadrotor_height,
                          quadrotor_length * quadrotor_height,
                          quadrotor_length ** 2))

    def wind_force(state):
        """Return wind force.

        Homogeneous wind, this does not create any torques.

        Parameters
        ----------
        state :
        """
        # Project surface areas into the wind direction
        area = np.abs(direction.dot(state.R)) * norm_area
        force = np.sum(area) * strength * direction
        return force

    return wind_force


def random_disturbance_creator(covariance, mean=None):
    """Add gaussian disturbance forces with a certain covariance function.

    Parameters
    ----------
    covariance: np.array
        A 3x3 array of the covariance matrix
    mean: np.array
        A 1d array of the 3 mean values (defaults to zero-mean)

    Returns
    -------
    disturbance: callable
        A function that can be used as an external force in quadsim
    """
    if mean is None:
        mean = np.zeros((3,))

    def random_force(state):
        """Return wind force.

        Parameters
        ----------
        state: State

        Returns
        -------
        force: np.array
        """
        return np.random.multivariate_normal(mean, covariance)

    return random_force
