"""
MPC
"""
# Imports
from casadi import *
import numpy as np
from Abgabe.Normalize.MinMax import minmax_norm, minmax_norm_back

class MPC:
    """
    Class: MPC
    """
    def __init__(self, S, N, Q, R, dist, dist_flag, model=None):
        self.factor = 1
        self.dist = dist_flag
        self.a = np.array([[0.8511, 0],
                           [0, 0.99999]])
        self.nx = self.a.shape[1]

        self.b = np.array([[4, 0],
                           [0, -2500*self.factor]])
        self.nu = self.b.shape[1]

        self.e = np.array([[0.22217, 0.017912, 0.42212],
                           [0, 0, 0]])

        self.nd = self.e.shape[1]

        self.x = SX.sym("x", self.nx, 1)
        self.u = SX.sym("u", self.nu, 1)
        self.d = SX.sym("d", self.nd, 1)

        self.int_gains = dist['int_gains'] * dist_flag /10
        self.room_temp = dist['room_temp'] * dist_flag /10
        self.sol_rad = dist['sol_rad'] * dist_flag /10

        if model is not None:
            # model.get_conf_weights()
            output_sys = transpose(model.nn_casadi(model.weights, model.config, horzcat(self.x.T, self.d.T, self.u.T)))

        else:
            output_sys = mtimes(self.a, self.x) + mtimes(self.b, self.u) + mtimes(self.e, self.d)
        self.system_nominal = Function("sys", [self.x, self.u, self.d], [output_sys])

        output_sys_real = mtimes(self.a, self.x) + mtimes(self.b, self.u) + mtimes(self.e, self.d)
        self.system_nominal_real = Function("sys", [self.x, self.u, self.d], [output_sys_real])
        self.S = S
        self.N = N



        self.x_ref = SX.sym("x_ref", self.nx, 1)
        # stage cost
        J_stage = 0.5 * mtimes(transpose(self.x_ref - self.x), mtimes(Q, (self.x_ref - self.x))) \
                  + 0.5 * mtimes(transpose(self.u), mtimes(R, self.u))

        self.J_function_stage = Function('J_stage', [self.x_ref, self.x, self.u], [J_stage])

        # terminal cost
        J_terminal = 0.5 * mtimes(transpose(self.x_ref - self.x), mtimes(Q, (self.x_ref - self.x)))
        self.J_function_terminal = Function('J_terminal', [self.x_ref, self.x], [J_terminal])

    def mpc_step(self, x_ref_values, x_init, NN_flag, min1=-inf, min2=-inf, mindist1=-inf, mindist2=-inf, mindist3=-inf,
                 max1=inf, max2=inf, maxdist1=inf, maxdist2=inf, maxdist3=inf):

        # Symbolic variables
        X = SX.sym("X", (self.N + 1) * self.nx, 1)
        U = SX.sym("U", self.N * self.nu, 1)
        lbu = np.array([[-1], [-1]])
        ubu = np.array([[1], [1]])
        if NN_flag == 1:
            #lbx = np.array([[minmax_norm(0, min1, max1)], [minmax_norm(0, min2, max2)]])
            #ubx = np.array([[minmax_norm(1, min1, max1)], [minmax_norm(1, min2, max2)]])
            lbx = np.array([[0], [0]])
            ubx = np.array([[1], [1]])
            #lbu = np.array([[-inf], [-inf]])
            #ubu = np.array([[inf], [inf]])
        else:
            lbx = np.array([[20], [0]])
            ubx = np.array([[25], [200000*self.factor]])

        mpc_x = np.zeros((self.S + 1 - self.N, self.nx))
        mpc_x[0, :] = x_init.T
        mpc_u = np.zeros((self.S - self.N, self.nu))

        for step in range(self.S - self.N):
            J = 0

            # system discription
            G = []
            lbg = []
            ubg = []
            lb_X = []
            ub_X = []
            lb_U = []
            ub_U = []

            for k in range(self.N):

                x_k = X[k * self.nx:(k + 1) * self.nx, :]
                x_k_next = X[(k + 1) * self.nx:(k + 2) * self.nx, :]
                u_k = U[k * self.nu:(k + 1) * self.nu, :]
                if NN_flag == 1 and self.dist != 0:
                    # normalize disturbances
                    room_temp = minmax_norm(self.room_temp.item(step+k), mindist1, maxdist1)
                    sol_rad = minmax_norm(self.sol_rad.item(step+k), mindist2, maxdist2)
                    int_gains = minmax_norm(self.int_gains.item(step + k), mindist3, maxdist3)
                    d_k = np.array([room_temp, sol_rad, int_gains])
                else:
                    d_k = np.array([[self.room_temp.item(step + k)], [self.sol_rad.item(step + k)],
                                [self.int_gains.item(step + k)]])

                # objective
                print(step + k)
                x_ref = x_ref_values[:, step + k].T
                J += self.J_function_stage(x_ref, x_k, u_k)

                # equality constraints (system equation)
                x_next = self.system_nominal(x_k, u_k, d_k)

                if k == 0:
                    G.append(x_k)
                    lbg.append(x_init)
                    ubg.append(x_init)

                G.append(minus(x_next, x_k_next))
                lbg.append(np.zeros((self.nx, 1)))
                ubg.append(np.zeros((self.nx, 1)))
                # inequality constraints
                lb_X.append(lbx)
                ub_X.append(ubx)
                lb_U.append(lbu)
                ub_U.append(ubu)

            # Terminal cost and constraints
            x_k = X[self.N * self.nx:(self.N + 1) * self.nx, :]
            J += self.J_function_terminal(x_ref_values[:, step], x_k)
            lb_X.append(lbx)
            ub_X.append(ubx)

            # solve optimization problem
            # f - function , x - varaibles, g - constrains
            lb = vertcat(vertcat(*lb_X), vertcat(*lb_U))
            ub = vertcat(vertcat(*ub_X), vertcat(*ub_U))
            prob = {'f': J, 'x': vertcat(X, U), 'g': vertcat(*G)}
            opts = {}
            # opts["ipopt.print_level"] = 0
            # opts["print_time"] = 0
            solver = nlpsol('solver', 'ipopt', prob, opts)  # nlpsol  ipopt,  qpsol qpoases
            res = solver(lbx=lb, ubx=ub, lbg=vertcat(*lbg), ubg=vertcat(*ubg))

            u_opt = res['x'][(self.N + 1) * self.nx:(self.N + 1) * self.nx + self.nu, :]
            d = np.array([[self.room_temp.item(step)], [self.sol_rad.item(step)], [self.int_gains.item(step)]])

            if NN_flag == 1:
                x_init[0,0] = minmax_norm_back(x_init[0,0], min1, max1)
                x_init[1, 0] = minmax_norm_back(x_init[1,0], min2, max2)
            x_plus = self.system_nominal_real(x_init, u_opt, d)
            if NN_flag == 1:
                x_plus[0,0] = minmax_norm(x_plus[0,0], min1, max1)
                x_plus[1, 0] = minmax_norm(x_plus[1, 0], min2, max2)
            mpc_x[step + 1, :] = x_plus.T
            mpc_u[step, :] = u_opt.T
            x_init = x_plus

        return mpc_x, mpc_u
