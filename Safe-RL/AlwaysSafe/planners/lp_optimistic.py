import os
import numpy as np
import cvxpy as cv

from .lp import LinearProgrammingPlanner
from util.grb import *


class OptimisticLinearProgrammingPlanner(LinearProgrammingPlanner):

    def __init__(self, *args, reward_ci=None, cost_ci=None, transition_ci=None, **kwargs):
        super().__init__(*args, **kwargs)
        if reward_ci is None:
            self.r_ci = np.zeros(shape=self.reward.shape)
        else:
            self.r_ci = reward_ci
            self.r_ci[self.terminal] = 0
        if cost_ci is None:
            self.c_ci = np.zeros(shape=self.cost.shape)
        else:
            self.c_ci = cost_ci
            self.c_ci[self.terminal] = 0
        if transition_ci is None:
            self.t_ci = np.zeros(shape=self.transition.shape)
        else:
            self.t_ci = transition_ci
            self.t_ci[self.terminal] = 0
        self.y = []
        self.reward_ub = np.clip(self.reward + self.r_ci, self.min_reward, self.max_reward)
        self.cost_lb = np.clip(self.cost - self.c_ci, self.min_cost, self.max_cost)
        self.transition_ub = np.clip(self.transition + self.t_ci, 0, 1)
        self.transition_lb = np.clip(self.transition - self.t_ci, 0, 1)

    def instantiate_lp_cvxpy(self):
        # variables
        # y is the occupancy on time step h of the tuple s,a,s'
        self.y = [
            [cv.Variable(shape=(self.na, self.ns), nonneg=True) for s in self.states]
            for h in range(self.horizon)
        ]
        # x is the occupancy on time step h of the tuple s,a
        self.x = [cv.Variable(shape=(self.ns, self.na)) for h in range(self.horizon)]

        # expressions
        self.exp_reward = cv.Constant(0)
        self.exp_cost = cv.Constant(0)

        for h in range(self.horizon):
            self.exp_cost += cv.sum(cv.multiply(self.x[h], self.cost_lb))
            for s in self.states:
                self.exp_reward += cv.sum(cv.multiply(self.x[h][s], self.reward_ub[s]))

        # objective
        obj = cv.Maximize(self.exp_reward)

        # constraints
        if self.horizon > 0:
            constraints = []
            for h in range(self.horizon):
                for s in self.states:
                    constraints.append(self.x[h][s] == cv.sum(self.y[h][s], axis=1))
                    if h > 0:
                        inflow = cv.Constant(0)
                        for t in self.states:
                            inflow += cv.sum(self.y[h - 1][t][:, s])

                        if self.terminal[s]:
                            constraints += [cv.sum(self.y[h][s]) == 0]
                        else:
                            constraints += [inflow == cv.sum(self.y[h][s])]
                    else:
                        constraints += [
                            # outflow == inflow (first time step)
                            cv.sum(self.y[0][s]) == self.isd[s]
                            for s in self.states
                        ]
                    constraints += [
                        self.y[h][s][a] <= self.x[h][s, a] * self.transition_ub[s, a]
                        for a in self.actions
                    ]
                    constraints += [
                        self.y[h][s][a] >= self.x[h][s, a] * self.transition_lb[s, a]
                        for a in self.actions
                    ]
        else:
            constraints = []
        if self.cost_bound is not None:
            constraints.append(self.exp_cost <= self.cost_bound)

        # problem
        self.lp = cv.Problem(obj, constraints)

    def instantiate_lp_grb(self):
        self.lp = Model('ConstrainedMDP')
        if not self.verbose:
            self.lp.Params.OutputFlag = 0

        if self.verbose:
            print("adding variables")
        self.x = x = self.lp.addVars(range(self.horizon), self.states, self.actions, lb=0.0, name='x')
        self.y = y = self.lp.addVars(range(self.horizon), self.states, self.actions, self.states, lb=0.0, name='x')

        if self.verbose:
            print("adding constraints")

        self.exp_reward = quicksum(
            quicksum(
                quicksum(
                    x[h, s, a] * self.reward_ub[s, a]
                    for h in range(self.horizon)
                )
                for a in self.actions
            )
            for s in self.non_terminal_states
        )
        self.exp_cost = quicksum(
            quicksum(
                quicksum(
                    x[h, s, a] * self.cost_lb[s, a]
                    for h in range(self.horizon)
                )
                for a in self.actions
            )
            for s in self.non_terminal_states
        )
        if self.cost_bound is not None:
            self.lp.addConstr(
                self.exp_cost <= self.cost_bound,
                "expected_cost_bound"
            )

        if self.horizon > 0:
            self.lp.addConstrs((
                quicksum(x[0, s, a] for a in self.actions) == self.isd[s]
                for s in self.states
            ), 'isd')

        self.lp.addConstrs((
            x[h, s, a] == quicksum(y[h, s, a, suc] for suc in self.states)
            for s in self.states for h in range(self.horizon) for a in self.actions
        ), 'x_is_sum_y')

        self.lp.addConstrs((
            y[h, s, a, suc] <= self.transition_ub[s, a, suc] * x[h, s, a]
            for h in range(self.horizon) for suc in self.states for a in self.actions for s in self.states
        ), 'transition_ub')
        self.lp.addConstrs((
            y[h, s, a, suc] >= self.transition_lb[s, a, suc] * x[h, s, a]
            for h in range(self.horizon) for suc in self.states for a in self.actions for s in self.states
        ), 'transition_lb')

        self.lp.addConstrs((
            # out(suc) == in(suc)
            quicksum(x[h, suc, a] for a in self.actions)
            == quicksum(
                quicksum(
                    y[h-1, s, a, suc]
                    for s in self.states
                ) for a in self.actions)
            for suc in self.non_terminal_states for h in range(1, self.horizon)
        ), 'transient')

        self.lp.addConstrs((
            quicksum(
                x[h, suc, a]
                for a in self.actions
            ) == 0
            for suc in self.states if self.terminal[suc] for h in range(self.horizon)
        ), 'terminal_flow')

        if self.verbose:
            print("setting objective")
        self.lp.setObjective(
            self.exp_reward,
            GRB.MAXIMIZE
        )

        self.lp.update()

    def to_dot(self, filename, form='pdf'):
        with open(filename, 'w') as out:
            out.write(self.get_graph_header())
            out.write( self.get_graph_dot())
            out.write("}\n")
        self.try_draw(filename, form)

    def get_graph_dot(self):
        res = ""
        for h in range(self.horizon):
            states_h = []
            for s in self.states:
                outflow_s = 0
                inflow_s = 0
                for a in self.actions:
                    for t in self.states:
                        if self.solver == 'grb':
                            outflow_s += self.y[h, s, a, t].x
                            if h > 0:
                                inflow_s += self.y[h-1, t, a, s].x
                        else:
                            outflow_s += self.y[h][s][a, t].value
                            if h > 0:
                                inflow_s += self.y[h-1][t][a, s].value
                if outflow_s < 0.00001 and inflow_s < 0.000001:
                    continue
                decoded_s = "".join([str(x) for x in self.env.decode(s)])
                from_s = "h" + str(h) + "s" + str(s) + "f" + str(decoded_s)
                label = "conc: {}".format(s)
                res += "\t{} [label=\"{}\"]\n".format(from_s, label)
                states_h.append(from_s)
                if h == self.horizon - 1:
                    continue
                for a in self.actions:
                    for t in self.states:
                        if self.solver == 'grb':
                            o = self.y[h, s, a, t].x
                        else:
                            o = self.y[h][s][a, t].value
                        if o > 0.001:
                            decoded_t = "".join([str(x) for x in self.env.decode(t)])
                            to_t = "h" + str(h + 1) + "s" + str(t) + "f" + decoded_t
                            res += "\t{} -> {} [label=< a={} p={:.3f}>]\n".format(from_s, to_t, a, o)
            sh = ""
            for s in states_h:
                sh += "{}; ".format(s)
            res += "subgraph t"+ str(h)+ " {rank = same; label = \"t=" + str(h) + "\"; labeljust=\"l\"; " + sh + "  } \n"
        return res

    @staticmethod
    def get_graph_header():
        return(
            "digraph {\n"
            + "\tgraph [ dpi=\"360\", pad=\"0.5\", nodesep=\"1\", ranksep=\"1\"]\n"
            + "\tnode [ fontname = STIXGeneral fontsize = 10 shape=circle style=filled color=\"#E0E0E0\" fillcolor=\"#E0E0E0\"]\n"
            + "\tedge [ fontname = STIXGeneral fontsize = 10 color = \"#E0E0E0\"]\n"
        )

    @staticmethod
    def try_draw(filename, form):
        command = "dot -T{1} {0} -o {0}.{1}".format(filename, form)
        res = os.system(command)
        if res != 0:
            print("the following command failed")
            print(command)
