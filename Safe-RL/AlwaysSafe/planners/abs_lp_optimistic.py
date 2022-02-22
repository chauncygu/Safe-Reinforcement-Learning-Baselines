import cvxpy as cv
import numpy as np
from typing import Sequence

from gym_factored.envs.base import DiscreteEnv

from .lp_optimistic import OptimisticLinearProgrammingPlanner
from util.mdp import get_mdp_functions_partial
from util.mdp import get_mdp_functions
from util.grb import *


class AbsOptimisticLinearProgrammingPlanner(OptimisticLinearProgrammingPlanner):
    def __init__(self, *args, features: Sequence = None, reward_ci=None, transition_ci=None,
                 abs_transition=None, abs_cost=None, abs_terminal=None, abs_map=None, policy_type='ground',
                 cost_bound_coefficient=1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_type = policy_type
        self.features = features
        self.cost_bound_coefficient = cost_bound_coefficient

        if reward_ci is None:
            self.r_ci = np.zeros(shape=self.reward.shape)
        else:
            self.r_ci = reward_ci
            self.r_ci[self.terminal] = 0
        if transition_ci is None:
            self.t_ci = np.zeros(shape=self.transition.shape)
        else:
            self.t_ci = transition_ci
            self.t_ci[self.terminal] = 0
        self.y = []
        self.z = []
        self.reward_ub = np.clip(self.reward + self.r_ci, self.min_reward, self.max_reward)
        self.transition_ub = np.clip(self.transition + self.t_ci, 0, 1)
        self.transition_lb = np.clip(self.transition - self.t_ci, 0, 1)

        self.abs_transition = abs_transition
        self.abs_cost = abs_cost
        self.abs_terminal = abs_terminal
        self.abs_map = abs_map  # map from state to abstract_state
        self.abs_states = np.arange(self.abs_map.shape[0])

    @classmethod
    def from_discrete_env(cls, env: DiscreteEnv, features: Sequence = None, **kwargs)\
            -> 'AbsOptimisticLinearProgrammingPlanner':
        transition, reward, _, terminal = get_mdp_functions(env)
        cost = np.zeros(shape=reward.shape)
        for s in np.arange(env.nS)[terminal]:
            transition[s, :, :] = 0
            transition[s, :, s] = 1
            reward[s, :] = 0
        max_reward, min_reward = reward.max(), reward.min()
        abs_transition, abs_reward, abs_cost, abs_terminal, abs_map = get_mdp_functions_partial(env, features)
        for s in np.arange(abs_terminal.shape[0])[abs_terminal]:
            abs_transition[s, :, :] = 0
            abs_transition[s, :, s] = 1
        return cls(transition, reward, cost, terminal, env.isd, env, max_reward, min_reward, 0, 0,
                   features=features,
                   abs_transition=abs_transition, abs_cost=abs_cost,  abs_terminal=abs_terminal, abs_map=abs_map,
                   **kwargs)

    def instantiate_lp_cvxpy(self):
        # variables
        # y is the occupancy on time step h of the tuple s,a,s'
        self.y = [
            [cv.Variable(shape=(self.na, self.ns), nonneg=True) for _ in self.states]
            for _ in range(self.horizon)
        ]
        # x is the occupancy on time step h of the tuple s,a
        self.x = [cv.Variable(shape=(self.ns, self.na), nonneg=True) for _ in range(self.horizon)]
        # z is the occupancy on time step h of the tuple abs_s, a
        self.z = [cv.Variable(shape=(len(self.abs_states), self.na), nonneg=True) for _ in range(self.horizon)]

        # expressions
        self.exp_reward = cv.Constant(0)
        self.exp_cost = cv.Constant(0)

        for h in range(self.horizon):
            self.exp_cost += cv.sum(cv.multiply(self.z[h], self.abs_cost))
            for s in self.states:
                self.exp_reward += cv.sum(cv.multiply(self.x[h][s], self.reward_ub[s]))

        # objective
        obj = cv.Maximize(self.exp_reward)

        # constraints
        constraints = []
        if self.horizon > 0:
            for h in range(self.horizon):
                for s in self.states:
                    constraints.append(self.x[h][s] == cv.sum(self.y[h][s], axis=1))
                    if h > 0:
                        if self.terminal[s]:
                            constraints += [cv.sum(self.y[h][s]) == 0]
                        else:
                            inflow = cv.Constant(0)
                            for t in self.states:
                                inflow += cv.sum(self.y[h - 1][t][:, s])
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
                for abs_s in self.abs_states:
                    constraints.append(cv.sum(self.x[h][self.abs_map[abs_s], :], axis=0) == self.z[h][abs_s])
                if h > 0:
                    constraints += [
                        # outflow == inflow
                        cv.sum(self.z[h][abs_s])
                        == cv.sum(cv.multiply(self.z[h - 1], self.abs_transition[:, :, abs_s]))
                        for abs_s in self.abs_states[~self.abs_terminal]
                    ]
                    if self.abs_terminal.any():
                        constraints += [
                            cv.sum(self.z[h][self.abs_terminal], axis=1) == 0
                        ]
        if self.cost_bound is not None:
            constraints.append(self.exp_cost <= self.cost_bound * self.cost_bound_coefficient)

        # problem
        self.lp = cv.Problem(obj, constraints)

    def instantiate_lp_grb(self):
        self.lp = Model("ConstrainedMDP")
        if not self.verbose:
            self.lp.Params.OutputFlag = 0
        self.lp.Params.Presolve = 0

        if self.verbose:
            print("adding variables")
        self.x = self.lp.addVars(range(self.horizon), self.states, self.actions, lb=0.0, name='x')
        self.y = self.lp.addVars(range(self.horizon), self.states, self.actions, self.states, lb=0.0, name='y')
        self.z = self.lp.addVars(range(self.horizon), self.abs_states, self.actions, lb=0.0, name='z')

        if self.verbose:
            print("adding constraints")

        self.exp_reward = quicksum(
            quicksum(
                quicksum(
                    self.x[h, s, a] * self.reward_ub[s, a]
                    for h in range(self.horizon)
                )
                for a in self.actions
            )
            for s in self.states
        )
        self.exp_cost = quicksum(
            quicksum(
                quicksum(
                    self.z[h, s, a] * self.abs_cost[s, a]
                    for h in range(self.horizon)
                )
                for a in self.actions
            )
            for s in self.abs_states
        )
        if self.cost_bound is not None:
            self.lp.addConstr(
                self.exp_cost <= self.cost_bound * self.cost_bound_coefficient,
                "expected_cost_bound"
            )

        if self.horizon > 0:
            self.lp.addConstrs((
                # quicksum(self.x[0, s, a] for a in self.actions) == self.isd[s]
                quicksum(self.y[0, s, a, suc] for a in self.actions for suc in self.states) == self.isd[s]
                for s in self.states
            ), "isd")

        self.lp.addConstrs((
            self.x[h, s, a] == quicksum(self.y[h, s, a, suc] for suc in self.states)
            for h in range(self.horizon) for s in self.states for a in self.actions
        ), "x_is_sum_y")
        self.lp.addConstrs((
            self.z[h, abs_s, a] == quicksum(self.x[h, s, a] for s in self.states[self.abs_map[abs_s]])
            for h in range(self.horizon) for abs_s in self.abs_states for a in self.actions
        ), "z_is_sum_x")

        self.lp.addConstrs((
            self.y[h, s, a, suc] <= self.transition_ub[s, a, suc] * self.x[h, s, a]
            for h in range(self.horizon) for s in self.states for a in self.actions for suc in self.states
        ), "transition_ub")
        self.lp.addConstrs((
            self.y[h, s, a, suc] >= self.transition_lb[s, a, suc] * self.x[h, s, a]
            for h in range(self.horizon) for s in self.states for a in self.actions for suc in self.states
        ), "transition_lb")

        self.lp.addConstrs((
            # out(suc) == in(suc)
            quicksum(self.y[h, s, a, suc] for a in self.actions for suc in self.states)
            == quicksum(
                quicksum(
                    self.y[h-1, pred, a, s]
                    for pred in self.states
                ) for a in self.actions)
            # for s in self.states for h in range(1, self.horizon)
            for s in self.states[~self.terminal] for h in range(1, self.horizon)
        ), "transient")

        self.lp.addConstrs((
            quicksum(
                self.y[h, s, a, suc]
                for a in self.actions for suc in self.states
            ) == 0
            for s in self.states[self.terminal] for h in range(self.horizon)
        ), "terminal_flow")

        self.lp.addConstrs((
            # abs_out(suc) == abs_in(suc)
            quicksum(self.z[h, abs_s, a] for a in self.actions)
            == quicksum(
                self.abs_transition[abs_pred, a, abs_s] * self.z[h-1, abs_pred, a]
                for abs_pred in self.abs_states for a in self.actions
            )
            for abs_s in self.abs_states[~self.abs_terminal] for h in range(1, self.horizon)
        ), "transient_abs")

        self.lp.addConstrs((
            quicksum(
                self.z[h, s, a]
                for a in self.actions
            ) == 0
            for s in self.abs_states[self.abs_terminal] for h in range(self.horizon)
        ), "terminal_flow_abs")

        if self.verbose:
            print("setting objective")
        self.lp.setObjective(
            self.exp_reward,
            GRB.MAXIMIZE
        )

        self.lp.update()

    def extract_policy(self):
        if self.policy_type == 'ground':
            self.extract_ground_policy()
        elif self.policy_type == 'abs':
            self.extract_abs_policy()
        elif self.policy_type == 'global_test':
            self.extract_safe_policy_global_test()
        elif self.policy_type == 'local_test':
            self.extract_safe_policy_local_test()
        elif self.policy_type == 'adaptive':
            self.extract_ground_policy_for_worst_case()

    def extract_abs_policy(self):
        for h in range(self.horizon):
            for s in self.abs_states:
                self.set_abs_policy(h, s)

    def set_abs_policy(self, h, s):
        occupancy = self.get_occupancy_abs(h, s)
        state_occupancy = occupancy.sum()
        if state_occupancy > 0:
            self.policy[h][self.abs_map[s]] = occupancy / state_occupancy
        else:
            self.policy[h][self.abs_map[s]] = np.full(self.na, fill_value=1. / self.na)

    def get_occupancy_abs(self, h, s):
        occupancy = np.zeros(self.na, dtype=float)
        if self.solver == "grb":
            for a in self.actions:
                occupancy[a] = max(self.z[h, s, a].x, 0)
        else:
            occupancy = np.maximum(self.z[h][s].value, np.zeros(self.na))
        return occupancy

    def extract_safe_policy_global_test(self):
        if self.max_expected_cost() <= self.cost_bound:
            self.extract_ground_policy()
        else:
            self.extract_abs_policy()

    def extract_safe_policy_local_test(self):
        for h in range(self.horizon):
            for abs_s in self.abs_states:
                if self.test_safety_locally(h, abs_s):
                    for s in self.states[self.abs_map[abs_s]]:
                        self.set_policy(h, s)
                else:
                    self.set_abs_policy(h, abs_s)

    def extract_ground_policy_for_worst_case(self):
        """
        this method searches for the largest cost_bound
        that yields a policy that is safe in all MDPs of the uncertainty set
        """
        while True:
            violation = max(self.max_expected_cost() - self.cost_bound, 0)
            if violation == 0:
                self.extract_ground_policy()
                return
            # reduce coefficient
            lr = 0.5
            self.cost_bound_coefficient -= lr * (violation / self.cost_bound)
            if self.verbose:
                print("new cost bound {}".format(self.cost_bound * self.cost_bound_coefficient))
            # TODO: here we could just adjust the bound on the cost function instead of re-instantiating the LP
            self.instantiate_lp()
            solved = self.solve_lp()
            if not solved:
                if self.verbose:
                    print("cost bound got too tight and the lp became infeasible")
                # assume problem is infeasible and return original abstract policy
                self.cost_bound_coefficient = 1
                self.instantiate_lp()
                self.solve_lp()
                self.extract_abs_policy()
                return


    def to_dot_abs(self, filename, form='pdf'):
        with open(filename, 'w') as out:
            out.write(self.get_graph_header())
            out.write(self.get_abs_graph_dot())
            out.write("}\n")
        self.try_draw(filename, form)

    def get_abs_graph_dot(self):
        res = ""
        for h in range(self.horizon):
            states_h = []
            for s in self.abs_states:
                outflow_s = 0
                inflow_s = 0
                for a in self.actions:
                    for t in self.abs_states:
                        if self.solver == 'grb':
                            outflow_s += self.z[h, s, a].x * self.abs_transition[s, a, t]
                            if h > 0:
                                inflow_s += self.z[h-1, t, a].x * self.abs_transition[t, a, s]
                        else:
                            outflow_s += self.z[h][s, a].value * self.abs_transition[s, a, t]
                            if h > 0:
                                inflow_s += self.z[h-1][t, a].value * self.abs_transition[t, a, s]
                if outflow_s < 0.000001 and inflow_s < 0.000001:
                    continue
                decoded_ground_s = list(self.env.decode(self.states[self.abs_map[s]][0]))
                decoded_s = "".join([str(decoded_ground_s[x]) for x in self.features])
                from_s = "h" + str(h) + "s" + str(s) + "f" + str(decoded_s)
                label = "abs: " + " ".join([str(decoded_ground_s[x]) for x in self.features])
                res += "\t{} [label=\"{}\"]\n".format(from_s, label)
                states_h.append(from_s)
                if h == self.horizon - 1:
                    continue
                for a in self.actions:
                    for t in self.abs_states:
                        if self.solver == 'grb':
                            o = self.z[h, s, a].x * self.abs_transition[s, a, t]
                        else:
                            o = self.z[h][s, a].value * self.abs_transition[s, a, t]
                        if o > 0.001:
                            decoded_ground_t = list(self.env.decode(self.states[self.abs_map[t]][0]))
                            decoded_t = "".join([str(decoded_ground_t[x]) for x in self.features])
                            to_t = "h" + str(h + 1) + "s" + str(t) + "f" + decoded_t
                            var = "\t{} -> {} [label=< a={} p={:.3f}>]\n".format(from_s, to_t, a, o)
                            res += var
            sh = ""
            for s in states_h:
                sh += "{}; ".format(s)
            res += "subgraph {rank=same; " + sh + "  } \n"
        return res

    def max_expected_cost(self):
        if self.solver == 'grb':
            return self.max_expected_cost_grb()
        else:
            return self.max_expected_cost_cvxpy()

    def max_expected_cost_grb(self):
        lp_test = Model('TestSafety')
        if not self.verbose:
            lp_test.Params.OutputFlag = 0

        if self.verbose:
            print("adding variables")
        x = lp_test.addVars(range(self.horizon), self.states, self.actions, lb=0.0, name='x')
        y = lp_test.addVars(range(self.horizon), self.states, self.actions, self.states, lb=0.0, name='y')

        if self.verbose:
            print("adding constraints")

        if self.horizon > 0:
            lp_test.addConstrs((
                quicksum(y[0, s, a, suc] for a in self.actions for suc in self.states) == self.isd[s]
                for s in self.states
            ), 'isd')

        lp_test.addConstrs((
            x[h, s, a] == quicksum(y[h, s, a, suc] for suc in self.states)
            for h in range(self.horizon) for s in self.states for a in self.actions
        ), 'x_is_sum_y')

        lp_test.addConstrs((
            self.x[h, s, a].x / sum(self.x[h, s, a2].x for a2 in self.actions)
            *
            quicksum(x[h, s, a2] for a2 in self.actions)
            ==
            x[h, s, a]
            for h in range(self.horizon) for s in self.states for a in self.actions
            if sum(self.x[h, s, a2].x for a2 in self.actions) > 0
        ), 'policy_ground_states_fixed')

        lp_test.addConstrs((
            y[h, s, a, suc] <= self.transition_ub[s, a, suc] * x[h, s, a]
            for h in range(self.horizon) for s in self.states for a in self.actions for suc in self.states
        ), 'transition_ub')
        lp_test.addConstrs((
            y[h, s, a, suc] >= self.transition_lb[s, a, suc] * x[h, s, a]
            for h in range(self.horizon) for s in self.states for a in self.actions for suc in self.states
        ), 'transition_lb')

        lp_test.addConstrs((
            quicksum(y[h, s, a, suc] for a in self.actions for suc in self.states)
            == quicksum(
                quicksum(
                    y[h-1, pred, a, s]
                    for pred in self.states
                ) for a in self.actions)
            for s in self.states[~self.terminal] for h in range(1, self.horizon)
        ), 'transient')

        lp_test.addConstrs((
            quicksum(
                y[h, s, a, suc]
                for a in self.actions for suc in self.states
            ) == 0
            for s in self.states[self.terminal] for h in range(self.horizon)
        ), 'terminal_flow')

        if self.verbose:
            print("setting objective")
        max_exp_cost = quicksum(
            x[h, s, a] * self.abs_cost[np.flatnonzero(self.abs_map.T[s])[0], a]
            for h in range(self.horizon) for s in self.states for a in self.actions
        )
        lp_test.setObjective(
            max_exp_cost,
            GRB.MAXIMIZE
        )

        lp_test.update()
        solve_gurobi_lp(lp_test, self.verbose)
        obj = lp_test.getObjective()
        if self.verbose:
            # self.to_dot('/tmp/rluc/flow_worst_case.dot', form='png')
            print(obj.getValue())
        return obj.getValue()

    def max_expected_cost_cvxpy(self):
        # variables
        # y is the occupancy on time step h of the tuple s,a,s'
        y = [
            [cv.Variable(shape=(self.na, self.ns), nonneg=True) for _ in self.states]
            for _ in range(self.horizon)
        ]
        # x is the occupancy on time step h of the tuple s,a
        x = [cv.Variable(shape=(self.ns, self.na), nonneg=True) for _ in range(self.horizon)]

        # expressions
        exp_cost = cv.Constant(0)

        for h in range(self.horizon):
            for s in self.states:
                exp_cost += cv.sum(cv.multiply(x[h][s], self.abs_cost[np.flatnonzero(self.abs_map.T[s])[0], :]))
        # objective
        obj = cv.Maximize(exp_cost)

        # constraints
        constraints = []
        for h in range(self.horizon):
            for s in self.states:
                # x is sum y
                constraints.append(x[h][s] == cv.sum(y[h][s], axis=1))
                if h > 0:
                    if self.terminal[s]:
                        # terminal flow
                        constraints += [cv.sum(y[h][s]) == 0]
                    else:
                        # transient
                        inflow = cv.Constant(0)
                        for t in self.states:
                            inflow += cv.sum(y[h - 1][t][:, s])
                        constraints += [inflow == cv.sum(y[h][s])]
                else:
                    # isd
                    constraints += [
                        # outflow == inflow (first time step)
                        cv.sum(y[0][s]) == self.isd[s]
                        for s in self.states
                    ]
                # trans ub
                constraints += [
                    y[h][s][a] <= x[h][s, a] * self.transition_ub[s, a]
                    for a in self.actions
                ]
                # trans lb
                constraints += [
                    y[h][s][a] >= x[h][s, a] * self.transition_lb[s, a]
                    for a in self.actions
                ]

                occupancy = self.get_occupancy(h, s)
                if occupancy.sum() > 0:
                    policy = occupancy / np.sum(occupancy)
                    # policy_ground_states_fixed
                    constraints += [
                        policy[a] * cv.sum(x[h][s])
                        ==
                        x[h][s, a]
                        for a in self.actions
                    ]
        # problem
        lp_test = cv.Problem(obj, constraints)
        lp_test.solve()


        if self.verbose:
            print(obj.value)
        return obj.value

    def test_safety_locally(self, time_step, abstract_state):
        lp_test = Model('TestSafetyLocal')
        if not self.verbose:
            lp_test.Params.OutputFlag = 0

        if self.verbose:
            print("adding variables")
        x = lp_test.addVars(range(self.horizon), self.states, self.actions, lb=0.0, name='x')
        y = self.y = lp_test.addVars(range(self.horizon), self.states, self.actions, self.states, lb=0.0, name='y')

        if self.verbose:
            print("adding constraints")

        if self.horizon > 0:
            lp_test.addConstrs((
                quicksum(y[0, s, a, suc] for a in self.actions for suc in self.states) == self.isd[s]
                for s in self.states
            ), "isd")

        lp_test.addConstrs((
            x[h, s, a] == quicksum(y[h, s, a, suc] for suc in self.states)
            for h in range(self.horizon) for s in self.states for a in self.actions
        ), "x_is_sum_y")

        lp_test.addConstr(
            sum(self.z[time_step, abstract_state, a].x for a in self.actions) ==
            quicksum(x[time_step, s, a] for s in self.states[self.abs_map[abstract_state]] for a in self.actions),
            "fix_flow_of_abs_states")

        lp_test.addConstrs((
            self.x[h, s, a].x / sum(self.x[h, s, a2].x for a2 in self.actions)
            * quicksum(x[h, s, a2] for a2 in self.actions)
            == x[h, s, a]
            for h in [time_step]
            for s in self.states[self.abs_map[abstract_state]]
            for a in self.actions
            if sum(self.x[h, s, a2].x for a2 in self.actions) > 0
        ), "policy_ground_states_fixed")

        lp_test.addConstrs((
            y[h, s, a, suc] <= self.transition_ub[s, a, suc] * x[h, s, a]
            for h in range(self.horizon) for s in self.states for a in self.actions for suc in self.states
        ), "transition_ub")
        lp_test.addConstrs((
            y[h, s, a, suc] >= self.transition_lb[s, a, suc] * x[h, s, a]
            for h in range(self.horizon) for s in self.states for a in self.actions for suc in self.states
        ), "transition_lb")

        lp_test.addConstrs((
            quicksum(y[h, s, a, suc] for a in self.actions for suc in self.states)
            == quicksum(
                quicksum(
                    y[h-1, pred, a, s]
                    for pred in self.states
                ) for a in self.actions)
            for s in self.states[~self.terminal] for h in range(1, self.horizon)
        ), "transient")

        lp_test.addConstrs((
            quicksum(
                y[h, s, a, suc]
                for a in self.actions for suc in self.states
            ) == 0
            for s in self.states[self.terminal] for h in range(self.horizon)
        ), "terminal_flow")

        if self.verbose:
            print("setting objective")
        max_exp_cost = quicksum(
            x[time_step, s, a] * self.abs_cost[np.flatnonzero(self.abs_map.T[s])[0], a]
            for s in self.states[self.abs_map[abstract_state]] for a in self.actions
        )
        lp_test.setObjective(
            max_exp_cost,
            GRB.MAXIMIZE
        )
        lp_test.update()
        solve_gurobi_lp(lp_test, self.verbose)
        obj = lp_test.getObjective()
        if self.verbose:
            self.to_dot('/tmp/rluc/flow_worst_case_{}_{}.dot'.format(time_step, abstract_state), form='png')
            print(time_step, abstract_state, obj.getValue(), self.get_exp_cost_abs_state(time_step, abstract_state))
        return obj.getValue() <= self.get_exp_cost_abs_state(time_step, abstract_state)

    def get_exp_cost_abs_state(self, time_step, abstract_state):
        return sum(
            self.abs_cost[abstract_state, a] * self.x[time_step, s, a].x
            for s in self.states[self.abs_map[abstract_state]]
            for a in self.actions
        )
