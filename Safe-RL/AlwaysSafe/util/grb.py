try:
    from gurobipy import Model, quicksum, GRB, GurobiError
    GUROBI_FOUND = True
except ModuleNotFoundError as e:
    GUROBI_FOUND = False


def solve_gurobi_lp(model, verbose=False, check_if_infeasible=False):
    if not verbose:
        model.Params.OutputFlag = 0
    model.optimize()

    if model.status == GRB.Status.INF_OR_UNBD:
        # Turn presolve off to determine whether model is infeasible or unbounded
        model.setParam(GRB.Param.Presolve, 0)
        model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        # model.write('model.lp')
        # model.write('model.sol')
        if verbose:
            print('Optimal objective: {}'.format(model.objVal))
        return model
    elif model.status == GRB.Status.UNBOUNDED:
        model.write('model_unbounded.lp')
        raise GurobiError(model.status,
                          'Optimization stopped (UNBOUNDED), check the file model_unbounded.lp')
    elif model.status == GRB.Status.INFEASIBLE:
        if check_if_infeasible:
            model.write('model_infeasible.lp')
            model.computeIIS()
            model.write("model.ilp")
            raise GurobiError(model.status,
                              'Optimization stopped (INFEASIBLE), check files model_infeasible.lp and model.ilp')
    return model
