import mdptoolbox
import numpy as np
from safemdp.grid_world import grid_world_graph


__all__ = ['reward_w_exp_bonus', 'reward_oracle', 'calc_opt_policy',
           'calculate_path', 'calculate_P_opti_pess']


def reward_oracle(x, true_reward):
    """ Calculate the reward that is nominal reward + exploration bonus

    Parameters
    ----------
    x: safemdp class
    true_reward: true reward 
        - This reward is defined for only states
        - Virtual state is NOT considered

    Returns
    -------
    reward_oracle: reward function used by the oracle agent
        - This reward is defined for state and action pairs
        - Virtual state is considered
    """
    n, m = x.world_shape

    # reward with virtual state
    reward_w_vs = np.zeros((n*m+1, 1))
    reward_w_vs[:-1, 0] = true_reward
    reward_w_vs[-1, 0] = 0

    G = grid_world_graph_w_stay(x)
    P = calculate_P_grid(G, x)

    reward_oracle = np.zeros((n*m+1, 5))
    ind_a, ind_s, ind_sp = np.where(P)
    for i in range(len(ind_a)):
        reward_oracle[ind_s[i], ind_a[i]] = reward_w_vs[ind_sp[i]]

    # For the virtual state, reward is set to very low
    reward_oracle[-1, :] = -1e5
    # reward_oracle[-1, :] = 0

    return reward_oracle


def reward_w_exp_bonus(safe_x, reward_x, R_max=100):
    """ Calculate the reward that is nominal reward + exploration bonus

    Parameters
    ----------
    safe_x: safemdp class
    reward_x: reward object
    gp_reward: GP for reward
    R_max: maximum value of reward
    beta_reward: scaling factor of confidence interval for reward

    Returns
    -------
    reward_w_bonus: nominal reward + exploration bonus
    """
    beta_reward = reward_x.beta

    n, m = safe_x.world_shape
    r_w_eb = np.zeros((n*m+1, 5))

    # Upper bound of reward
    mu_reward = reward_x.gp.predict(safe_x.coord)[0]
    var_reward = reward_x.gp.predict(safe_x.coord)[1]
    std_reward = var_reward**0.5

    u_reward = np.zeros((n*m+1, 1))
    u_reward[:-1] = mu_reward + beta_reward*std_reward[:]
    u_reward[-1] = 0

    G = grid_world_graph_w_stay(safe_x)
    P = calculate_P_grid(G, safe_x)

    ind_a, ind_s, ind_sp = np.where(P)
    for i in range(len(ind_a)):
        r_w_eb[ind_s[i], ind_a[i]] = min(R_max, u_reward[ind_sp[i]])

    # For the virtual state, reward is set to very low
    r_w_eb[-1, :] = -1e5
    # r_w_eb[-1, :] = 0

    return r_w_eb


def grid_world_graph_w_stay(x):
    """Create a graph that represents a grid world.

    In the grid world there are five actions, (0, 1, 2, 3, 4), which correspond
    to going (stay, up, right, down, left) in the x-y plane.

    Parameters
    ----------
    world_size: tuple
        The size of the grid world (rows, columns)

    Returns
    -------
    graph: nx.DiGraph()
        The directed graph representing the grid world.
    """
    world_size = x.world_shape
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    # grid world graph without "stay" action
    graph = grid_world_graph(world_size)

    # action 0: stay (not move)
    graph.add_edges_from(zip(grid_nodes[:, :].reshape(-1),
                             grid_nodes[:, :].reshape(-1)),
                         action=0)

    return graph


def calculate_P_grid(G, x):
    """ To calculate the value function in MDP, the transition probability
    must be stochastic. This function slightly modifies the graph so that the
    transition probability is strochastic especially for the boundary.

    Action: 0 (stay), 1 (up), 2 (right), 3 (down), 4 (left)
    """
    n, m = x.world_shape
    P = np.zeros((5, n*m+1, n*m+1))

    for i in range(n*m):
        action_list = [0, 1, 2, 3, 4]
        for node, neighbor_node, tmp_a in G.edges_iter(i, data='action'):
            action_list.remove(tmp_a)
            P[tmp_a, node, neighbor_node] = 1

        for j in range(len(action_list)):
            if action_list[j] == 0:
                P[action_list[j], i, i] = 1
            else:
                P[action_list[j], i, -1] = 1

    return P


def calculate_bin_P(P, x, cal_type='pes'):
    """ Calculate the virtual, binary transition function.
    That is, this function is to calculate the transition function which
    a state and action pair may visit the virtual state $z$
    """
    n, m = x.world_shape
    # P_z is defined for the n*m states $s$ and a virtual state $z$
    # index 0 - n*m-1: real state
    #             n*m: virtual state
    P_z = np.zeros((5, n*m+1, n*m+1))
    ind_a, ind_s, ind_sp = np.where(P)

    if cal_type == 'pes':
        safe_space = x.S_hat
    elif cal_type == 'opt':
        safe_space = x.S_bar

    for i in range(len(ind_a)):
        if safe_space[ind_s[i], ind_a[i]]:
            P_z[ind_a[i], ind_s[i], ind_sp[i]] = 1
        else:
            P_z[ind_a[i], ind_s[i], -1] = 1

    # For any action, transition probability from z to z is equal to 1
    P_z[:, -1, -1] = 1

    return P_z


def calculate_cont_P(P, p_safe, world_size):
    """ Calculate the virtual transition function. That is, this function is
    to calculate the transition function which a state and action pair may
    visit the virtual state $z$
    """
    n, m = world_size
    # P_z is defined for the n*m states $s$ and a virtual state $z$
    # index 0 - n*m-1: real state
    #             n*m: virtual state
    P_z = np.zeros((5, n*m+1, n*m+1))
    ind_a, ind_s, ind_sp = np.where(P)

    for i in range(len(ind_a)):
        P_z[ind_a[i], ind_s[i], ind_sp[i]] = p_safe[ind_s[i], ind_a[i]]
        P_z[ind_a[i], ind_s[i], -1] = 1 - p_safe[ind_s[i], ind_a[i]]

    # For any action, transition probability from z to z is equal to 1
    P_z[:, -1, -1] = 1

    return P_z


def calculate_path(x, V, H, source, next_pos):
    opt_path = [source, next_pos]
    G = grid_world_graph_w_stay(x)
    for i in range(H-2):
        _set_neighbors = []
        for _, neighbor_node, _ in G.edges_iter(next_pos, data='action'):
            _set_neighbors.append(neighbor_node)

        ind_neighbor_maxV = np.argmax(V[_set_neighbors, 1])
        next_pos = _set_neighbors[ind_neighbor_maxV]
        opt_path.append(next_pos)

    return opt_path


def check_safe_exp_exit_cond(pes_safe_x, opt_safe_x, reward_x, args,
                             bin_tran=True):
    """ To check if the exit condition for exploration of safety is satisfied.

    Parameters
    ----------
    world_shape: tuple
        The size of the grid world (rows, columns)
    gamma: discounted factor
    horizon: (finite) horizon
    R: reward function: array

    Note: x.S_hat represents the safe space characterized by GPs
    """
    G = grid_world_graph_w_stay(pes_safe_x)
    P = calculate_P_grid(G, pes_safe_x)

    n, m = pes_safe_x.world_shape
    R = np.zeros((n*m+1, 5))
    ind_a, ind_s, ind_sp = np.where(P)

    coord = pes_safe_x.coord
    mu_r = reward_x.gp.predict(coord)[0]
    var_reward = reward_x.gp.predict(coord)[1]
    std_reward = var_reward**0.5

    upper_reward = mu_r + reward_x.beta*std_reward
    lower_reward = mu_r - reward_x.beta*std_reward

    for i in range(len(ind_a)):
        if ind_sp[i] < n*m:
            if opt_safe_x.S_bar[ind_s[i], ind_a[i]]:
                R[ind_s[i], ind_a[i]] = upper_reward[ind_sp[i]]
            if pes_safe_x.S_hat[ind_s[i], ind_a[i]]:
                R[ind_s[i], ind_a[i]] = lower_reward[ind_sp[i]]

    R[-1, :] = -100

    # Transition probability in optimistic safe space
    if bin_tran:
        P_z = calculate_bin_P(P, opt_safe_x, cal_type='opt')
    else:
        p_safe = pes_safe_x.probability_safe()
        P_z = calculate_cont_P(P, p_safe, pes_safe_x.world_shape)

    pi = mdptoolbox.mdp.PolicyIteration(P_z, R, args.gamma)
    pi.run()
    policy = pi.policy

    safe_states = np.where(pes_safe_x.S_hat[:, 0])[0]
    cnt_safe, cnt_unsafe = 0, 0
    for ss in safe_states:
        next_s = np.where(P[int(policy[ss]), int(ss), :])[0][0]
        if next_s in safe_states:
            cnt_safe += 1
        else:
            cnt_unsafe += 1

    if cnt_unsafe == 0:
        return True
    else:
        return False


def calc_opt_policy(x, R, pos, args, bin_tran=True):
    """ To calculate action and next state by solving infinite-horizon MDP.

    Parameters
    ----------
    x: (Pessimistic) SafeMDP object
    R: Reward function: array
    pos: current position
    p_safe: probability of state $s$ being safe
    args: list of arguments
    """
    gamma = args.gamma

    G = grid_world_graph_w_stay(x)
    P = calculate_P_grid(G, x)

    if bin_tran:
        P_z = calculate_bin_P(P, x, cal_type='pes')
    else:
        p_safe = x.probability_safe()
        P_z = calculate_cont_P(P, p_safe, x.world_shape)

    if args.mdp_alg == 'vi':
        vi = mdptoolbox.mdp.ValueIteration(P_z, R, gamma)
        vi.run()
        V, pi = vi.V, vi.policy
    elif args.mdp_alg == 'pi':
        pi = mdptoolbox.mdp.PolicyIteration(P_z, R, gamma)
        pi.run()
        V, pi = pi.V, pi.policy

    next_action = pi[pos]
    next_state = np.where(P[next_action, pos, :])[0][0]

    return (next_state, next_action), V, pi, P_z


def calculate_P_opti_pess(P, pes_x, opt_x):
    """ Calculate the virtual transition function. That is, this function is to
    calculate the transition function which a state and action pair may visit
    the virtual state $z$
    """
    n, m = pes_x.world_shape
    # P_z is defined for the n*m states $s$ and a virtual state $z$
    # index 0 - n*m-1: real state
    #             n*m: virtual state
    P_pess = np.zeros((5, n*m+1, n*m+1))
    P_opti = np.zeros((5, n*m+1, n*m+1))
    ind_a, ind_s, ind_sp = np.where(P)

    for i in range(len(ind_a)):
        if pes_x.S_hat[ind_s[i], ind_a[i]]:
            P_pess[ind_a[i], ind_s[i], ind_sp[i]] = 1
            P_pess[ind_a[i], ind_s[i], -1] = 0
        else:
            P_pess[ind_a[i], ind_s[i], ind_sp[i]] = 0
            P_pess[ind_a[i], ind_s[i], -1] = 1
    P_pess[:, -1, -1] = 1

    for i in range(len(ind_a)):
        if opt_x.S_hat[ind_s[i], ind_a[i]]:
            P_opti[ind_a[i], ind_s[i], ind_sp[i]] = 1
            P_opti[ind_a[i], ind_s[i], -1] = 0
        else:
            P_opti[ind_a[i], ind_s[i], ind_sp[i]] = 0
            P_opti[ind_a[i], ind_s[i], -1] = 1
    P_opti[:, -1, -1] = 1

    return P_pess, P_opti
