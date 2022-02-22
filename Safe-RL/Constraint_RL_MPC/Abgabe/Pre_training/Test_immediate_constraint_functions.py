"""
Generate an optimization problem, that minimizes the euclidean norm between the old action and the new,
with subject to the generated constraint function
"""

from casadi import *
from Abgabe.Model.Linear_Env import LinearEnv
import matplotlib.pyplot as plt
from Abgabe.Neural_Network.NeuralNetwork import NN
import matplotlib as mpl
from Abgabe.Normalize.MinMax import minmax_norm


mpl.rcParams['lines.linewidth'] = 4.5
mpl.rcParams['font.size'] = 30
mpl.rcParams['legend.fontsize'] = 'medium'

'''--- Load neural networks -----------------------------------------------------------------------------------------'''
num_in = 3
num_out = 1
num_hidden = 10
activation = 'tanh'
activation_out = 'linear'
optimizer = 'Adam'

# neural network for temperature lower bound
path = 'constraints_{}_weights.h5f'.format('test_T_low')

constraint_T_low = NN(num_in, num_out, num_hidden, activation, activation, activation_out, optimizer, path)

# neural network for temperature upper bound
path = 'constraints_{}_weights.h5f'.format('test_T_up')

constraint_T_up = NN(num_in, num_out, num_hidden, activation, activation, activation_out, optimizer, path)

# neural network for energy lower bound
path = 'constraints_{}_weights.h5f'.format('test_E_low')

constraint_E_low = NN(num_in, num_out, num_hidden, activation, activation, activation_out, optimizer, path)

# neural network for energy upper bound
path = 'constraints_{}_weights.h5f'.format('test_E_up')

constraint_E_up = NN(num_in, num_out, num_hidden, activation, activation, activation_out, optimizer, path)


'''--- Define optimization problem ----------------------------------------------------------------------------------'''
# init
na = 2
param = MX.sym('param', 1, 4)
a = MX.sym('a', na, 1)
G = []

# define cost function J
J = 0.5*((sqrt(a[0]-param[0,0]))**2)**2 + 0.5*((sqrt(a[1]-param[0,1]))**2)**2

# convert keras structure to be used in symbolic casadi framework
constraints_T_low = constraint_T_low.nn_casadi(constraint_T_low.weights, constraint_T_low.config, horzcat(param[0,2],a[0], a[1]))
constraints_T_up = constraint_T_up.nn_casadi(constraint_T_up.weights, constraint_T_up.config, horzcat(param[0,2], a[0], a[1]))
constraints_E_low = constraint_E_low.nn_casadi(constraint_E_low.weights, constraint_E_low.config, horzcat(param[0,3],a[0], a[1]))
constraints_E_up = constraint_E_up.nn_casadi(constraint_E_up.weights, constraint_E_up.config, horzcat(param[0,3], a[0], a[1]))

G.append(constraints_T_low)
G.append(constraints_T_up)
G.append(constraints_E_low)
G.append(constraints_E_up)

# define upper and lower bounds for the inputs and constraint functions
lbax = np.array([[-1], [-1]])
ubax = np.array([[1], [1]])

ubg = np.array([[inf], [0.99], [inf], [0.99]])
lbg = np.array([[0.01], [-inf], [0.], [-inf]])

opts = {}
# opts["ipopt.tol"] = 1e-5
opts["ipopt.print_level"] = 2
opts["print_time"] = 0

nlp = {'x': vertcat(a), 'f': J, 'g': horzcat(*G), 'p':  param}
solver = nlpsol('solver', 'ipopt', nlp, opts)


'''--- Simulate system ----------------------------------------------------------------------------------------------'''
num_sim = 300

# load environment
path_dist = '../Disturbances/external_disturbances_old.mat'
env = LinearEnv(path_dist=path_dist)
env.reset_states()
env_original = LinearEnv(path_dist=path_dist)
env_original.x = env.x

# initialize arrays to save data
state_old = np.copy(env.x)
save_temp = []
save_engy = []
save_temp_original = []
save_engy_original = []
save_temp_old = []
save_engy_old = []

actions_new_0 = []
actions_new_1 = []
actions_new_2 = []

actions_old_1 = []
actions_old_2 = []
actions_old_3 = []

np.random.seed(123)

# counter to determine the number of constraint violations
num_constraint_t = 0
num_constraint_e = 0

num_unconstraint_t = 0
num_unconstraint_e = 0

for i in range(num_sim):
    # sample new random action
    env.reset_states()
    env_original.x = np.copy(env.x)
    state_old = np.copy(env.x)

    action1 = np.random.uniform(0, 1)
    action2 = np.random.uniform(-1, 1)
    action = np.array([action1, action2])

    state_old[0, 0] = minmax_norm(state_old[0, 0], 20, 25)
    state_old[1, 0] = minmax_norm(state_old[1, 0], 0, 200000)

    param = np.reshape(np.concatenate((action, np.reshape(state_old, (1, 2))[0])), (1, 4))

    # with the old state and the random input, a new projected input is obtained
    res = solver(lbx=lbax, ubx=ubax, lbg=lbg, ubg=ubg, p=param)  # p = mu
    new_action = res['x']

    # apply new projected and old input to the system
    _, _ = env.step(new_action)
    _, _ = env_original.step(action)

    # count the number of constraint violations
    if env_original.x[0, 0] > 25 or env_original.x[0, 0]<20:
        num_unconstraint_t += 1
    if env_original.x[1, 0] > 200000 or env_original.x[1, 0]<0:
        num_unconstraint_e += 1
    if env.x[0, 0] > 25 or env.x[0, 0] < 20:
        num_constraint_t += 1
    if env.x[1, 0] > 200000 or env.x[1, 0] < 0:
        num_constraint_e += 1

    # store data
    save_temp_old.append(state_old[0, 0])
    save_engy_old.append(state_old[1, 0])
    save_temp.append(env.x[0, 0])
    save_engy.append(env.x[1, 0])
    save_temp_original.append(env_original.x[0, 0])
    save_engy_original.append(env_original.x[1, 0])
    actions_new_0.append(float(new_action[0]))
    actions_new_1.append(float(new_action[1]))

    actions_old_1.append(action1)
    actions_old_2.append(action2)


print('Num constraint errors T', num_unconstraint_t, '/',  num_constraint_t, '\n'
      'Num constraint errors E', num_unconstraint_e, '/',  num_constraint_e)

'''---Plot ----------------------------------------------------------------------------------------------------------'''
plot_T_in = []
index_T_in = []

plot_T_out = []
index_T_out = []

plot_E_in = []
index_E_in = []

plot_E_out = []
index_E_out = []
for i in range(num_sim):
    if save_temp_original[i] >= 20 and save_temp_original[i] <= 25:
        plot_T_in.append(save_temp_original[i])
        index_T_in.append(i)
    else:
        plot_T_out.append(save_temp_original[i])
        index_T_out.append(i)

    if save_engy_original[i] > 0 and save_engy_original[i] < 200000:
        plot_E_in.append(save_engy_original[i])
        index_E_in.append(i)
    else:
        plot_E_out.append(save_engy_original[i])
        index_E_out.append(i)

my_color = np.where((np.asarray(save_temp_original) >= 20) & (np.asarray(save_temp_original )<= 25), 'green', 'red')
x = np.linspace(0, num_sim, num_sim)

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(x, save_temp,  s=100, label='Constraint', zorder=-1, color='blue')
plt.scatter(index_T_in, plot_T_in, s=60, label='Original In', zorder=1,  color='green')
plt.scatter(index_T_out, plot_T_out, s=60, label='Original Out', zorder=1,  color='red')
plt.plot(20 * np.ones((len(save_temp),)), '--', color='grey')
plt.plot(25 * np.ones((len(save_temp),)), '--', color='grey')
plt.legend()
plt.ylabel('Temperature')
plt.xlim(0,num_sim)

my_color = np.where((np.asarray(save_engy_original) >= 0) & (np.asarray(save_engy_original )<= 200000), 'green', 'red')
plt.subplot(2, 1, 2)
plt.scatter(x, save_engy, s=100, label='Constraint', zorder=-1, color='blue')
plt.scatter(index_E_in, plot_E_in, s=60, label='Original Out', zorder=1,  color='green')
plt.scatter(index_E_out, plot_E_out, s=60, label='Original In', zorder=1,  color='red')
plt.plot(0.0001 * np.ones((len(save_engy),)),  '--', color='grey')
plt.plot(200000 * np.ones((len(save_engy),)),  '--', color='grey')
plt.xlabel('Steps')
plt.ylabel('Energy')
plt.ylim(-10000, 210000)
plt.xlim(0, num_sim)

plt.figure()
plt.subplot(211)
plt.plot(actions_new_0,  label='Phvac new')
plt.plot(actions_old_1, label='Phvac old')
plt.subplot(212)
plt.plot(actions_new_1, label='Phvac new')
plt.plot(actions_old_2, label='Phvac old')
plt.legend()
plt.show()

