"""
Main function to execute the system identification
"""

from Abgabe.Neural_Network.NeuralNetwork import NN
import numpy as np
from Abgabe.Model.Linear_Env import LinearEnv
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from Abgabe.Normalize.MinMax import minmax_norm, minmax_norm_back
from matplotlib import rc
import time
rc('font',**{'family':'serif','serif':['Computer Modern']})

rc('text', usetex=True)
# plot definitions
mpl.rcParams['lines.linewidth'] = 4.5
mpl.rcParams['font.size'] = 30
mpl.rcParams['legend.fontsize'] = 'medium'
"""----Initialization------------------------------------------------------------------------------------------------"""
num_samples = 1
num_episodes = 500

# network parameter
num_hidden = 10
activation = 'relu'
activation_out = 'linear'
optimizer = 'Adam'

# model paramter
Q = np.array([[1, 0], [0, 0.00000001]])
R = np.array([[0, 0], [0, 0]])

dist_flag = 1

path_dist = '../Disturbances/external_disturbances_old.mat'
path = 'SI_MPC_weights.h5f'
minmax_path = 'SI_MinMax'

env = LinearEnv(disturbance=dist_flag, nb_tracking=0, Q=Q, R=R, path_dist=path_dist)

# init arrays to save the data
states_new = np.array([21, 100000])
actions = np.array([0, 0])
states_old = np.array([21, 100000])
disturbances = np.array([0, 0, 0])


"""----generate D : random state, action, next state pairs-----------------------------------------------------------"""
print('start generate data:')
for i in range(num_episodes):
    _ = env.reset_states()
    old_state = env.x.T
    #old_state = env.x.T

    for j in range(num_samples):
        # randomly sample action
        action1 = np.random.uniform(-1, 1)
        action2 = np.random.uniform(-1, 1)
        action = np.array([action1, action2])

        dist1 = np.random.uniform(0, 2.5)*dist_flag
        dist2 = np.random.uniform(0, 80)*dist_flag
        dist3 = np.random.uniform(0, 2)*dist_flag
        disturbance = np.array([[dist1], [dist2], [dist3]])
        env.d = disturbance

        new_state, _ = env.step(action)

        # save action, old state, new state and disturbance
        states_new = np.vstack([states_new,  env.x.T])
        states_old = np.vstack([states_old, old_state])
        actions = np.vstack([actions, action])
        disturbances = np.vstack([disturbances, disturbance.T])

        old_state = env.x.T
print('end generate data.')

"""----normalize D --------------------------------------------------------------------------------------------------"""
# max min normalization of data (x-min)/(max-min)
states_old_norm = np.copy(states_old)
states_new_norm = np.copy(states_new)
disturbances_norm = np.copy(disturbances)

# Load data of min and max values if they are saved
if os.path.isfile(minmax_path):
    minmax_values = np.load(minmax_path)
    min_state1 = minmax_values[0]
    max_state1 = minmax_values[1]

    min_state2 = minmax_values[2]
    max_state2 = minmax_values[3]

    min_dist1 = minmax_values[4]
    max_dist1 = minmax_values[5]
    min_dist2 = minmax_values[6]
    max_dist2 = minmax_values[7]
    min_dist3 = minmax_values[8]
    max_dist3 = minmax_values[9]
else:

    min_state1 = min(states_old[:, 0])
    max_state1 = max(states_old[:,0])

    min_state2 = min(states_old[:, 1])
    max_state2 = max(states_old[:, 1])

    min_dist1 = min(disturbances[:, 0])
    max_dist1 = max(disturbances[:, 0])
    min_dist2 = min(disturbances[:, 1])
    max_dist2 = max(disturbances[:, 1])
    min_dist3 = min(disturbances[:, 2])
    max_dist3 = max(disturbances[:, 2])

states_old_norm[:, 0] = minmax_norm(states_old_norm[:, 0], min_state1, max_state1)
states_new_norm[:, 0] = minmax_norm(states_new_norm[:, 0], min_state1, max_state1)
states_old_norm[:, 1] = minmax_norm(states_old_norm[:, 1], min_state2, max_state2)
states_new_norm[:, 1] = minmax_norm(states_new_norm[:, 1], min_state2, max_state2)

if dist_flag != 0:

    disturbances_norm[:, 0] = minmax_norm(disturbances_norm[:, 0], min_dist1, max_dist1)
    disturbances_norm[:, 1] = minmax_norm(disturbances_norm[:, 1], min_dist2, max_dist2)
    disturbances_norm[:, 2] = minmax_norm(disturbances_norm[:, 2], min_dist3, max_dist3)


"""----Split into test and trainigs data-----------------------------------------------------------------------------"""
[train_states_new, test_states_new] = np.split(states_new_norm, [round(num_episodes*num_samples*0.9), ])
[train_states_old, test_states_old] = np.split(states_old_norm, [round(num_episodes*num_samples*0.9), ])
[train_actions, test_actions] = np.split(actions, [round(num_episodes*num_samples*0.9), ])
[train_disturbances, test_disturbances] = np.split(disturbances_norm, [round(num_episodes*num_samples*0.9), ])

train_in = np.concatenate((train_states_old, train_disturbances, train_actions), axis=1)
train_out = train_states_new

test_in = np.concatenate((test_states_old, test_disturbances, test_actions), axis=1)
test_out = test_states_new


"""----Generate and train neural network-----------------------------------------------------------------------------"""
num_in = train_in.shape[1]
num_out = train_out.shape[1]

if os.path.isfile(path):
    network = NN(num_in, num_out, num_hidden, activation, activation, activation_out, optimizer, path)

else:
    network = NN(num_in, num_out, num_hidden, activation, activation, activation_out, optimizer)
start = time.time()


history = network.model.fit(train_in, train_out,  batch_size=64, epochs=200)
cost = history.history['loss']
end = time.time()
print("TIME in minutes:", (end - start)/60)
"""----Test neural network---------------------------------------------------------------------------------------"""
predicted_states = network.model.predict(test_in)
error1 = 1/len(predicted_states[:,0])*np.sum((minmax_norm_back(predicted_states[:,0], min_state1, max_state1) -
                                              minmax_norm_back(test_states_new[:,0], min_state1, max_state1))**2)
error2 = 1/len(predicted_states[:,1])*np.sum((minmax_norm_back(predicted_states[:,1], min_state2, max_state2) -
                                              minmax_norm_back(test_states_new[:,1], min_state2, max_state2))**2)

print(error1, error2) # 0.0012885619312584568 0.00013151326324537213
steps = np.linspace(0, len(predicted_states) -1, len(predicted_states))
fig, _ = plt.subplots()

plt.subplot(311)
plt.scatter(steps, abs((minmax_norm_back(predicted_states[:,0], min_state1, max_state1) -
                                              minmax_norm_back(test_states_new[:,0], min_state1, max_state1))), s=20)
plt.grid()
plt.ylabel('Diff $T_r$')
plt.subplot(312)
plt.scatter(steps, abs(minmax_norm_back(predicted_states[:,1], min_state2, max_state2) -
                                              minmax_norm_back(test_states_new[:,1], min_state2, max_state2)), s=20)
plt.ylabel('Diff $E_{bat}$')
plt.xlabel('Samples')
plt.grid()
plt.subplot(313)
plt.semilogy(cost)
plt.grid()
plt.ylabel('Cost')
plt.xlabel('Epochs')
fig.tight_layout()
plt.show()
"""----Save neural network---------------------------------------------------------------------------------------"""
network.model.save_weights(path)
np.save(minmax_path, np.array([min_state1, max_state1, min_state2, max_state2, min_dist1, max_dist1, min_dist2,
                               max_dist2, min_dist3, max_dist3]))
