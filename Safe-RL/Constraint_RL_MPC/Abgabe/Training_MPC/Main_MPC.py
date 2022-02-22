"""
Main function to execute MPC
"""
from Abgabe.Neural_Network.NeuralNetwork import NN
import numpy as np
from Abgabe.Model.Linear_Env import LinearEnv
import matplotlib.pyplot as plt
from Abgabe.Training_MPC.MPC import MPC
import scipy.io
import matplotlib as mpl
from Abgabe.Normalize.MinMax import minmax_norm, minmax_norm_back
import time

# plot definitions
mpl.rcParams['lines.linewidth'] = 4.5
mpl.rcParams['font.size'] = 30
mpl.rcParams['legend.fontsize'] = 'medium'
"""----Initialization------------------------------------------------------------------------------------------------"""
num_samples = 5
num_episodes = 1

# network parameter
num_in = 7
num_out = 2
num_hidden = 10
activation = 'relu'
activation_out = 'linear'
optimizer = 'Adam'

# model parameter
Q = np.array([[1, 0], [0, 0.00000001]])
R = np.array([[0, 0], [0, 0]])

dist_flag = 1

path_dist = '../Disturbances/external_disturbances_old.mat'
env = LinearEnv(disturbance=dist_flag, nb_tracking=0, Q=Q, R=R, path_dist=path_dist)

states_new = np.array([21, 100000])
actions = np.array([0, 0])
states_old = np.array([21, 100000, 0, 0, 0])

path = 'SI_MPC_weights.h5f'
minmax_path = 'SI_MinMax.npy'
"""----Load minmax values--------------------------------------------------------------------------------------------"""
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

"""----Generate neural network---------------------------------------------------------------------------------------"""

network = NN(num_in, num_out, num_hidden, activation, activation, activation_out, optimizer, path)

"""----DO MPC--------------------------------------------------------------------------------------------------------"""
N = 5
S = 90

dist = scipy.io.loadmat(path_dist)


ref = np.ones((2,S+N+1))
ref[0, :] = ref[0, :]*22.5
# ref[0][45:-1] = ref[0][45:-1] + 4
ref[1, :] = ref[1, :]*100000

x_init = np.array([[20], [0]])

mpc_model = MPC(S, N, Q, R, dist, dist_flag)

states_model, actions_model = mpc_model.mpc_step(ref, x_init, 0, min_state1, min_state2,  min_dist1, min_dist2,
                                                 min_dist3, max_state1, max_state2, max_dist1, max_dist2, max_dist3)

ref = np.ones((2, S+N+1))
ref[0,:]=minmax_norm(22.5, min_state1, max_state1)
ref[1,:]=minmax_norm(100000, min_state2, max_state2)
x_init = np.array([[minmax_norm(20, min_state1, max_state1)], [minmax_norm(0, min_state2, max_state2)]])
mpc_network = MPC(S, N, Q, R, dist, dist_flag, network)
start = time.time()
states_network, actions_network = mpc_network.mpc_step(ref, x_init, 1, min_state1, min_state2, min_dist1, min_dist2,
                                                       min_dist3, max_state1, max_state2, max_dist1, max_dist2, max_dist3)

end = time.time()
print("TIME in minutes:", (end - start)/60)
plt.figure()
plt.subplot(311)
plt.plot(20 * np.ones((len(states_network),)), '--', color='grey')
plt.plot(25 * np.ones((len(states_network),)), '--', color='grey')
plt.plot(minmax_norm_back(states_network[:,0], min_state1, max_state1), label='Network')
plt.plot(states_model[:,0], 'g--', label='Model')
plt.ylabel('Temperature [C]')
plt.legend()
plt.grid()

plt.subplot(312)
plt.plot(0.1 * np.ones((len(states_network),)), '--', color='grey')
plt.plot(200000 * np.ones((len(states_network),)), '--', color='grey')
plt.plot(minmax_norm_back(states_network[:,1], min_state2, max_state2), label='Network')
plt.plot(states_model[:,1], 'g--', label='Model')
plt.ylabel('Energy [W]')
plt.grid()

plt.subplot(313)
plt.plot(abs(minmax_norm_back(states_network[:,0], min_state1, max_state1)-states_model[:,0]), label='Temp', color='darkred')
plt.plot(abs(minmax_norm_back(states_network[:,1], min_state2, max_state2)-states_model[:,1]), label='Energy', color='tomato')
plt.ylabel('Difference')
plt.xlabel('Steps')
plt.legend()
plt.grid()

plt.figure()
plt.subplot(311)
plt.plot(actions_network[:,0], label='Network')
plt.plot(actions_model[:,0], 'g--', label='Model')
plt.ylabel('Input P1')
plt.grid()

plt.subplot(312)
plt.plot(actions_network[:,1], label='Network')
plt.plot(actions_model[:,1],'g--', label='Model')
plt.legend()
plt.ylabel('Input P2')
plt.grid()

plt.subplot(313)
plt.plot(abs(actions_model[:,0] - actions_network[:,0]), label='diff P1', color='darkred')
plt.plot(abs(actions_model[:,1] - actions_network[:,1]), label='diff P2', color='tomato')
plt.ylabel('Difference')
plt.xlabel('Steps')
plt.legend()
plt.grid()

plt.show()
