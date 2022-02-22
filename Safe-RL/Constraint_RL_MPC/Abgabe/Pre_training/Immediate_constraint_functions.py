"""
Generate neural network, which describes the immediate constraint function ci(x,u)
"""
import numpy as np
from Abgabe.Model.Linear_Env import LinearEnv
import os
from Abgabe.Neural_Network.NeuralNetwork import NN
from Abgabe.Normalize.MinMax import minmax_norm

"""----SET PARAMETERS------------------------------------------------------------------------------------------------"""
num_samples = 1  # number of samples per episode
num_episodes = 100  # number of episodes

# define network parameters
num_in = 3
num_out = 1
num_hidden = 10
activation = 'tanh'
activation_out = 'linear'
optimizer = 'Adam'

# here, the states to be constrained are defined
# define safety signal
# 0-> Temperature low, 1-> Energy low,  2-> Temperature up, 3-> Energy up
state_flag = 3
if state_flag == 0:
    ENV_NAME = 'test_T_low'
    state = 0
    div = 21
elif state_flag == 1:
    ENV_NAME = 'test_E_low'
    state = 1
    div = 200000
elif state_flag == 2:
    ENV_NAME = 'test_T_up'
    state = 0
    div = 21
elif state_flag == 3:
    ENV_NAME = 'test_E_up'
    state = 1
    div = 200000

# define paths
path_dist = '../Disturbances/external_disturbances_old.mat'
path = 'constraints_{}_weights.h5f'.format(ENV_NAME)

# load environment
env = LinearEnv(path_dist=path_dist)


"""----GENERATE DATASET----------------------------------------------------------------------------------------------"""
# generate D : random state, action, next state pairs

# initialize arrays to store transitions
states_new = np.array([div])
actions_train = np.array([0, 0])
states_old = np.array([div])

# loop over all episodes and all samples per episode
for i in range(num_episodes):
    _ = env.reset_states()
    old_state = env.x

    for j in range(num_samples):
        # randomly sample action
        action1 = np.random.uniform(-1, 1)
        action2 = np.random.uniform(-1, 1)

        action = np.array([action1, action2])

        # apply action to the environment
        new_state, _ = env.step(action)

        # terminate episode if new state out of bounds -> to be changed for new model
        if state_flag == 1 and env.x[1][0] < 0:
            env.x[1][0] = 0
            print("under")
            break
        elif state_flag == 3 and env.x[1][0] > 200000:
            env.x[1][0] = 200000
            print("over")
            break

        elif state_flag == 0 and env.x[0][0] < 20:
            env.x[0][0] = 20
            print("under")
            break

        elif state_flag == 2 and env.x[0][0] > 25:
            env.x[0][0] = 25
            print("over")
            break

        # save action, old state and new state
        states_new = np.vstack([states_new, [env.x[state, 0]]])
        states_old = np.vstack([states_old, [old_state[state, 0]]])
        actions_train = np.vstack([actions_train, action])
        old_state = env.x


"""----NORMALIZE DATASET---------------------------------------------------------------------------------------------"""
# max min normalization of data x = (x-min)/(max-min)

states_old_norm = minmax_norm(states_old, min(states_old), max(states_old))
states_new_norm = minmax_norm(states_new, min(states_new), max(states_new))

"""----GENERATE NEURAL NETWORK---------------------------------------------------------------------------------------"""
# generate and train neural network which describes c(s,a)
# inintalize neural network
network = NN(num_in, num_out, num_hidden, activation, activation, activation_out, optimizer)

# load weights if they allready exist
if os.path.isfile('constraints_{}_weights.h5f'.format(ENV_NAME)):
    path = 'constraints_{}_weights.h5f'.format(ENV_NAME)
    network.model.load_weights(path)
    print("load weights")

# train constraint function
inputs = np.concatenate((states_old_norm, actions_train), axis=1)
network.model.fit(inputs, states_new_norm, batch_size=64, epochs=20)

# save network parameters
network.model.save_weights(path)



