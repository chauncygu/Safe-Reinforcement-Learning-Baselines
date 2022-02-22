
from Abgabe.Neural_Network.Actor_Model import ActorModel
from Abgabe.Neural_Network.Critic_Model import CriticModel
from Abgabe.Training_RL.DDPG import DDPG_Agent
from Abgabe.Pre_training.constraints import Constraints
from Abgabe.Model.Linear_Env import LinearEnv
from Abgabe.Buffer.ReplayBuffer import ReplayBuffer

import numpy as np
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess
import matplotlib.pyplot as plt

from keras import backend as K
import tensorflow as tf
import time
import matplotlib as mpl
from Abgabe.Training_MPC.MPC import MPC
import scipy.io
import os

# plot definitions
mpl.rcParams['lines.linewidth'] = 4.5
mpl.rcParams['font.size'] = 24
mpl.rcParams['legend.fontsize'] = 'small'


"""----SET PARAMETERS------------------------------------------------------------------------------------------------"""

# The number of elements in the episodesTrain array define, over how many runs the Training is executed, the size of
# future_steps_tracing, future_steps_dist and ENV_NAME must be the same

episodesTrain = [5, 5, 5]
episodesTest = 1
stepsEpisodes = 100
stepsEpisodes_test = 100
future_steps_tracing = [0, 1, 1]  # number of steps the tracing trajectory is used from the future -> 0 = Nonw
buffersize = 10000

# disturbance parameter
path_dist = '../Disturbances/external_disturbances_uniform.mat'
path_dist_test = '../Disturbances/external_disturbances_old.mat'
disturbance = 0  # 0 -> no disturbance , 1 -> added disturbance
future_steps_dist = [1, 1, 0]  # number of steps the disturbance is used from the future -> 0 = None

# parameter of the noise process
sigma = 0.25
theta = 0.15
mu = 0

# Flag constraints
constraints = 'Rewardshaping'  # [None, SafetyLayer, Rewardshaping]

# Environmental details -> to be adapted for new model
Q = np.array([[10, 0], [0, 0.0000001]])
R = np.array([[0, 0], [0, 0]])

ENV_NAME = ['Test1', 'Test2', 'Test3']

x_init_T = 23
x_init_E = 0

# MPC parameters
do_MPC = True
N = 5  # Prediction horizon for MPC result

# Plotting information
label = ENV_NAME
color = ['blue', 'orange', 'yellow']

"""----Initialization------------------------------------------------------------------------------------------------"""
# create constraint function network
constraint_temp_up = None
constraint_energy_up = None
constraint_temp_low = None
constraint_energy_low = None

# to be adapted for new model
if constraints == 'SafetyLayer':
    constraint_temp_up = Constraints(path='../Pre_training/constraints_{}_weights.h5f'.format('test_T_up'))
    constraint_energy_up = Constraints(path='../Pre_training/constraints_{}_weights.h5f'.format('test_E_up'))
    constraint_temp_low = Constraints(path='../Pre_training/constraints_{}_weights.h5f'.format('test_T_low'))
    constraint_energy_low = Constraints(path='../Pre_training/constraints_{}_weights.h5f'.format('test_E_low'))
constraints_all = [constraint_temp_up, constraint_temp_low, constraint_energy_up, constraint_energy_low]

# define replay buffer
replayBuffer = ReplayBuffer(buffer_size=buffersize)

start = time.time()

"""---- LOOP --------------------------------------------------------------------------------------------------------"""
fig, ax = plt.subplots(3)
plt.title("Temperature Tracking")
count = 0
for number in episodesTrain:

    # Load Environment
    env = LinearEnv(disturbance=disturbance, nb_tracking=future_steps_tracing[count], Q=Q, R=R, path_dist=path_dist,
                    reward_shaping=constraints)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    nb_actions = env.action_space.shape[0]
    nb_observations = env.observation_space.shape[0]
    nb_disturbance = 3 * future_steps_dist[count]  # dim d x future steps

    # define exploration noise
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=theta, mu=mu, sigma=sigma)
    if constraints == 'SafetyLayer':
        random_process = GaussianWhiteNoiseProcess(size=nb_actions, mu=mu, sigma=sigma)
    # create Actor
    actor = ActorModel(sess, nb_observations, nb_actions, nb_disturbance+future_steps_tracing[count])
    # create Critic
    critic = CriticModel(sess, nb_observations, nb_actions, env, nb_disturbance+future_steps_tracing[count])

    """----FIT DDPG--------------------------------------------------------------------------------------------------"""
    # generate an DDPG object
    agent = DDPG_Agent(actor=actor, critic=critic, critic_action_input=critic.InputActions,
                       constraints_net=constraints_all, memory=replayBuffer, random_process=random_process, constraint=constraints,
                       nb_disturbance=nb_disturbance, nb_tracing=future_steps_tracing[count])

    print("DDPG object generated for run", count+1, "\nTraining starts...")

    # load weights of actor and critic if excisting
    if os.path.isfile('ddpg_{}_weights.h5f'.format(ENV_NAME)):
        agent.load_weights('ddpg_{}_weights.h5f'.format(ENV_NAME[count]))
    agent.compile()

    # train the agent
    reward, constraintviolations = agent.fit(env, nb_episodes=number, nb_max_episode_steps=stepsEpisodes)
    plot_val, episodes = agent.get_plot_val()

    print(constraintviolations)

    """----TEST DDPG-------------------------------------------------------------------------------------------------"""
    # Finally, we evaluate our algorithm for one episode
    # generate a new model object
    print("Testing the algorithm...")
    env = LinearEnv(disturbance=disturbance, nb_tracking=future_steps_tracing[count], Q=Q, R=R,
                    path_dist=path_dist_test)
    env.reset_states()

    # set inital state
    env.x = np.array([[x_init_T], [x_init_E]])

    # test run
    agent.test(env, nb_episodes=episodesTest, nb_steps_per_episode=stepsEpisodes_test)
    plt.subplot(311)
    plt.plot(env.T_plot[1:-1], label=label[count], color=color[count])

    # calculate error
    E_ref = env.E_ref_bat * np.ones((len(env.Ebat_plot),))
    error = 1/stepsEpisodes_test * sum((env.Ebat_plot - E_ref)**2)
    error2 = 1 / stepsEpisodes_test * sum((env.T_plot - env.T_ref_plot) ** 2)
    print('Test error Temperature (MSE) of run', count+1, ':', error)
    print('Test error Energy (MSE) of run', count+1, ':', error2)

    plt.subplot(312)
    plt.plot(env.Ebat_plot[1:-1], label=label[count], color=color[count])
    plt.subplot(313)
    plt.plot(env.u1_plot[1:-1], color='g')
    plt.plot(env.u2_plot[1:-1], color='y')

    agent.save_weights('ddpg_{}_{}_weights.h5f'.format(ENV_NAME[count], episodes), overwrite=True)
    if episodesTrain[count] > 1 and len(reward[1:-1]) > 0:
        print('Maximum Reward:', max(reward[1:-1]))
    print("Saving...")

    replayBuffer.erase()

    val = env.get_val()
    count += 1

end = time.time()
print("TIME in minutes:", (end - start)/60)

if do_MPC is True:
    """----DO MPC ---------------------------------------------------------------------------------------------------"""
    # MPC results
    dist = scipy.io.loadmat(path_dist_test)

    int_gains = dist['int_gains']
    room_temp = dist['room_temp']
    sol_rad = dist['sol_rad']
    S = episodesTest*stepsEpisodes_test

    doMPC = MPC(S, N, Q, R, dist=dist, dist_flag=1)

    ref = np.ones((2, S))
    ref[0, :] = ref[0, :]*21.5
    ref[1, :] = ref[1, :]*100000

    state = np.array([[20], [100000]])

    x_mpc, u_mpc = doMPC.mpc_step(ref, state, NN_flag=0)
    mpcT = x_mpc[:,0]
    error = 1 / 20 * sum((mpcT[40:60] - env.T_ref_plot[40:60]) ** 2)
    print('Error MPC', error)

    # plot MPC
    plt.subplot(311)
    plt.plot(x_mpc[:, 0], 'g--', label='MPC, N=5')
    plt.subplot(312)
    plt.plot(x_mpc[:, 1], 'g--', label='MPC, N=5')


"""----Plot results--------------------------------------------------------------------------------------------------"""
plt.subplot(311)
plt.plot(25 * np.ones((stepsEpisodes_test,)), '--', color='grey')
plt.plot(20 * np.ones((stepsEpisodes_test,)), '--', color='grey')
plt.plot(env.T_ref_plot[1:-1], label='Reference', color='red')
plt.grid()
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Temperature $[^\circ]$')
# plt.ylim(19.5,25.5)

plt.subplot(312)
plt.plot(0.1 * np.ones((stepsEpisodes_test,)), '--', color='grey')
plt.plot(200000 * np.ones((stepsEpisodes_test,)), '--', color='grey')
plt.plot(env.E_ref_bat * np.ones((stepsEpisodes_test,)), label='Reference', color='red')
plt.grid()
plt.xlabel('Steps')
plt.ylabel('Energy [mAh]')
# plt.ylim(-20000, 220000)

plt.subplot(313)
plt.plot(-1 * np.ones((stepsEpisodes_test,)), '--', color='grey')
plt.plot( np.ones((stepsEpisodes_test,)), '--', color='grey')
plt.plot(env.u1_plot[1:-1], label='Phvac', color='g')
plt.plot(env.u2_plot[1:-1], label='Pbat', color='y')
plt.legend()
plt.grid()
plt.xlabel('Steps')
plt.ylabel('Input')

plt.show()
