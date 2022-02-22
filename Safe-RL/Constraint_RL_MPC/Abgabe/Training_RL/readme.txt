
##### Main_RL.py ##################################################################

Training phase and test phase of the DDPG algorithm
it can be evaluated together with MPC, if enabled


PARAMETER:

num_samples =  number of samples per episode
num_episodes =  number of episodes

episodesTrain =  number of episodes for the training
episodesTest =  number of episodes for the test
stepsEpisodes =  number of samples per episode training
stepsEpisodes_test = number of samples per episode training

future_steps_tracing = number of steps the tracing trajectory is used from the future -> 0 = Nonw
buffersize = size of replay buffer

disturbance =  0 -> no disturbance , 1 -> added disturbance
future_steps_dist = number of steps the disturbance is used from the future -> 0 = None

# parameter of the noise process
sigma
theta
mu 

constraints = Flag constraints [None, SafetyLayer, Rewardshaping]

# Environmental details
Q 
R 
ENV_NAME = Name where the weights are saved


# MPC parameters
do_MPC = Flag whether MPC should be evaluated
N =  Prediction horizon for MPC result

OUTPUT:

network weights of the trained network is saved in the same folder
evolution of states and inputs is ploted


