
##### Main_System_Identification.py ################################################

trains neural network for system identification of the model
has to be evaluated first

PARAMETER:

num_samples =  number of samples per episode
num_episodes =  number of episodes

# network parameter
num_hidden 
activation 
activation_out
optimizer 

# model paramter
Q 
R 

dist_flag = 0-> train without disturbances 1-> train with disturbances 


OUTPUT:

network weights of the trained network is saved in the same folder
evolution of error is plotted


##### Main_MPC.py ##################################################################

Execution of the MPC algorithm with the trained network


PARAMETER:

network parameters have to be the same as the SI parameters

N = prediction horizon
S = samples to be evaluated

OUTPUT:

evolution of states and inputs is plotted


