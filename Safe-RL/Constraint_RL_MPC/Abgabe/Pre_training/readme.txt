
##### Immediate_constraint_functions.py ##################################################################

Pre-training phase to learn immediate constraint functions
has to be evaluated for every constraint


PARAMETER:

num_samples =  number of samples per episode
num_episodes =  number of episodes

state_flag = 0-> Temperature low, 1-> Energy low,  2-> Temperature up, 3-> Energy up,  define safety signal

# define network parameters
num_in 
num_out 
num_hidden
activation 
activation_out
optimizer 

OUTPUT:

network weights of the trained network is saved in the same folder


##### Test_Immediate_constraint_functions.py ##############################################################

evaluation of safety layer, to make sure that the constraints are working
loads neural network weights, so they have to be trained before
