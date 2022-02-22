# Constraint_RL_MPC
Safe control of unknown dynamic systems with reinforcement learning and model predictive control

STRUCTURE:

---Pre_training---
first main file for the pre-training phase where the safety layer's immediate constraint functions are trained

---Training_MPC---
main file for the system identification
main file for the MPC training and test
containing the class MPC, where control strategy is implemeted
system identification has to be evaluated first

---Training_RL---
main file for the the RL training and test
containing the class DDPG, where the actor-critic method is implemented
when using the safety layer, the pre-training phase has to be evaluated first

---Buffer---
contains the class ReplayBuffer.py, where the training data is stored during the training of the RL algorithm

---Disturbances---
contains different sources of disturbances
- external_disturbnaces_old.m -> original disturbance, used for test
- external_disturbances_normal.m -> normally distributed disturbance in the same range as the original values
- external_disturbances_randn.m -> uniform distributed disturbance in the same range as the original values

---Model---
contains the model of the system

---Neural_Network---
contains a class for the actor, a class for the critic network and a class for a general network

---Normalize---
realizes functions for the min-max normailzation and renormalization for the data used for neural networks


HOW TO:

---Integrate a new model---
To integrate a new model in the structure, the Linear_Env.py file has to be changed.
__init__: Here, the matrices and dimensions are defined.
step: Here, the system equations are described and executed.
All other functions in the Linear_Env.py file have to be adapted to possible changes of the new model parameters.

In the Pre_trainig, the constraints of the states have to be adapted to the new model.

In the Training_RL phase, again the constraints have to be adapted in the Main file and in the DDPG file.



