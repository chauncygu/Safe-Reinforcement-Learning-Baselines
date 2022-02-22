'''
Importance sampling estimators

Harshit Sikchi
'''
import numpy as np
import math

'''
Simple importance sampling

'''


def simple_is(pi_b,pi_e,reward):
    estimated_reward = reward
    for i,action_hist_prob in enumerate(pi_b):
        estimated_reward*= pi_e[i]/pi_b[i]
    return estimated_reward



'''
Per Decision Importance Sampling
reward: list of reward obtained per time step

gamma: discount factor
trajectory_reward_high: Maximum value of sum of discounted rewards in a trajectory 
trajectory_reward_low: Minimum value of sum of discounted rewards in a trajectory 

returns normalized estimate of reward under evaluation policy
'''

def per_decision_is(pi_b,pi_e,gamma,reward,trajectory_reward_high,trajectory_reward_low):
    horizon = len(reward)
    expected_reward = 0
    gamma_t = 1
    importance_weight = 1
    for t in range(1,horizon+1):
        importance_weight *= pi_e[t-1]/pi_b[t-1] 
        expected_reward+= gamma_t * reward[t-1] *importance_weight  
        gamma_t *= gamma

    return (expected_reward - trajectory_reward_low)/(trajectory_reward_high-trajectory_reward_low)


'''
Normalized Per Decision Importance Sampling
reward: list of reward obtained per time step
trajectory_reward_high: Maximum value of sum of discounted rewards in a trajectory 
trajectory_reward_low: Minimum value of sum of discounted rewards in a trajectory 
reward_high: Maximum reward obtained per time step
reward_low: Minimum reward obtained per time step
gamma: discount factor
returns normalized estimate of reward under evaluation policy
'''

def normalized_per_decision_is(pi_b,pi_e,gamma,reward,trajectory_reward_high,trajectory_reward_low,reward_low):
    horizon = len(reward)
    expected_reward = 0
    gamma_t = 1
    importance_weight = 1
    for t in range(1,horizon+1):
        importance_weight *= pi_e[t-1]/pi_b[t-1] 
        expected_reward+= gamma_t * (reward[t-1]-reward_low) *importance_weight  
        gamma_t *= gamma

    return (expected_reward +(reward_low*((gamma_t-1)/(gamma-1)))- trajectory_reward_low)/(trajectory_reward_high-trajectory_reward_low)



'''
Weighted Importance Sampling
* Works in a batch setting

pi_b : batch containing histories sampled from behavorial policy 
pi_e : batch containing histories sampled from evaluation policy 

reward: batch of list of reward obtained per time step


returns normalized estimate of performance under evaluation policy
'''

def weighted_is(pi_b,pi_e,reward):

    estimated_reward = 0
    estimated_weight = 0
    for history_b,history_e,history_reward in zip(pi_b,pi_e,reward):
        estimated_history_reward = history_reward
        estimated_history_weight = 1
        for i,action_hist_prob in enumerate(history_b):
            estimated_history_reward*= history_e[i]/history_b[i]
            estimated_history_weight*= history_e[i]/history_b[i]
        estimated_reward+= estimated_history_reward
        estimated_weight+= estimated_history_weight
    return estimated_reward/estimated_weight





'''
Weighted Per Decision Importance Sampling
* Works in a batch setting

pi_b : batch containing histories sampled from behavorial policy 
pi_e : batch containing histories sampled from evaluation policy 

reward: batch of list of reward obtained per time step

gamma: discount factor
trajectory_reward_high: Maximum value of sum of discounted rewards in a trajectory 
trajectory_reward_low: Minimum value of sum of discounted rewards in a trajectory 

returns normalized estimate of reward under evaluation policy
'''

def weighted_per_decision_is(pi_b,pi_e,gamma,reward,trajectory_reward_high,trajectory_reward_low):
    
    estimated_reward = 0
    estimated_weight = 0
    for history_b,history_e,history_reward in zip(pi_b,pi_e,reward):    
        horizon = len(history_reward)
        estimated_history_reward = 0
        estimated_history_weight = 0 
        gamma_t = 1
        importance_weight = 1
        for t in range(1,horizon+1):
            importance_weight *= history_e[t-1]/history_b[t-1] 
            estimated_history_reward+= gamma_t * history_reward[t-1] *importance_weight  
            estimated_history_weight+=gamma_t*importance_weight
            gamma_t *= gamma
        estimated_weight+= estimated_history_weight
        estimated_reward+= estimated_history_reward

    return ((estimated_reward/estimated_weight) - trajectory_reward_low)/(trajectory_reward_high-trajectory_reward_low)


'''
Consistent Weighted Per Decision Importance Sampling
* Works in a batch setting

pi_b : batch containing histories sampled from behavorial policy 
pi_e : batch containing histories sampled from evaluation policy 

reward: batch of list of reward obtained per time step

gamma: discount factor
trajectory_reward_high: Maximum value of sum of discounted rewards in a trajectory 
trajectory_reward_low: Minimum value of sum of discounted rewards in a trajectory 

returns normalized estimate of reward under evaluation policy
'''

# TODO
# Conceptualize and formulate

def consistent_weighted_per_decision_is(pi_b,pi_e,gamma,reward,trajectory_reward_high,trajectory_reward_low):
    
    estimated_reward = 0
    estimated_weight = 0
    for history_b,history_e,history_reward in zip(pi_b,pi_e,reward):    
        horizon = len(history_reward)
        estimated_history_reward = 0
        estimated_history_weight = 0 
        gamma_t = 1
        importance_weight = 1
        for t in range(1,horizon+1):
            importance_weight *= history_e[t-1]/history_b[t-1] 
            estimated_history_reward+= gamma_t * history_reward[t-1] *importance_weight  
            estimated_history_weight+=gamma_t*importance_weight
            gamma_t *= gamma
        estimated_weight+= estimated_history_weight
        estimated_reward+= estimated_history_reward

    return ((estimated_reward/estimated_weight) - trajectory_reward_low)/(trajectory_reward_high-trajectory_reward_low)