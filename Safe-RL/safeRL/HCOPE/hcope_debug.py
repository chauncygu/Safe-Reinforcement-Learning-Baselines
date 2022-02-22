import numpy as np 
import random
import math
from scipy.optimize import minimize
from scipy.special import j1
from scipy.optimize import minimize_scalar


# TODO
# Add Sanity Check

def hcope_estimator(d_pre, d_post, pi_b,pi_e,delta):
	"""
	d_pre : float, size = (dataset_split,)
		Trajectory rewards from the behavior policy 

	d_post : float, size = (dataset_size - dataset_split, )
		Trajectory rewards from the behavior policy 

	delta : float, size = scalar
		1-delta is the confidence of the estimator
	
	pi_b : Probabilities for respective trajectories in behaviour policy

	pi_e : Probabilities for respective trajectories in evaluation policy

	RETURNS: lower bound for the mean, mu as per Theorem 1 of Thomas et al. High Confidence Off-Policy Evaluation
	"""
	d_pre = np.asarray(d_pre)
	d_post = np.asarray(d_post)
	n_post = len(d_post)
	n_pre = len(d_pre)

    # Estimate c which maximizes the lower bound using estimates from d_pre

	c_estimate = 4.0

	def f(x):
		n_pre = len(d_pre)
		Y = np.asarray([min((d_pre[i] * pi_e[i])/pi_b[i], x) for i in range(n_pre)], dtype=float)

		# Empirical mean
		EM = np.sum(Y)/n_pre

		# Second term
		term2 = (7.*x*np.log(2/delta)) / (3*(len(d_post)-1))

		# Third term
		term3 = np.sqrt( ((2*np.log(2/delta))/(n_post*n_pre*(n_pre-1)) * (n_pre*np.sum(np.square(Y)) - np.square(np.sum(Y))) ))
		
		return (-EM+term2+term3) 

	c_estimate = minimize(f,np.array([c_estimate]),method='BFGS').x

	# Use the estimated c for computing the maximum lower bound
	c = c_estimate

	if ~isinstance(c, list):
		c = np.full((n_post,), c, dtype=float)

	
	if n_post<=1:
		raise(ValueError("The value of 'n' must be greater than 1"))


	Y = np.asarray([min((d_post[i] * pi_e[i])/pi_b[i], c[i]) for i in range(len(d_post))], dtype=float)

	# Empirical mean
	EM = np.sum(Y/c)/np.sum(1/c)

	# Second term
	term2 = (7.*n_post*np.log(2/delta)) / (3*(n_post-1)*np.sum(1/c))

	# Third term
	term3 = np.sqrt( ((2*np.log(2/delta))/(n_post-1)) * (n_post*np.sum(np.square(Y/c)) - np.square(np.sum(Y/c))) ) / np.sum(1/c)


	# Final estimate
	return EM - term2 - term3



'''
Sample test for checking performance bounds by HCOPE -

Thomas, Philip S., Georgios Theocharous, and Mohammad Ghavamzadeh. "High-Confidence Off-Policy Evaluation." AAAI. 2015.

'''

if __name__=="__main__":

    dataset_size = 100
    # Rewards obtained per trajectories
    dataset = random.sample(xrange(100),dataset_size)
    d_pre = dataset[:int(0.05*dataset_size)]
    d_post = dataset[int(0.05*dataset_size):]
    # Ensure small divergence between both policy
    pi_b = np.random.uniform(low=0.02, high=0.1, size=(dataset_size,))
    pi_e = np.random.uniform(low=0.001, high=0.006, size=(dataset_size,))+ pi_b

    delta_confidence = 0.1
    print('Performance of behaviour policy: ' ,np.sum(dataset)/len(dataset))
    print('Estimated lower bound performance: ',hcope_estimator(d_pre,d_post,pi_b,pi_e,delta_confidence))
