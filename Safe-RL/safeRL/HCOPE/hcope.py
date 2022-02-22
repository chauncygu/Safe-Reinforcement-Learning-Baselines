import numpy as np
import gym
from policies import *
import math
from scipy.optimize import minimize
from scipy.special import j1
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import matplotlib.pyplot as plt

class HCOPE(object):

    def __init__(self,env,policy,eval_policy,rollout_length,delta=0.1):
        self.env = env
        self.policy= policy
        self.eval_policy=eval_policy
        self.rollout_length = rollout_length
        self.w_policy = self.policy.get_weights()
        # Set up maximum and minimum reward in a trajectory
        self.R_max = 200
        self.R_min = 1
        self.delta=delta
        if eval_policy is None:
            self.e_policy = None
        else:
            self.e_policy=self.eval_policy.get_weights()


    # Method to generate evaluation policy with gaussian noise added to our behaviour policy
    def setup_e_policy(self):
        noise =  np.random.normal(0,0.01,self.w_policy.shape)
        self.e_policy = self.w_policy - noise
        self.eval_policy.update_weights(self.e_policy)

    def rollout(self,shift = 0.,policy = None, rollout_length = None,render = False):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        total_reward = 0.
        steps = 0

        if(rollout_length==None):
            rollout_length=self.rollout_length

        ob = self.env.reset()
        for i in range(rollout_length):
            action,prob = policy.act(ob)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if(render):
                env.render()
            if done:
                break
            
        return total_reward, steps

    # Modified rollout method for HCOPE evaluation. Returns probs of each action that were taken in behavorial as well as evaluation policy
    def mod_rollout(self,shift = 0., rollout_length = None,render = False,random =False,greedy=True):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        

        total_reward = 0.
        steps = 0
        rewards = []
        probs = []
        eval_probs =[]
        if(rollout_length==None):
            rollout_length=self.rollout_length

        ob = self.env.reset()
        for i in range(rollout_length):
            if random== True:
                action = np.random.randint(0,env.action_space.n)
                action,prob = self.policy.act_action(ob,action)
                eval_action,eval_prob = self.eval_policy.act_action(ob,action)
            elif greedy==False:
                action,prob = self.policy.act(ob,greedy=greedy)               
                eval_action,eval_prob = self.eval_policy.act_action(ob,action)
                 
            else:
                action,prob = self.policy.act(ob)
                
                eval_action,eval_prob = self.eval_policy.act_action(ob,action)
            
            ob, reward, done, _ = self.env.step(action)
            rewards.append(reward- shift)
            probs.append(prob)
            eval_probs.append(eval_prob)
            steps += 1
            total_reward += (reward - shift)
            if(render):
                env.render()
            if done:
                break
            
        return total_reward, steps,rewards,probs,eval_probs


    # Evaluate any policy
    def evaluate(self,policy=None,shift=0.,n_rollouts=100,render = False):
        self.policy.update_weights(self.w_policy)
        self.policy.update_filter = False
        rewards = []
        for i in  range(n_rollouts):
            total_reward,steps = self.rollout(render=render,shift =shift    ,policy = policy)
            rewards.append(total_reward)            

        rewards = np.asarray(rewards)
        rewards = self.normalize_reward(rewards,self.R_min,self.R_max)

        return(np.mean(rewards))


    # Method to normalize trajectory rewards
    def normalize_reward(self, rewards,R_minus,R_plus):
        return (rewards-R_minus)/(R_plus-R_minus)


    # Method to generate dataset if it is not provided
    def generate_dataset(self,dataset_size = 100,shift = 0.,render=False):
        # Stop updating filter 
        self.policy.update_weights(self.w_policy)
        self.policy.update_filter = False
        self.eval_policy.update_weights(self.e_policy)
        self.eval_policy.update_filter = False
        rewards = []
        probs = []
        eval_probs = []


        for i in  range(dataset_size):
            total_reward,steps,rewards_list,probs_list,eval_probs_list = self.mod_rollout(render=render,shift = shift,greedy=False)
            rewards.append(rewards_list)
            probs.append(probs_list)
            eval_probs.append(eval_probs_list)            

        rewards = np.asarray(rewards)
        probs = np.asarray(probs)
        eval_probs = np.asarray(eval_probs)

        # Shuffle our dataset
        permutation = np.random.permutation(probs.shape[0])
        
        rewards = rewards[permutation,:]
        #rewards=self.normalize_reward(rewards,self.R_min,self.R_max)

        probs = probs[permutation,:]
        eval_probs =eval_probs[permutation,:]

        # Break the dataset into two parts for estimating c* 
        d_pre = rewards[:int(0.05*dataset_size),:]
        d_post = rewards[int(0.05*dataset_size):,:]
        
        pi_b_pre = probs[:int(0.05*dataset_size),:]
        pi_b_post = probs[int(0.05*dataset_size):,:]

        pi_e_pre = eval_probs[:int(0.05*dataset_size),:]
        pi_e_post = eval_probs[int(0.05*dataset_size):,:]

        return [d_pre,d_post,pi_b_pre,pi_b_post,pi_e_pre,pi_e_post]


        
    def visualize_IS_distribution(self):
        episodes = 1000
        probs=[]
        self.policy.update_weights(self.w_policy)
        self.policy.update_filter = False
        self.eval_policy.update_weights(self.e_policy)
        self.eval_policy.update_filter = False

        eval_probs=[]
        for i in  range(episodes):
            total_reward,steps,rewards_list,probs_list,eval_probs_list = self.mod_rollout(greedy=False)
            probs.append(probs_list)
            eval_probs.append(eval_probs_list)            

        
        probs = np.asarray(probs)
        eval_probs = np.asarray(eval_probs)

        importance_weight = np.log(np.asarray([ np.prod(np.asarray(eval_probs[i])/np.asarray(probs[i])) for i in range(episodes)], dtype=float))
        plt.hist(importance_weight, color = 'blue', edgecolor = 'black',bins = int(100))

        plt.savefig("IS_dist.png")
        


    def estimate_behavior_policy(self,dataset):
        d_pre,d_post,pi_b_pre,pi_b_post,pi_e_pre,pi_e_post = dataset
        eval_estimate = self.hcope_estimator(d_pre, d_post, pi_b_pre,pi_b_post,pi_e_pre,pi_e_post,self.delta)
        print("Estimate of evaluation policy: {}".format(eval_estimate))

    
    def hcope_estimator(self,d_pre, d_post, pi_b_pre,pi_b_post,pi_e_pre,pi_e_post,delta):
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
        
        print("Running HCOPE estimator on the evaluation policy..........")

        d_pre = np.asarray(d_pre)
        d_post = np.asarray(d_post)
        n_post = len(d_post)
        n_pre = len(d_pre)

        # Estimate c which maximizes the lower bound using estimates from d_pre

        c_estimate = 4.0
        print("Intial estimate of c {}.".format(c_estimate))

        def f(x):
            n_pre = len(d_pre)
            Y = np.asarray([min(self.normalize_reward(np.sum(d_pre[i]),self.R_min,self.R_max) * np.prod(pi_e_pre[i]/pi_b_pre[i].astype(np.float64)), x) for i in range(n_pre)], dtype=float)
            importance_weights = np.asarray([ np.prod(pi_e_pre[i]/pi_b_pre[i].astype(np.float64)) for i in range(n_pre)], dtype=float)
            # Empirical mean
            EM = np.sum(Y)/n_pre
            #print(EM)
            # Second term
            term2 = (7.*x*np.log(2./delta)) / (3*(n_post-1))
            # print(term2)
            square_term = ((n_pre*np.sum(np.square(Y))) - np.square(np.sum(Y)))
            if square_term<0:
                square_term=0
            # Third term
            term3 = np.sqrt( (((2.*np.log(2./delta))/(n_post*n_pre*(n_pre-1))) * square_term ))
            # print(term3)
            return (-EM+term2+term3) 

        c_estimate = minimize(f,np.array([c_estimate]),method='BFGS').x

        print("The estimate for c* was found to be {}.".format(c_estimate))

        # Use the estimated c for computing the maximum lower bound
        c = c_estimate

        if ~isinstance(c, list):
            c = np.full((n_post,), c, dtype=float)

        
        
        if n_post<=1:
            raise(ValueError("The value of 'n' must be greater than 1"))


        Y = np.asarray([min(self.normalize_reward(np.sum(d_post[i]),self.R_min,self.R_max) * np.prod(pi_e_post[i]/pi_b_post[i].astype(np.float64)), c[i]) for i in range(len(d_post))], dtype=float)
        importance_weights = np.asarray([ np.prod(pi_e_post[i]/pi_b_post[i].astype(np.float64)) for i in range(n_post)], dtype=float)
    
        # Empirical mean
        c = np.asarray([max(1,i) for i in c])

        EM = np.sum(Y/c[0])/(np.sum(1/c))

        # Second term
        term2 = (7.*n_post*np.log(2./delta)) / (3*(n_post-1)*np.sum(1/c))

        # Third term
        square_term = (n_post*np.sum(np.square(Y/c)) - np.square(np.sum(Y/c)))
        if square_term<0:
            square_term = 0
        term3 = np.sqrt( ((2*np.log(2./delta))/(n_post-1)) * square_term) / np.sum(1/c)


        # Sanity check on determinant

        k1 = (7.*n_post)/(3*(n_post-1)) 
        k2 = (n_post*np.sum(np.square(Y/c)) - np.square(np.sum(Y/c)))*(2./(n_post-1))
        k3 = (EM - term2 - term3)*np.sum(1/c) - (np.sum(Y/c))

        if(k2-4*k1*k3<0):
            print("The estimate of u_ is of zero confidence")
        else:
            if(-np.sqrt(k2)+np.sqrt(k2-4*k1*k3))<0:
                print("The estimate of u_ is of zero confidence")

        # Final estimate
        return EM - term2 - term3











if __name__=="__main__":
    # Create a gym environment
    env_name = "MountainCar-v0"
    env = gym.make(env_name)

    # Assuming discrete action space
    action_size = env.action_space.n
    ob_size = env.observation_space.shape[0]

    # Create a bilayer mlp with softmax
    policy_params={'type':'bilayer',
                   'ob_filter':'MeanStdFilter',
                   'ob_dim':ob_size,
                   'ac_dim':action_size}
    policy = BilayerPolicy_softmax(policy_params)
    eval_policy = BilayerPolicy_softmax(policy_params)

    my_hcope = HCOPE(env,policy,eval_policy,rollout_length = 1000,delta =0.1)
    my_hcope.setup_e_policy()

    dataset = my_hcope.generate_dataset(dataset_size=100,shift=-2)
    print("Estimate of behavorial policy: {}".format(my_hcope.evaluate(policy=my_hcope.policy,shift = -2,n_rollouts=100,render =False)))

    my_hcope.estimate_behavior_policy(dataset)
    print("True estimate of evaluation policy: {}".format(my_hcope.evaluate(policy=my_hcope.eval_policy,shift = -2,n_rollouts=100,render =False)))

    #my_hcope.visualize_IS_distribution()