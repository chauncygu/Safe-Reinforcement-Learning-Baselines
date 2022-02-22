"""

Code to load a policy and generate rollout data. Adapted from https://github.com/berkeleydeeprlcourse. 
Example usage:
    python run_policy.py ../trained_policies/Humanoid-v1/policy_reward_11600/lin_policy_plus.npz Humanoid-v1 --render \
            --num_rollouts 20
"""
import numpy as np
import gym
import MADRaS
import matplotlib.pyplot as plt
from policies_safe import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.set_printoptions(threshold=np.nan)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert rollouts')
    args = parser.parse_args()

    print('loading and building expert policy from {}'.format(args.expert_policy_file))
    lin_policy = np.load(args.expert_policy_file)
    
    lin_policy1 = lin_policy.items()[0][1]
    print('------')

    # lin_policy = np.load(args.expert_policy_file)
    # lin_policy = lin_policy.items()[0][1]
    
    #M = lin_policy1[0]
    # mean and std of state vectors estimated online by ARS. 
    mean = lin_policy1[1]
    print(mean.shape)

    std = lin_policy1[2]
        
    env = gym.make(args.envname)
    policy_params={'type':'bilayer_safe_explorer',
                   'ob_filter':'MeanStdFilter',
                   'ob_dim':env.observation_space.shape[0],
                   'ac_dim':env.action_space.shape[0]}
    policy = SafeBilayerExplorerPolicy(policy_params)
    PATH = "/home/harshit/work/ARS/trained_policies/Madras-explore8/bi_policy_num_plus_torch459.pt"
    policy.net.load_state_dict(torch.load(PATH))
    policy.net.eval()
    PATH2 = "/home/harshit/work/ARS/trained_policies/Madras-explore7/safeQ_torch119.pt"
    policy.safeQ.load_state_dict(torch.load(PATH2))

    #print(policy.net.parameters())
    returns = []
    observations = []
    actions = []
    obs_hist=open('obs_hist.txt','w')
    violate=open('debug_violations.txt','w')

    step_list=[]
    for i in range(args.num_rollouts):
        print('iter', i)
        
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            #print(obs[20])
            obs_n = torch.from_numpy((obs-mean)/std).float()
            #obs_n = torch.from_numpy(obs).float()
            weights = policy.getQ(obs)
            action = policy.net((obs_n)).detach().float().numpy()
            C = 0.1
            # Solve the lagrangian
            lagrangian = max(float(np.sum(weights*action) + obs[20] -C)/(np.sum(weights**2)),0)
            #my_f.write("lagrangian: {} \n".format(lagrangian))
            a_star = action - lagrangian*weights
            violate.write("Debug1: {} \n".format(float(np.sum(weights*action) + obs[20] -C)))
            violate.write("Debug2: {} \n".format(float(np.sum(weights*action) + obs[20] -C)/(np.sum(weights**2))))
            violate.write("Lagrangian: {} \n".format(lagrangian))
            violate.write("Cost: {} \n".format(obs[20]+float(np.sum(weights*action))))

            violate.write("------------------------------------ \n")

            print("Weights: {}".format(weights))
            print("Debug1: {}".format(float(np.sum(weights*action) + obs[20] -C)))
            print("Debug2: {}".format(float(np.sum(weights*action) + obs[20] -C)/(np.sum(weights**2))))
            print("Cost:{}".format(obs[20]+float(np.sum(weights*action))))
            print("Action output: {}".format(action))
            print("Action taken: {}".format(a_star))
            print("Lagrangian: {}".format(lagrangian))
            print("------------------------------------")


            #print(action)
            # action = np.dot(M, (obs - mean)/std)
            observations.append(obs)
            actions.append(action)
            a_star[1] = 0.0
            a_star[0] = 0.0
            print("Action taken Jugad: {}".format(a_star))
            obs, r, done, _ = env.step(a_star)
            if(i==0):
                obs_hist.write("Observation: {}\n".format(str(obs)))
            
            # print("Observation: {} \n".format(str(obs)))
            # print('----------------------------------------------\n')

            totalr += r
            steps += 1
            step_list.append(steps)
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, env.spec.timestep_limit))
            if steps >= env.spec.timestep_limit:
                break
        
        # obs_hist.close()
        print(np.asarray(observations).shape)
        # for i in range(len(observations[0])):
        #     plt.plot(step_list,np.asarray(observations)[:,i])
        #     plt.xlabel('Steps')
        #     plt.ylabel('observations')
        # plt.show()
        print('Reward in this episode: {0}'.format(totalr))
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    
if __name__ == '__main__':
    main()
