'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from filter import get_filter
import torch.optim as optim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Policy(object):

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux

class MLP(nn.Module):

    def __init__(self, input, output):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input, 32)
        self.fc2 = nn.Linear(32, output)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # print(x)
        #count = x.detach().numpy()
        #count = np.where(count==0.0)
        #print("COUNT",count[0].shape)
        #print("ZERO {}".format(128-np.sum(np.nonzero(count)[0])))
        x = self.fc2(x)
        return x   


class linear(nn.Module):

    def __init__(self, input, output):
        super(linear, self).__init__()
        self.fc1 = nn.Linear(input, output)

    def forward(self, x):
        x = self.fc1(x)
        # print(x)
        #count = x.detach().numpy()
        #count = np.where(count==0.0)
        #print("COUNT",count[0].shape)
        #print("ZERO {}".format(128-np.sum(np.nonzero(count)[0])))
        return x   


class MLP_probs(nn.Module):

    def __init__(self, input, output):
        super(MLP_probs, self).__init__()
        self.fc1 = nn.Linear(input, 32)
        self.fc2 = nn.Linear(32, output)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # print(x)
        #count = x.detach().numpy()
        #count = np.where(count==0.0)
        #print("COUNT",count[0].shape)
        #print("ZERO {}".format(128-np.sum(np.nonzero(count)[0])))
        x = self.fc2(x)
        x = F.softmax(x)
        return x   




class BilayerPolicy_softmax(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params,trained_weights= None):
        Policy.__init__(self, policy_params)
        self.net = MLP_probs(self.ob_dim, self.ac_dim)
        #lin_policy = np.load('/home/harshit/work/ARS/trained_policies/Policy_Testerbi2/bi_policy_num_plus149.npz')
    
        #lin_policy = lin_policy.items()[0][1]
        #self.weights=None

        self.weights = parameters_to_vector(self.net.parameters()).detach().double().numpy()
        # if trained_weights is not None:
        #     print("hieohrfoiahfoidanfkjahdfj")
        #     vector_to_parameters(torch.tensor(trained_weights), self.net.parameters())
        #     self.weights = trained_weights

    def update_weights(self, new_weights):
        vector_to_parameters(torch.tensor(new_weights), self.net.parameters())
        return



    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        obs = torch.from_numpy(ob)
        #print(np.argmax(self.net(obs).detach().double().numpy()))
        return np.argmax(self.net(obs).detach().double().numpy())    

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        #aux = np.asarray([self.weights.detach().double().numpy(), mu, std])
        aux = np.asarray([self.weights, mu, std])

        return aux


class BilayerPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params,trained_weights= None):
        Policy.__init__(self, policy_params)
        self.net = MLP(self.ob_dim, self.ac_dim)
        #lin_policy = np.load('/home/harshit/work/ARS/trained_policies/Policy_Testerbi2/bi_policy_num_plus149.npz')
    
        #lin_policy = lin_policy.items()[0][1]
        #self.weights=None

        self.weights = parameters_to_vector(self.net.parameters()).detach().double().numpy()
        # if trained_weights is not None:
        #     print("hieohrfoiahfoidanfkjahdfj")
        #     vector_to_parameters(torch.tensor(trained_weights), self.net.parameters())
        #     self.weights = trained_weights

    def update_weights(self, new_weights):
        print("UPDATE")
        vector_to_parameters(torch.tensor(new_weights), self.net.parameters())
        return



    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        obs = torch.from_numpy(ob)
        return self.net(obs).detach().double().numpy()

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        #aux = np.asarray([self.weights.detach().double().numpy(), mu, std])
        aux = np.asarray([self.weights, mu, std])

        return aux

class SafeBilayerPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params,trained_weights= None):
        Policy.__init__(self, policy_params)
        self.net = MLP(self.ob_dim, self.ac_dim)
        self.safeQ = MLP(self.ob_dim+self.ac_dim, 1)
        #lin_policy = np.load('/home/harshit/work/ARS/trained_policies/Policy_Testerbi2/bi_policy_num_plus149.npz')
    
        #lin_policy = lin_policy.items()[0][1]
        #self.weights=None

        self.weights = parameters_to_vector(self.net.parameters()).detach().double().numpy()
        # if trained_weights is not None:
        #     print("hieohrfoiahfoidanfkjahdfj")
        #     vector_to_parameters(torch.tensor(trained_weights), self.net.parameters())
        #     self.weights = trained_weights

    def update_weights(self, new_weights):
        print("UPDATE")
        vector_to_parameters(torch.tensor(new_weights), self.net.parameters())
        return


    def getQ(self,ob,action):
        ob = self.observation_filter(ob, update=self.update_filter)
        # input_to_network = np.append(ob,action).astype(np.float64)
        ob_ = torch.from_numpy(ob).float()
        return self.safeQ(ob_).detach().double().numpy()
      
    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        obs = torch.from_numpy(ob)
        return self.net(obs).detach().double().numpy()

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        #aux = np.asarray([self.weights.detach().double().numpy(), mu, std])
        aux = np.asarray([self.weights, mu, std])

        return aux

class SafeBilayerExplorerPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params,trained_weights= None):
        Policy.__init__(self, policy_params)
        # if trained_weights is not None:
        self.net = MLP(self.ob_dim, self.ac_dim)
        self.safeQ = linear(self.ob_dim,self.ac_dim).to(device)
        self.optimizer = optim.RMSprop(self.safeQ.parameters())
        # else:
        #     self.safeQ = MLP(self.ob_dim,self.ac_dim).to(device)

        #lin_policy = np.load('/home/harshit/work/ARS/trained_policies/Policy_Testerbi2/bi_policy_num_plus149.npz')
    
        #lin_policy = lin_policy.items()[0][1]
        #self.weights=None

        self.weights = parameters_to_vector(self.net.parameters()).detach().double().numpy()
        if trained_weights is not None:
            self.safeQ.load_state_dict(torch.load(trained_weights))
            self.safeQ.to(device)
        # if trained_weights is not None:
        #     print("hieohrfoiahfoidanfkjahdfj")
        #     vector_to_parameters(torch.tensor(trained_weights), self.net.parameters())
        #     self.weights = trained_weights

    def update_weights(self, new_weights):
        print("UPDATE")
        vector_to_parameters(torch.tensor(new_weights), self.net.parameters())
        return


    def getQ(self,ob):
        # ob = self.observation_filter(ob, update=self.update_filter)
        input_to_network = ob.astype(np.float64)
        input_to_network_ = torch.from_numpy(ob).float().to(device)
        return self.safeQ(input_to_network_).cpu().detach().double().numpy()
      
    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        obs = torch.from_numpy(ob)
        #print(np.argmax(self.net(obs).detach().double().numpy()))
        return self.net(obs).detach().double().numpy()   

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        #aux = np.asarray([self.weights.detach().double().numpy(), mu, std])
        aux = np.asarray([self.weights, mu, std])

        return aux



class SafeBilayerDiscretePolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params,trained_weights= None):
        Policy.__init__(self, policy_params)
        # if trained_weights is not None:
        self.net = MLP_probs(self.ob_dim, self.ac_dim)
        self.safeQ = MLP(self.ob_dim,self.ac_dim).to(device)
        self.target_net = MLP(self.ob_dim,self.ac_dim).to(device)
        self.target_net.load_state_dict(self.safeQ.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.safeQ.parameters())
        # else:
        #     self.safeQ = MLP(self.ob_dim,self.ac_dim).to(device)

        #lin_policy = np.load('/home/harshit/work/ARS/trained_policies/Policy_Testerbi2/bi_policy_num_plus149.npz')
    
        #lin_policy = lin_policy.items()[0][1]
        #self.weights=None

        self.weights = parameters_to_vector(self.net.parameters()).detach().double().numpy()
        # if trained_weights is not None:
        #     print("hieohrfoiahfoidanfkjahdfj")
        #     vector_to_parameters(torch.tensor(trained_weights), self.net.parameters())
        #     self.weights = trained_weights

    def update_weights(self, new_weights):
        print("UPDATE")
        vector_to_parameters(torch.tensor(new_weights), self.net.parameters())
        return


    def getQ(self,ob,action):
        ob = self.observation_filter(ob, update=self.update_filter)
        #input_to_network = np.append(ob,action).astype(np.float64)
        input_to_network_ = torch.from_numpy(ob).float()
        return self.safeQ(input_to_network_).detach().double().numpy()[action]
      
    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        obs = torch.from_numpy(ob)
        #print(np.argmax(self.net(obs).detach().double().numpy()))
        return self.net(obs).detach().double().numpy()

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        #aux = np.asarray([self.weights.detach().double().numpy(), mu, std])
        aux = np.asarray([self.weights, mu, std])

        return aux
def check_implementation():
    policy_params={'type':'linear',
                   'ob_filter':'MeanStdFilter',
                   'ob_dim': 24,
                   'ac_dim': 4}
    policy = BilayerPolicy(policy_params)
    print(policy.net)

#check_implementation()
