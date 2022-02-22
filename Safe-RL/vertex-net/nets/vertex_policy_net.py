import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class VertexPolicyNetwork(nn.Module):
    def __init__(self, env, obs_dim, num_vertex, hidden_dim, init_w=3e-3):
        super(VertexPolicyNetwork, self).__init__()

        self.env = env

        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_vertex)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=1)
        action_vertex = self.env.get_action_vertex(state.numpy())
        action_vertex = torch.FloatTensor(action_vertex).to(device)
        x = torch.bmm(x.unsqueeze(1), action_vertex).squeeze(1)
        # x = torch.sum(x * action_vertex, dim=1).unsqueeze(1)
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]