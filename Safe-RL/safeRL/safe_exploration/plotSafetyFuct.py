import torch
from policies_safe import *
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sys
sys.path.append('/home/harshit/work')
import MADRaS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lin_policy = np.load("/home/harshit/work/ARS/trained_policies/Madras-explore5/bi_policy_num_plus59.npz")

lin_policy1 = lin_policy.items()[0][1]
print('------')

# lin_policy = np.load(args.expert_policy_file)
# lin_policy = lin_policy.items()[0][1]

#M = lin_policy1[0]
# mean and std of state vectors estimated online by ARS. 
mean = lin_policy1[1]
print(mean.shape)

std = lin_policy1[2]
# mean = np.asarray([0,0])
# std=np.asarray([1,1])

env = gym.make("Madras-v0")
policy_params={'type':'bilayer_safe_explore',
                'ob_filter':'MeanStdFilter',
                'ob_dim':env.observation_space.shape[0],
                'ac_dim':env.action_space.shape[0]}


policy = SafeBilayerExplorerPolicy(policy_params)


PATH = "/home/harshit/work/ARS/trained_policies/Madras-explore5/safeQ_torch59.pt"
policy.safeQ.load_state_dict(torch.load(PATH))
policy.safeQ.eval()
policy.safeQ.to(device)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


X = np.arange(-1.0,1.0,0.1)
Y = np.arange(-0.1,0.1,0.001)
X, Y = np.meshgrid(X, Y)
Z= np.zeros(X.shape[0]*X.shape[1])
ct = 0 
su= 0
obs = env.reset()
for x,y in zip(X.reshape(-1),Y.reshape(-1)):
    obs[20] = x
    #obs =np.asarray([x,y]).astype(np.float64)
    #print(obs)
    obs_n = torch.from_numpy(obs).float().to(device)
    #obs_n = torch.from_numpy((obs-mean)/std).float().to(device)
    value = policy.safeQ((obs_n)).detach().cpu().float().numpy()
    action = np.random.randn(env.action_space.shape[0]).astype(np.float64)
    #print(value*action)
    cost = np.sum(value*action)
    su+=cost
    #print(su)
    # print(np.mean(value))
    # print('---------------')
    Z[ct] = su
    ct+=1

Z = Z.reshape(X.shape[0],X.shape[1])

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-200000, 500)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig('foo.png')
#plt.show()
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)


# for i in np.arange(-1.2,0.6,0.1):
#     for j in np.arange(-0.07,0.07,0.2):
#         obs =np.asarray([i,j]).astype(np.float64)
#         obs_n = torch.from_numpy((obs-mean)/std).float().to(device)
#         np.append(value,policy.safeQ((obs_n)).detach().cpu().float().numpy())
#         ax.scatter(i, j,policy.safeQ((obs_n)).detach().cpu().float().numpy() , c='r', marker='o')

# plt.savefig('foo.png')



