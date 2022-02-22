from safety_multiagent_mujoco.mujoco_multi import MujocoMulti
import numpy as np
import time


def main():

    # Swimmer
    # env_args = {"scenario": "manyagent_swimmer",
    #             "agent_conf": "10x2",
    #             "agent_obsk": 1,
    #             "episode_limit": 1000}

    # coupled_half_cheetah
    # env_args = {"scenario": "coupled_half_cheetah",
    #             "agent_conf": "1p1",
    #             "agent_obsk": 1,
    #             "episode_limit": 1000}

    # ANT 4
    # env_args = {"scenario": "manyagent_ant",
    #               "agent_conf": "3x2",
    #               "agent_obsk": 1,
    #               "episode_limit": 1000}

    # env_args = {"scenario": "manyagent_swimmer",
    #             "agent_conf": "10x2",
    #             "agent_obsk": 1,
    #             "episode_limit": 1000}

    env_args = {"scenario": "HalfCheetah-v2",
                "agent_conf": "2x3",
                "agent_obsk": 1,
                "episode_limit": 1000}

    # env_args = {"scenario": "Hopper-v2",
    #             "agent_conf": "3x1",
    #             "agent_obsk": 1,
    #             "episode_limit": 1000}

    # env_args = {"scenario": "Humanoid-v2",
    #             "agent_conf": "9|8",
    #             "agent_obsk": 1,
    #             "episode_limit": 1000}

    # env_args = {"scenario": "Humanoid-v2",
    #             "agent_conf": "17x1",
    #             "agent_obsk": 1,
    #             "episode_limit": 1000}

    # env_args = {"scenario": "Ant-v2",
    #             "agent_conf": "2x4",
    #             "agent_obsk": 1,
    #             "episode_limit": 1000}

    # env_args = {"scenario": "Ant-v2",
    #             "agent_conf": "2x4d",
    #             "agent_obsk": 1,
    #             "episode_limit": 1000}

    # env_args = {"scenario": "Ant-v2",
    #             "agent_conf": "4x2",
    #             "agent_obsk": 1,
    #             "episode_limit": 1000}

    env = MujocoMulti(env_args=env_args)
    env_info = env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    n_episodes = 10

    for e in range(n_episodes):
        ob=env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.uniform(-10, 0.0, n_actions)
                actions.append(action)

            # reward, terminated, _ = env.step(actions)
            # print("env.step(actions): ", env.step(actions))
            get_obs, get_state, reward, dones, infos, get_avail_actions= env.step(actions)
            # episode_reward += reward
            # print("reward: ", reward)
            cost_x= [[item['cost']] for item in infos]
            print("cost_x:", cost_x)
            print("reward:", reward)

            # time.sleep(0.1)
            env.render()


        # print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()

if __name__ == "__main__":
    main()
    """
    infos[cost]: [{'cost': 0.0, 'reward_forward': -0.6434413402233052, 'reward_ctrl': -4.010836585120964,
                   'reward_contact': -1.2071856383999997e-13, 'reward_survive': 1.0, 'cost_obj': 0.0, 'cost_done': 0.0},
                  {'cost': 0.0, 'reward_forward': -0.6434413402233052, 'reward_ctrl': -4.010836585120964,
                   'reward_contact': -1.2071856383999997e-13, 'reward_survive': 1.0, 'cost_obj': 0.0, 'cost_done': 0.0},
                  {'cost': 0.0, 'reward_forward': -0.6434413402233052, 'reward_ctrl': -4.010836585120964,
                   'reward_contact': -1.2071856383999997e-13, 'reward_survive': 1.0, 'cost_obj': 0.0, 'cost_done': 0.0}]
    """