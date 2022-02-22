import argparse
import gym
import numpy as np
import safety_gym
import torch
import time
from tqdm import tqdm

from mbrl import SafeMPC, RegressionModelEnsemble, CostModel
from utils.logx import EpochLogger
from utils.run_utils import setup_logger_kwargs, combined_shape, load_config, seed_torch
from utils.env_utils import SafetyGymEnv


DEFAULT_ENV_CONFIG_POINT = dict(
    action_repeat=3,
    max_episode_length=300,
    use_dist_reward=False,
    stack_obs=False,
)
DEFAULT_ENV_CONFIG_CAR = dict(
    action_repeat=2,
    max_episode_length=150,
    use_dist_reward=False,
    stack_obs=False,
)

def run(logger, config, args):
    if args.robot.lower() == "point":
        env_config = DEFAULT_ENV_CONFIG_POINT
    elif args.robot.lower() == "car":
        env_config = DEFAULT_ENV_CONFIG_CAR
    env = SafetyGymEnv(robot=args.robot, task="goal", level=args.level, seed=args.seed, config=env_config)
    # MPC and dynamic model config
    mpc_config = config['mpc_config']
    mpc_config["optimizer"] = args.optimizer.upper()
    cost_config = config['cost_config']
    dynamic_config = config['dynamic_config']
    if args.load is not None:
        dynamic_config["load"] = True
        dynamic_config["load_folder"] = args.load
        cost_config["load"] = True
        cost_config["load_folder"] = args.load
    if args.save:
        dynamic_config["save"] = True
        dynamic_config["save_folder"] = logger.output_dir
        cost_config["save"] = True
        cost_config["save_folder"] = logger.output_dir

    config["arguments"] = vars(args)
    logger.save_config(config)

    state_dim, action_dim = env.observation_size, env.action_size
    if args.ensemble>0:
        dynamic_config["n_ensembles"] = args.ensemble
    dynamic_model = RegressionModelEnsemble(state_dim+action_dim, state_dim, config=dynamic_config)
    cost_model = CostModel(env, cost_config)
    mpc_controller = SafeMPC(env, mpc_config, cost_model=cost_model, n_ensembles=dynamic_config["n_ensembles"])

    # Prepare random collected dataset
    start_time = time.time()
    pretrain_episodes = 1000 if args.load is None else 10
    pretrain_max_step = 50
    print("collecting random episodes...")
    data_num = 0
    for epi in tqdm(range(pretrain_episodes)):
        obs = env.reset()
        done = False
        i = 0
        while not done and i<pretrain_max_step:
            action = env.action_space.sample()
            obs_next, reward, done, info = env.step(action)
            if not info["goal_met"] and not done:  # otherwise the goal position will change
                x, y = np.concatenate((obs, action)), obs_next
                dynamic_model.add_data_point(x, y)
                cost = 1 if info["cost"]>0 else 0
                cost_model.add_data_point(obs_next, cost)
                data_num += 1
                i += 1
            obs = obs_next
    print("Finish to collect %i data "%data_num)

    # training the model
    if args.load is None:
        dynamic_model.reset_model()
        print("resetting model")
        dynamic_model.fit(use_data_buf=True, normalize=True)
    cost_model.fit()

    # Main loop: collect experience in env and update/log each epoch
    total_len = 0 # total interactions
    total_epi = 0
    for epoch in tqdm(range(args.epoch)): # update models per epoch
        for test_episode in range(args.episode): # collect data for episodes length
            obs, ep_ret, ep_cost, done = env.reset(), 0, 0, False
            mpc_controller.reset()
            if args.render:
                    env.render()
            while not done:    
                action = np.squeeze(np.array([mpc_controller.act(model=dynamic_model, state=obs)]))
                obs_next, reward, done, info = env.step(action)
                if args.render:
                    env.render()
                ep_ret += reward
                total_len += 1
                ep_cost += info["cost"]
                if not info["goal_met"] and not done:
                    x = np.concatenate((obs, action))
                    y = obs_next #- obs
                    dynamic_model.add_data_point(x, y)
                    cost = 1 if info["cost"]>0 else 0
                    cost_model.add_data_point(obs_next, cost)
                obs = obs_next     
            logger.store(EpRet=ep_ret, EpCost=ep_cost)
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('Episode', total_epi)
            logger.log_tabular('EpRet', average_only=True) #with_min_and_max = False
            logger.log_tabular('EpCost', average_only=True)
            logger.log_tabular('TotalEnvInteracts', total_len)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
            total_epi += 1
        # training the model
        if not args.test:
            dynamic_model.fit(use_data_buf=True, normalize=True)
            cost_model.fit()
    env.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='point', help="robot model, selected from `point` or `car` ")
    parser.add_argument('--level', type=int, default=1, help="environment difficulty, selected from `1` or `2`, where `2` would be more difficult than `1`")
    parser.add_argument('--epoch', type=int, default=60, help="maximum epochs to train")
    parser.add_argument('--episode', type=int, default=10, help="determines how many episodes data to collect for each epoch")
    parser.add_argument('--render','-r', action='store_true', help="render the environment")
    parser.add_argument('--test', '-t', action='store_true', help="test the performance of pretrained models without training")

    parser.add_argument('--seed', '-s', type=int, default=1, help="seed for Gym, PyTorch and Numpy")
    parser.add_argument('--dir', '-d',type=str, default='./data/', help="directory to save the logging information")
    parser.add_argument('--name','-n', type=str, default='test', help="name of the experiment, used to save data in a folder named by this parameter")
    parser.add_argument('--save', action='store_true', help="save the trained dynamic model, data buffer, and cost model")
    parser.add_argument('--load',type=str, default=None, help="load the trained dynamic model, data buffer, and cost model from a specified directory")
    parser.add_argument('--ensemble',type=int, default=0, help="number of model ensembles, if this argument is greater than 0, then it will replace the default ensembles number in config.yml") # number of ensembles
    parser.add_argument('--optimizer','-o',type=str, default="rce", help=" determine the optimizer, selected from `rce`, `cem`, or `random` ") # random, cem or CCE
    parser.add_argument('--config', '-c', type=str, default='./config.yml', help="specify the path to the configuation file of the models")

    args = parser.parse_args()
    logger_kwargs = setup_logger_kwargs(args.name, args.seed, args.dir)
    logger = EpochLogger(**logger_kwargs)
    config = load_config(args.config)

    run(logger, config, args)
    
