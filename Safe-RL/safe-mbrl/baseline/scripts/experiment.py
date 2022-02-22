#!/usr/bin/env python
import gym 
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir) 
from utils.env_utils import SafetyGymEnv

DEFAULT_ENV_CONFIG_P = dict(
    action_repeat=1,
    max_episode_length=300,
    use_dist_reward=False,
    stack_obs=False,
)
DEFAULT_ENV_CONFIG_C = dict(
    action_repeat=2,
    max_episode_length=150,
    use_dist_reward=False,
    stack_obs=False,
)

def main(robot, task, algo, seed, exp_name, cpu, wrapper):

    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    # Hyperparameters
    exp_name = algo + '_' + robot + task
    if robot=='Car':
        num_steps = 1e7
        steps_per_epoch = 30000
        max_ep_len = 150
        env_config = DEFAULT_ENV_CONFIG_C
    else: #Point
        num_steps = 1e7
        steps_per_epoch = 30000
        max_ep_len = 300
        env_config = DEFAULT_ENV_CONFIG_P

    epochs = int(num_steps / steps_per_epoch)
    save_freq = 10
    target_kl = 0.01
    cost_lim = 5

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    exp_name = exp_name or (algo + '_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    env_name = 'Safexp-'+robot+task+'-v0'

    if not wrapper:
        env_fn=lambda: gym.make(env_name)
    else:
        env_fn=lambda: SafetyGymEnv(robot=robot, task=task[:-1], level=int(task[-1]), seed=seed, config=env_config)

    algo(env_fn=env_fn,
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
         epochs=epochs,
         max_ep_len=max_ep_len,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs
         )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--wrapper', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    exp_name = args.exp_name if not(args.exp_name=='') else None
    main(args.robot, args.task, args.algo, args.seed, exp_name, args.cpu, args.wrapper)
