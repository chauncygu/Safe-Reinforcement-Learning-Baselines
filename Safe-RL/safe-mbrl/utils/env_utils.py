'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-05-29 19:49:11
@LastEditTime: 2020-07-29 21:25:14
@Description:
'''
import numpy as np
import re
import gym
import safety_gym

ROBOTS = ['Point','Car', 'Doggo']
TASKS = ['Goal', 'Button']

XYZ_SENSORS = dict(
    Point=['velocimeter'],
    Car=['velocimeter'],#,'accelerometer'],#,'ballquat_rear', 'right_wheel_vel', 'left_wheel_vel'],
    Doggo=['velocimeter','accelerometer']
    )

ANGLE_SENSORS = dict(
    Point=['gyro','magnetometer'],
    Car=['magnetometer','gyro'],
    Doggo=['magnetometer','gyro']
    )

CONSTRAINTS = dict(
    Goal=['vases', 'hazards'],
    Button=['hazards','gremlins','buttons'],)

DEFAULT_CONFIG = dict(
    action_repeat=5,
    max_episode_length=1000,
    use_dist_reward=False,
    stack_obs=False,
)

class Dict2Obj(object):
    #Turns a dictionary into a class
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])
        
    def __repr__(self):
        return "%s" % self.__dict__

class SafetyGymEnv():
    def __init__(self, robot='Point', task='Goal', level=1, seed=0, config=DEFAULT_CONFIG):
        self.robot = robot.capitalize()
        self.task = task.capitalize()
        assert self.robot in ROBOTS, "can not recognize the robot type {}".format(robot)
        assert self.task in TASKS, "can not recognize the task type {}".format(task)
        self.config = Dict2Obj(config)
        env_name = 'Safexp-'+self.robot+self.task+str(level)+'-v0'
        print("Creating environment: ", env_name)
        self.env = gym.make(env_name)
        self.env.seed(seed)

        print("Environment configuration: ", self.config)
        self.init_sensor()

         #for uses with ppo in baseline
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (self.env.action_space.shape[0],), dtype=np.float32)

    def init_sensor(self):
        self.xyz_sensors = XYZ_SENSORS[self.robot]
        self.angle_sensors = ANGLE_SENSORS[self.robot]
        self.constraints_name = CONSTRAINTS[self.task]
        #self.distance_name = ["goal_dist"] + [x+"_dist" for x in self.constraints_name]

        self.base_state_name = self.xyz_sensors + self.angle_sensors
        self.flatten_order = self.base_state_name + ["goal"] + self.constraints_name #+ self.distance_name

        # get state space vector size
        self.env.reset()
        obs = self.get_obs()
        #print(obs)
        self.obs_flat_size = sum([np.prod(i.shape) for i in obs.values()])
        self.state_dim = self.obs_flat_size
        if self.config.stack_obs:
            self.state_dim = self.state_dim*self.config.action_repeat
        self.key_to_slice = {}
        offset = 0
        for k in self.flatten_order:
            k_size = np.prod(obs[k].shape)
            self.key_to_slice[k] = slice(offset, offset + k_size)
            print("obs key: ", k, " slice: ", self.key_to_slice[k])
            offset += k_size

        self.base_state_dim = sum([np.prod(obs[k].shape) for k in self.base_state_name])
        self.action_dim = self.env.action_space.shape[0]
        self.key_to_slice["base_state"] = slice(0, self.base_state_dim)

    def reset(self):
        self.t = 0    # Reset internal timer
        self.env.reset()
        obs = self.get_obs_flatten()
        
        if self.config.stack_obs:
            for k in range(self.config.action_repeat):
                cat_obs = obs if k == 0 else np.concatenate((cat_obs, obs))
            return cat_obs
        else:
            return obs
    
    def step(self, action):
        # 2 dimensional numpy array, [vx, w]
        
        reward = 0
        cost = 0

        if self.config.stack_obs:
            cat_obs = np.zeros(self.config.action_repeat*self.obs_flat_size)

        for k in range(self.config.action_repeat):
            control = action
            state, reward_k, done, info = self.env.step(control)
            if self.config.use_dist_reward:
                reward_k = self.get_dist_reward()
            reward += reward_k
            cost += info["cost"]
            self.t += 1    # Increment internal timer
            observation = self.get_obs_flatten()
            if self.config.stack_obs:
                cat_obs[k*self.obs_flat_size :(k+1)*self.obs_flat_size] = observation 
            goal_met = ("goal_met" in info.keys()) # reach the goal
            done = done or self.t == self.config.max_episode_length
            if done or goal_met:
                if k != self.config.action_repeat-1 and self.config.stack_obs:
                    for j in range(k+1,self.config.action_repeat):
                        cat_obs[j*self.obs_flat_size :(j+1)*self.obs_flat_size] = observation 
                break
        cost = 1 if cost>0 else 0

        info = {"cost":cost, "goal_met":goal_met}
        if self.config.stack_obs:
            return cat_obs, reward, done, info
        else:
            return observation, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def recenter(self, pos):
        ''' Return the egocentric XY vector to a position from the robot '''
        return self.env.ego_xy(pos)

    def dist_xy(self, pos):
        ''' Return the distance from the robot to an XY position, 3 dim or 2 dim '''
        return self.env.dist_xy(pos)

    def get_obs(self):
        '''
        We will ingnore the z-axis coordinates in every poses.
        The returned obs coordinates are all in the robot coordinates.
        '''
        obs = {}
        robot_pos = self.env.robot_pos
        goal_pos = self.env.goal_pos
        vases_pos_list = self.env.vases_pos # list of shape (3,) ndarray
        hazards_pos_list = self.env.hazards_pos # list of shape (3,) ndarray
        gremlins_pos_list = self.env.gremlins_obj_pos # list of shape (3,) ndarray
        buttons_pos_list = self.env.buttons_pos # list of shape (3,) ndarray

        ego_goal_pos = self.recenter(goal_pos[:2])
        ego_vases_pos_list = [self.env.ego_xy(pos[:2]) for pos in vases_pos_list] # list of shape (2,) ndarray
        ego_hazards_pos_list = [self.env.ego_xy(pos[:2]) for pos in hazards_pos_list] # list of shape (2,) ndarray
        ego_gremlins_pos_list = [self.env.ego_xy(pos[:2]) for pos in gremlins_pos_list] # list of shape (2,) ndarray
        ego_buttons_pos_list = [self.env.ego_xy(pos[:2]) for pos in buttons_pos_list] # list of shape (2,) ndarray
        
        # append obs to the dict
        for sensor in self.xyz_sensors:  # Explicitly listed sensors
            if sensor=='accelerometer':
                obs[sensor] = self.env.world.get_sensor(sensor)[:1] # only x axis matters
            elif sensor=='ballquat_rear':
                obs[sensor] = self.env.world.get_sensor(sensor)
            else:
                obs[sensor] = self.env.world.get_sensor(sensor)[:2] # only x,y axis matters

        for sensor in self.angle_sensors:
            if sensor == 'gyro':
                obs[sensor] = self.env.world.get_sensor(sensor)[2:] #[2:] # only z axis matters
                #pass # gyro does not help
            else:
                obs[sensor] = self.env.world.get_sensor(sensor)

        obs["vases"] = np.array(ego_vases_pos_list) # (vase_num, 2)
        obs["hazards"] = np.array(ego_hazards_pos_list) # (hazard_num, 2)
        obs["goal"] = ego_goal_pos # (2,)
        obs["gremlins"] = np.array(ego_gremlins_pos_list) # (vase_num, 2)
        obs["buttons"] = np.array(ego_buttons_pos_list) # (hazard_num, 2)
        return obs

    def get_obs_flatten(self):
        # get the flattened obs
        self.obs = self.get_obs()
        #obs_flat_size = sum([np.prod(i.shape) for i in obs.values()])
        flat_obs = np.zeros(self.obs_flat_size)
        for k in self.flatten_order:
            idx = self.key_to_slice[k]
            flat_obs[idx] = self.obs[k].flat
        return flat_obs

    def get_dist_reward(self):
        '''
        @return reward: negative distance from robot to the goal
        '''
        return -self.env.dist_goal()

    @property
    def observation_size(self):
        return self.state_dim

    @property
    def action_size(self):
        return self.env.action_space.shape[0]

    @property
    def action_range(self):
        return float(self.env.action_space.low[0]), float(self.env.action_space.high[0])

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return self.env.action_space.sample()
