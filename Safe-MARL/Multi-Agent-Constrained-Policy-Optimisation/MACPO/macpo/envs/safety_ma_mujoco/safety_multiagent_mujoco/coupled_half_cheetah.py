import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from macpo.envs.safety_ma_mujoco.safety_multiagent_mujoco import mujoco_env
import os
import mujoco_py as mjp
from gym import error, spaces

class CoupledHalfCheetah(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'coupled_half_cheetah.xml'), 5)
        utils.EzPickle.__init__(self)

    def step(self, action):

        #ADDED
        # xposbefore = self.sim.data.qpos[1]
        # t = self.data.time
        # wall_act = .02 * np.sin(t / 3) ** 2 - .004
        # mjp.functions.mj_rnePostConstraint(self.sim.model,
        #                                    self.sim.data)  #### calc contacts, this is a mujoco py version mismatch issue with mujoco200
        # action_p_wall = np.concatenate((np.squeeze(action), [wall_act]))
        # self.do_simulation(action_p_wall, self.frame_skip)
        # xposafter = self.sim.data.qpos[1]
        # wallpos = self.data.get_geom_xpos("obj_geom")[0]
        # wallvel = self.data.get_body_xvelp("obj1")[0]
        # xdist = wallpos - xposafter
        # obj_cost = int(np.abs(xdist) < 2)
        # if obj_cost > 0:
        #     self.model.geom_rgba[9] = [1.0, 0, 0, 1.0]
        # else:
        #     self.model.geom_rgba[9] = [1.0, 0.5, 0.5, .8]
        # ob = self._get_obs()
        # reward_ctrl = - 0.1 * np.square(action).sum()
        # reward_run = (xposafter - xposbefore) / self.dt
        # reward = reward_ctrl + reward_run
        # done = False




        # xposbefore1 = self.sim.data.qpos[0]
        # xposbefore2 = self.sim.data.qpos[len(self.sim.data.qpos) // 2]
        # print("self.sim.data.qpos", self.sim.data.qpos)

        xposbefore1 = self.get_body_com("torso")[0]
        xposbefore2 = self.get_body_com("torso2")[0]

        yposbefore1 = self.get_body_com("torso")[1]
        yposbefore2 = self.get_body_com("torso2")[1]

        # ADDED
        t = self.data.time
        wall_act = .02 * np.sin(t / 3) ** 2 - .004
        mjp.functions.mj_rnePostConstraint(self.sim.model,
                                           self.sim.data)  #### calc contacts, this is a mujoco py version mismatch issue with mujoco200
        action_p_wall = np.concatenate((np.squeeze(action), [wall_act]))
        # print("action_p_wall", np.array(action_p_wall).shape)
        # print("action", np.array(action).shape)
        # print("self.frame_skip", self.frame_skip)
        self.do_simulation(action_p_wall, self.frame_skip)

        # self.do_simulation(action, self.frame_skip)
        # xposafter1 = self.sim.data.qpos[0]
        # xposafter2 = self.sim.data.qpos[len(self.sim.data.qpos)//2]
        xposafter1 = self.get_body_com("torso")[0]
        xposafter2 = self.get_body_com("torso2")[0]

        yposafter1 = self.get_body_com("torso")[1]
        yposafter2 = self.get_body_com("torso2")[1]

        # ADDED
        wallpos = self.data.get_geom_xpos("obj_geom")[0]
        # wallpos1 = self.data.get_geom_xpos("obj_geom1")[0]
        y_wallpos1 = self.data.get_geom_xpos("wall1")[1]
        y_wallpos2 = self.data.get_geom_xpos("wall2")[1]
        # print("x_wallpos1 = self.data.get_geom_xpos", x_wallpos1)
        # print("x_wallpos2 = self.data.get_geom_xpos", x_wallpos2)
        wallvel = self.data.get_body_xvelp("obj1")[0]
        xdist = np.abs(wallpos - xposafter1)  #+ np.abs(wallpos - xposafter2) #+ (wallpos1 - xposafter1)  + (wallpos1 - xposafter2)
        obj_cost = 0 # or int(np.abs(wallpos1 - xposafter2) < 5) or int(np.abs(wallpos1 - xposafter2) < 5)\
        #
        if int(np.abs(wallpos - xposafter1) < 5) or int(np.abs(wallpos - xposafter2) < 5) \
                or int(np.abs(y_wallpos1 - yposafter1) < 5) or int(np.abs(y_wallpos2 - yposafter2) < 5):
            obj_cost = 1

        # obj_cost = int(np.abs(xdist) < 5)
        # print("xposbefore1", xposbefore1)
        # print("xposbefore2", xposbefore2)
        # print("yposafter1", yposafter1)
        # print("yposafter2", yposafter2)
        # print("np.abs(x_wallpos1 - yposafter1)", np.abs(x_wallpos1 - yposafter1))
        # print("xposafter1", xposafter1)
        # print("xposafter2", xposafter2)
        # print("wallpos", wallpos)
        # print("wallpos1", wallpos1)
        # print("xdist", xdist)
        # print("(wallpos1 - xposafter2)", (wallpos1 - xposafter2))
        # print("(wallpos - xposafter1)", (wallpos - xposafter1))
        # print("(wallpos - xposafter2)", (wallpos - xposafter2))
        if obj_cost > 0:
            self.model.geom_rgba[9] = [1.0, 0, 0, 1.0]
        else:
            self.model.geom_rgba[9] = [1.0, 0.5, 0.5, .8]
        ob = self._get_obs()

        ob = self._get_obs()
        reward_ctrl1 = - 0.1 * np.square(action[0:len(action)//2]).sum()
        reward_ctrl2 = - 0.1 * np.square(action[len(action)//2:]).sum()
        reward_run1 = (xposafter1 - xposbefore1)/self.dt
        reward_run2 = (xposafter2 - xposbefore2) / self.dt
        reward = (reward_ctrl1 + reward_ctrl2)/2.0 + (reward_run1 + reward_run2)/2.0
        done = False
        return ob, reward, done, dict(cost=obj_cost, reward_run1=reward_run1, reward_ctrl1=reward_ctrl1,
                                      reward_run2=reward_run2, reward_ctrl2=reward_ctrl2)

    def _get_obs(self):

        #AADED
        wallvel = self.data.get_body_xvelp("obj1")[0]
        wall_f = .02 * np.sin(self.data.time / 3) ** 2 - .004
        xdist = (self.data.get_geom_xpos("obj_geom")[0] - self.sim.data.qpos[1]) / 10

        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[1:],
            [wallvel],
            [wall_f],
            np.clip([xdist], -5, 5),
        ])

        # return np.concatenate([
        #     self.sim.data.qpos.flat[1:],
        #     self.sim.data.qvel.flat,
        # ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_env_info(self):
        return {"episode_limit": self.episode_limit}

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low, high = low[:-1], high[:-1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space