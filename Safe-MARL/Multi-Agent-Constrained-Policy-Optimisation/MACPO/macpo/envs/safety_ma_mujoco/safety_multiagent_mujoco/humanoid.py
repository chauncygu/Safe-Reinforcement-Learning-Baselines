import numpy as np
# from mujoco_safety_gym.envs import mujoco_env
from macpo.envs.safety_ma_mujoco.safety_multiagent_mujoco import mujoco_env
from gym import utils
import mujoco_py as mjp


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        x = data.qpos.flat[0]
        y = data.qpos.flat[1]
        if x < 20:
            y_off = y - x * np.tan(30 / 360 * 2 * np.pi)
        elif x > 20 and x < 60:
            y_off = y + (x - 40) * np.tan(30 / 360 * 2 * np.pi)
        elif x > 60 and x < 100:
            y_off = y - (x - 80) * np.tan(30 / 360 * 2 * np.pi)
        else:
            y_off = y - 20 * np.tan(30 / 360 * 2 * np.pi)

        return np.concatenate([data.qpos.flat[2:-42],
                               data.qvel.flat[:-36],
                               [x / 5],
                               [y_off]])

        # return np.concatenate([data.qpos.flat[2:],
        #                        data.qvel.flat,
        #                        data.cinert.flat,
        #                        data.cvel.flat,
        #                        data.qfrc_actuator.flat,
        #                        data.cfrc_ext.flat])

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        mjp.functions.mj_rnePostConstraint(self.sim.model,
                                           self.sim.data)  #### calc contacts, this is a mujoco py version mismatch issue with mujoco200
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

        yposafter = self.get_body_com("torso")[1]
        ywall = np.array([-2.3, 2.3])
        if pos_after < 20:
            y_walldist = yposafter - pos_after * np.tan(30 / 360 * 2 * np.pi) + ywall
        elif pos_after > 20 and pos_after < 60:
            y_walldist = yposafter + (pos_after - 40) * np.tan(30 / 360 * 2 * np.pi) - ywall
        elif pos_after > 60 and pos_after < 100:
            y_walldist = yposafter - (pos_after - 80) * np.tan(30 / 360 * 2 * np.pi) + ywall
        else:
            y_walldist = yposafter - 20 * np.tan(30 / 360 * 2 * np.pi) + ywall
        obj_cost = (abs(y_walldist) < 1.8).any() * 1.0

        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        done_cost = done * 1.0
        cost = np.clip(obj_cost + done_cost, 0, 1)

        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost,
                                                   cost_obj=obj_cost,
                                                   cost_done=done_cost,
                                                   cost=cost,
                                                   )

    def reset_model(self):
        c = 0.01
        # self.set_state(
        #     self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
        #     self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        # )
        # return self._get_obs()
        qpos = self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq)
        qpos[-42:] = self.init_qpos[-42:]
        qvel = self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv, )
        qvel[-36:] = self.init_qvel[-36:]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20