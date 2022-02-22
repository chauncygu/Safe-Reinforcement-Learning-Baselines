import numpy as np
# from mujoco_safety_gym.envs import mujoco_env
from macpo.envs.safety_ma_mujoco.safety_multiagent_mujoco import mujoco_env
from gym import utils
import mujoco_py as mjp


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        mjp.functions.mj_rnePostConstraint(self.sim.model,
                                           self.sim.data)  #### calc contacts, this is a mujoco py version mismatch issue with mujoco200
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0

        ### safety stuff
        yposafter = self.get_body_com("torso")[1]
        ywall = np.array([-5, 5])
        if xposafter < 20:
            y_walldist = yposafter - xposafter * np.tan(30 / 360 * 2 * np.pi) + ywall
        elif xposafter > 20 and xposafter < 60:
            y_walldist = yposafter + (xposafter - 40) * np.tan(30 / 360 * 2 * np.pi) - ywall
        elif xposafter > 60 and xposafter < 100:
            y_walldist = yposafter - (xposafter - 80) * np.tan(30 / 360 * 2 * np.pi) + ywall
        else:
            y_walldist = yposafter - 20 * np.tan(30 / 360 * 2 * np.pi) + ywall

        obj_cost = (abs(y_walldist) < 1.8).any() * 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        body_quat = self.data.get_body_xquat('torso')
        z_rot = 1 - 2 * (
                    body_quat[1] ** 2 + body_quat[2] ** 2)  ### normally xx-rotation, not sure what axes mujoco uses
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0 \
                  and z_rot >= -0.7
        done = not notdone
        done_cost = done * 1.0
        cost = np.clip(obj_cost + done_cost, 0, 1)
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            cost_obj=obj_cost,
            cost_done=done_cost,
            cost=cost,
        )

    def _get_obs(self):
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        if x < 20:
            y_off = y - x * np.tan(30 / 360 * 2 * np.pi)
        elif x > 20 and x < 60:
            y_off = y + (x - 40) * np.tan(30 / 360 * 2 * np.pi)
        elif x > 60 and x < 100:
            y_off = y - (x - 80) * np.tan(30 / 360 * 2 * np.pi)
        else:
            y_off = y - 20 * np.tan(30 / 360 * 2 * np.pi)

        return np.concatenate([
            self.sim.data.qpos.flat[2:-42],
            self.sim.data.qvel.flat[:-36],
            [x / 5],
            [y_off],
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qpos[-42:] = self.init_qpos[-42:]
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        qvel[-36:] = self.init_qvel[-36:]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5