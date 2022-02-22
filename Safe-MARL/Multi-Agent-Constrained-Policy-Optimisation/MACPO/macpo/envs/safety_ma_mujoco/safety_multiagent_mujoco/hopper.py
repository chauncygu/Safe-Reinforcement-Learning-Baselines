import numpy as np
from macpo.envs.safety_ma_mujoco.safety_multiagent_mujoco import mujoco_env
from gym import utils
import mujoco_py as mjp


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)
        self.last_mocx = 5  #### vel readings are super noisy for mocap weld

    def step(self, a):
        posbefore = self.sim.data.qpos[3]
        t = self.data.time
        pos = (t + np.sin(t)) + 3
        self.data.set_mocap_pos('mocap1', [pos, 0, 0.5])

        mjp.functions.mj_rnePostConstraint(self.sim.model,
                                           self.sim.data)  #### calc contacts, this is a mujoco py version mismatch issue with mujoco200
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[3:6]
        alive_bonus = 1.0

        mocapx = self.sim.data.qpos[0]
        xdist = mocapx - posafter
        cost = int(np.abs(xdist) < 1)

        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        # done = not (np.isfinite(s).all() and (np.abs(s[5:]) < 100).all() and
        #             (height > .7) and (abs(ang) < .2))

        done = not (
                np.isfinite(s).all()
                and (np.abs(s[2:]) < 100).all()
                and (height > 0.7)
                and (abs(ang) < 0.2)
        )
        print("np.isfinite(s).all()", np.isfinite(s).all())
        print("np.abs(s[5:])", (np.abs(s[2:]) < 100).all())
        print("height", (height > 0.7))
        print("abs(ang) ", (abs(ang) < 0.2))

        ob = self._get_obs()
        return ob, reward, done, dict(cost=cost)

    def _get_obs(self):
        x = self.sim.data.qpos[3]
        mocapx = self.sim.data.qpos[0]
        mocvel = 1 + np.cos(self.data.time)
        mocacc = -np.sin(self.data.time)
        return np.concatenate([
            self.sim.data.qpos.flat[4:],
            np.clip(self.sim.data.qvel[3:].flat, -10, 10),
            [mocvel],
            [mocacc],
            [mocapx - x],
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def last_mocap_x(self):
        return self.last_mocx

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20