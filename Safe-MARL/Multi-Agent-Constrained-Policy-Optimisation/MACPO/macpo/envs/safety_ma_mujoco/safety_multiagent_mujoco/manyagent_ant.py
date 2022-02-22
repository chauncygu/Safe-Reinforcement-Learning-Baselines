import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from jinja2 import Template

import mujoco_py as mjp

import os

class ManyAgentAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        # Return Flag: Distinguish the mujoco and Wrapper env.
        self.rflag = 0
        agent_conf = kwargs.get("agent_conf")
        n_agents = int(agent_conf.split("x")[0])
        n_segs_per_agents = int(agent_conf.split("x")[1])
        n_segs = n_agents * n_segs_per_agents

        # Check whether asset file exists already, otherwise create it
        asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets',
                                                  'manyagent_ant_{}_agents_each_{}_segments.auto.xml'.format(n_agents,
                                                                                                                 n_segs_per_agents))
        # if not os.path.exists(asset_path):
        # print("Auto-Generating Manyagent Ant asset with {} segments at {}.".format(n_segs, asset_path))
        self._generate_asset(n_segs=n_segs, asset_path=asset_path)

        #asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets',git p
        #                          'manyagent_swimmer.xml')

        mujoco_env.MujocoEnv.__init__(self, asset_path, 4)
        utils.EzPickle.__init__(self)

    def _generate_asset(self, n_segs, asset_path):
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets',
                                                  'manyagent_ant.xml.template')
        with open(template_path, "r") as f:
            t = Template(f.read())
        body_str_template = """
        <body name="torso_{:d}" pos="-1 0 0">
           <!--<joint axis="0 1 0" name="nnn_{:d}" pos="0.0 0.0 0.0" range="-1 1" type="hinge"/>-->
            <geom density="100" fromto="1 0 0 0 0 0" size="0.1" type="capsule"/>
            <body name="front_right_leg_{:d}" pos="0 0 0">
              <geom fromto="0.0 0.0 0.0 0.0 0.2 0.0" name="aux1_geom_{:d}" size="0.08" type="capsule"/>
              <body name="aux_2_{:d}" pos="0.0 0.2 0">
                <joint axis="0 0 1" name="hip1_{:d}" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="right_leg_geom_{:d}" size="0.08" type="capsule"/>
                <body pos="-0.2 0.2 0">
                  <joint axis="1 1 0" name="ankle1_{:d}" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="right_ankle_geom_{:d}" size="0.08" type="capsule"/>
                </body>
              </body>
            </body>
            <body name="back_leg_{:d}" pos="0 0 0">
              <geom fromto="0.0 0.0 0.0 0.0 -0.2 0.0" name="aux2_geom_{:d}" size="0.08" type="capsule"/>
              <body name="aux2_{:d}" pos="0.0 -0.2 0">
                <joint axis="0 0 1" name="hip2_{:d}" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="back_leg_geom_{:d}" size="0.08" type="capsule"/>
                <body pos="-0.2 -0.2 0">
                  <joint axis="-1 1 0" name="ankle2_{:d}" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="third_ankle_geom_{:d}" size="0.08" type="capsule"/>
                </body>
              </body>
            </body>
        """

        body_close_str_template ="</body>\n"
        actuator_str_template = """\t     <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip1_{:d}" gear="150"/>
                                          <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle1_{:d}" gear="150"/>
                                          <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip2_{:d}" gear="150"/>
                                          <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle2_{:d}" gear="150"/>\n"""

        body_str = ""
        for i in range(1,n_segs):
            body_str += body_str_template.format(*([i]*16))
        body_str += body_close_str_template*(n_segs-1)

        actuator_str = ""
        for i in range(n_segs):
            actuator_str += actuator_str_template.format(*([i]*8))

        rt = t.render(body=body_str, actuators=actuator_str)
        with open(asset_path, "w") as f:
            f.write(rt)
        pass

    def step(self, a):
        xposbefore = self.get_body_com("torso_0")[0]
        self.do_simulation(a, self.frame_skip)

        #ADDED
        mjp.functions.mj_rnePostConstraint(self.sim.model,
                                           self.sim.data)  #### calc contacts, this is a mujoco py version mismatch issue with mujoco200

        xposafter = self.get_body_com("torso_0")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0

        ### ADDED safety stuff
        yposafter = self.get_body_com("torso_0")[1]
        ywall = np.array([-2.3, 2.3])
        if xposafter < 20:
            y_walldist = yposafter - xposafter * np.tan(30 / 360 * 2 * np.pi) + ywall
        elif xposafter>20 and xposafter<60:
            y_walldist = yposafter + (xposafter-40)*np.tan(30/360*2*np.pi) - ywall
        elif xposafter>60 and xposafter<100:
            y_walldist = yposafter - (xposafter-80)*np.tan(30/360*2*np.pi) + ywall
        else:
            y_walldist = yposafter - 20*np.tan(30/360*2*np.pi) + ywall
        obj_cost = (abs(y_walldist) < 1.8).any() * 1.0

        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        #### ADDED
        body_quat = self.data.get_body_xquat('torso_0')
        z_rot = 1-2*(body_quat[1]**2+body_quat[2]**2)  ### normally xx-rotation, not sure what axes mujoco uses

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0\
            and z_rot>=-0.7 #ADDED

        done = not notdone
        # print("done", done)
        print("y_walldist", y_walldist)
        print("obj_cost", obj_cost)
        #ADDED
        done_cost = done * 1.0
        cost = np.clip(obj_cost + done_cost, 0, 1)
        # print("reward", reward)
        # print("cost-manyagent_ant.py",cost)
        ob = self._get_obs()
        if self.rflag == 0:
            self.rflag += 1
            return ob, reward, done, dict(
                cost=cost,
                reward_forward=forward_reward, #
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
                cost_obj=obj_cost,  # ADDED
                cost_done=done_cost,  # ADDED
            )
        else:
            return ob, reward, done, dict(
                cost=cost,
                reward_forward=forward_reward, # cost = cost,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
                cost_obj=obj_cost, #ADDED
                cost_done=done_cost, #ADDED
            )

    def _get_obs(self):
        x = self.sim.data.qpos.flat[0] #ADDED
        y = self.sim.data.qpos.flat[1] #ADDED

        #ADDED
        if x<20:
            y_off = y - x*np.tan(30/360*2*np.pi)
        elif x>20 and x<60:
            y_off = y + (x-40)*np.tan(30/360*2*np.pi)
        elif x>60 and x<100:
            y_off = y - (x-80)*np.tan(30/360*2*np.pi)
        else:
            y_off = y - 20*np.tan(30/360*2*np.pi)
        # return np.concatenate([
        #     self.sim.data.qpos.flat[2:],
        #     self.sim.data.qvel.flat,
        #     # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        # ])
        return np.concatenate([
            self.sim.data.qpos.flat[2:-42], # size = 3
            self.sim.data.qvel.flat[:-36], # size = 6
            [x/5],
            [y_off],
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qpos[-42:] = self.init_qpos[-42:]
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        qvel[-36:] = self.init_qvel[-36:]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5