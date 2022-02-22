"""
DDPG Agent- Actor Critic method for Reinforcement Learning
"""
from casadi import *
import numpy as np
from Abgabe.Normalize.MinMax import minmax_norm

class DDPG_Agent:
    """
    Class for Actor Critic DDPG algorithm
    """
    def __init__(self, actor, critic, critic_action_input, memory, constraints_net, random_process=None, batch_size=64, gamma=.99,
                 nb_steps_warmup_critic=100, nb_steps_warmup_actor=100, target_model_update=.001, constraint=False,
                 nb_disturbance=0, nb_tracing=0):

        # parameters
        self.compiled = False
        self.random_process = random_process
        self.batch_size = batch_size
        self.gamma = gamma
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.target_model_update = target_model_update
        self.training = False
        self.critic_action_input = critic_action_input
        self.constraint = False
        if constraint == 'SafetyLayer':
            self.constraint = True
        self.nb_disturbance = nb_disturbance
        self.nb_tracing = nb_tracing

        self.step = 0
        self.episodes = 0
        self.plot_val = []

        # networks
        self.actor = actor
        self.critic = critic
        
        # constraints -> To be adapted when model is changed
        self.constraint_T_1 = constraints_net[0]
        self.constraint_T_2 = constraints_net[1]

        self.constraint_E_1 = constraints_net[2]
        self.constraint_E_2 = constraints_net[3]

        # init target networks Q' and mu'
        self.target_actor = actor.target_model
        self.target_critic = critic.target_model

        # init replay buffer
        self.memory = memory

        # init solver for constraint optimization
        if self.constraint is True:
            param = MX.sym('param', 1, 4)

            a = MX.sym('a', 2, 1)
            J = 0.5*((sqrt(a[0]-param[0,0]))**2)**2 + 0.5*((sqrt(a[1]-param[0,1]))**2)**2
            G = []

            # To be adapted when model is changed
            constraints_en_up = self.constraint_E_1.nn_casadi(self.constraint_E_1.weights, self.constraint_E_1.config,
                                                         horzcat(param[0, 3], a[0], a[1]))
            constraints_temp_up = self.constraint_T_1.nn_casadi(self.constraint_T_1.weights, self.constraint_T_1.config,
                                                           horzcat(param[0, 2], a[0], a[1]))
            constraints_en_low = self.constraint_E_2.nn_casadi(self.constraint_E_2.weights, self.constraint_E_2.config,
                                                         horzcat(param[0, 3], a[0], a[1]))
            constraints_temp_low = self.constraint_T_2.nn_casadi(self.constraint_T_2.weights, self.constraint_T_2.config,
                                                           horzcat(param[0, 2], a[0], a[1]))
            G.append(constraints_temp_low)
            G.append(constraints_temp_up)
            G.append(constraints_en_low)
            G.append(constraints_en_up)

            self.lbax = np.array([[-1], [-1]])
            self.ubax = np.array([[1], [1]])
            self.ubg = np.array([[inf], [0.85], [inf], [0.98]])
            self.lbg = np.array([[0.15], [-inf], [0], [-inf]])

            opts = {}
            opts["ipopt.tol"] = 1e-5
            opts["ipopt.print_level"] = 0
            opts["print_time"] = 0
            
            nlp = {'x': vertcat(a), 'f': J, 'g': horzcat(*G), 'p':  param}
            self.solver = nlpsol('solver', 'ipopt', nlp, opts)

        self.reset_states()

    def compile(self):
        """
        Function to compile neural networks
        :return:
        """
        self.actor.model.reset_states()
        self.critic.model.reset_states()
        self.target_actor.reset_states()
        self.target_critic.reset_states()
        self.compiled = True

    def set_weights_target(self):
        """
        Asserting weights of target networks
        :return:
        """

        # actor target
        actor_weights = self.actor.model.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            target_actor_weights[i] = self.target_model_update * actor_weights[i] + (1 - self.target_model_update) \
                                      * target_actor_weights[i]
        self.target_actor.set_weights(target_actor_weights)

        # critic target
        critic_weights = self.critic.model.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            target_critic_weights[i] = self.target_model_update * critic_weights[i] + (1 - self.target_model_update) \
                                       * target_critic_weights[i]
        self.target_critic.set_weights(target_critic_weights)
        return

    def fit(self, env, nb_episodes, nb_max_episode_steps):
        """
        training of the neural networks
        :param env:
        :param nb_episodes:
        :param nb_max_episode_steps:
        :return:
        """
        done = False
        self.training = True
        self.step = 0
        self.plot_val = []
        nb_actions = env.action_space.shape[0]
        nb_observations = env.observation_space.shape[0]
        mean_rew = 0
        reward_print = []
        dist = []
        tracing = []
        violations = []
        try:
            for eps in range(nb_episodes):

                # initial observation state
                observation_now = env.reset_states()
                self.reset_states()

                for t in range(nb_max_episode_steps):
                    # select action a_t
                    action = self.forward(env, observation_now)
                    if self.constraint is True:
                        action_new = self.constrain_action(action, env)
                    else:
                        action_new = action

                    # execute action
                    observation_next, reward = env.step(action_new)
                    # TODO: change to reset episode
                    '''
                    if self.constraint is True:
                        # if states stagnates, reset episode
                        if abs(observation_next[0,0] - observation_now[0,0]) < 5e-2 and observation_next[0,0] != env.T_ref:
                            break
                    '''
                    if self.nb_disturbance > 0:
                        dist = env.get_future_dist(self.nb_disturbance)
                    if self.nb_tracing > 0:
                        tracing = env.get_future_tracking()

                    if self.nb_disturbance > 0 or self.nb_tracing > 0:
                        future = dist + tracing
                    mean_rew += reward
                    if self.step % nb_max_episode_steps == 0:
                        mean_rew = mean_rew / nb_max_episode_steps
                        print("Episode",  eps, "Mean Reward:", mean_rew)
                        reward_print.append(mean_rew)
                        if abs(mean_rew) < 0.05 and self.step > 400:  # abs(mean_rew) < 0.05
                            return reward_print, violations
                        mean_rew = 0

                    # store transition in buffer R (state, action, reward, new_state, done)
                    if self.nb_disturbance + self.nb_tracing > 0:
                        self.memory.add_with_dist(np.reshape(observation_now, (nb_observations,)), np.reshape(action, (nb_actions,)),
                                    reward, np.reshape(observation_next, (nb_observations,)), done, np.reshape(future,(self.nb_disturbance + self.nb_tracing,)) )
                    else:
                        self.memory.add(np.reshape(observation_now, (nb_observations,)), np.reshape(action, (nb_actions,)),
                                    reward, np.reshape(observation_next, (nb_observations,)), done)

                    observation_now = observation_next
                    if self.step > self.nb_steps_warmup_actor:
                        # sample a random mini batch of N transitions from R
                        batch = self.memory.sample(self.batch_size)
                        states = np.asarray([e[0] for e in batch])
                        actions = np.asarray([e[1] for e in batch])
                        rewards = np.asarray([e[2] for e in batch])
                        new_states = np.asarray([e[3] for e in batch])

                        # Set y = r + gamma * Q_target(s_i+1, mu'(s_i+1|theta'_mu)|theta_Q')
                        if self.nb_disturbance + self.nb_tracing > 0:
                            future = np.asarray([e[5] for e in batch])

                            target_actions = self.target_actor.predict(np.concatenate((new_states, future),axis=1))
                            target_q_values = self.target_critic.predict([np.concatenate((new_states, future),axis=1), target_actions])

                        else:
                            target_actions = self.target_actor.predict(new_states)
                            target_q_values = self.target_critic.predict([new_states, target_actions])

                        targets = rewards + self.gamma * np.reshape(target_q_values, (self.batch_size,))

                        if self.nb_disturbance + self.nb_tracing > 0:
                            # Update critic by minimizing the loss
                            self.critic.model.fit([np.concatenate((states, future),axis=1), actions], np.reshape(targets, (self.batch_size, 1)),
                                              batch_size=self.batch_size, verbose=0)
                            # Update actor by using the sampled policy gradient
                            actions_for_grad = self.actor.model.predict(np.concatenate((states, future), axis=1))
                            gradient_q = self.critic.criticGradients(np.concatenate((states, future),axis=1), actions_for_grad)
                            self.actor.train_model(np.concatenate((states, future), axis=1), gradient_q)
                        else:
                            # Update critic by minimizing the loss
                            self.critic.model.fit([states, actions], np.reshape(targets, (self.batch_size, 1)),
                                              batch_size=self.batch_size, verbose=0)
                            # Update actor by using the sampled policy gradient
                            actions_for_grad = self.actor.model.predict(states)
                            gradient_q = self.critic.criticGradients(states, actions_for_grad)
                            self.actor.train_model(states, gradient_q)

                        # update target networks
                        self.set_weights_target()
                    self.step += 1
                self.plot_val.append(env.get_val())
                violations.append(env.constraint_violations)
                self.episodes += 1
        except KeyboardInterrupt:
            print('Training interrupted.')
        return reward_print, violations

    def reset_states(self):
        # reset random process
        if self.random_process is not None:
            self.random_process.reset_states()
        # reset network states
        if self.compiled:
            self.actor.model.reset_states()
            self.critic.model.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()
        return

    def forward(self, env, observation):
        # Select an action
        observation = np.reshape(observation, (1, env.observation_space.shape[0]))
        dist = []
        tracing = []
        if self.nb_disturbance > 0:
            dist = env.get_future_dist(self.nb_disturbance)
        if self.nb_tracing > 0:
            tracing = env.get_future_tracking()
        if self.nb_tracing + self.nb_disturbance > 0:
            future = np.array([dist + tracing])
            future = np.reshape(future.T, (1, self.nb_tracing + self.nb_disturbance))
            action = self.actor.model.predict(np.concatenate((observation, future), axis=1))
        else:
            action = self.actor.model.predict(observation)

        # add noise
        if self.random_process is not None and self.training is True:
            noise = self.random_process.sample()
            action += noise
        return action

    def test(self, env, nb_episodes, nb_steps_per_episode):
        self.training = False
        for episode in range(nb_episodes):
            observation = env._get_obs()
            for step in range(nb_steps_per_episode):
                action = self.forward(env, observation)
                if self.constraint is True:
                    action = self.constrain_action(action, env)
                observation, r = env.step(action)
                print("Reward:", r)
        return

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.model.load_weights(actor_filepath)
        self.critic.model.load_weights(critic_filepath)
        self.set_weights_target()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.model.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.model.save_weights(critic_filepath, overwrite=overwrite)

    def constrain_action(self, action, env):
        # solve optimization problem

        state = env.x
        state_new = np.copy(state)
        state_new[0, 0] = minmax_norm(state[0, 0], env.lbx[0], env.ubx[0])
        state_new[1, 0] = minmax_norm(state[1, 0], env.lbx[1], env.ubx[1])
        param = np.reshape(np.concatenate((np.reshape(action, (1,2))[0], np.reshape(state_new, (1,2))[0])), (1,4))
        res = self.solver( lbx=self.lbax, ubx=self.ubax, lbg=self.lbg, ubg=self.ubg, p= param)  # p = mu
        new_action = res['x']
        return new_action

    def get_plot_val(self):
        return self.plot_val, self.episodes
