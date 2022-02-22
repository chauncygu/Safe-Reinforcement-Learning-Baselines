import h5py
import numpy as np
import torch
from tqdm import trange

from .env.util import env_dims, isdiscrete, get_max_episode_steps
from .env.batch import BaseBatchedEnv, ProductEnv
from .torch_util import device, Module, torchify, random_indices
from .util import discounted_sum


class SampleBuffer(Module):
    COMPONENT_NAMES = ('states', 'actions', 'next_states', 'rewards', 'dones')

    def __init__(self, state_dim, action_dim, capacity, discrete_actions=False,
                 device=device):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.discrete_actions = discrete_actions
        self.device = device

        self._bufs = {}
        self.register_buffer('_pointer', torch.tensor(0, dtype=torch.long))

        if discrete_actions:
            assert action_dim == 1
            action_dtype = torch.int
            action_shape = []
        else:
            action_dtype = torch.float
            action_shape = [action_dim]

        components = (
            ('states', torch.float, [state_dim]),
            ('actions', action_dtype, action_shape),
            ('next_states', torch.float, [state_dim]),
            ('rewards', torch.float, []),
            ('dones', torch.bool, [])
        )
        for name, dtype, shape in components:
            self._create_buffer(name, dtype, shape)

    @classmethod
    def from_state_dict(cls, state_dict, device=device):
        # Must have same keys
        assert set(state_dict.keys()) == {*(f'_{name}' for name in cls.COMPONENT_NAMES), '_pointer'}
        states, actions = state_dict['_states'], state_dict['_actions']

        # Check that length (size of first dimension) matches
        l = len(states)
        for name in cls.COMPONENT_NAMES:
            tensor = state_dict[f'_{name}']
            assert torch.is_tensor(tensor)
            assert len(tensor) == l

        # Capacity, dimensions, and type of action inferred from state_dict
        buffer = cls(state_dim=states.shape[1], action_dim=actions.shape[1], capacity=l,
                     discrete_actions=(not actions.dtype.is_floating_point),
                     device=device)
        buffer.load_state_dict(state_dict)
        return buffer

    @classmethod
    def from_h5py(cls, path, device=device):
        with h5py.File(path, 'r') as f:
            data = {name: torchify(np.array(f[name]), device=device) for name in f.keys()}
        n_steps = len(data['rewards'])
        if 'next_states' not in data:
            all_states = data['states']
            assert len(all_states) == n_steps + 1
            data['states'] = all_states[:-1]
            data['next_states'] = all_states[1:]
        for v in data.values():
            assert len(v) == n_steps

        # Capacity, dimensions, and type of action inferred from h5py file
        states, actions = data['states'], data['actions']
        buffer = cls(state_dim=states.shape[1], action_dim=actions.shape[1], capacity=n_steps,
                     discrete_actions=(not actions.dtype.is_floating_point),
                     device=device)
        buffer.extend(*(data[name] for name in cls.COMPONENT_NAMES))
        return buffer

    def __len__(self):
        return min(self._pointer, self.capacity)

    def _create_buffer(self, name, dtype, shape):
        assert name not in self._bufs
        _name = f'_{name}'
        buffer_shape = [self.capacity, *shape]
        buffer = torch.empty(*buffer_shape, dtype=dtype, device=self.device)
        self.register_buffer(_name, buffer)
        self._bufs[name] = buffer

    def _get1(self, name):
        buf = self._bufs[name]
        if self._pointer <= self.capacity:
            return buf[:self._pointer]
        else:
            i = self._pointer % self.capacity
            return torch.cat([buf[i:], buf[:i]])

    def get(self, *names, device=device, as_dict=False):
        """
        Retrieves data from the buffer. Pass a vararg list of names.
        What is returned depends on how many names are given:
            * a list of all components if no names are given
            * a single component if one name is given
            * a list with one component for each name otherwise
        """
        if len(names) == 0:
            names = self.COMPONENT_NAMES
        bufs = [self._get1(name).to(device) for name in names]
        if as_dict:
            return dict(zip(names, bufs))
        else:
            return bufs if len(bufs) > 1 else bufs[0]

    def append(self, **kwargs):
        assert set(kwargs.keys()) == set(self.COMPONENT_NAMES)
        i = self._pointer % self.capacity
        for name in self.COMPONENT_NAMES:
            self._bufs[name][i] = kwargs[name]
        self._pointer += 1

    def extend(self, **kwargs):
        assert set(kwargs.keys()) == set(self.COMPONENT_NAMES)
        batch_size = len(list(kwargs.values())[0])
        assert batch_size <= self.capacity, 'We do not support extending by more than buffer capacity'
        i = self._pointer % self.capacity
        end = i + batch_size
        if end <= self.capacity:
            for name in self.COMPONENT_NAMES:
                self._bufs[name][i:end] = kwargs[name]
        else:
            fit = self.capacity - i
            overflow = end - self.capacity
            # Note: fit + overflow = batch_size
            for name in self.COMPONENT_NAMES:
                buf, arg = self._bufs[name], kwargs[name]
                buf[-fit:] = arg[:fit]
                buf[:overflow] = arg[-overflow:]
        self._pointer += batch_size

    def sample(self, batch_size, replace=True, device=device, include_indices=False):
        indices = torch.randint(len(self), [batch_size], device=device) if replace else \
            random_indices(len(self), size=batch_size, replace=False)
        bufs = [self._bufs[name][indices].to(device) for name in self.COMPONENT_NAMES]
        return (bufs, indices) if include_indices else bufs

    def split_episodes(self, max_length):
        """Use to split a single buffer into episodes that make it up.
        Note: this method computes the episode structure assuming the samples in the dataset are ordered sequentially.
        If this is not the case, the returned "episodes" are meaningless."""
        assert self._pointer <= self.capacity, 'split_episodes will give bad results on a circular buffer'
        states, actions, next_states, rewards, dones = self.get()
        n = len(self)
        done_indices = list(map(int, dones.nonzero()))
        episodes = []
        offset = 0
        used_indices = set()
        while offset < n:
            max_end = min(offset + max_length, n)
            actual_end = max_end
            if len(done_indices) > 0:
                next_done_index = done_indices[0]
                if next_done_index <= max_end:
                    actual_end = next_done_index + 1
                    done_indices.pop(0)

            episode_indices = set(range(offset, actual_end))
            assert len(episode_indices) > 0, 'Cannot have empty episode!'
            assert len(used_indices & episode_indices) == 0, 'Indices should not overlap!'
            traj_buffer = SampleBuffer(self.state_dim, self.action_dim, len(episode_indices),
                                       discrete_actions=self.discrete_actions)
            traj_buffer.extend(
                states[offset:actual_end],
                actions[offset:actual_end],
                next_states[offset:actual_end],
                rewards[offset:actual_end],
                dones[offset:actual_end]
            )
            episodes.append(traj_buffer)

            offset = actual_end
            used_indices |= episode_indices

        # Sanity checks
        assert len(done_indices) == 0
        assert sum(len(traj) for traj in episodes) == n
        assert used_indices == set(range(n))
        return episodes

    def trimmed_copy(self):
        new_buffer = self.__class__(self.state_dim, self.action_dim, len(self),
                                    discrete_actions=self.discrete_actions)
        new_buffer.extend(*self.get())
        return new_buffer

    def save_h5py(self, path, remove_duplicate_states=True):
        data = self.get(as_dict=True, device='cpu')
        if remove_duplicate_states:
            next_states = data.pop('next_states')
            data['states'] = torch.cat((
                data['states'], next_states[-1].unsqueeze(0)
            ))

        with h5py.File(path, 'w') as f:
            for k, v in data.items():
                f.create_dataset(k, data=v.numpy())



def concat_sample_buffers(buffers):
    state_dim, action_dim = buffers[0].state_dim, buffers[0].action_dim
    discrete_actions = buffers[0].discrete_actions
    total_capacity = 0
    for buffer in buffers:
        assert buffer.state_dim == state_dim
        assert buffer.action_dim == action_dim
        assert buffer.discrete_actions == discrete_actions
        total_capacity += len(buffer)
    combined_buffer = SampleBuffer(state_dim, action_dim, total_capacity,
                                   discrete_actions=discrete_actions)
    for buffer in buffers:
        combined_buffer.extend(*buffer.get())
    return combined_buffer


class UnbatchedStepSampler:
    """
    For sampling individual steps/transitions. Not suitable for episodes (use sample_episodes below)
    """
    def __init__(self, env):
        self.env = env
        self.samples_taken = 0
        self.reset()

    def reset(self):
        self._state = self.env.reset()
        self._return = 0.
        self._n_steps = 0

    def run(self, policy,
            n_steps=None,
            given_buffer=None,
            eval=False,
            progress_bar=False,
            post_step_callback=None,
            post_episode_callback=None):
        state_dim, action_dim = env_dims(self.env)
        max_episode_steps = get_max_episode_steps(self.env)
        buffer = given_buffer if given_buffer is not None else SampleBuffer(state_dim, action_dim, n_steps)
        range_fn = trange if progress_bar else range
        for _ in range_fn(n_steps):
            action = policy.act1(self._state, eval)
            next_state, reward, done, info = self.env.step(action)
            self.samples_taken += 1
            self._return += reward
            self._n_steps += 1
            buffer.append(self._state, action, next_state, reward, done)
            if callable(post_step_callback):
                post_step_callback(buffer)
            timeout = self._n_steps >= max_episode_steps
            if timeout or done:
                if callable(post_episode_callback):
                    post_episode_callback(self._return, self._n_steps, buffer)
                self.reset()
            else:
                self._state.copy_(next_state)
        return buffer


class BatchedStepSampler:
    """
    For sampling individual steps/transitions. Not suitable for episodes (use sample_episodes below)
    """
    def __init__(self, env):
        self.env = env if isinstance(env, BaseBatchedEnv) else ProductEnv([env])
        self.samples_taken = 0
        self.reset()

    def reset(self):
        self.set_states(self.env.reset(), set_env_states=False)

    def set_states(self, states, set_env_states=True):
        self._states = states.clone()
        if set_env_states:
            self.env.set_states(states)
        self._n_steps = torch.zeros(self.env.n_envs, dtype=int, device=device)

    def run(self, policy,
            n_samples=None, n_steps=None,
            given_buffer=None,
            eval=False,
            progress_bar=False,
            post_step_callback=None):
        if n_samples is None and n_steps is None:
            raise ValueError('StepSampler.run() must be passed n_samples or n_steps')
        elif n_samples is not None and n_steps is not None:
            raise ValueError('StepSampler.run() cannot be passed both n_samples and n_steps')
        elif n_samples is None:
            assert isinstance(n_steps, int)
            n_samples = n_steps * self.env.n_envs
        elif n_steps is None:
            assert isinstance(n_samples, int)
            assert n_samples % self.env.n_envs == 0, f'n_samples ({n_samples}) is not divisible by n_envs {self.env.n_envs}'
            n_steps = n_samples // self.env.n_envs

        state_dim, action_dim = env_dims(self.env)
        max_episode_steps = get_max_episode_steps(self.env)
        buffer = given_buffer if given_buffer is not None else SampleBuffer(state_dim, action_dim, n_samples)
        range_fn = trange if progress_bar else range
        for _ in range_fn(n_steps):
            actions = policy.act(self._states, eval)
            next_states, rewards, dones, infos = self.env.step(actions)
            buffer.extend(self._states, actions, next_states, rewards, dones)
            if callable(post_step_callback):
                post_step_callback(buffer)
            self._n_steps += 1
            timeouts = self._n_steps >= max_episode_steps
            indices = torch.nonzero(dones | timeouts).flatten()
            if len(indices) > 0:
                next_states = next_states.clone()
                next_states[indices] = self.env.partial_reset(indices)
                self._n_steps[indices] = 0
            self._states.copy_(next_states)
        self.samples_taken += n_samples
        return buffer


def sample_episode_unbatched(env, policy, eval=False,
                             max_steps=None,
                             post_step_callback=None,
                             recorder=None, render=False):
    state_dim, action_dim = env_dims(env)
    discrete_actions = isdiscrete(env.action_space)
    T = max_steps if max_steps is not None else get_max_episode_steps(env)
    episode = SampleBuffer(state_dim, 1 if discrete_actions else action_dim, T,
                           discrete_actions=discrete_actions)
    state = env.reset()

    if recorder:
        recorder.capture_frame()
    elif render:
        env.unwrapped.render()

    for t in range(T):
        action = policy.act1(state, eval=eval)
        next_state, reward, done, info = env.step(action)
        episode.append(state, action, next_state, reward, done)

        if post_step_callback is not None:
            post_step_callback()

        if recorder:
            recorder.capture_frame()
        elif render:
            env.unwrapped.render()

        if done:
            break
        else:
            state = next_state

    return episode


def sample_episodes_batched(env, policy, n_traj, eval=False):
    if not isinstance(env, BaseBatchedEnv):
        env = ProductEnv([env])

    state_dim, action_dim = env_dims(env)
    discrete_actions = isdiscrete(env.action_space)
    traj_buffer_factory = lambda: SampleBuffer(state_dim, 1 if discrete_actions else action_dim, env._max_episode_steps,
                                               discrete_actions=discrete_actions)
    traj_buffers = [traj_buffer_factory() for _ in range(env.n_envs)]
    complete_episodes = []

    states = env.reset()
    while True:
        actions = policy.act(states, eval=eval)
        next_states, rewards, dones, infos = env.step(actions)

        _next_states = next_states.clone()
        reset_indices = []

        for i in range(env.n_envs):
            traj_buffers[i].append(states=states[i], actions=actions[i], next_states=next_states[i],
                                   rewards=rewards[i], dones=dones[i])
            if dones[i] or len(traj_buffers[i]) == env._max_episode_steps:
                complete_episodes.append(traj_buffers[i])
                if len(complete_episodes) == n_traj:
                    # Done!
                    return complete_episodes

                reset_indices.append(i)
                traj_buffers[i] = traj_buffer_factory()

        if reset_indices:
            reset_indices = np.array(reset_indices)
            _next_states[reset_indices] = env.partial_reset(reset_indices)

        states.copy_(_next_states)


def evaluate_policy(env, policy, n_episodes=10, discount=1, reward_function=None):
    returns = []
    for episode in sample_episodes_batched(env, policy, n_episodes, eval=True):
        states, actions, next_states, rewards, dones = episode.get()
        if reward_function is not None:
            rewards = reward_function(states, actions, next_states)
        returns.append(discounted_sum(rewards, discount))
    return torchify(returns)