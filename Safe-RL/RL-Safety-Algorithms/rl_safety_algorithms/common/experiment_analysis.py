import os
import json
import numpy as np
from collections import namedtuple, OrderedDict
import matplotlib.pyplot as plt
import warnings
import os
import gym
import torch
import atexit
import pandas
import rl_safety_algorithms.common.mpi_tools as mpi_tools


def find_nested_item(obj, key):
    if key in obj:
        return obj[key]
    for (k, v) in obj.items():
        if isinstance(v, dict):
            return find_nested_item(v, key)  # added return statement


def search_nested_key(dic, key, default=None):
    """Return a value corresponding to the specified key in the (possibly
    nested) dictionary d. If there is no item with that key, return
    default.
    """
    stack = [iter(dic.items())]
    while stack:
        for k, v in stack[-1]:
            if isinstance(v, dict):
                stack.append(iter(v.items()))
                break
            elif k == key:
                return v
        else:
            stack.pop()
    return default


def get_file_contents(file_path: str,
                      skip_header: bool = False):
    """Open the file with given path and return Python object."""
    assert os.path.isfile(file_path), 'No file exists at: {}'.format(file_path)

    if file_path.endswith('.json'):  # return dict
        with open(file_path, 'r') as fp:
            data = json.load(fp)

    elif file_path.endswith('.csv'):
        if skip_header:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        else:
            data = np.loadtxt(file_path, delimiter=",")
        if len(data.shape) == 2:  # use pandas for tables..
            data = pandas.read_csv(file_path)
    else:
        raise NotImplementedError
    return data


def get_experiment_paths(path: str
                         ) -> tuple:
    """ Walk through path recursively and find experiment log files.

        Note:
            In a directory must exist a config.json and metrics.json file, such
            that path is detected.

    Parameters
    ----------
    path
        Path that is walked through recursively.

    Raises
    ------
    AssertionError
        If no experiment runs where found.

    Returns
    -------
    list
        Holding path names to directories.
    """
    experiment_paths = []
    for root, dirs, files in os.walk(path):  # walk recursively trough basedir
        config_json_in_dir = False
        metrics_json_in_dir = False
        for file in files:
            if file.endswith("config.json"):
                config_json_in_dir = True
            if file.endswith("progress.csv"):
                metrics_json_in_dir = True
        if config_json_in_dir and metrics_json_in_dir:
            experiment_paths.append(root)

    assert experiment_paths, f'No experiments found at: {path}'

    return tuple(experiment_paths)


class EnvironmentEvaluator(object):
    def __init__(self, log_dir, log_costs=True):

        self.log_dir = log_dir
        self.env = None
        self.ac = None
        self.log_costs = log_costs

        # open returns.csv file at the beginning to avoid disk access errors
        # on our HPC servers...
        if mpi_tools.proc_id() == 0:
            os.makedirs(log_dir, exist_ok=True)
            self.ret_file_name = 'returns.csv'
            self.ret_file = open(os.path.join(log_dir, self.ret_file_name), 'w')
            # Register close function is executed for normal program termination
            atexit.register(self.ret_file.close)
            if log_costs:
                self.c_file_name = 'costs.csv'
                self.costs_file = open(os.path.join(log_dir, self.c_file_name), 'w')
                atexit.register(self.costs_file.close)
        else:
            self.ret_file_name = None
            self.ret_file = None
            if log_costs:
                self.c_file_name = None
                self.costs_file = None

    def close(self):
        """Close opened output files immediately after training in order to
        avoid number of open files overflow. Avoids the following error:
        OSError: [Errno 24] Too many open files
        """
        if mpi_tools.proc_id() == 0:
            self.ret_file.close()
            if self.log_costs:
                self.costs_file.close()

    def eval(self, env, ac, num_evaluations):
        """ Evaluate actor critic module for given number of evaluations.
        """
        self.ac = ac
        self.ac.eval()  # disable exploration noise

        if isinstance(env, gym.Env):
            self.env = env
        elif isinstance(env, str):
            self.env = gym.make(env)
        else:
            raise TypeError('Env is not of type: str, gym.Env')

        size = mpi_tools.num_procs()
        num_local_evaluations = num_evaluations // size
        returns = np.zeros(num_local_evaluations, dtype=np.float32)
        costs = np.zeros(num_local_evaluations, dtype=np.float32)
        ep_lengths = np.zeros(num_local_evaluations, dtype=np.float32)

        for i in range(num_local_evaluations):
            returns[i], ep_lengths[i], costs[i] = self.eval_once()
        # Gather returns from all processes
        # Note: only root process owns valid data...
        returns = list(mpi_tools.gather_and_stack(returns))
        costs = list(mpi_tools.gather_and_stack(costs))

        # now write returns as column into output file...
        if mpi_tools.proc_id() == 0:
            self.write_to_file(self.ret_file, contents=returns)
            print('Saved to:', os.path.join(self.log_dir, self.ret_file_name))
            if self.log_costs:
                self.write_to_file(self.costs_file, contents=costs)
            print(f'Mean Ret: { np.mean(returns)} \t'
                  f'Mean EpLen: {np.mean(ep_lengths)} \t'
                  f'Mean Costs: {np.mean(costs)}')

        self.ac.train()  # back to train mode
        return np.array(returns), np.array(ep_lengths), np.array(costs)

    def eval_once(self):
        assert not self.ac.training, 'Call actor_critic.eval() beforehand.'
        done = False
        x = self.env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0

        while not done:
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, value, *_ = self.ac(obs)
            x, r, done, info = self.env.step(action)
            ret += r
            costs += info.get('cost', 0.)
            episode_length += 1

        return ret, episode_length, costs

    @staticmethod
    def write_to_file(file, contents: list):
        if mpi_tools.proc_id() == 0:
            column = [str(x) for x in contents]
            file.write("\n".join(column) + "\n")
            file.flush()


class ParameterContainer:
    def __init__(self):
        self.parameter_settings = dict()

    @classmethod
    def _parameters_to_string(
            cls,
            params: tuple
    ) -> str:
        return '/'.join([str(x) for x in params])

    def __contains__(self, items):
        if isinstance(items, list):
            item_as_string = self._parameters_to_string(items)
            return item_as_string in self.parameter_settings
        elif isinstance(items, str):
            return items in self.parameter_settings
        else:
            raise NotImplementedError

    def add(self,
            items: tuple,
            values: np.ndarray
            ) -> None:
        items_string = self._parameters_to_string(items)
        if items_string in self:
            self.parameter_settings[items_string].append(values)
        else:
            self.parameter_settings[items_string] = [values]

    def all_items(self):
        """ returns all stored data."""
        return self.parameter_settings.items()

    def clear(self):
        self.parameter_settings = dict()

    def get_data(self):
        return self.parameter_settings


class ExperimentAnalyzer(object):
    def __init__(self, base_dir, data_file_name, debug=True):
        self.base_dir = base_dir
        self.data_file_name = data_file_name
        self.param_container = ParameterContainer()
        self.exp_paths = get_experiment_paths(base_dir)
        print(f'Found {len(self.exp_paths)} files at: {base_dir}') if debug else None
        self.filtered_paths = list()

    # def _find_nested_item(self, obj, key):
    #     if key in obj:
    #         return obj[key]
    #     for (k, v) in obj.items():
    #         if isinstance(v, dict):
    #             return self._find_nested_item(v, key)  # added return statement
    #
    # def _search_nested_key(self, dic, key, default=None):
    #     """Return a value corresponding to the specified key in the (possibly
    #     nested) dictionary d. If there is no item with that key, return
    #     default.
    #     """
    #     stack = [iter(dic.items())]
    #     while stack:
    #         for k, v in stack[-1]:
    #             if isinstance(v, dict):
    #                 stack.append(iter(v.items()))
    #                 break
    #             elif k == key:
    #                 return v
    #         else:
    #             stack.pop()
    #     return default

    def _fill_param_container(self,
                              params: tuple,
                              filter: dict
                              ) -> None:
        """ Fill up the internal data container with the data created in the
            experiments.
            If filter dict is provided, only those keys are processed.
         """
        for path in self.exp_paths:
            # fetch config.json first and determine parameter values#
            config_file_path = os.path.join(path, 'config.json')
            config = get_file_contents(config_file_path)

            # Check if filter matches to current config
            if filter is not None:  # iterates when filter is not empty
                skip_path = False
                for key, v in filter.items():
                    found_value = search_nested_key(config, key)
                    # skip if filter does not match
                    if found_value is None:
                        skip_path = True
                        warnings.warn(
                            f'Filter {filter} did not apply at: {path}')
                    if found_value != v:
                        skip_path = True  # skip if filter does not match
                if skip_path:
                    continue

            fetched_config_values = OrderedDict()

            for param in params:
                # config typically holds nested dictionaries...
                is_present = search_nested_key(config, param)

                if is_present:
                    fetched_config_values[param] = is_present
                # if parameter is not found, return Not A Number
                else:
                    fetched_config_values[param] = np.NaN
            vals = fetched_config_values.values()

            data_file_path = os.path.join(path, self.data_file_name)
            try:
                data = get_file_contents(data_file_path)
            except ValueError:
                data = get_file_contents(data_file_path, skip_header=True)
            except AssertionError:
                print(f'WARNING: no thing found at: {data_file_path}')
                continue
            # assert isinstance(data, np.ndarray)
            # only add data if file is not empty
            if data.size > 0:
                self.param_container.add(vals, data)
            else:
                msg = f'Empty file: {data_file_path}'
                warnings.warn(msg)

    def get_data(self,
                 params: tuple,
                 filter: dict
                 ) -> dict:
        """fetch data from the experiment directories and merge
        runs with same parameters"""

        self.param_container.clear()
        # fill internal data container first
        self._fill_param_container(params, filter=filter)

        return self.param_container.get_data()

    def get_mean_return(self,
                        params: tuple,
                        filter: dict
                        ):
        data_dic = self.get_data(params, filter=filter)
        return_scores = []
        for values in data_dic.values():
            for vs in values:  # iterate over individual runs
                # print(vs)
                mean_return = np.nanmean(vs)  # build mean of return.csv
                if mean_return.size == 0:
                    warnings.warn(f'Found return array of size zero.')
                return_scores.append(mean_return)
        return np.mean(return_scores)


if __name__ == '__main__':
    base_dir = '/Users/sven/Experiments/RCC/exp_1/data'
    actors = ['mlp', 'linear', 'rbf', 'elm']

    for actor in actors:

        if actor == 'mlp':
            parameters = ('activation', 'hidden_sizes', 'pi_lr')
            filter = dict(actor=actor)
        elif actor == 'linear':
            parameters = ('actor', 'pi_lr', 'lam', 'mini_batch_size')
            filter = dict(actor='linear')
        elif actor == 'rbf':
            parameters = ('actor', 'bandwidth', 'num_features')
            filter = dict(actor='rbf')
        elif actor == 'elm':
            parameters = ('activation', 'pi_lr', 'hidden_sizes')
            filter = dict(actor=actor)
        else:
            raise NotImplementedError

        ea = ExperimentAnalyzer(base_dir)
        data_dict = ea.get_data(parameters, filter)

        # iterate over parameter setting of container and calculate mean performance
        setting_with_values = dict()
        for k, vs in data_dict.items():
            means = [np.mean(v) for v in vs]
            setting_with_values[k] = np.mean(vs)

        output = json.dumps(setting_with_values,
                            separators=(',', ':\t'),
                            indent=4,
                            sort_keys=True)
        print('=' * 65, f'\nActor architecture: {actor}')
        print(f'Parameters: {parameters}')
        print(output)
        print('Best value:')
        max_key = max(setting_with_values, key=setting_with_values.get)
        print(max_key, ":", setting_with_values[max_key])

        all_values_max_key = data_dict[max_key]
        stacked_values = np.concatenate(all_values_max_key, axis=1)
        plt.plot(np.mean(stacked_values, axis=1), label=actor)
    plt.legend()
    plt.show()
