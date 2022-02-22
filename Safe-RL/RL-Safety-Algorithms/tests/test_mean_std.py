import unittest
import numpy as np
import torch
from rl_safety_algorithms.common.online_mean_std import OnlineMeanStd
import rl_safety_algorithms.common.mpi_tools as mpi_tools


class TestOnlineMeanStd(unittest.TestCase):
    """ Testing the non-MPI version.
    """

    @staticmethod
    def perform_single_pass(rms, input_shape) -> bool:
        x = torch.from_numpy(np.random.normal(size=input_shape))
        rms(x)  # perform one call
        return True

    @staticmethod
    def get_data(M, N, epoch):
        """Returns data matrix of shape MxN."""
        np.random.seed(epoch)
        # start = 10000 + 4 * epoch
        # stop = pid*10000 + M * N + 4 * epoch
        data = np.random.normal(size=(M, N))
        return data
    
    def test_vector_updates(self):
        """ OnlineMeanStd module is updated with a batch of vector inputs,
            i.e. inputs of shape M x N.
            Note that std dev might differ more than 1e-5 when epochs > 10.
        """
        epochs = 20
        T = 500
        obs_shape = (1, )

        # === calculation through online updates
        rms = OnlineMeanStd(shape=obs_shape)
        for ep in range(epochs):
            # shape of batch: T x obs_shape
            vector_input = self.get_data(T, obs_shape[0], ep).flatten()
            rms.update(vector_input)
        rms_mean = rms.mean.numpy()
        rms_std = rms.std.numpy()

        # ===== calculate ground truths
        obs_list = [self.get_data(T, obs_shape[0], ep) for ep in range(epochs)]
        obs = np.vstack(obs_list)
        gt_mean = np.mean(obs, axis=0)
        gt_std = np.std(obs, axis=0)

        self.assertTrue(np.allclose(rms_mean, gt_mean))
        self.assertTrue(np.allclose(rms_std, gt_std, rtol=1e-2))
        self.assertTrue(self.perform_single_pass(rms, obs_shape))


if __name__ == '__main__':
    unittest.main()
