import unittest
import numpy as np
import torch
from rl_safety_algorithms.common.online_mean_std import OnlineMeanStd
import rl_safety_algorithms.common.mpi_tools as mpi_tools
import sys


class TestOnlineMeanStd(unittest.TestCase):
    @staticmethod
    def get_data(M, N, epoch, pid=mpi_tools.proc_id()):
        """Returns data matrix of shape MxN."""
        start = pid*10000 + 4 * epoch
        stop = pid*10000 + M * N + 4 * epoch
        step = 1
        return 0.001 * np.arange(start, stop, step).reshape((M, N))

    @staticmethod
    def perform_single_pass(rms, input_shape) -> bool:
        x = torch.from_numpy(np.random.normal(size=input_shape))
        rms(x)  # perform one call
        return True

    def test_mpi_matrix_updates(self):
        """ OnlineMeanStd module is updated with a batch of vector inputs,
            i.e. inputs of shape M x N.
            Note that std dev might differ more than 1e-5 when epochs > 10.
        """
        epochs = 20
        cores = 4
        T = 500
        obs_shape = (6, )

        try:
            if mpi_tools.mpi_fork(n=cores):
                sys.exit()
            pid = mpi_tools.proc_id()
            # print(f'Here is process:', mpi_tools.proc_id())

            # === calculation through online updates
            rms = OnlineMeanStd(shape=obs_shape)
            for ep in range(epochs):
                # shape of batch: T x obs_shape
                obs_batch = self.get_data(T, obs_shape[0], ep, pid)
                rms.update(obs_batch)
            mpi_mean = rms.mean.numpy()
            mpi_std = rms.std.numpy()

            # ===== calculate ground truths
            obs_list = [self.get_data(T, obs_shape[0], ep, pid=i)
                        for i in range(cores)
                        for ep in range(epochs)
                        ]
            obs = np.vstack(obs_list)
            gt_mean = np.mean(obs, axis=0)
            gt_std = np.std(obs, axis=0)

            # if mpi_tools.proc_id() == 0:
            #     print('gt_mean:', gt_mean)
            #     print('mpi_mean:', mpi_mean)
            #     print('gt_std:', gt_std)
            #     print('mpi_std:', mpi_std)
            self.assertTrue(np.allclose(mpi_mean, gt_mean))
            self.assertTrue(np.allclose(mpi_std, gt_std))
            self.assertTrue(self.perform_single_pass(rms, obs_shape))

        # necessary to prevent system exit with error...
        except SystemExit:
            print('Join....')

    def test_mpi_vector_updates(self):
        """Test with vector inputs.
        Note that std dev might differ more than 1e-5 when epochs > 10.
        """
        epochs = 10
        cores = 4
        T = 500
        input_shape = (1,)

        try:
            mpi_tools.mpi_fork(n=cores)
            p = mpi_tools.proc_id()
            # print(f'Here is process:', mpi_tools.proc_id())

            # === calculation through online updates
            rms = OnlineMeanStd(shape=input_shape)
            for ep in range(epochs):
                # shape of batch: T x input_shape
                vector_input = self.get_data(T, input_shape[0], ep, p).flatten()
                rms.update(vector_input)
            mpi_mean = rms.mean.numpy()
            mpi_std = rms.std.numpy()

            # ===== calculate ground truths
            obs_list = [self.get_data(T, input_shape[0], ep, pid=i)
                        for i in range(cores)
                        for ep in range(epochs)
                        ]
            obs = np.vstack(obs_list)
            gt_mean = np.mean(obs, axis=0)
            gt_std = np.std(obs, axis=0)

            # if mpi_tools.proc_id() == 0:
            #     print('gt_mean:', gt_mean)
            #     print('mpi_mean:', mpi_mean)
            #     print('gt_std:', gt_std)
            #     print('mpi_std:', mpi_std)
            self.assertTrue(np.allclose(mpi_mean, gt_mean))
            self.assertTrue(np.allclose(mpi_std, gt_std, rtol=1e-2))
            self.assertTrue(self.perform_single_pass(rms, input_shape))

        # necessary to prevent system exit with error...
        except SystemExit:
            print('Join....')


if __name__ == '__main__':
    unittest.main()
