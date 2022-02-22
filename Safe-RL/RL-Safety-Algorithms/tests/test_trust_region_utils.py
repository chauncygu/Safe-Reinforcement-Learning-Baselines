import unittest
import numpy as np
import gym
from rl_safety_algorithms.algs import core
import rl_safety_algorithms.algs.utils as U
import torch


def create_random_positive_definite_matrix(n):
    """from
    https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab"""
    A = np.random.uniform(-1, 1, (n, n))
    # construct a symmetric matrix using either
    A = 0.5 * (A + A.transpose())
    # since A(i,j) < 1 by construction and a symmetric diagonally dominant
    # matrix is symmetric positive definite, which can be ensured by adding nI
    A += n * np.eye(n)
    return A


def cg(Avp, b, nsteps, residual_tol=1e-10, eps=1e-6):
    x = np.zeros_like(b)
    r = b - Avp(x)
    p = r.copy()
    rdotr = np.dot(r, r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    verbose = False

    for i in range(nsteps):
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = Avp(p)
        alpha = rdotr / (np.dot(p, z) + eps)
        x += alpha * p
        r -= alpha * z
        new_rdotr = np.dot(r, r)
        if np.sqrt(new_rdotr) < residual_tol:
            break
        mu = new_rdotr / (rdotr + eps)
        p = r + mu * p
        rdotr = new_rdotr

    return x


class TestTrustRegionUtilities(unittest.TestCase):

    def test_cg_simple(self):
        A = np.array([[4., 1], [1, 3]])
        b = np.array([1, 2.])

        # solution is: x: [0.09090909 0.63636364]

        def Avp(p): return torch.matmul(torch.Tensor(A), p)

        # find x in Ax = b
        x = np.linalg.inv(A) @ b
        x_1 = U.conjugate_gradients(Avp, torch.Tensor(b), nsteps=15)

        def Mvp(p): return A @ p

        x_2 = cg(Mvp, b, nsteps=15)

        # print('x:', x)
        # print('x_1:', x_1)
        # print('x_2:', x_2)

        self.assertTrue(np.allclose(x, x_1.numpy()))
        self.assertTrue(np.allclose(x, x_2))

    def test_conjugate_gradient(self):
        """ test CG algorithms.
        See Also
        https://en.wikipedia.org/wiki/Conjugate_gradient_method
        """
        M = 150
        A = create_random_positive_definite_matrix(M)
        b = np.random.random(M) * 0.01

        # find x in Ax = b
        x = np.linalg.inv(A) @ b

        def Avp(p): return torch.matmul(torch.Tensor(A), p)

        x_1 = U.conjugate_gradients(Avp, torch.Tensor(b), nsteps=25)

        def Mvp(p): return A @ p

        x_2 = cg(Mvp, b, nsteps=25)

        # print('x:', x)
        # print('x_1:', x_1)
        # print('x_2:', x_2)

        self.assertTrue(np.allclose(x, x_1.numpy(), atol=1e-5))
        self.assertTrue(np.allclose(x, x_2, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
