import numpy as np
from scipy.stats import norm
from safemdp.grid_world import (GridWorld)


__all__ = ['SafetyObj']


class SafetyObj(GridWorld):
    """Safety Object in MDPs."""
    def __init__(self, gp, world_shape, step_size, beta, safety, h, S0,
                 S_hat0, L, update_dist):
        super(SafetyObj, self).__init__(gp, world_shape, step_size, beta,
                                        safety, h, S0, S_hat0, L, update_dist)

        self.S_bar = np.empty(S0.shape, dtype=np.bool)

    def probability_safe(self):
        """ Compute the probability that state and action pairs are safe.

        Parameters
        ----------
        x: safemdp class

        Returns
        -------
        prob_safety: probability of a state being safe
        """
        n, m = self.world_shape
        mu = 0.5 * (self.u + self.l)
        std = (self.u - mu) / self.beta

        prob_safety = np.zeros((n*m, 5))
        for i in range(n*m):
            for j in range(5):
                # For the boundary, standard deviation is defined as 0.
                # In this case, probability of being safety is also set to 0.
                if std[i, j] == 0:
                    if mu[i, j] >= self.h:
                        prob_safety[i, j] = 1
                    else:
                        prob_safety[i, j] = 0
                else:
                    prob_safety[i, j] = norm.sf(self.h, mu[i, j], std[i, j])

        # For "stay" action, transition probability is zero (always safe)
        prob_safety[:, 0] = 1

        return prob_safety

    def compute_S_bar(self):
        """ compute the optimistic safe space """
        return self.compute_S_hat()

    def update_opt_sets(self):
        """
        Update the sets S, S_hat and G taking with the available observation
        """
        self.update_confidence_interval()
        self.S[:] = self.u >= self.h

        self.compute_S_bar()
        self.S_bar = self.S_hat.copy()
