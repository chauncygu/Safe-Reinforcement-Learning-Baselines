import numpy as np


__all__ = ['RewardObj']


class RewardObj(object):
    """Reward Object in MDPs.

    Parameters
    ----------
    gp_r: GPy.core.GPRegression
        A Gaussian process model that can be used to determine the reward.
    beta_r: float
        The confidence interval used by the GP model.
    """
    def __init__(self, gp_r, beta_r):
        super(RewardObj, self).__init__()

        # Scalar for gp confidence intervals
        self.beta = beta_r
        # GP model
        self.gp = gp_r

    def add_gp_observations(self, x_new, y_new):
        """Add observations to the gp."""
        # Update GP with observations
        self.gp.set_XY(np.vstack((self.gp.X, x_new)),
                       np.vstack((self.gp.Y, y_new)))
