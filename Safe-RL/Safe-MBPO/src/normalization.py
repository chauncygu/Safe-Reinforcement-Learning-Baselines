import torch

from .torch_util import Module


class Normalizer(Module):
    def __init__(self, dim, epsilon=1e-6):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.zeros(dim))

    def fit(self, X):
        assert torch.is_tensor(X)
        assert X.dim() == 2
        assert X.shape[1] == self.dim
        self.mean.data.copy_(X.mean(dim=0))
        self.std.data.copy_(X.std(dim=0))

    def forward(self, x):
        return (x - self.mean) / (self.std + self.epsilon)

    def unnormalize(self, normal_X):
        return self.mean + (self.std * normal_X)