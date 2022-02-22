"""
Min - max normalization
"""


def minmax_norm(x, min_x, max_x):
    """
    This function normalizes data
    :param x: input data
    :param min_x: minimum value
    :param max_x: output data
    :return: normalized input data x_norm
    """
    x_norm = (x - min_x)/(max_x - min_x)

    return x_norm


def minmax_norm_back(x_norm, min_x, max_x):
    """
    This function denormalizes data
    :param x_norm: input data
    :param min_x: minimum value
    :param max_x: output data
    :return: real input data x
    """
    x = x_norm * (max_x - min_x) + min_x

    return x
