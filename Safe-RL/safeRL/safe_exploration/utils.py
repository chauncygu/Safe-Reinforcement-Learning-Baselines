# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

import numpy as np
import matplotlib.pyplot as plt

def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)



def batched_weighted_sum(weights, vecs, batch_size):
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size),
                                         itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float64),
                        np.asarray(batch_vecs, dtype=np.float64))
        num_items_summed += len(batch_weights)
    return total, num_items_summed

def plot_info(param_dict, logdir):
    for key, value in param_dict.items():
        x = value[0]
        y = value[1]
        x_name = value[2]
        y_name = value[3]
        print(x,y)
        plt.plot(x, y)
        plt.title(key)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.savefig((logdir + "/plot_"  +key + ".png"))
        plt.clf()


