import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import display, HTML
from mpl_toolkits.mplot3d import Axes3D

from safe_learning.utilities import (with_scope, get_storage, set_storage,
                                     get_feed_dict)


__all__ = ['plot_lyapunov_1d', 'plot_triangulation', 'show_graph']


# An object to store graph elements
_STORAGE = {}


@with_scope('plot_lyapunov_1d')
def plot_lyapunov_1d(lyapunov, true_dynamics, legend=False):
    """Plot the lyapunov function of a 1D system

    Parameters
    ----------
    lyapunov : instance of `Lyapunov`
    true_dynamics : callable
    legend : bool, optional
    """
    sess = tf.get_default_session()
    feed_dict = get_feed_dict(sess.graph)

    # Get the storage (specific to the lyapunov function)
    storage = get_storage(_STORAGE, index=lyapunov)

    if storage is None:
        # Lyapunov function
        states = lyapunov.discretization.all_points
        actions = lyapunov.policy(states)
        next_states = lyapunov.dynamics(states, actions)
        v_bounds = lyapunov.v_decrease_confidence(states, next_states)
        true_next_states = true_dynamics(states, actions, noise=False)
        delta_v_true, _ = lyapunov.v_decrease_confidence(states,
                                                         true_next_states)

        storage = [('states', states),
                   ('next_states', next_states),
                   ('v_bounds', v_bounds),
                   ('true_next_states', true_next_states),
                   ('delta_v_true', delta_v_true)]
        set_storage(_STORAGE, storage, index=lyapunov)
    else:
        (states, next_states, v_bounds,
         true_next_states, delta_v_true) = storage.values()

    extent = [np.min(states), np.max(states)]
    safe_set = lyapunov.safe_set
    threshold = lyapunov.threshold(states)

    # Create figure axes
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Format axes
    axes[0].set_title('GP model of the dynamics')
    axes[0].set_xlim(extent)
    axes[1].set_xlim(extent)
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel(r'Upper bound of $\Delta V(x)$')
    axes[1].set_title(r'Determining stability with $\Delta V(x)$')

    # Plot dynamics
    axes[0].plot(states,
                 true_next_states.eval(feed_dict=feed_dict),
                 color='black', alpha=0.8)

    mean, bound = sess.run(next_states, feed_dict=feed_dict)
    axes[0].fill_between(states[:, 0],
                         mean[:, 0] - bound[:, 0],
                         mean[:, 0] + bound[:, 0],
                         color=(0.8, 0.8, 1))

    if hasattr(lyapunov.dynamics, 'X'):
        axes[0].plot(lyapunov.dynamics.X[:, 0],
                     lyapunov.dynamics.Y[:, 0],
                     'x', ms=8, mew=2)

    v_dot_mean, v_dot_bound = sess.run(v_bounds, feed_dict=feed_dict)
    # # Plot V_dot
    print(v_dot_mean.shape)
    print(v_dot_bound.shape)
    plt.fill_between(states[:, 0],
                     v_dot_mean[:, 0] - v_dot_bound[:, 0],
                     v_dot_mean[:, 0] + v_dot_bound[:, 0],
                     color=(0.8, 0.8, 1))

    threshold_plot = plt.plot(extent, [threshold, threshold],
                              'k-.', label=r'Safety threshold ($L \tau$ )')

    # # Plot the true V_dot or Delta_V
    delta_v = delta_v_true.eval(feed_dict=feed_dict)
    v_dot_true_plot = axes[1].plot(states[:, 0],
                                   delta_v,
                                   color='k',
                                   label=r'True $\Delta V(x)$')

    # # Create twin axis
    ax2 = axes[1].twinx()
    ax2.set_ylabel(r'$V(x)$')
    ax2.set_xlim(extent)

    # # Plot Lyapunov function
    V_unsafe = np.ma.masked_where(safe_set, lyapunov.values)
    V_safe = np.ma.masked_where(~safe_set, lyapunov.values)
    unsafe_plot = ax2.plot(states, V_unsafe,
                           color='b',
                           label=r'$V(x)$ (unsafe, $\Delta V(x) > L \tau$)')
    safe_plot = ax2.plot(states, V_safe,
                         color='r',
                         label=r'$V(x)$ (safe, $\Delta V(x) \leq L \tau$)')

    if legend:
        lns = unsafe_plot + safe_plot + threshold_plot + v_dot_true_plot
        labels = [x.get_label() for x in lns]
        plt.legend(lns, labels, loc=4, fancybox=True, framealpha=0.75)

    # Create helper lines
    if np.any(safe_set):
        max_id = np.argmax(lyapunov.values[safe_set])
        x_safe = states[safe_set][max_id]
        y_range = axes[1].get_ylim()
        axes[1].plot([x_safe, x_safe], y_range, 'k-.')
        axes[1].plot([-x_safe, -x_safe], y_range, 'k-.')

    # Show plot
    plt.show()


def plot_triangulation(triangulation, axis=None, three_dimensional=False,
                       xlabel=None, ylabel=None, zlabel=None, **kwargs):
    """Plot a triangulation.
    
    Parameters
    ----------
    values: ndarray
    axis: optional
    three_dimensional: bool, optional
        Whether to plot 3D
        
    Returns
    -------
    axis:
        The axis on which we plotted.
    """
    values = triangulation.parameters[0].eval()

    if three_dimensional:
        if axis is None:
            axis = Axes3D(plt.figure())

        # Get the simplices and plot
        delaunay = triangulation.tri
        state_space = triangulation.discretization.all_points
        
        simplices = delaunay.simplices(np.arange(delaunay.nsimplex))
        c = axis.plot_trisurf(state_space[:, 0], state_space[:, 1], values[:, 0],
                            triangles=simplices.copy(),
                            cmap='viridis', lw=0.1, **kwargs)
        cbar = plt.colorbar(c)
    else:
        if axis is None:
            axis = plt.figure().gca()
            
        domain = triangulation.discretization.limits.tolist()
        num_points = triangulation.discretization.num_points
            
        # Some magic reshaping to go to physical coordinates
        vals = values.reshape(num_points[0], num_points[1]).T[::-1]
        axis = plt.imshow(vals, origin='upper',
                          extent=domain[0] + domain[1],
                          aspect='auto', cmap='viridis', interpolation='bilinear', **kwargs)
        cbar = plt.colorbar(axis)
        axis = axis.axes
        
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    if zlabel is not None:
        cbar.set_label(zlabel)
        
    return axis


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def.

    Taken from
    http://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-
    tensorflow-graph-in-jupyter
    """
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = str.encode("<stripped %d bytes>" % size)
    return strip_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph.

    Taken from
    http://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-
    tensorflow-graph-in-jupyter
    """
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script src="//cdnjs.cloudflare.com/ajax/libs/polymer/0.3.3/platform.js"></script>
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)),
               id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:100%;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

