# https://arxiv.org/pdf/1806.06988.pdf

from jax._src.api import value_and_grad
from jax._src.numpy import lax_numpy as np
from jax._src.nn import functions as nn
from jax._src.random import PRNGKey, uniform
from jax._src.tree_util import tree_map, tree_reduce
from jax.experimental.optimizers import adam

def jax_bin(inputs, params):
    D = params.shape[0]
    W = np.reshape(np.linspace(1.0, D + 1.0, D + 1), [1, -1])
    b = np.cumsum(np.concatenate([np.zeros([1]), -np.sort(params)], 0),0)
    return nn.softmax(np.matmul(inputs, W) + b)

def loss_fn(cut_points_list, leaf_score, inputs, targets):
    leaf  tree_reduce(np.kron, tree_map(lambda z: jax_bin(inputs[:, z[0]:z[0] + 1], z[1]), enumerate(cut_points_list)))
    preds = np.matmul(leaf, leaf_score)
    return -np.sum(nn.log_softmax(preds) * targets, axis = -1)

# set seed
key = PRNGKey(0)

# load data
x = np.vstack([np.array(i[0]) for i in demo])
y = np.vstack([np.array(i[1]) for i in demo])
x_dim = x.shape[1]
y_dim = y.shape[1]

# set params
cut_points_list = [uniform(key, [1]) for i in np.ones([x_dim])]
leaf_score = uniform(key, [2 ** x_dim, y_dim])
step_size = 1e-3
opt_init, opt_update, get_params = adam(step_size)
opt_state = opt_init(cut_points_list + [leaf_score])
num_epochs = 10

for i in range(num_epochs):
    loss, grads = value_and_grad(loss_fn)(cut_points_list, leaf_score, x, y)
    opt_state = opt_update(i, grads, opt_state)
    cut_points_list, leaf_score = get_params(opt_state)
