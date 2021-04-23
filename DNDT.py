from functools import reduce
from jax._src.api import value_and_grad
from jax._src.tree_util import tree_map, tree_reduce
from jax._src.numpy import lax_numpy as np
from jax._src.nn import functions as nn
from jax._src.random import PRNGKey, uniform
from jax.experimental.optimizers import adam

x = np.vstack([np.array(i[0]) for i in demo])
y = np.vstack([np.array(i[1]) for i in demo])

def jax_kron_prod(a, b):
    res = np.einsum('ij,ik->ijk', [a, b])
    return np.reshape(res, [-1, int(np.prod(res.shape[1:]))]) 

def jax_bin(x, cut_points):
    D = cut_points.shape[0]
    W = np.reshape(np.linspace(1.0, D + 1.0, D + 1), [1, -1])
    cut_points = np.sort(cut_points)
    b = np.cumsum(np.concatenate([np.zeros([1]), -cut_points], 0),0)
    h = np.matmul(x, W) + b
    return nn.softmax(h)

def loss_fn(params, inputs, targets):
    cut_points_list, leaf_score = params
    leaf = tree_reduce(jax_kron_prod, tree_map(lambda z: jax_bin(x[:, z[0]:z[0] + 1], z[1]), enumerate(cut_points_list)))
    preds = np.matmul(leaf, leaf_score)
    return -np.sum(nn.log_softmax(preds) * targets, axis = -1)

key = PRNGKey(0)

num_cut = np.ones([4])
num_leaf = int(np.prod(np.array(num_cut) + 1))
num_class = 3
cut_points_list = [uniform(PRNGKey(0), [1]) for i in num_cut]
leaf_score = uniform(PRNGKey(0), [num_leaf, num_class])
params = cut_points_list, leaf_score
step_size = 1e-3
opt_init, opt_update, get_params = adam(step_size)
opt_state = opt_init(params)
num_epochs = 10

for i in range(num_epochs):
    loss, grads = value_and_grad(loss_fn)(params, x, y)
    opt_state = opt_update(i, grads, opt_state)
    params = get_params(opt_state)
