from functools import partial
from typing import Optional, List, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import linen as nn
from jax import jit, vmap, value_and_grad, random
from jax.scipy.stats import norm, multivariate_normal


class NNTraining:
    def __init__(self, x_dim: int, y_dim: int, prior_h: float, features: List[int], nll_scale: float = 0.1):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.prior_h = prior_h
        self.features = features
        self.prior_kernel = self.prepare_prior_kernel(h=self.prior_h ** 2)
        self.model = MLP(features=features, output_dim=self.y_dim)
        self.nll_scale = nll_scale
        self.tx = optax.adam(learning_rate=0.01)
        self.key = random.PRNGKey(0)

    @partial(jit, static_argnums=0)
    def update_step(self, xs, ys, opt_state, params, stats):
        (loss, updated_state), grads = value_and_grad(self.loss, has_aux=True, argnums=0)(
            params, stats, xs, ys)
        updates, opt_state = self.tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state, params, updated_state, loss

    def _train(self, params, batch_stats, xs, ys, num_steps):
        opt_state = self.tx.init(params)
        for i in range(num_steps):
            opt_state, params, batch_stats, loss = self.update_step(xs, ys, opt_state, params, batch_stats)
            if i % 100 == 0:
                print(loss)
        return params, batch_stats

    def fit_model(self, xs, ys, num_steps: int):
        self.key, subkey = random.split(self.key)
        self.params, self.batch_stats = self.init_params(subkey)
        self.params, self.batch_stats = self._train(self.params, self.batch_stats, xs, ys, num_steps)

    @partial(jit, static_argnums=0)
    def _train_one(self, params, batch_stats, x):
        assert x.shape == (self.x_dim,)
        net_out, updates = self.model.apply({'params': params, 'batch_stats': batch_stats}, x,
                                            mutable=['batch_stats'], train=True)
        batch_stats = updates['batch_stats']
        return net_out, batch_stats

    @partial(jit, static_argnums=0)
    def _eval_one(self, params, batch_stats, x):
        assert x.shape == (self.x_dim,)
        return self.model.apply({'params': params, 'batch_stats': batch_stats}, x)

    @staticmethod
    def prepare_prior_kernel(h=1.0 ** 2):
        def k(x, y):
            return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * h))

        v_k = vmap(k, in_axes=(0, None), out_axes=0)
        m_k = vmap(v_k, in_axes=(None, 0), out_axes=1)

        def kernel(fs):
            kernel_matrix = m_k(fs, fs)
            return kernel_matrix

        return kernel

    def _neg_log_posterior(self, pred_raw: jnp.ndarray, x_stacked: jnp.ndarray, y_batch: jnp.ndarray,
                           scale: jnp.ndarray, num_train_points):
        assert pred_raw.shape == y_batch.shape == scale.shape
        nll = self._nll(pred_raw, y_batch, scale)
        neg_log_prior = - self._gp_prior_log_prob(x_stacked, pred_raw) / num_train_points
        neg_log_post = nll + neg_log_prior
        return neg_log_post

    @staticmethod
    def _nll(pred_raw: jnp.ndarray, y_batch: jnp.ndarray, scale: jnp.ndarray):
        log_prob = norm.logpdf(y_batch, loc=pred_raw, scale=scale)
        return - jnp.mean(log_prob)

    def _gp_prior_log_prob(self, x: jnp.array, y: jnp.array, eps: float = 1e-4) -> jnp.ndarray:
        # Multiple dimension outputs are handled independently per dimension
        k = self.prior_kernel(x) + eps * jnp.eye(x.shape[0])

        def evaluate_fs(fs):
            assert fs.shape == (x.shape[0],) and fs.ndim == 1
            return multivariate_normal.logpdf(fs, mean=jnp.zeros(x.shape[0]), cov=k)

        evaluate_fs_multiple_dims = vmap(evaluate_fs, in_axes=1, out_axes=0)
        return jnp.mean(evaluate_fs_multiple_dims(y))

    def loss(self, params, stats, xs, ys):
        assert xs.shape[1] == self.x_dim and ys.shape[1] == self.y_dim
        assert xs.shape[0] == ys.shape[0]
        f_raw, new_stats = vmap(self._train_one, in_axes=(None, None, 0),
                                out_axes=(0, None), axis_name='batch')(params, stats, xs)

        scale = self.nll_scale * jnp.ones(shape=f_raw.shape)

        assert f_raw.shape == ys.shape == scale.shape
        num_train_points = xs.shape[0]
        loss = self._neg_log_posterior(f_raw, xs, ys, scale, num_train_points)
        return loss, new_stats

    def init_params(self, key):
        variables = self.model.init(key, jnp.ones(shape=(self.x_dim,)))
        # Split batch_stats and params (which are updated by optimizer).
        params = variables['params']
        batch_stats = variables['batch_stats']
        del variables  # Delete variables to avoid wasting resources
        return params, batch_stats


class MLP(nn.Module):
    features: Sequence[int]
    output_dim: Optional[int]

    @nn.compact
    def __call__(self, x, train: bool = False):
        for feat in self.features:
            x = nn.Dense(features=feat)(x)
            x = nn.BatchNorm(use_running_average=not train, axis_name="batch")(x)
            x = nn.swish(x)
        if self.output_dim is not None:
            x = nn.Dense(features=self.output_dim)(x)
        return x


if __name__ == '__main__':
    key = random.PRNGKey(0)
    xs = jnp.linspace(0, 10, 20).reshape(-1, 1)
    ys = jnp.sin(xs)

    model = NNTraining(x_dim=1, y_dim=1, prior_h=0.1, features=[100, 100, 20], nll_scale=0.1)
    model.fit_model(xs=xs, ys=ys, num_steps=1000)

    test_xs = jnp.linspace(0, 10, 100).reshape(-1, 1)
    preds = vmap(model._eval_one, in_axes=(None, None, 0))(model.params, model.batch_stats, test_xs)

    plt.scatter(xs, ys, label='Data', color='red')
    plt.plot(test_xs, preds, label='NN prediction', color='blue')
    plt.legend()
    plt.show()
