# python3
# pylint: disable=g-bad-file-header
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Efficient ensemble implementations for JAX/Haiku via einsum."""
from typing import Optional, Sequence

import chex
from enn import base
from enn.networks import indexers
import haiku as hk
import jax
import jax.numpy as jnp


def make_einsum_ensemble_mlp_enn(
    output_sizes: Sequence[int], num_ensemble: int, nonzero_bias: bool = True,
) -> base.EpistemicNetwork:
    """Factory method to create fast einsum MLP ensemble ENN.

  This is a specialized implementation for ReLU MLP without a prior network.

  Args:
    output_sizes: Sequence of integer sizes for the MLPs.
    num_ensemble: Integer number of elements in the ensemble.
    nonzero_bias: Whether to make the initial layer bias nonzero.
  Returns:
    EpistemicNetwork as an ensemble of MLP.
  """

    def ensemble_forward(x: base.Array) -> base.OutputWithPrior:
        """Forwards the entire ensemble at given input x."""
        model = EnsembleMLP(output_sizes, num_ensemble, nonzero_bias)
        return model(x)

    transformed = hk.without_apply_rng(hk.transform(ensemble_forward))

    # Apply function selects the appropriate index of the ensemble output.
    def apply(params: hk.Params, x: base.Array, z: base.Index) -> base.OutputWithPrior:
        net_out = transformed.apply(params, x)
        one_hot_index = jax.nn.one_hot(z, num_ensemble)
        return jnp.dot(net_out, one_hot_index)

    def init(key: base.RngKey, x: base.Array, z: base.Index) -> hk.Params:
        del z
        return transformed.init(key, x)

    indexer = indexers.EnsembleIndexer(num_ensemble)

    return base.EpistemicNetwork(apply, init, indexer)


def make_ensemble_mlp_with_prior_enn(
    output_sizes: Sequence[int],
    dummy_input: chex.Array,
    num_ensemble: int,
    prior_scale: float = 1.0,
    nonzero_bias: bool = True,
    seed: int = 999,
) -> base.EpistemicNetwork:
    """Factory method to create fast einsum MLP ensemble with matched prior.

  Args:
    output_sizes: Sequence of integer sizes for the MLPs.
    dummy_input: Example x input for prior initialization.
    num_ensemble: Integer number of elements in the ensemble.
    prior_scale: Float rescaling of the prior MLP.
    nonzero_bias: Whether to make the initial layer bias nonzero.
    seed: integer seed for prior init.

  Returns:
    EpistemicNetwork ENN of the ensemble of MLP with matches prior.
  """

    enn = make_einsum_ensemble_mlp_enn(output_sizes, num_ensemble, nonzero_bias)
    init_key, _ = jax.random.split(jax.random.PRNGKey(seed))
    prior_params = enn.init(init_key, dummy_input, jnp.array([]))

    # Apply function selects the appropriate index of the ensemble output.
    def apply_with_prior(
        params: hk.Params, x: base.Array, z: base.Index
    ) -> base.OutputWithPrior:
        ensemble_train = enn.apply(params, x, z)
        ensemble_prior = enn.apply(prior_params, x, z) * prior_scale
        return base.OutputWithPrior(train=ensemble_train, prior=ensemble_prior)

    return base.EpistemicNetwork(apply_with_prior, enn.init, enn.indexer)


################################################################################
# Einsum implementation of MLP


class EnsembleBranch(hk.Module):
    """Branches a single linear layer to num_ensemble, output_size."""

    def __init__(
        self,
        num_ensemble: int,
        output_size: int,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: str = "ensemble_branch",
    ):
        super().__init__(name=name)
        self.num_ensemble = num_ensemble
        self.output_size = output_size
        if b_init:
            self._b_init = b_init
        else:
            self._b_init = jnp.zeros

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:  # [B, H] -> [B, D, K]
        assert inputs.ndim == 2
        unused_batch, input_size = inputs.shape
        w_init = hk.initializers.TruncatedNormal(stddev=(1.0 / jnp.sqrt(input_size)))
        w = hk.get_parameter(
            "w", [input_size, self.output_size, self.num_ensemble], init=w_init
        )
        b = hk.get_parameter(
            "b", [self.output_size, self.num_ensemble], init=self._b_init
        )
        return jnp.einsum("bi,ijk->bjk", inputs, w) + b


class EnsembleLinear(hk.Module):
    """Keeps num_ensemble linear layers in parallel without interactions."""

    def __init__(self, output_size: int, name: str = "linear"):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:  # [B, H, K] -> [B. D, K]
        assert inputs.ndim == 3
        unused_batch, input_size, self.num_ensemble = inputs.shape
        w_init = hk.initializers.TruncatedNormal(stddev=(1.0 / jnp.sqrt(input_size)))
        w = hk.get_parameter(
            "w", [input_size, self.output_size, self.num_ensemble], init=w_init
        )
        b = hk.get_parameter("b", [self.output_size, self.num_ensemble], init=jnp.zeros)
        return jnp.einsum("bik,ijk->bjk", inputs, w) + b


class EnsembleMLP(hk.Module):
    """Parallel num_ensemble MLPs all with same output_sizes.

  In the first layer, the input is 'branched' to num_ensemble linear layers.
  Then, in subsequent layers it is purely parallel EnsembleLinear.
  """

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensemble: int,
        nonzero_bias: bool = True,
        name: str = "ensemble_mlp",
    ):
        super().__init__(name=name)
        self.num_ensemble = num_ensemble
        layers = []
        for index, output_size in enumerate(output_sizes):
            if index == 0:
                if nonzero_bias:
                    b_init = hk.initializers.TruncatedNormal(stddev=1)
                else:
                    b_init = jnp.zeros
                layers.append(EnsembleBranch(num_ensemble, output_size, b_init))
            else:
                layers.append(EnsembleLinear(output_size))
        self.layers = tuple(layers)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:  # [B, H] -> [B, D, K]
        num_layers = len(self.layers)
        out = hk.Flatten()(inputs)
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < num_layers - 1:
                out = jax.nn.relu(out)
        return out
