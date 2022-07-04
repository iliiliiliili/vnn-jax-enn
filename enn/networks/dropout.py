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
"""Implementing Dropout as an ENN in JAX."""
from typing import Optional, Sequence

from enn import base
from enn.networks import indexers
import haiku as hk


class MLPDropoutENN(base.EpistemicNetwork):
    """MLP with dropout as an ENN."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        dropout_rate: float = 0.05,
        dropout_input: bool = True,
        seed: int = 0,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
    ):
        """MLP with dropout as an ENN."""

        def enn_fn(inputs: base.Array, index: base.Index) -> base.Output:
            x = hk.Flatten()(inputs)

            # Note that we consider a dropout layer after the input to be consistent
            # with the paper "Dropout as a Bayesian Approximation: Representing Model
            # Uncertainty in Deep Learning" (2015),
            # https://github.com/yaringal/DropoutUncertaintyExps/blob/master/net/net.py
            if dropout_input:
                # We use index as rng for the dropout layer.
                x = hk.dropout(index, dropout_rate, x)
            net = hk.nets.MLP(output_sizes, w_init=w_init, b_init=b_init)

            # We use index as rng for the dropout layer.
            return net(x, dropout_rate=dropout_rate, rng=index)

        # Note that although our enn_fn is stochastic because of the dropout layer,
        # since we pass an index as rng directly, we can strill wrap transformed
        # function with hk.without_apply_rng.
        transformed = hk.without_apply_rng(hk.transform(enn_fn))

        # We use a simple indexer which is basically an identity map.
        indexer = indexers.PrngIndexer()

        # Apply function for enn_fn requires a rng key. We use the index z in f(x,z)
        # as the rng key. This apply method hides this step from outside.
        def apply(params: hk.Params, x: base.Array, z: base.Index) -> base.Output:
            net_out = transformed.apply(params, x, z)
            return net_out

        super().__init__(apply, transformed.init, indexer)
