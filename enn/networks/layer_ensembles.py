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

"""Implementing some types of ENN ensembles in JAX."""
from typing import Callable, Optional, Sequence

from enn import base
from enn import utils
from enn.networks import indexers
from enn.networks import priors
import haiku as hk
import jax


class EnsembleLayer(hk.Module):
    def __init__(self, ensemble: Sequence[hk.Module], priors: Sequence[hk.Module], prior_scale: int = 1.0):
        super().__init__(name="layer_ensemble")
        self.ensemble = list(ensemble)
        self.priors = list(priors)
        self.prior_scale = prior_scale

    def __call__(self, inputs: base.Array, index: base.Index) -> base.Array:
        """Index must be a single integer indicating which layer to forward."""
        # during init ensure all module parameters are created.
        _ = [model(inputs) for model in self.ensemble]  # pytype:disable=not-callable
        _ = [model(inputs) for model in self.priors]  # pytype:disable=not-callable
        model_output = hk.switch(index, self.ensemble, inputs)
        if len(self.priors) > 0:
            prior_output = hk.switch(index, self.priors, inputs)
            output = base.OutputWithPrior(model_output, self.prior_scale * prior_output).preds
        else:
            output = model_output

        return output


class LayerEnsembleNetwork(base.EpistemicNetwork):
    """A layer-ensemble MLP (with flatten) and without any prior."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensembles: Sequence[int],
        module: hk.Module = hk.Linear
    ):

        def enn_fn(inputs: base.Array, full_index: base.Index) -> base.Output:
            x = hk.Flatten()(inputs)

            layers = [
                EnsembleLayer([module(output_size) for _ in range(num_ensemble)], [])
                for num_ensemble, output_size in zip(num_ensembles, output_sizes)
            ]

            for layer, index in zip(layers, full_index):
                x = layer(x, index)

            return x

        transformed = hk.without_apply_rng(hk.transform(enn_fn))
        indexer = indexers.LayerEnsembleIndexer(self.num_ensembles),

        def apply(
            params: hk.Params, x: base.Array, z: base.Index
        ) -> base.Output:
            net_out = transformed.apply(params, x, z)
            return net_out
        
        super().__init__(apply, transformed.init, indexer)



class LayerEnsembleNetworkWithPriors(base.EpistemicNetwork):
    """A layer-ensemble MLP (with flatten) and without any prior."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensembles: Sequence[int],
        prior_scale: float = 1.0,
        module: hk.Module = hk.Linear,
        seed : int = 0,
    ):

        def enn_fn(inputs: base.Array, full_index: base.Index) -> base.Output:
            x = hk.Flatten()(inputs)

            layers = [
                EnsembleLayer(
                    [module(output_size) for _ in range(num_ensemble)],
                    [module(output_size) for _ in range(num_ensemble)],
                    prior_scale=prior_scale,
                )
                for num_ensemble, output_size in zip(num_ensembles, output_sizes)
            ]

            for layer, index in zip(layers, full_index):
                x = layer(x, index)

            return x

        transformed = hk.without_apply_rng(hk.transform(enn_fn))
        indexer = indexers.LayerEnsembleIndexer(num_ensembles)

        def apply(
            params: hk.Params, x: base.Array, z: base.Index
        ) -> base.Output:
            net_out = transformed.apply(params, x, z)
            return net_out
        
        super().__init__(apply, transformed.init, indexer)

def init_module(
    net_fn,
    dummy_input: base.Array,
    rng: int = 0,
) -> Sequence[Callable[[base.Array], base.Array]]:
    transformed = hk.without_apply_rng(hk.transform(net_fn))
    params = transformed.init(next(rng), dummy_input)
    return lambda x, params=params: transformed.apply(params, x)