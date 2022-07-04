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

"""Implementing some ensembles with categorical outputs.

Next step is to integrate more with the rest of the ENN code.
"""
from typing import Sequence

from enn import base
from enn import utils
from enn.networks import ensembles
from enn.networks import indexers
from enn.networks import priors
import haiku as hk
import jax
import jax.numpy as jnp


class CatOutputWithPrior(base.OutputWithPrior):
    """Categorical outputs with a real-valued prior."""

    @property
    def preds(self) -> base.Array:
        train = jnp.sum(jax.nn.softmax(self.train) * self.extra["atoms"], axis=-1)
        return train + jax.lax.stop_gradient(self.prior)


class CategoricalRegressionMLP(hk.Module):
    """Categorical MLP designed for regression ala MuZero value."""

    def __init__(self, output_sizes: Sequence[int], atoms: base.Array):
        """Categorical MLP designed for regression ala MuZero value."""
        super().__init__(name="categorical_regression_mlp")
        self.dim_out = output_sizes[-1]
        self.atoms = jnp.array(atoms)
        self.output_sizes = list(output_sizes[:-1]) + [self.dim_out * len(atoms)]

    def __call__(self, inputs: base.Array) -> base.Array:
        """Apply MLP and wrap outputs appropriately."""
        out = hk.Flatten()(inputs)
        out = hk.nets.MLP(self.output_sizes)(out)
        return CatOutputWithPrior(
            train=jnp.reshape(out, [-1, self.dim_out, len(self.atoms)]),
            extra={"atoms": self.atoms},
        )


class CatMLPEnsemble(base.EpistemicNetwork):
    """An ensemble of categorical MLP for regression."""

    def __init__(
        self, output_sizes: Sequence[int], atoms: base.Array, num_ensemble: int
    ):
        """An ensemble of categorical MLP for regression."""
        net_ctor = lambda: CategoricalRegressionMLP(output_sizes, atoms)
        enn = utils.epistemic_network_from_module(
            enn_ctor=lambda: ensembles.Ensemble(  # pylint: disable=g-long-lambda
                [net_ctor() for _ in range(num_ensemble)]
            ),
            indexer=indexers.EnsembleIndexer(num_ensemble),
        )
        super().__init__(enn.apply, enn.init, enn.indexer)


class CatMLPEnsembleGpPrior(base.EpistemicNetwork):
    """An ensemble of categorical MLP with a real-valued GP prior."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        atoms: base.Array,
        input_dim: int,
        num_ensemble: int,
        num_feat: int,
        gamma: priors.GpGamma = 1.0,
        prior_scale: float = 1,
        seed: int = 0,
    ):
        """An ensemble of categorical MLP with a real-valued GP prior."""
        gp_priors = ensembles.make_random_gp_ensemble_prior_fns(
            input_dim, 1, num_feat, gamma, num_ensemble, seed
        )
        enn = priors.EnnWithAdditivePrior(
            enn=CatMLPEnsemble(output_sizes, atoms, num_ensemble),
            prior_fn=ensembles.wrap_sequence_as_prior(gp_priors),
            prior_scale=prior_scale,
        )
        super().__init__(enn.apply, enn.init, enn.indexer)


class CatMLPEnsembleMlpPrior(base.EpistemicNetwork):
    """An ensemble of categorical MLP with real-valued MLP prior."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        atoms: base.Array,
        dummy_input: base.Array,
        num_ensemble: int,
        prior_scale: float = 1,
        seed: int = 0,
    ):
        """An ensemble of categorical MLP with real-valued MLP prior."""
        mlp_priors = ensembles.make_mlp_ensemble_prior_fns(
            output_sizes, dummy_input, num_ensemble, seed
        )
        enn = priors.EnnWithAdditivePrior(
            enn=CatMLPEnsemble(output_sizes, atoms, num_ensemble),
            prior_fn=ensembles.wrap_sequence_as_prior(mlp_priors),
            prior_scale=prior_scale,
        )
        super().__init__(enn.apply, enn.init, enn.indexer)
