# Copyright 2020-2021 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Implementation by Illia Oleksiienko
"""Implementing Variational Neural Network as an ENN in JAX."""

from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import chex
from haiku._src.batch_norm import BatchNorm
from enn import base
from enn import utils
from enn.networks import hypermodels
from enn.networks import indexers
import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp


Activation = Callable[[jnp.ndarray], jnp.ndarray]


def create_initializer(names):

    result = []

    for name in names:

        if name == None:
            result.append((None, None))
        elif name == "he_uniform":
            init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
            result.append((init, init))
        elif name == "he_normal":
            init = hk.initializers.VarianceScaling(
                2.0, "fan_in", "truncated_normal"
            )
            result.append((init, init))
        elif name == "glorot_normal":
            init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
            result.append((init, init))
        elif name == "glorot_uniform":
            init = hk.initializers.VarianceScaling(
                1.0, "fan_avg", "truncated_normal"
            )
            result.append((init, init))
        elif name == "1":
            init = hk.initializers.Constant(1.0)
            result.append((None, init))
        elif name == "2":
            init = hk.initializers.Constant(2.0)
            result.append((None, init))
        else:
            raise ValueError(str(name) + " is an unknown initializer name")

    return result


class VariationalBase(hk.Module):

    GLOBAL_STD: float = 0
    LOG_STDS = False

    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        means: Any,
        stds: Any,
        batch_norm_module: Any,
        batch_norm_size: int,
        activation: Optional[Union[Activation, List[Activation]]] = None,
        activation_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal["none"], Literal["replace"], Literal["multiply"]
        ] = "none",
    ) -> None:

        super().__init__()

        self.means = means
        self.stds = stds
        self.batch_norm_module = batch_norm_module
        self.batch_norm_mode = batch_norm_mode
        self.batch_norm_size = batch_norm_size
        self.activation = activation
        self.activation_mode = activation_mode
        self.use_batch_norm = use_batch_norm
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.global_std_mode = global_std_mode

    def __call__(self, x, index, global_std=2):

        end_activation = None
        end_batch_norm = None

        means = self.means()
        stds = self.stds()

        if self.use_batch_norm:

            batch_norm_targets = self.batch_norm_mode.split("+")

            for i, target in enumerate(batch_norm_targets):

                if target == "mean":
                    means = hk.Sequential(
                        [
                            means,
                            lambda x: self.batch_norm_module(
                                True,
                                True,
                                eps=self.batch_norm_eps,
                                decay_rate=self.batch_norm_momentum,
                            )(x, is_training=True),
                        ]
                    )
                elif target == "std":
                    if stds is not None:
                        stds = hk.Sequential(
                            [
                                stds,
                                lambda x: self.batch_norm_module(
                                    True,
                                    True,
                                    eps=self.batch_norm_eps,
                                    decay_rate=self.batch_norm_momentum,
                                    is_training=True,
                                )(x, is_training=True),
                            ]
                        )
                elif target == "end":
                    self.end_batch_norm = (
                        lambda x: self.batch_norm_module(
                            True,
                            True,
                            eps=self.batch_norm_eps,
                            decay_rate=self.batch_norm_momentum,
                            is_training=True,
                        )(x, is_training=True),
                    )
                else:
                    raise ValueError("Unknown batch norm target: " + target)

        if self.activation is not None:

            activation_targets = self.activation_mode.split("+")

            for i, target in enumerate(activation_targets):

                if len(activation_targets) == 1:
                    current_activation: Activation = self.activation  # type: ignore
                else:
                    current_activation: Activation = self.activation[  # type: ignore
                        i
                    ]

                if target == "mean":
                    means = hk.Sequential([means, current_activation])
                elif target == "std":
                    if stds is not None:
                        stds = hk.Sequential([stds, current_activation,])
                elif target == "end":
                    end_activation = current_activation
                elif target == "none":
                    pass
                else:
                    raise ValueError("Unknown activation target: " + target)

        mean_values = means(x)

        if stds:
            std_values = stds(x)
        else:
            std_values = 0

        if self.global_std_mode == "replace":
            std_values = global_std
        elif self.global_std_mode == "multiply":
            std_values = global_std * std_values

        result = mean_values + std_values * index

        if end_batch_norm is not None:
            result = end_batch_norm(result)

        if end_activation is not None:
            result = end_activation(result)

        return result


class VariationalLinear(VariationalBase):
    def __init__(
        self,
        out_features: int,
        activation: Optional[Union[Activation, List[Activation]]] = None,
        activation_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal["none"], Literal["replace"], Literal["multiply"]
        ] = "none",
        bias=True,
        initializer: Tuple[
            Union[
                Literal["he_uniform"],
                Literal["he_normal"],
                Literal["glorot_normal"],
                Literal["glorot_uniform"],
                None,
            ],
            Union[
                Literal["1"],
                Literal["2"],
                Literal["he_uniform"],
                Literal["he_normal"],
                Literal["glorot_normal"],
                Literal["glorot_uniform"],
                None,
            ],
        ] = (None, None),
        **kwargs,
    ) -> None:

        super().__init__()

        if use_batch_norm:
            bias = False

        initializers_mean, initializers_std = create_initializer(initializer)

        means = lambda: hk.Linear(
            out_features,
            with_bias=bias,
            w_init=initializers_mean[0],
            b_init=initializers_mean[1],
            **kwargs,
        )

        if global_std_mode == "replace":
            stds = lambda: None
        else:
            stds = lambda: hk.Linear(
                out_features,
                with_bias=bias,
                w_init=initializers_std[0],
                b_init=initializers_std[1],
                **kwargs,
            )

        super().build(
            means,
            stds,
            hk.BatchNorm,
            out_features,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=use_batch_norm,
            batch_norm_mode=batch_norm_mode,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
            global_std_mode=global_std_mode,
        )


class VariationalLinear_S(hk.Module):
    def __init__(
        self,
        out_features: int,
        activation: Optional[Union[Activation, List[Activation]]] = None,
        activation_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal["none"], Literal["replace"], Literal["multiply"]
        ] = "none",
        bias=True,
        **kwargs,
    ) -> None:

        self.kwargs = kwargs
        self.out_features = out_features
        self.bias = bias

        super().__init__()

    def __call__(self, x, index):
        means = hk.Linear(
            self.out_features, with_bias=self.bias, **self.kwargs
        )
        stds = hk.Linear(self.out_features, with_bias=self.bias, **self.kwargs)

        result = means(x) + stds(x) * index

        return result


class MLPVariationalENN(base.EpistemicNetwork):
    """MLP VNN as an ENN."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        activation: Optional[Union[Activation, List[Activation]]] = None,
        activation_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        global_std_mode: Union[
            Literal["none"], Literal["replace"], Literal["multiply"]
        ] = "none",
        seed: int = 0,
        initializer: Tuple[
            Union[
                Literal["he_uniform"],
                Literal["he_normal"],
                Literal["glorot_normal"],
                Literal["glorot_uniform"],
                None,
            ],
            Union[
                Literal["1"],
                Literal["2"],
                Literal["he_uniform"],
                Literal["he_normal"],
                Literal["glorot_normal"],
                Literal["glorot_uniform"],
                None,
            ],
        ] = (None, None),
    ):
        def enn_fn(inputs: base.Array, full_index: base.Index) -> base.Output:

            indices = []
            i = 0
            for output_size in output_sizes:
                indices.append(full_index[i : i + output_size])
                i += output_size

            x = hk.Flatten()(inputs)

            for output_size, index in zip(output_sizes, indices):
                x = VariationalLinear(
                    output_size,
                    activation,
                    activation_mode,
                    use_batch_norm,
                    batch_norm_mode,
                    global_std_mode=global_std_mode,
                    initializer=initializer,
                )(x, index)

            return x

        transformed = hk.without_apply_rng(hk.transform(enn_fn))

        index_dim = sum(output_sizes)

        indexer = indexers.ScaledGaussianIndexer(
            index_dim, jnp.sqrt(index_dim)
        )

        def apply(
            params: hk.Params, x: base.Array, z: base.Index
        ) -> base.Output:
            net_out = transformed.apply(params, x, z)
            return net_out

        super().__init__(apply, transformed.init, indexer)


class MLPVariationalENN_O(base.EpistemicNetwork):
    """MLP VNN as an ENN."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        activation: Optional[Union[Activation, List[Activation]]] = None,
        activation_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        global_std_mode: Union[
            Literal["none"], Literal["replace"], Literal["multiply"]
        ] = "none",
        seed: int = 0,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
    ):
        def enn_fn(inputs: base.Array, full_index: base.Index) -> base.Output:

            indices = []
            i = 0
            for output_size in output_sizes:
                indices.append(full_index[i : i + output_size])
                i += output_size

            x = hk.Flatten()(inputs)

            for output_size, index in zip(output_sizes, indices):
                x = VariationalLinear(
                    output_size,
                    activation,
                    activation_mode,
                    use_batch_norm,
                    batch_norm_mode,
                    global_std_mode=global_std_mode,
                    w_init=w_init,
                    b_init=b_init,
                )(x, index)

            return x

        transformed = hk.without_apply_rng(hk.transform(enn_fn))

        index_dim = sum(output_sizes)

        indexer = indexers.ScaledGaussianIndexer(
            index_dim, jnp.sqrt(index_dim)
        )

        def apply(
            params: hk.Params, x: base.Array, z: base.Index
        ) -> base.Output:
            net_out = transformed.apply(params, x, z)
            return net_out

        super().__init__(apply, transformed.init, indexer)
