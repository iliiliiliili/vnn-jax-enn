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

"""Helpful functions relating to losses."""
from typing import Any, Dict, Callable, List, Optional, Sequence, Tuple, Union

import dataclasses
from enn import base
from enn.losses import single_index
import haiku as hk
import jax
import jax.numpy as jnp


# Maps Haiku params (module_name, name, value) -> include or not
PredicateFn = Callable[[str, str, Any], bool]


def l2_weights_with_predicate(
    params: hk.Params, predicate: Optional[PredicateFn] = None
) -> jnp.DeviceArray:
    """Sum of squares of parameter weights that passes predicate_fn."""
    if predicate is not None:
        params = hk.data_structures.filter(predicate, params)
    return sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))


def add_l2_weight_decay(
    loss_fn: base.LossFn,
    scale: Union[float, Callable[[hk.Params], hk.Params]],
    predicate: Optional[PredicateFn] = None,
) -> base.LossFn:
    """Adds scale * l2 weight decay to an existing loss function."""
    try:  # Scale is numeric.
        scale = jnp.sqrt(scale)
        scale_fn = lambda ps: jax.tree_map(lambda p: scale * p, ps)
    except TypeError:
        scale_fn = scale  # Assuming scale is a Callable.

    def new_loss(a, params: hk.Params, c, d) -> base.Array:
        loss, metrics = loss_fn(a, params, c, d)
        decay = l2_weights_with_predicate(scale_fn(params), predicate)
        total_loss = loss + decay
        metrics["decay"] = decay
        metrics["raw_loss"] = loss
        return total_loss, metrics

    return new_loss


def combine_single_index_losses_as_metric(
    train_loss: single_index.SingleIndexLossFn,
    extra_losses: Dict[str, single_index.SingleIndexLossFn],
) -> single_index.SingleIndexLossFn:
    """Combines train_loss for training with extra_losses in metrics."""

    def combined_loss(
        apply: base.ApplyFn, params: hk.Params, batch: base.Batch, index: base.Index
    ) -> Tuple[base.Array, base.LossMetrics]:
        loss, metrics = train_loss(apply, params, batch, index)
        for name, loss_fn in extra_losses.items():
            extra_loss, extra_metrics = loss_fn(apply, params, batch, index)
            metrics[f"{name}:loss"] = extra_loss
            for key, value in extra_metrics.items():
                metrics[f"{name}:{key}"] = value
        return loss, metrics

    return combined_loss


def combine_losses_as_metric(
    train_loss: base.LossFn, extra_losses: Dict[str, base.LossFn],
) -> base.LossFn:
    """Combines train_loss for training with extra_losses in metrics."""

    def combined_loss(
        enn: base.EpistemicNetwork,
        params: hk.Params,
        batch: base.Batch,
        key: base.RngKey,
    ) -> Tuple[base.Array, base.LossMetrics]:
        loss, metrics = train_loss(enn, params, batch, key)
        for name, loss_fn in extra_losses.items():
            extra_loss, extra_metrics = loss_fn(enn, params, batch, key)
            metrics[f"{name}:loss"] = extra_loss
            for key, value in extra_metrics.items():
                metrics[f"{name}:{key}"] = value
        return loss, metrics

    return combined_loss


@dataclasses.dataclass
class CombineLossConfig:
    loss_fn: base.LossFn
    name: str = "unnamed"  # Name for the loss function
    weight: float = 1.0  # Weight to scale the loss by


def combine_losses(
    losses: Sequence[Union[CombineLossConfig, base.LossFn]]
) -> base.LossFn:
    """Combines multiple losses into a single loss."""
    clean_losses: List[CombineLossConfig] = []
    for i, loss in enumerate(losses):
        if not isinstance(loss, CombineLossConfig):
            loss = CombineLossConfig(loss, name=f"loss_{i}")
        clean_losses.append(loss)

    def loss_fn(
        enn: base.EpistemicNetwork,
        params: hk.Params,
        batch: base.Batch,
        key: base.RngKey,
    ) -> Tuple[base.Array, base.LossMetrics]:
        combined_loss = 0.0
        combined_metrics = {}
        for loss_config in clean_losses:
            # Compute the loss types for use in conditional computation
            loss, metrics = loss_config.loss_fn(enn, params, batch, key)
            combined_metrics[f"{loss_config.name}:loss"] = loss
            for name, value in metrics.items():
                combined_metrics[f"{loss_config.name}:{name}"] = value
            combined_loss += loss_config.weight * loss
        return combined_loss, combined_metrics

    return loss_fn
