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

"""An standard experiment operating by SGD."""

import functools
from typing import Callable, Dict, NamedTuple, Optional, Tuple

from acme.utils import loggers
from enn import base
from enn.supervised import base as supervised_base
import haiku as hk
import jax
import optax


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState


class Experiment(supervised_base.BaseExperiment):
    """Class to handle supervised training.

  Optional eval_datasets which is a collection of datasets to *evaluate*
  the loss on every eval_log_freq steps.
  """

    def __init__(
        self,
        enn: base.EpistemicNetwork,
        loss_fn: base.LossFn,
        optimizer: optax.GradientTransformation,
        dataset: base.BatchIterator,
        seed: int = 0,
        logger: Optional[loggers.Logger] = None,
        train_log_freq: int = 1,
        eval_datasets: Optional[Dict[str, base.BatchIterator]] = None,
        eval_log_freq: int = 1,
    ):
        self.enn = enn
        self.dataset = dataset
        self.rng = hk.PRNGSequence(seed)

        # Internalize the loss_fn
        self._loss = jax.jit(functools.partial(loss_fn, self.enn))

        # Internalize the eval datasets
        self._eval_datasets = eval_datasets
        self._eval_log_freq = eval_log_freq

        # Forward network at random index
        def forward(
            params: hk.Params, inputs: base.Array, key: base.RngKey
        ) -> base.Array:
            index = self.enn.indexer(key)
            return self.enn.apply(params, inputs, index)

        self._forward = jax.jit(forward)

        # Define the SGD step on the loss
        def sgd_step(
            state: TrainingState, batch: base.Batch, key: base.RngKey,
        ) -> Tuple[TrainingState, base.LossMetrics]:
            # Calculate the loss, metrics and gradients
            (loss, metrics), grads = jax.value_and_grad(self._loss, has_aux=True)(
                state.params, batch, key
            )
            metrics.update({"loss": loss})
            updates, new_opt_state = optimizer.update(grads, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)
            new_state = TrainingState(params=new_params, opt_state=new_opt_state,)
            return new_state, metrics

        self._sgd_step = jax.jit(sgd_step)

        # Initialize networks
        batch = next(self.dataset)
        index = self.enn.indexer(next(self.rng))
        params = self.enn.init(next(self.rng), batch.x, index)
        opt_state = optimizer.init(params)
        self.state = TrainingState(params, opt_state)
        self.step = 0
        self.logger = logger or loggers.make_default_logger("experiment", time_delta=0)
        self._train_log_freq = train_log_freq

    def train(self, num_batches: int, evaluate: Callable = None, log_file_name: str = None):
        """Train the ENN for num_batches."""
        for _ in range(num_batches):
            self.step += 1
            self.state, loss_metrics = self._sgd_step(
                self.state, next(self.dataset), next(self.rng)
            )

            # Periodically log this performance as dataset=train.
            if self.step % self._train_log_freq == 0:
                loss_metrics.update(
                    {"dataset": "train", "step": self.step, "sgd": True}
                )
                self.logger.write(loss_metrics)

                if log_file_name is not None:
                    with open(log_file_name, "a") as f:
                        f.write("step=" + str(loss_metrics["step"]) + " loss=" + str(float(loss_metrics["loss"])) + "\n")

            if evaluate is not None and self.step % self._eval_log_freq == 0:
                if evaluate(): # KL started decreasing
                    return

            # Periodically evaluate the other datasets.
            if self._eval_datasets and self.step % self._eval_log_freq == 0:
                for name, dataset in self._eval_datasets.items():
                    loss, metrics = self._loss(
                        self.state.params, next(dataset), next(self.rng)
                    )
                    metrics.update(
                        {
                            "dataset": name,
                            "step": self.step,
                            "sgd": False,
                            "loss": loss,
                        }
                    )
                    self.logger.write(metrics)

    def predict(self, inputs: base.Array, seed: int) -> base.Array:
        """Evaluate the trained model at given inputs."""
        return self._forward(self.state.params, inputs, jax.random.PRNGKey(seed))

    def loss(self, batch: base.Batch, seed: int) -> base.Array:
        """Evaluate the loss for one batch of data."""
        return self._loss(self.state.params, batch, jax.random.PRNGKey(seed))
