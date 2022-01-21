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
"""Example running an ENN on Thompson bandit task."""

from absl import app
from absl import flags
from jax.interpreters.xla import primitive_uses_outfeed
from enn.experiments.neurips_2021 import agent_factories
from enn.experiments.neurips_2021 import agents
from enn.experiments.neurips_2021 import load
from jax.config import config
import os

# Double-precision in JAX helps with numerical stability
config.update("jax_enable_x64", True)

# GP configuration
flags.DEFINE_multi_integer("input_dim", [1, 10, 100], "Input dimension.")
flags.DEFINE_multi_float(
    "data_ratio", [1.0, 10.0, 100.0], "Ratio of num_train to input_dim."
)
flags.DEFINE_multi_float(
    "noise_std", [0.01, 0.1, 1.0], "Additive noise standard deviation."
)
flags.DEFINE_integer("seed", 1, "Seed for testbed problem.")


# ENN agent
flags.DEFINE_integer("agent_id_start", 0, "Which agent id start")
flags.DEFINE_integer("agent_id_end", -1, "Which agent id end")
flags.DEFINE_enum(
    "agent",
    "all",
    [
        "all",
        "all_old",
        "ensemble",
        "dropout",
        "hypermodel",
        "bbb",
        "vnn",
        "vnn_selected",
        "vnn_lrelu",
        "vnn_lrelu_init",
        "layer_ensemble",
    ],
    "Which agent family.",
)

flags.DEFINE_string("experiment_group", "", "Name of the experiment group")

FLAGS = flags.FLAGS


def main(_):

    os.makedirs("results", exist_ok=True)

    for input_dim in FLAGS.input_dim:
        for data_ratio in FLAGS.data_ratio:
            for noise_std in FLAGS.noise_std:
                # Load the appropriate testbed problem
                problem = load.regression_load(
                    input_dim=input_dim,
                    data_ratio=data_ratio,
                    seed=FLAGS.seed,
                    noise_std=noise_std,
                )

                all_results = []

                sweep = agent_factories.load_agent_config_sweep(FLAGS.agent)
                sweep = (
                    sweep[FLAGS.agent_id_start :]
                    if FLAGS.agent_id_end == -1
                    else sweep[FLAGS.agent_id_start : FLAGS.agent_id_end]
                )

                for i, agent_config in enumerate(sweep):

                    agent_id = FLAGS.agent_id_start + i

                    print(
                        "input_dim",
                        input_dim,
                        "data_ratio",
                        data_ratio,
                        "noise_std",
                        noise_std,
                    )
                    print("agent_id", agent_id, "of", len(sweep))

                    # Form the appropriate agent for training
                    agent = agents.VanillaEnnAgent(agent_config.config_ctor())

                    # Train
                    enn_sampler = agent(
                        problem.train_data, problem.prior_knowledge
                    )

                    # Evaluate the quality of the ENN sampler after training
                    kl_quality = problem.evaluate_quality(enn_sampler)
                    print(
                        f"kl_estimate={kl_quality.kl_estimate}"
                        + "mean_error="
                        + str(kl_quality.extra["mean_error"])
                        + " "
                        + "std_error="
                        + str(kl_quality.extra["std_error"])
                    )
                    all_results.append(kl_quality)

                    with open(
                        "results/results_"
                        + FLAGS.experiment_group
                        + ("_" if len(FLAGS.experiment_group) > 0 else "")
                        + FLAGS.agent
                        + "_id"
                        + str(input_dim)
                        + "dr"
                        + str(data_ratio)
                        + "ns"
                        + str(noise_std)
                        + ".txt",
                        "a",
                    ) as f:

                        f.write(
                            str(agent_id)
                            + " "
                            + str(kl_quality.kl_estimate)
                            + " "
                            + "mean_error="
                            + str(kl_quality.extra["mean_error"])
                            + " "
                            + "std_error="
                            + str(kl_quality.extra["std_error"])
                            + " "
                            + " ".join(
                                [
                                    str(k) + "=" + str(v)
                                    for (
                                        k,
                                        v,
                                    ) in agent_config.settings.items()
                                ]
                            )
                            + "\n"
                        )

                print(all_results)


if __name__ == "__main__":
    app.run(main)
