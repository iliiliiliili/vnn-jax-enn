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

# Double-precision in JAX helps with numerical stability
config.update("jax_enable_x64", True)

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

FLAGS = flags.FLAGS


def main(_):

    sweep = agent_factories.load_agent_config_sweep(FLAGS.agent)
    sweep = (
        sweep[FLAGS.agent_id_start :]
        if FLAGS.agent_id_end == -1
        else sweep[FLAGS.agent_id_start : FLAGS.agent_id_end]
    )

    with open("agents_" + FLAGS.agent + ".txt", "w") as f:
        for i, agent_config in enumerate(sweep):

            agent_id = FLAGS.agent_id_start + i

            print("agent_id", agent_id, "of", len(sweep))

            f.write(
                "agent_id=" + str(agent_id) + " " +
                " ".join(
                    [
                        str(k) + "=" + str(v)
                        for (k, v,) in agent_config.settings.items()
                    ]
                )
                + "\n"
            )


if __name__ == "__main__":
    app.run(main)
