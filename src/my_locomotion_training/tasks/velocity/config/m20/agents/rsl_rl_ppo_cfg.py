# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2-style PPO baselines with separate M20 experiment directories."""

from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg


@configclass
class M20RoughPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    experiment_name = "m20_rough"
    max_iterations = 3000


@configclass
class M20FlatPPORunnerCfg(M20RoughPPORunnerCfg):
    experiment_name = "m20_flat"
    max_iterations = 1500

    def __post_init__(self):
        super().__post_init__()
        self.actor.hidden_dims = [256, 128, 128]
        self.critic.hidden_dims = [256, 128, 128]
