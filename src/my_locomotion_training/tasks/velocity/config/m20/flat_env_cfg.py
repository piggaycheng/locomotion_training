# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""M20 velocity tracking on flat ground."""

from isaaclab.utils import configclass

from .rough_env_cfg import M20RoughEnvCfg


@configclass
class M20FlatEnvCfg(M20RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        for reward_name in ("rolling_leg_posture", "rolling_leg_vel", "wheel_stance"):
            reward = getattr(self.rewards, reward_name, None)
            if reward is not None and "terrain_sensor_cfg" in reward.params:
                reward.params["terrain_sensor_cfg"] = None
        self.curriculum.terrain_levels = None
