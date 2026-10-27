# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deep Robotics M20 velocity-tracking environments."""

import gymnasium as gym

from . import agents

for terrain, cfg_name in (("Flat", "M20FlatEnvCfg"), ("Rough", "M20RoughEnvCfg")):
    gym.register(
        id=f"MyLocomotionTraining-Velocity-{terrain}-M20",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{terrain.lower()}_env_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:M20{terrain}PPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
