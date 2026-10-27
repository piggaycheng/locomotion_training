# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in real PhysX checks; see the M20 commands in README.md."""

import os

import gymnasium as gym
import pytest
import torch

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils import resolve_presets

import my_locomotion_training.tasks  # noqa: F401
from my_locomotion_training.tasks.velocity.config.m20.flat_env_cfg import M20FlatEnvCfg
from my_locomotion_training.tasks.velocity.config.m20.rough_env_cfg import M20RoughEnvCfg

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("M20_SIM_TESTS") != "1", reason="Requires the Isaac Sim binaries runtime and GPU."
    ),
]


@pytest.fixture(scope="module")
def simulation(request):
    with launch_simulation(resolve_presets(M20FlatEnvCfg())):
        yield
        # Kit exits the process on close, before pytest can propagate its status.
        if request.session.testsfailed:
            raise RuntimeError("M20 simulation tests failed.")


@pytest.mark.parametrize("terrain,observation_size", [("Flat", 60), ("Rough", 247)])
def test_m20_physics_and_finite_observations(simulation, terrain, observation_size):
    cfg = resolve_presets(M20FlatEnvCfg() if terrain == "Flat" else M20RoughEnvCfg())
    cfg.scene.num_envs = 2
    cfg.seed = 42
    cfg.observations.policy.enable_corruption = False
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    if terrain == "Rough":
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 4
    env = gym.make(f"MyLocomotionTraining-Velocity-{terrain}-M20", cfg=cfg)
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (2, observation_size)
        robot = env.unwrapped.scene["robot"]
        assert robot.num_joints == 16
        actions = torch.zeros((2, 16), device=env.unwrapped.device)
        with torch.inference_mode():
            for _ in range(150):
                obs, reward, terminated, _, _ = env.step(actions)
                assert torch.isfinite(obs["policy"]).all()
                assert torch.isfinite(reward).all()
                if terrain == "Flat":
                    assert not terminated.any()
            if terrain == "Flat":
                assert (robot.data.root_pos_w.torch[:, 2] > 0.35).all()
                actions[:, 12:] = 0.5
                for _ in range(50):
                    obs, reward, _, _, _ = env.step(actions)
                    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                wheel_ids, _ = robot.find_joints(".*_wheel_joint")
                assert robot.data.joint_vel.torch[:, wheel_ids].abs().max() > 1.0
        env.reset()
        assert torch.isfinite(env.unwrapped.scene["robot"].data.joint_pos.torch).all()
    finally:
        env.close()
