# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wheel-specific behavior without launching the simulator."""

import re
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg

from my_locomotion_training.tasks.velocity.config.m20.flat_env_cfg import M20FlatEnvCfg
from my_locomotion_training.tasks.velocity.config.m20.rough_env_cfg import M20RoughEnvCfg
from my_locomotion_training.tasks.velocity.mdp.wheels import wheel_contacts, wheel_slip, wheel_vel_at_rest


def _buffer(value):
    return SimpleNamespace(torch=torch.tensor(value, dtype=torch.float32))


def test_wheel_slip_distinguishes_rolling_sliding_and_airborne():
    # Rolling forward/backward, locked-wheel sliding, lateral sliding, and airborne spin.
    linear = [[[0.9, 0, 0]], [[-0.9, 0, 0]], [[0.9, 0, 0]], [[0.9, 0.5, 0]], [[0, 0, 0]]]
    angular = [[[0, 10, 0]], [[0, -10, 0]], [[0, 0, 0]], [[0, 10, 0]], [[0, 100, 0]]]
    forces = [[[0, 0, 20]]] * 4 + [[[0, 0, 0]]]
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(body_lin_vel_w=_buffer(linear), body_ang_vel_w=_buffer(angular))
            ),
            "contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=_buffer(forces))),
        }
    )
    asset = SceneEntityCfg("robot")
    sensor = SceneEntityCfg("contact_forces")
    actual = wheel_slip(env, asset, sensor, wheel_radius=0.09)
    torch.testing.assert_close(actual, torch.tensor([0.0, 0.0, 0.81, 0.25, 0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(wheel_contacts(env, sensor).flatten(), torch.tensor([1, 1, 1, 1, 0]).float())


def test_rest_penalty_allows_translation_and_turning_in_place():
    command = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_vel=_buffer([[2.0, 3.0]] * 3)))},
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    actual = wheel_vel_at_rest(env, SceneEntityCfg("robot"), command_name="base_velocity")
    torch.testing.assert_close(actual, torch.tensor([13.0, 0.0, 0.0]))


@pytest.mark.parametrize("cfg_type", [M20FlatEnvCfg, M20RoughEnvCfg])
def test_control_and_position_terms_exclude_continuous_wheels(cfg_type):
    cfg = cfg_type()
    cfg.validate()
    names = [f"{leg}_{joint}_joint" for leg in ("fl", "fr", "hl", "hr") for joint in ("hipx", "hipy", "knee", "wheel")]

    def selected(patterns):
        return {name for name in names if any(re.fullmatch(pattern, name) for pattern in patterns)}

    legs = selected(cfg.actions.joint_pos.joint_names)
    wheels = selected(cfg.actions.wheel_vel.joint_names)
    assert len(legs) == 12 and len(wheels) == 4
    assert legs.isdisjoint(wheels) and legs | wheels == set(names)
    for term in (cfg.observations.policy.joint_pos, cfg.rewards.leg_posture, cfg.rewards.dof_pos_limits):
        assert selected(term.params["asset_cfg"].joint_names) == legs
    assert selected(cfg.rewards.hipx_posture.params["asset_cfg"].joint_names) == {
        f"{leg}_hipx_joint" for leg in ("fl", "fr", "hl", "hr")
    }
    assert selected(cfg.observations.policy.wheel_vel.params["asset_cfg"].joint_names) == wheels
    assert selected(cfg.rewards.dof_acc_l2.params["asset_cfg"].joint_names) == legs
    assert selected(cfg.rewards.wheel_acc_l2.params["asset_cfg"].joint_names) == wheels
    assert cfg.scene.robot.actuators["wheels"].stiffness == 0.0
    assert cfg.actions.wheel_vel.use_default_offset is False
    assert cfg.rewards.feet_air_time is None


def test_flat_configuration_does_not_mutate_rough_height_terms():
    flat = M20FlatEnvCfg()
    rough = M20RoughEnvCfg()
    assert flat.scene.height_scanner is None
    assert flat.rewards.base_height_l2.params["sensor_cfg"] is None
    assert rough.scene.height_scanner.prim_path.endswith("/base_link")
    assert rough.rewards.base_height_l2.params["sensor_cfg"].name == "height_scanner"
    assert rough.scene.terrain.terrain_generator.curriculum
    for name in ("add_base_mass", "base_com", "base_external_force_torque"):
        assert getattr(rough.events, name).params["asset_cfg"].body_names == ["base_link"]
