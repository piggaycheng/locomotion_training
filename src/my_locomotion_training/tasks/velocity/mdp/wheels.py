# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wheel contact observations and rolling rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def wheel_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Return one contact flag per wheel from the current net contact forces."""
    forces = env.scene[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return (torch.linalg.vector_norm(forces, dim=-1) > threshold).float()


def wheel_slip(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    wheel_radius: float,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize horizontal velocity at touching wheels' bottom points.

    The upright-wheel approximation uses a world-vertical radius. Body angular
    velocity includes wheel spin, so pure rolling is not mistaken for foot slip,
    and no assumptions about joint-axis signs are necessary.
    """
    robot = env.scene[asset_cfg.name]
    linear = robot.data.body_lin_vel_w.torch[:, asset_cfg.body_ids]
    angular = robot.data.body_ang_vel_w.torch[:, asset_cfg.body_ids]
    radius = torch.zeros_like(linear)
    radius[..., 2] = -wheel_radius
    surface_velocity = linear + torch.linalg.cross(angular, radius, dim=-1)
    contact = wheel_contacts(env, sensor_cfg, threshold)
    return (surface_velocity[..., :2].square().sum(dim=-1) * contact).sum(dim=-1)


def wheel_vel_at_rest(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Discourage spinning wheels when both translation and yaw commands are zero."""
    velocity = env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids]
    command = env.command_manager.get_command(command_name)
    at_rest = torch.linalg.vector_norm(command[:, :3], dim=-1) < command_threshold
    return velocity.square().sum(dim=-1) * at_rest
