# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wheel contact observations and rolling rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

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


def _no_lateral_command(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    return env.command_manager.get_command(command_name)[:, 1].abs() < threshold


def _flat_terrain(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg | None, max_relief: float) -> torch.Tensor:
    """True where the height scan under the robot varies less than ``max_relief`` [m]."""
    if sensor_cfg is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    hits_z = env.scene[sensor_cfg.name].data.ray_hits_w.torch[..., 2]
    finite = torch.isfinite(hits_z)
    high = torch.where(finite, hits_z, torch.full_like(hits_z, -torch.inf)).amax(dim=1)
    low = torch.where(finite, hits_z, torch.full_like(hits_z, torch.inf)).amin(dim=1)
    return (high - low) < max_relief


def joint_deviation_no_lateral(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    lateral_threshold: float = 0.1,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    max_relief: float = 0.05,
) -> torch.Tensor:
    """L1 deviation from default joints, active only without a lateral command.

    Pure forward/backward/yaw commands should be tracked by wheel rolling with the legs held
    at their default pose, while lateral commands need the hips and legs to move. If
    ``terrain_sensor_cfg`` is given, the penalty is also disabled on non-flat terrain (stairs).
    """
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    deviation = robot.data.joint_pos.torch[:, ids] - robot.data.default_joint_pos.torch[:, ids]
    active = _no_lateral_command(env, command_name, lateral_threshold) & _flat_terrain(
        env, terrain_sensor_cfg, max_relief
    )
    return deviation.abs().sum(dim=-1) * active


def leg_vel_no_lateral(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    lateral_threshold: float = 0.1,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    max_relief: float = 0.05,
) -> torch.Tensor:
    """Penalize leg joint speed without a lateral command, on flat terrain only."""
    velocity = env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids]
    active = _no_lateral_command(env, command_name, lateral_threshold) & _flat_terrain(
        env, terrain_sensor_cfg, max_relief
    )
    return velocity.square().sum(dim=-1) * active


def wheel_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_time: float = 0.5,
    max_lifted: int = 2,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize having more than ``max_lifted`` wheels off the ground (anti hopping/bounding).

    Up to ``max_lifted`` wheels may be lifted at once for stepping or diagonal trotting;
    the clipped air time of every additional lifted wheel is penalized.
    """
    air_time = env.scene[sensor_cfg.name].data.current_air_time.torch[:, sensor_cfg.body_ids]
    lifted = air_time > threshold
    excess = (lifted.sum(dim=-1) - max_lifted).clamp(min=0)
    return air_time.clamp(max=max_time).sum(dim=-1) * (excess > 0) * excess


def wheel_lateral_clearance_shortfall(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.12,
    command_name: str = "base_velocity",
    lateral_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize laterally swinging wheels that stay too close to the ground.

    Active only when a lateral command is given. Wheel height is measured relative to the
    lowest wheel in each environment (the supporting stance height), weighted by the wheel's
    base-frame lateral speed so stance wheels are not penalized and swinging wheels must lift
    by ``target_height`` [m] to avoid dragging.
    """
    robot = env.scene[asset_cfg.name]
    pos_w = robot.data.body_pos_w.torch[:, asset_cfg.body_ids]
    vel_w = robot.data.body_lin_vel_w.torch[:, asset_cfg.body_ids]
    root_quat = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, pos_w.shape[1], -1)
    vel_b = quat_apply_inverse(root_quat, vel_w)
    z = pos_w[..., 2]
    clearance = z - z.amin(dim=-1, keepdim=True)
    shortfall = (target_height - clearance).clamp(min=0.0).square()
    lateral_speed = vel_b[..., 1].abs()
    active = ~_no_lateral_command(env, command_name, lateral_threshold)
    return (shortfall * lateral_speed).sum(dim=-1) * active


def wheel_long_lift(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, grace_time: float = 0.4) -> torch.Tensor:
    """Penalize a wheel held in the air longer than ``grace_time`` [s] (a leg that stops being used)."""
    air_time = env.scene[sensor_cfg.name].data.current_air_time.torch[:, sensor_cfg.body_ids]
    return (air_time - grace_time).clamp(min=0.0, max=1.0).sum(dim=-1)


def wheel_air_time_variance(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_time: float = 0.5) -> torch.Tensor:
    """Penalize uneven air/contact time across wheels so every leg takes part in stepping."""
    data = env.scene[sensor_cfg.name].data
    ids = sensor_cfg.body_ids
    last_air = data.last_air_time.torch[:, ids].clamp(max=max_time)
    last_contact = data.last_contact_time.torch[:, ids].clamp(max=max_time)
    return last_air.var(dim=-1) + last_contact.var(dim=-1)


def wheel_stance_shortfall(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_ratio: float = 0.9,
    command_name: str = "base_velocity",
    lateral_threshold: float = 0.1,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    max_relief: float = 0.05,
) -> torch.Tensor:
    """Penalize front/rear wheel spacing falling below ``min_ratio`` of its reference value.

    Wheel positions are expressed in the base frame, so the gap is independent of heading.
    The reference gaps are taken once from the median over environments on the first call,
    when the robots are still near the default standing pose. Only shrinkage is penalized, and
    only without a lateral command on flat terrain when ``terrain_sensor_cfg`` is given (legs
    move freely when stepping sideways or climbing stairs).
    Wheels must be ordered as selected by ``asset_cfg`` (front/rear are split by base-frame x).
    """
    robot = env.scene[asset_cfg.name]
    pos_w = robot.data.body_pos_w.torch[:, asset_cfg.body_ids]
    root_pos = robot.data.root_pos_w.torch.unsqueeze(1)
    root_quat = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, pos_w.shape[1], -1)
    pos_b = quat_apply_inverse(root_quat, pos_w - root_pos)
    front = pos_b[..., 0] > 0.0
    n_front = front.sum(dim=-1).clamp(min=1)
    n_rear = (~front).sum(dim=-1).clamp(min=1)
    x_front = (pos_b[..., 0] * front).sum(dim=-1) / n_front
    x_rear = (pos_b[..., 0] * ~front).sum(dim=-1) / n_rear
    gap = x_front - x_rear
    ref = getattr(env, "_wheel_stance_ref", None)
    if ref is None:
        ref = gap.median().detach()
        env._wheel_stance_ref = ref
    shortfall = (min_ratio * ref - gap).clamp(min=0.0).square()
    active = _no_lateral_command(env, command_name, lateral_threshold) & _flat_terrain(
        env, terrain_sensor_cfg, max_relief
    )
    return shortfall * active
