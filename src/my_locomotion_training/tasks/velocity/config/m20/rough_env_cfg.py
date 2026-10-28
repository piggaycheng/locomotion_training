# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""M20 mixed position/velocity control on rough terrain."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)

from my_locomotion_training.assets.m20 import (
    LEG_JOINTS,
    M20_CFG,
    STANDING_HEIGHT,
    WHEEL_BODIES,
    WHEEL_JOINTS,
    WHEEL_RADIUS,
)

from ... import mdp


@configclass
class M20ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[LEG_JOINTS],
        scale=0.25,
        use_default_offset=True,
        clip={LEG_JOINTS: (-1.5, 1.5)},
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[WHEEL_JOINTS],
        scale=20.0,
        use_default_offset=False,
        clip={WHEEL_JOINTS: (-60.0, 60.0)},
    )


@configclass
class M20ObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[LEG_JOINTS])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[LEG_JOINTS])},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        wheel_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[WHEEL_JOINTS])},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.05,
        )
        wheel_contact = ObsTerm(
            func=mdp.wheel_contacts,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[WHEEL_BODIES])},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class M20RewardsCfg(RewardsCfg):
    feet_air_time = None
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-15.0,
        params={"target_height": STANDING_HEIGHT, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    leg_posture = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[LEG_JOINTS])},
    )
    hipx_posture = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hipx_joint"])},
    )
    wheel_slip = RewTerm(
        func=mdp.wheel_slip,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[WHEEL_BODIES]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[WHEEL_BODIES]),
            "wheel_radius": WHEEL_RADIUS,
        },
    )
    wheel_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[WHEEL_JOINTS])},
    )
    wheel_vel_at_rest = RewTerm(
        func=mdp.wheel_vel_at_rest,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[WHEEL_JOINTS]),
            "command_name": "base_velocity",
        },
    )


@configclass
class M20RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: M20ActionsCfg = M20ActionsCfg()
    observations: M20ObservationsCfg = M20ObservationsCfg()
    rewards: M20RewardsCfg = M20RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics.default = self.sim.physics.isaacsim_physx
        self.scene.num_envs = 4096
        self.scene.env_spacing = 3.0
        self.scene.robot = M20_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # Keep the baseline independent of unrelated remote marble/HDR assets.
        self.scene.terrain.visual_material = None
        self.scene.sky_light.spawn.texture_file = None
        self.scene.terrain.max_init_terrain_level = 2
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        terrains["boxes"].grid_height_range = (0.025, 0.08)
        terrains["random_rough"].noise_range = (0.01, 0.05)
        terrains["random_rough"].noise_step = 0.01
        terrains["pyramid_stairs"].step_height_range = (0.025, 0.1)
        terrains["pyramid_stairs_inv"].step_height_range = (0.025, 0.1)

        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.debug_vis = False

        for event_name in ("add_base_mass", "base_com", "base_external_force_torque"):
            getattr(self.events, event_name).params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.reset_base.params["velocity_range"] = {
            axis: (-0.1, 0.1) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }

        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[LEG_JOINTS])
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[LEG_JOINTS])
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_(hipx|hipy|knee)"]
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link"]
