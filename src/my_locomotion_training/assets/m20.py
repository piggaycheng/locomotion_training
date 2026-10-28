# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deep Robotics M20 configuration, based on the Isaac Sim 6.2 asset."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

M20_USD_PATH = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/6.2/Isaac/Robots/DeepRobotics/M20/M20.usd"
)
LEG_JOINTS = ".*_(hipx|hipy|knee)_joint"
WHEEL_JOINTS = ".*_wheel_joint"
WHEEL_BODIES = ".*_wheel"
WHEEL_RADIUS = 0.09
STANDING_HEIGHT = 0.49

M20_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=M20_USD_PATH,
        activate_contact_sensors=True,
        variants={"Physics": "PhysX", "Sensor": "None", "Robot": "Robot"},
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=1.0,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
        ),
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, STANDING_HEIGHT),
        joint_pos={
            ".*_hipx_joint": 0.0,
            "f[lr]_hipy_joint": -0.65,
            "h[lr]_hipy_joint": 0.65,
            "f[lr]_knee_joint": 1.3,
            "h[lr]_knee_joint": -1.3,
            WHEEL_JOINTS: 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[LEG_JOINTS],
            joint_effort_limit=76.4,
            joint_velocity_limit=22.4,
            stiffness=80.0,
            damping=2.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[WHEEL_JOINTS],
            joint_effort_limit=21.6,
            joint_velocity_limit=79.3,
            stiffness=0.0,
            damping=2.0,
        ),
    },
)
