# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab.managers import ActionTermCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg

from . import mdp

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg

##
# Scene definition
##
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        "/home/arusso/git/Pendulum/robot/robot.urdf",
        force_usd_conversion=True, # force reconvert USD to ensure latest changes
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            # Just tells to use the PID. I can control joints in position if I set non-zero stiffness and damping
            drive_type="force",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg( # those are overidded
                stiffness=0.0,
                damping=0.0
            ),
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32, 
            solver_velocity_iteration_count=32,
        ),
    ),
    # initial state not picked up... Works from URDF
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0), joint_pos={"gim6010": 3.14159, "free_0": 0}
    ),

    actuators={
        "motor_actuator": ImplicitActuatorCfg(
            joint_names_expr=["gim6010"],
            effort_limit=11.0,  # N.m (Actual motor rated 11 N.m peak, 5 N.m continuous)
            effort_limit_sim=11.0,
            velocity_limit=21.0,  # rad/s (200 RPM, actual motor rated 195-242 RPM)
            velocity_limit_sim=21.0,
            stiffness=2.0,
            damping=0.1,
        ),
        "free_joint_actuator": ImplicitActuatorCfg(
            joint_names_expr=["free_0"],
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.05,
        ),
    },
)

class PendulumSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    # robot
    robot = PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



##
# MDP settings
##


@configclass
class ActionsCfg:
    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["gim6010"], scale=11.0)
    motor_position = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gim6010"],
            scale= 10.0,      # network output [-1, 1] × scale
            offset= 0.0,      # added after scaling, useful to bias the neutral point
            use_default_offset= False,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        motor_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"])}
        )
        motor_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"])}
        )

        # --- Free joint (pole) ---
        pole_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"])}
        )
        pole_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"])}
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"]),
            "position_range": (math.pi - 0.1, math.pi + 0.1),
            "velocity_range": (-0.1 , 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )





@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    pole_upright = RewTerm(
        func=mdp.reward_pole_upright,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["bearing_axe"])},
    )

    # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    # )

    # (4) Shaping tasks: lower cart velocity
    actuator_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Too many rotations
    too_many_rotations = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"]), "bounds": (-7., 7.)},
    )


##
# Environment configuration
##


@configclass
class PendulumEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: PendulumSceneCfg = PendulumSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 8
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation