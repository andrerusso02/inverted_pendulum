# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import ActionTermCfg, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

##
# Scene definition
##
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from . import mdp

PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        "/home/andre/git/inverted_pendulum/Pendulum/robot/robot.urdf",
        merge_fixed_joints=False,
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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    # initial state not picked up... Works from URDF
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0), joint_pos={"gim6010": 0.0, "free_0": 0.0}
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
            func=mdp.delayed_observation,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"]),
                                "base_func": mdp.joint_pos,
                "latency_choices": [1, 2],
                "obs_name": "motor_pos",
            }
        )
        motor_vel = ObsTerm(
            func=mdp.delayed_observation,
            noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"]),
                "base_func": mdp.joint_vel,
                "latency_choices": [1, 2],
                "obs_name": "motor_vel",
            }
        )

        # --- Free joint (pole) ---
        pole_pos = ObsTerm(
            func=mdp.delayed_observation,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"]),
                "base_func": mdp.joint_pos,
                "latency_choices": [1, 2, 3],
                "obs_name": "pole_pos",
            }
        )
        pole_vel = ObsTerm(
            func=mdp.delayed_observation,
            noise=UniformNoiseCfg(n_min=-0.15, n_max=0.15),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"]),
                "base_func": mdp.joint_vel,
                "latency_choices": [1, 2, 3],
                "obs_name": "pole_vel",
            }
        )

        action_history = ObsTerm(
            func=mdp.last_action,
            history_length=3,
            flatten_history_dim=True
        )


        gim6010_pos_observation_history = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"])},
            history_length=1,
            flatten_history_dim=True
        )

        gim6010_vel_observation_history = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"])},
            history_length=1,
            flatten_history_dim=True
        )

        pole_pos_observation_history = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"])},
            history_length=1,
            flatten_history_dim=True
        )

        pole_vel_observation_history = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"])},
            history_length=1,
            flatten_history_dim=True
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
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
            "position_range": (0.1, 0.1),
            "velocity_range": (-0.1 , 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # 1. Mass & Inertia Randomization
    # Varies the mass of all links by +/- 20%
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # 2. Joint Friction & Armature Randomization
    # Real joints experience stiction and varying friction profiles
    randomize_joint_physics = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.5, 1.5),
            "armature_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )

    # 3. Actuator Gain Randomization
    # Motor torque limits and electrical dynamics vary as motors heat up
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # (3) Primary task: keep pole upright
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
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    # pole_joint_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"])},
    # )

    # actuator_upright = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"]), "target": 3.14159},
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Too many rotations
    too_many_rotations = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"]), "bounds": (-2., 2.)},
    )

    # too_many_rotations_free_0 = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"]), "bounds": (-14., 14.)},
    # )


##
# Environment configuration
##


@configclass
class PendulumEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: PendulumSceneCfg = PendulumSceneCfg(num_envs=6000, env_spacing=1.0)
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
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
