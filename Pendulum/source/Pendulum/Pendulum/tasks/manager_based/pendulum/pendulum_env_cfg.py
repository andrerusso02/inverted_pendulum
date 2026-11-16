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
from isaaclab.envs.mdp.actions import EMAJointPositionToLimitsActionCfg, JointPositionActionCfg
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
            # Real drive chain (11-14): ros2_control position interface -> ODrive in
            # CONTROL_MODE_POSITION_CONTROL + INPUT_MODE_PASSTHROUGH ("direct"), i.e. the
            # ODrive internal cascade pos_gain(20) -> vel PI (vel_gain 0.1, Kt 0.097,
            # gear_ratio 8) -> torque_lim 0.3 N.m motor-side. Output-referred equivalent PD:
            #   k_eff = pos_gain * vel_gain * Kt * gear^2 = 20*0.1*0.097*64 ~= 12.4 N.m/rad
            #   c_eff = vel_gain * Kt * gear^2            ~= 0.62 N.m/(rad/s)
            #   tau_max = torque_lim * gear = 0.3*8 = 2.4 N.m  (bag plateau: 2.3-2.9 N.m)
            #   vel_max = vel_limit = 30 rad/s               (bag peak: 17.3 rad/s)
            # The old sim actuator (k=2, c=0.1, 11 N.m) was 5x softer and 4.5x stronger,
            # so policies learned torques the hardware cannot deliver: hardware runs reached
            # upright repeatedly but never captured.
            effort_limit=2.4,  # N.m
            effort_limit_sim=2.4,
            velocity_limit=30.0,  # rad/s
            velocity_limit_sim=30.0,
            stiffness=12.0,
            damping=0.6,
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
    motor_position = EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["gim6010"],
            scale=2.0,
            rescale_to_limits=False,   # URDF limits are +-100000, so rescaling to them is meaningless
            clip={"gim6010": (-2.0, 2.0)},
            alpha=0.15,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        motor_pos = ObsTerm(
            func=mdp.DelayedObservation,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=2,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"]),
                "base_func": mdp.joint_pos,
                "latency_choices": [1, 2, 3],
                "latency_group": "motor_encoder",
            }
        )
        motor_vel = ObsTerm(
            func=mdp.DelayedObservation,
            noise=UniformNoiseCfg(n_min=-0.3, n_max=0.3),
            history_length=2,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"]),
                "base_func": mdp.joint_vel,
                "latency_choices": [1, 2, 3],
                "latency_group": "motor_encoder",
            }
        )

        # --- Free joint (pole) ---
        pole_pos = ObsTerm(
            func=mdp.DelayedObservation,
            noise=UniformNoiseCfg(func=mdp.sin_cos_uniform_noise, n_min=-0.05, n_max=0.05),
            # 20 was worse than 22
            history_length=22,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"]),
                "base_func": mdp.free_joint_pos_sin_cos,
                "latency_choices": [1, 2, 3, 4, 5],
                "latency_group": "pole_encoder",
            }
        )
        # pole_vel = ObsTerm(
        #     func=mdp.DelayedObservation,
        #     # AS5600 is 12-bit and the node finite-differences at ~3.5-12 ms: quantization
        #     # alone gives ~+-0.44 rad/s of velocity noise, 3x the previous +-0.15 setting.
        #     noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        #     history_length=22,
        #     flatten_history_dim=True,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"]),
        #         "base_func": mdp.joint_vel,
        #         "latency_choices": [1, 2, 3, 4, 5],
        #         "latency_group": "pole_encoder",
        #     }
        # )

        action_history = ObsTerm(
            func=mdp.last_action,
            history_length=5,
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
            # Spread the start state over the whole hanging->upright arc instead of clustering it
            # at hanging. free_0=0 is hanging (tilt=pi), free_0=pi is upright (tilt=0). Envs that
            # start near upright learn to balance immediately, which is what supplies the reward
            # gradient the all-hanging start never produced.
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # 1. Mass & Inertia Randomization
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.7, 1.3),
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
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"]),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    # Zeroed: with the pole starting at tilt=pi, this paid 5.0 per episode for doing nothing
    # while `terminating` charged -2.0 for any motion that reached the +-2 rad motor bound.
    # That made passivity the global optimum of the local basin (observed: mean_std -> 0.009,
    # tilt stuck at 82-138 deg even after 1700 iterations). `terminating` alone still makes
    # surviving preferable, since upright reward only accrues while the episode continues.
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # (3) Primary task: keep pole upright
    # Linear in tilt rather than squared: the squared form has zero gradient at tilt=pi, which is
    # exactly where every episode starts, so it gives no signal to begin pumping.
    # Weight 2.0 (was 1.0): the 11-08 run plateaued at mean tilt ~25-35 deg, where the squared
    # reward is flat-ish; doubling it strengthens the gradient toward precise balance relative
    # to the alive bonus, so "just survive" is no longer a competitive strategy.
    pole_upright = RewTerm(
        func=mdp.reward_pole_upright,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["bearing_axe"])},
    )
    # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    # )

    # (4) Shaping tasks: lower cart velocity
    # Weight -0.1 (was -0.02): hardware-gentleness requirement. The 11-08 run settled at
    # mean |motor vel| ~ 2.5 rad/s; the 5x stronger L1 tax pushes the policy toward slow,
    # resonant pumping (~1 Hz) instead of fast corrections.
    actuator_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"])},
    )
    # (4c) Kapitza probe, NOT a shaping term. Keep the weight ~1e-9: any weight large enough to
    # change behaviour is larger at random init than in a converged shake (scale x init_noise_std
    # slams the motor into its bound), so PPO's cheapest response is to crush mean_std to ~0.03
    # and the policy never escapes the bound-termination trap. Read it as a vibration gauge:
    # divide the logged value by the weight for mean(alpha^2). Measured: 845 rad/s^2 RMS for the
    # Kapitza policy vs 62 for the EMA-filtered one. Absolute values are inflated by hard
    # velocity_limit_sim constraint spikes (torque alone caps alpha at 500 rad/s^2), so use it
    # for relative comparisons only. The EMA action filter is what actually suppresses vibration.
    motor_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-9,  # MEASUREMENT ONLY: any larger weight crushes init_noise_std before the policy escapes the bound-termination trap
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"])},
    )

    actuator_centered = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"]), "target": 0.},
    )

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
    scene: PendulumSceneCfg = PendulumSceneCfg(num_envs=12000, env_spacing=1.0)
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
        self.decimation = 1
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 100
        self.sim.render_interval = self.decimation
