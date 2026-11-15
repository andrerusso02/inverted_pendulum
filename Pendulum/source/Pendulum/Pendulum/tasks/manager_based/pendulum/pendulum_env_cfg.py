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
    # EMA-filtered position target. Measured on the 18-02 policy: the action spectrum peaks at
    # 12.5-14.2 Hz with 98-99% of its power above 6 Hz and the torque pinned at ~10 of 11 N.m —
    # Kapitza vibration, not the ~1.03 Hz resonant pumping that a real swing-up needs. A reward
    # penalty cannot fix this: any accel/action-rate penalty is LARGER at random init than in the
    # converged shake, so it crushes init_noise_std before the policy escapes the bound-termination
    # trap (observed: mean_std -> 0.03 within 25 iterations, never recovers). Filtering the command
    # removes the mechanism instead. alpha=0.15 at 100 Hz is a 2.6 Hz first-order low-pass:
    # 1 Hz passes at 93%, 13 Hz is attenuated 5x. DC gain is unchanged, so slow authority is intact.
    motor_position = EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=["gim6010"],
            # Span the reachable envelope: the termination bound is +-2 rad, so a target beyond
            # that is unreachable and just saturates the PD into bang-bang torque.
            scale=2.0,
            rescale_to_limits=False,   # URDF limits are +-100000, so rescaling to them is meaningless
            # Hard-enforce the +-2 rad envelope on the command itself. The termination bound only
            # resets sim episodes; unbounded raw actions (|raw| > 1) produced targets the real
            # controller happily tracked into full rotations. Clip is applied AFTER scale, so the
            # values are in rad. Mirror: EMAJointPositionFilter in pendulum_inference_pipeline.py.
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
            func=mdp.delayed_observation,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=2,                # <-- Added history here
            flatten_history_dim=True,        # <-- Flatten current and past
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"]),
                "base_func": mdp.joint_pos,
                # 11-13 hardware run: capture failed although obs signs/latency looked sane.
                # Widen latency coverage (CAN + controller adds up to ~30 ms on top of the
                # encoder path) and noise so the policy stays robust at the real plant's edge.
                "latency_choices": [1, 2, 3],
                "obs_name": "motor_pos",
            }
        )
        motor_vel = ObsTerm(
            func=mdp.delayed_observation,
            noise=UniformNoiseCfg(n_min=-0.3, n_max=0.3),
            history_length=2,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["gim6010"]),
                "base_func": mdp.joint_vel,
                "latency_choices": [1, 2, 3],
                "obs_name": "motor_vel",
            }
        )

        # --- Free joint (pole) ---
        # Wrapped to +-pi to match the real AS5600 encoder (encoder_hw_node wraps every
        # sample), PLUS a per-env randomized encoder-zero offset. The 11-14 20 s hardware run
        # showed the failure this fixes: with upright == the wrap point (as in plain-wrapped
        # training), the real wrap sat ~0.2 rad off true upright and every crossing triggered
        # 5x command chatter (0.167 vs 0.033 rad |dcmd|), degrading the hold to ~17 deg mean
        # tilt and eventually losing it.
        # +-0.35 from scratch was UNLEARNABLE (iter 9: raw upright capped at 0.42, sigma
        # inflated to 3.0 - the 32x32 net could not infer verticality from dynamics when the
        # position obs was that ambiguous). Phase 2 instead FINE-TUNES the proven iter-3
        # checkpoint with a mild +-0.12 (~7 deg): enough to move the wrap off upright and
        # teach mismatch tolerance, small enough that the balance skill stays learnable.
        pole_pos = ObsTerm(
            func=mdp.delayed_observation,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=2,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"]),
                "base_func": mdp.pole_pos_with_random_zero_offset,
                "base_func_kwargs": {"offset_range": (-0.12, 0.12)},
                "latency_choices": [1, 2, 3, 4, 5],
                "obs_name": "pole_pos",
            }
        )
        pole_vel = ObsTerm(
            func=mdp.delayed_observation,
            # AS5600 is 12-bit and the node finite-differences at ~3.5-12 ms: quantization
            # alone gives ~+-0.44 rad/s of velocity noise, 3x the previous +-0.15 setting.
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
            history_length=22,
            flatten_history_dim=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"]),
                "base_func": mdp.joint_vel,
                "latency_choices": [1, 2, 3, 4, 5],
                "obs_name": "pole_vel",
            }
        )

        action_history = ObsTerm(
            func=mdp.last_action,
            history_length=3,
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
    # Back to the proven iter-3 spread (+-0.1). The +-1.2 widening (iter 4) produced a coarse
    # swing-up style that never captured on hardware (0/2 runs); +-0.3 (5-series) changed
    # nothing decisive. Real runs start near 0 anyway: a balancing policy walks the motor back
    # to ~0 (measured starts: -0.007 / -0.095 rad).
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
    # Varies the mass of all links by +/- 30% (widened from 20% after the 11-13 hardware run:
    # the real plant sat at the edge of the trained distribution - it reached upright
    # repeatedly but never captured).
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
    # Motor torque limits and electrical dynamics vary as motors heat up.
    # Widened to 0.5-2.0x (was 0.8-1.2x): the real ODrive position loop (pos_gain 20,
    # vel_limit 30 rad/s) is far stiffer than the sim PD (k=2, 11 Nm cap), and the narrow
    # range left that gap uncovered.
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

    # 4. Transient pole pushes (iter 11) - trains push RECOVERY.
    # Sim push-probe on the deployed iter-10 policy: a gentle 0.2 s force pulse on the wheel
    # (finger-push equivalent, ~2-3 rad/s pole velocity) was recovered by only 7%/1.6% of envs
    # (+/- direction) - both directions panic-slam the motor into the bound. Cause: the pole
    # is never physically disturbed during balance in training (obs noise is not force), so
    # "quiet hold + sudden pole velocity" is fully OOD. On hardware this shows up as the
    # observed push asymmetry: one side's panic happens to chase correctly, the other spins.
    # NOT the persistent-bias approach of iter 7/8 (equilibrium shift / stagnation): two
    # interleaved interval events make telegraph-noise pulses - "on" samples a random local-
    # frame force on the wheel (tip, longest lever) every 0.5-2.0 s, "off" zeroes it within
    # 0.1-0.25 s, so pulses last ~0.125 s on average: dOmega ~ F*l*dt/I ~= up to ~2.5 rad/s,
    # matching the measured human push (2.1 rad/s initial pole velocity in the 11-14 bag).
    pole_push_on = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.5, 2.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["wheel"]),
            "force_range": (-0.8, 0.8),
            "torque_range": (0.0, 0.0),
        },
    )

    pole_push_off = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.1, 0.25),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["wheel"]),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
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
    # (4b) Kapitza stabilisation is open loop, so it survives latency that breaks feedback
    # control. This taxes exactly its high-frequency action content. Computed on the normalised
    # action, so the weight stays valid if the action scale changes.
    # action_rate = RewTerm(
    #     func=mdp.action_rate_l2,
    #     weight=-0.1,
    # )
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
    # (5) Shaping tasks: lower pole angular velocity
    # Re-enabled for hardware gentleness: a quietly balancing pole means slow, low-torque motor
    # moves. Kept small so the swing-up phase (which needs pole speed to pump) is not strangled.
    pole_joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["free_0"])},
    )

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
    # Back to +-2 (iter 6). The +-3 widening (iters 4-5) removed a useful shaping pressure:
    # with +-2, parking/riding the motor at extremes terminates the episode, which pushed PPO
    # out of the "spin-survive" local optimum early (iter 3 escaped by iter ~900 to raw upright
    # 0.71). With +-3 that gait survives indefinitely: iter 5f stagnated at raw upright 0.475.
    # Hardware consequence (motor can ride the real +-2 clip in an OOD state) is accepted;
    # iter 3 escaped it on hardware anyway (captured at ~4.4 s after ~2 s at the clip).
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
        self.decimation = 2
        # Back to 5 s (11-14): the 8 s experiment destabilised PPO in both directions -
        # entropy 0.01 inflated sigma to 10 (iter 5b, NaN at 2659) and 0.005 collapsed it to
        # 0.12 and plateaued at raw upright 0.35 (iter 5c, killed at ~800). 5 s + entropy 0.01
        # is the proven-stable recipe (iters 2-4 all reached raw upright 0.71-0.79).
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
