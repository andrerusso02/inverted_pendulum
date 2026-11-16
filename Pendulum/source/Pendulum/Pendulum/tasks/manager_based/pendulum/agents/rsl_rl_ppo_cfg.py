# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "cartpole_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 11-08 run: mean_std collapsed 0.98 -> 0.07 (entropy -1.13) and pole_upright plateaued
        # at ~0.65-0.75, then regressed. More entropy bonus keeps exploration alive.
        # 0.02 over-corrected: std inflated past 2.0 (full-scale flailing through the +-2 clip)
        # because the entropy gradient 0.02/sigma out-earns the upright gradient. 0.01 splits
        # the difference between collapse (0.005) and inflation (0.02).
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        # Adaptive schedule rode the LR up to its 1e-2 clamp early (10x init) while std collapsed,
        # which destabilised the late-training reward. Tighter KL target keeps updates smaller.
        desired_kl=0.005,
        max_grad_norm=1.0,
    )