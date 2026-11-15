# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    # Iter 11: RESUME from iter-10 (2026-11-14_22-22-50/model_4498.pt, the deployed policy)
    # and fine-tune +1500 iters with transient pole-push events enabled (see EventCfg
    # pole_push_on/off). Reason: sim push-probe showed the iter-10 policy recovers from a
    # finger-strength push in only 7%/1.6% of envs (+/- direction) - push recovery was never
    # trained (obs noise != physical force), which on hardware shows up as the observed
    # one-sided push failure. Resume semantics: max_iterations is ADDITIONAL (4498+1500).
    max_iterations = 1500
    seed = 42
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
        # because the entropy gradient 0.02/sigma out-earns the upright gradient.
        # 0.01 + ADAPTIVE LR + 5 s episodes is the proven-stable combo (iters 2-4: sigma ended
        # 0.23-0.47, raw upright 0.71-0.79; iter 3 captured on hardware). Fixed-LR variants all
        # failed: 1e-3 inflated sigma to 10 (5b, 8s), 5e-4 with 0.01 inflated to 2.8 (5d),
        # 0.0075 was stable but converged too slowly (5e: raw upright 0.37 @ iter 1066, killed).
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        # Adaptive is safe AGAIN for 5 s episodes: iters 2-4 ran it stably (LR decayed ~2e-3 ->
        # 8e-5). The 5a divergence (LR pinned at the 1e-2 clamp, value loss 1e8) happened only in
        # combination with 8 s episodes, which are abandoned. desired_kl 0.005 (was 0.01) still
        # guards against the clamp-riding seen in the original 11-08 run.
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )