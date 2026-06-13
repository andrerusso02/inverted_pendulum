# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, wrap_to_pi
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def reward_pole_upright(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Args:
        env:       The RL environment.
        asset_cfg: SceneEntityCfg pointing at the pole body
                   (e.g. SceneEntityCfg("robot", body_names=["pole_link"])).


    Returns:
        Tensor of shape (num_envs,) with values in (0, 1].
    """
    asset = env.scene[asset_cfg.name]

    # Body quaternions: (num_envs, num_bodies, 4) — scalar-first (w, x, y, z)
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # (N, 1, 4)
    quat = quat.squeeze(1)                                    # (N, 4)

    # Local Z axis of the body expressed in world frame
    local_axis = torch.zeros(quat.shape[0], 3, device=quat.device)
    local_axis[:, 0] = 1.0                                       # [0, 1, 0]
    pole_axis_in_world = quat_apply(quat, local_axis)               # (N, 3)

    # World up vector
    world_up = torch.zeros_like(pole_axis_in_world)
    world_up[:, 2] = 1.0                                      # [0, 0, 1]

    # Cosine of the angle between the two axes, clamped for numerical safety
    cos_angle = (pole_axis_in_world * world_up).sum(dim=-1).clamp(-1.0, 1.0)  # (N,)

    # Angular error in radians [0, π]
    angle_error = torch.acos(cos_angle)                       # (N,)

    # L2
    reward = 1.0 - angle_error / torch.pi                     # (N,)

    # print(f"Debug reward - Angle error (deg): {(angle_error * 180 / torch.pi).cpu().numpy()}, Reward: {reward.cpu().numpy()}")

    return reward