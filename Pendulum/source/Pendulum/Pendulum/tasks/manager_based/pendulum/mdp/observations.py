import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi


def free_joint_pos_modifier(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns joint positions wrapped to the range [-pi, pi]."""
    asset = env.scene[asset_cfg.name]
    # Extract positions and wrap angles
    return wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
