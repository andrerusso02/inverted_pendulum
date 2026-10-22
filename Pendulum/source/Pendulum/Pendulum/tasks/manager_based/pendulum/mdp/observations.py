import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.math import wrap_to_pi


def free_joint_pos_modifier(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns joint positions wrapped to the range [-pi, pi]."""
    asset = env.scene[asset_cfg.name]
    # Extract positions and wrap angles
    return wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])



def delayed_observation(
    env, 
    asset_cfg: SceneEntityCfg,
    base_func: callable,
    latency_choices: list[int],
    obs_name: str
) -> torch.Tensor:
    """Computes a delayed observation using a dynamically attached environment buffer."""

    # 1. Compute the current observation using the passed base function
    current_obs = base_func(env, asset_cfg=asset_cfg)

    # 2. Create a unique attribute key for this specific observation buffer
    buffer_key = f"_delay_buffer_{obs_name}"

    # 3. Initialize and attach the buffer to the environment on the first call
    if not hasattr(env, buffer_key):
        max_history = max(latency_choices) + 1
        delay_buffer = DelayBuffer(
            history_length=max_history,
            batch_size=env.num_envs,
            device=env.device
        )

        choices_tensor = torch.tensor(latency_choices, dtype=torch.long, device=env.device)
        random_indices = torch.randint(0, len(latency_choices), (env.num_envs,), device=env.device)
        delay_buffer.set_time_lag(choices_tensor[random_indices])

        # Store the buffer in the environment instance
        setattr(env, buffer_key, delay_buffer)

    # 4. Retrieve the buffer and compute the delayed output
    buffer = getattr(env, buffer_key)
    return buffer.compute(current_obs)
