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


# Encoder-zero mismatch between sim and the real AS5600 mount. The 11-14 20 s hardware run
# showed why this must be randomized: trained with upright == the +-pi wrap point, the policy
# treated every wrap crossing as "still upright". On hardware the wrap sits ~0.2 rad away from
# true upright, so crossings happened at 10-20 deg tilt and the 2*pi obs jump triggered violent
# corrections: 2.4 crossings/s, 5x command chatter near crossings, 17 deg mean hold tilt, hold
# eventually lost.
# Magnitude is a curriculum knob (settable per env_cfg instance): +-0.35 from scratch was
# UNLEARNABLE (iter 9: raw upright never exceeded 0.42, sigma inflated to 3.0 - position obs
# too ambiguous, and the 32x32 net could not infer verticality from dynamics alone). The
# working plan: phase 1 trains at 0.0 (exactly the proven iter-3 recipe), phase 2 resumes
# with +-0.12 to fine-tune tolerance around an already-competent balance policy.
POLE_ZERO_OFFSET_RANGE = (-0.35, 0.35)


def pole_pos_with_random_zero_offset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Wrapped joint position with a per-env randomized encoder-zero offset applied.

    The offset is resampled whenever ``episode_length_buf == 0`` (at init and at resets that
    happen before obs computation), so each env behaves like a robot with its own encoder
    mount. ``offset_range`` overrides the module default; the env config passes its own
    field so training phases can curriculum the magnitude.
    """
    asset = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    if offset_range is None:
        offset_range = POLE_ZERO_OFFSET_RANGE
    key = "_pole_zero_offset"
    if not hasattr(env, key):
        setattr(env, key, torch.zeros(env.num_envs, 1, device=env.device))
    offset = getattr(env, key)
    just_reset = env.episode_length_buf <= 0
    lo, hi = offset_range
    if lo == 0.0 and hi == 0.0:
        offset.zero_()
    elif just_reset.any():
        offset[just_reset] = torch.empty(int(just_reset.sum()), 1, device=env.device).uniform_(lo, hi)
    return wrap_to_pi(pos + offset)



def delayed_observation(
    env,
    asset_cfg: SceneEntityCfg,
    base_func: callable,
    latency_choices: list[int],
    obs_name: str,
    base_func_kwargs: dict | None = None
) -> torch.Tensor:
    """Computes a delayed observation using a dynamically attached environment buffer."""

    # 1. Compute the current observation using the passed base function
    current_obs = base_func(env, asset_cfg=asset_cfg, **(base_func_kwargs or {}))

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
