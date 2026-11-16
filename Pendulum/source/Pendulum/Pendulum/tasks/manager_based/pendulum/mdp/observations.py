from collections.abc import Callable, Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.noise import UniformNoiseCfg

from .buffers import ObservationHistoryBuffer


def free_joint_pos_sin_cos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return selected joint orientations as [sin(q), cos(q)] feature blocks."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.cat((torch.sin(q), torch.cos(q)), dim=-1)


def sin_cos_uniform_noise(data: torch.Tensor, cfg: UniformNoiseCfg) -> torch.Tensor:
    """Perturb each orientation by uniform angular noise, preserving the unit circle.

    Rotating the encoded pair is equivalent to adding noise in radians before
    encoding. Keeping this in the noise hook respects enable_corruption=False.
    """
    sin_q, cos_q = data.chunk(2, dim=-1)
    delta = torch.empty_like(sin_q).uniform_(cfg.n_min, cfg.n_max)
    sin_delta, cos_delta = torch.sin(delta), torch.cos(delta)
    return torch.cat(
        (sin_q * cos_delta + cos_q * sin_delta, cos_q * cos_delta - sin_q * sin_delta),
        dim=-1,
    )


class DelayedObservation(ManagerTermBase):
    """Delay observations once per control step, with per-environment episode resets.

    The manager owns this term and calls ``reset`` for the selected environments.
    Repeated reads within ``common_step_counter`` return the same delayed sample.
    Terms with the same ``latency_group`` share per-environment delays.
    Noise and observation history remain the observation manager's responsibility.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        choices = cfg.params["latency_choices"]
        if not choices or any(type(lag) is not int or lag < 0 for lag in choices):
            raise ValueError("latency_choices must contain non-negative integer delays.")
        self._choices = torch.tensor(choices, dtype=torch.int, device=self.device)
        self._buffer = ObservationHistoryBuffer(max(choices) + 1, self.num_envs, self.device)
        self._time_lags = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        # Groups belong to this environment, never to a global registry.
        group_name = cfg.params.get("latency_group")
        self._latency_lags = [self._time_lags]
        if group_name is not None:
            if not hasattr(env, "_observation_latency_groups"):
                env._observation_latency_groups = {}
            groups = env._observation_latency_groups
            if group_name not in groups:
                groups[group_name] = (tuple(choices), [])
            group_choices, lags = groups[group_name]
            if group_choices != tuple(choices):
                raise ValueError(f"Latency group {group_name!r} must use identical latency_choices.")
            lags.append(self._time_lags)
            self._latency_lags = lags
        self._last_step = None
        self._age = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._initial_obs = None
        self._cached_obs = None
        self._reset_pending = False
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear selected histories and sample their delays for the new episode."""
        count = self.num_envs if env_ids is None else len(env_ids)
        if count == 0:
            return
        ids = slice(None) if env_ids is None else env_ids
        self._buffer.reset(batch_ids=env_ids)
        if env_ids is None:
            self._age.fill_(-1)
        else:
            reset_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            self._age.index_fill_(0, reset_ids, -1)
        self._reset_pending = True
        indices = torch.randint(len(self._choices), (count,), device=self.device)
        # Each term is reset separately. Propagate each draw to all members so
        # the final draw is shared regardless of the manager's reset order.
        lags = self._choices[indices]
        # Choices were validated once at construction. Assigning a sampled choice
        # needs no GPU min/max reduction or scalar readback on each reset.
        for time_lags in self._latency_lags:
            time_lags[ids] = lags

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        base_func: Callable[..., torch.Tensor],
        latency_choices: list[int],
        latency_group: str | None = None,
    ) -> torch.Tensor:
        new_step = self._last_step != env.common_step_counter
        if new_step or self._reset_pending:
            fresh = self._age < 0
            current_obs = base_func(env, asset_cfg=asset_cfg)
            if self._initial_obs is None:
                self._initial_obs = torch.empty_like(current_obs)
                self._cached_obs = torch.empty_like(current_obs)
            mask_shape = (-1,) + (1,) * (current_obs.ndim - 1)
            self._initial_obs = torch.where(fresh.view(mask_shape), current_obs, self._initial_obs)
            if new_step:
                self._age += 1
                self._buffer.append(current_obs)
                self._cached_obs = self._buffer[self._time_lags]
                self._last_step = env.common_step_counter
            self._age.masked_fill_(fresh, 0)
            self._reset_pending = False

            # A recorder may read before a partial reset in the same step. Do not
            # push again for continuing environments. Hold each reset sample until
            # enough new steps exist to read its delay from the buffer instead.
            warmup = self._age <= self._time_lags
            self._cached_obs = torch.where(warmup.view(mask_shape), self._initial_obs, self._cached_obs)

        return self._cached_obs.clone()
