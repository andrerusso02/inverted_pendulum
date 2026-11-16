"""Pendulum environment with observation buffers local to this project."""

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.buffers import ObservationHistoryBuffer


class PendulumEnv(ManagerBasedRLEnv):
    def load_managers(self) -> None:
        super().load_managers()
        # Isaac Lab currently has no history-buffer factory in ObservationManager.
        # Replace its freshly constructed, empty buffers before the first reset.
        for group in self.observation_manager._group_obs_term_history_buffer.values():
            for name, history in group.items():
                if history._buffer is not None:
                    raise RuntimeError("Observation histories must be replaced before their first append.")
                group[name] = ObservationHistoryBuffer(history.max_length, history.batch_size, history.device)
