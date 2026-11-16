"""Run with the Isaac Lab Python environment: python scripts/test_observation_buffers.py."""

import unittest
from types import SimpleNamespace

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import torch

from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.buffers import CircularBuffer

from Pendulum.tasks.manager_based.pendulum.mdp.buffers import ObservationHistoryBuffer
from Pendulum.tasks.manager_based.pendulum.mdp.observations import DelayedObservation

DEVICES = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]


class TestObservationBuffers(unittest.TestCase):
    def test_history_matches_isaac_lab(self):
        """Compare wraparound, history order, partial/full resets and lag reads."""
        for device in DEVICES:
            for length in (1, 2, 6, 22):
                for shape in ((1,), (2,), (2, 3)):
                    with self.subTest(device=device, length=length, shape=shape):
                        original = CircularBuffer(length, 7, device)
                        optimized = ObservationHistoryBuffer(length, 7, device)
                        rng = torch.Generator(device=device).manual_seed(42)
                        for step in range(70):
                            if step % 11 == 0:
                                original.reset()
                                optimized.reset()
                            elif step % 3 == 0:
                                # Multiple resets before an append, including empty resets.
                                for ids in ([1, 4], [4, 6], []):
                                    ids = torch.tensor(ids, device=device, dtype=torch.long)
                                    original.reset(ids)
                                    optimized.reset(ids)
                            if original._buffer is not None:
                                self.assertTrue(torch.equal(original.buffer, optimized.buffer))
                            data = torch.randn((7, *shape), generator=rng, device=device)
                            original.append(data)
                            optimized.append(data)
                            self.assertTrue(torch.equal(original.buffer, optimized.buffer))
                            self.assertTrue(torch.equal(original.current_length, optimized.current_length))
                            lag = torch.randint(0, length + 3, (7,), generator=rng, device=device)
                            self.assertTrue(torch.equal(original[lag], optimized[lag]))
                            snapshot = optimized.buffer
                            snapshot.zero_()
                            self.assertTrue(torch.equal(original.buffer, optimized.buffer))

    def make_term(self, device, choices=(0, 1, 3), group=None, env=None):
        if env is None:
            env = SimpleNamespace(num_envs=3, device=device, common_step_counter=0)
        cfg = ObservationTermCfg(func=DelayedObservation, params={
            "latency_choices": list(choices), "latency_group": group,
        })
        return env, DelayedObservation(cfg, env)

    def test_delay_warmup_and_same_step_reset(self):
        for device in DEVICES:
            with self.subTest(device=device):
                env, term = self.make_term(device)
                term._time_lags.copy_(torch.tensor([0, 1, 3], device=device))
                current = torch.zeros(3, 2, device=device)
                def read():
                    return term(env, SceneEntityCfg("robot"), lambda *args, **kwargs: current, [0, 1, 3])

                for step in range(6):
                    env.common_step_counter = step
                    current.fill_(step)
                    expected = torch.tensor([step, max(0, step - 1), max(0, step - 3)], device=device)
                    self.assertTrue(torch.equal(read(), expected[:, None].expand(-1, 2)))
                    current.fill_(999)
                    self.assertTrue(torch.equal(read(), expected[:, None].expand(-1, 2)))

                # A recorder already read step 5. Reset one environment and read
                # again without advancing any continuing environment's history.
                term.reset(torch.tensor([2], device=device))
                term._time_lags[2] = 3
                current.fill_(50)
                self.assertTrue(torch.equal(read()[:, 0], torch.tensor([5, 4, 50], device=device)))
                for step in range(6, 10):
                    env.common_step_counter = step
                    current.fill_(step)
                    expected = [step, step - 1, 50 if step <= 8 else 6]
                    self.assertTrue(torch.equal(read()[:, 0], torch.tensor(expected, device=device)))
                term.reset([])
                self.assertTrue(torch.equal(read()[:, 0], torch.tensor([9, 8, 6], device=device)))
                term.reset()
                current.fill_(77)
                self.assertTrue(torch.equal(read(), current))

    def test_grouped_delays_and_isolation(self):
        for device in DEVICES:
            env, first = self.make_term(device, group="encoder")
            _, second = self.make_term(device, group="encoder", env=env)
            first.reset([0, 2])
            self.assertTrue(torch.equal(first._time_lags, second._time_lags))
            second.reset([1])
            self.assertTrue(torch.equal(first._time_lags, second._time_lags))
            other_env, other = self.make_term(device, choices=(0,), group="encoder")
            self.assertIsNot(env._observation_latency_groups, other_env._observation_latency_groups)
            self.assertEqual(other._time_lags.count_nonzero().item(), 0)
            with self.assertRaises(ValueError):
                self.make_term(device, choices=(0,), group="encoder", env=env)


if __name__ == "__main__":
    try:
        unittest.main()
    finally:
        app.close()
