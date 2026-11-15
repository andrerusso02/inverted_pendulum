# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE)
#
# SPDX-License-Identifier: BSD-3-Clause

"""Directional push-recovery probe: is the recovery asymmetry in the policy or the hardware?

Lets the greedy policy capture the pole, then applies mirrored GENTLE pushes (world-Y force
on the wheel for ~0.2 s, like a finger) to two equal env groups and measures per-direction
recovery. Episodes are extended and bound-termination disabled so the probe mirrors hardware
(the real rig neither times out at 5 s nor terminates at the +-2 motor clip).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1000)
parser.add_argument("--task", type=str, default="Template-Pendulum-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--push_force", type=float, default=0.6, help="World-Y force on the wheel [N] (~0.15 N.m hinge torque).")
parser.add_argument("--push_steps", type=int, default=20, help="Push duration in control steps (20 = 0.2 s).")
parser.add_argument("--capture_steps", type=int, default=300, help="Steps allowed to capture (3 s).")
parser.add_argument("--eval_steps", type=int, default=300, help="Steps observed after push (3 s).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Pendulum.tasks  # noqa: F401

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False
    # Mirror hardware: no time-out resets mid-probe, no bound termination (the real rig rides
    # the +-2 clip and can re-capture from it).
    env_cfg.episode_length_s = 60.0
    env_cfg.terminations.too_many_rotations = None

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uw = env.unwrapped
    robot = uw.scene["robot"]
    body_id = robot.find_bodies("bearing_axe")[0][0]
    wheel_id = robot.find_bodies("wheel")[0][0]
    motor_id = robot.find_joints("gim6010")[0][0]

    def tilt_deg_per_env():
        quat = robot.data.body_quat_w[:, body_id, :]
        local_axis = torch.zeros(uw.num_envs, 3, device=uw.device)
        local_axis[:, 0] = 1.0
        axis_w = quat_apply(quat, local_axis)
        return torch.rad2deg(torch.acos(axis_w[:, 2].clamp(-1.0, 1.0)))

    def set_wheel_force(ids: torch.Tensor, fy: float):
        forces = torch.zeros(len(ids), 1, 3, device=uw.device)
        forces[:, 0, 1] = fy
        torques = torch.zeros_like(forces)
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torques, body_ids=[wheel_id], env_ids=ids
        )

    def clear_wheel_force(ids: torch.Tensor):
        set_wheel_force(ids, 0.0)

    obs = env.get_observations()
    with torch.inference_mode():
        # phase 1: swing up and capture
        for _ in range(args_cli.capture_steps):
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)

        tilt = tilt_deg_per_env()
        ready_ids = (tilt < 10.0).nonzero(as_tuple=True)[0]
        n = len(ready_ids)
        print(f"captured & upright at push time: {n}/{uw.num_envs} ({100.0 * n / uw.num_envs:.0f}%)")
        if n < 20:
            print("too few captured envs; aborting")
            env.close()
            return

        half = n // 2
        pos_ids = ready_ids[:half]
        neg_ids = ready_ids[half:2 * half]
        groups = (("+push", pos_ids, +args_cli.push_force), ("-push", neg_ids, -args_cli.push_force))

        # phase 2a: gentle mirrored pushes for push_steps
        for ids, fy in ((pos_ids, +args_cli.push_force), (neg_ids, -args_cli.push_force)):
            set_wheel_force(ids, fy)
        for _ in range(args_cli.push_steps):
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
        clear_wheel_force(pos_ids)
        clear_wheel_force(neg_ids)

        # phase 2b: observe recovery
        stats = {
            name: dict(
                ok=torch.ones(len(ids), dtype=torch.bool, device=uw.device),
                tilt_sum=torch.zeros(len(ids), device=uw.device),
                tilt_max=torch.zeros(len(ids), device=uw.device),
                gim_absmax=torch.zeros(len(ids), device=uw.device),
            )
            for name, ids, _ in groups
        }
        for _ in range(args_cli.eval_steps):
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
            t = tilt_deg_per_env()
            g = robot.data.joint_pos[:, motor_id].abs()
            d = dones.reshape(-1).bool()
            for name, ids, _ in groups:
                tt = t[ids]
                s = stats[name]
                s["tilt_sum"] += tt
                s["tilt_max"] = torch.maximum(s["tilt_max"], tt)
                s["gim_absmax"] = torch.maximum(s["gim_absmax"], g[ids])
                s["ok"] &= (tt < 25.0) & ~d[ids]

        print(f"\n=== push recovery: F={args_cli.push_force} N on wheel for "
              f"{args_cli.push_steps * uw.step_dt:.2f} s, {args_cli.eval_steps * uw.step_dt:.0f} s window ===")
        for name, ids, _ in groups:
            s = stats[name]
            print(f"{name} (n={len(ids)}): recovered {100.0 * s['ok'].float().mean():5.1f}% | "
                  f"mean tilt {s['tilt_sum'].mean() / args_cli.eval_steps:5.1f} deg | "
                  f"median max-tilt {s['tilt_max'].median():6.1f} deg | "
                  f"p90 max-tilt {s['tilt_max'].quantile(0.9):6.1f} deg | "
                  f"median max|gim| {s['gim_absmax'].median():.2f} rad")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
