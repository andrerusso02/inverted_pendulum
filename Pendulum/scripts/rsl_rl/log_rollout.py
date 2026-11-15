# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE)
#
# SPDX-License-Identifier: BSD-3-Clause

"""Log a greedy sim rollout (full episode incl. swing-up) to CSV for sim-vs-real comparison.

Columns: t, raw_action, cmd_target, gim_pos, gim_vel, gim_torque, free_pos, free_vel, tilt_deg
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="Template-Pendulum-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--out", type=str, default="/tmp/sim_rollout.csv")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
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
    motor_id = robot.find_joints("gim6010")[0][0]
    pole_id = robot.find_joints("free_0")[0][0]
    body_id = robot.find_bodies("bearing_axe")[0][0]
    action_term = uw.action_manager.get_term("motor_position")

    n_steps = int(round(args_cli.seconds / uw.step_dt))
    rows = []
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(n_steps):
            actions = policy(obs)
            raw = actions[0, 0].item()
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
            tgt = action_term.processed_actions[0, 0].item() if hasattr(action_term, "processed_actions") else float("nan")
            quat = robot.data.body_quat_w[0, body_id, :]
            local_axis = torch.tensor([1.0, 0.0, 0.0], device=uw.device)
            axis_w = quat_apply(quat, local_axis)
            tilt = torch.rad2deg(torch.acos(axis_w[2].clamp(-1.0, 1.0))).item()
            rows.append([
                (step + 1) * uw.step_dt,
                raw,
                tgt,
                robot.data.joint_pos[0, motor_id].item(),
                robot.data.joint_vel[0, motor_id].item(),
                robot.data.applied_torque[0, motor_id].item(),
                robot.data.joint_pos[0, pole_id].item(),
                robot.data.joint_vel[0, pole_id].item(),
                tilt,
            ])

    with open(args_cli.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "raw_action", "cmd_target", "gim_pos", "gim_vel", "gim_torque", "free_pos", "free_vel", "tilt_deg"])
        w.writerows(rows)
    print(f"[INFO]: wrote {len(rows)} rows to {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
