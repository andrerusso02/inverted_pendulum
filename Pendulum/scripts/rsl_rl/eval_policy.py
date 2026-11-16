# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE)
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless greedy-policy evaluation: swing-up success, balance accuracy, and motion gentleness."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint greedily and report motion stats.")
parser.add_argument("--num_envs", type=int, default=1000)
parser.add_argument("--task", type=str, default="Template-Pendulum-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_episodes", type=int, default=3, help="Episodes per env to evaluate.")
parser.add_argument(
    "--no-disturbance",
    action="store_true",
    help="Zero the pole-disturbance event torques for a clean-condition eval.",
)
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
from packaging import version
from rsl_rl.runners import OnPolicyRunner

installed_version = metadata.version("rsl-rl-lib")

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Pendulum.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # greedy evaluation on clean observations (deployment-style)
    env_cfg.observations.policy.enable_corruption = False
    if args_cli.no_disturbance:
        # zero any disturbance event terms present in the env cfg
        for term_name in ("pole_disturbance", "pole_push_on", "pole_push_off"):
            term = getattr(env_cfg.events, term_name, None)
            if term is not None:
                term.params["force_range"] = (0.0, 0.0)
                term.params["torque_range"] = (0.0, 0.0)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export deployment artifacts next to the checkpoint (same layout as play.py)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        print(f"[INFO]: Exported policy to: {export_model_dir}")

    uw = env.unwrapped
    robot = uw.scene["robot"]
    motor_id = robot.find_joints("gim6010")[0][0]
    pole_id = robot.find_joints("free_0")[0][0]
    body_id = robot.find_bodies("bearing_axe")[0][0]

    episode_steps = int(round(env_cfg.episode_length_s / uw.step_dt))
    total_steps = episode_steps * args_cli.num_episodes
    # steady-state window: last 40% of each episode (after swing-up)
    steady_start = int(episode_steps * 0.6)

    def tilt_deg():
        quat = robot.data.body_quat_w[:, body_id, :]
        local_axis = torch.zeros(uw.num_envs, 3, device=uw.device)
        local_axis[:, 0] = 1.0
        axis_w = quat_apply(quat, local_axis)
        return torch.rad2deg(torch.acos(axis_w[:, 2].clamp(-1.0, 1.0)))

    ep_tilt, all_tilt, all_mvel, all_pvel, all_torque, all_mpos = [], [], [], [], [], []

    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(total_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)

            if (step % episode_steps) >= steady_start:
                t = tilt_deg()
                ep_tilt.append(t)
                all_tilt.append(t)
                all_mvel.append(robot.data.joint_vel[:, motor_id].abs())
                all_pvel.append(robot.data.joint_vel[:, pole_id].abs())
                all_torque.append(robot.data.applied_torque[:, motor_id].abs())
                all_mpos.append(robot.data.joint_pos[:, motor_id].abs())

            if step % episode_steps == episode_steps - 1:
                ep_mean = torch.stack(ep_tilt).mean(dim=0)
                print(
                    f"episode {step // episode_steps + 1}/{args_cli.num_episodes}: "
                    f"steady tilt mean={ep_mean.mean():6.1f} deg | "
                    f"balanced <15deg: {100.0 * (ep_mean < 15.0).float().mean():5.1f}% | "
                    f"<30deg: {100.0 * (ep_mean < 30.0).float().mean():5.1f}%"
                )
                ep_tilt.clear()

    def stats(name, stack, unit=""):
        s = torch.stack(stack)
        per_env_mean = s.mean(dim=0)
        print(
            f"{name:20s} mean={per_env_mean.mean():7.3f}{unit}  "
            f"median={per_env_mean.median():7.3f}{unit}  "
            f"p90(env)={per_env_mean.quantile(0.9):7.3f}{unit}  "
            f"peak={s.max():7.3f}{unit}"
        )

    print(f"\n=== Greedy eval: {args_cli.num_episodes} episodes x {uw.num_envs} envs, steady-state window ===")
    stats("tilt error", all_tilt, " deg")
    stats("motor |vel|", all_mvel, " rad/s")
    stats("pole |vel|", all_pvel, " rad/s")
    stats("motor |torque|", all_torque, " N.m")
    stats("motor |pos|", all_mpos, " rad")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
