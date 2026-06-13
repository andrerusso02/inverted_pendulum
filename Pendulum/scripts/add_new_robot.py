# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from Pendulum.tasks.manager_based.pendulum.pendulum_env_cfg import PendulumSceneCfg


ROBOT_ENTITY_KEY = "robot"


def wave(scene: InteractiveScene, sim_time: float, sim_dt: float):
    """A simple wave motion for the pendulum."""
    wave_action = scene[ROBOT_ENTITY_KEY].data.default_joint_pos.clone()
    wave_action[0] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)
    scene[ROBOT_ENTITY_KEY].set_joint_position_target(wave_action)

current_goal = 0.0
frequency = 1.

def rotate(scene: InteractiveScene, sim_time: float, sim_dt: float):
    """A simple rotation for the pendulum using position targets."""
    global current_goal
    
    # 1. Increment the goal position smoothly
    current_goal += 2 * np.pi * frequency * sim_dt
    
    # 2. Wrap the goal to the [-pi, pi] range to match PhysX joint wrapping
    # wrapped_goal = np.arctan2(np.sin(current_goal), np.cos(current_goal))
    
    # 3. Get current default positions for all envs and joints
    rotation_action = scene[ROBOT_ENTITY_KEY].data.default_joint_pos.clone()
    
    # 4. Find the specific index of the motor joint
    motor_joint_idx = scene[ROBOT_ENTITY_KEY].find_joints("gim6010")[0][0]
    
    # 5. Apply the wrapped target position only to the motor joint across all envs
    rotation_action[:, motor_joint_idx] = current_goal

    current_pos = scene[ROBOT_ENTITY_KEY].data.joint_pos.clone()
    print(f"Current motor joint position: {current_pos[:, motor_joint_idx].cpu().numpy()}, Target position: {rotation_action[:, motor_joint_idx].cpu().numpy()}")

    # 6. Set the targets
    scene[ROBOT_ENTITY_KEY].set_joint_position_target(rotation_action)
    

def inpulses(scene: InteractiveScene, sim_time: float, sim_dt: float):
    
    if sim_time % 10.0 < 5.:
        goal = 10.0
    else:
        goal = -0.0

    rotation_action = scene[ROBOT_ENTITY_KEY].data.default_joint_pos.clone()
    motor_joint_idx = scene[ROBOT_ENTITY_KEY].find_joints("gim6010")[0][0]
    rotation_action[:, motor_joint_idx] = goal
    scene[ROBOT_ENTITY_KEY].set_joint_position_target(rotation_action)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset
        # if count % 500 == 0:
        #     # reset counters
        #     count = 0
        #     # reset the scene entities to their initial positions offset by the environment origins

        #     root_robot_state = scene[ROBOT_ENTITY_KEY].data.default_root_state.clone()
        #     root_robot_state[:, :3] += scene.env_origins

        #     # copy the default root state to the sim for the robot's orientation and velocity
        #     scene[ROBOT_ENTITY_KEY].write_root_pose_to_sim(root_robot_state[:, :7])
        #     scene[ROBOT_ENTITY_KEY].write_root_velocity_to_sim(root_robot_state[:, 7:])

        #     # copy the default joint states to the sim
        #     joint_pos, joint_vel = (
        #         scene[ROBOT_ENTITY_KEY].data.default_joint_pos.clone(),
        #         scene[ROBOT_ENTITY_KEY].data.default_joint_vel.clone(),
        #     )
        #     scene[ROBOT_ENTITY_KEY].write_joint_state_to_sim(joint_pos, joint_vel)
        #     # clear internal buffers
        #     scene.reset()
        #     print(f"[INFO]: Resetting state...")

        inpulses(scene, sim_time, sim_dt)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = PendulumSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()