import torch
from isaaclab.app import AppLauncher

# 1. Initialize the simulation app natively (opens the GUI)
launcher = AppLauncher(headless=False)
app = launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
# Import your specific reward function
from Pendulum.tasks.manager_based.pendulum import mdp

from Pendulum.tasks.manager_based.pendulum.pendulum_env_cfg import PendulumEnvCfg


def set_damped_joints(env_cfg):
    """
    Iterates through the scene configuration and forcefully sets all joint 
    to damped behavior
    """
    # Iterate through all configured entities in the scene
    for entity_name, entity_cfg in vars(env_cfg.scene).items():
        
        # 1. Identify Articulation configurations (robots)
        if hasattr(entity_cfg, "actuators") and isinstance(entity_cfg.actuators, dict):
            
            # 2. Zero out explicit actuator configurations
            for actuator_name, actuator in entity_cfg.actuators.items():
                if hasattr(actuator, "stiffness"):
                    # Handle both single float assignments and per-joint dictionary assignments
                    actuator.stiffness = {k: 0.0 for k in actuator.stiffness} if isinstance(actuator.stiffness, dict) else 0.0
                if hasattr(actuator, "damping"):
                    actuator.damping = {k: 0.01 for k in actuator.damping} if isinstance(actuator.damping, dict) else 0.01
                if hasattr(actuator, "effort_limit_sim"):
                    actuator.effort_limit_sim = {k: 1.0 for k in actuator.effort_limit_sim} if isinstance(actuator.effort_limit_sim, dict) else 1.0
                if hasattr(actuator, "effort_limit"):
                    actuator.effort_limit = {k: None for k in actuator.effort_limit} if isinstance(actuator.effort_limit, dict) else None
            
            # 3. Zero out global USD/URDF joint_drive fallbacks (if present)
            if hasattr(entity_cfg, "spawn") and hasattr(entity_cfg.spawn, "joint_drive"):
                if hasattr(entity_cfg.spawn.joint_drive, "gains"):
                    entity_cfg.spawn.joint_drive.gains.stiffness = 0.0
                    entity_cfg.spawn.joint_drive.gains.damping = 0.0


def disable_terminations(env_cfg):
    """
    Iterates through the terminations configuration and sets them to None.
    This prevents the environment from issuing a 'done' signal and resetting the robot.
    """
    if hasattr(env_cfg, "terminations"):
        for term_name in list(vars(env_cfg.terminations).keys()):
            # Ignore built-in python/class attributes
            if not term_name.startswith("__"):
                setattr(env_cfg.terminations, term_name, None)                    


# Instantiate the configuration
env_cfg = PendulumEnvCfg()

# Override the number of environments to 1 for manual debugging
env_cfg.scene.num_envs = 1

env_cfg.sim.device = "cpu"
env_cfg.sim.use_gpu_pipeline = False
env_cfg.sim.physx.use_gpu = False

# Disable all gains for manual debugging
# set_damped_joints(env_cfg)

print(vars(env_cfg.scene.robot))

# Disable terminations for manual debugging
disable_terminations(env_cfg)

# 4. PREVENT EPISODIC RESETS AND VELOCITY INJECTIONS
env_cfg.episode_length_s = 100000.0  # Set an artificially massive timeout
env_cfg.events = None                # Disable the random state resets

# Pass the instance to the environment
env = ManagerBasedRLEnv(cfg=env_cfg)


# 2. Recreate the exact parameter object from your RewTerm
debug_asset_cfg = SceneEntityCfg("robot", body_names=["bearing_axe"])

# 3. Resolve the configuration against the active scene
# This step is critical: it parses "bearing_axe" into the actual integer physics indices 
# so your reward function can extract the correct tensor slices.
debug_asset_cfg.resolve(env.scene)

dummy_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
print("Starting interactive debugging loop...")

# Get the robot object from the scene
robot = env.scene["robot"]

# Find the exact physics index for the gim6010 joint
motor_idx = robot.data.joint_names.index("gim6010")

print("Starting interactive debugging loop...")

# 4. Run the simulation loop
while env.unwrapped.sim.is_playing():
    env.step(dummy_action)
    
    # Extract the processed action from the action manager (after any scaling/clipping)
    # Your ActionsCfg scales the raw action by 100.0, so we check what is actually computed.
    commanded_action = env.action_manager.action[0].cpu().numpy()
    
    
    # Extract joint velocity to monitor the spinning behavior
    joint_vel = robot.data.joint_vel[0, motor_idx].item()
    
    # Evaluate your reward function
    reward_tensor = mdp.reward_pole_upright(env, asset_cfg=debug_asset_cfg)
    
    # Print the diagnostics
    # print(f"Cmd Action: {commanded_action} | Vel: {joint_vel:.4f} rad/s | Reward: {reward_tensor[0].item():.4f}")
    # print(vars(robot.data))

app.close()
