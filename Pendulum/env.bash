# Source this in env activate script

# 1. Clear all ROS-specific environment variables
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset ROS_LOCALHOST_ONLY
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH

# 2. Clean up PATH, LD_LIBRARY_PATH, and PYTHONPATH to remove system ROS 2 traces
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "opt/ros" | paste -sd: -)
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "opt/ros" | paste -sd: -)
export PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\n' | grep -v "opt/ros" | paste -sd: -)

# 3. Set Isaac Sim's required internal ROS 2 variables
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/jazzy/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/andre/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib
