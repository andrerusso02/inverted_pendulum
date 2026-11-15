#!/usr/bin/env bash
# Short, bounded deployment run: starts the bringup and a rosbag, waits for the inference
# node to begin commanding the motor, records for RUN_SECONDS (default 5, keep <= 5 for
# hardware safety), then gracefully shuts everything down (SIGINT to launch first so motor
# commands stop, then SIGTERM to ros2 bag so the bag is finalized).
#
# Usage: utils/safe_deploy_run.sh [run_seconds]
# ROS setup scripts reference unbound variables; keep -u off entirely.
RUN_SECONDS="${1:-5}"
BAG_ROOT="$HOME/pendulum_ws/bags"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
BAG_DIR="$BAG_ROOT/run_$STAMP"
LOG_DIR="/tmp/deploy_$STAMP"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$BAG_ROOT" "$LOG_DIR"

source /opt/ros/jazzy/setup.bash
source "$HOME/pendulum_ws/install/setup.bash"

LAUNCH_PID=""
BAG_PID=""

stop_pid_group() {
  local pid="$1" sig="${2:-INT}"
  [[ -z "$pid" ]] && return 0
  kill -0 "$pid" 2>/dev/null || return 0
  kill -"$sig" -- "-$pid" 2>/dev/null || kill -"$sig" "$pid" 2>/dev/null
}

# Wait up to N seconds for a pid to disappear; escalate INT -> TERM -> KILL.
reap_pid_group() {
  local pid="$1" i
  [[ -z "$pid" ]] && return 0
  for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.5; done
  stop_pid_group "$pid" TERM
  for i in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.5; done
  stop_pid_group "$pid" KILL
}

cleanup() {
  trap - EXIT INT TERM
  # motor commands stop first, then close the bag
  stop_pid_group "$LAUNCH_PID" INT
  reap_pid_group "$LAUNCH_PID"
  stop_pid_group "$BAG_PID" TERM
  reap_pid_group "$BAG_PID"
}
trap cleanup EXIT INT TERM

echo "[deploy] bag:   $BAG_DIR"
echo "[deploy] logs:  $LOG_DIR"

setsid ros2 bag record -o "$BAG_DIR" --topics \
  /dynamic_joint_states \
  /joint_group_position_controller/commands \
  /joint_states > "$LOG_DIR/bag.log" 2>&1 &
BAG_PID=$!

echo "[deploy] starting bringup..."
setsid ros2 launch ros_pendulum bringup.launch.py > "$LOG_DIR/bringup.log" 2>&1 &
LAUNCH_PID=$!

echo "[deploy] waiting for first motor command (max 120s)..."
if ! python3 "$SCRIPT_DIR/wait_first_cmd.py" --timeout 120 > "$LOG_DIR/first_cmd.log" 2>&1; then
  echo "[deploy] ERROR: no command received. Aborting (cleanup will stop everything)."
  cat "$LOG_DIR/first_cmd.log"
  echo "[deploy] last 40 lines of bringup log:"
  tail -40 "$LOG_DIR/bringup.log"
  exit 1
fi
cat "$LOG_DIR/first_cmd.log"
echo "[deploy] motor is live. Holding ${RUN_SECONDS}s..."
sleep "$RUN_SECONDS"

echo "[deploy] stopping launch (motor commands cease)..."
stop_pid_group "$LAUNCH_PID" INT
reap_pid_group "$LAUNCH_PID"
LAUNCH_PID=""
echo "[deploy] closing bag..."
stop_pid_group "$BAG_PID" TERM
reap_pid_group "$BAG_PID"
BAG_PID=""

echo "[deploy] done. Bag: $BAG_DIR"
ros2 bag info "$BAG_DIR"
