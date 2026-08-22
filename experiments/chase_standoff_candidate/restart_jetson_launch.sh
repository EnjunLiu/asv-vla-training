#!/bin/bash
set -eo pipefail
TASK_TEXT="${1:?task text required}"
LOG_NAME="${2:?log name required}"
LOG="/home/jetson/jetson_asv_ws/logs/${LOG_NAME}"

# Kill by numeric PID only. Pattern must match install/lib/vla/* entrypoints.
mapfile -t PIDS < <(pgrep -f 'ros2 launch bringup vla_closed_loop|/install/lib/bridge/bridge_node|/install/lib/vla/language|/install/lib/vla/perception|/install/lib/vla/decision' || true)
if ((${#PIDS[@]})); then
  echo "KILL_PIDS ${PIDS[*]}"
  kill "${PIDS[@]}" 2>/dev/null || true
  sleep 3
  kill -9 "${PIDS[@]}" 2>/dev/null || true
fi
sleep 3

for i in $(seq 1 30); do
  left=$(pgrep -f '/install/lib/vla/(language|perception|decision)|bridge_node|ros2 launch bringup vla_closed_loop' || true)
  if [ -z "$left" ] && ! ss -lntp 2>/dev/null | grep -q ':8080'; then
    echo 8080_FREE
    break
  fi
  if [ -n "$left" ]; then
    echo "STILL_ALIVE $left"
    kill -9 $left 2>/dev/null || true
  fi
  sleep 1
done

# Confirm GPU is clear before Qwen load.
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true

rm -f "$LOG"
set +u
source /opt/ros/humble/setup.bash
source /home/jetson/jetson_asv_ws/install/setup.bash
set +u
cd /home/jetson/jetson_asv_ws
nohup ros2 launch bringup vla_closed_loop.launch.py \
  models_dir:=/home/jetson/jetson_asv_ws/models \
  execution_address:=192.168.137.1 \
  execution_port:=8081 \
  "task_text:=${TASK_TEXT}" \
  >"$LOG" 2>&1 </dev/null &
echo LAUNCH_PID=$!
echo LOG="$LOG"
sleep 2
echo LOG_SIZE=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
