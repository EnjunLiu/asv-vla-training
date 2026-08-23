#!/bin/bash
set -eo pipefail
LOG="/home/jetson/jetson_asv_ws/logs/closed_loop_red3_entityfeat_v2_run2_20260823.log"
TASK_TEXT=$'跟随红色目标船，保持3米距离'

mapfile -t PIDS < <(pgrep -f 'ros2 launch bringup vla_closed_loop|/install/lib/bridge/bridge_node|/install/lib/vla/task|/install/lib/vla/perception|/install/lib/vla/decision' || true)
if ((${#PIDS[@]})); then
  kill "${PIDS[@]}" 2>/dev/null || true
  sleep 3
  kill -9 "${PIDS[@]}" 2>/dev/null || true
fi
sleep 2
rm -f "$LOG"
source /opt/ros/humble/setup.bash
source /home/jetson/jetson_asv_ws/install/setup.bash
cd /home/jetson/jetson_asv_ws
nohup ros2 launch bringup vla_closed_loop.launch.py \
  models_dir:=/home/jetson/jetson_asv_ws/models \
  execution_address:=192.168.137.1 \
  execution_port:=8081 \
  "task_text:=${TASK_TEXT}" \
  >"$LOG" 2>&1 </dev/null &
echo LAUNCH_PID=$!
