#!/bin/bash
set -eo pipefail
LOG_NAME="${LOG_NAME:-closed_loop.log}"
LOG="/home/jetson/jetson_asv_ws/logs/${LOG_NAME}"
TASK_TEXT="${TASK_TEXT:-follow the red boat, keep 3 meters distance}"

mapfile -t PIDS < <(pgrep -f 'ros2 launch bringup vla_closed_loop|/install/lib/bridge/bridge_node|/install/lib/vla/' || true)
if ((${#PIDS[@]})); then
  kill "${PIDS[@]}" 2>/dev/null || true
  sleep 5
  kill -9 "${PIDS[@]}" 2>/dev/null || true
fi
sleep 8
# Free fragmented CUDA / page cache before loading vision+policy.
sync || true
rm -f "$LOG"
source /opt/ros/humble/setup.bash
source /home/jetson/jetson_asv_ws/install/setup.bash
cd /home/jetson/jetson_asv_ws
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup ros2 launch bringup vla_closed_loop.launch.py \
  models_dir:=/home/jetson/jetson_asv_ws/models \
  perception_model_path:=/home/jetson/jetson_asv_ws/models/vision.pt \
  perception_start_delay_sec:=5.0 \
  policy_start_delay_sec:=20.0 \
  policy_device:=cpu \
  visual_device:=cuda \
  execution_address:=192.168.137.1 \
  execution_port:=8081 \
  "task_text:=${TASK_TEXT}" \
  >"$LOG" 2>&1 </dev/null &
echo LAUNCH_PID=$!
