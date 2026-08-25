#!/bin/bash
set -eo pipefail
REMOTE_ROOT="${REMOTE_ROOT:-/tmp/asv_moving_collection_expand_20260823}"
mkdir -p "$REMOTE_ROOT"
cd /home/jetson/jetson_asv_ws
# ROS setup scripts reference unset vars; do not enable nounset around them.
source /opt/ros/humble/setup.bash
source install/setup.bash
pkill -f '/install/lib/bridge/bridge_node' || true
sleep 2
nohup ros2 run bridge bridge_node --ros-args --params-file src/bridge/config/ue_bridge.yaml \
  -p outbound_command_mode:=disabled -p use_sim_time:=true \
  >"$REMOTE_ROOT/bridge.log" 2>&1 </dev/null &
echo $! >"$REMOTE_ROOT/bridge.pid"
sleep 3
pgrep -af bridge_node
echo BRIDGE_READY
