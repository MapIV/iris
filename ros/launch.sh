#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
EXECUTE_DIR=$(pwd)

# Since the path of the pcd file is written in the config.yaml,
# the file cannot be opened unless this scripts is called from an appropriate location. 
if [ ${SCRIPT_DIR} != ${EXECUTE_DIR} ]; then
  echo "Please call this script from ${SCRIPT_DIR}"
  exit 1
fi

# analyze the arguments
if [ $# -ne 1 ]; then
  echo "This script requires 1 arguments."
  echo "You set $# arguments"
  echo "(ex) ./launch.sh ./moriyama.pcd"
  exit 1
fi

# manage `roscore`
if pgrep -x "roscore" > /dev/null
then
  echo "roscore is running"
else
  echo "Because roscore is stopping, it starts roscore"
  roscore &
  sleep 3
fi

# start rviz
#rosrun rviz rviz &

# execution
${SCRIPT_DIR}/devel/lib/vllm/vllm_node -c $1
