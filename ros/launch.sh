#!/bin/bash

if [ $# -ne 1 ]; then
  echo "指定された引数は$#個です。"
  echo "実行するには1個の引数が必要です。"
  exit 1
fi


if pgrep -x "roscore" > /dev/null
then
  echo "roscore is running"
else
  echo "Because roscore is stopping, it starts roscore"
  roscore &
fi

./devel/lib/vllm/vllm_node -c $1
