#!/bin/bash
#set -xeuo pipefail

#Envs
ray stop --force
rm -rf /tmp/ray
rm -rf /root/.triton/cache/
rm -rf /root/.triton/dump/
rm -rf /tmp/torchinductor_root/*
rm -rf /root/.cache/torch_extensions
SCRIPT_PATH=$(readlink -f "$0")

ulimit -n 65536 
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR=/home/z00841464/project/logs/verl/test-node1-$TIMESTAMP
mkdir -p $LOG_DIR
LOG_PATH=$LOG_DIR/test.log
SCRIPT_PATH=$(readlink -f "$0")
cp $SCRIPT_PATH $LOG_DIR


# 关闭训练图模式，待修复训练走入图模式
export TORCHDYNAMO_VERBOSE=1
export TORCH_COMPILE_DEBUG=1
export TORCHDYNAMO_DISABLE=1

#通信
NNODES=2
NPUS_PER_NODE=16

MASTER_ADDR=172.26.4.181
export HCCL_SOCKET_IFNAME=enp162s0f0
export GLOO_SOCKET_IFNAME=enp162s0f0

export ASCEND_PROCESS_LOG_PATH=/home/z00841464/project/plogs/$TIMESTAMP
# VLLM env 
export USE_MULTI_BLOCK_POOL=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=400
export ACL_OP_INIT_MODE=0
export ASCEND_A3_ENABLE=1
export VLLM_VERSION=0.13.0
export TRITON_ALLWAYS_COMPILE=1

export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"

# Project Configuration
project_name='Geneva2'
exp_name='Geneva2-single-node'
# Necessary env
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TASK_QUEUE_ENABLE=1
export STREAMS_PER_DEVICE=32
export CUDA_DEVICE_MAX_CONNECTIONS=1

#other
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1


#获取当前节点IP
# CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
# CURRENT_IP=$(hostname -I | grep -o '172\.26\.4\.1[0-9]\+' | head -n 1)
CURRENT_IP=172.26.4.188
echo $CURRENT_IP
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # 主节点启动
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU":'$NPUS_PER_NODE'}'
  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)pwd
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断 device_count 是否与 NNODES 相等
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
          LOG_DIR=/home/z00841464/project/logs/test-node8-$TIMESTAMP
          mkdir -p $LOG_DIR
          LOG_PATH=$LOG_DIR/test.log
          cp /home/z00841464/project/scripts/sh/run_rl.sh $LOG_DIR
	      bash /home/z00841464/project/scripts/sh/run_rl.sh 2>&1 | tee  $LOG_PATH
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # 子节点尝试往主节点注册ray直到成功
  while true; do
      # 尝试连接 Ray 集群
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # 检查连接是否成功
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi
