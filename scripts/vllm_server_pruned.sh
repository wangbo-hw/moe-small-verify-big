#!/bin/bash
# ============================================================================
# 以小验大 Step4: 启动裁剪后小模型 vLLM 服务（精度验证用）
#
# 用法:
#   MODEL_PATH=/path/to/model-top64-pruned PORT=8006 \
#     bash scripts/vllm_server_pruned.sh 2>&1 | tee run_dir/pruned-server.log
#
# 与 base 版差异: 默认不带 --enable-return-routed-experts（不再采集）。
#
# 全部变量均有默认值（默认值取自 DeepSeek-V4-Flash 历史经验，仅作示意）。
# ============================================================================
set -o pipefail

# ---- 服务目标（必填 MODEL_PATH）----
MODEL_PATH="${MODEL_PATH:?请设置 MODEL_PATH=裁剪后模型目录}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL_PATH")}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"          # 模型自带模板时留空

# ---- 并行与资源 ----
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8006}"
TP_SIZE="${TP_SIZE:-4}"
DP_SIZE="${DP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-22528}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-22528}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
MEM_UTIL="${MEM_UTIL:-0.9}"

# ---- Ascend 环境变量（历史经验值，按镜像/硬件核对后保留或修改）----
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ---- 组装命令 ----
CMD=(python -m vllm.entrypoints.openai.api_server "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --data-parallel-size "$DP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$MEM_UTIL"
  --enable-expert-parallel
  --enforce-eager
  # 历史经验: 裁剪模型可尝试仅 decode 走图, 提升吞吐; 首次验证建议先用 --enforce-eager,
  # 稳定后再注释上一行并放开下面一行
  # --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
)
[ -n "$CHAT_TEMPLATE" ] && CMD+=(--chat-template "$CHAT_TEMPLATE")
if [ -n "${EXTRA_ARGS:-}" ]; then eval "CMD+=($EXTRA_ARGS)"; fi

echo "[vllm_server_pruned] 启动: ${CMD[*]}"
exec "${CMD[@]}"

# 就绪检查（另开终端执行）:
#   curl --fail http://127.0.0.1:$PORT/v1/models
