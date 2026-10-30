export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1

#环境变量，所有节点执行
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

python -m vllm.entrypoints.openai.api_server \
  --model /home/z00841464/data/DeepSeek-V4-Flash-bf16-top64-pruned \
  --host 0.0.0.0 \
  --max_model_len 22528 \
  --max-num-batched-tokens 22528 \
  --served-model-name DeepSeek-V4-Flash-bf16-top64-pruned \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --max-num-seqs 48 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'\
  --async-scheduling \
  --enable-expert-parallel \
  --chat-template /home/z00841464/data/DeepSeek-V4-Flash-bf16-top64-pruned/chat_template.jinja \
  --port 8006 \
  --enforce-eager


# curl http://127.0.0.1:8006/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "DeepSeek-V4-Flash-bf16-pruned",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "2+2="
#             }
#         ],
#         "max_tokens": 256,
#         "temperature": 0
#     }'