# 以小验大操作命令手册

本手册对应四步流程：启动服务（`--enable-return-routed-experts`）→ 数据集采集 routed experts → 统计热点专家 → 裁剪模型，外加可选的权重转换、aisbench 评测与 RL 下游。代码块是 Linux/Bash 下的模板，不是需要全部执行的安装脚本；路径先在目标环境核实。脚本已随 skill 内置于 `scripts/`，接口以实际 `--help` 与源码为准。

## 1. 变量与脚本接口

以下 `/absolute/...` 均为待替换示意值。`RUN_DIR` 应位于用户指定的日志/共享盘目录，且为全新目录。

```bash
RUNTIME_PYTHON=/absolute/path/to/runtime/bin/python
VLLM_CLI=/absolute/path/to/runtime/bin/vllm
SKILL_DIR=/absolute/path/to/moe-small-verify-big        # skill 所在目录
SCRIPTS_DIR="$SKILL_DIR/scripts"
MODEL_BASE=/absolute/path/to/original-model             # 采集用原模型
MODEL_PRUNED=                                           # Step4 自动派生
RUN_DIR=/absolute/path/to/new-run
KEEP_EXPERTS=64

test -d "$SKILL_DIR" && test -d "$MODEL_BASE"
test ! -e "$RUN_DIR" && test ! -L "$RUN_DIR"
mkdir -- "$RUN_DIR"
```

先核对脚本接口（原文 `conver` 拼写保留，不擅自改成 `convert`）：

```bash
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/collect_routed_experts.py" --help
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/stat_moe.py" --help
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/expert_pruned.py" --help
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/conver_model_flash.py" --help
```

## 2. 配置 CANN、算子与源码路径

按目标镜像已有启动脚本解析实际位置；只 source 本轮确实需要的脚本。

- `CANN_ENV_SCRIPT`：toolkit 的 `set_env.sh`。
- `ATB_ENV_SCRIPT` / `CUSTOM_OPP_ENV_SCRIPT` / `CUSTOM_OPP_ROOT`：需要 ATB 与自定义算子时。
- `VLLM_ASCEND_ROOT`、`VLLM_ROOT`：PYTHONPATH 仅在确实使用源码 checkout 时加入。

```bash
test -f "$CANN_ENV_SCRIPT" && source "$CANN_ENV_SCRIPT"
test -d "$CUSTOM_OPP_ROOT" && export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP_ROOT${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"

"$RUNTIME_PYTHON" -c 'import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)'
```

历史位置（`/usr/local/Ascend/cann-9.0.0-beta.2`、个人目录）只是案例，不作默认值。

## 3. 可选：权重转换与再量化（w4a8 等）

仅在确认权重格式、脚本适配与任务需要后执行。`conver_model_flash.py` 支持从 FP8 源转 BF16，或再量化为 `w8a8-int / w8a8-mx / w4a8-mx / w4a4-mx`；**必须与 `mx_quantize.py` 保持同目录**（import 依赖）。

```bash
test ! -e "$MODEL_CONVERTED" && test ! -L "$MODEL_CONVERTED"
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/conver_model_flash.py" \
  --input_fp8_hf_path="$MODEL_SOURCE" \
  --output_hf_path="$MODEL_CONVERTED" \
  --quant_type=bfloat16        # 或 w8a8-int / w8a8-mx / w4a8-mx / w4a4-mx
```

- 脚本按 `element_size()==1` 识别 FP8 张量，要求源目录含 `model.safetensors.index.json`、`config.json`（DeepSeek-V4 系还需 `compress_ratios`）。裁剪后的模型（config 的 `n_routed_experts` 已更新）可直接再量化。
- ignore/quant_layers 名单按 DeepSeek-V4-Flash 结构写死，换模型必须核对覆盖。
- **w4a8 与原始权重部署推理差异**（精度损失、路由漂移、离线模拟与部署 kernel 不一致、口径归因）见 README「w4a8 与原始模型权重部署推理的差异」，采集-部署尽量同 dtype 口径。
- 选择采集用权重：未转换时 `MODEL_BASE="$MODEL_SOURCE"`；转换已验证时 `MODEL_BASE="$MODEL_CONVERTED"`。

## 4. 可选：安装 AIS-Bench

优先独立评测环境（venv/独立容器），固定源码 commit：

```bash
: "${AIS_ENV:?新 venv 目录}"; : "${AIS_SOURCE:?新源码目录}"; : "${AIS_COMMIT:?选定 commit}"; : "${PACKAGE_INDEX:?可达的 PyPI 源}"
"$EVAL_BASE_PYTHON" -m venv "$AIS_ENV"
AIS_PYTHON="$AIS_ENV/bin/python"; AIS_CLI="$AIS_ENV/bin/ais_bench"
git clone https://github.com/AISBench/benchmark.git "$AIS_SOURCE"
git -C "$AIS_SOURCE" checkout --detach "$AIS_COMMIT"
"$AIS_PYTHON" -I -m pip install -e "$AIS_SOURCE" --use-pep517 -i "$PACKAGE_INDEX"
"$AIS_PYTHON" -I -m pip install -r "$AIS_SOURCE/requirements/api.txt" -i "$PACKAGE_INDEX"
"$AIS_PYTHON" -I -m pip install -r "$AIS_SOURCE/requirements/extra.txt" -i "$PACKAGE_INDEX"
"$AIS_PYTHON" -I -m pip check && "$AIS_CLI" -h
```

经验坑（追溯用，非通用约束）：`setuptools<81`（pkg_resources）、仅装 `opencv-python-headless`、系统 torch 用 `.pth` 挂入 venv（mmengine collect_env 需要 torch）；gitee 源 `https://gitee.com/aisbench/benchmark.git` 与华为云 PyPI 镜像可按可达性替换。

## 5. Step1: 启动原模型服务

用参数化模板 `scripts/vllm_server_base.sh`（或 prompt 指定的其他脚本）。**硬性要求：启动参数必须含 `--enable-return-routed-experts`**，否则响应不附带 routed experts，流程无法继续。

| 变量 | 含义（默认值取自历史经验，仅示意） |
|---|---|
| `MODEL_PATH`（必填） | 原模型目录 |
| `SERVED_NAME` / `PORT` | 服务名 / 端口（默认模型目录名 / 8006） |
| `TP_SIZE` / `DP_SIZE` | `4` / `4` |
| `MAX_MODEL_LEN` / `MAX_BATCHED_TOKENS` / `MAX_NUM_SEQS` / `MEM_UTIL` | `22528` / `22528` / `48` / `0.9` |
| `CHAT_TEMPLATE` | 外置模板路径；模型自带时留空 |
| `EXTRA_ARGS` | 追加参数，如 `--trust-remote-code --dtype bfloat16` |

```bash
MODEL_PATH="$MODEL_BASE" SERVED_NAME=verify-base PORT=8006 \
  bash "$SCRIPTS_DIR/vllm_server_base.sh" 2>&1 | tee "$RUN_DIR/base-server.log"
```

首次观测优先降低并发与 token 预算（如 `MAX_NUM_SEQS=8 MAX_MODEL_LEN=2048 MEM_UTIL=0.9`），并保持 `--enforce-eager`（模板默认）。服务就绪检查与单请求验证：

```bash
curl --fail http://127.0.0.1:8006/v1/models
curl http://127.0.0.1:8006/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"verify-base","messages":[{"role":"user","content":"2+2="}],"max_tokens":8,"temperature":0}'
# 响应中应能找到 "routed_experts" 字段；找不到时排查启动参数与版本支持
```

注意：服务端可能执行 warmup/dummy 请求，但 Step2 采集是客户端行为，只统计采集脚本自己发送的请求，服务端 warmup 天然不计入。

## 6. Step2: 数据集采集 routed experts

prompt 指定数据集。`collect_routed_experts.py` 仅用 Python 标准库，把响应中的 `routed_experts`（base64 字符串或列表）按 `(T, L, K)` 解析，逐层追加写 `<output-dir>/layers.N.txt`（每行一个 token 的 K 个专家 ID），即 `stat_moe.py` 的输入格式。

```bash
TOPK_DIR="$RUN_DIR/topk_ids_by_layer"

# 先小批量核对形状与层号映射
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/collect_routed_experts.py" \
  --api-base http://127.0.0.1:8006 --model verify-base \
  --dataset "$DATASET" --column Question \
  --model-config "$MODEL_BASE/config.json" \
  --output-dir "$TOPK_DIR" \
  --limit 5 --raw-dir "$RUN_DIR/raw"

# 核对无误后全量
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/collect_routed_experts.py" \
  --api-base http://127.0.0.1:8006 --model verify-base \
  --dataset "$DATASET" --column Question \
  --model-config "$MODEL_BASE/config.json" \
  --output-dir "$TOPK_DIR"
```

核对要点（首个请求的日志与 `_collect_manifest.json`）：

1. `T`（数组 token 维）应约等于 `prompt_tokens+completion_tokens-1`；明显不符说明 L/K/dtype 推断错误。
2. `L`（MoE 层数）、`K`（每 token 激活专家数）默认从 `--model-config` 推断（`num_hidden_layers - first_k_dense_replace` 与 `num_experts_per_tok` 等字段）；不符时用 `--num-moe-layers`/`--experts-per-token` 显式指定。
3. 数组层号与模型全局层号不对应（如数组只含 MoE 层）时，用 `--layer-indices "0,1,2,4,..."` 显式映射，或 `--layer-offset` 平移。
4. base64 解码 dtype 默认 `int32`，按实际实现换 `--dtype`（int8/uint8/int16/int32/int64）。
5. 响应里的专家 ID 应为逻辑专家 ID；若该版本回传的是经 EP/log2phy 重映射的物理 ID，需同次运行的反向映射再统计。
6. 失败比例高时不要用于统计（manifest.failures 有详情）；`--raw-dir` 保存原始响应便于排查格式。
7. 数据集格式：CSV 用 `--column`（列名区分大小写，CountBenchQA 类 MCQ 数据用 `Question`；历史数据中该列在第 7 列/0-based）；JSONL/JSON 用 `--field`；纯文本每行一条。校准集应与目标业务分布一致。

## 7. Step3: 统计热点专家

```bash
set -o pipefail
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/stat_moe.py" \
  --top-k="$KEEP_EXPERTS" \
  --log-dir="$TOPK_DIR" 2>&1 | tee "$RUN_DIR/stat-moe.log"
# 输出 $RUN_DIR/top${KEEP_EXPERTS}_experts.json（--log-dir 的上级目录）
```

检查：MoE 层数是否齐全；每层计数 ≈ 有效 token 数 × K；日志有"无有效数据/行无有效数据"警告时排查采集完整性。改 `KEEP_EXPERTS` 重跑会生成新的 `topN_experts.json`，后续步骤不要引用旧文件。

```bash
TOPK_JSON="$RUN_DIR/top${KEEP_EXPERTS}_experts.json"
test -f "$TOPK_JSON" && sha256sum "$TOPK_JSON"
```

## 8. Step4: 裁剪模型并启动验证

```bash
test ! -e "${MODEL_BASE}-top${KEEP_EXPERTS}-pruned"
set -o pipefail
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/expert_pruned.py" \
  --top-k="$KEEP_EXPERTS" \
  --json-path="$TOPK_JSON" \
  --model-dir="$MODEL_BASE" 2>&1 | tee "$RUN_DIR/expert-pruned.log"

MODEL_PRUNED="${MODEL_BASE}-top${KEEP_EXPERTS}-pruned"
```

脚本语义与核对清单（详见 [routing-and-pruning.md](routing-and-pruning.md)）：

- 输出目录自动派生为 `{MODEL_BASE}-top{K}-pruned`，必须为全新目录。
- 前 3 层（`SKIP_LAYERS={0,1,2}`，DeepSeek-V4 的 hash 层）保留全部原专家；其余层裁到 top-K；shared experts 原样保留；`config.json` 的 `n_routed_experts` 改写为 K。
- 该语义要求服务加载器支持"部分层全量 + 其余层 top-K"的异构专家数；不满足时先确认加载适配，不能悄悄改全局专家数凑形状。
- 裁剪后用 `scripts/vllm_server_pruned.sh` 启动（不再需要 `--enable-return-routed-experts`）：

```bash
MODEL_PATH="$MODEL_PRUNED" SERVED_NAME=verify-pruned PORT=8007 \
  bash "$SCRIPTS_DIR/vllm_server_pruned.sh" 2>&1 | tee "$RUN_DIR/pruned-server.log"
```

确认加载成功、每层专家数与路由 ID 范围正确、生成正常后进入评测。

## 9. 可选：aisbench 精度验证与专家数优化

```bash
BASE_API_CONFIG=vllm_api_general_V4          # 自定义模型配置，按实际源码适配 endpoint/name/参数
DATASET_CONFIG=gsm8k_gen
CALIBRATION_COUNT=500

"$AIS_CLI" --models "$BASE_API_CONFIG" --datasets "$DATASET_CONFIG" --search
set -o pipefail
"$AIS_CLI" --models "$BASE_API_CONFIG" --datasets "$DATASET_CONFIG" \
  --num-prompts "$CALIBRATION_COUNT" --num-warmups=0 \
  --work-dir "$RUN_DIR/base-evaluation" 2>&1 | tee "$RUN_DIR/base-evaluation.log"

PRUNED_API_CONFIG=vllm_api_general_V4_prune   # 指向裁剪模型的同构配置
"$AIS_CLI" --models "$PRUNED_API_CONFIG" --datasets "$DATASET_CONFIG" \
  --num-prompts "$CALIBRATION_COUNT" --num-warmups=0 \
  --work-dir "$RUN_DIR/pruned-evaluation" 2>&1 | tee "$RUN_DIR/pruned-evaluation.log"
```

- 历史参数 `--dataset/--num-prompt` 是旧写法，以当前 CLI 帮助为准；`--num-warmups=0` 不能清除服务端已发生的 warmup（采集口径下无影响）。
- 数据集配置变量名必须以 `_datasets` 结尾，否则 aisbench 静默 0 tasks（日志表现为 `Partitioned into 0 tasks`）；CSV 数据集按列号取数（如 GPQADataset 从第 7 列取 Question）。
- **专家数优化**：对多个 `KEEP_EXPERTS`（如 16/32/48/64）重复 Step3→Step4→评测，比较精度与显存/吞吐，选平衡点；每轮的 `topN_experts.json` 与评测目录分开存放。用于选专家的数据集分数只能标注为校准集效果。
- 采集与部署权重 dtype 不同（如采集用 BF16、部署 w4a8）时，先读 README 的差异章节，分口径归因。

## 10. 可选：RL 训练下游

`scripts/rl/run_rl.sh` 以 `-top64-pruned` 裁剪模型为底模跑 verl/GRPO；`run_ray.sh`、`test_1node-beta2.sh` 为 Ray 环境脚本，`docker_rl_run_beta.sh` 为容器启动模板。这些脚本按历史环境写死了路径/卡数/数据集，使用前逐项修改；训练验收（真实反向、训推一致性）见 [validation.md](validation.md)。
