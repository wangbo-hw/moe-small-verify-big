# 以小验大操作命令手册

本页保留用户经验中的实际命令及参数含义，按安装环境替换路径。代码块是 Linux/Bash 下分步骤使用的模板，不是需要全部执行的安装脚本；先核对 [环境说明](environment.md)与[采集裁剪约定](routing-and-pruning.md)，再选择本轮所需步骤。内部脚本尚未随技能分发，命令接口需以下载的实现为准。

## 1. 确定路径、取代码和查接口

先在目标容器内探测实际安装位置，再填写变量。以下 `/absolute/...` 均为待替换示意值，不是约定目录；`RUN_DIR` 应位于用户指定的日志/共享盘目录。

```bash
RUNTIME_PYTHON=/absolute/path/to/runtime/bin/python
VLLM_CLI=/absolute/path/to/runtime/bin/vllm
PROJECT_DIR=/absolute/path/to/project
SCRIPTS_DIR="$PROJECT_DIR/scripts"
MODEL_SOURCE=/absolute/path/to/original-model
MODEL_CONVERTED=/absolute/path/to/new-converted-model
MODEL_PRUNED=/absolute/path/to/new-pruned-model
RUN_DIR=/absolute/path/to/new-run
TOPK_SAVE_PATH="$RUN_DIR/topk_ids_by_layer"
TOPK_JSON="$RUN_DIR/top64_experts.json"

test -x "$RUNTIME_PYTHON" && test -x "$VLLM_CLI"
test -d "$PROJECT_DIR" && test -d "$MODEL_SOURCE"
test ! -e "$RUN_DIR" && test ! -L "$RUN_DIR"
mkdir -- "$RUN_DIR"
tree -L 2 -d "$PROJECT_DIR"
```

`tree` 不存在时用 `find "$PROJECT_DIR" -maxdepth 2 -type d`。目录布局可能与原文不同，通过实际文件确定 `SCRIPTS_DIR` 等变量。模型源目录与两个输出目录必须独立，输出不能位于源目录内部。

从 [SKILL.md 的两个入口](../SKILL.md#补丁与裁剪脚本下载入口)获取采集补丁和配套脚本。若使用 Git，以下变量填写站点提供的真实克隆地址、新下载目录和需要的提交，不把网页 `/files/...` 地址填入 `CLONE_URL`：

```bash
: "${CLONE_URL:?填写站点给出的真实 Git 克隆地址}"
: "${DOWNLOAD_DIR:?填写本次新的下载目录}"
: "${SOURCE_REF:?填写要获取的提交或分支}"
test ! -e "$DOWNLOAD_DIR" && test ! -L "$DOWNLOAD_DIR"
git clone -- "$CLONE_URL" "$DOWNLOAD_DIR"
git -C "$DOWNLOAD_DIR" checkout --detach "$SOURCE_REF"
git -C "$DOWNLOAD_DIR" rev-parse HEAD
git -C "$DOWNLOAD_DIR" status --short
```

对 vLLM-Ascend 采集补丁使用原文固定提交 `ba1d0ca445b9ca59bf906ee23b4469f83499742e`；project/scripts 若取 `master`，记录解析后的完整 commit。下载 checkout 用于审阅和适配，不直接替换正在运行的安装目录。

```bash
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/conver_model_flash.py" --help
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/stat_moe.py" --help
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/expert_pruned.py" --help
"$VLLM_CLI" serve --help
```

只检查本轮会用到的脚本；如果接口与下面不同，以实际 `--help` 和源码为准，保留差异记录。原文的 `conver` 拼写保留，不擅自改成 `convert`。

## 2. 配置 CANN、算子与源码路径

原文环境脚本是 `project/set_rl_env_beta.sh`。已有与当前环境匹配的脚本可审阅后 source；需要单独构建时，先解析以下实际位置：

- `CANN_ENV_SCRIPT`：toolkit 的 `set_env.sh`。
- `ATB_ENV_SCRIPT`：需要 ATB 时的 `set_env.sh`。
- `CUSTOM_OPP_ENV_SCRIPT`、`CUSTOM_OPP_ROOT`：需要的自定义算子环境脚本与 vendor 根目录。
- `BISHENGIR_BIN`、`CUSTOM_OP_API_LIB`：所需编译工具与算子动态库目录。
- `VLLM_ASCEND_ROOT`、`VLLM_ROOT`、`TRANSFORMERS_ROOT`；训练另需 `VERL_ROOT`、`BRIDGE_ROOT`。

确认存在且本轮确实需要后，分别执行相应片段：

```bash
test -f "$CANN_ENV_SCRIPT" && source "$CANN_ENV_SCRIPT"
# 以下两项仅在目标运行栈需要时执行。
test -f "$ATB_ENV_SCRIPT" && source "$ATB_ENV_SCRIPT"
test -f "$CUSTOM_OPP_ENV_SCRIPT" && source "$CUSTOM_OPP_ENV_SCRIPT"

test -d "$CUSTOM_OPP_ROOT"
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP_ROOT${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export PATH="$BISHENGIR_BIN:$PATH"
export LD_LIBRARY_PATH="$CUSTOM_OP_API_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

检查 source 后已有条目，避免重复添加同一 vendor。多套自定义算子需要同时保留时，将各自实际目录纳入列表，不能覆盖为单一路径。

```bash
export PYTHONPATH="$VLLM_ASCEND_ROOT:$VLLM_ROOT:$TRANSFORMERS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# 训练场景才需要；按目标实现使用 mbridge 或 MindSpeed-Bridge 的实际根目录。
export PYTHONPATH="$VERL_ROOT:$BRIDGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$RUNTIME_PYTHON" -c 'import sys; print(sys.executable)'
"$RUNTIME_PYTHON" -c 'import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)'
```

导入检查可能触发扩展加载，执行前确认当前运行环境允许。原文的 `/usr/local/Ascend/cann-9.0.0-beta.2`、`/root/geneva2` 只是历史位置，不用于自动推定新环境路径。

## 3. 可选：FP8 权重转 BF16

仅在确认当前权重格式、转换脚本和任务需要后执行。此处恢复原脚本参数，不代表所有 NPU 模型都需要转换。

```bash
test ! -e "$MODEL_CONVERTED" && test ! -L "$MODEL_CONVERTED"
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/conver_model_flash.py" \
  --input_fp8_hf_path="$MODEL_SOURCE" \
  --output_hf_path="$MODEL_CONVERTED" \
  --quant_type=bfloat16
```

转换后校验实际 tensor dtype、scale 处理、索引和加载结果。只有确认不再依赖相应量化元数据时，才修改输出的 `config.json/quantization_config`。chat template 从匹配的模型版本获取；原文的 [DeepSeek-V4 模板](https://modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp/file/view/master/chat_template.jinja?status=0)不能直接套给其他模型。

选择本轮采集的原模型权重：未转换时 `MODEL_BASE="$MODEL_SOURCE"`；转换已验证时 `MODEL_BASE="$MODEL_CONVERTED"`。随后设置并核对 `CHAT_TEMPLATE_BASE`，不要假定每个模型都有独立的 `chat_template.jinja` 文件。

## 4. 可选：安装 AIS-Bench

优先独立评测环境，客户端通过 API 访问推理服务。下列变量分别指目标兼容的基础 Python、新 venv 目录、评测源码目录、固定 commit 和可达的包源。

```bash
: "${EVAL_BASE_PYTHON:?填写兼容该版本 AIS-Bench 的基础 Python}"
: "${AIS_ENV:?填写新的评测 venv 目录}"
: "${AIS_SOURCE:?填写新的评测源码目录}"
: "${AIS_COMMIT:?填写选定的源码 commit}"
: "${PACKAGE_INDEX:?填写当前可达且允许使用的 PyPI 源}"
test ! -e "$AIS_ENV" && test ! -L "$AIS_ENV"
"$EVAL_BASE_PYTHON" -m venv "$AIS_ENV"
AIS_PYTHON="$AIS_ENV/bin/python"
AIS_CLI="$AIS_ENV/bin/ais_bench"

test ! -e "$AIS_SOURCE" && test ! -L "$AIS_SOURCE"
git clone https://github.com/AISBench/benchmark.git "$AIS_SOURCE"
git -C "$AIS_SOURCE" checkout --detach "$AIS_COMMIT"
"$AIS_PYTHON" -I -m pip install -e "$AIS_SOURCE" --use-pep517 -i "$PACKAGE_INDEX"
"$AIS_PYTHON" -I -m pip install -r "$AIS_SOURCE/requirements/api.txt" -i "$PACKAGE_INDEX"
"$AIS_PYTHON" -I -m pip install -r "$AIS_SOURCE/requirements/extra.txt" -i "$PACKAGE_INDEX"
"$AIS_PYTHON" -I -m pip check
"$AIS_CLI" -h
```

原文使用 `git clone https://gitee.com/aisbench/benchmark.git` 与 `https://repo.huaweicloud.com/repository/pypi/simple`。按可达性选用，并核对镜像源码版本；自定义 V4 API 配置可能在 project 下的 benchmark fork，不能假定官方 checkout 自带它。

如果环境已准备好全部依赖，需要恢复原文的“跳过依赖安装”方式，使用下面这一分支，而非同时执行两套安装。`--no-deps` 不保证兼容，`pip check` 失败时仍需处理：

```bash
"$AIS_PYTHON" -I -m pip install -e "$AIS_SOURCE" --use-pep517 --no-deps -i "$PACKAGE_INDEX"
"$AIS_PYTHON" -I -m pip install -r "$AIS_SOURCE/requirements/api.txt" --no-deps -i "$PACKAGE_INDEX"
"$AIS_PYTHON" -I -m pip install -r "$AIS_SOURCE/requirements/extra.txt" --no-deps -i "$PACKAGE_INDEX"
"$AIS_PYTHON" -I -m pip check
```

原文记录过下面的版本修复组合，保留用于追溯；它不是通用依赖约束，也不应自动安装到训练环境：

```text
pip install datasets==4.8.4 fsspec==2026.2.0 antlr4-python3-runtime==4.9.3 -i https://repo.huaweicloud.com/repository/pypi/simple
```

## 5. 启动原模型并采集专家路由

先确定资源归属及实际设备编号，配置 `VISIBLE_DEVICES`、`TP_SIZE`、`DP_SIZE`、`MAX_MODEL_LEN`、`MAX_BATCHED_TOKENS`、`MAX_SEQS`、`MEMORY_UTILIZATION`、`SERVICE_HOST`、`SERVICE_PORT` 和 `BASE_SERVED_NAME`。

| 变量/参数 | 原文示例，仅供对照 |
|---|---|
| `VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15` |
| `TP_SIZE` / `DP_SIZE` | `4` / `4` |
| `MAX_MODEL_LEN` / `MAX_BATCHED_TOKENS` | `22528` / `22528` |
| `MAX_SEQS` / `MEMORY_UTILIZATION` | `48` / `0.9` |
| `SERVICE_HOST` / `SERVICE_PORT` | `0.0.0.0` / `8006` |

上表不是默认启动配置；首次观测优先降低并发和 token 预算，确定端口和监听范围。原经验的 Ascend 环境开关如下，仅保留目标镜像确认支持且需要的项：

```bash
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

在新目录采集，并在独立终端前台启动原模型；完整原始日志保存到本轮目录。若 tokenizer 自带模板且不需要外部文件，核对后去掉 `--chat-template` 一项。

```bash
test ! -e "$TOPK_SAVE_PATH" && test ! -L "$TOPK_SAVE_PATH"
mkdir -- "$TOPK_SAVE_PATH"
export TOPK_SAVE_PATH
set -o pipefail
"$VLLM_CLI" serve "$MODEL_BASE" \
  --host "$SERVICE_HOST" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --served-model-name "$BASE_SERVED_NAME" \
  --gpu-memory-utilization "$MEMORY_UTILIZATION" \
  --data-parallel-size "$DP_SIZE" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-num-seqs "$MAX_SEQS" \
  --async-scheduling \
  --enable-expert-parallel \
  --chat-template "$CHAT_TEMPLATE_BASE" \
  --port "$SERVICE_PORT" \
  --enforce-eager 2>&1 | tee "$RUN_DIR/base-server.log"
```

原文使用 `--max_model_len`；此处展示常见的 `--max-model-len`，实际名称以当前 CLI 为准。`--async-scheduling` 也来自原文，须验证与当前采集补丁及后端兼容，不支持时明确移除。`TOPK_SAVE_PATH` 必须由已适配的补丁实际读取，单独设置变量不会凭空产生路由记录。

## 6. 服务就绪后排除 dummy/warmup

在另一个终端用实际 API 地址确认服务就绪，检查本轮日志：

```bash
: "${API_BASE_URL:?填写客户端可访问的 API 地址，例如 http://127.0.0.1:8006}"
curl --fail --silent --show-error "$API_BASE_URL/v1/models"
find "$TOPK_SAVE_PATH" -maxdepth 2 -type f -name '*.txt' -print
```

原文在此执行 `rm -rf topk_ids_by_layer/*` 清缓存。保留这个历史说明，但不把它做成自动操作：服务可能仍在写文件，路径也可能指向别轮结果。

按实际 hook 支持的功能，在采样边界启用采集或按请求/phase 过滤 warmup。若旧脚本只能清空文件后计数，先核对写入是否已静止、目录是否仅属于本轮，并保存 warmup 记录，再为这份实现制定具体清理或切换步骤。不能假定重命名目录或更改外部 shell 环境就会让 worker 自动切换。

## 7. 用 AIS-Bench 驱动校准请求

原自定义配置位置是 `project/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_V4.py`，按实际源码布局寻找并检查 endpoint、served model name、模板和生成参数。

```bash
BASE_API_CONFIG=vllm_api_general_V4
DATASET_CONFIG=gsm8k_gen
CALIBRATION_COUNT=500

"$AIS_CLI" --models "$BASE_API_CONFIG" --datasets "$DATASET_CONFIG" --search
set -o pipefail
"$AIS_CLI" --models "$BASE_API_CONFIG" --datasets "$DATASET_CONFIG" \
  --num-prompts "$CALIBRATION_COUNT" --num-warmups=0 \
  --work-dir "$RUN_DIR/base-evaluation" 2>&1 | tee "$RUN_DIR/base-evaluation.log"
```

本节展示明确的复数参数；原文命令是 `--dataset gsm8k_gen --num-prompt 500 --num-warmups=0`，若旧版帮助确实使用该写法再采用。`--num-warmups=0` 不能清除服务启动阶段已有的 dummy 数据。固定实际样本 ID 并检查请求失败/截断，测试集若被用于选专家，应将相应分数标注为校准集效果。

## 8. 统计高频专家

确认采集完整、有效 token 过滤和 rank 去重后，恢复原 `stat_moe.py` 命令：

```bash
KEEP_EXPERTS=64
set -o pipefail
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/stat_moe.py" \
  --top-k="$KEEP_EXPERTS" \
  --log-dir="$TOPK_SAVE_PATH" 2>&1 | tee "$RUN_DIR/stat-moe.log"
```

`64` 是原文保留数示例，按任务改成 32、64 或其他已验证数量。检查该脚本真实输出目录和文件名，将 `TOPK_JSON` 更新为实际生成的文件；修改 `KEEP_EXPERTS` 时不要继续指向旧的 `top64_experts.json`。脚本若缺少有效 token/rank 处理，先补足统计逻辑。

```bash
test -f "$TOPK_JSON"
sha256sum "$TOPK_JSON"
```

## 9. 按高频专家裁剪模型

下面保留内部 `expert_pruned.py` 的原接口。执行前检查源码：它没有在原命令里显式指定输出目录，必须查清派生出的目标路径，并将 `MODEL_PRUNED` 设置为该实际路径。确认该目录全新、源权重不会被改写，hash/shared/MTP 处理与任务一致，再运行：

```bash
test ! -e "$MODEL_PRUNED" && test ! -L "$MODEL_PRUNED"
set -o pipefail
"$RUNTIME_PYTHON" "$SCRIPTS_DIR/expert_pruned.py" \
  --top-k="$KEEP_EXPERTS" \
  --json-path="$TOPK_JSON" \
  --model-dir="$MODEL_BASE" 2>&1 | tee "$RUN_DIR/expert-pruned.log"
```

如果所取得的实现改用 `--topk-json`、支持 `--output-dir` 或 `--dry-run`，按其真实接口改写；不要向原脚本传不存在的参数。没有预检查功能时先审阅或补上只读计划，核对每层 ID 映射、router 切片、shared/hash 保留及索引。原文的自动输出后缀是 `-top64-pruned` 示例，不保证所有版本相同。

## 10. 启动裁剪模型并复测

先让本任务的原模型服务退出并确认资源与端口释放，再运行裁剪模型；不停止其他用户的进程。核对裁剪结果路径及 `CHAT_TEMPLATE_PRUNED`，在新的服务进程中关闭采集。原文只注释 `export` 不会清除当前 shell 已继承的变量，因此这里显式 `unset`：

```bash
unset TOPK_SAVE_PATH
set -o pipefail
"$VLLM_CLI" serve "$MODEL_PRUNED" \
  --host "$SERVICE_HOST" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --served-model-name "$PRUNED_SERVED_NAME" \
  --gpu-memory-utilization "$MEMORY_UTILIZATION" \
  --data-parallel-size "$DP_SIZE" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-num-seqs "$MAX_SEQS" \
  --async-scheduling \
  --enable-expert-parallel \
  --chat-template "$CHAT_TEMPLATE_PRUNED" \
  --port "$SERVICE_PORT" \
  --enforce-eager 2>&1 | tee "$RUN_DIR/pruned-server.log"
```

复测客户端使用匹配的自定义配置：

```bash
PRUNED_API_CONFIG=vllm_api_general_V4_prune
"$AIS_CLI" --models "$PRUNED_API_CONFIG" --datasets "$DATASET_CONFIG" --search
set -o pipefail
"$AIS_CLI" --models "$PRUNED_API_CONFIG" --datasets "$DATASET_CONFIG" \
  --num-prompts "$CALIBRATION_COUNT" --num-warmups=0 \
  --work-dir "$RUN_DIR/pruned-evaluation" 2>&1 | tee "$RUN_DIR/pruned-evaluation.log"
```

比较相同样本、模板、生成参数及评分方式；对独立保留集另建评测目录和样本清单。训练或 RL 验收继续按 [validation.md](validation.md)验证真实反向与更新，不能只凭以上推理命令宣告训练可用。
