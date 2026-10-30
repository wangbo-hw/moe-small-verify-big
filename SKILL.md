---
name: moe-small-verify-big
description: MoE 以小验大：启动 vLLM 服务（必须带 --enable-return-routed-experts），用数据集请求服务并从响应采集 routed experts，统计逐层热点专家，再按热点专家裁剪出小模型；可选用 aisbench 精度验证优化保留专家数。适用于 vLLM/vLLM-Ascend 部署的 MoE 模型（DeepSeek-V4-Flash 等）。
---

# MoE 以小验大

完整流程四步：**启动服务（--enable-return-routed-experts）→ 数据集采集 routed experts → 统计热点专家 → 裁剪小模型**，外加可选的 aisbench 精度验证来优化保留专家数。全部脚本随本 skill 内置于 `scripts/`，无需另下补丁或克隆仓库。

```
原模型 ──Step1 启动服务──> 响应携带 routed_experts
             │
数据集 ──Step2 采集──> run_dir/topk_ids_by_layer/layers.N.txt
             │
             └─Step3 stat_moe.py──> run_dir/top64_experts.json
             │
             └─Step4 expert_pruned.py──> {model}-top64-pruned（新模型目录）
                                          │
                                          └─ (可选) aisbench 精度验证 / 多保留数扫描
```

## 执行前确认的输入

从当前对话（用户 prompt）获取，缺失且影响执行的才追问，一次问全：

1. **Step1**：启动脚本（默认用本 skill 的 `scripts/vllm_server_base.sh`）、模型路径、服务参数（TP/DP/端口/max-model-len 等）；无论用哪个脚本，启动参数必须包含 `--enable-return-routed-experts`。
2. **Step2**：数据集文件（csv/jsonl/txt，问题列名或字段）。
3. **Step3/4**：保留专家数 `KEEP_EXPERTS`（如 64）；输出目录 `RUN_DIR`（必须全新目录，不覆盖源权重）。
4. 是否需要可选的 aisbench 精度验证（及是否扫描多个 KEEP 值）。

## Step1 启动原模型服务

用 `scripts/vllm_server_base.sh`（环境变量参数化：`MODEL_PATH`、`SERVED_NAME`、`PORT`、`TP_SIZE`、`DP_SIZE`、`MAX_MODEL_LEN`、`MAX_NUM_SEQS`、`MEM_UTIL`、`CHAT_TEMPLATE`、`EXTRA_ARGS`），或用户 prompt 指定的脚本。硬性要求：

- **必须带 `--enable-return-routed-experts`**，否则响应中不会附带 routed experts，后续步骤全部无法进行。
- 服务就绪后先用单请求确认响应里有 `routed_experts` 字段（模板见脚本尾部注释），再进入 Step2。
- 首次观测用低并发、小 token 预算；确认 `v1/models` 返回正常后再放量。

## Step2 用数据集访问服务并采集 routed experts

prompt 指定数据集后执行：

```bash
python scripts/collect_routed_experts.py \
  --api-base http://127.0.0.1:8006 --model <SERVED_NAME> \
  --dataset <数据集> [--column Question | --field question] \
  --model-config <模型目录/config.json> \
  --output-dir "$RUN_DIR/topk_ids_by_layer" \
  [--limit 5 --raw-dir "$RUN_DIR/raw"]     # 先小批量核对
```

- 脚本把响应中的 routed_experts（base64 或列表）按 `(T, L, K)` 解析，逐层写成 `layers.N.txt`（每行一个 token 的 K 个专家 ID），正好是 `stat_moe.py` 的输入格式。
- **先 `--limit 5` 小批量核对**：看 manifest 与日志里 `T` 与 prompt+completion token 数的关系是否符合预期（约等于 `prompt+completion-1`）；`L`（MoE 层数）与 `K`（每 token 激活专家数）默认从 config.json 推断，推断不符时用 `--num-moe-layers`/`--experts-per-token` 显式指定；数组层号与模型全局层号不对应时用 `--layer-indices` 校正。
- 核对无误后去掉 `--limit` 全量采集；失败比例高时不要用于统计。
- 校准数据集的选取决定热点专家的覆盖面，应与目标业务分布一致；数据集问题列名区分大小写（CountBenchQA 类 MCQ 数据用 `--column Question`）。

## Step3 统计热点专家

```bash
python scripts/stat_moe.py --top-k "$KEEP_EXPERTS" --log-dir "$RUN_DIR/topk_ids_by_layer"
# 输出 $RUN_DIR/top${KEEP_EXPERTS}_experts.json（位于 --log-dir 的上级目录）
```

检查日志：MoE 层数是否齐全、每层计数是否约等于"有效 token 数 × K"。换 `KEEP_EXPERTS` 重跑时输出文件名会随之变化，注意不要引用旧的 JSON。

## Step4 用热点专家裁剪模型

```bash
python scripts/expert_pruned.py \
  --top-k "$KEEP_EXPERTS" \
  --json-path "$RUN_DIR/top${KEEP_EXPERTS}_experts.json" \
  --model-dir "$MODEL_BASE"
```

- 输出目录为 `{MODEL_BASE}-top{K}-pruned`（脚本自动派生）；必须为全新目录，源模型只读。
- 脚本语义：前 3 层（hash/dense 层，`SKIP_LAYERS={0,1,2}`）保留全部原专家，其余层裁到 top-K，shared experts 原样保留，`config.json` 的 `n_routed_experts` 改写为 K。换模型时核对该语义与目标结构是否匹配（详见 [references/routing-and-pruning.md](references/routing-and-pruning.md)），尤其要求加载器支持"hash 层全量 + 其余层 top-K"的异构专家数。
- 裁剪完成后用 `scripts/vllm_server_pruned.sh`（`MODEL_PATH` 指向裁剪目录）启动验证，确认能加载、专家数与路由 ID 范围正确、能正常生成。

## 可选：aisbench 精度验证与专家数优化

aisbench 安装与配置见 [references/commands.md](references/commands.md)；评测口径见 [references/validation.md](references/validation.md)。用法：

1. 分别对原模型与裁剪模型跑同一数据集、同一生成参数，比较精度（默认校准集口径；对外结论需独立保留集）。
2. 优化保留专家数：对多个 `KEEP_EXPERTS`（如 16/32/48/64）重复 Step3→Step4→评测，选精度-规模平衡点；重跑统计会生成新的 `topN_experts.json`，不要混用。
3. 若采集时服务是 w4a8 量化权重、部署验证用原始精度（或反之），先读 README「w4a8 与原始权重部署推理差异」——量化与裁剪两类差异需分别归因，不要混在一次对比里下结论。

## 注意事项与参考

- **w4a8 差异**：`scripts/conver_model_flash.py`（依赖同目录 `scripts/mx_quantize.py`）支持 FP8→BF16 及再量化为 `w8a8-int/w8a8-mx/w4a8-mx/w4a4-mx`；量化口径带来的部署差异与对以小验大结论的影响，见 README「w4a8 与原始权重部署推理差异」。
- 执行纪律：所有路径在目标环境核实后再执行（容器内 vs 主机路径分开记录）；不覆盖源权重与他人目录；保存校准样本 ID、命令、日志与 manifest 到 `RUN_DIR`，标注已实测与未验证项。
- 详细经验：环境与路径 [references/environment.md](references/environment.md)；采集与裁剪核对 [references/routing-and-pruning.md](references/routing-and-pruning.md)；命令手册 [references/commands.md](references/commands.md)；验收 [references/validation.md](references/validation.md)；脚本来源 [references/source-notes.md](references/source-notes.md)。
- 小模型通过只证明已覆盖路径，不能替代大模型的规模、通信与内存验证；带回大模型前读 [references/validation.md](references/validation.md) 的「带回大模型」一节。
