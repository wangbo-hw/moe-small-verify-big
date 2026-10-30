# moe-small-verify-big — MoE 以小验大 Skill

把一个可运行的大 MoE 模型变成保留目标计算路径的小模型，用于低成本验证路由、热点专家、权重加载与训练/推理行为。

流程四步：

```
Step1  启动 vLLM 服务（必须带 --enable-return-routed-experts）
          │
Step2  用数据集访问服务，从响应中采集 routed experts
          │        scripts/collect_routed_experts.py
          ▼
       run_dir/topk_ids_by_layer/layers.N.txt
          │
Step3  统计逐层热点专家
          │        scripts/stat_moe.py
          ▼
       run_dir/top64_experts.json
          │
Step4  按热点专家裁剪模型
          │        scripts/expert_pruned.py
          ▼
       {model}-top64-pruned/   ──(可选)──>  aisbench 精度验证、扫描最优保留专家数
```

全部脚本随本仓库（=skill）分发，**不需要再打补丁**：`--enable-return-routed-experts` 是 vLLM/vLLM-Ascend 原生参数，响应中直接附带 routed experts。

## 目录结构

```
├── SKILL.md                          # skill 主入口（agent 按此文执行）
├── README.md                         # 本文件
├── agents/openai.yaml                # agent 展示元数据
├── scripts/
│   ├── vllm_server_base.sh           # Step1 启动原模型（参数化，含 --enable-return-routed-experts）
│   ├── vllm_server_pruned.sh         # Step4 启动裁剪模型（参数化）
│   ├── collect_routed_experts.py     # Step2 数据集访问 + routed experts 落盘
│   ├── stat_moe.py                   # Step3 逐层热点专家统计
│   ├── expert_pruned.py              # Step4 按热点专家裁剪权重
│   ├── conver_model_flash.py         # 可选: FP8→BF16 / 再量化 w8a8 / w4a8-mx 等（须与 mx_quantize.py 同目录）
│   ├── mx_quantize.py                # 上一步的依赖库
│   └── rl/                           # 可选: 以小验大→RL 训练调试的下游脚本（verl/Ray，DeepSeek-V4 经验）
│       ├── run_rl.sh                 #   用裁剪模型(top64-pruned)跑 GRPO RL
│       ├── run_ray.sh / test_1node-beta2.sh / docker_rl_run_beta.sh
└── references/                       # agent 执行时阅读的详细经验
    ├── commands.md                   #   命令手册（环境/安装/评测）
    ├── environment.md                #   环境、路径与 dtype 适配
    ├── routing-and-pruning.md        #   采集与裁剪的结构核对清单
    ├── validation.md                 #   验收口径（推理/训练/一致性）
    └── source-notes.md               #   脚本来源与适配说明
```

## 环境要求

- vLLM / vLLM-Ascend 版本支持 `--enable-return-routed-experts`，且该模型的 MoE 实现会向响应回传 routed experts（DeepSeek-V4 系已验证）。启动前先用单请求确认响应含 `routed_experts` 字段。
- Python ≥3.8；`safetensors`、`torch`（Step4 裁剪与可选转换需要；Step2 采集脚本仅用标准库）。
- Ascend 环境：按镜像实际参数核对启动脚本中的 `ASCEND_RT_VISIBLE_DEVICES`、`ACL_OP_INIT_MODE` 等（历史经验值，不保证通用）。
- 可选 aisbench：安装方法见 [references/commands.md](references/commands.md)。

## 快速上手（命令级）

```bash
# 0) 变量
MODEL_BASE=/path/to/DeepSeek-V4-Flash-bf16     # 采集用的原模型
KEEP=64                                        # 保留专家数
RUN_DIR=/path/to/run_$(date +%m%d_%H%M)        # 本轮输出目录（全新）
mkdir -p "$RUN_DIR"

# 1) 启动原模型服务（必须 --enable-return-routed-experts，脚本已内置）
MODEL_PATH="$MODEL_BASE" SERVED_NAME=verify-base PORT=8006 \
  bash scripts/vllm_server_base.sh 2>&1 | tee "$RUN_DIR/base-server.log"
# 另开终端: curl --fail http://127.0.0.1:8006/v1/models

# 2) 数据集采集 routed experts（先 5 条核对形状，再全量）
python scripts/collect_routed_experts.py \
  --api-base http://127.0.0.1:8006 --model verify-base \
  --dataset data/countbenchqa.csv --column Question \
  --model-config "$MODEL_BASE/config.json" \
  --output-dir "$RUN_DIR/topk_ids_by_layer" \
  --limit 5 --raw-dir "$RUN_DIR/raw"
# 核对 manifest 中 T ≈ prompt+completion-1 后，去掉 --limit 全量跑

# 3) 统计热点专家
python scripts/stat_moe.py --top-k "$KEEP" --log-dir "$RUN_DIR/topk_ids_by_layer"
# → $RUN_DIR/top64_experts.json

# 4) 裁剪模型
python scripts/expert_pruned.py \
  --top-k "$KEEP" \
  --json-path "$RUN_DIR/top${KEEP}_experts.json" \
  --model-dir "$MODEL_BASE"
# → $MODEL_BASE-top64-pruned/

# 4b) 启动裁剪模型验证
MODEL_PATH="$MODEL_BASE-top${KEEP}-pruned" SERVED_NAME=verify-pruned PORT=8007 \
  bash scripts/vllm_server_pruned.sh 2>&1 | tee "$RUN_DIR/pruned-server.log"
```

数据集支持三种格式：CSV（`--column` 指定问题列名或 0-based 序号）、JSONL/JSON（`--field` 指定字段）、纯文本（每行一条）。采集参数（层数 L、每 token 专家数 K、层号映射）见 `collect_routed_experts.py` 文件头说明与 [SKILL.md](SKILL.md) Step2。

## 怎么写 prompt 调用本 skill

安装本 skill 后，把下面模板补全成你的 prompt 即可；agent 会按 [SKILL.md](SKILL.md) 执行，细节读 `references/`。

**完整流程：**

> 使用 moe-small-verify-big 技能，在 <服务器/容器说明> 上对模型 <模型路径> 执行以小验大：
> 1. 用技能自带 scripts/vllm_server_base.sh 启动服务（端口 8006，TP<8> DP<1>，务必带 --enable-return-routed-experts）；
> 2. 用数据集 <csv/jsonl 路径，问题列名/字段> 采集 routed experts，输出到 <RUN_DIR>；
> 3. 统计 top<64> 热点专家；
> 4. 用热点专家裁剪模型，输出目录 <裁剪模型路径>，并用 scripts/vllm_server_pruned.sh 启动验证。

**只跑某一步**（各步可独立调用）：

> 使用 moe-small-verify-big：服务已在 http://127.0.0.1:8006（模型 xxx），用数据集 <路径>（列 Question）采集 routed experts 到 <RUN_DIR>/topk_ids_by_layer，并统计 top<64> 热点专家。

> 使用 moe-small-verify-big：热点统计 JSON 在 <RUN_DIR>/top64_experts.json，对模型 <路径> 执行裁剪并启动裁剪服务验证。

**加可选的 aisbench 精度验证 / 专家数优化：**

> ……裁剪完成后用 aisbench 对原模型和裁剪模型各跑一遍 <数据集配置，如 gsm8k_gen>（500 条，--num-warmups=0），比较精度；再对 top<32/48/64> 各裁剪一版做对比，给出精度-规模平衡建议。

Prompt 里给全「模型路径、服务参数、数据集、KEEP、输出目录」五要素，agent 基本不需要追问。

## w4a8 与原始模型权重部署推理的差异（重要）

`scripts/conver_model_flash.py` 可把 FP8 权重转 BF16，或再量化为 `w8a8-int / w8a8-mx / w4a8-mx / w4a4-mx`（MoE 专家 4bit、shared experts/attention/MTP 8bit、激活 8bit per-token 动态量化、KV/Li cache 量化；router gate 不量化）。**以小验大各环节若混用 w4a8 与原始（BF16/FP8）权重，会引入量化差异，解读结论时注意：**

1. **数值精度损失**：4bit 权重相对 BF16/FP8 有明显量化误差，输出分布整体偏移。对"热点专家统计"而言，头部热点通常稳定，但 **top-K 边界附近的专家可能互换**，KEEP 越小越敏感。
2. **路由漂移**：gate 本身不量化，但专家输出变了 → 下层输入分布变化 → 逐层路由频次漂移。在 A 权重上采集的热点集合，换 B 权重部署后不再严格最优。
3. **离线模拟 ≠ 部署 kernel**：`conver_model_flash.py` 的量化是离线 round-to-nearest/minmax 模拟；实际部署的反量化/MC2/fused kernel（clip 策略、scale 粒度）数值实现不同，精度会有额外出入。
4. **差异归因**：裁剪与量化是两类独立差异。推荐至少保持「采集-部署同 dtype 口径」（如都用 BF16），需要 w4a8 结论时单独跑 (原模型w4a8 vs 裁剪w4a8) 对照，不要把量化损失记在裁剪头上。
5. **ignore 名单是模型特定的**：脚本内 `generate_ignore_item/generate_quant_layers` 按 DeepSeek-V4-Flash 结构（attn `wq_a/wkv/indexer/compressor`、compress_ratios 等）写死，换模型必须核对 ignore/quant_layers 覆盖，否则关键模块被误量化。
6. **输入预期**：脚本按 `element_size()==1` 识别 FP8 张量（含 mxfp4 int8 打包），输入目录需含 `model.safetensors.index.json` 与 `config.json`（需 `compress_ratios` 字段）。裁剪后模型（config 已是 `n_routed_experts=K`）可直接再量化。
7. **收益与代价**：w4a8 主要省 MoE 权重显存/带宽；精度损失要靠 aisbench 等实测确认，不能引用离线压缩率推断。

## 可选：aisbench 精度验证与专家数优化

- 安装与独立 venv 配置：[references/commands.md](references/commands.md)「可选：安装 AIS-Bench」。
- 自定义 API 配置（endpoint/served name/生成参数）与数据集配置的坑（配置变量名必须以 `_datasets` 结尾，否则静默 0 tasks）：同文件「用 AIS-Bench 驱动请求」。
- 用法：同一数据集、同一生成参数分别评测原模型与各裁剪模型；对多个 KEEP（16/32/48/64…）重复 Step3→Step4→评测，选精度-规模平衡点。用于选专家的数据集分数只能算**校准集效果**，对外结论用独立保留集。

## RL 训练调试（可选下游）

`scripts/rl/` 保留 DeepSeek-V4 的 verl/Ray RL 环境脚本，`run_rl.sh` 直接以 `-top64-pruned` 裁剪模型为训练底模——即"以小验大"在 RL 场景的下游用法。路径、卡数、数据集均按历史环境写死，使用前需按当前环境修改；训练验收要求（真实反向、训推一致性）见 [references/validation.md](references/validation.md)。

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| 响应里没有 `routed_experts` | 服务没带 `--enable-return-routed-experts`，或该版本模型实现不回传；用 `--raw-dir` 存响应排查 |
| 采集报"元素数不是 L×K 整数倍" | L/K 推断不符：核对 `--num-moe-layers/--experts-per-token`，base64 数组 dtype 不对时换 `--dtype` |
| `stat_moe.py` 层数不全 | 有请求失败或 warmup 混入：看 `_collect_manifest.json` 的 ok/failed 与失败详情 |
| 裁剪模型加载失败（专家数不符） | 加载器需支持"hash 层全量 + 其余层 top-K"异构专家数；核对 config、index 与逐层张量数 |
| aisbench 显示 `Partitioned into 0 tasks` | 数据集配置变量名未以 `_datasets` 结尾（静默丢弃） |
| 图模式/显存问题 | 采集验证阶段先用 `--enforce-eager`；图模式额外占捕获工作区，必要时调低 `--gpu-memory-utilization` |

## 历史版本说明

早期版本依赖 vLLM-Ascend 补丁（ba1d0ca 设备侧落盘、v0.30 图模式引擎侧聚合）采集路由，现已废弃：流程改为原生 `--enable-return-routed-experts` + 客户端采集，见 git 历史获取旧补丁材料。
