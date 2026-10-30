# vLLM-Ascend 热点专家采集补丁分析（commit ba1d0ca）

## 来源信息

| 项 | 值 |
|---|---|
| 仓库 | `https://codehub-g.huawei.com/DeepSeekV4_expert_pruned/vllm-ascend`（内网，需 SSO） |
| Commit | `ba1d0ca445b9ca59bf906ee23b4469f83499742e` |
| 父提交 | `fdcdb88`（镜像原改动） |
| 作者 | z00841464 <zhangwangze@huawei.com> |
| 日期 | 2026-05-12 13:28:53 +0800 |
| 主题 | 以小验大适配，支持 moe 异构专家 |
| 规模 | 5 个文件，+131 / -36 行 |

## 补丁目的

让 DeepseekV4 在 vLLM-Ascend 上支持 **异构 MoE 层**（不同层专家数不同），服务"以小验大"：
- 用少量真实专家的"小模型"验证大规模模型的行为（路由/热点统计、权重加载、通信）；
- 提供基于环境变量 `TOPK_SAVE_PATH` 的 **topk_ids 落盘采集**，即热点专家统计入口。

## 逐文件改动分析

### 1. `vllm_ascend/models/deepseek_v4.py`（+97，核心）
- **每层独立 config**：新增 `make_custom_layer(layer_prefix)`，对每层 `deepcopy(hf_config)`；前 `hash_layers` 层（字段 `n_hash_layers`，缺省回退 `num_hash_layers=3`）强制：
  - `n_routed_experts = 256`
  - `num_experts_per_tok = 6`（同步 `n_activated_experts = 6`）
  然后用该层专属 config 构造 `DeepseekV2DecoderLayer`，替代原先所有层共用一个 config 的 lambda。
- **专家映射按层计算**：`get_expert_mapping` 遍历各层取 `max(n_routed_experts)`；`n_shared_experts` 仅在 `mix_placement` 时计入。
- **权重加载按层路由**：
  - 删除全局 `expert_params_mapping`，改为按层缓存 `_get_layer_expert_mapping(layer_id)`，`num_redundant_experts` 裁剪后置 0；
  - checkpoint 中超出当前 `num_hidden_layers` 的层直接跳过（小模型加载大裁剪 checkpoint 的关键）；
  - `mlp.shared_experts` 合并命名时使用该层实际 `n_routed_experts`（原为全局值）；
  - 含 `mlp.experts.` 的权重按层选择 mapping，其余置空。

### 2. `vllm_ascend/ops/fused_moe/experts_selector.py`（+15，采集入口）
- `select_experts` 新增 `layer_name` 参数；
- 环境变量 `TOPK_SAVE_PATH` 非空时，创建目录并把该层 `topk_ids` 追加写入
  `{TOPK_SAVE_PATH}/{layer_name}.txt`（`np.savetxt`，逗号分隔）；
  失败抛 `RuntimeError`。**这就是热点专家/路由统计的数据出口。**

### 3. `vllm_ascend/ops/fused_moe/fused_moe.py`（+29，通信按层切换）
- `apply` / `dummy_run` 调用 `select_experts` 时透传 `layer_name`；
- `AscendFusedMoE.__init__` 中 `setup_moe_comm_method(moe_config, layer_name)`；
- `forward` 时按 `layer_name` 从缓存取当前层专家数对应的 `moe_comm_method` 并写回
  `forward_context`；缓存未命中则懒创建——异构层专家数不同，通信实例必须每层独立。

### 4. `vllm_ascend/ops/fused_moe/moe_comm_method.py`（+24，缓存键改造）
- 全局缓存 `_MoECommMethods` 的 key 从 `MoECommType` 改为 `(MoECommType, layer_name)` 二元组；
- `get_moe_comm_method(type, layer_name)`：`layer_name=None` 时返回 None；
- `setup_moe_comm_method(moe_config, layer_name)` 为四种通信实现（ALLTOALL/ALLGATHER/MC2/FUSED_MC2）按层各建实例。

### 5. `vllm_ascend/ascend_forward_context.py`（2）
- 适配新签名：`get_moe_comm_method(moe_comm_type, None)`（占位，真正实例由每层 forward 覆盖）。

## 使用方式（采集热点专家）

```bash
export TOPK_SAVE_PATH=/path/to/topk_out   # 非空即开启
# 启动 vllm serve（打上本补丁），每层生成一个 <layer_name>.txt，逗号分隔的 topk_ids
```

离线统计可配合仓库根目录的 `stat_moe.py` 使用。

## 移植注意事项

1. `n_routed_experts=256 / topk=6` 是为特定"小模型"**硬编码**，移植时应改为从目标模型 config/CLI 注入；
2. `hash_layers` 字段名在不同 config 中可能是 `n_hash_layers` 或 `num_hash_layers`，需与目标 hf_config 核对；
3. 补丁基于内部 `DeepseekV4MoE / DeepseekV2DecoderLayer` 结构，需 diff 目标 vLLM-Ascend 版本后再套用；
4. `get_expert_mapping` 的 `n_shared` 分支有 `mix_placement` 与 `rocm_aiter_moe_shared_expert_enabled` 两套条件，别混用；
5. `TOPK_SAVE_PATH` 落盘是同步追加写，长序列高并发场景注意 IO 开销与文件体积。

## 本目录文件清单

| 文件 | 说明 |
|---|---|
| `analysis.md` | 本分析报告 |
| `ba1d0ca.patch` | 原始 git patch（`git show ba1d0ca`） |
| `files/vllm_ascend/...` | 补丁后 5 个变更文件的完整快照 |
