# 推理、训练与大模型回归验收

原模型与裁剪模型的完整服务及评测命令见 [commands.md](commands.md)，本页定义结果如何比较和验收。

## 最小可比实验

保存原模型和裁剪模型的权重身份、代码/补丁哈希、dtype、并行布局、tokenizer/template、固定样本 ID、生成参数和评测器版本。先各跑少量相同请求核对接口与解析，再运行约定数量的评测。并行布局变化不可避免时记录它，不把数值差异全部归因于裁剪。

用于选择高频专家的数据是校准数据。如果再用同一批题评分，只能称为校准集效果；对外质量结论使用独立保留集。训练任务应明确 train/dev/test，避免用 test 的路由统计或分数反复选择裁剪方案。已有同集实验可以保留，但标注用途与局限。

采集是客户端行为，服务端没有落盘开关需要清理；但**性能测试**应在未开启 `--enable-return-routed-experts` 的服务上进行（回传大数组有额外开销，会污染性能数字）。

## 使用 AIS-Bench 时

使用隔离评测环境中的 CLI，先检查该版本 `ais_bench -h`，再核对 model/dataset 配置搜索结果。历史命令中的 `--dataset`、`--num-prompt` 可能是旧参数或前缀缩写，不作为跨版本接口保证。

以下仅在实际帮助确认支持这些参数时使用；变量来自本轮运行映射：

```bash
"$AIS_CLI" --models "$AIS_MODEL_CONFIG" --datasets "$AIS_DATASET_CONFIG" --search
"$AIS_CLI" --models "$AIS_MODEL_CONFIG" --datasets "$AIS_DATASET_CONFIG" \
  --num-prompts "$EVAL_COUNT" --num-warmups 0 --work-dir "$EVAL_OUTPUT"
```

`vllm_api_general_V4`/`vllm_api_general_V4_prune` 是历史经验中的自定义配置名，不保证官方安装自带。若缺失，检查已有配置源码并适配实际 API；确认 endpoint、served model name、chat/completion 类型、模板、max tokens、stop、重试及答案解析。数据集配置变量名必须以 `_datasets` 结尾，否则 aisbench 静默丢弃（`Partitioned into 0 tasks`）。

以 GSM8K 为例，记录实际评测条数、选题 ID、正确率、请求失败/截断数，以及最终答案提取规则；不要把服务失败样本悄悄排除。历史材料中的 500 条、92.60 只是一次报告，不能作为新模型的预期结果或验收线。

## 专家数优化（可选）

对多个 `KEEP_EXPERTS`（如 16/32/48/64）重复"统计→裁剪→评测"，比较精度与显存/吞吐后选择平衡点：

- 每轮使用独立的 `topN_experts.json`、裁剪输出目录与评测 work-dir，不要混用。
- 扫描用的数据集分数属于校准集效果；最终选定后用独立保留集复核一次。
- 评估时报出每档的"热点覆盖率"（`topk_counts` 总和占全部计数比例）与精度变化，便于判断继续减专家的收益边界。

## 推理验收

先验证模型正常加载、每层专家数量与路由 ID 范围、返回内容和评分解析。比较原模型与裁剪模型的正确率/质量损失、权重大小、峰值内存、启动时间和吞吐。阈值由任务约定；未约定时报告实测与权衡，不虚构"通过"结论。

裁剪改变了专家池和路由分布，原模型与裁剪模型不要求逐 token 完全一致。采集权重与部署权重 dtype 不同（如 w4a8 对比 BF16）时，量化与裁剪是两类差异，需分口径对照归因（见 README「w4a8 与原始模型权重部署推理的差异」）。

## 训练与训推一致性

训练模式必须覆盖实际 autograd 路径；仅有冻结前向结果不能验收训练补丁。对改动的模块检查一般输入下的梯度数值和参考实现，而不只看"梯度非空"。

小 batch 真实更新中记录 loss、grad norm、非有限值、optimizer step 和参数变化，确认使用的是训练实现。分别记录权重/计算、梯度累加/通信、optimizer 状态 dtype。

训推一致性使用同一 checkpoint 的 rollout token 及对应 logprob，与 actor 更新前在同一 token 序列上的重算比较。对齐 prompt/response 偏移、mask、padding/EOS、position、温度和分布处理。分别报告概率 Pearson、logprob Pearson、MAE/max abs、有效 token 数、非有限值和方差。相关系数高不代表绝对误差小；常量序列或样本过少时不能据此宣告通过。

更新后重新同步权重，再按新的权重版本进行下一次比较。RL reward 趋势和固定 dev/test 表现另行验收，不能用一致性指标代替学习效果。

## 带回大模型

列出小模型已覆盖的计算路径和未覆盖项。大模型需要重新检查专家数量/层数、算子选择、TP/EP/PP/CP 通信、负载分布、权重同步临时内存和长序列。先运行可界定的小请求/短步数回归，确认相同改动确实执行，再扩大规模。记录失败的首个阶段并据此定位，不能因小模型成功就跳过大模型验证。
