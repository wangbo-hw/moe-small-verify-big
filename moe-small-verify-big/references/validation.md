# 推理、训练与大模型回归验收

原模型与裁剪模型的完整服务及评测命令见 [commands.md](commands.md)，本页定义结果如何比较和验收。

## 最小可比实验

保存原模型和裁剪模型的权重身份、代码/补丁哈希、dtype、并行布局、tokenizer/template、固定样本 ID、生成参数和评测器版本。先各跑少量相同请求核对接口与解析，再运行约定数量的评测。并行布局变化不可避免时记录它，不把数值差异全部归因于裁剪。

用于选择高频专家的数据是校准数据。如果再用同一批题评分，只能称为校准集效果；对外质量结论使用独立保留集。训练任务应明确 train/dev/test，避免用 test 的路由统计或分数反复选择裁剪方案。已有同集实验可以保留，但标注用途与局限。

关闭采集须核对实际服务进程：重新启动服务时不继承 `TOPK_SAVE_PATH`，或验证运行时开关已生效。性能测试应在关闭 dump 的条件下进行。

## 使用 AIS-Bench 时

使用隔离评测环境中的 CLI，先检查该版本 `ais_bench -h`，再核对 model/dataset 配置搜索结果。历史命令中的 `--dataset`、`--num-prompt` 可能是旧参数或前缀缩写，不作为跨版本接口保证。

以下仅在实际帮助确认支持这些参数时使用；变量来自本轮运行映射：

```bash
"$AIS_CLI" --models "$AIS_MODEL_CONFIG" --datasets "$AIS_DATASET_CONFIG" --search
"$AIS_CLI" --models "$AIS_MODEL_CONFIG" --datasets "$AIS_DATASET_CONFIG" \
  --num-prompts "$EVAL_COUNT" --num-warmups 0 --work-dir "$EVAL_OUTPUT"
```

`vllm_api_general_V4`/`vllm_api_general_V4_prune` 是原经验中的自定义配置名，不保证官方安装自带。若缺失，检查已有配置源码并适配实际 API；确认 endpoint、served model name、chat/completion 类型、模板、max tokens、stop、重试及答案解析。不能只替换成任意可搜索到的模型配置。

以 GSM8K 为例，记录实际评测条数、选题 ID、正确率、请求失败/截断数，以及最终答案提取规则；不要把服务失败样本悄悄排除。原材料中 500 条、92.60 的数字仅是其一次报告，不能作为新模型的预期结果或验收线。

## 推理验收

先验证模型正常加载、每层专家数量与路由 ID 范围、返回内容和评分解析。比较原模型与裁剪模型的正确率/质量损失、权重大小、峰值内存、启动时间和吞吐。阈值由任务约定；未约定时报告实测与权衡，不虚构“通过”结论。

裁剪改变了专家池和路由分布，原模型与裁剪模型不要求逐 token 完全一致。不要把二者输出相似度与同一裁剪模型的 actor/rollout 一致性混为一项。

## 训练与训推一致性

训练模式必须覆盖实际 autograd 路径；仅有冻结前向结果不能验收训练补丁。对改动的模块检查一般输入下的梯度数值和参考实现，而不只看“梯度非空”。attention sink 的分母、Q/K 耦合、投影权重及 checkpoint 重计算等是否需要专项检查，依据目标实现和改动决定。

小 batch 真实更新中记录 loss、grad norm、非有限值、optimizer step 和参数变化，确认使用的是训练实现。分别记录权重/计算、梯度累加/通信、optimizer 状态 dtype；不要从 optimizer dtype 推断冻结前向误差。

训推一致性使用同一 checkpoint 的 rollout token 及对应 logprob，与 actor 更新前在同一 token 序列上的重算比较。对齐 prompt/response 偏移、mask、padding/EOS、position、温度和分布处理；若 rollout 返回采样后分布而 actor 使用原始分布，先明确统一比较口径。路由 replay 开关也应纳入实验身份。

分别报告概率 Pearson、logprob Pearson、MAE/max abs、有效 token 数、非有限值和方差。相关系数高不代表绝对误差小；常量序列或样本过少时不能据此宣告通过。验收阈值沿用用户要求，不把某次 0.99/0.999 当作所有项目通用阈值。

更新后重新同步权重，再按新的权重版本进行下一次比较。RL reward 趋势和固定 dev/test 表现另行验收，不能用一致性指标代替学习效果。

## 带回大模型

列出小模型已覆盖的计算路径和未覆盖项。大模型需要重新检查专家数量/层数、算子选择、TP/EP/PP/CP 通信、负载分布、权重同步临时内存和长序列。先运行可界定的小请求/短步数回归，确认相同补丁确实执行，再扩大规模。记录失败的首个阶段并据此定位，不能因小模型成功就跳过大模型验证。
