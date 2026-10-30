# 专家采集与结构裁剪

原模型启动、采集、`stat_moe.py` 和 `expert_pruned.py` 的参数化命令见 [commands.md](commands.md)，执行前用本页核对结构与映射。

## 确认逐层结构

结合模型配置、真实实现及权重 header/索引建立逐层表：层号、路由类型、routed 数量、shared 数量/宽度、每 token 激活数、router/校正 bias、hash 表、MTP 和量化布局。字段可能位于嵌套 text config，不能只检查配置顶层。

`KEEP_EXPERTS=64` 表示每个目标层保留 64 个 routed experts；`experts_per_token=6/8` 表示每个 token 的激活数，两者不是同一个 top-k。仅缩减专家池时保留原激活数和路由算法，并检查 `KEEP_EXPERTS >= experts_per_token`。分组路由还需检查组数、每组候选数和分组 top-k 的约束；不满足时重新制定裁剪方案，不能悄悄改算法凑形状。

hash 路由层默认保留全部原专家及原 hash 映射（`expert_pruned.py` 的 `SKIP_LAYERS={0,1,2}` 即此语义，换模型需核对）。若某些层保留原数量、其他层裁剪，loader 必须支持逐层不同专家数；仅修改全局 `n_routed_experts` 可能造成保留层也被错误实例化。未支持这种结构时，先适配并验证 loader，或明确报告该方案当前无法运行。

shared expert 默认原样保留。注意 shared 与 routed 可能复用 MLP 构造代码但 clamp/激活配置不同；禁止因为共用代码而把 routed 配置强加到 shared。MTP、embedding、attention、norm、LM head 的处理逐项记录，不按名称相似推定能删除。

## 通过响应通道观测路由

采集入口是 vLLM 原生 `--enable-return-routed-experts`：服务启动后响应中附带 routed experts，`scripts/collect_routed_experts.py` 逐请求解析并落盘。不需要也不允许再用改 vllm/vllm-ascend 源码打补丁的方式采集。

执行前核对：

1. 单请求确认响应确实含 `routed_experts` 字段（`--raw-dir` 保存原始响应排查格式）。
2. 数组形状 `(T, L, K)` 的语义：`T` 与 `prompt+completion` token 数的关系（约 `prompt+completion-1`）、`L` 是 MoE 层数、`K` 是每 token 激活数。首请求必须人工核对后再放量。
3. 响应中的专家 ID 是否为最终 dispatch 的逻辑专家 ID；若该版本回传的是经 EP/log2phy/负载均衡重映射的物理 ID，必须有同次运行的反向映射，不能把物理 ID 直接当全局专家编号。
4. TP/EP 下 token 分片与 rank 聚合由引擎原生通道负责（服务端已聚合为每请求一份数组）；若实测发现各 rank 重复记录，需去重后再统计。
5. 采集是客户端行为：只有采集脚本发送的请求会计入，服务端 warmup/dummy 流量天然不计入，无需清理服务端落盘目录。
6. 采集并发默认 4，可通过 `--concurrency` 调整；大批量采集时确认服务无请求失败/截断（manifest.failures），统计前先看 ok/failed 比例。

原模型不能在当前资源上启动时，先解决原模型采集的资源或后端支持，或选择另一个经用户认可的可运行校准模型并标注范围；不能把随机裁剪模型的频次宣称为原模型频次。

## 汇总频次与生成映射

固定校准数据集、split、样本 ID、提示模板、生成参数和 seed。记录输入 token 与生成 token 的贡献。每层检查有效记录数量、专家 ID 范围、缺层、重复、异常重复 ID，以及计数是否与有效 token×激活数一致（对照 `_collect_manifest.json` 与 `stat_moe.py` 输出）。

`stat_moe.py` 按出现频次选每层 top-K，并保留 `topk_counts` 便于复核边界平局。报告保留选择次数占总选择次数的比例；频率覆盖率不是准确率保证。

保存每层完整频次、选中旧 ID 和明确的 old→new 映射。`expert_pruned.py` 内部按"保留 ID 升序 → 新 ID 0..K-1"重排，专家权重与 router 使用同一映射。

## 写出裁剪权重

`expert_pruned.py` 接口：`--top-k`（保留数）、`--json-path`（统计 JSON）、`--model-dir`（源模型）；输出目录自动派生为 `{model-dir}-top{K}-pruned`，脚本不提供输出目录参数——确认磁盘空间与派生路径后再运行。只读计划应明确：每层 copy/slice/drop 的 tensor、源和目标 shape、切片轴。

确认以下不变量后再使用输出：

- 原模型只读，目标不与源相同或位于源目录内部；已有输出不覆盖。
- 前 `SKIP_LAYERS` 层保留全部原专家；其余层选中 ID 唯一有效，router 权重/bias 与专家 gate/up/down 权重使用相同映射。
- shared experts、tokenizer、配置、模板原样保留；`config.json` 的 `n_routed_experts` 改写为 K，`model.safetensors.index.json` 重写。
- 量化模型的 packed 张量与 scale 不能直接套本脚本（scale 键名不匹配裁剪正则）；需先 BF16 化或扩展脚本处理 scale 切片，逐 tensor 校验后再用。

完整权重较大时分片读取与写出，并预留源+目标+临时文件空间。只有数据与索引校验通过后才能启动服务。保留映射和源文件校验信息，不能把"文件少了"当作裁剪正确的证据。
