# 脚本来源与适配说明

本 skill 整理自《以小验大实践》：在 Ascend 上运行原始 MoE 模型，采集指定数据集的 token 级专家路由，汇总逐层高频专家，裁剪 routed experts，并对比评测。历史版本中的个人目录、镜像位置和并行参数只是案例，不构成本 skill 的固定运行环境。

当前版本已把全部脚本内置到 `scripts/`，并把采集入口统一为 vLLM 原生 `--enable-return-routed-experts`（响应附带 routed experts + 客户端落盘），**不再依赖任何 vLLM/vLLM-Ascend 源码补丁**。历史补丁材料（内网 codehub 的 ba1d0ca 设备侧落盘、v0.30 图模式引擎侧聚合）已废弃，仅在仓库 git 历史中可查。

## 内置脚本清单

| 脚本 | 用途 | 适配要点 |
|---|---|---|
| `scripts/vllm_server_base.sh` | Step1 启动原模型服务（环境变量参数化） | **必须含 `--enable-return-routed-experts`**；Ascend 环境变量按镜像核对；改编自历史 `vllm_server_V4.sh` |
| `scripts/vllm_server_pruned.sh` | Step4 启动裁剪模型 | 不开回传；验证稳定后可尝试 `FULL_DECODE_ONLY` 图模式；改编自 `vllm_server_V4_prune.sh` |
| `scripts/collect_routed_experts.py` | Step2 数据集访问 + routed experts 落盘 | 本 skill 新增；纯标准库；`(T, L, K)` 形状/层号映射需按首请求核对 |
| `scripts/stat_moe.py` | Step3 逐层频次统计 → `top{K}_experts.json` | 输出写在 `--log-dir` 的上级目录；输入为 `layers.N.txt`（每行一个 token 的逗号分隔专家 ID） |
| `scripts/expert_pruned.py` | Step4 按热点专家裁剪权重 | `SKIP_LAYERS={0,1,2}`（hash 层全保留）；输出目录自动派生 `-top{K}-pruned`；兼容 ffn/mlp 两套命名 |
| `scripts/conver_model_flash.py` | 可选 FP8→BF16 / 再量化（w8a8/w4a8-mx/w4a4-mx） | 原文拼写如此；**必须与 `mx_quantize.py` 同目录**；ignore/quant_layers 按 DeepSeek-V4-Flash 结构写死 |
| `scripts/mx_quantize.py` | 上一步依赖（MX FP8/FP4 量化原语） | 供 `conver_model_flash.py` import，不单独运行 |
| `scripts/rl/*.sh` | 可选 RL 下游（verl/GRPO/Ray/docker） | 按历史环境写死路径/卡数，使用前逐项修改 |

## 数据格式约定（脚本间契约）

```
collect_routed_experts.py            stat_moe.py                 expert_pruned.py
layers.N.txt ──────────────────────> top{K}_experts.json ───────> {model}-top{K}-pruned/
(每行=一个token的K个专家ID,逗号分隔)   {"layers.N": {"topk_ids":     (SKIP_LAYERS 全保留,
 文件名 N 需匹配 "layers.<数字>"       "topk_counts": [...]}}       其余层裁到 top-K)
```

## 外部参考

- [AIS-Bench 官方安装说明](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/get_started/install.html)与[源码](https://github.com/AISBench/benchmark)：可选评测器，实际使用以选定 commit、requirements 和 CLI 帮助为准。
- gitee 镜像 `https://gitee.com/aisbench/benchmark.git` 与华为云 PyPI 源按可达性替换。
- DeepSeek-V4-Flash 的 chat template 从匹配的模型版本获取（如 ModelScope 模型仓库），不能套用到其他模型。

不要直接复制历史材料中的包版本修复命令、全局目录清理命令或个人绝对路径。它们描述历史操作，不是可移植前提。
