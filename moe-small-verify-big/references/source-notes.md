# 经验来源与实现线索

本技能整理自用户提供的《以小验大实践》：在 Ascend 上运行原始 MoE 模型，采集指定数据集的 token 级专家路由，汇总逐层高频专家，裁剪 routed experts，并对比评测。文中的个人目录、镜像位置和并行参数不构成本技能的固定运行环境。

原文的操作命令已整理到 [commands.md](commands.md)：保留转换、安装、环境变量、服务、统计、裁剪、评测的实际接口，个人目录改为本轮变量；版本修复和清缓存做法以历史记录保留。

## 补丁与脚本下载来源

- [vLLM-Ascend 专家采集适配提交](https://codehub-g.huawei.com/DeepSeekV4_expert_pruned/vllm-ascend/files/commit/ba1d0ca445b9ca59bf906ee23b4469f83499742e?ref=ba1d0ca445b9ca59bf906ee23b4469f83499742e)
- [project/scripts 裁剪工程](https://codehub-g.huawei.com/DeepSeekV4_expert_pruned/project/files?ref=master&filePath=scripts)

用户明确说明这两处可以获取补丁和配套脚本，应优先复用其中与当前模型/版本匹配的实现。创建本技能时未验证远端内容，也未将其实现打包进技能。使用时从当前授权环境访问，固定实际 commit 并审阅源码；不可访问时寻找用户已有本地 checkout/补丁，必要时索取所需源码。

| 原文脚本线索 | 用途及适配点 |
|---|---|
| `scripts/conver_model_flash.py` | 原文拼写如此；检查是否存在及支持的量化格式、解量化规则、输出 dtype |
| `scripts/sh/vllm_server_V4.sh` | 服务示例；适配架构、设备编号、并行与序列预算 |
| `set_rl_env_beta.sh` | 环境示例；重新解析 CANN、自定义算子和源码路径 |
| `scripts/stat_moe.py` | token 路由统计；核对日志 schema、rank 去重、有效 token 过滤 |
| `scripts/expert_pruned.py` | 专家裁剪；核对 hash/shared/MTP、packed 布局与 router 映射 |
| `benchmark/.../vllm_api_general_V4.py` | 自定义评测配置；不能假定标准 AIS-Bench 包包含 |

## 外部参考

- [AIS-Bench 官方安装说明](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/get_started/install.html)与[源码](https://github.com/AISBench/benchmark)：创建技能时已查阅；实际使用仍以选定 commit、requirements 和 CLI 帮助为准。
- 原材料给出的 [DeepSeek-V4 模板样例](https://modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp/file/view/master/chat_template.jinja?status=0)：仅作为寻找该模型模板的线索；创建技能时未验证其内容，与其他模型不保证兼容。

不要直接复制原文中的包版本修复命令、全局目录清理命令或个人绝对路径。它们描述历史操作，不是可移植前提。
