# 环境、路径和 dtype 适配

具体环境变量、转换与安装命令见 [commands.md](commands.md)，以下说明用于判断哪些命令适用于本轮环境。

## 获取补丁和脚本

先检查 [SKILL.md 中的两个下载入口](../SKILL.md#补丁与裁剪脚本下载入口)：vLLM-Ascend 固定提交提供采集补丁来源，project/scripts 提供统计、裁剪等配套脚本来源。优先检查已有副本的来源与版本；缺失时获取所需代码并记录 commit/文件哈希，再映射到当前安装路径。网页浏览地址不直接当作 Git 克隆地址，历史部署目录不当作补丁安装目标。

## 先建立实际运行映射

在将使用的容器或虚拟环境内探测，不能用 Windows 编排机的包版本代替目标环境。只检查已授权的节点、容器和相关目录；不因历史工单存在就推定当前仍可用。

建议在 `runtime-map.json` 记录下列字段；不存在的可选组件用 null，并说明不需要的原因：

| 字段 | 确认方法 |
|---|---|
| host/container/image、资源期限 | 当前任务记录和容器 inspect；确认 NPU 占用及归属 |
| runtime_python、cli_executable | `command -v`、`sys.executable`、CLI shebang；排除混用解释器 |
| source_roots、commits、dirty_files | 实际导入路径、Git HEAD/status；记录本地补丁 |
| source_model、output_model | 目标环境真实绝对路径；输出必须是新的独立目录 |
| conversion/statistics/pruning scripts | 在已知项目目录用 `rg --files` 找到，检查实现及参数 |
| CANN env scripts、custom OPP roots | 已安装文件、软链接解析、现有启动配置及实际算子来源 |
| calibration/evaluation data | 文件版本、样本 ID、split 与哈希 |
| run_root、capture_root、eval env/source | 新的运行目录及实际挂载关系；符合用户日志位置要求 |

在目标环境中用已确认的解释器做轻量检查，例如：

```bash
"$RUNTIME_PYTHON" -m pip show vllm vllm-ascend torch torch-npu transformers
"$RUNTIME_PYTHON" -c 'import sys; print(sys.executable); print(sys.version)'
```

需要执行导入探测时先确认模块是否会初始化设备。在 CPU 安全的路径上核对 `__file__`/`inspect.getfile`、模型注册入口和版本分发代码。仅 `pip show` 成功不能证明运行用了目标 checkout。

## 配置环境

从目标镜像已有启动脚本和安装目录中解析 CANN、ATB、自定义算子的实际路径；只 source 当前任务确实需要且已存在的脚本。不要把历史 `cann-9.0.0-beta.2` 或某个用户目录当成默认值。

`ASCEND_CUSTOM_OPP_PATH`、`LD_LIBRARY_PATH` 可能已有多个必要项。查明每项作用后追加/去重，不直接覆盖成材料中的单个目录。若运行依赖 Ray 或其他常驻进程，确认其启动时已继承所需环境；修改当前 shell 不会更新已运行 worker。

`PYTHONPATH` 仅加入选定的源代码根目录，控制顺序并再次检查导入位置。训练需要的 verl、bridge、transformers 路径按目标实现选择；纯推理不必加入训练栈。

设备可见编号、TP/DP/EP、并发和上下文根据硬件及该版本参数含义设置。材料里的 16 个设备编号、DP4/TP4、22528 token、48 并发只是一次实验配置；不能从物理卡数直接推定可见设备数或任意相乘并行维度。

## 权重转换不是默认步骤

先检查 checkpoint 实际张量 dtype、量化 scale/布局、`quantization_config` 和 loader 支持能力。已有可用 BF16/FP32 权重无需重复转换；FP8 或其他量化格式也不代表必须转换。

如果任务确需转换，先核对目标格式、存储容量和算子支持，使用与源格式匹配的解量化实现，写入新目录。量化权重可能需要 scale、block size 和布局转换，单纯 `.to(bfloat16)` 不等价于正确解量化。保存源/目标 dtype、转换脚本提交、配置差异和数值验证。

只有输出确已完全解除相应量化、且目标 loader 要求时，才修改或移除输出配置中的 `quantization_config`。对混合量化模型保留实际需要的元数据。不得删除原模型的配置来伪装转换完成。

区分权重、激活/累加、通信、optimizer 状态的 dtype；不能把一处 BF16 配置推广到全部路径。FP32→BF16→FP32 不会恢复已丢失的有效位，是否转换应由算子约定及对照验证决定。

## Tokenizer 与模板

优先使用该模型发布的 tokenizer 和 chat template，比较服务端与评测端的应用方式，避免重复套模板。仅当实际缺失且确认兼容时补入文件；不能给 HY4 复制 DeepSeek-V4 模板。记录模板哈希、BOS/EOS、special tokens 和 reasoning/stop 配置。

## AIS-Bench 与离线包

AIS-Bench 是可选评测器，已有等效评测路径可复用。优先为 API 客户端建立独立 venv/Conda 或独立评测容器，固定源码 commit；核对所选版本支持的 Python 版本与 requirements，再安装所需依赖。官方参考：[安装说明](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/get_started/install.html)。

若必须在已有训练环境内安装，先保存依赖清单并核对兼容性。`--no-deps` 仅跳过依赖解析，并不证明依赖齐全或兼容；不要照抄经验中的 datasets/fsspec/antlr 版本。安装后检查 `pip check`、CLI 和所需评测路径；若会破坏训练依赖，改用隔离环境。

网络慢时可在已授权的本地环境下载后上传，校验大小与 SHA256。二进制 wheel 必须匹配目标 Linux 架构、Python ABI 和系统兼容性，不能上传 Windows wheel 当作 Linux 包。纯 Python 包也需检查 Python 版本要求。固定源码包版本并记录构建依赖；镜像站、Git 源码和 PyPI 包之间的替换须明确版本/提交，不假定 HEAD 等同某个发布版。

需要代理时使用已知环境配置，避免把代理凭据打印到日志或写入技能。完整包缓存和安装日志写到运行目录，不放入技能本身。
