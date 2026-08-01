# ScaleFlow

English Version: [README_EN.md](README_EN.md)

## 项目简介

ScaleFlow 是一个面向边缘资源受限场景的多规模语言模型协同推理与调度研究框架。项目关注在有限 GPU 资源、差异化质量需求、时延约束和云端访问成本同时存在时，如何在多个不同规模的本地语言模型之间进行路由、验证与动态升级，并仅在必要时调用云端模型。

ScaleFlow 不实现新的底层推理引擎，也不训练或修改语言模型参数。真实推理使用成熟运行时；当前已通过 vLLM 接入 `Qwen/Qwen3.5-0.8B`，同时保留确定性的 MockBackend 验证调度逻辑和实验数据流。

## 论文研究背景

本项目服务于硕士学位论文“面向边缘资源受限场景的多规模语言模型协同推理与调度优化”。核心研究问题是：面对难度、质量要求和实时性要求不同的请求，系统应如何选择本地模型、何时升级到更大模型、何时返回当前结果，以及何时使用云端兜底，从而权衡任务质量、端到端时延、吞吐、GPU 资源消耗和云端调用比例。

研究聚焦边缘计算和云边协同部署，不涉及无线物理层建模、模型参数训练、复杂显存管理系统或为使用强化学习而强行引入 DQN/PPO。

## 边缘资源受限场景

目标场景包含以下约束：

- 终端请求具有不同难度、质量门槛和最大允许时延；
- 边缘节点仅有有限 GPU、显存和并发服务能力；
- 边缘到云端存在网络往返时延、传输开销和 API 调用成本；
- 简单请求应尽量由小模型低成本完成，复杂请求才逐级升级；
- 云端模型作为兜底和质量参考，而不是论文研究的主要对象。

## 多规模模型协同方案

本地模型族确定为 Qwen3.5：

1. `Qwen/Qwen3.5-0.8B`
2. `Qwen/Qwen3.5-2B`
3. `Qwen/Qwen3.5-4B`
4. `Qwen/Qwen3.5-9B`

质量感知级联的目标路径为 `0.8B -> 2B -> 4B -> 9B`。模型顺序和置信度阈值由 YAML 配置，不写死在调度代码中。规划中的云端兜底模型为 DeepSeek-V4-Flash；该云端接口当前尚未实现，也不会在 Mock 实验中被调用。

## 当前已实现

当前完成 Phase 2，已经实现：

- 四个统一数据结构：`InferenceRequest`、`ModelResponse`、`DecisionRecord`、`InferenceResult`；
- `AlwaysModelPolicy` 固定模型策略；
- `ConfidenceCascadePolicy` 配置驱动的置信度级联策略；
- 可通过 YAML 配置输出、置信度、模拟时延和成功/失败状态的 MockBackend；
- easy、medium、hard 三条确定性请求；
- 完整 `decision_trace`、升级次数和累计模拟时延；
- 简单 JSONL 结果写入；
- CPU-only 单元测试和 CLI 集成测试；
- 基于稳定版 vLLM 的 `Qwen/Qwen3.5-0.8B` 单模型同步文本推理；
- 显式纯文本、非思考模式和固定模型 revision；
- 每个生成 token 的真实选中 token logprob、长度归一化 confidence、端到端推理时延和 NVML 显存读数；
- 由 YAML 定义的 5 条固定真实模型冒烟请求；
- SGLang 的未接入占位接口。

当前没有接入 2B、4B、9B、SGLang 或 DeepSeek API，也没有实现多模型真实级联、并发队列、资源感知调度、正式数据集评测、网络模拟、正式 benchmark、绘图或学习式策略。

## 项目结构

```text
ScaleFlow/
├── configs/
│   ├── mock_qwen35.yaml
│   └── qwen35_0_8b_vllm.yaml
├── src/scaleflow/
│   ├── backends/
│   │   ├── base.py
│   │   ├── mock.py
│   │   ├── sglang.py
│   │   └── vllm.py
│   ├── scheduler/
│   │   ├── policies.py
│   │   └── runner.py
│   ├── schemas.py
│   ├── config.py
│   ├── cli.py
│   └── __main__.py
├── tests/
├── environment.yml
└── pyproject.toml
```

运行生成的 `results/` 目录和实验产物不会提交到 Git。

## 环境创建与安装

要求 Linux、conda 和 Python 3.12。仅运行 Mock 时不需要 GPU；真实模型需要 NVIDIA GPU、兼容驱动和 vLLM 可选依赖。

```bash
conda create -n scaleflow python=3.12 pip -y
conda run -n scaleflow python -m pip install -e '.[dev]'
```

也可以使用仓库中的环境文件：

```bash
conda env create -f environment.yml
conda run -n scaleflow python -m pip install -e '.[dev]'
```

运行真实模型时安装固定的稳定版 vLLM；其依赖会安装匹配的 PyTorch 和 CUDA 用户态运行库：

```bash
conda run -n scaleflow python -m pip install -e '.[dev,vllm]'
```

## 运行 Mock 示例

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow \
  python -m scaleflow run-mock \
  --config configs/mock_qwen35.yaml \
  --output results/mock_results.jsonl
```

该命令运行三条固定请求并生成 JSONL。JSONL 是运行产物，不纳入版本控制。Mock 中的时延和置信度是配置值，不是真实模型实验结果。

## 运行测试

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m pytest -q
```

测试不使用 GPU、不下载模型，也不调用远程 API。

## 运行 Qwen3.5-0.8B

下面的命令只暴露一张 GPU，通过 vLLM 运行 5 条固定文本请求。首次运行会将模型下载到 Hugging Face 缓存；权重、缓存和结果均被 Git 排除。

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-vllm \
  --config configs/qwen35_0_8b_vllm.yaml \
  --output results/qwen35_0_8b_results.jsonl
```

如果服务器无法直接访问 Hugging Face，可通过环境变量指定经过确认的兼容 endpoint。当前受限网络环境验证时使用：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-vllm \
  --config configs/qwen35_0_8b_vllm.yaml \
  --output results/qwen35_0_8b_results.jsonl
```

模型加载只接受文本，`enable_thinking` 固定为 `false`。命令结束时会在标准输出打印模型加载时延、加载前后显存和 vLLM 版本；逐请求 JSONL 保存输出、推理时延、显存、决策记录和原始 logprob。

### Logprob 与 confidence

配置使用 `logprobs: 1`。对于生成 token 序列中的第 `i` 个实际选中 token，vLLM 返回：

```text
l_i = log p(token_i | prompt, token_1, ..., token_{i-1})
```

`token_logprobs` 原样保存所有 `l_i`。ScaleFlow 使用生成 token 概率的几何平均作为当前实验 confidence：

```text
confidence = exp((1 / T) * sum(l_i))
```

JSONL 中的 `confidence_method` 固定记录为 `exp(mean(output_token_logprobs))`。该变换减弱输出长度对概率连乘的影响，但它尚未经过任务准确率校准，不能直接解释为答案正确概率。

`total_latency_ms` 从调用 `LLM.chat` 前开始，到文本、logprob 和 confidence 解析完成后结束，不包含一次性模型加载。首请求可能包含 Triton kernel JIT，不能将这 5 条冒烟结果视为正式性能 benchmark。`gpu_memory_used_mb` 是 NVML 在所选可见 GPU 上读取的总已用显存；CLI 摘要另外给出加载前后增量。

## 配置文件

[`configs/mock_qwen35.yaml`](configs/mock_qwen35.yaml) 包含：

- 固定随机种子；
- 调度策略名称；
- 置信度阈值；
- 模型调用顺序；
- easy、medium、hard 请求；
- 每个模型针对每条请求的输出、置信度、模拟时延、成功状态和错误信息。

配置中不包含真实模型路径、GPU 编号或 API Key。未来涉及敏感信息时必须通过未提交的环境变量或本地 `.env` 提供。

[`configs/qwen35_0_8b_vllm.yaml`](configs/qwen35_0_8b_vllm.yaml) 固定模型 ID 和 revision、BF16、4096 上下文、显存利用率、非思考模式、确定性采样参数、logprob 数量和 5 条文本请求。GPU 由 `CUDA_VISIBLE_DEVICES` 指定，下载 endpoint 由 `HF_ENDPOINT` 指定，二者均不写死在配置中。

## 开发状态与后续路线

当前完成 Phase 0（独立环境与项目骨架）、Phase 1（MockBackend 最小调度流程）和 Phase 2（Qwen3.5-0.8B 单模型真实推理）。后续计划按实验需求逐步推进：

1. 分别接入并验证 Qwen3.5-2B、4B 和 9B；
2. 在标注任务上校准真实 token logprob 与质量的关系；
3. 扩展真实固定路由、动态级联、Local-Only、Cloud-Only 和 Oracle 基线；
4. 加入队列、GPU 负载、截止时间和通信成本等资源状态；
5. 运行不同负载与网络条件的可重复实验并生成 Pareto 分析；
6. 仅在规则式策略暴露明确不足时评估简单学习式方法。

以上均为规划，尚未实现的能力不会作为当前项目功能或实验结论陈述。

## 实验可复现性

Mock 场景由 YAML 完整定义，输入顺序、模型响应、阈值和模拟时延固定，相同配置会产生一致结果。真实模型配置固定模型 revision、随机种子、推理参数和请求顺序；GPU 型号、驱动、CUDA、PyTorch、vLLM、代码 commit、加载时延及逐请求结果应随实验记录。生成的模型缓存和原始结果默认不进入 Git；公开论文数据时将使用独立、经过脱敏和说明的发布流程。

## 许可证状态

当前项目处于私人研究阶段，正式公开前将确定开源许可证。
