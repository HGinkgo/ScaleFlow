# ScaleFlow

English Version: [README_EN.md](README_EN.md)

## 项目简介

ScaleFlow 是一个面向边缘资源受限场景的多规模语言模型协同推理与调度研究框架。项目关注在有限 GPU 资源、差异化质量需求、时延约束和云端访问成本同时存在时，如何在多个不同规模的本地语言模型之间进行路由、验证与动态升级，并仅在必要时调用云端模型。

ScaleFlow 不实现新的底层推理引擎，也不训练或修改语言模型参数。后续真实推理将优先接入成熟的 vLLM 或 SGLang，当前阶段使用确定性的 MockBackend 验证调度逻辑和实验数据流。

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

当前处于 Phase 1，已经实现：

- 四个统一数据结构：`InferenceRequest`、`ModelResponse`、`DecisionRecord`、`InferenceResult`；
- `AlwaysModelPolicy` 固定模型策略；
- `ConfidenceCascadePolicy` 配置驱动的置信度级联策略；
- 可通过 YAML 配置输出、置信度、模拟时延和成功/失败状态的 MockBackend；
- easy、medium、hard 三条确定性请求；
- 完整 `decision_trace`、升级次数和累计模拟时延；
- 简单 JSONL 结果写入；
- CPU-only 单元测试和 CLI 集成测试；
- vLLM 与 SGLang 的未接入占位接口，不包含对应运行时依赖。

当前没有实现真实模型加载、真实置信度、GPU 调度、数据集评测、资源监控、网络模拟、DeepSeek API、正式 benchmark、绘图或学习式策略。

## 项目结构

```text
ScaleFlow/
├── configs/
│   └── mock_qwen35.yaml
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

要求 Linux、conda 和 Python 3.12。本阶段不需要 CUDA、PyTorch、vLLM 或 SGLang。

```bash
conda create -n scaleflow python=3.12 pip -y
conda run -n scaleflow python -m pip install -e '.[dev]'
```

也可以使用仓库中的环境文件：

```bash
conda env create -f environment.yml
conda run -n scaleflow python -m pip install -e '.[dev]'
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

## 配置文件

[`configs/mock_qwen35.yaml`](configs/mock_qwen35.yaml) 包含：

- 固定随机种子；
- 调度策略名称；
- 置信度阈值；
- 模型调用顺序；
- easy、medium、hard 请求；
- 每个模型针对每条请求的输出、置信度、模拟时延、成功状态和错误信息。

配置中不包含真实模型路径、GPU 编号或 API Key。未来涉及敏感信息时必须通过未提交的环境变量或本地 `.env` 提供。

## 开发状态与后续路线

当前完成 Phase 0（独立环境与项目骨架）和 Phase 1（MockBackend 最小调度流程）。后续计划按实验需求逐步推进：

1. 接入真实推理后端并完成单模型性能剖面；
2. 获取真实 token log probability，校准置信度阈值；
3. 扩展固定路由、动态级联、Local-Only、Cloud-Only 和 Oracle 基线；
4. 加入队列、GPU 负载、截止时间和通信成本等资源状态；
5. 运行不同负载与网络条件的可重复实验并生成 Pareto 分析；
6. 仅在规则式策略暴露明确不足时评估简单学习式方法。

以上均为规划，尚未实现的能力不会作为当前项目功能或实验结论陈述。

## 实验可复现性

当前 Mock 场景由 YAML 完整定义，输入顺序、模型响应、阈值和模拟时延固定，相同配置会产生一致结果。正式实验将同时记录代码 commit、环境版本、配置文件、随机种子、数据集版本、模型版本和逐请求结果。生成的原始结果默认不进入 Git；公开论文数据时将使用独立、经过脱敏和说明的发布流程。

## 许可证状态

当前项目处于私人研究阶段，正式公开前将确定开源许可证。
