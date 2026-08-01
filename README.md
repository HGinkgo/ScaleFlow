<div align="center">
  <h1>ScaleFlow</h1>
  <p>面向资源受限边缘环境的多规模语言模型协同推理实验框架</p>
  <p>
    <strong>简体中文</strong> |
    <a href="README_EN.md">English</a>
  </p>
  <p>
    <a href="https://www.python.org/"><img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white"></a>
    <a href="https://pypi.org/project/vllm/0.26.0/"><img alt="vLLM 0.26.0" src="https://img.shields.io/badge/vLLM-0.26.0-4B8BBE"></a>
    <a href="https://huggingface.co/Qwen/Qwen3.5-0.8B"><img alt="Qwen3.5" src="https://img.shields.io/badge/Model-Qwen3.5-7C3AED"></a>
    <a href="LICENSE"><img alt="Apache-2.0 License" src="https://img.shields.io/badge/License-Apache%202.0-D22128?logo=apache"></a>
  </p>
</div>

## 简介

ScaleFlow 是一个轻量、配置驱动的语言模型推理实验框架，用于研究有限 GPU 资源下不同规模本地模型之间的请求路由、结果验证与动态升级。

项目不实现新的底层推理运行时，也不训练或修改模型参数。真实推理统一使用 vLLM；MockBackend 用于确定性验证调度和结果记录逻辑。

## 核心能力

- 基于 YAML 的模型、采样参数和调度策略配置；
- `AlwaysModelPolicy` 固定模型策略与 `ConfidenceCascadePolicy` 置信度级联策略；
- 可重复的 MockBackend 输出、时延、置信度和失败模拟；
- 基于 vLLM 的 Qwen3.5-0.8B 纯文本、非思考模式推理；
- 真实输出 token logprob、长度归一化 confidence、端到端时延和显存读数；
- 完整决策轨迹和直接 JSONL 结果输出；
- 不使用 GPU 的单元测试与 CLI 集成测试。

当前本地模型路线为：

```text
Qwen3.5-0.8B -> Qwen3.5-2B -> Qwen3.5-4B -> Qwen3.5-9B
```

目前仅 Qwen3.5-0.8B 已接入真实推理，其余模型和云端兜底仍属于后续工作。

## 项目结构

```text
ScaleFlow/
├── configs/                 # 可复现的实验配置
├── src/scaleflow/
│   ├── backends/            # MockBackend 与 VLLMBackend
│   ├── scheduler/           # 调度策略与同步执行流程
│   ├── schemas.py           # 共享数据结构
│   ├── config.py            # YAML 配置读取
│   └── cli.py               # 命令行入口
├── tests/                   # CPU-only 测试
├── environment.yml
└── pyproject.toml
```

模型缓存、数据集和生成结果均位于 Git 忽略目录，不会提交到仓库。

## 安装

ScaleFlow 使用 Python 3.12 和独立 conda 环境：

```bash
conda env create -f environment.yml
conda run -n scaleflow python -m pip install -e '.[dev]'
```

运行真实模型时安装固定版本的 vLLM：

```bash
conda run -n scaleflow python -m pip install -e '.[dev,vllm]'
```

## 快速开始

运行确定性 Mock 场景：

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow \
  python -m scaleflow run-mock \
  --config configs/mock_qwen35.yaml \
  --output results/mock_results.jsonl
```

运行 Qwen3.5-0.8B 真实推理：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-vllm \
  --config configs/qwen35_0_8b_vllm.yaml \
  --output results/qwen35_0_8b_results.jsonl
```

GPU 通过 `CUDA_VISIBLE_DEVICES` 选择。模型 ID、revision 和生成参数来自 YAML；下载 endpoint 可通过 `HF_ENDPOINT` 环境变量指定。

运行全部测试：

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m pytest -q
```

## 配置与测量

- `configs/mock_qwen35.yaml`：四个 Qwen3.5 规模的确定性 Mock 级联场景；
- `configs/qwen35_0_8b_vllm.yaml`：Qwen3.5-0.8B 的固定 revision、BF16、非思考模式和确定性采样参数。

对于实际生成 token 的条件对数概率 `l_i`，当前 confidence 定义为：

```text
confidence = exp(mean(output_token_logprobs))
```

JSONL 保留完整 token logprob、confidence 计算方法、时延、显存读数和决策记录。该 confidence 尚未校准为答案正确概率。

## 状态

当前已完成项目骨架、Mock 调度流程和 Qwen3.5-0.8B 单模型真实推理链路。多模型真实级联、标注数据集评测、云端兜底、并发队列和资源感知调度尚未实现。

所有公开结果都应同时记录配置、随机种子、模型 revision、运行环境和代码版本；未经正式评测的数据不会作为性能结论。

## 许可证

Copyright 2026 Pengfei_He. 本项目基于 [Apache License 2.0](LICENSE) 发布。
