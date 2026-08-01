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
- 固定 GSM8K 测试集、64 个样本 ID、预热流程和自动评分基线；
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
│   ├── baseline.py          # GSM8K 评分、统计与结果写入
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

运行固定的 GSM8K-64 单模型基线：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-gsm8k \
  --config configs/qwen35_0_8b_gsm8k.yaml \
  --output results/qwen35_0_8b_gsm8k_64.jsonl \
  --summary results/qwen35_0_8b_gsm8k_64_summary.json
```

该命令首次运行会下载并校验官方 GSM8K `test` 数据；原始数据、逐条 JSONL 和汇总 JSON 均被 Git 忽略。

运行全部测试：

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m pytest -q
```

## 配置与测量

- `configs/mock_qwen35.yaml`：四个 Qwen3.5 规模的确定性 Mock 级联场景；
- `configs/qwen35_0_8b_vllm.yaml`：Qwen3.5-0.8B 的固定 revision、BF16、非思考模式和确定性采样参数。
- `configs/qwen35_0_8b_gsm8k.yaml`：固定 GSM8K commit、SHA256、64 个样本索引、Prompt、8 条预热请求和生成参数。

GSM8K 基线使用 OpenAI 官方测试集 commit `3101c7d5072418e28b9008a6636bde82a006892c`，并校验 SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。后续不同规模模型应复用同一配置中的样本索引和 Prompt。

对于实际生成 token 的条件对数概率 `l_i`，当前 confidence 定义为：

```text
confidence = exp(mean(output_token_logprobs))
```

JSONL 保留完整 token logprob、confidence 计算方法、时延、显存读数和决策记录。该 confidence 尚未校准为答案正确概率。

GSM8K 结果将 `correct`、`incorrect`、`parse_failure` 和 `inference_failure` 分开记录；parse failure 不会被隐藏在回答错误中。64 条样本上的 confidence 与正确性关系只作初步观察，不构成正式统计结论。

## 状态

当前已完成项目骨架、Mock 调度流程、Qwen3.5-0.8B 单模型真实推理链路和固定 GSM8K-64 基线。多模型真实级联、云端兜底、并发队列和资源感知调度尚未实现。

一次固定运行得到 33/64 正确、28 条答案错误和 3 条格式解析失败。配置将 `gpu_memory_utilization` 从早期冒烟测试的 0.4 调整为 0.25；当前 vLLM 日志中权重约 1.53 GiB、预留 KV cache 约 3.36 GiB，NVML 观测的加载显存增量约 6.27 GiB。总显存还包含激活和运行时开销，不能等同于模型权重。8 条预热请求排除了模型加载和主要 Triton JIT 影响；准确率、时延和 confidence 关系仅作初步观察，不构成正式性能或统计结论。

## 许可证

Copyright 2026 Pengfei_He. 本项目基于 [Apache License 2.0](LICENSE) 发布。
