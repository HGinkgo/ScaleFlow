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
- 基于 vLLM 的 Qwen3.5-0.8B、2B、4B 与 9B 纯文本、非思考模式推理；
- 真实输出 token logprob、长度归一化 confidence、端到端时延和显存读数；
- 完整决策轨迹和直接 JSONL 结果输出；
- 固定 GSM8K 测试集、64 个样本 ID、预热流程和自动评分基线；
- 支持任意两个及以上有序模型结果的严格离线对齐、模型对挽救分析和逐级 oracle 统计；
- 不使用 GPU 的单元测试与 CLI 集成测试。

当前本地模型路线为：

```text
Qwen3.5-0.8B -> Qwen3.5-2B -> Qwen3.5-4B -> Qwen3.5-9B
```

目前四个 Qwen3.5 本地模型均已接入真实推理；云端兜底仍属于后续工作。

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

使用相同实验契约依次运行 2B、4B 和 9B，再离线对齐四组结果：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-gsm8k \
  --config configs/qwen35_2b_gsm8k.yaml \
  --output results/qwen35_2b_gsm8k_64.jsonl \
  --summary results/qwen35_2b_gsm8k_64_summary.json

CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-gsm8k \
  --config configs/qwen35_4b_gsm8k.yaml \
  --output results/qwen35_4b_gsm8k_64.jsonl \
  --summary results/qwen35_4b_gsm8k_64_summary.json

CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-gsm8k \
  --config configs/qwen35_9b_gsm8k.yaml \
  --output results/qwen35_9b_gsm8k_64.jsonl \
  --summary results/qwen35_9b_gsm8k_64_summary.json

CUDA_VISIBLE_DEVICES="" conda run -n scaleflow \
  python -m scaleflow compare-gsm8k-multi \
  --inputs results/qwen35_0_8b_gsm8k_64.jsonl \
           results/qwen35_2b_gsm8k_64.jsonl \
           results/qwen35_4b_gsm8k_64.jsonl \
           results/qwen35_9b_gsm8k_64.jsonl \
  --output results/qwen35_multi_comparison.json
```

各模型应在独立进程中依次运行，不同时驻留显存。`--inputs` 顺序表示模型能力顺序；比较器校验样本、Prompt、参考答案和公共实验配置，并保留全部正确性组合与样本 ID。`compare-gsm8k` 继续作为双模型兼容入口。比较命令不执行实际级联。

运行全部测试：

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m pytest -q
```

## 配置与测量

- `configs/mock_qwen35.yaml`：四个 Qwen3.5 规模的确定性 Mock 级联场景；
- `configs/qwen35_0_8b_vllm.yaml`：Qwen3.5-0.8B 的固定 revision、BF16、非思考模式和确定性采样参数。
- `configs/qwen35_0_8b_gsm8k.yaml`：固定 GSM8K commit、SHA256、64 个样本索引、Prompt、8 条预热请求和生成参数。
- `configs/qwen35_2b_gsm8k.yaml`：保持相同实验契约，仅固定 2B 模型 revision 和运行时显存配置。
- `configs/qwen35_4b_gsm8k.yaml`：保持相同实验契约，固定 4B 模型 revision，并使用独立的运行时显存配置。
- `configs/qwen35_9b_gsm8k.yaml`：保持相同实验契约，固定 9B BF16 revision，单卡运行时显存比例为 `0.90`。

GSM8K 基线使用 OpenAI 官方测试集 commit `3101c7d5072418e28b9008a6636bde82a006892c`，并校验 SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。后续不同规模模型应复用同一配置中的样本索引和 Prompt。

对于实际生成 token 的条件对数概率 `l_i`，当前 confidence 定义为：

```text
confidence = exp(mean(output_token_logprobs))
```

JSONL 保留完整 token logprob、confidence 计算方法、时延、显存读数和决策记录。该 confidence 尚未校准为答案正确概率。

GSM8K 结果将 `correct`、`incorrect`、`parse_failure` 和 `inference_failure` 分开记录；parse failure 不会被隐藏在回答错误中。64 条样本上的 confidence 与正确性关系只作初步观察，不构成正式统计结论。

## 许可证

Copyright 2026 Pengfei_He. 本项目基于 [Apache License 2.0](LICENSE) 发布。
