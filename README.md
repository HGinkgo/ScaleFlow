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

ScaleFlow 是一个轻量、配置驱动的语言模型推理实验框架，用于研究有限 GPU 资源下不同规模本地模型之间的请求选择、结果验证与协同推理。

项目不实现新的推理 runtime，也不训练模型参数。真实推理统一使用 vLLM；MockBackend 用于验证调度和结果记录逻辑。

## 核心能力

- YAML 配置驱动的模型、采样参数和实验流程；
- Qwen3.5-0.8B、2B、4B、9B 的纯文本、非思考模式推理；
- MockBackend、固定模型策略和置信度级联策略；
- JSONL 结果记录、GSM8K 评分和多模型离线对齐；
- CPU-only 单元测试与 CLI 集成测试。

## 项目结构

```text
ScaleFlow/
├── configs/                 # 实验配置
├── src/scaleflow/
│   ├── backends/            # MockBackend 与 VLLMBackend
│   ├── scheduler/           # 策略与执行流程
│   ├── baseline.py          # 评分与结果写入
│   ├── offline.py           # 离线分析
│   ├── routing.py           # 文本路由分析
│   ├── performance.py       # 性能测试
│   ├── schemas.py           # 共享数据结构
│   ├── config.py            # YAML 读取
│   └── cli.py               # CLI 入口
├── tests/                   # 测试
├── environment.yml
└── pyproject.toml
```

模型缓存、数据集、日志和实验结果均位于 Git 忽略目录，不会提交到仓库。

## 安装

```bash
conda env create -f environment.yml
conda run -n scaleflow python -m pip install -e '.[dev,analysis]'
```

真实模型运行需另外安装固定版本的 vLLM 可选依赖 `.[vllm]`。

## 快速开始

运行确定性 Mock 场景：

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow \
  python -m scaleflow run-mock \
  --config configs/mock_qwen35.yaml \
  --output results/mock_results.jsonl
```

运行测试：

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m pytest -q
```

其他实验通过 `python -m scaleflow` 子命令运行，参数和协议直接查看 `configs/` 及对应源代码；README 不重复展开实验步骤和结果。

三模型离线级联实验的配置模板为 `configs/qwen35_gsm8k_three_cascade.yaml`。

## 许可证

Copyright 2026 Pengfei_He. 本项目基于 [Apache License 2.0](LICENSE) 发布。
