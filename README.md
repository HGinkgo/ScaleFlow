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
- 固定 GSM8K 测试集、64 条小样本和 1319 条全量评测配置，包含预热流程和自动评分基线；
- 支持任意两个及以上有序模型结果的严格离线对齐、模型对挽救分析和逐级 oracle 统计；
- 基于确定性留出集的 confidence 验证、Pareto 阈值搜索和离线级联重放；
- 只使用请求文本的人工规则与 TF-IDF/逻辑回归离线路由分析；
- 基于 vLLM OpenAI 服务和异步流式客户端的真实闭环并发压测；
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
│   ├── offline.py           # Confidence 分析与离线级联重放
│   ├── routing.py           # CPU-only 推理前文本路由分析
│   ├── performance.py       # 闭环并发压测与流式时延测量
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

运行 Phase 10 的 CPU 文本路由分析还需要安装轻量分析依赖：

```bash
conda run -n scaleflow python -m pip install -e '.[analysis,dev]'
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

运行完整 GSM8K 测试集（1319 条）时，分别使用以下四个配置，并让模型进程依次运行：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow \
  python -m scaleflow run-gsm8k \
  --config configs/qwen35_0_8b_gsm8k_full.yaml \
  --output results/qwen35_0_8b_gsm8k_full.jsonl \
  --summary results/qwen35_0_8b_gsm8k_full_summary.json
```

将上述命令中的配置和输出文件替换为 `qwen35_2b_gsm8k_full.yaml`、`qwen35_4b_gsm8k_full.yaml` 和 `qwen35_9b_gsm8k_full.yaml`，不要并行驻留模型。完成四组运行后进行离线比较：

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow \
  python -m scaleflow compare-gsm8k-multi \
  --inputs results/qwen35_0_8b_gsm8k_full.jsonl \
           results/qwen35_2b_gsm8k_full.jsonl \
           results/qwen35_4b_gsm8k_full.jsonl \
           results/qwen35_9b_gsm8k_full.jsonl \
  --output results/qwen35_gsm8k_full_comparison.json
```

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

使用已有四组全量结果完成 confidence 验证、开发集阈值搜索和一次性留出集评估：

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m scaleflow \
  analyze-gsm8k-confidence --config configs/qwen35_gsm8k_offline.yaml \
  --inputs results/qwen35_0_8b_gsm8k_full.jsonl results/qwen35_2b_gsm8k_full.jsonl \
           results/qwen35_4b_gsm8k_full.jsonl results/qwen35_9b_gsm8k_full.jsonl \
  --output results/qwen35_gsm8k_confidence_development.json

CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m scaleflow \
  search-gsm8k-cascade --config configs/qwen35_gsm8k_offline.yaml \
  --confidence-report results/qwen35_gsm8k_confidence_development.json \
  --inputs results/qwen35_0_8b_gsm8k_full.jsonl results/qwen35_2b_gsm8k_full.jsonl \
           results/qwen35_4b_gsm8k_full.jsonl results/qwen35_9b_gsm8k_full.jsonl \
  --output results/qwen35_gsm8k_cascade_policy.json

CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m scaleflow \
  evaluate-gsm8k-cascade --config configs/qwen35_gsm8k_offline.yaml \
  --policy results/qwen35_gsm8k_cascade_policy.json \
  --inputs results/qwen35_0_8b_gsm8k_full.jsonl results/qwen35_2b_gsm8k_full.jsonl \
           results/qwen35_4b_gsm8k_full.jsonl results/qwen35_9b_gsm8k_full.jsonl \
  --output results/qwen35_gsm8k_cascade_evaluation.json
```

阈值只在660条开发集上选择；659条留出集只用于冻结策略的一次评估。评估完成后会在策略文件旁写入 `.evaluated` 标记，重复执行会被拒绝。以上命令不加载模型，也不使用 GPU。

运行单模型闭环并发实验（四个模型需依次执行）：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n scaleflow python -m scaleflow \
  run-gsm8k-concurrency --config configs/qwen35_gsm8k_concurrency.yaml \
  --model-id Qwen/Qwen3.5-0.8B \
  --reference results/qwen35_0_8b_gsm8k_full.jsonl \
  --reference-summary results/qwen35_0_8b_gsm8k_full_summary.json \
  --output results/phase9_qwen35_0_8b_concurrency.jsonl \
  --summary results/phase9_qwen35_0_8b_concurrency_summary.json \
  --server-log results/phase9_qwen35_0_8b_concurrency_server.log
```

使用 Phase 8 已保存的划分，对 Phase 7 的 2B、4B、9B 全量结果执行一次推理前文本路由分析：

```bash
CUDA_VISIBLE_DEVICES="" conda run -n scaleflow python -m scaleflow \
  analyze-gsm8k-routing \
  --config configs/qwen35_gsm8k_routing.yaml \
  --split-report results/phase8_confidence_development.json \
  --inputs results/phase7_persistent_qwen35_2b_gsm8k_full.jsonl \
           results/phase7_persistent_qwen35_4b_gsm8k_full.jsonl \
           results/phase7_persistent_qwen35_9b_gsm8k_full.jsonl \
  --output results/phase10_text_routing.json
```

该命令只在 660 条开发集拟合人工规则和 TF-IDF/逻辑回归，随后冻结两者并对 659 条已在 Phase 8 使用过的探索性评估集执行一次。结果文件位于 Git 忽略目录，不重复运行或依据评估集调参。

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
- `configs/qwen35_*_gsm8k_full.yaml`：使用同一数据 commit、Prompt、生成参数和预热流程，按原始顺序选择完整 1319 条测试记录。
- `configs/qwen35_gsm8k_offline.yaml`：固定留出划分、confidence bootstrap、阈值候选步长和随机接受基线种子集合。
- `configs/qwen35_gsm8k_concurrency.yaml`：固定128条请求、四个模型 revision、统一 `gpu_memory_utilization=0.90`、预热和五档并发协议。
- `configs/qwen35_gsm8k_routing.yaml`：固定 Phase 8 的 SHA256 划分、文本特征、规则阈值候选、TF-IDF/逻辑回归参数和匹配比例随机种子。

GSM8K 基线使用 OpenAI 官方测试集 commit `3101c7d5072418e28b9008a6636bde82a006892c`，并校验 SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。后续不同规模模型应复用同一配置中的样本索引和 Prompt。

对于实际生成 token 的条件对数概率 `l_i`，当前 confidence 定义为：

```text
confidence = exp(mean(output_token_logprobs))
```

JSONL 保留完整 token logprob、confidence 计算方法、时延、显存读数和决策记录。该 confidence 尚未校准为答案正确概率。

GSM8K 结果将 `correct`、`incorrect`、`parse_failure` 和 `inference_failure` 分开记录；parse failure 不会被隐藏在回答错误中。64 条样本上的 confidence 与正确性关系只作初步观察，不构成正式统计结论。

## 全量 GSM8K 评测

本次全量评测使用上述 1319 条测试记录的原始顺序，保持 64 条基线的 Prompt、模型 revision、BF16、非思考模式、生成参数和 8 条预热请求不变；四个模型在同一张 RTX 3090 上分别运行，结果和数据均不提交 Git。

| 模型 | 正确率 | incorrect | parse_failure | inference_failure | 平均时延 ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 52.24% | 540 | 90 | 0 | 3962.16 | 3376.18 | 8968.62 |
| Qwen3.5-2B | 64.59% | 389 | 78 | 0 | 2954.64 | 2611.49 | 6165.59 |
| Qwen3.5-4B | 89.61% | 120 | 17 | 0 | 3956.88 | 3363.12 | 8033.22 |
| Qwen3.5-9B | 92.95% | 60 | 33 | 0 | 4698.13 | 4150.91 | 9594.98 |

| 模型 | 输出 token 均值 | aggregate tokens/s | 峰值显存 MiB | confidence 与正确性的点二列相关 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 165.87 | 41.86 | 6882.06 | 0.403 |
| Qwen3.5-2B | 121.09 | 40.98 | 6992.06 | 0.413 |
| Qwen3.5-4B | 122.30 | 30.91 | 11662.06 | 0.397 |
| Qwen3.5-9B | 148.36 | 31.58 | 22678.06 | 0.431 |

4B 与 9B 的逐请求对齐结果为：两者都正确 1143 条、只有 4B 正确 39 条、只有 9B 正确 83 条、两者都未正确 54 条。4B 未正确的 137 条中，9B 挽救 83 条，挽救率为 60.58%；其中挽救 `incorrect` 74/120，挽救 `parse_failure` 9/17，`inference_failure` 为 0/0。9B 未正确但至少一个更小模型正确的非单调样本共有 50 条；全部样本 ID、四模型 16 种正确性组合及各组合 ID 保存在 `results/qwen35_gsm8k_full_comparison.json` 的 `per_request`、`correctness_combinations` 字段中。

正确性组合的位顺序为 `0.8B, 2B, 4B, 9B`，计数为：

```text
0000:43  0001:51  0010:24  0011:210
0100:4   0101:20  0110:9   0111:269
1000:3   1001:4   1010:3   1011:129
1100:4   1101:8   1110:3   1111:535
```

逐级加入模型的事后 Oracle 正确率为：0.8B 52.24%（689 条）、加入 2B 后 75.13%（991 条，增量 302）、加入 4B 后 92.87%（1225 条，增量 234）、加入 9B 后 96.74%（1276 条，增量 51）。因此完整 GSM8K 能稳定区分四个模型；4B 与 9B 仍较接近，但全量数据的 3.34 个百分点差距明显大于 64 条样本上的小差距。GSM8K 对高规模模型相对容易，尚不能视为完全饱和；后续若要检验更细的调度收益，应加入难度更高且协议独立的评测集。上述 confidence 相关性仅为本次固定评测上的探索性观察，不构成校准或正式统计结论。

## 离线 Confidence 级联

1319条结果按 `SHA256(seed=42, sample_id)` 固定划分为660条开发集和659条留出集。0.8B、2B、4B在开发集上的 AUROC 分别为0.701、0.743和0.779，95% bootstrap 区间下界均高于0.5；最低 confidence 的20%样本未正确率增量及 AURC 改善的区间下界也均高于0，因此三个模型均保留在候选链中。

51个候选点包含“始终接受”和“始终升级”边界。132,651组阈值中有120组达到开发集9B准确率，最低累计均值时延的阈值为0.9256、0.9414和0.9421。该策略在开发集与9B同为607/660正确。

冻结策略在留出集上得到611/659正确（92.72%），低于9B单模型的619/659（93.93%）；最终失败为39条 `incorrect`、9条 `parse_failure` 和0条 `inference_failure`。9B调用率为65.86%，相当于减少34.14%的9B调用，但串行累计时延为13,744/12,189/27,033 ms（平均/P50/P95），明显高于9B单模型的4,611/4,154/8,912 ms。

固定1000个种子的随机接受基线使用近似相同的逐级接受率和9B调用率，平均准确率为90.74%，随机化区间为89.38%至92.11%。confidence 级联相对其平均提升1.97个百分点，差值随机化区间为0.61至3.34个百分点。这说明 confidence 排序提供了有效信息，但当前阈值策略没有在留出集保持9B质量，也没有降低串行端到端时延。

上述时延是已有单模型记录的离线求和，不包含模型加载、切换、排队和并发干扰；随机化区间只描述固定留出集上的随机策略波动，不是总体性能的统计置信区间。

## 并发服务评测

从GSM8K测试集以 `seed=42` 固定选择同一组128条请求，四个模型在单张RTX 3090上依次运行。服务启动后先执行8条预热，每档并发再执行一轮同并发度预热；正式测试使用闭环并发1、2、4、8、16，正常生成至EOS，`max_tokens=384`仅作为安全上限。TTFT从客户端收到首个非空文本token计时，TPOT为包含本机HTTP流式传输和事件处理开销的客户端观测均值。

| 模型 | 并发1 req/s | 并发2 | 并发4 | 并发8 | 并发16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 0.259 | 0.483 | 0.878 | 1.725 | 3.063 |
| Qwen3.5-2B | 0.364 | 0.700 | 1.340 | 2.568 | 4.502 |
| Qwen3.5-4B | 0.251 | 0.474 | 0.948 | 1.783 | 3.037 |
| Qwen3.5-9B | 0.215 | 0.413 | 0.780 | 1.472 | 2.635 |

| 模型（并发16） | 输出 tok/s | 平均时延 ms | P95 ms | TTFT P95 ms | TPOT均值 ms | 解析答案一致率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 489.9 | 4403 | 8344 | 132 | 27.08 | 87.50% |
| Qwen3.5-2B | 521.4 | 3279 | 5964 | 182 | 27.60 | 88.28% |
| Qwen3.5-4B | 372.5 | 4534 | 10058 | 326 | 35.98 | 97.66% |
| Qwen3.5-9B | 377.7 | 5173 | 9964 | 541 | 35.12 | 97.66% |

| 模型 | 权重 GiB | KV Cache GiB | KV Cache tokens | NVML峰值 MiB |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 1.53 | 18.68 | 877714 | 22807 |
| Qwen3.5-2B | 3.63 | 16.57 | 778532 | 22819 |
| Qwen3.5-4B | 7.99 | 12.21 | 221476 | 22883 |
| Qwen3.5-9B | 16.80 | 3.39 | 61440 | 22747 |

全部2560条正式请求成功，未出现OOM、KV Cache不足或服务排队；日志中最高KV Cache使用率为0.8B/2B/4B/9B的2.1%/2.4%/8.5%/30.6%。统一的显存利用率会让vLLM把权重之外的显存用于KV Cache及运行时空间，因此约22.2 GiB的NVML峰值不能直接解释为模型权重。

2B在所有并发档位的请求吞吐和平均时延上均优于0.8B，结合全量GSM8K质量结果，0.8B在本次并发范围内没有实际服务优势。4B相对9B具有约15%至21%的请求吞吐优势，同时保持接近的质量，适合作为主要边缘质量层；9B适合作为本地高质量兜底。并发16时9B的TTFT P95升至541 ms，但没有等待队列或KV Cache压力，当前瓶颈更接近批处理下的计算竞争。建议核心链保留 `2B -> 4B -> 9B`，0.8B仅保留为对照基线。

并发1的解析答案和全文均与单模型基线完全一致。并发大于1时，即使贪心参数和固定seed不变，GPU批处理数值路径仍使完整文本一致率下降；因此主要复现指标采用解析后的最终答案一致率，全文一致率作为严格指标。

## 推理前文本路由（Phase 10，探索性）

本分析不重新运行模型，只使用 Phase 7 的 2B、4B、9B 全量结果。样本按 Phase 8 已保存的 `SHA256(seed=42, sample_id)` 划分为 660 条开发集和 659 条评估集；评估集已在 Phase 8 使用，因此以下仅是探索性离线结果，不是独立测试结论。输入只包含问题文本及字符数、词数、数字数、运算符数和关键词数，不使用输出、confidence、正确性或实际时延。

| 方法 | 准确率 | incorrect / parse / inference | none_correct | 2B / 4B / 9B 调用比例 | 历史时延均值 / P50 / P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 始终 2B | 62.67% | 207 / 39 / 0 | 24 | 100 / 0 / 0% | 3053 / 2679 / 6864 |
| 始终 4B | 89.83% | 58 / 9 / 0 | 24 | 0 / 100 / 0% | 3987 / 3361 / 8115 |
| 始终 9B | 93.93% | 29 / 11 / 0 | 24 | 0 / 0 / 100% | 4611 / 4154 / 8912 |
| 人工规则 | 81.34% | 101 / 22 / 0 | 24 | 50.53 / 4.40 / 45.07% | 3915 / 3367 / 8580 |
| TF-IDF + 逻辑回归 | 70.71% | 164 / 29 / 0 | 24 | 73.14 / 23.07 / 3.79% | 3361 / 2933 / 7356 |
| 事后理想最低正确模型 | 96.36% | 17 / 7 / 0 | 24 | 62.67 / 29.44 / 7.89% | 3545 / 2954 / 7868 |

时延表中的“历史”只重放被选模型的自然完成时延；加入实测路由器开销后，人工规则均值为 3915.41 ms（预测开销均值 0.032 ms），TF-IDF 均值为 3361.47 ms（预测开销均值 0.547 ms）。训练耗时分别为 43.9 ms 和 10.42 s。两种路由均未达到预设的“评估集准确率不低于 9B 下降 1 个百分点”标准：人工规则比 9B 低 12.59 个百分点，TF-IDF 低 23.22 个百分点。

匹配调用比例的 1000 个固定种子随机基线中，人工规则对应随机准确率均值 77.93%（区间 76.02%–79.82%），人工规则提升 3.40 个百分点，差值区间 1.52–5.31 个百分点；TF-IDF 对应随机均值 70.11%（68.44%–71.78%），提升 0.60 个百分点，差值区间 -1.06–2.28 个百分点。人工规则有有限的排序信号，但质量仍不足；TF-IDF 在开发集 96.52% 而评估集降至 70.71%，说明泛化不稳定，不能据此继续调参。三模型均未正确的请求单独记为 `none_correct`，没有混入“错误选择 9B”。结论是：仅依赖请求文本的轻量路由不足以替代当前质量基线。

## 许可证

Copyright 2026 Pengfei_He. 本项目基于 [Apache License 2.0](LICENSE) 发布。
