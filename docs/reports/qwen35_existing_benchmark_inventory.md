# Qwen3.5 已有实验结果盘点

盘点范围为当前工作区已有文件，未重新运行模型、未下载模型、未启动 vLLM 服务。`results/` 及相关日志目录均为 Git 忽略目录；本文档只记录已存在的结果。

## 结果文件清单

### 真实模型单模型 GSM8K

| 模型 | 结果 JSONL | 汇总 JSON | 样本数 | 结果性质 |
|---|---|---|---:|---|
| Qwen3.5-0.8B | `results/phase7_persistent_qwen35_0_8b_gsm8k_full.jsonl` | 同名 `_summary.json` | 1319 | 真实 vLLM，Phase 7 全量 |
| Qwen3.5-2B | `results/phase7_persistent_qwen35_2b_gsm8k_full.jsonl` | 同名 `_summary.json` | 1319 | 真实 vLLM，Phase 7 全量 |
| Qwen3.5-4B | `results/phase7_persistent_qwen35_4b_gsm8k_full.jsonl` | 同名 `_summary.json` | 1319 | 真实 vLLM，Phase 7 全量 |
| Qwen3.5-9B | `results/phase7_persistent_qwen35_9b_gsm8k_full.jsonl` | 同名 `_summary.json` | 1319 | 真实 vLLM，Phase 7 全量 |

另有 Phase 3 至 Phase 6 的 64 条结果和比较文件，例如 `phase4_*`、`phase5_*`、`phase6_*`；它们是较早阶段结果，不替代 Phase 7 全量结果。

### 真实模型并发性能

四个模型各有一个 Phase 9 汇总文件和 640 条 JSONL 记录，覆盖并发度 1、2、4、8、16，每档128条请求：

| 模型 | 汇总 | 逐请求结果 | 状态 |
|---|---|---|---|
| 0.8B | `results/phase9_qwen35_0_8b_concurrency_summary.json` | `results/phase9_qwen35_0_8b_concurrency.jsonl` | completed |
| 2B | `results/phase9_qwen35_2b_concurrency_summary.json` | `results/phase9_qwen35_2b_concurrency.jsonl` | completed |
| 4B | `results/phase9_qwen35_4b_concurrency_summary.json` | `results/phase9_qwen35_4b_concurrency.jsonl` | completed |
| 9B | `results/phase9_qwen35_9b_concurrency_summary.json` | `results/phase9_qwen35_9b_concurrency.jsonl` | completed |

每个并发汇总还包含预检、服务日志路径、显存概况和请求级 TTFT/TPOT。单模型并发 JSONL 的记录数为 640，而汇总文件中每档的统计结果是可直接使用的聚合指标。

### 路由和级联

| 类型 | 文件 | 真实性和口径 |
|---|---|---|
| 文本路由 | `results/phase10_text_routing.json` | 读取真实模型历史输出的 CPU 离线分析，不重新推理；评估集为 Phase 8 已使用的探索性评估集 |
| 三模型级联 | `results/phase11_three_model_cascade_final_v2.json` | 读取真实模型历史输出的 confidence 离线重放；最终版本，使用 2B→4B→9B |
| 三模型级联旧文件 | `phase11_three_model_cascade.json`、`phase11_three_model_cascade_final.json` | 早期重复结果，分析时应优先使用 `final_v2` |
| 四模型 confidence 级联 | `results/phase8_cascade_evaluation.json` | 读取历史真实结果的离线级联评估 |

`results/mock_results.jsonl` 是 MockBackend 测试输出，不属于真实模型实验，不能用于论文模型结果分析。

## 全量 GSM8K 单模型指标

以下均来自 Phase 7 全量1319条结果；时延不包含模型加载和预热。解析失败单独计数，准确率按正确样本数除以1319计算。

| 模型 | 准确率 | 正确/解析失败/推理失败 | 解析成功率 | 平均时延 ms | P50 ms | P95 ms | tokens/s 聚合 | 峰值显存 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8B | 52.24% | 689 / 90 / 0 | 93.18% | 3962.2 | 3376.2 | 8968.6 | 41.86 | 6882.1 |
| 2B | 64.59% | 852 / 78 / 0 | 94.09% | 2954.6 | 2611.5 | 6165.6 | 40.98 | 6992.1 |
| 4B | 89.61% | 1182 / 17 / 0 | 98.71% | 3956.9 | 3363.1 | 8033.2 | 30.91 | 11662.1 |
| 9B | 92.95% | 1226 / 33 / 0 | 97.50% | 4698.1 | 4150.9 | 9595.0 | 31.58 | 22678.1 |

全量单模型汇总文件没有 TTFT、TPOT 或请求吞吐字段；这些指标不能从汇总 JSON 中补写。全量基线有输出 token 数和 tokens/s，但没有独立的实时 TTFT/TPOT 测量。

## 并发性能指标

Phase 9 使用同一组128条请求、vLLM 0.26.0、BF16、非思考模式、`temperature=0`、`max_tokens=384`、`gpu_memory_utilization=0.90`。真实服务请求均成功，以下为并发1和并发16的代表性结果；完整五档数据在对应汇总 JSON 中。

| 模型 | 并发 | 成功/总数 | requests/s | tokens/s | 平均总时延 ms | P50/P95 ms | 平均 TTFT ms | 平均 TPOT ms | 峰值显存 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8B | 1 | 128/128 | 0.259 | 41.44 | 3857.8 | 3359.8 / 8569.2 | 67.7 | 23.87 | 22807.4 |
| 0.8B | 16 | 128/128 | 3.063 | 489.88 | 4402.6 | 3876.4 / 8343.6 | 102.5 | 27.08 | 22807.4 |
| 2B | 1 | 128/128 | 0.364 | 40.96 | 2750.1 | 2455.9 / 5054.1 | 67.5 | 24.02 | 22819.4 |
| 2B | 16 | 128/128 | 4.502 | 521.37 | 3278.9 | 2929.9 / 5964.1 | 109.7 | 27.60 | 22819.4 |
| 4B | 1 | 128/128 | 0.251 | 30.72 | 3979.4 | 3441.0 / 8618.5 | 88.7 | 32.07 | 22883.4 |
| 4B | 16 | 128/128 | 3.037 | 372.48 | 4534.0 | 3920.9 / 10058.2 | 154.6 | 35.97 | 22883.4 |
| 9B | 1 | 128/128 | 0.215 | 31.09 | 4661.1 | 3914.5 / 9472.2 | 89.0 | 31.76 | 22747.4 |
| 9B | 16 | 128/128 | 2.635 | 377.65 | 5172.9 | 4353.9 / 9964.5 | 178.4 | 35.12 | 22747.4 |

并发汇总中的 requests/s 和 tokens/s 是服务端实际闭环统计；TTFT 是客户端收到第一个非空文本 token 的时间，TPOT 是客户端观测值，包含 HTTP 流处理开销。显存峰值为服务运行时 NVML 总占用，不能直接等同于模型权重大小。并发1与并发16的准确率分别保存在汇总文件的 `accuracy` 字段中；本盘点表不将其与1319条质量基线混为一谈。

## 配置一致性检查

- Phase 7 四个全量单模型结果均为 GSM8K test，同一数据集 commit 和 SHA256，同一1319条请求、同一 Prompt、`temperature=0`、`top_p=1`、`max_tokens=384`、BF16、非思考模式、vLLM 0.26.0。
- 四个 Phase 7 结果的模型 revision、模型 ID 和 `gpu_memory_utilization` 分别记录在各自汇总 JSON 中；显存比例为0.25、0.25、0.45、0.90，因模型规模不同而有意不同，不能视为完全相同的显存配置。
- Phase 9 四个模型使用同一128条请求和同一采样参数，统一 `gpu_memory_utilization=0.90`，但每个模型有独立的实验 fingerprint；其测量配置与 Phase 7 的单请求基线不完全相同，因此并发指标不能直接替代 Phase 7 单模型时延。
- Phase 3 至 Phase 6 的64条结果使用固定64条样本，但属于早期阶段；其配置中的模型显存比例按模型大小不同设置。

## 是否可直接用于论文分析

可以直接用于当前论文的探索性和方法验证分析：

- 四模型全量 GSM8K 的准确率、解析失败率、单请求时延、tokens/s 和峰值显存；
- 四模型在同一128条请求上的真实并发吞吐、总时延、TTFT、TPOT 和显存；
- 真实单模型结果上的多模型 Oracle、confidence 离线级联和文本路由分析。

使用时必须保留实验口径：级联和文本路由是离线重放，Phase 8/10/11 的659条评估集已被探索性使用；累计级联时延不包含模型加载、切换、排队、并发和网络开销。

## 缺失或不可直接比较的数据

- Phase 7 全量单模型结果没有 TTFT、TPOT 和真实并发 requests/s；只有总时延和聚合 tokens/s。
- Phase 9 汇总文件没有一个适合所有四模型的单一准确率基线表，但逐档 `outcome_counts`、accuracy 和一致性字段存在；需要按并发档位读取。
- Phase 9 的显存统一使用 `gpu_memory_utilization=0.90`，Phase 7 则按模型使用不同显存比例，两者显存数值不能直接做同配置比较。
- Phase 9 结果只覆盖128条固定请求，每模型五个并发档位，不是完整 GSM8K 质量基线。
- 路由和级联没有真实在线路由器、模型切换、队列或并发资源开销；它们不能解释为在线服务性能。
- 云端 DeepSeek 结果、9B 以上模型、数据集级别的独立泛化测试和资源感知调度结果均不存在。
- `phase9_*_idle_check_failure.json` 和若干启动日志属于过程记录，不是有效实验汇总。

## 下一步最小补测建议

优先补齐与论文主表直接相关的缺口：对已有四模型单模型基线保留同一请求协议，补充与 Phase 9 相同定义的 TTFT、TPOT 和吞吐字段；同时单独整理 Phase 9 五档完整表格。不要将 Mock 输出或离线重放指标混入真实在线结果。
