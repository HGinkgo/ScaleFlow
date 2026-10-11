# ScaleFlow 已有 SLM 协同设计总结

## 1. 已有请求处理流程

ScaleFlow 的在线单请求入口是 `src/scaleflow/scheduler/runner.py` 中的 `run_request()`。它按照策略给出的 `model_order` 调用对应 Backend；每次调用后把结果交给策略判断，累计已调用模型的推理时延，并在策略返回 `return` 时结束请求。

当前存在两种在线策略：

- `AlwaysModelPolicy`：初始模型由配置或调用方直接指定，只调用一个模型并返回结果。它不根据请求文本、置信度或资源状态重新选择模型。
- `ConfidenceCascadePolicy`：调用方预先给出有序模型链和一个 confidence 阈值。当前模型成功且 confidence 达到阈值时立即返回；低于阈值时升级到下一个模型；模型失败时升级；最后一个模型无论是否达到阈值都终止并返回。

真实模型由 `VLLMBackend` 执行，Mock 场景由 `MockBackend` 执行。模型链可以配置为 `0.8B → 2B → 4B → 9B`，也可以使用较短链；Phase 11 的离线分析专门使用 `2B → 4B → 9B`。当前在线策略没有根据请求难度、质量等级或实时性要求进行初始模型预测。

在线返回条件是：固定策略完成唯一模型调用，或级联策略在某一级 confidence 达标、到达最大模型、或最大模型失败后终止。解析失败主要在 GSM8K 评分和离线重放中单独记录；在线 `ConfidenceCascadePolicy` 直接使用 `ModelResponse.success` 和 confidence，不能等同于 GSM8K 解析器级别的完整错误检测。

## 2. 已有协同机制

| 机制 | 代码或配置位置 | 决策依据 | 是否真实在线执行 | 已考虑的系统因素 |
|---|---|---|---|---|
| 固定模型策略 | `src/scaleflow/scheduler/policies.py` 的 `AlwaysModelPolicy`；`configs/mock_qwen35.yaml` 等 | 外部指定一个 `model_id` | 是；Mock 已运行，VLLM 可通过 `run-vllm` 或 `run-gsm8k` 使用 | 只累计当前模型的响应时延；不考虑队列、负载、通信或多模型资源竞争 |
| 在线 confidence 级联基础策略 | `src/scaleflow/scheduler/policies.py` 的 `ConfidenceCascadePolicy`；`configs/mock_qwen35.yaml` | success、confidence 是否达到固定阈值、是否到达最后一级 | Mock 已真实执行过；真实 Qwen 主要通过单模型基线执行，未作为在线多模型服务级联运行 | 只考虑响应 success、confidence 和顺序调用；不考虑 GPU 负载、队列、切换开销、网络时延或并发调度 |
| 四模型 confidence 门槛分析 | `src/scaleflow/offline.py`；`configs/qwen35_gsm8k_offline.yaml`；CLI `analyze-gsm8k-confidence`、`search-gsm8k-cascade`、`evaluate-gsm8k-cascade` | 只用开发集 confidence 判断中间模型是否具备区分能力，再搜索阈值 | 离线读取 Phase 7 真实 JSONL 并重放，没有重新调用模型 | 固定 SHA256 划分、失败强制升级、9B 终止模型；累计时延是历史时延求和，不含切换、排队、并发和网络开销 |
| 三模型 confidence 级联 | `src/scaleflow/cli.py` 的 `cascade_gsm8k_three_offline()`；`configs/qwen35_gsm8k_three_cascade.yaml`；结果 `results/phase11_three_model_cascade_final_v2.json` | 在660条开发集上从 confidence 分位点选择 `2B→4B` 或 `2B→4B→9B` 阈值，质量下限为开发集9B准确率减1个百分点；解析失败、推理失败、confidence 无效强制升级 | 未在线运行；只对 Phase 7 真实结果做离线重放 | 记录调用率、最终来源模型、升级次数和累计历史时延；未建模在线队列、切换、并发和资源状态 |
| 推理前文本路由 | `src/scaleflow/routing.py`；`configs/qwen35_gsm8k_routing.yaml`；CLI `analyze-gsm8k-routing`；结果 `results/phase10_text_routing.json` | 只看原始问题及字符数、词数、数字数、运算符和关键词等特征；包括人工规则和 TF-IDF+逻辑回归 | 未在线执行；在开发集拟合后对659条已使用过的评估集做一次离线分析 | 分开记录历史模型时延和路由器开销；不考虑 GPU 负载、队列、模型切换、网络或并发资源 |
| 多模型 Oracle 与比较 | `src/scaleflow/baseline.py` 的 `compare_model_records()`；CLI `compare-gsm8k-multi` | 事后读取各模型的 correctness，统计全部正确性组合、模型对挽救和逐步 Oracle | 不是决策机制；完全离线分析 | 只提供理论上界和互补性分析，不包含任何在线系统成本 |
| 并发服务测试 | `src/scaleflow/performance.py`；`configs/qwen35_gsm8k_concurrency.yaml`；CLI `run-gsm8k-concurrency` | 不做模型选择，固定一个模型并改变真实服务并发度 | 是真实 vLLM 服务压测，但不是协同路由 | 测量真实批处理、请求吞吐、TTFT、客户端 TPOT、排队影响和显存；未实现多模型共享资源调度 |

## 3. 已有实验能够支持的结论

真实 vLLM 单模型结果已经支持 Qwen3.5 模型规模与 GSM8K 能力之间的明显差异：Phase 7 全量1319条结果中，0.8B、2B、4B、9B 准确率分别为52.24%、64.59%、89.61%和92.95%。四模型结果还显示模型之间存在互补，事后 Oracle 达到96.74%；这证明“不同模型可能为彼此挽救错误”这一现象真实存在。

Phase 9 提供了真实 vLLM 并发服务数据。四个模型均在同一组128条请求和并发度1、2、4、8、16下运行成功，记录了 requests/s、tokens/s、总时延、TTFT、TPOT和显存。结果可以支持不同模型在真实连续批处理下的吞吐和尾时延比较，但不能直接推出一个在线协同路由器的收益。

固定模型策略和基础 confidence 级联已经在 MockBackend 上真实运行，说明请求执行、升级、决策轨迹、累计时延和 JSONL 记录链路可用。真实 Qwen 结果主要用于单模型基线和离线分析，不应表述为已经完成真实多模型在线级联服务。

Phase 8 的 confidence 分析、Phase 10 的文本路由、Phase 11 的三模型级联都使用了真实模型历史输出，因此可以支持探索性离线判断：

- confidence 在部分模型上具有一定错误区分能力，但不足以稳定实现低成本高质量选择；
- 文本规则和 TF-IDF 逻辑回归没有达到接近9B质量的预设目标；
- `2B→4B→9B` 离线级联虽然把评估集9B调用率降到7.13%，但准确率为91.50%，低于 always-9B 的93.93%，累计离线时延却高于 always-9B；
- `2B→4B` 级联没有满足开发集质量约束，且累计离线时延高于直接4B。

这些结果只说明已有离线策略的表现，不能证明在线系统中的模型切换、排队、并发批处理或资源感知调度收益。Phase 8、10、11 的659条评估集已被探索性使用，也不能当作独立泛化测试。

## 4. 当前设计的局限

- 初始模型选择目前由外部固定配置决定。系统没有基于请求难度、质量目标、时延预算或历史负载的在线选择器。
- 在线 confidence 级联的决策输入较少，基础策略只看 success 和单个 confidence 阈值；GSM8K 中的答案解析、失败类型和校准置信度没有完整接入在线策略。
- 离线级联把多个历史单模型时延相加，忽略模型切换、显存重新分配、排队、批处理变化和通信开销，因此不是在线端到端时延。
- 文本路由使用简单特征或轻量 TF-IDF，且评估集被 Phase 8 复用；结果只能作为探索性观察，不能作为独立测试结论。
- Oracle 使用真实正确性，属于事后理想上界，在线决策不可获得。
- Phase 7 单请求基线没有 TTFT、TPOT和真实并发吞吐；Phase 9 的并发测试虽然真实，但请求集合和显存配置与全量质量基线不同。
- 当前没有把 GPU 显存、KV Cache、队列长度、请求优先级、并发度或节点健康状态纳入模型选择决策。

## 5. 尚未涉及的内容

- 多模型同时驻留或多节点部署下的在线资源感知路由；
- 真实在线 confidence 级联服务，包括模型切换成本、队列和并发批处理；
- 动态请求优先级、时延约束、成本约束和 SLA 驱动的策略；
- 通过云端 DeepSeek-V4-Flash 进行兜底或质量上界验证；
- 训练模型参数、微调 Qwen、强化学习或学习型在线调度；
- 无线物理层、网络拓扑和真实边缘到云端通信优化；
- 其他独立困难数据集上的泛化验证。

对后续论文规划的直接启示：现有结果已经建立了“模型能力和时延存在差异、模型结果存在互补，但可观测的 confidence 和请求文本信号不足以直接获得有效协同收益”这一事实基础。后续论文需要把研究重点放在可验证的在线选择信号及其系统成本上，同时严格区分真实服务测量、离线重放和事后 Oracle。
