# Phase 9 并发服务基准设计

## 目标

在同一张 RTX 3090 和统一 vLLM 0.26.0 配置下，测量 Qwen3.5-0.8B、2B、4B、9B 在真实闭环并发负载中的服务性能。实验仅改变模型和并发度，不改变 GSM8K 数据、Prompt、模型 revision、BF16、非思考模式、贪心生成或评分规则。

## 实验契约

- 从固定 GSM8K test commit 的 1,319 条记录中使用 seed 42 确定性选择 128 条，并在 YAML 中固化索引；所有模型、并发等级保持相同请求及顺序。
- OpenAI Chat Completions 请求使用单条 user message，逐请求设置 `chat_template_kwargs.enable_thinking=false`、`temperature=0`、`top_p=1`、`top_k=0`、`min_p=0`、`presence_penalty=0`、`max_tokens=384` 和固定 seed。
- 允许自然 EOS，不设置 `ignore_eos`。四个模型固定 BF16、各自 revision、`max_model_len=2048`、`gpu_memory_utilization=0.90`、eager execution，并关闭 prefix cache。
- 模型分别启动和卸载，不并行驻留。每个模型依次运行并发度 1、2、4、8、16，每级处理完整 128 条请求。

## 执行路径

ScaleFlow 启动独立 `vllm serve` 子进程，通过本机 OpenAI 兼容接口执行异步流式请求。客户端为每个并发等级创建固定数量 worker；worker 完成一条请求后再从共享 FIFO 队列领取下一条，因此最多保持指定数量的在途请求，并让 vLLM 实际执行调度与连续批处理。

模型服务健康后先执行固定 8 条启动预热。每个并发等级开始前，再以相同并发度执行一轮不计时预热；完成后重置该级计时和 NVML 峰值统计。正式长测前用并发 1 的少量样本核对请求参数、解析答案和 Phase 7 贪心输出；明显不一致时停止。

## 计时与统计

- 请求开始：客户端发起 HTTP 请求前的单调时钟。
- TTFT：客户端收到首个非空文本 token 的时间减请求开始时间。
- 完整时延：收到流终止事件的时间减请求开始时间。
- 客户端观测平均 TPOT：`(完整时延 - TTFT) / (输出 token 数 - 1)`；不足两个输出 token 时不计算。
- token 数采用最终 usage 信息，不由文本估算。
- 输出 requests/s、tokens/s、完整时延/TTFT/TPOT 的 mean、P50、P95、平均输入/输出 token 数、成功/失败数、评分结果及显存读数。

一致性以解析后的最终答案是否等于对应 Phase 7 贪心结果为主要指标，同时记录完整生成文本严格一致率。

## 显存与失败处理

NVML 在模型启动前记录空闲基线，在启动和每个正式并发等级期间轮询峰值。模型权重和 KV Cache 预算从 vLLM 启动日志提取并连同原始日志保存。NVML 总占用包含权重、KV Cache 预留、运行时工作区和框架开销，不能解释为模型权重。

若出现 OOM、KV Cache 不足、服务进程退出或健康检查失败，保留已完成结果和完整日志，停止该模型更高并发等级，不改变精度、量化、显存比例或输出上限。

## 产物边界

代码只增加轻量性能模块、CLI、独立 YAML、测试和 README 使用说明。逐请求 JSONL、汇总 JSON、服务日志、数据与模型缓存继续位于 Git 忽略目录。Phase 9 不实现路由、级联、云端 API、新数据集或资源分配算法。
